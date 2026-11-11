#!/usr/bin/env bash
# Static shader and pipeline precompile on Linux (tools/local/static-precompile), precompile-windows.ps1's run:
#   ./precompile-linux.sh                  every shader and pipeline of the game into the static pipeline cache
#                                          _PipelineCache/static/<title>_<version>.bin, which the emulator looks up
#                                          before compiling (hours the first time: run it overnight); with a driver that
#                                          has VK_KHR_pipeline_binary (NVIDIA 5xx) the pipelines' binaries instead,
#                                          <title>_<version>.binaries; without them, each pipeline's own pipeline cache
#                                          data, <title>_<version>.blobs (as precompile-windows.ps1 explains)
#   ./precompile-linux.sh --jobs 8         8 processes at a time (default: 3 threads each on the allowed CPUs)
#   ./precompile-linux.sh --quick          only the shaders as recorded play used them (the release's hints and this
#                                          PC's recordings): about half of a full run. A full run compiles those first
#                                          too, and the game can start once "ready to play" shows
#   ./precompile-linux.sh --full           the seeds' guessed specializations of recorded shaders too
#   ./precompile-linux.sh --coverage       no pipelines, only what the seeds compile to (precompile.py coverage)
#   ./precompile-linux.sh --inputs-only    only the compiled inputs of the emulator's shader prefetch
#                                          (_PipelineCache/static/<title>_<version>.shaders; every full run writes them)
#   ./precompile-linux.sh --export-hints   nothing compiled: the specialization hints of this PC's recorded play
#                                          (my-hints-<title>_<version>.hints, without the game's code), for an issue
# The game: --game <folder or .zar> (the folder with eboot.bin and sce_sys, or that folder packed into a ZArchive), else
# KYTY_GAME (as run_ds.sh takes it). Other options: --threads N, --shards N, --seeds <file>, --recorded <file|none>,
# --cpus <list> (taskset's, e.g. 1-7: only those CPUs; default: the ones this shell may use), --exe <program>.
#
# The NVIDIA driver compiles big compute shaders nearly one at a time per process, so the work is split into shards, a
# niced process each (kyty_shader_precompile --shard i/n) with an NVIDIA disk cache of its own, merged into the static
# cache at the end, about 100 pipelines each (as the plan counts them). A shard stops at 160 (KYTY_PRECOMPILE_SHARD_CAP)
# or where NVIDIA's own dictionary compression came anyway, leaves the rest (exit code 3) and runs again as two; a
# dictionary from a shard's first pipeline on was the driver's before, which no smaller shard escapes: the run ends
# (issue #107). An interrupted run resumes: finished shards are merged first, and what the static cache holds is not
# compiled again.
# Build the program with ./build-linux.sh kyty_shader_precompile; a release package has it next to this script. The
# seed file (seeds-<title>_<version>.seeds) is made when there is none, and made again when the program changed.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
ROOT="$(pwd -P)"

GAME="${KYTY_GAME:-}"
SEEDS=''
RECORDED='*'
JOBS=0
THREADS=3
SHARDS=512
CPU_LIST=''
MODE=full-default # full-default | full | quick | coverage | inputs | hints
EXE=''
usage() { sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
	case "$1" in
		--game) GAME="$2"; shift 2 ;;
		--seeds) SEEDS="$2"; shift 2 ;;
		--recorded) RECORDED="$2"; [ "$RECORDED" = none ] && RECORDED=''; shift 2 ;;
		--jobs) JOBS="$2"; shift 2 ;;
		--threads) THREADS="$2"; shift 2 ;;
		--shards) SHARDS="$2"; shift 2 ;;
		--cpus) CPU_LIST="$2"; shift 2 ;;
		--exe) EXE="$2"; shift 2 ;;
		--quick) MODE=quick; shift ;;
		--full) MODE=full; shift ;;
		--coverage) MODE=coverage; shift ;;
		--inputs-only) MODE=inputs; shift ;;
		--export-hints) MODE=hints; shift ;;
		-h | --help) usage 0 ;;
		-*) echo "unknown option $1 (--help lists them)" >&2; exit 1 ;;
		*) GAME="$1"; shift ;; # (the game alone, as a first argument)
	esac
done
die() { echo "$*" >&2; exit 1; }

if [ -z "$EXE" ]; then
	EXE="$ROOT/kyty_shader_precompile"
	[ -x "$EXE" ] || EXE="$ROOT/_Build/linux/kyty_shader_precompile"
fi
[ -x "$EXE" ] || die "missing $EXE: build it with ./build-linux.sh kyty_shader_precompile (or chmod +x it)"
# Named like the emulator: drivers key per-application caches and profiles by the executable's name, and what
# kyty_shader_precompile compiled the emulator then did not find (precompile-windows.ps1: AMD on Windows). A hard link in
# _PipelineCache (a copy where the file system has none).
RUN_AS="$ROOT/_PipelineCache/precompile-as-emulator"
mkdir -p "$RUN_AS"
rm -f "$RUN_AS/kyty_emulator"
ln "$EXE" "$RUN_AS/kyty_emulator" 2>/dev/null || cp "$EXE" "$RUN_AS/kyty_emulator" || die "cannot place $RUN_AS/kyty_emulator"
EXE="$RUN_AS/kyty_emulator"

# The game's sce_sys/param.json (title and version): in its folder, or copied out of a .zar by the program.
json_field() { tr -d '\r\n' < "$1" | sed -n "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p"; }
[ -n "$GAME" ] || die "which game? --game <folder or .zar> (or KYTY_GAME)"
GAME="${GAME%/}"
PARAM="$GAME/sce_sys/param.json"
PARAM_COPY=0
if [ ! -f "$PARAM" ] && [[ "$GAME" == *.zar ]] && [ -f "$GAME" ]; then
	PARAM="$(mktemp)"
	PARAM_COPY=1
	"$EXE" --game "$GAME" --param "$PARAM" >/dev/null 2>&1 || { rm -f "$PARAM"; PARAM=''; }
fi
[ -n "$PARAM" ] && [ -f "$PARAM" ] || die "no sce_sys/param.json in $GAME (--game <folder or .zar>)"
TITLE="$(json_field "$PARAM" titleId)"
VERSION="$(json_field "$PARAM" contentVersion)"
[ "$PARAM_COPY" = 1 ] && rm -f "$PARAM"
[ -n "$TITLE" ] && [ -n "$VERSION" ] || die "no titleId or contentVersion in $GAME's param.json"
ID="${TITLE}_${VERSION}"

DEFAULT_SEEDS=0
if [ -z "$SEEDS" ]; then
	DEFAULT_SEEDS=1
	if [ -f "$ROOT/seeds-$ID.seeds" ] || [ ! -d "$ROOT/_Build" ]; then
		SEEDS="$ROOT/seeds-$ID.seeds"
	else
		SEEDS="$ROOT/_Build/static-precompile/seeds-$ID.seeds"
	fi
fi
SEED_DIR="$(dirname "$SEEDS")"
STATES="$ROOT/tools/local/static-precompile/pass-states.json"
# The seed file another program made (<file>.program: its size and time) is made again (a release unpacked over the
# last), and replaced where it lists something else: the static cache is then out of date, and the run compiles what
# is new.
PROGRAM_STAMP="$(stat -L -c '%s %Y' "$EXE")"
MADE_BY="$SEEDS.program"
REMAKE=0
if [ "$DEFAULT_SEEDS" = 1 ] && [ -f "$SEEDS" ] && [ "$(cat "$MADE_BY" 2>/dev/null)" != "$PROGRAM_STAMP" ]; then REMAKE=1; fi
if [ ! -f "$SEEDS" ] || [ "$REMAKE" = 1 ]; then
	# Every shader the game ships, with the pipelines it draws them with (from the game files).
	mkdir -p "$SEED_DIR"
	made="$SEEDS"
	[ "$REMAKE" = 1 ] && made="$SEEDS.new"
	"$EXE" --game "$GAME" --make-seeds "$made" --states "$STATES" ||
		die "making the seed file failed (kyty_shader_precompile --make-seeds)"
	if [ "$REMAKE" = 1 ]; then
		if cmp -s "$made" "$SEEDS"; then rm -f "$made"; else mv -f "$made" "$SEEDS"; fi
	fi
	echo "$PROGRAM_STAMP" > "$MADE_BY"
fi

shopt -s nullglob
RECORDINGS=("$ROOT"/_PipelineCache/warmup-v2/*/"$ID".shaders)
if [ "$MODE" = hints ]; then
	[ ${#RECORDINGS[@]} -gt 0 ] || die "no recorded play of $ID in _PipelineCache/warmup-v2 (play first, with the shader warmup on)"
	out="$ROOT/my-hints-$ID.hints"
	arguments=(--game "$GAME" --seeds "$SEEDS" --make-hints "$out")
	for file in "${RECORDINGS[@]}"; do arguments+=(--recorded "$file"); done
	"$EXE" "${arguments[@]}" || die "making the hints failed (kyty_shader_precompile --make-hints)"
	# Zipped as well when zip or python3 is there: GitHub takes a .zip as an attachment, not a .hints file.
	rm -f "$out.zip"
	if command -v zip > /dev/null; then
		(cd "$(dirname "$out")" && zip -q "$(basename "$out").zip" "$(basename "$out")")
	elif command -v python3 > /dev/null; then
		python3 -c 'import sys, zipfile, os; zipfile.ZipFile(sys.argv[1] + ".zip", "w", zipfile.ZIP_DEFLATED).write(sys.argv[1], os.path.basename(sys.argv[1]))' "$out"
	fi
	[ -f "$out.zip" ] && out="$out.zip"
	echo "hints: $out ($(( ($(stat -c %s "$out") + 1023) / 1024 )) KiB, no game code in it): attach it to a GitHub issue"
	exit 0
fi

RECORDED_LOST=0
if [ "$RECORDED" = '*' ]; then
	RECORDED=''
	hints="$ROOT/tools/local/static-precompile/hints-$ID.hints"
	# Another region's Demon's Souls of the same version shares its shaders.
	if [ ! -f "$hints" ] && [[ "$TITLE" =~ ^PPSA0134[0-2]$ ]]; then
		region=("$ROOT"/tools/local/static-precompile/hints-PPSA0134?_"$VERSION".hints)
		for file in "${region[@]}"; do [[ "$(basename "$file")" == hints-PPSA01341_* ]] && region=("$file") && break; done
		[ ${#region[@]} -gt 0 ] && hints="${region[0]}"
	fi
	# Recorded play, all of it (made again each run, with the code of this seed file): the release's hints, this PC's
	# warmup recordings and a recorded seed file next to the seed file.
	sources=()
	released="$SEED_DIR/released-$ID.seeds"
	if [ -f "$hints" ]; then
		if "$EXE" --game "$GAME" --seeds "$SEEDS" --apply-hints "$hints" --out "$released"; then
			sources+=("$released")
		else
			echo "the release's specialization hints could not be applied (kyty_shader_precompile --apply-hints)"
		fi
	fi
	sources+=("${RECORDINGS[@]}")
	[ -f "$SEED_DIR/recorded-$ID.seeds" ] && sources+=("$SEED_DIR/recorded-$ID.seeds")
	if [ ${#sources[@]} -eq 1 ] && [ "${sources[0]}" = "$released" ]; then
		RECORDED="$released"
	elif [ ${#sources[@]} -gt 0 ]; then
		# Their hints together (each specialization once), completed with the code of the seed file.
		together="$SEED_DIR/recorded-play-$ID.hints"
		arguments=(--game "$GAME" --seeds "$SEEDS" --make-hints "$together")
		for file in "${sources[@]}"; do arguments+=(--recorded "$file"); done
		RECORDED="$SEED_DIR/hinted-$ID.seeds"
		if ! "$EXE" "${arguments[@]}" || ! "$EXE" --game "$GAME" --seeds "$SEEDS" --apply-hints "$together" --out "$RECORDED"; then
			echo "the specializations of recorded play could not be gathered (kyty_shader_precompile --make-hints/--apply-hints): the seeds alone"
			RECORDED=''
			RECORDED_LOST=1
		fi
	fi
fi

# The pipelines a seed file's dry run planned as its own (its plan line), 0 without one.
planned_pipelines() {
	local line
	line=$(grep -F "of them this seed file's" "$ROOT/$LOGS/$1.out.log" 2>/dev/null | tail -1)
	if [[ "$line" =~ \(([0-9]+)\ pipelines,\ ([0-9]+)\ of\ them ]]; then echo "${BASH_REMATCH[2]}"; else echo 0; fi
}
# The pipeline recipes of a seed file (its body after the identity line and checksum: the records, each its word count
# and words, then the recipes' count). Without Python: 16000, a recording of every world.
seed_pipelines() {
	command -v python3 >/dev/null 2>&1 || { echo 16000; return; }
	python3 -I - "$1" <<'EOF' 2>/dev/null || echo 16000
import struct, sys
with open(sys.argv[1], 'rb') as f:
    while f.read(1) not in (b'\n', b''):
        pass
    f.read(8)
    (records,) = struct.unpack('<I', f.read(4))
    for _ in range(records):
        (words,) = struct.unpack('<I', f.read(4))
        f.seek(words * 4, 1)
    print(struct.unpack('<I', f.read(4))[0])
EOF
}

TASKSET=()
if [ -n "$CPU_LIST" ]; then
	command -v taskset >/dev/null 2>&1 || die "--cpus needs taskset (util-linux)"
	TASKSET=(taskset -c "$CPU_LIST")
	CPUS=$("${TASKSET[@]}" nproc)
else
	CPUS=$(nproc)
fi
# A process scales to a few threads (4: 85%), and each translates the programs its share needs.
[ "$JOBS" -gt 0 ] 2>/dev/null || JOBS=$(( (CPUS + THREADS - 1) / THREADS ))
[ "$JOBS" -ge 1 ] || JOBS=1
STAMP="$(date +%Y%m%d-%H%M%S)"
if [ -d "$ROOT/_Build" ]; then LOGS="_Build/run-logs/$STAMP-precompile"; else LOGS="logs/$STAMP-precompile"; fi
mkdir -p "$ROOT/$LOGS"
BEGIN=$(date +%s)
# Each process gets an empty NVIDIA disk cache of its own (nothing trained yet), removed when it is done.
CACHES="$(mktemp -d "${TMPDIR:-/tmp}/kyty-precompile-$STAMP.XXXX")"

declare -A RUN_PID RUN_SEEDS RUN_SHARD
cleanup() {
	# A failed shard ends the run: the others are stopped too.
	local name
	for name in "${!RUN_PID[@]}"; do kill "${RUN_PID[$name]}" 2>/dev/null; done
	wait 2>/dev/null
	rm -rf "$CACHES"
}
trap cleanup EXIT
trap 'echo; echo "interrupted: run it again to resume"; exit 130' INT TERM

# Starts the program (niced, its own NVIDIA cache, its logs) in the background: LAST_PID.
start_precompile() {
	local name="$1"
	shift
	mkdir -p "$CACHES/$name"
	__GL_SHADER_DISK_CACHE_PATH="$CACHES/$name" "${TASKSET[@]}" nice -n 10 "$EXE" --game "$GAME" "$@" \
		> "$ROOT/$LOGS/$name.out.log" 2> "$ROOT/$LOGS/$name.err.log" &
	LAST_PID=$!
}
run_precompile() { # name what arguments...: in the foreground, the run ends when it fails
	local name="$1" what="$2" rc
	shift 2
	start_precompile "$name" "$@"
	wait "$LAST_PID"
	rc=$?
	rm -rf "${CACHES:?}/$name"
	[ "$rc" = 0 ] || die "$what failed; logs: $LOGS"
}
elapsed() { local s=$(( $(date +%s) - BEGIN )); printf '%02d:%02d:%02d' $((s / 3600)) $((s % 3600 / 60)) $((s % 60)); }
# A background process is done once it is gone or a zombie.
finished() { [ ! -d "/proc/$1" ] || [ "$(cut -d' ' -f3 "/proc/$1/stat" 2>/dev/null)" = Z ]; }

echo "precompile: $SEEDS, $SHARDS shards, $JOBS at a time on $CPUS CPUs; logs $LOGS"
if [ -n "$RECORDED" ]; then
	echo "recorded:   $RECORDED (the release's hints and this PC's recorded play)"
else
	echo "recorded:   none: the seeds alone"
fi
if [ "$MODE" = coverage ]; then
	out="${SEEDS%.seeds}.compiled.shaders"
	run_precompile coverage coverage --seeds "$SEEDS" --no-pipelines --threads "$CPUS" --out "$out"
	echo "wrote $out"
	exit 0
fi
# The compiled inputs for the emulator's shader prefetch: the seeds translated as the full run does, without pipelines.
INPUTS=(--seeds "$SEEDS" --no-pipelines --threads "$CPUS" --static-inputs)
if [ "$MODE" = inputs ]; then
	run_precompile inputs inputs "${INPUTS[@]}"
	echo "wrote _PipelineCache/static/$ID.shaders"
	exit 0
fi
# The shards an interrupted run finished first: what they hold is not compiled again.
run_precompile merge-before merge --merge
# The plan: every pipeline of the run with the work item that makes it first (a dry run of each seed file on all CPUs, a
# minute or two). The shards then compile each pipeline once, whichever of them or of the seed files makes it.
PLAN="$ROOT/$LOGS/plan.bin"
HAS_RECORDED=0
[ -n "$RECORDED" ] && [ -f "$RECORDED" ] && HAS_RECORDED=1
[ "$MODE" = quick ] && [ "$HAS_RECORDED" = 0 ] &&
	die "--quick needs the specializations of recorded play (a recording or the release's hints)"
# Two tiers: the specializations of recorded play first, then every shader the game ships (--quick: not those).
if [ "$HAS_RECORDED" = 1 ]; then
	run_precompile plan-recorded plan --seeds "$RECORDED" --dry-run --plan "$PLAN" --threads "$CPUS"
fi
if [ "$MODE" != quick ]; then
	seed_plan=(--seeds "$SEEDS" --dry-run --plan "$PLAN" --threads "$CPUS" --static-inputs)
	[ "$HAS_RECORDED" = 1 ] && [ "$MODE" != full ] && seed_plan+=(--covered-by "$RECORDED")
	run_precompile plan-seeds plan "${seed_plan[@]}"
fi
FIRST=()
SECOND=()
TAB=$'\t'
# About 100 pipelines a shard, as the plan counts them. (The recorded seed file's graphics recipes alone counted half of
# its pipelines: 1.07's hints made shards of about 240, past where NVIDIA's driver compresses with its dictionary.)
if [ "$HAS_RECORDED" = 1 ]; then
	planned=$(planned_pipelines plan-recorded)
	[ "$planned" -gt 0 ] || planned=$(( 2 * $(seed_pipelines "$RECORDED") ))
	count=$(( (planned + 99) / 100 ))
	[ "$count" -ge 1 ] || count=1
	for ((i = 0; i < count; i++)); do FIRST+=("$RECORDED$TAB$i/$count"); done
fi
if [ "$MODE" != quick ]; then
	count=$(( ($(planned_pipelines plan-seeds) + 99) / 100 ))
	[ "$count" -ge "$SHARDS" ] || count=$SHARDS
	for ((i = 0; i < count; i++)); do SECOND+=("$SEEDS$TAB$i/$count"); done
fi
# The shards share the pipelines they hold (PipelineBinaryWriter::Claimed): one that several shards make is compiled once.
export KYTY_PRECOMPILE_CLAIMS="$CACHES/claims"
mkdir -p "$KYTY_PRECOMPILE_CLAIMS"
# Pipelines the graphics driver never finished compiling (a shard's hang guard: exit 4, issue #126), listed here: every
# shard leaves them out.
STUCK_LIST="$ROOT/$LOGS/stuck-pipelines.txt"
export KYTY_PRECOMPILE_STUCK="$STUCK_LIST"
RESTARTED=0
DONE_COUNT=0
SPLIT=0
SHOWN=$(date +%s)
TIER_MERGED=0
[ ${#FIRST[@]} -eq 0 ] && TIER_MERGED=1
while [ ${#FIRST[@]} -gt 0 ] || [ ${#SECOND[@]} -gt 0 ] || [ ${#RUN_PID[@]} -gt 0 ]; do
	# The first tier done (no shard runs then: the merge replaces the file they read): merged into the static cache and
	# kept for the final merge, so the game can use it while the rest compiles.
	if [ "$TIER_MERGED" = 0 ] && [ ${#FIRST[@]} -eq 0 ] && [ ${#RUN_PID[@]} -eq 0 ]; then
		TIER_MERGED=1
		if [ ${#SECOND[@]} -gt 0 ]; then
			# (Without the claims: a merge copying the store would claim every pipeline in it.)
			claims="$KYTY_PRECOMPILE_CLAIMS"
			unset KYTY_PRECOMPILE_CLAIMS
			run_precompile merge-recorded merge --merge --keep-shards
			export KYTY_PRECOMPILE_CLAIMS="$claims"
			echo "  $(elapsed) ready to play: the shaders of recorded play are in the static cache; the game's other shaders follow"
		fi
	fi
	while [ ${#RUN_PID[@]} -lt "$JOBS" ]; do
		if [ ${#FIRST[@]} -gt 0 ]; then
			item="${FIRST[0]}"
			FIRST=("${FIRST[@]:1}")
		elif [ "$TIER_MERGED" = 1 ] && [ ${#SECOND[@]} -gt 0 ]; then
			item="${SECOND[0]}"
			SECOND=("${SECOND[@]:1}")
		else
			break
		fi
		seed_file="${item%%"$TAB"*}"
		shard="${item#*"$TAB"}"
		base="$(basename "$seed_file")"
		name="shard${shard/\//of}-${base%.*}"
		start_precompile "$name" --seeds "$seed_file" --shard "$shard" --threads "$THREADS" --plan "$PLAN"
		RUN_PID[$name]=$LAST_PID
		RUN_SEEDS[$name]="$seed_file"
		RUN_SHARD[$name]="$shard"
	done
	sleep 0.25
	for name in "${!RUN_PID[@]}"; do
		pid="${RUN_PID[$name]}"
		finished "$pid" || continue
		wait "$pid"
		rc=$?
		seed_file="${RUN_SEEDS[$name]}"
		shard="${RUN_SHARD[$name]}"
		unset "RUN_PID[$name]" "RUN_SEEDS[$name]" "RUN_SHARD[$name]"
		rm -rf "${CACHES:?}/$name"
		if [ "$rc" = 3 ]; then
			# NVIDIA's dictionary from the shard's first pipeline on: the driver had it before the shard compiled anything,
			# and every smaller shard would too (issue #107).
			if grep -q -F 'dictionary from here on (after 0 captured' "$ROOT/$LOGS/$name.out.log" 2>/dev/null; then
				die "$name: NVIDIA's driver compressed even its first pipeline with a dictionary it had before, so the precompile cannot keep its pipelines and smaller shards would not help. Please report it in issue #107 (github.com/chenxiao07/KytyPS5) with the logs in $LOGS; what the precompile finished is kept."
			fi
			i="${shard%/*}"
			n="${shard#*/}"
			[ "$n" -lt 65536 ] || die "$name cannot be split further; logs: $LOGS"
			halves=("$seed_file$TAB$i/$((2 * n))" "$seed_file$TAB$((i + n))/$((2 * n))")
			if [ "$seed_file" = "$RECORDED" ]; then FIRST+=("${halves[@]}"); else SECOND+=("${halves[@]}"); fi
			SPLIT=$((SPLIT + 1))
		elif [ "$rc" = 4 ]; then
			# A pipeline the driver never finished compiling (named in the shard's log and in stuck-pipelines.txt): the
			# shard starts again without it, what it made kept (its .part file, merged like the shards').
			RESTARTED=$((RESTARTED + 1))
			[ "$RESTARTED" -le 100 ] || die "more than 100 shards stopped at pipelines the driver never finished; logs: $LOGS"
			for stream in out err; do
				[ -f "$ROOT/$LOGS/$name.$stream.log" ] && mv "$ROOT/$LOGS/$name.$stream.log" "$ROOT/$LOGS/$name.stuck$RESTARTED.$stream.log"
			done
			if [ "$seed_file" = "$RECORDED" ]; then FIRST+=("$seed_file$TAB$shard"); else SECOND+=("$seed_file$TAB$shard"); fi
			echo "  $(elapsed) $name: the graphics driver never finished compiling a pipeline; started again without it"
		elif [ "$rc" != 0 ]; then
			die "$name failed (exit code $rc); logs: $LOGS"
		else
			DONE_COUNT=$((DONE_COUNT + 1))
		fi
	done
	if [ $(( $(date +%s) - SHOWN )) -ge 60 ]; then
		SHOWN=$(date +%s)
		echo "  $(elapsed) shards: $DONE_COUNT done, ${#RUN_PID[@]} running, $(( ${#FIRST[@]} + ${#SECOND[@]} )) waiting ($SPLIT split)"
	fi
done
unset KYTY_PRECOMPILE_CLAIMS KYTY_PRECOMPILE_STUCK
# (--quick: what the store held of the other shaders stays. Recorded play that could not be gathered: the pipelines an
# earlier run made for it stay too.) (--stamp: the store's translation features, those this run used.)
merge_arguments=(--merge --stamp)
[ "$MODE" != quick ] && [ "$RECORDED_LOST" = 0 ] && merge_arguments+=(--prune)
run_precompile merge merge "${merge_arguments[@]}"
[ "$MODE" = quick ] && run_precompile inputs inputs "${INPUTS[@]}"
cache="$(ls -t "$ROOT/_PipelineCache/static/$ID".{bin,binaries,blobs} 2>/dev/null | head -1)"
if [ -n "$cache" ]; then
	echo "done in $(elapsed): $cache ($(( $(stat -c %s "$cache") / 1048576 )) MB; $SPLIT shards split)"
	if [ -f "$STUCK_LIST" ]; then
		echo "$(wc -l < "$STUCK_LIST") pipeline(s) left out: the graphics driver never finished compiling them ($LOGS/stuck-pipelines.txt). The game compiles them when it first needs them and may stop there the same way: please report it on GitHub with that file."
	fi
else
	echo "done in $(elapsed), but there is no _PipelineCache/static/$ID.bin, .binaries or .blobs: see the logs in $LOGS"
	exit 1
fi
