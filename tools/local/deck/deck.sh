#!/usr/bin/env bash
# The Steam Deck loop from the Windows PC (Git Bash): build in WSL, deploy and run on the Deck over SSH.
#   deck.sh sync              this checkout (HEAD, uncommitted changes, untracked files) into WSL's ~/kyty-deck
#   deck.sh build [targets]   sync, then build-deck.sh there (default kyty_emulator)
#   deck.sh deploy            the Deck build, deckbench.py and the bench save into the Deck's ~/KytyPS5
#   deck.sh run [options]     one measured run on the Deck (deckbench.py run --help)
#   deck.sh prof <label>      a run's --prof samples by thread, module and function (deckprof.py)
#   deck.sh marks <label>     a run's --tracem capture: GPU time per shader, barriers by the code that recorded them
#   deck.sh trace <label>     a run's --trace timeline: frames, the render thread's gaps and what ended them
#   deck.sh census <label>    a run's --census capture: render-thread time per kind, barrier callers (--kind 13)
#   deck.sh ab "label|+cvar=v" ...  same-boot steps on a game kept running (run --keep): fps and GPU ms per step
#   deck.sh ssh [command]     a command on the Deck
# KYTY_DECK (default deck@192.168.0.229); the key ~/.ssh/kyty_deck (Windows and WSL), authorized on the Deck once.
# WSL needs the build-linux.yml packages and a clone of this repository at ~/kyty-deck (git clone /mnt/c/...).
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"
DECK="${KYTY_DECK:-deck@192.168.0.229}"
ROOT_WSL="$(wsl.exe -e wslpath -a "$(cygpath -w "$ROOT")" | tr -d '\r')"
SSH=(/c/Windows/System32/OpenSSH/ssh.exe -i "$(cygpath -w ~/.ssh/kyty_deck)" -o BatchMode=yes -o ConnectTimeout=10 "$DECK")
WSL_SSH="ssh -i ~/.ssh/kyty_deck -o BatchMode=yes -o ConnectTimeout=10"

in_wsl() { wsl.exe -e bash -lc "$1"; }

sync() {
	local patch="$ROOT/_Build/deck-sync.patch" list="$ROOT/_Build/deck-sync.untracked"
	mkdir -p "$ROOT/_Build"
	git -C "$ROOT" diff HEAD --binary > "$patch"
	git -C "$ROOT" ls-files --others --exclude-standard -- src tools CMakeLists.txt cmake > "$list"
	# HEAD's commit from the repository itself: a worktree's .git names it by a Windows path WSL's git cannot follow.
	local common head
	common="$(wsl.exe -e wslpath -a "$(cygpath -w "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")" | tr -d '\r')"
	head="$(git -C "$ROOT" rev-parse HEAD)"
	in_wsl "set -e; cd ~/kyty-deck; git fetch -q '$common' $head; git checkout -q -f --detach $head;
		git clean -fdq -e _Build; git submodule update -q --init --recursive;
		[ ! -s '$ROOT_WSL/_Build/deck-sync.patch' ] || git apply --whitespace=nowarn '$ROOT_WSL/_Build/deck-sync.patch';
		[ ! -s '$ROOT_WSL/_Build/deck-sync.untracked' ] || (cd '$ROOT_WSL' && tr -d '\r' < _Build/deck-sync.untracked | tar -cf - -T -) | tar -xf -;
		echo \"synced \$(git rev-parse --short HEAD) + \$(git status --porcelain | wc -l) changed files\""
}

case "${1:-}" in
sync) sync ;;
build)
	shift
	sync
	in_wsl "cd ~/kyty-deck && nice -n 10 tools/local/deck/build-deck.sh $*" ;;
deploy)
	"${SSH[@]}" 'mkdir -p ~/KytyPS5/bench-save ~/KytyPS5/logs ~/KytyPS5/prof'
	in_wsl "set -e; rsync -t --info=stats1 -e '$WSL_SSH' ~/kyty-deck/_Build/deck/kyty_emulator ~/kyty-deck/_Build/deck/translator-id '$ROOT_WSL/tools/local/deck/deckbench.py' '$ROOT_WSL/tools/local/deck/deckab.py' '$ROOT_WSL/tools/local/deck/gpumetrics.py' '$ROOT_WSL/tools/local/deck/decklive.py' '$ROOT_WSL/tools/local/deck/deck-perf.sh' '$ROOT_WSL/tools/local/live-trace.py' $DECK:KytyPS5/;
		mkdir -p ~/kyty-deck/_Build/deck-deployed; cp -p ~/kyty-deck/_Build/deck/kyty_emulator ~/kyty-deck/_Build/deck-deployed/;
		rsync -rt -e '$WSL_SSH' '$ROOT_WSL/_Build/bench-baselines/rec-m02_00_00_00-c0000_0006/' $DECK:KytyPS5/bench-save/" ;;
run)
	shift
	"${SSH[@]}" "python3 ~/KytyPS5/deckbench.py run $(printf '%q ' "$@")" ;;
prof)
	# deck.sh prof <label>: the run's process CPU samples into WSL, by thread, module and function (deckprof.py).
	in_wsl "set -e; mkdir -p ~/kyty-deck/_Build/deck-prof ~/kyty-deck/_Build/deck-libs; rsync -t -e '$WSL_SSH' '$DECK:KytyPS5/prof/$2.prof*' ~/kyty-deck/_Build/deck-prof/;
		rsync -t -e '$WSL_SSH' '$DECK:/usr/lib/libc.so.6' '$DECK:/usr/lib/libvulkan_radeon.so' '$DECK:/usr/lib/libstdc++.so.6' ~/kyty-deck/_Build/deck-libs/ 2>/dev/null || true;
		python3 ~/kyty-deck/tools/local/deck/deckprof.py ~/kyty-deck/_Build/deck-prof/$2.prof ~/kyty-deck/_Build/deck-deployed/kyty_emulator --libs ~/kyty-deck/_Build/deck-libs ${3:-}" ;;
marks)
	# deck.sh marks <label> [live-trace.py marks options]: GPU time per shader of a run's --tracem capture.
	label="$2"
	shift 2
	in_wsl "set -e; mkdir -p ~/kyty-deck/_Build/deck-prof; rsync -t -e '$WSL_SSH' '$DECK:KytyPS5/prof/$label.trace*' ~/kyty-deck/_Build/deck-prof/;
		python3 ~/kyty-deck/tools/local/live-trace.py marks ~/kyty-deck/_Build/deck-prof/$label.trace --exe ~/kyty-deck/_Build/deck-deployed/kyty_emulator $*" ;;
trace)
	# deck.sh trace <label>: a run's --trace timeline: per frame (live-trace.py frames), the render thread's idle gaps
	# and the guest submissions that ended them (live-trace-report.py), with the threads' names.
	label="$2"
	in_wsl "set -e; mkdir -p ~/kyty-deck/_Build/deck-prof; rsync -t -e '$WSL_SSH' '$DECK:KytyPS5/prof/$label.timeline*' ~/kyty-deck/_Build/deck-prof/;
		cd ~/kyty-deck/_Build/deck-prof; python3 ~/kyty-deck/tools/local/live-trace.py frames $label.timeline --detail 4 | head -40;
		python3 ~/kyty-deck/tools/local/live-trace-report.py $label.timeline --hz \$(cat $label.timeline.hz) --frames 4 | head -80;
		echo threads:; cat $label.timeline.threads | sort -n | tr '\n' ' '" ;;
census)
	# deck.sh census <label> [deckcensus.py options, e.g. --kind 13]: a run's --census capture, callers symbolized.
	label="$2"
	shift 2
	in_wsl "set -e; mkdir -p ~/kyty-deck/_Build/deck-prof; rsync -t -e '$WSL_SSH' '$DECK:KytyPS5/prof/$label.census' ~/kyty-deck/_Build/deck-prof/;
		python3 ~/kyty-deck/tools/local/deck/deckcensus.py ~/kyty-deck/_Build/deck-prof/$label.census ~/kyty-deck/_Build/deck-deployed/kyty_emulator $*" ;;
ab)
	# deck.sh ab "label|+name=value;..." ...: same-boot steps on a game left running by `deck.sh run --keep` (deckab.py).
	shift
	"${SSH[@]}" "python3 ~/KytyPS5/deckab.py $(printf '%q ' "$@")" ;;
ssh)
	shift
	"${SSH[@]}" "$@" ;;
*)
	sed -n '2,10p' "$0"
	exit 1 ;;
esac
