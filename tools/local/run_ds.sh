#!/bin/bash
# Production switch set for Demon's Souls (from run-windows.json) + best binary.
# Work in the repository root when this is the checked-out script (tools/local/), else in the folder
# of a packaged copy (the executable next to it).
_kyty_here="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
if [ -d "$_kyty_here/../../_Build" ]; then cd "$_kyty_here/../.." || exit 1
elif [ -d "$_kyty_here/../_Build" ]; then cd "$_kyty_here/.." || exit 1
else cd "$_kyty_here" || exit 1; fi
# A switch set before this script keeps its value (KYTY_TABLE_DISPATCH=0 ./run_ds.sh: issue #51, where the script
# set it back to 1 and the test changed nothing).
export KYTY_SHADER_WARMUP=${KYTY_SHADER_WARMUP:-1}
export KYTY_GPU_BUFFER_PAGES=${KYTY_GPU_BUFFER_PAGES:-1}
export KYTY_RECORDING_CPUS=${KYTY_RECORDING_CPUS:-auto}
export KYTY_SRT_NATIVE=${KYTY_SRT_NATIVE:-1}
export KYTY_SRT_PREDICATES=${KYTY_SRT_PREDICATES:-1}
export KYTY_PREPARATION_SCRATCH=${KYTY_PREPARATION_SCRATCH:-1}
export KYTY_PREPARATION_LOOKUP=${KYTY_PREPARATION_LOOKUP:-1}
export KYTY_PREPARATION_TRIM=${KYTY_PREPARATION_TRIM:-1}
export KYTY_SPECIALIZATION_GUARD=${KYTY_SPECIALIZATION_GUARD:-1}
export KYTY_PIPELINE_INDEX=${KYTY_PIPELINE_INDEX:-1}
export KYTY_BINDING_SCRATCH=${KYTY_BINDING_SCRATCH:-1}
export KYTY_DRAW_RUN_RANGES=${KYTY_DRAW_RUN_RANGES:-1}
export KYTY_BUFFER_RESIDENCY=${KYTY_BUFFER_RESIDENCY:-1}
export KYTY_COPY_FEEDBACK=${KYTY_COPY_FEEDBACK:-1}
export KYTY_ASYNC_LOD_STATS=${KYTY_ASYNC_LOD_STATS:-1}
export KYTY_BACKING_READ=${KYTY_BACKING_READ:-1}
export KYTY_STREAM_UPLOAD=${KYTY_STREAM_UPLOAD:-1}
export KYTY_FRAME_PIPELINE=${KYTY_FRAME_PIPELINE:-1}
export KYTY_IMAGE_BARRIER_DEDUPE=${KYTY_IMAGE_BARRIER_DEDUPE:-1}
export KYTY_IMAGE_POOL=${KYTY_IMAGE_POOL:-1}
export KYTY_PENDING_DRAIN=${KYTY_PENDING_DRAIN:-1}
export KYTY_DISPATCH_BATCH=${KYTY_DISPATCH_BATCH:-128}
export KYTY_VULKAN_RECORDING=${KYTY_VULKAN_RECORDING:-1}
export KYTY_DEFERRED_SUBMIT=${KYTY_DEFERRED_SUBMIT:-1}
export KYTY_NATIVE_XPR=${KYTY_NATIVE_XPR:-1}
export KYTY_WRITE_WINDOW_HANDOFF=${KYTY_WRITE_WINDOW_HANDOFF:-1}
export KYTY_READBACK_DETACH=${KYTY_READBACK_DETACH:-1}
export KYTY_ASYNC_WRITE_READBACK=${KYTY_ASYNC_WRITE_READBACK:-1}
export KYTY_READBACK_SLOTS=${KYTY_READBACK_SLOTS:-1}
export KYTY_ASYNC_UPLOAD=${KYTY_ASYNC_UPLOAD:-2}
export KYTY_BDA_DIRTY_REGIONS=${KYTY_BDA_DIRTY_REGIONS:-1}
export KYTY_GLOBAL_BARRIER_DEDUPE=${KYTY_GLOBAL_BARRIER_DEDUPE:-1}
export KYTY_RANGE_SET_FAST=${KYTY_RANGE_SET_FAST:-1}
export KYTY_IMAGE_GRANULES=${KYTY_IMAGE_GRANULES:-2}
export KYTY_NATIVE_XPR_PREDICT=${KYTY_NATIVE_XPR_PREDICT:-1}
export KYTY_NATIVE_IMAGE_PROOF=${KYTY_NATIVE_IMAGE_PROOF:-1}
export KYTY_TEXTURE_RESOLVE_PAGES=${KYTY_TEXTURE_RESOLVE_PAGES:-1}
export KYTY_ASYNC_REPROTECT=${KYTY_ASYNC_REPROTECT:-1}
export KYTY_READBACK_NARROW=${KYTY_READBACK_NARROW:-1}
export KYTY_SHADER_WARMUP_THREADS=${KYTY_SHADER_WARMUP_THREADS:-16}
export KYTY_READBACK_QUEUE=${KYTY_READBACK_QUEUE:-1}
export KYTY_TABLE_XPR=${KYTY_TABLE_XPR:-2}
export KYTY_TABLE_DISPATCH=${KYTY_TABLE_DISPATCH:-1}
# The rest of run-windows.ps1's defaults (src/local/performance-switches.h).
export KYTY_BUFFER_RECLAIM=${KYTY_BUFFER_RECLAIM:-1}
export KYTY_DRAW_PACKETS=${KYTY_DRAW_PACKETS:-1}
export KYTY_PARTIAL_IMAGE_DIRTY=${KYTY_PARTIAL_IMAGE_DIRTY:-1}
export KYTY_PARTIAL_ROW_BANDS=${KYTY_PARTIAL_ROW_BANDS:-1}
export KYTY_ASYNC_XPR_PIPELINES=${KYTY_ASYNC_XPR_PIPELINES:-1}
export KYTY_NATIVE_XPR_RELOCATE=${KYTY_NATIVE_XPR_RELOCATE:-1}
export KYTY_NATIVE_XPR_STORE_BUDGET=${KYTY_NATIVE_XPR_STORE_BUDGET:-256}
export KYTY_NATIVE_XPR_KEEP_FRAMES=${KYTY_NATIVE_XPR_KEEP_FRAMES:-600}
export KYTY_NATIVE_XPR_INSTANCES=${KYTY_NATIVE_XPR_INSTANCES:-1}
export KYTY_UNMAP_PROTECT_SKIP=${KYTY_UNMAP_PROTECT_SKIP:-1}
# Demon's Souls' own engine setting (src/loader/gameArgs.h), as run-windows.ps1 gives it: no XPR index-buffer
# culling, ~2800 compute dispatches a frame at Boletaria 1-1 that cull triangles the GPU then draws anyway
# (KYTY_GAME_CONVARS= empty: none).
export KYTY_GAME_CONVARS=${KYTY_GAME_CONVARS-doIndexBufferCulling=false}
# One driver pipeline cache for every build (_PipelineCache/local/<key>), as run-windows.ps1: keyed by the
# executable, every update compiled the warmup pipelines again (minutes), and a modified source tree had none.
export KYTY_DRIVER_CACHE_KEY=${KYTY_DRIVER_CACHE_KEY:-1f3c29b70e536c8f5a5e812e793b33ac2eb6858b4810ed98ca2a51b9fa5eed97}
# GPU render thread pinned off CPU 0 (interrupt-heavy); physical cores 1-7.
export KYTY_RENDER_CPUS=${KYTY_RENDER_CPUS:-"1,2,3,4,5,6,7"}
# KYTY_BIN, else the first there of _Build/linux-nt, _Build/linux (build-linux.sh) and a packaged copy's executable
# next to the script.
BIN="${KYTY_BIN:-}"
for candidate in ./_Build/linux-nt/kyty_emulator ./_Build/linux/kyty_emulator ./kyty_emulator; do
  [ -n "$BIN" ] || [ ! -x "$candidate" ] || BIN="$candidate"
done
[ -n "$BIN" ] || { echo "no kyty_emulator (build it with build-linux.sh, or set KYTY_BIN)" >&2; exit 1; }
# Hands-off stutter capture (KYTY_LIVE_FILE): a fresh frame-interval (flips) file every ~12 s.
# The game writes the last 3000 presented-frame intervals (us) per file, so any hitch is recorded
# without sending any input. Analyse with tools/local/stutter-report.py (or manually).
# KYTY_MONITOR_LOG=<file>: a system monitor for the stutter diagnosis, 1 Hz memory / swap / PSI-memory, while the
# emulator runs.
if [ -n "$KYTY_MONITOR_LOG" ]; then
(
  sleep 5
  while :; do
    pid=$(pgrep -x kyty_emulator | head -1)
    rss=0; [ -n "$pid" ] && rss=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null)
    read -r avail swt swf < <(awk '/MemAvailable/{a=$2}/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print a,t,f}' /proc/meminfo)
    psi=$(awk -F'avg10=' '/^some/{split($2,x," ");print x[1]}' /proc/pressure/memory 2>/dev/null)
    printf '%s rss_mb=%d avail_mb=%d swap_used_mb=%d psi_mem=%s\n' "$(date +%H:%M:%S)" \
      "$((rss/1024))" "$((avail/1024))" "$(( (swt-swf)/1024 ))" "$psi"
    sleep 1
    [ -n "$pid" ] || break
  done
) > "$KYTY_MONITOR_LOG" 2>&1 &
fi
if [ -n "$KYTY_LIVE_FILE" ]; then
  flips_dir="${KYTY_FLIPS_DIR:-logs/flips}"
  mkdir -p "$flips_dir"
  printf 'id 0\n' > "$KYTY_LIVE_FILE"
  (
    sleep 6; n=0
    while :; do
      n=$((n+1))
      printf 'id %d\nflips 3000 %s/%05d_%s.txt\n' "$n" "$flips_dir" "$n" "$(date +%H%M%S)" \
        > "$KYTY_LIVE_FILE.tmp" && mv "$KYTY_LIVE_FILE.tmp" "$KYTY_LIVE_FILE"
      sleep 12
    done
  ) &
fi
# Mailbox = vblank-aligned, no tearing, low latency (smooth). Override with PRESENT_MODE=Fifo|Immediate.
# MANGO=1 (default): MangoHud FPS/CPU/GPU overlay; MANGO=0 disables it.
# The game: KYTY_GAME (its folder, or a .zar archive).
GAME="${KYTY_GAME:-}"
[ -n "$GAME" ] || { echo "set KYTY_GAME to the game (its folder, or a .zar archive)" >&2; exit 1; }
# The vblank: KYTY_VBLANK (default 60), or a --vblank-frequency among the arguments. Above 60 Hz Demon's Souls needs
# gameMinDeltaTime (as run-windows.ps1 gives it, 5550c1bb): it steps its game by at least a frame of its target rate
# each frame, so every frame past 60 fps (30 in the cinematic mode) played the game that much faster.
VBLANK="${KYTY_VBLANK:-60}"
previous=""
for argument in "$@"; do
  [ "$previous" != "--vblank-frequency" ] || VBLANK="$argument"
  previous="$argument"
done
if [ "$VBLANK" -gt 60 ] 2>/dev/null && [[ "$KYTY_GAME_CONVARS" != *gameMinDeltaTime* ]]; then
  export KYTY_GAME_CONVARS="${KYTY_GAME_CONVARS:+$KYTY_GAME_CONVARS;}gameMinDeltaTime=0.004"
fi
RUN=("$BIN")
if [ "${MANGO:-1}" = 1 ] && command -v mangohud >/dev/null 2>&1; then
  RUN=(mangohud "$BIN")
  export MANGOHUD_CONFIG=${MANGOHUD_CONFIG:-fps,frame_timing,vram,ram,gpu_temp,cpu_temp,gpu_power,cpu_power}
fi
exec "${RUN[@]}" --game "$GAME" \
  --screen-width 1920 --screen-height 1080 --console-language 1 \
  --shader-validation false --shader-log-direction Silent --printf-direction Silent \
  --vblank-frequency "$VBLANK" --present-mode "${PRESENT_MODE:-Mailbox}" "$@"