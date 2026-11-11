#!/bin/bash
# Runs Demon's Souls with the best binary (the emulator's own switch defaults are the release settings).
# Work in the repository root when this is the checked-out script (tools/local/), else in the folder
# of a packaged copy (the executable next to it).
_kyty_here="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
if [ -d "$_kyty_here/../../_Build" ]; then cd "$_kyty_here/../.." || exit 1
elif [ -d "$_kyty_here/../_Build" ]; then cd "$_kyty_here/.." || exit 1
else cd "$_kyty_here" || exit 1; fi
# The emulator's switches default to the release settings (src/local/performance-switches.h); one set before this
# script reaches it unchanged (KYTY_TABLE_DISPATCH=0 ./run_ds.sh: an A/B or a workaround, as in issue #51).
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