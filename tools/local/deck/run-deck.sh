#!/bin/bash
# Demon's Souls on the Steam Deck with the fastest settings measured (docs/STEAM-DECK.md): the Lowest graphics preset
# (shorter sun shadows, no ambient or torch shadows, simpler distant detail, textures a quarter the size), the 3D at
# 640x360, no projected decals (blood, ground marks), sun shadows only within 20 m, the game's 1080p output shown full
# screen at 1280x800 with 16:9 kept (bars top and bottom), the mouse cursor hidden. Best with the GPU clock at 1300 MHz
# (Steam's Quick Access menu > Performance > Manual GPU Clock 1300; Game Mode keeps it per game).
#
# It runs the kyty_emulator next to it: the Linux package's, or a Steam Deck build (tools/local/deck/build-deck.sh). The
# Steam Deck build's defaults are set here too, so the Linux package runs as it does; one set before the script wins.
#
# The game: KYTY_GAME, else ~/Games/PPSA01341-app0.zar, else the folder ~/Games/PPSA01341-app0 (eboot.bin in it).
# Switches (Steam: the shortcut's Properties > Launch Options, e.g. `KYTY_DECK_SUN_SHADOWS=full %command%`):
#   KYTY_DECK_SUN_SHADOWS=full  sun shadows to 60 m, the Lowest preset's (slower)
#   KYTY_DECK_SUN_SHADOWS=off   no sun shadows (fastest, but the sun lights the character in dark places)
#   KYTY_DECK_DECALS=1          projected decals back on (~7% slower)
#   KYTY_RENDER_HEIGHT=432      a sharper 3D (768x432; ~5% slower than 360)
#   KYTY_FPS_HUD=0              no frame-rate panel at the top right (F8 switches it)
#   KYTY_HIDE_CURSOR=0          the cursor shows when the mouse moves
#   KYTY_CONSOLE_LANGUAGE=<n>   the console's language (1 English, the default; 0 Japanese)
# Extra emulator options go after the script's name; the log is written to logs/ next to it.
here="$(dirname "$(readlink -f "$0")")"
cd "$here" || exit 1
[ -x ./kyty_emulator ] || { echo "no kyty_emulator next to $0 (chmod +x kyty_emulator?)" >&2; exit 1; }
# One copy at a time: two starts at once (a double-click on a desktop icon, which opens on one click) ran two games
# sharing the CPU, the GPU and the save. The lock stays held while the emulator runs (it inherits it).
exec 9>"$here/.run-deck.lock"
if ! flock -n 9; then
	echo "Demon's Souls (KytyPS5) is already running"
	exit 1
fi
if [ -z "${KYTY_GAME:-}" ]; then
	for candidate in "$HOME/Games/PPSA01341-app0.zar" "$HOME/Games/PPSA01341-app0"; do
		[ -e "$candidate" ] && { KYTY_GAME="$candidate"; break; }
	done
fi
[ -n "${KYTY_GAME:-}" ] || { echo "no game: put it in ~/Games/PPSA01341-app0(.zar) or set KYTY_GAME" >&2; exit 1; }
export KYTY_GAME
export KYTY_RENDER_HEIGHT="${KYTY_RENDER_HEIGHT:-360}"
export KYTY_HD_OUTPUT=1
export KYTY_TEXTURE_REDUCE="${KYTY_TEXTURE_REDUCE:-2}"
deck_convars="r_forceViewportHeightForPixelSizeCalc=480;r_enableAmbientShadowing=false;r_enableSpotShadows=false"
[ "${KYTY_DECK_DECALS:-0}" = 1 ] || deck_convars="$deck_convars;r_projectedDecalEnable=false"
# Sun shadows: short (20 m) by default; without them the sun lit the character under roofs and in dark corners (its
# highlights far too bright there). (r_forceShadowLOD=1 showed stray wall pieces in the tutorial: left out.)
case "${KYTY_DECK_SUN_SHADOWS:-short}" in
	0 | off) deck_convars="$deck_convars;r_enableCascadeShadows=false" ;;
	1 | full) deck_convars="$deck_convars;sunShadowMaxDistance=60" ;;
	*) deck_convars="$deck_convars;sunShadowMaxDistance=20" ;;
esac
export KYTY_GAME_CONVARS="$deck_convars${KYTY_GAME_CONVARS:+;$KYTY_GAME_CONVARS}"
export KYTY_PRESENT_ASPECT=fit
export KYTY_HIDE_CURSOR="${KYTY_HIDE_CURSOR:-1}"
# The emulator's own panel at the top right (frame rate and frame times; F8 switches it).
export KYTY_FPS_HUD="${KYTY_FPS_HUD:-1}"
# The Steam Deck build's defaults (the Linux package's are a PC's): the write watch by userfaultfd (write protection, and
# minor mode for the GPU's read protections), the sound through ALSA (PipeWire plays it; PulseAudio's main loop took
# ~0.35 of a CPU more), address loads sharing page walks, the render thread making the upload worker's copies while it
# waits, and the render thread alone on the Deck's second core.
export KYTY_UFFD_WP="${KYTY_UFFD_WP:-1}"
export KYTY_UFFD_MINOR="${KYTY_UFFD_MINOR:-1}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-alsa,pulseaudio}"
export KYTY_BDA_GROUP_WALKS="${KYTY_BDA_GROUP_WALKS:-1}"
export KYTY_UPLOAD_HELP="${KYTY_UPLOAD_HELP:-1}"
export KYTY_RENDER_CORE="${KYTY_RENDER_CORE:-2}"
# (run_ds.sh's own: the render thread off CPU 0.)
export KYTY_RENDER_CPUS="${KYTY_RENDER_CPUS:-1,2,3,4,5,6,7}"
# Live commands for tuning over SSH (`measure 20` and the like, written to this file).
export KYTY_LIVE_FILE="${KYTY_LIVE_FILE:-$here/live-commands.txt}"
# A development build's translator identity (deck.sh deploy writes translator-id): the shaders it translated stay cached
# across its builds (without it a new executable translates every shader again).
[ -f translator-id ] && export KYTY_PREFETCH_CACHE_BUILD="${KYTY_PREFETCH_CACHE_BUILD:-$(cat translator-id)}"
mkdir -p logs
log="logs/$(date +%Y%m%d-%H%M%S).log"
echo "log: $here/$log"
./kyty_emulator --game "$KYTY_GAME" --screen-width 1280 --screen-height 800 --fullscreen \
  --console-language "${KYTY_CONSOLE_LANGUAGE:-1}" \
  --shader-validation false --shader-log-direction Silent --printf-direction Silent \
  --vblank-frequency 60 --present-mode "${PRESENT_MODE:-Mailbox}" "$@" 2>&1 | tee "$log"
