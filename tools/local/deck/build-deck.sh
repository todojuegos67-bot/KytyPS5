#!/usr/bin/env bash
# The Steam Deck build (KYTY_STEAM_DECK, -march=znver2: the Deck's Zen 2) in _Build/deck, with build-linux.sh, in WSL or
# on Linux. SteamOS 3.9 has glibc 2.43 and libstdc++ 6.0.35: a build on a distribution no newer than that runs there
# (Ubuntu 26.04 in WSL matches it). Targets as build-linux.sh takes them (default kyty_emulator).
# _Build/deck/translator-id: a hash of the shader translator's sources, which deckbench.py gives the emulator as
# KYTY_PREFETCH_CACHE_BUILD: builds that leave the translator alone keep the Deck's cache of translated shaders.
# The PGO profile: tools/pgo/kyty-deck.profdata (trained on a Steam Deck) where there is one, else kyty-linux.profdata
# (a PC's); KYTY_DECK_PGO=<file> another, KYTY_DECK_PGO=none none; KYTY_DECK_BUILD_DIR another build directory.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"
BUILD="${KYTY_DECK_BUILD_DIR:-$ROOT/_Build/deck}"
args="-DKYTY_STEAM_DECK=ON -DKYTY_BUILD_LINUX_LAUNCHER=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5"
profile="${KYTY_DECK_PGO:-}"
if [ -z "$profile" ]; then
	profile="$ROOT/tools/pgo/kyty-deck.profdata"
	[ -f "$profile" ] || profile="$ROOT/tools/pgo/kyty-linux.profdata"
fi
[ "$profile" != none ] && [ -f "$profile" ] || profile=''
args+=" -DKYTY_PGO_USE=$profile"
# (A configured directory keeps its first profile: configured again when another one is asked for.)
if [ -f "$BUILD/CMakeCache.txt" ] && ! grep -qx "KYTY_PGO_USE:FILEPATH=$profile" "$BUILD/CMakeCache.txt"; then
	cmake -S "$ROOT" -B "$BUILD" -DKYTY_PGO_USE="$profile" > /dev/null
fi
[ $# -gt 0 ] || set -- kyty_emulator
KYTY_MARCH=znver2 KYTY_BUILD_DIR="$BUILD" KYTY_CMAKE_ARGS="$args" "$ROOT/build-linux.sh" "$@"
(cd "$ROOT" && find src/graphics/shader src/graphics/host_gpu/renderer/pipeline -type f \( -name '*.cpp' -o -name '*.h' \) |
	sort | xargs cat | sha1sum | cut -c1-16) > "$BUILD/translator-id"
echo "translator $(cat "$BUILD/translator-id"), PGO profile ${profile:-none}"
