#!/usr/bin/env bash
# Linux build: clang + Ninja, like the Windows one (build-windows.cmd).
#   ./build-linux.sh              configure (first time) and build kyty_emulator and kyty_launcher (the settings window)
#   ./build-linux.sh <target>...  build other targets (for example all, kyty_shader_precompile)
#
# Needs Git, Clang, Ninja, CMake, glslangValidator (glslang-tools) and Python 3, plus the SDL2
# development libraries (README.md, "Building on Linux"). Missing submodules are checked out first.
#
# Release is -O3 with ThinLTO. KYTY_MARCH (default x86-64-v3, AVX2/FMA/BMI2) is the -march the whole
# build uses: x86-64-v3 makes an executable other PCs can run, native also takes this CPU's newer
# extensions (which crash CPUs without them). KYTY_BUILD_DIR overrides the build directory
# (default _Build/linux), KYTY_CMAKE_ARGS adds arguments to the first configure (e.g. -DKYTY_PGO_USE=).
# The Qt launcher is off: this builds the emulator alone, as the release packages do.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
BUILD="${KYTY_BUILD_DIR:-$ROOT/_Build/linux}"
MARCH="${KYTY_MARCH:-x86-64-v3}"

if [ ! -d "$ROOT/.git" ]; then
	echo "No Git checkout: clone the repository with git (a source archive has no submodules)." >&2
	exit 1
fi
for tool in git clang clang++ cmake ninja glslangValidator python3; do
	command -v "$tool" >/dev/null 2>&1 || { echo "$tool not found (README.md, 'Building on Linux')." >&2; exit 1; }
done

# The submodules a clone without --recurse-submodules lacks (only those: one checked out at another
# commit on purpose stays as it is).
if git -C "$ROOT" submodule status | grep -q '^-'; then
	echo "Checking out submodules..."
	git -C "$ROOT" submodule update --init --recursive
fi

# xbyak and zydis are fetched by CMake at configure time, unless copies are under _Build/deps.
DEPS_ARGS=()
[ -f "$ROOT/_Build/deps/xbyak-src/CMakeLists.txt" ] && DEPS_ARGS+=(-DFETCHCONTENT_SOURCE_DIR_XBYAK="$ROOT/_Build/deps/xbyak-src")
[ -f "$ROOT/_Build/deps/zydis-src/CMakeLists.txt" ] && DEPS_ARGS+=(-DFETCHCONTENT_SOURCE_DIR_ZYDIS="$ROOT/_Build/deps/zydis-src")

if [ ! -f "$BUILD/build.ninja" ]; then
	echo "Configuring $BUILD (-march=$MARCH)"
	cmake -S "$ROOT" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
		-DKYTY_BUILD_LAUNCHER=OFF \
		-DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -march=$MARCH" \
		-DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=$MARCH -flto=thin" \
		-DCMAKE_EXE_LINKER_FLAGS="-flto=thin" \
		"${DEPS_ARGS[@]}" ${KYTY_CMAKE_ARGS:-}
fi

[ $# -gt 0 ] || set -- kyty_emulator kyty_launcher
cmake --build "$BUILD" --target "$@" --parallel
