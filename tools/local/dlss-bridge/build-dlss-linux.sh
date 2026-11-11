#!/usr/bin/env bash
# NVIDIA DLSS on Linux (KYTY_UPSCALER=dlss): fetches NVIDIA's DLSS SDK (headers, the static NGX loader and the Linux DLSS
# libraries; github.com/NVIDIA/DLSS, under NVIDIA's license), builds the bridge (libkyty_dlss.so) and copies it with
# libnvidia-ngx-dlss.so.* and libnvidia-ngx-dlssg.so.* next to kyty_emulator.
#
#   tools/local/dlss-bridge/build-dlss-linux.sh [folder ...]
#
# The folders get the libraries (default: _Build/linux, where build-linux.sh puts kyty_emulator); a release folder can be
# given too. DLSS_SDK_DIR: an SDK checkout to use instead of _Build/deps/dlss-sdk.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"
SDK="${DLSS_SDK_DIR:-$ROOT/_Build/deps/dlss-sdk}"
BUILD="$ROOT/_Build/dlss-bridge"

for tool in git cmake ninja; do
	command -v "$tool" >/dev/null 2>&1 || { echo "$tool not found" >&2; exit 1; }
done
CXX="${CXX:-$(command -v clang++ || command -v g++ || true)}"
[ -n "$CXX" ] || { echo "no C++ compiler (clang++ or g++)" >&2; exit 1; }

have_sdk() {
	[ -f "$SDK/include/nvsdk_ngx_vk.h" ] && [ -f "$SDK/lib/Linux_x86_64/libnvsdk_ngx.a" ] &&
		compgen -G "$SDK/lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.*" >/dev/null &&
		compgen -G "$SDK/lib/Linux_x86_64/rel/libnvidia-ngx-dlssg.so.*" >/dev/null
}
if ! have_sdk; then
	if [ -n "${DLSS_SDK_DIR:-}" ]; then
		echo "DLSS_SDK_DIR=$SDK is not a DLSS SDK checkout (include/, lib/Linux_x86_64/)" >&2
		exit 1
	fi
	echo "Fetching the NVIDIA DLSS SDK into $SDK"
	rm -rf "$SDK"
	mkdir -p "$(dirname "$SDK")"
	git clone --filter=blob:none --no-checkout --depth 1 https://github.com/NVIDIA/DLSS "$SDK"
	git -C "$SDK" sparse-checkout set --no-cone '/include/*' '/lib/Linux_x86_64/libnvsdk_ngx.a' \
		'/lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.*' '/lib/Linux_x86_64/rel/libnvidia-ngx-dlssg.so.*' '/LICENSE.txt'
	git -C "$SDK" checkout
fi
library=$(ls "$SDK"/lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.* | sort -V | tail -1)
library_fg=$(ls "$SDK"/lib/Linux_x86_64/rel/libnvidia-ngx-dlssg.so.* | sort -V | tail -1)

[ -f "$ROOT/3rdparty/Vulkan-Headers/include/vulkan/vulkan.h" ] || git -C "$ROOT" submodule update --init 3rdparty/Vulkan-Headers
cmake -S "$ROOT/tools/local/dlss-bridge" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" \
	-DDLSS_SDK_ROOT="$SDK" -DVULKAN_INCLUDE="$ROOT/3rdparty/Vulkan-Headers/include" >/dev/null
ninja -C "$BUILD" >/dev/null

targets=("$@")
[ ${#targets[@]} -gt 0 ] || targets=("$ROOT/_Build/linux")
for target in "${targets[@]}"; do
	if [ ! -d "$target" ]; then
		echo "skipped $target (no such folder)" >&2
		continue
	fi
	rm -f "$target"/libnvidia-ngx-dlss.so.* "$target"/libnvidia-ngx-dlssg.so.*
	cp "$BUILD/libkyty_dlss.so" "$library" "$library_fg" "$target/"
	cp "$SDK/LICENSE.txt" "$target/LICENSE-DLSS-SDK.txt"
	echo "DLSS: $target/libkyty_dlss.so, $(basename "$library"), $(basename "$library_fg")"
done
