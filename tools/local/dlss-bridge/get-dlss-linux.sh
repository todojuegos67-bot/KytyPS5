#!/usr/bin/env bash
# get-dlss-linux.sh: DLSS super resolution, DLAA and DLSS frame generation for the Linux package (KYTY_UPSCALER=dlss,
# KYTY_FRAMEGEN). The package carries none of NVIDIA's files: this downloads NVIDIA's DLSS SDK (github.com/NVIDIA/DLSS,
# under NVIDIA's license) and builds the bridge libkyty_dlss.so on this PC from its source (dlss-bridge/, MIT), then puts
# it with the SDK's libnvidia-ngx-dlss.so.* and libnvidia-ngx-dlssg.so.* next to kyty_emulator. Once is enough (again:
# another SDK). In a checkout, tools/local/dlss-bridge/build-dlss-linux.sh does the same for _Build/linux.
#
#   ./get-dlss-linux.sh [folder]     (default: this script's folder, the package's)
#
# It needs git and a C++ compiler (g++ or clang++): what is missing it offers to install with the distribution's
# package manager (pacman, apt, dnf, zypper or xbps; sudo asks for the password). Vulkan's headers come from the system
# or are downloaded too. DLSS_SDK_REF: another SDK version (a tag of github.com/NVIDIA/DLSS; default v310.9.16).
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SDK_REF="${DLSS_SDK_REF:-v310.9.16}"
VULKAN_REF=v1.4.335

# The bridge's source: dlss-bridge/ in the package, or this folder and src/local in a checkout.
if [ -f "$HERE/dlss-bridge/kyty_dlss_bridge.cpp" ]; then
	SOURCE="$HERE/dlss-bridge"
	HEADER_DIR="$HERE/dlss-bridge"
	TARGET="${1:-$HERE}"
else
	SOURCE="$HERE"
	HEADER_DIR="$(cd "$HERE/../../../src/local" && pwd)"
	TARGET="${1:-$(cd "$HERE/../../.." && pwd)/_Build/linux}"
fi
[ -f "$SOURCE/kyty_dlss_bridge.cpp" ] && [ -f "$HEADER_DIR/kyty-dlss-bridge.h" ] ||
	{ echo "the bridge's source (kyty_dlss_bridge.cpp, kyty-dlss-bridge.h) is missing" >&2; exit 1; }
[ -d "$TARGET" ] || { echo "no folder $TARGET" >&2; exit 1; }
[ -f "$TARGET/kyty_emulator" ] || echo "(no kyty_emulator in $TARGET: the libraries go there anyway)"

find_compiler() { CXX="${CXX:-$(command -v g++ || command -v clang++ || true)}"; }
find_compiler
# What is missing, with the package manager's names for it.
missing=()
command -v git >/dev/null 2>&1 || missing+=(git)
[ -n "$CXX" ] || missing+=(compiler)
if [ ${#missing[@]} -gt 0 ]; then
	packages() { # the package names: $1 the compiler's
		local name
		for name in "${missing[@]}"; do if [ "$name" = compiler ]; then echo "$1"; else echo "$name"; fi; done
	}
	install=()
	if command -v pacman >/dev/null 2>&1; then
		install=(pacman -S --needed --noconfirm $(packages gcc))
	elif command -v apt-get >/dev/null 2>&1; then
		install=(apt-get install -y $(packages g++))
	elif command -v dnf >/dev/null 2>&1; then
		install=(dnf install -y $(packages gcc-c++))
	elif command -v zypper >/dev/null 2>&1; then
		install=(zypper --non-interactive install $(packages gcc-c++))
	elif command -v xbps-install >/dev/null 2>&1; then
		install=(xbps-install -Sy $(packages gcc))
	fi
	[ ${#install[@]} -gt 0 ] ||
		{ echo "git and a C++ compiler (g++) are needed, and this distribution's package manager is not one this script knows: install them, then run it again" >&2; exit 1; }
	[ "$(id -u)" = 0 ] || install=(sudo "${install[@]}")
	# (Asked first: it changes the system. Without a terminal to ask on, the command is shown instead.)
	needs=()
	for name in "${missing[@]}"; do if [ "$name" = compiler ]; then needs+=('a C++ compiler'); else needs+=("$name"); fi; done
	list="${needs[0]}"
	[ ${#needs[@]} -lt 2 ] || list="${needs[0]} and ${needs[1]}"
	echo "DLSS's build needs $list: ${install[*]}"
	answer=''
	[ -t 0 ] && read -r -p "Install it now? [y/N] " answer
	[[ "$answer" =~ ^[Yy] ]] || { echo "install it (the command above), then run this again" >&2; exit 1; }
	[ "${install[0]}" != sudo ] || [ "${install[1]}" != apt-get ] || sudo apt-get update -qq || true
	"${install[@]}" ||
		{ echo "the install failed (a read-only system such as SteamOS? there: a distrobox, or install git and g++ some other way), then run it again" >&2; exit 1; }
	hash -r
	find_compiler
	command -v git >/dev/null 2>&1 && [ -n "$CXX" ] || { echo "git or the C++ compiler is still missing after the install" >&2; exit 1; }
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/kyty-dlss.XXXX")"
trap 'rm -rf "$WORK"' EXIT

# Only what the bridge needs: the headers, the static NGX loader, the Linux DLSS libraries and the license.
echo "Downloading NVIDIA's DLSS SDK $SDK_REF (github.com/NVIDIA/DLSS)"
SDK="$WORK/dlss-sdk"
git clone --quiet --filter=blob:none --no-checkout --depth 1 --branch "$SDK_REF" https://github.com/NVIDIA/DLSS "$SDK"
git -C "$SDK" sparse-checkout set --no-cone '/include/*' '/lib/Linux_x86_64/libnvsdk_ngx.a' \
	'/lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.*' '/lib/Linux_x86_64/rel/libnvidia-ngx-dlssg.so.*' '/LICENSE.txt'
git -C "$SDK" checkout --quiet
library=$(ls "$SDK"/lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.* 2>/dev/null | sort -V | tail -1)
library_fg=$(ls "$SDK"/lib/Linux_x86_64/rel/libnvidia-ngx-dlssg.so.* 2>/dev/null | sort -V | tail -1)
[ -n "$library" ] && [ -n "$library_fg" ] && [ -f "$SDK/lib/Linux_x86_64/libnvsdk_ngx.a" ] ||
	{ echo "the download has no Linux DLSS libraries (DLSS_SDK_REF=$SDK_REF)" >&2; exit 1; }

# Vulkan's headers: the system's, else Khronos' (the version the emulator is built with).
VULKAN_INCLUDE=''
for dir in "${VULKAN_SDK:+$VULKAN_SDK/include}" /usr/include /usr/local/include; do
	[ -n "$dir" ] && [ -f "$dir/vulkan/vulkan.h" ] && [ -f "$dir/vk_video/vulkan_video_codecs_common.h" ] &&
		{ VULKAN_INCLUDE="$dir"; break; }
done
if [ -z "$VULKAN_INCLUDE" ]; then
	echo "Downloading Vulkan's headers $VULKAN_REF (github.com/KhronosGroup/Vulkan-Headers)"
	git clone --quiet --filter=blob:none --no-checkout --depth 1 --branch "$VULKAN_REF" \
		https://github.com/KhronosGroup/Vulkan-Headers "$WORK/vulkan-headers"
	git -C "$WORK/vulkan-headers" sparse-checkout set include
	git -C "$WORK/vulkan-headers" checkout --quiet
	VULKAN_INCLUDE="$WORK/vulkan-headers/include"
fi

echo "Building libkyty_dlss.so ($(basename "$CXX"))"
# (As tools/local/dlss-bridge/CMakeLists.txt: libnvsdk_ngx.a loads the driver's libnvidia-ngx.so.1 at run time. Built
# on this PC, it takes this PC's libstdc++.so: no static C++ library, a separate package on some distributions.)
"$CXX" -std=c++20 -O2 -DNDEBUG -fPIC -shared -fvisibility=hidden -Wno-format-security -o "$WORK/libkyty_dlss.so" "$SOURCE/kyty_dlss_bridge.cpp" \
	-I"$HEADER_DIR" -I"$SDK/include" -I"$VULKAN_INCLUDE" \
	"$SDK/lib/Linux_x86_64/libnvsdk_ngx.a" -ldl -Wl,--no-undefined

rm -f "$TARGET"/libnvidia-ngx-dlss.so.* "$TARGET"/libnvidia-ngx-dlssg.so.*
cp "$WORK/libkyty_dlss.so" "$library" "$library_fg" "$TARGET/"
cp "$SDK/LICENSE.txt" "$TARGET/LICENSE-DLSS-SDK.txt"
echo "DLSS: $TARGET/libkyty_dlss.so, $(basename "$library"), $(basename "$library_fg") (NVIDIA's license: LICENSE-DLSS-SDK.txt)"
echo "Then: KYTY_UPSCALER=dlss (KYTY_RENDER_HEIGHT 2160 DLAA, 1440 Quality, 1080 Performance), KYTY_FRAMEGEN=1 (2x)"
