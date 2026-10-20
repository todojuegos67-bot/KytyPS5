#include "loader/demonsSoulsSuperResolution.h"

#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"
#include "local/frame-gen.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace Loader::DemonsSoulsSuperResolution {
namespace {

// The renderer's constructor: `if (IsTrinity()) mSuperResolution = new SuperResolution` (0x88 bytes; 01.007.000
// eboot+0x9e4b33): the call, its test, the allocation and the constructor's call.
constexpr std::string_view CreateCheck = "e8 ?? ?? ?? ?? 84 c0 74 ?? bf 88 00 00 00 e8 ?? ?? ?? ?? 48 89 c7 49 89 c7 e8";
// mov al, 1; nop dword [rax] (the call's five bytes).
constexpr uint8_t Taken[5] {0xb0, 0x01, 0x0f, 0x1f, 0x00};

// SuperResolution::IsEnabled (01.007.000 eboot+0xa53700): display mode 5 and the render target's height (a float at
// +4 of the render job's size) at least 1080.0 (vucomiss [rip + 1080.0f]; setae al): below it the renderer skips the
// upscaler for a plain copy. The compare (8 bytes) becomes cmp dword [rax+4], 720.0f (a positive float compares as an
// integer), setae kept: 720 and above in mode 5. (Any height: the menus' 480x270 background went through it too, the
// upscaler remade for every switch between it and the 3D view, a second a frame.)
constexpr std::string_view EnabledCheck =
    "83 78 3c 05 75 15 48 8b 46 50 c5 fa 10 40 04 c5 f8 2e 05 ?? ?? ?? ?? 0f 93 c0 c3";
constexpr uint8_t AnyHeightBytes[11] {0x81, 0x78, 0x04, 0x00, 0x00, 0x34, 0x44, 0x0f, 0x93, 0xc0, 0x90};

bool g_active     = false;
bool g_any_height = false;

} // namespace

bool Active() {
	return g_active;
}

bool AnyHeight() {
	return g_any_height;
}

void Install(Program* program) {
	const char* upscaler = std::getenv("KYTY_UPSCALER");
	const std::string_view kind = upscaler != nullptr ? std::string_view(upscaler) : std::string_view();
	if ((kind != "dlss" && kind != "xess" && kind != "fsr") || program == nullptr ||
	    !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin")
		return;
	// (The device exists by now: without DLSS on it, the game's own TAA rather than a stretch without anti-aliasing.)
	if (!Libs::Graphics::FrameGen::UpscalerReady()) {
		if (kind == "dlss")
			std::printf("Demon's Souls super resolution: DLSS is not available here (an NVIDIA RTX GPU and NVIDIA's Streamline "
			            "runtime); the game's own TAA kept\n");
		else
			std::printf("Demon's Souls super resolution: %s is not available here (its DLL next to the emulator); the game's "
			            "own TAA kept\n", kind == "xess" ? "XeSS" : "FSR 3.1");
		return;
	}
	const auto site = GuestCode::FindUnique(*program, GuestCode::Pattern(CreateCheck));
	if (!site) {
		std::printf("Demon's Souls super resolution: this build has no PS5 Pro path (01.007.000 has); the game's own TAA kept\n");
		return;
	}
	// Installation occurs before module initializers or guest threads run; program memory is writable to the host.
	std::memcpy(reinterpret_cast<void*>(*site), Taken, sizeof(Taken));
	g_active = true;
	if (const auto enabled = GuestCode::FindUnique(*program, GuestCode::Pattern(EnabledCheck)); enabled) {
		std::memcpy(reinterpret_cast<void*>(*enabled + 15), AnyHeightBytes, sizeof(AnyHeightBytes));
		g_any_height = true;
	} else {
		std::printf("Demon's Souls super resolution: its 1080 least render height check not found; 1080 at least\n");
	}
	std::printf("Demon's Souls super resolution: the PS5 Pro path (display mode 5, its upscaler in place of the TAA) is on\n");
}

} // namespace Loader::DemonsSoulsSuperResolution
