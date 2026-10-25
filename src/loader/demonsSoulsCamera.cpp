// (10-25) Demon's Souls' free camera turn ramp, removed for keyboard and mouse (KYTY_KBM_NO_RAMP).
//
// The free camera's stick step (ChrExFollowCam, 01.007.000 eboot+0x18cf00; this = the follow camera, [this] = its
// camera data cd, dt at this+0x44, the right stick at this+0x120 (pitch) / +0x124 (yaw), its length at this+0x130):
//   if |stick| > 0.998 and the stick is mostly horizontal (|yaw| above a function of cd+0x220):
//     ramp = cd+0x128 += dt / cd+0x1d0 (towards +-1 with the stick's sign, from 0 on a reversal), and
//     yaw += sign(stick) * dt * lerp(cd+0x1f0, cd+0x208, |ramp|)     (the stick's size not used: full tilt)
//     pitch eased towards a position from the stick's vertical part
//   else:
//     ramp = 0
//     pitch += dt * cd+0x1ec * stick.pitch, yaw += dt * cd+0x1f0 * stick.yaw   (linear, no ramp)
// cd+0x1ec/0x1f0 (base) and 0x204/0x208 (top) are lerp(slowest, fastest, t^2) with t = the options' camera speed / 10,
// clamped to 1 (eboot+0x181b75 and eboot+0x192cc1: speeds above 10 turn as 10 does).
// So a full stick turns at the base rate and then speeds up for cd+0x1d0 s to the top rate, and stops dead when the
// stick leaves full tilt: the mouse saturates the stick at a quick swipe, so the camera accelerated over ~0.3 s (the
// video: 26, 27, 28 ... 35 px a frame) and stopped at once on release.
// The patch (only with keyboard and mouse): the full-tilt branch is never taken (jbe -> jmp to "ramp = 0"), so every
// stick takes the linear branch: an immediate yaw rate proportional to the stick. KYTY_KBM_NO_RAMP=1 (default) also
// makes the linear yaw use the top rate cd+0x208 (a full stick turns at the game's fastest at once), =2 keeps the base
// rate cd+0x1f0 (slower), =0 leaves the game's code.
#include "loader/demonsSoulsCamera.h"

#include "common/virtualMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Loader::DemonsSoulsCamera {
namespace {

// vmovss xmm0, [rbx + 0x130]; vucomiss xmm0, [rip + 0.998f]; jbe <ramp = 0>; vbroadcastss xmm1, [rip + abs mask];
// vmovss xmm0, [rbx + 0x124].
constexpr const char* FullTilt =
    "c5 fa 10 83 30 01 00 00 c5 f8 2e 05 ?? ?? ?? ?? 0f 86 ?? ?? ?? ?? c4 e2 79 18 0d ?? ?? ?? ?? c5 fa 10 83 24 01 00 00";
// ramp = 0: mov dword [rax + 0x128], 0; vmovss xmm0, [rbx + 0x44]; vmulss xmm1, xmm0, [rax + 0x1ec]; lea rcx, [rip];
// vmulss xmm0, xmm0, [rax + 0x1f0]; lea rdi, [rip]; lea rsi, [rip]; vmulss xmm1, xmm1, [rbx + 0x120];
// vmulss xmm0, xmm0, [rbx + 0x124].
constexpr const char* Linear = "c7 80 28 01 00 00 00 00 00 00 c5 fa 10 43 44 c5 fa 59 88 ec 01 00 00 48 8d 0d ?? ?? ?? ?? "
                               "c5 fa 59 80 f0 01 00 00 48 8d 3d ?? ?? ?? ?? 48 8d 35 ?? ?? ?? ?? c5 f2 59 8b 20 01 00 00 "
                               "c5 fa 59 83 24 01 00 00";
constexpr uint64_t JbeOffset      = 16; // in FullTilt
constexpr uint64_t YawRateOffset  = 34; // in Linear: the disp32 of [rax + 0x1f0]
constexpr uint32_t BaseYawRate    = 0x1f0;
constexpr uint32_t TopYawRate     = 0x208;

bool Is1(const char* name) {
	const char* v = std::getenv(name);
	return v != nullptr && v[0] == '1' && v[1] == '\0';
}

} // namespace

int NoRampMode() {
	static const int mode = [] {
		if (!Is1("KYTY_KBM") && !Is1("KYTY_KBM_CALIBRATE")) return 0;
		const char* v = std::getenv("KYTY_KBM_NO_RAMP");
		if (v == nullptr || *v == '\0') return 1;
		const int m = std::atoi(v);
		return m == 0 || m == 2 ? m : 1;
	}();
	return mode;
}

void InstallNoRamp(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin")
		return;
	const int mode = NoRampMode();
	if (mode == 0) {
		if (Is1("KYTY_KBM") || Is1("KYTY_KBM_CALIBRATE"))
			std::printf("Demon's Souls camera: the game's full-tilt turn ramp kept (KYTY_KBM_NO_RAMP=0)\n");
		std::fflush(stdout);
		return;
	}
	const auto tilt   = GuestCode::FindUnique(*program, GuestCode::Pattern(FullTilt));
	const auto linear = GuestCode::FindUnique(*program, GuestCode::Pattern(Linear));
	bool       ok     = tilt.has_value() && linear.has_value();
	uint64_t   jbe    = 0;
	if (ok) {
		// The compare's constant is 0.998f, the jbe lands on the linear branch's "ramp = 0".
		int32_t disp = 0, rel = 0;
		std::memcpy(&disp, reinterpret_cast<const void*>(*tilt + 12), sizeof(disp));
		std::memcpy(&rel, reinterpret_cast<const void*>(*tilt + JbeOffset + 2), sizeof(rel));
		const uint64_t constant = *tilt + 16 + static_cast<int64_t>(disp);
		float          limit    = 0.0f;
		std::memcpy(&limit, reinterpret_cast<const void*>(constant), sizeof(limit));
		jbe = *tilt + JbeOffset;
		ok  = limit > 0.99f && limit < 1.0f && jbe + 6 + static_cast<int64_t>(rel) == *linear;
	}
	if (!ok) {
		std::printf("Demon's Souls camera: the free camera's turn ramp of a known build not found; the game's ramp kept "
		            "(KYTY_KBM_NO_RAMP)\n");
		std::fflush(stdout);
		return;
	}
	// jmp <ramp = 0>; nop (the 6-byte jbe).
	const int32_t rel = static_cast<int32_t>(static_cast<int64_t>(*linear) - static_cast<int64_t>(jbe + 5));
	uint8_t       jmp[6] {0xe9, 0, 0, 0, 0, 0x90};
	std::memcpy(jmp + 1, &rel, sizeof(rel));
	std::memcpy(reinterpret_cast<void*>(jbe), jmp, sizeof(jmp));
	Common::VirtualMemory::FlushInstructionCache(jbe, sizeof(jmp));
	if (mode == 1) {
		const uint64_t at = *linear + YawRateOffset;
		std::memcpy(reinterpret_cast<void*>(at), &TopYawRate, sizeof(TopYawRate));
		Common::VirtualMemory::FlushInstructionCache(at, sizeof(TopYawRate));
	}
	std::printf("Demon's Souls camera (keyboard and mouse): the free camera's full-tilt turn ramp removed (eboot+0x%llx: a "
	            "full stick no longer speeds up from the base turn rate over the camera's ramp time, nor stops dead when it "
	            "leaves full tilt); the yaw is linear in the stick at the game's %s turn rate (camera data +0x%x), pitch "
	            "at its base rate (KYTY_KBM_NO_RAMP=%d; 0 the game's ramp, 1 top rate, 2 base rate)\n",
	            static_cast<unsigned long long>(jbe - program->base_vaddr), mode == 1 ? "top" : "base",
	            mode == 1 ? TopYawRate : BaseYawRate, mode);
	std::fflush(stdout);
#endif
}

} // namespace Loader::DemonsSoulsCamera
