// Demon's Souls 01.007.000 has a debug console variable cp11_debug_skipSplashScreens ("Skip Splash Screens"; its
// registration is at eboot+0x7cda63, value byte at eboot+0x2e9fd38, default 0, read through the variable's value pointer
// at eboot+0x2e9fcb0). Its only reader is the frontend boot step at eboot+0x7c6f0a, which sets frontend flag 0x3c
// ("play the splash screens", the flag table at [eboot+0x27f0498] + 0x24 + 4 * 0x3c, written by eboot+0x484aa0):
//   mov rax, [skipSplashScreens.value]; cmp byte [rax], 0; jne no_splash      ; <- patched to jmp
//   cmp qword [eboot+0x2ea0a50], 0; jne no_splash; cmp byte [eboot+0x2ea0a68], 1; mov esi, 1; jne set
//   no_splash: xor esi, esi
//   set: mov edi, 0x3c; call SetFrontendFlag
// The splash-screen movies (movies/ps_studios_logo.bk2, logo_intro.bk2) are named by the game's data, not by code, so the
// flag the data checks is what turns them off: the jne becomes a jmp, as if the switch were on (eboot+0x7c6f14).
//
// The opening in-engine cinematic (scn080000 "voidwalk", cutscene_scn080000_00; table entry built at eboot+0x7b8033) is
// started by the game's data, so it is ended the way the player ends it: the game's Hold-to-Skip. Its per-frame update
// (eboot+0x158c150) asks whether the cinematic can be skipped (eboot+0x158bda0, cp11_cutscenes_cinematicSkipMode), then
// advances the hold (float at this+0x118 against 1.0, the X button's value > 0.99) into cl = "skip now", stores cl at
// this+0x120 (eboot+0x158c351) and skips when it is 1 (eboot+0x158c40f). At eboot+0x158c341 `test cl, cl; jne store`
// becomes `mov cl, 1; jmp store`: the first update of a skippable cinematic skips it, as a completed hold would. That
// would skip every cinematic, so the patch is one-shot: Disarm() (called from the pad read, libs/controller.cpp, a few
// seconds after the first cinematic or 4 minutes after boot) puts the original bytes back.
#include "loader/demonsSoulsSkipIntro.h"

#include "common/virtualMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Loader::DemonsSoulsSkipIntro {
namespace {

constexpr const char* BootStep = "48 8b 05 ?? ?? ?? ?? 80 38 00 75 18 48 83 3d ?? ?? ?? ?? 00 75 0e 80 3d ?? ?? ?? ?? 01 "
                                 "be 01 00 00 00 75 02 31 f6 bf 3c 00 00 00 e8";
constexpr uint64_t JumpOffset = 10; // the jne (75 18)

constexpr const char* HoldUpdate = "76 16 c5 fa 10 0d ?? ?? ?? ?? b1 01 c7 83 18 01 00 00 00 00 80 3f eb 10 84 c9 75 0c c5 f8 2e c1 "
                                   "75 06 0f 8b ?? ?? ?? ?? 88 8b 20 01 00 00 a8 02";
constexpr uint64_t                HoldOffset = 24; // test cl, cl; jne +0xc
constexpr std::array<uint8_t, 4> HoldOriginal {0x84, 0xc9, 0x75, 0x0c};
constexpr std::array<uint8_t, 4> HoldSkip {0xb1, 0x01, 0xeb, 0x0c}; // mov cl, 1; jmp +0xc
std::atomic<uint64_t>             g_hold_site {0};
uint64_t                          g_base = 0;

void InstallCinematicSkip(Program* program) {
	const auto found = GuestCode::FindUnique(*program, GuestCode::Pattern(HoldUpdate));
	if (!found) {
		std::printf("[port] intro: cinematic skip patch NOT applied (hold-to-skip bytes not found: not 01.007.000?)\n");
		return;
	}
	auto* site = reinterpret_cast<uint8_t*>(*found + HoldOffset);
	if (std::memcmp(site, HoldOriginal.data(), HoldOriginal.size()) != 0) {
		std::printf("[port] intro: cinematic skip patch NOT applied (unexpected bytes)\n");
		return;
	}
	std::memcpy(site, HoldSkip.data(), HoldSkip.size());
	Common::VirtualMemory::FlushInstructionCache(reinterpret_cast<uint64_t>(site), HoldSkip.size());
	g_base = program->base_vaddr;
	g_hold_site.store(reinterpret_cast<uint64_t>(site), std::memory_order_release);
	std::printf("[port] intro: cinematic skip patch applied at eboot+0x%llx (the first skippable cinematic ends at once)\n",
	            static_cast<unsigned long long>(reinterpret_cast<uint64_t>(site) - g_base));
}

} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	const char* v = std::getenv("KYTY_SKIP_INTRO");
	if (v == nullptr || v[0] == '\0' || v[0] == '0') return;
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin")
		return;
	InstallCinematicSkip(program);
	const auto step = GuestCode::FindUnique(*program, GuestCode::Pattern(BootStep));
	if (!step) {
		std::printf("[port] intro: skip patch NOT applied (boot step bytes not found: not 01.007.000?)\n");
		return;
	}
	auto* jump = reinterpret_cast<uint8_t*>(*step + JumpOffset);
	if (jump[0] != 0x75 || jump[1] != 0x18) {
		std::printf("[port] intro: skip patch NOT applied (unexpected bytes %02x %02x)\n", jump[0], jump[1]);
		return;
	}
	// Installation occurs before module initializers or guest threads run; program memory is writable to the host.
	jump[0] = 0xeb;
	Common::VirtualMemory::FlushInstructionCache(reinterpret_cast<uint64_t>(jump), 2);
	std::printf("[port] intro: skip patch applied at eboot+0x%llx (cp11_debug_skipSplashScreens forced on)\n",
	            static_cast<unsigned long long>(reinterpret_cast<uint64_t>(jump) - program->base_vaddr));
#endif
}

bool CinematicSkipArmed() {
	return g_hold_site.load(std::memory_order_acquire) != 0;
}

void Disarm(const char* reason) {
	const uint64_t site = g_hold_site.exchange(0, std::memory_order_acq_rel);
	if (site == 0) return;
	auto* bytes = reinterpret_cast<uint8_t*>(site);
	if (std::memcmp(bytes, HoldSkip.data(), HoldSkip.size()) == 0) {
		std::memcpy(bytes, HoldOriginal.data(), HoldOriginal.size());
		Common::VirtualMemory::FlushInstructionCache(site, HoldOriginal.size());
	}
	std::printf("[port] intro: cinematic skip patch removed at eboot+0x%llx (%s); later cinematics play normally\n",
	            static_cast<unsigned long long>(site - g_base), reason);
}

} // namespace Loader::DemonsSoulsSkipIntro
