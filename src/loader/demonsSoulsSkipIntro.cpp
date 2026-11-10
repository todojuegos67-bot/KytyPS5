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
#include "loader/demonsSoulsSkipIntro.h"

#include "common/virtualMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Loader::DemonsSoulsSkipIntro {
namespace {

constexpr const char* BootStep = "48 8b 05 ?? ?? ?? ?? 80 38 00 75 18 48 83 3d ?? ?? ?? ?? 00 75 0e 80 3d ?? ?? ?? ?? 01 "
                                 "be 01 00 00 00 75 02 31 f6 bf 3c 00 00 00 e8";
constexpr uint64_t JumpOffset = 10; // the jne (75 18)

} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	const char* v = std::getenv("KYTY_SKIP_INTRO");
	if (v == nullptr || v[0] == '\0' || v[0] == '0') return;
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin")
		return;
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

} // namespace Loader::DemonsSoulsSkipIntro
