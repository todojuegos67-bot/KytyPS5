#include "loader/demonsSoulsRenderSize.h"
#include "loader/demonsSoulsSuperResolution.h"

#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "loader/elf.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"
#include "local/local-platform.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <span>
#include <vector>

namespace Loader::DemonsSoulsRenderSize {
namespace {

bool Readable(uint64_t address, uint64_t size) {
	return Libs::Graphics::HostMemoryRangeIsReadable(address, size);
}

bool HoldsPair(uint64_t address, uint32_t width, uint32_t height) {
	uint32_t pair[2] {};
	if (!Readable(address, sizeof(pair))) return false;
	std::memcpy(pair, reinterpret_cast<const void*>(address), sizeof(pair));
	return pair[0] == width && pair[1] == height;
}

// The one address of a mode's render size the mode cases (`pattern`) load (both cases load the same constant).
std::optional<uint64_t> FindRenderConstant(const Program& program, std::string_view pattern, uint32_t width,
                                           uint32_t height) {
	std::vector<uint64_t> constants;
	for (const auto site: GuestCode::Find(program, GuestCode::Pattern(pattern), 64)) {
		int32_t output_displacement = 0, render_displacement = 0;
		std::memcpy(&output_displacement, reinterpret_cast<const void*>(site + 4), sizeof(int32_t));
		std::memcpy(&render_displacement, reinterpret_cast<const void*>(site + 12), sizeof(int32_t));
		const uint64_t output = site + 8 + static_cast<int64_t>(output_displacement);
		const uint64_t render = site + 16 + static_cast<int64_t>(render_displacement);
		if (HoldsPair(output, OutputWidth, OutputHeight) && HoldsPair(render, width, height) &&
		    std::find(constants.begin(), constants.end(), render) == constants.end())
			constants.push_back(render);
	}
	return constants.size() == 1 ? std::optional(constants[0]) : std::nullopt;
}

// The one place `bytes` is in the program's loaded segments.
std::optional<uint64_t> FindUniqueData(const Program& program, std::span<const uint8_t> bytes) {
	const auto* elf = program.elf.get();
	if (elf == nullptr || elf->GetEhdr() == nullptr || elf->GetPhdr() == nullptr) return std::nullopt;
	std::vector<uint64_t> found;
	const auto*           phdr = elf->GetPhdr();
	for (size_t i = 0; i < elf->GetEhdr()->e_phnum && found.size() < 2; i++) {
		if (phdr[i].p_type != PT_LOAD || phdr[i].p_filesz < bytes.size()) continue;
		const auto address = program.base_vaddr + phdr[i].p_vaddr;
		if (!Readable(address, phdr[i].p_filesz)) continue;
		const auto* begin = reinterpret_cast<const uint8_t*>(address);
		const auto* end   = begin + phdr[i].p_filesz;
		for (const auto* at = std::search(begin, end, bytes.begin(), bytes.end()); at != end && found.size() < 2;
		     at             = std::search(at + 1, end, bytes.begin(), bytes.end()))
			found.push_back(reinterpret_cast<uint64_t>(at));
	}
	return found.size() == 1 ? std::optional(found[0]) : std::nullopt;
}

double   g_wide_aspect = 0.0; // the aspect KYTY_RENDER_ASPECT installed (0: 16:9)
uint32_t g_wide_output = 0;   // the output's width InstallWideOutput set (0: the game's 3840)

// The RIP-relative target of the disp32 at `site` + `at`, the instruction `length` bytes long.
uint64_t RipTarget(uint64_t site, uint32_t at, uint32_t length) {
	int32_t displacement = 0;
	std::memcpy(&displacement, reinterpret_cast<const void*>(site + at), sizeof(displacement));
	return site + length + static_cast<int64_t>(displacement);
}

// The wide output (WideOutputWidth) for display mode 5 (`mode`), where 01.007.000 has its places (false, nothing
// changed, when one is not found):
// - the output pair its cases load, `width` x 2160. Display mode 4's (Cinematic: its 3D and output, natively) is the same
//   constant: its two loads are moved to the same 3840x2160 in the mode 3 preset (CinematicSource), Cinematic kept as
//   it is (rendered at 5120x2160 natively the game ran out of its memory pools and stopped);
// - the upscaler's context (SuperResolution::InitContext's 3840, 2160);
// - the fixed 3840x2160 the video settings' three record builders store, which sizes the upscaler's image and the
//   processing after it: each mode's own output instead (rcx = xmm1 in the setter and the constructor, rdx = r10 << 32
//   | r9 in the size query);
// - the mode's entry of the output width table.
bool InstallWideOutput(const Program& program, int mode, uint32_t width) {
	std::vector<uint64_t> outputs; // (the setter's and the constructor's mode 5 cases load the same pair)
	for (const auto site: GuestCode::Find(program, GuestCode::Pattern(TrinityCase), 8)) {
		const uint64_t output = RipTarget(site, 4, 8);
		if (HoldsPair(output, OutputWidth, OutputHeight) && std::find(outputs.begin(), outputs.end(), output) == outputs.end())
			outputs.push_back(output);
	}
	if (outputs.size() != 1) return false;
	std::vector<uint64_t> cinematic;
	for (const auto site: GuestCode::Find(program, GuestCode::Pattern(CinematicCase), 8))
		if (RipTarget(site, 4, 8) == outputs[0]) cinematic.push_back(site);
	const auto preset   = FindUniqueData(program, std::span(reinterpret_cast<const uint8_t*>(CinematicSource), sizeof(CinematicSource)));
	const auto contexts = GuestCode::Find(program, GuestCode::Pattern(UpscalerContext), 2);
	const auto setter   = GuestCode::Find(program, GuestCode::Pattern(SetterFixedPair), 2);
	const auto builder  = GuestCode::Find(program, GuestCode::Pattern(ConstructorFixedPair), 2);
	const auto query    = GuestCode::Find(program, GuestCode::Pattern(QueryFixedPair), 2);
	const auto widths   = FindUniqueData(program, std::span(reinterpret_cast<const uint8_t*>(OutputWidths), sizeof(OutputWidths)));
	if (cinematic.size() != 2 || !preset || contexts.size() != 1 || setter.size() != 1 || builder.size() != 1 ||
	    query.size() != 1 || !widths)
		return false;
	// (The 3840x2160 after the preset's 0x80ffffff and 2560x1440; nothing writes the preset.)
	const uint64_t source = *preset + 12;
	for (const auto site: cinematic) {
		const auto displacement = static_cast<int32_t>(static_cast<int64_t>(source) - static_cast<int64_t>(site + 8));
		std::memcpy(reinterpret_cast<void*>(site + 4), &displacement, sizeof(displacement));
	}
	const uint32_t pair[2] {width, OutputHeight};
	std::memcpy(reinterpret_cast<void*>(outputs[0]), pair, sizeof(pair));
	std::memcpy(reinterpret_cast<void*>(contexts[0] + 1), &width, sizeof(width));
	// vmovq rcx, xmm1; nop dword [rax + rax] (10 bytes, as the mov rcx, imm64 they replace).
	constexpr uint8_t from_xmm1[10] {0xc4, 0xe1, 0xf9, 0x7e, 0xc9, 0x0f, 0x1f, 0x44, 0x00, 0x00};
	std::memcpy(reinterpret_cast<void*>(setter[0] + 4), from_xmm1, sizeof(from_xmm1));
	std::memcpy(reinterpret_cast<void*>(builder[0] + 7), from_xmm1, sizeof(from_xmm1));
	// mov edx, r10d; shl rdx, 32; or rdx, r9 (10 bytes, as the mov rdx, imm64).
	constexpr uint8_t from_query[10] {0x44, 0x89, 0xd2, 0x48, 0xc1, 0xe2, 0x20, 0x4c, 0x09, 0xca};
	std::memcpy(reinterpret_cast<void*>(query[0] + 3), from_query, sizeof(from_query));
	std::memcpy(reinterpret_cast<void*>(*widths + mode * sizeof(uint32_t)), &width, sizeof(width));
	return true;
}

// The save's graphics mode read: call (read int); mov r13d, [rbp - 0x218]; mov rdi, r12; call; xor eax, eax;
// cmp r13d, 1; mov edi, 4; mov [r15 + 0x64], r13d.
constexpr std::string_view SavedModeRead =
    "e8 ?? ?? ?? ?? 44 8b ad e8 fd ff ff 4c 89 e7 e8 ?? ?? ?? ?? 31 c0 41 83 fd 01 bf 04 00 00 00 45 89 6f 64";
// The default from the system's game preset: mov r12d, 2; call IsTrinity; test al, al; jne store; mov eax, [rbp-0x54];
// test; je; xor r12d, r12d; cmp eax, 1; setne r12b; store: mov [r14 + 0x64], r12d.
constexpr std::string_view PresetMode =
    "41 bc 02 00 00 00 e8 ?? ?? ?? ?? 84 c0 75 11 8b 45 ac 85 c0 74 1a 45 31 e4 83 f8 01 41 0f 95 c4 45 89 66 64";

} // namespace

void InstallForcedMode(Program* program) {
	const char* setting = std::getenv("KYTY_DS_FORCE_MODE");
	if (setting == nullptr || *setting == 0 || program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    program->file_name.filename() != "eboot.bin")
		return;
	const std::string_view name(setting);
	uint32_t               value = 0;
	if (name == "cinematic") {
		value = 1;
	} else if (name != "performance") {
		std::printf("Demon's Souls mode: KYTY_DS_FORCE_MODE=%s is not cinematic or performance; the game's choice kept\n",
		            setting);
		return;
	}
	const auto saved  = GuestCode::FindUnique(*program, GuestCode::Pattern(SavedModeRead));
	const auto preset = GuestCode::FindUnique(*program, GuestCode::Pattern(PresetMode));
	if (!saved || !preset) {
		std::printf("Demon's Souls mode: the graphics mode's boot reads of a known build not found; the game's choice kept\n");
		return;
	}
	// mov r13d, value; nop (the 7-byte load from the save).
	uint8_t from_save[7] {0x41, 0xbd, 0, 0, 0, 0, 0x90};
	std::memcpy(from_save + 2, &value, sizeof(value));
	std::memcpy(reinterpret_cast<void*>(*saved + 5), from_save, sizeof(from_save));
	// mov r12d, value; jmp store (over the preset's mapping).
	uint8_t from_preset[8] {0x41, 0xbc, 0, 0, 0, 0, 0xeb, 0x18};
	std::memcpy(from_preset + 2, &value, sizeof(value));
	std::memcpy(reinterpret_cast<void*>(*preset), from_preset, sizeof(from_preset));
	const char* ratio = std::getenv("KYTY_UPSCALE_RATIO");
	std::printf("Demon's Souls mode: the game starts in its %s mode (KYTY_DS_FORCE_MODE; still changeable in its options)%s\n",
	            value == 1 ? "cinematic" : "performance",
	            value == 1 && (ratio == nullptr || *ratio == 0) ? ", native 3840x2160" : "");
}

// KYTY_DS_SIXTY_STEP (default on, =0 the game's): the per-mode render setter (01.007.000 eboot 0x90090b3a0) reads a
// frame rate {59.94, 59.94, 29.97, 19.98} (render setup +0x28: the time step 1/rate, eboot 0x900b280e0, that the logo,
// menus, movies and cutscene sequencers advance by each frame) and a flip rate {0, 0, 1, 2} (sceVideoOutSetFlipRate)
// by mode. On the emulator's virtual vblank the flips stay at 60, so Cinematic's 1/30 step ran them twice too fast:
// every mode gets the performance mode's 59.94 and flip rate 0 (Cinematic keeps its 4K picture).
void InstallSixtyStep(Program* program) {
	const char* setting = std::getenv("KYTY_DS_SIXTY_STEP");
	if ((setting != nullptr && std::strcmp(setting, "0") == 0) || program == nullptr ||
	    !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin")
		return;
	const auto site = GuestCode::FindUnique(
	    *program, GuestCode::Pattern("48 8d 0d ?? ?? ?? ?? c5 fa 10 04 81 48 8d 0d ?? ?? ?? ?? 8b 34 81 c5 fa 11 47 28"));
	if (!site) {
		std::printf("Demon's Souls step: the render setter of a known build not found; the game's step kept\n");
		return;
	}
	const uint64_t rates_at = RipTarget(*site, 3, 7);
	const uint64_t flips_at = RipTarget(*site, 15, 19);
	float          rates[4] {};
	int32_t        flips[4] {};
	if (!Readable(rates_at, sizeof(rates)) || !Readable(flips_at, sizeof(flips))) {
		std::printf("Demon's Souls step: the mode tables are not readable; the game's step kept\n");
		return;
	}
	std::memcpy(rates, reinterpret_cast<const void*>(rates_at), sizeof(rates));
	std::memcpy(flips, reinterpret_cast<const void*>(flips_at), sizeof(flips));
	if (rates[0] < 59.0f || rates[0] > 60.5f || flips[0] != 0 || flips[2] != 1) {
		std::printf("Demon's Souls step: unexpected mode tables (%.2f, flip %d/%d); the game's step kept\n",
		            static_cast<double>(rates[0]), flips[0], flips[2]);
		return;
	}
	for (int i = 1; i < 4; i++) {
		rates[i] = rates[0];
		flips[i] = 0;
	}
	std::memcpy(reinterpret_cast<void*>(rates_at), rates, sizeof(rates));
	std::memcpy(reinterpret_cast<void*>(flips_at), flips, sizeof(flips));
	std::printf("Demon's Souls step: every graphics mode at the 59.94 fps step and flip rate 0 (Cinematic's 1/30 step "
	            "and 30 fps flips off; KYTY_DS_SIXTY_STEP=0: the game's)\n");
}

double WideAspect() {
	return g_wide_aspect;
}

uint32_t WideOutputWidth() {
	return g_wide_output;
}

void Install(Program* program) {
	const char* setting = std::getenv("KYTY_RENDER_HEIGHT");
	const char* wide    = std::getenv("KYTY_RENDER_ASPECT");
	const char* ratio_s = std::getenv("KYTY_UPSCALE_RATIO");
	const bool  by_ratio = (setting == nullptr || *setting == 0) && ratio_s != nullptr && *ratio_s != 0;
	if (((setting == nullptr || *setting == 0) && !by_ratio && (wide == nullptr || *wide == 0)) || program == nullptr ||
	    !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin")
		return;
	// (The game's super resolution path: display mode 5 is the performance mode, loader/demonsSoulsSuperResolution.h.)
	const bool     trinity     = DemonsSoulsSuperResolution::Active();
	const uint32_t game_width  = trinity ? TrinityWidth : RenderWidth;
	const uint32_t game_height = trinity ? TrinityHeight : RenderHeight;
	const int      mode        = trinity ? TrinityMode : PerformanceMode;
	auto requested = setting != nullptr && *setting != 0 ? static_cast<uint32_t>(std::strtoul(setting, nullptr, 10))
	                                                     : game_height;
	// KYTY_UPSCALE_RATIO=<1..4> (the launcher's DLSS/FSR/XeSS presets: DLAA 1.0, Quality 1.5, Balanced 1.72,
	// Performance 2.0, Ultra Performance 3.0) with no KYTY_RENDER_HEIGHT: the render height is the monitor's own height
	// (the primary display's current mode; KYTY_DISPLAY_HEIGHT=<n> overrides it) over the ratio, so the 3D renders at
	// what the preset means for that monitor. The game's upscaler output stays 3840x2160 (its frame after it, the HUD
	// and the composite are made for it), the presenter scales that to the window; within 720-2160.
	if (by_ratio) {
		const double ratio   = std::strtod(ratio_s, nullptr);
		const char*  forced  = std::getenv("KYTY_DISPLAY_HEIGHT");
		uint32_t     display = forced != nullptr && *forced != 0 ? static_cast<uint32_t>(std::strtoul(forced, nullptr, 10))
		                                                         : LocalPlatform::DisplayHeight(nullptr);
		if (display == 0) display = OutputHeight;
		if (ratio >= 1.0 && ratio <= 4.0) {
			const auto height = static_cast<uint32_t>(std::lround(display / ratio));
			requested         = std::clamp<uint32_t>(height, 720, OutputHeight);
			std::printf("Demon's Souls render size: KYTY_UPSCALE_RATIO=%s on a %u high display: %u%s\n", ratio_s, display,
			            requested, requested != height ? " (720-2160)" : "");
		} else {
			std::printf("Demon's Souls render size: KYTY_UPSCALE_RATIO=%s is not 1-4; ignored\n", ratio_s);
		}
	}
	const auto aspect    = wide != nullptr && *wide != 0 ? ParseAspect(wide) : 0.0;
	if (wide != nullptr && *wide != 0 && aspect == 0.0)
		std::printf("Demon's Souls render size: KYTY_RENDER_ASPECT=%s is not wider than 16:9 and at most 32:9; 16:9 kept\n",
		            wide);
	if (trinity && !DemonsSoulsSuperResolution::AnyHeight() && requested >= 720 && requested < SuperResolutionLeastHeight) {
		std::printf("Demon's Souls render size: KYTY_RENDER_HEIGHT=%u is below the game's super resolution (1080 at "
		            "least); 1080 instead\n",
		            requested);
		requested = SuperResolutionLeastHeight;
	}
	// The Cinematic mode (display mode 4: 3840x2160 natively, no upscaler) with KYTY_UPSCALE_RATIO: its config asks for
	// display mode 5 as the performance mode's does (loader/gameArgs.cpp), so it renders at this same size (within
	// the same 1080 least / 720 and render target memory limits) and goes through the upscaler.
	const auto cinematic_note = [&](uint32_t width, uint32_t height) {
		if (trinity && by_ratio)
			std::printf("Demon's Souls render size: cinematic mode renders %ux%u instead of %ux%u (output %ux%u, through the "
			            "upscaler)\n",
			            width, height, OutputWidth, OutputHeight, g_wide_output != 0 ? g_wide_output : OutputWidth,
			            OutputHeight);
	};
	if (requested == game_height && aspect == 0.0) {
		cinematic_note(game_width, game_height);
		return;
	}
	auto size = RenderSizeFor(requested, aspect != 0.0 ? aspect : GameAspect);
	if (const auto most = MaxRenderHeight(aspect != 0.0 ? aspect : GameAspect); size.height > most) {
		const auto lower = RenderSizeFor(most, aspect != 0.0 ? aspect : GameAspect);
		std::printf("Demon's Souls render size: %ux%u does not fit the game's render target memory (made for 3840x2160: "
		            "a red or broken picture, or a crash); %ux%u instead\n",
		            size.width, size.height, lower.width, lower.height);
		size = lower;
	}
	if (size.height == 0) {
		std::printf("Demon's Souls render size: KYTY_RENDER_HEIGHT=%s is not 720-2160; the game's %ux%u kept\n", setting,
		            game_width, game_height);
		return;
	}
	auto constant = FindRenderConstant(*program, trinity ? TrinityCase : ModeCase, game_width, game_height);
	if (!constant && !trinity) constant = FindRenderConstant(*program, ModeCaseDuplicated, game_width, game_height);
	const auto widths   = FindUniqueData(*program, std::span(reinterpret_cast<const uint8_t*>(ModeWidths), sizeof(ModeWidths)));
	const auto heights  = FindUniqueData(*program, std::span(reinterpret_cast<const uint8_t*>(ModeHeights), sizeof(ModeHeights)));
	// (The render setup's 16:9: the one such store in the code.)
	std::optional<uint64_t> aspect_store;
	if (aspect != 0.0) {
		const auto sites = GuestCode::Find(*program, GuestCode::Pattern(AspectStore), 2);
		if (sites.size() == 1) aspect_store = sites[0];
	}
	if (!constant || !widths || !heights || (aspect != 0.0 && !aspect_store)) {
		std::printf("Demon's Souls render size: the video settings of a known build not found; the game's %ux%u kept\n",
		            game_width, game_height);
		return;
	}
	// Installation occurs before module initializers or guest threads run; program memory is writable to the host.
	const uint32_t pair[2] {size.width, size.height};
	std::memcpy(reinterpret_cast<void*>(*constant), pair, sizeof(pair));
	std::memcpy(reinterpret_cast<void*>(*widths + mode * sizeof(uint32_t)), &size.width, sizeof(uint32_t));
	std::memcpy(reinterpret_cast<void*>(*heights + mode * sizeof(uint32_t)), &size.height, sizeof(uint32_t));
	if (aspect_store) {
		// The projection's aspect: the render size's own, as the game's 2560x1440 has 16/9.
		const float value = static_cast<float>(static_cast<double>(size.width) / size.height);
		std::memcpy(reinterpret_cast<void*>(*aspect_store + 3), &value, sizeof(value));
		g_wide_aspect = value;
	}
	// The super resolution path with a wider aspect: the upscaler makes the wide output (WideOutputWidth).
	const char* wide_output = std::getenv("KYTY_WIDE_OUTPUT");
	if (trinity && aspect != 0.0 && (wide_output == nullptr || std::strcmp(wide_output, "0") != 0)) {
		const auto output = RenderSizeFor(OutputHeight, aspect);
		if (InstallWideOutput(*program, mode, output.width)) {
			g_wide_output = output.width;
		} else {
			std::printf("Demon's Souls render size: the wide output's settings of a known build not found; the 3D squeezed "
			            "into the 3840x2160 output\n");
		}
	}
	std::printf("Demon's Souls render size: performance mode renders %ux%u instead of %ux%u (output %ux%u)%s\n", size.width,
	            size.height, game_width, game_height, g_wide_output != 0 ? g_wide_output : OutputWidth, OutputHeight,
	            g_wide_output != 0 ? ", its projection and the output as wide" : aspect_store ? ", its projection as wide" : "");
	cinematic_note(size.width, size.height);
}

} // namespace Loader::DemonsSoulsRenderSize
