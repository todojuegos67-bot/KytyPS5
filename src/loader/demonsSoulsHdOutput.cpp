#include "loader/demonsSoulsHdOutput.h"
#include "loader/demonsSoulsRenderSize.h"
#include "loader/demonsSoulsSuperResolution.h"

#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>

namespace Loader::DemonsSoulsHdOutput {
namespace {

bool                    g_active    = false;
bool                    g_upscaled  = false; // with the PS5 Pro path (DemonsSoulsSuperResolution::HdOutput)
bool                    g_layout_4k = false; // InterfaceAtOutput
uint32_t                g_wide      = 0;     // WideOutputWidth
std::array<uint64_t, 3> g_modes {};          // the three choices' immediates: video out, renderer, interface

// A nop of `length` bytes (1-5) in place of a cmovne or a load.
void Nop(uint64_t address, uint32_t length) {
	constexpr uint8_t nops[5][5] {{0x90}, {0x66, 0x90}, {0x0f, 0x1f, 0x00}, {0x0f, 0x1f, 0x40, 0x00}, {0x0f, 0x1f, 0x44, 0x00, 0x00}};
	EXIT_IF(length == 0 || length > 5);
	std::memcpy(reinterpret_cast<void*>(address), nops[length - 1], length);
}

} // namespace

uint32_t CustomOutputHeight() {
	static const uint32_t height = [] {
		const char* value = std::getenv("KYTY_OUTPUT_HEIGHT");
		if (value == nullptr || *value == 0) return 0u;
		const auto parsed = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
		if (parsed == DemonsSoulsRenderSize::OutputHeight) return 0u;
		if (parsed < HdHeight || parsed > DemonsSoulsRenderSize::OutputHeight) {
			std::printf("Demon's Souls output: KYTY_OUTPUT_HEIGHT=%s is not 1080-2160; not used\n", value);
			return 0u;
		}
		return parsed;
	}();
	return height;
}

uint32_t UpscaledHeight() {
	return CustomOutputHeight() != 0 ? CustomOutputHeight() : HdHeight;
}

uint32_t UpscaledWidth() {
	return CustomOutputHeight() != 0 ? DemonsSoulsRenderSize::RenderSizeFor(CustomOutputHeight()).width : HdWidth;
}

bool InterfaceScaled() {
	return g_active && g_upscaled && CustomOutputHeight() > HdHeight && !g_layout_4k;
}

bool InterfaceAtOutput() {
	return g_active && g_upscaled && g_layout_4k;
}

uint32_t WideOutputWidth() {
	return InterfaceAtOutput() ? g_wide : 0;
}

bool Requested() {
	// KYTY_OUTPUT_HEIGHT: the PS5 Pro path's output at that height (with an upscaler only: the game's own HD path has no
	// other size).
	if (const char* upscaler = std::getenv("KYTY_UPSCALER");
	    CustomOutputHeight() != 0 && upscaler != nullptr && *upscaler != 0 && std::strcmp(upscaler, "off") != 0)
		return true;
	const char* value = std::getenv("KYTY_HD_OUTPUT");
	if (value != nullptr && *value != 0 && std::strcmp(value, "auto") != 0) return std::strcmp(value, "0") != 0;
	// Unset or "auto" (the default): the HD output when the automatic render height (the window's) is at most 1080 lines,
	// so that the game's output is the window's 1920x1080 (loader/demonsSoulsRenderSize.h); never with the PS5 Pro path
	// (KYTY_UPSCALER: its automatic height is the 4K output's) or a wider aspect (KYTY_RENDER_ASPECT). A render height set
	// (the launchers': they choose the output themselves) keeps the 4K output. Once not the default: an RX 9070 XT
	// (RADV) showed the HD path's picture far darker and no faster; on current Mesa the same card measured the same
	// brightness and 4.6% more fps (issue #80), and the menus' 3D it left black at 1080 is drawn now (#138).
	const char* upscaler = std::getenv("KYTY_UPSCALER");
	const char* aspect   = std::getenv("KYTY_RENDER_ASPECT");
	if ((upscaler != nullptr && *upscaler != 0 && std::strcmp(upscaler, "off") != 0) || (aspect != nullptr && *aspect != 0))
		return false;
	const auto height = DemonsSoulsRenderSize::AutoHeight(false);
	return height != 0 && height <= HdHeight;
}

bool Active() {
	return g_active;
}

void SetMode(int mode) {
	if (!g_active) return;
	// (With the PS5 Pro path the video out and the renderer take its display mode 5 in every render config: the
	// character creator's and the Cinematic mode's ask for 4, native 4K, which the 1080p interface would not fit.)
	const auto     value = static_cast<uint32_t>(mode);
	const uint32_t values[3] {g_upscaled ? 5u : value, g_upscaled ? 5u : value, g_layout_4k ? 5u : value};
	for (size_t i = 0; i < g_modes.size(); ++i)
		std::memcpy(reinterpret_cast<void*>(g_modes[i] + 1), &values[i], sizeof(values[i]));
}

void Install(Program* program) {
	if (!Requested() || program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    program->file_name.filename() != "eboot.bin")
		return;
	// With the PS5 Pro path (DemonsSoulsSuperResolution::HdOutput): the interface's part - its layout's choice, the
	// display buffers and its images' scale -, the video out and the renderer at its display mode 5 (SetMode), whose
	// sizes DemonsSoulsRenderSize makes 1080p (the ceilings below are each mode's own output there).
	const bool upscaled  = DemonsSoulsSuperResolution::HdOutput();
	const auto video     = GuestCode::FindUnique(*program, GuestCode::Pattern(VideoOutChoice));
	const auto renderer  = GuestCode::FindUnique(*program, GuestCode::Pattern(RendererChoice));
	const auto interface = GuestCode::FindUnique(*program, GuestCode::Pattern(InterfaceChoice));
	const auto buffers   = GuestCode::Find(*program, GuestCode::Pattern(DisplayBuffer), 3);
	auto       setter    = GuestCode::Find(*program, GuestCode::Pattern(DemonsSoulsRenderSize::SetterFixedPair), 2);
	if (setter.empty())
		setter = GuestCode::Find(*program, GuestCode::Pattern(DemonsSoulsRenderSize::SetterFixedPairAfterFlag), 2);
	const auto builder   = GuestCode::Find(*program, GuestCode::Pattern(DemonsSoulsRenderSize::ConstructorFixedPair), 2);
	const auto query     = GuestCode::Find(*program, GuestCode::Pattern(DemonsSoulsRenderSize::QueryFixedPair), 2);
	if (!video || !renderer || !interface || buffers.size() != 2 || setter.size() != 1 || builder.size() != 1 ||
	    query.size() != 1) {
		std::printf("Demon's Souls HD output: the display mode settings of a known build (01.007.000, 01.005.000) not found; the 4K "
		            "output kept\n");
		return;
	}
	// Installation occurs before module initializers or guest threads run; program memory is writable to the host.
	g_modes = {*video, *renderer, *interface};
	for (const auto at: VideoOutOverrides) Nop(*video + at, 3);
	for (const auto at: RendererOverrides) Nop(*renderer + at, 3);
	Nop(*interface + InterfaceOverrides[0], 3);
	Nop(*interface + InterfaceOverrides[1], 4);
	const uint32_t width  = upscaled ? UpscaledWidth() : HdWidth;
	const uint32_t height = upscaled ? UpscaledHeight() : HdHeight;
	// KYTY_OUTPUT_HEIGHT above 1080 (InterfaceAtOutput; KYTY_OUTPUT_LAYOUT=hd: not): the interface's 4K layout at the
	// output's size, below. With a wider KYTY_RENDER_ASPECT the wide output at that height (WideOutputWidth), the display
	// buffers as wide: the game copies its output into them for the frames without the 3D (issue #198).
	const char* layout = std::getenv("KYTY_OUTPUT_LAYOUT");
	const auto  screen = upscaled && CustomOutputHeight() > HdHeight && (layout == nullptr || std::strcmp(layout, "hd") != 0)
	                         ? GuestCode::FindUnique(*program, GuestCode::Pattern(ScreenSize))
	                         : std::nullopt;
	if (screen) {
		const char* aspect = std::getenv("KYTY_RENDER_ASPECT");
		const char* wide   = std::getenv("KYTY_WIDE_OUTPUT");
		const auto  value  = aspect != nullptr && *aspect != 0 ? DemonsSoulsRenderSize::ParseAspect(aspect) : 0.0;
		if (value != 0.0 && (wide == nullptr || std::strcmp(wide, "0") != 0))
			g_wide = DemonsSoulsRenderSize::RenderSizeFor(height, value).width;
	} else if (upscaled && CustomOutputHeight() > HdHeight && (layout == nullptr || std::strcmp(layout, "hd") != 0)) {
		std::printf("Demon's Souls HD output: the interface's screen size not found; its HD layout scaled instead\n");
	}
	const uint32_t buffer_width = g_wide != 0 ? g_wide : width;
	for (const auto site: buffers) {
		std::memcpy(reinterpret_cast<void*>(site + 1), &buffer_width, sizeof(buffer_width));
		std::memcpy(reinterpret_cast<void*>(site + 6), &height, sizeof(height));
	}
	if (!upscaled) {
		// The ceilings' mov r64, 2160 << 32 | 3840 (the immediate after the opcode's two bytes).
		const uint64_t ceiling = static_cast<uint64_t>(HdHeight) << 32u | HdWidth;
		std::memcpy(reinterpret_cast<void*>(setter[0] + 6), &ceiling, sizeof(ceiling));
		std::memcpy(reinterpret_cast<void*>(builder[0] + 9), &ceiling, sizeof(ceiling));
		std::memcpy(reinterpret_cast<void*>(query[0] + 5), &ceiling, sizeof(ceiling));
	}
	// The interface's textures are 4K's in every mode, and an image sized from its texture divides it by the screen's
	// pixels per layout unit (2 at 4K, 1 at 1080p): at 1080p the HUD's images and frames, the menus' and the health
	// bars' came twice as large (the user 10-20). The divisor kept at 2.0: mov eax, 0x40000000; vmovd xmm0, eax.
	constexpr uint8_t divisor[16] {0xb8, 0x00, 0x00, 0x00, 0x40, 0xc5, 0xf9, 0x6e,
	                               0xc0, 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00};
	// (01.005.000's load before the vmovaps is a byte longer.)
	auto     image      = GuestCode::FindUnique(*program, GuestCode::Pattern(ImageScale));
	uint32_t image_load = 4;
	if (!image) {
		image      = GuestCode::FindUnique(*program, GuestCode::Pattern(ImageScaleR12));
		image_load = 5;
	}
	if (image) {
		Nop(*image, image_load);
		std::memcpy(reinterpret_cast<void*>(*image + image_load + 5), divisor, sizeof(divisor));
		std::printf("Demon's Souls HD output: the interface's images sized at the 4K scale\n");
	} else {
		std::printf("Demon's Souls HD output: the interface's image sizing not found; its images come twice as large\n");
	}
	// Another kind of image divides the same way: at 1080p the souls counter's backdrop, a gradient strip from the
	// icon to the number at 4K, came as a small black square. The same 9 bytes, without the nop7.
	auto     backdrop    = GuestCode::FindUnique(*program, GuestCode::Pattern(BackdropScale));
	uint32_t backdrop_at = BackdropDivisor;
	if (!backdrop) {
		backdrop    = GuestCode::FindUnique(*program, GuestCode::Pattern(BackdropScaleShort));
		backdrop_at = BackdropShortDivisor;
	}
	if (backdrop) {
		std::memcpy(reinterpret_cast<void*>(*backdrop + backdrop_at), divisor, 9);
		std::printf("Demon's Souls HD output: the souls counter's backdrop sized at the 4K scale\n");
	} else {
		std::printf("Demon's Souls HD output: the souls counter's backdrop sizing not found; it comes as a small square\n");
	}
	// A third: the health and stamina bars' pieces, each its texture's width over the divisor. At 1080p their ends came
	// twice as wide, a dark wedge over the bars' left ends where the 4K output fills them to the emblem (issue #187).
	// The store of rbx kept, then 2.0 for the call: xor eax, eax; mov al, 2; vcvtsi2ss xmm0, xmm0, eax (15 bytes too).
	if (const auto gauge = GuestCode::FindUnique(*program, GuestCode::Pattern(GaugeScale))) {
		uint8_t           code[15] {};
		constexpr uint8_t two[8] {0x31, 0xc0, 0xb0, 0x02, 0xc5, 0xfa, 0x2a, 0xc0};
		std::memcpy(code, reinterpret_cast<const void*>(*gauge + GaugeStore), 7);
		std::memcpy(code + 7, two, sizeof(two));
		std::memcpy(reinterpret_cast<void*>(*gauge + GaugeDivisor), code, sizeof(code));
		std::printf("Demon's Souls HD output: the health and stamina bars' pieces sized at the 4K scale\n");
	} else {
		std::printf("Demon's Souls HD output: the bars' piece sizing not found; their ends come twice as wide\n");
	}
	// The interface's 4K layout (its choice made 5 by SetMode) at the output's size (its 16:9 part with the wide output):
	// mov rax, imm64 (the width and height as floats); mov [rbp - 0x388], rax; jmp to the store's lea. The images keep
	// the 4K's divisor above (the 4K output's images scaled down with the rest).
	if (screen) {
		const float size[2] {static_cast<float>(width), static_cast<float>(height)};
		uint8_t     code[19] {0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0x48, 0x89, 0x85, 0x78, 0xfc, 0xff, 0xff, 0xeb,
		                      static_cast<uint8_t>(ScreenSizeBytes - 19)};
		std::memcpy(code + 2, size, sizeof(size));
		std::memcpy(reinterpret_cast<void*>(*screen), code, sizeof(code));
		g_layout_4k = true;
	}
	g_active   = true;
	g_upscaled = upscaled;
	SetMode(2);
	if (upscaled) {
		std::printf("Demon's Souls HD output: the interface at %s over the PS5 Pro path's %ux%u output%s\n",
		            g_layout_4k ? g_wide != 0 ? "the output's 16:9 part (its 4K layout)" : "the output's size (its 4K layout)"
		                        : "1920x1080 (its HD layout)",
		            g_wide != 0 ? g_wide : width, height,
		            CustomOutputHeight() != 0 && !g_layout_4k ? " (KYTY_OUTPUT_HEIGHT; the interface scaled to it)" : "");
	} else {
		std::printf("Demon's Souls HD output: the game's HD path (display mode 2: output, post-processing and interface at "
		            "1920x1080) is on\n");
	}
}

} // namespace Loader::DemonsSoulsHdOutput
