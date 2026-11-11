#pragma once

#include <cstdint>
#include <string_view>

namespace Loader {
struct Program;
// KYTY_HD_OUTPUT=1 (the launcher's "Game output: 1080p"): Demon's Souls 01.007.000 runs its own HD path, the one a PS5
// takes on a 1080p TV: its output, post-processing and interface at 1920x1080 instead of 3840x2160 (the HUD in the
// game's own HD layout, not the 4K one scaled), the presenter (or Gamescope) scaling that to the window. GPU-bound PCs
// save the 4K post-processing the performance mode does after its 2560x1440 3D. Its display mode is 1 (the game's
// 1600x900 3D upscaled to the 1080p output), its render size made the render height's (KYTY_RENDER_HEIGHT,
// KYTY_RENDER_ASPECT: loader/demonsSoulsRenderSize.h; 1920x1080 by default). Display mode 2 (its own 3D at 1920x1080)
// is Install's first choice only: it leaves the menus' 3D black (the equipment menu's character, the character
// creator; issue #138).
// The game picks its display mode in three places - its video out, its renderer and its interface (which lays out at
// 4K for modes 3-5) - each the same way: r_Wants4K, else r_Wants4K_Game (kept in the save), else a compiled 3. Each
// gets the HD mode, its two overrides removed (a save's 4K mode would come back on loading it). The mode alone keeps
// the 4K display buffers and the video settings' 3840x2160 ceilings (the 1080p picture in a corner of them): both made
// 1920x1080. Excludes the wide output (KYTY_WIDE_OUTPUT). With the PS5 Pro path (KYTY_UPSCALER) only the interface's
// part is taken: the 3D upscaled to a 1080p output there (loader/demonsSoulsSuperResolution.h).
// (SharpEmu's TitleDisplayPatches found the same places in 01.004.000; these are 01.007.000's, found by their code, and
// 01.005.000's, where its compiler made them differ: issue #226.)
namespace DemonsSoulsHdOutput {
// KYTY_HD_OUTPUT is set (before Install: the other patches keep out of its way), or KYTY_OUTPUT_HEIGHT with an upscaler.
[[nodiscard]] bool Requested();
// KYTY_OUTPUT_HEIGHT=<1080..2159>, with the PS5 Pro path (KYTY_UPSCALER): its output at that height instead of the HD
// output's 1080, 16:9 (2560x1440 for 1440: a 1440p screen gets DLSS's own picture, neither the 4K one scaled down nor
// the 1080p one scaled up) - the 1080p output's places, the display buffers that size, the interface in its HD layout
// scaled to the output (InterfaceScaled); 0 when unset or out of range.
[[nodiscard]] uint32_t CustomOutputHeight();
// The PS5 Pro path's output with the HD output: 1920x1080, or KYTY_OUTPUT_HEIGHT's size.
[[nodiscard]] uint32_t UpscaledWidth();
[[nodiscard]] uint32_t UpscaledHeight();
// KYTY_OUTPUT_HEIGHT's output above 1080: the interface in its 4K layout at the output's size (InterfaceAtOutput), or
// with KYTY_OUTPUT_LAYOUT=hd its HD layout's 1920x1080 in the output, which src/local/hud-anchor.h scales to it
// (ScaleScope; InterfaceScaled). Scaled, the HD layout lost the thin strokes of its letters at 1.33x (a T read as an I,
// issue #187), and the menus' 3D views (the character creator's, the equipment menu's) came 1.33x too large from the
// top left corner (issue #202): the game draws them over the output in its pixels already.
[[nodiscard]] bool InterfaceScaled();
// The game's own layout code puts the 4K layout at 1.33 pixels per layout unit for 1440 (ScreenSize, found by
// Psychowarden for issue #199), the same picture as the 4K output's scaled down, nothing scaled after. (The interface's
// screen size left at 3840x2160 put what lay past the output's size in its layout outside it: the souls, the items,
// the prompts.)
[[nodiscard]] bool InterfaceAtOutput();
// With InterfaceAtOutput and KYTY_RENDER_ASPECT wider than 16:9 (KYTY_WIDE_OUTPUT=0: squeezed into 16:9 instead): the
// width of the wide output at KYTY_OUTPUT_HEIGHT (3416x1440 at 21:9; DemonsSoulsRenderSize makes it, issue #199), the
// display buffers as wide, the interface in its 16:9 part (2560x1440); 0 otherwise.
[[nodiscard]] uint32_t WideOutputWidth();
void Install(Program* program);
// Install found its places and took the game to the HD path.
[[nodiscard]] bool Active();
// The display mode the HD path runs (2 at first): DemonsSoulsRenderSize's other render size asks for mode 1.
void SetMode(int mode);

// The three display mode choices: mov reg, 3; then the two cmovne taking r_Wants4K's and r_Wants4K_Game's values.
// The video out's (01.007.000 0x9fe6a3): mov esi, 3; test ecx, ecx; cmovne esi, ecx; test eax, eax; cmovne esi, eax.
inline constexpr std::string_view VideoOutChoice = "be 03 00 00 00 85 c9 0f 45 f1 85 c0 0f 45 f0 e8";
inline constexpr uint32_t         VideoOutOverrides[2] {7, 12};
// The renderer's (0xb26055), r_Wants4K and r_Wants4K_Game read in between, then compared with the current mode.
inline constexpr std::string_view RendererChoice =
    "be 03 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 8b 8f 38 0d 00 00 8b 87 30 0d 00 00 85 c9 0f 45 f1 85 c0 0f 45 f0 39 b7";
inline constexpr uint32_t RendererOverrides[2] {27, 32};
// The interface's (0xde88b6): mov eax, 3; ...; cmovne eax, ecx; ...; cmovne eax, r15d; add eax, -3; cmp eax, 3 (its
// 4K layout for modes 3-5).
inline constexpr std::string_view InterfaceChoice =
    "b8 03 00 00 00 85 c9 0f 45 c1 45 85 ff 48 8b 0a 41 0f 45 c7 83 c0 fd 83 f8 03";
inline constexpr uint32_t InterfaceOverrides[2] {7, 16};
// The display buffers' size (0x90af74, 0x90b061): mov edi, 3840; mov esi, 2160; mov edx, 1; mov ecx, 1.
inline constexpr std::string_view DisplayBuffer = "bf 00 0f 00 00 be 70 08 00 00 ba 01 00 00 00 b9 01 00 00 00";
inline constexpr uint32_t         HdWidth = 1920, HdHeight = 1080;
// An interface image's size from its texture's (0xe6041b, in 0xe603c0): mov rax, [r13 + 0x10]; vmovaps [rbp - 0x40],
// xmm4; mov rdi, [rax + disp32]; add rdi, 8; call 0xdda210 (the screen's pixels per layout unit, the divisor).
inline constexpr std::string_view ImageScale = "49 8b 45 10 c5 f8 29 65 c0 48 8b b8 ?? ?? ?? ?? 48 83 c7 08 e8";
// 01.005.000's (0xe3f7dc): mov rax, [r12 + 0x10], a byte longer, then the same.
inline constexpr std::string_view ImageScaleR12 = "49 8b 44 24 10 c5 f8 29 65 c0 48 8b b8 ?? ?? ?? ?? 48 83 c7 08 e8";
// The same division for another kind of image, the souls counter's backdrop among them (0xe641e8): cmp byte
// [rax + 0x198], 1; jne; lea rdi, [rbx + 8]; call 0xdda210; mov rax, [r14 + 0x110].
inline constexpr std::string_view BackdropScale =
    "80 b8 98 01 00 00 01 0f 85 ?? ?? ?? ?? 48 8d 7b 08 e8 ?? ?? ?? ?? 49 8b 86 10 01 00 00";
inline constexpr uint32_t BackdropDivisor = 13; // the lea and the call, 9 bytes
// 01.005.000's (0xe43551): cmp byte [rax + 0x198], 0; je short past them; the same lea and call.
inline constexpr std::string_view BackdropScaleShort =
    "80 b8 98 01 00 00 00 74 ?? 48 8d 7b 08 e8 ?? ?? ?? ?? 49 8b 86 10 01 00 00";
inline constexpr uint32_t BackdropShortDivisor = 9;
// And for the pieces of the health and stamina bars (0x155924a, in 0x1559140): each texture's width over the divisor.
// add rbx, 8; mov rdi, rbx; mov [rbp + disp32], rbx; call 0xdda210; mov rbx, [r14 + 0x18]; vmovss [rbp + disp32], xmm0.
inline constexpr std::string_view GaugeScale = "48 83 c3 08 48 89 df 48 89 9d ?? ?? ?? ?? e8 ?? ?? ?? ?? 49 8b 5e 18 c5 fa 11 85";
inline constexpr uint32_t         GaugeDivisor = 4; // mov rdi, rbx; the store; the call: 15 bytes
inline constexpr uint32_t         GaugeStore   = 7; // the store of rbx, 7 bytes (kept)
// The interface's screen size (0xde8910, in its per-frame setup; found by Psychowarden, issue #199): the size it is given,
// scaled by 2 or 0.5 (0x24f9f10) when its layout and the video out's differ, then stored by 0xddb4f0 (the screen at +8:
// its pixels over its 1080 layout units, 0xdda210, are the pixels per layout unit). The 43 bytes from the je over the
// scaling to the store's lea rdi, [rbx + 8].
inline constexpr std::string_view ScreenSize =
    "74 29 31 c9 83 f8 03 48 8d 05 ?? ?? ?? ?? c5 fb 10 85 78 fc ff ff 0f 92 c1 c4 e2 79 18 0c 88 c5 f8 59 c1 c5 f8 13 "
    "85 78 fc ff ff 48 8d 7b 08";
inline constexpr uint32_t ScreenSizeBytes = 43;
} // namespace DemonsSoulsHdOutput
} // namespace Loader
