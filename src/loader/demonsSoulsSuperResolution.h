#pragma once

namespace Loader {
struct Program;
// KYTY_UPSCALER=dlss (run-windows.ps1 -Upscaler, the launcher's "Upscaler"): Demon's Souls 01.007.000's own PS5 Pro
// path. On a PS5 Pro (sceKernelIsTrinityMode) the game's performance mode is its display mode 5: the 3D rendered at
// 2688x1512, then Sony's super resolution (PSSR, libScePsml) in place of its TAA makes the 3840x2160 image the rest of
// the frame (bloom, depth of field, motion blur, the HUD) is drawn on; elsewhere mode 3: 2560x1440, the TAA at that
// size, the 4K output a spatial upscale in the final composite. Here the renderer makes its SuperResolution object
// (the one sceKernelIsTrinityMode check before it, nothing else of the PS5 Pro: its shaders, memory and settings stay
// the PS5's) and the performance mode's render config asks for mode 5 (kernel/fileSystem.cpp: loader/gameArgs.h);
// libs/libPsml.cpp hands the dispatch to the emulator's upscaler (graphics/guest_gpu/superResolution.h). Only where
// DLSS is ready on the device (frame-gen.h UpscalerReady, made before the game is loaded); elsewhere the game's TAA.
// KYTY_RENDER_HEIGHT then sizes mode 5 (loader/demonsSoulsRenderSize.h), at least 1080: the game's super resolution
// is off below (SuperResolution::IsEnabled).
namespace DemonsSoulsSuperResolution {
void Install(Program* program);
// Install turned the game's super resolution path on.
[[nodiscard]] bool Active();
} // namespace DemonsSoulsSuperResolution
} // namespace Loader
