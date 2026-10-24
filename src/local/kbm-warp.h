#ifndef EMULATOR_SRC_LOCAL_KBM_WARP_H_
#define EMULATOR_SRC_LOCAL_KBM_WARP_H_
// KYTY_KBM_WARP=1 (with KYTY_KBM=1): the pending mouse turn reprojected late, in the present thread
// (swapchain.cpp, Swapchain::RecordPresentCommands). With the frame's HUD-less image (the game's composite drawn without
// its HUD texture, FrameGen::FrameInputs::hudless) only the 3D moves: kbm-warp.comp keeps the final image where it
// differs from the HUD-less one (the HUD) and samples the HUD-less image shifted elsewhere.
// FSR 3 frame generation: the same warped image is the real frame; the frame's HUD-less image, depth and motion vectors
// (its own copies, FrameInputs) are shifted alike and the vectors get the shift's change since the previous real frame,
// so the generated frames interpolate between two consistently warped frames. Nothing of FSR 3 itself is touched.
//
// KYTY_KBM_WARP_HUDLESS=0: no HUD-less image (the whole frame moves, the HUD too; and, without frame generation, the
// HUD-less capture is not turned on for the warp alone).
// KYTY_KBM_WARP_FSR3=1: warp with FSR 3 frame generation too (10-25: off by default, double images; DLSS-G: never).
// KYTY_KBM_WARP_ZOOM (0.05): the constant edge crop (fraction of each half-size), the most the 3D can be shifted.
// The shift itself (swapchain.cpp, HostInputKbmWarpDegrees): what the pad reads after the presented frame's flip sent
// plus the counts not read yet, in the camera's degrees as learned from the camera: it hands over to the game's own
// turn the frame that has it is presented, so it does not jump back and forth.
// KYTY_KBM_WARP_THRESHOLD / KYTY_KBM_WARP_RADIUS: the HUD mask (as KYTY_FSR3_HUD_THRESHOLD / _RADIUS).
#include "graphics/host_gpu/graphicContext.h"

namespace Libs::Graphics {
namespace FrameGen {
struct FrameInputs;
}
namespace KbmWarp {

// KYTY_KBM=1 and KYTY_KBM_WARP=1.
[[nodiscard]] bool On();
// ... and the HUD-less image wanted (FrameGen captures it even without frame generation then).
[[nodiscard]] bool HudlessWanted();
[[nodiscard]] bool Fsr3Wanted();

// Present thread, in the present's command buffer before the present blit. `final_image` (TRANSFER_SRC, sampled) is the
// frame; `inputs` its frame generation inputs (HUD-less image needed). `shift_*`: output pixels, out(p) = in(p + shift).
// With `fsr3`, the inputs' HUD-less image, depth and motion vectors are shifted too (all or nothing).
// Returns the image to blit instead of `final_image` (TRANSFER_SRC, `final_image`'s size), or null: nothing recorded.
[[nodiscard]] VulkanImage* Record(GraphicContext& graphics, vk::CommandBuffer command, VulkanImage& final_image,
                                  FrameGen::FrameInputs* inputs, float shift_x, float shift_y, bool fsr3);
// The warp should run this frame though nothing moves: the edge zoom is constant (KYTY_KBM_WARP_ZOOM), or (FSR 3) the
// last real frame's input shift is not zero and the next frame's vectors need the change back.
[[nodiscard]] bool InputsShifted();
// FSR 3: a real frame whose inputs were not shifted (none, or generation off): the next starts from no shift.
void NoteUnshifted();

} // namespace KbmWarp
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_LOCAL_KBM_WARP_H_
