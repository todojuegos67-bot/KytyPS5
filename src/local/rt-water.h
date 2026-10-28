#ifndef EMULATOR_SRC_LOCAL_RT_WATER_H_
#define EMULATOR_SRC_LOCAL_RT_WATER_H_
// KYTY_HWRT_WATER=0|1|2|3 (No / Bajo / Medio / Alto; default 0: off, nothing captured, created or recorded; needs
// KYTY_HWRT=1): sharp ray-traced mirror reflections on water, puddles and wet stone (rt-water.cpp, rt-water.comp).
// The water surfaces are identified by their pixel shader (KYTY_HWRT_WATER_PS=<hex>[,<hex>...], default Demon's Souls'
// 7a59c1a20cbe5d4a) and captured into a TLAS of their own by KYTY_HWRT (hwrt-scene.cpp); puddles by the game's own SSR
// (KYTY_RT_REFLECTIONS' guide) on level floors.
#include "local/frame-gen.h"

#include <cstdint>

namespace Libs::Graphics::FrameGen {

[[nodiscard]] int RtWaterLevel();
// A water-surface pixel shader (false whenever the level is 0). Any thread.
[[nodiscard]] bool RtWaterIsWaterPs(uint64_t ps_hash);
// (table-xpr.inc, the table store) a water shader's draw state, logged once per (VS, PS).
void RtWaterNoteState(uint64_t vs_hash, uint64_t ps_hash, uint32_t colors, uint32_t depth_width, bool depth_write, int topology);
// (renderDraw.cpp) a water shader's draw on the normal path (not captured; counted for the log).
void RtWaterNoteNormalDraw();
// In Upscale, after RtReflectionsApply: inputs.color becomes the composited copy. False: nothing done.
bool RtWaterApply(vk::CommandBuffer command, UpscaleInputs& inputs);

} // namespace Libs::Graphics::FrameGen

#endif // EMULATOR_SRC_LOCAL_RT_WATER_H_
