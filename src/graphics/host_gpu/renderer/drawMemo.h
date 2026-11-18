#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWMEMO_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWMEMO_H_

#include <cstdint>
#include <string>

// gt7-drawmemo-1118, KYTY_DRAW_MEMO (off by default): a memo of a draw stage's resolved images
// and samplers (RenderExecutor::PrepareBindings), by stable key: the stage's program and the
// words of its T#s and S#s. GT7 builds its per-draw descriptor tables in fresh linear memory each
// frame (FUN_0000acf0, a lock-free bump allocator, called from its binders such as FUN_000e62f0),
// so the tables' addresses never repeat, but it fills them from persistent objects (FUN_00005f60
// and FUN_00005ca0 build each descriptor from a resource's address, stride and format; FUN_000ede70
// binds a persistent shader object): the words, and the shader pointers, stay the same from one
// frame to the next. A stage whose program and words were seen before, with every image the
// texture memo would give it still current (its entry, view, image set stamp and image), takes
// the images and samplers of last time without one texture lookup.
//
//   KYTY_DRAW_MEMO=1  the memo is used.
//   KYTY_DRAW_MEMO=2  verifier: the normal path runs and is used; the memo's answer is compared
//                     with it and differences are counted and logged (the first 16).
//
// A "Desglose draw" line every 10 s (GpuPhases) gives where a draw's programs and textures time
// goes, with the memo's counters. GPU thread only.
namespace Libs::Graphics::DrawMemo {

// 0 (off), 1 (on) or 2 (verifier), from KYTY_DRAW_MEMO.
[[nodiscard]] int Mode();

// Set around a draw's PrepareBindings calls on the GPU thread (TranslateDrawBindings): only draw
// stages use the memo and are timed into the draw steps.
class DrawScope {
public:
	DrawScope();
	~DrawScope();
	DrawScope(const DrawScope&)            = delete;
	DrawScope& operator=(const DrawScope&) = delete;

private:
	bool m_previous;
};
[[nodiscard]] bool InDraw();

// The memo's counters since the last call, as text for the "Desglose draw" line (and reset).
[[nodiscard]] std::string TakeReport(double cycles_per_us);

} // namespace Libs::Graphics::DrawMemo

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWMEMO_H_
