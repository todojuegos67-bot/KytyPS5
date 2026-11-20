#pragma once
// (f5b) What the verification's CPU side (pass-verify.cpp) hands its GPU side (verify-gpu.cpp), and the results back.
// Internal to src/native. GPU thread only.
#include <cstdint>

namespace NativeRender::Verify {

constexpr uint32_t MaxTargets = 8;

// A render target a pass wrote: the TextureCache image (id, its generation) and the view's level and layer.
struct TargetRef {
	uint32_t index = 0, generation = 0, level = 0, layer = 0;
	bool     depth = false;
	bool operator==(const TargetRef& o) const {
		return index == o.index && generation == o.generation && level == o.level && layer == o.layer && depth == o.depth;
	}
};

enum class JobKind : uint8_t {
	Entry, // the scope begin: the targets the pass's last instance wrote, copied as they are before it draws
	Exit,  // the scope end: the targets this instance wrote; the emulated copy and the control pass's copy
};

struct Job {
	JobKind   kind     = JobKind::Exit;
	uint32_t  pass     = 0;
	uint64_t  instance = 0; // serial of the pass instance (Entry and Exit of one instance share it)
	int32_t   queue    = 0; // run with the command buffer of this queue's walk only
	uint32_t  count    = 0;
	TargetRef targets[MaxTargets];
};

// Why a target was not compared (verify-gpu.cpp).
enum Refusal : uint32_t {
	RNoImage,      // the image is gone, or another one is in its slot
	RFormat,       // a format whose texel size is not known here (compressed, planar, ...)
	RSamples,      // multisampled
	RType,         // not a 2D image
	RLayout,       // undefined contents (never written)
	RShadow,       // the shadow image could not be made
	RBudget,       // this frame's readback space is used up
	RNoBuffer,     // the command buffer is not recording / the readback buffer could not be made
	Refusals
};
inline constexpr const char* RefusalNames[Refusals] = {"sin_imagen", "formato", "msaa", "no_2d",
                                                        "sin_contenido", "sombra", "presupuesto", "sin_buffer"};

// pass-verify.cpp: the CPU side's queue (the front job, if it is for the queue translating now).
[[nodiscard]] bool TakeJob(Job& out);
[[nodiscard]] uint64_t FlipCount() noexcept;

// verify-gpu.cpp -> pass-verify.cpp, when a comparison's readback is in (8 frames later).
struct ImageOutcome {
	uint32_t pass = 0;
	uint64_t instance = 0;
	uint32_t format = 0, width = 0, height = 0, layer = 0, level = 0;
	bool     depth  = false;
	uint64_t texels = 0;      // compared (sampled tiles)
	uint64_t differing = 0;   // emulated exit vs the control pass's exit: must be 0
	bool     entry_known = false;
	uint64_t changed = 0;     // emulated exit vs the entry copy: what the pass changed (the comparator sees changes)
};
void ImageResult(const ImageOutcome& outcome);
void ImageRefused(uint32_t pass, Refusal why);
// verify-gpu.cpp: results whose frame is old enough (called at the flip; reads mapped memory only).
void CollectGpu(uint64_t flip);

} // namespace NativeRender::Verify
