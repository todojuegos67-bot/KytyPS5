#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MTGPUWORKERS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MTGPUWORKERS_H_

#include "graphics/host_gpu/renderer/drawPrep.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/render.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

// Multi-threaded GPU thread, stage d with workers (gt7-mt-1117; KYTY_MT_GPU=1 and 4, off by
// default; KYTY_MT_GPU_THREADS workers, 3 by default).
//
// The draw-prep scanner (drawPrepScanner.cpp) reads each graphics submission ahead of the GPU
// thread on its own register copy, from the moment the submission is enqueued (GuestGpu::
// Enqueue). Each chained INDIRECT_BUFFER (IT 0x3F, control bit 20) starts a chunk, which goes to
// the workers and the GPU thread right away (empty); for every draw of the chunk it prepares
// (the shader pair known, the registers followed), the scanner exports what its registers make
// of the draw (PipelineCache::DrawPrepInputs: the stages' parameters and inputs, from its
// registers at that draw) with the draw's key, and the draw can be taken at once.
//
// A worker takes the oldest draw no worker has taken (several workers share a chunk): walks
// the stages' descriptors as the scanner's walks do (DrawPrepWalk: only bytes no GPU work wrote,
// recorded), then, without any lock (no try-lock, no renderer lock, not m_translation_mutex),
// finds the stages' programs in the permutations each program entry publishes and translates the
// draw's bindings in deferred mode in its own TranslationContext (RenderExecutor::SpeculateDraw),
// reading the texture cache through its worker view (textureWorkerView.h: read-copy-update
// records the GPU thread publishes when it registers, unregisters or changes an image, and the
// overlap results FindImage reached with no change to the image set). It creates nothing: an
// image, null image, sampler or program not there yet, colour metadata, a copy, a bindless heap,
// a diagnostic that logs in order fail the draw, and the GPU thread translates it as always.
// The 64 KiB zones the translation read are recorded with the write registry's epoch (stage A),
// read before anything else: every change the GPU thread publishes is stamped after it is
// published.
//
// The GPU thread, at each draw (EmitPreparedDraw): when no worker has finished the draw it is
// passed over (it never waits; a worker not started on it never starts); when one has, the
// worker's translation is used if its programs and descriptors are the GPU thread's own for the draw, no
// zone it read was written (GPU, host, an image made or retired) after its epoch, and the images
// it bound are still there: its BindImage calls are made again in the GPU thread's context, its
// bindings moved in, its deferred actions run (EmitPreparedDraw then goes on as for a
// translated draw). Otherwise the draw is translated the normal way. The order of what is
// emitted is the normal order. Dispatches always go the normal way.
// KYTY_MT_GPU=4: the same, but the normal translation is always the one used, and each worker
// translation that would have been used is compared with it (differences: Log::WriteUnmuted).
// A line every 10 s (both): chunks speculated, accepted, rejected by reason, draws emitted from
// a worker, and the GPU thread's milliseconds saved (estimated).
namespace Libs::Graphics {
class RenderContext;
} // namespace Libs::Graphics

namespace Libs::Graphics::MtGpu {

// One draw of a chunk.
struct SpecDraw {
	DrawPrep::Key                 key;
	PipelineCache::DrawPrepInputs inputs;
	// Worker (read by the GPU thread once the chunk is done):
	bool                                  translated = false;
	const char*                           failure    = nullptr; // why not (null: translated)
	uint64_t                              epoch      = 0;       // the registry's, at translation
	std::vector<uint64_t>                 zones;                // read by the translation
	DrawPrep::PacketResult                walked;               // the stages' resources (owned)
	std::array<ShaderStageRuntime, 4>     stages {};            // DrawPrepWalk's order: pixel first
	std::array<bool, 4>                   normalized {};
	RenderExecutor::TranslationContext    tc;
	std::vector<std::pair<ImageId, bool>> binds; // BindImage's calls, in order
	TextureWorkerView::FoundLog           found; // what its FindImage calls found
	// Who has it: Free (no one yet), Running (a worker), Done (the worker's results above are
	// complete: release/acquire), Abandoned (the GPU thread got there first: no worker starts it).
	static constexpr uint32_t Free      = 0;
	static constexpr uint32_t Running   = 1;
	static constexpr uint32_t Done      = 2;
	static constexpr uint32_t Abandoned = 3;
	std::atomic<uint32_t>     state {Free};
	uint64_t                  done_ns = 0; // worker: when it was Done
	// GPU thread: taken already.
	bool consumed = false;
};

// KYTY_MT_GPU=1 or 4, with the draw-prep scanner on.
[[nodiscard]] bool WorkersEnabled();
// KYTY_MT_GPU=4: compare, use the normal translation.
[[nodiscard]] bool WorkersCompare();

// The graphics scanner (drawPrepScanner.cpp), in its thread: a chained buffer starts a chunk
// (and ends the one before), a draw it prepared goes into the open chunk, the submission's end
// (or a stop of the scan) ends it.
void SpecChunkBegin(uint64_t seq, uint64_t address);
void SpecAddDraw(const DrawPrep::Key& key, const PipelineCache::DrawPrepInputs& inputs);
void SpecChunkEnd();

// The workers: started and stopped with the graphics scanner.
void StartWorkers(RenderContext& renderer);
void StopWorkers();

// GPU thread, at a draw: the worker's translation of the draw `key` names, or null (`reason`:
// why, or null when the draw was not speculated at all). Valid until the next call.
[[nodiscard]] SpecDraw* TakeSpecDraw(const DrawPrep::Key* key, const char** reason);
// GPU thread: what became of the draw TakeSpecDraw gave (rejected: `reason`; `emitted`: its
// translation was used; KYTY_MT_GPU=4: `differs`, compared with the normal one), and the time
// spent taking and checking it.
void NoteSpecOutcome(const char* reason, bool emitted, bool differs, uint64_t consume_ns);
// GPU thread: a draw translated the normal way while the workers run, and how long it took.
void NoteNormalTranslation(uint64_t ns);
// Worker: a translation done (or failed) and how long it held the locks.
void NoteWorkerDraw(const char* failure, uint64_t held_ns, uint32_t lock_retries);

// GPU thread, every 10 s (MtGpu::Report): the line of KYTY_MT_GPU=1 and 4.
void ReportWorkers(double seconds);

} // namespace Libs::Graphics::MtGpu

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MTGPUWORKERS_H_
