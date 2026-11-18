#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MTGPUSTAGEA_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MTGPUSTAGEA_H_

#include <cstdint>
#include <vector>

// Multi-threaded GPU thread, stage A (gt7-mt-1117; KYTY_MT_GPU=2, off by default): record only.
//
// A write registry with stamps: a global epoch (one per recorded write) and, per 64 KiB zone of
// guest memory, the epoch and kind of the last write to it (relaxed atomics, no locks). Writes
// come from the GPU (BufferCache/TextureCache MarkGpuWritten, InvalidateMemoryFromGPU), the host
// (write faults and InvalidateMemory in RenderContext) and retired images (TextureCache).
//
// Each chained INDIRECT_BUFFER (IT 0x3F, control bit 20) of a graphics stream is a chunk. The
// draw-prep scanner stamps a chunk "prepared" when it walks it; the GPU thread stamps it
// "consumed" when it starts it, and records the zones its draws read (buffers obtained, images
// found, the chunk's own packets). A chunk could have been translated ahead and accepted when
// none of the zones it read was written between prepared and consumed. Nothing changes what the
// emulator does: the line every 10 s only says how many chunks would have passed.
namespace Libs::Graphics::MtGpu {

[[nodiscard]] bool StageAEnabled();
// KYTY_MT_GPU as a number (0 when unset or not a number): 1 workers, 2 stage A, 3 self-check,
// 4 workers checked against the normal translation.
[[nodiscard]] uint32_t Mode();
// KYTY_MT_GPU=1 or 4: translation workers (mtGpuWorkers.h).
[[nodiscard]] bool WorkersMode();
// KYTY_MT_GPU=1/4 (mtGpuWorkers.cpp), measurement. Worker: why FindImage found no image of the
// same backing ("imagen sin la misma memoria"): what lies at the descriptor's memory. GPU
// thread: how its FindImage ended while the workers run. Both counted in the 10 s line.
void NoteImageMiss(const char* what);
void NoteGpuImageLookup(const char* what);
// The write registry (NoteWrite) is kept: KYTY_MT_GPU=1, 2 or 4.
[[nodiscard]] bool RegistryEnabled();

enum class WriteKind : uint32_t { None = 0, Gpu = 1, Host = 2, ImageRetired = 3 };

// Any thread.
void NoteWrite(WriteKind kind, uint64_t vaddr, uint64_t size);
// Draw-prep scanner: it walked the chained buffer at `address`.
void NotePrepared(uint64_t address);
// GPU thread (the command processor): a chained buffer starts / the stream ended.
void ChunkBegin(uint64_t address, uint64_t size_bytes);
void StreamEnd();
// GPU thread: a draw or dispatch packet of the current chunk.
void NoteDraw();
// GPU thread (ignored elsewhere and outside a chunk): the current chunk reads this range.
void NoteRead(uint64_t vaddr, uint64_t size);

// Stage B self-check (KYTY_MT_GPU=3, off by default): the GPU thread translates the first draw
// of each chained buffer a second time with another RenderExecutor::TranslationContext and
// compares bindings and pipeline (renderDraw.cpp); differences go to the log at once.
[[nodiscard]] bool SelfCheckEnabled();
// GPU thread: true once per chained buffer (its first draw that reaches PrepareBindings).
[[nodiscard]] bool TakeSelfCheckDraw();
void NoteSelfCheck(bool same);

// Stage d (KYTY_MT_GPU=3 for now): a draw is split in TRANSLATE (its bindings, into its own
// RenderExecutor::TranslationContext, recording nothing) and EMIT (the context's deferred
// actions, in order). What translation cannot defer (a texture cache copy or image creation,
// a scheduler Finish) calls NonDeferrable: under a SpeculationScope on that thread it marks
// the translation failed (the draw then goes the normal way); elsewhere it does nothing.
[[nodiscard]] bool DeferredDrawEnabled();
void NonDeferrable(const char* reason);
class SpeculationScope {
public:
	SpeculationScope();
	~SpeculationScope();
	SpeculationScope(const SpeculationScope&)            = delete;
	SpeculationScope& operator=(const SpeculationScope&) = delete;
	// The first NonDeferrable reason seen in the scope, or null.
	[[nodiscard]] const char* Failure() const { return m_failure; }

private:
	friend void NonDeferrable(const char* reason);
	friend void SpeculationFail(const char* reason);
	friend bool SpeculationFailed();
	SpeculationScope* m_previous;
	const char*       m_failure = nullptr;
};
// GPU thread: one translated draw, accepted (null) or failed for `reason`.
void NoteSpeculation(const char* reason);

// KYTY_MT_GPU=1/4 (mtGpuWorkers.h): the translation workers' side of the above.
// The calling thread is one of them (set once per thread, by the pool).
[[nodiscard]] bool OnWorker();
void               SetOnWorker();
// A worker stops a translation it cannot do without side effects (an image to make, a copy):
// marks the scope failed, as NonDeferrable, but nothing was done (no alert).
void SpeculationFail(const char* reason);
// Whether the speculation scope of this thread has failed.
[[nodiscard]] bool SpeculationFailed();
// The registry's epoch now (every write later has a larger one).
[[nodiscard]] uint64_t Epoch();
// Worker: while one lives, NoteRead records the 64 KiB zones read here (sorted, unique at end).
class ReadRecorder {
public:
	explicit ReadRecorder(std::vector<uint64_t>& zones);
	~ReadRecorder();
	ReadRecorder(const ReadRecorder&)            = delete;
	ReadRecorder& operator=(const ReadRecorder&) = delete;

private:
	std::vector<uint64_t>* m_previous;
	std::vector<uint64_t>& m_zones;
};
// GPU thread: null when no zone of `zones` (nor a wide write) was written after `epoch`; else
// what wrote one ("escritura GPU", "escritura host", "imagen creada/retirada").
[[nodiscard]] const char* ZonesWrittenSince(const std::vector<uint64_t>& zones, uint64_t epoch);
// The same for the zones NoteRead records for [vaddr, vaddr + size).
[[nodiscard]] const char* RangeWrittenSince(uint64_t vaddr, uint64_t size, uint64_t epoch);
// NonDeferrable reached on a worker (a path the workers' guards missed): counted and logged.
[[nodiscard]] uint64_t WorkerAlerts();

// GPU thread, after the "Spec" line: the "MT GPU stage A" line (Log::WriteUnmuted).
void Report(uint64_t frames, double seconds);

} // namespace Libs::Graphics::MtGpu

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MTGPUSTAGEA_H_
