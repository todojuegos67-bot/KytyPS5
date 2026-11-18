#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MTGPUSTAGEA_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MTGPUSTAGEA_H_

#include <cstdint>

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

// GPU thread, after the "Spec" line: the "MT GPU stage A" line (Log::WriteUnmuted).
void Report(uint64_t frames, double seconds);

} // namespace Libs::Graphics::MtGpu

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MTGPUSTAGEA_H_
