#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_REPLAY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_REPLAY_H_

#include <cstdint>
#include <string>

// gt7-replay-1119, KYTY_REPLAY (off by default): a draw's translation kept by its place in the
// stream, from one frame to the next.
//
// GT7 sends one graphics DCB a frame (FUN_0011a640). Its workers write ~520 chunks a frame, each
// ended by FUN_0002aac0 with the pending Set{Sh,Cx,Uc}RegistersIndirect lists and an
// INDIRECT_BUFFER (0xc0023f00, control 0x0f300000 | size) that chains the next chunk; a chunk
// inherits the registers of the one before it. The descriptor tables its binders fill (such as
// FUN_000e62f0) come from a lock-free linear allocator (FUN_0000acf0: 2 MiB blocks from
// FUN_00005710, a fresh address each frame), and only their address goes into the user data (a
// 64-bit pointer in two SGPRs); the words in them are built from persistent objects
// (FUN_00005f60/FUN_00005ca0) and repeat. So the stream has the same shape almost every frame
// (99.5 %) while its bytes almost never repeat (1.3 %).
//
// The shape of a chunk: its packets' opcodes and lengths and the registers they set (the
// structural key, without values), chained with the chunk before it (the state it inherits).
// A draw's place: that key and its number in the chunk. What a draw's translation resolved there
// is kept by place (the stage's images and samplers, its color targets' descriptions) together
// with the inputs it came from (program, descriptor words, render target registers, texture memo
// stamps), and a later draw in the same place whose inputs are the same takes it again instead of
// resolving it. The place is only where to look: every input is compared before anything is
// used, so a wrong place costs a miss, never a wrong binding.
//
//   KYTY_REPLAY=1  the kept translation is used.
//   KYTY_REPLAY=2  verifier: the normal path runs and is used; what the replay would have given
//                  is compared with it and the differences are counted and logged (the first 16).
//
// A "Replay" line every 10 s (Log::WriteUnmuted). GPU thread only.
namespace Libs::Graphics::Replay {

// 0 (off), 1 (on) or 2 (verifier), from KYTY_REPLAY.
[[nodiscard]] int Mode();

// A command stream's place (in its Pm4Execution): the key of the open chunk and how many draw
// and dispatch packets it has had.
struct StreamState {
	uint64_t        chain       = 0; // key of the open chunk (0: the stream's own buffer)
	uint32_t        ordinal     = 0;
	const uint32_t* last_packet = nullptr; // the last numbered packet (one suspended runs again)
};

// Command processor, GPU thread. A chained buffer starts a chunk; a draw or dispatch packet;
// the stream ended.
void ChunkBegin(StreamState& stream, const uint32_t* commands, uint32_t dwords);
void NotePacket(StreamState& stream, const uint32_t* packet);
void StreamEnd(StreamState& stream);

// The stream whose packets run on this thread while it lives.
class StreamScope {
public:
	explicit StreamScope(StreamState* stream);
	~StreamScope();
	StreamScope(const StreamScope&)            = delete;
	StreamScope& operator=(const StreamScope&) = delete;

private:
	StreamState* m_previous;
};

// Around a draw (DrawIndex, DrawAuto): its place, its time, and whether every part of it came
// from the replay (=1) or would have (=2).
class DrawScope {
public:
	DrawScope();
	~DrawScope();
	DrawScope(const DrawScope&)            = delete;
	DrawScope& operator=(const DrawScope&) = delete;

private:
	uint64_t m_previous_key;
	uint32_t m_previous_parts;
	uint32_t m_previous_missed;
	int64_t  m_begin;
	bool     m_active;
};

// The place of the draw being translated on this thread, 0 when there is none (off, no stream,
// not in a draw). Never 0 for a place.
[[nodiscard]] uint64_t DrawKey();
// `DrawKey` mixed with a part's own number (a stage's type, a target slot).
[[nodiscard]] uint64_t PartKey(uint64_t draw_key, uint32_t kind, uint32_t index);

// What a part of a draw found. Images and samplers are a stage's; targets are color targets;
// depth is the depth target's description; descriptors and constants are the draw's push of its
// descriptor set and push constants, left out when the command buffer holds them already.
enum class Part : uint32_t { Image, Samplers, Target, Depth, Descriptors, Constants, Count };
enum class Miss : uint32_t {
	Ineligible, // the stage or image cannot be kept (bindless, storage, diagnostics, ...)
	NoEntry,    // nothing kept at this place
	Program,    // another program at this place
	Words,      // other descriptor words, or other render target registers
	NotKept,    // kept but not reusable (not a texture memo answer, a light grid)
	MemoStale,  // the texture memo entry changed or the image set was dropped
	Stamp,      // images made or dropped over the image's memory
	Image,      // the image gone or to be bound again
	Count
};
// Per part: reused (=1) or reusable (=2), or missed for `why`.
void NoteHit(Part part);
void NoteMiss(Part part, Miss why);
// =2: the replay's answer was compared with the normal path's; `what` differed (null: the same).
void NoteVerified(Part part, const char* what, uint64_t shader_hash, uint32_t index);
// A part stored (a new or changed entry).
void NoteStored(Part part);
// The kept entries were dropped (too many).
void NoteDropped(Part part);

// GPU thread, every 10 s after the GPU phases: the "Replay" line (nothing while off).
void Report(double seconds);

} // namespace Libs::Graphics::Replay

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_REPLAY_H_
