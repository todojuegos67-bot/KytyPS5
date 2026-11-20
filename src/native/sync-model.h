#pragma once
// (f5) Native barrier / sync model: the interface (design; no implementation in this build).
//
// The game orders its work with labels: the CPU sets a label to 0 when it records the signing pass
// (0x912890 DCB / 0x9151d0 ACB), the pass ends with RELEASE_MEM writing 1 at end of pipe, and the waiter does
// WAIT_REG_MEM == 1 (VmW0Tdpy420 DCB / htn36gPnBk4 ACB, poll 400). The three label blocks of a frame are
// L0 = [ctx+0x110], L1 = [ctx+0x118], L2 = [ctx+0x120] (acb1). The CPU never writes the 1 (acb1: proven for the 9
// skinning/particle waits and seen for the ~35 other ACB stops).
//
// Native equivalent: a label is a point on a timeline semaphore of its signing queue. Signal = the native pass (or the
// emulated range that holds the RELEASE_MEM) ends; wait = the consuming queue waits that value. Inside one queue a
// label wait becomes a pipeline barrier (or nothing, when the frame graph already orders it). EVENT_WRITE cache
// flushes inside native passes are dropped: the frame graph's hazards give the barriers (frame-graph.h).
//
// Hybrid mode: a label signed by a native pass and waited by an emulated one (or the reverse) keeps today's mechanism
// (the emulator writes the label when it translates / completes the RELEASE_MEM: P7 LabelsAtCompletion); the native
// pass just records the same RELEASE_MEM effect at its end. Labels are replaced by semaphores only when both ends
// are native.
#include <cstdint>

namespace NativeRender {

enum class LabelBlock : uint8_t {
	L0 = 0, // [ctx+0x110]
	L1 = 1, // [ctx+0x118]
	L2 = 2, // [ctx+0x120]
};

struct LabelRef {
	LabelBlock block  = LabelBlock::L0;
	uint32_t   offset = 0; // bytes into the block (e.g. L2+0x380 DepthPrePass, L2+0x4c0 end of particles)
};

enum class SyncKind : uint8_t {
	ReleaseLabel = 0,
	AcquireLabel = 1,
};

struct SyncEdge {
	LabelRef label;
	SyncKind kind      = SyncKind::ReleaseLabel;
	uint32_t pass      = 0;     // pass-registry id
	bool     compute   = false; // the ACB side
};

class SyncModel {
public:
	virtual ~SyncModel() = default;
	// One frame's label edges, from the frame map (static) and the pass captures (this frame's addresses).
	virtual void BeginFrame(uint64_t frame, uint64_t l0, uint64_t l1, uint64_t l2) = 0;
	// The native pass `pass` ends and signals `label` (records the timeline signal or the RELEASE_MEM equivalent).
	virtual void SignalLabel(uint32_t pass, const LabelRef& label, void* cmd) = 0;
	// The native pass `pass` needs `label` signalled first: a barrier on the same queue or a semaphore wait.
	virtual void WaitLabel(uint32_t pass, const LabelRef& label, void* cmd) = 0;
	// The guest address of a label this frame (for the hybrid path that still writes it).
	[[nodiscard]] virtual uint64_t Address(const LabelRef& label) const = 0;
};

} // namespace NativeRender
