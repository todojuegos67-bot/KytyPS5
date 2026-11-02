#pragma once
// KYTY_PRIO_RANGE_WAIT (A/B, off by default): which deferred priority operations a guest readback must wait for.
// A priority operation (CommandScheduler::DeferPriorityOperation) runs on the priority thread after its tick completes.
// Each declares what guest memory it writes:
//   Unknown - not declared: it may write anything (waited for by every readback, as before);
//   None    - it writes no guest memory (flip completion, interrupt / event delivery);
//   Range   - it writes guest memory in [begin, end) only (a texture download's write-back).
// A readback of [window_begin, window_end) whose bytes the GPU finished by `tick` waits, as before, for every
// operation of a tick <= `tick`, unless range-aware: then only for those that may write its window (Unknown, or a
// Range overlapping it). An operation that may write the window keeps its order before the readback's write-back;
// one that cannot write it has no order with the readback's bytes to keep.
#include <cstdint>

namespace PrioWait {

enum Class : uint8_t { Unknown = 0, None = 1, Range = 2 };

struct Op {
	uint64_t tick  = 0;
	uint8_t  cls   = Unknown;
	uint64_t begin = 0, end = 0;
};

[[nodiscard]] inline bool MayWrite(const Op& op, uint64_t window_begin, uint64_t window_end) {
	if (op.cls == None) return false;
	if (op.cls == Range) return op.begin < window_end && window_begin < op.end;
	return true;
}

// True while the readback must keep waiting. `queued` in tick order (FIFO, ticks non-decreasing).
template <typename Queue>
[[nodiscard]] bool Blocks(const Queue& queued, bool active, const Op& active_op, uint64_t tick, uint64_t window_begin,
                          uint64_t window_end, bool range_aware) {
	if (active && active_op.tick <= tick && (!range_aware || MayWrite(active_op, window_begin, window_end))) return true;
	for (const auto& op: queued) {
		if (op.tick > tick) break;
		if (!range_aware || MayWrite(op, window_begin, window_end)) return true;
	}
	return false;
}

} // namespace PrioWait
