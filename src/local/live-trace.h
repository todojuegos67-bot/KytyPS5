#pragma once
// Local diagnostic: a timeline of scheduling events (live `trace SECONDS PATH`).
// Recording costs one relaxed load while tracing is off.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "local-platform.h"

#include <x86intrin.h>

namespace LiveTrace {

// Values 3-6 and 15-18 belonged to events no longer recorded; they are not reused.
enum Type : uint32_t {
	Submit        = 1,  // a: queue | type << 8, b: dwords
	GuestDone     = 2,  // a: submitted frame
	EqueueWait    = 7,  // a: 1 begin / 0 end, b: events received
	FlipSubmit    = 8,  // a: request id
	FlipComplete  = 9,  // a: request id
	FrontSuspend  = 10, // a: address, b: value
	GpuSubmit     = 11, // a: tick
	GpuDone       = 12, // a: tick
	GuestReadback = 13, // a: 1 begin / 0 end, b: address
	RenderIdle    = 14, // a: 1 begin / 0 end, b: 0 empty, 1 blocked
	ReadbackTicks = 19, // a: read address, b: size | tick the copy is recorded in << 32
	TickDone      = 20, // a: tick the GPU completed (monitor thread)
	RenderSlice   = 21, // a: queue | type << 8 | complete << 16 | begin << 17, b: epoch | dwords << 32
	GpuSpan       = 22, // a: tick, b: GPU nanoseconds begin << 32... see GpuSpanNs
	GpuSpanNs     = 23, // a: GPU begin timestamp (ns), b: GPU end timestamp (ns); follows GpuSpan
	GpuMark       = 24, // a: mark slot, b: tag (shader address of the draw/dispatch recorded)
	GpuMarkValue  = 25, // a: mark slot, b: GPU timestamp (ns) after the command completed
	GpuWrite      = 26, // a: address, b: size | tick the writing command is recorded in << 32
	LabelWrite    = 27, // a: address, b: value (low 48 bits) | source << 56 (1 end of pipe, 2 WRITE_DATA)
	SyncReadback  = 28, // a: address, b: size | tick the copy is recorded in << 32 (drains the queue)
	RbMismatch    = 29, // a: address, b: size | tick the copy engine waited for << 32 (KYTY_READBACK_QUEUE=3)
	RbFast        = 30, // a: window address, b: size | tick the copy engine waits for << 32
	BufferUse     = 31, // a: address, b: size | 1 << 63 when written (every ObtainBuffer, `tracew`)
	FaultSite     = 32, // a: faulting guest instruction, b: fault address | 1 << 63 for a write (tracked pages)
	FaultCaller   = 33, // follows FaultSite for host code: a: the qword at the stack top, b: faulting instruction
	ImageUse      = 34, // a: image address, b: size (low 32 bits) | layout << 32 | 1 << 63 when written (`tracew`)
	SlowOsCall    = 35, // a: address, b: call << 56 | TSC ticks / 1024 (24 bits) << 32 | size in 4 KiB pages
	FaultDone     = 36, // the handled fault of the thread's last FaultSite: a: 1 write / 0 read, b: fault address
	GuestCommand  = 37, // render thread runs a guest thread's command (SendCommand): a: 1 begin / 0 end
};

// SlowOsCall ids: address-space calls that hold the process's memory locks (a page fault anywhere
// in the process waits for them) and the guest address space's own lock.
enum OsCall : uint64_t {
	OsMapView = 1, OsUnmapView = 2, OsProtect = 3, OsFree = 4, OsAlloc = 5, OsUnmapBacking = 6, OsMapBacking = 7,
};

struct Record {
	uint64_t tsc;
	uint32_t tid;
	uint32_t type;
	uint64_t a, b;
};

constexpr size_t           Capacity = 1u << 22;
inline std::atomic_bool     g_on {false};
inline std::atomic<uint64_t> g_count {0};
inline Record               g_records[Capacity];

// GPU buffer writes and readback ranges as events (live `tracew`): producers are matched offline.
inline std::atomic_bool g_writes_on {false};
inline bool             WriteTicks() {
	return g_writes_on.load(std::memory_order_relaxed) && g_on.load(std::memory_order_relaxed);
}

// GPU timestamps per command buffer while tracing: query pair `slot` of g_timestamp_pool.
constexpr uint32_t          TimestampSlots = 8192;
inline void*                g_timestamp_pool = nullptr; // VkQueryPool
inline double               g_timestamp_period = 1.0;  // ns per tick
inline std::atomic<uint32_t> g_timestamp_next {0};
inline std::atomic<uint32_t> g_tick_slot[TimestampSlots]; // tick % TimestampSlots -> slot + 1

// Per-draw/dispatch GPU timestamps while tracing (`trace` with marks): a timestamp after each
// recorded draw or dispatch, tagged with the recording thread's current tag.
constexpr uint32_t           MarkSlots = 1u << 20;
inline void*                 g_mark_pool = nullptr; // VkQueryPool
inline std::atomic_bool      g_marks_on {false};
inline std::atomic<uint32_t> g_mark_next {0};
inline thread_local uint64_t g_mark_tag = 0;
// Set by the renderer: reads the used marks into the trace (after tracing stops).
inline void (*g_read_marks)(uint32_t count) = nullptr;

inline void Append(uint32_t type, uint64_t a, uint64_t b) {
	const auto index = g_count.fetch_add(1, std::memory_order_relaxed);
	if (index < Capacity) g_records[index] = {__rdtsc(), 0, type, a, b};
}

inline uint32_t ThreadId() {
	static thread_local uint32_t tid = LocalPlatform::ThreadId();
	return tid;
}

inline void Event(uint32_t type, uint64_t a = 0, uint64_t b = 0) {
	if (!g_on.load(std::memory_order_relaxed)) return;
	const auto index = g_count.fetch_add(1, std::memory_order_relaxed);
	if (index >= Capacity) return;
	g_records[index] = {__rdtsc(), ThreadId(), type, a, b};
}

// Records the call when it took at least ~0.25 ms (and tracing is on).
struct SlowOsCallScope {
	uint64_t call, address, size, start;
	SlowOsCallScope(uint64_t c, uint64_t a, uint64_t s): call(c), address(a), size(s), start(__rdtsc()) {}
	~SlowOsCallScope() {
		const uint64_t ticks = __rdtsc() - start;
		if (ticks < 800000 || !g_on.load(std::memory_order_relaxed)) return;
		const uint64_t units = std::min<uint64_t>(ticks >> 10u, 0xffffff);
		Event(SlowOsCall, address, call << 56u | units << 32u | std::min<uint64_t>(size >> 12u, 0xffffffffu));
	}
};

// Called by the live thread.
inline void Dump(const char* path) {
	const auto count = std::min<uint64_t>(g_count.load(), Capacity);
	if (auto* out = std::fopen(path, "wb")) {
		std::fwrite(g_records, sizeof(Record), count, out);
		std::fclose(out);
	}
}

} // namespace LiveTrace
