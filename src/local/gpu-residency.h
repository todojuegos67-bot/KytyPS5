#pragma once
// GPU residency (KYTY_GPU_RESIDENCY=0 turns it off; default on, Windows): every 30 s, how much of the guest memory under
// the GPU caches' registrations is in the process's working set (RAM), by use. A measurement only: no hook, no
// protection change, nothing written to guest memory.
//
// The GPU thread copies the registered images' and buffers' guest ranges at the next flip after a request
// (GpuResourceManager::AdvanceFrame: the texture cache's lock held for the copy of its images; the buffer cache is the
// GPU thread's); the measurement's own thread (ram-guard.cpp, low priority) then queries the working set
// (K32QueryWorkingSetEx) at one page in each 64 KiB of every range, through both views of the page: the game's (the
// guest address) and the emulator's (the backing store's alias, through which the emulator reads uploads and writes
// GPU results back). See ram-guard.cpp's PrintGpuResidency for the line.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

namespace GpuResidency {

enum Kind : uint8_t {
	Texture          = 0, // sampled-only images
	Target           = 1, // render targets, depth targets, video outs
	Storage          = 2, // storage images and other images the GPU wrote (not targets)
	GpuBuffer        = 3, // buffers no GPU write is pending in
	GpuWrittenBuffer = 4, // buffers with a page the GPU wrote (MemoryTracker)
	Kinds            = 5,
};

struct Range {
	uint64_t address = 0;
	uint64_t size    = 0;
	Kind     kind    = Texture;
};

inline bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_GPU_RESIDENCY");
		return value == nullptr || !(value[0] == '0' && value[1] == '\0');
	}();
	return on;
}

// The measurement thread's request and the GPU thread's answer.
struct Exchange {
	std::mutex              mutex;
	std::condition_variable delivered;
	std::atomic<bool>       requested {false};
	uint64_t                request = 0; // the request's number
	uint64_t                answer  = 0; // the number of the request `ranges` answers
	std::vector<Range>      ranges;      // (the measurement's reserved vector while a request is open)
	uint64_t                images = 0, buffers = 0;
	double                  copy_ms = 0.0;
};

inline Exchange& Get() {
	static auto* exchange = new Exchange; // (never destroyed: the detached thread may outlive static destructors)
	return *exchange;
}

// The GPU thread at a flip: whether a request is open (an atomic load).
[[nodiscard]] inline bool Requested() noexcept {
	return Get().requested.load(std::memory_order_acquire);
}

// The GPU thread: the request's vector (empty, reserved by the measurement) in `out`, and its number; 0 (nothing taken,
// the request left open for the next flip) when the measurement's thread holds the mutex: the GPU thread never waits on
// it (a low-priority holder descheduled under load could otherwise stall a flip).
inline uint64_t Take(std::vector<Range>* out) {
	auto&                        exchange = Get();
	std::unique_lock<std::mutex> lock(exchange.mutex, std::try_to_lock);
	if (!lock.owns_lock()) return 0;
	out->swap(exchange.ranges);
	out->clear();
	exchange.requested.store(false, std::memory_order_relaxed);
	return exchange.request;
}

// The GPU thread: the ranges copied for request `request` (dropped, never waited for, when the mutex is held: that pass
// then goes unmeasured).
inline void Deliver(uint64_t request, std::vector<Range>&& ranges, uint64_t images, uint64_t buffers, double copy_ms) {
	auto& exchange = Get();
	{
		std::unique_lock<std::mutex> lock(exchange.mutex, std::try_to_lock);
		if (!lock.owns_lock()) return;
		exchange.ranges.swap(ranges); // (the previous contents freed by the caller, outside the lock)
		exchange.answer  = request;
		exchange.images  = images;
		exchange.buffers = buffers;
		exchange.copy_ms = copy_ms;
	}
	exchange.delivered.notify_all();
}

// The measurement's thread, once (ram-guard.cpp; nothing off Windows or with KYTY_GPU_RESIDENCY=0).
void Start();

} // namespace GpuResidency
