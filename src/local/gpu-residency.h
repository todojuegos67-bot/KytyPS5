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

#include "mem1.h" // (mem1: KYTY_MEM1=1 needs the answer and the ranges too)

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

// (ram19) KYTY_RAM19=1 (off by default): the "[ram19]" lines every 30 s (ram-guard.cpp, PrintRam19): the game's memory
// per allocator split by what the GPU caches hold over it and by where it is (RAM, taken by an offload, out of RAM but
// accessible: Windows' compression store, the page file or never written), and the emulator's own allocations the same
// way. It also answers the request at the end of a submission (GpuResourceManager::EndSubmission): with the game's own
// flips (KYTY_RAIZ) no FlipPreparation comes, so AdvanceFrame never answered ("no answer from the GPU thread in 10 s" in
// every log of 11-16/11-17). Off: one static bool read at a submission's end, nothing else.
// (ram20) KYTY_RAM20=1 (off by default; see Ram20On below) needs the same answer and lines: it turns them on too.
inline bool Ram20On();
inline bool Ram19On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_RAM19");
		return (value != nullptr && value[0] == '1' && value[1] == '\0') || Ram20On() || Mem1::On(); // (mem1: its census)
	}();
	return on;
}

// (ram20) KYTY_RAM20=1 (off by default; with native memory, KYTY_RAIZ=1): the game's memory that has a copy in video
// memory (under a GPU cache's image or buffer, as the last GPU residency pass saw it) let go of in RAM and in Windows'
// compression store: written to the RAM17 disk store and reset (texture-offload.h, the RAM18 cold-range path: no-access
// with read watchers, any touch reads it back from the file on its own thread), "[ram20]" line. It turns on KYTY_RAM17
// (and with it RAM16) and the "[ram19]" lines (ram-guard.cpp). Off: one static bool read where it is asked.
inline bool Ram20On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_RAM20");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return on;
}

// (ram20) The GPU caches' ranges of the last answered residency pass (ram-guard.cpp's measurement thread stores them;
// the RAM18/RAM20 thread reads them). `pass`: its number (0: none yet).
struct Ram20Snapshot {
	std::mutex         mutex;
	std::vector<Range> ranges;
	uint64_t           pass = 0;
};
inline Ram20Snapshot& Ram20Ranges() {
	static auto* snapshot = new Ram20Snapshot; // (never destroyed: detached threads)
	return *snapshot;
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
