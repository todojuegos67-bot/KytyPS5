#pragma once
// KYTY_GPU_ONLY_WRITES=1 (off by default): what the GPU writes stays in video memory, as on a native port, unless the
// game's CPU touches it. Nothing here frees, decommits, trims or moves guest memory: pages that would have been written
// only by the emulator are simply never written, so they never become resident.
//
// The pattern is the one the trackers already use: a page the GPU wrote stays no-access in the game's view (the buffer
// cache's GPU-modified state) and its bytes are copied into guest memory on the first CPU access, read or write
// (GpuResourceManager::HandleFault -> BufferCache::ReadMemory). What this changes:
//
// 1. Readback windows. A CPU access to GPU-written memory was read back with a speculative window around it: 512 KiB
//    (BeginGuestReadback, ReadMemoryOnGpu) or a whole 64 KiB copy-feedback snapshot (TryGuestReadFromFeedback). Every
//    GPU-written page of the window was written into guest memory through the emulator's alias of the store, whether
//    the game ever read it or not. Field log 20261114-022456 (KYTY_NATIVE_MEM=2, gameplay windows): 9.2-12.3k
//    write-backs/s, 1.6-2.6 GB/s, about one per fault (faults: write 6.3-6.9k/s, read 3.6-4.0k/s); "GPU-written buffers:
//    the emulator's alias only 214-385 MiB" resident (pages written back that the game's view never touched). For memory
//    of the engine's GPU allocators (EMemoryType 2: VirtualMemoryGPU, the GPU Heap/Page/Bucket allocators, the GPU
//    Default TaggedArena, the render-target arenas) the window is cut to the pages of the access: the other pages stay
//    GPU-owned and no-access, and fault on their own first touch. The engine's CPU readback memory is its CPU context
//    (CBuffer::Create flag 0x1 -> CPU context 0x2ea0d00: "readback or CPU-side buffers"); the CPU allocators keep the
//    window (in the field log the CPU's read faults are there: HeapAllocator #4 3.0-3.1k/s, TaggedArena #10 0.4-0.5k/s;
//    GPU allocators: GPU Heap #8 80-88/s, the rest 0-11/s, GPU TaggedArena #11 0 with 1.1-1.3 GB/s written back).
//    Memory no engine run holds (stacks, libc heap, flexible memory, runs newer than the census walk) is not narrowed.
//    The classification is a policy, not a correctness condition: a narrowed window only leaves more pages GPU-owned,
//    which the trackers already handle.
// 2. Never-written guest bytes: the zero-fill of never-written upload sources this branch first carried here is
//    KYTY_NATIVE_GUESTMEM=1's (src/local/native-guestmem.h: the same sites, plus the async ReadMappedOrZeroCall), one
//    implementation; set KYTY_NATIVE_GUESTMEM=1 alongside this switch for both halves.
//
// KYTY_GPU_ONLY_WRITES_WINDOW_KB=<n> (default 4: the pages of the access): the window kept around an access to GPU-class
// memory, a power of two from 4 to 512 (512: as before). Wider windows trade written-back pages for fewer faults.
//
// Needs the engine census (KYTY_NATIVE_MEM, which this turns on at level 1 when unset) for the GPU-class runs; until the
// first walk nothing is narrowed. Logged with the census every 10 s ("[native-mem]   GPU-only writes ...").

#include "graphics/host_gpu/uploadHeap.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

namespace GpuOnlyWrites {

inline bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_GPU_ONLY_WRITES");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return on;
}

// The window around an access to GPU-class memory, bytes (a power of two, 4 KiB..512 KiB).
inline uint64_t WindowBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_GPU_ONLY_WRITES_WINDOW_KB");
		uint64_t    kib   = value != nullptr ? std::strtoull(value, nullptr, 10) : 4;
		if (kib < 4) kib = 4;
		if (kib > 512) kib = 512;
		uint64_t pow2 = 4;
		while (pow2 * 2 <= kib) pow2 *= 2;
		return pow2 << 10u;
	}();
	return bytes;
}

// GPU-class map: one bit per 64 KiB granule of guest address space below 64 GiB (the engine's page runs are 64 KiB
// aligned: WalkRuns checks it). Written by the census thread only, read lock-free by any thread.
constexpr uint64_t GranuleBits = 16;
constexpr uint64_t SpanBits    = 36;
constexpr uint64_t Granules    = uint64_t {1} << (SpanBits - GranuleBits);
constexpr uint64_t Words       = Granules / 64;

struct State {
	std::atomic<uint64_t> gpu[Words] {};
	std::atomic<uint64_t> gpu_bytes {0};  // bytes of whole granules in the map now
	std::atomic<uint64_t> publishes {0};  // walks published
	// Counters (since start).
	std::atomic<uint64_t> narrowed {0};       // accesses whose window was cut
	std::atomic<uint64_t> skipped_bytes {0};  // window bytes not taken (an upper bound of what was not written back)
	// Write-backs into guest memory by source (bytes): readback (fault windows, asynchronous), download (synchronous:
	// ReadMemoryOnGpu, discards' edge pages, collector), feedback (copy-feedback snapshots), image, other.
	std::atomic<uint64_t> written_back[5] {};
};

inline State& Get() {
	static State state;
	return state;
}

// The census thread: the GPU-class page runs now ([begin, end) pairs, any order). Granules wholly inside a run are set,
// the others cleared; only words that changed are stored.
inline void Publish(const std::vector<std::pair<uint64_t, uint64_t>>& runs) {
	auto&                 s = Get();
	std::vector<uint64_t> words(Words, 0);
	uint64_t              bytes = 0;
	for (const auto& [begin, end]: runs) {
		if (end <= begin) continue;
		const uint64_t first = (begin + (uint64_t {1} << GranuleBits) - 1) >> GranuleBits;
		const uint64_t last  = std::min<uint64_t>(end >> GranuleBits, Granules); // (exclusive)
		for (uint64_t g = first; g < last; ++g) {
			if ((words[g / 64] & (uint64_t {1} << (g % 64))) == 0) bytes += uint64_t {1} << GranuleBits;
			words[g / 64] |= uint64_t {1} << (g % 64);
		}
	}
	for (uint64_t i = 0; i < Words; ++i) {
		if (s.gpu[i].load(std::memory_order_relaxed) != words[i]) s.gpu[i].store(words[i], std::memory_order_release);
	}
	s.gpu_bytes.store(bytes, std::memory_order_relaxed);
	s.publishes.fetch_add(1, std::memory_order_release);
}

// Whether all of [vaddr, vaddr + size) is memory of the engine's GPU allocators (as of the last walk).
inline bool IsGpuClass(uint64_t vaddr, uint64_t size) {
	if (size == 0 || vaddr > UINT64_MAX - size || vaddr + size > (uint64_t {1} << SpanBits)) return false;
	auto&          s     = Get();
	const uint64_t first = vaddr >> GranuleBits, last = (vaddr + size - 1) >> GranuleBits;
	for (uint64_t g = first; g <= last; ++g) {
		if ((s.gpu[g / 64].load(std::memory_order_acquire) & (uint64_t {1} << (g % 64))) == 0) return false;
	}
	return true;
}

// A readback window [begin, end) around a CPU access [address, address + size), inside a buffer [lo, hi): cut to the
// access's pages (or the configured window around them) when the access is to GPU-class memory. The result holds the
// access's pages and stays inside [lo, hi) and the old window's union with the access's pages. `page`: the trackers'
// page size (a power of two). True when it was cut.
inline bool Narrow(uint64_t address, uint64_t size, uint64_t lo, uint64_t hi, uint64_t page, uint64_t& begin, uint64_t& end) {
	if (size == 0 || begin >= end) return false;
	// KYTY_ARENA_NATIVE: the GPU frame arena (uploadHeap.h) on its own, with its own window.
	const bool gpu   = Enabled() && IsGpuClass(address, size);
	const bool arena = !gpu && Libs::Graphics::UploadHeap::InArena(address, size);
	if (!gpu && !arena) return false;
	const uint64_t window = std::max(gpu ? WindowBytes() : Libs::Graphics::UploadHeap::ArenaWindow, page);
	uint64_t       first  = std::max(address & ~(window - 1), lo);
	uint64_t       last   = std::min(((address + size + window - 1) & ~(window - 1)), hi);
	// (Never more than the old window, never less than the access's own pages.)
	first = std::max(first, std::min(begin, address & ~(page - 1)));
	last  = std::min(last, std::max(end, (address + size + page - 1) & ~(page - 1)));
	if (first >= last || (first <= begin && last >= end)) return false;
	auto& s = Get();
	s.narrowed.fetch_add(1, std::memory_order_relaxed);
	s.skipped_bytes.fetch_add((end - begin) - std::min(end - begin, last - first), std::memory_order_relaxed);
	begin = first;
	end   = last;
	return true;
}

inline void NoteWriteBack(const char* source, uint64_t bytes) {
	if (bytes == 0) return;
	const char c     = source != nullptr ? source[0] : '?';
	const int  index = c == 'r' ? 0 : c == 'd' ? 1 : c == 'f' ? 2 : c == 'i' ? 3 : 4;
	Get().written_back[index].fetch_add(bytes, std::memory_order_relaxed);
}

} // namespace GpuOnlyWrites
