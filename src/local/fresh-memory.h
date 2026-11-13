#pragma once
// Fresh memory (KYTY_FRESH_MEMORY=0 turns it off): the 64 KiB granules of the guest backing store nothing wrote since
// the emulator started, whose bytes are zeros. An upload of such bytes fills its staging with zeros instead of reading
// them: the read through the guest view made the pages resident (1-1 on a 16 GB PC: ~0.75 GB of zero pages under images
// only the GPU writes - render targets, a 180 MiB volume, video outs - read by the async upload worker when they were
// made; the working set 0.7-1.4 GB smaller at each of six worlds).
//
// Every write marks its granules: GPU-accessible guest memory the CPU may write is write-watched until its first write
// (GpuResourceManager: the fault marks it), writes into the backing view mark theirs (TryWriteBacking,
// TryWriteCpuBacking), host overwrites theirs (InvalidateMemoryForWrite), and memory the guest can write without being
// watched is marked when it is mapped so (CPU-only mappings, a protection change that lets the CPU write, a view mapped
// again by a partial unmap). Memory only the GPU may write needs no watch: its guest bytes change only through the
// backing view (write-backs). A granule once written stays written.

#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string_view>

namespace FreshMemory {

constexpr uint64_t GranuleBits = 16;
constexpr uint64_t Granule     = uint64_t {1} << GranuleBits;

inline bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_FRESH_MEMORY");
		return value == nullptr || std::string_view(value) != "0";
	}();
	return on;
}

struct State {
	std::atomic<uint64_t>* written = nullptr; // a bit per granule of the backing store
	std::atomic<uint64_t>* armed   = nullptr; // granules a guest mapping write-watches (GpuResourceManager)
	uint64_t               base    = 0, size = 0, granules = 0;
	// Counters (bytes): granules marked by a first write (fault), by other writes, uploads filled with zeros.
	std::atomic<uint64_t> faults {0}, marked {0}, zero_upload_bytes {0}, watched_bytes {0};
};

// Made at the first use once the backing store exists (the GPU side's first mapping comes after it).
inline State* Get() {
	static State                state;
	static std::atomic<bool>    ready {false};
	static std::mutex           mutex;
	if (ready.load(std::memory_order_acquire)) return state.written != nullptr ? &state : nullptr;
	if (!Enabled()) return nullptr;
	std::lock_guard lock(mutex);
	if (!ready.load(std::memory_order_relaxed)) {
		uint64_t base = 0, size = 0;
		if (!Libs::LibKernel::Memory::BackingView(&base, &size) || base == 0 || size == 0) return nullptr;
		state.base     = base;
		state.size     = size;
		state.granules = (size + Granule - 1) >> GranuleBits;
		state.written  = new std::atomic<uint64_t>[(state.granules + 63) / 64]();
		state.armed    = new std::atomic<uint64_t>[(state.granules + 63) / 64]();
		ready.store(true, std::memory_order_release);
		std::printf("Fresh memory: on (KYTY_FRESH_MEMORY=0 turns it off), %llu MiB of guest backing store in %llu KiB "
		            "granules\n",
		            static_cast<unsigned long long>(size >> 20u), static_cast<unsigned long long>(Granule >> 10u));
	}
	return &state;
}

// The granule of a backing-view address (UINT64_MAX: outside the backing store).
inline uint64_t GranuleOf(const State& state, uint64_t backing) {
	return backing >= state.base && backing - state.base < state.size ? (backing - state.base) >> GranuleBits : UINT64_MAX;
}

inline bool IsWritten(const State& state, uint64_t granule) {
	return granule >= state.granules ||
	       (state.written[granule >> 6u].load(std::memory_order_acquire) & (uint64_t {1} << (granule & 63u))) != 0;
}

// True when it was not written before.
inline bool MarkGranule(State& state, uint64_t granule) {
	if (granule >= state.granules) return false;
	const uint64_t bit = uint64_t {1} << (granule & 63u);
	return (state.written[granule >> 6u].fetch_or(bit, std::memory_order_acq_rel) & bit) == 0;
}

// The granules of backing-view bytes [backing, backing + size) written.
inline void MarkBacking(uint64_t backing, uint64_t size) {
	auto* state = Get();
	if (state == nullptr || size == 0) return;
	const auto first = GranuleOf(*state, backing), last = GranuleOf(*state, backing + size - 1);
	if (first == UINT64_MAX || last == UINT64_MAX) return;
	for (auto granule = first; granule <= last; ++granule) {
		if (MarkGranule(*state, granule)) state->marked.fetch_add(Granule, std::memory_order_relaxed);
	}
}

inline void SetArmed(State& state, uint64_t granule, bool armed) {
	if (granule >= state.granules) return;
	const uint64_t bit = uint64_t {1} << (granule & 63u);
	if (armed) state.armed[granule >> 6u].fetch_or(bit, std::memory_order_acq_rel);
	else state.armed[granule >> 6u].fetch_and(~bit, std::memory_order_acq_rel);
}

// A guest view of backing-view bytes [backing, backing + size) mapped again (a partial unmap of its view): the
// write-watched granules there are written, as a CPU write between the map and its protection would go unseen. Bits
// only (callable under the address-space lock); the watch itself goes at the next write fault.
inline void MarkArmedBacking(uint64_t backing, uint64_t size) {
	auto* state = Get();
	if (state == nullptr || size == 0) return;
	const auto first = GranuleOf(*state, backing), last = GranuleOf(*state, backing + size - 1);
	if (first == UINT64_MAX || last == UINT64_MAX) return;
	for (auto granule = first; granule <= last; ++granule) {
		const uint64_t bit = uint64_t {1} << (granule & 63u);
		if ((state->armed[granule >> 6u].load(std::memory_order_acquire) & bit) != 0 && MarkGranule(*state, granule))
			state->marked.fetch_add(Granule, std::memory_order_relaxed);
	}
}

// The granules of guest range [vaddr, vaddr + size) written (its bytes in the backing store; bits only).
inline void MarkGuestRange(uint64_t vaddr, uint64_t size) {
	if (Get() == nullptr || size == 0) return;
	for (uint64_t done = 0; done < size;) {
		const uint64_t at  = vaddr + done;
		const auto     end = Libs::LibKernel::Memory::BackingMappingEnd(at);
		if (end <= at) {
			done += std::min(size - done, Granule - (at & (Granule - 1)));
			continue;
		}
		const auto piece = std::min(end - at, size - done);
		if (const auto* backing = Libs::LibKernel::Memory::TryGetBackingPointer(at, piece))
			MarkBacking(reinterpret_cast<uint64_t>(backing), piece);
		done += piece;
	}
}

// The same for backing-view bytes [backing, backing + size) (one mapping's: bits only, no kernel query).
template <typename Run>
bool ForEachBackingRun(const uint8_t* backing, uint64_t size, Run&& run) {
	auto* state = Get();
	if (state == nullptr || size == 0 || backing == nullptr) return false;
	uint64_t run_begin = 0;
	bool     run_fresh = false;
	for (uint64_t offset = 0; offset < size;) {
		const auto address = reinterpret_cast<uint64_t>(backing) + offset;
		const auto granule = GranuleOf(*state, address);
		const auto bytes   = granule == UINT64_MAX
		                         ? size - offset
		                         : std::min(size - offset, Granule - ((address - state->base) & (Granule - 1)));
		const bool fresh   = granule != UINT64_MAX && !IsWritten(*state, granule);
		if (offset == 0) {
			run_fresh = fresh;
		} else if (fresh != run_fresh) {
			run(run_begin, offset - run_begin, run_fresh);
			run_begin = offset;
			run_fresh = fresh;
		}
		offset += bytes;
	}
	run(run_begin, size - run_begin, run_fresh);
	return true;
}

// Calls run(offset, size, fresh) for the runs of guest range [vaddr, vaddr + size) in order: fresh runs are bytes of
// granules nobody wrote; bytes outside the backing store are never fresh. False (nothing called) when tracking is off.
template <typename Run>
bool ForEachRun(uint64_t vaddr, uint64_t size, Run&& run) {
	auto* state = Get();
	if (state == nullptr || size == 0) return false;
	uint64_t   run_begin = 0, run_end = 0;
	bool       run_fresh = false;
	const auto add       = [&](uint64_t begin, uint64_t end, bool fresh) {
        if (run_end == run_begin) {
            run_begin = begin;
            run_fresh = fresh;
        } else if (fresh != run_fresh) {
            run(run_begin, run_end - run_begin, run_fresh);
            run_begin = begin;
            run_fresh = fresh;
        }
        run_end = end;
	};
	for (uint64_t done = 0; done < size;) {
		const uint64_t at      = vaddr + done;
		const auto     end     = Libs::LibKernel::Memory::BackingMappingEnd(at);
		const auto     piece   = end > at ? std::min(end - at, size - done) : size - done;
		const auto*    backing = end > at ? Libs::LibKernel::Memory::TryGetBackingPointer(at, piece) : nullptr;
		if (backing == nullptr) {
			add(done, done + piece, false);
			done += piece;
			continue;
		}
		for (uint64_t offset = 0; offset < piece;) {
			const auto address = reinterpret_cast<uint64_t>(backing) + offset;
			const auto granule = GranuleOf(*state, address);
			const auto bytes   = granule == UINT64_MAX
			                         ? piece - offset
			                         : std::min(piece - offset, Granule - ((address - state->base) & (Granule - 1)));
			add(done + offset, done + offset + bytes, granule != UINT64_MAX && !IsWritten(*state, granule));
			offset += bytes;
		}
		done += piece;
	}
	if (run_end != run_begin) run(run_begin, run_end - run_begin, run_fresh);
	return true;
}

} // namespace FreshMemory
