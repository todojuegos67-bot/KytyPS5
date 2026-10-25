#pragma once
// Texture offload (Windows). The guest-memory copy of a texture GPU work never wrote, that only shaders sample, uploaded
// and clean for a while, duplicates its image in video memory: on a PS5 the game's memory holds it once.
//
// KYTY_TEXTURE_OFFLOAD_PROBE=1 (diagnostics): such a texture's pages are made inaccessible in both views of guest memory
// (the guest's mapping: read watchers of the page manager; the backing view the emulator reads and writes: no access) and
// nothing else. Every later touch faults, is counted with who made it and how long after, and gives the range its
// protection back for good: what keeping the texture only in video memory would cost.
//
// KYTY_TEXTURE_OFFLOAD=1 (default with 20 GiB of RAM or less): the same for textures whose download was checked to give
// their guest bytes back exactly (16 MiB of images a scan: the downloads go with the frame's GPU work), and their pages
// are dropped by the offload worker (DiscardVirtualMemory in the backing view a MiB at a time, the guest view's pages
// trimmed: the RAM given back, no pagefile write). On a 16 GB PC with a 1-1 -> Stonefang -> 1-1 route: up to 2 GB of
// guest memory out of RAM. Whoever needs the bytes again gets them back first, downloaded from the image by the render
// thread (TextureCache::RestoreOffloaded, waiting for the GPU): an upload of an image over them (InitializeImage), a host
// write over part of them (InvalidateMemory), the game's code touching them (its fault: the render thread through
// SendCommandSync); the image going (UnregisterImage: the collector retires images no draw used for 20 s) has them
// written back by the offload worker when the GPU is done, touches waiting till then. A host overwrite of all of them or
// an unmap only drops them. A touch by a thread that cannot wait for the render thread (the upload worker, deferred GPU
// work, a speculation's recorder) is a miss: counted and logged, the bytes it sees zeros, and the render thread restores
// them after (Drain).
//
// Read watchers go at once after a touch by game code (a guest thread holds no lock of the emulator's) or a host write or
// unmap (normal context); after one by the emulator's own code the fault handler only gives the pages a protection that
// lets the access through, and the render thread removes them (Drain).
//
// KYTY_TEXTURE_SPILL=1 (default in the low-RAM profile, with the offload): the levels KYTY_TEXTURE_REDUCE leaves out of a
// large 2D texture (descriptors.cpp HoldFromBaseLevel: the image holds only the first bytes of the texture's memory, its
// smaller levels) are bytes the emulator never reads again: no image holds them, nothing uploads them, and the offload
// cannot drop them (no image to download them from). With one level left out they are 3/4 of each such texture (at 1-1
// its BC textures took 3.7 GiB whole, 0.9 GiB reduced). Their whole pages (a "tail" range: no image over them, nothing
// the GPU wrote) are taken as the offload takes a texture (read watchers on the guest view), written to a scratch file
// by the offload worker (the store alias read-only meanwhile: a write there is a touch), dropped from RAM
// (DiscardVirtualMemory) and the alias made inaccessible. A touch reads them back from the file on the thread that
// touched (any thread: no GPU work, so no misses), under the offload's lock, then goes on as the offload's touch; an
// upload of an image over them, a host write over part of them likewise; an unmap or a host write over all of them only
// drops them. Never the page file and no working set trim of pages that hold data (the 0x1A bug checks), and no
// guess: the bytes come back as they were. KYTY_TEXTURE_SPILL_MB (default 64): the tails taken a scan (every 2 s).

#include "guest-lazy-commit.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "local-platform.h"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace TextureOffload {

enum class Mode { Off, Probe, Offload };

// KYTY_TEXTURE_OFFLOAD=1|0 (default: on with 20 GiB of RAM or less, where the game's memory and the emulator's caches
// press on it; with more, its restores' stalls buy nothing).
inline Mode GetMode() {
#if defined(_WIN32)
	static const Mode mode = [] {
		const char* offload = std::getenv("KYTY_TEXTURE_OFFLOAD");
		if (offload != nullptr && offload[0] != '\0') {
			if (offload[0] == '1') return Mode::Offload;
		} else if (const auto ram = LocalPlatform::PhysicalMemory(); ram != 0 && ram <= (uint64_t {20} << 30u)) {
			return Mode::Offload;
		}
		const char* probe = std::getenv("KYTY_TEXTURE_OFFLOAD_PROBE");
		return probe != nullptr && probe[0] == '1' ? Mode::Probe : Mode::Off;
	}();
	return mode;
#else
	return Mode::Off;
#endif
}

inline bool Enabled() {
	return GetMode() != Mode::Off;
}

enum class Touch : uint8_t {
	GameRead,
	GameWrite,
	EmulatorGuestRead,
	EmulatorGuestWrite,
	BackingRead,
	BackingWrite,
	HostWrite,
	HostOverwrite,
	Unmap,
	Upload,
	ImageGone,
	SpillFailed,
	Count
};

inline const char* TouchName(Touch touch) {
	static const char* names[] = {"game read",
	                              "game write",
	                              "emulator read (guest view)",
	                              "emulator write (guest view)",
	                              "emulator read (backing view)",
	                              "emulator write (backing view)",
	                              "host write over part",
	                              "host write over all",
	                              "unmap",
	                              "image upload over it",
	                              "image gone",
	                              "spill file write failed"};
	return names[static_cast<int>(touch)];
}

struct Piece {
	uint64_t backing = 0, size = 0;
};

struct Range {
	uint64_t                              guest_end = 0;
	std::vector<Piece>                    pieces;
	uint64_t                              serial = 0, image = 0; // (the image's serial and packed id)
	std::chrono::steady_clock::time_point since;
	bool                                  discarded = false;
	bool                                  touched   = false; // (protection given back; its read watchers go: Drain)
	bool                                  missed    = false; // (discarded and touched by a thread that could not wait)
	bool                                  restoring = false; // (its restore runs: other touches wait for its end)
	std::thread::id                       restorer {};       // (the thread restoring it; none: the offload worker)
	uint64_t                              spill   = ~uint64_t {0}; // (KYTY_TEXTURE_SPILL: a tail range's file offset)
	bool                                  spilled = false;         // (its bytes in the file and dropped from RAM)
	[[nodiscard]] bool                    IsTail() const { return spill != ~uint64_t {0}; }
};

// Set by the texture cache: the page manager's read watchers of a guest range removed; a restore of the discarded ranges
// over a guest range by the render thread (any thread CanWait allows: it returns when they are back); whether the calling
// thread may wait for the render thread.
using RemoveWatchers = void (*)(uint64_t vaddr, uint64_t size);
using Restore        = void (*)(uint64_t vaddr, uint64_t size, Touch touch);
using CanWait        = bool (*)();

struct State {
	std::mutex                   mutex;
	std::map<uint64_t, Range>    guest;   // by guest start
	std::map<uint64_t, uint64_t> backing; // backing piece start -> guest start
	std::vector<uint64_t>        pending; // touched ranges (guest starts) whose read watchers go (Drain)
	std::unordered_set<uint64_t> touched_serials; // images touched once: not taken again
	std::atomic<RemoveWatchers>  remove {nullptr};
	std::atomic<Restore>         restore {nullptr};
	std::atomic<CanWait>         can_wait {nullptr};
	std::atomic<uint64_t>        guest_lo {~uint64_t {0}}, guest_hi {0}, backing_lo {~uint64_t {0}}, backing_hi {0};
	// Statistics.
	uint64_t now = 0, peak = 0, total = 0, ranges_total = 0;
	uint64_t touches[static_cast<int>(Touch::Count)] {}, touch_bytes[static_cast<int>(Touch::Count)] {};
	uint64_t ages[4] {}; // <1 s, <10 s, <60 s, later
	uint64_t restores = 0, restore_bytes = 0, restore_us = 0, restore_us_max = 0, lost = 0, missed = 0, missed_bytes = 0;
	uint64_t checked = 0, checked_bytes = 0, check_failed = 0;
	// Ranges being restored (any thread), and buffer uploads that waited for an asynchronous restore (WaitBeforeRead).
	std::atomic<uint64_t> restoring_ranges {0}, read_waits {0}, read_wait_bytes {0};
	std::unordered_map<uint64_t, uint64_t> rips;    // the emulator's instructions -> touches
	std::unordered_map<uint64_t, uint64_t> callers; // their callers (one frame unwound) -> touches
	// KYTY_TEXTURE_SPILL: the scratch file, its next free offset and freed blocks (offset, size), and statistics.
	uint64_t                                   spill_file = 0, spill_end = 0;
	std::vector<std::pair<uint64_t, uint64_t>> spill_free;
	uint64_t tails_now = 0, tails_peak = 0, tails_taken = 0, tails_taken_bytes = 0, tails_written = 0, tails_written_bytes = 0;
	uint64_t tails_write_us = 0, tails_restores = 0, tails_restore_bytes = 0, tails_restore_us = 0, tails_restore_us_max = 0;
	uint64_t tails_lost = 0, tails_dropped = 0, tails_open_failed = 0;
};

inline State& Get() {
	static State state;
	return state;
}

// KYTY_TEXTURE_SPILL=1 (see above; with the offload only).
inline bool SpillEnabled() {
#if defined(_WIN32)
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TEXTURE_SPILL");
		return value != nullptr && value[0] == '1' && GetMode() == Mode::Offload;
	}();
	return on;
#else
	return false;
#endif
}
// A tail range's serial: its image's with this bit (never an image's own).
constexpr uint64_t TailSerialBit = uint64_t {1} << 63u;
constexpr uint64_t SpillBlock    = uint64_t {64} << 10u;

// (state.mutex held) A file block for `size` bytes: a freed one that fits (first fit), else the file's end.
inline uint64_t AllocateSpillLocked(State& state, uint64_t size) {
	size = (size + SpillBlock - 1) & ~(SpillBlock - 1);
	for (auto it = state.spill_free.begin(); it != state.spill_free.end(); ++it) {
		if (it->second < size) continue;
		const uint64_t offset = it->first;
		if (it->second == size) state.spill_free.erase(it);
		else *it = {it->first + size, it->second - size};
		return offset;
	}
	const uint64_t offset = state.spill_end;
	state.spill_end += size;
	return offset;
}

// (state.mutex held) A tail range leaving the maps: its file block free again.
inline void ReleaseSpillLocked(State& state, const Range& range, uint64_t size) {
	if (!range.IsTail()) return;
	state.spill_free.emplace_back(range.spill, (size + SpillBlock - 1) & ~(SpillBlock - 1));
}

#if defined(_WIN32)
inline bool Protect(uint64_t address, uint64_t size, DWORD protection) {
	DWORD old = 0;
	return GuestLazyCommit::Protect(reinterpret_cast<void*>(address), size, protection, &old); // (reserved pages first)
}

inline bool IsHostCode(uint64_t rip) {
	HMODULE module = nullptr;
	return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                          reinterpret_cast<LPCWSTR>(rip), &module) != 0;
}

// The return address of the faulting function (one frame unwound from the fault's context; a leaf's at rsp).
inline uint64_t CallerOf(const void* native_context) {
	CONTEXT context    = *static_cast<const CONTEXT*>(native_context);
	DWORD64 image_base = 0;
	auto*   function   = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
	if (function == nullptr) return *reinterpret_cast<const DWORD64*>(context.Rsp);
	void*   handler_data = nullptr;
	DWORD64 establisher  = 0;
	RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, function, &context, &handler_data, &establisher, nullptr);
	return context.Rip;
}
#endif

// A guest range [vaddr, vaddr + size) taken, with its backing pieces and image (texture cache, render thread; the
// caller protects, and discards, after).
inline void Add(uint64_t vaddr, uint64_t size, std::vector<Piece> pieces, uint64_t serial, uint64_t image, bool discarded,
                bool tail = false) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	Range           range;
	if (tail) {
		range.spill = AllocateSpillLocked(state, size);
		state.tails_taken++;
		state.tails_taken_bytes += size;
	}
	range.guest_end = vaddr + size;
	range.pieces    = std::move(pieces);
	range.serial    = serial;
	range.image     = image;
	range.since     = std::chrono::steady_clock::now();
	range.discarded = discarded;
	for (const auto& piece: range.pieces) {
		state.backing[piece.backing] = vaddr;
		if (piece.backing < state.backing_lo.load(std::memory_order_relaxed)) state.backing_lo.store(piece.backing, std::memory_order_relaxed);
		if (piece.backing + piece.size > state.backing_hi.load(std::memory_order_relaxed))
			state.backing_hi.store(piece.backing + piece.size, std::memory_order_relaxed);
	}
	if (vaddr < state.guest_lo.load(std::memory_order_relaxed)) state.guest_lo.store(vaddr, std::memory_order_relaxed);
	if (vaddr + size > state.guest_hi.load(std::memory_order_relaxed)) state.guest_hi.store(vaddr + size, std::memory_order_relaxed);
	const bool is_tail = range.IsTail();
	state.guest.emplace(vaddr, std::move(range));
	if (is_tail) return; // (counted when written: SpillRange)
	state.now += size;
	state.peak = std::max(state.peak, state.now);
	state.total += size;
	state.ranges_total++;
}

#if defined(_WIN32)
// The worker thread's drop of range `begin`'s pages (discarded in the backing view, then no access there): unless it was
// touched, restored or dropped since it was taken. Under the state's lock: no touch or restore of it meanwhile.
inline bool DiscardRange(uint64_t begin) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	const auto      it = state.guest.find(begin);
	if (it == state.guest.end() || it->second.touched || it->second.restoring || !it->second.discarded) return false;
	// A MiB at a time: a discard holds the process's working set lock over its pages, which every page fault and
	// protection change of the game's threads waits for (whole ranges: frames of 30-60 ms after each scan).
	constexpr uint64_t Chunk = uint64_t {1} << 20u;
	for (const auto& piece: it->second.pieces) {
		// (A discard is refused on pages without access.)
		for (uint64_t at = 0; at < piece.size; at += Chunk) {
			(void)DiscardVirtualMemory(reinterpret_cast<void*>(piece.backing + at), std::min(Chunk, piece.size - at));
			SwitchToThread();
		}
		(void)Protect(piece.backing, piece.size, PAGE_NOACCESS);
	}
	return true;
}

#endif

// Not to be taken: a range of it is taken, or the image was touched once.
[[nodiscard]] inline bool Taken(uint64_t vaddr, uint64_t size, uint64_t serial) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	if (state.touched_serials.count(serial) != 0) return true;
	auto it = state.guest.upper_bound(vaddr);
	if (it != state.guest.begin() && std::prev(it)->second.guest_end > vaddr) return true;
	return it != state.guest.end() && it->first < vaddr + size;
}

// (state.mutex held) The first range over [vaddr, ...), or end.
inline std::map<uint64_t, Range>::iterator FirstOver(State& state, uint64_t vaddr) {
	auto it = state.guest.upper_bound(vaddr);
	if (it != state.guest.begin() && std::prev(it)->second.guest_end > vaddr) --it;
	return it;
}

// A discarded range not yet given back over [vaddr, vaddr + size) (cheap when none is near).
[[nodiscard]] inline bool Overlaps(uint64_t vaddr, uint64_t size) {
	auto& state = Get();
	if (size == 0 || vaddr + size <= state.guest_lo.load(std::memory_order_relaxed) || vaddr >= state.guest_hi.load(std::memory_order_relaxed))
		return false;
	std::lock_guard lock(state.mutex);
	for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + size; ++it)
		if (it->second.discarded && !it->second.touched) return true;
	return false;
}

// (state.mutex held) A restore by another thread (or the offload worker) runs over [vaddr, vaddr + size).
inline bool RestoringLocked(State& state, uint64_t vaddr, uint64_t size) {
	for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + size; ++it)
		if (it->second.restoring && it->second.restorer != std::this_thread::get_id()) return true;
	return false;
}

// Whether a restore by another thread (or the offload worker) runs over [vaddr, vaddr + size).
[[nodiscard]] inline bool IsRestoring(uint64_t vaddr, uint64_t size) {
	auto& state = Get();
	if (size == 0 || vaddr + size <= state.guest_lo.load(std::memory_order_relaxed) || vaddr >= state.guest_hi.load(std::memory_order_relaxed))
		return false;
	std::lock_guard lock(state.mutex);
	return RestoringLocked(state, vaddr, size);
}

// Until no other thread (nor the offload worker) restores a range over [vaddr, vaddr + size).
inline void WaitRestores(uint64_t vaddr, uint64_t size) {
	while (IsRestoring(vaddr, size)) std::this_thread::yield();
}

// Whether a read of [vaddr, vaddr + size) through the backing view must wait for a restore first (cheap when none
// runs). An image gone (UnregisterImage) is restored asynchronously: its backing pieces are given read-write access at
// once and its bytes land when the GPU's download is done, so a reader of the backing view in between (a buffer upload,
// the async upload worker) read the discarded pages: zeros. At a death and reload under a small video memory budget
// the collector takes hundreds of offloaded textures at once and the reload reads their memory again: dark textures.
[[nodiscard]] inline bool MustWaitBeforeRead(uint64_t vaddr, uint64_t size) {
	auto& state = Get();
	return state.restoring_ranges.load(std::memory_order_acquire) != 0 && IsRestoring(vaddr, size);
}
inline void NoteReadWait(uint64_t size) {
	auto& state = Get();
	state.read_waits.fetch_add(1, std::memory_order_relaxed);
	state.read_wait_bytes.fetch_add(size, std::memory_order_relaxed);
}

// (state.mutex held) Counted as touched by `touch`.
inline void CountLocked(State& state, std::map<uint64_t, Range>::iterator it, Touch touch, uint64_t rip = 0, uint64_t caller = 0) {
	auto&      range = it->second;
	const auto size  = range.guest_end - it->first;
	state.touches[static_cast<int>(touch)]++;
	state.touch_bytes[static_cast<int>(touch)] += size;
	const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - range.since).count();
	state.ages[seconds < 1 ? 0 : seconds < 10 ? 1 : seconds < 60 ? 2 : 3]++;
	if (rip != 0) state.rips[rip]++;
	if (caller != 0) state.callers[caller]++;
	state.touched_serials.insert(range.serial);
	if (range.IsTail()) {
		// (A tail touched while dropped: its restore came first, or an unmap or a whole overwrite drops it.)
		if (range.spilled) state.tails_now -= std::min(state.tails_now, size);
		range.spilled = false;
		return;
	}
	state.now -= std::min(state.now, size);
}

// (state.mutex held) The touch of range `it`: counted, its views given a protection that lets accesses through (the
// guest's read-only: its write watchers stay) and queued for its read watchers to go.
inline void TouchLocked(State& state, std::map<uint64_t, Range>::iterator it, Touch touch, uint64_t rip = 0, uint64_t caller = 0) {
	auto& range = it->second;
	if (range.touched) return;
	range.touched = true;
	CountLocked(state, it, touch, rip, caller);
#if defined(_WIN32)
	for (const auto& piece: range.pieces) (void)Protect(piece.backing, piece.size, PAGE_READWRITE);
	(void)Protect(it->first, range.guest_end - it->first, PAGE_READONLY);
#endif
	state.pending.push_back(it->first);
}

#if defined(_WIN32)
// (state.mutex held) A tail range's bytes back in RAM (KYTY_TEXTURE_SPILL): the alias writable, the file's bytes read
// into it when they were dropped. After it the range is an untouched offload range that holds its bytes: the caller
// touches it (TouchLocked) as any other.
inline void RestoreSpillLocked(State& state, Range& range, uint64_t size) {
	if (!range.IsTail() || !range.discarded) return;
	for (const auto& piece: range.pieces) (void)Protect(piece.backing, piece.size, PAGE_READWRITE);
	if (range.spilled) {
		const auto start  = std::chrono::steady_clock::now();
		uint64_t   offset = range.spill;
		bool       ok     = true;
		for (const auto& piece: range.pieces) {
			ok = LocalPlatform::ReadScratchFile(state.spill_file, offset, reinterpret_cast<void*>(piece.backing), piece.size) && ok;
			offset += piece.size;
		}
		const auto us = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
		state.tails_restores++;
		state.tails_restore_bytes += size;
		state.tails_restore_us += us;
		state.tails_restore_us_max = std::max(state.tails_restore_us_max, us);
		state.tails_now -= std::min(state.tails_now, size);
		if (!ok) {
			state.tails_lost++;
			std::printf("Texture tails: LOST 0x%llx+0x%llx (the spill file could not be read back: zeros)\n",
			            static_cast<unsigned long long>(range.pieces.empty() ? 0 : range.pieces.front().backing),
			            static_cast<unsigned long long>(size));
		}
		range.spilled = false;
	}
	range.discarded = false;
}

// The offload worker's spill of tail range `begin` (KYTY_TEXTURE_SPILL): its bytes written to the scratch file with the
// alias read-only (a write through it faults: a touch, and the range is left alone), then, unless it was touched
// meanwhile, dropped (DiscardVirtualMemory a MiB at a time) and the alias made inaccessible. True when dropped (the
// caller trims the guest view's pages, as after DiscardRange).
inline bool SpillRange(uint64_t begin) {
	auto&              state = Get();
	std::vector<Piece> pieces;
	uint64_t           offset = 0, size = 0;
	{
		std::lock_guard lock(state.mutex);
		const auto      it = state.guest.find(begin);
		if (it == state.guest.end() || !it->second.IsTail() || it->second.touched || !it->second.discarded || it->second.spilled)
			return false;
		if (state.spill_file == 0) {
			state.spill_file = LocalPlatform::OpenScratchFile();
			if (state.spill_file == 0) {
				if (state.tails_open_failed++ == 0) std::printf("Texture tails: no scratch file could be made: KYTY_TEXTURE_SPILL does nothing\n");
				return false;
			}
		}
		pieces = it->second.pieces;
		offset = it->second.spill;
		size   = it->second.guest_end - begin;
		for (const auto& piece: pieces) (void)Protect(piece.backing, piece.size, PAGE_READONLY);
	}
	const auto start = std::chrono::steady_clock::now();
	bool       ok    = true;
	uint64_t   at    = offset;
	for (const auto& piece: pieces) {
		ok = ok && LocalPlatform::WriteScratchFile(state.spill_file, at, reinterpret_cast<const void*>(piece.backing), piece.size);
		at += piece.size;
	}
	const auto us = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
	std::lock_guard lock(state.mutex);
	const auto      it = state.guest.find(begin);
	// (Touched meanwhile: the touch gave its protections back and the range goes with its watchers: Drain.)
	if (it == state.guest.end() || it->second.touched) {
		state.tails_dropped++;
		return false;
	}
	auto& range = it->second;
	if (!ok) {
		// (Nothing dropped: its bytes are all there. Its watchers go as a touch's.)
		range.discarded = false;
		TouchLocked(state, it, Touch::SpillFailed);
		return false;
	}
	constexpr uint64_t Chunk = uint64_t {1} << 20u;
	for (const auto& piece: pieces) {
		for (uint64_t done = 0; done < piece.size; done += Chunk) {
			const auto bytes = std::min(Chunk, piece.size - done);
			auto*      where = reinterpret_cast<void*>(piece.backing + done);
			if (DiscardVirtualMemory(where, bytes) != ERROR_SUCCESS) {
				// (Refused on read-only pages: writable for the call, under the lock: a write meanwhile is the emulator's
				// own through the alias, which the read-only pages made fault until now.)
				(void)Protect(piece.backing + done, bytes, PAGE_READWRITE);
				(void)DiscardVirtualMemory(where, bytes);
			}
		}
		(void)Protect(piece.backing, piece.size, PAGE_NOACCESS);
	}
	range.spilled = true;
	state.tails_now += size;
	state.tails_peak = std::max(state.tails_peak, state.tails_now);
	state.tails_written++;
	state.tails_written_bytes += size;
	state.tails_write_us += us;
	return true;
}
#endif

// (state.mutex held) The queued ranges out of the maps (missed discarded ones stay for their restore: Drain): their guest
// ranges, for the caller to remove the watchers of.
inline std::vector<std::pair<uint64_t, uint64_t>> TakePendingLocked(State& state) {
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	std::vector<uint64_t>                      kept;
	for (const auto begin: state.pending) {
		auto range = state.guest.find(begin);
		if (range == state.guest.end()) continue;
		if (range->second.missed) {
			kept.push_back(begin);
			continue;
		}
		ranges.emplace_back(begin, range->second.guest_end - begin);
		for (const auto& piece: range->second.pieces) state.backing.erase(piece.backing);
		ReleaseSpillLocked(state, range->second, range->second.guest_end - begin);
		state.guest.erase(range);
	}
	state.pending.swap(kept);
	return ranges;
}

inline void RemoveAll(State& state, const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	const auto remove = state.remove.load(std::memory_order_acquire);
	if (remove == nullptr) return;
	for (const auto& [begin, size]: ranges) remove(begin, size);
}

// A restore's ranges (TextureCache::RestoreOffloaded, render thread): the discarded ranges over [vaddr, vaddr + size) not
// given back yet (and the missed ones: `late`), marked as restored by this thread and counted as touched by `touch`.
struct Restoring {
	uint64_t           begin = 0, end = 0, serial = 0, image = 0;
	std::vector<Piece> pieces;
};
// `async`: the bytes are written back by the offload worker (EndRestore there).
inline std::vector<Restoring> BeginRestore(uint64_t vaddr, uint64_t size, Touch touch, bool late = false, bool async = false) {
	auto&                  state = Get();
	std::vector<Restoring> ranges;
	std::lock_guard        lock(state.mutex);
	for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + size; ++it) {
		auto& range = it->second;
		// (Tails come back from their file first: RestoreSpilled.)
		if (range.IsTail() || !range.discarded || range.restoring || (range.touched && !(late && range.missed))) continue;
		if (!range.touched && touch != Touch::Count) CountLocked(state, it, touch);
		range.touched   = true;
		range.restoring = true;
		range.restorer  = async ? std::thread::id {} : std::this_thread::get_id();
		state.restoring_ranges.fetch_add(1, std::memory_order_acq_rel);
		ranges.push_back({it->first, range.guest_end, range.serial, range.image, range.pieces});
	}
	return ranges;
}

// The restored ranges out of the maps (their bytes back and read watchers removed: other touches go on).
inline void EndRestore(const std::vector<Restoring>& ranges, uint64_t bytes, uint64_t us, uint64_t lost) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	for (const auto& restored: ranges) {
		auto range = state.guest.find(restored.begin);
		if (range == state.guest.end()) continue;
		if (range->second.restoring) state.restoring_ranges.fetch_sub(1, std::memory_order_acq_rel);
		for (const auto& piece: range->second.pieces) state.backing.erase(piece.backing);
		ReleaseSpillLocked(state, range->second, range->second.guest_end - restored.begin);
		state.guest.erase(range);
	}
	std::erase_if(state.pending, [&](uint64_t begin) { return state.guest.find(begin) == state.guest.end(); });
	state.restores++;
	state.restore_bytes += bytes;
	state.restore_us += us;
	state.restore_us_max = std::max(state.restore_us_max, us);
	state.lost += lost;
}

#if defined(_WIN32)
// The tails over [vaddr, vaddr + size) back from their file and touched (TextureCache::RestoreOffloaded, first: an image
// upload over them, a restore's caller); their read watchers go at once (the render thread).
inline void RestoreSpilled(uint64_t vaddr, uint64_t size, Touch touch) {
	auto& state = Get();
	if (size == 0 || vaddr + size <= state.guest_lo.load(std::memory_order_relaxed) || vaddr >= state.guest_hi.load(std::memory_order_relaxed))
		return;
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	{
		std::lock_guard lock(state.mutex);
		bool            any = false;
		for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + size; ++it) {
			if (!it->second.IsTail() || it->second.touched) continue;
			RestoreSpillLocked(state, it->second, it->second.guest_end - it->first);
			TouchLocked(state, it, touch == Touch::Count ? Touch::Upload : touch);
			any = true;
		}
		if (!any) return;
		ranges = TakePendingLocked(state);
	}
	RemoveAll(state, ranges);
}
#endif

inline void NoteCheck(uint64_t images, uint64_t bytes, uint64_t failed) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	state.checked += images;
	state.checked_bytes += bytes;
	state.check_failed += failed;
}

#if defined(_WIN32)
// The exception filter's probe handler (any thread): a touch of a taken range, either view.
inline bool OnFault(uint64_t address, uint64_t rip, const void* native_context, bool write) {
	auto&      state      = Get();
	const bool in_guest   = address >= state.guest_lo.load(std::memory_order_relaxed) && address < state.guest_hi.load(std::memory_order_relaxed);
	const bool in_backing = address >= state.backing_lo.load(std::memory_order_relaxed) && address < state.backing_hi.load(std::memory_order_relaxed);
	if (!in_guest && !in_backing) return false;
	// A fault raised while its range was still taken, served after another thread's touch gave the pages back (and the
	// range left the maps): the access goes again. (Only this protects the backing view.)
	const auto stale = [address] {
		MEMORY_BASIC_INFORMATION info {};
		return VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) == sizeof(info) && info.Protect == PAGE_READWRITE;
	};
	std::vector<std::pair<uint64_t, uint64_t>> now;
	uint64_t restore_begin = 0, restore_size = 0, wait_begin = 0, wait_size = 0;
	Touch    restore_touch = Touch::GameRead;
	{
		std::unique_lock lock(state.mutex);
		std::map<uint64_t, Range>::iterator it = state.guest.end();
		bool                                backing_hit = false;
		if (in_guest) {
			auto found = state.guest.upper_bound(address);
			if (found != state.guest.begin() && address < std::prev(found)->second.guest_end) it = std::prev(found);
		}
		if (it == state.guest.end() && in_backing) {
			auto piece = state.backing.upper_bound(address);
			if (piece == state.backing.begin()) return stale();
			--piece;
			auto range = state.guest.find(piece->second);
			if (range == state.guest.end()) return stale();
			const bool inside = std::any_of(range->second.pieces.begin(), range->second.pieces.end(), [&](const Piece& p) {
				return address >= p.backing && address < p.backing + p.size;
			});
			if (!inside) return stale();
			it          = range;
			backing_hit = true;
		}
		if (it == state.guest.end()) return false;
		auto&      range = it->second;
		const bool host  = backing_hit || IsHostCode(rip);
		if (range.restoring && range.restorer != std::this_thread::get_id()) {
			// Being restored (the render thread or the offload worker): the access goes again after.
			wait_begin = it->first;
			wait_size  = range.guest_end - it->first;
		} else if (range.touched) {
			// (Its protection came back before its watchers went, or this thread restores it.)
			if (host) {
				if (backing_hit) {
					for (const auto& piece: range.pieces) (void)Protect(piece.backing, piece.size, PAGE_READWRITE);
				} else {
					(void)Protect(it->first, range.guest_end - it->first, PAGE_READONLY);
				}
				return true;
			}
			const auto can_wait = state.can_wait.load(std::memory_order_acquire);
			if (range.missed && state.restore.load(std::memory_order_acquire) != nullptr && can_wait != nullptr && can_wait()) {
				// (A missed one keeps its watchers until its restore: it comes back now.)
				restore_begin = it->first;
				restore_size  = range.guest_end - it->first;
				restore_touch = Touch::Count;
			} else {
				now = TakePendingLocked(state);
			}
		} else {
			// A tail's bytes back from its file on this thread (no GPU work: any thread may), then a touch as any other's.
			if (range.IsTail()) RestoreSpillLocked(state, range, range.guest_end - it->first);
			const Touch touch = backing_hit ? (write ? Touch::BackingWrite : Touch::BackingRead)
			                    : host      ? (write ? Touch::EmulatorGuestWrite : Touch::EmulatorGuestRead)
			                                : (write ? Touch::GameWrite : Touch::GameRead);
			const auto  can_wait = state.can_wait.load(std::memory_order_acquire);
			if (range.discarded && state.restore.load(std::memory_order_acquire) != nullptr && can_wait != nullptr && can_wait()) {
				restore_begin = it->first;
				restore_size  = range.guest_end - it->first;
				restore_touch = touch;
			} else {
				if (range.discarded) {
					range.missed = true;
					state.missed++;
					state.missed_bytes += range.guest_end - it->first;
					std::printf("Texture offload: MISSED %s of 0x%llx..0x%llx (zeros until restored) at rip 0x%llx\n", TouchName(touch),
					            static_cast<unsigned long long>(it->first), static_cast<unsigned long long>(range.guest_end),
					            static_cast<unsigned long long>(rip));
				}
				TouchLocked(state, it, touch, host ? rip : 0, host ? CallerOf(native_context) : 0);
				if (host) return true;
				now = TakePendingLocked(state);
			}
		}
	}
	if (wait_size != 0) {
		WaitRestores(wait_begin, wait_size);
		return true;
	}
	if (restore_size != 0) {
		state.restore.load(std::memory_order_acquire)(restore_begin, restore_size, restore_touch);
		return true;
	}
	RemoveAll(state, now); // (a game thread: no lock of the emulator's held)
	return true;
}
#endif

// A host write (InvalidateMemory: file reads, network; `over_all`: it writes all of [vaddr, vaddr + size)) or an unmap
// over [vaddr, vaddr + size): the taken ranges there touched. A discarded one a write covers part of comes back first
// (the restore); the others are dropped. Normal context: their read watchers go at once.
inline void OnHostAccess(uint64_t vaddr, uint64_t size, bool unmap, bool over_all) {
	if (!Enabled() || size == 0) return;
	auto& state = Get();
	if (vaddr + size <= state.guest_lo.load(std::memory_order_relaxed) || vaddr >= state.guest_hi.load(std::memory_order_relaxed)) return;
	WaitRestores(vaddr, size);
	std::vector<std::pair<uint64_t, uint64_t>> ranges, restore;
	{
		std::lock_guard lock(state.mutex);
		for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + size; ++it) {
			if (it->second.restoring) continue; // (this thread's own restore)
			const bool covered = over_all && it->first >= vaddr && it->second.guest_end <= vaddr + size;
			if (it->second.touched) {
				// A missed one: dropped with the memory, or written over whole.
				if (it->second.missed && (unmap || covered)) {
					it->second.missed = false;
					state.pending.push_back(it->first);
				}
				continue;
			}
			if (!unmap && !covered && it->second.discarded) {
#if defined(_WIN32)
				if (it->second.IsTail()) {
					RestoreSpillLocked(state, it->second, it->second.guest_end - it->first);
					TouchLocked(state, it, Touch::HostWrite);
					continue;
				}
#endif
				restore.emplace_back(it->first, it->second.guest_end - it->first);
				continue;
			}
			TouchLocked(state, it, unmap ? Touch::Unmap : covered ? Touch::HostOverwrite : Touch::HostWrite);
		}
		ranges = TakePendingLocked(state);
	}
	RemoveAll(state, ranges);
	const auto restore_fn = state.restore.load(std::memory_order_acquire);
	const auto can_wait   = state.can_wait.load(std::memory_order_acquire);
	for (const auto& [begin, bytes]: restore) {
		if (restore_fn != nullptr && can_wait != nullptr && can_wait()) {
			restore_fn(begin, bytes, Touch::HostWrite);
			continue;
		}
		std::lock_guard lock(state.mutex);
		auto            it = state.guest.find(begin);
		if (it == state.guest.end() || it->second.touched) continue;
		it->second.missed = true;
		state.missed++;
		state.missed_bytes += bytes;
		std::printf("Texture offload: MISSED host write over part of 0x%llx..0x%llx (zeros until restored)\n",
		            static_cast<unsigned long long>(begin), static_cast<unsigned long long>(begin + bytes));
		TouchLocked(state, it, Touch::HostWrite);
	}
}

// Render thread: the read watchers of ranges the emulator's code touched go; missed discarded ranges come back first
// (`restore` with Touch::Count: a late restore, on the render thread itself).
inline void Drain() {
	auto&                                      state = Get();
	std::vector<std::pair<uint64_t, uint64_t>> ranges, missed;
	{
		std::lock_guard lock(state.mutex);
		if (state.pending.empty()) return;
		for (const auto begin: state.pending) {
			auto it = state.guest.find(begin);
			if (it != state.guest.end() && it->second.missed && !it->second.restoring)
				missed.emplace_back(begin, it->second.guest_end - begin);
		}
		ranges = TakePendingLocked(state);
	}
	RemoveAll(state, ranges);
	const auto restore = state.restore.load(std::memory_order_acquire);
	for (const auto& [begin, size]: missed)
		if (restore != nullptr) restore(begin, size, Touch::Count);
}

inline void Report(bool force) {
	auto&       state = Get();
	static auto last  = std::chrono::steady_clock::now();
	const auto  time  = std::chrono::steady_clock::now();
	if (!force && time - last < std::chrono::seconds(30)) return;
	last = time;
	std::lock_guard  lock(state.mutex);
	constexpr double MiB = 1024.0 * 1024.0;
	std::string      line;
	for (int i = 0; i < static_cast<int>(Touch::Count); i++) {
		if (state.touches[i] == 0) continue;
		char part[160];
		std::snprintf(part, sizeof(part), "%s%s %llu (%.0f MiB)", line.empty() ? "" : ", ", TouchName(static_cast<Touch>(i)),
		              static_cast<unsigned long long>(state.touches[i]), state.touch_bytes[i] / MiB);
		line += part;
	}
	const bool offload = GetMode() == Mode::Offload;
	std::printf("Texture offload: %.0f MiB %s now (peak %.0f, %llu ranges / %.0f MiB ever); touched: %s; after <1 s %llu, <10 s %llu, "
	            "<60 s %llu, later %llu\n",
	            state.now / MiB, offload ? "out of RAM" : "probed", state.peak / MiB, static_cast<unsigned long long>(state.ranges_total),
	            state.total / MiB, line.empty() ? "none" : line.c_str(), static_cast<unsigned long long>(state.ages[0]),
	            static_cast<unsigned long long>(state.ages[1]), static_cast<unsigned long long>(state.ages[2]),
	            static_cast<unsigned long long>(state.ages[3]));
	if (offload) {
		std::printf("Texture offload: checked %llu images (%.0f MiB, %llu differ); %llu restores (%.0f MiB, %.1f ms, at most %.1f ms), "
		            "%llu lost, %llu missed (%.0f MiB); %llu buffer reads waited for a restore (%.0f MiB)\n",
		            static_cast<unsigned long long>(state.checked), state.checked_bytes / MiB,
		            static_cast<unsigned long long>(state.check_failed), static_cast<unsigned long long>(state.restores),
		            state.restore_bytes / MiB, state.restore_us / 1000.0, state.restore_us_max / 1000.0,
		            static_cast<unsigned long long>(state.lost), static_cast<unsigned long long>(state.missed), state.missed_bytes / MiB,
		            static_cast<unsigned long long>(state.read_waits.load()), state.read_wait_bytes.load() / MiB);
	}
	if (SpillEnabled()) {
		std::printf("Texture tails (KYTY_TEXTURE_SPILL): %.0f MiB out of RAM now in the spill file (peak %.0f); taken %llu (%.0f MiB), "
		            "written %llu (%.0f MiB, %.0f ms), dropped before written %llu; read back %llu (%.0f MiB, %.1f ms, at most %.1f ms), "
		            "lost %llu; file %.0f MiB (%zu free blocks)\n",
		            state.tails_now / MiB, state.tails_peak / MiB, static_cast<unsigned long long>(state.tails_taken),
		            state.tails_taken_bytes / MiB, static_cast<unsigned long long>(state.tails_written), state.tails_written_bytes / MiB,
		            state.tails_write_us / 1000.0, static_cast<unsigned long long>(state.tails_dropped),
		            static_cast<unsigned long long>(state.tails_restores), state.tails_restore_bytes / MiB, state.tails_restore_us / 1000.0,
		            state.tails_restore_us_max / 1000.0, static_cast<unsigned long long>(state.tails_lost), state.spill_end / MiB,
		            state.spill_free.size());
	}
#if defined(_WIN32)
	const auto exe  = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	const auto list = [exe](const std::unordered_map<uint64_t, uint64_t>& addresses, const char* what) {
		std::vector<std::pair<uint64_t, uint64_t>> top(addresses.begin(), addresses.end());
		std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
		for (size_t i = 0; i < std::min<size_t>(top.size(), 10); i++) {
			HMODULE module = nullptr;
			GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			                   reinterpret_cast<LPCWSTR>(top[i].first), &module);
			wchar_t name[MAX_PATH] {};
			if (module != nullptr) GetModuleFileNameW(module, name, MAX_PATH);
			const wchar_t* base = wcsrchr(name, L'\\');
			std::printf("Texture offload:   %s %llu touches at %ls+0x%llx%s\n", what, static_cast<unsigned long long>(top[i].second),
			            base != nullptr ? base + 1 : name, static_cast<unsigned long long>(top[i].first - reinterpret_cast<uint64_t>(module)),
			            reinterpret_cast<uint64_t>(module) == exe ? " (the emulator)" : "");
		}
	};
	list(state.rips, "at");
	list(state.callers, "called from");
#endif
	std::fflush(stdout);
}

} // namespace TextureOffload
