#pragma once
// Texture offload (Windows, Linux). The guest-memory copy of a texture GPU work never wrote, that only shaders sample,
// uploaded and clean for a while, duplicates its image in video memory: on a PS5 the game's memory holds it once.
//
// KYTY_TEXTURE_OFFLOAD_PROBE=1 (diagnostics): such a texture's pages are made inaccessible in both views of guest memory
// (the guest's mapping: read watchers of the page manager; the backing view the emulator reads and writes: no access, a
// guard page on Windows with guards, KYTY_READ_GUARD) and nothing else. Every later touch faults, is counted with who
// made it and how long after, and gives the range its protection back for good: what keeping the texture only in video
// memory would cost.
//
// KYTY_TEXTURE_OFFLOAD=1 (off by default, see GetMode): the same for textures whose download was checked to give
// their guest bytes back exactly (16 MiB of images a scan: the downloads go with the frame's GPU work), and their pages
// are dropped by the offload worker (Windows: DiscardVirtualMemory in the backing view a MiB at a time, the guest view's
// pages trimmed; Linux: a hole punched in the backing store's memfd, which frees them in every view: the RAM given back,
// no pagefile or swap write). On a 16 GB PC with a 1-1 -> Stonefang -> 1-1 route: up to 2 GB of
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

#include "file-offload.h"
#include "local-platform.h"

#if defined(_WIN32)
#include "kernel/memory.h"

#include <windows.h>
#elif defined(__linux__)
#include "common/virtualMemory.h"
#include "kernel/memory.h"

#include <cstring>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <ucontext.h>
#endif

// Where it runs: Windows and Linux (macOS: no backing store to drop the pages from).
#if defined(_WIN32) || defined(__linux__)
#define KYTY_TEXTURE_OFFLOAD_SUPPORTED 1
#else
#define KYTY_TEXTURE_OFFLOAD_SUPPORTED 0
#endif

namespace TextureOffload {

enum class Mode { Off, Probe, Offload };

// KYTY_TEXTURE_OFFLOAD=1|0 (default: off; it was on with 20 GiB of RAM or less until v20261024: on 16 GB PCs textures
// went black at a distance and characters lost their hair (#5, #159, #157), mostly gone with it off though its own
// checks counted nothing lost; the ~0.5 GB it saved is covered by the engine's free blocks given back
// (loader/demonsSoulsPageFree.h) and the game files closed when idle).
inline Mode GetMode() {
#if KYTY_TEXTURE_OFFLOAD_SUPPORTED
	static const Mode mode = [] {
		const char* offload = std::getenv("KYTY_TEXTURE_OFFLOAD");
		if (offload != nullptr && offload[0] == '1') return Mode::Offload;
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

// The texture offload or the file offload (file-offload.h, whose ranges this state keeps too).
inline bool AnyEnabled() {
	return Enabled() || FileOffload::Enabled();
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
	BufferUpload,
	ImageGone,
	MappedAgain,
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
	                              "buffer upload over it",
	                              "image gone",
	                              "its memory mapped again"};
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
	// File-backed (file-offload.h): its file's id (0: an image's range) and the file's offset of
	// its first byte; whether its bytes were found equal to the file's. (Its `serial` tells it from
	// a range taken at the same address later.)
	uint32_t file        = 0;
	uint64_t file_offset = 0;
	bool     verified    = false;
};

// Set by the texture cache: the page manager's read watchers of a guest range removed; a restore of the discarded ranges
// over a guest range by the render thread (any thread CanWait allows: it returns when they are back); whether the calling
// thread may wait for the render thread.
using RemoveWatchers = void (*)(uint64_t vaddr, uint64_t size);
using Restore        = void (*)(uint64_t vaddr, uint64_t size, Touch touch);
using CanWait        = bool (*)();
using Submit         = void (*)();

struct State {
	std::mutex                   mutex;
	std::map<uint64_t, Range>    guest;   // by guest start
	std::map<uint64_t, uint64_t> backing; // backing piece start -> guest start
	std::vector<uint64_t>        pending; // touched ranges (guest starts) whose read watchers go (Drain)
	std::unordered_set<uint64_t> touched_serials; // images touched once: not taken again
	std::atomic<RemoveWatchers>  remove {nullptr};
	std::atomic<Restore>         restore {nullptr};
	std::atomic<CanWait>         can_wait {nullptr};
	// (The render thread's unsubmitted restore downloads submitted.)
	std::atomic<Submit>          submit {nullptr};
	std::atomic<uint64_t>        guest_lo {~uint64_t {0}}, guest_hi {0}, backing_lo {~uint64_t {0}}, backing_hi {0};
	std::atomic<uint64_t>        restoring {0}; // ranges whose restore runs (BeginRestore to EndRestore)
	// Statistics.
	uint64_t now = 0, peak = 0, total = 0, ranges_total = 0;
	uint64_t touches[static_cast<int>(Touch::Count)] {}, touch_bytes[static_cast<int>(Touch::Count)] {};
	uint64_t ages[4] {}; // <1 s, <10 s, <60 s, later
	uint64_t restores = 0, restore_bytes = 0, restore_us = 0, restore_us_max = 0, lost = 0, missed = 0, missed_bytes = 0;
	uint64_t checked = 0, checked_bytes = 0, check_failed = 0;
	uint64_t read_waits = 0, read_wait_bytes = 0; // buffer uploads that waited for an asynchronous restore
	// Host writes that could not wait for an asynchronous restore under them.
	uint64_t                               write_unwaited = 0;
	std::unordered_map<uint64_t, uint64_t> rips;    // the emulator's instructions -> touches
	std::unordered_map<uint64_t, uint64_t> callers; // their callers (one frame unwound) -> touches
	// The file-backed ranges' own (file-offload.h): taken (`file_now`: still taken, verified or
	// not), checked, touched.
	uint64_t file_now = 0, file_peak = 0, file_total = 0, file_ranges = 0, file_dropped = 0;
	uint64_t file_dropped_peak = 0, file_verified = 0, file_differ = 0, file_differ_bytes = 0;
	uint64_t file_unreadable = 0;
	uint64_t file_touches[static_cast<int>(Touch::Count)] {};
	uint64_t file_touch_bytes[static_cast<int>(Touch::Count)] {};
	uint64_t file_ages[4] {};
	uint64_t file_restores = 0, file_restore_bytes = 0, file_restore_us = 0,
	         file_restore_us_max = 0;
	uint64_t file_lost = 0, file_read_failures = 0, file_serial = 0;
	uint64_t file_scans = 0, file_scan_us = 0, file_scan_us_max = 0, file_scan_ranges = 0;
	std::unordered_map<uint32_t, uint64_t> file_touched; // file -> touches
	std::unordered_map<uint64_t, uint64_t> file_rips;    // the emulator's instructions -> touches
};

inline State& Get() {
	static State state;
	return state;
}

// The ranges in `guest` per MiB of the guest's address space (below 1 TiB), read without the lock:
// OnHostAccess's check that a taken range may lie under a host write. The render thread's linear
// copies (hundreds a frame) and downloads rarely touch one, and skip the lock the offload worker
// holds through a range's discard. Static storage: the counters of guest memory with taken ranges
// become resident only (2 KiB a GiB).
constexpr uint64_t           BlockBits  = 20;
constexpr uint64_t           BlockLimit = uint64_t {1} << 40u;
inline std::atomic<uint16_t> g_blocks[BlockLimit >> BlockBits];

// (state.mutex held) A range in or out of `guest`: its blocks counted (+1) or not (-1).
inline void CountBlocksLocked(uint64_t vaddr, uint64_t end, int delta) {
	end = std::min(end, BlockLimit);
	if (vaddr >= end) return;
	for (uint64_t block = vaddr >> BlockBits; block <= (end - 1) >> BlockBits; ++block)
		g_blocks[block].fetch_add(static_cast<uint16_t>(delta), std::memory_order_relaxed);
}

// Whether a range in `guest` may overlap [vaddr, vaddr + size) (beyond BlockLimit: always), without
// the lock.
[[nodiscard]] inline bool MayOverlap(uint64_t vaddr, uint64_t size) {
	const uint64_t end = vaddr + size;
	if (end > BlockLimit || end < vaddr) return true;
	for (uint64_t block = vaddr >> BlockBits; block <= (end - 1) >> BlockBits; ++block)
		if (g_blocks[block].load(std::memory_order_relaxed) != 0) return true;
	return false;
}

// The pieces in `backing` per MiB of the backing view, read without the lock: OnBackingMapped's
// check that a guest map may cover a taken range's pieces (most guest maps fall between backing_lo
// and backing_hi once a few images are taken). Indexed by the MiB modulo BlockLimit, wherever the
// view lies: a store smaller than BlockLimit shares no counter within itself, and a shared one
// would only take the lock in vain.
inline std::atomic<uint16_t> g_backing_blocks[BlockLimit >> BlockBits];

// (state.mutex held) A piece in or out of `backing`: its blocks counted (+1) or not (-1).
inline void CountBackingLocked(uint64_t backing, uint64_t size, int delta) {
	if (size == 0) return;
	constexpr uint64_t Mask = (BlockLimit >> BlockBits) - 1;
	for (uint64_t block = backing >> BlockBits; block <= (backing + size - 1) >> BlockBits; ++block)
		g_backing_blocks[block & Mask].fetch_add(static_cast<uint16_t>(delta),
		                                         std::memory_order_relaxed);
}

// Whether a piece in `backing` may overlap [backing, backing + size), without the lock.
[[nodiscard]] inline bool MayOverlapBacking(uint64_t backing, uint64_t size) {
	if (size > BlockLimit) return true;
	constexpr uint64_t Mask = (BlockLimit >> BlockBits) - 1;
	for (uint64_t block = backing >> BlockBits; block <= (backing + size - 1) >> BlockBits; ++block)
		if (g_backing_blocks[block & Mask].load(std::memory_order_relaxed) != 0) return true;
	return false;
}

enum class Access { None, Read, ReadWrite };

#if defined(_WIN32)
inline bool Protect(uint64_t address, uint64_t size, Access access) {
	// (With guard pages (KYTY_READ_GUARD) a guard over read-only, as in the guest view: a page of a paged section view
	// with no access blue-screens Windows once its memory leaves RAM (memory.cpp kyty_local_read_guard), and a
	// discarded one has left it. A guard goes with the fault it raises: RearmBackingGuard puts it back.)
	const DWORD none       = Libs::LibKernel::Memory::ReadGuardsOn() ? PAGE_GUARD | PAGE_READONLY : PAGE_NOACCESS;
	const DWORD protection = access == Access::None   ? none
	                         : access == Access::Read ? PAGE_READONLY
	                                                  : PAGE_READWRITE;
	// (Windows unlocks a locked page given no access: the pages read back there leave the hot set first.)
	if (access == Access::None) Libs::LibKernel::Memory::DropHotReadsBacking(address, size);
	DWORD       old        = 0;
	return VirtualProtect(reinterpret_cast<void*>(address), size, protection, &old) != 0;
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
#elif defined(__linux__)
// (Through the emulator's record of protections, as the page manager's changes of the guest view.)
inline bool Protect(uint64_t address, uint64_t size, Access access) {
	using Common::VirtualMemory::Mode;
	return Common::VirtualMemory::Protect(address, size,
	                                      access == Access::None   ? Mode::NoAccess
	                                      : access == Access::Read ? Mode::Read
	                                                               : Mode::ReadWrite);
}

// The emulator's code or a library's (guest code lies in no loaded object).
inline bool IsHostCode(uint64_t rip) {
	Dl_info info {};
	return dladdr(reinterpret_cast<const void*>(rip), &info) != 0;
}

// The return address at the stack top: a leaf's caller (memcpy's), else the faulting function's own
// return address or a stack slot (no unwind of the faulting frame here).
inline uint64_t CallerOf(const void* native_context) {
	const auto* context = static_cast<const ucontext_t*>(native_context);
	return *reinterpret_cast<const uint64_t*>(context->uc_mcontext.gregs[REG_RSP]);
}
#endif

// A guest range [vaddr, vaddr + size) taken, with its backing pieces and image (texture cache,
// render thread; the caller protects the backing view, and discards, after). `hold` puts the page
// manager's hold on its guest view, in force when it returns, under the state's lock and after the
// range is in the maps: another thread's touch (a host write's OnHostAccess) cannot give the range
// back, and remove the hold, before the hold is there (a no-access applied after the hold went
// stayed for good, and the game's next read of the page faulted forever), and a fault on a page
// just protected finds its range.
template <typename Hold>
inline void Add(uint64_t vaddr, uint64_t size, std::vector<Piece> pieces, uint64_t serial,
                uint64_t image, bool discarded, Hold&& hold) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	Range           range;
	range.guest_end = vaddr + size;
	range.pieces    = std::move(pieces);
	range.serial    = serial;
	range.image     = image;
	range.since     = std::chrono::steady_clock::now();
	range.discarded = discarded;
	for (const auto& piece: range.pieces) {
		state.backing[piece.backing] = vaddr;
		CountBackingLocked(piece.backing, piece.size, 1);
		if (piece.backing < state.backing_lo.load(std::memory_order_relaxed)) state.backing_lo.store(piece.backing, std::memory_order_relaxed);
		if (piece.backing + piece.size > state.backing_hi.load(std::memory_order_relaxed))
			state.backing_hi.store(piece.backing + piece.size, std::memory_order_relaxed);
	}
	if (vaddr < state.guest_lo.load(std::memory_order_relaxed)) state.guest_lo.store(vaddr, std::memory_order_relaxed);
	if (vaddr + size > state.guest_hi.load(std::memory_order_relaxed)) state.guest_hi.store(vaddr + size, std::memory_order_relaxed);
	if (state.guest.emplace(vaddr, std::move(range)).second)
		CountBlocksLocked(vaddr, vaddr + size, 1);
	state.now += size;
	state.peak = std::max(state.peak, state.now);
	state.total += size;
	state.ranges_total++;
	hold();
}

// A file-backed range [vaddr, vaddr + size) taken (render thread; the offload worker checks it
// against the file after): its backing view gets no access, then `hold` as for Add. False, and
// nothing taken, when a protection of the backing view fails (vm.max_map_count reached): a view
// left readable would read the hole punched later as zeros, without a fault.
template <typename Hold>
inline bool AddFile(uint64_t vaddr, uint64_t size, std::vector<Piece> pieces, uint32_t file,
                    uint64_t file_offset, Hold&& hold) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
#if KYTY_TEXTURE_OFFLOAD_SUPPORTED
	for (size_t i = 0; i < pieces.size(); i++) {
		if (Protect(pieces[i].backing, pieces[i].size, Access::None)) continue;
		for (size_t done = 0; done < i; done++)
			(void)Protect(pieces[done].backing, pieces[done].size, Access::ReadWrite);
		return false;
	}
#endif
	Range range;
	range.guest_end   = vaddr + size;
	range.pieces      = std::move(pieces);
	range.since       = std::chrono::steady_clock::now();
	range.serial      = ++state.file_serial;
	range.file        = file;
	range.file_offset = file_offset;
	for (const auto& piece: range.pieces) {
		state.backing[piece.backing] = vaddr;
		CountBackingLocked(piece.backing, piece.size, 1);
		if (piece.backing < state.backing_lo.load(std::memory_order_relaxed))
			state.backing_lo.store(piece.backing, std::memory_order_relaxed);
		if (piece.backing + piece.size > state.backing_hi.load(std::memory_order_relaxed))
			state.backing_hi.store(piece.backing + piece.size, std::memory_order_relaxed);
	}
	if (vaddr < state.guest_lo.load(std::memory_order_relaxed))
		state.guest_lo.store(vaddr, std::memory_order_relaxed);
	if (vaddr + size > state.guest_hi.load(std::memory_order_relaxed))
		state.guest_hi.store(vaddr + size, std::memory_order_relaxed);
	if (state.guest.emplace(vaddr, std::move(range)).second)
		CountBlocksLocked(vaddr, vaddr + size, 1);
	state.file_now += size;
	state.file_peak = std::max(state.file_peak, state.file_now);
	state.file_total += size;
	state.file_ranges++;
	hold();
	return true;
}

#if KYTY_TEXTURE_OFFLOAD_SUPPORTED
// The worker thread's drop of range `begin`'s pages (discarded in the backing view and no access there): unless it was
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
#if defined(_WIN32)
		// (A discard is refused on pages without access; a locked page would stay in RAM: the pages read back there leave
		// the hot set first.)
		Libs::LibKernel::Memory::DropHotReadsBacking(piece.backing, piece.size);
		for (uint64_t at = 0; at < piece.size; at += Chunk) {
			(void)DiscardVirtualMemory(reinterpret_cast<void*>(piece.backing + at), std::min(Chunk, piece.size - at));
			SwitchToThread();
		}
		(void)Protect(piece.backing, piece.size, Access::None);
#else
		// (Linux: no access first, then the hole, which frees the pages in every view: a touch in
		// between would have read zeros. A MiB at a time too: faults on the memfd's pages wait for
		// a punch over them.)
		(void)Protect(piece.backing, piece.size, Access::None);
		for (uint64_t at = 0; at < piece.size; at += Chunk) {
			(void)Libs::LibKernel::Memory::DiscardBacking(piece.backing + at,
			                                              std::min(Chunk, piece.size - at));
			sched_yield();
		}
#endif
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

// A range is taken over [vaddr, vaddr + size).
[[nodiscard]] inline bool AnyTaken(uint64_t vaddr, uint64_t size) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	auto            it = state.guest.upper_bound(vaddr);
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

// (state.mutex held) A restore by another thread (or the offload worker) runs over [vaddr, vaddr +
// size); `files`: a file read back only.
inline bool RestoringLocked(State& state, uint64_t vaddr, uint64_t size, bool files = false) {
	for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + size; ++it)
		if (it->second.restoring && it->second.restorer != std::this_thread::get_id() &&
		    (!files || it->second.file != 0))
			return true;
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

// Whether a read of [vaddr, vaddr + size) through the backing view waits for a restore first (cheap while none runs). An
// image that goes is restored asynchronously (RestoreOffloaded): its backing pieces are readable at once and its bytes land
// when the GPU's download is done, so a reader of the backing view in between (a buffer upload) reads the discarded pages'
// zeros. Image uploads wait already.
[[nodiscard]] inline bool MustWaitBeforeRead(uint64_t vaddr, uint64_t size) {
	return Get().restoring.load(std::memory_order_acquire) != 0 && IsRestoring(vaddr, size);
}

// The render thread's asynchronous restores (an image gone) not submitted yet, submitted: their
// bytes land when the GPU is done with their downloads (the offload worker writes them). Before a
// wait of its own for another thread, which may wait for one of them (a guest readback's copy: its
// write-back waits for the restores under it, WaitRestoresBeforeWrite). Other threads: nothing.
inline void SubmitRestores() {
	if (const auto submit = Get().submit.load(std::memory_order_acquire); submit != nullptr)
		submit();
}

// A host write's wait for the restores under [vaddr, vaddr + size). An asynchronous one (an image
// gone) lands when the GPU is done with its download: the render thread submits that first (nothing
// else would; a game thread's wait ends at the render thread's next submission, which waits for
// nothing the game thread holds: SubmitRestores). Deferred GPU work and the threads CanWait refuses
// wait for the files' read backs only (an image's restore could be behind their own work): its
// bytes then land after the write, counted, reported once.
inline void WaitRestoresBeforeWrite(uint64_t vaddr, uint64_t size) {
	auto& state = Get();
	if (!MustWaitBeforeRead(vaddr, size)) return;
	const auto can_wait = state.can_wait.load(std::memory_order_acquire);
	if (can_wait == nullptr || can_wait()) {
		SubmitRestores();
		WaitRestores(vaddr, size);
		return;
	}
	for (;;) {
		{
			std::lock_guard lock(state.mutex);
			if (!RestoringLocked(state, vaddr, size, true)) {
				if (RestoringLocked(state, vaddr, size) && state.write_unwaited++ == 0)
					std::printf(
					    "Texture offload: a host write over 0x%llx..0x%llx did not wait for the "
					    "restore there\n",
					    static_cast<unsigned long long>(vaddr),
					    static_cast<unsigned long long>(vaddr + size));
				return;
			}
		}
		std::this_thread::yield();
	}
}

inline void NoteReadWait(uint64_t size) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	state.read_waits++;
	state.read_wait_bytes += size;
}

// (state.mutex held) Counted as touched by `touch`.
inline void CountLocked(State& state, std::map<uint64_t, Range>::iterator it, Touch touch, uint64_t rip = 0, uint64_t caller = 0) {
	auto&      range = it->second;
	const auto size  = range.guest_end - it->first;
	if (range.file != 0) {
		const double seconds =
		    std::chrono::duration<double>(std::chrono::steady_clock::now() - range.since).count();
		state.file_touches[static_cast<int>(touch)]++;
		state.file_touch_bytes[static_cast<int>(touch)] += size;
		state.file_ages[seconds < 1 ? 0 : seconds < 10 ? 1 : seconds < 60 ? 2 : 3]++;
		state.file_touched[range.file]++;
		if (seconds < 10 && touch != Touch::Unmap) FileOffload::QuickTouch(range.file);
		if (rip != 0) state.file_rips[rip]++;
		(void)caller;
		state.file_now -= std::min(state.file_now, size);
		if (range.discarded) state.file_dropped -= std::min(state.file_dropped, size);
		return;
	}
	state.touches[static_cast<int>(touch)]++;
	state.touch_bytes[static_cast<int>(touch)] += size;
	const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - range.since).count();
	state.ages[seconds < 1 ? 0 : seconds < 10 ? 1 : seconds < 60 ? 2 : 3]++;
	if (rip != 0) state.rips[rip]++;
	if (caller != 0) state.callers[caller]++;
	state.touched_serials.insert(range.serial);
	state.now -= std::min(state.now, size);
}

// (state.mutex held) The touch of range `it`: counted, its views given a protection that lets accesses through (the
// guest's read-only: its write watchers stay) and queued for its read watchers to go.
inline void TouchLocked(State& state, std::map<uint64_t, Range>::iterator it, Touch touch, uint64_t rip = 0, uint64_t caller = 0) {
	auto& range = it->second;
	if (range.touched) return;
	range.touched = true;
	CountLocked(state, it, touch, rip, caller);
#if KYTY_TEXTURE_OFFLOAD_SUPPORTED
	for (const auto& piece: range.pieces)
		(void)Protect(piece.backing, piece.size, Access::ReadWrite);
	(void)Protect(it->first, range.guest_end - it->first, Access::Read);
#endif
	state.pending.push_back(it->first);
}

// (state.mutex held) A range out of the maps.
inline void EraseLocked(State& state, std::map<uint64_t, Range>::iterator range) {
	for (const auto& piece: range->second.pieces) {
		state.backing.erase(piece.backing);
		CountBackingLocked(piece.backing, piece.size, -1);
	}
	CountBlocksLocked(range->first, range->second.guest_end, -1);
	state.guest.erase(range);
}

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
		EraseLocked(state, range);
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
		// (A file-backed range is read back by RestoreFiles.)
		if (range.file != 0 || !range.discarded || range.restoring ||
		    (range.touched && !(late && range.missed)))
			continue;
		if (!range.touched && touch != Touch::Count) CountLocked(state, it, touch);
		range.touched   = true;
		range.restoring = true;
		range.restorer  = async ? std::thread::id {} : std::this_thread::get_id();
		ranges.push_back({it->first, range.guest_end, range.serial, range.image, range.pieces});
	}
	state.restoring.fetch_add(ranges.size(), std::memory_order_acq_rel);
	return ranges;
}

// The restored ranges out of the maps (their bytes back and read watchers removed: other touches go on).
inline void EndRestore(const std::vector<Restoring>& ranges, uint64_t bytes, uint64_t us, uint64_t lost) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	for (const auto& restored: ranges) {
		auto range = state.guest.find(restored.begin);
		if (range == state.guest.end()) continue;
		EraseLocked(state, range);
	}
	std::erase_if(state.pending, [&](uint64_t begin) { return state.guest.find(begin) == state.guest.end(); });
	state.restoring.fetch_sub(ranges.size(), std::memory_order_acq_rel);
	state.restores++;
	state.restore_bytes += bytes;
	state.restore_us += us;
	state.restore_us_max = std::max(state.restore_us_max, us);
	state.lost += lost;
}

inline void NoteCheck(uint64_t images, uint64_t bytes, uint64_t failed) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	state.checked += images;
	state.checked_bytes += bytes;
	state.check_failed += failed;
}

// A scan of the file offload's (TextureCache::OffloadFiles, render thread): its ranges taken and
// its time.
inline void NoteFileScan(uint64_t ranges, uint64_t us) {
	auto&           state = Get();
	std::lock_guard lock(state.mutex);
	state.file_scans++;
	state.file_scan_ranges += ranges;
	state.file_scan_us += us;
	state.file_scan_us_max = std::max(state.file_scan_us_max, us);
}

#if KYTY_FILE_OFFLOAD_SUPPORTED
// Any thread (a fault handler's too): the dropped file-backed range at `begin` read back from its
// file (a pread into the spare view of the backing store: its views keep their protection
// meanwhile), then touched as `touch`. Other threads' touches of it wait meanwhile (`restoring`).
// False when it was none to read back (not dropped, touched, or another thread's restore of it
// runs).
inline bool RestoreFileRange(uint64_t begin, Touch touch, uint64_t rip = 0) {
	auto&              state = Get();
	std::vector<Piece> pieces;
	uint32_t           file   = 0;
	uint64_t           offset = 0;
	{
		std::lock_guard lock(state.mutex);
		const auto      it = state.guest.find(begin);
		if (it == state.guest.end() || it->second.file == 0 || !it->second.discarded ||
		    it->second.touched || it->second.restoring)
			return false;
		it->second.restoring = true;
		it->second.restorer  = std::this_thread::get_id();
		pieces               = it->second.pieces;
		file                 = it->second.file;
		offset               = it->second.file_offset;
		state.restoring.fetch_add(1, std::memory_order_acq_rel);
	}
	const auto start = std::chrono::steady_clock::now();
	const int  fd    = FileOffload::Open(file);
	int        error = fd < 0 ? errno : 0;
	uint64_t   bytes = 0, lost = 0, lost_at = 0;
	for (const auto& piece: pieces) {
		auto*    spare = Libs::LibKernel::Memory::SpareBackingPointer(piece.backing);
		uint64_t done  = 0;
		while (spare != nullptr && fd >= 0 && done < piece.size) {
			const auto read =
			    pread(fd, spare + done, piece.size - done, static_cast<off_t>(offset + done));
			if (read <= 0) {
				if (read < 0 && error == 0) error = errno;
				break;
			}
			done += static_cast<uint64_t>(read);
		}
		bytes += done;
		if (done < piece.size && lost == 0) lost_at = offset + done;
		lost += piece.size - done; // (zeros: the hole's)
		offset += piece.size;
		// (The spare view's mappings of the pages go: they are the store's, mapped in the other
		// views too.)
		if (spare != nullptr) (void)madvise(spare, piece.size, MADV_DONTNEED);
	}
	if (fd >= 0) close(fd);
	const auto us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                          std::chrono::steady_clock::now() - start)
	                                          .count());
	if (lost != 0) {
		// (Loud: the game now reads zeros where its file's bytes were.)
		static std::atomic<uint32_t> told {0};
		if (told.fetch_add(1, std::memory_order_relaxed) < 32) {
			std::printf("File offload: READ BACK FAILED: %s at offset 0x%llx (%s): %llu bytes "
			            "are zeros\n",
			            FileOffload::Name(file).c_str(), static_cast<unsigned long long>(lost_at),
			            error != 0 ? std::strerror(error) : "the file is shorter",
			            static_cast<unsigned long long>(lost));
			std::fflush(stdout);
		}
	}
	std::lock_guard lock(state.mutex);
	state.restoring.fetch_sub(1, std::memory_order_acq_rel);
	state.file_restores++;
	state.file_restore_bytes += bytes;
	state.file_restore_us += us;
	state.file_restore_us_max = std::max(state.file_restore_us_max, us);
	state.file_lost += lost;
	state.file_read_failures += lost != 0 ? 1 : 0;
	// (Still there: a range whose restore runs leaves the maps only through its restorer.)
	const auto it = state.guest.find(begin);
	if (it == state.guest.end()) return true;
	it->second.restoring = false;
	TouchLocked(state, it, touch, rip);
	it->second.discarded = false;
	return true;
}

// The dropped file-backed ranges over [vaddr, vaddr + size) read back on this thread
// (RestoreFileRange).
inline void RestoreFiles(uint64_t vaddr, uint64_t size, Touch touch) {
	auto& state = Get();
	if (size == 0 || vaddr + size <= state.guest_lo.load(std::memory_order_relaxed) ||
	    vaddr >= state.guest_hi.load(std::memory_order_relaxed))
		return;
	std::vector<uint64_t> begins;
	{
		std::lock_guard lock(state.mutex);
		for (auto it = FirstOver(state, vaddr); it != state.guest.end() && it->first < vaddr + size;
		     ++it)
			if (it->second.file != 0 && it->second.discarded && !it->second.touched)
				begins.push_back(it->first);
	}
	for (const auto begin: begins)
		(void)RestoreFileRange(begin, touch);
}

// The offload worker: the taken file-backed range at `begin` compared with its file (through the
// spare view: both views of it are protected). Equal, it is dropped (`drop`: KYTY_FILE_OFFLOAD=1)
// or stays taken; else, or when the file cannot be read, it is given back (its read watchers go
// with the next Drain). (The file's pages stay in the page cache, where the game's read put them: a
// read-back from there took a sixth of the time one from the disk took.)
inline void CheckFileRange(uint64_t begin, bool drop) {
	auto&              state = Get();
	std::vector<Piece> pieces;
	uint32_t           file   = 0;
	uint64_t           offset = 0, serial = 0;
	{
		std::lock_guard lock(state.mutex);
		const auto      it = state.guest.find(begin);
		if (it == state.guest.end() || it->second.file == 0 || it->second.touched ||
		    it->second.restoring)
			return;
		pieces = it->second.pieces;
		file   = it->second.file;
		offset = it->second.file_offset;
		serial = it->second.serial;
	}
	const int                         fd    = FileOffload::Open(file);
	bool                              equal = fd >= 0;
	thread_local std::vector<uint8_t> bytes;
	for (const auto& piece: pieces) {
		const auto* spare = Libs::LibKernel::Memory::SpareBackingPointer(piece.backing);
		if (!equal || spare == nullptr) {
			equal = false;
			break;
		}
		bytes.resize(piece.size);
		uint64_t done = 0;
		while (done < piece.size) {
			const auto read = pread(fd, bytes.data() + done, piece.size - done,
			                        static_cast<off_t>(offset + done));
			if (read <= 0) break;
			done += static_cast<uint64_t>(read);
		}
		equal = done == piece.size && std::memcmp(bytes.data(), spare, piece.size) == 0;
		offset += piece.size;
		(void)madvise(const_cast<uint8_t*>(spare), piece.size, MADV_DONTNEED);
	}
	if (fd >= 0) close(fd);
	std::lock_guard lock(state.mutex);
	// (The same range: one taken at its address since has another serial.)
	const auto it = state.guest.find(begin);
	if (it == state.guest.end() || it->second.file != file || it->second.serial != serial ||
	    it->second.touched || it->second.restoring)
		return;
	auto&      range = it->second;
	const auto size  = range.guest_end - begin;
	if (!equal) {
		// Given back, not counted as a touch.
		(fd < 0 ? state.file_unreadable : state.file_differ)++;
		state.file_differ_bytes += size;
		state.file_now -= std::min(state.file_now, size);
		range.touched = true;
		for (const auto& piece: range.pieces)
			(void)Protect(piece.backing, piece.size, Access::ReadWrite);
		(void)Protect(begin, size, Access::Read);
		state.pending.push_back(begin);
		return;
	}
	range.verified = true;
	state.file_verified += size;
	if (!drop) return;
	// (The backing view has had no access since the range was taken.)
	for (const auto& piece: range.pieces)
		(void)Libs::LibKernel::Memory::DiscardBacking(piece.backing, piece.size);
	range.discarded = true;
	state.file_dropped += size;
	state.file_dropped_peak = std::max(state.file_dropped_peak, state.file_dropped);
}
#endif

#if KYTY_TEXTURE_OFFLOAD_SUPPORTED
// The exception filter's (Linux: the SIGSEGV handler's) probe handler (any thread): a touch of a
// taken range, either view.
inline bool OnFaultTaken(uint64_t address, uint64_t rip, const void* native_context, bool write) {
	auto&      state      = Get();
	const bool in_guest   = address >= state.guest_lo.load(std::memory_order_relaxed) && address < state.guest_hi.load(std::memory_order_relaxed);
	const bool in_backing = address >= state.backing_lo.load(std::memory_order_relaxed) && address < state.backing_hi.load(std::memory_order_relaxed);
	if (!in_guest && !in_backing) return false;
	// A fault raised while its range was still taken, served after another thread's touch gave the pages back (and the
	// range left the maps): the access goes again. (Only this protects the backing view.)
#if defined(_WIN32)
	const auto stale = [address] {
		MEMORY_BASIC_INFORMATION info {};
		return VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) == sizeof(info) && info.Protect == PAGE_READWRITE;
	};
#else
	// (Linux has no cheap query: only the offload takes the backing view's access, and gives it
	// back before a range leaves the maps, so a fault there outside them is such a race. Again and
	// again on one address it is none: not ours.)
	const auto stale = [address] {
		thread_local uint64_t last    = 0;
		thread_local uint32_t repeats = 0;
		repeats                       = address == last ? repeats + 1 : 0;
		last                          = address;
		return repeats < 64;
	};
#endif
	std::vector<std::pair<uint64_t, uint64_t>> now;
	uint64_t restore_begin = 0, restore_size = 0, wait_begin = 0, wait_size = 0, file_begin = 0,
	         file_rip      = 0;
	Touch    restore_touch = Touch::GameRead;
	bool     file_host = false, in_backing_hit = false;
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
			// (Its protection came back before its watchers went, or this thread restores it. The
			// emulator's write through the guest view has its watchers go now, as a game thread's
			// touch: read-only, it would fault forever.)
			if (host && (backing_hit || !write)) {
				if (backing_hit) {
					for (const auto& piece: range.pieces)
						(void)Protect(piece.backing, piece.size, Access::ReadWrite);
				} else {
					(void)Protect(it->first, range.guest_end - it->first, Access::Read);
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
			const Touch touch = backing_hit ? (write ? Touch::BackingWrite : Touch::BackingRead)
			                    : host      ? (write ? Touch::EmulatorGuestWrite : Touch::EmulatorGuestRead)
			                                : (write ? Touch::GameWrite : Touch::GameRead);
			const auto  can_wait = state.can_wait.load(std::memory_order_acquire);
			if (range.file != 0 && range.discarded) {
				// (A dropped file-backed range: read back by this thread, whichever it is.)
				file_begin     = it->first;
				restore_touch  = touch;
				file_rip       = host ? rip : 0;
				file_host      = host;
				in_backing_hit = backing_hit;
			} else if (range.discarded &&
			           state.restore.load(std::memory_order_acquire) != nullptr &&
			           can_wait != nullptr && can_wait()) {
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
				if (host && (backing_hit || !write)) return true;
				now = TakePendingLocked(state);
			}
		}
	}
	if (wait_size != 0) {
		WaitRestores(wait_begin, wait_size);
		return true;
	}
#if KYTY_FILE_OFFLOAD_SUPPORTED
	if (file_begin != 0) {
		(void)RestoreFileRange(file_begin, restore_touch, file_rip);
		if (!file_host || (write && !in_backing_hit)) {
			{
				std::lock_guard lock(state.mutex);
				now = TakePendingLocked(state);
			}
			RemoveAll(state, now); // (a game thread)
		}
		return true;
	}
#endif
	if (restore_size != 0) {
#if defined(__linux__)
		// (The signal handler runs with its signal blocked: a fault in the restore, which runs here
		// on the render thread, would end the process. Unblocked till the handler returns, which
		// gives the thread its mask back.)
		sigset_t faults;
		sigemptyset(&faults);
		sigaddset(&faults, SIGSEGV);
		sigaddset(&faults, SIGBUS);
		pthread_sigmask(SIG_UNBLOCK, &faults, nullptr);
#endif
		state.restore.load(std::memory_order_acquire)(restore_begin, restore_size, restore_touch);
		return true;
	}
	RemoveAll(state, now); // (a game thread: no lock of the emulator's held)
	return true;
}

#if defined(_WIN32)
// The exception filter's (memory.cpp ArmReadGuard, any thread, before OnFault): a guard of the backing view raised a
// fault and went with it. The page of a taken range neither touched nor being restored gets it back at once (the next
// read would see the discarded bytes without a fault). Under the state's lock, as the touches and restores that give
// the pages their access: one put back after a restore gave them access would fault the restore's writes (the offload
// worker's, which would wait for its own restore). Other pages have their access, or get it from the restore.
inline void RearmBackingGuard(uint64_t address) {
	auto& state = Get();
	if (address < state.backing_lo.load(std::memory_order_relaxed) ||
	    address >= state.backing_hi.load(std::memory_order_relaxed))
		return;
	std::lock_guard lock(state.mutex);
	auto            piece = state.backing.upper_bound(address);
	if (piece == state.backing.begin()) return;
	--piece;
	const auto range = state.guest.find(piece->second);
	if (range == state.guest.end() || range->second.touched || range->second.restoring) return;
	const bool inside = std::any_of(range->second.pieces.begin(), range->second.pieces.end(), [&](const Piece& p) {
		return address >= p.backing && address < p.backing + p.size;
	});
	if (!inside) return;
	constexpr uint64_t Page = 0x1000;
	DWORD              old  = 0;
	(void)VirtualProtect(reinterpret_cast<void*>(address & ~(Page - 1)), Page, PAGE_GUARD | PAGE_READONLY, &old);
}
#endif

inline bool OnFault(uint64_t address, uint64_t rip, const void* native_context, bool write) {
	const bool handled = OnFaultTaken(address, rip, native_context, write);
	// (Diagnostics: one address faulting again and again on a thread is a loop; told once, with the
	// range there.) (2 s on it: a loop through a slow path, a readback's wait for the render
	// thread, is one too.)
	thread_local uint64_t                              last_address = 0, repeats = 0;
	thread_local std::chrono::steady_clock::time_point first;
	repeats      = address == last_address ? repeats + 1 : 0;
	last_address = address;
	if (repeats == 0) first = std::chrono::steady_clock::now();
	if (repeats >= 1000 && repeats % 1000 == 0 &&
	    std::chrono::steady_clock::now() - first >= std::chrono::seconds(2) &&
	    std::exchange(first, std::chrono::steady_clock::time_point::max()) !=
	        std::chrono::steady_clock::time_point::max()) {
		auto&           state = Get();
		std::lock_guard lock(state.mutex);
		const auto      it    = FirstOver(state, address);
		const bool      found = it != state.guest.end() && it->first <= address;
		char            name[32] {};
#if defined(__linux__)
		pthread_getname_np(pthread_self(), name, sizeof(name));
#endif
		std::printf(
		    "Texture offload: fault loop at 0x%llx (rip 0x%llx, %s, host %d, handled %d, "
		    "thread %s): %s range 0x%llx..0x%llx file %u touched %d discarded %d restoring %d "
		    "missed %d verified %d, %zu pending\n",
		    static_cast<unsigned long long>(address), static_cast<unsigned long long>(rip),
		    write ? "write" : "read", IsHostCode(rip) ? 1 : 0, handled ? 1 : 0, name,
		    found ? "in" : "no", found ? static_cast<unsigned long long>(it->first) : 0ull,
		    found ? static_cast<unsigned long long>(it->second.guest_end) : 0ull,
		    found ? it->second.file : 0u, found && it->second.touched,
		    found && it->second.discarded, found && it->second.restoring,
		    found && it->second.missed, found && it->second.verified, state.pending.size());
		std::fflush(stdout);
	}
	return handled;
}
#endif

// A host write (InvalidateMemory: file reads, network; `over_all`: it writes all of [vaddr, vaddr +
// size)) or an unmap over [vaddr, vaddr + size): the taken ranges there touched. A discarded one a
// write covers part of comes back first (the restore; a file-backed one before an unmap too); the
// others are dropped. Normal context: their read watchers go at once.
inline void OnHostAccess(uint64_t vaddr, uint64_t size, bool unmap, bool over_all) {
	if (!AnyEnabled() || size == 0) return;
	auto& state = Get();
	if (vaddr + size <= state.guest_lo.load(std::memory_order_relaxed) || vaddr >= state.guest_hi.load(std::memory_order_relaxed)) return;
	if (!MayOverlap(vaddr, size)) return;
	WaitRestoresBeforeWrite(vaddr, size);
	std::vector<std::pair<uint64_t, uint64_t>> ranges, restore, file_restore;
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
				(it->second.file != 0 ? file_restore : restore)
				    .emplace_back(it->first, it->second.guest_end - it->first);
				continue;
			}
			// (A dropped file-backed range comes back before an unmap too: direct memory keeps its
			// bytes for a later mapping (the APR's unmap and map of the same memory), and the read
			// costs no GPU work.)
			if (unmap && it->second.discarded && it->second.file != 0) {
				file_restore.emplace_back(it->first, it->second.guest_end - it->first);
				continue;
			}
			TouchLocked(state, it, unmap ? Touch::Unmap : covered ? Touch::HostOverwrite : Touch::HostWrite);
		}
		ranges = TakePendingLocked(state);
	}
	RemoveAll(state, ranges);
#if KYTY_FILE_OFFLOAD_SUPPORTED
	if (!file_restore.empty()) {
		for (const auto& [begin, bytes]: file_restore)
			(void)RestoreFileRange(begin, unmap ? Touch::Unmap : Touch::HostWrite);
		{
			std::lock_guard lock(state.mutex);
			ranges = TakePendingLocked(state);
		}
		RemoveAll(state, ranges);
	}
#endif
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

// A guest mapping of the backing store's [backing, backing + size) about to be made (MapBacking,
// any thread: direct memory mapped again, at the same address or another): the taken ranges with
// pieces there are given back first. The new view would read a dropped range's hole without a
// fault, and a restore would overwrite what it wrote. A dropped file-backed range is read back on
// this thread; a dropped image range comes back late (Drain: a guest memory call does not wait for
// the render thread), as a host write that could not wait.
inline void OnBackingMapped(uint64_t backing, uint64_t size) {
	if (!AnyEnabled() || size == 0) return;
	auto& state = Get();
	if (backing + size <= state.backing_lo.load(std::memory_order_relaxed) ||
	    backing >= state.backing_hi.load(std::memory_order_relaxed) ||
	    !MayOverlapBacking(backing, size))
		return;
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	std::vector<uint64_t>                      files;
	{
		std::lock_guard lock(state.mutex);
		auto            piece = state.backing.upper_bound(backing);
		if (piece != state.backing.begin()) --piece;
		for (; piece != state.backing.end() && piece->first < backing + size; ++piece) {
			const auto it = state.guest.find(piece->second);
			if (it == state.guest.end() || it->second.touched || it->second.restoring) continue;
			const auto& pieces = it->second.pieces;
			const auto  found  = std::find_if(pieces.begin(), pieces.end(), [&](const Piece& p) {
				return p.backing == piece->first;
			});
			if (found == pieces.end() || found->backing + found->size <= backing) continue;
			if (it->second.discarded && it->second.file != 0) {
				files.push_back(it->first);
				continue;
			}
			if (it->second.discarded) {
				it->second.missed = true;
				state.missed++;
				state.missed_bytes += it->second.guest_end - it->first;
				std::printf("Texture offload: MISSED a new mapping of the memory of 0x%llx..0x%llx "
				            "(zeros until restored)\n",
				            static_cast<unsigned long long>(it->first),
				            static_cast<unsigned long long>(it->second.guest_end));
			}
			TouchLocked(state, it, Touch::MappedAgain);
		}
		ranges = TakePendingLocked(state);
	}
	RemoveAll(state, ranges);
#if KYTY_FILE_OFFLOAD_SUPPORTED
	if (files.empty()) return;
	for (const auto begin: files)
		(void)RestoreFileRange(begin, Touch::MappedAgain);
	{
		std::lock_guard lock(state.mutex);
		ranges = TakePendingLocked(state);
	}
	RemoveAll(state, ranges);
#endif
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

// (state.mutex held) The file offload's lines.
inline void ReportFilesLocked(State& state) {
	constexpr double MiB = 1024.0 * 1024.0;
	std::string      line;
	for (int i = 0; i < static_cast<int>(Touch::Count); i++) {
		if (state.file_touches[i] == 0) continue;
		char part[160];
		std::snprintf(part, sizeof(part), "%s%s %llu (%.0f MiB)", line.empty() ? "" : ", ",
		              TouchName(static_cast<Touch>(i)),
		              static_cast<unsigned long long>(state.file_touches[i]),
		              state.file_touch_bytes[i] / MiB);
		line += part;
	}
	uint64_t pending = 0, reads = 0, read_bytes = 0, hot_bytes = 0, hot_files = 0, small_bytes = 0;
	FileOffload::Pending(&pending, &reads, &read_bytes, &hot_bytes, &hot_files, &small_bytes);
	std::printf(
	    "File offload: t=%.0f s: %.0f MiB taken now (peak %.0f, %llu ranges / %.0f MiB ever), "
	    "%.0f MiB out of RAM (peak %.0f); %llu APR reads (%.0f MiB), %.0f MiB waiting, %.0f MiB "
	    "left alone (%llu files touched soon after), %.0f MiB in pieces under 64 KiB; checked "
	    "equal %.0f MiB, %llu differ, %llu unreadable (%.0f MiB); touched: %s; after <1 s %llu, "
	    "<10 s %llu, <60 s %llu, later %llu\n",
	    std::chrono::duration<double>(std::chrono::steady_clock::now() - FileOffload::Get().start)
	        .count(),
	    state.file_now / MiB, state.file_peak / MiB,
	    static_cast<unsigned long long>(state.file_ranges), state.file_total / MiB,
	    state.file_dropped / MiB, state.file_dropped_peak / MiB,
	    static_cast<unsigned long long>(reads), read_bytes / MiB, pending / MiB, hot_bytes / MiB,
	    static_cast<unsigned long long>(hot_files), small_bytes / MiB, state.file_verified / MiB,
	    static_cast<unsigned long long>(state.file_differ),
	    static_cast<unsigned long long>(state.file_unreadable), state.file_differ_bytes / MiB,
	    line.empty() ? "none" : line.c_str(), static_cast<unsigned long long>(state.file_ages[0]),
	    static_cast<unsigned long long>(state.file_ages[1]),
	    static_cast<unsigned long long>(state.file_ages[2]),
	    static_cast<unsigned long long>(state.file_ages[3]));
	if (state.file_restores != 0)
		std::printf(
		    "File offload: %llu reads back (%.0f MiB, %.1f ms, at most %.2f ms), %llu "
		    "failed (%.0f MiB lost)\n",
		    static_cast<unsigned long long>(state.file_restores), state.file_restore_bytes / MiB,
		    state.file_restore_us / 1000.0, state.file_restore_us_max / 1000.0,
		    static_cast<unsigned long long>(state.file_read_failures), state.file_lost / MiB);
	if (state.file_scans != 0)
		std::printf("File offload: %llu scans took %.1f ms on the render thread (at most %.2f ms), "
		            "%llu ranges taken\n",
		            static_cast<unsigned long long>(state.file_scans), state.file_scan_us / 1000.0,
		            state.file_scan_us_max / 1000.0,
		            static_cast<unsigned long long>(state.file_scan_ranges));
	std::vector<std::pair<uint32_t, uint64_t>> top(state.file_touched.begin(),
	                                               state.file_touched.end());
	std::sort(top.begin(), top.end(),
	          [](const auto& a, const auto& b) { return a.second > b.second; });
	for (size_t i = 0; i < std::min<size_t>(top.size(), 12); i++)
		std::printf("File offload:   %llu touches of %s\n",
		            static_cast<unsigned long long>(top[i].second),
		            FileOffload::Name(top[i].first).c_str());
#if defined(__linux__)
	// The emulator's instructions that touched them.
	std::vector<std::pair<uint64_t, uint64_t>> rips(state.file_rips.begin(), state.file_rips.end());
	std::sort(rips.begin(), rips.end(),
	          [](const auto& a, const auto& b) { return a.second > b.second; });
	for (size_t i = 0; i < std::min<size_t>(rips.size(), 8); i++) {
		Dl_info     info {};
		const bool  found = dladdr(reinterpret_cast<const void*>(rips[i].first), &info) != 0 &&
		                    info.dli_fname != nullptr;
		const char* name  = found ? info.dli_fname : "?";
		const char* base  = std::strrchr(name, '/');
		std::printf("File offload:   %llu touches at %s+0x%llx (%s)\n",
		            static_cast<unsigned long long>(rips[i].second),
		            base != nullptr ? base + 1 : name,
		            static_cast<unsigned long long>(
		                rips[i].first - (found ? reinterpret_cast<uint64_t>(info.dli_fbase) : 0)),
		            found && info.dli_sname != nullptr ? info.dli_sname : "?");
	}
#endif
}

inline void Report(bool force) {
	auto&       state = Get();
	static auto last  = std::chrono::steady_clock::now();
	const auto  time  = std::chrono::steady_clock::now();
	// (KYTY_OFFLOAD_REPORT_S: seconds between reports, default 30.)
	static const auto period = [] {
		const char* value = std::getenv("KYTY_OFFLOAD_REPORT_S");
		return std::chrono::seconds(
		    value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) : 30);
	}();
	if (!force && time - last < period) return;
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
	if (FileOffload::Enabled()) ReportFilesLocked(state);
	if (!Enabled()) {
		std::fflush(stdout);
		return;
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
		            "%llu lost, %llu missed (%.0f MiB); %llu buffer uploads waited for a restore (%.0f MiB), %llu host writes "
		            "could not\n",
		            static_cast<unsigned long long>(state.checked), state.checked_bytes / MiB,
		            static_cast<unsigned long long>(state.check_failed), static_cast<unsigned long long>(state.restores),
		            state.restore_bytes / MiB, state.restore_us / 1000.0, state.restore_us_max / 1000.0,
		            static_cast<unsigned long long>(state.lost), static_cast<unsigned long long>(state.missed), state.missed_bytes / MiB,
		            static_cast<unsigned long long>(state.read_waits), state.read_wait_bytes / MiB,
		            static_cast<unsigned long long>(state.write_unwaited));
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
#elif defined(__linux__)
	Dl_info self {};
	(void)dladdr(reinterpret_cast<const void*>(&Report), &self);
	const auto list = [&self](const std::unordered_map<uint64_t, uint64_t>& addresses,
	                          const char*                                   what) {
		std::vector<std::pair<uint64_t, uint64_t>> top(addresses.begin(), addresses.end());
		std::sort(top.begin(), top.end(),
		          [](const auto& a, const auto& b) { return a.second > b.second; });
		for (size_t i = 0; i < std::min<size_t>(top.size(), 10); i++) {
			Dl_info     info {};
			const bool  found = dladdr(reinterpret_cast<const void*>(top[i].first), &info) != 0 &&
			                    info.dli_fname != nullptr;
			const char* name  = found ? info.dli_fname : "?";
			const char* base  = std::strrchr(name, '/');
			std::printf(
			    "Texture offload:   %s %llu touches at %s+0x%llx%s\n", what,
			    static_cast<unsigned long long>(top[i].second), base != nullptr ? base + 1 : name,
			    static_cast<unsigned long long>(
			        top[i].first - (found ? reinterpret_cast<uint64_t>(info.dli_fbase) : 0)),
			    found && info.dli_fbase == self.dli_fbase ? " (the emulator)" : "");
		}
	};
	list(state.rips, "at");
	list(state.callers, "called from");
#endif
	std::fflush(stdout);
}

} // namespace TextureOffload
