#include "graphics/host_gpu/pageManager.h"
#include "slow-log.h"
#include "raiz.h" // (raiz1: KYTY_RAIZ)
#include "lock-watch.h" // (raiz10: the granule locks' waits)
#include "native/g1-probe.h" // (g1, KYTY_G1_PROBE: the pages GPU work marks no-access)

#include "graphics/host_gpu/regionDefinitions.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <sched.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {
namespace {
thread_local std::vector<PageManager::DeferredRange>* g_deferred_write_protect = nullptr;
thread_local uint64_t g_unmapping_begin = 0;
thread_local uint64_t g_unmapping_end   = 0;
// PageManager::DeferReadProtection: the GPU thread's pending no-access ranges, disjoint and not adjacent (a range
// next to one extends it).
thread_local bool g_defer_read_protect = false;
struct PendingRange {
	uint64_t begin, end;
};
thread_local std::vector<PendingRange> g_read_protects;
thread_local uint64_t                  g_observation_epoch = 0; // (PageManager::ObservationEpoch)
// PageManager::CaptureCpuWrites: the pages read (and reset) from the write watch right before the emulator wrote them,
// for the next SweepCpuWrites (unsorted; a leaf lock).
std::mutex            g_captured_mutex;
std::vector<uint64_t> g_captured_pages;
// PageManager::NativeWriteWatch: KYTY_NATIVE_MEMORY=1.
bool ReadNativeWriteWatch() noexcept {
	(void)Raiz::On(); // (raiz1: KYTY_RAIZ=1 forces KYTY_NATIVE_MEMORY=1, KYTY_WRITE_WATCH=1, KYTY_WRITE_WATCH_MODE=protect)
	const char* v = std::getenv("KYTY_NATIVE_MEMORY");
	return v != nullptr && v[0] == '1' && v[1] == '\0';
}
// (11-14) Read at first use, not at static initialization: the launcher sets KYTY_NATIVE_MEMORY after the exe started
// (a static initializer read it unset: no sweep ever ran).
inline bool NativeWriteWatchOn() noexcept {
	static const bool on = ReadNativeWriteWatch();
	return on;
}
#define g_native_write_watch (NativeWriteWatchOn())
// (nativemem1) PageManager::NativeReadFaults: KYTY_WRITE_WATCH=1 with the write watch above (LibKernel::Memory::
// NativeReadFaultsEnabled: KYTY_NATIVE_MEMORY=1 too). Read at first use, as above.
inline bool NativeReadFaultsOn() noexcept {
	static const bool on = g_native_write_watch && Libs::LibKernel::Memory::NativeReadFaultsEnabled();
	return on;
}
// (ww2) KYTY_WRITE_WATCH_MODE, with KYTY_NATIVE_MEMORY=1 and KYTY_WRITE_WATCH=1 only (Off otherwise: nothing below
// changes anything then). Read at first use, as above (the launcher sets it after the exe started).
//   poll (the default): the write watch is still read by sweeps of every region a write watcher holds, but only at the
//     hand-overs that can carry a write no sweep has seen: a submission the game queued after the last sweep began
//     (PageManager::SweptSince) and a WAIT_REG_MEM satisfied by a value the command processors did not write themselves
//     (graphicsRun.cpp). And no written page is lost to GPU work: the write watch of a range is read right before the
//     buffers make it GPU-written (TakeWritesBeforeGpuWrite) and right after its pages became no-access
//     (CaptureAfterNoAccess).
//   protect: no sweep at all. The write watchers protect their pages read-only again and the game's writes fault, as
//     without native memory - on private memory now, not on a section's pages -.
//   sweep: ram4's sweeps, unchanged (every submission, every satisfied WAIT_REG_MEM), to compare.
enum class WatchMode : uint8_t { Off, Sweep, Poll, Protect };
WatchMode ReadWatchMode() noexcept {
	if (!NativeReadFaultsOn()) return WatchMode::Off;
	(void)Raiz::On(); // (raiz1)
	WatchMode   mode = WatchMode::Poll;
	const char* v    = std::getenv("KYTY_WRITE_WATCH_MODE");
	if (v != nullptr && (std::strcmp(v, "sweep") == 0 || std::strcmp(v, "old") == 0 || std::strcmp(v, "0") == 0))
		mode = WatchMode::Sweep;
	else if (v != nullptr && (std::strcmp(v, "protect") == 0 || std::strcmp(v, "fault") == 0 || std::strcmp(v, "2") == 0))
		mode = WatchMode::Protect;
	std::printf("Write watch: mode %s (KYTY_WRITE_WATCH_MODE=poll|protect|sweep; poll is the default)%s\n",
	            mode == WatchMode::Poll ? "poll" : mode == WatchMode::Protect ? "protect" : "sweep",
	            mode == WatchMode::Poll      ? ": sweeps only at hand-overs no sweep has covered; written pages read before GPU "
	                                           "work takes them"
	            : mode == WatchMode::Protect ? ": no sweeps; write watchers protect read-only (private memory), writes fault"
	                                         : ": a sweep at every submission and satisfied WAIT_REG_MEM (as before)");
	std::fflush(stdout);
	return mode;
}
inline WatchMode GetWatchMode() noexcept {
	static const WatchMode mode = ReadWatchMode();
	return mode;
}
// (ww2, poll) The pages TakeWritesBeforeGpuWrite read, for the images over them (the buffers took them at once), and
// the pages CaptureAfterNoAccess read: both for the next SweepCpuWrites (g_captured_mutex).
std::vector<uint64_t> g_captured_images;
std::vector<uint64_t> g_captured_forced;
// (ww2, poll) The full sweeps begun (PageManager::SweepStamp): moved before a sweep reads anything.
std::atomic<uint64_t> g_sweep_stamp {0};
// (ww2, poll) Moves when a region not in the sweep's cached list gets its first write-watched page (Region::listed).
std::atomic<uint64_t> g_watch_generation {1};
// (ww2) PageManager::TakeWatchStats: since the last take.
std::atomic<uint64_t> g_stat_noaccess_pages {0}, g_stat_before_gpu_calls {0}, g_stat_before_gpu_pages {0},
    g_stat_after_protect_calls {0}, g_stat_after_protect_pages {0}, g_stat_list_rebuilds {0}, g_stat_capture_us {0};
uint64_t ElapsedUs(std::chrono::steady_clock::time_point start) noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}
// (nativemem1, KYTY_WRITE_WATCH) SweepCpuWrites' partial sweeps: KYTY_WRITE_WATCH_FULL_EVERY=N (default 1: every sweep
// reads the write watch of every region a write watcher holds, as KYTY_NATIVE_MEMORY=1 does). N > 1: only every N-th
// sweep - and any sweep KYTY_WRITE_WATCH_FULL_MS (default 8) after the last full one - is full; the others read only the
// regions some sweep found written in the last HOT_SWEEPS sweeps. A write to another region is kept by the write watch
// (no reset there) and seen by the next full sweep: until then the GPU may use its old bytes.
struct SweepConfig {
	uint32_t full_every = 1;
	uint64_t full_ms    = 8;
};
const SweepConfig& GetSweepConfig() noexcept {
	static const SweepConfig config = [] {
		SweepConfig c;
		if (const char* v = std::getenv("KYTY_WRITE_WATCH_FULL_EVERY"); v != nullptr && *v != 0)
			c.full_every = std::max<uint32_t>(1u, static_cast<uint32_t>(std::strtoul(v, nullptr, 10)));
		if (const char* v = std::getenv("KYTY_WRITE_WATCH_FULL_MS"); v != nullptr && *v != 0)
			c.full_ms = std::strtoull(v, nullptr, 10);
		if (NativeReadFaultsOn()) {
			std::printf("Write watch: on (KYTY_WRITE_WATCH=1, native memory): CPU writes from the write watch, no read-only "
			            "protection; GPU-written pages no-access (read faults kept); full sweep every %u (and after %llu ms)\n",
			            c.full_every, static_cast<unsigned long long>(c.full_ms));
			std::fflush(stdout);
			(void)GetWatchMode(); // (ww2: its line)
		}
		return c;
	}();
	return config;
}
constexpr uint64_t HOT_SWEEPS = 64;
// (nativemem1) PageManager::TakeSweepStats: since the last take (the GPU thread's sweeps only).
std::atomic<uint64_t> g_sweep_full {0}, g_sweep_partial {0}, g_sweep_queried {0}, g_sweep_regions {0};

constexpr uint64_t PAGE_SIZE    = TRACKER_PAGE_SIZE;
constexpr uint64_t REGION_SIZE  = TRACKER_REGION_SIZE;
constexpr uint64_t ADDRESS_SIZE = TRACKER_ADDRESS_SIZE;
constexpr uint64_t REGION_COUNT = ADDRESS_SIZE / REGION_SIZE;

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
// The tracker reuses Win32 memory-protection tags as internal page-state values.
// Mirror their canonical numeric values so the shared state-machine logic is identical.
constexpr uint32_t PAGE_NOACCESS  = 0x01;
constexpr uint32_t PAGE_READONLY  = 0x02;
constexpr uint32_t PAGE_READWRITE = 0x04;
#endif
constexpr uint64_t REGION_PAGES = REGION_SIZE / PAGE_SIZE;
// HasReadWatchers' filter: a bit per 64 KiB granule.
constexpr uint64_t READ_GRANULE_SIZE  = 64 * 1024;
constexpr uint64_t READ_GRANULE_PAGES = READ_GRANULE_SIZE / PAGE_SIZE;
// The lowest 64 KiB of the address space: never guest memory (UpdateRegionWatchers).
constexpr size_t NULL_PAGES = 64 * 1024 / PAGE_SIZE;

constexpr uint32_t NO_ACCESS_PROTECTION  = PAGE_NOACCESS;
constexpr uint32_t READ_ONLY_PROTECTION  = PAGE_READONLY;
constexpr uint32_t READ_WRITE_PROTECTION = PAGE_READWRITE;

// KYTY_NATIVE_MEMORY=1 (off by default): no read watcher becomes a host protection. The watcher counts and read hints
// stay (HasReadWatchers still answers for the emulator's own reads of GPU-written memory: SyncGpuCleanBacking,
// ReadFaults, TryReadGpuCleanBackingOnWatchedPage), but no protection of any kind is applied (Protect returns: 11-14).
// The game's writes are read from the write watch (SweepCpuWrites); its reads of GPU-written memory fault nowhere and
// are served only by BufferCache's eager write-back (WriteBackNativeWrites).
static bool NativeMemoryNoReadWatch() noexcept {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_NATIVE_MEMORY");
		// (nativemem1: not with KYTY_WRITE_WATCH=1 - read watchers and the offload's holds protect again -.)
		return v != nullptr && v[0] == '1' && !NativeReadFaultsOn();
	}();
	return on;
}

// The raw return addresses of a fatal end, on stderr (resolve them with the build's map: the end itself names no
// caller; 10-18 a 1-1 tour ended on "invalid range vaddr=0, size=0x400000" with nothing else).
void WriteBacktrace() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	void*      frames[32];
	const auto count = RtlCaptureStackBackTrace(0, 32, frames, nullptr);
	std::fprintf(stderr, "--- Backtrace --- (exe base %p)\n", static_cast<void*>(GetModuleHandleW(nullptr)));
	for (USHORT i = 0; i < count; ++i) std::fprintf(stderr, "  %p\n", frames[i]);
	std::fflush(stderr);
#endif
}

[[noreturn]] void FailFast(const char* reason = nullptr) noexcept {
	std::fputs("PageManager fail-fast: ", stderr);
	std::fputs(reason != nullptr ? reason : "invalid page state", stderr);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	WriteBacktrace();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(EXCEPTION_NONCONTINUABLE_EXCEPTION));
#endif
	std::_Exit(322);
}

[[noreturn]] void Fatal(const char* format, ...) {
	std::fputs("PageManager fatal: ", stderr);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	WriteBacktrace();
	std::_Exit(322);
}

Common::VirtualMemory::Mode ToMemoryMode(uint32_t protection) {
	switch (protection) {
		case NO_ACCESS_PROTECTION: return Common::VirtualMemory::Mode::NoAccess;
		case READ_ONLY_PROTECTION: return Common::VirtualMemory::Mode::Read;
		case READ_WRITE_PROTECTION: return Common::VirtualMemory::Mode::ReadWrite;
		default: Fatal("unmappable protection 0x%08" PRIx32, protection);
	}
}

// (raiz10) A page manager lock taken: free, at once; taken, a short spin, then the CPU yielded (owners call VirtualProtect
// inside: see TrackingSpinLock). (raiz9, KYTY_RAIZ) A wait is timed by thread class (the "[raiz]" line's PROTECT COST,
// "region lock") and shown to the lock watch (raiz10).
// (raiz11, KYTY_PM_LOCK_HOLDERS: default on with KYTY_RAIZ; =0 off) Who holds a lock: its taker's role and origin,
// stored with it when taken (LockTag), read by a waiter that finds it taken; and how long each role holds one.
// 20261116-052640 (merge2): the game's threads ("others") waited 127.7 times a frame, 3.57 ms, for these locks, more
// than before the granule locks (2.03) - with the put-in of kept bytes 3x dearer, done inside them -. This says by whom.
bool PmLockHoldersOn() noexcept {
	static const bool on = [] {
		const char* v   = std::getenv("KYTY_PM_LOCK_HOLDERS");
		const bool  set = Raiz::On() && !(v != nullptr && v[0] == '0');
		if (Raiz::On()) {
			std::printf("Page manager: lock holders %s (KYTY_PM_LOCK_HOLDERS%s; raiz11)\n", set ? "counted" : "not counted",
			            set ? "=0 turns it off" : "=0");
			std::fflush(stdout);
		}
		return set;
	}();
	return on;
}
struct LockTag {
	std::atomic<uint8_t>  role {0}, origin {0};
	std::atomic<uint64_t> since {0}; // (steady_clock ns at the take; 0 untimed)
};
uint32_t PmRoleNow() noexcept {
	if (Raiz::t_pm_role != Raiz::RoleOther) return Raiz::t_pm_role;
	if (PageManagerOnGpuThread()) return Raiz::RoleGpu;
	if (Raiz::t_upload_thread) return Raiz::RoleUpload;
	return Raiz::RoleOther;
}
uint64_t PmNowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
// (raiz11) A lock taken: its tag set (this thread's role and origin, the time).
void LockTagSet(LockTag* tag) noexcept {
	if (tag == nullptr || !PmLockHoldersOn()) return;
	tag->role.store(static_cast<uint8_t>(PmRoleNow()), std::memory_order_relaxed);
	tag->origin.store(static_cast<uint8_t>(Raiz::t_protect_origin < Raiz::ProtOriginCount ? Raiz::t_protect_origin : Raiz::ProtOriginCount),
	                  std::memory_order_relaxed);
	tag->since.store(PmNowNs(), std::memory_order_relaxed);
}
// (raiz11) A lock about to be let go: its time held, by its holder's role.
void LockTagRelease(LockTag* tag) noexcept {
	if (tag == nullptr || !PmLockHoldersOn()) return;
	const uint64_t since = tag->since.exchange(0, std::memory_order_relaxed);
	if (since == 0) return;
	const uint64_t now  = PmNowNs();
	const uint64_t held = now > since ? (now - since) / 1000u : 0;
	const uint32_t role = tag->role.load(std::memory_order_relaxed) % Raiz::RoleCount;
	Raiz::g_rl_hold_n[role].fetch_add(1, std::memory_order_relaxed);
	Raiz::g_rl_hold_us[role].fetch_add(held, std::memory_order_relaxed);
	if (held >= 100) Raiz::g_rl_hold_long[role].fetch_add(1, std::memory_order_relaxed);
}

void SpinLockTake(std::atomic_flag& lock, LockTag* tag = nullptr) noexcept {
	if (!lock.test_and_set(std::memory_order_acquire)) {
		LockTagSet(tag);
		return;
	}
	const bool      timed = Raiz::On();
	const auto      start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	// (raiz11: the holder found, as its tag says now.)
	const uint32_t holder_role   = tag != nullptr ? tag->role.load(std::memory_order_relaxed) % Raiz::RoleCount : Raiz::RoleOther;
	const uint32_t holder_origin = tag != nullptr ? std::min<uint32_t>(tag->origin.load(std::memory_order_relaxed), Raiz::ProtOriginCount)
	                                              : Raiz::ProtOriginCount;
	{
		LockWatch::Wait watched(LockWatch::PmGranule);
		uint32_t        spins = 0;
		do {
			do {
				if (++spins < 512) {
					_mm_pause();
				} else {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
					SwitchToThread();
#else
					sched_yield();
#endif
				}
			} while (lock.test(std::memory_order_relaxed));
		} while (lock.test_and_set(std::memory_order_acquire));
	}
	LockWatch::Release(LockWatch::PmGranule); // (only the wait is watched: the holder is not marked)
	if (timed) {
		const auto us = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
		auto& cost = Raiz::g_cost[Raiz::ClassOf(PageManagerOnGpuThread())];
		cost.rl_waits.fetch_add(1, std::memory_order_relaxed);
		cost.rl_wait_us.fetch_add(us, std::memory_order_relaxed);
		if (tag != nullptr && PmLockHoldersOn()) {
			const uint32_t waiter = PmRoleNow();
			Raiz::g_rl_wait_n[waiter][holder_role].fetch_add(1, std::memory_order_relaxed);
			Raiz::g_rl_wait_us[waiter][holder_role].fetch_add(us, std::memory_order_relaxed);
			Raiz::g_rl_wait_origin_us[holder_origin].fetch_add(us, std::memory_order_relaxed);
		}
	}
	LockTagSet(tag);
}

// (raiz10) KYTY_PM_GRANULE_LOCKS (default on; =0: one lock per 4 MiB region, as before): a lock per 64 KiB granule of a
// region, and the protections applied with only their own granules' locks held. 20261116-044058 (raiz9, KYTY_RAIZ=1): the
// game's threads waited 129 times a frame, 2.03 ms/frame, for a region's lock - held by whoever protected ANY page of the
// same 4 MiB, for as many VirtualProtect calls as its range had runs (the upload worker's re-arming: 525-897 a frame) -.
// Now a fault waits only for a holder of its own 64 KiB, for that holder's run. Locks are taken in ascending address order
// (granule after granule, region after region), as before for a region.
bool GranuleLocksOn() noexcept {
	static const bool on = [] {
		const char* v   = std::getenv("KYTY_PM_GRANULE_LOCKS");
		const bool  set = !(v != nullptr && v[0] == '0');
		Raiz::g_pm_granule_locks.store(set ? 1 : 0, std::memory_order_relaxed);
		std::printf("Page manager: a lock per %s (KYTY_PM_GRANULE_LOCKS%s)\n", set ? "64 KiB granule" : "4 MiB region",
		            set ? "=0 for one per region" : "=0");
		std::fflush(stdout);
		return set;
	}();
	return on;
}

void ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		Fatal("invalid range vaddr=0x%016" PRIx64 ", size=0x%016" PRIx64, vaddr, size);
	}
}

uint64_t PageStart(uint64_t vaddr) {
	return vaddr & ~(PAGE_SIZE - 1);
}

uint64_t PageEnd(uint64_t vaddr, uint64_t size) {
	ValidateRange(vaddr, size);
	return PageStart(vaddr + size - 1) + PAGE_SIZE;
}

// ---- KYTY_PROTECT_PAUSE ----
// Bug check 0x1A / 0x41202: Windows pages a page out while this process changes its protection. While a dangerous
// window lasts - free RAM under KYTY_PROTECT_PAUSE_FREE_MB (default 1500, read at most every 100 ms), or a zone change
// (KYTY_PROTECT_PAUSE_ZONE_MB, default 256, of texture ranges unmapped in 5 s, or KYTY_PROTECT_PAUSE_COPY_MB, default
// 768, of streamed texture copies in 5 s: NoteZoneSignal) - and KYTY_PROTECT_PAUSE_HOLD_MS (default 8000) after it, no
// write protection (read-only) is applied: its pages are kept in `resync` (and `host_rw`: their host protection stayed
// read-write, so a release of them to read-write is skipped too). The GPU thread invalidates them as CPU-written every
// KYTY_PROTECT_PAUSE_RESYNC_MS (default 1000) and when the pause ends, the skipped protections applied again first
// (GpuResourceManager::SyncProtectPause). No-access protections (read watchers, texture offload) are always applied.
struct PauseConfig {
	bool     on        = true;
	uint64_t free_mb   = 1500;
	int64_t  hold_ms   = 8000;
	uint64_t zone_mb   = 256;
	uint64_t copy_mb   = 768;
	int64_t  resync_ms = 1000;
};
uint64_t EnvU64(const char* name, uint64_t fallback) {
	const char* v = std::getenv(name);
	if (v == nullptr || v[0] < '0' || v[0] > '9') return fallback;
	return std::strtoull(v, nullptr, 10);
}
const PauseConfig& GetPauseConfig() {
	static const PauseConfig config = [] {
		PauseConfig c;
		const char* v = std::getenv("KYTY_PROTECT_PAUSE");
		c.on          = (v != nullptr && v[0] == '1'); // (11-14) off by default: froze the game (fault loop)
		// (native memory: no protection at all anyway; with KYTY_WRITE_WATCH no read-only one, all the pause skips)
		if (c.on && (NativeMemoryNoReadWatch() || NativeReadFaultsOn())) c.on = false;
		c.free_mb   = EnvU64("KYTY_PROTECT_PAUSE_FREE_MB", c.free_mb);
		c.hold_ms   = static_cast<int64_t>(EnvU64("KYTY_PROTECT_PAUSE_HOLD_MS", static_cast<uint64_t>(c.hold_ms)));
		c.zone_mb   = EnvU64("KYTY_PROTECT_PAUSE_ZONE_MB", c.zone_mb);
		c.copy_mb   = EnvU64("KYTY_PROTECT_PAUSE_COPY_MB", c.copy_mb);
		c.resync_ms = static_cast<int64_t>(EnvU64("KYTY_PROTECT_PAUSE_RESYNC_MS", static_cast<uint64_t>(c.resync_ms)));
		if (c.on) {
			std::printf("Protect pause: enabled (KYTY_PROTECT_PAUSE=0 off): free RAM < %llu MiB, or zone change (>= %llu MiB "
			            "texture ranges unmapped or >= %llu MiB streamed copies in 5 s), held %lld ms; resync every %lld ms\n",
			            static_cast<unsigned long long>(c.free_mb), static_cast<unsigned long long>(c.zone_mb),
			            static_cast<unsigned long long>(c.copy_mb), static_cast<long long>(c.hold_ms),
			            static_cast<long long>(c.resync_ms));
		} else {
			std::printf("Protect pause: disabled (%s)\n", NativeMemoryNoReadWatch() ? "KYTY_NATIVE_MEMORY=1" : "KYTY_PROTECT_PAUSE=0");
		}
		std::fflush(stdout);
		return c;
	}();
	return config;
}

int64_t PauseNowMs() noexcept {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

using RangeMap = std::map<uint64_t, uint64_t>; // begin -> end, disjoint, touching ranges merged

void RangeAdd(RangeMap& m, uint64_t begin, uint64_t end) {
	auto it = m.upper_bound(begin);
	if (it != m.begin()) {
		auto prev = std::prev(it);
		if (prev->second >= begin) {
			begin = prev->first;
			end   = std::max(end, prev->second);
			it    = m.erase(prev);
		}
	}
	while (it != m.end() && it->first <= end) {
		end = std::max(end, it->second);
		it  = m.erase(it);
	}
	m.emplace(begin, end);
}

bool RangeContains(const RangeMap& m, uint64_t begin, uint64_t end) {
	auto it = m.upper_bound(begin);
	if (it == m.begin()) return false;
	--it;
	return it->first <= begin && it->second >= end;
}

void RangeSubtract(RangeMap& m, uint64_t begin, uint64_t end) {
	auto it = m.upper_bound(begin);
	if (it != m.begin()) --it;
	while (it != m.end() && it->first < end) {
		const auto b = it->first;
		const auto e = it->second;
		if (e <= begin) {
			++it;
			continue;
		}
		it = m.erase(it);
		if (b < begin) m.emplace(b, begin);
		if (e > end) {
			m.emplace(end, e);
			break;
		}
	}
}

struct PauseState {
	std::mutex mutex;
	RangeMap   resync;
	RangeMap   host_rw;
	bool       active   = false;
	int64_t    until_ms = 0, last_resync_ms = 0;
	uint64_t   skipped = 0, applied = 0, pages = 0;
	// The zone signal's 5 s window.
	int64_t  window_ms = 0;
	uint64_t window_unmapped = 0, window_copied = 0;
};
PauseState& GetPause() {
	static auto* state = new PauseState(); // (never destroyed: other threads may protect during exit)
	return *state;
}
std::atomic<bool>    g_pause_active {false};
std::atomic<int64_t> g_pause_ram_poll_ms {0};

// Under the pause's lock.
void PauseEnterLocked(PauseState& p, int64_t now, const char* reason) {
	const auto& c = GetPauseConfig();
	p.until_ms    = std::max(p.until_ms, now + c.hold_ms);
	if (p.active) return;
	p.active         = true;
	p.last_resync_ms = now;
	p.skipped        = 0;
	p.applied        = 0;
	p.pages          = 0;
	g_pause_active.store(true, std::memory_order_release);
	std::printf("Protect pause: on (%s)\n", reason);
	std::fflush(stdout);
}

void PausePollRam(int64_t now) noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	auto last = g_pause_ram_poll_ms.load(std::memory_order_relaxed);
	if (now - last < 100) return;
	if (!g_pause_ram_poll_ms.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	if (GlobalMemoryStatusEx(&status) == 0) return;
	const uint64_t free_mb = static_cast<uint64_t>(status.ullAvailPhys) / (1024u * 1024u);
	if (free_mb >= GetPauseConfig().free_mb) return;
	char reason[96];
	std::snprintf(reason, sizeof(reason), "low RAM: %llu MiB free", static_cast<unsigned long long>(free_mb));
	auto&           p = GetPause();
	std::lock_guard lock(p.mutex);
	PauseEnterLocked(p, now, reason);
#else
	(void)now;
#endif
}

// ProtectHost's filter: true when the protection is skipped (the pause owns it).
bool PauseSkips(uint64_t vaddr, uint64_t size, uint32_t protection) noexcept {
	if (!GetPauseConfig().on) return false;
	PausePollRam(PauseNowMs());
	if (!g_pause_active.load(std::memory_order_acquire)) return false;
	auto&           p = GetPause();
	std::lock_guard lock(p.mutex);
	if (!p.active) return false;
	const auto end = vaddr + size;
	if (protection == READ_ONLY_PROTECTION) {
		RangeAdd(p.resync, vaddr, end);
		RangeAdd(p.host_rw, vaddr, end);
		p.skipped++;
		return true;
	}
	if (protection == READ_WRITE_PROTECTION) {
		// (Pages the pause left read-write on the host: nothing to release.)
		if (RangeContains(p.host_rw, vaddr, end)) {
			p.skipped++;
			return true;
		}
	} else {
		RangeSubtract(p.host_rw, vaddr, end); // (no access now: a later release must be applied)
	}
	p.applied++;
	return false;
}

// ---- (raiz8) KYTY_RAIZ_ASYNC_RELEASE (raiz.h): the GPU thread's read releases left to others ----------------------
// Whether this thread's read releases are left to others: the GPU thread's only (a fault's own thread, a guest thread
// or the release thread apply them at once: the access that waits for them is theirs).
bool DeferReleasesHere() noexcept {
	return NativeReadFaultsOn() && Raiz::AsyncReleaseOn() && PageManagerOnGpuThread();
}
// (raiz11) KYTY_RAIZ_RELEASE_LAZY=1: the GPU thread's deferred releases left for whoever needs them (DeferRelease).
// (sync1, 11-23, KYTY_RAIZ_RELEASE_RUN_GRANULE, on by default; =0 as before) The release thread's runs of deferred releases
// end at its granule's lock: a run spanning several 64 KiB granules held all their locks during its VirtualProtect and the
// kept bytes' put-in (log 20261123-045950: the release thread held page manager locks 27 ms/frame, 60 holds >= 100 us a
// frame; the game thread applying its own ranges waited 2.84 ms/frame on it and the GPU thread 0.55 ms/frame). Same pages,
// same protections, only cut where a granule ends (as runs already are when a page in between is not marked).
bool ReleaseRunGranuleOn() noexcept {
	static const bool on = [] {
		const char* v  = std::getenv("KYTY_RAIZ_RELEASE_RUN_GRANULE");
		const bool  ok = !(v != nullptr && v[0] == '0');
		if (Raiz::On())
			std::printf("Raiz: release thread runs %s (KYTY_RAIZ_RELEASE_RUN_GRANULE%s; sync1)\n",
			            ok ? "cut at each granule lock (shorter lock holds)" : "as long as the marked pages go (as before)", ok ? "=0: as before" : "=1 to cut them");
		return ok;
	}();
	return on;
}

// (sync2, 11-23, KYTY_RAIZ_APPLY_SHORT_HOLDS: on by default; =0 as before; =2 on + verified) A deferred-release walk
// (ApplyRegionDeferred: guest thread, release thread, fault) holds only the granule locks of the run it is protecting.
// Before: once past a run it kept its last granule's lock while scanning on, and reaching the next marked page took EVERY
// lock in between (RegionLocker::Reach), granules with nothing to do included; and the guest thread's runs went on over
// several granules, each lock held for the whole VirtualProtect and taken while waiting on the next one. Log
// 20261123-063653: "held by guest apply" 20-34 ms/frame, 64-90 holds >= 100 us a frame, for 1.85 ms/frame of guest
// applies (ASYNC RELEASE) - the locks were held while waiting, not while working -; the release thread's own cut (sync1)
// could not change that. Now: the locks behind the walk go before a later granule's is taken (ascending order kept: no
// deadlock), and the guest thread's runs end at a granule's lock as the release thread's do. Same pages, same protections.
// =2: at each run's VirtualProtect the locks held are checked to be the run's granules' only; a breach printed ("Raiz: VERIFY sync2").
uint32_t ApplyShortHoldsMode() noexcept {
	static const uint32_t mode = [] {
		const char*    v = std::getenv("KYTY_RAIZ_APPLY_SHORT_HOLDS");
		const uint32_t m = v != nullptr && v[0] == '0' ? 0u : (v != nullptr && v[0] == '2' ? 2u : 1u);
		if (Raiz::On()) {
			std::printf("Raiz: deferred-release walks %s (KYTY_RAIZ_APPLY_SHORT_HOLDS=%u; sync2)\n",
			            m == 0 ? "hold their locks as before" : m == 2 ? "hold only their run's granule locks, VERIFIED" : "hold only their run's granule locks",
			            m);
			std::fflush(stdout);
		}
		return m;
	}();
	return mode;
}

bool ReleaseLazyOn() noexcept {
	static const bool on = [] {
		const char* v   = std::getenv("KYTY_RAIZ_RELEASE_LAZY");
		const bool  set = Raiz::On() && v != nullptr && v[0] == '1';
		if (Raiz::On()) {
			std::printf("Raiz: deferred releases %s (KYTY_RAIZ_RELEASE_LAZY%s; raiz11)\n",
			            set ? "LEFT for a fault, the guest thread waiting or the GPU watching again - the release thread off -"
			                : "applied by the release thread as well",
			            set ? "=1, experiment" : "=1 for the experiment");
			std::fflush(stdout);
		}
		return set;
	}();
	return on;
}
// The release thread's queue: [begin, end) ranges whose pages were marked (Region::deferred). The thread applies what
// is still marked of each (a guest thread or a fault may have done it first: nothing then).
struct ReleaseQueue {
	std::mutex                mutex;
	std::condition_variable   ready;
	std::vector<PendingRange> ranges;
	bool                      started = false;
	// The page manager the thread applies to (PageManager's constructor and destructor; the destructor waits, under
	// apply_mutex, for a batch under way).
	std::atomic<PageManager*> owner {nullptr};
	std::mutex                apply_mutex;
	size_t                    max_ranges = 0;
};
ReleaseQueue& GetReleaseQueue() {
	static auto* queue = new ReleaseQueue(); // (never destroyed: its thread may outlive static destructors)
	return *queue;
}
void ReleaseThreadLoop();
// (Under a region's lock: the queue's mutex is a leaf.)
void QueueRelease(uint64_t begin, uint64_t end) {
	auto& q     = GetReleaseQueue();
	bool  wake  = false;
	bool  start = false;
	{
		std::lock_guard lock(q.mutex);
		if (!q.ranges.empty() && q.ranges.back().end == begin) {
			q.ranges.back().end = end;
		} else {
			wake = q.ranges.empty();
			q.ranges.push_back({begin, end});
		}
		q.max_ranges = std::max(q.max_ranges, q.ranges.size());
		if (!q.started) {
			q.started = true;
			start     = true;
		}
	}
	if (start) std::thread(ReleaseThreadLoop).detach();
	if (wake) q.ready.notify_one();
}

} // namespace

struct PageManager::Impl {
	struct PageState {
		uint8_t write_watchers  : 7 = 0;
		uint8_t access_watchers : 1 = 0;

		[[nodiscard]] uint32_t Perms() const noexcept {
			if (access_watchers != 0) {
				return NO_ACCESS_PROTECTION;
			}
			// (KYTY_NATIVE_MEMORY: write watchers keep the count only; the write watch sees the writes, SweepCpuWrites.)
			// (ww2, KYTY_WRITE_WATCH_MODE=protect: read-only again, on native memory; no sweep reads the write watch then.)
			if (write_watchers != 0 && (!g_native_write_watch || GetWatchMode() == WatchMode::Protect)) {
				return READ_ONLY_PROTECTION;
			}
			return READ_WRITE_PROTECTION;
		}

		template <int delta, bool is_read>
		uint32_t AddDelta(uint64_t address) {
			static_assert(delta >= -1 && delta <= 1);
			if constexpr (is_read) {
				if constexpr (delta == 1) {
					if (access_watchers != 0) {
						Fatal("read-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++access_watchers;
				} else if constexpr (delta == -1) {
					if (access_watchers == 0) {
						Fatal("read-watcher underflow at 0x%016" PRIx64, address);
					}
					return --access_watchers;
				} else {
					return access_watchers;
				}
			} else {
				if constexpr (delta == 1) {
					if (write_watchers == 0x7f) {
						Fatal("write-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++write_watchers;
				} else if constexpr (delta == -1) {
					if (write_watchers == 0) {
						Fatal("write-watcher underflow at 0x%016" PRIx64, address);
					}
					return --write_watchers;
				} else {
					return write_watchers;
				}
			}
		}
	};
	static_assert(sizeof(PageState) == 1);

	// (raiz10) A lock per 64 KiB granule (the read granule: SyncReadGranules recomputes a granule's bit under its lock);
	// KYTY_PM_GRANULE_LOCKS=0: every granule uses lock 0 (one per region, as before).
	static constexpr size_t LOCK_GRANULES = REGION_PAGES / READ_GRANULE_PAGES;
	static_assert(LOCK_GRANULES == 64 && REGION_PAGES % READ_GRANULE_PAGES == 0);

	struct Region {
		std::array<std::atomic_flag, LOCK_GRANULES> locks; // (raiz10; clear: C++20's default)
		std::array<LockTag, LOCK_GRANULES>          tags;  // (raiz11: who holds each, KYTY_PM_LOCK_HOLDERS)
		std::array<PageState, REGION_PAGES> pages;
		// Nonzero while a page has read watchers (HasReadWatchers).
		std::array<std::atomic<uint8_t>, REGION_PAGES> read_hint {};
		// The texture offload's holds of a page (src/local/texture-offload.h: its guest bytes dropped from RAM): no access
		// while any lasts, whatever its watchers. A count of its own: as a read watcher it took the one read bit a page
		// has, and the GPU's tracking of a GPU write over such a texture's memory (a pool reused while a level loads)
		// ended the emulator with "read-watcher overflow" (issue #91, an RTX 3050 with 16 GB of RAM). (Atomic: WatchedRuns
		// reads without the lock.)
		std::array<std::atomic<uint8_t>, REGION_PAGES> offload {};
		// (raiz8, KYTY_RAIZ_ASYNC_RELEASE) Pages whose read release the GPU thread left to others: their host protection
		// is still no access (stricter than their state), their read hint still set. Set and cleared under the lock; read
		// without it by the fault path (a hint: rechecked under the lock). `deferred_count`: how many are set.
		std::array<std::atomic<uint8_t>, REGION_PAGES> deferred {};
		std::atomic<uint32_t>                          deferred_count {0};
		// (nativemem1) Pages a write watcher holds (changed under the lock, read without it: SweepCpuWrites), and the
		// last sweep that found one of the region's pages written (+1; the sweeping thread's only).
		std::atomic<uint32_t> write_pages {0};
		std::atomic<uint64_t> hot_sweep {0};
		// (ww2, poll) In the sweep's cached list of regions (SweepList). Read after write_pages is raised, cleared before
		// write_pages is read again (both sequentially consistent): a region that gets its first write-watched page is in
		// the list or moves g_watch_generation, never neither.
		std::atomic<uint8_t> listed {0};
	};

	// A page's protection: its watchers', no access while the offload holds it.
	static uint32_t PermsOf(const Region& region, size_t page) noexcept {
		return region.offload[page].load(std::memory_order_relaxed) != 0 ? NO_ACCESS_PROTECTION : region.pages[page].Perms();
	}

	// (raiz10) The lock a page of a region is under (REGION_PAGES: one past the last), and the first page under lock `g`.
	static size_t LockOf(size_t page) noexcept { return GranuleLocksOn() ? page / READ_GRANULE_PAGES : 0; }
	static size_t LockFirstPage(size_t g) noexcept { return GranuleLocksOn() ? g * READ_GRANULE_PAGES : 0; }

	// (raiz10) The locks of one region a walk in ascending page order holds: [lo, hi], taken as the walk reaches their
	// pages (Reach), let go once the walk is past them (ReleaseBelow: every lock under the page's; `done(first, end)` told
	// the pages of each lock let go, first). One region at a time, never a lower lock after a higher one: no deadlock.
	class RegionLocker final {
	public:
		explicit RegionLocker(Region& region): m_region(region) {}
		~RegionLocker() { ReleaseBelow(REGION_PAGES); }
		KYTY_CLASS_NO_COPY(RegionLocker);

		// The lock of `page` held (and every one between the highest held and it). A page under a lock held: nothing.
		void Reach(size_t page) noexcept {
			const size_t g = LockOf(page);
			if (m_held && g <= m_hi) return;
			for (size_t i = m_held ? m_hi + 1 : g; i <= g; ++i) SpinLockTake(m_region.locks[i], &m_region.tags[i]);
			if (!m_held) m_lo = g;
			m_hi   = g;
			m_held = true;
		}
		// The locks under `page`'s let go (all of them past the region's end).
		void ReleaseBelow(size_t page) noexcept {
			ReleaseBelow(page, [](size_t, size_t) {});
		}
		template <typename Done>
		void ReleaseBelow(size_t page, Done&& done) noexcept {
			const size_t g = page >= REGION_PAGES ? LOCK_GRANULES : LockOf(page);
			while (m_held && m_lo < g) {
				const size_t from = LockFirstPage(m_lo);
				const size_t to   = GranuleLocksOn() ? from + READ_GRANULE_PAGES : REGION_PAGES;
				done(from, to);
				LockTagRelease(&m_region.tags[m_lo]); // (raiz11)
				m_region.locks[m_lo].clear(std::memory_order_release);
				if (m_lo == m_hi) {
					m_held = false;
					break;
				}
				++m_lo;
			}
		}
		[[nodiscard]] size_t HeldLo() const noexcept { return m_held ? m_lo : static_cast<size_t>(-1); } // (sync2: the verify)
		[[nodiscard]] size_t HeldHi() const noexcept { return m_held ? m_hi : static_cast<size_t>(-1); }
		[[nodiscard]] bool Holds(size_t page) const noexcept { return m_held && LockOf(page) >= m_lo && LockOf(page) <= m_hi; }

	private:
		Region& m_region;
		size_t  m_lo = 0, m_hi = 0;
		bool    m_held = false;
	};

	// (raiz10) One page's lock, alone.
	class PageLock final {
	public:
		PageLock(Region& region, size_t page): m_lock(region.locks[LockOf(page)]), m_tag(&region.tags[LockOf(page)]) {
			SpinLockTake(m_lock, m_tag);
		}
		~PageLock() {
			LockTagRelease(m_tag); // (raiz11)
			m_lock.clear(std::memory_order_release);
		}
		KYTY_CLASS_NO_COPY(PageLock);

	private:
		std::atomic_flag& m_lock;
		LockTag*          m_tag;
	};

	Impl() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		if (info.dwPageSize != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32,
			      static_cast<uint32_t>(info.dwPageSize));
		}
#elif defined(__APPLE__)
		// Under Rosetta the host page size is 4 KB, matching TRACKER_PAGE_SIZE.
		if (static_cast<uint64_t>(getpagesize()) != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32, static_cast<uint32_t>(getpagesize()));
		}
#else
		const auto host_page_size = ::sysconf(_SC_PAGESIZE);
		if (host_page_size < 0 || static_cast<uint64_t>(host_page_size) != PAGE_SIZE) {
			Fatal("unsupported host page size %ld", static_cast<long>(host_page_size));
		}
#endif
		(void)GetPauseConfig(); // (its start-up line)
		regions = std::make_unique<std::atomic<Region*>[]>(REGION_COUNT);
		for (uint64_t i = 0; i < REGION_COUNT; i++) {
			regions[i].store(nullptr, std::memory_order_relaxed);
		}
	}

	~Impl() {
		for (const auto& region: region_storage) {
			RegionLocker lock(*region); // (raiz10: all of its locks)
			lock.Reach(0);
			lock.Reach(REGION_PAGES - 1);
			for (size_t page = 0; page < REGION_PAGES; page++) {
				if (PermsOf(*region, page) != READ_WRITE_PROTECTION) {
					FailFast("PageManager destroyed with live page state");
				}
			}
		}
	}

	Region* FindRegion(uint64_t vaddr) const noexcept {
		return vaddr < ADDRESS_SIZE ? regions[vaddr / REGION_SIZE].load(std::memory_order_acquire)
		                            : nullptr;
	}

	Region* GetOrCreateRegion(uint64_t vaddr) {
		const auto index = vaddr / REGION_SIZE;
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		std::lock_guard lock(region_mutex);
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		auto  region = std::make_unique<Region>();
		auto* ptr    = region.get();
		region_storage.push_back(std::move(region));
		region_list.emplace_back(index * REGION_SIZE, ptr);
		regions[index].store(ptr, std::memory_order_release);
		return ptr;
	}

	static void ProtectHost(uint64_t vaddr, uint64_t size, uint32_t protection) noexcept {
		if (PauseSkips(vaddr, size, protection)) return; // (KYTY_PROTECT_PAUSE)
		const LoteTimer lote_timer; // (PROT-LOTE: the render thread's own calls, counted and timed)
		if (!Libs::LibKernel::Memory::ProtectGuestHostMemory(vaddr, size,
		                                                     ToMemoryMode(protection))) {
			Fatal("address-space protection failed at 0x%016" PRIx64 ", new=0x%08" PRIx32, vaddr,
			      protection);
		}
		if (protection == NO_ACCESS_PROTECTION && GetWatchMode() == WatchMode::Poll) CaptureAfterNoAccess(vaddr, size);
	}

	// (ww2, poll) Right after pages became no-access: what the write watch holds of them is read (and reset) now and
	// kept for the next sweep, which takes those pages as written whatever their state then. No game write lands there
	// from here on without faulting, and TakeWritesBeforeGpuWrite read the range before the buffers made it GPU-written:
	// what is left is a write made between the two (the upload of the range's dirty pages; the wait of a deferred
	// protection, DeferReadProtection). With write faults that write found the page read-only and came after the GPU's
	// (read back first); the sweep's invalidation does the same (GpuResourceManager::InvalidateCpuWrittenPages).
	// Before, such a page was no-access at the next sweep and its write was dropped. (The address space's lock inside,
	// as the protection just applied took it; then a leaf mutex.)
	static void CaptureAfterNoAccess(uint64_t vaddr, uint64_t size) noexcept {
		thread_local std::vector<uint64_t> pages;
		pages.clear();
		const auto start = std::chrono::steady_clock::now();
		(void)Libs::LibKernel::Memory::NativeWriteWatch(vaddr, size, &CollectWritten, &pages, true);
		g_stat_capture_us.fetch_add(ElapsedUs(start), std::memory_order_relaxed);
		g_stat_after_protect_calls.fetch_add(1, std::memory_order_relaxed);
		if (pages.empty()) return;
		g_stat_after_protect_pages.fetch_add(pages.size(), std::memory_order_relaxed);
		std::lock_guard lock(g_captured_mutex);
		g_captured_forced.insert(g_captured_forced.end(), pages.begin(), pages.end());
	}

	static void CollectWritten(uint64_t page_vaddr, void* user) {
		static_cast<std::vector<uint64_t>*>(user)->push_back(page_vaddr);
	}

	// ---- (PROT-LOTE) KYTY_PROT_LOTE=1 (off by default) --------------------------------------------------------------------
	// Log 20261127-014257-039 (Latria), [raiz] PROTECT COST: the render thread makes 323 VirtualProtect a frame, 0.94 ms;
	// by origin read-flush 121 calls / 1,488 pages 0.66 ms (no-access of what GPU work writes: ~0.44 us a page, the
	// working-set drop), write-release 188 calls 0.26 ms, write-track 26 calls 0.09 ms. KYTY_FLUSH_ASYNC moved the
	// read-flush to a worker but joined it at EVERY later protection of the render thread (write-release/track,
	// ~210/frame) and before every hand-off: the render thread waited for it (gain 0, ESTADO).
	// Here: one worker per producer thread with a FIFO of batches (sequence numbers). A hand-off never waits. A later
	// protection of the render thread waits only for the batches that cover its own pages (overlap); a sync point (label,
	// memory write, wait, guest command, submit: FlushReadProtect) waits for all of them, as before: what the guest can
	// observe is unchanged (the no-access is in effect before anything tells the guest the GPU work is done). The pages
	// stay deferred until then exactly as DeferReadProtection already left them pending until the next flush.
	struct LoteFlusher {
		std::mutex                                                   mutex;
		std::condition_variable                                      cv;
		std::vector<std::pair<uint64_t, std::vector<PendingRange>>> queue;
		std::atomic<uint64_t>                                        done {0};
		// (producer only)
		uint64_t                                         issued = 0;
		std::vector<std::pair<PendingRange, uint64_t>> inflight;
		LoteFlusher() {
			std::thread([this] {
				for (;;) {
					std::vector<std::pair<uint64_t, std::vector<PendingRange>>> work;
					{
						std::unique_lock lock(mutex);
						cv.wait(lock, [&] { return !queue.empty(); });
						work.swap(queue);
					}
					for (auto& [seq, ranges]: work) {
						const auto start = std::chrono::steady_clock::now();
						{
							const Raiz::OriginScope origin(Raiz::ProtReadFlush);
							for (const auto& range: ranges) ProtectHost(range.begin, range.end - range.begin, NO_ACCESS_PROTECTION);
						}
						g_lote_apply_ns.fetch_add(ElapsedNs(start), std::memory_order_relaxed);
						g_lote_applied_ranges.fetch_add(ranges.size(), std::memory_order_relaxed);
						done.store(seq, std::memory_order_release);
						done.notify_all();
					}
				}
			}).detach();
		}
	};
	static inline std::atomic<uint64_t> g_lote_apply_ns {0}, g_lote_applied_ranges {0}, g_lote_batches {0}, g_lote_ranges {0},
	    g_lote_overlap_waits {0}, g_lote_overlap_ns {0}, g_lote_sync_waits {0}, g_lote_sync_ns {0}, g_lote_sync_flushes {0},
	    g_lote_rt_calls {0}, g_lote_rt_ns {0};
	// (PROT-LOTE v2) Where each sync wait came from: 0 other, 1 guest command loop, 2 submit, 0x100 + PM4 opcode.
	static inline std::array<std::atomic<uint64_t>, 0x200> g_lote_site_waits {}, g_lote_site_ns {}, g_lote_site_flushes {};
	static inline thread_local uint32_t     t_lote_site     = 0;
	static inline thread_local LoteFlusher* t_lote          = nullptr;
	static inline thread_local bool         t_lote_producer = false; // (a thread that defers read protections: the render thread)

	static uint64_t ElapsedNs(std::chrono::steady_clock::time_point start) noexcept {
		return static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
	}
	static bool LoteOn() noexcept {
		static const bool on = [] {
			const char* v = std::getenv("KYTY_PROT_LOTE");
			const bool  r = v != nullptr && v[0] == '1';
			std::printf("[prot-lote] KYTY_PROT_LOTE=%d: %s\n", r ? 1 : 0,
			            r ? "ENCENDIDO (el sin-acceso de lo que escribe la GPU lo aplica un hilo aparte; el hilo de render solo "
			                "espera en etiquetas/esperas/envios o si toca esas mismas paginas)"
			              : "apagado");
			std::fflush(stdout);
			return r;
		}();
		return on;
	}
	struct LoteTimer {
		std::chrono::steady_clock::time_point start;
		bool                                  on;
		LoteTimer() noexcept: on(t_lote_producer && LoteOn()) {
			if (on) start = std::chrono::steady_clock::now();
		}
		~LoteTimer() {
			if (!on) return;
			g_lote_rt_calls.fetch_add(1, std::memory_order_relaxed);
			g_lote_rt_ns.fetch_add(ElapsedNs(start), std::memory_order_relaxed);
		}
	};
	static LoteFlusher& Lote() noexcept {
		if (t_lote == nullptr) t_lote = new LoteFlusher(); // (never destroyed: the worker lives as long as the process)
		return *t_lote;
	}
	static void LoteWaitSeq(LoteFlusher& f, uint64_t seq) noexcept {
		for (int spin = 0; spin < 256 && f.done.load(std::memory_order_acquire) < seq; ++spin) _mm_pause();
		for (uint64_t now = f.done.load(std::memory_order_acquire); now < seq; now = f.done.load(std::memory_order_acquire))
			f.done.wait(now, std::memory_order_acquire);
	}
	static void LotePrune(LoteFlusher& f) noexcept {
		const auto done = f.done.load(std::memory_order_acquire);
		if (f.inflight.empty() || done < f.inflight.front().second) return;
		std::erase_if(f.inflight, [done](const auto& e) { return e.second <= done; });
	}
	static void LoteWaitOverlapping(uint64_t begin, uint64_t end) noexcept {
		if (t_lote == nullptr) return;
		auto& f = *t_lote;
		LotePrune(f);
		uint64_t need = 0;
		for (const auto& [range, seq]: f.inflight)
			if (range.begin < end && begin < range.end) need = std::max(need, seq);
		if (need == 0 || f.done.load(std::memory_order_acquire) >= need) return;
		const auto start = std::chrono::steady_clock::now();
		LoteWaitSeq(f, need);
		g_lote_overlap_waits.fetch_add(1, std::memory_order_relaxed);
		g_lote_overlap_ns.fetch_add(ElapsedNs(start), std::memory_order_relaxed);
		LotePrune(f);
	}
	static void LoteHandOff() noexcept {
		t_lote_producer = true;
		LoteReport();
		if (g_read_protects.empty()) return;
		auto&      f   = Lote();
		const auto seq = ++f.issued;
		for (const auto& range: g_read_protects) f.inflight.emplace_back(range, seq);
		g_lote_batches.fetch_add(1, std::memory_order_relaxed);
		g_lote_ranges.fetch_add(g_read_protects.size(), std::memory_order_relaxed);
		{
			std::lock_guard lock(f.mutex);
			f.queue.emplace_back(seq, std::move(g_read_protects));
		}
		f.cv.notify_one();
		g_read_protects.clear();
	}
	// A sync point: the pending ranges applied here (as before) after every batch in flight (order kept).
	static void LoteSyncFlush() noexcept {
		t_lote_producer = true;
		const uint32_t site = t_lote_site & 0x1ffu;
		t_lote_site         = 0;
		if (t_lote != nullptr && t_lote->done.load(std::memory_order_acquire) < t_lote->issued) {
			auto&      f     = *t_lote;
			const auto start = std::chrono::steady_clock::now();
			LoteWaitSeq(f, f.issued);
			const auto ns = ElapsedNs(start);
			g_lote_sync_waits.fetch_add(1, std::memory_order_relaxed);
			g_lote_sync_ns.fetch_add(ns, std::memory_order_relaxed);
			g_lote_site_waits[site].fetch_add(1, std::memory_order_relaxed);
			g_lote_site_ns[site].fetch_add(ns, std::memory_order_relaxed);
		}
		if (t_lote != nullptr) t_lote->inflight.clear();
		if (!g_read_protects.empty()) {
			g_lote_sync_flushes.fetch_add(1, std::memory_order_relaxed);
			g_lote_site_flushes[site].fetch_add(1, std::memory_order_relaxed);
		}
		const Raiz::OriginScope origin(Raiz::ProtReadFlush); // (raiz8)
		for (const auto& range: g_read_protects) ProtectHost(range.begin, range.end - range.begin, NO_ACCESS_PROTECTION);
		g_read_protects.clear();
	}
	// One line every 5 s: the render thread's own VirtualProtect (calls and ms per second; per frame = / fps) and its
	// waits for the worker; the worker's work moved off the render thread.
	static void LoteReport() noexcept {
		static thread_local auto last = std::chrono::steady_clock::now();
		const auto               now  = std::chrono::steady_clock::now();
		if (now - last < std::chrono::seconds(5)) return;
		const double s = std::chrono::duration<double>(now - last).count();
		last           = now;
		const auto x   = [s](std::atomic<uint64_t>& a) { return static_cast<double>(a.exchange(0, std::memory_order_relaxed)) / s; };
		const double calls = x(g_lote_rt_calls), rt_ms = x(g_lote_rt_ns) / 1e6, b = x(g_lote_batches), r = x(g_lote_ranges);
		const double ow = x(g_lote_overlap_waits), ow_ms = x(g_lote_overlap_ns) / 1e6, sw = x(g_lote_sync_waits),
		             sw_ms = x(g_lote_sync_ns) / 1e6, sf = x(g_lote_sync_flushes), ap_ms = x(g_lote_apply_ns) / 1e6,
		             ar = x(g_lote_applied_ranges);
		std::printf("[prot-lote] %.1f s, hilo de render por segundo: VirtualProtect %.0f llamadas %.2f ms; esperas al hilo "
		            "aparte: por solape %.0f (%.2f ms), en puntos de sincronia %.0f (%.2f ms); lotes entregados %.0f (%.0f rangos), "
		            "flushes que siguen en el hilo de render %.0f; hilo aparte aplico %.0f rangos %.2f ms (fuera del hilo de "
		            "render). Por fotograma: dividir por los fps (y comparar con [raiz] PROTECT COST GPU thread)\n",
		            s, calls, rt_ms, ow, ow_ms, sw, sw_ms, b, r, sf, ar, ap_ms);
		// (v2) The sync points that waited or still flushed on the render thread, the 6 costliest (per second).
		std::array<std::pair<uint64_t, uint32_t>, 0x200> by {};
		uint64_t                                         waits[0x200], flushes[0x200];
		for (uint32_t i = 0; i < 0x200; ++i) {
			waits[i]   = g_lote_site_waits[i].exchange(0, std::memory_order_relaxed);
			flushes[i] = g_lote_site_flushes[i].exchange(0, std::memory_order_relaxed);
			by[i]      = {g_lote_site_ns[i].exchange(0, std::memory_order_relaxed) + flushes[i], i};
		}
		std::sort(by.begin(), by.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
		std::printf("[prot-lote] puntos de sincronia (esperas/s ms/s flushes/s):");
		for (int k = 0; k < 6 && by[k].first != 0; ++k) {
			const uint32_t i = by[k].second;
			if (i >= 0x100) std::printf(" pm4 0x%02x", i - 0x100);
			else std::printf(" %s", i == 1 ? "comando" : i == 2 ? "envio" : "otro");
			std::printf(" %.0f %.2f %.0f;", static_cast<double>(waits[i]) / s,
			            static_cast<double>(by[k].first - flushes[i]) / 1e6 / s, static_cast<double>(flushes[i]) / s);
		}
		std::printf("\n");
		std::fflush(stdout);
	}

	static void FlushReadProtect() noexcept {
		if (LoteOn()) {
			LoteSyncFlush();
			return;
		}
		AsyncReadFlushJoin(); // (KYTY_FLUSH_ASYNC: a batch handed off before comes first)
		NoteSyncFlush();
		const Raiz::OriginScope origin(Raiz::ProtReadFlush); // (raiz8)
		for (const auto& range: g_read_protects) ProtectHost(range.begin, range.end - range.begin, NO_ACCESS_PROTECTION);
		g_read_protects.clear();
	}

	// ---- (FLUSH-ASYNC) KYTY_FLUSH_ASYNC=1 (off by default) -------------------------------------------------------------
	// Log 20261124-084052-337, [raiz] PROTECT BY ORIGIN: the GPU (render) thread spends ~1.0 ms a frame (143 calls,
	// ~2,500 pages) applying its deferred no-access ranges ("read-flush"), ~60% of its 1.69 ms of protection changes, on
	// a CPU-bound frame (cpu 12.8 of 15.8 ms, gpu_wait ~0). Many flushes come before packets that let the guest learn
	// nothing (NOP, EVENT_WRITE, ACQUIRE_MEM, INDIRECT_BUFFER...): there the batch goes to a worker thread and the render
	// thread goes on translating. Before a label, a memory write, a wait, a guest command or any other protection change
	// of the render thread the batch is joined (applied first): the order the guest can observe is unchanged.
	static bool AsyncReadFlushOn() noexcept {
		static const bool on = [] {
			const char* v = std::getenv("KYTY_FLUSH_ASYNC");
			const bool  r = v != nullptr && v[0] == '1';
			std::printf("[flush-async] KYTY_FLUSH_ASYNC=%d: %s\n", r ? 1 : 0,
			            r ? "ENCENDIDO (la proteccion no-access de lo que escribe la GPU se aplica en un hilo aparte)" : "apagado");
			std::fflush(stdout);
			return r;
		}();
		return on;
	}
	struct AsyncReadFlush {
		std::mutex                mutex;
		std::condition_variable   cv;
		std::vector<PendingRange> batch;
		std::atomic<bool>         busy {false};
		std::atomic<uint64_t>     batches {0}, ranges {0}, joins_waited {0}, join_wait_ns {0}, apply_ns {0}, sync_flushes {0};
		AsyncReadFlush() {
			std::thread([this] {
				for (;;) {
					std::vector<PendingRange> work;
					{
						std::unique_lock lock(mutex);
						cv.wait(lock, [&] { return !batch.empty(); });
						work.swap(batch);
					}
					const auto start = std::chrono::steady_clock::now();
					{
						const Raiz::OriginScope origin(Raiz::ProtReadFlush);
						for (const auto& range: work) ProtectHost(range.begin, range.end - range.begin, NO_ACCESS_PROTECTION);
					}
					apply_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
					                                             std::chrono::steady_clock::now() - start)
					                                             .count()),
					                   std::memory_order_relaxed);
					busy.store(false, std::memory_order_release);
					busy.notify_all();
				}
			}).detach();
		}
	};
	static AsyncReadFlush& AsyncFlusher() noexcept {
		static auto* flusher = new AsyncReadFlush(); // (never destroyed: the worker lives as long as the process)
		return *flusher;
	}
	static void AsyncReadFlushJoin() noexcept {
		if (!AsyncReadFlushOn()) return;
		auto& f = AsyncFlusher();
		if (!f.busy.load(std::memory_order_acquire)) return;
		const auto start = std::chrono::steady_clock::now();
		for (int spin = 0; spin < 2000 && f.busy.load(std::memory_order_acquire); ++spin) _mm_pause();
		while (f.busy.load(std::memory_order_acquire)) f.busy.wait(true, std::memory_order_acquire);
		f.joins_waited.fetch_add(1, std::memory_order_relaxed);
		f.join_wait_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                                   std::chrono::steady_clock::now() - start)
		                                                   .count()),
		                         std::memory_order_relaxed);
	}
	static void NoteSyncFlush() noexcept {
		if (AsyncReadFlushOn() && !g_read_protects.empty()) AsyncFlusher().sync_flushes.fetch_add(1, std::memory_order_relaxed);
	}
	static void AsyncReadFlushReport() noexcept {
		auto&       f    = AsyncFlusher();
		static auto last = std::chrono::steady_clock::now();
		const auto  now  = std::chrono::steady_clock::now();
		if (now - last < std::chrono::seconds(5)) return;
		const double s = std::chrono::duration<double>(now - last).count();
		last           = now;
		const double b = static_cast<double>(f.batches.exchange(0)), r = static_cast<double>(f.ranges.exchange(0));
		const double j = static_cast<double>(f.joins_waited.exchange(0)), jw = static_cast<double>(f.join_wait_ns.exchange(0));
		const double a = static_cast<double>(f.apply_ns.exchange(0)), sf = static_cast<double>(f.sync_flushes.exchange(0));
		std::printf("[flush-async] %.1f s: lotes al hilo %.0f/s (%.0f rangos/s), aplicados fuera del hilo de render %.2f ms/s; "
		            "flushes que siguen en el hilo de render %.0f/s; esperas del hilo de render al lote %.0f/s, %.2f ms/s\n",
		            s, b / s, r / s, a / 1e6 / s, sf / s, j / s, jw / 1e6 / s);
		std::fflush(stdout);
	}
	// The pending ranges to the worker (after the batch before it). Nothing pending: nothing done.
	static void FlushReadProtectAsync() noexcept {
		if (LoteOn()) {
			LoteHandOff();
			return;
		}
		if (!AsyncReadFlushOn()) {
			FlushReadProtect();
			return;
		}
		AsyncReadFlushReport();
		if (g_read_protects.empty()) return;
		AsyncReadFlushJoin();
		auto& f = AsyncFlusher();
		f.batches.fetch_add(1, std::memory_order_relaxed);
		f.ranges.fetch_add(g_read_protects.size(), std::memory_order_relaxed);
		{
			std::lock_guard lock(f.mutex);
			f.busy.store(true, std::memory_order_release);
			f.batch.swap(g_read_protects);
		}
		f.cv.notify_one();
		g_read_protects.clear();
	}

	// The pending no-access ranges that overlap [begin, end), applied first (a protection change of those pages
	// comes after them; pages of the others keep theirs pending).
	static void FlushReadProtectOverlapping(uint64_t begin, uint64_t end) noexcept {
		if (LoteOn()) LoteWaitOverlapping(begin, end); // (PROT-LOTE: only batches over these pages)
		else AsyncReadFlushJoin(); // (KYTY_FLUSH_ASYNC: the batch handed off before comes first)
		const Raiz::OriginScope origin(Raiz::ProtReadFlush); // (raiz8)
		auto& ranges = g_read_protects;
		for (size_t i = 0; i < ranges.size();) {
			if (ranges[i].begin < end && begin < ranges[i].end) {
				ProtectHost(ranges[i].begin, ranges[i].end - ranges[i].begin, NO_ACCESS_PROTECTION);
				ranges[i] = ranges.back();
				ranges.pop_back();
			} else {
				++i;
			}
		}
	}

	// A no-access range joins the pending ones: merged with those it overlaps or touches (one call for all).
	static void AddReadProtect(uint64_t begin, uint64_t end) noexcept {
		G1::NoteReadTrack(begin, end); // (g1: by pass and label; one relaxed load when off)
		auto& ranges = g_read_protects;
		for (size_t i = 0; i < ranges.size();) {
			if (ranges[i].begin <= end && begin <= ranges[i].end) {
				begin     = std::min(begin, ranges[i].begin);
				end       = std::max(end, ranges[i].end);
				ranges[i] = ranges.back();
				ranges.pop_back();
			} else {
				++i;
			}
		}
		ranges.push_back({begin, end});
	}

	void Protect(uint64_t vaddr, uint64_t size, uint32_t protection) noexcept {
		// (11-14) Native memory: no host protection change at all (bug check 0x1A: paging while protections change).
		if (NativeMemoryNoReadWatch()) return;
		if (g_defer_read_protect) {
			// (PROT-LOTE: a no-access range after one in flight gives the same result in any order; any other protection
			// waits for the batches over its pages only - FlushReadProtectOverlapping -: disjoint pages do not depend on
			// each other's order.)
			if (!LoteOn()) AsyncReadFlushJoin(); // (KYTY_FLUSH_ASYNC: a later protection of this thread comes after the batch)
			if (protection == NO_ACCESS_PROTECTION) {
				AddReadProtect(vaddr, vaddr + size);
				return;
			}
			// (Another protection of the pages of a pending range comes after it; the other ranges stay pending:
			// protections of disjoint pages do not depend on each other's order.)
			FlushReadProtectOverlapping(vaddr, vaddr + size);
		}
		const auto end = vaddr + size;
		// (ram4, KYTY_WRITE_WATCH: not there. Native memory an unmap parks keeps its pages committed, with the protection
		// they had: a release left out would leave them no-access for good, and the emulator's late writes there - GPU
		// results written back, file reads and labels queued before the unmap - are plain stores: an access violation in
		// the exe. kyty_local_unmap_protect_skip_mode is 0 by default; this makes it harmless.)
		if (protection == READ_WRITE_PROTECTION && !NativeReadFaultsOn() && g_unmapping_end > g_unmapping_begin &&
		    vaddr < g_unmapping_end && end > g_unmapping_begin) {
			// Releases inside the range being unmapped are left to the unmap (SetUnmappingRange).
			if (vaddr < g_unmapping_begin) ProtectHost(vaddr, g_unmapping_begin - vaddr, protection);
			if (end > g_unmapping_end) ProtectHost(g_unmapping_end, end - g_unmapping_end, protection);
			return;
		}
		ProtectHost(vaddr, size, protection);
	}

	// ---- (raiz8) KYTY_RAIZ_ASYNC_RELEASE ------------------------------------------------------------------------------
	// (Region lock held.) Pages [first, first + count) of the region: their read release (state changed already) is left
	// to others; their host protection stays no access and their read hints stay set until it is applied.
	void DeferRelease(Region& region, uint64_t base_addr, size_t first, size_t count) noexcept {
		uint64_t marked = 0;
		for (size_t page = first; page < first + count; ++page)
			if (region.deferred[page].exchange(1, std::memory_order_relaxed) == 0) ++marked;
		region.deferred_count.fetch_add(static_cast<uint32_t>(marked), std::memory_order_relaxed);
		Raiz::g_rel_deferred_runs.fetch_add(1, std::memory_order_relaxed);
		Raiz::g_rel_deferred_pages.fetch_add(count, std::memory_order_relaxed);
		Raiz::g_rel_pending_pages.fetch_add(marked, std::memory_order_relaxed);
		// (raiz11, KYTY_RAIZ_RELEASE_LAZY=1, an experiment, off by default) Not queued for the release thread: the pages
		// stay no-access - their kept GPU bytes kept - until something needs them: a fault on one (ApplyDeferredAt), the
		// guest thread that waits for them (ApplyDeferredReleases), an emulator window (its put-in), or the GPU watching them
		// again (ClearDeferredAfterProtect: no put-in at all). 052640: the release thread put in 43,216 pages/s, 2.28
		// ms/frame, before anyone asked; how many of them the game would have touched is what this measures ("[raiz]":
		// ASYNC RELEASE "fault" and "watched again before applied", LAZY "kept now", "store full").
		if (!ReleaseLazyOn()) QueueRelease(base_addr + first * PAGE_SIZE, base_addr + (first + count) * PAGE_SIZE);
	}

	// (sync2, KYTY_RAIZ_APPLY_SHORT_HOLDS=2) At a run's VirtualProtect the walk must hold exactly the granule locks of the
	// run's pages [page, next) - none before, none after - and, cut at granules, one. A breach printed (the first 8), and a
	// count every 2^20 runs ("Raiz: VERIFY sync2").
	static void VerifyRunLocks(const RegionLocker& locker, size_t page, size_t next, bool cut) noexcept {
		static std::atomic<uint64_t> checks {0}, bad {0};
		const size_t lo = LockOf(page), hi = LockOf(next - 1);
		// (The walk may hold the lock of `next` too: reached to see that the run ends there.)
		const bool   hi_ok = locker.HeldHi() == hi || (next < REGION_PAGES && locker.HeldHi() == LockOf(next));
		const bool   ok    = locker.HeldLo() == lo && hi_ok && (!cut || lo == hi);
		if (!ok) {
			const uint64_t n = bad.fetch_add(1, std::memory_order_relaxed) + 1;
			if (n <= 8) {
				std::printf("Raiz: VERIFY sync2 BREACH: run pages %zu-%zu (granules %zu-%zu) protected holding granules %zu-%zu%s\n",
				            page, next, lo, hi, locker.HeldLo(), locker.HeldHi(), cut && lo != hi ? " (run not cut)" : "");
				std::fflush(stdout);
			}
		}
		const uint64_t c = checks.fetch_add(1, std::memory_order_relaxed) + 1;
		if ((c & 0xFFFFFu) == 0) {
			std::printf("Raiz: VERIFY sync2: %llu runs checked, %llu breaches\n", static_cast<unsigned long long>(c),
			            static_cast<unsigned long long>(bad.load(std::memory_order_relaxed)));
			std::fflush(stdout);
		}
	}

	// (Region lock held.) The marked pages of [first, last) given their state's protection now (a run of adjacent ones
	// with the same protection per call; the kernel puts their kept bytes in, as at any release), their marks and - once
	// readable - their read hints cleared. A page watched again meanwhile (no access in its state) only loses its mark: its
	// host protection is no access already. Returns the pages whose mark was cleared.
	// (raiz10) `locker`: the region's locks, taken here as the walk reaches a marked page (a mark is set and cleared under
	// its page's lock: read before as a hint, again under it) and let go behind it; all let go on return, each granule's
	// read bit synced first.
	size_t ApplyRegionDeferred(Region& region, RegionLocker& locker, uint64_t base_addr, size_t first, size_t last,
	                           uint32_t applier) noexcept {
		if (region.deferred_count.load(std::memory_order_relaxed) == 0) return 0;
		const Raiz::OriginScope origin(Raiz::ProtDeferred);
		const uint32_t          holds_mode   = ApplyShortHoldsMode(); // (sync2)
		const bool              short_holds  = holds_mode != 0 && GranuleLocksOn();
		const bool              verify_holds = holds_mode == 2 && short_holds;
		size_t                  applied = 0, cleared = 0, runs = 0;
		size_t                  touched_lo = REGION_PAGES, touched_hi = 0; // pages whose marks were cleared
		const auto              sync = [&](size_t from, size_t to) {
            from = std::max(from, touched_lo);
            to   = std::min(to, touched_hi);
            if (from < to) SyncReadGranules(region, base_addr, from, to);
		};
		for (size_t page = first; page < last;) {
			if (region.deferred[page].load(std::memory_order_relaxed) == 0) {
				++page;
				continue;
			}
			// (sync2) The locks behind the walk let go before this page's is taken: none taken for the granules between.
			if (short_holds) locker.ReleaseBelow(page, sync);
			locker.Reach(page);
			if (region.deferred[page].load(std::memory_order_relaxed) == 0) { // (applied meanwhile)
				++page;
				continue;
			}
			const auto perms = PermsOf(region, page);
			size_t     next  = page + 1;
			// (sync1) The release thread stops a run where the next page's granule lock begins.
			// (sync2) The guest thread's too (KYTY_RAIZ_APPLY_SHORT_HOLDS); a fault's run is one granule's already.
			const bool cut_at_granule = GranuleLocksOn() && ((applier == Raiz::ApplierWorker && ReleaseRunGranuleOn()) ||
			                                                 (applier != Raiz::ApplierWorker && short_holds));
			while (next < last && region.deferred[next].load(std::memory_order_relaxed) != 0) {
				if (cut_at_granule && LockOf(next) != LockOf(page)) break;
				locker.Reach(next);
				if (region.deferred[next].load(std::memory_order_relaxed) == 0 || PermsOf(region, next) != perms) break;
				++next;
			}
			const bool release = perms != NO_ACCESS_PROTECTION;
			if (release) {
				Protect(base_addr + page * PAGE_SIZE, (next - page) * PAGE_SIZE, perms);
				if (verify_holds) VerifyRunLocks(locker, page, next, cut_at_granule);
				applied += next - page;
				++runs;
			}
			for (size_t p = page; p < next; ++p) {
				region.deferred[p].store(0, std::memory_order_relaxed);
				// (The hint goes only now that the page is readable on the host: NativeGuardedAccess trusts a clear one.)
				if (release && NativeReadFaultsOn()) region.read_hint[p].store(false, std::memory_order_release);
			}
			cleared += next - page;
			region.deferred_count.fetch_sub(static_cast<uint32_t>(next - page), std::memory_order_relaxed);
			touched_lo = std::min(touched_lo, page);
			touched_hi = std::max(touched_hi, next);
			page       = next;
			locker.ReleaseBelow(page, sync); // (raiz10: the granules behind the walk)
		}
		locker.ReleaseBelow(REGION_PAGES, sync);
		if (cleared == 0) return 0;
		Raiz::g_rel_pending_pages.fetch_sub(cleared, std::memory_order_relaxed);
		Raiz::g_rel_skipped_pages.fetch_add(cleared - applied, std::memory_order_relaxed);
		const uint32_t who = applier < Raiz::ApplierCount ? applier : Raiz::ApplierWorker;
		Raiz::g_rel_applied_pages[who].fetch_add(applied, std::memory_order_relaxed);
		Raiz::g_rel_applied_runs[who].fetch_add(runs, std::memory_order_relaxed);
		return cleared;
	}

	// (Region lock held.) After a protection the page manager applied itself to [first, first + count) (its state's,
	// `perms`): marked pages there are no longer waiting for their release - their host protection is their state's now -.
	void ClearDeferredAfterProtect(Region& region, uint64_t base_addr, size_t first, size_t count, uint32_t perms) noexcept {
		if (region.deferred_count.load(std::memory_order_relaxed) == 0) return;
		size_t cleared = 0;
		for (size_t p = first; p < first + count && p < REGION_PAGES; ++p) {
			if (region.deferred[p].load(std::memory_order_relaxed) == 0) continue;
			region.deferred[p].store(0, std::memory_order_relaxed);
			++cleared;
			if (perms != NO_ACCESS_PROTECTION && NativeReadFaultsOn() && PermsOf(region, p) != NO_ACCESS_PROTECTION)
				region.read_hint[p].store(false, std::memory_order_release);
		}
		if (cleared == 0) return;
		region.deferred_count.fetch_sub(static_cast<uint32_t>(cleared), std::memory_order_relaxed);
		Raiz::g_rel_pending_pages.fetch_sub(cleared, std::memory_order_relaxed);
		Raiz::g_rel_skipped_pages.fetch_add(cleared, std::memory_order_relaxed);
		SyncReadGranules(region, base_addr, first, std::min<size_t>(first + count, REGION_PAGES));
	}

	// Any thread: the marked pages of [begin, end) applied (ApplyRegionDeferred), each region's lock taken only where
	// some page of the range is marked.
	void ApplyDeferredRange(uint64_t begin, uint64_t end, uint32_t applier) noexcept {
		if (end > ADDRESS_SIZE) end = ADDRESS_SIZE;
		begin = PageStart(begin);
		if (begin >= end) return;
		const auto start   = std::chrono::steady_clock::now();
		size_t     cleared = 0;
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, (chunk_begin / REGION_SIZE + 1) * REGION_SIZE);
			const auto region_base = chunk_begin / REGION_SIZE * REGION_SIZE;
			if (auto* region = FindRegion(chunk_begin); region != nullptr && region->deferred_count.load(std::memory_order_relaxed) != 0) {
				const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
				const auto last  = static_cast<size_t>((chunk_end - region_base + PAGE_SIZE - 1) / PAGE_SIZE);
				bool       any   = false;
				for (size_t p = first; p < last && !any; ++p) any = region->deferred[p].load(std::memory_order_relaxed) != 0;
				if (any) {
					RegionLocker locker(*region); // (raiz10: taken page by page inside)
					cleared += ApplyRegionDeferred(*region, locker, region_base, first, last, applier);
				}
			}
			chunk_begin = chunk_end;
		}
		if (cleared != 0 && applier < Raiz::ApplierCount)
			Raiz::g_rel_apply_us[applier].fetch_add(ElapsedUs(start), std::memory_order_relaxed);
	}

	// A fault at `vaddr`: when its page is marked, the run of marked pages around it is applied (true: retry the access).
	bool ApplyDeferredAt(uint64_t vaddr) noexcept {
		if (vaddr >= ADDRESS_SIZE) return false;
		auto* region = FindRegion(vaddr);
		if (region == nullptr || region->deferred_count.load(std::memory_order_relaxed) == 0) return false;
		const auto page = static_cast<size_t>((vaddr % REGION_SIZE) / PAGE_SIZE);
		if (region->deferred[page].load(std::memory_order_relaxed) == 0) return false;
		const auto start       = std::chrono::steady_clock::now();
		const auto region_base = vaddr / REGION_SIZE * REGION_SIZE;
		struct RoleScope { // (raiz11: a fault applying a deferred release, for the lock holders' count)
			uint32_t saved = Raiz::t_pm_role;
			RoleScope() {
				if (saved == Raiz::RoleOther) Raiz::t_pm_role = Raiz::RoleFault;
			}
			~RoleScope() { Raiz::t_pm_role = saved; }
		} role_scope;
		{
			RegionLocker locker(*region);
			locker.Reach(page);
			// (Applied meanwhile by another thread: the access is retried all the same.)
			if (region->deferred[page].load(std::memory_order_relaxed) == 0) return true;
			// (raiz10: the run of marked pages around it, inside the lock held - its granule -.)
			const size_t held_first = LockFirstPage(LockOf(page));
			const size_t held_end   = GranuleLocksOn() ? held_first + READ_GRANULE_PAGES : REGION_PAGES;
			size_t       lo = page, hi = page + 1;
			while (lo > held_first && region->deferred[lo - 1].load(std::memory_order_relaxed) != 0) --lo;
			while (hi < held_end && region->deferred[hi].load(std::memory_order_relaxed) != 0) ++hi;
			(void)ApplyRegionDeferred(*region, locker, region_base, lo, hi, Raiz::ApplierFault);
		}
		Raiz::g_rel_apply_us[Raiz::ApplierFault].fetch_add(ElapsedUs(start), std::memory_order_relaxed);
		return true;
	}

	template <bool track, bool is_read, bool masked, bool offload = false>
	void UpdateRegionWatchers(Region& region, uint64_t base_addr, size_t first, size_t last,
	                          const RegionBits* mask = nullptr) {
		// (raiz8: the origin of the protections below, for the "[raiz]" line's PROTECT BY ORIGIN.)
		const Raiz::OriginScope origin(offload ? Raiz::ProtOffload
		                               : is_read ? (track ? Raiz::ProtReadTrack : Raiz::ProtReadRelease)
		                                         : (track ? Raiz::ProtWriteTrack : Raiz::ProtWriteRelease));
		// The lowest 64 KiB hold no guest memory (no mapping is ever made there): a range a stale descriptor put there
		// is not watched, both ways alike. Watched, the deferred write protection of its region's first pages started
		// at address 0, an invalid range, and the upload worker's protection job ended the emulator ("invalid range
		// vaddr=0x0, size=0x400000", issue #70 and a 1-1 tour, both with DLSS super resolution on).
		if (base_addr == 0 && first < NULL_PAGES) {
			static std::atomic<bool> reported {false};
			if (!reported.exchange(true, std::memory_order_relaxed)) {
				std::printf("Page watchers: pages below 64 KiB (%s 0x%llx-0x%llx) left unwatched: no guest memory there\n",
				            is_read ? "read" : "write", static_cast<unsigned long long>(first * PAGE_SIZE),
				            static_cast<unsigned long long>(last * PAGE_SIZE));
				std::fflush(stdout);
			}
			first = NULL_PAGES;
			if (first >= last) return;
		}
		// (raiz10) The locks of the granules walked, taken as it reaches them and let go once behind it - no run left to
		// protect there -, each granule's read hints and bit settled first (`finish`, below). One lock per region with
		// KYTY_PM_GRANULE_LOCKS=0: held to the end, as before.
		RegionLocker locker(region);
		locker.Reach(first);
		auto      perms                 = PermsOf(region, first);
		uint64_t  range_begin           = 0;
		uint64_t  range_bytes           = 0;
		uint64_t  potential_range_bytes = 0;

		const auto release_pending = [&] {
			if (range_bytes != 0) {
				if (track && !is_read && !offload && g_deferred_write_protect != nullptr) {
					g_deferred_write_protect->push_back({base_addr + range_begin * PAGE_SIZE, range_bytes});
				} else if (!track && is_read && !offload && perms != NO_ACCESS_PROTECTION && DeferReleasesHere()) {
					// (raiz8, KYTY_RAIZ_ASYNC_RELEASE: a read release of the GPU thread, left to others. A pending no-access
					// range over these pages is applied first, as Protect does: the release comes after it.)
					if (g_defer_read_protect)
						FlushReadProtectOverlapping(base_addr + range_begin * PAGE_SIZE, base_addr + range_begin * PAGE_SIZE + range_bytes);
					DeferRelease(region, base_addr, static_cast<size_t>(range_begin), static_cast<size_t>(range_bytes / PAGE_SIZE));
				} else {
					Protect(base_addr + range_begin * PAGE_SIZE, range_bytes, perms);
					// (raiz8: marked pages this protected have their state's protection now.)
					ClearDeferredAfterProtect(region, base_addr, static_cast<size_t>(range_begin),
					                          static_cast<size_t>(range_bytes / PAGE_SIZE), perms);
				}
				range_bytes           = 0;
				potential_range_bytes = 0;
			}
		};

		// (raiz10) The pages [from, to) of a lock let go: done (their protections applied), so their read hints - a hint goes
		// once its page is readable on the host - and their granules' bits are settled now, under that lock.
		const auto finish = [&](size_t from, size_t to) {
			if constexpr (is_read || offload) {
				from = std::max(from, first);
				to   = std::min(to, last);
				if (from >= to) return;
				// (raiz8: not the pages whose release was left to others - DeferRelease -: still no access on the host.)
				if (NativeReadFaultsOn())
					for (size_t page_index = from; page_index < to; page_index++)
						if (PermsOf(region, page_index) != NO_ACCESS_PROTECTION &&
						    region.deferred[page_index].load(std::memory_order_relaxed) == 0)
							region.read_hint[page_index].store(false, std::memory_order_release);
				SyncReadGranules(region, base_addr, from, to);
			} else {
				(void)from, (void)to;
			}
		};

		for (size_t page_index = first; page_index < last; page_index++) {
			// (raiz10) A new granule: the ones behind the run still to protect (all, with none) let go, this one taken.
			if (page_index != first && LockOf(page_index) != LockOf(page_index - 1)) {
				locker.ReleaseBelow(range_bytes == 0 ? page_index : static_cast<size_t>(range_begin), finish);
				locker.Reach(page_index);
			}
			auto&      page    = region.pages[page_index];
			const auto address = base_addr + page_index * PAGE_SIZE;
			const bool update  = !masked || mask->Get(page_index);

			const auto old_perms = PermsOf(region, page_index);
			uint32_t   new_count = 0;
			if constexpr (offload) {
				auto& holds = region.offload[page_index];
				new_count   = holds.load(std::memory_order_relaxed);
				if (update) {
					if (new_count == (track ? 255u : 0u)) Fatal("offload-hold %s at 0x%016" PRIx64, track ? "overflow" : "underflow", address);
					new_count += track ? 1 : -1;
					holds.store(static_cast<uint8_t>(new_count), std::memory_order_relaxed);
				}
			} else {
				new_count = update ? page.AddDelta<track ? 1 : -1, is_read>(address) : page.AddDelta<0, is_read>(address);
				// (nativemem1: the region's write-watched pages, SweepCpuWrites' filter.)
				if constexpr (!is_read) {
					// (ww2: a region the sweep's cached list does not hold yet makes it build the list again, Region::listed.)
					if (update && track && new_count == 1 && region.write_pages.fetch_add(1) == 0 && g_native_write_watch &&
					    region.listed.load() == 0)
						g_watch_generation.fetch_add(1);
					if (update && !track && new_count == 0) region.write_pages.fetch_sub(1, std::memory_order_relaxed);
				}
			}
			const auto new_perms = PermsOf(region, page_index);
			if constexpr (is_read || offload) {
				// (A page the offload holds is read-watched to HasReadWatchers: a read of it faults.)
				const bool faults = new_perms == NO_ACCESS_PROTECTION;
				if (faults) MarkReadGranule(address); // (before the page's hint: the bits stay a superset)
				// (nativemem1, KYTY_WRITE_WATCH: a hint is cleared only once the page's release is applied, below - the
				// emulator's copies trust a clear hint to mean an accessible page, NativeGuardedAccess -.)
				if (faults || !NativeReadFaultsOn()) region.read_hint[page_index].store(faults, std::memory_order_release);
			}

			if (new_perms != perms) [[unlikely]] {
				release_pending();
				perms = new_perms;
			} else if (range_bytes != 0) {
				potential_range_bytes += PAGE_SIZE;
			}

			if (!update) {
				continue;
			}

			const bool watcher_edge = (track && new_count == 1) || (!track && new_count == 0);
			if (watcher_edge && old_perms != new_perms) {
				if (range_bytes == 0) {
					range_begin           = page_index;
					potential_range_bytes = PAGE_SIZE;
				}
				range_bytes = potential_range_bytes;
			}
		}

		release_pending();
		locker.ReleaseBelow(REGION_PAGES, finish); // (raiz10: the rest, settled)
	}

	// A bit per 64 KiB granule: some page there may have read watchers. Set before a page's read_hint, cleared after
	// the granule's last one (both under the region's lock). HasReadWatchers asks it first: a hint byte per page sat on
	// its own cold line, the region pointer on another, for every page a table read touched (1% of the GPU thread at
	// 1-1); a line of these bits covers 32 MiB.
	std::unique_ptr<std::atomic<uint64_t>[]> read_granules =
	    std::make_unique<std::atomic<uint64_t>[]>(ADDRESS_SIZE / READ_GRANULE_SIZE / 64);

	void MarkReadGranule(uint64_t address) noexcept {
		const auto granule = address / READ_GRANULE_SIZE;
		read_granules[granule / 64].fetch_or(uint64_t {1} << (granule % 64), std::memory_order_release);
	}

	void SyncReadGranules(const Region& region, uint64_t base_addr, size_t first, size_t last) noexcept {
		if (first >= last) return;
		for (size_t g = first / READ_GRANULE_PAGES; g <= (last - 1) / READ_GRANULE_PAGES; ++g) {
			bool any = false;
			for (size_t p = g * READ_GRANULE_PAGES; p < (g + 1) * READ_GRANULE_PAGES; ++p)
				any |= region.read_hint[p].load(std::memory_order_relaxed) != 0;
			const auto granule = base_addr / READ_GRANULE_SIZE + g;
			const auto bit     = uint64_t {1} << (granule % 64);
			if (any)
				read_granules[granule / 64].fetch_or(bit, std::memory_order_release);
			else if ((read_granules[granule / 64].load(std::memory_order_relaxed) & bit) != 0)
				read_granules[granule / 64].fetch_and(~bit, std::memory_order_release);
		}
	}

	[[nodiscard]] bool MayHaveReadWatchers(uint64_t vaddr, uint64_t last_byte) const noexcept {
		if (last_byte >= ADDRESS_SIZE) return true;
		for (auto g = vaddr / READ_GRANULE_SIZE; g <= last_byte / READ_GRANULE_SIZE; ++g)
			if (((read_granules[g / 64].load(std::memory_order_acquire) >> (g % 64)) & 1u) != 0) return true;
		return false;
	}

	// (raiz10) The run of pages from `first` (below `last`) with `first`'s protection, its granules' locks taken as it
	// is read (the next page's too: a run ends where its protection differs, read under its lock). Returns its end.
	static size_t RunLocked(Region& region, RegionLocker& locker, size_t first, size_t last) noexcept {
		locker.Reach(first);
		const auto perms = PermsOf(region, first);
		auto       next  = first + 1;
		while (next < last) {
			locker.Reach(next);
			if (PermsOf(region, next) != perms) break;
			next++;
		}
		return next;
	}

	void SyncProtection(uint64_t vaddr, uint64_t size) {
		const Raiz::OriginScope origin(Raiz::ProtSync); // (raiz8)
		const auto begin = PageStart(vaddr);
		const auto end   = PageEnd(vaddr, size);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, (chunk_begin / REGION_SIZE + 1) * REGION_SIZE);
			const auto region_base = chunk_begin / REGION_SIZE * REGION_SIZE;
			if (auto* region = FindRegion(chunk_begin); region != nullptr) {
				// (raiz10: a run's granules held for its protection only - RunLocked -.)
				RegionLocker locker(*region);
				const auto   last = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
				for (auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE); first < last;) {
					const auto next = RunLocked(*region, locker, first, last);
					Protect(region_base + first * PAGE_SIZE, (next - first) * PAGE_SIZE, PermsOf(*region, first));
					first = next;
					locker.ReleaseBelow(first);
				}
			}
			chunk_begin = chunk_end;
		}
	}

	void WatchedRuns(uint64_t vaddr, uint64_t size, std::vector<PageManager::WatchedRun>* runs) const {
		const auto begin = PageStart(vaddr);
		const auto end   = PageEnd(vaddr, size);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, (chunk_begin / REGION_SIZE + 1) * REGION_SIZE);
			const auto region_base = chunk_begin / REGION_SIZE * REGION_SIZE;
			// (No region lock: a page's state is one byte, and a change made meanwhile protects after the caller.)
			if (const auto* region = FindRegion(chunk_begin); region != nullptr) {
				const auto last = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
				for (auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE); first < last;) {
					const auto perms = PermsOf(*region, first);
					auto       next  = first + 1;
					while (next < last && PermsOf(*region, next) == perms) next++;
					if (perms != READ_WRITE_PROTECTION) {
						const auto address = region_base + first * PAGE_SIZE;
						const auto mode    = ToMemoryMode(perms);
						if (!runs->empty() && runs->back().address + runs->back().size == address && runs->back().mode == mode) {
							runs->back().size += (next - first) * PAGE_SIZE;
						} else {
							runs->push_back({address, (next - first) * PAGE_SIZE, mode});
						}
					}
					first = next;
				}
			}
			chunk_begin = chunk_end;
		}
	}

	void ReapplyProtection(uint64_t vaddr, uint64_t size) {
		const Raiz::OriginScope origin(Raiz::ProtReapply); // (raiz8)
		const auto begin = PageStart(vaddr);
		const auto end   = PageEnd(vaddr, size);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, (chunk_begin / REGION_SIZE + 1) * REGION_SIZE);
			const auto region_base = chunk_begin / REGION_SIZE * REGION_SIZE;
			if (auto* region = FindRegion(chunk_begin); region != nullptr) {
				// (raiz10) A run's granules held for its protection only (RunLocked), not the whole region for all of the
				// range's runs: the upload worker's re-arming had the game's faults anywhere in the 4 MiB waiting for it.
				RegionLocker locker(*region);
				const auto   last = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
				for (auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE); first < last;) {
					const auto next  = RunLocked(*region, locker, first, last);
					const auto perms = PermsOf(*region, first);
					if (perms != READ_WRITE_PROTECTION) {
						Protect(region_base + first * PAGE_SIZE, (next - first) * PAGE_SIZE, perms);
					}
					first = next;
					locker.ReleaseBelow(first);
				}
			}
			chunk_begin = chunk_end;
		}
	}

	// A page some watcher holds (a write, read or offload hold): the pages a write fault was taken on before.
	static bool Watched(const Region& region, size_t page) noexcept {
		return region.pages[page].write_watchers != 0 || region.pages[page].access_watchers != 0 ||
		       region.offload[page].load(std::memory_order_relaxed) != 0;
	}
	// Whether some page of the region at `base` may be watched now, read without its lock (as WatchedRuns: watchers are
	// added by the GPU thread, the sweep's own; one removed meanwhile only makes the sweep read one region more). A
	// page's state is one byte, nonzero while a write or read watcher holds it; the offload's holds set the region's
	// read granule bits (a superset), which are one word for a whole region.
	bool AnyWatched(const Region& region, uint64_t base) const noexcept {
		static_assert(REGION_SIZE / READ_GRANULE_SIZE == 64 && sizeof(PageState) == 1);
		if (read_granules[base / REGION_SIZE].load(std::memory_order_acquire) != 0) return true;
		for (size_t page = 0; page < REGION_PAGES; page += 8) {
			uint64_t word = 0;
			std::memcpy(&word, &region.pages[page], sizeof(word));
			if (word != 0) return true;
		}
		return false;
	}

	static void AppendPage(std::vector<PageManager::DeferredRange>* runs, uint64_t address) {
		if (!runs->empty() && runs->back().address + runs->back().size == address) {
			runs->back().size += PAGE_SIZE;
		} else {
			runs->push_back({address, PAGE_SIZE});
		}
	}

	// (ww2, poll) The regions a write watcher holds pages of, sorted, and their runs of adjacent regions: built again
	// only when a region outside the list got its first write-watched page (g_watch_generation) and every 1024 sweeps
	// (to let go of the regions whose watchers left). The sweeping thread's only. Before, every sweep walked every
	// region ever made (a cold line each) under region_mutex and sorted what it found.
	struct SweepList {
		std::vector<std::pair<uint64_t, Region*>>  regions;
		std::vector<std::pair<uint64_t, uint64_t>> spans; // [begin, end) of each run of adjacent regions
		std::vector<uint32_t>                      span_first; // the index in `regions` of each span's first region
		uint64_t                                   generation = 0;
		uint32_t                                   age        = 0;
	};
	const SweepList& GetSweepList() {
		static SweepList list;
		const auto       generation = g_watch_generation.load();
		if (generation == list.generation && ++list.age < 1024) return list;
		list.generation = generation;
		list.age        = 0;
		list.regions.clear();
		list.spans.clear();
		list.span_first.clear();
		g_stat_list_rebuilds.fetch_add(1, std::memory_order_relaxed);
		{
			std::lock_guard lock(region_mutex);
			for (const auto& entry: region_list) {
				// (Out first, then asked again: see Region::listed.)
				entry.second->listed.store(0);
				if (entry.second->write_pages.load() == 0) continue;
				entry.second->listed.store(1);
				list.regions.push_back(entry);
			}
		}
		std::sort(list.regions.begin(), list.regions.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
		for (size_t i = 0; i < list.regions.size();) {
			size_t j = i + 1;
			while (j < list.regions.size() && list.regions[j].first == list.regions[j - 1].first + REGION_SIZE) j++;
			list.spans.emplace_back(list.regions[i].first, list.regions[j - 1].first + REGION_SIZE);
			list.span_first.push_back(static_cast<uint32_t>(i));
			i = j;
		}
		return list;
	}

	// (In address order, overlapping and adjacent runs merged.)
	static void SortMergeRuns(std::vector<PageManager::DeferredRange>* runs) {
		std::sort(runs->begin(), runs->end(), [](const auto& a, const auto& b) { return a.address < b.address; });
		size_t out = 0;
		for (size_t k = 0; k < runs->size(); k++) {
			auto& last = (*runs)[out == 0 ? 0 : out - 1];
			if (out != 0 && last.address + last.size >= (*runs)[k].address) {
				last.size = std::max(last.address + last.size, (*runs)[k].address + (*runs)[k].size) - last.address;
			} else {
				(*runs)[out++] = (*runs)[k];
			}
		}
		runs->resize(out);
	}

	void SweepCpuWrites(std::vector<PageManager::DeferredRange>* runs, std::vector<PageManager::DeferredRange>* image_runs) {
		// The regions with a watched page now, sorted (a region is made at its first watcher and never freed).
		// (nativemem1, KYTY_WRITE_WATCH: only those a write watcher holds pages of - read-watched and offloaded pages are
		// no-access, their writes fault -, by a counter instead of a scan of the region's 1024 page states; and in a
		// partial sweep only the hot ones, GetSweepConfig.)
		const bool read_faults = NativeReadFaultsOn();
		const bool poll        = GetWatchMode() == WatchMode::Poll;
		static uint64_t sweep_no = 0, full_no = 0;
		static auto     full_at  = std::chrono::steady_clock::now();
		bool            full     = true;
		if (read_faults) {
			const auto& config = GetSweepConfig();
			++sweep_no;
			const auto now = std::chrono::steady_clock::now();
			full = config.full_every <= 1 || sweep_no - full_no >= config.full_every ||
			       now - full_at >= std::chrono::milliseconds(config.full_ms);
			if (full) {
				full_no = sweep_no;
				full_at = now;
			}
			(full ? g_sweep_full : g_sweep_partial).fetch_add(1, std::memory_order_relaxed);
		}
		// (ww2, poll) Before anything is read: a submission queued while the stamp was lower made its writes before this
		// sweep reads the write watch (PageManager::SweptSince).
		if (poll && full) g_sweep_stamp.fetch_add(1);
		thread_local std::vector<std::pair<uint64_t, Region*>> list;
		list.clear();
		const SweepList* cached = poll && full ? &GetSweepList() : nullptr;
		if (cached == nullptr) {
			std::lock_guard lock(region_mutex);
			if (read_faults) {
				for (const auto& entry: region_list) {
					if (entry.second->write_pages.load(std::memory_order_relaxed) == 0) continue;
					const auto hot = entry.second->hot_sweep.load(std::memory_order_relaxed);
					if (full || (hot != 0 && sweep_no - hot <= HOT_SWEEPS)) list.push_back(entry);
				}
			} else {
				for (const auto& entry: region_list)
					if (AnyWatched(*entry.second, entry.first)) list.push_back(entry);
			}
		}
		const size_t region_count = cached != nullptr ? cached->regions.size() : list.size();
		if (read_faults) {
			g_sweep_regions.fetch_add(region_count, std::memory_order_relaxed);
			g_sweep_queried.fetch_add(region_count * REGION_SIZE, std::memory_order_relaxed);
		}
		if (cached == nullptr) std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
		// The pages kept by CaptureCpuWrites, then those of the write watch (read and reset), adjacent regions at once.
		thread_local std::vector<uint64_t> written;
		written.clear();
		// (ww2, poll) CaptureAfterNoAccess's pages (taken whatever their state) and TakeWritesBeforeGpuWrite's (the images').
		thread_local std::vector<uint64_t> forced, images;
		forced.clear();
		images.clear();
		{
			std::lock_guard lock(g_captured_mutex);
			written.insert(written.end(), g_captured_pages.begin(), g_captured_pages.end());
			g_captured_pages.clear();
			forced.swap(g_captured_forced);
			images.swap(g_captured_images);
		}
		thread_local std::vector<std::pair<uint64_t, Region*>> unreadable; // regions whose write watch was refused
		unreadable.clear();
		if (cached != nullptr) {
			// (ww2, poll) Every run of adjacent regions under one hold of the address space's lock.
			thread_local std::vector<uint32_t> failed;
			failed.clear();
			if (!cached->spans.empty() &&
			    !Libs::LibKernel::Memory::NativeWriteWatchRanges(cached->spans.data(), cached->spans.size(), true, &written, &failed)) {
				// (Some part refused: again a region at a time, to know which. What the first call read and reset is in
				// `written` already; the parts it read report nothing new.)
				for (const auto span: failed) {
					for (size_t k = cached->span_first[span];
					     k < cached->regions.size() && cached->regions[k].first < cached->spans[span].second; k++) {
						if (!Libs::LibKernel::Memory::NativeWriteWatch(cached->regions[k].first, REGION_SIZE, &CollectWritten, &written,
						                                             true))
							unreadable.push_back(cached->regions[k]);
					}
				}
			}
		}
		for (size_t i = 0; i < list.size();) {
			size_t j = i + 1;
			while (j < list.size() && list[j].first == list[j - 1].first + REGION_SIZE) j++;
			const auto begin = list[i].first;
			const auto end   = list[j - 1].first + REGION_SIZE;
			if (!Libs::LibKernel::Memory::NativeWriteWatch(begin, end - begin, &CollectWritten, &written, true)) {
				// (Some part refused: again a region at a time, to know which. What the first call read and reset is in
				// `written` already; the parts it read report nothing new.)
				for (size_t k = i; k < j; k++) {
					if (!Libs::LibKernel::Memory::NativeWriteWatch(list[k].first, REGION_SIZE, &CollectWritten, &written, true))
						unreadable.push_back(list[k]);
				}
			}
			i = j;
		}
		std::sort(written.begin(), written.end());
		written.erase(std::unique(written.begin(), written.end()), written.end());
		uint64_t dropped_no_access = 0; // (ram4, KYTY_WRITE_WATCH: see below)
		// The written pages some watcher holds, under each region's lock (nothing is called under it).
		for (size_t at = 0; at < written.size();) {
			const auto region_base = written[at] / REGION_SIZE * REGION_SIZE;
			size_t     stop        = at;
			while (stop < written.size() && written[stop] / REGION_SIZE * REGION_SIZE == region_base) stop++;
			if (auto* region = FindRegion(region_base); region != nullptr) {
				if (read_faults) region->hot_sweep.store(sweep_no, std::memory_order_relaxed);
				RegionLocker lock(*region); // (raiz10: each page's granule, in order)
				for (size_t k = at; k < stop; k++) {
					const auto page = static_cast<size_t>((written[k] - region_base) / PAGE_SIZE);
					lock.ReleaseBelow(page);
					lock.Reach(page);
					// (KYTY_WRITE_WATCH: a write watcher's page that is not no-access; a write to a no-access one faulted
					// and was handled then, a bit there is the emulator's own opened copy, NativeGuardedAccess.)
					const bool take = read_faults ? region->pages[page].write_watchers != 0 &&
					                                    PermsOf(*region, page) != NO_ACCESS_PROTECTION
					                              : Watched(*region, page);
					if (take) AppendPage(runs, region_base + page * PAGE_SIZE);
					// (ram4) Counted, not taken: a written page a write watcher holds that is no-access now. The emulator's
					// own copies forget their writes (NativeForgetWrites), so this is a game write from before GPU work
					// made the page no-access (or one that slipped in while a copy had it open): the other caches over
					// that page keep bytes older than the game's. Taking it here would be wrong the other way (an image
					// the GPU wrote after it would be replaced by guest memory). "Write watch: ... dropped" below.
					// (ww2, poll: the game's writes from before the GPU's were read then - TakeWritesBeforeGpuWrite,
					// CaptureAfterNoAccess -; what is counted here is written while no-access: a copy of the emulator's
					// that did not forget its write, or a game write that slipped in while one had the page open.)
					else if (read_faults && region->pages[page].write_watchers != 0)
						dropped_no_access++;
				}
			}
			at = stop;
		}
		// (ww2, poll) The pages written before their no-access protection took effect: taken whatever they are now, as
		// long as a watcher holds them (the write came after the GPU's: CaptureAfterNoAccess).
		if (!forced.empty()) {
			std::sort(forced.begin(), forced.end());
			forced.erase(std::unique(forced.begin(), forced.end()), forced.end());
			for (const auto address: forced) {
				if (auto* region = FindRegion(address); region != nullptr) {
					PageLock lock(*region, static_cast<size_t>((address % REGION_SIZE) / PAGE_SIZE)); // (raiz10)
					if (Watched(*region, static_cast<size_t>((address % REGION_SIZE) / PAGE_SIZE)))
						runs->push_back({address, PAGE_SIZE});
				}
			}
			SortMergeRuns(runs);
		}
		// (ww2, poll) The pages the buffers took before GPU work wrote them: the images over them, if a watcher holds
		// them still (the image's own).
		if (!images.empty() && image_runs != nullptr) {
			std::sort(images.begin(), images.end());
			images.erase(std::unique(images.begin(), images.end()), images.end());
			for (const auto address: images) {
				if (auto* region = FindRegion(address); region != nullptr) {
					PageLock lock(*region, static_cast<size_t>((address % REGION_SIZE) / PAGE_SIZE)); // (raiz10)
					if (region->pages[(address % REGION_SIZE) / PAGE_SIZE].write_watchers != 0) AppendPage(image_runs, address);
				}
			}
		}
		g_stat_noaccess_pages.fetch_add(dropped_no_access, std::memory_order_relaxed);
		if (read_faults && (!poll || !unreadable.empty())) {
			// (ram4) One line every 5 s while there is something to say: the written pages left out (above) and the
			// regions whose write watch Windows refused. Nothing printed is the good case.
			// (ww2, poll: the pages are in the "Write watch (5.0 s)" line; here only the regions that could not be read.)
			static uint64_t dropped_total = 0, dropped_said = 0, unreadable_sweeps = 0, unreadable_said = 0;
			static auto     said_at = std::chrono::steady_clock::now();
			dropped_total += dropped_no_access;
			if (!unreadable.empty()) unreadable_sweeps++;
			const auto now = std::chrono::steady_clock::now();
			if ((dropped_total != dropped_said || unreadable_sweeps != unreadable_said) && now - said_at >= std::chrono::seconds(5)) {
				std::printf("Write watch: %llu written pages dropped (no-access when swept: a game write right before GPU work "
				            "took the page; %llu in all), %llu sweeps with regions that could not be read (%llu in all)\n",
				            static_cast<unsigned long long>(dropped_total - dropped_said), static_cast<unsigned long long>(dropped_total),
				            static_cast<unsigned long long>(unreadable_sweeps - unreadable_said),
				            static_cast<unsigned long long>(unreadable_sweeps));
				std::fflush(stdout);
				dropped_said    = dropped_total;
				unreadable_said = unreadable_sweeps;
				said_at         = now;
			}
		}
		if (unreadable.empty()) return;
		// Regions whose write watch could not be read: every page a write watcher holds is taken as written (correct,
		// slow). (Pages only read-watched - GPU-written - are not: each would be read back at every sweep.)
		static std::atomic<bool> said {false};
		if (!said.exchange(true, std::memory_order_relaxed)) {
			std::printf("Native write watch: %zu tracker regions (the first at 0x%016" PRIx64 ") could not be read: their "
			            "write-watched pages are taken as written at every sweep\n",
			            unreadable.size(), unreadable.front().first);
			std::fflush(stdout);
		}
		for (const auto& [region_base, region]: unreadable) {
			RegionLocker lock(*region); // (raiz10: granule after granule)
			for (size_t page = 0; page < REGION_PAGES; page++) {
				lock.ReleaseBelow(page);
				lock.Reach(page);
				if (region->pages[page].write_watchers != 0) AppendPage(runs, region_base + page * PAGE_SIZE);
			}
		}
		SortMergeRuns(runs);
	}

	template <bool track, bool is_read, bool offload = false>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
		const auto begin = PageStart(vaddr);
		const auto end   = PageEnd(vaddr, size);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, (chunk_begin / REGION_SIZE + 1) * REGION_SIZE);
			const auto region_base = chunk_begin / REGION_SIZE * REGION_SIZE;
			auto*      region = track ? GetOrCreateRegion(chunk_begin) : FindRegion(chunk_begin);
			if (region == nullptr) {
				Fatal("untracking unknown page 0x%016" PRIx64, chunk_begin);
			}
			const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			UpdateRegionWatchers<track, is_read, false, offload>(*region, region_base, first, last);
			chunk_begin = chunk_end;
		}
	}

	std::unique_ptr<std::atomic<Region*>[]> regions;
	std::vector<std::unique_ptr<Region>>    region_storage;
	// The regions made, with their base, in the order they were made (under region_mutex: SweepCpuWrites).
	std::vector<std::pair<uint64_t, Region*>> region_list;
	std::mutex                              region_mutex;
};

static_assert(std::atomic<void*>::is_always_lock_free);

// (raiz8) The release thread's way in (a friend of PageManager: Impl is private).
struct DeferredReleaseAccess {
	static void Apply(PageManager* owner, uint64_t begin, uint64_t end, uint32_t applier) noexcept {
		owner->m_impl->ApplyDeferredRange(begin, end, applier);
	}
};

namespace {
// (raiz8) KYTY_RAIZ_ASYNC_RELEASE's release thread: the queued ranges, sorted and joined where they touch, applied
// (what is still marked of them: a guest thread or a fault may have done it first).
void ReleaseThreadLoop() {
	Raiz::t_pm_role = Raiz::RoleRelease; // (raiz11: for the lock holders' count)
	auto&                     q = GetReleaseQueue();
	std::vector<PendingRange> batch;
	for (;;) {
		{
			std::unique_lock lock(q.mutex);
			q.ready.wait(lock, [&q] { return !q.ranges.empty(); });
			batch.swap(q.ranges);
		}
		std::sort(batch.begin(), batch.end(), [](const PendingRange& a, const PendingRange& b) { return a.begin < b.begin; });
		size_t out = 0;
		for (size_t i = 1; i < batch.size(); ++i) {
			if (batch[i].begin <= batch[out].end) {
				batch[out].end = std::max(batch[out].end, batch[i].end);
			} else {
				batch[++out] = batch[i];
			}
		}
		if (!batch.empty()) batch.resize(out + 1);
		{
			std::lock_guard apply(q.apply_mutex);
			if (auto* owner = q.owner.load(std::memory_order_acquire); owner != nullptr)
				for (const auto& range: batch) DeferredReleaseAccess::Apply(owner, range.begin, range.end, Raiz::ApplierWorker);
		}
		batch.clear();
	}
}
} // namespace

PageManager::PageManager(): m_impl(std::make_unique<Impl>()) {
	GetReleaseQueue().owner.store(this, std::memory_order_release); // (raiz8)
}

PageManager::~PageManager() {
	// (raiz8: the release thread lets go of it first - a batch under way finishes -.)
	auto& q = GetReleaseQueue();
	PageManager* self = this;
	if (q.owner.compare_exchange_strong(self, nullptr, std::memory_order_acq_rel)) {
		std::lock_guard apply(q.apply_mutex);
	}
}

bool PageManager::ApplyDeferredReleaseAt(uint64_t vaddr) noexcept {
	return m_impl->ApplyDeferredAt(vaddr);
}

void PageManager::ApplyDeferredReleases(uint64_t vaddr, uint64_t size) noexcept {
	if (size == 0 || Raiz::g_rel_pending_pages.load(std::memory_order_relaxed) == 0) return;
	if (auto* owner = GetReleaseQueue().owner.load(std::memory_order_acquire); owner != nullptr) {
		const uint64_t end   = UINT64_MAX - vaddr < size ? UINT64_MAX : vaddr + size;
		const uint32_t saved = Raiz::t_pm_role; // (raiz11)
		if (saved == Raiz::RoleOther) Raiz::t_pm_role = Raiz::RoleGuestApply;
		DeferredReleaseAccess::Apply(owner, vaddr, end, Raiz::ApplierGuest);
		Raiz::t_pm_role = saved;
	}
}

uint64_t PageManager::GetPageSize() const {
	return PAGE_SIZE;
}

const std::atomic<uint64_t>* PageManager::ReadGranules() const noexcept {
	static_assert(READ_GRANULE_SIZE == uint64_t {1} << READ_GRANULE_BITS);
	return m_impl->read_granules.get();
}

bool PageManager::HasReadWatchers(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) return false;
	if (!m_impl->MayHaveReadWatchers(vaddr, vaddr + size - 1)) return false;
	const auto end = PageStart(vaddr + size - 1);
	for (auto page = PageStart(vaddr);; page += PAGE_SIZE) {
		if (const auto* region = m_impl->FindRegion(page);
		    region != nullptr &&
		    region->read_hint[(page % REGION_SIZE) / PAGE_SIZE].load(std::memory_order_acquire))
			return true;
		if (page == end) return false;
	}
}

void PageManager::SetDeferredWriteProtectSink(std::vector<DeferredRange>* sink) noexcept {
	g_deferred_write_protect = sink;
}

void PageManager::SyncProtection(uint64_t vaddr, uint64_t size) {
	m_impl->SyncProtection(vaddr, size);
}

void PageManager::SetUnmappingRange(uint64_t begin, uint64_t end) noexcept {
	g_unmapping_begin = end == 0 ? 0 : begin;
	g_unmapping_end   = end;
}

bool PageManager::RestoreIfUnwatched(uint64_t vaddr) noexcept {
	// A region that does not exist yet never held a watcher: the protection is not ours.
	auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return false;
	}
	Impl::PageLock lock(*region, static_cast<size_t>((vaddr % REGION_SIZE) / PAGE_SIZE)); // (raiz10: its granule's)
	if (Impl::PermsOf(*region, (vaddr % REGION_SIZE) / PAGE_SIZE) != READ_WRITE_PROTECTION) {
		return false;
	}
	const Raiz::OriginScope origin(Raiz::ProtRestore); // (raiz8)
	return Libs::LibKernel::Memory::RestoreGuestWritable(PageStart(vaddr), PAGE_SIZE);
}

bool PageManager::WatchedSpan(uint64_t vaddr, uint64_t size, uint64_t* first, uint64_t* end) const noexcept {
	bool           any  = false;
	const uint64_t stop = UINT64_MAX - vaddr < size ? UINT64_MAX : vaddr + size;
	for (uint64_t at = PageStart(vaddr); at < stop && at < ADDRESS_SIZE;) {
		const uint64_t until = std::min(stop, (at / REGION_SIZE + 1) * REGION_SIZE);
		// (A region that does not exist yet never held a watcher.)
		if (auto* region = m_impl->FindRegion(at); region != nullptr) {
			Impl::RegionLocker lock(*region); // (raiz10: granule after granule)
			for (uint64_t page = at; page < until; page += PAGE_SIZE) {
				lock.ReleaseBelow(static_cast<size_t>((page % REGION_SIZE) / PAGE_SIZE));
				lock.Reach(static_cast<size_t>((page % REGION_SIZE) / PAGE_SIZE));
				if (Impl::PermsOf(*region, (page % REGION_SIZE) / PAGE_SIZE) == READ_WRITE_PROTECTION) continue;
				if (!any) *first = page;
				any  = true;
				*end = page + PAGE_SIZE;
			}
		}
		at = until;
	}
	return any;
}

void PageManager::ReapplyProtection(uint64_t vaddr, uint64_t size) {
	m_impl->ReapplyProtection(vaddr, size);
}

void PageManager::WatchedRuns(uint64_t vaddr, uint64_t size, std::vector<WatchedRun>* runs) const {
	m_impl->WatchedRuns(vaddr, size, runs);
}

template <bool track>
void PageManager::UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
	SlowLog::Scope slow([&](double ms) {
		std::printf("SLOW UpdatePageWatchers %.1f ms track=%d addr=0x%llx size=0x%llx\n", ms, track ? 1 : 0,
		            static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size));
	});
	m_impl->UpdatePageWatchers<track, false>(vaddr, size);
}

template void PageManager::UpdatePageWatchers<true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false>(uint64_t, uint64_t);

template <bool track>
void PageManager::UpdateReadWatchers(uint64_t vaddr, uint64_t size) {
	m_impl->UpdatePageWatchers<track, false, true>(vaddr, size);
}

template void PageManager::UpdateReadWatchers<true>(uint64_t, uint64_t);
template void PageManager::UpdateReadWatchers<false>(uint64_t, uint64_t);

void PageManager::DeferReadProtection(bool on) noexcept {
	if (!on) {
		Impl::FlushReadProtect();
		++g_observation_epoch;
	}
	g_defer_read_protect = on;
}

void PageManager::SetLoteSite(uint32_t site) noexcept {
	Impl::t_lote_site = site;
}

// (PROT-LOTE v2) At a submit: with KYTY_PROT_LOTE=1 the pending ranges go to the worker (no wait). A submit lets the
// guest learn nothing: a label or memory write the guest can see is a packet of its own, flushed (and waited) at its
// translation before the submit; the GPU's results reach guest memory only through readbacks, which fault on (or wait
// for: Protect / FlushReadProtectOverlapping) those pages. Off: as before.
void PageManager::FlushDeferredProtectionAtSubmit() noexcept {
	if (Impl::LoteOn()) {
		Impl::LoteHandOff();
		++g_observation_epoch;
		return;
	}
	FlushDeferredProtection();
}

void PageManager::FlushDeferredProtection() noexcept {
	Impl::FlushReadProtect();
	++g_observation_epoch;
}

// (FLUSH-ASYNC) KYTY_FLUSH_ASYNC=1: the pending ranges applied by a worker (joined at the next flush or protection
// change of this thread); off, the same as FlushDeferredProtection.
void PageManager::FlushDeferredProtectionAsync() noexcept {
	Impl::FlushReadProtectAsync();
	++g_observation_epoch;
}

uint64_t PageManager::ObservationEpoch() noexcept {
	return g_observation_epoch;
}

bool PageManager::NativeWriteWatch() noexcept {
	return g_native_write_watch;
}

bool PageManager::NativeReadFaults() noexcept {
	return NativeReadFaultsOn();
}

void PageManager::TakeSweepStats(uint64_t* full, uint64_t* partial, uint64_t* regions, uint64_t* queried_bytes) noexcept {
	if (full != nullptr) *full = g_sweep_full.exchange(0, std::memory_order_relaxed);
	if (partial != nullptr) *partial = g_sweep_partial.exchange(0, std::memory_order_relaxed);
	if (regions != nullptr) *regions = g_sweep_regions.exchange(0, std::memory_order_relaxed);
	if (queried_bytes != nullptr) *queried_bytes = g_sweep_queried.exchange(0, std::memory_order_relaxed);
}

void PageManager::SweepCpuWrites(std::vector<DeferredRange>* runs, std::vector<DeferredRange>* image_runs) {
	if (!g_native_write_watch) return;
	if (GetWatchMode() == WatchMode::Protect) return; // (ww2: the writes fault)
	m_impl->SweepCpuWrites(runs, image_runs);
}

bool PageManager::WatchPolled() noexcept {
	return GetWatchMode() == WatchMode::Poll;
}

bool PageManager::WatchProtects() noexcept {
	return GetWatchMode() == WatchMode::Protect;
}

uint64_t PageManager::SweepStamp() noexcept {
	return g_sweep_stamp.load();
}

bool PageManager::SweptSince(uint64_t stamp) noexcept {
	return GetWatchMode() == WatchMode::Poll && g_sweep_stamp.load() > stamp;
}

void PageManager::TakeWatchStats(WatchStats* stats) noexcept {
	stats->noaccess_pages      = g_stat_noaccess_pages.exchange(0, std::memory_order_relaxed);
	stats->before_gpu_calls    = g_stat_before_gpu_calls.exchange(0, std::memory_order_relaxed);
	stats->before_gpu_pages    = g_stat_before_gpu_pages.exchange(0, std::memory_order_relaxed);
	stats->after_protect_calls = g_stat_after_protect_calls.exchange(0, std::memory_order_relaxed);
	stats->after_protect_pages = g_stat_after_protect_pages.exchange(0, std::memory_order_relaxed);
	stats->list_rebuilds       = g_stat_list_rebuilds.exchange(0, std::memory_order_relaxed);
	stats->capture_us          = g_stat_capture_us.exchange(0, std::memory_order_relaxed);
}

void PageManager::TakeWritesBeforeGpuWrite(uint64_t vaddr, uint64_t size, std::vector<uint64_t>* pages) noexcept {
	pages->clear();
	if (GetWatchMode() != WatchMode::Poll || size == 0 || UINT64_MAX - vaddr < size) return;
	const auto begin = PageStart(vaddr);
	const auto end   = (vaddr + size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
	const auto start = std::chrono::steady_clock::now();
	(void)Libs::LibKernel::Memory::NativeWriteWatch(begin, end - begin, &Impl::CollectWritten, pages, true);
	g_stat_capture_us.fetch_add(ElapsedUs(start), std::memory_order_relaxed);
	g_stat_before_gpu_calls.fetch_add(1, std::memory_order_relaxed);
	if (pages->empty()) return;
	g_stat_before_gpu_pages.fetch_add(pages->size(), std::memory_order_relaxed);
	std::lock_guard lock(g_captured_mutex);
	g_captured_images.insert(g_captured_images.end(), pages->begin(), pages->end());
}

bool PageManager::HasWriteProtection(uint64_t vaddr, uint64_t size) const noexcept {
	if (GetWatchMode() != WatchMode::Protect || !GuestRange {vaddr, size}.Valid()) return false;
	const auto end = PageStart(vaddr + size - 1);
	for (auto page = PageStart(vaddr);; page += PAGE_SIZE) {
		if (const auto* region = m_impl->FindRegion(page); region != nullptr) {
			// (A page's state is one byte, nonzero while a watcher holds it; read without the region's lock, as
			// AnyWatched: a hint.)
			uint8_t state = 0;
			std::memcpy(&state, &region->pages[(page % REGION_SIZE) / PAGE_SIZE], sizeof(state));
			if (state != 0) return true;
		} else {
			// (No region: nothing watched up to the next one.)
			const auto next = (page / REGION_SIZE + 1) * REGION_SIZE;
			if (next > end || next < page) return false;
			page = next - PAGE_SIZE;
			continue;
		}
		if (page == end) return false;
	}
}

void PageManager::CaptureCpuWrites(uint64_t vaddr, uint64_t size) noexcept {
	if (!g_native_write_watch || size == 0 || UINT64_MAX - vaddr < size) return;
	if (GetWatchMode() == WatchMode::Protect) return; // (ww2: no sweep reads them; the game's writes fault)
	const auto begin = PageStart(vaddr);
	const auto end   = (vaddr + size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
	thread_local std::vector<uint64_t> pages;
	pages.clear();
	// (Read and reset at once: a game write after this is seen by the next sweep as any other.)
	(void)Libs::LibKernel::Memory::NativeWriteWatch(begin, end - begin, &Impl::CollectWritten, &pages, true);
	if (pages.empty()) return;
	std::lock_guard lock(g_captured_mutex);
	g_captured_pages.insert(g_captured_pages.end(), pages.begin(), pages.end());
}

void PageManager::NoteZoneSignal(uint64_t unmapped_bytes, uint64_t copied_bytes) noexcept {
	const auto& c = GetPauseConfig();
	if (!c.on || (unmapped_bytes == 0 && copied_bytes == 0)) return;
	const auto      now = PauseNowMs();
	auto&           p   = GetPause();
	std::lock_guard lock(p.mutex);
	if (now - p.window_ms >= 5000) {
		p.window_ms       = now;
		p.window_unmapped = 0;
		p.window_copied   = 0;
	}
	p.window_unmapped += unmapped_bytes;
	p.window_copied += copied_bytes;
	constexpr uint64_t MiB = 1024u * 1024u;
	const bool zone = p.window_unmapped >= c.zone_mb * MiB;
	const bool copy = p.window_copied >= c.copy_mb * MiB;
	if (!zone && !copy) return;
	char reason[128];
	std::snprintf(reason, sizeof(reason), "zone change: %llu MiB unmapped, %llu MiB copied in 5 s",
	              static_cast<unsigned long long>(p.window_unmapped / MiB),
	              static_cast<unsigned long long>(p.window_copied / MiB));
	PauseEnterLocked(p, now, reason);
}

bool PageManager::TakeProtectPause(std::vector<DeferredRange>* reprotect, std::vector<DeferredRange>* resync,
                                   bool* ending) noexcept {
	const auto& c = GetPauseConfig();
	if (!c.on) return false;
	const auto now = PauseNowMs();
	PausePollRam(now);
	if (!g_pause_active.load(std::memory_order_acquire)) return false;
	auto&           p = GetPause();
	std::lock_guard lock(p.mutex);
	if (!p.active) return false;
	*ending = now >= p.until_ms;
	if (!*ending && (now - p.last_resync_ms < c.resync_ms || p.resync.empty())) return false;
	p.last_resync_ms = now;
	for (const auto& [b, e]: p.resync) resync->push_back({b, e - b});
	p.resync.clear();
	if (*ending) {
		for (const auto& [b, e]: p.host_rw) reprotect->push_back({b, e - b});
		p.host_rw.clear();
		p.active = false;
		g_pause_active.store(false, std::memory_order_release);
	}
	return true;
}

void PageManager::KeepProtectPauseResync(uint64_t vaddr, uint64_t size) noexcept {
	auto&           p = GetPause();
	std::lock_guard lock(p.mutex);
	if (!p.active) return;
	// (Only what the pause left read-write on the host.)
	const auto end = vaddr + size;
	auto       it  = p.host_rw.upper_bound(vaddr);
	if (it != p.host_rw.begin()) --it;
	thread_local std::vector<std::pair<uint64_t, uint64_t>> add;
	add.clear();
	for (; it != p.host_rw.end() && it->first < end; ++it) {
		const auto b = std::max(it->first, vaddr);
		const auto e = std::min(it->second, end);
		if (b < e) add.emplace_back(b, e);
	}
	for (const auto& [b, e]: add) RangeAdd(p.resync, b, e);
}

void PageManager::EndProtectPause(bool ending, uint64_t pages) noexcept {
	auto&           p = GetPause();
	std::lock_guard lock(p.mutex);
	p.pages += pages;
	if (!ending) return;
	std::printf("Protect pause: off (hold over), %llu protections skipped, %llu applied, %llu pages resynced\n",
	            static_cast<unsigned long long>(p.skipped), static_cast<unsigned long long>(p.applied),
	            static_cast<unsigned long long>(p.pages));
	std::fflush(stdout);
	if (!p.active) {
		p.skipped = 0;
		p.applied = 0;
		p.pages   = 0;
	}
}

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask) {
	if (base_addr % REGION_SIZE != 0 || base_addr >= ADDRESS_SIZE ||
	    REGION_SIZE > ADDRESS_SIZE - base_addr) {
		Fatal("invalid tracking region base 0x%016" PRIx64, base_addr);
	}

	const auto start_range = mask.FirstRange();
	const auto end_range   = mask.LastRange();
	if (start_range.first == REGION_PAGES) {
		FailFast("empty region watcher mask");
	}
	const auto first = start_range.first;
	const auto last  = end_range.second;
	if (start_range.second == end_range.second) {
		m_impl->UpdatePageWatchers<track, is_read>(base_addr + first * PAGE_SIZE,
		                                           (last - first) * PAGE_SIZE);
		return;
	}

	auto* region = track ? m_impl->GetOrCreateRegion(base_addr) : m_impl->FindRegion(base_addr);
	if (region == nullptr) {
		Fatal("untracking unknown region 0x%016" PRIx64, base_addr);
	}
	m_impl->UpdateRegionWatchers<track, is_read, true>(*region, base_addr, first, last, &mask);
}

template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&);

} // namespace Libs::Graphics
