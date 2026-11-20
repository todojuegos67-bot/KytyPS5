#ifndef EMULATOR_SRC_LOCAL_RAIZ_H_
#define EMULATOR_SRC_LOCAL_RAIZ_H_

// (raiz1) KYTY_RAIZ=1 (off by default; read once, at first use: the launcher sets it after the exe started).
//
// Bug check 0x1A / 0x41202 (minidumps 11-15): always in this process, inside NtProtectVirtualMemory or
// NtQueryVirtualMemory over a page of a VIEW OF THE GUEST STORE'S SECTION whose PTE was in prototype form while the
// working-set trimmer took pages of that view (16 GB, ~34,000 protection changes a second). A page of PRIVATE memory
// has no prototype PTE: the state the bug check finds cannot exist for it. So the guest's memory is made private:
//
//   KYTY_RAIZ=1  ==  KYTY_NATIVE_MEMORY=1 + KYTY_WRITE_WATCH=1 + KYTY_WRITE_WATCH_MODE=protect  (forced, whatever the
//   launcher or [Avanzado] extra say about those three), plus what raiz1 adds:
//    - the game's writes are still seen by WRITE FAULTS on read-only pages (as today, the proven path: nothing lost by
//      design), on private memory now; the write watch is NOT polled for them (ww2's poll mode: 12-61 ms a frame);
//    - the emulator's own copies over protected pages (no alias view any more) open them for the copy under a gate that
//      every protection change of guest memory respects; reads open no-access pages READ-ONLY (a game write during the
//      copy still faults and waits), writes open them read-write;
//    - a write window over READ-ONLY pages (a write watcher's: CPU and GPU copies equal) can hide a game write that
//      lands in the same pages meanwhile: those pages are handed to the GPU thread as CPU-written (invalidated before
//      the next submission: one upload more, never a lost write). KYTY_RAIZ_WINDOW_RESYNC=0 turns that off (to
//      measure its cost only);
//    - KYTY_RAIZ_VERIFY=1: the write watch (MEM_WRITE_WATCH, which the private memory has) audits the invariant "no
//      byte of a protected page changes without a fault": reset when a page becomes protected, read before its
//      protection changes again and before any window opens it. A page found written while it was protected is a lost
//      write (logged, must be 0).
// The "[raiz]" line every 5 s (GPU thread) has the counters.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

namespace Raiz {

inline void PutEnv(const char* key, const char* value) {
#if defined(_WIN32)
	(void)_putenv_s(key, value);
#else
	(void)setenv(key, value, 1);
#endif
}

[[nodiscard]] inline bool On() noexcept {
	static const bool on = [] {
		const char* v   = std::getenv("KYTY_RAIZ");
		const bool  set = v != nullptr && v[0] == '1';
		if (!set) return false;
		// (Forced: [Avanzado] extra or the launcher may hold KYTY_NATIVE_MEMORY=0 - ds-settings' default -.)
		PutEnv("KYTY_NATIVE_MEMORY", "1");
		PutEnv("KYTY_WRITE_WATCH", "1");
		PutEnv("KYTY_WRITE_WATCH_MODE", "protect");
		// (raiz3) The validated write policies (fewer read-only protections, fewer windows over them), unless set:
		// KYTY_DS_NEIGHBOR_SPLIT=2, KYTY_DS_WRITE_ANNOUNCED=2, KYTY_DS_READBACK_HOOK=1, KYTY_DS_LINEAR_COPY_HOOK=1.
		// (=0 in [Avanzado] extra turns one off.)
		const auto default_env = [](const char* key, const char* value) {
			const char* now = std::getenv(key);
			if (now == nullptr || now[0] == '\0') PutEnv(key, value);
		};
		default_env("KYTY_DS_NEIGHBOR_SPLIT", "2");
		default_env("KYTY_DS_WRITE_ANNOUNCED", "2");
		default_env("KYTY_DS_READBACK_HOOK", "1");
		default_env("KYTY_DS_LINEAR_COPY_HOOK", "1");
		// (ganchos2, KYTY_FRAME_NATIVO=1) The frame memory by the game's own allocator and submit hooks, not by protection:
		// the frame policy (KYTY_DS_WRITE_FRAME, anuncio1/2) with its stale-use check (=2) and KYTY_RAIZ_ANNOUNCE. Without it,
		// a frame reservation over no-watch pages asks them back to the watch (20261120-020146: 101609 per 5 s, "reservas de
		// frame que pidieron paginas de vuelta"), and frame memory is 57% of the write faults (20261115-203605 audit).
		if (const char* fn = std::getenv("KYTY_FRAME_NATIVO"); fn != nullptr && fn[0] == '1') {
			default_env("KYTY_RAIZ_ANNOUNCE", "1");
			default_env("KYTY_DS_WRITE_FRAME", "2");
			std::printf("[frame-nativo] KYTY_FRAME_NATIVO=1: memoria de frame por los ganchos del asignador y de los envios "
			            "(KYTY_RAIZ_ANNOUNCE=%s, KYTY_DS_WRITE_FRAME=%s: con comprobacion de datos viejos)\n",
			            std::getenv("KYTY_RAIZ_ANNOUNCE"), std::getenv("KYTY_DS_WRITE_FRAME"));
		}
		const char* r = std::getenv("KYTY_RAIZ_WINDOW_RESYNC");
		const char* f = std::getenv("KYTY_RAIZ_VERIFY");
		std::printf("Raiz: policies NEIGHBOR_SPLIT=%s WRITE_ANNOUNCED=%s READBACK_HOOK=%s LINEAR_COPY_HOOK=%s\n",
		            std::getenv("KYTY_DS_NEIGHBOR_SPLIT"), std::getenv("KYTY_DS_WRITE_ANNOUNCED"), std::getenv("KYTY_DS_READBACK_HOOK"),
		            std::getenv("KYTY_DS_LINEAR_COPY_HOOK"));
		std::printf("Raiz: on (KYTY_RAIZ=1): guest memory private (no section, no alias view, no prototype PTE); game "
		            "writes by write faults on read-only pages; emulator copies through guarded windows; window resync %s; "
		            "verify %s\n",
		            (r != nullptr && r[0] == '1') ? "ON (KYTY_RAIZ_WINDOW_RESYNC=1)" : "off (raiz2 default)",
		            (f != nullptr && f[0] == '1') ? "ON (KYTY_RAIZ_VERIFY=1: write watch audit)" : "off");
		std::fflush(stdout);
		return true;
	}();
	return on;
}

[[nodiscard]] inline bool VerifyOn() noexcept {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RAIZ_VERIFY");
		return On() && v != nullptr && v[0] == '1';
	}();
	return on;
}

// (raiz2) Off by default now - KYTY_RAIZ_WINDOW_RESYNC=1 turns it on -: raiz1 invalidated ~5,000 pages/s as CPU-written
// (Boletaria, 11-16), images included (TextureCache::InvalidateCpuWrittenPages), where the backing view's writes never
// invalidated anything: an image dropped and made again from guest memory is a one-frame flash, the defect seen with
// raiz1. Off: the emulator's writes over read-only pages are what they were through the alias (no invalidation); a game
// write landing in the same page during the copy's microseconds is not seen (NOTAS.md).
[[nodiscard]] inline bool ResyncOn() noexcept {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RAIZ_WINDOW_RESYNC");
		return On() && v != nullptr && v[0] == '1';
	}();
	return on;
}

// Counters since start (the report takes differences).
struct Counters {
	std::atomic<uint64_t> read_windows {0};       // emulator reads that had to open pages
	std::atomic<uint64_t> write_windows {0};      // emulator writes that had to open pages
	std::atomic<uint64_t> window_protects {0};    // their VirtualProtect calls (open + restore)
	std::atomic<uint64_t> window_ro_pages {0};    // pages opened from read-only (write watchers)
	std::atomic<uint64_t> window_na_pages {0};    // pages opened from no-access (read watchers)
	std::atomic<uint64_t> resync_pages {0};       // pages queued as CPU-written after a write window (KYTY_RAIZ_WINDOW_RESYNC)
	std::atomic<uint64_t> resync_taken_pages {0}; // of those, invalidated by the GPU thread (mapped ones)
	std::atomic<uint64_t> verify_checked {0};     // pages whose write watch the audit read
	std::atomic<uint64_t> verify_lost {0};        // pages written while protected, with no fault (MUST be 0)
	std::atomic<uint64_t> verify_read_window {0}; // pages written during a READ window (MUST be 0: opened read-only)
	std::atomic<uint64_t> verify_us {0};          // the audit's time
	std::atomic<uint64_t> gate_waits {0};         // protection changes that waited for a window (contention)
	// (raiz2)
	std::atomic<uint64_t> probe_skips {0};          // reads whose pages were all readable: no window
	std::atomic<uint64_t> empty_windows {0};        // windows that found nothing to open
	std::atomic<uint64_t> window_us {0};            // windows' own time (gate, VirtualQuery, VirtualProtect; not the copy)
	std::atomic<uint64_t> merged_writebacks {0};    // write-back pieces joined to the one before (one window less each)
	std::atomic<uint64_t> guarded_reads_direct {0}; // NativeGuardedRead (upload worker) with no window
	std::atomic<uint64_t> guarded_reads_window {0}; // NativeGuardedRead that needed one
	// (raiz5)
	std::atomic<uint64_t> gate_waits_gpu {0};   // waits at the gate on the GPU (render) thread
	std::atomic<uint64_t> gate_wait_us_gpu {0}; // and their time
	std::atomic<uint64_t> windows_gpu {0};      // windows that opened something, on the GPU thread
	std::atomic<uint64_t> release_fast {0};     // releases to read-write with kept bytes: one VirtualProtect
	std::atomic<uint64_t> release_slow {0};     // the others with kept bytes (a VirtualQuery + one a run)
	std::atomic<uint64_t> shadow_hits {0};      // reads served by shadows, no window
	std::atomic<uint64_t> shadow_fills {0};     // pages kept as shadows
	std::atomic<uint64_t> shadow_drops {0};     // shadows forgotten
	std::atomic<uint64_t> shadow_full {0};      // fills refused: 16,384 pages kept
	std::atomic<uint64_t> shadow_mismatch {0};  // (raiz6, VERIFY) shadow reads that differed from guest memory
	// (raiz7) The GPU thread's protection changes of guest memory and their whole time (gate, kept bytes, VirtualProtect).
	std::atomic<uint64_t> protects_gpu {0};
	std::atomic<uint64_t> protect_us_gpu {0};
	// (raiz8) Releases with kept bytes to a readable protection that is not read-write (read-only: a write watcher
	// holds the page): one VirtualProtect per run of kept pages, no VirtualQuery (raiz7's "slow" path, 5960/s in
	// 20261116-031219: a VirtualQuery and a VirtualProtect per region).
	std::atomic<uint64_t> release_ranged {0};
};
inline Counters g_counters;

// ---- (raiz8) Who asked for each protection change of guest memory -------------------------------------------------
// The kernel's ProtectMappedUnlocked counts each change (calls, pages, its whole time) under the calling thread's
// origin (t_protect_origin, set by the page manager around its Protect calls), the GPU (render) thread apart from the
// others. The "[raiz]" line prints them: what is left of the 532 changes/frame (raiz7) and where.
enum ProtectOrigin : uint32_t {
	ProtOther = 0,     // not the page manager's: a guest mprotect, the kernel's own
	ProtReadTrack,     // a read watcher added: no access (GPU work writes the pages)
	ProtReadFlush,     // the deferred no-access ranges applied (PageManager::DeferReadProtection)
	ProtReadRelease,   // a read watcher released: no access -> readable (a readback's end)
	ProtWriteTrack,    // a write watcher added: read-only (an upload made the pages clean)
	ProtWriteRelease,  // a write watcher released: read-only -> read-write (the pages became CPU-dirty)
	ProtOffload,       // the texture offload's holds
	ProtSync,          // PageManager::SyncProtection
	ProtReapply,       // PageManager::ReapplyProtection (KYTY_ASYNC_REPROTECT's deferred write protections, remaps)
	ProtRestore,       // PageManager::RestoreIfUnwatched (a write fault on a page no watcher holds)
	ProtDeferred,      // (raiz8) a deferred read release applied (KYTY_RAIZ_ASYNC_RELEASE)
	ProtOriginCount
};
inline constexpr const char* ProtectOriginNames[ProtOriginCount] = {
    "other", "read-track", "read-flush", "read-release", "write-track", "write-release", "offload", "sync", "reapply",
    "restore", "deferred-release"};
inline thread_local uint32_t t_protect_origin = ProtOther;
struct OriginScope {
	uint32_t saved;
	explicit OriginScope(uint32_t origin): saved(t_protect_origin) { t_protect_origin = origin; }
	~OriginScope() { t_protect_origin = saved; }
	OriginScope(const OriginScope&)            = delete;
	OriginScope& operator=(const OriginScope&) = delete;
};
// [0] the GPU thread, [1] the other threads.
inline std::atomic<uint64_t> g_origin_calls[2][ProtOriginCount];
inline std::atomic<uint64_t> g_origin_pages[2][ProtOriginCount];
inline std::atomic<uint64_t> g_origin_us[2][ProtOriginCount];
inline void NoteProtect(bool gpu_thread, uint64_t pages, uint64_t us) {
	const uint32_t origin = t_protect_origin < ProtOriginCount ? t_protect_origin : ProtOther;
	const int      who    = gpu_thread ? 0 : 1;
	g_origin_calls[who][origin].fetch_add(1, std::memory_order_relaxed);
	g_origin_pages[who][origin].fetch_add(pages, std::memory_order_relaxed);
	g_origin_us[who][origin].fetch_add(us, std::memory_order_relaxed);
}

// ---- (raiz8) KYTY_RAIZ_ASYNC_RELEASE: the GPU thread's read releases applied off it ------------------------------
// 20261116-031219: the GPU thread's protection changes 532/frame, 4.17 ms/frame; of it 2.55 ms the kept bytes put in
// at release (87,031 pages/s, 5,960 releases/s, all "slow"), i.e. read releases (no access -> readable) at the end of
// the guest readbacks (BufferCache::FinishGuestReadback). A release only makes a page LESS protected: left protected a
// while longer, an access to it faults and the fault applies it (GpuResourceManager::HandleFault, first thing), the
// emulator's own accesses keep going through windows (the page's read hint stays set until the release is applied).
// So the GPU thread now only records the page state and marks the pages; the release is applied (the current state's
// protection, the kept bytes in) by: the guest thread whose readback it was (right after its SendCommandSync: it was
// waiting for the GPU thread to do exactly that), a fault on the page, or the "release" thread (whatever is left).
// KYTY_RAIZ_ASYNC_RELEASE=0: off (raiz7: the GPU thread applies them).
enum ReleaseApplier : uint32_t { ApplierWorker = 0, ApplierGuest = 1, ApplierFault = 2, ApplierCount = 3 };
inline constexpr const char* ReleaseApplierNames[ApplierCount] = {"release thread", "guest thread", "fault"};
inline std::atomic<uint64_t> g_rel_deferred_runs {0}, g_rel_deferred_pages {0};
inline std::atomic<uint64_t> g_rel_applied_pages[ApplierCount];
inline std::atomic<uint64_t> g_rel_applied_runs[ApplierCount];
inline std::atomic<uint64_t> g_rel_apply_us[ApplierCount];
inline std::atomic<uint64_t> g_rel_skipped_pages {0}; // no access again by then (watched again): nothing to apply
inline std::atomic<uint64_t> g_rel_pending_pages {0}; // gauge: marked, not applied yet
[[nodiscard]] inline bool AsyncReleaseOn() noexcept {
	static const bool on = [] {
		if (!On()) return false;
		const char* v   = std::getenv("KYTY_RAIZ_ASYNC_RELEASE");
		const bool  set = !(v != nullptr && v[0] == '0');
		std::printf("Raiz: async read release %s (KYTY_RAIZ_ASYNC_RELEASE%s): the GPU thread's read releases applied by the "
		            "guest thread that waits for them, a fault on the page or the release thread\n",
		            set ? "ON" : "off", set ? "=0 turns it off" : "=0");
		std::fflush(stdout);
		return set;
	}();
	return on;
}

// ---- (raiz9) What a protection change of guest memory costs, by thread class ---------------------------------------
// 20261116-034739 (raiz8) against the old memory (203605, 215649) at the same load: the upload worker's protection jobs
// (upload_call_us) 2.3-3.2 ms/frame against 1.0-1.3 for about as many re-armed pages (reprotect_pages 14-19 k/frame both),
// and the GPU thread's wait for that worker (upload) 2.9 ms/frame against 1.3-1.5. Of the worker's protection time only
// ~1.4 ms/frame shows inside the kernel's change (PROTECT BY ORIGIN, "reapply", other threads): the rest is outside it
// (region locks, the address space's lock) or not timed. These split it: per class (the GPU thread, the upload threads -
// worker, its protection thread, its lane -, the others), the changes, the gate's waits and their time, the
// VirtualProtect calls' own time, and the page manager's region locks waited for.
enum ThreadClass : uint32_t { ClassGpu = 0, ClassUpload = 1, ClassOther = 2, ClassCount = 3 };
inline constexpr const char* ThreadClassNames[ClassCount] = {"GPU thread", "upload threads", "others"};
inline thread_local bool     t_upload_thread = false; // (set by the upload worker's threads: async-upload.h)
[[nodiscard]] inline uint32_t ClassOf(bool gpu_thread) noexcept {
	return gpu_thread ? ClassGpu : (t_upload_thread ? ClassUpload : ClassOther);
}
struct ProtectCost {
	std::atomic<uint64_t> calls {0}, ns {0};                // protection changes and their whole time (as raiz7's timer, ns)
	std::atomic<uint64_t> gate_waits {0}, gate_wait_us {0}; // the gate (NativeProtectLock / NativeWindowLock) waited for
	std::atomic<uint64_t> vp_calls {0}, vp_ns {0};          // VirtualProtect itself, inside those changes (ns)
	std::atomic<uint64_t> rl_waits {0}, rl_wait_us {0};     // the page manager's region lock found taken, and the wait
};
inline ProtectCost g_cost[ClassCount];
// The gate held alone, by what: [0] a lazy write-back kept (NativeGuardedWrite), [1] an emulator window
// (NativeGuardedAccess). Its time held, every protection change of every thread waiting meanwhile.
inline std::atomic<uint64_t> g_gate_alone_n[2], g_gate_alone_us[2];
inline std::atomic<uint64_t> g_keep_reset_skipped {0}; // (raiz9) ResetWriteWatch calls not made at a keep
// (raiz10) KYTY_RAIZ_KEEP_STRIPES (memoryAddressSpace.inc): a lazy keep holds the gate SHARED and its stripes (1 MiB of
// address each, hashed into 512) alone; a protection change that makes pages readable holds its stripes shared. Counted:
// the keeps so made and their time held, and the stripes' waits - [0] a keep's (alone), [1] a protection's (shared) -.
inline std::atomic<uint64_t> g_keep_stripe_n {0}, g_keep_stripe_us {0};
inline std::atomic<uint64_t> g_stripe_waits[2], g_stripe_wait_us[2];
// (raiz10) The page manager's locks (pageManager.cpp, KYTY_PM_GRANULE_LOCKS): 1 a lock per 64 KiB granule, 0 one per
// 4 MiB region (as before), -1 not read yet.
inline std::atomic<int> g_pm_granule_locks {-1};

// ---- (raiz11) The put-in of kept GPU bytes (KYTY_RAIZ_MERGE_FAST), who holds the page manager's locks, stale reads --
// 20261116-052640 (merge2): putting in at release 4.56 ms/frame on other threads (raiz8's plain copy: 1.53), 8,244
// runs/s, 91,653 pages/s. Counted here: the runs, their pages found already holding the kept bytes (nothing to write,
// left read-only - no read-write opening, no compare-and-swap -) and those that differed, the words written by CAS (and
// how many of them two at a time, 16 bytes), the VirtualProtect calls made, the final checks left to the caller's own
// protection (one VirtualProtect less a run), the time.
inline std::atomic<uint64_t> g_merge_runs {0}, g_merge_pages_same {0}, g_merge_pages_diff {0}, g_merge_words {0};
inline std::atomic<uint64_t> g_merge_cas16 {0}, g_merge_vp {0}, g_merge_after {0}, g_merge_raced {0};
// The page manager's locks (pageManager.cpp, KYTY_PM_LOCK_HOLDERS, default on with KYTY_RAIZ): who waited for whom.
// A role per thread: the GPU (render) thread, the upload threads, the release thread (raiz8), a guest thread applying
// the releases it waits for (BufferCache::ReadMemory), a fault applying a deferred release (ApplyDeferredAt), others
// (the game's other faults: write faults, restores). The holder's role and origin (t_protect_origin) are stored with the
// lock when it is taken; a waiter reads them when it finds the lock taken.
enum PmRole : uint32_t { RoleOther = 0, RoleGpu, RoleUpload, RoleRelease, RoleGuestApply, RoleFault, RoleCount };
inline constexpr const char* PmRoleNames[RoleCount] = {"other", "GPU", "upload", "release thread", "guest apply", "fault apply"};
inline thread_local uint32_t t_pm_role = RoleOther; // (set by the release thread and the appliers' scopes)
inline std::atomic<uint64_t> g_rl_wait_n[RoleCount][RoleCount], g_rl_wait_us[RoleCount][RoleCount]; // [waiter][holder]
inline std::atomic<uint64_t> g_rl_wait_origin_us[ProtOriginCount + 1]; // by the holder's origin (+1: none recorded)
inline std::atomic<uint64_t> g_rl_hold_n[RoleCount], g_rl_hold_us[RoleCount]; // locks held, by the holder's role
inline std::atomic<uint64_t> g_rl_hold_long[RoleCount];                      // of those, held 100 us or more
// Stale reads (KYTY_RAIZ_STALE_CHECK, default on with KYTY_RAIZ): an emulator read straight from guest memory (no
// window) that overlapped a put-in of kept GPU bytes still under way - the page readable, the GPU's bytes not in yet -.
// [0] the upload worker's direct read (Memory::NativeGuardedRead), [1] the GPU thread's reads after SyncGpuCleanBacking
// (indirect arguments, SRT tables), [2] any other emulator read that needed no window (NativeGuardedAccess's direct
// way; checked with KYTY_RAIZ_VERIFY only: it is every read). `fixed`: read again through a window / waited for.
enum StaleSite : uint32_t { StaleUpload = 0, StaleGpuSync = 1, StaleDirect = 2, StaleCount = 3 };
inline constexpr const char* StaleSiteNames[StaleCount] = {"upload direct", "GPU thread sync", "other direct (VERIFY)"};
// `n`: the read's 64 KiB overlapped a put-in (under way, or begun and ended during it); `exact`: a put-in under way at
// the check covered one of its pages (certainly old bytes read).
inline std::atomic<uint64_t> g_stale_n[StaleCount], g_stale_exact[StaleCount], g_stale_fixed[StaleCount];
inline std::atomic<uint64_t> g_putin_inflight_runs {0}; // runs registered (the detector's writes)

// ---- (raiz12) KYTY_RAIZ_SETTLE: the game's readers of GPU results wait for their buffer's put-ins first ----------
// 20261116-060451 (raiz11): 57,328 pages/s put in with their OLD bytes readable (305.6 us each) and 223 game read faults/s
// just before them: the game's threads read GPU results half put in (STALE READS of the emulator's own readers: 0). With
// private memory a page must be readable before its kept bytes can go in (no second view: a section's is bug check
// 0x1A's ingredient; WriteProcessMemory opens the page itself and refuses no-access/read-only pages; NtWriteVirtualMemory
// refuses them), so the window cannot be closed at the page: it is closed at the game's readers. Ghidra (projelf_un3):
//  - 0xab6540 "XPR Index Cull" creates the GPU's output over table 0x2d460's entry [5] (+0x28): 15,000 draw arguments of
//    0x14 bytes, the first dword an index count (CBuffer::Create(.., 0x14, 15000, 0x804, entry[5]));
//  - 0xa0d2d0 fills table 0x2cd30 (0x28 bytes an entry) with [1] = the address of a 0x2d460 entry (0x2cd38 = +0x2d460,
//    0x2cd60 = +0x2d4b0, ...);
//  - 0xab6080 calls 0xa1f020(object, pass, &0x2cd30[i]) for each object; 0xa1f020 reads
//    *(int*)(*(entry[1] + 0x28) + index * 0x14) and clears the object's visible bit when it is 0.
// So the visibility of the world's objects is read by the CPU from the index cull's counters - GPU results -. The hook
// on 0xa1f020's entry (loader/demonsSoulsWriteAudit.cpp) and the readback accessors' (0x909940 / 0x909970) settle
// their buffer before the game reads it (Memory::RaizSettleRange): read back what the GPU still owns, apply the
// deferred releases (their put-in by this thread), wait for any put-in under way. KYTY_RAIZ_SETTLE=1 (default with
// KYTY_RAIZ), =2 only counts (the A/B), =0 off.
// Here: the buffers the 0xa1f020 hook saw (their range, and how many of its calls run over each now), and the put-ins
// over them - the ones that began while a call was reading the buffer are the game's reads of old bytes -.
[[nodiscard]] inline int SettleMode() noexcept {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_RAIZ_SETTLE");
		if (v != nullptr && (v[0] == '0' || v[0] == '1' || v[0] == '2') && v[1] == 0) return v[0] - '0';
		return 1;
	}();
	return On() ? mode : 0;
}
struct SettleBuffer {
	std::atomic<uint64_t> begin {0}, end {0};
	std::atomic<uint32_t> active {0}; // 0xa1f020 calls over it now
};
inline constexpr uint32_t    SettleBufferMax = 32;
inline SettleBuffer          g_settle_buffers[SettleBufferMax];
inline std::atomic<uint32_t> g_settle_buffer_count {0};
// Counters: [0] 0xa1f020 (the index cull's counters), [1] the readback accessors.
// (parpadeo3) [2] 0xa1f560, the XPR index cull's read of the per-section chunk-visibility bits a GPU dispatch wrote
// (0xa1f930 / 0xa20750; KYTY_PARPADEO_XPR, loader/demonsSoulsWriteAudit.cpp). (parpadeo5) Also 0xa1f290's, the reader
// that really runs (0xab6080 -> 0xa1f290 -> 0xad8bc0 / 0xad92f0; KYTY_PARPADEO5=2, local/parpadeo5.h).
enum SettleSiteKind : uint32_t { SettleXpr = 0, SettleAccessor = 1, SettleChunkBits = 2, SettleSiteCount = 3 };
inline constexpr const char* SettleSiteNames[SettleSiteCount] = {"0xa1f020 counters", "accessors 0x909940/70",
                                                                 "chunk bits 0xa1f560 (parpadeo3) / 0xa1f290 (parpadeo5)"};
inline std::atomic<uint64_t> g_settle_calls[SettleSiteCount], g_settle_memo[SettleSiteCount];
inline std::atomic<uint64_t> g_settle_busy[SettleSiteCount];     // a put-in under way over the buffer at entry
inline std::atomic<uint64_t> g_settle_pending[SettleSiteCount];  // kept GPU bytes not in yet (no-access: would fault)
inline std::atomic<uint64_t> g_settle_readback[SettleSiteCount]; // read back here
inline std::atomic<uint64_t> g_settle_waited[SettleSiteCount], g_settle_wait_us[SettleSiteCount];
inline std::atomic<uint64_t> g_settle_left[SettleSiteCount];     // still under way or kept after the wait's cap (4 ms)
inline std::atomic<uint64_t> g_settle_us[SettleSiteCount];       // the hook's own time
inline std::atomic<uint64_t> g_settle_putin_pages {0};           // differing pages put in over a 0xa1f020 buffer
inline std::atomic<uint64_t> g_settle_putin_during {0};          // ... while a 0xa1f020 call was reading it
inline std::atomic<uint64_t> g_settle_full {0};                  // buffers not registered (table full)
// Moves at every keep and every put-in's start: a settle that saw none since its last one over the same buffer, on the
// same thread, has nothing new to wait for (the memo).
inline std::atomic<uint64_t> g_settle_epoch {0};
// The slot of [begin, end) (made when new; -1 when the table is full).
inline int SettleBufferSlot(uint64_t begin, uint64_t end) noexcept {
	const uint32_t n = g_settle_buffer_count.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < n && i < SettleBufferMax; ++i)
		if (g_settle_buffers[i].begin.load(std::memory_order_relaxed) == begin &&
		    g_settle_buffers[i].end.load(std::memory_order_relaxed) == end)
			return static_cast<int>(i);
	static std::mutex make;
	std::lock_guard   lock(make);
	const uint32_t    m = g_settle_buffer_count.load(std::memory_order_relaxed);
	for (uint32_t i = 0; i < m; ++i)
		if (g_settle_buffers[i].begin.load(std::memory_order_relaxed) == begin &&
		    g_settle_buffers[i].end.load(std::memory_order_relaxed) == end)
			return static_cast<int>(i);
	if (m >= SettleBufferMax) {
		g_settle_full.fetch_add(1, std::memory_order_relaxed);
		return -1;
	}
	g_settle_buffers[m].begin.store(begin, std::memory_order_relaxed);
	g_settle_buffers[m].end.store(end, std::memory_order_relaxed);
	g_settle_buffer_count.store(m + 1, std::memory_order_release);
	return static_cast<int>(m);
}
// The buffer slot this thread's 0xa1f020 call is reading (-1: none): a put-in it makes itself - its own read fault's, the
// thread waiting for it - shows it nothing old, and is not counted as one during the read.
inline thread_local int t_settle_slot = -1;
// (Kernel, a put-in's run of differing pages [at, at + size), before its first VirtualProtect.)
inline void SettleNotePutIn(uint64_t at, uint64_t size) noexcept {
	const uint32_t n = g_settle_buffer_count.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < n && i < SettleBufferMax; ++i) {
		const auto&    b = g_settle_buffers[i];
		const uint64_t lo = (std::max)(at, b.begin.load(std::memory_order_relaxed));
		const uint64_t hi = (std::min)(at + size, b.end.load(std::memory_order_relaxed));
		if (lo >= hi) continue;
		const uint64_t pages = ((hi + 0xfffu) >> 12u) - (lo >> 12u);
		g_settle_putin_pages.fetch_add(pages, std::memory_order_relaxed);
		const uint32_t own = t_settle_slot == static_cast<int>(i) ? 1u : 0u;
		if (b.active.load(std::memory_order_seq_cst) > own) g_settle_putin_during.fetch_add(pages, std::memory_order_relaxed);
	}
}

// ---- (raiz9) The upload worker's protection thread (KYTY_UPLOAD_PROT_THREAD, async-upload.h) --------------------
// Counted in any memory mode; printed by the upload worker's own "[upload]" line (AsyncUpload::Drain, every 5 s).
inline std::atomic<uint64_t> g_up_prot_jobs {0};          // protection jobs run (any thread)
inline std::atomic<uint64_t> g_up_prot_us_thread {0};     // their time on the protection thread
inline std::atomic<uint64_t> g_up_prot_us_worker {0};     // their time on the worker (it had nothing protected to copy)
inline std::atomic<uint64_t> g_up_prot_lock_us {0};       // the worker's waits for the protection side's lock
inline std::atomic<uint64_t> g_up_adjacent {0};           // protection jobs starting where the one before ended (same call)
inline std::atomic<uint64_t> g_up_copy_us {0}, g_up_copy_runs {0}, g_up_copy_bytes {0}; // the worker's copies
inline std::atomic<uint64_t> g_up_waits {0}, g_up_wait_us {0}; // the render thread's waits (Drain) and their time

// (raiz2) Who opened a write window (pages actually opened): the calling thread's tag, set by the kernel's write entry
// points (WriteBacking's source, TryWriteCpuBacking, the rest), counted at each window that opened something.
enum Site : uint32_t { SiteOther = 0, SiteReadback, SiteDownload, SiteFeedback, SiteImage, SiteCpuWrite, SiteWriteBacking, SiteCount };
inline constexpr const char* SiteNames[SiteCount] = {"other", "readback", "download", "feedback", "image", "linear/cpu", "backing"};
inline thread_local Site     t_site = SiteOther;
inline std::atomic<uint64_t> g_site_windows[SiteCount];
inline Site SiteOf(const char* source) {
	if (source == nullptr) return SiteWriteBacking;
	auto is = [&](const char* name) {
		for (size_t i = 0;; ++i) {
			if (name[i] == 0) return true;
			if (source[i] != name[i]) return false;
		}
	};
	if (is("readback")) return SiteReadback;
	if (is("download")) return SiteDownload;
	if (is("feedback")) return SiteFeedback;
	if (is("image")) return SiteImage;
	return SiteWriteBacking;
}
struct SiteScope {
	Site saved;
	explicit SiteScope(Site site): saved(t_site) { t_site = site; }
	~SiteScope() { t_site = saved; }
};

// (raiz2) Exposure of the windows over NO-ACCESS pages (GPU-written; the game's reads of them should fault and be read
// back): while one is open the game's threads read those pages without a fault - stale bytes (a read window opens them
// read-only) or half-written ones (a write window, read-write). No read is seen, so it is estimated: each such page's
// window (open time, close time) is kept in a small table; a game READ fault on a page within NEAR_US after its window
// closed is a near miss (the same read a little earlier would have hit the window). Hits/s ~= near misses/s x (mean
// window time / NEAR_US). Plus the total time pages stay open.
inline constexpr uint64_t NEAR_US = 200;
struct ExposureSlot {
	std::atomic<uint64_t> page {0};
	std::atomic<uint64_t> closed_ns {0};
	std::atomic<uint32_t> open_us {0};
	std::atomic<uint32_t> write {0};
};
inline ExposureSlot          g_exposure[4096];
inline std::atomic<uint64_t> g_exposed_pages[2];    // [read, write] no-access pages opened
inline std::atomic<uint64_t> g_exposed_page_us[2];  // their open time summed
inline std::atomic<uint64_t> g_near_misses[2];      // game read faults within NEAR_US after such a window closed
inline uint64_t              NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}
inline size_t ExposureSlotOf(uint64_t page) {
	return static_cast<size_t>(((page >> 12u) * 0x9E3779B97F4A7C15ull) >> 52u) % 4096u;
}
// A no-access run [at, at + size) given its protection back after `open_us` open.
inline void NoteExposure(uint64_t at, uint64_t size, uint64_t open_us, bool write) {
	const uint64_t pages = size >> 12u;
	g_exposed_pages[write].fetch_add(pages, std::memory_order_relaxed);
	g_exposed_page_us[write].fetch_add(pages * open_us, std::memory_order_relaxed);
	const uint64_t now = NowNs();
	for (uint64_t i = 0; i < pages && i < 64; ++i) {
		auto& slot = g_exposure[ExposureSlotOf(at + (i << 12u))];
		slot.page.store(at + (i << 12u), std::memory_order_relaxed);
		slot.open_us.store(static_cast<uint32_t>(open_us), std::memory_order_relaxed);
		slot.write.store(write ? 1u : 0u, std::memory_order_relaxed);
		slot.closed_ns.store(now, std::memory_order_release);
	}
}
// (raiz11) The put-ins' exposure: from a put-in's first VirtualProtect (read-only) to its last CAS, a page whose kept
// bytes differ shows the game its OLD bytes with no fault. Estimated as the windows': a game read fault on the page
// within NEAR_US BEFORE the put-in began is a near miss (the page was still no-access; the same read a little later
// would have read old bytes). Puts-in by the faulting thread itself (ApplyDeferredAt) are not counted.
struct FaultSeen {
	std::atomic<uint64_t> page {0};
	std::atomic<uint64_t> ns {0};
};
inline FaultSeen             g_fault_seen[4096];
inline std::atomic<uint64_t> g_putin_exposed_pages {0}, g_putin_exposed_us {0}, g_putin_near {0};
// (parpadeo3) The pages of the last near misses (the game read faults counted in "est. game reads of OLD bytes"), for
// the "[parpadeo3]" line (loader/demonsSoulsWriteAudit.cpp: where they are, by the memory map's label).
inline constexpr uint32_t    NearPageRing = 256;
inline std::atomic<uint64_t> g_near_page[NearPageRing];
inline std::atomic<uint64_t> g_near_page_next {0};
inline void NotePutInNearPage(uint64_t page) noexcept {
	const uint64_t i = g_near_page_next.fetch_add(1, std::memory_order_relaxed);
	g_near_page[i % NearPageRing].store(page, std::memory_order_relaxed);
}
inline void NotePutInExposure(uint64_t page, uint64_t start_ns, uint64_t open_us) {
	g_putin_exposed_pages.fetch_add(1, std::memory_order_relaxed);
	g_putin_exposed_us.fetch_add(open_us, std::memory_order_relaxed);
	const auto& seen = g_fault_seen[ExposureSlotOf(page)];
	if (seen.page.load(std::memory_order_relaxed) != page) return;
	const uint64_t at = seen.ns.load(std::memory_order_acquire);
	if (at != 0 && at <= start_ns && start_ns - at <= NEAR_US * 1000u) {
		g_putin_near.fetch_add(1, std::memory_order_relaxed);
		NotePutInNearPage(page);
	}
}
// A game read fault (GpuResourceManager::HandleFault).
inline void NoteGameReadFault(uint64_t vaddr) {
	if (!On()) return;
	const uint64_t page = vaddr & ~uint64_t {0xfff};
	{
		auto& seen = g_fault_seen[ExposureSlotOf(page)]; // (raiz11)
		seen.page.store(page, std::memory_order_relaxed);
		seen.ns.store(NowNs(), std::memory_order_release);
	}
	auto&          slot = g_exposure[ExposureSlotOf(page)];
	const uint64_t closed = slot.closed_ns.load(std::memory_order_acquire);
	if (closed == 0 || slot.page.load(std::memory_order_relaxed) != page) return;
	const uint64_t now = NowNs();
	if (now >= closed && now - closed <= NEAR_US * 1000u)
		g_near_misses[slot.write.load(std::memory_order_relaxed) != 0 ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
}

// (raiz3) Lazy write-back (kernel's NativePendingStore & co.): pages of GPU results kept, put in at a release, in a
// window, at a remap; dropped at a map or unmap; write-backs that found the store full.
enum LazyApply : uint32_t { LazyAtRelease = 0, LazyInWindow = 1, LazyAtRemap = 2, LazyHowCount = 3 };
inline std::atomic<uint64_t> g_lazy_stored {0};
inline std::atomic<uint64_t> g_lazy_applied[LazyHowCount];
inline std::atomic<uint64_t> g_lazy_dropped {0};
inline std::atomic<uint64_t> g_lazy_full {0};
// (raiz4) Pages new to the store (the "kept now" gauge: raiz3 took every store, a page kept again included, for a new
// one), and the time of keeping and of putting in at release, the GPU thread's apart from the others'.
inline std::atomic<uint64_t> g_lazy_new {0};
// (raiz6) Kept pages dropped as stale (written since they were kept), takes whose write watch could not be read, runs
// not kept because no read watcher of the trackers held them (a no-access that is not the trackers').
inline std::atomic<uint64_t> g_lazy_stale {0}, g_lazy_unchecked {0}, g_lazy_unwatched {0};
inline std::atomic<uint64_t> g_lazy_store_us_gpu {0}, g_lazy_store_us_other {0};
inline std::atomic<uint64_t> g_lazy_apply_us_gpu {0}, g_lazy_apply_us_other {0};

// The pages a write window opened from read-only, for the GPU thread (GpuResourceManager::SweepCpuWrites).
inline std::mutex                                 g_resync_mutex;
inline std::vector<std::pair<uint64_t, uint64_t>> g_resync; // [begin, end)
inline std::atomic<bool>                          g_resync_pending {false};

inline void QueueResync(uint64_t begin, uint64_t end) {
	if (end <= begin) return;
	g_counters.resync_pages.fetch_add((end - begin) >> 12u, std::memory_order_relaxed);
	std::lock_guard lock(g_resync_mutex);
	if (!g_resync.empty() && g_resync.back().second == begin) {
		g_resync.back().second = end;
	} else {
		g_resync.emplace_back(begin, end);
	}
	g_resync_pending.store(true, std::memory_order_release);
}

inline void TakeResync(std::vector<std::pair<uint64_t, uint64_t>>* out) {
	out->clear();
	if (!g_resync_pending.load(std::memory_order_acquire)) return;
	std::lock_guard lock(g_resync_mutex);
	out->swap(g_resync);
	g_resync_pending.store(false, std::memory_order_relaxed);
}

// A lost write the audit found: counted, the first 32 printed.
inline void NoteLost(uint64_t page, const char* where) {
	g_counters.verify_lost.fetch_add(1, std::memory_order_relaxed);
	static std::atomic<uint32_t> said {0};
	if (said.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::printf("[raiz] VERIFY: page 0x%llx written while protected, with no fault (%s): a LOST write\n",
		            static_cast<unsigned long long>(page), where);
		std::fflush(stdout);
	}
}

// The "[raiz]" line, every 5 s (GPU thread: GpuResourceManager's report). `write_faults`, `window_faults`,
// `read_faults`: LiveCounters' totals since start (the game's writes seen by fault, and reads).
// (raiz2) Plus the windows' own time (and per frame: `flips` since start), the reads a probe let through with no
// window, the windows that opened nothing, the write-back pieces joined, the upload worker's guarded reads; VERIFY's
// time per frame.
inline void Report(uint64_t write_faults, uint64_t window_faults, uint64_t read_faults, uint64_t protects, uint64_t flips,
                   uint64_t busy_waits) {
	if (!On()) return;
	constexpr int   N        = 23;
	static auto     last     = std::chrono::steady_clock::now();
	static uint64_t prev[N]  = {};
	static bool     first    = true;
	const auto      now      = std::chrono::steady_clock::now();
	if (!first && now - last < std::chrono::seconds(5)) return;
	auto& c = g_counters;
	const uint64_t cur[N] = {write_faults,
	                         window_faults,
	                         read_faults,
	                         protects,
	                         c.read_windows.load(),
	                         c.write_windows.load(),
	                         c.window_protects.load(),
	                         c.window_ro_pages.load(),
	                         c.window_na_pages.load(),
	                         c.resync_pages.load(),
	                         c.resync_taken_pages.load(),
	                         c.verify_checked.load(),
	                         c.verify_lost.load(),
	                         c.verify_read_window.load(),
	                         c.verify_us.load(),
	                         c.gate_waits.load(),
	                         c.probe_skips.load(),
	                         c.empty_windows.load(),
	                         c.window_us.load(),
	                         c.merged_writebacks.load(),
	                         c.guarded_reads_direct.load(),
	                         c.guarded_reads_window.load(),
	                         flips};
	if (first) {
		first = false;
		last  = now;
		for (int i = 0; i < N; ++i) prev[i] = cur[i];
		return;
	}
	const double secs = std::chrono::duration<double>(now - last).count();
	const double ps   = secs > 0.0 ? 1.0 / secs : 0.0;
	uint64_t     d[N];
	for (int i = 0; i < N; ++i) d[i] = cur[i] - prev[i];
	const double frames = static_cast<double>(d[22] != 0 ? d[22] : 1);
	std::printf("[raiz] (%.1f s, %llu frames): private guest memory. Game writes seen by FAULT %.0f/s (+%.0f/s window "
	            "faults), by WRITE WATCH (window resync) %.0f pages/s; read faults %.0f/s; guest protection changes %.0f/s. "
	            "Emulator windows: %.0f reads/s + %.0f writes/s (+%.0f/s that opened nothing; %.0f reads/s needed none: "
	            "probe), %.0f VirtualProtect/s, pages opened %.0f/s read-only + %.0f/s no-access; write-backs joined %.0f/s; upload "
	            "worker reads %.0f/s direct + %.0f/s with a window; windows' own time %.2f ms/frame (all threads); gate "
	            "waits %.0f/s; resync invalidated %.0f pages/s",
	            secs, static_cast<unsigned long long>(d[22]), static_cast<double>(d[0]) * ps, static_cast<double>(d[1]) * ps,
	            static_cast<double>(d[9]) * ps, static_cast<double>(d[2]) * ps, static_cast<double>(d[3]) * ps,
	            static_cast<double>(d[4]) * ps, static_cast<double>(d[5]) * ps, static_cast<double>(d[17]) * ps,
	            static_cast<double>(d[16]) * ps, static_cast<double>(d[6]) * ps, static_cast<double>(d[7]) * ps,
	            static_cast<double>(d[8]) * ps, static_cast<double>(d[19]) * ps, static_cast<double>(d[20]) * ps,
	            static_cast<double>(d[21]) * ps, static_cast<double>(d[18]) / 1000.0 / frames,
	            static_cast<double>(d[15]) * ps, static_cast<double>(d[10]) * ps);
	{
		// (raiz2) The no-access windows' exposure (above): pages opened/s, mean open time, near misses/s, estimated hits/s.
		static uint64_t prev_e[6] = {};
		const uint64_t  e[6] = {g_exposed_pages[0].load(), g_exposed_page_us[0].load(), g_near_misses[0].load(),
		                        g_exposed_pages[1].load(), g_exposed_page_us[1].load(), g_near_misses[1].load()};
		for (int k = 0; k < 2; ++k) {
			const uint64_t pages = e[3 * k] - prev_e[3 * k], us = e[3 * k + 1] - prev_e[3 * k + 1], near_n = e[3 * k + 2] - prev_e[3 * k + 2];
			const double   mean  = pages != 0 ? static_cast<double>(us) / static_cast<double>(pages) : 0.0;
			std::printf("; EXPOSURE %s windows over no-access: %.0f pages/s open %.1f us each, game read faults just after "
			            "%.1f/s, est. game reads INSIDE %.2f/s",
			            k == 0 ? "read" : "write", static_cast<double>(pages) * ps, mean, static_cast<double>(near_n) * ps,
			            static_cast<double>(near_n) * ps * mean / static_cast<double>(NEAR_US));
		}
		for (int i = 0; i < 6; ++i) prev_e[i] = e[i];
		// (raiz11) The put-ins' (above).
		static uint64_t prev_p[3] = {};
		const uint64_t  pe[3]     = {g_putin_exposed_pages.load(), g_putin_exposed_us.load(), g_putin_near.load()};
		const uint64_t  pages = pe[0] - prev_p[0], us = pe[1] - prev_p[1], near_n = pe[2] - prev_p[2];
		const double    mean  = pages != 0 ? static_cast<double>(us) / static_cast<double>(pages) : 0.0;
		std::printf("; EXPOSURE put-ins of kept GPU bytes (raiz11): %.0f differing pages/s readable with old bytes %.1f us each, "
		            "game read faults just before %.1f/s, est. game reads of OLD bytes %.2f/s",
		            static_cast<double>(pages) * ps, mean, static_cast<double>(near_n) * ps,
		            static_cast<double>(near_n) * ps * mean / static_cast<double>(NEAR_US));
		for (int i = 0; i < 3; ++i) prev_p[i] = pe[i];
	}
	{
		// (raiz5) The render thread apart, the releases, the shadows.
		static uint64_t prev_r[9] = {};
		const uint64_t  r[9]      = {c.gate_waits_gpu.load(), c.gate_wait_us_gpu.load(), c.windows_gpu.load(), c.release_fast.load(),
		                             c.release_slow.load(),  c.shadow_hits.load(),      c.shadow_fills.load(), c.shadow_drops.load(),
		                             c.shadow_full.load()};
		{
			static uint64_t prev_p[2] = {};
			const uint64_t  p[2]      = {c.protects_gpu.load(), c.protect_us_gpu.load()};
			std::printf("; GPU thread protection changes %.0f/frame, %.2f ms/frame in all (raiz7)",
			            static_cast<double>(p[0] - prev_p[0]) / frames, static_cast<double>(p[1] - prev_p[1]) / 1000.0 / frames);
			prev_p[0] = p[0];
			prev_p[1] = p[1];
		}
		std::printf("; GPU thread: windows %.0f/s, gate waits %.0f/s (%.2f ms/frame); releases with kept bytes %.0f/s fast + "
		            "%.0f/s slow; SHADOWS: reads served %.0f/s, pages kept %.0f/s, forgotten %.0f/s, full %.0f/s",
		            static_cast<double>(r[2] - prev_r[2]) * ps, static_cast<double>(r[0] - prev_r[0]) * ps,
		            static_cast<double>(r[1] - prev_r[1]) / 1000.0 / frames, static_cast<double>(r[3] - prev_r[3]) * ps,
		            static_cast<double>(r[4] - prev_r[4]) * ps, static_cast<double>(r[5] - prev_r[5]) * ps,
		            static_cast<double>(r[6] - prev_r[6]) * ps, static_cast<double>(r[7] - prev_r[7]) * ps,
		            static_cast<double>(r[8] - prev_r[8]) * ps);
		for (int i = 0; i < 9; ++i) prev_r[i] = r[i];
	}
	{
		// (raiz3) Lazy write-back.
		static uint64_t prev_l[11] = {};
		const uint64_t  l[11]      = {g_lazy_stored.load(), g_lazy_applied[LazyAtRelease].load(), g_lazy_applied[LazyInWindow].load(),
		                              g_lazy_applied[LazyAtRemap].load(), g_lazy_dropped.load(), g_lazy_full.load(),
		                              g_lazy_new.load(), g_lazy_store_us_gpu.load(), g_lazy_store_us_other.load(),
		                              g_lazy_apply_us_gpu.load(), g_lazy_apply_us_other.load()};
		const uint64_t kept = l[6] - l[1] - l[2] - l[3] - l[4];
		std::printf("; LAZY write-back: kept %.0f pages/s (%.0f new), put in %.0f/s at release + %.0f/s in a window + %.0f/s at "
		            "a remap, dropped %.0f/s, store full %.0f/s, kept now %llu pages (%.0f MiB); time per frame: keeping %.2f "
		            "ms GPU thread + %.2f others, putting in at release %.2f ms GPU thread + %.2f others",
		            static_cast<double>(l[0] - prev_l[0]) * ps, static_cast<double>(l[6] - prev_l[6]) * ps,
		            static_cast<double>(l[1] - prev_l[1]) * ps, static_cast<double>(l[2] - prev_l[2]) * ps,
		            static_cast<double>(l[3] - prev_l[3]) * ps, static_cast<double>(l[4] - prev_l[4]) * ps,
		            static_cast<double>(l[5] - prev_l[5]) * ps, static_cast<unsigned long long>(kept),
		            static_cast<double>(kept) * 4608.0 / 1048576.0, static_cast<double>(l[7] - prev_l[7]) / 1000.0 / frames,
		            static_cast<double>(l[8] - prev_l[8]) / 1000.0 / frames, static_cast<double>(l[9] - prev_l[9]) / 1000.0 / frames,
		            static_cast<double>(l[10] - prev_l[10]) / 1000.0 / frames);
		for (int i = 0; i < 11; ++i) prev_l[i] = l[i];
		static uint64_t prev_x[3] = {};
		const uint64_t  x[3]      = {g_lazy_stale.load(), g_lazy_unchecked.load(), g_lazy_unwatched.load()};
		std::printf(" (raiz6: dropped as stale - written after kept - %.0f/s, total %llu; takes unchecked %.0f/s; not kept, no "
		            "read watcher %.0f/s)",
		            static_cast<double>(x[0] - prev_x[0]) * ps, static_cast<unsigned long long>(x[0]),
		            static_cast<double>(x[1] - prev_x[1]) * ps, static_cast<double>(x[2] - prev_x[2]) * ps);
		for (int i = 0; i < 3; ++i) prev_x[i] = x[i];
	}
	{
		static uint64_t prev_busy = 0;
		std::printf("; protections that waited for a map/unmap %.0f/s", static_cast<double>(busy_waits - prev_busy) * ps);
		prev_busy = busy_waits;
	}
	{
		// (raiz8) The protection changes by origin: GPU thread per frame (calls, pages, ms), the other threads per second
		// (calls) and their time per frame. Only the origins with something.
		static uint64_t prev_o[3][2][ProtOriginCount] = {};
		std::printf("; PROTECT BY ORIGIN (raiz8) GPU thread /frame:");
		for (int who = 0; who < 2; ++who) {
			if (who == 1) std::printf(" | other threads:");
			for (uint32_t o = 0; o < ProtOriginCount; ++o) {
				const uint64_t calls = g_origin_calls[who][o].load(), pages = g_origin_pages[who][o].load(),
				               us    = g_origin_us[who][o].load();
				const uint64_t dc = calls - prev_o[0][who][o], dp = pages - prev_o[1][who][o], du = us - prev_o[2][who][o];
				prev_o[0][who][o] = calls;
				prev_o[1][who][o] = pages;
				prev_o[2][who][o] = us;
				if (dc == 0) continue;
				if (who == 0)
					std::printf(" %s %.1f calls %.0f pages %.2f ms", ProtectOriginNames[o], static_cast<double>(dc) / frames,
					            static_cast<double>(dp) / frames, static_cast<double>(du) / 1000.0 / frames);
				else
					std::printf(" %s %.0f calls/s %.0f pages/s %.2f ms/frame", ProtectOriginNames[o], static_cast<double>(dc) * ps,
					            static_cast<double>(dp) * ps, static_cast<double>(du) / 1000.0 / frames);
			}
		}
		static uint64_t prev_ranged = 0;
		const uint64_t  ranged      = c.release_ranged.load();
		std::printf("; releases with kept bytes to read-only, by runs (raiz8) %.0f/s", static_cast<double>(ranged - prev_ranged) * ps);
		prev_ranged = ranged;
	}
	{
		// (raiz8) KYTY_RAIZ_ASYNC_RELEASE.
		static uint64_t prev_d[2] = {}, prev_a[3][ApplierCount] = {}, prev_s = 0;
		const uint64_t  d[2]      = {g_rel_deferred_runs.load(), g_rel_deferred_pages.load()};
		std::printf("; ASYNC RELEASE (raiz8): GPU thread left %.1f releases/frame (%.0f pages/frame) to others; applied by",
		            static_cast<double>(d[0] - prev_d[0]) / frames, static_cast<double>(d[1] - prev_d[1]) / frames);
		for (uint32_t a = 0; a < ApplierCount; ++a) {
			const uint64_t runs = g_rel_applied_runs[a].load(), pages = g_rel_applied_pages[a].load(), us = g_rel_apply_us[a].load();
			std::printf(" %s %.0f runs/s %.0f pages/s %.2f ms/frame%s", ReleaseApplierNames[a],
			            static_cast<double>(runs - prev_a[0][a]) * ps, static_cast<double>(pages - prev_a[1][a]) * ps,
			            static_cast<double>(us - prev_a[2][a]) / 1000.0 / frames, a + 1 < ApplierCount ? "," : "");
			prev_a[0][a] = runs;
			prev_a[1][a] = pages;
			prev_a[2][a] = us;
		}
		const uint64_t s = g_rel_skipped_pages.load();
		std::printf("; watched again before applied %.0f pages/s; pending now %llu pages", static_cast<double>(s - prev_s) * ps,
		            static_cast<unsigned long long>(g_rel_pending_pages.load()));
		prev_s = s;
		prev_d[0] = d[0];
		prev_d[1] = d[1];
	}
	{
		static uint64_t prev_site[SiteCount] = {};
		std::printf("; write windows by source/s:");
		for (uint32_t i = 0; i < SiteCount; ++i) {
			const uint64_t v = g_site_windows[i].load();
			std::printf(" %s %.0f", SiteNames[i], static_cast<double>(v - prev_site[i]) * ps);
			prev_site[i] = v;
		}
	}
	{
		// (raiz9) Protection cost by thread class, per frame: changes, their whole time, the gate's waits, VirtualProtect
		// itself, the region locks waited for; and the gate held alone (lazy keeps, windows).
		static uint64_t prev_k[ClassCount][8] = {};
		std::printf("; PROTECT COST (raiz9) per frame:");
		for (uint32_t k = 0; k < ClassCount; ++k) {
			auto&          g    = g_cost[k];
			const uint64_t v[8] = {g.calls.load(),    g.ns.load(),    g.gate_waits.load(), g.gate_wait_us.load(),
			                       g.vp_calls.load(), g.vp_ns.load(), g.rl_waits.load(),   g.rl_wait_us.load()};
			double         dv[8];
			for (int i = 0; i < 8; ++i) {
				dv[i]          = static_cast<double>(v[i] - prev_k[k][i]);
				prev_k[k][i] = v[i];
			}
			std::printf(" %s %.0f changes %.2f ms (gate %.1f waits %.2f ms, VirtualProtect %.0f %.2f ms, region lock %.1f waits "
			            "%.2f ms)%s",
			            ThreadClassNames[k], dv[0] / frames, dv[1] / 1e6 / frames, dv[2] / frames, dv[3] / 1000.0 / frames,
			            dv[4] / frames, dv[5] / 1e6 / frames, dv[6] / frames, dv[7] / 1000.0 / frames,
			            k + 1 < ClassCount ? "," : "");
		}
		static uint64_t prev_g[5] = {};
		const uint64_t  g[5]      = {g_gate_alone_n[0].load(), g_gate_alone_us[0].load(), g_gate_alone_n[1].load(),
		                             g_gate_alone_us[1].load(), g_keep_reset_skipped.load()};
		std::printf("; gate held alone: lazy keeps %.0f/s %.2f ms/frame, windows %.0f/s %.2f ms/frame; keep resets skipped %.0f/s",
		            static_cast<double>(g[0] - prev_g[0]) * ps, static_cast<double>(g[1] - prev_g[1]) / 1000.0 / frames,
		            static_cast<double>(g[2] - prev_g[2]) * ps, static_cast<double>(g[3] - prev_g[3]) / 1000.0 / frames,
		            static_cast<double>(g[4] - prev_g[4]) * ps);
		for (int i = 0; i < 5; ++i) prev_g[i] = g[i];
		// (raiz10) The keeps off the gate's exclusive hold, and the page manager's lock granularity.
		static uint64_t prev_s[6] = {};
		const uint64_t  s[6]      = {g_keep_stripe_n.load(),    g_keep_stripe_us.load(), g_stripe_waits[0].load(),
		                             g_stripe_wait_us[0].load(), g_stripe_waits[1].load(), g_stripe_wait_us[1].load()};
		const int       pm         = g_pm_granule_locks.load();
		std::printf("; KEEP STRIPES (raiz10): keeps %.0f/s held %.2f ms/frame, stripe waits: keeps %.0f/s %.2f ms/frame, "
		            "protections %.0f/s %.2f ms/frame; page manager locks %s",
		            static_cast<double>(s[0] - prev_s[0]) * ps, static_cast<double>(s[1] - prev_s[1]) / 1000.0 / frames,
		            static_cast<double>(s[2] - prev_s[2]) * ps, static_cast<double>(s[3] - prev_s[3]) / 1000.0 / frames,
		            static_cast<double>(s[4] - prev_s[4]) * ps, static_cast<double>(s[5] - prev_s[5]) / 1000.0 / frames,
		            pm == 1 ? "per 64 KiB granule" : pm == 0 ? "per 4 MiB region" : "(not used yet)");
		for (int i = 0; i < 6; ++i) prev_s[i] = s[i];
	}
	{
		// (raiz11) The put-in's work, the page manager's locks by waiter and holder, the stale reads.
		static uint64_t prev_m[8] = {};
		const uint64_t  m[8]      = {g_merge_runs.load(), g_merge_pages_same.load(), g_merge_pages_diff.load(), g_merge_words.load(),
		                             g_merge_cas16.load(), g_merge_vp.load(),        g_merge_after.load(),      g_merge_raced.load()};
		double          dm[8];
		for (int i = 0; i < 8; ++i) dm[i] = static_cast<double>(m[i] - prev_m[i]), prev_m[i] = m[i];
		std::printf("; MERGE (raiz11): runs %.0f/s, pages same %.0f/s differing %.0f/s, CAS words %.0f/s (16-byte %.0f/s), "
		            "VirtualProtect %.0f/s, checks after the caller's protection %.0f/s, game writes found %llu (total)",
		            dm[0] * ps, dm[1] * ps, dm[2] * ps, dm[3] * ps, dm[4] * ps, dm[5] * ps, dm[6] * ps,
		            static_cast<unsigned long long>(m[7]));
		static uint64_t prev_w[RoleCount][RoleCount][2] = {};
		static uint64_t prev_h[RoleCount][3]            = {};
		static uint64_t prev_o[ProtOriginCount + 1]     = {};
		std::printf("; PM LOCKS (raiz11) waits ms/frame [waiter<-holder]:");
		for (uint32_t w = 0; w < RoleCount; ++w)
			for (uint32_t h = 0; h < RoleCount; ++h) {
				const uint64_t n = g_rl_wait_n[w][h].load(), us = g_rl_wait_us[w][h].load();
				const double   dn = static_cast<double>(n - prev_w[w][h][0]), dus = static_cast<double>(us - prev_w[w][h][1]);
				prev_w[w][h][0] = n, prev_w[w][h][1] = us;
				if (dus / 1000.0 / frames >= 0.01)
					std::printf(" %s<-%s %.1f/frame %.2f", PmRoleNames[w], PmRoleNames[h], dn / frames, dus / 1000.0 / frames);
			}
		std::printf(", by holder's origin:");
		for (uint32_t o = 0; o <= ProtOriginCount; ++o) {
			const uint64_t us = g_rl_wait_origin_us[o].load();
			const double   d  = static_cast<double>(us - prev_o[o]) / 1000.0 / frames;
			prev_o[o]         = us;
			if (d >= 0.01) std::printf(" %s %.2f", o < ProtOriginCount ? ProtectOriginNames[o] : "(none)", d);
		}
		std::printf(", held by:");
		for (uint32_t h = 0; h < RoleCount; ++h) {
			const uint64_t v[3] = {g_rl_hold_n[h].load(), g_rl_hold_us[h].load(), g_rl_hold_long[h].load()};
			const double   dn = static_cast<double>(v[0] - prev_h[h][0]), dus = static_cast<double>(v[1] - prev_h[h][1]),
			             dl = static_cast<double>(v[2] - prev_h[h][2]);
			for (int i = 0; i < 3; ++i) prev_h[h][i] = v[i];
			if (dn != 0)
				std::printf(" %s %.0f/frame %.2f ms (>=100us %.1f/frame)", PmRoleNames[h], dn / frames, dus / 1000.0 / frames,
				            dl / frames);
		}
		static uint64_t prev_st[StaleCount][3] = {};
		std::printf("; STALE READS (raiz11, during a put-in; same 64 KiB / same pages):");
		for (uint32_t s = 0; s < StaleCount; ++s) {
			const uint64_t n = g_stale_n[s].load(), x = g_stale_exact[s].load(), f = g_stale_fixed[s].load();
			std::printf(" %s %.0f/s / %.0f/s (read again %.0f/s; totals %llu / %llu)", StaleSiteNames[s],
			            static_cast<double>(n - prev_st[s][0]) * ps, static_cast<double>(x - prev_st[s][1]) * ps,
			            static_cast<double>(f - prev_st[s][2]) * ps, static_cast<unsigned long long>(n), static_cast<unsigned long long>(x));
			prev_st[s][0] = n, prev_st[s][1] = x, prev_st[s][2] = f;
		}
		// (raiz12) KYTY_RAIZ_SETTLE.
		if (SettleMode() != 0) {
			static uint64_t prev_se[SettleSiteCount][9] = {};
			static uint64_t prev_sp[2]                  = {};
			std::printf("; SETTLE (raiz12, KYTY_RAIZ_SETTLE=%d %s):", SettleMode(), SettleMode() == 1 ? "fix" : "count only");
			for (uint32_t s = 0; s < SettleSiteCount; ++s) {
				const uint64_t v[9] = {g_settle_calls[s].load(),    g_settle_memo[s].load(),    g_settle_busy[s].load(),
				                       g_settle_pending[s].load(),  g_settle_readback[s].load(), g_settle_waited[s].load(),
				                       g_settle_wait_us[s].load(),  g_settle_left[s].load(),     g_settle_us[s].load()};
				double         dv[9];
				for (int i = 0; i < 9; ++i) dv[i] = static_cast<double>(v[i] - prev_se[s][i]), prev_se[s][i] = v[i];
				std::printf(" %s: calls %.0f/s (memo %.0f/s), at entry a put-in UNDER WAY %.1f/s (total %llu) / kept bytes not in "
				            "%.1f/s, read back %.1f/s, waited %.1f/s %.3f ms/frame, still under way or kept after 4 ms %llu (total), hook "
				            "%.3f ms/frame;",
				            SettleSiteNames[s], dv[0] * ps, dv[1] * ps, dv[2] * ps, static_cast<unsigned long long>(v[2]), dv[3] * ps,
				            dv[4] * ps, dv[5] * ps, dv[6] / 1000.0 / frames, static_cast<unsigned long long>(v[7]),
				            dv[8] / 1000.0 / frames);
			}
			const uint64_t sp[2] = {g_settle_putin_pages.load(), g_settle_putin_during.load()};
			std::printf(" put-ins over the 0xa1f020 buffers (%u known): %.1f differing pages/s, of them while the game was "
			            "READING the buffer %.1f/s (total %llu: with =1 must stay 0)",
			            g_settle_buffer_count.load(), static_cast<double>(sp[0] - prev_sp[0]) * ps,
			            static_cast<double>(sp[1] - prev_sp[1]) * ps, static_cast<unsigned long long>(sp[1]));
			prev_sp[0] = sp[0], prev_sp[1] = sp[1];
		}
	}
	if (VerifyOn()) std::printf("; VERIFY shadows differing from memory: %llu (total, must be 0)",
	                            static_cast<unsigned long long>(c.shadow_mismatch.load()));
	if (VerifyOn()) {
		std::printf("; VERIFY: %.0f pages audited/s (%.2f ms/frame, all threads), LOST %llu (total %llu, must be 0), written "
		            "in a read window %llu (total %llu, must be 0)",
		            static_cast<double>(d[11]) * ps, static_cast<double>(d[14]) / 1000.0 / frames,
		            static_cast<unsigned long long>(d[12]), static_cast<unsigned long long>(cur[12]),
		            static_cast<unsigned long long>(d[13]), static_cast<unsigned long long>(cur[13]));
	}
	std::printf("\n");
	std::fflush(stdout);
	last = now;
	for (int i = 0; i < N; ++i) prev[i] = cur[i];
}

} // namespace Raiz

#endif // EMULATOR_SRC_LOCAL_RAIZ_H_
