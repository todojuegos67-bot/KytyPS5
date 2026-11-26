// Internal memory address-space helpers included by memory.cpp.
// Kept in the same translation unit because they depend on memory.cpp-local helpers.

#include "cold-store.h" // (ram11, KYTY_COLD_STORE)
#include "nativo-cubierto.h" // (VIGILANCIA-DOBLE: KYTY_SIN_VIGILANCIA_DOBLE)
#include "lock-watch.h" // (ram13b: who waits for which lock, KYTY_LOCK_WATCH)

static constexpr uint64_t HOST_SYSTEM_MANAGED_MIN  = 0x0000040000ull;
static constexpr uint64_t HOST_SYSTEM_MANAGED_MAX  = 0x07ffffbfffull;
static constexpr uint64_t HOST_SYSTEM_RESERVED_MIN = HOST_SYSTEM_MANAGED_MAX + 1u;
#if defined(__APPLE__) && (defined(__x86_64__) || defined(_M_X64))
static constexpr uint64_t HOST_SYSTEM_RESERVED_MAX = 0x0fbfffffffull;
static constexpr uint64_t HOST_USER_MIN            = 0x7000000000ull;
#else
static constexpr uint64_t HOST_SYSTEM_RESERVED_MAX = 0x0fffffffffull;
static constexpr uint64_t HOST_USER_MIN            = 0x1000000000ull;
#endif
static constexpr uint64_t HOST_USER_MAX = 0xfbffffffffull;

static bool GuestFreeRangeContains(uint64_t base, uint64_t range_size, uint64_t candidate,
                                   uint64_t size) {
	if (UINT64_MAX - base < range_size || candidate < base) {
		return false;
	}
	const auto range_end = base + range_size;
	return candidate <= range_end && size <= range_end - candidate;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// (A placeholder split or coalesce, counted with its time: placeholder_calls, placeholder_kt.)
static bool PlaceholderFree(void* vaddr, uint64_t size, DWORD type) {
	const auto start = __rdtsc();
	const bool done  = VirtualFree(vaddr, size, type) != 0;
	LiveCounters::Add(LiveCounters::PlaceholderCalls);
	LiveCounters::Add(LiveCounters::PlaceholderKt, (__rdtsc() - start) >> 10u);
	return done;
}

static bool CoalescePlaceholders(uint64_t vaddr, uint64_t size) {
	return size != 0 && PlaceholderFree(reinterpret_cast<void*>(vaddr), size, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS);
}

// Kyty.Trim's trim of a parked range: GuestAddressSpace::TrimParked while the address space lives (below).
static void TrimParkedRange(uint64_t vaddr, uint64_t size);

// Ranges parked in their views (GuestBackingStore::UnmapOne) taken out of the working set by a thread of its own: mapped,
// their pages stayed counted in use (~1.5 GB of the game's freed streaming memory at a time), and dropping them costs
// ~0.5 us per 4 KiB page (VirtualUnlock on pages not locked), which the game's threads no longer pay. Harmless when late:
// on a range unmapped meanwhile it fails, a range mapped again has its pages faulted back in.
class ParkedRangeTrimmer {
public:
	static void Trim(uint64_t vaddr, uint64_t size) {
		static auto* trimmer = new ParkedRangeTrimmer; // (never destroyed: its thread may outlive static destructors)
		{
			std::lock_guard lock(trimmer->m_mutex);
			trimmer->m_ranges.emplace_back(vaddr, size);
		}
		trimmer->m_ready.notify_one();
	}

private:
	ParkedRangeTrimmer() {
		std::thread([this] {
			LocalPlatform::SetThreadName("Kyty.Trim");
			SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
			std::vector<std::pair<uint64_t, uint64_t>> ranges;
			for (;;) {
				{
					std::unique_lock lock(m_mutex);
					m_ready.wait(lock, [this] { return !m_ranges.empty(); });
					ranges.swap(m_ranges);
				}
				for (const auto& [vaddr, size]: ranges) TrimParkedRange(vaddr, size);
				ranges.clear();
			}
		}).detach();
	}
	std::mutex                                 m_mutex;
	std::condition_variable                    m_ready;
	std::vector<std::pair<uint64_t, uint64_t>> m_ranges;
};
#endif

// Guest ranges that still hold their data but were mapped again with their view's protection:
// the neighbours of a partly unmapped view (a Windows view cannot be split, so the whole view is
// unmapped and its remaining parts mapped again) and views an unmap rolled back. The GPU
// trackers' write and read protection of their pages was lost with the old view; without it CPU
// writes to watched pages went unseen and the GPU kept using stale data (a frozen character
// model). It goes back once the address-space lock is released (GuestAddressSpace::UnmapBacking;
// the page manager protects through that lock).
thread_local std::vector<std::pair<uint64_t, uint64_t>> g_remapped_views;
// The view a partial unmap maps again, from its unmap to the end of UnmapBacking (InRemapWindow).
static std::atomic<uint64_t> g_remap_window_begin {0};
static std::atomic<uint64_t> g_remap_window_end {0};

// kyty_local_protective_remap as UnmapBacking read it: its views are protected by ProtectRemapped or left to
// ReapplyRemappedViews, not both.
thread_local bool     g_protective_remap = true;
thread_local uint32_t g_remapped_count   = 0;
static bool ProtectiveRemap() {
	return g_protective_remap;
}

// KYTY_NATIVE_MEMORY=1 (Windows; off by default, read once): the guest's direct and flexible memory is private memory at
// its guest address (VirtualAlloc2 over the placeholder, MEM_WRITE_WATCH) instead of views of a page-file section, and
// the emulator's backing pointers are the guest addresses themselves (no second view: Memory::NativeWriteWatch reports
// the pages written). Paged while their protection changed, the section's pages were the bug check 0x1A/0x41202.
// What changes with it (memory.h, NativeMemoryEnabled): no alias of a physical range at two guest addresses (refused),
// no bytes of a physical range kept while nothing maps it, and the backing accessors no longer bypass guest protection.
static bool NativeMemoryRequested();

// (KYTY_NATIVE_MEMORY) KYTY_NATIVE_PARK_DECOMMIT=1: a guest unmap decommits the parked pages at once (the first native
// builds). Off by default: they stay committed (and are trimmed, as the backing views' parked pages always were) until a
// map takes them (AdoptParkedNative: decommitted and committed again there, zeros) or their allocation goes. The emulator
// writes into guest memory the game just unmapped (GPU results written back, offloaded textures restored, file reads and
// labels queued before the unmap): through the backing view those writes landed in the section, harmless; decommitted,
// each one was an access violation in the exe (state MEM_RESERVE, type MEM_PRIVATE: "Unhandled host exception").
[[maybe_unused]] static bool NativeParkDecommit() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_NATIVE_PARK_DECOMMIT");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

static bool NativeMemoryRequested();

// (nativemem1) KYTY_DS_FREED_DISCARD=1 with KYTY_NATIVE_MEMORY=1: parked ranges discarded, not trimmed
// (GuestAddressSpace::TrimParked).
[[maybe_unused]] static bool NativeParkDiscard() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DS_FREED_DISCARD");
		const bool  asked = value != nullptr && value[0] == '1';
		if (asked && NativeMemoryRequested()) {
			std::printf("Freed discard (native memory): the game's unmapped ranges are discarded (DiscardVirtualMemory), "
			            "not trimmed to the page file\n");
			std::fflush(stdout);
		}
		return asked && NativeMemoryRequested();
	}();
	return on;
}

static bool NativeMemoryRequested() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static const bool on = [] {
		(void)Raiz::On(); // (raiz1: KYTY_RAIZ=1 forces KYTY_NATIVE_MEMORY=1, KYTY_WRITE_WATCH=1, mode protect)
		const char* value = std::getenv("KYTY_NATIVE_MEMORY");
		return value != nullptr && value[0] == '1';
	}();
	return on;
#else
	return false;
#endif
}

// (nativemem1) KYTY_WRITE_WATCH=1 with KYTY_NATIVE_MEMORY=1 (Windows; off by default; read once, at first use: the
// launcher sets it after the exe started): native memory with the read faults kept (memory.h, NativeReadFaultsEnabled).
// The trackers' no-access protection of GPU-written pages applies again (PageManager), their read-only protection never
// (the write watch sees the CPU's writes). With no backing view to bypass it, the emulator's own copies over such pages
// go through NativeGuardedAccess.
static bool NativeReadFaultsRequested() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static const bool on = [] {
		(void)Raiz::On(); // (raiz1)
		const char* value = std::getenv("KYTY_WRITE_WATCH");
		const bool  asked = value != nullptr && value[0] == '1';
		if (asked && !NativeMemoryRequested()) {
			std::printf("Write watch: KYTY_WRITE_WATCH=1 ignored: it needs KYTY_NATIVE_MEMORY=1 (private guest memory)\n");
			std::fflush(stdout);
		}
		return asked && NativeMemoryRequested();
	}();
	return on;
#else
	return false;
#endif
}

// (ww2) Memory::NativeWriteWatchCounters: GetWriteWatch calls since start, and those refused and made again a region
// at a time (a range over two allocations, or one not write-watched).
static std::atomic<uint64_t> g_native_watch_calls {0};
// (raiz2) Set by the start-up self-test while it maps one physical range twice (memory.cpp).
[[maybe_unused]] static bool g_native_alias_selftest = false;
static std::atomic<uint64_t> g_native_watch_retries {0};

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// (nativemem1) Every protection change of guest memory (ProtectMappedUnlocked) and every opening of NativeGuardedAccess
// take it (KYTY_WRITE_WATCH only): no protection changes while a copy has pages opened, so the protection it restores is
// still the right one, and none is added under a copy that found its pages accessible.
// (raiz1) A gate instead of one recursive mutex: the protection changes take it SHARED - they ran side by side before
// native memory too (under the address space's shared lock): the mutex had put each of the ~34,000 a second behind the
// others -, a window (NativeGuardedAccess, NativeOpenParked) takes it ALONE. Re-entry on the thread that holds it is not
// a second take (the old mutex was recursive for that: a fault inside a copy - its other side guest memory - may protect
// or open pages on the same thread): t_native_window_depth / t_native_protect_depth. KYTY_RAIZ_VERIFY: the protection
// changes take it alone too (the audit reads and resets the write watch around them, NativeAuditProtect).
static std::shared_mutex& NativeProtectGate() {
	static std::shared_mutex gate;
	return gate;
}
static thread_local uint32_t t_native_window_depth  = 0;
static thread_local uint32_t t_native_protect_depth = 0;
// Windows waiting for the gate: a protection change lets them in first (a shared lock taken by one thread after another
// - ~34,000 a second - would keep a window out for good: reader preference, seen in the stub test with glibc).
static std::atomic<uint32_t> g_native_windows_waiting {0};

// (raiz5) A wait at the gate: counted, and on the GPU (render) thread timed apart ("[raiz]": its waits and their ms).
struct NativeGateWait {
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	NativeGateWait() { Raiz::g_counters.gate_waits.fetch_add(1, std::memory_order_relaxed); }
	~NativeGateWait() {
		const bool     gpu = Graphics::GuestGpu::IsGpuThread();
		const uint64_t us  = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
		// (raiz9: every class's waits and their time, the upload threads' apart.)
		auto& cost = Raiz::g_cost[Raiz::ClassOf(gpu)];
		cost.gate_waits.fetch_add(1, std::memory_order_relaxed);
		cost.gate_wait_us.fetch_add(us, std::memory_order_relaxed);
		if (!gpu) return;
		Raiz::g_counters.gate_waits_gpu.fetch_add(1, std::memory_order_relaxed);
		Raiz::g_counters.gate_wait_us_gpu.fetch_add(us, std::memory_order_relaxed);
	}
};

class NativeWindowLock {
public:
	// (raiz9) `kind`: what holds the gate alone, for the "[raiz]" line's time held (0 a lazy keep, 1 a window or other).
	// (raiz10) `take` false: nothing at all (NativeGuardedWrite's keep with KYTY_RAIZ_KEEP_STRIPES).
	explicit NativeWindowLock(uint32_t kind = 1, bool take = true): m_kind(kind < 2 ? kind : 1), m_skip(!take) {
		if (m_skip) return;
		if (t_native_window_depth++ != 0 || t_native_protect_depth != 0) return; // (held by this thread already)
		if (!NativeProtectGate().try_lock()) {
			NativeGateWait  wait;
			LockWatch::Wait watched(LockWatch::GateAlone); // (ram13b)
			g_native_windows_waiting.fetch_add(1, std::memory_order_acq_rel);
			NativeProtectGate().lock();
			g_native_windows_waiting.fetch_sub(1, std::memory_order_acq_rel);
		} else {
			LockWatch::Held(LockWatch::GateAlone); // (ram13b)
		}
		m_locked = true;
		if (Raiz::On()) m_held = std::chrono::steady_clock::now();
	}
	~NativeWindowLock() {
		if (m_skip) return;
		--t_native_window_depth;
		if (m_locked) {
			if (Raiz::On()) {
				Raiz::g_gate_alone_n[m_kind].fetch_add(1, std::memory_order_relaxed);
				Raiz::g_gate_alone_us[m_kind].fetch_add(
				    static_cast<uint64_t>(
				        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_held).count()),
				    std::memory_order_relaxed);
			}
			NativeProtectGate().unlock();
			LockWatch::Release(LockWatch::GateAlone); // (ram13b)
		}
	}
	KYTY_CLASS_NO_COPY(NativeWindowLock);

private:
	bool                                  m_locked = false;
	uint32_t                              m_kind   = 1;
	bool                                  m_skip   = false; // (raiz10)
	std::chrono::steady_clock::time_point m_held {};
};

class NativeProtectLock {
public:
	explicit NativeProtectLock(bool wanted) {
		if (!wanted || t_native_window_depth != 0 || t_native_protect_depth++ != 0) {
			if (wanted && t_native_window_depth == 0) m_counted = true; // (depth was raised: undone below)
			return;
		}
		m_counted   = true;
		m_exclusive = Raiz::VerifyOn();
		if (m_exclusive) {
			LockWatch::Wait watched(LockWatch::GateAlone); // (ram13b)
			NativeProtectGate().lock();
		} else {
			if (g_native_windows_waiting.load(std::memory_order_acquire) != 0) {
				NativeGateWait  wait;
				LockWatch::Wait watched(LockWatch::GateShared); // (ram13b: the spin behind a window waiting)
				while (g_native_windows_waiting.load(std::memory_order_acquire) != 0) std::this_thread::yield();
			}
			if (!NativeProtectGate().try_lock_shared()) {
				NativeGateWait  wait;
				LockWatch::Wait watched(LockWatch::GateShared); // (ram13b)
				NativeProtectGate().lock_shared();
			} else {
				LockWatch::Held(LockWatch::GateShared); // (ram13b)
			}
		}
		m_locked = true;
	}
	~NativeProtectLock() {
		if (m_counted) --t_native_protect_depth;
		if (!m_locked) return;
		if (m_exclusive) {
			NativeProtectGate().unlock();
			LockWatch::Release(LockWatch::GateAlone); // (ram13b)
		} else {
			NativeProtectGate().unlock_shared();
			LockWatch::Release(LockWatch::GateShared); // (ram13b)
		}
	}
	KYTY_CLASS_NO_COPY(NativeProtectLock);

private:
	bool m_counted = false, m_locked = false, m_exclusive = false;
};

// ---- (raiz10) KYTY_RAIZ_KEEP_STRIPES: the lazy keeps off the gate's exclusive hold ------------------------------------
// 20261116-044058 (raiz9): 16,188 lazy keeps a second (NativeGuardedWrite: a GPU result for no-access pages kept aside)
// took the gate ALONE - 0.58 ms/frame held -, and each one stopped every protection change of every thread: a protection
// waits for a window waiting, then for the keep (the game's threads 53 gate waits, 1.56 ms a frame; the GPU thread 0.66).
// What a keep needs is narrower: between its check (the pages no-access) and its store, no protection change may make
// THOSE pages readable (it would take the kept bytes out before they are in, and leave them behind a readable page), and
// no window may open them (a window holds the gate alone: a keep's shared hold keeps it out). So now (default; =0: as
// before) a keep holds the gate SHARED and the stripes of its range alone, and a protection change that makes pages
// readable holds its range's stripes shared around the kept bytes' take and its VirtualProtect. A stripe: 1 MiB of
// address, 512 of them (hashed: the address's MiB modulo 512; a range over 512 MiB or more takes them all). Taken in
// ascending order (the two parts of a range that wraps the table: the low one first), after the gate, before the lazy
// store's mutex: keeps among themselves, and protections, cannot deadlock. A protection change to no access takes none
// (a keep then either saw the page no-access - its bytes kept, as before - or not - a window writes them, as before -).
static bool NativeKeepStripesOn() {
	static const bool on = [] {
		const char* v   = std::getenv("KYTY_RAIZ_KEEP_STRIPES");
		const bool  set = !(v != nullptr && v[0] == '0');
		if (Raiz::On()) {
			std::printf("Raiz: lazy keeps hold the gate %s (KYTY_RAIZ_KEEP_STRIPES%s)\n",
			            set ? "shared and their 1 MiB stripes alone" : "alone (as raiz9)", set ? "=0 for the gate alone" : "=0");
			std::fflush(stdout);
		}
		return set;
	}();
	return on;
}
static constexpr uint32_t    NativeStripeCount = 512;
static constexpr uint32_t    NativeStripeShift = 20;
static std::shared_mutex     g_native_stripes[NativeStripeCount];
static thread_local uint32_t t_native_stripe_depth = 0; // (a nested take - none known - takes nothing more)

template <bool Alone>
class NativeStripeLock {
public:
	NativeStripeLock(uint64_t vaddr, uint64_t size, bool wanted) {
		if (!wanted || size == 0 || t_native_stripe_depth != 0) return;
		const uint64_t c0 = vaddr >> NativeStripeShift, c1 = (vaddr + size - 1u) >> NativeStripeShift;
		if (c1 - c0 + 1u >= NativeStripeCount) {
			m_a0 = 0, m_a1 = NativeStripeCount - 1u;
		} else {
			const auto i0 = static_cast<uint32_t>(c0 % NativeStripeCount), i1 = static_cast<uint32_t>(c1 % NativeStripeCount);
			if (i0 <= i1) {
				m_a0 = i0, m_a1 = i1;
			} else { // (wraps: [0, i1] first, then [i0, last])
				m_a0 = 0, m_a1 = i1, m_b0 = i0, m_b1 = NativeStripeCount - 1u, m_two = true;
			}
		}
		++t_native_stripe_depth;
		m_locked = true;
		std::chrono::steady_clock::time_point start {};
		bool                                  waited = false;
		Take(m_a0, m_a1, &waited, &start);
		if (m_two) Take(m_b0, m_b1, &waited, &start);
		if (waited) {
			Raiz::g_stripe_waits[Alone ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
			Raiz::g_stripe_wait_us[Alone ? 0 : 1].fetch_add(
			    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()),
			    std::memory_order_relaxed);
		}
	}
	~NativeStripeLock() {
		if (!m_locked) return;
		if (m_two) Give(m_b0, m_b1);
		Give(m_a0, m_a1);
		--t_native_stripe_depth;
	}
	KYTY_CLASS_NO_COPY(NativeStripeLock);

private:
	static void Take(uint32_t from, uint32_t to, bool* waited, std::chrono::steady_clock::time_point* start) {
		for (uint32_t i = from; i <= to; ++i) {
			auto& stripe = g_native_stripes[i];
			if (Alone ? stripe.try_lock() : stripe.try_lock_shared()) continue;
			if (!*waited) *start = std::chrono::steady_clock::now();
			*waited = true;
			{
				LockWatch::Wait watched(Alone ? LockWatch::StripeAlone : LockWatch::StripeShared);
				if (Alone) stripe.lock();
				else stripe.lock_shared();
			}
			LockWatch::Release(Alone ? LockWatch::StripeAlone : LockWatch::StripeShared); // (the wait watched, not the hold)
		}
	}
	static void Give(uint32_t from, uint32_t to) {
		for (uint32_t i = to + 1u; i-- > from;) {
			if (Alone) g_native_stripes[i].unlock();
			else g_native_stripes[i].unlock_shared();
		}
	}
	uint32_t m_a0 = 0, m_a1 = 0, m_b0 = 0, m_b1 = 0;
	bool     m_two = false, m_locked = false;
};

static std::atomic<uint64_t> g_native_protects {0};      // protection changes of guest memory (native memory only)
static std::atomic<uint64_t> g_native_copy_protects {0}; // NativeGuardedAccess's (opened + restored)
static std::atomic<uint64_t> g_native_guarded_copies {0};

// Whether [vaddr, vaddr + size) may hold pages a tracker made no-access (read watchers, the offload's holds): the page
// manager's read hints, a superset of them (set before such a protection, cleared after the release). False for memory
// no tracker covers.
static bool NativeMayBeProtected(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr && size != 0 && IsGpuAddressRange(vaddr, size) && g_gpu_resources->HasReadWatchers(vaddr, size);
}

// (ww2, KYTY_WRITE_WATCH_MODE=protect only; false otherwise) Whether [vaddr, vaddr + size) may hold pages a write
// watcher made read-only (PageManager::HasWriteProtection, a hint): there is no backing view to write them through.
static bool NativeMayBeWriteProtected(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr && size != 0 && IsGpuAddressRange(vaddr, size) && g_gpu_resources->HasWriteProtection(vaddr, size);
}

// (raiz1, KYTY_RAIZ_VERIFY) The pages of [at, at + size) - inside one allocation: one region of VirtualQuery - the write
// watch holds written since its last reset, read and reset. Returned: their count. `lost_where` set: each is a write to
// a page that was protected all along (the caller's knowledge), with no fault: noted as a lost write.
static uint64_t RaizTakeWritten(uint64_t at, uint64_t size, const char* lost_where, bool reset = true) {
	if (size == 0) return 0;
	const auto                       start = std::chrono::steady_clock::now();
	constexpr ULONG_PTR              Batch = 1024;
	thread_local std::vector<void*>  found;
	found.resize(Batch);
	uint64_t       total = 0;
	const uint64_t end   = at + size;
	for (uint64_t from = at; from < end;) {
		ULONG_PTR count       = Batch;
		DWORD     granularity = 0;
		g_native_watch_calls.fetch_add(1, std::memory_order_relaxed);
		if (GetWriteWatch(reset ? WRITE_WATCH_FLAG_RESET : 0, reinterpret_cast<void*>(from), end - from, found.data(), &count,
		                  &granularity) != 0)
			break; // (an allocation without MEM_WRITE_WATCH: Windows refused it over a placeholder, said at the map)
		for (ULONG_PTR i = 0; i < count; ++i) {
			++total;
			if (lost_where != nullptr) Raiz::NoteLost(reinterpret_cast<uint64_t>(found[i]), lost_where);
		}
		if (count < Batch) break;
		from = reinterpret_cast<uint64_t>(found[count - 1]) + 0x1000u;
	}
	Raiz::g_counters.verify_checked.fetch_add(size >> 12u, std::memory_order_relaxed);
	Raiz::g_counters.verify_us.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                                               std::chrono::steady_clock::now() - start)
	                                                               .count()),
	                                     std::memory_order_relaxed);
	return total;
}

// (raiz1) Whether a page protection (low byte) lets nothing write it: what the trackers set (read-only, no-access).
static bool RaizProtected(DWORD protect) {
	const DWORD p = protect & 0xffu;
	return p == PAGE_NOACCESS || p == PAGE_READONLY || p == PAGE_EXECUTE_READ || p == PAGE_EXECUTE;
}

// (raiz1, KYTY_RAIZ_VERIFY; the caller holds the gate alone) Around a protection change of guest memory [vaddr, vaddr +
// size) the GPU trackers cover: `before` (true, ahead of the change), the runs protected now are read - a page written
// there since it became protected took no fault: a lost write -; after it (false), when the new protection is
// read-only or no-access, the write watch of the range is reset (what was written while it was writable was seen or
// was not the trackers' to see), so the next audit finds only writes made while protected.
static void NativePendingDropStale(uint64_t vaddr, uint64_t size);
static void NativeAuditProtect(uint64_t vaddr, uint64_t size, bool before, bool now_protected) {
	if (!Raiz::VerifyOn() || size == 0 || !IsGpuAddressRange(vaddr, size)) return;
	if (!before && !now_protected) return;
	const uint64_t end = vaddr + size;
	for (uint64_t at = vaddr; at < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) break;
		const uint64_t stop = std::min(end, reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize));
		if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE && stop > at) {
			// (raiz6: read, not reset - the lazy write-back's take reads the same bits right after, NativePendingTake: a
			// page written while kept must still show it -; the reset after the change, below, when it protects.)
			if (before && RaizProtected(info.Protect)) (void)RaizTakeWritten(at, stop - at, "before a protection change", false);
			if (!before) {
				// (raiz6: the kept bytes written over first dropped - this reset would hide that from NativePendingTake.)
				NativePendingDropStale(at, stop - at);
				(void)RaizTakeWritten(at, stop - at, nullptr);
			}
		}
		at = std::max(stop, at + 0x1000u);
	}
}

// (raiz2) A run of pages a window opened.
struct NativeOpenedRun {
	uint64_t at, size;
	DWORD    old;
	bool     write; // opened for a write (a read-only page's resync at its restore)
};

// (raiz2) The runs given their protection back at a window's end: counters, KYTY_RAIZ_WINDOW_RESYNC's
// queue for the pages a write opened from read-only, KYTY_RAIZ_VERIFY's check. `runs` in the order they were opened.
static void NativeRestoreRuns(const NativeOpenedRun* runs, size_t count, uint64_t open_us) {
	constexpr uint64_t Page   = 0x1000;
	const bool         raiz   = Raiz::On();
	const bool         verify = Raiz::VerifyOn();
	uint64_t           calls  = 0;
	// (Last opened first: a page opened read-only for a read, then read-write for a write, gets read-only, then its own.)
	for (size_t i = count; i-- > 0;) {
		uint64_t at = runs[i].at, size = runs[i].size;
		// Adjacent runs (going down) with the same protection to give back: one call.
		while (i > 0 && runs[i - 1].old == runs[i].old && runs[i - 1].at + runs[i - 1].size == at) {
			--i;
			at = runs[i].at;
			size += runs[i].size;
		}
		DWORD ignored = 0;
		(void)VirtualProtect(reinterpret_cast<void*>(at), size, runs[i].old, &ignored);
		++calls;
	}
	g_native_copy_protects.fetch_add(calls, std::memory_order_relaxed);
	if (!raiz) return;
	auto& c = Raiz::g_counters;
	c.window_protects.fetch_add(calls, std::memory_order_relaxed);
	for (size_t i = 0; i < count; ++i) {
		const auto&    run           = runs[i];
		const DWORD    old           = run.old & 0xffu;
		const uint64_t pages         = run.size / Page;
		const bool     was_read_only = old == PAGE_READONLY || old == PAGE_EXECUTE_READ;
		(was_read_only ? c.window_ro_pages : c.window_na_pages).fetch_add(pages, std::memory_order_relaxed);
		if (old == PAGE_NOACCESS) Raiz::NoteExposure(run.at, run.size, open_us, run.write); // (raiz2)
		if (run.write && was_read_only && Raiz::ResyncOn()) Raiz::QueueResync(run.at, run.at + run.size);
		if (verify) {
			const uint64_t written = RaizTakeWritten(run.at, run.size, nullptr);
			if (!run.write && written != 0) {
				c.verify_read_window.fetch_add(written, std::memory_order_relaxed);
				static std::atomic<uint32_t> said {0};
				if (said.fetch_add(1, std::memory_order_relaxed) < 16) {
					std::printf("[raiz] VERIFY: %llu page(s) of 0x%llx+0x%llx written during an emulator READ window (must "
					            "not happen: opened read-only)\n",
					            static_cast<unsigned long long>(written), static_cast<unsigned long long>(run.at),
					            static_cast<unsigned long long>(run.size));
					std::fflush(stdout);
				}
			}
		}
	}
}

// ---- (raiz3) Lazy write-back: GPU results for NO-ACCESS pages kept aside, put in the page only when it opens --------
// 11-16 (raiz2, Boletaria): 95,600 no-access pages/s opened read-write for 74.5 us each to write GPU results back
// (readback 7.7-10.3 k windows/s, feedback 4.7 k/s), the game's threads able to read them half-written meanwhile
// (estimated 17 reads/s inside: the flashes' likely cause), 3.2 ms/frame of windows. Now a write-back (WriteBacking's
// readback / feedback / download) over pages that are no-access does not touch them: its bytes go to this store
// (a page's bytes and a mask of those written), with the gate held alone so no protection change slips between the
// check and the store. The page stays no-access - a game read faults as on the console's timeline -, and its bytes go
// in at the one moment the page opens:
//  - a protection change that makes it readable (ProtectMappedUnlocked: the read fault's release, a readback's
//    release, a guest mprotect): read-write for the copy, then its new protection;
//  - an emulator window over it (NativeGuardedAccess: read or write; a read's window opens it read-write for the copy
//    first, then read-only);
//  - the native remap's copy of a view (rare).
// A map or unmap over a page drops what is kept for it. KYTY_RAIZ_LAZY=0: off (raiz2's windows). KYTY_RAIZ_LAZY_MB
// (default 256): the store's ceiling; past it, write-backs open windows as before (counted).
struct NativePendingPage {
	uint8_t  bytes[4096];
	uint64_t mask[64]; // a bit per byte written
	// (raiz7) Copies into `bytes` still running (NativePendingStore's second phase, outside the gate and the mutex):
	// whoever takes the page out or drops it waits for 0 first (NativePendingWaitFilled).
	std::atomic<uint32_t> filling {0};
};
static void NativePendingWaitFilled(const NativePendingPage& page) {
	while (page.filling.load(std::memory_order_acquire) != 0) std::this_thread::yield();
}
// (raiz7) A copy of NativePendingStore's second phase: [from, from + len) of the caller's bytes into `page` at `off`.
struct NativePendingFill {
	NativePendingPage* page;
	uint64_t           off, len;
	const uint8_t*     from;
};
static std::mutex                                                      g_pending_mutex;
static std::unordered_map<uint64_t, std::unique_ptr<NativePendingPage>> g_pending;
// (raiz5) Pages given back after their apply, reused (raiz4 allocated and freed one per page: ~76,000 a second).
static std::vector<std::unique_ptr<NativePendingPage>> g_pending_free;
static std::atomic<uint64_t>                                           g_pending_count {0};
// Kept pages per bucket of 64 KiB granules (hashed): NativePendingIn asks the mutex only where some are (the ~60,000
// protection changes a second check it).
static std::atomic<uint32_t> g_pending_buckets[16384];
static size_t                NativePendingBucket(uint64_t page) {
	return static_cast<size_t>(((page >> 16u) * 0x9E3779B97F4A7C15ull) >> 50u) % 16384u;
}

// ---- (raiz5) Shadows of no-access pages for the emulator's reads ----------------------------------------------------
// 11-16 (raiz4, Boletaria): 21,000-25,000 read windows/s (the render thread's reads of GPU-clean bytes on pages a read
// watcher holds: SRT spans, tables, indirect arguments - TryReadGpuCleanBacking* -), each the gate alone, a VirtualQuery
// and two VirtualProtect: 1.7-2.0 ms/frame of windows, gate waits 7,000-9,900/s. The backing view read them for
// nothing. A no-access page's bytes change only when its protection changes (the game cannot write it; it opens at a
// protection change), when the emulator writes it (a window, a kept write-back) or when it is mapped or unmapped: so the
// bytes a read window sees are kept (a 4 KiB copy a page) and serve the next reads of that page with no window, until
// one of those events forgets them. KYTY_RAIZ_SHADOW=0: off. At most 16,384 pages (64 MiB); past it, none more kept.
struct NativeShadowPage {
	uint8_t bytes[4096];
};
static std::shared_mutex                                               g_shadow_mutex;
static std::unordered_map<uint64_t, std::unique_ptr<NativeShadowPage>> g_shadow;
static std::atomic<uint64_t>                                           g_shadow_count {0};
static std::atomic<uint32_t>                                           g_shadow_buckets[16384];
static constexpr size_t                                                NativeShadowMax = 16384;
static thread_local bool t_native_fill_shadow = false; // (set by NativeGuardedReadFrom for its window)

static bool NativeShadowOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RAIZ_SHADOW");
		return Raiz::On() && !(v != nullptr && v[0] == '0');
	}();
	return on;
}
static size_t NativeShadowBucket(uint64_t page) {
	return static_cast<size_t>(((page >> 16u) * 0x9E3779B97F4A7C15ull) >> 50u) % 16384u;
}
static bool NativeShadowMaybe(uint64_t vaddr, uint64_t size) {
	if (g_shadow_count.load(std::memory_order_relaxed) == 0 || size == 0) return false;
	for (uint64_t granule = vaddr & ~uint64_t {0xffff}; granule < vaddr + size; granule += 0x10000)
		if (g_shadow_buckets[NativeShadowBucket(granule)].load(std::memory_order_relaxed) != 0) return true;
	return false;
}
// The shadows of [vaddr, vaddr + size) forgotten (a protection change, a write, a map or unmap there).
static void NativeShadowDrop(uint64_t vaddr, uint64_t size) {
	if (!NativeShadowMaybe(vaddr, size)) return;
	std::unique_lock lock(g_shadow_mutex);
	uint64_t         dropped = 0;
	const uint64_t   first   = vaddr & ~uint64_t {0xfff};
	if (((vaddr + size - first + 0xfff) >> 12u) < g_shadow.size()) {
		for (uint64_t page = first; page < vaddr + size; page += 0x1000) {
			auto it = g_shadow.find(page);
			if (it == g_shadow.end()) continue;
			g_shadow_buckets[NativeShadowBucket(page)].fetch_sub(1, std::memory_order_relaxed);
			g_shadow.erase(it);
			++dropped;
		}
	} else {
		for (auto it = g_shadow.begin(); it != g_shadow.end();) {
			if (it->first >= first && it->first < vaddr + size) {
				g_shadow_buckets[NativeShadowBucket(it->first)].fetch_sub(1, std::memory_order_relaxed);
				it = g_shadow.erase(it);
				++dropped;
			} else {
				++it;
			}
		}
	}
	g_shadow_count.fetch_sub(dropped, std::memory_order_relaxed);
	Raiz::g_counters.shadow_drops.fetch_add(dropped, std::memory_order_relaxed);
}
// The pages of [at, at + size) - open for reading now, inside a window - kept (those not kept yet, while there is room).
static void NativeShadowFill(uint64_t at, uint64_t size) {
	std::unique_lock lock(g_shadow_mutex);
	uint64_t         filled = 0;
	for (uint64_t page = at & ~uint64_t {0xfff}; page < at + size; page += 0x1000) {
		if (g_shadow.size() >= NativeShadowMax) {
			Raiz::g_counters.shadow_full.fetch_add(1, std::memory_order_relaxed);
			break;
		}
		auto& slot = g_shadow[page];
		if (slot) continue;
		slot = std::make_unique<NativeShadowPage>();
		std::memcpy(slot->bytes, reinterpret_cast<const void*>(page), 4096);
		g_shadow_buckets[NativeShadowBucket(page)].fetch_add(1, std::memory_order_relaxed);
		++filled;
	}
	g_shadow_count.fetch_add(filled, std::memory_order_relaxed);
	Raiz::g_counters.shadow_fills.fetch_add(filled, std::memory_order_relaxed);
}
// [vaddr, vaddr + bytes) from the shadows into `out` when every page of it has one. False otherwise (nothing copied).
static bool NativeShadowRead(uint64_t vaddr, uint64_t bytes, uint8_t* out) {
	if (!NativeShadowMaybe(vaddr, bytes)) return false;
	std::shared_lock lock(g_shadow_mutex);
	const NativeShadowPage* pages[64];
	const uint64_t          first = vaddr & ~uint64_t {0xfff};
	const uint64_t          count = ((vaddr + bytes - first) + 0xfff) >> 12u;
	if (count > 64) return false;
	for (uint64_t i = 0; i < count; ++i) {
		auto it = g_shadow.find(first + (i << 12u));
		if (it == g_shadow.end()) return false;
		pages[i] = it->second.get();
	}
	for (uint64_t at = vaddr, end = vaddr + bytes; at < end;) {
		const uint64_t page = at & ~uint64_t {0xfff};
		const uint64_t stop = std::min(end, page + 0x1000);
		std::memcpy(out + (at - vaddr), pages[(page - first) >> 12u]->bytes + (at - page), stop - at);
		at = stop;
	}
	return true;
}

static bool NativeLazyOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RAIZ_LAZY");
		return Raiz::On() && !(v != nullptr && v[0] == '0');
	}();
	return on;
}
static uint64_t NativeLazyMaxPages() {
	static const uint64_t pages = [] {
		const char* v  = std::getenv("KYTY_RAIZ_LAZY_MB");
		uint64_t    mb = (v != nullptr && v[0] >= '0' && v[0] <= '9') ? std::strtoull(v, nullptr, 10) : 256u;
		return mb * 1024u * 1024u / sizeof(NativePendingPage);
	}();
	return pages;
}
// Whether this thread's write is a write-back of GPU results (Raiz::t_site, WriteBacking's source).
static bool NativeLazySite() {
	return Raiz::t_site == Raiz::SiteReadback || Raiz::t_site == Raiz::SiteFeedback || Raiz::t_site == Raiz::SiteDownload;
}

// [from, from + bytes) of `data` kept for guest [vaddr, vaddr + bytes) (all in no-access pages). False: the store is full
// (nothing kept of the pages not yet there).
// (raiz4) Bits [from, to) of a page's mask set, a 64-bit word at a time.
static void NativeMaskSet(uint64_t* mask, uint64_t from, uint64_t to) {
	while (from < to) {
		const uint64_t word = from >> 6u, bit = from & 63u;
		const uint64_t n    = std::min<uint64_t>(64u - bit, to - from);
		mask[word] |= (n == 64u ? ~uint64_t {0} : ((uint64_t {1} << n) - 1u)) << bit;
		from += n;
	}
}

// (raiz7) Phase 1 only, under the gate and the mutex: the pages made (or found) and their masks set; the copies are
// appended to `fills` for the caller to run after it let both go (NativePendingRunFills). raiz6 copied under both: the
// gate held alone by the readback thread while it copied (keeping 1.42 ms/frame there), the GPU thread's protections
// waiting behind it (GPU thread gate waits 0.41 ms/frame, 015108).
static bool NativePendingStore(uint64_t vaddr, const uint8_t* data, uint64_t bytes, std::vector<NativePendingFill>* fills) {
	Raiz::g_settle_epoch.fetch_add(1, std::memory_order_acq_rel); // (raiz12: a keep - KYTY_RAIZ_SETTLE's memo -)
	std::lock_guard lock(g_pending_mutex);
	// (Room for every new page first: all of the range kept, or none.)
	uint64_t fresh = 0;
	for (uint64_t page = vaddr & ~uint64_t {0xfff}; page < vaddr + bytes; page += 0x1000) fresh += g_pending.count(page) == 0 ? 1u : 0u;
	if (g_pending.size() + fresh > NativeLazyMaxPages()) return false;
	for (uint64_t at = vaddr, end = vaddr + bytes; at < end;) {
		const uint64_t page = at & ~uint64_t {0xfff};
		const uint64_t stop = std::min(end, page + 0x1000);
		auto&          slot = g_pending[page];
		if (!slot) {
			if (!g_pending_free.empty()) {
				slot = std::move(g_pending_free.back());
				g_pending_free.pop_back();
			} else {
				slot = std::make_unique<NativePendingPage>();
			}
			std::memset(slot->mask, 0, sizeof(slot->mask));
			g_pending_count.fetch_add(1, std::memory_order_relaxed);
			g_pending_buckets[NativePendingBucket(page)].fetch_add(1, std::memory_order_relaxed);
			Raiz::g_lazy_new.fetch_add(1, std::memory_order_relaxed); // (raiz4: pages new to the store)
		}
		const uint64_t off = at - page, len = stop - at;
		// (raiz4: the mask a word at a time - raiz3 set it a bit at a time, 4,096 steps a whole page, ~61,000 pages/s -.)
		NativeMaskSet(slot->mask, off, off + len);
		slot->filling.fetch_add(1, std::memory_order_relaxed);
		fills->push_back({slot.get(), off, len, data + (at - vaddr)});
		at = stop;
	}
	return true;
}

// (raiz7) Phase 2: the copies, with no lock held; each page's `filling` released when its copy is in.
static void NativePendingRunFills(std::vector<NativePendingFill>& fills) {
	for (const auto& fill: fills) {
		std::memcpy(fill.page->bytes + fill.off, fill.from, fill.len);
		fill.page->filling.fetch_sub(1, std::memory_order_release);
	}
	fills.clear();
}

// Whether any page of [vaddr, vaddr + size) has kept bytes (one relaxed load when the store is empty).
static bool NativePendingIn(uint64_t vaddr, uint64_t size) {
	if (g_pending_count.load(std::memory_order_relaxed) == 0 || size == 0) return false;
	bool any = false;
	for (uint64_t granule = vaddr & ~uint64_t {0xffff}; granule < vaddr + size && !any; granule += 0x10000)
		any = g_pending_buckets[NativePendingBucket(granule)].load(std::memory_order_relaxed) != 0;
	if (!any) return false;
	std::lock_guard lock(g_pending_mutex);
	for (uint64_t page = vaddr & ~uint64_t {0xfff}; page < vaddr + size; page += 0x1000)
		if (g_pending.count(page) != 0) return true;
	return false;
}

// The kept bytes of [vaddr, vaddr + size) put in their pages - which the caller made writable - and forgotten.
// (raiz6) The kept pages of [vaddr, vaddr + size), taken out of the store BEFORE their pages open - so that opening
// and copying are back to back (raiz5's release opened a whole range, then looked each page up under the mutex: the
// game's threads could write the open pages for that long, and a copy after their write put GPU bytes over theirs:
// the 11-16 crash in TaggedFrameAllocator, a free-list node read back as 0) -. And a kept page that was written since
// it was kept (its write watch, reset when it was kept: the page was writable some time - a protection restored behind
// the trackers' back, as the cold store's probe does -) is dropped, never copied: its bytes are newer than the GPU's.
// `vaddr, size` inside one allocation (a VirtualQuery region, or one guest mapping); where the write watch cannot be
// read (an allocation without MEM_WRITE_WATCH) the pages are taken as before (counted).
using NativeTaken = std::vector<std::pair<uint64_t, std::unique_ptr<NativePendingPage>>>;
// KYTY_RAIZ_LAZY_CHECK=0: no write watch read at a take (one GetWriteWatch a released run less; the stale check off).
// (raiz7) Default: on only with KYTY_COLD_STORE=1, the one path known to make a kept page writable behind the
// trackers' back (the 01:48 crash). Without it: 0 stale pages in every log (raiz6 VERIFY 022938: total 0), and the
// check is a GetWriteWatch per released run on the GPU thread. KYTY_RAIZ_LAZY_CHECK=1 / =0 forces it.
static bool NativeLazyCheck() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RAIZ_LAZY_CHECK");
		if (v != nullptr && (v[0] == '0' || v[0] == '1')) return v[0] == '1';
		const char* cold = std::getenv("KYTY_COLD_STORE");
		return cold != nullptr && cold[0] == '1';
	}();
	return on;
}
// (raiz9) KYTY_RAIZ_KEEP_RESET=1: a lazy keep resets the write watch of its pages even when nothing reads it (raiz8).
static bool NativeKeepResetForced() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RAIZ_KEEP_RESET");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}
static void NativePendingTake(uint64_t vaddr, uint64_t size, NativeTaken* taken) {
	taken->clear();
	if (!NativePendingIn(vaddr, size)) return;
	const uint64_t first = vaddr & ~uint64_t {0xfff};
	const uint64_t end   = (vaddr + size + 0xfff) & ~uint64_t {0xfff};
	// The pages written since they were kept (no reset: the audit's and the next take's alike).
	thread_local std::vector<void*>    found;
	thread_local std::vector<uint64_t> written;
	written.clear();
	bool watch_ok = true;
	found.resize(1024);
	for (uint64_t from = first; NativeLazyCheck() && from < end;) {
		ULONG_PTR count       = found.size();
		DWORD     granularity = 0;
		if (GetWriteWatch(0, reinterpret_cast<void*>(from), end - from, found.data(), &count, &granularity) != 0) {
			watch_ok = false;
			break;
		}
		for (ULONG_PTR i = 0; i < count; ++i) written.push_back(reinterpret_cast<uint64_t>(found[i]));
		if (count < found.size()) break;
		from = reinterpret_cast<uint64_t>(found[count - 1]) + 0x1000u;
	}
	if (!watch_ok) Raiz::g_lazy_unchecked.fetch_add(1, std::memory_order_relaxed);
	std::lock_guard lock(g_pending_mutex);
	for (uint64_t page = first; page < end; page += 0x1000) {
		auto it = g_pending.find(page);
		if (it == g_pending.end()) continue;
		NativePendingWaitFilled(*it->second); // (raiz7: its copy in first)
		const bool stale = std::find(written.begin(), written.end(), page) != written.end();
		g_pending_count.fetch_sub(1, std::memory_order_relaxed);
		g_pending_buckets[NativePendingBucket(page)].fetch_sub(1, std::memory_order_relaxed);
		if (stale) {
			Raiz::g_lazy_stale.fetch_add(1, std::memory_order_relaxed);
			if (g_pending_free.size() < 4096) g_pending_free.push_back(std::move(it->second));
		} else {
			taken->emplace_back(page, std::move(it->second));
		}
		g_pending.erase(it);
	}
}

// (raiz6, KYTY_RAIZ_VERIFY) Before the audit resets the write watch of [vaddr, vaddr + size) (one allocation): the kept
// pages written since they were kept dropped now (as NativePendingTake would), the others kept.
static void NativePendingDropStale(uint64_t vaddr, uint64_t size) {
	if (!NativeLazyCheck()) return;
	thread_local NativeTaken taken;
	NativePendingTake(vaddr, size, &taken);
	if (taken.empty()) return;
	std::lock_guard lock(g_pending_mutex);
	for (auto& [page, kept]: taken) {
		auto& slot = g_pending[page];
		if (slot) continue;
		slot = std::move(kept);
		g_pending_count.fetch_add(1, std::memory_order_relaxed);
		g_pending_buckets[NativePendingBucket(page)].fetch_add(1, std::memory_order_relaxed);
	}
	taken.clear();
}

// (raiz6) The taken pages copied in (their pages writable now, opened right before) and given back to the pool.
static void NativePendingCopy(NativeTaken& taken, Raiz::LazyApply how) {
	if (!taken.empty()) Raiz::g_settle_epoch.fetch_add(1, std::memory_order_acq_rel); // (raiz12: KYTY_RAIZ_SETTLE's memo)
	// (raiz4: whole 64-byte words of the mask at once, and a fully written page in one copy - raiz3 walked the mask a
	// bit at a time, 4,096 steps a page, 52,000 pages/s at release -.)
	for (const auto& [page, kept]: taken) {
		auto*    dst  = reinterpret_cast<uint8_t*>(page);
		uint64_t full = ~uint64_t {0};
		for (const auto w: kept->mask) full &= w;
		if (full == ~uint64_t {0}) {
			std::memcpy(dst, kept->bytes, 4096);
			continue;
		}
		uint64_t run_begin = 0;
		bool     in_run    = false;
		for (uint64_t word = 0; word < 64; ++word) {
			const uint64_t m = kept->mask[word];
			if (m == ~uint64_t {0} || m == 0) {
				const bool on = m != 0;
				if (on && !in_run) run_begin = word * 64u, in_run = true;
				if (!on && in_run) std::memcpy(dst + run_begin, kept->bytes + run_begin, word * 64u - run_begin), in_run = false;
				continue;
			}
			for (uint64_t bit = 0; bit < 64; ++bit) {
				const bool     on = ((m >> bit) & 1u) != 0;
				const uint64_t at = word * 64u + bit;
				if (on && !in_run) run_begin = at, in_run = true;
				if (!on && in_run) std::memcpy(dst + run_begin, kept->bytes + run_begin, at - run_begin), in_run = false;
			}
		}
		if (in_run) std::memcpy(dst + run_begin, kept->bytes + run_begin, 4096u - run_begin);
	}
	if (!taken.empty()) { // (raiz5: the pages back to the pool, up to 4,096 of them)
		std::lock_guard lock(g_pending_mutex);
		for (auto& [page, kept]: taken)
			if (g_pending_free.size() < 4096) g_pending_free.push_back(std::move(kept));
	}
	if (!taken.empty()) Raiz::g_lazy_applied[how].fetch_add(taken.size(), std::memory_order_relaxed);
	taken.clear();
}

// ---- (crash-ba2b1f) Kept bytes put in with the game's threads running --------------------------------------------
// A kept page opens (VirtualProtect) for every thread at once: raiz3-8 opened it read-write and copied the GPU's bytes
// after, so a game write landing in between (a job worker writing the page the moment another thread's release opened
// it) was replaced by the GPU's older bytes, and one landing after the copy but before the page's final protection
// (read-only: clean, write-watched) reached no fault and no upload - the next write-back over the page undid it.
// 20261116-043751: shared_ptr control blocks of page 0x37b97e000 back to that page's old bytes, eboot+0xba2b1f.
// Now, a run of adjacent kept pages at a time:
//  1. read-only first: the bytes the pages hold are read with no game write possible (one faults and waits);
//  2. read-write, and the kept bytes put in 8 bytes at a time by compare-and-swap against what was read: a word the
//     game wrote meanwhile keeps the game's value (the game's write came later, as on the console);
//  3. a final protection other than read-write is applied here, then the pages are compared with what they should hold:
//     a difference is a game write made before that protection, with no fault.
// Pages with a game write found in 2 or 3: marked (local/raiz-wb-guard.h: no write-back over them while GPU-owned) and
// queued as CPU-written (uploaded again). Their write watch is then reset (the copy's own writes; KYTY_RAIZ_WB_GUARD).
// A run that cannot be opened goes back to the store, as before. KYTY_RAIZ_MERGE_PUTIN=0: raiz8's copy (open, then copy).
static bool NativeMergePutIn() {
	static const bool on = [] {
		const char* v  = std::getenv("KYTY_RAIZ_MERGE_PUTIN");
		const bool  on = Raiz::On() && !(v != nullptr && v[0] == '0');
		if (Raiz::On()) {
			std::printf("Raiz: kept GPU bytes put in %s (KYTY_RAIZ_MERGE_PUTIN%s; crash-ba2b1f)\n",
			            on ? "read-only first, then by compare-and-swap: a game write in the open page wins and one before "
			                 "its final protection is found"
			               : "as raiz8: opened read-write, then copied",
			            on ? "=0 for raiz8's" : "=0");
			std::fflush(stdout);
		}
		return on;
	}();
	return on;
}
// Bits [8w, 8w + 8) of a page's mask (a bit per byte) as a mask of those bytes.
static uint64_t NativeByteMask(uint64_t bits8) {
	if (bits8 == 0xffu) return ~uint64_t {0};
	uint64_t mask = 0;
	for (uint64_t b = 0; b < 8; ++b)
		if (((bits8 >> b) & 1u) != 0) mask |= uint64_t {0xff} << (b * 8u);
	return mask;
}
// ---- (raiz11) Put-ins under way, for the readers that read guest memory straight (no window) ---------------------
// A put-in makes its pages readable (read-only, step 1) before the kept GPU bytes are in (step 2): from the first
// VirtualProtect to the last compare-and-swap a direct reader - the upload worker's GuardedCopy (Memory::NativeGuardedRead),
// the GPU thread's reads after SyncGpuCleanBacking (indirect arguments, tables) - gets the page's OLD bytes with no fault:
// a GPU upload of stale data (a one-frame flash), the same with raiz8's copy (opened read-write, then copied). Each run
// under way marks its stripes (64 KiB of address, 4,096, hashed by address): `busy` while under way, `gen` moved at
// its start and at its end; and takes a slot of 64 with its exact range while under way (NativePutInExact). A reader takes both before and after its copy: busy, or gen moved = it may have read old bytes
// (counted, Raiz::g_stale_n; read again through a window, which waits for the put-in at the gate - KYTY_RAIZ_STALE_FIX,
// default on -). KYTY_RAIZ_STALE_CHECK=0: none of this.
struct NativePutInStripe {
	std::atomic<uint32_t> busy {0};
	std::atomic<uint32_t> gen {0};
};
static constexpr uint32_t NativePutInStripeCount = 4096;
static NativePutInStripe  g_putin_stripes[NativePutInStripeCount];
static std::atomic<uint64_t> g_putin_slot_begin[64], g_putin_slot_end[64];
static std::atomic<uint32_t> g_putin_busy_total {0};
static bool NativeStaleCheckOn() {
	static const bool on = [] {
		const char* v   = std::getenv("KYTY_RAIZ_STALE_CHECK");
		const bool  set = Raiz::On() && !(v != nullptr && v[0] == '0');
		if (Raiz::On()) {
			std::printf("Raiz: stale-read check %s (KYTY_RAIZ_STALE_CHECK%s; raiz11): direct reads of guest memory during a "
			            "put-in of kept GPU bytes counted%s\n",
			            set ? "ON" : "off", set ? "=0 turns it off" : "=0", set ? "" : " - not -");
			std::fflush(stdout);
		}
		return set;
	}();
	return on;
}
static bool NativeStaleFixOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RAIZ_STALE_FIX");
		const bool  set = NativeStaleCheckOn() && !(v != nullptr && v[0] == '0');
		if (Raiz::On()) {
			std::printf("Raiz: stale reads %s (KYTY_RAIZ_STALE_FIX%s; raiz11)\n",
			            set ? "read again / waited for (the put-in finished first)" : "only counted", set ? "=0: only counted" : "=0");
			std::fflush(stdout);
		}
		return set;
	}();
	return on;
}
// The stripes of [at, at + size): first index and count (all of them past 256 MiB).
static void NativePutInStripes(uint64_t at, uint64_t size, uint32_t* first, uint32_t* count) {
	const uint64_t lo = at >> 16u, hi = (at + (size == 0 ? 1 : size) - 1u) >> 16u;
	*first = static_cast<uint32_t>(lo % NativePutInStripeCount);
	*count = hi - lo + 1u >= NativePutInStripeCount ? NativePutInStripeCount : static_cast<uint32_t>(hi - lo + 1u);
}
// Whether a put-in under way now covers a page of [at, at + size) exactly (its slot; a run that found none: not seen).
static bool NativePutInExact(uint64_t at, uint64_t size) {
	for (uint32_t i = 0; i < 64; ++i) {
		const uint64_t b = g_putin_slot_begin[i].load(std::memory_order_seq_cst);
		if (b == 0) continue;
		const uint64_t e = g_putin_slot_end[i].load(std::memory_order_seq_cst);
		if (b < at + size && at < e) return true;
	}
	return false;
}
// A run's put-in under way (begin) and done (end): its bytes are all in.
static void NativePutInMark(uint64_t at, uint64_t size, bool begin) {
	if (begin) Raiz::g_settle_epoch.fetch_add(1, std::memory_order_acq_rel); // (raiz12: KYTY_RAIZ_SETTLE's memo)
	if (!NativeStaleCheckOn()) return;
	uint32_t first = 0, count = 0;
	NativePutInStripes(at, size, &first, &count);
	thread_local int32_t slot = -1;
	if (begin) {
		for (uint32_t i = 0; i < 64 && slot < 0; ++i) {
			uint64_t expected = 0;
			if (g_putin_slot_begin[i].load(std::memory_order_relaxed) == 0 &&
			    g_putin_slot_begin[i].compare_exchange_strong(expected, ~uint64_t {0}, std::memory_order_seq_cst)) {
				g_putin_slot_end[i].store(at + size, std::memory_order_seq_cst);
				g_putin_slot_begin[i].store(at, std::memory_order_seq_cst);
				slot = static_cast<int32_t>(i);
			}
		}
	}
	for (uint32_t i = 0; i < count; ++i) {
		auto& s = g_putin_stripes[(first + i) % NativePutInStripeCount];
		if (begin) {
			s.busy.fetch_add(1, std::memory_order_seq_cst);
			s.gen.fetch_add(1, std::memory_order_seq_cst);
		} else {
			s.gen.fetch_add(1, std::memory_order_seq_cst);
			s.busy.fetch_sub(1, std::memory_order_seq_cst);
		}
	}
	if (begin) {
		g_putin_busy_total.fetch_add(1, std::memory_order_seq_cst);
		Raiz::g_putin_inflight_runs.fetch_add(1, std::memory_order_relaxed);
	} else {
		g_putin_busy_total.fetch_sub(1, std::memory_order_seq_cst);
		if (slot >= 0) g_putin_slot_begin[slot].store(0, std::memory_order_seq_cst);
		slot = -1;
	}
}
// A direct read's ticket: taken before the read (Take), checked after it (Clean: no put-in overlapped it).
struct NativeReadTicket {
	uint32_t first = 0, count = 0;
	uint64_t sum   = 0; // (the stripes' gens added up)
	bool     busy  = false, any = false;
};
static void NativeReadTicketTake(uint64_t at, uint64_t size, NativeReadTicket* t) {
	t->any = NativeStaleCheckOn() && size != 0;
	if (!t->any) return;
	NativePutInStripes(at, size, &t->first, &t->count);
	t->sum  = 0;
	t->busy = false;
	for (uint32_t i = 0; i < t->count; ++i) {
		auto& s = g_putin_stripes[(t->first + i) % NativePutInStripeCount];
		t->busy = t->busy || s.busy.load(std::memory_order_seq_cst) != 0;
		t->sum += s.gen.load(std::memory_order_seq_cst);
	}
}
static bool NativeReadTicketClean(const NativeReadTicket& t) {
	if (!t.any) return true;
	std::atomic_thread_fence(std::memory_order_seq_cst);
	uint64_t sum  = 0;
	bool     busy = t.busy;
	for (uint32_t i = 0; i < t.count; ++i) {
		auto& s = g_putin_stripes[(t.first + i) % NativePutInStripeCount];
		busy    = busy || s.busy.load(std::memory_order_seq_cst) != 0;
		sum += s.gen.load(std::memory_order_seq_cst);
	}
	return !busy && sum == t.sum;
}
// Whether a put-in is under way over [at, at + size) now.
static bool NativePutInBusy(uint64_t at, uint64_t size) {
	if (!NativeStaleCheckOn() || g_putin_busy_total.load(std::memory_order_seq_cst) == 0) return false;
	uint32_t first = 0, count = 0;
	NativePutInStripes(at, size, &first, &count);
	for (uint32_t i = 0; i < count; ++i)
		if (g_putin_stripes[(first + i) % NativePutInStripeCount].busy.load(std::memory_order_seq_cst) != 0) return true;
	return false;
}
// A stale read found: counted; the first 48 printed with the address, its allocator's label (DemonsSoulsMemMap) and the
// reader (`what`: the upload worker's site, the GPU thread's shader hash).
static void NativeNoteStale(Raiz::StaleSite site, uint64_t at, uint64_t size, uint64_t what, bool fixed);
// (The same, `exact` when a put-in still under way covers its pages - not only their 64 KiB -.)
static bool NativeStaleExact(uint64_t at, uint64_t size) { return NativePutInExact(at, size); }

// ---- (raiz11) KYTY_RAIZ_MERGE_FAST (default on with the merge put-in; =0: merge2's NativePendingMergeOpen) -----------
// 20261116-052640 (merge2): the put-in cost 4.56 ms/frame (raiz8's plain copy: 1.53) for 8,244 runs/s, 91,653 pages/s:
// three VirtualProtect a run (read-only, read-write, final) and a compare-and-swap for every 8 bytes that differ - a
// locked instruction each: 512 a page, ~10x a plain copy (measured here: 1,300 pages, CAS 5.0-5.4 ms, memcpy 0.4) -.
// Now, with the same guarantees (no game write lost: one in the open page wins, one before the final protection found):
//  1. read-only first (as before); each page compared IN PLACE with its kept bytes (no copy): a page that already holds
//     them needs nothing more - it stays read-only (the caller's protection follows), no read-write opening, no CAS, no
//     write watch reset -;
//  2. only the pages that differ: copied aside (their old bytes), opened read-write as runs of their own, their differing
//     words put in by CAS - two words at a time (16 bytes, one locked instruction) where both differ -;
//  3. the final protection: when the caller gives the range read-only itself right after (ProtectMappedUnlocked's
//     ranged release, t_native_merge_after), not applied here - one VirtualProtect less a run -: the check runs after the
//     caller's (NativeMergeAfterProtect). Otherwise as before.
static bool NativeMergeFastOn() {
	static const bool on = [] {
		const char* v   = std::getenv("KYTY_RAIZ_MERGE_FAST");
		const bool  set = !(v != nullptr && v[0] == '0');
		if (Raiz::On()) {
			std::printf("Raiz: kept bytes merge %s (KYTY_RAIZ_MERGE_FAST%s; raiz11)\n",
			            set ? "fast: pages holding them already left alone, differing words only, 16-byte CAS, the final check "
			                  "after the caller's protection"
			                : "as merge2",
			            set ? "=0 for merge2's" : "=0");
			std::fflush(stdout);
		}
		return set;
	}();
	return on;
}
// (Set by ProtectMappedUnlocked around NativePendingBeforeProtect for a change to read-only: it protects the range
// itself right after; the checks wait in t_native_merge_checks for NativeMergeAfterProtect.)
static thread_local bool t_native_merge_after = false;
struct NativeMergeCheck {
	uint64_t at, size;      // a run of differing pages, read-write now
	size_t   words;         // its expected words in t_native_merge_expect, from here
	size_t   raced;         // its per-page game-write flags in t_native_merge_raced, from here
};
static thread_local std::vector<NativeMergeCheck> t_native_merge_checks;
static thread_local std::vector<uint64_t>         t_native_merge_expect;
static thread_local std::vector<uint8_t>          t_native_merge_raced;

// One 16-byte compare-and-swap (both words or neither); `expected` gets what memory held on failure.
static bool NativeCas16(uint64_t* dst, uint64_t expected[2], const uint64_t desired[2]) {
#if defined(__clang__) || defined(__GNUC__)
	bool ok;
	__asm__ __volatile__("lock cmpxchg16b %1"
	                     : "=@ccz"(ok), "+m"(*reinterpret_cast<volatile __int128*>(dst)), "+a"(expected[0]), "+d"(expected[1])
	                     : "b"(desired[0]), "c"(desired[1])
	                     : "memory");
	return ok;
#else
	return _InterlockedCompareExchange128(reinterpret_cast<volatile long long*>(dst), static_cast<long long>(desired[1]),
	                                      static_cast<long long>(desired[0]), reinterpret_cast<long long*>(expected)) != 0;
#endif
}

// The checks of a run left for after the caller's protection, or done now (`final_protect` applied here first).
static void NativeMergeCheckRun(const NativeMergeCheck& check) {
	const uint64_t pages = check.size >> 12u;
	for (uint64_t p = 0; p < pages; ++p) {
		const uint64_t page = check.at + (p << 12u);
		if (t_native_merge_raced[check.raced + p] != 0) continue;
		if (std::memcmp(reinterpret_cast<const void*>(page), t_native_merge_expect.data() + check.words + p * 512u, 4096) != 0) {
			t_native_merge_raced[check.raced + p] = 1;
			Raiz::g_merge_raced.fetch_add(1, std::memory_order_relaxed);
			Libs::LibKernel::Memory::RaizWbNotePutInRace(page, 0, true);
		}
	}
	Libs::LibKernel::Memory::RaizWbEmulatorWrote(check.at, check.size);
	for (uint64_t p = 0; p < pages; ++p)
		if (t_native_merge_raced[check.raced + p] != 0)
			Libs::LibKernel::Memory::RaizWbNoteGameWrote(check.at + (p << 12u), "kept GPU bytes put in");
}
// (ProtectMappedUnlocked, right after its VirtualProtect: the runs' final protection is on.)
static void NativeMergeAfterProtect() {
	for (const auto& check: t_native_merge_checks) {
		NativeMergeCheckRun(check);
		Raiz::g_merge_after.fetch_add(1, std::memory_order_relaxed);
	}
	t_native_merge_checks.clear();
	t_native_merge_expect.clear();
	t_native_merge_raced.clear();
}

// (putin-seguro, KYTY_RAIZ_PUTIN_SEGURO, default on; =0: as before) crash-ba2b1f, eboot+0xc4eb9f (log 20261126-084435):
// the CAS above keeps a game word only when the game's value differs from the page's old one. A game write of the same
// value the page held (a field set to 0 / null over an old 0: an object built in the open page) is not seen, and the
// GPU's word goes over it (an ABA): a CPU object with GPU bytes in its fields (0x3c00000000000000, fp16 1.0, at +0x20 /
// +0x30) and a virtual call through them crashed. A page where any game write is found was written by the game while
// the GPU's bytes were kept: it is the game's page (as KYTY_RAIZ_WB_GUARD leaves such pages alone at a write-back). Every
// word this put-in wrote there goes back to the page's old value by CAS (a game write since keeps the game's); the page
// is then queued as CPU-written (RaizWbNoteGameWrote), so the GPU's copy gets the game's bytes.
static bool NativePutInSeguroOn() {
	static const bool on = [] {
		const char* v   = std::getenv("KYTY_RAIZ_PUTIN_SEGURO");
		const bool  set = !(v != nullptr && v[0] == '0');
		if (Raiz::On()) {
			std::printf("[putin-seguro] KYTY_RAIZ_PUTIN_SEGURO=%d (crash-ba2b1f): %s\n", set ? 1 : 0,
			            set ? "a page the game wrote during a put-in of kept GPU bytes stays the game's (GPU bytes thrown away)"
			                : "off: word by word as before");
			std::fflush(stdout);
		}
		return set;
	}();
	return on;
}
// The words of a page this put-in wrote (`ours`) back to `before` (the page's old bytes); `words` (what each word should
// hold, for the checks after) follows. Returns the words given back.
static uint64_t NativePutInSeguroRevert(uint64_t page, uint64_t* dst, uint64_t* words, const uint64_t* before,
                                        const uint8_t* ours) {
	uint64_t back = 0, game = 0;
	for (uint64_t w = 0; w < 512u; ++w) {
		if (ours[w] == 0) continue;
		uint64_t e = words[w];
		if (std::atomic_ref<uint64_t>(dst[w]).compare_exchange_strong(e, before[w], std::memory_order_relaxed)) {
			words[w] = before[w];
			++back;
		} else {
			words[w] = e; // (the game wrote it since: the game's)
			++game;
		}
	}
	static std::atomic<uint32_t> said {0};
	if (said.fetch_add(1, std::memory_order_relaxed) < 64) {
		std::printf("[putin-seguro] pagina 0x%llx: el juego la escribio, bytes de GPU descartados (%llu palabras; %llu ya "
		            "reescritas por el juego)\n",
		            static_cast<unsigned long long>(page), static_cast<unsigned long long>(back),
		            static_cast<unsigned long long>(game));
		std::fflush(stdout);
	}
	return back;
}

static void NativePendingMergeOpenFast(NativeTaken& taken, DWORD final_protect, Raiz::LazyApply how) {
	constexpr uint64_t Page       = 0x1000;
	const bool         final_rw   = final_protect == PAGE_READWRITE || final_protect == PAGE_EXECUTE_READWRITE;
	const bool         final_read = final_protect == PAGE_READONLY || final_protect == PAGE_EXECUTE_READ;
	const bool         after      = t_native_merge_after && final_protect == PAGE_READONLY;
	if (!after) {
		t_native_merge_checks.clear();
		t_native_merge_expect.clear();
		t_native_merge_raced.clear();
	}
	thread_local std::vector<uint8_t> differs;
	uint64_t                          applied = 0;
	for (size_t i = 0; i < taken.size();) {
		size_t j = i + 1;
		while (j < taken.size() && taken[j].first == taken[j - 1].first + Page) ++j;
		const uint64_t run_at   = taken[i].first;
		const uint64_t run_size = (j - i) * Page;
		DWORD          old = 0, ignored = 0;
		NativePutInMark(run_at, run_size, true);
		const uint64_t exposed_ns = Raiz::On() ? Raiz::NowNs() : 0; // (raiz11: the old bytes readable from here)
		Raiz::g_merge_runs.fetch_add(1, std::memory_order_relaxed);
		Raiz::g_merge_vp.fetch_add(1, std::memory_order_relaxed);
		if (VirtualProtect(reinterpret_cast<void*>(run_at), run_size, PAGE_READONLY, &old) == 0) {
			NativePutInMark(run_at, run_size, false);
			// (Not opened: back to the store, as they were - no write could land on them -.)
			std::lock_guard lock(g_pending_mutex);
			for (size_t k = i; k < j; ++k) {
				auto& slot = g_pending[taken[k].first];
				if (slot) continue;
				slot = std::move(taken[k].second);
				g_pending_count.fetch_add(1, std::memory_order_relaxed);
				g_pending_buckets[NativePendingBucket(taken[k].first)].fetch_add(1, std::memory_order_relaxed);
			}
			i = j;
			continue;
		}
		Raiz::g_counters.window_protects.fetch_add(1, std::memory_order_relaxed);
		// 1. Each page against its kept bytes, in place (read-only: no game write meanwhile).
		differs.assign(j - i, 0);
		uint64_t n_diff = 0;
		for (size_t k = i; k < j; ++k) {
			const auto&     kept = *taken[k].second;
			const uint64_t* cur  = reinterpret_cast<const uint64_t*>(taken[k].first);
			bool            diff = false;
			for (uint64_t w = 0; w < Page / 8u && !diff; ++w) {
				const uint64_t bits = (kept.mask[w >> 3u] >> ((w & 7u) * 8u)) & 0xffu;
				if (bits == 0) continue;
				const uint64_t mask = NativeByteMask(bits);
				uint64_t       value;
				std::memcpy(&value, kept.bytes + w * 8u, 8);
				diff = ((cur[w] ^ value) & mask) != 0;
			}
			differs[k - i] = diff ? 1 : 0;
			n_diff += diff ? 1u : 0u;
		}
		Raiz::g_merge_pages_same.fetch_add((j - i) - n_diff, std::memory_order_relaxed);
		Raiz::g_merge_pages_diff.fetch_add(n_diff, std::memory_order_relaxed);
		// 2. The runs of differing pages: their old bytes aside, read-write, the differing words by CAS.
		for (size_t a = i; a < j;) {
			if (differs[a - i] == 0) {
				++a;
				continue;
			}
			size_t b = a + 1;
			while (b < j && differs[b - i] != 0) ++b;
			const uint64_t sub_at = taken[a].first, sub_size = (b - a) * Page;
			const size_t   words_at = t_native_merge_expect.size(), raced_at = t_native_merge_raced.size();
			Raiz::SettleNotePutIn(sub_at, sub_size); // (raiz12: over a buffer the game's 0xa1f020 reads, while it reads?)
			t_native_merge_expect.resize(words_at + sub_size / 8u);
			t_native_merge_raced.resize(raced_at + (b - a), 0);
			std::memcpy(t_native_merge_expect.data() + words_at, reinterpret_cast<const void*>(sub_at), sub_size);
			Raiz::g_merge_vp.fetch_add(1, std::memory_order_relaxed);
			if (VirtualProtect(reinterpret_cast<void*>(sub_at), sub_size, PAGE_READWRITE, &ignored) == 0) {
				// (Read-only still: the old bytes are what the page holds. The kept ones back to the store - as when a
				// run cannot be opened -: a later opening puts them in.)
				t_native_merge_expect.resize(words_at);
				t_native_merge_raced.resize(raced_at);
				std::lock_guard lock(g_pending_mutex);
				for (size_t k = a; k < b; ++k) {
					auto& slot = g_pending[taken[k].first];
					if (slot) continue;
					slot = std::move(taken[k].second);
					g_pending_count.fetch_add(1, std::memory_order_relaxed);
					g_pending_buckets[NativePendingBucket(taken[k].first)].fetch_add(1, std::memory_order_relaxed);
				}
				a = b;
				continue;
			}
			Raiz::g_counters.window_protects.fetch_add(1, std::memory_order_relaxed);
			uint64_t n_words = 0, n_cas16 = 0;
			for (size_t k = a; k < b; ++k) {
				const auto& kept  = *taken[k].second;
				auto*       dst   = reinterpret_cast<uint64_t*>(taken[k].first);
				uint64_t*   words = t_native_merge_expect.data() + words_at + (k - a) * (Page / 8u);
				uint64_t    game  = 0;
				thread_local uint64_t before[Page / 8u]; // (putin-seguro: the page's old bytes, and the words put in)
				thread_local uint8_t  ours[Page / 8u];
				std::memcpy(before, words, Page);
				std::memset(ours, 0, sizeof(ours));
				for (uint64_t w = 0; w < Page / 8u; w += 2) {
					uint64_t desired[2] = {words[w], words[w + 1]};
					bool     change[2]  = {false, false};
					for (uint64_t h = 0; h < 2; ++h) {
						const uint64_t bits = (kept.mask[(w + h) >> 3u] >> (((w + h) & 7u) * 8u)) & 0xffu;
						if (bits == 0) continue;
						const uint64_t mask = NativeByteMask(bits);
						uint64_t       value;
						std::memcpy(&value, kept.bytes + (w + h) * 8u, 8);
						desired[h] = (words[w + h] & ~mask) | (value & mask);
						change[h]  = desired[h] != words[w + h];
					}
					if (!change[0] && !change[1]) continue;
					if (change[0] && change[1]) {
						uint64_t expected[2] = {words[w], words[w + 1]};
						++n_cas16;
						n_words += 2;
						if (NativeCas16(dst + w, expected, desired)) {
							words[w]     = desired[0];
							words[w + 1] = desired[1];
							ours[w] = ours[w + 1] = 1;
							continue;
						}
						// (The game wrote one of the two, or both: each word on its own against what it held.)
						for (uint64_t h = 0; h < 2; ++h) {
							if (expected[h] != words[w + h]) {
								words[w + h] = expected[h]; // (the game's value, kept)
								++game;
								continue;
							}
							uint64_t e = words[w + h];
							if (std::atomic_ref<uint64_t>(dst[w + h]).compare_exchange_strong(e, desired[h], std::memory_order_relaxed)) {
								words[w + h] = desired[h];
								ours[w + h]  = 1;
							} else {
								words[w + h] = e;
								++game;
							}
						}
						continue;
					}
					const uint64_t h = change[0] ? 0 : 1;
					uint64_t       e = words[w + h];
					++n_words;
					if (std::atomic_ref<uint64_t>(dst[w + h]).compare_exchange_strong(e, desired[h], std::memory_order_relaxed)) {
						words[w + h] = desired[h];
						ours[w + h]  = 1;
					} else {
						words[w + h] = e; // (the game's value, kept)
						++game;
					}
				}
				if (game != 0) {
					t_native_merge_raced[raced_at + (k - a)] = 1;
					Raiz::g_merge_raced.fetch_add(1, std::memory_order_relaxed);
					Libs::LibKernel::Memory::RaizWbNotePutInRace(taken[k].first, game, false);
					if (NativePutInSeguroOn()) (void)NativePutInSeguroRevert(taken[k].first, dst, words, before, ours);
				}
			}
			Raiz::g_merge_words.fetch_add(n_words, std::memory_order_relaxed);
			Raiz::g_merge_cas16.fetch_add(n_cas16, std::memory_order_relaxed);
			if (exposed_ns != 0 && Raiz::t_pm_role != Raiz::RoleFault) {
				const uint64_t open_us = (Raiz::NowNs() - exposed_ns) / 1000u;
				for (size_t k = a; k < b; ++k) Raiz::NotePutInExposure(taken[k].first, exposed_ns, open_us);
			}
			const NativeMergeCheck check {sub_at, sub_size, words_at, raced_at};
			// 3. The final protection: the caller's, right after (the check then), or here.
			if (after) {
				t_native_merge_checks.push_back(check);
			} else {
				if (!final_rw) {
					(void)VirtualProtect(reinterpret_cast<void*>(sub_at), sub_size, final_protect, &ignored);
					Raiz::g_merge_vp.fetch_add(1, std::memory_order_relaxed);
					Raiz::g_counters.window_protects.fetch_add(1, std::memory_order_relaxed);
				}
				if (!final_rw && final_read) {
					NativeMergeCheckRun(check);
				} else {
					// (Left read-write, or not readable: no check, as before.)
					Libs::LibKernel::Memory::RaizWbEmulatorWrote(sub_at, sub_size);
					for (size_t k = a; k < b; ++k)
						if (t_native_merge_raced[raced_at + (k - a)] != 0)
							Libs::LibKernel::Memory::RaizWbNoteGameWrote(taken[k].first, "kept GPU bytes put in");
				}
			}
			a = b;
		}
		NativePutInMark(run_at, run_size, false); // (every byte in, or the page's own)
		{
			std::lock_guard lock(g_pending_mutex); // (raiz5: the pages back to the pool, up to 4,096 of them)
			for (size_t k = i; k < j; ++k) {
				if (!taken[k].second) continue; // (back in the store)
				++applied;
				if (g_pending_free.size() < 4096) g_pending_free.push_back(std::move(taken[k].second));
			}
		}
		i = j;
	}
	if (!after) {
		t_native_merge_checks.clear();
		t_native_merge_expect.clear();
		t_native_merge_raced.clear();
	}
	if (applied != 0) Raiz::g_lazy_applied[how].fetch_add(applied, std::memory_order_relaxed);
	taken.clear();
}

// `taken` (sorted, as NativePendingTake leaves it) put in; the runs end with `final_protect` (PAGE_READWRITE /
// PAGE_EXECUTE_READWRITE: left read-write, no check after). The caller holds the gate, as for NativePendingCopy.
static void NativePendingMergeOpenMerge2(NativeTaken& taken, DWORD final_protect, Raiz::LazyApply how) {
	constexpr uint64_t                 Page = 0x1000;
	thread_local std::vector<uint64_t> snap; // (the run's bytes, then what each word should hold)
	const bool final_rw   = final_protect == PAGE_READWRITE || final_protect == PAGE_EXECUTE_READWRITE;
	const bool final_read = final_protect == PAGE_READONLY || final_protect == PAGE_EXECUTE_READ;
	uint64_t   applied    = 0;
	for (size_t i = 0; i < taken.size();) {
		size_t j = i + 1;
		while (j < taken.size() && taken[j].first == taken[j - 1].first + Page) ++j;
		const uint64_t run_at   = taken[i].first;
		const uint64_t run_size = (j - i) * Page;
		auto*          run      = reinterpret_cast<void*>(run_at);
		DWORD          old = 0, ignored = 0;
		bool           ok = VirtualProtect(run, run_size, PAGE_READONLY, &old) != 0;
		if (ok) {
			snap.resize(run_size / 8u);
			std::memcpy(snap.data(), run, run_size);
			ok = VirtualProtect(run, run_size, PAGE_READWRITE, &ignored) != 0;
			if (!ok) (void)VirtualProtect(run, run_size, old, &ignored);
		}
		if (!ok) {
			// (Not opened: back to the store, as they were - no write could land on them -.)
			std::lock_guard lock(g_pending_mutex);
			for (size_t k = i; k < j; ++k) {
				auto& slot = g_pending[taken[k].first];
				if (slot) continue;
				slot = std::move(taken[k].second);
				g_pending_count.fetch_add(1, std::memory_order_relaxed);
				g_pending_buckets[NativePendingBucket(taken[k].first)].fetch_add(1, std::memory_order_relaxed);
			}
			i = j;
			continue;
		}
		Raiz::g_counters.window_protects.fetch_add(2, std::memory_order_relaxed);
		thread_local std::vector<uint8_t> raced;
		raced.assign(j - i, 0);
		for (size_t k = i; k < j; ++k) {
			const auto&    kept  = *taken[k].second;
			auto*          dst   = reinterpret_cast<uint64_t*>(taken[k].first);
			uint64_t*      words = snap.data() + (k - i) * (Page / 8u);
			uint64_t       game  = 0;
			thread_local uint64_t before[Page / 8u]; // (putin-seguro: the page's old bytes, and the words put in)
			thread_local uint8_t  ours[Page / 8u];
			std::memcpy(before, words, Page);
			std::memset(ours, 0, sizeof(ours));
			for (uint64_t w = 0; w < Page / 8u; ++w) {
				const uint64_t bits = (kept.mask[w >> 3u] >> ((w & 7u) * 8u)) & 0xffu;
				if (bits == 0) continue;
				const uint64_t mask = NativeByteMask(bits);
				uint64_t       value;
				std::memcpy(&value, kept.bytes + w * 8u, 8);
				uint64_t       expected = words[w];
				const uint64_t desired  = (expected & ~mask) | (value & mask);
				if (desired == expected) continue;
				if (std::atomic_ref<uint64_t>(dst[w]).compare_exchange_strong(expected, desired, std::memory_order_relaxed)) {
					words[w] = desired;
					ours[w]  = 1;
				} else {
					words[w] = expected; // (the game's value, kept)
					++game;
				}
			}
			if (game != 0) {
				raced[k - i] = 1;
				Libs::LibKernel::Memory::RaizWbNotePutInRace(taken[k].first, game, false);
				if (NativePutInSeguroOn()) (void)NativePutInSeguroRevert(taken[k].first, dst, words, before, ours);
			}
		}
		if (!final_rw) {
			(void)VirtualProtect(run, run_size, final_protect, &ignored);
			Raiz::g_counters.window_protects.fetch_add(1, std::memory_order_relaxed);
			if (final_read) {
				for (size_t k = i; k < j; ++k) {
					if (raced[k - i] != 0) continue;
					if (std::memcmp(reinterpret_cast<const void*>(taken[k].first), snap.data() + (k - i) * (Page / 8u), Page) != 0) {
						raced[k - i] = 1;
						Libs::LibKernel::Memory::RaizWbNotePutInRace(taken[k].first, 0, true);
					}
				}
			}
		}
		// (The copy's own writes are not the game's: KYTY_RAIZ_WB_GUARD's write watch forgets them; the game's found
		// above are marked instead.)
		Libs::LibKernel::Memory::RaizWbEmulatorWrote(run_at, run_size);
		for (size_t k = i; k < j; ++k)
			if (raced[k - i] != 0) Libs::LibKernel::Memory::RaizWbNoteGameWrote(taken[k].first, "kept GPU bytes put in");
		{
			std::lock_guard lock(g_pending_mutex); // (raiz5: the pages back to the pool, up to 4,096 of them)
			for (size_t k = i; k < j; ++k)
				if (g_pending_free.size() < 4096) g_pending_free.push_back(std::move(taken[k].second));
		}
		applied += j - i;
		i = j;
	}
	if (applied != 0) Raiz::g_lazy_applied[how].fetch_add(applied, std::memory_order_relaxed);
	taken.clear();
}

// (raiz11) KYTY_RAIZ_MERGE_FAST: NativePendingMergeOpenFast; =0: merge2's.
static void NativePendingMergeOpen(NativeTaken& taken, DWORD final_protect, Raiz::LazyApply how) {
	if (taken.empty()) return;
	if (NativeMergeFastOn()) {
		NativePendingMergeOpenFast(taken, final_protect, how);
		return;
	}
	// (merge2's: the put-in under way marked for the stale-read check all the same, the whole call.)
	const uint64_t at = taken.front().first, size = taken.back().first + 0x1000u - taken.front().first;
	NativePutInMark(at, size, true);
	Raiz::SettleNotePutIn(at, size); // (raiz12: every page, merge2 does not tell the differing ones)
	NativePendingMergeOpenMerge2(taken, final_protect, how);
	NativePutInMark(at, size, false);
}

// ---- (raiz12) KYTY_RAIZ_SETTLE (local/raiz.h): a game reader of GPU results waits for its buffer's put-ins -------------
// Memory::RaizSettleRange's work (here, so the stub test runs it): `readback()` reads back what the GPU still owns of the
// range (true: done), `apply()` applies its deferred releases (their put-ins, on this thread, under the page manager's
// locks). Mode 1: both, then until no put-in is under way over the range and no kept bytes are left over it (another
// thread's readback between its copy and its release: its put-in would begin while the game reads), applying again the
// deferred releases that appear, at most 4 ms. Mode 2: only the checks at entry. Counted (Raiz::g_settle_*).
// The memo: the same range on this thread with no keep and no put-in begun anywhere since (Raiz::g_settle_epoch) and
// none under way: nothing can have changed for it.
// (parpadeo6) `mode_override` >= 0: that mode instead of KYTY_RAIZ_SETTLE's (Memory::RaizSettleRange's `mode`).
enum : uint32_t { NativeSettleBusy = 1u, NativeSettlePending = 2u, NativeSettleReadback = 4u, NativeSettleWaited = 8u, NativeSettleLeft = 16u };
template <typename Readback, typename Apply>
static uint32_t NativeSettle(uint64_t vaddr, uint64_t size, uint32_t site, Readback&& readback, Apply&& apply, int mode_override = -1) {
	const int mode = mode_override >= 0 ? (Raiz::On() ? mode_override : 0) : Raiz::SettleMode();
	if (mode == 0 || size == 0 || site >= Raiz::SettleSiteCount) return 0;
	const auto start = std::chrono::steady_clock::now();
	Raiz::g_settle_calls[site].fetch_add(1, std::memory_order_relaxed);
	struct Memo {
		uint64_t vaddr = 0, size = 0, epoch = UINT64_MAX;
	};
	thread_local Memo memo[Raiz::SettleSiteCount];
	const uint64_t    epoch = Raiz::g_settle_epoch.load(std::memory_order_acquire);
	if (mode == 1 && memo[site].vaddr == vaddr && memo[site].size == size && memo[site].epoch == epoch && !NativePutInBusy(vaddr, size)) {
		Raiz::g_settle_memo[site].fetch_add(1, std::memory_order_relaxed);
		return 0;
	}
	uint32_t flags = 0;
	if (NativePutInBusy(vaddr, size)) flags |= NativeSettleBusy;
	if (NativePendingIn(vaddr, size)) flags |= NativeSettlePending;
	if ((flags & NativeSettleBusy) != 0) Raiz::g_settle_busy[site].fetch_add(1, std::memory_order_relaxed);
	if ((flags & NativeSettlePending) != 0) Raiz::g_settle_pending[site].fetch_add(1, std::memory_order_relaxed);
	if (mode == 1) {
		if (readback()) {
			flags |= NativeSettleReadback;
			Raiz::g_settle_readback[site].fetch_add(1, std::memory_order_relaxed);
		}
		std::chrono::steady_clock::time_point wait {};
		for (;;) {
			apply();
			if (!NativePutInBusy(vaddr, size) && !NativePendingIn(vaddr, size)) break;
			const auto now = std::chrono::steady_clock::now();
			if ((flags & NativeSettleWaited) == 0) {
				flags |= NativeSettleWaited;
				wait = now;
			} else if (now - wait >= std::chrono::milliseconds(4)) {
				flags |= NativeSettleLeft;
				Raiz::g_settle_left[site].fetch_add(1, std::memory_order_relaxed);
				break;
			}
			std::this_thread::yield();
		}
		if ((flags & NativeSettleWaited) != 0) {
			Raiz::g_settle_waited[site].fetch_add(1, std::memory_order_relaxed);
			Raiz::g_settle_wait_us[site].fetch_add(
			    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wait).count()),
			    std::memory_order_relaxed);
		}
		// (The epoch read before the work: a keep or a put-in begun meanwhile makes the next call settle again.)
		memo[site] = {vaddr, size, epoch};
	}
	Raiz::g_settle_us[site].fetch_add(
	    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()),
	    std::memory_order_relaxed);
	return flags;
}

// (raiz11) A stale read: counted; the first 48 printed - the address, the allocator label of the game's memory map, the
// reader -, for the flashes: which of the game's buffers the GPU got old bytes of.
static void NativeNoteStale(Raiz::StaleSite site, uint64_t at, uint64_t size, uint64_t what, bool fixed) {
	const bool exact = NativeStaleExact(at, size);
	Raiz::g_stale_n[site].fetch_add(1, std::memory_order_relaxed);
	if (exact) Raiz::g_stale_exact[site].fetch_add(1, std::memory_order_relaxed);
	if (fixed) Raiz::g_stale_fixed[site].fetch_add(1, std::memory_order_relaxed);
	static std::atomic<uint32_t> said {0};
	if (!exact || said.fetch_add(1, std::memory_order_relaxed) >= 48) return; // (only the certain ones printed)
	char label[64] = {};
	if (!::Loader::DemonsSoulsMemMap::LabelAt(at, label, sizeof(label)) || label[0] == 0) std::snprintf(label, sizeof(label), "?");
	std::printf("[raiz11] STALE READ (%s): 0x%llx+0x%llx during a put-in of kept GPU bytes, label %s, reader 0x%llx%s, "
	            "kept now %llu pages\n",
	            Raiz::StaleSiteNames[site], static_cast<unsigned long long>(at), static_cast<unsigned long long>(size), label,
	            static_cast<unsigned long long>(what), fixed ? " - read again after it -" : " - NOT fixed (KYTY_RAIZ_STALE_FIX=0) -",
	            static_cast<unsigned long long>(g_pending_count.load(std::memory_order_relaxed)));
	std::fflush(stdout);
}

// The kept bytes of [vaddr, vaddr + size) dropped: a map or an unmap there.
static void NativePendingDrop(uint64_t vaddr, uint64_t size) {
	NativeShadowDrop(vaddr, size); // (raiz5: a map or unmap forgets the shadows too)
	if (g_pending_count.load(std::memory_order_relaxed) == 0 || size == 0) return;
	std::lock_guard lock(g_pending_mutex);
	uint64_t        dropped = 0;
	const uint64_t  first   = vaddr & ~uint64_t {0xfff};
	if (((vaddr + size - first + 0xfff) >> 12u) < g_pending.size()) {
		// (A range smaller than the store: its pages looked up.)
		for (uint64_t page = first; page < vaddr + size; page += 0x1000) {
			auto it = g_pending.find(page);
			if (it == g_pending.end()) continue;
			NativePendingWaitFilled(*it->second); // (raiz7)
			g_pending_buckets[NativePendingBucket(page)].fetch_sub(1, std::memory_order_relaxed);
			g_pending.erase(it);
			++dropped;
		}
		g_pending_count.fetch_sub(dropped, std::memory_order_relaxed);
		Raiz::g_lazy_dropped.fetch_add(dropped, std::memory_order_relaxed);
		return;
	}
	for (auto it = g_pending.begin(); it != g_pending.end();) {
		if (it->first >= (vaddr & ~uint64_t {0xfff}) && it->first < vaddr + size) {
			NativePendingWaitFilled(*it->second); // (raiz7)
			g_pending_buckets[NativePendingBucket(it->first)].fetch_sub(1, std::memory_order_relaxed);
			it = g_pending.erase(it);
			++dropped;
		} else {
			++it;
		}
	}
	g_pending_count.fetch_sub(dropped, std::memory_order_relaxed);
	Raiz::g_lazy_dropped.fetch_add(dropped, std::memory_order_relaxed);
}

// (The page-run walk of a protection change, KYTY_RAIZ_LAZY: the kept bytes of [vaddr, vaddr + size) in, the runs that
// are no-access now made read-write for it, before the change gives them their new protection. Only when the new one
// is readable - a no-access one keeps them kept -.) The caller holds the gate (shared or alone).
static bool NativePendingBeforeProtect(uint64_t vaddr, uint64_t size, bool now_readable, bool now_read_write = false) {
	// (raiz4) Returns true when every page of the range is read-write now (opened for the copy or already): a change to
	// read-write then has nothing left to do (the caller skips its VirtualProtect: raiz3 made two, read-write for the
	// copy and read-write again). Timed for the "[raiz]" line (GPU thread apart).
	if (!now_readable || !NativePendingIn(vaddr, size)) return false;
	const auto     start     = std::chrono::steady_clock::now();
	// (raiz5) A change to read-write: the whole range read-write at once - it is what is asked -, the kept bytes in, and
	// nothing else: one VirtualProtect, no VirtualQuery (raiz4: a VirtualQuery and a VirtualProtect a run, 1.7-2.9 ms a
	// frame on the GPU thread at Boletaria, 76,000 pages/s).
	// (raiz6: the kept pages taken out first - and the stale ones dropped -; the copy right after the opening.)
	thread_local NativeTaken taken;
	if (now_read_write && NativeMergePutIn()) {
		// (crash-ba2b1f) The kept pages first, read-write with their bytes merged in (NativePendingMergeOpen), then the
		// whole range read-write. Not opened (the range's VirtualProtect refused): the slow path below, as before.
		NativePendingTake(vaddr, size, &taken);
		NativePendingMergeOpen(taken, PAGE_READWRITE, Raiz::LazyAtRelease);
		DWORD old = 0;
		if (VirtualProtect(reinterpret_cast<void*>(vaddr), size, PAGE_READWRITE, &old) != 0) {
			Raiz::g_counters.release_fast.fetch_add(1, std::memory_order_relaxed);
			const auto us = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
			(Graphics::GuestGpu::IsGpuThread() ? Raiz::g_lazy_apply_us_gpu : Raiz::g_lazy_apply_us_other).fetch_add(us, std::memory_order_relaxed);
			return true;
		}
	} else if (now_read_write) {
		NativePendingTake(vaddr, size, &taken);
		DWORD old = 0;
		if (VirtualProtect(reinterpret_cast<void*>(vaddr), size, PAGE_READWRITE, &old) != 0) {
			NativePendingCopy(taken, Raiz::LazyAtRelease);
			Raiz::g_counters.release_fast.fetch_add(1, std::memory_order_relaxed);
			const auto us = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
			(Graphics::GuestGpu::IsGpuThread() ? Raiz::g_lazy_apply_us_gpu : Raiz::g_lazy_apply_us_other).fetch_add(us, std::memory_order_relaxed);
			return true;
		}
		// (Not opened: the pages go back to the store, as they were - no write could land on them -.)
		for (auto& [page, kept]: taken) {
			std::lock_guard lock(g_pending_mutex);
			auto&           slot = g_pending[page];
			if (!slot) {
				slot = std::move(kept);
				g_pending_count.fetch_add(1, std::memory_order_relaxed);
				g_pending_buckets[NativePendingBucket(page)].fetch_add(1, std::memory_order_relaxed);
			}
		}
		taken.clear();
	}
	// (raiz8) A change to another readable protection (read-only: a write watcher holds the pages; 5,960/s in
	// 20261116-031219, every one of raiz7's releases with kept bytes took the slow path below): only the kept pages are
	// opened read-write, a run of adjacent ones per VirtualProtect, no VirtualQuery; their bytes in; the caller then gives
	// the whole range its protection (its own VirtualProtect, as before). The kept pages are no-access ones
	// (NativePendingStore keeps bytes only for those, the gate held alone; a change that makes one readable takes them
	// out first), so this opens a subset of what the slow path opened (it opened whole VirtualQuery regions holding a
	// kept page). KYTY_RAIZ_RANGED_RELEASE=0: the slow path, as raiz7.
	static const bool ranged_release = [] {
		const char* v = std::getenv("KYTY_RAIZ_RANGED_RELEASE");
		return !(v != nullptr && v[0] == '0');
	}();
	if (!now_read_write && ranged_release && NativeMergePutIn()) {
		// (crash-ba2b1f) The kept pages read-only at the end (NativePendingMergeOpen checks them there); the caller's
		// VirtualProtect then gives the whole range its protection, as before.
		NativePendingTake(vaddr, size, &taken);
		NativePendingMergeOpen(taken, PAGE_READONLY, Raiz::LazyAtRelease);
		Raiz::g_counters.release_ranged.fetch_add(1, std::memory_order_relaxed);
		const auto us = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
		(Graphics::GuestGpu::IsGpuThread() ? Raiz::g_lazy_apply_us_gpu : Raiz::g_lazy_apply_us_other).fetch_add(us, std::memory_order_relaxed);
		return false;
	}
	if (!now_read_write && ranged_release) {
		NativePendingTake(vaddr, size, &taken);
		thread_local NativeTaken opened;
		opened.clear();
		for (size_t i = 0; i < taken.size();) {
			size_t j = i + 1;
			while (j < taken.size() && taken[j].first == taken[j - 1].first + 0x1000u) ++j;
			const uint64_t run_at   = taken[i].first;
			const uint64_t run_size = (j - i) * 0x1000u;
			DWORD          old      = 0;
			if (VirtualProtect(reinterpret_cast<void*>(run_at), run_size, PAGE_READWRITE, &old) != 0) {
				for (size_t k = i; k < j; ++k) opened.emplace_back(taken[k].first, std::move(taken[k].second));
				Raiz::g_counters.window_protects.fetch_add(1, std::memory_order_relaxed);
			} else {
				// (Not opened: back to the store, as they were - no write could land on them -.)
				std::lock_guard lock(g_pending_mutex);
				for (size_t k = i; k < j; ++k) {
					auto& slot = g_pending[taken[k].first];
					if (slot) continue;
					slot = std::move(taken[k].second);
					g_pending_count.fetch_add(1, std::memory_order_relaxed);
					g_pending_buckets[NativePendingBucket(taken[k].first)].fetch_add(1, std::memory_order_relaxed);
				}
			}
			i = j;
		}
		taken.clear();
		NativePendingCopy(opened, Raiz::LazyAtRelease);
		Raiz::g_counters.release_ranged.fetch_add(1, std::memory_order_relaxed);
		const auto us = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
		(Graphics::GuestGpu::IsGpuThread() ? Raiz::g_lazy_apply_us_gpu : Raiz::g_lazy_apply_us_other).fetch_add(us, std::memory_order_relaxed);
		return false; // (the caller's VirtualProtect gives the range its read-only protection)
	}
	Raiz::g_counters.release_slow.fetch_add(1, std::memory_order_relaxed);
	const uint64_t end       = vaddr + size;
	bool           all_rw    = true;
	for (uint64_t at = vaddr; at < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) {
			all_rw = false;
			break;
		}
		const uint64_t stop = std::min(end, reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize));
		const DWORD    prot     = info.Protect & 0xffu;
		const bool     writable = info.State == MEM_COMMIT && prot == PAGE_READWRITE;
		bool           now_rw   = writable;
		if (info.State == MEM_COMMIT && NativePendingIn(at, stop - at)) {
			NativePendingTake(at, stop - at, &taken); // (raiz6: before the opening)
			DWORD old = 0;
			if (NativeMergePutIn()) {
				// (crash-ba2b1f) The kept pages merged in first; to read-write, then the region read-write. To another
				// readable protection: the kept pages end read-only (checked there) and the caller protects the range.
				if (now_read_write) {
					NativePendingMergeOpen(taken, PAGE_READWRITE, Raiz::LazyAtRelease);
					now_rw = writable || VirtualProtect(reinterpret_cast<void*>(at), stop - at, PAGE_READWRITE, &old) != 0;
					Raiz::g_counters.window_protects.fetch_add(writable ? 0 : 1, std::memory_order_relaxed);
				} else {
					NativePendingMergeOpen(taken, PAGE_READONLY, Raiz::LazyAtRelease);
					now_rw = false;
				}
			} else if (writable || VirtualProtect(reinterpret_cast<void*>(at), stop - at, PAGE_READWRITE, &old) != 0) {
				NativePendingCopy(taken, Raiz::LazyAtRelease);
				Raiz::g_counters.window_protects.fetch_add(writable ? 0 : 1, std::memory_order_relaxed);
				now_rw = true;
			}
		}
		all_rw = all_rw && now_rw;
		at     = std::max(stop, at + 0x1000u);
	}
	const auto us = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
	(Graphics::GuestGpu::IsGpuThread() ? Raiz::g_lazy_apply_us_gpu : Raiz::g_lazy_apply_us_other).fetch_add(us, std::memory_order_relaxed);
	return all_rw;
}

// (raiz2) A read that may need no window: each page of [vaddr, vaddr + bytes) read one byte by Common::GuardedCopy (a
// fault there is resumed, no handler runs). True when every one could be read: no page is no-access, the caller reads
// without the gate, VirtualQuery or VirtualProtect (raiz1 did all three whenever the 64 KiB read hint was set). A page
// made no-access meanwhile is the race the hint check had already (its fault takes the guest fault path, as before).
[[maybe_unused]] static bool NativeAllReadable(uint64_t vaddr, uint64_t bytes) {
	constexpr uint64_t Page = 0x1000;
	uint8_t            sink = 0;
	for (uint64_t at = vaddr & ~(Page - 1u), end = vaddr + bytes; at < end; at += Page) {
		const uint64_t probe = std::max(at, vaddr);
		if (Common::GuardedCopy(&sink, reinterpret_cast<const void*>(probe), 1) != 0) return false;
	}
	return true;
}

// (nativemem1, KYTY_WRITE_WATCH) `access()` reads or writes host bytes [host, host + bytes) - guest memory itself with
// native memory. Pages no tracker protects: done at once (one check of the read hints). Otherwise, with the gate held
// alone (NativeWindowLock), the runs of committed pages that are not writable are made writable, `access()` runs, and
// each run gets its own protection back. Without native memory or KYTY_WRITE_WATCH: `access()` alone.
// (ww2) `writes`: `access()` writes the bytes. KYTY_WRITE_WATCH_MODE=protect: a write also opens the pages a write
// watcher made read-only (as the backing view let it write them, with no fault and the watcher kept), and a read
// leaves read-only pages alone.
// (raiz1, KYTY_RAIZ) Without an alias view, a page opened for the copy is open to the game's threads too:
//  - a read opens no-access pages READ-ONLY only: a game write meanwhile still faults (and waits at the gate for the
//    window to close: its protection change), none is lost; a game read meanwhile sees the bytes it would see after
//    its own read fault's readback, less what is not read back yet (the residual risk, NOTAS.md);
//  - a write opens them read-write: a game write that lands meanwhile in pages opened from READ-ONLY (a write
//    watcher's: their CPU and GPU copies equal) is not seen by a fault, so those pages are queued as CPU-written for the
//    GPU thread (Raiz::QueueResync; KYTY_RAIZ_WINDOW_RESYNC=0 off): invalidated before its next submission, an upload
//    more instead of a lost write. Pages opened from no-access (GPU-written: read back here) are not: they may hold
//    GPU bytes not read back yet, which an invalidation as CPU-written would drop.
//  - KYTY_RAIZ_VERIFY: before opening, the write watch of each run is read (a page written while protected: lost);
//    after, a read window's runs must show no write (opened read-only), a write window's are reset (the copy's own).
// (raiz2) A read first probes its pages (NativeAllReadable): no window when none is no-access. The time of the windows
// themselves (gate, VirtualQuery, VirtualProtect, not the copy) is counted for the "[raiz]" line.
template <typename Access>
static void NativeGuardedAccess(const uint8_t* host, uint64_t bytes, Access&& access, bool writes = false) {
	const auto vaddr = reinterpret_cast<uint64_t>(host);
	// (ram15, KYTY_NATIVE_TEXTURE_OFFLOAD: dropped texture bytes under the window given back first; memory.cpp.)
	if (const auto hook = g_native_window_hook.load(std::memory_order_acquire); hook != nullptr && bytes != 0)
		hook(vaddr, bytes, writes, t_native_window_depth != 0 || t_native_protect_depth != 0);
	if (bytes == 0 || !NativeReadFaultsRequested() ||
	    !(NativeMayBeProtected(vaddr, bytes) || (writes && NativeMayBeWriteProtected(vaddr, bytes)))) {
		// (raiz11, KYTY_RAIZ_VERIFY) A read with no window: checked against the put-ins under way (only counted: no hint
		// is set over a put-in's pages until it ends, so this must stay 0).
		if (!writes && Raiz::VerifyOn()) {
			NativeReadTicket ticket;
			NativeReadTicketTake(vaddr, bytes, &ticket);
			access();
			if (!NativeReadTicketClean(ticket)) NativeNoteStale(Raiz::StaleDirect, vaddr, bytes, static_cast<uint64_t>(Raiz::t_site), false);
			return;
		}
		access();
		return;
	}
	const bool raiz = Raiz::On();
	auto&      c    = Raiz::g_counters;
	// (raiz5: no probe. raiz2-4 read one byte of each page first - GuardedCopy - to skip the window: 0 reads/s needed
	// none in every log, every probe was an access violation resumed, thousands a second on the render thread.)
	const auto       start      = std::chrono::steady_clock::now();
	const bool       reads_only = !writes && Graphics::PageManager::WatchProtects();
	const bool       verify     = Raiz::VerifyOn();
	NativeWindowLock lock;
	thread_local std::vector<NativeOpenedRun> opened;
	const size_t        first      = opened.size(); // (a window inside this one's access - a fault in a copy - nests)
	constexpr uint64_t  Page       = 0x1000;
	const uint64_t      end        = (vaddr + bytes + Page - 1u) & ~(Page - 1u);
	for (uint64_t at = vaddr & ~(Page - 1u); at < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) break;
		const uint64_t stop = std::min(end, reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize));
		const DWORD    prot = info.Protect & 0xffu;
		const bool     writable =
		    prot == PAGE_READWRITE || prot == PAGE_EXECUTE_READWRITE || prot == PAGE_WRITECOPY || prot == PAGE_EXECUTE_WRITECOPY;
		const bool readable = prot == PAGE_READONLY || prot == PAGE_EXECUTE_READ;
		if (info.State == MEM_COMMIT && !writable && !(reads_only && readable)) {
			const bool exec = prot == PAGE_EXECUTE || prot == PAGE_EXECUTE_READ;
			// (raiz1: a read opens read-only - a game write still faults -.)
			const DWORD open = (raiz && !writes) ? (exec ? PAGE_EXECUTE_READ : PAGE_READONLY)
			                                     : (exec ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE);
			// (raiz3: GPU results kept for these pages - lazy write-back - go in first: read-write for that, then the
			// window's own protection.)
			// (raiz6: taken out - stale ones dropped - before the opening, and before the audit's reset of the write watch.)
			thread_local NativeTaken window_taken;
			NativePendingTake(at, stop - at, &window_taken);
			const bool  pending    = !window_taken.empty();
			if (verify && info.Type == MEM_PRIVATE) (void)RaizTakeWritten(at, stop - at, "before an emulator copy");
			if (pending && NativeMergePutIn()) {
				// (crash-ba2b1f) The kept pages merged in first (NativePendingMergeOpen), ending with the window's own
				// protection; then the whole run (one VirtualQuery region: all of it had info.Protect) opened as before.
				NativePendingMergeOpen(window_taken, open, Raiz::LazyInWindow);
				DWORD ignored = 0;
				if (VirtualProtect(reinterpret_cast<void*>(at), stop - at, open, &ignored) != 0) {
					opened.push_back({at, stop - at, info.Protect, writes});
					if (verify && info.Type == MEM_PRIVATE) (void)RaizTakeWritten(at, stop - at, nullptr); // (its own write)
				} else {
					(void)VirtualProtect(reinterpret_cast<void*>(at), stop - at, info.Protect, &ignored); // (the kept pages back)
				}
				at = std::max(stop, at + Page);
				continue;
			}
			const DWORD first_open = pending ? (exec ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE) : open;
			DWORD old = 0;
			if (VirtualProtect(reinterpret_cast<void*>(at), stop - at, first_open, &old) != 0) {
				opened.push_back({at, stop - at, old, writes});
				if (pending) {
					NativePendingCopy(window_taken, Raiz::LazyInWindow);
					DWORD ignored = 0;
					if (first_open != open) (void)VirtualProtect(reinterpret_cast<void*>(at), stop - at, open, &ignored);
					if (verify && info.Type == MEM_PRIVATE) (void)RaizTakeWritten(at, stop - at, nullptr); // (its own write)
				}
			}
		}
		at = std::max(stop, at + Page);
	}
	const size_t opened_now = opened.size() - first;
	const auto   ready      = std::chrono::steady_clock::now();
	if (raiz && writes) NativeShadowDrop(vaddr, bytes); // (raiz5: written: the shadows of these pages go)
	// (raiz5: a read's pages opened from no-access - and the kept GPU bytes in - kept as shadows for the next reads.)
	if (raiz && !writes && t_native_fill_shadow && NativeShadowOn())
		for (size_t i = first; i < opened.size(); ++i)
			if ((opened[i].old & 0xffu) == PAGE_NOACCESS) NativeShadowFill(opened[i].at, opened[i].size);
	access();
	// (crash-ba2b1f, KYTY_RAIZ_WB_GUARD) An emulator write: not the game's for the next write-back's check.
	if (raiz && writes) Libs::LibKernel::Memory::RaizWbEmulatorWrote(vaddr, bytes);
	const auto done = std::chrono::steady_clock::now();
	if (opened_now != 0) {
		g_native_copy_protects.fetch_add(opened_now, std::memory_order_relaxed);
		g_native_guarded_copies.fetch_add(1, std::memory_order_relaxed);
	}
	if (raiz) {
		if (opened_now == 0) {
			c.empty_windows.fetch_add(1, std::memory_order_relaxed);
		} else {
			(writes ? c.write_windows : c.read_windows).fetch_add(1, std::memory_order_relaxed);
			if (Graphics::GuestGpu::IsGpuThread()) c.windows_gpu.fetch_add(1, std::memory_order_relaxed); // (raiz5)
			c.window_protects.fetch_add(opened_now, std::memory_order_relaxed);
			if (writes) Raiz::g_site_windows[Raiz::t_site].fetch_add(1, std::memory_order_relaxed); // (raiz2)
		}
	}
	if (opened_now != 0)
		NativeRestoreRuns(opened.data() + first, opened_now,
		                  static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
		                                            std::chrono::steady_clock::now() - ready)
		                                            .count()));
	opened.resize(first);
	if (raiz)
		c.window_us.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
		                                                (ready - start) + (std::chrono::steady_clock::now() - done))
		                                                .count()),
		                      std::memory_order_relaxed);
}

// (raiz3) A write of the emulator's into guest [host, host + bytes): `write_to(dst)` writes the bytes to dst (the guest
// address, or a buffer). A write-back of GPU results (NativeLazySite) over no-access pages is kept (above) instead of
// opening them; the rest goes through NativeGuardedAccess as before. The caller holds the address space's map lock
// (the gate is taken after it, as a protection change takes it).
template <typename WriteTo>
static void NativeGuardedWrite(uint8_t* host, uint64_t bytes, WriteTo&& write_to) {
	const auto vaddr = reinterpret_cast<uint64_t>(host);
	if (bytes == 0 || !NativeLazyOn() || !NativeLazySite() || !NativeMayBeProtected(vaddr, bytes)) {
		NativeGuardedAccess(host, bytes, [&] { write_to(host); }, true);
		return;
	}
	// (ram20, KYTY_RAM20=1) A write-back of GPU results over a range RAM20 dropped (no-access by its read watchers: it would
	// be kept aside below, with no window to ask for the bytes first) gets the range back from the disk store first, as a
	// window's write does (the hook; nothing there when no taken range is near). Off: one static bool read.
	static const bool ram20 = [] {
		const char* v = std::getenv("KYTY_RAM20");
		return v != nullptr && v[0] == '1' && v[1] == '\0';
	}();
	if (ram20) {
		if (const auto hook = g_native_window_hook.load(std::memory_order_acquire); hook != nullptr)
			hook(vaddr, bytes, true, t_native_window_depth != 0 || t_native_protect_depth != 0);
	}
	const auto start = std::chrono::steady_clock::now(); // (raiz4: timed, the "[raiz]" line)
	// (raiz4: the calling thread's buffer, not one allocated each write-back; a nested write - a fault in the copy -
	// gets its own.)
	thread_local std::vector<uint8_t> buffer;
	thread_local uint32_t             depth = 0;
	std::vector<uint8_t>              nested;
	auto&                             temp  = depth++ == 0 ? buffer : nested;
	struct Leave {
		uint32_t& d;
		~Leave() { --d; }
	} leave {depth};
	if (temp.size() < bytes) temp.resize(bytes);
	write_to(temp.data());
	struct Rest {
		uint64_t at, size;
	};
	std::vector<Rest>              rest;
	std::vector<NativePendingFill> fills; // (raiz7)
	{
		// (raiz10, KYTY_RAIZ_KEEP_STRIPES) The gate shared - a window waiting goes first; a window open keeps this out -
		// and this range's stripes alone (no protection change making these pages readable meanwhile); off: the gate alone
		// (raiz9: timed as a lazy keep). Inside a window or a protection of this thread: as before (nothing more taken).
		const bool striped  = NativeKeepStripesOn() && t_native_window_depth == 0 && t_native_protect_depth == 0;
		const auto held_at  = striped && Raiz::On() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		NativeWindowLock       alone(0, !striped);
		NativeProtectLock      shared(striped);
		NativeStripeLock<true> stripes(vaddr, bytes, striped);
		struct HeldTimer {
			bool                                  on;
			std::chrono::steady_clock::time_point at;
			~HeldTimer() {
				if (!on) return;
				Raiz::g_keep_stripe_n.fetch_add(1, std::memory_order_relaxed);
				Raiz::g_keep_stripe_us.fetch_add(
				    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - at).count()),
				    std::memory_order_relaxed);
			}
		} held_timer {striped && Raiz::On(), held_at};
		constexpr uint64_t Page = 0x1000;
		const uint64_t     end  = vaddr + bytes;
		for (uint64_t at = vaddr; at < end;) {
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) {
				rest.push_back({at, end - at});
				break;
			}
			const uint64_t stop =
			    std::min(end, reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize));
			const bool no_access = info.State == MEM_COMMIT && (info.Protect & 0xffu) == PAGE_NOACCESS;
			if (no_access) NativeShadowDrop(at, stop - at); // (raiz5: its bytes change: the shadow goes)
			// (raiz6: only where a tracker's read watcher holds the run - its no-access is the trackers', not a probe's -,
			// and its write watch reset once kept: a write to it from now on means its kept bytes are stale,
			// NativePendingTake.)
			const bool watched = no_access && g_gpu_resources != nullptr && g_gpu_resources->HasReadWatchers(at, stop - at);
			if (no_access && !watched) Raiz::g_lazy_unwatched.fetch_add(1, std::memory_order_relaxed);
			if (watched && NativePendingStore(at, temp.data() + (at - vaddr), stop - at, &fills)) {
				const uint64_t page_begin = at & ~(Page - 1u), page_end = (stop + Page - 1u) & ~(Page - 1u);
				// (raiz9) Only where the write watch is read at the take (NativeLazyCheck: the stale check, on with
				// KYTY_COLD_STORE=1): otherwise nothing reads what this reset, and it was a system call with the gate held
				// alone - every protection change of every thread waiting -. A kept page is no-access: no write of the
				// game's sets its bit, so the reset clears only what KYTY_RAIZ_VERIFY's audit would rightly call a lost
				// write. KYTY_RAIZ_KEEP_RESET=1: as raiz8 (reset always).
				if (NativeLazyCheck() || NativeKeepResetForced())
					(void)ResetWriteWatch(reinterpret_cast<void*>(page_begin), page_end - page_begin);
				else
					Raiz::g_keep_reset_skipped.fetch_add(1, std::memory_order_relaxed);
				Raiz::g_lazy_stored.fetch_add((page_end - page_begin) / Page, std::memory_order_relaxed);
			} else {
				if (watched) Raiz::g_lazy_full.fetch_add(1, std::memory_order_relaxed);
				rest.push_back({at, stop - at});
			}
			at = std::max(stop, (at & ~(Page - 1u)) + Page);
		}
	}
	NativePendingRunFills(fills); // (raiz7: the copies after the gate and the mutex)
	(Graphics::GuestGpu::IsGpuThread() ? Raiz::g_lazy_store_us_gpu : Raiz::g_lazy_store_us_other)
	    .fetch_add(static_cast<uint64_t>(
	                   std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()),
	               std::memory_order_relaxed);
	for (const auto& r: rest) {
		auto* to = reinterpret_cast<uint8_t*>(r.at);
		NativeGuardedAccess(to, r.size, [&] { std::memcpy(to, temp.data() + (r.at - vaddr), r.size); }, true);
	}
}

// (raiz5) A read of the emulator's of guest [host, host + bytes): `read_from(src)` reads the bytes from src (the guest
// address, or a buffer holding them). Pages with shadows: from them, no window (NativeShadowRead). Otherwise
// NativeGuardedAccess as before, its no-access pages kept as shadows for the next time.
template <typename ReadFrom>
static void NativeGuardedReadFrom(const uint8_t* host, uint64_t bytes, ReadFrom&& read_from) {
	const auto vaddr = reinterpret_cast<uint64_t>(host);
	if (bytes == 0 || !NativeShadowOn() || !NativeReadFaultsRequested() || !NativeMayBeProtected(vaddr, bytes)) {
		NativeGuardedAccess(host, bytes, [&] { read_from(host); }, false);
		return;
	}
	thread_local std::vector<uint8_t> buffer;
	if (bytes <= 64 * 4096) {
		if (buffer.size() < bytes) buffer.resize(bytes);
		if (NativeShadowRead(vaddr, bytes, buffer.data())) {
			Raiz::g_counters.shadow_hits.fetch_add(1, std::memory_order_relaxed);
			// (raiz6, KYTY_RAIZ_VERIFY: the shadow against guest memory itself, read through a window - its pending bytes
			// in first -: a difference is a shadow some path should have forgotten. Counted, the first 16 printed; the
			// read takes memory's bytes then.)
			if (Raiz::VerifyOn()) {
				thread_local std::vector<uint8_t> real;
				if (real.size() < bytes) real.resize(bytes);
				NativeGuardedAccess(host, bytes, [&] { std::memcpy(real.data(), host, bytes); }, false);
				if (std::memcmp(real.data(), buffer.data(), bytes) != 0) {
					Raiz::g_counters.shadow_mismatch.fetch_add(1, std::memory_order_relaxed);
					static std::atomic<uint32_t> said {0};
					if (said.fetch_add(1, std::memory_order_relaxed) < 16) {
						std::printf("[raiz] VERIFY: shadow of 0x%llx+0x%llx differs from guest memory (a stale shadow)\n",
						            static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(bytes));
						std::fflush(stdout);
					}
					read_from(real.data());
					return;
				}
			}
			read_from(buffer.data());
			return;
		}
	}
	const bool was = t_native_fill_shadow;
	t_native_fill_shadow = true;
	NativeGuardedAccess(host, bytes, [&] { read_from(host); }, false);
	t_native_fill_shadow = was;
}

// (ram11, KYTY_COLD_STORE: local/cold-store.h) The emulator's side of the cold store. Its thread probes and evicts with
// the gate taken alone, as a window takes it (no window open, no protection change of guest memory meanwhile).
static void ColdGateLock() {
	++t_native_window_depth; // (what it calls there takes no gate again)
	g_native_windows_waiting.fetch_add(1, std::memory_order_acq_rel);
	NativeProtectGate().lock();
	g_native_windows_waiting.fetch_sub(1, std::memory_order_acq_rel);
}
static void ColdGateUnlock() {
	NativeProtectGate().unlock();
	--t_native_window_depth;
}
static bool ColdTracked(uint64_t vaddr, uint64_t size) {
	return NativeMayBeProtected(vaddr, size) || NativeMayBeWriteProtected(vaddr, size);
}
static bool ColdPending(uint64_t vaddr, uint64_t size) {
	return NativePendingIn(vaddr, size);
}
// (ram13) Where the cold store may act: ONLY the map's ranges of the labels that hold data written once and then left
// (KYTY_COLD_STORE_LABELS, default VirtualMemo,gpupage,cpupage: the textures' streaming VA - memcpy 0x901459 -, the
// GPU and CPU page allocators' blocks over 1 MiB - mesh indices 0xac26c0 and vertices 0xaa1f70, CBuffers with data
// 0x90954b, load buffers -), and never anything overlapping one that is touched as allocator bookkeeping or every frame
// (Ghidra, eboot 01.007.000): TaggedFrameAllocator 0x811f60 keeps its free list INSIDE the arena (the loop at 0x8120b8
// reads it at each allocation: the crash of 11-16 01:48), the engine heap 0x8034b0 keeps its red-black tree of free
// blocks inside them (each alloc/free), the bucket allocators likewise; the frame rings 0x902f20/0x902f3f, the DCBs
// 0x9112ff, the 112 MiB pool backend+0xde8, the render-target arenas (0x91c82e/0x91c965). Without the map's snapshot
// nothing. (The cold store's thread only.)
static std::vector<std::string> ColdLabelList(const char* env, std::vector<std::string> fallback) {
	const char* v = std::getenv(env);
	if (v == nullptr || v[0] == '\0') return fallback;
	std::vector<std::string> out;
	std::string              item;
	for (const char* c = v;; ++c) {
		if (*c == ',' || *c == '\0') {
			if (!item.empty()) out.push_back(item);
			item.clear();
			if (*c == '\0') break;
		} else {
			item += *c;
		}
	}
	return out;
}
static bool ColdRangesOf(const std::vector<std::string>& labels, std::vector<uint64_t>* out) {
	std::vector<const char*> names;
	for (const auto& l: labels) names.push_back(l.c_str());
	std::vector<uint64_t> got(2 * 8192);
	const size_t          n = Loader::DemonsSoulsMemMap::RangesOf(names.data(), names.size(), got.data(), 8192);
	if (n == SIZE_MAX) return false;
	got.resize(2 * std::min<size_t>(n, 8192));
	out->swap(got);
	return true;
}
static bool ColdExcluded(uint64_t vaddr, uint64_t size) {
	if (Loader::DemonsSoulsMemMap::RtArenaOf(vaddr, vaddr + size) != 0) return true;
	static const std::vector<std::string> in_labels = ColdLabelList("KYTY_COLD_STORE_LABELS", {"VirtualMemo", "gpupage", "cpupage"});
	static const std::vector<std::string> out_labels =
	    ColdLabelList("KYTY_COLD_STORE_EXCLUDE", {"DefaultTagg", "ctx", "ring", "pool", "cpuheap", "gpuheap", "cpubkt", "gpubkt"});
	static std::vector<uint64_t> in_ranges, out_ranges; // [begin, end) pairs
	static bool                  known = false;
	static auto                  last  = std::chrono::steady_clock::time_point {};
	const auto                   now   = std::chrono::steady_clock::now();
	if (!known || now - last > std::chrono::seconds(2)) {
		last  = now;
		known = ColdRangesOf(in_labels, &in_ranges) && ColdRangesOf(out_labels, &out_ranges);
	}
	if (!known) return true;
	for (size_t i = 0; i + 1 < out_ranges.size(); i += 2)
		if (vaddr < out_ranges[i + 1] && out_ranges[i] < vaddr + size) return true;
	for (size_t i = 0; i + 1 < in_ranges.size(); i += 2)
		if (vaddr >= in_ranges[i] && vaddr + size <= in_ranges[i + 1]) return false;
	return true;
}
// (ram13) The gate as a protection change takes it (shared; a window waiting goes first): the probes take it so - they
// add a guard page by page and check each page's old protection, so only windows have to be kept out -. With
// KYTY_RAIZ_VERIFY alone, as the protection changes take it then.
static void ColdGateLockShared() {
	if (Raiz::VerifyOn()) {
		ColdGateLock();
		return;
	}
	++t_native_protect_depth;
	while (g_native_windows_waiting.load(std::memory_order_acquire) != 0) std::this_thread::yield();
	NativeProtectGate().lock_shared();
}
static void ColdGateUnlockShared() {
	if (Raiz::VerifyOn()) {
		ColdGateUnlock();
		return;
	}
	NativeProtectGate().unlock_shared();
	--t_native_protect_depth;
}
static void ColdForget(uint64_t vaddr, uint64_t size) {
	NativeShadowDrop(vaddr, size);
}
static bool ColdGpuThread() {
	return Graphics::GuestGpu::IsGpuThread();
}

// (ram4, KYTY_WRITE_WATCH) A native range the game unmapped and the emulator keeps committed (parked): pages a tracker
// still held no-access when the mapping went (a watcher that outlives the unmap: its later release finds no mapping to
// apply to, ProtectMappedUnlocked) are made read-write here. The game has no mapping there any more, and the
// emulator's late writes into a parked range - GPU results written back, file reads and labels queued before the unmap -
// are plain stores: on a no-access page each was an access violation in the exe ("Unhandled host exception"), which
// KYTY_NATIVE_MEMORY=1 alone never had (it protects nothing). Only where the read hints say a page may be protected
// (one check otherwise); a map there commits the pages again with the mapping's protection (AdoptParkedNative).
static void NativeOpenParked(uint64_t vaddr, uint64_t size) {
	NativePendingDrop(vaddr, size); // (raiz3: unmapped - parked -: what was kept for it is nobody's)
	// (ww2, KYTY_WRITE_WATCH_MODE=protect: the pages a write watcher left read-only too.)
	if (size == 0 || !NativeReadFaultsRequested() || !(NativeMayBeProtected(vaddr, size) || NativeMayBeWriteProtected(vaddr, size)))
		return;
	const bool       read_only_too = Graphics::PageManager::WatchProtects();
	NativeWindowLock lock; // (raiz1: the gate, alone)
	constexpr uint64_t Page   = 0x1000;
	const uint64_t     end    = vaddr + size;
	uint64_t           opened = 0;
	for (uint64_t at = vaddr; at < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) break;
		const uint64_t stop = std::min(end, reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize));
		if (info.State == MEM_COMMIT &&
		    ((info.Protect & 0xffu) == PAGE_NOACCESS || (read_only_too && (info.Protect & 0xffu) == PAGE_READONLY)) && stop > at) {
			DWORD old = 0;
			if (VirtualProtect(reinterpret_cast<void*>(at), stop - at, PAGE_READWRITE, &old) != 0) opened += stop - at;
		}
		at = std::max(stop, at + Page);
	}
	if (opened == 0) return;
	g_native_copy_protects.fetch_add(1, std::memory_order_relaxed);
	static std::atomic<uint32_t> said {0};
	if (said.fetch_add(1, std::memory_order_relaxed) < 8) {
		std::printf("Write watch: unmapped range 0x%llx+0x%llx had %llu KiB still no-access: opened (parked memory is "
		            "read-write)\n",
		            static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
		            static_cast<unsigned long long>(opened >> 10u));
		std::fflush(stdout);
	}
}
#else
template <typename Access>
static void NativeGuardedAccess(const uint8_t* /*host*/, uint64_t /*bytes*/, Access&& access, bool /*writes*/ = false) {
	access();
}
template <typename WriteTo>
static void NativeGuardedWrite(uint8_t* host, uint64_t /*bytes*/, WriteTo&& write_to) {
	write_to(host);
}
template <typename ReadFrom>
static void NativeGuardedReadFrom(const uint8_t* host, uint64_t /*bytes*/, ReadFrom&& read_from) {
	read_from(host);
}

#endif

// (nativemem1, KYTY_WRITE_WATCH) A pointer to guest bytes some tracker may have made no-access is not handed out (the
// backing accessors return none: their callers copy instead - TryReadBacking & co., NativeGuardedAccess -): a later read
// through it would fault in the emulator's own code (an upload worker, a hash) with nothing to bypass the protection.
// (raiz2) Set by Memory::TryGetUploadSourcePointer for its call: the upload worker's copies read through
// Common::GuardedCopy and a window where that stops (Memory::NativeGuardedRead): a pointer over protected pages is safe
// for them.
static thread_local bool t_native_allow_protected_pointer = false;
static bool NativeRefusePointer(uint64_t vaddr, uint64_t size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (t_native_allow_protected_pointer) return false;
	return NativeReadFaultsRequested() && NativeMayBeProtected(vaddr, size);
#else
	(void)vaddr, (void)size;
	return false;
#endif
}

// A view mapped again: counted, and left to ReapplyRemappedViews unless ProtectRemapped protected it.
static void NoteRemapped(uint64_t vaddr, uint64_t size) {
	++g_remapped_count;
	if (!g_protective_remap) g_remapped_views.emplace_back(vaddr, size);
}

static void ReapplyRemappedViews() {
	for (const auto& [vaddr, size]: std::exchange(g_remapped_views, {})) {
		if (g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size)) {
			g_gpu_resources->ReapplyProtection(vaddr, size);
		}
	}
}

// Backing aliases bypass guest page protection. Invalidate hoisted preparation
// across every such write and mapping/protection mutation on every host OS.
struct GuestBackingMutation {
	Graphics::GpuResourceManager* resources = g_gpu_resources;
	GuestBackingMutation() { if (resources) resources->BeginAliasWrite(); }
	~GuestBackingMutation() { if (resources) resources->EndAliasWrite(); }
};

class GuestBackingStore {
public:
	enum class FailureReason {
		None,
		BackingUnavailable,
		PhysicalRangeOutOfBounds,
		FixedMapViewFailed,
		PlaceholderMapFailed,
		ProtectFailed,
		WrongHostAddress,
		AddressRangeUnavailable,
		NativeAliasRefused, // (KYTY_NATIVE_MEMORY) the physical range is mapped at another address already
	};

	explicit GuestBackingStore(uint64_t size): m_size(size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		// KYTY_NATIVE_MEMORY=1: no section and no backing view at all (no commit charge for them either); each guest
		// mapping is private memory committed when it is mapped (ReplacePlaceholderNative). KYTY_BACKING_ON_DEMAND,
		// KYTY_BACKING_FILE and KYTY_GUEST_LAZY_COMMIT do not apply then.
		m_native = NativeMemoryRequested();
		if (m_native) {
			std::printf("Guest memory: %llu MiB of physical memory, native: private memory at the guest addresses, "
			            "committed when mapped, write-watched (KYTY_NATIVE_MEMORY; no backing view)\n",
			            static_cast<unsigned long long>(m_size >> 20u));
			std::fflush(stdout);
			{ // (ram11) The cold store's thread (it does nothing without KYTY_COLD_STORE=1, read in it 10 s later).
				ColdStore::Hooks hooks;
				hooks.tracked     = &ColdTracked;
				hooks.pending     = &ColdPending;
				hooks.gate_lock   = &ColdGateLock;
				hooks.gate_unlock = &ColdGateUnlock;
				hooks.gate_lock_shared   = &ColdGateLockShared;   // (ram13)
				hooks.gate_unlock_shared = &ColdGateUnlockShared; // (ram13)
				hooks.gpu_thread  = &ColdGpuThread;
				hooks.ranges      = &ColdRanges;
				hooks.ranges_user = this;
				hooks.excluded    = &ColdExcluded;
				hooks.forget      = &ColdForget;
				ColdStore::Start(hooks);
				F0Measure::SetGate(&ColdGateLock, &ColdGateUnlock); // (f0: the read sampler's guards, gate alone)
			}
			return;
		}
		// KYTY_BACKING_ON_DEMAND=1: the section is only reserved and each range is committed when a guest
		// mapping first covers it (ReplacePlaceholderView). SEC_COMMIT charged the whole 13.5 GiB to the
		// commit limit at launch: 17.5-18.7 GB committed for 5-10 GB resident (MEMORY_MANAGEMENT reboots).
		m_on_demand = [] {
			const char* v = std::getenv("KYTY_BACKING_ON_DEMAND");
			return v != nullptr && v[0] == '1';
		}();
		// KYTY_BACKING_FILE=1 (or a folder): the section backed by a sparse, delete-on-close file on disk (the
		// emulator's folder by default) instead of the page file. File-backed pages are not charged to the commit
		// limit at all (13.5 GiB less commit charge: 16 GB PCs bug-checked / rebooted at the limit), and under memory
		// pressure Windows writes the cold ones to that file and drops them instead of failing. Same bytes, same
		// zero-filled pages, same views: nothing the game reads changes. Falls back to the page file when the file
		// cannot be made or its drive has less than the store's size + 4 GiB free.
		m_file = OpenBackingFile(m_size);
		if (m_file != INVALID_HANDLE_VALUE) m_on_demand = false; // (SEC_RESERVE is for page-file sections only)
		// KYTY_GUEST_LAZY_COMMIT=1 (guest-lazy-commit.h, off by default): reserved too, but each granule committed at its
		// first touch (a fault, a kernel-mode access), not when a mapping covers it; nothing decommitted or moved later.
		m_lazy = m_file == INVALID_HANDLE_VALUE && GuestLazyCommit::Requested();
		if (m_lazy) m_on_demand = true;
		m_handle = CreateFileMappingA(
		    m_file, nullptr,
		    PAGE_EXECUTE_READWRITE | (m_on_demand ? SEC_RESERVE : SEC_COMMIT),
		    static_cast<DWORD>(m_size >> 32u), static_cast<DWORD>(m_size & 0xffffffffu), nullptr);
		if (m_handle == nullptr && m_file != INVALID_HANDLE_VALUE) {
			std::printf("Guest memory: file backing refused (CreateFileMapping 0x%08x), the page file instead\n",
			            static_cast<unsigned>(GetLastError()));
			CloseHandle(m_file);
			m_file   = INVALID_HANDLE_VALUE;
			m_handle = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr,
			                              PAGE_EXECUTE_READWRITE | (m_on_demand ? SEC_RESERVE : SEC_COMMIT),
			                              static_cast<DWORD>(m_size >> 32u), static_cast<DWORD>(m_size & 0xffffffffu),
			                              nullptr);
		}
		std::printf("Guest memory: %llu MiB backing store %s\n", static_cast<unsigned long long>(m_size >> 20u),
		            m_file != INVALID_HANDLE_VALUE ? "in a sparse file on disk (KYTY_BACKING_FILE: no commit charge)"
		            : m_lazy      ? "in the page file, reserved: committed at first touch (KYTY_GUEST_LAZY_COMMIT)"
		            : m_on_demand ? "in the page file, committed on demand (KYTY_BACKING_ON_DEMAND)"
		                          : "in the page file, all committed at launch");
		std::fflush(stdout);
		if (m_handle == nullptr) {
			LOGF_COLOR(Log::Color::Red,
			           "\t direct-memory backing: CreateFileMapping failed: 0x%08" PRIx32 "\n",
			           static_cast<uint32_t>(GetLastError()));
			return;
		}
		m_backing_base = static_cast<uint8_t*>(
		    MapViewOfFile(m_handle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, m_size));
		if (m_backing_base == nullptr) {
			LOGF_COLOR(Log::Color::Red,
			           "\t direct-memory backing: writable alias failed: 0x%08" PRIx32 "\n",
			           static_cast<uint32_t>(GetLastError()));
			CloseHandle(m_handle);
			m_handle = nullptr;
		}
		if (m_lazy && m_backing_base != nullptr) GuestLazyCommit::Start(reinterpret_cast<uint64_t>(m_backing_base), m_size);
		// (The RAM guard counts the store's commit charge: all of it, what is committed of it, or none.)
		if (m_backing_base != nullptr)
			RamGuard::NoteGuestStore(m_backing_base, m_size,
			                         m_file != INVALID_HANDLE_VALUE ? RamGuard::StoreCharge::File
			                         : m_lazy                       ? RamGuard::StoreCharge::Lazy
			                         : m_on_demand                  ? RamGuard::StoreCharge::OnDemand
			                                                        : RamGuard::StoreCharge::AllCommitted);
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#if defined(__APPLE__)
		{
			char shm_name[64];
			snprintf(shm_name, sizeof(shm_name), "/kyty-dm-%d-%llx", static_cast<int>(getpid()),
			         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(this)));
			m_fd = shm_open(shm_name, O_RDWR | O_CREAT | O_EXCL, 0600);
			if (m_fd >= 0) {
				shm_unlink(shm_name);
			}
		}
#else
		m_fd = static_cast<int>(syscall(SYS_memfd_create, "KytyDirectMemory", 0));
#endif
		if (m_fd < 0) {
			LOGF_COLOR(Log::Color::Red, "\t direct-memory backing: shared fd creation failed: %d\n",
			           errno);
			return;
		}
		if (ftruncate(m_fd, static_cast<off_t>(m_size)) != 0) {
			LOGF_COLOR(Log::Color::Red, "\t direct-memory backing: ftruncate failed: %d\n", errno);
			close(m_fd);
			m_fd = -1;
			return;
		}
		auto* backing = mmap(nullptr, m_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);
		if (backing == MAP_FAILED) {
			LOGF_COLOR(Log::Color::Red, "\t direct-memory backing: writable alias failed: %d\n",
			           errno);
			close(m_fd);
			m_fd = -1;
			return;
		}
		m_backing_base = static_cast<uint8_t*>(backing);
#endif
	}

	~GuestBackingStore() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (const auto& [base, size]: m_views) {
			ReleaseView(reinterpret_cast<void*>(base), size);
		}
		m_views.clear();
		m_parked.clear();
		m_maps.clear();
		m_coverage.clear();
		if (m_backing_base != nullptr) {
			UnmapViewOfFile(m_backing_base);
		}
		if (m_handle != nullptr) {
			CloseHandle(m_handle);
		}
		if (m_file != INVALID_HANDLE_VALUE) {
			CloseHandle(m_file); // (delete on close: the file goes with the process)
		}
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		for (const auto& [_, entry]: m_maps) {
			EXIT_IF(
			    !UnmapViewPreservePlaceholder(reinterpret_cast<void*>(entry.vaddr), entry.size));
		}
		m_maps.clear();
		m_coverage.clear();
		if (m_backing_base != nullptr) {
			munmap(m_backing_base, m_size);
		}
		if (m_fd >= 0) {
			close(m_fd);
		}
#endif
	}

	KYTY_CLASS_NO_COPY(GuestBackingStore);

	static const char* GetFailureReasonName(FailureReason reason) {
		switch (reason) {
			case FailureReason::None: return "n/a";
			case FailureReason::BackingUnavailable: return "backing-unavailable";
			case FailureReason::PhysicalRangeOutOfBounds: return "physical-range-out-of-bounds";
			case FailureReason::FixedMapViewFailed: return "fixed-map-view-failed";
			case FailureReason::PlaceholderMapFailed: return "placeholder-map-failed";
			case FailureReason::ProtectFailed: return "protect-failed";
			case FailureReason::WrongHostAddress: return "wrong-host-address";
			case FailureReason::AddressRangeUnavailable: return "address-range-unavailable";
			case FailureReason::NativeAliasRefused: return "native-alias-refused";
			default: return "unknown";
		}
	}

	[[nodiscard]] bool IsAvailable() const {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (m_native) return true;
		return m_handle != nullptr && m_backing_base != nullptr;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		return m_fd >= 0 && m_backing_base != nullptr;
#else
		return false;
#endif
	}

	// 0 with KYTY_NATIVE_MEMORY (no backing view: its users - FreshMemory, the RAM flushes, BackingView - stand down).
	[[nodiscard]] uint64_t GetBackingBase() const {
		return reinterpret_cast<uint64_t>(m_backing_base);
	}

	[[nodiscard]] bool IsNative() const { return m_native; }

	[[nodiscard]] uint64_t GetBackingSize() const { return m_size; }
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	bool   m_on_demand = false;
	bool   m_lazy      = false;
	HANDLE m_file      = INVALID_HANDLE_VALUE;

	// The sparse backing file (KYTY_BACKING_FILE), or INVALID_HANDLE_VALUE (off, or not possible: the page file).
	static HANDLE OpenBackingFile(uint64_t size) {
		const char* v = std::getenv("KYTY_BACKING_FILE");
		if (v == nullptr || v[0] == '\0' || (v[0] == '0' && v[1] == '\0')) return INVALID_HANDLE_VALUE;
		std::wstring folder;
		if (v[0] == '1' && v[1] == '\0') {
			wchar_t exe[MAX_PATH] {};
			const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
			folder.assign(exe, n);
			const auto slash = folder.find_last_of(L"\\/");
			folder = slash == std::wstring::npos ? L"." : folder.substr(0, slash);
		} else {
			const int n = MultiByteToWideChar(CP_UTF8, 0, v, -1, nullptr, 0);
			folder.resize(n > 0 ? n - 1 : 0);
			if (n > 0) MultiByteToWideChar(CP_UTF8, 0, v, -1, folder.data(), n);
		}
		ULARGE_INTEGER free_bytes {};
		if (GetDiskFreeSpaceExW(folder.c_str(), &free_bytes, nullptr, nullptr) == 0 ||
		    free_bytes.QuadPart < size + (uint64_t {4} << 30u)) {
			std::printf("Guest memory: KYTY_BACKING_FILE needs %llu MiB free on its drive (%llu MiB): the page file instead\n",
			            static_cast<unsigned long long>((size >> 20u) + 4096),
			            static_cast<unsigned long long>(free_bytes.QuadPart >> 20u));
			return INVALID_HANDLE_VALUE;
		}
		const std::wstring path = folder + L"\\kyty-guest-memory-" + std::to_wstring(GetCurrentProcessId()) + L".tmp";
		// Temporary + delete on close: Windows keeps its pages in RAM while there is room (no lazy write-back of a
		// temporary file's pages) and removes the file when the process ends, however it ends.
		HANDLE file = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE | GENERIC_EXECUTE, 0, nullptr, CREATE_ALWAYS,
		                          FILE_ATTRIBUTE_TEMPORARY | FILE_ATTRIBUTE_NOT_CONTENT_INDEXED | FILE_FLAG_DELETE_ON_CLOSE,
		                          nullptr);
		if (file == INVALID_HANDLE_VALUE) {
			std::printf("Guest memory: KYTY_BACKING_FILE could not create its file (0x%08x): the page file instead\n",
			            static_cast<unsigned>(GetLastError()));
			return INVALID_HANDLE_VALUE;
		}
		DWORD returned = 0;
		const bool sparse = DeviceIoControl(file, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr) != 0;
		LARGE_INTEGER end {};
		end.QuadPart = static_cast<LONGLONG>(size);
		if (!sparse || SetFilePointerEx(file, end, nullptr, FILE_BEGIN) == 0 || SetEndOfFile(file) == 0) {
			std::printf("Guest memory: KYTY_BACKING_FILE could not size its sparse file (0x%08x): the page file instead\n",
			            static_cast<unsigned>(GetLastError()));
			CloseHandle(file);
			return INVALID_HANDLE_VALUE;
		}
		return file;
	}
#endif

	// Locking: m_mutex (shared) keeps the mappings stable; mapping changes take it exclusively.
	// m_data_mutex serializes the copies into and out of the backing (the write fence below
	// assumes one writer at a time), so mapping queries such as IsUniqueRange no longer wait
	// behind a GPU readback being written back into guest memory.
	// Diagnostic (live measure: backing_lock_*, backing_hold_*): the render thread's waits for
	// either lock are counted when it is contended, and how long the writers hold them.
	template <typename Lock>
	static void AcquireCounted(Lock& lock, bool write = false) {
		if (lock.try_lock()) return;
		const auto start = std::chrono::steady_clock::now();
		lock.lock();
		if (Graphics::GuestGpu::IsGpuThread()) {
			const auto us = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
			LiveCounters::Add(write ? LiveCounters::BackingWriteWaits : LiveCounters::BackingLockWaits);
			LiveCounters::Add(write ? LiveCounters::BackingWriteWaitUs : LiveCounters::BackingLockWaitUs, us);
			LogWaitStack(us, std::is_same_v<Lock, std::shared_lock<std::shared_mutex>> ? "map" : write ? "write" : "data");
		}
	}
	// KYTY_BACKING_WAIT_STACK=<us> (diagnostics): the render thread's stack at a backing lock wait of at least <us>
	// microseconds, once per distinct stack (64 at most), with the wait counted per stack in the summary lines.
	static void LogWaitStack([[maybe_unused]] uint64_t us, [[maybe_unused]] const char* kind) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		static const uint64_t threshold = [] {
			const char* value = std::getenv("KYTY_BACKING_WAIT_STACK");
			return value != nullptr ? std::strtoull(value, nullptr, 10) : UINT64_MAX;
		}();
		if (us < threshold) return;
		std::array<void*, 16> frames {};
		unsigned long         hash  = 0;
		const auto            count = RtlCaptureStackBackTrace(1, static_cast<unsigned long>(frames.size()), frames.data(), &hash);
		struct Seen { uint64_t waits = 0, us = 0; };
		static std::mutex                               mutex;
		static std::unordered_map<unsigned long, Seen> seen;
		uint64_t                                        waits = 0, sum = 0;
		{
			std::lock_guard guard(mutex);
			if (seen.size() >= 64 && !seen.contains(hash)) return;
			auto& entry = seen[hash];
			waits = ++entry.waits;
			sum = entry.us += us;
		}
		if (waits != 1 && (waits & (waits - 1)) != 0) return;
		std::string text = "BACKING-WAIT " + std::string(kind) + " " + std::to_string(us) + " us (stack " +
		                   std::to_string(hash) + ": " + std::to_string(waits) + " waits, " + std::to_string(sum) + " us):";
		for (unsigned i = 0; i < count; i++) {
			const auto address = reinterpret_cast<uint64_t>(frames[i]);
			char       part[40];
			std::snprintf(part, sizeof(part),
			              address >= 0x140000000ull && address < 0x150000000ull ? " exe+0x%llx" : " 0x%llx",
			              static_cast<unsigned long long>(address >= 0x140000000ull && address < 0x150000000ull
			                                                  ? address - 0x140000000ull
			                                                  : address));
			text += part;
		}
		std::printf("%s\n", text.c_str());
		std::fflush(stdout);
#endif
	}
	std::shared_lock<std::shared_mutex> MapReadLock() {
		std::shared_lock lock(m_mutex, std::defer_lock);
		AcquireCounted(lock);
		return lock;
	}
	std::unique_lock<std::mutex> DataLock(bool write = false) {
		std::unique_lock lock(m_data_mutex, std::defer_lock);
		AcquireCounted(lock, write);
		return lock;
	}
	struct HoldTimer {
		LiveCounters::Id                      id;
		std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		~HoldTimer() {
			LiveCounters::Add(id, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                              std::chrono::steady_clock::now() - start)
			                                              .count()));
		}
	};

	bool ZeroRange(uint64_t backing_offset, uint64_t size) {
		GuestBackingMutation mutation;
		if (!IsAvailable() || !RangeOk(backing_offset, size)) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		HoldTimer       hold {LiveCounters::BackingHoldZeroUs};
		BackingWriteFence fence(*this);
		// Only the granules a guest mapping has covered before are zeroed: the others hold the zeros
		// the OS gives untouched backing. Zeroing everything touched every page of each new allocation
		// (a 1.35 GB streaming pool the game then fills only in part), which stayed resident: system
		// RAM grew by what the game reserved, not by what it used.
		const uint64_t end = backing_offset + size;
		for (uint64_t offset = backing_offset; offset < end;) {
			const uint64_t granule = offset >> UsedGranuleBits;
			const uint64_t next    = std::min(end, (granule + 1) << UsedGranuleBits);
			if (IsUsedGranule(granule)) {
				if (m_native) ZeroMappedNative(offset, next);
				else ZeroCommitted(offset, next);
			}
			offset = next;
		}
		NoteUsedUnlocked(backing_offset, size);
		return true;
	}

	// A guest mapping is about to cover the range: its granules count as (possibly) written.
	void NoteUsed(uint64_t backing_offset, uint64_t size) {
		if (!IsAvailable() || !RangeOk(backing_offset, size)) return;
		std::lock_guard lock(m_mutex);
		NoteUsedUnlocked(backing_offset, size);
	}

	bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
		GuestBackingMutation mutation;
		if (data == nullptr) {
			return false;
		}
		const auto* source = static_cast<const uint8_t*>(data);
		return TryTransferBacking<true>(vaddr, size,
		                          [source](uint8_t* backing, uint64_t data_offset, size_t bytes) {
			                          std::memcpy(backing, source + data_offset, bytes);
			                          FreshMemory::MarkBacking(reinterpret_cast<uint64_t>(backing), bytes);
		                          });
	}

	// A write of bytes no data writer writes, such as a linear copy's into pages the CPU owns (TryWriteBacking's GPU
	// results go to GPU-owned pages only: BufferCache::WriteBackGpuOwned): its own mutex and fence, so it never waits
	// behind a data write (the render thread's linear copies waited there ~1.5 times a frame, 0.13 ms at Latria). A
	// reader excludes both kinds of writer (the slow path both mutexes, the fast one both fences). False when one
	// mapping does not cover the range.
	bool TryWriteCpuBacking(uint64_t vaddr, const void* data, uint64_t size) {
		GuestBackingMutation mutation;
		if (data == nullptr || !IsAvailable() || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		auto            lock = MapReadLock();
		std::lock_guard cpu_lock(m_cpu_write_mutex);
		MapEntry        entry {};
		if (!FindContainingUnlocked(vaddr, size, &entry)) {
			return false;
		}
		const auto begin = entry.backing_offset + vaddr - entry.vaddr;
		if (!RangeOk(begin, size)) {
			return false;
		}
		DataWriteFence fence(*this, begin, begin + size, m_cpu_write_pages);
		uint8_t* host = HostAt(entry, vaddr);
		NativeGuardedAccess(host, size, [&] { std::memcpy(host, data, size); }, true); // (nativemem1; ww2: a write)
		FreshMemory::MarkBacking(reinterpret_cast<uint64_t>(host), size);
		return true;
	}

	bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
		if (data == nullptr) {
			return false;
		}
		auto* destination = static_cast<uint8_t*>(data);
		// Large reads are timed: their pages are often first touched through the backing view.
		const bool large = size >= 0x10000;
		const auto start = large ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		const bool ok    = TryTransferBacking<false>(
            vaddr, size, [destination](uint8_t* backing, uint64_t data_offset, size_t bytes) {
                std::memcpy(destination + data_offset, backing, bytes);
            });
		if (large && ok) {
			LiveCounters::Add(LiveCounters::BackingReadBytes, size);
			LiveCounters::Add(LiveCounters::BackingReadUs,
			                  static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                            std::chrono::steady_clock::now() - start)
			                                            .count()));
		}
		return ok;
	}

	bool TryReadBackingToHost(uint64_t vaddr, void* data, uint64_t size) {
		if (kyty_local_backing_read_mode.load(std::memory_order_relaxed) != 0 && IsAvailable() &&
		    data && size && size <= 0x10000 && UINT64_MAX - vaddr >= size) {
			BackingReadTransaction read(*this);
			if (read.Acquired()) {
				auto&      cached = HostReadTranslationOf(vaddr);
				const auto epoch  = RegionEpoch(vaddr);
				MapEntry   entry {};
				if (cached.epoch == epoch && GuestFreeRangeContains(cached.entry.vaddr, cached.entry.size, vaddr, size)) {
					entry = cached.entry;
				} else if (FindContainingUnlocked(vaddr, size, &entry)) {
					cached = {entry, epoch};
				}
				if (entry.size && RangeOk(entry.backing_offset + vaddr - entry.vaddr, size)) {
					const auto offset = entry.backing_offset + vaddr - entry.vaddr;
					if (read.Narrow(offset, offset + size)) {
						const uint8_t* host = HostAt(entry, vaddr);
						NativeGuardedReadFrom(host, size, [&](const uint8_t* from) { std::memcpy(data, from, size); }); // (raiz5: shadows)
						return true;
					}
				}
			}
			// The transaction is released here, before a spanning transfer or a
			// writer-held mutex can wait.
		}
		return TryReadBacking(vaddr, data, size);
	}

	// The backing bytes of [vaddr, vaddr + size) when one mapping covers the range, else null.
	// The backing view lives as long as the store; unmapping only detaches the guest range.
	const uint8_t* BackingPointer(uint64_t vaddr, uint64_t size) {
		if (!IsAvailable() || size == 0 || UINT64_MAX - vaddr < size) {
			return nullptr;
		}
		MapEntry entry {};
		{
			BackingReadTransaction read(*this);
			if (read.Acquired()) {
				auto&      cached = HostReadTranslationOf(vaddr);
				const auto epoch  = RegionEpoch(vaddr);
				if (cached.epoch == epoch && GuestFreeRangeContains(cached.entry.vaddr, cached.entry.size, vaddr, size)) {
					entry = cached.entry;
				} else if (FindContainingUnlocked(vaddr, size, &entry)) {
					cached = {entry, epoch};
				}
			}
		}
		if (entry.size == 0) {
			auto lock = MapReadLock();
			if (!FindContainingUnlocked(vaddr, size, &entry)) {
				return nullptr;
			}
		}
		const auto offset = entry.backing_offset + vaddr - entry.vaddr;
		// (nativemem1, KYTY_WRITE_WATCH: none over pages a tracker may have made no-access: NativeRefusePointer.)
		return RangeOk(offset, size) && !NativeRefusePointer(vaddr, size) ? HostAt(entry, vaddr) : nullptr;
	}

	// The end of the mapping `vaddr` is in (0: none), through this thread's host read translations (BackingPointer's
	// lookups of the same mappings then find them there).
	uint64_t MappingEndOf(uint64_t vaddr) {
		if (!IsAvailable()) {
			return 0;
		}
		{
			BackingReadTransaction read(*this);
			if (read.Acquired()) {
				auto&      cached = HostReadTranslationOf(vaddr);
				const auto epoch  = RegionEpoch(vaddr);
				if (cached.epoch == epoch && GuestFreeRangeContains(cached.entry.vaddr, cached.entry.size, vaddr, 1)) {
					return cached.entry.vaddr + cached.entry.size;
				}
				MapEntry entry {};
				if (!FindContainingUnlocked(vaddr, 1, &entry)) return 0;
				cached = {entry, epoch};
				return entry.vaddr + entry.size;
			}
		}
		auto     lock = MapReadLock();
		MapEntry entry {};
		return FindContainingUnlocked(vaddr, 1, &entry) ? entry.vaddr + entry.size : 0;
	}

	// The backing bytes of [vaddr, vaddr + size) as one piece per guest mapping it spans, in
	// order (the pieces are contiguous in the guest range); false if a part is not mapped.
	bool BackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces) {
		pieces.clear();
		if (!IsAvailable() || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		if (NativeRefusePointer(vaddr, size)) return false; // (nativemem1, KYTY_WRITE_WATCH: the caller copies instead)
		auto lock = MapReadLock();
		const auto end = vaddr + size;
		for (auto current = vaddr; current < end;) {
			MapEntry entry {};
			if (!FindContainingUnlocked(current, 1, &entry)) {
				return false;
			}
			const auto bytes  = std::min(end, entry.vaddr + entry.size) - current;
			const auto offset = entry.backing_offset + current - entry.vaddr;
			if (!RangeOk(offset, bytes)) {
				return false;
			}
			pieces.emplace_back(HostAt(entry, current), bytes);
			current += bytes;
		}
		return true;
	}

	// As BackingPieces, for a range with unmapped parts: a hole is a piece with a null pointer. False if
	// nothing of the range is mapped (or a mapping's backing is out of range).
	bool SparseBackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces) {
		pieces.clear();
		if (!IsAvailable() || size == 0 || UINT64_MAX - vaddr < size) return false;
		// (nativemem1, KYTY_WRITE_WATCH: none over pages a tracker may have made no-access; the caller reads the range with
		// TryReadMappedOrZero instead, NativeGuardedAccess.)
		if (NativeRefusePointer(vaddr, size)) return false;
		auto       lock   = MapReadLock();
		const auto end    = vaddr + size;
		bool       mapped = false;
		auto       first  = m_maps.lower_bound(vaddr);
		if (first != m_maps.begin()) {
			auto previous = std::prev(first);
			if (previous->second.vaddr + previous->second.size > vaddr) first = previous;
		}
		uint64_t current = vaddr;
		for (auto mapping = first; mapping != m_maps.end() && mapping->second.vaddr < end; ++mapping) {
			const auto& entry = mapping->second;
			const auto  start = std::max(vaddr, entry.vaddr);
			const auto  bytes = std::min(end, entry.vaddr + entry.size) - start;
			if (!RangeOk(entry.backing_offset + start - entry.vaddr, bytes)) return false;
			if (start > current) pieces.emplace_back(nullptr, start - current);
			pieces.emplace_back(HostAt(entry, start), bytes);
			current = start + bytes;
			mapped  = true;
		}
		if (current < end) pieces.emplace_back(nullptr, end - current);
		return mapped;
	}

	bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size) {
		if (!IsAvailable() || data == nullptr || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		auto            lock      = MapReadLock();
		auto            data_lock = DataLock();
		std::lock_guard cpu_lock(m_cpu_write_mutex);
		const auto      end   = vaddr + size;
		auto            first = m_maps.lower_bound(vaddr);
		if (first != m_maps.begin()) {
			auto previous = std::prev(first);
			if (previous->second.vaddr + previous->second.size > vaddr) {
				first = previous;
			}
		}
		for (auto mapping = first; mapping != m_maps.end() && mapping->second.vaddr < end;
		     ++mapping) {
			const auto& entry = mapping->second;
			const auto  start = std::max(vaddr, entry.vaddr);
			const auto  bytes = std::min(end, entry.vaddr + entry.size) - start;
			if (!RangeOk(entry.backing_offset + start - entry.vaddr, bytes)) {
				return false;
			}
		}

		auto* destination = static_cast<uint8_t*>(data);
		std::memset(destination, 0, size);
		for (auto mapping = first; mapping != m_maps.end() && mapping->second.vaddr < end;
		     ++mapping) {
			const auto& entry  = mapping->second;
			const auto  start  = std::max(vaddr, entry.vaddr);
			const auto  bytes  = std::min(end, entry.vaddr + entry.size) - start;
			const uint8_t* host = HostAt(entry, start);
			NativeGuardedReadFrom(host, bytes, [&](const uint8_t* from) { std::memcpy(destination + start - vaddr, from, bytes); }); // (raiz5)
		}
		return true;
	}

	// `protective`: a view mapped again (UnmapOne, Unmap) gets its watched pages' protection before it is published
	// (ProtectRemapped; the caller holds the address-space lock).
	bool MapExistingPlaceholderFixed(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                                 VirtualMemory::Mode mode,
	                                 FailureReason*      failure_reason = nullptr, bool protective = false) {
		GuestBackingMutation mutation;
		if (!IsAvailable() || !RangeOk(backing_offset, size)) {
			SetFailure(failure_reason, !IsAvailable() ? FailureReason::BackingUnavailable
			                                          : FailureReason::PhysicalRangeOutOfBounds);
			return false;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		// (Native: private memory holds one guest address's bytes; a physical range already mapped elsewhere - the PS5's
		// aliases - cannot be mapped a second time.)
		if (m_native && !NativeCoverageFree(backing_offset, size, vaddr)) {
			SetFailure(failure_reason, FailureReason::NativeAliasRefused);
			return false;
		}
		void* mapped = ReplacePlaceholderView(vaddr, size, backing_offset, mode, failure_reason);
		if (mapped == nullptr || reinterpret_cast<uint64_t>(mapped) != vaddr) {
			if (mapped != nullptr) {
				ReleaseView(mapped, size);
				SetFailure(failure_reason, FailureReason::WrongHostAddress);
			}
			return false;
		}
		if (protective) ProtectRemapped(vaddr, size, mode);

		{
			std::lock_guard lock(m_mutex);
			HoldTimer hold {LiveCounters::BackingHoldMapUs};
			BackingWriteFence fence(*this);
			InvalidateMappingLookup();
			EXIT_IF(!MapsEmplace(MapEntry {vaddr, size, backing_offset, mode}));
			EXIT_IF(!m_views.emplace(vaddr, size).second);
			NoteAliasesUnlocked(vaddr, size, backing_offset);
		}
		SetFailure(failure_reason, FailureReason::None);
		return true;
#else
		(void)protective;
		void* mapped = MapView(vaddr, size, backing_offset, mode, failure_reason);
		if (mapped == nullptr || reinterpret_cast<uint64_t>(mapped) != vaddr) {
			return false;
		}
		{
			std::lock_guard lock(m_mutex);
			HoldTimer hold {LiveCounters::BackingHoldMapUs};
			BackingWriteFence fence(*this);
			InvalidateMappingLookup();
			EXIT_IF(!MapsEmplace(MapEntry {vaddr, size, backing_offset, mode}));
		}
		SetFailure(failure_reason, FailureReason::None);
		return true;
#endif
	}

	// KYTY_ALIAS_STATS (default off since 10-23: it walks every mapping on each map under the lock; with ~40k mappings
	// late in a session every area change stalled the render thread, 38-56 fps instead of 75): whether the physical memory a view maps is mapped at another address too (the PS5's
	// aliases: what makes the backing a shared section, the first ingredient of bug check 0x1A/0x41202). Counted per map,
	// the first ones logged; the totals every 512 maps: whether the guest memory could be private memory instead.
	void NoteAliasesUnlocked(uint64_t vaddr, uint64_t size, uint64_t backing_offset) {
		static const bool on = [] {
			const char* value = std::getenv("KYTY_ALIAS_STATS");
			return value != nullptr && *value == '1';
		}();
		if (!on) return;
		static uint64_t maps = 0, aliased_maps = 0, aliased_bytes = 0, mapped_bytes = 0;
		uint64_t        overlap = 0;
		for (const auto& [_, entry]: m_maps) {
			if (entry.vaddr == vaddr) continue;
			const auto begin = std::max(entry.backing_offset, backing_offset);
			const auto end   = std::min(entry.backing_offset + entry.size, backing_offset + size);
			if (begin < end) {
				overlap += end - begin;
				if (aliased_maps < 16) {
					std::printf("Alias: guest 0x%llx+0x%llx (physical 0x%llx) also at guest 0x%llx (0x%llx bytes shared)\n",
					            static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
					            static_cast<unsigned long long>(backing_offset), static_cast<unsigned long long>(entry.vaddr),
					            static_cast<unsigned long long>(end - begin));
					std::fflush(stdout);
				}
			}
		}
		++maps;
		mapped_bytes += size;
		if (overlap != 0) {
			++aliased_maps;
			aliased_bytes += overlap;
		}
		if (maps % 512 == 0 || (overlap != 0 && aliased_maps <= 16)) {
			std::printf("Alias stats: %llu maps (%llu MiB), %llu aliased (%llu MiB shared with another address); %zu views now\n",
			            static_cast<unsigned long long>(maps), static_cast<unsigned long long>(mapped_bytes >> 20u),
			            static_cast<unsigned long long>(aliased_maps), static_cast<unsigned long long>(aliased_bytes >> 20u),
			            m_maps.size());
			std::fflush(stdout);
		}
	}

	// [vaddr, vaddr + size) unmapped. The ranges that became free placeholders (each one placeholder) are appended to
	// `freed`: on Windows a range left parked in its view is none, and a view released whole frees its parked ranges too.
	bool Unmap(uint64_t vaddr, uint64_t size, bool* placeholder_preserved,
	           std::vector<std::pair<uint64_t, uint64_t>>* freed) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		if (placeholder_preserved != nullptr) {
			*placeholder_preserved = false;
		}

		const auto            end = vaddr + size;
		std::vector<MapEntry> targets;
		{
			std::lock_guard lock(m_mutex);
			HoldTimer hold {LiveCounters::BackingHoldMapUs};
			auto            current = vaddr;
			while (current < end) {
				MapEntry entry {};
				if (!FindContainingUnlocked(current, 1, &entry)) {
					return false;
				}
				const auto part_size = std::min(end, entry.vaddr + entry.size) - current;
				targets.push_back(
				    {current, part_size, entry.backing_offset + current - entry.vaddr, entry.mode});
				current += part_size;
			}
		}

		std::vector<MapEntry> removed;
		removed.reserve(targets.size());
		const auto freed_before = freed->size();
		bool       parked       = false;
		for (const auto& entry: targets) {
			bool part_placeholder_preserved = false;
			bool part_parked                = false;
			if (!UnmapOne(entry.vaddr, entry.size, &part_placeholder_preserved, freed, &part_parked)) {
				// (A part parked in its view or released with it is not given back: those fail only fatally.)
				EXIT_IF(parked);
				for (auto it = removed.rbegin(); it != removed.rend(); ++it) {
					EXIT_IF(!MapExistingPlaceholderFixed(it->vaddr, it->size, it->backing_offset,
					                                     it->mode, nullptr, ProtectiveRemap()));
					NoteRemapped(it->vaddr, it->size);
				}
				freed->resize(freed_before);
				if (placeholder_preserved != nullptr) {
					*placeholder_preserved = false;
				}
				return false;
			}
			EXIT_IF(!part_placeholder_preserved);
			parked = parked || part_parked;
			removed.push_back(entry);
		}

		if (placeholder_preserved != nullptr) {
			*placeholder_preserved = true;
		}
		return true;
	}

	// The span of the views over [vaddr, vaddr + size), widening [*begin, *end) (an unmap maps the other parts of the
	// first and the last view again, or releases a view whole: GuestAddressSpace::UnmapBacking marks it busy).
	void ViewSpan(uint64_t vaddr, uint64_t size, uint64_t* begin, uint64_t* end) {
		auto lock = MapReadLock();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (const auto view = FindViewUnlocked(vaddr); view != m_views.end()) *begin = std::min(*begin, view->first);
		if (const auto view = FindViewUnlocked(vaddr + size - 1); view != m_views.end())
			*end = std::max(*end, view->first + view->second);
#else
		MapEntry entry {};
		if (FindContainingUnlocked(vaddr, 1, &entry)) *begin = std::min(*begin, entry.vaddr);
		if (FindContainingUnlocked(vaddr + size - 1, 1, &entry)) *end = std::max(*end, entry.vaddr + entry.size);
#endif
	}

	// `freed`: as Unmap's. `parked` (Windows): the range was left parked in its view, or released with it.
	bool UnmapOne(uint64_t vaddr, uint64_t size, bool* placeholder_preserved,
	              std::vector<std::pair<uint64_t, uint64_t>>* freed, bool* parked = nullptr) {
		GuestBackingMutation mutation;
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_backing_store_unmaps_before_failure == 0) {
			g_test_backing_store_unmaps_before_failure = UINT32_MAX;
			return false;
		}
		if (g_test_backing_store_unmaps_before_failure != UINT32_MAX) {
			g_test_backing_store_unmaps_before_failure--;
		}
#endif
		if (parked != nullptr) {
			*parked = false;
		}

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
		// Only the unmapped pages get a placeholder, under the lock. Mapping the
		// neighbours again would reset their protection (GPU page tracking, guest
		// mprotect) and leave a window in which their still-mapped pages have no entry.
		std::lock_guard lock(m_mutex);
		HoldTimer hold {LiveCounters::BackingHoldMapUs};
		const auto      next = m_maps.upper_bound(vaddr);
		if (next == m_maps.begin()) {
			return false;
		}
		const auto entry = std::prev(next);
		const auto old   = entry->second;
		if (vaddr + size > old.vaddr + old.size) {
			return false;
		}
		BackingWriteFence fence(*this);
		if (!UnmapViewPreservePlaceholder(reinterpret_cast<void*>(vaddr), size)) {
			return false;
		}
		InvalidateMappingLookup();
		MapsErase(entry);
		if (vaddr != old.vaddr) {
			MapsEmplace(MapEntry {old.vaddr, vaddr - old.vaddr, old.backing_offset, old.mode});
		}
		if (const auto right = vaddr + size; right != old.vaddr + old.size) {
			MapsEmplace(MapEntry {right, old.vaddr + old.size - right, old.backing_offset + (right - old.vaddr),
			                      old.mode});
		}
		freed->emplace_back(vaddr, size);
		if (placeholder_preserved != nullptr) {
			*placeholder_preserved = true;
		}
		return true;
#else
		// Whole: the range is its view. Split: the view unmapped and its other parts mapped again (a view cannot be split).
		// Park: the range left in its view (m_parked). Release: the view unmapped whole, nothing of it mapped.
		enum class How { Whole, Split, Park, Release };
		const auto park_mode = kyty_local_park_unmaps.load(std::memory_order_relaxed);
		const bool park_new  = park_mode != 0;
		MapEntry   old {};
		uint64_t   view_base = 0, view_size = 0;
		How        how = How::Whole;
		{
			std::lock_guard lock(m_mutex);
			HoldTimer hold {LiveCounters::BackingHoldMapUs};
			const auto      next = m_maps.upper_bound(vaddr);
			if (next == m_maps.begin()) {
				return false;
			}
			const auto entry = std::prev(next);
			if (vaddr + size > entry->second.vaddr + entry->second.size) {
				return false;
			}
			old             = entry->second;
			const auto view = FindViewUnlocked(old.vaddr);
			EXIT_IF(view == m_views.end() || old.vaddr + old.size > view->first + view->second);
			view_base             = view->first;
			view_size             = view->second;
			const bool whole_view = old.vaddr == view_base && old.size == view_size;
			const bool exact      = vaddr == old.vaddr && size == old.size;
			// (Native: never Split - private memory mapped again would lose the other parts' bytes -, always Park.)
			how = !whole_view ? How::Park : exact ? How::Whole : (park_new || m_native) ? How::Park : How::Split;
			BackingWriteFence fence(*this);
			InvalidateMappingLookup();
			MapsErase(entry);
			if (how != How::Park) {
				m_views.erase(view);
			} else {
				if (vaddr != old.vaddr) {
					EXIT_IF(!MapsEmplace(MapEntry {old.vaddr, vaddr - old.vaddr, old.backing_offset, old.mode}));
				}
				if (const auto right = vaddr + size; right != old.vaddr + old.size) {
					EXIT_IF(!MapsEmplace(MapEntry {right, old.vaddr + old.size - right,
					                               old.backing_offset + (right - old.vaddr), old.mode}));
				}
				const auto view_end = view_base + view_size;
				if (const auto live = m_maps.lower_bound(view_base); live == m_maps.end() || live->first >= view_end) {
					how = How::Release;
					m_parked.erase(m_parked.lower_bound(view_base), m_parked.lower_bound(view_end));
					m_views.erase(view);
				} else {
					ParkUnlocked(vaddr, size, view_base, view_end);
				}
			}
		}

		if (how == How::Park) {
			// The range keeps its protection: the guest has no mapping there (a map over it splits the view first), and the
			// emulator's reads and writes of guest memory find none either (m_maps). Made no-access, ~0.5 us per 4 KiB page
			// on the game's thread, it cost a third of what splitting the view did (kyty_local_park_unmaps 2: a stray access
			// faults, for debugging); its pages leave the working set on the trimming thread instead. (The address space
			// holds the range busy: no protection of it changes meanwhile.)
			// (Native: the parked pages decommitted - their bytes are no one's, their RAM and commit given back; a map there
			// commits them again, AdoptParkedNative -; the trim when Windows refuses.)
			// (Native: decommitted only with KYTY_NATIVE_PARK_DECOMMIT=1, NativeParkDecommit; otherwise kept committed and
			// trimmed like a view's parked pages, so a late write of the emulator's there does not fault.)
			if (m_native) NativeOpenParked(vaddr, size); // (ram4, KYTY_WRITE_WATCH: no page of a parked range stays no-access)
			if (m_native && NativeParkDecommit() && VirtualFree(reinterpret_cast<void*>(vaddr), size, MEM_DECOMMIT) != 0) {
				LiveCounters::Add(LiveCounters::ParkedUnmaps);
				if (parked != nullptr) *parked = true;
				if (placeholder_preserved != nullptr) *placeholder_preserved = true;
				return true;
			}
			// (Native: never made no-access - its pages' protection does not change in that mode -, trimmed instead.)
			if (park_mode == 2 && !m_native) {
				EXIT_IF(!ApplyProtection(reinterpret_cast<void*>(vaddr), size, VirtualMemory::Mode::NoAccess));
			} else {
				ParkedRangeTrimmer::Trim(vaddr, size);
			}
			LiveCounters::Add(LiveCounters::ParkedUnmaps);
			if (parked != nullptr) *parked = true;
			if (placeholder_preserved != nullptr) *placeholder_preserved = true;
			return true;
		}
		if (how == How::Release) {
			EXIT_IF(!ReleaseView(reinterpret_cast<void*>(view_base), view_size));
			LiveCounters::Add(LiveCounters::ParkedReleases);
			freed->emplace_back(view_base, view_size);
			if (parked != nullptr) *parked = true;
			if (placeholder_preserved != nullptr) *placeholder_preserved = true;
			return true;
		}
		if (how == How::Whole) {
			const bool unmapped = ReleaseView(reinterpret_cast<void*>(old.vaddr), old.size);
			if (!unmapped) {
				std::lock_guard lock(m_mutex);
				HoldTimer hold {LiveCounters::BackingHoldMapUs};
				BackingWriteFence fence(*this);
				InvalidateMappingLookup();
				MapsEmplace(old);
				m_views.emplace(old.vaddr, old.size);
				return false;
			}
			freed->emplace_back(vaddr, size);
			if (placeholder_preserved != nullptr) {
				*placeholder_preserved = true;
			}
			return true;
		}

		const uint64_t left_size  = vaddr - old.vaddr;
		const uint64_t right_addr = vaddr + size;
		const uint64_t old_end    = old.vaddr + old.size;
		const uint64_t right_size = old_end - right_addr;
		g_remap_window_end.store(old_end, std::memory_order_relaxed);
		g_remap_window_begin.store(old.vaddr, std::memory_order_release);

		EXIT_IF(m_native); // (native unmaps park instead)
		if (UnmapViewPreservePlaceholder(reinterpret_cast<void*>(old.vaddr), old.size)) {
			bool ok = true;
			if (left_size != 0 && left_size != old.size) {
				ok = SplitPlaceholder(old.vaddr, left_size) && ok;
			}
			if (right_size != 0) {
				ok = SplitPlaceholder(vaddr, size) && ok;
			}
			if (left_size != 0) {
				const bool mapped = MapExistingPlaceholderFixed(old.vaddr, left_size, old.backing_offset, old.mode,
				                                                nullptr, ProtectiveRemap());
				if (mapped) NoteRemapped(old.vaddr, left_size);
				ok = mapped && ok;
			}
			if (right_size != 0) {
				const bool mapped = MapExistingPlaceholderFixed(right_addr, right_size,
				                                                old.backing_offset + (right_addr - old.vaddr), old.mode,
				                                                nullptr, ProtectiveRemap());
				if (mapped) NoteRemapped(right_addr, right_size);
				ok = mapped && ok;
			}
			if (ok) {
				freed->emplace_back(vaddr, size);
				if (placeholder_preserved != nullptr) {
					*placeholder_preserved = true;
				}
				return true;
			}

			std::vector<std::pair<uint64_t, uint64_t>> scratch;
			if (left_size != 0 && Contains(old.vaddr, left_size)) {
				EXIT_IF(!UnmapOne(old.vaddr, left_size, nullptr, &scratch));
			}
			if (right_size != 0 && Contains(right_addr, right_size)) {
				EXIT_IF(!UnmapOne(right_addr, right_size, nullptr, &scratch));
			}
			EXIT_IF(VirtualFree(reinterpret_cast<void*>(old.vaddr), old.size,
			                    MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) == 0);
			EXIT_IF(!MapExistingPlaceholderFixed(old.vaddr, old.size, old.backing_offset, old.mode, nullptr,
			                                     ProtectiveRemap()));
			NoteRemapped(old.vaddr, old.size);
			return false;
		}
		std::lock_guard lock(m_mutex);
		HoldTimer hold {LiveCounters::BackingHoldMapUs};
		BackingWriteFence fence(*this);
		InvalidateMappingLookup();
		MapsEmplace(old);
		m_views.emplace(old.vaddr, old.size);
		return false;
#endif
	}

	// Whether ranges parked in views (UnmapOne) overlap [vaddr, vaddr + size); the span of their views widens
	// [*begin, *end).
	bool ParkedViews(uint64_t vaddr, uint64_t size, uint64_t* begin, uint64_t* end) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		auto lock  = MapReadLock();
		bool found = false;
		for (auto it = FirstParkedUnlocked(vaddr, size); it != m_parked.end() && it->first < vaddr + size; ++it) {
			const auto view = FindViewUnlocked(it->first);
			EXIT_IF(view == m_views.end());
			*begin = std::min(*begin, view->first);
			*end   = std::max(*end, view->first + view->second);
			found  = true;
		}
		return found;
#else
		(void)vaddr, (void)size, (void)begin, (void)end;
		return false;
#endif
	}

	// The end of the parked range holding `vaddr`, 0 when none.
	uint64_t ParkedEnd(uint64_t vaddr) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		auto       lock = MapReadLock();
		const auto it   = FirstParkedUnlocked(vaddr, 1);
		return it != m_parked.end() ? it->first + it->second : 0;
#else
		(void)vaddr;
		return 0;
#endif
	}

	// The views holding ranges parked over [vaddr, vaddr + size) split for real, before something is mapped there: each
	// unmapped whole and its mapped parts mapped again as views of their own (as a partial unmap did, protective); its
	// parked ranges, free placeholders now, are appended to `freed`. (GuestAddressSpace::MaterializeParked holds the
	// views busy.)
	void Materialize(uint64_t vaddr, uint64_t size, std::vector<std::pair<uint64_t, uint64_t>>* freed) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		GuestBackingMutation mutation;
		for (;;) {
			uint64_t                                   view_base = 0, view_size = 0;
			std::vector<MapEntry>                      mapped;
			std::vector<std::pair<uint64_t, uint64_t>> parts;
			{
				std::lock_guard lock(m_mutex);
				HoldTimer hold {LiveCounters::BackingHoldMapUs};
				const auto      parked = FirstParkedUnlocked(vaddr, size);
				if (parked == m_parked.end()) return;
				const auto view = FindViewUnlocked(parked->first);
				EXIT_IF(view == m_views.end());
				view_base           = view->first;
				view_size           = view->second;
				const auto view_end = view_base + view_size;
				BackingWriteFence fence(*this);
				InvalidateMappingLookup();
				for (auto it = m_maps.lower_bound(view_base); it != m_maps.end() && it->first < view_end;) {
					mapped.push_back(it->second);
					parts.emplace_back(it->second.vaddr, it->second.size);
					MapsErase(it++);
				}
				for (auto it = m_parked.lower_bound(view_base); it != m_parked.end() && it->first < view_end;) {
					parts.emplace_back(it->first, it->second);
					freed->emplace_back(it->first, it->second);
					it = m_parked.erase(it);
				}
				m_views.erase(view);
			}
			g_remap_window_end.store(view_base + view_size, std::memory_order_relaxed);
			g_remap_window_begin.store(view_base, std::memory_order_release);
			// (Native: the bytes of the mapped parts copied aside first - private memory released is gone - and back once
			// they are mapped again. Rare: only a map or commit over a parked range that AdoptParkedNative could not take.)
			std::vector<std::pair<MapEntry, uint8_t*>> saved;
			if (m_native) {
				for (const auto& entry: mapped) {
					auto* copy = static_cast<uint8_t*>(VirtualAlloc(nullptr, entry.size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
					EXIT_IF(copy == nullptr);
					DWORD old_protect = 0;
					// (Its pages may be no-access - the guest's or a tracker's protection -: readable for the copy.)
					{
						// (raiz3: GPU results kept for its pages go in first.)
						NativeWindowLock pending_lock;
						NativePendingBeforeProtect(entry.vaddr, entry.size, true);
						NativeShadowDrop(entry.vaddr, entry.size); // (raiz5)
					}
					EXIT_IF(VirtualProtect(reinterpret_cast<void*>(entry.vaddr), entry.size, PAGE_READONLY, &old_protect) == 0);
					std::memcpy(copy, reinterpret_cast<const void*>(entry.vaddr), entry.size);
					saved.emplace_back(entry, copy);
				}
				std::printf("Guest memory (native): view 0x%llx+0x%llx split for a map over its parked range, %zu part(s) "
				            "copied\n",
				            static_cast<unsigned long long>(view_base), static_cast<unsigned long long>(view_size),
				            saved.size());
				std::fflush(stdout);
			}
			EXIT_IF(!ReleaseView(reinterpret_cast<void*>(view_base), view_size));
			// The view's placeholder cut at the parts (they tile it), then the mapped ones mapped again.
			std::sort(parts.begin(), parts.end());
			uint64_t at = view_base;
			for (size_t i = 0; i < parts.size(); ++i) {
				EXIT_IF(parts[i].first != at);
				at += parts[i].second;
				if (i + 1 < parts.size()) EXIT_IF(!SplitPlaceholder(parts[i].first, parts[i].second));
			}
			EXIT_IF(at != view_base + view_size);
			if (m_native) {
				for (const auto& [entry, copy]: saved) {
					// (Mapped writable - entry.mode is the host mode, read-write -, filled, then protected as a view is.)
					EXIT_IF(!MapExistingPlaceholderFixed(entry.vaddr, entry.size, entry.backing_offset, entry.mode, nullptr,
					                                     false));
					std::memcpy(reinterpret_cast<void*>(entry.vaddr), copy, entry.size);
					EXIT_IF(VirtualFree(copy, 0, MEM_RELEASE) == 0);
					if (ProtectiveRemap()) ProtectRemapped(entry.vaddr, entry.size, entry.mode);
					NoteRemapped(entry.vaddr, entry.size);
				}
				LiveCounters::Add(LiveCounters::ParkedSplits);
				continue;
			}
			for (const auto& entry: mapped) {
				EXIT_IF(!MapExistingPlaceholderFixed(entry.vaddr, entry.size, entry.backing_offset, entry.mode, nullptr,
				                                     ProtectiveRemap()));
				NoteRemapped(entry.vaddr, entry.size);
			}
			LiveCounters::Add(LiveCounters::ParkedSplits);
		}
#else
		(void)vaddr, (void)size, (void)freed;
#endif
	}

	// (KYTY_NATIVE_MEMORY) A map of [vaddr, vaddr + size) inside one range parked in a native allocation: its pages
	// committed again (zeros) and the range mapped there, the allocation kept (no copy, Materialize). 1: mapped; 0: not
	// such a range (or not native), map it as usual; -1: refused (`failure_reason`). The caller holds the range busy.
	int AdoptParkedNative(uint64_t vaddr, uint64_t size, uint64_t backing_offset, VirtualMemory::Mode mode,
	                      FailureReason* failure_reason) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (!m_native || size == 0 || UINT64_MAX - vaddr < size || !RangeOk(backing_offset, size)) return 0;
		{
			auto       lock   = MapReadLock();
			const auto parked = FirstParkedUnlocked(vaddr, size);
			if (parked == m_parked.end() || parked->first > vaddr || parked->first + parked->second < vaddr + size) return 0;
		}
		if (!NativeCoverageFree(backing_offset, size, vaddr)) {
			SetFailure(failure_reason, FailureReason::NativeAliasRefused);
			return -1;
		}
		GuestBackingMutation mutation;
		void*                ptr = nullptr;
		{
			LiveTrace::SlowOsCallScope slow(LiveTrace::OsMapView, vaddr, size);
			ptr = MemGuard::RetryOs("native guest memory (parked range)", size, [&] {
				// (Parked pages kept committed - NativeParkDecommit off - hold the old mapping's bytes and protection: dropped
				// first, so the new mapping reads zeros as a decommitted range did. Decommitting reserved pages is harmless.)
				(void)VirtualFree(reinterpret_cast<void*>(vaddr), size, MEM_DECOMMIT);
				return VirtualAlloc(reinterpret_cast<void*>(vaddr), size, MEM_COMMIT, GetProtect(mode));
			});
		}
		if (ptr == nullptr) {
			SetFailure(failure_reason, FailureReason::PlaceholderMapFailed);
			return -1;
		}
		{
			std::lock_guard lock(m_mutex);
			HoldTimer         hold {LiveCounters::BackingHoldMapUs};
			BackingWriteFence fence(*this);
			InvalidateMappingLookup();
			const auto parked = FirstParkedUnlocked(vaddr, size);
			EXIT_IF(parked == m_parked.end() || parked->first > vaddr || parked->first + parked->second < vaddr + size);
			const uint64_t parked_begin = parked->first, parked_end = parked->first + parked->second;
			m_parked.erase(parked);
			if (parked_begin < vaddr) m_parked.emplace(parked_begin, vaddr - parked_begin);
			if (vaddr + size < parked_end) m_parked.emplace(vaddr + size, parked_end - (vaddr + size));
			EXIT_IF(!MapsEmplace(MapEntry {vaddr, size, backing_offset, mode}));
		}
		SetFailure(failure_reason, FailureReason::None);
		return 1;
#else
		(void)vaddr, (void)size, (void)backing_offset, (void)mode, (void)failure_reason;
		return 0;
#endif
	}

	bool Contains(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		auto lock = MapReadLock();
		const auto      end     = vaddr + size;
		auto            current = vaddr;

		while (current < end) {
			MapEntry entry {};
			if (!FindContainingUnlocked(current, 1, &entry)) {
				return false;
			}
			const auto entry_end = entry.vaddr + entry.size;
			if (entry_end <= current) {
				return false;
			}
			current = std::min(end, entry_end);
		}

		return true;
	}

	// The other virtual ranges mapping bytes of [vaddr, vaddr + size) (the aliases), as {vaddr, size} pairs within
	// the range's physical span; false: the range is not mapped by one entry. Used on a range IsUniqueRange refused.
	bool Aliases(uint64_t vaddr, uint64_t size, std::vector<std::pair<uint64_t, uint64_t>>* aliases) {
		aliases->clear();
		if (size == 0 || UINT64_MAX - vaddr < size) return false;
		auto     lock = MapReadLock();
		MapEntry owner {};
		if (!FindContainingUnlocked(vaddr, size, &owner)) return false;
		const auto physical_begin = owner.backing_offset + (vaddr - owner.vaddr), physical_end = physical_begin + size;
		for (const auto& [at, entry]: m_maps) {
			if (at == owner.vaddr) continue;
			const auto begin = std::max(physical_begin, entry.backing_offset);
			const auto end   = std::min(physical_end, entry.backing_offset + entry.size);
			if (begin < end) aliases->emplace_back(entry.vaddr + (begin - entry.backing_offset), end - begin);
		}
		return true;
	}

	// The same over consecutive mappings (the game maps its heaps a few hundred KiB at a time: a 4 MiB copy of its
	// constants spans ten): unique when each mapping's part of the range is (no byte of it mapped twice), and no
	// address of it is unmapped. (Its callers read and write the range through its guest addresses only.)
	bool IsUniqueRangeAcrossMappings(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size) return false;
		auto           lock = MapReadLock();
		const uint64_t end  = vaddr + size;
		for (uint64_t at = vaddr; at < end;) {
			// (Each mapping through this thread's translations, held under the lock: a search of m_maps per mapping, ten
			// of them for the 4 MiB copies, was 0.9% of the render thread at Latria.)
			auto&      cached = HostReadTranslationOf(at);
			const auto epoch  = RegionEpoch(at);
			MapEntry   entry {};
			if (cached.epoch == epoch && GuestFreeRangeContains(cached.entry.vaddr, cached.entry.size, at, 1)) {
				entry = cached.entry;
			} else if (FindContainingUnlocked(at, 1, &entry)) {
				cached = {entry, epoch};
			} else {
				return false;
			}
			const uint64_t piece    = std::min(end, entry.vaddr + entry.size) - at;
			const uint64_t physical = entry.backing_offset + at - entry.vaddr;
			if (!CoverageUnique(physical, physical + piece)) return false;
			at += piece;
		}
		return true;
	}

	bool IsUniqueRange(uint64_t vaddr, uint64_t size) {
		// KYTY_ALIAS_VERIFY: every answer is compared with the full rebuild it replaces.
		static const bool verify = std::getenv("KYTY_ALIAS_VERIFY") != nullptr;
		if (!verify) {
			if (size == 0 || UINT64_MAX - vaddr < size) return false; // (as FindContainingUnlocked)
			auto lock = MapReadLock();
			// The mapping over each 256 KiB of address and whether no byte of it is mapped twice (the usual case), kept
			// while its region's mappings (RegionEpoch) and the aliases anywhere (m_alias_epoch) stay the same: any range in
			// such a mapping is unique. The game asks about ~1000 ranges a frame (and every write fault one), most of them
			// new (ring buffers), in a few mappings: two cold tree lookups each. (By 4 MiB and one epoch for every mapping
			// change, the game's ever finer mappings evicted each other and its streaming invalidated them all.)
			struct Owner {
				const GuestBackingStore* store;
				uint64_t                 epoch, alias_epoch;
				MapEntry                 entry;
				bool                     unique;
			};
			thread_local std::array<Owner, 1024> owners {};
			auto&      owner = owners[(vaddr >> 18u) % owners.size()];
			const auto epoch = RegionEpoch(vaddr);
			if (owner.store != this || owner.epoch != epoch || owner.alias_epoch != m_alias_epoch ||
			    !GuestFreeRangeContains(owner.entry.vaddr, owner.entry.size, vaddr, size)) {
				MapEntry entry {};
				if (!FindContainingUnlocked(vaddr, size, &entry)) return false;
				owner = {this, epoch, m_alias_epoch, entry, CoverageUnique(entry.backing_offset, entry.backing_offset + entry.size)};
			}
			if (owner.unique) return true;
			const auto physical = owner.entry.backing_offset + vaddr - owner.entry.vaddr;
			return CoverageUnique(physical, physical + size);
		}
		std::lock_guard lock(m_mutex); // the full rebuild writes its intervals
		const bool unique = IsUniqueRangeUnlocked(vaddr, size);
		if (unique != IsUniqueRangeRebuilt(vaddr, size)) {
			EXIT("alias coverage disagrees with the rebuilt intervals: vaddr=0x%016" PRIx64
			     " size=0x%016" PRIx64 " coverage=%d mappings=%zu segments=%zu\n",
			     vaddr, size, unique ? 1 : 0, m_maps.size(), m_coverage.size());
		}
		return unique;
	}

private:
	bool IsUniqueRangeUnlocked(uint64_t vaddr, uint64_t size) const {
		MapEntry owner {};
		if (!FindContainingUnlocked(vaddr, size, &owner)) return false;
		const auto physical = owner.backing_offset + vaddr - owner.vaddr;
		return CoverageUnique(physical, physical + size);
	}

	// The previous answer: the alias intervals of every mapping, rebuilt after each change.
	bool IsUniqueRangeRebuilt(uint64_t vaddr, uint64_t size) {
		MapEntry owner {};
		if (!FindContainingUnlocked(vaddr, size, &owner)) return false;
		if (!m_alias_ranges_valid) {
			// Diagnostic (live measure: alias_rebuilds, alias_rebuild_us, alias_maps).
			HoldTimer rebuild_time {LiveCounters::AliasRebuildUs};
			LiveCounters::Add(LiveCounters::AliasRebuilds);
			LiveCounters::Add(LiveCounters::AliasMaps, m_maps.size());
			// Build physical intervals shared by at least two guest mappings.
			// An alias elsewhere in a large allocation does not affect this range.
			std::vector<std::pair<uint64_t, int>> endpoints;
			endpoints.reserve(m_maps.size() * 2);
			for (const auto& [_, entry] : m_maps) {
				endpoints.emplace_back(entry.backing_offset, 1);
				endpoints.emplace_back(entry.backing_offset + entry.size, -1);
			}
			std::sort(endpoints.begin(), endpoints.end());
			m_aliased_physical_ranges.clear();
			uint64_t previous = 0;
			int active = 0;
			for (size_t i = 0; i < endpoints.size();) {
				const auto position = endpoints[i].first;
				if (active > 1 && previous < position) {
					if (!m_aliased_physical_ranges.empty() &&
					    m_aliased_physical_ranges.back().second == previous)
						m_aliased_physical_ranges.back().second = position;
					else m_aliased_physical_ranges.emplace_back(previous, position);
				}
				do { active += endpoints[i++].second; }
				while (i < endpoints.size() && endpoints[i].first == position);
				previous = position;
			}
			EXIT_IF(active != 0);
			m_alias_ranges_valid = true;
		}
		const auto physical = owner.backing_offset + vaddr - owner.vaddr;
		const auto found = std::lower_bound(m_aliased_physical_ranges.begin(),
		    m_aliased_physical_ranges.end(), physical,
		    [](const auto& interval, uint64_t value) { return interval.second <= value; });
		return found == m_aliased_physical_ranges.end() || physical + size <= found->first;
	}

private:
	// One writer at a time of each kind: data writers hold m_data_mutex (and m_mutex shared), CPU-owned
	// writers m_cpu_write_mutex (bytes no data writer writes: TryWriteCpuBacking), zeroing and
	// mapping mutations hold m_mutex exclusively, which excludes both.
	// A writer announces itself before inspecting the reader slots. Sequential
	// consistency makes it impossible for both entrants to miss the other: either
	// the reader falls back, or the writer waits for its bounded host copy to end.
	// No GPU work, guest faults, or other locks occur inside that copy transaction.
	class BackingWriteFence {
	public:
		KYTY_CLASS_NO_COPY(BackingWriteFence);
		explicit BackingWriteFence(GuestBackingStore& store): m_store(store) {
			m_store.m_host_write_pending.store(true, std::memory_order_seq_cst);
			unsigned spins = 0;
			for (const auto& slot: m_store.m_host_read_slots) {
				while (slot.pages.load(std::memory_order_seq_cst) != 0) {
					_mm_pause();
					if (++spins == 256) { std::this_thread::yield(); spins = 0; }
				}
			}
		}
		~BackingWriteFence() {
			m_store.m_host_write_pending.store(false, std::memory_order_release);
		}
	private:
		GuestBackingStore& m_store;
	};

	// A data write (guest readback results, TryWriteBacking) into backing bytes [begin, end) that
	// one mapping covers: it waits only for a reader of overlapping bytes, and a reader falls back
	// only when its bytes overlap (BackingReadTransaction::Narrow). Before, every small render-
	// thread read fell back to the data mutex, and waited there for the whole unrelated copy
	// (about 11 times and 0.35 ms a frame on the walk). Both sides publish their range before
	// they inspect the other's, so overlapping bytes are never read and written at once.
	// Backing byte ranges as one word: first page << 32 | end page (4 KiB pages; 0: none).
	static constexpr uint64_t AllPages = 0xffffffffull;
	static uint64_t PackPages(uint64_t begin, uint64_t end) {
		const auto first = std::min<uint64_t>(begin >> 12u, 0xfffffffeull);
		const auto last  = std::min<uint64_t>((end + 4095u) >> 12u, 0xffffffffull);
		return first << 32u | std::max(last, first + 1);
	}
	static bool PagesOverlap(uint64_t a, uint64_t b) {
		return a != 0 && b != 0 && (a >> 32u) < (b & 0xffffffffull) && (b >> 32u) < (a & 0xffffffffull);
	}

	class DataWriteFence {
	public:
		KYTY_CLASS_NO_COPY(DataWriteFence);
		// `published`: m_data_write_pages, or m_cpu_write_pages for a CPU-owned write (TryWriteCpuBacking).
		DataWriteFence(GuestBackingStore& store, uint64_t begin, uint64_t end, std::atomic<uint64_t>& published)
		    : m_published(published) {
			const auto pages = PackPages(begin, end);
			m_published.store(pages, std::memory_order_seq_cst);
			unsigned spins = 0;
			for (const auto& slot: store.m_host_read_slots) {
				while (PagesOverlap(slot.pages.load(std::memory_order_seq_cst), pages)) {
					_mm_pause();
					if (++spins == 256) { std::this_thread::yield(); spins = 0; }
				}
			}
		}
		~DataWriteFence() { m_published.store(0, std::memory_order_release); }
	private:
		std::atomic<uint64_t>& m_published;
	};

	class BackingReadTransaction {
	public:
		KYTY_CLASS_NO_COPY(BackingReadTransaction);
		explicit BackingReadTransaction(GuestBackingStore& store): m_store(store) {
			// A reader per slot (this thread's first, then the next free one); it holds every page until Narrow
			// names the ones it copies.
			if (m_store.m_host_write_pending.load(std::memory_order_relaxed)) return;
			thread_local const size_t first = std::hash<std::thread::id> {}(std::this_thread::get_id());
			for (size_t i = 0; i < HostReadSlots && m_slot == nullptr; ++i) {
				auto&    slot = m_store.m_host_read_slots[(first + i) % HostReadSlots].pages;
				uint64_t idle = 0;
				if (slot.load(std::memory_order_relaxed) == 0 &&
				    slot.compare_exchange_strong(idle, AllPages, std::memory_order_seq_cst))
					m_slot = &slot;
			}
			if (m_slot != nullptr && m_store.m_host_write_pending.load(std::memory_order_seq_cst)) Release();
		}
		~BackingReadTransaction() { Release(); }
		bool Acquired() const { return m_slot != nullptr; }
		// The read copies backing bytes [begin, end): false (and released) while a data or CPU-owned write
		// into overlapping pages is under way.
		bool Narrow(uint64_t begin, uint64_t end) {
			const auto pages = PackPages(begin, end);
			m_slot->store(pages, std::memory_order_seq_cst);
			if (PagesOverlap(m_store.m_data_write_pages.load(std::memory_order_seq_cst), pages) ||
			    PagesOverlap(m_store.m_cpu_write_pages.load(std::memory_order_seq_cst), pages)) {
				Release();
				return false;
			}
			return true;
		}
	private:
		void Release() {
			if (m_slot != nullptr) {
				m_slot->store(0, std::memory_order_release);
				m_slot = nullptr;
			}
		}
		GuestBackingStore&     m_store;
		std::atomic<uint64_t>* m_slot = nullptr;
	};

	template <bool Write, typename Transfer>
	bool TryTransferBacking(uint64_t vaddr, uint64_t size, Transfer transfer) {
		if (!IsAvailable() || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		auto lock      = MapReadLock();
		auto data_lock = DataLock(Write);
		// (A reader excludes the CPU-owned writers too; a data writer's bytes are never theirs.)
		std::unique_lock cpu_lock(m_cpu_write_mutex, std::defer_lock);
		if constexpr (!Write) cpu_lock.lock();
		if constexpr (Write) {
			HoldTimer hold {LiveCounters::BackingHoldWriteUs};
			// The shared map lock keeps the mapping, so its backing bytes, fixed.
			MapEntry entry {};
			if (FindContainingUnlocked(vaddr, size, &entry)) {
				const auto      begin = entry.backing_offset + vaddr - entry.vaddr;
				DataWriteFence fence(*this, begin, begin + size, m_data_write_pages);
				return TransferBackingUnlocked(vaddr, size, transfer, true); // (ww2: a write)
			}
			// A range over several mappings (validated whole first: all or nothing) is written a mapping's piece at a
			// time, each under the fence of its own pages. The fence that excludes every reader made each lock-free
			// read of other bytes (the render thread's table copies) fall back to the data mutex and wait there for the
			// whole copy (~160 us). A reader that spans mappings takes that mutex anyway, so it still sees all or none.
			LiveCounters::Add(LiveCounters::BackingSpanWrites);
			const auto end = vaddr + size;
			for (auto current = vaddr; current < end;) {
				MapEntry piece {};
				if (!FindContainingUnlocked(current, 1, &piece) ||
				    !RangeOk(piece.backing_offset + current - piece.vaddr, std::min(end, piece.vaddr + piece.size) - current))
					return false;
				current = std::min(end, piece.vaddr + piece.size);
			}
			for (auto current = vaddr; current < end;) {
				MapEntry piece {};
				EXIT_IF(!FindContainingUnlocked(current, 1, &piece));
				const auto     bytes = std::min(end, piece.vaddr + piece.size) - current;
				const auto     begin = piece.backing_offset + current - piece.vaddr;
				DataWriteFence fence(*this, begin, begin + bytes, m_data_write_pages);
				uint8_t* host = HostAt(piece, current);
				// (raiz3: NativeGuardedWrite - a write-back of GPU results over no-access pages is kept, lazy -.)
				NativeGuardedWrite(host, bytes, [&](uint8_t* to) { transfer(to, current - vaddr, static_cast<size_t>(bytes)); });
				current += bytes;
			}
			return true;
		} else {
			return TransferBackingUnlocked(vaddr, size, transfer);
		}
	}

	// (ww2) `writes`: the transfer writes the backing (NativeGuardedAccess).
	template <typename Transfer>
	bool TransferBackingUnlocked(uint64_t vaddr, uint64_t size, Transfer transfer, bool writes = false) {
		const auto      end     = vaddr + size;
		if (kyty_local_preparation_lookup_mode.load(std::memory_order_relaxed)) {
			MapEntry entry{};
			if (FindContainingUnlocked(vaddr, size, &entry)) {
				const auto offset = entry.backing_offset + vaddr - entry.vaddr;
				if (!RangeOk(offset, size)) return false;
				// The lock keeps this whole mapping valid through the transfer.
				// Spanning requests retain the original validate-all-then-copy path.
				uint8_t* host = HostAt(entry, vaddr);
				// (raiz3: a write through NativeGuardedWrite, lazy write-back.)
				if (writes)
					NativeGuardedWrite(host, size, [&](uint8_t* to) { transfer(to, 0, static_cast<size_t>(size)); });
				else
					NativeGuardedReadFrom(host, size, [&](const uint8_t* from) { transfer(const_cast<uint8_t*>(from), 0, static_cast<size_t>(size)); }); // (raiz5)
				return true;
			}
		}
		auto            current = vaddr;
		while (current < end) {
			MapEntry entry {};
			if (!FindContainingUnlocked(current, 1, &entry)) {
				return false;
			}
			const auto bytes = std::min(end, entry.vaddr + entry.size) - current;
			if (!RangeOk(entry.backing_offset + current - entry.vaddr, bytes)) {
				return false;
			}
			current += bytes;
		}

		current = vaddr;
		while (current < end) {
			MapEntry entry {};
			EXIT_IF(!FindContainingUnlocked(current, 1, &entry));
			const auto bytes  = std::min(end, entry.vaddr + entry.size) - current;
			uint8_t* host = HostAt(entry, current);
			if (writes) // (raiz3: NativeGuardedWrite, lazy write-back)
				NativeGuardedWrite(host, bytes, [&](uint8_t* to) { transfer(to, current - vaddr, static_cast<size_t>(bytes)); });
			else
				NativeGuardedReadFrom(host, bytes, [&](const uint8_t* from) { transfer(const_cast<uint8_t*>(from), current - vaddr, static_cast<size_t>(bytes)); }); // (raiz5)
			current += bytes;
		}
		return true;
	}

	struct MapEntry {
		uint64_t            vaddr          = 0;
		uint64_t            size           = 0;
		uint64_t            backing_offset = 0;
		VirtualMemory::Mode mode           = VirtualMemory::Mode::NoAccess;
	};

	// The coverage key at `at` (inserted with the count of the segment it splits).
	std::map<uint64_t, uint32_t>::iterator CoverageSplit(uint64_t at) {
		auto next = m_coverage.lower_bound(at);
		if (next != m_coverage.end() && next->first == at) return next;
		const uint32_t count = next == m_coverage.begin() ? 0 : std::prev(next)->second;
		return m_coverage.emplace_hint(next, at, count);
	}

	void AddCoverage(uint64_t begin, uint64_t end, int delta) {
		if (begin >= end) return;
		auto first = CoverageSplit(begin);
		auto last  = CoverageSplit(end);
		for (auto it = first; it != last; ++it) {
			EXIT_IF(delta < 0 && it->second == 0);
			const bool aliased = it->second > 1;
			it->second         = static_cast<uint32_t>(static_cast<int>(it->second) + delta);
			if ((it->second > 1) != aliased) ++m_alias_epoch; // (a byte gained or lost its second mapping)
		}
		// Merge keys that no longer start a different count.
		for (auto it: {first, last}) {
			if (it == m_coverage.end()) continue;
			const uint32_t before = it == m_coverage.begin() ? 0 : std::prev(it)->second;
			if (it->second == before) m_coverage.erase(it);
		}
	}

	// (ram11) The game's mappings, for the cold store's thread (its shared lock: a snapshot).
	static void ColdRanges(void* self, std::vector<std::pair<uint64_t, uint64_t>>* out) {
		auto*            store = static_cast<GuestBackingStore*>(self);
		std::shared_lock lock(store->m_mutex);
		for (const auto& [key, entry]: store->m_maps) {
			(void)key;
			out->emplace_back(entry.vaddr, entry.vaddr + entry.size);
		}
	}

	// Every m_maps change goes through these two (they keep m_coverage and the region epochs in step).
	bool MapsEmplace(const MapEntry& entry) {
		if (!m_maps.emplace(entry.vaddr, entry).second) return false;
		AddCoverage(entry.backing_offset, entry.backing_offset + entry.size, 1);
		TouchRegions(entry.vaddr, entry.size);
		return true;
	}
	void MapsErase(std::map<uint64_t, MapEntry>::iterator entry) {
		AddCoverage(entry->second.backing_offset, entry->second.backing_offset + entry->second.size, -1);
		TouchRegions(entry->second.vaddr, entry->second.size);
		m_maps.erase(entry);
	}

	// The mappings' change count of each 64 MiB of guest address (MapsEmplace and MapsErase move the ones a mapping
	// spans; under the mapping mutation, which readers' transactions exclude), each region's counts its own (its index in
	// the high bits): a translation cached for an address holds while its region's count is the same, whatever was mapped
	// or unmapped elsewhere. (With one count for all mappings, the game's streaming invalidated every cached translation
	// several times a frame.) Past the regions' 2 TiB, the global epoch.
	static constexpr uint64_t RegionEpochBits = 26;
	static constexpr uint64_t RegionEpochCount = uint64_t {1} << (41u - RegionEpochBits);
	std::unique_ptr<std::atomic<uint64_t>[]> m_region_epochs = [] {
		auto epochs = std::make_unique<std::atomic<uint64_t>[]>(RegionEpochCount);
		for (uint64_t region = 0; region < RegionEpochCount; ++region)
			epochs[region].store((region + 1) << 32u, std::memory_order_relaxed);
		return epochs;
	}();
	void TouchRegions(uint64_t vaddr, uint64_t size) {
		if (size == 0) return;
		for (uint64_t region = vaddr >> RegionEpochBits, last = (vaddr + size - 1) >> RegionEpochBits;
		     region <= last && region < RegionEpochCount; ++region)
			m_region_epochs[region].fetch_add(1, std::memory_order_relaxed);
	}
	[[nodiscard]] uint64_t RegionEpoch(uint64_t vaddr) const {
		const auto region = vaddr >> RegionEpochBits;
		return region < RegionEpochCount ? m_region_epochs[region].load(std::memory_order_relaxed) : m_mapping_epoch;
	}

	// True when no physical byte of [begin, end) is mapped by more than one guest mapping.
	bool CoverageUnique(uint64_t begin, uint64_t end) const {
		auto it = m_coverage.upper_bound(begin);
		if (it != m_coverage.begin()) --it;
		for (; it != m_coverage.end() && it->first < end; ++it) {
			const auto next = std::next(it);
			if (next != m_coverage.end() && next->first <= begin) continue;
			if (it->second > 1) return false;
		}
		return true;
	}

	void InvalidateMappingLookup() {
		m_alias_ranges_valid = false;
		if (++m_mapping_epoch == 0) m_mapping_epoch = 1;
	}

	struct HostReadTranslation { MapEntry entry; uint64_t epoch = 0; };
	// The host read translation of `vaddr`'s 256 KiB granule, this thread's (readers of different slots fill them at
	// once): 4096 cover 1 GiB without two granules sharing one. An entry holds while its epoch is its region's
	// (RegionEpoch, read under a read slot). (By 4 MiB granule, the mappings of the game's heaps - 43 to 208 of 192-512
	// KiB in each of its 120 MiB heaps, more as its direct memory gets cut finer - evicted each other, and their reads
	// searched m_maps.)
	HostReadTranslation& HostReadTranslationOf(uint64_t vaddr) {
		struct Translations {
			const GuestBackingStore*               store = nullptr;
			std::array<HostReadTranslation, 4096> entries {};
		};
		thread_local Translations translations;
		if (translations.store != this) {
			translations.store = this;
			translations.entries.fill({});
		}
		return translations.entries[(vaddr >> 18) % translations.entries.size()];
	}

	bool FindContainingUnlocked(uint64_t vaddr, uint64_t size, MapEntry* out) const {
		EXIT_IF(out == nullptr);

		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		auto next = m_maps.upper_bound(vaddr);
		if (next == m_maps.begin()) {
			return false;
		}
		const auto& entry = std::prev(next)->second;
		if (vaddr + size <= entry.vaddr + entry.size) {
			*out = entry;
			return true;
		}

		return false;
	}

	[[nodiscard]] bool RangeOk(uint64_t backing_offset, uint64_t size) const {
		return size != 0 && backing_offset < m_size && size <= m_size - backing_offset;
	}

	static void SetFailure(FailureReason* out, FailureReason reason) {
		if (out != nullptr) {
			*out = reason;
		}
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static DWORD GetProtect(VirtualMemory::Mode mode) {
		switch (mode) {
			case VirtualMemory::Mode::Read: return PAGE_READONLY;
			case VirtualMemory::Mode::Write:
			case VirtualMemory::Mode::ReadWrite: return PAGE_READWRITE;
			case VirtualMemory::Mode::Execute: return PAGE_EXECUTE;
			case VirtualMemory::Mode::ExecuteRead: return PAGE_EXECUTE_READ;
			case VirtualMemory::Mode::ExecuteWrite:
			case VirtualMemory::Mode::ExecuteReadWrite: return PAGE_EXECUTE_READWRITE;
			case VirtualMemory::Mode::NoAccess:
			default: return PAGE_NOACCESS;
		}
	}

	static bool ApplyProtection(void* ptr, uint64_t size, VirtualMemory::Mode mode) {
		LiveTrace::SlowOsCallScope slow(LiveTrace::OsProtect, reinterpret_cast<uint64_t>(ptr), size);
		DWORD old_protect = 0;
		// (Lazy commit: a reserved page cannot be protected; its granule is committed first.)
		return GuestLazyCommit::Protect(ptr, size, GetProtect(mode), &old_protect);
	}

	static bool SplitPlaceholder(uint64_t vaddr, uint64_t size) {
		if (size == 0) {
			return true;
		}
		LiveTrace::SlowOsCallScope slow(LiveTrace::OsFree, vaddr, size);
		return VirtualFree(reinterpret_cast<void*>(vaddr), size,
		                   MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) != 0;
	}

	// A view mapped again (MapExistingPlaceholderFixed, `protective`) gets the GPU trackers' protection of the pages a
	// watcher holds at once. The views are busy (UnmapBacking), and every protection change of the trackers over them
	// waits for that: their state is read without their regions' locks, and a change made meanwhile is applied after
	// this. Given only after the lock (ReapplyRemappedViews), a CPU write to a watched page in between went
	// unseen. (The view keeps its own mode until here, a few system calls: one mapped read-only cannot be made writable
	// later, and made read-only at once, the game's own writes to its unwatched pages faulted by the thousands.)
	static void ProtectRemapped(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) return;
		thread_local std::vector<Graphics::PageManager::WatchedRun> runs;
		runs.clear();
		g_gpu_resources->WatchedRuns(vaddr, size, &runs);
		for (const auto& run: runs) {
			const auto begin = std::max(run.address, vaddr);
			const auto end   = std::min(run.address + run.size, vaddr + size);
			if (begin < end && run.mode != mode)
				EXIT_IF(!ApplyProtection(reinterpret_cast<void*>(begin), end - begin, run.mode));
		}
	}

	// KYTY_NATIVE_MEMORY: the placeholder [vaddr, vaddr + size) replaced by private memory, committed and write-watched
	// (MEM_WRITE_WATCH; where Windows refuses it with a placeholder, without: NativeWriteWatch then fails for it, said once).
	void* ReplacePlaceholderNative(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode, FailureReason* failure_reason) {
		const DWORD protect = GetProtect(mode);
		void*       ptr     = nullptr;
		// (commit-al-tocar, KYTY_COMMIT_AL_TOCAR=1: reserved only, committed at the first touch - NativeLazyCommit.)
		const bool  lazy    = NativeLazyCommit::Start();
		const DWORD commit  = lazy ? 0u : static_cast<DWORD>(MEM_COMMIT);
		if (lazy) NativeLazyCommit::Register(vaddr, size);
		{
			LiveTrace::SlowOsCallScope slow(LiveTrace::OsMapView, vaddr, size);
			ptr = MemGuard::RetryOs("native guest memory", size, [&] {
				return VirtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(vaddr), size,
				                     MEM_RESERVE | commit | MEM_REPLACE_PLACEHOLDER | MEM_WRITE_WATCH, protect, nullptr, 0);
			});
			if (ptr == nullptr && GetLastError() == ERROR_INVALID_PARAMETER) {
				static std::atomic_bool said {false};
				if (!said.exchange(true))
					std::printf("Guest memory (native): MEM_WRITE_WATCH refused over a placeholder (0x57): mapped without "
					            "it, NativeWriteWatch reports nothing for such mappings\n");
				std::fflush(stdout);
				ptr = MemGuard::RetryOs("native guest memory", size, [&] {
					return VirtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(vaddr), size,
					                     MEM_RESERVE | commit | MEM_REPLACE_PLACEHOLDER, protect, nullptr, 0);
				});
			}
		}
		if (ptr == nullptr || reinterpret_cast<uint64_t>(ptr) != vaddr) {
			if (lazy) NativeLazyCommit::Unregister(vaddr, size);
			LOGF_COLOR(Log::Color::Red, "\t native guest memory: 0x%016" PRIx64 "+0x%" PRIx64 " failed: 0x%08" PRIx32 "\n", vaddr,
			           size, static_cast<uint32_t>(GetLastError()));
			if (ptr != nullptr) VirtualFree(ptr, 0, MEM_RELEASE);
			SetFailure(failure_reason, FailureReason::PlaceholderMapFailed);
			return nullptr;
		}
		MapaGrande::Note(vaddr, size); // (KYTY_MAPA_GRANDE)
		return ptr;
	}

	// A guest view or native allocation [vaddr, vaddr + size) given back as a placeholder.
	bool ReleaseView(void* vaddr, uint64_t size) {
		if (!m_native) return UnmapViewPreservePlaceholder(vaddr, size);
		NativeLazyCommit::Unregister(reinterpret_cast<uint64_t>(vaddr), size); // (commit-al-tocar)
		LiveTrace::SlowOsCallScope slow(LiveTrace::OsUnmapView, reinterpret_cast<uint64_t>(vaddr), size);
		return VirtualFree(vaddr, size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) != 0;
	}

	void* ReplacePlaceholderView(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                             VirtualMemory::Mode mode, FailureReason* failure_reason) {
		if (m_native) return ReplacePlaceholderNative(vaddr, size, mode, failure_reason);
		const DWORD map_protect =
		    mode == VirtualMemory::Mode::NoAccess ? PAGE_READWRITE : GetProtect(mode);
		void* ptr = nullptr;
		// (Lazy commit: the view known before it is mapped, so a fault on it finds the protections recorded for its pages.)
		GuestLazyCommit::Register(vaddr, size, backing_offset);
		// Committing through the writable alias commits the section's pages for every view of them.
		// (Refused for lack of memory: a release pass and another try, MemGuard.)
		if (m_on_demand && !m_lazy &&
		    MemGuard::RetryOs("direct-memory commit", size, [&] {
			    return VirtualAlloc(m_backing_base + backing_offset, size, MEM_COMMIT, PAGE_READWRITE);
		    }) == nullptr) {
			LOGF_COLOR(Log::Color::Red, "\t direct-memory backing: commit of %" PRIu64 " bytes failed: 0x%08" PRIx32 "\n",
			           size, static_cast<uint32_t>(GetLastError()));
			GuestLazyCommit::Unregister(vaddr);
			SetFailure(failure_reason, FailureReason::PlaceholderMapFailed);
			return nullptr;
		}
		{
			LiveTrace::SlowOsCallScope slow(LiveTrace::OsMapView, vaddr, size);
			ptr = MemGuard::RetryOs("MapViewOfFile3", size, [&] {
				return MapViewOfFile3(m_handle, GetCurrentProcess(), reinterpret_cast<void*>(vaddr), backing_offset, size,
				                      MEM_REPLACE_PLACEHOLDER, map_protect, nullptr, 0);
			});
		}
		if (ptr == nullptr || reinterpret_cast<uint64_t>(ptr) != vaddr) {
			if (ptr != nullptr) {
				UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);
			}
			GuestLazyCommit::Unregister(vaddr);
			SetFailure(failure_reason, FailureReason::PlaceholderMapFailed);
			return nullptr;
		}

		// (Mapped with its protection, but for no access, which a view is not mapped with. A protection change of the whole
		// view right after it is mapped is part of the stand-alone reproducer of bug check 0x1A/0x41202 (issues #5, #50: a
		// view mapped again while protections change and pages are paged), and changed nothing.)
		if (mode == VirtualMemory::Mode::NoAccess && !ApplyProtection(ptr, size, mode)) {
			UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);
			GuestLazyCommit::Unregister(vaddr);
			SetFailure(failure_reason, FailureReason::ProtectFailed);
			return nullptr;
		}
		// (A view mapped again comes back without the trackers' protection until it is applied: a write meanwhile goes
		// unseen, so its write-watched never-written granules count as written.)
		FreshMemory::MarkArmedBacking(GetBackingBase() + backing_offset, size);
		return ptr;
	}

	static bool UnmapViewPreservePlaceholder(void* vaddr, uint64_t size) {
		bool unmapped = false;
		{
			LiveTrace::SlowOsCallScope slow(LiveTrace::OsUnmapView, reinterpret_cast<uint64_t>(vaddr), size);
			unmapped = UnmapViewOfFile2(GetCurrentProcess(), vaddr, MEM_PRESERVE_PLACEHOLDER) != 0;
		}
		// (Lazy commit: the view and the protections recorded for it forgotten once it is gone.)
		if (unmapped) GuestLazyCommit::Unregister(reinterpret_cast<uint64_t>(vaddr));
		return unmapped;
	}

	HANDLE m_handle = nullptr;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	[[nodiscard]] static uint64_t GetMappableTestSize() { return 0x4000; }

	static int GetProtect(VirtualMemory::Mode mode) {
		switch (mode) {
			case VirtualMemory::Mode::Read: return PROT_READ;
			case VirtualMemory::Mode::Write:
			case VirtualMemory::Mode::ReadWrite: return PROT_READ | PROT_WRITE;
			case VirtualMemory::Mode::Execute: return PROT_EXEC;
			case VirtualMemory::Mode::ExecuteRead: return PROT_EXEC | PROT_READ;
			case VirtualMemory::Mode::ExecuteWrite:
			case VirtualMemory::Mode::ExecuteReadWrite: return PROT_EXEC | PROT_READ | PROT_WRITE;
			case VirtualMemory::Mode::NoAccess:
			default: return PROT_NONE;
		}
	}

	void* MapView(uint64_t vaddr, uint64_t size, uint64_t backing_offset, VirtualMemory::Mode mode,
	              FailureReason* failure_reason) {
		void* ptr = mmap(reinterpret_cast<void*>(vaddr), size, GetProtect(mode),
		                 MAP_SHARED | MAP_FIXED, m_fd, static_cast<off_t>(backing_offset));
		if (ptr == MAP_FAILED) {
			SetFailure(failure_reason, FailureReason::FixedMapViewFailed);
			return nullptr;
		}
		return ptr;
	}

	// Replace the file view atomically instead of exposing an unmapped hole in the
	// process-lifetime guest reservation.
	static bool UnmapViewPreservePlaceholder(void* vaddr, uint64_t size) {
		void* ptr = mmap(vaddr, size, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
		return ptr != MAP_FAILED && ptr == vaddr;
	}

	int m_fd = -1;
#endif

	static constexpr uint64_t UsedGranuleBits = 16; // 64 KiB
	[[nodiscard]] bool IsUsedGranule(uint64_t granule) const {
		const uint64_t word = granule >> 6;
		return word < m_used.size() && (m_used[word] >> (granule & 63)) & 1;
	}
	void NoteUsedUnlocked(uint64_t backing_offset, uint64_t size) {
		if (size == 0) return;
		const uint64_t last = (backing_offset + size - 1) >> UsedGranuleBits;
		if (m_used.size() <= (last >> 6)) m_used.resize((last >> 6) + 1, 0);
		uint64_t first_covered = 0;
		for (uint64_t granule = backing_offset >> UsedGranuleBits; granule <= last; ++granule) {
			const uint64_t bit = uint64_t {1} << (granule & 63);
			if ((m_used[granule >> 6] & bit) == 0) ++first_covered;
			m_used[granule >> 6] |= bit;
		}
		if (first_covered != 0) GuestLazyCommit::NoteMapped(first_covered << UsedGranuleBits);
	}
	// Zeroes [offset, end) of the backing (lock held), but not its pages still reserved (KYTY_GUEST_LAZY_COMMIT): those
	// were never committed, so they read zeros once committed, and a memset would only commit them. Nothing is freed.
	void ZeroCommitted(uint64_t offset, uint64_t end) {
		if (!GuestLazyCommit::Enabled()) {
			std::memset(m_backing_base + offset, 0, end - offset);
			return;
		}
		const auto base = reinterpret_cast<uint64_t>(m_backing_base);
		while (offset < end) {
			const uint64_t next = GuestLazyCommit::RunEnd(base + offset, base + end) - base;
			const uint64_t stop = next > offset ? next : end;
			if (!GuestLazyCommit::IsReservedPage(m_backing_base + offset)) std::memset(m_backing_base + offset, 0, stop - offset);
			offset = stop;
		}
	}
	std::vector<uint64_t>        m_used; // 1 bit per 64 KiB granule of the backing a mapping has covered

	// The host bytes of guest address `vaddr` inside mapping `entry`: in the backing view, or (native) the address itself.
	[[nodiscard]] uint8_t* HostAt(const MapEntry& entry, uint64_t vaddr) const {
		if (m_native) return reinterpret_cast<uint8_t*>(vaddr);
		return m_backing_base + entry.backing_offset + (vaddr - entry.vaddr);
	}
	// (Native, lock held) Zeroes the mapped bytes of physical [offset, end) (a granule a mapping covered before; the others
	// read zeros already): physical memory unmapped has no bytes to zero.
	void ZeroMappedNative(uint64_t offset, uint64_t end) {
		for (const auto& [_, entry]: m_maps) {
			const uint64_t begin = std::max(offset, entry.backing_offset);
			const uint64_t stop  = std::min(end, entry.backing_offset + entry.size);
			if (begin < stop) {
				// (ram4, KYTY_WRITE_WATCH: through NativeGuardedAccess as the other writes of the emulator's - a page a
				// tracker holds no-access was an access violation here, with this lock held -.)
				auto* host = reinterpret_cast<uint8_t*>(entry.vaddr + (begin - entry.backing_offset));
				NativeGuardedAccess(host, stop - begin, [&] { std::memset(host, 0, stop - begin); }, true);
			}
		}
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
public: // (ram1) used by DiscardUncoveredBacking
	// (ram1, KYTY_DS_FREED_DISCARD) The store's bytes of physical [offset, offset + size) dropped from RAM when no guest
	// mapping covers any of them (m_coverage 0 all over): DiscardVirtualMemory through the alias (MEM_RESET when Windows
	// refuses it), committed runs only. The pages stay committed (a page-file section's pages cannot be decommitted), so
	// the commit charge is the same; their RAM goes back without a page-file write (unmapped they were trimmed to the
	// modified list and written out). Under the shared lock: a map covering them (MapsEmplace, exclusive) waits for the
	// drop, so a range the game maps again is never dropped after. Not with native memory, a backing file or lazy commit.
	// Returns the bytes dropped; `covered` / `reserved` / `refused` get the bytes left because a mapping covers them,
	// because they were never committed, or because Windows refused both calls.
	uint64_t DiscardUncovered(uint64_t offset, uint64_t size, uint64_t* covered, uint64_t* reserved, uint64_t* refused) {
		if (m_native || m_lazy || m_file != INVALID_HANDLE_VALUE || m_backing_base == nullptr || size == 0 || offset >= m_size)
			return 0;
		size = std::min(size, m_size - offset);
		std::shared_lock lock(m_mutex);
		const uint64_t   end = offset + size;
		auto             it  = m_coverage.upper_bound(offset);
		if (it != m_coverage.begin()) --it;
		for (; it != m_coverage.end() && it->first < end; ++it) {
			const auto next = std::next(it);
			if (next != m_coverage.end() && next->first <= offset) continue;
			if (it->second != 0) {
				*covered += size;
				return 0;
			}
		}
		uint64_t dropped = 0;
		for (uint64_t at = offset; at < end;) {
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(m_backing_base + at, &info, sizeof(info)) != sizeof(info)) {
				*refused += end - at;
				break;
			}
			const uint64_t region_end = static_cast<uint64_t>(static_cast<uint8_t*>(info.BaseAddress) - m_backing_base) +
			                            info.RegionSize;
			const uint64_t stop = std::min(end, std::max(region_end, at + 0x1000));
			if (info.State != MEM_COMMIT) {
				*reserved += stop - at;
			} else if (DiscardVirtualMemory(m_backing_base + at, stop - at) == ERROR_SUCCESS ||
			           VirtualAlloc(m_backing_base + at, stop - at, MEM_RESET, PAGE_READWRITE) != nullptr) {
				dropped += stop - at;
			} else {
				*refused += stop - at;
			}
			at = stop;
		}
		return dropped;
	}
	// (ram2, KYTY_DS_FREED_DISCARD=2) Whether every byte of physical [begin, end) has exactly `count` guest mappings
	// (m_coverage is a step function: a key's count holds up to the next key, 0 before the first). Lock held.
	[[nodiscard]] bool CoverageIsUnlocked(uint64_t begin, uint64_t end, uint32_t count) const {
		auto it = m_coverage.upper_bound(begin);
		if ((it == m_coverage.begin() ? 0u : std::prev(it)->second) != count) return false;
		for (; it != m_coverage.end() && it->first < end; ++it) {
			if (it->second != count) return false;
		}
		return true;
	}
	// (ram2, KYTY_DS_FREED_DISCARD=2) Called before a guest unmap of [vaddr, vaddr + size): the parts of it that map the
	// page allocator's pool (physical [pool_begin, pool_end)) and are the only guest mapping of their bytes are reset
	// through the game's own view, while its pages are still in the working set (VirtualAlloc MEM_RESET, documented for a
	// view of a page-file section: "the data is no longer of interest, the pages should not be read from or written to
	// the paging file"). Unmapped right after, they leave the working set clean: no page-file write, no copy kept by
	// Windows' memory compression, and the page-file space they had is released. ram1's discard went through the
	// emulator's alias later (pages already trimmed, most never touched through the alias: nothing for MEM_RESET to
	// act on there). No VirtualQuery, no thread, no page touched; the store's shared lock (a map of these bytes waits).
	// Not with native memory, a backing file or lazy commit. Returns the bytes reset; `shared`: bytes left because
	// another mapping covers them too; `refused`: bytes Windows refused.
	uint64_t ResetSoleMapping(uint64_t vaddr, uint64_t size, uint64_t pool_begin, uint64_t pool_end, uint64_t* shared,
	                          uint64_t* refused) {
		if (m_native || m_lazy || m_file != INVALID_HANDLE_VALUE || size == 0 || UINT64_MAX - vaddr < size) return 0;
		std::shared_lock lock(m_mutex);
		const uint64_t   end  = vaddr + size;
		uint64_t         done = 0;
		auto             it   = m_maps.upper_bound(vaddr);
		if (it != m_maps.begin()) --it;
		for (; it != m_maps.end() && it->first < end; ++it) {
			const MapEntry& entry = it->second;
			const uint64_t  first = std::max(vaddr, entry.vaddr);
			const uint64_t  last  = std::min(end, entry.vaddr + entry.size);
			if (first >= last) continue;
			const uint64_t offset = entry.backing_offset + (first - entry.vaddr);
			if (offset < pool_begin || offset + (last - first) > pool_end) continue;
			if (!CoverageIsUnlocked(offset, offset + (last - first), 1)) {
				*shared += last - first;
			} else if (VirtualAlloc(reinterpret_cast<void*>(first), last - first, MEM_RESET, PAGE_READWRITE) != nullptr) {
				done += last - first;
			} else {
				*refused += last - first;
			}
		}
		return done;
	}
	// (ram6, KYTY_DS_FREED_RELEASE) The counters of ReleaseSoleMapping, bytes; `skipped` (optional, room for
	// `skipped_max`): the guest ranges left as they were (mapped twice, or refused by Windows).
	struct ReleaseCounts {
		uint64_t                       done = 0, discarded = 0, reset = 0, shared = 0, refused = 0;
		std::pair<uint64_t, uint64_t>* skipped       = nullptr;
		size_t                         skipped_max   = 0;
		size_t                         skipped_count = 0;
		void Skip(uint64_t begin, uint64_t end) {
			if (skipped == nullptr) return;
			if (skipped_count != 0 && skipped[skipped_count - 1].second == begin) skipped[skipped_count - 1].second = end;
			else if (skipped_count < skipped_max) skipped[skipped_count++] = {begin, end};
		}
	};
	// (ram6, KYTY_DS_FREED_RELEASE) The parts of guest [vaddr, vaddr + size) that map the page allocator's pool
	// (physical [pool_begin, pool_end)) and are the only guest mapping of their bytes, given back to Windows (`act`) or
	// only counted (KYTY_DS_FREED_RELEASE=2). A page-file section's committed pages cannot be decommitted ("Once the
	// memory is committed, it cannot be decommitted ... You can get sort of the effect ... by resetting it (MEM_RESET).
	// The memory is still committed." - R. Chen, The Old New Thing, 2015-01-30), so the commit charge stays; what goes
	// is the RAM and any page-file or compression-store copy: DiscardVirtualMemory (= reset and out of the working set at
	// once) through the game's view AND the emulator's alias (the uploads read through the alias, KYTY_BACKING_READ: a
	// page left valid there stays in RAM). When Windows refuses the discard: MEM_RESET and VirtualUnlock (an unlocked
	// range leaves the working set; the call "fails" with ERROR_NOT_LOCKED). The pages stay mapped and committed: the
	// next touch reads zeros or the old bytes (undefined), a write just works - no fault of ours, nothing to map again.
	// Each run of bytes is judged by its own mapping count (m_coverage). The store's shared lock (a map or unmap of these
	// bytes waits). Not with native memory, a backing file or lazy commit.
	uint64_t ReleaseSoleMapping(uint64_t vaddr, uint64_t size, uint64_t pool_begin, uint64_t pool_end, bool act,
	                            ReleaseCounts* counts) {
		if (m_native || m_lazy || m_file != INVALID_HANDLE_VALUE || m_backing_base == nullptr || size == 0 ||
		    UINT64_MAX - vaddr < size)
			return 0;
		// (ram8) MEM_RESET only: no write to the pages, whatever their protection and whoever changes it meanwhile.
		// DiscardVirtualMemory was the crash of log 22:14 (0xC0000005 writing 0x109016c000, PAGE_READONLY, at ~45 s in
		// Boletaria): on a view of a shared section it does not drop the pages but ZEROES them in user mode - the dump's
		// stack holds its QueryWorkingSetEx array (a page each 4 KiB from 0x1090130000, attributes 0x4000804f: valid,
		// shared, read-write when asked) and the faulting code is ntdll's memset (rep stosb, zeros already written below
		// the fault). A page the page manager made read-only between its query and its memset (a read/write watcher of
		// the GPU caches) faulted in the emulator's own code: no handler of ours takes that. The zeroing also brought the
		// pages INTO RAM ("DiscardVirtualMemory 197 MiB" of ram6 was mostly that). The working set trim (VirtualUnlock) is
		// done after, out of the game's lock and with the guest memory quiet: TrimSole.
		const auto drop = [](void* at, uint64_t bytes, bool* discarded) {
			*discarded = false;
			return VirtualAlloc(at, bytes, MEM_RESET, PAGE_READWRITE) != nullptr;
		};
		std::shared_lock lock(m_mutex);
		const uint64_t   end  = vaddr + size;
		uint64_t         done = 0;
		auto             it   = m_maps.upper_bound(vaddr);
		if (it != m_maps.begin()) --it;
		for (; it != m_maps.end() && it->first < end; ++it) {
			const MapEntry& entry = it->second;
			const uint64_t  first = std::max(vaddr, entry.vaddr);
			const uint64_t  last  = std::min(end, entry.vaddr + entry.size);
			if (first >= last) continue;
			const uint64_t piece_begin = entry.backing_offset + (first - entry.vaddr);
			const uint64_t piece_end   = piece_begin + (last - first);
			if (piece_begin < pool_begin || piece_end > pool_end || piece_end > m_size) continue;
			// (m_coverage is a step function: a key's count holds up to the next key, 0 before the first.)
			auto     cov   = m_coverage.upper_bound(piece_begin);
			uint32_t count = cov == m_coverage.begin() ? 0u : std::prev(cov)->second;
			for (uint64_t at = piece_begin; at < piece_end;) {
				const uint64_t stop  = cov != m_coverage.end() && cov->first < piece_end ? cov->first : piece_end;
				const uint64_t bytes = stop - at;
				const uint64_t guest = first + (at - piece_begin);
				if (bytes == 0) {
				} else if (count != 1) {
					counts->shared += bytes;
					counts->Skip(guest, guest + bytes);
				} else if (!act) {
					done += bytes;
				} else {
					bool discarded = false;
					if (!drop(reinterpret_cast<void*>(guest), bytes, &discarded)) {
						counts->refused += bytes;
						counts->Skip(guest, guest + bytes);
					} else {
						bool alias_discarded = false;
						(void)drop(m_backing_base + at, bytes, &alias_discarded); // (same pages, dead already: best effort)
						(discarded ? counts->discarded : counts->reset) += bytes;
						done += bytes;
					}
				}
				at = stop;
				if (cov != m_coverage.end() && cov->first == stop) {
					count = cov->second;
					++cov;
				}
			}
		}
		counts->done += done;
		return done;
	}
	// (ram8, KYTY_DS_FREED_RELEASE) The pages of guest [vaddr, vaddr + size) that ReleaseSoleMapping reset (the pool's,
	// mapped once) out of the working set, through the game's view and the alias: VirtualUnlock (fails "not locked" but
	// trims; nothing written, the bytes stay as they are). The caller holds the guest memory quiet (RunQuietWith).
	void TrimSole(uint64_t vaddr, uint64_t size, uint64_t pool_begin, uint64_t pool_end) {
		if (m_native || m_lazy || m_file != INVALID_HANDLE_VALUE || m_backing_base == nullptr || size == 0 ||
		    UINT64_MAX - vaddr < size)
			return;
		std::shared_lock lock(m_mutex);
		const uint64_t   end = vaddr + size;
		auto             it  = m_maps.upper_bound(vaddr);
		if (it != m_maps.begin()) --it;
		for (; it != m_maps.end() && it->first < end; ++it) {
			const MapEntry& entry = it->second;
			const uint64_t  first = std::max(vaddr, entry.vaddr);
			const uint64_t  last  = std::min(end, entry.vaddr + entry.size);
			if (first >= last) continue;
			const uint64_t piece_begin = entry.backing_offset + (first - entry.vaddr);
			const uint64_t piece_end   = piece_begin + (last - first);
			if (piece_begin < pool_begin || piece_end > pool_end || piece_end > m_size) continue;
			auto     cov   = m_coverage.upper_bound(piece_begin);
			uint32_t count = cov == m_coverage.begin() ? 0u : std::prev(cov)->second;
			for (uint64_t at = piece_begin; at < piece_end;) {
				const uint64_t stop = cov != m_coverage.end() && cov->first < piece_end ? cov->first : piece_end;
				if (stop > at && count == 1) {
					(void)VirtualUnlock(reinterpret_cast<void*>(first + (at - piece_begin)), stop - at);
					(void)VirtualUnlock(m_backing_base + at, stop - at);
				}
				at = stop;
				if (cov != m_coverage.end() && cov->first == stop) {
					count = cov->second;
					++cov;
				}
			}
		}
	}
	// (ram9, KYTY_STORE_TRIM / KYTY_STORE_HOTSET) The game's mappings of the store cut in granules of at most `granule`
	// bytes: guest address, the alias's address of the same bytes, length. Shared lock: a snapshot (a granule unmapped
	// since is trimmed for nothing: VirtualUnlock of reserved memory fails and does nothing).
	bool InFile() const { return m_file != INVALID_HANDLE_VALUE; } // (ram10: KYTY_BACKING_FILE in use)
	struct Granule {
		uint64_t guest = 0;
		uint8_t* alias = nullptr;
		uint64_t bytes = 0;
	};
	void Granules(uint64_t granule, std::vector<Granule>* out) {
		out->clear();
		// (ram10: with KYTY_BACKING_FILE too - there the trim is what sends the cold pages to the file -.)
		if (m_native || m_lazy || m_backing_base == nullptr || granule == 0) return;
		std::shared_lock lock(m_mutex);
		for (const auto& [key, entry]: m_maps) {
			(void)key;
			for (uint64_t at = 0; at < entry.size; at += granule) {
				const uint64_t bytes = std::min(granule, entry.size - at);
				if (entry.backing_offset + at + bytes > m_size) break;
				out->push_back({entry.vaddr + at, m_backing_base + entry.backing_offset + at, bytes});
			}
		}
	}
	// (ram6, KYTY_DS_FREED_RELEASE) For each guest address: bit 0 its page is in the working set through the game's
	// view, bit 1 through the emulator's alias (0 when not mapped by a single mapping known here). K32QueryWorkingSetEx,
	// no page touched. False when Windows refused.
	bool SampleResidency(const uint64_t* vaddrs, size_t count, uint8_t* flags) {
		if (m_native || m_backing_base == nullptr || count == 0) return false;
		std::vector<PSAPI_WORKING_SET_EX_INFORMATION> entries(count * 2);
		{
			std::shared_lock lock(m_mutex);
			for (size_t i = 0; i < count; i++) {
				const uint64_t v = vaddrs[i] & ~uint64_t {0xfff};
				entries[2 * i].VirtualAddress = reinterpret_cast<void*>(v);
				entries[2 * i + 1].VirtualAddress = nullptr;
				auto it = m_maps.upper_bound(v);
				if (it == m_maps.begin()) continue;
				--it;
				if (v >= it->second.vaddr && v < it->second.vaddr + it->second.size &&
				    it->second.backing_offset + (v - it->second.vaddr) < m_size)
					entries[2 * i + 1].VirtualAddress = m_backing_base + it->second.backing_offset + (v - it->second.vaddr);
			}
		}
		std::vector<PSAPI_WORKING_SET_EX_INFORMATION> query;
		query.reserve(entries.size());
		for (const auto& e: entries)
			if (e.VirtualAddress != nullptr) query.push_back(e);
		if (!query.empty() &&
		    K32QueryWorkingSetEx(GetCurrentProcess(), query.data(), static_cast<DWORD>(query.size() * sizeof(query[0]))) == 0)
			return false;
		size_t q = 0;
		for (size_t i = 0; i < count; i++) {
			flags[i] = 0;
			if (entries[2 * i].VirtualAddress != nullptr && q < query.size() && query[q].VirtualAddress == entries[2 * i].VirtualAddress)
				flags[i] |= query[q++].VirtualAttributes.Valid != 0 ? 1u : 0u;
			if (entries[2 * i + 1].VirtualAddress != nullptr && q < query.size() &&
			    query[q].VirtualAddress == entries[2 * i + 1].VirtualAddress)
				flags[i] |= query[q++].VirtualAttributes.Valid != 0 ? 2u : 0u;
		}
		return true;
	}
private:
#endif
	// (Native) Whether no guest mapping covers physical [offset, offset + size); the first alias is printed (16 at most).
	bool NativeCoverageFree(uint64_t offset, uint64_t size, uint64_t vaddr) {
		std::shared_lock lock(m_mutex);
		const uint64_t   end = offset + size;
		auto             it  = m_coverage.upper_bound(offset);
		if (it != m_coverage.begin()) --it;
		for (; it != m_coverage.end() && it->first < end; ++it) {
			const auto next = std::next(it);
			if (next != m_coverage.end() && next->first <= offset) continue;
			if (it->second != 0) {
				static std::atomic<uint32_t> said {0};
				if (said.fetch_add(1) < 16) {
					// (raiz2: the start-up self-test - Memory's SelfTestSub64SharedPlaceholderAlias, before the game loads -
					// asks for exactly this refusal; said as such, so a game alias stands out.)
					std::printf("Guest memory (native): physical 0x%llx+0x%llx mapped at 0x%llx is mapped elsewhere already: "
					            "refused (KYTY_NATIVE_MEMORY has no aliases)%s\n",
					            static_cast<unsigned long long>(offset), static_cast<unsigned long long>(size),
					            static_cast<unsigned long long>(vaddr),
					            g_native_alias_selftest ? " - start-up self-test, expected" : " - BY THE GAME");
					std::fflush(stdout);
				}
				return false;
			}
		}
		return true;
	}
	bool                         m_native       = false; // KYTY_NATIVE_MEMORY (NativeMemoryRequested), Windows only
	uint8_t*                     m_backing_base = nullptr;
	uint64_t                     m_size         = 0;
	std::shared_mutex            m_mutex;
	std::mutex                   m_data_mutex;
	std::mutex                   m_cpu_write_mutex; // (TryWriteCpuBacking)
	std::vector<std::pair<uint64_t, uint64_t>> m_aliased_physical_ranges;
	bool m_alias_ranges_valid = false;
	std::map<uint64_t, MapEntry> m_maps;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// The file views of the guest mappings (base -> size): each view's mapped parts (m_maps) and parked ranges (m_parked)
	// tile it. A view is one mapping until a partial unmap parks a range in it.
	std::map<uint64_t, uint64_t> m_views;
	// Ranges the guest unmapped from views still mapped (base -> size, each inside one view). A view cannot be
	// split: unmapping it whole and mapping its other parts again cost ~0.05 ms per MiB of the view (1.3 s in 3 minutes
	// of play, up to 29 ms a call, on the game's job threads, all its memory calls waiting), while the game mapped
	// something over under 1% of such ranges within a second (96%: after 10 s or never). Their views are split for real
	// when something is mapped over one (Materialize), or unmapped whole once nothing of them stays mapped.
	std::map<uint64_t, uint64_t> m_parked;

	std::map<uint64_t, uint64_t>::iterator FindViewUnlocked(uint64_t vaddr) {
		const auto next = m_views.upper_bound(vaddr);
		if (next == m_views.begin()) return m_views.end();
		const auto view = std::prev(next);
		return vaddr < view->first + view->second ? view : m_views.end();
	}

	// The first parked range that overlaps [vaddr, vaddr + size), or end.
	std::map<uint64_t, uint64_t>::iterator FirstParkedUnlocked(uint64_t vaddr, uint64_t size) {
		auto it = m_parked.upper_bound(vaddr);
		if (it != m_parked.begin() && std::prev(it)->first + std::prev(it)->second > vaddr) --it;
		return it != m_parked.end() && it->first < vaddr + size ? it : m_parked.end();
	}

	// [vaddr, vaddr + size) parked, joined to the parked ranges next to it in the same view [view_begin, view_end).
	void ParkUnlocked(uint64_t vaddr, uint64_t size, uint64_t view_begin, uint64_t view_end) {
		uint64_t begin = vaddr, end = vaddr + size;
		auto     next = m_parked.lower_bound(vaddr);
		if (next != m_parked.begin()) {
			const auto previous = std::prev(next);
			if (previous->first >= view_begin && previous->first + previous->second == begin) {
				begin = previous->first;
				m_parked.erase(previous);
			}
		}
		if (next != m_parked.end() && next->first == end && next->first + next->second <= view_end) {
			end = next->first + next->second;
			m_parked.erase(next);
		}
		EXIT_IF(!m_parked.emplace(begin, end - begin).second);
	}
#endif
	// Physical coverage of m_maps: [key, next key) is mapped by `value` guest mappings (the
	// last key closes the covered space with 0). Changed with m_maps (MapsEmplace/MapsErase),
	// so IsUniqueRange looks only at the requested range instead of rebuilding the alias
	// intervals of every mapping after each map or unmap (~0.7 ms with 6700 mappings, two to
	// three times a frame while the game streams).
	std::map<uint64_t, uint32_t> m_coverage;
	alignas(64) std::atomic_bool m_host_write_pending {false};
	// The pages of the data write in progress (DataWriteFence), 0 when none.
	alignas(64) std::atomic<uint64_t> m_data_write_pages {0};
	// ... of the CPU-owned write in progress (TryWriteCpuBacking).
	alignas(64) std::atomic<uint64_t> m_cpu_write_pages {0};
	// The pages each active host read copies (all until it knows), 0 when none: a reader per slot. With one slot,
	// a read that met another (an upload worker's BackingPointer, hundreds a frame) took the data mutex instead and
	// waited there behind the copies of the writers (the render thread's per-draw reads: ~2.6 waits, ~0.23 ms a frame).
	struct alignas(64) HostReadSlot {
		std::atomic<uint64_t> pages {0};
	};
	static constexpr size_t                    HostReadSlots = 8;
	std::array<HostReadSlot, HostReadSlots> m_host_read_slots {};
	uint64_t m_mapping_epoch = 1;
	// Moves when some physical byte gains or loses its second guest mapping (AddCoverage): uniqueness proven before holds
	// while it stays (and the mapping's region epoch, RegionEpoch).
	uint64_t m_alias_epoch = 1;
};

// The address range ReserveGuestRegions holds free for frame generation's device creation (ReleaseFrameGenAddressHole).
static constexpr uint64_t FrameGenHoleSize = uint64_t {64} << 30u;
static void*              g_frame_gen_hole = nullptr;

class GuestAddressSpace;
// (The address space while it lives: Kyty.Trim's trims go through it, TrimParkedRange.)
static std::atomic<GuestAddressSpace*> g_trim_space {nullptr};

class GuestAddressSpace {
	// m_mutex held exclusively, counted with its time (address_space_holds, address_space_hold_kt): every protection
	// change of the GPU trackers waits for such a hold (ProtectTransient: address_space_waits).
	class ExclusiveLock {
	public:
		explicit ExclusiveLock(std::shared_mutex& mutex): m_lock(mutex), m_start(__rdtsc()) {}
		~ExclusiveLock() {
			LiveCounters::Add(LiveCounters::AddressSpaceHolds);
			LiveCounters::Add(LiveCounters::AddressSpaceHoldKt, (__rdtsc() - m_start) >> 10u);
		}
		KYTY_CLASS_NO_COPY(ExclusiveLock);
		std::unique_lock<std::shared_mutex>& Lock() { return m_lock; }

	private:
		std::unique_lock<std::shared_mutex> m_lock;
		uint64_t                            m_start;
	};

public:
	explicit GuestAddressSpace(uint64_t backing_size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		m_granularity = info.dwAllocationGranularity;
#endif
		ReserveGuestRegions();
		m_backing = new GuestBackingStore(backing_size);
		g_trim_space.store(this, std::memory_order_release);
		if (!m_backing->IsAvailable()) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			EXIT("could not reserve %" PRIu64 " MB for guest direct memory. Windows commits\n"
			     "this up front, so the paging file is usually what needs to be larger.\n",
			     backing_size / (1024U * 1024U));
#else
			EXIT("could not reserve %" PRIu64 " MB for guest direct memory\n",
			     backing_size / (1024U * 1024U));
#endif
		}
	}

	~GuestAddressSpace() {
		g_trim_space.store(nullptr, std::memory_order_release);
		delete m_backing;
		m_backing = nullptr;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (const auto& [start, size]: m_owned) {
			const auto end     = start + size;
			auto       current = start;
			while (current < end) {
				MEMORY_BASIC_INFORMATION info {};
				EXIT_IF(VirtualQuery(reinterpret_cast<const void*>(current), &info, sizeof(info)) ==
				        0);
				const auto region_start = reinterpret_cast<uint64_t>(info.BaseAddress);
				const auto region_end   = region_start + static_cast<uint64_t>(info.RegionSize);
				if (info.State == MEM_COMMIT) {
					EXIT_IF(VirtualFree(reinterpret_cast<void*>(region_start), info.RegionSize,
					                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0);
					continue;
				}
				if (info.State == MEM_RESERVE) {
					EXIT_IF(VirtualFree(info.AllocationBase, 0, MEM_RELEASE) == 0);
				}
				current = region_end;
			}
		}
#else
		for (const auto& [start, size]: m_owned) {
			munmap(reinterpret_cast<void*>(start), size);
		}
#endif
	}

	KYTY_CLASS_NO_COPY(GuestAddressSpace);

	[[nodiscard]] uint64_t GetGranularity() const { return m_granularity; }
	[[nodiscard]] uint64_t GetBackingBase() const { return m_backing->GetBackingBase(); }
	[[nodiscard]] uint64_t GetBackingSize() const { return m_backing->GetBackingSize(); }
	[[nodiscard]] bool     IsNative() const { return m_backing->IsNative(); }

	// (KYTY_NATIVE_MEMORY) The 4 KiB pages of [vaddr, vaddr + size) written since their last reset, appended to `pages`
	// in order, over the guest mappings there (direct and flexible); `reset`: cleared at once, 64 MiB per call. False:
	// not native, or a part was refused (an allocation without MEM_WRITE_WATCH): its pages are missing. The address
	// space's lock is held shared (no map or unmap meanwhile; protection changes go on).
	bool NativeWrittenPages(uint64_t vaddr, uint64_t size, bool reset, std::vector<uint64_t>* pages) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (!m_backing->IsNative() || size == 0 || UINT64_MAX - vaddr < size) return false;
		std::shared_lock lock(m_mutex);
		return NativeWrittenPagesUnlocked(vaddr, size, reset, pages);
#else
		(void)vaddr, (void)size, (void)reset, (void)pages;
		return false;
#endif
	}

	// (ww2) NativeWrittenPages for `count` ranges [first, second) under one hold of the lock (a sweep's: one shared lock
	// for all its runs of regions instead of one each). `failed` (may be null): the index of each range a part of which
	// was refused. False: not native, or some range failed.
	bool NativeWrittenRanges(const std::pair<uint64_t, uint64_t>* ranges, size_t count, bool reset, std::vector<uint64_t>* pages,
	                         std::vector<uint32_t>* failed) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (!m_backing->IsNative()) return false;
		std::shared_lock lock(m_mutex);
		bool             ok = true;
		for (size_t i = 0; i < count; ++i) {
			if (ranges[i].second <= ranges[i].first) continue;
			if (!NativeWrittenPagesUnlocked(ranges[i].first, ranges[i].second - ranges[i].first, reset, pages)) {
				ok = false;
				if (failed != nullptr) failed->push_back(static_cast<uint32_t>(i));
			}
		}
		return ok;
#else
		(void)ranges, (void)count, (void)reset, (void)pages, (void)failed;
		return false;
#endif
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// (m_mutex held, shared or not.)
	bool NativeWrittenPagesUnlocked(uint64_t vaddr, uint64_t size, bool reset, std::vector<uint64_t>* pages) {
		constexpr uint64_t Page  = 0x1000;
		// 64 MiB of pages: one call never overflows, so a reset loses nothing. (ww2: 16 MiB before - a call for every
		// 16 MiB of a large mapping, at every sweep.)
		constexpr ULONG_PTR Slots = 16384;
		thread_local std::vector<void*> found(Slots);
		bool             ok  = true;
		const uint64_t   end = vaddr + size;
		const auto query = [&](uint64_t begin, uint64_t stop) {
			ULONG_PTR count = Slots;
			ULONG     granularity = 0;
			g_native_watch_calls.fetch_add(1, std::memory_order_relaxed);
			if (GetWriteWatch(reset ? WRITE_WATCH_FLAG_RESET : 0, reinterpret_cast<void*>(begin), stop - begin, found.data(),
			                  &count, &granularity) != 0)
				return false;
			for (ULONG_PTR i = 0; i < count; ++i) pages->push_back(reinterpret_cast<uint64_t>(found[i]));
			return true;
		};
		auto it = m_mapped.upper_bound(vaddr);
		if (it != m_mapped.begin()) --it;
		for (; it != m_mapped.end() && it->second.vaddr < end; ++it) {
			const uint64_t begin = std::max(vaddr, it->second.vaddr) & ~(Page - 1u);
			const uint64_t stop  = std::min(end, it->second.vaddr + it->second.size);
			for (uint64_t at = begin; at < stop;) {
				const uint64_t next = std::min(stop, at + Slots * Page);
				if (!query(at, next)) {
					// (A chunk over two allocations, or one not write-watched: again a region at a time.)
					g_native_watch_retries.fetch_add(1, std::memory_order_relaxed);
					for (uint64_t part = at; part < next;) {
						MEMORY_BASIC_INFORMATION info {};
						if (VirtualQuery(reinterpret_cast<const void*>(part), &info, sizeof(info)) != sizeof(info)) {
							ok = false;
							break;
						}
						const uint64_t part_end =
						    std::min(next, reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize));
						if (info.State == MEM_COMMIT && !query(part, part_end)) ok = false;
						part = std::max(part_end, part + Page);
					}
				}
				at = next;
			}
		}
		return ok;
	}
#endif

	bool ZeroBacking(uint64_t backing_offset, uint64_t size) {
		return m_backing->ZeroRange(backing_offset, size);
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// (ram1, KYTY_DS_FREED_DISCARD) GuestBackingStore::DiscardUncovered.
	uint64_t DiscardUncoveredBacking(uint64_t backing_offset, uint64_t size, uint64_t* covered, uint64_t* reserved,
	                                 uint64_t* refused) {
		return m_backing->DiscardUncovered(backing_offset, size, covered, reserved, refused);
	}
	// (ram2, KYTY_DS_FREED_DISCARD=2) GuestBackingStore::ResetSoleMapping.
	uint64_t ResetBackingBeforeUnmap(uint64_t vaddr, uint64_t size, uint64_t pool_begin, uint64_t pool_end,
	                                 uint64_t* shared, uint64_t* refused) {
		return m_backing->ResetSoleMapping(vaddr, size, pool_begin, pool_end, shared, refused);
	}
	// (ram6, KYTY_DS_FREED_RELEASE) GuestBackingStore::ReleaseSoleMapping / SampleResidency.
	uint64_t ReleaseBackingSole(uint64_t vaddr, uint64_t size, uint64_t pool_begin, uint64_t pool_end, bool act,
	                            GuestBackingStore::ReleaseCounts* counts) {
		return m_backing->ReleaseSoleMapping(vaddr, size, pool_begin, pool_end, act, counts);
	}
	bool StoreInFile() const { return m_backing->InFile(); } // (ram10)
	void StoreGranules(uint64_t granule, std::vector<GuestBackingStore::Granule>* out) { // (ram9)
		m_backing->Granules(granule, out);
	}
	void TrimBackingSole(uint64_t vaddr, uint64_t size, uint64_t pool_begin, uint64_t pool_end) { // (ram8)
		m_backing->TrimSole(vaddr, size, pool_begin, pool_end);
	}
	bool SampleBackingResidency(const uint64_t* vaddrs, size_t count, uint8_t* flags) {
		return m_backing->SampleResidency(vaddrs, count, flags);
	}
#endif

	[[nodiscard]] bool Owns(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		std::shared_lock lock(m_mutex);
		return std::any_of(m_owned.begin(), m_owned.end(), [vaddr, size](const auto& range) {
			return vaddr >= range.first && vaddr + size <= range.first + range.second;
		});
	}

	// Whether a guest mapping holds vaddr, once a mapping change under way is done: on Windows a partial
	// unmap unmaps the whole view and maps its other parts back (a view cannot be split), all under the
	// exclusive lock, and an access to those parts meanwhile faults on unmapped memory.
	[[nodiscard]] bool IsMappedAfterChanges(uint64_t vaddr) {
		std::shared_lock lock(m_mutex);
		WaitNotBusy(lock, vaddr, 1);
		const auto       it = FindMappedContainingUnlocked(vaddr);
		return it != m_mapped.end();
	}

	[[nodiscard]] bool OverlapsOwned(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		std::shared_lock lock(m_mutex);
		return std::any_of(m_owned.begin(), m_owned.end(), [vaddr, size](const auto& range) {
			return VirtualRangesOverlap(vaddr, size, range.first, range.second);
		});
	}

	bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
		return m_backing->TryWriteBacking(vaddr, data, size);
	}
	bool TryWriteCpuBacking(uint64_t vaddr, const void* data, uint64_t size) {
		return m_backing->TryWriteCpuBacking(vaddr, data, size);
	}

	bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
		return m_backing->TryReadBacking(vaddr, data, size);
	}
	bool TryReadBackingToHost(uint64_t vaddr, void* data, uint64_t size) {
		return m_backing->TryReadBackingToHost(vaddr, data, size);
	}

	const uint8_t* BackingPointer(uint64_t vaddr, uint64_t size) { return m_backing->BackingPointer(vaddr, size); }
	uint64_t       BackingMappingEnd(uint64_t vaddr) { return m_backing->MappingEndOf(vaddr); }
	bool BackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces) {
		return m_backing->BackingPieces(vaddr, size, pieces);
	}

	bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size) {
		return m_backing->TryReadSparseBacking(vaddr, data, size);
	}
	bool SparseBackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces) {
		return m_backing->SparseBackingPieces(vaddr, size, pieces);
	}

	bool IsUniqueBackingRange(uint64_t vaddr, uint64_t size) { return m_backing->IsUniqueRange(vaddr, size); }
	bool BackingAliases(uint64_t vaddr, uint64_t size, std::vector<std::pair<uint64_t, uint64_t>>* aliases) {
		return m_backing->Aliases(vaddr, size, aliases);
	}
	bool IsUniqueBackingRangeAcrossMappings(uint64_t vaddr, uint64_t size) {
		return m_backing->IsUniqueRangeAcrossMappings(vaddr, size);
	}

	bool BackingContains(uint64_t vaddr, uint64_t size) { return m_backing->Contains(vaddr, size); }

	bool Protect(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		GuestBackingMutation mutation;
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		ExclusiveLock lock(m_mutex);
		return ProtectMappedUnlocked(vaddr, size, mode);
	}

	// Returns once no change of the address space is under way (Memory::WriteBacking: a partial unmap's views mapped
	// again).
	void WaitForChanges(uint64_t vaddr, uint64_t size) {
		std::shared_lock lock(m_mutex);
		WaitNotBusy(lock, vaddr, size);
	}

	// Protect, keeping the GPU trackers' protection of the pages a watcher holds (what ReapplyProtection gives back):
	// under the lock, which every protection change of the trackers waits for, no CPU write reaches a watched page
	// between the guest's protection and theirs (a guest mprotect to writable made such writes unseen).
	bool ProtectKeepingWatched(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		GuestBackingMutation mutation;
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		thread_local std::vector<Graphics::PageManager::WatchedRun> runs;
		runs.clear();
		ExclusiveLock lock(m_mutex);
		if (g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size)) g_gpu_resources->WatchedRuns(vaddr, size, &runs);
		uint64_t at = vaddr;
		for (const auto& run: runs) {
			const auto begin = std::max(run.address, vaddr);
			const auto end   = std::min(run.address + run.size, vaddr + size);
			if (begin >= end) continue;
			if (at < begin && !ProtectMappedUnlocked(at, begin - at, mode)) return false;
			if (!ProtectMappedUnlocked(begin, end - begin, run.mode)) return false;
			at = end;
		}
		return at >= vaddr + size || ProtectMappedUnlocked(at, vaddr + size - at, mode);
	}

	bool ProtectTransient(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		LiveCounters::Add(LiveCounters::ProtectCalls);
		if (LiveCensus::g_render) {
			LiveCounters::Add(LiveCounters::ProtectCallsRender);
			if (LiveCensus::g_queue == 0) LiveCounters::Add(LiveCounters::ProtectCallsGfx);
		}

		// The page manager serializes changes of one region under its lock; different regions
		// never share pages, so these calls may run concurrently.
		std::shared_lock lock(m_mutex, std::try_to_lock);
		if (!lock.owns_lock()) {
			const auto start = __rdtsc();
			lock.lock();
			const auto waited = (__rdtsc() - start) >> 10u;
			LiveCounters::Add(LiveCounters::AddressSpaceWaits);
			LiveCounters::Add(LiveCounters::AddressSpaceWaitKt, waited);
			if (LiveCensus::g_render) LiveCounters::Add(LiveCounters::AddressSpaceWaitKtRender, waited);
		}
		WaitNotBusy(lock, vaddr, size);
		// (FRAME-NATIVO2 v2) The excluded (imported) DefaultTagg arenas stay read-write for good: a protection of the
		// trackers or of RAIZ over them is dropped, the rest of the range protected as asked. Read-write passes.
		if (mode != VirtualMemory::Mode::ReadWrite && mode != VirtualMemory::Mode::ExecuteReadWrite &&
		    ::FrameNativo2::AnyExcluded()) {
			// (v3) Arena bases are found at run time, in any order: the excluded ones walked by address.
			uint64_t bases[::FrameNativo2::kArenas];
			int      nb = 0;
			for (int i = 0; i < ::FrameNativo2::kArenas; ++i)
				if (::FrameNativo2::g_excluded[i].load(std::memory_order_acquire) && ::FrameNativo2::Base(i) != 0)
					bases[nb++] = ::FrameNativo2::Base(i);
			std::sort(bases, bases + nb);
			uint64_t at = vaddr;
			for (int k = 0; k < nb; ++k) {
				const uint64_t b = std::max(at, bases[k]);
				const uint64_t e = std::min(vaddr + size, bases[k] + ::FrameNativo2::kArenaSize);
				if (b >= e) continue;
				::FrameNativo2::g_skipped.fetch_add(1, std::memory_order_relaxed);
				if (at < b && !ProtectMappedUnlocked(at, b - at, mode)) return false;
				at = e;
			}
			return at >= vaddr + size || ProtectMappedUnlocked(at, vaddr + size - at, mode);
		}
		// (FRAME-NATIVO2 v11, KYTY_FRAME_NATIVO2=4) Bloques de 1 MiB de instancias de solo lectura (AllocTag 0x7fe350):
		// sin vigilar, su proteccion se descarta (siguen RW); el resto del rango se protege como se pide.
		if (mode != VirtualMemory::Mode::ReadWrite && mode != VirtualMemory::Mode::ExecuteReadWrite &&
		    ::FrameNativo2::SoloLectura() && ::FrameNativo2::RoAny(vaddr, size)) {
			uint64_t at = vaddr, b = 0, e = 0;
			while (::FrameNativo2::RoNextRun(at, vaddr + size, &b, &e)) {
				::FrameNativo2::g_skipped.fetch_add(1, std::memory_order_relaxed);
				if (at < b && !ProtectMappedUnlocked(at, b - at, mode)) return false;
				at = e;
			}
			return at >= vaddr + size || ProtectMappedUnlocked(at, vaddr + size - at, mode);
		}
		return ProtectMappedUnlocked(vaddr, size, mode);
	}

	bool MapBacking(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                VirtualMemory::Mode               mode,
	                GuestBackingStore::FailureReason* failure_reason = nullptr) {
		LiveTrace::SlowOsCallScope slow(LiveTrace::OsMapBacking, vaddr, size);
		// (KYTY_NATIVE_MEMORY: a map inside a parked range takes it in its allocation, AdoptParkedNative.)
		if (m_backing->IsNative()) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			NativePendingDrop(vaddr, size); // (raiz3: nothing kept for an older mapping lands in this one)
#endif
			{
				ExclusiveLock lock(m_mutex);
				m_busy_done.wait(lock.Lock(), [&] { return !BusyUnlocked(vaddr, size); });
				m_busy.emplace_back(vaddr, vaddr + size);
			}
			const auto adopt_mode = VirtualMemory::IsExecute(mode) ? VirtualMemory::Mode::ExecuteReadWrite
			                                                       : VirtualMemory::Mode::ReadWrite;
			const int  adopted    = m_backing->AdoptParkedNative(vaddr, size, backing_offset, adopt_mode, failure_reason);
			if (adopted > 0) m_backing->NoteUsed(backing_offset, size);
			{
				ExclusiveLock lock(m_mutex);
				if (adopted > 0) EXIT_IF(!AddMappedUnlocked(vaddr, size, MappingType::Backing));
				EraseBusyUnlocked(vaddr, vaddr + size);
			}
			m_busy_done.notify_all();
			if (adopted != 0) return adopted > 0;
		}
		MaterializeParked(vaddr, size);
		FreeSplit split;
		{
			ExclusiveLock lock(m_mutex);
			// (Guest maps and unmaps run one at a time: another one over the range is only waited for here.)
			m_busy_done.wait(lock.Lock(), [&] { return !BusyUnlocked(vaddr, size); });
			if (!ConsumeDeferredUnlocked(vaddr, size, &split)) {
				if (failure_reason != nullptr) {
					*failure_reason = GuestBackingStore::FailureReason::AddressRangeUnavailable;
				}
				return false;
			}
			m_busy.emplace_back(vaddr, vaddr + size);
		}
		// The system calls without the lock (BusyUnlocked): the trackers' protections of other ranges go on.
		const bool split_done = SplitPlaceholders(split, vaddr, size);
		const auto host_mode  = VirtualMemory::IsExecute(mode)
		                            ? VirtualMemory::Mode::ExecuteReadWrite
		                            : VirtualMemory::Mode::ReadWrite;
		m_backing->NoteUsed(backing_offset, size);
		const bool mapped =
		    split_done && m_backing->MapExistingPlaceholderFixed(vaddr, size, backing_offset, host_mode, failure_reason);
		{
			ExclusiveLock lock(m_mutex);
			if (mapped) {
				EXIT_IF(!AddMappedUnlocked(vaddr, size, MappingType::Backing));
			} else {
				RestoreSplitUnlocked(split, vaddr, size);
			}
			EraseBusyUnlocked(vaddr, vaddr + size);
		}
		m_busy_done.notify_all();
		return mapped;
	}

	bool UnmapBacking(uint64_t vaddr, uint64_t size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		NativePendingDrop(vaddr, size); // (raiz3: what was kept for the range is nobody's now)
		ColdStore::Drop(vaddr, size);   // (ram11: what the cold store holds there is nobody's either)
#endif
		bool unmapped = false;
		{
			LiveTrace::SlowOsCallScope slow(LiveTrace::OsUnmapBacking, vaddr, size);
			struct UnmapHold {
				uint64_t start = __rdtsc();
				~UnmapHold() { LiveCounters::Add(LiveCounters::AddressSpaceUnmapHoldKt, (__rdtsc() - start) >> 10u); }
			} unmap_hold;
			g_protective_remap = kyty_local_protective_remap.load(std::memory_order_relaxed) != 0;
			// The views over the range busy, the system calls without the lock (BusyUnlocked): a partial unmap of a big
			// view held it for milliseconds (4.5 ms for a 52 MiB one, 1.5-5 ms a call late in a laptop session), every
			// protection of the GPU trackers anywhere waiting behind it.
			uint64_t busy_begin = vaddr, busy_end = vaddr + size;
			bool     contained  = false;
			{
				ExclusiveLock lock(m_mutex);
				// (Guest maps and unmaps run one at a time: another one over the views is only waited for here.)
				for (;;) {
					contained = ContainsMappedUnlocked(vaddr, size, MappingType::Backing);
					if (!contained) break;
					busy_begin = vaddr;
					busy_end   = vaddr + size;
					m_backing->ViewSpan(vaddr, size, &busy_begin, &busy_end);
					if (!BusyUnlocked(busy_begin, busy_end - busy_begin)) {
						m_busy.emplace_back(busy_begin, busy_end);
						break;
					}
					m_busy_done.wait(lock.Lock());
				}
			}
			if (contained) {
				bool                                       placeholder_preserved = false;
				std::vector<std::pair<uint64_t, uint64_t>> freed, spans;
				const bool done = m_backing->Unmap(vaddr, size, &placeholder_preserved, &freed) && placeholder_preserved;
				{
					ExclusiveLock lock(m_mutex);
					if (done) {
						EXIT_IF(!RemoveMappedUnlocked(vaddr, size, MappingType::Backing));
						spans    = AddFreedUnlocked(freed);
						unmapped = true;
					}
					EraseBusyUnlocked(busy_begin, busy_end);
				}
				m_busy_done.notify_all();
				// (The free placeholders around them made one without the lock: guest maps and unmaps, the only users of
				// the free list, run one at a time.)
				CoalesceSpans(spans);
			}
		}
		if (g_remapped_count != 0) {
			LiveCounters::Add(LiveCounters::RemappedViews, std::exchange(g_remapped_count, 0u));
			if (const auto us = kyty_local_remap_window_us.load(std::memory_order_relaxed); us != 0)
				std::this_thread::sleep_for(std::chrono::microseconds(us));
		}
		ReapplyRemappedViews();
		g_remap_window_begin.store(0, std::memory_order_release);
		return unmapped;
	}

	// The free-range queries below only read, so they share the lock with the page trackers'
	// protection changes (ProtectTransient, hundreds a frame on the render thread). Taken
	// exclusively, the hundreds of queries of one hint-less guest mapping (FindGuestFreeRange)
	// kept the render thread waiting up to 15 ms at a time while rolling in Boletaria.
	uint64_t ReserveAligned(uint64_t search_addr, uint64_t size, uint64_t alignment) {
		if (size == 0) {
			return 0;
		}
		alignment = std::max<uint64_t>(alignment != 0 ? alignment : PageSize(), PageSize());
		std::shared_lock lock(m_mutex);
		if (search_addr == 0) {
			auto addr = FindFreeAlignedUnlocked(HOST_SYSTEM_MANAGED_MIN,
			                                    HOST_SYSTEM_MANAGED_MAX + 1u, size, alignment);
			return addr != 0 ? addr
			                 : FindFreeAlignedUnlocked(HOST_USER_MIN, HOST_USER_MAX + 1u, size,
			                                           alignment);
		}
		return FindFreeAlignedUnlocked(search_addr, HOST_USER_MAX + 1u, size, alignment);
	}

	uint64_t FindFreeAligned(uint64_t search_start, uint64_t search_end, uint64_t size,
	                         uint64_t alignment) {
		if (size == 0 || alignment == 0) {
			return 0;
		}
		alignment = std::max<uint64_t>(alignment, PageSize());
		std::shared_lock lock(m_mutex);
		return FindFreeAlignedUnlocked(search_start, search_end, size, alignment);
	}

	bool ReserveFixed(uint64_t vaddr, uint64_t size) {
		if (vaddr == 0 || size == 0 || !IsPageAligned(vaddr) || !IsPageAligned(size)) {
			return false;
		}
		std::shared_lock lock(m_mutex);
		return FreeOrParkedUnlocked(vaddr, size);
	}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	bool TestContainsFree(uint64_t vaddr, uint64_t size) {
		ExclusiveLock lock(m_mutex);
		return FreeOrParkedUnlocked(vaddr, size);
	}
#endif

	bool ReleaseFree(uint64_t vaddr, uint64_t size) {
		std::shared_lock lock(m_mutex);
		return FreeOrParkedUnlocked(vaddr, size);
	}

	bool Commit(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		MaterializeParked(vaddr, size);
		ExclusiveLock lock(m_mutex);
		if (!ConsumeUnlocked(vaddr, size)) {
			return false;
		}
		const auto host_mode = VirtualMemory::IsExecute(mode)
		                           ? VirtualMemory::Mode::ExecuteReadWrite
		                           : VirtualMemory::Mode::ReadWrite;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		LiveTrace::SlowOsCallScope slow(LiveTrace::OsAlloc, vaddr, size);
		// (commit-al-tocar, KYTY_COMMIT_AL_TOCAR=1: native guest memory reserved only, committed at the first touch.)
		const bool  lazy        = m_backing->IsNative() && NativeLazyCommit::Start();
		const DWORD lazy_commit = lazy ? 0u : static_cast<DWORD>(MEM_COMMIT);
		if (lazy) NativeLazyCommit::Register(vaddr, size);
		// (Refused for lack of commit: a release pass and another try, MemGuard; the guard's releasers never take
		// this lock.)
		auto* ptr = MemGuard::RetryOs("guest private commit", size, [&] {
			// (KYTY_NATIVE_MEMORY: write-watched too, NativeWriteWatch; without it where Windows refuses that.)
			void* done = nullptr;
			if (m_backing->IsNative())
				done = VirtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(vaddr), size,
				                     MEM_RESERVE | lazy_commit | MEM_REPLACE_PLACEHOLDER | MEM_WRITE_WATCH, GetProtect(host_mode),
				                     nullptr, 0);
			if (done == nullptr)
				done = VirtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(vaddr), size,
				                     MEM_RESERVE | lazy_commit | MEM_REPLACE_PLACEHOLDER, GetProtect(host_mode), nullptr, 0);
			return done;
		});
		if (ptr == nullptr || reinterpret_cast<uint64_t>(ptr) != vaddr) {
			if (lazy) NativeLazyCommit::Unregister(vaddr, size);
			if (ptr != nullptr) {
				VirtualFree(ptr, 0, MEM_RELEASE);
			}
			AddFreeUnlocked(vaddr, size);
			return false;
		}
		MapaGrande::Note(vaddr, size); // (KYTY_MAPA_GRANDE)
		EXIT_IF(!AddMappedUnlocked(vaddr, size, MappingType::Private));
		return true;
#else
		if (VirtualMemory::Protect(vaddr, size, host_mode)) {
			EXIT_IF(!AddMappedUnlocked(vaddr, size, MappingType::Private));
			return true;
		}
		AddFreeUnlocked(vaddr, size);
		return false;
#endif
	}

	bool ReleaseCommitted(uint64_t vaddr, uint64_t size) {
		if (vaddr == 0 || size == 0 || !IsPageAligned(vaddr) || !IsPageAligned(size)) {
			return false;
		}
		ExclusiveLock lock(m_mutex);
		if (!ContainsMappedUnlocked(vaddr, size, MappingType::Private)) {
			return false;
		}
		const auto end     = vaddr + size;
		auto       current = vaddr;
		std::vector<std::pair<uint64_t, uint64_t>> regions;
		while (current < end) {
			const auto it       = FindMappedContainingUnlocked(current);
			const auto part_end = std::min(end, it->second.vaddr + it->second.size);
			regions.emplace_back(current, part_end - current);
			current = part_end;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (const auto& [region_vaddr, region_size]: regions) {
			NativeLazyCommit::Unregister(region_vaddr, region_size); // (commit-al-tocar)
			LiveTrace::SlowOsCallScope slow(LiveTrace::OsFree, region_vaddr, region_size);
			EXIT_IF(VirtualFree(reinterpret_cast<void*>(region_vaddr), region_size,
			                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0);
		}
		// ReleaseCommitted also publishes the aggregate span as one logical free range.
		EXIT_IF(regions.size() > 1 && !CoalescePlaceholders(vaddr, size));
#else
		for (const auto& [region_vaddr, region_size]: regions) {
			RestorePlaceholder(region_vaddr, region_size);
		}
#endif
		EXIT_IF(!RemoveMappedUnlocked(vaddr, size, MappingType::Private));
		AddFreeUnlocked(vaddr, size);
		return true;
	}

	bool SelfTest() {
		const auto addr = ReserveAligned(0, PageSize(), PageSize());
		if (addr == 0) {
			LOGF_COLOR(Log::Color::Red,
			           "\t guest address-space self-test: 16 KiB range unavailable\n");
			return false;
		}
		if (!Commit(addr, PageSize(), VirtualMemory::Mode::ReadWrite)) {
			LOGF_COLOR(Log::Color::Red, "\t guest address-space self-test: commit failed\n");
			return false;
		}

		auto* ptr = reinterpret_cast<uint64_t*>(addr);
		*ptr      = 0x4b59545950484f4cull; // "KYTYPHOL"

		if (!ReleaseCommitted(addr, PageSize())) {
			LOGF_COLOR(Log::Color::Red,
			           "\t guest address-space self-test: release-committed failed\n");
			return false;
		}

		LOGF_COLOR(Log::Color::Green, "\t guest address-space self-test: ok\n");
		return true;
	}

private:
	enum class MappingType { Backing, Private };

	struct MappedRegion {
		uint64_t    vaddr = 0;
		uint64_t    size  = 0;
		MappingType type  = MappingType::Private;
	};

	static constexpr uint64_t PageSize() { return 0x4000; }

	static uint64_t AlignUp(uint64_t value, uint64_t alignment) {
		if (alignment == 0) {
			return 0;
		}
		const auto remainder = value % alignment;
		const auto increment = (remainder != 0 ? alignment - remainder : 0);
		return increment <= UINT64_MAX - value ? value + increment : 0;
	}

	static uint64_t AlignDown(uint64_t value, uint64_t alignment) {
		return value & ~(alignment - 1u);
	}

	[[nodiscard]] static bool IsPageAligned(uint64_t value) {
		return (value & (PageSize() - 1u)) == 0;
	}

	[[nodiscard]] bool IsGranularityAligned(uint64_t value) const {
		return (value & (m_granularity - 1u)) == 0;
	}

	// Whether a backing map or unmap is changing a range that overlaps [vaddr, vaddr + size) outside the lock: its page-table
	// pages (2 MiB of addresses each, and the next one where it ends on one's start) as a whole. Windows' memory manager
	// bug-checked (0x1A/0x41202, issues #5 and #50, 16 GB PCs) in a protection change of a host thread from the first page
	// of a 2 MiB-aligned range (a texture's) while another view was mapped or unmapped: busy ranges were per view, and a
	// protection change of a neighbouring view in the same page-table page went on during the system calls. Maps and
	// unmaps run one at a time anyway; protection changes in those pages wait for them.
	// Any map or unmap anywhere makes a protection change wait, not only one in its page-table pages (the views are all of
	// one section: their prototype PTEs are shared): two reporters of #5 still bug-checked with v20261018, whose waits
	// were per page-table page; no measurable cost (an RTX 5090 desktop and an RTX 3080 laptop, 1-1 routes A-B-A).
	// KYTY_PROTECT_SERIAL=0: the per page-table page waits only.
	[[nodiscard]] bool BusyUnlocked(uint64_t vaddr, uint64_t size) const {
		// (raiz2, KYTY_RAIZ: off by default - KYTY_PROTECT_SERIAL=1 puts it back -. The serial waits were for views of
		// one section, whose prototype PTEs all maps share; private memory has none: a protection change waits only for
		// a map or unmap in its own page-table pages.)
		static const bool serial = [] {
			const char* value = std::getenv("KYTY_PROTECT_SERIAL");
			if (value == nullptr || *value == '\0') return !Raiz::On();
			return *value != '0';
		}();
		if (serial && !m_busy.empty()) return true;
		constexpr uint64_t PageTablePage = uint64_t {2} << 20u;
		for (const auto& [begin, end]: m_busy) {
			const auto page_begin = begin & ~(PageTablePage - 1u);
			const auto page_end   = (end + PageTablePage) & ~(PageTablePage - 1u);
			if (vaddr < page_end && page_begin < vaddr + size) return true;
		}
		return false;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
public:
	// Kyty.Trim's trim of a parked range (VirtualUnlock), 16 MiB at a time, each busy meanwhile: no map or unmap runs in
	// its page-table pages then, nor it during theirs (it ran unlocked before, during a view's map again too: one of the
	// overlaps of the bug check above). No lock is held during the trim.
	void TrimParked(uint64_t vaddr, uint64_t size) {
		constexpr uint64_t Chunk = uint64_t {16} << 20u;
		for (uint64_t at = vaddr, end = vaddr + size; at < end;) {
			const uint64_t next = std::min(end, (at & ~(Chunk - 1u)) + Chunk);
			{
				ExclusiveLock lock(m_mutex);
				m_busy_done.wait(lock.Lock(), [&] { return !BusyUnlocked(at, next - at); });
				m_busy.emplace_back(at, next);
			}
			// (nativemem1: with native memory and KYTY_DS_FREED_DISCARD=1 - ram1's switch, which needs the backing view
			// otherwise - the parked private pages are discarded instead: their RAM goes back with no page-file write and they
			// stay committed and writable, so a late write of the emulator's there does not fault. A map there zeroes them
			// anyway, AdoptParkedNative. The trim when Windows refuses.)
			if (!NativeParkDiscard() || DiscardVirtualMemory(reinterpret_cast<void*>(at), next - at) != ERROR_SUCCESS)
				VirtualUnlock(reinterpret_cast<void*>(at), next - at);
			{
				ExclusiveLock lock(m_mutex);
				EraseBusyUnlocked(at, next);
			}
			m_busy_done.notify_all();
			at = next;
		}
	}

private:
#endif

	// (With m_mutex held shared.) Returns once no map or unmap is changing [vaddr, vaddr + size).
	void WaitNotBusy(std::shared_lock<std::shared_mutex>& lock, uint64_t vaddr, uint64_t size) {
		if (!BusyUnlocked(vaddr, size)) return;
		LiveCounters::Add(LiveCounters::AddressSpaceBusyWaits);
		m_busy_done.wait(lock, [&] { return !BusyUnlocked(vaddr, size); });
	}

	void EraseBusyUnlocked(uint64_t begin, uint64_t end) {
		const auto it = std::find(m_busy.begin(), m_busy.end(), std::pair<uint64_t, uint64_t> {begin, end});
		EXIT_IF(it == m_busy.end());
		m_busy.erase(it);
	}

	bool ProtectMappedUnlocked(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		// (nativemem1, KYTY_WRITE_WATCH: not while a copy has pages opened, NativeGuardedAccess.)
		// (raiz7) The GPU (render) thread's protection changes timed whole - the gate's wait, the kept bytes' copy, the
		// VirtualProtect - for the "[raiz]" line: what they cost that thread a frame, measured, not inferred.
		// (raiz8) Every thread's, by origin (Raiz::t_protect_origin, the page manager's), with its pages: the "[raiz]"
		// line's PROTECT BY ORIGIN.
		struct GpuProtectTimer {
			bool                                  on    = Raiz::On();
			bool                                  gpu   = on && Graphics::GuestGpu::IsGpuThread();
			uint64_t                              pages = 0;
			uint64_t                              range_vaddr = 0, range_size = 0; // (g1)
			std::chrono::steady_clock::time_point start = on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
			~GpuProtectTimer() {
				if (!on) return;
				const auto ns = static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
				const auto us = ns / 1000u;
				Raiz::NoteProtect(gpu, pages, us);
				G1::NoteProtect(gpu, Raiz::t_protect_origin, range_vaddr, range_size, ns); // (g1, KYTY_G1_PROBE: by label and pass)
				auto& cost = Raiz::g_cost[Raiz::ClassOf(gpu)]; // (raiz9: in ns - a change is a few us -)
				cost.calls.fetch_add(1, std::memory_order_relaxed);
				cost.ns.fetch_add(ns, std::memory_order_relaxed);
				if (!gpu) return;
				Raiz::g_counters.protects_gpu.fetch_add(1, std::memory_order_relaxed);
				Raiz::g_counters.protect_us_gpu.fetch_add(us, std::memory_order_relaxed);
			}
		} gpu_timer;
		gpu_timer.pages = (size + 0xfffu) >> 12u;
		gpu_timer.range_vaddr = vaddr; // (g1)
		gpu_timer.range_size  = size;
		// (VIGILANCIA-DOBLE, local/nativo-cubierto.h) Pages a native piece serves (MarcarCubierto) get no watch protection
		// at all (read-track / read-flush / write-track / sync / reapply to read-only or no-access): nothing is done.
		if (::NativoCubierto::SkipWatch(vaddr, size, Raiz::t_protect_origin,
		                                mode == VirtualMemory::Mode::NoAccess || mode == VirtualMemory::Mode::Read)) {
			gpu_timer.on = false;
			return true;
		}
		::FrameNativo::NoteProtect(vaddr, size); // (FRAME-NATIVO) every change, by the memory map's label
		::NativoCubierto::NoteProtect(vaddr, Raiz::t_protect_origin); // (VIGILANCIA-DOBLE: the [vigilancia] line, zone x reason)
		// (raiz1: the gate, shared - alone with KYTY_RAIZ_VERIFY -: NativeProtectLock.)
		NativeProtectLock guard(NativeReadFaultsRequested());
		// (ram11; ram12: with the gate held - the cold store probes and evicts with it alone -, so nothing of the range
		// can be probed or evicted between this and the change below.)
		ColdStore::Ensure(vaddr, size);
		F0Measure::Ensure(vaddr, size); // (f0: our guards off first)
#endif
		const auto end = vaddr + size;
		auto       it  = m_mapped.upper_bound(vaddr);
		if (it != m_mapped.begin()) {
			--it;
		}
		for (; it != m_mapped.end() && it->second.vaddr < end; ++it) {
			const auto& region      = it->second;
			const auto  range_start = std::max(vaddr, region.vaddr);
			const auto  range_end   = std::min(end, region.vaddr + region.size);
			if (range_start >= range_end) continue;
			LiveTrace::SlowOsCallScope slow(LiveTrace::OsProtect, range_start, range_end - range_start);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (m_backing->IsNative()) g_native_protects.fetch_add(1, std::memory_order_relaxed); // (nativemem1, the 5 s line)
			// (raiz1, KYTY_RAIZ_VERIFY: the write watch audit around it, NativeAuditProtect.)
			const bool audit         = m_backing->IsNative() && Raiz::VerifyOn();
			const bool now_protected = mode == VirtualMemory::Mode::NoAccess || mode == VirtualMemory::Mode::Read ||
			                           mode == VirtualMemory::Mode::Execute || mode == VirtualMemory::Mode::ExecuteRead;
			if (audit) NativeAuditProtect(range_start, range_end - range_start, true, now_protected);
			// (raiz5: the shadows of these pages - no-access no more, or again - forgotten first.)
			if (m_backing->IsNative()) NativeShadowDrop(range_start, range_end - range_start);
			// (raiz10, KYTY_RAIZ_KEEP_STRIPES) A change that makes pages readable: its stripes shared from the kept bytes'
			// take through its VirtualProtect, so no lazy keep of these pages slips in between (a keep holds them alone).
			// Not inside a window (the gate alone: no keep runs) nor nested in another protection of this thread.
			NativeStripeLock<false> stripes(range_start, range_end - range_start,
			                                m_backing->IsNative() && NativeLazyOn() && NativeKeepStripesOn() &&
			                                    mode != VirtualMemory::Mode::NoAccess && t_native_window_depth == 0 &&
			                                    t_native_protect_depth == 1);
			// (raiz3, KYTY_RAIZ_LAZY: GPU results kept for pages this makes readable go in first.)
			// (raiz4: when that left every page read-write and read-write is what is asked, nothing more to do.)
			// (raiz11, KYTY_RAIZ_MERGE_FAST) A change to read-only: the put-in leaves its runs' final protection to the
			// VirtualProtect below and checks them after it (NativeMergeAfterProtect) - one VirtualProtect less a run -.
			t_native_merge_after = m_backing->IsNative() && NativeLazyOn() && mode == VirtualMemory::Mode::Read && NativeMergeFastOn();
			const bool lazy_done = m_backing->IsNative() && NativeLazyOn() &&
			                       NativePendingBeforeProtect(range_start, range_end - range_start, mode != VirtualMemory::Mode::NoAccess,
			                                                  mode == VirtualMemory::Mode::ReadWrite) &&
			                       mode == VirtualMemory::Mode::ReadWrite;
			t_native_merge_after = false;
			struct MergeAfter {
				~MergeAfter() {
					if (!t_native_merge_checks.empty()) NativeMergeAfterProtect(); // (after the VirtualProtect, or its failure)
				}
			} merge_after;
#else
			const bool lazy_done = false;
#endif
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			// (raiz9: VirtualProtect's own time, by thread class.)
			const auto vp_start = gpu_timer.on && !lazy_done ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
#endif
			if (!lazy_done && !VirtualMemory::Protect(range_start, range_end - range_start, mode)) {
				return false;
			}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (gpu_timer.on && !lazy_done) {
				auto& cost = Raiz::g_cost[Raiz::ClassOf(gpu_timer.gpu)];
				cost.vp_calls.fetch_add(1, std::memory_order_relaxed);
				cost.vp_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				                                               std::chrono::steady_clock::now() - vp_start)
				                                               .count()),
				                     std::memory_order_relaxed);
			}
#endif
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (audit) NativeAuditProtect(range_start, range_end - range_start, false, now_protected);
#endif
		}
		return true;
	}

	// A free entry consumed with its placeholder split later (SplitPlaceholders, outside the lock).
	struct FreeSplit {
		uint64_t base = 0, end = 0; // the free entry
	};

	// ConsumeUnlocked's free-list change without its system calls: the remainders become free entries now.
	bool ConsumeDeferredUnlocked(uint64_t vaddr, uint64_t size, FreeSplit* split) {
		auto it = FindContaining(vaddr, size);
		if (it == m_free.end()) {
			return false;
		}
		split->base = it->first;
		split->end  = it->first + it->second;
		m_free.erase(it);
		if (split->base != vaddr) m_free.emplace(split->base, vaddr - split->base);
		if (vaddr + size != split->end) m_free.emplace(vaddr + size, split->end - (vaddr + size));
		return true;
	}

	// The placeholder of a consumed free entry split around [vaddr, vaddr + size) (SplitUnlocked's calls). False when
	// a split failed (RestoreSplitUnlocked).
	static bool SplitPlaceholders(const FreeSplit& split, uint64_t vaddr, uint64_t size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (split.base != vaddr &&
		    !PlaceholderFree(reinterpret_cast<void*>(split.base), vaddr - split.base, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER))
			return false;
		if (vaddr + size != split.end &&
		    !PlaceholderFree(reinterpret_cast<void*>(vaddr), size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER))
			return false;
#else
		(void)split, (void)vaddr, (void)size;
#endif
		return true;
	}

	// A map that failed gives its free entry back whole, its placeholders (the ones split) made one again.
	void RestoreSplitUnlocked(const FreeSplit& split, uint64_t vaddr, uint64_t size) {
		m_free.erase(m_free.lower_bound(split.base), m_free.lower_bound(split.end));
		(void)vaddr, (void)size;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		// (A coalesce of one placeholder fails: nothing was split then.)
		(void)CoalescePlaceholders(split.base, split.end - split.base);
#endif
		m_free.emplace(split.base, split.end - split.base);
	}

	// AddFreeUnlocked's free-list change without its system calls, for the free placeholders of an unmap
	// (GuestBackingStore::Unmap's `freed`): each joins the list as its own entry. Returned: the spans they make with the
	// free entries next to them, where there are such, to be coalesced outside the lock (CoalesceSpans).
	std::vector<std::pair<uint64_t, uint64_t>> AddFreedUnlocked(const std::vector<std::pair<uint64_t, uint64_t>>& freed) {
		for (const auto& [vaddr, size]: freed) {
			auto [it, inserted] = m_free.emplace(vaddr, size);
			if (!inserted) it->second = size;
		}
		std::vector<std::pair<uint64_t, uint64_t>> spans;
		for (const auto& [vaddr, size]: freed) {
			const auto it    = m_free.find(vaddr);
			uint64_t   begin = it->first, end = it->first + it->second;
			for (auto first = it; first != m_free.begin() && std::prev(first)->first + std::prev(first)->second == begin;) {
				--first;
				begin = first->first;
			}
			for (auto last = std::next(it); last != m_free.end() && last->first == end; ++last) end = last->first + last->second;
			if (end - begin > size &&
			    std::none_of(spans.begin(), spans.end(), [begin](const auto& span) { return span.first == begin; }))
				spans.emplace_back(begin, end);
		}
		return spans;
	}

	// The free entries of each span made one, their placeholders first (outside the lock: guest maps and unmaps, the only
	// users of the free list, run one at a time).
	void CoalesceSpans(const std::vector<std::pair<uint64_t, uint64_t>>& spans) {
		for (const auto& [begin, end]: spans) {
			if (CoalesceFreePlaceholders(begin, end)) {
				ExclusiveLock lock(m_mutex);
				MergeFreeUnlocked(begin, end);
			}
		}
	}

	// Whether [vaddr, vaddr + size) is free: free placeholders, or ranges parked in views (GuestBackingStore::UnmapOne),
	// which a map or commit there makes free placeholders first (MaterializeParked).
	bool FreeOrParkedUnlocked(uint64_t vaddr, uint64_t size) {
		if (FindContaining(vaddr, size) != m_free.end()) return true;
		const auto end = vaddr + size;
		for (auto at = vaddr; at < end;) {
			if (const auto it = FindContaining(at, 1); it != m_free.end()) {
				at = it->first + it->second;
			} else if (const auto parked_end = m_backing->ParkedEnd(at); parked_end != 0) {
				at = parked_end;
			} else {
				return false;
			}
		}
		return true;
	}

	// `call` with the whole address space busy: no map, unmap or protection change of the guest's views runs meanwhile
	// (they wait for it, as for each other; the working set trim, LocalPlatform::SetTrimRunner).
public:
	// (At most 200 ms waiting for the maps and unmaps under way: past that the trim is skipped this time, never stalled.)
	bool RunQuiet(void (*call)()) {
		{
			ExclusiveLock lock(m_mutex);
			if (!m_busy_done.wait_for(lock.Lock(), std::chrono::milliseconds(200), [&] { return m_busy.empty(); }))
				return false;
			m_busy.emplace_back(0, uint64_t {1} << 47u);
		}
		call();
		{
			ExclusiveLock lock(m_mutex);
			EraseBusyUnlocked(0, uint64_t {1} << 47u);
		}
		m_busy_done.notify_all();
		return true;
	}
	// The same for a callable (the periodic RAM flush's trim, ram-guard.h).
	template <typename F>
	bool RunQuietWith(F&& call) {
		{
			ExclusiveLock lock(m_mutex);
			if (!m_busy_done.wait_for(lock.Lock(), std::chrono::milliseconds(200), [&] { return m_busy.empty(); }))
				return false;
			m_busy.emplace_back(0, uint64_t {1} << 47u);
		}
		call();
		{
			ExclusiveLock lock(m_mutex);
			EraseBusyUnlocked(0, uint64_t {1} << 47u);
		}
		m_busy_done.notify_all();
		return true;
	}

private:

	// The ranges parked in views over [vaddr, vaddr + size) made free placeholders before a map or commit there: their
	// views split for real (GuestBackingStore::Materialize), busy meanwhile (their mapped parts are mapped again).
	void MaterializeParked(uint64_t vaddr, uint64_t size) {
		uint64_t begin = UINT64_MAX, end = 0;
		if (!m_backing->ParkedViews(vaddr, size, &begin, &end)) return;
		g_protective_remap = kyty_local_protective_remap.load(std::memory_order_relaxed) != 0;
		{
			ExclusiveLock lock(m_mutex);
			m_busy_done.wait(lock.Lock(), [&] { return !BusyUnlocked(begin, end - begin); });
			m_busy.emplace_back(begin, end);
		}
		std::vector<std::pair<uint64_t, uint64_t>> freed, spans;
		m_backing->Materialize(vaddr, size, &freed);
		{
			ExclusiveLock lock(m_mutex);
			spans = AddFreedUnlocked(freed);
			EraseBusyUnlocked(begin, end);
		}
		m_busy_done.notify_all();
		CoalesceSpans(spans);
		if (g_remapped_count != 0) {
			LiveCounters::Add(LiveCounters::RemappedViews, std::exchange(g_remapped_count, 0u));
		}
		ReapplyRemappedViews();
		g_remap_window_begin.store(0, std::memory_order_release);
	}

	static bool CoalesceFreePlaceholders(uint64_t begin, uint64_t end) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return CoalescePlaceholders(begin, end - begin);
#else
		(void)begin, (void)end;
		return true;
#endif
	}

	// The free entries of [begin, end) made one (their placeholders were: CoalesceFreePlaceholders).
	void MergeFreeUnlocked(uint64_t begin, uint64_t end) {
		m_free.erase(m_free.lower_bound(begin), m_free.lower_bound(end));
		m_free.emplace(begin, end - begin);
	}

	bool ConsumeUnlocked(uint64_t vaddr, uint64_t size) {
		auto it = FindContaining(vaddr, size);
		if (it == m_free.end() || !SplitUnlocked(it, vaddr, size)) {
			return false;
		}
		it = m_free.find(vaddr);
		if (it == m_free.end() || it->second != size) {
			return false;
		}
		m_free.erase(it);
		return true;
	}

	std::map<uint64_t, MappedRegion>::iterator FindMappedContainingUnlocked(uint64_t vaddr) {
		auto next = m_mapped.upper_bound(vaddr);
		if (next == m_mapped.begin()) {
			return m_mapped.end();
		}
		auto it = std::prev(next);
		return vaddr < it->second.vaddr + it->second.size ? it : m_mapped.end();
	}

	bool ContainsMappedUnlocked(uint64_t vaddr, uint64_t size, MappingType type) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		const auto end     = vaddr + size;
		auto       current = vaddr;
		while (current < end) {
			const auto it = FindMappedContainingUnlocked(current);
			if (it == m_mapped.end() || it->second.type != type) {
				return false;
			}
			current = std::min(end, it->second.vaddr + it->second.size);
		}
		return true;
	}

	bool AddMappedUnlocked(uint64_t vaddr, uint64_t size, MappingType type) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		const auto end  = vaddr + size;
		auto       next = m_mapped.lower_bound(vaddr);
		if (next != m_mapped.end() && next->second.vaddr < end) {
			return false;
		}
		if (next != m_mapped.begin()) {
			const auto& previous = std::prev(next)->second;
			if (previous.vaddr + previous.size > vaddr) {
				return false;
			}
		}
		return m_mapped.emplace(vaddr, MappedRegion {vaddr, size, type}).second;
	}

	bool RemoveMappedUnlocked(uint64_t vaddr, uint64_t size, MappingType type) {
		if (!ContainsMappedUnlocked(vaddr, size, type)) {
			return false;
		}
		const auto end     = vaddr + size;
		auto       current = vaddr;
		while (current < end) {
			auto       it       = FindMappedContainingUnlocked(current);
			const auto original = it->second;
			const auto part_end = std::min(end, original.vaddr + original.size);
			m_mapped.erase(it);
			if (original.vaddr < current) {
				const auto left_size = current - original.vaddr;
				EXIT_IF(!AddMappedUnlocked(original.vaddr, left_size, original.type));
			}
			if (part_end < original.vaddr + original.size) {
				const auto right_size = original.vaddr + original.size - part_end;
				EXIT_IF(!AddMappedUnlocked(part_end, right_size, original.type));
			}
			current = part_end;
		}
		return true;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static DWORD GetProtect(VirtualMemory::Mode mode) {
		switch (mode) {
			case VirtualMemory::Mode::Read: return PAGE_READONLY;
			case VirtualMemory::Mode::Write:
			case VirtualMemory::Mode::ReadWrite: return PAGE_READWRITE;
			case VirtualMemory::Mode::Execute: return PAGE_EXECUTE;
			case VirtualMemory::Mode::ExecuteRead: return PAGE_EXECUTE_READ;
			case VirtualMemory::Mode::ExecuteWrite:
			case VirtualMemory::Mode::ExecuteReadWrite: return PAGE_EXECUTE_READWRITE;
			case VirtualMemory::Mode::NoAccess:
			default: return PAGE_NOACCESS;
		}
	}
#endif

	// DLSS frame generation (KYTY_FRAMEGEN) or DLSS super resolution (KYTY_UPSCALE=dlss, KYTY_TAA_REPLACE=dlss): both
	// go through NGX, which reserves its address space as the device is made. (DLSS without frame generation closed
	// the emulator there: the guest had taken the whole range, the ceiling was only for KYTY_FRAMEGEN.)
	static bool NvidiaNgxWanted() {
		const char* framegen = std::getenv("KYTY_FRAMEGEN");
		if (framegen != nullptr && std::atoi(framegen) > 0) return true;
		const char* upscale = std::getenv("KYTY_UPSCALE");
		if (upscale != nullptr && std::strcmp(upscale, "dlss") == 0) return true;
		const char* taa = std::getenv("KYTY_TAA_REPLACE");
		return taa != nullptr && std::strcmp(taa, "dlss") == 0;
	}

	void ReserveGuestRegions() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		constexpr uint64_t MinimumRegionSize = 0x1000000;
		const auto         process           = GetCurrentProcess();
		auto               guest_end         = HOST_USER_MAX + 1u;
		auto               next_addr         = HOST_SYSTEM_MANAGED_MIN;
		// KYTY_RESERVE_CEILING=<hex>: stop below it, leaving the top of the 40-bit range to the
		// driver. DLSS (frame generation or super resolution: NvidiaNgxWanted) needs it: NVIDIA's optical flow and NGX
		// interop reserve about 44 GiB of host address space below 1 TiB when the device is
		// created, and fail it (VK_ERROR_INITIALIZATION_FAILED) when the guest took all of it.
		// With frame generation the guest stops at 512 GiB (its mappings take the lowest free
		// addresses, tens of GiB above HOST_USER_MIN): a ceiling of 0xE0_0000_0000 left an RTX 5090
		// enough, an RTX 5080's device creation failed until 0x80_0000_0000 (issue #32).
		// (DLSS super resolution, KYTY_UPSCALER=dlss, makes its device through NGX the same way.)
		const char* framegen   = std::getenv("KYTY_FRAMEGEN");
		const char* upscaler   = std::getenv("KYTY_UPSCALER");
		const bool  streamline = (framegen != nullptr && std::atoi(framegen) > 0) ||
		                        (upscaler != nullptr && std::strcmp(upscaler, "dlss") == 0);
		uint64_t ceiling = 0;
		if (const char* value = std::getenv("KYTY_RESERVE_CEILING"); value != nullptr && *value != '\0') {
			ceiling = std::strtoull(value, nullptr, 16);
		} else if (NvidiaNgxWanted() || streamline) {
			// (DLSS frame generation's optical flow and NGX reserve ~44 GiB below 1 TiB at device creation: an RTX 5080's
			// device was refused with the ceiling at 0xE0_0000_0000; upstream issue #32.)
			ceiling = 0x8000000000ull;
		}
		if (ceiling != 0) {
			guest_end = std::clamp<uint64_t>(ceiling, HOST_USER_MIN, guest_end);
		}
		// KYTY_FG_ADDRESS_HOLE=0|1 (default 1 with frame generation): 64 GiB right above the guest held free until the
		// Vulkan device is made (ReleaseFrameGenAddressHole). The executable is not randomized (/DYNAMICBASE:NO), so the
		// process's allocations after the guest's take the lowest free addresses one after another, from the guest's end
		// up: on an RTX 5080 the device creation with frame generation failed (VK_ERROR_INITIALIZATION_FAILED) until the
		// user turned on bottom-up address randomization for the emulator in Windows' exploit protection (issue #45),
		// which scatters them. (The size NVIDIA's interop reserved below 1 TiB on the 5090: about 44 GiB.)
		if ((streamline || NvidiaNgxWanted()) && guest_end <= (uint64_t {1} << 40u) - FrameGenHoleSize) {
			const char* hole = std::getenv("KYTY_FG_ADDRESS_HOLE");
			if (hole == nullptr || *hole != '0') {
				g_frame_gen_hole = VirtualAlloc(reinterpret_cast<void*>(guest_end), FrameGenHoleSize, MEM_RESERVE,
				                                PAGE_NOACCESS);
			}
		}

		while (next_addr < guest_end) {
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<const void*>(next_addr), &info, sizeof(info)) == 0) {
				EXIT("failed to inspect guest address space at 0x%016" PRIx64 "\n", next_addr);
			}

			const auto region_base = reinterpret_cast<uint64_t>(info.BaseAddress);
			const auto region_end =
			    std::min<uint64_t>(guest_end, region_base + static_cast<uint64_t>(info.RegionSize));
			const auto reserve_start =
			    AlignUp(std::max<uint64_t>(region_base, HOST_SYSTEM_MANAGED_MIN), m_granularity);
			const auto reserve_end = AlignDown(region_end, m_granularity);

			if (info.State == MEM_FREE && reserve_end > reserve_start &&
			    reserve_end - reserve_start > MinimumRegionSize) {
				auto* ptr = VirtualAlloc2(
				    process, reinterpret_cast<void*>(reserve_start), reserve_end - reserve_start,
				    MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
				if (reinterpret_cast<uint64_t>(ptr) != reserve_start) {
					EXIT("failed to reserve guest address space at 0x%016" PRIx64
					     ", size 0x%016" PRIx64 "\n",
					     reserve_start, reserve_end - reserve_start);
				}
				AddFreeUnlocked(reserve_start, reserve_end - reserve_start);
				m_owned.emplace_back(reserve_start, reserve_end - reserve_start);
			}

			if (region_end <= next_addr) {
				EXIT("guest address-space scan did not advance at 0x%016" PRIx64 "\n", next_addr);
			}
			next_addr = region_end;
		}
#else
		const std::array<std::pair<uint64_t, uint64_t>, 3> regions {{
		    {HOST_SYSTEM_MANAGED_MIN, HOST_SYSTEM_MANAGED_MAX + 1u},
		    {HOST_SYSTEM_RESERVED_MIN, HOST_SYSTEM_RESERVED_MAX + 1u},
		    {HOST_USER_MIN, HOST_USER_MAX + 1u},
		}};

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
		std::vector<std::pair<uint64_t, uint64_t>> occupied;
		FILE*                                      maps = fopen("/proc/self/maps", "r");
		EXIT_IF(maps == nullptr);

		char line[512];
		while (fgets(line, sizeof(line), maps) != nullptr) {
			unsigned long long mapping_start = 0;
			unsigned long long mapping_end   = 0;
			if (sscanf(line, "%llx-%llx", &mapping_start, &mapping_end) == 2) {
				occupied.emplace_back(static_cast<uint64_t>(mapping_start),
				                      static_cast<uint64_t>(mapping_end));
			}
		}
		fclose(maps);

		auto reserve_range = [this](uint64_t start, uint64_t end) {
			start = AlignUp(start, PageSize());
			end   = AlignDown(end, PageSize());
			if (start == 0 || end <= start) {
				return;
			}

			const auto size = end - start;
			int        flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#if defined(KYTY_LINKED_GUEST_ADDRESS_SPACE)
			flags |= MAP_FIXED;
#elif defined(MAP_FIXED_NOREPLACE)
			flags |= MAP_FIXED_NOREPLACE;
#endif

			void* ptr = mmap(reinterpret_cast<void*>(start), size, PROT_NONE, flags, -1, 0);
			if (ptr == MAP_FAILED || reinterpret_cast<uint64_t>(ptr) != start) {
				if (ptr != MAP_FAILED) {
					munmap(ptr, size);
				}
				return;
			}

			AddFreeUnlocked(start, size);
			m_owned.emplace_back(start, size);
		};

		for (const auto& [region_start, region_end]: regions) {
			auto current = region_start;

			for (const auto& [mapping_start, mapping_end]: occupied) {
				if (mapping_end <= current) {
					continue;
				}
				if (mapping_start >= region_end) {
					break;
				}

				if (mapping_start > current) {
					reserve_range(current, std::min(mapping_start, region_end));
				}

				current = std::max(current, mapping_end);
				if (current >= region_end) {
					break;
				}
			}

			if (current < region_end) {
				reserve_range(current, region_end);
			}
		}
#else
		for (const auto& [start, end]: regions) {
			int flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#if defined(KYTY_LINKED_GUEST_ADDRESS_SPACE)
			flags |= MAP_FIXED;
#elif defined(MAP_FIXED_NOREPLACE)
			flags |= MAP_FIXED_NOREPLACE;
#endif
			void* ptr = mmap(reinterpret_cast<void*>(start), end - start, PROT_NONE, flags, -1, 0);
			if (ptr == MAP_FAILED || reinterpret_cast<uint64_t>(ptr) != start) {
				if (ptr != MAP_FAILED) {
					munmap(ptr, end - start);
				}
				EXIT("failed to reserve guest address space at 0x%016" PRIx64
				     ", size 0x%016" PRIx64 "\n",
				     start, end - start);
			}
			AddFreeUnlocked(start, end - start);
			m_owned.emplace_back(start, end - start);
		}
#endif
#endif
	}

	uint64_t FindFreeAlignedUnlocked(uint64_t search_start, uint64_t search_end, uint64_t size,
	                                 uint64_t alignment) {
		if (search_start >= search_end || size > search_end - search_start) {
			return 0;
		}
		// The free ranges are sorted and disjoint: the first that can hold a candidate at or past
		// search_start is the one containing it (or the next). Scanning from the bottom of the
		// address space walked thousands of free ranges below it, again for each guest
		// reservation a hint-less mapping skips (FindGuestFreeRange).
		auto it = m_free.upper_bound(search_start);
		if (it != m_free.begin()) {
			--it;
		}
		for (; it != m_free.end() && it->first < search_end; ++it) {
			const auto& [base, range_size] = *it;
			const auto candidate           = AlignUp(std::max(base, search_start), alignment);
			if (candidate < search_end && size <= search_end - candidate &&
			    GuestFreeRangeContains(base, range_size, candidate, size)) {
				return candidate;
			}
		}
		return 0;
	}

	std::map<uint64_t, uint64_t>::iterator FindContaining(uint64_t vaddr, uint64_t size) {
		auto next = m_free.upper_bound(vaddr);
		if (next == m_free.begin()) {
			return m_free.end();
		}
		auto it  = std::prev(next);
		auto end = it->first + it->second;
		return (vaddr >= it->first && vaddr + size <= end ? it : m_free.end());
	}

	bool SplitUnlocked(std::map<uint64_t, uint64_t>::iterator it, uint64_t vaddr, uint64_t size) {
		const auto base = it->first;
		const auto end  = it->first + it->second;
		if (base == vaddr && end == vaddr + size) {
			return true;
		}

		if (base != vaddr) {
			const auto left_size = vaddr - base;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (!PlaceholderFree(reinterpret_cast<void*>(base), left_size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
				LOGF_COLOR(Log::Color::Red,
				           "\t guest placeholder split failed: base=0x%016" PRIx64
				           " size=0x%016" PRIx64 " requested=0x%016" PRIx64
				           " request_size=0x%016" PRIx64 " error=0x%08" PRIx32 "\n",
				           base, end - base, vaddr, size, static_cast<uint32_t>(GetLastError()));
				return false;
			}
#endif
			it->second = left_size;
			it         = m_free.emplace(vaddr, end - vaddr).first;
		}

		if (it->second != size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (!PlaceholderFree(reinterpret_cast<void*>(it->first), size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
				LOGF_COLOR(Log::Color::Red,
				           "\t guest placeholder split failed: base=0x%016" PRIx64
				           " size=0x%016" PRIx64 " requested=0x%016" PRIx64
				           " request_size=0x%016" PRIx64 " error=0x%08" PRIx32 "\n",
				           it->first, it->second, vaddr, size,
				           static_cast<uint32_t>(GetLastError()));
				return false;
			}
#endif
			const auto right_start = it->first + size;
			const auto right_size  = it->second - size;
			it->second             = size;
			m_free.emplace(right_start, right_size);
		}

		return true;
	}

	void AddFreeUnlocked(uint64_t vaddr, uint64_t size) {
		if (vaddr == 0 || size == 0) {
			return;
		}
		auto [it, inserted] = m_free.emplace(vaddr, size);
		if (!inserted) {
			it->second = size;
		}

		auto base = it->first;
		auto end  = it->first + it->second;

		auto first = it;
		while (first != m_free.begin()) {
			auto prev = std::prev(first);
			if (prev->first + prev->second != base) {
				break;
			}
			base  = prev->first;
			first = prev;
		}

		auto last = std::next(it);
		while (last != m_free.end() && end == last->first) {
			end  = last->first + last->second;
			last = std::next(last);
		}

		if (base != it->first || end != it->first + it->second) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (CoalescePlaceholders(base, end - base)) {
				m_free.erase(first, last);
				m_free.emplace(base, end - base);
			}
#else
			m_free.erase(first, last);
			m_free.emplace(base, end - base);
#endif
		}
	}

	void RestorePlaceholder(uint64_t vaddr, uint64_t size) {
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
		// The address remains owned for the process lifetime; MAP_FIXED only replaces one of our
		// mappings and never acquires a new guest address.
		void* ptr = mmap(reinterpret_cast<void*>(vaddr), size, PROT_NONE,
		                 MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
		EXIT_IF(ptr == MAP_FAILED || reinterpret_cast<uint64_t>(ptr) != vaddr);
#else
		(void)vaddr;
		(void)size;
#endif
	}

	// Shared by the transient (page-watch) protection changes and the ownership queries, which
	// only read the maps: guest write faults and GPU readbacks on many threads otherwise queue on
	// it for every VirtualProtect. Mapping changes take it exclusively.
	std::shared_mutex                          m_mutex;
	// Ranges a backing map or unmap is changing outside m_mutex (MapBacking, UnmapBacking: their system calls), under
	// it; what overlaps one waits for it (WaitNotBusy). Guest maps and unmaps run one at a time (the kernel's memory
	// operation mutex).
	std::vector<std::pair<uint64_t, uint64_t>> m_busy;
	std::condition_variable_any                m_busy_done;
	std::map<uint64_t, uint64_t>               m_free;
	std::map<uint64_t, MappedRegion>            m_mapped;
	std::vector<std::pair<uint64_t, uint64_t>> m_owned;
	uint64_t                                   m_granularity = 0x10000;
	GuestBackingStore*                         m_backing     = nullptr;
};

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
static void TrimParkedRange(uint64_t vaddr, uint64_t size) {
	if (auto* space = g_trim_space.load(std::memory_order_acquire); space != nullptr) {
		space->TrimParked(vaddr, size);
	} else {
		VirtualUnlock(reinterpret_cast<void*>(vaddr), size);
	}
}
#endif
