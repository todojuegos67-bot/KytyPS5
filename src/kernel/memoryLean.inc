// Lean's memory API that upstream's memory.cpp does not have (session 61, 2026-11-12), kept in its own file so that
// upstream's kernel commits apply to memory.cpp as upstream wrote them. Included by memory.cpp inside its namespace.
// The full port of upstream's kernel block starts these as forwarders; lean's features come back on upstream's store
// one commit at a time (side-session-notes/61-kernel-plan.md).

bool TryReadBackingCached(uint64_t vaddr, void* data, uint64_t size) {
	return TryReadBacking(vaddr, data, size);
}

bool TryReadGpuCleanBytes(uint64_t vaddr, void* data, uint64_t size) {
	if (g_gpu_resources == nullptr || size == 0 || !IsGpuAddressRange(vaddr, size) || !Graphics::GuestGpu::IsGpuThread()) {
		return false;
	}
	auto& buffers = GetGpuResources().GetBufferCache();
	// A page no shader wrote does not fault; bytes a shader wrote need their read back.
	if (!buffers.IsRegionGpuModified(vaddr, size) || buffers.HasGpuDirtyBytes(vaddr, size)) {
		return false;
	}
	return TryReadBacking(vaddr, data, size);
}

const uint8_t* FindGpuCleanBacking(uint64_t vaddr, uint64_t size) {
	if (g_guest_address_space == nullptr) {
		return nullptr;
	}
	if (g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size)) {
		if (!Graphics::GuestGpu::IsGpuThread() || GetGpuResources().GetBufferCache().HasGpuDirtyBytes(vaddr, size)) {
			return nullptr;
		}
	}
	return g_guest_address_space->FindBacking(vaddr, size);
}

// Lean (b68e4e42f): for an address read from data that may be stale (the compute-ahead scanner). Backed pages lie in
// committed ranges (upstream's fast path, as in ClampRangeSize), and the page table bounds-checks any address; otherwise
// the range walk under the state lock shared.
bool IsCommittedRange(uint64_t vaddr, uint64_t size) {
	if (g_virtual_ranges == nullptr) {
		return false;
	}
	if (g_guest_address_space != nullptr && g_guest_address_space->IsBacked(vaddr, size)) {
		return true;
	}
	std::shared_lock state(g_memory_state_mutex);
	return g_virtual_ranges->ClampRangeSize(vaddr, size) == size;
}

// Lean (7ab7e1b55): upstream's ClampRangeSize, through reserved ranges too when asked (unmapped pages there read as
// zero), with upstream's fast path: a fully backed range is committed either way.
uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size, bool through_reserved) {
	if (!through_reserved) {
		return ClampRangeSize(vaddr, size);
	}
	if (size <= (uint64_t {1} << 24u) && g_guest_address_space->IsBacked(vaddr, size)) {
		return size;
	}
	std::shared_lock state(g_memory_state_mutex);
	const auto       clamped_size = g_virtual_ranges->ClampRangeSize(vaddr, size, true);
	state.unlock();
	if (clamped_size == 0) {
		EXIT("Memory: attempted to access invalid address 0x%016" PRIx64 " with size 0x%016" PRIx64 "\n", vaddr, size);
	}
	if (clamped_size != size) {
		LOGF("Memory: clamped buffer range addr=0x%016" PRIx64 " size=0x%016" PRIx64 " to 0x%016" PRIx64 "\n", vaddr,
		     size, clamped_size);
	}
	return clamped_size;
}

// Commit on first touch (lean 438cb01c5, fb82648a3; Windows, on by default, KYTY_FIRST_TOUCH=0 turns it off) with lazy
// protection (08d26ed16, c7f883880) and the second view committed first (4ece93a3f), on upstream's store.
void EnableFirstTouchCommit() {
	const char* value    = std::getenv("KYTY_FIRST_TOUCH");
	g_first_touch_commit = value == nullptr || value[0] != '0';
}

bool CommitOnFault(uint64_t vaddr) noexcept {
	return g_guest_address_space != nullptr && g_guest_address_space->CommitOnFault(vaddr);
}

bool HandleFirstTouch(uint64_t vaddr, bool write, bool execute) noexcept {
	return g_guest_address_space != nullptr && g_guest_address_space->HandleFirstTouch(vaddr, write, execute);
}

// Never under the state lock (it may wait for a make-ready, which takes it shared).
void CommitGuestRange(uint64_t vaddr, uint64_t size) {
	if (g_guest_address_space != nullptr) g_guest_address_space->CommitGuestRange(vaddr, size);
}

FirstTouchStats GetFirstTouchStats() {
	if (g_guest_address_space == nullptr || !g_guest_address_space->IsFirstTouch()) return {};
	return {true, g_guest_address_space->CommittedBytes(), g_guest_address_space->FirstTouchFaults()};
}

// Lazy protection: a guest view's piece of a granule just committed gets the guest's protection, from the virtual
// ranges (the view's map mode where they give none), and the tracker's state on top, in one protect per run
// (PageManager::ApplyHostProtection).
struct GuestModes {
	std::vector<VirtualRanges::Range> ranges;
	VirtualMemory::Mode               map_mode;
};

static VirtualMemory::Mode GuestModeAt(uint64_t vaddr, void* context) {
	const auto* modes = static_cast<const GuestModes*>(context);
	for (const auto& range: modes->ranges) {
		VirtualMemory::Mode mode = modes->map_mode;
		if (vaddr - range.start < range.size && IsCommittedRangeType(range.type) &&
		    DecodeMemoryProtection(range.protection, &mode)) {
			return mode;
		}
	}
	return modes->map_mode;
}

// A guest mprotect (host protect and virtual-ranges update) and a make-ready (guest mode read, then protect) never
// interleave: otherwise make-ready could apply the old mode after the guest's protect, and leave a page more open than
// the guest asked. Lock order: this one, then the tracker's locks, then the state lock; never the reverse.
static std::recursive_mutex g_guest_protect_mutex;

// From a fault on a guest view (GuestBackingStore::MakeReady), with no lock of the store held.
static void MakeGranuleReady(uint64_t vaddr, uint64_t size, VirtualMemory::Mode map_mode) {
	std::lock_guard lock(g_guest_protect_mutex);
	GuestModes      modes {{}, map_mode};
	if (g_virtual_ranges != nullptr) {
		// Only around the read: the protect below goes through the tracker, which takes the state lock itself.
		std::shared_lock state(g_memory_state_mutex);
		g_virtual_ranges->QuerySpan(vaddr, size, &modes.ranges);
	}
	if (g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size)) {
		GetGpuResources().ApplyHostProtection(vaddr, size, GuestModeAt, &modes);
		return;
	}
	for (uint64_t page = vaddr; page < vaddr + size;) {
		const auto mode = GuestModeAt(page, &modes);
		auto       next = page + 4096u;
		while (next < vaddr + size && GuestModeAt(next, &modes) == mode) next += 4096u;
		VirtualMemory::Protect(page, next - page, mode);
		page = next;
	}
}

// Known zero (lean 494a5d9fa, 00504e03a, 10d1948a0): direct memory no mapping has had yet reads as zeros, so the
// buffer cache uploads it as zeros instead of reading it, until a write ends that; a mapping that aliases another's
// pages ends it for the other. The store keeps the bookkeeping beside its page table.
static std::atomic<bool> g_known_zero_enabled {false};
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
static void (*g_test_gpu_zero)(bool, uint64_t, size_t) = nullptr;
#endif

// From MapGpuRange, after the operation's state block.
static void MarkFreshGpuRange(uint64_t vaddr, uint64_t size) {
	if (g_guest_address_space == nullptr) {
		return;
	}
	std::vector<std::pair<uint64_t, uint64_t>> aliases;
	std::vector<std::pair<uint64_t, uint64_t>> fresh;
	{
		// The bookkeeping changes under the state lock; the buffer cache hears of it after (no graphics call under it).
		std::lock_guard state(g_memory_state_mutex);
		aliases = g_guest_address_space->FindBackingAliases(vaddr, size);
		fresh   = g_guest_address_space->TakeFreshRuns(vaddr, size, PhysicalMemory::Size());
	}
	if (!g_known_zero_enabled.load(std::memory_order_acquire)) {
		fresh.clear(); // taken, so not marked later either
	}
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	if (g_test_gpu_zero != nullptr) {
		for (const auto& [alias, bytes]: aliases) g_test_gpu_zero(false, alias, bytes);
		for (const auto& [run, bytes]: fresh) g_test_gpu_zero(true, run, bytes);
	}
#endif
	if (g_gpu_resources == nullptr) {
		return;
	}
	auto& buffers = GetGpuResources().GetBufferCache();
	for (const auto& [alias, bytes]: aliases) {
		if (IsGpuAddressRange(alias, bytes)) {
			buffers.UnmarkMemoryZero(alias, bytes);
		}
	}
	for (const auto& [run, bytes]: fresh) {
		if (IsGpuAddressRange(run, bytes)) {
			buffers.MarkMemoryZero(run, bytes);
		}
	}
}

// Known-zero marking needs the fault handler that ends it: before the loader installs it, a host write to a
// write-protected fresh page would be fatal.
void EnableKnownZero() noexcept {
	g_known_zero_enabled.store(true, std::memory_order_release);
}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
void TestSetGpuZeroCallback(void (*callback)(bool, uint64_t, size_t)) {
	g_test_gpu_zero = callback;
}

void TestForgetMappedPages() {
	EXIT_IF(g_guest_address_space == nullptr);
	g_guest_address_space->TestForgetMappedPages();
}
#endif

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// Lazy protection on a standalone store with a second guest view: the make-ready callback runs with no store lock held,
// sees the view's pages committed but inaccessible, and its protection is what the view ends with.
static GuestBackingStore* g_test_lazy_store    = nullptr;
static int                g_test_lazy_calls    = 0;
static bool               g_test_lazy_unlocked = true;
static DWORD              g_test_lazy_seen     = 0;

static DWORD TestProtectionAt(uint64_t vaddr) {
	MEMORY_BASIC_INFORMATION info {};
	if (VirtualQuery(reinterpret_cast<void*>(vaddr), &info, sizeof(info)) == 0) return 0xffffffffu;
	return info.State == MEM_COMMIT ? info.Protect : 0;
}

static void TestLazyMakeReady(uint64_t vaddr, uint64_t size, VirtualMemory::Mode /*map_mode*/) {
	g_test_lazy_calls++;
	g_test_lazy_unlocked = g_test_lazy_unlocked && g_test_lazy_store->TestLocksFree();
	g_test_lazy_seen     = TestProtectionAt(vaddr);
	VirtualMemory::Protect(vaddr, size, VirtualMemory::Mode::Read);
}

static const char* TestLazyProtection() {
	constexpr uint64_t MiB = 1024u * 1024u;
	GuestBackingStore  store(16 * MiB, true);
	g_test_lazy_store = &store;
	g_test_lazy_calls = 0;
	store.SetMakeReady(TestLazyMakeReady);
	const auto alias = store.GetBackingBase();
	const auto view  = store.TestMapView(4 * MiB, 4 * MiB); // granules 2 and 3
	if (view == 0) return "test view not mapped";
	const char* result = [&]() -> const char* {
		if (TestProtectionAt(view) != 0) return "view committed at start";
		if (!store.ProtectCommittedRuns(view, 4 * MiB, 4 * MiB, VirtualMemory::Mode::NoAccess) ||
		    TestProtectionAt(view) != 0) {
			return "a protect of reserved granules did not skip them";
		}
		if (!store.CommitOnFault(view + 2 * MiB + 100)) return "view first touch not handled";
		if (g_test_lazy_calls != 1 || !g_test_lazy_unlocked) return "make ready not called once without store locks";
		if (g_test_lazy_seen != PAGE_NOACCESS) return "view usable before it was ready";
		if (TestProtectionAt(view + 2 * MiB) != PAGE_READONLY) return "view did not end with the make-ready protection";
		if (TestProtectionAt(alias + 6 * MiB) != PAGE_READWRITE) return "alias not read-write";
		if (!store.TestIsReady(3) || store.CommitOnFault(view + 2 * MiB)) return "ready granule taken again";
		if (TestProtectionAt(view) != 0) return "the other granule committed";
		// Through the alias: committed, the view inaccessible until a fault on it makes it ready.
		if (!store.CommitOnFault(alias + 4 * MiB) || g_test_lazy_calls != 1) return "alias first touch made ready";
		if (TestProtectionAt(view) != PAGE_NOACCESS || store.TestIsReady(2)) return "alias commit opened the view";
		if (!store.CommitOnFault(view + 8) || g_test_lazy_calls != 2 || TestProtectionAt(view) != PAGE_READONLY) {
			return "a view fault after an alias commit did not make ready";
		}
		if (!store.ProtectCommittedRuns(view, 4 * MiB, 4 * MiB, VirtualMemory::Mode::ReadWrite) ||
		    TestProtectionAt(view) != PAGE_READWRITE || TestProtectionAt(view + 2 * MiB) != PAGE_READWRITE) {
			return "a protect of committed granules";
		}
		return "";
	}();
	store.TestUnmapView(view);
	g_test_lazy_store = nullptr;
	return result;
}

// A second view over a granule not committed yet commits it first: the first view stays inaccessible, nothing is made
// ready, and a fault on the first view then makes both views ready.
static const char* TestSecondView() {
	constexpr uint64_t MiB = 1024u * 1024u;
	GuestBackingStore  store(16 * MiB, true);
	g_test_lazy_store = &store;
	g_test_lazy_calls = 0;
	store.SetMakeReady(TestLazyMakeReady);
	const auto first  = store.TestMapView(4 * MiB, 4 * MiB); // granules 2 and 3
	const auto second = store.TestMapView(6 * MiB, 2 * MiB); // granule 3
	const char* result = [&]() -> const char* {
		if (first == 0 || second == 0) return "test views not mapped";
		if (TestProtectionAt(first) != 0) return "a granule with one view committed";
		if (TestProtectionAt(first + 2 * MiB) != PAGE_NOACCESS) return "second view: first view not committed inaccessible";
		if (TestProtectionAt(second) != PAGE_READWRITE) return "second view not mapped over committed memory";
		if (g_test_lazy_calls != 0 || store.TestIsReady(3)) return "second view made the granule ready";
		if (!store.CommitOnFault(first + 2 * MiB) || g_test_lazy_calls != 2) return "a fault did not make both views ready";
		if (TestProtectionAt(first + 2 * MiB) != PAGE_READONLY || TestProtectionAt(second) != PAGE_READONLY) {
			return "views did not end with the make-ready protection";
		}
		return "";
	}();
	if (second != 0) store.TestUnmapView(second);
	if (first != 0) store.TestUnmapView(first);
	g_test_lazy_store = nullptr;
	return result;
}

// Upstream's partial unmap re-maps the survivors and re-applies the protections it read from them (59723ae19). On a
// first-touch view (granule 0 reserved, 1 ready, 2 committed through the alias but not ready) that read skips the
// reserved pages, which stay reserved in both survivors; the ready granule keeps its protection and the other stays
// inaccessible until a fault makes it ready.
static const char* TestFirstTouchPartialUnmap() {
	constexpr uint64_t MiB  = 1024u * 1024u;
	constexpr uint64_t Hole = 16u * 1024u;
	GuestBackingStore  store(16 * MiB, true);
	g_test_lazy_store = &store;
	g_test_lazy_calls = 0;
	store.SetMakeReady(TestLazyMakeReady);
	void* placeholder = VirtualAlloc2(GetCurrentProcess(), nullptr, 6 * MiB, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
	                                  PAGE_NOACCESS, nullptr, 0);
	if (placeholder == nullptr) return "placeholder not reserved";
	const auto base = reinterpret_cast<uint64_t>(placeholder);
	if (!store.MapFixed(base, 6 * MiB, 0, VirtualMemory::Mode::ReadWrite)) return "view not mapped";
	const char* result = [&]() -> const char* {
		if (!store.CommitOnFault(base + 2 * MiB + 8) || TestProtectionAt(base + 2 * MiB) != PAGE_READONLY) {
			return "granule 1 not made ready";
		}
		if (!store.CommitOnFault(store.GetBackingBase() + 4 * MiB) || TestProtectionAt(base + 4 * MiB) != PAGE_NOACCESS) {
			return "granule 2 not committed inaccessible";
		}
		store.Unmap(base + MiB, Hole); // survivors [0, 1 MiB) and [1 MiB + 16 KiB, 6 MiB), both over granule 0
		if (TestProtectionAt(base) != 0 || TestProtectionAt(base + MiB + Hole) != 0) return "a reserved granule committed";
		if (TestProtectionAt(base + 2 * MiB) != PAGE_READONLY) return "a ready granule lost its protection";
		if (TestProtectionAt(base + 4 * MiB) != PAGE_NOACCESS || store.TestIsReady(2)) {
			return "a granule not ready became usable";
		}
		if (!store.CommitOnFault(base + MiB + Hole) || g_test_lazy_calls != 3) return "a fault did not make both survivors ready";
		if (TestProtectionAt(base) != PAGE_READONLY || TestProtectionAt(base + MiB + Hole) != PAGE_READONLY) {
			return "survivors did not end with the make-ready protection";
		}
		if (!store.CommitOnFault(base + 4 * MiB) || g_test_lazy_calls != 4 || TestProtectionAt(base + 4 * MiB) != PAGE_READONLY) {
			return "a fault on a survivor did not make its granule ready";
		}
		return "";
	}();
	store.Unmap(base, MiB);
	store.Unmap(base + MiB + Hole, 5 * MiB - Hole);
	VirtualFree(placeholder, 6 * MiB, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS);
	VirtualFree(placeholder, 0, MEM_RELEASE);
	g_test_lazy_store = nullptr;
	return result;
}
#endif

// First touch on a standalone 16 MB store, through its alias. Empty on success, else what failed.
const char* TestFirstTouchStore() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	constexpr uint64_t MiB = 1024u * 1024u;
	GuestBackingStore  store(16 * MiB, true);
	if (store.CommittedBytes() != 0) return "committed at start";
	const auto alias = store.GetBackingBase();
	if (!store.CommitOnFault(alias + 3 * MiB + 5)) return "first touch not handled";
	if (store.CommittedBytes() != 2 * MiB || store.FirstTouchFaults() != 1) return "not one 2 MB granule";
	auto* byte = reinterpret_cast<volatile uint8_t*>(alias + 3 * MiB + 5);
	*byte      = 0x5a;
	if (*byte != 0x5a) return "write lost";
	if (store.CommitOnFault(alias + 3 * MiB + 5)) return "a committed granule taken before the GPU path";
	if (!store.HandleFirstTouch(alias + 3 * MiB + 5, true, false)) return "a committed page's retry refused";
	if (!store.HandleFirstTouch(alias + 7 * MiB, false, false) || store.CommittedBytes() != 4 * MiB) return "late first touch";
	store.ZeroRange(0, 16 * MiB);
	if (store.CommittedBytes() != 4 * MiB) return "zeroing committed more";
	if (*byte != 0) return "not zeroed";
	store.EnsureCommitted(10 * MiB, 4 * MiB);
	if (store.CommittedBytes() != 8 * MiB) return "EnsureCommitted";
	if (store.HandleFirstTouch(0x1000, false, false) || store.CommitOnFault(0x1000)) return "an address outside handled";
	if (const char* lazy = TestLazyProtection(); *lazy != '\0') return lazy;
	if (const char* second = TestSecondView(); *second != '\0') return second;
	return TestFirstTouchPartialUnmap();
#else
	return "";
#endif
}
#endif

void ReleaseDriverAddressRoom() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (g_driver_room != nullptr) {
		VirtualFree(g_driver_room, 0, MEM_RELEASE);
		g_driver_room = nullptr;
	}
#endif
}
