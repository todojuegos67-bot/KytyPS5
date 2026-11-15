// Internal memory address-space helpers included by memory.cpp.
// Kept in the same translation unit because they depend on memory.cpp-local helpers.

static constexpr uint64_t HOST_SYSTEM_MANAGED_MIN  = 0x0000040000ull;
static constexpr uint64_t HOST_SYSTEM_MANAGED_MAX  = 0x07ffffbfffull;
#if defined(__APPLE__) && (defined(__x86_64__) || defined(_M_X64))
static constexpr uint64_t HOST_SYSTEM_RESERVED_MAX = 0x0fbfffffffull;
static constexpr uint64_t HOST_USER_MIN            = 0x7000000000ull;
#else
static constexpr uint64_t HOST_SYSTEM_RESERVED_MAX = 0x0fffffffffull;
static constexpr uint64_t HOST_USER_MIN            = 0x1000000000ull;
#endif
static constexpr uint64_t HOST_USER_MAX = 0xfbffffffffull;

// The room below 1 TB kept for the GPU driver's CUDA side when DLSS is asked for (ReserveGuestRegions), held until the
// Vulkan device is made (ReleaseDriverAddressRoom).
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
static void* g_driver_room = nullptr;
#endif

static bool GuestFreeRangeContains(uint64_t base, uint64_t range_size, uint64_t candidate,
                                   uint64_t size) {
	const auto range_end = base + range_size;
	return candidate <= range_end && size <= range_end - candidate;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
static bool CoalescePlaceholders(uint64_t vaddr, uint64_t size) {
	return VirtualFree(reinterpret_cast<void*>(vaddr), size,
	                   MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) != 0;
}
#endif

class GuestBackingStore {
public:
	enum class FailureReason {
		None,
		FixedMapViewFailed,
		PlaceholderMapFailed,
		ProtectFailed,
		AddressRangeUnavailable,
	};

	explicit GuestBackingStore(uint64_t size, [[maybe_unused]] bool first_touch = false): m_size(size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		// Lean (438cb01c5): SEC_COMMIT charges the whole section to the commit limit at start, ~5 GB of it never touched by
		// a game. With first touch it is reserved, and each CommitGranule is committed when first touched (CommitOnFault),
		// when a range is protected (ProtectViews) or before a kernel-mode copy (CommitViews).
		m_first_touch = first_touch;
		m_committed   = std::make_unique<std::atomic<uint64_t>[]>((m_size / CommitGranule + 64u) / 64u);
		m_ready       = std::make_unique<std::atomic<uint64_t>[]>((m_size / CommitGranule + 64u) / 64u);
		m_readying    = std::make_unique<std::atomic<uint64_t>[]>((m_size / CommitGranule + 64u) / 64u);
		m_handle      = CreateFileMappingA(
		    INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE | (m_first_touch ? SEC_RESERVE : SEC_COMMIT),
		    static_cast<DWORD>(m_size >> 32u), static_cast<DWORD>(m_size & 0xffffffffu), nullptr);
		if (m_handle == nullptr) {
			EXIT("could not reserve %" PRIu64 " MB for guest direct memory: 0x%08" PRIx32
			     ". Windows commits\n"
			     "this up front, so the paging file is usually what needs to be larger.\n",
			     m_size / (1024U * 1024U), static_cast<uint32_t>(GetLastError()));
		}
		m_backing_base = static_cast<uint8_t*>(
		    MapViewOfFile(m_handle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, m_size));
		if (m_backing_base == nullptr) {
			EXIT("direct-memory backing: writable alias failed: 0x%08" PRIx32 "\n",
			     static_cast<uint32_t>(GetLastError()));
		}
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
			EXIT("direct-memory backing: shared fd creation failed: %d\n", errno);
		}
		if (ftruncate(m_fd, static_cast<off_t>(m_size)) != 0) {
			EXIT("direct-memory backing: ftruncate to %" PRIu64 " MB failed: %d\n",
			     m_size / (1024U * 1024U), errno);
		}
		auto* backing = mmap(nullptr, m_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);
		if (backing == MAP_FAILED) {
			EXIT("direct-memory backing: writable alias failed: %d\n", errno);
		}
		m_backing_base = static_cast<uint8_t*>(backing);
#endif
	}

	// Guest views outlive the alias and the handle; the address-space owner releases them.
	~GuestBackingStore() {
		for (auto& bucket: m_pages) {
			delete[] bucket.load(std::memory_order_relaxed);
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		UnmapViewOfFile(m_backing_base);
		CloseHandle(m_handle);
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		munmap(m_backing_base, m_size);
		close(m_fd);
#endif
	}

	KYTY_CLASS_NO_COPY(GuestBackingStore);

	static const char* GetFailureReasonName(FailureReason reason) {
		switch (reason) {
			case FailureReason::None: return "n/a";
			case FailureReason::FixedMapViewFailed: return "fixed-map-view-failed";
			case FailureReason::PlaceholderMapFailed: return "placeholder-map-failed";
			case FailureReason::ProtectFailed: return "protect-failed";
			case FailureReason::AddressRangeUnavailable: return "address-range-unavailable";
			default: return "unknown";
		}
	}

	[[nodiscard]] uint64_t GetBackingBase() const {
		return reinterpret_cast<uint64_t>(m_backing_base);
	}

	[[nodiscard]] uint64_t GetBackingSize() const { return m_size; }

	void ZeroRange(uint64_t backing_offset, uint64_t size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (m_first_touch) {
			// A granule not committed yet reads 0 when it is; writing it would commit it.
			const auto end = backing_offset + size;
			for (auto offset = backing_offset; offset < end;) {
				const auto granule_end = std::min(end, (offset / CommitGranule + 1u) * CommitGranule);
				if (IsCommitted(offset / CommitGranule)) {
					std::memset(m_backing_base + offset, 0, granule_end - offset);
				}
				offset = granule_end;
			}
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
			g_test_backing_bytes_zeroed += size;
#endif
			return;
		}
#endif
		std::memset(m_backing_base + backing_offset, 0, size);
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		g_test_backing_bytes_zeroed += size;
#endif
	}

	// Callers hold the state lock, so no view changes under the write. Unbacked gaps are skipped,
	// mirroring sparse reads that zero-fill them.
	void WriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
		const auto* source = static_cast<const uint8_t*>(data);
		ForEachRun(
		    vaddr, size,
		    [&](uint64_t offset, uint64_t start, uint64_t bytes) {
			    std::memcpy(m_backing_base + offset, source + (start - vaddr), bytes);
		    },
		    [](uint64_t, uint64_t) {});
	}

	// Needs no lock: page entries are atomic and buckets live as long as the store.
	[[nodiscard]] bool IsBacked(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		for (auto page = vaddr >> PageBits; page <= (vaddr + size - 1u) >> PageBits; ++page) {
			if (LoadPage(page) == 0) {
				return false;
			}
		}
		return true;
	}

	// Needs no lock either, as the alias stays mapped. A read racing an unmap copies the backing's
	// current bytes, or zeros for a page unmapped between the check and the copy.
	bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) const {
		if (data == nullptr || !IsBacked(vaddr, size)) {
			return false;
		}
		ReadSparseBacking(vaddr, data, size, [](uint64_t, uint8_t* destination, uint64_t bytes) {
			std::memset(destination, 0, bytes);
		});
		return true;
	}

	// Lean (ca980f90f): the backing bytes of [vaddr, vaddr + size) when contiguous backing holds all of them, else
	// nullptr. Needs no lock, as IsBacked; the bytes stay this range's until the guest remaps it.
	[[nodiscard]] const uint8_t* FindBacking(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return nullptr;
		}
		const auto first = vaddr >> PageBits;
		const auto entry = LoadPage(first);
		if (entry == 0) {
			return nullptr;
		}
		for (auto page = first + 1u; page <= (vaddr + size - 1u) >> PageBits; ++page) {
			if (LoadPage(page) != entry + (page - first)) {
				return nullptr;
			}
		}
		return m_backing_base + (uint64_t {entry - 1u} << PageBits) + (vaddr & PageMask);
	}

	template <typename Gap>
	void ReadSparseBacking(uint64_t vaddr, void* data, uint64_t size, Gap&& gap) const {
		auto* destination = static_cast<uint8_t*>(data);
		ForEachRun(
		    vaddr, size,
		    [&](uint64_t offset, uint64_t start, uint64_t bytes) {
			    std::memcpy(destination + (start - vaddr), m_backing_base + offset, bytes);
		    },
		    [&](uint64_t start, uint64_t bytes) {
			    gap(start, destination + (start - vaddr), bytes);
		    });
	}

	// Lean's known-zero bookkeeping (494a5d9fa, 00504e03a, 10d1948a0), changed with the page table in MapFixed, Unmap and
	// UnmapOne; callers hold the state lock exclusively, as for mapping.
	// The runs of [vaddr, vaddr + size) whose backing pages, below backing_limit, no mapping had before
	// theirs: they read as zeros until written. Each run is returned once.
	std::vector<std::pair<uint64_t, uint64_t>> TakeFreshRuns(uint64_t vaddr, uint64_t size,
	                                                         uint64_t backing_limit) {
		std::vector<std::pair<uint64_t, uint64_t>> runs;
		std::erase_if(m_fresh, [&](const MapEntry& entry) {
			const auto begin = std::max(vaddr, entry.vaddr);
			const auto end   = std::min(vaddr + size, entry.vaddr + entry.size);
			if (begin >= end) {
				return false;
			}
			if (entry.backing_offset + entry.size <= backing_limit) {
				runs.emplace_back(begin, end - begin);
			}
			return true;
		});
		return runs;
	}

	// The other mappings of the backing pages that [vaddr, vaddr + size) maps.
	std::vector<std::pair<uint64_t, uint64_t>> FindAliases(uint64_t vaddr, uint64_t size) const {
		std::vector<std::pair<uint64_t, uint64_t>> aliases;
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return aliases;
		}
		// No backing page of the range in a second mapping: nothing to find, without the two walks over all
		// the mappings (Marvel's Wolverine: ~12,000-19,000 of them, ~0.3 ms a call, ~1,400 calls a second in
		// the white transition, and none found an alias). The page table gives the range's backing.
		if (m_map_counts_exact) {
			bool shared = false;
			ForEachRun(
			    vaddr, size,
			    [&](uint64_t offset, uint64_t, uint64_t bytes) {
				    for (auto page = offset / MapCountPage; page <= (offset + bytes - 1) / MapCountPage && !shared;
				         page++) {
					    shared = page < m_map_counts.size() && m_map_counts[page] > 1;
				    }
			    },
			    [](uint64_t, uint64_t) {});
			if (!shared) {
				return aliases;
			}
		}
		for (const auto& [start, entry]: m_maps) {
			const auto begin = std::max(vaddr, entry.vaddr);
			const auto end   = std::min(vaddr + size, entry.vaddr + entry.size);
			if (begin >= end) {
				continue;
			}
			const auto backing_begin = entry.backing_offset + (begin - entry.vaddr);
			const auto backing_end   = backing_begin + (end - begin);
			for (const auto& [other_start, other]: m_maps) {
				if (other.vaddr < vaddr + size && vaddr < other.vaddr + other.size) {
					continue; // inside the range itself
				}
				const auto overlap_begin = std::max(backing_begin, other.backing_offset);
				const auto overlap_end   = std::min(backing_end, other.backing_offset + other.size);
				if (overlap_begin < overlap_end) {
					aliases.emplace_back(other.vaddr + (overlap_begin - other.backing_offset),
					                     overlap_end - overlap_begin);
				}
			}
		}
		return aliases;
	}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	void TestForgetMappedPages() {
		std::fill(m_mapped_once.begin(), m_mapped_once.end(), uint64_t {0});
		m_fresh.clear();
	}
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Lean (438cb01c5, fb82648a3, 08d26ed16, 4ece93a3f): commit on first touch with lazy protection.
	static constexpr uint64_t CommitGranule = 2u * 1024u * 1024u;

	[[nodiscard]] bool     IsFirstTouch() const { return m_first_touch; }
	[[nodiscard]] uint64_t CommittedBytes() const { return m_committed_bytes.load(std::memory_order_relaxed); }
	[[nodiscard]] uint64_t FirstTouchFaults() const { return m_first_touch_faults.load(std::memory_order_relaxed); }

	// Lazy protection: a granule not committed is reserved, which faults on every access like a protect, so protects skip it
	// (ProtectCommittedRuns). It is committed through each guest view with PAGE_NOACCESS (a view's own protection; the alias
	// comes up read-write), then made ready: every view gets the guest's protection and the page tracker's state back
	// (m_make_ready, set by memory.cpp), with no lock of the store or the state lock held. Until then its views stay
	// inaccessible, and a fault on them waits for the thread making it ready.
	using MakeReadyFn = void (*)(uint64_t vaddr, uint64_t size, VirtualMemory::Mode map_mode);
	void SetMakeReady(MakeReadyFn fn) { m_make_ready = fn; }

	// Commits and makes ready the granules of [backing_offset, +size). Never under the state lock: it waits for another
	// thread's make-ready, which takes the state lock shared.
	void EnsureCommitted(uint64_t backing_offset, uint64_t size) {
		if (!m_first_touch || size == 0 || backing_offset >= m_size || m_size - backing_offset < size) {
			return;
		}
		for (auto granule = backing_offset / CommitGranule; granule <= (backing_offset + size - 1u) / CommitGranule;
		     ++granule) {
			CommitGranuleOnce(granule);
			while (!IsReady(granule)) {
				if (!MakeReady(granule)) std::this_thread::yield();
			}
		}
	}

	// The guest views and the alias under [vaddr, +size), committed and ready (before a kernel-mode copy, which fails on a
	// reserved or inaccessible page instead of faulting). The backing comes from the page table, which needs no lock.
	void CommitViews(uint64_t vaddr, uint64_t size) {
		if (!m_first_touch || size == 0 || UINT64_MAX - vaddr < size) {
			return;
		}
		uint64_t offset = 0;
		if (AliasOffset(vaddr, &offset)) {
			EnsureCommitted(offset, std::min(size, m_size - offset));
			return;
		}
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
		ForEachRun(
		    vaddr, size, [&](uint64_t backing_offset, uint64_t, uint64_t bytes) { ranges.emplace_back(backing_offset, bytes); },
		    [](uint64_t, uint64_t) {});
		for (const auto& [backing_offset, bytes]: ranges) EnsureCommitted(backing_offset, bytes);
	}

	// First in the exception handler, before the GPU fault path (which takes any fault in a GPU-mapped range for a tracking
	// fault and retries it). A granule not committed is committed; a view's granule not ready is made ready (or waited for).
	// The alias never makes ready: emulator code writes it under the state lock shared, and making ready takes the
	// tracker's region locks and the state lock. False for anything else.
	bool CommitOnFault(uint64_t vaddr) {
		uint64_t   offset = 0;
		const bool alias  = AliasOffset(vaddr, &offset);
		if (!m_first_touch || (!alias && !ViewOffset(vaddr, &offset))) {
			return false;
		}
		const auto granule = offset / CommitGranule;
		if (IsReady(granule) || (alias && IsCommitted(granule))) {
			return false;
		}
		if (CommitGranuleOnce(granule)) {
			m_first_touch_faults.fetch_add(1, std::memory_order_relaxed);
		}
		if (!alias) MakeReady(granule);
		return true;
	}

	// Protects the runs of committed granules under a view range [vaddr, +size) at backing_offset; skips the rest (reserved).
	bool ProtectCommittedRuns(uint64_t vaddr, uint64_t size, uint64_t backing_offset, VirtualMemory::Mode mode) {
		if (!m_first_touch) {
			return VirtualMemory::Protect(vaddr, size, mode);
		}
		bool ok = true;
		for (uint64_t done = 0; done < size;) {
			const auto granule = (backing_offset + done) / CommitGranule;
			const auto bytes   = std::min(size - done, (granule + 1u) * CommitGranule - (backing_offset + done));
			if (IsCommitted(granule)) {
				ok = VirtualMemory::Protect(vaddr + done, bytes, mode) && ok;
			}
			done += bytes;
		}
		return ok;
	}

	// A guest protect over mapped views: VirtualProtect fails on a reserved page, so only committed granules are protected.
	// One view per call (VirtualProtect can't cross views): the caller passes one m_mapped region at a time.
	bool ProtectViews(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		bool ok = true;
		ForEachRun(
		    vaddr, size,
		    [&](uint64_t backing_offset, uint64_t start, uint64_t bytes) {
			    ok = ProtectCommittedRuns(start, bytes, backing_offset, mode) && ok;
		    },
		    [&](uint64_t, uint64_t) { ok = false; });
		return ok;
	}

	// Unit tests: a guest view of [offset, +size) anywhere, read-write; whether a granule is ready; no store lock held.
	uint64_t TestMapView(uint64_t offset, uint64_t size) {
		const auto place = CommitSharedGranules(offset, size);
		void*      ptr   = MapViewOfFile3(m_handle, GetCurrentProcess(), nullptr, offset, size, 0, PAGE_READWRITE, nullptr, 0);
		if (ptr == nullptr) return 0;
		std::lock_guard lock(m_view_mutex);
		m_views[reinterpret_cast<uint64_t>(ptr)] = {size, offset, VirtualMemory::Mode::ReadWrite};
		return reinterpret_cast<uint64_t>(ptr);
	}
	void TestUnmapView(uint64_t vaddr) {
		{
			std::lock_guard lock(m_view_mutex);
			m_views.erase(vaddr);
		}
		UnmapViewOfFile(reinterpret_cast<void*>(vaddr));
	}
	[[nodiscard]] bool TestIsReady(uint64_t granule) const { return IsReady(granule); }
	bool               TestLocksFree() {
		if (!m_view_mutex.try_lock()) return false;
		m_view_mutex.unlock();
		if (!m_commit_mutex.try_lock()) return false;
		m_commit_mutex.unlock();
		return true;
	}

	bool HandleFirstTouch(uint64_t vaddr, bool write, bool execute) {
		if (!m_first_touch) {
			return false;
		}
		uint64_t   offset = 0;
		const bool alias  = AliasOffset(vaddr, &offset);
		if (!alias && !ViewOffset(vaddr, &offset)) {
			return false;
		}
		const auto granule = offset / CommitGranule;
		if (!IsCommitted(granule)) {
			if (CommitGranuleOnce(granule)) {
				m_first_touch_faults.fetch_add(1, std::memory_order_relaxed);
			}
			if (!alias) MakeReady(granule);
			return true;
		}
		if (!alias && !IsReady(granule)) {
			MakeReady(granule);
			return true;
		}
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<void*>(vaddr), &info, sizeof(info)) == 0 || info.State != MEM_COMMIT ||
		    (info.Protect & PAGE_GUARD) != 0) {
			return false;
		}
		const DWORD p = info.Protect & 0xffu;
		if (execute) return p == PAGE_EXECUTE || p == PAGE_EXECUTE_READ || p == PAGE_EXECUTE_READWRITE;
		if (write) return p == PAGE_READWRITE || p == PAGE_EXECUTE_READWRITE;
		return p != PAGE_NOACCESS && p != 0;
	}
#else
	using MakeReadyFn = void (*)(uint64_t vaddr, uint64_t size, VirtualMemory::Mode map_mode);
	void                   SetMakeReady(MakeReadyFn) {}
	void                   EnsureCommitted(uint64_t, uint64_t) {}
	void                   CommitViews(uint64_t, uint64_t) {}
	bool                   HandleFirstTouch(uint64_t, bool, bool) { return false; }
	bool                   CommitOnFault(uint64_t) { return false; }
	[[nodiscard]] bool     IsFirstTouch() const { return false; }
	[[nodiscard]] uint64_t CommittedBytes() const { return 0; }
	[[nodiscard]] uint64_t FirstTouchFaults() const { return 0; }
#endif

	// Mapping and unmapping hold the state lock exclusively.
	bool MapFixed(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	              VirtualMemory::Mode mode, FailureReason* failure_reason = nullptr) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (const auto callback = g_test_before_backing_map) {
			g_test_before_backing_map = nullptr;
			callback(vaddr, size);
		}
#endif
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (!ReplacePlaceholderView(vaddr, size, backing_offset, mode, failure_reason)) {
			return false;
		}
#else
		if (!MapView(vaddr, size, backing_offset, mode, failure_reason)) {
			return false;
		}
#endif
		StorePages(vaddr, size, backing_offset);
		NoteMapUnlocked(vaddr, size, backing_offset, mode);
		return true;
	}

	// The caller validated the span; failing here is an emulator fault.
	void Unmap(uint64_t vaddr, uint64_t size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		const auto end   = vaddr + size;
		uint64_t   views = 0;
		for (auto current = vaddr; current < end; ++views) {
			const auto view     = QueryView(current);
			const auto part_end = std::min(end, view.base + view.size);
			UnmapOne(view, current, part_end - current);
			current = part_end;
		}
		// Each unmapped file view becomes a separate Windows placeholder. The address-space owner
		// records this contiguous guest span as one free range, so normalize fragmented backing
		// views into one placeholder before publishing it there.
		EXIT_IF(views > 1 && !CoalescePlaceholders(vaddr, size));
#else
		// Replace only the requested range so concurrent accesses and GPU watch
		// protections on the surviving mappings remain intact.
		EXIT_IF(!UnmapViewPreservePlaceholder(reinterpret_cast<void*>(vaddr), size));
		ClearPages(vaddr, size);
		NoteUnmapUnlocked(vaddr, size);
#endif
	}

private:
	static constexpr uint64_t PageBits    = 14;
	static constexpr uint64_t PageMask    = (uint64_t {1} << PageBits) - 1u;
	static constexpr uint64_t BucketPages = uint64_t {1} << 16u;
	static constexpr uint64_t BucketCount =
	    ((kExtendedMemoryBase + kExtendedMemorySize) >> PageBits) / BucketPages;

	// Zero for an unbacked page, otherwise the backing page index plus one.
	[[nodiscard]] uint32_t LoadPage(uint64_t page) const {
		if (page / BucketPages >= BucketCount) {
			return 0;
		}
		const auto* entries = m_pages[page / BucketPages].load(std::memory_order_acquire);
		return entries != nullptr ? entries[page % BucketPages].load(std::memory_order_relaxed) : 0;
	}

	// Walks the span in runs of contiguous backing, as (backing offset, start, bytes), and runs of
	// unbacked pages, as (start, bytes).
	template <typename Region, typename Gap>
	void ForEachRun(uint64_t vaddr, uint64_t size, Region&& region, Gap&& gap) const {
		const auto end = vaddr + size;
		for (auto start = vaddr; start < end;) {
			const auto     entry    = LoadPage(start >> PageBits);
			const uint32_t step     = entry != 0 ? 1 : 0;
			auto           expected = entry + step;
			auto           stop     = (start | PageMask) + 1u;
			while (stop < end && LoadPage(stop >> PageBits) == expected) {
				stop += PageMask + 1u;
				expected += step;
			}
			stop = std::min(stop, end);
			if (entry != 0) {
				region((uint64_t {entry - 1u} << PageBits) + (start & PageMask), start,
				       stop - start);
			} else {
				gap(start, stop - start);
			}
			start = stop;
		}
	}

	void StorePages(uint64_t vaddr, uint64_t size, uint64_t backing_offset) {
		const auto end = vaddr + size;
		EXIT_IF(((vaddr | size | backing_offset) & PageMask) != 0 ||
		        (end >> PageBits) > BucketCount * BucketPages || backing_offset + size > m_size);
		auto entry = static_cast<uint32_t>(backing_offset >> PageBits) + 1u;
		for (auto page = vaddr >> PageBits; page < end >> PageBits; ++page, ++entry) {
			auto& bucket  = m_pages[page / BucketPages];
			auto* entries = bucket.load(std::memory_order_relaxed);
			if (entries == nullptr) {
				entries = new std::atomic<uint32_t>[BucketPages] {};
				bucket.store(entries, std::memory_order_release);
			}
			entries[page % BucketPages].store(entry, std::memory_order_relaxed);
		}
	}

	void ClearPages(uint64_t vaddr, uint64_t size) {
		for (auto page = vaddr >> PageBits; page < (vaddr + size) >> PageBits; ++page) {
			m_pages[page / BucketPages].load(std::memory_order_relaxed)[page % BucketPages].store(
			    0, std::memory_order_relaxed);
		}
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

	// Collects the page runs of the span whose protection differs from the view's mode.
	static void QueryProtections(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode,
	                             std::vector<MEMORY_BASIC_INFORMATION>* out) {
		const auto end = vaddr + size;
		for (auto addr = vaddr; addr < end;) {
			MEMORY_BASIC_INFORMATION info {};
			EXIT_IF(VirtualQuery(reinterpret_cast<void*>(addr), &info, sizeof(info)) == 0);
			info.RegionSize = std::min<SIZE_T>(info.RegionSize, end - addr);
			// Lean (first touch): a granule not committed yet is reserved (Protect 0) and stays so; nothing to re-apply.
			if (info.State == MEM_COMMIT && info.Protect != GetProtect(mode)) {
				out->push_back(info);
			}
			addr += info.RegionSize;
		}
	}

	static bool ApplyProtection(void* ptr, uint64_t size, DWORD protect) {
		DWORD old_protect = 0;
		return VirtualProtect(ptr, size, protect, &old_protect) != 0;
	}

	static bool SplitPlaceholder(uint64_t vaddr, uint64_t size) {
		return VirtualFree(reinterpret_cast<void*>(vaddr), size,
		                   MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) != 0;
	}

	struct View {
		uint64_t base    = 0;
		uint64_t size    = 0;
		DWORD    protect = 0;
	};

	// The OS records a view as the regions sharing its allocation base; guest and tracker
	// protections split it into several.
	static View QueryView(uint64_t vaddr) {
		MEMORY_BASIC_INFORMATION info {};
		EXIT_IF(VirtualQuery(reinterpret_cast<void*>(vaddr), &info, sizeof(info)) == 0 ||
		        info.Type != MEM_MAPPED);
		View view {reinterpret_cast<uint64_t>(info.AllocationBase), 0, info.AllocationProtect};
		auto end = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
		while (VirtualQuery(reinterpret_cast<void*>(end), &info, sizeof(info)) != 0 &&
		       reinterpret_cast<uint64_t>(info.AllocationBase) == view.base) {
			end += info.RegionSize;
		}
		view.size = end - view.base;
		return view;
	}

	void UnmapOne(const View& view, uint64_t vaddr, uint64_t size) {
		const uint64_t offset     = uint64_t {LoadPage(view.base >> PageBits) - 1u} << PageBits;
		const uint64_t left_size  = vaddr - view.base;
		const uint64_t right_addr = vaddr + size;
		const uint64_t right_size = view.base + view.size - right_addr;
		// Survivors are re-mapped read-write, executable if the view was, and the protections guest
		// mprotect and write tracking set on them are re-applied afterwards.
		const auto mode =
		    (view.protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE)) != 0
		        ? VirtualMemory::Mode::ExecuteReadWrite
		        : VirtualMemory::Mode::ReadWrite;
		std::vector<MEMORY_BASIC_INFORMATION> protections;
		QueryProtections(view.base, left_size, mode, &protections);
		QueryProtections(right_addr, right_size, mode, &protections);

		EXIT_IF(!UnmapViewPreservePlaceholder(reinterpret_cast<void*>(view.base), view.size));
		ClearPages(vaddr, size);
		NoteUnmapUnlocked(vaddr, size); // the survivors' MapFixed below records them again
		EXIT_IF(left_size != 0 && !SplitPlaceholder(view.base, left_size));
		EXIT_IF(right_size != 0 && !SplitPlaceholder(vaddr, size));
		EXIT_IF(left_size != 0 && !MapFixed(view.base, left_size, offset, mode));
		EXIT_IF(right_size != 0 &&
		        !MapFixed(right_addr, right_size, offset + (right_addr - view.base), mode));
		for (const auto& info: protections) {
			EXIT_IF(!ApplyProtection(info.BaseAddress, info.RegionSize, info.Protect));
		}
	}

	bool ReplacePlaceholderView(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                            VirtualMemory::Mode mode, FailureReason* failure_reason) {
		// Lean (4ece93a3f): held until the view is in m_views.
		std::unique_lock<std::mutex> place;
		if (m_first_touch) {
			place = CommitSharedGranules(backing_offset, size);
		}
		const DWORD map_protect =
		    mode == VirtualMemory::Mode::NoAccess ? PAGE_READWRITE : GetProtect(mode);
		void* ptr =
		    MapViewOfFile3(m_handle, GetCurrentProcess(), reinterpret_cast<void*>(vaddr),
		                   backing_offset, size, MEM_REPLACE_PLACEHOLDER, map_protect, nullptr, 0);
		if (ptr == nullptr || reinterpret_cast<uint64_t>(ptr) != vaddr) {
			if (ptr != nullptr) {
				UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);
			}
			SetFailure(failure_reason, FailureReason::PlaceholderMapFailed);
			return false;
		}

		// Granules not committed stay reserved (no access until made ready, which applies NoAccess from the guest's mode).
		if (mode == VirtualMemory::Mode::NoAccess &&
		    !(m_first_touch ? ProtectCommittedRuns(vaddr, size, backing_offset, mode)
		                    : ApplyProtection(ptr, size, PAGE_NOACCESS))) {
			UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);
			SetFailure(failure_reason, FailureReason::ProtectFailed);
			return false;
		}
		if (m_first_touch) {
			std::lock_guard lock(m_view_mutex);
			m_views[vaddr] = {size, backing_offset, mode};
		}
		return true;
	}

	bool UnmapViewPreservePlaceholder(void* vaddr, uint64_t /*size*/) {
		if (m_first_touch) {
			std::lock_guard lock(m_view_mutex);
			m_views.erase(reinterpret_cast<uint64_t>(vaddr));
		}
		return UnmapViewOfFile2(GetCurrentProcess(), vaddr, MEM_PRESERVE_PLACEHOLDER) != 0;
	}
	// No granule is first committed while two guest views show it: the commit through one view would show the page in
	// the other at that view's own protection. So a granule that already has a view is committed (through those views,
	// NoAccess) before another maps over it, and that one maps over committed memory as before first touch. The lock
	// returned is held across the map and its insert into m_views, so two maps can't both see no view.
	std::unique_lock<std::mutex> CommitSharedGranules(uint64_t backing_offset, uint64_t size) {
		std::unique_lock<std::mutex> place(m_place_mutex);
		std::vector<uint64_t>        shared;
		{
			std::lock_guard lock(m_view_mutex);
			for (const auto& [vaddr, view]: m_views) {
				const auto from = std::max(backing_offset, view.backing_offset);
				const auto to   = std::min(backing_offset + size, view.backing_offset + view.size);
				for (auto granule = from / CommitGranule; from < to && granule * CommitGranule < to; ++granule) {
					if (!IsCommitted(granule)) shared.push_back(granule);
				}
			}
		}
		for (const auto granule: shared) {
			CommitGranuleOnce(granule);
		}
		return place;
	}

	[[nodiscard]] bool IsCommitted(uint64_t granule) const {
		return (m_committed[granule / 64u].load(std::memory_order_acquire) >> (granule % 64u) & 1u) != 0;
	}

	[[nodiscard]] bool IsReady(uint64_t granule) const {
		return (m_ready[granule / 64u].load(std::memory_order_acquire) >> (granule % 64u) & 1u) != 0;
	}

	struct ViewPiece {
		uint64_t            vaddr;
		uint64_t            size;
		VirtualMemory::Mode mode;
	};

	// The guest views over a granule, each clipped to it.
	std::vector<ViewPiece> ViewsOver(uint64_t granule) {
		const auto             begin = granule * CommitGranule;
		const auto             end   = std::min(begin + CommitGranule, m_size);
		std::vector<ViewPiece> pieces;
		std::lock_guard        lock(m_view_mutex);
		for (const auto& [vaddr, view]: m_views) {
			const auto view_end = view.backing_offset + view.size;
			if (view.backing_offset < end && begin < view_end) {
				const auto from = std::max(begin, view.backing_offset);
				pieces.push_back({vaddr + (from - view.backing_offset), std::min(end, view_end) - from, view.mode});
			}
		}
		return pieces;
	}

	// Gives a committed granule's views the guest's protection and the tracker's state (m_make_ready), once. False if
	// another thread is doing it. The caller holds no store lock and is not on the alias path.
	bool MakeReady(uint64_t granule) {
		const auto bit = uint64_t {1} << (granule % 64u);
		if (IsReady(granule)) return true;
		if ((m_readying[granule / 64u].fetch_or(bit, std::memory_order_acq_rel) & bit) != 0) return false;
		// After the committed bit: the tracker reads it under its region lock, which m_make_ready takes to read its state.
		for (const auto& piece: ViewsOver(granule)) {
			if (m_make_ready != nullptr) {
				m_make_ready(piece.vaddr, piece.size, piece.mode);
			} else {
				VirtualMemory::Protect(piece.vaddr, piece.size, piece.mode);
			}
		}
		m_ready[granule / 64u].fetch_or(bit, std::memory_order_release);
		return true;
	}

	// True if this call committed it (false: another thread did first). Through each guest view with PAGE_NOACCESS (the
	// view's own protection; the alias comes up read-write), so no view is usable before it is made ready.
	bool CommitGranuleOnce(uint64_t granule) {
		if (IsCommitted(granule)) {
			return false;
		}
		const auto      pieces = ViewsOver(granule);
		std::lock_guard lock(m_commit_mutex);
		if (IsCommitted(granule)) {
			return false;
		}
		const auto offset = granule * CommitGranule;
		const auto bytes  = std::min(CommitGranule, m_size - offset);
		for (const auto& piece: pieces) {
			// A view unmapped since the list fails here; the alias below commits the granule anyway.
			VirtualAlloc(reinterpret_cast<void*>(piece.vaddr), piece.size, MEM_COMMIT, PAGE_NOACCESS);
		}
		if (VirtualAlloc(m_backing_base + offset, bytes, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
			EXIT("Out of memory: Windows could not commit %" PRIu64 " MB of guest memory (error 0x%08" PRIx32
			     ", %" PRIu64 " MB committed so far).\nClose other programs or enlarge the page file (System > About > "
			     "Advanced system settings > Performance > Virtual memory).\n",
			     bytes >> 20u, static_cast<uint32_t>(GetLastError()), CommittedBytes() >> 20u);
		}
		m_committed_bytes.fetch_add(bytes, std::memory_order_relaxed);
		m_committed[granule / 64u].fetch_or(uint64_t {1} << (granule % 64u), std::memory_order_release);
		if (pieces.empty()) {
			// No guest view: ready at once (nothing to protect).
			m_ready[granule / 64u].fetch_or(uint64_t {1} << (granule % 64u), std::memory_order_release);
		}
		return true;
	}

	[[nodiscard]] bool AliasOffset(uint64_t vaddr, uint64_t* offset) const {
		const auto base = reinterpret_cast<uint64_t>(m_backing_base);
		if (vaddr < base || vaddr - base >= m_size) return false;
		*offset = vaddr - base;
		return true;
	}

	[[nodiscard]] bool ViewOffset(uint64_t vaddr, uint64_t* offset) {
		std::lock_guard lock(m_view_mutex);
		auto            it = m_views.upper_bound(vaddr);
		if (it == m_views.begin()) return false;
		--it;
		if (vaddr - it->first >= it->second.size) return false;
		*offset = it->second.backing_offset + (vaddr - it->first);
		return true;
	}

	// The fault path's own table of guest views: filled before MapFixed fills the page table and emptied after it is
	// cleared, so it covers each OS view's whole life.
	struct GuestView {
		uint64_t            size;
		uint64_t            backing_offset;
		VirtualMemory::Mode mode;
	};
	bool                                     m_first_touch = false;
	MakeReadyFn                              m_make_ready  = nullptr;
	std::unique_ptr<std::atomic<uint64_t>[]> m_committed; // one bit per CommitGranule of the backing
	std::unique_ptr<std::atomic<uint64_t>[]> m_ready;     // committed and its views protected as they should be
	std::unique_ptr<std::atomic<uint64_t>[]> m_readying;  // a thread is making it ready
	std::mutex                               m_commit_mutex;
	std::atomic<uint64_t>                    m_committed_bytes {0};
	std::atomic<uint64_t>                    m_first_touch_faults {0};
	std::mutex                               m_view_mutex;  // the fault path's own view table
	std::map<uint64_t, GuestView>            m_views;       // guest views by base address
	std::mutex                               m_place_mutex; // a view's map with its CommitSharedGranules

	HANDLE m_handle = nullptr;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
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

	bool MapView(uint64_t vaddr, uint64_t size, uint64_t backing_offset, VirtualMemory::Mode mode,
	             FailureReason* failure_reason) {
		if (mmap(reinterpret_cast<void*>(vaddr), size, GetProtect(mode), MAP_SHARED | MAP_FIXED,
		         m_fd, static_cast<off_t>(backing_offset)) == MAP_FAILED) {
			SetFailure(failure_reason, FailureReason::FixedMapViewFailed);
			return false;
		}
		return true;
	}

	// Replace the file view atomically instead of exposing an unmapped hole in the
	// process-lifetime guest reservation.
	static bool UnmapViewPreservePlaceholder(void* vaddr, uint64_t size) {
		return mmap(vaddr, size, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) !=
		       MAP_FAILED;
	}

	int m_fd = -1;
#endif

	// Lean's known-zero bookkeeping: the mappings by guest address (the page table maps pages, not mappings, and
	// FindAliases needs the reverse direction), which backing pages any mapping had, and per-page mapping counts.
	struct MapEntry {
		uint64_t            vaddr          = 0;
		uint64_t            size           = 0;
		uint64_t            backing_offset = 0;
		VirtualMemory::Mode mode           = VirtualMemory::Mode::NoAccess;
	};

	void NoteMapUnlocked(uint64_t vaddr, uint64_t size, uint64_t backing_offset, VirtualMemory::Mode mode) {
		RemoveMappingsUnlocked(vaddr, size);
		DropFreshUnlocked(vaddr, size);
		EXIT_IF(!EmplaceMapUnlocked(MapEntry {vaddr, size, backing_offset, mode}));
		NoteMappedUnlocked(vaddr, size, backing_offset, mode);
	}

	void NoteUnmapUnlocked(uint64_t vaddr, uint64_t size) {
		RemoveMappingsUnlocked(vaddr, size);
		DropFreshUnlocked(vaddr, size);
	}

	void RemoveMappingsUnlocked(uint64_t vaddr, uint64_t size) {
		const auto end = vaddr + size;
		auto       it  = m_maps.upper_bound(vaddr);
		if (it != m_maps.begin()) {
			--it;
		}
		while (it != m_maps.end() && it->first < end) {
			const auto old     = it->second;
			const auto old_end = old.vaddr + old.size;
			if (old_end <= vaddr) {
				++it;
				continue;
			}
			it = EraseMapUnlocked(it);
			if (old.vaddr < vaddr) {
				EmplaceMapUnlocked(MapEntry {old.vaddr, vaddr - old.vaddr, old.backing_offset, old.mode});
			}
			if (end < old_end) {
				EmplaceMapUnlocked(MapEntry {end, old_end - end, old.backing_offset + end - old.vaddr, old.mode});
			}
		}
	}

	// The backing is a fresh pagefile-backed section (or memfd): a page no mapping has had yet reads as
	// zeros. Each mapping records its pages here; the runs whose pages were all new wait in m_fresh for
	// the caller (TakeFreshRuns) until the mapping goes away.
	// Executable mappings are recorded but never fresh: the write protection that ends the known-zero
	// state would take away execute.
	void NoteMappedUnlocked(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                        VirtualMemory::Mode mode) {
		constexpr uint64_t page = 0x1000;
		const bool         keep = !VirtualMemory::IsExecute(mode);
		if (m_mapped_once.empty()) {
			m_mapped_once.resize((m_size / page + 63) / 64);
		}
		uint64_t run_start = 0;
		uint64_t run_size  = 0;
		for (uint64_t offset = 0; offset < size; offset += page) {
			const auto index = (backing_offset + offset) / page;
			auto&      word  = m_mapped_once[index / 64];
			const auto bit   = uint64_t {1} << (index % 64);
			if ((word & bit) == 0) {
				word |= bit;
				if (run_size == 0) {
					run_start = offset;
				}
				run_size += page;
				continue;
			}
			if (run_size != 0) {
				if (keep) m_fresh.push_back({vaddr + run_start, run_size, backing_offset + run_start});
				run_size = 0;
			}
		}
		if (run_size != 0) {
			if (keep) m_fresh.push_back({vaddr + run_start, run_size, backing_offset + run_start});
		}
	}

	// Each mapping counts once on every 4 KiB backing page it touches. A count that would pass 255 (or go
	// below 0) turns the counts off for good: FindAliases then always walks the mappings.
	static constexpr uint64_t MapCountPage = 0x1000;

	void CountMappingUnlocked(const MapEntry& entry, bool add) {
		if (!m_map_counts_exact || entry.size == 0) {
			return;
		}
		if (m_map_counts.empty()) {
			m_map_counts.resize((m_size + MapCountPage - 1) / MapCountPage);
		}
		const auto first = entry.backing_offset / MapCountPage;
		const auto last  = (entry.backing_offset + entry.size - 1) / MapCountPage;
		if (last >= m_map_counts.size()) {
			m_map_counts_exact = false;
			return;
		}
		for (auto page = first; page <= last; page++) {
			auto& count = m_map_counts[page];
			if (add ? count == UINT8_MAX : count == 0) {
				m_map_counts_exact = false;
				return;
			}
			count = static_cast<uint8_t>(add ? count + 1 : count - 1);
		}
	}

	bool EmplaceMapUnlocked(const MapEntry& entry) {
		if (!m_maps.emplace(entry.vaddr, entry).second) {
			return false;
		}
		CountMappingUnlocked(entry, true);
		return true;
	}

	std::map<uint64_t, MapEntry>::iterator EraseMapUnlocked(std::map<uint64_t, MapEntry>::iterator entry) {
		CountMappingUnlocked(entry->second, false);
		return m_maps.erase(entry);
	}

	void DropFreshUnlocked(uint64_t vaddr, uint64_t size) {
		std::erase_if(m_fresh, [&](const MapEntry& entry) {
			return entry.vaddr < vaddr + size && vaddr < entry.vaddr + entry.size;
		});
	}

	std::map<uint64_t, MapEntry> m_maps;
	std::vector<uint64_t>        m_mapped_once; // one bit per 4 KiB page of the backing
	std::vector<uint8_t>         m_map_counts;  // mappings per 4 KiB page of the backing (CountMappingUnlocked)
	bool                         m_map_counts_exact = true;
	std::vector<MapEntry>        m_fresh;       // vaddr, size, backing_offset (mode unused)

	uint8_t*                                                     m_backing_base = nullptr;
	uint64_t                                                     m_size         = 0;
	std::array<std::atomic<std::atomic<uint32_t>*>, BucketCount> m_pages {};
};

// Callers change it with the state lock held exclusively and query it under the operation lock or
// the state lock. TryWriteBacking, CopySparseMemory and ProtectTransient take the state lock shared
// themselves.
class GuestAddressSpace {
public:
	explicit GuestAddressSpace(uint64_t backing_size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		m_granularity = info.dwAllocationGranularity;
#endif
		ReserveGuestRegions();
		m_backing = new GuestBackingStore(backing_size, g_first_touch_commit);
	}

	~GuestAddressSpace() {
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
				if (info.Type == MEM_MAPPED) {
					EXIT_IF(UnmapViewOfFile2(GetCurrentProcess(), info.AllocationBase,
					                         MEM_PRESERVE_PLACEHOLDER) == 0);
					continue;
				}
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

	void ZeroBacking(uint64_t backing_offset, uint64_t size) {
		m_backing->ZeroRange(backing_offset, size);
	}

	// The owned ranges are fixed at construction.
	[[nodiscard]] bool Owns(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		return std::any_of(m_owned.begin(), m_owned.end(), [vaddr, size](const auto& range) {
			return vaddr >= range.first && vaddr + size <= range.first + range.second;
		});
	}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	[[nodiscard]] bool OverlapsOwned(uint64_t vaddr, uint64_t size) const {
		return std::any_of(m_owned.begin(), m_owned.end(), [vaddr, size](const auto& range) {
			return VirtualRangesOverlap(vaddr, size, range.first, range.second);
		});
	}
#endif

	// Writes the backed bytes of a range that holds no private mappings.
	bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
		if (!Owns(vaddr, size)) {
			return false;
		}
		std::shared_lock state(g_memory_state_mutex);
		if (OverlapsPrivateUnlocked(vaddr, size)) {
			return false;
		}
		m_backing->WriteBacking(vaddr, data, size);
		return true;
	}

	bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
		return m_backing->TryReadBacking(vaddr, data, size);
	}

	[[nodiscard]] const uint8_t* FindBacking(uint64_t vaddr, uint64_t size) {
		return m_backing->FindBacking(vaddr, size);
	}

	// Lean's known zero: callers hold the state lock exclusively (MarkFreshGpuRange).
	std::vector<std::pair<uint64_t, uint64_t>> TakeFreshRuns(uint64_t vaddr, uint64_t size, uint64_t backing_limit) {
		return m_backing->TakeFreshRuns(vaddr, size, backing_limit);
	}

	std::vector<std::pair<uint64_t, uint64_t>> FindBackingAliases(uint64_t vaddr, uint64_t size) const {
		return m_backing->FindAliases(vaddr, size);
	}

	// Lean (first touch): see GuestBackingStore. CommitGuestRange never under the state lock.
	[[nodiscard]] bool     HandleFirstTouch(uint64_t vaddr, bool write, bool execute) {
		return m_backing->HandleFirstTouch(vaddr, write, execute);
	}
	[[nodiscard]] bool     CommitOnFault(uint64_t vaddr) { return m_backing->CommitOnFault(vaddr); }
	void                   SetMakeReady(GuestBackingStore::MakeReadyFn fn) { m_backing->SetMakeReady(fn); }
	void                   CommitGuestRange(uint64_t vaddr, uint64_t size) { m_backing->CommitViews(vaddr, size); }
	[[nodiscard]] bool     IsFirstTouch() const { return m_backing->IsFirstTouch(); }
	[[nodiscard]] uint64_t CommittedBytes() const { return m_backing->CommittedBytes(); }
	[[nodiscard]] uint64_t FirstTouchFaults() const { return m_backing->FirstTouchFaults(); }

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	void TestForgetMappedPages() {
		std::lock_guard state(g_memory_state_mutex);
		m_backing->TestForgetMappedPages();
	}
#endif

	[[nodiscard]] bool IsBacked(uint64_t vaddr, uint64_t size) const {
		return m_backing->IsBacked(vaddr, size);
	}

	template <bool copy_private>
	bool CopySparseMemory(uint64_t vaddr, void* data, uint64_t size) {
		if (m_backing->TryReadBacking(vaddr, data, size)) {
			return true;
		}
		if (data == nullptr || !Owns(vaddr, size)) {
			return false;
		}
		std::shared_lock state(g_memory_state_mutex);
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_backing_read != nullptr) {
			g_test_backing_read(vaddr, size);
		}
#endif
		if constexpr (!copy_private) {
			if (OverlapsPrivateUnlocked(vaddr, size)) {
				return false;
			}
		}
		m_backing->ReadSparseBacking(
		    vaddr, data, size, [&](uint64_t start, uint8_t* destination, uint64_t bytes) {
			    if constexpr (copy_private) {
				    // Backing gaps can contain private mappings or reservations only.
				    CopyPrivateUnlocked(start, destination, bytes);
			    } else {
				    std::memset(destination, 0, bytes);
			    }
		    });
		return true;
	}

	bool Protect(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		return ProtectMappedUnlocked(vaddr, size, mode);
	}

	// Tracker protections only read m_mapped; the page tracker orders them per page.
	bool ProtectTransient(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		std::shared_lock state(g_memory_state_mutex);
		return ProtectMappedUnlocked(vaddr, size, mode);
	}

	bool MapBacking(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                VirtualMemory::Mode               mode,
	                GuestBackingStore::FailureReason* failure_reason = nullptr) {
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		const bool replacing = ContainsMappedUnlocked(vaddr, size, MappingType::Backing);
#else
		constexpr bool replacing = false;
#endif
		if (!replacing && !ConsumeUnlocked(vaddr, size)) {
			if (failure_reason != nullptr) {
				*failure_reason = GuestBackingStore::FailureReason::AddressRangeUnavailable;
			}
			return false;
		}
		const auto host_mode = VirtualMemory::IsExecute(mode)
		                           ? VirtualMemory::Mode::ExecuteReadWrite
		                           : VirtualMemory::Mode::ReadWrite;
		if (m_backing->MapFixed(vaddr, size, backing_offset, host_mode, failure_reason)) {
			if (!replacing) {
				AddMappedUnlocked(vaddr, size, MappingType::Backing);
			}
			return true;
		}
		if (!replacing) {
			AddFreeUnlocked(vaddr, size);
		}
		return false;
	}

	bool UnmapBacking(uint64_t vaddr, uint64_t size) {
		if (!ContainsMappedUnlocked(vaddr, size, MappingType::Backing)) {
			return false;
		}
		m_backing->Unmap(vaddr, size);
		RemoveMappedUnlocked(vaddr, size);
		AddFreeUnlocked(vaddr, size);
		return true;
	}

	uint64_t FindFreeAligned(uint64_t search_start, uint64_t search_end, uint64_t size,
	                         uint64_t alignment) {
		alignment = std::max<uint64_t>(alignment, PageSize());
		return FindFreeAlignedUnlocked(search_start, search_end, size, alignment);
	}

	bool ContainsFree(uint64_t vaddr, uint64_t size) {
		return FindContaining(vaddr, size) != m_free.end();
	}

	bool Commit(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		if (!ConsumeUnlocked(vaddr, size)) {
			return false;
		}
		const auto host_mode = VirtualMemory::IsExecute(mode)
		                           ? VirtualMemory::Mode::ExecuteReadWrite
		                           : VirtualMemory::Mode::ReadWrite;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		auto* ptr = VirtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(vaddr), size,
		                          MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
		                          GetProtect(host_mode), nullptr, 0);
		if (ptr == nullptr || reinterpret_cast<uint64_t>(ptr) != vaddr) {
			if (ptr != nullptr) {
				VirtualFree(ptr, 0, MEM_RELEASE);
			}
			AddFreeUnlocked(vaddr, size);
			return false;
		}
		AddMappedUnlocked(vaddr, size, MappingType::Private);
		return true;
#else
		if (VirtualMemory::Protect(vaddr, size, host_mode)) {
			AddMappedUnlocked(vaddr, size, MappingType::Private);
			return true;
		}
		AddFreeUnlocked(vaddr, size);
		return false;
#endif
	}

	bool ReleaseCommitted(uint64_t vaddr, uint64_t size) {
		if (!IsPageAligned(vaddr) || !IsPageAligned(size)) {
			return false;
		}
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
		RemoveMappedUnlocked(vaddr, size);
		AddFreeUnlocked(vaddr, size);
		return true;
	}

	bool SelfTest() {
		auto addr = FindFreeAligned(HOST_SYSTEM_MANAGED_MIN, HOST_SYSTEM_MANAGED_MAX + 1u,
		                            PageSize(), PageSize());
		if (addr == 0) {
			addr = FindFreeAligned(HOST_USER_MIN, HOST_USER_MAX + 1u, PageSize(), PageSize());
		}
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
		const auto remainder = value % alignment;
		return remainder != 0 ? value + (alignment - remainder) : value;
	}

	static uint64_t AlignDown(uint64_t value, uint64_t alignment) {
		return value & ~(alignment - 1u);
	}

	[[nodiscard]] static bool IsPageAligned(uint64_t value) {
		return (value & (PageSize() - 1u)) == 0;
	}

	bool ProtectMappedUnlocked(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		const auto end = vaddr + size;
		auto       it  = m_mapped.upper_bound(vaddr);
		if (it != m_mapped.begin()) {
			--it;
		}
		for (; it != m_mapped.end() && it->second.vaddr < end; ++it) {
			const auto& region      = it->second;
			const auto  range_start = std::max(vaddr, region.vaddr);
			const auto  range_end   = std::min(end, region.vaddr + region.size);
			if (range_start >= range_end) {
				continue;
			}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			// Lean (first touch): a backing region is one view; VirtualProtect fails on its reserved granules.
			if (region.type == MappingType::Backing && m_backing->IsFirstTouch()) {
				if (!m_backing->ProtectViews(range_start, range_end - range_start, mode)) {
					return false;
				}
				continue;
			}
#endif
			if (!VirtualMemory::Protect(range_start, range_end - range_start, mode)) {
				return false;
			}
		}
		return true;
	}

	bool ConsumeUnlocked(uint64_t vaddr, uint64_t size) {
		const auto it = FindContaining(vaddr, size);
		if (it == m_free.end() || !SplitUnlocked(it, vaddr, size)) {
			return false;
		}
		m_free.erase(vaddr);
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

	[[nodiscard]] bool OverlapsPrivateUnlocked(uint64_t vaddr, uint64_t size) const {
		auto mapping = m_mapped.upper_bound(vaddr);
		if (mapping != m_mapped.begin()) {
			--mapping;
		}
		for (; mapping != m_mapped.end() && mapping->second.vaddr < vaddr + size; ++mapping) {
			const auto& region = mapping->second;
			if (region.type == MappingType::Private && region.vaddr + region.size > vaddr) {
				return true;
			}
		}
		return false;
	}

	// Copies the private mappings of a span and zero-fills the rest.
	void CopyPrivateUnlocked(uint64_t vaddr, uint8_t* destination, uint64_t size) const {
		const auto end    = vaddr + size;
		auto       cursor = vaddr;
		auto       it     = m_mapped.upper_bound(vaddr);
		if (it != m_mapped.begin()) {
			--it;
		}
		for (; it != m_mapped.end() && it->second.vaddr < end; ++it) {
			const auto start = std::max(cursor, it->second.vaddr);
			const auto stop  = std::min(end, it->second.vaddr + it->second.size);
			if (start < stop) {
				std::memset(destination + (cursor - vaddr), 0, start - cursor);
				std::memcpy(destination + (start - vaddr), reinterpret_cast<const void*>(start),
				            stop - start);
				cursor = stop;
			}
		}
		std::memset(destination + (cursor - vaddr), 0, end - cursor);
	}

	void AddMappedUnlocked(uint64_t vaddr, uint64_t size, MappingType type) {
		m_mapped.emplace(vaddr, MappedRegion {vaddr, size, type});
	}

	void RemoveMappedUnlocked(uint64_t vaddr, uint64_t size) {
		const auto end     = vaddr + size;
		auto       current = vaddr;
		while (current < end) {
			const auto it           = FindMappedContainingUnlocked(current);
			const auto original     = it->second;
			const auto original_end = original.vaddr + original.size;
			const auto part_end     = std::min(end, original_end);
			m_mapped.erase(it);
			if (original.vaddr < current) {
				AddMappedUnlocked(original.vaddr, current - original.vaddr, original.type);
			}
			if (part_end < original_end) {
				AddMappedUnlocked(part_end, original_end - part_end, original.type);
			}
			current = part_end;
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
#endif

	void ReserveGuestRegions() {
		// DLSS: NVIDIA's CUDA side (VK_NVX_binary_import, cuInit) reserves its arena below 1 TB, 9 GB at the least beside an
		// 8 GB allocation of its own, and refuses the device otherwise. The guest regions leave 16 GB there, which this
		// process's bottom-up allocations fill (no high-entropy ASLR: /DYNAMICBASE:NO). With DLSS asked for, the user region
		// stops 48 GB short, as guest maps fill it from the bottom, and the room up to 1 TB is held until the Vulkan device is made:
		// the host's own allocations before it (some 270 GB) would fill it too (session 58, 2026-11-11: 64 GB free passes,
		// 16 fails). Windows only: Linux's CUDA wants other addresses (0x200000000 on), not looked at.
		const bool dlss = KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS && Config::Ps5ProEnabled() && !Config::GetDlssFolder().empty();
		const uint64_t user_end = dlss ? HOST_USER_MAX + 1u - (48ull << 30u) : HOST_USER_MAX + 1u;
		const std::array<std::pair<uint64_t, uint64_t>, 3> regions {{
		    {HOST_SYSTEM_MANAGED_MIN, HOST_SYSTEM_RESERVED_MAX + 1u},
		    {HOST_USER_MIN, user_end},
		    {kExtendedMemoryBase, kExtendedMemoryBase + kExtendedMemorySize},
		}};

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		constexpr uint64_t MinimumRegionSize = 0x1000000;
		const auto         process           = GetCurrentProcess();
		for (const auto& [guest_start, guest_end]: regions) {
			auto next_addr = guest_start;
			while (next_addr < guest_end) {
				MEMORY_BASIC_INFORMATION info {};
				if (VirtualQuery(reinterpret_cast<const void*>(next_addr), &info, sizeof(info)) == 0) {
					EXIT("failed to inspect guest address space at 0x%016" PRIx64 "\n", next_addr);
				}

				const auto region_base = reinterpret_cast<uint64_t>(info.BaseAddress);
				const auto region_end = std::min<uint64_t>(
				    guest_end, region_base + static_cast<uint64_t>(info.RegionSize));
				const auto reserve_start =
				    AlignUp(std::max(region_base, guest_start), m_granularity);
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

				next_addr = region_end;
			}
		}
		if (dlss) {
			g_driver_room = VirtualAlloc(reinterpret_cast<void*>(user_end), (1ull << 40u) - user_end, MEM_RESERVE, PAGE_NOACCESS);
			if (g_driver_room == nullptr) {
				printf("DLSS: the address room below 1 TB is taken (%lu); the driver may refuse DLSS\n", GetLastError());
			}
		}
#else
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
			if (end <= start) {
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
			// Free ranges merge across region boundaries, so owned ranges do as well.
			if (!m_owned.empty() && m_owned.back().first + m_owned.back().second == start) {
				m_owned.back().second += size;
			} else {
				m_owned.emplace_back(start, size);
			}
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
	                                 uint64_t alignment) const {
		// Include the predecessor because search_start can lie inside a free range.
		auto it = m_free.upper_bound(search_start);
		if (it != m_free.begin()) {
			--it;
		}
		for (; it != m_free.end() && it->first < search_end; ++it) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
			g_test_free_range_visits.fetch_add(1, std::memory_order_relaxed);
#endif
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
		auto it = std::prev(next);
		return GuestFreeRangeContains(it->first, it->second, vaddr, size) ? it : m_free.end();
	}

	bool SplitUnlocked(std::map<uint64_t, uint64_t>::iterator it, uint64_t vaddr, uint64_t size) {
		const auto base = it->first;
		const auto end  = it->first + it->second;
		if (base != vaddr) {
			const auto left_size = vaddr - base;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (VirtualFree(reinterpret_cast<void*>(base), left_size,
			                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0) {
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
			if (VirtualFree(reinterpret_cast<void*>(it->first), size,
			                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0) {
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
		EXIT_IF(ptr == MAP_FAILED);
#else
		(void)vaddr;
		(void)size;
#endif
	}

	std::map<uint64_t, uint64_t>               m_free;
	std::map<uint64_t, MappedRegion>            m_mapped;
	std::vector<std::pair<uint64_t, uint64_t>> m_owned;
	uint64_t                                   m_granularity = 0x10000;
	GuestBackingStore*                         m_backing     = nullptr;
};
