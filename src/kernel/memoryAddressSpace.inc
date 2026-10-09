// Internal memory address-space helpers included by memory.cpp.
// Kept in the same translation unit because they depend on memory.cpp-local helpers.

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
static bool CoalescePlaceholders(uint64_t vaddr, uint64_t size) {
	return size != 0 && VirtualFree(reinterpret_cast<void*>(vaddr), size,
	                                MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) != 0;
}
#endif

// Guest ranges that still hold their data but were mapped again with their view's protection:
// the neighbours of a partly unmapped view (a Windows view cannot be split, so the whole view is
// unmapped and its remaining parts mapped again) and views an unmap rolled back. The GPU
// trackers' write and read protection of their pages was lost with the old view; without it CPU
// writes to watched pages went unseen and the GPU kept using stale data (a frozen character
// model). It goes back once the address-space lock is released (GuestAddressSpace::UnmapBacking;
// the page manager protects through that lock).
thread_local std::vector<std::pair<uint64_t, uint64_t>> g_remapped_views;

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
	};

	explicit GuestBackingStore(uint64_t size): m_size(size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		m_handle = CreateFileMappingA(
		    INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE | SEC_COMMIT,
		    static_cast<DWORD>(m_size >> 32u), static_cast<DWORD>(m_size & 0xffffffffu), nullptr);
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
		for (const auto& [_, entry]: m_maps) {
			UnmapViewPreservePlaceholder(reinterpret_cast<void*>(entry.vaddr), entry.size);
		}
		m_maps.clear();
		m_coverage.clear();
		if (m_backing_base != nullptr) {
			UnmapViewOfFile(m_backing_base);
		}
		if (m_handle != nullptr) {
			CloseHandle(m_handle);
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
			default: return "unknown";
		}
	}

	[[nodiscard]] bool IsAvailable() const {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return m_handle != nullptr && m_backing_base != nullptr;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		return m_fd >= 0 && m_backing_base != nullptr;
#else
		return false;
#endif
	}

	[[nodiscard]] uint64_t GetBackingBase() const {
		return reinterpret_cast<uint64_t>(m_backing_base);
	}

	[[nodiscard]] uint64_t GetBackingSize() const { return m_size; }

	// Locking: m_mutex (shared) keeps the mappings stable; mapping changes take it exclusively.
	// m_data_mutex serializes the copies into and out of the backing (the write fence below
	// assumes one writer at a time), so mapping queries such as IsUniqueRange no longer wait
	// behind a GPU readback being written back into guest memory.
	// Diagnostic (live measure: backing_lock_*, backing_hold_*): the render thread's waits for
	// either lock are counted when it is contended, and how long the writers hold them.
	template <typename Lock>
	static void AcquireCounted(Lock& lock) {
		if (lock.try_lock()) return;
		const auto start = std::chrono::steady_clock::now();
		lock.lock();
		if (Graphics::GuestGpu::IsGpuThread()) {
			LiveCounters::Add(LiveCounters::BackingLockWaits);
			LiveCounters::Add(LiveCounters::BackingLockWaitUs,
			                  static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                            std::chrono::steady_clock::now() - start)
			                                            .count()));
		}
	}
	std::shared_lock<std::shared_mutex> MapReadLock() {
		std::shared_lock lock(m_mutex, std::defer_lock);
		AcquireCounted(lock);
		return lock;
	}
	std::unique_lock<std::mutex> DataLock() {
		std::unique_lock lock(m_data_mutex, std::defer_lock);
		AcquireCounted(lock);
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
			if (IsUsedGranule(granule)) std::memset(m_backing_base + offset, 0, next - offset);
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
		                          });
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
				auto& cached = m_host_read_translations[(vaddr >> 22) % m_host_read_translations.size()];
				MapEntry entry {};
				if (cached.epoch == m_mapping_epoch &&
				    GuestFreeRangeContains(cached.entry.vaddr, cached.entry.size, vaddr, size)) {
					entry = cached.entry;
				} else if (FindContainingUnlocked(vaddr, size, &entry)) {
					cached = {entry, m_mapping_epoch};
				}
				if (entry.size && RangeOk(entry.backing_offset + vaddr - entry.vaddr, size)) {
					const auto offset = entry.backing_offset + vaddr - entry.vaddr;
					if (read.Narrow(offset, offset + size)) {
						std::memcpy(data, m_backing_base + offset, size);
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
				auto& cached = m_host_read_translations[(vaddr >> 22) % m_host_read_translations.size()];
				if (cached.epoch == m_mapping_epoch &&
				    GuestFreeRangeContains(cached.entry.vaddr, cached.entry.size, vaddr, size)) {
					entry = cached.entry;
				} else if (FindContainingUnlocked(vaddr, size, &entry)) {
					cached = {entry, m_mapping_epoch};
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
		return RangeOk(offset, size) ? m_backing_base + offset : nullptr;
	}

	// The backing bytes of [vaddr, vaddr + size) as one piece per guest mapping it spans, in
	// order (the pieces are contiguous in the guest range); false if a part is not mapped.
	bool BackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces) {
		pieces.clear();
		if (!IsAvailable() || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
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
			pieces.emplace_back(m_backing_base + offset, bytes);
			current += bytes;
		}
		return true;
	}

	bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size) {
		if (!IsAvailable() || data == nullptr || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		auto lock      = MapReadLock();
		auto data_lock = DataLock();
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
			const auto  offset = entry.backing_offset + start - entry.vaddr;
			std::memcpy(destination + start - vaddr, m_backing_base + offset, bytes);
		}
		return true;
	}

	bool MapExistingPlaceholderFixed(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                                 VirtualMemory::Mode mode,
	                                 FailureReason*      failure_reason = nullptr) {
		GuestBackingMutation mutation;
		if (!IsAvailable() || !RangeOk(backing_offset, size)) {
			SetFailure(failure_reason, !IsAvailable() ? FailureReason::BackingUnavailable
			                                          : FailureReason::PhysicalRangeOutOfBounds);
			return false;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		void* mapped = ReplacePlaceholderView(vaddr, size, backing_offset, mode, failure_reason);
		if (mapped == nullptr || reinterpret_cast<uint64_t>(mapped) != vaddr) {
			if (mapped != nullptr) {
				UnmapViewPreservePlaceholder(mapped, size);
				SetFailure(failure_reason, FailureReason::WrongHostAddress);
			}
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
#else
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

	bool Unmap(uint64_t vaddr, uint64_t size, bool* placeholder_preserved = nullptr) {
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
		for (const auto& entry: targets) {
			bool part_placeholder_preserved = false;
			if (!UnmapOne(entry.vaddr, entry.size, &part_placeholder_preserved)) {
				for (auto it = removed.rbegin(); it != removed.rend(); ++it) {
					EXIT_IF(!MapExistingPlaceholderFixed(it->vaddr, it->size, it->backing_offset,
					                                     it->mode));
					g_remapped_views.emplace_back(it->vaddr, it->size);
				}
				if (placeholder_preserved != nullptr) {
					*placeholder_preserved = false;
				}
				return false;
			}
			EXIT_IF(!part_placeholder_preserved);
			removed.push_back(entry);
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		// Each unmapped file view becomes a separate Windows placeholder. The address-space owner
		// records this contiguous guest span as one free range, so normalize fragmented backing
		// views into one placeholder before publishing it there.
		if (removed.size() > 1 && !CoalescePlaceholders(vaddr, size)) {
			for (auto it = removed.rbegin(); it != removed.rend(); ++it) {
				EXIT_IF(!MapExistingPlaceholderFixed(it->vaddr, it->size, it->backing_offset,
				                                     it->mode));
				g_remapped_views.emplace_back(it->vaddr, it->size);
			}
			return false;
		}
#endif

		if (placeholder_preserved != nullptr) {
			*placeholder_preserved = true;
		}
		return true;
	}

	bool UnmapOne(uint64_t vaddr, uint64_t size, bool* placeholder_preserved = nullptr) {
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
		if (placeholder_preserved != nullptr) {
			*placeholder_preserved = true;
		}
		return true;
#else
		MapEntry old {};
		{
			std::lock_guard lock(m_mutex);
			HoldTimer hold {LiveCounters::BackingHoldMapUs};
			auto            next = m_maps.upper_bound(vaddr);
			if (next != m_maps.begin()) {
				auto entry = std::prev(next);
				if (vaddr + size <= entry->second.vaddr + entry->second.size) {
					old = entry->second;
					BackingWriteFence fence(*this);
					InvalidateMappingLookup();
					MapsErase(entry);
				}
			}
		}

		if (old.size == 0) {
			return false;
		}

		const bool exact_unmap = (vaddr == old.vaddr && size == old.size);
		if (exact_unmap) {
			const bool unmapped =
			    UnmapViewPreservePlaceholder(reinterpret_cast<void*>(old.vaddr), old.size);
			if (!unmapped) {
				std::lock_guard lock(m_mutex);
				HoldTimer hold {LiveCounters::BackingHoldMapUs};
				BackingWriteFence fence(*this);
				InvalidateMappingLookup();
				MapsEmplace(old);
				return false;
			}
			if (placeholder_preserved != nullptr) {
				*placeholder_preserved = true;
			}
			return true;
		}

		const uint64_t left_size  = vaddr - old.vaddr;
		const uint64_t right_addr = vaddr + size;
		const uint64_t old_end    = old.vaddr + old.size;
		const uint64_t right_size = old_end - right_addr;

		if (UnmapViewPreservePlaceholder(reinterpret_cast<void*>(old.vaddr), old.size)) {
			bool ok = true;
			if (left_size != 0 && left_size != old.size) {
				ok = SplitPlaceholder(old.vaddr, left_size) && ok;
			}
			if (right_size != 0) {
				ok = SplitPlaceholder(vaddr, size) && ok;
			}
			if (left_size != 0) {
				const bool mapped =
				    MapExistingPlaceholderFixed(old.vaddr, left_size, old.backing_offset, old.mode);
				if (mapped) g_remapped_views.emplace_back(old.vaddr, left_size);
				ok = mapped && ok;
			}
			if (right_size != 0) {
				const bool mapped = MapExistingPlaceholderFixed(
				    right_addr, right_size, old.backing_offset + (right_addr - old.vaddr), old.mode);
				if (mapped) g_remapped_views.emplace_back(right_addr, right_size);
				ok = mapped && ok;
			}
			if (ok) {
				if (placeholder_preserved != nullptr) {
					*placeholder_preserved = true;
				}
				return true;
			}

			if (left_size != 0 && Contains(old.vaddr, left_size)) {
				EXIT_IF(!UnmapOne(old.vaddr, left_size));
			}
			if (right_size != 0 && Contains(right_addr, right_size)) {
				EXIT_IF(!UnmapOne(right_addr, right_size));
			}
			EXIT_IF(VirtualFree(reinterpret_cast<void*>(old.vaddr), old.size,
			                    MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) == 0);
			EXIT_IF(
			    !MapExistingPlaceholderFixed(old.vaddr, old.size, old.backing_offset, old.mode));
			g_remapped_views.emplace_back(old.vaddr, old.size);
			return false;
		}
		std::lock_guard lock(m_mutex);
		HoldTimer hold {LiveCounters::BackingHoldMapUs};
		BackingWriteFence fence(*this);
		InvalidateMappingLookup();
		MapsEmplace(old);
		return false;
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

	bool IsUniqueRange(uint64_t vaddr, uint64_t size) {
		// KYTY_ALIAS_VERIFY: every answer is compared with the full rebuild it replaces.
		static const bool verify = std::getenv("KYTY_ALIAS_VERIFY") != nullptr;
		if (!verify) {
			auto lock = MapReadLock();
			return IsUniqueRangeUnlocked(vaddr, size);
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
	// One writer at a time: data writers hold m_data_mutex (and m_mutex shared), zeroing and
	// mapping mutations hold m_mutex exclusively, which excludes the data writers.
	// A writer announces itself before inspecting the one reader slot. Sequential
	// consistency makes it impossible for both entrants to miss the other: either
	// the reader falls back, or the writer waits for its bounded host copy to end.
	// No GPU work, guest faults, or other locks occur inside that copy transaction.
	class BackingWriteFence {
	public:
		KYTY_CLASS_NO_COPY(BackingWriteFence);
		explicit BackingWriteFence(GuestBackingStore& store): m_store(store) {
			m_store.m_host_write_pending.store(true, std::memory_order_seq_cst);
			unsigned spins = 0;
			while (m_store.m_host_read_pages.load(std::memory_order_seq_cst) != 0) {
				_mm_pause();
				if (++spins == 256) { std::this_thread::yield(); spins = 0; }
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
		DataWriteFence(GuestBackingStore& store, uint64_t begin, uint64_t end): m_store(store) {
			const auto pages = PackPages(begin, end);
			m_store.m_data_write_pages.store(pages, std::memory_order_seq_cst);
			unsigned spins = 0;
			while (PagesOverlap(m_store.m_host_read_pages.load(std::memory_order_seq_cst), pages)) {
				_mm_pause();
				if (++spins == 256) { std::this_thread::yield(); spins = 0; }
			}
		}
		~DataWriteFence() { m_store.m_data_write_pages.store(0, std::memory_order_release); }
	private:
		GuestBackingStore& m_store;
	};

	class BackingReadTransaction {
	public:
		KYTY_CLASS_NO_COPY(BackingReadTransaction);
		explicit BackingReadTransaction(GuestBackingStore& store): m_store(store) {
			// One reader at a time; it holds every page until Narrow names the ones it copies.
			uint64_t idle = 0;
			if (m_store.m_host_write_pending.load(std::memory_order_relaxed) ||
			    !m_store.m_host_read_pages.compare_exchange_strong(idle, AllPages, std::memory_order_seq_cst))
				return;
			m_acquired = true;
			if (m_store.m_host_write_pending.load(std::memory_order_seq_cst)) Release();
		}
		~BackingReadTransaction() { Release(); }
		bool Acquired() const { return m_acquired; }
		// The read copies backing bytes [begin, end): false (and released) while a data write
		// into overlapping pages is under way.
		bool Narrow(uint64_t begin, uint64_t end) {
			const auto pages = PackPages(begin, end);
			m_store.m_host_read_pages.store(pages, std::memory_order_seq_cst);
			if (PagesOverlap(m_store.m_data_write_pages.load(std::memory_order_seq_cst), pages)) {
				Release();
				return false;
			}
			return true;
		}
	private:
		void Release() {
			if (m_acquired) {
				m_store.m_host_read_pages.store(0, std::memory_order_release);
				m_acquired = false;
			}
		}
		GuestBackingStore& m_store;
		bool m_acquired = false;
	};

	template <bool Write, typename Transfer>
	bool TryTransferBacking(uint64_t vaddr, uint64_t size, Transfer transfer) {
		if (!IsAvailable() || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		auto lock      = MapReadLock();
		auto data_lock = DataLock();
		if constexpr (Write) {
			HoldTimer hold {LiveCounters::BackingHoldWriteUs};
			// The shared map lock keeps the mapping, so its backing bytes, fixed. A range
			// that spans mappings keeps the fence that excludes every reader.
			MapEntry entry {};
			if (FindContainingUnlocked(vaddr, size, &entry)) {
				const auto      begin = entry.backing_offset + vaddr - entry.vaddr;
				DataWriteFence fence(*this, begin, begin + size);
				return TransferBackingUnlocked(vaddr, size, transfer);
			}
			BackingWriteFence fence(*this);
			return TransferBackingUnlocked(vaddr, size, transfer);
		} else {
			return TransferBackingUnlocked(vaddr, size, transfer);
		}
	}

	template <typename Transfer>
	bool TransferBackingUnlocked(uint64_t vaddr, uint64_t size, Transfer transfer) {
		const auto      end     = vaddr + size;
		if (kyty_local_preparation_lookup_mode.load(std::memory_order_relaxed)) {
			MapEntry entry{};
			if (FindContainingUnlocked(vaddr, size, &entry)) {
				const auto offset = entry.backing_offset + vaddr - entry.vaddr;
				if (!RangeOk(offset, size)) return false;
				// The lock keeps this whole mapping valid through the transfer.
				// Spanning requests retain the original validate-all-then-copy path.
				transfer(m_backing_base + offset, 0, static_cast<size_t>(size));
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
			const auto offset = entry.backing_offset + current - entry.vaddr;
			transfer(m_backing_base + offset, current - vaddr, static_cast<size_t>(bytes));
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
			it->second = static_cast<uint32_t>(static_cast<int>(it->second) + delta);
		}
		// Merge keys that no longer start a different count.
		for (auto it: {first, last}) {
			if (it == m_coverage.end()) continue;
			const uint32_t before = it == m_coverage.begin() ? 0 : std::prev(it)->second;
			if (it->second == before) m_coverage.erase(it);
		}
	}

	// Every m_maps change goes through these two (they keep m_coverage in step).
	bool MapsEmplace(const MapEntry& entry) {
		if (!m_maps.emplace(entry.vaddr, entry).second) return false;
		AddCoverage(entry.backing_offset, entry.backing_offset + entry.size, 1);
		return true;
	}
	void MapsErase(std::map<uint64_t, MapEntry>::iterator entry) {
		AddCoverage(entry->second.backing_offset, entry->second.backing_offset + entry->second.size, -1);
		m_maps.erase(entry);
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
		if (++m_mapping_epoch == 0) {
			m_mapping_epoch = 1;
			// In place: assigning {} builds a 40 KiB array on the stack (every caller's frame, called or not).
			m_host_read_translations.fill({});
		}
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
		return VirtualProtect(ptr, size, GetProtect(mode), &old_protect) != 0;
	}

	static bool SplitPlaceholder(uint64_t vaddr, uint64_t size) {
		if (size == 0) {
			return true;
		}
		LiveTrace::SlowOsCallScope slow(LiveTrace::OsFree, vaddr, size);
		return VirtualFree(reinterpret_cast<void*>(vaddr), size,
		                   MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) != 0;
	}

	void* ReplacePlaceholderView(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                             VirtualMemory::Mode mode, FailureReason* failure_reason) {
		const DWORD map_protect =
		    mode == VirtualMemory::Mode::NoAccess ? PAGE_READWRITE : GetProtect(mode);
		void* ptr = nullptr;
		{
			LiveTrace::SlowOsCallScope slow(LiveTrace::OsMapView, vaddr, size);
			ptr = MapViewOfFile3(m_handle, GetCurrentProcess(), reinterpret_cast<void*>(vaddr), backing_offset, size,
			                     MEM_REPLACE_PLACEHOLDER, map_protect, nullptr, 0);
		}
		if (ptr == nullptr || reinterpret_cast<uint64_t>(ptr) != vaddr) {
			if (ptr != nullptr) {
				UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);
			}
			SetFailure(failure_reason, FailureReason::PlaceholderMapFailed);
			return nullptr;
		}

		if (!ApplyProtection(ptr, size, mode)) {
			UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);
			SetFailure(failure_reason, FailureReason::ProtectFailed);
			return nullptr;
		}
		return ptr;
	}

	static bool UnmapViewPreservePlaceholder(void* vaddr, uint64_t size) {
		LiveTrace::SlowOsCallScope slow(LiveTrace::OsUnmapView, reinterpret_cast<uint64_t>(vaddr), size);
		return UnmapViewOfFile2(GetCurrentProcess(), vaddr, MEM_PRESERVE_PLACEHOLDER) != 0;
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
		for (uint64_t granule = backing_offset >> UsedGranuleBits; granule <= last; ++granule) {
			m_used[granule >> 6] |= uint64_t {1} << (granule & 63);
		}
	}
	std::vector<uint64_t>        m_used; // 1 bit per 64 KiB granule of the backing a mapping has covered

	uint8_t*                     m_backing_base = nullptr;
	uint64_t                     m_size         = 0;
	std::shared_mutex            m_mutex;
	std::mutex                   m_data_mutex;
	std::vector<std::pair<uint64_t, uint64_t>> m_aliased_physical_ranges;
	bool m_alias_ranges_valid = false;
	std::map<uint64_t, MapEntry> m_maps;
	// Physical coverage of m_maps: [key, next key) is mapped by `value` guest mappings (the
	// last key closes the covered space with 0). Changed with m_maps (MapsEmplace/MapsErase),
	// so IsUniqueRange looks only at the requested range instead of rebuilding the alias
	// intervals of every mapping after each map or unmap (~0.7 ms with 6700 mappings, two to
	// three times a frame while the game streams).
	std::map<uint64_t, uint32_t> m_coverage;
	alignas(64) std::atomic_bool m_host_write_pending {false};
	// The pages of the data write in progress (DataWriteFence), 0 when none.
	alignas(64) std::atomic<uint64_t> m_data_write_pages {0};
	// The pages the one active host read copies (all until it knows), 0 when none.
	alignas(64) std::atomic<uint64_t> m_host_read_pages {0};
	struct HostReadTranslation { MapEntry entry; uint64_t epoch = 0; };
	// By 4 MiB granule: 1024 slots cover 4 GiB without two granules sharing one (with 64, granules 256 MiB apart
	// evicted each other and their reads searched m_maps).
	std::array<HostReadTranslation, 1024> m_host_read_translations {};
	uint64_t m_mapping_epoch = 1;
};

class GuestAddressSpace {
public:
	explicit GuestAddressSpace(uint64_t backing_size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		m_granularity = info.dwAllocationGranularity;
#endif
		ReserveGuestRegions();
		m_backing = new GuestBackingStore(backing_size);
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

	bool ZeroBacking(uint64_t backing_offset, uint64_t size) {
		return m_backing->ZeroRange(backing_offset, size);
	}

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

	bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
		return m_backing->TryReadBacking(vaddr, data, size);
	}
	bool TryReadBackingToHost(uint64_t vaddr, void* data, uint64_t size) {
		return m_backing->TryReadBackingToHost(vaddr, data, size);
	}

	const uint8_t* BackingPointer(uint64_t vaddr, uint64_t size) { return m_backing->BackingPointer(vaddr, size); }
	bool BackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces) {
		return m_backing->BackingPieces(vaddr, size, pieces);
	}

	bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size) {
		return m_backing->TryReadSparseBacking(vaddr, data, size);
	}

	bool IsUniqueBackingRange(uint64_t vaddr, uint64_t size) { return m_backing->IsUniqueRange(vaddr, size); }

	bool BackingContains(uint64_t vaddr, uint64_t size) { return m_backing->Contains(vaddr, size); }

	bool Protect(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		GuestBackingMutation mutation;
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		return ProtectMappedUnlocked(vaddr, size, mode);
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
		std::shared_lock lock(m_mutex);
		return ProtectMappedUnlocked(vaddr, size, mode);
	}

	bool MapBacking(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                VirtualMemory::Mode               mode,
	                GuestBackingStore::FailureReason* failure_reason = nullptr) {
		std::lock_guard lock(m_mutex);
		LiveTrace::SlowOsCallScope slow(LiveTrace::OsMapBacking, vaddr, size);
		if (!ConsumeUnlocked(vaddr, size)) {
			if (failure_reason != nullptr) {
				*failure_reason = GuestBackingStore::FailureReason::AddressRangeUnavailable;
			}
			return false;
		}
		const auto host_mode = VirtualMemory::IsExecute(mode)
		                           ? VirtualMemory::Mode::ExecuteReadWrite
		                           : VirtualMemory::Mode::ReadWrite;
		m_backing->NoteUsed(backing_offset, size);
		if (m_backing->MapExistingPlaceholderFixed(vaddr, size, backing_offset, host_mode,
		                                           failure_reason)) {
			EXIT_IF(!AddMappedUnlocked(vaddr, size, MappingType::Backing));
			return true;
		}
		AddFreeUnlocked(vaddr, size);
		return false;
	}

	bool UnmapBacking(uint64_t vaddr, uint64_t size) {
		bool unmapped = false;
		{
			std::lock_guard lock(m_mutex);
			LiveTrace::SlowOsCallScope slow(LiveTrace::OsUnmapBacking, vaddr, size);
			bool            placeholder_preserved = false;
			if (ContainsMappedUnlocked(vaddr, size, MappingType::Backing) &&
			    m_backing->Unmap(vaddr, size, &placeholder_preserved) && placeholder_preserved) {
				EXIT_IF(!RemoveMappedUnlocked(vaddr, size, MappingType::Backing));
				AddFreeUnlocked(vaddr, size);
				unmapped = true;
			}
		}
		ReapplyRemappedViews();
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
		return FindContaining(vaddr, size) != m_free.end();
	}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	bool TestContainsFree(uint64_t vaddr, uint64_t size) {
		std::lock_guard lock(m_mutex);
		return FindContaining(vaddr, size) != m_free.end();
	}
#endif

	bool ReleaseFree(uint64_t vaddr, uint64_t size) {
		std::shared_lock lock(m_mutex);
		return FindContaining(vaddr, size) != m_free.end();
	}

	bool Commit(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		std::lock_guard lock(m_mutex);
		if (!ConsumeUnlocked(vaddr, size)) {
			return false;
		}
		const auto host_mode = VirtualMemory::IsExecute(mode)
		                           ? VirtualMemory::Mode::ExecuteReadWrite
		                           : VirtualMemory::Mode::ReadWrite;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		LiveTrace::SlowOsCallScope slow(LiveTrace::OsAlloc, vaddr, size);
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
		std::lock_guard lock(m_mutex);
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
			if (range_start >= range_end) continue;
			LiveTrace::SlowOsCallScope slow(LiveTrace::OsProtect, range_start, range_end - range_start);
			if (!VirtualMemory::Protect(range_start, range_end - range_start, mode)) {
				return false;
			}
		}
		return true;
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

	void ReserveGuestRegions() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		constexpr uint64_t MinimumRegionSize = 0x1000000;
		const auto         process           = GetCurrentProcess();
		auto               guest_end         = HOST_USER_MAX + 1u;
		auto               next_addr         = HOST_SYSTEM_MANAGED_MIN;
		// KYTY_RESERVE_CEILING=<hex>: stop below it, leaving the top of the 40-bit range to the
		// driver. DLSS frame generation (KYTY_FRAMEGEN) needs it: NVIDIA's optical flow and NGX
		// interop reserve about 44 GiB of host address space below 1 TiB when the device is
		// created, and fail it (VK_ERROR_INITIALIZATION_FAILED) when the guest took all of it.
		uint64_t ceiling = 0;
		if (const char* value = std::getenv("KYTY_RESERVE_CEILING"); value != nullptr && *value != '\0') {
			ceiling = std::strtoull(value, nullptr, 16);
		} else if (const char* framegen = std::getenv("KYTY_FRAMEGEN"); framegen != nullptr && std::atoi(framegen) > 0) {
			ceiling = 0xE000000000ull;
		}
		if (ceiling != 0) {
			guest_end = std::clamp<uint64_t>(ceiling, HOST_USER_MIN, guest_end);
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
	std::map<uint64_t, uint64_t>               m_free;
	std::map<uint64_t, MappedRegion>            m_mapped;
	std::vector<std::pair<uint64_t, uint64_t>> m_owned;
	uint64_t                                   m_granularity = 0x10000;
	GuestBackingStore*                         m_backing     = nullptr;
};
