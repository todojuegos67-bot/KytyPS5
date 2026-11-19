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

static bool GuestFreeRangeContains(uint64_t base, uint64_t range_size, uint64_t candidate,
                                   uint64_t size) {
	if (UINT64_MAX - base < range_size || candidate < base) {
		return false;
	}
	const auto range_end = base + range_size;
	return candidate <= range_end && size <= range_end - candidate;
}

// Guest ranges whose view was mapped again with the view's own access while their data stayed:
// the neighbours of a partly unmapped view (a Windows view cannot be split, so the whole view is
// unmapped and its remaining parts mapped again) and views an unmap rolled back. The page
// tracker's protection of their pages went with the old view; it is applied again once the
// address-space lock is released (GuestAddressSpace::UnmapBacking), since the tracker protects
// through that lock. From Jetsku d0111331.
thread_local std::vector<std::pair<uint64_t, uint64_t>> g_remapped_views;

static void ResyncRemappedViews() {
	for (const auto& [vaddr, size]: std::exchange(g_remapped_views, {})) {
		ResyncGpuRangeProtection(vaddr, size);
	}
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
static bool CoalescePlaceholders(uint64_t vaddr, uint64_t size) {
	return size != 0 && VirtualFree(reinterpret_cast<void*>(vaddr), size,
	                                MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) != 0;
}
#endif

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
		// SEC_COMMIT charged all of guest direct memory (13.5 GB) to the system commit limit at
		// boot, before the game had mapped any of it (ISSUES #21: with a small page file the
		// emulator could not start, or ran out of commit later). The section is reserved
		// instead, and each range is committed when the guest first maps it (EnsureCommitted);
		// a page committed through the alias below is committed in every view of the section.
		// KYTY_LAZY_COMMIT=0 commits everything up front as before.
		m_lazy_commit = [] {
			const char* value = std::getenv("KYTY_LAZY_COMMIT");
			return value == nullptr || value[0] != '0';
		}();
		m_committed.assign((m_size + CommitGranule * 64u - 1u) / (CommitGranule * 64u), 0u);
		m_handle = CreateFileMappingA(
		    INVALID_HANDLE_VALUE, nullptr,
		    PAGE_EXECUTE_READWRITE | (m_lazy_commit ? SEC_RESERVE : SEC_COMMIT),
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
		m_maps_generation.fetch_add(1, std::memory_order_acq_rel);
		m_maps.clear();
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
		m_maps_generation.fetch_add(1, std::memory_order_acq_rel);
		m_maps.clear();
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

	bool ZeroRange(uint64_t backing_offset, uint64_t size) {
		if (!IsAvailable() || !RangeOk(backing_offset, size)) {
			return false;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (m_lazy_commit) {
			// Pages not committed yet are zero when they are; touching them would fault.
			std::lock_guard lock(m_commit_mutex);
			const auto      end = backing_offset + size;
			for (auto offset = backing_offset; offset < end;) {
				const auto granule_end =
				    std::min(end, (offset / CommitGranule + 1u) * CommitGranule);
				if (IsCommittedUnlocked(offset / CommitGranule)) {
					std::memset(m_backing_base + offset, 0, granule_end - offset);
				}
				offset = granule_end;
			}
			return true;
		}
#endif
		std::memset(m_backing_base + backing_offset, 0, size);
		return true;
	}

	bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
		if (data == nullptr) {
			return false;
		}
		const auto* source = static_cast<const uint8_t*>(data);
		// Writes copy under the lock (round 82): a write racing an unmap could otherwise land in
		// backing that was released, zeroed and given to a new owner meanwhile.
		return TryTransferBacking(
		    vaddr, size,
		    [source](uint8_t* backing, uint64_t data_offset, size_t bytes) {
			    std::memcpy(backing, source + data_offset, bytes);
		    },
		    true);
	}

	bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
		if (data == nullptr) {
			return false;
		}
		auto* destination = static_cast<uint8_t*>(data);
		return TryTransferBacking(
		    vaddr, size, [destination](uint8_t* backing, uint64_t data_offset, size_t bytes) {
			    std::memcpy(destination + data_offset, backing, bytes);
		    });
	}

	// The backing bytes of [vaddr, vaddr + size) when one mapping holds all of them, else nullptr.
	[[nodiscard]] const uint8_t* FindBacking(uint64_t vaddr, uint64_t size) {
		if (!IsAvailable() || size == 0 || UINT64_MAX - vaddr < size) {
			return nullptr;
		}
		// The GPU thread's SRT walks ask for every guest page they read (GuestPageReadCache),
		// tens of thousands of times a frame, and each took this lock and a map lookup. A thread
		// remembers its recent mappings until the mapping table changes (m_maps_generation).
		// KYTY_BACKING_CACHE=0 looks up every time.
		struct Cached {
			const GuestBackingStore* owner      = nullptr;
			uint64_t                 generation = 0;
			uint64_t                 vaddr      = 0;
			uint64_t                 size       = 0;
			uint64_t                 offset     = 0;
		};
		static const bool cache_enabled = [] {
			const char* value = std::getenv("KYTY_BACKING_CACHE");
			return value == nullptr || value[0] != '0';
		}();
		thread_local std::array<Cached, 8> cache {};
		thread_local uint32_t              cache_next = 0;
		if (cache_enabled) {
			const auto generation = m_maps_generation.load(std::memory_order_acquire);
			for (const auto& cached: cache) {
				if (cached.owner == this && cached.generation == generation &&
				    vaddr >= cached.vaddr && vaddr + size <= cached.vaddr + cached.size) {
					const auto offset = cached.offset + vaddr - cached.vaddr;
					return RangeOk(offset, size) ? m_backing_base + offset : nullptr;
				}
			}
		}
		std::lock_guard lock(m_mutex);
		MapEntry        entry {};
		if (!FindContainingUnlocked(vaddr, size, &entry)) {
			return nullptr;
		}
		if (cache_enabled) {
			cache[cache_next++ % cache.size()] = {this,
			                                      m_maps_generation.load(std::memory_order_acquire),
			                                      entry.vaddr, entry.size, entry.backing_offset};
		}
		const auto offset = entry.backing_offset + vaddr - entry.vaddr;
		return RangeOk(offset, size) ? m_backing_base + offset : nullptr;
	}

	bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size) {
		if (!IsAvailable() || data == nullptr || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		std::lock_guard lock(m_mutex);
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
		auto  cursor      = vaddr;
		for (auto mapping = first; mapping != m_maps.end() && mapping->second.vaddr < end;
		     ++mapping) {
			const auto& entry  = mapping->second;
			const auto  start  = std::max(vaddr, entry.vaddr);
			const auto  bytes  = std::min(end, entry.vaddr + entry.size) - start;
			const auto  offset = entry.backing_offset + start - entry.vaddr;
			if (cursor < start) {
				std::memset(destination + (cursor - vaddr), 0, start - cursor);
			}
			std::memcpy(destination + (start - vaddr), m_backing_base + offset, bytes);
			cursor = start + bytes;
		}
		if (cursor < end) {
			std::memset(destination + (cursor - vaddr), 0, end - cursor);
		}
		return true;
	}

	bool MapFixed(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	              VirtualMemory::Mode mode, FailureReason* failure_reason = nullptr) {
		if (!IsAvailable() || !RangeOk(backing_offset, size)) {
			SetFailure(failure_reason, !IsAvailable() ? FailureReason::BackingUnavailable
			                                          : FailureReason::PhysicalRangeOutOfBounds);
			return false;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		EnsureCommitted(backing_offset, size);
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
			m_maps_generation.fetch_add(1, std::memory_order_acq_rel);
			EXIT_IF(!m_maps.emplace(vaddr, MapEntry {vaddr, size, backing_offset, mode}).second);
		}
		SetFailure(failure_reason, FailureReason::None);
		return true;
#else
		std::lock_guard lock(m_mutex);
		void* mapped = MapView(vaddr, size, backing_offset, mode, failure_reason);
		if (mapped == nullptr || reinterpret_cast<uint64_t>(mapped) != vaddr) {
			return false;
		}
		RemoveMappingsUnlocked(vaddr, size);
		m_maps_generation.fetch_add(1, std::memory_order_acq_rel);
		EXIT_IF(!m_maps.emplace(vaddr, MapEntry {vaddr, size, backing_offset, mode}).second);
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
					EXIT_IF(!MapFixed(it->vaddr, it->size, it->backing_offset, it->mode));
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
				EXIT_IF(!MapFixed(it->vaddr, it->size, it->backing_offset, it->mode));
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
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_backing_store_unmaps_before_failure == 0) {
			g_test_backing_store_unmaps_before_failure = UINT32_MAX;
			return false;
		}
		if (g_test_backing_store_unmaps_before_failure != UINT32_MAX) {
			g_test_backing_store_unmaps_before_failure--;
		}
#endif

		MapEntry old {};
		{
			std::lock_guard lock(m_mutex);
			auto            next = m_maps.upper_bound(vaddr);
			if (next != m_maps.begin()) {
				auto entry = std::prev(next);
				if (vaddr + size <= entry->second.vaddr + entry->second.size) {
					old = entry->second;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
					m_maps_generation.fetch_add(1, std::memory_order_acq_rel);
					m_maps.erase(entry);
#endif
				}
			}
		}

		if (old.size == 0) {
			return false;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		const uint64_t left_size  = vaddr - old.vaddr;
		const uint64_t right_addr = vaddr + size;
		const uint64_t right_size = old.vaddr + old.size - right_addr;
		const bool exact_unmap = (vaddr == old.vaddr && size == old.size);
		if (exact_unmap) {
			const bool unmapped =
			    UnmapViewPreservePlaceholder(reinterpret_cast<void*>(old.vaddr), old.size);
			if (!unmapped) {
				std::lock_guard lock(m_mutex);
				m_maps_generation.fetch_add(1, std::memory_order_acq_rel);
				m_maps.emplace(old.vaddr, old);
				return false;
			}
			if (placeholder_preserved != nullptr) {
				*placeholder_preserved = true;
			}
			return true;
		}

		if (UnmapViewPreservePlaceholder(reinterpret_cast<void*>(old.vaddr), old.size)) {
			bool ok = true;
			if (left_size != 0 && left_size != old.size) {
				ok = SplitPlaceholder(old.vaddr, left_size) && ok;
			}
			if (right_size != 0) {
				ok = SplitPlaceholder(vaddr, size) && ok;
			}
			if (left_size != 0) {
				const bool mapped = MapFixed(old.vaddr, left_size, old.backing_offset, old.mode);
				if (mapped) {
					g_remapped_views.emplace_back(old.vaddr, left_size);
				}
				ok = mapped && ok;
			}
			if (right_size != 0) {
				const bool mapped = MapFixed(right_addr, right_size,
				                             old.backing_offset + (right_addr - old.vaddr), old.mode);
				if (mapped) {
					g_remapped_views.emplace_back(right_addr, right_size);
				}
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
			EXIT_IF(!MapFixed(old.vaddr, old.size, old.backing_offset, old.mode));
			g_remapped_views.emplace_back(old.vaddr, old.size);
			return false;
		}
		std::lock_guard lock(m_mutex);
		m_maps_generation.fetch_add(1, std::memory_order_acq_rel);
		m_maps.emplace(old.vaddr, old);
		return false;
#else
		// Replace only the requested range so concurrent accesses and GPU watch
		// protections on the surviving mappings remain intact.
		if (!UnmapViewPreservePlaceholder(reinterpret_cast<void*>(vaddr), size)) {
			return false;
		}

		std::lock_guard lock(m_mutex);
		RemoveMappingsUnlocked(vaddr, size);

		if (placeholder_preserved != nullptr) {
			*placeholder_preserved = true;
		}
		return true;
#endif
	}

	bool Contains(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		std::lock_guard lock(m_mutex);
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

private:
	template <typename Transfer>
	bool TryTransferBacking(uint64_t vaddr, uint64_t size, Transfer transfer,
	                        bool locked_copy = false) {
		if (!IsAvailable() || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		// Round 82: a read's backing pieces are found under the lock and copied after it (from
		// the Frank fork, cecbeebb). Copying under it made game threads' reads of guest memory
		// (the clean-read emulation of their page faults) wait for the GPU thread's uploads,
		// ~30 MB a frame in a race. The backing's writable alias stays mapped for the whole run,
		// so a read that races an unmap of the range reads bytes of the old mapping, as a guest
		// read racing its own unmap would. Up to 8 pieces; a range of more mappings is copied
		// under the lock, as are writes and everything with KYTY_BISECT's backing:locked.
		struct Piece {
			uint64_t offset;
			uint64_t done;
			size_t   bytes;
		};
		std::array<Piece, 8> pieces {};
		size_t               count = 0;
		{
			std::lock_guard lock(m_mutex);
			const auto      end     = vaddr + size;
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
				if (count < pieces.size()) {
					pieces[count] = {entry.backing_offset + current - entry.vaddr, current - vaddr,
					                 static_cast<size_t>(bytes)};
				}
				count++;
				current += bytes;
			}

			if (locked_copy || count > pieces.size() || BackingCopyLocked()) {
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
		}
		for (size_t i = 0; i < count; i++) {
			transfer(m_backing_base + pieces[i].offset, pieces[i].done, pieces[i].bytes);
		}
		return true;
	}

	struct MapEntry {
		uint64_t            vaddr          = 0;
		uint64_t            size           = 0;
		uint64_t            backing_offset = 0;
		VirtualMemory::Mode mode           = VirtualMemory::Mode::NoAccess;
	};

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	void RemoveMappingsUnlocked(uint64_t vaddr, uint64_t size) {
		const auto end = vaddr + size;
		auto       it = m_maps.upper_bound(vaddr);
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
			m_maps_generation.fetch_add(1, std::memory_order_acq_rel);
			it = m_maps.erase(it);
			if (old.vaddr < vaddr) {
				m_maps_generation.fetch_add(1, std::memory_order_acq_rel);
				m_maps.emplace(old.vaddr,
				               MapEntry {old.vaddr, vaddr - old.vaddr, old.backing_offset, old.mode});
			}
			if (end < old_end) {
				m_maps_generation.fetch_add(1, std::memory_order_acq_rel);
				m_maps.emplace(end, MapEntry {end, old_end - end,
				                             old.backing_offset + end - old.vaddr, old.mode});
			}
		}
	}
#endif

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

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static constexpr uint64_t CommitGranule = 64u * 1024u;

	[[nodiscard]] bool IsCommittedUnlocked(uint64_t granule) const {
		return (m_committed[granule / 64u] >> (granule % 64u) & 1u) != 0;
	}

	// Commits the backing pages of [backing_offset, +size) that are not committed yet; see the
	// constructor. Ends the emulator with a clear message when the system is out of commit.
	void EnsureCommitted(uint64_t backing_offset, uint64_t size) {
		if (!m_lazy_commit || size == 0) {
			return;
		}
		std::lock_guard lock(m_commit_mutex);
		const auto      first = backing_offset / CommitGranule;
		const auto      last  = (backing_offset + size - 1u) / CommitGranule;
		for (auto granule = first; granule <= last;) {
			if (IsCommittedUnlocked(granule)) {
				granule++;
				continue;
			}
			auto run_end = granule + 1u;
			while (run_end <= last && !IsCommittedUnlocked(run_end)) {
				run_end++;
			}
			const auto offset = granule * CommitGranule;
			const auto bytes  = std::min(run_end * CommitGranule, m_size) - offset;
			if (VirtualAlloc(m_backing_base + offset, bytes, MEM_COMMIT, PAGE_READWRITE) ==
			    nullptr) {
				EXIT("Out of memory: Windows could not commit %" PRIu64
				     " MB of guest memory (error 0x%08" PRIx32 ", %" PRIu64
				     " MB committed so far).\nClose other programs or enlarge the page file "
				     "(System > About > Advanced system settings > Performance > Virtual "
				     "memory).\n",
				     bytes >> 20u, static_cast<uint32_t>(GetLastError()),
				     (m_committed_bytes) >> 20u);
			}
			m_committed_bytes += bytes;
			for (; granule < run_end; granule++) {
				m_committed[granule / 64u] |= uint64_t {1} << (granule % 64u);
			}
		}
	}
#endif

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
		DWORD old_protect = 0;
		return VirtualProtect(ptr, size, GetProtect(mode), &old_protect) != 0;
	}

	static bool SplitPlaceholder(uint64_t vaddr, uint64_t size) {
		if (size == 0) {
			return true;
		}
		return VirtualFree(reinterpret_cast<void*>(vaddr), size,
		                   MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) != 0;
	}

	void* ReplacePlaceholderView(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                             VirtualMemory::Mode mode, FailureReason* failure_reason) {
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
			return nullptr;
		}

		// MapViewOfFile3 gave the view its protection already, except for no access (mapped
		// read-write, then protected): upstream's "avoid redundant Windows view protection".
		if (mode == VirtualMemory::Mode::NoAccess && !ApplyProtection(ptr, size, mode)) {
			UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);
			SetFailure(failure_reason, FailureReason::ProtectFailed);
			return nullptr;
		}
		return ptr;
	}

	static bool UnmapViewPreservePlaceholder(void* vaddr, uint64_t /*size*/) {
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
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (const auto callback = g_test_before_backing_map) {
			g_test_before_backing_map = nullptr;
			callback(vaddr, size);
		}
#endif
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

	uint8_t*                     m_backing_base = nullptr;
	uint64_t                     m_size         = 0;
	std::mutex                   m_mutex;
	std::map<uint64_t, MapEntry> m_maps;
	// Changes with every change of m_maps, made under m_mutex (FindBacking's thread caches).
	std::atomic<uint64_t>        m_maps_generation {1};
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	bool                  m_lazy_commit     = false;
	std::mutex            m_commit_mutex;
	std::vector<uint64_t> m_committed; // one bit per CommitGranule of the backing
	uint64_t              m_committed_bytes = 0;
#endif
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
		ScanStamp::LayoutChanging(); // KYTY_SCAN_STAMP: before the change
		return m_backing->ZeroRange(backing_offset, size);
	}

	[[nodiscard]] bool Owns(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		return std::any_of(m_owned.begin(), m_owned.end(), [vaddr, size](const auto& range) {
			return vaddr >= range.first && vaddr + size <= range.first + range.second;
		});
	}

	[[nodiscard]] bool OverlapsOwned(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		std::lock_guard lock(m_mutex);
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

	[[nodiscard]] const uint8_t* FindBacking(uint64_t vaddr, uint64_t size) {
		return m_backing->FindBacking(vaddr, size);
	}

	bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size) {
		if (!Owns(vaddr, size)) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		auto            mapping = m_mapped.upper_bound(vaddr);
		if (mapping != m_mapped.begin()) {
			--mapping;
		}
		for (; mapping != m_mapped.end() && mapping->second.vaddr < vaddr + size; ++mapping) {
			const auto& region = mapping->second;
			if (region.type == MappingType::Private && region.vaddr + region.size > vaddr) {
				return false;
			}
		}
		return m_backing->TryReadSparseBacking(vaddr, data, size);
	}

	bool BackingContains(uint64_t vaddr, uint64_t size) { return m_backing->Contains(vaddr, size); }

	bool Protect(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		ScanStamp::LayoutChanging(); // KYTY_SCAN_STAMP: before the change
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		return ProtectMappedUnlocked(vaddr, size, mode);
	}

	// The page tracker's protections (ProtectGuestHostMemory). Round 112: only memory with
	// backing, the memory the GPU can use; the program's code, stacks and runtime memory (private
	// mappings) keep their own. A GPU buffer the game pointed at the start of its code (at an exit
	// to the menu after a license test) made that code page read-only, then writable without
	// execute, and every thread calling through it faulted on it for good: a freeze.
	bool ProtectTransient(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}

		// Shared (mainline #1815): the page tracker protects from every faulting game thread and
		// from the GPU thread, and this only reads m_mapped; the page manager's region lock
		// already orders changes to the same pages. Everything that changes m_mapped, and the
		// game's own mprotect, still takes the lock exclusively (and these too with
		// KYTY_BISECT's protect:exclusive, to compare).
		if (ProtectLockExclusive()) {
			std::unique_lock lock(m_mutex);
			return ProtectMappedUnlocked(vaddr, size, mode, true);
		}
		std::shared_lock lock(m_mutex);
		return ProtectMappedUnlocked(vaddr, size, mode, true);
	}

	// Round 127: the stacks of live threads that run on memory the game gave them
	// (scePthreadAttrSetstack), in its own mapped memory. The page tracker's protections leave them
	// writable (ProtectMappedUnlocked). A write to a protected page of the stack a thread runs on
	// cannot be reported: Windows puts the fault's record on that same stack, finds it read-only,
	// and ends the process at once, with no handler run and nothing in the log. In Scapes, about
	// 14 minutes in as a scene loaded, a GPU buffer over a 2 MB block made a worker's stack in it
	// read-only (an access violation in ntdll on that stack, from a crash dump). What the tracker
	// protected in the range before is made writable at once.
	void AddLiveStack(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size + HostPageSize()) {
			return;
		}
		const auto      start = AlignDown(vaddr, HostPageSize());
		const auto      end   = AlignUp(vaddr + size, HostPageSize());
		// KYTY_SCAN_STAMP: known as a live stack (never watched) before it becomes writable.
		ScanStamp::AddLiveStack(start, end);
		std::lock_guard lock(m_mutex);
		m_live_stacks.emplace(start, end);
		auto it = m_mapped.upper_bound(start);
		if (it != m_mapped.begin()) {
			--it;
		}
		for (; it != m_mapped.end() && it->second.vaddr < end; ++it) {
			const auto& region      = it->second;
			const auto  range_start = std::max(start, region.vaddr);
			const auto  range_end   = std::min(end, region.vaddr + region.size);
			if (region.type == MappingType::Backing && range_start < range_end) {
				(void)VirtualMemory::Protect(range_start, range_end - range_start,
				                             VirtualMemory::Mode::ReadWrite);
			}
		}
	}

	// Whether the range was a live stack (AddLiveStack); the tracker may protect it again.
	bool RemoveLiveStack(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size + HostPageSize()) {
			return false;
		}
		const auto      start = AlignDown(vaddr, HostPageSize());
		const auto      end   = AlignUp(vaddr + size, HostPageSize());
		std::lock_guard lock(m_mutex);
		const auto [first, last] = m_live_stacks.equal_range(start);
		for (auto it = first; it != last; ++it) {
			if (it->second == end) {
				m_live_stacks.erase(it);
				ScanStamp::RemoveLiveStack(start, end);
				return true;
			}
		}
		return false;
	}

	bool MapBacking(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                VirtualMemory::Mode               mode,
	                GuestBackingStore::FailureReason* failure_reason = nullptr) {
		ScanStamp::LayoutChanging(); // KYTY_SCAN_STAMP: before the change
		std::lock_guard lock(m_mutex);
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
				EXIT_IF(!AddMappedUnlocked(vaddr, size, MappingType::Backing));
			}
			return true;
		}
		if (!replacing) {
			AddFreeUnlocked(vaddr, size);
		}
		return false;
	}

	bool UnmapBacking(uint64_t vaddr, uint64_t size) {
		ScanStamp::LayoutChanging(); // KYTY_SCAN_STAMP: before the change
		bool unmapped = false;
		{
			std::lock_guard lock(m_mutex);
			bool            placeholder_preserved = false;
			if (ContainsMappedUnlocked(vaddr, size, MappingType::Backing) &&
			    m_backing->Unmap(vaddr, size, &placeholder_preserved) && placeholder_preserved) {
				EXIT_IF(!RemoveMappedUnlocked(vaddr, size, MappingType::Backing));
				AddFreeUnlocked(vaddr, size);
				unmapped = true;
			}
		}
		// Outside the lock: the page tracker protects through it (ProtectTransient).
		ResyncRemappedViews();
		return unmapped;
	}

	uint64_t ReserveAligned(uint64_t search_addr, uint64_t size, uint64_t alignment) {
		if (size == 0) {
			return 0;
		}
		alignment = std::max<uint64_t>(alignment != 0 ? alignment : PageSize(), PageSize());
		std::lock_guard lock(m_mutex);
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
		std::lock_guard lock(m_mutex);
		return FindFreeAlignedUnlocked(search_start, search_end, size, alignment);
	}

	bool ReserveFixed(uint64_t vaddr, uint64_t size) {
		ScanStamp::LayoutChanging(); // KYTY_SCAN_STAMP: before the change
		if (vaddr == 0 || size == 0 || !IsPageAligned(vaddr) || !IsPageAligned(size)) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		return FindContaining(vaddr, size) != m_free.end();
	}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	bool TestContainsFree(uint64_t vaddr, uint64_t size) {
		std::lock_guard lock(m_mutex);
		return FindContaining(vaddr, size) != m_free.end();
	}
#endif

	bool ReleaseFree(uint64_t vaddr, uint64_t size) {
		ScanStamp::LayoutChanging(); // KYTY_SCAN_STAMP: before the change
		std::lock_guard lock(m_mutex);
		return FindContaining(vaddr, size) != m_free.end();
	}

	bool Commit(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		ScanStamp::LayoutChanging(); // KYTY_SCAN_STAMP: before the change
		std::lock_guard lock(m_mutex);
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
		ScanStamp::LayoutChanging(); // KYTY_SCAN_STAMP: before the change
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
	// The host's page, the page tracker's (TRACKER_PAGE_SIZE).
	static constexpr uint64_t HostPageSize() { return 0x1000; }

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

	bool ProtectMappedUnlocked(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode,
	                           bool backing_only = false) {
		const auto end = vaddr + size;
		auto       it  = m_mapped.upper_bound(vaddr);
		if (it != m_mapped.begin()) {
			--it;
		}
		for (; it != m_mapped.end() && it->second.vaddr < end; ++it) {
			const auto& region      = it->second;
			const auto  range_start = std::max(vaddr, region.vaddr);
			const auto  range_end   = std::min(end, region.vaddr + region.size);
			if (backing_only && region.type != MappingType::Backing) {
				if (range_start < range_end) {
					static std::atomic<uint32_t> reported {0};
					if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
						LOGF(
						    "Page tracking: left the protection of private memory (program code, a "
						    "stack) at 0x%016" PRIx64 "+0x%" PRIx64
						    " alone (a GPU buffer over it)\n",
						    range_start, range_end - range_start);
					}
				}
				continue;
			}
			if (range_start >= range_end) {
				continue;
			}
			// Round 127: a protection that takes write access away leaves live stacks out.
			if (backing_only && !m_live_stacks.empty() &&
			    (static_cast<uint32_t>(mode) & static_cast<uint32_t>(VirtualMemory::Mode::Write)) ==
			        0) {
				if (!ProtectAroundLiveStacksUnlocked(range_start, range_end, mode)) {
					return false;
				}
				continue;
			}
			if (!VirtualMemory::Protect(range_start, range_end - range_start, mode)) {
				return false;
			}
		}
		return true;
	}

	// [start, end) less the live stacks in it (AddLiveStack), each piece given `mode`.
	bool ProtectAroundLiveStacksUnlocked(uint64_t start, uint64_t end, VirtualMemory::Mode mode) {
		auto at = start;
		for (const auto& [stack_start, stack_end]: m_live_stacks) {
			if (stack_start >= end) {
				break;
			}
			if (stack_end <= at) {
				continue;
			}
			if (stack_start > at && !VirtualMemory::Protect(at, stack_start - at, mode)) {
				return false;
			}
			static std::atomic<uint32_t> reported {0};
			if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
				LOGF("Page tracking: left a live thread's stack at 0x%016" PRIx64 "+0x%" PRIx64
				     " writable (a GPU buffer over it; round 127)\n",
				     std::max(at, stack_start),
				     std::min(end, stack_end) - std::max(at, stack_start));
			}
			at = std::max(at, stack_end);
			if (at >= end) {
				return true;
			}
		}
		return VirtualMemory::Protect(at, end - at, mode);
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
		const std::array<std::pair<uint64_t, uint64_t>, 3> regions {{
		    {HOST_SYSTEM_MANAGED_MIN, HOST_SYSTEM_RESERVED_MAX + 1u},
		    {HOST_USER_MIN, HOST_USER_MAX + 1u},
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

				if (region_end <= next_addr) {
					EXIT("guest address-space scan did not advance at 0x%016" PRIx64 "\n", next_addr);
				}
				next_addr = region_end;
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
		for (const auto& [base, range_size]: m_free) {
			const auto candidate = AlignUp(std::max(base, search_start), alignment);
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

	std::shared_mutex                           m_mutex;
	std::map<uint64_t, uint64_t>               m_free;
	std::map<uint64_t, MappedRegion>            m_mapped;
	std::vector<std::pair<uint64_t, uint64_t>> m_owned;
	// AddLiveStack's ranges, host-page aligned: start to end.
	std::multimap<uint64_t, uint64_t>          m_live_stacks;
	uint64_t                                   m_granularity = 0x10000;
	GuestBackingStore*                         m_backing     = nullptr;
};
