#include "kernel/memory.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "libs/errno.h"
#include "libs/libs.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <magic_enum.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#ifndef MEM_RESERVE_PLACEHOLDER
#define MEM_RESERVE_PLACEHOLDER 0x00040000
#endif
#ifndef MEM_REPLACE_PLACEHOLDER
#define MEM_REPLACE_PLACEHOLDER 0x00004000
#endif
#ifndef MEM_PRESERVE_PLACEHOLDER
#define MEM_PRESERVE_PLACEHOLDER 0x00000002
#endif
#ifndef MEM_COALESCE_PLACEHOLDERS
#define MEM_COALESCE_PLACEHOLDERS 0x00000001
#endif
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::LibKernel::Memory {

namespace VirtualMemory = Common::VirtualMemory;

LIB_NAME("libkernel", "libkernel");

constexpr int PROT_CPU_READ  = 0x01;
constexpr int PROT_CPU_WRITE = 0x02;
constexpr int PROT_CPU_EXEC  = 0x04;
constexpr int PROT_GPU_READ  = 0x10;
constexpr int PROT_GPU_WRITE = 0x20;

enum class GpuAccessMode { NoAccess, Read, Write, ReadWrite };

constexpr uint64_t PAGE_TABLE_POOL_SIZE   = 4ull * 1024ull * 1024ull * 1024ull;
constexpr uint64_t PAGE_TABLE_GRANULARITY = 2ull * 1024ull * 1024ull;
constexpr int      PAGE_TABLE_POOL_ENTRIES =
    static_cast<int>(PAGE_TABLE_POOL_SIZE / PAGE_TABLE_GRANULARITY);
constexpr uint64_t DEFAULT_FLEXIBLE_MEMORY_SIZE = 1ull * 1024ull * 1024ull * 1024ull;

static uint64_t                      g_flexible_memory_size        = DEFAULT_FLEXIBLE_MEMORY_SIZE;
static bool                          g_flexible_memory_size_frozen = false;

// Round 81: room past the guest's direct memory (and its flexible memory) for direct allocations
// that do not fit in it. The game asks only within the size it is told (0x320000000, 12800 MiB),
// and some loads of the same race took it past that: one run held a 2.3 GB block another run of
// the race did without, its next allocations failed, and it crashed on the null pointer one of
// them returned (round 79 and 80 logs, the files it read the same in both). Such an allocation
// is given room here instead, logged; the size the game is told does not change, so nothing it
// plans by grows. KYTY_DMEM_OVERFLOW_MB (default 16384; 0: none, as before) sets the room. The
// backing behind it is reserved, not committed, until it is used; with KYTY_LAZY_COMMIT=0
// (Windows commits the whole backing at boot) there is none unless KYTY_DMEM_OVERFLOW_MB asks.
static uint64_t DirectMemoryOverflowSize() {
	static const uint64_t size = [] {
		// 16 GB (round 89; 8 GB from round 82, 4 GB in round 81): the race's 5.3 GB pool did not
		// fit in 4 GB when the game held a 2.3 GB block or two more than usual, and the game froze
		// at the track menu. Loading a track (the first one after the start too, about one load
		// in ten), it still held two 2.3 GB blocks and a 1.6 GB one past its memory, the pool did
		// not fit in 8 GB either, and its loading screen never ended.
		uint64_t mb = 16384;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (const char* eager = std::getenv("KYTY_LAZY_COMMIT");
		    eager != nullptr && eager[0] == '0') {
			mb = 0;
		}
#endif
		// Decimal MiB only (a value with anything else keeps the default).
		if (const char* value = std::getenv("KYTY_DMEM_OVERFLOW_MB");
		    value != nullptr && value[0] != '\0' &&
		    std::strspn(value, "0123456789") == std::strlen(value) && std::strlen(value) <= 6) {
			mb = std::strtoull(value, nullptr, 10);
		}
		// Whole 2 MiB pages (rounded up), at most 64 GiB.
		return ((std::min<uint64_t>(mb, 65536u) + 1u) & ~uint64_t {1}) << 20u;
	}();
	return size;
}
// The guest code of the direct allocation this thread is making (round 82), for its log lines.
static thread_local uint64_t          t_direct_memory_caller        = 0;
static Graphics::RenderContext*       g_gpu_resources               = nullptr;

static Graphics::RenderContext& GetGpuResources() {
	EXIT_IF(g_gpu_resources == nullptr);
	return *g_gpu_resources;
}

static bool IsGpuAddressRange(uint64_t vaddr, uint64_t size) {
	return Graphics::GuestRange {vaddr, size}.Valid();
}

static void MapGpuRange(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) {
		return;
	}
	GetGpuResources().MapMemory(vaddr, size);
}

static void UnmapGpuRange(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) {
		return;
	}
	GetGpuResources().UnmapMemory(vaddr, size);
}

// KYTY_REMAP_TRACKER_PROTECTION=0 turns it off: after a range's host protection was replaced
// behind the page tracker (the neighbours of a partly unmapped Windows view mapped again with the
// view's access, views an unmap rolled back, a guest mprotect that allows writes), its watched
// pages get the tracker's protection back. Without it CPU writes to still-tracked buffers there
// stopped faulting and the GPU kept using stale bytes. From Jetsku d0111331 (chenxiao07
// 3f3825158 / e6dfd8e6b).
static bool RemapTrackerProtectionEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_REMAP_TRACKER_PROTECTION");
		return value == nullptr || !(value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

static void ResyncGpuRangeProtection(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size) ||
	    !RemapTrackerProtectionEnabled()) {
		return;
	}
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
		LOGF("Remap protection: resync 0x%016" PRIx64 " size 0x%" PRIx64 " (first 8 logged)\n", vaddr,
		     size);
	}
	GetGpuResources().ResyncHostProtection(vaddr, size);
}

static bool DecodeMemoryProtection(int prot, VirtualMemory::Mode* mode, GpuAccessMode* gpu_mode) {
	EXIT_IF(mode == nullptr);
	EXIT_IF(gpu_mode == nullptr);

	bool cpu_read  = (prot & PROT_CPU_READ) != 0;
	bool cpu_write = (prot & PROT_CPU_WRITE) != 0;
	bool cpu_exec  = (prot & PROT_CPU_EXEC) != 0;
	bool gpu_read  = (prot & PROT_GPU_READ) != 0;
	bool gpu_write = (prot & PROT_GPU_WRITE) != 0;

	if ((prot &
	     (PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC | PROT_GPU_READ | PROT_GPU_WRITE)) == 0 &&
	    prot != 0) {
		return false;
	}

	if (gpu_read && gpu_write) {
		*gpu_mode = GpuAccessMode::ReadWrite;
	} else if (gpu_read) {
		*gpu_mode = GpuAccessMode::Read;
	} else if (gpu_write) {
		*gpu_mode = GpuAccessMode::Write;
	} else {
		*gpu_mode = GpuAccessMode::NoAccess;
	}

	bool host_read  = cpu_read || gpu_read;
	bool host_write = cpu_write || gpu_write;

	if (host_write) {
		host_read = true;
	}

	*mode = VirtualMemory::Mode::NoAccess;
	if (cpu_exec) {
		*mode = (host_write ? VirtualMemory::Mode::ExecuteReadWrite
		                    : (host_read ? VirtualMemory::Mode::ExecuteRead
		                                 : VirtualMemory::Mode::Execute));
	} else if (host_write) {
		*mode = VirtualMemory::Mode::ReadWrite;
	} else if (host_read) {
		*mode = VirtualMemory::Mode::Read;
	}

	return true;
}

static void CopyVirtualRangeName(char* dst, const char* name) {
	EXIT_IF(dst == nullptr);

	std::memset(dst, 0, KERNEL_MAXIMUM_NAME_LENGTH);

	if (name != nullptr) {
		std::strncpy(dst, name, KERNEL_MAXIMUM_NAME_LENGTH - 1);
	}
}

static bool VirtualRangesOverlap(uint64_t left_start, uint64_t left_size, uint64_t right_start,
                                 uint64_t right_size) {
	if (left_size == 0 || right_size == 0) {
		return false;
	}

	auto left_end = (UINT64_MAX - left_start < left_size ? UINT64_MAX : left_start + left_size);
	auto right_end =
	    (UINT64_MAX - right_start < right_size ? UINT64_MAX : right_start + right_size);

	return left_start < right_end && right_start < left_end;
}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
static uint32_t        g_test_backing_store_unmaps_before_failure = UINT32_MAX;
static callback_func_t g_test_before_backing_map                   = nullptr;
#endif

// KYTY_BISECT's backing:locked (round 82; KYTY_BACKING_COPY_LOCKED=1 for the whole run): copies
// to and from guest memory hold the backing store's lock while they copy, as before.
static std::atomic<bool> g_backing_copy_locked_step {false};

static bool BackingCopyLocked() {
	static const bool always = [] {
		const char* value = std::getenv("KYTY_BACKING_COPY_LOCKED");
		return value != nullptr && value[0] == '1';
	}();
	return always || g_backing_copy_locked_step.load(std::memory_order_relaxed);
}

// Round 135, KYTY_BISECT's protect:exclusive (KYTY_PROTECT_LOCK=exclusive for the whole run): the
// page tracker's protections take the address space's lock exclusively, as before mainline #1815
// let them share it.
static std::atomic<bool> g_protect_exclusive_step {false};

static bool ProtectLockExclusive() {
	static const bool always = [] {
		const char* value = std::getenv("KYTY_PROTECT_LOCK");
		return value != nullptr && std::strcmp(value, "exclusive") == 0;
	}();
	return always || g_protect_exclusive_step.load(std::memory_order_relaxed);
}

#include "memoryAddressSpace.inc"

enum class VirtualRangeType {
	Reserved,
	PoolReserved,
	Direct,
	Flexible,
	Pooled,
	Stack,
	Code,
	Runtime,
};

static bool IsReservedRangeType(VirtualRangeType type) {
	return type == VirtualRangeType::Reserved || type == VirtualRangeType::PoolReserved;
}

static bool IsPooledRangeType(VirtualRangeType type) {
	return type == VirtualRangeType::Pooled || type == VirtualRangeType::PoolReserved;
}

static bool IsCommittedRangeType(VirtualRangeType type) {
	return !IsReservedRangeType(type);
}

static bool IsPrivateCommittedRangeType(VirtualRangeType type) {
	return type == VirtualRangeType::Stack || type == VirtualRangeType::Code ||
	       type == VirtualRangeType::Runtime;
}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
static bool g_test_fail_next_range_replace = false;
#endif

// KYTY_TRACE_MEMORY (memory.h): the watched ranges.
struct MemoryWatch {
	uint64_t address = 0;
	uint64_t size    = 0;
};

static const std::vector<MemoryWatch>& MemoryWatches() {
	static const std::vector<MemoryWatch> watches = [] {
		std::vector<MemoryWatch> result;
		const char*              text = std::getenv("KYTY_TRACE_MEMORY");
		while (text != nullptr && *text != '\0') {
			char*      end     = nullptr;
			const auto address = std::strtoull(text, &end, 16);
			if (end == text) {
				break;
			}
			uint64_t size = 1;
			if (*end == '+') {
				text = end + 1;
				size = std::max<uint64_t>(std::strtoull(text, &end, 16), 1u);
			}
			result.push_back({address, size});
			text = *end == ',' ? end + 1 : nullptr;
		}
		for (const auto& watch: result) {
			LOGF("Trace memory: watching 0x%016" PRIx64 "+0x%" PRIx64 " (KYTY_TRACE_MEMORY)\n",
			     watch.address, watch.size);
		}
		return result;
	}();
	return watches;
}

bool TracesMemory(uint64_t vaddr, uint64_t size) {
	const auto& watches = MemoryWatches();
	if (watches.empty() || size == 0) {
		return false;
	}
	return std::any_of(watches.begin(), watches.end(), [&](const MemoryWatch& watch) {
		return vaddr < watch.address + watch.size && watch.address < vaddr + size;
	});
}

void TraceMemory(const char* format, ...) {
	char    text[1024];
	va_list args;
	va_start(args, format);
	std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	LOGF("Trace memory: %s\n", text);
}

// The kernel's own mapping changes over a watched range: the first 400, then at powers of two.
static void TraceRangeChange(const char* what, uint64_t start, uint64_t size, const char* type,
                             int protection, uint64_t offset) {
	if (!TracesMemory(start, size)) {
		return;
	}
	static std::atomic<uint64_t> changes {0};
	const auto                   count = ++changes;
	if (count > 400 && !std::has_single_bit(count)) {
		return;
	}
	TraceMemory("%s 0x%016" PRIx64 "+0x%" PRIx64 " %s prot 0x%x offset 0x%" PRIx64 " (%" PRIu64
	            " changes so far)",
	            what, start, size, type, protection, offset, count);
}

// The guest's virtual address ranges, sorted by start, never overlapping, and merged with their
// neighbours where SameMergeKey allows. Round 88: an ordered map instead of a vector, and the
// committed bytes kept as merged spans as the ranges change. Each change rebuilt and re-sorted
// the whole vector (and ReplaceSpan and QuerySpan walked it from the start): the game maps
// streaming memory thousands of times while it loads a track, over tens of thousands of ranges,
// and its loading threads spent minutes there while it drew nothing (2026-10-18, warm87: 80 s at
// a time with no frame). Every change is now local, logarithmic in the number of ranges.
class VirtualRanges {
public:
	struct Range {
		uint64_t         start          = 0;
		uint64_t         size           = 0;
		uint64_t         offset         = 0;
		int              protection     = 0;
		int              memory_type    = 0;
		VirtualRangeType type           = VirtualRangeType::Reserved;
		bool             disallow_merge = false;
		char             name[KERNEL_MAXIMUM_NAME_LENGTH];
	};

	bool Add(uint64_t start, uint64_t size, uint64_t offset, int protection, int memory_type,
	         VirtualRangeType type, const char* name, bool disallow_merge = false) {
		Common::LockGuard lock(m_mutex);
		m_generation.fetch_add(1, std::memory_order_acq_rel);
		m_coverage_generation.fetch_add(1, std::memory_order_acq_rel);

		if (start == 0 || size == 0 || size > UINT64_MAX - start) {
			return false;
		}
		if (FindOverlap(start, size) != nullptr) {
			return false;
		}

		Range r {};
		r.start          = start;
		r.size           = size;
		r.offset         = offset;
		r.protection     = protection;
		r.memory_type    = memory_type;
		r.type           = type;
		r.disallow_merge = disallow_merge;
		CopyVirtualRangeName(r.name, name);
		MergeAroundUnlocked(InsertUnlocked(r));
		TraceRangeChange("mapped", start, size, magic_enum::enum_name(type).data(), protection,
		                 offset);
		return true;
	}

	bool Remove(uint64_t start, uint64_t size) {
		Common::LockGuard lock(m_mutex);
		m_generation.fetch_add(1, std::memory_order_acq_rel);
		m_coverage_generation.fetch_add(1, std::memory_order_acq_rel);
		TraceRangeChange("unmapped", start, size, "-", 0, 0);

		if (auto exact = m_ranges.find(start);
		    exact != m_ranges.end() && exact->second.size == size) {
			EraseUnlocked(exact);
			return true;
		}
		// Removing makes no new neighbours, so nothing new can merge.
		return RemoveUnlocked(start, size);
	}

	bool HasOverlap(uint64_t start, uint64_t size) {
		Common::LockGuard lock(m_mutex);

		return FindOverlap(start, size) != nullptr;
	}

	bool QueryOverlap(uint64_t start, uint64_t size, Range* out) {
		EXIT_IF(out == nullptr);
		Common::LockGuard lock(m_mutex);

		const auto* overlap = FindOverlap(start, size);
		if (overlap == nullptr) {
			return false;
		}
		*out = *overlap;
		return true;
	}

	bool ReleaseReserved(uint64_t start, uint64_t size) {
		Common::LockGuard lock(m_mutex);
		m_generation.fetch_add(1, std::memory_order_acq_rel);
		m_coverage_generation.fetch_add(1, std::memory_order_acq_rel);
		TraceRangeChange("reservation released", start, size, "-", 0, 0);

		if (auto exact = m_ranges.find(start); exact != m_ranges.end() &&
		                                       exact->second.size == size &&
		                                       IsReservedRangeType(exact->second.type)) {
			EraseUnlocked(exact);
		}
		return true;
	}

	bool ReplaceSpan(uint64_t start, uint64_t size, VirtualRangeType expected_type,
	                 uint64_t offset, int protection, int memory_type, VirtualRangeType type,
	                 const char* name, bool disallow_merge = false) {
		Common::LockGuard lock(m_mutex);
		m_generation.fetch_add(1, std::memory_order_acq_rel);
		m_coverage_generation.fetch_add(1, std::memory_order_acq_rel);

		if (start == 0 || size == 0 || size > UINT64_MAX - start) {
			return false;
		}

		// Every byte must lie in a range of the expected type.
		auto       current = start;
		const auto end     = start + size;
		while (current < end) {
			const auto containing = ContainingUnlocked(current);
			if (containing == m_ranges.end() || containing->second.type != expected_type) {
				return false;
			}
			current = std::min(end, End(containing->second.start, containing->second.size));
		}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_fail_next_range_replace) {
			g_test_fail_next_range_replace = false;
			return false;
		}
#endif
		TraceRangeChange("replaced", start, size, magic_enum::enum_name(type).data(), protection,
		                 offset);
		Range replacement {};
		replacement.start          = start;
		replacement.size           = size;
		replacement.offset         = offset;
		replacement.protection     = protection;
		replacement.memory_type    = memory_type;
		replacement.type           = type;
		replacement.disallow_merge = disallow_merge;
		CopyVirtualRangeName(replacement.name, name);
		RemoveUnlocked(start, size);
		MergeAroundUnlocked(InsertUnlocked(replacement));
		return true;
	}

	void Rename(uint64_t start, uint64_t size, const char* name) {
		Common::LockGuard lock(m_mutex);
		m_generation.fetch_add(1, std::memory_order_acq_rel);

		if (auto exact = m_ranges.find(start);
		    exact != m_ranges.end() && exact->second.size == size) {
			CopyVirtualRangeName(exact->second.name, name);
			MergeAroundUnlocked(exact);
			return;
		}
		EditUnlocked(start, size, [name](Range* r) { CopyVirtualRangeName(r->name, name); });
	}

	void Protect(uint64_t start, uint64_t size, int protection) {
		Common::LockGuard lock(m_mutex);
		m_generation.fetch_add(1, std::memory_order_acq_rel);
		TraceRangeChange("protected", start, size, "-", protection, 0);

		EditUnlocked(start, size, [protection](Range* r) { r->protection = protection; });
	}

	void SetMemoryType(uint64_t start, uint64_t size, int memory_type) {
		Common::LockGuard lock(m_mutex);
		m_generation.fetch_add(1, std::memory_order_acq_rel);

		EditUnlocked(start, size, [memory_type](Range* r) { r->memory_type = memory_type; });
	}

	bool Query(uint64_t addr, int flags, Range* out) {
		EXIT_IF(out == nullptr);

		Common::LockGuard lock(m_mutex);

		auto next = m_ranges.upper_bound(addr);
		if (next != m_ranges.begin()) {
			const auto& current = std::prev(next)->second;
			if (addr < End(current.start, current.size)) {
				*out = current;
				return true;
			}
		}
		if (flags != 1 || next == m_ranges.end()) {
			return false;
		}

		*out = next->second;
		return true;
	}

	bool QuerySpan(uint64_t start, uint64_t size, std::vector<Range>* out) {
		EXIT_IF(out == nullptr);

		Common::LockGuard lock(m_mutex);
		out->clear();
		if (start == 0 || size == 0 || size > UINT64_MAX - start) {
			return false;
		}

		const auto end     = start + size;
		auto       current = start;
		for (auto it = FirstEndingAfterUnlocked(start); it != m_ranges.end(); ++it) {
			const auto& range     = it->second;
			const auto  range_end = End(range.start, range.size);
			if (range_end <= current) {
				continue;
			}
			if (range.start > current) {
				break;
			}

			Range part = range;
			part.start = current;
			part.size  = std::min(end, range_end) - current;
			if (part.type == VirtualRangeType::Direct) {
				part.offset += current - range.start;
			}
			out->push_back(part);
			current += part.size;
			if (current == end) {
				return true;
			}
		}

		out->clear();
		return false;
	}

	[[nodiscard]] uint64_t Generation() const noexcept {
		return m_generation.load(std::memory_order_acquire);
	}

	// Bumped only by the changes that can change which bytes are committed (add, remove,
	// replace, release); protection, name and memory type changes leave it.
	[[nodiscard]] uint64_t CoverageGeneration() const noexcept {
		return m_coverage_generation.load(std::memory_order_acquire);
	}

	// DescribeMappings (memory.h).
	std::string Describe(uint64_t start, uint64_t size) {
		Common::LockGuard lock(m_mutex);
		if (start == 0 || size == 0 || size > UINT64_MAX - start) {
			return "(none)";
		}
		const auto  end     = start + size;
		auto        current = start;
		std::string text;
		size_t      pieces = 0;
		const auto  add    = [&](uint64_t from, uint64_t to, const char* what, int protection) {
            if (++pieces > 48) {
                return;
            }
            char piece[160];
            std::snprintf(piece, sizeof(piece), "%s+0x%" PRIx64 " 0x%" PRIx64 " %s%s",
                          text.empty() ? "" : ", ", from - start, to - from, what,
                          protection != 0 ? (protection & 2 ? " rw" : " r") : "");
            text += piece;
		};
		for (auto it = FirstEndingAfterUnlocked(start); it != m_ranges.end(); ++it) {
			const auto& r         = it->second;
			const auto  range_end = End(r.start, r.size);
			if (range_end <= current) {
				continue;
			}
			if (r.start >= end) {
				break;
			}
			if (r.start > current) {
				add(current, r.start, "hole", 0);
				current = r.start;
			}
			const auto to = std::min(end, range_end);
			add(current, to, magic_enum::enum_name(r.type).data(), r.protection);
			current = to;
			if (current == end) {
				break;
			}
		}
		if (current < end) {
			add(current, end, "hole", 0);
		}
		if (pieces > 48) {
			text += ", ... (" + std::to_string(pieces) + " pieces)";
		}
		return text;
	}

	bool CommittedSpans(uint64_t start, uint64_t size, size_t max_spans,
	                    std::vector<std::pair<uint64_t, uint64_t>>* spans) {
		Common::LockGuard lock(m_mutex);
		spans->clear();
		if (start == 0 || size == 0 || size > UINT64_MAX - start) {
			return true;
		}
		const auto end  = start + size;
		auto       span = m_coverage.upper_bound(start);
		if (span != m_coverage.begin()) {
			--span;
		}
		for (; span != m_coverage.end() && span->first < end; ++span) {
			const auto from = std::max(start, span->first);
			const auto to   = std::min(end, span->second);
			if (from >= to) {
				continue;
			}
			if (spans->size() >= max_spans) {
				spans->clear();
				return false;
			}
			spans->emplace_back(from, to - from);
		}
		return true;
	}

	uint64_t CommittedExtent(uint64_t start, uint64_t size) {
		Common::LockGuard lock(m_mutex);
		if (start == 0 || size == 0 || size > UINT64_MAX - start) {
			return 0;
		}
		const auto end = start + size;
		// The last span that starts before the end.
		auto span = m_coverage.lower_bound(end);
		if (span == m_coverage.begin()) {
			return 0;
		}
		--span;
		return span->second > start ? std::min(end, span->second) - start : 0;
	}

	// Round 133: span_begin and span_end, when given, get the merged committed span the answer
	// came from (KYTY_CLAMP_SPANS), for TryClampRangeSize's span memo; left alone otherwise.
	uint64_t ClampRangeSize(uint64_t virtual_addr, uint64_t size, uint64_t* span_begin = nullptr,
	                        uint64_t* span_end = nullptr) {
		Common::LockGuard lock(m_mutex);

		if (virtual_addr == 0 || size == 0 || size > UINT64_MAX - virtual_addr) {
			return 0;
		}
		// Committed bytes as merged spans: a heap-wide descriptor spans thousands of mappings,
		// and while the game streams (it maps memory hundreds of times a second) the per-thread
		// answers of TryClampRangeSize keep missing, so the walk below ran under the lock at ~4 us
		// a call (the helicopter cut, 2026-10-09). KYTY_CLAMP_SPANS=0 walks the ranges as before.
		static const bool spans = [] {
			const char* value = std::getenv("KYTY_CLAMP_SPANS");
			return value == nullptr || value[0] != '0';
		}();
		if (spans) {
			auto span = m_coverage.upper_bound(virtual_addr);
			if (span == m_coverage.begin()) {
				return 0;
			}
			--span;
			if (virtual_addr >= span->second) {
				return 0;
			}
			if (span_begin != nullptr && span_end != nullptr) {
				*span_begin = span->first;
				*span_end   = span->second;
			}
			return std::min(size, span->second - virtual_addr);
		}

		auto vma = m_ranges.upper_bound(virtual_addr);
		if (vma == m_ranges.begin()) {
			return 0;
		}
		--vma;

		const auto vma_end = End(vma->second.start, vma->second.size);
		if (virtual_addr < vma->second.start || virtual_addr >= vma_end ||
		    !IsCommittedRangeType(vma->second.type)) {
			return 0;
		}

		uint64_t clamped_size = std::min(size, vma_end - virtual_addr);
		uint64_t expected     = virtual_addr + clamped_size;
		++vma;

		while (vma != m_ranges.end() && vma->second.start == expected &&
		       IsCommittedRangeType(vma->second.type) && clamped_size < size) {
			const auto chunk = std::min(size - clamped_size, vma->second.size);
			clamped_size += chunk;
			expected += chunk;
			++vma;
		}

		return clamped_size;
	}

	uint64_t CountPageTableEntries(bool gpu) {
		Common::LockGuard lock(m_mutex);

		uint64_t used = 0;
		for (const auto& [start, r]: m_ranges) {
			(void)start;
			if (!IsCommittedRangeType(r.type) || r.size == 0) {
				continue;
			}

			const bool has_cpu_access =
			    (r.protection & (PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC)) != 0;
			const bool has_gpu_access = (r.protection & (PROT_GPU_READ | PROT_GPU_WRITE)) != 0;
			if (gpu ? !has_gpu_access : !has_cpu_access) {
				continue;
			}

			const auto end         = End(r.start, r.size);
			const auto first_entry = r.start / PAGE_TABLE_GRANULARITY;
			const auto last_entry  = (end - 1) / PAGE_TABLE_GRANULARITY;
			used += last_entry - first_entry + 1;
		}

		return used;
	}

private:
	using RangeMap = std::map<uint64_t, Range>;

	static uint64_t End(uint64_t start, uint64_t size) {
		return (UINT64_MAX - start < size ? UINT64_MAX : start + size);
	}

	static bool SameMergeKey(const Range& left, const Range& right) {
		if (left.disallow_merge || right.disallow_merge || left.type == VirtualRangeType::Direct ||
		    right.type == VirtualRangeType::Direct) {
			return false;
		}

		return left.type == right.type && left.protection == right.protection &&
		       left.memory_type == right.memory_type &&
		       std::strncmp(left.name, right.name, KERNEL_MAXIMUM_NAME_LENGTH) == 0;
	}

	// The piece [start, end) of a range, as AddPiece made it.
	static Range Piece(const Range& source, uint64_t start, uint64_t end) {
		Range piece = source;
		piece.start = start;
		piece.size  = end - start;
		if (piece.type == VirtualRangeType::Direct) {
			piece.offset += start - source.start;
		}
		return piece;
	}

	// The range holding addr, or end.
	RangeMap::iterator ContainingUnlocked(uint64_t addr) {
		auto next = m_ranges.upper_bound(addr);
		if (next == m_ranges.begin()) {
			return m_ranges.end();
		}
		auto current = std::prev(next);
		return addr < End(current->second.start, current->second.size) ? current : m_ranges.end();
	}

	// The first range that ends after addr (the one holding it, else the next one).
	RangeMap::iterator FirstEndingAfterUnlocked(uint64_t addr) {
		auto next = m_ranges.upper_bound(addr);
		if (next != m_ranges.begin()) {
			auto current = std::prev(next);
			if (addr < End(current->second.start, current->second.size)) {
				return current;
			}
		}
		return next;
	}

	// A new range (nothing overlaps it), with its committed bytes.
	RangeMap::iterator InsertUnlocked(const Range& range) {
		auto [inserted, ok] = m_ranges.emplace(range.start, range);
		EXIT_IF(!ok);
		if (IsCommittedRangeType(range.type)) {
			CoverAddUnlocked(range.start, End(range.start, range.size));
		}
		return inserted;
	}

	// Removes a range, with its committed bytes; the iterator after it.
	RangeMap::iterator EraseUnlocked(RangeMap::iterator range) {
		if (IsCommittedRangeType(range->second.type)) {
			CoverRemoveUnlocked(range->second.start, End(range->second.start, range->second.size));
		}
		return m_ranges.erase(range);
	}

	// Joins the range with its neighbours while they may merge (as before, the one before it,
	// then those after it). Merging leaves the committed bytes as they are: merged ranges have
	// one type.
	RangeMap::iterator MergeAroundUnlocked(RangeMap::iterator current) {
		if (current == m_ranges.end()) {
			return current;
		}
		if (current != m_ranges.begin()) {
			auto previous = std::prev(current);
			if (End(previous->second.start, previous->second.size) == current->second.start &&
			    SameMergeKey(previous->second, current->second)) {
				previous->second.size += current->second.size;
				m_ranges.erase(current);
				current = previous;
			}
		}
		for (;;) {
			auto next = std::next(current);
			if (next == m_ranges.end() ||
			    End(current->second.start, current->second.size) != next->second.start ||
			    !SameMergeKey(current->second, next->second)) {
				break;
			}
			current->second.size += next->second.size;
			m_ranges.erase(next);
		}
		return current;
	}

	// Cuts the range holding addr in two at addr (no bytes change); the range starting at addr,
	// or the first one after it.
	RangeMap::iterator SplitAtUnlocked(uint64_t addr) {
		const auto containing = ContainingUnlocked(addr);
		if (containing == m_ranges.end()) {
			return m_ranges.lower_bound(addr);
		}
		if (containing->second.start == addr) {
			return containing;
		}
		const auto source       = containing->second;
		const auto end          = End(source.start, source.size);
		containing->second.size = addr - source.start;
		auto [tail, ok]         = m_ranges.emplace(addr, Piece(source, addr, end));
		EXIT_IF(!ok);
		return tail;
	}

	template <typename EditFunc>
	void EditUnlocked(uint64_t start, uint64_t size, EditFunc edit) {
		if (size == 0) {
			return;
		}
		const auto edit_end = End(start, size);
		auto       first    = SplitAtUnlocked(start);
		if (first == m_ranges.end() || first->second.start >= edit_end) {
			return;
		}
		SplitAtUnlocked(edit_end);
		auto last = first;
		while (last != m_ranges.end() && last->second.start < edit_end) {
			edit(&last->second);
			++last;
		}
		// Only the edited ranges and their two neighbours can merge anew.
		auto current = first != m_ranges.begin() ? std::prev(first) : first;
		while (current != m_ranges.end()) {
			const bool past = current->second.start >= edit_end;
			current         = std::next(MergeAroundUnlocked(current));
			if (past) {
				break;
			}
		}
	}

	bool RemoveUnlocked(uint64_t start, uint64_t size) {
		if (size == 0) {
			return false;
		}

		const auto rem_end = End(start, size);
		bool       removed = false;
		auto       it      = FirstEndingAfterUnlocked(start);
		while (it != m_ranges.end() && it->second.start < rem_end) {
			const auto r     = it->second;
			const auto r_end = End(r.start, r.size);
			removed          = true;
			it               = EraseUnlocked(it);
			if (r.start < start) {
				InsertUnlocked(Piece(r, r.start, start));
			}
			if (rem_end < r_end) {
				it = InsertUnlocked(Piece(r, rem_end, r_end));
				break;
			}
		}
		return removed;
	}

	// As before: the range starting at or after start, else the one before it.
	const Range* FindOverlap(uint64_t start, uint64_t size) {
		auto position = m_ranges.lower_bound(start);
		if (position != m_ranges.end() &&
		    VirtualRangesOverlap(start, size, position->second.start, position->second.size)) {
			return &position->second;
		}
		if (position != m_ranges.begin()) {
			auto previous = std::prev(position);
			if (VirtualRangesOverlap(start, size, previous->second.start, previous->second.size)) {
				return &previous->second;
			}
		}
		return nullptr;
	}

	// The committed bytes, as disjoint spans that do not touch: [first, second).
	void CoverAddUnlocked(uint64_t start, uint64_t end) {
		if (start >= end) {
			return;
		}
		auto it = m_coverage.upper_bound(start);
		if (it != m_coverage.begin()) {
			auto previous = std::prev(it);
			if (previous->second >= start) {
				start = previous->first;
				end   = std::max(end, previous->second);
				it    = m_coverage.erase(previous);
			}
		}
		while (it != m_coverage.end() && it->first <= end) {
			end = std::max(end, it->second);
			it  = m_coverage.erase(it);
		}
		m_coverage.emplace(start, end);
	}

	void CoverRemoveUnlocked(uint64_t start, uint64_t end) {
		if (start >= end) {
			return;
		}
		auto it = m_coverage.upper_bound(start);
		if (it != m_coverage.begin()) {
			--it;
		}
		while (it != m_coverage.end() && it->first < end) {
			const auto span_start = it->first;
			const auto span_end   = it->second;
			if (span_end <= start) {
				++it;
				continue;
			}
			it = m_coverage.erase(it);
			if (span_start < start) {
				m_coverage.emplace(span_start, start);
			}
			if (span_end > end) {
				m_coverage.emplace(end, span_end);
				break;
			}
		}
	}

	RangeMap                     m_ranges;
	std::map<uint64_t, uint64_t> m_coverage;
	Common::Mutex                m_mutex;
	// Bumped by every change of the ranges.
	std::atomic<uint64_t> m_generation {1};
	// Bumped by the changes of committed coverage (CoverageGeneration): TryClampRangeSize's
	// per-thread answers are valid while it is unchanged.
	std::atomic<uint64_t> m_coverage_generation {1};
};

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
static uint32_t g_test_physical_memory_unmaps_before_failure = UINT32_MAX;
static bool     g_test_fail_next_fixed_reserve_range_add     = false;
#endif

class PhysicalMemory {
public:
	enum class AllocationKind { Direct, Pooled, Automatic };
	using PhysicalRanges = std::vector<std::pair<uint64_t, uint64_t>>;

	struct AllocatedBlock {
		uint64_t            start_addr;
		uint64_t            size;
		uint64_t            map_vaddr;
		uint64_t            map_size;
		uint64_t            host_vaddr;
		uint64_t            host_size;
		int                 prot;
		VirtualMemory::Mode mode;
		GpuAccessMode       gpu_mode;
		int                 memory_type;
		AllocationKind      kind;
		char                name[KERNEL_MAXIMUM_NAME_LENGTH];
	};

	PhysicalMemory() {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
		m_free.emplace(0, Size());
		// The overflow room, past the flexible memory: searches of the direct memory stop at
		// Size(), so only Alloc's overflow search reaches it.
		if (DirectMemoryOverflowSize() != 0) {
			m_free.emplace(TotalSize(), DirectMemoryOverflowSize());
		}
	}
	virtual ~PhysicalMemory() = default;

	KYTY_CLASS_NO_COPY(PhysicalMemory);

	static constexpr uint64_t TotalSize() { return static_cast<uint64_t>(13824) * 1024 * 1024; }
	static uint64_t           Size() {
		EXIT_IF(g_flexible_memory_size >= TotalSize());
		return TotalSize() - g_flexible_memory_size;
	}

	bool Alloc(uint64_t search_start, uint64_t search_end, size_t len, size_t alignment,
	           uint64_t* phys_addr_out, int memory_type,
	           AllocationKind kind = AllocationKind::Direct);
	bool ReserveAutomatic(uint64_t size, PhysicalRanges* ranges);
	void RestoreAutomatic(const PhysicalRanges& ranges);
	bool Available(uint64_t search_start, uint64_t search_end, size_t alignment,
	               uint64_t* phys_addr_out, uint64_t* size_out);
	bool Release(uint64_t start, size_t len, uint64_t* vaddr, uint64_t* size,
	             GpuAccessMode* gpu_mode);
	bool Map(uint64_t vaddr, uint64_t phys_addr, size_t len, int prot, VirtualMemory::Mode mode,
	         GpuAccessMode gpu_mode);
	bool Unmap(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode,
	           uint64_t* host_vaddr_to_release = nullptr);
	bool Find(uint64_t vaddr, uint64_t* base_addr, size_t* len, int* prot,
	          VirtualMemory::Mode* mode, GpuAccessMode* gpu_mode);
	bool Find(uint64_t phys_addr, bool next, PhysicalMemory::AllocatedBlock* out);
	bool CanMapDirect(uint64_t phys_addr, size_t len);
	bool ReleasePoolExpansion(uint64_t phys_addr, size_t len);
	bool GetAllocatedSpan(uint64_t phys_addr, size_t len, std::vector<AllocatedBlock>* blocks);
	std::vector<AllocatedBlock> FindMappings(uint64_t phys_addr, size_t len);
	void ProtectMapping(uint64_t vaddr, uint64_t size, int prot, VirtualMemory::Mode mode,
	                    GpuAccessMode gpu_mode);
	void SetVirtualRangeName(uint64_t vaddr, uint64_t len, const char* name);
	void SetVirtualRangeMemoryType(uint64_t vaddr, uint64_t len, int memory_type);

	[[nodiscard]] Common::Mutex&                            GetMutex() { return m_mutex; }
	[[nodiscard]] const std::map<uint64_t, AllocatedBlock>& GetPhysicalBlocks() const {
		return m_physical;
	}
	[[nodiscard]] const std::vector<AllocatedBlock>& GetMappings() const { return m_mappings; }

private:
	static void RemoveFreeRange(std::map<uint64_t, uint64_t>& ranges, uint64_t start,
	                            uint64_t size);
	static void AddFreeRange(std::map<uint64_t, uint64_t>& ranges, uint64_t start, uint64_t size);
	void        ReclaimAutomatic(uint64_t start, uint64_t size);

	std::map<uint64_t, AllocatedBlock> m_physical;
	std::map<uint64_t, uint64_t>       m_free;
	std::map<uint64_t, uint64_t>       m_automatic_free;
	std::vector<AllocatedBlock>        m_mappings;
	Common::Mutex                      m_mutex;
	// Direct allocations given room past the direct memory (DirectMemoryOverflowSize).
	uint64_t m_overflow_allocations = 0;
};

class FlexibleMemory {
public:
	struct AllocatedBlock {
		uint64_t            map_vaddr;
		uint64_t            map_size;
		uint64_t            backing_offset;
		uint64_t            host_vaddr;
		uint64_t            host_size;
		int                 prot;
		VirtualMemory::Mode mode;
		GpuAccessMode       gpu_mode;
		char                name[KERNEL_MAXIMUM_NAME_LENGTH];
	};

	FlexibleMemory() {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
		m_free.emplace(PhysicalMemory::Size(), Size());
	}
	virtual ~FlexibleMemory() = default;

	KYTY_CLASS_NO_COPY(FlexibleMemory);

	static uint64_t Size() { return g_flexible_memory_size; }
	uint64_t        Available();

	bool Map(uint64_t vaddr, size_t len, int prot, VirtualMemory::Mode mode, GpuAccessMode gpu_mode,
	         const char* name);
	bool Unmap(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode,
	           uint64_t* host_vaddr_to_release = nullptr);
	bool Find(uint64_t vaddr, uint64_t* base_addr, size_t* len, int* prot,
	          VirtualMemory::Mode* mode, GpuAccessMode* gpu_mode);
	bool Snapshot(uint64_t vaddr, uint64_t size, std::vector<AllocatedBlock>* blocks);
	bool Restore(const std::vector<AllocatedBlock>& blocks);
	void Protect(uint64_t vaddr, uint64_t size, int prot, VirtualMemory::Mode mode,
	             GpuAccessMode gpu_mode);
	void SetVirtualRangeName(uint64_t vaddr, uint64_t len, const char* name);

	[[nodiscard]] Common::Mutex&                     GetMutex() { return m_mutex; }
	[[nodiscard]] const std::vector<AllocatedBlock>& GetBlocks() const { return m_allocated; }

private:
	void ConsumeFreeRange(std::map<uint64_t, uint64_t>::iterator range, uint64_t start,
	                      uint64_t size);
	void AddFreeRange(uint64_t start, uint64_t size);

	std::vector<AllocatedBlock>  m_allocated;
	std::map<uint64_t, uint64_t> m_free;
	uint64_t                     m_allocated_total = 0;
	Common::Mutex                m_mutex;
};

class PooledMemory {
public:
	struct Mapping {
		uint64_t      vaddr;
		uint64_t      size;
		uint64_t      phys_addr;
		GpuAccessMode gpu_mode;
	};

	void                   Expand(uint64_t phys_addr, uint64_t size);
	bool                   ReleaseExpansion(uint64_t phys_addr, uint64_t size);
	bool                   Allocate(uint64_t vaddr, uint64_t size, GpuAccessMode gpu_mode,
	                                std::vector<Mapping>* mappings);
	bool                   Query(uint64_t vaddr, uint64_t size, std::vector<Mapping>* mappings);
	bool                   Release(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode);
	[[nodiscard]] uint64_t Available();
	[[nodiscard]] std::vector<Mapping> GetMappings();

private:
	struct PhysicalRange {
		uint64_t start;
		uint64_t size;
	};

	void AddFreeUnlocked(uint64_t start, uint64_t size);
	bool QueryUnlocked(uint64_t vaddr, uint64_t size, std::vector<Mapping>* mappings) const;

	std::vector<PhysicalRange> m_free;
	std::vector<PhysicalRange> m_expansions;
	std::vector<Mapping>       m_mappings;
	Common::Mutex              m_mutex;
};

static std::unique_ptr<PhysicalMemory>    g_physical_memory;
static std::unique_ptr<FlexibleMemory>    g_flexible_memory;
static std::unique_ptr<PooledMemory>      g_pooled_memory;
static std::unique_ptr<VirtualRanges>     g_virtual_ranges;
static std::unique_ptr<GuestAddressSpace> g_guest_address_space;
static callback_func_t                    g_alloc_callback        = nullptr;
static callback_func_t                    g_free_callback         = nullptr;
static std::atomic<uint64_t>              g_memory_pool_committed = 0;
static void                               MemoryPoolSubtractCommitted(uint64_t len);
// Keep host mappings, physical blocks, placeholders, and virtual ranges in step.
static std::recursive_mutex g_memory_operation_mutex;

// The base address the PS5 kernel hands out for hint-less user mappings. Guest code can
// assume mappings it did not place explicitly are at or above this (Sony's libc rejects a
// heap below it), so hint-less searches must not fall back to the low system-managed range.
static constexpr uint64_t GUEST_DEFAULT_MAP_BASE = 0x200000000ull;

static uint64_t FindGuestFreeRange(uint64_t search_addr, uint64_t size, uint64_t alignment) {
	EXIT_IF(g_guest_address_space == nullptr || g_virtual_ranges == nullptr);

	auto find_in = [&](uint64_t begin, uint64_t end) {
		auto current = begin;
		while (current < end && size <= end - current) {
			const auto candidate =
			    g_guest_address_space->FindFreeAligned(current, end, size, alignment);
			if (candidate == 0) {
				return uint64_t {0};
			}
			VirtualRanges::Range overlap {};
			if (!g_virtual_ranges->QueryOverlap(candidate, size, &overlap)) {
				return candidate;
			}
			const auto overlap_end = overlap.start + overlap.size;
			if (overlap_end <= current) {
				return uint64_t {0};
			}
			current = overlap_end;
		}
		return uint64_t {0};
	};

	if (search_addr != 0) {
		return find_in(search_addr, HOST_USER_MAX + 1u);
	}
	auto addr = find_in(GUEST_DEFAULT_MAP_BASE, HOST_SYSTEM_MANAGED_MAX + 1u);
	if (addr == 0) {
		addr = find_in(HOST_USER_MIN, HOST_USER_MAX + 1u);
	}
	return addr;
}

namespace {

// The write journal (JournalWrite): two rings, the emulator's writes and the game threads' write
// faults (tens of thousands a second in races, which would push the others out). An entry is
// published by its sequence number, stored last; a reader skips one being rewritten.
struct JournalEntry {
	std::atomic<uint64_t> sequence {0}; // the entry's index in its ring + 1, once written
	std::atomic<uint64_t> vaddr {0};
	std::atomic<uint64_t> size_source {0}; // size << 8 | source
	std::atomic<uint64_t> time_ns {0};
	std::atomic<uint32_t> tid {0};
};

template <size_t N>
struct JournalRing {
	static_assert(std::has_single_bit(N));
	std::atomic<uint64_t>       next {0};
	std::array<JournalEntry, N> entries {};
};

// Made by Initialize, before any thread could append (a write fault's handler among them).
// Round 133: 131072 entries each (5 MB each; 16384 and 65536 before): at a race's rates the
// emulator's ring held its writes of the last few seconds only, so a crash report's "none" said
// little.
constexpr size_t             JournalEntries     = 131072;
JournalRing<JournalEntries>* g_emulator_journal = nullptr;
JournalRing<JournalEntries>* g_fault_journal    = nullptr;

void InitializeJournal() {
	if (g_emulator_journal == nullptr) {
		g_emulator_journal = new JournalRing<JournalEntries>();
		g_fault_journal    = new JournalRing<JournalEntries>();
	}
}

thread_local WriteSource t_write_source = WriteSource::Other;

template <size_t N>
void Append(JournalRing<N>& ring, uint64_t vaddr, uint64_t size, WriteSource source,
            uint64_t now) noexcept {
	const auto index = ring.next.fetch_add(1, std::memory_order_relaxed);
	auto&      entry = ring.entries[index & (N - 1)];
	entry.sequence.store(0, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_release);
	entry.vaddr.store(vaddr, std::memory_order_relaxed);
	entry.size_source.store((std::min<uint64_t>(size, uint64_t {1} << 55u) << 8u) |
	                            static_cast<uint8_t>(source),
	                        std::memory_order_relaxed);
	entry.time_ns.store(now, std::memory_order_relaxed);
	entry.tid.store(Common::PerfStats::CurrentOsThreadId(), std::memory_order_relaxed);
	entry.sequence.store(index + 1, std::memory_order_release);
}

struct JournalHit {
	uint64_t    vaddr  = 0;
	uint64_t    size   = 0;
	uint64_t    time   = 0;
	uint32_t    tid    = 0;
	WriteSource source = WriteSource::Other;
};

template <size_t N>
void Collect(const JournalRing<N>& ring, uint64_t vaddr, uint64_t size, uint64_t since,
             std::vector<JournalHit>& hits) {
	const auto next = ring.next.load(std::memory_order_acquire);
	for (uint64_t back = 0; back < std::min<uint64_t>(next, N); back++) {
		const auto  index    = next - 1 - back;
		const auto& entry    = ring.entries[index & (N - 1)];
		const auto  sequence = entry.sequence.load(std::memory_order_acquire);
		if (sequence != index + 1) {
			continue;
		}
		JournalHit hit;
		hit.vaddr              = entry.vaddr.load(std::memory_order_relaxed);
		const auto size_source = entry.size_source.load(std::memory_order_relaxed);
		hit.time               = entry.time_ns.load(std::memory_order_relaxed);
		hit.tid                = entry.tid.load(std::memory_order_relaxed);
		hit.size               = size_source >> 8u;
		hit.source             = static_cast<WriteSource>(size_source & 0xffu);
		std::atomic_thread_fence(std::memory_order_acquire);
		if (entry.sequence.load(std::memory_order_relaxed) != sequence) {
			continue;
		}
		if (hit.time < since) {
			// Older (stamped just before its slot was taken, a newer one may follow). A second
			// older than that, the rest are too (round 133: the rings are larger).
			if (hit.time + 1000000000u < since) {
				break;
			}
			continue;
		}
		if (hit.vaddr < vaddr + size && vaddr < hit.vaddr + hit.size) {
			hits.push_back(hit);
		}
	}
}

// Round 133: how far back the ring reaches, in ns before `now` (UINT64_MAX: to the start of the
// run, it has not wrapped).
template <size_t N>
uint64_t Reach(const JournalRing<N>& ring, uint64_t now) {
	const auto next = ring.next.load(std::memory_order_acquire);
	if (next <= N) {
		return UINT64_MAX;
	}
	const auto& oldest = ring.entries[next & (N - 1)];
	const auto  time   = oldest.time_ns.load(std::memory_order_relaxed);
	return now > time ? now - time : 0;
}

std::string ReachText(uint64_t ns) {
	return ns == UINT64_MAX ? std::string("since the start")
	                        : fmt::format("for the last {:.1f} s", static_cast<double>(ns) / 1e9);
}

} // namespace

const char* WriteSourceName(WriteSource source) noexcept {
	switch (source) {
		case WriteSource::Other: return "backing write";
		case WriteSource::Download: return "download";
		case WriteSource::FastReadback: return "fast readback";
		case WriteSource::StoreInPlace: return "store in place";
		case WriteSource::ImageWriteBack: return "image write-back";
		case WriteSource::MetadataClear: return "metadata clear";
		case WriteSource::GpuThreadStore: return "WRITE_DATA";
		case WriteSource::Fill: return "DMA fill";
		case WriteSource::Copy: return "DMA copy";
		case WriteSource::FileRead: return "file read";
		case WriteSource::CpuWriteFault: return "game write fault";
		case WriteSource::Count: break;
	}
	return "?";
}

uint64_t JournalNowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

void JournalWrite(uint64_t vaddr, uint64_t size, WriteSource source) noexcept {
	if (size == 0) {
		return;
	}
	const auto now = JournalNowNs();
	if (source == WriteSource::CpuWriteFault) {
		if (auto* ring = g_fault_journal; ring != nullptr) {
			Append(*ring, vaddr, size, source, now);
		}
	} else if (auto* ring = g_emulator_journal; ring != nullptr) {
		Append(*ring, vaddr, size, source, now);
	}
}

WriteSource CurrentWriteSource() noexcept {
	return t_write_source;
}

WriteSourceScope::WriteSourceScope(WriteSource source) noexcept: m_previous(t_write_source) {
	t_write_source = source;
}

WriteSourceScope::~WriteSourceScope() {
	t_write_source = m_previous;
}

std::string DescribeJournal(uint64_t vaddr, uint64_t size, uint64_t ms, size_t max) {
	const auto              now   = JournalNowNs();
	const auto              since = now > ms * 1000000u ? now - ms * 1000000u : 0;
	std::vector<JournalHit> hits;
	if (const auto* ring = g_emulator_journal; ring != nullptr) {
		Collect(*ring, vaddr, size, since, hits);
	}
	if (const auto* ring = g_fault_journal; ring != nullptr) {
		Collect(*ring, vaddr, size, since, hits);
	}
	if (hits.empty()) {
		// Round 133: and how far back that holds.
		if (g_emulator_journal == nullptr || g_fault_journal == nullptr) {
			return "none";
		}
		return fmt::format("none (kept: the emulator's writes {}, the game's write faults {})",
		                   ReachText(Reach(*g_emulator_journal, now)),
		                   ReachText(Reach(*g_fault_journal, now)));
	}
	std::sort(hits.begin(), hits.end(),
	          [](const JournalHit& a, const JournalHit& b) { return a.time > b.time; });
	std::string text;
	for (size_t i = 0; i < hits.size() && i < max; i++) {
		const auto& hit = hits[i];
		const auto  age = static_cast<double>(now - std::min(now, hit.time)) / 1e6;
		text += fmt::format("{}{} 0x{:012x}+0x{:x} {:.1f} ms ago ({})", i == 0 ? "" : "; ",
		                    WriteSourceName(hit.source), hit.vaddr, hit.size, age,
		                    Common::PerfStats::ThreadNameOf(hit.tid));
	}
	if (hits.size() > max) {
		text += fmt::format("; {} more", hits.size() - max);
	}
	return text;
}

bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
	const bool written = g_guest_address_space != nullptr &&
	                     g_guest_address_space->TryWriteBacking(vaddr, data, size);
	if (written) {
		JournalWrite(vaddr, size, t_write_source);
	}
	return written;
}

bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
	return g_guest_address_space != nullptr &&
	       g_guest_address_space->TryReadBacking(vaddr, data, size);
}

bool TryReadBufferBacking(uint64_t vaddr, void* data, uint64_t size) {
	if (g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size)) {
		if (!Graphics::GuestGpu::IsGpuThread()) {
			return false;
		}
		auto& buffers = GetGpuResources().GetBufferCache();
		// Round 152: bytes a fast readback's copy is still bringing in are put there first (a
		// short wait, not the fault's path, which may drain the GPU for the page's other bytes).
		buffers.FinishFastReadbacks(vaddr, size);
		if (buffers.HasGpuDirtyBytes(vaddr, size)) {
			buffers.ReadMemory(vaddr, size);
		}
	}
	return TryReadBacking(vaddr, data, size);
}

static thread_local bool t_gpu_read_delegate = false;

bool IsGpuBufferWritten(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) {
		return false;
	}
	if (t_gpu_read_delegate) {
		return GetGpuResources().GetBufferCache().HasGpuDirtyBytesShared(vaddr, size);
	}
	if (!Graphics::GuestGpu::IsGpuThread()) {
		return false;
	}
	return GetGpuResources().GetBufferCache().HasGpuDirtyBytes(vaddr, size);
}

void SetGpuReadDelegate(bool delegate) {
	t_gpu_read_delegate = delegate;
}

// Whether [vaddr, vaddr + size) may hold GPU-written bytes, for the GPU thread or its delegate
// (true for any other thread: it cannot ask).
static bool MaybeGpuDirtyForClean(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) {
		return false;
	}
	if (t_gpu_read_delegate) {
		return GetGpuResources().GetBufferCache().HasGpuDirtyBytesShared(vaddr, size);
	}
	if (!Graphics::GuestGpu::IsGpuThread()) {
		return true;
	}
	// Round 152: as TryReadBufferBacking, bytes a fast readback's copy is still bringing in are
	// put there first.
	auto& buffers = GetGpuResources().GetBufferCache();
	buffers.FinishFastReadbacks(vaddr, size);
	return buffers.HasGpuDirtyBytes(vaddr, size);
}

const uint8_t* FindGpuCleanBacking(uint64_t vaddr, uint64_t size) {
	if (g_guest_address_space == nullptr || MaybeGpuDirtyForClean(vaddr, size)) {
		return nullptr;
	}
	return g_guest_address_space->FindBacking(vaddr, size);
}

const uint8_t* FindBackingUnchecked(uint64_t vaddr, uint64_t size) {
	return g_guest_address_space != nullptr ? g_guest_address_space->FindBacking(vaddr, size)
	                                        : nullptr;
}

bool TryReadGpuCleanBacking(uint64_t vaddr, void* data, uint64_t size) {
	return !MaybeGpuDirtyForClean(vaddr, size) && TryReadBacking(vaddr, data, size);
}

bool TryReadCleanFaultingBytes(uint64_t fault_vaddr, uint64_t vaddr, void* data, uint64_t size) {
	return g_gpu_resources != nullptr &&
	       g_gpu_resources->CanServeCleanRead(fault_vaddr, vaddr, size) &&
	       TryReadBacking(vaddr, data, size);
}

uint64_t TryClampRangeSize(uint64_t vaddr, uint64_t size) {
	EXIT_IF(g_virtual_ranges == nullptr);

	// Every buffer of every draw asks; a heap-wide descriptor spans thousands of mappings, each
	// walked under the lock (3.8 % of the GPU thread in the jungle, 2026-10-03). The answer stays
	// valid until the ranges change, so each thread remembers its recent ones.
	struct Answer {
		uint64_t generation = 0;
		uint64_t vaddr      = 0;
		uint64_t size       = 0;
		uint64_t clamped    = 0;
	};
	// KYTY_CLAMP_CACHE=0 walks the ranges every time, as before.
	static const bool cached = [] {
		const char* value = std::getenv("KYTY_CLAMP_CACHE");
		return value == nullptr || value[0] != '0';
	}();
	thread_local std::array<Answer, 64> answers {};
	const auto generation = g_virtual_ranges->CoverageGeneration();
	auto&      answer     = answers[((vaddr >> 12u) ^ (size >> 12u) ^ size) & 63u];
	if (cached && answer.generation == generation && answer.vaddr == vaddr && answer.size == size) {
		return answer.clamped;
	}
	// Round 133: the merged committed spans the last answers came from. A range inside one is
	// whole without the lock and the search: a draw's buffers lie in a few large spans, and the
	// answers above, by address and size, kept missing (1.4 % of the GPU thread in a race, its
	// lock waits besides). A range that runs past its span takes the walk, which logs the cut.
	struct Span {
		uint64_t generation = 0;
		uint64_t begin      = 0;
		uint64_t end        = 0;
	};
	thread_local std::array<Span, 8> spans {};
	thread_local uint32_t            next_span = 0;
	if (cached && vaddr != 0 && size != 0 && size <= UINT64_MAX - vaddr) {
		for (const auto& span: spans) {
			if (span.generation == generation && vaddr >= span.begin && vaddr + size <= span.end) {
				answer = {generation, vaddr, size, size};
				return size;
			}
		}
	}
	uint64_t   span_begin   = 0;
	uint64_t   span_end     = 0;
	const auto clamped_size = g_virtual_ranges->ClampRangeSize(vaddr, size, &span_begin, &span_end);
	answer                  = {generation, vaddr, size, clamped_size};
	// A range past its span's end comes here again and again: its span is kept once.
	if (cached && span_end > span_begin && std::ranges::none_of(spans, [&](const Span& span) {
		    return span.generation == generation && span.begin == span_begin &&
		           span.end == span_end;
	    })) {
		spans[next_span++ & 7u] = {generation, span_begin, span_end};
	}
	if (clamped_size != 0 && clamped_size != size) {
		LOGF("Memory: clamped buffer range addr=0x%016" PRIx64 " size=0x%016" PRIx64
		     " to 0x%016" PRIx64 "\n",
		     vaddr, size, clamped_size);
		if (TracesMemory(vaddr, size)) {
			TraceMemory("range 0x%016" PRIx64 "+0x%" PRIx64 " cut to 0x%" PRIx64
			            " at a hole in the mappings: %s",
			            vaddr, size, clamped_size, DescribeMappings(vaddr, size).c_str());
		}
	}
	return clamped_size;
}

std::string DescribeMappings(uint64_t vaddr, uint64_t size) {
	return g_virtual_ranges != nullptr ? g_virtual_ranges->Describe(vaddr, size)
	                                   : "(no memory map)";
}

std::vector<std::string> DescribeGpuImages(uint64_t vaddr, uint64_t size, size_t count) {
	if (g_gpu_resources == nullptr || size == 0) {
		return {};
	}
	return g_gpu_resources->DescribeImages(vaddr, size, count);
}

bool IsCommittedRange(uint64_t vaddr, uint64_t size) {
	// No guest memory map (unit tests that drive the libraries on host memory): nothing to check.
	if (g_virtual_ranges == nullptr) {
		return true;
	}

	return size == 0 || g_virtual_ranges->ClampRangeSize(vaddr, size) == size;
}

bool CommittedSpans(uint64_t vaddr, uint64_t size, size_t max_spans,
                    std::vector<std::pair<uint64_t, uint64_t>>* spans) {
	EXIT_IF(spans == nullptr);
	if (g_virtual_ranges == nullptr) {
		spans->assign(1, {vaddr, size});
		return true;
	}
	return g_virtual_ranges->CommittedSpans(vaddr, size, max_spans, spans);
}

uint64_t CommittedExtent(uint64_t vaddr, uint64_t size) {
	return g_virtual_ranges != nullptr ? g_virtual_ranges->CommittedExtent(vaddr, size) : size;
}

uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size) {
	const auto clamped_size = TryClampRangeSize(vaddr, size);
	if (clamped_size == 0) {
		EXIT("Memory: attempted to access invalid address 0x%016" PRIx64 " with size 0x%016" PRIx64
		     "\n",
		     vaddr, size);
	}
	return clamped_size;
}

void WriteBacking(uint64_t vaddr, const void* data, uint64_t size) noexcept {
	if (!TryWriteBacking(vaddr, data, size)) {
		EXIT("Memory: required direct-backing write failed, addr=0x%016" PRIx64
		     " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
}

void InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (size == 0) {
		return;
	}
	(void)GetGpuResources().InvalidateMemory(vaddr, size);
}

void InstallGpuResources(Graphics::RenderContext* resources) noexcept {
	EXIT_IF(resources != nullptr && g_gpu_resources != nullptr);
	g_gpu_resources = resources;
}

bool HandleGpuFault(Graphics::PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	return g_gpu_resources != nullptr && g_gpu_resources->HandleFault(access, fault_vaddr);
}

bool IsGpuWrittenPage(uint64_t fault_vaddr) noexcept {
	return g_gpu_resources != nullptr && g_gpu_resources->IsGpuWrittenPage(fault_vaddr);
}

std::string DescribeGpuBytes(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr ? g_gpu_resources->DescribeGpuBytes(vaddr, size)
	                                  : std::string("no GPU");
}

bool TryWriteGpuFaultingBytes(uint64_t pc, uint64_t fault_vaddr, uint64_t vaddr, const void* data,
                              uint64_t size) {
	return g_gpu_resources != nullptr &&
	       g_gpu_resources->TryWriteFaulting(pc, fault_vaddr, vaddr, data, size);
}

struct PrtAperture {
	uint64_t address = 0;
	uint64_t size    = 0;
};

constexpr int      PRT_APERTURE_MAX_INDEX = 2;
constexpr uint64_t PRT_PAGE_SIZE          = 0x4000;
constexpr uint64_t PRT_APERTURE_START     = 0x0f00000000ull;
constexpr uint64_t PRT_APERTURE_END       = 0xfc00000000ull;

static std::array<PrtAperture, PRT_APERTURE_MAX_INDEX + 1> g_prt_apertures {};
static Common::Mutex                                       g_prt_aperture_mutex;

static bool IsInPrtAperture(uint64_t address, uint64_t size = 1) {
	if (size == 0 || UINT64_MAX - address < size) {
		return false;
	}
	Common::LockGuard lock(g_prt_aperture_mutex);

	for (const auto& aperture: g_prt_apertures) {
		if (address >= aperture.address) {
			const auto offset = address - aperture.address;
			if (offset < aperture.size && size <= aperture.size - offset) {
				return true;
			}
		}
	}

	return false;
}

bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size) {
	return g_guest_address_space->TryReadSparseBacking(vaddr, data, size);
}

static bool SelfTestSub64SharedPlaceholderAlias() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	constexpr uint64_t PageSize    = 0x4000;
	const auto         granularity = g_guest_address_space->GetGranularity();
	if (granularity < PageSize * 2u) {
		LOGF_COLOR(
		    Log::Color::Yellow,
		    "\t direct-memory sub-64K placeholder self-test skipped: granularity too small\n");
		return true;
	}

	const auto base = FindGuestFreeRange(0, granularity, granularity);
	if (base == 0) {
		LOGF_COLOR(Log::Color::Red,
		           "\t direct-memory sub-64K placeholder self-test: reserve unavailable\n");
		return false;
	}

	const auto alias          = base + PageSize;
	bool       ok             = false;
	auto       failure_reason = GuestBackingStore::FailureReason::None;
	if (g_guest_address_space->MapBacking(alias, PageSize, PageSize, VirtualMemory::Mode::ReadWrite,
	                                      &failure_reason)) {
		auto* ptr = reinterpret_cast<uint64_t*>(alias);
		*ptr      = 0x4b59545953553634ull; // "KYTYSU64"
		ok        = (*ptr == 0x4b59545953553634ull);
		std::memset(ptr, 0, PageSize);

		ok = g_guest_address_space->UnmapBacking(alias, PageSize) && ok;
	}

	LOGF_COLOR(
	    ok ? Log::Color::Green : Log::Color::Red,
	    "\t direct-memory sub-64K placeholder self-test: %s%s%s\n", ok ? "ok" : "failed",
	    ok ? "" : ", reason = ", ok ? "" : GuestBackingStore::GetFailureReasonName(failure_reason));
	return ok;
#else
	return true;
#endif
}

static bool ReplaceFixedRangeWithReserved(uint64_t start, uint64_t size);

void Initialize() {
	InitializeJournal();
	g_flexible_memory_size_frozen = true;
	VirtualMemory::Init();
	g_guest_address_space = std::make_unique<GuestAddressSpace>(PhysicalMemory::TotalSize() +
	                                                            DirectMemoryOverflowSize());
	g_physical_memory     = std::make_unique<PhysicalMemory>();
	g_flexible_memory     = std::make_unique<FlexibleMemory>();
	g_pooled_memory       = std::make_unique<PooledMemory>();
	g_virtual_ranges      = std::make_unique<VirtualRanges>();
	EXIT_IF(!g_guest_address_space->SelfTest());
	EXIT_IF(!SelfTestSub64SharedPlaceholderAlias());
}

void Shutdown() {
	g_pooled_memory.reset();
	g_flexible_memory.reset();
	g_physical_memory.reset();
	g_virtual_ranges.reset();
	g_guest_address_space.reset();
}

struct AlignedPos {
	uint64_t value = 0;
	bool     valid = false;
};

static constexpr AlignedPos GetAlignedPos(uint64_t pos, size_t alignment) {
	if (alignment == 0) {
		return {pos, true};
	}

	const auto remainder = pos % alignment;
	const auto increment = (remainder != 0 ? alignment - remainder : 0);
	if (increment > UINT64_MAX - pos) {
		return {};
	}

	return {pos + increment, true};
}

static_assert(!GetAlignedPos(UINT64_MAX - 1, 4).valid);

void RegisterCallbacks(callback_func_t alloc_func, callback_func_t free_func) {
	EXIT_IF(g_alloc_callback != nullptr || g_free_callback != nullptr);
	EXIT_IF(alloc_func == nullptr || free_func == nullptr);

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	g_alloc_callback = alloc_func;
	g_free_callback  = free_func;

	g_physical_memory->GetMutex().Lock();
	for (const auto& b: g_physical_memory->GetMappings()) {
		if (b.map_vaddr != 0 && b.map_size != 0) {
			g_alloc_callback(b.map_vaddr, b.map_size);
		}
	}
	g_physical_memory->GetMutex().Unlock();

	g_flexible_memory->GetMutex().Lock();
	for (const auto& b: g_flexible_memory->GetBlocks()) {
		g_alloc_callback(b.map_vaddr, b.map_size);
	}
	g_flexible_memory->GetMutex().Unlock();

	for (const auto& mapping: g_pooled_memory->GetMappings()) {
		g_alloc_callback(mapping.vaddr, mapping.size);
	}
}

void SetFlexibleMemorySize(uint64_t size) {
	constexpr uint64_t GuestPageSize = 0x4000;
	EXIT_IF(g_flexible_memory_size_frozen || g_guest_address_space != nullptr);
	EXIT_IF(size == 0 || (size & (GuestPageSize - 1u)) != 0 || size >= PhysicalMemory::TotalSize());
	g_flexible_memory_size = size;
	LOGF("\t flexible memory size = 0x%016" PRIx64 " (%" PRIu64 " MiB)\n", size,
	     size / (1024ull * 1024ull));
}

bool PhysicalMemory::Alloc(uint64_t search_start, uint64_t search_end, size_t len, size_t alignment,
                           uint64_t* phys_addr_out, int memory_type, AllocationKind kind) {
	if (phys_addr_out == nullptr) {
		return false;
	}

	Common::LockGuard lock(m_mutex);

	const auto requested_end = search_end;
	search_end               = std::min<uint64_t>(search_end, Size());
	if (search_start >= search_end) {
		return false;
	}

	// The first free place for `len` in [first, last), or UINT64_MAX.
	const auto find = [&](uint64_t first, uint64_t last) -> uint64_t {
		auto range = m_free.upper_bound(first);
		if (range != m_free.begin()) {
			range--;
		}
		for (; range != m_free.end() && range->first < last; ++range) {
			const auto range_end   = std::min<uint64_t>(range->first + range->second, last);
			const auto lower_bound = std::max(range->first, first);
			const auto aligned     = GetAlignedPos(lower_bound, alignment);
			const auto free_pos    = aligned.value;
			if (!aligned.valid || free_pos < lower_bound || free_pos > range_end ||
			    len > range_end - free_pos) {
				continue;
			}
			return free_pos;
		}
		return UINT64_MAX;
	};
	auto free_pos = find(search_start, search_end);
	if (free_pos == UINT64_MAX && kind == AllocationKind::Direct &&
	    DirectMemoryOverflowSize() != 0 && requested_end >= Size()) {
		// A search up to the end of the direct memory that found no room there: the overflow.
		free_pos = find(TotalSize(), TotalSize() + DirectMemoryOverflowSize());
		if (free_pos != UINT64_MAX) {
			const auto n = ++m_overflow_allocations;
			if (n <= 16 || std::has_single_bit(n)) {
				uint64_t in_overflow = 0;
				for (const auto& [start, block]: m_physical) {
					if (start >= TotalSize()) {
						in_overflow += block.size;
					}
				}
				LOGF("Direct memory: 0x%" PRIx64 " bytes did not fit in the game's %" PRIu64
				     " MiB: given room past it at 0x%" PRIx64 " (round 81; %" PRIu64
				     " MiB of the %" PRIu64 " MiB there in use before it; %" PRIu64
				     " such allocations so far, KYTY_DMEM_OVERFLOW_MB=0 fails them as before)\n",
				     static_cast<uint64_t>(len), Size() >> 20u, free_pos, in_overflow >> 20u,
				     DirectMemoryOverflowSize() >> 20u, n);
			}
		}
	}
	if (free_pos != UINT64_MAX) {
		AllocatedBlock b {};
		b.size           = len;
		b.start_addr     = free_pos;
		b.gpu_mode       = GpuAccessMode::NoAccess;
		b.map_size       = 0;
		b.map_vaddr      = 0;
		b.prot           = 0;
		b.mode           = VirtualMemory::Mode::NoAccess;
		b.memory_type    = memory_type;
		b.kind           = kind;

		RemoveFreeRange(m_free, free_pos, len);
		EXIT_IF(!m_physical.emplace(b.start_addr, b).second);
		if (kind == AllocationKind::Automatic) {
			AddFreeRange(m_automatic_free, free_pos, len);
		}

		*phys_addr_out = free_pos;
		return true;
	}

	return false;
}

bool PhysicalMemory::Available(uint64_t search_start, uint64_t search_end, size_t alignment,
                               uint64_t* phys_addr_out, uint64_t* size_out) {
	if (phys_addr_out == nullptr || size_out == nullptr) {
		return false;
	}

	Common::LockGuard lock(m_mutex);

	search_end = std::min<uint64_t>(search_end, Size());
	if (search_start >= search_end) {
		return false;
	}

	uint64_t best_addr = 0;
	uint64_t best_size = 0;

	for (const auto& [range_start, range_size]: m_free) {
		if (range_start >= search_end) {
			break;
		}
		const auto range_end   = std::min<uint64_t>(range_start + range_size, search_end);
		const auto lower_bound = std::max(range_start, search_start);
		const auto aligned     = GetAlignedPos(lower_bound, alignment);
		const auto free_pos    = aligned.value;
		if (aligned.valid && free_pos >= lower_bound && free_pos < range_end &&
		    range_end - free_pos > best_size) {
			best_addr = free_pos;
			best_size = range_end - free_pos;
		}
	}

	if (best_size == 0) {
		return false;
	}

	*phys_addr_out = best_addr;
	*size_out      = best_size;
	return true;
}

void PhysicalMemory::RemoveFreeRange(std::map<uint64_t, uint64_t>& ranges, uint64_t start,
                                     uint64_t size) {
	const auto end   = start + size;
	auto       range = ranges.upper_bound(start);
	if (range != ranges.begin()) {
		--range;
	}
	while (range != ranges.end() && range->first < end) {
		const auto range_start = range->first;
		const auto range_end   = range_start + range->second;
		if (range_end <= start) {
			++range;
			continue;
		}
		range = ranges.erase(range);
		if (range_start < start) {
			ranges.emplace(range_start, start - range_start);
		}
		if (end < range_end) {
			ranges.emplace(end, range_end - end);
		}
	}
}

void PhysicalMemory::AddFreeRange(std::map<uint64_t, uint64_t>& ranges, uint64_t start,
                                  uint64_t size) {
	auto end  = start + size;
	auto next = ranges.lower_bound(start);
	if (next != ranges.begin()) {
		auto previous = std::prev(next);
		if (previous->first + previous->second >= start) {
			start = previous->first;
			end   = std::max(end, previous->first + previous->second);
			next  = ranges.erase(previous);
		}
	}
	while (next != ranges.end() && next->first <= end) {
		end  = std::max(end, next->first + next->second);
		next = ranges.erase(next);
	}
	ranges.emplace(start, end - start);
}

bool PhysicalMemory::ReserveAutomatic(uint64_t size, PhysicalRanges* ranges) {
	Common::LockGuard lock(m_mutex);
	for (const auto& [start, available]: m_automatic_free) {
		const auto length = std::min(size, available);
		ranges->emplace_back(start, length);
		size -= length;
		if (size == 0) {
			for (const auto& [offset, length]: *ranges) {
				RemoveFreeRange(m_automatic_free, offset, length);
			}
			return true;
		}
	}
	ranges->clear();
	return false;
}

void PhysicalMemory::RestoreAutomatic(const PhysicalRanges& ranges) {
	Common::LockGuard lock(m_mutex);
	for (const auto& [start, size]: ranges) {
		AddFreeRange(m_automatic_free, start, size);
	}
}

void PhysicalMemory::ReclaimAutomatic(uint64_t start, uint64_t size) {
	const auto end   = start + size;
	auto       owner = m_physical.upper_bound(start);
	if (owner != m_physical.begin()) {
		--owner;
	}
	bool automatic = false;
	for (; owner != m_physical.end() && owner->first < end; ++owner) {
		const auto& block = owner->second;
		if (block.kind != AllocationKind::Automatic || block.start_addr + block.size <= start) {
			continue;
		}
		const auto first = std::max(start, block.start_addr);
		const auto last  = std::min(end, block.start_addr + block.size);
		AddFreeRange(m_automatic_free, first, last - first);
		automatic = true;
	}
	if (automatic) {
		for (const auto& mapping: m_mappings) {
			const auto first = std::max(start, mapping.start_addr);
			const auto last  = std::min(end, mapping.start_addr + mapping.size);
			if (first < last) {
				RemoveFreeRange(m_automatic_free, first, last - first);
			}
		}
	}
}

bool PhysicalMemory::Release(uint64_t start, size_t len, uint64_t* vaddr, uint64_t* size,
                             GpuAccessMode* gpu_mode) {
	EXIT_IF(vaddr == nullptr);
	EXIT_IF(size == nullptr);
	EXIT_IF(gpu_mode == nullptr);

	Common::LockGuard lock(m_mutex);

	auto next = m_physical.upper_bound(start);
	if (next == m_physical.begin()) {
		return false;
	}
	auto  it = std::prev(next);
	auto& b  = it->second;
	if (b.kind == AllocationKind::Pooled || start < b.start_addr ||
	    start >= b.start_addr + b.size || len > b.start_addr + b.size - start) {
		return false;
	}

	if (b.kind == AllocationKind::Automatic) {
		RemoveFreeRange(m_automatic_free, start, len);
	}

	if (start == b.start_addr && len == b.size) {
		*vaddr    = b.map_vaddr;
		*size     = b.map_size;
		*gpu_mode = b.gpu_mode;

		m_physical.erase(it);
		AddFreeRange(m_free, start, len);
		return true;
	}
	if (start > b.start_addr && start + len < b.start_addr + b.size) {
		auto old_start = b.start_addr;
		auto old_end   = b.start_addr + b.size;

		*vaddr    = (b.map_vaddr != 0 ? b.map_vaddr + (start - old_start) : 0);
		*size     = (b.map_vaddr != 0 ? len : 0);
		*gpu_mode = b.gpu_mode;

		AllocatedBlock right = b;
		right.start_addr     = start + len;
		right.size           = old_end - right.start_addr;
		if (right.map_vaddr != 0) {
			right.map_vaddr += right.start_addr - old_start;
			right.map_size = right.size;
		}

		b.size = start - old_start;
		if (b.map_vaddr != 0) {
			b.map_size = b.size;
		}

		m_physical.emplace(right.start_addr, right);
		AddFreeRange(m_free, start, len);
		return true;
	}
	if (start == b.start_addr && len < b.size) {
		*vaddr    = b.map_vaddr;
		*size     = (b.map_vaddr != 0 ? len : 0);
		*gpu_mode = b.gpu_mode;

		AllocatedBlock remaining = b;
		m_physical.erase(it);
		remaining.start_addr += len;
		remaining.size -= len;
		if (remaining.map_vaddr != 0) {
			remaining.map_vaddr += len;
			remaining.map_size -= len;
		}
		m_physical.emplace(remaining.start_addr, remaining);
		AddFreeRange(m_free, start, len);
		return true;
	}
	if (start > b.start_addr && start + len == b.start_addr + b.size) {
		*vaddr    = (b.map_vaddr != 0 ? b.map_vaddr + (start - b.start_addr) : 0);
		*size     = (b.map_vaddr != 0 ? len : 0);
		*gpu_mode = b.gpu_mode;

		b.size = start - b.start_addr;
		if (b.map_vaddr != 0) {
			b.map_size = b.size;
		}
		AddFreeRange(m_free, start, len);
		return true;
	}

	return false;
}

bool PhysicalMemory::Map(uint64_t vaddr, uint64_t phys_addr, size_t len, int prot,
                         VirtualMemory::Mode mode, GpuAccessMode gpu_mode) {
	Common::LockGuard lock(m_mutex);

	if (len == 0 || UINT64_MAX - phys_addr < len) {
		return false;
	}

	auto current = phys_addr;
	auto next    = m_physical.upper_bound(current);
	if (next == m_physical.begin()) {
		return false;
	}
	auto first = std::prev(next);
	bool has_automatic = false;
	while (current < phys_addr + len) {
		auto block = m_physical.upper_bound(current);
		if (block == m_physical.begin()) {
			return false;
		}
		--block;
		const auto block_end = block->second.start_addr + block->second.size;
		if (block->second.kind == AllocationKind::Pooled || current < block->second.start_addr ||
		    current >= block_end) {
			return false;
		}
		has_automatic |= block->second.kind == AllocationKind::Automatic;
		current = std::min<uint64_t>(phys_addr + len, block_end);
		if (current < phys_addr + len) {
			const auto following = std::next(block);
			if (following == m_physical.end() || following->second.start_addr != current) {
				return false;
			}
		}
	}

	AllocatedBlock mapping = first->second;
	mapping.start_addr     = phys_addr;
	mapping.size           = len;
	mapping.map_vaddr      = vaddr;
	mapping.map_size       = len;
	mapping.host_vaddr     = vaddr;
	mapping.host_size      = len;
	mapping.prot           = prot;
	mapping.mode           = mode;
	mapping.gpu_mode       = gpu_mode;
	m_mappings.push_back(mapping);
	if (has_automatic) {
		RemoveFreeRange(m_automatic_free, phys_addr, len);
	}

	return true;
}

bool PhysicalMemory::CanMapDirect(uint64_t phys_addr, size_t len) {
	Common::LockGuard lock(m_mutex);

	if (len == 0 || UINT64_MAX - phys_addr < len) {
		return false;
	}

	const auto end     = phys_addr + len;
	auto       current = phys_addr;
	while (current < end) {
		auto block = m_physical.upper_bound(current);
		if (block == m_physical.begin()) {
			return false;
		}
		--block;
		const auto block_end = block->second.start_addr + block->second.size;
		if (block->second.kind == AllocationKind::Pooled || current < block->second.start_addr ||
		    current >= block_end) {
			return false;
		}
		current = std::min(end, block_end);
		if (current < end) {
			const auto following = std::next(block);
			if (following == m_physical.end() || following->second.start_addr != current) {
				return false;
			}
		}
	}
	return true;
}

bool PhysicalMemory::ReleasePoolExpansion(uint64_t phys_addr, size_t len) {
	Common::LockGuard lock(m_mutex);
	const auto        it = m_physical.find(phys_addr);
	if (it == m_physical.end() || it->second.kind != AllocationKind::Pooled ||
	    it->second.size != len) {
		return false;
	}
	m_physical.erase(it);
	AddFreeRange(m_free, phys_addr, len);
	return true;
}

bool PhysicalMemory::GetAllocatedSpan(uint64_t phys_addr, size_t len,
                                      std::vector<AllocatedBlock>* blocks) {
	EXIT_IF(blocks == nullptr);
	blocks->clear();
	if (len == 0 || UINT64_MAX - phys_addr < len) {
		return false;
	}

	Common::LockGuard lock(m_mutex);
	const auto        end     = phys_addr + len;
	auto              current = phys_addr;
	while (current < end) {
		auto block = m_physical.upper_bound(current);
		if (block == m_physical.begin()) {
			blocks->clear();
			return false;
		}
		--block;
		const auto block_end = block->second.start_addr + block->second.size;
		if (block->second.kind == AllocationKind::Pooled || current < block->second.start_addr ||
		    current >= block_end) {
			blocks->clear();
			return false;
		}

		auto part       = block->second;
		part.start_addr = current;
		part.size       = std::min(end, block_end) - current;
		blocks->push_back(part);
		current += part.size;
		if (current < end) {
			const auto following = std::next(block);
			if (following == m_physical.end() || following->second.start_addr != current) {
				blocks->clear();
				return false;
			}
		}
	}
	return true;
}

bool PhysicalMemory::Unmap(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode,
                           uint64_t* host_vaddr_to_release) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	if (g_test_physical_memory_unmaps_before_failure == 0) {
		g_test_physical_memory_unmaps_before_failure = UINT32_MAX;
		return false;
	}
	if (g_test_physical_memory_unmaps_before_failure != UINT32_MAX) {
		g_test_physical_memory_unmaps_before_failure--;
	}
#endif

	EXIT_IF(gpu_mode == nullptr);

	Common::LockGuard lock(m_mutex);

	if (host_vaddr_to_release != nullptr) {
		*host_vaddr_to_release = 0;
	}

	auto set_host_release_if_last = [this, host_vaddr_to_release](uint64_t host_vaddr,
	                                                              uint64_t host_size) {
		if (host_vaddr_to_release == nullptr || host_vaddr == 0 || host_size == 0) {
			return;
		}
		const bool still_mapped = std::any_of(
		    m_mappings.begin(), m_mappings.end(), [host_vaddr, host_size](const auto& block) {
			    return block.host_vaddr == host_vaddr && block.host_size == host_size;
		    });
		if (!still_mapped) {
			*host_vaddr_to_release = host_vaddr;
		}
	};

	for (size_t index = 0; index < m_mappings.size(); ++index) {
		auto& b = m_mappings[index];
		if (vaddr < b.map_vaddr || vaddr >= b.map_vaddr + b.map_size ||
		    size > b.map_vaddr + b.map_size - vaddr) {
			continue;
		}
		*gpu_mode             = b.gpu_mode;
		const auto phys_addr  = b.start_addr + vaddr - b.map_vaddr;
		const auto host_vaddr = b.host_vaddr;
		const auto host_size  = b.host_size;
		if (vaddr == b.map_vaddr && size == b.map_size) {
			m_mappings.erase(m_mappings.begin() + static_cast<std::ptrdiff_t>(index));
			set_host_release_if_last(host_vaddr, host_size);
		} else if (vaddr > b.map_vaddr && vaddr + size < b.map_vaddr + b.map_size) {
			AllocatedBlock right = b;
			right.start_addr += (vaddr + size) - b.map_vaddr;
			right.size      = b.map_vaddr + b.map_size - (vaddr + size);
			right.map_size  = right.size;
			right.map_vaddr = vaddr + size;
			b.size     = vaddr - b.map_vaddr;
			b.map_size = b.size;
			m_mappings.push_back(right);
		} else if (vaddr == b.map_vaddr) {
			b.start_addr += size;
			b.size -= size;
			b.map_vaddr += size;
			b.map_size -= size;
		} else {
			b.size     = vaddr - b.map_vaddr;
			b.map_size = b.size;
		}
		ReclaimAutomatic(phys_addr, size);
		return true;
	}

	return false;
}

void PhysicalMemory::ProtectMapping(uint64_t vaddr, uint64_t size, int prot,
                                    VirtualMemory::Mode mode, GpuAccessMode gpu_mode) {
	Common::LockGuard lock(m_mutex);
	if (size == 0 || UINT64_MAX - vaddr < size) {
		return;
	}

	const auto                  end = vaddr + size;
	std::vector<AllocatedBlock> updated;
	updated.reserve(m_mappings.size() + 2);
	for (const auto& block: m_mappings) {
		const auto block_end = block.map_vaddr + block.map_size;
		if (!VirtualRangesOverlap(vaddr, size, block.map_vaddr, block.map_size)) {
			updated.push_back(block);
			continue;
		}
		const auto overlap_start = std::max(vaddr, block.map_vaddr);
		const auto overlap_end   = std::min(end, block_end);
		if (block.map_vaddr < overlap_start) {
			auto left     = block;
			left.size     = overlap_start - block.map_vaddr;
			left.map_size = left.size;
			updated.push_back(left);
		}
		auto middle       = block;
		middle.start_addr = block.start_addr + overlap_start - block.map_vaddr;
		middle.size       = overlap_end - overlap_start;
		middle.map_vaddr  = overlap_start;
		middle.map_size   = middle.size;
		middle.prot       = prot;
		middle.mode       = mode;
		middle.gpu_mode   = gpu_mode;
		updated.push_back(middle);
		if (overlap_end < block_end) {
			auto right       = block;
			right.start_addr = block.start_addr + overlap_end - block.map_vaddr;
			right.size       = block_end - overlap_end;
			right.map_vaddr  = overlap_end;
			right.map_size   = right.size;
			updated.push_back(right);
		}
	}
	m_mappings = std::move(updated);
}

bool PhysicalMemory::Find(uint64_t phys_addr, bool next, AllocatedBlock* out) {
	EXIT_IF(out == nullptr);

	Common::LockGuard lock(m_mutex);

	auto following = m_physical.upper_bound(phys_addr);
	if (following != m_physical.begin()) {
		const auto& block = std::prev(following)->second;
		if (phys_addr < block.start_addr + block.size) {
			*out = block;
			return true;
		}
	}

	if (next && following != m_physical.end()) {
		*out = following->second;
		return true;
	}

	return false;
}

std::vector<PhysicalMemory::AllocatedBlock> PhysicalMemory::FindMappings(uint64_t phys_addr,
                                                                         size_t   len) {
	Common::LockGuard lock(m_mutex);

	std::vector<AllocatedBlock> mappings;
	for (const auto& m: m_mappings) {
		if (!VirtualRangesOverlap(phys_addr, len, m.start_addr, m.size)) {
			continue;
		}

		const uint64_t overlap_start = std::max<uint64_t>(phys_addr, m.start_addr);
		const uint64_t overlap_end   = std::min<uint64_t>(phys_addr + len, m.start_addr + m.size);
		AllocatedBlock part          = m;
		part.start_addr              = overlap_start;
		part.size                    = overlap_end - overlap_start;
		part.map_vaddr += overlap_start - m.start_addr;
		part.map_size = part.size;
		mappings.push_back(part);
	}

	std::sort(mappings.begin(), mappings.end(),
	          [](const auto& a, const auto& b) { return a.map_vaddr < b.map_vaddr; });
	return mappings;
}

bool PhysicalMemory::Find(uint64_t vaddr, uint64_t* base_addr, size_t* len, int* prot,
                          VirtualMemory::Mode* mode, GpuAccessMode* gpu_mode) {
	Common::LockGuard lock(m_mutex);

	return std::any_of(m_mappings.begin(), m_mappings.end(),
	                   [vaddr, base_addr, len, prot, mode, gpu_mode](auto& b) {
		                   if (vaddr >= b.map_vaddr && vaddr < b.map_vaddr + b.map_size) {
			                   if (base_addr != nullptr) {
				                   *base_addr = b.map_vaddr;
			                   }
			                   if (len != nullptr) {
				                   *len = b.map_size;
			                   }
			                   if (prot != nullptr) {
				                   *prot = b.prot;
			                   }
			                   if (mode != nullptr) {
				                   *mode = b.mode;
			                   }
			                   if (gpu_mode != nullptr) {
				                   *gpu_mode = b.gpu_mode;
			                   }

			                   return true;
		                   }
		                   return false;
	                   });
}

void PhysicalMemory::SetVirtualRangeName(uint64_t vaddr, uint64_t len, const char* name) {
	Common::LockGuard lock(m_mutex);

	for (auto& b: m_mappings) {
		if (VirtualRangesOverlap(vaddr, len, b.map_vaddr, b.map_size)) {
			CopyVirtualRangeName(b.name, name);
		}
	}
}

bool FlexibleMemory::Map(uint64_t vaddr, size_t len, int prot, VirtualMemory::Mode mode,
                         GpuAccessMode gpu_mode, const char* name) {
	Common::LockGuard lock(m_mutex);

	const auto available = (Size() >= m_allocated_total ? Size() - m_allocated_total : 0);
	if (len == 0 || len > available) {
		LOGF_COLOR(Log::Color::Red,
		           "\t flexible memory exhausted: configured = 0x%016" PRIx64
		           ", allocated = 0x%016" PRIx64 ", available = 0x%016" PRIx64
		           ", requested = 0x%016" PRIx64 "\n",
		           Size(), m_allocated_total, available, static_cast<uint64_t>(len));
		return false;
	}

	std::vector<AllocatedBlock> blocks;
	auto                        current   = vaddr;
	auto                        remaining = static_cast<uint64_t>(len);
	for (const auto& [backing_offset, free_size]: m_free) {
		if (remaining == 0) {
			break;
		}
		const auto     chunk = std::min(remaining, free_size);
		AllocatedBlock block {};
		block.map_vaddr      = current;
		block.map_size       = chunk;
		block.backing_offset = backing_offset;
		block.host_vaddr     = vaddr;
		block.host_size      = len;
		block.prot           = prot;
		block.mode           = mode;
		block.gpu_mode       = gpu_mode;
		CopyVirtualRangeName(block.name, name);
		blocks.push_back(block);
		current += chunk;
		remaining -= chunk;
	}
	if (remaining != 0) {
		return false;
	}

	std::vector<AllocatedBlock> mapped;
	for (const auto& block: blocks) {
		if (!g_guest_address_space->ZeroBacking(block.backing_offset, block.map_size)) {
			for (auto it = mapped.rbegin(); it != mapped.rend(); ++it) {
				EXIT_IF(!g_guest_address_space->UnmapBacking(it->map_vaddr, it->map_size));
			}
			return false;
		}
		if (!g_guest_address_space->MapBacking(block.map_vaddr, block.map_size,
		                                       block.backing_offset, block.mode)) {
			for (auto it = mapped.rbegin(); it != mapped.rend(); ++it) {
				EXIT_IF(!g_guest_address_space->UnmapBacking(it->map_vaddr, it->map_size));
			}
			return false;
		}
		mapped.push_back(block);
	}

	for (const auto& block: blocks) {
		auto next = m_free.upper_bound(block.backing_offset);
		EXIT_IF(next == m_free.begin());
		auto range = std::prev(next);
		EXIT_IF(block.backing_offset < range->first ||
		        block.map_size > range->first + range->second - block.backing_offset);
		ConsumeFreeRange(range, block.backing_offset, block.map_size);
		m_allocated.push_back(block);
	}
	std::sort(m_allocated.begin(), m_allocated.end(),
	          [](const auto& left, const auto& right) { return left.map_vaddr < right.map_vaddr; });
	m_allocated_total += len;

	return true;
}

bool FlexibleMemory::Unmap(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode,
                           uint64_t* host_vaddr_to_release) {
	EXIT_IF(gpu_mode == nullptr);

	Common::LockGuard lock(m_mutex);

	if (host_vaddr_to_release != nullptr) {
		*host_vaddr_to_release = 0;
	}
	if (size == 0 || UINT64_MAX - vaddr < size) {
		return false;
	}

	const auto end     = vaddr + size;
	auto       current = vaddr;
	bool       found   = false;
	for (const auto& block: m_allocated) {
		if (block.map_vaddr + block.map_size <= current) {
			continue;
		}
		if (block.map_vaddr > current || block.map_vaddr >= end) {
			break;
		}
		if (!found) {
			*gpu_mode = block.gpu_mode;
			found     = true;
		}
		current = std::min(end, block.map_vaddr + block.map_size);
		if (current == end) {
			break;
		}
	}
	if (!found || current != end || !g_guest_address_space->UnmapBacking(vaddr, size)) {
		return false;
	}

	std::vector<AllocatedBlock> remaining;
	remaining.reserve(m_allocated.size() + 1);
	uint64_t removed = 0;
	for (const auto& block: m_allocated) {
		const auto block_end = block.map_vaddr + block.map_size;
		if (!VirtualRangesOverlap(vaddr, size, block.map_vaddr, block.map_size)) {
			remaining.push_back(block);
			continue;
		}
		const auto overlap_start = std::max(vaddr, block.map_vaddr);
		const auto overlap_end   = std::min(end, block_end);
		const auto overlap_size  = overlap_end - overlap_start;
		AddFreeRange(block.backing_offset + overlap_start - block.map_vaddr, overlap_size);
		removed += overlap_size;

		if (block.map_vaddr < overlap_start) {
			auto left     = block;
			left.map_size = overlap_start - block.map_vaddr;
			remaining.push_back(left);
		}
		if (overlap_end < block_end) {
			auto right           = block;
			right.map_vaddr      = overlap_end;
			right.map_size       = block_end - overlap_end;
			right.backing_offset = block.backing_offset + overlap_end - block.map_vaddr;
			remaining.push_back(right);
		}
	}
	EXIT_IF(removed != size || removed > m_allocated_total);
	m_allocated = std::move(remaining);
	m_allocated_total -= removed;
	return true;
}

void FlexibleMemory::ConsumeFreeRange(std::map<uint64_t, uint64_t>::iterator range, uint64_t start,
                                      uint64_t size) {
	const auto range_start = range->first;
	const auto range_end   = range->first + range->second;
	m_free.erase(range);
	if (range_start < start) {
		m_free.emplace(range_start, start - range_start);
	}
	if (start + size < range_end) {
		m_free.emplace(start + size, range_end - start - size);
	}
}

void FlexibleMemory::AddFreeRange(uint64_t start, uint64_t size) {
	auto end  = start + size;
	auto next = m_free.lower_bound(start);
	if (next != m_free.begin()) {
		auto previous = std::prev(next);
		if (previous->first + previous->second >= start) {
			start = previous->first;
			end   = std::max(end, previous->first + previous->second);
			next  = m_free.erase(previous);
		}
	}
	while (next != m_free.end() && next->first <= end) {
		end  = std::max(end, next->first + next->second);
		next = m_free.erase(next);
	}
	m_free.emplace(start, end - start);
}

bool FlexibleMemory::Snapshot(uint64_t vaddr, uint64_t size, std::vector<AllocatedBlock>* blocks) {
	EXIT_IF(blocks == nullptr);
	Common::LockGuard lock(m_mutex);
	blocks->clear();
	if (size == 0 || UINT64_MAX - vaddr < size) {
		return false;
	}

	const auto end     = vaddr + size;
	auto       current = vaddr;
	for (const auto& block: m_allocated) {
		const auto block_end = block.map_vaddr + block.map_size;
		if (block_end <= current) {
			continue;
		}
		if (block.map_vaddr > current || block.map_vaddr >= end) {
			break;
		}
		const auto part_end = std::min(end, block_end);
		auto       part     = block;
		part.map_vaddr      = current;
		part.map_size       = part_end - current;
		part.backing_offset += current - block.map_vaddr;
		blocks->push_back(part);
		current = part_end;
		if (current == end) {
			return true;
		}
	}
	blocks->clear();
	return false;
}

bool FlexibleMemory::Restore(const std::vector<AllocatedBlock>& blocks) {
	Common::LockGuard lock(m_mutex);
	if (blocks.empty()) {
		return false;
	}

	for (const auto& block: blocks) {
		auto next = m_free.upper_bound(block.backing_offset);
		if (next == m_free.begin()) {
			return false;
		}
		const auto range = std::prev(next);
		if (block.backing_offset < range->first ||
		    block.map_size > range->first + range->second - block.backing_offset) {
			return false;
		}
	}

	std::vector<AllocatedBlock> mapped;
	for (const auto& block: blocks) {
		if (!g_guest_address_space->MapBacking(block.map_vaddr, block.map_size,
		                                       block.backing_offset, block.mode)) {
			for (auto it = mapped.rbegin(); it != mapped.rend(); ++it) {
				EXIT_IF(!g_guest_address_space->UnmapBacking(it->map_vaddr, it->map_size));
			}
			return false;
		}
		mapped.push_back(block);
	}

	for (const auto& block: blocks) {
		auto next = m_free.upper_bound(block.backing_offset);
		EXIT_IF(next == m_free.begin());
		auto range = std::prev(next);
		ConsumeFreeRange(range, block.backing_offset, block.map_size);
		m_allocated.push_back(block);
		m_allocated_total += block.map_size;
	}
	std::sort(m_allocated.begin(), m_allocated.end(),
	          [](const auto& left, const auto& right) { return left.map_vaddr < right.map_vaddr; });
	return true;
}

void FlexibleMemory::Protect(uint64_t vaddr, uint64_t size, int prot, VirtualMemory::Mode mode,
                             GpuAccessMode gpu_mode) {
	Common::LockGuard lock(m_mutex);
	if (size == 0 || UINT64_MAX - vaddr < size) {
		return;
	}

	const auto                  end = vaddr + size;
	std::vector<AllocatedBlock> updated;
	updated.reserve(m_allocated.size() + 2);
	for (const auto& block: m_allocated) {
		const auto block_end = block.map_vaddr + block.map_size;
		if (!VirtualRangesOverlap(vaddr, size, block.map_vaddr, block.map_size)) {
			updated.push_back(block);
			continue;
		}
		const auto overlap_start = std::max(vaddr, block.map_vaddr);
		const auto overlap_end   = std::min(end, block_end);
		if (block.map_vaddr < overlap_start) {
			auto left     = block;
			left.map_size = overlap_start - block.map_vaddr;
			updated.push_back(left);
		}
		auto middle           = block;
		middle.map_vaddr      = overlap_start;
		middle.map_size       = overlap_end - overlap_start;
		middle.backing_offset = block.backing_offset + overlap_start - block.map_vaddr;
		middle.prot           = prot;
		middle.mode           = mode;
		middle.gpu_mode       = gpu_mode;
		updated.push_back(middle);
		if (overlap_end < block_end) {
			auto right           = block;
			right.map_vaddr      = overlap_end;
			right.map_size       = block_end - overlap_end;
			right.backing_offset = block.backing_offset + overlap_end - block.map_vaddr;
			updated.push_back(right);
		}
	}
	m_allocated = std::move(updated);
}

bool FlexibleMemory::Find(uint64_t vaddr, uint64_t* base_addr, size_t* len, int* prot,
                          VirtualMemory::Mode* mode, GpuAccessMode* gpu_mode) {
	Common::LockGuard lock(m_mutex);

	return std::any_of(m_allocated.begin(), m_allocated.end(),
	                   [vaddr, base_addr, len, prot, mode, gpu_mode](auto& b) {
		                   if (vaddr >= b.map_vaddr && vaddr < b.map_vaddr + b.map_size) {
			                   if (base_addr != nullptr) {
				                   *base_addr = b.map_vaddr;
			                   }
			                   if (len != nullptr) {
				                   *len = b.map_size;
			                   }
			                   if (prot != nullptr) {
				                   *prot = b.prot;
			                   }
			                   if (mode != nullptr) {
				                   *mode = b.mode;
			                   }
			                   if (gpu_mode != nullptr) {
				                   *gpu_mode = b.gpu_mode;
			                   }

			                   return true;
		                   }
		                   return false;
	                   });
}

void FlexibleMemory::SetVirtualRangeName(uint64_t vaddr, uint64_t len, const char* name) {
	Common::LockGuard lock(m_mutex);

	for (auto& b: m_allocated) {
		if (VirtualRangesOverlap(vaddr, len, b.map_vaddr, b.map_size)) {
			CopyVirtualRangeName(b.name, name);
		}
	}
}

void PhysicalMemory::SetVirtualRangeMemoryType(uint64_t vaddr, uint64_t len, int memory_type) {
	Common::LockGuard lock(m_mutex);

	for (auto& b: m_mappings) {
		if (VirtualRangesOverlap(vaddr, len, b.map_vaddr, b.map_size)) {
			b.memory_type = memory_type;
		}
	}
}

uint64_t FlexibleMemory::Available() {
	Common::LockGuard lock(m_mutex);

	return (Size() >= m_allocated_total ? Size() - m_allocated_total : 0);
}

void PooledMemory::AddFreeUnlocked(uint64_t start, uint64_t size) {
	if (size == 0) {
		return;
	}

	m_free.push_back({start, size});
	std::sort(m_free.begin(), m_free.end(),
	          [](const auto& left, const auto& right) { return left.start < right.start; });

	std::vector<PhysicalRange> merged;
	for (const auto& range: m_free) {
		if (!merged.empty() && range.start <= merged.back().start + merged.back().size) {
			const auto end =
			    std::max(merged.back().start + merged.back().size, range.start + range.size);
			merged.back().size = end - merged.back().start;
		} else {
			merged.push_back(range);
		}
	}
	m_free = std::move(merged);
}

void PooledMemory::Expand(uint64_t phys_addr, uint64_t size) {
	Common::LockGuard lock(m_mutex);
	m_expansions.push_back({phys_addr, size});
	AddFreeUnlocked(phys_addr, size);
}

bool PooledMemory::ReleaseExpansion(uint64_t phys_addr, uint64_t size) {
	Common::LockGuard lock(m_mutex);
	const auto        expansion = std::find_if(m_expansions.begin(), m_expansions.end(),
	                                           [phys_addr, size](const auto& range) {
		                                    return range.start == phys_addr && range.size == size;
	                                           });
	if (expansion == m_expansions.end()) {
		return false;
	}
	if (std::any_of(m_mappings.begin(), m_mappings.end(), [phys_addr, size](const auto& mapping) {
		    return VirtualRangesOverlap(phys_addr, size, mapping.phys_addr, mapping.size);
	    })) {
		return false;
	}

	const auto end = phys_addr + size;
	const auto free_range =
	    std::find_if(m_free.begin(), m_free.end(), [phys_addr, end](const auto& r) {
		    return phys_addr >= r.start && end <= r.start + r.size;
	    });
	if (free_range == m_free.end()) {
		return false;
	}

	const auto old        = *free_range;
	const auto old_end    = old.start + old.size;
	const auto left_size  = phys_addr - old.start;
	const auto right_size = old_end - end;
	m_free.erase(free_range);
	if (left_size != 0) {
		m_free.push_back({old.start, left_size});
	}
	if (right_size != 0) {
		m_free.push_back({end, right_size});
	}
	std::sort(m_free.begin(), m_free.end(),
	          [](const auto& left, const auto& right) { return left.start < right.start; });
	m_expansions.erase(expansion);
	return true;
}

bool PooledMemory::Allocate(uint64_t vaddr, uint64_t size, GpuAccessMode gpu_mode,
                            std::vector<Mapping>* mappings) {
	EXIT_IF(mappings == nullptr);
	mappings->clear();
	if (vaddr == 0 || size == 0 || UINT64_MAX - vaddr < size) {
		return false;
	}

	Common::LockGuard lock(m_mutex);
	auto              free      = m_free;
	auto              current   = vaddr;
	auto              remaining = size;
	for (auto& range: free) {
		if (remaining == 0 || range.size == 0) {
			continue;
		}
		const auto part_size = std::min(range.size, remaining);
		mappings->push_back({current, part_size, range.start, gpu_mode});
		range.start += part_size;
		range.size -= part_size;
		current += part_size;
		remaining -= part_size;
	}
	if (remaining != 0) {
		mappings->clear();
		return false;
	}

	free.erase(
	    std::remove_if(free.begin(), free.end(), [](const auto& range) { return range.size == 0; }),
	    free.end());
	m_free = std::move(free);
	m_mappings.insert(m_mappings.end(), mappings->begin(), mappings->end());
	return true;
}

bool PooledMemory::QueryUnlocked(uint64_t vaddr, uint64_t size,
                                 std::vector<Mapping>* mappings) const {
	EXIT_IF(mappings == nullptr);
	mappings->clear();
	if (vaddr == 0 || size == 0 || UINT64_MAX - vaddr < size) {
		return false;
	}

	const auto end     = vaddr + size;
	auto       current = vaddr;
	while (current < end) {
		const auto it =
		    std::find_if(m_mappings.begin(), m_mappings.end(), [current](const auto& m) {
			    return current >= m.vaddr && current < m.vaddr + m.size;
		    });
		if (it == m_mappings.end()) {
			mappings->clear();
			return false;
		}
		const auto part_size = std::min(end, it->vaddr + it->size) - current;
		mappings->push_back(
		    {current, part_size, it->phys_addr + (current - it->vaddr), it->gpu_mode});
		current += part_size;
	}
	return true;
}

bool PooledMemory::Query(uint64_t vaddr, uint64_t size, std::vector<Mapping>* mappings) {
	Common::LockGuard lock(m_mutex);
	return QueryUnlocked(vaddr, size, mappings);
}

bool PooledMemory::Release(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode) {
	EXIT_IF(gpu_mode == nullptr);

	Common::LockGuard    lock(m_mutex);
	std::vector<Mapping> released;
	if (!QueryUnlocked(vaddr, size, &released)) {
		return false;
	}

	*gpu_mode                = released.front().gpu_mode;
	const auto           end = vaddr + size;
	std::vector<Mapping> kept;
	for (const auto& mapping: m_mappings) {
		const auto mapping_end = mapping.vaddr + mapping.size;
		const auto cut_start   = std::max(vaddr, mapping.vaddr);
		const auto cut_end     = std::min(end, mapping_end);
		if (cut_start >= cut_end) {
			kept.push_back(mapping);
			continue;
		}
		if (mapping.vaddr < cut_start) {
			auto left = mapping;
			left.size = cut_start - mapping.vaddr;
			kept.push_back(left);
		}
		if (cut_end < mapping_end) {
			auto right      = mapping;
			right.vaddr     = cut_end;
			right.size      = mapping_end - cut_end;
			right.phys_addr = mapping.phys_addr + (cut_end - mapping.vaddr);
			kept.push_back(right);
		}
	}
	m_mappings = std::move(kept);
	for (const auto& mapping: released) {
		AddFreeUnlocked(mapping.phys_addr, mapping.size);
	}
	return true;
}

uint64_t PooledMemory::Available() {
	Common::LockGuard lock(m_mutex);
	uint64_t          available = 0;
	for (const auto& range: m_free) {
		available += range.size;
	}
	return available;
}

std::vector<PooledMemory::Mapping> PooledMemory::GetMappings() {
	Common::LockGuard lock(m_mutex);
	return m_mappings;
}

static bool UnmapPooledBackingTransactional(const std::vector<PooledMemory::Mapping>& mappings,
                                            VirtualMemory::Mode                       mode) {
	std::vector<PooledMemory::Mapping> removed;
	for (const auto& mapping: mappings) {
		if (!g_guest_address_space->UnmapBacking(mapping.vaddr, mapping.size)) {
			for (auto it = removed.rbegin(); it != removed.rend(); ++it) {
				auto       failure_reason = GuestBackingStore::FailureReason::None;
				const bool restored       = g_guest_address_space->MapBacking(
				    it->vaddr, it->size, it->phys_addr, mode, &failure_reason);
				if (!restored) {
					EXIT("pooled-memory unmap rollback failed: %s\n",
					     GuestBackingStore::GetFailureReasonName(failure_reason));
				}
			}
			return false;
		}
		removed.push_back(mapping);
	}
	return true;
}

int32_t KYTY_SYSV_ABI KernelMapNamedFlexibleMemory(void** addr_in_out, size_t len, int prot,
                                                   int flags, const char* name) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	EXIT_NOT_IMPLEMENTED(addr_in_out == nullptr);

	constexpr size_t   PAGE_SIZE                = 0x4000;
	constexpr size_t   MAXIMUM_NAME_SIZE        = 32;
	constexpr uint64_t DEFAULT_PS5_BASE         = 0x200000000;
	constexpr uint32_t GUEST_MAP_FIXED          = 0x10;
	constexpr uint32_t GUEST_MAP_NO_OVERWRITE   = 0x80;
	constexpr uint32_t GUEST_MAP_DMEM_COMPAT    = 0x400;
	constexpr uint32_t GUEST_MAP_UNKNOWN_8000   = 0x8000;
	constexpr uint32_t GUEST_MAP_NO_COALESCE    = 0x400000;
	constexpr uint32_t GUEST_MAP_ALIGNMENT_MASK = 0xff000000;
	constexpr uint32_t SUPPORTED_MAP_BITS       = GUEST_MAP_FIXED | GUEST_MAP_NO_OVERWRITE |
	                                              GUEST_MAP_DMEM_COMPAT | GUEST_MAP_UNKNOWN_8000 |
	                                              GUEST_MAP_NO_COALESCE | GUEST_MAP_ALIGNMENT_MASK;

	if (len == 0 || (len & (PAGE_SIZE - 1)) != 0) {
		return KERNEL_ERROR_EINVAL;
	}
	if (len > g_flexible_memory->Available()) {
		return KERNEL_ERROR_ENOMEM;
	}

	if (name == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (std::strlen(name) >= MAXIMUM_NAME_SIZE) {
		return KERNEL_ERROR_ENAMETOOLONG;
	}

	const auto map_flags       = static_cast<uint32_t>(flags);
	const auto alignment_shift = (map_flags & GUEST_MAP_ALIGNMENT_MASK) >> 24u;
	if ((map_flags & ~SUPPORTED_MAP_BITS) != 0 ||
	    (alignment_shift != 0 && (alignment_shift < 14 || alignment_shift > 31))) {
		LOGF_COLOR(Log::Color::Red, "\t unsupported flags = 0x%08" PRIx32 "\n", map_flags);
		return KERNEL_ERROR_EINVAL;
	}
	const uint64_t map_alignment =
	    alignment_shift != 0 ? uint64_t {1} << alignment_shift : PAGE_SIZE;

	VirtualMemory::Mode mode     = VirtualMemory::Mode::NoAccess;
	GpuAccessMode       gpu_mode = GpuAccessMode::NoAccess;

	if (!DecodeMemoryProtection(prot, &mode, &gpu_mode)) {
		EXIT("unknown prot: %d\n", prot);
	}

	auto     in_addr         = reinterpret_cast<uint64_t>(*addr_in_out);
	uint64_t out_addr        = 0;
	bool     reserved_target = false;

	if ((flags & GUEST_MAP_FIXED) != 0) {
		if (in_addr == 0 || (in_addr & (PAGE_SIZE - 1)) != 0 ||
		    (in_addr & (map_alignment - 1u)) != 0) {
			return KERNEL_ERROR_EINVAL;
		}
		if ((flags & GUEST_MAP_NO_OVERWRITE) != 0 && g_virtual_ranges->HasOverlap(in_addr, len)) {
			return KERNEL_ERROR_ENOMEM;
		}
		std::vector<VirtualRanges::Range> reserved_ranges;
		if (g_virtual_ranges->QuerySpan(in_addr, len, &reserved_ranges) &&
		    std::all_of(reserved_ranges.begin(), reserved_ranges.end(), [](const auto& range) {
			    return range.type == VirtualRangeType::Reserved;
		    })) {
			UnmapGpuRange(in_addr, len);
			reserved_target = true;
			out_addr        = in_addr;
		}
		if (!reserved_target && ReplaceFixedRangeWithReserved(in_addr, len)) {
			reserved_target = true;
			out_addr        = in_addr;
		}
	} else {
		const auto search_addr = (in_addr != 0 ? in_addr : DEFAULT_PS5_BASE);
		out_addr               = FindGuestFreeRange(search_addr, len, map_alignment);
		if (out_addr != 0) {
			UnmapGpuRange(out_addr, len);
		}
	}

	*addr_in_out = reinterpret_cast<void*>(out_addr);

	if (out_addr == 0) {
		return KERNEL_ERROR_ENOMEM;
	}

	if (!g_flexible_memory->Map(out_addr, len, prot, mode, gpu_mode, name)) {
		LOGF_COLOR(Log::Color::Red, "\t [Fail]\n");
		return KERNEL_ERROR_ENOMEM;
	}

	const bool published = reserved_target
	                           ? g_virtual_ranges->ReplaceSpan(
	                                 out_addr, len, VirtualRangeType::Reserved, 0, prot, 0,
	                                 VirtualRangeType::Flexible, name,
	                                 (map_flags & GUEST_MAP_NO_COALESCE) != 0)
	                           : g_virtual_ranges->Add(out_addr, len, 0, prot, 0,
	                                                   VirtualRangeType::Flexible, name,
	                                                   (map_flags & GUEST_MAP_NO_COALESCE) != 0);
	if (!published) {
		GpuAccessMode rollback_gpu_mode = GpuAccessMode::NoAccess;
		EXIT_IF(!g_flexible_memory->Unmap(out_addr, len, &rollback_gpu_mode));
		return KERNEL_ERROR_EBUSY;
	}

	LOGF("\t in_addr  = 0x%016" PRIx64 "\n"
	     "\t out_addr = 0x%016" PRIx64 "\n"
	     "\t size     = %" PRIu64 "\n"
	     "\t mode     = %s\n"
	     "\t flags    = 0x%08" PRIx32 "\n"
	     "\t name     = %s\n"
	     "\t gpu_mode = %s\n",
	     in_addr, out_addr, len, magic_enum::enum_name(mode), static_cast<uint32_t>(flags), name,
	     magic_enum::enum_name(gpu_mode));

	MapGpuRange(out_addr, len);

	if (g_alloc_callback != nullptr) {
		g_alloc_callback(out_addr, len);
	}

	return OK;
}

int KYTY_SYSV_ABI KernelMapFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags) {
	return KernelMapNamedFlexibleMemory(addr_in_out, len, prot, flags, "");
}

int KYTY_SYSV_ABI KernelSetPrtAperture(int index, void* addr, size_t len) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	const auto address = reinterpret_cast<uint64_t>(addr);

	LOGF("\t index = %d\n"
	     "\t addr  = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n",
	     index, address, static_cast<uint64_t>(len));

	if (index < 0 || index > PRT_APERTURE_MAX_INDEX) {
		return KERNEL_ERROR_EINVAL;
	}

	if (len != 0 && (address == 0 || (address & (PRT_PAGE_SIZE - 1u)) != 0 ||
	                 (len & (PRT_PAGE_SIZE - 1u)) != 0 || address < PRT_APERTURE_START ||
	                 len > PRT_APERTURE_END - address)) {
		return KERNEL_ERROR_EINVAL;
	}

	PrtAperture old {};
	{
		Common::LockGuard lock(g_prt_aperture_mutex);
		old = g_prt_apertures[static_cast<size_t>(index)];
	}
	if (old.size != 0) {
		UnmapGpuRange(old.address, old.size);
	}
	{
		Common::LockGuard lock(g_prt_aperture_mutex);
		g_prt_apertures[static_cast<size_t>(index)] =
		    len == 0 ? PrtAperture {} : PrtAperture {address, static_cast<uint64_t>(len)};
	}
	if (len != 0) {
		MapGpuRange(address, len);
	}

	LOGF_COLOR(Log::Color::Green, "\t[Ok]\n");

	return OK;
}

int KYTY_SYSV_ABI KernelGetPrtAperture(int index, void** addr, size_t* len) {
	PRINT_NAME();

	LOGF("\t index = %d\n"
	     "\t addr  = %p\n"
	     "\t len   = %p\n",
	     index, static_cast<void*>(addr), static_cast<void*>(len));

	if (index < 0 || index > PRT_APERTURE_MAX_INDEX) {
		return KERNEL_ERROR_EINVAL;
	}

	if (addr == nullptr || len == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	PrtAperture aperture {};
	{
		Common::LockGuard lock(g_prt_aperture_mutex);
		aperture = g_prt_apertures[static_cast<size_t>(index)];
	}

	*addr = reinterpret_cast<void*>(aperture.address);
	*len  = static_cast<size_t>(aperture.size);

	LOGF_COLOR(Log::Color::Green,
	           "\t *addr = 0x%016" PRIx64 "\n"
	           "\t *len  = 0x%016" PRIx64 "\n"
	           "\t[Ok]\n",
	           aperture.address, aperture.size);

	return OK;
}

int KYTY_SYSV_ABI KernelSetVirtualRangeName(const void* addr, uint64_t len, const char* name) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	auto vaddr = reinterpret_cast<uint64_t>(addr);

	LOGF("\t addr = 0x%016" PRIx64 "\n"
	     "\t len  = 0x%016" PRIx64 "\n"
	     "\t name = %s\n",
	     vaddr, len, name != nullptr ? name : "(null)");

	if (name == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (std::strlen(name) >= KERNEL_MAXIMUM_NAME_LENGTH) {
		return KERNEL_ERROR_ENAMETOOLONG;
	}

	g_physical_memory->SetVirtualRangeName(vaddr, len, name);
	g_flexible_memory->SetVirtualRangeName(vaddr, len, name);
	g_virtual_ranges->Rename(vaddr, len, name);

	return OK;
}

int KYTY_SYSV_ABI KernelClearVirtualRangeName(const void* addr, uint64_t len) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	auto vaddr = reinterpret_cast<uint64_t>(addr);

	LOGF("\t addr = 0x%016" PRIx64 "\n"
	     "\t len  = 0x%016" PRIx64 "\n",
	     vaddr, len);

	g_physical_memory->SetVirtualRangeName(vaddr, len, "");
	g_flexible_memory->SetVirtualRangeName(vaddr, len, "");
	g_virtual_ranges->Rename(vaddr, len, "");

	return OK;
}

static bool FreeGuestMemoryOwner(uint64_t vaddr, uint64_t size) {
	return g_guest_address_space->ReleaseCommitted(vaddr, size) &&
	       g_virtual_ranges->Remove(vaddr, size);
}

static int UnmapMemoryRange(uint64_t vaddr, size_t len) {
	if (len == 0 || UINT64_MAX - vaddr < len) {
		return KERNEL_ERROR_EINVAL;
	}

	VirtualRanges::Range range {};
	if (!g_virtual_ranges->Query(vaddr, 0, &range)) {
		return KERNEL_ERROR_EACCES;
	}
	const auto chunk_len = std::min<uint64_t>(len, range.size - (vaddr - range.start));
	if (chunk_len < len) {
		const int ret = UnmapMemoryRange(vaddr, chunk_len);
		return ret == OK ? UnmapMemoryRange(vaddr + chunk_len, len - chunk_len) : ret;
	}
	if (IsReservedRangeType(range.type)) {
		if (!g_guest_address_space->ReleaseFree(vaddr, len)) {
			return KERNEL_ERROR_EACCES;
		}
		g_virtual_ranges->Remove(vaddr, len);
		return OK;
	}
	if (range.type == VirtualRangeType::Code || range.type == VirtualRangeType::Runtime) {
		return FreeGuestMemoryOwner(vaddr, len) ? OK : KERNEL_ERROR_EACCES;
	}

	GpuAccessMode gpu_mode       = GpuAccessMode::NoAccess;
	bool          owner_unmapped = false;
	if (range.type == VirtualRangeType::Pooled) {
		std::vector<PooledMemory::Mapping> mappings;
		if (g_pooled_memory->Query(vaddr, len, &mappings)) {
			VirtualMemory::Mode mode        = VirtualMemory::Mode::NoAccess;
			GpuAccessMode       decoded_gpu = GpuAccessMode::NoAccess;
			owner_unmapped = DecodeMemoryProtection(range.protection, &mode, &decoded_gpu) &&
			                 UnmapPooledBackingTransactional(mappings, mode);
			if (owner_unmapped && !g_pooled_memory->Release(vaddr, len, &gpu_mode)) {
				EXIT("failed to release unmapped pooled-memory range\n");
			}
		}
		if (!owner_unmapped) {
			return KERNEL_ERROR_EACCES;
		}
	} else if (range.type == VirtualRangeType::Direct) {
		VirtualMemory::Mode direct_mode   = VirtualMemory::Mode::NoAccess;
		GpuAccessMode       direct_gpu    = GpuAccessMode::NoAccess;
		const auto          direct_offset = range.offset + vaddr - range.start;
		if (DecodeMemoryProtection(range.protection, &direct_mode, &direct_gpu) &&
		    g_guest_address_space->BackingContains(vaddr, len) &&
		    g_guest_address_space->UnmapBacking(vaddr, len)) {
			uint64_t ignored_host = 0;
			if (g_physical_memory->Unmap(vaddr, len, &gpu_mode, &ignored_host)) {
				owner_unmapped = true;
			} else {
				EXIT_IF(!g_guest_address_space->MapBacking(vaddr, len, direct_offset, direct_mode));
			}
		}
	} else if (range.type == VirtualRangeType::Stack) {
		owner_unmapped = g_guest_address_space->ReleaseCommitted(vaddr, len);
	} else {
		uint64_t ignored_host = 0;
		owner_unmapped        = g_flexible_memory->Unmap(vaddr, len, &gpu_mode, &ignored_host);
	}
	if (!owner_unmapped) {
		return KERNEL_ERROR_EACCES;
	}

	g_virtual_ranges->Remove(vaddr, len);

	if (g_free_callback != nullptr && IsCommittedRangeType(range.type)) {
		g_free_callback(vaddr, len);
	}
	if (range.type == VirtualRangeType::Pooled) {
		MemoryPoolSubtractCommitted(len);
	}

	return OK;
}

int KYTY_SYSV_ABI KernelMunmap(uint64_t vaddr, size_t len) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t start = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n",
	     vaddr, len);

	if (len == 0 || UINT64_MAX - vaddr < len) {
		return KERNEL_ERROR_EINVAL;
	}
	std::vector<VirtualRanges::Range> ranges;
	if (!g_virtual_ranges->QuerySpan(vaddr, len, &ranges)) {
		return KERNEL_ERROR_EACCES;
	}
	UnmapGpuRange(vaddr, len);
	return UnmapMemoryRange(vaddr, len);
}

DirectMemoryUse GetDirectMemoryUse() {
	DirectMemoryUse use;
	if (g_physical_memory == nullptr) {
		return use;
	}
	use.size = PhysicalMemory::Size();
	Common::LockGuard lock(g_physical_memory->GetMutex());
	for (const auto& [start, block]: g_physical_memory->GetPhysicalBlocks()) {
		use.allocated += block.size;
		use.blocks++;
		if (start >= PhysicalMemory::TotalSize()) {
			use.overflow += block.size;
		}
	}
	return use;
}

void SetBackingCopyLocked(bool locked) {
	g_backing_copy_locked_step.store(locked, std::memory_order_relaxed);
}

void SetProtectLockExclusive(bool exclusive) {
	g_protect_exclusive_step.store(exclusive, std::memory_order_relaxed);
}

size_t KYTY_SYSV_ABI KernelGetDirectMemorySize() {
	PRINT_NAME();

	return PhysicalMemory::Size();
}

// A request for 0 bytes of direct memory gets EINVAL, as from the kernel, and the guest may ask
// again at once. GT7 did, forever, when time trial was selected on its track map (2026-10-14), and
// the log of each try and of the AvailableDirectMemorySize query before it grew to 34 GB. Such a
// request is described the first times, with the guest code it came from (to see how it came to
// ask for 0 bytes); once the same one has failed four times in a row, it and those queries are
// logged at powers of two only. Guarded by g_memory_operation_mutex.
struct ZeroLengthRequest {
	int64_t  search_start = 0;
	int64_t  search_end   = 0;
	size_t   alignment    = 0;
	int      memory_type  = 0;
	uint64_t caller       = 0;
	uint64_t count        = 0; // in a row
};
static ZeroLengthRequest g_zero_length_request;
static uint32_t          g_zero_length_described = 0;
static uint64_t          g_quiet_queries         = 0;

// Direct memory running out: the guest may try again at once, forever (GT7 on its time selection
// screen, 2026-10-14: a loop of queries, small allocations, an 8 MB one that fails and releases,
// 5 GB of log). Each failure is counted; at powers of two the log gets what the memory is used
// for, and after 256 failures the lines of each AllocateDirectMemory, AvailableDirectMemorySize
// and ReleaseDirectMemory call are left out until 4,096 calls in a row went without one.
// Guarded by g_memory_operation_mutex.
struct DirectMemoryPressure {
	uint64_t failures         = 0;
	uint64_t calls_since_fail = 0;
	bool     quiet            = false;
	// Unchecked releases of ranges that were not one allocated span (see
	// KernelReleaseDirectMemory): how many, and what they freed.
	uint64_t partial_releases = 0;
	uint64_t partial_freed    = 0;
};
static DirectMemoryPressure g_direct_memory_pressure;

static bool DirectMemoryQuiet() {
	auto& pressure = g_direct_memory_pressure;
	if (pressure.quiet && ++pressure.calls_since_fail >= 4096) {
		pressure.quiet = false;
		LOGF("Direct memory: 4096 calls without a failed allocation; their lines are logged "
		     "again\n");
	}
	return pressure.quiet;
}

static void ReportDirectMemoryFailure(uint64_t len, uint64_t alignment, int memory_type) {
	auto& pressure            = g_direct_memory_pressure;
	pressure.calls_since_fail = 0;
	const auto failures       = ++pressure.failures;
	if (failures >= 256 && !pressure.quiet) {
		pressure.quiet = true;
		LOGF("Direct memory: %" PRIu64 " failed allocations; the lines of each allocation, query "
		     "and release are left out until 4096 calls in a row go without a failure\n",
		     failures);
	}
	if (!std::has_single_bit(failures)) {
		return;
	}
	uint64_t                                   allocated[3] = {};
	uint64_t                                   blocks       = 0;
	uint64_t                                   mapped       = 0;
	uint64_t                                   largest_free = 0;
	uint64_t                                   free_at      = 0;
	std::map<int, uint64_t>                    by_type;
	std::vector<std::pair<uint64_t, uint64_t>> largest; // size, start
	{
		Common::LockGuard lock(g_physical_memory->GetMutex());
		for (const auto& [start, block]: g_physical_memory->GetPhysicalBlocks()) {
			allocated[static_cast<int>(block.kind)] += block.size;
			by_type[block.memory_type] += block.size;
			blocks++;
			largest.emplace_back(block.size, start);
		}
		for (const auto& mapping: g_physical_memory->GetMappings()) {
			mapped += mapping.map_size;
		}
	}
	(void)g_physical_memory->Available(0, PhysicalMemory::Size(), 0, &free_at, &largest_free);
	std::ranges::sort(largest, std::greater {});
	char        part[64];
	std::string types;
	for (const auto& [type, bytes]: by_type) {
		std::snprintf(part, sizeof(part), "%stype %d %" PRIu64 " MiB", types.empty() ? "" : ", ",
		              type, bytes >> 20u);
		types += part;
	}
	std::string biggest;
	for (size_t i = 0; i < largest.size() && i < 8; i++) {
		std::snprintf(part, sizeof(part), "%s0x%" PRIx64 "+0x%" PRIx64, i == 0 ? "" : ", ",
		              largest[i].second, largest[i].first);
		biggest += part;
	}
	const auto total = allocated[0] + allocated[1] + allocated[2];
	LOGF("Direct memory: no 0x%" PRIx64 " bytes (alignment 0x%" PRIx64 ", type %d) to allocate, "
	     "%" PRIu64 " failures so far. %" PRIu64 " of %" PRIu64 " MiB allocated in %" PRIu64
	     " blocks (direct %" PRIu64 ", pooled %" PRIu64 ", automatic %" PRIu64 " MiB; %s), %" PRIu64
	     " MiB mapped; the largest free range 0x%" PRIx64 "+0x%" PRIx64 "; the largest blocks %s; "
	     "%" PRIu64 " releases of ranges that were not one allocation freed %" PRIu64
	     " MiB; asked from 0x%016" PRIx64 "\n",
	     len, alignment, memory_type, failures, total >> 20u, PhysicalMemory::Size() >> 20u, blocks,
	     allocated[0] >> 20u, allocated[1] >> 20u, allocated[2] >> 20u, types.c_str(),
	     mapped >> 20u, free_at, largest_free, biggest.c_str(), pressure.partial_releases,
	     pressure.partial_freed >> 20u, t_direct_memory_caller);
}

// Whether this AvailableDirectMemorySize query's lines are left out (see ZeroLengthRequest).
static bool QuietDirectMemoryQuery() {
	if (g_zero_length_request.count < 4) {
		return false;
	}
	return !std::has_single_bit(++g_quiet_queries);
}

static void NoteZeroLengthRequest(int64_t search_start, int64_t search_end, size_t alignment,
                                  int memory_type, uint64_t caller) {
	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);
	auto&                                 request = g_zero_length_request;
	const bool same = request.count != 0 && request.search_start == search_start &&
	                  request.search_end == search_end && request.alignment == alignment &&
	                  request.memory_type == memory_type && request.caller == caller;
	if (!same) {
		request         = {search_start, search_end, alignment, memory_type, caller, 0};
		g_quiet_queries = 0;
	}
	request.count++;
	if (request.count >= 4 && std::has_single_bit(request.count)) {
		LOGF("AllocateDirectMemory: the same request for 0 bytes (EINVAL) %" PRIu64
		     " times in a row, from 0x%016" PRIx64
		     "; it and the AvailableDirectMemorySize queries are logged at powers of two only\n",
		     request.count, caller);
	}
	if (request.count != 1 || g_zero_length_described >= 3) {
		return;
	}
	g_zero_length_described++;
	LOGF("AllocateDirectMemory: a request for 0 bytes (EINVAL), type %d, alignment 0x%" PRIx64
	     ", search 0x%" PRIx64 "-0x%" PRIx64 ", from guest code at 0x%016" PRIx64
	     "; its code before that:\n",
	     memory_type, static_cast<uint64_t>(alignment), search_start, search_end, caller);
	constexpr uint64_t                  Before = 0x200;
	constexpr uint64_t                  After  = 0x40;
	std::array<uint8_t, Before + After> code {};
	if (caller < Before || !TryReadBacking(caller - Before, code.data(), code.size())) {
		LOGF("AllocateDirectMemory: (the code is not readable)\n");
		return;
	}
	for (uint64_t row = 0; row < code.size(); row += 32) {
		char line[32 * 2 + 1] {};
		for (uint64_t k = 0; k < 32; k++) {
			std::snprintf(line + k * 2, 3, "%02x", code[row + k]);
		}
		LOGF("AllocateDirectMemory: code 0x%016" PRIx64 ": %s\n", caller - Before + row, line);
	}
}

int KYTY_SYSV_ABI KernelAvailableDirectMemorySize(int64_t search_start, int64_t search_end,
                                                  size_t alignment, int64_t* phys_addr_out,
                                                  size_t* size_out) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	const bool zero_length_quiet = QuietDirectMemoryQuery();
	const bool pressure_quiet    = DirectMemoryQuiet();
	const bool quiet             = zero_length_quiet || pressure_quiet;
	if (!quiet) {
		LOGF("\t search_start = 0x%016" PRIx64 "\n"
		     "\t search_end   = 0x%016" PRIx64 "\n"
		     "\t alignment    = 0x%016" PRIx64 "\n",
		     search_start, search_end, static_cast<uint64_t>(alignment));
	}

	if (phys_addr_out == nullptr || size_out == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*phys_addr_out = 0;
	*size_out      = 0;

	if (search_start < 0 || search_end < 0) {
		return KERNEL_ERROR_EINVAL;
	}

	if (search_end <= search_start) {
		LOGF_COLOR(Log::Color::Red, "\t[Fail]\n");
		return KERNEL_ERROR_ENOMEM;
	}

	uint64_t phys_addr = 0;
	uint64_t size      = 0;
	if (!g_physical_memory->Available(static_cast<uint64_t>(search_start),
	                                  static_cast<uint64_t>(search_end), alignment, &phys_addr,
	                                  &size)) {
		if (!quiet) {
			LOGF_COLOR(Log::Color::Red, "\t[Fail]\n");
		}
		return KERNEL_ERROR_ENOMEM;
	}

	*phys_addr_out = static_cast<int64_t>(phys_addr);
	*size_out      = static_cast<size_t>(size);

	if (!quiet) {
		LOGF_COLOR(Log::Color::Green,
		           "\t phys_addr = 0x%016" PRIx64 "\n"
		           "\t size      = 0x%016" PRIx64 "\n"
		           "\t[Ok]\n",
		           phys_addr, size);
	}

	return OK;
}

int KYTY_SYSV_ABI KernelGetPageTableStats(int* cpu_total, int* cpu_available, int* gpu_total,
                                          int* gpu_available) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	if (cpu_total == nullptr || cpu_available == nullptr || gpu_total == nullptr ||
	    gpu_available == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	const auto cpu_used =
	    (g_virtual_ranges != nullptr ? g_virtual_ranges->CountPageTableEntries(false) : 0);
	const auto gpu_used =
	    (g_virtual_ranges != nullptr ? g_virtual_ranges->CountPageTableEntries(true) : 0);

	*cpu_total     = PAGE_TABLE_POOL_ENTRIES;
	*gpu_total     = PAGE_TABLE_POOL_ENTRIES;
	*cpu_available = PAGE_TABLE_POOL_ENTRIES -
	                 static_cast<int>(std::min<uint64_t>(cpu_used, PAGE_TABLE_POOL_ENTRIES));
	*gpu_available = PAGE_TABLE_POOL_ENTRIES -
	                 static_cast<int>(std::min<uint64_t>(gpu_used, PAGE_TABLE_POOL_ENTRIES));

	LOGF_COLOR(Log::Color::Green,
	           "\t cpu_total     = %d\n"
	           "\t cpu_available = %d\n"
	           "\t gpu_total     = %d\n"
	           "\t gpu_available = %d\n"
	           "\t[Ok]\n",
	           *cpu_total, *cpu_available, *gpu_total, *gpu_available);

	return OK;
}

int KYTY_SYSV_ABI KernelDirectMemoryQuery(int64_t offset, int flags, void* info, size_t info_size) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t offset    = 0x%016" PRIx64 "\n"
	     "\t flags     = 0x%08" PRIx32 "\n"
	     "\t info_size = 0x%016" PRIx64 "\n",
	     offset, flags, info_size);

	struct QueryInfo {
		int64_t start;
		int64_t end;
		int     memory_type;
	};

	if (offset < 0 || (flags != 0 && flags != 1) || info_size != sizeof(QueryInfo) ||
	    info == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	auto* query_info = static_cast<QueryInfo*>(info);

	PhysicalMemory::AllocatedBlock block {};
	{
		Common::LockGuard lock(g_physical_memory->GetMutex());
		const auto&       blocks  = g_physical_memory->GetPhysicalBlocks();
		auto              current = blocks.upper_bound(static_cast<uint64_t>(offset));
		if (current != blocks.begin()) {
			auto previous = std::prev(current);
			if (static_cast<uint64_t>(offset) <
			    previous->second.start_addr + previous->second.size) {
				current = previous;
			}
		}
		// A walk of the direct memory (flags 1) from inside it ends at its end: the blocks given
		// room past it (round 81) answer only queries of their own addresses.
		const bool past_end = flags == 1 && current != blocks.end() &&
		                      static_cast<uint64_t>(offset) < PhysicalMemory::Size() &&
		                      current->second.start_addr >= PhysicalMemory::Size();
		if (current == blocks.end() || past_end ||
		    (flags == 0 && (static_cast<uint64_t>(offset) < current->second.start_addr ||
		                    static_cast<uint64_t>(offset) >=
		                        current->second.start_addr + current->second.size))) {
			if (flags == 1 && static_cast<uint64_t>(offset) < PhysicalMemory::Size()) {
				query_info->start       = static_cast<int64_t>(PhysicalMemory::Size());
				query_info->end         = static_cast<int64_t>(PhysicalMemory::Size());
				query_info->memory_type = 0;
				LOGF_COLOR(Log::Color::Green, "\t terminal    = true\n\t[Ok]\n");
				return OK;
			}

			LOGF_COLOR(Log::Color::Red, "\t[Fail]\n");
			return KERNEL_ERROR_EACCES;
		}

		block        = current->second;
		uint64_t end = block.start_addr + block.size;
		for (auto following = std::next(current);
		     following != blocks.end() && following->second.start_addr == end &&
		     following->second.memory_type == block.memory_type;
		     ++following) {
			end += following->second.size;
		}
		block.size = end - block.start_addr;
	}

	query_info->start       = static_cast<int64_t>(block.start_addr);
	query_info->end         = static_cast<int64_t>(block.start_addr + block.size);
	query_info->memory_type = block.memory_type;

	LOGF_COLOR(Log::Color::Green,
	           "\t start       = %016" PRIx64 "\n"
	           "\t end         = %016" PRIx64 "\n"
	           "\t memory_type = %d\n"
	           "\t[Ok]\n",
	           query_info->start, query_info->end, query_info->memory_type);

	return OK;
}

// Round 82: every direct allocation and release of 256 MB or more, with the guest code that asked
// and the memory in use after it, logged however quiet the allocation lines are. The race's pool
// (5.3 GB) failed in some runs because the game held one or two 2.3 GB blocks other runs did not
// have at that point: these lines show where those come from and whether they are released.
static void NoteBigDirectMemoryChange(bool allocated, uint64_t start, uint64_t len, int memory_type,
                                      uint64_t caller) {
	constexpr uint64_t Big = 256ull << 20u;
	if (len < Big) {
		return;
	}
	uint64_t in_use = 0;
	uint64_t blocks = 0;
	{
		Common::LockGuard lock(g_physical_memory->GetMutex());
		for (const auto& [block_start, block]: g_physical_memory->GetPhysicalBlocks()) {
			(void)block_start;
			in_use += block.size;
			blocks++;
		}
	}
	LOGF("Direct memory: %s 0x%" PRIx64 " bytes (%" PRIu64 " MiB) at 0x%" PRIx64
	     "%s, from 0x%016" PRIx64 " on %s: %" PRIu64 " MiB in use in %" PRIu64 " blocks\n",
	     allocated ? "allocated" : "released", len, len >> 20u, start,
	     allocated ? fmt::format(" (type {})", memory_type).c_str() : "", caller,
	     Common::PerfStats::ThreadNameOf(Common::PerfStats::CurrentOsThreadId()).c_str(),
	     in_use >> 20u, blocks);
}

int AllocateDirectMemory(int64_t search_start, int64_t search_end, size_t len, size_t alignment,
                         int memory_type, int64_t* phys_addr_out, bool automatic) {

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	if (len != 0) {
		g_zero_length_request.count = 0;
	}
	const bool quiet = DirectMemoryQuiet();
	if (!quiet && (len != 0 || g_zero_length_request.count < 4 ||
	               std::has_single_bit(g_zero_length_request.count))) {
		LOGF("\t search_start = 0x%016" PRIx64 "\n"
		     "\t search_end   = 0x%016" PRIx64 "\n"
		     "\t len          = 0x%016" PRIx64 "\n"
		     "\t alignment    = 0x%016" PRIx64 "\n"
		     "\t memory_type  = %d\n",
		     search_start, search_end, len, alignment, memory_type);
	}

	constexpr uint64_t PAGE_SIZE = 0x4000;
	if (search_start < 0 || search_end <= search_start || len == 0 ||
	    (len & (PAGE_SIZE - 1u)) != 0 || (alignment != 0 && (alignment & (PAGE_SIZE - 1u)) != 0) ||
	    phys_addr_out == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	uint64_t addr = 0;
	const auto kind = automatic ? PhysicalMemory::AllocationKind::Automatic
	                            : PhysicalMemory::AllocationKind::Direct;
	if (!g_physical_memory->Alloc(search_start, search_end, len, alignment, &addr, memory_type,
	                              kind)) {
		if (!quiet) {
			LOGF_COLOR(Log::Color::Red, "\t[Fail]\n");
		}
		ReportDirectMemoryFailure(len, alignment, memory_type);
		return KERNEL_ERROR_EAGAIN;
	}

	*phys_addr_out = static_cast<int64_t>(addr);

	if (!quiet) {
		LOGF_COLOR(Log::Color::Green, "\tphys_addr    = %016" PRIx64 "\n\t[Ok]\n", addr);
	}
	NoteBigDirectMemoryChange(true, addr, len, memory_type, t_direct_memory_caller);

	return OK;
}

int KYTY_SYSV_ABI KernelAllocateDirectMemory(int64_t search_start, int64_t search_end, size_t len,
                                             size_t alignment, int memory_type,
                                             int64_t* phys_addr_out) {
	PRINT_NAME();
	if (len == 0) {
		NoteZeroLengthRequest(search_start, search_end, alignment, memory_type,
		                      reinterpret_cast<uint64_t>(__builtin_return_address(0)));
	}
	// The caller of KernelAllocateMainDirectMemory when it is the one calling.
	const bool outer = t_direct_memory_caller == 0;
	if (outer) {
		t_direct_memory_caller = reinterpret_cast<uint64_t>(__builtin_return_address(0));
	}
	const auto result =
	    AllocateDirectMemory(search_start, search_end, len, alignment, memory_type, phys_addr_out);
	if (outer) {
		t_direct_memory_caller = 0;
	}
	return result;
}

int KYTY_SYSV_ABI KernelAllocateMainDirectMemory(size_t len, size_t alignment, int memory_type,
                                                 int64_t* phys_addr_out) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t len          = 0x%016" PRIx64 "\n"
	     "\t alignment    = 0x%016" PRIx64 "\n"
	     "\t memory_type  = %d\n",
	     len, alignment, memory_type);

	t_direct_memory_caller = reinterpret_cast<uint64_t>(__builtin_return_address(0));
	const int result = KernelAllocateDirectMemory(0, static_cast<int64_t>(PhysicalMemory::Size()),
	                                              len, alignment, memory_type, phys_addr_out);
	t_direct_memory_caller = 0;
	return result;
}

// `not_one_span`: set when the range is not one allocated span (nothing was released).
static int ReleaseDirectMemoryInternal(int64_t start, size_t len, bool* not_one_span = nullptr) {
	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	if (g_pooled_memory->ReleaseExpansion(static_cast<uint64_t>(start), len)) {
		if (!g_physical_memory->ReleasePoolExpansion(static_cast<uint64_t>(start), len)) {
			EXIT("failed to release physical pool expansion\n");
		}
		return OK;
	}

	std::vector<PhysicalMemory::AllocatedBlock> allocated_span;
	if (!g_physical_memory->GetAllocatedSpan(static_cast<uint64_t>(start), len, &allocated_span)) {
		if (not_one_span != nullptr) {
			*not_one_span = true;
		}
		return KERNEL_ERROR_EACCES;
	}

	const auto mapped_aliases = g_physical_memory->FindMappings(start, len);
	for (const auto& alias: mapped_aliases) {
		VirtualRanges::Range range {};
		if (!g_guest_address_space->BackingContains(alias.map_vaddr, alias.map_size) ||
		    !g_virtual_ranges->Query(alias.map_vaddr, 0, &range) ||
		    range.type != VirtualRangeType::Direct ||
		    alias.map_size > range.start + range.size - alias.map_vaddr) {
			EXIT("direct-memory alias escaped guest address-space ownership\n");
		}
	}

	for (const auto& alias: mapped_aliases) {
		UnmapGpuRange(alias.map_vaddr, alias.map_size);
	}

	auto restore_gpu_aliases = [&mapped_aliases]() {
		for (const auto& alias: mapped_aliases) {
			MapGpuRange(alias.map_vaddr, alias.map_size);
		}
	};
	auto restore_owner_aliases = [](const std::vector<PhysicalMemory::AllocatedBlock>& aliases) {
		for (auto it = aliases.rbegin(); it != aliases.rend(); ++it) {
			EXIT_IF(!g_guest_address_space->MapBacking(it->map_vaddr, it->map_size, it->start_addr,
			                                           it->mode));
		}
	};

	std::vector<PhysicalMemory::AllocatedBlock> owner_unmapped;
	for (const auto& alias: mapped_aliases) {
		if (!g_guest_address_space->UnmapBacking(alias.map_vaddr, alias.map_size)) {
			restore_owner_aliases(owner_unmapped);
			restore_gpu_aliases();
			return KERNEL_ERROR_EACCES;
		}
		owner_unmapped.push_back(alias);
	}

	std::vector<PhysicalMemory::AllocatedBlock> metadata_unmapped;
	for (const auto& alias: mapped_aliases) {
		GpuAccessMode alias_gpu_mode = GpuAccessMode::NoAccess;
		if (!g_physical_memory->Unmap(alias.map_vaddr, alias.map_size, &alias_gpu_mode)) {
			for (const auto& removed: metadata_unmapped) {
				EXIT_IF(!g_physical_memory->Map(removed.map_vaddr, removed.start_addr,
				                                removed.map_size, removed.prot, removed.mode,
				                                removed.gpu_mode));
			}
			restore_owner_aliases(owner_unmapped);
			restore_gpu_aliases();
			return KERNEL_ERROR_EACCES;
		}
		metadata_unmapped.push_back(alias);
	}

	for (const auto& alias: mapped_aliases) {
		EXIT_IF(!g_virtual_ranges->Remove(alias.map_vaddr, alias.map_size));
	}

	for (const auto& block: allocated_span) {
		uint64_t      unused_vaddr = 0;
		uint64_t      unused_size  = 0;
		GpuAccessMode unused_gpu   = GpuAccessMode::NoAccess;
		EXIT_IF(!g_physical_memory->Release(block.start_addr, block.size, &unused_vaddr,
		                                    &unused_size, &unused_gpu));
	}

	if (g_free_callback != nullptr) {
		for (const auto& alias: mapped_aliases) {
			g_free_callback(alias.map_vaddr, alias.map_size);
		}
	}

	return OK;
}

static int ValidateDirectReleaseRange(int64_t start, size_t len) {
	constexpr uint64_t PAGE_SIZE = 0x4000;
	return start < 0 || (static_cast<uint64_t>(start) & (PAGE_SIZE - 1u)) != 0 ||
	               (len & (PAGE_SIZE - 1u)) != 0
	           ? KERNEL_ERROR_EINVAL
	           : OK;
}

// The allocated parts of [start, start + len) (pooled ones left out), as runs of adjacent blocks.
static std::vector<std::pair<uint64_t, uint64_t>> AllocatedRuns(uint64_t start, uint64_t len) {
	std::vector<std::pair<uint64_t, uint64_t>> runs;
	if (len == 0 || len > UINT64_MAX - start) {
		return runs;
	}
	const auto        end = start + len;
	Common::LockGuard lock(g_physical_memory->GetMutex());
	const auto&       blocks = g_physical_memory->GetPhysicalBlocks();
	auto              block  = blocks.upper_bound(start);
	if (block != blocks.begin()) {
		--block;
	}
	for (; block != blocks.end() && block->second.start_addr < end; ++block) {
		const auto& b     = block->second;
		const auto  b_end = b.start_addr + b.size;
		if (b_end <= start || b.kind == PhysicalMemory::AllocationKind::Pooled) {
			continue;
		}
		const auto part_start = std::max(start, b.start_addr);
		const auto part_end   = std::min(end, b_end);
		if (!runs.empty() && runs.back().first + runs.back().second == part_start) {
			runs.back().second += part_end - part_start;
		} else {
			runs.emplace_back(part_start, part_end - part_start);
		}
	}
	return runs;
}

int KYTY_SYSV_ABI KernelReleaseDirectMemory(int64_t start, size_t len) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);
	if (!DirectMemoryQuiet()) {
		LOGF("\t start = 0x%016" PRIx64 "\n"
		     "\t len   = 0x%016" PRIx64 "\n",
		     start, len);
	}

	const int validation = ValidateDirectReleaseRange(start, len);
	if (validation != OK) {
		return validation;
	}
	bool       not_one_span = false;
	const auto caller       = reinterpret_cast<uint64_t>(__builtin_return_address(0));
	const int  released = len != 0 ? ReleaseDirectMemoryInternal(start, len, &not_one_span) : OK;
	if (len != 0 && released == OK) {
		NoteBigDirectMemoryChange(false, static_cast<uint64_t>(start), len, 0, caller);
	}
	if (len != 0 && released != OK && !not_one_span) {
		// Round 82: the release failed (its mappings could not be undone): the guest is told it
		// worked, as before, but the memory stays allocated, so it is logged.
		static std::atomic<uint32_t> failures {0};
		const auto                   failure = failures.fetch_add(1, std::memory_order_relaxed);
		if (failure < 16u || std::has_single_bit(failure + 1u)) {
			LOGF("ReleaseDirectMemory: 0x%016" PRIx64 "+0x%" PRIx64 " was not released (error "
			     "0x%08x), from 0x%016" PRIx64 ": it stays allocated (%u such releases so far)\n",
			     start, static_cast<uint64_t>(len), static_cast<uint32_t>(released), caller,
			     failure + 1u);
		}
	}
	if (len != 0 && released != OK && not_one_span) {
		// Not one allocated span (a gap, or a part already released): the unchecked release
		// frees the allocated parts of the range and passes over the rest, where this used to
		// free nothing at all and keep the whole range allocated.
		uint64_t freed = 0;
		uint32_t parts = 0;
		for (const auto& [part_start, part_size]: AllocatedRuns(start, len)) {
			if (ReleaseDirectMemoryInternal(static_cast<int64_t>(part_start), part_size) == OK) {
				freed += part_size;
				parts++;
			}
		}
		NoteBigDirectMemoryChange(false, static_cast<uint64_t>(start), freed, 0, caller);
		auto& pressure = g_direct_memory_pressure;
		pressure.partial_releases++;
		pressure.partial_freed += freed;
		if (pressure.partial_releases <= 32 || std::has_single_bit(pressure.partial_releases)) {
			LOGF("ReleaseDirectMemory: 0x%016" PRIx64 "+0x%" PRIx64 " is not one allocated span: "
			     "its %u allocated parts were released (0x%" PRIx64 " bytes; %" PRIu64
			     " such releases, %" PRIu64 " MiB, so far)\n",
			     start, static_cast<uint64_t>(len), parts, freed, pressure.partial_releases,
			     pressure.partial_freed >> 20u);
		}
	}
	return OK;
}

int KYTY_SYSV_ABI KernelCheckedReleaseDirectMemory(int64_t start, size_t len) {
	PRINT_NAME();

	LOGF("\t start = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n",
	     start, len);

	const int validation = ValidateDirectReleaseRange(start, len);
	if (validation != OK || len == 0) {
		return validation;
	}

	const int result = ReleaseDirectMemoryInternal(start, len);
	if (result == OK) {
		NoteBigDirectMemoryChange(false, static_cast<uint64_t>(start), len, 0,
		                          reinterpret_cast<uint64_t>(__builtin_return_address(0)));
	}
	return result == KERNEL_ERROR_EACCES ? KERNEL_ERROR_ENOENT : result;
}

int KYTY_SYSV_ABI KernelMapDirectMemory(void** addr, size_t len, int prot, int flags,
                                        int64_t direct_memory_start, size_t alignment) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	if (addr == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}
	constexpr uint64_t PAGE_SIZE              = 0x4000;
	constexpr int      GUEST_MAP_FIXED        = 0x10;
	constexpr int      GUEST_MAP_NO_OVERWRITE = 0x80;

	if (len == 0 || (len & (PAGE_SIZE - 1u)) != 0 || direct_memory_start < 0 ||
	    (static_cast<uint64_t>(direct_memory_start) & (PAGE_SIZE - 1u)) != 0 ||
	    (alignment != 0 && (alignment & (alignment - 1u)) != 0 &&
	     (alignment & (PAGE_SIZE - 1u)) != 0)) {
		return KERNEL_ERROR_EINVAL;
	}
	if ((prot & PROT_CPU_EXEC) != 0) {
		return KERNEL_ERROR_EACCES;
	}

	bool fixed        = ((flags & GUEST_MAP_FIXED) != 0);
	bool no_overwrite = ((flags & GUEST_MAP_NO_OVERWRITE) != 0);

	VirtualMemory::Mode mode     = VirtualMemory::Mode::NoAccess;
	GpuAccessMode       gpu_mode = GpuAccessMode::NoAccess;

	if (!DecodeMemoryProtection(prot, &mode, &gpu_mode)) {
		return KERNEL_ERROR_EINVAL;
	}
	if (!g_physical_memory->CanMapDirect(static_cast<uint64_t>(direct_memory_start), len)) {
		return KERNEL_ERROR_ENOMEM;
	}

	auto     in_addr         = reinterpret_cast<uint64_t>(*addr);
	uint64_t out_addr        = 0;
	bool     shared_backing  = false;
	bool     reserved_target = false;
	auto     shared_failure  = GuestBackingStore::FailureReason::None;
	// Direct mappings must remain views of the single backing object. Anonymous fallbacks break
	// aliasing and lose direct-memory contents when a range is unmapped and mapped again.
	auto map_shared_fixed = [&](uint64_t target_addr) -> bool {
		return g_guest_address_space->MapBacking(target_addr, len, direct_memory_start, mode,
		                                         &shared_failure);
	};
	std::vector<PhysicalMemory::AllocatedBlock> replaced_mappings;
	auto restore_mappings = [&](size_t count, bool restore_views) {
		for (size_t index = 0; index < count; ++index) {
			const auto& old = replaced_mappings[index];
			if (restore_views) {
				EXIT_IF(!g_guest_address_space->MapBacking(old.map_vaddr, old.map_size,
				                                           old.start_addr, old.mode));
			}
			EXIT_IF(!g_physical_memory->Map(old.map_vaddr, old.start_addr, old.map_size,
			                               old.prot, old.mode, old.gpu_mode));
			g_physical_memory->SetVirtualRangeName(old.map_vaddr, old.map_size, old.name);
			g_physical_memory->SetVirtualRangeMemoryType(old.map_vaddr, old.map_size,
			                                          old.memory_type);
		}
		MapGpuRange(in_addr, len);
	};
	if (fixed) {
		if (in_addr == 0 || (in_addr & (PAGE_SIZE - 1u)) != 0 ||
		    (alignment != 0 && in_addr % alignment != 0)) {
			return KERNEL_ERROR_EINVAL;
		}
		if (no_overwrite && g_virtual_ranges->HasOverlap(in_addr, len)) {
			return KERNEL_ERROR_ENOMEM;
		}

		std::vector<VirtualRanges::Range> reserved_ranges;
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		if (g_virtual_ranges->QuerySpan(in_addr, len, &reserved_ranges) &&
		    std::all_of(reserved_ranges.begin(), reserved_ranges.end(), [](const auto& range) {
			    return range.type == VirtualRangeType::Direct;
		    })) {
			uint64_t covered = 0;
			{
				Common::LockGuard lock(g_physical_memory->GetMutex());
				for (const auto& mapping: g_physical_memory->GetMappings()) {
					const auto start = std::max(in_addr, mapping.map_vaddr);
					const auto end = std::min(in_addr + len, mapping.map_vaddr + mapping.map_size);
					if (start >= end) {
						continue;
					}
					auto old = mapping;
					old.start_addr += start - old.map_vaddr;
					old.map_vaddr = start;
					old.size = old.map_size = end - start;
					covered += old.map_size;
					replaced_mappings.push_back(old);
				}
			}
			if (covered != len) {
				return KERNEL_ERROR_EBUSY;
			}
			UnmapGpuRange(in_addr, len);
			for (size_t index = 0; index < replaced_mappings.size(); ++index) {
				const auto& old = replaced_mappings[index];
				GpuAccessMode old_gpu_mode = GpuAccessMode::NoAccess;
				if (!g_physical_memory->Unmap(old.map_vaddr, old.map_size, &old_gpu_mode)) {
					restore_mappings(index, false);
					return KERNEL_ERROR_EBUSY;
				}
			}
			// Keep the old view accessible until mmap atomically replaces it. In particular,
			// do not publish a temporary reserved/PROT_NONE range between two direct views.
			if (!map_shared_fixed(in_addr)) {
				restore_mappings(replaced_mappings.size(), true);
				return KERNEL_ERROR_ENOMEM;
			}
			out_addr       = in_addr;
			shared_backing = true;
		} else
#endif
		{
			if (g_virtual_ranges->QuerySpan(in_addr, len, &reserved_ranges) &&
			    std::all_of(reserved_ranges.begin(), reserved_ranges.end(), [](const auto& range) {
				    return range.type == VirtualRangeType::Reserved;
			    })) {
				UnmapGpuRange(in_addr, len);
				reserved_target = true;
			}
			if (!reserved_target && ReplaceFixedRangeWithReserved(in_addr, len)) {
				reserved_target = true;
			}
			if (!reserved_target) {
				return KERNEL_ERROR_ENOMEM;
			}
			if (map_shared_fixed(in_addr)) {
				out_addr       = in_addr;
				shared_backing = true;
			}
		}
	} else {
		constexpr size_t DEFAULT_ALIGNMENT = 0x4000;
		alignment                          = (alignment != 0 ? alignment : DEFAULT_ALIGNMENT);
		std::vector<VirtualRanges::Range> reserved_ranges;
		if (in_addr != 0 && g_virtual_ranges->QuerySpan(in_addr, len, &reserved_ranges) &&
		    std::all_of(reserved_ranges.begin(), reserved_ranges.end(), [](const auto& range) {
			    return range.type == VirtualRangeType::Reserved;
		    })) {
			UnmapGpuRange(in_addr, len);
			reserved_target = true;
			if (map_shared_fixed(in_addr)) {
				out_addr       = in_addr;
				shared_backing = true;
			}
		}
		if (!reserved_target) {
			out_addr = FindGuestFreeRange(in_addr, len, alignment);
			if (out_addr != 0) {
				UnmapGpuRange(out_addr, len);
				shared_backing = map_shared_fixed(out_addr);
				if (!shared_backing) {
					out_addr = 0;
				}
			}
		}
	}

	*addr = reinterpret_cast<void*>(out_addr);

	const char* shared_reason = "n/a";
	if (!shared_backing) {
		shared_reason = GuestBackingStore::GetFailureReasonName(shared_failure);
	}

	LOGF("\t in_addr  = 0x%016" PRIx64 "\n"
	     "\t out_addr = 0x%016" PRIx64 "\n"
	     "\t dmem     = 0x%016" PRIx64 "\n"
	     "\t size     = 0x%016" PRIx64 "\n"
	     "\t mode     = %s\n"
	     "\t flags    = 0x%08" PRIx32 "\n"
	     "\t align    = 0x%016" PRIx64 "\n"
	     "\t gpu_mode = %s\n"
	     "\t shared   = %s\n"
	     "\t reason   = %s\n",
	     in_addr, out_addr, static_cast<uint64_t>(direct_memory_start), len,
	     magic_enum::enum_name(mode), static_cast<uint32_t>(flags), alignment,
	     magic_enum::enum_name(gpu_mode), shared_backing ? "yes" : "no", shared_reason);

	if (out_addr == 0) {
		return KERNEL_ERROR_ENOMEM;
	}

	if (!g_physical_memory->Map(out_addr, direct_memory_start, len, prot, mode, gpu_mode)) {
		LOGF_COLOR(Log::Color::Red, "\t [Fail]\n");
		if (replaced_mappings.empty()) {
			EXIT_IF(!g_guest_address_space->UnmapBacking(out_addr, len));
		} else {
			restore_mappings(replaced_mappings.size(), true);
		}
		return KERNEL_ERROR_EBUSY;
	}

	PhysicalMemory::AllocatedBlock mapped_block {};
	g_physical_memory->Find(direct_memory_start, false, &mapped_block);
	const bool published = !replaced_mappings.empty() || reserved_target
	                           ? g_virtual_ranges->ReplaceSpan(
	                                 out_addr, len, replaced_mappings.empty()
	                                                    ? VirtualRangeType::Reserved
	                                                    : VirtualRangeType::Direct,
	                                 direct_memory_start,
	                                 prot, mapped_block.memory_type, VirtualRangeType::Direct, "")
	                           : g_virtual_ranges->Add(out_addr, len, direct_memory_start, prot,
	                                                   mapped_block.memory_type,
	                                                   VirtualRangeType::Direct, "");
	if (!published) {
		GpuAccessMode rollback_gpu_mode = GpuAccessMode::NoAccess;
		EXIT_IF(!g_physical_memory->Unmap(out_addr, len, &rollback_gpu_mode));
		if (replaced_mappings.empty()) {
			EXIT_IF(!g_guest_address_space->UnmapBacking(out_addr, len));
		} else {
			restore_mappings(replaced_mappings.size(), true);
		}
		return KERNEL_ERROR_EBUSY;
	}

	MapGpuRange(out_addr, len);

	if (g_alloc_callback != nullptr) {
		g_alloc_callback(out_addr, len);
	}

	LOGF_COLOR(Log::Color::Green, "\t [Ok]\n");

	return OK;
}

int KYTY_SYSV_ABI KernelMapDirectMemory2(void** addr, size_t len, int type, int prot, int flags,
                                         int64_t direct_memory_start, size_t alignment) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t type = %d\n", type);

	auto ret = KernelMapDirectMemory(addr, len, prot, flags, direct_memory_start, alignment);
	if (ret == OK && addr != nullptr && *addr != nullptr) {
		const auto out_addr = reinterpret_cast<uint64_t>(*addr);
		g_physical_memory->SetVirtualRangeMemoryType(out_addr, len, type);
		g_virtual_ranges->SetMemoryType(out_addr, len, type);
	}

	return ret;
}

int MapAutomaticMemory(uint64_t vaddr, size_t size, int type, int prot) {
	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);
	constexpr uint64_t                    page_size = 0x4000;
	if (size == 0 || (vaddr & (page_size - 1)) != 0 || (size & (page_size - 1)) != 0 ||
	    vaddr < kExtendedMemoryBase || vaddr >= kExtendedMemoryBase + kExtendedMemorySize ||
	    size > kExtendedMemoryBase + kExtendedMemorySize - vaddr) {
		return KERNEL_ERROR_EINVAL;
	}

	PhysicalMemory::PhysicalRanges ranges;
	if (!g_physical_memory->ReserveAutomatic(size, &ranges)) {
		return KERNEL_ERROR_EAGAIN;
	}
	uint64_t mapped = 0;
	for (const auto& [offset, length]: ranges) {
		void*     address = reinterpret_cast<void*>(vaddr + mapped);
		const int result  = KernelMapDirectMemory2(&address, length, type, prot, 0x10,
		                                           static_cast<int64_t>(offset), page_size);
		if (result != OK) {
			if (mapped != 0) {
				EXIT_IF(KernelMunmap(vaddr, mapped) != OK);
			}
			g_physical_memory->RestoreAutomatic(ranges);
			return result;
		}
		mapped += length;
	}
	return OK;
}

int KYTY_SYSV_ABI KernelMapNamedDirectMemory(void** addr, size_t len, int prot, int flags,
                                             int64_t direct_memory_start, size_t alignment,
                                             const char* name) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t name = %s\n", name != nullptr ? name : "(null)");

	if (name == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (std::strlen(name) >= KERNEL_MAXIMUM_NAME_LENGTH) {
		return KERNEL_ERROR_ENAMETOOLONG;
	}

	auto ret = KernelMapDirectMemory(addr, len, prot, flags, direct_memory_start, alignment);
	if (ret == OK && addr != nullptr) {
		g_physical_memory->SetVirtualRangeName(reinterpret_cast<uint64_t>(*addr), len, name);
		g_virtual_ranges->Rename(reinterpret_cast<uint64_t>(*addr), len, name);
	}

	return ret;
}

int KYTY_SYSV_ABI KernelIsAddressSanitizerEnabled() {
	PRINT_NAME();

	return 0;
}

int KYTY_SYSV_ABI KernelQueryMemoryProtection(void* addr, void** start, void** end, int* prot) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	EXIT_NOT_IMPLEMENTED(addr == nullptr);

	VirtualRanges::Range range {};
	if (!g_virtual_ranges->Query(reinterpret_cast<uint64_t>(addr), 0, &range)) {
		return KERNEL_ERROR_EACCES;
	}

	if (start != nullptr) {
		*start = reinterpret_cast<void*>(range.start);
	}
	if (end != nullptr) {
		*end = reinterpret_cast<void*>(range.start + range.size);
	}
	if (prot != nullptr) {
		*prot = range.protection;
	}

	return OK;
}

static bool ReplaceFixedRangeWithReserved(uint64_t start, uint64_t size) {

	struct ReplacedChunk {
		VirtualRanges::Range                        range {};
		VirtualMemory::Mode                         mode     = VirtualMemory::Mode::NoAccess;
		GpuAccessMode                               gpu_mode = GpuAccessMode::NoAccess;
		std::vector<FlexibleMemory::AllocatedBlock> flexible_blocks;
		bool                                        host_unmapped    = false;
		bool                                        backend_unmapped = false;
	};

	std::vector<ReplacedChunk> chunks;
	const auto                 end     = start + size;
	auto                       current = start;

	while (current < end) {
		VirtualRanges::Range range {};
		if (!g_virtual_ranges->Query(current, 1, &range)) {
			break;
		}
		if (range.start >= end) {
			break;
		}
		if (current < range.start) {
			current = std::min<uint64_t>(end, range.start);
			continue;
		}

		const auto range_end = range.start + range.size;
		const auto chunk     = std::min<uint64_t>(end, range_end) - current;

		ReplacedChunk replaced {};
		replaced.range       = range;
		replaced.range.start = current;
		replaced.range.size  = chunk;
		if (range.type == VirtualRangeType::Direct) {
			replaced.range.offset += current - range.start;
		}
		DecodeMemoryProtection(replaced.range.protection, &replaced.mode, &replaced.gpu_mode);
		if (range.type == VirtualRangeType::Direct &&
		    !g_guest_address_space->BackingContains(current, chunk)) {
			return false;
		}
		if (range.type == VirtualRangeType::Flexible &&
		    !g_flexible_memory->Snapshot(current, chunk, &replaced.flexible_blocks)) {
			return false;
		}
		if (range.type == VirtualRangeType::Pooled) {
			EXIT("reserve-fixed replacement of pooled memory is unsupported: addr=0x%016" PRIx64
			     " size=0x%016" PRIx64 "\n",
			     replaced.range.start, replaced.range.size);
		}
		chunks.push_back(replaced);

		current += chunk;
	}

	auto restore_chunks = [&chunks]() -> bool {
		bool ok = true;
		for (auto it = chunks.rbegin(); it != chunks.rend(); ++it) {
			const auto& chunk            = *it;
			bool        host_restored    = true;
			bool        backend_restored = true;

			if (chunk.range.type == VirtualRangeType::Direct) {
				if (chunk.host_unmapped) {
					host_restored = g_guest_address_space->MapBacking(
					    chunk.range.start, chunk.range.size, chunk.range.offset, chunk.mode);
				}
				if (chunk.backend_unmapped) {
					backend_restored = g_physical_memory->Map(
					    chunk.range.start, chunk.range.offset, chunk.range.size,
					    chunk.range.protection, chunk.mode, chunk.gpu_mode);
				}
			} else if (chunk.range.type == VirtualRangeType::Flexible && chunk.backend_unmapped) {
				host_restored = backend_restored =
				    g_flexible_memory->Restore(chunk.flexible_blocks);
			} else if (IsPrivateCommittedRangeType(chunk.range.type) && chunk.host_unmapped) {
				host_restored =
				    g_guest_address_space->Commit(chunk.range.start, chunk.range.size, chunk.mode);
			}
			ok = host_restored && backend_restored && ok;
		}

		for (const auto& chunk: chunks) {
			const bool range_restored = g_virtual_ranges->Add(
			    chunk.range.start, chunk.range.size, chunk.range.offset, chunk.range.protection,
			    chunk.range.memory_type, chunk.range.type, chunk.range.name,
			    chunk.range.disallow_merge);
			ok = range_restored && ok;
		}
		if (ok) {
			for (const auto& chunk: chunks) {
				if (IsCommittedRangeType(chunk.range.type)) {
					MapGpuRange(chunk.range.start, chunk.range.size);
				}
			}
		}
		return ok;
	};

	UnmapGpuRange(start, size);
	g_virtual_ranges->Remove(start, size);

	for (auto& chunk: chunks) {
		GpuAccessMode gpu_mode = GpuAccessMode::NoAccess;
		bool          unmapped = true;

		if (chunk.range.type == VirtualRangeType::Direct) {
			if (!g_guest_address_space->UnmapBacking(chunk.range.start, chunk.range.size)) {
				unmapped = false;
			} else {
				chunk.host_unmapped = true;
			}
			if (unmapped) {
				unmapped = g_physical_memory->Unmap(chunk.range.start, chunk.range.size, &gpu_mode);
				chunk.gpu_mode         = gpu_mode;
				chunk.backend_unmapped = unmapped;
			}
		} else if (chunk.range.type == VirtualRangeType::Flexible) {
			unmapped = g_flexible_memory->Unmap(chunk.range.start, chunk.range.size, &gpu_mode);
			chunk.host_unmapped    = unmapped;
			chunk.backend_unmapped = unmapped;
			chunk.gpu_mode         = gpu_mode;
		} else if (IsPrivateCommittedRangeType(chunk.range.type)) {
			unmapped = g_guest_address_space->ReleaseCommitted(chunk.range.start, chunk.range.size);
			chunk.host_unmapped    = unmapped;
			chunk.backend_unmapped = unmapped;
		} else if (IsReservedRangeType(chunk.range.type)) {
			unmapped = g_guest_address_space->ReleaseFree(chunk.range.start, chunk.range.size);
		} else {
			unmapped = false;
		}
		if (chunk.range.type == VirtualRangeType::Direct) {
			chunk.gpu_mode         = gpu_mode;
			chunk.backend_unmapped = chunk.backend_unmapped && unmapped;
		}

		if (!unmapped) {
			LOGF_COLOR(Log::Color::Red,
			           "\t reserve-fixed replace: backend unmap failed at 0x%016" PRIx64
			           ", size=0x%016" PRIx64 ", type=%s\n",
			           chunk.range.start, chunk.range.size,
			           magic_enum::enum_name(chunk.range.type));
			if (!restore_chunks()) {
				EXIT("reserve-fixed backend-unmap rollback failed\n");
			}
			return false;
		}
	}

	if (!g_guest_address_space->ReserveFixed(start, size)) {
		if (!restore_chunks()) {
			EXIT("reserve-fixed host-reservation rollback failed\n");
		}
		return false;
	}

	bool range_added = false;
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	if (g_test_fail_next_fixed_reserve_range_add) {
		g_test_fail_next_fixed_reserve_range_add = false;
	} else
#endif
	{
		range_added =
		    g_virtual_ranges->Add(start, size, 0, 0, 0, VirtualRangeType::Reserved, "anon");
	}
	if (!range_added) {
		LOGF_COLOR(Log::Color::Red,
		           "\t reserve-fixed replace: range add failed at 0x%016" PRIx64
		           ", size=0x%016" PRIx64 "\n",
		           start, size);
		if (!restore_chunks()) {
			EXIT("reserve-fixed range-registration rollback failed\n");
		}
		auto free_start = start;
		for (const auto& chunk: chunks) {
			if (free_start < chunk.range.start &&
			    !g_guest_address_space->ReleaseFree(free_start, chunk.range.start - free_start)) {
				EXIT("reserve-fixed range-registration gap cleanup failed\n");
			}
			free_start = chunk.range.start + chunk.range.size;
		}
		if (free_start < start + size &&
		    !g_guest_address_space->ReleaseFree(free_start, start + size - free_start)) {
			EXIT("reserve-fixed range-registration tail cleanup failed\n");
		}
		return false;
	}

	for (const auto& chunk: chunks) {
		if (g_free_callback != nullptr && IsCommittedRangeType(chunk.range.type)) {
			g_free_callback(chunk.range.start, chunk.range.size);
		}
	}

	return true;
}

int KYTY_SYSV_ABI KernelReserveVirtualRange(void** addr, size_t len, int flags, size_t alignment) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	const auto in_addr = (addr != nullptr ? reinterpret_cast<uint64_t>(*addr) : 0);

	LOGF("\t in_addr   = 0x%016" PRIx64 "\n"
	     "\t len       = 0x%016" PRIx64 "\n"
	     "\t flags     = 0x%08" PRIx32 "\n"
	     "\t alignment = 0x%016" PRIx64 "\n",
	     in_addr, len, flags, alignment);

	constexpr size_t PAGE_SIZE              = 0x4000;
	constexpr int    GUEST_MAP_FIXED        = 0x10;
	constexpr int    GUEST_MAP_NO_OVERWRITE = 0x80;

	if (addr == nullptr || len == 0 || (len & (PAGE_SIZE - 1)) != 0) {
		return KERNEL_ERROR_EINVAL;
	}
	if (alignment != 0 && (alignment & (alignment - 1)) != 0 &&
	    (alignment & (PAGE_SIZE - 1)) != 0) {
		return KERNEL_ERROR_EINVAL;
	}

	uint64_t out_addr            = 0;
	bool     range_already_added = false;
	if ((flags & GUEST_MAP_FIXED) != 0) {
		if (in_addr == 0 || (in_addr & (PAGE_SIZE - 1)) != 0) {
			return KERNEL_ERROR_EINVAL;
		}
		if ((flags & GUEST_MAP_NO_OVERWRITE) != 0 && g_virtual_ranges->HasOverlap(in_addr, len)) {
			return KERNEL_ERROR_ENOMEM;
		}
		if (ReplaceFixedRangeWithReserved(in_addr, len)) {
			out_addr            = in_addr;
			range_already_added = true;
		}
	} else {
		alignment = (alignment != 0 ? alignment : PAGE_SIZE);
		out_addr  = FindGuestFreeRange(in_addr, len, alignment);
		if (out_addr != 0) {
			UnmapGpuRange(out_addr, len);
		}
	}

	if (out_addr == 0) {
		return KERNEL_ERROR_ENOMEM;
	}

	if (!range_already_added &&
	    !g_virtual_ranges->Add(out_addr, len, 0, 0, 0, VirtualRangeType::Reserved, "anon")) {
		return KERNEL_ERROR_EBUSY;
	}

	*addr = reinterpret_cast<void*>(out_addr);

	LOGF("\t out_addr  = 0x%016" PRIx64 "\n", out_addr);

	return OK;
}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
void TestBeforeNextBackingMap(callback_func_t callback) {
	g_test_before_backing_map = callback;
}

void TestFailNextPhysicalMemoryUnmap() {
	TestFailPhysicalMemoryUnmapAfter(0);
}

void TestFailPhysicalMemoryUnmapAfter(uint32_t successful_unmaps) {
	g_test_physical_memory_unmaps_before_failure = successful_unmaps;
}

void TestFailGuestBackingStoreUnmapAfter(uint32_t successful_unmaps) {
	g_test_backing_store_unmaps_before_failure = successful_unmaps;
}

void TestFailNextFixedReserveRangeRegistration() {
	g_test_fail_next_fixed_reserve_range_add = true;
}

void TestFailNextVirtualRangeReplacement() {
	g_test_fail_next_range_replace = true;
}

bool TestPlaceholderRangeIsFree(uint64_t vaddr, uint64_t size) {
	return g_guest_address_space->TestContainsFree(vaddr, size);
}

bool TestGuestAddressRangeIsOwned(uint64_t vaddr, uint64_t size) {
	return g_guest_address_space->Owns(vaddr, size);
}

bool TestGuestBackingOutsideAddressSpace() {
	return !g_guest_address_space->OverlapsOwned(g_guest_address_space->GetBackingBase(),
	                                             g_guest_address_space->GetBackingSize());
}

uint64_t TestGuestBackingSize() {
	return g_guest_address_space->GetBackingSize();
}

uint64_t TestDirectMemoryOverflowSize() {
	return DirectMemoryOverflowSize();
}

bool TestGuestFreeRangeBounds() {
	return GuestFreeRangeContains(0x10000, 0x20000, 0x18000, 0x4000) &&
	       !GuestFreeRangeContains(0x10000, 0x20000, 0x40000, 0x4000) &&
	       !GuestFreeRangeContains(UINT64_MAX - 0x1000, 0x2000, UINT64_MAX - 0x800, 0x400);
}
#endif

int KYTY_SYSV_ABI KernelVirtualQuery(const void* addr, int flags, VirtualQueryInfo* info,
                                     uint64_t info_size) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	auto vaddr = reinterpret_cast<uint64_t>(addr);

	LOGF("\t addr      = 0x%016" PRIx64 "\n"
	     "\t flags     = 0x%08" PRIx32 "\n"
	     "\t info_size = 0x%016" PRIx64 "\n",
	     vaddr, flags, info_size);

	if (info == nullptr || info_size != sizeof(VirtualQueryInfo) || (flags != 0 && flags != 1)) {
		return KERNEL_ERROR_EINVAL;
	}

	VirtualRanges::Range candidate {};
	if (!g_virtual_ranges->Query(vaddr, flags, &candidate)) {
		return KERNEL_ERROR_EACCES;
	}

	std::memset(info, 0, sizeof(VirtualQueryInfo));
	info->start        = candidate.start;
	info->end          = candidate.start + candidate.size;
	info->offset       = candidate.offset;
	info->protection   = candidate.protection;
	info->memory_type  = candidate.memory_type;
	info->is_flexible  = (candidate.type == VirtualRangeType::Flexible ? 1 : 0);
	info->is_direct    = (candidate.type == VirtualRangeType::Direct ? 1 : 0);
	info->is_stack     = (candidate.type == VirtualRangeType::Stack ? 1 : 0);
	info->is_pooled    = (IsPooledRangeType(candidate.type) ? 1 : 0);
	info->is_committed = (IsCommittedRangeType(candidate.type) ? 1 : 0);
	info->is_gpu_prt   = IsInPrtAperture(vaddr);
	CopyVirtualRangeName(info->name, candidate.name);

	static std::atomic<uint32_t> log_count {0};
	if (log_count.fetch_add(1) < 64) {
		LOGF("\t start       = 0x%016" PRIx64 "\n"
		     "\t end         = 0x%016" PRIx64 "\n"
		     "\t offset      = 0x%016" PRIx64 "\n"
		     "\t protection  = 0x%08" PRIx32 "\n"
		     "\t memory_type = %d\n"
		     "\t flexible    = %d\n"
		     "\t direct      = %d\n"
		     "\t name        = %s\n",
		     static_cast<uint64_t>(info->start), static_cast<uint64_t>(info->end), info->offset,
		     info->protection, info->memory_type, static_cast<int>(info->is_flexible),
		     static_cast<int>(info->is_direct), info->name);
	}

	return OK;
}

int KYTY_SYSV_ABI KernelIsStack(void* addr, void** start, void** end) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	auto vaddr = reinterpret_cast<uint64_t>(addr);

	LOGF("\t addr = 0x%016" PRIx64 "\n", vaddr);

	VirtualRanges::Range candidate {};
	if (!g_virtual_ranges->Query(vaddr, 0, &candidate)) {
		return KERNEL_ERROR_EACCES;
	}

	uint64_t stack_start = 0;
	uint64_t stack_end   = 0;
	if (candidate.type == VirtualRangeType::Stack) {
		stack_start = candidate.start;
		stack_end   = candidate.start + candidate.size;
	}

	if (start != nullptr) {
		*start = reinterpret_cast<void*>(stack_start);
	}

	if (end != nullptr) {
		*end = reinterpret_cast<void*>(stack_end);
	}

	LOGF("\t start = 0x%016" PRIx64 "\n"
	     "\t end   = 0x%016" PRIx64 "\n",
	     stack_start, stack_end);

	return OK;
}

int KYTY_SYSV_ABI KernelAvailableFlexibleMemorySize(size_t* size) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	if (size == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*size = g_flexible_memory->Available();

	LOGF("\t *size = 0x%016" PRIx64 "\n", *size);

	return OK;
}

int KYTY_SYSV_ABI KernelConfiguredFlexibleMemorySize(size_t* size) {
	PRINT_NAME();

	if (size == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*size = FlexibleMemory::Size();

	LOGF("\t *size = 0x%016" PRIx64 "\n", *size);

	return OK;
}

static int ProgramProtection(VirtualMemory::Mode mode) {
	const auto protection = static_cast<int>(mode);
	if ((protection & ~(PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC)) != 0) {
		EXIT("unsupported program-memory protection: 0x%08x\n", protection);
	}
	return protection;
}

static std::vector<VirtualRanges::Range> RequireGuestRuntimeMemory(uint64_t vaddr, uint64_t size) {
	std::vector<VirtualRanges::Range> ranges;
	if (g_virtual_ranges == nullptr || !g_virtual_ranges->QuerySpan(vaddr, size, &ranges) ||
	    std::any_of(ranges.begin(), ranges.end(), [](const auto& range) {
		    return range.type != VirtualRangeType::Code && range.type != VirtualRangeType::Runtime;
	    })) {
		EXIT("guest runtime range is not fully mapped: addr=0x%016" PRIx64 " size=0x%016" PRIx64
		     "\n",
		     vaddr, size);
	}
	return ranges;
}

static uint64_t AllocateGuestRuntimeMemory(uint64_t search_addr, uint64_t size,
                                           VirtualMemory::Mode mode, const char* name,
                                           VirtualRangeType type, bool fixed) {
	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	constexpr uint64_t GuestPageSize = 0x4000;
	if (size == 0 || size > UINT64_MAX - (GuestPageSize - 1u) || name == nullptr ||
	    (fixed && (search_addr == 0 || (search_addr & (GuestPageSize - 1u)) != 0))) {
		return 0;
	}
	const auto mapped_size = (size + GuestPageSize - 1u) & ~(GuestPageSize - 1u);
	const auto vaddr =
	    fixed ? search_addr : FindGuestFreeRange(search_addr, mapped_size, GuestPageSize);
	if (vaddr == 0 || g_virtual_ranges->HasOverlap(vaddr, mapped_size)) {
		return 0;
	}
	UnmapGpuRange(vaddr, mapped_size);
	if (!g_guest_address_space->Commit(vaddr, mapped_size, mode)) {
		return 0;
	}
	if (!g_virtual_ranges->Add(vaddr, mapped_size, 0, ProgramProtection(mode), 0, type, name)) {
		EXIT_IF(!g_guest_address_space->ReleaseCommitted(vaddr, mapped_size));
		return 0;
	}
	MapGpuRange(vaddr, mapped_size);
	return vaddr;
}

uint64_t AllocateProgramMemory(uint64_t search_addr, uint64_t size, VirtualMemory::Mode mode,
                               const char* name) {
	return AllocateGuestRuntimeMemory(search_addr, size, mode, name, VirtualRangeType::Code, false);
}

void SetProgramMemoryProtection(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	const auto ranges = RequireGuestRuntimeMemory(vaddr, size);
	if (std::any_of(ranges.begin(), ranges.end(),
	                [](const auto& range) { return range.type != VirtualRangeType::Code; })) {
		EXIT("program-memory range is not fully mapped: addr=0x%016" PRIx64 " size=0x%016" PRIx64
		     "\n",
		     vaddr, size);
	}

	const auto host_mode = VirtualMemory::IsExecute(mode) ? VirtualMemory::Mode::ExecuteReadWrite
	                                                      : VirtualMemory::Mode::ReadWrite;
	EXIT_IF(!g_guest_address_space->Protect(vaddr, size, host_mode));
	g_virtual_ranges->Protect(vaddr, size, ProgramProtection(mode));
}

uint64_t AllocateRuntimeMemory(uint64_t search_addr, uint64_t size, VirtualMemory::Mode mode,
                               const char* name, bool fixed) {
	return AllocateGuestRuntimeMemory(search_addr, size, mode, name, VirtualRangeType::Runtime,
	                                  fixed);
}

uint64_t AllocateGuestStackMemory(uint64_t search_addr, uint64_t size, VirtualMemory::Mode mode,
                                  const char* name) {
	const auto vaddr =
	    AllocateGuestRuntimeMemory(search_addr, size, mode, name, VirtualRangeType::Stack, false);
	if (vaddr != 0 && g_alloc_callback != nullptr) {
		g_alloc_callback(vaddr, size);
	}
	return vaddr;
}

bool ProtectGuestMemory(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode,
                        VirtualMemory::Mode* old_mode) {
	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);
	constexpr uint64_t                    GuestPageSize = 0x4000;
	if (vaddr == 0 || size == 0 || size > UINT64_MAX - (vaddr & (GuestPageSize - 1u))) {
		return false;
	}
	const auto aligned_addr = vaddr & ~(GuestPageSize - 1u);
	const auto aligned_size =
	    (size + (vaddr - aligned_addr) + GuestPageSize - 1u) & ~(GuestPageSize - 1u);
	const auto ranges = RequireGuestRuntimeMemory(aligned_addr, aligned_size);
	if (old_mode != nullptr) {
		*old_mode = static_cast<VirtualMemory::Mode>(
		    ranges.front().protection & (PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC));
	}
	if (!g_guest_address_space->Protect(aligned_addr, aligned_size, mode)) {
		return false;
	}
	g_virtual_ranges->Protect(aligned_addr, aligned_size, ProgramProtection(mode));
	return true;
}

bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
	return g_guest_address_space != nullptr &&
	       g_guest_address_space->ProtectTransient(vaddr, size, mode);
}

void AddLiveThreadStack(uint64_t vaddr, uint64_t size) {
	if (g_guest_address_space != nullptr) {
		g_guest_address_space->AddLiveStack(vaddr, size);
	}
}

void RemoveLiveThreadStack(uint64_t vaddr, uint64_t size) {
	// The tracker's protection of what it still watches there comes back (outside the
	// address-space lock: the tracker protects through it).
	if (g_guest_address_space != nullptr && g_guest_address_space->RemoveLiveStack(vaddr, size)) {
		ResyncGpuRangeProtection(vaddr, size);
	}
}

bool FreeGuestMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	constexpr uint64_t GuestPageSize = 0x4000;
	if (vaddr == 0 || size == 0 || size > UINT64_MAX - (GuestPageSize - 1u)) {
		return false;
	}
	const auto mapped_size = (size + GuestPageSize - 1u) & ~(GuestPageSize - 1u);
	(void)RequireGuestRuntimeMemory(vaddr, mapped_size);
	UnmapGpuRange(vaddr, mapped_size);
	return FreeGuestMemoryOwner(vaddr, mapped_size);
}

int KYTY_SYSV_ABI KernelMprotect(const void* addr, size_t len, int prot) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	auto vaddr = reinterpret_cast<uint64_t>(addr);

	LOGF("\t addr = 0x%016" PRIx64 "\n"
	     "\t len  = 0x%016" PRIx64 "\n",
	     vaddr, static_cast<uint64_t>(len));

	constexpr uint64_t PAGE_SIZE    = 0x4000;
	auto               aligned_addr = vaddr & ~(PAGE_SIZE - 1);
	const auto         page_offset  = vaddr - aligned_addr;
	if (len > UINT64_MAX - page_offset || len + page_offset > UINT64_MAX - (PAGE_SIZE - 1)) {
		EXIT("memory-protection range overflows: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, static_cast<uint64_t>(len));
	}
	auto aligned_len = (len + page_offset + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);

	if (aligned_len == 0) {
		return OK;
	}

	VirtualMemory::Mode mode     = VirtualMemory::Mode::NoAccess;
	GpuAccessMode       gpu_mode = GpuAccessMode::NoAccess;

	if (!DecodeMemoryProtection(prot, &mode, &gpu_mode)) {
		return KERNEL_ERROR_EINVAL;
	}
	std::vector<VirtualRanges::Range> old_ranges;
	if (!g_virtual_ranges->QuerySpan(aligned_addr, aligned_len, &old_ranges) ||
	    std::any_of(old_ranges.begin(), old_ranges.end(),
	                [](const auto& range) { return !IsCommittedRangeType(range.type); })) {
		EXIT("memory-protection range is not fully mapped: addr=0x%016" PRIx64 " size=0x%016" PRIx64
		     "\n",
		     aligned_addr, aligned_len);
	}
	const auto old_mode = static_cast<VirtualMemory::Mode>(
	    old_ranges.front().protection & (PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC));
	bool ok = g_guest_address_space->Protect(aligned_addr, aligned_len, mode);

	if (!ok) {
		EXIT("host memory-protection update failed: addr=0x%016" PRIx64 " size=0x%016" PRIx64
		     " prot=0x%08x\n",
		     aligned_addr, aligned_len, prot);
	}
	for (const auto& old_range: old_ranges) {
		if (old_range.type == VirtualRangeType::Direct) {
			g_physical_memory->ProtectMapping(old_range.start, old_range.size, prot, mode,
			                                  gpu_mode);
		} else if (old_range.type == VirtualRangeType::Flexible) {
			g_flexible_memory->Protect(old_range.start, old_range.size, prot, mode, gpu_mode);
		}
	}
	g_virtual_ranges->Protect(aligned_addr, aligned_len, prot);
	// A writable guest protection replaced the tracker's on watched pages: their CPU writes
	// would stop faulting.
	if ((prot & PROT_CPU_WRITE) != 0) {
		ResyncGpuRangeProtection(aligned_addr, aligned_len);
	}

	LOGF("\t prot: %s -> %s\n", magic_enum::enum_name(old_mode), magic_enum::enum_name(mode));

	return OK;
}

int KYTY_SYSV_ABI KernelMtypeprotect(const void* addr, size_t len, int type, int prot) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t addr = 0x%016" PRIx64 "\n"
	     "\t len  = 0x%016" PRIx64 "\n"
	     "\t type = 0x%08" PRIx32 "\n"
	     "\t prot = 0x%08" PRIx32 "\n",
	     reinterpret_cast<uint64_t>(addr), static_cast<uint64_t>(len), static_cast<uint32_t>(type),
	     static_cast<uint32_t>(prot));

	return KernelMprotect(addr, len, prot);
}

int KYTY_SYSV_ABI KernelBatchMap2(KernelBatchMapEntry* entries, int num_entries,
                                  int* num_entries_out, int flags) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t entries         = %p\n"
	     "\t num_entries     = %d\n"
	     "\t num_entries_out = %p\n"
	     "\t flags           = 0x%08" PRIx32 "\n",
	     static_cast<void*>(entries), num_entries, static_cast<void*>(num_entries_out),
	     static_cast<uint32_t>(flags));

	enum MemoryOpTypes {
		MAP_OP_MAP_DIRECT   = 0,
		MAP_OP_UNMAP        = 1,
		MAP_OP_PROTECT      = 2,
		MAP_OP_MAP_FLEXIBLE = 3,
		MAP_OP_TYPE_PROTECT = 4,
	};

	if (entries == nullptr || num_entries < 0) {
		return KERNEL_ERROR_EINVAL;
	}

	int processed = 0;

	for (int i = 0; i < num_entries; i++, processed++) {
		auto* entry = &entries[i];

		LOGF("\t [%d] start = %p, offset = 0x%016" PRIx64 ", length = 0x%016" PRIx64
		     ", prot = 0x%02" PRIx32 ", type = 0x%02" PRIx32 ", op = %d\n",
		     i, entry->start, entry->offset, entry->length,
		     static_cast<uint32_t>(entry->protection), static_cast<uint32_t>(entry->type),
		     entry->operation);

		if (entry->length == 0 || entry->operation < MAP_OP_MAP_DIRECT ||
		    entry->operation > MAP_OP_TYPE_PROTECT) {
			break;
		}

		int result = OK;
		switch (entry->operation) {
			case MAP_OP_MAP_DIRECT:
				result = KernelMapNamedDirectMemory(&entry->start, entry->length, entry->protection,
				                                    flags, static_cast<int64_t>(entry->offset), 0,
				                                    "anon");
				break;
			case MAP_OP_UNMAP:
				result = KernelMunmap(reinterpret_cast<uint64_t>(entry->start), entry->length);
				break;
			case MAP_OP_PROTECT:
				result = KernelMprotect(entry->start, entry->length, entry->protection);
				break;
			case MAP_OP_MAP_FLEXIBLE:
				result = KernelMapNamedFlexibleMemory(&entry->start, entry->length,
				                                      entry->protection, flags, "anon");
				break;
			case MAP_OP_TYPE_PROTECT:
				result =
				    KernelMtypeprotect(entry->start, entry->length, entry->type, entry->protection);
				break;
			default: result = KERNEL_ERROR_EINVAL; break;
		}

		if (result != OK) {
			if (num_entries_out != nullptr) {
				*num_entries_out = processed;
			}
			return result;
		}
	}

	if (num_entries_out != nullptr) {
		*num_entries_out = processed;
	}

	return (processed == num_entries ? OK : KERNEL_ERROR_EINVAL);
}

int KYTY_SYSV_ABI KernelBatchMap(KernelBatchMapEntry* entries, int num_entries,
                                 int* num_entries_out) {
	constexpr int GUEST_MAP_FIXED = 0x10;
	return KernelBatchMap2(entries, num_entries, num_entries_out, GUEST_MAP_FIXED);
}

static bool IsAligned(uint64_t value, uint64_t alignment) {
	return alignment == 0 || (value & (alignment - 1u)) == 0;
}

static void MemoryPoolSubtractCommitted(uint64_t len) {
	auto current = g_memory_pool_committed.load(std::memory_order_relaxed);
	while (current != 0) {
		const auto next = (current > len ? current - len : 0);
		if (g_memory_pool_committed.compare_exchange_weak(current, next,
		                                                  std::memory_order_relaxed)) {
			return;
		}
	}
}

int KYTY_SYSV_ABI KernelMemoryPoolExpand(int64_t search_start, int64_t search_end, size_t len,
                                         size_t alignment, int64_t* phys_addr_out) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	constexpr uint64_t POOL_PAGE_SIZE = 0x10000;
	if (search_start < 0 || search_end <= search_start || len == 0 ||
	    (len & (POOL_PAGE_SIZE - 1u)) != 0 || phys_addr_out == nullptr ||
	    (alignment != 0 &&
	     (!std::has_single_bit(alignment) || (alignment & (POOL_PAGE_SIZE - 1u)) != 0))) {
		return KERNEL_ERROR_EINVAL;
	}
	if (static_cast<uint64_t>(search_end - search_start) < len) {
		return KERNEL_ERROR_ENOMEM;
	}

	const auto effective_alignment = (alignment != 0 ? alignment : POOL_PAGE_SIZE);
	uint64_t   phys_addr           = 0;
	if (!g_physical_memory->Alloc(static_cast<uint64_t>(search_start),
	                              static_cast<uint64_t>(search_end), len, effective_alignment,
	                              &phys_addr, 0, PhysicalMemory::AllocationKind::Pooled)) {
		return KERNEL_ERROR_ENOMEM;
	}

	g_pooled_memory->Expand(phys_addr, len);
	*phys_addr_out = static_cast<int64_t>(phys_addr);

	LOGF("\t search_start = 0x%016" PRIx64 "\n"
	     "\t search_end   = 0x%016" PRIx64 "\n"
	     "\t len          = 0x%016" PRIx64 "\n"
	     "\t alignment    = 0x%016" PRIx64 "\n"
	     "\t phys_addr    = 0x%016" PRIx64 "\n",
	     search_start, search_end, static_cast<uint64_t>(len), static_cast<uint64_t>(alignment),
	     phys_addr);
	return OK;
}

int KYTY_SYSV_ABI KernelMemoryPoolReserve(void* addr_in, size_t len, size_t alignment, int flags,
                                          void** addr_out) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t addr_in   = 0x%016" PRIx64 "\n"
	     "\t len       = 0x%016" PRIx64 "\n"
	     "\t alignment = 0x%016" PRIx64 "\n"
	     "\t flags     = 0x%08" PRIx32 "\n"
	     "\t addr_out  = %p\n",
	     reinterpret_cast<uint64_t>(addr_in), static_cast<uint64_t>(len),
	     static_cast<uint64_t>(alignment), static_cast<uint32_t>(flags),
	     static_cast<void*>(addr_out));

	constexpr uint64_t POOL_RESERVE_ALIGNMENT = 0x200000;

	if (addr_out == nullptr || len == 0 || !IsAligned(len, POOL_RESERVE_ALIGNMENT)) {
		return KERNEL_ERROR_EINVAL;
	}
	if (alignment != 0 &&
	    (!std::has_single_bit(alignment) || !IsAligned(alignment, POOL_RESERVE_ALIGNMENT))) {
		return KERNEL_ERROR_EINVAL;
	}

	void*      out_addr          = addr_in;
	const auto reserve_alignment = (alignment != 0 ? alignment : POOL_RESERVE_ALIGNMENT);
	const int  ret = KernelReserveVirtualRange(&out_addr, len, flags, reserve_alignment);
	if (ret == OK) {
		const auto           out_vaddr = reinterpret_cast<uint64_t>(out_addr);
		VirtualRanges::Range reserved_range {};
		if (!g_virtual_ranges->Query(out_vaddr, 0, &reserved_range) ||
		    reserved_range.start != out_vaddr || reserved_range.size != len ||
		    !IsReservedRangeType(reserved_range.type)) {
			return KERNEL_ERROR_EBUSY;
		}
		if (!g_virtual_ranges->ReplaceSpan(out_vaddr, len, VirtualRangeType::Reserved, 0, 0, 0,
		                                   VirtualRangeType::PoolReserved, reserved_range.name)) {
			return KERNEL_ERROR_EBUSY;
		}
		*addr_out = out_addr;
		LOGF("\t out_addr  = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(out_addr));
	}

	return ret;
}

int KYTY_SYSV_ABI KernelMemoryPoolCommit(void* addr, size_t len, int type, int prot, int flags) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t addr  = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n"
	     "\t type  = 0x%08" PRIx32 "\n"
	     "\t prot  = 0x%08" PRIx32 "\n"
	     "\t flags = 0x%08" PRIx32 "\n",
	     reinterpret_cast<uint64_t>(addr), static_cast<uint64_t>(len), static_cast<uint32_t>(type),
	     static_cast<uint32_t>(prot), static_cast<uint32_t>(flags));

	constexpr uint64_t POOL_COMMIT_ALIGNMENT = 0x10000;
	constexpr int      PROT_CPU_EXEC         = 0x04;

	if (addr == nullptr || len == 0 || !IsAligned(len, POOL_COMMIT_ALIGNMENT) ||
	    (prot & PROT_CPU_EXEC) != 0) {
		return KERNEL_ERROR_EINVAL;
	}

	VirtualMemory::Mode mode     = VirtualMemory::Mode::NoAccess;
	GpuAccessMode       gpu_mode = GpuAccessMode::NoAccess;
	if (!DecodeMemoryProtection(prot, &mode, &gpu_mode)) {
		return KERNEL_ERROR_EINVAL;
	}

	const auto vaddr = reinterpret_cast<uint64_t>(addr);

	VirtualRanges::Range old_range {};
	if (!g_virtual_ranges->Query(vaddr, 0, &old_range) ||
	    old_range.type != VirtualRangeType::PoolReserved ||
	    len > old_range.size - (vaddr - old_range.start)) {
		return KERNEL_ERROR_EACCES;
	}

	std::vector<PooledMemory::Mapping> mappings;
	if (!g_pooled_memory->Allocate(vaddr, len, gpu_mode, &mappings)) {
		return KERNEL_ERROR_ENOMEM;
	}

	std::vector<PooledMemory::Mapping> mapped;
	auto                               rollback = [&]() {
		if (!UnmapPooledBackingTransactional(mapped, mode)) {
			EXIT("failed to roll back pooled-memory backing maps\n");
		}
		GpuAccessMode rollback_gpu_mode = GpuAccessMode::NoAccess;
		if (!g_pooled_memory->Release(vaddr, len, &rollback_gpu_mode)) {
			EXIT("failed to release pooled-memory rollback allocation\n");
		}
	};

	for (const auto& mapping: mappings) {
		auto       failure_reason = GuestBackingStore::FailureReason::None;
		const bool ok = g_guest_address_space->MapBacking(mapping.vaddr, mapping.size,
		                                                  mapping.phys_addr, mode, &failure_reason);
		if (!ok) {
			LOGF_COLOR(Log::Color::Red, "\t pool backing map failed: %s\n",
			           GuestBackingStore::GetFailureReasonName(failure_reason));
			rollback();
			return KERNEL_ERROR_ENOMEM;
		}
		mapped.push_back(mapping);
	}

	if (!g_virtual_ranges->ReplaceSpan(vaddr, len, VirtualRangeType::PoolReserved, 0, prot, type,
	                                   VirtualRangeType::Pooled, old_range.name)) {
		rollback();
		return KERNEL_ERROR_EBUSY;
	}

	MapGpuRange(vaddr, len);

	if (g_alloc_callback != nullptr) {
		g_alloc_callback(vaddr, len);
	}

	g_memory_pool_committed.fetch_add(len, std::memory_order_relaxed);

	return OK;
}

static int DecommitMemoryPoolRange(uint64_t vaddr, size_t len) {
	VirtualRanges::Range old_range {};
	if (!g_virtual_ranges->Query(vaddr, 0, &old_range)) {
		return KERNEL_ERROR_EACCES;
	}
	const auto chunk_len = std::min<uint64_t>(len, old_range.size - (vaddr - old_range.start));
	if (old_range.type == VirtualRangeType::PoolReserved) {
		return chunk_len < len ? DecommitMemoryPoolRange(vaddr + chunk_len, len - chunk_len) : OK;
	}
	if (old_range.type != VirtualRangeType::Pooled) {
		return KERNEL_ERROR_EACCES;
	}
	if (chunk_len < len) {
		const int ret = DecommitMemoryPoolRange(vaddr, chunk_len);
		return ret == OK ? DecommitMemoryPoolRange(vaddr + chunk_len, len - chunk_len) : ret;
	}

	std::vector<PooledMemory::Mapping> mappings;
	if (!g_pooled_memory->Query(vaddr, len, &mappings)) {
		return KERNEL_ERROR_EACCES;
	}

	VirtualMemory::Mode mode        = VirtualMemory::Mode::NoAccess;
	GpuAccessMode       decoded_gpu = GpuAccessMode::NoAccess;
	if (!DecodeMemoryProtection(old_range.protection, &mode, &decoded_gpu)) {
		return KERNEL_ERROR_EACCES;
	}
	if (!UnmapPooledBackingTransactional(mappings, mode)) {
		EXIT("pooled-memory backing transaction failed after GPU unmap: addr=0x%016" PRIx64
		     " size=0x%016" PRIx64 "\n",
		     vaddr, len);
	}

	GpuAccessMode gpu_mode = GpuAccessMode::NoAccess;
	if (!g_pooled_memory->Release(vaddr, len, &gpu_mode)) {
		EXIT("failed to release decommitted pooled-memory range\n");
	}

	g_virtual_ranges->Remove(vaddr, len);
	g_virtual_ranges->Add(vaddr, len, 0, 0, 0, VirtualRangeType::PoolReserved, old_range.name);

	if (g_free_callback != nullptr) {
		g_free_callback(vaddr, len);
	}

	MemoryPoolSubtractCommitted(len);
	return OK;
}

int KYTY_SYSV_ABI KernelMemoryPoolDecommit(void* addr, size_t len, int flags) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	LOGF("\t addr  = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n"
	     "\t flags = 0x%08" PRIx32 "\n",
	     reinterpret_cast<uint64_t>(addr), static_cast<uint64_t>(len),
	     static_cast<uint32_t>(flags));

	constexpr uint64_t POOL_COMMIT_ALIGNMENT = 0x10000;

	if (addr == nullptr || len == 0 || !IsAligned(len, POOL_COMMIT_ALIGNMENT)) {
		return KERNEL_ERROR_EINVAL;
	}

	const auto vaddr = reinterpret_cast<uint64_t>(addr);
	if (UINT64_MAX - vaddr < len) {
		return KERNEL_ERROR_EINVAL;
	}
	const auto end  = vaddr + len;
	auto       scan = vaddr;
	while (scan < end) {
		VirtualRanges::Range scan_range {};
		if (!g_virtual_ranges->Query(scan, 0, &scan_range) ||
		    (scan_range.type != VirtualRangeType::Pooled &&
		     scan_range.type != VirtualRangeType::PoolReserved)) {
			return KERNEL_ERROR_EACCES;
		}
		const auto next = std::min(end, scan_range.start + scan_range.size);
		if (next <= scan) {
			return KERNEL_ERROR_EACCES;
		}
		scan = next;
	}

	UnmapGpuRange(vaddr, len);
	return DecommitMemoryPoolRange(vaddr, len);
}

int KYTY_SYSV_ABI KernelMemoryPoolBatch(const KernelMemoryPoolBatchEntry* entries, int num_entries,
                                        int* num_entries_out, int flags) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	if (entries == nullptr || num_entries < 0) {
		return KERNEL_ERROR_EINVAL;
	}

	enum MemoryPoolOp {
		POOL_OP_COMMIT       = 1,
		POOL_OP_DECOMMIT     = 2,
		POOL_OP_PROTECT      = 3,
		POOL_OP_TYPE_PROTECT = 4,
		POOL_OP_MOVE         = 5,
	};

	int processed = 0;
	int result    = OK;

	for (int i = 0; i < num_entries; i++, processed++) {
		const auto& entry = entries[i];
		switch (entry.op) {
			case POOL_OP_COMMIT:
				result = KernelMemoryPoolCommit(entry.commit.addr, entry.commit.len,
				                                entry.commit.type, entry.commit.prot, entry.flags);
				break;
			case POOL_OP_DECOMMIT:
				result =
				    KernelMemoryPoolDecommit(entry.decommit.addr, entry.decommit.len, entry.flags);
				break;
			case POOL_OP_PROTECT:
				result = KernelMprotect(entry.protect.addr, entry.protect.len, entry.protect.prot);
				break;
			case POOL_OP_TYPE_PROTECT:
				result = KernelMtypeprotect(entry.type_protect.addr, entry.type_protect.len,
				                            entry.type_protect.type, entry.type_protect.prot);
				break;
			case POOL_OP_MOVE:
			default: result = KERNEL_ERROR_EINVAL; break;
		}

		if (result != OK) {
			break;
		}
	}

	if (num_entries_out != nullptr) {
		*num_entries_out = processed;
	}

	(void)flags;
	return result;
}

int KYTY_SYSV_ABI KernelMemoryPoolGetBlockStats(KernelMemoryPoolBlockStats* output,
                                                size_t                      output_size) {
	PRINT_NAME();

	std::lock_guard<std::recursive_mutex> memory_operation_lock(g_memory_operation_mutex);

	if (output == nullptr && output_size != 0) {
		return KERNEL_ERROR_EFAULT;
	}

	KernelMemoryPoolBlockStats stats {};
	constexpr uint64_t         BLOCK_SIZE = 0x10000;
	const uint64_t             committed  = g_memory_pool_committed.load(std::memory_order_relaxed);
	const uint64_t available = (g_pooled_memory != nullptr ? g_pooled_memory->Available() : 0);

	stats.available_flushed_blocks = static_cast<int32_t>(available / BLOCK_SIZE);
	stats.available_cached_blocks  = 0;
	stats.allocated_flushed_blocks = static_cast<int32_t>(committed / BLOCK_SIZE);
	stats.allocated_cached_blocks  = 0;

	const auto copy_size = std::min(output_size, sizeof(stats));
	if (copy_size != 0) {
		std::memcpy(output, &stats, copy_size);
	}

	return OK;
}

} // namespace Libs::LibKernel::Memory
