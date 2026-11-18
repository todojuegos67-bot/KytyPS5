#include "common/hostException.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/imageZoneStamps.h"
#include "graphics/host_gpu/renderer/cache/pagedRangeSet.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <semaphore>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach-o/dyld.h>
#include <csignal>
#include <limits.h>
#include <map>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#include <csignal>
#include <map>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using Libs::Graphics::GuestRange;
using Libs::Graphics::MemoryTracker;
using Libs::Graphics::PageManager;
using Libs::Graphics::RangeSet;
using Libs::Graphics::TRACKER_ADDRESS_SIZE;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "MemoryTrackerTests: failed: %s\n", text);
    std::abort();
  }
}

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
using DWORD = uint32_t;
constexpr uint32_t PAGE_NOACCESS = 1;
constexpr uint32_t PAGE_READONLY = 2;
constexpr uint32_t PAGE_READWRITE = 3;
constexpr uint32_t MEM_RESERVE = 0;
constexpr uint32_t MEM_COMMIT = 0;
constexpr uint32_t MEM_RELEASE = 0;

int ToHostProt(uint32_t protection) {
  switch (protection) {
  case PAGE_NOACCESS:
    return PROT_NONE;
  case PAGE_READONLY:
    return PROT_READ;
  default:
    return PROT_READ | PROT_WRITE;
  }
}

uint32_t Protection(const void *address) {
#if defined(__APPLE__)
  mach_vm_address_t region_address =
      reinterpret_cast<mach_vm_address_t>(address);
  mach_vm_size_t region_size = 0;
  vm_region_basic_info_data_64_t info{};
  mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object_name = MACH_PORT_NULL;
  Check(mach_vm_region(mach_task_self(), &region_address, &region_size,
                       VM_REGION_BASIC_INFO_64,
                       reinterpret_cast<vm_region_info_t>(&info), &info_count,
                       &object_name) == KERN_SUCCESS,
        "mach_vm_region failed");
  if (object_name != MACH_PORT_NULL) {
    mach_port_deallocate(mach_task_self(), object_name);
  }
  return (info.protection & VM_PROT_WRITE) != 0
             ? PAGE_READWRITE
             : (info.protection & VM_PROT_READ) != 0 ? PAGE_READONLY
                                                     : PAGE_NOACCESS;
#else
  const auto addr = reinterpret_cast<uintptr_t>(address);
  std::FILE *maps = std::fopen("/proc/self/maps", "r");
  Check(maps != nullptr, "open /proc/self/maps failed");
  char line[512];
  uint32_t result = 0;
  while (std::fgets(line, sizeof(line), maps) != nullptr) {
    unsigned long start = 0;
    unsigned long end = 0;
    char perms[8]{};
    if (std::sscanf(line, "%lx-%lx %7s", &start, &end, perms) != 3) {
      continue;
    }
    if (addr >= start && addr < end) {
      result = perms[1] == 'w'   ? PAGE_READWRITE
               : perms[0] == 'r' ? PAGE_READONLY
                                 : PAGE_NOACCESS;
      break;
    }
  }
  std::fclose(maps);
  return result;
#endif
}

std::map<void *, size_t> &AllocationSizes() {
  static std::map<void *, size_t> sizes;
  return sizes;
}

void *VirtualAlloc(void *address, size_t size, DWORD, uint32_t protection) {
#if defined(__APPLE__)
  mach_vm_address_t raw_address = reinterpret_cast<mach_vm_address_t>(address);
  const auto flags = address != nullptr ? VM_FLAGS_FIXED : VM_FLAGS_ANYWHERE;
  if (mach_vm_allocate(mach_task_self(), &raw_address, size, flags) !=
      KERN_SUCCESS) {
    return nullptr;
  }
  if (mach_vm_protect(mach_task_self(), raw_address, size, false,
                      static_cast<vm_prot_t>(ToHostProt(protection))) !=
      KERN_SUCCESS) {
    mach_vm_deallocate(mach_task_self(), raw_address, size);
    return nullptr;
  }
  void *raw = reinterpret_cast<void *>(raw_address);
  AllocationSizes()[raw] = size;
  return raw;
#else
  const int extra = address != nullptr ? MAP_FIXED_NOREPLACE : 0;
  void *raw = ::mmap(address, size, ToHostProt(protection),
                     MAP_PRIVATE | MAP_ANONYMOUS | extra, -1, 0);
  if (raw == MAP_FAILED) {
    return nullptr;
  }
  AllocationSizes()[raw] = size;
  return raw;
#endif
}

int VirtualFree(void *address, size_t, DWORD) {
  auto &sizes = AllocationSizes();
  auto it = sizes.find(address);
  if (it == sizes.end()) {
    return 0;
  }
#if defined(__APPLE__)
  const int ok = mach_vm_deallocate(mach_task_self(),
                                    reinterpret_cast<mach_vm_address_t>(address),
                                    it->second) == KERN_SUCCESS
                     ? 1
                     : 0;
#else
  const int ok = ::munmap(address, it->second) == 0 ? 1 : 0;
#endif
  sizes.erase(it);
  return ok;
}

int VirtualProtect(void *address, size_t size, uint32_t protection,
                   DWORD *old_protection) {
  if (old_protection != nullptr) {
    *old_protection = Protection(address);
  }
#if defined(__APPLE__)
  return mach_vm_protect(mach_task_self(),
                         reinterpret_cast<mach_vm_address_t>(address), size,
                         false, static_cast<vm_prot_t>(ToHostProt(protection))) ==
                 KERN_SUCCESS
             ? 1
             : 0;
#else
  return ::mprotect(address, size, ToHostProt(protection)) == 0 ? 1 : 0;
#endif
}
#else
uint32_t Protection(const void *address) {
  MEMORY_BASIC_INFORMATION info{};
  Check(VirtualQuery(address, &info, sizeof(info)) != 0, "VirtualQuery failed");
  return info.Protect;
}
#endif

bool IsWritable(const void *address) {
  return Protection(address) == PAGE_READWRITE;
}

uint64_t g_protection_calls = 0;

struct ProtectionCall {
  uint64_t address;
  uint64_t size;
  Common::VirtualMemory::Mode mode;
};

std::vector<ProtectionCall> g_protection_log;

void ResetProtectionLog() {
  g_protection_calls = 0;
  g_protection_log.clear();
}

bool ProtectAddressSpace(uint64_t vaddr, uint64_t size,
                         Common::VirtualMemory::Mode mode) {
  uint32_t protection = PAGE_NOACCESS;
  if (mode == Common::VirtualMemory::Mode::Read) {
    protection = PAGE_READONLY;
  } else if (mode == Common::VirtualMemory::Mode::ReadWrite) {
    protection = PAGE_READWRITE;
  }
  DWORD old_protection = 0;
  g_protection_calls++;
  g_protection_log.push_back({vaddr, size, mode});
  return VirtualProtect(reinterpret_cast<void *>(vaddr), size, protection,
                        &old_protection) != 0;
}

struct TrackerHarness {
  TrackerHarness() : tracker(page_manager) {}

  PageManager page_manager;
  MemoryTracker tracker;
};

uint8_t *AllocateFixedGuestRange(uint64_t size, uintptr_t offset) {
  // Keep the mapping inside the tracker's guest address space and preserve the
  // requested region alignment. A fixed address can be occupied by the host
  // process (notably by the macOS runner's ASLR layout).
  constexpr uintptr_t first_base = 0x0000000200000000ull;
  constexpr uintptr_t stride = 0x0000000100000000ull;
  for (uintptr_t attempt = 0; attempt < 256; attempt++) {
    auto *wanted = reinterpret_cast<void *>(first_base + offset + attempt * stride);
    auto *memory = static_cast<uint8_t *>(
        VirtualAlloc(wanted, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (memory == wanted) {
      return memory;
    }
    if (memory != nullptr) {
      Check(VirtualFree(memory, 0, MEM_RELEASE) != 0,
            "release unexpected fixed allocation failed");
    }
  }
  Check(false, "no free fixed guest address found");
  return nullptr;
}

uint8_t *Allocate(PageManager &manager, uint64_t pages) {
  const auto size = manager.GetPageSize() * pages;
  return AllocateFixedGuestRange(size, 0x10000);
}

void Release(uint8_t *memory) {
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

void TestRangeSet() {
  RangeSet ranges;
  ranges.Add(0x1000, 0x80);
  ranges.Add(0x1080, 0x80);
  ranges.Add(0x1200, 0x40);
  Check(ranges.Contains(0x1010, 0xe0) && !ranges.Contains(0x1010, 0x200),
        "range set containment did not require full coverage");
  Check(ranges.Intersects(0x0fff, 2) && ranges.Intersects(0x11ff, 2) &&
            !ranges.Intersects(0x1100, 0x100) &&
            !ranges.Intersects(0x1240, 1),
        "range set intersection did not preserve half-open boundaries");
  std::vector<std::pair<uint64_t, uint64_t>> intersections;
  ranges.ForEachInRange(0x1070, 0x1b0, [&](uint64_t start, uint64_t end) {
    intersections.emplace_back(start, end);
  });
  Check(intersections.size() == 2 && intersections[0].first == 0x1070 &&
            intersections[0].second == 0x1100 &&
            intersections[1].first == 0x1200 && intersections[1].second == 0x1220,
        "range set did not merge and intersect exact byte ranges");
  ranges.Subtract(0x1040, 0x1e0);
  intersections.clear();
  ranges.ForEachInRange(0x1000, 0x300, [&](uint64_t start, uint64_t end) {
    intersections.emplace_back(start, end);
  });
  Check(intersections.size() == 2 && intersections[0].first == 0x1000 &&
            intersections[0].second == 0x1040 &&
            intersections[1].first == 0x1220 && intersections[1].second == 0x1240,
        "range set subtraction did not preserve both exact tails");
}

// The page bitmap of PagedRangeSet (round 80): its answers are the RangeSet's, over random
// additions, subtractions and queries of every size, and every page holding bytes stays marked.
void TestPagedRangeSet() {
	using Libs::Graphics::PagedRangeSet;
	std::mt19937_64 rng(0x5eed80);
	for (int round = 0; round < 60; round++) {
		PagedRangeSet  paged;
		RangeSet       plain;
		const uint64_t base = (rng() % 4) * (uint64_t {1} << 36) + (rng() % 1000) * 4096;
		for (int op = 0; op < 2000; op++) {
			uint64_t address = base + rng() % (uint64_t {64} << 20u);
			uint64_t size    = 0;
			switch (rng() % 4) {
				case 0: size = 1 + rng() % 64; break;
				case 1: size = 1 + rng() % 8192; break;
				case 2: size = 1 + rng() % (uint64_t {1} << 20u); break;
				default: size = 1 + rng() % (uint64_t {20} << 20u); break;
			}
			if (rng() % 5 == 0) {
				address &= ~uint64_t {4095};
			}
			if (rng() % 5 == 0) {
				size = (size + 4095) & ~uint64_t {4095};
			}
			const auto kind = rng() % 10;
			if (kind < 3) {
				paged.Add(address, size);
				plain.Add(address, size);
			} else if (kind < 6) {
				paged.Subtract(address, size);
				plain.Subtract(address, size);
			} else if (kind == 9 && rng() % 50 == 0) {
				paged.Clear();
				plain.Clear();
			} else {
				Check(paged.Intersects(address, size) == plain.Intersects(address, size),
				      "the paged range set answered an intersection unlike the range set");
			}
			if (op % 97 == 0) {
				bool marked = true;
				plain.ForEach([&](uint64_t begin, uint64_t end) {
					for (uint64_t page = begin & ~uint64_t {4095}; page < end; page += 4096) {
						marked = marked && paged.PageMarked(page);
					}
				});
				Check(marked, "a page holding bytes of the paged range set was not marked");
			}
		}
	}
	PagedRangeSet edge;
	edge.Add(0x10000, 0x10);
	edge.Add(0x10ff0, 0x20);
	edge.Subtract(0x10000, 0x10);
	Check(edge.Intersects(0x10ff0, 1) && edge.Intersects(0x11000, 1) &&
	          !edge.Intersects(0x10000, 0x800),
	      "a subtraction cleared a page another range still holds bytes of");
	edge.Subtract(0x10ff0, 0x20);
	Check(!edge.PageMarked(0x10000) && !edge.PageMarked(0x11000) && edge.Empty(),
	      "the last subtraction left a page marked");
	// The last page of the address space: its check for bytes left must not wrap.
	PagedRangeSet top;
	top.Add(0xfffffffffffff000ull, 0x10);
	top.Subtract(0xfffffffffffff800ull, 0x10);
	Check(top.Intersects(0xfffffffffffff000ull, 0x10) && top.PageMarked(0xfffffffffffff000ull),
	      "a subtraction on the last page unmarked it while it holds bytes");
	top.Subtract(0xfffffffffffff000ull, 0x10);
	Check(!top.PageMarked(0xfffffffffffff000ull) && top.Empty(),
	      "the last page stayed marked without bytes");
}

// The BDA walk's spans of one region (round 80): their CPU-dirty pages become clean and
// write-protected in one protection change, bridging the clean pages between them; a CPU-dirty
// page outside every span stays dirty and writable (and splits the change in two).
void TestCpuDirtySpans() {
	TrackerHarness harness;
	auto&          tracker   = harness.tracker;
	const auto     page_size = harness.page_manager.GetPageSize();
	auto*          memory    = Allocate(harness.page_manager, 16);
	const auto     address   = reinterpret_cast<uint64_t>(memory);
	tracker.ForEachUploadRange(
	    address, page_size * 16, false, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
	const auto page = [&](uint64_t index) { return address + page_size * index; };
	for (const uint64_t index: {1u, 2u, 5u, 9u}) {
		tracker.MarkRegionAsCpuModified(page(index) + 8, 8);
	}
	const std::vector<std::pair<uint64_t, uint64_t>> spans = {
	    {page(1), page_size * 2}, {page(5), page_size}, {page(9), page_size}};
	std::vector<std::tuple<size_t, uint64_t, uint64_t>> runs;
	ResetProtectionLog();
	tracker.TakeCpuDirtySpans(spans, [&](size_t span, uint64_t run, uint64_t bytes) noexcept {
		runs.emplace_back(span, run, bytes);
	});
	Check(runs.size() == 3 && std::get<0>(runs[0]) == 0 && std::get<1>(runs[0]) == page(1) &&
	          std::get<2>(runs[0]) == page_size * 2 && std::get<0>(runs[1]) == 1 &&
	          std::get<1>(runs[1]) == page(5) && std::get<0>(runs[2]) == 2 &&
	          std::get<1>(runs[2]) == page(9),
	      "the spans' CPU-dirty runs were not given span by span");
	Check(g_protection_calls == 1 && !tracker.IsRegionCpuModified(page(1), page_size * 9) &&
	          Protection(memory + page_size * 1) == PAGE_READONLY &&
	          Protection(memory + page_size * 9) == PAGE_READONLY,
	      "the spans were not write-protected in one change");

	tracker.MarkRegionAsCpuModified(page(2) + 8, 8);
	tracker.MarkRegionAsCpuModified(page(7) + 8, 8);
	tracker.MarkRegionAsCpuModified(page(9) + 8, 8);
	const std::vector<std::pair<uint64_t, uint64_t>> apart = {{page(2), page_size},
	                                                          {page(9), page_size}};
	runs.clear();
	ResetProtectionLog();
	tracker.TakeCpuDirtySpans(apart, [&](size_t span, uint64_t run, uint64_t bytes) noexcept {
		runs.emplace_back(span, run, bytes);
	});
	Check(runs.size() == 2 && g_protection_calls == 2 &&
	          tracker.IsRegionCpuModified(page(7), page_size) && IsWritable(memory + page_size * 7),
	      "a CPU-dirty page outside the spans was taken or protected");

	tracker.MarkRegionAsCpuModified(address, page_size * 16);
	tracker.UntrackMemory(address, page_size * 16);
	Release(memory);
}

void TestGuestRange() {
  constexpr GuestRange empty{};
  constexpr GuestRange first_byte{1, 1};
  constexpr uint64_t extended_end = Libs::LibKernel::Memory::kExtendedMemoryBase +
                                    Libs::LibKernel::Memory::kExtendedMemorySize;
  constexpr GuestRange last_byte{extended_end - 1, 1};

  static_assert(empty.Empty() && !empty.Valid() && empty.ValidOrEmpty());
  static_assert(!first_byte.Empty() && first_byte.Valid() &&
                first_byte.ValidOrEmpty() && first_byte.End() == 2);
  static_assert(last_byte.Valid() && last_byte.End() == extended_end);

  Check(!GuestRange{0, 1}.Empty() && !GuestRange{0, 1}.ValidOrEmpty(),
        "zero-address nonempty guest range is rejected");
  Check(!GuestRange{1, 0}.Empty() && !GuestRange{1, 0}.ValidOrEmpty(),
        "nonzero-address empty guest range is rejected");
  Check(GuestRange{Libs::LibKernel::Memory::kExtendedMemoryBase, 1}.Valid() &&
            !GuestRange{Libs::Graphics::LOWER_ADDRESS_SIZE, 1}.Valid() &&
            !GuestRange{Libs::LibKernel::Memory::kExtendedMemoryBase - 1, 2}.Valid() &&
            !GuestRange{extended_end - 1, 2}.Valid(),
        "extended range and gap boundaries are enforced");
  Check(!GuestRange{TRACKER_ADDRESS_SIZE, 1}.Valid(),
        "first address beyond the guest range is rejected");
  Check(!GuestRange{TRACKER_ADDRESS_SIZE - 1, 2}.Valid(),
        "guest range crossing the address-space end is rejected");
  Check(!GuestRange{UINT64_MAX, 2}.Valid(), "wrapping guest range is rejected");
}

void TestQueriesDoNotRequireMappedOwnership() {
  constexpr uint64_t address = 0x0000000203000000ull;
  TrackerHarness harness;
  const auto page_size = harness.page_manager.GetPageSize();
  Check(harness.tracker.IsRegionCpuModified(address, page_size) &&
            !harness.tracker.IsRegionGpuModified(address, page_size),
        "unowned tracker range did not expose its initial CPU-dirty state");
}

void TestConcurrentRegionPublication() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  std::binary_semaphore start_first{0};
  std::binary_semaphore start_second{0};
  std::atomic_uint32_t cpu_dirty_results{0};
  std::jthread first([&] {
    start_first.acquire();
    if (tracker.IsRegionCpuModified(address, page_size)) {
      cpu_dirty_results.fetch_add(1, std::memory_order_relaxed);
    }
  });
  std::jthread second([&] {
    start_second.acquire();
    if (tracker.IsRegionCpuModified(address, page_size)) {
      cpu_dirty_results.fetch_add(1, std::memory_order_relaxed);
    }
  });
  start_first.release();
  start_second.release();
  first.join();
  second.join();

  tracker.UntrackMemory(address, page_size);
  Release(memory);
  Check(cpu_dirty_results.load(std::memory_order_relaxed) == 2,
        "concurrent region publication lost initial CPU ownership");
}

void TestCpuDirtyUpload() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);
  Check(tracker.IsRegionCpuModified(address + 16, 32),
        "new region was not CPU dirty");

  uint32_t ranges = 0;
  bool uploaded = false;
  tracker.ForEachUploadRange(
      address + 16, 32, false,
      [&](uint64_t upload_address, uint64_t upload_size) noexcept {
        Check(upload_address == address && upload_size == page_size,
              "upload range was not page aligned");
        ranges++;
      },
      [&]() noexcept { uploaded = true; });
  Check(ranges == 1 && uploaded &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "upload did not clear CPU dirty state and arm protection");

  tracker.MarkRegionAsCpuModified(address + 16, 32);
  Check(tracker.IsRegionCpuModified(address, page_size) && IsWritable(memory),
        "explicit CPU dirtiness did not release write protection");
  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

void TestCleanUploadPreservesOwnership() {
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = AllocateFixedGuestRange(region_size * 2, region_size);
  const auto address = reinterpret_cast<uint64_t>(memory);
  tracker.ForEachUploadRange(address, region_size * 2, false,
                             [](uint64_t, uint64_t) noexcept {},
                             []() noexcept {});

  for (const auto start : {address + 63 * page_size,
                           address + region_size - page_size}) {
    const auto before = start - page_size;
    const auto after = start + 2 * page_size;
    tracker.MarkRegionAsCpuModified(before, page_size);
    tracker.MarkRegionAsCpuModified(after, page_size);
    uint32_t ranges = 0;
    uint32_t completions = 0;
    ResetProtectionLog();
    tracker.ForEachUploadRange(
        start, 2 * page_size, false,
        [&](uint64_t, uint64_t) noexcept { ranges++; },
        [&]() noexcept { completions++; });
    Check(ranges == 0 && completions == 1 && g_protection_calls == 0 &&
              !tracker.IsRegionCpuModified(start, 2 * page_size),
          "clean read-only upload changed dirty state or protection");

    tracker.ForEachUploadRange(
        start, 2 * page_size, true,
        [&](uint64_t, uint64_t) noexcept { ranges++; },
        [&]() noexcept { completions++; });
    Check(ranges == 0 && completions == 2 &&
              tracker.IsRegionGpuModified(start, page_size) &&
              tracker.IsRegionGpuModified(start + page_size, page_size) &&
              Protection(reinterpret_cast<void *>(start)) == PAGE_NOACCESS &&
              Protection(reinterpret_cast<void *>(start + page_size)) ==
                  PAGE_NOACCESS &&
              tracker.IsRegionCpuModified(before, page_size) &&
              tracker.IsRegionCpuModified(after, page_size) &&
              !tracker.IsRegionGpuModified(before, page_size) &&
              !tracker.IsRegionGpuModified(after, page_size) &&
              IsWritable(reinterpret_cast<void *>(before)) &&
              IsWritable(reinterpret_cast<void *>(after)),
          "clean written upload lost GPU ownership or changed dirty neighbors");
    tracker.UnmarkRegionAsGpuModified(start, 2 * page_size);
  }
  tracker.UntrackMemory(address, region_size * 2);
  Release(memory);
}

void BenchmarkCleanUploads() {
  constexpr uint64_t size = 256ull * 1024 * 1024;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = AllocateFixedGuestRange(size, 0);
  const auto address = reinterpret_cast<uint64_t>(memory);
  tracker.ForEachUploadRange(address, size, false,
                             [](uint64_t, uint64_t) noexcept {},
                             []() noexcept {});
  std::puts("chunk_bytes,sweeps,calls,elapsed_ns,ns_per_sweep,ns_per_call");
  for (const uint64_t chunk : {size, Libs::Graphics::TRACKER_REGION_SIZE,
                               uint64_t{64 * 1024}}) {
    uint64_t ranges = 0;
    uint64_t completions = 0;
    uint64_t sweeps = 0;
    ResetProtectionLog();
    const auto start = std::chrono::steady_clock::now();
    std::chrono::nanoseconds elapsed{};
    do {
      for (uint64_t offset = 0; offset < size; offset += chunk) {
        tracker.ForEachUploadRange(
            address + offset, chunk, false,
            [&](uint64_t, uint64_t) noexcept { ranges++; },
            [&]() noexcept { completions++; });
      }
      sweeps++;
      elapsed = std::chrono::steady_clock::now() - start;
    } while (elapsed < std::chrono::milliseconds(250));
    Check(ranges == 0 && completions == sweeps * (size / chunk) &&
              g_protection_calls == 0,
          "clean upload benchmark changed ranges, completion, or protection");
    std::printf("%llu,%llu,%llu,%lld,%.2f,%.2f\n",
                static_cast<unsigned long long>(chunk),
                static_cast<unsigned long long>(sweeps),
                static_cast<unsigned long long>(completions),
                static_cast<long long>(elapsed.count()),
                static_cast<double>(elapsed.count()) / sweeps,
                static_cast<double>(elapsed.count()) / completions);
  }
  tracker.UntrackMemory(address, size);
  Release(memory);
}

void TestRangeInvalidation() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t size = Libs::Graphics::TRACKER_REGION_SIZE * 2;
  auto *memory = AllocateFixedGuestRange(size, 0x1000000);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, size) && !IsWritable(memory),
        "range invalidation setup did not establish GPU ownership");
  uint32_t flushes = 0;
  tracker.InvalidateRegion(address + 16, size - 32, [&] {
    flushes++;
    tracker.ForEachDownloadRange<true>(address + 16, size - 32,
                                       [](uint64_t, uint64_t) noexcept {});
    tracker.MarkRegionAsCpuModified(address + 16, size - 32);
  });
  Check(flushes == 1 && !tracker.IsRegionGpuModified(address, size) &&
            tracker.IsRegionCpuModified(address, size) && IsWritable(memory) &&
            IsWritable(memory + size - 1),
        "range invalidation did not batch ownership transfer across regions");
  tracker.InvalidateRegion(address + 16, size - 32, [&] { flushes++; });
  Check(flushes == 1,
        "clean range invalidation unnecessarily requested a GPU flush");
  tracker.UntrackMemory(address, size);
  Release(memory);
}

// A write fault's whole block (RenderContext::HandleFault): released at once
// when none of it is GPU-written, left as it was when a page of it is.
void TestReleaseForCpuWrite() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  constexpr uint64_t pages = 16;
  auto *memory = Allocate(page_manager, pages);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto size = page_size * pages;

  tracker.ForEachUploadRange(
      address, size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(!tracker.IsRegionCpuModified(address, size) && !IsWritable(memory) &&
            !IsWritable(memory + size - 1),
        "block release setup did not arm write tracking");
  Check(
      tracker.ReleaseForCpuWrite(address, size) &&
          tracker.IsRegionCpuModified(address, page_size) &&
          tracker.IsRegionCpuModified(address + size - page_size, page_size) &&
          IsWritable(memory) && IsWritable(memory + size - 1),
      "a clean block was not released whole");

  tracker.ForEachUploadRange(
      address, size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.ForEachUploadRange(
      address + page_size * 5, page_size, true,
      [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  Check(tracker.IsRegionGpuModified(address + page_size * 5, page_size) &&
            !IsWritable(memory),
        "block release setup did not take GPU ownership of one page");
  Check(!tracker.ReleaseForCpuWrite(address, size) &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            !IsWritable(memory) &&
            tracker.IsRegionGpuModified(address + page_size * 5, page_size),
        "a block holding a GPU-written page was released");

  tracker.UnmarkRegionAsGpuModified(address + page_size * 5, page_size);
  tracker.MarkRegionAsCpuModified(address, size);
  tracker.UntrackMemory(address, size);
  Release(memory);
}

void TestHotPages() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  constexpr uint64_t pages = 16;
  auto *memory = Allocate(page_manager, pages);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto size = page_size * pages;
  const auto hot = address + page_size * 2;
  std::vector<std::pair<uint64_t, uint64_t>> uploaded;
  const auto upload = [&](uint64_t vaddr, uint64_t bytes, bool written) {
    uploaded.clear();
    tracker.ForEachUploadRange(
        vaddr, bytes, written,
        [&](uint64_t range, uint64_t range_bytes) noexcept {
          uploaded.emplace_back(range, range_bytes);
        },
        []() noexcept {});
  };

  upload(address, size, false);
  Check(!IsWritable(memory + page_size * 2),
        "hot page setup did not arm write tracking");
  // A write fault on the page, which then becomes hot: it stays writable after
  // its upload.
  tracker.MarkRegionAsCpuModified(hot, page_size);
  Check(tracker.MarkHot(hot, page_size) == 1 && tracker.HotPages() == 1,
        "a CPU-dirty page did not become hot");
  upload(address, size, false);
  Check(uploaded.size() == 1 && uploaded[0].first == hot &&
            !tracker.IsRegionCpuModified(hot, page_size) &&
            IsWritable(memory + page_size * 2) &&
            !IsWritable(memory + page_size * 3),
        "a hot page was write-protected after its upload");
  // Each refresh makes it CPU-dirty again, without a protection change.
  ResetProtectionLog();
  Check(tracker.RedirtyHot() == 1 &&
            tracker.IsRegionCpuModified(hot, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            g_protection_calls == 0,
        "a refresh did not make the hot page CPU-dirty again on its own");
  Check(tracker.RedirtyHot() == 0,
        "a refresh counted a page that was CPU-dirty already");
  upload(address, size, false);
  Check(uploaded.size() == 1 && uploaded[0].first == hot &&
            IsWritable(memory + page_size * 2),
        "a refreshed hot page was not uploaded again");
  // Written by the GPU: the CPU may have written the clean hot page since its
  // upload, unseen, so it is uploaded first; then it is no longer hot, and
  // protected as GPU-written pages are.
  upload(hot, page_size, true);
  Check(uploaded.size() == 1 && uploaded[0].first == hot,
        "a GPU-written sync did not upload the clean hot page first");
  Check(tracker.IsRegionGpuModified(hot, page_size) &&
            !IsWritable(memory + page_size * 2) && tracker.RedirtyHot() == 0 &&
            tracker.HotPages() == 0,
        "a page the GPU wrote stayed hot");
  tracker.UnmarkRegionAsGpuModified(hot, page_size);
  Check(!IsWritable(memory + page_size * 2),
        "a page that stopped being hot became writable");
  // Cleared: the page may hold unseen writes, so it is CPU-dirty (uploaded
  // next) and write-protected by that upload.
  const auto other = address + page_size * 7;
  tracker.MarkRegionAsCpuModified(other, page_size);
  Check(tracker.MarkHot(other, page_size) == 1,
        "a second page did not become hot");
  upload(address, size, false);
  Check(IsWritable(memory + page_size * 7),
        "the second hot page was protected");
  tracker.ClearHot();
  Check(tracker.HotPages() == 0 &&
            tracker.IsRegionCpuModified(other, page_size) &&
            IsWritable(memory + page_size * 7) && tracker.RedirtyHot() == 0,
        "a cleared hot page was not left CPU-dirty");
  upload(address, size, false);
  Check(uploaded.size() == 1 && uploaded[0].first == other &&
            !IsWritable(memory + page_size * 7),
        "a cleared hot page was not uploaded and protected");
  // A store that does not fault (the GPU thread's own) makes a clean hot page
  // CPU-dirty.
  tracker.MarkRegionAsCpuModified(other, page_size);
  Check(tracker.MarkHot(other, page_size) == 1,
        "the page did not become hot again");
  upload(address, size, false);
  tracker.DirtyHotIn(address, size);
  Check(tracker.IsRegionCpuModified(other, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size),
        "a store to a hot page did not make it CPU-dirty");
  tracker.ClearHot();

  tracker.MarkRegionAsCpuModified(address, size);
  tracker.UntrackMemory(address, size);
  Release(memory);
}

void TestGpuReacquisitionAfterInvalidation() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size),
        "reacquisition setup did not establish GPU ownership");

  uint32_t flushes = 0;
  uint32_t uploads = 0;
  std::binary_semaphore reacquire{0};
  std::binary_semaphore reacquired{0};
  std::jthread publisher([&] {
    reacquire.acquire();
    tracker.ForEachUploadRange(
        address + 16, 32, true,
        [&](uint64_t, uint64_t) noexcept { uploads++; }, []() noexcept {});
    reacquired.release();
  });
  tracker.InvalidateRegion(address + 16, 32, [&] {
    flushes++;
    tracker.ForEachDownloadRange<true>(address + 16, 32,
                                       [](uint64_t, uint64_t) noexcept {});
    tracker.MarkRegionAsCpuModified(address + 16, 32);
    reacquire.release();
    reacquired.acquire();
  });
  publisher.join();
  Check(flushes == 1 && uploads == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            !IsWritable(memory),
        "invalidation rejected a new generation of GPU ownership");

  tracker.UnmarkRegionAsGpuModified(address, page_size);
  tracker.MarkRegionAsCpuModified(address, page_size);
  tracker.UntrackMemory(address, page_size);
  Release(memory);
}

void TestGpuDirtyBits() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionGpuModified(address + page_size, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "GPU dirty state escaped the requested range");
  tracker.UnmarkRegionAsGpuModified(address, page_size);
  Check(!tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "GPU dirty state did not restore write-only tracking");
  tracker.MarkRegionAsCpuModified(address, page_size);
  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

void TestExactDirtyIntervalsSharingTrackerPage() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  RangeSet exact_dirty;
  exact_dirty.Add(address + 64, 16);
  exact_dirty.Add(address + 192, 32);

  ResetProtectionLog();
  tracker.MarkRegionAsGpuModified(address + 64, 16);
  tracker.MarkRegionAsGpuModified(address + 192, 32);
  Check(g_protection_calls == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "disjoint byte dirtiness duplicated the page watcher");

  exact_dirty.Subtract(address + 64, 16);
  if (!exact_dirty.Intersects(address, page_size)) {
    tracker.UnmarkRegionAsGpuModified(address, page_size);
  }
  Check(g_protection_calls == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "draining one exact interval prematurely released its shared page");

  exact_dirty.Subtract(address + 192, 32);
  if (!exact_dirty.Intersects(address, page_size)) {
    tracker.UnmarkRegionAsGpuModified(address, page_size);
  }
  Check(g_protection_calls == 2 &&
            !tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "draining the final exact interval did not release its tracker page");

  tracker.UntrackMemory(address, page_size);
  Release(memory);
}

void TestGpuDownloadProtectionMirrors() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size * 4, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(address + 16, 32);
  tracker.MarkRegionAsGpuModified(address + page_size * 2 + 16, 32);

  std::vector<std::pair<uint64_t, uint64_t>> visited;
  ResetProtectionLog();
  tracker.ForEachDownloadRange<false>(
      address, page_size * 3,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        visited.push_back({range_address, range_size});
      });
  Check(visited.size() == 2 && visited[0].first == address &&
            visited[0].second == page_size &&
            visited[1].first == address + page_size * 2 &&
            visited[1].second == page_size && g_protection_calls == 0 &&
            tracker.IsRegionGpuModified(address, page_size * 3),
        "non-clearing download changed protection or lost sparse ranges");

  visited.clear();
  bool protected_during_download = false;
  tracker.ForEachDownloadRange<true>(
      address + 16, 32,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        protected_during_download = Protection(memory) == PAGE_NOACCESS;
        visited.push_back({range_address, range_size});
      });
  Check(protected_during_download && visited.size() == 1 &&
            visited[0].first == address &&
            visited[0].second == page_size && g_protection_log.size() == 1 &&
            g_protection_log[0].address == address &&
            g_protection_log[0].size == page_size &&
            g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
            !tracker.IsRegionGpuModified(address, page_size) &&
            tracker.IsRegionGpuModified(address + page_size * 2, page_size) &&
            Protection(memory) == PAGE_READONLY &&
            Protection(memory + page_size * 2) == PAGE_NOACCESS,
        "partial download did not preserve the CPU/GPU protection mirrors");

  visited.clear();
  ResetProtectionLog();
  tracker.ForEachDownloadRange<true>(
      address + 16, 32,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        visited.push_back({range_address, range_size});
      });
  Check(visited.empty() && g_protection_calls == 0 &&
            tracker.IsRegionGpuModified(address + page_size * 2, page_size),
        "idempotent partial download disturbed another GPU-owned page");

  tracker.UnmarkRegionAsGpuModified(address, page_size * 3);
  Check(!tracker.IsRegionGpuModified(address, page_size * 3) &&
            Protection(memory + page_size * 2) == PAGE_READONLY,
        "broad final unmark did not restore write-only tracking");
  ResetProtectionLog();
  tracker.MarkRegionAsCpuModified(address + 16, 32);
  Check(
      g_protection_log.size() == 1 && g_protection_log[0].address == address &&
          g_protection_log[0].size == page_size &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::ReadWrite &&
          IsWritable(memory) && !IsWritable(memory + page_size),
      "CPU-dirty transition did not release only its write watcher");

  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

void TestCrossRegionUpload() {
  constexpr uint64_t region_size = 4ull * 1024ull * 1024ull;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = AllocateFixedGuestRange(region_size * 2, 0x10000);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto boundary = (address + region_size - 1) & ~(region_size - 1);
  uint32_t ranges = 0;
  tracker.ForEachUploadRange(
      boundary - page_size, page_size * 2, false,
      [&](uint64_t, uint64_t) noexcept { ranges++; }, []() noexcept {});
  Check(ranges == 2 &&
            !tracker.IsRegionCpuModified(boundary - page_size, page_size * 2) &&
            !IsWritable(reinterpret_cast<void *>(boundary - page_size)) &&
            !IsWritable(reinterpret_cast<void *>(boundary)),
        "cross-region upload did not clear and protect both regions");
  tracker.MarkRegionAsCpuModified(boundary - page_size, page_size * 2);
  tracker.UntrackMemory(address, region_size * 2);
  Release(memory);
}

// IsRangeCpuCleanHint mirrors the regions' CPU summaries one bit per region; the read-only upload
// walk is skipped only when every region of the range exists and is clean.
void TestRegionBitmapHint() {
  // Region aligned (4 MiB), clear of the other tests' memory.
  constexpr uint64_t region_size = 4ull * 1024ull * 1024ull;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = AllocateFixedGuestRange(region_size * 2, 0x400000);
  const auto address = reinterpret_cast<uint64_t>(memory);
  Check(!tracker.IsRangeCpuCleanHint(address, region_size * 2),
        "regions that do not exist yet were reported clean");

  uint32_t ranges = 0;
  tracker.ForEachUploadRange(
      address, region_size * 2, false,
      [&](uint64_t, uint64_t) noexcept { ranges++; }, []() noexcept {});
  Check(ranges == 2 && tracker.IsRangeCpuCleanHint(address, region_size * 2) &&
            !tracker.IsRegionCpuModified(address, region_size * 2),
        "uploading two whole regions did not leave them clean in the hint");

  tracker.MarkRegionAsCpuModified(address + region_size + page_size, 16);
  Check(!tracker.IsRangeCpuCleanHint(address, region_size * 2) &&
            !tracker.IsRangeCpuCleanHint(address + region_size, page_size) &&
            tracker.IsRangeCpuCleanHint(address, region_size),
        "a CPU write did not clear the hint of its region only");

  ranges = 0;
  bool uploaded = false;
  tracker.ForEachUploadRange(
      address, region_size * 2, false,
      [&](uint64_t upload_address, uint64_t upload_size) noexcept {
        Check(upload_address == address + region_size + page_size &&
                  upload_size == page_size,
              "the dirty page was not the one uploaded");
        ranges++;
      },
      [&]() noexcept { uploaded = true; });
  Check(ranges == 1 && uploaded &&
            tracker.IsRangeCpuCleanHint(address, region_size * 2),
        "the upload walk missed the dirty page or left the hint dirty");

  Check(!tracker.IsRangeCpuCleanHint(address, region_size * 3),
        "a range reaching a region that was never created was reported clean");
  tracker.MarkRegionAsCpuModified(address, region_size * 2);
  tracker.UntrackMemory(address, region_size * 2);
  Release(memory);
}

void TestDirtyRegionWalk() {
  // A read-only upload over three existing regions that starts and ends inside them, with dirty
  // pages in the first and last region only (the bitmap-driven walk, KYTY_TRACKER_DIRTY_WALK):
  // exactly the dirty pages inside the range upload, and one just past its end stays dirty.
  constexpr uint64_t region_size = 4ull * 1024ull * 1024ull;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = AllocateFixedGuestRange(region_size * 3, 0x800000);
  const auto address = reinterpret_cast<uint64_t>(memory);
  tracker.ForEachUploadRange(address, region_size * 3, false,
                             [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  Check(tracker.IsRangeCpuCleanHint(address, region_size * 3),
        "the first upload did not leave the three regions clean");

  const auto begin = address + page_size * 3;
  const auto end = address + region_size * 2 + page_size * 5;
  tracker.MarkRegionAsCpuModified(address + page_size * 3, 16);
  tracker.MarkRegionAsCpuModified(address + region_size * 2 + page_size * 4, 16);
  tracker.MarkRegionAsCpuModified(address + region_size * 2 + page_size * 6, 16);
  uint32_t ranges = 0;
  bool first = false;
  bool last = false;
  tracker.ForEachUploadRange(
      begin, end - begin, false,
      [&](uint64_t upload_address, uint64_t upload_size) noexcept {
        Check(upload_size == page_size, "a dirty page uploaded with the wrong size");
        first |= upload_address == address + page_size * 3;
        last |= upload_address == address + region_size * 2 + page_size * 4;
        ranges++;
      },
      []() noexcept {});
  Check(ranges == 2 && first && last,
        "the dirty walk did not upload exactly the dirty pages inside the range");
  Check(tracker.IsRegionCpuModified(address + region_size * 2 + page_size * 6, 16) &&
            !tracker.IsRegionCpuModified(begin, end - begin),
        "the dirty walk cleared a page outside the range or left one inside dirty");

  tracker.MarkRegionAsCpuModified(address, region_size * 3);
  tracker.UntrackMemory(address, region_size * 3);
  Release(memory);
}

// The texture memo's zone stamps (round 78): a range's stamp moves when an image is registered
// or unregistered over it, and stays when that happens elsewhere.
void TestImageZoneStamps() {
  auto stamps = std::make_unique<Libs::Graphics::ImageZoneStamps>();
  constexpr uint64_t MB = 1ull << 20u;
  const uint64_t texture = 0x9'0000'0000ull + 3 * MB + 0x1000; // a small texture
  const uint64_t big = 0x12'0000'0000ull;                      // 32 MB: regions, not zones
  const auto t0 = stamps->RangeStamp(texture, 0x4000);
  const auto b0 = stamps->RangeStamp(big, 32 * MB);
  Check(t0 != 0, "a range has a stamp");
  Check(stamps->RangeStamp(texture, 0x4000) == t0, "a stamp stays while nothing changes");

  // A light grid in per-frame memory far away: neither range moves.
  stamps->Stamp(0x10'0400'0000ull, 0x1000);
  Check(stamps->RangeStamp(texture, 0x4000) == t0, "a change elsewhere leaves a range's stamp");
  Check(stamps->RangeStamp(big, 32 * MB) == b0, "a change elsewhere leaves a long range's stamp");

  // An image over the texture's zone (not over its bytes): the stamp moves.
  stamps->Stamp(texture + 0x8000, 0x1000);
  const auto t1 = stamps->RangeStamp(texture, 0x4000);
  Check(t1 != t0, "a change in the zone moves the stamp");
  Check(stamps->RangeStamp(texture, 0x4000) == t1, "the new stamp stays");

  // An image that covers the texture from far below it.
  stamps->Stamp(texture - 2 * MB, 4 * MB);
  Check(stamps->RangeStamp(texture, 0x4000) != t1, "a covering image moves the stamp");

  // A change inside the long range's regions moves it.
  stamps->Stamp(big + 20 * MB, 0x2000);
  Check(stamps->RangeStamp(big, 32 * MB) != b0, "a change in a long range's region moves it");

  // A huge image moves every stamp.
  const auto t2 = stamps->RangeStamp(texture, 0x4000);
  stamps->Stamp(0x20'0000'0000ull, 8192 * MB);
  Check(stamps->RangeStamp(texture, 0x4000) != t2, "a huge image moves every stamp");

  // A range too long to look at never keeps its stamp.
  const auto huge = stamps->RangeStamp(0x30'0000'0000ull, 65ull * 64 * MB);
  Check(stamps->RangeStamp(0x30'0000'0000ull, 65ull * 64 * MB) != huge,
        "a range of more than 64 regions has no stamp that stays");
}

void TestBdaPendingRegions() {
  // The BDA walk's pending regions (MemoryTracker::TakeBdaPending): a region is
  // pending when it is created, when pages of it become CPU-dirty and when
  // MarkBdaPending names it; a take empties them. CollectCpuDirtyRuns gives the
  // dirty pages as runs.
  constexpr uint64_t region_size = 4ull * 1024ull * 1024ull;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = AllocateFixedGuestRange(region_size * 3, 0x800000);
  const auto address = reinterpret_cast<uint64_t>(memory);
  std::vector<uint64_t> taken;
  const auto take = [&] {
    taken.clear();
    tracker.TakeBdaPending([&](uint64_t region) { taken.push_back(region); });
  };
  take();
  Check(taken.empty(), "a new tracker had pending regions");

  tracker.ForEachUploadRange(
      address, region_size * 3, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  take();
  Check(taken.size() == 3 && taken[0] == address &&
            taken[1] == address + region_size &&
            taken[2] == address + region_size * 2,
        "the regions an upload created were not pending, in order");
  take();
  Check(taken.empty(), "a take left regions pending");

  std::vector<std::pair<uint64_t, uint64_t>> runs;
  tracker.CollectCpuDirtyRuns(address, region_size * 3, runs);
  Check(runs.empty(), "clean regions had CPU-dirty runs");

  tracker.MarkRegionAsCpuModified(address + region_size * 2 + page_size * 7,
                                  16);
  tracker.MarkRegionAsCpuModified(address + region_size * 2 + page_size * 8,
                                  16);
  tracker.MarkRegionAsCpuModified(address + region_size * 2 + page_size * 20,
                                  16);
  take();
  Check(taken.size() == 1 && taken[0] == address + region_size * 2,
        "a CPU-dirty page did not make its region (only) pending");
  tracker.CollectCpuDirtyRuns(address, region_size * 3, runs);
  Check(runs.size() == 2 &&
            runs[0] ==
                std::pair<uint64_t, uint64_t>(
                    address + region_size * 2 + page_size * 7, page_size * 2) &&
            runs[1] ==
                std::pair<uint64_t, uint64_t>(
                    address + region_size * 2 + page_size * 20, page_size),
        "the CPU-dirty runs were not the dirty pages");
  runs.clear();
  tracker.CollectCpuDirtyRuns(address + region_size * 2 + page_size * 8,
                              page_size, runs);
  Check(runs.size() == 1 &&
            runs[0].first == address + region_size * 2 + page_size * 8 &&
            runs[0].second == page_size,
        "a run was not clipped to the range asked for");

  tracker.MarkBdaPending(address + region_size + 5, 1);
  take();
  Check(taken.size() == 1 && taken[0] == address + region_size,
        "MarkBdaPending did not make its region pending");
  tracker.MarkBdaPending(address + region_size - 1, 2);
  take();
  Check(taken.size() == 2 && taken[0] == address &&
            taken[1] == address + region_size,
        "MarkBdaPending across a region boundary did not make both pending");

  tracker.MarkRegionAsCpuModified(address, region_size * 3);
  tracker.UntrackMemory(address, region_size * 3);
  Release(memory);
}

void TestRedirtyForMap() {
  // A new buffer's regions (MemoryTracker::TrackRegions), and memory mapped
  // again (MemoryTracker::RedirtyForMap): its clean pages become
  // CPU-dirty, GPU-written ones stay as they are, and its region is pending
  // for the BDA walk again, also when its pages were CPU-dirty already.
  // DescribePages counts the pages by state.
  constexpr uint64_t region_size = 4ull * 1024ull * 1024ull;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  // The tracker's pages in a host page.
  const auto per = page_size / Libs::Graphics::TRACKER_PAGE_SIZE;
  auto *memory = AllocateFixedGuestRange(region_size, 0x400000);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto take = [&] {
    std::vector<uint64_t> taken;
    tracker.TakeBdaPending([&](uint64_t region) { taken.push_back(region); });
    return taken;
  };
  // TrackRegions (a new cached buffer's regions): created all CPU-dirty and
  // pending, once.
  Check(tracker.DescribePages(address, page_size).present == 0 &&
            tracker.TrackRegions(address, region_size) == 1,
        "TrackRegions did not create the missing region");
  Check(tracker.DescribePages(address, page_size * 8).cpu == 8 * per &&
            tracker.IsBdaPending(address),
        "a region TrackRegions created was not CPU-dirty and pending");
  Check(tracker.TrackRegions(address, region_size) == 0,
        "TrackRegions created a region that existed");
  tracker.ForEachUploadRange(
      address, page_size * 8, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  (void)take();
  auto states = tracker.DescribePages(address, page_size * 8);
  Check(states.present == 8 * per && states.cpu == 0 && states.gpu == 0 &&
            states.hot == 0,
        "uploaded pages were not described as clean");
  Check(!tracker.IsBdaPending(address), "a taken region was still pending");

  tracker.MarkRegionAsGpuModified(address + page_size * 2, page_size);
  const auto dirtied = tracker.RedirtyForMap(address, page_size * 8);
  states = tracker.DescribePages(address, page_size * 8);
  Check(dirtied == 7 * per && states.cpu == 7 * per && states.gpu == per,
        "a map did not make the clean pages (only) CPU-dirty");
  Check(tracker.IsBdaPending(address) && take().size() == 1,
        "a map did not make its region pending");
  Check(tracker.RedirtyForMap(address, page_size * 2) == 0 &&
            tracker.IsBdaPending(address),
        "a map over CPU-dirty pages did not make their region pending again");
  (void)take();

  tracker.UnmarkRegionAsGpuModified(address + page_size * 2, page_size);
  tracker.MarkRegionAsCpuModified(address, region_size);
  tracker.UntrackMemory(address, region_size);
  Release(memory);
}

void TestUploadDoesNotSerializeDisjointRegion() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto second_region =
      (allocation_base & ~(region_size - 1)) + region_size;
  Check(second_region + page_size <= allocation_base + region_size * 2,
        "test allocation does not span two tracker regions");

  // Publish both managers before the concurrent section so this test measures
  // tracker access serialization rather than manager allocation.
  Check(tracker.IsRegionCpuModified(allocation_base, page_size) &&
            tracker.IsRegionCpuModified(second_region, page_size),
        "disjoint upload setup did not initialize both regions");

  std::binary_semaphore upload_entered{0};
  std::binary_semaphore finish_upload{0};
  std::binary_semaphore query_finished{0};
  std::atomic_bool query_result{false};
  std::jthread uploader([&] {
    tracker.ForEachUploadRange(
        allocation_base, page_size, true, [](uint64_t, uint64_t) noexcept {},
        [&]() noexcept {
          upload_entered.release();
          finish_upload.acquire();
        });
  });
  upload_entered.acquire();
  std::jthread query([&] {
    query_result.store(tracker.IsRegionCpuModified(second_region, page_size),
                       std::memory_order_relaxed);
    query_finished.release();
  });

  const bool completed_while_upload_blocked =
      query_finished.try_acquire_for(std::chrono::seconds(5));
  finish_upload.release();
  uploader.join();
  query.join();

  tracker.UnmarkRegionAsGpuModified(allocation_base, page_size);
  tracker.MarkRegionAsCpuModified(allocation_base, page_size);
  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
  Check(completed_while_upload_blocked &&
            query_result.load(std::memory_order_relaxed),
        "upload callback serialized an unrelated tracker region");
}

void TestDownloadDoesNotSerializeDisjointRegion() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto second_region =
      (allocation_base & ~(region_size - 1)) + region_size;
  Check(second_region + page_size <= allocation_base + region_size * 2,
        "test allocation does not span two tracker regions");

  tracker.ForEachUploadRange(
      allocation_base, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.ForEachUploadRange(
      second_region, page_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});

  std::binary_semaphore download_entered{0};
  std::binary_semaphore finish_download{0};
  std::binary_semaphore mutation_finished{0};
  std::jthread downloader([&] {
    tracker.ForEachDownloadRange<false>(
        allocation_base, second_region + page_size - allocation_base,
        [&](uint64_t address, uint64_t) noexcept {
          if (address == allocation_base) {
            download_entered.release();
            finish_download.acquire();
          }
        });
  });
  download_entered.acquire();
  std::jthread mutation([&] {
    tracker.MarkRegionAsGpuModified(second_region, page_size);
    mutation_finished.release();
  });

  const bool completed_while_download_blocked =
      mutation_finished.try_acquire_for(std::chrono::seconds(5));
  finish_download.release();
  downloader.join();
  mutation.join();

  const bool both_gpu_owned =
      tracker.IsRegionGpuModified(allocation_base, page_size) &&
      tracker.IsRegionGpuModified(second_region, page_size);
  tracker.UnmarkRegionAsGpuModified(allocation_base, page_size);
  tracker.UnmarkRegionAsGpuModified(second_region, page_size);
  tracker.MarkRegionAsCpuModified(allocation_base, page_size);
  tracker.MarkRegionAsCpuModified(second_region, page_size);
  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
  Check(completed_while_download_blocked && both_gpu_owned,
        "download callback serialized an unrelated tracker region");
}

void TestGpuUnmarkUsesRegionMask() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  Check(region_base + region_size + page_size <=
            allocation_base + region_size * 2,
        "test allocation does not span two complete tracker regions");

  const auto sparse_begin = region_base + page_size;
  tracker.ForEachUploadRange(
      sparse_begin, page_size * 3, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(sparse_begin, page_size);
  tracker.MarkRegionAsGpuModified(sparse_begin + page_size * 2, page_size);
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(sparse_begin, page_size * 3);
  Check(
      g_protection_calls == 1 && g_protection_log.size() == 1 &&
          g_protection_log[0].address == sparse_begin &&
          g_protection_log[0].size == page_size * 3 &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
          !tracker.IsRegionGpuModified(sparse_begin, page_size * 3) &&
          Protection(reinterpret_cast<void *>(sparse_begin)) == PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(sparse_begin + page_size)) ==
              PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(sparse_begin + page_size * 2)) ==
              PAGE_READONLY,
      "GPU unmark did not coalesce a sparse 4 MiB region mask");
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(sparse_begin, page_size * 3);
  Check(g_protection_calls == 0,
        "idempotent GPU unmark performed a protection call");

  const auto boundary = region_base + region_size;
  const auto cross_begin = boundary - page_size;
  tracker.ForEachUploadRange(
      cross_begin, page_size * 2, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(cross_begin, page_size * 2);
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(cross_begin, page_size * 2);
  Check(g_protection_calls == 2 && g_protection_log.size() == 2 &&
            g_protection_log[0].address == cross_begin &&
            g_protection_log[0].size == page_size &&
            g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
            g_protection_log[1].address == boundary &&
            g_protection_log[1].size == page_size &&
            g_protection_log[1].mode == Common::VirtualMemory::Mode::Read &&
            !tracker.IsRegionGpuModified(cross_begin, page_size * 2),
        "cross-region GPU unmark did not use one update per 4 MiB region");

  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
}

void TestFullRegionGpuUnmarkBatching() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  Check(region_base + region_size <= allocation_base + region_size * 2,
        "test allocation does not contain a complete tracker region");

  tracker.ForEachUploadRange(
      region_base, region_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(region_base, region_size);
  Check(tracker.IsRegionGpuModified(region_base, region_size) &&
            Protection(reinterpret_cast<void *>(region_base)) ==
                PAGE_NOACCESS &&
            Protection(reinterpret_cast<void *>(region_base + region_size -
                                                page_size)) == PAGE_NOACCESS,
        "full-region setup did not establish GPU read protection");

  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(region_base, region_size);
  Check(
      g_protection_log.size() == 1 &&
          g_protection_log[0].address == region_base &&
          g_protection_log[0].size == region_size &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
          !tracker.IsRegionGpuModified(region_base, region_size) &&
          Protection(reinterpret_cast<void *>(region_base)) == PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(region_base + region_size -
                                              page_size)) == PAGE_READONLY,
      "full-region GPU unmark did not use one exact 4 MiB protection request");

  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
}

[[noreturn]] void RunDeathCase(const char *name) {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);
  if (std::strcmp(name, "gpu-dirty-explicit-cpu") == 0) {
    tracker.ForEachUploadRange(
        address, page_size, true, [](uint64_t, uint64_t) noexcept {},
        []() noexcept {});
    tracker.MarkRegionAsCpuModified(address, page_size);
  } else if (std::strcmp(name, "reentrant-upload") == 0) {
    tracker.ForEachUploadRange(
        address, page_size, true, [](uint64_t, uint64_t) noexcept {},
        [&]() noexcept {
          (void)tracker.IsRegionCpuModified(address, page_size);
        });
  } else if (std::strcmp(name, "recursive-tracking-lock") == 0) {
    Libs::Graphics::TrackingSpinLock lock;
    lock.lock();
    lock.lock();
  } else if (std::strcmp(name, "non-owner-tracking-unlock") == 0) {
    Libs::Graphics::TrackingSpinLock lock;
    lock.lock();
    std::thread worker([&] { lock.unlock(); });
    worker.join();
  }
  std::_Exit(0x7f);
}

void CheckDeathCase(const char *name) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
  char path[MAX_PATH]{};
  Check(GetModuleFileNameA(nullptr, path, MAX_PATH) != 0,
        "GetModuleFileName failed");
  std::string command = std::string("\"") + path + "\" --death " + name;
  std::vector<char> mutable_command(command.begin(), command.end());
  mutable_command.push_back('\0');
  STARTUPINFOA startup{sizeof(startup)};
  PROCESS_INFORMATION process{};
  Check(CreateProcessA(nullptr, mutable_command.data(), nullptr, nullptr, FALSE,
                       CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                       &process) != 0,
        "CreateProcess failed");
  Check(WaitForSingleObject(process.hProcess, 10000) == WAIT_OBJECT_0,
        "MemoryTracker death test timed out");
  DWORD exit_code = 0;
  Check(
      GetExitCodeProcess(process.hProcess, &exit_code) != 0 &&
          (exit_code == 321 || exit_code == EXCEPTION_NONCONTINUABLE_EXCEPTION),
      "MemoryTracker death path used the wrong exit");
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
#else
#if defined(__APPLE__)
  std::vector<char> path(PATH_MAX);
  uint32_t path_size = static_cast<uint32_t>(path.size());
  if (_NSGetExecutablePath(path.data(), &path_size) != 0) {
    path.resize(path_size);
    Check(_NSGetExecutablePath(path.data(), &path_size) == 0,
          "_NSGetExecutablePath failed");
  }
#endif
  const pid_t pid = ::fork();
  Check(pid >= 0, "fork failed");
  if (pid == 0) {
#if defined(__APPLE__)
    ::execl(path.data(), "MemoryTrackerTests", "--death", name, nullptr);
#else
    ::execl("/proc/self/exe", "MemoryTrackerTests", "--death", name, nullptr);
#endif
    std::_Exit(0x7e);
  }
  int status = 0;
  Check(::waitpid(pid, &status, 0) == pid, "waitpid failed");
  const bool fatal_exit =
      WIFEXITED(status) && WEXITSTATUS(status) == (321 & 0xff);
  Check(fatal_exit || WIFSIGNALED(status),
        "MemoryTracker death path used the wrong exit");
#endif
}

void TestFatalPaths() {
  for (const char *name : {"gpu-dirty-explicit-cpu", "reentrant-upload",
                           "recursive-tracking-lock", "non-owner-tracking-unlock"}) {
    CheckDeathCase(name);
  }
}


// KYTY_NATIVE_MEM=1 (graphics/host_gpu/nativeMem.h): kept pages. A model of the
// game writing pages of an upload heap between publish points (refreshes)
// while uploads copy guest bytes to a GPU copy: after every upload the GPU copy
// of the uploaded range equals guest memory, with or without kept pages, and
// kept pages fault less. The hash is the bytes' own (XXH3 in the emulator).
uint64_t HashBytes(const uint8_t *bytes, size_t size) {
  uint64_t value = 1469598103934665603ull;
  for (size_t i = 0; i < size; i++) {
    value = (value ^ bytes[i]) * 1099511628211ull;
  }
  return value;
}

uint64_t RunKeptModel(bool keep, uint32_t seed, uint64_t *stale) {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  constexpr uint64_t pages = 64;
  auto *memory = Allocate(page_manager, pages);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto size = page_size * pages;
  std::vector<uint8_t> gpu(size, 0);
  std::memset(memory, 0, size);
  const auto upload = [&](uint64_t vaddr, uint64_t bytes) {
    tracker.ForEachUploadRange(
        vaddr, bytes, false,
        [&](uint64_t range, uint64_t range_bytes) noexcept {
          std::memcpy(gpu.data() + (range - address),
                      reinterpret_cast<const void *>(range), range_bytes);
        },
        []() noexcept {});
  };
  const auto hash = [&](uint64_t vaddr, uint64_t *value) {
    *value = HashBytes(reinterpret_cast<const uint8_t *>(vaddr), page_size);
    return true;
  };
  upload(address, size);
  std::mt19937 random(seed);
  uint64_t faults = 0;
  uint64_t frame = 0;
  // A ring of pages written in bursts: each burst writes a few bytes of a
  // window of pages, several times between publishes, as the game fills its
  // GPU pool.
  uint64_t head = 0;
  for (int step = 0; step < 6000; step++) {
    const auto op = random() % 10;
    if (op < 6) {
      const auto page = (head + random() % 4) % pages;
      const auto offset = random() % page_size;
      auto *byte = memory + page * page_size + offset;
      if (!IsWritable(byte)) {
        faults++;
        // The fault handler: the page becomes CPU-dirty (and writable), and
        // with keep, kept.
        const auto page_address = address + page * page_size;
        tracker.MarkRegionAsCpuModified(page_address, page_size);
        if (keep) {
          (void)tracker.MarkKept(page_address, page_size, 1024);
        }
        Check(IsWritable(byte), "a faulted page did not become writable");
      }
      *byte = static_cast<uint8_t>(random());
      if (random() % 8 == 0) {
        head = (head + 1) % pages;
      }
    } else {
      // A publish point (a slice every fourth one, else a wait), then uses.
      const bool slice = op == 6;
      if (keep) {
        const bool settle = slice && random() % 4 == 0;
        // The model's clock: a step is a microsecond; kept 300 unchanged.
        (void)tracker.RefreshKept(slice, settle, static_cast<uint64_t>(step), 300,
                                  hash);
      }
      if (slice) {
        frame++;
      }
      for (int use = 0; use < 3; use++) {
        const auto first = random() % pages;
        const auto count = 1 + random() % 8;
        const auto bytes = std::min<uint64_t>(count, pages - first) * page_size;
        const auto vaddr = address + first * page_size;
        upload(vaddr, bytes);
        if (std::memcmp(gpu.data() + first * page_size,
                        reinterpret_cast<const void *>(vaddr), bytes) != 0) {
          (*stale)++;
        }
      }
    }
  }
  tracker.ClearHot();
  tracker.MarkRegionAsCpuModified(address, size);
  tracker.UntrackMemory(address, size);
  Release(memory);
  return faults;
}

void TestKeptPages() {
  for (uint32_t seed = 1; seed <= 6; seed++) {
    uint64_t stale_tracked = 0;
    uint64_t stale_kept = 0;
    const auto tracked = RunKeptModel(false, seed, &stale_tracked);
    const auto kept = RunKeptModel(true, seed, &stale_kept);
    std::printf("MemoryTrackerTests: kept pages, seed %u: write faults %llu "
                "tracked, %llu kept; stale uploads %llu tracked, %llu kept\n",
                seed, static_cast<unsigned long long>(tracked),
                static_cast<unsigned long long>(kept),
                static_cast<unsigned long long>(stale_tracked),
                static_cast<unsigned long long>(stale_kept));
    Check(stale_tracked == 0 && stale_kept == 0,
          "an upload after a publish missed a CPU write");
    Check(kept < tracked, "kept pages did not fault less");
  }
}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
void *g_fault_stack = nullptr;
constexpr size_t FAULT_STACK_SIZE = 64 * 1024;
volatile sig_atomic_t g_stack_faults = 0;

bool HandleStackFault(const Common::HostException::ExceptionInfo &info) {
  using namespace Common::HostException;
  stack_t active_stack{};
  const auto fault_address = reinterpret_cast<uintptr_t>(g_fault_stack) +
                             FAULT_STACK_SIZE - sizeof(uintptr_t);
  if (info.type != ExceptionType::AccessViolation ||
      info.access_violation_type != AccessViolationType::Write ||
      info.access_violation_vaddr != fault_address ||
      ::sigaltstack(nullptr, &active_stack) != 0 ||
      (active_stack.ss_flags & SS_ONSTACK) == 0) {
    std::_Exit(1);
  }
  g_stack_faults = 1;
  return ::mprotect(g_fault_stack, FAULT_STACK_SIZE,
                    PROT_READ | PROT_WRITE) == 0;
}

// A stack write must fault before any signal frame can use the protected stack.
__attribute__((naked)) void WriteProtectedStack(void *) {
  asm volatile("mov %rsp, %rax\n"
               "mov %rdi, %rsp\n"
               "push %rax\n"
               "pop %rsp\n"
               "ret\n");
}

void TestFaultOnProtectedStack() {
  const pid_t pid = ::fork();
  Check(pid >= 0, "stack fault fork failed");
  if (pid == 0) {
    std::thread worker([] {
      Check(Common::HostException::InitializeThreadSignalStack(),
            "initialize thread signal stack failed");
      Check(Common::HostException::InstallHandler(HandleStackFault),
            "install stack fault handler failed");
      g_fault_stack = ::mmap(nullptr, FAULT_STACK_SIZE, PROT_READ,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      Check(g_fault_stack != MAP_FAILED, "allocate protected stack failed");
      WriteProtectedStack(static_cast<char *>(g_fault_stack) + FAULT_STACK_SIZE);
      Check(g_stack_faults == 1, "protected stack write did not resume");
      struct sigaction action{};
      Check(::sigaction(SIGSEGV, nullptr, &action) == 0 &&
                action.sa_handler != SIG_DFL,
            "stack fault reset the process handler");
      Check(::munmap(g_fault_stack, FAULT_STACK_SIZE) == 0,
            "release protected stack failed");
    });
    worker.join();
    std::_Exit(0);
  }
  int status = 0;
  Check(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
            WEXITSTATUS(status) == 0,
        "fault on a protected stack did not recover");
}
#endif

} // namespace

namespace Libs::LibKernel::Memory {

bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size,
                            Common::VirtualMemory::Mode mode) {
  return ProtectAddressSpace(vaddr, size, mode);
}

} // namespace Libs::LibKernel::Memory

int main(int argc, char **argv) {
  if (argc == 3 && std::strcmp(argv[1], "--death") == 0) {
    RunDeathCase(argv[2]);
  }
  if (argc == 2 && std::strcmp(argv[1], "--benchmark-clean-upload") == 0) {
    BenchmarkCleanUploads();
    return 0;
  }
  TestGuestRange();
  TestRangeSet();
  TestPagedRangeSet();
  TestCpuDirtySpans();
  TestQueriesDoNotRequireMappedOwnership();
  TestConcurrentRegionPublication();
  TestCpuDirtyUpload();
  TestCleanUploadPreservesOwnership();
  TestRangeInvalidation();
  TestReleaseForCpuWrite();
  TestHotPages();
  TestKeptPages();
  TestGpuReacquisitionAfterInvalidation();
  TestGpuDirtyBits();
  TestExactDirtyIntervalsSharingTrackerPage();
  TestGpuDownloadProtectionMirrors();
  TestCrossRegionUpload();
  TestRegionBitmapHint();
  TestDirtyRegionWalk();
  TestBdaPendingRegions();
  TestRedirtyForMap();
  TestImageZoneStamps();
  TestUploadDoesNotSerializeDisjointRegion();
  TestDownloadDoesNotSerializeDisjointRegion();
  TestGpuUnmarkUsesRegionMask();
  TestFullRegionGpuUnmarkBatching();
  TestFatalPaths();
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
  TestFaultOnProtectedStack();
#endif
  std::puts("MemoryTrackerTests: all cases passed");
  return 0;
}
