#include "common/hostException.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/preflushTable.h"
#include "graphics/host_gpu/rangeSet.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <semaphore>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
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
std::mutex g_protection_log_mutex;

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
  {
    std::lock_guard lock(g_protection_log_mutex);
    g_protection_calls++;
    g_protection_log.push_back({vaddr, size, mode});
  }
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

void MarkGpuModified(MemoryTracker &tracker, uint64_t address, uint64_t size) {
  tracker.ForEachUploadRange(
      address, size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
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

void TestResolveWriteBitmap() {
  // A binding of six pages whose bitmap starts 0x80 before it (the descriptor's alignment); the
  // dispatch wrote its third bitmap page only.
  constexpr uint64_t base = 0x10000 - 0x80, vaddr = 0x10000, size = 0x6000;
  const std::array<uint32_t, 1> bits{1u << 2};
  RangeSet dirty, known, keep;
  dirty.Add(0xf000, 0x1000); // outside the binding
  dirty.Add(vaddr, size);
  known.Add(0x14100, 0x100); // an untracked writer's bytes
  keep.Add(0x15000, 0x800);  // a later tracked dispatch's binding
  std::vector<std::pair<uint64_t, uint64_t>> cleared, left, written;
  Libs::Graphics::ResolveWriteBitmap(dirty, known, keep, base, vaddr, size, bits,
                                     [&](uint64_t begin, uint64_t end) { cleared.emplace_back(begin, end); });
  dirty.ForEach([&](uint64_t begin, uint64_t end) { left.emplace_back(begin, end); });
  known.ForEach([&](uint64_t begin, uint64_t end) { written.emplace_back(begin, end); });
  using Ranges = std::vector<std::pair<uint64_t, uint64_t>>;
  Check(cleared == Ranges{{0x10000, 0x11f80}, {0x12f80, 0x14100}, {0x14200, 0x15000}, {0x15800, 0x16000}},
        "a write bitmap cleared the wrong bytes");
  Check(left == Ranges{{0xf000, 0x10000}, {0x11f80, 0x12f80}, {0x14100, 0x14200}, {0x15000, 0x15800}},
        "a write bitmap left the wrong bytes GPU-written");
  Check(written == Ranges{{0x11f80, 0x12f80}, {0x14100, 0x14200}},
        "a write bitmap did not make the bytes it showed known written");

  // keep's ranges outside the binding, touching its ends or not, change nothing (the buffer cache
  // passes only the later writes that overlap the binding).
  RangeSet dirty2, known2, keep2;
  dirty2.Add(0xf000, 0x1000);
  dirty2.Add(vaddr, size);
  known2.Add(0x14100, 0x100);
  keep2.Add(0x15000, 0x800);
  keep2.Add(0x8000, vaddr - 0x8000);  // ends where the binding starts
  keep2.Add(vaddr + size, 0x2000);    // starts where it ends
  keep2.Add(0x40000, 0x1000);         // far away
  std::vector<std::pair<uint64_t, uint64_t>> cleared2, left2, written2;
  Libs::Graphics::ResolveWriteBitmap(dirty2, known2, keep2, base, vaddr, size, bits,
                                     [&](uint64_t begin, uint64_t end) { cleared2.emplace_back(begin, end); });
  dirty2.ForEach([&](uint64_t begin, uint64_t end) { left2.emplace_back(begin, end); });
  known2.ForEach([&](uint64_t begin, uint64_t end) { written2.emplace_back(begin, end); });
  Check(cleared2 == cleared && left2 == left && written2 == written,
        "keep's ranges outside the binding changed a write bitmap's result");

  // Runs across words: bitmap pages 0, 33-35 and 95 of a 96-page binding were written.
  constexpr uint64_t wide = 96 * 0x1000;
  const std::array<uint32_t, 3> wide_bits{1u, 0b111u << 1, 1u << 31};
  RangeSet wide_dirty, wide_known;
  wide_dirty.Add(0x100000, wide);
  cleared.clear();
  written.clear();
  Libs::Graphics::ResolveWriteBitmap(wide_dirty, wide_known, {}, 0x100000, 0x100000, wide, wide_bits,
                                     [&](uint64_t begin, uint64_t end) { cleared.emplace_back(begin, end); });
  wide_known.ForEach([&](uint64_t begin, uint64_t end) { written.emplace_back(begin, end); });
  Check(cleared == Ranges{{0x101000, 0x121000}, {0x124000, 0x15f000}},
        "a write bitmap's runs across words cleared the wrong bytes");
  Check(written == Ranges{{0x100000, 0x101000}, {0x121000, 0x124000}, {0x15f000, 0x160000}},
        "a write bitmap's runs across words made the wrong bytes known written");
}

// Maximal runs of the set bytes of a byte model, from `origin`.
std::vector<std::pair<uint64_t, uint64_t>> Runs(const std::vector<uint8_t> &model, uint64_t origin) {
  std::vector<std::pair<uint64_t, uint64_t>> runs;
  for (size_t i = 0; i < model.size();) {
    if (model[i] == 0u) {
      i++;
      continue;
    }
    auto end = i;
    while (end < model.size() && model[end] != 0u) end++;
    runs.emplace_back(origin + i, origin + end);
    i = end;
  }
  return runs;
}

std::vector<std::pair<uint64_t, uint64_t>> Runs(const RangeSet &set) {
  std::vector<std::pair<uint64_t, uint64_t>> runs;
  set.ForEach([&](uint64_t begin, uint64_t end) { runs.emplace_back(begin, end); });
  return runs;
}

uint64_t NextRandom(uint64_t &state) {
  state ^= state << 13u;
  state ^= state >> 7u;
  state ^= state << 17u;
  return state;
}

// Add and Subtract against a byte model: Subtract splits ranges in place.
void TestRangeSetModel() {
  constexpr uint64_t origin = 0x10000, span = 2048;
  uint64_t state = 0x9e3779b97f4a7c15ull;
  RangeSet set;
  std::vector<uint8_t> model(span, 0u);
  for (int step = 0; step < 20000; step++) {
    const auto at = NextRandom(state) % span;
    const auto size = 1 + NextRandom(state) % std::min<uint64_t>(span - at, (step % 7 == 0) ? span : 64);
    const bool add = NextRandom(state) % 3 != 0;
    if (add) {
      set.Add(origin + at, size);
    } else {
      set.Subtract(origin + at, size);
    }
    std::fill(model.begin() + static_cast<ptrdiff_t>(at), model.begin() + static_cast<ptrdiff_t>(at + size),
              add ? 1u : 0u);
    Check(Runs(set) == Runs(model, origin), "range set differs from its byte model");
  }
}

// ResolveWriteBitmap against a byte model, with many small known ranges (a fragmented binding).
void TestResolveWriteBitmapModel() {
  constexpr uint64_t page = 0x1000, origin = 0x100000 - 0x8000, span = 0x48000;
  uint64_t state = 0x2545f4914f6cdd1dull;
  for (int round = 0; round < 3000; round++) {
    const uint64_t base = 0x100000 - (NextRandom(state) % 2) * 0x80;
    const uint64_t vaddr = 0x100000 + NextRandom(state) % 0x3000;
    const uint64_t size = 1 + NextRandom(state) % (40 * page - (vaddr - 0x100000));
    std::array<uint32_t, 2> bits{static_cast<uint32_t>(NextRandom(state)), static_cast<uint32_t>(NextRandom(state))};
    if (round % 4 == 0) bits[0] &= static_cast<uint32_t>(NextRandom(state)); // longer clear runs
    std::vector<uint8_t> dirty_model(span, 0u), known_model(span, 0u), keep_model(span, 0u);
    RangeSet dirty, known, keep;
    const auto add = [&](RangeSet &set, std::vector<uint8_t> &model, int count, uint64_t max_size) {
      for (int i = 0; i < count; i++) {
        const auto at = NextRandom(state) % (span - max_size);
        const auto bytes = 1 + NextRandom(state) % max_size;
        set.Add(origin + at, bytes);
        std::fill(model.begin() + static_cast<ptrdiff_t>(at), model.begin() + static_cast<ptrdiff_t>(at + bytes), 1u);
      }
    };
    add(dirty, dirty_model, 3, 0x30000);
    add(known, known_model, static_cast<int>(NextRandom(state) % 60), 0x800);
    add(keep, keep_model, static_cast<int>(NextRandom(state) % 4), 0x3000);
    std::vector<uint8_t> cleared_model(span, 0u);
    for (uint64_t address = vaddr; address < vaddr + size; address++) {
      const auto i = address - origin;
      const auto bit_index = (address - base) / page;
      const bool set = ((bits[bit_index / 32u] >> (bit_index % 32u)) & 1u) != 0u;
      if (dirty_model[i] == 0u) continue;
      if (set) {
        known_model[i] = 1u;
      } else if (known_model[i] == 0u && keep_model[i] == 0u) {
        dirty_model[i] = 0u;
        cleared_model[i] = 1u;
      }
    }
    std::vector<std::pair<uint64_t, uint64_t>> cleared;
    Libs::Graphics::ResolveWriteBitmap(dirty, known, keep, base, vaddr, size, bits,
                                       [&](uint64_t begin, uint64_t end) { cleared.emplace_back(begin, end); });
    Check(cleared == Runs(cleared_model, origin), "a write bitmap cleared other bytes than its byte model");
    Check(Runs(dirty) == Runs(dirty_model, origin), "a write bitmap left other GPU-written bytes than its byte model");
    Check(Runs(known) == Runs(known_model, origin), "a write bitmap made other bytes known than its byte model");
  }
}

// Kept runs in one call, across a region boundary: as the call per run, with no protection change.
void TestKeptGpuProtectionRuns() {
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  constexpr auto region = Libs::Graphics::TRACKER_REGION_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = Allocate(harness.page_manager, 3 * region / harness.page_manager.GetPageSize());
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto base = (allocation_base + region - 1) & ~(region - 1);
  const auto boundary = base + region;
  tracker.ForEachUploadRange(base, 2 * region, false, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  tracker.MarkRegionAsGpuModified(base, 2 * region);
  const std::vector<std::pair<uint64_t, uint64_t>> runs{{base + page_size, base + 3 * page_size},
                                                        {boundary - 3 * page_size, boundary + 2 * page_size},
                                                        {boundary + 5 * page_size, boundary + 6 * page_size}};
  ResetProtectionLog();
  tracker.KeepGpuProtection(runs);
  Check(g_protection_calls == 0, "keeping runs changed protection");
  const auto check_page = [&](uint64_t address) {
    const bool in_run = std::any_of(runs.begin(), runs.end(), [&](const auto &run) {
      return address >= run.first && address < run.second;
    });
    Check(tracker.IsRegionGpuModified(address, page_size) == !in_run, "kept runs left the wrong pages GPU-written");
  };
  for (uint64_t i = 0; i < 8; i++) check_page(base + i * page_size);
  for (uint64_t i = 0; i < 16; i++) check_page(boundary - 8 * page_size + i * page_size);
  Check(tracker.ReleaseKept(boundary + 5 * page_size, page_size) &&
            !tracker.ReleaseKept(boundary + 4 * page_size, page_size) &&
            tracker.ReleaseKept(boundary - 2 * page_size, page_size),
        "a page of a kept run was not kept, or one between runs was");
  tracker.UnmarkRegionAsGpuModified(base, 2 * region);
  tracker.UntrackMemory(allocation_base, 3 * region);
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

void TestPreflushCopyLimit() {
  // At most two copies between two read backs: a third write leaves the page missing until it is read back again.
  Libs::Graphics::PreflushTable table(4);
  uint32_t copies = 0;
  const auto take = [&](uint64_t tick) {
    table.TakeCopies(tick, [&](uint64_t, uint32_t) {
      copies++;
      return true;
    });
  };
  table.OnReadBack(7, 1);
  for (uint64_t tick = 1; tick <= 2; tick++) {
    table.OnWrite(7, 7, tick);
    take(tick);
    Check(table.Lookup(7) && table.Lookup(7)->second == tick, "a copy within the limit does not serve the page");
  }
  table.OnWrite(7, 7, 3);
  take(3);
  Check(copies == 2 && !table.Lookup(7), "a page past its copy limit hit with an older copy");
  table.OnReadBack(7, 1);
  table.OnWrite(7, 7, 4);
  take(4);
  Check(copies == 3 && table.Lookup(7) && table.Lookup(7)->second == 4, "a read back did not reset the copy limit");
}

void TestPreflushTable() {
  Libs::Graphics::PreflushTable table(2, 100);
  std::vector<std::pair<uint64_t, uint32_t>> copied;
  const auto take = [&](uint64_t tick, bool record = true) {
    copied.clear();
    table.TakeCopies(tick, [&](uint64_t page, uint32_t slot) {
      copied.emplace_back(page, slot);
      return record;
    });
  };
  const auto hit_tick = [&](uint64_t page) {
    const auto found = table.Lookup(page);
    return found ? found->second : 0;
  };

  // Pages never read back are not tracked.
  table.OnWrite(10, 10, 5);
  take(5);
  Check(copied.empty() && !table.Lookup(10), "a page with no read back was pre-flushed");

  // Read back, then written in command buffer 6: a miss until 6's end copies it.
  table.OnReadBack(10, 1);
  Check(!table.Lookup(10), "an unwritten page hit");
  table.OnWrite(9, 11, 6);
  Check(!table.Lookup(10), "a page written in the recording command buffer hit");
  take(6);
  Check(copied.size() == 1 && copied[0].first == 10, "the written page was not copied");
  Check(hit_tick(10) == 6, "the copy at the writer's end does not serve the page");
  take(7);
  Check(copied.empty(), "an unwritten page was copied again");

  // A write after the pre-copy: a miss until that command buffer's own copy.
  table.OnWrite(10, 10, 8);
  Check(!table.Lookup(10), "a page written after its pre-copy hit (stale slot)");
  take(8);
  Check(hit_tick(10) == 8, "the newer copy does not serve the page");

  // Two writes in one command buffer: one copy, after both.
  table.OnWrite(10, 10, 9);
  table.OnWrite(10, 10, 9);
  take(9);
  Check(copied.size() == 1 && hit_tick(10) == 9, "a page written twice in a command buffer");

  // A copy that could not be recorded leaves the page missing; a later command buffer's copy serves it.
  table.OnWrite(10, 10, 10);
  take(10, false);
  Check(!table.Lookup(10), "a page whose copy was not recorded hit");
  table.OnWrite(10, 10, 11);
  take(12);
  Check(hit_tick(10) == 12, "a later command buffer's copy does not serve the page");

  // A read back (or a publish from the slot) starts over.
  table.OnReadBack(10, 2);
  Check(!table.Lookup(10), "a page read back again hit without a new write");

  // A write in the command buffer that also reads the page back, after the read back: copied at its end.
  table.OnWrite(10, 10, 13);
  table.OnReadBack(10, 2);
  table.OnWrite(10, 10, 13);
  take(13);
  Check(hit_tick(10) == 13, "a write after a read back in one command buffer");

  // Slots run out: a third page is not tracked.
  table.OnReadBack(20, 2);
  table.OnReadBack(30, 2);
  Check(table.Pages() == 2, "a page took a slot beyond the limit");
  table.OnWrite(30, 30, 14);
  take(14);
  Check(copied.empty() && !table.Lookup(30), "an untracked page was pre-flushed");

  // Expiry: pages not read back in this frame or the previous one give their slots back; a page written before
  // it expired is not copied into a slot that another page has taken since.
  table.OnWrite(20, 20, 15);
  table.Expire(3);
  Check(table.Pages() == 2, "a page read back in the previous frame expired");
  table.Expire(4);
  Check(table.Pages() == 0 && !table.Lookup(10), "expired pages kept their slots");
  table.OnReadBack(40, 4);
  take(15);
  Check(copied.empty(), "a page that expired while pending was copied");
  Check(table.SlotsUsed() == 2, "an expired slot was not reused");
  table.OnWrite(40, 40, 16);
  take(16);
  Check(copied.size() == 1 && copied[0].first == 40 && hit_tick(40) == 16, "a reused slot does not serve its page");

  // A command buffer whose end was not seen: its writes count, and the next end copies them.
  table.OnReadBack(41, 4);
  table.OnWrite(40, 41, 17);
  table.OnWrite(50, 50, 18);
  Check(!table.Lookup(40) && !table.Lookup(41), "a page written in an unseen command buffer hit");
  take(18);
  Check(copied.size() == 2 && hit_tick(40) == 18 && hit_tick(41) == 18,
        "the next end did not copy the pages of an unseen command buffer");
}

void TestKnownZero() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto zeros = [&] {
    std::vector<std::pair<uint64_t, uint64_t>> runs;
    tracker.ForEachZeroRange(address, page_size * 4,
                             [&](uint64_t run, uint64_t bytes) {
                               runs.emplace_back(run, bytes);
                             });
    return runs;
  };

  // New memory is CPU-dirty and writable; known zero keeps it write-protected.
  tracker.MarkRegionAsZero(address, page_size * 4);
  Check(tracker.IsRegionCpuModified(address, page_size * 4) &&
            Protection(memory) == PAGE_READONLY &&
            Protection(memory + page_size * 3) == PAGE_READONLY,
        "known-zero CPU-dirty pages were not write-protected");

  // A CPU write (the fault's invalidation) on page 1 ends its zero state and
  // opens it; the others stay zero and protected.
  tracker.InvalidateRegion(address + page_size + 16, 32, [] {});
  Check(IsWritable(memory + page_size) && Protection(memory) == PAGE_READONLY &&
            zeros() == std::vector<std::pair<uint64_t, uint64_t>>{
                           {address, page_size},
                           {address + page_size * 2, page_size * 2}},
        "a CPU write did not end the zero state of only its page");

  // The upload reads page 1 and fills the zero runs.
  std::vector<std::tuple<uint64_t, uint64_t, bool>> runs;
  tracker.ForEachUploadRange(
      address, page_size * 4, false,
      [&](uint64_t run, uint64_t bytes, bool zero) noexcept {
        runs.emplace_back(run, bytes, zero);
      },
      []() noexcept {});
  Check(runs == std::vector<std::tuple<uint64_t, uint64_t, bool>>{
                    {address + page_size, page_size, false},
                    {address, page_size, true},
                    {address + page_size * 2, page_size * 2, true}} &&
            !tracker.IsRegionCpuModified(address, page_size * 4) &&
            Protection(memory + page_size) == PAGE_READONLY &&
            zeros().size() == 2,
        "upload did not split zero runs or lost the zero state");

  // A GPU write ends the zero state of its page.
  tracker.MarkRegionAsGpuModified(address + page_size * 2 + 16, 32);
  Check(zeros() == std::vector<std::pair<uint64_t, uint64_t>>{
                       {address, page_size}, {address + page_size * 3, page_size}},
        "a GPU write did not end the zero state");
  tracker.UnmarkRegionAsGpuModified(address + page_size * 2, page_size);

  // Unmarking (an alias appeared) releases the write protection a CPU-dirty
  // page kept only for the zero state.
  tracker.MarkRegionAsCpuModified(address + page_size * 3, page_size);
  tracker.MarkRegionAsZero(address + page_size * 3, page_size);
  Check(Protection(memory + page_size * 3) == PAGE_READONLY,
        "re-marking a CPU-dirty page zero did not protect it");
  tracker.UnmarkRegionAsZero(address, page_size * 4);
  Check(zeros().empty() && IsWritable(memory + page_size * 3) &&
            Protection(memory) == PAGE_READONLY,
        "unmarking did not end the zero state or changed clean pages");

  tracker.UntrackMemory(address, page_size * 4);
  Check(IsWritable(memory) && IsWritable(memory + page_size * 3),
        "untracking left protection");
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

std::vector<std::pair<uint64_t, uint64_t>>
ConsumeCpuWrites(MemoryTracker &tracker, uint64_t address, uint64_t size) {
  std::vector<std::pair<uint64_t, uint64_t>> runs;
  tracker.ConsumeCpuWrites(address, size, [&](uint64_t run, uint64_t bytes) {
    runs.emplace_back(run, bytes);
  });
  return runs;
}

void TestConsumeCpuWrites() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto upload = [&] {
    tracker.ForEachUploadRange(
        address, page_size * 4, false, [](uint64_t, uint64_t) noexcept {},
        []() noexcept {});
  };

  // A region the tracker never saw counts as written, clamped to the query.
  auto runs = ConsumeCpuWrites(tracker, address + 16, page_size * 4 - 32);
  Check(runs.size() == 1 && runs[0].first == address + 16 &&
            runs[0].second == page_size * 4 - 32,
        "untracked pages were not reported as CPU written");
  // Pages still CPU dirty keep being reported: writes to them do not fault.
  runs = ConsumeCpuWrites(tracker, address, page_size * 4);
  Check(runs.size() == 1 && runs[0].second == page_size * 4,
        "CPU-dirty pages were not reported again");
  upload();
  runs = ConsumeCpuWrites(tracker, address, page_size * 4);
  Check(runs.empty(), "clean pages were reported as CPU written");

  // A write after the upload is reported once the page is clean again.
  tracker.MarkRegionAsCpuModified(address + page_size, 8);
  upload();
  runs = ConsumeCpuWrites(tracker, address, page_size * 4);
  Check(runs.size() == 1 && runs[0].first == address + page_size &&
            runs[0].second == page_size,
        "a CPU write between uploads was not reported");
  runs = ConsumeCpuWrites(tracker, address, page_size * 4);
  Check(runs.empty(), "a consumed CPU write was reported twice");

  // A write outside the query stays pending for a later query over it.
  tracker.MarkRegionAsCpuModified(address + page_size * 3, 8);
  upload();
  runs = ConsumeCpuWrites(tracker, address, page_size * 2);
  Check(runs.empty(), "a CPU write outside the query was reported");
  runs = ConsumeCpuWrites(tracker, address, page_size * 4);
  Check(runs.size() == 1 && runs[0].first == address + page_size * 3,
        "a CPU write outside an earlier query was forgotten");
  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

void TestGpuMarkingSkipsCpuDirtyPages() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 3);
  const auto address = reinterpret_cast<uint64_t>(memory);
  tracker.ForEachUploadRange(
      address, page_size * 3, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  // The CPU wrote the middle page after the GPU work was recorded.
  tracker.MarkRegionAsCpuModified(address + page_size, 8);

  std::vector<std::pair<uint64_t, uint64_t>> marked;
  tracker.MarkRegionAsGpuModifiedUnlessCpuDirty(
      address + 16, page_size * 3 - 32, [&](uint64_t run, uint64_t bytes) {
        marked.emplace_back(run, bytes);
      });
  Check(marked.size() == 2 && marked[0].first == address &&
            marked[0].second == page_size &&
            marked[1].first == address + page_size * 2 &&
            marked[1].second == page_size,
        "GPU marking did not skip exactly the CPU-dirty page");
  Check(tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionGpuModified(address + page_size, page_size) &&
            tracker.IsRegionGpuModified(address + page_size * 2, page_size) &&
            tracker.IsRegionCpuModified(address + page_size, page_size),
        "GPU marking changed the wrong pages");
  Check(Protection(memory) == PAGE_NOACCESS && IsWritable(memory + page_size) &&
            Protection(memory + page_size * 2) == PAGE_NOACCESS,
        "GPU marking did not protect the marked pages only");

  marked.clear();
  tracker.MarkRegionAsGpuModifiedUnlessCpuDirty(
      address + page_size, page_size, [&](uint64_t run, uint64_t bytes) {
        marked.emplace_back(run, bytes);
      });
  Check(marked.empty(), "a CPU-dirty page alone was marked GPU modified");

  tracker.UnmarkRegionAsGpuModified(address, page_size * 3);
  tracker.UntrackMemory(address, page_size * 3);
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
  // CPU-dirtying the next page does not conflict with the GPU-dirty one.
  tracker.MarkRegionAsCpuModified(address + page_size, page_size);
  tracker.UnmarkRegionAsGpuModified(address, page_size);
  Check(!tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "GPU dirty state did not restore write-only tracking");
  // Nor does CPU-dirtying the previous page.
  tracker.ForEachUploadRange(
      address + page_size, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsCpuModified(address, page_size);
  tracker.UnmarkRegionAsGpuModified(address + page_size, page_size);
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
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  auto *memory = AllocateFixedGuestRange(region_size * 2, 0x10000);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto boundary = (address + region_size - 1) & ~(region_size - 1);
  for (const bool is_written : {false, true}) {
    TrackerHarness harness;
    auto &tracker = harness.tracker;
    uint32_t ranges = 0;
    tracker.ForEachUploadRange(
        boundary - page_size, page_size * 2, is_written,
        [&](uint64_t start, uint64_t size) noexcept {
          Check(start == boundary - page_size + ranges * page_size &&
                    size == page_size,
                "cold upload did not visit both region boundaries exactly");
          ranges++;
        },
        [&]() noexcept {
          Check(ranges == 2 &&
                    Protection(reinterpret_cast<void *>(boundary - page_size)) ==
                        PAGE_READONLY &&
                    Protection(reinterpret_cast<void *>(boundary)) == PAGE_READONLY,
                "cold upload completed before clearing and protecting both regions");
        });
    for (const auto page : {boundary - page_size, boundary}) {
      Check(!tracker.IsRegionCpuModified(page, page_size) &&
                tracker.IsRegionGpuModified(page, page_size) == is_written &&
                Protection(reinterpret_cast<void *>(page)) ==
                    (is_written ? PAGE_NOACCESS : PAGE_READONLY),
            "cross-region upload did not preserve final ownership and protection");
    }
    Check(IsWritable(reinterpret_cast<void *>(boundary - 2 * page_size)) &&
              IsWritable(reinterpret_cast<void *>(boundary + page_size)),
          "cross-region upload changed neighboring pages");
    tracker.UnmarkRegionAsGpuModified(boundary - page_size, 2 * page_size);
    tracker.UntrackMemory(address, region_size * 2);
  }
  Release(memory);
}

void TestConcurrentColdUploads() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = AllocateFixedGuestRange(region_size * 2, region_size);
  const auto address = reinterpret_cast<uint64_t>(memory);
  std::counting_semaphore<2> start{0};
  std::counting_semaphore<2> upload_entered{0};
  std::counting_semaphore<2> finish_upload{0};
  std::vector<std::jthread> workers;
  for (const auto page : {address, address + region_size}) {
    workers.emplace_back([&, page] {
      start.acquire();
      uint32_t ranges = 0;
      tracker.ForEachUploadRange(
          page, page_size, true,
          [&](uint64_t upload_address, uint64_t upload_size) noexcept {
            Check(upload_address == page && upload_size == page_size,
                  "concurrent cold upload lost its dirty page");
            ranges++;
          },
          [&]() noexcept {
            Check(ranges == 1, "concurrent cold upload skipped its dirty page");
            upload_entered.release();
            finish_upload.acquire();
          });
    });
  }
  start.release(2);
  const bool first_entered = upload_entered.try_acquire_for(std::chrono::seconds(5));
  const bool second_entered = upload_entered.try_acquire_for(std::chrono::seconds(5));
  finish_upload.release(2);
  for (auto &worker : workers) {
    worker.join();
  }
  Check(first_entered && second_entered,
        "cold writable uploads serialized disjoint regions");
  for (const auto page : {address, address + region_size}) {
    Check(!tracker.IsRegionCpuModified(page, page_size) &&
              tracker.IsRegionGpuModified(page, page_size) &&
              Protection(reinterpret_cast<void *>(page)) == PAGE_NOACCESS,
          "concurrent cold upload did not retain GPU ownership");
    tracker.UnmarkRegionAsGpuModified(page, page_size);
  }
  tracker.UntrackMemory(address, region_size * 2);
  Release(memory);
}

// Runs probe on a thread while an upload of the page at address is parked in
// its callback, holding that region's lock, and returns whether the probe
// finished within 5 s.
template <typename Probe>
bool RunDuringParkedUpload(MemoryTracker &tracker, uint64_t address,
                           Probe &&probe) {
  std::binary_semaphore upload_entered{0};
  std::binary_semaphore finish_upload{0};
  std::binary_semaphore probe_finished{0};
  std::jthread uploader([&] {
    tracker.ForEachUploadRange(
        address, Libs::Graphics::TRACKER_PAGE_SIZE, true,
        [](uint64_t, uint64_t) noexcept {},
        [&]() noexcept {
          upload_entered.release();
          finish_upload.acquire();
        });
  });
  upload_entered.acquire();
  std::jthread worker([&] {
    probe();
    probe_finished.release();
  });
  const bool finished = probe_finished.try_acquire_for(std::chrono::seconds(5));
  finish_upload.release();
  uploader.join();
  worker.join();
  return finished;
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

  const bool completed_while_upload_blocked =
      RunDuringParkedUpload(tracker, allocation_base, [&] {
        MarkGpuModified(tracker, second_region, page_size);
      });
  const bool second_marked =
      tracker.IsRegionGpuModified(second_region, page_size);

  tracker.UnmarkRegionAsGpuModified(allocation_base, page_size);
  tracker.UnmarkRegionAsGpuModified(second_region, page_size);
  tracker.MarkRegionAsCpuModified(allocation_base, page_size);
  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
  Check(completed_while_upload_blocked && second_marked,
        "upload callback serialized an unrelated tracker region");
}

void TestQueriesSkipRegionLock() {
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = Allocate(harness.page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  bool cpu_modified = true;
  bool gpu_modified = true;
  // The upload holds the region lock until its writes are marked.
  const bool completed_while_upload_blocked =
      RunDuringParkedUpload(tracker, address, [&] {
        cpu_modified = tracker.IsRegionCpuModified(address, page_size);
        gpu_modified = tracker.IsRegionGpuModified(address, page_size);
      });

  const bool marked = tracker.IsRegionGpuModified(address, page_size);
  tracker.UnmarkRegionAsGpuModified(address, page_size);
  tracker.MarkRegionAsCpuModified(address, page_size);
  tracker.UntrackMemory(address, page_size);
  Release(memory);
  Check(completed_while_upload_blocked && !cpu_modified && !gpu_modified &&
            marked,
        "dirty queries waited for an upload holding the region lock");
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

// A tracked dispatch's unwritten pages stay read-protected after its resolve, so that the next
// dispatch's mark changes no protection; the CPU's first touch lifts them in one change, and pages
// the CPU has touched are kept again at later resolves, so its next touch faults again.
void TestKeptGpuProtection() {
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  constexpr uint64_t window = 0x40000;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = Allocate(harness.page_manager, 3 * window / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto base = (allocation_base + window - 1) & ~(window - 1);
  const auto next = base + window;
  const auto *page = reinterpret_cast<const void *>(base + page_size);
  const auto at = [](uint64_t address) { return reinterpret_cast<const void *>(address); };

  tracker.ForEachUploadRange(base, 2 * window, false, [](uint64_t, uint64_t) noexcept {},
                             []() noexcept {});
  tracker.MarkRegionAsGpuModified(base, window);
  ResetProtectionLog();
  tracker.KeepGpuProtection(base, window);
  Check(g_protection_calls == 0 && !tracker.IsRegionGpuModified(base, window) &&
            Protection(page) == PAGE_NOACCESS,
        "kept pages changed protection or stayed GPU-written");
  tracker.MarkRegionAsGpuModified(base, window);
  Check(g_protection_calls == 0 && tracker.IsRegionGpuModified(base, window),
        "marking kept pages again changed their protection");

  // A CPU write on a kept page: read-write in one change, no read back, the others stay kept.
  tracker.KeepGpuProtection(base, window);
  bool flushed = false;
  ResetProtectionLog();
  tracker.InvalidateRegion(base + page_size, page_size, [&] { flushed = true; });
  Check(!flushed && g_protection_calls == 1 &&
            g_protection_log[0].mode == Common::VirtualMemory::Mode::ReadWrite && IsWritable(page) &&
            Protection(at(base)) == PAGE_NOACCESS && tracker.IsRegionCpuModified(base + page_size, page_size),
        "a CPU write on a kept page did not make it read-write in one change");

  // A read fault on a kept page lifts the kept pages of its window (the written page is a run of its own).
  ResetProtectionLog();
  Check(tracker.ReleaseKept(base + 5 * page_size + 8, window) && g_protection_calls == 2 &&
            Protection(at(base)) == PAGE_READONLY && IsWritable(page) &&
            Protection(at(base + window - page_size)) == PAGE_READONLY,
        "a fault on a kept page did not lift its window");
  ResetProtectionLog();
  Check(!tracker.ReleaseKept(base + 5 * page_size, window) && g_protection_calls == 0,
        "a page that is not kept was released");

  // The range release (before a copy under the address-space lock) lifts only what is kept.
  tracker.MarkRegionAsGpuModified(next, 2 * page_size);
  tracker.KeepGpuProtection(next, page_size);
  ResetProtectionLog();
  tracker.ReleaseKeptRange(next, 4 * page_size);
  Check(g_protection_calls == 1 && Protection(at(next)) == PAGE_READONLY &&
            Protection(at(next + page_size)) == PAGE_NOACCESS && tracker.IsRegionGpuModified(next + page_size, page_size),
        "the range release lifted a GPU-written page or missed a kept one");
  ResetProtectionLog();
  tracker.ReleaseKeptRange(next, 4 * page_size);
  Check(g_protection_calls == 0, "a range release with nothing kept changed protection");
  tracker.UnmarkRegionAsGpuModified(next + page_size, page_size);

  // The CPU touched every page of the first window while it was kept: the next resolve keeps them
  // with no protection change, and the next touch lifts them again.
  tracker.ForEachUploadRange(base, window, false, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  tracker.MarkRegionAsGpuModified(base, window);
  ResetProtectionLog();
  tracker.KeepGpuProtection(base, window);
  Check(g_protection_calls == 0 && !tracker.IsRegionGpuModified(base, window) &&
            Protection(at(base)) == PAGE_NOACCESS && Protection(at(base + window - page_size)) == PAGE_NOACCESS,
        "pages the CPU touched were not kept again");
  Check(tracker.ReleaseKept(base + 8, window) && Protection(at(base)) == PAGE_READONLY &&
            Protection(at(base + window - page_size)) == PAGE_READONLY,
        "a fault on a page kept again did not lift its window");

  tracker.UntrackMemory(allocation_base, 3 * window);
  Check(IsWritable(at(base)) && IsWritable(at(next)), "untracking left a kept page protected");
  Release(memory);
}

// A written binding marked again over pages that are already GPU-written keeps exactly their protection (no call at all),
// and a CPU write after the GPU write still faults: the page stays no-access, the fault reads the GPU bytes back, and the
// write is then uploaded. A lift window narrower than 256 KB lifts only the kept pages inside it: a read of a kept page
// past it still faults.
void TestGpuRemarkKeepsProtection() {
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  constexpr uint64_t window = 0x40000;
  constexpr uint64_t narrow = 0x4000;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = Allocate(harness.page_manager, 2 * window / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto base = (allocation_base + window - 1) & ~(window - 1);
  const auto at = [](uint64_t address) { return reinterpret_cast<const void *>(address); };
  const auto all_no_access = [&](uint64_t begin, uint64_t end) {
    for (auto address = begin; address < end; address += page_size) {
      if (Protection(at(address)) != PAGE_NOACCESS) return false;
    }
    return true;
  };

  tracker.ForEachUploadRange(base, window, false, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  MarkGpuModified(tracker, base, 4 * page_size);
  Check(all_no_access(base, base + 4 * page_size) && Protection(at(base + 4 * page_size)) == PAGE_READONLY,
        "a written binding did not protect exactly its pages");
  ResetProtectionLog();
  MarkGpuModified(tracker, base, 4 * page_size);
  MarkGpuModified(tracker, base + page_size, 2 * page_size);
  MarkGpuModified(tracker, base + 8, 16);
  Check(g_protection_calls == 0 && all_no_access(base, base + 4 * page_size) &&
            tracker.IsRegionGpuModified(base, 4 * page_size) && Protection(at(base + 4 * page_size)) == PAGE_READONLY,
        "marking GPU-written pages again changed their protection");

  // The CPU writes page 1 after the GPU did: the page isn't writable, so the write faults. The fault's invalidation
  // reads it back (flush), the page becomes CPU-written, and the next upload walk reports it.
  Check(!IsWritable(at(base + page_size)), "a CPU write after the GPU write would not fault");
  uint32_t flushes = 0;
  tracker.InvalidateRegion(base + page_size, page_size, [&] {
    flushes++;
    tracker.ForEachDownloadRange<true>(base + page_size, page_size, [](uint64_t, uint64_t) noexcept {});
    tracker.MarkRegionAsCpuModified(base + page_size, page_size);
  });
  std::vector<std::pair<uint64_t, uint64_t>> uploads;
  tracker.ForEachUploadRange(
      base, 4 * page_size, false,
      [&](uint64_t address, uint64_t size) noexcept { uploads.emplace_back(address, size); }, []() noexcept {});
  Check(flushes == 1 && uploads.size() == 1 && uploads[0].first == base + page_size &&
            uploads[0].second == page_size && tracker.IsRegionGpuModified(base, page_size) &&
            tracker.IsRegionGpuModified(base + 2 * page_size, 2 * page_size) &&
            !tracker.IsRegionGpuModified(base + page_size, page_size) && Protection(at(base)) == PAGE_NOACCESS,
        "a CPU write after the GPU write was not read back first and then uploaded");
  tracker.UnmarkRegionAsGpuModified(base, 4 * page_size);

  // A narrow lift: a read fault lifts only the kept pages of its 16 KB; one past it still faults and lifts its own.
  MarkGpuModified(tracker, base, window);
  tracker.KeepGpuProtection(base, window);
  ResetProtectionLog();
  Check(tracker.ReleaseKept(base + 5 * page_size + 8, narrow) && g_protection_calls == 1 &&
            Protection(at(base + 4 * page_size)) == PAGE_READONLY &&
            Protection(at(base + 7 * page_size)) == PAGE_READONLY && all_no_access(base, base + 4 * page_size) &&
            all_no_access(base + 8 * page_size, base + window),
        "a narrow lift lifted other pages than its window");
  Check(!tracker.ReleaseKept(base + 6 * page_size, narrow) &&
            tracker.ReleaseKept(base + 9 * page_size, narrow) &&
            Protection(at(base + 8 * page_size)) == PAGE_READONLY && all_no_access(base + 12 * page_size, base + window),
        "a read past a narrow lift did not fault on its own window");

  tracker.UntrackMemory(allocation_base, 2 * window);
  Check(IsWritable(at(base)), "untracking left a page protected");
  Release(memory);
}

// A range is CPU-clean (no upload walk needed) only while every region of it is tracked and no page of it is CPU-dirty:
// a CPU write anywhere in a region ends it for the whole region, the next upload walk restores it.
void TestRangeCpuClean() {
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  constexpr auto region = Libs::Graphics::TRACKER_REGION_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = Allocate(harness.page_manager, 3 * region / harness.page_manager.GetPageSize());
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto base = (allocation_base + region - 1) & ~(region - 1);
  const auto walk = [&](uint64_t address, uint64_t size) {
    uint32_t ranges = 0;
    tracker.ForEachUploadRange(address, size, false, [&](uint64_t, uint64_t) noexcept { ranges++; }, []() noexcept {});
    return ranges;
  };
  Check(!tracker.IsRangeCpuClean(base, page_size), "an untracked region was clean");
  Check(walk(base, 2 * region) != 0 && tracker.IsRangeCpuClean(base, 2 * region) &&
            tracker.IsRangeCpuClean(base + region - page_size, 2 * page_size),
        "a walked range was not clean");
  Check(!tracker.IsRangeCpuClean(base + region, region + page_size), "a range into an untracked region was clean");

  // A CPU write on the last page of the first region: the region is no longer clean, the second one still is, and a
  // walk of the first one's other pages (which finds nothing) leaves it dirty.
  tracker.MarkRegionAsCpuModified(base + region - page_size, 16);
  Check(!tracker.IsRangeCpuClean(base, page_size) && !tracker.IsRangeCpuClean(base, 2 * region) &&
            tracker.IsRangeCpuClean(base + region, region),
        "a CPU write did not end its region's clean state, or ended another's");
  Check(walk(base, page_size) == 0 && !tracker.IsRangeCpuClean(base, page_size),
        "a walk that missed the dirty page made its region clean");
  Check(walk(base, region) == 1 && tracker.IsRangeCpuClean(base, 2 * region),
        "the walk that uploaded the dirty page did not make its region clean");

  // A write fault on a GPU-written page (the invalidation's flush marks it CPU-written) ends the clean state too.
  MarkGpuModified(tracker, base + page_size, page_size);
  Check(tracker.IsRangeCpuClean(base, region), "a GPU write ended the CPU-clean state");
  tracker.InvalidateRegion(base + page_size, page_size, [&] {
    tracker.ForEachDownloadRange<true>(base + page_size, page_size, [](uint64_t, uint64_t) noexcept {});
    tracker.MarkRegionAsCpuModified(base + page_size, page_size);
  });
  Check(!tracker.IsRangeCpuClean(base, region) && walk(base, region) == 1 && tracker.IsRangeCpuClean(base, region),
        "a CPU write after a GPU write was not seen by the clean state");

  tracker.UntrackMemory(allocation_base, 3 * region);
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
          tracker.MarkRegionAsCpuModified(address, page_size);
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
  TestResolveWriteBitmap();
  TestRangeSetModel();
  TestResolveWriteBitmapModel();
  TestQueriesDoNotRequireMappedOwnership();
  TestConcurrentRegionPublication();
  TestCpuDirtyUpload();
  TestKnownZero();
  TestPreflushTable();
  TestPreflushCopyLimit();
  TestCleanUploadPreservesOwnership();
  TestConsumeCpuWrites();
  TestGpuMarkingSkipsCpuDirtyPages();
  TestRangeInvalidation();
  TestGpuReacquisitionAfterInvalidation();
  TestGpuDirtyBits();
  TestExactDirtyIntervalsSharingTrackerPage();
  TestGpuDownloadProtectionMirrors();
  TestCrossRegionUpload();
  TestConcurrentColdUploads();
  TestUploadDoesNotSerializeDisjointRegion();
  TestQueriesSkipRegionLock();
  TestDownloadDoesNotSerializeDisjointRegion();
  TestGpuUnmarkUsesRegionMask();
  TestFullRegionGpuUnmarkBatching();
  TestKeptGpuProtection();
  TestKeptGpuProtectionRuns();
  TestGpuRemarkKeepsProtection();
  TestRangeCpuClean();
  TestFatalPaths();
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
  TestFaultOnProtectedStack();
#endif
  std::puts("MemoryTrackerTests: all cases passed");
  return 0;
}
