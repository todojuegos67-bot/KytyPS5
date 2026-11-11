#include "common/common.h"

#if KYTY_PLATFORM != KYTY_PLATFORM_LINUX
// #error "KYTY_PLATFORM != KYTY_PLATFORM_LINUX"
#else

#include "common/assert.h"
#include "common/pageProtectionTable.h"
#include "common/platform/sysVirtual.h"
#include "common/virtualMemory.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

#if !defined(__APPLE__)
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

// IWYU pragma: no_include <asm/mman-common.h>
// IWYU pragma: no_include <asm/mman.h>
// IWYU pragma: no_include <bits/pthread_types.h>
// IWYU pragma: no_include <linux/mman.h>

#if defined(MAP_FIXED_NOREPLACE) && KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#define KYTY_FIXED_NOREPLACE
#endif

namespace Common {

static pthread_mutex_t              g_virtual_mutex {};
static std::map<uintptr_t, size_t>* g_allocs   = nullptr;
static PageProtectionTable*         g_protects = nullptr;

#if !defined(__APPLE__)
// KYTY_UFFD_WP=1: the GPU trackers' write watch of guest views as userfaultfd write protection (SysVirtualProtectWriteWatch).
// A view's page keeps its mprotect protection writable and takes the write protection instead: a protect-fault-unprotect
// cycle costs ~45% less (one ioctl each way, no VMA split or merge), and a write raises SIGBUS (BUS_ADRERR, the page-fault
// error code's write bit set), which the host fault handler serves as it serves SIGSEGV. UFFD_USER_MODE_ONLY needs no
// privilege; a kernel write into such a page fails with EFAULT, as into a read-only one. Pages not of a registered view,
// or past the tracker's 1 << 40 window, keep mprotect.
static int g_uffd = -1;
// A byte per 4 KiB page of the low 1 TiB (reserved, committed as touched): the page's mprotect protection (PROT_*), as
// last set, whether its view is registered, whether it is write-protected, and its minor mode (below).
static uint8_t*         g_watch_pages     = nullptr;
static constexpr uint64_t WATCH_PAGE_COUNT = (uint64_t {1} << 40u) >> 12u;
static constexpr uint8_t  WATCH_PROT       = 0x07;
static constexpr uint8_t  WATCH_MINOR      = 0x08;
static constexpr uint8_t  WATCH_ZAPPED     = 0x10;
static constexpr uint8_t  WATCH_REGISTERED = 0x40;
static constexpr uint8_t  WATCH_WP         = 0x80;

// KYTY_UFFD_MINOR=1 (with KYTY_UFFD_WP; on by default in the Steam Deck build): the GPU thread's deferred read
// protections (no access: PageManager's flush, SysVirtualZapNoAccess) of runs of up to MINOR_RUN_PAGES pages of a
// registered view drop the pages' PTEs (MADV_DONTNEED) under userfaultfd minor mode instead of an mprotect: the next
// access faults (SIGBUS, as a write protection's), and lifting it maps the page cache page again (UFFDIO_CONTINUE, its
// write protection with it). No VMA split or merge and no mmap write lock: at 1-1 those protections were ~95 single
// pages a frame (GPU results the game reads back), each cycle two mprotects, ~6 ms a frame of the GPU thread walking; a
// microbenchmark with 7 busy threads and 7,000 views: 35.7 us a cycle as mprotects, 10.6 us zapped. Minor mode is
// registered by aligned blocks of MINOR_BLOCK_PAGES within a view (pages registered one by one split the views into
// thousands of VMAs, and the remaining mprotects over them took ~120 us instead of ~25), up to MINOR_BLOCK_CAP of them
// (~120 more a world load), for as long as their view lasts. Never left: a registration with fewer modes keeps the ones
// a VMA has (userfaultfd_register skips it), and an unregistration (then a registration for write protection only)
// fails other threads' write protections of the pages meanwhile, which then take them for pages of no registered view
// while the kernel still write-protects them: a write faulted forever at a world load. A page of a block
// whose PTE is missing otherwise (first touched there, reclaimed) faults too and is mapped at once
// (SysVirtualResolveMinorFault); a kernel access to it fails (EFAULT), as to a protected page. Pages not in the page
// cache (holes, swapped out: their access would not fault) and other protections keep mprotect. 1-1 at 360p, same
// boot alternating: stand 34.0/34.1/34.2 -> 35.9/35.8/35.8 fps (render 23.1 -> 21.9 ms), walk legs 33.8 -> 35.4 fps
// (p99 37.5 -> 35.8 ms).
static bool              g_minor_on      = false;
static thread_local bool g_zap_no_access = false;
static std::mutex        g_minor_mutex;
static std::map<uint64_t, uint64_t>* g_minor_blocks = nullptr; // (first page -> end page; under g_minor_mutex)
static constexpr uint64_t           MINOR_RUN_PAGES   = 64;
static constexpr uint64_t           MINOR_BLOCK_PAGES = 64;
static constexpr uint64_t           MINOR_BLOCK_CAP   = 8192;

// KYTY_WRITE_GUARD_UFFD=1 (default 0; issue #289): a guest view's mapping (SysVirtualWriteWatchMapped) records its
// protection in g_protects, as SysVirtualProtect does (a placeholder keeps the old record: no guard arms there, it has no
// backing view; and its mmap may fail with the view still mapped). Without
// it a view page's recorded protection is unknown (read as no access) until an mprotect of it, and with the write watch
// by userfaultfd the trackers make none on a page they keep writable. KYTY_WRITE_GUARD's Arm then saved no access for
// a label page it had just written, and its lift left the page no-access with no guard and no tracker holding it: the
// GPU thread's WAIT_REG_MEM read of the label faulted forever (base 2dd7a061a: page byte 0x40, a 4 KiB ---s VMA,
// g_pages empty). With this switch the guard also stays on with KYTY_UFFD_WP (graphicsRun.cpp).
static bool g_record_view_protection = false;

static int OpenUffd(uint64_t features) {
	const int fd = static_cast<int>(syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY));
	uffdio_api api {};
	api.api      = UFFD_API;
	api.features = features;
	if (fd >= 0 && ioctl(fd, UFFDIO_API, &api) != 0) {
		close(fd);
		return -1;
	}
	return fd;
}

static void WriteWatchInit() {
	// VirtualMemory::Init() runs more than once (main and the kernel memory module).
	if (const char* record = std::getenv("KYTY_WRITE_GUARD_UFFD"); record != nullptr && record[0] == '1')
		g_record_view_protection = true;
	if (g_uffd >= 0) return;
	const char* value = std::getenv("KYTY_UFFD_WP");
	const char* minor = std::getenv("KYTY_UFFD_MINOR");
#if defined(KYTY_STEAM_DECK)
	// On by default in the Steam Deck build (KYTY_UFFD_WP=0: mprotect): there the upload worker's re-arms took 4.4 ms a
	// frame as mprotect calls, 2.3 ms as write protections; 1-1 at 360p, same boots alternating: stand 29.0/29.3 ->
	// 29.8/30.2 fps, a camera spin 28.0/26.9 -> 29.4/29.0.
	if (value == nullptr || value[0] == '\0') value = "1";
	if (minor == nullptr || minor[0] == '\0') minor = "1";
#endif
	if (value == nullptr || value[0] != '1') return;
	constexpr uint64_t features = UFFD_FEATURE_SIGBUS | UFFD_FEATURE_WP_HUGETLBFS_SHMEM | UFFD_FEATURE_WP_UNPOPULATED;
	const bool         want_minor = minor != nullptr && minor[0] == '1';
	int                fd         = want_minor ? OpenUffd(features | UFFD_FEATURE_MINOR_SHMEM) : -1;
	const bool         has_minor  = fd >= 0;
	if (fd < 0) fd = OpenUffd(features);
	void* pages = MAP_FAILED;
	if (fd >= 0) {
		pages = mmap(nullptr, WATCH_PAGE_COUNT, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
	}
	if (pages == MAP_FAILED) {
		std::printf("KYTY_UFFD_WP: userfaultfd write protection unavailable, mprotect kept\n");
		if (fd >= 0) close(fd);
		return;
	}
	g_watch_pages = static_cast<uint8_t*>(pages);
	g_uffd        = fd;
	std::printf("KYTY_UFFD_WP: write watch by userfaultfd write protection\n");
	if (!want_minor) return;
	g_minor_on = has_minor;
	if (g_minor_on) g_minor_blocks = new std::map<uint64_t, uint64_t>;
	std::printf("KYTY_UFFD_MINOR: %s\n", g_minor_on ? "the GPU's read protections of short runs by userfaultfd minor mode"
	                                                : "userfaultfd minor mode unavailable, mprotect kept");
}

static bool WatchRange(uint64_t address, uint64_t size) {
	return g_uffd >= 0 && size != 0 && (address & 0xfffu) == 0 && address >> 12u < WATCH_PAGE_COUNT &&
	       size <= (WATCH_PAGE_COUNT << 12u) - address;
}

static uint8_t WatchPage(uint64_t page) {
	return __atomic_load_n(&g_watch_pages[page], __ATOMIC_RELAXED);
}

static void SetWatchPages(uint64_t first, uint64_t end, uint8_t value) {
	for (auto page = first; page < end; page++) __atomic_store_n(&g_watch_pages[page], value, __ATOMIC_RELAXED);
}

// Bits of [first, end) set or cleared, each page's others as they are now: another thread's protection of the pages
// (its PROT_* bits) is not undone by a byte read before it.
static void SetWatchBits(uint64_t first, uint64_t end, uint8_t bits) {
	for (auto page = first; page < end; page++) __atomic_fetch_or(&g_watch_pages[page], bits, __ATOMIC_RELAXED);
}

static void ClearWatchBits(uint64_t first, uint64_t end, uint8_t bits) {
	for (auto page = first; page < end; page++)
		__atomic_fetch_and(&g_watch_pages[page], static_cast<uint8_t>(~bits), __ATOMIC_RELAXED);
}

// After an mprotect of [address, address + size): its pages' protection, their other bits kept.
static void RecordWatchProtection(uint64_t address, uint64_t size, int protect) {
	if (!WatchRange(address, (size + 0xfffu) & ~uint64_t {0xfff})) return;
	for (auto page = address >> 12u, end = (address + size + 0xfffu) >> 12u; page < end; page++) {
		// (Its other bits as other threads set them meanwhile: a minor page's zap among them.)
		auto old = WatchPage(page);
		while ((old & WATCH_PROT) != protect &&
		       !__atomic_compare_exchange_n(&g_watch_pages[page], &old, static_cast<uint8_t>((old & ~WATCH_PROT) | protect), true,
		                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
		}
	}
}

static bool WriteProtect(uint64_t address, uint64_t size, bool on) {
	uffdio_writeprotect wp {};
	wp.range.start = address;
	wp.range.len   = size;
	wp.mode        = on ? UFFDIO_WRITEPROTECT_MODE_WP : 0;
	return ioctl(g_uffd, UFFDIO_WRITEPROTECT, &wp) == 0;
}

static bool RegisterView(uint64_t address, uint64_t size, uint64_t mode) {
	uffdio_register reg {};
	reg.range.start = address;
	reg.range.len   = size;
	reg.mode        = mode;
	return ioctl(g_uffd, UFFDIO_REGISTER, &reg) == 0;
}

// A zapped run mapped again, write-protected or writable; page by page where a fault mapped some of it first (their
// write protection set as asked) or a page left the page cache (nothing to map: its next access gets zeros, as any
// hole's).
static void MapZapped(uint64_t address, uint64_t size, bool wp) {
	uffdio_continue map {};
	map.range.start = address;
	map.range.len   = size;
	map.mode        = wp ? UFFDIO_CONTINUE_MODE_WP : 0;
	if (ioctl(g_uffd, UFFDIO_CONTINUE, &map) == 0) return;
	for (auto page = address; page < address + size; page += 0x1000u) {
		map.range.start = page;
		map.range.len   = 0x1000u;
		map.mapped      = 0;
		if (ioctl(g_uffd, UFFDIO_CONTINUE, &map) != 0 && errno == EEXIST) WriteProtect(page, 0x1000u, wp);
	}
}

// The no-access protection of a registered run of `state` (SysVirtualProtectWriteWatch) by zapping its PTEs, when
// minor mode can hold it: false keeps mprotect. A run outside minor mode first registers its aligned block's pages of
// the view (those registered next to it, not in minor mode yet). The run's bits say zapped before its PTEs go (a fault
// in between is served as a protection's), its block's say minor before its registration.
static bool ZapRun(uint64_t address, uint64_t size, uint8_t state) {
	if (!g_minor_on || !g_zap_no_access) return false;
	if ((state & WATCH_ZAPPED) != 0) return true;
	const auto first = address >> 12u, end = (address + size) >> 12u;
	if ((state & WATCH_PROT) != (PROT_READ | PROT_WRITE) || end - first > MINOR_RUN_PAGES) return false;
	unsigned char resident[MINOR_RUN_PAGES] {};
	if (mincore(reinterpret_cast<void*>(address), size, resident) != 0) return false;
	for (uint64_t i = 0; i < end - first; i++)
		if ((resident[i] & 1u) == 0) return false;
	{
		std::lock_guard lock(g_minor_mutex);
		if ((state & WATCH_MINOR) == 0) {
			const auto block_first = first & ~(MINOR_BLOCK_PAGES - 1);
			const auto block_end   = (end + MINOR_BLOCK_PAGES - 1) & ~(MINOR_BLOCK_PAGES - 1);
			auto       lo = first, hi = end;
			while (lo > block_first && (WatchPage(lo - 1) & (WATCH_REGISTERED | WATCH_MINOR)) == WATCH_REGISTERED) lo--;
			while (hi < block_end && (WatchPage(hi) & (WATCH_REGISTERED | WATCH_MINOR)) == WATCH_REGISTERED) hi++;
			if (g_minor_blocks->size() >= MINOR_BLOCK_CAP) return false;
			SetWatchBits(lo, hi, WATCH_MINOR);
			if (!RegisterView(lo << 12u, (hi - lo) << 12u, UFFDIO_REGISTER_MODE_MINOR | UFFDIO_REGISTER_MODE_WP)) {
				ClearWatchBits(lo, hi, WATCH_MINOR);
				return false;
			}
			(*g_minor_blocks)[lo] = hi;
		}
	}
	SetWatchBits(first, end, WATCH_ZAPPED);
	if (madvise(reinterpret_cast<void*>(address), size, MADV_DONTNEED) != 0) {
		ClearWatchBits(first, end, WATCH_ZAPPED);
		return false;
	}
	return true;
}

// The minor blocks over [first, end) forgotten (their view unmapped or replaced: its registration went with it).
static void ForgetMinor(uint64_t first, uint64_t end) {
	if (!g_minor_on) return;
	std::lock_guard lock(g_minor_mutex);
	auto it = g_minor_blocks->upper_bound(first);
	if (it != g_minor_blocks->begin() && std::prev(it)->second > first) --it;
	while (it != g_minor_blocks->end() && it->first < end) it = g_minor_blocks->erase(it);
}
#else
static void WriteWatchInit() {}
static void RecordWatchProtection(uint64_t /*address*/, uint64_t /*size*/, int /*protect*/) {}
#endif

void SysVirtualInit() {
	pthread_mutexattr_t attr {};

	pthread_mutexattr_init(&attr);
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
	pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_FAST_NP); // glibc-only fast mutex
#else
	pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_NORMAL);
#endif
	pthread_mutex_init(&g_virtual_mutex, &attr);
	pthread_mutexattr_destroy(&attr);

	g_allocs   = new std::map<uintptr_t, size_t>;
	g_protects = new PageProtectionTable;
	WriteWatchInit();
}

static int get_protection_flag(VirtualMemory::Mode mode) {
	int protect = PROT_NONE;
	switch (mode) {
		case VirtualMemory::Mode::Read: protect = PROT_READ; break;
		case VirtualMemory::Mode::Write:
		case VirtualMemory::Mode::ReadWrite: protect = PROT_READ | PROT_WRITE; break; // NOLINT
		case VirtualMemory::Mode::Execute: protect = PROT_EXEC; break;
		case VirtualMemory::Mode::ExecuteRead: protect = PROT_EXEC | PROT_READ; break; // NOLINT
		case VirtualMemory::Mode::ExecuteWrite:
		case VirtualMemory::Mode::ExecuteReadWrite:
			protect = PROT_EXEC | PROT_WRITE | PROT_READ;
			break; // NOLINT
		case VirtualMemory::Mode::NoAccess:
		default: protect = PROT_NONE; break;
	}
	return protect;
}

static VirtualMemory::Mode get_protection_flag(int mode) {
	switch (mode) {
		case PROT_NONE: return VirtualMemory::Mode::NoAccess;
		case PROT_READ: return VirtualMemory::Mode::Read;
		case PROT_WRITE: return VirtualMemory::Mode::Write;
		case PROT_READ | PROT_WRITE: return VirtualMemory::Mode::ReadWrite; // NOLINT
		case PROT_EXEC: return VirtualMemory::Mode::Execute;
		case PROT_EXEC | PROT_WRITE: return VirtualMemory::Mode::ExecuteWrite; // NOLINT
		case PROT_EXEC | PROT_READ: return VirtualMemory::Mode::ExecuteRead;   // NOLINT
		case PROT_EXEC | PROT_WRITE | PROT_READ:
			return VirtualMemory::Mode::ExecuteReadWrite; // NOLINT
		default: return VirtualMemory::Mode::NoAccess;
	}
}

// Keep automatic mappings inside the guest and GPU-addressable low window.
#ifdef KYTY_FIXED_NOREPLACE
static constexpr uintptr_t LOW_ARENA_LIMIT = 0x000000FC00000000ULL; // libc mspace window ceiling
static constexpr uintptr_t LOW_ARENA_FLOOR = 0x000000A000000000ULL; // 640 GiB
static constexpr uintptr_t LOW_ARENA_GRAIN = 0x0000000000010000ULL; // 64 KiB

static_assert(LOW_ARENA_LIMIT <= 0x0000010000000000ULL,
              "arena must stay inside the GPU page tracker's 1<<40 window");
static_assert(LOW_ARENA_FLOOR < LOW_ARENA_LIMIT, "arena floor must sit below its ceiling");

static std::atomic<uintptr_t> g_low_arena_next {LOW_ARENA_LIMIT};
#endif

// Caller holds g_virtual_mutex.
static void record_alloc(uintptr_t addr, size_t size) {
	auto next = g_allocs->upper_bound(addr);
	if (next != g_allocs->begin()) {
		auto       it         = std::prev(next);
		const auto alloc_addr = it->first;
		const auto alloc_end  = alloc_addr + it->second;
		if (alloc_addr <= addr && addr + size <= alloc_end) {
			g_allocs->erase(it);
			if (alloc_addr < addr) {
				(*g_allocs)[alloc_addr] = addr - alloc_addr;
			}
			if (addr + size < alloc_end) {
				(*g_allocs)[addr + size] = alloc_end - (addr + size);
			}
		}
	}
	(*g_allocs)[addr] = size;
}

#ifdef KYTY_FIXED_NOREPLACE
static uintptr_t align_up_to(uintptr_t addr, uint64_t alignment) {
	return (addr + alignment - 1) & ~(alignment - 1);
}
#endif

// Freed arena addresses are not reused while GPU caches remain keyed by address.
static void* map_anonymous(uintptr_t addr, size_t size, int protect, int flags) {
	if (addr != 0) {
		return mmap(reinterpret_cast<void*>(addr), size, protect, flags, -1, 0); // NOLINT
	}

#ifdef KYTY_FIXED_NOREPLACE
	const auto step = align_up_to(size, LOW_ARENA_GRAIN);
	for (int attempt = 0; attempt < 256; attempt++) {
		const auto top = g_low_arena_next.fetch_sub(step, std::memory_order_relaxed);
		if (top < step || top - step < LOW_ARENA_FLOOR) {
			break;
		}
		const auto hint = (top - step) & ~(LOW_ARENA_GRAIN - 1);
		void* ptr = mmap(reinterpret_cast<void*>(hint), size, protect, flags | MAP_FIXED_NOREPLACE,
		                 -1, 0); // NOLINT
		if (ptr != MAP_FAILED) {
			return ptr;
		}
	}
#endif

	return mmap(nullptr, size, protect, flags, -1, 0); // NOLINT
}

uint64_t SysVirtualAlloc(uint64_t address, uint64_t size, VirtualMemory::Mode mode) {
	EXIT_IF(g_allocs == nullptr);

	auto addr = static_cast<uintptr_t>(address);

	int protect = get_protection_flag(mode);

	void* ptr = map_anonymous(addr, size, protect, MAP_PRIVATE | MAP_ANON);

	auto ret_addr = reinterpret_cast<uintptr_t>(ptr);

	if (ptr != MAP_FAILED) {
		pthread_mutex_lock(&g_virtual_mutex);
		record_alloc(ret_addr, size);
		uintptr_t page_start = ret_addr >> 12u;
		uintptr_t page_end   = (ret_addr + size - 1) >> 12u;
		g_protects->Set(page_start, page_end, static_cast<uint8_t>(protect));
		pthread_mutex_unlock(&g_virtual_mutex);
	}

	return ret_addr;
}

static uintptr_t align_up(uintptr_t addr, uint64_t alignment) {
	return (addr + alignment - 1) & ~(alignment - 1);
}

uint64_t SysVirtualAllocAligned(uint64_t address, uint64_t size, VirtualMemory::Mode mode,
                                uint64_t alignment) {
	if (alignment == 0) {
		return 0;
	}

	EXIT_IF(g_allocs == nullptr);

	auto addr    = static_cast<uintptr_t>(address);
	int  protect = get_protection_flag(mode);

	void* ptr = map_anonymous(addr, size, protect, MAP_PRIVATE | MAP_ANON);

	auto ret_addr = reinterpret_cast<uintptr_t>(ptr);

	if (ptr != MAP_FAILED && ((ret_addr & (alignment - 1)) != 0)) {
		munmap(ptr, size);

		ptr =
		    map_anonymous(addr, size + alignment, protect, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE);
		ret_addr = reinterpret_cast<uintptr_t>(ptr);
		if (ptr != MAP_FAILED) {
#if defined(__APPLE__)
			// Carve the aligned subrange out of the live mapping with MAP_FIXED (in-place
			// replacement) and trim the slack; never munmap the whole range first, or a
			// concurrent host mapping (dyld, Rosetta, Metal) could claim the hole and be
			// destroyed by the MAP_FIXED. Other platforms keep the original path below.
			auto aligned_addr = align_up(ret_addr, alignment);
			// NOLINTNEXTLINE
			void* fixed = mmap(reinterpret_cast<void*>(aligned_addr), size, protect,
			                   MAP_FIXED | MAP_PRIVATE | MAP_ANON, -1, 0);
			if (fixed == MAP_FAILED) {
				munmap(ptr, size + alignment);
				ret_addr = 0;
				ptr      = MAP_FAILED;
			} else {
				if (aligned_addr > ret_addr) {
					munmap(reinterpret_cast<void*>(ret_addr), aligned_addr - ret_addr);
				}
				const uintptr_t tail_start = aligned_addr + size;
				const uintptr_t resv_end   = ret_addr + size + alignment;
				if (resv_end > tail_start) {
					munmap(reinterpret_cast<void*>(tail_start), resv_end - tail_start);
				}
				ptr      = fixed;
				ret_addr = aligned_addr;
			}
#else
			munmap(ptr, size + alignment);
			auto aligned_addr = align_up(ret_addr, alignment);
#ifdef KYTY_FIXED_NOREPLACE
			// NOLINTNEXTLINE
			ptr = mmap(reinterpret_cast<void*>(aligned_addr), size, protect,
			           MAP_FIXED_NOREPLACE | MAP_PRIVATE | MAP_ANON, -1, 0);
#else
			// NOLINTNEXTLINE
			ptr = mmap(reinterpret_cast<void*>(aligned_addr), size, protect,
			           MAP_FIXED | MAP_PRIVATE | MAP_ANON, -1, 0);
#endif
			ret_addr = reinterpret_cast<uintptr_t>(ptr);
			if (ptr != MAP_FAILED && ((ret_addr & (alignment - 1)) != 0)) {
				munmap(ptr, size);
				ret_addr = 0;
				ptr      = MAP_FAILED;
			}
#endif
		}
	}

	if (ptr == MAP_FAILED) {
		return SysVirtualAllocAligned(address, size, mode, alignment << 1u);
	}

	pthread_mutex_lock(&g_virtual_mutex);
	record_alloc(ret_addr, size);
	uintptr_t page_start = ret_addr >> 12u;
	uintptr_t page_end   = (ret_addr + size - 1) >> 12u;
	g_protects->Set(page_start, page_end, static_cast<uint8_t>(protect));
	pthread_mutex_unlock(&g_virtual_mutex);

	return ret_addr;
}

#if defined(__APPLE__)
// macOS has no /proc/self/maps; query the Mach VM map directly. mach_vm_region returns
// the first mapped region at or above `region_addr`; if it begins before the end of the
// requested range, the range overlaps an existing mapping.
static bool is_mapped(void* ptr, size_t length) {
	auto                           query_addr  = reinterpret_cast<mach_vm_address_t>(ptr);
	mach_vm_address_t              region_addr = query_addr;
	mach_vm_size_t                 region_size = 0;
	vm_region_basic_info_data_64_t info {};
	mach_msg_type_number_t         count       = VM_REGION_BASIC_INFO_COUNT_64;
	mach_port_t                    object_name = MACH_PORT_NULL;

	kern_return_t kr =
	    mach_vm_region(mach_task_self(), &region_addr, &region_size, VM_REGION_BASIC_INFO_64,
	                   reinterpret_cast<vm_region_info_t>(&info), &count, &object_name);
	if (kr != KERN_SUCCESS) {
		return false; // no region at or above the address → unmapped
	}
	return region_addr < (query_addr + length);
}
#else
static bool is_mapped(void* ptr, size_t length) {
	FILE* file = fopen("/proc/self/maps", "r");
	char  line[1024];
	bool  ret  = false;
	auto  addr = reinterpret_cast<uintptr_t>(ptr);

	while (feof(file) == 0) {
		if (fgets(line, 1024, file) == nullptr) {
			break;
		}
		uint64_t start = 0;
		uint64_t end   = 0;
		// NOLINTNEXTLINE(cert-err34-c)
		if (sscanf(line, "%" SCNx64 "-%" SCNx64, &start, &end) != 2) {
			continue;
		}
		if (addr >= start && addr + length <= end) {
			ret = true;
			break;
		}
	}
	fclose(file);
	return ret;
}
#endif

bool SysVirtualAllocFixed(uint64_t address, uint64_t size, VirtualMemory::Mode mode) {
	EXIT_IF(g_allocs == nullptr);

	auto addr    = static_cast<uintptr_t>(address);
	int  protect = get_protection_flag(mode);

#ifdef KYTY_FIXED_NOREPLACE
	// NOLINTNEXTLINE
	void* ptr = mmap(reinterpret_cast<void*>(addr), size, protect,
	                 MAP_FIXED_NOREPLACE | MAP_PRIVATE | MAP_ANON, -1, 0);
#else
	// NOLINTNEXTLINE
	void* ptr = (is_mapped(reinterpret_cast<void*>(addr), size)
	                 ? MAP_FAILED
	                 : mmap(reinterpret_cast<void*>(addr), size, protect,
	                        MAP_FIXED | MAP_PRIVATE | MAP_ANON, -1, 0));
#endif

	auto ret_addr = reinterpret_cast<uintptr_t>(ptr);

	if (ptr != MAP_FAILED && ret_addr != addr) {
		munmap(ptr, size);
		ret_addr = 0;
		ptr      = MAP_FAILED;
	}

	if (ptr != MAP_FAILED) {
		pthread_mutex_lock(&g_virtual_mutex);
		record_alloc(ret_addr, size);
		uintptr_t page_start = ret_addr >> 12u;
		uintptr_t page_end   = (ret_addr + size - 1) >> 12u;
		g_protects->Set(page_start, page_end, static_cast<uint8_t>(protect));
		pthread_mutex_unlock(&g_virtual_mutex);

		return true;
	}

	return false;
}

bool SysVirtualCommit(uint64_t address, uint64_t size, VirtualMemory::Mode mode) {
	return SysVirtualProtect(address, size, mode);
}

uint64_t SysVirtualReserve(uint64_t address, uint64_t size) {
	return SysVirtualReserveAligned(address, size, 1);
}

uint64_t SysVirtualReserveAligned(uint64_t address, uint64_t size, uint64_t alignment) {
	if (alignment == 0) {
		return 0;
	}

	EXIT_IF(g_allocs == nullptr);

	auto addr = static_cast<uintptr_t>(address);

	void* ptr = map_anonymous(addr, size, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE);

	auto ret_addr = reinterpret_cast<uintptr_t>(ptr);

	if (ptr != MAP_FAILED && ((ret_addr & (alignment - 1)) != 0)) {
		munmap(ptr, size);

		ptr      = map_anonymous(addr, size + alignment, PROT_NONE,
		                         MAP_PRIVATE | MAP_ANON | MAP_NORESERVE);
		ret_addr = reinterpret_cast<uintptr_t>(ptr);
		if (ptr != MAP_FAILED) {
#if defined(__APPLE__)
			// Carve the aligned subrange out of the live reservation with MAP_FIXED (an
			// in-place replacement), then trim the slack. The range must never be
			// returned to the OS in between: another thread (dyld, Rosetta, Metal,
			// malloc) could claim the hole, and the subsequent MAP_FIXED would silently
			// destroy its mapping. Other platforms keep the original path below.
			auto aligned_addr = align_up(ret_addr, alignment);
			// NOLINTNEXTLINE
			void* fixed = mmap(reinterpret_cast<void*>(aligned_addr), size, PROT_NONE,
			                   MAP_FIXED | MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
			if (fixed == MAP_FAILED) {
				munmap(ptr, size + alignment);
				ret_addr = 0;
				ptr      = MAP_FAILED;
			} else {
				if (aligned_addr > ret_addr) {
					munmap(reinterpret_cast<void*>(ret_addr), aligned_addr - ret_addr);
				}
				const uintptr_t tail_start = aligned_addr + size;
				const uintptr_t resv_end   = ret_addr + size + alignment;
				if (resv_end > tail_start) {
					munmap(reinterpret_cast<void*>(tail_start), resv_end - tail_start);
				}
				ptr      = fixed;
				ret_addr = aligned_addr;
			}
#else
			munmap(ptr, size + alignment);
			auto aligned_addr = align_up(ret_addr, alignment);
#ifdef KYTY_FIXED_NOREPLACE
			// NOLINTNEXTLINE
			ptr = mmap(reinterpret_cast<void*>(aligned_addr), size, PROT_NONE,
			           MAP_FIXED_NOREPLACE | MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
#else
			// NOLINTNEXTLINE
			ptr = mmap(reinterpret_cast<void*>(aligned_addr), size, PROT_NONE,
			           MAP_FIXED | MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
#endif
			ret_addr = reinterpret_cast<uintptr_t>(ptr);
			if (ptr != MAP_FAILED && ((ret_addr & (alignment - 1)) != 0)) {
				munmap(ptr, size);
				ret_addr = 0;
				ptr      = MAP_FAILED;
			}
#endif
		}
	}

	if (ptr == MAP_FAILED) {
		return SysVirtualReserveAligned(address, size, alignment << 1u);
	}

	pthread_mutex_lock(&g_virtual_mutex);
	record_alloc(ret_addr, size);
	pthread_mutex_unlock(&g_virtual_mutex);

	return ret_addr;
}

bool SysVirtualReserveFixed(uint64_t address, uint64_t size) {
	EXIT_IF(g_allocs == nullptr);

	auto addr = static_cast<uintptr_t>(address);

#ifdef KYTY_FIXED_NOREPLACE
	// NOLINTNEXTLINE
	void* ptr = mmap(reinterpret_cast<void*>(addr), size, PROT_NONE,
	                 MAP_FIXED_NOREPLACE | MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
#else
	// NOLINTNEXTLINE
	void* ptr = (is_mapped(reinterpret_cast<void*>(addr), size)
	                 ? MAP_FAILED
	                 : mmap(reinterpret_cast<void*>(addr), size, PROT_NONE,
	                        MAP_FIXED | MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0));
#endif

	auto ret_addr = reinterpret_cast<uintptr_t>(ptr);

	if (ptr != MAP_FAILED && ret_addr != addr) {
		munmap(ptr, size);
		ret_addr = 0;
		ptr      = MAP_FAILED;
	}

	if (ptr != MAP_FAILED) {
		pthread_mutex_lock(&g_virtual_mutex);
		record_alloc(ret_addr, size);
		pthread_mutex_unlock(&g_virtual_mutex);

		return true;
	}

	return false;
}

bool SysVirtualDecommit(uint64_t address, uint64_t size) {
	// Drop physical pages while preserving the reservation.
	if (!SysVirtualProtect(address, size, VirtualMemory::Mode::NoAccess)) {
		return false;
	}

	if (size != 0) {
#if defined(__APPLE__)
		constexpr int RECLAIM_ADVICE = MADV_FREE;
#else
		constexpr int RECLAIM_ADVICE = MADV_DONTNEED;
#endif
		const auto page_size = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
		if (page_size != 0) {
			// Do not discard pages outside the requested range.
			const auto begin = (static_cast<uintptr_t>(address) + page_size - 1) & ~(page_size - 1);
			const auto end   = (static_cast<uintptr_t>(address) + size) & ~(page_size - 1);
			if (end > begin) {
				::madvise(reinterpret_cast<void*>(begin), end - begin, RECLAIM_ADVICE);
			}
		}
	}

	return true;
}

bool SysVirtualFree(uint64_t address) {
	EXIT_IF(g_allocs == nullptr);
	size_t size = 0;

	auto addr = static_cast<uintptr_t>(address & ~static_cast<uint64_t>(0xfffu));

	pthread_mutex_lock(&g_virtual_mutex);
	if (auto s = g_allocs->find(addr); s != g_allocs->end()) {
		size = s->second;
		g_allocs->erase(s);
	}
	pthread_mutex_unlock(&g_virtual_mutex);

	if (size == 0) {
		return false;
	}

	if (munmap(reinterpret_cast<void*>(addr), size) == 0) {
		uintptr_t page_start = addr >> 12u;
		uintptr_t page_end   = (addr + size - 1) >> 12u;
		pthread_mutex_lock(&g_virtual_mutex);
		g_protects->Erase(page_start, page_end);
		pthread_mutex_unlock(&g_virtual_mutex);
		return true;
	}

	return false;
}

bool SysVirtualFreeRange(uint64_t address, uint64_t size) {
	EXIT_IF(g_allocs == nullptr);
	if (size == 0 || (address & 0xfffu) != 0 || (size & 0xfffu) != 0) {
		return false;
	}

	const auto addr = static_cast<uintptr_t>(address);
	const auto end  = addr + size;
	if (end < addr) {
		return false;
	}

	pthread_mutex_lock(&g_virtual_mutex);
	auto next = g_allocs->upper_bound(addr);
	if (next == g_allocs->begin()) {
		pthread_mutex_unlock(&g_virtual_mutex);
		return false;
	}

	// A reservation may have been split into several adjacent records.
	auto       first      = std::prev(next);
	const auto alloc_addr = first->first;
	if (addr < alloc_addr || alloc_addr + first->second <= addr) {
		pthread_mutex_unlock(&g_virtual_mutex);
		return false;
	}

	auto      last   = first;
	uintptr_t cursor = alloc_addr + first->second;
	while (cursor < end) {
		auto following = std::next(last);
		if (following == g_allocs->end() || following->first != cursor) {
			pthread_mutex_unlock(&g_virtual_mutex);
			return false;
		}
		last   = following;
		cursor = following->first + following->second;
	}
	const auto alloc_end = cursor;

	if (munmap(reinterpret_cast<void*>(addr), size) != 0) {
		pthread_mutex_unlock(&g_virtual_mutex);
		return false;
	}

	g_allocs->erase(first, std::next(last));
	if (alloc_addr < addr) {
		(*g_allocs)[alloc_addr] = addr - alloc_addr;
	}
	if (end < alloc_end) {
		(*g_allocs)[end] = alloc_end - end;
	}
	g_protects->Erase(addr >> 12u, (end - 1u) >> 12u);
	pthread_mutex_unlock(&g_virtual_mutex);
	return true;
}

bool SysVirtualProtect(uint64_t address, uint64_t size, VirtualMemory::Mode mode,
                       VirtualMemory::Mode* old_mode) {
	auto addr = static_cast<uintptr_t>(address);

	pthread_mutex_lock(&g_virtual_mutex);
	if (old_mode != nullptr) {
		*old_mode = get_protection_flag(g_protects->Get(addr >> 12u));
	}
	pthread_mutex_unlock(&g_virtual_mutex);

	uintptr_t page_start = addr >> 12u;
	uintptr_t page_end   = (addr + size - 1) >> 12u;
	if (mprotect(reinterpret_cast<void*>(page_start << 12u), (page_end - page_start + 1) << 12u,
	             get_protection_flag(mode)) == 0) {
		pthread_mutex_lock(&g_virtual_mutex);
		g_protects->Set(page_start, page_end, static_cast<uint8_t>(get_protection_flag(mode)));
		pthread_mutex_unlock(&g_virtual_mutex);
		RecordWatchProtection(page_start << 12u, (page_end - page_start + 1) << 12u, get_protection_flag(mode));
		return true;
	}

	return false;
}

#if !defined(__APPLE__)
// Runs of pages in one state: those of registered views take the write protection for a write watch (Read) and lose it
// for a writable mode, and an mprotect only where their protection differs (a write watch keeps the view writable:
// after a no-access watch, made writable again under the write protection set first).
bool SysVirtualProtectWriteWatch(uint64_t address, uint64_t size, VirtualMemory::Mode mode) {
	const auto begin = address & ~uint64_t {0xfff};
	const auto end   = (address + size + 0xfffu) & ~uint64_t {0xfff};
	if (!WatchRange(begin, end - begin)) return SysVirtualProtect(address, size, mode);
	const int  protect  = get_protection_flag(mode);
	const bool watch    = protect == PROT_READ;
	const bool writable = (protect & PROT_WRITE) != 0;
	for (auto first = begin >> 12u, last = end >> 12u; first < last;) {
		auto state = WatchPage(first);
		auto next  = first + 1;
		while (next < last && WatchPage(next) == state) next++;
		const auto run_address = first << 12u;
		const auto run_size    = (next - first) << 12u;
		first                  = next;
		if ((state & WATCH_REGISTERED) == 0) {
			if (!SysVirtualProtect(run_address, run_size, mode)) return false;
			continue;
		}
		if (protect == PROT_NONE && ZapRun(run_address, run_size, state)) continue;
		if ((state & WATCH_ZAPPED) != 0 && protect != PROT_NONE) {
			// (Lifted: the page cache pages mapped again, write-protected for a watch; the bits say so before the run stops
			// being zapped, which the fault paths below leave alone.)
			MapZapped(run_address, run_size, watch);
			if (watch) SetWatchBits(run_address >> 12u, next, WATCH_WP);
			ClearWatchBits(run_address >> 12u, next, static_cast<uint8_t>(WATCH_ZAPPED | (watch ? 0 : WATCH_WP)));
			state = WatchPage(run_address >> 12u);
		}
		const bool wp = (state & WATCH_WP) != 0;
		if ((watch && !wp) || (writable && wp)) {
			// The bits first, then the kernel's write protection: a minor fault mapping a page meanwhile maps it as the bits
			// say and then looks at them again (SysVirtualResolveMinorFault), and a write the kernel still protects while the
			// bits say none only finds the lift early. The other way round, a minor fault between the two mapped a page
			// write-protected from the bits read before a lift cleared them: the kernel kept a protection the bits no longer
			// showed, and the page's writes faulted forever (Steam Deck 10-28: a job worker's 27,000 SIGBUS a second at a
			// world load, the game stopped).
			if (watch) SetWatchBits(run_address >> 12u, next, WATCH_WP);
			else ClearWatchBits(run_address >> 12u, next, WATCH_WP);
			if (!WriteProtect(run_address, run_size, watch)) {
				// (No longer a registered view: something else was mapped there.)
				SetWatchPages(run_address >> 12u, next, state & WATCH_PROT);
				if (!SysVirtualProtect(run_address, run_size, mode)) return false;
				continue;
			}
		}
		const int target = watch ? PROT_READ | PROT_WRITE : protect;
		if ((state & WATCH_PROT) != target &&
		    !SysVirtualProtect(run_address, run_size, watch ? VirtualMemory::Mode::ReadWrite : mode))
			return false;
	}
	return true;
}

// (KYTY_WRITE_GUARD_UFFD.) A view mapped over [address, address + size): its protection recorded.
static void RecordMappedProtection(uint64_t address, uint64_t size, int protect) {
	if (!g_record_view_protection || size == 0) return;
	pthread_mutex_lock(&g_virtual_mutex);
	g_protects->Set(address >> 12u, (address + size - 1) >> 12u, static_cast<uint8_t>(protect));
	pthread_mutex_unlock(&g_virtual_mutex);
}

void SysVirtualWriteWatchMapped(uint64_t address, uint64_t size, VirtualMemory::Mode mode) {
	RecordMappedProtection(address, size, get_protection_flag(mode));
	if (!WatchRange(address, size)) return;
	uffdio_register reg {};
	reg.range.start = address;
	reg.range.len   = size;
	reg.mode        = UFFDIO_REGISTER_MODE_WP;
	const bool registered = ioctl(g_uffd, UFFDIO_REGISTER, &reg) == 0;
	ForgetMinor(address >> 12u, (address + size) >> 12u);
	SetWatchPages(address >> 12u, (address + size) >> 12u,
	              static_cast<uint8_t>(get_protection_flag(mode) | (registered ? WATCH_REGISTERED : 0)));
}

void SysVirtualWriteWatchUnmapped(uint64_t address, uint64_t size) {
	if (!WatchRange(address, size)) return;
	ForgetMinor(address >> 12u, (address + size) >> 12u);
	SetWatchPages(address >> 12u, (address + size) >> 12u, 0);
}

void SysVirtualZapNoAccess(bool on) {
	g_zap_no_access = on;
}

// SIGBUS (before the fault handlers): a page not zapped whose PTE is missing, in minor mode (first touched in the view,
// reclaimed, a zap's lift that found it mapped by a fault) mapped again, whatever its bits say (a registration they do
// not show would fault forever), write-protected as its bits say once mapped (a protection's change meanwhile went to a
// page not mapped: its bits changed first, SysVirtualProtectWriteWatch). And a write the kernel write-protects on a page
// of a registered view whose bits say no write protection, not zapped: the kernel's protection lifted (a lift under way,
// its bits cleared already; or one the kernel kept that the bits lost, which faulted forever). False for any other fault
// (a write protection the bits show, a zapped page's: the fault handlers serve them as protections'; a page outside
// minor mode: UFFDIO_CONTINUE refuses it).
static std::atomic<uint64_t> g_stale_write_protects {0};

bool SysVirtualResolveMinorFault(uint64_t address, bool present, bool write) {
	if (g_uffd < 0 || address >> 12u >= WATCH_PAGE_COUNT) return false;
	const auto page  = address >> 12u;
	auto       state = WatchPage(page);
	if (present) {
		if (!write || (state & (WATCH_REGISTERED | WATCH_ZAPPED | WATCH_WP)) != WATCH_REGISTERED ||
		    (state & WATCH_PROT) != (PROT_READ | PROT_WRITE) || !WriteProtect(page << 12u, 0x1000u, false))
			return false;
		// (A protection of the page meanwhile, its bit set before this lift reached the kernel: set again.)
		for (int tries = 0; tries < 8; tries++) {
			const auto now = WatchPage(page);
			if ((now & WATCH_ZAPPED) != 0 || ((now ^ state) & WATCH_WP) == 0) break;
			(void)WriteProtect(page << 12u, 0x1000u, (now & WATCH_WP) != 0);
			state = now;
		}
		// (Mostly a lift under way: another thread's write between its bits and its ioctl.)
		if (const auto count = g_stale_write_protects.fetch_add(1, std::memory_order_relaxed); (count & (count - 1)) == 0) {
			std::printf("KYTY_UFFD_WP: %llu write(s) the kernel still write-protected on pages whose bits had none (a lift "
			            "under way, or a protection the bits lost), the protection lifted (0x%llx)\n",
			            static_cast<unsigned long long>(count + 1), static_cast<unsigned long long>(page << 12u));
			std::fflush(stdout);
		}
		return true;
	}
	if (!g_minor_on || (state & WATCH_ZAPPED) != 0) return false;
	uffdio_continue map {};
	map.range.start = page << 12u;
	map.range.len   = 0x1000u;
	map.mode        = (state & WATCH_WP) != 0 ? UFFDIO_CONTINUE_MODE_WP : 0;
	if (ioctl(g_uffd, UFFDIO_CONTINUE, &map) != 0) return false;
	for (int tries = 0; tries < 8; tries++) {
		const auto now = WatchPage(page);
		if ((now & WATCH_ZAPPED) != 0 || ((now ^ state) & WATCH_WP) == 0) break;
		(void)WriteProtect(page << 12u, 0x1000u, (now & WATCH_WP) != 0);
		state = now;
	}
	return true;
}
#else
bool SysVirtualProtectWriteWatch(uint64_t address, uint64_t size, VirtualMemory::Mode mode) {
	return SysVirtualProtect(address, size, mode);
}

void SysVirtualWriteWatchMapped(uint64_t /*address*/, uint64_t /*size*/, VirtualMemory::Mode /*mode*/) {}

void SysVirtualWriteWatchUnmapped(uint64_t /*address*/, uint64_t /*size*/) {}

void SysVirtualZapNoAccess(bool /*on*/) {}

bool SysVirtualResolveMinorFault(uint64_t /*address*/, bool /*present*/, bool /*write*/) {
	return false;
}
#endif

bool SysVirtualFlushInstructionCache(uint64_t /*address*/, uint64_t /*size*/) {
	return true;
}

} // namespace Common

#endif
