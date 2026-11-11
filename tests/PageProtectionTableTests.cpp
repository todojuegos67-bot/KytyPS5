#include "common/pageProtectionTable.h"
#include "common/platform/sysVirtual.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <csignal>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "PageProtectionTableTests: %s\n", message);
		std::abort();
	}
}

void CheckSparseTable() {
	Common::PageProtectionTable table;
	constexpr uintptr_t         high = uintptr_t {0x7f0012345000} >> 12;
	Check(table.Get(0) == -1 && table.Get(high) == -1, "empty table has permissions");
	table.Set(1023, 1024, 3);
	table.Set(high, high + 2, 0); // Known PROT_NONE is distinct from an absent page.
	Check(table.Get(1022) == -1 && table.Get(1023) == 3 && table.Get(1024) == 3 &&
	          table.Get(1025) == -1 && table.Get(high) == 0 && table.RegionCount() == 3,
	      "region boundary, high address or known no-access page was lost");
	table.Erase(1023, 1023);
	Check(table.Get(1023) == -1 && table.Get(1024) == 3 && table.RegionCount() == 2,
	      "partial erase affected a neighboring region");
	table.Set(1024, 2047, 5);
	table.Erase(1025, 2046);
	Check(table.Get(1024) == 5 && table.Get(1025) == -1 && table.Get(2047) == 5,
	      "middle erase lost region endpoints");
	table.Erase(0, std::numeric_limits<uintptr_t>::max());
	Check(table.RegionCount() == 0, "full sparse erase retained a region");
	const auto top = std::numeric_limits<uintptr_t>::max();
	table.Set(top - 2, top, 7);
	Check(table.Get(top) == 7 && table.Get(top - 3) == -1, "maximum page number wrapped");
	table.Erase(top - 1, top);
	Check(table.Get(top - 2) == 7 && table.Get(top) == -1, "maximum page erase wrapped");
	table.Erase(top - 2, top - 2);
	Check(table.RegionCount() == 0, "empty final region was not reclaimed");
}

void CheckRandomTransactions() {
	Common::PageProtectionTable table;
	std::map<uintptr_t, int>    reference;
	uint32_t                    random = 0x91abcd34;
	const auto                  next   = [&] {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        return random;
	};
	for (uint32_t iteration = 0; iteration < 10000; ++iteration) {
		const auto first = next() % 8192;
		const auto last  = first + next() % 3072;
		if (next() % 3 == 0) {
			table.Erase(first, last);
			reference.erase(reference.lower_bound(first), reference.upper_bound(last));
		} else {
			const auto mode = next() & 7;
			table.Set(first, last, mode);
			for (uintptr_t page = first; page <= last; ++page)
				reference[page] = mode;
		}
		for (const auto page: {first - uintptr_t(first != 0), uintptr_t(first), uintptr_t(last),
		                       uintptr_t(last + 1), uintptr_t(next() % 12288)}) {
			const auto found = reference.find(page);
			Check(table.Get(page) == (found == reference.end() ? -1 : found->second),
			      "random range update differs from per-page map");
		}
	}
	for (uintptr_t page = 0; page < 12288; ++page) {
		const auto found = reference.find(page);
		Check(table.Get(page) == (found == reference.end() ? -1 : found->second),
		      "final full comparison differs from per-page map");
	}
}

void CheckPlatformProtection() {
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	// This metadata table serves the POSIX backend. Use the host page size on macOS too.
	const auto page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
	using namespace Common;
	using Mode = VirtualMemory::Mode;
	SysVirtualInit();
	const auto address = SysVirtualAllocAligned(0, (page * 9), Mode::ReadWrite, (page * 16));
	Check(address && address != UINT64_MAX, "platform allocation failed");
	*reinterpret_cast<volatile uint32_t*>(address) = 0xabcddcba;
	Mode old                                       = Mode::NoAccess;
	Check(SysVirtualProtect(address + 1, page, Mode::Read, &old) && old == Mode::ReadWrite,
	      "unaligned protection lost prior permissions");
	Check(SysVirtualProtect(address + page, 1, Mode::ReadWrite, &old) && old == Mode::Read,
	      "rounded second page did not retain its permissions");
	Check(SysVirtualProtect(address + (page * 2), 1, Mode::ReadWrite, &old) &&
	          old == Mode::ReadWrite,
	      "protection changed an adjacent page");
	Check(SysVirtualProtect(address, page, Mode::ReadWrite, &old) && old == Mode::Read &&
	          *reinterpret_cast<volatile uint32_t*>(address) == 0xabcddcba,
	      "restoring access lost permissions or memory contents");
	Check(SysVirtualFreeRange(address + (page * 4), page), "middle unmap failed");
	Check(!SysVirtualProtect(address + (page * 4), page, Mode::Read, &old) && old == Mode::NoAccess,
	      "unmapped page retained metadata or mprotect succeeded");
	Check(SysVirtualAllocFixed(address + (page * 4), page, Mode::NoAccess), "middle remap failed");
	Check(SysVirtualProtect(address + (page * 4), page, Mode::ReadWrite, &old) &&
	          old == Mode::NoAccess,
	      "new allocation inherited retired permissions");
	*reinterpret_cast<volatile uint32_t*>(address + (page * 4)) = 0x12345678;
	Check(SysVirtualFreeRange(address, (page * 9)), "split allocation release failed");
#endif
}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
using namespace Common;

volatile sig_atomic_t g_watch_signal  = 0;
volatile uint64_t     g_watch_address = 0;

// As the GPU trackers' fault handler: note the write, then let it through.
void WatchFault(int signal_number, siginfo_t* info, void* /*context*/) {
	const auto address = reinterpret_cast<uint64_t>(info->si_addr) & ~uint64_t {0xfff};
	g_watch_signal     = signal_number;
	g_watch_address    = address;
	if (!SysVirtualProtectWriteWatch(address, 0x1000, VirtualMemory::Mode::ReadWrite)) std::abort();
}

void Write(uint64_t address, uint32_t value) {
	g_watch_signal                                 = 0;
	*reinterpret_cast<volatile uint32_t*>(address) = value;
}
#endif

// KYTY_UFFD_WP=1: the trackers' Read is userfaultfd write protection (SIGBUS) on a shared view, else mprotect (SIGSEGV);
// either way a write to a watched page faults once and lands, and other pages do not fault.
void CheckWriteWatch() {
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
	using Mode = VirtualMemory::Mode;
	setenv("KYTY_UFFD_WP", "1", 1);
	SysVirtualInit();
	constexpr uint64_t page = 0x1000;
	constexpr uint64_t size = page * 4;
	const int          fd   = memfd_create("write_watch", MFD_CLOEXEC);
	Check(fd >= 0 && ftruncate(fd, size) == 0, "memfd failed");
	// Inside the watched low 1 TiB, as the guest's views.
	auto* ptr = mmap(reinterpret_cast<void*>(uint64_t {0x7e0000000}), size, PROT_READ | PROT_WRITE,
	                 MAP_SHARED | MAP_FIXED_NOREPLACE, fd, 0);
	Check(ptr != MAP_FAILED, "view map failed");
	const auto address = reinterpret_cast<uint64_t>(ptr);
	SysVirtualWriteWatchMapped(address, size, Mode::ReadWrite);

	struct sigaction action {};
	struct sigaction old_bus {};
	struct sigaction old_segv {};
	action.sa_sigaction = WatchFault;
	action.sa_flags     = SA_SIGINFO;
	sigemptyset(&action.sa_mask);
	sigaction(SIGBUS, &action, &old_bus);
	sigaction(SIGSEGV, &action, &old_segv);

	Write(address + page, 1);
	Check(SysVirtualProtectWriteWatch(address + page, page, Mode::Read), "watch failed");
	Check(*reinterpret_cast<volatile uint32_t*>(address + page) == 1, "watched page not readable");
	Write(address, 2);
	Check(g_watch_signal == 0, "write next to a watched page faulted");
	Write(address + page, 3);
	const int signal_number = g_watch_signal;
	Check((signal_number == SIGBUS || signal_number == SIGSEGV) && g_watch_address == address + page,
	      "write to a watched page did not fault there");
	Check(*reinterpret_cast<volatile uint32_t*>(address + page) == 3, "faulted write was lost");
	Write(address + page, 4);
	Check(g_watch_signal == 0, "unwatched page faulted again");

	// Watch, unwatch: no fault; no access and back keeps the contents.
	Check(SysVirtualProtectWriteWatch(address, size, Mode::Read), "range watch failed");
	Check(SysVirtualProtectWriteWatch(address, size, Mode::ReadWrite), "range unwatch failed");
	Write(address + (page * 3), 5);
	Check(g_watch_signal == 0, "unwatched range faulted");
	Check(SysVirtualProtectWriteWatch(address + page, page, Mode::NoAccess) &&
	          SysVirtualProtectWriteWatch(address + page, page, Mode::ReadWrite) &&
	          *reinterpret_cast<volatile uint32_t*>(address + page) == 4,
	      "no access and back lost the contents");

	// A view mapped again over the old one is watched again.
	SysVirtualWriteWatchUnmapped(address, size);
	Check(mmap(ptr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) == ptr, "view remap failed");
	SysVirtualWriteWatchMapped(address, size, Mode::ReadWrite);
	Check(SysVirtualProtectWriteWatch(address + (page * 2), page, Mode::Read), "watch after remap failed");
	Write(address + (page * 2), 6);
	Check(g_watch_signal == signal_number && g_watch_address == address + (page * 2),
	      "write after remap did not fault the same way");

	sigaction(SIGBUS, &old_bus, nullptr);
	sigaction(SIGSEGV, &old_segv, nullptr);
	SysVirtualWriteWatchUnmapped(address, size);
	munmap(ptr, size);
	close(fd);
	unsetenv("KYTY_UFFD_WP");
	std::printf("PageProtectionTableTests: write watch by %s\n",
	            signal_number == SIGBUS ? "userfaultfd write protection" : "mprotect");
#endif
}
} // namespace

int main() {
	CheckSparseTable();
	CheckRandomTransactions();
	CheckPlatformProtection();
	CheckWriteWatch();
	std::puts("PageProtectionTableTests: all cases passed");
}
