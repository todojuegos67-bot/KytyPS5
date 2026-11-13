#include "kernel/regionProbe.h"

#include "common/common.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep

#include <malloc.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

namespace RegionProbe {

namespace {

constexpr uint64_t DefaultAddress = 0xff60000000ull;
constexpr uint64_t MiB            = uint64_t {1} << 20u;

std::atomic<const char*>              g_last {"(before the first checkpoint)"};
std::once_flag                        g_start;
std::chrono::steady_clock::time_point g_t0;

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_REGION_PROBE");
		return !(value != nullptr && value[0] == '0');
	}();
	return enabled;
}

uint64_t Address() {
	static const uint64_t address = [] {
		const char* value = std::getenv("KYTY_REGION_PROBE_ADDR");
		if (value == nullptr || value[0] == '\0') return DefaultAddress;
		const uint64_t parsed = std::strtoull(value, nullptr, 16);
		return parsed != 0 ? parsed : DefaultAddress;
	}();
	return address;
}

unsigned long long Elapsed() {
	return static_cast<unsigned long long>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - g_t0).count());
}

unsigned long long UnixMs() {
	return static_cast<unsigned long long>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

struct Allocation {
	bool     free          = true;
	uint64_t base          = 0;
	uint64_t reserved      = 0; // free: the size of the free hole there
	uint64_t committed     = 0;
	uint32_t regions       = 0;
	DWORD    type          = 0;
	DWORD    alloc_protect = 0;
	DWORD    base_state    = 0;
	DWORD    base_protect  = 0;
};

bool Query(uint64_t at, MEMORY_BASIC_INFORMATION* info) {
	return VirtualQuery(reinterpret_cast<LPCVOID>(at), info, sizeof(*info)) == sizeof(*info);
}

// The whole allocation (one AllocationBase) that holds address: reserved and committed bytes, read-only.
Allocation Describe(uint64_t address) {
	Allocation               a;
	MEMORY_BASIC_INFORMATION info {};
	if (!Query(address, &info)) return a;
	if (info.State == MEM_FREE) {
		a.reserved = static_cast<uint64_t>(info.RegionSize);
		return a;
	}
	a.free          = false;
	a.base          = reinterpret_cast<uint64_t>(info.AllocationBase);
	a.type          = info.Type;
	a.alloc_protect = info.AllocationProtect;
	for (uint64_t at = a.base; a.regions < 100000u;) {
		if (!Query(at, &info)) break;
		if (info.State == MEM_FREE || reinterpret_cast<uint64_t>(info.AllocationBase) != a.base) break;
		if (at == a.base) {
			a.base_state   = info.State;
			a.base_protect = info.Protect;
		}
		a.reserved += static_cast<uint64_t>(info.RegionSize);
		if (info.State == MEM_COMMIT) a.committed += static_cast<uint64_t>(info.RegionSize);
		a.regions++;
		const uint64_t next = reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize);
		if (next <= at) break;
		at = next;
	}
	return a;
}

const char* TypeName(DWORD type) {
	switch (type) {
		case MEM_PRIVATE: return "private";
		case MEM_MAPPED: return "mapped";
		case MEM_IMAGE: return "image";
		default: return "?";
	}
}

// One line: what is at address now.
void Print(const char* what, uint64_t address, const Allocation& a, const Allocation& next) {
	if (a.free) {
		std::printf("[region-probe] +%llu ms (unix %llu) %s: 0x%llx free (a hole of %llu MiB)\n", Elapsed(), UnixMs(), what,
		            static_cast<unsigned long long>(address), static_cast<unsigned long long>(a.reserved / MiB));
	} else {
		std::printf("[region-probe] +%llu ms (unix %llu) %s: 0x%llx in allocation 0x%llx: %llu MiB reserved, %llu MiB "
		            "committed (%u regions), %s, allocation protect 0x%lx, base page state 0x%lx protect 0x%lx; next "
		            "allocation 0x%llx: %s %llu MiB reserved, %llu MiB committed\n",
		            Elapsed(), UnixMs(), what, static_cast<unsigned long long>(address),
		            static_cast<unsigned long long>(a.base), static_cast<unsigned long long>(a.reserved / MiB),
		            static_cast<unsigned long long>(a.committed / MiB), a.regions, TypeName(a.type),
		            static_cast<unsigned long>(a.alloc_protect), static_cast<unsigned long>(a.base_state),
		            static_cast<unsigned long>(a.base_protect), static_cast<unsigned long long>(a.base + a.reserved),
		            next.free ? "free," : TypeName(next.type), static_cast<unsigned long long>(next.reserved / MiB),
		            static_cast<unsigned long long>(next.committed / MiB));
	}
	std::fflush(stdout);
}

Allocation DescribeNext(const Allocation& a) {
	if (a.free || a.reserved == 0) return {};
	return Describe(a.base + a.reserved);
}

// The allocation's first 0x60 bytes (only when its first page is committed and readable; ReadProcessMemory fails
// instead of faulting): an NT heap segment has 0xffeeffee at +0x10 and its heap at +0x28 (x64 _HEAP_SEGMENT), a Segment
// Heap 0xddeeddee at +0x10. Returns whether it could read them.
bool PrintHeader(const Allocation& a) {
	constexpr DWORD Readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ |
	                           PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
	if (a.free || a.base_state != MEM_COMMIT || (a.base_protect & (PAGE_GUARD | PAGE_NOACCESS)) != 0 ||
	    (a.base_protect & Readable) == 0) {
		return false;
	}
	unsigned char bytes[0x60] {};
	SIZE_T        read = 0;
	if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(a.base), bytes, sizeof(bytes), &read) == 0 ||
	    read != sizeof(bytes)) {
		return false;
	}
	uint32_t signature = 0;
	uint64_t heap      = 0;
	uint64_t base      = 0;
	uint32_t pages     = 0;
	std::memcpy(&signature, bytes + 0x10, sizeof(signature));
	std::memcpy(&heap, bytes + 0x28, sizeof(heap));
	std::memcpy(&base, bytes + 0x30, sizeof(base));
	std::memcpy(&pages, bytes + 0x38, sizeof(pages));

	HANDLE      heaps[1024] {};
	const DWORD count     = GetProcessHeaps(1024, heaps);
	int         base_heap = -1;
	int         ptr_heap  = -1;
	if (count <= 1024) {
		for (DWORD i = 0; i < count; i++) {
			const auto handle = reinterpret_cast<uint64_t>(heaps[i]);
			if (handle == a.base) base_heap = static_cast<int>(i);
			if (handle == heap) ptr_heap = static_cast<int>(i);
		}
	}
	const auto process_heap = reinterpret_cast<uint64_t>(GetProcessHeap());
	const auto crt_heap     = static_cast<uint64_t>(_get_heap_handle());
	const char* kind        = signature == 0xffeeffeeu ? "NT heap segment signature"
	                          : signature == 0xddeeddeeu ? "Segment Heap signature"
	                                                     : "no heap signature";
	std::printf("[region-probe] header of 0x%llx: +0x10 = 0x%08x (%s), +0x28 = 0x%llx (%s%d), +0x30 = 0x%llx, +0x38 = %u "
	            "pages; base is heap #%d of %lu; process heap 0x%llx, CRT heap 0x%llx\n",
	            static_cast<unsigned long long>(a.base), signature, kind, static_cast<unsigned long long>(heap),
	            heap == process_heap ? "the process heap, heap #" : "heap #", ptr_heap,
	            static_cast<unsigned long long>(base), pages, base_heap, static_cast<unsigned long>(count),
	            static_cast<unsigned long long>(process_heap), static_cast<unsigned long long>(crt_heap));
	char hex[sizeof(bytes) * 2 + 1] {};
	for (size_t i = 0; i < sizeof(bytes); i++) {
		static constexpr char Digits[] = "0123456789abcdef";
		hex[i * 2]                     = Digits[bytes[i] >> 4u];
		hex[i * 2 + 1]                 = Digits[bytes[i] & 0xfu];
	}
	std::printf("[region-probe] first bytes of 0x%llx: %s\n", static_cast<unsigned long long>(a.base), hex);
	std::fflush(stdout);
	return true;
}

void Poll() {
	const uint64_t address  = Address();
	constexpr auto Duration = std::chrono::minutes(10);
	// Until something is there: one VirtualQuery a millisecond.
	for (;;) {
		if (std::chrono::steady_clock::now() - g_t0 > Duration) return;
		MEMORY_BASIC_INFORMATION info {};
		if (Query(address, &info) && info.State != MEM_FREE) break;
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	Allocation a = Describe(address);
	char       what[256];
	std::snprintf(what, sizeof(what), "poll: appeared, the last checkpoint passed: %s", g_last.load());
	Print(what, address, a, DescribeNext(a));
	bool     header_done = PrintHeader(a);
	uint64_t shown_base  = a.base;
	uint64_t shown_size  = a.reserved;
	uint64_t shown       = a.committed;
	// Then each change of its commit by 32 MiB or more (or of the allocation itself), four times a second.
	while (std::chrono::steady_clock::now() - g_t0 <= Duration) {
		std::this_thread::sleep_for(std::chrono::milliseconds(250));
		a = Describe(address);
		const uint64_t delta = a.committed > shown ? a.committed - shown : shown - a.committed;
		if (a.free || a.base != shown_base || a.reserved != shown_size || delta >= 32 * MiB) {
			std::snprintf(what, sizeof(what), "poll: changed, the last checkpoint passed: %s", g_last.load());
			Print(what, address, a, DescribeNext(a));
			shown_base = a.base;
			shown_size = a.reserved;
			shown      = a.committed;
		}
		if (!header_done) header_done = PrintHeader(a);
	}
	std::printf("[region-probe] polling stopped after 10 minutes\n");
	std::fflush(stdout);
}

} // namespace

void Check(const char* checkpoint) {
	if (!Enabled() || checkpoint == nullptr) return;
	std::call_once(g_start, [] {
		g_t0 = std::chrono::steady_clock::now();
		std::printf("[region-probe] watching 0x%llx (KYTY_REGION_PROBE=0: off)\n",
		            static_cast<unsigned long long>(Address()));
		std::fflush(stdout);
		std::thread(Poll).detach();
	});
	g_last.store(checkpoint);
	const Allocation a = Describe(Address());
	Print(checkpoint, Address(), a, DescribeNext(a));
}

} // namespace RegionProbe

#else

namespace RegionProbe {

void Check(const char* /*checkpoint*/) {}

} // namespace RegionProbe

#endif
