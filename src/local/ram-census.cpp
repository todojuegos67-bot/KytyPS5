#include "ram-census.h"

#include "lean-hostmem.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// (After windows.h.)
#include <psapi.h>

#include <chrono>
#include <thread>
#endif

namespace RamCensus {

namespace {

std::mutex g_query_mutex; // (the census thread and vma.cpp's allocator creation / destruction only)
HeapQuery  g_query = nullptr;

} // namespace

void SetHeapQuery(HeapQuery query) {
	const std::lock_guard lock(g_query_mutex);
	g_query = query;
}

#if !defined(_WIN32)

void Start() {}

#else

namespace {

constexpr uint64_t MiB = uint64_t {1} << 20u;

// pdh.dll at run time (not linked): the few functions and the one value type used.
struct PdhValue {
	DWORD CStatus;
	union {
		LONG     longValue;
		double   doubleValue;
		LONGLONG largeValue;
		LPCSTR   AnsiStringValue;
		LPCWSTR  WideStringValue;
	};
};
constexpr DWORD PdhFmtLarge = 0x00000400;
using PdhOpenQueryW_t           = LONG(WINAPI*)(LPCWSTR, DWORD_PTR, HANDLE*);
using PdhAddEnglishCounterW_t   = LONG(WINAPI*)(HANDLE, LPCWSTR, DWORD_PTR, HANDLE*);
using PdhCollectQueryData_t     = LONG(WINAPI*)(HANDLE);
using PdhGetFormattedCounterValue_t = LONG(WINAPI*)(HANDLE, DWORD, LPDWORD, PdhValue*);

using EnumProcesses_t  = BOOL(WINAPI*)(DWORD*, DWORD, DWORD*);
using QueryImageName_t = BOOL(WINAPI*)(HANDLE, DWORD, LPWSTR, DWORD*);

enum Counter : int {
	Modified,
	StandbyNormal,
	StandbyReserve,
	StandbyCore,
	FreeZero,
	PoolNonpaged,
	PoolPagedResident,
	DriverResident,
	CodeResident,
	CacheBytes,
	CounterCount
};
constexpr const wchar_t* CounterPaths[CounterCount] = {
    L"\\Memory\\Modified Page List Bytes",
    L"\\Memory\\Standby Cache Normal Priority Bytes",
    L"\\Memory\\Standby Cache Reserve Bytes",
    L"\\Memory\\Standby Cache Core Bytes",
    L"\\Memory\\Free & Zero Page List Bytes",
    L"\\Memory\\Pool Nonpaged Bytes",
    L"\\Memory\\Pool Paged Resident Bytes",
    L"\\Memory\\System Driver Resident Bytes",
    L"\\Memory\\System Code Resident Bytes",
    L"\\Memory\\Cache Bytes",
};

struct Pdh {
	PdhCollectQueryData_t         collect = nullptr;
	PdhGetFormattedCounterValue_t value   = nullptr;
	HANDLE                        query   = nullptr;
	HANDLE                        counters[CounterCount] {};

	bool Open() {
		HMODULE dll = LoadLibraryW(L"pdh.dll");
		if (dll == nullptr) return false;
		auto open = reinterpret_cast<PdhOpenQueryW_t>(reinterpret_cast<void*>(GetProcAddress(dll, "PdhOpenQueryW")));
		auto add  = reinterpret_cast<PdhAddEnglishCounterW_t>(
            reinterpret_cast<void*>(GetProcAddress(dll, "PdhAddEnglishCounterW")));
		collect = reinterpret_cast<PdhCollectQueryData_t>(reinterpret_cast<void*>(GetProcAddress(dll, "PdhCollectQueryData")));
		value   = reinterpret_cast<PdhGetFormattedCounterValue_t>(
            reinterpret_cast<void*>(GetProcAddress(dll, "PdhGetFormattedCounterValue")));
		if (open == nullptr || add == nullptr || collect == nullptr || value == nullptr) return false;
		if (open(nullptr, 0, &query) != 0) return false;
		for (int i = 0; i < CounterCount; ++i)
			if (add(query, CounterPaths[i], 0, &counters[i]) != 0) counters[i] = nullptr; // (that one reads -1)
		(void)collect(query);
		return true;
	}
	// Bytes, or -1 when the counter is not there.
	void Read(int64_t* out) {
		const bool ok = query != nullptr && collect(query) == 0;
		for (int i = 0; i < CounterCount; ++i) {
			out[i] = -1;
			if (!ok || counters[i] == nullptr) continue;
			PdhValue v {};
			DWORD    type = 0;
			if (value(counters[i], PdhFmtLarge, &type, &v) == 0 && v.CStatus <= 1) out[i] = v.largeValue;
		}
	}
};

long long M(int64_t bytes) {
	return bytes < 0 ? -1 : static_cast<long long>(bytes / static_cast<int64_t>(MiB));
}
long long M(uint64_t bytes) {
	return static_cast<long long>(bytes / MiB);
}

struct Top {
	uint64_t ws = 0;
	char     name[24] {};
};

void NameOf(HANDLE process, QueryImageName_t query_name, char* out, size_t size) {
	out[0] = '?';
	out[1] = '\0';
	if (query_name == nullptr) return;
	wchar_t path[MAX_PATH];
	DWORD   length = MAX_PATH;
	if (!query_name(process, 0, path, &length) || length == 0) return;
	const wchar_t* base = path;
	for (DWORD i = 0; i < length; ++i)
		if (path[i] == L'\\' || path[i] == L'/') base = path + i + 1;
	size_t n = 0;
	for (; base[n] != L'\0' && n + 1 < size; ++n) out[n] = base[n] < 128 ? static_cast<char>(base[n]) : '_';
	out[n] = '\0';
}

void Run() {
	SetThreadDescription(GetCurrentThread(), L"Kyty.RamCensus");
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	Pdh  pdh;
	const bool have_pdh = pdh.Open();
	HMODULE    kernel   = GetModuleHandleW(L"kernel32.dll");
	auto enum_processes = kernel != nullptr ? reinterpret_cast<EnumProcesses_t>(
	                                              reinterpret_cast<void*>(GetProcAddress(kernel, "K32EnumProcesses")))
	                                        : nullptr;
	auto query_name = kernel != nullptr ? reinterpret_cast<QueryImageName_t>(
	                                          reinterpret_cast<void*>(GetProcAddress(kernel, "QueryFullProcessImageNameW")))
	                                    : nullptr;
	static DWORD pids[4096]; // (reused; one thread)
	const DWORD  self_pid  = GetCurrentProcessId();
	DWORD        last_faults = 0;
	ULONGLONG    last_tick   = GetTickCount64();
	std::printf("[ram-census] on: a line every 30 s (MiB; KYTY_RAM_CENSUS=0 turns it off)%s\n",
	            have_pdh ? "" : "; PDH unavailable: lists and pools read -1");
	std::fflush(stdout);
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(30));
		MEMORYSTATUSEX status {};
		status.dwLength = sizeof(status);
		if (GlobalMemoryStatusEx(&status) == 0) continue;
		const uint64_t total = status.ullTotalPhys, avail = status.ullAvailPhys, in_use = total - avail;

		int64_t c[CounterCount];
		if (have_pdh) pdh.Read(c);
		else
			for (auto& v: c) v = -1;
		const auto pos     = [](int64_t v) { return v > 0 ? v : 0; };
		const int64_t standby = c[StandbyNormal] < 0 ? -1 : pos(c[StandbyNormal]) + pos(c[StandbyReserve]) + pos(c[StandbyCore]);

		PROCESS_MEMORY_COUNTERS_EX2 self {};
		self.cb = sizeof(self);
		uint64_t self_pws = 0;
		if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&self), sizeof(self)) != 0) {
			self_pws = self.PrivateWorkingSetSize;
		} else {
			self.cb = sizeof(PROCESS_MEMORY_COUNTERS_EX);
			(void)K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&self),
			                              sizeof(PROCESS_MEMORY_COUNTERS_EX));
		}
		const ULONGLONG tick    = GetTickCount64();
		const double    seconds = tick > last_tick ? static_cast<double>(tick - last_tick) / 1000.0 : 1.0;
		const auto      faults  = static_cast<long long>(static_cast<double>(self.PageFaultCount - last_faults) / seconds);
		last_faults             = self.PageFaultCount;
		last_tick               = tick;

		uint64_t others = 0;
		int      counted = 0;
		Top      top[3] {};
		DWORD    bytes   = 0;
		if (enum_processes != nullptr && enum_processes(pids, sizeof(pids), &bytes)) {
			for (DWORD i = 0; i < bytes / sizeof(DWORD); ++i) {
				if (pids[i] == 0 || pids[i] == self_pid) continue;
				HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pids[i]);
				if (process == nullptr) continue;
				PROCESS_MEMORY_COUNTERS counters {};
				counters.cb = sizeof(counters);
				if (K32GetProcessMemoryInfo(process, &counters, sizeof(counters)) != 0) {
					const uint64_t ws = counters.WorkingSetSize;
					others += ws;
					counted++;
					int slot = -1;
					for (int t = 0; t < 3; ++t)
						if (ws > top[t].ws) {
							slot = t;
							break;
						}
					if (slot >= 0) {
						for (int t = 2; t > slot; --t) top[t] = top[t - 1];
						top[slot].ws = ws;
						NameOf(process, query_name, top[slot].name, sizeof(top[slot].name));
					}
				}
				CloseHandle(process);
			}
		}

		Heap heaps[16] {};
		int  heap_count = 0;
		{
			const std::lock_guard lock(g_query_mutex);
			if (g_query != nullptr) heap_count = g_query(heaps, 16);
		}
		char     vk[400];
		size_t   at   = 0;
		uint64_t host = 0;
		vk[0]         = '\0';
		for (int h = 0; h < heap_count && at + 64 < sizeof(vk); ++h) {
			if (!heaps[h].device_local) host += heaps[h].usage;
			at += static_cast<size_t>(std::snprintf(vk + at, sizeof(vk) - at, " h%d%s %lld/%lld vma %lld", h,
			                                        heaps[h].device_local ? "D" : "", M(heaps[h].usage),
			                                        M(heaps[h].budget), M(heaps[h].blocks)));
		}

		const int64_t explained = static_cast<int64_t>(self.WorkingSetSize + others) + pos(c[Modified]) + pos(c[PoolNonpaged]) +
		                          pos(c[PoolPagedResident]) + pos(c[DriverResident]) + pos(c[CodeResident]);
		const int64_t unexplained = static_cast<int64_t>(in_use) - explained;

		std::printf("[ram-census] sys %lld/%lld/%lld lists mod %lld stby %lld free %lld kern np %lld pg %lld drv %lld code %lld "
		            "cache %lld self ws %lld pws %lld commit %lld pf/s %lld others %lld (%d: %s %lld, %s %lld, %s %lld) vk%s "
		            "host %lld lean %s skip %lld/%lld of %lld unexplained %lld\n",
		            M(total), M(in_use), M(avail), M(c[Modified]), M(standby), M(c[FreeZero]), M(c[PoolNonpaged]),
		            M(c[PoolPagedResident]), M(c[DriverResident]), M(c[CodeResident]), M(c[CacheBytes]),
		            M(static_cast<uint64_t>(self.WorkingSetSize)), M(self_pws), M(static_cast<uint64_t>(self.PrivateUsage)),
		            faults, M(others), counted, top[0].name, M(top[0].ws), top[1].name, M(top[1].ws), top[2].name,
		            M(top[2].ws), heap_count != 0 ? vk : " -", M(host), LeanHostMem::Enabled() ? "on" : "off",
		            M(LeanHostMem::g_skipped_copy.load(std::memory_order_relaxed)),
		            M(LeanHostMem::g_skipped_zero.load(std::memory_order_relaxed)),
		            M(LeanHostMem::g_written.load(std::memory_order_relaxed)), M(unexplained));
		std::fflush(stdout);
	}
}

} // namespace

void Start() {
	static std::once_flag once;
	std::call_once(once, [] {
		if (const char* value = std::getenv("KYTY_RAM_CENSUS"); value != nullptr && value[0] == '0') return;
		std::thread(Run).detach();
	});
}

#endif

} // namespace RamCensus
