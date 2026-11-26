#include "libs/trim.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h> // _heapmin
#include <psapi.h>  // GetProcessMemoryInfo
#include <tlhelp32.h>
// (windows.h / rpcndr.h macros that clash with ordinary names)
#ifdef small
#undef small
#endif
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#ifdef interface
#undef interface
#endif
#endif

namespace Libs::Trim {
namespace {

std::mutex              g_mutex;
std::condition_variable g_wake;
bool                    g_prefetch_done = false;
std::atomic<bool>       g_started {false};

#if defined(_WIN32)

uint64_t EnvU64(const char* name, uint64_t fallback) {
	const char* v = std::getenv(name);
	if (v == nullptr || *v == '\0') return fallback;
	return std::strtoull(v, nullptr, 10);
}

constexpr uint64_t MiB(uint64_t bytes) { return bytes >> 20u; }


// HEAP_SUMMARY (heapapi.h), taken by name: not every SDK declares it.
struct HeapSummaryData {
	DWORD  cb;
	SIZE_T allocated;
	SIZE_T committed;
	SIZE_T reserved;
	SIZE_T max_reserve;
};
using HeapSummaryFn = BOOL(WINAPI*)(HANDLE, DWORD, HeapSummaryData*);

struct HeapNumbers {
	bool     ok        = false;
	uint64_t allocated = 0; // bytes in blocks in use
	uint64_t committed = 0;
};

HeapNumbers ProcessHeapNumbers() {
	static const auto fn = [] {
		HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
		return kernel != nullptr ? reinterpret_cast<HeapSummaryFn>(reinterpret_cast<void*>(GetProcAddress(kernel, "HeapSummary")))
		                         : static_cast<HeapSummaryFn>(nullptr);
	}();
	HeapNumbers n;
	if (fn == nullptr) return n;
	HeapSummaryData s {};
	s.cb = sizeof(s);
	if (fn(GetProcessHeap(), 0, &s) == 0) return n;
	n.ok        = true;
	n.allocated = s.allocated;
	n.committed = s.committed;
	return n;
}

// (private bytes, working set)
std::pair<uint64_t, uint64_t> ProcessUsage() {
	PROCESS_MEMORY_COUNTERS_EX counters {};
	counters.cb = sizeof(counters);
	if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters)) == 0)
		return {0, 0};
	return {counters.PrivateUsage, counters.WorkingSetSize};
}

void HeapPass(const char* why) {
	const auto heap_before = ProcessHeapNumbers();
	const auto use_before  = ProcessUsage();
	const auto begin       = std::chrono::steady_clock::now();

	(void)_heapmin();
	HANDLE      heaps[256];
	const DWORD count = GetProcessHeaps(256, heaps);
	double      longest_ms = 0;
	for (DWORD i = 0; i < count && i < 256; ++i) {
		const auto one = std::chrono::steady_clock::now();
		(void)HeapCompact(heaps[i], 0);
		longest_ms = (std::max)(longest_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - one).count());
	}
	// HeapOptimizeResources (3): each heap's caches dropped and as much of its free memory decommitted as it can.
	struct {
		DWORD version;
		DWORD flags;
	} optimize {1, 0};
	const bool optimized =
	    HeapSetInformation(nullptr, static_cast<HEAP_INFORMATION_CLASS>(3), &optimize, sizeof(optimize)) != 0;

	const double total_ms   = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
	const auto   heap_after = ProcessHeapNumbers();
	const auto   use_after  = ProcessUsage();
	const auto   saved_private =
	    static_cast<long long>(MiB(use_before.first)) - static_cast<long long>(MiB(use_after.first));
	const auto saved_ws = static_cast<long long>(MiB(use_before.second)) - static_cast<long long>(MiB(use_after.second));

	if (heap_before.ok && heap_after.ok) {
		std::printf("[trim] heap (%s): C++ heap committed %llu -> %llu MiB, in use %llu MiB (free kept before: %llu MiB); "
		            "process private %llu -> %llu MiB, working set %llu -> %llu MiB: RAM ahorrada %lld MiB (private), %lld MiB "
		            "(working set); %lu heaps, %.1f ms (the longest heap held %.1f ms), HeapOptimizeResources %s\n",
		            why, static_cast<unsigned long long>(MiB(heap_before.committed)),
		            static_cast<unsigned long long>(MiB(heap_after.committed)),
		            static_cast<unsigned long long>(MiB(heap_after.allocated)),
		            static_cast<unsigned long long>(MiB(heap_before.committed > heap_before.allocated
		                                                    ? heap_before.committed - heap_before.allocated
		                                                    : 0)),
		            static_cast<unsigned long long>(MiB(use_before.first)), static_cast<unsigned long long>(MiB(use_after.first)),
		            static_cast<unsigned long long>(MiB(use_before.second)), static_cast<unsigned long long>(MiB(use_after.second)),
		            saved_private, saved_ws, static_cast<unsigned long>(count), total_ms, longest_ms, optimized ? "ok" : "refused");
	} else {
		std::printf("[trim] heap (%s): process private %llu -> %llu MiB, working set %llu -> %llu MiB: RAM ahorrada %lld MiB "
		            "(private), %lld MiB (working set); %lu heaps, %.1f ms (HeapSummary not available), "
		            "HeapOptimizeResources %s\n",
		            why, static_cast<unsigned long long>(MiB(use_before.first)), static_cast<unsigned long long>(MiB(use_after.first)),
		            static_cast<unsigned long long>(MiB(use_before.second)), static_cast<unsigned long long>(MiB(use_after.second)),
		            saved_private, saved_ws, static_cast<unsigned long>(count), total_ms, optimized ? "ok" : "refused");
	}
	std::fflush(stdout);
}

uint64_t FileTime100ns(const FILETIME& t) { return (static_cast<uint64_t>(t.dwHighDateTime) << 32u) | t.dwLowDateTime; }

std::string ThreadName(HANDLE thread, uint32_t tid) {
	std::string name;
	PWSTR       description = nullptr;
	if (SUCCEEDED(GetThreadDescription(thread, &description)) && description != nullptr) {
		for (const wchar_t* c = description; *c != L'\0' && name.size() < 48; ++c)
			name.push_back(*c < 128 ? static_cast<char>(*c) : '?');
		LocalFree(description);
	}
	// Numbered workers ("Kyty.Prefetch 12", "SDLAudioP3") go in one group: trailing digits and blanks off.
	while (!name.empty() && ((name.back() >= '0' && name.back() <= '9') || name.back() == ' ' || name.back() == '#'))
		name.pop_back();
	if (name.empty()) name = "(no name)";
	(void)tid;
	return name;
}

struct Seen {
	uint64_t    cpu_100ns = 0;
	std::string name;
};

// One census: every thread's CPU time since the last one, grouped by name.
void Census(std::unordered_map<uint32_t, Seen>& seen, double period_s, bool print = true) {
	const auto  begin   = std::chrono::steady_clock::now();
	const DWORD process = GetCurrentProcessId();
	HANDLE      snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) return;
	struct Group {
		double   cpu_s   = 0;
		uint32_t threads = 0;
		uint32_t waiting = 0;
	};
	std::unordered_map<std::string, Group>  groups;
	std::unordered_map<uint32_t, Seen>      now_seen;
	uint32_t                                threads = 0, waiting = 0;
	double                                  total_s = 0;
	THREADENTRY32                           entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL more = Thread32First(snapshot, &entry); more != 0; more = Thread32Next(snapshot, &entry)) {
		if (entry.th32OwnerProcessID != process) continue;
		HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
		if (thread == nullptr) continue;
		FILETIME created {}, exited {}, kernel {}, user {};
		if (GetThreadTimes(thread, &created, &exited, &kernel, &user) != 0) {
			const uint64_t cpu = FileTime100ns(kernel) + FileTime100ns(user);
			const auto     it  = seen.find(entry.th32ThreadID);
			Seen           s;
			s.cpu_100ns     = cpu;
			s.name          = it != seen.end() ? it->second.name : ThreadName(thread, entry.th32ThreadID);
			const double d  = it != seen.end() && cpu >= it->second.cpu_100ns ? static_cast<double>(cpu - it->second.cpu_100ns) * 1e-7
			                                                                  : static_cast<double>(cpu) * 1e-7;
			auto& g = groups[s.name];
			g.cpu_s += d;
			g.threads++;
			threads++;
			total_s += d;
			if (d < period_s * 0.001) {
				g.waiting++;
				waiting++;
			}
			now_seen.emplace(entry.th32ThreadID, std::move(s));
		}
		CloseHandle(thread);
	}
	CloseHandle(snapshot);
	seen.swap(now_seen);
	if (!print) return;

	std::vector<std::pair<std::string, Group>> order(groups.begin(), groups.end());
	std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.second.cpu_s > b.second.cpu_s; });
	std::string line;
	char        piece[160];
	for (size_t i = 0; i < order.size() && i < 24; ++i) {
		const auto& [name, g] = order[i];
		std::snprintf(piece, sizeof(piece), "%s%s x%u %.1f%%%s", i == 0 ? "" : ", ", name.c_str(), g.threads,
		              100.0 * g.cpu_s / period_s, g.waiting == g.threads ? " (solo espera)" : "");
		line += piece;
	}
	std::string idle;
	for (const auto& [name, g]: order) {
		if (g.waiting != g.threads || g.threads == 0) continue;
		std::snprintf(piece, sizeof(piece), "%s%s x%u", idle.empty() ? "" : ", ", name.c_str(), g.threads);
		idle += piece;
		if (idle.size() > 1500) break;
	}
	const double took_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
	std::printf("[trim-cpu] %.0f s: process %.1f%% of a CPU in %u threads, %u of them under 0.1%% (only wait); by group: %s (%.1f ms)\n",
	            period_s, 100.0 * total_s / period_s, threads, waiting, line.c_str(), took_ms);
	std::printf("[trim-cpu] groups that only waited: %s\n", idle.empty() ? "-" : idle.c_str());
	std::fflush(stdout);
}

void Thread() {
	(void)SetThreadDescription(GetCurrentThread(), L"Kyty.Trim");
	const uint64_t census_s = EnvU64("KYTY_TRIM_CENSUS_S", 30);
	const auto     started  = std::chrono::steady_clock::now();
	bool           heap_done = false;
	std::unordered_map<uint32_t, Seen> seen;
	auto           last_census = std::chrono::steady_clock::now();
	if (census_s != 0) Census(seen, 1.0, false); // (the baseline only: nothing printed)
	for (;;) {
		bool prefetch_done = false;
		{
			std::unique_lock lock(g_mutex);
			g_wake.wait_for(lock, std::chrono::seconds(1), [] { return g_prefetch_done; });
			prefetch_done = g_prefetch_done;
		}
		const auto now = std::chrono::steady_clock::now();
		if (!heap_done && (prefetch_done || now - started >= std::chrono::seconds(180))) {
			heap_done = true;
			HeapPass(prefetch_done ? "shader prefetch done" : "180 s, no shader prefetch end seen");
		}
		if (census_s != 0 && now - last_census >= std::chrono::seconds(census_s)) {
			Census(seen, std::chrono::duration<double>(now - last_census).count());
			last_census = now;
		}
		if (heap_done && census_s == 0) return;
	}
}

#endif // _WIN32

} // namespace

bool Enabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TRIM");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

void Start() {
	if (!Enabled() || g_started.exchange(true)) return;
#if defined(_WIN32)
	std::printf("[trim] on (KYTY_TRIM=1): one C++ heap pass after the shader prefetch (free blocks decommitted, no page "
	            "file); thread census every %llu s (KYTY_TRIM_CENSUS_S, 0 = off). PS5 services: left as they are (already "
	            "bookkeeping only; trophies untouched)\n",
	            static_cast<unsigned long long>(EnvU64("KYTY_TRIM_CENSUS_S", 30)));
	std::thread(Thread).detach();
#else
	std::printf("[trim] KYTY_TRIM=1: Windows only, nothing done\n");
#endif
}

void AfterShaderPrefetch() {
	if (!Enabled()) return;
	{
		std::lock_guard lock(g_mutex);
		if (g_prefetch_done) return;
		g_prefetch_done = true;
	}
	g_wake.notify_all();
}

} // namespace Libs::Trim
