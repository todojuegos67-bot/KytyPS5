#include "soft-release.h"

#include "perf-overlay.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <malloc.h>
#include <psapi.h>
#endif

namespace SoftRelease {

namespace {
std::mutex                                               g_mutex;
std::vector<std::pair<const char*, uint64_t (*)()>>      g_sources;

#if defined(_WIN32)
// (private bytes, working set)
std::pair<uint64_t, uint64_t> Usage() {
	PROCESS_MEMORY_COUNTERS_EX counters {};
	counters.cb = sizeof(counters);
	if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters)) == 0)
		return {0, 0};
	return {counters.PrivateUsage, counters.WorkingSetSize};
}

// The heaps' free blocks decommitted (no page file: a decommitted page has nothing to write). Returns the heaps' count.
size_t CompactHeaps() {
	_heapmin();
	HANDLE heaps[256];
	const DWORD n = GetProcessHeaps(256, heaps);
	for (DWORD i = 0; i < n && i < 256; ++i) (void)HeapCompact(heaps[i], 0);
	return n;
}

// Every frame of the last ~2 s within budget (and at least 60 of them): no loading hitch, no busy fight.
bool FramesQuiet(double budget_ms) {
	std::vector<float> frames;
	PerfOverlay::RecentFrames(120, &frames, nullptr);
	if (frames.size() < 60) return false;
	double sum = 0;
	for (float ms: frames) {
		if (ms > budget_ms) return false;
		sum += ms;
	}
	return sum / static_cast<double>(frames.size()) <= budget_ms * 0.85;
}
#endif
} // namespace

bool Enabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_SOFT_RELEASE");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

void Register(const char* name, uint64_t (*release)()) {
	std::lock_guard lock(g_mutex);
	g_sources.emplace_back(name, release);
}

void LowerThreadMemoryPriority() {
#if defined(_WIN32)
	MEMORY_PRIORITY_INFORMATION info {};
	info.MemoryPriority = MEMORY_PRIORITY_LOW;
	(void)SetThreadInformation(GetCurrentThread(), ThreadMemoryPriority, &info, sizeof(info));
#endif
}

uint64_t DiscardPages(void* address, uint64_t size) {
#if defined(_WIN32)
	const auto begin = (reinterpret_cast<uintptr_t>(address) + 4095u) & ~uintptr_t {4095};
	const auto end   = (reinterpret_cast<uintptr_t>(address) + size) & ~uintptr_t {4095};
	if (end <= begin) return 0;
	// (DiscardVirtualMemory: Windows 8.1+; MEM_RESET otherwise. Either way the pages are dropped, never paged out.)
	if (DiscardVirtualMemory(reinterpret_cast<void*>(begin), end - begin) != ERROR_SUCCESS)
		if (VirtualAlloc(reinterpret_cast<void*>(begin), end - begin, MEM_RESET, PAGE_READWRITE) == nullptr) return 0;
	return end - begin;
#else
	(void)address;
	(void)size;
	return 0;
#endif
}

void Start() {
#if defined(_WIN32)
	if (!Enabled()) return;
	static std::once_flag once;
	std::call_once(once, [] {
		const char* s_text  = std::getenv("KYTY_SOFT_RELEASE_SECONDS");
		const char* b_text  = std::getenv("KYTY_SOFT_RELEASE_BUDGET_MS");
		const auto  seconds = std::max<unsigned long>(30, s_text != nullptr ? std::strtoul(s_text, nullptr, 10) : 30);
		const double budget = b_text != nullptr && std::atof(b_text) > 0 ? std::atof(b_text) : 20.0;
		std::printf("Soft release: every %lu s with frames within %.1f ms (no working set trim, no page file)\n", seconds,
		            budget);
		std::fflush(stdout);
		std::thread([seconds, budget] {
			SetThreadDescription(GetCurrentThread(), L"Kyty.SoftRelease");
			SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
			LowerThreadMemoryPriority();
			uint64_t total = 0, passes = 0, skipped = 0;
			for (;;) {
				std::this_thread::sleep_for(std::chrono::seconds(seconds));
				// (Busy: retried every 5 s, up to one more interval, then the next interval.)
				bool quiet = FramesQuiet(budget);
				for (int i = 0; !quiet && i < static_cast<int>(seconds / 5); ++i) {
					std::this_thread::sleep_for(std::chrono::seconds(5));
					quiet = FramesQuiet(budget);
				}
				if (!quiet) {
					if (++skipped % 10 == 1) {
						std::printf("Soft release: skipped (frames over %.1f ms), %llu skips\n", budget,
						            static_cast<unsigned long long>(skipped));
						std::fflush(stdout);
					}
					continue;
				}
				const auto [before, ws_before] = Usage();
				uint64_t       sources = 0;
				{
					std::vector<std::pair<const char*, uint64_t (*)()>> list;
					{
						std::lock_guard lock(g_mutex);
						list = g_sources;
					}
					for (const auto& [name, release]: list) sources += release();
				}
				const size_t   heaps    = CompactHeaps();
				const auto [after, ws_after] = Usage();
				// (Released: the commit dropped, or the resident pages dropped by discards, whichever is more.)
				const uint64_t released = std::max<uint64_t>(before > after ? before - after : 0,
				                                             ws_before > ws_after ? ws_before - ws_after : 0);
				total += released;
				++passes;
				if (passes <= 8 || released >= (16ull << 20u) || passes % 20 == 0) {
					std::printf("Soft release: pass %llu released %llu MiB (heaps %zu, sources discarded %llu MiB), "
					            "%llu MiB in all; private %llu MiB, working set %llu MiB\n",
					            static_cast<unsigned long long>(passes), static_cast<unsigned long long>(released >> 20u),
					            heaps, static_cast<unsigned long long>(sources >> 20u),
					            static_cast<unsigned long long>(total >> 20u), static_cast<unsigned long long>(after >> 20u),
					            static_cast<unsigned long long>(ws_after >> 20u));
					std::fflush(stdout);
				}
			}
		}).detach();
	});
#endif
}

} // namespace SoftRelease
