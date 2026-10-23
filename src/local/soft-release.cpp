#include "soft-release.h"

#include "perf-overlay.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
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
std::mutex                                               g_scratch_mutex;
std::vector<Scratch*>                                    g_scratches;

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

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

Scratch::Scratch(const char* name): source(name) {
	if (!Enabled()) return;
	std::lock_guard lock(g_scratch_mutex);
	g_scratches.push_back(this);
	registered = true;
}

Scratch::~Scratch() {
	if (!registered) return;
	std::lock_guard lock(g_scratch_mutex);
	g_scratches.erase(std::remove(g_scratches.begin(), g_scratches.end(), this), g_scratches.end());
}

std::unique_lock<std::mutex> Scratch::Use() {
	if (!registered) return {};
	last_use_ns.store(NowNs(), std::memory_order_relaxed);
	return std::unique_lock<std::mutex>(mutex);
}

uint64_t Scratch::ReleaseIfIdle(int64_t now_ns) {
	if (now_ns - last_use_ns.load(std::memory_order_relaxed) < 10'000'000'000ll) return 0;
	std::unique_lock lock(mutex, std::try_to_lock);
	if (!lock.owns_lock()) return 0;
	const uint64_t capacity = bytes.capacity();
	if (capacity < (64u << 10u)) return 0;
	std::vector<uint8_t>().swap(bytes);
	return capacity;
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
			std::map<std::string, uint64_t> scratch_total;
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
				// Idle scratch buffers, by source.
				std::string by_source;
				{
					std::vector<std::pair<const char*, uint64_t>> freed;
					const auto now = NowNs();
					std::lock_guard lock(g_scratch_mutex);
					for (auto* scratch: g_scratches) {
						const uint64_t bytes = scratch->ReleaseIfIdle(now);
						if (bytes == 0) continue;
						auto it = std::find_if(freed.begin(), freed.end(),
						                       [&](const auto& e) { return std::strcmp(e.first, scratch->source) == 0; });
						if (it == freed.end()) freed.emplace_back(scratch->source, bytes);
						else it->second += bytes;
					}
					for (const auto& [name, bytes]: freed) {
						char piece[160];
						std::snprintf(piece, sizeof(piece), "%s%s %.1f MiB", by_source.empty() ? "" : ", ", name,
						              static_cast<double>(bytes) / 1048576.0);
						by_source += piece;
						scratch_total[name] += bytes;
					}
				}
				const size_t   heaps    = CompactHeaps();
				const auto [after, ws_after] = Usage();
				// (Released: the commit dropped, or the resident pages dropped by discards, whichever is more.)
				const uint64_t released = std::max<uint64_t>(before > after ? before - after : 0,
				                                             ws_before > ws_after ? ws_before - ws_after : 0);
				total += released;
				++passes;
				if (!by_source.empty()) {
					std::string totals;
					for (const auto& [name, bytes]: scratch_total) {
						char piece[160];
						std::snprintf(piece, sizeof(piece), "%s%s %llu MiB", totals.empty() ? "" : ", ", name.c_str(),
						              static_cast<unsigned long long>(bytes >> 20u));
						totals += piece;
					}
					std::printf("Soft release: idle scratch freed: %s (in all: %s)\n", by_source.c_str(), totals.c_str());
				}
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
