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

// (HeapSetInformation's HeapOptimizeResources, Windows 8.1+: its own copy of the SDK's struct and class value.)
struct OptimizeResources {
	DWORD version;
	DWORD flags;
};
constexpr int kHeapOptimizeResources = 3;

// A heap's free commit decommitted (the NT heap's free segment pages and the Segment Heap's alike).
void OptimizeHeap(HANDLE heap) {
	OptimizeResources info {1, 0};
	(void)HeapSetInformation(heap, static_cast<HEAP_INFORMATION_CLASS>(kHeapOptimizeResources), &info, sizeof(info));
}

// The heaps' free blocks decommitted (no page file: a decommitted page has nothing to write). Returns the heaps' count.
size_t CompactHeaps() {
	_heapmin();
	HANDLE heaps[256];
	const DWORD n = GetProcessHeaps(256, heaps);
	for (DWORD i = 0; i < n && i < 256; ++i) {
		(void)HeapCompact(heaps[i], 0);
		OptimizeHeap(heaps[i]);
	}
	return n;
}

// kernel32's HeapSummary (Windows 10+, looked up: older SDKs lack it): a heap's allocated and committed bytes.
struct HeapSummaryInfo {
	DWORD  cb;
	SIZE_T allocated;
	SIZE_T committed;
	SIZE_T reserved;
	SIZE_T max_reserve;
};
using HeapSummaryFn = BOOL(WINAPI*)(HANDLE, DWORD, HeapSummaryInfo*);

HeapSummaryFn HeapSummaryFunction() {
	static const HeapSummaryFn fn = [] {
		HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
		return kernel != nullptr
		           ? reinterpret_cast<HeapSummaryFn>(reinterpret_cast<void*>(GetProcAddress(kernel, "HeapSummary")))
		           : nullptr;
	}();
	return fn;
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

uint64_t Scratch::ReleaseIfIdle(int64_t now_ns, int64_t idle_ns) {
	if (now_ns - last_use_ns.load(std::memory_order_relaxed) < idle_ns) return 0;
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

uint64_t FlushNow(int64_t scratch_idle_ms, char* detail, size_t detail_size) {
#if defined(_WIN32)
	// (One pass at a time: the soft-release thread's own passes and the RAM guard's.)
	static std::mutex flushing;
	const std::lock_guard serial(flushing);
	const auto [before, ws_before] = Usage();
	uint64_t sources = 0;
	{
		std::vector<std::pair<const char*, uint64_t (*)()>> list;
		{
			std::lock_guard lock(g_mutex);
			list = g_sources;
		}
		for (const auto& [name, release]: list) sources += release();
	}
	uint64_t scratch = 0;
	{
		const auto now = NowNs();
		std::lock_guard lock(g_scratch_mutex);
		for (auto* s: g_scratches) scratch += s->ReleaseIfIdle(now, std::max<int64_t>(0, scratch_idle_ms) * 1'000'000ll);
	}
	const size_t heaps = CompactHeaps();
	const auto [after, ws_after] = Usage();
	const uint64_t released = std::max<uint64_t>(before > after ? before - after : 0, ws_before > ws_after ? ws_before - ws_after : 0);
	if (detail != nullptr && detail_size != 0)
		std::snprintf(detail, detail_size, "heaps %zu compacted, scratch %.1f MiB, free pools %.1f MiB", heaps,
		              static_cast<double>(scratch) / 1048576.0, static_cast<double>(sources) / 1048576.0);
	return released;
#else
	(void)scratch_idle_ms;
	if (detail != nullptr && detail_size != 0) detail[0] = '\0';
	return 0;
#endif
}

void StartHeapReport() {
#if defined(_WIN32)
	// KYTY_HEAP_REPORT=0: off.
	if (const char* v = std::getenv("KYTY_HEAP_REPORT"); v != nullptr && v[0] == '0') return;
	static std::once_flag once;
	std::call_once(once, [] {
		std::thread([] {
			SetThreadDescription(GetCurrentThread(), L"Kyty.HeapReport");
			SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
			LowerThreadMemoryPriority();
			const HeapSummaryFn summary = HeapSummaryFunction();
#if defined(KYTY_SEGMENT_HEAP)
			const char* kind = "Segment Heap requested by the manifest";
#else
			const char* kind = "NT heap";
#endif
			uint64_t optimized_total = 0;
			for (;;) {
				std::this_thread::sleep_for(std::chrono::seconds(30));
				HANDLE      heaps[256];
				const DWORD n         = GetProcessHeaps(256, heaps);
				uint64_t    committed = 0, allocated = 0, reserved = 0;
				uint64_t    optimized = 0;
				bool        known     = summary != nullptr;
				for (DWORD i = 0; i < n && i < 256; ++i) {
					if (summary == nullptr) break;
					HeapSummaryInfo info {};
					info.cb = sizeof(info);
					if (summary(heaps[i], 0, &info) == 0) {
						known = false;
						continue;
					}
					committed += info.committed;
					allocated += info.allocated;
					reserved += info.reserved;
					// (A heap holding 256 MiB or more committed but free: that free commit decommitted now, not only on
					// a quiet soft-release pass.)
					const uint64_t free_bytes = info.committed > info.allocated ? info.committed - info.allocated : 0;
					if (free_bytes >= (256ull << 20u)) {
						OptimizeHeap(heaps[i]);
						HeapSummaryInfo after {};
						after.cb = sizeof(after);
						if (summary(heaps[i], 0, &after) != 0 && after.committed < info.committed)
							optimized += info.committed - after.committed;
					}
				}
				optimized_total += optimized;
				const auto [priv, ws] = Usage();
				if (known) {
					std::printf("Heap (30 s, %s): %lu heaps, committed %llu MiB, allocated %llu MiB, free %llu MiB, reserved "
					            "%llu MiB; decommitted now %llu MiB (%llu MiB in all); emulator private %llu MiB, working set "
					            "%llu MiB\n",
					            kind, static_cast<unsigned long>(n), static_cast<unsigned long long>(committed >> 20u),
					            static_cast<unsigned long long>(allocated >> 20u),
					            static_cast<unsigned long long>((committed > allocated ? committed - allocated : 0) >> 20u),
					            static_cast<unsigned long long>(reserved >> 20u),
					            static_cast<unsigned long long>(optimized >> 20u),
					            static_cast<unsigned long long>(optimized_total >> 20u),
					            static_cast<unsigned long long>(priv >> 20u), static_cast<unsigned long long>(ws >> 20u));
				} else {
					std::printf("Heap (30 s, %s): %lu heaps, HeapSummary unavailable; emulator private %llu MiB, working set "
					            "%llu MiB\n",
					            kind, static_cast<unsigned long>(n), static_cast<unsigned long long>(priv >> 20u),
					            static_cast<unsigned long long>(ws >> 20u));
				}
				std::fflush(stdout);
			}
		}).detach();
	});
#endif
}

void Start() {
#if defined(_WIN32)
	StartHeapReport();
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
