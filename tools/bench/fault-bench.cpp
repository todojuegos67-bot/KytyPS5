// ds-perf-1102 microbenchmark: the guest-thread cost of a write-window fault cycle (re-protect, write fault,
// VEH handler unprotect) vs a page kept writable (KYTY_HOT_PAGES), at the logged per-frame counts, with busy
// threads in the process (VirtualProtect downgrades send TLB shootdowns to every core running the process).
// Usage: fault-bench <pages per frame> <frames> <busy threads> <pages per window>
#include <windows.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <algorithm>

static char*             g_base = nullptr;
static size_t            g_size = 0;
static size_t            g_window = 4;
static std::atomic<long> g_faults {0};

static LONG CALLBACK Handler(EXCEPTION_POINTERS* info) {
	if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
	auto* addr = reinterpret_cast<char*>(info->ExceptionRecord->ExceptionInformation[1]);
	if (addr < g_base || addr >= g_base + g_size) return EXCEPTION_CONTINUE_SEARCH;
	const size_t page   = (addr - g_base) / 4096;
	const size_t first  = page / g_window * g_window;
	DWORD        old    = 0;
	VirtualProtect(g_base + first * 4096, g_window * 4096, PAGE_READWRITE, &old); // the write window opens
	g_faults.fetch_add(1, std::memory_order_relaxed);
	return EXCEPTION_CONTINUE_EXECUTION;
}

int main(int argc, char** argv) {
	const size_t pages   = argc > 1 ? std::atoi(argv[1]) : 475;
	const int    frames  = argc > 2 ? std::atoi(argv[2]) : 600;
	const int    busy    = argc > 3 ? std::atoi(argv[3]) : 8;
	g_window             = argc > 4 ? std::atoi(argv[4]) : 4;
	const size_t windows = pages; // one fault per window, as the logs count (window_faults)
	g_size               = windows * g_window * 4096 * 2;
	g_base               = static_cast<char*>(VirtualAlloc(nullptr, g_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
	std::vector<char> upload(4096);
	AddVectoredExceptionHandler(1, Handler);
	std::atomic<bool> stop {false};
	std::vector<std::thread> threads;
	for (int t = 0; t < busy; ++t)
		threads.emplace_back([&] { volatile uint64_t x = 0; while (!stop.load(std::memory_order_relaxed)) { x = x + 1; g_base[(x * 64) % g_size] ; } });
	// The page touched in each window: spread like constant buffers (every other window).
	auto run = [&](bool hot) {
		std::vector<double> ms;
		for (int f = 0; f < frames; ++f) {
			const auto start = std::chrono::steady_clock::now();
			for (size_t w = 0; w < windows; ++w) {
				char* p = g_base + (w * 2) * g_window * 4096;
				if (!hot) { DWORD old; VirtualProtect(p, g_window * 4096, PAGE_READONLY, &old); } // re-protect after upload
				std::memset(p, static_cast<int>(f), 256);                                          // the game's write (faults when protected)
				std::memcpy(upload.data(), p, 4096);                                               // the upload copy (both)
			}
			ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
		}
		std::sort(ms.begin(), ms.end());
		double sum = 0; for (double v: ms) sum += v;
		std::printf("%-5s pages/frame %zu window %zu busy %d: mean %.3f ms/frame, p50 %.3f, p99 %.3f (faults %ld)\n", hot ? "hot" : "base",
		            pages, g_window, busy, sum / ms.size(), ms[ms.size() / 2], ms[ms.size() * 99 / 100], g_faults.exchange(0));
		return sum / ms.size();
	};
	double b = 0, h = 0;
	for (int round = 0; round < 3; ++round) { b += run(false); h += run(true); } // A B A B A B
	std::printf("RESULT pages %zu busy %d: base %.3f ms, hot %.3f ms, saved %.3f ms/frame (%.2f us per fault cycle)\n", pages, busy,
	            b / 3, h / 3, (b - h) / 3, (b - h) / 3 * 1000.0 / pages);
	stop = true;
	for (auto& t: threads) t.join();
	return 0;
}
