#include "local/mem-guard.h"

#include "common/assert.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h> // _heapmin
#include <psapi.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace MemGuard {

namespace {

struct Releaser {
	int         id;
	const char* name;
	ReleaseFn   fn;
};

std::mutex& Mutex() {
	static auto* m = new std::mutex(); // (never destroyed: a release may run while the process ends)
	return *m;
}
std::condition_variable& Cv() {
	static auto* cv = new std::condition_variable();
	return *cv;
}

std::vector<Releaser>* g_releasers = nullptr; // under Mutex()
int                    g_next_id   = 1;
uint64_t               g_requested = 0;       // pass requests (under Mutex())
uint64_t               g_done      = 0;       // passes done, the last request each one served
uint64_t               g_last_freed = 0;
const char*            g_reason     = "";
bool                   g_thread_up  = false;
std::atomic<uint64_t>  g_epoch {0};
std::atomic<int64_t>   g_last_pressure_ms {0};
std::atomic<uint32_t>  g_passes_logged {0};
std::atomic<bool>      g_box_shown {false};
std::atomic<bool>      g_in_fatal {false};
thread_local bool      t_guard_thread = false;

int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

struct State {
	uint64_t process_commit = 0, working_set = 0, ram_total = 0, ram_avail = 0, commit_total = 0, commit_limit = 0;
};

State Query() {
	State s;
#if defined(_WIN32)
	PROCESS_MEMORY_COUNTERS_EX counters {};
	counters.cb = sizeof(counters);
	if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
	                            sizeof(counters)) != 0) {
		s.process_commit = counters.PrivateUsage;
		s.working_set    = counters.WorkingSetSize;
	}
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	if (GlobalMemoryStatusEx(&status) != 0) {
		s.ram_total = status.ullTotalPhys;
		s.ram_avail = status.ullAvailPhys;
	}
	PERFORMANCE_INFORMATION perf {};
	perf.cb = sizeof(perf);
	if (K32GetPerformanceInfo(&perf, sizeof(perf)) != 0) {
		s.commit_total = static_cast<uint64_t>(perf.CommitTotal) * perf.PageSize;
		s.commit_limit = static_cast<uint64_t>(perf.CommitLimit) * perf.PageSize;
	}
#elif defined(__linux__)
	struct sysinfo info {};
	if (sysinfo(&info) == 0) {
		s.ram_total    = static_cast<uint64_t>(info.totalram) * info.mem_unit;
		s.ram_avail    = static_cast<uint64_t>(info.freeram + info.bufferram) * info.mem_unit;
		s.commit_limit = static_cast<uint64_t>(info.totalram + info.totalswap) * info.mem_unit;
		s.commit_total = s.commit_limit - static_cast<uint64_t>(info.freeram + info.freeswap) * info.mem_unit;
	}
	if (FILE* f = std::fopen("/proc/self/statm", "r"); f != nullptr) {
		unsigned long long size = 0, resident = 0;
		if (std::fscanf(f, "%llu %llu", &size, &resident) == 2) {
			const auto page  = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
			s.process_commit = size * page;
			s.working_set    = resident * page;
		}
		std::fclose(f);
	}
#endif
	return s;
}

uint64_t CompactHeaps() {
#if defined(_WIN32)
	_heapmin();
	HANDLE      heaps[256];
	const DWORD n = GetProcessHeaps(256, heaps);
	for (DWORD i = 0; i < n && i < 256; ++i) (void)HeapCompact(heaps[i], 0);
#endif
	return 0; // (measured by the pass as the commit drop)
}

void RunPass(const char* reason) {
	const auto start  = NowMs();
	const auto before = Query();
	std::vector<Releaser> list;
	{
		std::lock_guard lock(Mutex());
		if (g_releasers != nullptr) list = *g_releasers;
	}
	uint64_t reported = 0;
	for (const auto& r: list) {
		try {
			reported += r.fn();
		} catch (...) {
			// (A releaser that itself ran out: the others still run.)
		}
	}
	CompactHeaps();
	const auto after = Query();
	const uint64_t dropped = before.process_commit > after.process_commit ? before.process_commit - after.process_commit : 0;
	const uint64_t freed   = std::max(reported, dropped);
	g_epoch.fetch_add(1, std::memory_order_acq_rel);
	if (g_passes_logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		char state[256];
		FormatState(state, sizeof(state));
		std::printf("Memory guard: release pass (%s): %zu releasers, %llu MiB freed (%llu MiB reported) in %lld ms; %s\n",
		            reason, list.size(), static_cast<unsigned long long>(dropped >> 20u),
		            static_cast<unsigned long long>(reported >> 20u), static_cast<long long>(NowMs() - start), state);
		std::fflush(stdout);
	}
	std::lock_guard lock(Mutex());
	g_last_freed = freed;
}

void ThreadMain() {
	t_guard_thread = true;
#if defined(_WIN32)
	SetThreadDescription(GetCurrentThread(), L"Kyty.MemGuard");
#endif
	std::unique_lock lock(Mutex());
	for (;;) {
		Cv().wait(lock, [] { return g_requested > g_done; });
		const uint64_t serving = g_requested;
		const char*    reason  = g_reason;
		lock.unlock();
		RunPass(reason);
		lock.lock();
		g_done = serving;
		Cv().notify_all();
	}
}

// Under Mutex(). False when no thread could be made (then the caller compacts the heaps inline).
bool EnsureThreadLocked() {
	if (g_thread_up) return true;
	try {
		std::thread(ThreadMain).detach();
		g_thread_up = true;
	} catch (...) {
		return false;
	}
	return true;
}

// operator new's handler (local-platform.cpp's KytyNew calls it in its loop): a release pass and a wait per call; after
// MaxRetries calls within a few seconds on one thread, the clear message and exit (instead of std::bad_alloc ending
// the process with no word).
void NewHandler() {
	if (g_in_fatal.load()) {
		// (The fatal report itself ran out: operator new throws std::bad_alloc now.)
		std::set_new_handler(nullptr);
		return;
	}
	// (A releaser's own allocation: std::bad_alloc, which the pass catches; no pass inside a pass.)
	if (t_guard_thread) throw std::bad_alloc();
	thread_local int     calls      = 0;
	thread_local int64_t first_call = 0;
	const auto           now        = NowMs();
	if (calls == 0 || now - first_call > 10000) {
		calls      = 0;
		first_call = now;
	}
	if (!ShouldRetry("operator new", 0, calls++)) FatalOutOfMemory("operator new (C++ heap)", 0);
}

#if defined(_WIN32)
using MessageBoxW_t = int(WINAPI*)(HWND, LPCWSTR, LPCWSTR, UINT);
struct BoxArgs {
	MessageBoxW_t  box  = nullptr;
	const wchar_t* text = nullptr;
};
BoxArgs g_box;
DWORD WINAPI BoxThread(LPVOID p) {
	const auto* a = static_cast<BoxArgs*>(p);
	a->box(nullptr, a->text, L"KytyPS5 - memoria / memory",
	       MB_OK | MB_ICONERROR | MB_TOPMOST | MB_SETFOREGROUND | MB_SYSTEMMODAL);
	return 0;
}

void ShowBox(const wchar_t* text) {
	if (const char* v = std::getenv("KYTY_NO_MESSAGEBOX"); v != nullptr && v[0] == '1') return;
	if (g_box_shown.exchange(true)) return;
	HMODULE user32      = LoadLibraryW(L"user32.dll");
	auto    box         = user32 != nullptr ? reinterpret_cast<MessageBoxW_t>(GetProcAddress(user32, "MessageBoxW")) : nullptr;
	if (box == nullptr) return;
	// On a thread of its own, at most 5 minutes: the exit is never held for good.
	g_box.box  = box;
	g_box.text = text;
	HANDLE thread = CreateThread(nullptr, 64 * 1024, BoxThread, &g_box, 0, nullptr);
	if (thread != nullptr) {
		WaitForSingleObject(thread, 5 * 60 * 1000);
		CloseHandle(thread);
	}
}

constexpr const wchar_t* OutOfMemoryText =
    L"No hay suficiente memoria RAM: el emulador se ha quedado sin memoria.\n"
    L"Cierra otros programas (navegadores, Discord, launchers) y vuelve a jugar.\n"
    L"En el launcher: marca \"Equipo de 16 GB RAM\" (con 16 GB se marca solo).\n\n"
    L"Not enough RAM: the emulator ran out of memory.\n"
    L"Close other programs (browsers, Discord, launchers) and start the game again.\n"
    L"In the launcher: tick \"16 GB RAM PC\" (on 16 GB PCs it is ticked for you).";
constexpr const wchar_t* NearlyOutText =
    L"El emulador se ha cerrado con la memoria RAM casi llena: lo mas probable es que falte memoria.\n"
    L"Cierra otros programas (navegadores, Discord, launchers) y vuelve a jugar.\n\n"
    L"The emulator stopped while RAM was nearly full: most likely it ran out of memory.\n"
    L"Close other programs (browsers, Discord, launchers) and start the game again.";
#endif

} // namespace

int RegisterReleaser(const char* name, ReleaseFn fn) {
	std::lock_guard lock(Mutex());
	if (g_releasers == nullptr) g_releasers = new std::vector<Releaser>();
	const int id = g_next_id++;
	g_releasers->push_back({id, name, std::move(fn)});
	return id;
}

void UnregisterReleaser(int id) {
	// (Waits for a pass in progress: the releaser's object may go right after.)
	std::unique_lock lock(Mutex());
	if (!t_guard_thread) Cv().wait_for(lock, std::chrono::seconds(2), [] { return g_requested == g_done; });
	if (g_releasers == nullptr) return;
	g_releasers->erase(std::remove_if(g_releasers->begin(), g_releasers->end(), [&](const Releaser& r) { return r.id == id; }),
	                   g_releasers->end());
}

uint64_t ReleaseNow(const char* reason, uint32_t wait_ms) {
	if (t_guard_thread) return 0; // (a releaser ran out itself: no pass inside a pass)
	std::unique_lock lock(Mutex());
	if (!EnsureThreadLocked()) {
		lock.unlock();
		CompactHeaps();
		return 0;
	}
	const uint64_t ticket = ++g_requested;
	g_reason              = reason != nullptr ? reason : "";
	Cv().notify_all();
	if (wait_ms == 0) return 0;
	if (!Cv().wait_for(lock, std::chrono::milliseconds(wait_ms), [&] { return g_done >= ticket; })) return 0;
	return g_last_freed;
}

void OnPressure(const char* reason) {
	const auto now  = NowMs();
	auto       last = g_last_pressure_ms.load(std::memory_order_relaxed);
	if (now - last < 2000 || !g_last_pressure_ms.compare_exchange_strong(last, now)) return;
	(void)ReleaseNow(reason, 0);
}

uint64_t ReleaseEpoch() {
	return g_epoch.load(std::memory_order_acquire);
}

void Start() {
	{
		std::lock_guard lock(Mutex());
		(void)EnsureThreadLocked();
	}
	static std::once_flag once;
	std::call_once(once, [] {
		if (std::get_new_handler() == nullptr) std::set_new_handler(NewHandler);
	});
}

bool IsOutOfMemoryError(uint32_t error) {
#if defined(_WIN32)
	switch (error) {
		case ERROR_NOT_ENOUGH_MEMORY:   // 8
		case ERROR_OUTOFMEMORY:         // 14
		case ERROR_COMMITMENT_MINIMUM:  // 635
		case ERROR_NO_SYSTEM_RESOURCES: // 1450
		case ERROR_NONPAGED_SYSTEM_RESOURCES:
		case ERROR_PAGED_SYSTEM_RESOURCES:
		case ERROR_WORKING_SET_QUOTA:
		case ERROR_PAGEFILE_QUOTA:      // 1454
		case ERROR_COMMITMENT_LIMIT:    // 1455
			return true;
		default: return false;
	}
#else
	return error == 12; // ENOMEM
#endif
}

uint32_t LastOsError() {
#if defined(_WIN32)
	return static_cast<uint32_t>(GetLastError());
#else
	return static_cast<uint32_t>(errno);
#endif
}

void SetLastOsError(uint32_t error) {
#if defined(_WIN32)
	SetLastError(static_cast<DWORD>(error));
#else
	errno = static_cast<int>(error);
#endif
}

bool ShouldRetry(const char* what, uint64_t bytes, int attempt) {
	if (attempt >= MaxRetries) return false;
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
		char state[256];
		FormatState(state, sizeof(state));
		std::printf("Memory guard: %s of %llu MiB failed for lack of memory (try %d of %d): releasing; %s\n",
		            what != nullptr ? what : "an allocation", static_cast<unsigned long long>(bytes >> 20u), attempt + 1,
		            MaxRetries, state);
		std::fflush(stdout);
	}
	if (t_guard_thread) {
		// (A releaser's own allocation: no pass to wait for, a short pause only.)
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		return attempt == 0;
	}
	(void)ReleaseNow(what, 300);
	// Other programs and the system's own trimming get a moment too: 50, 100, 200, 400 ms.
	std::this_thread::sleep_for(std::chrono::milliseconds(50u << static_cast<unsigned>(attempt)));
	return true;
}

void FatalOutOfMemory(const char* what, uint64_t bytes) {
	g_in_fatal.store(true);
	char state[256];
	FormatState(state, sizeof(state));
	std::fprintf(stderr, "Memory guard: OUT OF MEMORY: %s (%llu MiB) still failed after %d release passes; %s\n",
	             what != nullptr ? what : "an allocation", static_cast<unsigned long long>(bytes >> 20u), MaxRetries,
	             state);
	std::printf("Memory guard: OUT OF MEMORY: %s (%llu MiB) still failed after %d release passes; %s\n"
	            "Not enough RAM: close other programs (browsers, Discord) and start the game again "
	            "(launcher: the 16 GB RAM PC profile).\n",
	            what != nullptr ? what : "an allocation", static_cast<unsigned long long>(bytes >> 20u), MaxRetries,
	            state);
	std::fflush(nullptr);
#if defined(_WIN32)
	ShowBox(OutOfMemoryText);
#endif
	EXIT("Not enough RAM: %s (%llu MiB) could not be allocated. Close other programs and start the game again.\n",
	     what != nullptr ? what : "an allocation", static_cast<unsigned long long>(bytes >> 20u));
	std::_Exit(3); // (EXIT does not return; this keeps [[noreturn]] honest under a recoverable scope's throw)
}

void FormatState(char* out, size_t size) {
	if (out == nullptr || size == 0) return;
	const auto s = Query();
	std::snprintf(out, size,
	              "process commit %llu MiB, working set %llu MiB; system RAM available %llu of %llu MiB; "
	              "commit %llu of %llu MiB",
	              static_cast<unsigned long long>(s.process_commit >> 20u),
	              static_cast<unsigned long long>(s.working_set >> 20u), static_cast<unsigned long long>(s.ram_avail >> 20u),
	              static_cast<unsigned long long>(s.ram_total >> 20u),
	              static_cast<unsigned long long>(s.commit_total >> 20u),
	              static_cast<unsigned long long>(s.commit_limit >> 20u));
}

void LogState(const char* label) {
	char state[256];
	FormatState(state, sizeof(state));
	std::printf("Memory state (%s): %s\n", label != nullptr ? label : "", state);
	std::fflush(stdout);
}

bool SystemNearlyOut() {
	const auto     s         = Query();
	constexpr auto MiB       = uint64_t {1} << 20u;
	const bool     commit_ok = s.commit_limit == 0 || s.commit_total + std::max<uint64_t>(512 * MiB, s.commit_limit / 33) <
	                                                  s.commit_limit;
	const bool     ram_ok    = s.ram_total == 0 || s.ram_avail >= 256 * MiB;
	return !(commit_ok && ram_ok);
}

void NoteFatalExit() {
#if defined(_WIN32)
	if (g_box_shown.load()) return;
	if (!SystemNearlyOut()) return;
	std::printf("Memory guard: the process is ending with the system nearly out of memory (likely the cause)\n");
	std::fflush(stdout);
	ShowBox(NearlyOutText);
#endif
}

} // namespace MemGuard
