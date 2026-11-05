// Diagnostic: KYTY_CPU_LOAD=1 (env, default off). Every 5 s prints the CPU use of the busiest
// threads of the process (user+kernel time over wall time; 100% = one full core) with their names.
// A thread near 100% is a serial bottleneck; if none is, the frame waits on the GPU or on
// synchronization. Off, nothing runs.
#include "common/common.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#include <tlhelp32.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// KYTY_CPU_LOAD=1 also samples where GuestMain is: every ~5 ms its instruction pointer is read
// (SuspendThread/GetThreadContext/ResumeThread, <1% overhead) and the 5 s histogram is printed as
// "GuestMain PC (5 s)": a spin loop in guest code shows as one 64-byte bucket "guest 0x...", a wait
// in the OS as ntdll/kernelbase/win32u, emulator code as kyty_emulator.exe+offset. Together with
// "Phases (5 s)" this tells whether the guest thread is the bottleneck or only waiting for the GPU.
std::atomic<DWORD> g_guest_tid {0};

std::string DescribePc(uint64_t rip) {
	HMODULE mod = nullptr;
	if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       reinterpret_cast<LPCSTR>(rip), &mod) != 0 && mod != nullptr) {
		char path[MAX_PATH] {};
		GetModuleFileNameA(mod, path, MAX_PATH - 1);
		const char* base = path;
		for (const char* c = path; *c != 0; ++c) {
			if (*c == '\\' || *c == '/') base = c + 1;
		}
		char buf[MAX_PATH + 40] {};
		std::snprintf(buf, sizeof(buf), "%s+0x%llx", base, static_cast<unsigned long long>(rip - reinterpret_cast<uint64_t>(mod)));
		return buf;
	}
	char buf[48] {};
	std::snprintf(buf, sizeof(buf), "guest 0x%llx", static_cast<unsigned long long>(rip & ~63ull));
	return buf;
}

void SleepAndSampleGuest(std::chrono::seconds total, std::unordered_map<uint64_t, uint32_t>& hist, uint32_t& samples) {
	const auto end = std::chrono::steady_clock::now() + total;
	HANDLE     h   = nullptr;
	DWORD      tid = 0;
	while (std::chrono::steady_clock::now() < end) {
		const DWORD want = g_guest_tid.load(std::memory_order_relaxed);
		if (want != tid) {
			if (h != nullptr) CloseHandle(h);
			tid = want;
			h   = tid != 0 ? OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, tid) : nullptr;
		}
		if (h != nullptr && SuspendThread(h) != static_cast<DWORD>(-1)) {
			CONTEXT ctx {};
			ctx.ContextFlags = CONTEXT_CONTROL;
			const bool ok    = GetThreadContext(h, &ctx) != 0;
			ResumeThread(h);
			if (ok) {
				// 64-byte buckets for guest code; module code keeps its exact address (resolved at print time).
				++hist[ctx.Rip];
				++samples;
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
	if (h != nullptr) CloseHandle(h);
}

void PrintGuestHistogram(const std::unordered_map<uint64_t, uint32_t>& hist, uint32_t samples) {
	if (samples == 0) return;
	std::unordered_map<std::string, uint32_t> by_where;
	for (const auto& [rip, n]: hist) by_where[DescribePc(rip)] += n;
	std::vector<std::pair<uint32_t, std::string>> rows;
	rows.reserve(by_where.size());
	for (const auto& [where, n]: by_where) rows.emplace_back(n, where);
	std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
	std::printf("GuestMain PC (5 s): %u samples; where it was:", samples);
	for (size_t i = 0; i < rows.size() && i < 8; ++i) {
		std::printf(" [%.0f%% %s]", 100.0 * rows[i].first / samples, rows[i].second.c_str());
	}
	std::printf("\n");
	std::fflush(stdout);
}

void CpuLoadThread() {
	std::unordered_map<DWORD, uint64_t> last;
	auto                                last_time = std::chrono::steady_clock::now();
	bool                                primed    = false;
	const DWORD                         pid       = GetCurrentProcessId();
	for (;;) {
		const auto   now        = std::chrono::steady_clock::now();
		const double wall_100ns = std::chrono::duration<double>(now - last_time).count() * 1e7;
		last_time               = now;
		std::vector<std::pair<double, std::string>> rows;
		std::unordered_map<DWORD, uint64_t>         cur;
		double                                      total = 0.0;
		HANDLE                                      snap  = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
		if (snap != INVALID_HANDLE_VALUE) {
			THREADENTRY32 te {};
			te.dwSize = sizeof(te);
			for (BOOL ok = Thread32First(snap, &te); ok != 0; ok = Thread32Next(snap, &te)) {
				if (te.th32OwnerProcessID != pid) {
					continue;
				}
				HANDLE h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, te.th32ThreadID);
				if (h == nullptr) {
					continue;
				}
				FILETIME c {}, e {}, k {}, u {};
				if (GetThreadTimes(h, &c, &e, &k, &u) != 0) {
					const uint64_t t = ((static_cast<uint64_t>(k.dwHighDateTime) << 32u) | k.dwLowDateTime) +
					                   ((static_cast<uint64_t>(u.dwHighDateTime) << 32u) | u.dwLowDateTime);
					cur[te.th32ThreadID] = t;
					auto it              = last.find(te.th32ThreadID);
					if (primed && it != last.end() && wall_100ns > 0.0) {
						const double pct = 100.0 * static_cast<double>(t - it->second) / wall_100ns;
						total += pct;
						if (pct >= 2.0) {
							std::string name = "tid " + std::to_string(te.th32ThreadID);
							PWSTR       desc = nullptr;
							if (SUCCEEDED(GetThreadDescription(h, &desc)) && desc != nullptr) {
								if (desc[0] != 0) {
									char buf[128] {};
									WideCharToMultiByte(CP_UTF8, 0, desc, -1, buf, sizeof(buf) - 1, nullptr, nullptr);
									name = std::string(buf) + " (" + std::to_string(te.th32ThreadID) + ")";
									if (std::strncmp(buf, "GuestMain", 9) == 0) g_guest_tid.store(te.th32ThreadID, std::memory_order_relaxed);
								}
								LocalFree(desc);
							}
							rows.emplace_back(pct, name);
						}
					}
				}
				CloseHandle(h);
			}
			CloseHandle(snap);
		}
		if (primed) {
			std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
			std::printf("CPU load (5 s): process %.0f%% of one core; busiest threads:\n", total);
			for (size_t i = 0; i < rows.size() && i < 16; i++) {
				std::printf("  %5.1f%%  %s\n", rows[i].first, rows[i].second.c_str());
			}
			std::fflush(stdout);
		}
		last   = std::move(cur);
		primed = true;
		std::unordered_map<uint64_t, uint32_t> hist;
		uint32_t                               samples = 0;
		SleepAndSampleGuest(std::chrono::seconds(5), hist, samples);
		PrintGuestHistogram(hist, samples);
	}
}

struct CpuLoadStarter {
	CpuLoadStarter() {
		const char* v = std::getenv("KYTY_CPU_LOAD");
		if (v != nullptr && v[0] != '\0' && v[0] != '0') {
			std::thread(CpuLoadThread).detach();
		}
	}
};
const CpuLoadStarter g_cpu_load_starter;

// Windows rounds every timed wait (SleepConditionVariable*, WaitForSingleObject, std::condition_variable
// wait_for, Sleep) up to the system timer period, 15.6 ms by default, unless the process asks for a
// finer one. Guest timed waits and Thread_Gpu's poll of a suspended WAIT_REG_MEM ask for 1 ms or less,
// so each one cost ~15.6 ms and the frame serialized on them at a few % CPU. KYTY_TIMER_RES_MS
// (default 1, 0 = leave the system default) sets the period for the life of the process.
struct TimerResolution {
	TimerResolution() {
		const char* v  = std::getenv("KYTY_TIMER_RES_MS");
		const int   ms = v != nullptr ? std::atoi(v) : 1;
		if (ms <= 0) {
			return;
		}
		using func_t = unsigned(WINAPI*)(unsigned);
		HMODULE winmm = LoadLibraryA("winmm.dll");
		auto*   begin = winmm != nullptr ? reinterpret_cast<func_t>(GetProcAddress(winmm, "timeBeginPeriod")) : nullptr;
		const unsigned result = begin != nullptr ? begin(static_cast<unsigned>(ms)) : 1u;
		std::printf("Timer: system timer period %d ms requested (%s)\n", ms, result == 0 ? "ok" : "failed");
		std::fflush(stdout);
	}
};
const TimerResolution g_timer_resolution;

} // namespace

void KytyNameHostThread(const char* name) {
	wchar_t wide[64] {};
	MultiByteToWideChar(CP_UTF8, 0, name, -1, wide, 63);
	SetThreadDescription(GetCurrentThread(), wide);
}

#else
void KytyNameHostThread(const char* /*name*/) {}
#endif
