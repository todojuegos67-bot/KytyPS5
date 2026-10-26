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
		std::this_thread::sleep_for(std::chrono::seconds(5));
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
