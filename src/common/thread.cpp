// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <ctime>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
#include <fmt/format.h>

#include "core/libraries/fiber/fiber.h"
#include "core/libraries/kernel/threads/pthread.h"

#include "common/error.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "ntapi.h"
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <pthread.h>
#elif defined(_WIN32)
#include <windows.h>
#include "common/string_util.h"
#else
#if defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
#include <pthread_np.h>
#else
#include <pthread.h>
#endif
#include <sched.h>
#endif
#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __FreeBSD__
#define cpu_set_t cpuset_t
#endif

namespace Common {

#ifdef __APPLE__

void SetCurrentThreadRealtime(const std::chrono::nanoseconds period_ns) {
    // CPU time to grant.
    const std::chrono::nanoseconds computation_ns = period_ns / 2;

    // Determine the timebase for converting time to ticks.
    struct mach_timebase_info timebase{};
    mach_timebase_info(&timebase);
    const auto ticks_per_ns =
        static_cast<double>(timebase.denom) / static_cast<double>(timebase.numer);

    const auto period_ticks =
        static_cast<u32>(static_cast<double>(period_ns.count()) * ticks_per_ns);
    const auto computation_ticks =
        static_cast<u32>(static_cast<double>(computation_ns.count()) * ticks_per_ns);

    thread_time_constraint_policy policy = {
        .period = period_ticks,
        .computation = computation_ticks,
        // Should not matter since preemptible is false, but needs to be >= computation regardless.
        .constraint = computation_ticks,
        .preemptible = false,
    };

    int ret = thread_policy_set(
        pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
        reinterpret_cast<thread_policy_t>(&policy), THREAD_TIME_CONSTRAINT_POLICY_COUNT);
    if (ret != KERN_SUCCESS) {
        LOG_ERROR(Common, "Could not set thread to real-time with period {} ns: {}",
                  period_ns.count(), ret);
    }
}

#else

void SetCurrentThreadRealtime(const std::chrono::nanoseconds period_ns) {
    // Not implemented
}

#endif

#ifdef _WIN32

void SetCurrentThreadPriority(ThreadPriority new_priority) {
    auto handle = GetCurrentThread();
    int windows_priority = 0;
    switch (new_priority) {
    case ThreadPriority::Low:
        windows_priority = THREAD_PRIORITY_BELOW_NORMAL;
        break;
    case ThreadPriority::Normal:
        windows_priority = THREAD_PRIORITY_NORMAL;
        break;
    case ThreadPriority::High:
        windows_priority = THREAD_PRIORITY_ABOVE_NORMAL;
        break;
    case ThreadPriority::VeryHigh:
        windows_priority = THREAD_PRIORITY_HIGHEST;
        break;
    case ThreadPriority::Critical:
        windows_priority = THREAD_PRIORITY_TIME_CRITICAL;
        break;
    default:
        windows_priority = THREAD_PRIORITY_NORMAL;
        break;
    }
    SetThreadPriority(handle, windows_priority);
}

bool AccurateSleep(const std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   const bool interruptible) {
    const auto begin_sleep = std::chrono::high_resolution_clock::now();

    LARGE_INTEGER interval{
        .QuadPart = -1 * (duration.count() / 100u),
    };
    HANDLE timer = ::CreateWaitableTimer(NULL, TRUE, NULL);
    SetWaitableTimer(timer, &interval, 0, NULL, NULL, 0);
    const auto ret = WaitForSingleObjectEx(timer, INFINITE, interruptible);
    ::CloseHandle(timer);

    if (remaining) {
        const auto end_sleep = std::chrono::high_resolution_clock::now();
        const auto sleep_time = end_sleep - begin_sleep;
        *remaining = duration > sleep_time ? duration - sleep_time : std::chrono::nanoseconds(0);
    }
    return ret == WAIT_OBJECT_0;
}

#else

void SetCurrentThreadPriority(ThreadPriority new_priority) {
    pthread_t this_thread = pthread_self();

    const auto scheduling_type = SCHED_OTHER;
    s32 max_prio = sched_get_priority_max(scheduling_type);
    s32 min_prio = sched_get_priority_min(scheduling_type);
    u32 level = std::max(static_cast<u32>(new_priority) + 1, 4U);

    struct sched_param params;
    if (max_prio > min_prio) {
        params.sched_priority = min_prio + ((max_prio - min_prio) * level) / 4;
    } else {
        params.sched_priority = min_prio - ((min_prio - max_prio) * level) / 4;
    }

    pthread_setschedparam(this_thread, scheduling_type, &params);
}

bool AccurateSleep(const std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   const bool interruptible) {
    timespec request = {
        .tv_sec = duration.count() / 1'000'000'000,
        .tv_nsec = duration.count() % 1'000'000'000,
    };
    timespec remain;
    int ret;
    while ((ret = nanosleep(&request, &remain)) < 0 && errno == EINTR) {
        if (interruptible) {
            break;
        }
        request = remain;
    }
    if (remaining) {
        *remaining = std::chrono::nanoseconds(remain.tv_sec * 1'000'000'000 + remain.tv_nsec);
    }
    return ret == 0 || errno != EINTR;
}

#endif

#ifdef _WIN32

namespace {
struct BBPerfThread {
    std::string name;
    HANDLE handle = nullptr;
    u64 last_100ns = 0;
};
std::mutex g_bb_perf_mutex;
std::vector<BBPerfThread> g_bb_perf_threads;

bool BBEnvOn(const char* name, bool def) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return def;
    return !(v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F');
}

bool BBHotThread(const std::string& n) {
    return n.find("GpuCommandProcessor") != std::string::npos ||
           n.find("PresentThread") != std::string::npos ||
           n.find("GpuSchedPriorityPendingOpsRunner") != std::string::npos ||
           n.find("shadPS4:Main") != std::string::npos;
}

// The processors that share the largest L3 (on a 9950X3D: the V-Cache CCD). 0: unknown / uniform.
KAFFINITY BBBigCacheMask(WORD& group) {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);
    if (len == 0) return 0;
    std::vector<char> buf(len);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationCache, info, &len)) return 0;
    DWORD best = 0, smallest = 0xFFFFFFFFu;
    int caches = 0;
    KAFFINITY mask = 0;
    for (DWORD off = 0; off < len;) {
        auto* e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        if (e->Relationship == RelationCache && e->Cache.Level == 3) {
            ++caches;
            smallest = std::min(smallest, e->Cache.CacheSize);
            if (e->Cache.CacheSize > best) {
                best = e->Cache.CacheSize;
                mask = e->Cache.GroupMask.Mask;
                group = e->Cache.GroupMask.Group;
            }
        }
        off += e->Size;
    }
    // Only a real asymmetric part (one L3 bigger than another) gets a preference.
    return caches >= 2 && best > smallest ? mask : 0;
}

void BBTuneThread(const std::string& n) {
    if (!BBHotThread(n)) return;
    // The GPU command processor feeds the host GPU: above normal (BB_THREAD_PRIO=0: as before).
    if (n.find("GpuCommandProcessor") != std::string::npos && BBEnvOn("BB_THREAD_PRIO", false)) {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    }
    // BB_AFFINITY=1 (default off): the hot threads prefer the big-L3 CCD (ideal processor plus a
    // group affinity on that CCD's processors). Never set on a symmetric CPU.
    if (BBEnvOn("BB_AFFINITY", false)) {
        WORD group = 0;
        const KAFFINITY mask = BBBigCacheMask(group);
        if (mask != 0) {
            GROUP_AFFINITY ga{};
            ga.Mask = mask;
            ga.Group = group;
            SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr);
            LOG_INFO(Common, "[perf] BB_AFFINITY: {} on the big-L3 processors (mask {:#x})", n,
                     static_cast<u64>(mask));
        }
    }
}

void BBRegisterThread(const char* name) {
    HANDLE h = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &h,
                         THREAD_QUERY_LIMITED_INFORMATION | THREAD_SET_INFORMATION, FALSE, 0)) {
        return;
    }
    std::scoped_lock lk{g_bb_perf_mutex};
    if (g_bb_perf_threads.size() >= 512) {
        CloseHandle(h);
        return;
    }
    g_bb_perf_threads.push_back({name, h, 0});
}
} // namespace

std::string BBThreadBusyReport(double wall_seconds, int top) {
    if (wall_seconds <= 0.0) return {};
    std::vector<std::pair<double, std::string>> busy;
    {
        std::scoped_lock lk{g_bb_perf_mutex};
        for (auto it = g_bb_perf_threads.begin(); it != g_bb_perf_threads.end();) {
            FILETIME c, e, k, u;
            DWORD code = 0;
            if (!GetThreadTimes(it->handle, &c, &e, &k, &u) ||
                (GetExitCodeThread(it->handle, &code) && code != STILL_ACTIVE)) {
                CloseHandle(it->handle);
                it = g_bb_perf_threads.erase(it);
                continue;
            }
            const auto t = [](FILETIME f) {
                return (static_cast<u64>(f.dwHighDateTime) << 32) | f.dwLowDateTime;
            };
            const u64 now = t(k) + t(u);
            if (it->last_100ns != 0) {
                busy.emplace_back(static_cast<double>(now - it->last_100ns) / 1e7 / wall_seconds *
                                      100.0,
                                  it->name);
            }
            it->last_100ns = now;
            ++it;
        }
    }
    std::sort(busy.begin(), busy.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string out;
    for (int i = 0; i < top && i < static_cast<int>(busy.size()); ++i) {
        std::string n = busy[i].second;
        if (n.rfind("shadPS4:", 0) == 0) n = n.substr(8);
        out += fmt::format("{}{} {:.0f}%", i ? ", " : "", n, busy[i].first);
    }
    return out;
}

// Sets the debugger-visible name of the current thread.
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
    SetThreadDescription(GetCurrentThread(), UTF8ToUTF16W(name).data());
    // bb-crash-119: only emulator threads are registered / tuned by default; guest threads (e.g.
    // HavokWorkerThread) are left untouched unless BB_PERF=2.
    const std::string bb_n{name ? name : ""};
    const char* bb_perf = std::getenv("BB_PERF");
    const bool bb_emu = bb_n.rfind("shadPS4:", 0) == 0 || BBHotThread(bb_n);
    if (bb_emu || (bb_perf && bb_perf[0] == '2')) {
        BBRegisterThread(name);
    }
    if (bb_emu) {
        BBTuneThread(bb_n);
    }
}

void SetThreadName(void* thread, const char* name) {
    SetThreadDescription(thread, UTF8ToUTF16W(name).data());
}

#else // !_WIN32, so must be POSIX threads

// MinGW with the POSIX threading model does not support pthread_setname_np
#if !defined(_WIN32) || defined(_MSC_VER)
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
#ifdef __APPLE__
    pthread_setname_np(name);
#elif defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    pthread_set_name_np(pthread_self(), name);
#elif defined(__NetBSD__)
    pthread_setname_np(pthread_self(), "%s", (void*)name);
#elif defined(__linux__)
    // Linux limits thread names to 15 characters and will outright reject any
    // attempt to set a longer name with ERANGE.
    std::string truncated(name, std::min(strlen(name), static_cast<std::size_t>(15)));
    if (int e = pthread_setname_np(pthread_self(), truncated.c_str())) {
        errno = e;
        LOG_ERROR(Common, "Failed to set thread name to '{}': {}", truncated, GetLastErrorMsg());
    }
#else
    pthread_setname_np(pthread_self(), name);
#endif
}

void SetThreadName(void* thread, const char* name) {
    // TODO
}
#endif

#if defined(_WIN32)
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
    // Do Nothing on MinGW
}

void SetThreadName(void* thread, const char* name) {
    // Do Nothing on MinGW
}
#endif

#endif

AccurateTimer::AccurateTimer(std::chrono::nanoseconds target_interval)
    : target_interval(target_interval) {}

void AccurateTimer::Start() {
    const auto begin_sleep = std::chrono::high_resolution_clock::now();
    if (total_wait.count() > 0) {
        AccurateSleep(total_wait, nullptr, false);
    }
    start_time = std::chrono::high_resolution_clock::now();
    total_wait -= std::chrono::duration_cast<std::chrono::nanoseconds>(start_time - begin_sleep);
}

void AccurateTimer::End() {
    auto now = std::chrono::high_resolution_clock::now();
    total_wait +=
        target_interval - std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time);
}

std::string GetCurrentThreadName() {
    using namespace Libraries::Kernel;
    if (g_curthread && !g_curthread->name.empty()) {
        if (g_curthread->tcb->tcb_fiber) {
            return fmt::format("{}@@{}", g_curthread->name,
                               g_curthread->tcb->tcb_fiber->current_fiber->name);
        }
        return g_curthread->name;
    }
#ifdef _WIN32
    PWSTR name{};
    if (FAILED(GetThreadDescription(GetCurrentThread(), &name)) || name == nullptr) {
        return "<unknown name>";
    }
    const auto result = Common::UTF16ToUTF8(name);
    LocalFree(name);
    return result;
#else
    char name[256];
    if (pthread_getname_np(pthread_self(), name, sizeof(name)) != 0) {
        return "<unknown name>";
    }
    return std::string{name};
#endif
}

#ifndef _WIN32
std::string BBThreadBusyReport(double, int) {
    return {};
}
#endif

} // namespace Common
