#include "kernel/syncOnAddress.h"
#include "kernel/waitDiag.h"

#include "common/logging/log.h"
#include "common/threads.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <list>
#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
#include <cerrno>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#undef min
#undef max
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#endif

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::LibKernel::SyncOnAddress {

namespace {

constexpr uint32_t SIGNAL_POLL_MICROS = 10000;

using Clock = std::chrono::steady_clock;

template <typename T>
[[nodiscard]] bool IsValidWaitAddress(const volatile T* address) {
	return address != nullptr && (reinterpret_cast<uintptr_t>(address) & (alignof(T) - 1u)) == 0;
}

[[nodiscard]] bool IsValidWakeAddress(const volatile void* address) {
	return address != nullptr &&
	       (reinterpret_cast<uintptr_t>(address) & (alignof(uint32_t) - 1u)) == 0;
}

template <typename T>
[[nodiscard]] T ReadWord(const volatile T* address) {
	return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

struct WaitDeadline {
	bool              finite = false;
	Clock::time_point end {};
};

[[nodiscard]] WaitDeadline MakeDeadline(std::chrono::nanoseconds timeout) {
	const auto now = Clock::now();
	const auto remaining = Clock::time_point::max() - now;
	return {true, timeout >= remaining ? Clock::time_point::max() : now + timeout};
}

[[nodiscard]] WaitDeadline MakeDeadline(const uint32_t* timeout_micros) {
	if (timeout_micros == nullptr) {
		return {};
	}
	return MakeDeadline(std::chrono::microseconds(*timeout_micros));
}

[[nodiscard]] uint32_t GetWaitSliceMicros(const WaitDeadline& deadline, bool first_wait) {
	if (!deadline.finite) {
		return SIGNAL_POLL_MICROS;
	}

	const auto now = Clock::now();
	if (now >= deadline.end) {
		return first_wait ? 0u : UINT32_MAX;
	}

	const auto remaining =
	    std::chrono::ceil<std::chrono::microseconds>(deadline.end - now).count();
	return static_cast<uint32_t>(std::min<int64_t>(remaining, SIGNAL_POLL_MICROS));
}

void PollSignals(signal_poll_func_t signal_poll) {
	if (signal_poll != nullptr) {
		signal_poll();
	}
}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)

template <typename T>
int WaitLinux(volatile T* address, T expected, const WaitDeadline& deadline,
              signal_poll_func_t signal_poll) {
	bool       first_wait = true;

	for (;;) {
		if (ReadWord(address) != expected) {
			return OK;
		}
		const auto slice_micros = GetWaitSliceMicros(deadline, first_wait);
		if (slice_micros == UINT32_MAX) {
			return ReadWord(address) == expected ? KERNEL_ERROR_ETIMEDOUT : OK;
		}
		const timespec timeout = {
		    .tv_sec  = static_cast<time_t>(slice_micros / 1000000u),
		    .tv_nsec = static_cast<long>(slice_micros % 1000000u) * 1000L,
		};

		long result     = 0;
		int  wait_error = 0;
		result          = syscall(SYS_futex, const_cast<T*>(address), FUTEX_WAIT_PRIVATE,
		                          static_cast<uint32_t>(expected), &timeout, nullptr, 0);
		if (result != 0) {
			wait_error = errno;
		}
		if (result == 0 || wait_error == EAGAIN) {
			return OK;
		}
		if (wait_error != ETIMEDOUT && wait_error != EINTR) {
			return KERNEL_ERROR_EINVAL;
		}

		PollSignals(signal_poll);
		if (deadline.finite && Clock::now() >= deadline.end) {
			return ReadWord(address) == expected ? KERNEL_ERROR_ETIMEDOUT : OK;
		}
		first_wait = false;
	}
}

int WakeLinux(volatile void* address, int32_t count) {
	// The legacy FUTEX_WAKE path may wake one waiter when count is zero.
	if (count == 0) {
		return OK;
	}

	const auto result = syscall(SYS_futex, const_cast<void*>(address), FUTEX_WAKE_PRIVATE, count,
	                            nullptr, nullptr, 0);
	return result < 0 ? KERNEL_ERROR_EINVAL : OK;
}

#endif

struct PortableWaiter {
	Common::CondVar condition;
	// Round 132: read without the entry's lock while the waiter spins; set with it held.
	std::atomic<bool> wake_requested {false};
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Round 132: the waiting thread's own auto-reset event (null: the condition variable is used,
	// as before), and when Wake set it.
	HANDLE            event = nullptr;
	Clock::time_point woken_at {};
#endif
};

// Round 132: what the waits did, for the "Sync on address" line every 10 s (any thread).
struct WaitStats {
	std::atomic<uint64_t>                waits {0};   // calls that compared and could wait
	std::atomic<uint64_t>                at_once {0}; // the value differed at the first look
	std::atomic<uint64_t>                spun {0};    // released while spinning, before any sleep
	std::atomic<uint64_t>                slept {0};   // slept at least once
	std::atomic<uint64_t>                timed_out {0};
	std::atomic<uint64_t>                wakes {0};      // Wake calls
	std::atomic<uint64_t>                woken {0};      // waiters a Wake released
	std::atomic<uint64_t>                latency_n {0};  // sleepers woken by a Wake, measured
	std::atomic<uint64_t>                latency_us {0}; // their time from the Wake to their return
	std::atomic<uint64_t>                latency_max {0};
	std::array<std::atomic<uint64_t>, 5> latency_buckets {}; // <10, <50, <200, <1000, more us
	std::atomic<int64_t>                 last_report_ns {0};
};
WaitStats g_stats;

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
	    .count();
}

void NoteLatency(uint64_t us) {
	g_stats.latency_n.fetch_add(1, std::memory_order_relaxed);
	g_stats.latency_us.fetch_add(us, std::memory_order_relaxed);
	auto longest = g_stats.latency_max.load(std::memory_order_relaxed);
	while (us > longest &&
	       !g_stats.latency_max.compare_exchange_weak(longest, us, std::memory_order_relaxed)) {
	}
	const size_t bucket = us < 10 ? 0 : us < 50 ? 1 : us < 200 ? 2 : us < 1000 ? 3 : 4;
	g_stats.latency_buckets[bucket].fetch_add(1, std::memory_order_relaxed);
}

// Every 10 s, by the first wait that sees the time has come.
void MaybeReport() {
	const auto now  = NowNs();
	auto       last = g_stats.last_report_ns.load(std::memory_order_relaxed);
	if (last == 0) {
		(void)g_stats.last_report_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10'000'000'000 ||
	    !g_stats.last_report_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	const auto take = [](std::atomic<uint64_t>& value) {
		return value.exchange(0, std::memory_order_relaxed);
	};
	const auto              waits     = take(g_stats.waits);
	const auto              at_once   = take(g_stats.at_once);
	const auto              spun      = take(g_stats.spun);
	const auto              slept     = take(g_stats.slept);
	const auto              timed_out = take(g_stats.timed_out);
	const auto              wakes     = take(g_stats.wakes);
	const auto              woken     = take(g_stats.woken);
	const auto              n         = take(g_stats.latency_n);
	const auto              total_us  = take(g_stats.latency_us);
	const auto              longest   = take(g_stats.latency_max);
	std::array<uint64_t, 5> buckets {};
	for (size_t i = 0; i < buckets.size(); i++) {
		buckets[i] = take(g_stats.latency_buckets[i]);
	}
	if (waits == 0 && wakes == 0) {
		return;
	}
	const double seconds = static_cast<double>(now - last) / 1e9;
	LOGF("Sync on address (%.1f s): %" PRIu64 " waits (%" PRIu64
	     " found the value changed, %" PRIu64 " released while spinning, %" PRIu64
	     " slept, %" PRIu64 " timed out), %" PRIu64 " wakes releasing %" PRIu64
	     "; a sleeper a wake released returned %.1f us after it on "
	     "average (%" PRIu64 " measured: %" PRIu64 " under 10 us, %" PRIu64 " under 50, %" PRIu64
	     " under 200, %" PRIu64 " under 1000, %" PRIu64 " more; longest %" PRIu64 " us)\n",
	     seconds, waits, at_once, spun, slept, timed_out, wakes, woken,
	     n != 0 ? static_cast<double>(total_us) / static_cast<double>(n) : 0.0, n, buckets[0],
	     buckets[1], buckets[2], buckets[3], buckets[4], longest);
}

// Round 132: KYTY_SYNC_SPIN_US (default 20; 0: none): how long a waiter checks the value and its
// wake before it sleeps. A job system's handoffs come back within microseconds, and a sleeping
// thread takes the scheduler's wakeup to run again. Windows only (Linux waits in the futex).
uint32_t SpinMicros() {
	static const uint32_t micros = [] {
		const char* value = std::getenv("KYTY_SYNC_SPIN_US");
		return value != nullptr && *value != '\0' ? static_cast<uint32_t>(std::min<unsigned long>(
		                                                std::strtoul(value, nullptr, 10), 1000ul))
		                                          : 20u;
	}();
	return micros;
}

// Round 132: the waits of round 131 and before (KYTY_SYNC_WAIT=old, or KYTY_BISECT's
// syncwait:old through SetOldWaits): no spin, and the condition variable, whose waits end with
// up to 0.5 ms slept apart from it (CondVar::WaitFor), when a wake is not seen.
std::atomic<bool> g_old_waits {[] {
	const char* value = std::getenv("KYTY_SYNC_WAIT");
	return value != nullptr && std::string_view(value) == "old";
}()};

void Pause() {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#endif
}

struct PortableAddressEntry {
	Common::Mutex              mutex;
	std::list<PortableWaiter*> waiters;
};

struct PortableAddressRegistry {
	std::mutex                                                           mutex;
	std::unordered_map<uintptr_t, std::shared_ptr<PortableAddressEntry>> entries;
};

PortableAddressRegistry& GetPortableRegistry() {
	static PortableAddressRegistry registry;
	return registry;
}

std::shared_ptr<PortableAddressEntry> RegisterPortableWaiter(volatile void*  address,
                                                             PortableWaiter* waiter) {
	auto&           registry = GetPortableRegistry();
	std::lock_guard registry_lock(registry.mutex);
	auto&           entry = registry.entries[reinterpret_cast<uintptr_t>(address)];
	if (!entry) {
		entry = std::make_shared<PortableAddressEntry>();
	}
	entry->mutex.Lock();
	entry->waiters.push_back(waiter);
	return entry;
}

void UnregisterPortableWaiter(volatile void*                               address,
                              const std::shared_ptr<PortableAddressEntry>& entry,
                              PortableWaiter*                              waiter) {
	entry->waiters.remove(waiter);
	const bool empty = entry->waiters.empty();
	entry->mutex.Unlock();
	if (!empty) {
		return;
	}

	auto&           registry = GetPortableRegistry();
	std::lock_guard registry_lock(registry.mutex);
	entry->mutex.Lock();
	const auto it = registry.entries.find(reinterpret_cast<uintptr_t>(address));
	if (it != registry.entries.end() && it->second == entry && entry->waiters.empty()) {
		registry.entries.erase(it);
	}
	entry->mutex.Unlock();
}

// The condition variable's waits (the only ones before round 132; macOS, and Windows with
// KYTY_SYNC_WAIT=old or F8's syncwait:old).
template <typename T>
int WaitCondVar(volatile T* address, T expected, const WaitDeadline& deadline,
                signal_poll_func_t signal_poll) {
	PortableWaiter waiter;
	auto           entry      = RegisterPortableWaiter(address, &waiter);
	bool           first_wait = true;
	int            result     = OK;

	while (ReadWord(address) == expected && !waiter.wake_requested) {
		const auto slice_micros = GetWaitSliceMicros(deadline, first_wait);
		if (slice_micros == UINT32_MAX) {
			result = KERNEL_ERROR_ETIMEDOUT;
			break;
		}
		if (slice_micros == 0) {
			result = KERNEL_ERROR_ETIMEDOUT;
			break;
		}

		(void)waiter.condition.WaitFor(&entry->mutex, slice_micros);
		entry->mutex.Unlock();
		PollSignals(signal_poll);
		entry->mutex.Lock();

		if (deadline.finite && Clock::now() >= deadline.end && ReadWord(address) == expected &&
		    !waiter.wake_requested) {
			result = KERNEL_ERROR_ETIMEDOUT;
			break;
		}
		first_wait = false;
	}

	if (result == KERNEL_ERROR_ETIMEDOUT) {
		g_stats.timed_out.fetch_add(1, std::memory_order_relaxed);
	}
	UnregisterPortableWaiter(address, entry, &waiter);
	return result;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// Round 132: the calling thread's event and high-resolution timer for its waits (made once).
struct ThreadWaitHandles {
	HANDLE event = nullptr;
	HANDLE timer = nullptr;
	ThreadWaitHandles() {
		event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
		                               TIMER_ALL_ACCESS);
		if (timer == nullptr) {
			timer = CreateWaitableTimerW(nullptr, FALSE, nullptr);
		}
	}
	~ThreadWaitHandles() {
		if (event != nullptr) {
			CloseHandle(event);
		}
		if (timer != nullptr) {
			CloseHandle(timer);
		}
	}
	ThreadWaitHandles(const ThreadWaitHandles&)            = delete;
	ThreadWaitHandles& operator=(const ThreadWaitHandles&) = delete;
};

ThreadWaitHandles& WaitHandles() {
	thread_local ThreadWaitHandles handles;
	return handles;
}

// One sleep on the waiter's event of at most `micros` (the high-resolution timer bounds it; the
// 10 ms signal polls of a wait without a deadline need no more than milliseconds).
void SleepOnEvent(const ThreadWaitHandles& handles, uint32_t micros, bool precise) {
	if (precise && handles.timer != nullptr) {
		LARGE_INTEGER due {};
		due.QuadPart = -static_cast<LONGLONG>(micros) * 10;
		if (SetWaitableTimerEx(handles.timer, &due, 0, nullptr, nullptr, nullptr, 0) != 0) {
			const HANDLE both[2] = {handles.event, handles.timer};
			(void)WaitForMultipleObjects(2, both, FALSE, INFINITE);
			return;
		}
	}
	(void)WaitForSingleObject(handles.event, std::max<DWORD>(1, (micros + 999u) / 1000u));
}

// Round 132: a wait that a wake ends at any point of it. The waiter registers first, so no wake
// is missed; it then checks its value and its wake for SpinMicros before it sleeps on its own
// event, which Wake sets (CondVar::WaitFor slept the last part of each slice apart from the
// condition variable: a wake in it was seen up to 0.5 ms late, in about 5% of a job thread's
// waits in a race).
template <typename T>
int WaitEvent(volatile T* address, T expected, const WaitDeadline& deadline,
              signal_poll_func_t signal_poll) {
	// A poll (a zero or already passed timeout) checks the value and returns without registering,
	// so no wake can be handed to it while it spins (the condition variable's wait held the entry
	// lock from registering to unregistering for one). WaitImpl polls the signals after it.
	if (deadline.finite && Clock::now() >= deadline.end) {
		if (ReadWord(address) != expected) {
			return OK;
		}
		g_stats.timed_out.fetch_add(1, std::memory_order_relaxed);
		return KERNEL_ERROR_ETIMEDOUT;
	}
	auto& handles = WaitHandles();
	if (handles.event == nullptr) {
		return WaitCondVar(address, expected, deadline, signal_poll);
	}
	(void)ResetEvent(handles.event);
	PortableWaiter waiter;
	waiter.event = handles.event;
	auto entry   = RegisterPortableWaiter(address, &waiter);
	int  result  = OK;

	if (const auto spin = SpinMicros(); spin != 0 && ReadWord(address) == expected) {
		entry->mutex.Unlock();
		const auto spin_end = Clock::now() + std::chrono::microseconds(spin);
		bool       released = false;
		for (uint32_t i = 1;; i++) {
			if (ReadWord(address) != expected ||
			    waiter.wake_requested.load(std::memory_order_acquire)) {
				released = true;
				break;
			}
			if ((i & 63u) == 0) {
				const auto now = Clock::now();
				if (now >= spin_end || (deadline.finite && now >= deadline.end)) {
					break;
				}
			}
			Pause();
		}
		entry->mutex.Lock();
		if (released) {
			g_stats.spun.fetch_add(1, std::memory_order_relaxed);
		}
	}

	bool slept      = false;
	bool first_wait = true;
	while (ReadWord(address) == expected && !waiter.wake_requested) {
		const auto slice_micros = GetWaitSliceMicros(deadline, first_wait);
		if (slice_micros == UINT32_MAX || slice_micros == 0) {
			result = KERNEL_ERROR_ETIMEDOUT;
			break;
		}
		entry->mutex.Unlock();
		slept = true;
		SleepOnEvent(handles, slice_micros, deadline.finite);
		PollSignals(signal_poll);
		entry->mutex.Lock();

		if (deadline.finite && Clock::now() >= deadline.end && ReadWord(address) == expected &&
		    !waiter.wake_requested) {
			result = KERNEL_ERROR_ETIMEDOUT;
			break;
		}
		first_wait = false;
	}
	if (slept) {
		g_stats.slept.fetch_add(1, std::memory_order_relaxed);
		if (waiter.wake_requested) {
			const auto us = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
			                                                                      waiter.woken_at)
			                    .count();
			NoteLatency(static_cast<uint64_t>(std::max<int64_t>(us, 0)));
		}
	}
	if (result == KERNEL_ERROR_ETIMEDOUT) {
		g_stats.timed_out.fetch_add(1, std::memory_order_relaxed);
	}
	UnregisterPortableWaiter(address, entry, &waiter);
	return result;
}
#endif

template <typename T>
int WaitPortable(volatile T* address, T expected, const WaitDeadline& deadline,
                 signal_poll_func_t signal_poll) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (!g_old_waits.load(std::memory_order_relaxed)) {
		return WaitEvent(address, expected, deadline, signal_poll);
	}
#endif
	return WaitCondVar(address, expected, deadline, signal_poll);
}

int WakePortable(volatile void* address, int32_t count) {
	if (count == 0) {
		return OK;
	}
	auto&            registry = GetPortableRegistry();
	std::unique_lock registry_lock(registry.mutex);
	const auto       it = registry.entries.find(reinterpret_cast<uintptr_t>(address));
	if (it == registry.entries.end()) {
		return OK;
	}
	auto entry = it->second;
	entry->mutex.Lock();
	registry_lock.unlock();

	int32_t remaining = count;
	for (auto* waiter: entry->waiters) {
		if (!waiter->wake_requested) {
			g_stats.woken.fetch_add(1, std::memory_order_relaxed);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			waiter->woken_at = Clock::now();
#endif
			waiter->wake_requested.store(true, std::memory_order_release);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (waiter->event != nullptr) {
				(void)SetEvent(waiter->event);
			} else {
				waiter->condition.Signal();
			}
#else
			waiter->condition.Signal();
#endif
			if (remaining != INT_MAX && --remaining == 0) {
				break;
			}
		}
	}
	entry->mutex.Unlock();
	return OK;
}

template <typename T>
int WaitImpl(volatile T* address, T expected, const WaitDeadline& deadline,
             signal_poll_func_t signal_poll) {
	if (!IsValidWaitAddress(address)) {
		return KERNEL_ERROR_EINVAL;
	}
	g_stats.waits.fetch_add(1, std::memory_order_relaxed);
	if (ReadWord(address) != expected) {
		g_stats.at_once.fetch_add(1, std::memory_order_relaxed);
	}
	MaybeReport();

	int result = OK;
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
	result = WaitLinux(address, expected, deadline, signal_poll);
#else
	result = WaitPortable(address, expected, deadline, signal_poll);
#endif
	PollSignals(signal_poll);
	return result;
}

} // namespace

void SetOldWaits(bool old) {
	static const bool forced = [] {
		const char* value = std::getenv("KYTY_SYNC_WAIT");
		return value != nullptr && std::string_view(value) == "old";
	}();
	g_old_waits.store(forced || old, std::memory_order_relaxed);
}

int Wait32(volatile uint32_t* address, uint32_t expected, const uint32_t* timeout_micros,
           signal_poll_func_t signal_poll) {
	const WaitDiag::Scope wait_diag("sceKernelSyncOnAddressWait32", reinterpret_cast<uint64_t>(address), {},
	                                timeout_micros == nullptr ? -1 : static_cast<int64_t>(*timeout_micros));
	return WaitImpl(address, expected, MakeDeadline(timeout_micros), signal_poll);
}

int Wait64(volatile uint64_t* address, uint64_t expected, const uint32_t* timeout_micros,
           signal_poll_func_t signal_poll) {
	const WaitDiag::Scope wait_diag("sceKernelSyncOnAddressWait64", reinterpret_cast<uint64_t>(address), {},
	                                timeout_micros == nullptr ? -1 : static_cast<int64_t>(*timeout_micros));
	return WaitImpl(address, expected, MakeDeadline(timeout_micros), signal_poll);
}

int Wait64(volatile uint64_t* address, uint64_t expected, std::chrono::nanoseconds timeout,
           signal_poll_func_t signal_poll) {
	return WaitImpl(address, expected, MakeDeadline(timeout), signal_poll);
}

int Wake(volatile void* address, int32_t count) {
	if (!IsValidWakeAddress(address) || count < 0) {
		return KERNEL_ERROR_EINVAL;
	}
	g_stats.wakes.fetch_add(1, std::memory_order_relaxed);

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
	return WakeLinux(address, count);
#endif
	return WakePortable(address, count);
}

} // namespace Libs::LibKernel::SyncOnAddress
