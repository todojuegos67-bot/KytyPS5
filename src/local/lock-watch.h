#ifndef EMULATOR_SRC_LOCAL_LOCK_WATCH_H_
#define EMULATOR_SRC_LOCAL_LOCK_WATCH_H_

// (ram13b) Who waits for which lock, for a hang with no exception (log 11-16 02:52: "[flip] 0 flips/s" for good, the
// audio going on, no thread dump). Each instrumented lock is taken through Wait (the thread's slot says "waiting for X
// since t"; counted per lock with the time) and marked held until Release. A watchdog thread (KYTY_LOCK_WATCH=1, or
// with the cold store on) looks once a second: a wait older than KYTY_LOCK_WATCH_S (3 s) prints, once per stall, every
// thread waiting (which lock, for how long) and every thread holding an instrumented lock, plus the counters per lock.
// Lock-free, no allocation (256 thread slots; a thread past them is not watched).

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace LockWatch {

enum Id : uint32_t {
	GateAlone = 0, // raiz's protection gate, taken alone (a window, the cold store's eviction)
	GateShared,    // the same, shared (a protection change, the cold store's probe)
	ColdMutex,     // the cold store's table
	PendingMutex,  // raiz's lazy write-back store
	ShadowMutex,   // raiz's shadows
	PmGranule,     // (raiz10) the page manager's lock of a 64 KiB granule (only its waits: the holder is not marked)
	StripeAlone,   // (raiz10) a keep's stripe of the gate, alone (KYTY_RAIZ_KEEP_STRIPES)
	StripeShared,  // (raiz10) the same, shared (a protection change that makes pages readable)
	Ids
};
inline constexpr const char* Names[Ids] = {"gate-alone", "gate-shared", "cold-mutex", "lazy-mutex", "shadow-mutex",
                                           "pm-granule", "stripe-alone", "stripe-shared"};

struct Slot {
	std::atomic<uint32_t> tid {0};
	std::atomic<uint32_t> waiting {0}; // Id + 1, 0 none
	std::atomic<uint64_t> since_us {0};
	std::atomic<uint32_t> held {0}; // a bit per Id (depth not kept: a re-entry is not marked twice)
};
inline Slot                  g_slots[256];
inline std::atomic<uint32_t> g_used {0};
inline std::atomic<uint64_t> g_waits[Ids] {}, g_wait_us[Ids] {}, g_wait_max_us[Ids] {}, g_takes[Ids] {};

inline uint64_t NowUs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline uint32_t ThreadId() {
#if defined(_WIN32)
	return static_cast<uint32_t>(GetCurrentThreadId());
#else
	return static_cast<uint32_t>(syscall(SYS_gettid));
#endif
}
inline Slot* Mine() {
	thread_local Slot* slot = [] {
		const uint32_t i = g_used.fetch_add(1, std::memory_order_relaxed);
		if (i >= 256) return static_cast<Slot*>(nullptr);
		g_slots[i].tid.store(ThreadId(), std::memory_order_relaxed);
		return &g_slots[i];
	}();
	return slot;
}

// Around the taking of lock `id`: the slot says it waits meanwhile; held after.
class Wait {
public:
	explicit Wait(Id id): m_id(id), m_slot(Mine()), m_start(NowUs()) {
		if (m_slot != nullptr) {
			m_slot->since_us.store(m_start, std::memory_order_relaxed);
			m_slot->waiting.store(static_cast<uint32_t>(id) + 1u, std::memory_order_release);
		}
	}
	~Wait() {
		const uint64_t us = NowUs() - m_start;
		g_takes[m_id].fetch_add(1, std::memory_order_relaxed);
		if (us >= 50) {
			g_waits[m_id].fetch_add(1, std::memory_order_relaxed);
			g_wait_us[m_id].fetch_add(us, std::memory_order_relaxed);
			uint64_t m = g_wait_max_us[m_id].load(std::memory_order_relaxed);
			while (us > m && !g_wait_max_us[m_id].compare_exchange_weak(m, us, std::memory_order_relaxed)) {
			}
		}
		if (m_slot != nullptr) {
			m_slot->waiting.store(0, std::memory_order_release);
			m_slot->held.fetch_or(1u << m_id, std::memory_order_relaxed);
		}
	}
	Wait(const Wait&)            = delete;
	Wait& operator=(const Wait&) = delete;

private:
	Id       m_id;
	Slot*    m_slot;
	uint64_t m_start;
};
// Taken without waiting (a try_lock that succeeded): only marked held.
inline void Held(Id id) {
	if (Slot* s = Mine(); s != nullptr) s->held.fetch_or(1u << id, std::memory_order_relaxed);
	g_takes[id].fetch_add(1, std::memory_order_relaxed);
}
inline void Release(Id id) {
	if (Slot* s = Mine(); s != nullptr) s->held.fetch_and(~(1u << id), std::memory_order_relaxed);
}

// The watchdog's look (public for a test): true when it printed.
inline bool Look(uint64_t stall_us) {
	const uint64_t now   = NowUs();
	const uint32_t n     = std::min<uint32_t>(g_used.load(std::memory_order_relaxed), 256u);
	bool           stuck = false;
	for (uint32_t i = 0; i < n && !stuck; ++i)
		stuck = g_slots[i].waiting.load(std::memory_order_acquire) != 0 &&
		        now - g_slots[i].since_us.load(std::memory_order_relaxed) >= stall_us;
	static bool said = false; // (once per stall)
	if (!stuck) {
		said = false;
		return false;
	}
	if (said) return false;
	said = true;
	std::printf("[lockwatch] STALL: a lock waited for over %.1f s.", static_cast<double>(stall_us) / 1e6);
	for (uint32_t i = 0; i < n; ++i) {
		const uint32_t w = g_slots[i].waiting.load(std::memory_order_acquire), h = g_slots[i].held.load(std::memory_order_relaxed);
		if (w == 0 && h == 0) continue;
		std::printf(" | thread %u", g_slots[i].tid.load(std::memory_order_relaxed));
		if (w != 0)
			std::printf(" WAITS %s for %.2f s", Names[w - 1],
			            static_cast<double>(now - g_slots[i].since_us.load(std::memory_order_relaxed)) / 1e6);
		for (uint32_t k = 0; k < Ids; ++k)
			if ((h >> k) & 1u) std::printf(" holds %s", Names[k]);
	}
	std::printf(" | per lock (taken, waits over 50 us, total ms, longest ms):");
	for (uint32_t k = 0; k < Ids; ++k)
		std::printf(" %s %llu/%llu/%.0f/%.1f", Names[k], static_cast<unsigned long long>(g_takes[k].load()),
		            static_cast<unsigned long long>(g_waits[k].load()), static_cast<double>(g_wait_us[k].load()) / 1000.0,
		            static_cast<double>(g_wait_max_us[k].load()) / 1000.0);
	std::printf("\n");
	std::fflush(stdout);
	return true;
}

// Once; KYTY_LOCK_WATCH_S (3) the stall's length.
inline void Start() {
	static std::atomic<bool> started {false};
	if (started.exchange(true)) return;
	std::thread([] {
		const char* v       = std::getenv("KYTY_LOCK_WATCH_S");
		const auto  seconds = v != nullptr && v[0] >= '1' && v[0] <= '9' ? std::strtoull(v, nullptr, 10) : 3ull;
		for (;;) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
			(void)Look(seconds * 1000000ull);
		}
	}).detach();
}

} // namespace LockWatch

#endif // EMULATOR_SRC_LOCAL_LOCK_WATCH_H_
