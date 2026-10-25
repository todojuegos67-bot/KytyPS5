#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_STALL_WATCH_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_STALL_WATCH_H_

// Stall watch (astro-load-1025): finds what a game waits on when it stops making progress while the
// emulator keeps presenting (Astro Bot: a world load that never ends, the loading screen at 60 fps).
//
// Cost while nothing is stuck: each guest thread owns one slot; a blocking kernel wait (contended
// mutex, condition variable, semaphore, event flag, event queue, address wait, sleep, join, APR
// completion) writes what it waits on into the slot and clears it on return (a few relaxed stores
// and one clock read, next to a wait that blocks). File reads, draws/dispatches and flips bump
// relaxed counters. A watch thread wakes twice a second and reads them.
//
// A dump (STALLWATCH lines in the log) when, for KYTY_STALL_WATCH_S seconds (default 10):
//   - no flip at all (a hard hang), or
//   - flips go on, no file read (APR or kernel read) and the frames are light (under 300
//     draws+dispatches a flip: a loading screen or a menu), after reads had been going on, or
//   - light frames for 6x that time, reads or not (a loader stuck in a retry loop).
// Each dump lists every guest thread with its wait (object, name, how long) or "no known wait",
// its guest rip and the guest return addresses found on its stack (module+offset, for a
// disassembler), the AMPR/APR queues (what their heads wait on), and on Windows every host
// thread's stack (module+offset; symbolize with kyty_emulator.map). At most 6 dumps a run; one per
// stall, re-armed when the stall ends. KYTY_STALL_WATCH=0 turns it off.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>

namespace Libs::LibKernel {

// One per guest thread (see StallWatchRegisterThread). Written by its thread, read by the watch.
struct StallWatchSlot {
	std::atomic<void*>                thread {nullptr};
	std::atomic<uint64_t>             host_tid {0};
	std::atomic<const char*>          what {nullptr};
	std::atomic<uint64_t>             object {0};
	std::atomic<int64_t>              since {0};
	std::atomic<uint64_t>             sp {0};
	std::atomic<uint64_t>             waits {0};
	std::array<std::atomic<char>, 40> name {};

	void StoreName(const char* text) noexcept {
		size_t i = 0;
		if (text != nullptr) {
			for (; i + 1 < name.size() && text[i] != '\0'; i++) {
				name[i].store(text[i], std::memory_order_relaxed);
			}
		}
		name[i].store('\0', std::memory_order_relaxed);
	}
};

inline thread_local StallWatchSlot* t_stall_watch_slot = nullptr;

// Marks the calling guest thread as blocked in `what` on `object` (name may be null) until the
// scope ends. Nested scopes restore the outer wait. Only for guest threads (no-op otherwise, or
// when `what` is null). Inline, so the files that use it need no other part of the stall watch.
class GuestWaitScope {
public:
	GuestWaitScope(const char* what, const void* object, const char* name) noexcept {
		auto* slot = t_stall_watch_slot;
		if (slot == nullptr || what == nullptr) {
			return;
		}
		m_slot        = slot;
		m_prev_what   = slot->what.load(std::memory_order_relaxed);
		m_prev_object = slot->object.load(std::memory_order_relaxed);
		m_prev_since  = slot->since.load(std::memory_order_relaxed);
		m_prev_sp     = slot->sp.load(std::memory_order_relaxed);
		slot->StoreName(name);
		slot->object.store(reinterpret_cast<uint64_t>(object), std::memory_order_relaxed);
		slot->since.store(std::chrono::steady_clock::now().time_since_epoch().count(),
		                  std::memory_order_relaxed);
		slot->sp.store(reinterpret_cast<uint64_t>(this), std::memory_order_relaxed);
		slot->what.store(what, std::memory_order_release);
	}
	~GuestWaitScope() {
		if (m_slot == nullptr) {
			return;
		}
		m_slot->waits.fetch_add(1, std::memory_order_relaxed);
		m_slot->object.store(m_prev_object, std::memory_order_relaxed);
		m_slot->since.store(m_prev_since, std::memory_order_relaxed);
		m_slot->sp.store(m_prev_sp, std::memory_order_relaxed);
		if (m_prev_what == nullptr) {
			m_slot->StoreName(nullptr);
		}
		m_slot->what.store(m_prev_what, std::memory_order_release);
	}
	GuestWaitScope(const GuestWaitScope&)            = delete;
	GuestWaitScope& operator=(const GuestWaitScope&) = delete;

private:
	StallWatchSlot* m_slot        = nullptr;
	const char*     m_prev_what   = nullptr;
	uint64_t        m_prev_object = 0;
	int64_t         m_prev_since  = 0;
	uint64_t        m_prev_sp     = 0;
};

// pthread.cpp: a guest thread starts (pthread is its Pthread) or ends.
void StallWatchRegisterThread(void* pthread, uint64_t host_tid, uint64_t stack_low,
                              uint64_t stack_high) noexcept;
void StallWatchUnregisterThread() noexcept;

namespace StallWatchCounters {
inline std::atomic<uint64_t> g_io {0};
inline std::atomic<uint64_t> g_gpu_work {0};
inline std::atomic<uint64_t> g_flips {0};
inline std::atomic<uint64_t> g_ampr_rejected {0};
} // namespace StallWatchCounters

inline void StallWatchNoteIo() noexcept {
	StallWatchCounters::g_io.fetch_add(1, std::memory_order_relaxed);
}
inline void StallWatchNoteGpuWork() noexcept {
	StallWatchCounters::g_gpu_work.fetch_add(1, std::memory_order_relaxed);
}
inline void StallWatchNoteFlip() noexcept {
	StallWatchCounters::g_flips.fetch_add(1, std::memory_order_relaxed);
}
// astro-sync-1025: which files are read (kernel reads and APR reads), for a dump's "files read" lines: a
// loader stuck re-reading one file, or the stream that keeps the reads going while a load never finishes.
// A short lock and a map update per read (reads are system calls); nothing when the watch is off.
void StallWatchNoteFileRead(const std::filesystem::path& path, int64_t offset, uint64_t size) noexcept;
// astro-tunnel-1025: a full STALLWATCH dump on the watch thread within half a second (F10 in the game window;
// creating stalldump.txt in the emulator's folder does the same). Works with the watch on, up to 20 a run.
void StallWatchRequestDump() noexcept;

inline uint64_t StallWatchNoteAmprRejected() noexcept {
	return StallWatchCounters::g_ampr_rejected.fetch_add(1, std::memory_order_relaxed) + 1;
}

// Starts the watch thread (once; reads KYTY_STALL_WATCH / KYTY_STALL_WATCH_S).
void StallWatchStart();

} // namespace Libs::LibKernel

namespace Libs::LibAmpr {
// libAmpr.cpp: the APR/AMM queues for a dump (never blocks).
void DescribeStateForStallWatch(std::string* out);
} // namespace Libs::LibAmpr

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_STALL_WATCH_H_ */
