#pragma once
// KYTY_ASYNC_UPLOAD: buffer upload copies (guest memory -> host-visible staging memory) run on
// one worker, in the order the render thread recorded them; a queue submission first waits for
// the copies recorded before it. The render thread only records the copy commands.
// KYTY_ASYNC_UPLOAD=2: the staging copies of image uploads (ObtainBufferForImage) as well.
//
// The worker reads the guest range through the guest's mapping where it can (CopyFrom: a page without access
// ends a guarded copy) and through the backing view past that: a guest write or protection change never takes it
// into the guest's fault handler, and unmapping a range waits for the jobs on it before the range can be reused.

#include "common/guardedCopy.h"
#include "live-census.h"
#include "live-counters.h"
#include "local-platform.h"
#include "parallel-pages.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <thread>
#include <vector>
#include <x86intrin.h>

extern "C" {
extern volatile std::atomic<uint32_t> kyty_local_async_upload_mode;
}

namespace AsyncUpload {

class Worker {
public:
	static constexpr uint64_t Capacity = 1u << 16;

	Worker(): m_jobs(std::make_unique<Job[]>(Capacity)), m_thread([this] { Run(); }) {}
	~Worker() {
		m_stop.store(true, std::memory_order_seq_cst);
		m_head.fetch_add(0, std::memory_order_seq_cst);
		m_head.notify_all();
		m_thread.join();
	}
	Worker(const Worker&)            = delete;
	Worker& operator=(const Worker&) = delete;

	// Render thread only: copies the backing of guest [guest, guest + size). Returns the sequence number
	// that covers this copy.
	uint64_t Push(uint8_t* destination, const uint8_t* source, uint64_t size, uint64_t guest) {
		return Add({destination, source, size, nullptr, nullptr, guest, size, false});
	}
	// Render thread only: runs call(context, a, b) on the worker in queue order; it works on guest
	// [guest, guest + guest_size).
	uint64_t PushCall(void (*call)(void*, uint64_t, uint64_t), void* context, uint64_t a, uint64_t b, uint64_t guest,
	                  uint64_t guest_size) {
		return Add({reinterpret_cast<uint8_t*>(a), nullptr, b, call, context, guest, guest_size, false});
	}
	// Render thread only: as PushCall, for a write protection of guest pages the copies pushed after it read
	// (KYTY_ASYNC_REPROTECT): it may run before copies pushed ahead of it too (Run).
	uint64_t PushProtection(void (*call)(void*, uint64_t, uint64_t), void* context, uint64_t a, uint64_t b,
	                        uint64_t guest, uint64_t guest_size) {
		return Add({reinterpret_cast<uint8_t*>(a), nullptr, b, call, context, guest, guest_size, true});
	}
	// Render thread only: wakes the worker after a batch of pushes.
	void Kick() {
		if (m_sleeping.load(std::memory_order_seq_cst)) m_head.notify_one();
	}
	// Render thread only: the sequence number of the last pushed copy.
	[[nodiscard]] uint64_t Pushed() const { return m_head.load(std::memory_order_relaxed); }
	// Render thread only: the sequence number of the last pending job on guest [vaddr, vaddr + size), or 0.
	[[nodiscard]] uint64_t PendingOn(uint64_t vaddr, uint64_t size) const {
		const uint64_t head = m_head.load(std::memory_order_relaxed);
		uint64_t       last = 0;
		for (uint64_t sequence = m_done.load(std::memory_order_acquire); sequence < head; ++sequence) {
			const Job& job = m_jobs[sequence % Capacity];
			if (job.guest < vaddr + size && vaddr < job.guest + job.guest_size) last = sequence + 1;
		}
		return last;
	}

	// Any thread: returns once every copy up to `sequence` is in staging memory.
	void Wait(uint64_t sequence) {
		if (m_done.load(std::memory_order_acquire) >= sequence) return;
		const auto start = std::chrono::steady_clock::now();
		const auto note  = [&] {
			const auto ns = (std::chrono::steady_clock::now() - start).count();
			LiveCensus::g_upload_wait_ns.fetch_add(ns, std::memory_order_relaxed);
			LiveCounters::Add(LiveCounters::UploadWaitUs, static_cast<uint64_t>(ns) / 1000u);
		};
		for (int spin = 0; spin < 4000; ++spin) {
			if (m_done.load(std::memory_order_acquire) >= sequence) return note();
			_mm_pause();
		}
		m_waiters.fetch_add(1, std::memory_order_seq_cst);
		for (uint64_t done; (done = m_done.load(std::memory_order_seq_cst)) < sequence;)
			m_done.wait(done, std::memory_order_seq_cst);
		m_waiters.fetch_sub(1, std::memory_order_relaxed);
		note();
	}

private:
	struct Job {
		uint8_t*       destination;
		const uint8_t* source;
		uint64_t       size;
		void (*call)(void*, uint64_t, uint64_t); // with destination as `a`, size as `b`
		void*    context;
		uint64_t guest, guest_size;
		bool     protection; // a call PushProtection queued
	};

	uint64_t Add(const Job& job) {
		const uint64_t head = m_head.load(std::memory_order_relaxed);
		while (head - m_done.load(std::memory_order_acquire) >= Capacity) {
			Kick();
			_mm_pause();
		}
		m_jobs[head % Capacity] = job;
		m_head.store(head + 1, std::memory_order_seq_cst);
		return head + 1;
	}

	void Run() {
		LocalPlatform::SetThreadName("Kyty.Upload");
		// Created by the render thread: do not inherit its dedicated CPU.
		LocalPlatform::PinThreadToCpuList(std::getenv("KYTY_RECORDING_CPUS"));
		// Pause iterations the worker polls for new copies before it sleeps.
		constexpr uint32_t UploadSpins = 20000;
		uint64_t           done        = 0;
		for (;;) {
			uint64_t head = m_head.load(std::memory_order_acquire);
			for (uint32_t spin = 0; head == done && spin < UploadSpins; ++spin) {
				_mm_pause();
				head = m_head.load(std::memory_order_acquire);
			}
			if (head == done) {
				m_sleeping.store(true, std::memory_order_seq_cst);
				head = m_head.load(std::memory_order_seq_cst);
				if (head == done) {
					if (m_stop.load(std::memory_order_seq_cst)) return;
					m_head.wait(done, std::memory_order_seq_cst);
				}
				m_sleeping.store(false, std::memory_order_relaxed);
				continue;
			}
			const auto start = std::chrono::steady_clock::now();
			const auto since = [](std::chrono::steady_clock::time_point from) {
				return static_cast<uint64_t>((std::chrono::steady_clock::now() - from).count()) / 1000u;
			};
			const auto call = [](const Job& job) {
				job.call(job.context, reinterpret_cast<uint64_t>(job.destination), job.size);
			};
			while (done < head) {
				if (const Job& job = m_jobs[done % Capacity]; job.call != nullptr && !job.protection) {
					const auto called = std::chrono::steady_clock::now();
					call(job);
					LiveCounters::Add(LiveCounters::UploadCallUs, since(called));
					++done;
					continue;
				}
				// The jobs up to the next other call (which keeps its place): their write protections first, in their
				// order, then their copies, which are independent of each other. A copy follows the protection of its
				// pages (pushed before it) either way: one pushed after the copy only re-arms its pages sooner, and the
				// copy that is its own still comes after it. (Each upload pushes its protections then its copies: in
				// queue order the copies came in runs of one upload's few, a few KiB to tens of KiB each.)
				uint64_t end = done, bytes = 0;
				for (; end < head; ++end) {
					const Job& next = m_jobs[end % Capacity];
					if (next.call == nullptr) bytes += next.size;
					else if (!next.protection) break;
				}
				const auto protecting = std::chrono::steady_clock::now();
				for (uint64_t i = done; i < end; ++i)
					if (const Job& next = m_jobs[i % Capacity]; next.call != nullptr) call(next);
				LiveCounters::Add(LiveCounters::UploadCallUs, since(protecting));
				CopyRun(done, end, bytes);
				done = end;
			}
			LiveCounters::Add(LiveCounters::UploadBusyUs, since(start));
			m_done.store(done, std::memory_order_seq_cst);
			if (m_waiters.load(std::memory_order_seq_cst) != 0) m_done.notify_all();
		}
	}

	// The copies among jobs [first, last) (calls skipped), `bytes` in all. Copies of 1 MiB or more together are
	// shared with helper threads of their own (not the render thread's page comparisons' pool, which would wait
	// behind it), as one job over their bytes: an image the game streams in comes as many copies (its pieces, its
	// small neighbours), and a frame where it brings a new area (~1 GB of textures, every submission waiting for their
	// copies) stalled ~300 ms. (Copied one by one, a walk's slow frames at Boletarian Palace 1-3 waited 25-28 ms for
	// them; as runs, 14 ms.)
	void CopyRun(uint64_t first, uint64_t last, uint64_t bytes) {
		constexpr uint64_t Chunk = uint64_t {256} << 10u;
		if (bytes == 0) return;
		if (bytes < 4 * Chunk) {
			for (; first < last; ++first)
				if (const Job& job = m_jobs[first % Capacity]; job.call == nullptr)
					CopyFrom(job.destination, job.source, job.guest, job.size);
			return;
		}
		// The run's copies, each at its offset in the run's bytes.
		m_run.clear();
		uint64_t offset = 0;
		for (; first < last; ++first) {
			const Job& job = m_jobs[first % Capacity];
			if (job.call != nullptr) continue;
			m_run.push_back({offset, job.destination, job.source, job.guest});
			offset += job.size;
		}
		m_run.push_back({offset, nullptr, nullptr, 0});
		LiveCounters::Add(LiveCounters::UploadSharedRuns);
		static auto* const pool = new ParallelPages::Pool(7, "Kyty.UploadCopy"); // (never destroyed)
		pool->For(bytes, Chunk, [](void* at, uint64_t from, uint64_t to) {
			const auto& run = *static_cast<const std::vector<RunCopy>*>(at);
			// The copy holding byte `from`, then the ones after it up to `to`.
			auto it = std::prev(std::upper_bound(run.begin(), std::prev(run.end()), from,
			                                     [](uint64_t at_byte, const RunCopy& copy) { return at_byte < copy.offset; }));
			for (; from < to; ++it) {
				const auto end  = std::min(to, std::next(it)->offset);
				const auto skip = from - it->offset;
				CopyFrom(it->destination + skip, it->source + skip, it->guest + skip, end - from);
				from = end;
			}
		}, &m_run);
	}

	// The guest's bytes read through its own mapping where it can (Windows): a page of the backing view faults on its
	// first read (~1.2 us: 2.3-3.8 GB/s on one thread, 6-11 GB/s on eight), where most uploaded pages were just
	// written through the guest mapping, which has them (the waits of those 1-3 slow frames: 18 -> 4 ms). A page that
	// mapping cannot read (GPU-written, read-protected) ends that part (GuardedCopy); the rest comes from the backing
	// view, the same bytes (unmapping a range waits for its jobs: the guest mapping is the backing's).
	static void CopyFrom(uint8_t* destination, const uint8_t* backing, uint64_t guest, uint64_t size) {
#if defined(_WIN32)
		if (const auto left = Common::GuardedCopy(destination, reinterpret_cast<const void*>(guest), size); left != 0) {
			LiveCounters::Add(LiveCounters::UploadGuardStops);
			const auto done = size - left;
			std::memcpy(destination + done, backing + done, left);
		}
#else
		std::memcpy(destination, backing, size);
#endif
	}

	struct RunCopy {
		uint64_t       offset;
		uint8_t*       destination;
		const uint8_t* source;
		uint64_t       guest;
	};
	std::vector<RunCopy>               m_run; // (the worker's)
	std::unique_ptr<Job[]>             m_jobs;
	alignas(64) std::atomic<uint64_t>  m_head {0};
	alignas(64) std::atomic<uint64_t>  m_done {0};
	alignas(64) std::atomic<bool>      m_sleeping {false};
	std::atomic<uint32_t>              m_waiters {0};
	std::atomic<bool>                  m_stop {false};
	std::thread                        m_thread;
};

[[nodiscard]] inline bool Enabled() {
	return kyty_local_async_upload_mode.load(std::memory_order_relaxed) != 0;
}

// Set by the render thread before its first push; waits stay armed after the switch goes off.
inline std::atomic<bool> g_used {false};

inline Worker& Get() {
	// Never destroyed: threads that still record or submit may outlive static destruction.
	static Worker* const worker = new Worker();
	g_used.store(true, std::memory_order_relaxed);
	return *worker;
}

// Render thread: the sequence a submission recorded now must wait for (0: nothing pending).
[[nodiscard]] inline uint64_t SubmitSequence() {
	return g_used.load(std::memory_order_relaxed) ? Get().Pushed() : 0;
}

// Any thread, with a sequence taken on the render thread.
inline void Wait(uint64_t sequence) {
	if (sequence != 0) Get().Wait(sequence);
}

// Render thread: every recorded copy reaches staging memory before this returns.
inline void Drain() {
	Wait(SubmitSequence());
}

// Render thread: every recorded job on guest [vaddr, vaddr + size) is done before this returns.
inline void DrainRange(uint64_t vaddr, uint64_t size) {
	if (g_used.load(std::memory_order_relaxed)) Wait(Get().PendingOn(vaddr, size));
}

} // namespace AsyncUpload
