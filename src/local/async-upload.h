#pragma once
// KYTY_ASYNC_UPLOAD: buffer upload copies (guest backing -> host-visible staging memory) run on
// one worker, in the order the render thread recorded them; a queue submission first waits for
// the copies recorded before it. The render thread only records the copy commands.
// KYTY_ASYNC_UPLOAD=2: the staging copies of image uploads (ObtainBufferForImage) as well.
//
// The worker reads the backing view, never the guest range: a guest write or protection change
// cannot fault it, and unmapping a range waits for the jobs on it before the range can be reused.

#include "local-platform.h"
#include "parallel-pages.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
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
		return Add({destination, source, size, nullptr, nullptr, guest, size});
	}
	// Render thread only: runs call(context, a, b) on the worker in queue order; it works on guest
	// [guest, guest + guest_size).
	uint64_t PushCall(void (*call)(void*, uint64_t, uint64_t), void* context, uint64_t a, uint64_t b, uint64_t guest,
	                  uint64_t guest_size) {
		return Add({reinterpret_cast<uint8_t*>(a), nullptr, b, call, context, guest, guest_size});
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
		for (int spin = 0; spin < 4000; ++spin) {
			if (m_done.load(std::memory_order_acquire) >= sequence) return;
			_mm_pause();
		}
		m_waiters.fetch_add(1, std::memory_order_seq_cst);
		for (uint64_t done; (done = m_done.load(std::memory_order_seq_cst)) < sequence;)
			m_done.wait(done, std::memory_order_seq_cst);
		m_waiters.fetch_sub(1, std::memory_order_relaxed);
	}

private:
	struct Job {
		uint8_t*       destination;
		const uint8_t* source;
		uint64_t       size;
		void (*call)(void*, uint64_t, uint64_t); // with destination as `a`, size as `b`
		void*    context;
		uint64_t guest, guest_size;
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
			for (; done < head; ++done) {
				const Job& job = m_jobs[done % Capacity];
				if (job.call != nullptr) job.call(job.context, reinterpret_cast<uint64_t>(job.destination), job.size);
				else Copy(job.destination, job.source, job.size);
			}
			m_done.store(done, std::memory_order_seq_cst);
			if (m_waiters.load(std::memory_order_seq_cst) != 0) m_done.notify_all();
		}
	}

	// A copy of 1 MiB or more is shared with helper threads of its own (not the render thread's page comparisons'
	// pool, which would wait behind it). The first read of each source page through this view faults (~1.2 us a
	// page: 2.3-3.8 GB/s on one thread, 6-11 GB/s on eight), and an image the game streams in is read once: a frame
	// where it brings a new area (~1 GB of textures, every submission waiting for their copies) stalled ~300 ms.
	static void Copy(uint8_t* destination, const uint8_t* source, uint64_t size) {
		constexpr uint64_t Chunk = uint64_t {256} << 10u;
		if (size < 4 * Chunk) {
			std::memcpy(destination, source, size);
			return;
		}
		static auto* const pool = new ParallelPages::Pool(7, "Kyty.UploadCopy"); // (never destroyed)
		struct Context {
			uint8_t*       destination;
			const uint8_t* source;
		} context {destination, source};
		pool->For(size, Chunk, [](void* at, uint64_t first, uint64_t last) {
			const auto& copy = *static_cast<const Context*>(at);
			std::memcpy(copy.destination + first, copy.source + first, last - first);
		}, &context);
	}

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
