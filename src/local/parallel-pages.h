#pragma once
// A page-wise job of the render thread shared with helper threads kept off the render CPUs (KYTY_RENDER_CPUS): every
// participant takes chunks of it until none is left, the caller too, so a helper slow to wake leaves its share to
// the others; the caller returns once every chunk is done. For memory-bound work too big for one core (the
// comparison of a large linear copy's bytes, BufferCache::CopyGuestMemory: 7 helpers on the E-cores took the 4 MiB
// comparison of the culling chain from 0.39 to 0.24 ms, 3 to 0.27 ms; they wake late, the caller does the most). One
// job at a time (callers on other threads wait for it).

#include "local-platform.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <x86intrin.h>

namespace ParallelPages {

using Job = void (*)(void* context, uint64_t first, uint64_t last);

class Pool {
public:
	explicit Pool(uint32_t helpers = 7, const char* name = "Kyty.Pages"): m_name(name) {
		for (uint32_t i = 0; i < helpers; ++i) std::thread([this] { Run(); }).detach();
	}
	Pool(const Pool&)            = delete;
	Pool& operator=(const Pool&) = delete;

	// job(context, first, last) over [0, count) in chunks of `chunk` (fewer than 2^24 of them).
	void For(uint64_t count, uint64_t chunk, Job job, void* context) {
		std::lock_guard lock(m_mutex);
		m_job     = job;
		m_context = context;
		m_count   = count;
		m_chunk   = std::max<uint64_t>(chunk, 1);
		m_chunks  = (count + m_chunk - 1) / m_chunk;
		m_done.store(0, std::memory_order_relaxed);
		const uint32_t job_id = ++m_jobs & 0xffffu;
		// (The fields above are the job's for whoever takes a chunk of it: they acquire its ticket. The ticket holds the
		// chunk count too: a helper of an earlier job reads nothing this one may be writing.)
		m_tickets.store(uint64_t {job_id} << 48u | m_chunks << 24u, std::memory_order_release);
		m_posted.store(job_id, std::memory_order_seq_cst);
		m_posted.notify_all();
		Work(job_id);
		// (Only chunks a helper took and has not finished are left.)
		for (uint32_t spin = 0; m_done.load(std::memory_order_acquire) != m_chunks; ++spin) {
			if (spin < 20000) {
				_mm_pause();
				continue;
			}
			const auto done = m_done.load(std::memory_order_acquire);
			if (done != m_chunks) m_done.wait(done, std::memory_order_acquire);
		}
	}

private:
	// Takes chunks of job `job_id` until none is left; nothing once another job replaced it. (A chunk taken is not
	// done: its job's fields stay until it is.)
	void Work(uint32_t job_id) {
		for (;;) {
			auto ticket = m_tickets.load(std::memory_order_acquire);
			do {
				if ((ticket >> 48u) != job_id || (ticket & 0xffffffu) >= ((ticket >> 24u) & 0xffffffu)) return;
			} while (!m_tickets.compare_exchange_weak(ticket, ticket + 1, std::memory_order_acq_rel));
			const auto first = (ticket & 0xffffffu) * m_chunk;
			m_job(m_context, first, std::min(first + m_chunk, m_count));
			m_done.fetch_add(1, std::memory_order_acq_rel);
			m_done.notify_one();
		}
	}

	void Run() {
		LocalPlatform::SetThreadName(m_name);
		LocalPlatform::AvoidCpuList(std::getenv("KYTY_RENDER_CPUS"));
		uint32_t seen = 0;
		for (;;) {
			uint32_t posted = 0;
			while ((posted = m_posted.load(std::memory_order_acquire)) == seen) m_posted.wait(seen, std::memory_order_acquire);
			seen = posted;
			Work(posted);
		}
	}

	const char*                       m_name;
	std::mutex                        m_mutex;
	uint32_t                          m_jobs    = 0; // (the caller's, under the mutex)
	Job                               m_job     = nullptr;
	void*                             m_context = nullptr;
	uint64_t                          m_count = 0, m_chunk = 1, m_chunks = 0;
	alignas(64) std::atomic<uint32_t> m_posted {0};
	alignas(64) std::atomic<uint64_t> m_tickets {0}; // job id << 48 | chunks << 24 | the next chunk
	alignas(64) std::atomic<uint64_t> m_done {0};
};

inline void For(uint64_t count, uint64_t chunk, Job job, void* context) {
	static auto* pool = new Pool(); // (never destroyed: its threads wait until the process ends)
	pool->For(count, chunk, job, context);
}

} // namespace ParallelPages
