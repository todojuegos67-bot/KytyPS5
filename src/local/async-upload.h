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
#include "spin-gate.h"
#include "write-guard.h"

#include <algorithm>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#include <x86intrin.h>

extern "C" {
extern volatile std::atomic<uint32_t> kyty_local_async_upload_mode;
// KYTY_UPLOAD_LANES (live 1..3, default 1; PR #287): 2 = a run's write protections split by
// guest region (4 MiB, the trackers' and the page manager's region: their per-region locks) over the worker and a second thread
// (Kyty.Upload2), which sleeps on a futex between runs and never spins; the run's copies still start only after both lanes are done,
// and one region's protections stay in queue order in its one lane. 3 = 2 with the order checks (upload_lane_mismatch must stay 0).
// (At 1-1 the waited batches' protections were all single-region with no call among them, the lanes 45-57 / 43-55 %.)
extern volatile std::atomic<uint32_t> kyty_local_upload_lanes;
// KYTY_UPLOAD_HELP (live kyty_local_upload_help): whether the render thread's Drain runs the jobs the upload worker is not
// at itself (Worker::Help): 1 always (the Steam Deck build's default), 0 never, 2 (the default elsewhere) while the
// process's CPUs are busy, the spin gate shut (fewer than three of them idle, spin-gate.h). On the Deck's four cores the
// worker, asleep or waiting for a CPU, left the render thread waiting 7.2-7.7 ms a frame, 1.1-1.3 with it helping (stand
// 30.5 -> 32.8 fps). The 22-CPU desktop held to six of them: the wait 1.8-1.9 -> 0.6 ms, 63.7-66.3 -> 66.6-68.5 fps at
// 1-1, the same lows; to four: 4.1-4.4 -> 0.9-1.0 ms, +4% (the jobs at normal priority: Help). With CPUs to spare the
// worker is at its jobs at once, and the render thread taking them gained nothing and made the 1% lows worse (10-27).
inline volatile std::atomic<uint32_t> kyty_local_upload_help {[] {
	const char* value = std::getenv("KYTY_UPLOAD_HELP");
	if (value != nullptr && (value[0] == '0' || value[0] == '1' || value[0] == '2') && value[1] == '\0')
		return static_cast<uint32_t>(value[0] - '0');
#if defined(KYTY_STEAM_DECK)
	return 1u;
#else
	return 2u;
#endif
}()};
}

namespace AsyncUpload {

// True on the worker's thread (a fault there cannot wait for the render thread, which may wait for the worker).
inline thread_local bool t_worker = false;

// KYTY_SERVE_IN_WAITS bit 1 (GuestGpu::ServeInWaits), set by the GPU thread. Asked only by a Wait that was told to serve
// (the drains before a guest memory write of the command processor and at the frame boundary; a submit's or an
// unmap's drain never serves). With false: whether that Wait spins to the copies' end, running the guest commands
// queued for the GPU thread meanwhile (a game thread's readback fault waits for its command there); with true: run
// them, true when some ran.
inline bool (*g_serve_hook)(bool serve) = nullptr;

class Worker {
public:
	static constexpr uint64_t Capacity = 1u << 16;
	// KYTY_UPLOAD_LANES: the guest region a lane is chosen by: the trackers' region managers' and the page manager's region,
	// whose locks protections take (bufferCache.cpp asserts it is TRACKER_REGION_SIZE).
	static constexpr uint64_t LaneRegion = uint64_t {4} << 20u;

	Worker(): m_jobs(std::make_unique<Job[]>(Capacity)), m_thread([this] { Run(); }) {}
	~Worker() {
		m_stop.store(true, std::memory_order_seq_cst);
		m_head.fetch_add(0, std::memory_order_seq_cst);
		m_head.notify_all();
		m_thread.join();
		// (The wake below reads as a post: no jobs in it. The worker is gone and the second thread idle.)
		m_lane_jobs[1].clear();
		m_lane_posted.fetch_add(1, std::memory_order_seq_cst);
		m_lane_posted.notify_all();
		if (m_lane_thread.joinable()) m_lane_thread.join();
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
	// Render thread only: the sequence number of the last finished job (KYTY_WRITE_GUARD: whether any is pending).
	[[nodiscard]] uint64_t Done() const { return m_done.load(std::memory_order_acquire); }
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

	// Any thread: returns once every copy up to `sequence` is in staging memory. `serve`: GPU thread only, run the
	// guest commands queued meanwhile (g_serve_hook).
	void Wait(uint64_t sequence, bool serve = false) {
		if (m_done.load(std::memory_order_acquire) >= sequence) return;
		const auto start = std::chrono::steady_clock::now();
		const auto note  = [&] {
			const auto ns = (std::chrono::steady_clock::now() - start).count();
			LiveCensus::g_upload_wait_ns.fetch_add(ns, std::memory_order_relaxed);
			LiveCounters::Add(LiveCounters::UploadWaitUs, static_cast<uint64_t>(ns) / 1000u);
		};
		if (serve && g_serve_hook != nullptr && g_serve_hook(false)) {
			Kick();
			while (m_done.load(std::memory_order_acquire) < sequence)
				if (!g_serve_hook(true)) _mm_pause();
			return note();
		}
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

	// Render thread (Drain): as Wait, but the jobs up to `sequence` that the worker is not at (asleep, or not yet given a
	// CPU) are run here, in queue order: on the Steam Deck's four cores the render thread waited ~7 ms a frame for jobs
	// the worker ran in ~3.6 ms. Only write protections and copies: a call of another kind (a read through the guest's
	// mapping, which may fault into a readback) is left to the worker, and waited for.
	void Help(uint64_t sequence) {
		if (m_done.load(std::memory_order_acquire) >= sequence) return;
		const auto start = std::chrono::steady_clock::now();
		// The jobs at the worker's priority, not the render thread's above the game's threads: held to four CPUs, the game's
		// threads starved behind them, frames of 100-500 ms with the render thread idle (1-1, 1% lows 25-45 -> 7-13 fps;
		// at normal priority the lows as without the help, +4% fps). (The Steam Deck build keeps the render thread's core
		// for it alone, KYTY_RENDER_CORE: nothing of the game's runs there.)
		std::optional<LocalPlatform::NormalPriority> normal;
		for (int spin = 0; m_done.load(std::memory_order_acquire) < sequence;) {
			if (m_process.try_lock()) {
#if !defined(KYTY_STEAM_DECK)
				if (!normal) normal.emplace();
#endif
				const uint64_t done = m_done.load(std::memory_order_relaxed);
				const uint64_t head = m_head.load(std::memory_order_acquire);
				const uint64_t ran  = done < head ? Process(done, head, true) : done;
				m_process.unlock();
				if (ran == done) break; // (the next job is the worker's)
				continue;
			}
			if (++spin >= 4000) break; // (the worker is at them)
			_mm_pause();
		}
		normal.reset();
		const auto helped = (std::chrono::steady_clock::now() - start).count();
		if (m_done.load(std::memory_order_acquire) < sequence) {
			Kick();
			Wait(sequence);
		}
		LiveCounters::Add(LiveCounters::UploadHelpUs, static_cast<uint64_t>(helped) / 1000u);
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
		t_worker = true;
		WriteGuard::t_no_wait = true; // (KYTY_WRITE_GUARD: the lifts are this thread's jobs)
		// Created by the render thread: do not inherit its dedicated CPU.
		LocalPlatform::PinThreadToCpuList(std::getenv("KYTY_RECORDING_CPUS"));
		// Pause iterations the worker polls for new copies before it sleeps (while its CPUs idle: spin-gate.h).
		constexpr uint32_t UploadSpins = 20000;
		for (;;) {
			uint64_t       done  = m_done.load(std::memory_order_acquire);
			uint64_t       head  = m_head.load(std::memory_order_acquire);
			const uint32_t spins = head == done && SpinGate::Open(false) ? UploadSpins : 0;
			for (uint32_t spin = 0; head == done && spin < spins; ++spin) {
				_mm_pause();
				head = m_head.load(std::memory_order_acquire);
				done = m_done.load(std::memory_order_acquire);
			}
			if (head == done) {
				LiveCounters::Add(LiveCounters::UploadSleeps);
				m_sleeping.store(true, std::memory_order_seq_cst);
				head = m_head.load(std::memory_order_seq_cst);
				if (head == done) {
					if (m_stop.load(std::memory_order_seq_cst)) return;
					m_head.wait(done, std::memory_order_seq_cst);
				}
				m_sleeping.store(false, std::memory_order_relaxed);
				continue;
			}
			// (The render thread may be running jobs itself: Help.)
			std::lock_guard process(m_process);
			done = m_done.load(std::memory_order_relaxed);
			head = m_head.load(std::memory_order_acquire);
			if (done < head) Process(done, head, false);
		}
	}

	// Jobs [done, head) in queue order (the caller holds m_process), m_done published as they finish; `helper`: stops at a
	// call other than a write protection. Returns the jobs' new end.
	uint64_t Process(uint64_t done, uint64_t head, bool helper) {
		const auto start = std::chrono::steady_clock::now();
		const auto since = [](std::chrono::steady_clock::time_point from) {
			return static_cast<uint64_t>((std::chrono::steady_clock::now() - from).count()) / 1000u;
		};
		const auto     call  = [](const Job& job) { CallJob(job); };
		// (KYTY_UPLOAD_LANES, read once a batch. Only the worker splits: a helping render thread would wait on the second
		// thread, the stall Help exists to avoid, and the second thread takes its creator's priority and CPUs.)
		const uint32_t lanes = helper ? 1u : kyty_local_upload_lanes.load(std::memory_order_relaxed);
		while (done < head) {
			if (const Job& job = m_jobs[done % Capacity]; job.call != nullptr && !job.protection) {
				if (helper) break;
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
			Protections(done, end, lanes);
			LiveCounters::Add(LiveCounters::UploadCallUs, since(protecting));
			CopyRun(done, end, bytes);
			done = end;
		}
		if (!helper) LiveCounters::Add(LiveCounters::UploadBusyUs, since(start));
		m_done.store(done, std::memory_order_seq_cst);
		if (m_waiters.load(std::memory_order_seq_cst) != 0) m_done.notify_all();
		return done;
	}

	static void CallJob(const Job& job) { job.call(job.context, reinterpret_cast<uint64_t>(job.destination), job.size); }

	// KYTY_UPLOAD_LANES: the jobs each lane needs at least for a run to be split: the second thread's wake-up (~3-4 us)
	// plus the waker's FUTEX_WAKE (~2-3 us) against ~3 us a protection; lane 1 runs late by the wake-up, so it needs more.
	static constexpr size_t LaneMinJobs0 = 2;
	static constexpr size_t LaneMinJobs1 = 4;

	// The write protections among jobs [first, last), in queue order (lanes 1), or split by region over two lanes (2, 3 =
	// with the order checks) when no job spans regions and both lanes get LaneMinJobs. (The caller holds m_process.)
	void Protections(uint64_t first, uint64_t last, uint32_t lanes) {
		if (lanes >= 2) {
			m_lane_jobs[0].clear();
			m_lane_jobs[1].clear();
			bool serial = false;
			for (uint64_t i = first; i < last && !serial; ++i) {
				const Job& job = m_jobs[i % Capacity];
				if (job.call == nullptr) continue;
				const uint64_t region = job.guest / LaneRegion;
				if (region != (job.guest + std::max<uint64_t>(job.guest_size, 1) - 1) / LaneRegion) serial = true;
				else m_lane_jobs[region & 1u].push_back(i);
			}
			// (Only while the process's CPUs keep idle ones, the spin gate open: else the second thread waits for a CPU.)
			if (!serial && m_lane_jobs[0].size() >= LaneMinJobs0 && m_lane_jobs[1].size() >= LaneMinJobs1 &&
			    SpinGate::g_open.load(std::memory_order_relaxed)) {
				RunLanes(lanes == 3, first, last);
				return;
			}
		}
		for (uint64_t i = first; i < last; ++i)
			if (const Job& next = m_jobs[i % Capacity]; next.call != nullptr) CallJob(next);
	}

	// Lane 1's jobs to the second thread, lane 0's here, then waits for the second (a short spin: the lanes are about even;
	// then a futex), so the run's copies start after both. (The caller holds m_process: one run in lanes at a time.)
	void RunLanes(bool verify, uint64_t first, uint64_t last) {
		if (!m_lane_thread.joinable()) {
			m_lane_thread = std::thread([this] { LaneRun(); });
			std::printf("Upload lanes: on (KYTY_UPLOAD_LANES=%u: a second upload thread runs the odd 4 MiB regions' write protections; "
			            "it sleeps between runs)\n",
			            verify ? 3u : 2u);
			std::fflush(stdout);
		}
		m_lane_verify.store(verify, std::memory_order_relaxed);
		const uint32_t generation = m_lane_posted.load(std::memory_order_relaxed) + 1;
		m_lane_posted.store(generation, std::memory_order_seq_cst);
		m_lane_posted.notify_one();
		for (const uint64_t i: m_lane_jobs[0]) CallJob(m_jobs[i % Capacity]);
		// (The join, after this lane's own jobs: its wait is the split's cost, upload_lane_join_us.)
		const auto joining = std::chrono::steady_clock::now();
		for (int spin = 0; spin < 2000 && m_lane_finished.load(std::memory_order_acquire) != generation; ++spin) _mm_pause();
		if (m_lane_finished.load(std::memory_order_seq_cst) != generation) {
			LiveCounters::Add(LiveCounters::UploadLaneFutexJoins);
			m_lane_main_waiting.store(true, std::memory_order_seq_cst);
			for (uint32_t finished; (finished = m_lane_finished.load(std::memory_order_seq_cst)) != generation;)
				m_lane_finished.wait(finished, std::memory_order_seq_cst);
			m_lane_main_waiting.store(false, std::memory_order_relaxed);
		}
		LiveCounters::Add(LiveCounters::UploadLaneJoinUs,
		                  static_cast<uint64_t>((std::chrono::steady_clock::now() - joining).count()) / 1000u);
		LiveCounters::Add(LiveCounters::UploadLaneRuns);
		LiveCounters::Add(LiveCounters::UploadLaneJobs, m_lane_jobs[1].size());
		if (verify) LiveCounters::Add(LiveCounters::UploadLaneMismatch, CheckLanes(first, last));
	}

	// The second lane's thread: each posted run's lane-1 jobs in their order, then sleeps again (never spins: a spinning
	// thread in the process makes every protection's TLB flush reach one more CPU, ~0.27 us each).
	void LaneRun() {
		LocalPlatform::SetThreadName("Kyty.Upload2");
		t_worker = true;
		LocalPlatform::PinThreadToCpuList(std::getenv("KYTY_RECORDING_CPUS"));
		uint32_t seen = 0;
		for (;;) {
			uint32_t posted = 0;
			while ((posted = m_lane_posted.load(std::memory_order_acquire)) == seen) {
				if (m_stop.load(std::memory_order_seq_cst)) return;
				m_lane_posted.wait(seen, std::memory_order_acquire);
			}
			seen = posted;
			// (A post runs and finishes even when stopping: its poster waits for it; the destructor's wake carries no jobs
			// of its own: the worker has stopped by then.)
			const bool verify = m_lane_verify.load(std::memory_order_relaxed);
			if (verify) m_lane_ran.clear();
			for (const uint64_t i: m_lane_jobs[1]) {
				if (verify) m_lane_ran.push_back(i);
				CallJob(m_jobs[i % Capacity]);
			}
			m_lane_finished.store(seen, std::memory_order_seq_cst);
			if (m_lane_main_waiting.load(std::memory_order_seq_cst)) m_lane_finished.notify_one();
			if (m_stop.load(std::memory_order_seq_cst)) return;
		}
	}

	// KYTY_UPLOAD_LANES=3, after both lanes are done: the order rules a split must keep, the violations counted. Each lane holds
	// only its regions (one region never in both: its protections keep their queue order), in queue order; the second thread
	// ran exactly lane 1's jobs, in that order.
	[[nodiscard]] uint64_t CheckLanes(uint64_t first, uint64_t last) const {
		uint64_t mismatches = 0;
		// (Every protection job of the run in one of the lanes, each single-region.)
		size_t protections = 0;
		for (uint64_t i = first; i < last; ++i) {
			const Job& job = m_jobs[i % Capacity];
			if (job.call == nullptr) continue;
			protections++;
			if (job.guest / LaneRegion != (job.guest + std::max<uint64_t>(job.guest_size, 1) - 1) / LaneRegion) mismatches++;
		}
		if (protections != m_lane_jobs[0].size() + m_lane_jobs[1].size()) mismatches++;
		for (uint32_t lane = 0; lane < 2; ++lane) {
			for (size_t k = 0; k < m_lane_jobs[lane].size(); ++k) {
				const Job& job = m_jobs[m_lane_jobs[lane][k] % Capacity];
				if (((job.guest / LaneRegion) & 1u) != lane) mismatches++;
				if (k != 0 && m_lane_jobs[lane][k] <= m_lane_jobs[lane][k - 1]) mismatches++;
			}
		}
		for (const uint64_t a: m_lane_jobs[0])
			for (const uint64_t b: m_lane_jobs[1])
				if (m_jobs[a % Capacity].guest / LaneRegion == m_jobs[b % Capacity].guest / LaneRegion) mismatches++;
		if (m_lane_ran != m_lane_jobs[1]) mismatches++;
		return mismatches;
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
	// KYTY_UPLOAD_LANES: the run's jobs per lane (written under m_process before a post, read by the second thread until it
	// finishes that post), the second thread, its post / finish generations, and whether the poster sleeps on the finish.
	std::vector<uint64_t>              m_lane_jobs[2];
	std::vector<uint64_t>              m_lane_ran; // (=3: the second thread's)
	std::thread                        m_lane_thread;
	alignas(64) std::atomic<uint32_t>  m_lane_posted {0};
	alignas(64) std::atomic<uint32_t>  m_lane_finished {0};
	std::atomic<bool>                  m_lane_main_waiting {false};
	std::atomic<bool>                  m_lane_verify {false};
	std::unique_ptr<Job[]>             m_jobs;
	alignas(64) std::atomic<uint64_t>  m_head {0};
	alignas(64) std::atomic<uint64_t>  m_done {0};
	alignas(64) std::atomic<bool>      m_sleeping {false};
	std::atomic<uint32_t>              m_waiters {0};
	std::atomic<bool>                  m_stop {false};
	std::mutex                         m_process; // held by whichever thread runs jobs (the worker, or Help)
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

// Render thread: every recorded copy reaches staging memory before this returns; the jobs the worker is not at made
// here where kyty_local_upload_help says (above). `serve`: the guest commands queued for the GPU thread run meanwhile
// (KYTY_SERVE_IN_WAITS); only from a drain that no command and no submit is inside.
inline void Drain(bool serve = false) {
	if (const auto sequence = SubmitSequence(); sequence != 0) {
		const auto help = kyty_local_upload_help.load(std::memory_order_relaxed);
		if (help == 1 || (help == 2 && !SpinGate::g_open.load(std::memory_order_relaxed)))
			Get().Help(sequence);
		else
			Get().Wait(sequence, serve);
	}
}

// Render thread: every recorded job on guest [vaddr, vaddr + size) is done before this returns.
inline void DrainRange(uint64_t vaddr, uint64_t size) {
	if (g_used.load(std::memory_order_relaxed)) Wait(Get().PendingOn(vaddr, size));
}

} // namespace AsyncUpload
