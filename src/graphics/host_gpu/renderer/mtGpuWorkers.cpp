#include "graphics/host_gpu/renderer/mtGpuWorkers.h"

#include "common/cpuAffinity.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/renderer/mtGpuStageA.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <fmt/format.h>

namespace Libs::Graphics::MtGpu {

namespace {

// Chunks and draws waiting for the GPU thread, at most: past them the scanner drops new chunks
// and draws (a speculated draw holds its walks and bindings, some KB).
constexpr size_t MaxChunks = 64;
constexpr size_t MaxDraws  = 16384;
// Draws a chunk holds at most: past them the scanner goes on in a new chunk (the slots are
// fixed, so the workers read the chunk while the scanner appends to it).
constexpr uint32_t ChunkCapacity = 512;

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// A chunk goes to the workers and the GPU thread when the scanner opens it (each chained
// INDIRECT_BUFFER of a submission, as the submission is enqueued): its draws are handed out one
// by one as the scanner prepares them, not when the chunk is complete. Any worker takes the
// next draw not taken yet (draws of one chunk run on several workers at once); the GPU thread
// uses each draw a worker has finished, and passes over the others (never waits).
struct SpecChunk {
	uint64_t seq     = 0;
	uint64_t address = 0;
	// Written by the scanner at [count] before count is raised (release); fixed slots.
	std::array<std::unique_ptr<SpecDraw>, ChunkCapacity> draws;
	std::atomic<uint32_t> count {0};
	std::atomic<bool>     closed {false}; // no more draws (count is final)
	std::atomic<bool>     cancel {false}; // the GPU thread passed it
	// Under g_pool.mutex: the next draw a worker takes.
	uint32_t next = 0;
	// When a worker took its first draw (NowNs), 0: not yet (the lead over the GPU thread).
	std::atomic<uint64_t> start_ns {0};
	// GPU thread only.
	bool        reached = false;
	const char* reject  = nullptr; // the first reason a draw of it was not taken
	uint32_t    cursor  = 0;

	[[nodiscard]] uint32_t Published() const { return count.load(std::memory_order_acquire); }
};

uint32_t ThreadCount() {
	static const uint32_t count = [] {
		const char* value = std::getenv("KYTY_MT_GPU_THREADS");
		long        n     = value != nullptr ? std::strtol(value, nullptr, 10) : 3l;
		return static_cast<uint32_t>(std::clamp(n, 1l, 8l));
	}();
	return count;
}

// The pool and the chunks.
struct Pool {
	RenderContext*                          renderer = nullptr;
	std::mutex                              mutex;
	std::condition_variable                 ready;
	std::deque<std::shared_ptr<SpecChunk>>  work;    // for the workers, in order
	std::deque<std::shared_ptr<SpecChunk>>  pending; // for the GPU thread, in order
	std::atomic<size_t>                     pending_draws {0};
	// Chunks the GPU thread is done with, freed by a worker (not on the GPU thread).
	std::vector<std::shared_ptr<SpecChunk>> graveyard;
	uint32_t                                idle = 0; // workers waiting on `ready`
	bool                                    stop = false;
	std::vector<std::thread>                threads;
};
// Never destroyed: a worker may still run at exit when the scanner is not torn down.
Pool& g_pool = *new Pool;

// The scanner's open chunk (scanner thread only).
std::shared_ptr<SpecChunk> g_open;
// The GPU thread's current chunk.
std::shared_ptr<SpecChunk> g_current;

constexpr uint32_t MaxWorkers = 8;

// Where each worker's time goes, since the last report (its own slot: no lock).
struct WorkerTime {
	std::atomic<uint64_t> wait_ns {0};  // a lock not free: the attempt and the sleep after it
	std::atomic<uint64_t> work_ns {0};  // walking and translating
	std::atomic<uint64_t> idle_ns {0};  // nothing to do (waiting for a draw)
};
std::array<WorkerTime, MaxWorkers>& g_worker_time = *new std::array<WorkerTime, MaxWorkers>;

// Counters since the last report.
struct Stats {
	std::mutex                        mutex;
	uint64_t                          chunks_made    = 0; // given to the workers
	uint64_t                          chunks_dropped = 0; // queue full
	uint64_t                          draws_dropped  = 0; // too many draws waiting
	uint64_t                          chunks_seen    = 0; // reached by the GPU thread
	uint64_t                          chunks_ok      = 0;
	uint64_t                          chunks_missed  = 0; // passed without a draw taken
	std::map<std::string, uint64_t>   chunk_rejects;
	std::map<std::string, uint64_t>   draw_rejects;
	std::map<std::string, uint64_t>   worker_failures;
	uint64_t                          worker_draws   = 0;
	uint64_t                          worker_ok      = 0;
	uint64_t                          worker_held_ns = 0;
	uint64_t                          lock_retries   = 0;
	uint64_t                          taken          = 0; // draws looked at with a translation
	uint64_t                          emitted        = 0;
	uint64_t                          validated      = 0; // =4: would have been used
	uint64_t                          differ         = 0;
	uint64_t                          consume_ns     = 0;
	uint64_t                          normal_draws   = 0;
	uint64_t                          normal_ns      = 0;
	// The lead: how long before the GPU thread reached a chunk a worker had started it.
	uint64_t                          lead_ns        = 0;
	uint64_t                          lead_chunks    = 0; // started before the GPU thread
	uint64_t                          lead_late      = 0; // not started when reached
	// Per draw: how long before the GPU thread reached it a worker had finished it.
	uint64_t                          draw_lead_ns   = 0;
	uint64_t                          draw_lead_n    = 0;
	// "imagen sin la misma memoria", by what the worker found there (NoteImageMiss).
	std::map<std::string, uint64_t>   image_misses;
	// The GPU thread's FindImage while the workers run (NoteGpuImageLookup).
	std::map<const char*, uint64_t>   gpu_lookups; // by literal (one per outcome)
};
Stats& g_stats = *new Stats;

void CountChunkEnd(const SpecChunk& chunk) {
	std::scoped_lock lock(g_stats.mutex);
	if (!chunk.reached) {
		g_stats.chunks_missed++;
		return;
	}
	g_stats.chunks_seen++;
	if (chunk.reject == nullptr) {
		g_stats.chunks_ok++;
	} else {
		g_stats.chunk_rejects[chunk.reject]++;
	}
}

// Scanner: the chunk goes to the workers and the GPU thread now, empty; its draws follow.
bool Open(const std::shared_ptr<SpecChunk>& chunk) {
	{
		std::scoped_lock lock(g_pool.mutex);
		if (!g_pool.stop && g_pool.pending.size() < MaxChunks) {
			g_pool.pending.push_back(chunk);
			g_pool.work.push_back(chunk);
			std::scoped_lock stats(g_stats.mutex);
			g_stats.chunks_made++;
			return true;
		}
	}
	std::scoped_lock stats(g_stats.mutex);
	g_stats.chunks_dropped++;
	return false;
}

// Scanner: the chunk's draws are final.
void Close(SpecChunk& chunk) {
	{
		std::scoped_lock lock(g_pool.mutex);
		chunk.closed.store(true, std::memory_order_release);
	}
	g_pool.ready.notify_all(); // a worker may wait for this chunk's end to go to the next
}

// Worker, under g_pool.mutex: the next draw to translate, from the oldest chunk with one; the
// chunks done with (cancelled, or closed with every draw taken) leave the list.
std::shared_ptr<SpecChunk> NextDraw(SpecDraw** draw) {
	for (auto it = g_pool.work.begin(); it != g_pool.work.end();) {
		auto&      chunk = **it;
		const auto count = chunk.Published();
		if (chunk.cancel.load(std::memory_order_relaxed) ||
		    (chunk.next >= count && chunk.closed.load(std::memory_order_acquire) &&
		     chunk.next >= chunk.Published())) {
			it = g_pool.work.erase(it);
			continue;
		}
		if (chunk.next < count) {
			*draw = chunk.draws[chunk.next++].get();
			return *it;
		}
		++it;
	}
	return nullptr;
}

// Worker: one draw, unless the GPU thread got there first.
void RunDraw(RenderContext& renderer, SpecChunk& chunk, SpecDraw& d, WorkerTime& time) {
	uint32_t expected = SpecDraw::Free;
	if (!d.state.compare_exchange_strong(expected, SpecDraw::Running, std::memory_order_acq_rel)) {
		return; // the GPU thread reached it first: no longer wanted
	}
	auto&      executor = renderer.GetRenderExecutor();
	auto&      pipeline = renderer.GetPipelineCache();
	const auto passed   = [&] {
		return chunk.cancel.load(std::memory_order_relaxed) ||
		       !DrawPrep::GpuProgress(d.key.queue).Before(d.key);
	};
	const auto finish = [&] {
		d.done_ns = NowNs();
		d.state.store(SpecDraw::Done, std::memory_order_release);
	};
	if (passed()) {
		d.failure = "hilo GPU por delante";
		finish();
		return;
	}
	uint64_t started = 0;
	(void)chunk.start_ns.compare_exchange_strong(started, NowNs(), std::memory_order_relaxed);
	// The descriptor walk outside the locks: only bytes no GPU work wrote, recorded.
	const auto walk_begin = NowNs();
	const bool walked     = pipeline.DrawPrepWalk(d.inputs, d.walked);
	time.work_ns.fetch_add(NowNs() - walk_begin, std::memory_order_relaxed);
	if (!walked) {
		d.failure = "recorrido del SRT";
		NoteWorkerDraw(d.failure, 0, 0);
		finish();
		return;
	}
	uint32_t retries = 0;
	while (!passed()) {
		const auto begin = NowNs();
		if (executor.SpeculateDraw(d)) {
			const auto held = NowNs() - begin;
			time.work_ns.fetch_add(held, std::memory_order_relaxed);
			NoteWorkerDraw(d.failure, held, retries);
			finish();
			return;
		}
		retries++;
		std::this_thread::sleep_for(std::chrono::microseconds(20));
		time.wait_ns.fetch_add(NowNs() - begin, std::memory_order_relaxed);
	}
	d.translated = false;
	d.failure    = "hilo GPU por delante";
	finish();
}

void WorkerRun(uint32_t index) {
	auto& time = g_worker_time[index % MaxWorkers];
	KYTY_PROFILER_THREAD("Thread_MtGpuWorker");
	Common::RegisterGpuSideThread("Thread_MtGpuWorker");
	// The walks' clean-backing reads, as the draw-prep scanner's workers make them.
	LibKernel::Memory::SetGpuReadDelegate(true);
	SetOnWorker();
	for (;;) {
		std::shared_ptr<SpecChunk>              chunk;
		SpecDraw*                               draw = nullptr;
		std::vector<std::shared_ptr<SpecChunk>> dead;
		{
			std::unique_lock lock(g_pool.mutex);
			for (;;) {
				if (g_pool.stop) {
					return;
				}
				dead.swap(g_pool.graveyard);
				chunk = NextDraw(&draw);
				if (chunk != nullptr || !dead.empty()) {
					break;
				}
				const auto idle_begin = NowNs();
				g_pool.idle++;
				g_pool.ready.wait(lock);
				g_pool.idle--;
				time.idle_ns.fetch_add(NowNs() - idle_begin, std::memory_order_relaxed);
			}
		}
		dead.clear();
		if (chunk != nullptr) {
			RunDraw(*g_pool.renderer, *chunk, *draw, time);
		}
	}
}

} // namespace

bool WorkersEnabled() {
	static const bool enabled = [] {
		if (!WorkersMode()) {
			return false;
		}
		if (!DrawPrep::Enabled()) {
			Log::WriteUnmuted("MT GPU: KYTY_MT_GPU=1/4 needs the draw-prep scanner (KYTY_DRAW_PREP): "
			                  "workers off\n");
			return false;
		}
		(void)RegistryEnabled();
		Log::WriteUnmuted(fmt::format(
		    "MT GPU stage d: {} translation workers ahead of the GPU thread (KYTY_MT_GPU={}{}); "
		    "draws handed out as the scanner prepares them\n",
		    ThreadCount(), Mode(), Mode() == 4 ? ", compared with the normal translation" : ""));
		return true;
	}();
	return enabled;
}

bool WorkersCompare() {
	return WorkersEnabled() && Mode() == 4;
}

void SpecChunkBegin(uint64_t seq, uint64_t address) {
	SpecChunkEnd();
	auto chunk     = std::make_shared<SpecChunk>();
	chunk->seq     = seq;
	chunk->address = address;
	if (Open(chunk)) {
		g_open = std::move(chunk);
	}
}

void SpecAddDraw(const DrawPrep::Key& key, const PipelineCache::DrawPrepInputs& inputs) {
	if (g_open == nullptr || key.queue != 0 || key.seq != g_open->seq) {
		return;
	}
	if (g_open->count.load(std::memory_order_relaxed) >= ChunkCapacity) {
		// Full: the rest of the buffer in a new chunk.
		const auto seq     = g_open->seq;
		const auto address = g_open->address;
		SpecChunkBegin(seq, address);
		if (g_open == nullptr) {
			return;
		}
	}
	if (g_pool.pending_draws.load(std::memory_order_relaxed) >= MaxDraws) {
		std::scoped_lock stats(g_stats.mutex);
		g_stats.draws_dropped++;
		return;
	}
	auto&      chunk = *g_open;
	const auto index = chunk.count.load(std::memory_order_relaxed);
	auto       draw  = std::make_unique<SpecDraw>();
	draw->key        = key;
	draw->inputs     = inputs;
	chunk.draws[index] = std::move(draw);
	g_pool.pending_draws.fetch_add(1, std::memory_order_relaxed);
	bool wake = false;
	{
		// Under the pool's mutex: a worker that found nothing waits on `ready` under it.
		std::scoped_lock lock(g_pool.mutex);
		chunk.count.store(index + 1, std::memory_order_release);
		wake = g_pool.idle != 0;
	}
	if (wake) {
		g_pool.ready.notify_one();
	}
}

void SpecChunkEnd() {
	if (g_open == nullptr) {
		return;
	}
	auto chunk = std::move(g_open);
	g_open     = nullptr;
	Close(*chunk);
}

void StartWorkers(RenderContext& renderer) {
	if (!WorkersEnabled() || !g_pool.threads.empty()) {
		return;
	}
	g_pool.renderer = &renderer;
	for (uint32_t i = 0; i < ThreadCount(); i++) {
		g_pool.threads.emplace_back(WorkerRun, i);
	}
}

void StopWorkers() {
	{
		std::scoped_lock lock(g_pool.mutex);
		g_pool.stop = true;
		for (auto& chunk: g_pool.pending) {
			chunk->cancel.store(true, std::memory_order_relaxed);
		}
	}
	g_pool.ready.notify_all();
	for (auto& thread: g_pool.threads) {
		if (thread.joinable()) {
			thread.join();
		}
	}
	g_pool.threads.clear();
}

namespace {

// GPU thread: the current chunk is done with (passed, or ended empty).
void Retire() {
	g_current->cancel.store(true, std::memory_order_relaxed);
	CountChunkEnd(*g_current);
	g_pool.pending_draws.fetch_sub(g_current->Published(), std::memory_order_relaxed);
	{
		std::scoped_lock lock(g_pool.mutex);
		g_pool.graveyard.push_back(std::move(g_current));
	}
	g_current = nullptr;
	g_pool.ready.notify_one();
}

} // namespace

SpecDraw* TakeSpecDraw(const DrawPrep::Key* key, const char** reason) {
	*reason = nullptr;
	if (key == nullptr || key->queue != 0) {
		return nullptr;
	}
	for (;;) {
		if (g_current != nullptr) {
			// Read `closed` first: a closed chunk's count is final.
			const bool closed = g_current->closed.load(std::memory_order_acquire);
			const auto count  = g_current->Published();
			if (closed && (count == 0 || g_current->draws[count - 1]->key.Before(*key))) {
				Retire(); // passed: whatever the workers still do for it is not wanted
				continue;
			}
			break;
		}
		std::scoped_lock lock(g_pool.mutex);
		if (g_pool.pending.empty()) {
			return nullptr; // nothing for this draw (yet)
		}
		auto&      front  = *g_pool.pending.front();
		const bool closed = front.closed.load(std::memory_order_acquire);
		const auto count  = front.Published();
		if (count == 0 && !closed) {
			return nullptr; // opened, no draw prepared yet
		}
		if (count != 0 && key->Before(front.draws[0]->key)) {
			return nullptr;
		}
		g_current = std::move(g_pool.pending.front());
		g_pool.pending.pop_front();
	}
	auto&      chunk = *g_current;
	const auto count = chunk.Published();
	if (count == 0 || key->Before(chunk.draws[0]->key)) {
		return nullptr;
	}
	if (!chunk.reached) {
		chunk.reached      = true;
		const auto started = chunk.start_ns.load(std::memory_order_relaxed);
		const auto now     = NowNs();
		std::scoped_lock lock(g_stats.mutex);
		if (started != 0 && started <= now) {
			g_stats.lead_ns += now - started;
			g_stats.lead_chunks++;
		} else {
			g_stats.lead_late++;
		}
	}
	while (chunk.cursor < count && chunk.draws[chunk.cursor]->key.Before(*key)) {
		chunk.cursor++;
	}
	if (chunk.cursor >= count || !(chunk.draws[chunk.cursor]->key == *key)) {
		return nullptr; // a draw the scanner did not give the workers
	}
	auto& draw = *chunk.draws[chunk.cursor];
	if (draw.consumed) {
		return nullptr; // the packet run again (suspended): its translation went to the first run
	}
	draw.consumed = true;
	// Not finished: passed over, never waited for (a worker not started yet never starts it).
	uint32_t state = SpecDraw::Free;
	if (draw.state.compare_exchange_strong(state, SpecDraw::Abandoned,
	                                       std::memory_order_acq_rel)) {
		*reason = "trabajador sin empezar";
	} else if (state == SpecDraw::Running) {
		*reason = "trabajador sin terminar";
	} else if (state == SpecDraw::Done) {
		const auto now = NowNs();
		if (draw.done_ns <= now) {
			std::scoped_lock lock(g_stats.mutex);
			g_stats.draw_lead_ns += now - draw.done_ns;
			g_stats.draw_lead_n++;
		}
		if (!draw.translated) {
			*reason = draw.failure != nullptr ? draw.failure : "sin traducir";
		}
	}
	if (*reason != nullptr) {
		NoteSpecOutcome(*reason, false, false, 0);
		return nullptr;
	}
	return &draw;
}

void NoteSpecOutcome(const char* reason, bool emitted, bool differs, uint64_t consume_ns) {
	if (reason != nullptr && g_current != nullptr && g_current->reject == nullptr) {
		g_current->reject = reason;
	}
	std::scoped_lock lock(g_stats.mutex);
	g_stats.taken++;
	g_stats.consume_ns += consume_ns;
	if (reason != nullptr) {
		g_stats.draw_rejects[reason]++;
		return;
	}
	if (emitted) {
		g_stats.emitted++;
	} else {
		g_stats.validated++;
		g_stats.differ += differs ? 1u : 0u;
	}
}

void NoteNormalTranslation(uint64_t ns) {
	std::scoped_lock lock(g_stats.mutex);
	g_stats.normal_draws++;
	g_stats.normal_ns += ns;
}

void NoteWorkerDraw(const char* failure, uint64_t held_ns, uint32_t lock_retries) {
	std::scoped_lock lock(g_stats.mutex);
	g_stats.worker_draws++;
	g_stats.worker_held_ns += held_ns;
	g_stats.lock_retries += lock_retries;
	if (failure == nullptr) {
		g_stats.worker_ok++;
	} else {
		g_stats.worker_failures[failure]++;
	}
}

void NoteImageMiss(const char* what) {
	std::scoped_lock lock(g_stats.mutex);
	g_stats.image_misses[what]++;
}

void NoteGpuImageLookup(const char* what) {
	if (!WorkersEnabled()) {
		return;
	}
	std::scoped_lock lock(g_stats.mutex);
	g_stats.gpu_lookups[what]++;
}

void ReportWorkers(double seconds) {
	if (!WorkersEnabled()) {
		return;
	}
	std::scoped_lock lock(g_stats.mutex);
	auto& s    = g_stats;
	const auto list = [](const std::map<std::string, uint64_t>& m) {
		std::string out;
		for (const auto& [why, n]: m) {
			out += fmt::format("{}{} {}", out.empty() ? "" : ", ", why, n);
		}
		return out.empty() ? std::string("-") : out;
	};
	const double normal_us =
	    s.normal_draws == 0 ? 0.0 : static_cast<double>(s.normal_ns) / 1000.0 / s.normal_draws;
	const double used      = static_cast<double>(WorkersCompare() ? s.validated : s.emitted);
	const double saved_ms  = (used * normal_us * 1000.0 - static_cast<double>(s.consume_ns)) / 1e6;
	std::string  times;
	for (uint32_t i = 0; i < std::min(ThreadCount(), MaxWorkers); i++) {
		auto&      t    = g_worker_time[i];
		const auto ms   = [](std::atomic<uint64_t>& v) {
            return static_cast<double>(v.exchange(0, std::memory_order_relaxed)) / 1e6;
		};
		const double wait = ms(t.wait_ns);
		const double work = ms(t.work_ns);
		const double idle = ms(t.idle_ns);
		times += fmt::format("{}#{} esperando locks {:.0f} / traduciendo {:.0f} / inactivo {:.0f}",
		                     times.empty() ? "" : ", ", i, wait, work, idle);
	}
	Log::WriteUnmuted(fmt::format(
	    "MT GPU trabajadores ({:.0f} s, KYTY_MT_GPU={}): trozos especulados {} (descartados por "
	    "cola llena {}, draws descartados {}), alcanzados {}, aceptados {}, rechazados por: {}; sin alcanzar {}. Draws: "
	    "traducidos por trabajadores {} de {} (fallos: {}), mirados por el hilo GPU {}, emitidos "
	    "desde trabajador {}{}, rechazados por: {}. Hilo GPU: traduccion normal {:.1f} us/draw, "
	    "consumo {:.1f} ms, ahorro estimado {:.1f} ms{}; locks en manos de trabajadores {:.1f} ms "
	    "({} reintentos); alertas {}. Tiempo de los trabajadores (ms): {}; ventaja media {:.1f} us "
	    "({} trozos empezados antes de que llegue el hilo GPU, {} sin empezar), por draw terminado "
	    "{:.1f} us ({} draws). Imagen sin la misma "
	    "memoria por: {}. FindImage del hilo GPU: {}\n",
	    seconds, Mode(), s.chunks_made, s.chunks_dropped, s.draws_dropped, s.chunks_seen, s.chunks_ok,
	    list(s.chunk_rejects), s.chunks_missed, s.worker_ok, s.worker_draws,
	    list(s.worker_failures), s.taken, s.emitted,
	    WorkersCompare() ? fmt::format(" (=4: validados {}, diferencias {})", s.validated, s.differ)
	                     : std::string(),
	    list(s.draw_rejects), normal_us, static_cast<double>(s.consume_ns) / 1e6, saved_ms,
	    WorkersCompare() ? " (si se usaran)" : "", static_cast<double>(s.worker_held_ns) / 1e6,
	    s.lock_retries, WorkerAlerts(), times,
	    s.lead_chunks == 0 ? 0.0 : static_cast<double>(s.lead_ns) / 1000.0 / s.lead_chunks,
	    s.lead_chunks, s.lead_late,
	    s.draw_lead_n == 0 ? 0.0 : static_cast<double>(s.draw_lead_ns) / 1000.0 / s.draw_lead_n,
	    s.draw_lead_n, list(s.image_misses), [&] {
		    std::map<std::string, uint64_t> named;
		    for (const auto& [what, n]: s.gpu_lookups) {
			    named[what] += n;
		    }
		    return list(named);
	    }()));
	g_stats.chunks_made = g_stats.chunks_dropped = g_stats.chunks_seen = g_stats.chunks_ok = 0;
	g_stats.chunks_missed = 0;
	g_stats.chunk_rejects.clear();
	g_stats.draw_rejects.clear();
	g_stats.worker_failures.clear();
	g_stats.worker_draws = g_stats.worker_ok = g_stats.worker_held_ns = g_stats.lock_retries = 0;
	g_stats.taken = g_stats.emitted = g_stats.validated = g_stats.differ = 0;
	g_stats.consume_ns = g_stats.normal_draws = g_stats.normal_ns = 0;
	g_stats.lead_ns = g_stats.lead_chunks = g_stats.lead_late = 0;
	g_stats.draw_lead_ns = g_stats.draw_lead_n = g_stats.draws_dropped = 0;
	g_stats.image_misses.clear();
	g_stats.gpu_lookups.clear();
}

} // namespace Libs::Graphics::MtGpu
