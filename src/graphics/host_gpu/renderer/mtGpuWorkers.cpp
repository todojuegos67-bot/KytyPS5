#include "graphics/host_gpu/renderer/mtGpuWorkers.h"

#include "common/cpuAffinity.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/renderer/mtGpuStageA.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"

#include <algorithm>
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
// (a speculated draw holds its walks and bindings, some KB).
constexpr size_t MaxChunks = 64;
constexpr size_t MaxDraws  = 16384;

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

struct SpecChunk {
	uint64_t                               seq     = 0;
	uint64_t                               address = 0;
	std::vector<std::unique_ptr<SpecDraw>> draws;
	// 0 waiting, 1 done (the workers' results are complete).
	std::atomic<uint32_t> done {0};
	std::atomic<bool>     cancel {false};
	// GPU thread only.
	bool        reached = false;
	bool        dropped = false;   // not done when reached: none of it is used
	const char* reject  = nullptr; // the first reason a draw of it was not taken
	size_t      cursor  = 0;
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
	size_t                                  pending_draws = 0;
	// Chunks the GPU thread is done with, freed by a worker (not on the GPU thread).
	std::vector<std::shared_ptr<SpecChunk>> graveyard;
	bool                                    stop = false;
	std::vector<std::thread>                threads;
};
// Never destroyed: a worker may still run at exit when the scanner is not torn down.
Pool& g_pool = *new Pool;

// The scanner's open chunk (scanner thread only).
std::shared_ptr<SpecChunk> g_open;
// The GPU thread's current chunk.
std::shared_ptr<SpecChunk> g_current;

// Counters since the last report.
struct Stats {
	std::mutex                        mutex;
	uint64_t                          chunks_made    = 0; // given to the workers
	uint64_t                          chunks_dropped = 0; // queue full
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

void Publish(std::shared_ptr<SpecChunk> chunk) {
	{
		std::scoped_lock lock(g_pool.mutex);
		if (g_pool.stop || g_pool.pending.size() >= MaxChunks ||
		    g_pool.pending_draws + chunk->draws.size() > MaxDraws) {
			std::scoped_lock stats(g_stats.mutex);
			g_stats.chunks_dropped++;
			return;
		}
		g_pool.pending_draws += chunk->draws.size();
		g_pool.pending.push_back(chunk);
		g_pool.work.push_back(std::move(chunk));
	}
	{
		std::scoped_lock stats(g_stats.mutex);
		g_stats.chunks_made++;
	}
	g_pool.ready.notify_one();
}

// Worker: the chunk's draws in order, until the GPU thread reaches or passes them.
void RunChunk(RenderContext& renderer, SpecChunk& chunk) {
	auto& executor = renderer.GetRenderExecutor();
	auto& pipeline = renderer.GetPipelineCache();
	for (auto& draw: chunk.draws) {
		auto&      d      = *draw;
		const auto passed = [&] {
			return chunk.cancel.load(std::memory_order_relaxed) ||
			       !DrawPrep::GpuProgress(d.key.queue).Before(d.key);
		};
		if (passed()) {
			d.failure = "hilo GPU por delante";
			break;
		}
		// The descriptor walk outside the locks: only bytes no GPU work wrote, recorded.
		if (!pipeline.DrawPrepWalk(d.inputs, d.walked)) {
			d.failure = "recorrido del SRT";
			NoteWorkerDraw(d.failure, 0, 0);
			continue;
		}
		uint32_t retries = 0;
		bool     done    = false;
		while (!passed()) {
			const auto begin = NowNs();
			if (executor.SpeculateDraw(d)) {
				NoteWorkerDraw(d.failure, NowNs() - begin, retries);
				done = true;
				break;
			}
			retries++;
			std::this_thread::sleep_for(std::chrono::microseconds(20));
		}
		if (!done) {
			d.failure = "hilo GPU por delante";
			break;
		}
	}
	chunk.done.store(1, std::memory_order_release);
}

void WorkerRun() {
	KYTY_PROFILER_THREAD("Thread_MtGpuWorker");
	Common::RegisterGpuSideThread("Thread_MtGpuWorker");
	// The walks' clean-backing reads, as the draw-prep scanner's workers make them.
	LibKernel::Memory::SetGpuReadDelegate(true);
	SetOnWorker();
	for (;;) {
		std::shared_ptr<SpecChunk>              chunk;
		std::vector<std::shared_ptr<SpecChunk>> dead;
		{
			std::unique_lock lock(g_pool.mutex);
			g_pool.ready.wait(lock, [] {
				return g_pool.stop || !g_pool.work.empty() || !g_pool.graveyard.empty();
			});
			if (g_pool.stop) {
				return;
			}
			dead.swap(g_pool.graveyard);
			if (!g_pool.work.empty()) {
				chunk = std::move(g_pool.work.front());
				g_pool.work.pop_front();
			}
		}
		dead.clear();
		if (chunk != nullptr) {
			RunChunk(*g_pool.renderer, *chunk);
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
		    "MT GPU stage d: {} translation workers ahead of the GPU thread (KYTY_MT_GPU={}{})\n",
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
	g_open          = std::make_shared<SpecChunk>();
	g_open->seq     = seq;
	g_open->address = address;
}

void SpecAddDraw(const DrawPrep::Key& key, const PipelineCache::DrawPrepInputs& inputs) {
	if (g_open == nullptr || key.queue != 0 || key.seq != g_open->seq) {
		return;
	}
	auto draw    = std::make_unique<SpecDraw>();
	draw->key    = key;
	draw->inputs = inputs;
	g_open->draws.push_back(std::move(draw));
}

void SpecChunkEnd() {
	if (g_open == nullptr) {
		return;
	}
	auto chunk = std::move(g_open);
	g_open     = nullptr;
	if (!chunk->draws.empty()) {
		Publish(std::move(chunk));
	}
}

void StartWorkers(RenderContext& renderer) {
	if (!WorkersEnabled() || !g_pool.threads.empty()) {
		return;
	}
	g_pool.renderer = &renderer;
	for (uint32_t i = 0; i < ThreadCount(); i++) {
		g_pool.threads.emplace_back(WorkerRun);
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

SpecDraw* TakeSpecDraw(const DrawPrep::Key* key, const char** reason) {
	*reason = nullptr;
	if (key == nullptr || key->queue != 0) {
		return nullptr;
	}
	const auto first_of = [](const SpecChunk& c) -> const DrawPrep::Key& {
		return c.draws.front()->key;
	};
	const auto last_of = [](const SpecChunk& c) -> const DrawPrep::Key& {
		return c.draws.back()->key;
	};
	for (;;) {
		if (g_current != nullptr) {
			if (last_of(*g_current).Before(*key)) {
				// Passed: whatever the workers still do for it is not wanted.
				g_current->cancel.store(true, std::memory_order_relaxed);
				CountChunkEnd(*g_current);
				{
					std::scoped_lock lock(g_pool.mutex);
					g_pool.graveyard.push_back(std::move(g_current));
				}
				g_current = nullptr;
				g_pool.ready.notify_one();
				continue;
			}
			break;
		}
		std::scoped_lock lock(g_pool.mutex);
		if (g_pool.pending.empty() || key->Before(first_of(*g_pool.pending.front()))) {
			return nullptr; // nothing for this draw (yet)
		}
		g_current = std::move(g_pool.pending.front());
		g_pool.pending.pop_front();
		g_pool.pending_draws -= g_current->draws.size();
	}
	auto& chunk = *g_current;
	if (key->Before(first_of(chunk))) {
		return nullptr;
	}
	if (!chunk.reached) {
		chunk.reached = true;
		if (chunk.done.load(std::memory_order_acquire) == 0) {
			// Not finished: dropped, never waited for.
			chunk.cancel.store(true, std::memory_order_relaxed);
			chunk.dropped = true;
			chunk.reject  = "trabajador sin terminar";
		}
	}
	while (chunk.cursor < chunk.draws.size() && chunk.draws[chunk.cursor]->key.Before(*key)) {
		chunk.cursor++;
	}
	if (chunk.cursor >= chunk.draws.size() || !(chunk.draws[chunk.cursor]->key == *key)) {
		return nullptr; // a draw the scanner did not give the workers
	}
	auto& draw = *chunk.draws[chunk.cursor];
	if (draw.consumed) {
		return nullptr; // the packet run again (suspended): its translation went to the first run
	}
	draw.consumed = true;
	if (chunk.dropped) {
		*reason = "trabajador sin terminar";
	} else if (!draw.translated) {
		*reason = draw.failure != nullptr ? draw.failure : "sin traducir";
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
	Log::WriteUnmuted(fmt::format(
	    "MT GPU trabajadores ({:.0f} s, KYTY_MT_GPU={}): trozos especulados {} (descartados por "
	    "cola llena {}), alcanzados {}, aceptados {}, rechazados por: {}; sin alcanzar {}. Draws: "
	    "traducidos por trabajadores {} de {} (fallos: {}), mirados por el hilo GPU {}, emitidos "
	    "desde trabajador {}{}, rechazados por: {}. Hilo GPU: traduccion normal {:.1f} us/draw, "
	    "consumo {:.1f} ms, ahorro estimado {:.1f} ms{}; locks en manos de trabajadores {:.1f} ms "
	    "({} reintentos); alertas {}\n",
	    seconds, Mode(), s.chunks_made, s.chunks_dropped, s.chunks_seen, s.chunks_ok,
	    list(s.chunk_rejects), s.chunks_missed, s.worker_ok, s.worker_draws,
	    list(s.worker_failures), s.taken, s.emitted,
	    WorkersCompare() ? fmt::format(" (=4: validados {}, diferencias {})", s.validated, s.differ)
	                     : std::string(),
	    list(s.draw_rejects), normal_us, static_cast<double>(s.consume_ns) / 1e6, saved_ms,
	    WorkersCompare() ? " (si se usaran)" : "", static_cast<double>(s.worker_held_ns) / 1e6,
	    s.lock_retries, WorkerAlerts()));
	g_stats.chunks_made = g_stats.chunks_dropped = g_stats.chunks_seen = g_stats.chunks_ok = 0;
	g_stats.chunks_missed = 0;
	g_stats.chunk_rejects.clear();
	g_stats.draw_rejects.clear();
	g_stats.worker_failures.clear();
	g_stats.worker_draws = g_stats.worker_ok = g_stats.worker_held_ns = g_stats.lock_retries = 0;
	g_stats.taken = g_stats.emitted = g_stats.validated = g_stats.differ = 0;
	g_stats.consume_ns = g_stats.normal_draws = g_stats.normal_ns = 0;
}

} // namespace Libs::Graphics::MtGpu
