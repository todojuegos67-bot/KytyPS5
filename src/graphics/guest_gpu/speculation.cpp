#include "graphics/guest_gpu/speculation.h"

#include "common/assert.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/bdaDirtyRegions.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "async-upload.h"
#include "draw-state-observer.h"
#include "local-platform.h"
#include "mode-toggle.h"
#include "vulkan-recording.h"
#include "write-guard.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

extern "C" {
volatile std::atomic_uint32_t kyty_local_speculate_mode {0};
volatile std::atomic_uint32_t kyty_local_speculate_threads {4};
// Translations going on and results not finished (Translate to Finish). While there are any, a speculation's executor
// may hold what it read of the owners' catalogs (pairs, draw states, image sets) and a result's commands their
// pipelines: the catalog sweeps (NativeXprSweepStates, TableSweepPairs) and EvictIdlePipelines wait for none.
volatile std::atomic_uint32_t kyty_local_speculations_live {0};
// EvictIdlePipelines is destroying pipeline handles: no translation begins until it is done.
volatile std::atomic_uint32_t kyty_local_pipelines_evicting {0};
}

namespace Libs::Graphics {

bool CommitGuardedWrite(void* dst, const void* src, uint32_t size); // (graphicsRun.cpp: KYTY_WRITE_GUARD_SPEC)

// KYTY_SPECULATE_FLUSH_EVERY (live speculate_flush_every N; 8 by default under KYTY_SPECULATE=4, else 0; 0 and 1:
// every commit): a commit's held segment buffers go to the GPU at its end only every Nth commit; the rest wait for the
// next flush (a label's, a guest submission's end, a readback's, a chunk wait's). The GPU thread flushes them before
// it waits for a speculation's thread (which may wait for their tick), and when a commit blocks at a wait hole.
static uint32_t FlushEvery() {
	static auto& every = LiveSwitch::Find("speculate_flush_every", ModeToggle::Default("speculate_flush_every"));
	return static_cast<uint32_t>(std::max<int64_t>(every.load(std::memory_order_relaxed), 0));
}

uint64_t Speculation::NowNs() noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// The packets a refusal makes a hole of: draws, dispatches, end-of-pipe events and waits (what they change besides
// their work is checked when the commit translates them: Speculation::RunHole; a wait not satisfied then blocks it).
static bool IsHole(uint32_t header) {
	switch ((header >> 8u) & 0xffu) {
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_EVENT_WRITE_EOP:
		case Pm4::IT_EVENT_WRITE_EOS:
		case Pm4::IT_RELEASE_MEM:
		case Pm4::IT_WAIT_REG_MEM:
		case Pm4::IT_WAIT_REG_MEM_64: return true;
		// (A wait for a display buffer's flip: the commit waits there before the rendering after it. A super resolution
		// dispatch runs its upscaler there.)
		case Pm4::IT_NOP:
			return KYTY_PM4_R(header) == Pm4::R_RELEASE_MEM || KYTY_PM4_R(header) == Pm4::R_WAIT_FLIP_DONE ||
			       KYTY_PM4_R(header) == Pm4::R_FLIP || KYTY_PM4_R(header) == Pm4::R_SUPER_RESOLUTION;
		case Pm4::IT_GET_LOD_STATS: return true;
		default: return false;
	}
}

// A chunk's translation stopped where the next chunk begins (Result::stop).
static constexpr char ChunkEnd[] = "chunk end";

// "Work wrote what it read", by the write's kind and what logged it (null: not found).
static const char* WroteWhatItRead(const Spec::EffectsLog::Entry* entry) {
	static constexpr std::array<std::array<const char*, 3>, 2> names {
	    {{"commit: GPU work wrote what it read (normal)", "commit: GPU work wrote what it read (hole)",
	      "commit: GPU work wrote what it read (commit)"},
	     {"commit: a processor write wrote what it read (normal)", "commit: a processor write wrote what it read (hole)",
	      "commit: a processor write wrote what it read (commit)"}}};
	if (entry == nullptr) return "commit: work wrote what it read";
	return names[entry->kind == Spec::EffectsLog::Kind::GpuWrite ? 0 : 1][static_cast<size_t>(entry->source)];
}

Speculation::Translator::Translator(RenderContext& renderer, uint32_t number, int interrupt_id)
    : index(number), draw(std::make_unique<RenderExecutor>(renderer)), compute(std::make_unique<RenderExecutor>(renderer)),
      executors {draw.get(), compute.get()}, cp(std::make_unique<CommandProcessor>(renderer, interrupt_id)),
      table_upload(std::make_unique<StreamBuffer>(renderer.GetGraphics(), renderer.GetCommandScheduler(), MemoryUsage::Upload,
                                                  uint64_t {64} << 20u,
                                                  AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress)),
      shader_upload(std::make_unique<StreamBuffer>(renderer.GetGraphics(), renderer.GetCommandScheduler(),
                                                   MemoryUsage::Upload, uint64_t {16} << 20u)) {
	draw->ReadCatalogOf(renderer.DefaultRenderExecutor(), 1 + number);
	compute->ReadCatalogOf(renderer.DefaultComputeRenderExecutor(), 1 + number);
}

Speculation::Speculation(RenderContext& renderer, int interrupt_id, uint32_t base, uint32_t threads)
    : m_renderer(renderer), m_chain(std::make_unique<CommandProcessor>(renderer, interrupt_id)),
      m_state(std::make_unique<CommandProcessor>(renderer, interrupt_id)), m_interrupt_id(interrupt_id) {
	m_state->SetStateOnly(true);
	if (threads == 0)
		threads = kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4
		              ? kyty_local_speculate_threads.load(std::memory_order_relaxed)
		              : 1u;
	// (Packet slots and executor roles: Spec::MaxPacketSlots, RenderExecutor::Roles less the GPU thread's.)
	threads = std::clamp<uint32_t>(threads, 1u, static_cast<uint32_t>(Spec::MaxPacketSlots) - std::min<uint32_t>(base, Spec::MaxPacketSlots - 1));
	for (uint32_t i = 0; i < threads; ++i) m_translators.push_back(std::make_unique<Translator>(renderer, base + i, interrupt_id));
}

void Speculation::Stop() {
	{
		std::lock_guard lock(m_mutex);
		m_quit = true;
	}
	m_ahead_wake.notify_all();
	for (auto& translator: m_translators)
		if (translator->thread.joinable()) translator->thread.join();
	if (m_blocked) Finish(std::move(m_blocked), 0);
	for (;;) {
		std::unique_ptr<Result> result;
		{
			std::lock_guard lock(m_mutex);
			if (m_ahead.empty()) break;
			result = std::move(m_ahead.front());
			m_ahead.pop_front();
		}
		Finish(std::move(result), 0);
	}
}

Speculation::~Speculation() {
	{
		std::lock_guard lock(m_mutex);
		m_quit = true;
	}
	m_ahead_wake.notify_all();
	for (auto& translator: m_translators)
		if (translator->thread.joinable()) translator->thread.join();
	auto& scheduler = m_renderer.GetCommandScheduler();
	if (m_blocked) scheduler.DestroyRecorder(std::move(m_blocked->recorder));
	for (auto& result: m_ahead) scheduler.DestroyRecorder(std::move(result->recorder));
	for (auto& translator: m_translators)
		for (auto& result: translator->free) scheduler.DestroyRecorder(std::move(result->recorder));
}

Speculation::Epochs Speculation::CurrentEpochs() const {
	auto& resources = m_renderer.GetGpuResources();
	auto& textures  = m_renderer.GetTextureCache();
	return {m_renderer.GetBufferCache().RegistrationEpoch(), textures.ResolutionEpoch(), resources.MappingEpoch()};
}

Speculation::Outcome Speculation::TranslateAndCommit(CommandProcessor& cp, uint64_t serial,
                                                     std::span<const uint32_t> commands, Pm4Execution& execution) {
	auto result    = Translate(*m_translators[0], cp, commands, execution);
	result->serial = serial;
	return Commit(std::move(result), cp, execution);
}

Speculation::Outcome Speculation::Resume(CommandProcessor& cp, Pm4Execution& execution) {
	return Continue(std::move(m_blocked), cp, execution);
}

void Speculation::Push(const Job& job) {
	{
		std::lock_guard lock(m_mutex);
		if (!m_translators[0]->thread.joinable())
			for (auto& translator: m_translators)
				translator->thread = std::thread([this, worker = translator.get()] { AheadRun(*worker); });
		m_jobs.push_back({job});
		m_jobs.back().pushed_ns = NowNs();
	}
	m_ahead_wake.notify_all();
}

void Speculation::Done(uint64_t serial, const CommandProcessor& cp) {
	std::lock_guard lock(m_mutex);
	while (!m_jobs.empty() && m_jobs.front().job.serial <= serial) {
		// (Its records' writes are made: what read through them reads memory.)
		for (auto& chunk: m_jobs.front().chunks)
			if (auto record = std::move(chunk.record)) {
				Unlink(record.get(), nullptr);
				Retire(std::move(record));
			}
		m_jobs.pop_front();
	}
	m_done_serial = serial;
	if (!m_done_state) m_done_state = std::make_unique<CommandProcessor>(m_renderer, m_interrupt_id);
	m_done_state->CopyStateFrom(cp, false);
	m_ahead_wake.notify_all();
}

void Speculation::ResyncStatePass() {
	if (m_state_busy || m_done_serial == 0) return;
	// (It cannot follow, and the GPU thread did that submission; or it is behind the GPU thread.)
	if (m_broken ? m_done_serial >= m_broken_at : m_state_serial < m_done_serial) {
		m_state->CopyStateFrom(*m_done_state, false);
		m_state_serial = m_done_serial;
		m_broken       = false;
	}
}

bool Speculation::StatePass(const Job& job, uint64_t which) {
	if (job.reset) m_state->Reset();
	if (job.kind != Job::Kind::Graphics) return job.kind == Job::Kind::Flip; // (a flip preparation leaves the state as it is)
	m_state->ResetDeCe();
	m_state->SetFlip({});
	m_state->SetChunking(ChunkWork);
	Pm4Execution execution;
	for (;;) {
		// A chunk from here: where it starts, and its record, which reads through the one before while that is not done.
		Chunk chunk;
		chunk.start = std::make_unique<CommandProcessor>(m_renderer, m_interrupt_id);
		chunk.start->CopyStateFrom(*m_state, false);
		chunk.at = execution;
		{
			std::lock_guard lock(m_mutex);
			chunk.record = TakeRecord();
			if (const auto* entry = EntryOf(job.serial); entry != nullptr && !entry->chunks.empty())
				chunk.record->previous = entry->chunks.back().record.get();
			else if (const auto* before = EntryOf(job.serial - 1); before != nullptr && !before->chunks.empty())
				chunk.record->previous = before->chunks.back().record.get();
			m_pass_record = chunk.record.get();
		}
		chunk.record->Inherit(); // (a copy's source the records before it write is not read: Spec::State::Resolve)
		Spec::t_record       = chunk.record.get();
		const auto processed = m_state->Process(execution, job.commands);
		Spec::t_record       = nullptr;
		const bool boundary  = processed != Pm4ProcessResult::Complete && m_state->TakeBoundary();
		chunk.end            = execution;
		chunk.last           = !boundary;
		std::lock_guard lock(m_mutex);
		m_pass_record = nullptr;
		// (Not followed: what is not passed is translated the normal way.)
		auto*      entry    = EntryOf(job.serial);
		const bool followed = processed == Pm4ProcessResult::Complete || boundary;
		if (entry != nullptr && which == m_pass && followed) {
			entry->pass = which;
			entry->chunks.push_back(std::move(chunk));
			m_ahead_wake.notify_all();
		} else {
			Unlink(chunk.record.get(), nullptr);
			Retire(std::move(chunk.record));
		}
		if (!boundary || entry == nullptr || which != m_pass) {
			m_state->SetChunking(0);
			return followed && entry != nullptr;
		}
	}
}

bool Speculation::ChunkInOrder() {
	static auto& on = LiveSwitch::Find("spec_chunk_in_order", [] {
		const char* value = std::getenv("KYTY_SPEC_CHUNK_IN_ORDER");
		return value != nullptr ? std::atoll(value) : ModeToggle::Default("spec_chunk_in_order");
	}());
	return on.load(std::memory_order_relaxed) != 0;
}

const Spec::State* Speculation::RecordOf(uint64_t serial, uint32_t chunk) {
	const auto* entry = EntryOf(serial);
	return entry != nullptr && chunk < entry->chunks.size() ? entry->chunks[chunk].record.get() : nullptr;
}

Speculation::Entry* Speculation::EntryOf(uint64_t serial) {
	for (auto& entry: m_jobs)
		if (entry.job.serial == serial) return &entry;
	return nullptr;
}

void Speculation::Unlink(const Spec::State* from, const Spec::State* to) {
	const auto relink = [&](Spec::State& state) {
		if (state.previous.load(std::memory_order_relaxed) == from) state.previous = to;
	};
	for (auto& ahead: m_ahead) relink(ahead->state);
	for (auto& translator: m_translators)
		if (translator->current != nullptr) relink(translator->current->state);
	for (auto& entry: m_jobs)
		for (auto& chunk: entry.chunks)
			if (chunk.record) relink(*chunk.record);
	if (m_pass_record != nullptr) relink(*m_pass_record);
}

uint64_t Speculation::Horizon() const {
	uint64_t horizon = UINT64_MAX;
	for (const auto& translator: m_translators)
		if (translator->since != 0) horizon = std::min(horizon, translator->since);
	return horizon;
}

std::unique_ptr<Spec::State> Speculation::TakeRecord() {
	const auto horizon = Horizon();
	for (auto it = m_retired_records.begin(); it != m_retired_records.end(); ++it)
		if (it->first < horizon) {
			auto record = std::move(it->second);
			m_retired_records.erase(it);
			record->Clear();
			return record;
		}
	return std::make_unique<Spec::State>();
}

void Speculation::Retire(std::unique_ptr<Spec::State> record) {
	m_retired_records.emplace_back(++m_clock, std::move(record));
}

// The workers' CPUs: KYTY_SPECULATE_CPUS=<list> when set; otherwise the upper half of the performance cores
// (LocalPlatform::SpeculationCpuList: CPUs 8-15 on an 8 P-core + 16 E-core part), never an E-core, the GPU, record and
// upload threads still free over the whole render list; where there is no such list (Linux), off the render threads'
// CPUs (KYTY_RENDER_CPUS). On the E-cores a worker translates a chunk slower than the GPU thread can wait for it: in
// the 8 Nov placement trial the arena gained 2.8 fps in-process with the workers on CPUs 8-15, Latria and 1-1 were
// level, and the translation lost to cancels fell by a third to a half everywhere. Live: speculate_cpus, a CPU mask
// (0: this default), taken by each worker at its next wake. More workers than the list's CPUs share them (SMT).
static const std::pair<std::string, const char*>& SpeculationCpus() {
	static const auto cpus = []() -> std::pair<std::string, const char*> {
		if (const char* list = std::getenv("KYTY_SPECULATE_CPUS"); list != nullptr && *list != 0) return {list, "KYTY_SPECULATE_CPUS"};
		return {LocalPlatform::SpeculationCpuList(), "the upper half of the performance cores"};
	}();
	return cpus;
}

static void PinSpeculationWorker(int64_t mask) {
	if (mask != 0) {
		std::string list;
		for (uint32_t cpu = 0; cpu < 64; ++cpu)
			if ((static_cast<uint64_t>(mask) >> cpu) & 1u) list += (list.empty() ? "" : ",") + std::to_string(cpu);
		LocalPlatform::PinThreadToCpuList(list.c_str());
	} else if (const auto& cpus = SpeculationCpus(); !cpus.first.empty()) {
		LocalPlatform::PinThreadToCpuList(cpus.first.c_str());
	} else {
		// (Off the GPU thread's CPUs: it commits what they translate, and waits for it.)
		LocalPlatform::AvoidCpuList(std::getenv("KYTY_RENDER_CPUS"));
	}
}

void Speculation::AheadRun(Translator& translator) {
	LocalPlatform::SetThreadName("Kyty.Speculation");
	// (Worker packets, live spec_worker_packets: its segments recorded as packets, spliced at commit. Armed only
	// inside a segment recorded so: otherwise its calls go to the driver as before.)
	const LocalVulkanRecording::PassiveProducerScope passive_producer;
	static auto& live_cpus    = LiveSwitch::Find("speculate_cpus", 0);
	int64_t      applied_cpus = live_cpus.load(std::memory_order_relaxed);
	PinSpeculationWorker(applied_cpus);
	if (translator.index == 0) {
		const auto& cpus   = SpeculationCpus();
		const char* render = std::getenv("KYTY_RENDER_CPUS");
		std::printf("Speculation: %u workers on CPUs %s (%s); render threads on CPUs %s\n", static_cast<unsigned>(m_translators.size()),
		            cpus.first.empty() ? "off the render threads' CPUs" : cpus.first.c_str(), cpus.first.empty() ? "no list" : cpus.second,
		            render != nullptr && *render != 0 ? render : "any");
	}
	// (Its Vulkan calls go to the driver as they are made, no recording stream: what they name is alive then.)
	RenderContext::SetThreadExecutors(&translator.executors);
	GuestGpu::SetSpeculationThread(true);
	Spec::t_packet_slot = translator.index;
	std::unique_lock lock(m_mutex);
	for (;;) {
		Entry*   pass       = nullptr;
		Entry*   work       = nullptr;
		uint32_t work_chunk = 0;
		m_ahead_wake.wait(lock, [&] {
			if (m_quit) return true;
			ResyncStatePass();
			pass = !m_state_busy && !m_broken ? EntryOf(m_state_serial + 1) : nullptr;
			work = nullptr;
			if (pass == nullptr)
				for (auto& entry: m_jobs) {
					if (entry.pass != m_pass || entry.job.serial <= m_done_serial) continue;
					for (uint32_t c = 0; c < entry.chunks.size() && work == nullptr; ++c)
						if (!entry.chunks[c].taken && !Taken(entry.job.serial, c)) {
							work       = &entry;
							work_chunk = c;
						}
					if (work != nullptr) break;
				}
			return pass != nullptr || work != nullptr;
		});
		if (m_quit) return;
		if (const auto mask = live_cpus.load(std::memory_order_relaxed); mask != applied_cpus) {
			applied_cpus = mask;
			PinSpeculationWorker(mask);
		}
		if (pass != nullptr) {
			// The state pass of the next submission (one thread at a time; the state it starts from goes to it, and its
			// record, which reads through the one before while that is not done).
			m_state_busy     = true;
			translator.since = ++m_clock;
			const auto job   = pass->job;
			const auto which = m_pass;
			// (Chunks of an earlier pass over it go: what read through their records reads memory.)
			for (auto& chunk: pass->chunks)
				if (auto record = std::move(chunk.record)) {
					Unlink(record.get(), nullptr);
					Retire(std::move(record));
				}
			pass->chunks.clear();
			lock.unlock();
			const bool followed = StatePass(job, which);
			lock.lock();
			translator.since = 0;
			m_state_busy     = false;
			if (!followed) {
				m_broken    = true;
				m_broken_at = job.serial;
			}
			if (which == m_pass) m_state_serial = job.serial;
			m_ahead_wake.notify_all();
			continue;
		}
		// A translation, from the state the pass gave it (with the pending writes of the one before when it is done).
		auto& chunk                  = work->chunks[work_chunk];
		chunk.taken                  = true;
		translator.translating       = work->job.serial;
		translator.translating_chunk = work_chunk;
		translator.since             = ++m_clock; // (before `previous` can be retired: it is not reused meanwhile)
		translator.began_ns          = NowNs();
		translator.cancel.store(false, std::memory_order_relaxed);
		const auto         job       = work->job;
		const auto         which     = work->pass;
		auto               start     = std::move(chunk.start);
		const Pm4Execution at        = chunk.at;
		const auto         stop      = chunk.last ? std::optional<Pm4Execution> {} : std::optional {chunk.end};
		// The chunk before it: its translation not committed yet, else the pass's record of it.
		const auto* before_entry = work_chunk > 0 ? work : EntryOf(job.serial - 1);
		const auto  before_serial = work_chunk > 0 ? job.serial : job.serial - 1;
		const auto  before_chunk  = work_chunk > 0 ? work_chunk - 1
		                            : before_entry != nullptr && !before_entry->chunks.empty()
		                                ? static_cast<uint32_t>(before_entry->chunks.size() - 1)
		                                : UINT32_MAX;
		const Result* previous = nullptr;
		for (const auto& ahead: m_ahead)
			if (ahead->serial == before_serial && ahead->chunk == before_chunk && ahead->pass == which) previous = ahead.get();
		if (work_chunk == 0) {
			m_start_delay_ns += NowNs() - work->pushed_ns;
			++m_started;
		}
		const Spec::State* record = RecordOf(before_serial, before_chunk);
		const bool         blind  = previous == nullptr && record == nullptr && job.serial > m_done_serial + 1;
		lock.unlock();
		const auto began  = NowNs();
		auto       result = Translate(translator, *start, job.commands, at, previous, record, stop ? &*stop : nullptr);
		result->serial = job.serial;
		result->chunk  = work_chunk;
		result->pass   = which;
		result->blind  = blind;
		Spec::LeavePackets();
		lock.lock();
		{
			const auto took = NowNs() - began;
			const auto ms   = took / 1000000.0;
			++m_durations[ms < 0.5 ? 0 : ms < 1 ? 1 : ms < 2 ? 2 : ms < 4 ? 3 : ms < 8 ? 4 : 5];
			m_longest_ns = std::max(m_longest_ns, took);
		}
		translator.translating = 0;
		m_ahead.push_back(std::move(result));
		m_ahead_wake.notify_all();
	}
}

Speculation::Outcome Speculation::CommitReady(CommandProcessor& cp, const Job& job, uint32_t chunk, Pm4Execution& execution,
                                              std::optional<Pm4Execution>& end) {
	std::unique_ptr<Result> result;
	{
		std::unique_lock lock(m_mutex);
		// Where the chunk ends (the rest of it the normal way, up to there, when there is none for it or its commit
		// stops).
		const auto chunk_end = [&] {
			end.reset();
			if (const auto* entry = EntryOf(job.serial);
			    entry != nullptr && entry->pass == m_pass && chunk < entry->chunks.size() && !entry->chunks[chunk].last)
				end = entry->chunks[chunk].end;
		};
		// The normal way: the speculation's threads do not begin it, nor, its end unknown, any chunk of it after it.
		const auto take = [&] {
			chunk_end();
			// (KYTY_SPEC_CHUNK_IN_ORDER: a chunk the pass has not cut is translated here up to where the pass will end it;
			// the chunks after it stay the speculation's threads'.)
			const auto taken = end || ChunkInOrder() ? chunk : UINT32_MAX;
			if (job.serial > m_taken || (job.serial == m_taken && taken > m_taken_chunk)) {
				m_taken       = job.serial;
				m_taken_chunk = taken;
			}
		};
		for (;;) {
			// (Those of chunks before it, taken the normal way since, or of a state pass that went wrong.)
			for (size_t i = 0; i < m_ahead.size();) {
				if ((m_ahead[i]->serial > job.serial || (m_ahead[i]->serial == job.serial && m_ahead[i]->chunk >= chunk)) &&
				    m_ahead[i]->pass == m_pass) {
					++i;
					continue;
				}
				auto dropped = std::move(m_ahead[i]);
				m_ahead.erase(m_ahead.begin() + static_cast<ptrdiff_t>(i));
				lock.unlock();
				Finish(std::move(dropped), 0);
				lock.lock();
				i = 0;
			}
			const auto found = std::ranges::find_if(m_ahead, [&](const auto& ahead) {
				return ahead->serial == job.serial && ahead->chunk == chunk;
			});
			if (found != m_ahead.end()) {
				result = std::move(*found);
				m_ahead.erase(found);
				break;
			}
			// Being translated: its result (the normal translation would write what tells the guest the command
			// buffer is done, which it may then reuse under the speculation's thread). Not begun: this thread translates
			// it the normal way, the speculation's threads do not begin it.
			const auto translating = [&] {
				return std::ranges::any_of(m_translators, [&](const auto& translator) {
					return translator->translating == job.serial && translator->translating_chunk == chunk;
				});
			};
			if (translating()) {
				// KYTY_SPECULATE_CANCEL_US (live speculate_cancel_us, 0: never): a translation begun less than that long ago
				// has more left than this thread takes for the whole chunk (a speculation's thread is several times slower
				// per draw): it stops at its next packet, what it did is committed, the rest is translated here.
				static auto& cancel_us = LiveSwitch::Find("speculate_cancel_us", 1000000); // (default 1 s: always)
				if (const auto us = cancel_us.load(std::memory_order_relaxed); us > 0)
					for (auto& translator: m_translators)
						if (translator->translating == job.serial && translator->translating_chunk == chunk &&
						    NowNs() - translator->began_ns < static_cast<uint64_t>(us) * 1000u &&
						    !translator->cancel.exchange(true, std::memory_order_relaxed))
							++m_cancelled;
				++m_waits_for_ahead;
				const auto waited = NowNs();
				// (KYTY_SPECULATE_FLUSH_EVERY: the commits held go before waiting on a speculation's thread.)
				if (FlushEvery() > 1) {
					lock.unlock();
					m_renderer.GetCommandScheduler().FlushRecorded();
					lock.lock();
				}
				// (KYTY_SERVE_IN_WAITS 4: the guest commands queued for this thread run while it waits.)
				if ((GuestGpu::ServeInWaits() & 4u) != 0) {
					while (translating()) {
						if (GuestGpu::CommandsPending()) {
							lock.unlock();
							GuestGpu::ServePendingCommands();
							lock.lock();
							continue;
						}
						m_ahead_wake.wait_for(lock, std::chrono::microseconds(50));
					}
				} else {
					m_ahead_wake.wait(lock, [&] { return !translating(); });
				}
				m_wait_ns += NowNs() - waited;
				continue;
			}
			take();
			++m_ahead_misses;
			return Outcome::None;
		}
		if (!cp.SameState(*result->start_state) || !CommandProcessor::SamePosition(execution, result->start_at)) {
			// From another state or place: the state pass went wrong (what began from it goes); it goes on from the state
			// after this submission (Done), which is translated on the normal way.
			++m_stop_reasons["ahead: another start state"];
			++m_pass;
			m_broken    = true;
			m_broken_at = job.serial;
			m_taken     = job.serial;
			m_taken_chunk = UINT32_MAX;
			end.reset();
			lock.unlock();
			Finish(std::move(result), 0);
			return Outcome::None;
		}
		chunk_end();
	}
	return Commit(std::move(result), cp, execution);
}

Speculation::Outcome Speculation::CommitAhead(CommandProcessor& cp, const Job& job, std::span<const Job> queued,
                                              Pm4Execution& execution) {
	auto& translator = *m_translators[0];
	// Speculations of command buffers taken the normal way since, or begun from another state.
	const auto drop_ahead = [&] {
		while (!m_ahead.empty()) {
			auto dropped = std::move(m_ahead.front());
			m_ahead.pop_front();
			Finish(std::move(dropped), 0);
		}
	};
	while (!m_ahead.empty() && m_ahead.front()->serial < job.serial) {
		auto dropped = std::move(m_ahead.front());
		m_ahead.pop_front();
		Finish(std::move(dropped), 0);
	}
	std::unique_ptr<Result> result;
	if (!m_ahead.empty() && m_ahead.front()->serial == job.serial) {
		result = std::move(m_ahead.front());
		m_ahead.pop_front();
		if (!cp.SameState(*result->start_state)) {
			++m_stop_reasons["ahead: another start state"];
			Finish(std::move(result), 0);
			drop_ahead();
		}
	}
	if (!result) {
		result         = Translate(translator, cp, job.commands, execution);
		result->serial = job.serial;
	}
	// The ones queued after the last translated, from the state it leaves: while each takes the one before's place
	// and none stopped before its end.
	const Result* last = m_ahead.empty() ? result.get() : m_ahead.back().get();
	auto&         from = m_chain;
	from->CopyStateFrom(*last->end_state, false);
	uint64_t serial = last->serial;
	for (const auto& next: queued) {
		if (next.serial <= serial) continue;
		if (next.serial != serial + 1 || next.kind == Job::Kind::Other || last->stop != nullptr || last->failed ||
		    m_ahead.size() >= MaxAhead)
			break;
		serial = next.serial;
		if (next.reset) from->Reset();
		if (next.kind == Job::Kind::Flip) continue;
		from->ResetDeCe();
		from->SetFlip({});
		auto ahead    = Translate(translator, *from, next.commands, Pm4Execution {}, last);
		ahead->serial = next.serial;
		m_ahead.push_back(std::move(ahead));
		last = m_ahead.back().get();
		from->CopyStateFrom(*last->end_state, false);
	}
	return Commit(std::move(result), cp, execution);
}

std::unique_ptr<Speculation::Result> Speculation::Translate(Translator& translator, const CommandProcessor& cp,
                                                            std::span<const uint32_t> commands,
                                                            const Pm4Execution& execution, const Result* previous,
                                                            const Spec::State* record, const Pm4Execution* stop) {
	auto&                                      scheduler = m_renderer.GetCommandScheduler();
	std::unique_ptr<Result>                    result;
	std::vector<std::pair<uint64_t, uint64_t>> resolved;
	{
		std::lock_guard lock(m_mutex);
		if (translator.since == 0) translator.since = ++m_clock;
		// (One retired before every translation going on began: none of them reaches it through `previous` any more.)
		const auto horizon = Horizon();
		if (const auto reusable = std::ranges::find_if(translator.free, [&](const auto& kept) { return kept->retired < horizon; });
		    reusable != translator.free.end()) {
			result = std::move(*reusable);
			translator.free.erase(reusable);
		}
		resolved.swap(translator.resolved);
	}
	// (What the commits released of its rings, released on the thread that uses them.)
	for (const auto& [pending, tick]: resolved) ResolvePending(translator, pending, tick);
	if (!result) {
		result              = std::make_unique<Result>();
		result->owner       = &translator;
		result->recorder    = scheduler.CreateRecorder();
		result->start_state = std::make_unique<CommandProcessor>(m_renderer, m_interrupt_id);
		result->end_state   = std::make_unique<CommandProcessor>(m_renderer, m_interrupt_id);
	}
	auto& r = *result;
	// (A speculation's thread records and ends its buffers; the GPU thread's go through its recording stream.)
	r.recorder->own_thread   = !GuestGpu::IsGpuThread();
	r.recorder->pending_tick = m_pending.fetch_add(1, std::memory_order_relaxed) + 1;
	// KYTY_SPEC_WORKER_PACKETS (live spec_worker_packets; on under KYTY_SPECULATE=4): a speculation thread records its
	// segments as packets against a placeholder, spliced into the GPU thread's recording stream at commit (no command
	// buffers of its own, no submissions of them). The GPU thread's own translations record as before.
	static auto& worker_packets = LiveSwitch::Find("spec_worker_packets", ModeToggle::Default("spec_worker_packets"));
	r.recorder->packets = r.recorder->own_thread && worker_packets.load(std::memory_order_relaxed) != 0 &&
	                      LocalVulkanRecording::HasPassiveProducer();
	auto& processor          = *translator.cp;
	processor.CopyStateFrom(cp, false); // (no packet it takes uses the constant RAM)
	r.start_state->CopyStateFrom(cp, false);
	r.serial = 0;
	r.pass   = 0;
	r.blind  = false;
	r.state.Clear();
	r.state.table_upload  = translator.table_upload.get();
	r.state.shader_upload = translator.shader_upload.get();
	r.state.cancel        = &translator.cancel;
	{
		// (Linked under the lock: one committed or dropped since it was chosen is not; Finish unlinks the others.)
		std::lock_guard lock(m_mutex);
		r.retired = 0;
		kyty_local_speculations_live.fetch_add(1, std::memory_order_seq_cst); // (Finish takes it back)
		if (previous != nullptr && previous->retired == 0)
			r.state.previous = &previous->state;
		else if (record != nullptr && std::ranges::any_of(m_jobs, [&](const Entry& entry) {
			         return std::ranges::any_of(entry.chunks, [&](const Chunk& chunk) { return chunk.record.get() == record; });
		         }))
			r.state.previous = record;
		else
			r.state.previous = nullptr;
		// (Where its effects begin, as it becomes the translation going on: the log keeps what it may need.)
		r.effects_from     = m_log.End();
		r.state.log_end    = &m_log.end;
		translator.current = &r;
	}
	r.state.Inherit();
	r.holes.clear();
	r.epochs = CurrentEpochs();
	r.segment_ends.clear();
	r.stop           = nullptr;
	r.stop_execution = {};
	r.failed         = false;
	r.submitted      = 0;
	r.last_tick      = 0;
	r.next           = 0;
	r.blocked        = SIZE_MAX;
	r.start_ns       = NowNs();
	r.start_at       = execution;
	Pm4Execution spec = execution;
	// (A chunk's: up to where the next begins.)
	Pm4Execution stop_at;
	if (stop != nullptr) stop_at = *stop;
	processor.SetStopAt(stop != nullptr ? &stop_at : nullptr);
	// (What it translates is not done: nothing it does is an effect.)
	auto* const effects = std::exchange(Spec::t_effects, nullptr);
	auto* const effects_async = std::exchange(Spec::t_effects_async, nullptr);
	// (Counted above: EvictIdlePipelines either saw it and evicts none, or set its flag first and this waits for it
	// before any pipeline is bound.)
	while (kyty_local_pipelines_evicting.load(std::memory_order_seq_cst) != 0) std::this_thread::yield();
	RenderContext::SetThreadExecutors(&translator.executors);
	// (What it kept of the catalogs from its last translation, dropped when a sweep or reset happened since.)
	translator.draw->ForgetStaleCatalog();
	translator.compute->ForgetStaleCatalog();
	{
	Spec::t_state = &r.state;
	CommandScheduler::SetThreadRecorder(r.recorder.get());
	try {
		processor.BufferInit();
		for (;;) {
			const auto processed = processor.Process(spec, commands);
			if (r.state.refused == nullptr) {
				// (Its chunk's end, or a constant engine wait.)
				if (processed != Pm4ProcessResult::Complete) r.stop = processor.TakeStopReached() ? ChunkEnd : "suspended";
				break;
			}
			uint32_t   dwords  = 0;
			const auto packet  = CommandProcessor::SuspendedPacket(spec, dwords);
			const bool partial = LocalVulkanRecording::RecordedWrites() != r.state.checkpoint_work;
			r.state.Rollback();
			if (partial) { // (its recorded work cannot be taken back)
				static const auto names = [] {
					std::array<std::string, 256> all;
					for (uint32_t i = 0; i < 256; ++i) all[i] = "partly recorded packet " + std::to_string(i);
					return all;
				}();
				static std::atomic<uint32_t> reports {0};
				if (reports.fetch_add(1, std::memory_order_relaxed) < 16)
					std::printf("Speculation: packet 0x%08x refused (%s) after recording work\n", *packet, r.state.refused);
				r.failed = true;
				r.stop   = names[(*packet >> 8u) & 0xffu].c_str();
				break;
			}
			if (!IsHole(*packet) || r.state.stop) {
				r.stop = r.state.refused;
				break;
			}
			// A hole: the segment ends before it, the next begins after it, from the processor state there.
			if (r.hole_states.size() == r.holes.size()) r.hole_states.push_back(std::make_unique<CommandProcessor>(m_renderer, m_interrupt_id));
			r.hole_states[r.holes.size()]->CopyStateFrom(processor, false);
			auto& hole  = r.holes.emplace_back();
			hole.packet = packet;
			hole.dwords = dwords;
			hole.reason = r.state.refused;
			CommandProcessor::SkipSuspendedPacket(spec);
			hole.after = spec;
			scheduler.EndRecording(*r.recorder);
			r.state.CloseSegment(r.recorder->recorded.size());
			r.segment_ends.push_back(NowNs());
			r.state.refused = nullptr;
			// (The work after it does not continue the draw state of the work before it.)
			DrawStateObserver::Invalidate();
			translator.draw->NativeXprForgetState();
			processor.BufferInit();
		}
		scheduler.EndRecording(*r.recorder);
	} catch (const CommandScheduler::RecorderRefusal& refusal) {
		r.failed = true;
		r.stop   = refusal.what;
		// (Worker packets: the open segment ends here, on its thread, so the producer stops recording this thread's
		// calls; Finish drops it.)
		if (r.recorder->packets) scheduler.EndRecording(*r.recorder);
	}
	CommandScheduler::SetThreadRecorder(nullptr);
	Spec::t_state = nullptr;
	processor.SetStopAt(nullptr);
	(void)processor.TakeStopReached();
	// (The draw state the translation goes on from is not this thread's shadow any more.)
	DrawStateObserver::Invalidate();
	translator.draw->NativeXprForgetState();
	}
	if (GuestGpu::IsGpuThread()) RenderContext::SetThreadExecutors(nullptr);
	Spec::t_effects = effects;
	Spec::t_effects_async = effects_async;
	DrawStateObserver::Invalidate();
	r.end_ns     = NowNs();
	r.progressed = !r.holes.empty() || spec.MadeProgress();
	if (r.progressed && !r.failed) {
		r.state.CloseSegment(r.recorder->recorded.size());
		r.segment_ends.push_back(r.end_ns);
		r.stop_execution = std::move(spec);
	}
	r.end_state->CopyStateFrom(processor, false);
	{
		std::lock_guard lock(m_mutex);
		translator.since = 0;
	}
	return result;
}

Speculation::Outcome Speculation::Commit(std::unique_ptr<Result> result, CommandProcessor& cp, Pm4Execution& execution) {
	auto& r = *result;
	// Stopped at its first packet (a wait it does not find satisfied yet): nothing to commit.
	if (!r.progressed) {
		Finish(std::move(result), 0);
		return Outcome::None;
	}
	if (r.stop != nullptr && r.stop != ChunkEnd) ++m_stop_reasons[r.stop];
	m_effects.Clear(r.effects_from);
	return Continue(std::move(result), cp, execution);
}

Speculation::Outcome Speculation::Continue(std::unique_ptr<Result> result, CommandProcessor& cp, Pm4Execution& execution) {
	auto&      r        = *result;
	const auto start    = NowNs();
	const auto progress = r.failed ? Progress::Stopped : CommitSegments(r, cp);
	// (Its segments' buffers held for one submission go now: a speculation thread may wait for their tick.
	// KYTY_SPECULATE_FLUSH_EVERY: every Nth commit, and one that blocks at a wait hole.)
	if (FlushEvery() <= 1 || progress == Progress::Blocked || ++m_held_commits >= FlushEvery()) {
		m_held_commits = 0;
		m_renderer.GetCommandScheduler().FlushRecorded();
	}
	m_commit_ns += NowNs() - start;
	if (progress == Progress::Blocked) {
		m_blocked = std::move(result);
		return Outcome::Blocked;
	}
	const auto committed = r.next;
	m_blind += r.blind ? 1 : 0;
	if (progress == Progress::Stopped) {
		++m_stopped;
		m_blind_stopped += r.blind ? 1 : 0;
	}
	const auto kept      = committed == 0 ? r.start_ns : r.segment_ends[committed - 1];
	m_kept_ns += kept - r.start_ns;
	m_lost_ns += r.end_ns - kept;
	m_segments += committed;
	auto outcome = Outcome::None;
	if (progress == Progress::Complete) {
		EXIT_IF(committed != r.state.closed.size());
		// Committed whole: the processor state at its end (or at the packet it stopped at).
		cp.CopyStateFrom(*r.end_state, false);
		outcome = r.stop == nullptr ? Outcome::Whole : r.stop == ChunkEnd ? Outcome::Chunk : Outcome::Partly;
		if (r.stop != nullptr) execution = std::move(r.stop_execution);
	} else if (committed != 0) {
		// Stopped after the hole that followed the last segment committed (`cp` translated it).
		execution = std::move(r.holes[committed - 1].after);
		outcome   = Outcome::Partly;
	}
	++(outcome == Outcome::Whole || outcome == Outcome::Chunk ? m_whole : outcome == Outcome::Partly ? m_partly : m_none);
	Finish(std::move(result), committed);
	{
		// What no result began before: of those translated ahead (in the order they were done, not begun) and those
		// being translated (from when their translator took them: its current result's until then).
		std::lock_guard lock(m_mutex);
		auto            oldest = m_log.End();
		for (const auto& ahead: m_ahead) oldest = std::min(oldest, ahead->effects_from);
		for (const auto& translator: m_translators)
			if (translator->translating != 0 && translator->current != nullptr)
				oldest = std::min(oldest, translator->current->effects_from);
		m_log.DropBefore(oldest);
	}
	Report();
	return outcome;
}

const char* Speculation::Changed(const Result& result, size_t index) const {
	auto&       buffers  = m_renderer.GetBufferCache();
	auto&       textures = m_renderer.GetTextureCache();
	const auto& state    = result.state;
	const auto& epochs   = result.epochs;
	const auto& segment  = state.closed[index];
	const auto  from     = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
	// (Memory mapped since changes nothing it used; memory unmapped under what it read or uses does.)
	auto&      resources = m_renderer.GetGpuResources();
	const bool mapping   = resources.MappingEpoch() != epochs.mapping;
	// (Nothing registered, released or unmapped since it began, mostly: no range or image to look at.)
	if (mapping || buffers.RegistrationEpoch() != epochs.buffers)
		for (size_t i = from.buffer_ranges; i < segment.end.buffer_ranges; ++i) {
			const auto& range = state.buffer_ranges[i];
			if (buffers.RegisteredSince(range.begin, range.end - range.begin, epochs.buffers)) return "commit: buffers registered";
			if (mapping && resources.UnmappedSince(epochs.mapping, range.begin, range.end - range.begin))
				return "commit: memory unmapped";
		}
	if (mapping) {
		for (size_t i = from.clean_reads; i < segment.end.clean_reads; ++i) {
			const auto& read = state.clean_reads[i];
			if (resources.UnmappedSince(epochs.mapping, read.begin, read.end - read.begin)) return "commit: memory unmapped";
		}
		for (size_t i = from.late_reads; i < segment.end.late_reads; ++i) {
			const auto& read = state.late_reads[i];
			if (resources.UnmappedSince(epochs.mapping, read.begin, read.end - read.begin)) return "commit: memory unmapped";
		}
		for (const auto& use: segment.images)
			if (resources.UnmappedSince(epochs.mapping, use.image->info.data.address, use.image->info.data.size))
				return "commit: memory unmapped";
	}
	if (textures.ResolutionEpoch() == epochs.images) return nullptr;
	for (const auto& use: segment.images)
		if (!textures.StillResolved(*use.image, use.serial, epochs.images)) return "commit: images registered";
	for (size_t i = from.touched_images; i < segment.end.touched_images; ++i)
		if (!textures.StillResolved(*state.touched_images[i].first, state.touched_images[i].second, epochs.images))
			return "commit: images registered";
	return nullptr;
}

// KYTY_SPEC_COMMIT_VERIFY: whether this one is the Nth (the thread that commits).
static bool CommitVerifyDue() {
	static uint64_t count = 0;
	const auto      every = Spec::CommitVerifyEvery();
	return every > 0 && ++count % static_cast<uint64_t>(every) == 0;
}

Speculation::Progress Speculation::CommitSegments(Result& result, CommandProcessor& cp) {
	auto&       scheduler = m_renderer.GetCommandScheduler();
	const auto& state     = result.state;
	const auto  segments  = state.closed.size();
	EXIT_IF(result.holes.size() + 1 != segments);
	// (Blocked at a wait hole: from there.)
	if (const auto hole = std::exchange(result.blocked, SIZE_MAX); hole != SIZE_MAX) {
		const auto run = RunHole(result, cp, hole);
		if (run == HoleRun::Blocked) return result.blocked = hole, Progress::Blocked;
		if (run == HoleRun::Changed) return Progress::Stopped;
	}
	while (result.next < segments) {
		const auto index = result.next;
		m_effects.Take(m_log);
		// What the work assumed: nothing registered or released since; the surface metadata it found not cleared not
		// cleared since; the guest memory it read as no GPU work had written it not written since (the effects
		// logged since it began); the words it read from a speculation before it in memory now; the waits it found
		// satisfied satisfied.
		if (const char* why = Changed(result, index)) return ++m_stop_reasons[why], Progress::Stopped;
		// The images it uses as it found them: none written through a buffer since (a hole's storage-buffer write, fill or
		// copy over its memory, or an earlier segment's, replayed in its Before). The normal path's binding uploads such an
		// image from those bytes before the work; this work was recorded without the upload and would read it as it was
		// (the game's probe input, 0x27ce00000 at the tester's spot, written per face by a refused dispatch: the dark).
		for (const auto& use: state.closed[index].images)
			if (!use.found_buffer_modified && use.image->IsBufferModified())
				return ++m_stop_reasons["commit: GPU work wrote an image it uses"], Progress::Stopped;
		for (const auto& fact: state.meta_facts)
			if (!fact.cleared && fact.first <= index && index <= fact.last &&
			    m_renderer.GetTextureCache().IsMetaCleared(fact.address, fact.slice))
				return ++m_stop_reasons["commit: a surface cleared"], Progress::Stopped;
		{
			const auto  from    = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
			const auto& to      = state.closed[index].end;
			const auto& effects = m_effects;
			// The guest memory it read on the CPU as no GPU work wrote it (and as it was): not written since.
			// (Stamped: a page written since is a candidate, the log's entries since over the read decide.)
			const bool any_since = m_log.End() > result.effects_from;
			if (effects.stamped) {
				// (KYTY_SPEC_WALK_FROM_STAMP: the walk tries the entry the read's latest stamp names first;
				// KYTY_SPEC_COMMIT_VERIFY: every Nth such walk, the walk from the speculation's start as well.)
				if (any_since) {
					const bool from_stamp = Spec::WalkFromStampOn();
					for (size_t i = from.clean_reads; i < to.clean_reads; ++i) {
						const auto& read = state.clean_reads[i];
						const auto  size = read.end - read.begin;
						if (!m_log.stamps.MaybeSince(read.begin, size, result.effects_from)) continue;
						const Spec::EffectsLog::Entry* write = nullptr;
						if (from_stamp) {
							bool direct = false;
							write       = m_log.WriteOverFromStamp(result.effects_from, read.begin, size, direct);
							if (direct) LiveCounters::Add(LiveCounters::SpecWalkDirect);
							if (CommitVerifyDue()) {
								const bool first = m_log.FirstWriteOver(result.effects_from, read.begin, size) != nullptr;
								LiveCounters::Add(LiveCounters::SpecCommitVerified);
								if (first != (write != nullptr)) LiveCounters::Add(LiveCounters::SpecCommitMismatch);
							}
						} else {
							write = m_log.FirstWriteOver(result.effects_from, read.begin, size);
						}
						if (write != nullptr) return ++m_stop_reasons[WroteWhatItRead(write)], Progress::Stopped;
					}
				}
				// (KYTY_SPEC_LATE_READS: a late read, by the processor's writes logged after it was read, and GPU work's since
				// the speculation began: a GPU write is logged as its work is recorded, before it reaches memory, and its
				// page's read protection may be deferred, PageManager::DeferReadProtection.)
				if (any_since)
					for (size_t i = from.late_reads; i < to.late_reads; ++i) {
						const auto& read  = state.late_reads[i];
						const auto  bytes = read.end - read.begin;
						if (!m_log.stamps.MaybeSince(read.begin, bytes, result.effects_from)) continue;
						if (const auto* gpu = m_log.FirstGpuWriteOver(result.effects_from, read.begin, bytes))
							return ++m_stop_reasons[WroteWhatItRead(gpu)], Progress::Stopped;
						const auto since = std::max(result.effects_from, read.since);
						if (m_log.End() <= since || !m_log.stamps.MaybeSince(read.begin, bytes, since)) continue;
						if (const auto* write = m_log.FirstWriteOver(since, read.begin, bytes))
							return ++m_stop_reasons[WroteWhatItRead(write)], Progress::Stopped;
					}
			} else if (!effects.gpu_writes.Empty() || !effects.host_writes.Empty()) {
				for (size_t i = from.clean_reads; i < to.clean_reads; ++i) {
					const auto& read = state.clean_reads[i];
					if (effects.gpu_writes.Intersects(read.begin, read.end - read.begin) ||
					    effects.host_writes.Intersects(read.begin, read.end - read.begin))
						return ++m_stop_reasons[WroteWhatItRead(m_log.FirstWriteOver(result.effects_from, read.begin,
						                                                             read.end - read.begin))],
						       Progress::Stopped;
				}
				// (Late reads as clean ones here: every write since it began.)
				for (size_t i = from.late_reads; i < to.late_reads; ++i) {
					const auto& read = state.late_reads[i];
					if (effects.gpu_writes.Intersects(read.begin, read.end - read.begin) ||
					    effects.host_writes.Intersects(read.begin, read.end - read.begin))
						return ++m_stop_reasons[WroteWhatItRead(m_log.FirstWriteOver(result.effects_from, read.begin,
						                                                             read.end - read.begin))],
						       Progress::Stopped;
				}
			}
			// The descriptor sets it binds: none retired since (one may be reused under it).
			for (const auto retired: effects.retired_sets)
				if (std::find(state.sets.begin() + static_cast<ptrdiff_t>(from.sets),
				              state.sets.begin() + static_cast<ptrdiff_t>(to.sets), retired) !=
				    state.sets.begin() + static_cast<ptrdiff_t>(to.sets))
					return ++m_stop_reasons["commit: a hole retired a set"], Progress::Stopped;
			if (effects.stamped) {
				if (any_since)
					for (size_t i = from.cpu_copies; i < to.cpu_copies; ++i) {
						const auto& copy = state.cpu_copies[i];
						if (m_log.stamps.MaybeSince(copy.begin, copy.end - copy.begin, result.effects_from) &&
						    m_log.FirstGpuWriteOver(result.effects_from, copy.begin, copy.end - copy.begin) != nullptr)
							return ++m_stop_reasons["commit: GPU work wrote what it copies"], Progress::Stopped;
					}
			} else if (!effects.gpu_writes.Empty())
				for (size_t i = from.cpu_copies; i < to.cpu_copies; ++i) {
					const auto& copy = state.cpu_copies[i];
					if (effects.gpu_writes.Intersects(copy.begin, copy.end - copy.begin))
						return ++m_stop_reasons["commit: GPU work wrote what it copies"], Progress::Stopped;
				}
			for (size_t i = from.chain_reads; i < to.chain_reads; ++i) {
				const auto& read = state.chain_reads[i];
				// (KYTY_WRITE_GUARD_SPEC: a page a pending guest write guards, through the backing view: wg2's census had this
				// read fault ~1 a frame.)
				const void* word = reinterpret_cast<const void*>(read.address);
				if (WriteGuard::g_count.load(std::memory_order_relaxed) != 0)
					if (const auto* backing = WriteGuard::GuardedBacking(read.address, 4u)) word = backing;
				if (*static_cast<const uint32_t*>(word) != read.value)
					return ++m_stop_reasons["commit: a command buffer before wrote otherwise"], Progress::Stopped;
			}
			for (size_t i = from.waits; i < to.waits; ++i) {
				const auto& wait  = state.waits[i];
				uint64_t    value = 0;
				const void* from  = reinterpret_cast<const void*>(wait.address);
				if (WriteGuard::g_count.load(std::memory_order_relaxed) != 0)
					if (const auto* backing = WriteGuard::GuardedBacking(wait.address, wait.bytes)) from = backing;
				std::memcpy(&value, from, wait.bytes);
				if (!TestWaitRegMemValue(value, wait.reference, wait.mask, wait.function))
					return ++m_stop_reasons["commit: a wait not satisfied"], Progress::Stopped;
			}
		}
		const auto prepared = CurrentEpochs();
		Before(result, index);
		// (Its preparation registers what the work did not find registered (BDA ranges): a buffer merged away under
		// the recorded work leaves it reading a destroyed one. The work after it is translated in order.)
		if (CurrentEpochs() != prepared && Changed(result, index) != nullptr)
			return ++m_stop_reasons["commit: preparation registered buffers"], Progress::Stopped;
		// KYTY_SPEC_DRAW_INDIRECT_VERIFY: its sampled DRAW_INDIRECT table draws, here where their arguments are current.
		for (const auto& entry: state.draw_verifies)
			if (entry.segment == index) VerifyDrawIndirect(m_renderer, entry);
		const auto& segment = state.closed[index];
		if (const auto tick = scheduler.SubmitRecorded(*result.recorder, segment.recorded - result.submitted); tick != 0)
			result.last_tick = tick;
		result.submitted = segment.recorded;
		// (Worker packets: spliced into the open buffer, which goes on: the draw state this thread knew of it is not
		// the buffer's any more. The recording worker's replay state is: the same buffer, replayed on the same thread.)
		if (result.recorder->packets) {
			DrawStateObserver::Invalidate();
			m_renderer.GetRenderExecutor().NativeXprForgetState();
		}
		After(result, index);
		if ((GuestGpu::ServeInWaits() & 2u) != 0) GuestGpu::ServePendingCommands(); // (KYTY_SERVE_IN_WAITS 2)
		m_renderer.GetBufferCache().WriteBackCompletedFeedback(false); // (KYTY_EAGER_FEEDBACK)
		result.next = index + 1;
		if (index < result.holes.size()) {
			const auto run = RunHole(result, cp, index);
			if (run == HoleRun::Blocked) return result.blocked = index, Progress::Blocked;
			if (run == HoleRun::Changed) return Progress::Stopped;
		}
	}
	return Progress::Complete;
}

void Speculation::Before(Result& result, size_t index) {
	const Spec::EffectSourceScope source(Spec::EffectSource::Commit);
	auto&       resources = m_renderer.GetGpuResources();
	auto&       buffers   = m_renderer.GetBufferCache();
	auto&       textures  = m_renderer.GetTextureCache();
	auto&       scheduler = m_renderer.GetCommandScheduler();
	auto&       state     = result.state;
	const auto& segment   = state.closed[index];
	const auto  from      = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
	const auto& to        = segment.end;
	BdaDirtyRegions::Publish();
	// In the scheduler's command buffer (submitted before the segment's): the ranges it writes obtained written
	// (their CPU-dirty pages uploaded, then GPU-owned), those it reads made current, its BDA reads prepared, and each
	// image where the work found it.
	RenderLockGuard lock(m_renderer.GetMutex());
	for (size_t i = from.written; i < to.written; ++i) {
		const auto& range = state.written[i];
		(void)buffers.ObtainBuffer(range.begin, range.end - range.begin, true);
	}
	for (size_t i = from.current; i < to.current; ++i) {
		const auto& range = state.current[i];
		buffers.EnsureBufferContents(range.begin, range.end - range.begin);
	}
	// (What the holes wrote on the CPU since, over ranges it reads on the GPU.)
	if (m_effects.stamped) {
		if (m_log.End() > result.effects_from)
			for (size_t i = from.buffer_ranges; i < to.buffer_ranges; ++i) {
				const auto& range = state.buffer_ranges[i];
				m_log.ForEachHostWriteOver(result.effects_from, range.begin, range.end - range.begin,
				                           [&](uint64_t address, uint64_t size) { buffers.EnsureBufferContents(address, size); });
			}
	} else if (!m_effects.host_writes.Empty())
		for (size_t i = from.buffer_ranges; i < to.buffer_ranges; ++i) {
			const auto& range = state.buffer_ranges[i];
			m_effects.host_writes.ForEachIntersection(range.begin, range.end - range.begin, [&](RangeSet::Range written) {
				buffers.EnsureBufferContents(written.address, written.size);
			});
		}
	// The texel buffers it reads over images GPU work wrote (KYTY_SPECULATE_TEXEL_SYNC), synchronized from them as the
	// normal path does at the draw: after the uploads above (the image's bytes win, as there), before the work (what it
	// writes over them lands after the copy, as in order).
	for (size_t i = from.texel_syncs; i < to.texel_syncs; ++i) {
		const auto& range = state.texel_syncs[i];
		buffers.EnsureTexelContents(range.begin, range.end - range.begin);
	}
	if (segment.bda_all) {
		resources.PrepareBda();
	} else {
		for (size_t first = from.bda; first < to.bda; first += 128) {
			const std::span ranges(state.bda.data() + first, std::min<size_t>(128, to.bda - first));
			if (!resources.PrepareBdaReadRanges(ranges)) resources.PrepareBda();
		}
	}
	for (size_t i = from.invalidated; i < to.invalidated; ++i) {
		const auto& range = state.invalidated[i];
		textures.InvalidateMemoryFromGPU(range.begin, range.end - range.begin, "speculation");
	}
	for (const auto& use: segment.images) {
		EXIT_IF(use.image->serial != use.serial); // (nothing released images since: Epochs)
		use.image->RestoreState(use.found, use.found_subresources, scheduler.Current().Handle());
	}
}

void Speculation::After(Result& result, size_t index) {
	const Spec::EffectSourceScope source(Spec::EffectSource::Commit);
	auto&       buffers  = m_renderer.GetBufferCache();
	auto&       textures = m_renderer.GetTextureCache();
	auto&       state    = result.state;
	const auto& segment  = state.closed[index];
	const auto  from     = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
	const auto& to       = segment.end;
	// Its writes are the latest GPU work over those ranges; the images as it left them; the caches' LRU; the command
	// processor's guest memory writes.
	for (size_t i = from.written; i < to.written; ++i) {
		const auto& range = state.written[i];
		buffers.NoteSpeculativeWrite(range.begin, range.end - range.begin);
	}
	for (auto& use: state.closed[index].images) {
		const Image::StateLock lock(*use.image);
		use.image->backing.state              = use.left;
		use.image->backing.subresource_states = std::move(use.left_subresources);
		use.image->transit_group              = use.left_group;
		MoveImageStateEpoch(use.image); // (KYTY_TABLE_SERIALS: the image's serial moves, and the epoch)
	}
	// KYTY_SPEC_TOUCH_BATCH (live spec_touch_batch, default on): one lock for the segment's images (~0.4 ms a frame of
	// LRU touches at 1-1 were taken image by image); 0: one by one, as before.
	static auto& touch_batch = LiveSwitch::Find("spec_touch_batch", 1);
	if (touch_batch.load(std::memory_order_relaxed) == 0) {
		for (size_t i = from.touched_images; i < to.touched_images; ++i)
			textures.TouchSpeculated(*state.touched_images[i].first, state.touched_images[i].second);
	} else if (to.touched_images > from.touched_images) {
		textures.TouchSpeculated(std::span(state.touched_images).subspan(from.touched_images, to.touched_images - from.touched_images));
	}
	for (size_t i = from.touched_buffers; i < to.touched_buffers; ++i) buffers.TouchSpeculated(state.touched_buffers[i]);
	// The work before the processor's guest memory writes is on the GPU before them, as in order (BufferFlush): the
	// segments' buffers held for one submission go first. (Labels written while that work was held made ~1.5 frames
	// in 1,000 lose whole passes at the stone arch: speculate_batch_submits.) The buffer copies recorded before a label
	// are made first, as WriteGuestMemory does: issue #196.
	// (KYTY_WRITE_GUARD_SPEC: a label lands at once, guarded until the copies pushed before it are made, as on the normal
	// path; a copy, or a write the guard does not take, drains first as before.)
	const bool guard   = WriteGuard::InSpeculation();
	bool       drained = false;
	const auto drain   = [&] {
		if (drained) return;
		drained = true;
		AsyncUpload::Drain();
	};
	if (from.writes != to.writes) {
		m_renderer.GetCommandScheduler().FlushRecorded();
		if (!guard) drain();
	}
	for (size_t i = from.writes; i < to.writes; ++i) {
		const auto& write = state.writes[i];
		if (write.copy) {
			drain();
			buffers.CopyGuestMemory(write.address, write.value, write.size);
		} else {
			if (!guard || !CommitGuardedWrite(reinterpret_cast<void*>(write.address), &write.value, write.size)) {
				drain();
				std::memcpy(reinterpret_cast<void*>(write.address), &write.value, write.size);
			}
			// (A speculation translated after it that read memory there before: Spec::EffectsLog.)
			Spec::NoteHostWrite(write.address, write.size);
			Spec::NoteLabel(write.address);
		}
	}
	if (from.feedbacks != to.feedbacks) {
		RenderLockGuard lock(m_renderer.GetMutex());
		for (size_t i = from.feedbacks; i < to.feedbacks; ++i)
			buffers.ScheduleCopyFeedback(state.feedbacks[i].begin, state.feedbacks[i].end - state.feedbacks[i].begin);
	}
}

Speculation::HoleRun Speculation::RunHole(Result& result, CommandProcessor& cp, size_t index) {
	const Spec::EffectSourceScope source(Spec::EffectSource::Hole);
	const auto& hole  = result.holes[index];
	const auto& state = *result.hole_states[index];
	cp.CopyStateFrom(state, false);
	// (The draw state it starts from is not the shadow's.)
	auto& executor = m_renderer.GetRenderExecutor();
	DrawStateObserver::Invalidate();
	executor.NativeXprForgetState();
	Pm4Execution execution;
	const auto   start     = NowNs();
	const auto   processed = cp.Process(execution, {hole.packet, hole.dwords});
	m_hole_ns += NowNs() - start;
	DrawStateObserver::Invalidate();
	executor.NativeXprForgetState();
	// (A wait not satisfied yet: the commit waits there. Draws, dispatches and end-of-pipe events do not wait.)
	if (processed != Pm4ProcessResult::Complete) {
		EXIT_IF(!CommandProcessor::AtWait(execution));
		++m_waits_blocked;
		return HoleRun::Blocked;
	}
	++m_holes_run;
	// The work after it assumed it changed no processor state but its registers' (they are the same). (What it wrote
	// and retired: Spec::Effects, checked before each segment.)
	if (!cp.SameDrawState(state)) return ++m_stop_reasons["commit: a hole changed the draw state"], HoleRun::Changed;
	return HoleRun::Done;
}

void Speculation::Finish(std::unique_ptr<Result> result, size_t first) {
	auto& scheduler = m_renderer.GetCommandScheduler();
	auto& r         = *result;
	r.state.ReleaseConsumePending(); // (its holes' clears are consumed now, or never: spec_predict_dcc=3)
	for (const auto& hole: r.holes) ++m_hole_reasons[hole.reason];
	if (first < r.state.closed.size() || !r.recorder->recorded.empty() || r.recorder->open != SIZE_MAX)
		scheduler.DiscardRecorded(*r.recorder);
	// (Its rings' data is read by the command buffers it submitted, done with the last; none submitted: free now. Not
	// the open command buffer's tick: a speculation thread waiting for its ring would wait for a submission that may
	// wait for it.)
	const auto tick    = r.last_tick;
	const auto pending = r.recorder->pending_tick;
	auto&      owner   = *r.owner;
	if (!r.recorder->own_thread) ResolvePending(owner, pending, tick);
	std::lock_guard lock(m_mutex);
	// (A speculation thread's: released when it begins its next translation.)
	if (r.recorder->own_thread) owner.resolved.emplace_back(pending, tick);
	// (Its pending writes are made or dropped: what was translated after it reads through its command buffer's record
	// while that is not done, else memory.)
	Unlink(&r.state, RecordOf(r.serial, r.chunk));
	// (Its commands are submitted or discarded: none of them binds a pipeline later.)
	kyty_local_speculations_live.fetch_sub(1, std::memory_order_seq_cst);
	r.retired = ++m_clock;
	owner.free.push_back(std::move(result));
}

void Speculation::ResolvePending(Translator& translator, uint64_t pending, uint64_t tick) {
	translator.table_upload->ResolvePendingTicks(pending, tick);
	translator.shader_upload->ResolvePendingTicks(pending, tick);
	translator.draw->ResolvePendingTicks(pending, tick);
	translator.compute->ResolvePendingTicks(pending, tick);
}

void Speculation::Report() {
	if ((m_whole + m_partly + m_none) % 2000 != 0) return;
	const auto print = [](const char* title, std::unordered_map<const char*, uint64_t>& counts) {
		std::vector<std::pair<const char*, uint64_t>> sorted(counts.begin(), counts.end());
		std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second > b.second; });
		std::printf("; %s:", title);
		for (const auto& [reason, count]: sorted) std::printf(" %s %llu", reason, static_cast<unsigned long long>(count));
		counts.clear();
	};
	const auto frame  = m_renderer.FrameNumber();
	const auto frames = static_cast<double>(std::max<uint64_t>(frame - m_report_frame, 1));
	m_report_frame    = frame;
	const auto ms     = [&](uint64_t ns) { return static_cast<double>(ns) / 1e6 / frames; };
	std::printf("Speculation: %llu whole, %llu partly, %llu not committed; %llu segments, %llu holes (%llu waits blocked); "
	            "ahead: %llu being translated, %llu not yet; blind %llu (stopped %llu of %llu); per frame: queue %.2f ms, "
	            "translation kept %.2f ms, lost %.2f ms, commit %.2f ms (holes %.2f ms), waits %.2f ms; start delay %.2f ms; "
	            "translations %llu/%llu/%llu/%llu/%llu/%llu (<0.5/1/2/4/8 ms/more, longest %.1f ms)",
	            static_cast<unsigned long long>(m_whole), static_cast<unsigned long long>(m_partly),
	            static_cast<unsigned long long>(m_none), static_cast<unsigned long long>(m_segments),
	            static_cast<unsigned long long>(m_holes_run), static_cast<unsigned long long>(m_waits_blocked),
	            static_cast<unsigned long long>(m_waits_for_ahead), static_cast<unsigned long long>(m_ahead_misses),
	            static_cast<unsigned long long>(m_blind), static_cast<unsigned long long>(m_blind_stopped),
	            static_cast<unsigned long long>(m_stopped),
	            ms(m_queue_ns), ms(m_kept_ns), ms(m_lost_ns), ms(m_commit_ns - m_hole_ns), ms(m_hole_ns), ms(m_wait_ns),
	            m_started != 0 ? static_cast<double>(m_start_delay_ns) / 1e6 / static_cast<double>(m_started) : 0.0,
	            static_cast<unsigned long long>(m_durations[0]), static_cast<unsigned long long>(m_durations[1]),
	            static_cast<unsigned long long>(m_durations[2]), static_cast<unsigned long long>(m_durations[3]),
	            static_cast<unsigned long long>(m_durations[4]), static_cast<unsigned long long>(m_durations[5]),
	            static_cast<double>(m_longest_ns) / 1e6);
	m_waits_for_ahead = m_ahead_misses = 0;
	m_blind = m_stopped = m_blind_stopped = 0;
	m_kept_ns = m_lost_ns = m_commit_ns = m_hole_ns = m_queue_ns = 0;
	{
		std::lock_guard lock(m_mutex);
		m_wait_ns = m_start_delay_ns = m_started = m_longest_ns = 0;
		m_durations.fill(0);
	}
	print("stops", m_stop_reasons);
	print("holes", m_hole_reasons);
	{
		std::lock_guard lock(m_mutex);
		std::printf("; stopped for the GPU thread: %llu", static_cast<unsigned long long>(std::exchange(m_cancelled, 0)));
	}
	std::printf("\n");
	std::fflush(stdout);
	m_whole = m_partly = m_none = m_segments = m_holes_run = m_waits_blocked = 0;
}

} // namespace Libs::Graphics
