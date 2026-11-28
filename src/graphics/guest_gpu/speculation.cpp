#include "graphics/guest_gpu/speculation.h"
#include "local/buffercache-par.h" // (buffercache-par, KYTY_BC_PAR)

#include "common/assert.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/bdaDirtyRegions.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "draw-state-observer.h"
#include "envio2.h"
#include "envio3.h"
#include "local-platform.h"
#include "vulkan-recording.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
volatile std::atomic_uint32_t kyty_local_speculate_mode {0};
volatile std::atomic_uint32_t kyty_local_speculate_threads {2};
}

namespace Libs::Graphics {

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

Speculation::Translator::Translator(RenderContext& renderer, uint32_t number, int queue_event_id, uint32_t table_mib,
                                    uint32_t shader_mib)
    : index(number), draw(std::make_unique<RenderExecutor>(renderer)), compute(std::make_unique<RenderExecutor>(renderer)),
      executors {draw.get(), compute.get()}, cp(std::make_unique<CommandProcessor>(renderer, queue_event_id)),
      table_upload(std::make_unique<StreamBuffer>(renderer.GetGraphics(), renderer.GetCommandScheduler(), MemoryUsage::Upload,
                                                  uint64_t {table_mib} << 20u,
                                                  AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress)),
      shader_upload(std::make_unique<StreamBuffer>(renderer.GetGraphics(), renderer.GetCommandScheduler(),
                                                   MemoryUsage::Upload, uint64_t {shader_mib} << 20u)) {
	draw->ReadCatalogOf(renderer.DefaultRenderExecutor(), 1 + number);
	compute->ReadCatalogOf(renderer.DefaultComputeRenderExecutor(), 1 + number);
}

Speculation::Speculation(RenderContext& renderer)
    : m_renderer(renderer), m_chain(std::make_unique<CommandProcessor>(renderer, 0)),
      m_state(std::make_unique<CommandProcessor>(renderer, 0)) {
	m_state->SetStateOnly(true);
	const auto threads = kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4
	                         ? std::clamp<uint32_t>(kyty_local_speculate_threads.load(std::memory_order_relaxed), 1u,
	                                                static_cast<uint32_t>(Spec::MaxPacketSlots))
	                         : 1u;
	for (uint32_t i = 0; i < threads; ++i) m_translators.push_back(std::make_unique<Translator>(renderer, i));
}

// (envio2) No state pass, no Push: Hand / CommitHanded only.
Speculation::Speculation(RenderContext& renderer, int queue_event_id, std::function<void()> wake)
    : m_renderer(renderer), m_chain(std::make_unique<CommandProcessor>(renderer, queue_event_id)),
      m_state(std::make_unique<CommandProcessor>(renderer, queue_event_id)), m_detached_wake(std::move(wake)),
      m_name(Envio3::Mode() != 0 ? "Speculation (envio3, ACB)" : "Speculation (envio2, ACB)") {
	m_queue_event_id = queue_event_id;
	m_max_in_flight  = Envio3::Mode() != 0 ? Envio3::MaxInFlight : 1;
	m_translators.push_back(std::make_unique<Translator>(renderer, 0, queue_event_id));
	m_detached = std::thread([this] { DetachedRun(); });
}

// (tej, KYTY_TEJ F1) The shadow translator: number 7 (packet slot 7, catalog role 8; KYTY_SPECULATE=4 uses 0..threads-1,
// envio2 0), rings of 16 + 8 MiB instead of 64 + 16.
Speculation::Speculation(RenderContext& renderer, ShadowTag, uint32_t number, uint32_t table_mib, uint32_t shader_mib)
    : m_renderer(renderer), m_chain(std::make_unique<CommandProcessor>(renderer, 0)),
      m_state(std::make_unique<CommandProcessor>(renderer, 0)), m_name("Speculation (tej, sombra)") {
	m_translators.push_back(std::make_unique<Translator>(
	    renderer, std::min(number, static_cast<uint32_t>(Spec::MaxPacketSlots - 1)), 0, table_mib, shader_mib));
}

void Speculation::ShadowThreadInit() {
	auto& translator = *m_translators[0];
	RenderContext::SetThreadExecutors(&translator.executors);
	GuestGpu::SetSpeculationThread(true);
	Spec::t_packet_slot = translator.index;
	BcPar::t_reader     = true; // (buffercache-par: lee el indice de BufferCache con cerrojo compartido)
}

Speculation::ShadowOut Speculation::Shadow(const CommandProcessor& start, std::span<const uint32_t> commands,
                                           CommandProcessor& end) {
	auto&     translator = *m_translators[0];
	const auto t0         = NowNs();
	auto      result     = Translate(translator, start, commands, Pm4Execution {});
	const auto t1         = NowNs();
	// (Its packets passed: what the GPU thread retired while it read is destroyed once this is seen, Spec::PacketsPassed.)
	Spec::LeavePackets();
	ShadowOut out;
	out.stop       = result->stop == ChunkEnd ? nullptr : result->stop;
	out.failed     = result->failed;
	out.holes      = static_cast<uint32_t>(result->holes.size());
	out.progressed = result->progressed;
	out.translate_ns = t1 - t0;
	if (Spec::t_shadow_end_hook != nullptr) Spec::t_shadow_end_hook(&result->state); // (tej F1d)
	end.CopyStateFrom(*result->end_state, false);
	// (Never submitted: its command buffers discarded, its rings' and executors' pending work freed before the next.)
	Finish(std::move(result), 0);
	m_hole_reasons.clear(); // (Finish counts them for a report this translator never prints)
	return out;
}

void Speculation::DetachedRun() {
	LocalPlatform::SetThreadName("Kyty.Acb2");
	// (Off the GPU thread's CPUs, as the speculation's threads: it goes on meanwhile.)
	LocalPlatform::AvoidCpuList(std::getenv("KYTY_RENDER_CPUS"));
	auto& translator = *m_translators[0];
	// (Its Vulkan calls go to the driver as they are made, no recording stream: its recorder is its own.)
	RenderContext::SetThreadExecutors(&translator.executors);
	GuestGpu::SetSpeculationThread(true);
	Spec::t_packet_slot = translator.index;
	BcPar::t_reader     = true; // (buffercache-par: hilo lector del indice de BufferCache)
	std::unique_lock lock(m_detached_mutex);
	for (;;) {
		m_detached_job_cv.wait(lock, [&] { return m_detached_quit || !m_detached_jobs.empty(); });
		if (m_detached_quit) return;
		// (In the order handed: the GPU thread commits each queue's head; one handed later waits less.)
		auto job = std::move(m_detached_jobs.front());
		m_detached_jobs.pop_front();
		lock.unlock();
		// (Its start state is its own: Hand copied it.)
		const auto start  = NowNs();
		auto       result = Translate(translator, *job.start, job.commands, job.execution);
		result->serial    = job.serial;
		// (envio3) What the GPU thread did since it was handed is checked, not only since its translation began: the
		// log keeps it (m_in_flight). ENVIO2 alone kept the translation's start, as before.
		if (Envio3::Mode() != 0) result->effects_from = std::min(result->effects_from, job.effects_from);
		Envio2::Add(Envio2::g_totals.acb2_ns, NowNs() - start);
		lock.lock();
		m_detached_starts.push_back(std::move(job.start));
		m_detached_results.push_back(std::move(result));
		lock.unlock();
		m_detached_done_cv.notify_all();
		if (m_detached_wake) m_detached_wake(); // (the GPU thread may sleep with the submission blocked)
		lock.lock();
	}
}

bool Speculation::Hand(uint64_t serial, const CommandProcessor& cp, std::span<const uint32_t> commands,
                       const Pm4Execution& execution) {
	if (Full() || !DetachedOn()) return false;
	if (std::ranges::any_of(m_in_flight, [&](const Flying& f) { return f.serial == serial; })) return false;
	uint64_t effects_from = 0;
	{
		std::lock_guard lock(m_mutex);
		// (Nothing in flight: what the normal translation logged so far is needed by no result.)
		if (m_in_flight.empty()) m_log.DropBefore(m_log.End());
		effects_from = m_log.End();
	}
	Envio3::NoteDepth(m_in_flight.size() + 1);
	m_in_flight.push_back({serial, effects_from});
	{
		std::lock_guard lock(m_detached_mutex);
		DetachedJob job;
		if (!m_detached_starts.empty()) {
			job.start = std::move(m_detached_starts.back());
			m_detached_starts.pop_back();
		} else {
			job.start = std::make_unique<CommandProcessor>(m_renderer, m_queue_event_id);
		}
		job.start->CopyStateFrom(cp, false);
		job.serial       = serial;
		job.commands     = commands;
		job.execution    = execution;
		job.effects_from = effects_from;
		m_detached_jobs.push_back(std::move(job));
	}
	m_detached_job_cv.notify_one();
	return true;
}

void Speculation::Landed(uint64_t serial) {
	std::erase_if(m_in_flight, [&](const Flying& f) { return f.serial == serial; });
	// (What no result in flight began before is dropped.)
	std::lock_guard lock(m_mutex);
	auto            oldest = m_log.End();
	for (const auto& f: m_in_flight) oldest = std::min(oldest, f.effects_from);
	m_log.DropBefore(oldest);
}

Speculation::Outcome Speculation::CommitHanded(uint64_t serial, CommandProcessor& cp, Pm4Execution& execution,
                                               uint64_t& waited_ns) {
	EXIT_IF(!std::ranges::any_of(m_in_flight, [&](const Flying& f) { return f.serial == serial; }));
	std::unique_ptr<Result> result;
	{
		std::unique_lock lock(m_detached_mutex);
		waited_ns       = 0;
		const auto mine = [&] {
			return std::ranges::find_if(m_detached_results, [&](const auto& r) { return r->serial == serial; });
		};
		if (mine() == m_detached_results.end()) {
			const auto start = NowNs();
			m_detached_done_cv.wait(lock, [&] { return mine() != m_detached_results.end(); });
			waited_ns = NowNs() - start;
		}
		const auto found = mine();
		result           = std::move(*found);
		m_detached_results.erase(found);
	}
	const auto outcome = Commit(std::move(result), cp, execution);
	if (outcome == Outcome::Blocked) {
		m_detached_blocked.push_back(std::move(m_blocked)); // (still in flight: its effects are kept)
		return outcome;
	}
	Landed(serial);
	return outcome;
}

bool Speculation::BlockedHanded(uint64_t serial) const noexcept {
	return std::ranges::any_of(m_detached_blocked, [&](const auto& r) { return r->serial == serial; });
}

Speculation::Outcome Speculation::ResumeHanded(uint64_t serial, CommandProcessor& cp, Pm4Execution& execution) {
	const auto found = std::ranges::find_if(m_detached_blocked, [&](const auto& r) { return r->serial == serial; });
	EXIT_IF(found == m_detached_blocked.end());
	auto result = std::move(*found);
	m_detached_blocked.erase(found);
	const auto outcome = Continue(std::move(result), cp, execution);
	if (outcome == Outcome::Blocked) {
		m_detached_blocked.push_back(std::move(m_blocked));
		return outcome;
	}
	Landed(serial);
	return outcome;
}

Speculation::~Speculation() {
	if (m_detached.joinable()) {
		{
			std::lock_guard lock(m_detached_mutex);
			m_detached_quit = true;
		}
		m_detached_job_cv.notify_all();
		m_detached.join();
		for (auto& result: m_detached_results) m_renderer.GetCommandScheduler().DestroyRecorder(std::move(result->recorder));
		for (auto& result: m_detached_blocked) m_renderer.GetCommandScheduler().DestroyRecorder(std::move(result->recorder));
	}
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
	if (!m_done_state) m_done_state = std::make_unique<CommandProcessor>(m_renderer, 0);
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
	m_state->SetChunking(ChunkWorkSize());
	Pm4Execution execution;
	for (;;) {
		// A chunk from here: where it starts, and its record, which reads through the one before while that is not done.
		Chunk chunk;
		chunk.start = std::make_unique<CommandProcessor>(m_renderer, 0);
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

void Speculation::AheadRun(Translator& translator) {
	LocalPlatform::SetThreadName("Kyty.Speculation");
	// (Off the GPU thread's CPUs: it commits what they translate, and waits for it.)
	LocalPlatform::AvoidCpuList(std::getenv("KYTY_RENDER_CPUS"));
	// (Its Vulkan calls go to the driver as they are made, no recording stream: what they name is alive then.)
	RenderContext::SetThreadExecutors(&translator.executors);
	GuestGpu::SetSpeculationThread(true);
	Spec::t_packet_slot = translator.index;
	BcPar::t_reader     = true; // (buffercache-par: hilo lector del indice de BufferCache)
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
			const auto taken = end ? chunk : UINT32_MAX;
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
				++m_waits_for_ahead;
				const auto waited = NowNs();
				m_ahead_wake.wait(lock, [&] { return !translating(); });
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
		result->start_state = std::make_unique<CommandProcessor>(m_renderer, 0);
		result->end_state   = std::make_unique<CommandProcessor>(m_renderer, 0);
	}
	auto& r = *result;
	// (A speculation's thread records and ends its buffers; the GPU thread's go through its recording stream.)
	r.recorder->own_thread   = !GuestGpu::IsGpuThread();
	r.recorder->pending_tick = m_pending.fetch_add(1, std::memory_order_relaxed) + 1;
	auto& processor          = *translator.cp;
	processor.CopyStateFrom(cp, false); // (no packet it takes uses the constant RAM)
	r.start_state->CopyStateFrom(cp, false);
	r.serial = 0;
	r.pass   = 0;
	r.blind  = false;
	r.state.Clear();
	r.state.table_upload  = translator.table_upload.get();
	r.state.shader_upload = translator.shader_upload.get();
	{
		// (Linked under the lock: one committed or dropped since it was chosen is not; Finish unlinks the others.)
		std::lock_guard lock(m_mutex);
		r.retired = 0;
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
	RenderContext::SetThreadExecutors(&translator.executors);
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
			const bool journal = std::exchange(BcPar::t_journal, false);
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
			// (buffercache-par, KYTY_BC_PAR) Lee memoria que escribe el procesador en el segmento abierto: el segmento se
			// cierra antes del paquete (un corte sin paquete: el Commit hace las escrituras de la DCB de los segmentos
			// anteriores antes del trabajo del siguiente) y el paquete se traduce otra vez en el nuevo. Una vez por paquete; solo
			// draws/dispatches (los que ya serian hueco: misma suposicion de estado del procesador que RunHole).
			if (journal && IsHole(*packet) && !r.state.stop && BcPar::SplitOn() &&
			    (r.holes.empty() || r.holes.back().dwords != 0 || r.holes.back().packet != packet)) {
				if (r.hole_states.size() == r.holes.size()) r.hole_states.push_back(std::make_unique<CommandProcessor>(m_renderer, 0));
				r.hole_states[r.holes.size()]->CopyStateFrom(processor, false);
				auto& cut  = r.holes.emplace_back();
				cut.packet = packet;
				cut.dwords = 0; // (corte: ningun paquete en el Commit)
				cut.reason = "buffercache-par: corte por escrituras de la DCB";
				cut.after  = spec; // (el paquete otra vez, si el Commit para aqui)
				scheduler.EndRecording(*r.recorder);
				r.state.CloseSegment(r.recorder->recorded.size());
				r.segment_ends.push_back(NowNs());
				r.state.refused = nullptr;
				BcPar::g_splits.fetch_add(1, std::memory_order_relaxed);
				DrawStateObserver::Invalidate();
				translator.draw->NativeXprForgetState();
				processor.BufferInit();
				continue;
			}
			if (journal) BcPar::Refused(BcPar::JournalRefused);
			if (!IsHole(*packet) || r.state.stop) {
				r.stop = r.state.refused;
				break;
			}
			// A hole: the segment ends before it, the next begins after it, from the processor state there.
			if (r.hole_states.size() == r.holes.size()) r.hole_states.push_back(std::make_unique<CommandProcessor>(m_renderer, 0));
			r.hole_states[r.holes.size()]->CopyStateFrom(processor, false);
			auto& hole  = r.holes.emplace_back();
			hole.packet = packet;
			hole.dwords = dwords;
			hole.reason = r.state.refused;
			if (Spec::t_hole_hook != nullptr) Spec::t_hole_hook(packet, r.state.refused); // (tej: el hueco, contado)
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
	if (DetachedOn() && Envio3::Mode() != 0) { // (envio3: the ACB thread's time whose work went to the GPU)
		Envio2::Add(Envio3::g_totals.moved_ns, kept - r.start_ns);
		Envio2::Add(Envio3::g_totals.lost_ns, r.end_ns - kept);
	}
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
		// (envio2/envio3: the ACB thread's in flight, from when each was handed.)
		for (const auto& f: m_in_flight) oldest = std::min(oldest, f.effects_from);
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

// (spec5) The commit's "a surface cleared" stop by who cleared it (TextureCache::MetaClearOrigin).
static const char* ClearedWhy(uint8_t origin) {
	// (spec6: Spec::ClearOriginNow - bit 3 in a hole)
	static constexpr const char* names[16] = {
	    "commit: a surface cleared (other)",           "commit: a surface cleared (compute clear)",
	    "commit: a surface cleared (depth clear draw)", "commit: a surface cleared (CP fill)",
	    "commit: a surface cleared (DCC fill)",         "commit: a surface cleared (depth bind record)",
	    "commit: a surface cleared (?6)",               "commit: a surface cleared (?7)",
	    "commit: a surface cleared (other, in a hole)", "commit: a surface cleared (compute clear, in a hole)",
	    "commit: a surface cleared (depth clear draw, in a hole)", "commit: a surface cleared (CP fill, in a hole)",
	    "commit: a surface cleared (DCC fill, in a hole)", "commit: a surface cleared (depth bind record, in a hole)",
	    "commit: a surface cleared (?6, in a hole)",   "commit: a surface cleared (?7, in a hole)"};
	return names[origin & 15u];
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
		// (envio3) An image the work uses that a sweep found CPU-written since it was handed (logged as host writes by
		// GpuResourceManager::InvalidateCpuWrittenPages): its upload is the GPU thread's; the rest the normal way.
		// (spec5: and for the graphics speculation under KYTY_NATIVE_MEMORY, its sweeps logged the same way.)
		if (((DetachedOn() && Envio3::Mode() != 0) || PageManager::NativeWriteWatch()) && !m_effects.host_writes.Empty())
			for (const auto& use: state.closed[index].images)
				if (m_effects.host_writes.Intersects(use.image->info.data.address, use.image->info.data.size))
					return Envio2::Add(Envio3::g_totals.sweep_stops), ++m_stop_reasons["commit: CPU wrote an image (sweep)"],
					       Progress::Stopped;
		for (const auto& fact: state.meta_facts)
			if (!fact.cleared && fact.first <= index && index <= fact.last &&
			    m_renderer.GetTextureCache().IsMetaCleared(fact.address, fact.slice))
				return ++m_stop_reasons[ClearedWhy(m_renderer.GetTextureCache().MetaClearOrigin(fact.address))], Progress::Stopped;
		{
			const auto  from    = index == 0 ? Spec::Counts {} : state.closed[index - 1].end;
			const auto& to      = state.closed[index].end;
			const auto& effects = m_effects;
			// The guest memory it read on the CPU as no GPU work wrote it (and as it was): not written since.
			if (!effects.gpu_writes.Empty() || !effects.host_writes.Empty())
				for (size_t i = from.clean_reads; i < to.clean_reads; ++i) {
					const auto& read = state.clean_reads[i];
					if (effects.gpu_writes.Intersects(read.begin, read.end - read.begin) ||
					    effects.host_writes.Intersects(read.begin, read.end - read.begin))
						return ++m_stop_reasons[WroteWhatItRead(m_log.FirstWriteOver(result.effects_from, read.begin,
						                                                             read.end - read.begin))],
						       Progress::Stopped;
				}
			// The descriptor sets it binds: none retired since (one may be reused under it).
			for (const auto retired: effects.retired_sets)
				if (std::find(state.sets.begin() + static_cast<ptrdiff_t>(from.sets),
				              state.sets.begin() + static_cast<ptrdiff_t>(to.sets), retired) !=
				    state.sets.begin() + static_cast<ptrdiff_t>(to.sets))
					return ++m_stop_reasons["commit: a hole retired a set"], Progress::Stopped;
			if (!effects.gpu_writes.Empty())
				for (size_t i = from.cpu_copies; i < to.cpu_copies; ++i) {
					const auto& copy = state.cpu_copies[i];
					if (effects.gpu_writes.Intersects(copy.begin, copy.end - copy.begin))
						return ++m_stop_reasons["commit: GPU work wrote what it copies"], Progress::Stopped;
				}
			for (size_t i = from.chain_reads; i < to.chain_reads; ++i) {
				const auto& read = state.chain_reads[i];
				if (*reinterpret_cast<const uint32_t*>(read.address) != read.value)
					return ++m_stop_reasons["commit: a command buffer before wrote otherwise"], Progress::Stopped;
			}
			for (size_t i = from.waits; i < to.waits; ++i) {
				const auto& wait  = state.waits[i];
				uint64_t    value = 0;
				std::memcpy(&value, reinterpret_cast<const void*>(wait.address), wait.bytes);
				if (!TestWaitRegMemValue(value, wait.reference, wait.mask, wait.function))
					return ++m_stop_reasons["commit: a wait not satisfied"], Progress::Stopped;
			}
		}
		const auto prepared = CurrentEpochs();
		const auto prep_t0  = NowNs(); // (spec5: the report's main-thread split)
		Before(result, index);
		const auto prep_t1 = NowNs();
		m_prepare_ns += prep_t1 - prep_t0;
		// (Its preparation registers what the work did not find registered (BDA ranges): a buffer merged away under
		// the recorded work leaves it reading a destroyed one. The work after it is translated in order.)
		if (CurrentEpochs() != prepared && Changed(result, index) != nullptr)
			return ++m_stop_reasons["commit: preparation registered buffers"], Progress::Stopped;
		const auto& segment = state.closed[index];
		if (const auto tick = scheduler.SubmitRecorded(*result.recorder, segment.recorded - result.submitted); tick != 0)
			result.last_tick = tick;
		result.submitted = segment.recorded;
		const auto after_t0 = NowNs();
		m_submit_ns += after_t0 - prep_t1;
		After(result, index);
		m_after_ns += NowNs() - after_t0;
		++m_chunks_committed;
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
	if (!m_effects.host_writes.Empty())
		for (size_t i = from.buffer_ranges; i < to.buffer_ranges; ++i) {
			const auto& range = state.buffer_ranges[i];
			m_effects.host_writes.ForEachIntersection(range.begin, range.end - range.begin, [&](RangeSet::Range written) {
				buffers.EnsureBufferContents(written.address, written.size);
			});
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
	}
	if (!state.closed[index].images.empty()) MoveImageStateEpoch();
	for (size_t i = from.touched_images; i < to.touched_images; ++i)
		textures.TouchSpeculated(*state.touched_images[i].first, state.touched_images[i].second);
	for (size_t i = from.touched_buffers; i < to.touched_buffers; ++i) buffers.TouchSpeculated(state.touched_buffers[i]);
	for (size_t i = from.writes; i < to.writes; ++i) {
		const auto& write = state.writes[i];
		if (write.copy) {
			buffers.CopyGuestMemory(write.address, write.value, write.size);
		} else {
			std::memcpy(reinterpret_cast<void*>(write.address), &write.value, write.size);
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
	if (hole.dwords == 0) return HoleRun::Done; // (buffercache-par: corte sin paquete)
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
	std::printf("%s: %llu whole, %llu partly, %llu not committed; %llu segments, %llu holes (%llu waits blocked); "
	            "ahead: %llu being translated, %llu not yet; blind %llu (stopped %llu of %llu); per frame: queue %.2f ms, "
	            "translation kept %.2f ms, lost %.2f ms, commit %.2f ms (holes %.2f ms), waits %.2f ms; start delay %.2f ms; "
	            "translations %llu/%llu/%llu/%llu/%llu/%llu (<0.5/1/2/4/8 ms/more, longest %.1f ms)",
	            m_name, static_cast<unsigned long long>(m_whole), static_cast<unsigned long long>(m_partly),
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
	// (spec5) The GPU thread's time against the workers': what the speculation leaves on the main thread per frame
	// (its queue time: commits - preparation, submission, after, holes -, waits, and the normal translation), and the
	// segment count (per-segment delivery).
	std::printf("%s: main thread %.2f ms/frame (prepare %.2f, submit %.2f, after %.2f, holes %.2f, waits %.2f, rest %.2f), "
	            "workers %.2f ms/frame (kept %.2f, lost %.2f); %.1f segments/frame, chunk %u draws\n",
	            m_name, ms(m_queue_ns), ms(m_prepare_ns), ms(m_submit_ns), ms(m_after_ns), ms(m_hole_ns), ms(m_wait_ns),
	            ms(m_queue_ns > m_commit_ns + m_wait_ns ? m_queue_ns - m_commit_ns - m_wait_ns : 0),
	            ms(m_kept_ns + m_lost_ns), ms(m_kept_ns), ms(m_lost_ns), static_cast<double>(m_chunks_committed) / frames,
	            ChunkWorkSize());
	m_prepare_ns = m_submit_ns = m_after_ns = m_chunks_committed = 0;
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
	std::printf("\n");
	std::fflush(stdout);
	m_whole = m_partly = m_none = m_segments = m_holes_run = m_waits_blocked = 0;
}

} // namespace Libs::Graphics
