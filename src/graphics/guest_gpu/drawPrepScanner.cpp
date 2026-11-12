#include "graphics/guest_gpu/drawPrepScanner.h"

#include "common/cpuAffinity.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/speculation.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/drawPrep.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderDraw.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/agc.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <vector>

namespace Libs::Graphics {

namespace {

bool IsGraphicsDraw(uint32_t opcode) {
	return DrawPrep::CountsAsDrawOrDispatch(opcode) && opcode != Pm4::IT_DISPATCH_DIRECT &&
	       opcode != Pm4::IT_DISPATCH_INDIRECT;
}

// The register file a SET_*_REG packet writes, and whether the command processor's handler would
// take it: the handlers end the emulator on an offset they do not know, and the scanner reads each
// submission as it is queued, when its command stream can still hold garbage (GT7, while a menu
// waits: a SET_SH_REG with an offset past the SH file). Mirrors the handlers' checks.
bool RegisterPacketIsSane(uint32_t opcode, uint32_t header, const uint32_t* body) {
	constexpr uint32_t SelectorMask = 0x70000000u; // NormalizeRegisterOffset (pm4Handlers.cpp)
	const uint32_t     num_values   = KYTY_PM4_LEN(header) - 2u;
	const auto in_file = [&](uint32_t offset, uint32_t num, const auto& direct, const auto* indirect) {
		if (offset >= num) return false;
		if (direct[offset] != nullptr) return true;
		if (num_values == 0) return false;
		for (uint32_t i = 0; i < num_values; i++) {
			if (offset + i >= num || indirect[offset + i] == nullptr) return false;
		}
		return true;
	};
	switch (opcode) {
		case Pm4::IT_SET_CONTEXT_REG: {
			const auto offset = body[0] & ~SelectorMask;
			return offset == Pm4::CX_NOP ||
			       in_file(offset, Pm4::CX_NUM, g_hw_ctx_func, g_hw_ctx_indirect_func);
		}
		case Pm4::IT_SET_SH_REG: {
			const auto offset = body[0];
			if (offset == Pm4::SH_NOP) return true;
			if (!in_file(offset, Pm4::SH_NUM, g_hw_sh_func, g_hw_sh_indirect_func)) return false;
			// Round 112: every register the packet writes. The SH handlers are each set for every
			// register they take (the compute ones, a stage's user data) and end the emulator on
			// one past them ("unsupported compute SH register 0x22b", at a license test's retry
			// prompt); the generic one takes the registers that have an indirect handler.
			const auto direct = g_hw_sh_func[offset];
			if (direct == nullptr) return true;
			const bool generic = direct == g_hw_sh_func[Pm4::SPI_GRAPHICS_SHADER_CONTROL_PS];
			for (uint32_t i = 0; i < num_values; i++) {
				const auto reg = offset + i;
				if (reg >= Pm4::SH_NUM) return false;
				if (generic ? g_hw_sh_indirect_func[reg] == nullptr : g_hw_sh_func[reg] != direct) {
					return false;
				}
			}
			return true;
		}
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX: {
			if (Gen5::AgcIsInternalDataPacket(header, body)) return true;
			const auto offset = opcode == Pm4::IT_SET_UCONFIG_REG_INDEX ? body[0] & 0x0fffffffu
			                                                             : body[0] & ~SelectorMask;
			return offset == Pm4::UC_NOP ||
			       in_file(offset, Pm4::UC_NUM, g_hw_uc_func, g_hw_uc_indirect_func);
		}
		default: return true;
	}
}

// KYTY_DRAW_PREP_WORKERS (default 2, at most 8): the threads that run the scanner's walks; 0 runs
// them on the scanner thread, as before round 73.
uint32_t WorkerCount() {
	static const uint32_t count = [] {
		const char* value = std::getenv("KYTY_DRAW_PREP_WORKERS");
		long        n     = value != nullptr ? std::strtol(value, nullptr, 10) : 2l;
		// gt7-spec-1112: KYTY_SPECULATE=<n> asks for n walk workers at least.
		n = std::max(n, static_cast<long>(Speculation::Threads()));
		return static_cast<uint32_t>(std::clamp(n, 0l, 8l));
	}();
	return count;
}

// Jobs waiting for a worker, at most: past it the scanner waits (it is far enough ahead).
constexpr size_t MaxJobs = 128;

// Round 111: the scanner walks its own copy of each command stream, read through the guest
// memory's backing (TryReadBacking, which a racing unmap cannot fault). The streams are guest
// memory the game may be freeing or reusing as the scanner reads them (a submission the GPU
// thread has already run, an indirect buffer it called): read in place, one that went away ended
// the emulator in the scanner thread (an access violation, at a license test's retry prompt), and
// a packet could change between its check and its handler (round 109). As the scanner's other
// reads, only bytes no GPU work wrote (the backing holds their older value). False when the range
// cannot be read so.
bool CopyStream(uint64_t address, uint64_t dwords, std::vector<uint32_t>& words) {
	words.resize(static_cast<size_t>(dwords));
	return dwords == 0 || LibKernel::Memory::TryReadGpuCleanBacking(address, words.data(),
	                                                                dwords * sizeof(uint32_t));
}

// Indirect buffers called inside one another, at most (a garbage or looping call would otherwise
// copy a buffer a level without end).
constexpr size_t MaxScanDepth = 8;

void LogScanStopped(const char* what, uint32_t header, uint64_t value) {
	static std::atomic<uint32_t> reports {0};
	const auto                   report = reports.fetch_add(1, std::memory_order_relaxed);
	if (report < 16u || (report & (report + 1u)) == 0u) {
		LOGF("DrawPrepScanner: stopped at %s (header 0x%08" PRIx32 ", 0x%016" PRIx64
		     "); the GPU thread prepares the rest itself (%" PRIu32 " so far)\n",
		     what, header, value, report + 1u);
	}
}

} // namespace

DrawPrepScanner::DrawPrepScanner(RenderContext& renderer, const char* name): m_renderer(renderer) {
	const auto workers = WorkerCount();
	LOGF("DrawPrep: the %s scanner's walks run on %u worker thread%s (KYTY_DRAW_PREP_WORKERS)\n",
	     name, workers, workers == 1 ? "" : "s");
	for (uint32_t i = 0; i < workers; i++) {
		m_workers.emplace_back([this] { WorkerRun(); });
	}
	m_thread = std::thread([this] { Run(); });
}

DrawPrepScanner::QueueState& DrawPrepScanner::State(uint32_t queue) {
	auto& state = m_queues[queue % m_queues.size()];
	if (state == nullptr) {
		state            = std::make_unique<QueueState>();
		state->cp        = std::make_unique<CommandProcessor>(m_renderer, queue == 0 ? 0 : 0x1f + static_cast<int>(queue));
		state->saved_ctx = std::make_unique<HW::Context>();
		// As GuestGpu: the graphics processor starts reset, a compute one as constructed.
		if (queue == 0) {
			state->cp->Reset();
		}
	}
	return *state;
}

DrawPrepScanner::~DrawPrepScanner() {
	{
		std::lock_guard lock(m_mutex);
		m_stop = true;
	}
	{
		std::lock_guard lock(m_job_mutex);
		m_job_stop = true;
	}
	m_stopping.store(true, std::memory_order_relaxed);
	m_work.notify_one();
	m_job_ready.notify_all();
	m_job_space.notify_all();
	if (m_thread.joinable()) {
		m_thread.join();
	}
	for (auto& worker: m_workers) {
		if (worker.joinable()) {
			worker.join();
		}
	}
}

void DrawPrepScanner::WorkerRun() {
	KYTY_PROFILER_THREAD("Thread_DrawPrepWalk");
	Common::RegisterGpuSideThread("Thread_DrawPrepWalk");
	// As the scanner thread: clean-backing reads that check the GPU-dirty ranges themselves.
	LibKernel::Memory::SetGpuReadDelegate(true);
	for (;;) {
		Job job;
		{
			std::unique_lock lock(m_job_mutex);
			m_job_ready.wait(lock, [this] { return m_job_stop || !m_jobs.empty(); });
			if (m_job_stop) {
				return;
			}
			job = std::move(m_jobs.front());
			m_jobs.pop_front();
		}
		m_job_space.notify_one();
		Walk(job);
	}
}

void DrawPrepScanner::Dispatch(const DrawPrep::Key&                 key,
                               const PipelineCache::DrawPrepInputs& inputs) {
	Job job;
	job.index  = m_next_index++;
	job.key    = key;
	job.inputs = inputs;
	if (m_workers.empty() || Bisect::ScanOneThread()) {
		Walk(job);
		return;
	}
	{
		std::unique_lock lock(m_job_mutex);
		m_job_space.wait(lock, [this] { return m_job_stop || m_jobs.size() < MaxJobs; });
		if (m_job_stop) {
			return;
		}
		m_jobs.push_back(std::move(job));
	}
	m_job_ready.notify_one();
}

void DrawPrepScanner::Walk(Job& job) {
	// The GPU thread may have passed the packet while the job waited.
	if (!DrawPrep::GpuProgress(job.key.queue).Before(job.key)) {
		DrawPrep::Add(DrawPrep::Counter::SkipBehind);
		Finish(job.index, nullptr);
		return;
	}
	auto result = DrawPrep::Recycle();
	result->key = job.key;
	if (m_renderer.GetPipelineCache().DrawPrepWalk(job.inputs, *result)) {
		Finish(job.index, std::move(result));
	} else {
		DrawPrep::Discard(std::move(result));
		Finish(job.index, nullptr);
	}
}

void DrawPrepScanner::Finish(uint64_t index, std::unique_ptr<DrawPrep::PacketResult> result) {
	// Going away: nothing more is published (Publish can wait while the GPU thread is gone).
	if (m_stopping.load(std::memory_order_relaxed)) {
		return;
	}
	std::lock_guard lock(m_order_mutex);
	if (index != m_next_publish) {
		m_done.emplace(index, std::move(result));
		return;
	}
	DrawPrep::Publish(std::move(result));
	m_next_publish++;
	for (auto next = m_done.begin(); next != m_done.end() && next->first == m_next_publish;
	     next      = m_done.erase(next)) {
		DrawPrep::Publish(std::move(next->second));
		m_next_publish++;
	}
}

void DrawPrepScanner::EnqueueSubmission(uint32_t queue, uint64_t seq,
                                        std::span<const uint32_t> commands) {
	{
		std::lock_guard lock(m_mutex);
		m_items.push_back({queue, seq, commands});
	}
	m_work.notify_one();
}

void DrawPrepScanner::EnqueueReset(uint32_t queue) {
	{
		std::lock_guard lock(m_mutex);
		m_items.push_back({queue, 0, {}});
	}
	m_work.notify_one();
}

void DrawPrepScanner::Run() {
	KYTY_PROFILER_THREAD("Thread_DrawPrep");
	Common::RegisterGpuSideThread("Thread_DrawPrep");
	// The clean-backing reads (TryReadGpuCleanBacking) answer "maybe GPU-written" on any other
	// thread; a read delegate checks the buffer cache's GPU-dirty ranges under their shared lock,
	// which the GPU thread holds exclusively whenever it changes them.
	LibKernel::Memory::SetGpuReadDelegate(true);
	for (;;) {
		Item item;
		{
			std::unique_lock lock(m_mutex);
			m_work.wait(lock, [this] { return m_stop || !m_items.empty(); });
			if (m_stop) {
				return;
			}
			item = m_items.front();
			m_items.pop_front();
		}
		m_queue = item.queue;
		m_state = &State(item.queue);
		DrawPrep::SetScannerQueue(item.queue);
		if (item.seq == 0) {
			// The GPU thread resets its command processor at a suspend point.
			m_state->cp->Reset();
			m_state->pushed  = false;
			m_state->tainted = false;
		} else {
			Scan(item);
		}
		DrawPrep::LogCountersIfDue();
	}
}

// As CommandProcessor::ApplyContextStateOperation, without its exits: a sequence the scanner
// cannot follow only makes its register copy unreliable.
void DrawPrepScanner::ApplyContextState(uint32_t operation) {
	auto&      ctx        = m_state->cp->GetCtx();
	const bool was_tainted = m_state->tainted;
	struct Count {
		const bool& was;
		const bool& now;
		~Count() {
			if (!was && now) {
				DrawPrep::Add(DrawPrep::Counter::TaintContext);
			}
		}
	} count {was_tainted, m_state->tainted};
	switch (static_cast<ContextStateOperation>(operation)) {
		case ContextStateOperation::Clear: ctx.Reset(); break;
		case ContextStateOperation::Push:
			m_state->tainted |= m_state->pushed;
			*m_state->saved_ctx = ctx;
			m_state->pushed     = true;
			break;
		case ContextStateOperation::Pop:
			m_state->tainted |= !m_state->pushed;
			ctx      = *m_state->saved_ctx;
			m_state->pushed = false;
			break;
		case ContextStateOperation::PushClear:
			m_state->tainted |= m_state->pushed;
			*m_state->saved_ctx = ctx;
			m_state->pushed     = true;
			ctx.Reset();
			break;
		default: m_state->tainted = true; break;
	}
}

// SET_*_REG_INDIRECT: the register pairs are read with the clean backing read (no fault, no wait)
// and handed to the command processor's own handler from a local copy. False when the pairs
// cannot be read that way (the GPU wrote them): the register copy is then unreliable.
bool DrawPrepScanner::ApplyRegisterIndirect(uint32_t header, const uint32_t* body) {
	if (KYTY_PM4_LEN(header) != 5u) {
		return false;
	}
	const uint64_t address =
	    (static_cast<uint64_t>(body[0]) & 0xfffffffcu) | (static_cast<uint64_t>(body[1]) << 32u);
	const uint32_t count = body[3] & 0x3fffu;
	if (count == 0) {
		return true;
	}
	if (address == 0) {
		return false;
	}
	thread_local std::vector<uint32_t> pairs;
	pairs.resize(static_cast<size_t>(count) * 2u);
	if (!LibKernel::Memory::TryReadGpuCleanBacking(address, pairs.data(),
	                                               pairs.size() * sizeof(uint32_t))) {
		return false;
	}
	const auto                    local = reinterpret_cast<uint64_t>(pairs.data());
	const std::array<uint32_t, 4> copy {static_cast<uint32_t>(local), static_cast<uint32_t>(local >> 32u),
	                                    body[2], body[3]};
	const auto opcode = (header >> 8u) & 0xffu;
	(void)g_cp_op_func[opcode](*m_state->cp, header, copy.data(), 5u, 5u);
	return true;
}

void DrawPrepScanner::ScanDraw(uint64_t seq, uint32_t ordinal, const uint32_t* guest_packet) {
	// KYTY_BISECT's drawprep:off: the GPU thread walks every draw itself.
	if (Bisect::DrawPrepOff()) {
		return;
	}
	if (m_state->tainted) {
		DrawPrep::Add(DrawPrep::Counter::Tainted);
		return;
	}
	const DrawPrep::Key key {
	    .queue = m_queue, .seq = seq, .ordinal = ordinal, .packet = guest_packet};
	if (!DrawPrep::GpuProgress(m_queue).Before(key)) {
		DrawPrep::Add(DrawPrep::Counter::SkipBehind);
		return;
	}
	const auto& ctx  = m_state->cp->GetCtx();
	const auto& sh   = m_state->cp->GetShCtx();
	const auto& ucfg = m_state->cp->GetUcfg();
	if (!DrawPrepHasVertexShader(sh)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
		return;
	}
	std::array<Prospero::ColorComponentMapping, 8> mapping {};
	const bool ps_active = DrawPrepPixelState(ctx, sh, mapping);
	thread_local PipelineCache::DrawPrepInputs     inputs;
	if (m_renderer.GetPipelineCache().DrawPrepDescribeGraphics(sh.GetVs(), sh.GetPs(),
	                                                           ctx.GetShaderRegisters(), ctx, ucfg,
	                                                           mapping, ps_active, inputs)) {
		Dispatch(key, inputs);
	}
}

// A dispatch: the compute inputs RenderExecutor::DispatchDirect / DispatchIndirect build.
void DrawPrepScanner::ScanDispatch(uint64_t seq, uint32_t ordinal, const uint32_t* packet,
                                   const uint32_t* guest_packet, uint32_t opcode) {
	if (Bisect::DrawPrepOff()) {
		return;
	}
	if (m_state->tainted) {
		DrawPrep::Add(DrawPrep::Counter::Tainted);
		return;
	}
	const DrawPrep::Key key {
	    .queue = m_queue, .seq = seq, .ordinal = ordinal, .packet = guest_packet};
	if (!DrawPrep::GpuProgress(m_queue).Before(key)) {
		DrawPrep::Add(DrawPrep::Counter::SkipBehind);
		return;
	}
	// As CommandProcessor::DispatchDirect / DispatchIndirect: the dispatch initiator selects the
	// wave size, which is part of the program key. Before round 82 the scanner kept the default,
	// so a wave32 dispatch never found what it prepared (from the Frank fork, ed26428d).
	// KYTY_BISECT's scanwave:old.
	if (!Bisect::ScanWaveOld()) {
		uint32_t mode = 0;
		if (opcode == Pm4::IT_DISPATCH_DIRECT) {
			mode = KYTY_PM4_LEN(packet[0]) == 5u ? packet[4] : 0u;
		} else {
			mode = packet[0] == 0xc0021600u ? packet[3] : packet[2];
		}
		m_state->cp->GetShCtx().SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	}
	const auto& cs = m_state->cp->GetShCtx().GetCs();
	if (cs.cs_regs.data_addr == 0 || PipelineCache::IsSkipListed(cs.cs_regs.data_addr)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
		return;
	}
	ShaderComputeInputInfo input_info {};
	if (opcode == Pm4::IT_DISPATCH_DIRECT) {
		if (KYTY_PM4_LEN(packet[0]) != 5u) {
			DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
			return;
		}
		const uint32_t groups[3] = {packet[1], packet[2], packet[3]};
		const uint32_t mode      = packet[4];
		if (groups[0] == 0 || groups[1] == 0 || groups[2] == 0) {
			DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
			return;
		}
		const bool use_thread_dimensions =
		    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
		input_info.dispatch_thread_dimensions = use_thread_dimensions;
		const uint32_t group_sizes[3] = {cs.cs_regs.num_thread_x, cs.cs_regs.num_thread_y,
		                                 cs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			input_info.workgroup_counts[axis] = groups[axis];
			if (use_thread_dimensions) {
				const auto size = std::max(group_sizes[axis], 1u);
				input_info.workgroup_counts[axis] = groups[axis] / size + (groups[axis] % size != 0u);
			}
		}
	} else {
		const uint32_t mode = packet[0] == 0xc0021600u ? packet[3] : packet[2];
		const bool     use_thread_dimensions =
		    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
		input_info.dispatch_thread_dimensions   = use_thread_dimensions;
		input_info.dispatch_dimensions_indirect = use_thread_dimensions;
	}
	thread_local PipelineCache::DrawPrepInputs inputs;
	if (m_renderer.GetPipelineCache().DrawPrepDescribeCompute(
	        cs, m_state->cp->GetCtx().GetShaderRegisters(), input_info, inputs)) {
		Dispatch(key, inputs);
	}
}

// The packet walk of CommandProcessor::ProcessPm4, for registers only: register writes go to the
// scanner's command processor, draws are prepared, everything else is stepped over.
void DrawPrepScanner::Scan(const Item& item) {
	struct Cursor {
		std::span<const uint32_t> commands;         // the scanner's copy (CopyStream)
		const uint32_t*           guest  = nullptr; // where the stream is in guest memory
		uint32_t                  offset = 0;
	};
	// One copy a nesting level (a chained indirect buffer replaces its level's). Moving the outer
	// vector keeps each copy's storage.
	thread_local std::vector<std::vector<uint32_t>> copies;
	const auto copy = [&](size_t level, const uint32_t* guest, uint64_t dwords, Cursor& cursor) {
		if (copies.size() <= level) {
			copies.resize(level + 1);
		}
		auto& words = copies[level];
		if (!CopyStream(reinterpret_cast<uint64_t>(guest), dwords, words)) {
			return false;
		}
		// Round 112: a copy made after the GPU thread had run a later submission's draw may hold
		// what the game wrote over this one's memory once the GPU was done with it (garbage
		// packets, which the handlers end the emulator on).
		std::atomic_thread_fence(std::memory_order_acquire);
		if (DrawPrep::GpuProgress(m_queue).seq > item.seq) {
			return false;
		}
		cursor = {std::span<const uint32_t>(words.data(), words.size()), guest};
		return true;
	};
	const auto unreadable = [&](uint32_t header, const uint32_t* guest) {
		// Its registers are unknown here. Round 170: the submissions after it are prepared anyway,
		// as after register pairs the GPU wrote: a result is used only when its program entry,
		// shader address, user data, compute sizes and every byte its walk read are the GPU
		// thread's (TakeScanned). Before, the queue's register copy counted as unreliable up to
		// its next suspend point, and a compute queue the scanner fell behind on once went
		// unprepared for the rest of it (scan:oldtaint).
		if (Bisect::ScanOldTaint()) {
			m_state->tainted = true;
		}
		DrawPrep::Add(DrawPrep::Counter::TaintIndirect);
		LogScanStopped("a command buffer it could not copy, or one the GPU thread had passed",
		               header, reinterpret_cast<uint64_t>(guest));
	};
	std::vector<Cursor> stack(1);
	if (!copy(0, item.commands.data(), item.commands.size(), stack[0])) {
		unreadable(0, item.commands.data());
		return;
	}
	uint32_t ordinal = 0;
	bool     prepare = true; // false after a COND_EXEC it could not follow
	while (!stack.empty()) {
		auto& cursor = stack.back();
		if (cursor.offset >= cursor.commands.size()) {
			stack.pop_back();
			continue;
		}
		const auto* const packet    = cursor.commands.data() + cursor.offset;
		const auto* const guest     = cursor.guest + cursor.offset; // the GPU thread's key
		const auto        total     = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining = total - cursor.offset;
		const auto        header    = packet[0];
		if (header == 0x80000000u) {
			cursor.offset++;
			continue;
		}
		if (remaining < 2) {
			return;
		}
		if ((header >> 30u) == 0u) {
			const auto length = ((header >> 16u) & 0x3fffu) + 2u;
			if (length > remaining) {
				return;
			}
			cursor.offset += length;
			continue;
		}
		const auto opcode = (header >> 8u) & 0xffu;
		const auto length = KYTY_PM4_LEN(header);
		if (length > remaining) {
			return;
		}
		if (DrawPrep::CountsAsDrawOrDispatch(opcode)) {
			const auto this_ordinal = ordinal++;
			DrawPrep::Add(DrawPrep::Counter::Scanned);
			if (!prepare) {
				// After a COND_EXEC the scanner could not follow (its packet numbers may no
				// longer be the GPU thread's).
			} else if (IsGraphicsDraw(opcode)) {
				ScanDraw(item.seq, this_ordinal, guest);
			} else {
				ScanDispatch(item.seq, this_ordinal, packet, guest, opcode);
			}
			cursor.offset += length;
			continue;
		}
		const auto command = header & ~1u;
		switch (opcode) {
			case Pm4::IT_SET_CONTEXT_REG:
			case Pm4::IT_SET_SH_REG:
			case Pm4::IT_SET_UCONFIG_REG:
			case Pm4::IT_SET_UCONFIG_REG_INDEX: {
				// The stream is the scanner's copy (CopyStream): the packet checked is the one
				// applied. Before round 109 one the game changed between the check and the handler
				// ended the emulator in the handler ("cmd_offset >= Pm4::SH_NUM").
				if (length < 2u || !RegisterPacketIsSane(opcode, command, packet + 1)) {
					// Garbage in the stream: the registers after it are unknown.
					m_state->tainted = true;
					DrawPrep::Add(DrawPrep::Counter::TaintIndirect);
					LogScanStopped("an unknown register", header, packet[1]);
					return;
				}
				const auto consumed =
				    g_cp_op_func[opcode](*m_state->cp, command, packet + 1, length, length) + 1u;
				cursor.offset += consumed;
				continue;
			}
			case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
			case Pm4::IT_SET_SH_REG_INDIRECT:
			case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
				if (!ApplyRegisterIndirect(command, packet + 1)) {
					// Register pairs the GPU wrote: the registers they set keep their old values
					// here. The draws after them are prepared anyway: the GPU thread uses a result
					// only when its program entry (the static state), shader address, user data,
					// compute sizes and every byte its walk read are the same as its own
					// (TakeScanned), which is all a walk depends on; one prepared from a stale
					// register is not used, as before. Before round 70 the rest of the queue up to
					// the next suspend point went unprepared (~90 draws a frame at Tokyo, round
					// 69); KYTY_BISECT's drawprep:taint does that again.
					if (Bisect::DrawPrepTaint()) {
						m_state->tainted = true;
					}
					DrawPrep::Add(DrawPrep::Counter::TaintIndirect);
				}
				break;
			case Pm4::IT_COND_EXEC: {
				// As CpOpCondExec: the next exec_count dwords run only when the word at the
				// address is not zero. The draws in them were counted here and not on the GPU
				// thread, whose packet numbers then fell behind the scanner's for the rest of the
				// submission: none of its draws found its result after a COND_EXEC that skipped
				// (round 73). The word is read as the GPU thread will read it, when no GPU work
				// wrote it. When it cannot be read so, or the packet is predicated (the GPU thread
				// may step over it and run the dwords anyway), which packets the GPU thread meets
				// is unknown here: nothing more of the submission is prepared, and its register
				// writes are followed as if the dwords ran.
				DrawPrep::Add(DrawPrep::Counter::ScanCondExec);
				if (length < 5u) {
					DrawPrep::Add(DrawPrep::Counter::ScanCondStop);
					return;
				}
				const uint64_t address =
				    (static_cast<uint64_t>(packet[2]) << 32u) | (packet[1] & 0xfffffffcu);
				const uint32_t exec_count = packet[4] & 0x3fffu;
				uint32_t       value      = 0;
				if ((header & 1u) != 0 || (packet[1] & 0x3u) != 0 || packet[3] != 0 ||
				    address == 0 || exec_count > remaining - length ||
				    !LibKernel::Memory::TryReadGpuCleanBacking(address, &value, sizeof(value))) {
					DrawPrep::Add(DrawPrep::Counter::ScanCondStop);
					prepare = false;
					cursor.offset += length;
					continue;
				}
				if (value == 0) {
					DrawPrep::Add(DrawPrep::Counter::ScanCondSkip);
					cursor.offset += length + exec_count;
				} else {
					cursor.offset += length;
				}
				continue;
			}
			case Pm4::IT_CLEAR_STATE: m_state->cp->GetCtx().Reset(); break;
			case Pm4::IT_INDIRECT_BUFFER: {
				if (length != 4u) {
					// A conditional branch on guest memory: the scanner cannot follow it.
					return;
				}
				const auto* sub = reinterpret_cast<const uint32_t*>(
				    packet[1] | (static_cast<uint64_t>(packet[2]) << 32u));
				const uint32_t control = packet[3];
				const uint32_t count   = control & 0xfffffu;
				cursor.offset += length;
				if (count != 0 && sub != nullptr &&
				    !LibKernel::Memory::IsCommittedRange(reinterpret_cast<uint64_t>(sub),
				                                         uint64_t {count} * sizeof(uint32_t))) {
					// Not guest memory (yet): following it read an unmapped address.
					m_state->tainted = true;
					DrawPrep::Add(DrawPrep::Counter::TaintIndirect);
					LogScanStopped("an indirect buffer outside guest memory", header,
					               reinterpret_cast<uint64_t>(sub));
					return;
				}
				if (count != 0 && sub != nullptr) {
					// A chained buffer replaces this level's stream, a called one is a level more.
					// Neither uses `packet` or `cursor` after (the copy may move them).
					const bool chained = (control & (1u << 20u)) != 0;
					const auto level   = chained ? stack.size() - 1 : stack.size();
					Cursor     next;
					if (level >= MaxScanDepth) {
						m_state->tainted = true;
						DrawPrep::Add(DrawPrep::Counter::TaintIndirect);
						LogScanStopped("indirect buffers called too deep", header,
						               reinterpret_cast<uint64_t>(sub));
						return;
					}
					if (!copy(level, sub, count, next)) {
						unreadable(header, sub);
						return;
					}
					if (chained) {
						stack.back() = next;
					} else {
						stack.push_back(next);
					}
				}
				continue;
			}
			case Pm4::IT_NOP: {
				const auto r = KYTY_PM4_R(header);
				if (r == Pm4::R_ZERO) {
					if ((packet[1] & 0xffff0000u) == 0x68750000u) {
						const auto id = packet[1] & 0xfffu;
						if (id == 0x4u || id == 0xdu) {
							(void)g_cp_op_func[Pm4::IT_NOP](*m_state->cp, command, packet + 1, remaining,
							                                total);
						}
					}
				} else if (r == Pm4::R_CONTEXT_STATE) {
					ApplyContextState(packet[1]);
				} else if (r == Pm4::R_DISPATCH_RESET) {
					m_state->cp->Reset();
					m_state->pushed = false;
				}
				break;
			}
			default: break;
		}
		cursor.offset += length;
	}
}

} // namespace Libs::Graphics
