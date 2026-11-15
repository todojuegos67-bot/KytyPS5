#include "graphics/guest_gpu/computeAhead.h"

#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include "common/assert.h"
#include "common/logging/log.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <vector>

namespace Libs::Graphics {

bool ComputeAheadScanner::Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_COMPUTE_AHEAD");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

ComputeAheadScanner::ComputeAheadScanner(RenderContext& renderer): m_renderer(renderer) {
	m_thread = std::thread([this] { Run(); });
}

ComputeAheadScanner::~ComputeAheadScanner() {
	{
		std::lock_guard lock(m_mutex);
		m_stop = true;
	}
	m_work.notify_one();
	if (m_thread.joinable()) m_thread.join();
}

void ComputeAheadScanner::Enqueue(uint32_t queue, uint64_t seq, std::span<const uint32_t> commands) {
	{
		std::lock_guard lock(m_mutex);
		m_items.push_back({queue, seq, commands});
	}
	m_work.notify_one();
}

void ComputeAheadScanner::Reset() {
	{
		std::lock_guard lock(m_mutex);
		m_items.push_back({});
	}
	m_work.notify_one();
}

ComputeAheadScanner::QueueState& ComputeAheadScanner::State(uint32_t queue) {
	auto& state = m_queues[queue % m_queues.size()];
	if (state == nullptr) {
		state            = std::make_unique<QueueState>();
		// As GuestGpu::GetProcessor: compute queue n raises interrupts as 0x20 + n.
		state->cp        = std::make_unique<CommandProcessor>(m_renderer, queue == 0 ? 0 : 0x1f + static_cast<int>(queue));
		state->saved_ctx = std::make_unique<HW::Context>();
		state->cp->Reset();
	}
	return *state;
}

void ComputeAheadScanner::Run() {
	for (;;) {
		Item item;
		{
			std::unique_lock lock(m_mutex);
			m_work.wait(lock, [this] { return m_stop || !m_items.empty(); });
			if (m_stop) return;
			item = m_items.front();
			m_items.pop_front();
		}
		if (item.seq == 0) {
			auto& state = State(0);
			state.cp->Reset();
			state.pushed  = false;
			state.tainted = false;
		} else {
			// A packet the scanner's copy cannot take must not end the game: that queue's register
			// copy is untrusted from there on.
			try {
				Common::ScopedExitThrows throws;
				Scan(item);
			} catch (const Common::ExitException& error) {
				State(item.queue).tainted = true;
				if (static uint32_t logged = 0; logged++ < 20) LOGF("Compute ahead: scan stopped: %s\n", error.message.c_str());
			}
		}
	}
}

// As CommandProcessor::ApplyContextStateOperation, without its exits: a sequence the scanner
// cannot follow only makes its register copy unreliable.
void ComputeAheadScanner::ApplyContextState(QueueState& state, uint32_t operation) {
	auto& ctx = state.cp->GetCtx();
	switch (static_cast<ContextStateOperation>(operation)) {
		case ContextStateOperation::Clear: ctx.Reset(); break;
		case ContextStateOperation::Push:
			state.tainted |= state.pushed;
			*state.saved_ctx = ctx;
			state.pushed     = true;
			break;
		case ContextStateOperation::Pop:
			state.tainted |= !state.pushed;
			ctx          = *state.saved_ctx;
			state.pushed = false;
			break;
		case ContextStateOperation::PushClear:
			state.tainted |= state.pushed;
			*state.saved_ctx = ctx;
			state.pushed     = true;
			ctx.Reset();
			break;
		default: state.tainted = true; break;
	}
}

// The compute inputs RenderExecutor::DispatchDirect / DispatchIndirect build.
void ComputeAheadScanner::ScanDispatch(QueueState& state, const uint32_t* packet, uint32_t opcode) {
	const auto& cs = state.cp->GetShCtx().GetCs();
	// Also from a register copy the scanner could not follow all the way (as the old fork's
	// f27a4bcf): a dispatch's registers are mostly written right before it, and a wrong guess
	// only costs a worker's time. Only for a shader the game has created.
	if (cs.cs_regs.data_addr == 0 || !ShaderIsMapped(cs.cs_regs.data_addr)) return;
	ShaderComputeInputInfo input_info {};
	input_info.async_compute = state.cp->IsAsyncComputeQueue();
	const uint32_t mode = opcode == Pm4::IT_DISPATCH_DIRECT ? packet[4]
	                      : packet[0] == 0xc0021600u      ? packet[3]
	                                                      : packet[2];
	state.cp->GetShCtx().SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	if (opcode == Pm4::IT_DISPATCH_DIRECT) {
		const bool use_thread_dimensions = (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
		input_info.dispatch_thread_dimensions = use_thread_dimensions;
		if (KYTY_PM4_LEN(packet[0]) != 5u || packet[1] == 0 || packet[2] == 0 || packet[3] == 0) return;
		const uint32_t sizes[3] = {cs.cs_regs.num_thread_x, cs.cs_regs.num_thread_y, cs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto groups = packet[1 + axis];
			const auto size   = std::max(sizes[axis], 1u);
			input_info.workgroup_counts[axis] = use_thread_dimensions ? groups / size + (groups % size != 0u) : groups;
		}
	} else {
		input_info.indirect_args_address = packet[0] == 0xc0021600u
		                                       ? packet[1] | (static_cast<uint64_t>(packet[2]) << 32u)
		                                       : state.cp->GetDispatchIndirectArgsBaseAddress() + packet[1];
	}
	if (Behind()) return;
	m_renderer.GetPipelineCache().BuildComputeAhead(cs, state.cp->GetCtx().GetShaderRegisters(), input_info);
}

// A draw's pixel shader, from registers the scanner followed all the way: a graphics pipeline needs
// every one of them right to be the one the GPU thread builds.
void ComputeAheadScanner::ScanDraw(QueueState& state, const uint32_t* packet, uint32_t opcode) {
	if (state.tainted || Behind()) return;
	auto&                                          ctx = state.cp->GetCtx();
	auto&                                          sh  = state.cp->GetShCtx();
	std::array<Prospero::ColorComponentMapping, 8> mapping;
	if (!DrawPixelExports(ctx, sh, mapping)) return;
	// An indirect draw's vertex program reads its record's bases from the user data its packet patches, as the GPU
	// thread draws it (CommandProcessor::DrawIndirectMulti): another program, keyed apart from a direct draw's.
	ShaderIndirectOffsets offsets;
	bool                  indirect = false;
	if (opcode == Pm4::IT_DRAW_INDIRECT || opcode == Pm4::IT_DRAW_INDEX_INDIRECT ||
	    opcode == Pm4::IT_DRAW_INDIRECT_MULTI || opcode == Pm4::IT_DRAW_INDEX_INDIRECT_MULTI) {
		indirect = IndirectDrawOffsets(IndirectDrawPacketRegisters(packet[0] & ~1u, packet + 1), offsets);
		if (!indirect) offsets = {};
	}
	m_renderer.GetPipelineCache().BuildGraphicsAhead(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx,
	                                                 state.cp->GetUcfg(), mapping, offsets, indirect);
}

// The scanner reads a submission after the GPU thread may have passed it and the game reused its memory, so an address
// in a packet can be anything. Following a wild one would end the game (Marvel's Wolverine, once in a Trask run: a
// register list at 0x22fed760e0d5bc24). A range that isn't guest memory leaves the queue's registers untrusted.
bool ComputeAheadScanner::Followable(QueueState& state, uint64_t address, uint64_t bytes) {
	if (LibKernel::Memory::IsCommittedRange(address, bytes)) return true;
	state.tainted = true;
	if (static uint32_t logged = 0; logged++ < 20) {
		LOGF("Compute ahead: not following 0x%016" PRIx64 " (%" PRIu64 " bytes): not guest memory\n", address, bytes);
	}
	return false;
}

// The packet walk of CommandProcessor::ProcessPm4, for registers only: register writes go to the
// scanner's command processor, dispatches are built ahead, everything else is stepped over.
void ComputeAheadScanner::Scan(const Item& item) {
	// Every submission is followed for its registers; Behind() keeps from building for one the GPU
	// thread is already past. The one it is in is still ahead of it: a frame's draws are one
	// submission, and the GPU thread stands inside it while it compiles their pipelines.
	m_item = &item;
	PipelineCache::AheadSubmission(item.seq);
	auto& state = State(item.queue);
	struct Cursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset = 0;
	};
	std::vector<Cursor> stack {{item.commands}};
	while (!stack.empty()) {
		auto& cursor = stack.back();
		if (cursor.offset >= cursor.commands.size()) {
			stack.pop_back();
			continue;
		}
		const auto* const packet    = cursor.commands.data() + cursor.offset;
		const auto        total     = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining = total - cursor.offset;
		const auto        header    = packet[0];
		if (header == 0x80000000u) {
			cursor.offset++;
			continue;
		}
		if (remaining < 2) return;
		if ((header >> 30u) == 0u) {
			const auto length = ((header >> 16u) & 0x3fffu) + 2u;
			if (length > remaining) return;
			cursor.offset += length;
			continue;
		}
		const auto opcode = (header >> 8u) & 0xffu;
		const auto length = KYTY_PM4_LEN(header);
		if (length > remaining) return;
		const auto command = header & ~1u;
		switch (opcode) {
			case Pm4::IT_DISPATCH_DIRECT:
			case Pm4::IT_DISPATCH_INDIRECT: ScanDispatch(state, packet, opcode); break;
			case Pm4::IT_DRAW_INDIRECT:
			case Pm4::IT_DRAW_INDEX_INDIRECT:
			case Pm4::IT_DRAW_INDIRECT_MULTI:
			case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
			case Pm4::IT_DRAW_INDEX_2:
			case Pm4::IT_DRAW_INDEX_OFFSET_2:
			case Pm4::IT_DRAW_INDEX_AUTO: ScanDraw(state, packet, opcode); break;
			case Pm4::IT_SET_BASE:
			case Pm4::IT_SET_CONTEXT_REG:
			case Pm4::IT_SET_SH_REG:
			case Pm4::IT_SET_UCONFIG_REG:
			case Pm4::IT_SET_UCONFIG_REG_INDEX:
				cursor.offset += g_cp_op_func[opcode](*state.cp, command, packet + 1, remaining, total) + 1u;
				continue;
			case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
			case Pm4::IT_SET_SH_REG_INDIRECT:
			case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
				// As the GPU thread's handler: the register list is read where the game left it.
				if (length != 5u) {
					state.tainted = true;
					break;
				}
				if (const uint64_t list = (packet[4] & 0x3fffu) * 8ull;
				    list != 0 && !Followable(state, (packet[1] & 0xfffffffcu) | (static_cast<uint64_t>(packet[2]) << 32u), list)) {
					break;
				}
				(void)g_cp_op_func[opcode](*state.cp, command, packet + 1, remaining, total);
				break;
			case Pm4::IT_CLEAR_STATE: state.cp->GetCtx().Reset(); break;
			case Pm4::IT_INDIRECT_BUFFER: {
				if (length == 14u) {
					// A branch on a value in guest memory (CpOpBranch), mostly one the GPU writes: both
					// sides are followed, "then" first; a side not taken only costs workers' guesses.
					const auto* then_buffer = reinterpret_cast<const uint32_t*>((packet[8] & 0xfffffffcu) |
					                                                            (static_cast<uint64_t>(packet[9]) << 32u));
					const auto* else_buffer = reinterpret_cast<const uint32_t*>((packet[11] & 0xfffffffcu) |
					                                                            (static_cast<uint64_t>(packet[12]) << 32u));
					const uint32_t then_count = packet[10] & 0xfffffu, else_count = packet[13] & 0xfffffu;
					cursor.offset += length;
					if (else_buffer != nullptr && else_count != 0 &&
					    Followable(state, reinterpret_cast<uint64_t>(else_buffer), else_count * 4ull)) {
						stack.push_back({std::span(else_buffer, else_count)});
					}
					if (then_buffer != nullptr && then_count != 0 &&
					    Followable(state, reinterpret_cast<uint64_t>(then_buffer), then_count * 4ull)) {
						stack.push_back({std::span(then_buffer, then_count)});
					}
					continue;
				}
				if (length != 4u) return;
				const auto* sub = reinterpret_cast<const uint32_t*>(packet[1] | (static_cast<uint64_t>(packet[2]) << 32u));
				const uint32_t control = packet[3];
				const uint32_t count   = control & 0xfffffu;
				cursor.offset += length;
				if (count != 0 && sub != nullptr && Followable(state, reinterpret_cast<uint64_t>(sub), count * 4ull)) {
					if ((control & (1u << 20u)) != 0) {
						cursor = {std::span(sub, count)};
					} else {
						stack.push_back({std::span(sub, count)});
					}
				}
				continue;
			}
			case Pm4::IT_NOP: {
				const auto r = KYTY_PM4_R(header);
				if (r == Pm4::R_ZERO) {
					if ((packet[1] & 0xffff0000u) == 0x68750000u) {
						const auto id = packet[1] & 0xfffu;
						if (id == 0x4u || id == 0xdu) (void)g_cp_op_func[Pm4::IT_NOP](*state.cp, command, packet + 1, remaining, total);
					}
				} else if (r == Pm4::R_CONTEXT_STATE) {
					ApplyContextState(state, packet[1]);
				} else if (r == Pm4::R_DISPATCH_RESET) {
					state.cp->Reset();
					state.pushed = false;
				}
				break;
			}
			default: break;
		}
		cursor.offset += length;
	}
}

} // namespace Libs::Graphics
