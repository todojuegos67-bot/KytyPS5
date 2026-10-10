#ifndef EMULATOR_SRC_LOCAL_DRAW_STATE_OBSERVER_H_
#define EMULATOR_SRC_LOCAL_DRAW_STATE_OBSERVER_H_
// Tracks whether the graphics state changed between consecutive indexed draws
// (indirect or direct), for the native XPR draw path: a draw is "clean" when nothing
// but the per-draw user-data SGPRs (the XPR geometry root), the index state and the
// indirect offset changed since the previous one, so the previous draw-state key
// still applies.
//
// A register write only dirties the gap when it changes a value (shadow copies
// of the CX/SH/UC spaces). Graphics user-data SGPRs (PS 0x0c-0x2b, GS 0x8c-0xab,
// HS 0x10c-0x12b) and the compute space (0x200+) never dirty it; dispatches,
// synchronisation, index state and indirect bases never do either.
#include "graphics/guest_gpu/pm4.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics::DrawStateObserver {

struct Shadow {
	// Each register's last value (low 32 bits) and the generation it was written in (high 32 bits): known while that is
	// `generation` (Invalidate starts another), so a write is one compare and a reset is one increment (the command
	// processor's Reset invalidates at every submission; the UC space alone has 16384 registers).
	std::array<uint64_t, Pm4::CX_NUM> cx {};
	std::array<uint64_t, Pm4::SH_NUM> sh {};
	std::array<uint64_t, Pm4::UC_NUM> uc {};
	uint64_t generation = uint64_t {1} << 32u;
	bool chain      = false; // the previous graphics draw was an indexed indirect draw
	bool dirty      = true;  // graphics state changed since that draw
	bool last_clean = false; // the latest DRAW_INDEX_INDIRECT was clean
};
inline thread_local Shadow g_shadow;

[[nodiscard]] inline bool ShFree(uint32_t offset) {
	return (offset >= 0x0cu && offset <= 0x2bu) || (offset >= 0x8cu && offset <= 0xabu) ||
	       (offset >= 0x10cu && offset <= 0x12bu) || offset >= 0x200u;
}

// space: 0 CX, 1 SH, 2 UC.
inline void WriteRegister(uint32_t space, uint32_t raw_offset, uint32_t value) {
	auto&      s      = g_shadow;
	const auto offset = raw_offset & 0xffffu;
	const auto update = [&](auto& values) {
		if (offset >= values.size()) {
			s.dirty = true;
			return;
		}
		const uint64_t entry = s.generation | value;
		if (values[offset] != entry) {
			values[offset] = entry;
			s.dirty        = true;
		}
	};
	if (space == 1) {
		if (offset < Pm4::SH_NUM && ShFree(offset)) return;
		update(s.sh);
	} else if (space == 0) {
		update(s.cx);
	} else {
		update(s.uc);
	}
}

inline void WriteIndirectBlock(uint32_t space, const uint32_t* packet) {
	const auto* block = reinterpret_cast<const uint32_t*>((uint64_t(packet[1]) & 0xfffffffcu) |
	                                                      (uint64_t(packet[2]) << 32u));
	const uint32_t pairs = packet[4] & 0x3fffu;
	if (block == nullptr) return;
	for (uint32_t i = 0; i < pairs; ++i) {
		if (block[2 * i] == 0xffffffffu) continue;
		WriteRegister(space, block[2 * i], block[2 * i + 1]);
	}
}

// The registers changed in bulk (context clear/push/pop, a register reset): nothing the
// shadow holds is known any more.
inline void Invalidate() {
	auto& s = g_shadow;
	s.generation += uint64_t {1} << 32u;
	if (s.generation == 0) { // (wrapped: an entry of the first generation would be taken as known again)
		s.cx.fill(0);
		s.sh.fill(0);
		s.uc.fill(0);
		s.generation = uint64_t {1} << 32u;
	}
	s.chain = false;
	s.dirty = true;
}

// Every packet the command processor is about to execute.
inline void ObservePacket(uint32_t opcode, const uint32_t* packet, uint32_t packet_dw, bool predicated_skip) {
	auto& s = g_shadow;
	if (predicated_skip) {
		s.dirty = true;
		return;
	}
	switch (opcode) {
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
			s.last_clean = s.chain && !s.dirty;
			s.chain      = true;
			s.dirty      = false;
			return;
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX: {
			const uint32_t space = opcode == Pm4::IT_SET_CONTEXT_REG ? 0u
			                       : opcode == Pm4::IT_SET_SH_REG    ? 1u
			                                                         : 2u;
			for (uint32_t i = 2; i < packet_dw; ++i) WriteRegister(space, packet[1] + (i - 2), packet[i]);
			return;
		}
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
		case Pm4::IT_SET_SH_REG_INDIRECT:
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
			if (packet_dw < 5) {
				s.dirty = true;
				return;
			}
			WriteIndirectBlock(opcode == Pm4::IT_SET_CONTEXT_REG_INDIRECT ? 0u
			                   : opcode == Pm4::IT_SET_SH_REG_INDIRECT    ? 1u
			                                                              : 2u,
			                   packet);
			return;
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_INDEX_BUFFER_SIZE:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT:
		case Pm4::IT_SET_BASE:
		case Pm4::IT_NOP:
			// NOP-encoded commands: R_CONTEXT_STATE clears, pushes or pops the graphics
			// context and R_DISPATCH_RESET resets every register (CommandProcessor::Reset).
			if (const auto r = KYTY_PM4_R(packet[0]); r == Pm4::R_CONTEXT_STATE || r == Pm4::R_DISPATCH_RESET)
				Invalidate();
			return;
		case Pm4::IT_NUM_INSTANCES:
		case Pm4::IT_ACQUIRE_MEM:
		case Pm4::IT_RELEASE_MEM:
		case Pm4::IT_WAIT_REG_MEM:
		case Pm4::IT_WAIT_REG_MEM_64:
		case Pm4::IT_EVENT_WRITE:
		case Pm4::IT_EVENT_WRITE_EOP:
		case Pm4::IT_EVENT_WRITE_EOS:
		case Pm4::IT_WRITE_DATA:
		case Pm4::IT_PFP_SYNC_ME:
		case Pm4::IT_INDIRECT_BUFFER:
		case Pm4::IT_COPY_DATA:
		case Pm4::IT_DMA_DATA: return;
		default:
			// Any other draw ends the chain; anything unknown may change state.
			s.chain = false;
			s.dirty = true;
			return;
	}
}

} // namespace Libs::Graphics::DrawStateObserver

#endif // EMULATOR_SRC_LOCAL_DRAW_STATE_OBSERVER_H_
