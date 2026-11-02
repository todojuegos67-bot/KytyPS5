#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct FunctionLdsLayout {
	std::unordered_map<const Inst*, uint32_t> slots;
	uint32_t                                  dwords = 0;
};

// Graphics LDS currently uses invocation-private Function storage. If every access
// is a scalar dword at (LaneId << 2) + a constant offset, the lane term is identical
// for every access by that invocation. Distinct aligned offsets therefore identify
// distinct slots. Keep the original address/bounds checks in the emitter; only the
// storage indices change. Reject the entire layout if any LDS access differs.
inline FunctionLdsLayout PlanFunctionLdsLayout(const Program& program) {
	FunctionLdsLayout result;
	if (program.stage != ShaderType::Pixel) return result;
	std::vector<uint32_t>                     offsets;
	std::unordered_map<const Inst*, uint32_t> accesses;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (SharedAccessOf(inst.GetOpcode()) == SharedAccess::None) continue;
			const auto index = inst.Flags<MemoryFlags>().index;
			if (index >= program.memory_info.size()) return {};
			const auto& memory = program.memory_info[index];
			if (memory.kind != ResourceKind::Lds) continue;
			if ((inst.GetOpcode() != ValueOpcode::LoadSharedU32 &&
			     inst.GetOpcode() != ValueOpcode::WriteSharedU32) ||
			    memory.offset % 4u != 0u) {
				return {};
			}
			const auto* address = inst.Arg(0).Resolve().TryInstruction();
			if (!address || address->GetOpcode() != ValueOpcode::ShiftLeftLogical32) return {};
			const auto  shift = address->Arg(1).Resolve();
			const auto* lane  = address->Arg(0).Resolve().TryInstruction();
			if (!shift.IsImmediate() || shift.GetType() != Type::U32 || shift.U32() != 2u ||
			    !lane || (lane->GetOpcode() != ValueOpcode::LaneId && lane->GetOpcode() != ValueOpcode::WaveLaneId))
				return {};
			accesses.emplace(&inst, memory.offset);
			offsets.push_back(memory.offset);
		}
	}
	std::sort(offsets.begin(), offsets.end());
	offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
	for (const auto& [inst, offset]: accesses) {
		result.slots.emplace(
		    inst, static_cast<uint32_t>(std::lower_bound(offsets.begin(), offsets.end(), offset) -
		                                offsets.begin()));
	}
	result.dwords = static_cast<uint32_t>(offsets.size());
	return result;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
