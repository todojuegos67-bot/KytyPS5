#pragma once

#include "graphics/shader/recompiler/ir/Block.h"

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct ExecSelectStats {
	uint32_t bypassed = 0; // selects whose old operand skipped a select on their predicate
	uint32_t dropped  = 0; // selects replaced by their new operand
};

// A vector register write under EXEC is Select(exec, new, old) in every invocation: the lanes outside EXEC keep the
// old value. Where no use observes those lanes (each use reaches, through lane-wise operations alone, the new operand
// of a select on the same predicate), the select is its new operand. Select(p, b, Select(p, a, o)) is Select(p, b, o).
// Leaves the dropped selects as Void instructions and what fed only them dead (EliminateDeadCode).
[[nodiscard]] ExecSelectStats EliminateExecSelects(const BlockList& blocks);

} // namespace Libs::Graphics::ShaderRecompiler::IR
