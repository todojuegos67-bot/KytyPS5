#pragma once

#include "graphics/shader/recompiler/ir/Block.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct Program;

// The conditions every invocation running a block satisfies, sorted.
using BlockConditions = std::unordered_map<const Block*, std::vector<const Inst*>>;

struct BranchResult {
	uint32_t                        branches = 0;
	std::map<std::string, uint32_t> problems; // why the other EXEC branches stay votes
	BlockConditions                 block_conditions;
};

void RemoveIdentities(Program& program);
void EliminateDeadCode(const BlockList& blocks);
// Makes EXEC branches test each invocation's own condition where that is exact. Only for programs emitted one guest lane per invocation.
BranchResult BranchPerInvocation(Program& program);
// Replaces a use of a select of a lane's new or old value by the new value where that use needs
// it only in lanes where the select's condition holds, and by the old value where it needs it
// only in lanes where the condition fails: the consumer is masked by the same condition again,
// or runs only in invocations that satisfy it (block_conditions). Returns how many selects lost
// a use.
uint32_t EliminateMaskedSelects(const BlockList& blocks, const BlockConditions& block_conditions = {});

} // namespace Libs::Graphics::ShaderRecompiler::IR
