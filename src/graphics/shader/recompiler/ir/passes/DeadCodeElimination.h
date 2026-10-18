#pragma once

#include "graphics/shader/recompiler/ir/Block.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

struct Program;

void RemoveIdentities(Program& program);
// Replaces every web of phis that only ever carries one value with that value (the web becomes
// identities; run RemoveIdentities and EliminateDeadCode afterwards). Returns whether any did.
bool RemoveRedundantPhiWebs(const BlockList& blocks);
void EliminateDeadCode(const BlockList& blocks);

} // namespace Libs::Graphics::ShaderRecompiler::IR
