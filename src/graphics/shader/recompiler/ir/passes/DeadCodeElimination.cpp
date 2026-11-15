#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

void RemoveIdentities(Program& program) {
	// Resolve metadata before deleting its identity nodes; retain only folded native constants.
	for (auto& write: program.scalar_writes) {
		const auto value = write.value.Resolve();
		write.value = value.IsImmediate() && value.GetType() == Type::U32 ? value : Value {};
	}
	for (auto* block: program.blocks) {
		block->condition       = block->condition.Resolve();
		block->indirect_target = block->indirect_target.Resolve();
	}
	for (auto* block: program.blocks) {
		auto& instructions = block->Instructions();
		for (auto inst = instructions.begin(); inst != instructions.end();) {
			if (inst->GetOpcode() != ValueOpcode::Identity) {
				inst++;
				continue;
			}
			const auto replacement = inst->Arg(0);
			inst->ReplaceUsesWith(replacement, false);
			inst = instructions.erase(inst);
		}
	}
}

void EliminateDeadCode(const BlockList& blocks) {
	bool changed;
	do {
		std::vector<Inst*> pending;
		for (auto* block: blocks) {
			for (auto& inst: *block) {
				// Retained resource-planning instructions live outside the block list.
				inst.live = inst.MayHaveSideEffects() || std::ranges::any_of(inst.Uses(),
				    [](const Use& use) { return use.user->Parent() == nullptr; });
				if (inst.live) pending.push_back(&inst);
			}
		}
		while (!pending.empty()) {
			const auto* inst = pending.back();
			pending.pop_back();
			for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
				auto* input = inst->Arg(arg).TryInstruction();
				if (input != nullptr && input->Parent() != nullptr && !input->live) {
					input->live = true;
					pending.push_back(input);
				}
			}
		}
		// Unlink every dead operand before freeing nodes in a dead Phi cycle.
		for (auto* block: blocks)
			for (auto& inst: *block)
				if (!inst.live) inst.Invalidate();
		for (auto* block: blocks)
			std::erase_if(block->Instructions(), [](const Inst& inst) { return !inst.live; });

		changed = false;
		for (auto* block: blocks) {
			const auto& term = block->terminator;
			if (term.kind != CFG::TerminatorKind::ConditionalBranch || term.loop_header ||
			    block->condition.IsImmediate()) continue;
			for (const bool take_true: {false, true}) {
				const auto* empty = take_true ? term.false_block : term.true_block;
				const auto* merge = take_true ? term.true_block : term.false_block;
				if (empty == nullptr || merge == nullptr || !empty->empty() ||
				    empty->terminator.loop_header || merge->terminator.loop_header ||
				    empty->terminator.kind != CFG::TerminatorKind::Branch ||
				    empty->terminator.true_block != merge ||
				    std::ranges::any_of(*merge, [](const Inst& inst) {
					    return inst.GetOpcode() == ValueOpcode::Phi;
				    })) continue;
				// Keep the CFG and SSA edges intact; only the empty detour's predicate is dead.
				for (auto& inst: *block) {
					if (inst.GetOpcode() == ValueOpcode::Reference &&
					    inst.Arg(0).Resolve() == block->condition.Resolve())
						inst.SetArg(0, Value(take_true));
				}
				block->condition = Value(take_true);
				changed = true;
				break;
			}
		}
	} while (changed);
}

namespace {

// Whether an instruction computes each lane's result from that lane's operands alone, with no
// side effects, so its operands matter only in the lanes its result does.
bool IsLaneLocal(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::Identity:
		case ValueOpcode::BitCastU16F16:
		case ValueOpcode::BitCastF16U16:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::BitCastU64F64:
		case ValueOpcode::BitCastF64U64:
		case ValueOpcode::ConvertU16U32:
		case ValueOpcode::ConvertU32U16:
		case ValueOpcode::ConvertU8U32:
		case ValueOpcode::ConvertU32U8:
		case ValueOpcode::ConvertF32F16:
		case ValueOpcode::ConvertF16F32:
		case ValueOpcode::ConvertS32F32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32S32:
		case ValueOpcode::ConvertF64S32:
		case ValueOpcode::ConvertF32F64:
		case ValueOpcode::ConvertF64F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::ConvertF64U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeConstructU32x3:
		case ValueOpcode::CompositeConstructF32x2:
		case ValueOpcode::CompositeConstructU32x4:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::CompositeExtractU32x3:
		case ValueOpcode::CompositeExtractU32x4:
		case ValueOpcode::PackHalf2x16:
		case ValueOpcode::PackSnorm2x16:
		case ValueOpcode::PackUnorm2x16:
		case ValueOpcode::PackFloat2x16Rtz:
		case ValueOpcode::FPAbs32:
		case ValueOpcode::FPNeg32:
		case ValueOpcode::FPSaturate32:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::UDiv32:
		case ValueOpcode::SMulHi:
		case ValueOpcode::UMulHi:
		case ValueOpcode::IAbs32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::BitReverse32:
		case ValueOpcode::BitCount32:
		case ValueOpcode::FindUMsb32:
		case ValueOpcode::FindILsb32:
		case ValueOpcode::SMin32:
		case ValueOpcode::UMin32:
		case ValueOpcode::SMax32:
		case ValueOpcode::UMax32:
		case ValueOpcode::SMinTri32:
		case ValueOpcode::UMinTri32:
		case ValueOpcode::SMaxTri32:
		case ValueOpcode::UMaxTri32:
		case ValueOpcode::SMedTri32:
		case ValueOpcode::UMedTri32:
		case ValueOpcode::SLessThan32:
		case ValueOpcode::SLessThan64:
		case ValueOpcode::SLessThanEqual64:
		case ValueOpcode::ULessThanEqual64:
		case ValueOpcode::UGreaterThanEqual64:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThan64:
		case ValueOpcode::IEqual32:
		case ValueOpcode::IEqual64:
		case ValueOpcode::SLessThanEqual32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::SGreaterThan32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::UGreaterThan64:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::INotEqual64:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::UGreaterThanEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdEqual32:
		case ValueOpcode::FPOrdEqual64:
		case ValueOpcode::FPOrdLessThanEqual64:
		case ValueOpcode::FPOrdGreaterThanEqual64:
		case ValueOpcode::FPOrdNotEqual64:
		case ValueOpcode::FPOrdLessThan64:
		case ValueOpcode::FPOrdGreaterThan64:
		case ValueOpcode::FPUnordEqual64:
		case ValueOpcode::FPUnordNotEqual64:
		case ValueOpcode::FPUnordLessThan64:
		case ValueOpcode::FPUnordGreaterThan64:
		case ValueOpcode::FPUnordLessThanEqual64:
		case ValueOpcode::FPUnordGreaterThanEqual64:
		case ValueOpcode::FPUnordEqual32:
		case ValueOpcode::FPOrdNotEqual32:
		case ValueOpcode::FPUnordNotEqual32:
		case ValueOpcode::FPOrdLessThan32:
		case ValueOpcode::FPUnordLessThan32:
		case ValueOpcode::FPOrdGreaterThan32:
		case ValueOpcode::FPUnordGreaterThan32:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPUnordLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPUnordGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPIsNan64:
		case ValueOpcode::FPCmpClass32:
		case ValueOpcode::FPCmpClass16:
		case ValueOpcode::FPAdd32:
		case ValueOpcode::FPSub32:
		case ValueOpcode::FPFma32:
		case ValueOpcode::FPMad32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPAdd64:
		case ValueOpcode::FPMul64:
		case ValueOpcode::FPFma64:
		case ValueOpcode::FPRecip64:
		case ValueOpcode::FPMin32:
		case ValueOpcode::FPMax32:
		case ValueOpcode::FPMin64:
		case ValueOpcode::FPMax64:
		case ValueOpcode::FPMinTri32:
		case ValueOpcode::FPMaxTri32:
		case ValueOpcode::FPMedTri32:
		case ValueOpcode::FPRecip32:
		case ValueOpcode::FPRecipSqrt32:
		case ValueOpcode::FPSqrt:
		case ValueOpcode::FPSin:
		case ValueOpcode::FPCos:
		case ValueOpcode::FPExp2:
		case ValueOpcode::FPLog2:
		case ValueOpcode::FPLdexp:
		case ValueOpcode::FPRoundEven32:
		case ValueOpcode::FPFloor32:
		case ValueOpcode::FPCeil32:
		case ValueOpcode::FPTrunc32:
		case ValueOpcode::FPFloor64:
		case ValueOpcode::FPCeil64:
		case ValueOpcode::FPTrunc64:
		case ValueOpcode::FPFract32:
		case ValueOpcode::FPFract64: return true;
		default: return false;
	}
}

// Whether a memory operation reads or writes for a lane only when its trailing predicate is
// set, so its other operands matter only in those lanes. Image sampling is left out: its
// coordinates also feed the derivatives of neighboring lanes.
bool IsPredicatedPerLane(const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (inst.NumArgs() == 0 || inst.Arg(inst.NumArgs() - 1).GetType() != Type::U1) {
		return false;
	}
	switch (SharedAccessOf(op)) {
		case SharedAccess::Append:
		case SharedAccess::Consume: return false;
		case SharedAccess::None: break;
		default: return true;
	}
	switch (op) {
		case ValueOpcode::ImageRead:
		case ValueOpcode::ImageWrite: return true;
		default: break;
	}
	return BufferAccessOf(op) != BufferAccess::None ||
	       AddressOpcodeInfoOf(op).access != AddressAccess::None;
}

// A conjunction of U1 values, each maybe negated, sorted. A null value stands for false.
using Atoms = std::vector<std::pair<const Inst*, bool>>;

// The lanes a value is needed in, as a conjunction of conditions, each maybe negated: the lanes
// where all of them hold. No conditions means every lane; an undemanded value has not been
// reached yet.
struct Demand {
	bool  reached = false;
	Atoms conditions;
};

// Adds a condition, negated if asked, as a conjunction of instructions that hold or fail. False
// where a part of it is no instruction (an immediate false, or a register): leaving that part out
// only widens the lanes, which a demand may do but a select's test may not.
bool AddSignedAtoms(Value value, bool negate, Atoms& atoms) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		return value.U1() != negate;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	const auto op = inst->GetOpcode();
	if (op == ValueOpcode::LogicalNot) {
		return AddSignedAtoms(inst->Arg(0), !negate, atoms);
	}
	if ((op == ValueOpcode::LogicalAnd && !negate) || (op == ValueOpcode::LogicalOr && negate)) {
		const bool first = AddSignedAtoms(inst->Arg(0), negate, atoms);
		return AddSignedAtoms(inst->Arg(1), negate, atoms) && first;
	}
	const std::pair<const Inst*, bool> atom {inst, negate};
	if (const auto found = std::ranges::lower_bound(atoms, atom);
	    found == atoms.end() || *found != atom) {
		atoms.insert(found, atom);
	}
	return true;
}

bool IsSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 || op == ValueOpcode::SelectF32;
}

// The operand a select gives in every lane where the demand holds: 1 where its condition is set
// there, 2 where it is clear there, 0 where that is not known.
size_t ChosenOperand(const Inst& select, const Atoms& demand) {
	Atoms set;
	if (AddSignedAtoms(select.Arg(0), false, set) && !set.empty() && std::ranges::includes(demand, set)) {
		return 1;
	}
	Atoms clear;
	if (AddSignedAtoms(select.Arg(0), true, clear) && !clear.empty() && std::ranges::includes(demand, clear)) {
		return 2;
	}
	// A conjunction fails wherever one of its parts does.
	for (const auto& [inst, negated]: set) {
		if (std::ranges::binary_search(demand, std::pair<const Inst*, bool> {inst, !negated})) {
			return 2;
		}
	}
	return 0;
}

// With holding, also every value on the way that is true where the whole is: a node itself, or
// for a negated node, each LogicalNot of it (selects test their own copies).
void AddAtoms(Value value, bool negate, Atoms& atoms, std::vector<const Inst*>* holding = nullptr) {
	value = value.Resolve();
	const auto* inst = value.IsImmediate() ? nullptr : value.TryInstruction();
	if (inst == nullptr && value.IsImmediate() && value.U1() != negate) {
		return;
	}
	if (inst != nullptr && holding != nullptr) {
		if (!negate) {
			holding->push_back(inst);
		} else {
			for (const auto& use: inst->Uses()) {
				if (use.user->GetOpcode() == ValueOpcode::LogicalNot && use.user->Parent() != nullptr) {
					holding->push_back(use.user);
				}
			}
		}
	}
	if (inst != nullptr) {
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::LogicalNot) {
			AddAtoms(inst->Arg(0), !negate, atoms, holding);
			return;
		}
		if ((op == ValueOpcode::LogicalAnd && !negate) || (op == ValueOpcode::LogicalOr && negate)) {
			AddAtoms(inst->Arg(0), negate, atoms, holding);
			AddAtoms(inst->Arg(1), negate, atoms, holding);
			return;
		}
	}
	const std::pair<const Inst*, bool> atom {inst, negate};
	if (const auto found = std::ranges::lower_bound(atoms, atom);
	    found == atoms.end() || *found != atom) {
		atoms.insert(found, atom);
	}
}

bool IsZeroVote(CFG::BranchCondition kind) {
	return kind == CFG::BranchCondition::ExecZero || kind == CFG::BranchCondition::VccZero ||
	       kind == CFG::BranchCondition::SccZero;
}

// U1 values every invocation of a wave holds alike: the guest's branch conditions (votes or
// scalar tests) and logic over them. Phis of them are too, since every guest branch is
// wave-uniform before BranchPerInvocation.
std::unordered_set<const Inst*> UniformConditions(const BlockList& blocks) {
	std::unordered_set<const Inst*> uniform;
	for (const auto* block: blocks) {
		for (const auto& inst: *block) {
			switch (inst.GetOpcode()) {
				case ValueOpcode::ConditionRef:
				case ValueOpcode::LogicalAnd:
				case ValueOpcode::LogicalOr:
				case ValueOpcode::LogicalXor:
				case ValueOpcode::LogicalNot:
				case ValueOpcode::SelectU1: uniform.insert(&inst); break;
				case ValueOpcode::Phi:
					if (inst.GetType() == Type::U1) uniform.insert(&inst);
					break;
				default: break;
			}
		}
	}
	for (bool changed = true; changed;) {
		changed = false;
		std::erase_if(uniform, [&](const Inst* inst) {
			if (inst->GetOpcode() == ValueOpcode::ConditionRef) {
				return false;
			}
			for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
				const auto value = inst->Arg(arg).Resolve();
				if (!value.IsImmediate() && !uniform.contains(value.TryInstruction())) {
					changed = true;
					return true;
				}
			}
			return false;
		});
	}
	return uniform;
}

// Operations whose result in one invocation depends on others.
bool IsCrossInvocation(const Program& program, const Inst& inst) {
	switch (inst.GetOpcode()) {
		case ValueOpcode::Ballot:
		case ValueOpcode::ReadFirstLane:
		case ValueOpcode::ReadLane:
		case ValueOpcode::SubgroupReduceU32:
		case ValueOpcode::LaneReduceU32:
		case ValueOpcode::WriteLane:
		case ValueOpcode::Permlane16U32:
		case ValueOpcode::PermuteU32:
		case ValueOpcode::BpermuteU32:
		case ValueOpcode::SwizzleU32:
		case ValueOpcode::DppMoveU32:
		case ValueOpcode::DppUpdateU32:
		case ValueOpcode::WqmU64:
		case ValueOpcode::DataAppend:
		case ValueOpcode::DataConsume:
		case ValueOpcode::ImageQueryLod: return true;
		// Pixel shaders sample with implicit derivatives, which read the quad's other invocations.
		case ValueOpcode::ImageSampleRaw:
		case ValueOpcode::ImageGatherRaw: return program.stage == ShaderType::Pixel;
		default: return false;
	}
}

struct ExecRegion {
	Block*                          branch = nullptr;
	Inst*                           vote   = nullptr;
	Block*                          merge  = nullptr;
	bool                            negated_vote = false; // the body runs where Not(predicate)
	Atoms                           condition; // what the body's invocations satisfy
	Atoms                           varying;   // its atoms that can differ between invocations
	std::vector<Block*>             blocks;    // the body
	std::unordered_set<const Block*> body;
};

// The value an invocation that the vote alone sent into the region hands to the merge, as far as
// it follows from the region's selects: one whose condition fails for it keeps the old value.
struct Inactive {
	enum class Kind { Unknown, Open, Known } kind = Kind::Unknown;
	Value value;
};

Inactive InactiveValue(const ExecRegion& region, Value value,
                       std::unordered_map<const Inst*, Inactive>& memo) {
	value = value.Resolve();
	auto* inst = value.IsImmediate() ? nullptr : value.TryInstruction();
	if (inst == nullptr || !region.body.contains(inst->Parent())) {
		return {Inactive::Kind::Known, value};
	}
	if (const auto found = memo.find(inst); found != memo.end()) {
		return found->second; // Open: a loop-carried phi, which adds nothing of its own
	}
	memo[inst] = {Inactive::Kind::Open, {}};
	Inactive result;
	switch (inst->GetOpcode()) {
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32: {
			Atoms chosen;
			AddAtoms(inst->Arg(0), false, chosen);
			if (std::ranges::includes(chosen, region.varying)) {
				result = InactiveValue(region, inst->Arg(2), memo);
			}
			break;
		}
		case ValueOpcode::Phi: {
			result.kind = Inactive::Kind::Open;
			for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
				const auto incoming = InactiveValue(region, inst->Arg(arg), memo);
				if (incoming.kind == Inactive::Kind::Open) {
					continue;
				}
				if (incoming.kind == Inactive::Kind::Unknown ||
				    (result.kind == Inactive::Kind::Known && !(incoming.value == result.value))) {
					result = {};
					break;
				}
				result = incoming;
			}
			break;
		}
		default: break;
	}
	memo[inst] = result;
	return result;
}

const char* RegionProblem(const Program& program, const ExecRegion& region,
                          const std::unordered_set<const Inst*>& converted) {
	for (const auto* block: region.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (op == ValueOpcode::Barrier) {
				return "barrier"; // every invocation of the workgroup must reach it
			}
			if (IsCrossInvocation(program, inst)) {
				return "cross-lane";
			}
			if (op == ValueOpcode::ConditionRef &&
			    inst.Flags<CFG::BranchCondition>() != CFG::BranchCondition::ScalarInstruction &&
			    !converted.contains(&inst)) {
				return "vote";
			}
			if (inst.MayHaveSideEffects() && op != ValueOpcode::Reference &&
			    op != ValueOpcode::ReferenceU32) {
				// Only an invocation whose own condition holds may write.
				if (!IsPredicatedPerLane(inst)) {
					return "side effect";
				}
				Atoms active;
				AddAtoms(inst.Arg(inst.NumArgs() - 1), false, active);
				if (!std::ranges::includes(active, region.varying)) {
					return "side effect";
				}
			}
			for (const auto& use: inst.Uses()) {
				const auto* parent = use.user->Parent();
				if (parent == nullptr) {
					return "external";
				}
				if (!region.body.contains(parent) &&
				    (parent != region.merge || use.user->GetOpcode() != ValueOpcode::Phi)) {
					return "live-out";
				}
			}
		}
	}
	if (region.varying.empty()) {
		return nullptr;
	}
	std::unordered_map<const Inst*, Inactive> memo;
	for (const auto& phi: *region.merge) {
		if (phi.GetOpcode() != ValueOpcode::Phi) {
			break;
		}
		Value bypass;
		for (size_t arg = 0; arg < phi.NumArgs(); arg++) {
			if (phi.PhiBlock(arg) == region.branch) {
				bypass = phi.Arg(arg).Resolve();
			}
		}
		for (size_t arg = 0; arg < phi.NumArgs(); arg++) {
			if (!region.body.contains(phi.PhiBlock(arg))) {
				continue;
			}
			const auto inactive = InactiveValue(region, phi.Arg(arg), memo);
			if (inactive.kind != Inactive::Kind::Known || bypass.IsEmpty() ||
			    !(inactive.value == bypass)) {
				return "live-out";
			}
		}
	}
	return nullptr;
}

} // namespace

BranchResult BranchPerInvocation(Program& program) {
	// A guest branch on EXEC (s_cbranch_execz) is a subgroup vote: once one invocation wants the
	// block, all run it, and selects keep the old values of those whose EXEC is clear. The branch
	// can test each invocation's own condition instead where the invocations the vote alone sent
	// in gain nothing there: no cross-invocation operation, no write they could make, no exit, and
	// every value they hand to the merge is the one the skip edge carries. Inside, every select
	// on a condition the branch now guarantees is its new value (EliminateMaskedSelects).
	BranchResult result;
	if (program.dispatcher_fallback) {
		return result;
	}
	const auto uniform = UniformConditions(program.blocks);
	std::vector<ExecRegion> regions;
	for (auto* block: program.blocks) {
		const auto& term = block->terminator;
		if (term.kind != CFG::TerminatorKind::ConditionalBranch || term.loop_header ||
		    term.merge_block == nullptr || term.true_block == nullptr ||
		    term.false_block == nullptr || block->condition.IsEmpty()) {
			continue;
		}
		auto* condition = block->condition.Resolve().TryInstruction();
		auto* vote      = condition;
		const bool negated = condition != nullptr && condition->GetOpcode() == ValueOpcode::LogicalNot;
		if (negated) {
			vote = condition->Arg(0).Resolve().TryInstruction();
		}
		if (vote == nullptr || vote->GetOpcode() != ValueOpcode::ConditionRef ||
		    vote->Flags<CFG::BranchCondition>() == CFG::BranchCondition::ScalarInstruction) {
			continue;
		}
		const auto only_referenced = [&](const Inst* inst, const Inst* allowed) {
			return std::ranges::all_of(inst->Uses(), [&](const Use& use) {
				return use.user == allowed || use.user->GetOpcode() == ValueOpcode::Reference;
			});
		};
		if (!only_referenced(vote, negated ? condition : nullptr) ||
		    (negated && !only_referenced(condition, nullptr))) {
			continue;
		}
		// Any(P) and Not(All(P)) enter their true successor; per invocation the test becomes P.
		const bool any       = !IsZeroVote(vote->Flags<CFG::BranchCondition>());
		const bool body_true = any != negated;
		ExecRegion region {.branch = block, .vote = vote, .merge = term.merge_block};
		auto* entry = body_true ? term.true_block : term.false_block;
		if ((body_true ? term.false_block : term.true_block) != region.merge || entry == region.merge) {
			continue;
		}
		region.negated_vote = !any;
		AddAtoms(vote->Arg(0), !any, region.condition);
		for (const auto& atom: region.condition) {
			if (atom.first != nullptr && !uniform.contains(atom.first)) {
				region.varying.push_back(atom);
			}
		}
		region.blocks.push_back(entry);
		region.body.insert(entry);
		bool shaped = true;
		for (size_t next = 0; next < region.blocks.size() && shaped; next++) {
			const auto* current = region.blocks[next];
			const auto& kind    = current->terminator;
			if ((kind.kind != CFG::TerminatorKind::Branch &&
			     kind.kind != CFG::TerminatorKind::ConditionalBranch) ||
			    kind.true_block == nullptr ||
			    (kind.kind == CFG::TerminatorKind::ConditionalBranch &&
			     (kind.false_block == nullptr || current->condition.IsEmpty()))) {
				shaped = false; // an exit or an indirect branch
			}
			for (auto* successor: current->ImmSuccessors()) {
				if (successor != region.merge && region.body.insert(successor).second) {
					region.blocks.push_back(successor);
				}
			}
		}
		for (const auto* member: region.blocks) {
			for (const auto* predecessor: member->ImmPredecessors()) {
				if (!region.body.contains(predecessor) && !(member == entry && predecessor == block)) {
					shaped = false;
				}
			}
		}
		if (shaped) {
			regions.push_back(std::move(region));
		} else {
			result.problems["shape"]++;
		}
	}
	// A region holding another vote qualifies only once that one does.
	std::unordered_set<const Inst*> converted;
	for (const auto& region: regions) {
		converted.insert(region.vote);
	}
	std::vector<const char*> problems(regions.size(), nullptr);
	for (bool changed = true; changed;) {
		changed = false;
		for (size_t index = 0; index < regions.size(); index++) {
			if (problems[index] == nullptr &&
			    (problems[index] = RegionProblem(program, regions[index], converted)) != nullptr) {
				converted.erase(regions[index].vote);
				changed = true;
			}
		}
	}
	for (size_t index = 0; index < regions.size(); index++) {
		const auto& region = regions[index];
		if (problems[index] != nullptr) {
			result.problems[problems[index]]++;
			continue;
		}
		std::vector<const Inst*> known;
		Atoms                    atoms;
		AddAtoms(region.vote->Arg(0), region.negated_vote, atoms, &known);
		for (const auto* member: region.blocks) {
			auto& conditions = result.block_conditions[member];
			for (const auto* inst: known) {
				if (const auto found = std::ranges::lower_bound(conditions, inst);
				    found == conditions.end() || *found != inst) {
					conditions.insert(found, inst);
				}
			}
		}
		const auto predicate = region.vote->Arg(0);
		region.vote->ReplaceUsesWith(predicate);
		result.branches++;
	}
	return result;
}

uint32_t EliminateMaskedSelects(const BlockList& blocks, const BlockConditions& block_conditions) {
	// A guest vector write keeps an inactive lane's old value: select(exec, new, old). When the
	// result is only needed in lanes where exec is set, because every consumer is masked by it
	// again or runs only where it is set, the select is just its new value; where it is only
	// needed in lanes where exec is clear, it is the old one. In a chain of writes under one exec,
	// x1 = select(exec, f(x0), x0), the old value x0 is needed in both, so the select stays, but
	// each of its uses reads one side only: f reads the new value and x1's old operand the old.
	std::unordered_map<const Inst*, Demand> demands;
	const auto meet = [&](Value value, const Atoms& conditions) {
		value = value.Resolve();
		const auto* inst = value.IsImmediate() ? nullptr : value.TryInstruction();
		if (inst == nullptr) {
			return false;
		}
		auto& demand = demands[inst];
		if (!demand.reached) {
			demand = {true, conditions};
			return true;
		}
		// Needed wherever either use needs it: only the conditions both require remain.
		const auto before = demand.conditions.size();
		std::erase_if(demand.conditions, [&](const auto& condition) {
			return !std::ranges::binary_search(conditions, condition);
		});
		return demand.conditions.size() != before;
	};
	const Atoms every_lane;
	std::unordered_map<const Block*, Atoms> block_atoms;
	for (const auto& [block, conditions]: block_conditions) {
		auto& atoms = block_atoms[block];
		for (const auto* condition: conditions) {
			AddSignedAtoms(Value(const_cast<Inst*>(condition)), false, atoms);
		}
	}
	const auto running = [&](const Block* block) -> const Atoms& {
		const auto found = block_atoms.find(block);
		return found != block_atoms.end() ? found->second : every_lane;
	};
	// Calls visit(arg, lanes) with the lanes each operand of an instruction in a block running
	// in lanes "here" is needed in, given the lanes its result is. Nothing for one not reached.
	const auto operands = [&](const Inst& inst, const Atoms& here, auto&& visit) {
		const auto op = inst.GetOpcode();
		const bool external = std::ranges::any_of(inst.Uses(), [](const Use& use) {
			return use.user->Parent() == nullptr;
		});
		if (IsLaneLocal(op) && !external) {
			const auto found = demands.find(&inst);
			if (found == demands.end() || !found->second.reached) {
				return;
			}
			// A use in a block is needed only by the invocations that run it.
			Atoms conditions;
			std::ranges::set_union(found->second.conditions, here, std::back_inserter(conditions));
			if (!IsSelect(op)) {
				for (size_t arg = 0; arg < inst.NumArgs(); arg++) {
					visit(arg, conditions);
				}
				return;
			}
			// The new value is only chosen where the condition is set, the old one where it is clear.
			auto set   = conditions;
			auto clear = conditions;
			AddSignedAtoms(inst.Arg(0), false, set);
			AddSignedAtoms(inst.Arg(0), true, clear);
			visit(0, conditions);
			visit(1, set);
			visit(2, clear);
		} else if (IsPredicatedPerLane(inst) && !external) {
			const auto predicate = inst.Arg(inst.NumArgs() - 1);
			auto active = here;
			AddSignedAtoms(predicate, false, active);
			for (size_t arg = 0; arg + 1 < inst.NumArgs(); arg++) {
				visit(arg, active);
			}
			visit(inst.NumArgs() - 1, here);
		} else if (op == ValueOpcode::Phi && !external) {
			// An incoming value is needed by the invocations that come from its block.
			for (size_t arg = 0; arg < inst.NumArgs(); arg++) {
				visit(arg, running(inst.PhiBlock(arg)));
			}
		} else {
			for (size_t arg = 0; arg < inst.NumArgs(); arg++) {
				visit(arg, external ? every_lane : here);
			}
		}
	};
	for (bool changed = true; changed;) {
		changed = false;
		for (auto block = blocks.rbegin(); block != blocks.rend(); ++block) {
			const auto& here = running(*block);
			for (const auto& value: {(*block)->condition, (*block)->indirect_target}) {
				if (!value.IsEmpty()) changed |= meet(value, here);
			}
			auto& instructions = (*block)->Instructions();
			for (auto inst = instructions.rbegin(); inst != instructions.rend(); ++inst) {
				const auto* user = &*inst;
				operands(*user, here, [&](size_t arg, const Atoms& lanes) {
					changed |= meet(user->Arg(arg), lanes);
				});
			}
		}
	}
	const auto external = [](const Inst& inst) {
		return std::ranges::any_of(inst.Uses(), [](const Use& use) { return use.user->Parent() == nullptr; });
	};
	std::unordered_set<const Inst*> bypassed;
	// A select whose every use is decided gives way as a whole, block conditions included.
	for (auto* block: blocks) {
		for (auto& inst: *block) {
			if (!IsSelect(inst.GetOpcode()) || external(inst)) {
				continue;
			}
			const auto found = demands.find(&inst);
			if (found == demands.end() || !found->second.reached) {
				continue;
			}
			if (const auto chosen = ChosenOperand(inst, found->second.conditions); chosen != 0) {
				inst.ReplaceUsesWith(inst.Arg(chosen).Resolve());
				bypassed.insert(&inst);
			}
		}
	}
	// Then each use on its own. The value taken was needed through the select in those lanes
	// already, so the demands found above still hold.
	for (auto* block: blocks) {
		const auto& here = running(block);
		for (auto& inst: *block) {
			operands(inst, here, [&](size_t arg, const Atoms& lanes) {
				const auto old   = inst.Arg(arg).Resolve();
				auto       value = old;
				for (const auto* select = value.IsImmediate() ? nullptr : value.TryInstruction();
				     select != nullptr && IsSelect(select->GetOpcode());
				     select = value.IsImmediate() ? nullptr : value.TryInstruction()) {
					const auto chosen = ChosenOperand(*select, lanes);
					if (chosen == 0) {
						break;
					}
					bypassed.insert(select);
					value = select->Arg(chosen).Resolve();
				}
				if (!(value == old)) {
					inst.SetArg(arg, value);
				}
			});
		}
	}
	return static_cast<uint32_t>(bypassed.size());
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
