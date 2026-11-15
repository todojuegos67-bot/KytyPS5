#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"

#include <algorithm>
#include <optional>

#include <queue>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

struct ChainResult {
	Value value;
	Inst* write = nullptr;
};

ChainResult SearchChain(Value value, uint32_t lane, uint32_t wave_size) {
	for (;;) {
		value      = value.Resolve();
		auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::WriteLane) {
			return {value};
		}
		const auto selector = inst->Arg(2).Resolve();
		if (!selector.IsImmediate() || selector.GetType() != Type::U32) {
			return {value};
		}
		if (selector.U32() % wave_size == lane) {
			return {value, inst};
		}
		value = inst->Arg(0);
	}
}

bool IsPossibleToEliminate(Value source, uint32_t lane, uint32_t wave_size) {
	std::queue<Value>         queue;
	std::unordered_set<Inst*> visited;
	queue.push(source);

	while (!queue.empty()) {
		const auto chain = SearchChain(queue.front(), lane, wave_size);
		queue.pop();
		if (chain.write != nullptr) {
			continue;
		}
		auto* inst = chain.value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::Phi || inst->NumArgs() == 0) {
			return false;
		}
		if (!visited.insert(inst).second) {
			continue;
		}
		for (size_t index = inst->NumArgs(); index-- > 0;) {
			queue.push(inst->Arg(index));
		}
	}
	return true;
}

// The lane's bit of a scalar EXEC mask, (mask >> (lane & 31)) & 1 != 0, with the mask ~x | x:
// set in every lane, as x | !x is (Marvel's Wolverine's pixel shaders save EXEC in an SGPR).
bool IsEveryLaneMaskBit(Value predicate) {
	const auto get = [](Value value, ValueOpcode opcode) -> const Inst* {
		const auto* inst = value.Resolve().TryInstruction();
		return inst != nullptr && inst->GetOpcode() == opcode ? inst : nullptr;
	};
	const auto* test = get(predicate, ValueOpcode::INotEqual32);
	const auto* bit  = test != nullptr && test->Arg(1).Resolve() == Value(0u)
	                       ? get(test->Arg(0), ValueOpcode::BitwiseAnd32) : nullptr;
	const auto* shift = bit != nullptr && bit->Arg(1).Resolve() == Value(1u)
	                        ? get(bit->Arg(0), ValueOpcode::ShiftRightLogical32) : nullptr;
	const auto* mask  = shift != nullptr ? get(shift->Arg(0), ValueOpcode::BitwiseOr32) : nullptr;
	const auto* lane  = shift != nullptr ? get(shift->Arg(1), ValueOpcode::BitwiseAnd32) : nullptr;
	if (mask == nullptr || lane == nullptr || lane->Arg(1).Resolve() != Value(31u)) return false;
	for (size_t index = 0; index < 2; index++) {
		const auto* negate = get(mask->Arg(index), ValueOpcode::BitwiseNot32);
		if (negate != nullptr && negate->Arg(0).Resolve() == mask->Arg(1 - index).Resolve()) return true;
	}
	return false;
}

// A reduction over a row of sixteen lanes written as a scan: row shifts right by 1, 2, 4 and 8,
// each combined into the lanes EXEC writes, read from the row's last lane. With every lane
// running that lane holds the reduction of the row. `base` is the value before the scan.
bool MatchRowScan(const Inst& read, uint32_t lane, Value& base, ValueOpcode& combine) {
	if ((lane & 15u) != 15u) return false;
	auto value = read.Arg(0).Resolve();
	Value exec;
	for (uint32_t shift = 8u; shift != 0u; shift >>= 1u) {
		const auto* update = value.TryInstruction();
		if (update == nullptr || update->GetOpcode() != ValueOpcode::DppUpdateU32) return false;
		const auto  flags    = update->Flags<DppMoveFlags>();
		const auto  previous = update->Arg(1).Resolve();
		const auto* step     = update->Arg(0).Resolve().TryInstruction();
		if (flags.control != (0x110u | shift) || flags.row_mask != 0xfu || flags.bank_mask != 0xfu ||
		    flags.dpp8 || step == nullptr || step->NumArgs() != 2u) return false;
		const auto* move = step->Arg(0).Resolve().TryInstruction();
		auto        kept = step->Arg(1).Resolve();
		if (move == nullptr || move->GetOpcode() != ValueOpcode::DppMoveU32) {
			move = step->Arg(1).Resolve().TryInstruction();
			kept = step->Arg(0).Resolve();
		}
		if (move == nullptr || move->GetOpcode() != ValueOpcode::DppMoveU32 || kept != previous ||
		    move->Arg(0).Resolve() != previous || move->Flags<DppMoveFlags>().control != flags.control)
			return false;
		const auto step_exec = update->Arg(2).Resolve();
		if (shift == 8u) {
			exec    = step_exec;
			combine = step->GetOpcode();
		}
		// A source past the row's start that reads zero is the identity of these only.
		const bool zero_identity = combine == ValueOpcode::UMax32 || combine == ValueOpcode::IAdd32 ||
		                           combine == ValueOpcode::BitwiseOr32;
		if (step_exec != exec || move->Arg(1).Resolve() != exec || step->GetOpcode() != combine ||
		    (!zero_identity && (combine != ValueOpcode::UMin32 && combine != ValueOpcode::BitwiseAnd32)) ||
		    (!zero_identity && flags.bound_control)) return false;
		value = previous;
	}
	// The scan is the reduction only when no lane is left out of it: EXEC of every lane,
	// which the shader sets as x | ~x.
	if (!(exec.IsImmediate() && exec.GetType() == Type::U1 && exec.U1()) && !IsEveryLaneMaskBit(exec)) {
		const auto* any = exec.TryInstruction();
		if (any == nullptr || any->GetOpcode() != ValueOpcode::LogicalOr) return false;
		const auto negated = [](Value first, Value second) {
			const auto* inst = first.Resolve().TryInstruction();
			return inst != nullptr && inst->GetOpcode() == ValueOpcode::LogicalNot &&
			       inst->Arg(0).Resolve() == second.Resolve();
		};
		if (!negated(any->Arg(0), any->Arg(1)) && !negated(any->Arg(1), any->Arg(0))) return false;
	}
	base = value;
	return true;
}

using PhiMap = std::unordered_map<Inst*, Inst*>;

Value GetRealValue(PhiMap& phi_map, Value source, uint32_t lane, uint32_t wave_size) {
	const auto chain = SearchChain(source, lane, wave_size);
	if (chain.write != nullptr) {
		return chain.write->Arg(1);
	}

	auto* inst = chain.value.ResolveInstruction();
	EXIT_IF(inst->GetOpcode() != ValueOpcode::Phi);
	const auto [entry, is_new] = phi_map.try_emplace(inst);
	if (!is_new) {
		return Value(entry->second);
	}

	auto* block           = inst->Parent();
	auto  insertion_point = std::find_if(block->begin(), block->end(),
	                                     [&](const Inst& candidate) { return &candidate == inst; });
	EXIT_IF(insertion_point == block->end());
	auto& phi = *block->PrependNewInst(insertion_point, ValueOpcode::Phi);
	phi.SetFlags(Type::U32);
	entry->second = &phi;

	std::vector<Value> arguments;
	arguments.reserve(inst->NumArgs());
	for (size_t index = 0; index < inst->NumArgs(); index++) {
		arguments.push_back(GetRealValue(phi_map, inst->Arg(index), lane, wave_size));
	}
	const auto first = arguments.front().Resolve();
	if (std::ranges::all_of(arguments,
	                        [&](Value argument) { return argument.Resolve() == first; })) {
		phi.ReplaceUsesWith(first);
	} else {
		for (size_t index = 0; index < arguments.size(); index++) {
			phi.AddPhiOperand(inst->PhiBlock(index), arguments[index]);
		}
	}
	return Value(&phi);
}

struct Reduction {
	ValueOpcode fold;
	uint32_t    identity;
	Value       lhs;
	Value       rhs;
};

// The fold that joins the two half-wave reads, and the value lanes left out of it must hold.
std::optional<Reduction> ReductionOf(const Inst& inst) {
	switch (inst.GetOpcode()) {
		case ValueOpcode::IAdd32:
		case ValueOpcode::UMax32:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32: return Reduction {inst.GetOpcode(), 0u, inst.Arg(0), inst.Arg(1)};
		case ValueOpcode::UMin32:
		case ValueOpcode::BitwiseAnd32:
			return Reduction {inst.GetOpcode(), UINT32_MAX, inst.Arg(0), inst.Arg(1)};
		case ValueOpcode::SMin32: return Reduction {inst.GetOpcode(), 0x7fffffffu, inst.Arg(0), inst.Arg(1)};
		case ValueOpcode::SMax32: return Reduction {inst.GetOpcode(), 0x80000000u, inst.Arg(0), inst.Arg(1)};
		default: return std::nullopt;
	}
}

// A full-EXEC prefix scan with both half-wave endpoint reads is a reduction. Wave64 first joins
// the two rows of each half; a wave32 half is one row.
// Fragment subgroups may have no physical invocation at either endpoint.
class WaveReductionMatcher {
public:
	WaveReductionMatcher(Block* block, ValueOpcode fold, uint32_t identity)
	    : block(block), fold(fold), identity(identity) {}

	Value Match(Value lhs, Value rhs, uint32_t wave_size) {
		const auto* low  = Get(lhs, ValueOpcode::ReadLane);
		const auto* high = Get(rhs, ValueOpcode::ReadLane);
		if (low == nullptr || high == nullptr || low->Arg(0).Resolve() != high->Arg(0).Resolve()) {
			return {};
		}
		const auto lane = [&](Value value, uint32_t expected) {
			value = value.Resolve();
			return value.IsImmediate() && value.GetType() == Type::U32 &&
			       value.U32() % wave_size == expected;
		};
		const auto half = wave_size / 2u;
		if (!((lane(low->Arg(1), half - 1u) && lane(high->Arg(1), wave_size - 1u)) ||
		      (lane(low->Arg(1), wave_size - 1u) && lane(high->Arg(1), half - 1u)))) {
			return {};
		}
		if (wave_size == 32u) {
			auto attempt = *this;
			return attempt.MatchScan(low->Arg(0));
		}
		const auto* joined = Get(Unwrap(low->Arg(0)), fold);
		if (joined == nullptr) return {};
		for (size_t index = 0; index < 2; index++) {
			auto attempt = *this;
			const auto* cross = attempt.Get(attempt.Unwrap(joined->Arg(index)), ValueOpcode::Permlane16U32);
			if (cross == nullptr) continue;
			const auto flags = cross->Flags<PermlaneFlags>();
			auto prefix = attempt.Unwrap(joined->Arg(1 - index));
			if (!flags.x16 || flags.fetch_inactive || !flags.bound_control ||
			    cross->Arg(1).Resolve() != Value(UINT32_MAX) ||
			    cross->Arg(2).Resolve() != Value(UINT32_MAX) ||
			    !attempt.FullExec(cross->Arg(3)) || attempt.Unwrap(cross->Arg(0)) != prefix) {
				continue;
			}
			if (const auto seed = attempt.MatchScan(prefix); !seed.IsEmpty()) return seed;
		}
		return {};
	}

private:
	// The row scan's input. A scan seeded from a saved EXEC must give the inactive lanes the fold's
	// identity.
	Value MatchScan(Value prefix) {
		for (const uint32_t shift: {8u, 4u, 2u, 1u}) {
			prefix = MatchRow(prefix, shift);
		}
		prefix = Unwrap(prefix);
		if (prefix.IsEmpty() || saved_exec.IsEmpty()) return prefix;
		const auto* seed = Get(prefix, ValueOpcode::SelectU32);
		if (seed != nullptr && seed->Arg(2).Resolve() == Value(identity)) {
			const auto predicate = seed->Arg(0).Resolve();
			const auto* subset = Get(predicate, ValueOpcode::LogicalAnd);
			if (predicate == saved_exec ||
			    (subset != nullptr && (subset->Arg(0).Resolve() == saved_exec ||
			                           subset->Arg(1).Resolve() == saved_exec))) {
				return prefix;
			}
		}
		return {};
	}

	Inst* Get(Value value, ValueOpcode opcode) const {
		auto* inst = value.Resolve().TryInstruction();
		return inst != nullptr && inst->Parent() == block && inst->GetOpcode() == opcode
		           ? inst : nullptr;
	}

	// Constant propagation folds the x | ~x EXEC of s_orn2_saveexec to true; the unfolded form
	// records x as the saved EXEC.
	bool FullExec(Value predicate) {
		predicate = predicate.Resolve();
		if (predicate == Value(true)) return true;
		const auto* either = Get(predicate, ValueOpcode::LogicalOr);
		if (either == nullptr) return false;
		for (size_t index = 0; index < 2; index++) {
			const auto* negate = Get(either->Arg(index), ValueOpcode::LogicalNot);
			if (negate == nullptr || negate->Arg(0).Resolve() != either->Arg(1 - index).Resolve()) continue;
			const auto original = negate->Arg(0).Resolve();
			if (saved_exec.IsEmpty()) saved_exec = original;
			return saved_exec == original;
		}
		return false;
	}

	Value Unwrap(Value value) {
		while (const auto* select = Get(value, ValueOpcode::SelectU32)) {
			if (!FullExec(select->Arg(0))) break;
			value = select->Arg(1);
		}
		return value.Resolve();
	}

	static bool IsRowShift(const Inst& inst, uint32_t shift) {
		const auto flags = inst.Flags<DppMoveFlags>();
		return flags.control == 0x110u + shift && flags.row_mask == 0xf &&
		       flags.bank_mask == 0xf && !flags.fetch_inactive && !flags.bound_control && !flags.dpp8;
	}

	Value MatchRow(Value value, uint32_t shift) {
		const auto* update = Get(Unwrap(value), ValueOpcode::DppUpdateU32);
		if (update == nullptr || !IsRowShift(*update, shift) || !FullExec(update->Arg(2))) return {};
		const auto* combined = Get(update->Arg(0), fold);
		if (combined == nullptr) return {};
		for (size_t index = 0; index < 2; index++) {
			const auto* move = Get(combined->Arg(index), ValueOpcode::DppMoveU32);
			if (move == nullptr || !IsRowShift(*move, shift) || !FullExec(move->Arg(1))) continue;
			const auto previous = Unwrap(combined->Arg(1 - index));
			if (Unwrap(move->Arg(0)) == previous && Unwrap(update->Arg(1)) == previous) return previous;
		}
		return {};
	}

	Block*      block;
	ValueOpcode fold;
	uint32_t    identity;
	Value       saved_exec;
};

} // namespace

ReadLaneStats EliminateReadLane(Program& program, uint32_t wave_size) {
	ReadLaneStats stats;
	if (wave_size != 32u && wave_size != 64u) {
		return stats;
	}

	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			const auto opcode = inst.GetOpcode();
			if (const auto reduction = ReductionOf(inst)) {
				const auto seed = WaveReductionMatcher(block, reduction->fold, reduction->identity)
				                      .Match(reduction->lhs, reduction->rhs, wave_size);
				if (!seed.IsEmpty()) {
					const auto position = std::find_if(block->begin(), block->end(),
					                                  [&](const Inst& candidate) { return &candidate == &inst; });
					auto& subgroup = *block->PrependNewInst(position, ValueOpcode::SubgroupReduceU32, {seed});
					subgroup.SetFlags(reduction->fold);
					inst.ReplaceUsesWith(Value(&subgroup));
					stats.rewritten_reductions++;
				}
				continue;
			}
			if (opcode != ValueOpcode::ReadLane) {
				continue;
			}
			const auto selector = inst.Arg(1).Resolve();
			if (!selector.IsImmediate() || selector.GetType() != Type::U32) {
				continue;
			}

			const auto lane  = selector.U32() % wave_size;
			Value       base;
			ValueOpcode combine;
			if (MatchRowScan(inst, lane, base, combine)) {
				// A host subgroup can lack lanes of the guest wave: the scan through a missing
				// lane is lost and the read of a missing lane is undefined. The reduction over
				// the row's lanes that exist is what the scan gives the guest.
				const auto position = std::find_if(block->begin(), block->end(),
				    [&](const Inst& candidate) { return &candidate == &inst; });
				const auto emit = [&](ValueOpcode op, std::initializer_list<Value> args) {
					return &*block->PrependNewInst(position, op, args);
				};
				const auto row = emit(ValueOpcode::ShiftRightLogical32,
				                      {Value(emit(ValueOpcode::LaneId, {})), Value(4u)});
				auto* reduce = emit(ValueOpcode::LaneReduceU32,
				    {base, Value(emit(ValueOpcode::IEqual32, {Value(row), Value(lane >> 4u)}))});
				reduce->SetFlags(combine);
				inst.ReplaceUsesWith(Value(reduce));
				stats.rewritten_reads++;
				continue;
			}
			const auto chain = SearchChain(inst.Arg(0), lane, wave_size);
			if (chain.write != nullptr) {
				inst.ReplaceUsesWith(chain.write->Arg(1));
				stats.rewritten_reads++;
				continue;
			}
			auto* producer = chain.value.TryInstruction();
			if (producer == nullptr || producer->GetOpcode() != ValueOpcode::Phi ||
			    !IsPossibleToEliminate(chain.value, lane, wave_size)) {
				continue;
			}

			PhiMap phi_map;
			inst.ReplaceUsesWith(GetRealValue(phi_map, chain.value, lane, wave_size));
			stats.rewritten_reads++;
		}
	}
	return stats;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
