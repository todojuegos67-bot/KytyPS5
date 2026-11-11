#include "graphics/shader/recompiler/ir/passes/ExecSelects.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

// Values by instruction, open addressing with room for `count` (a node per entry, as std::unordered_map makes, took
// most of the walks' time in shaders of 20 thousand selects).
template <typename T>
class InstMap {
public:
	explicit InstMap(size_t count) {
		uint32_t bits = 4;
		while ((size_t {1} << bits) < count * 2) bits++;
		m_shift = 64u - bits;
		m_keys.assign(size_t {1} << bits, nullptr);
		m_values.resize(size_t {1} << bits);
	}

	// The key's value, `initial` when it had none (then also `true`).
	std::pair<T*, bool> Emplace(const Inst* key, T initial) {
		const auto mask = m_keys.size() - 1;
		for (auto slot = Slot(key);; slot = (slot + 1) & mask) {
			if (m_keys[slot] == key) return {&m_values[slot], false};
			if (m_keys[slot] == nullptr) {
				m_keys[slot]   = key;
				m_values[slot] = initial;
				return {&m_values[slot], true};
			}
		}
	}

	[[nodiscard]] const T* Find(const Inst* key) const {
		const auto mask = m_keys.size() - 1;
		for (auto slot = Slot(key);; slot = (slot + 1) & mask) {
			if (m_keys[slot] == key) return &m_values[slot];
			if (m_keys[slot] == nullptr) return nullptr;
		}
	}

private:
	[[nodiscard]] size_t Slot(const Inst* key) const {
		return static_cast<size_t>((reinterpret_cast<uintptr_t>(key) * 0x9e37'79b9'7f4a'7c15ull) >> m_shift);
	}

	std::vector<const Inst*> m_keys;
	std::vector<T>           m_values;
	uint32_t                 m_shift = 0;
};

// The dominator tree (Cooper, Harvey and Kennedy over the reverse post-order from the first block), with each block's
// interval in a depth-first walk of it: a block dominates those whose intervals lie in its own.
class DominatorTree {
public:
	explicit DominatorTree(const BlockList& blocks) {
		if (blocks.empty()) return;
		std::vector<Block*>                    post_order;
		std::vector<std::pair<Block*, size_t>> stack {{blocks.front(), 0}};
		m_index.emplace(blocks.front(), None);
		while (!stack.empty()) {
			auto*      block      = stack.back().first;
			const auto next       = stack.back().second;
			const auto successors = block->ImmSuccessors();
			if (next < successors.size()) {
				stack.back().second++;
				if (m_index.emplace(successors[next], None).second) stack.emplace_back(successors[next], 0);
				continue;
			}
			post_order.push_back(block);
			stack.pop_back();
		}
		const auto          count = static_cast<uint32_t>(post_order.size());
		std::vector<Block*> order(post_order.rbegin(), post_order.rend());
		for (uint32_t i = 0; i < count; i++) m_index[order[i]] = i;
		std::vector<uint32_t> idom(count, None);
		idom[0]              = 0;
		const auto intersect = [&](uint32_t a, uint32_t b) {
			while (a != b) {
				while (a > b) a = idom[a];
				while (b > a) b = idom[b];
			}
			return a;
		};
		for (bool changed = true; changed;) {
			changed = false;
			for (uint32_t i = 1; i < count; i++) {
				uint32_t dominator = None;
				for (const Block* predecessor: order[i]->ImmPredecessors()) {
					const auto found = m_index.find(predecessor);
					if (found == m_index.end() || idom[found->second] == None) continue;
					dominator = dominator == None ? found->second : intersect(found->second, dominator);
				}
				if (dominator != idom[i]) {
					idom[i] = dominator;
					changed = true;
				}
			}
		}
		std::vector<std::vector<uint32_t>> children(count);
		for (uint32_t i = 1; i < count; i++) children[idom[i]].push_back(i);
		m_enter.assign(count, 0);
		m_leave.assign(count, 0);
		uint32_t                                 clock = 0;
		std::vector<std::pair<uint32_t, size_t>> walk {{0u, 0}};
		m_enter[0] = clock++;
		while (!walk.empty()) {
			const auto node = walk.back().first;
			const auto next = walk.back().second;
			if (next < children[node].size()) {
				walk.back().second++;
				const auto child = children[node][next];
				m_enter[child]   = clock++;
				walk.emplace_back(child, 0);
				continue;
			}
			m_leave[node] = clock++;
			walk.pop_back();
		}
	}

	// `a` dominates `b` and is not `b` (a block the first one does not reach dominates none and has none).
	[[nodiscard]] bool StrictlyDominates(const Block* a, const Block* b) const {
		const auto found_a = m_index.find(a);
		const auto found_b = m_index.find(b);
		if (a == b || found_a == m_index.end() || found_b == m_index.end()) return false;
		return m_enter[found_a->second] < m_enter[found_b->second] && m_leave[found_b->second] < m_leave[found_a->second];
	}

private:
	static constexpr uint32_t None = std::numeric_limits<uint32_t>::max();

	std::unordered_map<const Block*, uint32_t> m_index;
	std::vector<uint32_t>                      m_enter;
	std::vector<uint32_t>                      m_leave;
};

bool IsSelect(ValueOpcode opcode) {
	return opcode == ValueOpcode::SelectU1 || opcode == ValueOpcode::SelectU32 || opcode == ValueOpcode::SelectF32;
}

// Each lane's result is of that lane's operands alone, without side effects and defined for any of them (UDiv32 is
// not: a zero divisor). Other lanes' values, memory, images and control flow are what it is not.
bool LaneWise(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::Identity:
		case ValueOpcode::BitCastU16F16:
		case ValueOpcode::BitCastF16U16:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU16U32:
		case ValueOpcode::ConvertU32U16:
		case ValueOpcode::ConvertU8U32:
		case ValueOpcode::ConvertU32U8:
		case ValueOpcode::ConvertF32F16:
		case ValueOpcode::ConvertF16F32:
		case ValueOpcode::ConvertS32F32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32S32:
		case ValueOpcode::ConvertF32U32:
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
		case ValueOpcode::SelectF32:
		case ValueOpcode::SelectU32:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
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
		case ValueOpcode::BitCount64:
		case ValueOpcode::FindUMsb32:
		case ValueOpcode::FindUMsb64:
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
		case ValueOpcode::FPCmpClass32:
		case ValueOpcode::FPAdd32:
		case ValueOpcode::FPSub32:
		case ValueOpcode::FPFma32:
		case ValueOpcode::FPMad32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPMin32:
		case ValueOpcode::FPMax32:
		case ValueOpcode::FPMinTri32:
		case ValueOpcode::FPMaxTri32:
		case ValueOpcode::FPMedTri32:
		case ValueOpcode::FPRecip32:
		case ValueOpcode::FPRecipIFlag32:
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
		case ValueOpcode::FPFract32: return true;
		default: return false;
	}
}

class Observation {
public:
	// `count`: the program's instructions.
	Observation(const BlockList& blocks, size_t count): m_tree(blocks), m_known(count), m_seen(count) {}

	// No use observes the select's lanes outside its predicate: its value there reaches, through lane-wise operations
	// alone, only the new operands of selects on the same predicate (which take their old operands in those lanes).
	// Whatever else it reaches (another lane, memory, an image, a branch) or a walk past the budget observes them.
	bool Unobserved(Inst* select) {
		const auto result = Walk(select);
		m_known.Emplace(select, result ? 1 : 0);
		return result;
	}

private:
	static constexpr size_t Budget = 4096;

	// First reached in this walk (a walk's stamp, not a cleared set: a set would be cleared per select).
	bool Reach(const Inst* inst) {
		const auto [walk, inserted] = m_seen.Emplace(inst, m_walk);
		if (inserted) return true;
		if (*walk == m_walk) return false;
		*walk = m_walk;
		return true;
	}

	bool Walk(Inst* select) {
		const auto  predicate  = select->Arg(0).Resolve();
		const auto* definition = predicate.TryInstruction();
		if (definition == nullptr || definition->Parent() == nullptr) return false;
		m_walk++;
		m_work.assign(1, select);
		Reach(select);
		size_t reached = 1;
		while (!m_work.empty()) {
			const auto* value = m_work.back();
			m_work.pop_back();
			for (const auto& use: value->Uses()) {
				auto*      user   = use.user;
				const auto opcode = user->GetOpcode();
				if (IsSelect(opcode) && use.operand != 0 && user->Arg(0).Resolve() == predicate) {
					if (use.operand == 1) continue;
					// The old operand: the user's own lanes outside the predicate.
					if (const auto* known = m_known.Find(user); known != nullptr) {
						if (*known == 0) return false;
						continue;
					}
				} else if (opcode == ValueOpcode::Phi) {
					// A phi's value may come around a loop, where a predicate defined in it is another value in the
					// next iteration: only a definition strictly dominating the phi's block lies outside every loop
					// through it (and leaves no edge into it between the select and the phi's uses).
					if (!m_tree.StrictlyDominates(definition->Parent(), user->Parent())) return false;
				} else if (!LaneWise(opcode)) {
					return false;
				}
				if (Reach(user)) {
					if (++reached > Budget) return false;
					m_work.push_back(user);
				}
			}
		}
		return true;
	}

	DominatorTree            m_tree;
	InstMap<uint8_t>         m_known; // decided selects: 1 unobserved
	InstMap<uint32_t>        m_seen;  // the walk that last reached an instruction
	std::vector<const Inst*> m_work;
	uint32_t                 m_walk = 0;
};

} // namespace

ExecSelectStats EliminateExecSelects(const BlockList& blocks) {
	ExecSelectStats    stats;
	std::vector<Inst*> selects;
	size_t             count = 0;
	for (auto* block: blocks) {
		for (auto& inst: block->Instructions()) {
			count++;
			const auto opcode = inst.GetOpcode();
			if (opcode != ValueOpcode::SelectU32 && opcode != ValueOpcode::SelectF32) continue;
			const auto predicate = inst.Arg(0).Resolve();
			if (predicate.IsImmediate()) continue;
			// Select(p, b, Select(p, a, o)): the inner select's lanes in p never reach this one.
			const auto old    = inst.Arg(2).Resolve();
			auto       bypass = old;
			for (const auto* inner = bypass.TryInstruction();
			     inner != nullptr && inner->GetOpcode() == opcode && inner->Arg(0).Resolve() == predicate;
			     inner = bypass.TryInstruction()) {
				bypass = inner->Arg(2).Resolve();
			}
			if (!(bypass == old)) {
				inst.SetArg(2, bypass);
				stats.bypassed++;
			}
			selects.push_back(&inst);
		}
	}
	if (selects.empty()) return stats;
	// Later selects first: a walk mostly ends at one already decided.
	Observation        observation(blocks, count);
	std::vector<Inst*> dropped;
	for (auto select = selects.rbegin(); select != selects.rend(); select++) {
		if ((*select)->HasUses() && observation.Unobserved(*select)) dropped.push_back(*select);
	}
	// (No identities: each leaves a Void instruction, which EliminateDeadCode takes with what fed only it.)
	for (auto* select: dropped) select->ReplaceUsesWith(select->Arg(1), false);
	stats.dropped = static_cast<uint32_t>(dropped.size());
	return stats;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
