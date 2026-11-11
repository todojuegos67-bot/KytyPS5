#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/LinearSrt.h"
#include "live-census.h"
#include "native-resource-runtime.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <unordered_map>
#include <unordered_set>

extern "C" {
volatile std::atomic<uint32_t> kyty_local_srt_predicate_mode {1};
volatile std::atomic<uint32_t> kyty_local_preparation_scratch_mode {1};
volatile std::atomic<uint32_t> kyty_local_preparation_lookup_mode {1};
volatile std::atomic<uint32_t> kyty_local_preparation_trim_mode {1};
}

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "vertex";
		case ShaderType::Pixel: return "pixel";
		case ShaderType::Fetch: return "fetch";
		case ShaderType::Compute: return "compute";
		default: return "unknown";
	}
}

std::string Diagnostic(const ResourcePlan& program, uint32_t pc, const std::string& message) {
	return fmt::format("shader SRT: hash=0x{:016x} stage={} pc=0x{:08x} {}", program.shader_hash,
	                   StageName(program.stage), pc, message);
}

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto kind = values.memory_info[index].kind;
	return (op == ValueOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) ||
	       (op == ValueOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer);
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::BitCount32:
		case ValueOpcode::FindILsb32:
		case ValueOpcode::FindUMsb32:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMin32:
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
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::UGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type)
	    : m_program(program), m_type(type) {}

	bool Run(Value value) { return Validate(value); }

	const std::string& Reason() const { return m_reason; }

private:
	bool Reject(std::string reason) {
		if (m_reason.empty()) {
			m_reason = std::move(reason);
		}
		return false;
	}

	std::string Describe(Value value, uint32_t depth = 0) const {
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return value.IsImmediate() && value.GetType() == Type::U32
			           ? fmt::format("0x{:08x}", value.U32())
			           : std::string("opaque");
		}
		const auto op   = inst->GetOpcode();
		auto       text = std::string(ValueOpcodeName(op));
		if (op == ValueOpcode::GetUserData && inst->NumArgs() == 1 &&
		    inst->Arg(0).GetType() == Type::ScalarReg) {
			return text + fmt::format(" s{}", RegIndex(inst->Arg(0).ScalarRegister()));
		}
		if (op == ValueOpcode::ReadConst && inst->NumArgs() == 2 &&
		    inst->Arg(1).Resolve().IsImmediate()) {
			return text + fmt::format(" slot={}", inst->Arg(1).Resolve().U32());
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
			const auto flags = inst->Flags<MemoryFlags>();
			text += fmt::format(" pc=0x{:08x}", flags.pc);
			if (flags.index < m_program.memory_info.size()) {
				text += fmt::format(" offset=0x{:x}", m_program.memory_info[flags.index].offset);
			}
			return text;
		}
		if (inst->NumArgs() == 0 || depth >= 3u) {
			return text;
		}
		text += '(';
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			text += index == 0 ? "" : ", ";
			text += Describe(inst->Arg(index), depth + 1u);
		}
		return text + ')';
	}

	std::string DescribePhi(Value value) const {
		std::vector<Value>              leaves;
		std::vector<Value>              pending {value};
		std::unordered_set<const Inst*> seen;
		while (!pending.empty()) {
			const auto current = pending.back().Resolve();
			pending.pop_back();
			const auto* inst = current.TryInstruction();
			if (inst != nullptr && inst->GetOpcode() == ValueOpcode::Phi) {
				if (seen.insert(inst).second) {
					for (size_t index = 0; index < inst->NumArgs(); index++) {
						pending.push_back(inst->Arg(index));
					}
				}
				continue;
			}
			if (std::ranges::none_of(leaves, [&](Value known) {
				    return EquivalentValue(m_program, known, current);
			    })) {
				leaves.push_back(current);
			}
		}
		auto text = fmt::format("Phi merges {} unequal values:", leaves.size());
		for (size_t index = 0; index < leaves.size() && index < 6u; index++) {
			text += fmt::format(" [{}] {}", index, Describe(leaves[index]));
		}
		return text;
	}

	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type == RuntimeValueType::Integer &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return Reject("operand is not an integer");
			}
		}
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		if (!m_visiting.insert(inst).second) {
			if (require_uniform) {
				Reject(fmt::format("{} is cyclic", ValueOpcodeName(inst->GetOpcode())));
			}
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			if (!valid && require_uniform) {
				Reject(fmt::format("{} cannot be evaluated before the draw",
				                   ValueOpcodeName(inst->GetOpcode())));
			}
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU8 ||
		    op == ValueOpcode::UndefU16 || op == ValueOpcode::UndefU32 ||
		    op == ValueOpcode::UndefU64 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type == RuntimeValueType::Integer && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				Reject(DescribePhi(value));
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u ||
			    (source->GetOpcode() != ValueOpcode::CompositeConstructU32x2 &&
			     source->GetOpcode() != ValueOpcode::IAddCarry32)) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
	std::string                     m_reason;
};

class PlanBuilder {
public:
	explicit PlanBuilder(Program& program): m_program(program) {}

	void Run() {
		m_program.srt_reads.clear();
		m_program.dynamic_reads.clear();
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				const auto op = inst.GetOpcode();
				if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
					const auto flags = inst.Flags<MemoryFlags>();
					if (flags.index < m_program.memory_info.size()) {
						const auto kind       = m_program.memory_info[flags.index].kind;
						const bool crosswired = (op == ValueOpcode::LoadAddressU32 &&
						                         kind == ResourceKind::ScalarBuffer) ||
						                        (op == ValueOpcode::ReadConstBuffer &&
						                         kind == ResourceKind::ScalarAddress);
						if (crosswired) {
							Fail(flags.pc,
							     fmt::format("{} has incompatible scalar memory metadata",
							                 ValueOpcodeName(op)));
						}
					}
				}
				if (IsDescriptorHandle(inst.GetOpcode())) {
					for (size_t index = 0; index < inst.NumArgs(); index++) {
						Collect(inst.Arg(index), 0);
					}
				}
			}
		}
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				if (inst.GetOpcode() == ValueOpcode::LoadAddressU32 && IsRawRead(m_program, inst) &&
				    inst.Arg(1).Resolve().IsImmediate() &&
				    ValidateRuntimeValue(m_program, Value(&inst))) {
					Collect(Value(&inst), inst.Flags<MemoryFlags>().pc);
				}
			}
		}
		PatchReads();
	}

private:
	struct Patch {
		Inst*    inst = nullptr;
		uint32_t slot = 0;
		bool     keep = false;
	};

	[[noreturn]] void Fail(uint32_t pc, const std::string& message) const {
		const auto diagnostic = Diagnostic(m_program, pc, message);
		EXIT("shader SRT planning failed: %s", diagnostic.c_str());
		std::abort();
	}

	void Collect(Value value, uint32_t use_pc) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			return;
		}
		auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			Fail(use_pc, "invalid typed planning value");
		}
		// A value is collected once; one reached again on its own path closes a cycle, which is fine through a
		// phi (m_phis[d] counts the phis above depth d) and an error otherwise.
		const auto [state, fresh] = m_states.try_emplace(inst, static_cast<uint32_t>(m_phis.size() - 1));
		if (!fresh) {
			if (state->second == Collected || m_phis.back() > m_phis[state->second]) {
				return;
			}
			Fail(use_pc, fmt::format("cyclic typed planning value {} without a phi",
			                         ValueOpcodeName(inst->GetOpcode())));
		}
		auto& depth = state->second;
		m_phis.push_back(m_phis.back() + (inst->GetOpcode() == ValueOpcode::Phi ? 1 : 0));
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			Collect(inst->Arg(index), use_pc);
		}
		m_phis.pop_back();
		depth = Collected;
		if (!IsRawRead(m_program, *inst)) {
			return;
		}
		const auto offset = inst->Arg(1).Resolve();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32) {
			if (std::ranges::find(m_program.dynamic_reads, value) ==
			    m_program.dynamic_reads.end()) {
				m_program.dynamic_reads.push_back(value);
			}
			return;
		}
		// Only reads of one structural hash can be equivalent; within it the earliest slot wins, as
		// when every read was compared (big compute shaders: thousands of reads, 100+ ms a translation).
		auto& bucket = m_buckets[StructuralHash(value)];
		for (const auto slot: bucket) {
			if (EquivalentValue(m_program, value, m_program.srt_reads[slot].value)) {
				m_patches.push_back({inst, slot, false});
				return;
			}
		}
		const auto slot = static_cast<uint32_t>(m_program.srt_reads.size());
		m_program.srt_reads.push_back({value, slot});
		bucket.push_back(slot);
		m_patches.push_back({inst, slot, true});
	}

	// Equal for every pair EquivalentValue accepts: it hashes only what that compares (opcode, type,
	// argument count, immediates, arguments; a phi's blocks, not its arguments, so no cycle is followed).
	uint64_t StructuralHash(Value value) {
		const auto mix = [](uint64_t hash, uint64_t part) {
			return hash ^ (part + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u));
		};
		value           = value.Resolve();
		const auto type = static_cast<uint64_t>(value.GetType());
		if (value.IsImmediate()) {
			return mix(type, value.GetType() == Type::U32 ? value.U32() : value.GetType() == Type::U64 ? value.U64() : 0);
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return type;
		}
		if (const auto found = m_hashes.find(inst); found != m_hashes.end()) {
			return found->second;
		}
		auto hash = mix(mix(type, static_cast<uint64_t>(inst->GetOpcode())), inst->NumArgs());
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			hash = mix(hash, inst->GetOpcode() == ValueOpcode::Phi ? reinterpret_cast<uintptr_t>(inst->PhiBlock(index))
			                                                         : StructuralHash(inst->Arg(index)));
		}
		m_hashes.emplace(inst, hash);
		return hash;
	}

	void PatchReads() {
		// Each patched read's place in its block, found in one pass (inserting keeps list positions valid).
		std::unordered_map<const Inst*, decltype(m_program.blocks.front()->Instructions().begin())> places;
		for (const auto& patch: m_patches) places.emplace(patch.inst, decltype(places)::mapped_type {});
		for (auto* block: m_program.blocks) {
			auto& list = block->Instructions();
			for (auto it = list.begin(); it != list.end(); ++it)
				if (const auto place = places.find(&*it); place != places.end()) place->second = it;
		}
		for (const auto& patch: m_patches) {
			auto* block = patch.inst->Parent();
			auto  where = places.at(patch.inst);
			const auto resource =
			    Value(&*block->PrependNewInst(where, ValueOpcode::GetSrtResource));
			const auto flat = Value(&*block->PrependNewInst(where, ValueOpcode::ReadConst,
			                                                {resource, Value(patch.slot)}));
			const auto uses = patch.inst->Uses();
			for (const auto& use: uses) {
				use.user->SetArg(use.operand, flat);
			}
			for (auto& info: m_program.block_info) {
				if (info.condition.Resolve() == Value(patch.inst)) {
					info.condition = flat;
				}
				if (info.indirect_target.Resolve() == Value(patch.inst)) {
					info.indirect_target = flat;
				}
			}
			if (patch.keep) {
				const auto memory = patch.inst->Flags<MemoryFlags>().index;
				if (memory < m_program.memory_info.size()) {
					m_program.memory_info[memory].planning_only = true;
				}
				block->AppendNewInst(ValueOpcode::ReferenceU32, {Value(patch.inst)});
			}
		}
	}

	static constexpr uint32_t Collected = UINT32_MAX;

	Program&                                                m_program;
	std::unordered_map<const Inst*, uint32_t>               m_states; // depth on the path, or Collected
	std::vector<uint32_t>                                   m_phis {0};
	std::vector<Patch>                                      m_patches;
	std::unordered_map<uint64_t, std::vector<uint32_t>>     m_buckets; // srt_reads slots by StructuralHash
	std::unordered_map<const Inst*, uint64_t>               m_hashes;
};

// Entries live only for one evaluation. Open addressing avoids allocating a
// node for each IR value; entries are never evicted, including on hash collision.
class EvaluationCache {
public:
	void Reserve(size_t expected) {
		m_entries.resize(std::bit_ceil(std::max<size_t>(16, expected * 2)));
	}
	bool Find(const Inst* inst, uint64_t& result) {
		if (m_entries.empty()) m_entries.resize(16);
		const auto slot = FindSlot(inst);
		if (m_entries[slot].inst == nullptr) {
			return false;
		}
		result = m_entries[slot].value;
		return true;
	}
	void Insert(const Inst* inst, uint64_t value) {
		if ((m_size + 1) * 2 >= m_entries.size()) {
			auto old = std::move(m_entries);
			m_entries.resize(old.size() * 2);
			for (const auto& entry: old) {
				if (entry.inst != nullptr) {
					m_entries[FindSlot(entry.inst)] = entry;
				}
			}
		}
		auto& entry = m_entries[FindSlot(inst)];
		if (entry.inst == nullptr) {
			++m_size;
		}
		entry = {inst, value};
	}
private:
	struct Entry {
		const Inst* inst = nullptr;
		uint64_t value = 0;
	};
	size_t FindSlot(const Inst* inst) const {
		auto hash = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(inst)) >> 4;
		hash ^= hash >> 32;
		hash *= 0x9e3779b97f4a7c15ull;
		hash ^= hash >> 29;
		const auto mask = m_entries.size() - 1;
		auto slot = static_cast<size_t>(hash) & mask;
		while (m_entries[slot].inst != nullptr && m_entries[slot].inst != inst) {
			slot = (slot + 1) & mask;
		}
		return slot;
	}
	std::vector<Entry> m_entries;
	size_t m_size = 0;
};

class Evaluator {
public:
	Evaluator(const ResourcePlan& program, const SrtRuntime& runtime,
	          std::span<const uint8_t> clean_flat_slots = {}, Evaluator* clean_evaluator = nullptr,
	          Value active_mask = {})
	    : m_program(program), m_runtime(runtime), m_clean_flat_slots(clean_flat_slots),
	      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask.Resolve()) {}

	bool Evaluate(Value value, uint32_t& result) {
		uint64_t wide = 0;
		if (!EvaluateWide(value, wide)) {
			return false;
		}
		result = static_cast<uint32_t>(wide);
		return true;
	}

private:
	static float Float32(uint64_t bits) {
		return std::bit_cast<float>(static_cast<uint32_t>(bits));
	}

	static uint64_t Float32Bits(float value) { return std::bit_cast<uint32_t>(value); }

	bool EvaluateWide(Value value, uint64_t& result) {
		if (value.IsImmediate()) {
			switch (value.GetType()) {
				case Type::U1: result = value.U1(); return true;
				case Type::U8: result = value.U8(); return true;
				case Type::U16: result = value.U16(); return true;
				case Type::U32: result = value.U32(); return true;
				case Type::U64: result = value.U64(); return true;
				case Type::F32: result = Float32Bits(value.F32Value()); return true;
				default: return false;
			}
		}
		auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return false;
		}
		if (!m_reserved) {
			// Most draws evaluate only the descriptor slice of a much larger IR.
			m_cache.Reserve(std::min<size_t>(m_program.value_storage.size(), 256));
			m_visiting.reserve(std::min<size_t>(m_program.value_storage.size(), 64));
			m_reserved = true;
		}
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) &&
		    inst->NumArgs() == 3 && inst->Arg(0).Resolve() == m_active_mask) {
			return EvaluateWide(inst->Arg(1), result);
		}
		if (m_cache.Find(inst, result)) {
			return true;
		}
		if (std::ranges::find(m_visiting, inst) != m_visiting.end()) {
			return false;
		}
		m_visiting.push_back(inst);
		uint64_t out = 0;
		const bool evaluated = EvaluateInst(*inst, out);
		m_visiting.pop_back();
		if (!evaluated) {
			return false;
		}
		m_cache.Insert(inst, out);
		result = out;
		return true;
	}

	bool Arg(const Inst& inst, size_t index, uint64_t& result) {
		return EvaluateWide(inst.Arg(index), result);
	}

	bool EvaluatePhi(const Inst& inst, uint64_t& result) {
		const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
		return !value.IsEmpty() && EvaluateWide(value, result);
	}

	bool EvaluateExtract(const Inst& inst, uint64_t& result) {
		const auto index = inst.Arg(1).Resolve();
		if (!index.IsImmediate() || index.GetType() != Type::U32) {
			return false;
		}
		const auto component = index.U32();
		if (component >= 2u) {
			return false;
		}
		if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
			uint64_t packed = 0;
			if (!Arg(inst, 0, packed)) {
				return false;
			}
			result = static_cast<uint32_t>(packed >> (component * 32u));
			return true;
		}
		const auto* source = inst.Arg(0).ResolveInstruction();
		if (source == nullptr) {
			return false;
		}
		if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
			return EvaluateWide(source->Arg(component), result);
		}
		if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
			uint64_t lhs = 0;
			uint64_t rhs = 0;
			if (!Arg(*source, 0, lhs) || !Arg(*source, 1, rhs)) {
				return false;
			}
			const auto sum =
			    static_cast<uint64_t>(static_cast<uint32_t>(lhs)) + static_cast<uint32_t>(rhs);
			result =
			    component == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
			return true;
		}
		return false;
	}

	bool EvaluateRawRead(const Inst& inst, uint64_t& result) {
		const auto flags = inst.Flags<MemoryFlags>();
		if (flags.index >= m_program.memory_info.size()) {
			return false;
		}
		const auto& mem    = m_program.memory_info[flags.index];
		const auto* handle = inst.Arg(0).ResolveInstruction();
		if (handle == nullptr) {
			return false;
		}
		uint64_t low    = 0;
		uint64_t high   = 0;
		uint64_t offset = 0;
		if (!Arg(*handle, 0, low) || !Arg(*handle, 1, high) || !Arg(inst, 1, offset)) {
			return false;
		}
		const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
		const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
		uint64_t   address   = 0;
		if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer) {
			uint64_t records = 0;
			uint64_t word3   = 0;
			if (handle->NumArgs() != 4u || !Arg(*handle, 2, records) || !Arg(*handle, 3, word3)) {
				return false;
			}
			if (immediate < 0) {
				return false;
			}
			const auto byte_offset =
			    static_cast<uint64_t>(immediate) + static_cast<uint32_t>(offset);
			const auto aligned = byte_offset & ~uint64_t {3};
			const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
			const auto size = stride == 0u
			                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
			                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
			if (aligned > size || size - aligned < sizeof(uint32_t)) {
				return false;
			}
			address = ((base & ~uint64_t {3}) + byte_offset) & ~uint64_t {3};
		} else {
			const auto relative = (immediate & ~int64_t {3}) +
			                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
			if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
				return false;
			}
		}
		uint32_t word = 0;
		if (m_runtime.read_memory != nullptr) {
			if (!m_runtime.read_memory(m_runtime.userdata, address, &word)) {
				return false;
			}
		} else {
			std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
		}
		result = word;
		return true;
	}

	bool EvaluateInst(const Inst& inst, uint64_t& result) {
		uint64_t   a       = 0;
		uint64_t   b       = 0;
		uint64_t   c       = 0;
		const auto binary  = [&]() { return Arg(inst, 0, a) && Arg(inst, 1, b); };
		const auto ternary = [&]() {
			return Arg(inst, 0, a) && Arg(inst, 1, b) && Arg(inst, 2, c);
		};
		switch (inst.GetOpcode()) {
			case ValueOpcode::Identity: return Arg(inst, 0, result);
			case ValueOpcode::GetUserData: {
				const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
				if (reg < m_program.user_data_base ||
				    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
					return false;
				}
				result = m_runtime.user_data[reg - m_program.user_data_base];
				return true;
			}
			case ValueOpcode::GetShaderBase: result = m_runtime.shader_base; return true;
			case ValueOpcode::Phi: return EvaluatePhi(inst, result);
			case ValueOpcode::ReadFirstLane: {
				Evaluator active(m_program, m_runtime, m_clean_flat_slots, m_clean_evaluator,
				                 inst.Arg(1));
				return active.EvaluateWide(inst.Arg(0), result);
			}
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32: return Arg(inst, 0, result);
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2: return EvaluateExtract(inst, result);
			case ValueOpcode::CompositeConstructU64:
				if (!binary()) {
					return false;
				}
				result = static_cast<uint32_t>(a) |
				         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
				return true;
			case ValueOpcode::ReadConst: {
				const auto slot = inst.Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
				    slot.U32() >= m_program.srt_reads.size()) {
					return false;
				}
				if (slot.U32() < m_clean_flat_slots.size() &&
				    m_clean_flat_slots[slot.U32()] != 0u && m_clean_evaluator != nullptr) {
					return m_clean_evaluator->EvaluateWide(m_program.srt_reads[slot.U32()].value,
					                                       result);
				}
				return EvaluateWide(m_program.srt_reads[slot.U32()].value, result);
			}
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer:
				if (IsRawRead(m_program, inst)) {
					return EvaluateRawRead(inst, result);
				}
				break;
			case ValueOpcode::IAdd32:
				if (binary()) {
					result = static_cast<uint32_t>(a + b);
					return true;
				}
				return false;
			case ValueOpcode::IAdd64:
				if (binary()) {
					result = a + b;
					return true;
				}
				return false;
			case ValueOpcode::ISub32:
				if (binary()) {
					result = static_cast<uint32_t>(a - b);
					return true;
				}
				return false;
			case ValueOpcode::ISub64:
				if (binary()) {
					result = a - b;
					return true;
				}
				return false;
			case ValueOpcode::IMul32:
				if (binary()) {
					result = static_cast<uint32_t>(a * b);
					return true;
				}
				return false;
			case ValueOpcode::IMul64:
				if (binary()) {
					result = a * b;
					return true;
				}
				return false;
			case ValueOpcode::UMin32:
				if (binary()) {
					result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
					return true;
				}
				return false;
			case ValueOpcode::ConvertF32U32:
				if (Arg(inst, 0, a)) {
					result = Float32Bits(static_cast<float>(static_cast<uint32_t>(a)));
					return true;
				}
				return false;
			case ValueOpcode::ConvertU32F32:
				if (Arg(inst, 0, a)) {
					const auto value = Float32(a);
					if (!std::isfinite(value) || value < 0.0f ||
					    static_cast<double>(value) > UINT32_MAX) {
						return false;
					}
					result = static_cast<uint32_t>(value);
					return true;
				}
				return false;
			case ValueOpcode::FPMul32:
				if (binary()) {
					result = Float32Bits(Float32(a) * Float32(b));
					return true;
				}
				return false;
			case ValueOpcode::FPTrunc32:
				if (Arg(inst, 0, a)) {
					result = Float32Bits(std::trunc(Float32(a)));
					return true;
				}
				return false;
			case ValueOpcode::FPIsNan32:
				if (Arg(inst, 0, a)) {
					result = std::isnan(Float32(a));
					return true;
				}
				return false;
			case ValueOpcode::FPOrdLessThanEqual32:
				if (binary()) {
					result = Float32(a) <= Float32(b);
					return true;
				}
				return false;
			case ValueOpcode::FPOrdGreaterThanEqual32:
				if (binary()) {
					result = Float32(a) >= Float32(b);
					return true;
				}
				return false;
			case ValueOpcode::BitwiseAnd32:
				if (binary()) {
					result = static_cast<uint32_t>(a & b);
					return true;
				}
				return false;
			case ValueOpcode::BitwiseAnd64:
				if (binary()) {
					result = a & b;
					return true;
				}
				return false;
			case ValueOpcode::BitwiseOr32:
				if (binary()) {
					result = static_cast<uint32_t>(a | b);
					return true;
				}
				return false;
			case ValueOpcode::BitwiseXor32:
				if (binary()) {
					result = static_cast<uint32_t>(a ^ b);
					return true;
				}
				return false;
			case ValueOpcode::BitwiseNot32:
				if (Arg(inst, 0, a)) {
					result = ~static_cast<uint32_t>(a);
					return true;
				}
				return false;
			case ValueOpcode::BitCount32:
				if (Arg(inst, 0, a)) {
					result = static_cast<uint32_t>(std::popcount(static_cast<uint32_t>(a)));
					return true;
				}
				return false;
			case ValueOpcode::FindILsb32:
				if (Arg(inst, 0, a)) {
					const auto bits = static_cast<uint32_t>(a);
					result          = bits == 0u ? UINT32_MAX
					                             : static_cast<uint32_t>(std::countr_zero(bits));
					return true;
				}
				return false;
			case ValueOpcode::FindUMsb32:
				if (Arg(inst, 0, a)) {
					const auto bits = static_cast<uint32_t>(a);
					result          = bits == 0u ? UINT32_MAX
					                             : static_cast<uint32_t>(31 - std::countl_zero(bits));
					return true;
				}
				return false;
			case ValueOpcode::ShiftLeftLogical32:
				if (binary()) {
					result = static_cast<uint32_t>(a) << (b & 31u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftLeftLogical64:
				if (binary()) {
					result = a << (b & 63u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightLogical32:
				if (binary()) {
					result = static_cast<uint32_t>(a) >> (b & 31u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightLogical64:
				if (binary()) {
					result = a >> (b & 63u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightArithmetic32:
				if (binary()) {
					result = static_cast<uint32_t>(
					    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightArithmetic64:
				if (binary()) {
					result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
					return true;
				}
				return false;
			case ValueOpcode::BitFieldUExtract:
				if (ternary()) {
					const auto offset = static_cast<uint32_t>(b);
					const auto width  = static_cast<uint32_t>(c);
					if (offset > 32u || width > 32u - offset) {
						return false;
					}
					const auto mask = width == 32u  ? UINT32_MAX
					                  : width == 0u ? 0u
					                                : (uint32_t {1} << width) - 1u;
					result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
					return true;
				}
				return false;
			case ValueOpcode::BitFieldSExtract:
				if (ternary()) {
					const auto offset = static_cast<uint32_t>(b);
					const auto width  = static_cast<uint32_t>(c);
					if (offset > 32u || width > 32u - offset) {
						return false;
					}
					if (width == 0u) {
						result = 0;
						return true;
					}
					const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
					auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
					if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
						bits |= ~mask;
					}
					result = bits;
					return true;
				}
				return false;
			case ValueOpcode::BitFieldInsert: {
				uint64_t d = 0;
				if (!ternary() || !Arg(inst, 3, d)) {
					return false;
				}
				const auto offset = static_cast<uint32_t>(c);
				const auto width  = static_cast<uint32_t>(d);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				if (width == 0u) {
					result = static_cast<uint32_t>(a);
					return true;
				}
				const auto mask =
				    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
				result = (static_cast<uint32_t>(a) & ~mask) |
				         ((static_cast<uint32_t>(b) << offset) & mask);
				return true;
			}
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectF32:
				if (ternary()) {
					result = a != 0u ? b : c;
					return true;
				}
				return false;
			case ValueOpcode::IEqual32:
				if (binary()) {
					result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::INotEqual32:
				if (binary()) {
					result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::ULessThan32:
				if (binary()) {
					result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::UGreaterThan32:
				if (binary()) {
					result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::UGreaterThanEqual32:
				if (binary()) {
					result = static_cast<uint32_t>(a) >= static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::LogicalAnd:
				if (binary()) {
					result = (a != 0u) && (b != 0u);
					return true;
				}
				return false;
			case ValueOpcode::LogicalOr:
				if (binary()) {
					result = (a != 0u) || (b != 0u);
					return true;
				}
				return false;
			case ValueOpcode::LogicalXor:
				if (binary()) {
					result = (a != 0u) != (b != 0u);
					return true;
				}
				return false;
			case ValueOpcode::LogicalNot:
				if (Arg(inst, 0, a)) {
					result = a == 0u;
					return true;
				}
				return false;
			case ValueOpcode::UndefU1:
			case ValueOpcode::UndefU8:
			case ValueOpcode::UndefU16:
			case ValueOpcode::UndefU32:
			case ValueOpcode::UndefU64: return false;
			default: break;
		}
		return false;
	}

	const ResourcePlan&                       m_program;
	const SrtRuntime&                         m_runtime;
	std::span<const uint8_t>                  m_clean_flat_slots;
	Evaluator*                                m_clean_evaluator = nullptr;
	Value                                     m_active_mask;
	EvaluationCache                           m_cache;
	std::vector<const Inst*>                  m_visiting;
	bool                                      m_reserved = false;
};

const DescriptorSource* Source(const ResourcePlan& program, uint32_t source) {
	if (source >= program.descriptor_sources.size()) {
		return nullptr;
	}
	return &program.descriptor_sources[source];
}

static void EvaluateSourceActivity(const ResourcePlan& program, const SrtRuntime& runtime,
                                    Evaluator& clean_evaluator, std::vector<uint8_t>& active) {
    active.assign(program.descriptor_sources.size(),1u);
    for(const auto& block:program.control_flow)
        for(const auto source:block.sources)active.at(source)=0u;
    if(program.control_flow.empty())return;
    std::vector<uint8_t> visited(program.control_flow.size());
    std::vector<uint32_t> pending{0};
    while(!pending.empty()) {
        const auto index=pending.back();pending.pop_back();
        if(visited.at(index))continue;
        visited[index]=1u;
        const auto& block=program.control_flow[index];
        for(const auto source:block.sources)active[source]=1u;
        uint32_t condition=0;
        if(!block.condition.IsEmpty() && runtime.read_specialization_memory!=nullptr &&
            clean_evaluator.Evaluate(block.condition,condition))
            pending.push_back(block.successors[condition!=0u?0u:1u]);
        else pending.insert(pending.end(),block.successors.begin(),block.successors.end());
    }
}

bool EvaluateRuntimeSourcesImpl(const ResourcePlan& program, std::span<const uint32_t> sources,
                                const SrtRuntime& runtime, std::vector<DescriptorValue>& results,
                                std::vector<uint32_t>& flat, bool evaluate_flat,
                                std::span<const uint8_t> clean_flat_slots,
                                std::vector<uint8_t>& active_sources) {
	if (!program.srt_plan_complete) {
		return false;
	}
	if (std::ranges::any_of(clean_flat_slots, [](uint8_t clean) { return clean != 0u; }) &&
	    runtime.read_specialization_memory == nullptr) {
		return false;
	}
	SrtRuntime clean_runtime  = runtime;
	clean_runtime.read_memory = runtime.read_specialization_memory;
	Evaluator            clean_evaluator(program, clean_runtime);
	Evaluator            evaluator(program, runtime, clean_flat_slots, &clean_evaluator);
	std::vector<uint8_t> active;
	if (evaluate_flat) EvaluateSourceActivity(program, runtime, clean_evaluator, active);
	std::vector<DescriptorValue> evaluated;
	evaluated.reserve(sources.size());
	for (const auto source_index: sources) {
		const auto* source = Source(program, source_index);
		if (source == nullptr) {
			return false;
		}
		DescriptorValue value;
		value.dword_count = source->dword_count;
		if (!evaluate_flat || active[source_index]) {
			for (uint32_t index = 0; index < source->dword_count; index++) {
				if (!evaluator.Evaluate(source->dwords[index], value.dwords[index])) {
					return false;
				}
			}
		}
		evaluated.push_back(value);
	}
	std::vector<uint32_t> flattened;
	if (evaluate_flat) {
		flattened.resize(program.srt_reads.size());
		for (const auto& read: program.srt_reads) {
			const bool clean    = read.flat_offset < clean_flat_slots.size() &&
			                      clean_flat_slots[read.flat_offset] != 0u;
			auto&      selected = clean ? clean_evaluator : evaluator;
			if (read.flat_offset >= flattened.size() ||
			    !selected.Evaluate(read.value, flattened[read.flat_offset])) {
				return false;
			}
		}
	}
	// Copied, not moved: the callers' vectors are reused storage that the linear evaluation swaps
	// with its own; a moved-in vector of exactly this size made those reallocate.
	results.assign(evaluated.begin(), evaluated.end());
	active_sources.assign(active.begin(), active.end());
	if (evaluate_flat) {
		flat.assign(flattened.begin(), flattened.end());
	}
	return true;
}

} // namespace

#if defined(__x86_64__) || defined(_M_X64)
static void BuildNativePredicate(const ResourcePlan& program, LinearSrtPlan& root,
                                  std::span<const uint32_t> conditional) {
    // A single reachable predicate is evaluated at most once. Its memo is not
    // shared with raw descriptor reads, so a separate compiled transaction is
    // equivalent, including a failed clean read's conservative two-way choice.
    if (conditional.size() != 1) return;
    const uint32_t condition = conditional.front();
    bool reached = false;
    for (uint32_t choice = 0; choice < 3; ++choice) {
        std::vector<uint8_t> active(program.descriptor_sources.size(), 1);
        std::vector<uint8_t> visited(program.control_flow.size());
        for (const auto& block : program.control_flow)
            for (const auto source : block.sources) active[source] = 0;
        std::vector<uint32_t> pending{0};
        while (!pending.empty()) {
            const auto index = pending.back(); pending.pop_back();
            if (visited[index]) continue;
            visited[index] = 1;
            const auto& block = program.control_flow[index];
            for (const auto source : block.sources) active[source] = 1;
            reached |= index == condition;
            if (index == condition && choice != 2) pending.push_back(block.successors[choice]);
            else pending.insert(pending.end(), block.successors.begin(), block.successors.end());
        }
        const auto leaf = std::ranges::find_if(root.control_variants,
            [&](const auto& candidate) { return candidate->active_sources == active; });
        EXIT_IF(leaf == root.control_variants.end());
        root.single_condition_variants[choice] = static_cast<uint32_t>(leaf - root.control_variants.begin());
    }
    if (!reached) return; // Never introduce a read from an unreachable block.
    auto compiled = std::make_shared<LinearSrtPlan>();
    if (!LinearSrtCompiler(program, *compiled).BuildPredicate(program.control_flow[condition].condition)) return;
    root.single_condition = std::move(compiled);
}

static std::optional<uint32_t> SelectNativePredicate(const LinearSrtPlan& root, const SrtRuntime& runtime) {
    if (kyty_local_srt_predicate_mode.load(std::memory_order_relaxed) == 0 || !root.single_condition) return {};
    const auto& predicate = *root.single_condition;
    EXIT_IF(predicate.nodes.size() > 128 || predicate.flat_words.size() != 1);
    std::array<uint64_t, 128> values; // Native code writes every dependency before use.
    const bool known = runtime.read_specialization_memory && predicate.function(&runtime, values.data());
    const auto choice = known ? (uint32_t(values[predicate.flat_words[0]]) != 0 ? 0u : 1u) : 2u;
    return root.single_condition_variants[choice];
}

static std::shared_ptr<LinearSrtPlan> BuildControlledLinearSrt(const ResourcePlan& program,
                                                                std::string& reason) {
    if(!program.srt_plan_complete || program.control_flow.empty() || program.control_flow.size()>256 ||
        program.descriptor_sources.empty()) {
        reason = "controlled: shape";
        return {};
    }
    if(std::ranges::any_of(program.clean_flat_slots,[](uint8_t value){return value!=0;})) {
        reason = "controlled: clean flat slots";
        return {};
    }
    // Predicate reads and raw transaction reads use separate memo domains in
    // the interpreter. Exclude clean flat slots until that shared memo can be
    // represented without duplicating a clean callback or changing its order.
    std::vector<uint32_t> conditional;
    for(uint32_t i=0;i<program.control_flow.size();++i) {
        const auto& block=program.control_flow[i];
        if(std::ranges::any_of(block.sources,[&](uint32_t id){return id>=program.descriptor_sources.size();}) ||
            std::ranges::any_of(block.successors,[&](uint32_t id){return id>=program.control_flow.size();})) {
            reason = "controlled: block ids";
            return {};
        }
        if(!block.condition.IsEmpty()) {
            if(block.successors.size()!=2) {
                reason = "controlled: successors";
                return {};
            }
            conditional.push_back(i);
        }
    }
    if(conditional.empty() || conditional.size()>2) {
        reason = fmt::format("controlled: {} conditionals", conditional.size());
        return {};
    }
    auto root=std::make_shared<LinearSrtPlan>();
    root->sources=program.materialization_sources;root->clean_slots=program.clean_flat_slots;
    uint32_t combinations=1;
    for(size_t i=0;i<conditional.size();++i)combinations*=3;
    for(uint32_t combination=0;combination<combinations;++combination) {
        // Three outcomes include failed/missing predicates: the existing
        // interpreter then visits both successors conservatively.
        std::vector<uint8_t> choices(program.control_flow.size(),2u);
        auto encoded=combination;
        for(const auto index:conditional){choices[index]=encoded%3;encoded/=3;}
        std::vector<uint8_t> active(program.descriptor_sources.size(),1u),visited(program.control_flow.size());
        for(const auto& block:program.control_flow)for(const auto source:block.sources)active[source]=0;
        std::vector<uint32_t> pending{0};
        while(!pending.empty()) {
            const auto index=pending.back();pending.pop_back();
            if(visited[index])continue;
            visited[index]=1;
            const auto& block=program.control_flow[index];
            for(const auto source:block.sources)active[source]=1;
            if(choices[index]<2)pending.push_back(block.successors[choices[index]]);
            else pending.insert(pending.end(),block.successors.begin(),block.successors.end());
        }
        if(std::ranges::any_of(root->control_variants,[&](const auto& variant){return variant->active_sources==active;}))continue;
        auto leaf=std::make_shared<LinearSrtPlan>();
        LinearSrtCompiler compiler(program,*leaf,active);
        if(!compiler.Build()) {
            reason = fmt::format("controlled leaf: {} {}", compiler.failure ? compiler.failure : "?",
                                 ValueOpcodeName(compiler.failed_op));
            return {};
        }
        root->control_variants.push_back(std::move(leaf));
    }
    BuildNativePredicate(program, *root, conditional);
    return root;
}
#endif

void BuildLinearSrtPlan(ResourcePlan& program) {
    program.linear_srt.reset();
#if defined(__x86_64__) || defined(_M_X64)
    auto compiled=std::make_shared<LinearSrtPlan>();
    LinearSrtCompiler compiler(program,*compiled);
    if (compiler.Build()) {
        program.linear_srt=std::move(compiled);
        return;
    }
    std::string controlled;
    program.linear_srt=BuildControlledLinearSrt(program, controlled);
    // KYTY_SRT_LINEAR_LOG: why a plan stays on the SRT interpreter (each evaluation walks the IR).
    static const bool log = std::getenv("KYTY_SRT_LINEAR_LOG") != nullptr;
    if (log && !program.linear_srt) {
        uint32_t dwords = 0, conditionals = 0;
        for (const auto& source: program.descriptor_sources) dwords += source.dword_count;
        for (const auto& block: program.control_flow) conditionals += block.condition.IsEmpty() ? 0u : 1u;
        std::printf("SRT interpreter plan: hash=0x%016llx stage=%u sources=%zu materialized=%zu dwords=%u "
                    "flat=%zu blocks=%zu conditionals=%u linear=%s %s controlled=%s\n",
                    static_cast<unsigned long long>(program.shader_hash), static_cast<unsigned>(program.stage),
                    program.descriptor_sources.size(), program.materialization_sources.size(), dwords,
                    program.srt_reads.size(), program.control_flow.size(), conditionals,
                    compiler.failure ? compiler.failure : "?",
                    std::string(ValueOpcodeName(compiler.failed_op)).c_str(), controlled.c_str());
    }
#endif
}

static bool LinearMaskMatches(std::span<const uint8_t> a,std::span<const uint8_t> b) {
    for(size_t i=0;i<std::max(a.size(),b.size());++i)
        if((i<a.size() && a[i]!=0)!=(i<b.size() && b[i]!=0))return false;
    return true;
}

// Whether the linear plan evaluates these sources under this clean mask. The plan copies its
// program's materialization sources and clean slots (BuildLinearSrtPlan), which never change
// afterwards: the program's own vectors, what every materialization passes, match without a scan.
static bool LinearInputsMatch(const ResourcePlan& program,const LinearSrtPlan& plan,
                              std::span<const uint32_t> sources,std::span<const uint8_t> clean) {
    if(sources.data()==program.materialization_sources.data() && sources.size()==program.materialization_sources.size() &&
        clean.data()==program.clean_flat_slots.data() && clean.size()==program.clean_flat_slots.size())
        return true;
    return std::ranges::equal(sources,plan.sources) && LinearMaskMatches(clean,plan.clean_slots);
}

static bool EvaluateLinearSrtOriginal(const LinearSrtPlan& plan,const SrtRuntime& runtime,
                             std::vector<DescriptorValue>& results,std::vector<uint32_t>& flat,
                             std::vector<uint8_t>& active_sources) {
    // A separate lease for each invocation also protects against reader reentry.
    thread_local std::vector<std::vector<uint64_t>> pool;
    std::vector<uint64_t> values;
    if(!pool.empty()){values=std::move(pool.back());pool.pop_back();}
    struct Recycle {
        std::vector<std::vector<uint64_t>>& pool;std::vector<uint64_t>& values;
        ~Recycle(){if(pool.size()<8)pool.push_back(std::move(values));}
    } recycle{pool,values};
    values.resize(plan.nodes.size());
    if(!plan.function(&runtime,values.data())) return false;
    NativePreparationScratch<LocalNativeResource::OutputWorkspace> output_storage;
    auto& evaluated = output_storage.Get().evaluated;
    auto& flattened = output_storage.Get().flattened;
    evaluated.assign(plan.descriptor_sizes.size(), {});
    flattened.assign(plan.flat_words.size(), 0u);
    size_t cursor=0;
    for(size_t i=0;i<evaluated.size();++i) {
        auto& output=evaluated[i];output.dword_count=plan.descriptor_sizes[i];
        for(uint32_t j=0;j<output.dword_count;++j) {
            const auto node=plan.descriptor_words[cursor++];
            output.dwords[j]=node==UINT32_MAX ? 0u : static_cast<uint32_t>(values[node]);
        }
    }
    for(size_t i=0;i<flattened.size();++i)
        if(plan.flat_words[i]!=UINT32_MAX)flattened[i]=static_cast<uint32_t>(values[plan.flat_words[i]]);
    results.swap(evaluated);flat.swap(flattened);
    if(plan.active_sources.empty())active_sources.assign(plan.active_count,1u);
    else active_sources=plan.active_sources;
    return true;
}

static bool EvaluateLinearSrt(const LinearSrtPlan& plan,const SrtRuntime& runtime,
                             std::vector<DescriptorValue>& results,std::vector<uint32_t>& flat,
                             std::vector<uint8_t>& active_sources) {
    return EvaluateLinearSrtOriginal(plan, runtime, results, flat, active_sources);
}

#if defined(__x86_64__) || defined(_M_X64)
// What a read node's address is based on (SrtReadTrace::Base): a user data pair, or two structural
// reads (`read_index`: their indices in the trace; empty: none).
static SrtReadTrace::Base PointerBase(const LinearSrtPlan& plan, const LinearSrtPlan::Node& node,
                                      std::span<const uint32_t> read_index) {
	using Kind = LinearSrtPlan::Kind;
	SrtReadTrace::Base base;
	if (node.op == ValueOpcode::ReadConstBuffer) return base;
	const auto& low  = plan.nodes[node.args[0]];
	const auto& high = plan.nodes[node.args[1]];
	if (low.kind == Kind::User && high.kind == Kind::User && high.immediate == low.immediate + 1 &&
	    low.immediate < SrtReadTrace::NoRoot) {
		base.root = static_cast<uint8_t>(low.immediate);
	} else if (low.kind == Kind::Read && high.kind == Kind::Read && !read_index.empty() &&
	           read_index[node.args[0]] != SrtReadTrace::NoBase && read_index[node.args[1]] != SrtReadTrace::NoBase) {
		base.low  = read_index[node.args[0]];
		base.high = read_index[node.args[1]];
	}
	return base;
}

static bool TraceLinearPlan(const ResourcePlan& program, const LinearSrtPlan& plan, const SrtRuntime& runtime,
                            SrtReadTrace& trace) {
	if (plan.function == nullptr) return false;
	// Per-thread scratch: native XPR retraces call this for every stored record.
	thread_local struct {
		std::vector<uint64_t>                                values;
		std::vector<uint8_t>                                 structural, data;
		std::vector<uint32_t>                                read_index;
		std::vector<std::pair<SrtReadTrace::Kind, uint32_t>> kinds;
	} scratch;
	auto& values = scratch.values;
	values.assign(plan.nodes.size(), 0);
	if (!plan.function(&runtime, values.data())) return false;
	using Kind = LinearSrtPlan::Kind;
	// A node is structural when another node or a descriptor dword consumes it.
	auto& structural = scratch.structural;
	structural.assign(plan.nodes.size(), 0);
	for (const auto& node: plan.nodes) {
		switch (node.kind) {
			case Kind::Read:
				if (node.op == ValueOpcode::ReadConstBuffer) {
					for (uint32_t i = 0; i < 5; ++i) structural[node.args[i]] |= SrtReadTrace::Address;
				} else {
					structural[node.args[0]] |= SrtReadTrace::Pointer;
					structural[node.args[1]] |= SrtReadTrace::Pointer;
					structural[node.args[2]] |= SrtReadTrace::Address;
				}
				break;
			case Kind::Unary: structural[node.args[0]] |= SrtReadTrace::Computed; break;
			case Kind::Binary:
				for (uint32_t i = 0; i < 2; ++i) structural[node.args[i]] |= SrtReadTrace::Computed;
				break;
			case Kind::Select:
				for (uint32_t i = 0; i < 3; ++i) structural[node.args[i]] |= SrtReadTrace::Computed;
				break;
			default: break;
		}
	}
	for (const auto node: plan.descriptor_words)
		if (node != UINT32_MAX) structural[node] |= SrtReadTrace::Descriptor;
	trace.flat_words = plan.flat_words.size();
	// Anything in the flattened SRT but a plain read is an immediate, user data or computed from
	// structural reads: its value stays, its inputs are compared.
	auto& data = scratch.data;
	data.assign(plan.nodes.size(), 0);
	for (const auto index: plan.flat_words)
		if (index != UINT32_MAX && plan.nodes[index].kind == Kind::Read && !structural[index] && !plan.nodes[index].clean)
			data[index] = 1;
	auto& read_index = scratch.read_index;
	read_index.assign(plan.nodes.size(), SrtReadTrace::NoBase);
	for (uint32_t index = 0; index < plan.nodes.size(); ++index) {
		const auto& node = plan.nodes[index];
		if (node.kind != Kind::Read || data[index]) continue;
		uint64_t address = 0;
		if (!LinearSrtPlan::ReadAddress(&node, values.data(), address)) return false;
		read_index[index] = static_cast<uint32_t>(trace.structural.size());
		trace.structural.push_back({address, static_cast<uint32_t>(values[index]), node.clean, structural[index],
		                            PointerBase(plan, node, read_index)});
	}
	for (uint32_t i = 0; i < plan.flat_words.size(); ++i) {
		const auto index = plan.flat_words[i];
		if (index == UINT32_MAX || !data[index]) continue;
		const auto& node    = plan.nodes[index];
		uint64_t    address = 0;
		if (!LinearSrtPlan::ReadAddress(&node, values.data(), address)) return false;
		trace.data.emplace_back(i, address);
		trace.data_bases.push_back(PointerBase(plan, node, read_index));
	}
	for (uint32_t i = 0; i < plan.flat_words.size(); ++i) {
		const auto node = plan.flat_words[i];
		if (node != UINT32_MAX && read_index[node] != UINT32_MAX) trace.flat_feeds.emplace_back(read_index[node], i);
	}
	// Descriptor sources in evaluation order are the buffers, the direct images
	// and the samplers of the snapshot (MaterializeSnapshot). Without that exact
	// correspondence no feed is reported.
	auto& kinds = scratch.kinds;
	kinds.clear();
	for (uint32_t i = 0; i < program.info.buffers.size(); ++i) kinds.emplace_back(SrtReadTrace::Kind::Buffer, i);
	for (uint32_t i = 0; i < program.info.images.size(); ++i) {
		const auto* source = Source(program, program.info.images[i].source);
		if (source != nullptr && source->indirect_image.has_value()) return true;
		kinds.emplace_back(SrtReadTrace::Kind::Image, i);
	}
	for (uint32_t i = 0; i < program.info.samplers.size(); ++i) kinds.emplace_back(SrtReadTrace::Kind::Sampler, i);
	if (kinds.size() != plan.descriptor_sizes.size()) return true;
	size_t word = 0;
	for (size_t source = 0; source < plan.descriptor_sizes.size(); ++source)
		for (uint32_t dword = 0; dword < plan.descriptor_sizes[source]; ++dword, ++word) {
			const auto node = plan.descriptor_words[word];
			if (node == UINT32_MAX || read_index[node] == UINT32_MAX) continue;
			trace.feeds.push_back({read_index[node], kinds[source].first, kinds[source].second, dword});
		}
	return true;
}

// Forwards to the caller's readers and logs every value read (control-flow
// predicates decide the plan, so their inputs are structural).
struct TraceReadLog {
	const SrtRuntime*                   inner = nullptr;
	std::vector<SrtReadTrace::Read>*    out   = nullptr;
	static bool Raw(void* self, uint64_t address, uint32_t* value) {
		auto& log = *static_cast<TraceReadLog*>(self);
		if (!log.inner->read_memory(log.inner->userdata, address, value)) return false;
		log.out->push_back({address, *value, false, SrtReadTrace::Predicate});
		return true;
	}
	static bool Clean(void* self, uint64_t address, uint32_t* value) {
		auto& log = *static_cast<TraceReadLog*>(self);
		if (!log.inner->read_specialization_memory(log.inner->userdata, address, value)) return false;
		log.out->push_back({address, *value, true, SrtReadTrace::Predicate});
		return true;
	}
	static bool Span(void* self, uint64_t address, uint32_t* values, uint32_t count, bool clean) {
		auto& log = *static_cast<TraceReadLog*>(self);
		// Logged spans stay SRT-group sized, as every logged word is kept.
		if (count > 16 || !log.inner->try_read_memory_span(log.inner->userdata, address, values, count, clean))
			return false;
		for (uint32_t i = 0; i < count; ++i)
			log.out->push_back({address + 4ull * i, values[i], clean, SrtReadTrace::Predicate});
		return true;
	}
	static bool Sync(void* self, uint64_t address, uint64_t size) {
		auto& log = *static_cast<TraceReadLog*>(self);
		return log.inner->sync_memory(log.inner->userdata, address, size);
	}
};
#endif

bool TraceLinearSrtReads(const ResourcePlan& program, const SrtRuntime& runtime, SrtReadTrace& trace) {
#if defined(__x86_64__) || defined(_M_X64)
	const auto* plan = program.linear_srt.get();
	trace.data.clear();
	trace.data_bases.clear();
	trace.structural.clear();
	trace.feeds.clear();
	trace.flat_feeds.clear();
	trace.unrooted = false;
	// Nothing to read: no descriptor, flattened word, uniform fill or table.
	if (plan == nullptr && program.srt_plan_complete && program.descriptor_sources.empty() &&
	    program.srt_reads.empty() && program.control_flow.empty() && program.uniform_fill.fill.words == 0 &&
	    !program.requires_specialization_memory) {
		trace.flat_words = 0;
		return true;
	}
	if (plan == nullptr || !program.srt_plan_complete ||
	    !LinearInputsMatch(program, *plan, program.materialization_sources, program.clean_flat_slots))
		return false;
	if (plan->control_variants.empty()) return TraceLinearPlan(program, *plan, runtime, trace);
	// The leaf EvaluateRuntimeSources selects: by the native predicate when it does (SelectNativePredicate),
	// whose reads are the plan's, with their roots.
	if (kyty_local_srt_predicate_mode.load(std::memory_order_relaxed) != 0 && plan->single_condition) {
		const auto&               predicate = *plan->single_condition;
		std::array<uint64_t, 128> values;
		if (predicate.nodes.size() > values.size() || predicate.flat_words.size() != 1 ||
		    runtime.read_specialization_memory == nullptr || !predicate.function(&runtime, values.data()))
			return false;
		const auto choice = uint32_t(values[predicate.flat_words[0]]) != 0 ? 0u : 1u;
		if (!TraceLinearPlan(program, *plan->control_variants[plan->single_condition_variants[choice]], runtime, trace))
			return false;
		for (uint32_t index = 0; index < predicate.nodes.size(); ++index) {
			const auto& node = predicate.nodes[index];
			if (node.kind != LinearSrtPlan::Kind::Read) continue;
			uint64_t address = 0;
			if (!LinearSrtPlan::ReadAddress(&node, values.data(), address)) return false;
			trace.structural.push_back({address, static_cast<uint32_t>(values[index]), node.clean,
			                            SrtReadTrace::Predicate, PointerBase(predicate, node, {})});
		}
		return true;
	}
	// Else from the interpreter's logged predicate reads.
	trace.unrooted = true;
	if (runtime.read_memory == nullptr || runtime.read_specialization_memory == nullptr ||
	    runtime.try_read_memory_span == nullptr || runtime.sync_memory == nullptr)
		return false;
	std::vector<SrtReadTrace::Read> predicate_reads;
	TraceReadLog log {&runtime, &predicate_reads};
	SrtRuntime logged = runtime;
	logged.userdata                   = &log;
	logged.read_memory                = TraceReadLog::Raw;
	logged.read_specialization_memory = TraceReadLog::Clean;
	logged.try_read_memory_span       = TraceReadLog::Span;
	logged.sync_memory                = TraceReadLog::Sync;
	auto clean        = logged;
	clean.read_memory = logged.read_specialization_memory;
	Evaluator            predicate_evaluator(program, clean);
	std::vector<uint8_t> active;
	EvaluateSourceActivity(program, logged, predicate_evaluator, active);
	const auto leaf = std::ranges::find_if(plan->control_variants,
	                                       [&](const auto& candidate) { return candidate->active_sources == active; });
	if (leaf == plan->control_variants.end() || !TraceLinearPlan(program, **leaf, runtime, trace)) return false;
	trace.structural.insert(trace.structural.end(), predicate_reads.begin(), predicate_reads.end());
	return true;
#else
	(void)program;
	(void)runtime;
	(void)trace;
	return false;
#endif
}

SrtUserDataUse UserDataUse(const ResourcePlan& program) {
#if defined(__x86_64__) || defined(_M_X64)
	// The uniform fill and indirect image tables are evaluated outside the plans.
	if (!program.srt_plan_complete || program.uniform_fill.fill.words != 0 || program.requires_specialization_memory)
		return {};
	const auto* root = program.linear_srt.get();
	if (root == nullptr)
		return program.descriptor_sources.empty() && program.srt_reads.empty() && program.control_flow.empty()
		           ? SrtUserDataUse {0, 0}
		           : SrtUserDataUse {};
	// The plans an evaluation runs: the linear plan, or the predicate and the variant it selects.
	std::vector<const LinearSrtPlan*> plans {root};
	if (!root->control_variants.empty()) {
		if (root->single_condition == nullptr) return {};
		plans = {root->single_condition.get()};
		for (const auto& variant: root->control_variants) plans.push_back(variant.get());
	}
	using Kind = LinearSrtPlan::Kind;
	// By user data index: read at all, the low or the high half of a pointer read's base, or anything else.
	uint32_t read = 0, low = 0, high = 0, other = 0;
	for (const auto* plan: plans) {
		const auto user = [&](uint32_t node) {
			const auto& n = plan->nodes[node];
			return n.kind == Kind::User && n.immediate < 31 ? static_cast<int>(n.immediate) : -1;
		};
		const auto any = [&](uint32_t node) {
			if (const auto k = user(node); k >= 0) other |= 1u << k;
		};
		for (const auto& node: plan->nodes) {
			if (node.kind == Kind::User) {
				if (node.immediate >= 32) return {};
				read |= 1u << node.immediate;
			}
			uint32_t used = 0;
			switch (node.kind) {
				case Kind::Read: used = node.op == ValueOpcode::ReadConstBuffer ? 5u : 3u; break;
				case Kind::Unary: used = 1; break;
				case Kind::Binary: used = 2; break;
				case Kind::Select: used = 3; break;
				default: break;
			}
			if (node.kind == Kind::Read && node.op != ValueOpcode::ReadConstBuffer) {
				const auto k = user(node.args[0]);
				if (k >= 0 && user(node.args[1]) == k + 1) {
					low |= 1u << k;
					high |= 1u << (k + 1);
					any(node.args[2]);
					continue;
				}
			}
			for (uint32_t i = 0; i < used; ++i) any(node.args[i]);
		}
		for (const auto node: plan->descriptor_words)
			if (node != UINT32_MAX) any(node);
		for (const auto node: plan->flat_words)
			if (node != UINT32_MAX) any(node);
	}
	// Each half in its role only.
	return {read, low & (high >> 1) & ~(other | high) & ~((other | low) >> 1)};
#else
	(void)program;
	return {};
#endif
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type,
                          std::string* reason) {
	RuntimeValidator validator(program, type);
	if (validator.Run(value)) {
		return true;
	}
	if (reason != nullptr) {
		*reason = validator.Reason();
	}
	return false;
}

void BuildSrtPlan(Program& program) {
	if (program.resource_tracking_complete) {
		EXIT("shader SRT planning failed: cannot rebuild SRT after resource tracking");
	}
	program.srt_plan_complete = false;
	PlanBuilder(program).Run();
	program.srt_plan_complete = true;
}

bool EvaluateUniformValues(const ResourcePlan& program, std::span<const Value> values,
                            const SrtRuntime& runtime, std::span<uint32_t> results) {
	if (values.size() != results.size()) {
		return false;
	}
	auto clean = runtime;
	clean.read_memory = runtime.read_specialization_memory != nullptr
	                        ? runtime.read_specialization_memory
	                        : +[](void*, uint64_t, uint32_t*) { return false; };
	Evaluator evaluator(program, clean);
	for (size_t i = 0; i < values.size(); ++i) {
		if (!evaluator.Evaluate(values[i], results[i])) {
			return false;
		}
	}
	return true;
}

bool EvaluateDescriptorSource(const ResourcePlan& program, uint32_t source,
                              const SrtRuntime& runtime, DescriptorValue& result) {
	std::vector<DescriptorValue> results;
	if (!EvaluateDescriptorSources(program, std::span {&source, 1}, runtime, results)) {
		return false;
	}
	result = results.front();
	return true;
}

bool EvaluateDescriptorSources(const ResourcePlan& program, std::span<const uint32_t> sources,
                               const SrtRuntime& runtime, std::vector<DescriptorValue>& results) {
	std::vector<uint32_t> ignored;
	std::vector<uint8_t>  active;
	return EvaluateRuntimeSourcesImpl(program, sources, runtime, results, ignored, false, {},
	                                  active);
}

bool EvaluateRuntimeSources(const ResourcePlan& program, std::span<const uint32_t> sources,
                            const SrtRuntime& runtime, std::vector<DescriptorValue>& results,
                            std::vector<uint32_t>& flat, std::span<const uint8_t> clean_flat_slots,
                            std::vector<uint8_t>& active_sources) {
    {
        const auto& linear=program.linear_srt;
        if(linear && LinearInputsMatch(program,*linear,sources,clean_flat_slots)) {
            if(!program.srt_plan_complete || (runtime.read_specialization_memory==nullptr &&
                std::ranges::any_of(clean_flat_slots,[](uint8_t v){return v!=0;})))return false;
            const auto run = [&] {
                if(!linear->control_variants.empty()) {
                    if (const auto variant = SelectNativePredicate(*linear, runtime)) {
                        return EvaluateLinearSrt(*linear->control_variants[*variant], runtime, results, flat, active_sources);
                    }
                    auto clean=runtime;clean.read_memory=runtime.read_specialization_memory;
                    Evaluator predicate_evaluator(program,clean);
                    std::vector<uint8_t> active;
                    EvaluateSourceActivity(program,runtime,predicate_evaluator,active);
                    const auto leaf=std::ranges::find_if(linear->control_variants,
                        [&](const auto& candidate){return candidate->active_sources==active;});
                    EXIT_IF(leaf==linear->control_variants.end());
                    return EvaluateLinearSrt(**leaf,runtime,results,flat,active_sources);
                }
                // A runtime failure is final: replaying the original evaluator could
                // duplicate a reader's side effects or mask a failed ownership check.
                return EvaluateLinearSrt(*linear,runtime,results,flat,active_sources);
            };
            return run();
        }
    }
	const auto& linear = program.linear_srt;
	LiveCensus::Scope census(LiveCensus::SrtInterpreter, program.shader_hash,
	                         !linear ? 1 : !std::ranges::equal(sources, linear->sources) ? 2 : 3);
	return EvaluateRuntimeSourcesImpl(program, sources, runtime, results, flat, true,
	                                  clean_flat_slots, active_sources);
}

bool WalkSrt(const ResourcePlan& program, const SrtRuntime& runtime, std::vector<uint32_t>& flat) {
	std::vector<DescriptorValue> ignored;
	std::vector<uint8_t>         active;
	return EvaluateRuntimeSources(program, {}, runtime, ignored, flat, {}, active);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
