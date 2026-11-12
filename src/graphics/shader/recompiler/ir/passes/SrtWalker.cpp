#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/profiler.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <cinttypes>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>

namespace Libs::Graphics::ShaderRecompiler::IR {

SrtRuntime CleanRuntime(SrtRuntime runtime) {
	if (runtime.read_specialization_memory == nullptr) {
		runtime.map_clean_page = nullptr;
	}
	runtime.read_memory = runtime.read_specialization_memory != nullptr
	                          ? runtime.read_specialization_memory
	                          : +[](void*, uint64_t, std::span<uint32_t>) { return false; };
	return runtime;
}

namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > UINT64_MAX - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer &&
	    op != ValueOpcode::LoadBufferU32) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto& memory = values.memory_info[index];
	if (op != ValueOpcode::LoadBufferU32) {
		return (op == ValueOpcode::LoadAddressU32 && memory.kind == ResourceKind::ScalarAddress) ||
		       (op == ValueOpcode::ReadConstBuffer && memory.kind == ResourceKind::ScalarBuffer);
	}
	if (inst.NumArgs() != 5u || memory.kind != ResourceKind::Buffer || memory.typed ||
	    memory.formatted || memory.coherent || memory.data_bits != 32u ||
	    memory.data_dwords != 1u || memory.idxen || memory.offen || memory.offset != 0u ||
	    inst.Arg(4).GetType() != Type::U1) {
		return false;
	}
	for (uint32_t arg = 1; arg <= 3; ++arg) {
		if (inst.Arg(arg).Resolve() != Value(0u)) return false;
	}
	return true;
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

bool IsUniformBufferRead(const ResourcePlan& program, const Inst& inst) {
	if (inst.GetOpcode() != ValueOpcode::LoadBufferU32 || inst.NumArgs() != 5u) return false;
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= program.memory_info.size()) return false;
	const auto& memory = program.memory_info[index];
	const auto* handle = inst.Arg(0).ResolveInstruction();
	return memory.kind == ResourceKind::Buffer && memory.data_bits == 32u &&
	       memory.data_dwords == 1u && !memory.typed && !memory.formatted && handle != nullptr &&
	       handle->GetOpcode() == ValueOpcode::GetBufferResource;
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::ConditionRef:
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
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::UMulHi:
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
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type)
	    : m_program(program), m_type(type) {}

	bool Run(Value value) { return Validate(value); }

private:
	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		if (require_uniform && !m_active_mask.IsEmpty() && value == m_active_mask) {
			return m_type != RuntimeValueType::Integer || Validate(value, false);
		}
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
				default: return false;
			}
		}
		if (require_uniform && !m_active_mask.IsEmpty() && value == m_active_mask) return true;
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		if (!m_visiting.insert(inst).second) {
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
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
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer ||
		    op == ValueOpcode::LoadBufferU32) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
			if (op == ValueOpcode::LoadBufferU32) {
				const auto guard = inst->Arg(4).Resolve();
				return finish(Validate(inst->Arg(0)) &&
				              ((!m_active_mask.IsEmpty() && guard == m_active_mask) || Validate(guard)));
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
};


} // namespace

namespace {

// Operand count of an instruction whose value is a pure function of its operands, evaluated in
// order (the first failing operand fails it), or -1. EvaluateInst and the replay trace share
// ApplyPureOp, so both compute exactly the same thing.
int PureArity(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::UndefU1:
		case ValueOpcode::UndefU8:
		case ValueOpcode::UndefU16:
		case ValueOpcode::UndefU32:
		case ValueOpcode::UndefU64: return 0;
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::FPTrunc32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::LogicalNot: return 1;
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMulHi:
		case ValueOpcode::UMin32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::IEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::LogicalXor: return 2;
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract: return 3;
		case ValueOpcode::BitFieldInsert: return 4;
		default: return -1;
	}
}

float AsFloat32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

bool ApplyPureOp(ValueOpcode op, const Inst& inst, const uint64_t* v, uint64_t& result) {
	const uint64_t a = v[0];
	const uint64_t b = v[1];
	const uint64_t c = v[2];
	switch (op) {
		case ValueOpcode::CompositeConstructU64:
			result =
			    static_cast<uint32_t>(a) | (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::IAdd32: result = static_cast<uint32_t>(a + b); return true;
		case ValueOpcode::IAdd64: result = a + b; return true;
		case ValueOpcode::ISub32: result = static_cast<uint32_t>(a - b); return true;
		case ValueOpcode::ISub64: result = a - b; return true;
		case ValueOpcode::IMul32: result = static_cast<uint32_t>(a * b); return true;
		case ValueOpcode::IMul64: result = a * b; return true;
		case ValueOpcode::UMulHi:
			result = (uint64_t {static_cast<uint32_t>(a)} * static_cast<uint32_t>(b)) >> 32u;
			return true;
		case ValueOpcode::UMin32:
			result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
			return true;
		case ValueOpcode::ConvertF32U32:
			result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
			return true;
		case ValueOpcode::ConvertU32F32: {
			const auto value = AsFloat32(a);
			if (!std::isfinite(value) || value < 0.0f || static_cast<double>(value) > UINT32_MAX) {
				return false;
			}
			result = static_cast<uint32_t>(value);
			return true;
		}
		case ValueOpcode::FPMul32: {
			// As MULSS (and the native walker): a NaN operand gives itself, quieted, the first
			// one when both are. Left to the compiler, a * b may be emitted as b * a.
			const auto bits_a = static_cast<uint32_t>(a);
			const auto bits_b = static_cast<uint32_t>(b);
			if ((bits_a & 0x7fffffffu) > 0x7f800000u) {
				result = bits_a | 0x00400000u;
			} else if ((bits_b & 0x7fffffffu) > 0x7f800000u) {
				result = bits_b | 0x00400000u;
			} else {
				result = std::bit_cast<uint32_t>(AsFloat32(a) * AsFloat32(b));
			}
			return true;
		}
		case ValueOpcode::FPTrunc32:
			result = std::bit_cast<uint32_t>(std::trunc(AsFloat32(a)));
			return true;
		case ValueOpcode::FPRecipIFlag32: {
			const auto bits     = static_cast<uint32_t>(a);
			const auto exponent = (bits >> 23u) & 255u;
			// Exact normal power-of-two reciprocals avoid host/device approximation drift.
			if ((bits & 0x807fffffu) != 0u || exponent == 0u || exponent >= 254u) return false;
			result = (254u - exponent) << 23u;
			return true;
		}
		case ValueOpcode::FPIsNan32: result = std::isnan(AsFloat32(a)); return true;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32: {
			const auto operand = [&](uint64_t bits) {
				if (inst.Flags<FPCompareFlags>().flush_input_denorms &&
				    (bits & 0x7fffffffu) < 0x00800000u) {
					bits &= 0x80000000u;
				}
				return AsFloat32(bits);
			};
			result = op == ValueOpcode::FPOrdLessThanEqual32 ? operand(a) <= operand(b)
			                                                 : operand(a) >= operand(b);
			return true;
		}
		case ValueOpcode::BitwiseAnd32: result = static_cast<uint32_t>(a & b); return true;
		case ValueOpcode::BitwiseAnd64: result = a & b; return true;
		case ValueOpcode::BitwiseOr32: result = static_cast<uint32_t>(a | b); return true;
		case ValueOpcode::BitwiseXor32: result = static_cast<uint32_t>(a ^ b); return true;
		case ValueOpcode::BitwiseNot32: result = ~static_cast<uint32_t>(a); return true;
		case ValueOpcode::ShiftLeftLogical32:
			result = static_cast<uint32_t>(a) << (b & 31u);
			return true;
		case ValueOpcode::ShiftLeftLogical64: result = a << (b & 63u); return true;
		case ValueOpcode::ShiftRightLogical32:
			result = static_cast<uint32_t>(a) >> (b & 31u);
			return true;
		case ValueOpcode::ShiftRightLogical64: result = a >> (b & 63u); return true;
		case ValueOpcode::ShiftRightArithmetic32:
			result = static_cast<uint32_t>(std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >>
			                               (b & 31u));
			return true;
		case ValueOpcode::ShiftRightArithmetic64:
			result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
			return true;
		case ValueOpcode::BitFieldUExtract: {
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			const auto mask = width == 32u  ? UINT32_MAX
			                  : width == 0u ? 0u
			                                : (uint32_t {1} << width) - 1u;
			result          = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
			return true;
		}
		case ValueOpcode::BitFieldSExtract: {
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
		case ValueOpcode::BitFieldInsert: {
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(v[3]);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result =
			    (static_cast<uint32_t>(a) & ~mask) | ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::IEqual32:
			result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::INotEqual32:
			result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::ULessThan32:
			result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::UGreaterThan32:
			result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::ULessThanEqual32:
			result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::SGreaterThanEqual32:
			result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
			         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
			return true;
		case ValueOpcode::LogicalXor: result = (a != 0u) != (b != 0u); return true;
		case ValueOpcode::LogicalNot: result = a == 0u; return true;
		default: return false; // Undef*
	}
}

// LoadAddressU32 / ReadConstBuffer after its operands (EvaluateRawRead's address computation):
// Read with the address to read, Zero for a constant buffer read past the end (reads zero), or
// Fail.
enum class RawAddress : uint8_t { Fail, Read, Zero };
RawAddress ComputeRawAddress(bool const_buffer, int64_t immediate, uint64_t low, uint64_t high,
                             uint64_t offset, uint64_t records, uint64_t& base, uint64_t& address) {
	// A raw scalar address keeps its upper bits; a constant buffer's base is 48 bits (upstream
	// cd9fc6ff, as EvaluateRawRead does).
	base = (high << 32u) | static_cast<uint32_t>(low);
	if (const_buffer) {
		if (immediate < 0) {
			return RawAddress::Fail;
		}
		const auto byte_offset = (static_cast<uint64_t>(immediate) & ~uint64_t {3}) +
		                         (static_cast<uint32_t>(offset) & ~3u);
		const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		const auto size   = stride == 0u
		                        ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                        : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
			// A scalar buffer read past the end returns zero (PS5 ISA, scalar buffer addressing),
			// as the shader's own load does (EmitReadConstBuffer). Failing skipped the draw.
			return RawAddress::Zero;
		}
		address = (base & AddressMask & ~uint64_t {3}) + byte_offset;
		return RawAddress::Read;
	}
	const auto relative =
	    (immediate & ~int64_t {3}) + static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
	return AddSignedAddress(base & ~uint64_t {3}, relative, address) ? RawAddress::Read
	                                                                  : RawAddress::Fail;
}

} // namespace

SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     std::span<const uint8_t> clean_flat_slots, SrtWalker* clean_evaluator,
                     Value active_mask)
    : m_program(program), m_runtime(runtime), m_clean_flat_slots(clean_flat_slots),
      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask.Resolve()),
      m_context(AcquireContext(program)) {
	BindNative();
}

namespace {

// KYTY_SRT_NATIVE=0 keeps every walker on the interpreter.
bool NativeEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_NATIVE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_SRT_NATIVE_VERIFY=1 evaluates every native result again with the interpreter and stops the
// emulator on a difference.
bool NativeVerify() {
	static const bool verify = std::getenv("KYTY_SRT_NATIVE_VERIFY") != nullptr;
	return verify;
}

// A plan is compiled once this many walkers were created for it (two per resource refresh), so
// plans used once or twice are not compiled.
constexpr uint32_t NativeAfterWalkers = 8;

// Set while VerifyNative's reference walkers run.
thread_local bool g_native_suppressed = false;

SrtNativeStats g_native_stats;
bool           g_native_report = false;
// KYTY_SPECULATE (gt7-spec-1112): ProgramCache::Get asks for the report on every program get
// (~130,000 a second in a GT7 race); a relaxed flag spares it the mutex while there is none.
std::atomic<bool> g_native_report_pending {false};
// The stage-preparation helper (pipelineCache.cpp) walks plans of its own concurrently.
std::mutex     g_native_stats_mutex;

} // namespace

bool TakeSrtNativeReport(SrtNativeStats& stats) {
	if (!g_native_report_pending.load(std::memory_order_relaxed)) {
		return false;
	}
	std::lock_guard lock(g_native_stats_mutex);
	g_native_report_pending.store(false, std::memory_order_relaxed);
	if (!g_native_report) {
		return false;
	}
	g_native_report = false;
	stats           = g_native_stats;
	return true;
}

void SrtWalker::BindNative() {
	if (!NativeEnabled() || g_native_suppressed || !m_active_mask.IsEmpty() ||
	    !SrtNativeCode::Supported()) {
		return;
	}
	const auto& program = m_program;
	if (program.native_code == nullptr) {
		if (program.native_attempted || ++program.native_uses < NativeAfterWalkers) {
			return;
		}
		program.native_attempted = true;
		program.native_code      = SrtNativeCode::Compile(program);
		std::lock_guard stats_lock(g_native_stats_mutex);
		auto&           stats = g_native_stats;
		if (program.native_code == nullptr) {
			stats.failed++;
		} else {
			stats.plans++;
			stats.bytes += program.native_code->CodeSize();
			stats.instructions += program.native_code->Instructions();
			stats.interpreted += program.native_code->Interpreted();
		}
		const auto total = stats.plans + stats.failed;
		if ((total & (total - 1u)) == 0u) {
			g_native_report = true;
			g_native_report_pending.store(true, std::memory_order_relaxed);
		}
		if (program.native_code == nullptr) {
			return;
		}
	}
	auto mode = SrtNativeMode::Self;
	if (m_clean_evaluator != nullptr) {
		// Split code evaluates the plan's clean slots and every select predicate in the clean
		// walker, which must itself run Self code for this plan.
		if (m_clean_evaluator->m_native != program.native_code.get() ||
		    m_clean_evaluator->m_native_mode != SrtNativeMode::Self ||
		    &m_clean_evaluator->m_program != &program ||
		    m_clean_flat_slots.size() != program.clean_flat_slots.size() ||
		    !std::equal(m_clean_flat_slots.begin(), m_clean_flat_slots.end(),
		                program.clean_flat_slots.begin())) {
			return;
		}
		mode = SrtNativeMode::Split;
	}
	if (m_context.values.size() < program.evaluation_value_count) {
		m_context.values.resize(program.evaluation_value_count);
	}
	m_native_frame.memo           = m_context.values.data();
	m_native_frame.generation     = m_context.generation;
	m_native_frame.user_data      = m_runtime.user_data.data();
	m_native_frame.user_data_size = m_runtime.user_data.size();
	m_native_frame.shader_base    = m_runtime.shader_base;
	m_native_frame.walker         = this;
	m_native_frame.clean =
	    mode == SrtNativeMode::Split ? &m_clean_evaluator->m_native_frame : nullptr;
	m_native_frame.failed = &m_failed_value;
	m_native              = program.native_code.get();
	m_native_mode         = mode;
}

bool SrtWalker::UseNativeTables() const {
	return m_native != nullptr;
}

bool SrtWalker::EvaluateNative(const SrtNativeValue& value, uint64_t& result) {
	switch (value.kind) {
		case SrtNativeValue::Immediate: result = value.immediate; return true;
		case SrtNativeValue::Routine: {
			// Upstream's walk evaluates branch conditions with the clean reader too: a value with
			// no routine compiled for this reader's mode goes through the interpreter.
			if (!m_native->Has(m_native_mode, value.index)) {
				return EvaluateWide(Value(const_cast<Inst*>(value.inst)), result);
			}
			const bool ok = m_native->Evaluate(m_native_frame, m_native_mode, value.index, result);
			return NativeVerify() ? VerifyNative(Value(const_cast<Inst*>(value.inst)), ok, result) : ok;
		}
		case SrtNativeValue::Interpret:
			return EvaluateWide(Value(const_cast<Inst*>(value.inst)), result);
		case SrtNativeValue::Fail: return false;
	}
	return false;
}

bool SrtWalker::VerifyNative(Value value, bool native_ok, uint64_t native_result) {
	const bool suppressed = g_native_suppressed;
	g_native_suppressed   = true;
	uint64_t reference    = 0;
	bool     reference_ok = false;
	if (m_native_mode == SrtNativeMode::Self) {
		SrtWalker walker(m_program, m_runtime);
		reference_ok = walker.EvaluateWide(value, reference);
	} else {
		SrtWalker clean(m_program, m_clean_evaluator->m_runtime);
		SrtWalker walker(m_program, m_runtime, m_clean_flat_slots, &clean);
		reference_ok = walker.EvaluateWide(value, reference);
	}
	g_native_suppressed = suppressed;
	if (reference_ok != native_ok || (native_ok && reference != native_result)) {
		EXIT("SRT native mismatch: shader %016" PRIx64 " opcode %u mode %u: native %s 0x%" PRIx64
		     ", interpreter %s 0x%" PRIx64 "\n",
		     m_program.shader_hash, static_cast<uint32_t>(value.Resolve().TryInstruction()->GetOpcode()),
		     static_cast<uint32_t>(m_native_mode), native_ok ? "ok" : "failed", native_result,
		     reference_ok ? "ok" : "failed", reference);
	}
	return native_ok;
}

SrtWalker::~SrtWalker() {
	--m_program.evaluation_depth;
}

bool SrtWalker::Evaluate(Value value, uint32_t& result) {
	uint64_t wide = 0;
	if (!EvaluateWide(value, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

ResourcePlan::EvaluationContext& SrtWalker::AcquireContext(const ResourcePlan& program) {
	if (program.evaluation_depth == program.evaluation_contexts.size()) {
		program.evaluation_contexts.emplace_back();
	}
	auto& context = program.evaluation_contexts[program.evaluation_depth++];
	context.generation += 2;
	return context;
}

float SrtWalker::Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

bool SrtWalker::EvaluateWideImpl(Value value, uint64_t& result) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		bool ok = true;
		switch (value.GetType()) {
			case Type::U1: result = value.U1(); break;
			case Type::U8: result = value.U8(); break;
			case Type::U16: result = value.U16(); break;
			case Type::U32: result = value.U32(); break;
			case Type::U64: result = value.U64(); break;
			case Type::F32: result = std::bit_cast<uint32_t>(value.F32Value()); break;
			default: ok = false; break;
		}
		if (m_trace != nullptr) {
			m_trace->OnImmediate(ok, result);
		}
		return ok;
	}
	if (!m_active_mask.IsEmpty() && value == m_active_mask) {
		result = 1u;
		return true;
	}
	auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		if (m_trace != nullptr) {
			m_trace->Abort("no instruction");
		}
		return false;
	}
	if (!m_active_mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) && inst->NumArgs() == 3 &&
	    inst->Arg(0).Resolve() == m_active_mask) {
		return EvaluateWide(inst->Arg(1), result);
	}
	const auto index = inst->EvaluationIndex(m_program.evaluation_value_count);
	if (index >= m_context.values.size()) {
		m_context.values.resize(m_program.evaluation_value_count);
		m_native_frame.memo = m_context.values.data();
	}
	if (m_native != nullptr && m_native->Has(m_native_mode, index)) {
		const bool ok = m_native->Evaluate(m_native_frame, m_native_mode, index, result);
		return NativeVerify() ? VerifyNative(value, ok, result) : ok;
	}
	if (m_context.values[index].generation == m_context.generation) {
		result = m_context.values[index].value;
		if (m_trace != nullptr) {
			m_trace->OnMemoHit(*this, index, result);
		}
		return true;
	}
	// The low generation bit marks an instruction that is still being evaluated.
	if (m_context.values[index].generation == (m_context.generation | 1u)) {
		if (m_trace != nullptr) {
			m_trace->Abort("cycle");
		}
		return false;
	}
	m_context.values[index].generation = m_context.generation | 1u;
	uint64_t   out                     = 0;
	const bool evaluated               = EvaluateInst(*inst, out);
	// Recursive evaluation may grow the dense memo vector.
	m_native_frame.memo = m_context.values.data();
	auto& memo          = m_context.values[index];
	if (m_trace != nullptr) {
		m_trace->OnInst(*this, *inst, index, evaluated, out);
	}
	if (!evaluated) {
		if (m_failed_value == nullptr) m_failed_value = inst;
		memo.generation = 0;
		return false;
	}
	memo.value      = out;
	memo.generation = m_context.generation;
	result          = out;
	return true;
}

bool SrtWalker::Arg(const Inst& inst, size_t index, uint64_t& result) {
	return EvaluateWide(inst.Arg(index), result);
}

bool SrtWalker::EvaluatePhi(const Inst& inst, uint64_t& result) {
	const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
	return !value.IsEmpty() && EvaluateWide(value, result);
}

bool SrtWalker::EvaluateExtract(const Inst& inst, uint64_t& result) {
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
		result = component == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
		return true;
	}
	return false;
}

bool SrtWalker::EvaluateRawRead(const Inst& inst, uint64_t& result) {
	const auto flags = inst.Flags<MemoryFlags>();
	if (flags.index >= m_program.memory_info.size()) {
		return false;
	}
	const auto& mem    = m_program.memory_info[flags.index];
	const bool  vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
	const auto  guard  = vector ? inst.Arg(4).Resolve() : Value {};
	if (vector && (guard.IsImmediate() || guard != m_active_mask)) {
		uint64_t enabled = 0;
		if (!Arg(inst, 4, enabled)) return false;
		if (enabled == 0u) {
			result = 0u;
			return true;
		}
	}
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
	const bool const_buffer = inst.GetOpcode() == ValueOpcode::ReadConstBuffer;
	const auto immediate    = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
	uint64_t   records      = 0;
	uint64_t   word3        = 0;
	if (const_buffer || vector) {
		if (handle->NumArgs() != 4u || !Arg(*handle, 2, records) || !Arg(*handle, 3, word3)) {
			return false;
		}
	}
	if (vector) {
		// Only the uniform, unswizzled structured DWORD address is evaluated on the host, through
		// the strict (specialization) reader. Not part of the replay traces.
		if (immediate < 0) {
			return false;
		}
		const auto base = (high << 32u) | static_cast<uint32_t>(low);
		const auto byte_offset =
		    (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
		const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u)
			return false;
		if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) {
			result = 0u;
			return true;
		}
		const auto size = static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
			result = 0;
			return true;
		}
		const auto address = (base & AddressMask & ~uint64_t {3}) + byte_offset;
		uint32_t   word    = 0;
		if (m_runtime.read_specialization_memory == nullptr) {
			return false;
		}
		if (!m_runtime.read_specialization_memory(m_runtime.userdata, address, {&word, 1})) {
			if (base == 0 && m_program.control_flow.empty()) {
				result = 0;
				return true;
			}
			m_read_failure         = "guest memory unreadable";
			m_read_failure_address = address;
			m_read_failure_offset  = 0;
			m_read_failure_size    = 0;
			return false;
		}
		result = word;
		return true;
	}
	uint64_t base    = 0;
	uint64_t address = 0;
	switch (ComputeRawAddress(const_buffer, immediate, low, high, offset, records, base, address)) {
		case RawAddress::Fail: return false;
		case RawAddress::Zero: result = 0; return true;
		case RawAddress::Read: break;
	}
	return ReadRawWord(address, base, result);
}

bool SrtWalker::ReadRawWord(uint64_t address, uint64_t base, uint64_t& result) {
	if (m_fill_active && m_runtime.gpu_written(address, sizeof(uint32_t))) {
		// Only the shader uses this slot: the GPU copies the bytes into the flat buffer before
		// the command (SrtGpuFill), where reading them here would wait for the GPU.
		m_runtime.gpu_fills->push_back({m_fill_offset, address});
		result = 0;
		return true;
	}
	uint32_t word = 0;
	if (m_runtime.map_clean_page != nullptr) {
		// The reader's own first step (a GPU-clean page is read from its backing), without the
		// callback chain and its page search for every dword. Raw reads are dword aligned.
		const auto page = address >> m_runtime.page_shift;
		if (page != m_mapped_page) {
			m_mapped_page  = page;
			m_mapped_bytes = m_runtime.map_clean_page(m_runtime.page_userdata,
			                                          page << m_runtime.page_shift);
		}
		if (m_mapped_bytes != nullptr && (address & 3u) == 0u) {
			std::memcpy(&word,
			            m_mapped_bytes + (address & ((uint64_t {1} << m_runtime.page_shift) - 1u)),
			            sizeof(word));
			if (m_runtime.capture_ranges != nullptr) {
				m_runtime.capture_ranges->emplace_back(address, sizeof(word));
			}
			result = word;
			return true;
		}
	}
	if (m_runtime.read_memory != nullptr) {
		if (!m_runtime.read_memory(m_runtime.userdata, address, {&word, 1})) {
			// A plan without control flow cannot tell a guarded read from an active one: a
			// failed read through a null base there reads zero (CS 0x0b4b91abfed42248 tests
			// s[4:5] against zero before its s_load; failing skipped its dispatch every frame).
			// With control flow, the reachability walk skips guarded reads and a null read in a
			// reachable block fails (ResourceTrackingTests, guarded scalar descriptor reads).
			if (base == 0 && m_program.control_flow.empty()) {
				result = 0;
				return true;
			}
			m_read_failure         = "guest memory unreadable";
			m_read_failure_address = address;
			m_read_failure_offset  = 0;
			m_read_failure_size    = 0;
			return false;
		}
	} else {
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	result = word;
	return true;
}

bool SrtWalker::EvaluateBufferRead(const Inst& inst, uint64_t& result) {
	if (!IsUniformBufferRead(m_program, inst)) return false;
	uint64_t enabled = 0;
	if (!Arg(inst, 4, enabled)) return false;
	if (enabled == 0u) {
		result = 0u;
		return true;
	}
	ShaderBufferResource descriptor;
	const auto&          handle = *inst.Arg(0).ResolveInstruction();
	for (uint32_t i = 0; i < 4u; i++) {
		if (!Evaluate(handle.Arg(i), descriptor.fields[i])) return false;
	}
	// Lane-indexed and swizzled addressing are not uniform raw linear reads.
	if (descriptor.Type() != 0u || descriptor.AddTid() || descriptor.SwizzleEnabled() ||
	    descriptor.OutOfBounds() != 0u || descriptor.Stride() < 4u)
		return false;
	uint64_t index = 0, offset = 0, scalar = 0;
	if (!Arg(inst, 1, index) || !Arg(inst, 2, offset) || !Arg(inst, 3, scalar)) return false;
	const auto& memory      = m_program.memory_info[inst.Flags<MemoryFlags>().index];
	const auto  byte_offset = static_cast<uint32_t>(offset + memory.offset);
	if (index >= descriptor.NumRecords() || byte_offset >= descriptor.Stride() ||
	    descriptor.RawFormat() == 0u) {
		result = 0u;
		return true;
	}
	const auto relative = static_cast<uint32_t>(index * descriptor.Stride() + byte_offset + scalar);
	const auto base     = descriptor.Base48();
	const auto aligned  = relative & ~uint64_t {3};
	if ((base & 3u) != 0u || aligned > descriptor.GetSize() ||
	    descriptor.GetSize() - aligned < 4u || aligned > AddressMask - base)
		return false;
	uint32_t word = 0;
	if (m_runtime.read_memory == nullptr ||
	    !m_runtime.read_memory(m_runtime.userdata, base + aligned, {&word, 1}))
		return false;
	result = word;
	return true;
}

bool SrtWalker::EvaluateInst(const Inst& inst, uint64_t& result) {
	const auto op = inst.GetOpcode();
	if (const int arity = PureArity(op); arity >= 0) {
		uint64_t operands[4] {};
		for (int i = 0; i < arity; i++) {
			if (!Arg(inst, static_cast<size_t>(i), operands[i])) {
				return false;
			}
		}
		return ApplyPureOp(op, inst, operands, result);
	}
	uint64_t a = 0;
	uint64_t b = 0;
	switch (op) {
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
			if (!m_active_mask.IsEmpty() && inst.Arg(1).Resolve() == m_active_mask) {
				return EvaluateWide(inst.Arg(0), result);
			}
			const auto clean_runtime = CleanRuntime(m_runtime);
			SrtWalker  clean_active(m_program, clean_runtime, {}, nullptr, inst.Arg(1));
			SrtWalker  active(m_program, m_runtime, m_clean_flat_slots, &clean_active, inst.Arg(1));
			return active.EvaluateWide(inst.Arg(0), result);
		}
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32: return Arg(inst, 0, result);
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2: return EvaluateExtract(inst, result);
		case ValueOpcode::ReadConst: {
			const auto slot = inst.Arg(1).Resolve();
			if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return false;
			}
			if (slot.U32() < m_clean_flat_slots.size() && m_clean_flat_slots[slot.U32()] != 0u &&
			    m_clean_evaluator != nullptr) {
				return m_clean_evaluator->EvaluateWide(m_program.srt_reads[slot.U32()].value,
				                                       result);
			}
			return EvaluateWide(m_program.srt_reads[slot.U32()].value, result);
		}
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32:
			if (IsRawRead(m_program, inst)) {
				if (inst.GetOpcode() == ValueOpcode::LoadBufferU32 && m_clean_evaluator != nullptr &&
				    m_clean_evaluator->m_active_mask == m_active_mask) {
					return m_clean_evaluator->EvaluateWide(Value(const_cast<Inst*>(&inst)), result);
				}
				return EvaluateRawRead(inst, result);
			}
			break;
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32: {
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (predicate.EvaluateWide(inst.Arg(0), a)) {
				return Arg(inst, a != 0u ? 1u : 2u, result);
			}
			return false;
		}
		case ValueOpcode::LogicalAnd: {
			const bool left = Arg(inst, 0, a);
			if (left && a == 0u) {
				result = 0u;
				return true;
			}
			if (!Arg(inst, 1, b) || (b != 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case ValueOpcode::LogicalOr: {
			const bool left = Arg(inst, 0, a);
			if (left && a != 0u) {
				result = 1u;
				return true;
			}
			if (!Arg(inst, 1, b) || (b == 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case ValueOpcode::ConditionRef: return Arg(inst, 0, result);
		default: break;
	}
	return false;
}

bool SrtWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_program.descriptor_sources.size()) {
		return false;
	}
	const auto& descriptor = m_program.descriptor_sources[source];
	result                 = {};
	result.dword_count     = descriptor.dword_count;
	if (UseNativeTables() && descriptor.dword_count <= 8u) {
		const auto* dwords = m_native->DescriptorDwords(source);
		for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
			uint64_t wide = 0;
			if (!EvaluateNative(dwords[index], wide)) {
				return false;
			}
			result.dwords[index] = static_cast<uint32_t>(wide);
		}
		return true;
	}
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		if (!Evaluate(descriptor.dwords[index], result.dwords[index])) {
			return false;
		}
	}
	return true;
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	if (!m_program.srt_plan_complete) return false;
	const auto refresh = [&](uint32_t slot) {
		if (slot >= m_program.srt_reads.size()) return false;
		const auto& read = m_program.srt_reads[slot];
		const bool clean = read.flat_offset < m_clean_flat_slots.size() &&
		                   m_clean_flat_slots[read.flat_offset] != 0u;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
			return false;
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		if (read.flat_offset >= flat.size()) return false;
		if (!clean && slot < m_program.gpu_fill_slots.size() && m_program.gpu_fill_slots[slot] != 0u &&
		    m_runtime.gpu_written != nullptr && m_runtime.gpu_fills != nullptr) {
			// Through the interpreter or a replay (both read through ReadRawWord), not the native
			// tables: ReadRawWord leaves GPU-written bytes to the GPU.
			m_fill_active = true;
			m_fill_offset = read.flat_offset;
			const bool ok = Evaluate(read.value, flat[read.flat_offset]);
			m_fill_active = false;
			return ok;
		}
		if (evaluator.UseNativeTables()) {
			uint64_t wide = 0;
			if (!evaluator.EvaluateNative(evaluator.m_native->FlatReads()[slot], wide)) return false;
			flat[read.flat_offset] = static_cast<uint32_t>(wide);
			return true;
		}
		return evaluator.Evaluate(read.value, flat[read.flat_offset]);
	};
	auto& active = m_program.active_sources;
	if (m_program.control_flow.empty()) {
		active.clear();
		flat.resize(m_program.srt_reads.size());
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); ++slot) {
			if (!refresh(slot)) return false;
		}
		return true;
	}
	// 0 false, 1 true, 2 not known (no condition, no reader, or it could not be evaluated).
	auto&      predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
	const auto outcome   = [&](uint32_t index, const ResourceBlock& block) -> uint8_t {
		if (block.condition.IsEmpty() || m_runtime.read_specialization_memory == nullptr) {
			return 2u;
		}
		uint32_t condition = 0;
		bool     known     = false;
		if (predicate.UseNativeTables()) {
			uint64_t wide = 0;
			known     = predicate.EvaluateNative(predicate.m_native->Conditions()[index], wide);
			condition = static_cast<uint32_t>(wide);
		} else {
			known = predicate.Evaluate(block.condition, condition);
		}
		return known ? (condition != 0u ? 1u : 0u) : 2u;
	};
	// Replay: in the jungle a program's walk visited ~88 of ~97 blocks for only ~4.7 conditions,
	// and 84 % of walks gave the same result as the program's previous one. Conditions that give
	// the outcomes of the last walk visit the same blocks, so only its reads are evaluated again
	// and its active sources taken over: no visited/pending bookkeeping per block.
	const uint8_t key = (m_runtime.read_specialization_memory != nullptr ? 1u : 0u) |
	                    (m_clean_evaluator != nullptr ? 2u : 0u);
	if (m_program.active_walk_valid && m_program.walk_key == key) {
		bool same = true;
		for (const auto& [index, recorded]: m_program.active_walk) {
			if (outcome(index, m_program.control_flow[index]) != recorded) {
				same = false;
				break;
			}
		}
		if (same) {
			flat.assign(m_program.srt_reads.size(), 0u);
			for (const auto slot: m_program.walk_slots) {
				if (!refresh(slot)) return false;
			}
			active = m_program.active_walk_result;
			// KYTY_WALK_VERIFY=1: walk anyway and stop if the replay differs.
			static const bool verify = std::getenv("KYTY_WALK_VERIFY") != nullptr;
			if (!verify) {
				return true;
			}
			const auto replay_flat   = flat;
			const auto replay_active = active;
			static std::atomic<uint64_t> verified {0};
			if (!WalkBlocks(flat, refresh, outcome, key) || flat != replay_flat ||
			    active != replay_active) {
				EXIT("SRT walk replay differs from the walk\n");
			}
			if (verified.fetch_add(1, std::memory_order_relaxed) % 100000 == 0) {
				std::fprintf(stderr, "SRT walk replay verified: %llu\n",
				             static_cast<unsigned long long>(verified.load()));
			}
			return true;
		}
	}
	return WalkBlocks(flat, refresh, outcome, key);
}

template <typename Refresh, typename Outcome>
bool SrtWalker::WalkBlocks(std::vector<uint32_t>& flat, const Refresh& refresh,
                           const Outcome& outcome, uint8_t key) {
	auto& active = m_program.active_sources;
	flat.assign(m_program.srt_reads.size(), 0u);
	// The sources no block guards are active on every walk: built once per plan, copied per call
	// (rebuilding it walked every block's sources on every draw).
	auto& initial = m_program.active_initial;
	if (initial.size() != m_program.descriptor_sources.size()) {
		initial.assign(m_program.descriptor_sources.size(), 1u);
		for (const auto& block: m_program.control_flow) {
			for (const auto source: block.sources) initial.at(source) = 0u;
		}
	}
	active = initial;
	auto& visited = m_program.visited_blocks;
	auto& pending = m_program.pending_blocks;
	auto& walk    = m_program.active_walk;
	auto& slots   = m_program.walk_slots;
	auto& stamps  = m_program.walk_slot_stamps;
	visited.assign(m_program.control_flow.size(), 0u);
	pending.clear();
	pending.push_back(0u);
	walk.clear();
	slots.clear();
	m_program.active_walk_valid = false;
	// A read listed under several blocks was evaluated once per block (91.6 evaluations per walk
	// for 65.3 reads in the jungle); each is evaluated once per walk now.
	if (stamps.size() != m_program.srt_reads.size() || ++m_program.walk_generation == 0u) {
		stamps.assign(m_program.srt_reads.size(), 0u);
		m_program.walk_generation = 1u;
	}
	const auto stamp = m_program.walk_generation;
	while (!pending.empty()) {
		const auto index = pending.back();
		pending.pop_back();
		if (visited.at(index)) continue;
		visited[index] = 1u;
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) active[source] = 1u;
		for (const auto slot: block.srt_reads) {
			if (slot < stamps.size()) {
				if (stamps[slot] == stamp) continue;
				stamps[slot] = stamp;
			}
			if (!refresh(slot)) return false;
			slots.push_back(slot);
		}
		const auto result = outcome(index, block);
		if (!block.condition.IsEmpty()) {
			walk.emplace_back(index, result);
		}
		if (result != 2u) {
			pending.push_back(block.successors[result != 0u ? 0u : 1u]);
		} else {
			pending.insert(pending.end(), block.successors.begin(), block.successors.end());
		}
	}
	m_program.active_walk_result = active;
	m_program.walk_key           = key;
	m_program.active_walk_valid  = true;
	return true;
}

// Replay traces (SrtWalker.h).

namespace {

bool TraceEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_TRACE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

bool TraceStatsEnabled() {
	static const bool enabled = std::getenv("KYTY_SRT_TRACE_STATS") != nullptr;
	return enabled;
}

// A program records once it has been refreshed this often; after this many abandoned replays in a
// row it stops recording for `TraceBackoff` refreshes.
constexpr uint32_t TraceAfterUses = 4;
constexpr uint32_t TraceMaxMisses = 4;
constexpr uint32_t TraceBackoff   = 256;
constexpr size_t   TraceMaxOps    = 8192;

struct TraceStats {
	uint64_t served = 0, abandoned = 0, recorded = 0, aborted = 0, off = 0, refreshes = 0;
	std::unordered_map<std::string, uint64_t> reasons;
};
TraceStats g_trace_stats;

void CountTrace(uint64_t TraceStats::*field, const char* reason = nullptr) {
	if (field == &TraceStats::refreshes) {
		Profiler::Add(Profiler::Counter::SrtRefreshes);
	} else if (field == &TraceStats::served) {
		Profiler::Add(Profiler::Counter::SrtReplayed);
	} else if (field == &TraceStats::abandoned) {
		Profiler::Add(Profiler::Counter::SrtAbandoned);
	} else if (field == &TraceStats::recorded) {
		Profiler::Add(Profiler::Counter::SrtRecorded);
	}
	if (!TraceStatsEnabled()) {
		return;
	}
	g_trace_stats.*field += 1;
	if (reason != nullptr) {
		g_trace_stats.reasons[reason]++;
	}
}

uint8_t TraceKey(const ResourcePlan& program, const SrtRuntime& runtime) {
	return (runtime.read_specialization_memory != nullptr ? 1u : 0u) |
	       (program.capture_specialization_reads ? 2u : 0u);
}

void PrintTraceStats() {
	std::string reasons;
	for (const auto& [reason, count]: g_trace_stats.reasons) {
		reasons += fmt::format(" {}={}", reason, count);
	}
	std::fprintf(stderr, "SRT trace: refreshes=%llu served=%llu abandoned=%llu recorded=%llu aborted=%llu "
	             "off=%llu;%s\n",
	             static_cast<unsigned long long>(g_trace_stats.refreshes),
	             static_cast<unsigned long long>(g_trace_stats.served),
	             static_cast<unsigned long long>(g_trace_stats.abandoned),
	             static_cast<unsigned long long>(g_trace_stats.recorded),
	             static_cast<unsigned long long>(g_trace_stats.aborted),
	             static_cast<unsigned long long>(g_trace_stats.off), reasons.c_str());
}

} // namespace

bool& SrtTraceSession::Suppressed() {
	thread_local bool suppressed = false;
	return suppressed;
}

SrtTraceSession::SrtTraceSession(const ResourcePlan& program, SrtWalker& clean, SrtWalker& walker,
                                 const SrtRuntime& runtime)
    : m_program(program), m_walkers {&clean, &walker} {
	if (!TraceEnabled() || Suppressed() || !program.srt_plan_complete) {
		return;
	}
	CountTrace(&TraceStats::refreshes);
	if (TraceStatsEnabled() && g_trace_stats.refreshes % 200000 == 0) {
		PrintTraceStats();
	}
	const auto key = TraceKey(program, runtime);
	if (program.srt_trace_backoff != 0) {
		program.srt_trace_backoff--;
		CountTrace(&TraceStats::off);
		return;
	}
	if (program.srt_trace != nullptr && program.srt_trace->key == key) {
		m_mode  = Mode::Serve;
		m_trace = program.srt_trace;
		thread_local std::vector<uint64_t> values;
		thread_local std::vector<uint8_t>  status;
		if (values.size() < m_trace->ops.size()) {
			values.resize(m_trace->ops.size());
			status.resize(m_trace->ops.size());
		}
		m_values = values.data();
		m_status = status.data();
	} else if (++program.srt_trace_uses >= TraceAfterUses) {
		m_mode      = Mode::Record;
		m_recording = std::make_unique<SrtTrace>();
		m_recording->key = key;
		for (auto& slots: m_slots) {
			slots.assign(program.evaluation_value_count, -1);
		}
	} else {
		return;
	}
	for (uint8_t id = 0; id < 2; id++) {
		m_walkers[id]->m_trace    = this;
		m_walkers[id]->m_trace_id = id;
		// Recording sees every evaluation only through the interpreter; replay needs none.
		m_native[id]              = m_walkers[id]->m_native;
		m_walkers[id]->m_native   = nullptr;
	}
}

SrtTraceSession::~SrtTraceSession() {
	if (m_mode == Mode::Off) {
		return;
	}
	for (auto* walker: m_walkers) {
		walker->m_trace = nullptr;
	}
	if (m_mode == Mode::Record) {
		if (m_succeeded && m_abort == nullptr && m_frames.empty()) {
			m_program.srt_trace = std::move(m_recording);
			CountTrace(&TraceStats::recorded);
		} else {
			CountTrace(&TraceStats::aborted, m_abort != nullptr ? m_abort : "refresh failed");
			if (m_abort != nullptr && ++m_program.srt_trace_misses >= TraceMaxMisses) {
				m_program.srt_trace_misses  = 0;
				m_program.srt_trace_backoff = TraceBackoff;
			}
		}
	} else if (m_mode == Mode::Serve) {
		m_program.srt_trace_misses = 0;
		CountTrace(&TraceStats::served);
	}
}

void SrtTraceSession::RestoreNative() {
	for (uint8_t id = 0; id < 2; id++) {
		m_walkers[id]->m_native = m_native[id];
	}
}

void SrtTraceSession::Abort(const char* reason) {
	if (m_mode == Mode::Record && m_abort == nullptr) {
		m_abort = reason;
	}
}

void SrtTraceSession::Abandon(const char* reason) {
	m_mode = Mode::Passthrough;
	RestoreNative();
	CountTrace(&TraceStats::abandoned, reason);
	// The trace no longer matches what the program does: record again, unless that keeps
	// happening.
	m_program.srt_trace = nullptr;
	if (++m_program.srt_trace_misses >= TraceMaxMisses) {
		m_program.srt_trace_misses  = 0;
		m_program.srt_trace_backoff = TraceBackoff;
	}
}

bool SrtTraceSession::EvaluateSlow(SrtWalker& walker, Value value, uint64_t& result) {
	if (m_mode == Mode::Serve) {
		if (m_frames.empty()) {
			return Serve(walker, value, result);
		}
	}
	if (m_mode != Mode::Record) {
		return walker.EvaluateWideImpl(value, result);
	}
	const bool top   = m_frames.empty();
	auto       first = static_cast<uint32_t>(m_recording->ops.size());
	(void)first;
	m_frames.emplace_back();
	const bool ok    = walker.EvaluateWideImpl(value, result);
	const auto frame = m_frames.back();
	m_frames.pop_back();
	if (m_abort != nullptr) {
		return ok;
	}
	if (!frame.have) {
		// Nothing told the frame what happened (native code or an evaluation the hooks do not
		// see): this refresh cannot be recorded.
		Abort("unrecorded evaluation");
		return ok;
	}
	if (frame.result.ok != ok) {
		Abort("result mismatch");
		return ok;
	}
	if (top) {
		const auto resolved = value.Resolve();
		SrtTraceCall call;
		call.raw    = value;
		call.inst   = resolved.IsImmediate() ? nullptr : resolved.TryInstruction();
		call.end    = static_cast<uint32_t>(m_recording->ops.size());
		call.result = frame.result.ref;
		call.walker = walker.m_trace_id;
		call.ok     = ok;
		if (call.inst == nullptr) {
			call.immediate = result;
		}
		m_recording->calls.push_back(call);
		return ok;
	}
	auto& parent = m_frames.back();
	if (parent.count == parent.events.size()) {
		Abort("too many operands");
		return ok;
	}
	parent.events[parent.count++] = frame.result;
	return ok;
}

void SrtTraceSession::OnImmediate(bool ok, uint64_t bits) {
	if (m_mode != Mode::Record || m_frames.empty()) {
		return;
	}
	auto& frame = m_frames.back();
	if (!ok) {
		frame.result = {Failed, false};
	} else {
		auto& imms = m_recording->immediates;
		imms.push_back(bits);
		frame.result = {-1 - static_cast<int32_t>(imms.size() - 1), true, bits};
	}
	frame.have = true;
}

void SrtTraceSession::OnMemoHit(const SrtWalker& walker, uint32_t index, uint64_t value) {
	if (m_mode != Mode::Record || m_frames.empty()) {
		return;
	}
	auto& slots = m_slots[walker.m_trace_id];
	if (index >= slots.size() || slots[index] < 0) {
		Abort("memo hit without op");
		return;
	}
	auto& frame  = m_frames.back();
	frame.result = {slots[index], true, value};
	frame.have   = true;
}

void SrtTraceSession::OnInst(const SrtWalker& walker, const Inst& inst, uint32_t index, bool ok,
                             uint64_t value) {
	if (m_mode != Mode::Record || m_frames.empty() || m_abort != nullptr) {
		return;
	}
	auto&      frame = m_frames.back();
	SrtTraceOp op;
	op.walker = walker.m_trace_id;
	op.ok     = ok;
	op.inst   = &inst;
	op.opcode = inst.GetOpcode();
	op.count  = static_cast<uint8_t>(frame.count);
	if (frame.count > std::size(op.args)) {
		Abort("too many operands");
		return;
	}
	for (uint32_t i = 0; i < frame.count; i++) {
		op.args[i] = frame.events[i].ref;
	}
	const auto strict = [&](uint32_t expected) {
		// Operands in order; the first that fails stops the instruction.
		if (frame.count > expected) return false;
		if (frame.count < expected) {
			return frame.count != 0 && !frame.events[frame.count - 1].ok && !ok;
		}
		return true;
	};
	bool supported = false;
	if (const int arity = PureArity(op.opcode); arity >= 0) {
		op.kind   = SrtTraceOp::Pure;
		supported = strict(static_cast<uint32_t>(arity));
	} else {
		switch (op.opcode) {
			case ValueOpcode::GetUserData: {
				const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
				op.kind        = SrtTraceOp::UserData;
				op.imm         = static_cast<int32_t>(reg) - static_cast<int32_t>(m_program.user_data_base);
				supported      = frame.count == 0;
				break;
			}
			case ValueOpcode::GetShaderBase:
				op.kind   = SrtTraceOp::ShaderBase;
				supported = frame.count == 0;
				break;
			case ValueOpcode::Phi:
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::ReadConst:
			case ValueOpcode::ConditionRef:
				op.kind   = SrtTraceOp::Pass;
				supported = frame.count == 1;
				break;
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2: {
				const auto component = inst.Arg(1).Resolve();
				if (!component.IsImmediate() || component.GetType() != Type::U32 ||
				    component.U32() >= 2u) {
					break;
				}
				op.flags            = static_cast<uint8_t>(component.U32());
				const auto* source  = inst.Arg(0).ResolveInstruction();
				if (op.opcode == ValueOpcode::CompositeExtractU64) {
					op.kind   = SrtTraceOp::Extract64;
					supported = frame.count == 1;
				} else if (source != nullptr &&
				           source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
					op.kind   = SrtTraceOp::Pass;
					supported = frame.count == 1;
				} else if (source != nullptr && source->GetOpcode() == ValueOpcode::IAddCarry32) {
					op.kind   = SrtTraceOp::AddCarryExtract;
					supported = strict(2);
				}
				break;
			}
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectF32:
				op.kind = SrtTraceOp::Select;
				if (frame.count == 1) {
					supported = !frame.events[0].ok && !ok;
				} else if (frame.count == 2 && frame.events[0].ok) {
					// The arm the predicate chose.
					op.flags  = frame.events[0].value != 0u ? 1u : 2u;
					supported = true;
				}
				break;
			case ValueOpcode::LogicalAnd:
				op.kind   = SrtTraceOp::LogicalAnd;
				supported = frame.count == 1 || frame.count == 2;
				break;
			case ValueOpcode::LogicalOr:
				op.kind   = SrtTraceOp::LogicalOr;
				supported = frame.count == 1 || frame.count == 2;
				break;
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer: {
				if (!IsRawRead(m_program, inst)) break;
				const auto  flags  = inst.Flags<MemoryFlags>();
				const auto* handle = inst.Arg(0).ResolveInstruction();
				if (handle == nullptr) break;
				op.kind  = SrtTraceOp::RawRead;
				op.imm   = static_cast<int32_t>(m_program.memory_info[flags.index].offset);
				op.flags = op.opcode == ValueOpcode::ReadConstBuffer ? 1u : 0u;
				if (op.flags != 0u && handle->NumArgs() != 4u) break;
				supported = strict(op.flags != 0u ? 5u : 3u);
				break;
			}
			default: break;
		}
	}
	if (!supported) {
		Abort("unsupported instruction");
		return;
	}
	auto& ops = m_recording->ops;
	if (ops.size() >= TraceMaxOps) {
		Abort("trace too long");
		return;
	}
	ops.push_back(op);
	const auto ref = static_cast<int32_t>(ops.size() - 1);
	if (ok) {
		auto& slots = m_slots[walker.m_trace_id];
		if (index < slots.size()) {
			slots[index] = ref;
		}
	}
	frame.result = {ref, ok, value};
	frame.have   = true;
}

bool SrtTraceSession::Serve(SrtWalker& walker, Value value, uint64_t& result) {
	const auto& trace = *m_trace;
	if (m_cursor >= trace.calls.size()) {
		Abandon("more calls than recorded");
		return walker.EvaluateWideImpl(value, result);
	}
	const auto& call = trace.calls[m_cursor];
	if (call.walker != walker.m_trace_id) {
		Abandon("different call");
		return walker.EvaluateWideImpl(value, result);
	}
	const Inst* inst = call.inst;
	if (!(call.raw == value)) {
		// The same value through another identity chain still matches.
		const auto resolved = value.Resolve();
		if ((resolved.IsImmediate() ? nullptr : resolved.TryInstruction()) != call.inst) {
			Abandon("different call");
			return walker.EvaluateWideImpl(value, result);
		}
	}
	if (inst == nullptr) {
		m_cursor++;
		return walker.EvaluateWideImpl(value, result);
	}
	if (!RunOps(call.end)) {
		Abandon("guard");
		return walker.EvaluateWideImpl(value, result);
	}
	m_cursor++;
	if (call.result == Failed) {
		return false;
	}
	if (call.result < 0) {
		result = trace.immediates[static_cast<size_t>(-1 - call.result)];
		return true;
	}
	result = m_values[static_cast<size_t>(call.result)];
	return m_status[static_cast<size_t>(call.result)] != 0u;
}

bool SrtTraceSession::RunOps(uint32_t end) {
	const auto& trace = *m_trace;
	const auto* ops   = trace.ops.data();
	const auto* imms  = trace.immediates.data();
	auto*       vals  = m_values;
	auto*       stat  = m_status;
	for (uint32_t i = m_next_op; i < end; i++) {
		const auto& op = ops[i];
		uint64_t    v[5] {};
		bool        ok[5] {};
		bool        all = true;
		for (uint32_t j = 0; j < op.count; j++) {
			const auto ref = op.args[j];
			if (ref >= 0) {
				v[j]  = vals[ref];
				ok[j] = stat[ref] != 0u;
			} else if (ref != Failed) {
				v[j]  = imms[-1 - ref];
				ok[j] = true;
			}
			all = all && ok[j];
		}
		bool     status = false;
		uint64_t value  = 0;
		switch (op.kind) {
			case SrtTraceOp::Pure:
				status = all && op.count == static_cast<uint32_t>(PureArity(op.opcode)) &&
				         ApplyPureOp(op.opcode, *op.inst, v, value);
				break;
			case SrtTraceOp::UserData: {
				const auto& user = m_walkers[op.walker]->m_runtime.user_data;
				status           = op.imm >= 0 && static_cast<size_t>(op.imm) < user.size();
				value            = status ? user[static_cast<size_t>(op.imm)] : 0u;
				break;
			}
			case SrtTraceOp::ShaderBase:
				status = true;
				value  = m_walkers[op.walker]->m_runtime.shader_base;
				break;
			case SrtTraceOp::Pass:
				status = ok[0];
				value  = v[0];
				break;
			case SrtTraceOp::Extract64:
				status = ok[0];
				value  = static_cast<uint32_t>(v[0] >> (op.flags * 32u));
				break;
			case SrtTraceOp::AddCarryExtract: {
				status = all && op.count == 2;
				const auto sum =
				    static_cast<uint64_t>(static_cast<uint32_t>(v[0])) + static_cast<uint32_t>(v[1]);
				value = op.flags == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
				break;
			}
			case SrtTraceOp::Select:
				if (op.count == 1) {
					// The predicate failed when recorded; if it succeeds now an arm is needed.
					if (ok[0]) return false;
					status = false;
				} else {
					if (!ok[0] || (v[0] != 0u ? 1u : 2u) != op.flags) return false;
					status = ok[1];
					value  = v[1];
				}
				break;
			case SrtTraceOp::LogicalAnd:
				if (ok[0] && v[0] == 0u) {
					if (op.count != 1) return false;
					status = true;
					value  = 0u;
				} else {
					if (op.count != 2) return false;
					status = ok[1] && !(v[1] != 0u && !ok[0]);
					value  = v[1] != 0u;
				}
				break;
			case SrtTraceOp::LogicalOr:
				if (ok[0] && v[0] != 0u) {
					if (op.count != 1) return false;
					status = true;
					value  = 1u;
				} else {
					if (op.count != 2) return false;
					status = ok[1] && !(v[1] == 0u && !ok[0]);
					value  = v[1] != 0u;
				}
				break;
			case SrtTraceOp::RawRead: {
				const bool const_buffer = op.flags != 0u;
				if (!all || op.count != (const_buffer ? 5u : 3u)) {
					status = false;
					break;
				}
				uint64_t base    = 0;
				uint64_t address = 0;
				switch (ComputeRawAddress(const_buffer, op.imm, v[0], v[1], v[2],
				                          const_buffer ? v[3] : 0u, base, address)) {
					case RawAddress::Fail: status = false; break;
					case RawAddress::Zero:
						status = true;
						value  = 0u;
						break;
					case RawAddress::Read:
						status = m_walkers[op.walker]->ReadRawWord(address, base, value);
						break;
				}
				break;
			}
		}
		if (status != op.ok) {
			return false;
		}
		vals[i] = value;
		stat[i] = status ? 1u : 0u;
	}
	if (end > m_next_op) {
		m_next_op = end;
	}
	return true;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}


} // namespace Libs::Graphics::ShaderRecompiler::IR
