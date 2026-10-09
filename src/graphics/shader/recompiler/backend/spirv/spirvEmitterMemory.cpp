#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t AndCondition(EmitterState& state, uint32_t lhs, uint32_t rhs) {
	return Binary(state, OpLogicalAnd, TypeBool(state), lhs, rhs);
}

uint32_t EmitDsMaskedLaneRead(EmitterState& state, uint32_t source, uint32_t target,
                              uint32_t exec) {
	if (state.lane_count == 2) {
		target = Binary(state, OpBitwiseAnd, TypeU32(state), target, ConstantU32(state, 31));
	}
	const auto shuffled = state.builder.AllocateId();
	state.builder.AddFunction({OpGroupNonUniformShuffle, TypeU32(state), shuffled,
	                           ConstantU32(state, ScopeSubgroup), source, target});
	const auto source_exec = state.builder.AllocateId();
	state.builder.AddFunction({OpGroupNonUniformShuffle, TypeBool(state), source_exec,
	                           ConstantU32(state, ScopeSubgroup), exec, target});
	const auto source_active =
	    AndCondition(state, source_exec, EmitSubgroupLaneActiveBool(state, target));
	return Select(state, TypeU32(state), source_active, shuffled, ConstantU32(state, 0));
}

uint32_t BufferByteAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                           uint32_t index, uint32_t offset, uint32_t soffset) {
	auto&          state   = ctx.state;
	const uint32_t packed  = StorageBufferPackedStride(state, mem);
	const uint32_t stride  = packed & 0x3fffu;
	const bool     swizzle = stride != 0u && ((packed >> 14u) & 1u) != 0u;
	if (((packed >> 20u) & 1u) != 0u) {
		// (ADD_TID: the GCN lane. A lane-local wave64 program's subgroup is half a wave: its index in the group.)
		const auto id   = state.lane_count == 1 && state.program.lane_local ? EmitLocalInvocationIndex(state)
		                                                                    : EmitSubgroupLocalInvocationId(state);
		const auto lane = Binary(state, OpBitwiseAnd, TypeU32(state), id, ConstantU32(state, 63));
		index           = Binary(state, OpIAdd, TypeU32(state), index, lane);
	}
	if (mem.offset != 0u) {
		offset = Binary(state, OpIAdd, TypeU32(state), offset, ConstantU32(state, mem.offset));
	}

	uint32_t address = 0;
	if (!swizzle && state.program.table_mode) {
		// Table mode: the V#'s own stride (EmitTableMode; 0 ignores the index).
		address = !mem.idxen && (packed & IR::BufferAddTidBit) == 0u
		              ? offset
		              : Binary(state, OpIAdd, TypeU32(state),
		                       Binary(state, OpIMul, TypeU32(state), index, state.table_strides.at(mem.resource)), offset);
	} else if (!swizzle && state.program.bindings.buffer_word_count != 0 && IR::RuntimeBufferStride(packed)) {
		// The stride of the buffer word: one module whatever the V# holds (0 ignores the index).
		if (!mem.idxen && (packed & IR::BufferAddTidBit) == 0u) {
			address = offset;
		} else {
			const auto word           = RuntimeBufferWord(state, mem);
			const auto runtime_stride = Binary(state, OpBitwiseAnd, TypeU32(state), word,
			                                   ConstantU32(state, IR::BufferWord::MaxStride));
			address = Binary(state, OpIAdd, TypeU32(state),
			                 Binary(state, OpIMul, TypeU32(state), index, runtime_stride), offset);
		}
	} else if (!swizzle) {
		if (stride == 0u) {
			address = offset;
		} else {
			const auto indexed = stride == 1u ? index
			                                  : Binary(state, OpIMul, TypeU32(state), index,
			                                           ConstantU32(state, stride));
			address            = Binary(state, OpIAdd, TypeU32(state), indexed, offset);
		}
	} else {
		const uint32_t stride_enum  = (packed >> 16u) & 3u;
		const uint32_t index_stride = 8u << stride_enum;
		const auto     index_msb    = Binary(state, OpShiftRightLogical, TypeU32(state), index,
		                                     ConstantU32(state, stride_enum + 3u));
		const auto     index_lsb    = Binary(state, OpBitwiseAnd, TypeU32(state), index,
		                                     ConstantU32(state, index_stride - 1u));
		const auto     offset_msb =
		    Binary(state, OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, ~3u));
		const auto offset_lsb =
		    Binary(state, OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, 3u));
		const auto indexed_msb = stride == 1u ? index_msb
		                                      : Binary(state, OpIMul, TypeU32(state), index_msb,
		                                               ConstantU32(state, stride));
		const auto msb = Binary(state, OpIMul, TypeU32(state),
		                        Binary(state, OpIAdd, TypeU32(state), indexed_msb, offset_msb),
		                        ConstantU32(state, index_stride));
		const auto lsb = Binary(
		    state, OpIAdd, TypeU32(state),
		    Binary(state, OpShiftLeftLogical, TypeU32(state), index_lsb, ConstantU32(state, 2u)),
		    offset_lsb);
		address = Binary(state, OpIAdd, TypeU32(state), msb, lsb);
	}

	const auto soffset_value = inst.Arg(3).Resolve();
	if (soffset_value.IsImmediate() && soffset_value.GetType() == IR::Type::U32 &&
	    soffset_value.U32() == 0u) {
		return address;
	}
	return Binary(state, OpIAdd, TypeU32(state), address, soffset);
}

uint32_t AddU64Low(EmitterState& state, uint32_t low, uint32_t high, uint32_t add_low,
                   uint32_t add_high, uint32_t& out_high) {
	const auto result = Binary(state, OpIAdd, TypeU32(state), low, add_low);
	const auto carry  = Binary(state, OpULessThan, TypeBool(state), result, low);
	out_high =
	    Binary(state, OpIAdd, TypeU32(state), Binary(state, OpIAdd, TypeU32(state), high, add_high),
	           Select(state, TypeU32(state), carry, ConstantU32(state, 1), ConstantU32(state, 0)));
	return result;
}

uint32_t ScratchByteAddress(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t low,
                            uint32_t high) {
	auto& state     = ctx.state;
	auto  immediate = static_cast<int32_t>(mem.offset);
	const auto immediate_low  = ConstantU32(state, static_cast<uint32_t>(immediate));
	const auto immediate_high = ConstantU32(state, immediate < 0 ? UINT32_MAX : 0u);
	low                      = AddU64Low(state, low, high, immediate_low, immediate_high, high);
	const auto valid =
	    Binary(state, OpIEqual, TypeBool(state), high, ConstantU32(state, 0));
	return Select(state, TypeU32(state), valid, low, ConstantU32(state, UINT32_MAX));
}

uint32_t ConstantDeviceAddress(EmitterState& state, uint64_t value) {
	return state.builder.Constant(OpConstant, TypeDeviceAddress(state),
	                              {static_cast<uint32_t>(value),
	                               static_cast<uint32_t>(value >> 32u)});
}

uint32_t DeviceAddressFromWords(EmitterState& state, uint32_t low, uint32_t high) {
	const auto low64  = Unary(state, OpUConvert, TypeDeviceAddress(state), low);
	const auto high64 = Binary(state, OpShiftLeftLogical, TypeDeviceAddress(state),
	                           Unary(state, OpUConvert, TypeDeviceAddress(state), high),
	                           ConstantDeviceAddress(state, 32));
	return Binary(state, OpBitwiseOr, TypeDeviceAddress(state), low64, high64);
}

uint32_t GuestAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto& state = ctx.state;
	auto  low   = ctx.Arg(inst, 1);
	if (mem.kind == IR::ResourceKind::ScalarAddress) {
		low = Binary(state, OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, ~3u));
	}
	uint32_t address = 0;
	if (mem.address_is_full) {
		address = DeviceAddressFromWords(state, low, ctx.Arg(inst, 2));
	} else {
		const auto* handle = inst.Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != IR::ValueOpcode::GetAddressResource ||
		    handle->NumArgs() != 2) {
			ctx.Fail(inst, "has no address base pair");
			return ConstantDeviceAddress(state, 0);
		}
		const auto base = DeviceAddressFromWords(state, ctx.Arg(*handle, 0), ctx.Arg(*handle, 1));
		address = Binary(state, OpIAdd, TypeDeviceAddress(state), base,
		                 Unary(state, OpUConvert, TypeDeviceAddress(state), low));
	}
	auto immediate = static_cast<int32_t>(mem.offset);
	if (mem.kind == IR::ResourceKind::ScalarAddress) {
		immediate = static_cast<int32_t>(static_cast<uint32_t>(immediate) & ~3u);
	}
	return immediate == 0
	           ? address
	           : Binary(state, OpIAdd, TypeDeviceAddress(state), address,
	                    ConstantDeviceAddress(state,
	                                          static_cast<uint64_t>(static_cast<int64_t>(immediate))));
}

uint32_t FaultElementPointer(EmitterState& state, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction({OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                           state.fault_buffer_variable, ConstantU32(state, 0), index});
	return pointer;
}

void RecordBdaFault(EmitterState& state, uint32_t page) {
	const auto word = Binary(state, OpShiftRightLogical, TypeU32(state), page,
	                         ConstantU32(state, 5));
	const auto bit = Binary(
	    state, OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1),
	    Binary(state, OpBitwiseAnd, TypeU32(state), page, ConstantU32(state, 31)));
	const auto pointer = FaultElementPointer(state, word);
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction({OpLoad, TypeU32(state), value, pointer});
	state.builder.AddFunction(
	    {OpStore, pointer, Binary(state, OpBitwiseOr, TypeU32(state), value, bit)});
}

uint32_t GetBdaPointer(ValueEmitContext& ctx, uint32_t address) {
	auto&      state  = ctx.state;
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(
	    {OpFunctionCall, TypeDeviceAddress(state), result, state.bda_pointer_function, address});
	return result;
}

uint32_t LoadBdaDword(ValueEmitContext& ctx, uint32_t address) {
	auto&      state   = ctx.state;
	const auto bda     = GetBdaPointer(ctx, address);
	const auto present = Binary(state, OpINotEqual, TypeBool(state), bda,
	                            ConstantDeviceAddress(state, 0));
	return EmitValueOrZeroIfCondition(state, present, [&]() {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(
		    {OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, bda});
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(
		    {OpLoad, TypeU32(state), value, pointer, MemoryAccessAlignedMask, sizeof(uint32_t)});
		return value;
	});
}

uint32_t LoadBda(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
	             uint32_t bits) {
	auto&      state   = ctx.state;
	const auto address = GuestAddress(ctx, inst, mem);
	const auto active  = ctx.Arg(inst, inst.NumArgs() - 1);
	return EmitValueOrZeroIfCondition(state, active, [&]() {
		const auto aligned = Binary(state, OpBitwiseAnd, TypeDeviceAddress(state), address,
		                            ConstantDeviceAddress(state, ~uint64_t {3}));
		const auto first   = LoadBdaDword(ctx, aligned);
		const auto byte = Binary(state, OpBitwiseAnd, TypeU32(state),
		                         Unary(state, OpUConvert, TypeU32(state), address),
		                         ConstantU32(state, 3));
		const auto crosses = bits == 8u
		                          ? ConstantBool(state, false)
		                          : Binary(state, bits == 16u ? OpUGreaterThan : OpINotEqual,
		                                   TypeBool(state), byte,
		                                   ConstantU32(state, bits == 16u ? 2u : 0u));
		const auto second = EmitValueOrZeroIfCondition(state, crosses, [&]() {
			return LoadBdaDword(
			    ctx, Binary(state, OpIAdd, TypeDeviceAddress(state), aligned,
			                ConstantDeviceAddress(state, sizeof(uint32_t))));
		});
		const auto shift = Binary(state, OpShiftLeftLogical, TypeU32(state), byte,
		                          ConstantU32(state, 3));
		const auto upper_shift = Binary(
		    state, OpShiftLeftLogical, TypeU32(state),
		    Binary(state, OpBitwiseAnd, TypeU32(state),
		           Binary(state, OpISub, TypeU32(state), ConstantU32(state, 4), byte),
		           ConstantU32(state, 3)),
		    ConstantU32(state, 3));
		const auto merged = Binary(
		    state, OpBitwiseOr, TypeU32(state),
		    Binary(state, OpShiftRightLogical, TypeU32(state), first, shift),
		    Binary(state, OpShiftLeftLogical, TypeU32(state), second, upper_shift));
		return bits == 32u
		           ? merged
		           : Binary(state, OpBitwiseAnd, TypeU32(state), merged,
		                    ConstantU32(state, bits == 8u ? 0xffu : 0xffffu));
	});
}

uint32_t ByteAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                     bool include_base_low_bits = true) {
	if (mem.kind == IR::ResourceKind::Buffer) {
		const auto address = BufferByteAddress(ctx, inst, mem, ctx.Arg(inst, 1), ctx.Arg(inst, 2),
		                                      ctx.Arg(inst, 3));
		if (!include_base_low_bits || !ctx.state.program.info.buffers[mem.resource].byte_base_offset ||
		    ctx.state.program.table_mode)
			return address;
		// The host descriptor is aligned down. Its whole-dword displacement is
		// added by EmitMemoryElementIndex; preserve the remaining byte displacement
		// BEFORE selecting a dword or extracting a byte, including carry into the
		// next dword. Scalar buffer reads deliberately ignore these base bits.
		const auto resource = ResourceForDescriptor(ctx.state, IR::DescriptorBindingKind::Buffers,
		                                            mem.resource);
		const auto low = Binary(ctx.state, OpBitwiseAnd, TypeU32(ctx.state),
		                        ctx.state.memory_byte_offsets[resource], ConstantU32(ctx.state, 3));
		return Binary(ctx.state, OpIAdd, TypeU32(ctx.state), address, low);
	}
	if (mem.kind == IR::ResourceKind::Lds || mem.kind == IR::ResourceKind::Gds) {
		if (mem.offset == 0u) {
			return ctx.Arg(inst, 0);
		}
		return Binary(ctx.state, OpIAdd, TypeU32(ctx.state), ctx.Arg(inst, 0),
		              ConstantU32(ctx.state, mem.offset));
	}
	if (mem.kind != IR::ResourceKind::Scratch) {
		EXIT("physical address memory must use the BDA emitter\n");
	}
	return ScratchByteAddress(ctx, mem, ctx.Arg(inst, 1), ctx.Arg(inst, 2));
}

uint32_t DwordIndex(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	const auto index = Binary(ctx.state, OpShiftRightLogical, TypeU32(ctx.state),
	                          ByteAddress(ctx, inst, mem), ConstantU32(ctx.state, 2));
	if (const auto slot = ctx.state.function_lds_slots.find(&inst);
	    slot != ctx.state.function_lds_slots.end()) {
		ctx.state.function_lds_index_slots.emplace(index, slot->second);
	}
	return index;
}

struct PreparedMemoryElement {
	MemoryResourceAccess resource;
	uint32_t             index = 0;
};

PreparedMemoryElement PrepareMemoryElement(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                           uint32_t raw_index) {
	auto       resource = PrepareMemoryResourceAccess(ctx.state, mem);
	const auto index    = EmitMemoryElementIndex(ctx.state, resource, raw_index);
	return {.resource = resource, .index = index};
}

uint32_t LoadWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                          uint32_t index);

uint32_t LoadSubwordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                             uint32_t address, uint32_t index, uint32_t bits, bool sign_extend);

uint32_t LoadWordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource) {
	const auto index = EmitMemoryElementIndex(ctx.state, resource, DwordIndex(ctx, inst, mem));
	return EmitValueOrZeroIfCondition(
	    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index),
	    [&]() { return LoadWordInBounds(ctx, resource, index); });
}

// A load runs for the lanes EXEC enables. A storage buffer load the device bounds needs no guard:
// a disabled lane loads harmlessly, and its register write (a Select on EXEC) drops the result.
static uint32_t LoadGuard(ValueEmitContext& ctx, const IR::Inst& inst, IR::ResourceKind kind) {
	return DeviceStorageBufferBounds(kind) ? ConstantBool(ctx.state, true)
	                                       : ctx.Arg(inst, inst.NumArgs() - 1);
}

uint32_t LoadWord(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem) {
	return EmitValueOrZeroIfCondition(ctx.state, LoadGuard(ctx, inst, mem.kind), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return LoadWordPrepared(ctx, inst, mem, resource);
	});
}

uint32_t LoadSubwordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                             const MemoryResourceAccess& resource, uint32_t bits,
                             bool sign_extend) {
	const auto address   = ByteAddress(ctx, inst, mem);
	const auto raw_index = Binary(ctx.state, OpShiftRightLogical, TypeU32(ctx.state), address,
	                              ConstantU32(ctx.state, 2));
	const auto index     = EmitMemoryElementIndex(ctx.state, resource, raw_index);
	return EmitValueOrZeroIfCondition(
	    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index), [&]() {
		    return LoadSubwordInBounds(ctx, resource, address, index, bits, sign_extend);
	    });
}

uint32_t LoadWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                          uint32_t index) {
	if (resource.bda_base != 0) {
		// Table mode: the buffer's device address (EmitTableMode). Out of bounds reads zero, as the device's bounds
		// make a storage buffer's: the load (of the first dword, which a null buffer's address also holds) is not
		// branched around, so loads issue together.
		auto&      state     = ctx.state;
		const auto in_bounds = EmitMemoryElementInBoundsExplicit(state, resource, index);
		const auto pointer   = EmitMemoryElementPointer(state, resource,
		                                                Select(state, TypeU32(state), in_bounds, index, ConstantU32(state, 0)));
		const auto value     = state.builder.AllocateId();
		state.builder.AddFunction({OpLoad, TypeU32(state), value, pointer, MemoryAccessAlignedMask, sizeof(uint32_t)});
		return Select(state, TypeU32(state), in_bounds, value, ConstantU32(state, 0));
	}
	const auto value   = ctx.state.builder.AllocateId();
	const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
	ctx.state.builder.AddFunction({OpLoad, TypeU32(ctx.state), value, pointer});
	return value;
}

uint32_t LoadSubwordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                             uint32_t address, uint32_t index, uint32_t bits, bool sign_extend) {
	const auto word = LoadWordInBounds(ctx, resource, index);
	const auto byte =
	    Binary(ctx.state, OpBitwiseAnd, TypeU32(ctx.state), address, ConstantU32(ctx.state, 3));
	const auto shift =
	    Binary(ctx.state, OpShiftLeftLogical, TypeU32(ctx.state), byte, ConstantU32(ctx.state, 3));
	const auto value =
	    Binary(ctx.state, OpBitwiseAnd, TypeU32(ctx.state),
	           Binary(ctx.state, OpShiftRightLogical, TypeU32(ctx.state), word, shift),
	           ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu));
	if (!sign_extend) return value;
	const auto left = Binary(ctx.state, OpShiftLeftLogical, TypeU32(ctx.state), value,
	                         ConstantU32(ctx.state, 32u - bits));
	return Binary(ctx.state, OpShiftRightArithmetic, TypeU32(ctx.state), left,
	              ConstantU32(ctx.state, 32u - bits));
}

uint32_t LoadSubword(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem, uint32_t bits,
                     bool sign_extend) {
	return EmitValueOrZeroIfCondition(ctx.state, LoadGuard(ctx, inst, mem.kind), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return LoadSubwordPrepared(ctx, inst, mem, resource, bits, sign_extend);
	});
}

Prospero::BufferFormat BufferFormat(const ValueEmitContext& ctx, const IR::MemoryInfo& mem) {
	return mem.typed ? Format::DecodeTBufferFormat(mem.data_format, mem.number_format)
	                 : StorageBufferFormat(ctx.state, mem);
}

IR::MemoryInfo RebaseFormattedComponent(IR::MemoryInfo mem, const Format::BufferFormatInfo& info,
                                        uint32_t component) {
	mem.offset += Format::GetFormatComponentByteOffset(info, component);
	mem.data_dwords     = 1u;
	mem.component_index = component;
	return mem;
}

IR::MemoryInfo RebaseRawComponent(IR::MemoryInfo mem, uint32_t component) {
	mem.offset += component * 4u;
	mem.data_dwords     = 1u;
	mem.component_index = component;
	return mem;
}

using Format::FormattedSource;
using Format::FormattedSourceKind;

FormattedSource ResolveFormattedSource(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                       const Format::BufferFormatInfo& info,
                                       uint32_t                        output_component) {
	if (mem.typed) {
		return output_component < info.component_count
		           ? FormattedSource {FormattedSourceKind::Memory, output_component}
		           : FormattedSource {};
	}
	const auto selector = GetDstSel(ctx.state.program.info.buffers[mem.resource].descriptor_swizzle,
	                                output_component);
	const auto source = Format::ResolveFormattedSource(info, selector);
	if (source.kind == FormattedSourceKind::Invalid) {
		ExitDescriptorBindingFailure(ctx.state, IR::DescriptorBindingKind::Buffers, mem.resource,
		                             "buffer descriptor has reserved dst_sel");
	}
	return source;
}

uint32_t FormattedConstant(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                           FormattedSourceKind kind) {
	return ConstantU32(ctx.state, Format::FormattedConstantBits(info, kind));
}

template <typename LoadWordFn, typename LoadSubwordFn>
uint32_t LoadFormattedComponent(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                const Format::BufferFormatInfo& info,
                                uint32_t output_component, LoadWordFn&& load_word,
                                LoadSubwordFn&& load_subword) {
	const auto source = ResolveFormattedSource(ctx, mem, info, output_component);
	if (source.kind != FormattedSourceKind::Memory) {
		return FormattedConstant(ctx, info, source.kind);
	}
	const auto component = source.component;
	const auto bits      = info.component_bits[component];
	uint32_t   raw       = 0;
	if (info.packed_bitfield) {
		raw = load_word(component);
		const auto type =
		    IsSignedFormatComponent(info.type) ? TypeI32(ctx.state) : TypeU32(ctx.state);
		const auto source_value =
		    type == TypeI32(ctx.state) ? Unary(ctx.state, OpBitcast, type, raw) : raw;
		const auto extracted = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(
		    {IsSignedFormatComponent(info.type) ? OpBitFieldSExtract : OpBitFieldUExtract, type,
		     extracted, source_value, ConstantU32(ctx.state, info.component_bit_offset[component]),
		     ConstantU32(ctx.state, bits)});
		raw = type == TypeI32(ctx.state)
		          ? Unary(ctx.state, OpBitcast, TypeU32(ctx.state), extracted)
		          : extracted;
	} else if (bits == 32u) {
		raw = load_word(component);
	} else {
		raw = load_subword(component, bits, IsSignedFormatComponent(info.type));
	}
	return NormalizeFormatComponent(ctx.state, info, component, raw);
}

// Runtime formats: a formatted buffer only loaded from (IR::RuntimeBufferFormat) whose
// specialization leaves the format open (IR::PortableFormats) takes its format and dst_sel from its
// buffer word (IR::BufferWord). The decode below gives the values of the specialized one
// (LoadFormattedComponent, LoadWideBuffer) for every format.
bool UsesRuntimeFormat(const EmitterState& state, const IR::MemoryInfo& mem) {
	return !mem.typed && (state.program.bindings.buffer_word_count != 0 || state.program.table_mode) &&
	       mem.resource < state.program.info.buffers.size() &&
	       IR::RuntimeBufferFormat(state.program.info.buffers[mem.resource]) &&
	       state.program.info.buffers[mem.resource].descriptor_format == Prospero::BufferFormat::kInvalid;
}

using SwitchCases = std::vector<std::pair<uint32_t, std::function<uint32_t()>>>;

// A value picked by a uniform runtime selector: one block per case, `fallback` for the others.
template <typename Fallback>
uint32_t EmitSwitchValue(EmitterState& state, uint32_t type, uint32_t selector, const SwitchCases& cases,
                         Fallback&& fallback) {
	const auto            default_label = state.builder.AllocateId();
	const auto            merge_label   = state.builder.AllocateId();
	std::vector<uint32_t> labels;
	std::vector<uint32_t> switch_words {OpSwitch, selector, default_label};
	for (const auto& [literal, value]: cases) {
		labels.push_back(state.builder.AllocateId());
		switch_words.push_back(literal);
		switch_words.push_back(labels.back());
	}
	state.builder.AddFunction({OpSelectionMerge, merge_label, SelectionControlNone});
	state.builder.AddFunction(switch_words);
	std::vector<uint32_t> phi_words {OpPhi, type, state.builder.AllocateId()};
	EmitLabel(state, default_label);
	phi_words.push_back(fallback());
	phi_words.push_back(state.current_label);
	state.builder.AddFunction({OpBranch, merge_label});
	size_t label = 0;
	for (const auto& [literal, value]: cases) {
		EmitLabel(state, labels[label++]);
		phi_words.push_back(value());
		phi_words.push_back(state.current_label);
		state.builder.AddFunction({OpBranch, merge_label});
	}
	EmitLabel(state, merge_label);
	state.builder.AddFunction(phi_words);
	return phi_words[2];
}

uint32_t WordField(EmitterState& state, uint32_t word, uint32_t shift, uint32_t bits) {
	return EmitAndConstant(state, EmitShiftRightConstant(state, word, shift), (1u << bits) - 1u);
}

// A packed layout's table (a byte per component), by layout: 11_11_10, 10_11_11, 2_10_10_10, 10_10_10_2.
uint32_t PackedLayoutTable(EmitterState& state, uint32_t layout, const std::array<uint32_t, 4>& tables) {
	auto value = ConstantU32(state, tables[3]);
	for (uint32_t index = 3; index-- > 0;) {
		value = Select(state, TypeU32(state),
		               EmitCompareU32Constant(state, OpIEqual, layout, IR::BufferWord::PackedLayouts + index),
		               ConstantU32(state, tables[index]), value);
	}
	return value;
}

} // namespace

// Once per buffer at the entry, instead of again at each component of each formatted load (a shader with
// hundreds of formatted loads compiled for minutes). Table mode: the buffer words EmitTableMode made from the
// V#s, by resource.
void EmitRuntimeFormats(EmitterState& state) {
	const auto* descriptor = IR::FindBinding(state.program.bindings, IR::DescriptorBindingKind::Buffers);
	const bool  table      = state.program.table_mode;
	if (descriptor == nullptr && !table) return;
	const auto count = table ? static_cast<uint32_t>(state.program.info.buffers.size())
	                         : std::min<uint32_t>(state.program.bindings.buffer_word_count,
	                                              static_cast<uint32_t>(descriptor->resources.size()));
	for (uint32_t index = 0; index < count; index++) {
		const auto resource = table ? index : descriptor->resources[index];
		if (resource >= state.program.info.buffers.size()) continue;
		const auto& buffer = state.program.info.buffers[resource];
		if (!IR::RuntimeBufferFormat(buffer) || buffer.descriptor_format != Prospero::BufferFormat::kInvalid) continue;
		auto& format  = state.runtime_formats[index];
		format.word   = table ? state.table_buffer_words.at(index) : state.buffer_words[index];
		format.layout = WordField(state, format.word, IR::BufferWord::LayoutShift, 4);
		format.type   = WordField(state, format.word, IR::BufferWord::TypeShift, 3);
		format.packed =
		    EmitCompareU32Constant(state, OpUGreaterThanEqual, format.layout, IR::BufferWord::PackedLayouts);
		// 11_11_10 and 10_11_11 have three components, 2_10_10_10 and 10_10_10_2 four.
		const auto packed_count = Select(
		    state, TypeU32(state),
		    EmitCompareU32Constant(state, OpULessThan, format.layout, IR::BufferWord::PackedLayouts + 2u),
		    ConstantU32(state, 3), ConstantU32(state, 4));
		format.count = Select(state, TypeU32(state), format.packed, packed_count,
		                      EmitAddU32(state, EmitAndConstant(state, format.layout, 3), ConstantU32(state, 1)));
		format.log_bytes    = EmitShiftRightConstant(state, format.layout, 2);
		format.packed_shift = PackedLayoutTable(state, format.layout, {0x00160b00u, 0x00150a00u, 0x160c0200u, 0x1e140a00u});
		format.packed_bits  = PackedLayoutTable(state, format.layout, {0x000a0b0bu, 0x000b0b0au, 0x0a0a0a02u, 0x020a0a0au});
		const auto integer  = EmitLogicalOrBool(
            state, EmitCompareU32Constant(state, OpIEqual, format.type, uint32_t(Format::ComponentType::Uint)),
            EmitCompareU32Constant(state, OpIEqual, format.type, uint32_t(Format::ComponentType::Sint)));
		format.one = Select(state, TypeU32(state), integer, ConstantU32(state, 1), ConstantU32(state, 0x3f800000u));
		format.is_signed = EmitCompareU32Constant(state, OpIEqual, EmitAndConstant(state, format.type, 1), 0);
	}
}

namespace {

const RuntimeFormat& LoadRuntimeFormat(EmitterState& state, const IR::MemoryInfo& mem) {
	return state.runtime_formats[state.program.table_mode
	                                 ? mem.resource
	                                 : ResourceForDescriptor(state, IR::DescriptorBindingKind::Buffers, mem.resource)];
}

// dst_sel of an output component: 0 zero, 1 one, 4-7 a memory component (2 and 3 reserved).
uint32_t RuntimeSelector(EmitterState& state, const RuntimeFormat& format, uint32_t output_component) {
	return WordField(state, format.word, IR::BufferWord::DstSelShift + 3u * output_component, 3);
}

// The memory component a selector reads (ResolveFormattedSource): formatted decoding expands the
// element's components before the swizzle.
uint32_t RuntimeSourceComponent(EmitterState& state, const RuntimeFormat& format, uint32_t selector) {
	return Binary(state, OpUMod, TypeU32(state), EmitAndConstant(state, selector, 3), format.count);
}

// The byte address of a buffer access with `extra` bytes added to its offset operand (as the
// specialized decode rebases a component's offset).
uint32_t BufferAddressPlus(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                           uint32_t extra) {
	auto&      state   = ctx.state;
	const auto offset  = Binary(state, OpIAdd, TypeU32(state), ctx.Arg(inst, 2), extra);
	const auto address = BufferByteAddress(ctx, inst, mem, ctx.Arg(inst, 1), offset, ctx.Arg(inst, 3));
	if (!state.program.info.buffers[mem.resource].byte_base_offset) return address;
	const auto resource = ResourceForDescriptor(state, IR::DescriptorBindingKind::Buffers, mem.resource);
	return Binary(state, OpIAdd, TypeU32(state), address,
	              EmitAndConstant(state, state.memory_byte_offsets[resource], 3));
}

// Where component `component` of the element lies: the byte address of its dword, and its bit
// offset and width there.
struct RuntimeComponent {
	uint32_t address = 0;
	uint32_t index   = 0; // memory element (dword) index
	uint32_t shift   = 0;
	uint32_t bits    = 0;
};

RuntimeComponent LocateRuntimeComponent(ValueEmitContext& ctx, const IR::Inst& inst,
                                        const IR::MemoryInfo& mem, const MemoryResourceAccess& resource,
                                        const RuntimeFormat& format, uint32_t component) {
	auto& state = ctx.state;
	// Byte-aligned layouts: components of 1, 2 or 4 bytes, one after another.
	const auto aligned_offset = Binary(state, OpShiftLeftLogical, TypeU32(state), component, format.log_bytes);
	const auto aligned_bits   = Binary(state, OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 8), format.log_bytes);
	// Packed layouts: bit fields of one dword, a byte per component in the layout's table.
	const auto by_layout = [&](uint32_t table) {
		return EmitAndConstant(state,
		                       Binary(state, OpShiftRightLogical, TypeU32(state), table,
		                              Binary(state, OpShiftLeftLogical, TypeU32(state), component,
		                                     ConstantU32(state, 3))),
		                       0xffu);
	};
	const auto packed_shift = by_layout(format.packed_shift);
	const auto packed_bits  = by_layout(format.packed_bits);

	RuntimeComponent located;
	located.address = BufferAddressPlus(ctx, inst, mem,
	                                    Select(state, TypeU32(state), format.packed, ConstantU32(state, 0),
	                                           aligned_offset));
	located.index   = EmitMemoryElementIndex(
	    state, resource,
	    Binary(state, OpShiftRightLogical, TypeU32(state), located.address, ConstantU32(state, 2)));
	// A dword component is the whole dword; a smaller one sits at its byte in it.
	const auto aligned_shift = Select(
	    state, TypeU32(state), EmitCompareU32Constant(state, OpIEqual, aligned_bits, 32),
	    ConstantU32(state, 0),
	    Binary(state, OpShiftLeftLogical, TypeU32(state), EmitAndConstant(state, located.address, 3),
	           ConstantU32(state, 3)));
	located.shift = Select(state, TypeU32(state), format.packed, packed_shift, aligned_shift);
	located.bits  = Select(state, TypeU32(state), format.packed, packed_bits, aligned_bits);
	return located;
}

// The component's value in its dword, converted as its format says (NormalizeFormatComponent). Every
// conversion is computed and the format's picked: branches per component made the portable module of a
// shader with hundreds of formatted loads tens of thousands of blocks (minutes to compile), and only a
// first encounter runs it (the next start compiles the specialized module).
uint32_t DecodeRuntimeComponent(EmitterState& state, const RuntimeFormat& format,
                                const RuntimeComponent& located, uint32_t dword) {
	using Format::ComponentType;
	// Its bits, zero- or sign-extended.
	const auto unused  = Binary(state, OpISub, TypeU32(state), ConstantU32(state, 32), located.bits);
	const auto shifted = Binary(state, OpShiftLeftLogical, TypeU32(state),
	                            Binary(state, OpShiftRightLogical, TypeU32(state), dword, located.shift), unused);
	const auto sign_extended = Unary(
	    state, OpBitcast, TypeU32(state),
	    Binary(state, OpShiftRightArithmetic, TypeI32(state), Unary(state, OpBitcast, TypeI32(state), shifted),
	           unused));
	const auto zero_extended = Binary(state, OpShiftRightLogical, TypeU32(state), shifted, unused);
	const auto raw = Select(state, TypeU32(state), format.is_signed, sign_extended, zero_extended);
	const auto max_value = [&](uint32_t all_ones) {
		return Unary(state, OpConvertUToF, TypeF32(state),
		             Binary(state, OpShiftRightLogical, TypeU32(state), ConstantU32(state, all_ones), unused));
	};
	const auto float_bits = [&](uint32_t value) { return EmitBitcastF32ToU32(state, value); };
	const auto unorm      = float_bits(Binary(state, OpFDiv, TypeF32(state),
	                                          Unary(state, OpConvertUToF, TypeF32(state), raw), max_value(0xffffffffu)));
	const auto normalized =
	    Binary(state, OpFDiv, TypeF32(state),
	           Unary(state, OpConvertSToF, TypeF32(state), EmitTBufferBitcastU32ToI32(state, raw)), max_value(0x7fffffffu));
	const auto clamped = state.builder.AllocateId();
	state.builder.AddFunction({OpExtInst, TypeF32(state), clamped, GlslStd450(state), GlslFMax, normalized,
	                           ConstantF32Value(state, -1.0f)});
	const auto snorm   = float_bits(clamped);
	const auto uscaled = float_bits(Unary(state, OpConvertUToF, TypeF32(state), raw));
	const auto sscaled = float_bits(Unary(state, OpConvertSToF, TypeF32(state), EmitTBufferBitcastU32ToI32(state, raw)));
	const auto pick    = [&](uint32_t condition, uint32_t value, uint32_t otherwise) {
		return Select(state, TypeU32(state), condition, value, otherwise);
	};
	const auto bits_are = [&](uint32_t bits) { return EmitCompareU32Constant(state, OpIEqual, located.bits, bits); };
	auto       floating = pick(bits_are(10u), EmitUFloatToF32Bits(state, raw, 10), raw);
	floating            = pick(bits_are(11u), EmitUFloatToF32Bits(state, raw, 11), floating);
	floating            = pick(bits_are(16u), EmitBitcastF32ToU32(state, EmitF16BitsToF32(state, raw)), floating);
	const auto type_is  = [&](ComponentType type) {
		return EmitCompareU32Constant(state, OpIEqual, format.type, uint32_t(type));
	};
	auto value = pick(type_is(ComponentType::Float), floating, raw);
	value      = pick(type_is(ComponentType::Sscaled), sscaled, value);
	value      = pick(type_is(ComponentType::Uscaled), uscaled, value);
	value      = pick(type_is(ComponentType::Snorm), snorm, value);
	return pick(type_is(ComponentType::Unorm), unorm, value);
}

// The value of an output component: its selector's constant, or its decoded memory component.
uint32_t SelectRuntimeOutput(EmitterState& state, const RuntimeFormat& format, uint32_t selector,
                             uint32_t decoded) {
	const auto memory = EmitCompareU32Constant(state, OpUGreaterThanEqual, selector, 4);
	const auto one    = EmitCompareU32Constant(state, OpIEqual, selector, 1);
	return Select(state, TypeU32(state), memory, decoded,
	              Select(state, TypeU32(state), one, format.one, ConstantU32(state, 0)));
}

// A single formatted load (FormattedLoadPrepared): each component's dword checked on its own, zero
// when out of bounds.
uint32_t RuntimeFormattedLoadPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                                      uint32_t output_component, const MemoryResourceAccess& resource) {
	auto&      state  = ctx.state;
	const auto format = LoadRuntimeFormat(state, mem);
	const auto load   = [&](uint32_t index) {
		return EmitValueOrZeroIfCondition(state, EmitMemoryElementInBounds(state, resource, index),
		                                  [&]() { return LoadWordInBounds(ctx, resource, index); });
	};
	return EmitSwitchValue(
	    state, TypeU32(state), format.type,
	    {{uint32_t(Format::ComponentType::Unknown),
	      [&] {
		      // No format: raw dwords, dst_sel ignored.
		      return LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, output_component), resource);
	      }}},
	    [&] {
		    const auto selector = RuntimeSelector(state, format, output_component);
		    const auto located  = LocateRuntimeComponent(ctx, inst, mem, resource, format,
		                                                 RuntimeSourceComponent(state, format, selector));
		    const auto decoded  = DecodeRuntimeComponent(state, format, located, load(located.index));
		    return SelectRuntimeOutput(state, format, selector, decoded);
	    });
}

uint32_t FormattedLoadPrepared(ValueEmitContext& ctx, const IR::Inst& inst,
                               const IR::MemoryInfo& mem, uint32_t output_component,
                               const MemoryResourceAccess& resource) {
	if (UsesRuntimeFormat(ctx.state, mem)) {
		return RuntimeFormattedLoadPrepared(ctx, inst, mem, output_component, resource);
	}
	const auto info = Format::GetFormatInfo(BufferFormat(ctx, mem));
	if (info.type == Format::ComponentType::Unknown) {
		return LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, output_component), resource);
	}
	return LoadFormattedComponent(
	    ctx, mem, info, output_component,
	    [&](uint32_t component) {
		    return LoadWordPrepared(ctx, inst, RebaseFormattedComponent(mem, info, component),
		                            resource);
	    },
	    [&](uint32_t component, uint32_t bits, bool sign_extend) {
		    return LoadSubwordPrepared(ctx, inst, RebaseFormattedComponent(mem, info, component),
		                               resource, bits, sign_extend);
	    });
}

uint32_t FormattedLoad(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	return EmitValueOrZeroIfCondition(ctx.state, LoadGuard(ctx, inst, mem.kind), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return FormattedLoadPrepared(ctx, inst, mem, 0u, resource);
	});
}

void StoreSubwordInBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource, uint32_t address, uint32_t index,
                          uint32_t bits, uint32_t data) {
	if (resource.bda_base != 0 && !std::exchange(ctx.state.table_guarded, true)) {
		// Table mode: out of bounds drops the store (LoadWordInBounds).
		EmitIfCondition(ctx.state, EmitMemoryElementInBoundsExplicit(ctx.state, resource, index),
		                [&]() { StoreSubwordInBounds(ctx, mem, resource, address, index, bits, data); });
		ctx.state.table_guarded = false;
		return;
	}
	const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
	const auto shift   = Binary(
	    ctx.state, OpShiftLeftLogical, TypeU32(ctx.state),
	    Binary(ctx.state, OpBitwiseAnd, TypeU32(ctx.state), address, ConstantU32(ctx.state, 3)),
	    ConstantU32(ctx.state, 3));
	const auto mask  = Binary(ctx.state, OpShiftLeftLogical, TypeU32(ctx.state),
	                          ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu), shift);
	const auto value = Binary(ctx.state, OpShiftLeftLogical, TypeU32(ctx.state),
	                          Binary(ctx.state, OpBitwiseAnd, TypeU32(ctx.state), data,
	                                 ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu)),
	                          shift);
	const auto merge = [&](uint32_t old) {
		return Binary(ctx.state, OpBitwiseOr, TypeU32(ctx.state),
		              Binary(ctx.state, OpBitwiseAnd, TypeU32(ctx.state), old,
		                     Unary(ctx.state, OpNot, TypeU32(ctx.state), mask)),
		              value);
	};
	if (mem.kind == IR::ResourceKind::Scratch) {
		const auto old = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction({OpLoad, TypeU32(ctx.state), old, pointer});
		ctx.state.builder.AddFunction({OpStore, pointer, merge(old)});
	} else {
		AtomicUpdate(ctx.state, pointer, mem.kind, merge);
	}
}

void StoreSubwordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource, uint32_t bits, uint32_t data) {
	const auto address   = ByteAddress(ctx, inst, mem);
	const auto raw_index = Binary(ctx.state, OpShiftRightLogical, TypeU32(ctx.state), address,
	                              ConstantU32(ctx.state, 2));
	const auto index     = EmitMemoryElementIndex(ctx.state, resource, raw_index);
	EmitIfCondition(
	    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index), [&]() {
		    StoreSubwordInBounds(ctx, mem, resource, address, index, bits, data);
	    });
}

void StoreSubword(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem, uint32_t bits) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		StoreSubwordPrepared(ctx, inst, mem, resource, bits, ctx.Arg(inst, inst.NumArgs() - 2));
	});
}

void StoreWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource, uint32_t index,
                       uint32_t data) {
	if (resource.bda_base != 0) { // table mode: out of bounds drops the store (LoadWordInBounds)
		EmitIfCondition(ctx.state, EmitMemoryElementInBoundsExplicit(ctx.state, resource, index), [&]() {
			ctx.state.builder.AddFunction({OpStore, EmitMemoryElementPointer(ctx.state, resource, index), data,
			                               MemoryAccessAlignedMask, sizeof(uint32_t)});
		});
		return;
	}
	const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
	ctx.state.builder.AddFunction({OpStore, pointer, data});
}

void StoreWordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                       const MemoryResourceAccess& resource, uint32_t data) {
	const auto index = EmitMemoryElementIndex(ctx.state, resource, DwordIndex(ctx, inst, mem));
	EmitIfCondition(ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index),
	                [&]() { StoreWordInBounds(ctx, resource, index, data); });
}

void StoreWord(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		StoreWordPrepared(ctx, inst, mem, resource, ctx.Arg(inst, inst.NumArgs() - 2));
	});
}

void FormattedStorePrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                            uint32_t component, const MemoryResourceAccess& resource,
                            uint32_t data) {
	const auto info = Format::GetFormatInfo(BufferFormat(ctx, mem));
	if (info.type == Format::ComponentType::Unknown) {
		StoreWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource, data);
		return;
	}
	if (component >= info.component_count) return;
	const auto bits          = info.component_bits[component];
	const auto component_mem = RebaseFormattedComponent(mem, info, component);
	const auto packed        = PackFormatComponent(ctx.state, info, component, data);
	if (info.packed_bitfield) {
		StoreWordPrepared(ctx, inst, component_mem, resource,
		                  Binary(ctx.state, OpShiftLeftLogical, TypeU32(ctx.state), packed,
		                         ConstantU32(ctx.state, info.component_bit_offset[component])));
	} else if (bits == 8u || bits == 16u) {
		StoreSubwordPrepared(ctx, inst, component_mem, resource, bits, packed);
	} else {
		StoreWordPrepared(ctx, inst, component_mem, resource, packed);
	}
}

void FormattedStore(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		FormattedStorePrepared(ctx, inst, mem, 0u, resource, ctx.Arg(inst, inst.NumArgs() - 2));
	});
}

uint32_t SpirvAtomicOpcode(IR::ValueOpcode opcode) {
	switch (opcode) {
		case IR::ValueOpcode::BufferAtomicCmpSwap32: return OpAtomicCompareExchange;
		case IR::ValueOpcode::BufferAtomicSwap32:
		case IR::ValueOpcode::BufferAtomicSwap64:
		case IR::ValueOpcode::SharedAtomicSwap32: return OpAtomicExchange;
		case IR::ValueOpcode::BufferAtomicIAdd32:
		case IR::ValueOpcode::SharedAtomicIAdd32: return OpAtomicIAdd;
		case IR::ValueOpcode::BufferAtomicISub32:
		case IR::ValueOpcode::SharedAtomicISub32: return OpAtomicISub;
		case IR::ValueOpcode::BufferAtomicSMin32:
		case IR::ValueOpcode::SharedAtomicSMin32: return OpAtomicSMin;
		case IR::ValueOpcode::BufferAtomicUMin32:
		case IR::ValueOpcode::SharedAtomicUMin32: return OpAtomicUMin;
		case IR::ValueOpcode::BufferAtomicSMax32:
		case IR::ValueOpcode::SharedAtomicSMax32: return OpAtomicSMax;
		case IR::ValueOpcode::BufferAtomicUMax32:
		case IR::ValueOpcode::SharedAtomicUMax32: return OpAtomicUMax;
		case IR::ValueOpcode::BufferAtomicAnd32:
		case IR::ValueOpcode::SharedAtomicAnd32: return OpAtomicAnd;
		case IR::ValueOpcode::BufferAtomicOr32:
		case IR::ValueOpcode::BufferAtomicOr64:
		case IR::ValueOpcode::SharedAtomicOr32: return OpAtomicOr;
		case IR::ValueOpcode::BufferAtomicXor32:
		case IR::ValueOpcode::SharedAtomicXor32: return OpAtomicXor;
		default: return 0;
	}
}

uint32_t EmitAtomicOperation(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t pointer,
                             uint32_t scope) {
	const auto old = ctx.state.builder.AllocateId();
	if (inst.GetOpcode() == IR::ValueOpcode::BufferAtomicCmpSwap32) {
		const auto desired    = ctx.Arg(inst, inst.NumArgs() - 3);
		const auto comparator = ctx.Arg(inst, inst.NumArgs() - 2);
		ctx.state.builder.AddFunction(
		    {OpAtomicCompareExchange, TypeU32(ctx.state), old, pointer, ConstantU32(ctx.state, scope),
		     ConstantU32(ctx.state, MemorySemanticsNone),
		     ConstantU32(ctx.state, MemorySemanticsNone), desired, comparator});
	} else {
		const auto value = ctx.Arg(inst, inst.NumArgs() - 2);
		ctx.state.builder.AddFunction(
		    {SpirvAtomicOpcode(inst.GetOpcode()), TypeU32(ctx.state), old, pointer,
		     ConstantU32(ctx.state, scope), ConstantU32(ctx.state, MemorySemanticsNone), value});
	}
	return old;
}

template <typename Fn>
uint32_t EmitAtomicAccess(ValueEmitContext& ctx, const IR::Inst& inst,
                          const IR::MemoryInfo& mem, Fn&& operation) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto access = PrepareMemoryElement(ctx, mem, DwordIndex(ctx, inst, mem));
		return EmitValueOrZeroIfCondition(
		    ctx.state, EmitMemoryElementInBoundsExplicit(ctx.state, access.resource, access.index), [&]() {
			    return operation(EmitMemoryElementPointer(ctx.state, access.resource, access.index));
		    });
	});
}

uint32_t EmitAtomic(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	return EmitAtomicAccess(ctx, inst, mem, [&](uint32_t pointer) {
		const auto scope = mem.kind == IR::ResourceKind::Lds ? ScopeWorkgroup : ScopeDevice;
		const auto old   = EmitAtomicOperation(ctx, inst, pointer, scope);
		if (mem.kind == IR::ResourceKind::Lds) {
			const auto semantics = MemorySemanticsAcquireRelease | MemorySemanticsWorkgroupMemory;
			ctx.state.builder.AddFunction({OpMemoryBarrier, ConstantU32(ctx.state, scope),
			                               ConstantU32(ctx.state, semantics)});
		} else {
			EmitDeviceAtomicMemoryBarrier(ctx.state);
		}
		return old;
	});
}

template <typename Fn>
uint32_t EmitAtomicUpdate(ValueEmitContext& ctx, const IR::Inst& inst,
                          const IR::MemoryInfo& mem, Fn&& replacement) {
	const auto value = ctx.Arg(inst, inst.NumArgs() - 2);
	return EmitAtomicAccess(ctx, inst, mem, [&](uint32_t pointer) {
		return AtomicUpdate(ctx.state, pointer, mem.kind, [&](uint32_t old) {
			return replacement(ctx.state, old, value);
		});
	});
}

uint32_t AtomicIncrement(EmitterState& state, uint32_t old, uint32_t limit) {
	// old >= limit ? 0 : old + 1 (unsigned).
	const auto wrap = Binary(state, OpUGreaterThanEqual, TypeBool(state), old, limit);
	const auto next = Binary(state, OpIAdd, TypeU32(state), old, ConstantU32(state, 1));
	return Select(state, TypeU32(state), wrap, ConstantU32(state, 0), next);
}

uint32_t AtomicDecrement(EmitterState& state, uint32_t old, uint32_t limit) {
	// old == 0 || old > limit ? limit : old - 1 (unsigned).
	const auto zero  = Binary(state, OpIEqual, TypeBool(state), old, ConstantU32(state, 0));
	const auto above = Binary(state, OpUGreaterThan, TypeBool(state), old, limit);
	const auto wrap  = Binary(state, OpLogicalOr, TypeBool(state), zero, above);
	const auto next  = Binary(state, OpISub, TypeU32(state), old, ConstantU32(state, 1));
	return Select(state, TypeU32(state), wrap, limit, next);
}

uint32_t EmitBufferAtomic64(ValueEmitContext& ctx, const IR::Inst& inst,
                            const IR::MemoryInfo& mem) {
	auto& state = ctx.state;
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU64(state), ConstantU64(state, 0), [&]() {
		    const auto resource = PrepareStorageBufferResourceAccess(
		        state, mem, state.storage_buffer_u64_variable, TypeStorageBufferU64Pointer(state));
		    const auto byte_address = Binary(state, OpIAdd, TypeU32(state),
		                                     ByteAddress(ctx, inst, mem, false), resource.byte_offset);
		    const auto index = Binary(state, OpShiftRightLogical, TypeU32(state), byte_address,
		                              ConstantU32(state, 3u));
		    return EmitValueOrDefaultIfCondition(
		        state, EmitMemoryElementInBoundsExplicit(state, resource, index), TypeU64(state),
		        ConstantU64(state, 0), [&]() {
			        const auto value = Unary(state, OpBitcast, TypeScalarU64(state),
			                                 ctx.Arg(inst, inst.NumArgs() - 2));
			        auto       pointer = 0u;
			        if (resource.bda_base != 0) {
				        // Table mode: the qword at the absolute address rounded down, as the bound range's byte offset
				        // makes it (a base 4 mod 8 would misalign it; buffers start at pages, so it stays inside).
				        pointer = state.builder.AllocateId();
				        state.builder.AddFunction(
				            {OpConvertUToPtr, TypePhysicalU64Pointer(state), pointer,
				             Binary(state, OpBitwiseAnd, TypeDeviceAddress(state),
				                    Binary(state, OpIAdd, TypeDeviceAddress(state), resource.bda_base,
				                           Unary(state, OpUConvert, TypeDeviceAddress(state), byte_address)),
				                    ConstantDeviceAddress(state, ~uint64_t {7}))});
			        } else {
				        pointer = EmitStorageBufferElementPointer(state, resource, index,
				                                                  TypeStorageBufferU64ElementPointer(state));
			        }
			        const auto old = state.builder.AllocateId();
			        state.builder.AddFunction(
			            {SpirvAtomicOpcode(inst.GetOpcode()), TypeScalarU64(state), old, pointer,
			             ConstantU32(state, ScopeDevice),
			             ConstantU32(state, MemorySemanticsNone), value});
			        EmitDeviceAtomicMemoryBarrier(state);
			        return Unary(state, OpBitcast, TypeU64(state), old);
		        });
	    });
}

uint32_t FloatAtomic(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                     bool max_value) {
	return EmitAtomicUpdate(ctx, inst, mem, [max_value](EmitterState& state, uint32_t old,
	                                                  uint32_t value) {
		return EmitFloatAtomicReplacement(state, old, value, max_value);
	});
}

uint32_t SharedFloatAtomic(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                           bool max_value) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto access = PrepareMemoryElement(ctx, mem, DwordIndex(ctx, inst, mem));
		EmitIfCondition(
		    ctx.state, EmitMemoryElementInBoundsExplicit(ctx.state, access.resource, access.index), [&]() {
			    ctx.state.builder.AddFunction(
			        {OpStore, ctx.scratch_u32_variable, ctx.Arg(inst, 1)});
			    const auto data = ctx.state.builder.AllocateId();
			    ctx.state.builder.AddFunction(
			        {OpLoad, TypeU32(ctx.state), data, ctx.scratch_u32_variable});
			    AtomicUpdate(
			        ctx.state, EmitMemoryElementPointer(ctx.state, access.resource, access.index),
			        mem.kind, [&](uint32_t old) {
				        const auto old_f = Unary(ctx.state, OpBitcast, TypeF32(ctx.state), old);
				        const auto compare_f =
				            Unary(ctx.state, OpBitcast, TypeF32(ctx.state), ctx.Arg(inst, 2));
				        const auto data_f = Unary(ctx.state, OpBitcast, TypeF32(ctx.state), data);
				        const auto compare =
				            Binary(ctx.state, max_value ? OpFOrdGreaterThan : OpFOrdLessThan,
				                   TypeBool(ctx.state), max_value ? old_f : compare_f,
				                   max_value ? compare_f : old_f);
				        return Unary(ctx.state, OpBitcast, TypeU32(ctx.state),
				                     Select(ctx.state, TypeF32(ctx.state), compare, data_f, old_f));
			        });
		    });
	});
	return 0;
}

uint32_t AppendConsume(ValueEmitContext& ctx, const IR::Inst& inst, bool append) {
	auto&      state = ctx.state;
	if (ctx.half == 1) {
		return ctx.other_half->Def(IR::Value(const_cast<IR::Inst*>(&inst)));
	}
	const auto m0    = ctx.Arg(inst, 0);
	const auto base =
	    Binary(state, OpShiftRightLogical, TypeU32(state), m0, ConstantU32(state, 16));
	const auto size = Binary(state, OpBitwiseAnd, TypeU32(state), m0, ConstantU32(state, 0xffffu));
	const auto address =
	    Binary(state, OpIAdd, TypeU32(state), base, ConstantU32(state, ctx.Memory(inst).offset));
	const auto raw_index =
	    Binary(state, OpShiftRightLogical, TypeU32(state), address, ConstantU32(state, 2));
	const auto mem    = ctx.Memory(inst);
	const auto access = PrepareMemoryResourceAccess(state, mem);
	const auto index  = EmitMemoryElementIndex(state, access, raw_index);
	const auto exec   = ctx.Arg(inst, 1);
	const auto ballot = ctx.Ballot(inst.Arg(1));
	const auto low  = state.builder.AllocateId();
	const auto high = state.builder.AllocateId();
	state.builder.AddFunction({OpCompositeExtract, TypeU32(state), low, ballot, 0});
	state.builder.AddFunction({OpCompositeExtract, TypeU32(state), high, ballot, 1});
	const auto count =
	    Binary(state, OpIAdd, TypeU32(state), Unary(state, OpBitCount, TypeU32(state), low),
	           Unary(state, OpBitCount, TypeU32(state), high));
	const auto first       = ctx.FirstLane(ballot);
	const auto source_lane = state.lane_count == 2 ? Binary(state, OpBitwiseAnd, TypeU32(state),
	                                                        first, ConstantU32(state, 31))
	                                               : first;
	const auto is_first =
	    Binary(state, OpIEqual, TypeBool(state), EmitSubgroupLocalInvocationId(state), source_lane);
	const auto storage_bounds = EmitMemoryElementInBoundsExplicit(state, access, index);
	const auto m0_bounds =
	    mem.kind == IR::ResourceKind::Gds
	        ? Binary(state, OpINotEqual, TypeBool(state), size, ConstantU32(state, 0))
	        : Binary(state, OpULessThan, TypeBool(state),
	                 ConstantU32(state, ctx.Memory(inst).offset + 3u), size);
	const auto condition = AndCondition(
	    state, is_first,
	    AndCondition(state,
	                 state.lane_count == 2
	                     ? Binary(state, OpINotEqual, TypeBool(state), count, ConstantU32(state, 0))
	                     : exec,
	                 AndCondition(state, storage_bounds, m0_bounds)));
	const auto atomic = EmitValueOrZeroIfCondition(state, condition, [&]() {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(
		    {append ? OpAtomicIAdd : OpAtomicISub, TypeU32(state), value,
		     EmitMemoryElementPointer(state, access, index),
		     ConstantU32(state, mem.kind == IR::ResourceKind::Gds ? ScopeDevice : ScopeWorkgroup),
		     ConstantU32(state, MemorySemanticsNone), count});
		return value;
	});
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction({OpGroupNonUniformShuffle, TypeU32(state), result,
	                           ConstantU32(state, ScopeSubgroup), atomic, source_lane});
	return result;
}

struct PreparedFormattedMemory {
	Format::BufferFormatInfo info;
	MemoryResourceAccess     resource;
	std::array<uint32_t, 4>  addresses {};
	std::array<uint32_t, 4>  indices {};
	uint32_t                 in_bounds = 0;
};

enum class FormattedAccess { Load, Store };

PreparedFormattedMemory PrepareFormattedMemory(ValueEmitContext& ctx, const IR::Inst& inst,
                                               const IR::MemoryInfo&       mem,
                                               const MemoryResourceAccess& resource,
                                               const Format::BufferFormatInfo& info,
                                               uint32_t components, FormattedAccess access) {
	PreparedFormattedMemory plan;
	plan.info     = info;
	plan.resource = resource;
	std::array<bool, 4> required_components {};
	if (access == FormattedAccess::Load) {
		for (uint32_t output = 0; output < components; output++) {
			const auto source = ResolveFormattedSource(ctx, mem, plan.info, output);
			if (source.kind == FormattedSourceKind::Memory) {
				required_components[source.component] = true;
			}
		}
	} else {
		for (uint32_t component = 0; component < std::min(components, plan.info.component_count);
		     component++) {
			required_components[component] = true;
		}
	}
	bool first_bound = true;
	for (uint32_t component = 0; component < plan.info.component_count; component++) {
		if (!required_components[component]) continue;
		const auto byte_offset = Format::GetFormatComponentByteOffset(plan.info, component);
		bool       reused      = false;
		for (uint32_t previous = 0; previous < component; previous++) {
			if (required_components[previous] &&
			    Format::GetFormatComponentByteOffset(plan.info, previous) == byte_offset) {
				plan.addresses[component] = plan.addresses[previous];
				plan.indices[component]   = plan.indices[previous];
				reused                    = true;
				break;
			}
		}
		if (reused) continue;
		const auto component_mem  = RebaseFormattedComponent(mem, plan.info, component);
		plan.addresses[component] = ByteAddress(ctx, inst, component_mem);
		const auto raw_index      = Binary(ctx.state, OpShiftRightLogical, TypeU32(ctx.state),
		                                   plan.addresses[component], ConstantU32(ctx.state, 2));
		plan.indices[component]   = EmitMemoryElementIndex(ctx.state, resource, raw_index);
		const auto component_bound =
		    EmitMemoryElementInBounds(ctx.state, resource, plan.indices[component]);
		if (first_bound) {
			plan.in_bounds = component_bound;
			first_bound    = false;
		} else {
			plan.in_bounds = AndCondition(ctx.state, plan.in_bounds, component_bound);
		}
	}
	if (first_bound) plan.in_bounds = ConstantBool(ctx.state, true);
	return plan;
}

uint32_t LoadFormattedInBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                               const PreparedFormattedMemory& plan, uint32_t output_component) {
	return LoadFormattedComponent(
	    ctx, mem, plan.info, output_component,
	    [&](uint32_t component) {
		    return LoadWordInBounds(ctx, plan.resource, plan.indices[component]);
	    },
	    [&](uint32_t component, uint32_t bits, bool sign_extend) {
		    return LoadSubwordInBounds(ctx, plan.resource, plan.addresses[component],
		                               plan.indices[component], bits, sign_extend);
	    });
}

uint32_t ConstructU32Composite(EmitterState& state, uint32_t components,
                               const std::array<uint32_t, 4>& values) {
	const auto            result = state.builder.AllocateId();
	std::vector<uint32_t> words {OpCompositeConstruct, TypeU32Composite(state, components), result};
	words.insert(words.end(), values.begin(), values.begin() + components);
	state.builder.AddFunction(words);
	return result;
}

uint32_t FormattedOutOfBoundsValue(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                   const PreparedFormattedMemory& plan, uint32_t components) {
	std::array<uint32_t, 4> values {};
	for (uint32_t component = 0; component < components; component++) {
		const auto source = ResolveFormattedSource(ctx, mem, plan.info, component);
		values[component] = FormattedConstant(ctx, plan.info, source.kind);
	}
	return ConstructU32Composite(ctx.state, components, values);
}

void StoreFormattedInBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                            const PreparedFormattedMemory& plan, uint32_t component,
                            uint32_t data) {
	if (component >= plan.info.component_count) return;
	const auto bits   = plan.info.component_bits[component];
	const auto packed = PackFormatComponent(ctx.state, plan.info, component, data);
	if (bits == 8u || bits == 16u) {
		StoreSubwordInBounds(ctx, mem, plan.resource, plan.addresses[component],
		                     plan.indices[component], bits, packed);
	} else {
		StoreWordInBounds(ctx, plan.resource, plan.indices[component], packed);
	}
}

// A wide formatted load with a runtime format (LoadWideBuffer): the element when every component its
// outputs read is in bounds, else each output's selector constant.
uint32_t RuntimeWideFormattedLoad(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                                  const MemoryResourceAccess& resource, uint32_t components) {
	auto&      state     = ctx.state;
	const auto format    = LoadRuntimeFormat(state, mem);
	const auto composite = TypeU32Composite(state, components);
	return EmitSwitchValue(
	    state, composite, format.type,
	    {{uint32_t(Format::ComponentType::Unknown),
	      [&] {
		      std::array<uint32_t, 4> values {};
		      for (uint32_t component = 0; component < components; component++) {
			      values[component] = LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource);
		      }
		      return ConstructU32Composite(state, components, values);
	      }}},
	    [&] {
		    std::array<uint32_t, 4>         selectors {}, memory {}, constants {};
		    std::array<RuntimeComponent, 4> located {};
		    uint32_t                        in_bounds = ConstantBool(state, true);
		    for (uint32_t output = 0; output < components; output++) {
			    selectors[output] = RuntimeSelector(state, format, output);
			    located[output]   = LocateRuntimeComponent(ctx, inst, mem, resource, format,
			                                               RuntimeSourceComponent(state, format, selectors[output]));
			    memory[output]    = EmitCompareU32Constant(state, OpUGreaterThanEqual, selectors[output], 4);
			    in_bounds         = AndCondition(
                    state, in_bounds,
                    EmitLogicalOrBool(state, EmitLogicalNotBool(state, memory[output]),
                                      EmitMemoryElementInBounds(state, resource, located[output].index)));
			    constants[output] = SelectRuntimeOutput(state, format, selectors[output], ConstantU32(state, 0));
		    }
		    return EmitValueOrDefaultIfCondition(
		        state, in_bounds, composite, ConstructU32Composite(state, components, constants), [&]() {
			        std::array<uint32_t, 4> values {};
			        for (uint32_t output = 0; output < components; output++) {
				        const auto dword = EmitValueOrZeroIfCondition(state, memory[output], [&]() {
					        return LoadWordInBounds(ctx, resource, located[output].index);
				        });
				        values[output] = SelectRuntimeOutput(
				            state, format, selectors[output],
				            DecodeRuntimeComponent(state, format, located[output], dword));
			        }
			        return ConstructU32Composite(state, components, values);
		        });
	    });
}

uint32_t LoadWideBufferPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                                const MemoryResourceAccess& resource, uint32_t components) {
	auto& state = ctx.state;
	if (mem.formatted && UsesRuntimeFormat(state, mem)) {
		return RuntimeWideFormattedLoad(ctx, inst, mem, resource, components);
	}
	const auto info = Format::GetFormatInfo(
	    mem.formatted ? BufferFormat(ctx, mem) : Prospero::BufferFormat::kInvalid);
	if (info.type != Format::ComponentType::Unknown) {
		const auto plan = PrepareFormattedMemory(ctx, inst, mem, resource, info, components,
		                                         FormattedAccess::Load);
		return EmitValueOrDefaultIfCondition(
		    state, plan.in_bounds, TypeU32Composite(state, components),
		    FormattedOutOfBoundsValue(ctx, mem, plan, components), [&]() {
			    std::array<uint32_t, 4> values {};
			    for (uint32_t component = 0; component < components; component++) {
				    values[component] = LoadFormattedInBounds(ctx, mem, plan, component);
			    }
			    return ConstructU32Composite(state, components, values);
		    });
	}
	std::array<uint32_t, 4> values {};
	for (uint32_t component = 0; component < components; component++) {
		values[component] = LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource);
	}
	return ConstructU32Composite(state, components, values);
}

uint32_t LoadWideBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto&      state = ctx.state;
	const auto mem   = ctx.Memory(inst);
	return EmitValueOrDefaultIfCondition(
	    state, LoadGuard(ctx, inst, mem.kind), TypeU32Composite(state, components),
	    ConstantU32CompositeZero(state, components), [&]() {
		    const auto resource = PrepareMemoryResourceAccess(state, mem);
		    return LoadWideBufferPrepared(ctx, inst, mem, resource, components);
	    });
}

void StoreWideBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto mem       = ctx.Memory(inst);
		const auto resource  = PrepareMemoryResourceAccess(state, mem);
		const auto composite = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto info = Format::GetFormatInfo(
		    mem.formatted ? BufferFormat(ctx, mem) : Prospero::BufferFormat::kInvalid);
		if (info.type != Format::ComponentType::Unknown) {
			const auto plan = PrepareFormattedMemory(ctx, inst, mem, resource, info, components,
			                                         FormattedAccess::Store);
			EmitIfCondition(state, plan.in_bounds, [&]() {
				const auto count = std::min(components, plan.info.component_count);
				uint32_t   word  = 0;
				for (uint32_t component = 0; component < count; component++) {
					const auto data = state.builder.AllocateId();
					state.builder.AddFunction(
					    {OpCompositeExtract, TypeU32(state), data, composite, component});
					if (!plan.info.packed_bitfield) {
						StoreFormattedInBounds(ctx, mem, plan, component, data);
						continue;
					}
					const auto packed  = PackFormatComponent(state, plan.info, component, data);
					const auto shifted = Binary(
					    state, OpShiftLeftLogical, TypeU32(state), packed,
					    ConstantU32(state, plan.info.component_bit_offset[component]));
					word = component == 0u
					           ? shifted
					           : Binary(state, OpBitwiseOr, TypeU32(state), word, shifted);
				}
				if (plan.info.packed_bitfield) {
					StoreWordInBounds(ctx, plan.resource, plan.indices[0], word);
				}
			});
			return;
		}
		for (uint32_t component = 0; component < components; component++) {
			const auto data = state.builder.AllocateId();
			state.builder.AddFunction(
			    {OpCompositeExtract, TypeU32(state), data, composite, component});
			StoreWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource, data);
		}
	});
}

uint32_t LoadWideShared(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU32Composite(state, components),
	    ConstantU32CompositeZero(state, components), [&]() {
		    const auto              mem      = ctx.Memory(inst);
		    const auto              resource = PrepareMemoryResourceAccess(state, mem);
		    const auto              base     = ByteAddress(ctx, inst, mem);
		    std::array<uint32_t, 4> values {};
		    for (uint32_t component = 0; component < components; component++) {
			    const auto address   = component == 0u ? base
			                                           : Binary(state, OpIAdd, TypeU32(state), base,
			                                                    ConstantU32(state, component * 4u));
			    const auto raw_index = Binary(state, OpShiftRightLogical, TypeU32(state), address,
			                                  ConstantU32(state, 2));
			    const auto index     = EmitMemoryElementIndex(state, resource, raw_index);
			    values[component]    = EmitValueOrZeroIfCondition(
			        state, EmitMemoryElementInBounds(state, resource, index),
			        [&]() { return LoadWordInBounds(ctx, resource, index); });
		    }
		    return ConstructU32Composite(state, components, values);
	    });
}

void StoreWideShared(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto mem      = ctx.Memory(inst);
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto base     = ByteAddress(ctx, inst, mem);
		for (uint32_t component = 0; component < components; component++) {
			const auto address = component == 0u ? base
			                                     : Binary(state, OpIAdd, TypeU32(state), base,
			                                              ConstantU32(state, component * 4u));
			const auto raw_index =
			    Binary(state, OpShiftRightLogical, TypeU32(state), address, ConstantU32(state, 2));
			const auto index = EmitMemoryElementIndex(state, resource, raw_index);
			EmitIfCondition(state, EmitMemoryElementInBounds(state, resource, index),
			                [&]() {
				                StoreWordInBounds(ctx, resource, index, ctx.Arg(inst, component + 1u));
			                });
		}
	});
}

} // namespace

void DefineGetBdaPointer(EmitterState& state) {
	if (!state.program.info.uses_dma) {
		return;
	}
	const auto type            = TypeDeviceAddress(state);
	const auto function_type   = state.builder.Type(OpTypeFunction, {type, type});
	state.bda_pointer_function = state.builder.AllocateId();
	const auto address         = state.builder.AllocateId();
	const auto entry_label     = state.builder.AllocateId();
	state.builder.AddName(state.bda_pointer_function, "get_bda_pointer");
	state.builder.AddFunction(
	    {OpFunction, type, state.bda_pointer_function, FunctionControlNone, function_type});
	state.builder.AddFunction({OpFunctionParameter, type, address});
	EmitLabel(state, entry_label);

	const auto page64 = Binary(
	    state, OpShiftRightLogical, type, address,
	    ConstantDeviceAddress(state, BufferCache::CACHING_PAGEBITS));
	const auto page = Unary(state, OpUConvert, TypeU32(state), page64);
	const auto entry_pointer = state.builder.AllocateId();
	state.builder.AddFunction({OpAccessChain, TypeDeviceAddressStoragePointer(state), entry_pointer,
	                           state.bda_pagetable_variable, ConstantU32(state, 0), page});
	const auto base = state.builder.AllocateId();
	state.builder.AddFunction({OpLoad, type, base, entry_pointer});
	const auto missing =
	    Binary(state, OpIEqual, TypeBool(state), base, ConstantDeviceAddress(state, 0));
	const auto fault_label     = state.builder.AllocateId();
	const auto available_label = state.builder.AllocateId();
	const auto merge_label     = state.builder.AllocateId();
	state.builder.AddFunction({OpSelectionMerge, merge_label, SelectionControlNone});
	state.builder.AddFunction(
	    {OpBranchConditional, missing, fault_label, available_label});

	EmitLabel(state, fault_label);
	RecordBdaFault(state, page);
	state.builder.AddFunction({OpBranch, merge_label});

	EmitLabel(state, available_label);
	const auto offset = Binary(
	    state, OpBitwiseAnd, type, address,
	    ConstantDeviceAddress(state, BufferCache::CACHING_PAGESIZE - 1));
	const auto available = Binary(state, OpIAdd, type, base, offset);
	state.builder.AddFunction({OpBranch, merge_label});

	EmitLabel(state, merge_label);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction({OpPhi, type, result, ConstantDeviceAddress(state, 0), fault_label,
	                           available, available_label});
	state.builder.AddFunction({OpReturnValue, result});
	state.builder.AddFunction({OpFunctionEnd});
}

uint32_t EmitTableBlockLoad(EmitterState& state, uint32_t index) {
	const auto address = TypeDeviceAddress(state);
	const auto offset  = Binary(state, OpShiftLeftLogical, address, Unary(state, OpUConvert, address, index),
	                            ConstantU32(state, 2u));
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction({OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
	                           Binary(state, OpIAdd, address, state.table_block, offset)});
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction({OpLoad, TypeU32(state), value, pointer, MemoryAccessAlignedMask, sizeof(uint32_t)});
	return value;
}

// Table mode, at the function's entry: the slots the shader reads (IR::TablePlan::gpu) from its block (the block's
// device address is two dwords of shader data, IR::BindingLayout::TableBlockDword), and each buffer's device
// address, size in dwords, buffer word and stride (the renderer resolved the V#).
void EmitTableMode(ValueEmitContext& ctx) {
	auto&       state   = ctx.state;
	const auto& program = state.program;
	if (!program.table_mode) return;
	const auto& plan    = program.table_plan;
	const auto  address = TypeDeviceAddress(state);
	const auto  block   = DeviceAddressFromWords(state, EmitShaderDataDwordLoad(state, program.bindings.TableBlockDword()),
	                                             EmitShaderDataDwordLoad(state, program.bindings.TableBlockDword() + 1));
	state.table_block   = block;
	const auto  load    = [&](uint32_t dword) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction({OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
		                           Binary(state, OpIAdd, address, block, ConstantDeviceAddress(state, uint64_t {dword} * 4))});
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction({OpLoad, TypeU32(state), value, pointer, MemoryAccessAlignedMask, sizeof(uint32_t)});
		return value;
	};
	state.table_slot_values.assign(plan.slots.size(), 0);
	for (size_t slot = 0; slot < plan.slots.size(); ++slot)
		if (plan.gpu[slot] != 0) state.table_slot_values[slot] = load(static_cast<uint32_t>(slot));
	// Each buffer as the renderer resolved it (IR::TablePlan::BufferDword).
	for (size_t i = 0; i < plan.buffers.size(); ++i) {
		const auto first       = plan.BufferDword(i);
		state.table_bases[i]   = DeviceAddressFromWords(state, load(first), load(first + 1));
		state.table_lengths[i] = load(first + 2);
		state.table_strides[i] = load(first + 4);
		const auto& resource   = program.info.buffers[i];
		if (IR::RuntimeBufferFormat(resource) && resource.descriptor_format == Prospero::BufferFormat::kInvalid)
			state.table_buffer_words.at(i) = load(first + 3);
	}
	EmitRuntimeFormats(state);
}

bool EmitValueMemory(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state             = ctx.state;
	const auto op                = inst.GetOpcode();
	const auto buffer_components = IR::BufferComponentCount(op);
	if (buffer_components > 1u) {
		const auto access = IR::BufferAccessOf(op);
		if (access == IR::BufferAccess::Read) {
			ctx.Define(inst, LoadWideBuffer(ctx, inst, buffer_components));
			return true;
		}
		if (access == IR::BufferAccess::Write) {
			StoreWideBuffer(ctx, inst, buffer_components);
			return true;
		}
	}
	const auto shared_components = IR::SharedComponentCount(op);
	if (shared_components > 1u) {
		if (IR::SharedAccessOf(op) == IR::SharedAccess::Read) {
			ctx.Define(inst, LoadWideShared(ctx, inst, shared_components));
		} else {
			StoreWideShared(ctx, inst, shared_components);
		}
		return true;
	}
	if ((op == IR::ValueOpcode::LoadAddressU32 || op == IR::ValueOpcode::ReadConstBuffer) &&
	    ctx.Memory(inst).planning_only) {
		return true;
	}
	if (op == IR::ValueOpcode::ReadConst && state.program.table_mode) {
		// A slot the block does not give the shader only addresses planning-only loads (IR::EnterTableMode), which
		// emit nothing.
		const auto slot = inst.Arg(1).Resolve().U32();
		ctx.Define(inst, slot < state.table_slot_values.size() && state.table_slot_values[slot] != 0
		                     ? state.table_slot_values[slot]
		                     : ConstantU32(state, 0));
		return true;
	}
	if (op == IR::ValueOpcode::ReadConst) {
		if (state.flattened_srt_variable == 0) {
			ctx.Fail(inst, "requires the flattened SRT descriptor");
			return true;
		}
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction({OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
		                           state.flattened_srt_variable, ConstantU32(state, 0),
		                           ctx.Arg(inst, 1)});
		ctx.Emit(inst, OpLoad, IR::Type::U32, {pointer});
		return true;
	}
	if (op == IR::ValueOpcode::ReadConstBuffer) {
		auto mem = ctx.Memory(inst);
		mem.kind = IR::ResourceKind::ScalarBuffer;
		const auto address =
		    Binary(state, OpIAdd, TypeU32(state), ctx.Arg(inst, 1), ConstantU32(state, mem.offset));
		const auto index =
		    Binary(state, OpShiftRightLogical, TypeU32(state), address, ConstantU32(state, 2));
		const auto access    = PrepareMemoryResourceAccess(state, mem);
		const auto element   = EmitMemoryElementIndex(state, access, index);
		const auto condition = EmitMemoryElementInBounds(state, access, element);
		ctx.Define(inst, EmitValueOrZeroIfCondition(state, condition,
		                                            [&]() { return LoadWordInBounds(ctx, access, element); }));
		return true;
	}
	const auto address_info = IR::AddressOpcodeInfoOf(op);
	const bool load_address = address_info.access == IR::AddressAccess::Read;
	if (load_address && ctx.Memory(inst).kind != IR::ResourceKind::Scratch) {
		ctx.Define(inst, LoadBda(ctx, inst, ctx.Memory(inst), address_info.data_bits));
		return true;
	}
	const bool load_buffer  = op == IR::ValueOpcode::LoadBufferU8 ||
	                          op == IR::ValueOpcode::LoadBufferU16 ||
	                          op == IR::ValueOpcode::LoadBufferU32;
	const bool load_shared  = IR::SharedAccessOf(op) == IR::SharedAccess::Read;
	if (load_address || load_buffer || load_shared) {
		const auto mem   = ctx.Memory(inst);
		uint32_t   value = 0;
		if (op == IR::ValueOpcode::LoadBufferU32 && mem.formatted)
			value = FormattedLoad(ctx, inst, mem);
		else if (op == IR::ValueOpcode::LoadAddressU8 || op == IR::ValueOpcode::LoadBufferU8 ||
		         op == IR::ValueOpcode::LoadSharedU8)
			value = LoadSubword(ctx, inst, mem, 8, false);
		else if (op == IR::ValueOpcode::LoadAddressU16 || op == IR::ValueOpcode::LoadBufferU16 ||
		         op == IR::ValueOpcode::LoadSharedU16)
			value = LoadSubword(ctx, inst, mem, 16, false);
		else
			value = LoadWord(ctx, inst, mem);
		ctx.Define(inst, value);
		return true;
	}
	const bool store_address = address_info.access == IR::AddressAccess::Write;
	const bool store_buffer  = op == IR::ValueOpcode::StoreBufferU8 ||
	                           op == IR::ValueOpcode::StoreBufferU16 ||
	                           op == IR::ValueOpcode::StoreBufferU32;
	const bool store_shared  = IR::SharedAccessOf(op) == IR::SharedAccess::Write;
	if (store_address || store_buffer || store_shared) {
		const auto mem = ctx.Memory(inst);
		if (op == IR::ValueOpcode::StoreBufferU32 && mem.formatted)
			FormattedStore(ctx, inst, mem);
		else if (op == IR::ValueOpcode::StoreAddressU8 || op == IR::ValueOpcode::StoreBufferU8 ||
		         op == IR::ValueOpcode::WriteSharedU8)
			StoreSubword(ctx, inst, mem, 8);
		else if (op == IR::ValueOpcode::StoreAddressU16 || op == IR::ValueOpcode::StoreBufferU16 ||
		         op == IR::ValueOpcode::WriteSharedU16)
			StoreSubword(ctx, inst, mem, 16);
		else
			StoreWord(ctx, inst, mem);
		return true;
	}
	const auto atomic_opcode = SpirvAtomicOpcode(op);
	if (atomic_opcode != 0 && inst.GetType() == IR::Type::U64) {
		ctx.Define(inst, EmitBufferAtomic64(ctx, inst, ctx.Memory(inst)));
		return true;
	}
	if (atomic_opcode != 0) {
		ctx.Define(inst, EmitAtomic(ctx, inst, ctx.Memory(inst)));
		return true;
	}
	if (op == IR::ValueOpcode::BufferAtomicFMin32 || op == IR::ValueOpcode::BufferAtomicFMax32) {
		ctx.Define(inst, FloatAtomic(ctx, inst, ctx.Memory(inst),
		                             op == IR::ValueOpcode::BufferAtomicFMax32));
		return true;
	}
	if (op == IR::ValueOpcode::SharedAtomicInc32 || op == IR::ValueOpcode::SharedAtomicDec32) {
		const auto replacement = op == IR::ValueOpcode::SharedAtomicInc32
		                             ? AtomicIncrement : AtomicDecrement;
		ctx.Define(inst, EmitAtomicUpdate(ctx, inst, ctx.Memory(inst), replacement));
		return true;
	}
	if (op == IR::ValueOpcode::SharedAtomicFMin32 || op == IR::ValueOpcode::SharedAtomicFMax32) {
		SharedFloatAtomic(ctx, inst, ctx.Memory(inst), op == IR::ValueOpcode::SharedAtomicFMax32);
		return true;
	}
	if (op == IR::ValueOpcode::DataAppend || op == IR::ValueOpcode::DataConsume) {
		ctx.Define(inst, AppendConsume(ctx, inst, op == IR::ValueOpcode::DataAppend));
		return true;
	}
	if (op == IR::ValueOpcode::SwizzleU32 || op == IR::ValueOpcode::BpermuteU32) {
		uint32_t source = ctx.Arg(inst, 0);
		uint32_t target = 0;
		if (op == IR::ValueOpcode::SwizzleU32) {
			state.builder.AddFunction({OpStore, ctx.scratch_u32_variable, source});
			source = state.builder.AllocateId();
			state.builder.AddFunction({OpLoad, TypeU32(state), source, ctx.scratch_u32_variable});
			target = EmitDsSwizzleTargetLane(state, EmitSubgroupLocalInvocationId(state),
			                                 inst.Arg(1).IsImmediate() ? inst.Arg(1).U32() : 0);
		} else {
			const auto index = Binary(
			    state, OpBitwiseAnd, TypeU32(state),
			    Binary(state, OpShiftRightLogical, TypeU32(state), ctx.Arg(inst, 1),
			           ConstantU32(state, 2)),
			    ConstantU32(state, 31));
			const auto base =
			    Binary(state, OpBitwiseAnd, TypeU32(state), EmitSubgroupLocalInvocationId(state),
			           ConstantU32(state, ~31u));
			target = Binary(state, OpBitwiseOr, TypeU32(state), base, index);
		}
		ctx.Define(inst, EmitDsMaskedLaneRead(state, source, target, ctx.Arg(inst, 2)));
		return true;
	}
	return false;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
