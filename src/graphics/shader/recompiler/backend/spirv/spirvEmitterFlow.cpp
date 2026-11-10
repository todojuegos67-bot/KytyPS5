#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

// KYTY_PS_WAVE_OR=0 turns off the reduction below (IsWaveOrScanRead).
static bool PsWaveOrEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PS_WAVE_OR");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// v_readlane 31 or 63 of `x | permlanex16(x)` in a wave64 non-compute program without two-half emulation.
static bool IsWaveOrScanRead(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (!PsWaveOrEnabled() || ctx.other_half != nullptr || ctx.state.program.wave_size != 64u ||
	    ctx.state.stage == ShaderType::Compute)
		return false;
	const auto lane = inst.Arg(1).Resolve();
	if (!lane.IsImmediate() || lane.GetType() != IR::Type::U32 || (lane.U32() != 31u && lane.U32() != 63u)) return false;
	const auto* value = inst.Arg(0).Resolve().TryInstruction();
	if (value != nullptr && value->GetOpcode() == IR::ValueOpcode::SelectU32)
		value = value->Arg(1).Resolve().TryInstruction();
	if (value == nullptr || value->GetOpcode() != IR::ValueOpcode::BitwiseOr32) return false;
	for (size_t i = 0; i < 2; i++) {
		const auto* arg = value->Arg(i).Resolve().TryInstruction();
		if (arg != nullptr && arg->GetOpcode() == IR::ValueOpcode::SelectU32) // a VGPR write under exec
			arg = arg->Arg(1).Resolve().TryInstruction();
		if (arg != nullptr && arg->GetOpcode() == IR::ValueOpcode::Permlane16U32 && arg->Flags<IR::PermlaneFlags>().x16)
			return true;
	}
	return false;
}
// Cooperative halves (IR::WaveFlags, Program::lane_cooperative): the wave's answer from both 32-lane halves, the
// workgroup's two subgroups, through workgroup memory. Every lane stores its word in its slot of the current phase,
// a barrier, then each reads the slots it needs. The phase alternates (two of 128 words), so an exchange's stores
// never overwrite words of the one before that another invocation has still to read: a barrier lies between.
static bool CooperativeWave(const ValueEmitContext& ctx, const IR::Inst& inst) {
	return ctx.state.lane_count == 1 && ctx.state.program.lane_cooperative && inst.Flags<IR::WaveFlags>().cooperative;
}

struct WaveExchange {
	uint32_t base = 0; // the phase's first slot
	uint32_t lane = 0; // the invocation's GCN lane: its slot
};

static WaveExchange BeginWaveExchange(EmitterState& state) {
	if (state.wave_exchange_variable == 0) {
		state.wave_exchange_variable = state.builder.DefineGlobalVariable(
		    TypeU32ArrayPointer(state, StorageClassWorkgroup, 256), StorageClassWorkgroup);
		state.builder.AddName(state.wave_exchange_variable, "wave_exchange");
		state.wave_phase_variable = state.builder.DefineInitializedVariable(
		    TypePointer(state, StorageClassPrivate, TypeU32(state)), StorageClassPrivate, ConstantU32(state, 0));
		state.builder.AddName(state.wave_phase_variable, "wave_phase");
	}
	const auto phase = state.builder.AllocateId();
	state.builder.AddFunction({OpLoad, TypeU32(state), phase, state.wave_phase_variable});
	state.builder.AddFunction(
	    {OpStore, state.wave_phase_variable, EmitBinaryU32(state, OpBitwiseXor, phase, ConstantU32(state, 1))});
	return {EmitBinaryU32(state, OpShiftLeftLogical, phase, ConstantU32(state, 7)),
	        EmitBinaryU32(state, OpBitwiseAnd, EmitLocalInvocationIndex(state), ConstantU32(state, 63))};
}

static uint32_t WaveSlot(EmitterState& state, uint32_t base, uint32_t offset) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction({OpAccessChain, TypePointer(state, StorageClassWorkgroup, TypeU32(state)), pointer,
	                           state.wave_exchange_variable, EmitAddU32(state, base, offset)});
	return pointer;
}

static void StoreWaveSlot(EmitterState& state, uint32_t base, uint32_t offset, uint32_t value) {
	state.builder.AddFunction({OpStore, WaveSlot(state, base, offset), value});
}

static uint32_t LoadWaveSlot(EmitterState& state, uint32_t base, uint32_t offset) {
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction({OpLoad, TypeU32(state), value, WaveSlot(state, base, offset)});
	return value;
}

static void WaveBarrier(EmitterState& state) {
	state.builder.AddFunction({OpControlBarrier, ConstantU32(state, ScopeWorkgroup), ConstantU32(state, ScopeWorkgroup),
	                           ConstantU32(state, MemorySemanticsAcquireRelease | MemorySemanticsWorkgroupMemory)});
}

namespace {

bool UserDataDwordIndex(const EmitterState& state, IR::ScalarReg reg, uint32_t& dword_index) {
	const auto register_index = IR::RegIndex(reg);
	const auto& registers = state.program.bindings.user_data_registers;
	const auto  found     = std::lower_bound(registers.begin(), registers.end(), register_index);
	if (found == registers.end() || *found != register_index) {
		return false;
	}
	dword_index = static_cast<uint32_t>(found - registers.begin());
	return true;
}

uint32_t EmitWqmU64(EmitterState& state, uint32_t value) {
	const auto shifted_one = state.builder.AllocateId();
	const auto merged_one  = state.builder.AllocateId();
	const auto shifted_two = state.builder.AllocateId();
	const auto merged_two  = state.builder.AllocateId();
	const auto quad_bits   = state.builder.AllocateId();
	const auto result      = state.builder.AllocateId();
	state.builder.AddFunction({OpShiftRightLogical, TypeU64(state), shifted_one, value,
	                           ConstantU64(state, 0x0000000100000001ull)});
	state.builder.AddFunction({OpBitwiseOr, TypeU64(state), merged_one, value, shifted_one});
	state.builder.AddFunction({OpShiftRightLogical, TypeU64(state), shifted_two, merged_one,
	                           ConstantU64(state, 0x0000000200000002ull)});
	state.builder.AddFunction({OpBitwiseOr, TypeU64(state), merged_two, merged_one, shifted_two});
	state.builder.AddFunction({OpBitwiseAnd, TypeU64(state), quad_bits, merged_two,
	                           ConstantU64(state, 0x1111111111111111ull)});
	state.builder.AddFunction(
	    {OpIMul, TypeU64(state), result, quad_bits, ConstantU64(state, 0x0000000f0000000full)});
	return result;
}

uint32_t EmitBuiltinU32(ValueEmitContext& ctx, IR::StageInputKind kind, uint32_t component) {
	auto& state = ctx.state;
	if (kind == IR::StageInputKind::LocalInvocationIndex) {
		return EmitLocalInvocationIndex(state);
	}
	if (state.lane_count == 2 && (kind == IR::StageInputKind::LocalInvocationId ||
	                              kind == IR::StageInputKind::GlobalInvocationId)) {
		const auto* cs      = ShaderWorkgroupInput(state.stage, state.input_info);
		uint32_t    divisor = 1;
		for (uint32_t axis = 0; axis < component; axis++) {
			divisor *= std::max(cs->threads_num[axis], 1u);
		}
		const auto size    = std::max(cs->threads_num[component], 1u);
		const auto divided = EmitBinaryU32(state, OpUDiv, EmitLocalInvocationIndex(state),
		                                   ConstantU32(state, divisor));
		const auto local   = EmitBinaryU32(state, OpUMod, divided, ConstantU32(state, size));
		if (kind == IR::StageInputKind::LocalInvocationId) {
			return local;
		}
		const auto group = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, component);
		return EmitAddU32(state, local,
		                  EmitBinaryU32(state, OpIMul, group, ConstantU32(state, size)));
	}
	const auto variable = InputVariableForKind(state, kind);
	if (variable == 0) {
		return ConstantU32(state, 0);
	}
	if (kind == IR::StageInputKind::FrontFacing) {
		const auto value = state.builder.AllocateId();
		const auto bits  = state.builder.AllocateId();
		state.builder.AddFunction({OpLoad, TypeBool(state), value, variable});
		state.builder.AddFunction(
		    {OpSelect, TypeU32(state), bits, value, ConstantU32(state, 1), ConstantU32(state, 0)});
		return bits;
	}
	if (kind == IR::StageInputKind::VertexIndex || kind == IR::StageInputKind::InstanceIndex ||
	    kind == IR::StageInputKind::BaseInstance || kind == IR::StageInputKind::Layer ||
	    kind == IR::StageInputKind::SampleId) {
		const auto value = state.builder.AllocateId();
		const auto bits  = state.builder.AllocateId();
		state.builder.AddFunction({OpLoad, TypeI32(state), value, variable});
		state.builder.AddFunction({OpBitcast, TypeU32(state), bits, value});
		return bits;
	}
	if (kind == IR::StageInputKind::FragCoord) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		const auto bits    = state.builder.AllocateId();
		state.builder.AddFunction({OpAccessChain,
		                           TypePointer(state, StorageClassInput, TypeF32(state)), pointer,
		                           variable, ConstantU32(state, component)});
		state.builder.AddFunction({OpLoad, TypeF32(state), value, pointer});
		state.builder.AddFunction({OpBitcast, TypeU32(state), bits, value});
		return bits;
	}
	if (kind == IR::StageInputKind::BaryCoordSmooth ||
	    kind == IR::StageInputKind::BaryCoordNoPerspective) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		const auto bits    = state.builder.AllocateId();
		// (I and J: the KHR 3-vector's second and third, the AMD 2-vector's first and second.)
		state.builder.AddFunction({OpAccessChain,
		                           TypePointer(state, StorageClassInput, TypeF32(state)), pointer,
		                           variable, ConstantU32(state, state.amd_barycentrics ? component : component + 1u)});
		state.builder.AddFunction({OpLoad, TypeF32(state), value, pointer});
		state.builder.AddFunction({OpBitcast, TypeU32(state), bits, value});
		return bits;
	}
	return EmitInputComponentU32(state, kind, component);
}

// Whether the GPU has quad operations in pixel shaders (Spirv::SetPixelQuadOperations). Vulkan does not require them:
// without them, quad_perm DPP moves stay shuffles (EmitDppQuadPermPixel below).
static std::atomic<bool> g_pixel_quad_operations {true};

// A quad_perm DPP move in a pixel shader, read by quad operations. Whole quad mode's helper pixels may be inactive for a
// subgroup shuffle, which then returns an undefined value, but Vulkan counts them in quad operations (on Demon's Souls'
// debug menu, PS b29a3e133609034e takes its font's edge width from DPP quad moves: lines along every partly covered
// quad, the diagonal of each glyph). Each source lane the control names is broadcast once, with its exec bit, and every
// lane takes the one its selector picks. (The capability only where it is used, as for PS_WAVE_OR below.)
static uint32_t EmitDppQuadPermPixel(ValueEmitContext& ctx, const IR::Inst& inst, const IR::DppMoveFlags& flags,
                                     uint32_t target_lane) {
	auto& state = ctx.state;
	state.builder.RequireCapability(CapabilityGroupNonUniformQuad);
	const auto scope    = ConstantU32(state, ScopeSubgroup);
	const auto selector = EmitBinaryU32(state, OpBitwiseAnd, target_lane, ConstantU32(state, 3));
	uint32_t   value    = 0;
	uint32_t   active   = 0;
	for (uint32_t source = 0; source < 4; source++) {
		bool used = false;
		for (uint32_t lane = 0; lane < 4; lane++) {
			used = used || ((flags.control >> (2u * lane)) & 3u) == source;
		}
		if (!used) {
			continue;
		}
		const auto index = ConstantU32(state, source);
		const auto read  = state.builder.AllocateId();
		state.builder.AddFunction({OpGroupNonUniformQuadBroadcast, TypeU32(state), read, scope, ctx.Arg(inst, 0), index});
		uint32_t read_active = 0;
		if (!flags.fetch_inactive) {
			read_active = state.builder.AllocateId();
			state.builder.AddFunction(
			    {OpGroupNonUniformQuadBroadcast, TypeBool(state), read_active, scope, ctx.Arg(inst, 1), index});
		}
		if (value == 0) {
			value  = read;
			active = read_active;
			continue;
		}
		const auto picked = state.builder.AllocateId();
		state.builder.AddFunction({OpIEqual, TypeBool(state), picked, selector, index});
		const auto picked_value = state.builder.AllocateId();
		state.builder.AddFunction({OpSelect, TypeU32(state), picked_value, picked, read, value});
		value = picked_value;
		if (!flags.fetch_inactive) {
			const auto picked_active = state.builder.AllocateId();
			state.builder.AddFunction({OpSelect, TypeBool(state), picked_active, picked, read_active, active});
			active = picked_active;
		}
	}
	if (flags.fetch_inactive) {
		return value;
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction({OpSelect, TypeU32(state), result, active, value, ConstantU32(state, 0)});
	return result;
}

uint32_t EmitDppWriteCondition(ValueEmitContext& ctx, const IR::DppMoveFlags& flags,
                               uint32_t exec) {
	auto&      state      = ctx.state;
	const auto lane       = EmitSubgroupLocalInvocationId(state);
	const auto bank_shift = state.builder.AllocateId();
	const auto row_shift  = state.builder.AllocateId();
	const auto bank       = state.builder.AllocateId();
	const auto row        = state.builder.AllocateId();
	const auto bank_bit   = state.builder.AllocateId();
	const auto row_bit    = state.builder.AllocateId();
	const auto bank_hit   = state.builder.AllocateId();
	const auto row_hit    = state.builder.AllocateId();
	const auto bank_ok    = state.builder.AllocateId();
	const auto row_ok     = state.builder.AllocateId();
	const auto masks_ok   = state.builder.AllocateId();
	state.builder.AddFunction(
	    {OpShiftRightLogical, TypeU32(state), bank_shift, lane, ConstantU32(state, 2)});
	state.builder.AddFunction(
	    {OpShiftRightLogical, TypeU32(state), row_shift, lane, ConstantU32(state, 4)});
	state.builder.AddFunction(
	    {OpBitwiseAnd, TypeU32(state), bank, bank_shift, ConstantU32(state, 3)});
	state.builder.AddFunction(
	    {OpBitwiseAnd, TypeU32(state), row, row_shift, ConstantU32(state, 3)});
	state.builder.AddFunction(
	    {OpShiftLeftLogical, TypeU32(state), bank_bit, ConstantU32(state, 1), bank});
	state.builder.AddFunction(
	    {OpShiftLeftLogical, TypeU32(state), row_bit, ConstantU32(state, 1), row});
	state.builder.AddFunction(
	    {OpBitwiseAnd, TypeU32(state), bank_hit, ConstantU32(state, flags.bank_mask), bank_bit});
	state.builder.AddFunction(
	    {OpBitwiseAnd, TypeU32(state), row_hit, ConstantU32(state, flags.row_mask), row_bit});
	state.builder.AddFunction(
	    {OpINotEqual, TypeBool(state), bank_ok, bank_hit, ConstantU32(state, 0)});
	state.builder.AddFunction(
	    {OpINotEqual, TypeBool(state), row_ok, row_hit, ConstantU32(state, 0)});
	state.builder.AddFunction({OpLogicalAnd, TypeBool(state), masks_ok, bank_ok, row_ok});
	uint32_t writable = masks_ok;
	if (!flags.bound_control) {
		const auto target  = EmitDppTargetLane(state, flags.control);
		const auto bounded = state.builder.AllocateId();
		state.builder.AddFunction({OpLogicalAnd, TypeBool(state), bounded, writable, target.valid});
		writable = bounded;
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction({OpLogicalAnd, TypeBool(state), result, exec, writable});
	return result;
}

uint32_t EmitAttribute(ValueEmitContext& ctx, uint32_t attr, uint32_t chan) {
	auto&    state        = ctx.state;
	uint32_t default_bits = 0;
	if (state.stage == ShaderType::Pixel &&
	    ShaderPixelParameterDefault(*state.input_info.pixel, attr, chan & 3u, default_bits)) {
		return ConstantU32(state, default_bits);
	}
	const auto* input = InputBindingForParameter(state, attr);
	if (input == nullptr || input->variable_id == 0) {
		return ConstantU32(state, 0);
	}
	if (state.stage == ShaderType::Vertex) {
		return EmitVertexParameterComponentU32(state, *input, chan & 3u);
	}
	const auto load_per_vertex = [&](uint32_t vertex) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		state.builder.AddFunction(
		    {OpAccessChain, TypePointer(state, StorageClassInput, TypeF32(state)), pointer,
		     input->variable_id, ConstantU32(state, vertex), ConstantU32(state, chan & 3u)});
		state.builder.AddFunction({OpLoad, TypeF32(state), value, pointer});
		return value;
	};
	if (input->per_vertex) {
		if (PixelParameterIsFlat(state, attr)) {
			const auto provoking = state.builder.AllocateId();
			state.builder.AddFunction(
			    {OpBitcast, TypeU32(state), provoking, load_per_vertex(0)});
			return provoking;
		}
		const auto barycentric_kind = state.input_info.pixel->ps_no_perspective
		                                  ? IR::StageInputKind::BaryCoordNoPerspective
		                                  : IR::StageInputKind::BaryCoordSmooth;
		const auto barycentric      = InputVariableForKind(state, barycentric_kind);
		uint32_t   sum              = 0;
		for (uint32_t vertex = 0; vertex < 3u; vertex++) {
			const auto pointer = state.builder.AllocateId();
			const auto weight  = state.builder.AllocateId();
			const auto product = state.builder.AllocateId();
			state.builder.AddFunction({OpAccessChain,
			                           TypePointer(state, StorageClassInput, TypeF32(state)),
			                           pointer, barycentric, ConstantU32(state, vertex)});
			state.builder.AddFunction({OpLoad, TypeF32(state), weight, pointer});
			state.builder.AddFunction(
			    {OpFMul, TypeF32(state), product, load_per_vertex(vertex), weight});
			if (vertex == 0u) {
				sum = product;
			} else {
				const auto next = state.builder.AllocateId();
				state.builder.AddFunction({OpFAdd, TypeF32(state), next, sum, product});
				sum = next;
			}
		}
		const auto bits = state.builder.AllocateId();
		state.builder.AddFunction({OpBitcast, TypeU32(state), bits, sum});
		return bits;
	}
	const auto vector    = state.builder.AllocateId();
	const auto component = state.builder.AllocateId();
	const auto bits      = state.builder.AllocateId();
	state.builder.AddFunction({OpLoad, TypeF32Vector(state, 4), vector, input->variable_id});
	state.builder.AddFunction({OpCompositeExtract, TypeF32(state), component, vector, chan & 3u});
	state.builder.AddFunction({OpBitcast, TypeU32(state), bits, component});
	return bits;
}

uint32_t EmitInterpolationParameter(ValueEmitContext& ctx, uint32_t attr, uint32_t chan,
                                    uint32_t mode) {
	auto&       state = ctx.state;
	const auto* input = InputBindingForParameter(state, attr);
	if (input == nullptr || !input->per_vertex) {
		return EmitAttribute(ctx, attr, chan);
	}
	const auto load_vertex = [&](uint32_t vertex) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		state.builder.AddFunction(
		    {OpAccessChain, TypePointer(state, StorageClassInput, TypeF32(state)), pointer,
		     input->variable_id, ConstantU32(state, vertex), ConstantU32(state, chan & 3u)});
		state.builder.AddFunction({OpLoad, TypeF32(state), value, pointer});
		return value;
	};

	const auto selected_vertex = (mode + 1u) % 3u;
	uint32_t   value           = load_vertex(selected_vertex);
	if (!PixelParameterIsCustom(state, attr) && mode < 2u) {
		const auto delta = state.builder.AllocateId();
		state.builder.AddFunction({OpFSub, TypeF32(state), delta, value, load_vertex(0)});
		value = delta;
	}
	const auto bits = state.builder.AllocateId();
	state.builder.AddFunction({OpBitcast, TypeU32(state), bits, value});
	return bits;
}

uint32_t MrtOutputMode(const EmitterState& state, const IR::ExportInfo& exp) {
	if (state.stage != ShaderType::Pixel || exp.kind != IR::ExportTargetKind::Mrt ||
	    exp.index >= std::size(state.input_info.pixel->target_output_mode)) {
		return 0;
	}
	return state.input_info.pixel->target_output_mode[exp.index];
}

uint32_t ExportRawComponent(ValueEmitContext& ctx, uint32_t vector, uint32_t component) {
	const auto value = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(
	    {OpCompositeExtract, TypeU32(ctx.state), value, vector, component});
	return value;
}

uint32_t ExportVector(ValueEmitContext& ctx, uint32_t data, const IR::ExportInfo& exp,
                      bool uint_output) {
	auto& state = ctx.state;
	if (exp.compr && !uint_output) {
		const auto unpack =
		    MrtOutputMode(state, exp) == 5u ? GlslUnpackUnorm2x16 : GlslUnpackHalf2x16;
		uint32_t f32[4] = {ConstantF32(state, 0), ConstantF32(state, 0), ConstantF32(state, 0),
		                   ConstantF32(state, 0x3f800000u)};
		for (uint32_t pair = 0; pair < 2u; pair++) {
			if ((exp.en & (3u << (pair * 2u))) == 0u) {
				continue;
			}
			const auto packed   = state.builder.AllocateId();
			const auto unpacked = state.builder.AllocateId();
			state.builder.AddFunction({OpCompositeExtract, TypeU32(state), packed, data, pair});
			state.builder.AddFunction(
			    {OpExtInst, TypeF32Vector(state, 2), unpacked, GlslStd450(state), unpack, packed});
			for (uint32_t lane = 0; lane < 2u; lane++) {
				const auto component = pair * 2u + lane;
				if (((exp.en >> component) & 1u) != 0u) {
					f32[component] = state.builder.AllocateId();
					state.builder.AddFunction(
					    {OpCompositeExtract, TypeF32(state), f32[component], unpacked, lane});
				}
			}
		}
		const auto vector = state.builder.AllocateId();
		state.builder.AddFunction({OpCompositeConstruct, TypeF32Vector(state, 4), vector, f32[0],
		                           f32[1], f32[2], f32[3]});
		return vector;
	}
	uint32_t raw[4] = {
	    ConstantU32(state, 0),
	    ConstantU32(state, 0),
	    ConstantU32(state, 0),
	    ConstantU32(state, uint_output ? 1u : 0x3f800000u),
	};
	if (exp.compr) {
		for (uint32_t pair = 0; pair < 2u; pair++) {
			if ((exp.en & (3u << (pair * 2u))) == 0u) {
				continue;
			}
			const auto packed = ExportRawComponent(ctx, data, pair);
			for (uint32_t lane = 0; lane < 2u; lane++) {
				const auto component = pair * 2u + lane;
				if (((exp.en >> component) & 1u) == 0u) {
					continue;
				}
				raw[component] = state.builder.AllocateId();
				state.builder.AddFunction({OpBitFieldUExtract, TypeU32(state), raw[component],
				                           packed, ConstantU32(state, lane * 16u),
				                           ConstantU32(state, 16)});
			}
		}
	} else {
		for (uint32_t component = 0; component < 4u; component++) {
			if (((exp.en >> component) & 1u) != 0u) {
				raw[component] = ExportRawComponent(ctx, data, component);
			}
		}
	}
	if (uint_output) {
		const auto vector = state.builder.AllocateId();
		state.builder.AddFunction({OpCompositeConstruct, TypeU32Vector(state, 4), vector, raw[0],
		                           raw[1], raw[2], raw[3]});
		return vector;
	}
	uint32_t f32[4] {};
	for (uint32_t component = 0; component < 4u; component++) {
		f32[component] = state.builder.AllocateId();
		state.builder.AddFunction({OpBitcast, TypeF32(state), f32[component], raw[component]});
	}
	const auto vector = state.builder.AllocateId();
	state.builder.AddFunction(
	    {OpCompositeConstruct, TypeF32Vector(state, 4), vector, f32[0], f32[1], f32[2], f32[3]});
	return vector;
}

void EmitAuxPositionExport(ValueEmitContext& ctx, uint32_t data, const IR::ExportInfo& exp) {
	auto& state = ctx.state;
	for (uint32_t component = 0; component < 4; component++) {
		if ((exp.en & (1u << component)) == 0) {
			continue;
		}
		const auto output = IR::DecodePositionExportComponent(
		    state.input_info.vertex->pa_cl_vs_out_cntl, exp.index, component);
		if (output.layer || output.viewport) {
			const auto raw = ExportRawComponent(ctx, data, component);
			if (output.layer) {
				const auto layer = state.builder.AllocateId();
				state.builder.AddFunction({OpBitwiseAnd, TypeU32(state), layer, raw,
				                           ConstantU32(state, 0x7ffu)});
				const auto pointer = state.stage == ShaderType::Mesh
				                         ? MeshOutputPointer(state, IR::StageOutputKind::Layer)
				                         : state.layer_variable;
				state.builder.AddFunction({OpStore, pointer, layer});
			}
			if (output.viewport) {
				// GFX10 MISC.z packs the viewport index in bits 16..19 alongside the layer.
				const auto viewport = state.builder.AllocateId();
				state.builder.AddFunction({OpBitFieldUExtract, TypeU32(state), viewport, raw,
				                           ConstantU32(state, 16), ConstantU32(state, 4)});
				state.builder.AddFunction({OpStore, state.viewport_index_variable, viewport});
			}
			continue;
		}
		if (!output.point_size && output.clip_distance == UINT32_MAX &&
		    output.cull_distance == UINT32_MAX) {
			continue;
		}

		const auto raw = ExportRawComponent(ctx, data, component);
		const auto f32 = state.builder.AllocateId();
		state.builder.AddFunction({OpBitcast, TypeF32(state), f32, raw});
		if (output.point_size) {
			state.builder.AddFunction({OpStore, state.point_size_variable, f32});
			continue;
		}
		auto StoreDistance = [&](uint32_t variable, uint32_t index) {
			if (index == UINT32_MAX) {
				return;
			}
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(
			    {OpAccessChain, TypePointer(state, StorageClassOutput, TypeF32(state)), pointer,
			     variable, ConstantU32(state, index)});
			state.builder.AddFunction({OpStore, pointer, f32});
		};
		StoreDistance(state.clip_distance_variable, output.clip_distance);
		StoreDistance(state.cull_distance_variable, output.cull_distance);
	}
}

uint32_t ConvertClipCoordinate(EmitterState& state, uint32_t coordinate, float scale,
                               float offset, float half_extent) {
	const auto window  = state.builder.AllocateId();
	const auto biased  = state.builder.AllocateId();
	const auto divided = state.builder.AllocateId();
	const auto ndc     = state.builder.AllocateId();
	state.builder.AddFunction(
	    {OpFMul, TypeF32(state), window, coordinate, ConstantF32Value(state, scale)});
	state.builder.AddFunction(
	    {OpFAdd, TypeF32(state), biased, window, ConstantF32Value(state, offset)});
	state.builder.AddFunction(
	    {OpFDiv, TypeF32(state), divided, biased, ConstantF32Value(state, half_extent)});
	state.builder.AddFunction(
	    {OpFSub, TypeF32(state), ndc, divided, ConstantF32Value(state, 1.0f)});
	return ndc;
}

uint32_t ConvertPositionToClipSpace(EmitterState& state, uint32_t position) {
	const auto& transform = state.input_info.vertex->clip_space;
	uint32_t    components[4] {};
	for (uint32_t i = 0; i < 4; i++) {
		components[i] = state.builder.AllocateId();
		state.builder.AddFunction(
		    {OpCompositeExtract, TypeF32(state), components[i], position, i});
	}
	components[0] = ConvertClipCoordinate(state, components[0], transform.scale[0],
	                                      transform.offset[0], transform.half_extent[0]);
	components[1] = ConvertClipCoordinate(state, components[1], transform.scale[1],
	                                      transform.offset[1], transform.half_extent[1]);
	const auto converted = state.builder.AllocateId();
	state.builder.AddFunction({OpCompositeConstruct, TypeF32Vector(state, 4), converted,
	                           components[0], components[1], components[2], components[3]});
	return converted;
}

void EmitExport(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state = ctx.state;
	const auto& exp   = ctx.Export(inst);
	const auto  exec  = ctx.Arg(inst, 1);
	if (state.stage == ShaderType::Pixel && exp.vm && state.requirements.pixel_valid_mask &&
	    state.pixel_valid_mask_variable != 0) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(
		    {OpSelect, TypeU32(state), value, exec, ConstantU32(state, 1), ConstantU32(state, 0)});
		state.builder.AddFunction({OpStore, state.pixel_valid_mask_variable, value});
	}
	if (exp.kind == IR::ExportTargetKind::Null || exp.en == 0u) {
		return;
	}
	EmitIfCondition(state, exec, [&]() {
		const auto data = ctx.Arg(inst, 0);
		if (exp.kind == IR::ExportTargetKind::Primitive) {
			if (state.stage == ShaderType::Mesh) {
				state.builder.AddFunction(
				    {OpStore, MeshPrimitivePointer(state), ExportRawComponent(ctx, data, 0)});
			}
			return;
		}
		if (exp.kind == IR::ExportTargetKind::Position && exp.index != 0) {
			EmitAuxPositionExport(ctx, data, exp);
			return;
		}
		if (exp.kind == IR::ExportTargetKind::MrtZ) {
			if ((exp.en & 1u) != 0u && state.depth_variable != 0) {
				const auto raw = ExportRawComponent(ctx, data, 0);
				const auto f32 = state.builder.AllocateId();
				state.builder.AddFunction({OpBitcast, TypeF32(state), f32, raw});
				state.builder.AddFunction({OpStore, state.depth_variable, f32});
			}
			if ((exp.en & 4u) != 0u && state.sample_mask_variable != 0) {
				const auto raw     = ExportRawComponent(ctx, data, 2);
				const auto value   = state.builder.AllocateId();
				const auto pointer = state.builder.AllocateId();
				state.builder.AddFunction({OpBitcast, TypeI32(state), value, raw});
				state.builder.AddFunction(
				    {OpAccessChain, TypePointer(state, StorageClassOutput, TypeI32(state)), pointer,
				     state.sample_mask_variable, ConstantU32(state, 0)});
				state.builder.AddFunction({OpStore, pointer, value});
			}
			return;
		}
		const auto variable =
		    state.stage == ShaderType::Mesh ? 0u : OutputVariableForExport(state, exp);
		if (state.stage != ShaderType::Mesh && variable == 0) {
			return;
		}
		const bool uint_output = MrtOutputMode(state, exp) == 7u;
		const auto vector_type = uint_output ? TypeU32Vector(state, 4) : TypeF32Vector(state, 4);
		auto       value       = ExportVector(ctx, data, exp, uint_output);
		if (state.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
		    exp.index < state.input_info.pixel->target_export_mapping.size()) {
			const auto mapping = state.input_info.pixel->target_export_mapping[exp.index];
			if (!mapping.IsIdentity()) {
				const auto mapped = state.builder.AllocateId();
				state.builder.AddFunction({OpVectorShuffle, vector_type, mapped, value, value,
				                           mapping.Map(0), mapping.Map(1), mapping.Map(2),
				                           mapping.Map(3)});
				value = mapped;
			}
		}
		if (exp.kind == IR::ExportTargetKind::Position &&
		    state.input_info.vertex->clip_space.enabled) {
			value = ConvertPositionToClipSpace(state, value);
		}
		if (state.stage == ShaderType::Mesh) {
			const auto kind = exp.kind == IR::ExportTargetKind::Position
			                      ? IR::StageOutputKind::Position
			                      : IR::StageOutputKind::Parameter;
			state.builder.AddFunction({OpStore, MeshOutputPointer(state, kind, exp.index), value});
		} else if (exp.kind == IR::ExportTargetKind::Position) {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(
			    {OpAccessChain, TypePointer(state, StorageClassOutput, TypeF32Vector(state, 4)),
			     pointer, variable, ConstantU32(state, 0)});
			state.builder.AddFunction({OpStore, pointer, value});
		} else {
			state.builder.AddFunction({OpStore, variable, value});
		}
	});
}

} // namespace

bool EmitValueFlow(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	switch (inst.GetOpcode()) {
		case IR::ValueOpcode::Identity: ctx.Define(inst, ctx.Arg(inst, 0)); return true;
		case IR::ValueOpcode::Void:
		case IR::ValueOpcode::Reference:
		case IR::ValueOpcode::ReferenceU32:
		case IR::ValueOpcode::ControlNop:
		case IR::ValueOpcode::Waitcnt:
		case IR::ValueOpcode::Sendmsg:
		case IR::ValueOpcode::TtraceData:
		case IR::ValueOpcode::InstPrefetch: return true;
		case IR::ValueOpcode::Barrier: {
			const auto semantics = MemorySemanticsAcquireRelease | MemorySemanticsWorkgroupMemory;
			state.builder.AddFunction({OpControlBarrier, ConstantU32(state, ScopeWorkgroup),
			                           ConstantU32(state, ScopeWorkgroup),
			                           ConstantU32(state, semantics)});
			return true;
		}
		case IR::ValueOpcode::MeshAllocate: EmitMeshAllocate(ctx, inst); return true;
		case IR::ValueOpcode::MeshDrawParameter: {
			const auto index = inst.Arg(0).U32();
			if (state.stage != ShaderType::Mesh || index >= IR::PushData::MeshDrawDwordCount) {
				ctx.Fail(inst, "invalid mesh draw parameter");
			}
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction({OpAccessChain, TypePushConstantElementPointer(state),
			                           pointer, state.push_constant_variable, ConstantU32(state, 0),
			                           ConstantU32(state, index)});
			state.builder.AddFunction({OpLoad, TypeU32(state), ctx.Result(inst), pointer});
			return true;
		}
		case IR::ValueOpcode::GetUserData: {
			const auto reg   = inst.Arg(0).ScalarRegister();
			uint32_t   dword = 0;
			if (!UserDataDwordIndex(state, reg, dword)) {
				ctx.Define(inst, ConstantU32(state, 0));
			} else {
				ctx.Define(inst, EmitShaderDataDwordLoad(state, dword));
			}
			return true;
		}
		case IR::ValueOpcode::GetBuiltin:
			ctx.Define(inst, EmitBuiltinU32(ctx, static_cast<IR::StageInputKind>(inst.Arg(0).U32()),
			                                inst.Arg(1).U32()));
			return true;
		case IR::ValueOpcode::UndefU1:
		case IR::ValueOpcode::UndefU8:
		case IR::ValueOpcode::UndefU16:
		case IR::ValueOpcode::UndefU32:
		case IR::ValueOpcode::UndefU64:
			state.builder.AddFunction({OpUndef, ctx.TypeId(inst.GetType()), ctx.Result(inst)});
			return true;
		case IR::ValueOpcode::DppMoveU32: {
			const auto flags    = inst.Flags<IR::DppMoveFlags>();
			const auto target   = EmitDppTargetLane(state, flags.control);
			// (A translation feature: a static store made without it keeps its pipelines, its next precompile brings it.)
			if (state.stage == ShaderType::Pixel && flags.control <= 0xffu && state.program.pixel_quad_reads &&
			    g_pixel_quad_operations.load(std::memory_order_relaxed)) {
				ctx.Define(inst, EmitDppQuadPermPixel(ctx, inst, flags, target.lane));
				return true;
			}
			const auto shuffled = ctx.Shuffle(inst, 0, target.lane);
			if (flags.fetch_inactive) {
				ctx.Define(inst, shuffled);
				return true;
			}
			const auto ballot        = ctx.Ballot(inst.Arg(1));
			const auto source_active = EmitBallotLaneActiveBool(state, ballot, target.lane);
			const auto can_fetch     = state.builder.AllocateId();
			state.builder.AddFunction(
			    {OpLogicalAnd, TypeBool(state), can_fetch, target.valid, source_active});
			ctx.Emit(inst, OpSelect, IR::Type::U32, {can_fetch, shuffled, ConstantU32(state, 0)});
			return true;
		}
		case IR::ValueOpcode::DppUpdateU32: {
			const auto flags = inst.Flags<IR::DppMoveFlags>();
			const auto write = EmitDppWriteCondition(ctx, flags, ctx.Arg(inst, 2));
			ctx.Emit(inst, OpSelect, IR::Type::U32, {write, ctx.Arg(inst, 0), ctx.Arg(inst, 1)});
			return true;
		}
		case IR::ValueOpcode::WqmU64:
			ctx.Define(inst, EmitWqmU64(ctx.state, ctx.Arg(inst, 0)));
			return true;
		case IR::ValueOpcode::LaneId:
			ctx.Define(inst, EmitSubgroupLocalInvocationId(state));
			return true;
		case IR::ValueOpcode::WaveLaneId:
			// (A lane-local wave64 program: its index in the group, as BufferByteAddress takes ADD_TID; otherwise LaneId's.)
			ctx.Define(inst, state.lane_count == 1 && state.program.lane_local
			                     ? Binary(state, OpBitwiseAnd, TypeU32(state), EmitLocalInvocationIndex(state),
			                              ConstantU32(state, 63))
			                     : EmitSubgroupLocalInvocationId(state));
			return true;
		case IR::ValueOpcode::Ballot: ctx.Define(inst, ctx.Ballot(inst.Arg(0))); return true;
		case IR::ValueOpcode::AnyLane: {
			const auto ballot = ctx.Ballot(inst.Arg(0));
			const auto low    = state.builder.AllocateId();
			const auto high   = state.builder.AllocateId();
			state.builder.AddFunction({OpCompositeExtract, TypeU32(state), low, ballot, 0});
			state.builder.AddFunction({OpCompositeExtract, TypeU32(state), high, ballot, 1});
			auto bits = EmitBinaryU32(state, OpBitwiseOr, low, high);
			if (CooperativeWave(ctx, inst)) {
				// (Each half's lanes, as every lane of the half stored them: the wave's.)
				const auto exchange = BeginWaveExchange(state);
				StoreWaveSlot(state, exchange.base, exchange.lane, bits);
				WaveBarrier(state);
				bits = EmitBinaryU32(state, OpBitwiseOr, LoadWaveSlot(state, exchange.base, ConstantU32(state, 0)),
				                     LoadWaveSlot(state, exchange.base, ConstantU32(state, 32)));
			}
			ctx.Emit(inst, OpINotEqual, IR::Type::U1, {bits, ConstantU32(state, 0)});
			return true;
		}
		case IR::ValueOpcode::ReadFirstLane: {
			const auto ballot = ctx.Ballot(inst.Arg(1));
			if (CooperativeWave(ctx, inst)) {
				// The wave's first active lane (none: lane 0), from both halves' active lanes.
				const auto active   = state.builder.AllocateId();
				state.builder.AddFunction({OpCompositeExtract, TypeU32(state), active, ballot, 0});
				const auto exchange = BeginWaveExchange(state);
				StoreWaveSlot(state, exchange.base, exchange.lane, ctx.Arg(inst, 0));
				StoreWaveSlot(state, exchange.base, EmitAddU32(state, exchange.lane, ConstantU32(state, 64)), active);
				WaveBarrier(state);
				const auto low       = LoadWaveSlot(state, exchange.base, ConstantU32(state, 64));
				const auto high      = LoadWaveSlot(state, exchange.base, ConstantU32(state, 96));
				const auto low_lane  = state.builder.AllocateId();
				const auto high_lane = state.builder.AllocateId();
				state.builder.AddFunction({OpExtInst, TypeU32(state), low_lane, GlslStd450(state), GlslFindILsb, low});
				state.builder.AddFunction({OpExtInst, TypeU32(state), high_lane, GlslStd450(state), GlslFindILsb, high});
				const auto in_low  = state.builder.AllocateId();
				const auto in_high = state.builder.AllocateId();
				state.builder.AddFunction({OpINotEqual, TypeBool(state), in_low, low, ConstantU32(state, 0)});
				state.builder.AddFunction({OpINotEqual, TypeBool(state), in_high, high, ConstantU32(state, 0)});
				const auto high_or_none = state.builder.AllocateId();
				const auto first        = state.builder.AllocateId();
				state.builder.AddFunction({OpSelect, TypeU32(state), high_or_none, in_high,
				                           EmitAddU32(state, high_lane, ConstantU32(state, 32)), ConstantU32(state, 0)});
				state.builder.AddFunction({OpSelect, TypeU32(state), first, in_low, low_lane, high_or_none});
				ctx.Define(inst, LoadWaveSlot(state, exchange.base, first));
				return true;
			}
			const auto lane   = ctx.FirstLane(ballot);
			ctx.Define(inst, ctx.Shuffle(inst, 0, lane));
			return true;
		}
		case IR::ValueOpcode::ReadLane:
			if (CooperativeWave(ctx, inst)) {
				// The value lane L stored, whichever half it is in.
				const auto exchange = BeginWaveExchange(state);
				StoreWaveSlot(state, exchange.base, exchange.lane, ctx.Arg(inst, 0));
				WaveBarrier(state);
				const auto lane = EmitBinaryU32(state, OpBitwiseAnd, ctx.Arg(inst, 1), ConstantU32(state, 63));
				ctx.Define(inst, LoadWaveSlot(state, exchange.base, lane));
				return true;
			}
			if (IsWaveOrScanRead(ctx, inst)) {
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1) < 64)
					std::printf("PS_WAVE_OR %016llx lane %u\n", static_cast<unsigned long long>(state.program.shader_hash),
					            inst.Arg(1).Resolve().U32());
				// A wave64 OR reduction (DPP row_shr 1/2/4/8, v_permlanex16, v_readlane 31/63) in a graphics shader that
				// runs on 32-wide host subgroups: lane 63 does not exist and fragment subgroups can have lanes without a
				// fragment, which break the scan's chain (dark columns on Demon's Souls' water, PS 7a59c1a20cbe5d4a).
				// Each lane's scanned value holds its own bits and only bits of the wave, so the subgroup's OR of them
				// is the union the scan meant. (The capability only where it is used: every other module keeps its
				// SPIR-V, so the pipelines precompiled for them stay valid.)
				state.builder.RequireCapability(CapabilityGroupNonUniformArithmetic);
				const auto result = state.builder.AllocateId();
				state.builder.AddFunction({OpGroupNonUniformBitwiseOr, TypeU32(state), result,
				                           ConstantU32(state, ScopeSubgroup), 0u /* Reduce */, ctx.Arg(inst, 0)});
				ctx.Define(inst, result);
				return true;
			}
			ctx.Define(inst, ctx.Shuffle(inst, 0, ctx.Arg(inst, 1)));
			return true;
		case IR::ValueOpcode::WriteLane: {
			const auto hit = state.builder.AllocateId();
			state.builder.AddFunction({OpIEqual, TypeBool(state), hit,
			                           EmitSubgroupLocalInvocationId(state), ctx.Arg(inst, 2)});
			ctx.Emit(inst, OpSelect, IR::Type::U32, {hit, ctx.Arg(inst, 1), ctx.Arg(inst, 0)});
			return true;
		}
		case IR::ValueOpcode::Permlane16U32: {
			const auto flags     = inst.Flags<IR::PermlaneFlags>();
			const auto subid     = EmitSubgroupLocalInvocationId(state);
			const auto row       = state.builder.AllocateId();
			const auto row_value = state.builder.AllocateId();
			const auto lane      = state.builder.AllocateId();
			const auto lane8     = state.builder.AllocateId();
			const auto shift     = state.builder.AllocateId();
			const auto upper     = state.builder.AllocateId();
			const auto selected  = state.builder.AllocateId();
			const auto shifted   = state.builder.AllocateId();
			const auto index     = state.builder.AllocateId();
			const auto target    = state.builder.AllocateId();
			state.builder.AddFunction(
			    {OpBitwiseAnd, TypeU32(state), row, subid, ConstantU32(state, 0xfffffff0u)});
			if (flags.x16) {
				state.builder.AddFunction(
				    {OpBitwiseXor, TypeU32(state), row_value, row, ConstantU32(state, 16)});
			} else {
				state.builder.AddFunction({OpCopyObject, TypeU32(state), row_value, row});
			}
			state.builder.AddFunction(
			    {OpBitwiseAnd, TypeU32(state), lane, subid, ConstantU32(state, 15)});
			state.builder.AddFunction(
			    {OpBitwiseAnd, TypeU32(state), lane8, lane, ConstantU32(state, 7)});
			state.builder.AddFunction(
			    {OpShiftLeftLogical, TypeU32(state), shift, lane8, ConstantU32(state, 2)});
			state.builder.AddFunction(
			    {OpUGreaterThanEqual, TypeBool(state), upper, lane, ConstantU32(state, 8)});
			state.builder.AddFunction(
			    {OpSelect, TypeU32(state), selected, upper, ctx.Arg(inst, 2), ctx.Arg(inst, 1)});
			state.builder.AddFunction(
			    {OpShiftRightLogical, TypeU32(state), shifted, selected, shift});
			state.builder.AddFunction(
			    {OpBitwiseAnd, TypeU32(state), index, shifted, ConstantU32(state, 15)});
			state.builder.AddFunction({OpBitwiseOr, TypeU32(state), target, row_value, index});
			const auto shuffled = ctx.Shuffle(inst, 0, target);
			uint32_t result = shuffled;
			if (!flags.fetch_inactive) {
				const auto source_exec = ctx.Shuffle(inst, 3, target);
				result = state.builder.AllocateId();
				state.builder.AddFunction({OpSelect, TypeU32(state), result, source_exec, shuffled,
				                           ConstantU32(state, 0)});
			}
			ctx.Define(inst, result);
			return true;
		}
		case IR::ValueOpcode::GetAttribute:
			ctx.Define(inst, EmitAttribute(ctx, inst.Arg(0).U32(), inst.Arg(1).U32()));
			return true;
		case IR::ValueOpcode::GetInterpolationParameter:
			ctx.Define(inst, EmitInterpolationParameter(ctx, inst.Arg(0).U32(), inst.Arg(1).U32(),
			                                            inst.Arg(2).U32()));
			return true;
		case IR::ValueOpcode::SetAttribute: EmitExport(ctx, inst); return true;
		case IR::ValueOpcode::GetShaderBase:
			// Guest S_GETPC values stay shader-relative in SPIR-V, matching the runtime ABI. The
			// runtime descriptor evaluator supplies the mapped shader base for host-side planning.
			ctx.Define(inst, ctx.Def(IR::Value(uint64_t {0})));
			return true;
		case IR::ValueOpcode::GetSrtResource:
		case IR::ValueOpcode::GetBufferResource:
		case IR::ValueOpcode::GetAddressResource:
		case IR::ValueOpcode::GetScratchResource:
		case IR::ValueOpcode::GetImageResource:
		case IR::ValueOpcode::GetSamplerResource:
		case IR::ValueOpcode::MakeImageAddress: return true;
		default: return false;
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter

namespace Libs::Graphics::ShaderRecompiler::Spirv {

void SetPixelQuadOperations(bool supported) {
	Emitter::g_pixel_quad_operations.store(supported, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
