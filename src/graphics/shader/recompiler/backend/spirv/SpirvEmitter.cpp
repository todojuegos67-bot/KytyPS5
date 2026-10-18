#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <optional>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

[[noreturn]] void Fail(const IR::Program& program, const char* reason) {
	EXIT("SPIR-V validation failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     program.shader_hash, static_cast<unsigned>(program.stage), reason);
	std::abort();
}

void ValidateNativeProgram(const IR::Program& program, bool lds_storage) {
	std::vector<bool> live_buffers(program.info.buffers.size());
	bool uses_lds = false;
	bool uses_gds = false;
	const auto uses_mapping = [](const auto& resource) {
		return resource.indirect_root != UINT32_MAX;
	};
	// Research: a bindless image or sampler reads its heap region from the flattened SRT.
	bool uses_flattened_srt = std::ranges::any_of(program.info.buffers, uses_mapping) ||
	                          std::ranges::any_of(program.info.images, uses_mapping) ||
	                          std::ranges::any_of(program.info.images, &IR::ImageResource::bindless) ||
	                          std::ranges::any_of(program.info.samplers, &IR::SamplerResource::bindless);
	const auto planning_only_handle = [&](const IR::Inst& handle) {
		return !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       const auto op = use.user->GetOpcode();
			       if (op != IR::ValueOpcode::LoadAddressU32 &&
			           op != IR::ValueOpcode::ReadConstBuffer) {
				       return false;
			       }
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].planning_only;
		       });
	};
	const auto indirect_buffer_handle = [&](const IR::Inst& handle) {
		return program.info.uses_dma && handle.NumArgs() == 4u && !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       if (IR::BufferAccessOf(use.user->GetOpcode()) != IR::BufferAccess::Read) {
				       return false;
			       }
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].kind == IR::ResourceKind::IndirectBuffer;
		       });
	};
	const auto local_flat_handle = [&](const IR::Inst& handle) {
		return !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       if (IR::AddressOpcodeInfoOf(use.user->GetOpcode()).access == IR::AddressAccess::None)
				       return false;
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].kind == IR::ResourceKind::FlatLocal;
		       });
	};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			uses_flattened_srt |= op == IR::ValueOpcode::ReadConst;
			if (IR::BufferAccessOf(op) != IR::BufferAccess::None ||
			    IR::SharedAccessOf(op) != IR::SharedAccess::None ||
			    IR::AddressOpcodeInfoOf(op).access != IR::AddressAccess::None) {
				const auto index = inst.Flags<IR::MemoryFlags>().index;
				if (index >= program.memory_info.size()) {
					Fail(program, "typed shader contains invalid memory metadata");
				}
				const auto& memory = program.memory_info[index];
				if (!memory.planning_only) {
					uses_lds |= memory.kind == IR::ResourceKind::FlatLocal;
					if (IR::SharedAccessOf(op) != IR::SharedAccess::None) {
						if (memory.kind != IR::ResourceKind::Lds && memory.kind != IR::ResourceKind::Gds) {
							Fail(program, "typed shader contains invalid shared-memory metadata");
						}
						uses_gds |= memory.kind == IR::ResourceKind::Gds;
						uses_lds |= memory.kind == IR::ResourceKind::Lds;
					} else if (memory.kind == IR::ResourceKind::Buffer ||
					           memory.kind == IR::ResourceKind::ScalarBuffer) {
						if (memory.resource >= program.info.buffers.size()) {
							Fail(program, "typed shader contains an invalid buffer resource");
						}
						live_buffers[memory.resource] = true;
						for (const auto child: program.info.buffers[memory.resource].indirect_resources) {
							if (child >= program.info.buffers.size()) {
								Fail(program, "typed shader contains an invalid indirect buffer resource");
							}
							live_buffers[child] = true;
						}
					}
				}
			}
			const auto dense = inst.Flags<uint32_t>();
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::GetBufferResource:
					if (planning_only_handle(inst) || indirect_buffer_handle(inst)) {
						break;
					}
					if (dense >= program.info.buffers.size()) {
						Fail(program, "typed buffer handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetBufferResourceIndex:
					if (dense >= program.info.buffers.size()) {
						Fail(program, "typed buffer selector has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetAddressResource:
					if (planning_only_handle(inst)) {
						break;
					}
					if (inst.NumArgs() != 2 || (!program.info.uses_dma && !local_flat_handle(inst))) {
						Fail(program, "typed address handle has invalid DMA metadata");
					}
					break;
				case IR::ValueOpcode::GetScratchResource:
					if (inst.NumArgs() != 0 || program.scratch_dwords == 0) {
						Fail(program, "typed scratch handle has invalid shader metadata");
					}
					break;
				case IR::ValueOpcode::GetImageResource:
					if (dense >= program.info.images.size()) {
						Fail(program, "typed image handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetSamplerResource:
					if (dense >= program.info.samplers.size()) {
						Fail(program, "typed sampler handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::ReadConst: {
					const auto slot = inst.Arg(1).Resolve();
					if (!slot.IsImmediate() || slot.GetType() != IR::Type::U32 ||
					    slot.U32() >= program.srt_reads.size()) {
						Fail(program, "flattened SRT read has an invalid dense slot");
					}
					break;
				}
				default: break;
			}
		}
	}
	using Kind = IR::DescriptorBindingKind;
	std::array<uint32_t, static_cast<size_t>(Kind::Count)> expected {};
	auto& buffer_count = expected[static_cast<size_t>(Kind::Buffers)];
	for (uint32_t index = 0; index < program.info.buffers.size(); ++index) {
		const auto slot = live_buffers[index] ? buffer_count++ : UINT32_MAX;
		if (program.info.buffers[index].live != live_buffers[index] ||
		    program.info.buffers[index].descriptor_index != slot) {
			Fail(program, "native buffer descriptor index does not match shader topology");
		}
	}
	for (const auto& image: program.info.images) {
		const auto kind = IR::DescriptorBindingForImage(image);
		if (!kind.has_value()) {
			Fail(program, "native shader plan has an invalid image class");
		}
		if (image.mip_count == 0u ||
		    (image.mip_mode != IR::ImageMipMode::Dynamic && image.mip_count != 1u)) {
			Fail(program, "native shader plan has an invalid image mip descriptor count");
		}
		auto& count = expected[static_cast<size_t>(*kind)];
		if (image.descriptor_index != count) {
			Fail(program, "native image descriptor index does not match shader topology");
		}
		count += image.mip_count;
	}
	expected[static_cast<size_t>(Kind::Samplers)] = static_cast<uint32_t>(program.info.samplers.size());
	expected[static_cast<size_t>(Kind::Gds)] = uses_gds;
	expected[static_cast<size_t>(Kind::SharedMemory)] = uses_lds && lds_storage;
	expected[static_cast<size_t>(Kind::BdaPagetable)] = program.info.uses_dma;
	expected[static_cast<size_t>(Kind::FaultBuffer)] = program.info.uses_dma;
	expected[static_cast<size_t>(Kind::FlattenedSrt)] = uses_flattened_srt;
	expected[static_cast<size_t>(Kind::ShaderData)] =
	    program.bindings.ShaderDataDwords() != 0 && !program.bindings.UsesPushData();
	uint64_t descriptor_mask = 0;
	for (uint32_t index = 0; index < expected.size(); ++index) {
		if (expected[index] != 0) descriptor_mask |= uint64_t{1} << index;
	}
	if (program.bindings.descriptor_counts != expected ||
	    program.bindings.descriptor_mask != descriptor_mask) {
		Fail(program, "native descriptor counts do not match shader topology");
	}
	const auto shader_data_dwords = program.bindings.ShaderDataDwords();
	const auto& registers = program.info.user_data_registers;
	const bool has_dispatch_threads = program.bindings.dispatch_thread_dword != IR::PushData::NoStart;
	if ((program.bindings.UsesPushData() &&
	     !IR::PushData::CanFit(program.bindings.push_data_start_dword, shader_data_dwords)) ||
	    (has_dispatch_threads && (program.stage != ShaderType::Compute ||
	                              program.bindings.dispatch_thread_dword != registers.size())) ||
	    program.bindings.memory_offset_dword != registers.size() + (has_dispatch_threads ? 3u : 0u) ||
	    !std::is_sorted(registers.begin(), registers.end()) ||
	    std::adjacent_find(registers.begin(), registers.end()) != registers.end()) {
		Fail(program, "native shader-data layout is inconsistent");
	}
}

// An upper bound of a U32 value built from constants, the lane id and simple integer arithmetic,
// or nothing. Phi cycles (a loop-carried address) are unbounded.
std::optional<uint64_t> UpperBoundU32(IR::Value value, uint32_t wave_size,
                                      std::vector<const IR::Inst*>& visiting) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		return value.GetType() == IR::Type::U32 ? std::optional<uint64_t>(value.U32())
		                                        : std::nullopt;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || visiting.size() > 64u || std::ranges::find(visiting, inst) != visiting.end()) {
		return std::nullopt;
	}
	visiting.push_back(inst);
	const auto arg = [&](size_t index) { return UpperBoundU32(inst->Arg(index), wave_size, visiting); };
	const auto immediate = [&](size_t index) -> std::optional<uint32_t> {
		const auto operand = inst->Arg(index).Resolve();
		return operand.IsImmediate() && operand.GetType() == IR::Type::U32
		           ? std::optional<uint32_t>(operand.U32())
		           : std::nullopt;
	};
	std::optional<uint64_t> result;
	switch (inst->GetOpcode()) {
		case IR::ValueOpcode::LaneId: result = wave_size - 1u; break;
		case IR::ValueOpcode::IAdd32:
			if (const auto a = arg(0), b = arg(1); a && b) {
				result = *a + *b;
			}
			break;
		case IR::ValueOpcode::IMul32:
			if (const auto a = arg(0), b = arg(1); a && b) {
				result = *a * *b;
			}
			break;
		case IR::ValueOpcode::ShiftLeftLogical32:
		case IR::ValueOpcode::ShiftRightLogical32: {
			const auto a = arg(0);
			const auto b = immediate(1);
			if (a && b && *b < 32u) {
				result = inst->GetOpcode() == IR::ValueOpcode::ShiftLeftLogical32 ? *a << *b : *a >> *b;
			}
			break;
		}
		case IR::ValueOpcode::BitwiseAnd32: {
			const auto a = arg(0), b = arg(1);
			if (a || b) {
				result = std::min(a.value_or(UINT32_MAX), b.value_or(UINT32_MAX));
			}
			break;
		}
		case IR::ValueOpcode::BitwiseOr32:
		case IR::ValueOpcode::BitwiseXor32:
			if (const auto a = arg(0), b = arg(1); a && b) {
				result = std::bit_ceil(std::max(*a, *b) + 1u) - 1u;
			}
			break;
		case IR::ValueOpcode::SelectU32:
			if (const auto a = arg(1), b = arg(2); a && b) {
				result = std::max(*a, *b);
			}
			break;
		case IR::ValueOpcode::Phi: {
			uint64_t bound = 0;
			bool     known = inst->NumArgs() != 0;
			for (size_t index = 0; known && index < inst->NumArgs(); index++) {
				const auto incoming = arg(index);
				known               = incoming.has_value();
				bound               = known ? std::max(bound, *incoming) : bound;
			}
			if (known) {
				result = bound;
			}
			break;
		}
		default: break;
	}
	visiting.pop_back();
	return result && *result <= UINT32_MAX ? result : std::nullopt;
}

} // namespace

// The Wolverine-side requirements IR::ShaderInfo does not collect: they are added to the
// program's info (subgroup features) or kept in EmitterState::requirements.
Emitter::SpirvRequirements Emitter::AnalyzeWolverineRequirements(IR::Program& program) {
	SpirvRequirements requirements {};
	auto&             info = program.info;
	// A non-compute stage keeps LDS in a function-scope array of its own, which NVIDIA places in
	// local memory: at the 8192-dword default that is 32 KB per invocation, several GB across the
	// GPU. Size it from the addresses the shader can form instead.
	bool     function_lds           = false;
	uint64_t function_lds_bytes     = 0;
	bool     function_lds_unbounded = false;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (IR::BufferAccessOf(inst.GetOpcode()) != IR::BufferAccess::None) {
				const auto memory_index = inst.Flags<IR::MemoryFlags>().index;
				if (memory_index < program.memory_info.size()) {
					const auto& memory = program.memory_info[memory_index];
					if (memory.kind == IR::ResourceKind::Buffer && memory.gpu_records) {
						info.subgroup_local_invocation_id = true;
					}
				}
			}
			const auto address_access = IR::AddressOpcodeInfoOf(inst.GetOpcode()).access;
			if (address_access != IR::AddressAccess::None) {
				const auto memory_index = inst.Flags<IR::MemoryFlags>().index;
				if (memory_index < program.memory_info.size() &&
				    program.memory_info[memory_index].kind == IR::ResourceKind::FlatLocal &&
				    program.stage != ShaderType::Compute && program.stage != ShaderType::Mesh) {
					// A FLAT access routed to LDS: its address is not bounded here.
					function_lds           = true;
					function_lds_unbounded = true;
				}
			}
			const auto shared_access = IR::SharedAccessOf(inst.GetOpcode());
			if (shared_access != IR::SharedAccess::None) {
				const auto index = inst.Flags<IR::MemoryFlags>().index;
				if (index < program.memory_info.size() &&
				    program.memory_info[index].kind == IR::ResourceKind::Lds &&
				    program.stage != ShaderType::Compute && program.stage != ShaderType::Mesh) {
					function_lds = true;
					std::vector<const IR::Inst*> visiting;
					const auto address = UpperBoundU32(inst.Arg(0), program.wave_size, visiting);
					if (address.has_value()) {
						const auto& memory = program.memory_info[index];
						function_lds_bytes = std::max(
						    function_lds_bytes, *address + std::max(memory.offset, memory.secondary_offset) +
						                            4u * IR::SharedComponentCount(inst.GetOpcode()));
					} else {
						function_lds_unbounded = true;
					}
				}
			}
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::ReadClockRealtime64: requirements.shader_clock = true; break;
				case IR::ValueOpcode::ReadLane:
					if (IR::MatchLaneReduction(inst, program.wave_size)) {
						info.subgroup_arithmetic          = true;
						info.subgroup_local_invocation_id = true;
					}
					break;
				default: break;
			}
		}
	}
	// Loop watchdog reports store the lane (EmitLoopWatchdog); after the upstream merge such
	// a program may have no other lane use (CS 0x1ca15ea9516b599e exited at boot).
	info.subgroup_local_invocation_id |= info.watchdog_reports;
	if (function_lds && !function_lds_unbounded && function_lds_bytes < 8192u * sizeof(uint32_t)) {
		requirements.function_lds_dwords =
		    static_cast<uint32_t>((function_lds_bytes + sizeof(uint32_t) - 1u) / sizeof(uint32_t));
	}
	return requirements;
}

namespace {
std::atomic_bool g_image_min_lod {false};
std::atomic_bool g_pixel_interlock {false};
} // namespace

void SetHostImageFeatures(const HostImageFeatures& features) {
	g_image_min_lod.store(features.min_lod, std::memory_order_relaxed);
	g_pixel_interlock.store(features.pixel_interlock, std::memory_order_relaxed);
}

HostImageFeatures GetHostImageFeatures() {
	return {.min_lod         = g_image_min_lod.load(std::memory_order_relaxed),
	        .pixel_interlock = g_pixel_interlock.load(std::memory_order_relaxed)};
}


std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t push_data_start_dword, bool maintenance8_enabled,
                                  bool emit_debug_info) {
	using namespace Emitter;

	if (program.stage != ShaderType::Compute && program.stage != ShaderType::Vertex &&
	    program.stage != ShaderType::Pixel && program.stage != ShaderType::Mesh &&
	    program.stage != ShaderType::Local && program.stage != ShaderType::TessellationControl &&
	    program.stage != ShaderType::TessellationEvaluation) {
		Fail(program, "binary SPIR-V emitter received an unsupported shader stage");
	}
	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	IR::ValidateProgram(program, true);
	program.bindings = {.push_data_start_dword = push_data_start_dword};
	const auto requirements = AnalyzeWolverineRequirements(program);
	EmitterState state(program, input_info, maintenance8_enabled);
	state.requirements = requirements;
	if (emit_debug_info) {
		char name[64];
		std::snprintf(name, sizeof(name), "rdna2-%016" PRIx64 "-block-byte-pc-plus-one",
		              program.shader_hash);
		state.debug_source = state.builder.AddString(name);
	}
	const auto* workgroup = ShaderWorkgroupInput(program.stage, input_info);
	state.lane_count =
	    workgroup != nullptr && program.wave_size == 64u && workgroup->host_subgroup_size == 32u
	        ? 2u
	        : 1u;
	if (program.stage == ShaderType::Mesh) {
		state.mesh_passes = std::max(input_info.vertex->mesh.passes, 1u);
	}
	DefineModule(state);
	ValidateNativeProgram(program, program.stage == ShaderType::Compute &&
	                                   input_info.compute != nullptr && input_info.compute->lds_storage);
	EmitProgram(state);
	state.builder.AddEntryPoint(ExecutionModelForStage(state.program.stage), state.main_func,
	                            "main", state.interface_variables);

	return state.builder.Build();
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
