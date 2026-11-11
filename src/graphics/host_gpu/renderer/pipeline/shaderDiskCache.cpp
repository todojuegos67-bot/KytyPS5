#include "graphics/host_gpu/renderer/pipeline/shaderDiskCache.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "shaderCacheVersion.h"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_map>
#include <xxhash.h>

namespace Libs::Graphics::ShaderDiskCache {

namespace IR = ShaderRecompiler::IR;

namespace {

// Bump when the encoding below changes in a way the source version cannot see (it hashes this
// file too, so this is belt and braces).
constexpr uint32_t FormatVersion = 4;           // Round 118: BindingLayout::partition_target_dword.
constexpr uint32_t FileMagic     = 0x4344534bu; // "KSDC"
constexpr uint64_t MaxFileSize   = uint64_t {256} << 20u;

template <typename T>
struct IsVector: std::false_type {};
template <typename T>
struct IsVector<std::vector<T>>: std::true_type {};
template <typename T>
struct IsArray: std::false_type {};
template <typename T, size_t N>
struct IsArray<std::array<T, N>>: std::true_type {};
template <typename T>
struct IsOptional: std::false_type {};
template <typename T>
struct IsOptional<std::optional<T>>: std::true_type {};

template <typename T, typename U>
concept Is = std::same_as<std::remove_const_t<T>, U>;

// Every serialized structure lists its fields once; the writer and the reader both walk this
// list, so they cannot disagree. A field added to one of these structures must be added here
// (ShaderIR.h says so next to them).
template <typename Io, typename T>
requires Is<T, IR::MemoryInfo>
void Visit(Io& io, T& v) {
	io(v.kind);
	io(v.resource);
	io(v.sampler);
	io(v.offset);
	io(v.secondary_offset);
	io(v.dmask);
	io(v.data_dwords);
	io(v.data_bits);
	io(v.component_index);
	io(v.component_count);
	io(v.data_format);
	io(v.number_format);
	io(v.image_sample_flags);
	io(v.image_dimension);
	io(v.image_address_components);
	io(v.address_is_full);
	io(v.data_signed);
	io(v.typed);
	io(v.formatted);
	io(v.image_has_mip);
	io(v.image_r128);
	io(v.idxen);
	io(v.offen);
	io(v.coherent);
	io(v.planning_only);
}

template <typename Io, typename T>
requires Is<T, IR::BufferResource>
void Visit(Io& io, T& v) {
	io(v.source);
	io(v.first_use_pc);
	io(v.max_byte_extent);
	io(v.packed_stride);
	io(v.descriptor_format);
	io(v.descriptor_swizzle);
	io(v.image_alias);
	io(v.read);
	io(v.written);
	io(v.atomic);
	io(v.formatted);
	io(v.scalar);
	io(v.write_tracked);
	io(v.indirect_root);
	io(v.indirect_mapping_offset);
	io(v.indirect_search_iterations);
	io(v.indirect_resources);
}

template <typename Io, typename T>
requires Is<T, IR::ImageResource>
void Visit(Io& io, T& v) {
	io(v.source);
	io(v.first_use_pc);
	io(v.resource_class);
	io(v.numeric_class);
	io(v.dimension);
	io(v.mip_mode);
	io(v.mip_count);
	io(v.conversion_format);
	io(v.shader_swizzle);
	io(v.read);
	io(v.written);
	io(v.atomic);
	io(v.atomic64);
	io(v.depth_compare);
	io(v.cube);
	io(v.r128);
	io(v.indirect_root);
	io(v.indirect_mapping_offset);
	io(v.indirect_search_iterations);
	io(v.bindless);
	io(v.runtime);
	io(v.clamp_half_float);
	io(v.indirect_resources);
}

template <typename Io, typename T>
requires Is<T, IR::SamplerResource>
void Visit(Io& io, T& v) {
	io(v.source);
	io(v.first_use_pc);
	io(v.snapshot_index);
	io(v.force_point_filtering);
	io(v.depth_compare);
	io(v.integer_border);
	io(v.gather_lod);
	io(v.bindless);
	io(v.bindless_mapping_offset);
}

template <typename Io, typename T>
requires Is<T, IR::SampledResourcePair>
void Visit(Io& io, T& v) {
	io(v.image);
	io(v.sampler);
	io(v.first_use_pc);
}

template <typename Io, typename T>
requires Is<T, IR::StageInput>
void Visit(Io& io, T& v) {
	io(v.kind);
	io(v.location);
	io(v.component_count);
	io(v.debug_name);
	io(v.per_vertex);
}

template <typename Io, typename T>
requires Is<T, IR::StageOutput>
void Visit(Io& io, T& v) {
	io(v.kind);
	io(v.index);
	io(v.location);
	io(v.debug_name);
}

template <typename Io, typename T>
requires Is<T, IR::ShaderInfo>
void Visit(Io& io, T& v) {
	io(v.buffers);
	io(v.images);
	io(v.samplers);
	io(v.sampled_pairs);
	io(v.inputs);
	io(v.outputs);
	io(v.vertex_fetch_components);
	io(v.vertex_offset_sgpr);
	io(v.instance_offset_sgpr);
	io(v.has_bitwise_xor);
	io(v.uses_dma);
	io(v.uses_bvh);
}

template <typename Io, typename T>
requires Is<T, IR::DescriptorBinding>
void Visit(Io& io, T& v) {
	io(v.kind);
	io(v.resources);
}

template <typename Io, typename T>
requires Is<T, IR::BindingLayout>
void Visit(Io& io, T& v) {
	io(v.push_data_start_dword);
	io(v.dispatch_thread_dword);
	io(v.partition_target_dword);
	io(v.memory_offset_dword);
	io(v.memory_offset_count);
	io(v.write_tracking);
	io(v.user_data_registers);
	io(v.descriptors);
}

template <typename Io, typename T>
requires Is<T, IR::CompiledShaderInfo>
void Visit(Io& io, T& v) {
	io(v.stage);
	io(v.shader_hash);
	io(v.wave_size);
	io(v.user_data_base);
	io(v.user_data_count);
	io(v.scratch_dwords);
	io(v.param_export_mask);
	io(v.has_address_writes);
	io(v.info);
	io(v.bindings);
}

template <typename Io, typename T>
requires Is<T, IR::DescriptorSource::IndirectDescriptor>
void Visit(Io& io, T& v) {
	io(v.material_source);
	io(v.table_source);
	io(v.selector_stride);
	io(v.selector_offset);
	io(v.table_offset);
	io(v.table_stride);
	io(v.workgroup_axis);
	io(v.selector_shift);
	io(v.key_shift);
	io(v.key_mask);
	io(v.bindless);
	io(v.runtime);
	io(v.key_count);
	io(v.selector_first);
	io(v.selector_mask);
	io(v.sources);
}

template <typename Io, typename T>
requires Is<T, IR::DescriptorSource::BindlessSampler>
void Visit(Io& io, T& v) {
	io(v.table_offset);
}

template <typename Io, typename T>
requires Is<T, IR::DescriptorSource>
void Visit(Io& io, T& v) {
	io(v.dwords);
	io(v.dword_count);
	io(v.indirect_descriptor);
	io(v.bindless_sampler);
}

template <typename Io, typename T>
requires Is<T, IR::ResourceBlock>
void Visit(Io& io, T& v) {
	io(v.condition);
	io(v.successors);
	io(v.sources);
	io(v.srt_reads);
}

template <typename Io, typename T>
requires Is<T, IR::SrtRead>
void Visit(Io& io, T& v) {
	io(v.value);
	io(v.flat_offset);
}

template <typename Io, typename T>
requires Is<T, IR::BufferWrite>
void Visit(Io& io, T& v) {
	io(v.buffer);
	io(v.immediate);
	io(v.index);
	io(v.offset);
	io(v.soffset);
	io(v.predicate);
}

template <typename Io, typename T>
requires Is<T, IR::UniformFill>
void Visit(Io& io, T& v) {
	io(v.kind);
	io(v.resource);
	io(v.group_stride);
	io(v.words);
	io(v.value);
}

template <typename Io, typename T>
requires Is<T, IR::UniformFillPlan>
void Visit(Io& io, T& v) {
	io(v.fill);
	io(v.values);
}

template <typename Io, typename T>
requires Is<T, IR::ResourceSpecialization::Buffer>
void Visit(Io& io, T& v) {
	io(v.packed_stride);
	io(v.descriptor_format);
	io(v.descriptor_swizzle);
	io(v.zero_stride_oob);
	io(v.write_tracked);
	io(v.indirect_root);
	io(v.indirect_mapping_offset);
	io(v.indirect_search_iterations);
}

template <typename Io, typename T>
requires Is<T, IR::ResourceSpecialization::Image>
void Visit(Io& io, T& v) {
	io(v.numeric_class);
	io(v.dimension);
	io(v.mip_count);
	io(v.conversion_format);
	io(v.shader_swizzle);
	io(v.indirect_root);
	io(v.indirect_mapping_offset);
	io(v.indirect_search_iterations);
	io(v.cube);
	io(v.fmask);
	io(v.fmask_fragments);
	io(v.bindless);
	io(v.runtime);
	io(v.clamp_half_float);
}

template <typename Io, typename T>
requires Is<T, IR::ResourceSpecialization::Sampler>
void Visit(Io& io, T& v) {
	io(v.bindless);
	io(v.bindless_mapping_offset);
}

template <typename Io, typename T>
requires Is<T, IR::ResourceSpecialization>
void Visit(Io& io, T& v) {
	io(v.buffers);
	io(v.images);
	io(v.samplers);
}

// The plan's own fields besides its instructions (which EncodePlan/DecodePlan handle first).
template <typename Io, typename T>
requires Is<T, IR::ResourcePlan>
void VisitPlanFields(Io& io, T& v) {
	io(v.stage);
	io(v.shader_hash);
	io(v.user_data_base);
	io(v.user_data_count);
	io(v.memory_info);
	io(v.descriptor_sources);
	io(v.control_flow);
	io(v.srt_reads);
	io(v.buffer_writes_bounded);
	io(v.buffer_writes);
	io(v.clean_flat_slots);
	io(v.gpu_fill_slots);
	io(v.requires_specialization_memory);
	io(v.capture_specialization_reads);
	io(v.bindless_images);
	io(v.srt_plan_complete);
	io(v.resource_tracking_complete);
	io(v.info);
	io(v.uniform_fill);
}

// Tripwires (Windows x64, the layout the sizes were taken from): a field added to a serialized
// structure changes its size and stops the build here, so that the field gets added to its Visit
// above as well (Visit lists every field; a field it leaves out is silently lost on a cache hit).
// After updating the Visit, update the size. ResourcePlan's covers its scratch members too.
#if defined(_MSC_VER) && defined(_M_X64)
static_assert(sizeof(IR::MemoryInfo) == 72);
static_assert(sizeof(IR::BufferResource) == 72);
static_assert(sizeof(IR::ImageResource) == 88);
static_assert(sizeof(IR::SamplerResource) == 24);
static_assert(sizeof(IR::SampledResourcePair) == 12);
static_assert(sizeof(IR::StageInput) == 56);
static_assert(sizeof(IR::StageOutput) == 48);
static_assert(sizeof(IR::ShaderInfo) == 192);
static_assert(sizeof(IR::DescriptorBinding) == 32);
static_assert(sizeof(IR::BindingLayout) == 72);
static_assert(sizeof(IR::CompiledShaderInfo) == 304);
static_assert(sizeof(IR::DescriptorSource::IndirectDescriptor) == 120);
static_assert(sizeof(IR::DescriptorSource::BindlessSampler) == 4);
static_assert(sizeof(IR::DescriptorSource) == 272);
static_assert(sizeof(IR::ResourceBlock) == 88);
static_assert(sizeof(IR::SrtRead) == 24);
static_assert(sizeof(IR::BufferWrite) == 72);
static_assert(sizeof(IR::UniformFill) == 28);
static_assert(sizeof(IR::UniformFillPlan) == 96);
static_assert(sizeof(IR::ResourceSpecialization::Buffer) == 28);
static_assert(sizeof(IR::ResourceSpecialization::Image) == 40);
static_assert(sizeof(IR::ResourceSpecialization::Sampler) == 8);
static_assert(sizeof(IR::ResourceSpecialization) == 72);
static_assert(sizeof(IR::ResourcePlan) == 856);
#endif

struct Writer {
	ByteWriter&                                    out;
	const std::unordered_map<const IR::Inst*, uint32_t>* index = nullptr;
	bool                                           ok          = true;

	template <typename T>
	void operator()(const T& value) {
		if constexpr (std::is_same_v<T, bool>) {
			out.Put(value);
		} else if constexpr (std::is_enum_v<T>) {
			out.Put(static_cast<std::underlying_type_t<T>>(value));
		} else if constexpr (std::is_arithmetic_v<T>) {
			out.Put(value);
		} else if constexpr (IsVector<T>::value || IsArray<T>::value) {
			if constexpr (IsVector<T>::value) {
				out.Put<uint64_t>(value.size());
			}
			for (const auto& element: value) {
				(*this)(element);
			}
		} else if constexpr (std::is_same_v<T, std::string>) {
			out.Put<uint64_t>(value.size());
			out.PutBytes({reinterpret_cast<const uint8_t*>(value.data()), value.size()});
		} else if constexpr (IsOptional<T>::value) {
			out.Put(value.has_value());
			if (value.has_value()) {
				(*this)(*value);
			}
		} else if constexpr (std::is_same_v<T, IR::Value>) {
			if (auto* inst = value.TryInstruction(); inst != nullptr) {
				if (index == nullptr) {
					ok = false;
					return;
				}
				const auto found = index->find(inst);
				if (found == index->end()) {
					ok = false;
					return;
				}
				out.Put<uint8_t>(1);
				out.Put<uint32_t>(found->second);
			} else {
				out.Put<uint8_t>(0);
				out.Put(static_cast<uint32_t>(value.GetType()));
				out.Put<uint64_t>(value.ImmediateBits());
			}
		} else {
			Visit(*this, value);
		}
	}
};

struct Reader {
	ByteReader&                   in;
	const std::vector<IR::Inst*>* insts = nullptr;

	template <typename T>
	void operator()(T& value) {
		if (!in.Ok()) {
			return;
		}
		if constexpr (std::is_same_v<T, bool>) {
			(void)in.Get(value);
		} else if constexpr (std::is_enum_v<T>) {
			std::underlying_type_t<T> raw {};
			if (in.Get(raw)) {
				value = static_cast<T>(raw);
			}
		} else if constexpr (std::is_arithmetic_v<T>) {
			(void)in.Get(value);
		} else if constexpr (IsVector<T>::value) {
			uint64_t count = 0;
			if (!in.GetCount(count)) {
				return;
			}
			value.clear();
			value.resize(count);
			for (auto& element: value) {
				(*this)(element);
			}
		} else if constexpr (IsArray<T>::value) {
			for (auto& element: value) {
				(*this)(element);
			}
		} else if constexpr (std::is_same_v<T, std::string>) {
			uint64_t count = 0;
			if (!in.GetCount(count)) {
				return;
			}
			value.resize(count);
			for (auto& c: value) {
				uint8_t byte = 0;
				(void)in.Get(byte);
				c = static_cast<char>(byte);
			}
		} else if constexpr (IsOptional<T>::value) {
			bool present = false;
			if (!in.Get(present)) {
				return;
			}
			value.reset();
			if (present) {
				// Assigned rather than emplace()d: libstdc++ does not count a struct nested in a
				// class with default member initializers as default-constructible (Linux build).
				value = typename T::value_type {};
				(*this)(*value);
			}
		} else if constexpr (std::is_same_v<T, IR::Value>) {
			uint8_t tag = 0;
			if (!in.Get(tag)) {
				return;
			}
			if (tag == 1u) {
				uint32_t position = 0;
				if (!in.Get(position) || insts == nullptr || position >= insts->size()) {
					in.Fail();
					return;
				}
				value = IR::Value((*insts)[position]);
			} else if (tag == 0u) {
				uint32_t type = 0;
				uint64_t bits = 0;
				if (!in.Get(type) || !in.Get(bits) ||
				    type == static_cast<uint32_t>(IR::Type::Opaque) ||
				    !IR::Value::FromImmediateBits(static_cast<IR::Type>(type), bits, value)) {
					in.Fail();
				}
			} else {
				in.Fail();
			}
		} else {
			Visit(*this, value);
		}
	}
};

bool EncodePlan(const IR::ResourcePlan& plan, ByteWriter& out) {
	std::unordered_map<const IR::Inst*, uint32_t> index;
	index.reserve(plan.value_storage.size());
	for (const auto& inst: plan.value_storage) {
		index.emplace(&inst, static_cast<uint32_t>(index.size()));
	}
	Writer writer {.out = out, .index = &index};
	out.Put<uint64_t>(plan.value_storage.size());
	for (const auto& inst: plan.value_storage) {
		out.Put(static_cast<uint16_t>(inst.GetOpcode()));
		out.Put(inst.Flags<uint64_t>());
	}
	for (const auto& inst: plan.value_storage) {
		out.Put<uint32_t>(static_cast<uint32_t>(inst.NumArgs()));
		for (size_t arg = 0; arg < inst.NumArgs(); arg++) {
			writer(inst.Arg(arg));
		}
	}
	VisitPlanFields(writer, plan);
	return writer.ok;
}

bool DecodePlan(ByteReader& in, IR::ResourcePlan& plan) {
	uint64_t count = 0;
	if (!in.GetCount(count)) {
		return false;
	}
	std::vector<IR::Inst*> insts;
	insts.reserve(count);
	for (uint64_t i = 0; i < count; i++) {
		uint16_t opcode = 0;
		uint64_t flags  = 0;
		if (!in.Get(opcode) || !in.Get(flags) ||
		    opcode >= static_cast<uint16_t>(IR::ValueOpcode::Count)) {
			return in.Fail();
		}
		insts.push_back(&plan.value_storage.emplace_back(static_cast<IR::ValueOpcode>(opcode), flags));
	}
	Reader reader {.in = in, .insts = &insts};
	for (auto* inst: insts) {
		uint32_t args = 0;
		if (!in.Get(args)) {
			return false;
		}
		const bool phi = inst->GetOpcode() == IR::ValueOpcode::Phi;
		if (!phi && args != inst->NumArgs()) {
			return in.Fail();
		}
		if (phi && args > 0xffffu) {
			return in.Fail();
		}
		for (uint32_t arg = 0; arg < args; arg++) {
			IR::Value value;
			reader(value);
			if (!in.Ok()) {
				return false;
			}
			if (phi) {
				inst->AddPhiOperand(nullptr, value);
			} else {
				inst->SetArg(arg, value);
			}
		}
	}
	VisitPlanFields(reader, plan);
	return in.Ok();
}

std::string HexKey(std::span<const uint8_t> key) {
	const auto hash = XXH3_128bits(key.data(), key.size());
	return fmt::format("{:016x}{:016x}", hash.high64, hash.low64);
}

template <typename... Args>
void CacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadFile(const std::filesystem::path& path, std::vector<uint8_t>& bytes) {
	std::FILE* file = nullptr;
#ifdef _WIN32
	if (_wfopen_s(&file, path.c_str(), L"rb") != 0) {
		file = nullptr;
	}
#else
	file = std::fopen(path.c_str(), "rb");
#endif
	if (file == nullptr) {
		return false;
	}
	bool ok = std::fseek(file, 0, SEEK_END) == 0;
	const auto size = ok ? std::ftell(file) : -1;
	ok = ok && size > 0 && static_cast<uint64_t>(size) <= MaxFileSize &&
	     std::fseek(file, 0, SEEK_SET) == 0;
	if (ok) {
		bytes.resize(static_cast<size_t>(size));
		ok = std::fread(bytes.data(), 1, bytes.size(), file) == bytes.size();
	}
	std::fclose(file);
	return ok;
}

bool WriteFile(const std::filesystem::path& path, std::span<const uint8_t> bytes) {
	auto temp = path;
	temp += ".tmp";
	std::FILE* file = nullptr;
#ifdef _WIN32
	if (_wfopen_s(&file, temp.c_str(), L"wb") != 0) {
		file = nullptr;
	}
#else
	file = std::fopen(temp.c_str(), "wb");
#endif
	if (file == nullptr) {
		return false;
	}
	const bool written = std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
	const bool closed  = std::fclose(file) == 0;
	std::error_code error;
	if (!written || !closed) {
		std::filesystem::remove(temp, error);
		return false;
	}
	std::filesystem::rename(temp, path, error);
	if (error) {
		std::filesystem::remove(temp, error);
		return false;
	}
	return true;
}

} // namespace

uint64_t SourceVersion() {
	return ShaderDiskCacheVersion::SourceVersion;
}

bool VerifyEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SHADER_DISK_CACHE_VERIFY");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

std::vector<uint8_t> SessionKey(const SessionInfo& session) {
	ByteWriter key;
	key.Put(FileMagic);
	key.Put(FormatVersion);
	key.Put(ShaderDiskCacheVersion::SourceVersion);
	key.Put(session.vendor_id);
	key.Put(session.device_id);
	key.Put(session.bindless_images);
	key.Put(session.float_image_atomics);
	// The emitter's device switches (vulkanWindow.cpp sets them before any shader is compiled).
	key.Put(ShaderRecompiler::Spirv::GetCoherentLoadAcquire());
	// Round 116: the atomic modes of coherent loads have sessions of their own.
	if (const auto mode = ShaderRecompiler::Spirv::GetCoherentLoadMode();
	    mode == ShaderRecompiler::Spirv::CoherentLoadMode::Acquire ||
	    mode == ShaderRecompiler::Spirv::CoherentLoadMode::Relaxed) {
		key.Put(static_cast<uint32_t>(mode));
	}
	key.Put(ShaderRecompiler::Spirv::GetDeviceClockShift());
	key.Put(ShaderRecompiler::Spirv::GetHostImageFeatures().min_lod);
	key.Put(ShaderRecompiler::Spirv::GetHostFloatFeatures().denorm_flush_f32);
	// Every KYTY_* switch named in the recompiler's sources, with its value.
	for (const auto name: ShaderDiskCacheVersion::RecompilerSwitches) {
		if (name.empty()) {
			continue;
		}
		const std::string name_text(name);
		const char*       value = std::getenv(name_text.c_str());
		key.Put<uint64_t>(name.size());
		key.PutBytes({reinterpret_cast<const uint8_t*>(name.data()), name.size()});
		key.Put(value != nullptr);
		if (value != nullptr) {
			const std::string_view text(value);
			key.Put<uint64_t>(text.size());
			key.PutBytes({reinterpret_cast<const uint8_t*>(text.data()), text.size()});
		}
	}
	return std::move(key.bytes);
}

void AppendInputKey(ByteWriter& key, const ShaderVertexInputInfo& info) {
	key.Put<uint8_t>('V');
	key.Put(static_cast<uint32_t>(info.logical_stage));
	key.Put<int32_t>(info.resources_num);
	key.Put<int32_t>(info.fetch_attrib_reg);
	key.Put<int32_t>(info.fetch_buffer_reg);
	key.Put(info.wave_size);
	key.Put(info.scratch_size_dwords);
	key.Put(info.pa_cl_vs_out_cntl);
	key.Put(info.clip_space.enabled);
	if (info.clip_space.enabled) {
		for (int i = 0; i < 2; i++) {
			key.Put(info.clip_space.scale[i]);
			key.Put(info.clip_space.offset[i]);
			key.Put(info.clip_space.half_extent[i]);
		}
	}
	const auto& mesh = info.mesh;
	for (const auto value: mesh.threads_num) {
		key.Put(value);
	}
	for (const auto value: {mesh.lds_size_dwords, mesh.scratch_size_dwords, mesh.host_subgroup_size,
	                        mesh.wave_size, mesh.input_primitive, mesh.primitives_per_group,
	                        mesh.vertices_per_group, mesh.max_vertices, mesh.max_primitives,
	                        mesh.provoking_vertex, mesh.passes}) {
		key.Put(value);
	}
	key.Put(mesh.fast_launch);
	const auto& tess = info.tess;
	for (const auto value:
	     {tess.input_control_points, tess.output_control_points, tess.ls_stride, tess.hs_stride,
	      tess.domain, tess.partitioning, tess.output_topology, tess.patch_base, tess.patch_size,
	      tess.min_level_bits, tess.max_level_bits}) {
		key.Put(value);
	}
	key.Put(info.fetch_external);
	key.Put(info.fetch_embedded);
	const int resources = std::clamp(info.resources_num, 0, ShaderVertexInputInfo::RES_MAX);
	for (int i = 0; i < resources; i++) {
		// The embedded fetch reads the V#'s format and destination selects (BuildStageStaticKey).
		key.Put(info.resources[i].fields[3] & 0x7ffffu);
		key.Put<int32_t>(info.resources_dst[i].registers_num);
		key.Put<int32_t>(info.resources_dst[i].attr_id);
	}
}

void AppendInputKey(ByteWriter& key, const ShaderPixelInputInfo& info) {
	key.Put<uint8_t>('P');
	const auto inputs = std::min<uint32_t>(info.input_num, std::size(info.interpolator_settings));
	key.Put(info.input_num);
	for (uint32_t i = 0; i < inputs; i++) {
		key.Put(info.interpolator_settings[i]);
	}
	for (const auto value: {info.wave_size, info.ps_system_input_base, info.custom_interpolation_mask,
	                        info.ps_perspective_center_vgpr, info.ps_perspective_centroid_vgpr,
	                        info.target_uint_mask, info.target_sint_mask,
	                        info.target_half_float_mask, info.scratch_size_dwords}) {
		key.Put(value);
	}
	for (const auto mode: info.target_output_mode) {
		key.Put(mode);
	}
	for (const auto& mapping: info.target_export_mapping) {
		key.Put(mapping.packed);
	}
	for (const bool flag: {info.ps_pos_x, info.ps_pos_y, info.ps_pos_z, info.ps_pos_w,
	                       info.ps_front_face, info.ps_ancillary, info.ps_no_perspective,
	                       info.ps_pixel_kill_enable, info.ps_depth_export_enable,
	                       info.ps_sample_mask_export_enable, info.ps_sample_shading,
	                       info.dual_source_blending, info.ps_early_z, info.ps_execute_on_noop}) {
		key.Put(flag);
	}
	key.Put(static_cast<uint8_t>(info.alpha_blend_source));
}

void AppendInputKey(ByteWriter& key, const ShaderComputeInputInfo& info) {
	key.Put<uint8_t>('C');
	for (const auto value: info.threads_num) {
		key.Put(value);
	}
	for (const auto value: {info.lds_size_dwords, info.scratch_size_dwords, info.host_subgroup_size,
	                        info.wave_size}) {
		key.Put(value);
	}
	key.Put(info.float_mode);
	for (const bool value: info.group_id) {
		key.Put(value);
	}
	key.Put(info.dispatch_thread_dimensions);
	key.Put(info.lds_storage);
	key.Put(info.dispatch_dimensions_indirect);
	key.Put<int32_t>(info.thread_ids_num);
	key.Put<int32_t>(info.workgroup_register);
	key.Put(info.tg_size_en);
}

void AppendSpecialization(ByteWriter& key, const IR::ResourceSpecialization& value) {
	Writer writer {.out = key};
	writer(value);
}

bool ReadSpecialization(ByteReader& in, IR::ResourceSpecialization& value) {
	Reader reader {.in = in};
	reader(value);
	return in.Ok();
}

bool EncodeSourceRecord(bool unsupported, bool uses_clock, const IR::ResourcePlan* plan,
                        std::vector<uint8_t>& out) {
	ByteWriter writer;
	writer.Put(unsupported);
	writer.Put(uses_clock);
	writer.Put(plan != nullptr);
	if (plan != nullptr && !EncodePlan(*plan, writer)) {
		return false;
	}
	out = std::move(writer.bytes);
	return true;
}

bool DecodeSourceRecord(std::span<const uint8_t> bytes, SourceRecord& record) {
	ByteReader reader(bytes);
	bool       has_plan = false;
	if (!reader.Get(record.unsupported) || !reader.Get(record.uses_clock) || !reader.Get(has_plan)) {
		return false;
	}
	if (has_plan == record.unsupported) {
		return false;
	}
	if (has_plan && !DecodePlan(reader, record.plan)) {
		return false;
	}
	return reader.AtEnd();
}

void EncodePermutationRecord(std::span<const uint32_t> spirv, const IR::CompiledShaderInfo& program,
                             std::vector<uint8_t>& out) {
	ByteWriter writer;
	writer.PutWords(spirv);
	Writer visit {.out = writer};
	visit(program);
	out = std::move(writer.bytes);
}

bool DecodePermutationRecord(std::span<const uint8_t> bytes, PermutationRecord& record) {
	ByteReader reader(bytes);
	if (!reader.GetWords(record.spirv) || record.spirv.empty()) {
		return false;
	}
	Reader visit {.in = reader};
	visit(record.program);
	return reader.AtEnd();
}

bool SourceRecordRoundTrips(std::span<const uint8_t> bytes) {
	SourceRecord         record;
	std::vector<uint8_t> again;
	return DecodeSourceRecord(bytes, record) &&
	       EncodeSourceRecord(record.unsupported, record.uses_clock,
	                          record.unsupported ? nullptr : &record.plan, again) &&
	       std::ranges::equal(bytes, again);
}

bool PermutationRecordRoundTrips(std::span<const uint8_t> bytes) {
	PermutationRecord    record;
	std::vector<uint8_t> again;
	if (!DecodePermutationRecord(bytes, record)) {
		return false;
	}
	EncodePermutationRecord(record.spirv, record.program, again);
	return std::ranges::equal(bytes, again);
}

std::unique_ptr<Store> Store::Open(std::filesystem::path directory) {
	if (const char* value = std::getenv("KYTY_SHADER_DISK_CACHE");
	    value != nullptr && value[0] == '0') {
		CacheLog("Shader disk cache: off (KYTY_SHADER_DISK_CACHE=0)");
		return nullptr;
	}
	if (directory.empty()) {
		CacheLog("Shader disk cache: off (no title id)");
		return nullptr;
	}
	CacheLog("Shader disk cache: {} (recompiler version {:016x}){}",
	         directory.generic_string(), ShaderDiskCacheVersion::SourceVersion,
	         VerifyEnabled() ? ", verifying every hit" : "");
	return std::make_unique<Store>(std::move(directory));
}

Store::Store(std::filesystem::path directory)
    : m_directory(std::move(directory)), m_writer([this] { WriterLoop(); }) {}

Store::~Store() {
	{
		std::lock_guard lock(m_mutex);
		m_stop = true;
	}
	m_wake.notify_all();
	m_writer.join();
	LogTotals("end");
}

std::filesystem::path Store::EntryPath(std::span<const uint8_t> key) const {
	return m_directory / (HexKey(key) + ".bin");
}

bool Store::Load(std::span<const uint8_t> key, std::vector<uint8_t>& payload) {
	std::vector<uint8_t> file;
	if (!ReadFile(EntryPath(key), file)) {
		return false;
	}
	counters.bytes_read += file.size();
	ByteReader reader(file);
	uint32_t   magic        = 0;
	uint32_t   format       = 0;
	uint64_t   key_size     = 0;
	uint64_t   payload_size = 0;
	uint64_t   payload_hash = 0;
	constexpr size_t header = sizeof(magic) + sizeof(format) + 3 * sizeof(uint64_t);
	const bool valid =
	    reader.Get(magic) && reader.Get(format) && reader.Get(key_size) &&
	    reader.Get(payload_size) && reader.Get(payload_hash) && magic == FileMagic &&
	    format == FormatVersion && key_size == key.size() && file.size() >= header + key_size &&
	    payload_size == file.size() - header - key_size &&
	    std::memcmp(file.data() + header, key.data(), key.size()) == 0 &&
	    XXH3_64bits(file.data() + header + key_size, payload_size) == payload_hash;
	if (!valid) {
		counters.rejected++;
		return false;
	}
	payload.assign(file.begin() + static_cast<std::ptrdiff_t>(header + key_size), file.end());
	return true;
}

void Store::Save(std::span<const uint8_t> key, std::span<const uint8_t> payload) {
	ByteWriter file;
	file.Put(FileMagic);
	file.Put(FormatVersion);
	file.Put<uint64_t>(key.size());
	file.Put<uint64_t>(payload.size());
	file.Put<uint64_t>(XXH3_64bits(payload.data(), payload.size()));
	file.PutBytes(key);
	file.PutBytes(payload);
	{
		std::lock_guard lock(m_mutex);
		m_queue.push_back({.path = EntryPath(key), .bytes = std::move(file.bytes)});
	}
	m_wake.notify_one();
}

void Store::Flush() {
	std::unique_lock lock(m_mutex);
	m_idle.wait(lock, [this] { return m_queue.empty() && !m_writing; });
}

void Store::WriterLoop() {
	std::unique_lock lock(m_mutex);
	for (;;) {
		m_wake.wait(lock, [this] { return m_stop || !m_queue.empty(); });
		if (m_queue.empty()) {
			// Stopping with nothing left to write.
			break;
		}
		auto pending = std::move(m_queue.front());
		m_queue.pop_front();
		m_writing = true;
		lock.unlock();
		bool ok = true;
		if (!m_directory_ready) {
			std::error_code error;
			std::filesystem::create_directories(m_directory, error);
			ok                = !error || std::filesystem::is_directory(m_directory, error);
			m_directory_ready = ok;
		}
		ok = ok && WriteFile(pending.path, pending.bytes);
		if (ok) {
			counters.writes++;
			counters.bytes_written += pending.bytes.size();
		} else if (counters.write_failures++ == 0) {
			CacheLog("Shader disk cache: failed to write {}", pending.path.generic_string());
		}
		lock.lock();
		m_writing = false;
		if (m_queue.empty()) {
			m_idle.notify_all();
		}
	}
	m_idle.notify_all();
}

void Store::LogTotals(std::string_view when) {
	CacheLog("Shader disk cache ({}): sources {} hits / {} misses, permutations {} hits / {} "
	         "misses, {} rejected, {} not encodable, {} written ({} KB), {} KB read{}",
	         when, counters.source_hits.load(), counters.source_misses.load(),
	         counters.permutation_hits.load(), counters.permutation_misses.load(),
	         counters.rejected.load(), counters.round_trip_failures.load(), counters.writes.load(),
	         counters.bytes_written.load() / 1024u, counters.bytes_read.load() / 1024u,
	         VerifyEnabled() ? fmt::format(", verified {} / {} mismatches", counters.verified.load(),
	                                       counters.verify_mismatches.load())
	                         : std::string());
}

} // namespace Libs::Graphics::ShaderDiskCache
