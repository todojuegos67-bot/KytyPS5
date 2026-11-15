#include "common/assert.h"
#include "common/upstreamPaths.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "gpu_blit_shaders/gpu_mesh_indirect_spv.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <chrono>
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <mutex>
#include <span>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

// IDK: maybe we can remove it?
constexpr uint8_t kTemporaryVertexAttribFormat113 =
    static_cast<uint8_t>(Prospero::VertexAttribFormat::k16_16SInt);
constexpr uint32_t kTemporaryPs5BufferFormat121 = 121u;

static bool NarrowInputFormat(vk::Format& format, uint32_t& size, uint32_t used_components) {
	if (used_components == 0 || used_components >= size) {
		return false;
	}

	switch (format) {
		case vk::Format::eR32G32B32A32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				case 3: format = vk::Format::eR32G32B32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR32G32B32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR16G16B16A16Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR16Sfloat; break;
				case 2: format = vk::Format::eR16G16Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Unorm:
			switch (used_components) {
				case 1: format = vk::Format::eR8Unorm; break;
				case 2: format = vk::Format::eR8G8Unorm; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Snorm:
			if (used_components != 2) {
				return false;
			}
			format = vk::Format::eR8G8Snorm;
			size   = 2;
			return true;
		case vk::Format::eR8G8B8A8Uint:
			switch (used_components) {
				case 1: format = vk::Format::eR8Uint; break;
				case 2: format = vk::Format::eR8G8Uint; break;
				default: return false;
			}
			size = used_components;
			return true;
		default: break;
	}

	return false;
}

static void GetInputFormat(const ShaderBufferResource& res, vk::Format& format, uint32_t& size,
                           uint32_t used_components) {
	const auto fmt        = res.Format();
	const auto raw_format = res.RawFormat();
	if (raw_format == kTemporaryVertexAttribFormat113) {
		static bool logged_113 = false;
		if (!logged_113) {
			LOGF("InputFormat: temporary: accepting invalid PS5 buffer format 113 as "
			     "vk::Format::eR32G32B32A32Sfloat\n");
			logged_113 = true;
		}
		format = vk::Format::eR32G32B32A32Sfloat;
		size   = 4;
		if (NarrowInputFormat(format, size, used_components)) {
			LOGF("InputFormat: narrowing fmt=%u to %s for used_components=%u\n", raw_format,
			     vk::to_string(format).c_str(), used_components);
		}
		return;
	}
	if (raw_format == kTemporaryPs5BufferFormat121) {
		static bool logged_121 = false;
		if (!logged_121) {
			LOGF("InputFormat: accepting PS5 buffer format 121 as vk::Format::eR16G16Sfloat\n");
			logged_121 = true;
		}
		format = vk::Format::eR16G16Sfloat;
		size   = 2;
		return;
	}

	format = VulkanFormat(fmt);
	size   = ShaderRecompiler::Format::GetFormatInfo(fmt).component_count;
	if (format == vk::Format::eUndefined || size == 0) {
		EXIT("unknown vertex format: fmt = %u\n", raw_format);
	}

	if (NarrowInputFormat(format, size, used_components)) {
		static std::atomic<uint64_t> log_count = 0;
		auto                         log_id    = log_count.fetch_add(1);
		if (log_id < 32) {
			LOGF("VertexInput: narrowed vertex format to %" PRIu32
			     " component(s) for shader fetch\n",
			     used_components);
		}
	}
}

static vk::BlendFactor GetBlendFactor(uint32_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kZero: return vk::BlendFactor::eZero;
		case Prospero::BlendFactor::kOne: return vk::BlendFactor::eOne;
		case Prospero::BlendFactor::kSrcColor: return vk::BlendFactor::eSrcColor;
		case Prospero::BlendFactor::kOneMinusSrcColor: return vk::BlendFactor::eOneMinusSrcColor;
		case Prospero::BlendFactor::kSrcAlpha: return vk::BlendFactor::eSrcAlpha;
		case Prospero::BlendFactor::kOneMinusSrcAlpha: return vk::BlendFactor::eOneMinusSrcAlpha;
		case Prospero::BlendFactor::kDstAlpha: return vk::BlendFactor::eDstAlpha;
		case Prospero::BlendFactor::kOneMinusDstAlpha: return vk::BlendFactor::eOneMinusDstAlpha;
		case Prospero::BlendFactor::kDstColor: return vk::BlendFactor::eDstColor;
		case Prospero::BlendFactor::kOneMinusDstColor: return vk::BlendFactor::eOneMinusDstColor;
		case Prospero::BlendFactor::kSrcAlphaSaturate: return vk::BlendFactor::eSrcAlphaSaturate;
		case Prospero::BlendFactor::kConstantColor: return vk::BlendFactor::eConstantColor;
		case Prospero::BlendFactor::kOneMinusConstantColor:
			return vk::BlendFactor::eOneMinusConstantColor;
		case Prospero::BlendFactor::kSrc1Color: return vk::BlendFactor::eSrc1Color;
		case Prospero::BlendFactor::kOneMinusSrc1Color: return vk::BlendFactor::eOneMinusSrc1Color;
		case Prospero::BlendFactor::kSrc1Alpha: return vk::BlendFactor::eSrc1Alpha;
		case Prospero::BlendFactor::kOneMinusSrc1Alpha: return vk::BlendFactor::eOneMinusSrc1Alpha;
		case Prospero::BlendFactor::kConstantAlpha: return vk::BlendFactor::eConstantAlpha;
		case Prospero::BlendFactor::kOneMinusConstantAlpha:
			return vk::BlendFactor::eOneMinusConstantAlpha;
		default: EXIT("unknown factor: %u\n", factor);
	}
	return vk::BlendFactor::eZero;
}

static vk::BlendOp GetBlendOp(uint32_t op) {
	switch (static_cast<Prospero::BlendOp>(op)) {
		case Prospero::BlendOp::kAdd: return vk::BlendOp::eAdd;
		case Prospero::BlendOp::kSubtract: return vk::BlendOp::eSubtract;
		case Prospero::BlendOp::kMin: return vk::BlendOp::eMin;
		case Prospero::BlendOp::kMax: return vk::BlendOp::eMax;
		case Prospero::BlendOp::kReverseSubtract: return vk::BlendOp::eReverseSubtract;
		default: EXIT("unknown op: %u\n", op);
	}
	return vk::BlendOp::eAdd;
}

static void AddLayoutBindings(std::vector<vk::DescriptorSetLayoutBinding>& descriptor_bindings,
                              const ShaderRecompiler::IR::CompiledShaderInfo& program,
                              vk::ShaderStageFlagBits              stage) {
	for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
		const auto index = std::countr_zero(mask);
		const auto kind = static_cast<ShaderRecompiler::IR::DescriptorBindingKind>(index);
		descriptor_bindings.push_back(
		    {ShaderRecompiler::IR::NativeBinding(program.stage, kind), NativeDescriptorType(kind),
		     program.bindings.descriptor_counts[index], stage, nullptr});
	}
}

// A program's image groups in its group set (IR::GroupSetFor): a binding an image kind, as the emitter defines them.
static std::vector<vk::DescriptorSetLayoutBinding> GroupBindings(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                                               vk::ShaderStageFlags                            stages) {
	std::vector<vk::DescriptorSetLayoutBinding> bindings;
	const auto&                                 counts = program.bindings.group_counts;
	for (uint32_t index = 0; index < counts.size(); ++index) {
		if (counts[index] == 0u) continue;
		const auto kind = static_cast<ShaderRecompiler::IR::DescriptorBindingKind>(index + ShaderRecompiler::IR::FirstImageBinding);
		bindings.push_back({ShaderRecompiler::IR::NativeBinding(program.stage, kind), NativeDescriptorType(kind), counts[index],
		                    stages, nullptr});
	}
	return bindings;
}

// Set numbers a layout skips (no group set below the last) take the empty layout, made on first use (the descriptor
// heap destroys it).
static void FillSetHoles(GraphicContext& graphics, std::span<vk::DescriptorSetLayout> sets) {
	{
		static std::mutex mutex;
		std::lock_guard   lock(mutex);
		if (graphics.empty_set_layout == nullptr) {
			vk::DescriptorSetLayoutCreateInfo empty {};
			EXIT_IF(graphics.device.createDescriptorSetLayout(&empty, nullptr, &graphics.empty_set_layout) !=
			        vk::Result::eSuccess);
		}
	}
	for (auto& set: sets) {
		if (set == nullptr) set = graphics.empty_set_layout;
	}
}

static void CreateDescriptorLayout(GraphicContext& graphics, vk::DescriptorSetLayout& layout, bool& uses_push_descriptors,
                                   std::span<const vk::DescriptorSetLayoutBinding> bindings, bool allow_push = true);

static void CreateGroupSetLayout(GraphicContext& graphics, vk::DescriptorSetLayout& layout,
                                 std::span<const vk::DescriptorSetLayoutBinding> bindings) {
	if (bindings.empty()) return;
	bool pushed = false;
	CreateDescriptorLayout(graphics, layout, pushed, bindings, false);
}

static void CreateDescriptorLayout(GraphicContext& graphics, vk::DescriptorSetLayout& layout, bool& uses_push_descriptors,
                                   std::span<const vk::DescriptorSetLayoutBinding> bindings, bool allow_push) {
	const auto& limits = graphics.GetPhysicalDeviceProperties().limits;
	uint64_t descriptor_count = 0;
	uint64_t storage_buffers = 0;
	uint64_t samplers = 0;
	std::array<uint64_t, 32> stage_buffers {}, stage_resources {}, stage_samplers {};
	for (const auto& binding: bindings) {
		descriptor_count += binding.descriptorCount;
		const bool storage = binding.descriptorType == vk::DescriptorType::eStorageBuffer;
		const bool sampler = binding.descriptorType == vk::DescriptorType::eSampler;
		if (storage) storage_buffers += binding.descriptorCount;
		if (sampler) samplers += binding.descriptorCount;
		for (auto stages = static_cast<uint32_t>(binding.stageFlags); stages != 0; stages &= stages - 1u) {
			const auto stage = std::countr_zero(stages);
			if (storage) stage_buffers[stage] += binding.descriptorCount;
			if (sampler) stage_samplers[stage] += binding.descriptorCount;
			else
				stage_resources[stage] += binding.descriptorCount;
			if (stage_buffers[stage] > limits.maxPerStageDescriptorStorageBuffers ||
			    stage_resources[stage] > limits.maxPerStageResources ||
			    stage_samplers[stage] > limits.maxPerStageDescriptorSamplers) {
				EXIT("native descriptor stage %u exceeds device limits: buffers=%" PRIu64
				     "/%u resources=%" PRIu64 "/%u samplers=%" PRIu64 "/%u\n", stage, stage_buffers[stage],
				     limits.maxPerStageDescriptorStorageBuffers, stage_resources[stage], limits.maxPerStageResources,
				     stage_samplers[stage], limits.maxPerStageDescriptorSamplers);
			}
		}
	}
	if (storage_buffers > limits.maxDescriptorSetStorageBuffers) {
		EXIT("native storage-buffer descriptors exceed device limit: %" PRIu64 "/%u\n",
		     storage_buffers, limits.maxDescriptorSetStorageBuffers);
	}
	if (samplers > limits.maxDescriptorSetSamplers) {
		EXIT("native sampler descriptors exceed device limit: %" PRIu64 "/%u\n",
		     samplers, limits.maxDescriptorSetSamplers);
	}
	uses_push_descriptors = allow_push && descriptor_count <= graphics.max_push_descriptors;

	vk::DescriptorSetLayoutCreateInfo create {};
	create.flags        = uses_push_descriptors
	                          ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	                          : vk::DescriptorSetLayoutCreateFlags {};
	create.bindingCount = static_cast<uint32_t>(bindings.size());
	create.pBindings    = bindings.data();
	// Image arrays are partially bound: a group of indirect candidates (ImageResource::indirect_capacity)
	// writes only its candidates, not null descriptors up to its capacity.
	std::vector<vk::DescriptorBindingFlags> binding_flags;
	vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info {};
	if (!uses_push_descriptors) {
		for (const auto& binding: bindings) {
			const bool image = binding.descriptorType == vk::DescriptorType::eSampledImage ||
			                   binding.descriptorType == vk::DescriptorType::eStorageImage;
			binding_flags.push_back(image && binding.descriptorCount > 1u
			                            ? vk::DescriptorBindingFlags {vk::DescriptorBindingFlagBits::ePartiallyBound}
			                            : vk::DescriptorBindingFlags {});
		}
		flags_info.bindingCount  = static_cast<uint32_t>(binding_flags.size());
		flags_info.pBindingFlags = binding_flags.data();
		create.pNext             = &flags_info;
	}
	EXIT_IF(graphics.device.createDescriptorSetLayout(
	            &create, nullptr, &layout) != vk::Result::eSuccess);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
vk::Pipeline PipelineLibraries::Get(uint64_t key, const std::function<vk::Pipeline()>& build) {
	{
		std::unique_lock lock(m_mutex);
		m_built.wait(lock, [&] { return !m_building.contains(key); });
		if (const auto found = m_parts.find(key); found != m_parts.end()) return found->second;
		m_building.insert(key);
	}
	vk::Pipeline part = nullptr;
	try {
		part = build();
	} catch (...) {
		std::lock_guard lock(m_mutex);
		m_building.erase(key);
		m_built.notify_all();
		throw;
	}
	std::lock_guard lock(m_mutex);
	m_parts.emplace(key, part);
	m_building.erase(key);
	m_built.notify_all();
	return part;
}

void PipelineLibraries::Keep(vk::PipelineLayout layout, std::initializer_list<vk::DescriptorSetLayout> sets) {
	std::lock_guard lock(m_mutex);
	m_layouts.push_back(layout);
	for (const auto set: sets) {
		if (set != nullptr) m_sets.push_back(set);
	}
}

void PipelineLibraries::Destroy(vk::Device device) {
	std::lock_guard lock(m_mutex);
	for (const auto& [key, part]: m_parts) device.destroyPipeline(part, nullptr);
	for (const auto layout: m_layouts) device.destroyPipelineLayout(layout, nullptr);
	for (const auto set: m_sets) device.destroyDescriptorSetLayout(set, nullptr);
	m_parts.clear();
	m_layouts.clear();
	m_sets.clear();
}

// A library part's key: the fields its state is built from, hashed with the part's kind.
// shortcut: a 64-bit hash, not the fields compared; a collision would link another part (about 2^-40 for a run's
// few thousand parts).
class LibraryKey {
public:
	explicit LibraryKey(uint64_t kind) { Add(kind); }
	LibraryKey& Add(uint64_t value) {
		m_words.push_back(value);
		return *this;
	}
	[[nodiscard]] uint64_t Hash() const { return XXH3_64bits(m_words.data(), m_words.size() * sizeof(uint64_t)); }

private:
	std::vector<uint64_t> m_words;
};

// Whether a vertex+pixel pipeline is linked from library parts (KYTY_GPL, GraphicContext::SplitsFragmentSet). Built whole
// instead:
// - tessellation and rect-list pipelines (shortcut: for now);
// - mesh pipelines.
// Pixel shaders with per-vertex inputs (VK_KHR_fragment_shader_barycentric PerVertexKHR) are linked from parts made
// with the whole layout (ReadsPerVertex).
bool BuildsFromLibraries(const GraphicContext& graphics, std::span<const ShaderVertexInputInfo> vertex_info,
                         const ShaderPixelInputInfo* ps_input_info, const PipelineCache::GraphicsPrograms& programs,
                         const PipelineStaticParameters& static_params) {
	if (ps_input_info == nullptr || !graphics.SplitsFragmentSet()) return false;
	const bool mesh         = vertex_info.front().stage.program->stage == ShaderType::Mesh;
	const bool tessellation = vertex_info.size() == 3;
	const bool rect_list    = !mesh && !tessellation && static_params.topology == vk::PrimitiveTopology::ePatchList;
	return !mesh && !tessellation && !rect_list && programs.vertex[0].spirv_hash != 0 && programs.pixel.spirv_hash != 0;
}

// A pipeline whose pixel shader reads per-vertex inputs is linked from parts made with its whole layout, without
// VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT. With independent-sets layouts, whole or from parts, the fire's
// gobo light in Marvel's Wolverine (PS 31e247c3beb225d2) drew dark and the device was lost (NVIDIA 616.92 and 621.01);
// with the whole layout it is right. The cause in the driver is not known: offline cases with per-vertex inputs, the
// same three sets, push descriptors and push constants are exact either way.
static bool ReadsPerVertex(const ShaderPixelInputInfo& ps_input_info) {
	return std::ranges::any_of(ps_input_info.stage.program->info.inputs,
	                           [](const ShaderRecompiler::IR::StageInput& input) { return input.per_vertex; });
}


void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache,
                            PipelineLibraries*                     libraries) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	const bool  tessellation   = vertex_info.size() == 3;
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(!vertex_program || (ps_active && !pixel_program));
	const bool with_depth = rendering.depth_format != vk::Format::eUndefined ||
	                        rendering.stencil_format != vk::Format::eUndefined;
	EXIT_IF(!vs_input_info.stage);
	const bool mesh = vs_input_info.stage.program->stage == ShaderType::Mesh;
	EXIT_NOT_IMPLEMENTED(mesh && !graphics.mesh_shader_enabled);
	const bool rect_list =
	    !mesh && !tessellation && static_params.topology == vk::PrimitiveTopology::ePatchList;

	vk::ShaderModule tess_control_shader_module = nullptr;
	vk::ShaderModule tess_eval_shader_module    = nullptr;
	// Upstream abc509d33: an indirect mesh draw's records are read by a task shader (gpu_mesh_indirect.task).
	const bool mesh_task = mesh && vs_input_info.indirect_draw;
	EXIT_NOT_IMPLEMENTED(mesh_task && !graphics.task_shader_enabled);
	const auto task_module =
	    mesh_task ? CompileSPV(GPU_MESH_INDIRECT_SPV, graphics.device) : vk::ShaderModule {};
	pipeline.task_stage = mesh_task;

	if (rect_list) {
		const auto shaders =
		    BuildRectListShaders(vs_input_info, ps_active ? ps_input_info : nullptr);
		tess_control_shader_module = CompileSPV(shaders.control, graphics.device);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TCS done module=%p\n",
			     static_cast<void*>(tess_control_shader_module));
		}

		tess_eval_shader_module = CompileSPV(shaders.evaluation, graphics.device);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TES done module=%p\n",
			     static_cast<void*>(tess_eval_shader_module));
		}
	}

	EXIT_NOT_IMPLEMENTED(rect_list &&
	                     (tess_control_shader_module == nullptr || tess_eval_shader_module == nullptr) &&
	                     !KYTY_UPSTREAM_PATH("4626e7fb7"));

	vk::PipelineShaderStageCreateInfo shader_stages[4] {};
	uint32_t                          shader_stage_count = 0;
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		shader_stages[shader_stage_count++] = {.stage =
		                                           NativeShaderStage(vertex_info[i].logical_stage),
		                                       .module = programs.vertex[i].module,
		                                       .pName  = "main"};
	}
	if (task_module) {
		shader_stages[shader_stage_count++] = {.stage  = vk::ShaderStageFlagBits::eTaskEXT,
		                                       .module = task_module,
		                                       .pName  = "main"};
	}
	if (rect_list) {
		shader_stages[shader_stage_count++] = {.stage =
		                                           vk::ShaderStageFlagBits::eTessellationControl,
		                                       .module = tess_control_shader_module,
		                                       .pName  = "main"};
		shader_stages[shader_stage_count++] = {.stage =
		                                           vk::ShaderStageFlagBits::eTessellationEvaluation,
		                                       .module = tess_eval_shader_module,
		                                       .pName  = "main"};
	}
	if (ps_active) {
		shader_stages[shader_stage_count++] = {.stage  = vk::ShaderStageFlagBits::eFragment,
		                                       .module = pixel_program.module,
		                                       .pName  = "main"};
	}

	vk::VertexInputAttributeDescription input_attr[ShaderVertexInputInfo::RES_MAX] {};
	vk::VertexInputBindingDescription   input_desc[ShaderVertexInputInfo::RES_MAX] {};

	for (uint32_t binding = 0; binding < vertex_input.binding_count; binding++) {
		input_desc[binding].binding   = binding;
		input_desc[binding].stride    = vertex_input.bindings[binding].stride;
		input_desc[binding].inputRate = vertex_input.bindings[binding].instance
		                                    ? vk::VertexInputRate::eInstance
		                                    : vk::VertexInputRate::eVertex;
	}
	for (uint32_t index = 0; index < vertex_input.attribute_count; index++) {
		input_attr[index].binding  = vertex_input.attributes[index].binding;
		input_attr[index].location = index;
		input_attr[index].offset   = vertex_input.attributes[index].offset;

		uint32_t   attr_size     = 4;
		const auto registers_num = vs_input_info.resources_dst[index].registers_num;
		const auto compiled_components =
		    vs_input_info.stage.program->info.vertex_fetch_components[index];
		const auto used_components =
		    compiled_components > 0 ? static_cast<int>(compiled_components) : registers_num;
		GetInputFormat(vs_input_info.resources[index], input_attr[index].format, attr_size,
		               static_cast<uint32_t>(used_components));

		if (graphics_debug_dump_enabled()) {
			static std::atomic_uint log_count = 0;
			const auto              log_id    = log_count.fetch_add(1, std::memory_order_relaxed);
			if (log_id < 128) {
				LOGF("VertexInputState[%u]: attr=%u binding=%u offset=%u stride=%u fmt=%d "
				     "src_fmt=%u dst=v%u regs=%u"
				     " fetched_components=%u attr_size=%u swizzle=%u,%u,%u,%u\n",
				     log_id, index, input_attr[index].binding, input_attr[index].offset,
				     input_desc[input_attr[index].binding].stride,
				     static_cast<int>(input_attr[index].format),
				     static_cast<uint32_t>(vs_input_info.resources[index].Format()),
				     static_cast<uint32_t>(vs_input_info.resources_dst[index].register_start),
				     static_cast<uint32_t>(registers_num), static_cast<uint32_t>(used_components),
				     attr_size, static_cast<uint32_t>(vs_input_info.resources[index].DstSelX()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelY()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelZ()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelW()));
			}
		}

		if (vs_input_info.resources[index].OutOfBounds() != 0) {
			static bool logged = false;
			if (!logged) {
				LOGF("VertexInput: temporary: accepting PS5 out-of-bounds behavior %" PRIu8 "\n",
				     vs_input_info.resources[index].OutOfBounds());
				logged = true;
			}
		}

		EXIT_IF(registers_num < 1 || registers_num > 4);
	}

	vk::PipelineVertexInputStateCreateInfo vertex_input_info {};
	vertex_input_info.vertexBindingDescriptionCount   = vertex_input.binding_count;
	vertex_input_info.pVertexBindingDescriptions      = input_desc;
	vertex_input_info.vertexAttributeDescriptionCount = vertex_input.attribute_count;
	vertex_input_info.pVertexAttributeDescriptions    = input_attr;

	vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
	input_assembly.topology = static_params.topology;
	input_assembly.primitiveRestartEnable =
	    static_params.primitive_restart_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineViewportDepthClipControlCreateInfoEXT depth_clip_control {};
	depth_clip_control.negativeOneToOne = (static_params.negative_one_to_one ? VK_TRUE : VK_FALSE);

	vk::PipelineViewportStateCreateInfo viewport_state {};
	viewport_state.pNext = &depth_clip_control;

	vk::PipelineRasterizationDepthClipStateCreateInfoEXT clip_ext {};
	clip_ext.depthClipEnable = static_params.depth_clip_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineRasterizationStateCreateInfo rasterizer {};
	// MoltenVK lacks VK_EXT_depth_clip_enable; omit the depth-clip struct on macOS and accept
	// Vulkan's default depth clipping (enabled) instead of the PS5's clamp behavior.
#if !defined(__APPLE__)
	// The DB clamps depth to the viewport range after polygon offset is applied.
	rasterizer.depthClampEnable = VK_TRUE;
	rasterizer.pNext = &clip_ext;
#endif
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex {};
	EXIT_NOT_IMPLEMENTED(static_params.provoking_vtx_last &&
	                     !graphics.provoking_vertex_last_enabled);
	if (graphics.provoking_vertex_last_enabled) {
		provoking_vertex.provokingVertexMode = static_params.provoking_vtx_last
		    ? vk::ProvokingVertexModeEXT::eLastVertex : vk::ProvokingVertexModeEXT::eFirstVertex;
		provoking_vertex.pNext = rasterizer.pNext;
		rasterizer.pNext = &provoking_vertex;
	}
	rasterizer.polygonMode = static_params.polygon_mode;
	rasterizer.lineWidth = 1.0f;

	vk::PipelineMultisampleStateCreateInfo multisampling {};
	multisampling.sampleShadingEnable  = static_params.sample_shading_enable ? VK_TRUE : VK_FALSE;
	multisampling.rasterizationSamples = vulkan_sample_count(static_params.samples);
	multisampling.minSampleShading     = 1.0f;
	multisampling.alphaToCoverageEnable =
	    static_params.alpha_to_coverage_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineColorBlendAttachmentState color_blend_attachment[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		EXIT_NOT_IMPLEMENTED((static_params.color_mask[i] & ~0x0fu) != 0);
		color_blend_attachment[i].colorWriteMask =
		    vk::ColorComponentFlags {static_params.color_mask[i]};
		color_blend_attachment[i].blendEnable = static_params.blend_enable[i] ? VK_TRUE : VK_FALSE;
		color_blend_attachment[i].srcColorBlendFactor =
		    GetBlendFactor(static_params.color_srcblend[i]);
		color_blend_attachment[i].dstColorBlendFactor =
		    GetBlendFactor(static_params.color_destblend[i]);
		color_blend_attachment[i].colorBlendOp = GetBlendOp(static_params.color_comb_fcn[i]);
		color_blend_attachment[i].srcAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_srcblend[i])
		         : color_blend_attachment[i].srcColorBlendFactor);
		color_blend_attachment[i].dstAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_destblend[i])
		         : color_blend_attachment[i].dstColorBlendFactor);
		color_blend_attachment[i].alphaBlendOp =
		    (static_params.separate_alpha_blend[i] ? GetBlendOp(static_params.alpha_comb_fcn[i])
		                                           : color_blend_attachment[i].colorBlendOp);
	}

	vk::Bool32 color_write_enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		color_write_enable[i] = VK_TRUE;
	}

	vk::PipelineColorWriteCreateInfoEXT color_write {};
	color_write.attachmentCount    = rendering.color_count;
	color_write.pColorWriteEnables = color_write_enable;

	vk::PipelineColorBlendStateCreateInfo color_blending {};
	// MoltenVK lacks VK_EXT_color_write_enable; drop the dynamic color-write struct on macOS
	// and rely on each attachment's static colorWriteMask (all channels enabled by default).
#if !defined(__APPLE__)
	color_blending.pNext = &color_write;
#endif
	color_blending.logicOp         = vk::LogicOp::eCopy;
	color_blending.attachmentCount = rendering.color_count;
	color_blending.pAttachments    = color_blend_attachment;

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	vk::ShaderStageFlags graphics_stages = vk::ShaderStageFlagBits::eFragment;
	if (mesh_task) {
		graphics_stages |= vk::ShaderStageFlagBits::eTaskEXT;
	}
	for (const auto& stage: vertex_info) {
		const auto native_stage = NativeShaderStage(stage.logical_stage);
		AddLayoutBindings(descriptor_bindings, *stage.stage.program, native_stage);
		pipeline.uses_image_heap |= stage.stage.program->info.uses_image_heap;
		graphics_stages |= native_stage;
	}
	// shortcut: with the fragment set split off, the device's per-layout totals are checked per set; check the
	// union too if a game nears maxDescriptorSetStorageBuffers.
	const bool split = ps_active && graphics.SplitsFragmentSet();
	std::vector<vk::DescriptorSetLayoutBinding> fragment_bindings;
	if (ps_active) {
		EXIT_IF(!ps_input_info->stage);
		AddLayoutBindings(split ? fragment_bindings : descriptor_bindings, *ps_input_info->stage.program,
		                  vk::ShaderStageFlagBits::eFragment);
		pipeline.uses_image_heap |= ps_input_info->stage.program->info.uses_image_heap;
	}
	// A pipeline layout may hold one push-descriptor set: with the split it is the pixel stage's, which binds most.
	CreateDescriptorLayout(graphics, pipeline.descriptor_set_layout, pipeline.uses_push_descriptors, descriptor_bindings,
	                       !split);
	if (split) {
		CreateDescriptorLayout(graphics, pipeline.fragment_set_layout, pipeline.fragment_uses_push_descriptors,
		                       fragment_bindings);
	}
	const bool from_libraries =
	    libraries != nullptr && split && BuildsFromLibraries(graphics, vertex_info, ps_input_info, programs, static_params);
	const vk::PushConstantRange push_constants {graphics_stages, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};
	// Set 1: the guest image heap in every guest pipeline while it is on, else the resident image heap
	// in the pipelines that use it.
	const bool       guest_heap = graphics.guest_image_heap_layout != nullptr;
	// The image groups' sets: IR::GroupSet the vertex stage's (only a Vertex program has one), IR::GroupSet + 1 the pixel's.
	const auto vertex_group_bindings = GroupBindings(*vs_input_info.stage.program, NativeShaderStage(vs_input_info.logical_stage));
	const auto fragment_group_bindings =
	    ps_active ? GroupBindings(*ps_input_info->stage.program, vk::ShaderStageFlagBits::eFragment)
	              : std::vector<vk::DescriptorSetLayoutBinding> {};
	CreateGroupSetLayout(graphics, pipeline.group_set_layouts[0], vertex_group_bindings);
	CreateGroupSetLayout(graphics, pipeline.group_set_layouts[1], fragment_group_bindings);
	std::array set_layouts {pipeline.descriptor_set_layout,
	                        guest_heap ? graphics.guest_image_heap_layout : graphics.image_heap_layout,
	                        pipeline.fragment_set_layout, pipeline.group_set_layouts[0], pipeline.group_set_layouts[1]};
	const uint32_t set_count = pipeline.group_set_layouts[1] != nullptr   ? ShaderRecompiler::IR::GroupSet + 2u
	                           : pipeline.group_set_layouts[0] != nullptr ? ShaderRecompiler::IR::GroupSet + 1u
	                           : split                                    ? 3u
	                           : guest_heap || pipeline.uses_image_heap   ? 2u
	                                                                      : 1u;
	FillSetHoles(graphics, std::span(set_layouts).first(set_count));

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = set_count;
	pipeline_layout_info.pSetLayouts            = set_layouts.data();
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;
	// Parts made with the whole layout may be linked into any pipeline whose layout is identically defined
	// (VUID-VkGraphicsPipelineCreateInfo-pLibraries-06613), so their keys hold its definition.
	const bool whole_layout = from_libraries && ReadsPerVertex(*ps_input_info);
	if (from_libraries && !whole_layout) pipeline_layout_info.flags = vk::PipelineLayoutCreateFlagBits::eIndependentSetsEXT;

	EXIT_IF(pipeline.pipeline_layout != nullptr);

	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreatePipelineLayout begin VS=%" PRIu64 " PS=%" PRIu64
		     " set_layouts=1 push_constants=%" PRIu32 "\n",
		     vertex_program.id, ps_active ? pixel_program.id : 0, 1u);
	}
	auto result = graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                   &pipeline.pipeline_layout);
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreatePipelineLayout done result=%s layout=%p\n",
		     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline_layout));
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);

	vk::PipelineDepthStencilStateCreateInfo depth_stencil_info {};

	std::vector<vk::DynamicState> dynamic_states {
	    vk::DynamicState::eViewportWithCount,
	    vk::DynamicState::eScissorWithCount,
	    vk::DynamicState::eLineWidth,
	    vk::DynamicState::eCullMode,
	    vk::DynamicState::eFrontFace,
	    vk::DynamicState::eDepthTestEnable,
	    vk::DynamicState::eDepthWriteEnable,
	    vk::DynamicState::eDepthCompareOp,
	    vk::DynamicState::eDepthBiasEnable,
	    vk::DynamicState::eDepthBias,
	    vk::DynamicState::eStencilTestEnable,
	    vk::DynamicState::eStencilOp,
	    vk::DynamicState::eStencilCompareMask,
	    vk::DynamicState::eStencilReference,
	    vk::DynamicState::eStencilWriteMask,
	    vk::DynamicState::eBlendConstants,
	};
#if !defined(__APPLE__)
	dynamic_states.push_back(vk::DynamicState::eDepthBoundsTestEnable);
	dynamic_states.push_back(vk::DynamicState::eDepthBounds);
#endif
	if (graphics.attachment_feedback_loop_enabled) {
		dynamic_states.push_back(vk::DynamicState::eAttachmentFeedbackLoopEnableEXT);
	}

	vk::PipelineDynamicStateCreateInfo dynamic_state {};
	dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic_state.pDynamicStates    = dynamic_states.data();

	vk::GraphicsPipelineCreateInfo  pipeline_info {};
	vk::PipelineRenderingCreateInfo rendering_info {};
	rendering_info.colorAttachmentCount    = rendering.color_count;
	rendering_info.pColorAttachmentFormats = rendering.color_formats.data();
	rendering_info.depthAttachmentFormat   = rendering.depth_format;
	rendering_info.stencilAttachmentFormat = rendering.stencil_format;
	pipeline_info.pNext                    = &rendering_info;
	pipeline_info.stageCount               = shader_stage_count;
	pipeline_info.pStages                  = shader_stages;
	pipeline_info.pVertexInputState        = mesh ? nullptr : &vertex_input_info;
	pipeline_info.pInputAssemblyState      = mesh ? nullptr : &input_assembly;
	vk::PipelineTessellationStateCreateInfo tessellation_state {};
	tessellation_state.patchControlPoints =
	    tessellation ? vs_input_info.tess.input_control_points : 3u;
	pipeline_info.pTessellationState = (rect_list || tessellation) ? &tessellation_state : nullptr;
	pipeline_info.pViewportState          = &viewport_state;
	pipeline_info.pRasterizationState     = &rasterizer;
	pipeline_info.pMultisampleState       = &multisampling;
	pipeline_info.pDepthStencilState      = (with_depth ? &depth_stencil_info : nullptr);
	pipeline_info.pColorBlendState        = &color_blending;
	pipeline_info.pDynamicState           = &dynamic_state;
	pipeline_info.layout                  = pipeline.pipeline_layout;
	pipeline_info.basePipelineIndex       = -1;

	EXIT_IF((pipeline.pipeline != nullptr) && !KYTY_UPSTREAM_PATH("86139e36f"));

	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines begin VS=%" PRIu64 " PS=%" PRIu64
		     " topology=%" PRIu32 " color_mask=0x%08" PRIx32
		     " depth=%s blend=%s dyn_states=%" PRIu32 "\n",
		     vertex_program.id, ps_active ? pixel_program.id : 0,
		     static_cast<uint32_t>(static_params.topology), static_params.color_mask[0],
		     (with_depth ? "true" : "false"), (static_params.blend_enable[0] ? "true" : "false"),
		     dynamic_state.dynamicStateCount);
	}
	if (from_libraries) {
		// Each part's layout holds its own sets only (all three with whole_layout); the linked pipeline's
		// (pipeline.pipeline_layout) all three.
		// A part may use its layouts until it is linked, and a worker's pipeline (whose sets these are) goes at once,
		// so a part gets sets of its own, alike, kept by the cache.
		const auto part_layout = [&](bool vertex_stages) {
			vk::DescriptorSetLayout vertex_set   = nullptr;
			vk::DescriptorSetLayout fragment_set = nullptr;
			bool                    pushed       = false;
			if (vertex_stages || whole_layout) {
				CreateDescriptorLayout(graphics, vertex_set, pushed, descriptor_bindings, false);
			}
			if (!vertex_stages || whole_layout) {
				CreateDescriptorLayout(graphics, fragment_set, pushed, fragment_bindings);
			}
			vk::DescriptorSetLayout vertex_group   = nullptr;
			vk::DescriptorSetLayout fragment_group = nullptr;
			if (vertex_stages || whole_layout) CreateGroupSetLayout(graphics, vertex_group, vertex_group_bindings);
			if (!vertex_stages || whole_layout) CreateGroupSetLayout(graphics, fragment_group, fragment_group_bindings);
			// A part's layout leaves the other part's sets out (null, independent sets); a whole layout has them all.
			std::array sets {vertex_set, set_layouts[1], fragment_set, vertex_group, fragment_group};
			if (whole_layout) FillSetHoles(graphics, std::span(sets).first(set_count));
			auto             info   = pipeline_layout_info;
			info.pSetLayouts        = sets.data();
			vk::PipelineLayout part = nullptr;
			EXIT_NOT_IMPLEMENTED(graphics.device.createPipelineLayout(&info, nullptr, &part) != vk::Result::eSuccess);
			libraries->Keep(part, {vertex_set, fragment_set, vertex_group, fragment_group});
			return part;
		};
		const auto add_whole_layout = [&](LibraryKey& key) {
			if (!whole_layout) return;
			key.Add(reinterpret_cast<uint64_t>(static_cast<VkDescriptorSetLayout>(set_layouts[1])));
			for (const auto* bindings: std::initializer_list<const std::vector<vk::DescriptorSetLayoutBinding>*> {&descriptor_bindings, &fragment_bindings, &vertex_group_bindings, &fragment_group_bindings}) {
				key.Add(bindings->size());
				for (const auto& binding: *bindings) {
					key.Add(binding.binding).Add(static_cast<uint64_t>(binding.descriptorType));
					key.Add(binding.descriptorCount).Add(static_cast<VkShaderStageFlags>(binding.stageFlags));
				}
			}
		};
		const auto build_part = [&](vk::GraphicsPipelineLibraryFlagsEXT kind, vk::GraphicsPipelineCreateInfo info,
		                            vk::PipelineLayout layout) {
			vk::GraphicsPipelineLibraryCreateInfoEXT library {};
			library.flags          = kind;
			library.pNext          = &rendering_info;
			info.pNext             = &library;
			info.flags             = vk::PipelineCreateFlagBits::eLibraryKHR;
			info.pDynamicState     = &dynamic_state;
			info.layout            = layout;
			info.basePipelineIndex = -1;
			vk::Pipeline part      = nullptr;
			EXIT_NOT_IMPLEMENTED(graphics.device.createGraphicsPipelines(driver_cache, 1, &info, nullptr, &part) !=
			                     vk::Result::eSuccess);
			return part;
		};
		const auto as_word = [](auto value) { return static_cast<uint64_t>(value); };

		LibraryKey vertex_input_key(1);
		vertex_input_key.Add(as_word(input_assembly.topology)).Add(input_assembly.primitiveRestartEnable);
		vertex_input_key.Add(vertex_input.binding_count);
		for (uint32_t i = 0; i < vertex_input.binding_count; i++) {
			vertex_input_key.Add(input_desc[i].stride).Add(as_word(input_desc[i].inputRate));
		}
		vertex_input_key.Add(vertex_input.attribute_count);
		for (uint32_t i = 0; i < vertex_input.attribute_count; i++) {
			vertex_input_key.Add(input_attr[i].binding).Add(input_attr[i].location).Add(input_attr[i].offset);
			vertex_input_key.Add(as_word(input_attr[i].format));
		}
		// A mesh pipeline has no vertex input: its mesh stage is in the pre-rasterization part.
		const auto vertex_part = mesh ? vk::Pipeline {} : libraries->Get(vertex_input_key.Hash(), [&] {
			vk::GraphicsPipelineCreateInfo info {};
			info.pVertexInputState   = &vertex_input_info;
			info.pInputAssemblyState = &input_assembly;
			return build_part(vk::GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface, info, nullptr);
		});

		LibraryKey raster_key(2);
		for (const auto& stage: programs.vertex) raster_key.Add(stage.spirv_hash);
		raster_key.Add(as_word(static_params.polygon_mode)).Add(static_params.depth_clip_enable);
		raster_key.Add(static_params.provoking_vtx_last).Add(static_params.negative_one_to_one);
		add_whole_layout(raster_key);
		const auto raster_part = libraries->Get(raster_key.Hash(), [&] {
			vk::GraphicsPipelineCreateInfo info {};
			info.stageCount          = static_cast<uint32_t>(vertex_info.size());
			info.pStages             = shader_stages;
			info.pViewportState      = &viewport_state;
			info.pRasterizationState = &rasterizer;
			return build_part(vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders, info,
			                  part_layout(true));
		});

		// Keyed by the pixel module alone: the targets and depth presence are not fragment-shader state, and the
		// multisample state is given to the part only with sample shading. So any worker's guessed build leaves the part
		// the draw needs, whatever pass it guessed (with whole_layout: whatever pass, for the same vertex-stage sets).
		LibraryKey fragment_key(3);
		fragment_key.Add(programs.pixel.spirv_hash).Add(multisampling.sampleShadingEnable);
		if (multisampling.sampleShadingEnable) fragment_key.Add(as_word(multisampling.rasterizationSamples));
		add_whole_layout(fragment_key);
		const auto fragment_part = libraries->Get(fragment_key.Hash(), [&] {
			vk::GraphicsPipelineCreateInfo info {};
			info.stageCount         = 1;
			info.pStages            = &shader_stages[shader_stage_count - 1];
			info.pDepthStencilState = &depth_stencil_info; // required in a fragment part (all of it is dynamic here)
			info.pMultisampleState  = multisampling.sampleShadingEnable ? &multisampling : nullptr;
			return build_part(vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader, info,
			                  part_layout(false));
		});

		LibraryKey output_key(4);
		output_key.Add(rendering.color_count);
		for (uint32_t i = 0; i < rendering.color_count; i++) {
			const auto& blend = color_blend_attachment[i];
			output_key.Add(as_word(rendering.color_formats[i])).Add(blend.blendEnable);
			output_key.Add(as_word(blend.srcColorBlendFactor)).Add(as_word(blend.dstColorBlendFactor));
			output_key.Add(as_word(blend.colorBlendOp)).Add(as_word(blend.srcAlphaBlendFactor));
			output_key.Add(as_word(blend.dstAlphaBlendFactor)).Add(as_word(blend.alphaBlendOp));
			output_key.Add(static_cast<VkColorComponentFlags>(blend.colorWriteMask));
		}
		output_key.Add(as_word(rendering.depth_format)).Add(as_word(rendering.stencil_format));
		output_key.Add(as_word(multisampling.rasterizationSamples)).Add(multisampling.sampleShadingEnable);
		const auto output_part = libraries->Get(output_key.Hash(), [&] {
			vk::GraphicsPipelineCreateInfo info {};
			info.pColorBlendState  = &color_blending;
			info.pMultisampleState = &multisampling;
			return build_part(vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface, info, nullptr);
		});

		const std::array                 parts {raster_part, fragment_part, output_part, vertex_part};
		vk::PipelineLibraryCreateInfoKHR linked {};
		linked.libraryCount = mesh ? 3u : 4u;
		linked.pLibraries   = parts.data();
		vk::GraphicsPipelineCreateInfo link {};
		link.pNext             = &linked;
		link.layout            = pipeline.pipeline_layout;
		link.basePipelineIndex = -1;
		result = graphics.device.createGraphicsPipelines(driver_cache, 1, &link, nullptr, &pipeline.pipeline);
	} else {
		result = graphics.device.createGraphicsPipelines(driver_cache, 1, &pipeline_info, nullptr,
		                                                 &pipeline.pipeline);
	}
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines done result=%s pipeline=%p\n",
		     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);

	if (tess_control_shader_module != nullptr) {
		graphics.device.destroyShaderModule(tess_control_shader_module, nullptr);
	}
	if (tess_eval_shader_module != nullptr) {
		graphics.device.destroyShaderModule(tess_eval_shader_module, nullptr);
	}
	if (task_module) {
		graphics.device.destroyShaderModule(task_module, nullptr);
	}
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache) {
	EXIT_IF(compute_module == nullptr);

	vk::PipelineShaderStageCreateInfo                     comp_shader_stage_info {};
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo comp_subgroup_size {};
	comp_shader_stage_info.stage  = vk::ShaderStageFlagBits::eCompute;
	comp_shader_stage_info.module = compute_module;
	comp_shader_stage_info.pName  = "main";
	EXIT_IF(!input_info.stage);
	// A wave64 program emitted for a 32-wide host subgroup (two lanes an invocation, or split over two subgroups) needs
	// that subgroup size: NVIDIA's is fixed at 32, but AMD would take the guest's 64 when asked.
	auto wave_size = input_info.stage.program->wave_size;
	if (input_info.host_subgroup_size != 0u) wave_size = std::min(wave_size, input_info.host_subgroup_size);
	if (graphics.compute_subgroup_size_control_enabled &&
	    wave_size >= graphics.min_subgroup_size && wave_size <= graphics.max_subgroup_size) {
		comp_subgroup_size.requiredSubgroupSize = wave_size;
		comp_shader_stage_info.pNext            = &comp_subgroup_size;
	}
	if (input_info.stage.program->wave_split) {
		// Its wave is two whole subgroups (local size 64, a multiple of the fixed subgroup size).
		EXIT_IF(!graphics.compute_full_subgroups_enabled);
		comp_shader_stage_info.flags |= vk::PipelineShaderStageCreateFlagBits::eRequireFullSubgroups;
	}

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	AddLayoutBindings(descriptor_bindings, *input_info.stage.program,
	                  vk::ShaderStageFlagBits::eCompute);
	pipeline.uses_image_heap = input_info.stage.program->info.uses_image_heap;
	CreateDescriptorLayout(graphics, pipeline.descriptor_set_layout, pipeline.uses_push_descriptors, descriptor_bindings);
	const vk::PushConstantRange push_constants {vk::ShaderStageFlagBits::eCompute, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};
	// Set 1: the guest image heap in every guest pipeline while it is on, else the resident image heap
	// in the pipelines that use it.
	const bool       guest_heap = graphics.guest_image_heap_layout != nullptr;
	CreateGroupSetLayout(graphics, pipeline.group_set_layouts[0],
	                     GroupBindings(*input_info.stage.program, vk::ShaderStageFlagBits::eCompute));
	std::array set_layouts {pipeline.descriptor_set_layout,
	                        guest_heap ? graphics.guest_image_heap_layout : graphics.image_heap_layout,
	                        vk::DescriptorSetLayout {}, pipeline.group_set_layouts[0]};
	const uint32_t set_count = pipeline.group_set_layouts[0] != nullptr ? ShaderRecompiler::IR::GroupSet + 1u
	                           : guest_heap || pipeline.uses_image_heap ? 2u
	                                                                    : 1u;
	FillSetHoles(graphics, std::span(set_layouts).first(set_count));

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = set_count;
	pipeline_layout_info.pSetLayouts            = set_layouts.data();
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;

	EXIT_IF(pipeline.pipeline_layout != nullptr);

	LOGF("PipelineTrace: vkCreatePipelineLayout CS begin set_layouts=1 push_constants=%u\n",
	     1u);
	auto result = graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                  &pipeline.pipeline_layout);
	LOGF("PipelineTrace: vkCreatePipelineLayout CS done result=%s layout=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline_layout));
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);

	vk::ComputePipelineCreateInfo info {};
	if (input_info.stage.program->info.uses_gds) {
		info.flags = vk::PipelineCreateFlagBits::eDispatchBase;
	}
	info.stage             = comp_shader_stage_info;
	info.layout            = pipeline.pipeline_layout;
	info.basePipelineIndex = -1;

	EXIT_IF((pipeline.pipeline != nullptr) && !KYTY_UPSTREAM_PATH("86139e36f"));

	LOGF("PipelineTrace: vkCreateComputePipelines begin layout=%p\n",
	     static_cast<void*>(pipeline.pipeline_layout));
	result = graphics.device.createComputePipelines(driver_cache, 1, &info, nullptr,
	                                                &pipeline.pipeline);
	LOGF("PipelineTrace: vkCreateComputePipelines done result=%s pipeline=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);
}

} // namespace Libs::Graphics
