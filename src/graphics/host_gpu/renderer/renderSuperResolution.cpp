// Sony's PS5 Pro super resolution as Demon's Souls dispatches it (graphics/guest_gpu/superResolution.h), done here:
// DLSS (KYTY_UPSCALER=dlss, frame-gen.h) where Streamline offers it, FSR 3 (KYTY_UPSCALER=fsr, fsr-native.h: Linux,
// any GPU), else a linear stretch of the input.
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/superResolution.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shaderBindings.h"
#include "common/assert.h"
#include "frame-gen.h"
#include "fsr-native.h"
#include "live-switch.h"
#include "gpu_tiler_shaders/sr_exposure_spv.h"
#include "gpu_tiler_shaders/sr_sharpen_spv.h"
#include "speculation-state.h"
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "vulkan-recording.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace {

// The parameter block's texture slots (01.007.000 SuperResolution::Apply; slot 3 is never set, 5 is a 1x1 exposure
// over the TAA's constants, 6 the game's reactive mask).
constexpr uint32_t SlotOutput = 0, SlotColor = 1, SlotMotion = 2, SlotDepth = 4, SlotExposure = 5, SlotReactive = 6;
// Its words: the projection (view to clip, row vectors) at +0x6c, the near and far planes at +0xe8, the jitter in
// render pixels at +0x100, the motion vectors' scale to render pixels towards the previous frame at +0x108 (the
// render size, negated), the reset flag at byte +0x11c (no history: the first frame, a cut).
constexpr uint32_t ViewToClipWord = 0x6c / 4, NearWord = 0xe8 / 4, FarWord = 0xec / 4, JitterWord = 0x100 / 4,
                   MotionScaleWord = 0x108 / 4, ResetByte = 0x11c;

float Word(const SuperResolution::Packet& packet, uint32_t index) {
	float value = 0.0f;
	std::memcpy(&value, &packet.parameters[index], sizeof(value));
	return value;
}

void Report(const char* text) {
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::printf("Super resolution: %s\n", text);
		std::fflush(stdout);
	}
}

// The game's exposure on the output (sr-exposure.comp): its PS5 Pro path takes Sony's upscaled image multiplied by it
// (DLSS's and the stretch's are not: the picture was 2.2x as bright as the PS5 path's, its fog a haze).
struct ExposurePass {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             sampler    = nullptr;
};

// A pass reading the upscaled image (sampled) into the output (storage) with the game's exposure (a buffer): `code` with
// `push_size` bytes of constants. (Made once each, kept for the process.)
ExposurePass MakePass(GraphicContext& graphics, const uint32_t* code, size_t code_size, uint32_t push_size) {
	ExposurePass made;
	std::array<vk::DescriptorSetLayoutBinding, 3> bindings {};
	bindings[0] = {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute, nullptr};
	bindings[1] = {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr};
	bindings[2] = {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr};
	vk::DescriptorSetLayoutCreateInfo descriptors {};
	descriptors.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptors.bindingCount = static_cast<uint32_t>(bindings.size());
	descriptors.pBindings    = bindings.data();
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&descriptors, nullptr, &made.set_layout),
	                     "create super resolution exposure descriptors");
	const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, push_size};
	vk::PipelineLayoutCreateInfo layout {};
	layout.setLayoutCount         = 1;
	layout.pSetLayouts            = &made.set_layout;
	layout.pushConstantRangeCount = 1;
	layout.pPushConstantRanges    = &push;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&layout, nullptr, &made.layout),
	                     "create super resolution exposure layout");
	vk::ShaderModuleCreateInfo shader {};
	shader.codeSize = code_size;
	shader.pCode    = code;
	vk::ShaderModule module;
	RequireVulkanSuccess(graphics.device.createShaderModule(&shader, nullptr, &module), "create super resolution exposure shader");
	vk::ComputePipelineCreateInfo pipeline {};
	pipeline.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline.stage.module = module;
	pipeline.stage.pName  = "main";
	// (The subgroup size the game's compute pipelines require here: GraphicContext::InternalComputeSubgroupSize.)
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup {};
	subgroup.requiredSubgroupSize = graphics.InternalComputeSubgroupSize();
	if (subgroup.requiredSubgroupSize != 0) pipeline.stage.pNext = &subgroup;
	pipeline.layout = made.layout;
	RequireVulkanSuccess(graphics.device.createComputePipelines(nullptr, 1, &pipeline, nullptr, &made.pipeline),
	                     "create super resolution exposure pipeline");
	graphics.device.destroyShaderModule(module, nullptr);
	vk::SamplerCreateInfo sampler {};
	sampler.magFilter    = vk::Filter::eNearest;
	sampler.minFilter    = vk::Filter::eNearest;
	sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	RequireVulkanSuccess(graphics.device.createSampler(&sampler, nullptr, &made.sampler),
	                     "create super resolution exposure sampler");
	return made;
}

const ExposurePass& Exposure(GraphicContext& graphics) {
	static const ExposurePass pass = MakePass(graphics, SR_EXPOSURE_SPV, sizeof(SR_EXPOSURE_SPV), 12);
	return pass;
}

// KYTY_DLSS_SHARPNESS=<0-1> (the launcher's "DLSS sharpening"; 0, the default: none): DLSS's image sharpened by RCAS
// (sr-sharpen.comp) on FSR 3's scale (KYTY_FSR_SHARPNESS, fsr-native.cpp, whose upscaler sharpens itself): 1 the most.
// The game's own TAA sharpens its image; DLSS in its place does not, and its pictures looked softer (issue #224).
float DlssSharpness() {
	static const float value = [] {
		const char* text = std::getenv("KYTY_DLSS_SHARPNESS");
		return text != nullptr && *text != '\0' ? std::clamp(std::strtof(text, nullptr), 0.0f, 1.0f) : 0.0f;
	}();
	return value;
}

const ExposurePass& Sharpen(GraphicContext& graphics) {
	static const ExposurePass pass = MakePass(graphics, SR_SHARPEN_SPV, sizeof(SR_SHARPEN_SPV), 16);
	return pass;
}

// DLSS's own output with sharpening, which the sharpening pass reads into the game's: one of the output's size and
// format, made where first needed. One made for an earlier size is kept (the GPU may still read it; sizes change rarely).
struct SharpenTarget {
	VulkanImage   image;
	vk::ImageView storage = nullptr, sampled = nullptr;
};
const SharpenTarget* SharpenImage(GraphicContext& graphics, vk::Format format, vk::Extent2D extent) {
	static std::mutex                                  mutex;
	static std::vector<std::unique_ptr<SharpenTarget>> made; // (the last one current)
	std::lock_guard                                    lock(mutex);
	if (!made.empty()) {
		const auto& last = made.back()->image;
		if (last.format == format && last.extent.width == extent.width && last.extent.height == extent.height)
			return made.back().get();
	}
	auto                target = std::make_unique<SharpenTarget>();
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, target->image)) return nullptr;
	vk::ImageViewCreateInfo view {};
	view.image            = target->image.image;
	view.viewType         = vk::ImageViewType::e2D;
	view.format           = format;
	view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view, nullptr, &target->storage) != vk::Result::eSuccess ||
	    graphics.device.createImageView(&view, nullptr, &target->sampled) != vk::Result::eSuccess)
		return nullptr;
	made.push_back(std::move(target));
	return made.back().get();
}

// DLSS records into the command buffer itself (Streamline, NGX), so it runs where the renderer's own commands are
// recorded, in their order: on the recording worker (KYTY_VULKAN_RECORDING) as a packet, else here.
struct UpscaleRecord {
	vk::CommandBuffer       command;
	FrameGen::UpscaleInputs inputs;
	uint32_t                color_level = 0, color_layer = 0, output_level = 0, output_layer = 0;
	ExposurePass            exposure_pass;                // (no pipeline: none)
	vk::ImageView           output_sampled = nullptr;     // the output read by the exposure pass
	vk::Buffer              exposure_buffer = nullptr;    // the game's exposure: its float at exposure_index
	vk::DeviceSize          exposure_base = 0, exposure_range = 0;
	uint32_t                exposure_index = 0;
	struct Timing*          timing         = nullptr;     // (KYTY_SR_TIMING)
	uint32_t                timing_entry   = 0;
	GraphicContext*         graphics       = nullptr;     // FSR 3 (fsr-native.h) in place of DLSS
	bool                    fsr            = false;
	// DLSS's sharpening (DlssSharpness): DLSS wrote inputs.output, the sharpening target's; the pass reads it (through
	// output_sampled) into the game's output, game_output.
	ExposurePass            sharpen_pass;                 // (no pipeline: none)
	FrameGen::UpscaleImage  game_output;
	float                   sharpness = 0.0f;             // the lobe's scale
};

// The output (general) multiplied by the exposure in place, after the upscale's writes (compute or transfer); left
// general and shader-written, as the renderer believes it.
void ApplyExposure(const UpscaleRecord& record, const vk::detail::DispatchLoaderDynamic& dispatch) {
	const auto& pass = record.exposure_pass;
	if (pass.pipeline == nullptr || record.output_sampled == nullptr || record.exposure_buffer == nullptr) return;
	const auto& output = record.inputs.output;
	// (The upscale's writes, the exposure's upload; the image stays general.)
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	record.command.pipelineBarrier2(dependency, dispatch);
	const vk::DescriptorImageInfo  source {pass.sampler, record.output_sampled, vk::ImageLayout::eGeneral};
	const vk::DescriptorImageInfo  target {nullptr, output.view, vk::ImageLayout::eGeneral};
	const vk::DescriptorBufferInfo values {record.exposure_buffer, record.exposure_base, record.exposure_range};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	writes[0].dstBinding      = 0;
	writes[0].descriptorCount = 1;
	writes[0].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
	writes[0].pImageInfo      = &source;
	writes[1].dstBinding      = 1;
	writes[1].descriptorCount = 1;
	writes[1].descriptorType  = vk::DescriptorType::eStorageImage;
	writes[1].pImageInfo      = &target;
	writes[2].dstBinding      = 2;
	writes[2].descriptorCount = 1;
	writes[2].descriptorType  = vk::DescriptorType::eStorageBuffer;
	writes[2].pBufferInfo     = &values;
	const uint32_t constants[] {output.extent.width, output.extent.height, record.exposure_index};
	record.command.bindPipeline(vk::PipelineBindPoint::eCompute, pass.pipeline, dispatch);
	record.command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, pass.layout, 0, writes, dispatch);
	record.command.pushConstants(pass.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(constants), constants, dispatch);
	record.command.dispatch((output.extent.width + 7u) / 8u, (output.extent.height + 7u) / 8u, 1, dispatch);
}

// DLSS's image (the sharpening target, general) sharpened into the game's output (general) with the exposure, after the
// upscale's writes; the output left general and shader-written, as the renderer believes it.
void ApplySharpen(const UpscaleRecord& record, const vk::detail::DispatchLoaderDynamic& dispatch) {
	const auto& pass   = record.sharpen_pass;
	const auto& output = record.game_output;
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	record.command.pipelineBarrier2(dependency, dispatch);
	const vk::DescriptorImageInfo  source {pass.sampler, record.output_sampled, vk::ImageLayout::eGeneral};
	const vk::DescriptorImageInfo  target {nullptr, output.view, vk::ImageLayout::eGeneral};
	const vk::DescriptorBufferInfo values {record.exposure_buffer, record.exposure_base, record.exposure_range};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	writes[0].dstBinding      = 0;
	writes[0].descriptorCount = 1;
	writes[0].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
	writes[0].pImageInfo      = &source;
	writes[1].dstBinding      = 1;
	writes[1].descriptorCount = 1;
	writes[1].descriptorType  = vk::DescriptorType::eStorageImage;
	writes[1].pImageInfo      = &target;
	writes[2].dstBinding      = 2;
	writes[2].descriptorCount = 1;
	writes[2].descriptorType  = vk::DescriptorType::eStorageBuffer;
	writes[2].pBufferInfo     = &values;
	uint32_t constants[4] {output.extent.width, output.extent.height, record.exposure_index, 0};
	std::memcpy(&constants[3], &record.sharpness, sizeof(float));
	record.command.bindPipeline(vk::PipelineBindPoint::eCompute, pass.pipeline, dispatch);
	record.command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, pass.layout, 0, writes, dispatch);
	record.command.pushConstants(pass.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(constants), constants, dispatch);
	record.command.dispatch((output.extent.width + 7u) / 8u, (output.extent.height + 7u) / 8u, 1, dispatch);
}

// With no DLSS after all (Streamline refused the evaluation): the input stretched to the output, in the layouts the
// renderer left for DLSS (the color read by shaders, the output general), and left as the renderer believes them.
void Stretch(const UpscaleRecord& record, const vk::detail::DispatchLoaderDynamic& dispatch) {
	const auto& in      = record.inputs;
	const auto  barrier = [&](vk::Image image, uint32_t level, uint32_t layer, vk::ImageLayout from, vk::ImageLayout to,
                             vk::PipelineStageFlags2 source_stage, vk::AccessFlags2 source_access,
                             vk::PipelineStageFlags2 destination_stage, vk::AccessFlags2 destination_access) {
		vk::ImageMemoryBarrier2 b {};
		b.srcStageMask        = source_stage;
		b.srcAccessMask       = source_access;
		b.dstStageMask        = destination_stage;
		b.dstAccessMask       = destination_access;
		b.oldLayout           = from;
		b.newLayout           = to;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image               = image;
		b.subresourceRange    = {vk::ImageAspectFlagBits::eColor, level, 1, layer, 1};
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers    = &b;
		record.command.pipelineBarrier2(dependency, dispatch);
	};
	const auto shaders = vk::PipelineStageFlagBits2::eAllGraphics | vk::PipelineStageFlagBits2::eComputeShader;
	barrier(in.color.image, record.color_level, record.color_layer, in.color.layout, vk::ImageLayout::eTransferSrcOptimal,
	        shaders, vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
	barrier(in.output.image, record.output_level, record.output_layer, in.output.layout, in.output.layout, shaders,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eTransferWrite);
	vk::ImageBlit blit {};
	blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, record.color_level, record.color_layer, 1};
	blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(in.color.extent.width), static_cast<int32_t>(in.color.extent.height), 1};
	blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, record.output_level, record.output_layer, 1};
	blit.dstOffsets[1]  = vk::Offset3D {static_cast<int32_t>(in.output.extent.width), static_cast<int32_t>(in.output.extent.height),
                                       1};
	record.command.blitImage(in.color.image, vk::ImageLayout::eTransferSrcOptimal, in.output.image, in.output.layout, 1, &blit,
	                         vk::Filter::eLinear, dispatch);
	barrier(in.color.image, record.color_level, record.color_layer, vk::ImageLayout::eTransferSrcOptimal, in.color.layout,
	        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eNone, shaders, vk::AccessFlagBits2::eShaderRead);
	barrier(in.output.image, record.output_level, record.output_layer, in.output.layout, in.output.layout,
	        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, shaders,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
}

// KYTY_SR_TIMING=1 (diagnostic): the GPU time of the upscale and of the exposure pass, from timestamps written around
// them (three per dispatch, a ring of 16), read back 8 dispatches later; their means printed every 300.
struct Timing {
	static constexpr uint32_t Entries = 16, Stamps = 3;
	vk::Device     device;
	vk::QueryPool  pool;
	double         period_ns = 1.0;
	uint64_t       written   = 0; // (dispatches timed: the next entry)
	uint64_t       read      = 0;
	double         upscale_ms = 0.0, exposure_ms = 0.0;
	uint32_t       samples    = 0;
};
Timing* SrTiming(GraphicContext& graphics) {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_SR_TIMING");
		return value != nullptr && *value == '1';
	}();
	if (!on) return nullptr;
	static Timing* timing = [&]() -> Timing* {
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = Timing::Entries * Timing::Stamps;
		vk::QueryPool pool;
		if (graphics.device.createQueryPool(&info, nullptr, &pool) != vk::Result::eSuccess) return nullptr;
		auto* made      = new Timing;
		made->device    = graphics.device;
		made->pool      = pool;
		made->period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
		return made;
	}();
	return timing;
}
// GPU thread: the results of entries 8 or more dispatches old that the GPU wrote.
void ReadTiming(Timing& timing) {
	while (timing.read + 8 <= timing.written) {
		const auto entry = static_cast<uint32_t>(timing.read % Timing::Entries);
		std::array<uint64_t, Timing::Stamps> stamps {};
		if (timing.device.getQueryPoolResults(timing.pool, entry * Timing::Stamps, Timing::Stamps, sizeof(stamps), stamps.data(),
		                                      sizeof(uint64_t), vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		timing.upscale_ms += static_cast<double>(stamps[1] - stamps[0]) * timing.period_ns / 1e6;
		timing.exposure_ms += static_cast<double>(stamps[2] - stamps[1]) * timing.period_ns / 1e6;
		++timing.read;
		if (++timing.samples == 300) {
			std::printf("Super resolution GPU: upscale %.3f ms, exposure %.3f ms (means of 300)\n", timing.upscale_ms / 300.0,
			            timing.exposure_ms / 300.0);
			std::fflush(stdout);
			timing.samples    = 0;
			timing.upscale_ms = timing.exposure_ms = 0.0;
		}
	}
}

void RunUpscale(const UpscaleRecord& record, const vk::detail::DispatchLoaderDynamic& dispatch) {
	const auto stamp = [&](uint32_t index) {
		if (record.timing == nullptr) return;
		const auto first = record.timing_entry * Timing::Stamps;
		if (index == 0) record.command.resetQueryPool(record.timing->pool, first, Timing::Stamps, dispatch);
		record.command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, record.timing->pool, first + index, dispatch);
	};
	stamp(0);
	if (record.sharpen_pass.pipeline != nullptr) {
		// (Its last frame's contents discarded, after the sharpening pass that read them.)
		vk::ImageMemoryBarrier2 b {};
		b.srcStageMask        = vk::PipelineStageFlagBits2::eComputeShader;
		b.srcAccessMask       = vk::AccessFlagBits2::eNone;
		b.dstStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
		b.dstAccessMask       = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite;
		b.oldLayout           = vk::ImageLayout::eUndefined;
		b.newLayout           = vk::ImageLayout::eGeneral;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image               = record.inputs.output.image;
		b.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers    = &b;
		record.command.pipelineBarrier2(dependency, dispatch);
	}
	const bool upscaled = record.fsr ? FsrNative::Upscale(*record.graphics, record.command, record.inputs)
	                                 : FrameGen::Upscale(record.command, record.inputs, dispatch);
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	LocalVulkanRecording::InvalidateReplayState(); // (what DLSS bound)
#endif
	if (!upscaled) Stretch(record, dispatch);
	stamp(1);
	if (record.sharpen_pass.pipeline != nullptr) {
		ApplySharpen(record, dispatch);
	} else {
		ApplyExposure(record, dispatch);
	}
	stamp(2);
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	LocalVulkanRecording::InvalidateReplayState(); // (the pass's pipeline)
#endif
}

#ifdef KYTY_LOCAL_VULKAN_RECORDING
void ReplayUpscale(std::span<const LocalVulkanRecording::Segment> segments, const vk::detail::DispatchLoaderDynamic& dispatch) {
	UpscaleRecord record;
	std::memcpy(static_cast<void*>(&record), segments[0].data, sizeof(record));
	RunUpscale(record, dispatch);
}
#endif

} // namespace

void RenderExecutor::RunSuperResolution(CommandBuffer& buffer, const SuperResolution::Packet& packet) {
	EXIT_IF(buffer.IsInvalid());
	// (A hole of a speculative translation: the commit, on the GPU thread, runs it - as DispatchDirect.)
	if (Spec::Current() != nullptr) return Report("a speculative translation reached a dispatch: skipped");
	m_context.GetCommandScheduler().PopPendingOperations();
	RenderLockGuard lock(m_context.GetMutex());
	FrameGen::NoteScene(); // (the game's 3D view, as its TAA pass notes it)
	auto&                             cache = m_context.GetTextureCache();
	constexpr std::array<uint32_t, 4> slots {SlotColor, SlotMotion, SlotDepth, SlotOutput};
	std::array<TextureBinding, 4>     bindings;
	for (uint32_t i = 0; i < slots.size(); i++) {
		const auto slot    = slots[i];
		const bool storage = slot == SlotOutput;
		if ((packet.textures_given & (1u << slot)) == 0) {
			ResetBindings();
			return Report("a dispatch without its color, depth, motion or output: skipped");
		}
		ShaderTextureResource descriptor {};
		std::memcpy(descriptor.fields, packet.textures[slot], sizeof(descriptor.fields));
		ShaderRecompiler::IR::ImageResource resource {};
		resource.source         = 0x5352u + slot; // (its own key in the resolution cache)
		resource.resource_class = storage ? ShaderRecompiler::IR::ImageResourceClass::Storage
		                                  : ShaderRecompiler::IR::ImageResourceClass::Sampled;
		resource.numeric_class  = Prospero::SampledTextureNumericClass(descriptor.Format());
		resource.dimension      = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
		resource.read           = !storage;
		resource.written        = storage;
		ShaderRecompiler::IR::DescriptorValue value {};
		std::copy(std::begin(descriptor.fields), std::end(descriptor.fields), value.dwords.begin());
		value.dword_count = 8;
		bindings[i]       = ResolveTexture(resource, value);
		if (bindings[i].desc.info.data.Empty()) {
			ResetBindings();
			return Report("an image the texture cache could not make (bound as null): skipped");
		}
		BindImage(bindings[i].image_id, storage);
	}
	// The game's reactive mask (slot 6), where given: the pixels Sony's upscaler is told to take from this frame rather
	// than from its history, the effects drawn with a pattern that changes every frame (fire, smoke, the soul form's
	// glow). Off by default: as FSR 3's reactive mask it took whole areas from this frame alone, the 1080p samples showing
	// on a 4K output as a grid of blocks and sparkling edges (issue #244, an RTX 3080 whose FSR frame generation runs FSR 3
	// in DLSS's place); DLSS's tag of it (its hint towards the current colour) was refused by Streamline every frame,
	// "[isFormatSupported] Cannot have undefined format" several times a frame on the console (issue #129), so DLSS never
	// had it. (The effects' flicker of #232 went with the texture LOD bias for the materials only, fa85edb0.) Live switch
	// sr_reactive (KYTY_SR_REACTIVE its start value): 0 (default) neither, 1 DLSS, 2 FSR 3 too.
	static auto& reactive_switch = LiveSwitch::Find("sr_reactive", 0);
	const bool   reactive_wanted = reactive_switch.load(std::memory_order_relaxed) != 0;
	TextureBinding reactive_binding;
	bool           has_reactive = false;
	if (ShaderTextureResource descriptor {}; reactive_wanted && (packet.textures_given & (1u << SlotReactive)) != 0) {
		std::memcpy(descriptor.fields, packet.textures[SlotReactive], sizeof(descriptor.fields));
		if (!descriptor.IsNull()) {
			ShaderRecompiler::IR::ImageResource resource {};
			resource.source         = 0x5352u + SlotReactive;
			resource.resource_class = ShaderRecompiler::IR::ImageResourceClass::Sampled;
			resource.numeric_class  = Prospero::SampledTextureNumericClass(descriptor.Format());
			resource.dimension      = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
			resource.read           = true;
			ShaderRecompiler::IR::DescriptorValue value {};
			std::copy(std::begin(descriptor.fields), std::end(descriptor.fields), value.dwords.begin());
			value.dword_count = 8;
			reactive_binding  = ResolveTexture(resource, value);
			has_reactive      = !reactive_binding.desc.info.data.Empty();
			if (has_reactive) BindImage(reactive_binding.image_id, false);
		}
	}
	if (has_reactive) {
		reactive_binding.image_view = cache.FindTexture(reactive_binding.image_id, reactive_binding.desc);
		cache.GetImage(reactive_binding.image_id).usage.texture = true;
	}
	// (As AcquireImages: the images' contents current, the output's write noted.)
	for (auto& binding: bindings) {
		auto       desc    = binding.desc;
		const bool storage = desc.type == TextureCache::BindingType::Storage;
		if (storage) desc.view_info.level_count = 1;
		binding.image_view = cache.FindTexture(binding.image_id, desc);
		auto& image        = cache.GetImage(binding.image_id);
		if (storage && !image.usage.storage) ++image.validity;
		image.usage.storage |= storage;
		image.usage.texture |= !storage;
	}
	// (Frame generation's depth and motion, as the TAA pass's.)
	if (FrameGen::Enabled())
		FrameGen::OnUpscale(bindings[2].image_id, bindings[2].desc.info.data.address, bindings[1].image_id,
		                    bindings[1].desc.info.data.address);
	buffer.EndRendering();
	const auto vk_buffer = buffer.Handle();
	auto&      color     = cache.GetImage(bindings[0].image_id);
	auto&      motion    = cache.GetImage(bindings[1].image_id);
	auto&      depth     = cache.GetImage(bindings[2].image_id);
	auto&      output    = cache.GetImage(bindings[3].image_id);
	const auto range     = [](const TextureBinding& binding) {
        const auto& view = binding.desc.view_info;
        return ImageSubresourceRange {view.base_level, 1, view.base_layer, 1};
	};
	const auto extent = [](const TextureBinding& binding, const Image& image) {
		const auto level = binding.desc.view_info.base_level;
		return vk::Extent2D {std::max(image.info.extent.width >> level, 1u), std::max(image.info.extent.height >> level, 1u)};
	};
	// KYTY_SR_STRETCH=1 (diagnostic): the stretch where DLSS is ready too (the PS5 Pro path's picture without it).
	// KYTY_SR_EXPOSURE=0 (diagnostic): the output without the game's exposure.
	static const bool stretch = [] {
		const char* value = std::getenv("KYTY_SR_STRETCH");
		return value != nullptr && *value == '1';
	}();
	static const bool exposure = [] {
		const char* value = std::getenv("KYTY_SR_EXPOSURE");
		return value == nullptr || *value != '0';
	}();
	const bool fsr          = FsrNative::UpscalerRequested() && !stretch;
	const bool dlss         = FrameGen::UpscalerReady() && !stretch && !fsr;
	const bool upscaler     = dlss || fsr;
	// (FSR's backend reads a depth image in the shader-read layout, as the TAA's depth copy is.)
	const auto depth_layout = depth.info.IsDepth() && !fsr ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
	                                                       : vk::ImageLayout::eShaderReadOnlyOptimal;
	const auto upscale_image = [&](const TextureBinding& binding, const Image& image, vk::ImageLayout layout) {
		FrameGen::UpscaleImage out;
		out.image  = image.backing.image;
		out.view   = binding.image_view;
		out.layout = layout;
		out.format = binding.desc.view_info.format;
		out.extent = extent(binding, image);
		out.usage  = image.backing.usage;
		out.base_level = binding.desc.view_info.base_level;
		out.base_layer = binding.desc.view_info.base_layer;
		out.image_format = image.backing.format;
		return out;
	};
	UpscaleRecord record;
	record.command        = vk_buffer;
	record.graphics       = &m_context.GetGraphics();
	record.fsr            = fsr;
	record.color_level    = bindings[0].desc.view_info.base_level;
	record.color_layer    = bindings[0].desc.view_info.base_layer;
	record.output_level   = bindings[3].desc.view_info.base_level;
	record.output_layer   = bindings[3].desc.view_info.base_layer;
	auto& inputs          = record.inputs;
	inputs.color          = upscale_image(bindings[0], color, vk::ImageLayout::eShaderReadOnlyOptimal);
	inputs.motion         = upscale_image(bindings[1], motion, vk::ImageLayout::eShaderReadOnlyOptimal);
	inputs.depth          = upscale_image(bindings[2], depth, depth_layout);
	inputs.output         = upscale_image(bindings[3], output, vk::ImageLayout::eGeneral);
	if (has_reactive && upscaler && (!fsr || reactive_switch.load(std::memory_order_relaxed) >= 2))
		inputs.reactive = upscale_image(reactive_binding, cache.GetImage(reactive_binding.image_id),
		                                vk::ImageLayout::eShaderReadOnlyOptimal);
	inputs.jitter_x       = Word(packet, JitterWord);
	inputs.jitter_y       = Word(packet, JitterWord + 1);
	inputs.motion_scale_x = Word(packet, MotionScaleWord);
	inputs.motion_scale_y = Word(packet, MotionScaleWord + 1);
	inputs.reset          = reinterpret_cast<const uint8_t*>(packet.parameters)[ResetByte] != 0;
	if (inputs.reset) FrameGen::NoteCut(); // (frame generation does not blend across it either)
	for (uint32_t i = 0; i < 16; i++) inputs.view_to_clip[i] = Word(packet, ViewToClipWord + i);
	inputs.near_plane = Word(packet, NearWord);
	inputs.far_plane  = Word(packet, FarWord);
	// The exposure: the float the 1x1 32-bit float texture of slot 5 holds (the game's constants), read as a buffer.
	ShaderTextureResource exposure_texture {};
	std::memcpy(exposure_texture.fields, packet.textures[SlotExposure], sizeof(exposure_texture.fields));
	if (exposure && (packet.textures_given & (1u << SlotExposure)) != 0 && !exposure_texture.IsNull() &&
	    exposure_texture.Format() == Prospero::BufferFormat::k32Float) {
		const auto [values, offset] = m_context.GetBufferCache().ObtainBuffer(exposure_texture.Base40(), sizeof(float), false);
		if (values != nullptr) {
			const auto alignment   = m_context.GetGraphics().GetPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;
			const auto base        = offset & ~(static_cast<uint64_t>(alignment) - 1u);
			record.exposure_buffer = values->Handle();
			record.exposure_base   = base;
			record.exposure_range  = offset - base + sizeof(float);
			record.exposure_index  = static_cast<uint32_t>((offset - base) / sizeof(float));
			record.exposure_pass   = Exposure(m_context.GetGraphics());
			// (A view the pass samples: the output's own, the image's to keep and destroy.)
			auto view_info    = bindings[3].desc.view_info;
			view_info.usage   = vk::ImageUsageFlagBits::eSampled;
			view_info.mapping = {};
			std::scoped_lock view_lock {cache.m_lock};
			record.output_sampled = output.FindView(view_info);
		}
	}
	// DLSS's sharpening: DLSS writes an image of its own, which the pass sharpens into the game's output.
	if (dlss && DlssSharpness() > 0.0f && record.exposure_pass.pipeline != nullptr) {
		static std::atomic<bool> reported {false};
		if (const auto* target = SharpenImage(m_context.GetGraphics(), inputs.output.format, inputs.output.extent)) {
			record.sharpen_pass   = Sharpen(m_context.GetGraphics());
			record.game_output    = inputs.output;
			record.sharpness      = std::exp2(2.0f * DlssSharpness() - 2.0f);
			record.output_sampled = target->sampled;
			record.output_level   = 0;
			record.output_layer   = 0;
			inputs.output.image        = target->image.image;
			inputs.output.view         = target->storage;
			inputs.output.usage        = target->image.usage;
			inputs.output.base_level   = 0;
			inputs.output.base_layer   = 0;
			inputs.output.image_format = inputs.output.format;
			if (!reported.exchange(true, std::memory_order_relaxed)) {
				std::printf("Super resolution: DLSS's image sharpened (KYTY_DLSS_SHARPNESS=%.2f, RCAS)\n", DlssSharpness());
				std::fflush(stdout);
			}
		} else if (!reported.exchange(true, std::memory_order_relaxed)) {
			std::printf("Super resolution: no image for DLSS's sharpening (%s %ux%u): not sharpened\n",
			            vk::to_string(inputs.output.format).c_str(), inputs.output.extent.width, inputs.output.extent.height);
			std::fflush(stdout);
		}
	}
	if (static std::atomic<uint32_t> images {0}; images.fetch_add(1, std::memory_order_relaxed) < 1) {
		// (What the images are made as, beside the views the bindings ask for: an upscaler making its own views needs it.)
		const auto describe = [](const char* name, const Image& image, const TextureBinding& binding) {
			std::printf("Super resolution: %s image %s %ux%u levels %u layers %u flags %s, view %s level %u layer %u\n", name,
			            vk::to_string(image.backing.format).c_str(), image.backing.extent.width, image.backing.extent.height,
			            image.info.resources.levels, image.info.resources.layers,
			            vk::to_string(image.backing.flags).c_str(), vk::to_string(binding.desc.view_info.format).c_str(),
			            binding.desc.view_info.base_level, binding.desc.view_info.base_layer);
		};
		describe("color", color, bindings[0]);
		describe("motion", motion, bindings[1]);
		describe("depth", depth, bindings[2]);
		describe("output", output, bindings[3]);
		if (has_reactive && inputs.reactive.image) describe("reactive mask", cache.GetImage(reactive_binding.image_id), reactive_binding);
		else std::printf("Super resolution: no reactive mask (%s)\n", !reactive_wanted ? "off: KYTY_SR_REACTIVE=1 DLSS, 2 FSR 3 too"
		                                                              : has_reactive  ? "FSR 3 without it: KYTY_SR_REACTIVE=2 gives it"
		                                                                              : "the game gave none");
	}
	if (static std::atomic<uint32_t> reports {0}; reports.fetch_add(1, std::memory_order_relaxed) < 2) {
		std::printf("Super resolution: %s, the game's exposure %s\n", dlss ? "DLSS" : fsr ? "FSR 3" : "stretch",
		            record.exposure_pass.pipeline != nullptr ? "applied" : "not applied");
		std::fflush(stdout);
	}
	if (auto* timing = SrTiming(m_context.GetGraphics()); timing != nullptr && upscaler) {
		ReadTiming(*timing);
		record.timing       = timing;
		record.timing_entry = static_cast<uint32_t>(timing->written++ % Timing::Entries);
	}
	if (upscaler) {
		color.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, range(bindings[0]), vk_buffer);
		motion.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, range(bindings[1]), vk_buffer);
		depth.Transit(depth_layout, vk::AccessFlagBits2::eShaderRead, range(bindings[2]), vk_buffer);
		output.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
		               range(bindings[3]), vk_buffer);
		if (inputs.reactive.image)
			cache.GetImage(reactive_binding.image_id)
			    .Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, range(reactive_binding),
			             vk_buffer);
#ifdef KYTY_LOCAL_VULKAN_RECORDING
		const LocalVulkanRecording::Segment segment {&record, sizeof(record)};
		if (!LocalVulkanRecording::EnqueuePacket(ReplayUpscale, {&segment, 1}))
			LocalVulkanRecording::ReplayInline(ReplayUpscale, {&segment, 1});
#else
		RunUpscale(record, VULKAN_HPP_DEFAULT_DISPATCHER);
#endif
		buffer.InvalidateGraphicsState(); // (what DLSS bound)
	} else {
		// No DLSS: the input stretched to the output (no anti-aliasing), then the exposure.
		color.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, range(bindings[0]), vk_buffer);
		output.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, range(bindings[3]), vk_buffer);
		vk::ImageBlit blit {};
		blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, record.color_level, record.color_layer, 1};
		blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(inputs.color.extent.width),
		                                    static_cast<int32_t>(inputs.color.extent.height), 1};
		blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, record.output_level, record.output_layer, 1};
		blit.dstOffsets[1]  = vk::Offset3D {static_cast<int32_t>(inputs.output.extent.width),
		                                    static_cast<int32_t>(inputs.output.extent.height), 1};
		vk_buffer.blitImage(color.backing.image, vk::ImageLayout::eTransferSrcOptimal, output.backing.image,
		                    vk::ImageLayout::eTransferDstOptimal, 1, &blit, vk::Filter::eLinear);
		output.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
		               range(bindings[3]), vk_buffer);
		ApplyExposure(record, VULKAN_HPP_DEFAULT_DISPATCHER);
	}
	ResetBindings();
}

} // namespace Libs::Graphics
