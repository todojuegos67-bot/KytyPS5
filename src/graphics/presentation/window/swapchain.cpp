#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "gpu_blit_shaders/gpu_blit_fs_triangle_spv.h"
#include "gpu_blit_shaders/gpu_video_out_downscale_spv.h"
#include "gpu_blit_shaders/gpu_video_out_fsr_easu_spv.h"
#include "gpu_blit_shaders/gpu_video_out_fsr_rcas_spv.h"
#include "gpu_blit_shaders/gpu_video_out_overlay_spv.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/drainStats.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window/frameGeneration.h"
#include "graphics/presentation/window/windowInternal.h"
#include "local/fg-discover.h"
#include "local/frame-gen.h"
#include "local/title-state.h"
#include "local/fsr3-fg.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>
#include <vulkan/vk_platform.h>

// IWYU pragma: no_include <intrin.h>

namespace Libs::Graphics {

struct Presenter::Frame {
	VulkanImage   image;
	vk::ImageView view         = nullptr;
	// Every mip level, for the filtered presentation downscale (see PresentFilter).
	vk::ImageView mips_view    = nullptr;
	uint64_t      present_tick = 0;
	bool          busy         = false;
	// bruno-astro-1025: frame generation's inputs for this frame (src/local/frame-gen.h: DLSS-G, FSR 3 frame
	// generation), copied at the guest flip; `scene`: the guest frame was the game's view (FrameGen::TakeScene).
	FrameGen::FrameInputs fg;
	bool                  scene = false;

	void Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format);
	void Transit(vk::CommandBuffer command, vk::ImageLayout layout, vk::AccessFlags2 access);
	void CopyFrom(CommandBuffer& command, Image& source);
	void Clear(CommandBuffer& command, const vk::ClearColorValue& color);
};

class FramePool final {
public:
	FramePool(WindowContext& window, CommandScheduler& scheduler)
	    : m_window(window), m_scheduler(scheduler) {}
	~FramePool() {
		m_scheduler.Wait(m_scheduler.CurrentTick() - 1);
		for (auto& frame: m_frames) {
			FrameGen::DestroyInputs(m_window.graphic_ctx, frame->fg);
			m_window.graphic_ctx.device.destroyImageView(frame->view, nullptr);
			m_window.graphic_ctx.device.destroyImageView(frame->mips_view, nullptr);
			if (frame->image.image != nullptr) {
				m_window.graphic_ctx.DeleteImage(frame->image);
			}
		}
	}
	KYTY_CLASS_NO_COPY(FramePool);

	void Initialize(uint32_t count, vk::Format format) {
		if (count == 0 || format == vk::Format::eUndefined) {
			EXIT("prepared-frame pool requires at least one frame\n");
		}
		Common::LockGuard lock(m_mutex);
		if (!m_frames.empty()) {
			EXIT("prepared-frame pool was initialized twice\n");
		}
		m_format = format;
		m_frames.reserve(count);
		m_free.reserve(count);
		for (uint32_t i = 0; i < count; i++) {
			auto frame = std::make_unique<Presenter::Frame>();
			m_free.push_back(frame.get());
			m_frames.push_back(std::move(frame));
		}
	}

	void SetFormat(vk::Format format) {
		if (format == vk::Format::eUndefined) {
			EXIT("prepared-frame pool requires a presentation format\n");
		}
		Common::LockGuard lock(m_mutex);
		m_format = format;
	}

	vk::Format GetFormat() {
		Common::LockGuard lock(m_mutex);
		if (m_format == vk::Format::eUndefined) {
			EXIT("prepared-frame pool has no presentation format\n");
		}
		return m_format;
	}

	Presenter::Frame* Acquire(vk::Extent2D extent, vk::Format format) {
		m_mutex.Lock();
		if (m_frames.empty()) {
			EXIT("prepared-frame pool was used before swapchain initialization\n");
		}
		// A synchronized flip may need more frames than the swapchain has images.
		// Queue capacity bounds growth; waiting here can prevent its master from recording.
		if (m_free.empty()) {
			auto frame = std::make_unique<Presenter::Frame>();
			m_free.push_back(frame.get());
			m_frames.push_back(std::move(frame));
		}
		const auto compatible = std::ranges::find_if(m_free, [&](const auto* frame) {
			return frame->image.extent.width == extent.width &&
			       frame->image.extent.height == extent.height && frame->image.format == format;
		});
		auto*      frame      = compatible == m_free.end() ? m_free.back() : *compatible;
		if (compatible != m_free.end()) {
			*compatible = m_free.back();
		}
		m_free.pop_back();
		if (frame->busy) {
			EXIT("prepared-frame pool returned an invalid frame\n");
		}
		frame->busy = true;
		m_mutex.Unlock();

		WaitForFrame(*frame);
		return frame;
	}

	void ValidateForPresent(Presenter::Frame* frame) {
		Common::LockGuard lock(m_mutex);
		if (frame == nullptr || !frame->busy) {
			EXIT("prepared frame has invalid presentation ownership\n");
		}
	}

	void Release(Presenter::Frame* frame) {
		if (frame == nullptr) {
			EXIT("cannot release a null prepared frame\n");
		}
		Common::LockGuard lock(m_mutex);
		if (!frame->busy) {
			EXIT("prepared frame was released twice\n");
		}
		frame->busy = false;
		m_free.push_back(frame);
	}

private:
	void WaitForFrame(Presenter::Frame& frame) {
		DrainStats::ReasonScope reason(DrainStats::Reason::PresentFrame);
		m_scheduler.Wait(frame.present_tick);
	}

	WindowContext&                                 m_window;
	CommandScheduler&                              m_scheduler;
	Common::Mutex                                  m_mutex;
	std::vector<std::unique_ptr<Presenter::Frame>> m_frames;
	std::vector<Presenter::Frame*>                 m_free;
	vk::Format                                     m_format = vk::Format::eUndefined;
};

void Presenter::Frame::Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format) {
	if (extent.width == 0 || extent.height == 0 || format == vk::Format::eUndefined) {
		EXIT("unsupported prepared frame, extent=%ux%u format=%d\n", extent.width, extent.height,
		     static_cast<int>(format));
	}
	auto&      dst        = image;
	const bool compatible = dst.image != nullptr && dst.extent.width == extent.width &&
	                        dst.extent.height == extent.height && dst.format == format;
	if (compatible) {
		return;
	}

	const auto features = graphics.GetFormatProperties(format).optimalTilingFeatures;
	const auto required =
	    vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eSampledImageFilterLinear |
	    vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferSrc |
	    vk::FormatFeatureFlagBits::eTransferDst;
	if ((features & required) != required) {
		EXIT("prepared presentation format lacks optimal blit support: format=%d features=0x%x\n",
		     static_cast<int>(format), static_cast<vk::FormatFeatureFlags::MaskType>(features));
	}

	if (dst.image != nullptr) {
		graphics.device.destroyImageView(view, nullptr);
		graphics.device.destroyImageView(mips_view, nullptr);
		view      = nullptr;
		mips_view = nullptr;
		graphics.DeleteImage(dst);
	}

	// The mip chain lets presentation shrink the frame without aliasing (see
	// PresentSourceLevel). Without blit-destination support, frames present from level 0 only.
	const bool mips = static_cast<bool>(features & vk::FormatFeatureFlagBits::eBlitDst);
	vk::ImageCreateInfo create {};
	create.sType         = vk::StructureType::eImageCreateInfo;
	create.imageType     = vk::ImageType::e2D;
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels =
	    mips ? static_cast<uint32_t>(std::bit_width(std::max(extent.width, extent.height))) : 1u;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
	               vk::ImageUsageFlagBits::eSampled;
	create.sharingMode = vk::SharingMode::eExclusive;
	create.samples     = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, dst)) {
		EXIT("failed to allocate prepared presentation image, extent=%ux%u format=%d\n",
		     extent.width, extent.height, static_cast<int>(format));
	}
}

void Presenter::Frame::Transit(vk::CommandBuffer command, vk::ImageLayout layout,
                               vk::AccessFlags2 access) {
	const auto     stage  = access == vk::AccessFlagBits2::eTransferRead ||
	                                access == vk::AccessFlagBits2::eTransferWrite
	                            ? vk::PipelineStageFlagBits2::eTransfer
	                            : vk::PipelineStageFlagBits2::eAllCommands;
	constexpr auto writes = vk::AccessFlagBits2::eTransferWrite |
	                        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eMemoryWrite;
	if (image.state.layout == layout && image.state.access_mask == access &&
	    !static_cast<bool>(image.state.access_mask & writes)) {
		return;
	}
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask                    = image.state.pl_stage;
	barrier.srcAccessMask                   = image.state.access_mask;
	barrier.dstStageMask                    = stage;
	barrier.dstAccessMask                   = access;
	barrier.oldLayout                       = image.state.layout;
	barrier.newLayout                       = layout;
	barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                           = image.image;
	barrier.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	barrier.subresourceRange.baseMipLevel   = 0;
	barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image.state = {stage, access, layout};
	image.subresource_states.clear();
}

void Presenter::Frame::CopyFrom(CommandBuffer& command_buffer, Image& source) {
	command_buffer.EndRendering();
	auto command = command_buffer.Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	vk::ImageCopy copy {};
	copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, source.backing.layers};
	copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, image.layers};
	copy.extent         = {std::min(source.backing.extent.width, image.extent.width),
	                       std::min(source.backing.extent.height, image.extent.height), 1};
	EXIT_IF(copy.srcSubresource.layerCount != copy.dstSubresource.layerCount);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, image.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy);
}

static vk::Extent2D MipExtent(vk::Extent2D extent, uint32_t level) {
	return {std::max(extent.width >> level, 1u), std::max(extent.height >> level, 1u)};
}

uint32_t PresentSourceLevel(vk::Extent2D source, vk::Extent2D target, uint32_t levels) noexcept {
	uint32_t level = 0;
	for (; level + 1 < levels; level++) {
		const auto extent = MipExtent(source, level);
		if (extent.width <= 2ull * target.width && extent.height <= 2ull * target.height) {
			break;
		}
	}
	return level;
}

void RecordPresentDownscale(vk::CommandBuffer command, vk::Image image, vk::Extent2D extent,
                            uint32_t level) {
	for (uint32_t mip = 1; mip <= level; mip++) {
		// Undefined discards the level; the transfer-stage source scope orders this write after
		// an earlier presentation's read of it.
		vk::ImageMemoryBarrier2 barrier {};
		barrier.srcStageMask        = vk::PipelineStageFlagBits2::eTransfer;
		barrier.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
		barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
		barrier.oldLayout           = vk::ImageLayout::eUndefined;
		barrier.newLayout           = vk::ImageLayout::eTransferDstOptimal;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image               = image;
		barrier.subresourceRange    = {vk::ImageAspectFlagBits::eColor, mip, 1, 0, 1};
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers    = &barrier;
		command.pipelineBarrier2(dependency);

		const auto  source      = MipExtent(extent, mip - 1);
		const auto  destination = MipExtent(extent, mip);
		vk::ImageBlit region {};
		region.srcSubresource = {vk::ImageAspectFlagBits::eColor, mip - 1, 0, 1};
		region.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(source.width),
		                                      static_cast<int32_t>(source.height), 1};
		region.dstSubresource = {vk::ImageAspectFlagBits::eColor, mip, 0, 1};
		region.dstOffsets[1]  = vk::Offset3D {static_cast<int32_t>(destination.width),
		                                      static_cast<int32_t>(destination.height), 1};
		command.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image,
		                  vk::ImageLayout::eTransferDstOptimal, 1, &region, vk::Filter::eLinear);

		barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
		barrier.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
		barrier.newLayout     = vk::ImageLayout::eTransferSrcOptimal;
		command.pipelineBarrier2(dependency);
	}
}

// A pipeline that draws the fullscreen triangle with `fragment` into one `format` attachment,
// optionally blending premultiplied alpha over it.
static vk::Pipeline CreateFullscreenPipeline(vk::Device device, vk::PipelineLayout layout,
                                             std::span<const uint32_t> fragment_code,
                                             vk::Format format, bool blend_over) {
	const auto vertex   = CompileSPV(GPU_BLIT_FS_TRIANGLE_SPV, device);
	const auto fragment = CompileSPV(fragment_code, device);
	std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
	stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
	stages[0].module = vertex;
	stages[0].pName  = "main";
	stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
	stages[1].module = fragment;
	stages[1].pName  = "main";
	vk::PipelineVertexInputStateCreateInfo   vertex_input {};
	vk::PipelineInputAssemblyStateCreateInfo assembly {};
	assembly.topology = vk::PrimitiveTopology::eTriangleList;
	vk::PipelineViewportStateCreateInfo viewport {};
	viewport.viewportCount = 1;
	viewport.scissorCount  = 1;
	vk::PipelineRasterizationStateCreateInfo rasterization {};
	rasterization.lineWidth = 1.0f;
	vk::PipelineMultisampleStateCreateInfo multisample {};
	multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
	vk::PipelineColorBlendAttachmentState attachment {};
	attachment.blendEnable         = blend_over ? VK_TRUE : VK_FALSE;
	attachment.srcColorBlendFactor = vk::BlendFactor::eOne;
	attachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
	attachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
	attachment.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
	attachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
	                            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
	vk::PipelineColorBlendStateCreateInfo blend {};
	blend.attachmentCount = 1;
	blend.pAttachments    = &attachment;
	const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo dynamic {};
	dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic.pDynamicStates    = dynamic_states.data();
	vk::PipelineRenderingCreateInfo rendering {};
	rendering.colorAttachmentCount    = 1;
	rendering.pColorAttachmentFormats = &format;
	vk::GraphicsPipelineCreateInfo create {};
	create.pNext               = &rendering;
	create.stageCount          = static_cast<uint32_t>(stages.size());
	create.pStages             = stages.data();
	create.pVertexInputState   = &vertex_input;
	create.pInputAssemblyState = &assembly;
	create.pViewportState      = &viewport;
	create.pRasterizationState = &rasterization;
	create.pMultisampleState   = &multisample;
	create.pColorBlendState    = &blend;
	create.pDynamicState       = &dynamic;
	create.layout              = layout;
	vk::Pipeline pipeline      = nullptr;
	RequireVulkanSuccess(device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &pipeline),
	                     "create video-out fullscreen pipeline");
	device.destroyShaderModule(fragment, nullptr);
	device.destroyShaderModule(vertex, nullptr);
	return pipeline;
}

// A combined image sampler at binding 0, pushed per draw.
static vk::DescriptorSetLayout CreatePushedSamplerLayout(vk::Device device) {
	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eCombinedImageSampler, 1,
	                                              vk::ShaderStageFlagBits::eFragment};
	vk::DescriptorSetLayoutCreateInfo    descriptors {};
	descriptors.flags              = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptors.bindingCount       = 1;
	descriptors.pBindings          = &binding;
	vk::DescriptorSetLayout layout = nullptr;
	RequireVulkanSuccess(device.createDescriptorSetLayout(&descriptors, nullptr, &layout),
	                     "create video-out descriptor layout");
	return layout;
}

bool PresentNeedsFilter(vk::Extent2D source, vk::Extent2D target) noexcept {
	const bool shrinks = source.width > target.width || source.height > target.height;
	const bool halves =
	    source.width == 2ull * target.width && source.height == 2ull * target.height;
	return shrinks && !halves && source.width <= 2ull * target.width &&
	       source.height <= 2ull * target.height;
}

void PresentFilter::Record(vk::Device device, vk::CommandBuffer command, vk::ImageView source_view,
                           uint32_t level, vk::Extent2D source, vk::ImageView target_view,
                           vk::Format target_format, vk::Extent2D target) {
	struct Constants {
		int32_t  size[2];
		float    scale[2];
		uint32_t level;
	};
	static_assert(sizeof(Constants) == 20);
	if (m_layout == nullptr) {
		vk::SamplerCreateInfo sampler {};
		sampler.magFilter    = vk::Filter::eLinear;
		sampler.minFilter    = vk::Filter::eLinear;
		sampler.mipmapMode   = vk::SamplerMipmapMode::eNearest;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		sampler.maxLod       = VK_LOD_CLAMP_NONE;
		RequireVulkanSuccess(device.createSampler(&sampler, nullptr, &m_sampler),
		                     "create video-out downscale sampler");
		m_descriptors = CreatePushedSamplerLayout(device);
		const vk::PushConstantRange  constants {vk::ShaderStageFlagBits::eFragment, 0,
		                                        sizeof(Constants)};
		vk::PipelineLayoutCreateInfo layout {};
		layout.setLayoutCount         = 1;
		layout.pSetLayouts            = &m_descriptors;
		layout.pushConstantRangeCount = 1;
		layout.pPushConstantRanges    = &constants;
		RequireVulkanSuccess(device.createPipelineLayout(&layout, nullptr, &m_layout),
		                     "create video-out downscale pipeline layout");
	}
	if (m_pipeline != nullptr && m_format != target_format) {
		device.destroyPipeline(m_pipeline, nullptr);
		m_pipeline = nullptr;
	}
	if (m_pipeline == nullptr) {
		m_pipeline = CreateFullscreenPipeline(device, m_layout, GPU_VIDEO_OUT_DOWNSCALE_SPV,
		                                      target_format, false);
		m_format   = target_format;
	}

	const vk::DescriptorImageInfo image {m_sampler, source_view,
	                                     vk::ImageLayout::eShaderReadOnlyOptimal};
	vk::WriteDescriptorSet        write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
	write.pImageInfo      = &image;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_layout, 0, 1, &write);
	const Constants constants {
	    {static_cast<int32_t>(source.width), static_cast<int32_t>(source.height)},
	    {static_cast<float>(source.width) / static_cast<float>(target.width),
		 static_cast<float>(source.height) / static_cast<float>(target.height)},
	    level};
	command.pushConstants(m_layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(constants),
	                      &constants);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);
	vk::RenderingAttachmentInfo attachment {};
	attachment.imageView   = target_view;
	attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachment.loadOp      = vk::AttachmentLoadOp::eDontCare;
	attachment.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = target;
	rendering.layerCount           = 1;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &attachment;
	command.beginRendering(&rendering);
	const vk::Viewport viewport {
	    0.0f, 0.0f, static_cast<float>(target.width), static_cast<float>(target.height),
	    0.0f, 1.0f};
	const vk::Rect2D scissor {{0, 0}, target};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &scissor);
	command.draw(3, 1, 0, 0);
	command.endRendering();
}

void PresentFilter::Release(vk::Device device) {
	device.destroyPipeline(m_pipeline, nullptr);
	device.destroyPipelineLayout(m_layout, nullptr);
	device.destroyDescriptorSetLayout(m_descriptors, nullptr);
	device.destroySampler(m_sampler, nullptr);
	m_pipeline    = nullptr;
	m_layout      = nullptr;
	m_descriptors = nullptr;
	m_sampler     = nullptr;
	m_format      = vk::Format::eUndefined;
}

void PresentFsr::Draw(vk::CommandBuffer command, vk::Pipeline pipeline, vk::ImageView source_view,
                      vk::ImageView target_view, vk::Extent2D target, const void* constants,
                      uint32_t constants_size) const {
	const vk::DescriptorImageInfo image {m_sampler, source_view,
	                                     vk::ImageLayout::eShaderReadOnlyOptimal};
	vk::WriteDescriptorSet        write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
	write.pImageInfo      = &image;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_layout, 0, 1, &write);
	command.pushConstants(m_layout, vk::ShaderStageFlagBits::eFragment, 0, constants_size, constants);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	vk::RenderingAttachmentInfo attachment {};
	attachment.imageView   = target_view;
	attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachment.loadOp      = vk::AttachmentLoadOp::eDontCare;
	attachment.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = target;
	rendering.layerCount           = 1;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &attachment;
	command.beginRendering(&rendering);
	const vk::Viewport viewport {
	    0.0f, 0.0f, static_cast<float>(target.width), static_cast<float>(target.height),
	    0.0f, 1.0f};
	const vk::Rect2D scissor {{0, 0}, target};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &scissor);
	command.draw(3, 1, 0, 0);
	command.endRendering();
}

void PresentFsr::Record(GraphicContext& graphics, vk::CommandBuffer command,
                        vk::ImageView source_view, vk::Extent2D source, vk::ImageView target_view,
                        vk::Format target_format, vk::Extent2D target) {
	const auto device = graphics.device;
	if (m_layout == nullptr) {
		vk::SamplerCreateInfo sampler {};
		sampler.magFilter    = vk::Filter::eLinear;
		sampler.minFilter    = vk::Filter::eLinear;
		sampler.mipmapMode   = vk::SamplerMipmapMode::eNearest;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		RequireVulkanSuccess(device.createSampler(&sampler, nullptr, &m_sampler),
		                     "create FSR sampler");
		m_descriptors = CreatePushedSamplerLayout(device);
		const vk::PushConstantRange  constants {vk::ShaderStageFlagBits::eFragment, 0, 64};
		vk::PipelineLayoutCreateInfo layout {};
		layout.setLayoutCount         = 1;
		layout.pSetLayouts            = &m_descriptors;
		layout.pushConstantRangeCount = 1;
		layout.pPushConstantRanges    = &constants;
		RequireVulkanSuccess(device.createPipelineLayout(&layout, nullptr, &m_layout),
		                     "create FSR pipeline layout");
	}
	if (m_easu != nullptr && m_format != target_format) {
		device.destroyPipeline(m_easu, nullptr);
		device.destroyPipeline(m_rcas, nullptr);
		m_easu = nullptr;
		m_rcas = nullptr;
	}
	if (m_easu == nullptr) {
		m_easu   = CreateFullscreenPipeline(device, m_layout, GPU_VIDEO_OUT_FSR_EASU_SPV,
		                                    target_format, false);
		m_rcas   = CreateFullscreenPipeline(device, m_layout, GPU_VIDEO_OUT_FSR_RCAS_SPV,
		                                    target_format, false);
		m_format = target_format;
	}
	if (m_scaled != nullptr && (m_scaled->extent.width != target.width ||
	                            m_scaled->extent.height != target.height ||
	                            m_scaled->format != target_format)) {
		device.destroyImageView(m_scaled_view, nullptr);
		graphics.DeleteImage(*m_scaled);
		m_scaled.reset();
		m_scaled_view = nullptr;
	}
	if (m_scaled == nullptr) {
		m_scaled = std::make_unique<VulkanImage>();
		vk::ImageCreateInfo create {};
		create.imageType     = vk::ImageType::e2D;
		create.extent        = vk::Extent3D {target.width, target.height, 1};
		create.mipLevels     = 1;
		create.arrayLayers   = 1;
		create.format        = target_format;
		create.tiling        = vk::ImageTiling::eOptimal;
		create.initialLayout = vk::ImageLayout::eUndefined;
		create.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
		create.sharingMode = vk::SharingMode::eExclusive;
		create.samples     = vk::SampleCountFlagBits::e1;
		if (!graphics.CreateImage(create, *m_scaled)) {
			EXIT("failed to allocate the FSR image, extent=%ux%u\n", target.width, target.height);
		}
		vk::ImageViewCreateInfo view {};
		view.image            = m_scaled->image;
		view.viewType         = vk::ImageViewType::e2D;
		view.format           = target_format;
		view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		RequireVulkanSuccess(device.createImageView(&view, nullptr, &m_scaled_view),
		                     "create FSR image view");
	}

	const auto barrier = [&](vk::ImageLayout from, vk::ImageLayout to, vk::AccessFlags source_access,
	                         vk::AccessFlags target_access, vk::PipelineStageFlags source_stage,
	                         vk::PipelineStageFlags target_stage) {
		vk::ImageMemoryBarrier change {};
		change.srcAccessMask       = source_access;
		change.dstAccessMask       = target_access;
		change.oldLayout           = from;
		change.newLayout           = to;
		change.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		change.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		change.image               = m_scaled->image;
		change.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		command.pipelineBarrier(source_stage, target_stage, vk::DependencyFlags {}, 0, nullptr, 0,
		                        nullptr, 1, &change);
	};
	// FsrEasuCon: the whole source to the whole target.
	const auto  bits = [](float value) { return std::bit_cast<uint32_t>(value); };
	const float sw   = static_cast<float>(source.width);
	const float sh   = static_cast<float>(source.height);
	const float tw   = static_cast<float>(target.width);
	const float th   = static_cast<float>(target.height);
	const uint32_t easu[16] {
	    bits(sw / tw),        bits(sh / th),        bits(0.5f * sw / tw - 0.5f), bits(0.5f * sh / th - 0.5f),
	    bits(1.0f / sw),      bits(1.0f / sh),      bits(1.0f / sw),             bits(-1.0f / sh),
	    bits(-1.0f / sw),     bits(2.0f / sh),      bits(1.0f / sw),             bits(2.0f / sh),
	    bits(0.0f),           bits(4.0f / sh),      0,                           0};
	barrier(vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {},
	        vk::AccessFlagBits::eColorAttachmentWrite, vk::PipelineStageFlagBits::eFragmentShader,
	        vk::PipelineStageFlagBits::eColorAttachmentOutput);
	Draw(command, m_easu, source_view, m_scaled_view, target, easu, sizeof(easu));
	barrier(vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
	        vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead,
	        vk::PipelineStageFlagBits::eColorAttachmentOutput,
	        vk::PipelineStageFlagBits::eFragmentShader);
	// FsrRcasCon: the sharpness in stops, 0 the sharpest. AMD suggests 0.2, which draws the
	// stair steps of a frame with little antialiasing; a stop is the default here.
	const float    sharpness = std::exp2(-0.1f * static_cast<float>(Config::FsrSoftnessTenths()));
	const uint32_t rcas[16] {bits(sharpness), 0, 0, 0};
	Draw(command, m_rcas, m_scaled_view, target_view, target, rcas, sizeof(rcas));
}

void PresentFsr::Release(GraphicContext& graphics) {
	const auto device = graphics.device;
	device.destroyPipeline(m_easu, nullptr);
	device.destroyPipeline(m_rcas, nullptr);
	device.destroyPipelineLayout(m_layout, nullptr);
	device.destroyDescriptorSetLayout(m_descriptors, nullptr);
	device.destroySampler(m_sampler, nullptr);
	if (m_scaled != nullptr) {
		device.destroyImageView(m_scaled_view, nullptr);
		graphics.DeleteImage(*m_scaled);
		m_scaled.reset();
	}
	m_easu        = nullptr;
	m_rcas        = nullptr;
	m_layout      = nullptr;
	m_descriptors = nullptr;
	m_sampler     = nullptr;
	m_scaled_view = nullptr;
	m_format      = vk::Format::eUndefined;
}

void Presenter::Frame::Clear(CommandBuffer& command_buffer, const vk::ClearColorValue& color) {
	command_buffer.EndRendering();
	auto command = command_buffer.Handle();
	Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	command.clearColorImage(image.image, vk::ImageLayout::eTransferDstOptimal, &color, 1, &range);
}

class Swapchain final {
public:
	enum class Status : uint8_t { Success, Recreate, SurfaceLost };

	explicit Swapchain(WindowContext& window): m_window(window) {}
	~Swapchain();
	KYTY_CLASS_NO_COPY(Swapchain);

	void                 Create();
	void                 Recreate(bool surface_lost = false);
	[[nodiscard]] bool   NeedsResize() const;
	[[nodiscard]] Status AcquireNextImage();
	[[nodiscard]] bool   PrepareSystemOverlay();
	// `generated`, when given, is shown in place of `source`: an image the size of the
	// swapchain, in the transfer-source layout (see FrameGeneration).
	void     RecordPresentCommands(CommandBuffer& command, Presenter::Frame* source,
	                               const Presenter::Layer& overlay, bool draw_system_overlay,
	                               vk::Image generated = nullptr);
	uint64_t Submit(CommandScheduler& scheduler);
	[[nodiscard]] Status Present();

	[[nodiscard]] uint32_t ImageCount() const noexcept {
		return static_cast<uint32_t>(m_images.size());
	}
	[[nodiscard]] vk::Format Format() const noexcept { return m_format; }
	[[nodiscard]] vk::Extent2D Extent() const noexcept { return m_extent; }
	// bruno-astro-1025 (KYTY_FSR3_FG): the swapchain is the FidelityFX frame generation swapchain.
	[[nodiscard]] bool             IsFsr3() const noexcept { return m_fsr3; }
	[[nodiscard]] vk::SwapchainKHR Handle() const noexcept { return m_handle; }

private:
	void Destroy();
	void DrawOverlay(vk::CommandBuffer command, const Presenter::Layer& layer);

	WindowContext&   m_window;
	vk::SwapchainKHR m_handle = nullptr;
	vk::Format       m_format = vk::Format::eUndefined;
	vk::Extent2D     m_extent {};
	// Drawable pixel size observed when this swapchain was created.
	vk::Extent2D                   m_window_extent {};
	std::vector<vk::Image>         m_images;
	std::vector<vk::ImageView>     m_image_views;
	std::vector<vk::Semaphore>     m_image_acquired;
	std::vector<vk::Semaphore>     m_render_complete;
	std::unique_ptr<SystemOverlay> m_system_overlay;
	vk::DescriptorSetLayout        m_overlay_descriptors = nullptr;
	vk::PipelineLayout             m_overlay_layout      = nullptr;
	vk::Pipeline                   m_overlay_pipeline    = nullptr;
	vk::Sampler                    m_overlay_sampler     = nullptr;
	PresentFilter                  m_filter;
	PresentFsr                     m_fsr;
	uint32_t                       m_image_index         = static_cast<uint32_t>(-1);
	uint32_t                       m_frame_index         = 0;
	bool                           m_fsr3                = false;
};

struct Presenter::Impl {
	explicit Impl(WindowContext& owner)
	    : renderer(*owner.render_context), window(owner), swapchain(owner),
	      present_scheduler(renderer, owner.graphic_ctx), frames(owner, present_scheduler) {
		EXIT_IF(owner.render_context == nullptr);
		swapchain.Create();
		frames.Initialize(swapchain.ImageCount(), swapchain.Format());
		// (bruno-astro-1025: brunoKytyPs5's own FSR 3 interpolation stays off while the todojuegos67-bot frame
		// generation is asked for: DLSS-G (KYTY_FRAMEGEN) or the FSR 3 frame generation swapchain (KYTY_FSR3_FG).)
		if (Config::FrameGenerationEnabled() && !FrameGen::Requested() && !Fsr3::Requested()) {
			frame_generation = std::make_unique<FrameGeneration>(owner.graphic_ctx);
			held_thread      = std::jthread([this](std::stop_token stop) { HeldThread(stop); });
		}
	}

	~Impl() {
		if (held_thread.joinable()) {
			held_thread.request_stop();
			held_cv.notify_all();
			held_thread.join();
		}
	}

	void RecoverSwapchain(Swapchain::Status status) {
		LOGF("Recovering Vulkan swapchain%s\n",
		     status == Swapchain::Status::SurfaceLost ? " and surface" : "");
		swapchain.Recreate(status == Swapchain::Status::SurfaceLost);
		frames.SetFormat(swapchain.Format());
	}

	Image& ResolveSurface(const ImageInfo& info, ImageId* id_out = nullptr) {
		TextureCache::ImageDesc desc {};
		desc.info                  = info;
		desc.view_info.format      = info.pixel_format;
		desc.view_info.type        = vk::ImageViewType::e2D;
		desc.view_info.aspect      = vk::ImageAspectFlagBits::eColor;
		desc.view_info.base_level  = 0;
		desc.view_info.level_count = 1;
		desc.view_info.base_layer  = 0;
		desc.view_info.layer_count = 1;
		desc.view_info.usage       = vk::ImageUsageFlagBits::eTransferSrc;
		desc.type                  = TextureCache::BindingType::VideoOut;

		auto&      cache      = renderer.GetTextureCache();
		const auto image_id   = cache.FindImage(desc);
		auto&      image      = cache.GetImage(image_id);
		image.usage.video_out = true;
		cache.UpdateImage(image_id);
		if (id_out != nullptr) {
			*id_out = image_id;
		}
		return image;
	}
	// new_frame is false when the last frame is shown again (the title counts only new ones).
	void Present(bool new_frame = true);
	// Presents once. With `generate_ms` above zero (the time since the previous new frame) it
	// tries to show the frame generated between the previous one and layer 0 instead of layer 0
	// itself, and returns whether it did.
	bool PresentOnce(bool new_frame, float generate_ms);
	// Shows the frame whose generated predecessor went out first, if it is still waiting.
	// Called with present_mutex held.
	void FlushHeld();
	void HeldThread(std::stop_token stop);

	RenderContext&        renderer;
	WindowContext&        window;
	Swapchain             swapchain;
	CommandScheduler      present_scheduler;
	FramePool             frames;
	Common::Mutex         present_mutex;
	std::array<Layer, 2>  layers {};
	std::atomic<uint64_t> presented_overlay_revision {0};

	// Frame generation. A new frame goes out in two presents, the generated one and then the
	// frame itself, both from held_thread at times of its own choosing. A game delivers its
	// frames unevenly (one vblank apart, then two), so the frames are shown on a steady clock
	// half a smoothed frame interval apart instead of as they arrive.
	using Clock = std::chrono::steady_clock;
	enum class Held : uint8_t { None, Generated, Frame };
	std::unique_ptr<FrameGeneration> frame_generation;
	Clock::time_point                last_new_frame {};
	// The smoothed time between new frames, and when the last frame was (or is to be) shown.
	float                            frame_interval_ms = 1000.0f / 60.0f;
	Clock::time_point                last_frame_shown {};
	std::mutex                       held_mutex;
	std::condition_variable          held_cv;
	// What held_thread still has to present of the current frame, and when.
	Held                             held = Held::None;
	Clock::time_point                held_generated_at {};
	Clock::time_point                held_frame_at {};
	std::jthread                          held_thread;
};

void Swapchain::Create() {
	auto& graphics = m_window.graphic_ctx;
	EXIT_IF(graphics.device == nullptr);
	EXIT_IF(m_window.surface == nullptr);

	// Capture the size before querying surface capabilities. If the window changes
	// during creation, the next presentation will detect the newer size.
	{
		Common::LockGuard lock(m_window.mutex);
		m_window_extent = {graphics.screen_width, graphics.screen_height};
	}
	EXIT_IF(m_window_extent.width == 0);
	EXIT_IF(m_window_extent.height == 0);
	m_window.RefreshSurfaceCapabilities();
	const auto& surface = m_window.surface_capabilities;
	EXIT_NOT_IMPLEMENTED(surface.formats.empty());

	m_extent = surface.capabilities.currentExtent;
	if (m_extent.width == std::numeric_limits<uint32_t>::max()) {
		m_extent.width =
		    std::clamp(m_window_extent.width, surface.capabilities.minImageExtent.width,
		               surface.capabilities.maxImageExtent.width);
		m_extent.height =
		    std::clamp(m_window_extent.height, surface.capabilities.minImageExtent.height,
		               surface.capabilities.maxImageExtent.height);
	}
	uint32_t image_count = surface.capabilities.minImageCount + 1;
	if (surface.capabilities.maxImageCount != 0) {
		image_count = std::min(image_count, surface.capabilities.maxImageCount);
	}
	const auto transform =
	    surface.capabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity
	        ? vk::SurfaceTransformFlagBitsKHR::eIdentity
	        : surface.capabilities.currentTransform;
	const auto composite =
	    surface.capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque
	        ? vk::CompositeAlphaFlagBitsKHR::eOpaque
	        : vk::CompositeAlphaFlagBitsKHR::eInherit;

	vk::SurfaceFormatKHR format {vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
	if (surface.formats.size() != 1 || surface.formats.front().format != vk::Format::eUndefined) {
		const auto it = std::find_if(surface.formats.begin(), surface.formats.end(),
		                             [](const vk::SurfaceFormatKHR& candidate) {
			                             return candidate.format == vk::Format::eB8G8R8A8Unorm ||
			                                    candidate.format == vk::Format::eR8G8B8A8Unorm;
		                             });
		if (it == surface.formats.end()) {
			EXIT("no supported UNORM swapchain format\n");
		}
		format = *it;
	}
	m_format                      = format.format;
	const auto swapchain_features = graphics.GetFormatProperties(m_format).optimalTilingFeatures;
	if (!static_cast<bool>(swapchain_features & vk::FormatFeatureFlagBits::eBlitDst)) {
		EXIT("swapchain format cannot be a blit destination: format=%d\n",
		     static_cast<int>(m_format));
	}

	vk::SwapchainCreateInfoKHR create_info {};
	create_info.sType            = vk::StructureType::eSwapchainCreateInfoKHR;
	create_info.surface          = m_window.surface;
	create_info.minImageCount    = image_count;
	create_info.imageFormat      = format.format;
	create_info.imageColorSpace  = format.colorSpace;
	create_info.imageExtent      = m_extent;
	create_info.imageArrayLayers = 1;
	create_info.imageUsage =
	    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
	create_info.imageSharingMode = vk::SharingMode::eExclusive;
	create_info.preTransform     = transform;
	create_info.compositeAlpha   = composite;
	switch (Config::GetPresentMode()) {
		case Config::PresentMode::Mailbox:
			create_info.presentMode = vk::PresentModeKHR::eMailbox;
			break;
		case Config::PresentMode::Immediate:
			create_info.presentMode = vk::PresentModeKHR::eImmediate;
			break;
		case Config::PresentMode::Fifo:
		default: create_info.presentMode = vk::PresentModeKHR::eFifo; break;
	}
	if (std::find(surface.present_modes.begin(), surface.present_modes.end(),
	              create_info.presentMode) == surface.present_modes.end()) {
		LOGF("warning: requested present mode is unavailable; falling back to Fifo\n");
		create_info.presentMode = vk::PresentModeKHR::eFifo;
	}
	create_info.clipped = VK_TRUE;
	// bruno-astro-1025 (KYTY_FSR3_FG): the FidelityFX frame generation swapchain in place of the driver's (false:
	// not asked for, or not made: the driver's).
	m_fsr3 = Fsr3::CreateSwapchain(graphics, create_info, &m_handle);
	if (!m_fsr3) {
		RequireVulkanSuccess(graphics.device.createSwapchainKHR(&create_info, nullptr, &m_handle),
		                     "vkCreateSwapchainKHR");
	}
	EXIT_IF(m_handle == nullptr);
	if (FrameGen::Requested() || m_fsr3) {
		std::printf("Swapchain: %ux%u, present mode %d, %u images%s\n", m_extent.width, m_extent.height,
		            static_cast<int>(create_info.presentMode), image_count,
		            m_fsr3 ? " (FSR 3 frame generation)" : " (DLSS-G through Streamline)");
		std::fflush(stdout);
	}

	m_images = EnumerateVulkan<vk::Image>(
	    "vkGetSwapchainImagesKHR", [&](uint32_t* count, vk::Image* images) {
		    return m_fsr3 ? Fsr3::GetSwapchainImages(graphics, m_handle, count, images)
		                  : graphics.device.getSwapchainImagesKHR(m_handle, count, images);
	    });
	EXIT_NOT_IMPLEMENTED(m_images.empty());

	m_image_views.resize(m_images.size());
	for (size_t i = 0; i < m_images.size(); i++) {
		vk::ImageViewCreateInfo view {};
		view.sType                           = vk::StructureType::eImageViewCreateInfo;
		view.image                           = m_images[i];
		view.viewType                        = vk::ImageViewType::e2D;
		view.format                          = m_format;
		view.components                      = {};
		view.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
		view.subresourceRange.baseArrayLayer = 0;
		view.subresourceRange.baseMipLevel   = 0;
		view.subresourceRange.layerCount     = 1;
		view.subresourceRange.levelCount     = 1;
		RequireVulkanSuccess(graphics.device.createImageView(&view, nullptr, &m_image_views[i]),
		                     "vkCreateImageView");
		EXIT_IF(m_image_views[i] == nullptr);
	}

	vk::SemaphoreCreateInfo semaphore_info {};
	semaphore_info.sType = vk::StructureType::eSemaphoreCreateInfo;
	m_image_acquired.resize(m_images.size());
	m_render_complete.resize(m_images.size());
	for (size_t i = 0; i < m_images.size(); i++) {
		RequireVulkanSuccess(
		    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_image_acquired[i]),
		    "create swapchain image-acquired semaphore");
		RequireVulkanSuccess(
		    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_render_complete[i]),
		    "create swapchain render-complete semaphore");
	}
	m_image_index = static_cast<uint32_t>(-1);
	m_frame_index = 0;
}

Swapchain::~Swapchain() {
	Destroy();
}

void Swapchain::Destroy() {
	if (m_handle == nullptr && m_image_acquired.empty() && m_render_complete.empty() &&
	    m_image_views.empty()) {
		return;
	}
	auto& graphics = m_window.graphic_ctx;

	{
		Common::LockGuard queue_lock(graphics.queue_mutex);
		RequireVulkanSuccess(graphics.queue.waitIdle(), "wait for swapchain queue");
	}
	if (m_system_overlay != nullptr) {
		m_system_overlay->ReleaseVulkan();
	}
	graphics.device.destroyPipeline(m_overlay_pipeline, nullptr);
	graphics.device.destroyPipelineLayout(m_overlay_layout, nullptr);
	graphics.device.destroyDescriptorSetLayout(m_overlay_descriptors, nullptr);
	graphics.device.destroySampler(m_overlay_sampler, nullptr);
	m_overlay_pipeline    = nullptr;
	m_overlay_layout      = nullptr;
	m_overlay_descriptors = nullptr;
	m_overlay_sampler     = nullptr;
	m_filter.Release(graphics.device);
	m_fsr.Release(graphics);

	for (const auto semaphore: m_image_acquired) {
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
	}
	for (const auto semaphore: m_render_complete) {
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
	}
	for (const auto view: m_image_views) {
		if (view != nullptr) {
			graphics.device.destroyImageView(view, nullptr);
		}
	}
	if (m_handle != nullptr) {
		if (m_fsr3) {
			Fsr3::DestroySwapchain(graphics, m_handle);
		} else {
			graphics.device.destroySwapchainKHR(m_handle, nullptr);
		}
	}
	m_fsr3 = false;

	m_handle        = nullptr;
	m_format        = vk::Format::eUndefined;
	m_extent        = {};
	m_window_extent = {};
	m_image_index   = static_cast<uint32_t>(-1);
	m_frame_index   = 0;
	m_images.clear();
	m_image_views.clear();
	m_image_acquired.clear();
	m_render_complete.clear();
}

void Swapchain::Recreate(bool surface_lost) {
	if (FrameGen::Enabled()) {
		FrameGen::SwapchainChanging();
	}
	Destroy();
	if (surface_lost) {
#if defined(__APPLE__)
		// Surface recreation goes through SDL_Vulkan_CreateSurface, which touches the
		// window's view/layer and must run on the main thread on macOS.
		EXIT_IF(!SDL_RunOnMainThread(
		    [](void* window) { static_cast<WindowContext*>(window)->RecreateSurface(); }, &m_window,
		    true));
#else
		m_window.RecreateSurface();
#endif
	}
	Create();
}

bool Swapchain::NeedsResize() const {
	Common::LockGuard lock(m_window.mutex);
	return m_window_extent.width != m_window.graphic_ctx.screen_width ||
	       m_window_extent.height != m_window.graphic_ctx.screen_height;
}

Swapchain::Status Swapchain::AcquireNextImage() {
	EXIT_IF(m_handle == nullptr || m_frame_index >= m_image_acquired.size());
	m_image_index     = static_cast<uint32_t>(-1);
	const auto result =
	    m_fsr3 ? Fsr3::AcquireNextImage(m_window.graphic_ctx, m_handle, m_image_acquired[m_frame_index],
	                                    &m_image_index)
	           : m_window.graphic_ctx.device.acquireNextImageKHR(
	                 m_handle, std::numeric_limits<uint64_t>::max(), m_image_acquired[m_frame_index],
	                 nullptr, &m_image_index);
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eSuboptimalKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eSuboptimalKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorOutOfDateKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorOutOfDateKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorUnknown:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorUnknown\n");
			return Status::Recreate;
		case vk::Result::eErrorSurfaceLostKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorSurfaceLostKHR\n");
			return Status::SurfaceLost;
		default: EXIT("vkAcquireNextImageKHR failed: %s\n", vk::to_string(result).c_str());
	}
	EXIT_IF(m_image_index >= m_images.size());
	return Status::Success;
}

bool Swapchain::PrepareSystemOverlay() {
	if (m_system_overlay == nullptr) {
		m_system_overlay = std::make_unique<SystemOverlay>(m_window.graphic_ctx);
	}
	return m_system_overlay->PrepareFrame(m_extent, m_format, ImageCount());
}

void Swapchain::DrawOverlay(vk::CommandBuffer command, const Presenter::Layer& layer) {
	auto device = m_window.graphic_ctx.device;
	if (m_overlay_pipeline == nullptr) {
		vk::SamplerCreateInfo sampler {};
		sampler.magFilter    = vk::Filter::eLinear;
		sampler.minFilter    = vk::Filter::eLinear;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		RequireVulkanSuccess(device.createSampler(&sampler, nullptr, &m_overlay_sampler),
		                     "create video-out overlay sampler");
		m_overlay_descriptors = CreatePushedSamplerLayout(device);
		const vk::PushConstantRange alpha {vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t)};
		vk::PipelineLayoutCreateInfo layout {};
		layout.setLayoutCount         = 1;
		layout.pSetLayouts            = &m_overlay_descriptors;
		layout.pushConstantRangeCount = 1;
		layout.pPushConstantRanges    = &alpha;
		RequireVulkanSuccess(device.createPipelineLayout(&layout, nullptr, &m_overlay_layout),
		                     "create video-out overlay pipeline layout");
		m_overlay_pipeline = CreateFullscreenPipeline(device, m_overlay_layout,
		                                              GPU_VIDEO_OUT_OVERLAY_SPV, m_format, true);
	}
	auto& frame = *layer.frame;
	if (frame.view == nullptr) {
		vk::ImageViewCreateInfo view {};
		view.image            = frame.image.image;
		view.viewType         = vk::ImageViewType::e2D;
		view.format           = frame.image.format;
		view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		RequireVulkanSuccess(device.createImageView(&view, nullptr, &frame.view),
		                     "create video-out overlay image view");
	}
	const vk::DescriptorImageInfo image {m_overlay_sampler, frame.view,
	                                     vk::ImageLayout::eShaderReadOnlyOptimal};
	vk::WriteDescriptorSet        write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
	write.pImageInfo      = &image;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_overlay_layout, 0, 1, &write);
	const uint32_t premultiplied = layer.premultiplied_alpha;
	command.pushConstants(m_overlay_layout, vk::ShaderStageFlagBits::eFragment, 0,
	                      sizeof(premultiplied), &premultiplied);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, m_overlay_pipeline);
	vk::RenderingAttachmentInfo attachment {};
	attachment.imageView   = m_image_views[m_image_index];
	attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
	attachment.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = m_extent;
	rendering.layerCount           = 1;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &attachment;
	command.beginRendering(&rendering);
	const vk::Viewport viewport {
	    0.0f, 0.0f, static_cast<float>(m_extent.width), static_cast<float>(m_extent.height),
	    0.0f, 1.0f};
	const vk::Rect2D scissor {{0, 0}, m_extent};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &scissor);
	command.draw(3, 1, 0, 0);
	command.endRendering();
}

void Swapchain::RecordPresentCommands(CommandBuffer& command, Presenter::Frame* source,
                                      const Presenter::Layer& overlay, bool draw_system_overlay,
                                      vk::Image generated) {
	if (generated != nullptr) {
		source = nullptr;
	}
	EXIT_IF(m_image_index >= m_images.size());
	auto       vk_command      = command.Handle();
	// The overlays bind their own graphics pipelines and dynamic state.
	command.InvalidateGraphicsState();
	const bool draw_overlay    = overlay.frame != nullptr;
	const bool draw_attachment = draw_overlay || draw_system_overlay;
	if (source != nullptr) {
		source->Transit(vk_command, vk::ImageLayout::eTransferSrcOptimal,
		                vk::AccessFlagBits2::eTransferRead);
	}
	if (draw_overlay) {
		overlay.frame->Transit(vk_command, vk::ImageLayout::eShaderReadOnlyOptimal,
		                       vk::AccessFlagBits2::eShaderRead);
	}

	// A large frame shrinks through its mip chain first, so the final blit never skips source
	// texels (see PresentSourceLevel). Shrinking by less than 2x draws with PresentFilter.
	uint32_t     source_level = 0;
	vk::Extent2D blit_extent {};
	bool         filtered = false;
	bool         fsr      = false; // Scaled up with FSR 1 (see PresentFsr); drawn, like filtered.
	if (source != nullptr) {
		const vk::Extent2D source_extent {source->image.extent.width, source->image.extent.height};
		source_level = PresentSourceLevel(source_extent, m_extent, source->image.mip_levels);
		RecordPresentDownscale(vk_command, source->image.image, source_extent, source_level);
		blit_extent = MipExtent(source_extent, source_level);
		filtered    = PresentNeedsFilter(blit_extent, m_extent);
		fsr         = Config::FsrUpscalingEnabled() && source_level == 0 &&
		              blit_extent.width <= m_extent.width && blit_extent.height <= m_extent.height &&
		              (blit_extent.width < m_extent.width || blit_extent.height < m_extent.height);
		filtered    = filtered || fsr;
		if (filtered) {
			if (source->mips_view == nullptr) {
				vk::ImageViewCreateInfo view {};
				view.image            = source->image.image;
				view.viewType         = vk::ImageViewType::e2D;
				view.format           = source->image.format;
				view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0,
				                         source->image.mip_levels, 0, 1};
				RequireVulkanSuccess(
				    m_window.graphic_ctx.device.createImageView(&view, nullptr, &source->mips_view),
				    "create presentation source view");
			}
			source->Transit(vk_command, vk::ImageLayout::eShaderReadOnlyOptimal,
			                vk::AccessFlagBits2::eShaderRead);
		}
	}

	vk::ImageMemoryBarrier to_transfer {};
	to_transfer.sType = vk::StructureType::eImageMemoryBarrier;
	to_transfer.dstAccessMask =
	    filtered ? vk::AccessFlagBits::eColorAttachmentWrite : vk::AccessFlagBits::eTransferWrite;
	to_transfer.oldLayout = vk::ImageLayout::eUndefined;
	to_transfer.newLayout =
	    filtered ? vk::ImageLayout::eColorAttachmentOptimal : vk::ImageLayout::eTransferDstOptimal;
	to_transfer.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.image                           = m_images[m_image_index];
	to_transfer.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_transfer.subresourceRange.baseMipLevel   = 0;
	to_transfer.subresourceRange.levelCount     = 1;
	to_transfer.subresourceRange.baseArrayLayer = 0;
	to_transfer.subresourceRange.layerCount     = 1;
	// Match the acquire wait stage so the layout transition cannot precede acquisition.
	const auto write_stage = filtered ? vk::PipelineStageFlagBits::eColorAttachmentOutput
	                                  : vk::PipelineStageFlagBits::eTransfer;
	vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, write_stage,
	                           vk::DependencyFlags {}, 0, nullptr, 0, nullptr, 1, &to_transfer);

	if (fsr) {
		m_fsr.Record(m_window.graphic_ctx, vk_command, source->mips_view, blit_extent,
		             m_image_views[m_image_index], m_format, m_extent);
	} else if (filtered) {
		m_filter.Record(m_window.graphic_ctx.device, vk_command, source->mips_view, source_level,
		                blit_extent, m_image_views[m_image_index], m_format, m_extent);
	} else if (source != nullptr) {
		vk::ImageBlit region {};
		region.srcSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
		region.srcSubresource.mipLevel       = source_level;
		region.srcSubresource.baseArrayLayer = 0;
		region.srcSubresource.layerCount     = 1;
		region.srcOffsets[1].x               = static_cast<int>(blit_extent.width);
		region.srcOffsets[1].y               = static_cast<int>(blit_extent.height);
		region.srcOffsets[1].z               = 1;
		region.dstSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
		region.dstSubresource.mipLevel       = 0;
		region.dstSubresource.baseArrayLayer = 0;
		region.dstSubresource.layerCount     = 1;
		region.dstOffsets[1].x               = static_cast<int>(m_extent.width);
		region.dstOffsets[1].y               = static_cast<int>(m_extent.height);
		region.dstOffsets[1].z               = 1;
		vk_command.blitImage(source->image.image, vk::ImageLayout::eTransferSrcOptimal,
		                     m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal, 1,
		                     &region, vk::Filter::eLinear);
	} else if (generated != nullptr) {
		vk::ImageBlit region {};
		region.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		region.srcOffsets[1]  = vk::Offset3D {static_cast<int>(m_extent.width),
		                                      static_cast<int>(m_extent.height), 1};
		region.dstSubresource = region.srcSubresource;
		region.dstOffsets[1]  = region.srcOffsets[1];
		vk_command.blitImage(generated, vk::ImageLayout::eTransferSrcOptimal,
		                     m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal, 1,
		                     &region, vk::Filter::eNearest);
	} else {
		const vk::ClearColorValue black {};
		vk_command.clearColorImage(m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal,
		                           &black, 1, &to_transfer.subresourceRange);
	}

	vk::ImageMemoryBarrier to_present {};
	to_present.sType               = vk::StructureType::eImageMemoryBarrier;
	to_present.srcAccessMask =
	    filtered ? vk::AccessFlagBits::eColorAttachmentWrite : vk::AccessFlagBits::eTransferWrite;
	to_present.dstAccessMask       = draw_attachment ? vk::AccessFlagBits::eColorAttachmentRead |
	                                                       vk::AccessFlagBits::eColorAttachmentWrite
	                                                 : vk::AccessFlagBits::eMemoryRead;
	to_present.oldLayout           = to_transfer.newLayout;
	to_present.newLayout           = draw_attachment ? vk::ImageLayout::eColorAttachmentOptimal
	                                                 : vk::ImageLayout::ePresentSrcKHR;
	to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_present.image               = m_images[m_image_index];
	to_present.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_present.subresourceRange.baseMipLevel   = 0;
	to_present.subresourceRange.levelCount     = 1;
	to_present.subresourceRange.baseArrayLayer = 0;
	to_present.subresourceRange.layerCount     = 1;
	vk_command.pipelineBarrier(write_stage,
	                           draw_attachment ? vk::PipelineStageFlagBits::eColorAttachmentOutput
	                                           : vk::PipelineStageFlagBits::eAllCommands,
	                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
	                           &to_present);
	if (draw_attachment) {
		if (draw_overlay) {
			DrawOverlay(vk_command, overlay);
			if (draw_system_overlay) {
				to_present.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
				to_present.oldLayout     = vk::ImageLayout::eColorAttachmentOptimal;
				vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
				                           vk::PipelineStageFlagBits::eColorAttachmentOutput,
				                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0,
				                           nullptr, 1, &to_present);
			}
		}
		if (draw_system_overlay) {
			m_system_overlay->Record(vk_command, m_image_views[m_image_index]);
		}
		to_present.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
		to_present.dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		to_present.oldLayout     = vk::ImageLayout::eColorAttachmentOptimal;
		to_present.newLayout     = vk::ImageLayout::ePresentSrcKHR;
		vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
		                           vk::PipelineStageFlagBits::eAllCommands,
		                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
		                           &to_present);
	}
}

uint64_t Swapchain::Submit(CommandScheduler& scheduler) {
	EXIT_IF(m_frame_index >= m_image_acquired.size() || m_image_index >= m_render_complete.size());
	SubmitInfo submit;
	submit.AddWait(m_image_acquired[m_frame_index], 1, vk::PipelineStageFlagBits::eTransfer);
	submit.AddSignal(m_render_complete[m_image_index]);
	return scheduler.Submit(submit);
}

Swapchain::Status Swapchain::Present() {
	EXIT_IF(m_image_index >= m_render_complete.size());
	const auto         ready = m_render_complete[m_image_index];
	vk::PresentInfoKHR present {};
	present.sType              = vk::StructureType::ePresentInfoKHR;
	present.swapchainCount     = 1;
	present.pSwapchains        = &m_handle;
	present.pImageIndices      = &m_image_index;
	present.pWaitSemaphores    = &ready;
	present.waitSemaphoreCount = 1;

	vk::Result result;
	if (m_fsr3) {
		// The frame generation swapchain queues it (its threads pace and present; its submissions take the
		// graphics queue's lock themselves, Fsr3::GameQueueSubmit).
		result = Fsr3::QueuePresent(m_window.graphic_ctx, present);
	} else {
		DrainStats::ReasonScope reason(DrainStats::Reason::PresentFrame);
		DrainStats::WaitTimer   held(DrainStats::Kind::Submit);
		Common::LockGuard       lock(m_window.graphic_ctx.queue_mutex);
		result = m_window.graphic_ctx.queue.presentKHR(&present);
	}
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eSuboptimalKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eSuboptimalKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorOutOfDateKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eErrorOutOfDateKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorSurfaceLostKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eErrorSurfaceLostKHR\n");
			return Status::SurfaceLost;
		default: EXIT("vkQueuePresentKHR failed: %s\n", vk::to_string(result).c_str());
	}
	m_frame_index = (m_frame_index + 1u) % static_cast<uint32_t>(m_images.size());
	return Status::Success;
}

// KYTY_DEBUG_FRAME_WATCH=<folder>: shrinks every presented frame, measures its brightness once
// the GPU has finished it, and saves (as PPM, with the frame before) the frames whose brightness
// or share of near-white pixels jumps away from the frames around them.
class FrameWatch final {
public:
	static FrameWatch* Get() {
		static FrameWatch* watch = [] () -> FrameWatch* {
			const char* folder = std::getenv("KYTY_DEBUG_FRAME_WATCH");
			return folder != nullptr && folder[0] != '\0' ? new FrameWatch(folder) : nullptr;
		}();
		return watch;
	}

	void Record(GraphicContext& graphics, CommandScheduler& scheduler, vk::CommandBuffer command,
	            const VulkanImage& source, uint64_t frame_num) {
		const auto index = m_next;
		auto&      slot  = m_slots[index];
		if (slot.busy.load(std::memory_order_acquire)) {
			return;
		}
		m_next = (m_next + 1u) % Slots;
		if (slot.image.image == nullptr && !Create(graphics, slot)) {
			return;
		}
		vk::ImageMemoryBarrier2 to_dst {};
		to_dst.srcStageMask        = vk::PipelineStageFlagBits2::eTransfer;
		to_dst.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
		to_dst.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
		to_dst.oldLayout           = vk::ImageLayout::eUndefined;
		to_dst.newLayout           = vk::ImageLayout::eTransferDstOptimal;
		to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		to_dst.image               = slot.image.image;
		to_dst.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers    = &to_dst;
		command.pipelineBarrier2(dependency);

		vk::ImageBlit region {};
		region.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		region.srcOffsets[1]  = vk::Offset3D {static_cast<int>(source.extent.width),
		                                      static_cast<int>(source.extent.height), 1};
		region.dstSubresource = region.srcSubresource;
		region.dstOffsets[1]  = vk::Offset3D {static_cast<int>(Width), static_cast<int>(Height), 1};
		command.blitImage(source.image, vk::ImageLayout::eTransferSrcOptimal, slot.image.image,
		                  vk::ImageLayout::eTransferDstOptimal, 1, &region, vk::Filter::eLinear);

		auto to_src          = to_dst;
		to_src.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		to_src.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
		to_src.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
		to_src.newLayout     = vk::ImageLayout::eTransferSrcOptimal;
		dependency.pImageMemoryBarriers = &to_src;
		command.pipelineBarrier2(dependency);

		vk::BufferImageCopy copy {};
		copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		copy.imageExtent      = vk::Extent3D {Width, Height, 1};
		command.copyImageToBuffer(slot.image.image, vk::ImageLayout::eTransferSrcOptimal,
		                          slot.buffer, 1, &copy);
		vk::MemoryBarrier2 to_host {};
		to_host.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		to_host.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		to_host.dstStageMask  = vk::PipelineStageFlagBits2::eHost;
		to_host.dstAccessMask = vk::AccessFlagBits2::eHostRead;
		vk::DependencyInfo host_dependency {};
		host_dependency.memoryBarrierCount = 1;
		host_dependency.pMemoryBarriers    = &to_host;
		command.pipelineBarrier2(host_dependency);

		slot.busy.store(true, std::memory_order_release);
		scheduler.DeferOperation([this, index, frame_num, &graphics] {
			auto& done = m_slots[index];
			vmaInvalidateAllocation(graphics.allocator, done.allocation, 0, VK_WHOLE_SIZE);
			Analyze(static_cast<const uint8_t*>(done.mapped), frame_num);
			done.busy.store(false, std::memory_order_release);
		});
	}

private:
	static constexpr uint32_t Width  = 480;
	static constexpr uint32_t Height = 270;
	static constexpr uint32_t Slots  = 8;
	static constexpr size_t   Bytes  = size_t {Width} * Height * 4u;
	static constexpr size_t   Window = 12;

	struct Slot {
		VulkanImage       image;
		VkBuffer          buffer     = VK_NULL_HANDLE;
		VmaAllocation     allocation = nullptr;
		void*             mapped     = nullptr;
		std::atomic<bool> busy {false};
	};

	explicit FrameWatch(std::string folder): m_folder(std::move(folder)) {
		std::error_code error;
		std::filesystem::create_directories(m_folder, error);
		std::printf("frame-watch: saving outlying frames to %s\n", m_folder.c_str());
	}

	static bool Create(GraphicContext& graphics, Slot& slot) {
		vk::ImageCreateInfo image {};
		image.imageType   = vk::ImageType::e2D;
		image.extent      = vk::Extent3D {Width, Height, 1};
		image.mipLevels   = 1;
		image.arrayLayers = 1;
		image.format      = vk::Format::eR8G8B8A8Unorm;
		image.tiling      = vk::ImageTiling::eOptimal;
		image.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
		image.samples = vk::SampleCountFlagBits::e1;
		if (!graphics.CreateImage(image, slot.image)) {
			return false;
		}
		vk::BufferCreateInfo buffer {};
		buffer.size  = Bytes;
		buffer.usage = vk::BufferUsageFlagBits::eTransferDst;
		VmaAllocationCreateInfo allocation {};
		allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		allocation.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
		                   VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		VmaAllocationInfo mapped {};
		if (vmaCreateBuffer(graphics.allocator, static_cast<const VkBufferCreateInfo*>(buffer),
		                    &allocation, &slot.buffer, &slot.allocation, &mapped) != VK_SUCCESS) {
			return false;
		}
		slot.mapped = mapped.pMappedData;
		return slot.mapped != nullptr;
	}

	static double Median(std::deque<double> values) {
		std::sort(values.begin(), values.end());
		return values[values.size() / 2];
	}

	void Save(const std::vector<uint8_t>& pixels, uint64_t frame_num, const char* kind) {
		const auto path = m_folder + "/frame_" + std::to_string(frame_num) + "_" + kind + ".ppm";
		std::FILE* file = std::fopen(path.c_str(), "wb");
		if (file == nullptr) {
			return;
		}
		std::fprintf(file, "P6\n%u %u\n255\n", Width, Height);
		std::vector<uint8_t> rgb(size_t {Width} * Height * 3u);
		for (size_t i = 0; i < size_t {Width} * Height; i++) {
			rgb[i * 3]     = pixels[i * 4];
			rgb[i * 3 + 1] = pixels[i * 4 + 1];
			rgb[i * 3 + 2] = pixels[i * 4 + 2];
		}
		std::fwrite(rgb.data(), 1, rgb.size(), file);
		std::fclose(file);
	}

	void Analyze(const uint8_t* data, uint64_t frame_num) {
		std::vector<uint8_t> pixels(data, data + Bytes);
		uint64_t             sum   = 0;
		uint64_t             white = 0;
		for (size_t i = 0; i < size_t {Width} * Height; i++) {
			const uint32_t r = pixels[i * 4], g = pixels[i * 4 + 1], b = pixels[i * 4 + 2];
			sum += r + g + b;
			white += r > 215 && g > 215 && b > 215 ? 1u : 0u;
		}
		const double mean      = static_cast<double>(sum) / (3.0 * 255.0 * Width * Height);
		const double whiteness = static_cast<double>(white) / (static_cast<double>(Width) * Height);
		std::lock_guard lock(m_mutex);
		if (m_means.size() >= 4) {
			const double usual_mean  = Median(m_means);
			const double usual_white = Median(m_whites);
			if ((std::abs(mean - usual_mean) > 0.10 || whiteness - usual_white > 0.20) &&
			    m_saved < 400) {
				m_saved += 2;
				std::printf("frame-watch: frame=%llu mean=%.3f usual=%.3f white=%.3f usual=%.3f\n",
				            static_cast<unsigned long long>(frame_num), mean, usual_mean,
				            whiteness, usual_white);
				Save(pixels, frame_num, "salto");
				if (!m_previous.empty()) {
					Save(m_previous, m_previous_frame, "antes");
				}
			}
		}
		if (frame_num % 300u == 0u) {
			std::printf("frame-watch: frame=%llu mean=%.3f white=%.3f\n",
			            static_cast<unsigned long long>(frame_num), mean, whiteness);
		}
		m_means.push_back(mean);
		m_whites.push_back(whiteness);
		if (m_means.size() > Window) {
			m_means.pop_front();
			m_whites.pop_front();
		}
		m_previous       = std::move(pixels);
		m_previous_frame = frame_num;
	}

	std::string             m_folder;
	std::array<Slot, Slots> m_slots;
	uint32_t                m_next = 0;
	std::mutex              m_mutex;
	std::deque<double>      m_means;
	std::deque<double>      m_whites;
	std::vector<uint8_t>    m_previous;
	uint64_t                m_previous_frame = 0;
	uint32_t                m_saved          = 0;
};

Presenter::Presenter(WindowContext& window): m_impl(std::make_unique<Impl>(window)) {}

Presenter::~Presenter() = default;

Presenter::Frame& Presenter::PrepareFrame(CommandBuffer& buffer, const ImageInfo& info) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(buffer.IsInvalid());
	auto frame_format = info.pixel_format;
	switch (frame_format) {
		case vk::Format::eR8G8B8A8Srgb: frame_format = vk::Format::eR8G8B8A8Unorm; break;
		case vk::Format::eB8G8R8A8Srgb: frame_format = vk::Format::eB8G8R8A8Unorm; break;
		default: break;
	}
	auto* frame = m_impl->frames.Acquire({info.extent.width, info.extent.height}, frame_format);
	Common::LockGuard render_lock(m_impl->renderer.GetMutex());
	ImageId           image_id;
	auto&             image = m_impl->ResolveSurface(info, &image_id);
	if (image.backing.format == vk::Format::eUndefined) {
		EXIT("unsupported presentation source, image=%p\n", static_cast<const void*>(&image));
	}
	frame->Configure(m_impl->window.graphic_ctx,
	                 {image.backing.extent.width, image.backing.extent.height}, frame_format);
	frame->CopyFrom(buffer, image);
	// bruno-astro-1025: frame generation's inputs, in the guest frame's command buffer (a title whose passes are not
	// named: zero motion vectors and a far depth at the flip image's size, FrameGen::GenericInputs).
	FgDiscover::OnFlip(image.info.data.address, image.backing.extent.width, image.backing.extent.height,
	                   static_cast<uint32_t>(image.backing.format), VideoOut::PlayerButtonSeen());
	frame->scene    = FrameGen::TakeScene();
	frame->fg.valid = false;
	if (FrameGen::Capturing()) {
		FrameGen::PrepareInputs(m_impl->window.graphic_ctx, m_impl->renderer.GetTextureCache(), buffer,
		                        frame->fg, image_id, image);
	}
	FrameGen::ReflexGuestFlip(frame->fg);
	if (auto* watch = FrameWatch::Get()) [[unlikely]] {
		frame->Transit(buffer.Handle(), vk::ImageLayout::eTransferSrcOptimal,
		               vk::AccessFlagBits2::eTransferRead);
		watch->Record(m_impl->window.graphic_ctx, m_impl->renderer.GetCommandScheduler(),
		              buffer.Handle(), frame->image,
		              static_cast<uint64_t>(m_impl->renderer.GetGpu().GetFrameNum()));
	}
	return *frame;
}

Presenter::Frame& Presenter::PrepareBlankFrame(uint32_t width, uint32_t height, bool opaque,
                                               CommandBuffer* producer) {
	KYTY_PROFILER_FUNCTION();
	auto              format = m_impl->frames.GetFormat();
	auto*             frame  = m_impl->frames.Acquire({width, height}, format);
	Common::LockGuard render_lock(m_impl->renderer.GetMutex());
	frame->Configure(m_impl->window.graphic_ctx, {width, height}, format);
	vk::ClearColorValue clear {};
	clear.float32[3] = opaque ? 1.0f : 0.0f;
	if (producer != nullptr) {
		EXIT_IF(producer->IsInvalid());
		frame->Clear(*producer, clear);
	} else {
		auto& command = m_impl->present_scheduler.BeginCommand();
		frame->Clear(command, clear);
		frame->present_tick = m_impl->present_scheduler.Submit();
	}
	return *frame;
}

bool Presenter::PresentLastFrame() {
	Common::LockGuard lock(m_impl->present_mutex);
	m_impl->FlushHeld();
	if (m_impl->layers[0].frame == nullptr && m_impl->layers[1].frame == nullptr) {
		return false;
	}
	m_impl->Present(false);
	return true;
}

bool Presenter::IsGuestPaused() const noexcept {
	return m_impl->window.loop.paused.load(std::memory_order_acquire);
}

bool Presenter::NeedsSystemOverlayRefresh() const noexcept {
	const auto visual = GetSystemOverlayVisualState();
	return visual.active ||
	       visual.revision != m_impl->presented_overlay_revision.load(std::memory_order_acquire);
}

RenderContext& Presenter::Renderer() const noexcept {
	return m_impl->renderer;
}

void Presenter::Present(Frame& frame) {
	const Layer layer {&frame, 0, false};
	Present(std::span(&layer, 1));
}

void Presenter::Present(std::span<const Layer> layers) {
	Common::LockGuard lock(m_impl->present_mutex);
	// The held frame is one of the layers about to be replaced.
	m_impl->FlushHeld();
	for (const auto& layer: layers) {
		EXIT_IF(layer.bus < 0 || layer.bus >= static_cast<int>(m_impl->layers.size()));
		m_impl->frames.ValidateForPresent(layer.frame);
		auto& previous = m_impl->layers[layer.bus];
		if (previous.frame != nullptr) {
			m_impl->frames.Release(previous.frame);
		}
		previous = layer;
	}
	m_impl->Present();
}

void Presenter::ClearLayer(int bus) {
	if (static_cast<size_t>(bus) >= m_impl->layers.size()) {
		return;
	}
	Common::LockGuard lock(m_impl->present_mutex);
	m_impl->FlushHeld();
	auto& layer = m_impl->layers[bus];
	if (layer.frame != nullptr) {
		m_impl->frames.Release(layer.frame);
		layer = {};
	}
}

void Presenter::Impl::FlushHeld() {
	{
		std::lock_guard held_lock(held_mutex);
		if (held == Held::None) {
			return;
		}
		// A generated frame that has not gone out yet is dropped: its turn has passed.
		held             = Held::None;
		last_frame_shown = Clock::now();
	}
	held_cv.notify_all();
	PresentOnce(true, 0.0f);
}

void Presenter::Impl::HeldThread(std::stop_token stop) {
	for (;;) {
		Held stage = Held::None;
		{
			std::unique_lock held_lock(held_mutex);
			held_cv.wait(held_lock, [&] { return stop.stop_requested() || held != Held::None; });
			if (stop.stop_requested()) {
				return;
			}
			stage               = held;
			const auto deadline = stage == Held::Generated ? held_generated_at : held_frame_at;
			// Someone else may show the frame first: a new frame arriving early, or a shutdown.
			held_cv.wait_until(held_lock, deadline,
			                   [&] { return stop.stop_requested() || held != stage; });
			if (stop.stop_requested()) {
				return;
			}
			if (held != stage) {
				continue;
			}
		}
		Common::LockGuard lock(present_mutex);
		{
			std::lock_guard held_lock(held_mutex);
			if (held != stage) {
				continue;
			}
			held = stage == Held::Generated ? Held::Frame : Held::None;
		}
		if (stage == Held::Generated) {
			if (!PresentOnce(true, frame_interval_ms)) {
				// No generated frame this time: the frame itself went out in its place.
				std::lock_guard held_lock(held_mutex);
				held             = Held::None;
				last_frame_shown = Clock::now();
			}
		} else {
			PresentOnce(true, 0.0f);
		}
	}
}

void Presenter::Impl::Present(bool new_frame) {
	FlushHeld();
	if (frame_generation != nullptr && frame_generation->Available() && new_frame &&
	    layers[0].frame != nullptr) {
		const auto now = Clock::now();
		if (last_new_frame != Clock::time_point {}) {
			const auto interval = std::clamp(
			    std::chrono::duration<float, std::milli>(now - last_new_frame).count(), 4.0f,
			    100.0f);
			frame_interval_ms += (interval - frame_interval_ms) * 0.1f;
		}
		last_new_frame  = now;
		const auto half = std::chrono::microseconds(static_cast<int64_t>(frame_interval_ms * 500.0f));
		{
			std::lock_guard held_lock(held_mutex);
			// Half an interval after the previous frame, but never in the past, and never so
			// late that a pause in the game would hold this frame back.
			auto generated_at = last_frame_shown + half;
			if (generated_at < now || generated_at > now + 2 * half) {
				generated_at = now;
			}
			held_generated_at = generated_at;
			held_frame_at     = generated_at + half;
			last_frame_shown  = held_frame_at;
			held              = Held::Generated;
		}
		held_cv.notify_all();
		return;
	}
	PresentOnce(new_frame, 0.0f);
}

bool Presenter::Impl::PresentOnce(bool new_frame, float generate_ms) {
	KYTY_PROFILER_FUNCTION();

	const auto overlay_visual = GetSystemOverlayVisualState();
	// Some window systems keep presenting an old swapchain after a resize.
	if (swapchain.NeedsResize()) {
		RecoverSwapchain(Swapchain::Status::Recreate);
	}
	bool                   generated = false;
	FrameGen::FrameInputs* fg_inputs = nullptr;
	for (uint32_t attempt = 0; attempt < 2; attempt++) {
		auto status = swapchain.AcquireNextImage();
		if (status != Swapchain::Status::Success) {
			RecoverSwapchain(status);
			continue;
		}
		{
			Common::LockGuard render_lock(renderer.GetMutex());
			auto&             command = present_scheduler.BeginCommand();
			const bool        draw_system_overlay =
			    overlay_visual.active && swapchain.PrepareSystemOverlay();
			vk::Image generated_image = nullptr;
			if (generate_ms > 0.0f) {
				// The frame is brought down through its mip chain as presentation does, and the
				// interpolation takes it from there at the size shown.
				auto*              source     = layers[0].frame;
				auto               vk_command = command.Handle();
				const vk::Extent2D source_extent {source->image.extent.width,
				                                  source->image.extent.height};
				source->Transit(vk_command, vk::ImageLayout::eTransferSrcOptimal,
				                vk::AccessFlagBits2::eTransferRead);
				const auto level = PresentSourceLevel(source_extent, swapchain.Extent(),
				                                      source->image.mip_levels);
				RecordPresentDownscale(vk_command, source->image.image, source_extent, level);
				if (frame_generation->Record(vk_command, source->image.image, level,
				                             MipExtent(source_extent, level), swapchain.Extent(),
				                             generate_ms)) {
					generated_image = frame_generation->Output();
				}
			}
			generated = generated_image != nullptr;
			swapchain.RecordPresentCommands(command, layers[0].frame, layers[1],
			                                draw_system_overlay, generated_image);
			// bruno-astro-1025: this present's frame generation inputs (a new frame of the game's only: an
			// idle refresh of the last frame, a menu or a movie has none; generation is off for it).
			fg_inputs = nullptr;
			if (new_frame && layers[0].frame != nullptr && layers[0].frame->fg.valid &&
			    (FrameGen::Enabled() || swapchain.IsFsr3())) {
				fg_inputs = &layers[0].frame->fg;
			}
			if (FrameGen::Enabled() && fg_inputs != nullptr) {
				FrameGen::ScaleHudless(window.graphic_ctx, command.Handle(), *fg_inputs, swapchain.Extent(),
				                       vk::Rect2D {{0, 0}, swapchain.Extent()}, swapchain.Format());
			}
			const auto tick = swapchain.Submit(present_scheduler);
			for (const auto& layer: layers) {
				if (layer.frame != nullptr) {
					layer.frame->present_tick = tick;
				}
			}
		}
		// KYTY_FRAMEGEN: Streamline takes the present (vkQueuePresentKHR is its hook) and adds the generated
		// frames; KYTY_FSR3_FG: the frame generation swapchain gets this frame's inputs.
		const bool generate = FrameGen::Enabled();
		if (generate) {
			FrameGen::SetPresentReflexIndex(fg_inputs != nullptr ? fg_inputs->reflex_index : 0u);
			FrameGen::BeforePresent(fg_inputs, swapchain.Extent(), vk::Rect2D {{0, 0}, swapchain.Extent()},
			                        swapchain.Format(), swapchain.ImageCount());
		}
		if (swapchain.IsFsr3()) {
			Fsr3::BeforePresent(window.graphic_ctx, fg_inputs, swapchain.Handle(), swapchain.Extent(),
			                    swapchain.Format());
		}
		status = swapchain.Present();
		if (generate) {
			FrameGen::AfterPresent();
		}
		if (status != Swapchain::Status::Success) {
			RecoverSwapchain(status);
			continue;
		}

		presented_overlay_revision.store(overlay_visual.revision, std::memory_order_release);
		if (frame_generation != nullptr && std::getenv("KYTY_DEBUG_FRAME_PACING") != nullptr) {
			static auto last = std::chrono::steady_clock::now();
			const auto  now  = std::chrono::steady_clock::now();
			LOGF("fg-present %s %.2f ms\n",
			     generated ? "gen" : (generate_ms > 0.0f ? "real-nogen" : "real"),
			     std::chrono::duration<float, std::milli>(now - last).count());
			last = now;
		}
		// A generated frame is a present, not a frame of the game.
		window.UpdateTitle(new_frame && !generated);
		return generated;
	}
	LOGF("Vulkan presentation retry exhausted; dropping frame\n");
	return false;
}

void Presenter::Discard(Frame& frame) {
	m_impl->frames.Release(&frame);
}

WindowContext::WindowContext() = default;

} // namespace Libs::Graphics
