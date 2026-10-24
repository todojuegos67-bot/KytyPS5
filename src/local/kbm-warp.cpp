#include "kbm-warp.h"

#include "frame-gen.h"
#include "gpu_tiler_shaders/kbm_warp_spv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace Libs::Graphics::KbmWarp {

namespace {

bool EnvIs(const char* name, const char* value) {
	const char* v = std::getenv(name);
	return v != nullptr && std::string_view(v) == value;
}
float EnvNum(const char* name, float def) {
	const char* v = std::getenv(name);
	if (v == nullptr || *v == '\0') return def;
	const float f = std::strtof(v, nullptr);
	return std::isfinite(f) ? f : def;
}

bool IsDepth(vk::Format format) {
	switch (format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint:
		case vk::Format::eD24UnormS8Uint:
		case vk::Format::eD32Sfloat:
		case vk::Format::eD32SfloatS8Uint:
		case vk::Format::eX8D24UnormPack32: return true;
		default: return false;
	}
}

// An image of ours, (re)made at a size and format.
struct Scratch {
	VulkanImage   image;
	vk::ImageView view = nullptr;
};
bool Ensure(GraphicContext& graphics, Scratch& s, vk::Format format, vk::Extent3D extent, vk::ImageUsageFlags usage) {
	if (s.image.image != nullptr && s.image.format == format && s.image.extent.width == extent.width &&
	    s.image.extent.height == extent.height)
		return true;
	if (s.view != nullptr) graphics.device.destroyImageView(s.view, nullptr);
	s.view = nullptr;
	if (s.image.image != nullptr) graphics.DeleteImage(s.image);
	s.image.image = nullptr;
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = usage;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, s.image)) {
		s.image.image = nullptr;
		return false;
	}
	s.image.state = {};
	if (!(usage & vk::ImageUsageFlagBits::eStorage)) return true;
	vk::ImageViewCreateInfo view {};
	view.image            = s.image.image;
	view.viewType         = vk::ImageViewType::e2D;
	view.format           = format;
	view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view, nullptr, &s.view) != vk::Result::eSuccess) {
		graphics.DeleteImage(s.image);
		s.image.image = nullptr;
		s.view        = nullptr;
		return false;
	}
	return true;
}

struct State {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             sampler    = nullptr;
	bool                    failed     = false;
	Scratch                 color, hudless, motion, depth;
	struct View {
		vk::Image     image  = nullptr;
		vk::Format    format = vk::Format::eUndefined;
		vk::ImageView view   = nullptr;
	};
	std::array<View, 6> views {}; // the presenter's frame images
	uint32_t            next_view = 0;
	float               prev_rect[4] = {0.0f, 0.0f, 1.0f, 1.0f}; // FSR 3: the last real frame's source rectangle (UV)
	// The edge zoom: the fraction of each half-size kept free around the source rectangle, so the pending turn never
	// uncovers an edge. (10-25) Constant (KYTY_KBM_WARP_ZOOM) from the first warped frame: before, it rose at once to what
	// a turn needed and eased back 0.35 s later, so the 3D's scale pumped by up to 5% at every turn (the background
	// "breathing", one of the dizziness reports).
	float               zoom      = 0.0f;
	uint32_t            logs = 0;
} g;

bool EnsurePipeline(GraphicContext& graphics) {
	if (g.pipeline != nullptr) return true;
	if (g.failed) return false;
	g.failed          = true; // until it is made
	const auto device = graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, 4> bindings {};
	for (uint32_t i = 0; i < 4; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = i < 2 ? vk::DescriptorType::eCombinedImageSampler : vk::DescriptorType::eStorageImage;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = static_cast<uint32_t>(bindings.size());
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, 64};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(KBM_WARP_SPV);
	module_info.pCode       = KBM_WARP_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g.layout;
	const auto created         = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g.pipeline = nullptr;
		return false;
	}
	vk::SamplerCreateInfo sampler {};
	sampler.magFilter    = vk::Filter::eLinear;
	sampler.minFilter    = vk::Filter::eLinear;
	sampler.mipmapMode   = vk::SamplerMipmapMode::eNearest;
	sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	sampler.maxLod       = 0.0f;
	if (device.createSampler(&sampler, nullptr, &g.sampler) != vk::Result::eSuccess) return false;
	g.failed = false;
	return true;
}

vk::ImageView FrameView(GraphicContext& graphics, const VulkanImage& image) {
	// A view made for each frame, kept for the next few (a frame image remade may reuse a handle: no lookup by it). A
	// slot is reused 6 frames later, long after its present finished (the presenter waits for each present's work).
	auto& slot  = g.views[g.next_view];
	g.next_view = (g.next_view + 1) % static_cast<uint32_t>(g.views.size());
	if (slot.view != nullptr) graphics.device.destroyImageView(slot.view, nullptr);
	slot = {};
	vk::ImageViewCreateInfo view {};
	view.image            = image.image;
	view.viewType         = vk::ImageViewType::e2D;
	view.format           = image.format;
	view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::ImageView created = nullptr;
	if (graphics.device.createImageView(&view, nullptr, &created) != vk::Result::eSuccess) return nullptr;
	slot = {image.image, image.format, created};
	return created;
}

vk::ImageMemoryBarrier2 Barrier(vk::Image image, vk::ImageLayout from, vk::ImageLayout to, vk::AccessFlags2 src,
                                vk::AccessFlags2 dst) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask       = src;
	barrier.dstStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask       = dst;
	barrier.oldLayout           = from;
	barrier.newLayout           = to;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image;
	barrier.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	return barrier;
}
void Barriers(vk::CommandBuffer command, std::initializer_list<vk::ImageMemoryBarrier2> barriers) {
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
	dependency.pImageMemoryBarriers    = barriers.begin();
	command.pipelineBarrier2(dependency);
}

struct Push {
	float   rect[4];
	float   prev[4];
	float   threshold;
	int32_t radius;
	int32_t mode;
	int32_t write1;
	float   mv_sign;
	float   pad[3];
};
static_assert(sizeof(Push) == 64);

bool Identity(const float* r) {
	return r[0] == 0.0f && r[1] == 0.0f && r[2] == 1.0f && r[3] == 1.0f;
}

void Dispatch(vk::CommandBuffer command, vk::ImageView a, vk::ImageView b, vk::ImageView out0, vk::ImageView out1,
              const Push& push, vk::Extent3D size) {
	vk::DescriptorImageInfo images[4] {};
	images[0] = {g.sampler, a, vk::ImageLayout::eShaderReadOnlyOptimal};
	images[1] = {g.sampler, b, vk::ImageLayout::eShaderReadOnlyOptimal};
	images[2] = {nullptr, out0, vk::ImageLayout::eGeneral};
	images[3] = {nullptr, out1, vk::ImageLayout::eGeneral};
	vk::WriteDescriptorSet writes[4] {};
	for (uint32_t i = 0; i < 4; ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = i < 2 ? vk::DescriptorType::eCombinedImageSampler : vk::DescriptorType::eStorageImage;
		writes[i].pImageInfo      = &images[i];
	}
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g.layout, 0, 4, writes);
	command.pushConstants(g.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
}


} // namespace

bool On() {
	static const bool on = EnvIs("KYTY_KBM_WARP", "1") && EnvIs("KYTY_KBM", "1");
	return on;
}
bool HudlessWanted() {
	static const bool wanted = On() && !EnvIs("KYTY_KBM_WARP_HUDLESS", "0");
	return wanted;
}
bool Fsr3Wanted() {
	// (10-25) Opt-in (KYTY_KBM_WARP_FSR3=1): FSR 3 interpolating warped frames gave double images (the HUD mask's edge,
	// the character against the shifted background), and frame generation already halves the step the eye sees.
	static const bool wanted = HudlessWanted() && EnvIs("KYTY_KBM_WARP_FSR3", "1");
	return wanted;
}
float ZoomMax() {
	static const float zoom_max = std::clamp(EnvNum("KYTY_KBM_WARP_ZOOM", 0.05f), 0.0f, 0.25f);
	return zoom_max;
}
bool InputsShifted() {
	// (A constant zoom: every frame goes through the warp, moving or not, so its scale never pops in or out.)
	return !Identity(g.prev_rect) || (On() && ZoomMax() > 0.0f);
}
void NoteUnshifted() {
	g.prev_rect[0] = g.prev_rect[1] = 0.0f;
	g.prev_rect[2] = g.prev_rect[3] = 1.0f;
}

VulkanImage* Record(GraphicContext& graphics, vk::CommandBuffer command, VulkanImage& final_image,
                    FrameGen::FrameInputs* inputs, float shift_x, float shift_y, bool fsr3) {
	if (!HudlessWanted() || inputs == nullptr || !inputs->valid || !inputs->has_hudless || !inputs->hudless_composite ||
	    inputs->hudless.image == nullptr || inputs->hudless_view == nullptr || final_image.image == nullptr ||
	    !(final_image.usage & vk::ImageUsageFlagBits::eSampled) ||
	    final_image.extent.width != inputs->hudless.extent.width ||
	    final_image.extent.height != inputs->hudless.extent.height || final_image.extent.width == 0 ||
	    final_image.extent.height == 0)
		return nullptr;
	if (!EnsurePipeline(graphics)) return nullptr;
	const vk::Extent3D size {final_image.extent.width, final_image.extent.height, 1};
	const auto         color_usage =
	    vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled;
	if (!Ensure(graphics, g.color, vk::Format::eR16G16B16A16Sfloat, size, color_usage)) return nullptr;
	const auto final_view = FrameView(graphics, final_image);
	if (final_view == nullptr) return nullptr;

	// FSR 3: the inputs too (render pixels), or nothing.
	if (fsr3) {
		auto& depth  = inputs->depth;
		auto& motion = inputs->motion;
		if (depth.image == nullptr || motion.image == nullptr || inputs->motion_view == nullptr ||
		    depth.extent.width != motion.extent.width || depth.extent.height != motion.extent.height ||
		    !IsDepth(depth.format) || IsDepth(motion.format) || !(inputs->hudless.usage & vk::ImageUsageFlagBits::eTransferDst) ||
		    !(motion.usage & vk::ImageUsageFlagBits::eTransferDst) || !(depth.usage & vk::ImageUsageFlagBits::eTransferDst))
			return nullptr;
		const auto blit_both = vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eBlitDst;
		if ((graphics.GetFormatProperties(depth.format).optimalTilingFeatures & blit_both) != blit_both ||
		    !(graphics.GetFormatProperties(motion.format).optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst) ||
		    !(graphics.GetFormatProperties(inputs->hudless.format).optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst))
			return nullptr;
		const vk::Extent3D render {motion.extent.width, motion.extent.height, 1};
		if (!Ensure(graphics, g.hudless, vk::Format::eR16G16B16A16Sfloat, size, color_usage) ||
		    !Ensure(graphics, g.motion, vk::Format::eR16G16B16A16Sfloat, render, color_usage) ||
		    !Ensure(graphics, g.depth, depth.format, render,
		            vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst))
			return nullptr;
	}

	// The edge zoom (constant, 10-25) and the source rectangle.
	{
		g.zoom         = ZoomMax();
		const float hw = 0.5f * static_cast<float>(size.width), hh = 0.5f * static_cast<float>(size.height);
		// The shift held inside the free margin (softly near its end).
		auto keep = [](float v, float m) {
			if (m <= 0.0f) return 0.0f;
			return m * std::tanh(v / m);
		};
		shift_x = keep(shift_x, g.zoom * hw);
		shift_y = keep(shift_y, g.zoom * hh);
	}
	const float scale = 1.0f - g.zoom;
	float       rect[4] = {0.5f - 0.5f * scale + shift_x / static_cast<float>(size.width),
	                       0.5f - 0.5f * scale + shift_y / static_cast<float>(size.height), scale, scale};
	int32_t     rx0 = 0, ry0 = 0, rw = 0, rh = 0; // FSR 3: the rectangle in whole render pixels (the depth blit)
	if (fsr3) {
		const auto& motion = inputs->motion;
		const int   W = static_cast<int>(motion.extent.width), H = static_cast<int>(motion.extent.height);
		rw  = std::clamp(static_cast<int32_t>(std::lround(static_cast<float>(W) * scale)), 1, W);
		rh  = std::clamp(static_cast<int32_t>(std::lround(static_cast<float>(H) * scale)), 1, H);
		rx0 = std::clamp(static_cast<int32_t>(std::lround(rect[0] * static_cast<float>(W))), 0, W - rw);
		ry0 = std::clamp(static_cast<int32_t>(std::lround(rect[1] * static_cast<float>(H))), 0, H - rh);
		// (The image through the same whole render pixels: the inputs stay exactly on it.)
		rect[0] = static_cast<float>(rx0) / static_cast<float>(W);
		rect[1] = static_cast<float>(ry0) / static_cast<float>(H);
		rect[2] = static_cast<float>(rw) / static_cast<float>(W);
		rect[3] = static_cast<float>(rh) / static_cast<float>(H);
	}

	// The color: F where it differs from H (the HUD), H shifted elsewhere.
	static const float threshold = EnvNum("KYTY_KBM_WARP_THRESHOLD", EnvNum("KYTY_FSR3_HUD_THRESHOLD", 3.0f / 1023.0f));
	static const int   radius =
	    std::clamp(static_cast<int>(EnvNum("KYTY_KBM_WARP_RADIUS", EnvNum("KYTY_FSR3_HUD_RADIUS", 2.0f))), 0, 6);
	Barriers(command, {Barrier(final_image.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
	                           vk::AccessFlagBits2::eTransferRead, vk::AccessFlagBits2::eShaderRead)});
	FrameGen::Transition(command, g.color.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
	                     vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader);
	auto* hudless_out = fsr3 ? &g.hudless : &g.color; // (without FSR 3 out1 is out0: the color is written last)
	if (fsr3)
		FrameGen::Transition(command, g.hudless.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
		                     vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader);
	{
		Push push {{rect[0], rect[1], rect[2], rect[3]}, {0.0f, 0.0f, 1.0f, 1.0f}, threshold, radius, 0, fsr3 ? 1 : 0, 1.0f, {}};
		Dispatch(command, final_view, inputs->hudless_view, g.color.view, hudless_out->view, push, size);
	}
	Barriers(command, {Barrier(final_image.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal,
	                           vk::AccessFlagBits2::eShaderRead, vk::AccessFlagBits2::eTransferRead)});
	FrameGen::Transition(command, g.color.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
	                     vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eTransfer);

	if (fsr3) {
		auto& depth  = inputs->depth;
		auto& motion = inputs->motion;
		const vk::Extent3D render {motion.extent.width, motion.extent.height, 1};
		// The vectors (UV units; FSR 3 takes vector * mv_sign * size as pixels to the previous position): the warped
		// pixel's previous position moved by the shift's change, in the vectors' units.
		static const float mv_sign = EnvNum("KYTY_FSR3_MV_SIGN", -1.0f) >= 0.0f ? 1.0f : -1.0f;
		FrameGen::Transition(command, g.motion.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
		                     vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader);
		FrameGen::Transition(command, motion, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
		                     vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader);
		{
			Push push {{rect[0], rect[1], rect[2], rect[3]},
			           {g.prev_rect[0], g.prev_rect[1], g.prev_rect[2], g.prev_rect[3]},
			           0.0f, 0, 1, 0, mv_sign, {}};
			Dispatch(command, inputs->motion_view, inputs->motion_view, g.motion.view, g.motion.view, push, render);
		}
		// Back into the frame's own copies (FSR 3's prepare and HUD-less blit read them after this command buffer).
		FrameGen::Transition(command, g.motion.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
		                     vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eTransfer);
		FrameGen::Transition(command, g.hudless.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
		                     vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eTransfer);
		FrameGen::Transition(command, motion, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
		                     vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
		FrameGen::Transition(command, inputs->hudless, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
		                     vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
		auto whole = [](vk::ImageAspectFlags aspect, vk::Extent3D extent) {
			vk::ImageBlit blit {};
			blit.srcSubresource = {aspect, 0, 0, 1};
			blit.dstSubresource = {aspect, 0, 0, 1};
			blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};
			blit.dstOffsets[1]  = blit.srcOffsets[1];
			return blit;
		};
		{
			const auto blit = whole(vk::ImageAspectFlagBits::eColor, render);
			command.blitImage(g.motion.image.image, vk::ImageLayout::eTransferSrcOptimal, motion.image,
			                  vk::ImageLayout::eTransferDstOptimal, 1, &blit, vk::Filter::eNearest);
		}
		{
			const auto blit = whole(vk::ImageAspectFlagBits::eColor, size);
			command.blitImage(g.hudless.image.image, vk::ImageLayout::eTransferSrcOptimal, inputs->hudless.image,
			                  vk::ImageLayout::eTransferDstOptimal, 1, &blit, vk::Filter::eNearest);
		}
		// The depth: shifted by blits (nearest; the uncovered strip repeats the edge) into ours, copied back.
		FrameGen::Transition(command, depth, vk::ImageAspectFlagBits::eDepth, vk::ImageLayout::eTransferSrcOptimal,
		                     vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eTransfer);
		FrameGen::Transition(command, g.depth.image, vk::ImageAspectFlagBits::eDepth, vk::ImageLayout::eTransferDstOptimal,
		                     vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
		{
			// (10-24) The source rectangle scaled onto the whole image (nearest), as the color and the vectors.
			vk::ImageBlit r {};
			r.srcSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1};
			r.dstSubresource = r.srcSubresource;
			r.srcOffsets[0]  = vk::Offset3D {rx0, ry0, 0};
			r.srcOffsets[1]  = vk::Offset3D {rx0 + rw, ry0 + rh, 1};
			r.dstOffsets[1]  = vk::Offset3D {static_cast<int32_t>(render.width), static_cast<int32_t>(render.height), 1};
			command.blitImage(depth.image, vk::ImageLayout::eTransferSrcOptimal, g.depth.image.image,
			                  vk::ImageLayout::eTransferDstOptimal, 1, &r, vk::Filter::eNearest);
		}
		FrameGen::Transition(command, g.depth.image, vk::ImageAspectFlagBits::eDepth, vk::ImageLayout::eTransferSrcOptimal,
		                     vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eTransfer);
		FrameGen::Transition(command, depth, vk::ImageAspectFlagBits::eDepth, vk::ImageLayout::eTransferDstOptimal,
		                     vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
		{
			vk::ImageCopy copy {};
			copy.srcSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1};
			copy.dstSubresource = copy.srcSubresource;
			copy.extent         = render;
			command.copyImage(g.depth.image.image, vk::ImageLayout::eTransferSrcOptimal, depth.image,
			                  vk::ImageLayout::eTransferDstOptimal, copy);
		}
		// As PrepareInputs left them.
		FrameGen::Transition(command, depth, vk::ImageAspectFlagBits::eDepth, vk::ImageLayout::eShaderReadOnlyOptimal,
		                     vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
		FrameGen::Transition(command, motion, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
		                     vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
		FrameGen::Transition(command, inputs->hudless, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
		                     vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
		for (int i = 0; i < 4; ++i) g.prev_rect[i] = rect[i];
	}
	if (g.logs < 3) {
		++g.logs;
		std::printf("KBM warp: 3D shifted %.1f,%.1f px under a still HUD, edge zoom %.1f%% (HUD-less composite %ux%u)%s\n",
		            shift_x, shift_y, g.zoom * 100.0f, size.width, size.height, fsr3 ? "; FSR 3 inputs (HUD-less, depth, motion) shifted alike" : "");
		std::fflush(stdout);
	}
	return &g.color.image;
}

} // namespace Libs::Graphics::KbmWarp
