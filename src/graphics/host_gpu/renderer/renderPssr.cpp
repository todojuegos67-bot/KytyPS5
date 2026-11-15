#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/upstreamPaths.h"
#include "common/threads.h"
#include "gpu_blit_shaders/gpu_pssr_linear_spv.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/fsr3FrameGen.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/upscaler/fsr.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <array>
#include <optional>

namespace Libs::Graphics {
namespace {

struct PssrParameters {
	uint32_t input_size[2];
	uint32_t output_size[2];
	uint32_t output_channels[4];
};

// Upscalers read and write with stages and accesses of their own.
void MemoryBarrier(vk::CommandBuffer command) {
	const vk::MemoryBarrier2 barrier {.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
	                                  .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
	                                  .dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
	                                  .dstAccessMask = vk::AccessFlagBits2::eMemoryRead |
	                                                   vk::AccessFlagBits2::eMemoryWrite};
	command.pipelineBarrier2(
	    vk::DependencyInfo {.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}

} // namespace

bool RenderExecutor::Upscale(vk::CommandBuffer command, const Pssr::Request& request,
                             Dlss::Inputs& inputs) {
	const auto release = [&](std::unique_ptr<Upscaler::Feature>& feature) {
		if (feature != nullptr) {
			m_context.GetCommandScheduler().DeferOperation(
			    [feature = std::move(feature)]() mutable { feature.reset(); });
		}
	};
	auto context =
	    std::ranges::find(m_upscaler_contexts, request.context, &UpscalerContext::context);
	bool reset = (request.flags & Pssr::kReset) != 0;
	if (context == m_upscaler_contexts.end() || !Dlss::Takes(context->creation, request, inputs)) {
		// A render size the upscaler can't take scales linearly.
		const auto creation = Upscaler::Choose(request, inputs);
		if (!creation) {
			return false;
		}
		if (context != m_upscaler_contexts.end()) {
			release(context->feature);
		} else {
			// A library has at most four contexts.
			if (m_upscaler_contexts.size() == 4) {
				release(m_upscaler_contexts.front().feature);
				m_upscaler_contexts.erase(m_upscaler_contexts.begin());
			}
			context = m_upscaler_contexts.insert(m_upscaler_contexts.end(),
			                                     {request.context, {}, nullptr});
		}
		context->creation = *creation;
		context->feature  = Upscaler::Create(command, *creation);
		reset             = true;
	}
	// A failed create or evaluate forgets the context, so the next frame creates it again (as UpscaleFsr).
	if (context->feature == nullptr) {
		m_upscaler_contexts.erase(context);
		return false;
	}
	MemoryBarrier(command);
	if (!context->feature->Evaluate(command, request, inputs, reset)) {
		release(context->feature);
		m_upscaler_contexts.erase(context);
		return false;
	}
	MemoryBarrier(command);
	return true;
}

bool RenderExecutor::UpscaleFsr(vk::CommandBuffer command, const Pssr::Request& request,
                                const Fsr3::Inputs& inputs) {
	const auto release = [&](Fsr3::Context* fsr) {
		if (fsr != nullptr) {
			m_context.GetCommandScheduler().DeferOperation([fsr] { Fsr3::Destroy(fsr); });
		}
	};
	auto creation = Fsr3::CreationFor(request, inputs);
	auto context  = std::ranges::find(m_fsr_contexts, request.context, &FsrContext::context);
	bool reset    = (request.flags & Pssr::kReset) != 0;
	// The game's dynamic resolution swaps between input images of different sizes (1664x936 and
	// 1792x1008 at the Pro title, ~6 times a second): a context made for the larger serves both.
	const auto fits = [&](const Fsr3::Creation& have) {
		return have.max_width >= creation.max_width && have.max_height >= creation.max_height &&
		       have.output_width == creation.output_width &&
		       have.output_height == creation.output_height && have.flags == creation.flags;
	};
	if (context == m_fsr_contexts.end() || !fits(context->creation)) {
		if (context != m_fsr_contexts.end()) {
			creation.max_width  = std::max(creation.max_width, context->creation.max_width);
			creation.max_height = std::max(creation.max_height, context->creation.max_height);
			release(context->fsr);
		} else {
			// A library has at most four contexts.
			if (m_fsr_contexts.size() == 4) {
				release(m_fsr_contexts.front().fsr);
				m_fsr_contexts.erase(m_fsr_contexts.begin());
			}
			context = m_fsr_contexts.insert(m_fsr_contexts.end(), {request.context, {}, nullptr});
		}
		context->creation = creation;
		context->fsr      = Fsr3::Create(m_context.GetGraphics(), creation);
		reset             = true;
	}
	// A failed context is forgotten: its creation would still fit the next request, which would then scale linearly
	// without trying again.
	if (context->fsr == nullptr) {
		m_fsr_contexts.erase(context);
		return false;
	}
	MemoryBarrier(command);
	if (!Fsr3::Dispatch(context->fsr, command, request, inputs, reset)) {
		release(context->fsr);
		m_fsr_contexts.erase(context);
		return false;
	}
	if (Fsr3::FrameGen::Active()) {
		Fsr3::FrameGen::Prepare(command, request, inputs, reset);
	}
	MemoryBarrier(command);
	return true;
}

void RenderExecutor::ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request) {
	EXIT_IF(buffer.IsInvalid() && !KYTY_UPSTREAM_PATH("45b35fafb"));
	m_context.GetCommandScheduler().PopPendingOperations();
	Common::LockGuard lock(m_context.GetMutex());
	ShaderTextureResource input_descriptor {}, output_descriptor {};
	std::copy_n(request.input, 8, input_descriptor.fields);
	std::copy_n(request.output, 8, output_descriptor.fields);
	const auto is_2d = [](const ShaderTextureResource& descriptor) {
		return !descriptor.IsNull() &&
		       (descriptor.Type() == Prospero::ImageType::kColor2D ||
		        descriptor.Type() == Prospero::ImageType::kColor2DArray) &&
		       descriptor.TileMode() != Prospero::TileMode::kDepth;
	};
	EXIT_IF(!is_2d(input_descriptor) || !is_2d(output_descriptor) ||
	        !Pssr::IsColorFormat(input_descriptor.Format()) ||
	        output_descriptor.Format() != Prospero::BufferFormat::k11_11_10Float ||
	        output_descriptor.TileMode() != Prospero::TileMode::kStandard256B ||
	        output_descriptor.BaseLevel() != 0 || output_descriptor.LastLevel() != 0 ||
	        output_descriptor.MaxMip() != 0 || output_descriptor.BaseArray5() != 0 ||
	        output_descriptor.Depth() != 0 || request.width == 0 || request.height == 0);
	const auto input_width = std::max(1u, (input_descriptor.Width5() + 1u) >>
	                                         input_descriptor.BaseLevel());
	const auto input_height = std::max(1u, (input_descriptor.Height5() + 1u) >>
	                                          input_descriptor.BaseLevel());
	EXIT_IF(request.width > input_width || request.height > input_height);
	PssrParameters parameters {{request.width, request.height},
	                           {output_descriptor.Width5() + 1u, output_descriptor.Height5() + 1u},
	                           {4, 4, 4, 4}};
	for (uint32_t component = 0; component < 3; ++component) {
		for (uint32_t source = 0; source < 4; ++source) {
			if (GetDstSel(output_descriptor.DstSelXYZW(), source) == 4u + component) {
				parameters.output_channels[component] = source;
				break;
			}
		}
	}
	const auto resolve = [&](const uint32_t* words, bool output) {
		ShaderRecompiler::IR::ImageResource resource {};
		resource.resource_class = output ? ShaderRecompiler::IR::ImageResourceClass::Storage
		                                 : ShaderRecompiler::IR::ImageResourceClass::Sampled;
		resource.numeric_class = Prospero::TextureNumericClass::Float;
		resource.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
		resource.read = !output;
		resource.written = output;
		ShaderRecompiler::IR::DescriptorValue value {};
		std::copy_n(words, 8, value.dwords.begin());
		value.dword_count = 8;
		auto binding = ResolveTexture(resource, value);
		binding.desc.view_info.level_count = 1;
		binding.desc.view_info.min_lod = 0;
		return binding;
	};
	auto& cache = m_context.GetTextureCache();
	auto input = resolve(request.input, false);
	BindImage(input.image_id, false);
	auto output = resolve(request.output, true);
	BindImage(output.image_id, true);
	EXIT_IF(input.desc.info.data.address < output.desc.info.data.End() &&
	        output.desc.info.data.address < input.desc.info.data.End());
	// Depth, motion vectors and exposure for DLSS or FSR, which write the output unswizzled.
	const std::array<const uint32_t*, 3>         dlss_words {request.depth, request.motion,
	                                                         request.exposure};
	std::array<std::optional<TextureBinding>, 3> dlss_bindings;
	// Upstream's upscaler (DLSS, or FSR from AMD's DLLs) when one is up; else the built-in FSR 3.1 when FSR is chosen.
	const bool dlss = Upscaler::Enabled() && Dlss::Accepts(request);
	const bool fsr  = !Upscaler::Enabled() && Config::GetPssrUpscaler() == Config::PssrUpscaler::FSR &&
	                 Fsr3::Accepts(request);
	if ((dlss || fsr) && parameters.output_channels[0] == 0 &&
	    parameters.output_channels[1] == 1 && parameters.output_channels[2] == 2) {
		for (size_t i = 0; i < dlss_words.size(); ++i) {
			ShaderTextureResource descriptor {};
			std::copy_n(dlss_words[i], 8, descriptor.fields);
			if (!descriptor.IsNull()) {
				dlss_bindings[i] = resolve(dlss_words[i], false);
				BindImage(dlss_bindings[i]->image_id, false);
			}
		}
	}
	const auto acquire = [&](TextureBinding& binding, const uint32_t* words, bool storage) {
		auto& previous = cache.GetImage(binding.image_id);
		if (!previous.registered || previous.binding.needs_rebind) {
			previous.binding = {};
			binding = resolve(words, storage);
			BindImage(binding.image_id, storage);
		}
		binding.image_view = cache.FindTexture(binding.image_id, binding.desc);
	};
	for (size_t i = 0; i < dlss_words.size(); ++i) {
		if (dlss_bindings[i]) {
			acquire(*dlss_bindings[i], dlss_words[i], false);
		}
	}
	acquire(input, request.input, false);
	acquire(output, request.output, true);
	auto& source = cache.GetImage(input.image_id);
	auto& destination = cache.GetImage(output.image_id);
	const auto& source_view = input.desc.view_info;
	const auto& destination_view = output.desc.view_info;
	EXIT_IF(source.info.IsDepth() || destination.info.IsDepth() ||
	        source.backing.samples != 1 || destination.backing.samples != 1 ||
	        (input.image_id == output.image_id &&
	         source_view.base_level == destination_view.base_level &&
	         source_view.base_layer == destination_view.base_layer));

	buffer.EndRendering();
	const auto command = buffer.Handle();
	const auto range   = [](const ImageViewInfo& view) {
		return ImageSubresourceRange {view.base_level, 1, view.base_layer, 1};
	};
	source.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead, range(source_view),
	               command);
	destination.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
	                    range(destination_view), command);
	if (dlss_bindings[0] && dlss) {
		// Whether a binding is a whole one-level, one-layer image, unswizzled in its channels.
		const auto raw = [&](const TextureBinding& binding, uint32_t channels) {
			const auto&      image  = cache.GetImage(binding.image_id);
			const auto&      view   = binding.desc.view_info;
			const auto       extent = binding.desc.info.MipExtent(view.base_level);
			const std::array swizzles {view.mapping.r, view.mapping.g, view.mapping.b};
			for (uint32_t i = 0; i < channels; ++i) {
				if (swizzles[i] != vk::ComponentSwizzle::eIdentity &&
				    swizzles[i] != static_cast<vk::ComponentSwizzle>(VK_COMPONENT_SWIZZLE_R + i)) {
					return false;
				}
			}
			return view.base_level == 0 && view.base_layer == 0 && image.backing.layers == 1 &&
			       image.backing.mip_levels == 1 && image.backing.samples == 1 &&
			       image.backing.extent.width == extent.width &&
			       image.backing.extent.height == extent.height;
		};
		const auto resource = [&](const TextureBinding& binding, bool write) {
			const auto& image  = cache.GetImage(binding.image_id);
			const auto& view   = binding.desc.view_info;
			const auto  extent = binding.desc.info.MipExtent(view.base_level);
			// Upscalers take a depth image in its backing format and depth aspect.
			const bool depth  = image.info.IsDepth();
			const auto aspect = depth ? vk::ImageAspectFlagBits::eDepth : view.aspect;
			return Dlss::Resource {
			    .view       = static_cast<VkImageView>(binding.image_view),
			    .image      = static_cast<VkImage>(image.backing.image),
			    .range      = {static_cast<VkImageAspectFlags>(aspect), view.base_level, 1,
			                   view.base_layer, 1},
			    .format     = static_cast<VkFormat>(depth ? image.backing.format : view.format),
			    .width      = extent.width,
			    .height     = extent.height,
			    .read_write = write};
		};
		for (const auto& binding: dlss_bindings) {
			if (binding) {
				cache.GetImage(binding->image_id)
				    .Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
					         range(binding->desc.view_info), command);
			}
		}
		auto exposure = dlss_bindings[2] ? resource(*dlss_bindings[2], false) : Dlss::Resource {};
		Dlss::Inputs inputs {resource(input, false), resource(*dlss_bindings[0], false),
		                     resource(*dlss_bindings[1], false),
		                     dlss_bindings[2] ? &exposure : nullptr, resource(output, true)};
		const bool   readable =
		    !Fsr::Enabled() || (Fsr::Reads(request, inputs) && raw(input, 3) &&
		                        raw(*dlss_bindings[0], 1) && raw(*dlss_bindings[1], 2) &&
		                        (!dlss_bindings[2] || raw(*dlss_bindings[2], 1)) && raw(output, 3));
		if (readable && Upscale(command, request, inputs)) {
			ResetBindings();
			return;
		}
	}
	if (dlss_bindings[0] && fsr) {
		// FSR's backend views and transitions whole images from level 0: one layer, one sample, the
		// view's format the image's own or one it can take.
		const auto whole = [&](const TextureBinding& binding, uint32_t width,
		                       uint32_t height) -> std::optional<Fsr3::Image> {
			const auto& image = cache.GetImage(binding.image_id);
			const auto& view  = binding.desc.view_info;
			const bool  depth = image.info.IsDepth();
			const auto  format = depth ? image.backing.format : view.format;
			if (view.base_level != 0 || view.base_layer != 0 || image.backing.layers != 1 ||
			    image.backing.samples != 1 || image.held != nullptr ||
			    (format != image.backing.format &&
			     !(image.backing.flags & vk::ImageCreateFlagBits::eMutableFormat)) ||
			    !Fsr3::Supports(static_cast<VkFormat>(format))) {
				return std::nullopt;
			}
			return Fsr3::Image {
			    .image   = static_cast<VkImage>(image.backing.image),
			    .format  = static_cast<VkFormat>(format),
			    .width   = width != 0 ? width : image.backing.extent.width,
			    .height  = height != 0 ? height : image.backing.extent.height,
			    .depth   = depth,
			    .stencil = depth && image.backing.format != vk::Format::eD32Sfloat &&
			               image.backing.format != vk::Format::eD16Unorm};
		};
		const auto color    = whole(input, 0, 0);
		const auto depth    = whole(*dlss_bindings[0], 0, 0);
		const auto motion   = whole(*dlss_bindings[1], 0, 0);
		const auto exposure = dlss_bindings[2] ? whole(*dlss_bindings[2], 0, 0) : std::nullopt;
		const auto upscaled = whole(output, parameters.output_size[0], parameters.output_size[1]);
		if (color && depth && motion && upscaled && (!dlss_bindings[2] || exposure)) {
			for (const auto& binding: dlss_bindings) {
				if (binding) {
					cache.GetImage(binding->image_id)
					    .Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
					             std::nullopt, command);
				}
			}
			source.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
			               std::nullopt, command);
			destination.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
			                    std::nullopt, command);
			const Fsr3::Inputs inputs {*color, *depth, *motion, exposure ? &*exposure : nullptr,
			                           *upscaled};
			if (UpscaleFsr(command, request, inputs)) {
				ResetBindings();
				return;
			}
		}
	}
	auto& graphics = m_context.GetGraphics();
	if (m_pssr_pipeline == nullptr) {
		const std::array bindings {
		    vk::DescriptorSetLayoutBinding {0, vk::DescriptorType::eSampledImage, 1,
		                                    vk::ShaderStageFlagBits::eCompute},
		    vk::DescriptorSetLayoutBinding {1, vk::DescriptorType::eStorageImage, 1,
		                                    vk::ShaderStageFlagBits::eCompute}};
		vk::DescriptorSetLayoutCreateInfo descriptor_info {};
		descriptor_info.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		descriptor_info.bindingCount = bindings.size();
		descriptor_info.pBindings = bindings.data();
		RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(
		                         &descriptor_info, nullptr, &m_pssr_descriptor_layout),
		                     "create upscaling descriptor layout");
		const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0,
		                                  sizeof(PssrParameters)};
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.setLayoutCount = 1;
		layout_info.pSetLayouts = &m_pssr_descriptor_layout;
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges = &push;
		RequireVulkanSuccess(graphics.device.createPipelineLayout(
		                         &layout_info, nullptr, &m_pssr_pipeline_layout),
		                     "create upscaling pipeline layout");
		m_pssr_pipeline =
		    CreateComputePipeline(graphics.device, GPU_PSSR_LINEAR_SPV, m_pssr_pipeline_layout,
			                      "create upscaling pipeline");
	}
	const std::array images {
	    vk::DescriptorImageInfo {nullptr, input.image_view, vk::ImageLayout::eGeneral},
	    vk::DescriptorImageInfo {nullptr, output.image_view, vk::ImageLayout::eGeneral}};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t i = 0; i < writes.size(); ++i) {
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = i == 0 ? vk::DescriptorType::eSampledImage
		                                 : vk::DescriptorType::eStorageImage;
		writes[i].pImageInfo = &images[i];
	}
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pssr_pipeline_layout,
	                             0, writes.size(), writes.data());
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pssr_pipeline);
	command.pushConstants(m_pssr_pipeline_layout, vk::ShaderStageFlagBits::eCompute,
	                      0, sizeof(parameters), &parameters);
	command.dispatch((parameters.output_size[0] + 7) / 8,
	                 (parameters.output_size[1] + 7) / 8, 1);
	ShaderAccessBarrier(command, vk::PipelineStageFlagBits::eComputeShader);
	ResetBindings();
}

} // namespace Libs::Graphics
