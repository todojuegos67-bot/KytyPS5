#include "graphics/host_gpu/renderer/image/image.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <fmt/format.h>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

[[nodiscard]] vk::ImageType HostImageType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1D: return vk::ImageType::e1D;
		case Prospero::ImageType::kColor3D: return vk::ImageType::e3D;
		case Prospero::ImageType::kColor2D: return vk::ImageType::e2D;
		default: EXIT("non-base image type: %u\n", static_cast<uint32_t>(type));
	}
}

[[nodiscard]] vk::ImageCreateFlags
ImageCreateFlags(const GraphicContext& graphics, const ImageInfo& info, vk::ImageUsageFlags usage) {
	vk::ImageCreateFlags flags {};
	if (DepthAspectTransferFormat(info.pixel_format) == vk::Format::eUndefined) {
		flags |= vk::ImageCreateFlagBits::eMutableFormat;
		if (info.IsBlock() && graphics.supports_block_texel_view) {
			flags |= vk::ImageCreateFlagBits::eBlockTexelViewCompatible;
		}
	}
	if (usage & vk::ImageUsageFlagBits::eStorage) {
		flags |= vk::ImageCreateFlagBits::eExtendedUsage;
	}
	if (info.IsVolume()) {
		flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
	}
	if (!info.IsVolume() || !info.IsBlock()) {
		return flags;
	}
	// RADV refuses a BC7 sRGB volume with block texel views (RX 9070 XT, ISSUES.md #16; found by
	// vMohammad24). Optional flags are dropped, least needed first, until the device accepts the
	// sampled image: uncompressed views (only for guest writes, which need storage usage as well),
	// 2D views (IsValidViewType then allows 3D views only), then other formats' views.
	using Bit = vk::ImageCreateFlagBits;
	const vk::ImageCreateFlags drops[] = {
	    {},
	    Bit::eBlockTexelViewCompatible,
	    Bit::eBlockTexelViewCompatible | Bit::e2DArrayCompatible,
	    Bit::e2DArrayCompatible | Bit::eBlockTexelViewCompatible | Bit::eExtendedUsage,
	    Bit::e2DArrayCompatible | Bit::eBlockTexelViewCompatible | Bit::eExtendedUsage |
	        Bit::eMutableFormat,
	};
	for (const auto drop: drops) {
		const auto candidate = flags & ~drop;
		if (graphics.GetImageFormatProperties(
		        info.pixel_format, vk::ImageType::e3D, vk::ImageTiling::eOptimal,
		        vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
		            vk::ImageUsageFlagBits::eSampled,
		        candidate, nullptr) == vk::Result::eSuccess) {
			if (drop) {
				static std::atomic_flag logged = ATOMIC_FLAG_INIT;
				if (!logged.test_and_set(std::memory_order_relaxed)) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Vulkan image: block-compressed volume {} created without flags 0x{:x} "
					    "(the device does not support them)\n",
					    vk::to_string(info.pixel_format),
					    static_cast<vk::ImageCreateFlags::MaskType>(flags & drop)));
				}
			}
			return candidate;
		}
	}
	return flags;
}

[[nodiscard]] bool HasFormatFeature(vk::FormatProperties      properties,
                                    vk::FormatFeatureFlagBits feature) {
	return static_cast<bool>(properties.optimalTilingFeatures & feature);
}

[[nodiscard]] vk::ImageUsageFlags ImageUsageFlags(GraphicContext& graphics, const ImageInfo& info) {
	auto usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	if (info.IsBlock()) {
		usage |= vk::ImageUsageFlagBits::eSampled;
		if (graphics.supports_block_texel_view) {
			const auto storage = usage | vk::ImageUsageFlagBits::eStorage;
			if (graphics.GetImageFormatProperties(
			        info.pixel_format, HostImageType(info.type), vk::ImageTiling::eOptimal, storage,
			        ImageCreateFlags(graphics, info, storage), nullptr) == vk::Result::eSuccess) {
				usage = storage;
			} else {
				static std::atomic_flag warned = ATOMIC_FLAG_INIT;
				if (!warned.test_and_set(std::memory_order_relaxed)) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Warning: format {} does not support storage access; block-compressed "
					    "textures written by the guest will not render.\n",
					    vk::to_string(info.pixel_format)));
				}
			}
		}
		return usage;
	}
	const auto properties = graphics.GetFormatProperties(info.pixel_format);
	if (HasFormatFeature(properties, vk::FormatFeatureFlagBits::eSampledImage)) {
		usage |= vk::ImageUsageFlagBits::eSampled;
	}
	if (DepthAspectTransferFormat(info.pixel_format) != vk::Format::eUndefined) {
		usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
		if (graphics.attachment_feedback_loop_enabled && (usage & vk::ImageUsageFlagBits::eSampled)) {
			usage |= vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT;
		}
		return usage;
	}
	if (HasFormatFeature(properties, vk::FormatFeatureFlagBits::eColorAttachment)) {
		usage |= vk::ImageUsageFlagBits::eColorAttachment;
	}
	if (info.samples == 1) {
		usage |= vk::ImageUsageFlagBits::eStorage;
	}
	return usage;
}

void ValidateOptionalRange(GuestRange range, const char* name) {
	if (!range.ValidOrEmpty()) {
		EXIT("invalid %s image range: address=0x%016llx size=0x%016llx\n", name,
		     static_cast<unsigned long long>(range.address),
		     static_cast<unsigned long long>(range.size));
	}
}

} // namespace

vk::ImageAspectFlags Image::FullAspectMask(vk::Format format) noexcept {
	switch (format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eX8D24UnormPack32:
		case vk::Format::eD32Sfloat: return vk::ImageAspectFlagBits::eDepth;
		case vk::Format::eS8Uint: return vk::ImageAspectFlagBits::eStencil;
		case vk::Format::eD16UnormS8Uint:
		case vk::Format::eD24UnormS8Uint:
		case vk::Format::eD32SfloatS8Uint:
			return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		default: return vk::ImageAspectFlagBits::eColor;
	}
}

Image::Barriers Image::GetBarriers(vk::ImageLayout                      destination_layout,
                                   vk::AccessFlags2                     destination_access,
                                   vk::PipelineStageFlags2              destination_stage,
                                   std::optional<ImageSubresourceRange> range) {
	auto& state              = backing.state;
	auto& subresource_states = backing.subresource_states;
	if (range && info.IsVolume()) {
		range->base_layer  = 0;
		range->layer_count = 1;
	}

	const bool partial =
	    range && (range->base_level != 0 || range->level_count != info.resources.levels ||
	              range->base_layer != 0 || range->layer_count != info.resources.layers);
	const bool has_subresource_states = !subresource_states.empty();

	// Round 167 (images:lean): with the guest's sync points ordering the GPU work, a draw's or
	// dispatch's use of the image in the layout it is in, with no copy before or now, makes no
	// barrier: its accesses join the ones the next barrier waits for. An emulator-made use since
	// the last barrier is not ordered by the guest, so the next use makes its barrier.
	const bool guest_use = GuestImageUses::Active();
	const bool lean =
	    guest_use && !internal_use && Bisect::LeanImages() && GuestGpu::GuestPm4Barriers();
	if (!guest_use) {
		internal_use = true;
	}
	constexpr auto transfer_access =
	    vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite;
	const auto joins = [&](const VulkanImageState& current) {
		return lean && current.layout == destination_layout &&
		       !static_cast<bool>((current.access_mask | destination_access) & transfer_access);
	};
	vk::AccessFlags2        joined_access {};
	vk::PipelineStageFlags2 joined_stage {};
	uint32_t                skipped = 0;

	Barriers barriers;
	if (partial || has_subresource_states) {
		if (!has_subresource_states) {
			subresource_states.resize(info.resources.levels * info.resources.layers, state);
		}

		const uint32_t base_level  = partial ? range->base_level : 0;
		const uint32_t level_count = partial ? range->level_count : info.resources.levels;
		const uint32_t base_layer  = partial ? range->base_layer : 0;
		const uint32_t layer_count = partial ? range->layer_count : info.resources.layers;
		for (uint32_t level = base_level; level < base_level + level_count; level++) {
			for (uint32_t layer = base_layer; layer < base_layer + layer_count; layer++) {
				const auto index = level * info.resources.layers + layer;
				EXIT_IF(index >= subresource_states.size());
				auto& subresource_state = subresource_states[index];
				if (joins(subresource_state)) {
					subresource_state.access_mask |= destination_access;
					subresource_state.pl_stage |= destination_stage;
					joined_access |= subresource_state.access_mask;
					joined_stage |= subresource_state.pl_stage;
					skipped++;
					continue;
				}

				constexpr auto write_access = vk::AccessFlagBits2::eTransferWrite |
				                              vk::AccessFlagBits2::eShaderWrite |
				                              vk::AccessFlagBits2::eMemoryWrite;
				const bool     repeated_write =
				    static_cast<bool>(subresource_state.access_mask & write_access);
				if (subresource_state.layout != destination_layout ||
				    subresource_state.access_mask != destination_access || repeated_write) {
					vk::ImageMemoryBarrier2 barrier {};
					barrier.srcStageMask                    = subresource_state.pl_stage;
					barrier.srcAccessMask                   = subresource_state.access_mask;
					barrier.dstStageMask                    = destination_stage;
					barrier.dstAccessMask                   = destination_access;
					barrier.oldLayout                       = subresource_state.layout;
					barrier.newLayout                       = destination_layout;
					barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
					barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
					barrier.image                           = backing.image;
					barrier.subresourceRange.aspectMask     = FullAspectMask(backing.format);
					barrier.subresourceRange.baseMipLevel   = level;
					barrier.subresourceRange.levelCount     = 1;
					barrier.subresourceRange.baseArrayLayer = layer;
					barrier.subresourceRange.layerCount     = 1;
					barriers.push_back(barrier);
					subresource_state = {destination_stage, destination_access, destination_layout};
				}
			}
		}

		if (!partial) {
			subresource_states.clear();
		}
	} else if (joins(state)) {
		state.access_mask |= destination_access;
		state.pl_stage |= destination_stage;
		RecordedCommands::Add(RecordedCommands::ImageBarriersSkipped);
		return {};
	} else {
		constexpr auto write_access   = vk::AccessFlagBits2::eTransferWrite |
		                                vk::AccessFlagBits2::eShaderWrite |
		                                vk::AccessFlagBits2::eMemoryWrite;
		const bool     repeated_write = static_cast<bool>(state.access_mask & write_access);
		if (state.layout == destination_layout && state.access_mask == destination_access &&
		    !repeated_write) {
			return {};
		}

		vk::ImageMemoryBarrier2 barrier {};
		barrier.srcStageMask                    = state.pl_stage;
		barrier.srcAccessMask                   = state.access_mask;
		barrier.dstStageMask                    = destination_stage;
		barrier.dstAccessMask                   = destination_access;
		barrier.oldLayout                       = state.layout;
		barrier.newLayout                       = destination_layout;
		barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
		barrier.image                           = backing.image;
		barrier.subresourceRange.aspectMask     = FullAspectMask(backing.format);
		barrier.subresourceRange.baseMipLevel   = 0;
		barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
		barriers.push_back(barrier);
	}

	if (state.layout != destination_layout) {
		NoteBindlessStateChange(*this);
	}
	// What joined (images:lean) stays for the next barrier to wait for.
	state = {destination_stage | joined_stage, destination_access | joined_access,
	         destination_layout};
	if (skipped != 0) {
		RecordedCommands::Add(RecordedCommands::ImageBarriersSkipped);
	}
	// A barrier for the whole image covers what an emulator-made use left; one for some of its
	// subresources may not.
	if (guest_use && !barriers.empty() && !has_subresource_states && !partial) {
		internal_use = false;
	}
	return barriers;
}

void Image::Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
                    std::optional<ImageSubresourceRange> range,
                    const CommandRecorder&               command_buffer) {
	const EmitStats::Scope stats(EmitStats::StepTransit); // gt7-emit-1119
	// Every use of the image goes through here (layouts), also when no barrier is needed: the
	// cache keeps a released image until the GPU has finished the last tick that used it.
	tick_accessed_last = std::max(tick_accessed_last, m_scheduler.CurrentTick());
	const auto transfer_access =
	    vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite;
	vk::PipelineStageFlags2 destination_stage {};
	if (static_cast<bool>(destination_access & transfer_access)) {
		destination_stage |= vk::PipelineStageFlagBits2::eTransfer;
	}
	if (!destination_access ||
	    static_cast<bool>(destination_access & ~vk::AccessFlags2 {transfer_access})) {
		destination_stage |=
		    vk::PipelineStageFlagBits2::eAllGraphics | vk::PipelineStageFlagBits2::eComputeShader;
	}
	const auto barriers =
	    GetBarriers(destination_layout, destination_access, destination_stage, range);
	if (barriers.empty()) {
		return;
	}
	EmitStats::Add(EmitStats::CountTransitBarriers);
	m_scheduler.EndRendering();
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
	dependency.pImageMemoryBarriers    = barriers.data();
	command_buffer.pipelineBarrier2(dependency);
}

void Image::Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
                   uint64_t size) {
	EXIT_IF(copies.empty() || buffer == nullptr || size == 0);
	m_scheduler.EndRendering();
	vk::BufferMemoryBarrier2 buffer_barrier {};
	buffer_barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.srcAccessMask       = vk::AccessFlagBits2::eMemoryWrite;
	buffer_barrier.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	buffer_barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferRead;
	buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.buffer              = buffer;
	buffer_barrier.offset              = offset;
	buffer_barrier.size                = size;
	const auto image_barriers =
	    GetBarriers(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
	                vk::PipelineStageFlagBits2::eCopy, {});
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &buffer_barrier;
	dependency.imageMemoryBarrierCount  = static_cast<uint32_t>(image_barriers.size());
	dependency.pImageMemoryBarriers     = image_barriers.data();
	m_scheduler.Current().MarkTransfer(CommandBufferDebugOp::ImageUpload, info.data.address, 0,
	                                   static_cast<uint32_t>(copies.size()),
	                                   static_cast<uint32_t>(std::min<uint64_t>(size, UINT32_MAX)),
	                                   static_cast<uint32_t>(backing.format));
	auto command                        = m_scheduler.Current().Recorder();
	command.pipelineBarrier2(dependency);
	command.copyBufferToImage(buffer, backing.image, vk::ImageLayout::eTransferDstOptimal,
	                          static_cast<uint32_t>(copies.size()), copies.data());
	buffer_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	buffer_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
	buffer_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.dstAccessMask =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	dependency.imageMemoryBarrierCount = 0;
	dependency.pImageMemoryBarriers    = nullptr;
	command.pipelineBarrier2(dependency);
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer,
                     uint64_t offset, uint64_t size) {
	EXIT_IF(copies.empty() || buffer == nullptr || size == 0);
	m_scheduler.EndRendering();
	vk::BufferMemoryBarrier2 buffer_barrier {};
	buffer_barrier.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.srcAccessMask =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	buffer_barrier.dstStageMask        = vk::PipelineStageFlagBits2::eCopy;
	buffer_barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
	buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.buffer              = buffer;
	buffer_barrier.offset              = offset;
	buffer_barrier.size                = size;
	const auto image_barriers =
	    GetBarriers(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
	                vk::PipelineStageFlagBits2::eCopy, {});
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &buffer_barrier;
	dependency.imageMemoryBarrierCount  = static_cast<uint32_t>(image_barriers.size());
	dependency.pImageMemoryBarriers     = image_barriers.data();
	m_scheduler.Current().MarkTransfer(CommandBufferDebugOp::ImageDownload, 0, info.data.address,
	                                   static_cast<uint32_t>(copies.size()),
	                                   static_cast<uint32_t>(std::min<uint64_t>(size, UINT32_MAX)),
	                                   static_cast<uint32_t>(backing.format));
	auto command                        = m_scheduler.Current().Recorder();
	command.pipelineBarrier2(dependency);
	command.copyImageToBuffer(backing.image, vk::ImageLayout::eTransferSrcOptimal, buffer,
	                          static_cast<uint32_t>(copies.size()), copies.data());
	buffer_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eCopy;
	buffer_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	buffer_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.dstAccessMask =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	dependency.imageMemoryBarrierCount = 0;
	dependency.pImageMemoryBarriers    = nullptr;
	command.pipelineBarrier2(dependency);
}

std::pair<uint32_t, uint32_t> Image::SanitizeCopyLayers(const Image& source,
                                                        const Image& destination, uint32_t depth) {
	const auto source_type        = source.backing.image_type;
	const auto destination_type   = destination.backing.image_type;
	uint32_t   source_layers      = source.backing.layers;
	uint32_t   destination_layers = destination.backing.layers;
	if (source_type == vk::ImageType::e3D) {
		source_layers = 1;
	}
	if (destination_type == vk::ImageType::e3D) {
		destination_layers = 1;
	}
	if (source_type == destination_type) {
		source_layers = destination_layers = std::min(source_layers, destination_layers);
	} else if (source_type == vk::ImageType::e2D && destination_type == vk::ImageType::e3D) {
		source_layers = depth;
	} else if (source_type == vk::ImageType::e3D && destination_type == vk::ImageType::e2D) {
		destination_layers = depth;
	}
	return {source_layers, destination_layers};
}

void Image::CopyImage(Image& source) {
	EXIT_IF(source.backing.samples != backing.samples);
	m_scheduler.EndRendering();
	const uint32_t levels     = std::min(source.backing.mip_levels, backing.mip_levels);
	const uint32_t base_depth = source.backing.image_type == backing.image_type
	                                ? std::min(source.backing.extent.depth, backing.extent.depth)
	                            : backing.image_type == vk::ImageType::e3D
	                                ? backing.extent.depth
	                                : source.backing.extent.depth;
	const auto     source_aspect =
	    FullAspectMask(source.backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	const auto destination_aspect =
	    FullAspectMask(backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	std::vector<vk::ImageCopy> copies;
	copies.reserve(levels);
	for (uint32_t level = 0; level < levels; level++) {
		// Within both images: an alias the guest describes smaller over the same memory (a
		// light-list grid whose size changes) took the larger one's extent, and the copy wrote
		// past its image; the device was lost ("WriteInvalid", Route X, 2026-10-14). Counted in
		// blocks: a direct copy may pair formats of the same block size and other block
		// extents (BC7 with RGBA32 UINT), and the extent is in the source's texels.
		const auto source_block      = vk::blockExtent(source.backing.format);
		const auto destination_block = vk::blockExtent(backing.format);
		const auto within = [&](uint32_t source_size, uint32_t destination_size, size_t axis) {
			const uint32_t source_texels      = std::max(source_size >> level, 1u);
			const uint32_t destination_texels = std::max(destination_size >> level, 1u);
			const uint32_t source_per         = std::max<uint32_t>(source_block[axis], 1u);
			const uint32_t destination_per    = std::max<uint32_t>(destination_block[axis], 1u);
			const uint32_t blocks =
			    std::min((source_texels + source_per - 1u) / source_per,
			             (destination_texels + destination_per - 1u) / destination_per);
			return std::min(source_texels, blocks * source_per);
		};
		const auto width  = within(source.backing.extent.width, backing.extent.width, 0);
		const auto height = within(source.backing.extent.height, backing.extent.height, 1);
		const auto depth  = std::max(base_depth >> level, 1u);
		const auto [source_layers, destination_layers] = SanitizeCopyLayers(source, *this, depth);
		vk::ImageCopy copy {};
		copy.srcSubresource = {source_aspect, level, 0, 1};
		copy.dstSubresource = {destination_aspect, level, 0, 1};
		if (source.backing.image_type == backing.image_type) {
			if (source.backing.image_type == vk::ImageType::e3D) {
				copy.extent = {width, height, depth};
			} else {
				copy.srcSubresource.layerCount = std::min(source_layers, destination_layers);
				copy.dstSubresource.layerCount = copy.srcSubresource.layerCount;
				copy.extent                    = {width, height, 1};
			}
		} else if (source.backing.image_type == vk::ImageType::e2D) {
			// Layers into depth slices: no more than either has.
			const auto slices              = std::min(source_layers, source.backing.layers);
			copy.srcSubresource.layerCount = slices;
			copy.extent                    = {width, height, slices};
		} else {
			const auto slices              = std::min(destination_layers, backing.layers);
			copy.dstSubresource.layerCount = slices;
			copy.extent                    = {width, height, slices};
		}
		copies.push_back(copy);
	}
	if (copies.empty()) {
		return;
	}
	m_scheduler.Current().MarkTransfer(
	    CommandBufferDebugOp::ImageCopy, info.data.address, source.info.data.address,
	    static_cast<uint32_t>(copies.size()), static_cast<uint32_t>(source.backing.format),
	    static_cast<uint32_t>(backing.format));
	auto command = m_scheduler.Current().Recorder();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
	                  vk::ImageLayout::eTransferDstOptimal, static_cast<uint32_t>(copies.size()),
	                  copies.data());
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::Resolve(Image& source, const ImageSubresourceRange& source_range,
                    const ImageSubresourceRange& destination_range) {
	EXIT_IF(backing.samples != 1 || source.backing.image_type != vk::ImageType::e2D ||
	        backing.image_type != vk::ImageType::e2D || source_range.level_count != 1 ||
	        destination_range.level_count != 1 ||
	        source_range.base_level >= source.backing.mip_levels ||
	        destination_range.base_level >= backing.mip_levels ||
	        source_range.base_layer >= source.backing.layers ||
	        destination_range.base_layer >= backing.layers);
	const auto layers       = std::min({source_range.layer_count, destination_range.layer_count,
	                                    source.backing.layers - source_range.base_layer,
	                                    backing.layers - destination_range.base_layer});
	const auto source_width = std::max(source.backing.extent.width >> source_range.base_level, 1u);
	const auto source_height =
	    std::max(source.backing.extent.height >> source_range.base_level, 1u);
	const auto destination_width =
	    std::max(backing.extent.width >> destination_range.base_level, 1u);
	const auto destination_height =
	    std::max(backing.extent.height >> destination_range.base_level, 1u);
	const bool copy = source.backing.samples == 1;
	EXIT_IF(layers == 0 || info.extent.width > source_width || info.extent.height > source_height ||
	        info.extent.width > destination_width || info.extent.height > destination_height ||
	        (copy ? !ImageViewOps::FormatsCompatible(source.backing.format, backing.format)
	              : source.backing.format != backing.format));
	auto resolved_source_range             = source_range;
	auto resolved_destination_range        = destination_range;
	resolved_source_range.layer_count      = layers;
	resolved_destination_range.layer_count = layers;
	const vk::Extent3D resolve_extent {info.extent.width, info.extent.height, 1};

	m_scheduler.EndRendering();
	m_scheduler.Current().MarkTransfer(CommandBufferDebugOp::ImageResolve, info.data.address,
	                                   source.info.data.address, source.backing.samples, layers,
	                                   static_cast<uint32_t>(backing.format));
	auto command = m_scheduler.Current().Recorder();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
	               resolved_source_range, command);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
	        resolved_destination_range, command);
	if (copy) {
		vk::ImageCopy region {};
		region.srcSubresource = {vk::ImageAspectFlagBits::eColor, resolved_source_range.base_level,
		                         resolved_source_range.base_layer, layers};
		region.dstSubresource = {vk::ImageAspectFlagBits::eColor,
		                         resolved_destination_range.base_level,
		                         resolved_destination_range.base_layer, layers};
		region.extent         = resolve_extent;
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
		                  vk::ImageLayout::eTransferDstOptimal, region);
	} else {
		vk::ImageResolve region {};
		region.srcSubresource = {vk::ImageAspectFlagBits::eColor, resolved_source_range.base_level,
		                         resolved_source_range.base_layer, layers};
		region.dstSubresource = {vk::ImageAspectFlagBits::eColor,
		                         resolved_destination_range.base_level,
		                         resolved_destination_range.base_layer, layers};
		region.extent         = resolve_extent;
		command.resolveImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                     backing.image, vk::ImageLayout::eTransferDstOptimal, region);
	}
}

uint32_t Image::CopyRows(uint64_t row_size, uint32_t rows, uint64_t capacity) noexcept {
	if (row_size == 0 || rows == 0 || row_size > capacity) {
		return 0;
	}
	return static_cast<uint32_t>(std::min<uint64_t>(rows, capacity / row_size));
}

void Image::CopyImageWithBuffer(Image& source, Buffer& buffer, TileManager& tiler) {
	EXIT_IF(buffer.Handle() == nullptr || source.backing.samples != 1 || backing.samples != 1);
	m_scheduler.EndRendering();
	const uint32_t levels = std::min(source.backing.mip_levels, backing.mip_levels);
	const auto     source_aspect =
	    FullAspectMask(source.backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	const auto destination_aspect =
	    FullAspectMask(backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	const auto     source_bytes      = DepthAspectTransferBytes(source.backing.format) != 0
	                                       ? DepthAspectTransferBytes(source.backing.format)
	                                       : source.info.bytes_per_block;
	const auto     destination_bytes = DepthAspectTransferBytes(backing.format) != 0
	                                       ? DepthAspectTransferBytes(backing.format)
	                                       : info.bytes_per_block;
	const uint32_t source_block      = source.info.IsBlock() ? 4u : 1u;
	const uint32_t destination_block = info.IsBlock() ? 4u : 1u;
	EXIT_IF(levels == 0 || source_bytes == 0 || source_bytes != destination_bytes ||
	        source_block != destination_block);
	const auto source_transform = source.info.GetColorTransform();
	const auto target_transform = info.GetColorTransform();

	vk::BufferMemoryBarrier2 barrier {};
	barrier.srcStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	barrier.srcAccessMask       = vk::AccessFlagBits2::eTransferRead;
	barrier.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = buffer.Handle();
	barrier.offset              = 0;
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &barrier;
	m_scheduler.Current().MarkTransfer(CommandBufferDebugOp::ImageCopyThroughBuffer,
	                                   info.data.address, source.info.data.address, levels,
	                                   static_cast<uint32_t>(source.backing.format),
	                                   static_cast<uint32_t>(backing.format));
	auto command                        = m_scheduler.Current().Recorder();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	for (uint32_t level = 0; level < levels; level++) {
		// Within both images, as CopyImage: the source's extent alone wrote past a smaller
		// destination. A block-compressed source is cut on a block boundary.
		const auto fit = [&](uint32_t source_size, uint32_t destination_size) {
			const uint32_t source_texels      = std::max(source_size >> level, 1u);
			const uint32_t destination_texels = std::max(destination_size >> level, 1u);
			const uint32_t texels             = std::min(source_texels, destination_texels);
			// A partial block only where it ends both images.
			return source_texels == destination_texels ? texels
			                                           : texels / source_block * source_block;
		};
		const auto width  = fit(source.backing.extent.width, backing.extent.width);
		const auto height = fit(source.backing.extent.height, backing.extent.height);
		if (width == 0 || height == 0) {
			continue;
		}
		const auto source_depth      = source.backing.image_type == vk::ImageType::e3D
		                                   ? std::max(source.backing.extent.depth >> level, 1u)
		                                   : source.backing.layers;
		const auto destination_depth = backing.image_type == vk::ImageType::e3D
		                                   ? std::max(backing.extent.depth >> level, 1u)
		                                   : backing.layers;
		const auto slices            = std::min(source_depth, destination_depth);
		const auto block_rows        = (height + source_block - 1) / source_block;
		const auto row_size =
		    static_cast<uint64_t>((width + source_block - 1) / source_block) * source_bytes;
		const auto rows_per_copy = CopyRows(row_size, block_rows, buffer.Size());
		EXIT_IF(slices == 0 || rows_per_copy == 0);
		for (uint32_t slice = 0; slice < slices; slice++) {
			for (uint32_t block_row = 0; block_row < block_rows; block_row += rows_per_copy) {
				const auto          copy_rows   = std::min(rows_per_copy, block_rows - block_row);
				const auto          y           = block_row * source_block;
				const auto          copy_height = std::min(copy_rows * source_block, height - y);
				const auto          copy_size   = row_size * copy_rows;
				vk::BufferImageCopy source_copy {};
				source_copy.imageSubresource = {
				    source_aspect, level,
				    source.backing.image_type == vk::ImageType::e3D ? 0u : slice, 1};
				source_copy.imageOffset           = {0, static_cast<int32_t>(y),
				                                     source.backing.image_type == vk::ImageType::e3D
				                                         ? static_cast<int32_t>(slice)
				                                         : 0};
				source_copy.imageExtent           = {width, copy_height, 1};
				auto destination_copy             = source_copy;
				destination_copy.imageSubresource = {
				    destination_aspect, level,
				    backing.image_type == vk::ImageType::e3D ? 0u : slice, 1};
				destination_copy.imageOffset.z =
				    backing.image_type == vk::ImageType::e3D ? static_cast<int32_t>(slice) : 0;
				barrier.size          = copy_size;
				barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
				barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
				command.pipelineBarrier2(dependency);
				command.copyImageToBuffer(source.backing.image,
				                          vk::ImageLayout::eTransferSrcOptimal, buffer.Handle(),
				                          source_copy);
				if (source_transform != target_transform) {
					const TileManager::Result bytes {buffer.Handle(), 0, copy_size};
					if (source_transform != ColorTransform::None) {
						tiler.TransformColor(bytes, bytes, source_transform, false);
					}
					if (target_transform != ColorTransform::None) {
						tiler.TransformColor(bytes, bytes, target_transform, true);
					}
				} else {
					barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
					barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
					command.pipelineBarrier2(dependency);
				}
				command.copyBufferToImage(buffer.Handle(), backing.image,
				                          vk::ImageLayout::eTransferDstOptimal, destination_copy);
			}
		}
	}
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::CopyMip(Image& source, uint32_t mip, uint32_t layer) {
	EXIT_IF(source.backing.samples != backing.samples || mip >= backing.mip_levels ||
	        layer >= backing.layers);
	m_scheduler.EndRendering();
	const auto width  = std::max(backing.extent.width >> mip, 1u);
	const auto height = std::max(backing.extent.height >> mip, 1u);
	const auto depth  = std::max(backing.extent.depth >> mip, 1u);
	EXIT_IF(width != source.backing.extent.width || height != source.backing.extent.height);
	auto [source_layers, destination_layers] = SanitizeCopyLayers(source, *this, depth);
	// From `layer` on, the destination has only so many layers left.
	if (backing.image_type != vk::ImageType::e3D &&
	    source.backing.image_type != vk::ImageType::e3D) {
		source_layers = destination_layers = std::min(source_layers, backing.layers - layer);
	}
	const auto aspects = FullAspectMask(source.backing.format);
	EXIT_IF(aspects != FullAspectMask(backing.format));
	std::array<vk::ImageCopy, 2> copies {};
	uint32_t                     copy_count = 0;
	for (const auto aspect: {vk::ImageAspectFlagBits::eColor, vk::ImageAspectFlagBits::eDepth,
	                         vk::ImageAspectFlagBits::eStencil}) {
		if (!static_cast<bool>(aspects & aspect)) {
			continue;
		}
		auto& copy          = copies[copy_count++];
		copy.srcSubresource = {aspect, 0, 0, source_layers};
		copy.dstSubresource = {aspect, mip, layer, destination_layers};
		copy.extent         = {width, height, depth};
	}
	m_scheduler.Current().MarkTransfer(CommandBufferDebugOp::ImageMipCopy, info.data.address,
	                                   source.info.data.address, mip, layer,
	                                   static_cast<uint32_t>(backing.format));
	auto command = m_scheduler.Current().Recorder();
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy_count, copies.data());
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

namespace ImageOps {

void Validate(const ImageInfo& info) {
	ValidateOptionalRange(info.data, "data");
	ValidateOptionalRange(info.stencil, "stencil");

	if (info.pixel_format == vk::Format::eUndefined) {
		const bool metadata_empty =
		    info.metadata.range.Empty() && info.metadata.kind == ImageMetadataKind::None &&
		    info.metadata.control == 0 &&
		    info.metadata.compression == VideoOutCompression::Uncompressed &&
		    !info.metadata.stencil_compressed;
		if (info.data.Empty() || info.HasStencil() || !metadata_empty || info.extent.width == 0 ||
		    info.extent.height == 0 || info.extent.depth == 0 || info.resources.levels != 1 ||
		    info.resources.layers != 1 || info.samples != 1 || info.pitch != 0 ||
		    info.bytes_per_block != 0) {
			EXIT("invalid stencil association image\n");
		}
		return;
	}

	if (info.extent.width == 0 || info.extent.height == 0 || info.extent.depth == 0 ||
	    info.resources.levels == 0 || info.resources.levels > info.mip_layout.size() ||
	    info.resources.layers == 0 || info.samples == 0 ||
	    vulkan_sample_count(info.samples) == vk::SampleCountFlagBits {} ||
	    info.bytes_per_block == 0 || (info.data.address != 0 && info.pitch == 0)) {
		EXIT("invalid image geometry or format\n");
	}

	switch (info.type) {
		case Prospero::ImageType::kColor1D:
			if (info.extent.height != 1 || info.extent.depth != 1) {
				EXIT("invalid 1D image shape\n");
			}
			break;
		case Prospero::ImageType::kColor3D:
			if (info.resources.layers != 1) {
				EXIT("3D images cannot have array layers\n");
			}
			break;
		case Prospero::ImageType::kColor2D:
			if (info.extent.depth != 1) {
				EXIT("invalid 2D image shape\n");
			}
			break;
		default: EXIT("non-base image type: %u\n", static_cast<uint32_t>(info.type));
	}
	if (info.samples > 1 && info.resources.levels != 1) {
		EXIT("multisampled images cannot have mip levels\n");
	}

	if (info.metadata.stencil_compressed && !info.HasStencil()) {
		EXIT("compressed stencil metadata requires a stencil plane\n");
	}
	switch (info.metadata.kind) {
		case ImageMetadataKind::None:
			if (!info.metadata.range.Empty() || info.metadata.control != 0 ||
			    info.metadata.compression != VideoOutCompression::Uncompressed ||
			    info.metadata.stencil_compressed) {
				EXIT("metadata-free image has metadata state\n");
			}
			break;
		case ImageMetadataKind::Htile:
			if (!info.metadata.range.Valid() ||
			    info.metadata.compression != VideoOutCompression::Uncompressed) {
				EXIT("invalid HTILE metadata\n");
			}
			break;
		case ImageMetadataKind::Dcc:
		case ImageMetadataKind::Cmask:
			if (!GuestRange {info.metadata.range.address,
			                 std::max<uint64_t>(info.metadata.range.size, 1)}.Valid() ||
			    info.metadata.compression == VideoOutCompression::Unsupported) {
				EXIT("invalid color metadata\n");
			}
			break;
	}
}

Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element) {
	switch (bytes_per_element) {
		case 1: return Prospero::BufferFormat::k8UNorm;
		case 2: return Prospero::BufferFormat::k16UNorm;
		case 4: return Prospero::BufferFormat::k32Float;
		case 8: return Prospero::BufferFormat::k16_16_16_16Float;
		case 16: return Prospero::BufferFormat::k32_32_32_32Float;
		default: EXIT("unsupported render-target element size: %u\n", bytes_per_element);
	}
}

} // namespace ImageOps

// Recycles the Vulkan images (and their views) of destroyed cache images for new ones of the same
// shape. GT7 binds hundreds of short-lived images a frame (per-frame CPU-written tables at new
// addresses, a light grid rebound with 8, 16, ... 48 layers), and the texture cache creates and
// destroys each: vmaCreateImage and vmaDestroyImage took 7.7 of the GPU thread's 21 s at Tokyo
// Expressway (70,000 of each, ~110 us a pair). A backing reaches the pool only once its image's
// last tick is done (TextureCache::ReleaseImage), so the GPU no longer uses it; its contents are
// undefined again, as a new image's are. Bounded by KYTY_IMAGE_POOL_MB (default 256, 0: off),
// images of at most 16 MB, kept for at most 4 s.
class ImageBackingPool {
public:
	struct Key {
		vk::ImageCreateFlags flags;
		vk::ImageType        type = vk::ImageType::e2D;
		vk::Extent3D         extent;
		uint32_t             levels  = 1;
		uint32_t             layers  = 1;
		vk::Format           format  = vk::Format::eUndefined;
		vk::ImageUsageFlags  usage;
		uint32_t             samples = 1;
		bool operator==(const Key& other) const {
			return flags == other.flags && type == other.type && extent == other.extent &&
			       levels == other.levels && layers == other.layers && format == other.format &&
			       usage == other.usage && samples == other.samples;
		}
	};
	struct KeyHash {
		size_t operator()(const Key& key) const {
			uint64_t words[6] = {
			    static_cast<uint64_t>(static_cast<VkImageCreateFlags>(key.flags)) |
			        (static_cast<uint64_t>(key.type) << 32u),
			    key.extent.width | (static_cast<uint64_t>(key.extent.height) << 32u),
			    key.extent.depth | (static_cast<uint64_t>(key.levels) << 32u),
			    key.layers | (static_cast<uint64_t>(key.format) << 32u),
			    static_cast<uint64_t>(static_cast<VkImageUsageFlags>(key.usage)),
			    key.samples};
			return static_cast<size_t>(XXH3_64bits(words, sizeof(words)));
		}
	};

	static ImageBackingPool& Get() {
		// Never destroyed: images may still be released during static destruction.
		static auto* pool = new ImageBackingPool;
		return *pool;
	}

	static uint64_t Budget() {
		static const uint64_t budget = [] {
			const char* value = std::getenv("KYTY_IMAGE_POOL_MB");
			return (value != nullptr ? std::strtoull(value, nullptr, 10) : 256ull) << 20u;
		}();
		return budget;
	}

	static Key KeyOf(const vk::ImageCreateInfo& create) {
		return {create.flags,       create.imageType, create.extent,
		        create.mipLevels,   create.arrayLayers, create.format,
		        create.usage,       static_cast<uint32_t>(create.samples)};
	}
	static Key KeyOf(const VulkanImage& backing) {
		return {backing.flags,      backing.image_type, backing.extent, backing.mip_levels,
		        backing.layers,     backing.format,     backing.usage,  backing.samples};
	}

	// A pooled backing of this shape, with its views, or false.
	bool Take(GraphicContext& graphics, const vk::ImageCreateInfo& create, VulkanImage& backing,
	          std::vector<CachedImageView>& views) {
		if (Budget() == 0) {
			return false;
		}
		const auto      key = KeyOf(create);
		std::lock_guard lock(m_mutex);
		Trim(graphics, Now());
		const auto found = m_index.find(key);
		if (found == m_index.end() || found->second.empty()) {
			m_misses++;
			return false;
		}
		const auto entry = found->second.back();
		found->second.pop_back();
		backing.format     = create.format;
		backing.image_type = create.imageType;
		backing.extent     = create.extent;
		backing.layers     = create.arrayLayers;
		backing.mip_levels = create.mipLevels;
		backing.samples    = static_cast<uint32_t>(create.samples);
		backing.usage      = create.usage;
		backing.flags      = create.flags;
		backing.image      = entry->image;
		backing.allocation = entry->allocation;
		backing.state      = {};
		backing.subresource_states.clear();
		views = std::move(entry->views);
		m_bytes -= entry->bytes;
		m_entries.erase(entry);
		Report();
		return true;
	}

	// Takes the backing (and views) of a destroyed image, or leaves them to be destroyed.
	bool Put(GraphicContext& graphics, VulkanImage& backing, std::vector<CachedImageView>& views) {
		if (Budget() == 0 || backing.image == nullptr || backing.allocation == nullptr) {
			return false;
		}
		VmaAllocationInfo allocation {};
		vmaGetAllocationInfo(graphics.allocator, backing.allocation, &allocation);
		VkMemoryPropertyFlags properties = 0;
		vmaGetAllocationMemoryProperties(graphics.allocator, backing.allocation, &properties);
		constexpr uint64_t MaxImageBytes = 16ull << 20u;
		if (allocation.size == 0 || allocation.size > MaxImageBytes ||
		    (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) {
			return false; // a large or spilled image: freed, as before
		}
		std::lock_guard lock(m_mutex);
		const auto      now = Now();
		m_entries.push_back({KeyOf(backing), backing.image, backing.allocation, std::move(views),
		                     allocation.size, now});
		m_index[m_entries.back().key].push_back(std::prev(m_entries.end()));
		m_bytes += allocation.size;
		m_puts++;
		backing.image      = nullptr;
		backing.allocation = nullptr;
		views.clear();
		Trim(graphics, now);
		return true;
	}

private:
	struct Entry {
		Key                          key;
		vk::Image                    image      = nullptr;
		VmaAllocation                allocation = nullptr;
		std::vector<CachedImageView> views;
		uint64_t                     bytes = 0;
		std::chrono::steady_clock::time_point stamp;
	};
	using Entries = std::list<Entry>;

	static std::chrono::steady_clock::time_point Now() { return std::chrono::steady_clock::now(); }

	// Oldest first, while over budget or older than the keep time.
	void Trim(GraphicContext& graphics, std::chrono::steady_clock::time_point now) {
		constexpr auto Keep = std::chrono::seconds(4);
		while (!m_entries.empty() &&
		       (m_bytes > Budget() || now - m_entries.front().stamp > Keep)) {
			auto  oldest = m_entries.begin();
			auto& list   = m_index[oldest->key];
			std::erase(list, oldest);
			if (list.empty()) {
				m_index.erase(oldest->key);
			}
			for (const auto& cached: oldest->views) {
				if (cached.view != nullptr) {
					NoteImageViewDestroyed(cached.view, 0, oldest->key.format);
					graphics.device.destroyImageView(cached.view, nullptr);
				}
			}
			VulkanImage doomed;
			doomed.image      = oldest->image;
			doomed.allocation = oldest->allocation;
			graphics.DeleteImage(doomed);
			m_bytes -= oldest->bytes;
			m_entries.erase(oldest);
			m_destroyed++;
		}
	}

	void Report() {
		if (++m_hits % 4096 == 1) {
			LOGF("Image pool: %" PRIu64 " images reused, %" PRIu64 " created anew, %" PRIu64
			     " destroyed; %zu kept (%" PRIu64 " MB)\n",
			     m_hits, m_misses, m_destroyed, m_entries.size(), m_bytes >> 20u);
		}
	}

	std::mutex                                                     m_mutex;
	Entries                                                        m_entries; // oldest first
	std::unordered_map<Key, std::vector<Entries::iterator>, KeyHash> m_index;
	uint64_t                                                       m_bytes     = 0;
	uint64_t                                                       m_hits      = 0;
	uint64_t                                                       m_misses    = 0;
	uint64_t                                                       m_puts      = 0;
	uint64_t                                                       m_destroyed = 0;
};

Image::Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& image_info)
    : info(image_info),
      stencil_subresources {0, image_info.resources.levels, 0, image_info.resources.layers},
      m_graphics(graphics), m_scheduler(scheduler) {
	KYTY_PROFILER_FUNCTION();
	ImageOps::Validate(info);
	m_cpu_dirty =
	    !info.data.Empty() && info.metadata.compression == VideoOutCompression::Uncompressed;
	if (info.pixel_format == vk::Format::eUndefined) {
		return;
	}

	vk::ImageCreateInfo create {};
	create.usage         = ImageUsageFlags(graphics, info);
	create.flags         = ImageCreateFlags(graphics, info, create.usage);
	create.imageType     = HostImageType(info.type);
	create.extent        = info.extent;
	create.mipLevels     = info.resources.levels;
	create.arrayLayers   = info.IsVolume() ? 1u : info.resources.layers;
	create.format        = info.pixel_format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.samples       = vulkan_sample_count(info.samples);

	if (ImageBackingPool::Get().Take(graphics, create, backing, views)) {
		return;
	}
	vk::ImageFormatProperties properties {};
	if (graphics.GetImageFormatProperties(create.format, create.imageType, create.tiling,
	                                      create.usage, create.flags,
	                                      &properties) != vk::Result::eSuccess ||
	    !static_cast<bool>(properties.sampleCounts & create.samples)) {
		EXIT("image format does not support required usage: format=%d type=%d usage=0x%x "
		     "flags=0x%x samples=%u\n",
		     static_cast<int>(create.format), static_cast<int>(create.imageType),
		     static_cast<vk::ImageUsageFlags::MaskType>(create.usage),
		     static_cast<vk::ImageCreateFlags::MaskType>(create.flags), info.samples);
	}

	if (!graphics.CreateImage(create, backing)) {
		EXIT("Out of memory: an image fits neither in video memory nor in system memory (the VMA "
		     "heap lines above give usage and budget). Close other programs; on a card with little "
		     "video memory lower the resolution; if Windows' commit limit is reached, enlarge the "
		     "page file. Image: extent=%ux%ux%u format=%d layers=%u levels=%u\n",
		     create.extent.width, create.extent.height, create.extent.depth,
		     static_cast<int>(create.format), create.arrayLayers, create.mipLevels);
	}
	SetVulkanObjectNameF(
	    graphics.device, backing.image,
	    "Kyty.Image[guest=0x{:016x} size=0x{:x} extent={}x{}x{} format={} mips={} layers={} samples={}]",
	    info.data.address, info.data.size, info.extent.width, info.extent.height, info.extent.depth,
	    static_cast<uint32_t>(info.pixel_format), info.resources.levels, info.resources.layers,
	    info.samples);
}

uint64_t Image::HashGuestEdges() const {
	std::array<uint8_t, TRACKER_PAGE_SIZE * 2> bytes {};
	const auto                                 range = info.data;
	const uint64_t head_end =
	    std::min(range.End(), Common::AlignUp(range.address, TRACKER_PAGE_SIZE));
	const uint64_t tail_begin =
	    std::max(range.address, Common::AlignDown(range.End(), TRACKER_PAGE_SIZE));
	const uint64_t head_size    = head_end - range.address;
	const uint64_t tail_address = tail_begin < head_end ? head_end : tail_begin;
	const uint64_t tail_size    = range.End() - tail_address;
	if ((head_size != 0 &&
	     !LibKernel::Memory::TryReadBacking(range.address, bytes.data(), head_size)) ||
	    (tail_size != 0 &&
	     !LibKernel::Memory::TryReadBacking(tail_address, bytes.data() + head_size, tail_size))) {
		EXIT("Image: failed to hash guest backing\n");
	}
	return XXH3_64bits(bytes.data(), static_cast<size_t>(head_size + tail_size));
}

Image::~Image() {
	KYTY_PROFILER_FUNCTION();
	if (backing.image != nullptr && ImageBackingPool::Get().Put(m_graphics, backing, views)) {
		return;
	}
	for (const auto& cached: views) {
		if (cached.view != nullptr) {
			NoteImageViewDestroyed(cached.view, info.data.address, info.pixel_format);
			m_graphics.device.destroyImageView(cached.view, nullptr);
		}
	}
	if (backing.image != nullptr) {
		m_graphics.DeleteImage(backing);
	}
}

} // namespace Libs::Graphics
