#include "local/screenshot.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace Libs::Graphics::Screenshot {

namespace {

std::mutex  g_mutex;
std::string g_requested; // (g_mutex)

// A frame series (RequestSeries): its frames are taken under g_mutex, written on the present thread.
constexpr const char* SeriesMark = "\x01series";
std::string g_series_path;     // (g_mutex)
uint32_t    g_series_left = 0; // frames still to take (g_mutex)
uint32_t    g_series_total = 0;
uint32_t    g_series_written = 0; // (present thread)
FILE*       g_series_file    = nullptr; // (present thread)

float HalfToFloat(uint16_t half) {
	const uint32_t sign     = (half >> 15) & 1u;
	const uint32_t exponent = (half >> 10) & 0x1fu;
	const uint32_t mantissa = half & 0x3ffu;
	float          value    = 0.0f;
	if (exponent == 0) value = std::ldexp(static_cast<float>(mantissa), -24);
	else if (exponent == 31) value = mantissa == 0 ? 65504.0f : 0.0f;
	else value = std::ldexp(static_cast<float>(mantissa | 0x400u), static_cast<int>(exponent) - 25);
	return sign != 0 ? -value : value;
}

uint8_t ToByte(float value) {
	return static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

// The texel's red, green and blue as bytes; false for a format not handled here.
bool Convert(vk::Format format, const uint8_t* texel, uint8_t* rgb) {
	switch (format) {
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb: rgb[0] = texel[2], rgb[1] = texel[1], rgb[2] = texel[0]; return true;
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb: rgb[0] = texel[0], rgb[1] = texel[1], rgb[2] = texel[2]; return true;
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32: {
			uint32_t word = 0;
			std::memcpy(&word, texel, sizeof(word));
			const uint32_t low = word & 0x3ffu, mid = (word >> 10) & 0x3ffu, high = (word >> 20) & 0x3ffu;
			const bool     bgr = format == vk::Format::eA2R10G10B10UnormPack32;
			rgb[0]             = static_cast<uint8_t>((bgr ? high : low) >> 2);
			rgb[1]             = static_cast<uint8_t>(mid >> 2);
			rgb[2]             = static_cast<uint8_t>((bgr ? low : high) >> 2);
			return true;
		}
		case vk::Format::eR16G16B16A16Sfloat: {
			uint16_t halves[3] {};
			std::memcpy(halves, texel, sizeof(halves));
			for (int i = 0; i < 3; ++i) rgb[i] = ToByte(HalfToFloat(halves[i]));
			return true;
		}
		default: return false;
	}
}

uint32_t TexelSize(vk::Format format) {
	return format == vk::Format::eR16G16B16A16Sfloat ? 8u : 4u;
}

} // namespace

void Request(const std::string& path) {
	std::lock_guard lock(g_mutex);
	g_requested = path;
}

void RequestSeries(const std::string& path, uint32_t count) {
	std::lock_guard lock(g_mutex);
	g_series_path  = path;
	g_series_left  = count;
	g_series_total = count;
}

std::string Take() {
	std::lock_guard lock(g_mutex);
	if (!g_requested.empty()) return std::exchange(g_requested, {});
	if (g_series_left == 0) return {};
	--g_series_left;
	return SeriesMark;
}

bool IsSeriesFrame(const std::string& path) {
	return path == SeriesMark;
}

namespace {

// One frame of the series: box averages (every few texels of each box) appended to the series file.
void AppendSeries(const Pending& pending, const uint8_t* texels) {
	const auto     size  = TexelSize(pending.format);
	const uint32_t box   = std::max(1u, pending.extent.width / SeriesWidth);
	const uint32_t w     = pending.extent.width / box, h = pending.extent.height / box;
	const uint32_t step  = std::max(1u, box / 4);
	std::string    path;
	uint32_t       total = 0, left = 0;
	{
		std::lock_guard lock(g_mutex);
		path = g_series_path, total = g_series_total, left = g_series_left;
	}
	if (g_series_file == nullptr) {
		g_series_written = 0;
		g_series_file    = std::fopen(path.c_str(), "wb");
		if (g_series_file == nullptr) {
			std::printf("SCREENSHOTS %s failed: cannot write the file\n", path.c_str());
			std::fflush(stdout);
			std::lock_guard lock(g_mutex);
			g_series_left = 0;
			return;
		}
		std::fprintf(g_series_file, "KSER %u %u %u\n", w, h, total);
	}
	std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
	bool                 converted = texels != nullptr;
	for (uint32_t y = 0; converted && y < h; ++y) {
		for (uint32_t x = 0; converted && x < w; ++x) {
			uint32_t sum[3] {}, count = 0;
			for (uint32_t dy = 0; dy < box; dy += step) {
				for (uint32_t dx = 0; dx < box; dx += step) {
					const auto index = static_cast<size_t>(y * box + dy) * pending.extent.width + (x * box + dx);
					uint8_t    texel[3] {};
					converted = converted && Convert(pending.format, texels + index * size, texel);
					sum[0] += texel[0], sum[1] += texel[1], sum[2] += texel[2], ++count;
				}
			}
			for (int c = 0; c < 3; ++c) rgb[(static_cast<size_t>(y) * w + x) * 3 + c] = static_cast<uint8_t>(sum[c] / count);
		}
	}
	if (converted) {
		std::fwrite(rgb.data(), 1, rgb.size(), g_series_file);
		++g_series_written;
	}
	if (left == 0) {
		std::fclose(g_series_file);
		g_series_file = nullptr;
		std::printf("SCREENSHOTS %s %u\n", path.c_str(), g_series_written);
		std::fflush(stdout);
	}
}

} // namespace

Pending Record(GraphicContext& graphics, vk::CommandBuffer command, VulkanImage& image, const std::string& path) {
	Pending pending;
	if (path.empty() || image.image == nullptr) return pending;
	pending.format = image.format;
	pending.extent = {image.extent.width, image.extent.height};
	pending.path   = path;
	VkBufferCreateInfo create {};
	create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	create.size  = static_cast<VkDeviceSize>(pending.extent.width) * pending.extent.height * TexelSize(image.format);
	create.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	allocate.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	if (vmaCreateBuffer(graphics.allocator, &create, &allocate, &pending.buffer, &pending.memory, nullptr) != VK_SUCCESS) {
		std::printf("SCREENSHOT %s failed: no buffer\n", path.c_str());
		std::fflush(stdout);
		pending = {};
		return pending;
	}
	// To a copy source and back to the image's layout, its tracked state kept.
	const auto barrier = [&](vk::ImageLayout from, vk::ImageLayout to, vk::PipelineStageFlags2 src_stage,
	                         vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
		vk::ImageMemoryBarrier2 b {};
		b.srcStageMask        = src_stage;
		b.srcAccessMask       = src_access;
		b.dstStageMask        = dst_stage;
		b.dstAccessMask       = dst_access;
		b.oldLayout           = from;
		b.newLayout           = to;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image               = image.image;
		b.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers    = &b;
		command.pipelineBarrier2(dependency);
	};
	const auto state = image.state;
	barrier(state.layout, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eAllCommands,
	        vk::AccessFlagBits2::eMemoryWrite, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
	vk::BufferImageCopy region {};
	region.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	region.imageExtent      = vk::Extent3D {pending.extent.width, pending.extent.height, 1};
	command.copyImageToBuffer(image.image, vk::ImageLayout::eTransferSrcOptimal, vk::Buffer(pending.buffer), 1, &region);
	if (state.layout != vk::ImageLayout::eUndefined) {
		barrier(vk::ImageLayout::eTransferSrcOptimal, state.layout, vk::PipelineStageFlagBits2::eTransfer,
		        vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eAllCommands,
		        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	} else {
		image.state = {vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
		               vk::ImageLayout::eTransferSrcOptimal};
	}
	return pending;
}

void Finish(GraphicContext& graphics, Pending& pending) {
	if (pending.buffer == VK_NULL_HANDLE) return;
	VmaAllocationInfo info {};
	vmaGetAllocationInfo(graphics.allocator, pending.memory, &info);
	(void)vmaInvalidateAllocation(graphics.allocator, pending.memory, 0, VK_WHOLE_SIZE);
	const auto* texels = static_cast<const uint8_t*>(info.pMappedData);
	if (pending.path == SeriesMark) {
		AppendSeries(pending, texels);
		vmaDestroyBuffer(graphics.allocator, pending.buffer, pending.memory);
		pending = {};
		return;
	}
	const auto  size   = TexelSize(pending.format);
	std::vector<uint8_t> rgb(static_cast<size_t>(pending.extent.width) * pending.extent.height * 3);
	bool                 converted = texels != nullptr;
	for (size_t i = 0; converted && i < static_cast<size_t>(pending.extent.width) * pending.extent.height; ++i)
		converted = Convert(pending.format, texels + i * size, rgb.data() + i * 3);
	if (converted) {
		if (auto* file = std::fopen(pending.path.c_str(), "wb"); file != nullptr) {
			std::fprintf(file, "P6\n%u %u\n255\n", pending.extent.width, pending.extent.height);
			std::fwrite(rgb.data(), 1, rgb.size(), file);
			std::fclose(file);
			std::printf("SCREENSHOT %s %ux%u\n", pending.path.c_str(), pending.extent.width, pending.extent.height);
		} else {
			std::printf("SCREENSHOT %s failed: cannot write the file\n", pending.path.c_str());
		}
	} else {
		std::printf("SCREENSHOT %s failed: format %s not handled\n", pending.path.c_str(),
		            vk::to_string(pending.format).c_str());
	}
	std::fflush(stdout);
	vmaDestroyBuffer(graphics.allocator, pending.buffer, pending.memory);
	pending = {};
}

} // namespace Libs::Graphics::Screenshot
