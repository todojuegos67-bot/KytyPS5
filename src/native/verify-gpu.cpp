// (f5b) The verification's GPU side (pass-verify.cpp has the CPU side; verify-jobs.h the hand-over).
//
// A job is the scope begin or end of a verified pass instance and the targets to copy. Recorded in the walk's command
// buffer at that point of the stream (the packet after the scope mark: nothing of the pass's queue ran in between):
//   - the rendering in progress is ended (as any transition does), each target subresource (one level, one layer) is
//     moved to TRANSFER_SRC with the image's own barrier tracking (Image::Transit) and moved back to the layout and
//     access it had after the copies;
//   - Entry: the target's sampled tiles -> readback E;
//   - Exit: the tiles -> readback A (the emulated pass's output) and, for the control pass, the tiles -> the control
//     pass's shadow image (an image of our own, same format) -> readback B (the native pass's output as =2 would read
//     a real native pass's shadow target).
// Tiles: a 12 x 12 grid of 16 x 16 texels over the subresource (2-3% of a 1080p target), the grid's origin moved per
// instance so that a long session covers the whole image. Raw texel bytes (any format whose texel size is known here;
// the depth aspect only for depth / stencil formats, its undefined top byte masked for 24-bit depth).
// Readback: one host-visible buffer of 10 frame slots of 6 MiB; a job's copies go to the slot of its frame and are
// compared on the CPU 8 frames later (CollectGpu, at a flip: the GPU is done with them by then; a slot is used again
// 10 frames later). Nothing here is read by the game or by the emulated path: the copies only read the game's images.
#include "native/verify-gpu.h"
#include "native/native-render.h"
#include "native/verify-jobs.h"

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include <vk_mem_alloc.h>

namespace NativeRender {
namespace {

namespace LG = Libs::Graphics;
using Verify::ImageOutcome;
using Verify::Job;
using Verify::JobKind;
using Verify::Refusal;
using Verify::TargetRef;

constexpr uint32_t Grid      = 12;
constexpr uint32_t TileSize  = 16;
constexpr uint32_t SlotCount = 10;
constexpr uint64_t SlotBytes = uint64_t {6} << 20u;
constexpr uint64_t Delay     = 8;

struct Shadow {
	VkImage       image  = VK_NULL_HANDLE;
	VmaAllocation memory = nullptr;
	vk::Format    format = vk::Format::eUndefined;
	uint32_t      width = 0, height = 0;
	uint64_t      used_flip = 0;
};

struct Tiles {
	std::vector<vk::BufferImageCopy> regions; // buffer offsets relative to 0
	uint64_t                         bytes  = 0;
	uint32_t                         texels = 0;
};

struct Pending {
	uint64_t     flip = 0;
	ImageOutcome outcome;
	uint32_t     texel = 0;
	bool         mask24 = false;
	uint64_t     a = 0, b = 0, e = 0, bytes = 0;
	bool         has_b = false;
};

struct EntryCopy {
	uint64_t  instance = 0, flip = 0, offset = 0, bytes = 0;
	TargetRef target;
};

struct GpuState {
	bool                   failed    = false;
	VmaAllocator           allocator = nullptr;
	vk::Device             device    = nullptr;
	VkBuffer               readback  = VK_NULL_HANDLE;
	VmaAllocation          readback_memory = nullptr;
	const uint8_t*         mapped    = nullptr;
	uint64_t               slot_flip[SlotCount] {};
	uint64_t               slot_used[SlotCount] {};
	std::vector<std::unique_ptr<Shadow>> shadows;
	std::vector<Pending>   pending;
	std::vector<EntryCopy> entries;
};
GpuState g;

// Bytes per texel as a buffer copy writes them (the depth aspect for depth formats); 0: not known here.
uint32_t TexelBytes(vk::Format f, bool* depth, bool* mask24) {
	*depth  = false;
	*mask24 = false;
	switch (f) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: *depth = true; return 2;
		case vk::Format::eX8D24UnormPack32:
		case vk::Format::eD24UnormS8Uint: *depth = true; *mask24 = true; return 4;
		case vk::Format::eD32Sfloat:
		case vk::Format::eD32SfloatS8Uint: *depth = true; return 4;
		case vk::Format::eR8Unorm:
		case vk::Format::eR8Snorm:
		case vk::Format::eR8Uint:
		case vk::Format::eR8Sint: return 1;
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR8G8Snorm:
		case vk::Format::eR8G8Uint:
		case vk::Format::eR8G8Sint:
		case vk::Format::eR16Unorm:
		case vk::Format::eR16Snorm:
		case vk::Format::eR16Uint:
		case vk::Format::eR16Sint:
		case vk::Format::eR16Sfloat:
		case vk::Format::eR5G6B5UnormPack16:
		case vk::Format::eB5G6R5UnormPack16:
		case vk::Format::eR4G4B4A4UnormPack16:
		case vk::Format::eB4G4R4A4UnormPack16:
		case vk::Format::eR5G5B5A1UnormPack16:
		case vk::Format::eB5G5R5A1UnormPack16:
		case vk::Format::eA1R5G5B5UnormPack16: return 2;
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Snorm:
		case vk::Format::eR8G8B8A8Uint:
		case vk::Format::eR8G8B8A8Sint:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA8B8G8R8UnormPack32:
		case vk::Format::eA8B8G8R8SrgbPack32:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2B10G10R10UintPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eB10G11R11UfloatPack32:
		case vk::Format::eE5B9G9R9UfloatPack32:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR16G16Snorm:
		case vk::Format::eR16G16Uint:
		case vk::Format::eR16G16Sint:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR32Uint:
		case vk::Format::eR32Sint:
		case vk::Format::eR32Sfloat: return 4;
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR16G16B16A16Snorm:
		case vk::Format::eR16G16B16A16Uint:
		case vk::Format::eR16G16B16A16Sint:
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32G32Uint:
		case vk::Format::eR32G32Sint:
		case vk::Format::eR32G32Sfloat: return 8;
		case vk::Format::eR32G32B32A32Uint:
		case vk::Format::eR32G32B32A32Sint:
		case vk::Format::eR32G32B32A32Sfloat: return 16;
		default: return 0;
	}
}

// The tiles of a w x h subresource for instance `instance` (the same for its entry and exit copies).
Tiles MakeTiles(uint32_t w, uint32_t h, uint32_t texel, vk::ImageAspectFlags aspect, uint32_t level, uint32_t layer,
                uint64_t instance) {
	Tiles          t;
	const uint32_t cw = std::max(1u, w / Grid), ch = std::max(1u, h / Grid);
	const uint32_t tw = std::min(TileSize, cw), th = std::min(TileSize, ch);
	const uint32_t ox = static_cast<uint32_t>((instance * 7u) % (cw - tw + 1u));
	const uint32_t oy = static_cast<uint32_t>((instance * 11u) % (ch - th + 1u));
	for (uint32_t gy = 0; gy < Grid; ++gy)
		for (uint32_t gx = 0; gx < Grid; ++gx) {
			const uint32_t x = gx * cw + ox, y = gy * ch + oy;
			if (x + tw > w || y + th > h) continue;
			vk::BufferImageCopy r {};
			r.bufferOffset      = t.bytes;
			r.bufferRowLength   = 0;
			r.bufferImageHeight = 0;
			r.imageSubresource  = vk::ImageSubresourceLayers {aspect, level, layer, 1};
			r.imageOffset       = vk::Offset3D {static_cast<int32_t>(x), static_cast<int32_t>(y), 0};
			r.imageExtent       = vk::Extent3D {tw, th, 1};
			t.regions.push_back(r);
			t.bytes += (uint64_t {tw} * th * texel + 15u) & ~uint64_t {15};
			t.texels += tw * th;
		}
	return t;
}

bool EnsureReadback(LG::GraphicContext& graphics) {
	if (g.readback != VK_NULL_HANDLE) return true;
	if (g.failed) return false;
	g.failed = true;
	VkBufferCreateInfo info {};
	info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	info.size  = SlotCount * SlotBytes;
	info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &info, &host, &g.readback, &g.readback_memory, &mapped) != VK_SUCCESS) {
		g.readback = VK_NULL_HANDLE;
		std::printf("[f5v] no se pudo crear el buffer de lectura (%llu MiB): sin comparacion de imagen\n",
		            static_cast<unsigned long long>(SlotCount * SlotBytes >> 20u));
		std::fflush(stdout);
		return false;
	}
	g.mapped    = static_cast<const uint8_t*>(mapped.pMappedData);
	g.allocator = graphics.allocator;
	g.device    = graphics.device;
	for (auto& f: g.slot_flip) f = UINT64_MAX;
	g.failed = false;
	return true;
}

// The control pass's shadow target for (format, w, h): kept while used; one unused for 600 frames is destroyed (the
// GPU is long done with it).
Shadow* EnsureShadow(vk::Format format, uint32_t w, uint32_t h, bool depth, uint64_t flip) {
	for (auto it = g.shadows.begin(); it != g.shadows.end();) {
		auto& s = **it;
		if (s.format == format && s.width == w && s.height == h) {
			s.used_flip = flip;
			return &s;
		}
		if (flip > s.used_flip + 600) {
			vmaDestroyImage(g.allocator, s.image, s.memory);
			it = g.shadows.erase(it);
		} else {
			++it;
		}
	}
	if (g.shadows.size() >= 12) return nullptr;
	auto              s = std::make_unique<Shadow>();
	VkImageCreateInfo create {};
	create.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	create.imageType   = VK_IMAGE_TYPE_2D;
	create.format      = static_cast<VkFormat>(format);
	create.extent      = {w, h, 1};
	create.mipLevels   = 1;
	create.arrayLayers = 1;
	create.samples     = VK_SAMPLE_COUNT_1_BIT;
	create.tiling      = VK_IMAGE_TILING_OPTIMAL;
	create.usage       = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	(void)depth;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateImage(g.allocator, &create, &allocate, &s->image, &s->memory, nullptr) != VK_SUCCESS) return nullptr;
	s->format    = format;
	s->width     = w;
	s->height    = h;
	s->used_flip = flip;
	g.shadows.push_back(std::move(s));
	return g.shadows.back().get();
}

void ShadowBarrier(vk::CommandBuffer cmd, VkImage image, vk::ImageAspectFlags aspects, vk::ImageLayout from,
                   vk::ImageLayout to, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src,
                   vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst) {
	vk::ImageMemoryBarrier2 b {};
	b.srcStageMask        = src_stage;
	b.srcAccessMask       = src;
	b.dstStageMask        = dst_stage;
	b.dstAccessMask       = dst;
	b.oldLayout           = from;
	b.newLayout           = to;
	b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image               = vk::Image(image);
	b.subresourceRange    = vk::ImageSubresourceRange {aspects, 0, 1, 0, 1};
	vk::DependencyInfo dep {};
	dep.imageMemoryBarrierCount = 1;
	dep.pImageMemoryBarriers    = &b;
	cmd.pipelineBarrier2(dep);
}

// Room in this frame's slot for `bytes`; UINT64_MAX: none.
uint64_t Reserve(uint64_t flip, uint64_t bytes) {
	const uint32_t s = static_cast<uint32_t>(flip % SlotCount);
	if (g.slot_flip[s] != flip) {
		g.slot_flip[s] = flip;
		g.slot_used[s] = 0;
	}
	if (g.slot_used[s] + bytes > SlotBytes) return UINT64_MAX;
	const uint64_t at = s * SlotBytes + g.slot_used[s];
	g.slot_used[s] += (bytes + 255u) & ~uint64_t {255};
	return at;
}

std::vector<vk::BufferImageCopy> At(const Tiles& t, uint64_t base) {
	std::vector<vk::BufferImageCopy> r = t.regions;
	for (auto& x: r) x.bufferOffset += base;
	return r;
}

void RunJob(LG::CommandBuffer& buffer, const Job& job, uint64_t flip) {
	auto& textures = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < job.count; ++i) {
		const TargetRef& t     = job.targets[i];
		LG::Image*       image = textures.TryGetImage(LG::ImageId(t.index, t.generation));
		if (image == nullptr || !image->registered || image->backing.image == nullptr) {
			Verify::ImageRefused(job.pass, Verify::RNoImage);
			continue;
		}
		const auto& b = image->backing;
		bool        depth = false, mask24 = false;
		const uint32_t texel = TexelBytes(b.format, &depth, &mask24);
		if (texel == 0) {
			Verify::ImageRefused(job.pass, Verify::RFormat);
			continue;
		}
		if (b.samples != 1) {
			Verify::ImageRefused(job.pass, Verify::RSamples);
			continue;
		}
		if (b.image_type != vk::ImageType::e2D || t.level >= b.mip_levels || t.layer >= b.layers) {
			Verify::ImageRefused(job.pass, Verify::RType);
			continue;
		}
		const uint32_t layers = image->info.resources.layers;
		const uint32_t index  = t.level * layers + t.layer;
		if (!b.subresource_states.empty() && index >= b.subresource_states.size()) {
			Verify::ImageRefused(job.pass, Verify::RType);
			continue;
		}
		const LG::VulkanImageState prev = b.subresource_states.empty() ? b.state : b.subresource_states[index];
		if (prev.layout == vk::ImageLayout::eUndefined) {
			Verify::ImageRefused(job.pass, Verify::RLayout);
			continue;
		}
		const uint32_t w = std::max(1u, b.extent.width >> t.level), h = std::max(1u, b.extent.height >> t.level);
		const vk::ImageAspectFlags aspect = depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;
		const Tiles tiles = MakeTiles(w, h, texel, aspect, t.level, t.layer, job.instance);
		if (tiles.regions.empty()) {
			Verify::ImageRefused(job.pass, Verify::RType);
			continue;
		}
		const bool    exit   = job.kind == JobKind::Exit;
		const uint64_t need  = exit ? tiles.bytes * 2u : tiles.bytes;
		const uint64_t at    = Reserve(flip, need);
		if (at == UINT64_MAX) {
			Verify::ImageRefused(job.pass, Verify::RBudget);
			continue;
		}
		Shadow* shadow = nullptr;
		if (exit) {
			shadow = EnsureShadow(b.format, w, h, depth, flip);
			if (shadow == nullptr) Verify::ImageRefused(job.pass, Verify::RShadow);
		}
		buffer.EndRendering();
		const vk::CommandBuffer cmd = buffer.Handle();
		const LG::ImageSubresourceRange range {t.level, 1, t.layer, 1};
		image->Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, range, cmd);
		const auto a_regions = At(tiles, at);
		cmd.copyImageToBuffer(b.image, vk::ImageLayout::eTransferSrcOptimal, vk::Buffer(g.readback),
		                      static_cast<uint32_t>(a_regions.size()), a_regions.data());
		uint64_t b_at = 0;
		if (shadow != nullptr) {
			// The control pass: its shadow target holds the emulated output (same texels), read back as a real native
			// pass's shadow target would be.
			const bool stencil = b.format == vk::Format::eD16UnormS8Uint || b.format == vk::Format::eD24UnormS8Uint ||
			                     b.format == vk::Format::eD32SfloatS8Uint;
			const vk::ImageAspectFlags all =
			    !depth ? vk::ImageAspectFlags {vk::ImageAspectFlagBits::eColor}
			           : vk::ImageAspectFlagBits::eDepth | (stencil ? vk::ImageAspectFlags {vk::ImageAspectFlagBits::eStencil}
			                                                         : vk::ImageAspectFlags {});
			ShadowBarrier(cmd, shadow->image, all, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
			              vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
			              vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
			std::vector<vk::ImageCopy> copies;
			copies.reserve(tiles.regions.size());
			for (const auto& r: tiles.regions) {
				vk::ImageCopy c {};
				c.srcSubresource = r.imageSubresource;
				c.srcOffset      = r.imageOffset;
				c.dstSubresource = vk::ImageSubresourceLayers {aspect, 0, 0, 1};
				c.dstOffset      = r.imageOffset;
				c.extent         = r.imageExtent;
				copies.push_back(c);
			}
			cmd.copyImage(b.image, vk::ImageLayout::eTransferSrcOptimal, vk::Image(shadow->image),
			              vk::ImageLayout::eTransferDstOptimal, static_cast<uint32_t>(copies.size()), copies.data());
			ShadowBarrier(cmd, shadow->image, all, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
			              vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
			              vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
			b_at = at + tiles.bytes;
			auto b_regions = At(tiles, b_at);
			for (auto& r: b_regions) r.imageSubresource = vk::ImageSubresourceLayers {aspect, 0, 0, 1};
			cmd.copyImageToBuffer(vk::Image(shadow->image), vk::ImageLayout::eTransferSrcOptimal, vk::Buffer(g.readback),
			                      static_cast<uint32_t>(b_regions.size()), b_regions.data());
		}
		vk::MemoryBarrier2 host {};
		host.srcStageMask  = vk::PipelineStageFlagBits2::eCopy;
		host.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		host.dstStageMask  = vk::PipelineStageFlagBits2::eHost;
		host.dstAccessMask = vk::AccessFlagBits2::eHostRead;
		vk::DependencyInfo dep {};
		dep.memoryBarrierCount = 1;
		dep.pMemoryBarriers    = &host;
		cmd.pipelineBarrier2(dep);
		// The target back where it was (its layout and access: the next draw finds it as it left it).
		image->Transit(prev.layout, prev.access_mask, range, cmd);
		buffer.InvalidateGraphicsState();
		if (!exit) {
			g.entries.push_back(EntryCopy {job.instance, flip, at, tiles.bytes, t});
			continue;
		}
		Pending p;
		p.flip               = flip;
		p.outcome.pass       = job.pass;
		p.outcome.instance   = job.instance;
		p.outcome.format     = static_cast<uint32_t>(b.format);
		p.outcome.width      = w;
		p.outcome.height     = h;
		p.outcome.level      = t.level;
		p.outcome.layer      = t.layer;
		p.outcome.depth      = depth;
		p.outcome.texels     = tiles.texels;
		p.texel              = texel;
		p.mask24             = mask24;
		p.a                  = at;
		p.b                  = b_at;
		p.has_b              = shadow != nullptr;
		p.bytes              = tiles.bytes;
		for (auto it = g.entries.begin(); it != g.entries.end(); ++it)
			if (it->instance == job.instance && it->target == t && it->bytes == tiles.bytes) {
				p.e                   = it->offset;
				p.outcome.entry_known = true;
				g.entries.erase(it);
				break;
			}
		if (!p.outcome.entry_known) p.e = UINT64_MAX;
		g.pending.push_back(p);
	}
}

// Texels of the tile set at `x` and `y` (the same layout: rows of tightly packed tiles, each padded to 16 bytes) that
// differ.
uint64_t Differ(const uint8_t* x, const uint8_t* y, uint64_t bytes, uint32_t texel, bool mask24) {
	uint64_t n = 0;
	if (mask24) {
		for (uint64_t i = 0; i + 4 <= bytes; i += 4) {
			uint32_t a = 0, b = 0;
			std::memcpy(&a, x + i, 4);
			std::memcpy(&b, y + i, 4);
			n += ((a ^ b) & 0x00ffffffu) != 0;
		}
		return n;
	}
	for (uint64_t i = 0; i + texel <= bytes; i += texel) n += std::memcmp(x + i, y + i, texel) != 0;
	return n;
}

} // namespace

void RunGpuWork(LG::CommandBuffer& buffer) {
	if (buffer.IsInvalid()) {
		Job job;
		while (Verify::TakeJob(job)) Verify::ImageRefused(job.pass, Verify::RNoBuffer);
		return;
	}
	auto& graphics = buffer.GetContext().GetGraphics();
	const bool ready = EnsureReadback(graphics);
	const uint64_t flip = Verify::FlipCount();
	Job job;
	while (Verify::TakeJob(job)) {
		if (!ready) {
			Verify::ImageRefused(job.pass, Verify::RNoBuffer);
			continue;
		}
		RunJob(buffer, job, flip);
	}
	// Entry copies whose exit never came.
	g.entries.erase(std::remove_if(g.entries.begin(), g.entries.end(), [&](const EntryCopy& e) { return flip > e.flip + 2; }),
	                g.entries.end());
}

namespace Verify {
void CollectGpu(uint64_t flip) {
	if (g.mapped == nullptr || g.pending.empty()) return;
	bool invalidated = false;
	for (size_t i = 0; i < g.pending.size();) {
		Pending& p = g.pending[i];
		if (flip < p.flip + Delay) {
			++i;
			continue;
		}
		if (!invalidated) {
			(void)vmaInvalidateAllocation(g.allocator, g.readback_memory, 0, VK_WHOLE_SIZE);
			invalidated = true;
		}
		ImageOutcome o = p.outcome;
		// Tile by tile, as the copies laid them out (each tile padded to 16 bytes: the padding is never compared).
		o.differing = 0;
		o.changed   = 0;
		const uint32_t cw = std::max(1u, o.width / Grid), ch = std::max(1u, o.height / Grid);
		const uint32_t tw = std::min(TileSize, cw), th = std::min(TileSize, ch);
		const uint64_t tile_bytes = uint64_t {tw} * th * p.texel;
		const uint64_t stride     = (tile_bytes + 15u) & ~uint64_t {15};
		for (uint64_t off = 0; off + tile_bytes <= p.bytes; off += stride) {
			if (p.has_b) o.differing += Differ(g.mapped + p.a + off, g.mapped + p.b + off, tile_bytes, p.texel, p.mask24);
			if (o.entry_known) o.changed += Differ(g.mapped + p.a + off, g.mapped + p.e + off, tile_bytes, p.texel, p.mask24);
		}
		if (p.has_b) ImageResult(o); // (no shadow: refused when it was copied, not a comparison)
		g.pending[i] = g.pending.back();
		g.pending.pop_back();
	}
}
} // namespace Verify

} // namespace NativeRender
