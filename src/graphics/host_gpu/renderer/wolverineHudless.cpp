#include "graphics/host_gpu/renderer/wolverineHudless.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <map>
#include <mutex>

namespace Libs::Graphics::WolverineHudless {

namespace {

// The draw that merges scene and UI layer into the flip buffer while loading (VS 726e434a3fc68410; one triangle).
constexpr uint64_t FinalMerge = 0x44082d40f4392cacull;
// The compute pass that does it at the title and in play: reads the scene and the UI layer (an 8-bit RGBA display-size
// image), writes the flip buffer as storage.
constexpr uint64_t FinalMergeCompute = 0x0cd99ace1ed451f6ull;

struct Capture {
	VulkanImage image;
	bool        fresh = false; // taken since the buffer's last flip
};

// The UI layer: the last display-size 8-bit RGBA target drawn this frame.
struct UiLayer {
	ImageId  id;
	uint64_t address = 0;
	int      frame   = -1;
};

// GPU time of the second merge (save, clear, draw, copy, restore), by a timestamp pair per merge.
constexpr uint32_t TimerSlots = 16;
struct Timer {
	vk::QueryPool                     pool = nullptr;
	std::array<bool, TimerSlots>      pending {};
	uint32_t                          next = 0;
	double                            ms_sum = 0.0;
	uint32_t                          count  = 0;
	std::chrono::steady_clock::time_point window {};
};

std::mutex                  g_mutex;
bool                        g_logged_compute = false;
std::map<uint64_t, Capture> g_captures; // by the flip buffer's guest address
UiLayer                     g_ui;
VulkanImage                 g_ui_saved;
Timer                       g_timer;
bool                        g_logged_images = false;
thread_local bool           t_in_merge      = false;

void Copy(Cmd command, vk::Image source, VulkanImage& destination) {
	Transit(command, destination, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	vk::ImageCopy copy {};
	copy.srcSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.dstSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.extent         = destination.extent;
	command.copyImage(source, vk::ImageLayout::eTransferSrcOptimal, destination.image,
	                  vk::ImageLayout::eTransferDstOptimal, 1, &copy);
}

// The address of each image the pixel program reads (0 where the descriptor is too short to tell).
std::vector<ShaderTextureResource> Images(const ShaderStageRuntime& pixel) {
	std::vector<ShaderTextureResource> images;
	for (const auto& value: pixel.resources->images) {
		ShaderTextureResource image {};
		std::memcpy(image.fields, value.dwords.data(),
		            sizeof(uint32_t) * std::min<size_t>(value.dword_count, std::size(image.fields)));
		images.push_back(image);
	}
	return images;
}

// Reads the timestamp pairs that are done, and logs the mean every 5 s.
void ReadTimer(GraphicContext& graphics) {
	auto& t = g_timer;
	for (uint32_t slot = 0; slot < TimerSlots; ++slot) {
		std::array<uint64_t, 2> stamps {};
		if (!t.pending[slot] ||
		    graphics.device.getQueryPoolResults(t.pool, slot * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess) {
			continue;
		}
		t.pending[slot] = false;
		t.ms_sum += static_cast<double>(stamps[1] - stamps[0]) *
		            graphics.physical_device_properties.limits.timestampPeriod / 1e6;
		t.count++;
	}
	const auto now = std::chrono::steady_clock::now();
	if (t.window == std::chrono::steady_clock::time_point {}) {
		t.window = now;
	}
	if (now - t.window >= std::chrono::seconds(5) && t.count > 0) {
		LOGF("Wolverine HUD-less (5 s): %u second merges, %.3f ms of GPU time each\n", t.count, t.ms_sum / t.count);
		t.ms_sum = 0.0;
		t.count  = 0;
		t.window = now;
	}
}

} // namespace

void Ensure(GraphicContext& graphics, VulkanImage& image, vk::Extent2D extent, vk::Format format,
            vk::ImageUsageFlags usage, CommandScheduler* scheduler) {
	if (image.image != nullptr && image.extent.width == extent.width && image.extent.height == extent.height &&
	    image.format == format && image.usage == usage) {
		return;
	}
	if (image.image != nullptr && scheduler != nullptr) {
		scheduler->DeferOperation([&graphics, handle = image.image, allocation = image.allocation] {
			VulkanImage old;
			old.image      = handle;
			old.allocation = allocation;
			graphics.DeleteImage(old);
		});
		image.image      = nullptr;
		image.allocation = nullptr;
		image.state      = {};
	} else if (image.image != nullptr) {
		graphics.DeleteImage(image);
		image.state = {};
	}
	vk::ImageCreateInfo create {};
	create.sType         = vk::StructureType::eImageCreateInfo;
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
	if (!graphics.CreateImage(create, image)) {
		EXIT("failed to allocate a HUD-less image, extent=%ux%u format=%d\n", extent.width, extent.height,
		     static_cast<int>(format));
	}
}

void Transit(Cmd command, VulkanImage& image, vk::ImageLayout layout, vk::AccessFlags2 access) {
	const auto stage = access == vk::AccessFlagBits2::eTransferRead || access == vk::AccessFlagBits2::eTransferWrite
	                       ? vk::PipelineStageFlagBits2::eTransfer
	                       : vk::PipelineStageFlagBits2::eAllCommands;
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
	barrier.subresourceRange.levelCount     = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount     = 1;
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image.state = {stage, access, layout};
}

// The layouts a dispatch's descriptors give the UI layer and the flip buffer: the merge's dispatch runs on them as
// bound, so both go back to them. A draw binds its images again itself (nullptr).
struct Bound {
	vk::ImageLayout ui;
	vk::ImageLayout flip;
};

// Keeps the UI layer, clears it, runs `merge` (into the flip buffer), keeps the flip buffer as its HUD-less copy and
// puts the UI layer back. Called with g_mutex held.
static void MergeWithoutUi(CommandBuffer& buffer, GraphicContext& graphics, Image& ui, Image& flip,
                           const std::function<void()>& merge, const Bound* bound) {
	auto&      capture     = g_captures[flip.info.data.address];
	const auto ui_extent   = vk::Extent2D {ui.backing.extent.width, ui.backing.extent.height};
	const auto flip_extent = vk::Extent2D {flip.backing.extent.width, flip.backing.extent.height};
	auto* scheduler = &buffer.GetContext().GetCommandScheduler();
	Ensure(graphics, g_ui_saved, ui_extent, ui.backing.format,
	       vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst, scheduler);
	Ensure(graphics, capture.image, flip_extent, flip.backing.format,
	       vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst, scheduler);
	if (g_timer.pool == nullptr) {
		vk::QueryPoolCreateInfo info {};
		info.sType      = vk::StructureType::eQueryPoolCreateInfo;
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = TimerSlots * 2;
		if (graphics.device.createQueryPool(&info, nullptr, &g_timer.pool) != vk::Result::eSuccess) {
			EXIT("failed to create the HUD-less timer\n");
		}
		LOGF("Wolverine HUD-less: merging twice, UI layer %ux%u format %d, flip buffer format %d\n", ui_extent.width,
		     ui_extent.height, static_cast<int>(ui.backing.format), static_cast<int>(flip.backing.format));
	}
	ReadTimer(graphics);
	const uint32_t slot  = g_timer.next;
	const bool     timed = !g_timer.pending[slot];
	g_timer.next         = (slot + 1) % TimerSlots;
	const auto shader_access = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;

	// Keep the UI layer, clear it, and merge into the flip buffer: the frame without its HUD.
	buffer.EndRendering();
	auto command = buffer.Recorder();
	if (timed) {
		command.resetQueryPool(g_timer.pool, slot * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_timer.pool, slot * 2);
	}
	ui.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	Copy(command, ui.backing.image, g_ui_saved);
	ui.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	const vk::ClearColorValue       clear {};
	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	command.clearColorImage(ui.backing.image, vk::ImageLayout::eTransferDstOptimal, clear, range);
	if (bound != nullptr) {
		ui.Transit(bound->ui, vk::AccessFlagBits2::eShaderRead, {}, command);
		flip.Transit(bound->flip, shader_access, {}, command);
	}
	t_in_merge = true;
	merge();
	t_in_merge = false;

	// Keep that frame, put the UI layer back; the game's own merge follows and overwrites the flip buffer.
	buffer.EndRendering();
	command = buffer.Recorder();
	flip.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	Copy(command, flip.backing.image, capture.image);
	capture.fresh = true;
	Transit(command, g_ui_saved, vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead);
	ui.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	vk::ImageCopy back {};
	back.srcSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	back.dstSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	back.extent         = g_ui_saved.extent;
	command.copyImage(g_ui_saved.image, vk::ImageLayout::eTransferSrcOptimal, ui.backing.image,
	                  vk::ImageLayout::eTransferDstOptimal, 1, &back);
	if (bound != nullptr) {
		ui.Transit(bound->ui, vk::AccessFlagBits2::eShaderRead, {}, command);
		flip.Transit(bound->flip, shader_access, {}, command);
	}
	if (timed) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_timer.pool, slot * 2 + 1);
		g_timer.pending[slot] = true;
	}
}

void BeforeDraw(CommandBuffer& buffer, const ShaderStageRuntime& pixel, const RenderColorInfo& target,
                const std::function<void()>& draw) {
	if (t_in_merge) {
		return;
	}
	auto&     context = buffer.GetContext();
	const int frame   = context.GetGpu().GetFrameNum();
	if (pixel.program->shader_hash != FinalMerge) {
		const auto extent = target.Extent();
		if (target.desc.info.pixel_format == vk::Format::eR8G8B8A8Unorm && extent.width >= 2560) {
			std::lock_guard lock(g_mutex);
			g_ui = {target.image_id, target.desc.info.data.address, frame};
		}
		return;
	}
	const auto images = Images(pixel);
	auto&      cache  = context.GetTextureCache();
	auto&      graphics = context.GetGraphics();
	std::lock_guard lock(g_mutex);
	if (!g_logged_images) {
		g_logged_images = true;
		for (size_t i = 0; i < images.size(); ++i) {
			LOGF("Wolverine HUD-less: final merge image[%zu] addr=0x%010" PRIx64 " fmt=%u extent=%ux%u%s\n", i,
			     images[i].Base40(), static_cast<uint32_t>(images[i].Format()),
			     static_cast<uint32_t>(images[i].Width5()) + 1u, static_cast<uint32_t>(images[i].Height5()) + 1u,
			     g_ui.frame == frame && images[i].Base40() == g_ui.address ? " (UI layer)" : "");
		}
	}
	const bool reads_ui = g_ui.frame == frame &&
	                      std::ranges::any_of(images, [](const auto& image) { return image.Base40() == g_ui.address; });
	if (!reads_ui) {
		return; // no HUD drawn this frame (or not found): the flip goes without a HUD-less copy
	}
	// shortcut: assumes the UI layer's cache slot lives from its draws to this one in the same frame; check the
	// address before use.
	auto& ui = cache.GetImage(g_ui.id);
	if (ui.info.data.address != g_ui.address) {
		return;
	}
	MergeWithoutUi(buffer, graphics, ui, cache.GetImage(target.image_id), draw, nullptr);
}

void BeforeDispatch(CommandBuffer& buffer, uint64_t hash, std::span<const ShaderRecompiler::IR::ImageResource> resources,
                    std::span<const TextureBinding> images, const std::function<void()>& dispatch) {
	if (hash != FinalMergeCompute || t_in_merge || resources.size() != images.size()) {
		return;
	}
	auto&                   context = buffer.GetContext();
	auto&                   cache   = context.GetTextureCache();
	const TextureBinding*   ui      = nullptr;
	const TextureBinding*   flip    = nullptr;
	uint32_t                uis     = 0;
	uint32_t                flips   = 0;
	for (size_t i = 0; i < images.size(); ++i) {
		const auto& info = images[i].desc.info;
		if (!images[i].image_id || info.extent.width < 2560) {
			continue;
		}
		if (resources[i].written && resources[i].resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
			flip = &images[i];
			flips++;
		} else if (!resources[i].written &&
		           (info.pixel_format == vk::Format::eR8G8B8A8Unorm || info.pixel_format == vk::Format::eR8G8B8A8Srgb)) {
			ui = &images[i];
			uis++;
		}
	}
	std::lock_guard lock(g_mutex);
	if (!g_logged_compute) {
		g_logged_compute = true;
		LOGF("Wolverine HUD-less: compute merge %016" PRIx64 ": %u UI layer(s), %u flip target(s)\n", hash, uis, flips);
		for (size_t i = 0; i < images.size(); ++i) {
			const auto& info = images[i].desc.info;
			LOGF("Wolverine HUD-less:   image[%zu] addr=0x%010" PRIx64 " %ux%u vk format %d %s%s\n", i, info.data.address,
			     info.extent.width, info.extent.height, static_cast<int>(info.pixel_format),
			     resources[i].written ? "written" : "read", &images[i] == ui ? " (UI layer)" : &images[i] == flip ? " (flip)" : "");
		}
	}
	if (uis != 1 || flips != 1) {
		return; // not the merge as seen at the title: leave it alone
	}
	const Bound bound {ui->layout, flip->layout};
	MergeWithoutUi(buffer, context.GetGraphics(), cache.GetImage(ui->image_id), cache.GetImage(flip->image_id), dispatch,
	               &bound);
}

bool TakeFor(Cmd command, GraphicContext& graphics, const Image& surface, VulkanImage& into) {
	std::lock_guard lock(g_mutex);
	if (g_timer.pool != nullptr) {
		ReadTimer(graphics);
	}
	const auto      found = g_captures.find(surface.info.data.address);
	if (found == g_captures.end() || !found->second.fresh) {
		return false;
	}
	auto& capture = found->second;
	capture.fresh = false;
	Ensure(graphics, into, {capture.image.extent.width, capture.image.extent.height}, capture.image.format,
	       vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
	           vk::ImageUsageFlagBits::eSampled);
	Transit(command, capture.image, vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead);
	Copy(command, capture.image.image, into);
	return true;
}

} // namespace Libs::Graphics::WolverineHudless
