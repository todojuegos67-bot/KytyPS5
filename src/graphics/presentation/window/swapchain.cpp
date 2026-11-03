#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "gpu_blit_shaders/gpu_blit_fs_triangle_spv.h"
#include "gpu_blit_shaders/present_dither_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window/windowInternal.h"
#include "loader/demonsSoulsRenderSize.h"
#include "loader/demonsSoulsWarp.h"
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "vulkan-recording.h"
#endif
#include "frame-gen.h"
#include "screenshot.h"
#include "anti-lag.h"
#include "fsr-native.h"
#include "frame-gen-hudless.h"
#include "live-counters.h"
#include "local-platform.h"
#include "perf-overlay.h"
#if defined(_WIN32)
#include "SDL_syswm.h"
#endif

#include <condition_variable>
#include <mutex>
#include <thread>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string_view>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <vulkan/vk_platform.h>

// IWYU pragma: no_include <intrin.h>

#define KYTY_ENABLE_DEBUG_PRINTF
#define KYTY_DBG_INPUT

namespace Libs::Graphics {

struct Presenter::Frame {
	VulkanImage            image;
	vk::ImageView          view = nullptr; // `image` sampled (the dithered copy, Swapchain::RecordDitheredCopy)
	FrameGen::FrameInputs  fg; // frame generation inputs (KYTY_FRAMEGEN)
	uint64_t               present_tick = 0;
	bool                   busy         = false;
	bool                   reusing_last = false;
	bool                   hdr          = false; // BT.2100 PQ pixels
	bool                   blank        = false; // a cleared frame (no guest picture)
	bool                   scene        = false; // the game's 3D view (FrameGen::TakeScene)
	bool                   menu         = false; // its interface (a menu) at 16:9 in the middle (FrameGen::TakeMenu)
	uint64_t               anti_lag_frame = 0;   // the frame AMD Anti-Lag numbered it (AntiLag::Flip), 0: none

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
			if (frame->view != nullptr) m_window.graphic_ctx.device.destroyImageView(frame->view, nullptr);
			if (frame->image.image != nullptr) {
				m_window.graphic_ctx.DeleteImage(frame->image);
			}
			FrameGen::DestroyInputs(m_window.graphic_ctx, frame->fg);
		}
		FrameGen::DestroyHudless(m_window.graphic_ctx);
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

	// `wait`: until a frame is free (else none when none is).
	Presenter::Frame* Acquire(bool wait = true) {
		m_mutex.Lock();
		if (m_frames.empty()) {
			EXIT("prepared-frame pool was used before swapchain initialization\n");
		}
		while (m_free.empty()) {
			if (!wait) {
				m_mutex.Unlock();
				return nullptr;
			}
			m_available.Wait(&m_mutex);
		}
		auto* frame = m_free.front();
		m_free.pop_front();
		if (frame->busy) {
			EXIT("prepared-frame pool returned an invalid frame\n");
		}
		if (m_last_frame == frame) {
			m_last_frame = nullptr;
		}
		frame->busy         = true;
		frame->reusing_last = false;
		m_mutex.Unlock();

		WaitForFrame(*frame);
		return frame;
	}

	Presenter::Frame* AcquireLast() {
		m_mutex.Lock();
		auto* frame = m_last_frame;
		if (frame == nullptr) {
			m_mutex.Unlock();
			return nullptr;
		}
		auto free = std::find(m_free.begin(), m_free.end(), frame);
		if (free == m_free.end() || frame->busy) {
			m_mutex.Unlock();
			EXIT("last submitted frame is not available for reuse\n");
		}
		m_free.erase(free);
		m_last_frame        = nullptr;
		frame->busy         = true;
		frame->reusing_last = true;
		m_mutex.Unlock();

		WaitForFrame(*frame);
		return frame;
	}

	void ValidateForPresent(Presenter::Frame* frame, bool reuse) {
		Common::LockGuard lock(m_mutex);
		if (frame == nullptr || !frame->busy || frame->reusing_last != reuse) {
			EXIT("prepared frame has invalid presentation ownership\n");
		}
	}

	void Release(Presenter::Frame* frame, bool make_last = false) {
		if (frame == nullptr) {
			EXIT("cannot release a null prepared frame\n");
		}
		Common::LockGuard lock(m_mutex);
		if (!frame->busy) {
			EXIT("prepared frame was released twice\n");
		}
		frame->busy         = false;
		frame->reusing_last = false;
		if (make_last) {
			m_last_frame = frame;
		}
		m_free.push_back(frame);
		m_available.Signal();
	}

private:
	void WaitForFrame(Presenter::Frame& frame) { m_scheduler.Wait(frame.present_tick); }

	WindowContext&                                 m_window;
	CommandScheduler&                              m_scheduler;
	Common::Mutex                                  m_mutex;
	Common::CondVar                                m_available;
	std::vector<std::unique_ptr<Presenter::Frame>> m_frames;
	std::deque<Presenter::Frame*>                  m_free;
	Presenter::Frame*                              m_last_frame = nullptr;
	vk::Format                                     m_format     = vk::Format::eUndefined;
};

void Presenter::Frame::Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format) {
	if (extent.width == 0 || extent.height == 0 || format == vk::Format::eUndefined) {
		EXIT("unsupported prepared frame, extent=%ux%u format=%d\n", extent.width, extent.height,
		     static_cast<int>(format));
	}
	const auto features = graphics.GetFormatProperties(format).optimalTilingFeatures;
	const auto required =
	    vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eSampledImageFilterLinear |
	    vk::FormatFeatureFlagBits::eTransferSrc | vk::FormatFeatureFlagBits::eTransferDst;
	if ((features & required) != required) {
		EXIT("prepared presentation format lacks optimal blit support: format=%d features=0x%x\n",
		     static_cast<int>(format), static_cast<vk::FormatFeatureFlags::MaskType>(features));
	}

	auto&      dst        = image;
	const bool compatible = dst.image != nullptr && dst.extent.width == extent.width &&
	                        dst.extent.height == extent.height && dst.format == format;
	if (compatible) {
		return;
	}
	if (dst.image != nullptr) {
		if (view != nullptr) graphics.device.destroyImageView(view, nullptr);
		view = nullptr;
		graphics.DeleteImage(dst);
	}

	vk::ImageCreateInfo create {};
	create.sType         = vk::StructureType::eImageCreateInfo;
	create.imageType     = vk::ImageType::e2D;
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
	               vk::ImageUsageFlagBits::eSampled; // (the dithered copy)
	create.sharingMode = vk::SharingMode::eExclusive;
	create.samples     = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, dst)) {
		EXIT("failed to allocate prepared presentation image, extent=%ux%u format=%d\n",
		     extent.width, extent.height, static_cast<int>(format));
	}
	vk::ImageViewCreateInfo info {};
	info.image            = dst.image;
	info.viewType         = vk::ImageViewType::e2D;
	info.format           = format;
	info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&info, nullptr, &view) != vk::Result::eSuccess) view = nullptr;
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
	Transit(command, vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead);
}

void Presenter::Frame::Clear(CommandBuffer& command_buffer, const vk::ClearColorValue& color) {
	command_buffer.EndRendering();
	auto command = command_buffer.Handle();
	Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	command.clearColorImage(image.image, vk::ImageLayout::eTransferDstOptimal, &color, 1, &range);
	Transit(command, vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead);
}

class Swapchain final {
public:
	enum class Status : uint8_t { Success, Recreate, SurfaceLost };

	explicit Swapchain(WindowContext& window): m_window(window) {}
	~Swapchain();
	KYTY_CLASS_NO_COPY(Swapchain);

	void                 Create();
	void                 Recreate(bool surface_lost = false);
	[[nodiscard]] Status AcquireNextImage();
	[[nodiscard]] bool   PrepareSystemOverlay(SystemOverlayHud* hud = nullptr);
	// `source_view`: `source` sampled (the dithered copy), or null.
	// `scene`: the frame shows the game's 3D view (Presenter::IsSceneFrame); `menu`: a menu at 16:9 in the middle of the
	// wide output (FrameGen::TakeMenu), shown alone between black bars with KYTY_PRESENT_ASPECT=fit.
	// `fsr_inputs`: the frame's inputs for FSR's HUD-less image (FrameGen::RecordHudless), or null.
	// `target` (Linux frame generation): the frame finished into that image (the swapchain's size and format; its
	// `target_view` for the system overlay) instead of the acquired swapchain image, left in the general layout.
	void                 RecordPresentCommands(CommandBuffer& command, VulkanImage& source, vk::ImageView source_view,
	                                           bool draw_system_overlay, bool scene, bool menu,
	                                           const FrameGen::FrameInputs* fsr_inputs = nullptr,
	                                           VulkanImage* target = nullptr, vk::ImageView target_view = nullptr);
	// The acquired swapchain image made a copy of `source` (the swapchain's size and format, general layout) and left for
	// the present (Linux frame generation: a generated frame, or the finished one).
	void                 RecordShow(CommandBuffer& command, const VulkanImage& source);
	uint64_t             Submit(CommandScheduler& scheduler);
	[[nodiscard]] Status Present();

	[[nodiscard]] uint32_t ImageCount() const noexcept {
		return static_cast<uint32_t>(m_images.size());
	}
	[[nodiscard]] vk::Format Format() const noexcept { return m_format; }
	[[nodiscard]] vk::Extent2D Extent() const noexcept { return m_extent; }
	[[nodiscard]] bool IsHdrSupported() const noexcept { return m_hdr_supported.load(std::memory_order_relaxed); }
	// The guest's frames are PQ (true) or SDR; true when the swapchain must be recreated for it.
	[[nodiscard]] bool SetHdr(bool hdr) {
		m_hdr_requested = hdr;
		return m_hdr_active != (hdr && IsHdrSupported());
	}
	// Where RecordPresentCommands put the frame in the swapchain image.
	[[nodiscard]] vk::Rect2D PresentRegion() const noexcept { return m_present_region; }
	// The presents wait for the display's vertical blank (VSync: FIFO).
	[[nodiscard]] bool IsFifo() const noexcept { return m_fifo.load(std::memory_order_relaxed); }

private:
	void                 Destroy();
	vk::SurfaceFormatKHR SelectFormat();
	// `source` (a 10-bit frame) drawn into the 8-bit `dst_image` at m_present_region with an ordered dither, where the blit
	// rounded each pixel (issue #234: the loading screens' fog banded); false (nothing recorded) where it does not apply.
	bool RecordDitheredCopy(vk::CommandBuffer command, VulkanImage& source, vk::ImageView source_view,
	                        vk::Image dst_image, vk::ImageView dst_view, uint32_t src_x, uint32_t src_w);

	WindowContext&              m_window;
	vk::SwapchainKHR            m_handle = nullptr;
	vk::Format                  m_format = vk::Format::eUndefined;
	std::atomic_bool            m_hdr_supported {false};
	std::atomic_bool            m_fifo {true};
	bool                        m_hdr_requested = false;
	bool                        m_hdr_active    = false;
	vk::Extent2D                m_extent {};
	std::vector<vk::Image>      m_images;
	std::vector<vk::ImageView>  m_image_views;
	std::vector<vk::Semaphore>  m_image_acquired;
	std::vector<vk::Semaphore>  m_render_complete;
	std::unique_ptr<SystemOverlay> m_system_overlay;
	uint32_t                    m_image_index = static_cast<uint32_t>(-1);
	uint32_t                    m_frame_index = 0;
	vk::Rect2D                  m_present_region {};
	// RecordDitheredCopy's: made on its first use and kept (a pipeline for each 8-bit format: B8G8R8A8, R8G8B8A8).
	vk::Sampler                 m_dither_sampler    = nullptr;
	vk::DescriptorSetLayout     m_dither_set_layout = nullptr;
	vk::PipelineLayout          m_dither_layout     = nullptr;
	std::array<vk::Pipeline, 2> m_dither_pipelines {};
	bool                        m_dither_failed = false;
};

struct Presenter::Impl {
	explicit Impl(WindowContext& owner)
	    : renderer(*owner.render_context), window(owner), swapchain(owner),
	      present_scheduler(renderer, owner.graphic_ctx), frames(owner, present_scheduler) {
		EXIT_IF(owner.render_context == nullptr);
		swapchain.Create();
		frames.Initialize(swapchain.ImageCount(), swapchain.Format());
	}

	void RecoverSwapchain(Swapchain::Status status) {
		LOGF("Recovering Vulkan swapchain%s\n",
		     status == Swapchain::Status::SurfaceLost ? " and surface" : "");
		// (DLSS-G off while the swapchain is made again; the next frame with its inputs turns it on.)
		if (FrameGen::Enabled()) FrameGen::SwapchainChanging();
		swapchain.Recreate(status == Swapchain::Status::SurfaceLost);
		frames.SetFormat(swapchain.Format());
#if defined(_WIN32)
		// The display's refresh again (PresentDue): a window moved to another display, its mode or full screen changed (a
		// TV between 60 and 120 Hz).
		min_present_interval = std::chrono::nanoseconds {-1};
#endif
	}

	Image& ResolveSurface(const ImageInfo& info) {
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
		return image;
	}

	RenderContext&        renderer;
	WindowContext&        window;
	Swapchain             swapchain;
	CommandScheduler      present_scheduler;
	FramePool             frames;
	std::atomic<uint64_t> presented_overlay_revision {0};
	// KYTY_FPS_HUD: the frame rate panel, its numbers refreshed once a second (the emulator's own frame rate: the
	// game's flips; with frame generation the frames presented, generated ones too, and the game's rate below), its
	// frame-time graph with every present; 2: the GPU, processor and memory rows too (PerfOverlay's sampling thread).
	SystemOverlayHud                      hud;
	std::chrono::steady_clock::time_point hud_window {};
	uint32_t                              hud_frames      = 0;
	bool                                  machine_started = false;

	// The detailed panel's rows from the machine's last sample.
	void UpdateMachineRows() {
		const auto machine = PerfOverlay::LatestMachine();
		auto&      perf    = hud.perf;
		perf.rows.clear();
		perf.core_load.clear();
		perf.core_note.clear();
		if (!machine.valid) return;
		constexpr double Gb   = 1024.0 * 1024.0 * 1024.0;
		const auto       text = [](const char* format, auto... values) {
            std::array<char, 160> line {};
            std::snprintf(line.data(), line.size(), format, values...);
            return std::string(line.data());
		};
		std::string gpu = machine.gpu_name;
		if (machine.gpu_valid)
			gpu += text("   %u%%   %u MHz   %u\xC2\xB0" "C   %u W", machine.gpu_load, machine.gpu_clock,
			            machine.gpu_temperature, machine.gpu_power);
		perf.rows.push_back({"GPU", gpu});
		std::string video;
		if (machine.video_usage != 0) video = text("emulator %.1f GB", static_cast<double>(machine.video_usage) / Gb);
		if (machine.gpu_memory_total != 0)
			video += text("%sGPU %.1f / %.1f GB", video.empty() ? "" : "   |   ",
			              static_cast<double>(machine.gpu_memory_used) / Gb, static_cast<double>(machine.gpu_memory_total) / Gb);
		else if (machine.video_budget != 0)
			video += text("  of %.1f GB", static_cast<double>(machine.video_budget) / Gb);
		if (!video.empty()) perf.rows.push_back({"VRAM", video});
		perf.rows.push_back({"CPU", machine.cpu_name + (machine.core_load.empty() ? std::string()
		                                                                          : text("   %.0f%%", machine.cpu_load * 100.0f))});
		perf.core_load = machine.core_load;
		// A hybrid processor's performance cores (the higher efficiency class) apart from its efficiency cores.
		const auto top = machine.core_class.empty() ? 0 : *std::max_element(machine.core_class.begin(), machine.core_class.end());
		const bool hybrid = std::any_of(machine.core_class.begin(), machine.core_class.end(), [&](uint8_t c) { return c != top; });
		perf.core_fast.assign(perf.core_load.size(), 0);
		double   mhz[2] {};
		uint32_t cores[2] {};
		for (size_t i = 0; i < perf.core_load.size(); ++i) {
			const bool fast = hybrid && i < machine.core_class.size() && machine.core_class[i] == top;
			perf.core_fast[i] = fast ? 1 : 0;
			if (i < machine.core_clock.size() && machine.core_clock[i] != 0) {
				mhz[fast ? 1 : 0] += machine.core_clock[i];
				cores[fast ? 1 : 0]++;
			}
		}
		if (hybrid && cores[0] != 0 && cores[1] != 0)
			perf.core_note = text("P-cores %.2f GHz   |   E-cores %.2f GHz", mhz[1] / cores[1] / 1000.0, mhz[0] / cores[0] / 1000.0);
		else if (cores[0] + cores[1] != 0)
			perf.core_note = text("%.2f GHz", (mhz[0] + mhz[1]) / (cores[0] + cores[1]) / 1000.0);
		perf.rows.push_back({"Emulator", text("%.1f cores   |   render thread %.0f%%   |   RAM %.1f GB", machine.process_cores,
		                                      machine.render_thread * 100.0f, static_cast<double>(machine.process_memory) / Gb)});
		if (machine.ram_total != 0)
			perf.rows.push_back({"RAM", text("%.1f / %.1f GB", static_cast<double>(machine.ram_used) / Gb,
			                                 static_cast<double>(machine.ram_total) / Gb)});
	}

	// The shader prefetch's progress line while it runs, and for a few seconds after.
	std::chrono::steady_clock::time_point prefetch_finished {};
	void UpdateStatus(std::chrono::steady_clock::time_point now) {
		const auto progress = renderer.GetPipelineCache().GetPrefetchProgress();
		hud.status.clear();
		if (progress.total == 0) {
			return;
		}
		std::array<char, 128> line {};
		if (progress.done < progress.total || progress.label != nullptr) {
			std::snprintf(line.data(), line.size(), "%s %zu%%  (%zu / %zu)",
			              progress.label != nullptr ? progress.label : "Preparing shaders", progress.done * 100 / progress.total,
			              progress.done, progress.total);
		} else {
			if (prefetch_finished == std::chrono::steady_clock::time_point {}) {
				prefetch_finished = now;
			}
			if (now - prefetch_finished >= std::chrono::seconds(3)) {
				return;
			}
			std::snprintf(line.data(), line.size(), "All shaders ready (%zu)", progress.total);
		}
		hud.status          = line.data();
		hud.status_fraction = static_cast<float>(progress.done) / static_cast<float>(progress.total);
	}

	// With the panel on, a note while the render thread compiles pipelines the precompiled set does not have (a first
	// use's stutter: issue #129 asked for it, as RPCS3 shows its shader compiles), and for 3 s after: how many in that
	// run of them and how long they took.
	PipelineCache::PlayCompiles           play_seen {}, play_shown {};
	std::chrono::steady_clock::time_point play_last {};
	void AddCompileNote(std::chrono::steady_clock::time_point now, int level) {
		const auto made = renderer.GetPipelineCache().GetPlayCompiles();
		if (made.count != play_seen.count) {
			if (now - play_last > std::chrono::seconds(3)) play_shown = {};
			play_shown.count += made.count - play_seen.count;
			play_shown.us += made.us - play_seen.us;
			play_seen = made;
			play_last = now;
		}
		if (level == 0 || play_shown.count == 0 || now - play_last > std::chrono::seconds(3)) return;
		std::array<char, 96> line {};
		std::snprintf(line.data(), line.size(), "Compiling shaders: %llu (%.1f s)", static_cast<unsigned long long>(play_shown.count),
		              static_cast<double>(play_shown.us) / 1e6);
		hud.notice += hud.notice.empty() ? line.data() : std::string("\n") + line.data();
	}

	[[nodiscard]] SystemOverlayHud* UpdateHud(bool reuse) {
		const int  level = PerfOverlay::Level();
		const auto now   = std::chrono::steady_clock::now();
		UpdateStatus(now);
		hud.notice = Loader::DemonsSoulsWarp::HudText();
		AddCompileNote(now, level);
		if (now.time_since_epoch().count() < present_wait_until.load(std::memory_order_relaxed)) {
			const char* text = "Presents held by the driver: NVIDIA Low Latency Mode Ultra?\nSet it to Off or On for kyty_emulator.exe";
			hud.notice += hud.notice.empty() ? std::string(text) : std::string("\n") + text;
		}
		if (level < 2) {
			// (Switched down from detailed with F8: its rows go.)
			hud.perf.rows.clear();
			hud.perf.core_load.clear();
			hud.perf.core_fast.clear();
			hud.perf.core_note.clear();
		}
		if (level == 0) {
			hud.title.clear();
			hud.detail.clear();
			hud.perf.frame_ms.clear();
			hud.perf.busy_ms.clear();
			hud_window = {}; // (the count starts again when it is switched on)
			hud_frames = 0;
			return hud.status.empty() && hud.notice.empty() ? nullptr : PlaceHud();
		}
		if (level >= 2 && !machine_started) {
			// The GPU NVML reports on: the Vulkan device's.
			machine_started = true;
			vk::PhysicalDeviceIDProperties id {};
			vk::PhysicalDeviceProperties2  properties {};
			properties.pNext = &id;
			window.graphic_ctx.physical_device.getProperties2(&properties);
			std::string name = properties.properties.deviceName.data();
			if (name.rfind("NVIDIA ", 0) == 0) name.erase(0, 7);
			PerfOverlay::StartMachine(id.deviceUUID.data(), name);
		}
		if (hud_window == std::chrono::steady_clock::time_point {}) {
			hud_window = now;
		}
		hud_frames += reuse ? 0u : 1u;
		// The graph: the last 300 frames (5 seconds at 60 fps).
		PerfOverlay::RecentFrames(300, &hud.perf.frame_ms, level >= 2 ? &hud.perf.busy_ms : nullptr);
		const double seconds = std::chrono::duration<double>(now - hud_window).count();
		// (Switched on with F8: the panel at once, from the last second's flips, then refreshed each second. It had
		// waited a second empty, which looked like a level without a panel, the user 10-19.)
		const bool first = hud.title.empty();
		if (seconds >= 1.0 || first) {
			// The emulator's own frame rate (the game's flips) is the number; with frame generation, the frames DLSS-G
			// presents (those it got, each with its generated ones) are, and the game's rate is below it.
			const double   game     = PerfOverlay::Summarize(std::max(seconds, 1.0)).fps;
			const uint32_t presents = FrameGen::Enabled() ? FrameGen::PresentedFrames() : 1u;
			std::array<char, 32>  title {};
			std::array<char, 128> detail {};
			if (presents > 1) {
				std::snprintf(title.data(), title.size(), "%.0f FPS",
				              (seconds >= 1.0 ? hud_frames / seconds : game) * presents);
				std::snprintf(detail.data(), detail.size(), "%s Frame Generation %ux  |  game %.0f fps",
				              FrameGen::UsesFsr() || FsrNative::InterpolationRequested() ? FrameGen::FsrName() : "DLSS",
				              presents, game);
			} else {
				std::snprintf(title.data(), title.size(), "%.0f FPS", game);
			}
			hud.title  = title.data();
			hud.detail = detail.data();
			// The last 10 seconds' frames: their average time, the render thread's busy part, the worst 1% as
			// fps and the longest.
			const auto summary = PerfOverlay::Summarize(10.0);
			std::array<char, 160> note {};
			if (summary.fps <= 0) {
				note[0] = '\0';
			} else if (level >= 2 && !hud.perf.busy_ms.empty()) {
				double busy = 0;
				for (const float ms: hud.perf.busy_ms) busy += ms;
				std::snprintf(note.data(), note.size(), "frame %.1f ms   |   render thread %.1f ms   |   1%% low %.0f fps   |   max %.0f ms",
				              1000.0 / summary.fps, busy / static_cast<double>(hud.perf.busy_ms.size()), summary.low1, summary.max_ms);
			} else {
				std::snprintf(note.data(), note.size(), "frame %.1f ms   |   1%% low %.0f fps   |   max %.0f ms",
				              1000.0 / summary.fps, summary.low1, summary.max_ms);
			}
			hud.perf.graph_note = note.data();
			if (level >= 2) UpdateMachineRows();
			hud_window = now;
			hud_frames = 0;
		}
		if (hud.title.empty() && hud.status.empty() && hud.notice.empty()) {
			return nullptr;
		}
		return PlaceHud();
	}
	SystemOverlayHud* PlaceHud() {
		hud.region = swapchain.PresentRegion();
		if (hud.region.extent.width == 0 || hud.region.extent.height == 0) {
			hud.region = vk::Rect2D {{0, 0}, swapchain.Extent()};
		}
		return &hud;
	}
	// Frames lost on the way to the display (issue #129: with DLSS and this thread a game at 30 fps showed ~5 presents a
	// second there): every 30 s, when a frame was replaced while waiting or dropped as not due, or a present call took
	// 50 ms or more, one log line. (Atomics: the vblank thread hands and replaces, the present thread presents.)
	struct PresentStats {
		std::atomic<uint64_t> handed {0}, presented {0}, replaced {0}, not_due {0}, present_us {0}, present_max_us {0};
		std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
	} present_stats;
	// Presents held by the driver (NotePresentCall): noted once; the notice on the game until then (steady_clock ticks).
	bool                 present_wait_noted = false;
	std::atomic<int64_t> present_wait_until {0};
	// Presenter::RequestSwapchainRecreate (the window's thread) for the present thread's next frame.
	std::atomic<bool> recreate_requested {false};
	void NotePresentCall(std::chrono::steady_clock::duration took) {
		auto&          st = present_stats;
		const uint64_t us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(took).count());
		st.presented.fetch_add(1, std::memory_order_relaxed);
		st.present_us.fetch_add(us, std::memory_order_relaxed);
		if (us > st.present_max_us.load(std::memory_order_relaxed)) st.present_max_us.store(us, std::memory_order_relaxed);
		const auto now = std::chrono::steady_clock::now();
		if (now - st.since < std::chrono::seconds(30)) return;
		const uint64_t handed = st.handed.exchange(0), presented = st.presented.exchange(0), replaced = st.replaced.exchange(0);
		const uint64_t not_due = st.not_due.exchange(0), total_us = st.present_us.exchange(0), max_us = st.present_max_us.exchange(0);
		const double   seconds = std::chrono::duration<double>(now - st.since).count();
		st.since               = now;
		// NVIDIA's Low Latency Mode Ultra holds the present call (VSync off: 2.7-6.3 ms on average against 0.4 with it Off
		// or On, frames lost, the frame times uneven: issue #129). Vulkan cannot read the driver's setting, so by the
		// symptom, once a session: a log line and a notice on the game for 20 s. (Not with frame generation, whose
		// presents pace themselves.)
		if (const double average_ms = presented != 0 ? static_cast<double>(total_us) / 1000.0 / static_cast<double>(presented) : 0.0;
		    !present_wait_noted && presented >= 300 && average_ms >= 2.0 && !swapchain.IsFifo() && !FrameGen::Enabled() &&
		    renderer.GetGraphics().GetPhysicalDeviceProperties().vendorID == 0x10de) {
			present_wait_noted = true;
			std::printf("Presenter: each present waited %.1f ms on average with VSync off: NVIDIA's Low Latency Mode Ultra does "
			            "that (frames are lost); set Low Latency Mode to Off or On for kyty_emulator.exe in the NVIDIA "
			            "Control Panel (issue #129)\n",
			            average_ms);
			present_wait_until.store((now + std::chrono::seconds(20)).time_since_epoch().count(), std::memory_order_relaxed);
		}
		if (replaced + not_due == 0 && max_us < 50000) return;
		std::printf("Presenter: %.0f s: %llu frames handed over, %llu presented, %llu replaced while waiting, %llu not due; "
		            "the present call %.2f ms on average, at most %.1f ms\n",
		            seconds, static_cast<unsigned long long>(handed), static_cast<unsigned long long>(presented),
		            static_cast<unsigned long long>(replaced), static_cast<unsigned long long>(not_due),
		            presented != 0 ? static_cast<double>(total_us) / 1000.0 / static_cast<double>(presented) : 0.0,
		            static_cast<double>(max_us) / 1000.0);
		std::fflush(stdout);
	}
	// The presenter's thread (Presenter::PresentLoop) and what it presents next (Submit, SubmitBlank).
	struct Waiting {
		Frame*   frame  = nullptr;
		bool     reuse  = false;
		bool     blank  = false; // a cleared frame to make (width, height, opaque)
		uint32_t width  = 0;
		uint32_t height = 0;
		bool     opaque = false;
		[[nodiscard]] bool Empty() const noexcept { return frame == nullptr && !blank; }
	};
	std::mutex                  waiting_mutex;
	std::condition_variable_any waiting_changed;
	std::condition_variable     waiting_taken; // the thread took the waiting frame (Replace, FrameGen::PresentEveryFrame)
	Waiting                     waiting;
	bool                        presenting = false; // (waiting_mutex) the thread took one and has not presented it yet
	std::jthread                thread;
	// A frame the vblank thread put in place of one still waiting: that one goes back to the pool unshown.
	void Replace(const Waiting& next) {
		Waiting dropped;
		{
			std::unique_lock lock(waiting_mutex);
			// (Each frame Streamline's DLSS upscaled is presented, FrameGen::PresentEveryFrame: a new frame waits until the
			// thread took the one waiting, 250 ms at most, rather than replace it.)
			if (next.frame != nullptr && !next.reuse && FrameGen::PresentEveryFrame()) {
				waiting_taken.wait_for(lock, std::chrono::milliseconds(250),
				                       [&] { return waiting.frame == nullptr || waiting.reuse; });
			}
			dropped = std::exchange(waiting, next);
		}
		if (next.frame != nullptr && !next.reuse) present_stats.handed.fetch_add(1, std::memory_order_relaxed);
		waiting_changed.notify_one();
		// (A game frame DLSS-G had inputs for never shown: the next one's motion spans two frames.)
		if (dropped.frame != nullptr && !dropped.reuse && dropped.frame->fg.valid) FrameGen::NoteDroppedFrame();
		// (A last frame shown again stays the last one.)
		if (dropped.frame != nullptr) {
			if (!dropped.reuse) {
				LiveCounters::Add(LiveCounters::PresentsReplaced);
				present_stats.replaced.fetch_add(1, std::memory_order_relaxed);
			}
			frames.Release(dropped.frame, dropped.reuse);
		}
	}

	// A cleared frame: into `producer` (the GPU thread's command buffer, a guest flip of a blank buffer), else on the
	// presenter's own scheduler (its thread's blank frames).
	void ClearFrame(Frame& frame, uint32_t width, uint32_t height, bool opaque, CommandBuffer* producer) {
		frame.fg.valid = false;
		frame.hdr      = false;
		frame.blank    = true;
		frame.scene    = false;
		frame.menu     = false;
		frame.Configure(window.graphic_ctx, {width, height}, frames.GetFormat());
		vk::ClearColorValue clear {};
		clear.float32[3] = opaque ? 1.0f : 0.0f;
		if (producer != nullptr) {
			frame.Clear(*producer, clear);
		} else {
			auto& command = present_scheduler.BeginCommand();
			frame.Clear(command, clear);
			frame.present_tick = present_scheduler.Submit();
		}
	}

#if defined(__linux__)
	// Linux frame generation (FrameGen::PresenterGenerates; Streamline does this inside vkQueuePresentKHR on Windows): the
	// frame finished at the swapchain's size with its system overlay (`composed`), and the frames NGX makes between two
	// such frames (`generated`), shown before it, evenly spaced over the time between the game's frames.
	VulkanImage                           composed;
	vk::ImageView                         composed_view = nullptr;
	std::array<VulkanImage, 3>            generated;
	std::array<vk::ImageView, 3>          generated_views {};
	bool                                  generation_off = false;
	std::chrono::steady_clock::time_point last_game_frame {};
	double                                game_frame_ms = 0.0; // the game's frame time, smoothed

	~Impl() { DestroyGenerationImages(); }

	void DestroyGenerationImages() {
		if (composed.image == nullptr && generated[0].image == nullptr) return;
		present_scheduler.Wait(present_scheduler.CurrentTick() - 1);
		auto&      graphics = window.graphic_ctx;
		const auto destroy  = [&](VulkanImage& image, vk::ImageView& view) {
			if (view != nullptr) graphics.device.destroyImageView(view, nullptr);
			view = nullptr;
			if (image.image != nullptr) graphics.DeleteImage(image);
		};
		destroy(composed, composed_view);
		for (size_t i = 0; i < generated.size(); ++i) destroy(generated[i], generated_views[i]);
	}

	// The images at the swapchain's size and format (made again when it changes); false when NGX cannot write that format.
	[[nodiscard]] bool ConfigureGenerationImages() {
		const auto extent = swapchain.Extent();
		const auto format = swapchain.Format();
		if (composed.image != nullptr && composed.extent.width == extent.width && composed.extent.height == extent.height &&
		    composed.format == format)
			return true;
		DestroyGenerationImages();
		auto&      graphics = window.graphic_ctx;
		const auto features = graphics.GetFormatProperties(format).optimalTilingFeatures;
		if (!(features & vk::FormatFeatureFlagBits::eStorageImage) || !(features & vk::FormatFeatureFlagBits::eSampledImage)) {
			std::printf("Frame generation: the swapchain's format (%d) cannot be NGX's output (no storage): off\n",
			            static_cast<int>(format));
			std::fflush(stdout);
			return false;
		}
		const auto make = [&](VulkanImage& image, vk::ImageView& view, vk::ImageUsageFlags usage) {
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
			if (!graphics.CreateImage(create, image)) return false;
			vk::ImageViewCreateInfo info {};
			info.image            = image.image;
			info.viewType         = vk::ImageViewType::e2D;
			info.format           = format;
			info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
			return graphics.device.createImageView(&info, nullptr, &view) == vk::Result::eSuccess;
		};
		const auto transfer = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
		bool       made     = make(composed, composed_view,
		                           transfer | vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
		for (size_t i = 0; made && i < generated.size(); ++i)
			made = make(generated[i], generated_views[i],
			            transfer | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled);
		if (!made) {
			std::printf("Frame generation: no memory for its %ux%u frames: off\n", extent.width, extent.height);
			std::fflush(stdout);
			DestroyGenerationImages();
			return false;
		}
		LOGF("frame generation: frames of %ux%u, format %d\n", extent.width, extent.height, static_cast<int>(format));
		return true;
	}

	// The frame finished into `composed`, NGX's frames between the last one and it into `generated`; those shown, then it.
	// False: the frame goes the normal way (frame generation off for good).
	[[nodiscard]] bool PresentGenerated(Frame& frame) {
		if (generation_off) return false;
		const auto overlay_visual = GetSystemOverlayVisualState();
		if (swapchain.SetHdr(frame.hdr)) RecoverSwapchain(Swapchain::Status::Recreate);
		if (!ConfigureGenerationImages()) {
			generation_off = true;
			return false;
		}
		auto*      hud    = UpdateHud(false);
		const auto extent = swapchain.Extent();
		const auto format = swapchain.Format();

		uint32_t count = 0;
		{
			// (The renderer's lock: frame generation's inputs and HUD-less images are the renderer's, as on Windows.)
			RenderLockGuard render_lock(renderer.GetMutex());
			auto&           command = present_scheduler.BeginCommand();
			const bool      draw_system_overlay =
			    (overlay_visual.active || hud != nullptr) && swapchain.PrepareSystemOverlay(hud);
			if (hud == nullptr || !draw_system_overlay) {
				FrameGen::SetStaticRect(0.0f, 0.0f, 0.0f, 0.0f); // (no panel: the whole picture moves)
			} else {
				// The panel's motion vectors are zeroed (a margin around it).
				const auto& region = hud->region;
				const float w = static_cast<float>(region.extent.width), h = static_cast<float>(region.extent.height);
				const float margin = 8.0f;
				FrameGen::SetStaticRect(
				    (static_cast<float>(hud->drawn.offset.x - region.offset.x) - margin) / w,
				    (static_cast<float>(hud->drawn.offset.y - region.offset.y) - margin) / h,
				    (static_cast<float>(hud->drawn.offset.x - region.offset.x) + static_cast<float>(hud->drawn.extent.width) + margin) / w,
				    (static_cast<float>(hud->drawn.offset.y - region.offset.y) + static_cast<float>(hud->drawn.extent.height) + margin) / h);
			}
			// (With the frame's inputs FSR 3 built in gets its HUD-less image too: FrameGen::RecordHudless.)
			swapchain.RecordPresentCommands(command, frame.image, frame.view, draw_system_overlay, frame.scene, frame.menu,
			                                frame.fg.valid && !frame.reusing_last ? &frame.fg : nullptr, &composed,
			                                composed_view);
			if (FrameGen::HudlessOn() && !frame.reusing_last) {
				FrameGen::ScaleHudless(window.graphic_ctx, command.Handle(), frame.fg, extent, swapchain.PresentRegion(),
				                       format);
			}
			std::array<FrameGen::UpscaleImage, 3> outputs {};
			for (size_t i = 0; i < generated.size(); ++i) {
				auto&                   image = generated[i];
				vk::ImageMemoryBarrier2 barrier {};
				barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
				barrier.srcAccessMask       = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
				barrier.dstStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
				barrier.dstAccessMask       = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
				barrier.oldLayout           = image.state.layout;
				barrier.newLayout           = vk::ImageLayout::eGeneral;
				barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barrier.image               = image.image;
				barrier.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
				vk::DependencyInfo dependency {};
				dependency.imageMemoryBarrierCount = 1;
				dependency.pImageMemoryBarriers    = &barrier;
				command.Handle().pipelineBarrier2(dependency);
				image.state = {vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderWrite, vk::ImageLayout::eGeneral};
				outputs[i]  = {image.image, generated_views[i], vk::ImageLayout::eGeneral, format, extent, image.usage, 0, 0};
			}
			const FrameGen::UpscaleImage backbuffer {composed.image, composed_view, vk::ImageLayout::eGeneral, format, extent,
			                                         composed.usage, 0, 0};
			count = FrameGen::Generate(window.graphic_ctx, command.Handle(), frame.fg.valid && !frame.reusing_last ? &frame.fg : nullptr,
			                           backbuffer, swapchain.PresentRegion(), outputs, frame.hdr);
			// (A live `screenshot`: the finished frame and the frame generated before it.)
			const auto shot      = Screenshot::Take();
			auto       capture   = Screenshot::Record(window.graphic_ctx, command.Handle(), composed, shot);
			auto       generated_capture =
			    count > 0 && !shot.empty() && !Screenshot::IsSeriesFrame(shot)
			        ? Screenshot::Record(window.graphic_ctx, command.Handle(), generated[0], shot + ".gen.ppm")
			        : Screenshot::Pending {};
			frame.present_tick = present_scheduler.Submit();
			if (capture.buffer != VK_NULL_HANDLE || generated_capture.buffer != VK_NULL_HANDLE) {
				present_scheduler.Wait(frame.present_tick);
				Screenshot::Finish(window.graphic_ctx, capture);
				Screenshot::Finish(window.graphic_ctx, generated_capture);
			}
		}

		// The game's frame time (the time between its frames here), smoothed; long gaps (a load, a pause) left out.
		const auto now = std::chrono::steady_clock::now();
		if (last_game_frame != std::chrono::steady_clock::time_point {}) {
			const double ms = std::chrono::duration<double, std::milli>(now - last_game_frame).count();
			if (ms > 2.0 && ms < 100.0) game_frame_ms = game_frame_ms <= 0.0 ? ms : game_frame_ms * 0.8 + ms * 0.2;
		}
		last_game_frame = now;
		const auto spacing = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
		    std::chrono::duration<double, std::milli>(game_frame_ms / static_cast<double>(count + 1)));
		// A newer game frame already waiting: the generated frames left are not shown (the game never waits for them).
		const auto behind = [&] {
			std::lock_guard lock(waiting_mutex);
			return !waiting.Empty();
		};
		const auto show = [&](const VulkanImage& image) {
			auto status = swapchain.AcquireNextImage();
			if (status != Swapchain::Status::Success) {
				RecoverSwapchain(status);
				return false;
			}
			auto& command = present_scheduler.BeginCommand();
			swapchain.RecordShow(command, image);
			(void)swapchain.Submit(present_scheduler);
			status = swapchain.Present();
			if (status != Swapchain::Status::Success) {
				RecoverSwapchain(status);
				return false;
			}
			return true;
		};
		const auto start = std::chrono::steady_clock::now();
		uint32_t   shown = 0;
		bool       ok    = true;
		for (uint32_t i = 0; ok && i < count; ++i) {
			if (i > 0) {
				if (behind()) break;
				std::this_thread::sleep_until(start + spacing * i);
			}
			ok = show(generated[i]);
			shown += ok ? 1u : 0u;
		}
		if (ok) {
			if (shown > 0 && !behind()) std::this_thread::sleep_until(start + spacing * shown);
			AntiLag::Present(frame.anti_lag_frame);
			ok = show(composed);
		}
		FrameGen::NotePresented(shown + 1);
		presented_overlay_revision.store(overlay_visual.revision, std::memory_order_release);
		window.UpdateTitle(shown + (ok ? 1u : 0u),
		                   FsrNative::InterpolationRequested() || FrameGen::UsesFsr() ? "FSR 3 FG" : "DLSS-G");
		frames.Release(&frame, true);
		return true;
	}
#endif

#if defined(_WIN32)
	// Presents faster than the display refreshes only pile up in the compositor, and on Windows
	// a present blocked there holds the driver's device lock (see Swapchain::Present).
	std::chrono::steady_clock::time_point last_present {};
	std::chrono::nanoseconds              min_present_interval {-1}; // -1: not queried yet

	// When PresentDue lets the next present go ahead.
	[[nodiscard]] std::chrono::steady_clock::time_point NextPresentDue() {
		(void)PresentDue(false); // (the display's refresh, queried once)
		return min_present_interval.count() > 0 ? last_present + min_present_interval * 3 / 4
		                                         : std::chrono::steady_clock::time_point {};
	}

	// `commit`: the present goes ahead now (the next one is timed from it).
	[[nodiscard]] bool PresentDue(bool commit = true) {
		if (min_present_interval.count() < 0) {
			SDL_DisplayMode mode {};
			const int       display = window.window != nullptr ? SDL_GetWindowDisplayIndex(window.window) : -1;
			const int       refresh =
			    display >= 0 && SDL_GetCurrentDisplayMode(display, &mode) == 0 ? mode.refresh_rate : 0;
			min_present_interval = std::chrono::nanoseconds(refresh > 0 ? 1000000000LL / refresh : 0);
			// (In release logs too: a refresh read wrong limits what is shown, the game's own frame rate unchanged; #129.)
			std::printf("Presents limited to the display refresh: %d Hz\n", refresh);
			std::fflush(stdout);
		}
		const auto now = std::chrono::steady_clock::now();
		// A quarter of the interval as slack: a 60 Hz guest on a 60 Hz display flips at the same
		// rate, and with jitter every third present came a little early (40 fps shown).
		if (min_present_interval.count() > 0 && now - last_present < min_present_interval * 3 / 4) return false;
		if (commit) last_present = now;
		return true;
	}
#endif
};

// --hdr: an HDR10 (ST 2084 PQ) surface format the presentation blit can write. The swapchain uses it
// while the guest's frames are PQ; SDR frames (and before the first PQ frame) keep an SDR swapchain.
vk::SurfaceFormatKHR Swapchain::SelectFormat() {
	const auto& formats    = m_window.surface_capabilities.formats;
	auto&       graphics   = m_window.graphic_ctx;
	const auto  hdr_format = std::find_if(formats.begin(), formats.end(), [&](const vk::SurfaceFormatKHR& candidate) {
		return candidate.colorSpace == vk::ColorSpaceKHR::eHdr10St2084EXT && IsHdrSwapchainFormat(candidate.format) &&
		       static_cast<bool>(graphics.GetFormatProperties(candidate.format).optimalTilingFeatures &
		                         vk::FormatFeatureFlagBits::eBlitDst);
	});
	const bool supported = Config::HdrEnabled() && hdr_format != formats.end();
	if (Config::HdrEnabled() && supported != m_hdr_supported.load(std::memory_order_relaxed)) {
		std::printf("HDR10 output: %s\n", supported ? "available" : "the display offers no HDR10 surface (SDR output)");
		std::fflush(stdout);
	}
	m_hdr_supported.store(supported, std::memory_order_relaxed);
	m_hdr_active = m_hdr_requested && supported;
	if (m_hdr_active) {
		return *hdr_format;
	}
	if (formats.size() == 1 && formats.front().format == vk::Format::eUndefined) {
		return {vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
	}
	// (Linux frame generation: first a format NGX can write its frames in, a storage image, PresentGenerated.)
	for (int pass = FrameGen::PresenterGenerates() ? 0 : 1; pass < 2; ++pass) {
		for (const auto& candidate: formats) {
			if (candidate.colorSpace != vk::ColorSpaceKHR::eSrgbNonlinear) {
				continue;
			}
			if (pass == 0 && !(graphics.GetFormatProperties(candidate.format).optimalTilingFeatures &
			                   vk::FormatFeatureFlagBits::eStorageImage)) {
				continue;
			}
			if (candidate.format == vk::Format::eB8G8R8A8Unorm || candidate.format == vk::Format::eR8G8B8A8Unorm) {
				return candidate;
			}
		}
	}
	EXIT("no supported UNORM swapchain format\n");
}

void Swapchain::Create() {
	auto& graphics = m_window.graphic_ctx;
	EXIT_IF(graphics.device == nullptr);
	EXIT_IF(m_window.surface == nullptr);

	Common::LockGuard lock(m_window.mutex);
	EXIT_IF(graphics.screen_width == 0);
	EXIT_IF(graphics.screen_height == 0);
	const auto&       surface = m_window.surface_capabilities;
	EXIT_NOT_IMPLEMENTED(surface.formats.empty());

	m_extent = surface.capabilities.currentExtent;
	if (m_extent.width == std::numeric_limits<uint32_t>::max()) {
		m_extent.width =
		    std::clamp(graphics.screen_width, surface.capabilities.minImageExtent.width,
		               surface.capabilities.maxImageExtent.width);
		m_extent.height =
		    std::clamp(graphics.screen_height, surface.capabilities.minImageExtent.height,
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

	const auto format             = SelectFormat();
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
	// (FSR's HUD-less image is a copy of the backbuffer: FrameGen::RecordHudless.)
	if (FrameGen::OwnsSwapchain()) create_info.imageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
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
	m_fifo.store(create_info.presentMode == vk::PresentModeKHR::eFifo, std::memory_order_relaxed);
	create_info.clipped          = VK_TRUE;
#if defined(_WIN32)
	// KYTY_FULLSCREEN_EXCLUSIVE (GraphicContext::full_screen_exclusive, issue #166): exclusive fullscreen disallowed or
	// allowed. (VkSurfaceFullScreenExclusiveInfoEXT is vulkan_win32.h's, not included here: its layout.) The AMD default
	// stays out of FSR's frame generation swapchain, which makes its own from this description.
	struct {
		VkStructureType sType                 = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT;
		const void*     pNext                 = nullptr;
		int32_t         full_screen_exclusive = 0; // VkFullScreenExclusiveEXT: 1 allowed, 2 disallowed
	} exclusive;
	if (graphics.full_screen_exclusive >= 0 && !(graphics.full_screen_exclusive_default && FrameGen::OwnsSwapchain())) {
		exclusive.pNext                 = create_info.pNext;
		exclusive.full_screen_exclusive = graphics.full_screen_exclusive == 0 ? 2 : 1;
		create_info.pNext               = &exclusive;
	}
#endif
	// KYTY_FRAMEGEN_BACKEND=fsr: FSR's frame generation swapchain, made and used through its own functions.
	if (FrameGen::OwnsSwapchain()) {
		RequireVulkanSuccess(FrameGen::CreateSwapchain(graphics, create_info, &m_handle), "FSR frame generation swapchain");
	} else {
		RequireVulkanSuccess(graphics.device.createSwapchainKHR(&create_info, nullptr, &m_handle),
		                     "vkCreateSwapchainKHR");
	}
	EXIT_IF(m_handle == nullptr);
	if (Config::HdrEnabled()) {
		std::printf("Vulkan presentation: %s\n", m_hdr_active ? "HDR10 (PQ)" : "SDR");
		std::fflush(stdout);
	}

	m_images = EnumerateVulkan<vk::Image>(
	    "vkGetSwapchainImagesKHR", [&](uint32_t* count, vk::Image* images) {
		    return FrameGen::OwnsSwapchain() ? FrameGen::GetSwapchainImages(graphics.device, m_handle, count, images)
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
	auto& device = m_window.graphic_ctx.device;
	for (const auto pipeline: m_dither_pipelines) {
		if (pipeline != nullptr) device.destroyPipeline(pipeline, nullptr);
	}
	if (m_dither_layout != nullptr) device.destroyPipelineLayout(m_dither_layout, nullptr);
	if (m_dither_set_layout != nullptr) device.destroyDescriptorSetLayout(m_dither_set_layout, nullptr);
	if (m_dither_sampler != nullptr) device.destroySampler(m_dither_sampler, nullptr);
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
	if (graphics.present_queue != nullptr) {
		Common::LockGuard queue_lock(graphics.present_queue_mutex);
		RequireVulkanSuccess(graphics.present_queue.waitIdle(), "wait for present queue");
	}
	if (m_system_overlay != nullptr) {
		m_system_overlay->ReleaseVulkan();
	}
	// FSR 3's frame generation swapchain presents from its own thread and queues, which the waits above do not cover:
	// its frames may still wait on the semaphores and read the images' views below. Its context goes first (its
	// destruction waits for its own work), then those. (An RTX 2060 SUPER on a 165 Hz display lost the device leaving
	// full screen with F11, issue #174; never on an RTX 5090 at 60 Hz.)
	if (m_handle != nullptr && FrameGen::OwnsSwapchain()) {
		FrameGen::DestroySwapchain(graphics, m_handle);
		m_handle = nullptr;
	}

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
		if (FrameGen::OwnsSwapchain()) {
			FrameGen::DestroySwapchain(graphics, m_handle);
		} else {
			graphics.device.destroySwapchainKHR(m_handle, nullptr);
		}
	}

	m_handle      = nullptr;
	m_format      = vk::Format::eUndefined;
	m_extent      = {};
	m_image_index = static_cast<uint32_t>(-1);
	m_frame_index = 0;
	m_images.clear();
	m_image_views.clear();
	m_image_acquired.clear();
	m_render_complete.clear();
}

void Swapchain::Recreate(bool surface_lost) {
	Destroy();
	if (surface_lost) {
#if defined(__APPLE__)
		// Surface recreation goes through SDL_Vulkan_CreateSurface, which touches the
		// window's view/layer and must run on the main thread on macOS.
		m_window.RunOnMainThread([this] { m_window.RecreateSurface(); });
#else
		m_window.RecreateSurface();
#endif
	}
	m_window.RefreshSurfaceCapabilities();
	Create();
}

Swapchain::Status Swapchain::AcquireNextImage() {
	EXIT_IF(m_handle == nullptr || m_frame_index >= m_image_acquired.size());
	m_image_index     = static_cast<uint32_t>(-1);
	const auto result =
	    FrameGen::OwnsSwapchain()
	        ? FrameGen::AcquireNextImage(m_window.graphic_ctx.device, m_handle, std::numeric_limits<uint64_t>::max(),
	                                     m_image_acquired[m_frame_index], &m_image_index)
	        : m_window.graphic_ctx.device.acquireNextImageKHR(m_handle, std::numeric_limits<uint64_t>::max(),
	                                                          m_image_acquired[m_frame_index], nullptr, &m_image_index);
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

bool Swapchain::PrepareSystemOverlay(SystemOverlayHud* hud) {
	if (m_system_overlay == nullptr) {
		m_system_overlay = std::make_unique<SystemOverlay>(m_window.graphic_ctx);
	}
	return m_system_overlay->PrepareFrame(m_extent, m_format, ImageCount(), hud);
}

// KYTY_PRESENT_DITHER=0: a 10-bit frame blitted into an 8-bit swapchain image as before, each pixel rounded (issue #234).
static bool PresentDither() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_PRESENT_DITHER");
		return value == nullptr || value[0] != '0';
	}();
	return on;
}

bool Swapchain::RecordDitheredCopy(vk::CommandBuffer command, VulkanImage& source, vk::ImageView source_view,
                                   vk::Image dst_image, vk::ImageView dst_view, uint32_t src_x, uint32_t src_w) {
	// Where the blit rounds away bits the frame has: a 10-bit (or half-float) frame into an 8-bit image (not HDR's 10-bit).
	const bool deep = source.format == vk::Format::eA2B10G10R10UnormPack32 ||
	                  source.format == vk::Format::eA2R10G10B10UnormPack32 ||
	                  source.format == vk::Format::eR16G16B16A16Sfloat;
	if (!deep || (m_format != vk::Format::eB8G8R8A8Unorm && m_format != vk::Format::eR8G8B8A8Unorm) ||
	    source_view == nullptr || dst_view == nullptr || m_dither_failed || !PresentDither()) {
		return false;
	}
	auto& device = m_window.graphic_ctx.device;
	auto  fail   = [this](const char* what) {
		std::printf("Presenter: no dithered copy (%s failed): frames blitted, each pixel rounded\n", what);
		std::fflush(stdout);
		m_dither_failed = true;
		return false;
	};
	if (m_dither_layout == nullptr) {
		vk::SamplerCreateInfo sampler {};
		sampler.magFilter    = vk::Filter::eLinear; // (as the blit scaled)
		sampler.minFilter    = vk::Filter::eLinear;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		if (device.createSampler(&sampler, nullptr, &m_dither_sampler) != vk::Result::eSuccess) return fail("sampler");
		vk::DescriptorSetLayoutBinding binding {};
		binding.binding            = 0;
		binding.descriptorType     = vk::DescriptorType::eCombinedImageSampler;
		binding.descriptorCount    = 1;
		binding.stageFlags         = vk::ShaderStageFlagBits::eFragment;
		binding.pImmutableSamplers = &m_dither_sampler;
		vk::DescriptorSetLayoutCreateInfo set_layout {};
		set_layout.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		set_layout.bindingCount = 1;
		set_layout.pBindings    = &binding;
		if (device.createDescriptorSetLayout(&set_layout, nullptr, &m_dither_set_layout) != vk::Result::eSuccess)
			return fail("descriptor set layout");
		vk::PushConstantRange range {};
		range.stageFlags = vk::ShaderStageFlagBits::eFragment;
		range.size       = 4 * sizeof(float);
		vk::PipelineLayoutCreateInfo layout {};
		layout.setLayoutCount         = 1;
		layout.pSetLayouts            = &m_dither_set_layout;
		layout.pushConstantRangeCount = 1;
		layout.pPushConstantRanges    = &range;
		if (device.createPipelineLayout(&layout, nullptr, &m_dither_layout) != vk::Result::eSuccess)
			return fail("pipeline layout");
	}
	auto& pipeline = m_dither_pipelines[m_format == vk::Format::eR8G8B8A8Unorm ? 1 : 0];
	if (pipeline == nullptr) {
		std::array<vk::ShaderModule, 2> modules {};
		const std::array<std::pair<const uint32_t*, size_t>, 2> code {
		    std::pair {GPU_BLIT_FS_TRIANGLE_SPV, std::size(GPU_BLIT_FS_TRIANGLE_SPV)},
		    std::pair {PRESENT_DITHER_SPV, std::size(PRESENT_DITHER_SPV)}};
		bool made = true;
		for (size_t i = 0; i < modules.size() && made; ++i) {
			vk::ShaderModuleCreateInfo create {};
			create.codeSize = code[i].second * sizeof(uint32_t);
			create.pCode    = code[i].first;
			made            = device.createShaderModule(&create, nullptr, &modules[i]) == vk::Result::eSuccess;
		}
		if (made) {
			std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
			stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
			stages[0].module = modules[0];
			stages[0].pName  = "main";
			stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
			stages[1].module = modules[1];
			stages[1].pName  = "main";
			vk::PipelineVertexInputStateCreateInfo   vertex_input {};
			vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
			input_assembly.topology = vk::PrimitiveTopology::eTriangleList;
			vk::PipelineViewportStateCreateInfo viewport {};
			viewport.viewportCount = 1;
			viewport.scissorCount  = 1;
			vk::PipelineRasterizationStateCreateInfo rasterization {};
			rasterization.lineWidth = 1.0f;
			vk::PipelineMultisampleStateCreateInfo multisample {};
			multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
			vk::PipelineColorBlendAttachmentState blend {};
			blend.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
			                       vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
			vk::PipelineColorBlendStateCreateInfo color_blend {};
			color_blend.attachmentCount = 1;
			color_blend.pAttachments    = &blend;
			const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
			vk::PipelineDynamicStateCreateInfo dynamic {};
			dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
			dynamic.pDynamicStates    = dynamic_states.data();
			vk::PipelineRenderingCreateInfo rendering {};
			rendering.colorAttachmentCount    = 1;
			rendering.pColorAttachmentFormats = &m_format;
			vk::GraphicsPipelineCreateInfo create {};
			create.pNext               = &rendering;
			create.stageCount          = static_cast<uint32_t>(stages.size());
			create.pStages             = stages.data();
			create.pVertexInputState   = &vertex_input;
			create.pInputAssemblyState = &input_assembly;
			create.pViewportState      = &viewport;
			create.pRasterizationState = &rasterization;
			create.pMultisampleState   = &multisample;
			create.pColorBlendState    = &color_blend;
			create.pDynamicState       = &dynamic;
			create.layout              = m_dither_layout;
			made = device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &pipeline) == vk::Result::eSuccess;
		}
		for (const auto module: modules) {
			if (module != nullptr) device.destroyShaderModule(module, nullptr);
		}
		if (!made) {
			pipeline = nullptr;
			return fail("pipeline");
		}
		std::printf("Presenter: 10-bit frames dithered into the 8-bit swapchain image (KYTY_PRESENT_DITHER=0: rounded)\n");
		std::fflush(stdout);
	}

	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	auto barrier = [&range](vk::Image image, vk::ImageLayout from, vk::ImageLayout to, vk::AccessFlags src_access,
	                        vk::AccessFlags dst_access) {
		vk::ImageMemoryBarrier b {};
		b.srcAccessMask       = src_access;
		b.dstAccessMask       = dst_access;
		b.oldLayout           = from;
		b.newLayout           = to;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image               = image;
		b.subresourceRange    = range;
		return b;
	};
	// The frame sampled; the swapchain image (its bars cleared by now) drawn on.
	const std::array before {
	    barrier(source.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, {},
	            vk::AccessFlagBits::eShaderRead),
	    barrier(dst_image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eColorAttachmentOptimal,
	            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eColorAttachmentWrite)};
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                        vk::DependencyFlags {}, 0, nullptr, 0, nullptr, static_cast<uint32_t>(before.size()),
	                        before.data());
	vk::RenderingAttachmentInfo color {};
	color.imageView   = dst_view;
	color.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	color.loadOp      = vk::AttachmentLoadOp::eDontCare; // (every pixel of the region drawn)
	color.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo rendering {};
	rendering.renderArea           = m_present_region;
	rendering.layerCount           = 1;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &color;
	command.beginRendering(rendering);
	const vk::Viewport viewport {static_cast<float>(m_present_region.offset.x), static_cast<float>(m_present_region.offset.y),
	                             static_cast<float>(m_present_region.extent.width),
	                             static_cast<float>(m_present_region.extent.height), 0.0f, 1.0f};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &m_present_region);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	vk::DescriptorImageInfo sampled {};
	sampled.imageView   = source_view;
	sampled.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	vk::WriteDescriptorSet write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
	write.pImageInfo      = &sampled;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_dither_layout, 0, 1, &write);
	// present_dither.frag: the region's top left, and the frame's texture coordinates per pixel (its `src_w` columns);
	// the frame's columns from `src_x` on: the region's left edge as if src_x of them came before it.
	const std::array<float, 4> constants {viewport.x - static_cast<float>(src_x) * viewport.width / static_cast<float>(src_w),
	                                      viewport.y,
	                                      static_cast<float>(src_w) / static_cast<float>(source.extent.width) / viewport.width,
	                                      1.0f / viewport.height};
	command.pushConstants(m_dither_layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(constants), constants.data());
	command.draw(3, 1, 0, 0);
	command.endRendering();
	// Both left as the blit leaves them: the frame a transfer source, the swapchain image a transfer destination (for
	// the HUD-less copy and the barriers after).
	const std::array after {
	    barrier(source.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal, {},
	            vk::AccessFlagBits::eTransferRead),
	    barrier(dst_image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferDstOptimal,
	            vk::AccessFlagBits::eColorAttachmentWrite,
	            vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)};
	command.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                        vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags {}, 0, nullptr, 0, nullptr,
	                        static_cast<uint32_t>(after.size()), after.data());
	return true;
}

void Swapchain::RecordPresentCommands(CommandBuffer& command, VulkanImage& source, vk::ImageView source_view,
                                      bool draw_system_overlay, bool scene, bool menu,
                                      const FrameGen::FrameInputs* fsr_inputs, VulkanImage* target,
                                      vk::ImageView target_view) {
	if (source.state.layout != vk::ImageLayout::eTransferSrcOptimal) {
		EXIT("invalid prepared presentation image, vk_image=%p layout=%d\n",
		     static_cast<void*>(source.image), static_cast<int>(source.state.layout));
	}
	EXIT_IF(target == nullptr && m_image_index >= m_images.size());
	auto vk_command = command.Handle();
	// (Linux frame generation: into `target`, left general for NGX and the copies to the swapchain, RecordShow.)
	const vk::Image       dst_image    = target != nullptr ? target->image : m_images[m_image_index];
	const vk::ImageView   dst_view     = target != nullptr ? target_view : m_image_views[m_image_index];
	const vk::ImageLayout final_layout = target != nullptr ? vk::ImageLayout::eGeneral : vk::ImageLayout::ePresentSrcKHR;

	vk::ImageMemoryBarrier to_transfer {};
	to_transfer.sType                           = vk::StructureType::eImageMemoryBarrier;
	to_transfer.dstAccessMask                   = vk::AccessFlagBits::eTransferWrite;
	to_transfer.oldLayout                       = vk::ImageLayout::eUndefined;
	to_transfer.newLayout                       = vk::ImageLayout::eTransferDstOptimal;
	to_transfer.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.image                           = dst_image;
	to_transfer.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_transfer.subresourceRange.baseMipLevel   = 0;
	to_transfer.subresourceRange.levelCount     = 1;
	to_transfer.subresourceRange.baseArrayLayer = 0;
	to_transfer.subresourceRange.layerCount     = 1;
	// Match the acquire wait stage so the layout transition cannot precede acquisition.
	vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                           vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags {}, 0,
	                           nullptr, 0, nullptr, 1, &to_transfer);

	// KYTY_PRESENT_ASPECT=fit: keep the frame's aspect ratio (black bars) instead of stretching
	// it over the window; a 3840x2160 frame on a 5120x2160 screen is then copied 1:1.
	static const bool aspect_fit = [] {
		const char* value = std::getenv("KYTY_PRESENT_ASPECT");
		return value != nullptr && std::string_view(value) == "fit";
	}();
	// KYTY_RENDER_ASPECT: the game's 3D was rendered wider and squeezed into its 16:9 output; every frame is shown at that
	// aspect, fitted to the window (within 2% of the window's aspect: filling it): the 3D as it was rendered, the others
	// (movies, menus, loading screens) stretched from their 16:9 (the user, 10-19: the opening movie filling the screen),
	// or with KYTY_PRESENT_ASPECT=fit kept at their 16:9 (black bars at the sides; issue #97).
	// With the wide output (DemonsSoulsRenderSize::WideOutputWidth) the game's frames are as wide as the 3D; one without
	// it (movies, menus, loading screens) holds its 16:9 picture at the left: that part is shown, stretched or fitted. A
	// menu over the 3D view with KYTY_PRESENT_ASPECT=fit (`menu`: hud-anchor put it at 16:9 in the middle) shows that
	// middle part alone at 16:9, between black bars (issue #250): the wide output's 16:9 part, or with the 3D squeezed
	// into the game's own 16:9 output the middle 16:9-over-the-aspect of it, widened back (the menu squeezed as much).
	int          dst_x = 0, dst_y = 0;
	int          dst_w = static_cast<int>(m_extent.width), dst_h = static_cast<int>(m_extent.height);
	uint32_t     src_x = 0, src_w = source.extent.width;
	const double wide        = Loader::DemonsSoulsRenderSize::WideAspect();
	const bool   wide_output = Loader::DemonsSoulsRenderSize::WideOutputWidth() != 0 && source.extent.height != 0;
	const bool   middle      = menu && aspect_fit && wide > 0.0 && source.extent.height != 0;
	if ((!scene || middle) && wide_output) {
		src_w = std::min(src_w, (source.extent.height * 16u + 8u) / 9u);
	} else if (middle) {
		src_w = std::clamp(static_cast<uint32_t>(std::lround(src_w * (16.0 / 9.0) / wide)), 1u, src_w);
	}
	if (middle) src_x = (source.extent.width - src_w) / 2;
	const double frame_aspect = middle                     ? 16.0 / 9.0
	                           : source.extent.height != 0 ? static_cast<double>(src_w) / source.extent.height
	                                                       : 0.0;
	double       aspect       = wide > 0.0 && ((scene && !middle) || !aspect_fit) ? wide : aspect_fit ? frame_aspect : 0.0;
	if (const double window = static_cast<double>(m_extent.width) / std::max(1u, m_extent.height);
	    wide > 0.0 && std::fabs(aspect / window - 1.0) < 0.02) {
		aspect = 0.0;
	}
	if (aspect > 0.0 && source.extent.width != 0 && source.extent.height != 0) {
		dst_w = std::max(1, static_cast<int>(std::lround(std::min<double>(m_extent.width, m_extent.height * aspect))));
		dst_h = std::max(1, static_cast<int>(std::lround(std::min<double>(m_extent.height, m_extent.width / aspect))));
		dst_x = (static_cast<int>(m_extent.width) - dst_w) / 2;
		dst_y = (static_cast<int>(m_extent.height) - dst_h) / 2;
		if (dst_w != static_cast<int>(m_extent.width) || dst_h != static_cast<int>(m_extent.height)) {
			const vk::ClearColorValue black {std::array<float, 4> {0.0f, 0.0f, 0.0f, 1.0f}};
			vk_command.clearColorImage(dst_image, vk::ImageLayout::eTransferDstOptimal, &black, 1,
			                           &to_transfer.subresourceRange);
			// (An image barrier: <windows.h> defines MemoryBarrier as a macro.)
			auto cleared          = to_transfer;
			cleared.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			cleared.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
			cleared.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
			cleared.newLayout     = vk::ImageLayout::eTransferDstOptimal;
			vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
			                           vk::DependencyFlags {}, 0, nullptr, 0, nullptr, 1, &cleared);
		}
	}

	m_present_region = vk::Rect2D {{dst_x, dst_y}, {static_cast<uint32_t>(dst_w), static_cast<uint32_t>(dst_h)}};
	vk::ImageBlit region {};
	region.srcSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
	region.srcSubresource.mipLevel       = 0;
	region.srcSubresource.baseArrayLayer = 0;
	region.srcSubresource.layerCount     = 1;
	region.srcOffsets[0].x               = static_cast<int>(src_x);
	region.srcOffsets[1].x               = static_cast<int>(src_x + src_w);
	region.srcOffsets[1].y               = static_cast<int>(source.extent.height);
	region.srcOffsets[1].z               = 1;
	region.dstSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
	region.dstSubresource.mipLevel       = 0;
	region.dstSubresource.baseArrayLayer = 0;
	region.dstSubresource.layerCount     = 1;
	region.dstOffsets[0].x               = dst_x;
	region.dstOffsets[0].y               = dst_y;
	region.dstOffsets[1].x               = dst_x + dst_w;
	region.dstOffsets[1].y               = dst_y + dst_h;
	region.dstOffsets[1].z               = 1;
	if (!RecordDitheredCopy(vk_command, source, source_view, dst_image, dst_view, src_x, src_w)) {
		vk_command.blitImage(source.image, vk::ImageLayout::eTransferSrcOptimal, dst_image,
		                     vk::ImageLayout::eTransferDstOptimal, 1, &region, vk::Filter::eLinear);
	}
	// KYTY_FRAMEGEN_BACKEND=fsr: the HUD-less image from this, before the system overlay (which counts as HUD then); the
	// Windows backend's of the swapchain image, FSR 3 built in's of the composed frame (`target`).
	if (fsr_inputs != nullptr && (FrameGen::OwnsSwapchain() ? target == nullptr : target != nullptr)) {
		FrameGen::RecordHudless(m_window.graphic_ctx, vk_command, fsr_inputs, dst_image, m_extent,
		                        m_present_region, m_format);
	}

	vk::ImageMemoryBarrier to_present {};
	to_present.sType         = vk::StructureType::eImageMemoryBarrier;
	to_present.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	to_present.dstAccessMask = draw_system_overlay ? vk::AccessFlagBits::eColorAttachmentRead |
	                                                     vk::AccessFlagBits::eColorAttachmentWrite
	                                               : vk::AccessFlagBits::eMemoryRead;
	to_present.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
	to_present.newLayout     = draw_system_overlay ? vk::ImageLayout::eColorAttachmentOptimal
	                                               : final_layout;
	to_present.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_present.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_present.image                           = dst_image;
	to_present.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_present.subresourceRange.baseMipLevel   = 0;
	to_present.subresourceRange.levelCount     = 1;
	to_present.subresourceRange.baseArrayLayer = 0;
	to_present.subresourceRange.layerCount     = 1;
	vk_command.pipelineBarrier(
	    vk::PipelineStageFlagBits::eTransfer,
	    draw_system_overlay ? vk::PipelineStageFlagBits::eColorAttachmentOutput
	                        : vk::PipelineStageFlagBits::eAllCommands,
	    vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1, &to_present);
	if (draw_system_overlay) {
		m_system_overlay->Record(vk_command, dst_view);
		to_present.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
		to_present.dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		to_present.oldLayout     = vk::ImageLayout::eColorAttachmentOptimal;
		to_present.newLayout     = final_layout;
		vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
		                           vk::PipelineStageFlagBits::eAllCommands,
		                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
		                           &to_present);
	}
	if (target != nullptr) {
		target->state = {vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead, vk::ImageLayout::eGeneral};
		target->subresource_states.clear();
	}
}

void Swapchain::RecordShow(CommandBuffer& command, const VulkanImage& source) {
	EXIT_IF(m_image_index >= m_images.size());
	auto vk_command = command.Handle();
	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	// The source as the earlier submissions left it (the composition, NGX's writes), the swapchain image for the copy
	// (after the acquire: its wait is at the transfer stage, Submit).
	std::array<vk::ImageMemoryBarrier, 2> before {};
	before[0].srcAccessMask       = vk::AccessFlagBits::eMemoryWrite;
	before[0].dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before[0].oldLayout           = vk::ImageLayout::eGeneral;
	before[0].newLayout           = vk::ImageLayout::eGeneral;
	before[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before[0].image               = source.image;
	before[0].subresourceRange    = range;
	before[1].dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
	before[1].oldLayout           = vk::ImageLayout::eUndefined;
	before[1].newLayout           = vk::ImageLayout::eTransferDstOptimal;
	before[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before[1].image               = m_images[m_image_index];
	before[1].subresourceRange    = range;
	vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
	                           vk::DependencyFlags {}, 0, nullptr, 0, nullptr, static_cast<uint32_t>(before.size()),
	                           before.data());
	vk::ImageCopy copy {};
	copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.extent         = {std::min(source.extent.width, m_extent.width), std::min(source.extent.height, m_extent.height), 1};
	vk_command.copyImage(source.image, vk::ImageLayout::eGeneral, m_images[m_image_index],
	                     vk::ImageLayout::eTransferDstOptimal, copy);
	vk::ImageMemoryBarrier to_present {};
	to_present.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	to_present.dstAccessMask       = vk::AccessFlagBits::eMemoryRead;
	to_present.oldLayout           = vk::ImageLayout::eTransferDstOptimal;
	to_present.newLayout           = vk::ImageLayout::ePresentSrcKHR;
	to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_present.image               = m_images[m_image_index];
	to_present.subresourceRange    = range;
	vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
	                           vk::DependencyFlags {}, 0, nullptr, 0, nullptr, 1, &to_present);
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
	if (auto& graphics = m_window.graphic_ctx; FrameGen::OwnsSwapchain() && graphics.present_queue != nullptr) {
		// FSR's swapchain: its game queue is the present queue (FrameGen::CreateSwapchain).
		Common::LockGuard lock(graphics.present_queue_mutex);
		result = FrameGen::QueuePresent(graphics.present_queue, present);
	} else if (FrameGen::OwnsSwapchain()) {
		// ... or the graphics queue, without a present queue (issue #170: one graphics queue).
		Common::LockGuard lock(graphics.queue_mutex);
		result = FrameGen::QueuePresent(graphics.queue, present);
	} else if (graphics.present_queue != nullptr) {
		// Its own queue: a present blocked in the kernel does not hold up other submissions.
		Common::LockGuard lock(graphics.present_queue_mutex);
		result = graphics.present_queue.presentKHR(&present);
	} else {
		Common::LockGuard lock(graphics.queue_mutex);
		result = graphics.queue.presentKHR(&present);
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

Presenter::Presenter(WindowContext& window): m_impl(std::make_unique<Impl>(window)) {
	m_impl->thread = std::jthread([this](std::stop_token token) { PresentLoop(std::move(token)); });
}

Presenter::~Presenter() {
	// The presenter's thread ends before what it presents with (a frame still waiting is not shown).
	m_impl->thread.request_stop();
	if (m_impl->thread.joinable()) m_impl->thread.join();
}

Presenter::Frame& Presenter::PrepareFrame(CommandBuffer& buffer, const ImageInfo& info, bool hdr) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(buffer.IsInvalid());
	auto*             frame = m_impl->frames.Acquire();
	frame->hdr              = hdr;
	frame->blank            = false;
	frame->anti_lag_frame   = AntiLag::Flip();
	RenderLockGuard render_lock(m_impl->renderer.GetMutex());
	auto&             image = m_impl->ResolveSurface(info);
	if (image.backing.format == vk::Format::eUndefined) {
		EXIT("unsupported presentation source, image=%p\n", static_cast<const void*>(&image));
	}

	auto frame_format = info.pixel_format;
	switch (frame_format) {
		case vk::Format::eR8G8B8A8Srgb: frame_format = vk::Format::eR8G8B8A8Unorm; break;
		case vk::Format::eB8G8R8A8Srgb: frame_format = vk::Format::eB8G8R8A8Unorm; break;
		default: break;
	}
	frame->Configure(m_impl->window.graphic_ctx,
	                 {image.backing.extent.width, image.backing.extent.height}, frame_format);
	frame->CopyFrom(buffer, image);
	frame->scene    = FrameGen::TakeScene();
	frame->menu     = FrameGen::TakeMenu();
	frame->fg.valid = false;
	// (A menu shown alone at 16:9 gets no frames generated: the inputs are the whole wide output's.)
	if (FrameGen::Enabled() && !frame->menu) {
		FrameGen::PrepareInputs(m_impl->window.graphic_ctx, m_impl->renderer.GetTextureCache(),
		                        buffer, frame->fg);
		if (FrameGen::HudlessOn()) {
			FrameGen::PrepareHudless(frame->fg, image);
		}
	}
	return *frame;
}

Presenter::Frame& Presenter::PrepareBlankFrame(uint32_t width, uint32_t height, bool opaque,
                                               CommandBuffer* producer) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(producer == nullptr || producer->IsInvalid());
	auto*           frame = m_impl->frames.Acquire();
	RenderLockGuard render_lock(m_impl->renderer.GetMutex());
	m_impl->ClearFrame(*frame, width, height, opaque, producer);
	return *frame;
}

Presenter::Frame* Presenter::PrepareLastFrame() {
	return m_impl->frames.AcquireLast();
}

// KYTY_PRESENT_DIRECT=1 (a test for issue #37, a variable refresh rate that stopped engaging in v20261016b): each frame
// is presented on the vblank thread as it is handed over, as before the presenter had a thread of its own (eadf554f).
// Not with FSR 3's frame generation: the vblank thread then waits in its swapchain's presents (two a game frame), and the
// game flipped at half the display's rate (at 60 Hz: 30.0 fps, done to flip 32 ms, against 60.2 and 16 ms on the
// presenter's thread; issue #236: an RTX 4090 at 120 Hz held at 30 in every session). DLSS-G keeps it: there the game ran
// at the full rate with it (120 shown of 60 at 120 Hz), and VRR displays ran smoother (v20261031 had it off with any
// frame generation; issue #236).
static bool PresentDirect() {
	static const bool requested = [] {
		const char* value = std::getenv("KYTY_PRESENT_DIRECT");
		return value != nullptr && value[0] == '1';
	}();
	if (!requested) return false;
	if (FrameGen::UsesFsr() || FsrNative::InterpolationRequested()) {
		static std::atomic_flag reported = ATOMIC_FLAG_INIT;
		if (!reported.test_and_set()) {
			std::printf("Presentation: KYTY_PRESENT_DIRECT off with FSR 3 frame generation (it held the game at half the "
			            "display's rate; issue #236)\n");
			std::fflush(stdout);
		}
		return false;
	}
	return true;
}

void Presenter::Submit(Frame& frame, bool reuse) {
	m_impl->frames.ValidateForPresent(&frame, reuse);
	if (PresentDirect()) {
		Present(frame, reuse);
		return;
	}
	m_impl->Replace({.frame = &frame, .reuse = reuse});
}

void Presenter::SubmitBlank(uint32_t width, uint32_t height, bool opaque) {
	if (PresentDirect()) {
		// (A free frame, else none, as the presenter's thread does.)
		if (auto* frame = m_impl->frames.Acquire(false); frame != nullptr) {
			m_impl->ClearFrame(*frame, width, height, opaque, nullptr);
			Present(*frame, false);
		}
		return;
	}
	m_impl->Replace({.blank = true, .width = width, .height = height, .opaque = opaque});
}

bool Presenter::PresentIdle() {
	std::lock_guard lock(m_impl->waiting_mutex);
	return m_impl->waiting.Empty() && !m_impl->presenting;
}

void Presenter::PresentLoop(std::stop_token token) {
	LocalPlatform::SetThreadName("Kyty.Present");
	auto& impl = *m_impl;
	for (;;) {
		Impl::Waiting next;
		{
			std::unique_lock lock(impl.waiting_mutex);
			impl.presenting = false;
			if (!impl.waiting_changed.wait(lock, token, [&] { return !impl.waiting.Empty(); })) return;
#if defined(_WIN32)
			// Held until the display can take it (Impl::PresentDue), a newer one taking its place meanwhile. (A frame
			// DLSS-G generates frames for goes at once: it paces its presents itself.)
			for (;;) {
				const auto& w = impl.waiting;
				if (w.frame != nullptr && !w.reuse && w.frame->fg.valid && FrameGen::Enabled()) break;
				const auto due = impl.NextPresentDue();
				if (std::chrono::steady_clock::now() >= due) break;
				impl.waiting_changed.wait_until(lock, token, due, [] { return false; });
				if (token.stop_requested()) return;
			}
#endif
			next            = std::exchange(impl.waiting, {});
			impl.presenting = true;
		}
		impl.waiting_taken.notify_all();
		if (next.blank) {
			// A free frame, else none: each frame held is a guest picture on its way here.
			auto* frame = impl.frames.Acquire(false);
			if (frame == nullptr) continue;
			impl.ClearFrame(*frame, next.width, next.height, next.opaque, nullptr);
			Present(*frame, false);
		} else {
			Present(*next.frame, next.reuse);
		}
	}
}

uint32_t Presenter::DisplayVblanksPerTick(uint32_t vblank_hz) {
#if defined(_WIN32)
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_VSYNC_LOCK");
		return value == nullptr || std::atoi(value) != 0;
	}();
	if (!enabled || vblank_hz == 0 || !m_impl->swapchain.IsFifo() || m_impl->window.window == nullptr) return 0;
	SDL_SysWMinfo info {};
	SDL_VERSION(&info.version);
	if (SDL_GetWindowWMInfo(m_impl->window.window, &info) != SDL_TRUE || info.subsystem != SDL_SYSWM_WINDOWS) return 0;
	// (A 59.94 Hz mode reports 59.)
	const int  refresh = LocalPlatform::DisplayRefreshHz(info.info.win.window);
	const auto hz      = static_cast<int>(vblank_hz);
	const int  k       = static_cast<int>(std::lround(static_cast<double>(refresh) / hz));
	if (refresh <= 0 || k < 1 || std::abs(refresh - k * hz) > 1) return 0;
	return LocalPlatform::StartDisplayVblanks(info.info.win.window) ? static_cast<uint32_t>(k) : 0;
#else
	(void)vblank_hz;
	return 0;
#endif
}

bool Presenter::WaitForDisplayVblank(uint64_t& seen, uint32_t timeout_us) {
	return LocalPlatform::WaitDisplayVblank(seen, timeout_us);
}

bool Presenter::IsGuestPaused() const noexcept {
	return m_impl->window.loop.paused.load(std::memory_order_acquire);
}

bool Presenter::IsHdrSupported() const noexcept {
	return m_impl->swapchain.IsHdrSupported();
}

bool Presenter::NeedsSystemOverlayRefresh() const noexcept {
	const auto visual = GetSystemOverlayVisualState();
	return visual.active ||
	       visual.revision != m_impl->presented_overlay_revision.load(std::memory_order_acquire);
}

RenderContext& Presenter::Renderer() const noexcept {
	return m_impl->renderer;
}

bool Presenter::FlipRateApplies(const Frame& frame) noexcept {
	// The game's 3D view only (with frame generation, the frames carrying its inputs): a capped movie played at half speed.
	return FrameGen::Enabled() ? frame.fg.valid : frame.scene;
}

bool Presenter::IsSceneFrame(const Frame& frame) noexcept {
	return frame.scene;
}

bool Presenter::GeneratesFrames() noexcept {
	return FrameGen::Enabled();
}

void Presenter::Present(Frame& frame, bool reuse) {
	KYTY_PROFILER_FUNCTION();
	m_impl->frames.ValidateForPresent(&frame, reuse);
#if defined(__linux__)
	// Linux frame generation: a new frame, and those NGX makes before it (PresentGenerated). A cleared frame, or an HDR
	// one for DLSS-G (FSR 3 built in takes PQ frames too), goes the normal way, and generation starts again after it; a
	// last frame shown again (an idle refresh) is not new.
	if (FrameGen::PresenterGenerates() && !reuse) {
		if (!frame.blank && (!frame.hdr || FrameGen::PresenterGeneratesHdr()) && m_impl->PresentGenerated(frame)) return;
		FrameGen::SwapchainChanging();
	}
#endif
#if defined(_WIN32)
	// (A frame DLSS-G generates frames for goes to it whatever the display's rate: it paces its presents itself, and
	// held to the refresh it got 57 of a 102 fps game's frames a second on a 60 Hz display, 5x showing 285, not 500.)
	const bool generated = FrameGen::Enabled() && frame.fg.valid && !reuse;
	if (!generated && !m_impl->PresentDue()) {
		// (PresentLoop waited until it was due.)
		LiveCounters::Add(LiveCounters::PresentsNotDue);
		if (!reuse) m_impl->present_stats.not_due.fetch_add(1, std::memory_order_relaxed);
		m_impl->frames.Release(&frame, true);
		return;
	}
#endif
	// KYTY_FRAMEGEN_BACKEND=fsr: the frame's inputs for FSR (none for an idle refresh of the last frame); a frame
	// without them between generated frames is not shown (FrameGen::SkipsFrame).
	const FrameGen::FrameInputs* fsr_inputs = frame.fg.valid && !frame.reusing_last && !reuse ? &frame.fg : nullptr;
	if (FrameGen::OwnsSwapchain() && FrameGen::SkipsFrame(fsr_inputs)) {
		m_impl->frames.Release(&frame, true);
		return;
	}
#if defined(KYTY_LOCAL_VULKAN_RECORDING) && defined(_WIN32)
	// The frame's rendering may still sit in the recording worker's queue (KYTY_DEFERRED_SUBMIT).
	// On Windows, presenting before it reaches the driver blocks vkQueuePresentKHR in the
	// kernel while it holds the queue lock that submission needs: a deadlock. Wait (no locks
	// held) until it has been submitted; this does not wait for the GPU.
	LocalVulkanRecording::WaitDeferredSubmits();
#endif

	const auto overlay_visual = GetSystemOverlayVisualState();
	auto&      swapchain  = m_impl->swapchain;
	auto*      hud        = m_impl->UpdateHud(reuse);
	// The guest's own frames choose the color space (PQ or SDR); black and blank flips keep the current one.
	if (!frame.blank && swapchain.SetHdr(frame.hdr)) {
		m_impl->RecoverSwapchain(Swapchain::Status::Recreate);
	}
	if (m_impl->recreate_requested.exchange(false, std::memory_order_acq_rel)) {
		std::printf("Presenter: the swapchain made again (the window went in or out of full screen)\n");
		std::fflush(stdout);
		m_impl->RecoverSwapchain(Swapchain::Status::Recreate);
	}
	for (uint32_t attempt = 0; attempt < 2; attempt++) {
		auto status = swapchain.AcquireNextImage();
		if (status != Swapchain::Status::Success) {
			m_impl->RecoverSwapchain(status);
			continue;
		}
		{
			// The present thread records and submits on its own scheduler (present_scheduler: its own pool, timeline
			// and command buffers; the queue has its own lock), from the presenter's frame image into the swapchain's:
			// nothing the GPU thread uses. Under the renderer's lock, its GPU thread waited for the whole recording
			// and submission once a frame (~60 us, ~110 us with the wake-up). Frame generation shares the renderer's.
			std::optional<RenderLockGuard> render_lock;
			if (FrameGen::Enabled()) render_lock.emplace(m_impl->renderer.GetMutex());
			auto&             command          = m_impl->present_scheduler.BeginCommand();
			const bool        draw_system_overlay =
			    (overlay_visual.active || hud != nullptr) && swapchain.PrepareSystemOverlay(hud);
			if (FrameGen::Enabled() && (hud == nullptr || !draw_system_overlay)) {
				FrameGen::SetStaticRect(0.0f, 0.0f, 0.0f, 0.0f); // (no panel: the whole picture moves)
			}
			if (hud != nullptr && draw_system_overlay && FrameGen::Enabled()) {
				// The panel's motion vectors are zeroed (a margin around it).
				const auto& region = hud->region;
				const float w = static_cast<float>(region.extent.width), h = static_cast<float>(region.extent.height);
				const float margin = 8.0f;
				FrameGen::SetStaticRect(
				    (static_cast<float>(hud->drawn.offset.x - region.offset.x) - margin) / w,
				    (static_cast<float>(hud->drawn.offset.y - region.offset.y) - margin) / h,
				    (static_cast<float>(hud->drawn.offset.x - region.offset.x) + static_cast<float>(hud->drawn.extent.width) + margin) / w,
				    (static_cast<float>(hud->drawn.offset.y - region.offset.y) + static_cast<float>(hud->drawn.extent.height) + margin) / h);
			}
			swapchain.RecordPresentCommands(command, frame.image, frame.view, draw_system_overlay, frame.scene, frame.menu,
			                                fsr_inputs);
			if (FrameGen::OwnsSwapchain()) {
				FrameGen::RecordPrepare(m_impl->window.graphic_ctx, command.Handle(), fsr_inputs, swapchain.PresentRegion());
			} else if (FrameGen::HudlessOn() && !frame.reusing_last) {
				// DLSS-G's HUD-less image, where the frame image went.
				FrameGen::ScaleHudless(m_impl->window.graphic_ctx, command.Handle(), frame.fg, swapchain.Extent(),
				                       swapchain.PresentRegion(), swapchain.Format());
			}
			// (A live `screenshot`: the game's picture as the presenter got it.)
			auto capture       = Screenshot::Record(m_impl->window.graphic_ctx, command.Handle(), frame.image, Screenshot::Take());
			frame.present_tick = swapchain.Submit(m_impl->present_scheduler);
			if (capture.buffer != VK_NULL_HANDLE) {
				m_impl->present_scheduler.Wait(frame.present_tick);
				Screenshot::Finish(m_impl->window.graphic_ctx, capture);
			}
		}
#if defined(_WIN32)
		// The Windows driver waits in vkQueuePresentKHR, inside the kernel, for the semaphore's
		// GPU work while it holds a device-wide lock: every queue submission waits behind it,
		// including those the GPU work may still depend on (a deadlock). Wait for the GPU here,
		// holding nothing; the present on its own queue then finds the semaphore signaled.
		m_impl->present_scheduler.Wait(frame.present_tick);
#endif
		// KYTY_FRAMEGEN: Streamline takes the present (vkQueuePresentKHR is its hook) and adds
		// the generated frames; an idle refresh of the last frame carries no new inputs.
		const bool generate = FrameGen::Enabled();
		if (generate) {
			FrameGen::BeforePresent(frame.fg.valid && !frame.reusing_last ? &frame.fg : nullptr,
			                        swapchain.Extent(), swapchain.PresentRegion(), swapchain.Format(),
			                        swapchain.ImageCount());
		}
		if (!reuse) AntiLag::Present(frame.anti_lag_frame);
		const auto present_start = std::chrono::steady_clock::now();
		status = swapchain.Present();
		const auto present_took = std::chrono::steady_clock::now() - present_start;
		if (generate) {
			FrameGen::AfterPresent();
		}
		if (status != Swapchain::Status::Success) {
			m_impl->RecoverSwapchain(status);
			continue;
		}

		LiveCounters::Add(LiveCounters::Presents);
		if (!reuse) m_impl->NotePresentCall(present_took);
		m_impl->presented_overlay_revision.store(overlay_visual.revision,
		                                         std::memory_order_release);
		m_impl->window.UpdateTitle();
		m_impl->frames.Release(&frame, true);
		return;
	}
	LOGF("Vulkan presentation retry exhausted; dropping frame\n");
	m_impl->frames.Release(&frame, reuse);
}

void Presenter::RequestSwapchainRecreate() noexcept {
	m_impl->recreate_requested.store(true, std::memory_order_release);
}

void Presenter::Discard(Frame& frame) {
	m_impl->frames.Release(&frame);
}

WindowContext::WindowContext() = default;

} // namespace Libs::Graphics
