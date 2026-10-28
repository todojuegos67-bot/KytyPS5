#include "local/port-live.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window/windowInternal.h"
#include "graphics/presentation/window/hostInput.h"
#include "loader/demonsSoulsRenderSize.h"
#include "loader/demonsSoulsWarp.h"
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "vulkan-recording.h"
#endif
#include "frame-gen.h"
#include "local/fsr3-fg.h"
#include "local/kbm-warp.h"
#include "local/anti-lag.h"
#include "local/rt-local-lights.h"
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
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <string>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <string_view>
#include <limits>
#include <memory>
#include <optional>
#include <vector>
#include <vulkan/vk_platform.h>

// IWYU pragma: no_include <intrin.h>

#define KYTY_ENABLE_DEBUG_PRINTF
#define KYTY_DBG_INPUT

namespace Libs::Graphics {

struct Presenter::Frame {
	VulkanImage            image;
	FrameGen::FrameInputs  fg; // frame generation inputs (KYTY_FRAMEGEN)
	uint64_t               present_tick = 0;
	bool                   busy         = false;
	bool                   reusing_last = false;
	bool                   hdr          = false; // BT.2100 PQ pixels
	bool                   blank        = false; // a cleared frame (no guest picture)
	bool                   scene        = false; // the game's 3D view (FrameGen::TakeScene)
	uint64_t               latency_index = 0;    // KYTY_ANTILAG (local/anti-lag.h)
	uint64_t               kbm_seq       = 0;    // KYTY_KBM_WARP: the pad reads it shows (HostInputKbmOnFlip)

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
			if (frame->image.image != nullptr) {
				m_window.graphic_ctx.DeleteImage(frame->image);
			}
			FrameGen::DestroyInputs(m_window.graphic_ctx, frame->fg);
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
	// (Sampled: KYTY_KBM_WARP's and FSR 3's HUD layer passes read it; the format's features hold it, checked above.)
	create.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
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
	// `scene`: the frame shows the game's 3D view (Presenter::IsSceneFrame).
	void                 RecordPresentCommands(CommandBuffer& command, VulkanImage& source, FrameGen::FrameInputs* fg,
	                                           bool draw_system_overlay, bool scene);
	uint64_t             Submit(CommandScheduler& scheduler);
	[[nodiscard]] Status Present();
	void                 SetPresentId(uint64_t id) noexcept { m_present_id = id; }
	// KYTY_KBM_WARP: the pad reads the next recorded frame shows (Presenter::Frame::kbm_seq).
	void                 SetKbmSeq(uint64_t seq) noexcept { m_kbm_seq = seq; }

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
	[[nodiscard]] vk::Image  CurrentImage() const noexcept { return m_images[m_image_index]; }
	[[nodiscard]] vk::SwapchainKHR Handle() const noexcept { return m_handle; }
	// The FSR 3 frame generation swapchain (KYTY_FSR3_FG) stands in for the swapchain.
	[[nodiscard]] bool IsFsr3() const noexcept { return m_fsr3; }
	// The frame generation swapchain keeps the window's size changes to itself (its presents of an out-of-date
	// swapchain are dropped: a frozen picture after going fullscreen); the surface's extent tells.
	[[nodiscard]] bool Fsr3SurfaceChanged() const {
		if (!m_fsr3) return false;
		vk::SurfaceCapabilitiesKHR capabilities {};
		if (m_window.graphic_ctx.physical_device.getSurfaceCapabilitiesKHR(m_window.surface, &capabilities) != vk::Result::eSuccess)
			return false;
		const auto extent = capabilities.currentExtent;
		return extent.width != std::numeric_limits<uint32_t>::max() && extent.width != 0 && extent.height != 0 &&
		       (extent.width != m_extent.width || extent.height != m_extent.height);
	}
	// The presents wait for the display's vertical blank (VSync: FIFO).
	[[nodiscard]] bool IsFifo() const noexcept { return m_fifo.load(std::memory_order_relaxed); }

private:
	void                 Destroy();
	vk::SurfaceFormatKHR SelectFormat();
	// KYTY_EXCLUSIVE_FULLSCREEN: the display taken (focused) or given back (focus lost, minimized) before a present.
	void UpdateExclusive();
	void ReleaseExclusive(const char* why);

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
	bool                        m_fsr3 = false;
	uint64_t                    m_present_id = 0; // KYTY_ANTILAG: VkPresentIdKHR (0 none)
	uint64_t                    m_kbm_seq    = 0; // KYTY_KBM_WARP (SetKbmSeq)
	bool                        m_exclusive_app      = false; // application controlled exclusive full screen
	bool                        m_exclusive_acquired = false;
	std::chrono::steady_clock::time_point m_exclusive_retry {};
	uint32_t                    m_exclusive_failures = 0;
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

	Image& ResolveSurface(const ImageInfo& info, ImageId* out_id = nullptr) {
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
		if (out_id != nullptr) *out_id = image_id;
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

	// (10-28) Frame generation's own per-second measurement, the "[fsr3] real X fps, shown Y fps" / "[dlssg] ... real X fps,
	// shown Y fps" lines in the run's log (KYTY_RUN_LOG; fsr3-fg.cpp's shown = the presents the FidelityFX DLL counted),
	// as metricas.ps1 reads them: the panel's numbers with FSR 3 / MFG and DLSS-G (the game's flips were the real frames
	// only, times the requested multiplier). Only a line written since the last read counts (a second with shown == real
	// is shown as measured, x1).
	struct FgMeasure {
		double                                real  = 0.0;
		double                                shown = 0.0;
		std::chrono::steady_clock::time_point at {};
		long long                             seen  = -1;
	} fg_measure;
	bool ReadFgMeasure(std::chrono::steady_clock::time_point now) {
		static const char* log = std::getenv("KYTY_RUN_LOG");
		if (log == nullptr || *log == '\0') return false;
		std::fflush(stdout);
#ifdef _WIN32
		static const std::wstring wlog = std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(log))).wstring();
		FILE* f = _wfopen(wlog.c_str(), L"rb");
#else
		FILE* f = std::fopen(log, "rb");
#endif
		if (f != nullptr) {
			std::fseek(f, 0, SEEK_END);
			const long long size = std::ftell(f);
			const long long from = size > 16384 ? size - 16384 : 0;
			std::string     text(static_cast<size_t>(size - from), '\0');
			std::fseek(f, static_cast<long>(from), SEEK_SET);
			text.resize(std::fread(text.data(), 1, text.size(), f));
			std::fclose(f);
			for (size_t at = text.rfind("\n["); at != std::string::npos; at = at == 0 ? std::string::npos : text.rfind("\n[", at - 1)) {
				const size_t end  = text.find('\n', at + 1);
				const std::string line = text.substr(at + 1, end == std::string::npos ? std::string::npos : end - at - 1);
				if (line.rfind("[fsr3] ", 0) != 0 && line.rfind("[dlssg] ", 0) != 0) continue;
				const size_t r = line.find("real ");
				double real = 0, shown = 0;
				if (r == std::string::npos || std::sscanf(line.c_str() + r, "real %lf fps, shown %lf fps", &real, &shown) != 2) continue;
				if (end == std::string::npos) continue; // (not finished yet)
				const long long pos = from + static_cast<long long>(at);
				if (pos > fg_measure.seen) {
					fg_measure.seen  = pos;
					fg_measure.real  = real;
					fg_measure.shown = shown;
					fg_measure.at    = now;
				}
				break;
			}
		}
		return fg_measure.at != std::chrono::steady_clock::time_point {} && now - fg_measure.at < std::chrono::seconds(3) &&
		       fg_measure.real > 0.0;
	}

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
	std::chrono::steady_clock::time_point status_shown {};
	void UpdateStatus(std::chrono::steady_clock::time_point now) {
		const auto progress = renderer.GetPipelineCache().GetPrefetchProgress();
		hud.status.clear();
		if (progress.total == 0) {
			return;
		}
		// KYTY_STATUS_HUD_SECONDS=<n> (default 6, 0: always): the progress line goes after n seconds on screen; the
		// work carries on behind (the launcher's choice: shown a moment, not over the whole game).
		static const int hide_after = [] {
			const char* value = std::getenv("KYTY_STATUS_HUD_SECONDS");
			return value != nullptr && *value != '\0' ? std::max(0, std::atoi(value)) : 6;
		}();
		if (status_shown == std::chrono::steady_clock::time_point {}) status_shown = now;
		if (hide_after > 0 && now - status_shown >= std::chrono::seconds(hide_after)) return;
		std::array<char, 128> line {};
		const bool es = PortLive::status_es.load(std::memory_order_relaxed); // (the PC port)
		if (es && (progress.done < progress.total || progress.label != nullptr)) {
			const bool pipelines = progress.label != nullptr && std::strstr(progress.label, "pipelines") != nullptr;
			std::snprintf(line.data(), line.size(), "%s %zu/%zu  (%zu%%)", pipelines ? "Preparando pipelines" : "Preparando shaders",
			              progress.done, progress.total, progress.done * 100 / progress.total);
		} else if (progress.done < progress.total || progress.label != nullptr) {
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
			std::snprintf(line.data(), line.size(), es ? "Shaders listos (%zu)" : "All shaders ready (%zu)", progress.total);
		}
		hud.status          = line.data();
		hud.status_fraction = static_cast<float>(progress.done) / static_cast<float>(progress.total);
	}

	[[nodiscard]] SystemOverlayHud* UpdateHud(bool reuse) {
		const int  level = PerfOverlay::Level();
		const auto now   = std::chrono::steady_clock::now();
		UpdateStatus(now);
		hud.notice = Loader::DemonsSoulsWarp::HudText();
		hud.trophy = Libs::Trophy::CurrentToast(&hud.trophy_seconds); // (the trophy popup: drawn with the panels)
		const bool markers = RtLocal::OverlayWanted(); // (KYTY_RTLS_DEBUG: drawn by PlaceHud)
		if (!markers) hud.markers.clear();
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
			return hud.status.empty() && hud.notice.empty() && !markers && hud.trophy == nullptr ? nullptr : PlaceHud();
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
			const bool fg_on = FrameGen::Enabled() || Fsr3::Active();
			if (fg_on && ReadFgMeasure(now)) {
				// (10-28) Measured: the frames shown on screen, the game's real ones and their ratio (not the requested one).
				const int mult = std::max(1, static_cast<int>(std::lround(fg_measure.shown / fg_measure.real)));
				PortLive::fg_measured.store(mult, std::memory_order_relaxed);
				std::snprintf(title.data(), title.size(), "%.0f FPS", fg_measure.shown);
				std::snprintf(detail.data(), detail.size(), "Reales %.0f FPS  |  FG x%d", fg_measure.real, mult);
			} else if (presents > 1) {
				PortLive::fg_measured.store(0, std::memory_order_relaxed);
				std::snprintf(title.data(), title.size(), "%.0f FPS",
				              (seconds >= 1.0 ? hud_frames / seconds : game) * presents);
				std::snprintf(detail.data(), detail.size(), "DLSS Frame Generation %ux  |  game %.0f fps", presents, game);
			} else {
				PortLive::fg_measured.store(0, std::memory_order_relaxed);
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
		if (hud.title.empty() && hud.status.empty() && hud.notice.empty() && !markers && hud.trophy == nullptr) {
			return nullptr;
		}
		return PlaceHud();
	}
	SystemOverlayHud* PlaceHud() {
		hud.region = swapchain.PresentRegion();
		if (hud.region.extent.width == 0 || hud.region.extent.height == 0) {
			hud.region = vk::Rect2D {{0, 0}, swapchain.Extent()};
		}
		if (RtLocal::OverlayWanted()) {
			std::vector<RtLocal::Marker> found;
			RtLocal::DebugMarkers(static_cast<float>(hud.region.offset.x), static_cast<float>(hud.region.offset.y),
			                      static_cast<float>(hud.region.extent.width), static_cast<float>(hud.region.extent.height), found);
			hud.markers.clear();
			for (auto& m: found) hud.markers.push_back({m.x, m.y, m.radius, m.rgba, m.shape, std::move(m.label)});
		}
		return &hud;
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
	Waiting                     waiting;
	bool                        presenting = false; // (waiting_mutex) the thread took one and has not presented it yet
	std::jthread                thread;
	// A frame the vblank thread put in place of one still waiting: that one goes back to the pool unshown.
	void Replace(const Waiting& next) {
		Waiting dropped;
		{
			std::lock_guard lock(waiting_mutex);
			dropped = std::exchange(waiting, next);
		}
		waiting_changed.notify_one();
		// (A game frame DLSS-G had inputs for never shown: the next one's motion spans two frames.)
		if (dropped.frame != nullptr && !dropped.reuse && dropped.frame->fg.valid) FrameGen::NoteDroppedFrame();
		// (A last frame shown again stays the last one.)
		if (dropped.frame != nullptr) frames.Release(dropped.frame, dropped.reuse);
	}

	// A cleared frame: into `producer` (the GPU thread's command buffer, a guest flip of a blank buffer), else on the
	// presenter's own scheduler (its thread's blank frames).
	void ClearFrame(Frame& frame, uint32_t width, uint32_t height, bool opaque, CommandBuffer* producer) {
		frame.fg.valid = false;
		frame.hdr      = false;
		frame.blank    = true;
		frame.scene    = false;
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
			LOGF("Presents limited to the display refresh: %d Hz\n", refresh);
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
	for (const auto& candidate: formats) {
		if (candidate.colorSpace != vk::ColorSpaceKHR::eSrgbNonlinear) {
			continue;
		}
		if (candidate.format == vk::Format::eB8G8R8A8Unorm || candidate.format == vk::Format::eR8G8B8A8Unorm) {
			return candidate;
		}
	}
	EXIT("no supported UNORM swapchain format\n");
}


// KYTY_EXCLUSIVE_FULLSCREEN=1 (Windows): VK_EXT_full_screen_exclusive. Its structures and functions are in
// vulkan_win32.h (behind VK_USE_PLATFORM_WIN32_KHR, which this build does not define); the same layouts here, with
// the structure types and results of vulkan_core.h.
namespace ExclusiveFullscreen {

bool Requested() {
#if defined(_WIN32)
	static const bool requested = [] {
		const char* value = std::getenv("KYTY_EXCLUSIVE_FULLSCREEN");
		return value != nullptr && *value == '1';
	}();
	return requested;
#else
	return false;
#endif
}

} // namespace ExclusiveFullscreen

namespace {

constexpr int32_t kFullScreenExclusiveAllowed               = 1; // VK_FULL_SCREEN_EXCLUSIVE_ALLOWED_EXT
constexpr int32_t kFullScreenExclusiveApplicationControlled = 3; // VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT

struct FullScreenExclusiveInfo { // VkSurfaceFullScreenExclusiveInfoEXT
	VkStructureType sType               = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT;
	void*           pNext               = nullptr;
	int32_t         fullScreenExclusive = kFullScreenExclusiveApplicationControlled;
};
struct FullScreenExclusiveWin32Info { // VkSurfaceFullScreenExclusiveWin32InfoEXT
	VkStructureType sType    = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_WIN32_INFO_EXT;
	const void*     pNext    = nullptr;
	void*           hmonitor = nullptr; // HMONITOR
};
struct FullScreenExclusiveCapabilities { // VkSurfaceCapabilitiesFullScreenExclusiveEXT
	VkStructureType sType                         = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_FULL_SCREEN_EXCLUSIVE_EXT;
	void*           pNext                         = nullptr;
	VkBool32        fullScreenExclusiveSupported  = VK_FALSE;
};
using PfnFullScreenExclusiveMode = VkResult(VKAPI_PTR*)(VkDevice, VkSwapchainKHR);

PfnFullScreenExclusiveMode ExclusiveFunction(vk::Device device, const char* name) {
	auto* get = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
	return get != nullptr ? reinterpret_cast<PfnFullScreenExclusiveMode>(get(static_cast<VkDevice>(device), name)) : nullptr;
}

void LogExclusive(const std::string& text) {
	static std::string last;
	if (text == last) return;
	last = text;
	std::printf("Exclusive fullscreen: %s\n", text.c_str());
	std::fflush(stdout);
	LOGF("Exclusive fullscreen: %s\n", text.c_str());
}

void* WindowMonitor(SDL_Window* window) {
#if defined(_WIN32)
	SDL_SysWMinfo info {};
	SDL_VERSION(&info.version);
	if (window != nullptr && SDL_GetWindowWMInfo(window, &info) == SDL_TRUE && info.subsystem == SDL_SYSWM_WINDOWS) {
		return LocalPlatform::MonitorHandle(info.info.win.window);
	}
#else
	(void)window;
#endif
	return nullptr;
}

} // namespace

void Swapchain::ReleaseExclusive(const char* why) {
	if (!m_exclusive_acquired || m_handle == nullptr) {
		m_exclusive_acquired = false;
		return;
	}
	m_exclusive_acquired = false;
	if (auto* release = ExclusiveFunction(m_window.graphic_ctx.device, "vkReleaseFullScreenExclusiveModeEXT"); release != nullptr) {
		const auto result = static_cast<vk::Result>(release(static_cast<VkDevice>(m_window.graphic_ctx.device),
		                                                    static_cast<VkSwapchainKHR>(m_handle)));
		LogExclusive(std::string("released (") + why + ", " + vk::to_string(result) + ")");
	}
}

void Swapchain::UpdateExclusive() {
	if (!m_exclusive_app || m_handle == nullptr) return;
	const bool wanted = ExclusiveFullscreen::g_wanted.load(std::memory_order_relaxed);
	if (!wanted) {
		ReleaseExclusive("focus lost or minimized");
		return;
	}
	if (m_exclusive_acquired) return;
	const auto now = std::chrono::steady_clock::now();
	if (now < m_exclusive_retry) return;
	auto* acquire = ExclusiveFunction(m_window.graphic_ctx.device, "vkAcquireFullScreenExclusiveModeEXT");
	if (acquire == nullptr) {
		LogExclusive("not supported (no vkAcquireFullScreenExclusiveModeEXT): borderless full screen");
		m_exclusive_app = false;
		return;
	}
	const auto result = static_cast<vk::Result>(
	    acquire(static_cast<VkDevice>(m_window.graphic_ctx.device), static_cast<VkSwapchainKHR>(m_handle)));
	if (result == vk::Result::eSuccess) {
		m_exclusive_acquired = true;
		m_exclusive_failures = 0;
		LogExclusive("acquired (" + std::to_string(m_extent.width) + "x" + std::to_string(m_extent.height) + ")");
		return;
	}
	// (Not focused yet, or the driver refuses: tried again later, at most every 2 s; given up after 10.)
	++m_exclusive_failures;
	m_exclusive_retry = now + std::chrono::seconds(2);
	LogExclusive("not acquired (" + vk::to_string(result) + ")" +
	             (m_exclusive_failures >= 10 ? ": given up, borderless full screen" : ": tried again"));
	if (m_exclusive_failures >= 10) m_exclusive_app = false;
}

void Swapchain::Create() {
	auto& graphics = m_window.graphic_ctx;
	EXIT_IF(graphics.device == nullptr);
	EXIT_IF(m_window.surface == nullptr);

	Common::LockGuard lock(m_window.mutex);
	EXIT_IF(graphics.screen_width == 0);
	EXIT_IF(graphics.screen_height == 0);
	auto&             surface = m_window.surface_capabilities;
	EXIT_NOT_IMPLEMENTED(surface.formats.empty());

	// KYTY_EXCLUSIVE_FULLSCREEN=1: while the window is full screen, the swapchain is made for exclusive full screen on
	// its monitor: application controlled (taken after creation and while focused, given back without focus), or, for
	// the FSR 3 frame generation swapchain (its real swapchain is AMD's, its handle not one the driver knows), left to
	// the driver (ALLOWED: the driver takes the display itself when the window covers it and has the focus).
	m_exclusive_app      = false;
	m_exclusive_acquired = false;
	m_exclusive_failures = 0;
	m_exclusive_retry    = {};
	FullScreenExclusiveInfo      exclusive_info {};
	FullScreenExclusiveWin32Info exclusive_win32 {};
	bool                         exclusive = false;
	if (ExclusiveFullscreen::Requested() && m_window.full_screen_exclusive_ext) {
		const auto window_flags = SDL_GetWindowFlags(m_window.window);
		const bool fullscreen   = (window_flags & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_FULLSCREEN_DESKTOP)) != 0;
		exclusive_win32.hmonitor = WindowMonitor(m_window.window);
		if (!fullscreen) {
			LogExclusive("off (the window is not full screen): windowed");
		} else if (exclusive_win32.hmonitor == nullptr) {
			LogExclusive("not supported (no monitor for the window): borderless full screen");
		} else {
			// (With FSR 3 or Streamline/DLSS-G the presented swapchain is wrapped by the frame generation layer:
			// left to the driver, never application controlled.)
			exclusive_info.fullScreenExclusive = (Fsr3::Requested() || FrameGen::StreamlineLoaded())
			                                         ? kFullScreenExclusiveAllowed
			                                         : kFullScreenExclusiveApplicationControlled;
			exclusive_info.pNext = &exclusive_win32;
			exclusive            = true;
			auto* get_capabilities2 = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceSurfaceCapabilities2KHR;
			if (m_window.surface_capabilities2_ext && get_capabilities2 != nullptr) {
				// The surface's capabilities for exclusive full screen on that monitor (they replace the plain ones).
				FullScreenExclusiveCapabilities exclusive_caps {};
				VkPhysicalDeviceSurfaceInfo2KHR surface_info {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR, &exclusive_info,
				                                              static_cast<VkSurfaceKHR>(m_window.surface)};
				VkSurfaceCapabilities2KHR       caps2 {VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR, &exclusive_caps, {}};
				const auto result = static_cast<vk::Result>(
				    get_capabilities2(static_cast<VkPhysicalDevice>(m_window.graphic_ctx.physical_device), &surface_info, &caps2));
				if (result != vk::Result::eSuccess) {
					LogExclusive("not supported (surface capabilities query: " + vk::to_string(result) + "): borderless full screen");
					exclusive = false;
				} else if (exclusive_caps.fullScreenExclusiveSupported == VK_FALSE) {
					LogExclusive("not supported (the driver offers none for this surface and monitor): borderless full screen");
					exclusive = false;
				} else {
					surface.capabilities = *reinterpret_cast<const vk::SurfaceCapabilitiesKHR*>(&caps2.surfaceCapabilities);
				}
			}
		}
	}

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
	// KYTY_FLIP_WHEN_READY: the shortest FIFO queue (2 images, or the surface's minimum) so a frame is shown the
	// refresh after it is done, not 1-2 queued presents later (with G-SYNC the display follows each present).
	// KYTY_SWAPCHAIN_IMAGES=N overrides the count (clamped to the surface's limits).
	if (VideoOut::FlipWhenReady()) image_count = std::max(2u, surface.capabilities.minImageCount);
	if (const char* value = std::getenv("KYTY_SWAPCHAIN_IMAGES"); value != nullptr && std::atoi(value) > 0) {
		image_count = std::max(static_cast<uint32_t>(std::atoi(value)), surface.capabilities.minImageCount);
	}
	// KYTY_FRAMEGEN: DLSS-G holds the presented image while it generates and shows its frames (one more per generated
	// frame); with the minimum the acquire of the next image waited for them, and the flip (done when the present
	// returns) with it: the game then waited for the flip (log 032429: 32 % of the render thread in WaitFlipDone).
	// (KYTY_DLSSG_IMAGES=1 asks for them: with them the game closed while its title loaded, logs 040442-040700.)
	static const bool more_images = [] {
		const char* value = std::getenv("KYTY_DLSSG_IMAGES");
		return value != nullptr && *value == '1';
	}();
	if (FrameGen::Requested() && more_images) image_count += static_cast<uint32_t>(FrameGen::GeneratedFrames());
	if (surface.capabilities.maxImageCount != 0) {
		image_count = std::min(image_count, surface.capabilities.maxImageCount);
	}
	if (FrameGen::Requested()) std::printf("Frame generation: swapchain of %u images\n", image_count);
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
	// (The chain lives until the swapchain is made: Streamline's vkCreateSwapchainKHR and AMD's frame interpolation
	// swapchain pass it on to the driver's; AMD's copies VkSurfaceFullScreenExclusive[Win32]InfoEXT, FFX 1.1.4.)
	if (exclusive) create_info.pNext = &exclusive_info;
	// KYTY_ANTILAG (Reflex): the latency mode for the swapchain (FSR 3's: its real one, with the patched DLL only).
	const void* chain_base = create_info.pNext;
	if (Fsr3::Requested()) create_info.pNext = AntiLag::ChainSwapchain(const_cast<void*>(chain_base), true);
	m_fsr3 = Fsr3::CreateSwapchain(graphics, create_info, &m_handle);
	if (!m_fsr3) create_info.pNext = AntiLag::ChainSwapchain(const_cast<void*>(chain_base), false);
	if (!m_fsr3) {
		if (exclusive && !FrameGen::StreamlineLoaded() && exclusive_info.fullScreenExclusive != kFullScreenExclusiveApplicationControlled) {
			// (FSR 3 asked for but its swapchain not made: the plain one, application controlled.)
			exclusive_info.fullScreenExclusive = kFullScreenExclusiveApplicationControlled;
		}
		auto result = graphics.device.createSwapchainKHR(&create_info, nullptr, &m_handle);
		if (result != vk::Result::eSuccess && exclusive) {
			LogExclusive("not supported (vkCreateSwapchainKHR with it: " + vk::to_string(result) + "): borderless full screen");
			exclusive         = false;
			create_info.pNext = AntiLag::ChainSwapchain(nullptr, false);
			m_handle          = nullptr;
			result            = graphics.device.createSwapchainKHR(&create_info, nullptr, &m_handle);
		}
		RequireVulkanSuccess(result, "vkCreateSwapchainKHR");
	}
	EXIT_IF(m_handle == nullptr);
	AntiLag::OnSwapchain(static_cast<VkSwapchainKHR>(m_handle), m_fsr3);
	if (exclusive) {
		if (!m_fsr3 && FrameGen::StreamlineLoaded()) {
			LogExclusive("left to the driver (Streamline / DLSS-G frame generation: not application controlled)");
		} else if (m_fsr3) {
			LogExclusive("left to the driver (FSR 3 frame generation swapchain: its real swapchain is AMD's, so not "
			             "application controlled)");
		} else {
			m_exclusive_app = true;
			UpdateExclusive();
		}
	}
	std::printf("Swapchain: %ux%u, present mode %d, %u images%s\n", m_extent.width, m_extent.height,
	            static_cast<int>(create_info.presentMode), image_count, m_fsr3 ? " (FSR 3 frame generation)" : "");
	std::fflush(stdout);
	if (Config::HdrEnabled()) {
		std::printf("Vulkan presentation: %s\n", m_hdr_active ? "HDR10 (PQ)" : "SDR");
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
	if (graphics.present_queue != nullptr) {
		Common::LockGuard queue_lock(graphics.present_queue_mutex);
		RequireVulkanSuccess(graphics.present_queue.waitIdle(), "wait for present queue");
	}
	if (m_system_overlay != nullptr) {
		m_system_overlay->ReleaseVulkan();
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
	ReleaseExclusive("swapchain made again");
	m_exclusive_app = false;
	if (m_handle != nullptr) {
		AntiLag::OnSwapchain(VK_NULL_HANDLE, false);
		if (m_fsr3) {
			Fsr3::DestroySwapchain(graphics, m_handle);
		} else {
			graphics.device.destroySwapchainKHR(m_handle, nullptr);
		}
	}
	m_fsr3 = false;

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
	    m_fsr3 ? Fsr3::AcquireNextImage(m_window.graphic_ctx, m_handle, m_image_acquired[m_frame_index], &m_image_index)
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
		case static_cast<vk::Result>(VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT):
			m_exclusive_acquired = false;
			LogExclusive("lost (vkAcquireNextImageKHR): swapchain made again");
			return Status::Recreate;
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

void Swapchain::RecordPresentCommands(CommandBuffer& command, VulkanImage& source, FrameGen::FrameInputs* fg,
                                      bool draw_system_overlay, bool scene) {
	if (source.state.layout != vk::ImageLayout::eTransferSrcOptimal) {
		EXIT("invalid prepared presentation image, vk_image=%p layout=%d\n",
		     static_cast<void*>(source.image), static_cast<int>(source.state.layout));
	}
	EXIT_IF(m_image_index >= m_images.size());
	auto vk_command = command.Handle();

	vk::ImageMemoryBarrier to_transfer {};
	to_transfer.sType                           = vk::StructureType::eImageMemoryBarrier;
	to_transfer.dstAccessMask                   = vk::AccessFlagBits::eTransferWrite;
	to_transfer.oldLayout                       = vk::ImageLayout::eUndefined;
	to_transfer.newLayout                       = vk::ImageLayout::eTransferDstOptimal;
	to_transfer.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.image                           = m_images[m_image_index];
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
	// (movies, menus, loading screens) stretched from their 16:9 (the user, 10-19: the opening movie filling the screen).
	// With the wide output (DemonsSoulsRenderSize::WideOutputWidth) the game's frames are as wide as the 3D; one without
	// it (movies, menus, loading screens) holds its 16:9 picture at the left: that part is shown, stretched.
	int          dst_x = 0, dst_y = 0;
	int          dst_w = static_cast<int>(m_extent.width), dst_h = static_cast<int>(m_extent.height);
	uint32_t     src_w = source.extent.width;
	if (!scene && Loader::DemonsSoulsRenderSize::WideOutputWidth() != 0 && source.extent.height != 0)
		src_w = std::min(src_w, (source.extent.height * 16u + 8u) / 9u);
	const double frame_aspect = source.extent.height != 0 ? static_cast<double>(src_w) / source.extent.height : 0.0;
	const double wide         = Loader::DemonsSoulsRenderSize::WideAspect();
	double       aspect       = wide > 0.0 ? wide : aspect_fit ? frame_aspect : 0.0;
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
			vk_command.clearColorImage(m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal, &black, 1,
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
	// KYTY_KBM_WARP=1 (with KYTY_KBM=1): late reprojection of the mouse turn the game has not shown yet. The source rect
	// moves by the pending yaw/pitch in pixels (pinhole: f = (w/2)/tan(hfov/2)); the uncovered strip repeats the edge
	// pixels. The whole final image moves (the HUD too), so the shift is capped (KYTY_KBM_WARP_MAX, of the width).
	// With the frame's HUD-less composite (KbmWarp::Record, kbm-warp.h) only the 3D moves, the HUD stays; with FSR 3
	// frame generation the frame's inputs (HUD-less image, depth, motion vectors) are shifted alike before FSR 3 takes
	// them. DLSS-G: not warped.
	int          shift_x = 0, shift_y = 0;
	VulkanImage* blit_source = &source;
	{
		struct WarpCfg {
			bool   on = false;
			double fov = 70.0, max = 0.06, invy = 1.0;
		};
		static const WarpCfg cfg = [] {
			WarpCfg     c;
			const char* on  = std::getenv("KYTY_KBM_WARP");
			const char* kbm = std::getenv("KYTY_KBM");
			c.on = on != nullptr && std::string_view(on) == "1" && kbm != nullptr && std::string_view(kbm) == "1";
			auto num = [](const char* name, double def, double lo, double hi) {
				const char* v = std::getenv(name);
				if (v == nullptr || *v == '\0') return def;
				const double d = std::strtod(v, nullptr);
				return std::isfinite(d) ? std::clamp(d, lo, hi) : def;
			};
			c.fov  = num("KYTY_KBM_WARP_FOV", 70.0, 20.0, 150.0);
			c.max  = num("KYTY_KBM_WARP_MAX", 0.06, 0.0, 0.25);
			c.invy = num("KYTY_KBM_WARP_INVY", 0.0, 0.0, 1.0) >= 0.5 ? -1.0 : 1.0;
			if (c.on) {
				LOGF("KBM warp: on (hfov %.0f deg, max shift %.1f%% of width, pitch %s; HUD-less composite %s: %s; FSR 3 frame "
				     "generation %s; DLSS-G: not warped)\n",
				     c.fov, c.max * 100.0, c.invy < 0.0 ? "inverted" : "normal",
				     KbmWarp::HudlessWanted() ? "on" : "off (KYTY_KBM_WARP_HUDLESS=0)",
				     KbmWarp::HudlessWanted() ? "the 3D moves under a still HUD (the whole image when a frame has none)"
				                              : "the whole image moves, the HUD too",
				     KbmWarp::Fsr3Wanted() ? "warped (its inputs shifted alike; KYTY_KBM_WARP_FSR3=1)"
				                           : "not warped (KYTY_KBM_WARP_FSR3=1 warps it)");
			}
			return c;
		}();
		static bool fg_logged = false;
		static const bool hud_layer = [] {
			const char* v = std::getenv("KYTY_FSR3_HUD_LAYER");
			return v != nullptr && *v == '1';
		}();
		// FSR 3: warped only with the frame's inputs shifted alike (its HUD layer mode compares F with H: not then).
		const bool fsr3    = IsFsr3();
		const bool fsr3_ok = fsr3 && !FrameGen::Enabled() && KbmWarp::Fsr3Wanted() && !hud_layer;
		const bool usable  = fg != nullptr && fg->valid;
		double     yaw = 0.0, pitch = 0.0;
		const bool pending =
		    cfg.on && scene && !Libs::VideoOut::MoviePlaying() && HostInputKbmWarpDegrees(m_kbm_seq, &yaw, &pitch);
		float      fx = 0.0f, fy = 0.0f;
		if (pending && src_w > 0 && source.extent.height > 0) {
			const double kPi = 3.14159265358979323846;
			const double f   = 0.5 * src_w / std::tan(cfg.fov * 0.5 * kPi / 180.0);
			const double cap = cfg.max * src_w;
			auto         px  = [&](double deg) {
                const double a = std::clamp(deg, -60.0, 60.0) * kPi / 180.0;
                return static_cast<float>(std::clamp(f * std::tan(a), -cap, cap));
			};
			fx = px(yaw);
			fy = px(pitch * cfg.invy);
		}
		const bool moving = std::fabs(fx) >= 0.5f || std::fabs(fy) >= 0.5f;
		if (cfg.on && (FrameGen::Enabled() || (fsr3 && !fsr3_ok))) {
			if (moving && !fg_logged) {
				fg_logged = true;
				LOGF("KBM warp: %s: frames are not warped\n", FrameGen::Enabled() ? "DLSS-G frame generation is on"
				                                                                   : "FSR 3 frame generation (KYTY_KBM_WARP_FSR3=1 "
				                                                                     "warps it, without KYTY_FSR3_HUD_LAYER=1)");
			}
			if (fsr3) KbmWarp::NoteUnshifted();
		} else if (fsr3) {
			// (A frame without inputs: FSR 3 generates nothing for it; the next starts from no shift.)
			VulkanImage* warped = nullptr;
			if (cfg.on && usable && (moving || KbmWarp::InputsShifted()))
				warped = KbmWarp::Record(m_window.graphic_ctx, command.Handle(), source, fg, moving ? fx : 0.0f,
				                         moving ? fy : 0.0f, true);
			if (warped != nullptr) blit_source = warped;
			else KbmWarp::NoteUnshifted();
		} else if (moving || (cfg.on && usable && KbmWarp::InputsShifted())) {
			// (10-24) Also while the edge zoom eases back after a turn (else it would snap off).
			VulkanImage* warped =
			    usable ? KbmWarp::Record(m_window.graphic_ctx, command.Handle(), source, fg, moving ? fx : 0.0f,
			                             moving ? fy : 0.0f, false)
			           : nullptr;
			if (warped != nullptr) {
				blit_source = warped;
			} else if (moving) {
				shift_x = static_cast<int>(std::lround(fx)); // (no HUD-less image: the whole image, as before)
				shift_y = static_cast<int>(std::lround(fy));
			}
		}
	}

	m_present_region = vk::Rect2D {{dst_x, dst_y}, {static_cast<uint32_t>(dst_w), static_cast<uint32_t>(dst_h)}};
	// Per axis up to two pieces: the shifted image and the edge strip (one source pixel stretched) it uncovers.
	struct Piece {
		int s0, s1; // source range
		int d0, d1; // destination range
	};
	auto pieces = [](int size, int shift, int d_origin, int d_size, Piece* out) {
		if (shift == 0 || std::abs(shift) >= size) {
			out[0] = {0, size, d_origin, d_origin + d_size};
			return 1;
		}
		const double scale = static_cast<double>(d_size) / size;
		const int    a0 = std::max(0, shift), a1 = std::min(size, size + shift);
		const int    d0 = d_origin + static_cast<int>(std::lround((a0 - shift) * scale));
		const int    d1 = d_origin + static_cast<int>(std::lround((a1 - shift) * scale));
		out[0]          = {a0, a1, d0, d1};
		if (shift > 0) out[1] = {size - 1, size, d1, d_origin + d_size};
		else out[1] = {0, 1, d_origin, d0};
		return out[1].d1 > out[1].d0 ? 2 : 1;
	};
	Piece      px[2], py[2];
	const int  nx = pieces(static_cast<int>(src_w), shift_x, dst_x, dst_w, px);
	const int  ny = pieces(static_cast<int>(source.extent.height), shift_y, dst_y, dst_h, py);
	vk::ImageBlit regions[4] {};
	uint32_t      count = 0;
	for (int ix = 0; ix < nx; ++ix) {
		for (int iy = 0; iy < ny; ++iy) {
			vk::ImageBlit& region                = regions[count++];
			region.srcSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
			region.srcSubresource.mipLevel       = 0;
			region.srcSubresource.baseArrayLayer = 0;
			region.srcSubresource.layerCount     = 1;
			region.srcOffsets[0]                 = vk::Offset3D {px[ix].s0, py[iy].s0, 0};
			region.srcOffsets[1]                 = vk::Offset3D {px[ix].s1, py[iy].s1, 1};
			region.dstSubresource                = region.srcSubresource;
			region.dstOffsets[0]                 = vk::Offset3D {px[ix].d0, py[iy].d0, 0};
			region.dstOffsets[1]                 = vk::Offset3D {px[ix].d1, py[iy].d1, 1};
		}
	}
	vk_command.blitImage(blit_source->image, vk::ImageLayout::eTransferSrcOptimal,
	                     m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal, count, regions,
	                     vk::Filter::eLinear);

	vk::ImageMemoryBarrier to_present {};
	to_present.sType         = vk::StructureType::eImageMemoryBarrier;
	to_present.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	to_present.dstAccessMask = draw_system_overlay ? vk::AccessFlagBits::eColorAttachmentRead |
	                                                     vk::AccessFlagBits::eColorAttachmentWrite
	                                               : vk::AccessFlagBits::eMemoryRead;
	to_present.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
	to_present.newLayout     = draw_system_overlay ? vk::ImageLayout::eColorAttachmentOptimal
	                                               : vk::ImageLayout::ePresentSrcKHR;
	to_present.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_present.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_present.image                           = m_images[m_image_index];
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
		m_system_overlay->Record(vk_command, m_image_views[m_image_index]);
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

// KYTY_FPS_CAP=<fps> (or auto: refresh - refresh^2/3600, 240 -> 224, under the G-SYNC range) with KYTY_FLIP_WHEN_READY:
// each present waits until 1/cap after the last one, by time (a high-resolution timer and a spin), not by vblanks.
static double FpsCapInterval(SDL_Window* window) {
	static const double interval = [window] {
		const char* value = std::getenv("KYTY_FPS_CAP");
		if (value == nullptr || *value == 0 || !VideoOut::FlipWhenReady()) return 0.0;
		double cap = 0.0;
		int    refresh = 0;
		if (std::string(value) == "auto") {
#if defined(_WIN32)
			SDL_SysWMinfo info {};
			SDL_VERSION(&info.version);
			if (window != nullptr && SDL_GetWindowWMInfo(window, &info) == SDL_TRUE && info.subsystem == SDL_SYSWM_WINDOWS) {
				refresh = LocalPlatform::DisplayRefreshHz(info.info.win.window);
			}
#else
			(void)window;
#endif
			if (refresh > 0) cap = refresh - static_cast<double>(refresh) * refresh / 3600.0;
		} else {
			cap = std::atof(value);
		}
		if (cap > 0.0) {
			std::printf("Present: fps cap %.2f (KYTY_FPS_CAP=%s%s)\n", cap, value,
			            refresh > 0 ? (", display " + std::to_string(refresh) + " Hz").c_str() : "");
		} else {
			std::printf("Present: no fps cap (KYTY_FPS_CAP=%s: no display refresh / invalid)\n", value);
		}
		std::fflush(stdout);
		return cap > 0.0 ? 1.0 / cap : 0.0;
	}();
	// (The PC port's menu: a cap chosen while playing, where the launch's applies.)
	if (const float live = PortLive::fps_cap.load(std::memory_order_relaxed); live > 0.0f && interval > 0.0) return 1.0 / live;
	return interval;
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
	// KYTY_ANTILAG (NVIDIA Reflex): the frame index as the present's id, as its latency markers have it.
	vk::PresentIdKHR present_id {};
	if (m_present_id != 0) { // (FSR 3: the patched DLL passes it to the real frame's present)
		present_id.swapchainCount = 1;
		present_id.pPresentIds    = &m_present_id;
		present.pNext             = &present_id;
	}

	UpdateExclusive();
	if (const double cap = FpsCapInterval(m_window.window); cap > 0.0) {
		static std::chrono::steady_clock::time_point next {};
		const auto step = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(cap));
		auto       now  = std::chrono::steady_clock::now();
		if (now < next) {
			LocalPlatform::SleepUntilPrecise(std::chrono::duration_cast<std::chrono::nanoseconds>(next.time_since_epoch()).count());
			now = std::chrono::steady_clock::now();
		}
		// (Late by more than a step: no catching up with presents back to back.)
		next = std::max(next + step, now);
	}
	vk::Result result;
	if (m_fsr3) {
		// The frame generation swapchain queues it (its threads pace and present; its submissions take the
		// graphics queue's lock themselves).
		result = Fsr3::QueuePresent(m_window.graphic_ctx, present);
	} else if (auto& graphics = m_window.graphic_ctx; graphics.present_queue != nullptr) {
		// Its own queue: a present blocked in the kernel does not hold up other submissions.
		Common::LockGuard lock(graphics.present_queue_mutex);
		result = graphics.present_queue.presentKHR(&present);
	} else {
		Common::LockGuard lock(graphics.queue_mutex);
		result = graphics.queue.presentKHR(&present);
	}
	VideoOut::NotePace(1);
	VideoOut::NotePresentDone();
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
		case static_cast<vk::Result>(VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT):
			m_exclusive_acquired = false;
			LogExclusive("lost (vkQueuePresentKHR): swapchain made again");
			return Status::Recreate;
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
	RenderLockGuard render_lock(m_impl->renderer.GetMutex());
	ImageId           image_id {};
	auto&             image = m_impl->ResolveSurface(info, &image_id);
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
	frame->fg.valid = false;
	if (FrameGen::Capturing()) {
		FrameGen::PrepareInputs(m_impl->window.graphic_ctx, m_impl->renderer.GetTextureCache(),
		                        buffer, frame->fg, image_id, image);
	}
	FrameGen::ReflexGuestFlip(frame->fg); // 10-23: DLSS-G's Reflex markers at the guest frame boundary
	return *frame;
}

Presenter::Frame& Presenter::PrepareBlankFrame(uint32_t width, uint32_t height, bool opaque,
                                               CommandBuffer* producer) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(producer == nullptr || producer->IsInvalid());
	auto*           frame = m_impl->frames.Acquire();
	frame->fg.reflex_index = 0; // (a blank frame: its Reflex markers at the present)
	RenderLockGuard render_lock(m_impl->renderer.GetMutex());
	m_impl->ClearFrame(*frame, width, height, opaque, producer);
	return *frame;
}

Presenter::Frame* Presenter::PrepareLastFrame() {
	return m_impl->frames.AcquireLast();
}

// KYTY_PRESENT_DIRECT=1 (a test for issue #37, a variable refresh rate that stopped engaging in v20261016b): each frame
// is presented on the vblank thread as it is handed over, as before the presenter had a thread of its own (eadf554f).
static bool PresentDirect() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_PRESENT_DIRECT");
		return value != nullptr && value[0] == '1';
	}();
	return on;
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
			// DLSS-G or FSR 3 generates frames for goes at once: it paces its presents itself.)
			for (;;) {
				const auto& w = impl.waiting;
				if (w.frame != nullptr && !w.reuse && w.frame->fg.valid && (FrameGen::Enabled() || Fsr3::Active())) break;
				const auto due = impl.NextPresentDue();
				if (std::chrono::steady_clock::now() >= due) break;
				impl.waiting_changed.wait_until(lock, token, due, [] { return false; });
				if (token.stop_requested()) return;
			}
#endif
			next            = std::exchange(impl.waiting, {});
			impl.presenting = true;
		}
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

bool Presenter::SystemOverlayChanged() const noexcept {
	return GetSystemOverlayVisualState().revision != m_impl->presented_overlay_revision.load(std::memory_order_acquire);
}

RenderContext& Presenter::Renderer() const noexcept {
	return m_impl->renderer;
}

bool Presenter::FlipRateApplies(const Frame& frame) noexcept {
	return !FrameGen::Enabled() || frame.fg.valid;
}

bool Presenter::IsGameView(const Frame& frame) noexcept {
	return frame.fg.valid;
}

bool Presenter::IsSceneFrame(const Frame& frame) noexcept {
	return frame.scene;
}

void Presenter::SetLatencyIndex(Frame& frame, uint64_t index) noexcept {
	frame.latency_index = index;
}

void Presenter::SetKbmSeq(Frame& frame, uint64_t seq) noexcept {
	frame.kbm_seq = seq;
}

bool Presenter::GeneratesFrames() noexcept {
	return FrameGen::Enabled() || Fsr3::Active();
}

void Presenter::Present(Frame& frame, bool reuse) {
	KYTY_PROFILER_FUNCTION();
	m_impl->frames.ValidateForPresent(&frame, reuse);
#if defined(_WIN32)
	// (A frame DLSS-G generates frames for goes to it whatever the display's rate: it paces its presents itself, and
	// held to the refresh it got 57 of a 102 fps game's frames a second on a 60 Hz display, 5x showing 285, not 500.)
	const bool generated = (FrameGen::Enabled() || Fsr3::Active()) && frame.fg.valid && !reuse;
	if (!generated && !m_impl->PresentDue()) {
		// (PresentLoop waited until it was due.)
		m_impl->frames.Release(&frame, true);
		return;
	}
#endif
#if defined(KYTY_LOCAL_VULKAN_RECORDING) && defined(_WIN32)
	// The frame's rendering may still sit in the recording worker's queue (KYTY_DEFERRED_SUBMIT).
	// On Windows, presenting before it reaches the driver blocks vkQueuePresentKHR in the
	// kernel while it holds the queue lock that submission needs: a deadlock. Wait (no locks
	// held) until it has been submitted; this does not wait for the GPU.
	LocalVulkanRecording::WaitDeferredSubmits();
#endif

	const auto overlay_visual = GetSystemOverlayVisualState();
	// The PC port's options menu (src/port): its frames go out without frame generation (no generated frame
	// drags the menu with the scene's motion).
	const bool port_menu      = PortLive::menu_open.load(std::memory_order_relaxed);
	auto&      swapchain  = m_impl->swapchain;
	auto*      hud        = m_impl->UpdateHud(reuse);
	// The guest's own frames choose the color space (PQ or SDR); black and blank flips keep the current one.
	if (!frame.blank && swapchain.SetHdr(frame.hdr)) {
		m_impl->RecoverSwapchain(Swapchain::Status::Recreate);
	}
	if (swapchain.Fsr3SurfaceChanged()) {
		LOGF("FSR3 FG: the window's size changed: swapchain re-created\n");
		std::printf("FSR3 FG: the window's size changed (fullscreen or resize): swapchain re-created\n");
		m_impl->RecoverSwapchain(Swapchain::Status::Recreate);
	}
	const bool timed      = FrameGen::Enabled();
	const auto flip_begin = std::chrono::steady_clock::now();
	for (uint32_t attempt = 0; attempt < 2; attempt++) {
		const auto acquire_begin = std::chrono::steady_clock::now();
		auto status = swapchain.AcquireNextImage();
		if (timed) FrameGen::NoteAcquireMs(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - acquire_begin).count());
		if (status != Swapchain::Status::Success) {
			m_impl->RecoverSwapchain(status);
			continue;
		}
		{
			// The present thread records and submits on its own scheduler (present_scheduler: its own pool, timeline
			// and command buffers; the queue has its own lock), from the presenter's frame image into the swapchain's:
			// nothing the GPU thread uses. Under the renderer's lock, its GPU thread waited for the whole recording
			// and submission once a frame (~60 us, ~110 us with the wake-up). Frame generation shares the renderer's.
			// (Frame generation's inputs are the frame's own copies: no renderer lock. Under it the GPU thread waited for
			// DLSS-G's present pacing: ~13 real fps.)
			const auto        submit_begin     = std::chrono::steady_clock::now();
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
			swapchain.SetKbmSeq(frame.kbm_seq);
			swapchain.RecordPresentCommands(command, frame.image, frame.fg.valid && !frame.reusing_last ? &frame.fg : nullptr,
			                                draw_system_overlay, frame.scene);
			// KYTY_FSR3_FG: the HUD as a layer FSR 3 composites over every frame shown, and the frame without it in the
			// backbuffer (generated frames then have no HUD to drag).
			if (swapchain.IsFsr3() && !draw_system_overlay)
				Fsr3::RecordHudLayer(m_impl->window.graphic_ctx, command.Handle(), swapchain.CurrentImage(), swapchain.Extent(),
				                     swapchain.PresentRegion(), frame.image,
				                     frame.fg.valid && !frame.reusing_last && !port_menu ? &frame.fg : nullptr);
			if (FrameGen::Enabled() && frame.fg.valid && !frame.reusing_last && !port_menu)
				FrameGen::ScaleHudless(m_impl->window.graphic_ctx, command.Handle(), frame.fg, swapchain.Extent(),
				                       swapchain.PresentRegion(), swapchain.Format());
			frame.present_tick = swapchain.Submit(m_impl->present_scheduler);
			if (timed) FrameGen::NoteSubmitMs(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - submit_begin).count());
		}
#if defined(_WIN32)
		// The Windows driver waits in vkQueuePresentKHR, inside the kernel, for the semaphore's
		// GPU work while it holds a device-wide lock: every queue submission waits behind it,
		// including those the GPU work may still depend on (a deadlock). Wait for the GPU here,
		// holding nothing; the present on its own queue then finds the semaphore signaled.
		{
			const auto wait_begin = std::chrono::steady_clock::now();
			m_impl->present_scheduler.Wait(frame.present_tick);
			if (timed) FrameGen::NoteGpuWaitMs(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wait_begin).count());
		}
#endif
		// KYTY_FRAMEGEN: Streamline takes the present (vkQueuePresentKHR is its hook) and adds
		// the generated frames; an idle refresh of the last frame carries no new inputs.
		const bool generate = FrameGen::Enabled();
		if (generate) {
			const auto before_begin = std::chrono::steady_clock::now();
			FrameGen::SetPresentReflexIndex(frame.reusing_last ? 0u : frame.fg.reflex_index);
			FrameGen::BeforePresent(frame.fg.valid && !frame.reusing_last && !port_menu ? &frame.fg : nullptr,
			                        swapchain.Extent(), swapchain.PresentRegion(), swapchain.Format(),
			                        swapchain.ImageCount());
			FrameGen::NoteBeforePresentMs(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before_begin).count());
		}
		if (swapchain.IsFsr3()) {
			// KYTY_FSR3_FG: this frame's depth and motion vectors (none: a menu, a loading screen, a movie, an idle
			// refresh of the last frame; generation off for it).
			Fsr3::BeforePresent(m_impl->window.graphic_ctx, frame.fg.valid && !frame.reusing_last && !port_menu ? &frame.fg : nullptr,
			                    swapchain.Handle(), swapchain.Extent(), swapchain.Format());
		}
		// KYTY_ANTILAG: the frame's PRESENT mark (a new guest frame only; with FSR 3 frame generation this is the real
		// frame going to its swapchain, never one it generates).
		const uint64_t latency_index = !reuse && !frame.reusing_last ? frame.latency_index : 0;
		if (latency_index != 0) {
			AntiLag::OnPresent(latency_index);
			frame.latency_index = 0;
		}
		swapchain.SetPresentId(AntiLag::PresentId(latency_index));
		status = swapchain.Present();
		swapchain.SetPresentId(0);
		AntiLag::AfterPresent(latency_index);
		if (status == Swapchain::Status::Success) AntiLag::OnPresented(latency_index); // KYTY_LATENCY: input -> present
		if (generate) {
			FrameGen::AfterPresent();
			FrameGen::NoteFlipPresentMs(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - flip_begin).count());
		}
		if (status != Swapchain::Status::Success) {
			m_impl->RecoverSwapchain(status);
			continue;
		}

		m_impl->presented_overlay_revision.store(overlay_visual.revision,
		                                         std::memory_order_release);
		if (PortLive::g_hooks != nullptr && PortLive::g_hooks->presented != nullptr) PortLive::g_hooks->presented();
		m_impl->window.UpdateTitle();
		m_impl->frames.Release(&frame, true);
		return;
	}
	LOGF("Vulkan presentation retry exhausted; dropping frame\n");
	m_impl->frames.Release(&frame, reuse);
}

void Presenter::Discard(Frame& frame) {
	m_impl->frames.Release(&frame);
}

WindowContext::WindowContext() = default;

} // namespace Libs::Graphics
