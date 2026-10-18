#include "frame-gen.h"
#include "fsr3-fg.h"
#include "xess-sr.h"
#include "graphics/presentation/videoOut.h"
#include <chrono>
#include "live-census.h"

#include "common/logging/log.h"
#include "local-platform.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include <tuple>
#include <unordered_map>
#include <utility>
#include "graphics/host_gpu/renderer/renderTarget.h"
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
#include "local/vulkan-recording.h"
#include "gpu_tiler_shaders/taa_bypass_spv.h"
#endif

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef KYTY_FRAMEGEN_STREAMLINE
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
// NVIDIA Streamline SDK headers (_Build/deps/streamline/sdk/include, see CMakeLists.txt).
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss.h>
#include <sl_dlss_g.h>
#include <sl_pcl.h>
#include <sl_reflex.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
#endif

namespace Libs::Graphics::FrameGen {

namespace {

struct Sources {
	ImageId  depth, motion;
	uint64_t depth_address = 0, motion_address = 0;
	bool     valid = false;
};

int      g_generate    = 0;     // frames to generate per rendered frame (0 = off)
bool g_dlss_sr = false; // KYTY_UPSCALE=dlss: DLSS super resolution (Streamline's DLSS plugin)
bool g_taa_dlss = false; // KYTY_TAA_REPLACE=dlss: DLSS (DLAA) in place of the game's TAA
// KYTY_TAA_REPLACE=upscaler (with KYTY_UPSCALE dlss|fsr|xess): the game's TAA skipped, its jittered image copied to
// the TAA's output, so the composite's upscaler gets the raw frame and does the anti-aliasing itself (on the TAA's
// output it ran a second temporal filter: smeared, "watercolor" images and ghosting).
bool g_taa_bypass = false;
// KYTY_TAA_BYPASS_MODE=copy: the first bypass (the TAA dispatch skipped, its output a copy of the raw frame: its alpha,
// which later passes read, lost: darker zones). Default: the TAA runs, then its color is replaced by the raw frame's
// (AfterTaa), its alpha kept. KYTY_TAA_BYPASS_GAIN=<f>: the raw frame times f there.
bool g_taa_bypass_copy = false;
// KYTY_TAA_BYPASS_MODE=match: the TAA's output replaced by the raw frame scaled to its mean brightness (10-17: the
// gain came out ~0.38 and the picture still darker: the raw frame's bright specks inflate the game's auto exposure).
// Default: the TAA's output is left to the game (its exposure, bloom and the rest see what they always saw) and only
// the upscaler takes the raw frame, a copy made right after the TAA (UpscaleComposite).
bool g_taa_bypass_match = false;
double g_real_interval = 0.0; // DLSS-G under the G-SYNC cap: the real frames' least interval (seconds)
bool g_sr_ready = false;
bool     g_ready       = false; // Streamline loaded and the feature available
Camera   g_camera;              // Kyty.Gpu: the current guest frame
Sources  g_sources;             // Kyty.Gpu: the current guest frame
uint64_t g_guest_frame = 0;     // Kyty.Gpu
std::atomic<bool> g_scene {false}; // the current guest frame ran the TAA pass
std::atomic<bool> g_last_scene {false}; // ... the last one the guest flipped
std::atomic<uint64_t> g_flips_taken {0}; // TakeScene calls (guest flips)

float AsFloat(uint32_t word) {
	float value = 0.0f;
	std::memcpy(&value, &word, sizeof(value));
	return value;
}

bool IsDepthFormat(vk::Format format) {
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

void Transition(vk::CommandBuffer command, VulkanImage& image, vk::ImageAspectFlags aspect,
                vk::ImageLayout layout, vk::AccessFlags2 access, vk::PipelineStageFlags2 stage) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = image.state.pl_stage;
	barrier.srcAccessMask       = image.state.access_mask;
	barrier.dstStageMask        = stage;
	barrier.dstAccessMask       = access;
	barrier.oldLayout           = image.state.layout;
	barrier.newLayout           = layout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image.image;
	barrier.subresourceRange    = {aspect, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image.state = {stage, access, layout};
}

// (Re)create `image` as a sampled copy target like `source`; with `attachment`, a color
// attachment too where the format allows it (*attachment tells).
void Configure(GraphicContext& graphics, VulkanImage& image, vk::ImageView& view,
               const VulkanImage& source, bool* attachment = nullptr) {
	if (image.image != nullptr && image.format == source.format &&
	    image.extent.width == source.extent.width && image.extent.height == source.extent.height) {
		return;
	}
	if (view != nullptr) {
		graphics.device.destroyImageView(view, nullptr);
		view = nullptr;
	}
	if (image.image != nullptr) {
		graphics.DeleteImage(image);
	}
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {source.extent.width, source.extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = source.format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled;
	if (attachment != nullptr) {
		*attachment = !IsDepthFormat(source.format) &&
		              static_cast<bool>(graphics.physical_device.getFormatProperties(source.format).optimalTilingFeatures &
		                                vk::FormatFeatureFlagBits::eColorAttachment);
		if (*attachment) {
			create.usage |= vk::ImageUsageFlagBits::eColorAttachment;
		}
	}
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, image)) {
		EXIT("frame generation: cannot allocate a %ux%u input image\n", source.extent.width,
		     source.extent.height);
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = image.image;
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = source.format;
	view_info.subresourceRange = {IsDepthFormat(source.format) ? vk::ImageAspectFlagBits::eDepth
	                                                           : vk::ImageAspectFlagBits::eColor,
	                              0, 1, 0, 1};
	RequireVulkanSuccess(graphics.device.createImageView(&view_info, nullptr, &view),
	                     "frame generation input view");
}

void CopyInto(vk::CommandBuffer command, Image& source, VulkanImage& target) {
	const auto aspect = IsDepthFormat(source.backing.format) ? vk::ImageAspectFlagBits::eDepth
	                                                         : vk::ImageAspectFlagBits::eColor;
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transition(command, target, aspect, vk::ImageLayout::eTransferDstOptimal,
	           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
	vk::ImageCopy copy {};
	copy.srcSubresource = {aspect, 0, 0, 1};
	copy.dstSubresource = {aspect, 0, 0, 1};
	copy.extent         = target.extent;
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, target.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy);
	Transition(command, target, aspect, vk::ImageLayout::eShaderReadOnlyOptimal,
	           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
}

// KYTY_FPS_HUD: frames shown per presented frame, and the panel's normalized rectangle (16 bits
// per coordinate: x0, y0, x1, y1; 0 = none).
std::atomic<uint32_t> g_presented_frames {1};
std::atomic<uint64_t> g_static_rect {0};

// No motion under the KYTY_FPS_HUD panel: DLSS-G would move it with the scene behind it.
void ZeroStaticRect(vk::CommandBuffer command, VulkanImage& motion, vk::ImageView view) {
	const auto packed = g_static_rect.load(std::memory_order_relaxed);
	if (packed == 0) {
		return;
	}
	const auto coordinate = [&](uint32_t index, uint32_t size) {
		return static_cast<int32_t>(((packed >> (16u * index)) & 0xffffu) * size / 0xffffu);
	};
	const int32_t x0 = coordinate(0, motion.extent.width), y0 = coordinate(1, motion.extent.height);
	const int32_t x1 = coordinate(2, motion.extent.width), y1 = coordinate(3, motion.extent.height);
	if (x1 <= x0 || y1 <= y0) {
		return;
	}
	Transition(command, motion, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal,
	           vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
	vk::RenderingAttachmentInfo color {};
	color.imageView   = view;
	color.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	color.loadOp      = vk::AttachmentLoadOp::eLoad;
	color.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = vk::Extent2D {motion.extent.width, motion.extent.height};
	rendering.layerCount           = 1;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &color;
	command.beginRendering(rendering);
	vk::ClearAttachment clear {};
	clear.aspectMask       = vk::ImageAspectFlagBits::eColor;
	clear.colorAttachment  = 0;
	clear.clearValue.color = vk::ClearColorValue {std::array<float, 4> {0.0f, 0.0f, 0.0f, 0.0f}};
	vk::ClearRect rect {};
	rect.rect       = vk::Rect2D {{x0, y0}, {static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0)}};
	rect.layerCount = 1;
	command.clearAttachments(1, &clear, 1, &rect);
	command.endRendering();
	Transition(command, motion, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
	           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
}

#ifdef KYTY_FRAMEGEN_STREAMLINE
// Streamline entry points (sl.interposer.dll, loaded at run time) and feature functions.
struct Api {
	PFun_slInit*                 init                = nullptr;
	PFun_slIsFeatureSupported*   is_feature_supported = nullptr;
	PFun_slIsFeatureLoaded*      is_feature_loaded   = nullptr;
	PFun_slGetFeatureFunction*   get_feature_function = nullptr;
	PFun_slGetNewFrameToken*     get_new_frame_token = nullptr;
	PFun_slSetConstants*         set_constants       = nullptr;
	PFun_slSetTagForFrame*       set_tag_for_frame   = nullptr;
	PFun_slDLSSGSetOptions*      dlssg_set_options   = nullptr;
	PFun_slDLSSGGetState*        dlssg_get_state     = nullptr;
	PFun_slReflexSetOptions*     reflex_set_options  = nullptr;
	PFun_slReflexSleep*          reflex_sleep        = nullptr;
	PFun_slPCLSetMarker*         pcl_set_marker      = nullptr;
	PFun_slEvaluateFeature*      evaluate            = nullptr;
	PFun_slDLSSSetOptions*       dlss_set_options    = nullptr;
};
Api              g_api;
sl::FrameToken*  g_token = nullptr; // present thread: the frame being presented
uint32_t         g_frame_index = 0;
bool             g_options_set = false;
vk::Extent2D     g_options_backbuffer {};
vk::Extent2D     g_options_render {};
vk::Format       g_options_format = vk::Format::eUndefined; // SDR or HDR10 swapchain
uint64_t         g_presents = 0;
int              g_mode_set = -1; // the DLSS-G mode last set (1 on, 0 off)
std::chrono::steady_clock::time_point g_stat_start {};
uint32_t         g_stat_real = 0, g_stat_on = 0;
double           g_stat_present_ms = 0, g_stat_present_max = 0;
double           g_stat_acquire_ms = 0, g_stat_acquire_max = 0, g_stat_gpuwait_ms = 0, g_stat_gpuwait_max = 0,
                 g_stat_flip_ms = 0, g_stat_flip_max = 0, g_stat_submit_ms = 0, g_stat_submit_max = 0,
                 g_stat_before_ms = 0, g_stat_before_max = 0;
std::chrono::steady_clock::time_point g_present_begin {};

// Matrices in Streamline's convention: row-major, row vectors (p' = p * M).
using Mat4 = sl::float4x4;

Mat4 Transposed(const std::array<float, 16>& m) { // our column-vector matrix -> SL
	Mat4 r;
	for (uint32_t i = 0; i < 4; ++i) {
		r[i] = sl::float4(m[i], m[4 + i], m[8 + i], m[12 + i]);
	}
	return r;
}

float At(const sl::float4& v, uint32_t i) {
	return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

Mat4 Multiply(const Mat4& a, const Mat4& b) {
	Mat4 r;
	for (uint32_t i = 0; i < 4; ++i) {
		float row[4] {};
		for (uint32_t j = 0; j < 4; ++j) {
			for (uint32_t k = 0; k < 4; ++k) row[j] += At(a[i], k) * At(b[k], j);
		}
		r[i] = sl::float4(row[0], row[1], row[2], row[3]);
	}
	return r;
}

bool Inverted(const Mat4& m, Mat4& out) {
	double a[4][8] {};
	for (uint32_t i = 0; i < 4; ++i) {
		for (uint32_t j = 0; j < 4; ++j) a[i][j] = At(m[i], j);
		a[i][4 + i] = 1.0;
	}
	for (uint32_t c = 0; c < 4; ++c) {
		uint32_t pivot = c;
		for (uint32_t r = c + 1; r < 4; ++r) {
			if (std::fabs(a[r][c]) > std::fabs(a[pivot][c])) pivot = r;
		}
		if (std::fabs(a[pivot][c]) < 1e-12) return false;
		for (uint32_t j = 0; j < 8; ++j) std::swap(a[c][j], a[pivot][j]);
		const double scale = 1.0 / a[c][c];
		for (uint32_t j = 0; j < 8; ++j) a[c][j] *= scale;
		for (uint32_t r = 0; r < 4; ++r) {
			if (r == c) continue;
			const double f = a[r][c];
			for (uint32_t j = 0; j < 8; ++j) a[r][j] -= f * a[c][j];
		}
	}
	for (uint32_t i = 0; i < 4; ++i) {
		out[i] = sl::float4(static_cast<float>(a[i][4]), static_cast<float>(a[i][5]),
		                    static_cast<float>(a[i][6]), static_cast<float>(a[i][7]));
	}
	return true;
}

// The game's world->clip matrix is P * V with a perspective P whose w row is (0, 0, 1, 0) (w =
// view depth) and whose z row is constant (reversed-Z, infinite far). Split it into camera
// axes, position and projection terms.
struct CameraFrame {
	sl::float3 right, up, forward, position;
	float      sx = 1.0f, sy = 1.0f, cx = 0.0f, cy = 0.0f, z_near = 0.1f;
	bool       valid = false;
};

CameraFrame Decompose(const std::array<float, 16>& m) {
	CameraFrame c;
	const auto row = [&](uint32_t i) { return std::array<double, 4> {m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]}; };
	const auto r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
	const double flen = std::sqrt(r3[0] * r3[0] + r3[1] * r3[1] + r3[2] * r3[2]);
	if (flen < 1e-6) return c;
	const double f[3] {r3[0] / flen, r3[1] / flen, r3[2] / flen};
	const double tf = r3[3] / flen;
	const auto split = [&](const std::array<double, 4>& r, double& s, double& off, double axis[3], double& t) {
		off = (r[0] * f[0] + r[1] * f[1] + r[2] * f[2]) / flen;
		double v[3] {r[0] / flen - off * f[0], r[1] / flen - off * f[1], r[2] / flen - off * f[2]};
		s = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		if (s < 1e-9) return false;
		for (int i = 0; i < 3; ++i) axis[i] = v[i] / s;
		t = (r[3] / flen - off * tf) / s;
		return true;
	};
	double sx = 0, cx = 0, sy = 0, cy = 0, x[3] {}, y[3] {}, tx = 0, ty = 0;
	if (!split(r0, sx, cx, x, tx) || !split(r1, sy, cy, y, ty)) return c;
	// Up = forward x right (Streamline's convention); a flipped clip y shows up as a negative sy.
	const double up[3] {f[1] * x[2] - f[2] * x[1], f[2] * x[0] - f[0] * x[2], f[0] * x[1] - f[1] * x[0]};
	if (up[0] * y[0] + up[1] * y[1] + up[2] * y[2] < 0) {
		sy = -sy;
		ty = -ty;
	}
	c.right    = sl::float3(float(x[0]), float(x[1]), float(x[2]));
	c.up       = sl::float3(float(up[0]), float(up[1]), float(up[2]));
	c.forward  = sl::float3(float(f[0]), float(f[1]), float(f[2]));
	// View translation (tx, ty, tf) = -axes . position.
	const double pos[3] {-(tx * x[0] + ty * up[0] + tf * f[0]), -(tx * x[1] + ty * up[1] + tf * f[1]),
	                     -(tx * x[2] + ty * up[2] + tf * f[2])};
	c.position = sl::float3(float(pos[0]), float(pos[1]), float(pos[2]));
	c.sx       = float(sx);
	c.sy       = float(sy);
	c.cx       = float(cx);
	c.cy       = float(cy);
	c.z_near   = float(r2[3] / flen);
	c.valid    = true;
	return c;
}

void LogMessage(sl::LogType type, const char* message) {
	LOGF("Streamline[%d]: %s", static_cast<int>(type), message);
	// KYTY_FRAMEGEN_VERBOSE: every message on stdout (the run log), not only errors.
	static const bool verbose = std::getenv("KYTY_FRAMEGEN_VERBOSE") != nullptr;
	if (type == sl::LogType::eError || verbose) {
		std::printf("Streamline %s: %s", type == sl::LogType::eError ? "error" : "log", message);
		std::fflush(stdout);
	}
}

template <typename F>
bool FeatureFunction(sl::Feature feature, const char* name, F*& function) {
	void* pointer = nullptr;
	if (g_api.get_feature_function(feature, name, pointer) != sl::Result::eOk || pointer == nullptr) {
		LOGF("frame generation: %s is unavailable\n", name);
		return false;
	}
	function = reinterpret_cast<F*>(pointer);
	return true;
}
#endif

} // namespace

bool Enabled() {
	return g_ready;
}

bool DlssUpscaleWanted();
bool TaaReplaceWanted();
bool Capturing() {
	return Enabled() || Fsr3::Active() || Fsr3::UpscaleWanted() || DlssUpscaleWanted() || Xess::Wanted() ||
	       TaaReplaceWanted();
}

bool Requested() {
	return g_generate > 0;
}

double RealFrameInterval() {
	return g_ready ? g_real_interval : 0.0;
}

bool StreamlineLoaded() {
	return g_generate > 0 || g_dlss_sr || g_taa_dlss;
}

int GeneratedFrames() {
	return g_generate;
}

#ifdef KYTY_FRAMEGEN_STREAMLINE
void NoteAcquireMs(double ms) {
	g_stat_acquire_ms += ms;
	g_stat_acquire_max = std::max(g_stat_acquire_max, ms);
}
void NoteGpuWaitMs(double ms) {
	g_stat_gpuwait_ms += ms;
	g_stat_gpuwait_max = std::max(g_stat_gpuwait_max, ms);
}
void NoteFlipPresentMs(double ms) {
	g_stat_flip_ms += ms;
	g_stat_flip_max = std::max(g_stat_flip_max, ms);
}
void NoteSubmitMs(double ms) {
	g_stat_submit_ms += ms;
	g_stat_submit_max = std::max(g_stat_submit_max, ms);
}
void NoteBeforePresentMs(double ms) {
	g_stat_before_ms += ms;
	g_stat_before_max = std::max(g_stat_before_max, ms);
}
#else
void NoteSubmitMs(double) {}
void NoteBeforePresentMs(double) {}
void NoteAcquireMs(double) {}
void NoteGpuWaitMs(double) {}
void NoteFlipPresentMs(double) {}
#endif

bool WantsCamera() {
	return g_ready && !g_camera.valid;
}

void BypassCommandHooks(vk::Device device) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// KYTY_FRAMEGEN_HOOKS=1 keeps Streamline's wrappers (for comparison).
	if ((g_generate == 0 && !g_dlss_sr && !g_taa_dlss) || std::getenv("KYTY_FRAMEGEN_HOOKS") != nullptr) {
		return;
	}
	// The real loader: Streamline itself forwards to it.
	HMODULE loader = GetModuleHandleW(L"vulkan-1.dll");
	auto*   get_device_proc = loader != nullptr ? reinterpret_cast<PFN_vkGetDeviceProcAddr>(
	                                                reinterpret_cast<void*>(GetProcAddress(loader, "vkGetDeviceProcAddr")))
	                                          : nullptr;
	if (get_device_proc == nullptr) {
		return;
	}
	auto&      dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;
	const auto native     = static_cast<VkDevice>(device);
	uint32_t   replaced   = 0;
	const auto direct     = [&](auto& function, const char* name) {
        if (auto* pointer = get_device_proc(native, name); pointer != nullptr) {
            function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(pointer);
            ++replaced;
        }
	};
	// Every per-command entry point the translation uses (10 000 draws and 3 000 dispatches a frame at the dragon's
	// path, each with its push constants, barriers and bindings): Streamline's wrappers only track state the emulator
	// does not need (eDisableCLStateTracking). Queue submission, presentation and the swapchain stay hooked.
	direct(dispatcher.vkBeginCommandBuffer, "vkBeginCommandBuffer");
	direct(dispatcher.vkEndCommandBuffer, "vkEndCommandBuffer");
	direct(dispatcher.vkCmdBindPipeline, "vkCmdBindPipeline");
	direct(dispatcher.vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets");
	direct(dispatcher.vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
	direct(dispatcher.vkCmdPipelineBarrier2, "vkCmdPipelineBarrier2");
	direct(dispatcher.vkCmdPushConstants, "vkCmdPushConstants");
	direct(dispatcher.vkCmdBindIndexBuffer, "vkCmdBindIndexBuffer");
	direct(dispatcher.vkCmdBindVertexBuffers, "vkCmdBindVertexBuffers");
	direct(dispatcher.vkCmdDraw, "vkCmdDraw");
	direct(dispatcher.vkCmdDrawIndexed, "vkCmdDrawIndexed");
	direct(dispatcher.vkCmdDrawIndirect, "vkCmdDrawIndirect");
	direct(dispatcher.vkCmdDrawIndexedIndirect, "vkCmdDrawIndexedIndirect");
	direct(dispatcher.vkCmdDispatch, "vkCmdDispatch");
	direct(dispatcher.vkCmdDispatchIndirect, "vkCmdDispatchIndirect");
	direct(dispatcher.vkCmdCopyBuffer, "vkCmdCopyBuffer");
	direct(dispatcher.vkCmdCopyBufferToImage, "vkCmdCopyBufferToImage");
	direct(dispatcher.vkCmdCopyImageToBuffer, "vkCmdCopyImageToBuffer");
	direct(dispatcher.vkCmdCopyImage, "vkCmdCopyImage");
	direct(dispatcher.vkCmdBlitImage, "vkCmdBlitImage");
	direct(dispatcher.vkCmdFillBuffer, "vkCmdFillBuffer");
	direct(dispatcher.vkCmdUpdateBuffer, "vkCmdUpdateBuffer");
	direct(dispatcher.vkCmdClearColorImage, "vkCmdClearColorImage");
	direct(dispatcher.vkCmdClearDepthStencilImage, "vkCmdClearDepthStencilImage");
	direct(dispatcher.vkCmdClearAttachments, "vkCmdClearAttachments");
	direct(dispatcher.vkCmdBeginRendering, "vkCmdBeginRendering");
	direct(dispatcher.vkCmdEndRendering, "vkCmdEndRendering");
	direct(dispatcher.vkCmdSetViewport, "vkCmdSetViewport");
	direct(dispatcher.vkCmdSetScissor, "vkCmdSetScissor");
	direct(dispatcher.vkCmdSetDepthBias, "vkCmdSetDepthBias");
	direct(dispatcher.vkCmdSetStencilReference, "vkCmdSetStencilReference");
	direct(dispatcher.vkCmdSetBlendConstants, "vkCmdSetBlendConstants");
	direct(dispatcher.vkCmdSetEvent, "vkCmdSetEvent");
	direct(dispatcher.vkCmdWaitEvents, "vkCmdWaitEvents");
	direct(dispatcher.vkCmdWriteTimestamp, "vkCmdWriteTimestamp");
	direct(dispatcher.vkCmdResetQueryPool, "vkCmdResetQueryPool");
	direct(dispatcher.vkCmdBeginQuery, "vkCmdBeginQuery");
	direct(dispatcher.vkCmdEndQuery, "vkCmdEndQuery");
	direct(dispatcher.vkCmdCopyQueryPoolResults, "vkCmdCopyQueryPoolResults");
	direct(dispatcher.vkCmdExecuteCommands, "vkCmdExecuteCommands");
	std::printf("Frame generation: %u command-buffer calls bypass Streamline\n", replaced);
#else
	(void)device;
#endif
}

void Initialize() {
	const char* value = std::getenv("KYTY_FRAMEGEN");
	g_generate        = value != nullptr ? std::clamp(std::atoi(value), 0, 5) : 0;
	{
		const char* upscale = std::getenv("KYTY_UPSCALE");
		g_dlss_sr           = upscale != nullptr && std::string(upscale) == "dlss";
		const char* taa     = std::getenv("KYTY_TAA_REPLACE");
		g_taa_dlss          = taa != nullptr && std::string(taa) == "dlss";
		g_taa_bypass        = taa != nullptr && std::string(taa) == "upscaler";
		if (const char* mode = std::getenv("KYTY_TAA_BYPASS_MODE"); mode != nullptr && std::string(mode) == "copy")
			g_taa_bypass_copy = true;
		// (Default: match. KYTY_TAA_BYPASS_MODE=raw: the TAA's output left to the game and the raw frame to the
		// upscaler alone: 10-17 the Nexus went black and lost the ghosts and glowing runes, drawn after the TAA.)
		g_taa_bypass_match = !(std::getenv("KYTY_TAA_BYPASS_MODE") != nullptr &&
		                       std::string(std::getenv("KYTY_TAA_BYPASS_MODE")) == "raw");
	}
	if (g_generate == 0 && !g_dlss_sr && !g_taa_dlss) {
		return;
	}
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// NVIDIA's Streamline runtime, which no release ships (the user's own copy: the Streamline SDK's bin\x64):
	// KYTY_STREAMLINE_DIR (the launcher's "Streamline..." folder, run-windows -StreamlineDir), else a "streamline"
	// folder next to the emulator, else the SDK unpacked beside the build (_Build\deps\streamline\sdk). A folder
	// may be the SDK's own (its bin\x64 is taken).
	std::vector<std::filesystem::path> candidates;
	wchar_t                            exe[MAX_PATH] {};
	GetModuleFileNameW(nullptr, exe, MAX_PATH);
	const auto exe_dir = std::filesystem::path(exe).parent_path();
	// (Wide: a folder's name in any language.)
	if (const wchar_t* env = _wgetenv(L"KYTY_STREAMLINE_DIR"); env != nullptr && *env != L'\0') {
		candidates.emplace_back(env);
	} else {
		candidates.push_back(exe_dir / "streamline");
		candidates.push_back(exe_dir.parent_path() / "deps" / "streamline" / "sdk");
	}
	std::filesystem::path dir;
	for (const auto& candidate: candidates) {
		for (const auto& folder: {candidate, candidate / "bin" / "x64"}) {
			std::error_code error;
			if (dir.empty() && std::filesystem::exists(folder / "sl.interposer.dll", error)) dir = folder;
		}
	}
	if (dir.empty()) {
		LOGF("frame generation: no sl.interposer.dll (KYTY_STREAMLINE_DIR or a streamline folder next to the emulator)\n");
		std::printf("Frame generation: NVIDIA's Streamline runtime was not found (choose its folder in the launcher)\n");
		g_generate = 0;
		g_dlss_sr  = false;
		g_taa_dlss = false;
		return;
	}
	const auto interposer = dir / "sl.interposer.dll";
	// Only NVIDIA's own (Streamline then checks its plugins' signatures itself).
	if (!LocalPlatform::IsNvidiaSigned(interposer.wstring())) {
		LOGF("frame generation: %s is not signed by NVIDIA: not loaded\n", interposer.string().c_str());
		std::printf("Frame generation: %s is not signed by NVIDIA, not loaded\n", interposer.string().c_str());
		g_generate = 0;
		g_dlss_sr  = false;
		g_taa_dlss = false;
		return;
	}
	if (std::getenv("KYTY_FRAMEGEN_VERBOSE") != nullptr) {
		// Graphics modules already loaded before Streamline (it must come first).
		for (const char* name: {"vulkan-1.dll", "nvoglv64.dll", "opengl32.dll", "dxgi.dll", "d3d11.dll", "d3d12.dll",
		                        "d3d9.dll", "nvapi64.dll", "nvcuda.dll", "nvofapi64.dll", "sl.interposer.dll"}) {
			if (GetModuleHandleA(name) != nullptr) std::printf("Frame generation: %s already loaded\n", name);
		}
	}
	HMODULE    module     = LoadLibraryW(interposer.c_str());
	if (module == nullptr) {
		LOGF("frame generation: cannot load %s\n", interposer.string().c_str());
		std::printf("Frame generation: cannot load %s\n", interposer.string().c_str());
		g_generate = 0;
		g_dlss_sr  = false;
		g_taa_dlss = false;
		return;
	}
	const auto load = [&](auto& function, const char* name) {
		function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
		    reinterpret_cast<void*>(GetProcAddress(module, name)));
		return function != nullptr;
	};
	if (!load(g_api.init, "slInit") || !load(g_api.is_feature_supported, "slIsFeatureSupported") ||
	    !load(g_api.is_feature_loaded, "slIsFeatureLoaded") ||
	    !load(g_api.get_feature_function, "slGetFeatureFunction") ||
	    !load(g_api.get_new_frame_token, "slGetNewFrameToken") ||
	    !load(g_api.set_constants, "slSetConstants") ||
	    !load(g_api.set_tag_for_frame, "slSetTagForFrame") || !load(g_api.evaluate, "slEvaluateFeature")) {
		LOGF("frame generation: sl.interposer.dll lacks an entry point\n");
		g_generate = 0;
		g_dlss_sr  = false;
		g_taa_dlss = false;
		return;
	}
	static const std::wstring plugin_dir = dir.wstring();
	static const std::wstring log_dir    = (dir.parent_path() / "logs").wstring();
	static const wchar_t*     plugin_paths[] {plugin_dir.c_str()};
	// KYTY_FRAMEGEN_FEATURES=dlssg,reflex,pcl: the plugins to load (a diagnostic subset).
	static std::vector<sl::Feature> features {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
	if (const char* list = std::getenv("KYTY_FRAMEGEN_FEATURES"); list != nullptr) {
		const std::string text = list;
		features.clear();
		if (text.find("dlssg") != std::string::npos) features.push_back(sl::kFeatureDLSS_G);
		if (text.find("reflex") != std::string::npos) features.push_back(sl::kFeatureReflex);
		if (text.find("pcl") != std::string::npos) features.push_back(sl::kFeaturePCL);
	}
	// DLSS super resolution (KYTY_UPSCALE=dlss, KYTY_TAA_REPLACE=dlss): its plugin too. Without frame generation the
	// same plugins are loaded all the same (DLSS-G then stays off): with only the DLSS plugin the emulator closed as the
	// Vulkan device was made (logs 005353, 013603), with the frame generation set it ran.
	if (g_dlss_sr || g_taa_dlss) features.push_back(sl::kFeatureDLSS);
	sl::Preferences           preferences {};
	preferences.showConsole        = false;
	preferences.logLevel           = std::getenv("KYTY_FRAMEGEN_VERBOSE") ? sl::LogLevel::eVerbose
	                                                                      : sl::LogLevel::eDefault;
	preferences.pathsToPlugins     = plugin_paths;
	preferences.numPathsToPlugins  = 1;
	preferences.pathToLogsAndData  = nullptr;
	preferences.logMessageCallback = LogMessage;
	preferences.flags              = sl::PreferenceFlags::eDisableCLStateTracking |
	                    sl::PreferenceFlags::eUseFrameBasedResourceTagging;
	preferences.featuresToLoad    = features.data();
	preferences.numFeaturesToLoad = static_cast<uint32_t>(features.size());
	preferences.engine            = sl::EngineType::eCustom;
	preferences.engineVersion     = "1.0";
	preferences.projectId         = "6a3f1d5e-4b7c-4f2a-9d8e-1c2b3a4d5e6f";
	preferences.renderAPI         = sl::RenderAPI::eVulkan;
	if (const auto result = g_api.init(preferences, sl::kSDKVersion); result != sl::Result::eOk) {
		LOGF("frame generation: slInit failed (%d)\n", static_cast<int>(result));
		std::printf("Frame generation: slInit failed (%d)\n", static_cast<int>(result));
		g_generate = 0;
		g_dlss_sr  = false;
		g_taa_dlss = false;
		return;
	}
	// SDL loads the interposer as the Vulkan loader: every instance/device/swapchain call then
	// goes through Streamline, which adds what DLSS-G needs.
	SetEnvironmentVariableW(L"SDL_VULKAN_LIBRARY", interposer.c_str());
	_putenv_s("SDL_VULKAN_LIBRARY", interposer.string().c_str());
	LOGF("frame generation: Streamline loaded from %s, %d generated frame(s)\n",
	     dir.string().c_str(), g_generate);
#else
	LOGF("frame generation: this build has no Streamline support\n");
	g_generate = 0;
		g_dlss_sr  = false;
		g_taa_dlss = false;
#endif
}

void OnDevice(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device) {
	(void)instance;
	(void)device;
	if (g_generate == 0 && !g_dlss_sr && !g_taa_dlss) {
		return;
	}
#ifdef KYTY_FRAMEGEN_STREAMLINE
	sl::AdapterInfo adapter {};
	adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
	bool loaded              = false;
	if (g_dlss_sr || g_taa_dlss) {
		bool dlss_loaded = false;
		if (g_api.is_feature_supported(sl::kFeatureDLSS, adapter) != sl::Result::eOk ||
		    g_api.is_feature_loaded(sl::kFeatureDLSS, dlss_loaded) != sl::Result::eOk || !dlss_loaded ||
		    !FeatureFunction(sl::kFeatureDLSS, "slDLSSSetOptions", g_api.dlss_set_options)) {
			std::printf("Upscaling: DLSS is not supported on this device: upscaling off\n");
		} else {
			g_sr_ready = true;
			std::printf("Upscaling: DLSS super resolution ready\n");
		}
		std::fflush(stdout);
	}
	if (g_generate == 0) {
		return;
	}
	if (g_api.is_feature_supported(sl::kFeatureDLSS_G, adapter) != sl::Result::eOk ||
	    g_api.is_feature_loaded(sl::kFeatureDLSS_G, loaded) != sl::Result::eOk || !loaded) {
		LOGF("frame generation: DLSS-G is not supported on this device\n");
		std::printf("Frame generation: DLSS-G is not supported on this device\n");
		return;
	}
	if (!FeatureFunction(sl::kFeatureDLSS_G, "slDLSSGSetOptions", g_api.dlssg_set_options) ||
	    !FeatureFunction(sl::kFeatureDLSS_G, "slDLSSGGetState", g_api.dlssg_get_state) ||
	    !FeatureFunction(sl::kFeatureReflex, "slReflexSetOptions", g_api.reflex_set_options) ||
	    !FeatureFunction(sl::kFeaturePCL, "slPCLSetMarker", g_api.pcl_set_marker)) {
		return;
	}
	// slReflexSleep each frame (KYTY_DLSSG_SLEEP=0: not): Reflex is then active, as DLSS-G requires; without it its
	// pacer held every present ~25 ms (the game at ~37 fps), and with Reflex off the game reached ~56 but DLSS-G
	// reported Reflex missing (a tinted picture).
	FeatureFunction(sl::kFeatureReflex, "slReflexSleep", g_api.reflex_sleep);
	sl::ReflexOptions reflex {};
	reflex.mode = sl::ReflexMode::eLowLatency;
	// KYTY_DLSSG_REFLEX=off|low|boost (a comparison of DLSS-G's pacing).
	if (const char* mode = std::getenv("KYTY_DLSSG_REFLEX"); mode != nullptr) {
		if (std::string(mode) == "off") reflex.mode = sl::ReflexMode::eOff;
		if (std::string(mode) == "boost") reflex.mode = sl::ReflexMode::eLowLatencyWithBoost;
	}
	// G-SYNC: the frames shown (rendered x (1 + generated)) must stay under the monitor's refresh, or the ones past it
	// leave the variable refresh range (a doubled / jumping picture at 227-228 of 240 Hz). Reflex's limiter holds the
	// rendered frames to cap / (1 + generated), the way NVIDIA's own G-SYNC + Reflex cap does (240 Hz: 224 shown,
	// 56 rendered at x4). KYTY_VRR_CAP_FPS=<shown fps> or auto (refresh - refresh^2 / 3600); unset: off.
	{
		uint32_t cap = 0;
		const char* text = std::getenv("KYTY_VRR_CAP_FPS");
		if (text == nullptr || *text == '\0') {
			cap = 0; // off unless asked for: no frame limit by default
		} else if (std::string(text) != "auto") {
			cap = static_cast<uint32_t>(std::strtoul(text, nullptr, 10));
		} else {
			DEVMODEA mode {};
			mode.dmSize = sizeof(mode);
			if (EnumDisplaySettingsA(nullptr, ENUM_CURRENT_SETTINGS, &mode) != 0 && mode.dmDisplayFrequency > 30) {
				const double hz = mode.dmDisplayFrequency;
				cap             = static_cast<uint32_t>(hz - hz * hz / 3600.0);
			}
		}
		if (cap != 0 && g_generate > 0) {
			reflex.frameLimitUs = static_cast<uint32_t>(1000000.0 * (g_generate + 1) / cap);
			g_real_interval     = static_cast<double>(g_generate + 1) / cap;
			std::printf("Frame generation: shown frames held under %u fps (rendered at most %.1f, Reflex limit %u us)\n", cap,
			            static_cast<double>(cap) / (g_generate + 1), reflex.frameLimitUs);
		}
	}
	std::printf("Frame generation: Reflex mode %d\n", static_cast<int>(reflex.mode));
	if (const auto result = g_api.reflex_set_options(reflex); result != sl::Result::eOk) {
		LOGF("frame generation: slReflexSetOptions failed (%d)\n", static_cast<int>(result));
	}
	g_ready = true;
	LOGF("frame generation: DLSS-G ready\n");
	std::printf("Frame generation: DLSS-G ready (%d generated frame(s) per frame)\n", g_generate);
	std::fflush(stdout);
#endif
}

#ifdef KYTY_FRAMEGEN_STREAMLINE
sl::Constants MakeConstants(bool usable, const Camera& camera_in, vk::Extent2D render) {
	sl::Constants constants {};
	const auto    camera = usable ? Decompose(camera_in.view_proj) : CameraFrame {};
	Mat4          identity;
	for (uint32_t i = 0; i < 4; ++i) {
		identity[i] = sl::float4(i == 0, i == 1, i == 2, i == 3);
	}
	constants.cameraViewToClip = identity;
	constants.clipToCameraView = identity;
	constants.clipToLensClip   = identity;
	constants.clipToPrevClip   = identity;
	constants.prevClipToClip   = identity;
	constants.jitterOffset     = sl::float2(0.0f, 0.0f);
	constants.mvecScale        = sl::float2(1.0f, 1.0f);
	constants.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
	constants.cameraPos        = sl::float3(0, 0, 0);
	constants.cameraUp         = sl::float3(0, 1, 0);
	constants.cameraRight      = sl::float3(1, 0, 0);
	constants.cameraFwd        = sl::float3(0, 0, 1);
	constants.cameraNear       = 0.1f;
	constants.cameraFar        = 10000.0f;
	constants.cameraFOV        = 1.0f;
	constants.cameraAspectRatio = 16.0f / 9.0f;
	constants.motionVectorsInvalidValue = sl::INVALID_FLOAT;
	constants.depthInverted       = sl::Boolean::eTrue;
	constants.cameraMotionIncluded = sl::Boolean::eTrue;
	constants.motionVectors3D     = sl::Boolean::eFalse;
	constants.reset               = usable ? sl::Boolean::eFalse : sl::Boolean::eTrue;
	constants.orthographicProjection = sl::Boolean::eFalse;
	constants.motionVectorsDilated   = sl::Boolean::eFalse;
	constants.motionVectorsJittered  = sl::Boolean::eFalse;
	if (usable && camera.valid) {
		// Projection in view space: x' = sx*x + cx*z, y' = sy*y + cy*z, z' = near, w' = z.
		Mat4 projection;
		projection[0] = sl::float4(camera.sx, 0, 0, 0);
		projection[1] = sl::float4(0, camera.sy, 0, 0);
		projection[2] = sl::float4(camera.cx, camera.cy, 0, 1);
		projection[3] = sl::float4(0, 0, camera.z_near, 0);
		constants.cameraViewToClip = projection;
		Inverted(projection, constants.clipToCameraView);
		const auto current  = Transposed(camera_in.view_proj);
		const auto previous = Transposed(camera_in.prev_view_proj);
		Mat4       clip_to_world;
		if (Inverted(current, clip_to_world)) {
			constants.clipToPrevClip = Multiply(clip_to_world, previous);
			Inverted(constants.clipToPrevClip, constants.prevClipToClip);
		}
		const float width  = static_cast<float>(render.width);
		const float height = static_cast<float>(render.height);
		constants.jitterOffset = sl::float2(camera_in.jitter_ndc[0] * width * 0.5f,
		                                    -camera_in.jitter_ndc[1] * height * 0.5f);
		static const bool pixels_mv = [] {
			const char* mv = std::getenv("KYTY_FRAMEGEN_MV");
			return mv != nullptr && std::string(mv) == "px";
		}();
		// The game's vectors point along the motion, in UV units: its TAA reads the history at
		// uv - mv. DLSS wants them towards the previous position (uv + mv = previous uv), so they
		// are negated; with them as they were, generated frames moved the scene the wrong way
		// when the camera turned. KYTY_FRAMEGEN_MV_SIGN=1 keeps the game's sign (a comparison).
		static const float mv_sign = [] {
			const char* sign = std::getenv("KYTY_FRAMEGEN_MV_SIGN");
			return sign != nullptr && std::atof(sign) > 0 ? 1.0f : -1.0f;
		}();
		constants.mvecScale = pixels_mv ? sl::float2(mv_sign / width, mv_sign / height) : sl::float2(mv_sign, mv_sign);
		constants.cameraPos   = camera.position;
		constants.cameraUp    = camera.up;
		constants.cameraRight = camera.right;
		constants.cameraFwd   = camera.forward;
		constants.cameraNear  = camera.z_near;
		constants.cameraFar   = 100000.0f;
		constants.cameraFOV   = 2.0f * std::atan(1.0f / std::fabs(camera.sy));
		constants.cameraAspectRatio = std::fabs(camera.sy / camera.sx);
	}
	return constants;
}

// DLSS super resolution of the composite's 3D image (UpscaleComposite): viewport 1, its own frame indices.
// The images' layouts as given (the composite path: color, depth and motion SHADER_READ_ONLY, output GENERAL; the TAA
// path: as the TAA dispatch's descriptors have them). `taa`: in place of the game's TAA (viewport 2): its color is
// jittered (the camera's jitter given), its output the TAA's.
struct DlssLayouts {
	vk::ImageLayout color  = vk::ImageLayout::eShaderReadOnlyOptimal;
	vk::ImageLayout depth  = vk::ImageLayout::eShaderReadOnlyOptimal;
	vk::ImageLayout motion = vk::ImageLayout::eShaderReadOnlyOptimal;
	vk::ImageLayout output = vk::ImageLayout::eGeneral;
};
bool DlssEvaluate(VkCommandBuffer command, const VulkanImage& color, vk::ImageView color_view, const VulkanImage& depth,
                  vk::ImageView depth_view, const VulkanImage& motion, vk::ImageView motion_view, const VulkanImage& output,
                  vk::ImageView output_view, vk::Extent2D render, vk::Extent2D upscale, const Camera& camera, bool reset,
                  bool taa, const DlssLayouts& layouts, bool composite = false);
bool DlssUpscale(VkCommandBuffer command, const VulkanImage& color, vk::ImageView color_view, const VulkanImage& depth,
                 vk::ImageView depth_view, const VulkanImage& motion, vk::ImageView motion_view, const VulkanImage& output,
                 vk::ImageView output_view, vk::Extent2D render, vk::Extent2D upscale, const Camera& camera, bool reset) {
	// (KYTY_TAA_REPLACE=upscaler: the color is the raw jittered frame: the camera's jitter given, as in the TAA's slot.)
	return DlssEvaluate(command, color, color_view, depth, depth_view, motion, motion_view, output, output_view, render,
	                    upscale, camera, reset, g_taa_bypass, DlssLayouts {}, true);
}
bool DlssEvaluate(VkCommandBuffer command, const VulkanImage& color, vk::ImageView color_view, const VulkanImage& depth,
                  vk::ImageView depth_view, const VulkanImage& motion, vk::ImageView motion_view, const VulkanImage& output,
                  vk::ImageView output_view, vk::Extent2D render, vk::Extent2D upscale, const Camera& camera, bool reset,
                  bool taa, const DlssLayouts& layouts, bool composite) {
	if (!g_sr_ready || g_api.dlss_set_options == nullptr || g_api.evaluate == nullptr) return false;
	// `taa`: the color is jittered (the camera's jitter given); `composite`: the composite's upscale (viewport 1).
	const bool               taa_slot = taa && !composite;
	const sl::ViewportHandle viewport {taa_slot ? 2u : 1u};
	static vk::Extent2D      set_render[2] {}, set_output[2] {};
	const int                slot = taa_slot ? 1 : 0;
	if (set_render[slot] != render || set_output[slot] != upscale) {
		const double ratio = static_cast<double>(upscale.height) / render.height;
		sl::DLSSOptions options {};
		// (NGX refuses DLSS's ultra quality mode, 0xbad00010 "unsupported parameter" on every evaluate: 1.3x renders
		// in the quality mode, whose render size range reaches up to the output size.)
		options.mode = ratio < 1.05   ? sl::DLSSMode::eDLAA
		               : ratio < 1.6  ? sl::DLSSMode::eMaxQuality
		               : ratio < 1.85 ? sl::DLSSMode::eBalanced
		               : ratio < 2.5  ? sl::DLSSMode::eMaxPerformance
		                              : sl::DLSSMode::eUltraPerformance;
		options.outputWidth     = upscale.width;
		options.outputHeight    = upscale.height;
		options.colorBuffersHDR = sl::Boolean::eTrue;
		options.useAutoExposure = sl::Boolean::eTrue;
		// KYTY_DLSS_PRESET=K (DLSS 4, transformer), L or M (DLSS 4.5), J; default: NVIDIA's choice per mode.
		sl::DLSSPreset preset = sl::DLSSPreset::eDefault;
		if (const char* text = std::getenv("KYTY_DLSS_PRESET"); text != nullptr && *text != '\0') {
			switch (*text) {
				case 'J': case 'j': preset = sl::DLSSPreset::ePresetJ; break;
				case 'K': case 'k': preset = sl::DLSSPreset::ePresetK; break;
				case 'L': case 'l': preset = sl::DLSSPreset::ePresetL; break;
				case 'M': case 'm': preset = sl::DLSSPreset::ePresetM; break;
				default: break;
			}
		}
		options.dlaaPreset = options.qualityPreset = options.balancedPreset = options.performancePreset =
		    options.ultraPerformancePreset = options.ultraQualityPreset = preset;
		if (const auto result = g_api.dlss_set_options(viewport, options); result != sl::Result::eOk) {
			std::printf("Upscaling: slDLSSSetOptions failed (%d): upscaling off\n", static_cast<int>(result));
			std::fflush(stdout);
			g_sr_ready = false;
			return false;
		}
		set_render[slot] = render;
		set_output[slot] = upscale;
		std::printf("Upscaling: DLSS %ux%u -> %ux%u, mode %u, preset %u\n", render.width, render.height, upscale.width,
		            upscale.height, static_cast<uint32_t>(options.mode), static_cast<uint32_t>(preset));
		std::fflush(stdout);
	}
	static uint32_t indices[2] {};
	uint32_t        index = ++indices[slot];
	sl::FrameToken* token = nullptr;
	if (g_api.get_new_frame_token(token, &index) != sl::Result::eOk || token == nullptr) return false;
	auto constants = MakeConstants(camera.valid, camera, render);
	// After the game's TAA its jitter is resolved in the image; in its place the color is jittered (MakeConstants: the
	// camera's, KYTY_TAA_JITTER_SIGN=-1 the other way round).
	if (!taa) {
		constants.jitterOffset = sl::float2(0.0f, 0.0f);
	} else {
		static const float jitter_sign = [] {
			const char* value = std::getenv("KYTY_TAA_JITTER_SIGN");
			return value != nullptr && std::atof(value) < 0 ? -1.0f : 1.0f;
		}();
		constants.jitterOffset = sl::float2(jitter_sign * constants.jitterOffset.x, jitter_sign * constants.jitterOffset.y);
	}
	constants.reset        = reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
	if (!camera.valid) {
		// (No camera this frame: the vectors as they are, in UV units towards the motion.)
		static const float mv_sign = [] {
			const char* sign = std::getenv("KYTY_FRAMEGEN_MV_SIGN");
			return sign != nullptr && std::atof(sign) > 0 ? 1.0f : -1.0f;
		}();
		constants.mvecScale = sl::float2(mv_sign, mv_sign);
	}
	if (g_api.set_constants(constants, *token, viewport) != sl::Result::eOk) return false;
	const auto resource = [](const VulkanImage& image, vk::ImageView view, vk::ImageLayout layout) {
		sl::Resource r(sl::ResourceType::eTex2d, static_cast<VkImage>(image.image), nullptr, static_cast<VkImageView>(view),
		               static_cast<uint32_t>(layout));
		r.width        = image.extent.width;
		r.height       = image.extent.height;
		r.nativeFormat = static_cast<uint32_t>(image.format);
		r.mipLevels    = 1;
		r.arrayLayers  = 1;
		r.flags        = 0;
		r.usage        = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image.usage));
		return r;
	};
	auto in_color  = resource(color, color_view, layouts.color);
	auto in_depth  = resource(depth, depth_view, layouts.depth);
	auto in_motion = resource(motion, motion_view, layouts.motion);
	auto out_color = resource(output, output_view, layouts.output);
	const sl::Extent render_extent {0, 0, render.width, render.height};
	const sl::Extent output_extent {0, 0, upscale.width, upscale.height};
	const sl::ResourceTag tags[] {
	    sl::ResourceTag(&in_color, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &render_extent),
	    sl::ResourceTag(&in_depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilEvaluate, &render_extent),
	    sl::ResourceTag(&in_motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilEvaluate, &render_extent),
	    sl::ResourceTag(&out_color, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &output_extent),
	};
	auto* cmd = reinterpret_cast<sl::CommandBuffer*>(command);
	if (g_api.set_tag_for_frame(*token, viewport, tags, static_cast<uint32_t>(std::size(tags)), cmd) != sl::Result::eOk)
		return false;
	const sl::BaseStructure* inputs[] {&viewport};
	static uint32_t failures = 0;
	if (const auto result = g_api.evaluate(sl::kFeatureDLSS, *token, inputs, 1, cmd); result != sl::Result::eOk) {
		if (failures < 4) {
			std::printf("Upscaling: DLSS evaluate failed (%d)\n", static_cast<int>(result));
			std::fflush(stdout);
		}
		// Each failed evaluate tries to create the NGX feature again (slow): off after 30 in a row.
		if (++failures >= 30) {
			std::printf("Upscaling: DLSS keeps failing: off (the game scales its 3D image itself)\n");
			std::fflush(stdout);
			g_sr_ready = false;
		}
		return false;
	}
	failures = 0;
	return true;
}
#endif

bool DlssUpscaleWanted() {
	return g_dlss_sr;
}

bool TaaReplaceWanted() {
	return g_taa_dlss || g_taa_bypass;
}


void BeforePresent(const FrameInputs* inputs, vk::Extent2D backbuffer, vk::Rect2D region,
                   vk::Format format, uint32_t image_count) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!g_ready) {
		return;
	}
	const sl::ViewportHandle viewport {0};
	// Movies, menus and loading screens (no TAA pass) are presented as they are: DLSS-G off for them.
	const bool               usable = inputs != nullptr && inputs->valid && !VideoOut::MoviePlaying();
	{
		const auto now = std::chrono::steady_clock::now();
		if (g_stat_start == std::chrono::steady_clock::time_point {}) g_stat_start = now;
		++g_stat_real;
		if (usable) ++g_stat_on;
		const double window = std::chrono::duration<double>(now - g_stat_start).count();
		if (window >= 1.0) {
			const double n = g_stat_real != 0 ? static_cast<double>(g_stat_real) : 1.0;
			std::printf("[dlssg] real %.1f fps, shown %.1f fps; generation on %u of %u frames; present call %.2f ms (max %.1f); "
			            "acquire %.2f (max %.1f), record+submit %.2f (max %.1f), gpu wait %.2f (max %.1f), before %.2f (max %.1f), flip present %.2f (max %.1f)\n",
			            g_stat_real / window, g_stat_real / window * static_cast<double>(PresentedFrames()), g_stat_on,
			            g_stat_real, g_stat_present_ms / n, g_stat_present_max, g_stat_acquire_ms / n, g_stat_acquire_max,
			            g_stat_submit_ms / n, g_stat_submit_max, g_stat_gpuwait_ms / n, g_stat_gpuwait_max, g_stat_before_ms / n,
			            g_stat_before_max, g_stat_flip_ms / n, g_stat_flip_max);
			std::fflush(stdout);
			g_stat_start = now;
			g_stat_real  = 0;
			g_stat_on    = 0;
			g_stat_present_ms = g_stat_present_max = 0;
			g_stat_acquire_ms = g_stat_acquire_max = g_stat_gpuwait_ms = g_stat_gpuwait_max = g_stat_flip_ms = g_stat_flip_max = 0;
			g_stat_submit_ms = g_stat_submit_max = g_stat_before_ms = g_stat_before_max = 0;
		}
	}
	// DLSS-G options: on while frames carry inputs; sizes are hints for its allocations.
	if (!g_options_set || g_mode_set != (usable ? 1 : 0) || g_options_format != format ||
	    (usable && (g_options_backbuffer != backbuffer || g_options_render != inputs->render_extent))) {
		sl::DLSSGOptions options {};
		options.mode                = usable ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
		if (g_mode_set != (usable ? 1 : 0)) {
			std::printf("DLSS-G: %s\n", usable ? "on (TAA pass)" : "off (no TAA pass: menu, loading or movie)");
			std::fflush(stdout);
		}
		g_mode_set = usable ? 1 : 0;
		options.numFramesToGenerate = static_cast<uint32_t>(g_generate);
		// KYTY_DLSSG_PARALLEL=1: DLSS-G's work blocks none of the emulator's queues (a comparison: by default it
		// blocks the presenting queue while it works).
		static const bool parallel = [] {
			const char* value = std::getenv("KYTY_DLSSG_PARALLEL");
			return value != nullptr && *value == '1';
		}();
		if (parallel) options.queueParallelismMode = sl::DLSSGQueueParallelismMode::eBlockNoClientQueues;
		options.numBackBuffers      = image_count;
		options.colorWidth          = backbuffer.width;
		options.colorHeight         = backbuffer.height;
		options.colorBufferFormat   = static_cast<uint32_t>(format);
		if (usable) {
			options.mvecDepthWidth    = inputs->render_extent.width;
			options.mvecDepthHeight   = inputs->render_extent.height;
			options.mvecBufferFormat  = static_cast<uint32_t>(inputs->motion.format);
			options.depthBufferFormat = static_cast<uint32_t>(inputs->depth.format);
			g_options_render          = inputs->render_extent;
		}
		if (const auto result = g_api.dlssg_set_options(viewport, options); result != sl::Result::eOk) {
			LOGF("frame generation: slDLSSGSetOptions failed (%d)\n", static_cast<int>(result));
		}
		g_options_set        = true;
		g_options_backbuffer = backbuffer;
		g_options_format     = format;
	}

	++g_frame_index;
	if (g_api.get_new_frame_token(g_token, &g_frame_index) != sl::Result::eOk || g_token == nullptr) {
		g_token = nullptr;
		return;
	}
	auto& token = *g_token;
	static const bool sleep = [] {
		const char* value = std::getenv("KYTY_DLSSG_SLEEP");
		return value == nullptr || *value != '0';
	}();
	if (sleep && g_api.reflex_sleep != nullptr) g_api.reflex_sleep(token);
	g_api.pcl_set_marker(sl::PCLMarker::eSimulationStart, token);
	g_api.pcl_set_marker(sl::PCLMarker::eSimulationEnd, token);
	g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitStart, token);

	sl::Constants constants = MakeConstants(usable, usable ? inputs->camera : Camera {}, usable ? inputs->render_extent : vk::Extent2D {});
	if (const auto result = g_api.set_constants(constants, token, viewport); result != sl::Result::eOk &&
	                                                                          g_presents < 8) {
		LOGF("frame generation: slSetConstants failed (%d)\n", static_cast<int>(result));
	}

	if (usable) {
		sl::Resource depth(sl::ResourceType::eTex2d, static_cast<VkImage>(inputs->depth.image), nullptr,
		                   static_cast<VkImageView>(inputs->depth_view),
		                   static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
		depth.width        = inputs->depth.extent.width;
		depth.height       = inputs->depth.extent.height;
		depth.nativeFormat = static_cast<uint32_t>(inputs->depth.format);
		depth.mipLevels    = 1;
		depth.arrayLayers  = 1;
		depth.flags        = 0;
		depth.usage        = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(inputs->depth.usage));
		sl::Resource motion(sl::ResourceType::eTex2d, static_cast<VkImage>(inputs->motion.image), nullptr,
		                    static_cast<VkImageView>(inputs->motion_view),
		                    static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
		motion.width        = inputs->motion.extent.width;
		motion.height       = inputs->motion.extent.height;
		motion.nativeFormat = static_cast<uint32_t>(inputs->motion.format);
		motion.mipLevels    = 1;
		motion.arrayLayers  = 1;
		motion.flags        = 0;
		motion.usage = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(inputs->motion.usage));
		const sl::Extent full {0, 0, inputs->render_extent.width, inputs->render_extent.height};
		const sl::Extent game {static_cast<uint32_t>(region.offset.y), static_cast<uint32_t>(region.offset.x),
		                       region.extent.width, region.extent.height};
		// The backbuffer is tagged only when the game image covers part of it (KYTY_PRESENT_ASPECT).
		const bool      partial = region.offset.x != 0 || region.offset.y != 0 ||
		                     region.extent.width != backbuffer.width || region.extent.height != backbuffer.height;
		// The HUD-less image (the flip image before the HUD), at the game image's size: DLSS-G keeps the HUD out.
		sl::Resource hudless(sl::ResourceType::eTex2d, static_cast<VkImage>(inputs->hudless_display.image), nullptr,
		                     static_cast<VkImageView>(inputs->hudless_display_view),
		                     static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
		hudless.width        = inputs->hudless_display.extent.width;
		hudless.height       = inputs->hudless_display.extent.height;
		hudless.nativeFormat = static_cast<uint32_t>(inputs->hudless_display.format);
		hudless.mipLevels    = 1;
		hudless.arrayLayers  = 1;
		hudless.flags        = 0;
		hudless.usage        = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(inputs->hudless_display.usage));
		const sl::Extent hudless_extent {0, 0, inputs->hudless_display.extent.width, inputs->hudless_display.extent.height};
		std::vector<sl::ResourceTag> tags {
		    sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &full),
		    sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent,
		                    &full),
		};
		if (partial)
			tags.emplace_back(nullptr, sl::kBufferTypeBackbuffer, sl::ResourceLifecycle::eValidUntilPresent, &game);
		static const bool hudless_tag = [] {
			const char* value = std::getenv("KYTY_FRAMEGEN_HUDLESS");
			return value == nullptr || *value != '0';
		}();
		if (hudless_tag && inputs->has_hudless_display && inputs->hudless_display.extent.width == backbuffer.width &&
		    inputs->hudless_display.extent.height == backbuffer.height)
			tags.emplace_back(&hudless, sl::kBufferTypeHUDLessColor, sl::ResourceLifecycle::eValidUntilPresent,
			                  &hudless_extent);
		if (const auto result = g_api.set_tag_for_frame(token, viewport, tags.data(), static_cast<uint32_t>(tags.size()), nullptr);
		    result != sl::Result::eOk && g_presents < 8) {
			LOGF("frame generation: slSetTagForFrame failed (%d)\n", static_cast<int>(result));
		}
	} else {
		// Menus, movies, loading: no valid depth and motion (DLSS-G then presents the frame as is).
		sl::ResourceTag tags[] {
		    sl::ResourceTag(nullptr, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent),
		    sl::ResourceTag(nullptr, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent),
		};
		g_api.set_tag_for_frame(token, viewport, tags, static_cast<uint32_t>(std::size(tags)), nullptr);
	}
	g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitEnd, token);
	g_api.pcl_set_marker(sl::PCLMarker::ePresentStart, token);
	g_present_begin = std::chrono::steady_clock::now();
#else
	(void)inputs;
	(void)backbuffer;
	(void)region;
	(void)format;
	(void)image_count;
#endif
}

void AfterPresent() {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (g_present_begin != std::chrono::steady_clock::time_point {}) {
		const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g_present_begin).count();
		g_stat_present_ms += ms;
		g_stat_present_max = std::max(g_stat_present_max, ms);
	}
	if (!g_ready || g_token == nullptr) {
		return;
	}
	g_api.pcl_set_marker(sl::PCLMarker::ePresentEnd, *g_token);
	const auto presents = ++g_presents;
	if (presents % 30 != 1) {
		return;
	}
	sl::DLSSGState state {};
	if (g_api.dlssg_get_state(sl::ViewportHandle {0}, state, nullptr) != sl::Result::eOk) {
		return;
	}
	g_presented_frames.store(std::max(state.numFramesActuallyPresented, 1u), std::memory_order_relaxed);
	// More generated frames asked for than the GPU makes (the launcher's 3x and up): as many as it makes.
	if (state.numFramesToGenerateMax != 0 && static_cast<uint32_t>(g_generate) > state.numFramesToGenerateMax) {
		LOGF("frame generation: %d generated frames asked for, the GPU makes up to %u\n", g_generate,
		     state.numFramesToGenerateMax);
		std::printf("Frame generation: %d generated frame(s) asked for, the GPU makes up to %u\n", g_generate,
		            state.numFramesToGenerateMax);
		g_generate    = static_cast<int>(state.numFramesToGenerateMax);
		g_options_set = false;
	}
	if (presents % 600 == 1) {
		LOGF("frame generation: status=0x%x presented=%u max_generated=%u vram=%" PRIu64 " MiB\n",
		     static_cast<uint32_t>(state.status), state.numFramesActuallyPresented,
		     state.numFramesToGenerateMax, state.estimatedVRAMUsageInBytes >> 20u);
		std::printf("Frame generation: status=0x%x presented=%u max_generated=%u\n",
		            static_cast<uint32_t>(state.status), state.numFramesActuallyPresented,
		            state.numFramesToGenerateMax);
		std::fflush(stdout);
	}
#endif
}

uint32_t PresentedFrames() {
	return g_presented_frames.load(std::memory_order_relaxed);
}

void SetStaticRect(float x0, float y0, float x1, float y1) {
	const auto pack = [](float value, uint32_t index) {
		return static_cast<uint64_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 65535.0f)) << (16u * index);
	};
	g_static_rect.store(pack(x0, 0) | pack(y0, 1) | pack(x1, 2) | pack(y1, 3), std::memory_order_relaxed);
}

void OnDraw(uint64_t vs_hash, std::span<const uint32_t> srt) {
	if (g_camera.valid || srt.size() < SrtJitter + 2 || !IsGbufferVertexShader(vs_hash)) {
		return;
	}
	for (uint32_t i = 0; i < 16; ++i) {
		g_camera.view_proj[i] = AsFloat(srt[SrtViewProj + i]);
	}
	// The previous matrix has rows x, y, w; its z row is taken from the current one.
	for (uint32_t i = 0; i < 4; ++i) {
		g_camera.prev_view_proj[i]      = AsFloat(srt[SrtPrevViewProj + i]);
		g_camera.prev_view_proj[4 + i]  = AsFloat(srt[SrtPrevViewProj + 4 + i]);
		g_camera.prev_view_proj[8 + i]  = g_camera.view_proj[8 + i];
		g_camera.prev_view_proj[12 + i] = AsFloat(srt[SrtPrevViewProj + 8 + i]);
	}
	g_camera.jitter_ndc = {AsFloat(srt[SrtJitter]), AsFloat(srt[SrtJitter + 1])};
	g_camera.valid      = true;
}

void NoteScene() {
	g_scene.store(true, std::memory_order_relaxed);
}

bool TakeScene() {
	const bool scene = g_scene.exchange(false, std::memory_order_relaxed);
	g_last_scene.store(scene, std::memory_order_relaxed);
	g_flips_taken.fetch_add(1, std::memory_order_relaxed);
	return scene;
}

uint64_t GuestFlips() {
	return g_flips_taken.load(std::memory_order_relaxed);
}

bool SceneNoted() {
	return g_scene.load(std::memory_order_relaxed) || g_last_scene.load(std::memory_order_relaxed);
}

void OnDispatch(uint64_t cs_hash, ImageId depth, uint64_t depth_address, ImageId motion,
                uint64_t motion_address) {
	if (!IsTaaShader(cs_hash)) {
		return;
	}
	g_sources = {depth, motion, depth_address, motion_address, true};
}

namespace {

// FSR 3's HUD-less image: the flip image before the game draws its HUD on it. The 3D scene reaches it in the first
// render pass of the frame into it (KYTY_FSR3_HUDLESS_PASS: the pass before which it is copied, 2 by default: the
// scene's full-screen pass is the first, the HUD's draws come after); the copy is taken as that pass begins.
struct FlipTarget {
	ImageId                    id;
	vk::Image                  image   = nullptr;
	uint64_t                   address = 0;
	std::vector<vk::ImageView> views;
};
std::array<FlipTarget, 4> g_flips {};
uint32_t                  g_flip_next    = 0;
TextureCache*             g_cache        = nullptr;
GraphicContext*           g_graphics_ctx = nullptr;
VulkanImage               g_hudless;
vk::ImageView             g_hudless_view = nullptr;
uint32_t                  g_flip_renders = 0;
bool                      g_hudless_taken = false;
uint32_t                  g_hudless_logs  = 0;

bool HudlessWanted() {
	static const bool wanted = [] {
		const char* value = std::getenv("KYTY_FSR3_HUDLESS");
		return value == nullptr || *value == '\0' || *value != '0';
	}();
	return wanted;
}
vk::Format UnormOf(vk::Format format) {
	switch (format) {
		case vk::Format::eR8G8B8A8Srgb: return vk::Format::eR8G8B8A8Unorm;
		case vk::Format::eB8G8R8A8Srgb: return vk::Format::eB8G8R8A8Unorm;
		default: return format;
	}
}
// `image` as a sampled copy target of that format and size.
void ConfigureAs(GraphicContext& graphics, VulkanImage& image, vk::ImageView& view, vk::Format format, vk::Extent3D extent) {
	if (image.image != nullptr && image.format == format && image.extent.width == extent.width &&
	    image.extent.height == extent.height)
		return;
	VulkanImage source {};
	source.format = format;
	source.extent = extent;
	Configure(graphics, image, view, source);
}
// A raw copy between two images of one size (formats of one class).
void CopyRaw(vk::CommandBuffer command, VulkanImage& source, VulkanImage& target) {
	Transition(command, source, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
	           vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eTransfer);
	Transition(command, target, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
	           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
	vk::ImageCopy copy {};
	copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.extent         = target.extent;
	command.copyImage(source.image, vk::ImageLayout::eTransferSrcOptimal, target.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy);
	for (auto* image: {&source, &target})
		Transition(command, *image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
		           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
}

} // namespace

namespace {
// The flip image's render pass in progress (renderer thread).
bool              g_flip_pass       = false;
const FlipTarget* g_flip_target     = nullptr;
uint32_t          g_flip_attachment = 0;
uint32_t          g_flip_draws      = 0;
uint32_t          g_flip_max_draws  = 0;
RenderState       g_flip_state {};
bool              g_resuming        = false;
uint32_t HudlessDraw() {
	static const uint32_t draw = [] {
		const char* value = std::getenv("KYTY_FSR3_HUDLESS_DRAW");
		const int   n     = value != nullptr ? std::atoi(value) : 1;
		return static_cast<uint32_t>(std::clamp(n, 1, 1000));
	}();
	return draw;
}
} // namespace

void OnBeginRendering(CommandBuffer& command, const RenderState& state) {
	(void)command;
	if (g_resuming || !LiveCensus::g_render || !Capturing() || !HudlessWanted() || g_cache == nullptr ||
	    g_graphics_ctx == nullptr)
		return;
	g_flip_pass   = false;
	g_flip_target = nullptr;
	for (uint32_t i = 0; i < state.num_color_attachments && g_flip_target == nullptr; ++i) {
		const auto view = state.color_attachments[i].image_view;
		if (view == nullptr) continue;
		for (const auto& target: g_flips)
			if (std::find(target.views.begin(), target.views.end(), view) != target.views.end()) {
				g_flip_target     = &target;
				g_flip_attachment = i;
				break;
			}
	}
	if (g_flip_target == nullptr) return;
	g_flip_pass  = true;
	g_flip_draws = 0;
	g_flip_state = state;
	++g_flip_renders;
}

void AfterDraw(CommandBuffer& command) {
	if (!g_flip_pass || !LiveCensus::g_render) return;
	++g_flip_draws;
	g_flip_max_draws = std::max(g_flip_max_draws, g_flip_draws);
	if (g_hudless_taken || g_flip_draws != HudlessDraw() || g_flip_target == nullptr) return;
	auto& image = g_cache->GetImage(g_flip_target->id);
	if (image.backing.image != g_flip_target->image || image.info.data.address != g_flip_target->address) return;
	ConfigureAs(*g_graphics_ctx, g_hudless, g_hudless_view, UnormOf(image.backing.format), image.backing.extent);
	command.EndRendering();
	auto vk_command = command.Handle();
	CopyInto(vk_command, image, g_hudless);
	image.Transit(g_flip_state.color_attachments[g_flip_attachment].image_layout,
	              vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite, {}, vk_command);
	g_resuming = true;
	command.ResumeRendering(g_flip_state);
	g_resuming      = false;
	g_hudless_taken = true;
}

namespace {
std::unordered_map<uint64_t, uint32_t> g_target_draws; // this frame's normal-path draws per color target
VulkanImage                            g_flip_backup;
vk::ImageView                          g_flip_backup_view = nullptr;
VulkanImage                            g_ui_backup;
vk::ImageView                          g_ui_backup_view = nullptr;
bool                                   g_composite      = false; // this frame's HUD-less image is a composite without HUD
uint32_t                               g_composite_logs = 0;
uint32_t                               g_flip_slot_hits = 0;
uint64_t KeyOf(ImageId id) { return (uint64_t {id.index} << 32u) | id.generation; }
// KYTY_FSR3_UI_IMAGE=n: the HUD texture is the composite's n-th candidate (the log lists them), not the guess.
int UiOverride() {
	static const int value = [] {
		const char* text = std::getenv("KYTY_FSR3_UI_IMAGE");
		return text != nullptr && *text != '\0' ? std::atoi(text) : -1;
	}();
	return value;
}
bool UiDebug() {
	static const bool value = [] {
		const char* text = std::getenv("KYTY_FSR3_UI_DEBUG");
		return text != nullptr && *text == '1';
	}();
	return value;
}
bool Rgba8(vk::Format format) {
	switch (format) {
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32: return true;
		default: return false;
	}
}
const FlipTarget* FlipOf(ImageId id) {
	if (!id || g_cache == nullptr) return nullptr;
	// (A table draw state's target id may name an image erased since: NativeXprTargetsCurrent asks before it checks
	// the ids are live. GetImage asserted on one at the dragon's path, 2026-10-11.)
	const auto* image = g_cache->TryGetImage(id);
	if (image == nullptr) return nullptr;
	for (const auto& target: g_flips)
		if (target.image != nullptr && image->backing.image == target.image && image->info.data.address == target.address)
			return &target;
	return nullptr;
}
// `backup` into `image` (mip 0, layer 0).
void CopyBack(vk::CommandBuffer command, VulkanImage& backup, Image& image) {
	Transition(command, backup, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
	           vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eTransfer);
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	vk::ImageCopy copy {};
	copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.extent         = backup.extent;
	command.copyImage(backup.image, vk::ImageLayout::eTransferSrcOptimal, image.backing.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy);
	Transition(command, backup, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
	           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
}
} // namespace

namespace {
// IsFlipImage's answers per image id (direct-mapped), valid while g_flip_epoch holds (it moves when a flip image changes).
struct FlipAnswer {
	ImageId  id;
	uint64_t epoch  = 0;
	bool     answer = false;
};
std::array<FlipAnswer, 512> g_flip_answers {};
uint64_t                    g_flip_epoch = 1;
} // namespace

bool IsFlipImage(ImageId id) {
	if (!id || g_cache == nullptr || !LiveCensus::g_render) return false;
	auto& cached = g_flip_answers[id.index & (g_flip_answers.size() - 1)];
	if (cached.epoch == g_flip_epoch && cached.id == id) return cached.answer;
	cached = {id, g_flip_epoch, FlipOf(id) != nullptr};
	return cached.answer;
}

void NoteDraw(ImageId target) {
	if (target && LiveCensus::g_render) ++g_target_draws[KeyOf(target)];
}

int FlipSlot(std::span<const ImageId> targets) {
	if (!LiveCensus::g_render || !HudlessWanted() || g_cache == nullptr || g_graphics_ctx == nullptr || g_hudless_taken)
		return -1;
	for (size_t i = 0; i < targets.size(); ++i)
		if (FlipOf(targets[i]) != nullptr) {
			++g_flip_slot_hits;
			return static_cast<int>(i);
		}
	return -1;
}

void BeforeFlipDraw(CommandBuffer& command, const RenderState& rendering, ImageId target, uint32_t slot,
                    std::span<const TextureBinding> images, const std::function<void()>& redraw) {
	if (!Enabled() && !Fsr3::Active()) return; // (capturing for upscaling only: no HUD-less image wanted)
	if (g_hudless_taken || slot >= rendering.num_color_attachments || FlipOf(target) == nullptr) return;
	auto& flip = g_cache->GetImage(target);
	// The HUD texture: of the full-screen-ish color images the composite samples, an 8-bit (or 10-bit) one with alpha
	// that the most draws went to this frame (the HUD's sprites and text; the scene's last passes are one draw each).
	struct Candidate {
		size_t   binding;
		ImageId  id;
		uint32_t draws;
		bool     rgba8;
	};
	std::vector<Candidate> candidates;
	for (size_t i = 0; i < images.size(); ++i) {
		const auto id = images[i].image_id;
		if (!id || id == target) continue;
		if (std::any_of(candidates.begin(), candidates.end(), [&](const Candidate& c) { return c.id == id; })) continue;
		auto& image = g_cache->GetImage(id);
		if (IsDepthFormat(image.backing.format) || image.backing.image == flip.backing.image) continue;
		if (image.backing.extent.width * 2 < flip.backing.extent.width ||
		    image.backing.extent.height * 2 < flip.backing.extent.height)
			continue;
		const auto found = g_target_draws.find(KeyOf(id));
		candidates.push_back({i, id, found != g_target_draws.end() ? found->second : 0u, Rgba8(image.backing.format)});
	}
	int chosen = -1;
	if (const int forced = UiOverride(); forced >= 0) {
		if (forced < static_cast<int>(candidates.size())) chosen = forced;
	} else {
		// Demon's Souls: the composite samples the scene at the internal resolution (2560x1440, upscaled by the draw)
		// and the HUD layer at the output resolution (3840x2160, RGBA16F, drawn on paths that are not counted). So the
		// HUD is first the image of the flip image's size, then the one with the most draws.
		const auto rank = [&](const Candidate& c) {
			auto& image = g_cache->GetImage(c.id);
			const bool full = image.backing.extent.width == flip.backing.extent.width &&
			                  image.backing.extent.height == flip.backing.extent.height;
			return std::tuple {full, c.draws > 0, c.rgba8, c.draws};
		};
		for (size_t i = 0; i < candidates.size(); ++i) {
			const auto& c = candidates[i];
			if (!std::get<0>(rank(c)) && c.draws == 0) continue;
			if (chosen < 0 || rank(c) > rank(candidates[chosen])) chosen = static_cast<int>(i);
		}
	}
	if (g_composite_logs < 8 || (UiDebug() && g_composite_logs < 200 && (++g_composite_logs % 20) == 0)) {
		++g_composite_logs;
		std::printf("HUD-less: the flip image's draw (%ux%u format %d) samples %zu images, candidates:",
		            flip.backing.extent.width, flip.backing.extent.height, static_cast<int>(flip.backing.format),
		            images.size());
		for (size_t i = 0; i < candidates.size(); ++i) {
			const auto& image = g_cache->GetImage(candidates[i].id);
			std::printf(" [%zu] binding %zu %ux%u format %d draws %u%s", i, candidates[i].binding, image.backing.extent.width,
			            image.backing.extent.height, static_cast<int>(image.backing.format), candidates[i].draws,
			            static_cast<int>(i) == chosen ? " (HUD)" : "");
		}
		std::printf("\n");
		std::fflush(stdout);
	}
	if (chosen < 0) return;
	const auto& binding = images[candidates[chosen].binding];
	auto&       ui      = g_cache->GetImage(candidates[chosen].id);
	const auto  ui_layout    = binding.layout == vk::ImageLayout::eUndefined ? vk::ImageLayout::eShaderReadOnlyOptimal
	                                                                         : binding.layout;
	const auto  flip_layout  = rendering.color_attachments[slot].image_layout;
	const auto  flip_access  = vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
	const bool  in_pass      = command.RenderingWith(rendering);
	auto        vk_command   = command.Handle();
	command.EndRendering();
	// Both as they are now.
	ConfigureAs(*g_graphics_ctx, g_flip_backup, g_flip_backup_view, flip.backing.format, flip.backing.extent);
	CopyInto(vk_command, flip, g_flip_backup);
	ConfigureAs(*g_graphics_ctx, g_ui_backup, g_ui_backup_view, ui.backing.format, ui.backing.extent);
	CopyInto(vk_command, ui, g_ui_backup);
	// No HUD.
	ui.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, vk_command);
	vk::ClearColorValue             transparent {};
	transparent.uint32 = std::array<uint32_t, 4> {0u, 0u, 0u, 0u};
	const vk::ImageSubresourceRange level {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk_command.clearColorImage(ui.backing.image, vk::ImageLayout::eTransferDstOptimal, &transparent, 1, &level);
	ui.Transit(ui_layout, vk::AccessFlagBits2::eShaderRead, {}, vk_command);
	flip.Transit(flip_layout, flip_access, {}, vk_command);
	// The draw without it (the pass as the game began it: cleared, unless it was already in progress).
	g_resuming = true;
	if (in_pass) command.ResumeRendering(rendering);
	else command.BeginRendering(rendering);
	redraw();
	command.EndRendering();
	ConfigureAs(*g_graphics_ctx, g_hudless, g_hudless_view, UnormOf(flip.backing.format), flip.backing.extent);
	CopyInto(vk_command, flip, g_hudless);
	// Everything back.
	CopyBack(vk_command, g_flip_backup, flip);
	// KYTY_FSR3_UI_DEBUG=1: the HUD texture is left cleared, so the game's own draw shows the picture without what
	// was taken as the HUD (everything that vanishes from the screen is what frame generation keeps out).
	if (!UiDebug()) CopyBack(vk_command, g_ui_backup, ui);
	ui.Transit(ui_layout, vk::AccessFlagBits2::eShaderRead, {}, vk_command);
	flip.Transit(flip_layout, flip_access, {}, vk_command);
	if (in_pass) command.ResumeRendering(rendering);
	g_resuming      = false;
	g_hudless_taken = true;
	g_composite     = true;
}

namespace {
VulkanImage   g_up_depth, g_up_motion, g_up_output;
vk::ImageView g_up_depth_view = nullptr, g_up_motion_view = nullptr, g_up_output_view = nullptr;
bool          g_up_reset = true;
std::chrono::steady_clock::time_point g_up_last {};
uint32_t      g_up_logs  = 0;
uint64_t      g_up_count = 0, g_up_missed = 0;

uint32_t UpscaleHeight() {
	static const uint32_t height = [] {
		const char* value = std::getenv("KYTY_UPSCALE_HEIGHT");
		const auto  h     = value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 1440u;
		return std::clamp(h, 720u, 2160u);
	}();
	return height;
}

void ConfigureOutput(GraphicContext& graphics, vk::Format format, vk::Extent2D extent) {
	if (g_up_output.image != nullptr && g_up_output.format == format && g_up_output.extent.width == extent.width &&
	    g_up_output.extent.height == extent.height)
		return;
	if (g_up_output_view != nullptr) {
		graphics.device.waitIdle();
		graphics.device.destroyImageView(g_up_output_view, nullptr);
		g_up_output_view = nullptr;
	}
	if (g_up_output.image != nullptr) graphics.DeleteImage(g_up_output);
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc |
	               vk::ImageUsageFlagBits::eTransferDst;
	create.sharingMode = vk::SharingMode::eExclusive;
	create.samples     = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, g_up_output)) {
		EXIT("upscaling: cannot allocate a %ux%u output image\n", extent.width, extent.height);
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = g_up_output.image;
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = format;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	RequireVulkanSuccess(graphics.device.createImageView(&view_info, nullptr, &g_up_output_view), "upscaling output view");
}
} // namespace

namespace {
// KYTY_TAA_REPLACE=upscaler: the TAA dispatch's jittered color (image 0) copied to its output (image 6), each image
// back in the layout the dispatch bound it in.
bool BypassTaa(vk::CommandBuffer vk_command, std::span<const TextureBinding> images) {
	if (g_cache == nullptr || images.size() < 7) return false;
	if (!Fsr3::UpscaleWanted() && !g_dlss_sr && !Xess::Wanted()) return false; // (no upscaler: the game's TAA)
	const auto& color_b = images[0];
	const auto& out_b   = images[6];
	if (!color_b.image_id || !out_b.image_id) return false;
	auto& color  = g_cache->GetImage(color_b.image_id);
	auto& output = g_cache->GetImage(out_b.image_id);
	static uint32_t logs = 0;
	if (color.backing.extent.width != output.backing.extent.width ||
	    color.backing.extent.height != output.backing.extent.height || color.backing.image == output.backing.image) {
		if (logs < 2) {
			++logs;
			std::printf("TAA: not skipped (color %ux%u, output %ux%u)\n", color.backing.extent.width,
			            color.backing.extent.height, output.backing.extent.width, output.backing.extent.height);
			std::fflush(stdout);
		}
		return false;
	}
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
	LocalVulkanRecording::Drain();
#endif
	const vk::ImageSubresourceLayers layers {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	color.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, vk_command);
	output.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, vk_command);
	if (color.backing.format == output.backing.format) {
		vk::ImageCopy copy {};
		copy.srcSubresource = layers;
		copy.dstSubresource = layers;
		copy.extent         = vk::Extent3D {color.backing.extent.width, color.backing.extent.height, 1};
		vk_command.copyImage(color.backing.image, vk::ImageLayout::eTransferSrcOptimal, output.backing.image,
		                     vk::ImageLayout::eTransferDstOptimal, copy);
	} else {
		vk::ImageBlit blit {};
		blit.srcSubresource = layers;
		blit.dstSubresource = layers;
		blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(color.backing.extent.width),
		                                    static_cast<int32_t>(color.backing.extent.height), 1};
		blit.dstOffsets[1]  = blit.srcOffsets[1];
		vk_command.blitImage(color.backing.image, vk::ImageLayout::eTransferSrcOptimal, output.backing.image,
		                     vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eNearest);
	}
	const auto shader = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	color.Transit(color_b.layout, shader, {}, vk_command);
	output.Transit(out_b.layout, shader, {}, vk_command);
	if (logs < 2) {
		++logs;
		std::printf("TAA: skipped, the upscaler takes the raw %ux%u frame (format %d -> %d)\n", color.backing.extent.width,
		            color.backing.extent.height, static_cast<int>(color.backing.format),
		            static_cast<int>(output.backing.format));
		std::fflush(stdout);
	}
	return true;
}
} // namespace

namespace {
struct TaaBypassPipeline {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             sampler    = nullptr;
	VkBuffer                stats      = VK_NULL_HANDLE; // (taa-bypass.comp's Stats: sums, the gain)
	VmaAllocation           stats_mem  = nullptr;
	VkBuffer                readback   = VK_NULL_HANDLE; // (a copy of it for the log)
	VmaAllocation           readback_mem = nullptr;
	const float*            readback_mapped = nullptr;
	VulkanImage             dlaa;                        // KYTY_TAA_REPLACE=dlss: DLAA's output, then into the TAA's
	vk::ImageView           dlaa_view = nullptr;
	uint64_t                raw_frame = UINT64_MAX;      // the guest frame whose raw color `dlaa` holds (bypass)
	// The TAA's own output (before the raw frame replaced it), put back as its history before its next dispatch: the
	// TAA then blends with its own past as on the console, and its output, compared with the raw frame, gives the
	// brightness the raw frame needs (10-17: with the replaced frame as its history the comparison fed on itself
	// and the gain ran to the 8x clamp).
	VulkanImage             truth;
	uint64_t                truth_frame = UINT64_MAX;
	bool                    failed     = false;
} g_bypass;

bool EnsureBypassPipeline(GraphicContext& graphics) {
	if (g_bypass.pipeline != nullptr) return true;
	if (g_bypass.failed) return false;
	g_bypass.failed   = true; // until it is made
	const auto device = graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, 3> bindings {};
	for (uint32_t i = 0; i < 3; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = i == 0   ? vk::DescriptorType::eCombinedImageSampler
		                              : i == 1 ? vk::DescriptorType::eStorageImage
		                                       : vk::DescriptorType::eStorageBuffer;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = static_cast<uint32_t>(bindings.size());
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_bypass.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, 16};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_bypass.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_bypass.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(TAA_BYPASS_SPV);
	module_info.pCode       = TAA_BYPASS_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_bypass.layout;
	const auto created = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_bypass.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_bypass.pipeline = nullptr;
		return false;
	}
	vk::SamplerCreateInfo sampler {};
	sampler.magFilter    = vk::Filter::eNearest;
	sampler.minFilter    = vk::Filter::eNearest;
	sampler.mipmapMode   = vk::SamplerMipmapMode::eNearest;
	sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	sampler.maxLod       = 0.0f;
	if (device.createSampler(&sampler, nullptr, &g_bypass.sampler) != vk::Result::eSuccess) return false;
	{
		VkBufferCreateInfo create {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		create.size  = 32;
		create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo allocate {};
		allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		if (vmaCreateBuffer(graphics.allocator, &create, &allocate, &g_bypass.stats, &g_bypass.stats_mem, nullptr) != VK_SUCCESS)
			return false;
		create.usage   = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		allocate.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo mapped {};
		if (vmaCreateBuffer(graphics.allocator, &create, &allocate, &g_bypass.readback, &g_bypass.readback_mem, &mapped) ==
		    VK_SUCCESS)
			g_bypass.readback_mapped = static_cast<const float*>(mapped.pMappedData);
	}
	g_bypass.failed = false;
	return true;
}

bool EnsureDlaaImage(GraphicContext& graphics, vk::Extent2D extent) {
	if (g_bypass.dlaa.image != nullptr && g_bypass.dlaa.extent.width == extent.width &&
	    g_bypass.dlaa.extent.height == extent.height)
		return true;
	if (g_bypass.dlaa_view != nullptr) {
		graphics.device.waitIdle();
		graphics.device.destroyImageView(g_bypass.dlaa_view, nullptr);
		g_bypass.dlaa_view = nullptr;
	}
	if (g_bypass.dlaa.image != nullptr) graphics.DeleteImage(g_bypass.dlaa);
	g_bypass.dlaa.image = nullptr;
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = vk::Format::eR16G16B16A16Sfloat;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
	               vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	create.sharingMode = vk::SharingMode::eExclusive;
	create.samples     = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, g_bypass.dlaa)) {
		g_bypass.dlaa.image = nullptr;
		return false;
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = g_bypass.dlaa.image;
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = create.format;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &g_bypass.dlaa_view) != vk::Result::eSuccess) {
		graphics.DeleteImage(g_bypass.dlaa);
		g_bypass.dlaa.image = nullptr;
		g_bypass.dlaa_view  = nullptr;
		return false;
	}
	g_bypass.dlaa.state = {};
	return true;
}


// A raw copy between a cache image and one of ours (one size and format), the cache image back in `layout`.
void CopyWithCacheImage(vk::CommandBuffer vk_command, Image& cache_image, vk::ImageLayout layout, VulkanImage& ours,
                        bool to_ours) {
	const vk::ImageSubresourceLayers layers {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	vk::ImageCopy copy {};
	copy.srcSubresource = layers;
	copy.dstSubresource = layers;
	copy.extent         = vk::Extent3D {ours.extent.width, ours.extent.height, 1};
	if (to_ours) {
		cache_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, vk_command);
		Transition(vk_command, ours, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
		           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
		vk_command.copyImage(cache_image.backing.image, vk::ImageLayout::eTransferSrcOptimal, ours.image,
		                     vk::ImageLayout::eTransferDstOptimal, copy);
	} else {
		Transition(vk_command, ours, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
		           vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eTransfer);
		cache_image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, vk_command);
		vk_command.copyImage(ours.image, vk::ImageLayout::eTransferSrcOptimal, cache_image.backing.image,
		                     vk::ImageLayout::eTransferDstOptimal, copy);
	}
	cache_image.Transit(layout, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, {}, vk_command);
}

bool EnsureTruthImage(GraphicContext& graphics, vk::Extent2D extent) {
	auto& t = g_bypass.truth;
	if (t.image != nullptr && t.extent.width == extent.width && t.extent.height == extent.height) return true;
	if (t.image != nullptr) {
		graphics.device.waitIdle();
		graphics.DeleteImage(t);
		t.image = nullptr;
	}
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = vk::Format::eR16G16B16A16Sfloat;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, t)) {
		t.image = nullptr;
		return false;
	}
	t.state             = {};
	g_bypass.truth_frame = UINT64_MAX;
	return true;
}

float EnvFloatOr(const char* name, float fallback) {
	const char* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? static_cast<float>(std::atof(value)) : fallback;
}
} // namespace

bool AfterTaa(CommandBuffer& command, vk::CommandBuffer vk_command, std::span<const TextureBinding> images) {
	const bool bypass = g_taa_bypass && !g_taa_bypass_copy;
	if ((!bypass && !g_taa_dlss) || g_cache == nullptr || g_graphics_ctx == nullptr || images.size() < 7) return false;
	if (bypass && !Fsr3::UpscaleWanted() && !g_dlss_sr && !Xess::Wanted()) return false; // (no upscaler: the TAA's)
	const auto& color_b = images[0];
	const auto& out_b   = images[6];
	if (!color_b.image_id || !out_b.image_id || color_b.image_view == nullptr || out_b.image_view == nullptr) return false;
	auto&           color  = g_cache->GetImage(color_b.image_id);
	auto&           output = g_cache->GetImage(out_b.image_id);
	const vk::Extent2D render {color.backing.extent.width, color.backing.extent.height};
	static uint32_t logs   = 0;
	if (output.backing.extent.width != render.width || output.backing.extent.height != render.height ||
	    color.backing.image == output.backing.image || output.backing.format != vk::Format::eR16G16B16A16Sfloat ||
	    !EnsureBypassPipeline(*g_graphics_ctx)) {
		if (logs < 2) {
			++logs;
			std::printf("TAA: kept (color %ux%u, output %ux%u format %d)\n", render.width, render.height,
			            output.backing.extent.width, output.backing.extent.height, static_cast<int>(output.backing.format));
			std::fflush(stdout);
		}
		return false;
	}
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
	LocalVulkanRecording::Drain();
#endif
	// The TAA's writes (and its reads) before what follows.
	const auto compute_barrier = [&](vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
		vk::MemoryBarrier2 barrier {};
		barrier.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
		barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead;
		barrier.dstStageMask  = dst_stage;
		barrier.dstAccessMask = dst_access;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &barrier;
		vk_command.pipelineBarrier2(dependency);
	};
	compute_barrier(vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);

	if (bypass && !g_taa_bypass_match) {
		// The raw jittered frame kept for the upscaler (UpscaleComposite); the TAA's output stays the game's.
		if (color.backing.format != vk::Format::eR16G16B16A16Sfloat || !EnsureDlaaImage(*g_graphics_ctx, render)) return false;
		color.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, vk_command);
		Transition(vk_command, g_bypass.dlaa, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
		           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
		vk::ImageCopy copy {};
		copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		copy.extent         = vk::Extent3D {render.width, render.height, 1};
		vk_command.copyImage(color.backing.image, vk::ImageLayout::eTransferSrcOptimal, g_bypass.dlaa.image,
		                     vk::ImageLayout::eTransferDstOptimal, copy);
		Transition(vk_command, g_bypass.dlaa, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
		           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader);
		color.Transit(color_b.layout, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, {}, vk_command);
		g_bypass.raw_frame = g_guest_frame;
		if (logs < 2) {
			++logs;
			std::printf("TAA: ran for the game; its raw %ux%u frame kept for the upscaler\n", render.width, render.height);
			std::fflush(stdout);
		}
		return true;
	}

	// The source of the color: the raw jittered frame, or DLAA's output (KYTY_TAA_REPLACE=dlss).
	vk::ImageView   source_view   = color_b.image_view;
	vk::ImageLayout source_layout = color_b.layout == vk::ImageLayout::eUndefined ? vk::ImageLayout::eGeneral : color_b.layout;
	if (!bypass) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
		const auto& depth_b = images[1];
		const auto& mv_b    = images[3];
		if (!g_sr_ready || !depth_b.image_id || !mv_b.image_id || depth_b.image_view == nullptr ||
		    mv_b.image_view == nullptr || !EnsureDlaaImage(*g_graphics_ctx, render))
			return false;
		auto& depth  = g_cache->GetImage(depth_b.image_id);
		auto& motion = g_cache->GetImage(mv_b.image_id);
		if (depth.backing.extent.width != render.width || motion.backing.extent.width != render.width) return false;
		static bool     reset = true;
		static uint64_t last  = 0;
		const uint64_t  frame = g_guest_frame;
		reset = reset || frame > last + 2; // (a gap: menus, loading screens or movies in between)
		last  = frame;
		Transition(vk_command, g_bypass.dlaa, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
		           vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
		           vk::PipelineStageFlagBits2::eComputeShader);
		DlssLayouts layouts;
		layouts.color  = color_b.layout;
		layouts.depth  = depth_b.layout;
		layouts.motion = mv_b.layout;
		layouts.output = vk::ImageLayout::eGeneral;
		const bool done = DlssEvaluate(static_cast<VkCommandBuffer>(vk_command), color.backing, color_b.image_view,
		                               depth.backing, depth_b.image_view, motion.backing, mv_b.image_view, g_bypass.dlaa,
		                               g_bypass.dlaa_view, render, render, g_camera, reset, true, layouts);
		command.InvalidateGraphicsState();
		if (!done) return false; // (the TAA's output stays)
		reset = false;
		Transition(vk_command, g_bypass.dlaa, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
		           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader);
		source_view   = g_bypass.dlaa_view;
		source_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
#else
		return false;
#endif
	}

	// The TAA's own output kept, for its history next frame (ReplaceTaa: RestoreTaaHistory).
	if (EnsureTruthImage(*g_graphics_ctx, render)) {
		CopyWithCacheImage(vk_command, output, out_b.layout, g_bypass.truth, true);
		g_bypass.truth_frame = g_guest_frame;
		compute_barrier(vk::PipelineStageFlagBits2::eAllCommands,
		                vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	}

	vk::DescriptorImageInfo  infos[2] {};
	infos[0] = {g_bypass.sampler, source_view, source_layout};
	infos[1] = {nullptr, out_b.image_view, vk::ImageLayout::eGeneral};
	vk::DescriptorBufferInfo stats {g_bypass.stats, 0, 32};
	vk::WriteDescriptorSet   writes[3] {};
	for (uint32_t i = 0; i < 3; ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = i == 0   ? vk::DescriptorType::eCombinedImageSampler
		                            : i == 1 ? vk::DescriptorType::eStorageImage
		                                     : vk::DescriptorType::eStorageBuffer;
		if (i < 2) writes[i].pImageInfo = &infos[i];
		else writes[i].pBufferInfo = &stats;
	}
	const bool out_general = out_b.layout == vk::ImageLayout::eGeneral;
	if (!out_general)
		output.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, {}, vk_command);
	vk_command.bindPipeline(vk::PipelineBindPoint::eCompute, g_bypass.pipeline);
	vk_command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_bypass.layout, 0, 3, writes);
	// KYTY_TAA_MATCH=0: no luminance match (the raw frame as it is); KYTY_TAA_MATCH_SMOOTH: this frame's weight.
	static const float scale     = std::max(EnvFloatOr("KYTY_TAA_BYPASS_GAIN", 1.0f), 0.01f);
	static const bool  match     = EnvFloatOr("KYTY_TAA_MATCH", 1.0f) != 0.0f;
	static const float smoothing = std::clamp(EnvFloatOr("KYTY_TAA_MATCH_SMOOTH", 0.1f), 0.001f, 1.0f);
	struct {
		float   scale;
		int32_t mode;
		float   smoothing;
		int32_t match;
	} push {scale, 0, smoothing, match ? 1 : 0};
	const uint32_t groups_x = (render.width + 7) / 8, groups_y = (render.height + 7) / 8;
	for (int32_t mode = 0; mode < 3; ++mode) {
		push.mode = mode;
		vk_command.pushConstants(g_bypass.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		if (mode == 1) vk_command.dispatch(1, 1, 1);
		else vk_command.dispatch(groups_x, groups_y, 1);
		compute_barrier(vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
		                vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferRead);
	}
	if (g_bypass.readback_mapped != nullptr) {
		const vk::BufferCopy copy {0, 0, 16};
		vk_command.copyBuffer(g_bypass.stats, g_bypass.readback, 1, &copy);
	}
	compute_barrier(vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	if (!out_general)
		output.Transit(out_b.layout, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, {}, vk_command);
	command.InvalidateGraphicsState(); // (this pass bound its own pipeline and descriptors)
	static uint64_t calls = 0;
	if (logs < 2 || (calls++ % 600 == 0 && g_bypass.readback_mapped != nullptr)) {
		const bool first = logs < 2;
		if (first) ++logs;
		float gain = 0.0f, measured = 0.0f;
		if (g_bypass.readback_mapped != nullptr) {
			(void)vmaInvalidateAllocation(g_graphics_ctx->allocator, g_bypass.readback_mem, 0, VK_WHOLE_SIZE);
			gain     = g_bypass.readback_mapped[2];
			measured = g_bypass.readback_mapped[3];
		}
		if (first)
			std::printf("TAA: ran, then its color replaced by %s %ux%u (its alpha kept, brightness matched to the TAA's: %s)\n",
			            bypass ? "the raw frame for the upscaler" : "DLAA's", render.width, render.height,
			            match ? "on" : "off");
		std::printf("TAA: brightness gain %.3f (TAA/raw this frame %.3f, fixed factor %.3f)\n", static_cast<double>(gain),
		            static_cast<double>(measured), static_cast<double>(scale));
		std::fflush(stdout);
	}
	return true;
}

// Before the TAA dispatch (AfterTaa replaces its output): its history image (binding 4) gets back the TAA's own
// output of the frame before, when that is the previous frame and the sizes match.
void RestoreTaaHistory(vk::CommandBuffer vk_command, std::span<const TextureBinding> images) {
	static const bool off = std::getenv("KYTY_TAA_TRUE_HISTORY") != nullptr && *std::getenv("KYTY_TAA_TRUE_HISTORY") == '0';
	if (off || g_cache == nullptr || images.size() < 7 || g_bypass.truth.image == nullptr) return;
	if (g_bypass.truth_frame == UINT64_MAX || g_guest_frame != g_bypass.truth_frame + 1) return;
	const auto& hist_b = images[4];
	const auto& out_b  = images[6];
	if (!hist_b.image_id || !out_b.image_id) return;
	auto& history = g_cache->GetImage(hist_b.image_id);
	auto& output  = g_cache->GetImage(out_b.image_id);
	static uint32_t logs = 0;
	if (history.backing.image == output.backing.image || history.backing.format != vk::Format::eR16G16B16A16Sfloat ||
	    history.backing.extent.width != g_bypass.truth.extent.width ||
	    history.backing.extent.height != g_bypass.truth.extent.height) {
		if (logs < 2) {
			++logs;
			std::printf("TAA: history not restored (history %ux%u format %d)\n", history.backing.extent.width,
			            history.backing.extent.height, static_cast<int>(history.backing.format));
			std::fflush(stdout);
		}
		return;
	}
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
	LocalVulkanRecording::Drain();
#endif
	const auto layout = hist_b.layout == vk::ImageLayout::eUndefined ? vk::ImageLayout::eShaderReadOnlyOptimal : hist_b.layout;
	CopyWithCacheImage(vk_command, history, layout, g_bypass.truth, false);
	if (logs < 2) {
		++logs;
		std::printf("TAA: its history is its own output (the raw frame went only to what follows)\n");
		std::fflush(stdout);
	}
}

bool ReplaceTaa(CommandBuffer& command, vk::CommandBuffer vk_command, std::span<const TextureBinding> images) {
	if ((g_taa_bypass && !g_taa_bypass_copy && g_taa_bypass_match) || (g_taa_dlss && !g_taa_bypass_copy))
		RestoreTaaHistory(vk_command, images);
	if (g_taa_bypass) return g_taa_bypass_copy && BypassTaa(vk_command, images);
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!g_taa_dlss || !g_taa_bypass_copy || !g_sr_ready || g_cache == nullptr || images.size() < 7) return false;
	const auto& color_b = images[0];
	const auto& depth_b = images[1];
	const auto& mv_b    = images[3];
	const auto& out_b   = images[6];
	if (!color_b.image_id || !depth_b.image_id || !mv_b.image_id || !out_b.image_id || color_b.image_view == nullptr ||
	    depth_b.image_view == nullptr || mv_b.image_view == nullptr || out_b.image_view == nullptr)
		return false;
	auto& color  = g_cache->GetImage(color_b.image_id);
	auto& depth  = g_cache->GetImage(depth_b.image_id);
	auto& motion = g_cache->GetImage(mv_b.image_id);
	auto& output = g_cache->GetImage(out_b.image_id);
	const vk::Extent2D render {color.backing.extent.width, color.backing.extent.height};
	if (output.backing.extent.width != render.width || output.backing.extent.height != render.height ||
	    depth.backing.extent.width != render.width || motion.backing.extent.width != render.width)
		return false;
	static bool                  reset   = true;
	static uint64_t              last    = 0;
	const uint64_t               frame   = g_guest_frame;
	reset = reset || frame > last + 2; // (a gap: menus, loading screens or movies in between)
	last  = frame;
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
	LocalVulkanRecording::Drain();
#endif
	DlssLayouts layouts;
	layouts.color  = color_b.layout;
	layouts.depth  = depth_b.layout;
	layouts.motion = mv_b.layout;
	layouts.output = out_b.layout;
	const bool done = DlssEvaluate(static_cast<VkCommandBuffer>(vk_command), color.backing, color_b.image_view, depth.backing,
	                               depth_b.image_view, motion.backing, mv_b.image_view, output.backing, out_b.image_view,
	                               render, render, g_camera, reset, true, layouts);
	command.InvalidateGraphicsState();
	static uint32_t logs = 0;
	if (done && logs < 2) {
		++logs;
		std::printf("TAA: replaced by DLSS (DLAA %ux%u, camera %s)\n", render.width, render.height,
		            g_camera.valid ? "valid" : "none");
		std::fflush(stdout);
	}
	if (done) reset = false;
	return done;
#else
	(void)command;
	(void)vk_command;
	(void)images;
	return false;
#endif
}

namespace {
std::atomic<uint64_t> g_taa_diag_count {0};
bool                  g_taa_diag_now = false; // this frame's TAA dispatch was logged: its composite is too
} // namespace

bool TaaDiagFrame() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TAA_DIAG");
		return value != nullptr && *value == '1';
	}();
	if (!on) return false;
	const auto n   = g_taa_diag_count.fetch_add(1, std::memory_order_relaxed);
	g_taa_diag_now = n < 6 || n % 600 == 0;
	if (g_taa_diag_now) {
		const auto& c = g_camera;
		std::printf("[taa] frame %" PRIu64 ": camera %s, jitter ndc %.6f %.6f\n", n, c.valid ? "valid" : "none",
		            c.jitter_ndc[0], c.jitter_ndc[1]);
		std::fflush(stdout);
	}
	return g_taa_diag_now;
}

void UpscaleComposite(CommandBuffer& command, const RenderState& rendering, std::span<TextureBinding> images) {
	if (g_taa_diag_now && g_cache != nullptr) {
		g_taa_diag_now = false;
		std::printf("[taa] composite samples %zu images:", images.size());
		for (size_t i = 0; i < images.size(); ++i) {
			const auto& b = images[i];
			if (!b.image_id) continue;
			const auto& image = g_cache->GetImage(b.image_id);
			std::printf(" [%zu] 0x%" PRIx64 " fmt %d %ux%u", i, b.desc.info.data.address,
			            static_cast<int>(image.backing.format), image.backing.extent.width, image.backing.extent.height);
		}
		std::printf("\n");
		std::fflush(stdout);
	}
	const bool dlss = DlssUpscaleWanted();
	const bool xess = Xess::Wanted();
	if ((!Fsr3::UpscaleWanted() && !dlss && !xess) || g_cache == nullptr || g_graphics_ctx == nullptr) return;
	if (!g_sources.valid) {
		g_up_reset = true; // menus, loading screens and movies: no TAA pass, the history starts again after them
		return;
	}
	auto& depth  = g_cache->GetImage(g_sources.depth);
	auto& motion = g_cache->GetImage(g_sources.motion);
	if (depth.info.data.address != g_sources.depth_address || motion.info.data.address != g_sources.motion_address ||
	    depth.backing.extent.width != motion.backing.extent.width ||
	    depth.backing.extent.height != motion.backing.extent.height) {
		g_up_reset = true;
		return;
	}
	const vk::Extent2D render {depth.backing.extent.width, depth.backing.extent.height};
	// The 3D image: the composite's RGBA16F image of the render size.
	TextureBinding* scene = nullptr;
	for (auto& binding: images) {
		if (!binding.image_id || !binding.mip_views.empty() || binding.image_view == nullptr) continue;
		auto& image = g_cache->GetImage(binding.image_id);
		if (image.backing.format == vk::Format::eR16G16B16A16Sfloat && image.backing.extent.width == render.width &&
		    image.backing.extent.height == render.height) {
			scene = &binding;
			break;
		}
	}
	if (scene == nullptr) {
		if (g_up_missed++ < 4) {
			std::printf("Upscaling: the composite samples no %ux%u RGBA16F image (%zu images): not upscaled\n", render.width,
			            render.height, images.size());
			std::fflush(stdout);
		}
		return;
	}
	const uint32_t out_height = UpscaleHeight();
	if (out_height < render.height) return;
	const auto out_width =
	    static_cast<uint32_t>(std::lround(static_cast<double>(render.width) * out_height / render.height / 2.0)) * 2u;
	const vk::Extent2D output {out_width, out_height};
	auto&              color = g_cache->GetImage(scene->image_id);
	ConfigureOutput(*g_graphics_ctx, color.backing.format, output);

	const bool in_pass    = command.RenderingWith(rendering);
	command.EndRendering();
	auto vk_command = command.Handle();
	Configure(*g_graphics_ctx, g_up_depth, g_up_depth_view, depth.backing);
	Configure(*g_graphics_ctx, g_up_motion, g_up_motion_view, motion.backing);
	CopyInto(vk_command, depth, g_up_depth);
	CopyInto(vk_command, motion, g_up_motion);
	color.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {}, vk_command);
	Transition(vk_command, g_up_output, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
	           vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader);
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
	// FSR records straight into the command buffer: what this thread queued so far goes first.
	LocalVulkanRecording::Drain();
#endif
	const auto now   = std::chrono::steady_clock::now();
	const auto delta = g_up_last == std::chrono::steady_clock::time_point {}
	                       ? 16.7f
	                       : std::chrono::duration<float, std::milli>(now - g_up_last).count();
	g_up_last = now;
	// KYTY_TAA_REPLACE=upscaler: the raw frame kept right after the TAA (AfterTaa), not the TAA's output.
	const VulkanImage* in_color = &color.backing;
	vk::ImageView      in_view  = scene->image_view;
	if (g_taa_bypass && !g_taa_bypass_copy && !g_taa_bypass_match && g_bypass.dlaa.image != nullptr &&
	    g_bypass.raw_frame == g_guest_frame && g_bypass.dlaa.extent.width == render.width &&
	    g_bypass.dlaa.extent.height == render.height) {
		in_color = &g_bypass.dlaa;
		in_view  = g_bypass.dlaa_view;
	} else if (g_taa_bypass && !g_taa_bypass_copy && !g_taa_bypass_match) {
		static uint32_t misses = 0;
		if (misses++ < 4) {
			std::printf("Upscaling: no raw frame of this frame kept (frame %llu, kept %llu): the TAA's output upscaled\n",
			            static_cast<unsigned long long>(g_guest_frame), static_cast<unsigned long long>(g_bypass.raw_frame));
			std::fflush(stdout);
		}
	}
	Fsr3::UpscaleDesc desc;
	desc.command  = static_cast<VkCommandBuffer>(vk_command);
	desc.color    = in_color;
	desc.depth    = &g_up_depth;
	desc.motion   = &g_up_motion;
	desc.output   = &g_up_output;
	desc.render   = render;
	desc.upscale  = output;
	desc.camera   = &g_camera;
	desc.reset    = g_up_reset;
	desc.delta_ms = delta;
	// KYTY_TAA_REPLACE=upscaler: the color is the raw jittered frame: the upscaler needs the camera's jitter, in pixels
	// (as DLSS takes it; KYTY_UPSCALE_JITTER_SIGN=-1 the other way round). After the game's TAA: none left.
	if (g_taa_bypass && g_camera.valid) {
		static const float sign = [] {
			const char* value = std::getenv("KYTY_UPSCALE_JITTER_SIGN");
			return value != nullptr && std::atof(value) < 0 ? -1.0f : 1.0f;
		}();
		desc.jitter[0] = sign * g_camera.jitter_ndc[0] * static_cast<float>(render.width) * 0.5f;
		desc.jitter[1] = sign * -g_camera.jitter_ndc[1] * static_cast<float>(render.height) * 0.5f;
	}
	bool done = false;
	if (dlss) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
		done = DlssUpscale(desc.command, *in_color, in_view, g_up_depth, g_up_depth_view, g_up_motion,
		                   g_up_motion_view, g_up_output, g_up_output_view, render, output, g_camera, g_up_reset);
#endif
	} else if (xess) {
		Xess::UpscaleDesc x;
		x.command     = desc.command;
		x.color       = in_color;
		x.color_view  = in_view;
		x.depth       = &g_up_depth;
		x.depth_view  = g_up_depth_view;
		x.motion      = &g_up_motion;
		x.motion_view = g_up_motion_view;
		x.output      = &g_up_output;
		x.output_view = g_up_output_view;
		x.render      = render;
		x.upscale     = output;
		x.reset       = g_up_reset;
		x.jitter[0]   = desc.jitter[0];
		x.jitter[1]   = desc.jitter[1];
		done          = Xess::Upscale(x);
	} else {
		done = Fsr3::Upscale(*g_graphics_ctx, desc);
	}
	command.InvalidateGraphicsState(); // (the upscaler bound its own pipelines and descriptors)
	Transition(vk_command, g_up_output, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
	           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
	if (in_pass) command.ResumeRendering(rendering);
	if (!done) return;
	scene->image_view = g_up_output_view;
	scene->layout     = vk::ImageLayout::eShaderReadOnlyOptimal;
	g_up_reset        = false;
	++g_up_count;
	if (g_up_logs < 2 || g_up_count % 3600 == 0) {
		++g_up_logs;
		std::printf("Upscaling: %ux%u -> %ux%u (%" PRIu64 " frames)\n", render.width, render.height, output.width,
		            output.height, g_up_count);
		std::fflush(stdout);
	}
}

void PrepareInputs(GraphicContext& graphics, TextureCache& cache, CommandBuffer& command,
                   FrameInputs& inputs, ImageId flip_id, Image& flip) {
	inputs.valid       = false;
	inputs.has_hudless = false;
	inputs.hudless_composite = false;
	g_cache            = &cache;
	g_graphics_ctx     = &graphics;
	// The flip images (two or three alternate): their views as render targets.
	{
		FlipTarget* target = nullptr;
		for (auto& candidate: g_flips)
			if (candidate.image == flip.backing.image && candidate.address == flip.info.data.address) target = &candidate;
		if (target == nullptr) {
			target  = &g_flips[g_flip_next];
			g_flip_next = (g_flip_next + 1) % static_cast<uint32_t>(g_flips.size());
		}
		if (target->image != flip.backing.image || target->address != flip.info.data.address || !(target->id == flip_id))
			++g_flip_epoch;
		target->id      = flip_id;
		target->image   = flip.backing.image;
		target->address = flip.info.data.address;
		target->views.clear();
		for (const auto& view: flip.views)
			if (view.view != nullptr) target->views.push_back(view.view);
	}
	const bool hudless       = g_hudless_taken && g_hudless.image != nullptr;
	const auto flip_renders  = g_flip_renders;
	const auto flip_draws    = g_flip_max_draws;
	g_hudless_taken          = false;
	const bool composite     = std::exchange(g_composite, false);
	g_target_draws.clear();
	g_flip_renders           = 0;
	g_flip_max_draws         = 0;
	g_flip_pass              = false;
	// (DLSS-G needs the camera; FSR 3 only the TAA pass's images: the G-buffer draws that hold the camera mostly
	// take the table path, which does not report it, and generation went on and off from frame to frame.)
	const bool ready = g_sources.valid;
	const auto sources = g_sources;
	const auto camera  = g_camera;
	g_sources.valid    = false;
	g_camera.valid     = false;
	++g_guest_frame;
	if (!ready) {
		return;
	}
	auto& depth  = cache.GetImage(sources.depth);
	auto& motion = cache.GetImage(sources.motion);
	if (depth.info.data.address != sources.depth_address ||
	    motion.info.data.address != sources.motion_address ||
	    depth.backing.extent.width != motion.backing.extent.width ||
	    depth.backing.extent.height != motion.backing.extent.height) {
		return; // the images were replaced after the TAA pass
	}
	Configure(graphics, inputs.depth, inputs.depth_view, depth.backing);
	Configure(graphics, inputs.motion, inputs.motion_view, motion.backing, &inputs.motion_attachment);
	command.EndRendering();
	auto vk_command = command.Handle();
	CopyInto(vk_command, depth, inputs.depth);
	CopyInto(vk_command, motion, inputs.motion);
	if (inputs.motion_attachment) {
		ZeroStaticRect(vk_command, inputs.motion, inputs.motion_view);
	}
	if (hudless && HudlessWanted()) {
		ConfigureAs(graphics, inputs.hudless, inputs.hudless_view, g_hudless.format, vk::Extent3D {g_hudless.extent.width, g_hudless.extent.height, 1});
		CopyRaw(vk_command, g_hudless, inputs.hudless);
		inputs.has_hudless       = true;
		inputs.hudless_composite = composite;
	}
	{
		// Every 600 frames with the TAA pass: how the HUD-less image was had (the composite without its HUD is the
		// good one; with the HUD, the frame's HUD is interpolated).
		static uint32_t frames = 0, composites = 0, with_hud = 0, flip_draws_seen = 0;
		++frames;
		composites += composite ? 1u : 0u;
		with_hud += hudless && !composite ? 1u : 0u;
		flip_draws_seen += std::exchange(g_flip_slot_hits, 0u) != 0 ? 1u : 0u;
		if (frames == 600) {
			std::printf("[hudless] 600 frames: composite without HUD %u, copy with HUD %u, none %u; flip draw on the normal path %u\n",
			            composites, with_hud, frames - composites - with_hud, flip_draws_seen);
			std::fflush(stdout);
			frames = composites = with_hud = flip_draws_seen = 0;
		}
	}
	if (g_hudless_logs < 12) {
		++g_hudless_logs;
		std::printf("HUD-less: %u render passes into the flip image this frame (most draws in one: %u), %s\n",
		            flip_renders, flip_draws, !hudless ? "none" : composite ? "the composite drawn without its HUD" : "copy after the first draw (with the HUD)");
		std::fflush(stdout);
	}
	inputs.render_extent = {motion.backing.extent.width, motion.backing.extent.height};
	inputs.camera        = camera;
	inputs.guest_frame   = g_guest_frame;
	inputs.valid         = true;
}

void DestroyInputs(GraphicContext& graphics, FrameInputs& inputs) {
	for (auto* view: {&inputs.depth_view, &inputs.motion_view, &inputs.hudless_view, &inputs.hudless_display_view}) {
		if (*view != nullptr) {
			graphics.device.destroyImageView(*view, nullptr);
			*view = nullptr;
		}
	}
	for (auto* image: {&inputs.depth, &inputs.motion, &inputs.hudless, &inputs.hudless_display}) {
		if (image->image != nullptr) {
			graphics.DeleteImage(*image);
		}
	}
	inputs.valid = false;
}

void ScaleHudless(GraphicContext& graphics, vk::CommandBuffer command, FrameInputs& inputs, vk::Extent2D backbuffer,
                  vk::Format format) {
	inputs.has_hudless_display = false;
	if (!inputs.valid || !inputs.has_hudless || inputs.hudless.image == nullptr) return;
	ConfigureAs(graphics, inputs.hudless_display, inputs.hudless_display_view, format,
	            vk::Extent3D {backbuffer.width, backbuffer.height, 1});
	Transition(command, inputs.hudless, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
	           vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eTransfer);
	Transition(command, inputs.hudless_display, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
	           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
	vk::ImageBlit region {};
	region.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	region.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	region.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(inputs.hudless.extent.width),
                                         static_cast<int32_t>(inputs.hudless.extent.height), 1};
	region.dstOffsets[1]  = vk::Offset3D {static_cast<int32_t>(backbuffer.width), static_cast<int32_t>(backbuffer.height), 1};
	command.blitImage(inputs.hudless.image, vk::ImageLayout::eTransferSrcOptimal, inputs.hudless_display.image,
	                  vk::ImageLayout::eTransferDstOptimal, 1, &region, vk::Filter::eLinear);
	for (auto* image: {&inputs.hudless, &inputs.hudless_display})
		Transition(command, *image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
		           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
	inputs.has_hudless_display = true;
}

} // namespace Libs::Graphics::FrameGen
