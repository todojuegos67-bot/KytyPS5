#include "frame-gen.h"
#include "kbm-warp.h"
#include "graphics/presentation/window/hostInput.h"
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
#include <mutex>
#include <unordered_map>
#include <utility>
#include "graphics/host_gpu/renderer/renderTarget.h"
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
#include "local/vulkan-recording.h"
#include "local/dyn-multiplier.h"
#include "local/hwrt-gi.h"
#include "gpu_tiler_shaders/taa_bypass_spv.h"
#include "gpu_tiler_shaders/sharpen_spv.h"
#include "gpu_tiler_shaders/ssgi_spv.h"
#include "gpu_tiler_shaders/gtao_spv.h"
#include "gpu_tiler_shaders/ssr_spv.h"
#include "local/rt-reflections.h"
#include "local/gbuf-view.h"
#include "local/shine.h"
#include "local/godrays.h"
#include "local/rt-local-lights.h"
#include "local/hw-lights.h"
#include "gpu_tiler_shaders/cshadow_spv.h"
#include "gpu_tiler_shaders/rtsun_spv.h"
#include "gpu_tiler_shaders/rtsun_hw_spv.h"
#include "local/hwrt-scene.h"
#include "kernel/memory.h"
#include <string>
#endif

#include <algorithm>
#include <atomic>
#include <bit>
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
#include <intrin.h>
#undef min
#undef max
// NVIDIA Streamline SDK headers (_Build/deps/streamline/sdk/include, see CMakeLists.txt).
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss.h>
#include <sl_dlss_g.h>
#include <sl_helpers.h>
#include <sl_pcl.h>
#include <sl_reflex.h>
#include <thread>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
#endif

extern "C" {
// DLSS-G frames' Reflex sleep (slReflexSleep, every present): 0 skips it (KYTY_FRAMEGEN_SLEEP=0).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_framegen_sleep {1};
}

namespace Libs::Graphics::FrameGen {

namespace {

struct Sources {
	ImageId  depth, motion;
	uint64_t depth_address = 0, motion_address = 0;
	bool     valid = false;
};

int      g_generate    = 0;     // frames to generate per rendered frame (0 = off)
// KYTY_DLSSG_DYNAMIC=1: an app-side dynamic multiplier. Streamline's eDynamic is D3D12 only (not on Vulkan, and it
// ignores numFramesToGenerate), so DLSS-G stays eOn and the emulator picks numFramesToGenerate itself, once a second:
// m = gen + 1 = the largest in [1, min(KYTY_FRAMEGEN (default 5), numFramesToGenerateMax (3 until known)) + 1] with
// real * m <= target (10-22; up after real * (m + 1) <= 0.97 target for ~1 s, down at once past 1.02 target). The real
// rate is never held to reach a multiplier; Reflex caps it at target / m. Target: KYTY_DLSSG_TARGET_FPS, default
// refresh - refresh^2 / 3600 (224 on 240 Hz).
bool     g_dlssg_dynamic   = false;
float    g_dlssg_target_fps = 0.0f;
int      g_dyn_gen         = 1;   // the current generated frame count in dynamic mode (g_generate is its maximum)
int      g_dyn_pending     = 0;   // a candidate value and how many windows in a row it came out
int      g_dyn_pending_n   = 0;
double   g_dyn_real_fps    = 0.0; // smoothed real fps
// 10-21: when one more generated frame would pass the target but costs at most ~10% of the real rate (60 real on
// 224: x4 at 56 real = 224 instead of x3 = 180), that multiplier is taken and Reflex holds the real rate at
// target / (gen + 1). KYTY_DLSSG_DYN_STRETCH=0: the old floor choice (no limit).
bool     g_dyn_stretch     = true;
uint32_t g_dyn_limit_us    = 0;
double   g_dyn_slept       = 0.0; // seconds our limiter waited in this window
int      g_dyn_block_s     = 0;   // seconds left without taking the extra frame (it cost too much real fps)
int      g_dyn_max         = 3;   // 10-22: the largest generated count in use (min(KYTY_FRAMEGEN, GPU max), 3 until known)
int      g_dyn_up_s        = 0;   // windows in a row the next multiplier fit under target * 0.97
int      g_dyn_pinned_s    = 0;   // windows in a row the real rate sat at our Reflex cap
int      g_dyn_noup_s      = 0;   // windows left without switching up (after a step down off the cap)
Kyty::Local::DynMultiplier g_dyn_dm; // 10-22: the free-rate multiplier choice
bool     g_fixed_floor     = false; // 10-23: fixed DLSS-G under a cap: the same flip floor / pacer choice per present
int      g_dyn_flip_asked  = -1;  // the last rate asked of SetDynamicFlipFps (-1: none since a clear)
int      g_dyn_flip_fps    = 0;   // 10-21: the game's own flip floor held for the extra frame (VideoOut::SetDynamicFlipFps)

// The flip floor for `gen` generated frames: the reach rate when gen + 1 at 60 passes the target, else none.
void DynApplyFlipFloor(int gen) {
	const int n    = gen + 1;
	const int want = 60.0 * n > static_cast<double>(g_dlssg_target_fps) + 0.5
	                     ? VideoOut::ReachFps(static_cast<double>(g_dlssg_target_fps), n)
	                     : 0;
	// (asked again only on a change: a refusal is printed once, not per present)
	if ((want >= 60 ? 0 : want) == g_dyn_flip_asked) return;
	g_dyn_flip_asked = want >= 60 ? 0 : want;
	const int fps    = VideoOut::SetDynamicFlipFps(g_dyn_flip_asked);
	if (fps != g_dyn_flip_fps) {
		if (fps > 0) std::printf("[dyn] x%d: game flips at %d (gameplay)\n", n, fps);
		else std::printf("[dyn] x%d: game flips at 60\n", n);
		std::fflush(stdout);
	}
	g_dyn_flip_fps = fps;
}

int CurrentGenerate() {
	return g_dlssg_dynamic ? std::min(g_dyn_gen, g_generate) : g_generate;
}
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
// DLSS-G under the G-SYNC cap: the real frames' least interval (seconds)
std::atomic<double> g_real_interval {0.0};
double g_pace_n = 0.0, g_pace_sum = 0.0, g_pace_sq = 0.0; // real present intervals this window
bool g_sr_ready = false;
bool     g_ready       = false; // Streamline loaded and the feature available
bool     g_upscale     = false; // KYTY_UPSCALER=dlss: DLSS super resolution asked for (Streamline loaded for it)
bool     g_upscale_ready = false; // ... and available on the device
// KYTY_UPSCALER=xess|fsr: the PS5 Pro path (renderSuperResolution.cpp) upscaled by our XeSS or FSR 3.1 instead of DLSS.
int      g_sr_kind = 0; // 0 none, 1 DLSS, 2 XeSS, 3 FSR
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

} // namespace

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
               const VulkanImage& source, bool* attachment) {
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

namespace {

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
	PFun_slDLSSGetOptimalSettings* dlss_get_optimal_settings = nullptr;
};
sl::ReflexOptions g_reflex_opts {}; // kept for the dynamic multiplier's limit changes
Api              g_api;
sl::FrameToken*  g_token = nullptr; // present thread: the frame being presented
uint32_t         g_frame_index = 0;
bool             g_options_set = false;
vk::Extent2D     g_options_backbuffer {};
vk::Extent2D     g_options_render {};
vk::Format       g_options_format = vk::Format::eUndefined; // SDR or HDR10 swapchain
uint64_t         g_presents = 0;
int              g_mode_set = -1; // the DLSS-G mode last set (1 on, 0 off)
std::atomic<bool> g_discontinuity {false}; // a frame with inputs left unshown (NoteDroppedFrame): the next resets
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

// DLSS evaluations NGX refused (Streamline's slEvaluateFeature says eOk all the same; its log has them): its output
// image is then never written, the game black (issue #87).
std::atomic<uint32_t> g_dlss_refusals {0};

void LogMessage(sl::LogType type, const char* message) {
	LOGF("Streamline[%d]: %s", static_cast<int>(type), message);
	if (type == sl::LogType::eError && message != nullptr && std::strstr(message, "[sl.dlss]") != nullptr &&
	    std::strstr(message, "evaluate feature failed") != nullptr)
		g_dlss_refusals.fetch_add(1, std::memory_order_relaxed);
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

// KYTY_DLSSG_PRESET=A|B|C|0: the DLSS Frame Generation preset, for this process only (the driver profile is not
// touched). nvngx_dlssg.dll reads it through NVAPI DRS GetSetting (nvapi_QueryInterface 0x73BF8338), setting
// 0x10E41DF1: 0 game/driver, 1 A, 2 B, 3 C ("Performance mode": a lighter model, DLSS-G >= 310.9.16; earlier
// builds ignore the value, so C is then left to the driver). The provider caches its answer once per process,
// so the hook must be in place before Streamline loads the provider: nvapi64.dll's nvapi_QueryInterface export
// is redirected (export table, a jump stub next to the image) before sl.interposer.dll loads, and its answer for
// 0x73BF8338 is wrapped. Everything else is forwarded unchanged.
// Approach and NVAPI ABI after MFG-Unlocker (github.com/dashdogy/MFG-Unlocker, source/native/dlssg_preset*,
// dlssg_provider_policy*), MIT, Copyright (c) 2026 Michael Robles (LICENSES/MFG-Unlocker-MIT.txt).
namespace DlssgPreset {

constexpr uint32_t kGetSettingInterface = 0x73BF8338u;
constexpr uint32_t kSettingId           = 0x10E41DF1u;
constexpr int32_t  kOk                  = 0;
constexpr int32_t  kSettingNotFound     = -160;

#pragma pack(push, 4)
union SettingValue {
	uint32_t dword;
	struct {
		uint32_t length;
		uint8_t  data[4096];
	} binary;
	uint16_t text[2048];
	uint64_t qword;
};
struct Setting { // NVDRS_SETTING_V1
	uint32_t     version;
	uint16_t     name[2048];
	uint32_t     id;
	uint32_t     type;
	uint32_t     location;
	uint32_t     current_predefined;
	uint32_t     predefined_valid;
	SettingValue predefined;
	SettingValue current;
};
#pragma pack(pop)
static_assert(sizeof(Setting) == 12320);
constexpr uint32_t kSettingVersion = sizeof(Setting) | (1u << 16);

using QueryInterfaceFn = void*(__cdecl*)(uint32_t);
using GetSettingFn     = int32_t(__cdecl*)(void*, void*, uint32_t, Setting*);

uint32_t                  g_preset = 0; // 0 off, else 1..3
std::atomic<QueryInterfaceFn> g_query {nullptr};
std::atomic<GetSettingFn>     g_get_setting {nullptr};
std::atomic<uint64_t>         g_reads {0};
std::atomic<uint64_t>         g_applied {0};
std::atomic<int>              g_provider_c {-1}; // -1 unknown, 0 lacks C, 1 has C
std::mutex                    g_log_mutex;

// The module's VS_FIXEDFILEINFO (from its own resources: no version.dll).
bool ModuleVersion(HMODULE module, uint16_t v[4]) {
	HRSRC resource = FindResourceW(module, MAKEINTRESOURCEW(1), MAKEINTRESOURCEW(16));
	if (resource == nullptr) return false;
	HGLOBAL     handle = LoadResource(module, resource);
	const DWORD size   = SizeofResource(module, resource);
	const auto* bytes  = handle != nullptr ? static_cast<const uint8_t*>(LockResource(handle)) : nullptr;
	if (bytes == nullptr) return false;
	for (DWORD i = 0; i + 52 <= size; i += 4) {
		uint32_t signature = 0;
		std::memcpy(&signature, bytes + i, 4);
		if (signature != 0xFEEF04BDu) continue;
		uint32_t ms = 0, ls = 0;
		std::memcpy(&ms, bytes + i + 8, 4);
		std::memcpy(&ls, bytes + i + 12, 4);
		v[0] = static_cast<uint16_t>(ms >> 16);
		v[1] = static_cast<uint16_t>(ms);
		v[2] = static_cast<uint16_t>(ls >> 16);
		v[3] = static_cast<uint16_t>(ls);
		return true;
	}
	return false;
}

char Letter(uint32_t preset) { return preset == 0 ? '0' : static_cast<char>('A' + preset - 1); }

// Whether the reading module (nvngx_dlssg.dll) implements Preset C; logs once.
bool ProviderHasC(const void* caller) {
	int known = g_provider_c.load(std::memory_order_acquire);
	if (known >= 0) return known == 1;
	std::lock_guard lock(g_log_mutex);
	known = g_provider_c.load(std::memory_order_acquire);
	if (known >= 0) return known == 1;
	HMODULE module = nullptr;
	GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                   static_cast<LPCWSTR>(caller), &module);
	char     name[MAX_PATH] = "?";
	uint16_t v[4] {};
	const bool versioned = module != nullptr && ModuleVersion(module, v);
	if (module != nullptr && GetModuleFileNameA(module, name, MAX_PATH) != 0) {
		if (const char* slash = std::strrchr(name, '\\'); slash != nullptr) std::memmove(name, slash + 1, std::strlen(slash));
	}
	const uint64_t version = versioned ? (uint64_t(v[0]) << 32) | (uint64_t(v[1]) << 16) | v[2] : 0;
	const bool     has_c   = versioned && version >= ((uint64_t(310) << 32) | (uint64_t(9) << 16) | 16);
	if (g_preset == 3) {
		LOGF("[dlssg] preset C requested; provider %s %u.%u.%u: %s\n", name, v[0], v[1], v[2],
		     has_c ? "applied" : "not supported (needs >= 310.9.16), left to the driver");
		std::printf("[dlssg] preset C requested; provider %s %u.%u.%u: %s\n", name, v[0], v[1], v[2],
		            has_c ? "applied" : "not supported (needs >= 310.9.16)");
	} else {
		LOGF("[dlssg] preset %c requested; provider %s %u.%u.%u: applied\n", Letter(g_preset), name, v[0], v[1], v[2]);
		std::printf("[dlssg] preset %c requested; provider %s %u.%u.%u: applied\n", Letter(g_preset), name, v[0], v[1],
		            v[2]);
	}
	std::fflush(stdout);
	g_provider_c.store(has_c ? 1 : 0, std::memory_order_release);
	return has_c;
}

__declspec(noinline) int32_t __cdecl GetSettingHook(void* session, void* profile, uint32_t id, Setting* setting) {
	const GetSettingFn original = g_get_setting.load(std::memory_order_acquire);
	const int32_t      result   = original(session, profile, id, setting);
	if (id != kSettingId || setting == nullptr || setting->version != kSettingVersion) return result;
	if (!(result == kSettingNotFound || (result == kOk && setting->id == id && setting->type == 0))) return result;
	// Only nvngx_dlssg*.dll's own reads are answered (as MFG-Unlocker does: its override is limited to the
	// retained DLSS-G provider); other readers of the key (NGX core, Streamline, the driver) keep the driver's
	// value. Each distinct reader is logged once, to see who reads the key. KYTY_DLSSG_PRESET_ALL=1: every reader.
	{
		HMODULE caller = nullptr;
		GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		                   static_cast<LPCWSTR>(_ReturnAddress()), &caller);
		char path[MAX_PATH] = "?";
		if (caller != nullptr) GetModuleFileNameA(caller, path, MAX_PATH);
		const char* slash = std::strrchr(path, '\\');
		const char* name  = slash != nullptr ? slash + 1 : path;
		const bool  provider = _strnicmp(name, "nvngx_dlssg", 11) == 0;
		static const bool all = [] {
			const char* v = std::getenv("KYTY_DLSSG_PRESET_ALL");
			return v != nullptr && *v != '\0' && *v != '0';
		}();
		{
			static HMODULE seen[16] {};
			std::lock_guard lock(g_log_mutex);
			bool known = false;
			for (HMODULE& m: seen) {
				if (m == caller) { known = true; break; }
				if (m == nullptr) { m = caller; break; }
			}
			if (!known) {
				LOGF("[dlssg] FG preset key read by %s (driver %s %u): %s\n", name, result == kOk ? "value" : "none",
				     result == kOk ? setting->current.dword : 0u, provider || all ? "overridden" : "left as is");
				std::printf("[dlssg] FG preset key read by %s (driver %s %u): %s\n", name,
				            result == kOk ? "value" : "none", result == kOk ? setting->current.dword : 0u,
				            provider || all ? "overridden" : "left as is");
				std::fflush(stdout);
			}
		}
		if (!provider && !all) return result;
	}
	const uint64_t reads = g_reads.fetch_add(1, std::memory_order_relaxed) + 1;
	const uint32_t was   = result == kOk ? setting->current.dword : 0;
	const bool has_c = ProviderHasC(_ReturnAddress()); // (Also logs the provider once.)
	if (g_preset == 3 && !has_c) return result;
	if (result == kSettingNotFound) {
		std::memset(setting, 0, sizeof(*setting));
		setting->version = kSettingVersion;
		setting->id      = id;
		constexpr char label[] = "Override DLSS-FG preset";
		for (size_t i = 0; i < sizeof(label); ++i) setting->name[i] = static_cast<uint16_t>(label[i]);
	}
	setting->location           = 0;
	setting->current_predefined = 0;
	std::memset(&setting->current, 0, sizeof(setting->current));
	setting->current.dword = g_preset;
	const uint64_t applied = g_applied.fetch_add(1, std::memory_order_relaxed) + 1;
	if ((reads & (reads - 1)) == 0) {
		LOGF("[dlssg] FG preset read #%llu by the provider (driver %s %u, answered %c, %llu overridden)\n",
		     static_cast<unsigned long long>(reads), result == kOk ? "value" : "none", was, Letter(g_preset),
		     static_cast<unsigned long long>(applied));
	}
	return kOk;
}

void* __cdecl QueryInterfaceHook(uint32_t id) {
	void* function = g_query.load(std::memory_order_acquire)(id);
	if (id != kGetSettingInterface || function == nullptr) return function;
	GetSettingFn expected = nullptr;
	if (!g_get_setting.compare_exchange_strong(expected, reinterpret_cast<GetSettingFn>(function)) &&
	    expected != reinterpret_cast<GetSettingFn>(function)) {
		return function; // (A second, different implementation: left alone.)
	}
	return reinterpret_cast<void*>(&GetSettingHook);
}

// Points nvapi64.dll's nvapi_QueryInterface export at QueryInterfaceHook (later GetProcAddress calls and imports
// resolve to it). Before sl.interposer.dll loads.
void Install() {
	const char* value = std::getenv("KYTY_DLSSG_PRESET");
	if (value == nullptr || *value == '\0') return;
	switch (*value) {
		case 'A': case 'a': case '1': g_preset = 1; break;
		case 'B': case 'b': case '2': g_preset = 2; break;
		case 'C': case 'c': case '3': g_preset = 3; break;
		case '0': return; // game / driver
		default: LOGF("[dlssg] KYTY_DLSSG_PRESET=%s: not A, B, C or 0, ignored\n", value); return;
	}
	const auto fail = [](const char* why) {
		LOGF("[dlssg] preset %c requested: not applied (%s)\n", Letter(g_preset), why);
		std::printf("[dlssg] preset %c requested: not applied (%s)\n", Letter(g_preset), why);
		g_preset = 0;
	};
	HMODULE nvapi = LoadLibraryExW(L"nvapi64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
	if (nvapi == nullptr) return fail("no nvapi64.dll");
	HMODULE pinned = nullptr; // (Never unloaded: the export now points into our stub.)
	GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
	                   reinterpret_cast<LPCWSTR>(nvapi), &pinned);
	auto*       base = reinterpret_cast<uint8_t*>(nvapi);
	const auto* dos  = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
	const auto* nt   = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
	const auto& dir  = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
	if (dir.VirtualAddress == 0) return fail("nvapi64.dll has no exports");
	const auto* exports   = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(base + dir.VirtualAddress);
	const auto* names     = reinterpret_cast<const DWORD*>(base + exports->AddressOfNames);
	const auto* ordinals  = reinterpret_cast<const WORD*>(base + exports->AddressOfNameOrdinals);
	auto*       functions = reinterpret_cast<DWORD*>(base + exports->AddressOfFunctions);
	DWORD*      slot      = nullptr;
	for (DWORD i = 0; i < exports->NumberOfNames; ++i) {
		if (std::strcmp(reinterpret_cast<const char*>(base + names[i]), "nvapi_QueryInterface") == 0) {
			slot = &functions[ordinals[i]];
			break;
		}
	}
	if (slot == nullptr) return fail("no nvapi_QueryInterface");
	auto* original = reinterpret_cast<QueryInterfaceFn>(GetProcAddress(nvapi, "nvapi_QueryInterface"));
	if (original == nullptr || reinterpret_cast<uint8_t*>(original) != base + *slot) return fail("unexpected export");
	// A jump stub above the image, within the 32-bit export RVA range.
	const uintptr_t image_end = reinterpret_cast<uintptr_t>(base) + nt->OptionalHeader.SizeOfImage;
	uint8_t*        stub      = nullptr;
	for (uintptr_t at = (image_end + 0xFFFF) & ~uintptr_t(0xFFFF);
	     stub == nullptr && at < reinterpret_cast<uintptr_t>(base) + 0x7FFF0000u; at += 0x10000) {
		stub = static_cast<uint8_t*>(
		    VirtualAlloc(reinterpret_cast<void*>(at), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
	}
	if (stub == nullptr) return fail("no memory near nvapi64.dll");
	const uint64_t target = reinterpret_cast<uint64_t>(&QueryInterfaceHook);
	stub[0] = 0xFF; stub[1] = 0x25; std::memset(stub + 2, 0, 4); // jmp [rip+0]
	std::memcpy(stub + 6, &target, 8);
	DWORD protection = 0;
	VirtualProtect(stub, 4096, PAGE_EXECUTE_READ, &protection);
	FlushInstructionCache(GetCurrentProcess(), stub, 14);
	g_query.store(original, std::memory_order_release);
	if (!VirtualProtect(slot, sizeof(DWORD), PAGE_READWRITE, &protection)) return fail("export table is read-only");
	*slot = static_cast<DWORD>(reinterpret_cast<uintptr_t>(stub) - reinterpret_cast<uintptr_t>(base));
	VirtualProtect(slot, sizeof(DWORD), protection, &protection);
	LOGF("[dlssg] preset %c requested: NVAPI DRS read hook installed (driver profile untouched)\n", Letter(g_preset));
	if (GetModuleHandleW(L"nvngx_dlssg.dll") != nullptr) LOGF("[dlssg] warning: nvngx_dlssg.dll was already loaded\n");
}

} // namespace DlssgPreset

#endif

} // namespace

bool Enabled() {
	return g_ready;
}

bool DlssUpscaleWanted();
bool TaaReplaceWanted();
bool Capturing() {
	// (KYTY_KBM_WARP: the frame's HUD-less composite, so that only the 3D is reprojected; kbm-warp.h.)
	return Enabled() || Fsr3::Active() || Fsr3::UpscaleWanted() || DlssUpscaleWanted() || Xess::Wanted() ||
	       TaaReplaceWanted() || KbmWarp::HudlessWanted();
}

bool Requested() {
	return g_generate > 0;
}

// 10-22: KYTY_DLSSG_PACE: how real frames are paced for DLSS-G.
//   adaptive (opt-in): VideoOut's flip pacer at max(target / m, mean + 0.75 sd of the last windows' present
//     intervals). Below the cap (log 20261022-103006: 43-51 real, 21 ms mean) nothing held the frames, so the game's
//     own frame-time spread reached DLSS-G (present interval sd 1.1-1.9 ms; FSR3's 0.37 was at a held 60); a steady
//     interval just above the typical frame evens it for a few % of real rate.
//   grid (default): the flip pacer at target / m only. Adaptive is held to at most 1.15 target / m and reset in
//   cutscenes, cinematics, movies and with DLSS-G off (it learned a 30 fps phase and held its own 36 ms after it).
//   reflex: Streamline's Reflex limit at target / m and our pacer off.
//   reflexfree (default with KYTY_FRAMEGEN_BASE=0, the game unlocked): as FSR3 unlocked (log 20261022-193327: smooth
//     at 45-60 real, present interval sd 1.08, flip->present 33.6 ms; DLSS-G fixed x4 locked, 193116: sd 1.74,
//     flip->present 48.7, 11.9 frames over 25 ms a window): no flip pacer and no flip floor on real frames, the only
//     limit Reflex's frameLimitUs = 1e6 m / target (its sleep, at the guest frame start, holds the game), flips when
//     ready; the game's own variable step above 60.
int DlssgPaceMode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_DLSSG_PACE");
		if (v != nullptr && std::strcmp(v, "adaptive") == 0) return 0;
		if (v != nullptr && std::strcmp(v, "grid") == 0) return 1;
		if (v != nullptr && std::strcmp(v, "reflex") == 0) return 2;
		if (v != nullptr && std::strcmp(v, "reflexfree") == 0) return 3;
		// 10-23: the game unlocked (KYTY_FRAMEGEN_BASE=0): reflexfree, as FSR3 unlocked runs.
		const char* base = std::getenv("KYTY_FRAMEGEN_BASE");
		if (base != nullptr && std::strcmp(base, "0") == 0) return 3;
		return 1; // grid (10-22: adaptive locked at 36 ms after a 30 fps phase, log 20261022-175339)
	}();
	return mode;
}
std::atomic<double> g_pace_adapt {0.0}; // adaptive: the steady interval from the present interval statistics

double RealFrameInterval() {
	if (!g_ready || DlssgPaceMode() >= 2) return 0.0;
	const double base = g_real_interval.load();
	return base > 0.0 && DlssgPaceMode() == 0 ? std::clamp(g_pace_adapt.load(), base, 1.15 * base) : base;
}

std::string SupportProblem(vk::PhysicalDevice physical_device) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (g_upscale) {
		sl::AdapterInfo adapter {};
		adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
		const auto result        = g_api.is_feature_supported(sl::kFeatureDLSS, adapter);
		std::printf("DLSS super resolution: %s\n", result == sl::Result::eOk
		                                               ? "Streamline offers it with this GPU and driver"
		                                               : (std::string("not offered here (") + sl::getResultAsStr(result) +
		                                                  "; an NVIDIA RTX GPU and a recent driver)")
		                                                     .c_str());
		std::fflush(stdout);
	}
#endif
	if (g_generate == 0) {
		return {};
	}
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// (Streamline answers before the device too: its programming guide asks while the adapters are enumerated. Issue
	// #45: a device refused with no word why; the driver's own refusal says nothing either.)
	sl::AdapterInfo adapter {};
	adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
	const auto  result       = g_api.is_feature_supported(sl::kFeatureDLSS_G, adapter);
	std::string problem;
	switch (result) {
		case sl::Result::eOk: break;
		case sl::Result::eErrorOSDisabledHWS:
			problem = "Windows' hardware-accelerated GPU scheduling is off (Settings > System > Display > Graphics > "
			          "Advanced graphics settings; then restart the PC)";
			break;
		case sl::Result::eErrorOSOutOfDate: problem = "this Windows version is too old for DLSS frame generation"; break;
		case sl::Result::eErrorDriverOutOfDate: problem = "the NVIDIA driver is too old for DLSS frame generation"; break;
		case sl::Result::eErrorNoSupportedAdapterFound:
		case sl::Result::eErrorAdapterNotSupported:
			problem = "this graphics card has no DLSS frame generation (NVIDIA RTX 40 or newer)";
			break;
		default: problem = std::string("Streamline does not offer DLSS frame generation here (") + sl::getResultAsStr(result) + ")"; break;
	}
	if (problem.empty()) {
		LOGF("frame generation: Streamline offers DLSS-G with this GPU, driver and Windows\n");
		std::printf("Frame generation: Streamline offers DLSS-G with this GPU, driver and Windows\n");
	} else {
		LOGF("frame generation: %s\n", problem.c_str());
		std::printf("Frame generation: %s\n", problem.c_str());
	}
	// (Where the driver's reservations go, for a refused device's log: RefusedDeviceAdvice.)
	PROCESS_MITIGATION_ASLR_POLICY aslr {};
	if (GetProcessMitigationPolicy(GetCurrentProcess(), ProcessASLRPolicy, &aslr, sizeof(aslr))) {
		std::printf("Frame generation: bottom-up ASLR %s for the emulator\n", aslr.EnableBottomUpRandomization != 0 ? "on" : "off");
	}
	std::fflush(stdout);
	return problem;
#else
	(void)physical_device;
	return {};
#endif
}

std::string RefusedDeviceAdvice() {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// (The emulator is built with fixed addresses, so without bottom-up ASLR its host allocations pack upward from the
	// guest's ceiling; an RTX 5080's driver refused the device until Exploit protection forced it on for the emulator.)
	PROCESS_MITIGATION_ASLR_POLICY aslr {};
	if (GetProcessMitigationPolicy(GetCurrentProcess(), ProcessASLRPolicy, &aslr, sizeof(aslr)) &&
	    aslr.EnableBottomUpRandomization == 0) {
		return " If it keeps failing, in Windows Security: App & browser control > Exploit protection > Program settings > "
		       "add kyty_emulator.exe > Randomize memory allocations (Bottom-up ASLR): Override system settings, On "
		       "(issue #45).";
	}
#endif
	return {};
}

bool StreamlineLoaded() {
	return g_generate > 0 || g_dlss_sr || g_taa_dlss || g_upscale;
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

bool InputsGathered() {
	return g_sources.valid && g_camera.valid;
}

void BypassCommandHooks(vk::Device device) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// KYTY_FRAMEGEN_HOOKS=1 keeps Streamline's wrappers (for comparison).
	if ((g_generate == 0 && !g_dlss_sr && !g_taa_dlss && !g_upscale) || std::getenv("KYTY_FRAMEGEN_HOOKS") != nullptr) {
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

namespace { int SsgiQuality(); }
void Initialize() {
	{
		const char* ssgi = std::getenv("KYTY_SSGI");
		std::printf("SSGI: KYTY_SSGI=%s read (quality %d)\n", ssgi != nullptr ? ssgi : "(unset)", SsgiQuality());
		std::fflush(stdout);
	}
	const char* value = std::getenv("KYTY_FRAMEGEN");
	g_generate        = value != nullptr ? std::clamp(std::atoi(value), 0, 5) : 0;
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (const char* dyn = std::getenv("KYTY_DLSSG_DYNAMIC"); dyn != nullptr && *dyn == '1') {
		if (value == nullptr) g_generate = 5; // KYTY_FRAMEGEN is the maximum (default 5: x6, capped by the GPU)
	}
	if (const char* dyn = std::getenv("KYTY_DLSSG_DYNAMIC"); dyn != nullptr && *dyn == '1' && g_generate > 0) {
		g_dlssg_dynamic = true;
		if (const char* st = std::getenv("KYTY_DLSSG_DYN_STRETCH"); st != nullptr && *st == '0') g_dyn_stretch = false;
		if (const char* t = std::getenv("KYTY_DLSSG_TARGET_FPS"); t != nullptr && std::atof(t) > 0) {
			g_dlssg_target_fps = static_cast<float>(std::atof(t));
		} else {
			DEVMODEA dm {};
			dm.dmSize = sizeof(dm);
			if (EnumDisplaySettingsA(nullptr, ENUM_CURRENT_SETTINGS, &dm) != 0 && dm.dmDisplayFrequency > 30) {
				const double hz    = dm.dmDisplayFrequency;
				g_dlssg_target_fps = static_cast<float>(std::floor(hz - hz * hz / 3600.0));
			}
		}
		if (g_dlssg_target_fps <= 0.0f) g_dlssg_target_fps = 224.0f;
		// First guess: the game's usual 60 real fps.
		g_dyn_max = std::min(g_generate, 3); // until Streamline reports numFramesToGenerateMax
		g_dyn_gen = std::clamp(static_cast<int>(std::floor(g_dlssg_target_fps / 60.0)) - 1, 1, g_dyn_max);
	}
#endif
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
		// KYTY_UPSCALER: the game's PS5 Pro path (its upscaler in place of the TAA) with DLSS, XeSS or FSR.
		const char* upscaler = std::getenv("KYTY_UPSCALER");
		const std::string kind = upscaler != nullptr ? upscaler : "";
		g_sr_kind = kind == "dlss" ? 1 : kind == "xess" ? 2 : kind == "fsr" ? 3 : 0;
		g_upscale = g_sr_kind == 1;
	}
	if (g_generate == 0 && !g_dlss_sr && !g_taa_dlss && !g_upscale) {
		return;
	}
	// (Every way out without Streamline turns both off.)
	struct Off {
		bool done = false;
		~Off() {
			if (!done) {
				g_generate = 0;
				g_upscale  = false;
			}
		}
	} off;
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
	DlssgPreset::Install(); // (Before Streamline loads nvngx_dlssg.dll.)
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
	static std::vector<sl::Feature> features;
	if (g_generate > 0) features = {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
	if (const char* list = std::getenv("KYTY_FRAMEGEN_FEATURES"); list != nullptr && g_generate > 0) {
		const std::string text = list;
		features.clear();
		if (text.find("dlssg") != std::string::npos) features.push_back(sl::kFeatureDLSS_G);
		if (text.find("reflex") != std::string::npos) features.push_back(sl::kFeatureReflex);
		if (text.find("pcl") != std::string::npos) features.push_back(sl::kFeaturePCL);
	}
	// DLSS super resolution (KYTY_UPSCALE=dlss, KYTY_TAA_REPLACE=dlss): its plugin too. Without frame generation the
	// same plugins are loaded all the same (DLSS-G then stays off): with only the DLSS plugin the emulator closed as the
	// Vulkan device was made (logs 005353, 013603), with the frame generation set it ran.
	if (g_dlss_sr || g_taa_dlss || g_upscale) features.push_back(sl::kFeatureDLSS);
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
	LOGF("frame generation: Streamline loaded from %s, %d generated frame(s), DLSS super resolution %s\n",
	     dir.string().c_str(), g_generate, g_upscale ? "asked for" : "off");
	off.done = true;
#else
	LOGF("frame generation: this build has no Streamline support\n");
	g_generate = 0;
		g_dlss_sr  = false;
		g_taa_dlss = false;
#endif
}

uint32_t DlssgReflexLimitUs(double interval);

void OnDevice(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device) {
	(void)instance;
	(void)device;
	if (g_generate == 0 && !g_dlss_sr && !g_taa_dlss && !g_upscale) {
		return;
	}
#ifdef KYTY_FRAMEGEN_STREAMLINE
	sl::AdapterInfo adapter {};
	adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
	bool loaded              = false;
	if (g_upscale) {
		bool dlss_loaded = false;
		if (g_api.is_feature_supported(sl::kFeatureDLSS, adapter) == sl::Result::eOk &&
		    g_api.is_feature_loaded(sl::kFeatureDLSS, dlss_loaded) == sl::Result::eOk && dlss_loaded &&
		    FeatureFunction(sl::kFeatureDLSS, "slDLSSSetOptions", g_api.dlss_set_options) &&
		    FeatureFunction(sl::kFeatureDLSS, "slDLSSGetOptimalSettings", g_api.dlss_get_optimal_settings)) {
			g_upscale_ready = true;
			std::printf("DLSS super resolution (PS5 Pro path): ready\n");
		} else {
			std::printf("DLSS super resolution (PS5 Pro path): not available on this device\n");
		}
		std::fflush(stdout);
	}
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
	sl::ReflexOptions& reflex = g_reflex_opts;
	reflex = {};
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
		// With KYTY_FLIP_FPS already holding the game at or under cap / (1 + generated) (x4: 56 on a 224 Hz vblank,
		// Reflex's limit 56 too) the emulator's flip pacing alone holds it: two limiters at the same rate beat against
		// each other (x4 uneven at 56, 10-19; x3 and x5, whose Reflex limit is above their flip rate, were smooth).
		// KYTY_DLSSG_REFLEX_LIMIT=1: Reflex's limit kept anyway.
		const int  flip_fps     = [] {
			const char* v = std::getenv("KYTY_FLIP_FPS");
			return v != nullptr ? std::atoi(v) : 0;
		}();
		const bool force_reflex = [] {
			const char* v = std::getenv("KYTY_DLSSG_REFLEX_LIMIT");
			return v != nullptr && *v == '1';
		}();
		if (g_dlssg_dynamic) {
			// 10-22: Reflex holds real <= target / multiplier, set again on every change (KYTY_VRR_CAP_FPS not used).
			cap                 = 0;
			g_real_interval     = static_cast<double>(g_dyn_gen + 1) / static_cast<double>(g_dlssg_target_fps);
			reflex.frameLimitUs = DlssgReflexLimitUs(g_real_interval);
			std::printf("Frame generation: dynamic multiplier, target %.0f fps: Reflex limit %u us (x%d)\n",
			            static_cast<double>(g_dlssg_target_fps), reflex.frameLimitUs, g_dyn_gen + 1);
		}
		if (cap != 0 && g_generate > 0 && flip_fps > 0 && !force_reflex &&
		    static_cast<double>(flip_fps) * (g_generate + 1) <= cap + 0.5 &&
		    static_cast<double>(cap) / (g_generate + 1) <= flip_fps * 1.03) { // (only where both bind: x4; others as they were)
			std::printf("Frame generation: shown frames %d x %d = %d (KYTY_FLIP_FPS), under %u: no Reflex limit\n",
			            flip_fps, g_generate + 1, flip_fps * (g_generate + 1), cap);
			cap = 0;
		}
		if (cap != 0 && g_generate > 0) {
			g_dlssg_target_fps  = static_cast<float>(cap); // 10-23: fixed too: the flip floor (DynApplyReflexLimit per present)
			g_fixed_floor       = true;
			g_real_interval     = static_cast<double>(g_generate + 1) / cap;
			reflex.frameLimitUs = DlssgReflexLimitUs(g_real_interval);
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
	// (A frame left unshown before this one: its motion and matrices are to the game's previous frame, not to the one
	// presented last.)
	const bool discontinuity      = g_discontinuity.exchange(false, std::memory_order_relaxed);
	constants.reset               = usable && !discontinuity ? sl::Boolean::eFalse : sl::Boolean::eTrue;
	constants.orthographicProjection = sl::Boolean::eFalse;
	constants.motionVectorsDilated   = sl::Boolean::eFalse;
	constants.motionVectorsJittered  = sl::Boolean::eFalse;
	{
		// The vectors' scale and sign whether the camera was captured or not (it was set only with the camera: without
		// it the game's vectors went in as +1, pointing the wrong way).
		static const float sign = [] {
			const char* value = std::getenv("KYTY_FRAMEGEN_MV_SIGN");
			return value != nullptr && std::atof(value) > 0 ? 1.0f : -1.0f;
		}();
		static const bool px = [] {
			const char* mv = std::getenv("KYTY_FRAMEGEN_MV");
			return mv != nullptr && std::string(mv) == "px";
		}();
		if (usable && render.width != 0 && render.height != 0)
			constants.mvecScale = px ? sl::float2(sign / static_cast<float>(render.width), sign / static_cast<float>(render.height))
			                         : sl::float2(sign, sign);
	}
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


// 10-23: KYTY_DLSSG_MARKERS=guest (default) | present. Before, every Reflex marker and slReflexSleep came at the
// present: the simulation marked as taking no time, and Reflex's sleep delayed the present itself (the driver's
// pacing then saw a frame of ~0 ms simulation and a variable wait before each present). Now at the guest flip on the
// render thread: render submit end for the frame that ends there, then for the next one Reflex's sleep (it paces the
// game's frames, not the presents), simulation start/end and render submit start; present start/end at the present
// with the same frame index. present: all at the present (the old placement).
bool ReflexGuestMarkers() {
	static const bool guest = [] {
		const char* v = std::getenv("KYTY_DLSSG_MARKERS");
		return v == nullptr || std::strcmp(v, "present") != 0;
	}();
	return guest;
}
std::atomic<uint32_t> g_rx_counter {0};   // Reflex frame indices (guest and present paths)
uint32_t              g_rx_open = 0;      // render thread: the frame begun at the last guest flip
std::atomic<uint32_t> g_rx_present {0};   // the index BeforePresent presents next (SetPresentReflexIndex)

// 10-22: real frames are paced by one limiter only: VideoOut's flip pacer (RealFrameInterval, the vblank grid, as
// FSR3's) holds them to target / m; Reflex stays for latency without a frame limit (two limiters at 56 beat against
// the game's 60: flip->present sd 1.90 ms vs FSR3's 0.55). KYTY_DLSSG_REFLEX_LIMIT=1: Reflex's limit too (old);
// KYTY_DLSSG_PACE=reflex: Reflex's limit only (DlssgPaceMode).
int DlssgPaceMode();
uint32_t DlssgReflexLimitUs(double interval) {
	static const bool force = [] {
		const char* v = std::getenv("KYTY_DLSSG_REFLEX_LIMIT");
		return v != nullptr && *v == '1';
	}();
	return (force || DlssgPaceMode() >= 2) && interval > 0.0 ? static_cast<uint32_t>(1000000.0 * interval) : 0u;
}

#ifdef KYTY_FRAMEGEN_STREAMLINE
// 10-22: the dynamic multiplier's Reflex cap: real <= target / (gen + 1), so shown <= target (G-Sync range).
// gen < 0: no limit (DLSS-G off: menus, movies, loading; x1 is not held to the last multiplier's target / m).
// 10-23: where the game's 60 would pass the target (x4 on 224), the game's own flips are held to the vblank's reach
// rate (840 Hz: 56, DynApplyFlipFloor) and our pacer is off: the pacer at 56 against the game's 60 queued a frame
// (log 20261022-190358, dynamic x4 at 55 real: flip->present mean 43.7 ms sd 3.9 vs 24.1 / 1.4 on fixed x4 at 45 real).
// Not in cinematics or held cutscenes, nor with DLSS-G off. KYTY_DLSSG_FLIP_FLOOR=0: the pacer only. Fixed DLSS-G under
// a cap (KYTY_VRR_CAP_FPS) likewise (10-23); it needs a vblank that 56 divides (840 Hz; 360 Hz refuses it: the pacer).
void DynApplyReflexLimit(int gen) {
	static const bool floor_on = [] {
		const char* v = std::getenv("KYTY_DLSSG_FLIP_FLOOR");
		return v == nullptr || *v != '0';
	}();
	if (floor_on && DlssgPaceMode() < 2 && gen >= 0 && !VideoOut::CinematicPlaying() && !VideoOut::CutsceneHeld()) {
		DynApplyFlipFloor(gen);
	} else if (g_dyn_flip_fps != 0) {
		VideoOut::SetDynamicFlipFps(0);
		g_dyn_flip_asked = -1;
		g_dyn_flip_fps = 0;
		std::printf("[dyn] game flips at 60\n");
	}
	const double interval = gen < 0 || g_dyn_flip_fps != 0
	                            ? 0.0
	                            : static_cast<double>(gen + 1) / std::max(static_cast<double>(g_dlssg_target_fps), 1.0);
	if (g_real_interval.load() != interval)
		std::printf("[dyn] pacer interval %.0f us for x%d%s\n", interval * 1e6, gen + 1, g_dyn_flip_fps != 0 ? " (game flips held instead)" : "");
	g_real_interval       = interval; // the flip pacer (VideoOut, as FSR3's) holds real frames to it on the vblank grid
	const uint32_t us     = DlssgReflexLimitUs(interval);
	if (g_reflex_opts.frameLimitUs == us) return;
	g_reflex_opts.frameLimitUs = us;
	if (const auto result = g_api.reflex_set_options(g_reflex_opts); result != sl::Result::eOk)
		LOGF("frame generation: slReflexSetOptions (dynamic limit) failed (%d)\n", static_cast<int>(result));
}
#endif

void BeforePresent(const FrameInputs* inputs, vk::Extent2D backbuffer, vk::Rect2D region,
                   vk::Format format, uint32_t image_count) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!g_ready) {
		return;
	}
	const sl::ViewportHandle viewport {0};
	// Movies, menus and loading screens (no TAA pass) are presented as they are: DLSS-G off for them.
	const bool               usable = inputs != nullptr && inputs->valid && !VideoOut::MoviePlaying() && VideoOut::PlayerInputSeen();
	// 10-21: an in-engine cinematic keeps 60 real: x3 at most, no flip floor, at once (not after two windows).
	if (g_dlssg_dynamic && VideoOut::CinematicPlaying() && (g_dyn_gen > 2 || g_dyn_flip_fps != 0)) {
		g_dyn_gen       = std::min(g_dyn_gen, std::min(2, std::max(g_generate, 1)));
		g_dyn_pending_n = 0;
		g_options_set   = false;
		VideoOut::SetDynamicFlipFps(0);
		g_dyn_flip_asked = -1;
		g_dyn_flip_fps = 0;
		DynApplyReflexLimit(g_dyn_gen);
		std::printf("[dyn] cinematic: back to 60, x%d\n", g_dyn_gen + 1);
		std::fflush(stdout);
	}
	if (g_dlssg_dynamic && g_dyn_limit_us != 0 && usable) {
		static std::chrono::steady_clock::time_point next {};
		const auto step = std::chrono::microseconds(g_dyn_limit_us);
		auto       t    = std::chrono::steady_clock::now();
		if (next > t) {
			g_dyn_slept += std::chrono::duration<double>(next - t).count();
			std::this_thread::sleep_until(next - std::chrono::milliseconds(1));
			while (std::chrono::steady_clock::now() < next) std::this_thread::yield();
			t = next;
		}
		next = std::max(next, t - step) + step;
	}
	{
		const auto now = std::chrono::steady_clock::now();
		if (g_stat_start == std::chrono::steady_clock::time_point {}) g_stat_start = now;
		++g_stat_real;
		if (usable) ++g_stat_on;
		{
			static std::chrono::steady_clock::time_point prev {};
			const double dt = prev == std::chrono::steady_clock::time_point {} ? 0.0 : std::chrono::duration<double>(now - prev).count();
			prev            = now;
			if (usable && dt > 0.0 && dt < 0.1) {
				g_pace_n += 1.0;
				g_pace_sum += dt;
				g_pace_sq += dt * dt;
			}
		}
		const double window = std::chrono::duration<double>(now - g_stat_start).count();
		if (window >= 1.0) {
			const double n = g_stat_real != 0 ? static_cast<double>(g_stat_real) : 1.0;
			if (VideoOut::CinematicPlaying() || VideoOut::CutsceneHeld() || VideoOut::MoviePlaying() || g_mode_set != 1) {
				g_pace_adapt = 0.0; // (nothing learned from held or ungenerated phases)
				g_pace_n     = 0.0;
			}
			if (g_pace_n >= 10.0) {
				const double mean = g_pace_sum / g_pace_n;
				const double sd   = std::sqrt(std::max(g_pace_sq / g_pace_n - mean * mean, 0.0));
				// Adaptive pacing: 0.98 mean + 0.75 sd, smoothed (the 2% lets it come down: held frames' mean is the pacer's
				// own interval, so mean + sd alone would only ever grow). At the cap target / m wins (max).
				const double want = 0.98 * mean + 0.75 * sd;
				const double prev = g_pace_adapt.load();
				g_pace_adapt      = prev <= 0.0 ? want : 0.7 * prev + 0.3 * want;
				std::printf("[dlssg] present interval sd %.2f ms (mean %.2f, %.0f frames); pacer %.2f ms (%s)\n", sd * 1e3, mean * 1e3,
				            g_pace_n, RealFrameInterval() * 1e3,
				            DlssgPaceMode() == 0 ? "adaptive" : DlssgPaceMode() == 1 ? "grid" : DlssgPaceMode() == 2 ? "reflex" : "reflexfree");
			} else {
				g_pace_adapt = 0.0;
			}
			g_pace_n = g_pace_sum = g_pace_sq = 0.0;
			if (g_dlssg_dynamic && g_mode_set == 1) {
				const double real = g_stat_real / window;
				// Free rate: frames over the time not spent in our limiter (what the game would do unlimited).
				const double freer = g_stat_real / std::max(window - g_dyn_slept, 0.05);
				g_dyn_slept       = 0.0;
				g_dyn_real_fps    = g_dyn_real_fps <= 0.0 ? freer : 0.5 * g_dyn_real_fps + 0.5 * freer;
				// 10-22: no flip floor (the game's real rate is not held to reach a multiplier). The largest m with
				// real * m <= target; up only after real * (m + 1) <= 0.97 target held 2 windows (EMA ~1 s), down at once
				// when real * m > 1.02 target. Reflex caps real at target / m: when the real rate sits on that cap for
				// 2 windows the game may run faster, so one step down (and no step up for 10 s).
				(void)freer;
				// 10-22: decided on the free rate (src/local/dyn-multiplier.h): per-level minimum real fps, one step up
				// at a time with a revert + 10 s block when it shows < 5% more, cap target / m only binding past it.
				const double T         = static_cast<double>(g_dlssg_target_fps);
				const bool   cinematic = VideoOut::CinematicPlaying();
				const int    maxgen    = cinematic ? std::min(2, g_dyn_max) : g_dyn_max;
				int          next      = std::clamp(g_dyn_dm.Window(g_dyn_gen + 1, real, T, maxgen + 1,
				                                                      cinematic || VideoOut::CutsceneHeld() || VideoOut::MoviePlaying()) - 1, 1, std::max(maxgen, 1));
				if (next != g_dyn_gen) {
					const int from_m = g_dyn_gen + 1;
					g_dyn_gen      = next;
					g_options_set  = false; // send numFramesToGenerate again
					g_dyn_limit_us = 0;
					g_dyn_pinned_s = 0;
					if (g_dyn_flip_fps != 0) {
						VideoOut::SetDynamicFlipFps(0);
		g_dyn_flip_asked = -1;
						g_dyn_flip_fps = 0;
					}
					DynApplyReflexLimit(g_dyn_gen);
					std::printf("[dyn] x%d (from x%d): real %.1f fps, shown %.1f (target %.0f), free %.1f\n", g_dyn_gen + 1, from_m,
					            real, std::min(real * from_m, T), T, g_dyn_dm.last_free);
				}
				std::printf("[dlssg] dynamic x%d; real %.1f fps (smoothed %.1f), shown %.1f, target %.0f\n", g_dyn_gen + 1, real,
				            g_dyn_real_fps, real * (g_dyn_gen + 1), static_cast<double>(g_dlssg_target_fps));
			}
			std::printf("[dlssg] %s x%u; real %.1f fps, shown %.1f fps; generation on %u of %u frames; present call %.2f ms (max %.1f); "
			            "acquire %.2f (max %.1f), record+submit %.2f (max %.1f), gpu wait %.2f (max %.1f), before %.2f (max %.1f), flip present %.2f (max %.1f)\n",
			            g_dlssg_dynamic ? "dynamic" : "fixed", PresentedFrames(),
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
	// DLSS-G on or off (its resources kept, eRetainResourcesWhenOff): off after 30 presents in a row without inputs
	// (menus, movies, loading), on after 10 with them, so a fade or a cut does not flap it (chenxiao07 d672b238).
	static uint32_t run_with = 0, run_without = 0;
	run_with           = usable ? run_with + 1 : 0;
	run_without        = usable ? 0 : run_without + 1;
	const bool mode_on = g_mode_set == 1 ? run_without < 30 : run_with >= 10;
	// 10-22: the dynamic Reflex limit is target / m of the current m while DLSS-G is on, none while it is off.
	if (g_dlssg_dynamic) DynApplyReflexLimit(mode_on ? g_dyn_gen : -1);
	else if (g_fixed_floor) DynApplyReflexLimit(mode_on ? g_generate : -1); // 10-23: fixed x4 likewise (flip floor 56)
	// DLSS-G options; sizes are hints for its allocations.
	if (!g_options_set || g_mode_set != (mode_on ? 1 : 0) || g_options_format != format ||
	    (usable && (g_options_backbuffer != backbuffer || g_options_render != inputs->render_extent))) {
		sl::DLSSGOptions options {};
		options.mode                = mode_on ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
		options.flags               = sl::DLSSGFlags::eRetainResourcesWhenOff;
		{
			static bool once = false;
			if (!once) {
				once = true;
				if (g_dlssg_dynamic)
					std::printf("[dlssg] mode: app-side dynamic multiplier (eOn; eDynamic is D3D12 only), up to %d generated, "
					            "target %.0f fps, starting x%d\n", g_generate, static_cast<double>(g_dlssg_target_fps), g_dyn_gen + 1);
				else
					std::printf("[dlssg] mode: fixed x%d (eOn)\n", g_generate + 1);
				std::fflush(stdout);
			}
		}
		if (g_mode_set != (mode_on ? 1 : 0)) {
			std::printf("DLSS-G: %s\n", mode_on ? "on (TAA pass)" : "off (no TAA pass: menu, loading or movie)");
			std::fflush(stdout);
		}
		g_mode_set = mode_on ? 1 : 0;
		options.numFramesToGenerate = static_cast<uint32_t>(CurrentGenerate());
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

	// 10-23: a frame begun at the guest flip (ReflexGuestFlip) keeps its index; others (menus, a repeated image)
	// take a new one with all markers here, as before.
	const uint32_t guest_index = g_rx_present.exchange(0, std::memory_order_relaxed);
	g_frame_index              = guest_index != 0 ? guest_index : g_rx_counter.fetch_add(1, std::memory_order_relaxed) + 1;
	if (g_api.get_new_frame_token(g_token, &g_frame_index) != sl::Result::eOk || g_token == nullptr) {
		g_token = nullptr;
		return;
	}
	auto& token = *g_token;
	static const bool sleep = [] {
		const char* value = std::getenv("KYTY_DLSSG_SLEEP");
		return value == nullptr || *value != '0';
	}();
	if (guest_index == 0) {
		if (sleep && g_api.reflex_sleep != nullptr && !ReflexGuestMarkers()) g_api.reflex_sleep(token);
		g_api.pcl_set_marker(sl::PCLMarker::eSimulationStart, token);
		g_api.pcl_set_marker(sl::PCLMarker::eSimulationEnd, token);
		g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitStart, token);
	}

	sl::Constants constants = MakeConstants(usable, usable ? inputs->camera : Camera {}, usable ? inputs->render_extent : vk::Extent2D {});
	if (const auto result = g_api.set_constants(constants, token, viewport); result != sl::Result::eOk) {
		static uint32_t failures = 0;
		if (failures++ < 8 || failures % 600 == 0) {
			LOGF("frame generation: slSetConstants failed (%d), %u time(s)\n", static_cast<int>(result), failures);
			std::printf("[dlssg] slSetConstants failed (%d), %u time(s)\n", static_cast<int>(result), failures);
		}
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
		const auto tag_result = g_api.set_tag_for_frame(token, viewport, tags.data(), static_cast<uint32_t>(tags.size()), nullptr);
		static uint32_t tag_failures = 0;
		if (tag_result != sl::Result::eOk && (tag_failures++ < 8 || tag_failures % 600 == 0)) {
			LOGF("frame generation: slSetTagForFrame failed (%d), %u time(s)\n", static_cast<int>(tag_result), tag_failures);
			std::printf("[dlssg] slSetTagForFrame failed (%d), %u time(s)\n", static_cast<int>(tag_result), tag_failures);
		}
		// Once: what DLSS-G is given (its constants, the tags' extents and formats), to check against FSR 3's inputs.
		static bool dumped = false;
		if (!dumped) {
			dumped = true;
			const auto m = [](const Mat4& a) {
				char text[320];
				std::snprintf(text, sizeof(text), "[%.4g %.4g %.4g %.4g | %.4g %.4g %.4g %.4g | %.4g %.4g %.4g %.4g | %.4g %.4g %.4g %.4g]",
				              a[0].x, a[0].y, a[0].z, a[0].w, a[1].x, a[1].y, a[1].z, a[1].w, a[2].x, a[2].y, a[2].z, a[2].w,
				              a[3].x, a[3].y, a[3].z, a[3].w);
				return std::string(text);
			};
			char text[2048];
			std::snprintf(text, sizeof(text),
			              "[dlssg] constants: frame %u camera %s viewToClip %s clipToPrevClip %s mvecScale %.6g %.6g jitter %.4g %.4g "
			              "near %.4g far %.4g fov %.4g aspect %.4g depthInverted %d cameraMotionIncluded %d mv3D %d dilated %d "
			              "jittered %d ortho %d reset %d | depth %ux%u fmt %u, motion %ux%u fmt %u, tag extent %ux%u, hudless %s %ux%u, "
			              "backbuffer %ux%u, region %d,%d %ux%u, setTag %d\n",
			              g_frame_index, inputs->camera.valid ? "valid" : "INVALID", m(constants.cameraViewToClip).c_str(),
			              m(constants.clipToPrevClip).c_str(), constants.mvecScale.x, constants.mvecScale.y, constants.jitterOffset.x,
			              constants.jitterOffset.y, constants.cameraNear, constants.cameraFar, constants.cameraFOV,
			              constants.cameraAspectRatio, static_cast<int>(constants.depthInverted),
			              static_cast<int>(constants.cameraMotionIncluded), static_cast<int>(constants.motionVectors3D),
			              static_cast<int>(constants.motionVectorsDilated), static_cast<int>(constants.motionVectorsJittered),
			              static_cast<int>(constants.orthographicProjection), static_cast<int>(constants.reset), depth.width,
			              depth.height, depth.nativeFormat, motion.width, motion.height, motion.nativeFormat, full.width, full.height,
			              tags.size() > 2 && tags.back().type == sl::kBufferTypeHUDLessColor ? "tagged" : "not tagged",
			              hudless.width, hudless.height, backbuffer.width, backbuffer.height, region.offset.x, region.offset.y,
			              region.extent.width, region.extent.height, static_cast<int>(tag_result));
			LOGF("%s", text);
			std::printf("%s", text);
			std::fflush(stdout);
		}
	} else {
		// Menus, movies, loading: no valid depth and motion (DLSS-G then presents the frame as is).
		sl::ResourceTag tags[] {
		    sl::ResourceTag(nullptr, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent),
		    sl::ResourceTag(nullptr, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent),
		};
		g_api.set_tag_for_frame(token, viewport, tags, static_cast<uint32_t>(std::size(tags)), nullptr);
	}
	if (guest_index == 0) g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitEnd, token);
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
	if (const auto result = g_api.dlssg_get_state(sl::ViewportHandle {0}, state, nullptr); result != sl::Result::eOk) {
		if (presents < 300) std::printf("[dlssg] slDLSSGGetState failed (%d)\n", static_cast<int>(result));
		return;
	}
	{
		// DLSS-G's own verdict on its inputs (eFailResolutionTooLow, eFailCommonConstantsInvalid, ... as flags).
		static uint32_t last_status = ~0u;
		const auto      status      = static_cast<uint32_t>(state.status);
		if (status != last_status || presents < 120) {
			last_status = status;
			LOGF("frame generation: DLSS-G state: status 0x%x, presented %u, generate max %u\n", status,
			     state.numFramesActuallyPresented, state.numFramesToGenerateMax);
			std::printf("[dlssg] state: status 0x%x, presented %u, generate max %u\n", status, state.numFramesActuallyPresented,
			            state.numFramesToGenerateMax);
		}
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
	if (g_dlssg_dynamic && state.numFramesToGenerateMax != 0) {
		const int max = std::min(g_generate, static_cast<int>(state.numFramesToGenerateMax));
		if (max != g_dyn_max) {
			std::printf("[dyn] up to x%d (GPU makes up to %u generated)\n", max + 1, state.numFramesToGenerateMax);
			std::fflush(stdout);
			g_dyn_max = max;
			if (g_dyn_gen > max) {
				g_dyn_gen     = max;
				g_options_set = false;
				DynApplyReflexLimit(g_dyn_gen);
			}
		}
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

void SwapchainChanging() {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// Off before the swapchain is made again (DLSS-G guide: before a resize or a full screen change).
	if (!g_ready || g_mode_set != 1) return;
	sl::DLSSGOptions options {};
	options.mode                = sl::DLSSGMode::eOff;
	options.flags               = sl::DLSSGFlags::eRetainResourcesWhenOff;
	options.numFramesToGenerate = static_cast<uint32_t>(CurrentGenerate());
	if (const auto result = g_api.dlssg_set_options(sl::ViewportHandle {0}, options); result != sl::Result::eOk)
		LOGF("frame generation: slDLSSGSetOptions (off) failed (%d)\n", static_cast<int>(result));
	g_mode_set    = 0;
	g_options_set = false; // (the next present sends its sizes again)
#endif
}

static bool g_cut = false; // GPU thread: the current guest frame's upscale started a new history (NoteCut)

void NoteCut() {
	g_cut = true;
#ifdef KYTY_FRAMEGEN_STREAMLINE
	g_discontinuity.store(true, std::memory_order_relaxed); // (DLSS-G: constants.reset at the next present)
#endif
	static std::atomic<uint32_t> cuts {0};
	if (const auto count = cuts.fetch_add(1, std::memory_order_relaxed) + 1; count <= 4 || count % 200 == 0)
		LOGF("frame generation: camera cut %u (the game reset its upscaler): no frame generated across it\n", count);
}

void NoteDroppedFrame() {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	g_discontinuity.store(true, std::memory_order_relaxed);
#endif
	static std::atomic<uint32_t> dropped {0};
	if (const auto count = dropped.fetch_add(1, std::memory_order_relaxed) + 1; count <= 8 || count % 600 == 0)
		LOGF("frame generation: %u frame(s) with inputs left unshown (the next resets DLSS-G's history)\n", count);
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

// The camera's capture, per guest frame window (the "[dlssg] camera valid" line): G-buffer draws seen by path, those
// whose SRT was too short to hold the camera.
std::atomic<uint32_t> g_cam_seen[3] {}; // normal, table, native record
std::atomic<uint32_t> g_cam_short {};

std::mutex            g_last_vp_mutex; // (KYTY_CONTACT_SHADOWS: the last captured camera, any frame)
std::array<float, 16> g_last_vp {};
bool                  g_have_last_vp = false;

void OnDraw(uint64_t vs_hash, std::span<const uint32_t> srt, char path) {
	if (g_camera.valid || !IsGbufferVertexShader(vs_hash)) {
		return;
	}
	g_cam_seen[path == 't' ? 1 : path == 'r' ? 2 : 0].fetch_add(1, std::memory_order_relaxed);
	if (srt.size() < SrtJitter + 2) {
		g_cam_short.fetch_add(1, std::memory_order_relaxed);
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
	// (The KYTY_KBM mouse: the camera's yaw, from the forward row.)
	::Libs::Graphics::HostInputCameraForward(g_camera.view_proj[12], g_camera.view_proj[13], g_camera.view_proj[14]);
	if (ContactShadowsWanted() || RtLocal::Wanted()) { // (rt-local-lights.cpp reads it too)
		std::lock_guard lock(g_last_vp_mutex);
		g_last_vp = g_camera.view_proj, g_have_last_vp = true;
	}
}

// KYTY_SHADER_CENSUS=1: every program the game runs (vertex, pixel, compute) with the first and last guest flip it ran
// in and how many draws or dispatches, printed every 10 s ("[census]" lines; the full list each time): the PlayStation
// Studios logo's programs are those it alone uses (10-18: to hold it at 60 without frame generation).
void NoteShader(uint64_t hash, char stage) {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_SHADER_CENSUS");
		return value != nullptr && *value == '1';
	}();
	if (!on || hash == 0) return;
	struct Entry {
		uint64_t first = 0, last = 0, count = 0;
		char     stage = '?';
	};
	static std::mutex                             mutex;
	static std::unordered_map<uint64_t, Entry>    seen;
	static auto                                   printed = std::chrono::steady_clock::now();
	const uint64_t                                flip    = g_flips_taken.load(std::memory_order_relaxed);
	std::lock_guard                               lock(mutex);
	auto& e = seen[hash];
	if (e.count == 0) {
		e.first = flip;
		e.stage = stage;
	}
	e.last = flip;
	e.count++;
	const auto now = std::chrono::steady_clock::now();
	if (now - printed < std::chrono::seconds(10)) return;
	printed = now;
	std::printf("[census] flip %llu: %zu programs\n", static_cast<unsigned long long>(flip), seen.size());
	for (const auto& [h, v]: seen)
		std::printf("[census] %c %016llx first %llu last %llu count %llu\n", v.stage, static_cast<unsigned long long>(h),
		            static_cast<unsigned long long>(v.first), static_cast<unsigned long long>(v.last),
		            static_cast<unsigned long long>(v.count));
	std::fflush(stdout);
}


void ReflexGuestFlip(FrameInputs& fg) {
	fg.reflex_index = 0;
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!g_ready || !ReflexGuestMarkers() || g_api.pcl_set_marker == nullptr) return;
	sl::FrameToken* token = nullptr;
	if (g_rx_open != 0) {
		uint32_t index = g_rx_open;
		if (g_api.get_new_frame_token(token, &index) == sl::Result::eOk && token != nullptr) {
			g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitEnd, *token);
			fg.reflex_index = g_rx_open;
		}
	}
	uint32_t next = g_rx_counter.fetch_add(1, std::memory_order_relaxed) + 1;
	token         = nullptr;
	if (g_api.get_new_frame_token(token, &next) != sl::Result::eOk || token == nullptr) {
		g_rx_open = 0;
		return;
	}
	static const bool sleep = [] {
		const char* value = std::getenv("KYTY_DLSSG_SLEEP");
		return value == nullptr || *value != '0';
	}();
	if (sleep && g_api.reflex_sleep != nullptr) g_api.reflex_sleep(*token);
	g_api.pcl_set_marker(sl::PCLMarker::eSimulationStart, *token);
	g_api.pcl_set_marker(sl::PCLMarker::eSimulationEnd, *token);
	g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitStart, *token);
	g_rx_open = next;
#endif
}

void SetPresentReflexIndex(uint32_t index) {
	g_rx_present.store(index, std::memory_order_relaxed);
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

void OnUpscale(ImageId depth, uint64_t depth_address, ImageId motion, uint64_t motion_address) {
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
	if (!Enabled() && !Fsr3::Active() && !KbmWarp::HudlessWanted()) return; // (capturing for upscaling only: no HUD-less image wanted)
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


// KYTY_UPSCALE_SHARPNESS=<0..1> after DLSS or XeSS (sharpen.comp): g_up_output into g_sharp, which the composite reads.
struct SharpenPass {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             sampler    = nullptr;
	VulkanImage             image;
	vk::ImageView           view   = nullptr;
	bool                    failed = false;
	// The PS5 Pro path when the output's format has no storage support: sharpen.comp into this rgba16f image, then
	// blitted back into the output.
	VulkanImage             out16;
	vk::ImageView           out16_view = nullptr;
} g_sharp;

bool StorageFormat(GraphicContext& graphics, vk::Format format) {
	return static_cast<bool>(graphics.physical_device.getFormatProperties(format).optimalTilingFeatures &
	                         vk::FormatFeatureFlagBits::eStorageImage);
}

bool EnsureSharpenOut16(GraphicContext& graphics, vk::Extent3D extent) {
	const auto device = graphics.device;
	if (g_sharp.out16.image != nullptr && g_sharp.out16.extent.width == extent.width && g_sharp.out16.extent.height == extent.height)
		return true;
	if (g_sharp.out16_view != nullptr) {
		device.waitIdle();
		device.destroyImageView(g_sharp.out16_view, nullptr);
		g_sharp.out16_view = nullptr;
	}
	if (g_sharp.out16.image != nullptr) graphics.DeleteImage(g_sharp.out16);
	g_sharp.out16.image = nullptr;
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = vk::Format::eR16G16B16A16Sfloat;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, g_sharp.out16)) {
		g_sharp.out16.image = nullptr;
		return false;
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = g_sharp.out16.image;
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = vk::Format::eR16G16B16A16Sfloat;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (device.createImageView(&view_info, nullptr, &g_sharp.out16_view) != vk::Result::eSuccess) return false;
	g_sharp.out16.state = {};
	return true;
}

bool EnsureSharpen(GraphicContext& graphics, const VulkanImage& like) {
	const auto device = graphics.device;
	if (g_sharp.pipeline == nullptr) {
		if (g_sharp.failed) return false;
		g_sharp.failed = true;
		std::array<vk::DescriptorSetLayoutBinding, 2> bindings {};
		for (uint32_t i = 0; i < 2; ++i) {
			bindings[i].binding         = i;
			bindings[i].descriptorType  = i == 0 ? vk::DescriptorType::eCombinedImageSampler : vk::DescriptorType::eStorageImage;
			bindings[i].descriptorCount = 1;
			bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
		}
		vk::DescriptorSetLayoutCreateInfo set_info {};
		set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		set_info.bindingCount = 2;
		set_info.pBindings    = bindings.data();
		if (device.createDescriptorSetLayout(&set_info, nullptr, &g_sharp.set_layout) != vk::Result::eSuccess) return false;
		vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, 4};
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.setLayoutCount         = 1;
		layout_info.pSetLayouts            = &g_sharp.set_layout;
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges    = &push;
		if (device.createPipelineLayout(&layout_info, nullptr, &g_sharp.layout) != vk::Result::eSuccess) return false;
		vk::ShaderModuleCreateInfo module_info {};
		module_info.codeSize    = sizeof(SHARPEN_SPV);
		module_info.pCode       = SHARPEN_SPV;
		vk::ShaderModule module = nullptr;
		if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
		pipeline_info.stage.module = module;
		pipeline_info.stage.pName  = "main";
		pipeline_info.layout       = g_sharp.layout;
		const auto created = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_sharp.pipeline);
		device.destroyShaderModule(module, nullptr);
		if (created != vk::Result::eSuccess) {
			g_sharp.pipeline = nullptr;
			return false;
		}
		vk::SamplerCreateInfo sampler {};
		sampler.magFilter    = vk::Filter::eNearest;
		sampler.minFilter    = vk::Filter::eNearest;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		if (device.createSampler(&sampler, nullptr, &g_sharp.sampler) != vk::Result::eSuccess) return false;
		g_sharp.failed = false;
	}
	if (g_sharp.image.image != nullptr && g_sharp.image.format == like.format &&
	    g_sharp.image.extent.width == like.extent.width && g_sharp.image.extent.height == like.extent.height)
		return true;
	if (g_sharp.view != nullptr) {
		device.waitIdle();
		device.destroyImageView(g_sharp.view, nullptr);
		g_sharp.view = nullptr;
	}
	if (g_sharp.image.image != nullptr) graphics.DeleteImage(g_sharp.image);
	g_sharp.image.image = nullptr;
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {like.extent.width, like.extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = like.format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	// (Storage only where the format has it: B10G11R11 may not.)
	if (StorageFormat(graphics, like.format)) create.usage |= vk::ImageUsageFlagBits::eStorage;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, g_sharp.image)) {
		g_sharp.image.image = nullptr;
		return false;
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = g_sharp.image.image;
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = like.format;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (device.createImageView(&view_info, nullptr, &g_sharp.view) != vk::Result::eSuccess) return false;
	g_sharp.image.state = {};
	return true;
}

float SharpenAmount() {
	static const float amount = [] {
		const char* value = std::getenv("KYTY_UPSCALE_SHARPNESS");
		return value != nullptr && *value != '\0' ? std::clamp(static_cast<float>(std::atof(value)), 0.0f, 1.0f) : 0.0f;
	}();
	return amount;
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
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, 24};
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
		create.size  = 1u << 20; // (the header and a ratio per 8x8 block, up to 3840x2160)
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
	vk::DescriptorBufferInfo stats {g_bypass.stats, 0, VK_WHOLE_SIZE};
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
	static const float smoothing = std::clamp(EnvFloatOr("KYTY_TAA_MATCH_SMOOTH", 0.25f), 0.001f, 1.0f);
	const uint32_t groups_x = (render.width + 7) / 8, groups_y = (render.height + 7) / 8;
	struct {
		float   scale;
		int32_t mode;
		float   smoothing;
		int32_t match;
		int32_t blocks_x;
		int32_t blocks_y;
	} push {scale, 0, smoothing, match ? 1 : 0, static_cast<int32_t>(groups_x), static_cast<int32_t>(groups_y)};
	// (The block ratios start at 0, "none yet", and again after a size change.)
	static uint32_t cleared_for = 0;
	if (cleared_for != groups_x * 65536u + groups_y) {
		cleared_for = groups_x * 65536u + groups_y;
		vk_command.fillBuffer(g_bypass.stats, 0, VK_WHOLE_SIZE, 0u);
		vk::MemoryBarrier2 filled {};
		filled.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		filled.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		filled.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
		filled.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
		vk::DependencyInfo filled_dep {};
		filled_dep.memoryBarrierCount = 1;
		filled_dep.pMemoryBarriers    = &filled;
		vk_command.pipelineBarrier2(filled_dep);
	}
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
			gain     = g_bypass.readback_mapped[2]; // (the mean of the block ratios)
			measured = gain;
		}
		if (first)
			std::printf("TAA: ran, then its color replaced by %s %ux%u (its alpha kept, brightness matched to the TAA's: %s)\n",
			            bypass ? "the raw frame for the upscaler" : "DLAA's", render.width, render.height,
			            match ? "on" : "off");
		(void)measured;
		std::printf("TAA: brightness per 8x8 block, mean gain %.3f (fixed factor %.3f)\n", static_cast<double>(gain),
		            static_cast<double>(scale));
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
	// Sharpening after DLSS or XeSS (FSR sharpens itself: KYTY_UPSCALE_SHARPNESS goes to its RCAS).
	bool sharpened = false;
	if (done && (dlss || xess) && SharpenAmount() > 0.0f && EnsureSharpen(*g_graphics_ctx, g_up_output)) {
		Transition(vk_command, g_sharp.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
		           vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader);
		vk::DescriptorImageInfo infos[2] {};
		infos[0] = {g_sharp.sampler, g_up_output_view, vk::ImageLayout::eShaderReadOnlyOptimal};
		infos[1] = {nullptr, g_sharp.view, vk::ImageLayout::eGeneral};
		vk::WriteDescriptorSet writes[2] {};
		for (uint32_t i = 0; i < 2; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = i == 0 ? vk::DescriptorType::eCombinedImageSampler : vk::DescriptorType::eStorageImage;
			writes[i].pImageInfo      = &infos[i];
		}
		vk_command.bindPipeline(vk::PipelineBindPoint::eCompute, g_sharp.pipeline);
		vk_command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_sharp.layout, 0, 2, writes);
		const float amount = SharpenAmount();
		vk_command.pushConstants(g_sharp.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(amount), &amount);
		vk_command.dispatch((g_up_output.extent.width + 7) / 8, (g_up_output.extent.height + 7) / 8, 1);
		Transition(vk_command, g_sharp.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
		           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
		command.InvalidateGraphicsState();
		sharpened = true;
	}
	if (in_pass) command.ResumeRendering(rendering);
	if (!done) return;
	scene->image_view = sharpened ? g_sharp.view : g_up_output_view;
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
	inputs.cut         = std::exchange(g_cut, false);
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
	auto camera        = g_camera;
	g_sources.valid    = false;
	g_camera.valid     = false;
	++g_guest_frame;
	{
		// A guest frame whose G-buffer draws did not give the camera: the last captured one, for up to
		// KYTY_DLSSG_CAMERA_HOLD (2) frames, its previous matrix the same (no camera motion assumed: the vectors carry it),
		// in place of identity matrices and a reset (DLSS-G then interpolated as from the image alone).
		static const uint32_t hold_frames = [] {
			const char* value = std::getenv("KYTY_DLSSG_CAMERA_HOLD");
			return value != nullptr ? static_cast<uint32_t>(std::atoi(value)) : 2u;
		}();
		static Camera   last {};
		static uint32_t since = ~0u;
		static uint32_t window = 0, valid = 0, held = 0;
		if (camera.valid) {
			last  = camera;
			since = 0;
			++valid;
		} else if (last.valid && since < hold_frames) {
			++since;
			camera                = last;
			camera.prev_view_proj = last.view_proj;
			++held;
		} else if (since != ~0u) {
			since = hold_frames + 1;
		}
		if (++window == 600) {
			const auto seen = [](uint32_t i) { return g_cam_seen[i].exchange(0, std::memory_order_relaxed); };
			const uint32_t normal = seen(0), table = seen(1), record = seen(2);
			const uint32_t shorter = g_cam_short.exchange(0, std::memory_order_relaxed);
			LOGF("[dlssg] camera valid %u of 600 (held %u; G-buffer draws before capture: normal %u, table %u, record %u, SRT "
			     "too short %u)\n", valid, held, normal, table, record, shorter);
			std::printf("[dlssg] camera valid %u of 600 (held %u; G-buffer draws before capture: normal %u, table %u, record "
			            "%u, SRT too short %u)\n", valid, held, normal, table, record, shorter);
			std::fflush(stdout);
			window = valid = held = 0;
		}
	}
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

bool UpscalerReady() {
	if (g_sr_kind == 2) return Xess::Ready();
	if (g_sr_kind == 3) return Fsr3::SrReady();
	return g_upscale_ready;
}

namespace {
// The PS5 Pro path's images as our upscalers take them.
void AsVulkanImage(const UpscaleImage& in, VulkanImage& out) {
	out.format = in.format;
	out.extent = vk::Extent3D {in.extent.width, in.extent.height, 1};
	out.usage  = in.usage;
	out.image  = in.image;
}
bool UpscaleOurs(vk::CommandBuffer command, const UpscaleInputs& inputs) {
	VulkanImage color, depth, motion, output, reactive;
	AsVulkanImage(inputs.color, color);
	AsVulkanImage(inputs.depth, depth);
	AsVulkanImage(inputs.motion, motion);
	AsVulkanImage(inputs.output, output);
	const bool have_reactive = inputs.reactive.view != nullptr && inputs.reactive.image != nullptr; // (rtparticles)
	if (have_reactive) AsVulkanImage(inputs.reactive, reactive);
	const float fov = inputs.view_to_clip[5] != 0.0f ? 2.0f * std::atan(1.0f / std::fabs(inputs.view_to_clip[5])) : 1.0f;
	bool done = false;
	if (g_sr_kind == 2) {
		Xess::UpscaleDesc x;
		x.command     = static_cast<VkCommandBuffer>(command);
		x.color       = &color;
		x.color_view  = inputs.color.view;
		x.depth       = &depth;
		x.depth_view  = inputs.depth.view;
		x.motion      = &motion;
		x.motion_view = inputs.motion.view;
		x.output      = &output;
		x.output_view = inputs.output.view;
		x.render      = inputs.color.extent;
		x.upscale     = inputs.output.extent;
		x.reset       = inputs.reset;
		x.jitter[0]   = inputs.jitter_x;
		x.jitter[1]   = inputs.jitter_y;
		x.mv_scale[0] = inputs.motion_scale_x;
		x.mv_scale[1] = inputs.motion_scale_y;
		if (have_reactive) x.mask = &reactive, x.mask_view = inputs.reactive.view;
		done          = Xess::Upscale(x);
	} else if (g_graphics_ctx != nullptr) {
		static auto last = std::chrono::steady_clock::now();
		const auto  now  = std::chrono::steady_clock::now();
		const float delta = std::chrono::duration<float, std::milli>(now - last).count();
		last              = now;
		Fsr3::UpscaleDesc desc;
		desc.command     = static_cast<VkCommandBuffer>(command);
		desc.color       = &color;
		desc.depth       = &depth;
		desc.motion      = &motion;
		desc.output      = &output;
		desc.render      = inputs.color.extent;
		desc.upscale     = inputs.output.extent;
		desc.reset       = inputs.reset;
		desc.delta_ms    = delta;
		desc.jitter[0]   = inputs.jitter_x;
		desc.jitter[1]   = inputs.jitter_y;
		desc.mv_scale[0] = inputs.motion_scale_x;
		desc.mv_scale[1] = inputs.motion_scale_y;
		desc.fov         = fov;
		desc.z_near      = inputs.near_plane;
		if (have_reactive) desc.reactive = &reactive;
		done             = Fsr3::Upscale(*g_graphics_ctx, desc);
	}
	static bool said = false;
	if (!said) {
		said = true;
		std::printf("Super resolution (PS5 Pro path): %s %ux%u -> %ux%u%s\n", g_sr_kind == 2 ? "XeSS" : "FSR 3.1",
		            inputs.color.extent.width, inputs.color.extent.height, inputs.output.extent.width,
		            inputs.output.extent.height, done ? "" : " (failed: stretched)");
		std::fflush(stdout);
	}
	return done;
}
} // namespace

static bool UpscaleStreamline(vk::CommandBuffer command, const UpscaleInputs& inputs) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!g_upscale_ready) return false;
	// Its own viewport: DLSS-G's (0) takes its constants and tags at the present.
	const sl::ViewportHandle viewport {1};
	// DLSS given up (the game's image stretched instead, renderSuperResolution.cpp): an input outside the sizes the
	// mode takes (the NVIDIA app's DLSS override of the program can ask for another scale: 3802x2138-3840x2160 for a
	// 1920x1080 input, issue #87), or NGX refusing its evaluations.
	static bool given_up = false;
	if (given_up) return false;
	if (g_dlss_refusals.load(std::memory_order_relaxed) >= 3) {
		given_up = true;
		std::printf("DLSS super resolution: NGX refused it %u times (Streamline errors above): the game's image is "
		            "upscaled without it from here on\n",
		            g_dlss_refusals.load(std::memory_order_relaxed));
		std::fflush(stdout);
		return false;
	}
	static vk::Extent2D options_render {}, options_output {};
	if (options_render != inputs.color.extent || options_output != inputs.output.extent) {
		// The mode by the input's share of the output's height (DLSS takes any input size; the mode picks its model and
		// limits): PS5 Pro's 1512 of 2160 and 1440 are quality, 1080 performance, the output's own DLAA; below half
		// the output (performance's least, 1080 of 2160) ultra performance, down to its third (720).
		const double share = static_cast<double>(inputs.color.extent.height) / inputs.output.extent.height;
		sl::DLSSOptions options {};
		options.mode = share >= 0.95   ? sl::DLSSMode::eDLAA
		               : share >= 0.62 ? sl::DLSSMode::eMaxQuality
		               : share >= 0.55 ? sl::DLSSMode::eBalanced
		               : share >= 0.495 ? sl::DLSSMode::eMaxPerformance // (10-21: 1079 of 2160 from a 1.334 ratio fell
		                                                                // to ultra performance, which takes 720 only: black)
		                               : sl::DLSSMode::eUltraPerformance;
		options.outputWidth     = inputs.output.extent.width;
		options.outputHeight    = inputs.output.extent.height;
		options.colorBuffersHDR = sl::Boolean::eTrue; // (the game's linear HDR scene, before its tone mapping)
		options.useAutoExposure = sl::Boolean::eTrue;
		// KYTY_DLSS_PRESET=<letter> (a test): that DLSS model for every mode instead of the mode's default.
		if (const char* preset = std::getenv("KYTY_DLSS_PRESET"); preset != nullptr && *preset >= 'A' && *preset <= 'O') {
			const auto value = static_cast<sl::DLSSPreset>(*preset - 'A' + 1);
			options.dlaaPreset = options.qualityPreset = options.balancedPreset = options.performancePreset =
			    options.ultraPerformancePreset = options.ultraQualityPreset = value;
		}
		if (const auto result = g_api.dlss_set_options(viewport, options); result != sl::Result::eOk) {
			std::printf("DLSS super resolution: slDLSSSetOptions failed (%s)\n", sl::getResultAsStr(result));
			std::fflush(stdout);
			return false;
		}
		sl::DLSSOptimalSettings settings {};
		(void)g_api.dlss_get_optimal_settings(options, settings);
		std::printf("DLSS super resolution: %ux%u -> %ux%u, mode %u (its range %ux%u-%ux%u)\n", inputs.color.extent.width,
		            inputs.color.extent.height, inputs.output.extent.width, inputs.output.extent.height,
		            static_cast<uint32_t>(options.mode), settings.renderWidthMin, settings.renderHeightMin,
		            settings.renderWidthMax, settings.renderHeightMax);
		std::fflush(stdout);
		if (settings.renderWidthMax != 0 && settings.renderHeightMax != 0 &&
		    (inputs.color.extent.width < settings.renderWidthMin || inputs.color.extent.width > settings.renderWidthMax ||
		     inputs.color.extent.height < settings.renderHeightMin || inputs.color.extent.height > settings.renderHeightMax)) {
			given_up = true;
			std::printf("DLSS super resolution: the game renders %ux%u, outside what DLSS takes in this mode: the game's "
			            "image is upscaled without it. A DLSS override for this program in the NVIDIA app (Super Resolution "
			            "mode or scaling ratio) does this; set it back to the default for kyty_emulator.exe\n",
			            inputs.color.extent.width, inputs.color.extent.height);
			std::fflush(stdout);
			return false;
		}
		options_render = inputs.color.extent;
		options_output = inputs.output.extent;
	}
	static uint32_t frame = 0;
	++frame;
	sl::FrameToken* token = nullptr;
	if (g_api.get_new_frame_token(token, &frame) != sl::Result::eOk || token == nullptr) return false;

	sl::Constants constants {};
	Mat4          identity;
	for (uint32_t i = 0; i < 4; ++i) {
		identity[i] = sl::float4(i == 0, i == 1, i == 2, i == 3);
	}
	Mat4 view_to_clip;
	for (uint32_t i = 0; i < 4; ++i) {
		view_to_clip[i] = sl::float4(inputs.view_to_clip[i * 4], inputs.view_to_clip[i * 4 + 1], inputs.view_to_clip[i * 4 + 2],
		                             inputs.view_to_clip[i * 4 + 3]);
	}
	constants.cameraViewToClip = view_to_clip;
	if (!Inverted(view_to_clip, constants.clipToCameraView)) constants.clipToCameraView = identity;
	constants.clipToLensClip      = identity;
	constants.clipToPrevClip      = identity;
	constants.prevClipToClip      = identity;
	constants.jitterOffset        = sl::float2(inputs.jitter_x, inputs.jitter_y);
	// (Streamline's motion vectors are in render-size UV units: its scale makes them so from the game's pixels.)
	constants.mvecScale = sl::float2(inputs.motion_scale_x / static_cast<float>(inputs.color.extent.width),
	                                 inputs.motion_scale_y / static_cast<float>(inputs.color.extent.height));
	constants.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
	constants.cameraPos           = sl::float3(0, 0, 0);
	constants.cameraUp            = sl::float3(0, 1, 0);
	constants.cameraRight         = sl::float3(1, 0, 0);
	constants.cameraFwd           = sl::float3(0, 0, 1);
	constants.cameraNear          = inputs.near_plane;
	constants.cameraFar           = inputs.far_plane;
	constants.cameraFOV           = inputs.view_to_clip[5] != 0.0f ? 2.0f * std::atan(1.0f / std::fabs(inputs.view_to_clip[5])) : 1.0f;
	constants.cameraAspectRatio   = static_cast<float>(inputs.color.extent.width) / static_cast<float>(inputs.color.extent.height);
	constants.motionVectorsInvalidValue = sl::INVALID_FLOAT;
	constants.depthInverted             = sl::Boolean::eTrue;
	constants.cameraMotionIncluded      = sl::Boolean::eTrue;
	constants.motionVectors3D           = sl::Boolean::eFalse;
	constants.reset                     = inputs.reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
	constants.orthographicProjection    = sl::Boolean::eFalse;
	constants.motionVectorsDilated      = sl::Boolean::eFalse;
	constants.motionVectorsJittered     = sl::Boolean::eFalse;
	if (const auto result = g_api.set_constants(constants, *token, viewport); result != sl::Result::eOk) {
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 8)
			std::printf("DLSS super resolution: slSetConstants failed (%s)\n", sl::getResultAsStr(result));
		return false;
	}
	const auto resource = [](const UpscaleImage& image) {
		sl::Resource r(sl::ResourceType::eTex2d, static_cast<VkImage>(image.image), nullptr, static_cast<VkImageView>(image.view),
		               static_cast<uint32_t>(static_cast<VkImageLayout>(image.layout)));
		r.width        = image.extent.width;
		r.height       = image.extent.height;
		r.nativeFormat = static_cast<uint32_t>(static_cast<VkFormat>(image.format));
		r.mipLevels    = 1;
		r.arrayLayers  = 1;
		r.flags        = 0;
		r.usage        = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image.usage));
		return r;
	};
	auto             color  = resource(inputs.color);
	auto             depth  = resource(inputs.depth);
	auto             motion = resource(inputs.motion);
	auto             output = resource(inputs.output);
	// (KYTY_SR_LOG: the first two evaluations' images and constants.)
	static const bool describe = std::getenv("KYTY_SR_LOG") != nullptr;
	if (static std::atomic<uint32_t> described {0}; describe && described.fetch_add(1, std::memory_order_relaxed) < 2) {
		const std::pair<const char*, const UpscaleImage*> images[] {
		    {"color", &inputs.color}, {"depth", &inputs.depth}, {"motion", &inputs.motion}, {"output", &inputs.output}};
		for (const auto& [name, image]: images) {
			std::printf("DLSS super resolution: %s image %p view %p layout %d format %d %ux%u usage 0x%x\n", name,
			            static_cast<void*>(static_cast<VkImage>(image->image)),
			            static_cast<void*>(static_cast<VkImageView>(image->view)), static_cast<int>(image->layout),
			            static_cast<int>(image->format), image->extent.width, image->extent.height,
			            static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image->usage)));
		}
		std::printf("DLSS super resolution: jitter %g %g, motion scale %g %g, reset %d, near %g far %g\n", inputs.jitter_x,
		            inputs.jitter_y, inputs.motion_scale_x, inputs.motion_scale_y, inputs.reset ? 1 : 0, inputs.near_plane,
		            inputs.far_plane);
		std::fflush(stdout);
	}
	const sl::Extent render {0, 0, inputs.color.extent.width, inputs.color.extent.height};
	const sl::Extent full {0, 0, inputs.output.extent.width, inputs.output.extent.height};
	// (rtparticles) KYTY_RT_PARTICLES=2: the particle mask as DLSS's bias-current-color hint (trust the current frame
	// where particles, drawn without motion vectors, are).
	const bool have_reactive = inputs.reactive.view != nullptr && inputs.reactive.image != nullptr;
	auto       reactive      = resource(inputs.reactive); // (null handles when there is none: the tag is not given)
	sl::ResourceTag tags[5] {
	    sl::ResourceTag(&color, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &render),
	    sl::ResourceTag(&output, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &full),
	    sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilEvaluate, &render),
	    sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilEvaluate, &render),
	    sl::ResourceTag(&reactive, sl::kBufferTypeBiasCurrentColorHint, sl::ResourceLifecycle::eValidUntilEvaluate, &render),
	};
	const uint32_t tag_count = have_reactive ? 5u : 4u;
	auto* command_buffer = reinterpret_cast<sl::CommandBuffer*>(static_cast<VkCommandBuffer>(command));
	if (const auto result = g_api.set_tag_for_frame(*token, viewport, tags, tag_count, command_buffer);
	    result != sl::Result::eOk) {
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 8)
			std::printf("DLSS super resolution: slSetTagForFrame failed (%s)\n", sl::getResultAsStr(result));
		return false;
	}
	const sl::BaseStructure* evaluate_inputs[] {&viewport};
	// (Result::eWarnOutOfVRAM is not a failure: Streamline returns it in place of eOk, DLSS recorded, while the process
	// uses all the video memory Windows budgets it - issue #85: an RTX 2060 at 5030 of 5187 MiB had DLSS's image
	// stretched over from then on.)
	if (const auto result = g_api.evaluate(sl::kFeatureDLSS, *token, evaluate_inputs, 1, command_buffer);
	    result == sl::Result::eWarnOutOfVRAM) {
		static std::atomic<bool> reported {false};
		if (!reported.exchange(true, std::memory_order_relaxed))
			std::printf("DLSS super resolution: the video memory budget is used up (Streamline's warning); DLSS still runs\n");
	} else if (result != sl::Result::eOk) {
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 8)
			std::printf("DLSS super resolution: slEvaluateFeature failed (%s)\n", sl::getResultAsStr(result));
		return false;
	}
	return true;
#else
	(void)command;
	(void)inputs;
	return false;
#endif
}

namespace {
// KYTY_UPSCALE_SHARPNESS on the PS5 Pro path after DLSS or XeSS (FSR's RCAS takes it itself): the output (general)
// copied into g_sharp, then sharpen.comp from it back into the output, which the game's composite reads.
void SharpenOutput(vk::CommandBuffer command, const UpscaleInputs& inputs, float amount) {
	const auto& out = inputs.output;
	// Any float color format (sharpen.comp writes without a format qualifier): in place where the output takes storage
	// writes, else into an rgba16f image blitted back (needs transfer dst).
	const bool float_format = out.format == vk::Format::eR16G16B16A16Sfloat || out.format == vk::Format::eB10G11R11UfloatPack32 ||
	                          out.format == vk::Format::eR32G32B32A32Sfloat || out.format == vk::Format::eA2B10G10R10UnormPack32 ||
	                          out.format == vk::Format::eR8G8B8A8Unorm || out.format == vk::Format::eB8G8R8A8Unorm;
	const bool direct = g_graphics_ctx != nullptr && (out.usage & vk::ImageUsageFlagBits::eStorage) &&
	                    StorageFormat(*g_graphics_ctx, out.format);
	const bool via16 = !direct && (out.usage & vk::ImageUsageFlagBits::eTransferDst);
	if (g_graphics_ctx == nullptr || out.view == nullptr || !float_format || !(out.usage & vk::ImageUsageFlagBits::eTransferSrc) ||
	    (!direct && !via16)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("Super resolution sharpening: skipped (output format %d usage 0x%x)\n", static_cast<int>(out.format),
			            static_cast<uint32_t>(static_cast<VkImageUsageFlags>(out.usage)));
			std::fflush(stdout);
		}
		return;
	}
	VulkanImage like;
	like.format = out.format;
	like.extent = vk::Extent3D {out.extent.width, out.extent.height, 1};
	if (!EnsureSharpen(*g_graphics_ctx, like)) return;
	if (via16 && !EnsureSharpenOut16(*g_graphics_ctx, like.extent)) return;
	{
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("Super resolution sharpening: output format %d %ux%u, %s\n", static_cast<int>(out.format), out.extent.width,
			            out.extent.height, direct ? "in place" : "via rgba16f + blit");
			std::fflush(stdout);
		}
	}
	const auto global = [&](vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src, vk::PipelineStageFlags2 dst_stage,
	                        vk::AccessFlags2 dst) {
		vk::MemoryBarrier2 barrier {};
		barrier.srcStageMask  = src_stage;
		barrier.srcAccessMask = src;
		barrier.dstStageMask  = dst_stage;
		barrier.dstAccessMask = dst;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &barrier;
		command.pipelineBarrier2(dependency);
	};
	// The upscaler's writes before the copy.
	global(vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite, vk::PipelineStageFlagBits2::eCopy,
	       vk::AccessFlagBits2::eTransferRead);
	Transition(command, g_sharp.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
	           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eCopy);
	vk::ImageCopy copy {};
	copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, out.level, out.layer, 1};
	copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.extent         = vk::Extent3D {out.extent.width, out.extent.height, 1};
	command.copyImage(out.image, vk::ImageLayout::eGeneral, g_sharp.image.image, vk::ImageLayout::eTransferDstOptimal, 1, &copy);
	Transition(command, g_sharp.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
	           vk::AccessFlagBits2::eShaderSampledRead, vk::PipelineStageFlagBits2::eComputeShader);
	// The copy's reads of the output before the pass writes it.
	global(vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eComputeShader,
	       vk::AccessFlagBits2::eShaderStorageWrite);
	vk::DescriptorImageInfo infos[2] {};
	infos[0] = {g_sharp.sampler, g_sharp.view, vk::ImageLayout::eShaderReadOnlyOptimal};
	if (via16)
		Transition(command, g_sharp.out16, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
		           vk::AccessFlagBits2::eShaderStorageWrite, vk::PipelineStageFlagBits2::eComputeShader);
	infos[1] = {nullptr, via16 ? g_sharp.out16_view : out.view, vk::ImageLayout::eGeneral};
	vk::WriteDescriptorSet writes[2] {};
	for (uint32_t i = 0; i < 2; ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = i == 0 ? vk::DescriptorType::eCombinedImageSampler : vk::DescriptorType::eStorageImage;
		writes[i].pImageInfo      = &infos[i];
	}
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_sharp.pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_sharp.layout, 0, 2, writes);
	command.pushConstants(g_sharp.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(amount), &amount);
	command.dispatch((out.extent.width + 7) / 8, (out.extent.height + 7) / 8, 1);
	if (via16) {
		Transition(command, g_sharp.out16, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
		           vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eBlit);
		global(vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eCopy,
		       vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eBlit,
		       vk::AccessFlagBits2::eTransferWrite);
		vk::ImageBlit blit {};
		blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, out.level, out.layer, 1};
		blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(out.extent.width), static_cast<int32_t>(out.extent.height), 1};
		blit.dstOffsets[1]  = blit.srcOffsets[1];
		command.blitImage(g_sharp.out16.image, vk::ImageLayout::eTransferSrcOptimal, out.image, vk::ImageLayout::eGeneral, 1, &blit,
		                  vk::Filter::eNearest);
		global(vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eAllCommands,
		       vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
		return;
	}
	// The pass's writes before whatever reads or writes the output next (the exposure pass, the composite).
	global(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	       vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
}
} // namespace

// ---- (rtparticles) KYTY_RT_PARTICLES: the particle pass's own emission for the RT "fire light", a reactive mask ----
// How Demon's Souls draws particles (tools/local/static-precompile/files-PPSA01341_01.007.000.files, the engine's
// coredata/enginesupport/shaders/_ps5/particle-*): 603 cooked particle compute shaders (one per emitter) simulate on
// the GPU; particle-sort-prep / particle-sort (bitonic) order them; one engine pass, particle-translucent (main_vs +
// 3 main_ps variants; ultralow / underwater / displacement / distortion / motionblur / fog / ui variants), draws them
// blended onto the HDR scene color with the depth test on and depth writes off (they leave no depth, so the depth
// buffer at a flame is the wall behind it); particle-lightmap-update lights them from the game's own lights;
// particle-apply-distortion is a compute pass after them. The TAA / PS5 Pro super-resolution input (slot 1) is that
// color after the pass: the SSGI emissive light already sees flames as bright pixels, but tells them from lit
// surfaces only by lit-colour / G-buffer-albedo ratio and takes the whole pixel (flame + wall) as the radiance.
//
// Level 1 (KYTY_RT_PARTICLES=1): right before the frame's first blended, depth-write-off draw into the scene color
// (the particle pass's first draw; KYTY_RT_PARTICLES_VS=<hex,...> pins the vertex shaders instead), the color is
// copied (one render-size RGBA16F copy, ~0.05-0.1 ms on an RTX 5090 at 1440p, and a render-pass break). The SSGI
// emissive light (ssgi.comp, binding 15) then takes color - copy, the particles' own emission, as the emitter
// radiance: exact for additive particles (embers, sparks, flames, spell glow), alpha x (particle - background) for
// alpha-blended ones; flames over bright lit surfaces are found too, lit stone is never a particle. Positions still
// come from the depth buffer (the surface behind the flame): a particle depth would need the pass re-drawn with
// depth writes, not done here.
// Level 2 (KYTY_RT_PARTICLES=2): also an R8 reactive mask of that emission share (ssgi.comp mode 6, needs KYTY_SSGI
// >= 1) for the upscaler: FSR 3.1's reactive + transparency-and-composition, XeSS's responsive pixels, DLSS's
// bias-current-color hint - less ghosting / smearing of particles in the upscaler (frame generation takes no mask).
// KYTY_RT_PARTICLES_REACTIVE (0..1, 0.8): the mask's strength.
// KYTY_RT_PARTICLES_DEBUG=1: every 10 s the ordered passes of the last flip onto the scene color (draws by vertex /
// pixel shader, blend, depth state, count; dispatches writing it; table / native-path draws), the copy's state, the
// game's own reactive-mask slot (renderSuperResolution.cpp).
namespace {
struct RtPartCensusKey {
	uint64_t vs = 0, ps = 0, address = 0;
	uint32_t width = 0, height = 0;
	int32_t  format = 0;
	uint8_t  blend = 0, src = 0, dst = 0, comb = 0, zt = 0, zw = 0, indirect = 0, path = 0;
	bool operator==(const RtPartCensusKey&) const = default;
};
uint64_t RtPartMix(uint64_t h, uint64_t v) {
	h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
	return h * 0xff51afd7ed558ccdull;
}
struct RtPartCensusKeyHash {
	size_t operator()(const RtPartCensusKey& k) const {
		uint64_t h = RtPartMix(RtPartMix(RtPartMix(k.vs, k.ps), k.address), (uint64_t {k.width} << 32) | k.height);
		h          = RtPartMix(h, static_cast<uint64_t>(static_cast<uint32_t>(k.format)));
		h          = RtPartMix(h, uint64_t {k.blend} | (uint64_t {k.src} << 8) | (uint64_t {k.dst} << 16) | (uint64_t {k.comb} << 24) |
		                              (uint64_t {k.zt} << 32) | (uint64_t {k.zw} << 40) | (uint64_t {k.indirect} << 48) | (uint64_t {k.path} << 56));
		return static_cast<size_t>(h);
	}
};
struct RtPartCensusEntry {
	uint32_t order = 0, draws = 0, instances = 0, indices = 0;
};
struct RtPartDispatchKey {
	uint64_t cs = 0, address = 0;
	uint32_t width = 0, height = 0;
	bool     operator==(const RtPartDispatchKey&) const = default;
};
struct RtPartDispatchKeyHash {
	size_t operator()(const RtPartDispatchKey& k) const {
		return static_cast<size_t>(RtPartMix(RtPartMix(k.cs, k.address), (uint64_t {k.width} << 32) | k.height));
	}
};
struct RtPartState {
	// The scene color images the upscaler read in the last flips (guest addresses; the game rotates them).
	std::array<uint64_t, 4> colors {};
	std::array<uint32_t, 4> color_w {}, color_h {};
	uint32_t                color_next = 0;
	// The pre-particle copy.
	VulkanImage   pre;
	vk::ImageView pre_view = nullptr;
	uint64_t      pre_flip = UINT64_MAX, pre_address = 0;
	bool          taken = false; // this flip
	uint64_t      flip  = UINT64_MAX;
	bool          resuming = false;
	// Statistics: copies taken, flips with none, the copy's draw's shaders.
	uint64_t copies = 0, flips_without = 0, uses = 0, misses = 0;
	uint64_t copy_vs = 0, copy_ps = 0;
	uint32_t copy_order = 0;
	// Census (debug): the last flip's passes onto the scene color (and, in order, the first draw of each key).
	std::unordered_map<RtPartCensusKey, RtPartCensusEntry, RtPartCensusKeyHash>       draws;
	std::unordered_map<RtPartDispatchKey, RtPartCensusEntry, RtPartDispatchKeyHash>   dispatches;
	uint32_t                                                                           order = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
};
RtPartState g_rtpart;

int RtParticlesLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_RT_PARTICLES");
		return value != nullptr && *value != '\0' ? std::clamp(std::atoi(value), 0, 2) : 0;
	}();
	return level;
}
bool RtPartColorKnown(uint64_t address, uint32_t width, uint32_t height, bool strict_size) {
	for (uint32_t i = 0; i < g_rtpart.colors.size(); ++i)
		if (g_rtpart.colors[i] != 0 && g_rtpart.colors[i] == address &&
		    (!strict_size || (g_rtpart.color_w[i] == width && g_rtpart.color_h[i] == height)))
			return true;
	return false;
}
// A new flip: the census of the last one logged (every 10 s), everything reset.
void RtPartNewFlip(uint64_t flip) {
	if (g_rtpart.flip == flip) return;
	if (g_rtpart.flip != UINT64_MAX && RtParticlesLevel() > 0 && !g_rtpart.taken) ++g_rtpart.flips_without;
	const auto now = std::chrono::steady_clock::now();
	if (now - g_rtpart.last_log >= std::chrono::seconds(10) && g_rtpart.flip != UINT64_MAX) {
		g_rtpart.last_log = now;
		if (RtParticlesDebug()) {
			std::vector<std::pair<RtPartCensusKey, RtPartCensusEntry>> list(g_rtpart.draws.begin(), g_rtpart.draws.end());
			std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.second.order < b.second.order; });
			std::printf("RT particles census (flip %llu): %zu draw groups onto the upscaler's color images, %zu storage dispatches; "
			            "copy taken %s (VS %016llx PS %016llx, pass #%u); %llu copies, %llu flips without, SSGI used %llu / missed %llu\n",
			            static_cast<unsigned long long>(g_rtpart.flip), list.size(), g_rtpart.dispatches.size(),
			            g_rtpart.taken ? "yes" : "no", static_cast<unsigned long long>(g_rtpart.copy_vs),
			            static_cast<unsigned long long>(g_rtpart.copy_ps), g_rtpart.copy_order, static_cast<unsigned long long>(g_rtpart.copies),
			            static_cast<unsigned long long>(g_rtpart.flips_without), static_cast<unsigned long long>(g_rtpart.uses),
			            static_cast<unsigned long long>(g_rtpart.misses));
			uint32_t shown = 0;
			for (const auto& [k, e]: list) {
				if (shown++ >= 48) break;
				const char* path = k.path == 1 ? "table" : k.path == 2 ? "native" : "normal";
				std::printf("  #%-3u %-6s VS %016llx PS %016llx -> 0x%llx %ux%u fmt %d %s%s%s: %u draws, %u inst, %u idx\n", e.order, path,
				            static_cast<unsigned long long>(k.vs), static_cast<unsigned long long>(k.ps), static_cast<unsigned long long>(k.address),
				            k.width, k.height, k.format, k.blend ? "blend" : "opaque", k.zw ? " zwrite" : (k.zt ? " ztest" : " nodepth"),
				            k.indirect ? " indirect" : "", e.draws, e.instances, e.indices);
				if (k.blend)
					std::printf("        blend src %u dst %u fn %u\n", k.src, k.dst, k.comb);
			}
			std::vector<std::pair<RtPartDispatchKey, RtPartCensusEntry>> ds(g_rtpart.dispatches.begin(), g_rtpart.dispatches.end());
			std::sort(ds.begin(), ds.end(), [](const auto& a, const auto& b) { return a.second.order < b.second.order; });
			shown = 0;
			for (const auto& [k, e]: ds) {
				if (shown++ >= 24) break;
				std::printf("  #%-3u CS %016llx writes 0x%llx %ux%u: %u dispatches\n", e.order, static_cast<unsigned long long>(k.cs),
				            static_cast<unsigned long long>(k.address), k.width, k.height, e.draws);
			}
			std::fflush(stdout);
		} else if (RtParticlesLevel() > 0) {
			std::printf("RT particles: %llu pre-particle copies, %llu flips without one, SSGI used %llu / missed %llu (copy at VS %016llx PS %016llx)\n",
			            static_cast<unsigned long long>(g_rtpart.copies), static_cast<unsigned long long>(g_rtpart.flips_without),
			            static_cast<unsigned long long>(g_rtpart.uses), static_cast<unsigned long long>(g_rtpart.misses),
			            static_cast<unsigned long long>(g_rtpart.copy_vs), static_cast<unsigned long long>(g_rtpart.copy_ps));
			std::fflush(stdout);
		}
	}
	g_rtpart.flip  = flip;
	g_rtpart.taken = false;
	g_rtpart.order = 0;
	g_rtpart.draws.clear();
	g_rtpart.dispatches.clear();
}
// The upscaler's color this frame (Upscale): remembered so the next frames' draws into it are known.
void RtPartNoteColor(uint64_t address, vk::Extent2D extent) {
	if (address == 0) return;
	for (uint32_t i = 0; i < g_rtpart.colors.size(); ++i)
		if (g_rtpart.colors[i] == address) {
			g_rtpart.color_w[i] = extent.width, g_rtpart.color_h[i] = extent.height;
			return;
		}
	const uint32_t at    = g_rtpart.color_next++ % g_rtpart.colors.size();
	g_rtpart.colors[at]  = address;
	g_rtpart.color_w[at] = extent.width;
	g_rtpart.color_h[at] = extent.height;
}
// Level >= 1 and the copy of this flip's scene color exists and is the upscaler's color: its view (else null).
vk::ImageView RtPartPreView(uint64_t color_address, vk::Extent2D extent) {
	if (RtParticlesLevel() == 0) return nullptr;
	const bool ok = g_rtpart.taken && g_rtpart.pre_view != nullptr && g_rtpart.pre_address == color_address &&
	                g_rtpart.pre_flip == GuestFlips() && g_rtpart.pre.extent.width == extent.width &&
	                g_rtpart.pre.extent.height == extent.height;
	if (ok) ++g_rtpart.uses;
	else ++g_rtpart.misses;
	return ok ? g_rtpart.pre_view : nullptr;
}
} // namespace

bool RtParticlesDebug() {
	static const bool debug = [] {
		const char* value = std::getenv("KYTY_RT_PARTICLES_DEBUG");
		return value != nullptr && *value == '1';
	}();
	return debug;
}
bool RtParticlesWanted() { return RtParticlesLevel() > 0 || RtParticlesDebug(); }

void NoteSceneDispatch(uint64_t cs_hash, uint64_t address, uint32_t width, uint32_t height) {
	if (!RtParticlesDebug() || !RtPartColorKnown(address, width, height, false)) return;
	RtPartNewFlip(GuestFlips());
	auto& e = g_rtpart.dispatches[RtPartDispatchKey {cs_hash, address, width, height}];
	if (e.draws == 0) e.order = ++g_rtpart.order;
	++e.draws;
}
void NoteOtherPathDraw(uint64_t address, uint64_t vs, bool blend, int path) {
	if (!RtParticlesDebug() || !RtPartColorKnown(address, 0, 0, false)) return;
	RtPartNewFlip(GuestFlips());
	RtPartCensusKey key;
	key.vs = vs, key.address = address, key.blend = blend ? 1 : 0, key.path = static_cast<uint8_t>(path);
	auto& e = g_rtpart.draws[key];
	if (e.draws == 0) e.order = ++g_rtpart.order;
	++e.draws;
}
void BeforeSceneDraw(CommandBuffer& command, TextureCache& cache, const RenderState& rendering, vk::ImageLayout layout,
                     const SceneDraw& draw) {
	if (g_rtpart.resuming || !RtParticlesWanted()) return;
	// Only the images the upscaler reads (the scene color, one of a few the game rotates), at their size.
	if (!RtPartColorKnown(draw.address, draw.width, draw.height, true)) return;
	RtPartNewFlip(GuestFlips());
	if (RtParticlesDebug()) {
		RtPartCensusKey key;
		key.vs = draw.vs, key.ps = draw.ps, key.address = draw.address, key.width = draw.width, key.height = draw.height;
		key.format = draw.format, key.blend = draw.blend ? 1 : 0, key.src = draw.src_blend, key.dst = draw.dst_blend;
		key.comb = draw.comb, key.zt = draw.depth_test ? 1 : 0, key.zw = draw.depth_write ? 1 : 0, key.indirect = draw.indirect ? 1 : 0;
		auto& e = g_rtpart.draws[key];
		if (e.draws == 0) e.order = ++g_rtpart.order;
		++e.draws;
		e.instances += draw.instances;
		e.indices += draw.index_count;
	}
	if (RtParticlesLevel() == 0 || g_rtpart.taken) return;
	// The particle pass's first draw: blended, depth test on, depth writes off (KYTY_RT_PARTICLES_VS pins the shaders).
	static const std::vector<uint64_t> pinned = [] {
		std::vector<uint64_t> out;
		if (const char* v = std::getenv("KYTY_RT_PARTICLES_VS"); v != nullptr)
			for (const char* at = v; *at != '\0';) {
				char*                    end = nullptr;
				const unsigned long long h   = std::strtoull(at, &end, 16);
				if (end == at) break;
				if (h != 0) out.push_back(h);
				if (*end != ',' && *end != ';') break;
				at = end + 1;
			}
		return out;
	}();
	// (KYTY_RT_PARTICLES_ANYDEPTH=1: a blended draw without the depth test counts too - a pass that fades its particles
	// against a sampled depth instead of testing it.)
	static const bool any_depth = [] {
		const char* v = std::getenv("KYTY_RT_PARTICLES_ANYDEPTH");
		return v != nullptr && *v == '1';
	}();
	// (A single quad / triangle is a full-screen pass - fog, composites - not particles: skipped by the heuristic.)
	const bool full_screen = !draw.indirect && draw.index_count <= 6 && draw.instances <= 1;
	const bool particle    = pinned.empty() ? (draw.blend && !draw.depth_write && (draw.depth_test || any_depth) && !full_screen)
	                                        : std::find(pinned.begin(), pinned.end(), draw.vs) != pinned.end();
	if (!particle) return;
	auto& image = cache.GetImage(draw.target);
	if (image.backing.image == nullptr || image.info.data.address != draw.address) return;
	if (image.backing.extent.width != draw.width || image.backing.extent.height != draw.height) return;
	if (image.backing.format != vk::Format::eR16G16B16A16Sfloat) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("RT particles: the scene color is not RGBA16F (format %d): no pre-particle copy\n",
			            static_cast<int>(image.backing.format));
			std::fflush(stdout);
		}
		return;
	}
	auto& graphics = command.GetGraphics();
	ConfigureAs(graphics, g_rtpart.pre, g_rtpart.pre_view, image.backing.format, image.backing.extent);
	command.EndRendering();
	const auto vk_command = command.Handle();
	CopyInto(vk_command, image, g_rtpart.pre); // (leaves the copy in SHADER_READ_ONLY_OPTIMAL)
	image.Transit(layout, vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite, {}, vk_command);
	g_rtpart.resuming = true;
	command.ResumeRendering(rendering);
	g_rtpart.resuming    = false;
	g_rtpart.taken       = true;
	g_rtpart.pre_flip    = g_rtpart.flip;
	g_rtpart.pre_address = draw.address;
	g_rtpart.copy_vs     = draw.vs;
	g_rtpart.copy_ps     = draw.ps;
	g_rtpart.copy_order  = g_rtpart.order;
	++g_rtpart.copies;
	static bool said = false;
	if (!said) {
		said = true;
		std::printf("RT particles: level %d: the scene color (0x%llx, %ux%u) copied before its first blended depth-write-off draw "
		            "(VS %016llx PS %016llx, blend src %u dst %u fn %u)%s\n",
		            RtParticlesLevel(), static_cast<unsigned long long>(draw.address), draw.width, draw.height,
		            static_cast<unsigned long long>(draw.vs), static_cast<unsigned long long>(draw.ps), draw.src_blend, draw.dst_blend,
		            draw.comb, pinned.empty() ? "" : " (KYTY_RT_PARTICLES_VS)");
		std::fflush(stdout);
	}
}

namespace {
// KYTY_SSGI=0|1|2 (ssgi.comp): screen-space ray-traced GI + AO on the PS5 Pro path's render-resolution color, recorded
// right before the upscaler (DLSS / XeSS / FSR), which then reads our composited copy in place of the game's color, so
// the upscaler and frame generation see it. Trace (1: half res, 1 ray/px; 2 "Alto": full res, 3 rays/px; 3 "Ultra": full res, 5 rays/px, 28 steps + refine),
// SVGF-style variance-guided temporal + a-trous (3 passes, 4 at Ultra), Ultra anti-flicker, temporal (alpha 0.1, clamped),
// 2 a-trous passes, bilateral upsample. KYTY_SSGI_INTENSITY (0.85 / 1.1 / 1.25 by quality), KYTY_SSGI_AO (0.25).
// Normals from depth derivatives; view positions from the game's view_to_clip (handles reverse-Z).
// (rtfire) KYTY_SSGI_EMISSIVE (strength, 0..4; default 1.0 at KYTY_SSGI >= 2, 0 at 1): "fire light": torches, bonfires,
// lava and other emissive pixels become area lights (one per cell of a 32x18 grid, ssgi.comp mode 5) found from the
// lit colour against the G-buffer albedo (KYTY_RT_REFLECTIONS' packed copy, layer 0 bytes 0..2), shadowed by
// screen-space rays, denoised (temporal + 2 a-trous) and added as albedo x irradiance. KYTY_SSGI_EMISSIVE_LUM (exposed
// luminance threshold, 1.5), KYTY_SSGI_EMISSIVE_RATIO (lit colour / albedo threshold, 12), KYTY_SSGI_EMISSIVE_RANGE
// (m, 8), KYTY_SSGI_EMISSIVE_VIEW (1 the emitter mask, 2 the added light only). Every 10 s "SSGI emissive:" stats.
struct SsgiImage {
	VulkanImage   image;
	vk::ImageView view = nullptr;
};
struct SsgiPass {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             point      = nullptr;
	vk::Sampler             linear     = nullptr;
	bool                    failed     = false;
	vk::Extent2D            render {}, trace {};
	SsgiImage               raw, tmp_a, tmp_b, out;
	SsgiImage               history[2], gbuf[2];
	SsgiImage               moments[2], var_a, var_b; // (luma m1, m2, count, variance) / filtered variance (trace res)
	SsgiImage               gihist[2];                // the composited GI (render res), level 3's anti-flicker
	// (rtfire) Emissive light: its trace output, temporal/a-trous ping-pong, history and moments (trace res; only
	// when KYTY_SSGI_EMISSIVE > 0), the light list + counters, their host copies (a ring of 4), a 1x1 R32_UINT stand-in.
	SsgiImage               emi_raw, emi_tmp, emi_hist[2], emi_mom[2];
	SsgiImage               reactive; // (rtparticles) KYTY_RT_PARTICLES=2: the particle reactive mask (R8_UNORM, render size)
	bool                    reactive_failed = false;
	VkBuffer                lights = VK_NULL_HANDLE, readback = VK_NULL_HANDLE;
	VmaAllocation           lights_mem = VK_NULL_HANDLE, readback_mem = VK_NULL_HANDLE;
	const uint32_t*         readback_map = nullptr;
	VkImage                 dummy_image = VK_NULL_HANDLE;
	VmaAllocation           dummy_mem   = VK_NULL_HANDLE;
	vk::ImageView           dummy_view  = nullptr;
	bool                    dummy_ready = false, emi_failed = false;
	double                  em_acc[8] {};
	uint32_t                em_frames = 0, em_gbuf_frames = 0;
	std::chrono::steady_clock::time_point em_last_log = std::chrono::steady_clock::now();
	uint32_t                frame         = 0;
	int                     quality       = 0;
	bool                    history_valid = false;
	// GPU time: two timestamps a frame, a ring of 8, read 4 frames later.
	vk::QueryPool           pool      = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms  = 0.0;
	uint32_t                samples = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
} g_ssgi;

struct SsgiPush {
	float   proj[4];
	float   depth_map[4];
	float   motion[4];
	float   trace[4];
	int32_t misc[4];
	float   extra[4];
};
static_assert(sizeof(SsgiPush) == 96);
// (rtfire) SSGI's own push constants: SsgiPush (shared with GTAO's, which extends it: unchanged) + the emissive terms.
struct SsgiPushE : SsgiPush {
	float   emis[4];  // strength x gain, exposed luminance threshold, lit/albedo ratio threshold, range (view units)
	int32_t emisc[4]; // flags (1 on, 2 G-buffer albedo), exposure index, view mode, -
};
static_assert(sizeof(SsgiPushE) == 128);
constexpr uint32_t SsgiCells       = 32 * 18;
constexpr uint32_t SsgiLightsBytes = 64 + SsgiCells * 4 + SsgiCells * 48; // hdr[16], list, 3 vec4 a cell
constexpr uint32_t SsgiStatWords   = 8;
// (rtparticles) 15 the pre-particle scene color (sampled), 16 the reactive mask (R8 storage).
constexpr uint32_t SsgiBindings    = 17;
vk::DescriptorType SsgiDescType(uint32_t i) {
	if (i == 13 || i == 14) return vk::DescriptorType::eStorageBuffer;
	if (i == 7 || i == 8 || i == 10 || i == 16) return vk::DescriptorType::eStorageImage;
	return vk::DescriptorType::eCombinedImageSampler;
}

int SsgiQuality() {
	static const int quality = [] {
		const char* value = std::getenv("KYTY_SSGI");
		return value != nullptr ? std::clamp(std::atoi(value), 0, 3) : 0;
	}();
	return quality;
}
float SsgiEnv(const char* name, float fallback, float hi) {
	const char* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? std::clamp(static_cast<float>(std::atof(value)), 0.0f, hi) : fallback;
}
// (rtfire) The emissive light's strength: 0 off (nothing created); on by default from KYTY_SSGI=2.
float SsgiEmissive() {
	static const float strength = SsgiQuality() <= 0 ? 0.0f : SsgiEnv("KYTY_SSGI_EMISSIVE", SsgiQuality() >= 2 ? 1.0f : 0.0f, 4.0f);
	return strength;
}
std::array<SsgiImage*, 6> SsgiEmisAll() {
	return {&g_ssgi.emi_raw, &g_ssgi.emi_tmp, &g_ssgi.emi_hist[0], &g_ssgi.emi_hist[1], &g_ssgi.emi_mom[0], &g_ssgi.emi_mom[1]};
}
// The light buffer, its host ring and the stand-in image (once; on failure the emissive light stays off).
bool SsgiEmisCreate(GraphicContext& graphics) {
	if (g_ssgi.lights != VK_NULL_HANDLE) return true;
	if (g_ssgi.emi_failed) return false;
	g_ssgi.emi_failed = true;
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = SsgiLightsBytes;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VmaAllocationCreateInfo dev {};
	dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_ssgi.lights, &g_ssgi.lights_mem, nullptr) != VK_SUCCESS) {
		g_ssgi.lights = VK_NULL_HANDLE;
		return false;
	}
	buffer.size  = 4 * SsgiStatWords * 4;
	buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_ssgi.readback, &g_ssgi.readback_mem, &mapped) != VK_SUCCESS) {
		g_ssgi.readback = VK_NULL_HANDLE;
		return false;
	}
	g_ssgi.readback_map = static_cast<const uint32_t*>(mapped.pMappedData);
	VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create.imageType   = VK_IMAGE_TYPE_2D;
	create.format      = VK_FORMAT_R32_UINT;
	create.extent      = {1, 1, 1};
	create.mipLevels   = 1;
	create.arrayLayers = 1;
	create.samples     = VK_SAMPLE_COUNT_1_BIT;
	create.tiling      = VK_IMAGE_TILING_OPTIMAL;
	create.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO;
	if (vmaCreateImage(graphics.allocator, &create, &allocate, &g_ssgi.dummy_image, &g_ssgi.dummy_mem, nullptr) != VK_SUCCESS) {
		g_ssgi.dummy_image = VK_NULL_HANDLE;
		return false;
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = vk::Image(g_ssgi.dummy_image);
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = vk::Format::eR32Uint;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &g_ssgi.dummy_view) != vk::Result::eSuccess) return false;
	g_ssgi.emi_failed = false;
	return true;
}
// Every frame: this frame's counters from 3 frames ago (statistics only); every 10 s the "SSGI emissive:" line.
void SsgiEmisStats(bool have_gbuf) {
	if (g_ssgi.readback_map == nullptr) return;
	if (g_ssgi.frame >= 4) {
		const uint32_t* w = g_ssgi.readback_map + ((g_ssgi.frame + 1) % 4) * SsgiStatWords; // (written 3 frames ago)
		for (uint32_t i = 0; i < SsgiStatWords; ++i)
			g_ssgi.em_acc[i] += static_cast<double>(w[i]);
		++g_ssgi.em_frames;
		if (have_gbuf) ++g_ssgi.em_gbuf_frames;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - g_ssgi.em_last_log < std::chrono::seconds(10) || g_ssgi.em_frames == 0) return;
	const double* a = g_ssgi.em_acc;
	const double  added = a[4] > 0 ? a[6] / 4096.0 / a[4] : 0.0, own = a[4] > 0 ? a[7] / 256.0 / a[4] : 0.0;
	std::printf("SSGI emissive: %.2f %% emissive pixels, %.1f lights a frame, mean added luminance %.4f exposed (%.1f %% of the "
	            "pixels' own %.3f), %.1f %% of the pixels lit > 0.002, G-buffer albedo %u/%u frames\n",
	            a[2] > 0 ? 100.0 * a[3] / a[2] : 0.0, a[0] / g_ssgi.em_frames, added, own > 0 ? 100.0 * added / own : 0.0, own,
	            a[4] > 0 ? 100.0 * a[5] / a[4] : 0.0, g_ssgi.em_gbuf_frames, g_ssgi.em_frames);
	std::fflush(stdout);
	for (double& v: g_ssgi.em_acc)
		v = 0.0;
	g_ssgi.em_frames = g_ssgi.em_gbuf_frames = 0;
	g_ssgi.em_last_log = now;
}

bool SsgiMakeImage(GraphicContext& graphics, SsgiImage& img, vk::Extent2D extent,
                   vk::Format format = vk::Format::eR16G16B16A16Sfloat) {
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
	               vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	create.sharingMode = vk::SharingMode::eExclusive;
	create.samples     = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, img.image)) {
		img.image.image = nullptr;
		return false;
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = img.image.image;
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = format;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &img.view) != vk::Result::eSuccess) return false;
	img.image.state = {};
	return true;
}
void SsgiFreeImage(GraphicContext& graphics, SsgiImage& img) {
	if (img.view != nullptr) graphics.device.destroyImageView(img.view, nullptr);
	img.view = nullptr;
	if (img.image.image != nullptr) graphics.DeleteImage(img.image);
	img.image.image = nullptr;
}

std::array<SsgiImage*, 14> SsgiAll() {
	return {&g_ssgi.raw,        &g_ssgi.tmp_a,      &g_ssgi.tmp_b,      &g_ssgi.out,   &g_ssgi.history[0],
	        &g_ssgi.history[1], &g_ssgi.gbuf[0],    &g_ssgi.gbuf[1],    &g_ssgi.moments[0], &g_ssgi.moments[1],
	        &g_ssgi.var_a,      &g_ssgi.var_b,      &g_ssgi.gihist[0],  &g_ssgi.gihist[1]};
}

bool SsgiEnsure(GraphicContext& graphics, vk::Extent2D render, int quality) {
	const auto device = graphics.device;
	if (g_ssgi.pipeline == nullptr) {
		if (g_ssgi.failed) return false;
		g_ssgi.failed = true;
		std::array<vk::DescriptorSetLayoutBinding, SsgiBindings> bindings {};
		for (uint32_t i = 0; i < SsgiBindings; ++i) {
			bindings[i].binding         = i;
			bindings[i].descriptorType  = SsgiDescType(i);
			bindings[i].descriptorCount = 1;
			bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
		}
		vk::DescriptorSetLayoutCreateInfo set_info {};
		set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		set_info.bindingCount = static_cast<uint32_t>(bindings.size());
		set_info.pBindings    = bindings.data();
		if (device.createDescriptorSetLayout(&set_info, nullptr, &g_ssgi.set_layout) != vk::Result::eSuccess) return false;
		vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(SsgiPushE)};
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.setLayoutCount         = 1;
		layout_info.pSetLayouts            = &g_ssgi.set_layout;
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges    = &push;
		if (device.createPipelineLayout(&layout_info, nullptr, &g_ssgi.layout) != vk::Result::eSuccess) return false;
		vk::ShaderModuleCreateInfo module_info {};
		module_info.codeSize    = sizeof(SSGI_SPV);
		module_info.pCode       = SSGI_SPV;
		vk::ShaderModule module = nullptr;
		if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
		pipeline_info.stage.module = module;
		pipeline_info.stage.pName  = "main";
		pipeline_info.layout       = g_ssgi.layout;
		const auto created = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_ssgi.pipeline);
		device.destroyShaderModule(module, nullptr);
		if (created != vk::Result::eSuccess) {
			g_ssgi.pipeline = nullptr;
			return false;
		}
		for (int i = 0; i < 2; ++i) {
			vk::SamplerCreateInfo sampler {};
			sampler.magFilter    = i == 0 ? vk::Filter::eNearest : vk::Filter::eLinear;
			sampler.minFilter    = sampler.magFilter;
			sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
			sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
			sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
			if (device.createSampler(&sampler, nullptr, i == 0 ? &g_ssgi.point : &g_ssgi.linear) != vk::Result::eSuccess) return false;
		}
		vk::QueryPoolCreateInfo query {};
		query.queryType  = vk::QueryType::eTimestamp;
		query.queryCount = 16;
		if (device.createQueryPool(&query, nullptr, &g_ssgi.pool) != vk::Result::eSuccess) g_ssgi.pool = nullptr;
		g_ssgi.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
		g_ssgi.failed    = false;
	}
	if (g_ssgi.out.image.image != nullptr && g_ssgi.render == render && g_ssgi.quality == quality) return true;
	device.waitIdle();
	for (auto* img : SsgiAll())
		SsgiFreeImage(graphics, *img);
	for (auto* img : SsgiEmisAll())
		SsgiFreeImage(graphics, *img);
	SsgiFreeImage(graphics, g_ssgi.reactive);
	const vk::Extent2D trace = quality >= 2 ? render : vk::Extent2D {std::max(render.width / 2, 1u), std::max(render.height / 2, 1u)};
	bool ok = SsgiMakeImage(graphics, g_ssgi.out, render);
	for (auto* img : {&g_ssgi.raw, &g_ssgi.tmp_a, &g_ssgi.tmp_b, &g_ssgi.history[0], &g_ssgi.history[1], &g_ssgi.gbuf[0], &g_ssgi.gbuf[1],
	                  &g_ssgi.moments[0], &g_ssgi.moments[1], &g_ssgi.var_a, &g_ssgi.var_b})
		ok = ok && SsgiMakeImage(graphics, *img, trace);
	for (auto* img : {&g_ssgi.gihist[0], &g_ssgi.gihist[1]})
		ok = ok && SsgiMakeImage(graphics, *img, render);
	if (ok && SsgiEmissive() > 0.0f && SsgiEmisCreate(graphics)) {
		bool emi_ok = true;
		for (auto* img : SsgiEmisAll())
			emi_ok = emi_ok && SsgiMakeImage(graphics, *img, trace);
		if (!emi_ok) {
			for (auto* img : SsgiEmisAll())
				SsgiFreeImage(graphics, *img);
			std::printf("SSGI: emissive light off (could not create its images)\n");
			std::fflush(stdout);
		}
	}
	// (rtparticles) The reactive mask (level 2); without it the level stays 1.
	if (ok && RtParticlesLevel() >= 2 && !g_ssgi.reactive_failed && !SsgiMakeImage(graphics, g_ssgi.reactive, render, vk::Format::eR8Unorm)) {
		SsgiFreeImage(graphics, g_ssgi.reactive);
		g_ssgi.reactive_failed = true;
		std::printf("RT particles: no reactive mask (its image could not be created): level 1\n");
		std::fflush(stdout);
	}
	if (!ok) {
		SsgiFreeImage(graphics, g_ssgi.reactive);
		for (auto* img : SsgiEmisAll())
			SsgiFreeImage(graphics, *img);
		for (auto* img : SsgiAll())
			SsgiFreeImage(graphics, *img);
		g_ssgi.failed = true;
		return false;
	}
	g_ssgi.render        = render;
	g_ssgi.quality       = quality;
	g_ssgi.trace         = trace;
	g_ssgi.history_valid = false;
	return true;
}

void SsgiReadTiming() {
	if (g_ssgi.pool == nullptr || g_graphics_ctx == nullptr) return;
	while (g_ssgi.read + 4 <= g_ssgi.written) {
		const auto entry = static_cast<uint32_t>(g_ssgi.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (g_graphics_ctx->device.getQueryPoolResults(g_ssgi.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                               vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		g_ssgi.sum_ms += static_cast<double>(stamps[1] - stamps[0]) * g_ssgi.period_ns / 1e6;
		++g_ssgi.samples;
		++g_ssgi.read;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - g_ssgi.last_log >= std::chrono::seconds(5) && g_ssgi.samples > 0) {
		std::printf("SSGI: GPU %.3f ms (mean of %u frames)\n", g_ssgi.sum_ms / g_ssgi.samples, g_ssgi.samples);
		std::fflush(stdout);
		g_ssgi.sum_ms   = 0.0;
		g_ssgi.samples  = 0;
		g_ssgi.last_log = now;
	}
}

// Records the effect; on success `color` is replaced by our composited rgba16f image (general layout).
bool SsgiApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int quality = SsgiQuality();
	const auto skip = [](const char* why) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("SSGI: skipped at the first upscaler frame (%s)\n", why);
			std::fflush(stdout);
		}
	};
	if (quality == 0) return skip("KYTY_SSGI is 0 or unset"), false;
	if (g_graphics_ctx == nullptr) return skip("no graphics context"), false;
	if (VideoOut::MoviePlaying()) {
		g_ssgi.history_valid = false;
		return false;
	}
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || inputs.motion.view == nullptr || color.extent.width < 16 ||
	    color.extent.height < 16)
		return skip("missing color/depth/motion view"), false;
	if (!SsgiEnsure(*g_graphics_ctx, color.extent, quality)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("SSGI: off (could not create its pipeline or images)\n");
			std::fflush(stdout);
		}
		return false;
	}
	const float intensity = SsgiEnv("KYTY_SSGI_INTENSITY", quality >= 3 ? 1.25f : quality >= 2 ? 1.1f : 0.85f, 4.0f);
	const float ao        = SsgiEnv("KYTY_SSGI_AO", 0.25f, 1.0f);
	const auto& m         = inputs.view_to_clip;
	SsgiPushE   push {};
	const bool  have_proj = m[0] != 0.0f && m[5] != 0.0f && (m[11] != 0.0f || m[15] != 0.0f);
	if (have_proj) {
		push.proj[0] = m[0], push.proj[1] = m[5], push.proj[2] = m[8], push.proj[3] = m[9];
		push.depth_map[0] = m[10], push.depth_map[1] = m[14], push.depth_map[2] = m[11], push.depth_map[3] = m[15];
	} else {
		// No projection: a 60 degree vertical FOV, reverse-Z infinite (d = near / z), the render's aspect.
		const float f  = 1.0f / std::tan(0.5f * 1.0472f);
		const float ar = static_cast<float>(color.extent.width) / static_cast<float>(color.extent.height);
		push.proj[0] = f / ar, push.proj[1] = f, push.proj[2] = 0.0f, push.proj[3] = 0.0f;
		push.depth_map[0] = 0.0f, push.depth_map[1] = inputs.near_plane, push.depth_map[2] = 1.0f, push.depth_map[3] = 0.0f;
	}
	push.motion[0] = inputs.motion_scale_x, push.motion[1] = inputs.motion_scale_y;
	push.motion[2] = static_cast<float>(color.extent.width), push.motion[3] = static_cast<float>(color.extent.height);
	push.trace[0] = static_cast<float>(g_ssgi.trace.width), push.trace[1] = static_cast<float>(g_ssgi.trace.height);
	push.trace[2] = intensity, push.trace[3] = ao;
	push.extra[1] = quality >= 3 ? 4.0f : 3.5f, push.extra[2] = 0.45f; // ray length / thickness at 1 m, grown with distance
	// (rtfire) The emissive light: on when its images exist; the G-buffer albedo when this flip's (or the last's) copy
	// matches the render; the game's exposure when the upscaler has it (else the light buffer's 1.0).
	const bool emissive = SsgiEmissive() > 0.0f && g_ssgi.lights != VK_NULL_HANDLE && g_ssgi.emi_raw.view != nullptr &&
	                      g_ssgi.emi_mom[1].view != nullptr;
	RtReflGbuffer gbuf;
	const bool    have_albedo   = emissive && RtReflGetGbuffer(gbuf) && gbuf.extent == color.extent && gbuf.views[0] != nullptr;
	const bool    game_exposure = inputs.exposure_buffer != nullptr && inputs.exposure_range >= sizeof(float);
	const float   em_lum   = SsgiEnv("KYTY_SSGI_EMISSIVE_LUM", 1.5f, 100.0f);
	const float   em_ratio = SsgiEnv("KYTY_SSGI_EMISSIVE_RATIO", 12.0f, 1000.0f);
	const float   em_range = std::max(SsgiEnv("KYTY_SSGI_EMISSIVE_RANGE", 8.0f, 50.0f), 0.5f);
	static const int em_view = static_cast<int>(SsgiEnv("KYTY_SSGI_EMISSIVE_VIEW", 0.0f, 2.0f));
	// (rtparticles) The pre-particle copy of this flip's scene color: the emitters' radiance is color - copy (flag 4);
	// level 2 also writes the reactive mask (mode 6) the upscaler takes.
	const vk::ImageView pre      = RtPartPreView(inputs.color_address, color.extent);
	const bool          reactive = pre != nullptr && RtParticlesLevel() >= 2 && g_ssgi.reactive.view != nullptr;
	static const float  reactive_strength = SsgiEnv("KYTY_RT_PARTICLES_REACTIVE", 0.8f, 1.0f);
	push.extra[3]                         = reactive_strength;
	if (emissive) {
		// (3: the internal gain over the physical L x A x cos x cos / d^2: game flames are far dimmer than real fire)
		push.emis[0] = SsgiEmissive() * 3.0f, push.emis[1] = std::max(em_lum, 0.05f), push.emis[2] = std::max(em_ratio, 1.0f);
		push.emis[3]  = em_range;
		push.emisc[0] = 1 | (have_albedo ? 2 : 0) | (pre != nullptr ? 4 : 0);
		push.emisc[1] = game_exposure ? static_cast<int32_t>(inputs.exposure_index) : 1;
		push.emisc[2] = em_view;
	}
	if (pre != nullptr) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("RT particles: the SSGI %s takes the particles' own emission (level %d%s)\n",
			            emissive ? "emissive light" : "pass (emissive light off: KYTY_SSGI_EMISSIVE=0 or quality 1)", RtParticlesLevel(),
			            reactive ? ", reactive mask on" : RtParticlesLevel() >= 2 ? ", reactive mask off: no image" : "");
			std::fflush(stdout);
		}
	}
	{
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("SSGI: on (quality %d, %ux%u, intensity %g, ao %g)%s\n", quality, g_ssgi.trace.width, g_ssgi.trace.height,
			            intensity, ao, have_proj ? "" : " (no projection: 60 deg FOV assumed)");
			std::printf("SSGI: color format %d, depth format %d, view_to_clip z terms %g %g %g %g\n", static_cast<int>(color.format),
			            static_cast<int>(inputs.depth.format), m[10], m[14], m[11], m[15]);
			if (emissive)
				std::printf("SSGI: emissive light on (strength %g, luminance > %g exposed, lit/albedo > %g, range %g m, %s, exposure %s%s)\n",
				            SsgiEmissive(), push.emis[1], push.emis[2], em_range,
				            have_albedo ? "G-buffer albedo" : "no G-buffer copy yet: luminance only until it comes",
				            game_exposure ? "the game's" : "1.0", em_view == 1 ? ", view: emitter mask" : em_view == 2 ? ", view: added light" : "");
			else if (SsgiEmissive() > 0.0f)
				std::printf("SSGI: emissive light off (its buffer or images could not be created)\n");
			else
				std::printf("SSGI: emissive light off (KYTY_SSGI_EMISSIVE=0%s)\n", quality < 2 ? ", the default at quality 1" : "");
			std::fflush(stdout);
		}
	}
	SsgiReadTiming();
	if (emissive) SsgiEmisStats(have_albedo);
	const auto     entry = static_cast<uint32_t>(g_ssgi.written % 8);
	const bool     timed = g_ssgi.pool != nullptr;
	if (timed) {
		command.resetQueryPool(g_ssgi.pool, entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_ssgi.pool, entry * 2);
	}
	const auto global = [&](vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src, vk::PipelineStageFlags2 dst_stage,
	                        vk::AccessFlags2 dst) {
		vk::MemoryBarrier2 barrier {};
		barrier.srcStageMask  = src_stage;
		barrier.srcAccessMask = src;
		barrier.dstStageMask  = dst_stage;
		barrier.dstAccessMask = dst;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &barrier;
		command.pipelineBarrier2(dependency);
	};
	// Our images general throughout (the first use from undefined).
	for (auto* img : SsgiAll())
		if (img->image.state.layout != vk::ImageLayout::eGeneral)
			Transition(command, img->image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
			           vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader);
	if (emissive) {
		for (auto* img : SsgiEmisAll())
			if (img->image.state.layout != vk::ImageLayout::eGeneral)
				Transition(command, img->image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
				           vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader);
		if (!g_ssgi.dummy_ready) {
			g_ssgi.dummy_ready = true;
			vk::ImageMemoryBarrier2 barrier {};
			barrier.srcStageMask     = vk::PipelineStageFlagBits2::eNone;
			barrier.dstStageMask     = vk::PipelineStageFlagBits2::eTransfer;
			barrier.dstAccessMask    = vk::AccessFlagBits2::eTransferWrite;
			barrier.oldLayout        = vk::ImageLayout::eUndefined;
			barrier.newLayout        = vk::ImageLayout::eGeneral;
			barrier.image            = vk::Image(g_ssgi.dummy_image);
			barrier.subresourceRange = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
			vk::DependencyInfo dependency {};
			dependency.imageMemoryBarrierCount = 1;
			dependency.pImageMemoryBarriers    = &barrier;
			command.pipelineBarrier2(dependency);
			vk::ClearColorValue zero {};
			zero.uint32 = std::array<uint32_t, 4> {0, 0, 0, 0};
			const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
			command.clearColorImage(vk::Image(g_ssgi.dummy_image), vk::ImageLayout::eGeneral, &zero, 1, &whole);
		}
		// This frame's light list and counters cleared; word 1 = 1.0 (the exposure without the game's). The previous
		// frame's reads of the buffer (and its stats copy) before.
		global(vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
		       vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
		std::array<uint32_t, 16> hdr {};
		hdr[1] = std::bit_cast<uint32_t>(1.0f);
		command.updateBuffer(vk::Buffer(g_ssgi.lights), 0, sizeof(hdr), hdr.data());
		global(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
		       vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	}
	// Earlier uses of our images (the previous frame's upscaler read of `out`) before this frame's writes.
	global(vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	       vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	const uint32_t cur  = g_ssgi.frame & 1u;
	const uint32_t prev = cur ^ 1u;
	push.misc[1] = static_cast<int32_t>(g_ssgi.frame);
	push.misc[2] = quality;
	push.misc[3] = (inputs.reset ? 1 : 0) | (g_ssgi.history_valid ? 2 : 0);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_ssgi.pipeline);
	const SsgiImage* aux_in  = &g_ssgi.moments[prev];
	const SsgiImage* aux_out = &g_ssgi.moments[cur];
	const auto run = [&](int mode, const SsgiImage& a, const SsgiImage& b, const SsgiImage& out, const SsgiImage& gout, float step,
	                     vk::Extent2D size) {
		const vk::Sampler       lin = g_ssgi.linear, pt = g_ssgi.point;
		vk::DescriptorImageInfo infos[SsgiBindings] {};
		infos[0]  = {lin, color.view, color.layout};
		infos[1]  = {pt, inputs.depth.view, inputs.depth.layout};
		infos[2]  = {pt, inputs.motion.view, inputs.motion.layout};
		infos[3]  = {lin, a.view, vk::ImageLayout::eGeneral};
		infos[4]  = {lin, b.view, vk::ImageLayout::eGeneral};
		infos[5]  = {pt, g_ssgi.gbuf[cur].view, vk::ImageLayout::eGeneral};
		infos[6]  = {pt, g_ssgi.gbuf[prev].view, vk::ImageLayout::eGeneral};
		infos[7]  = {nullptr, out.view, vk::ImageLayout::eGeneral};
		infos[8]  = {nullptr, gout.view, vk::ImageLayout::eGeneral};
		infos[9]  = {lin, aux_in->view, vk::ImageLayout::eGeneral};
		infos[10] = {nullptr, aux_out->view, vk::ImageLayout::eGeneral};
		// (rtfire) 11 the denoised emissive light, 12 G-buffer layer 0, 13 the light buffer, 14 the exposure (stand-ins
		// when the emissive light is off: nothing reads them then).
		infos[11] = {lin, emissive ? g_ssgi.emi_tmp.view : a.view, vk::ImageLayout::eGeneral};
		infos[12] = have_albedo ? vk::DescriptorImageInfo {pt, gbuf.views[0], vk::ImageLayout::eShaderReadOnlyOptimal}
		                        : vk::DescriptorImageInfo {pt, g_ssgi.dummy_view, vk::ImageLayout::eGeneral};
		// (rtparticles) 15 the pre-particle color (the color itself when there is none: read only with flag 4 / mode 6),
		// 16 the reactive mask (a storage stand-in otherwise; written only by mode 6).
		infos[15] = pre != nullptr ? vk::DescriptorImageInfo {pt, pre, vk::ImageLayout::eShaderReadOnlyOptimal}
		                           : vk::DescriptorImageInfo {pt, color.view, color.layout};
		infos[16] = {nullptr, reactive ? g_ssgi.reactive.view : out.view, vk::ImageLayout::eGeneral};
		const vk::DescriptorBufferInfo lights_info {vk::Buffer(g_ssgi.lights), 0, VK_WHOLE_SIZE};
		const vk::DescriptorBufferInfo exposure_info =
		    emissive && game_exposure ? vk::DescriptorBufferInfo {inputs.exposure_buffer, inputs.exposure_base, inputs.exposure_range}
		                              : lights_info;
		vk::WriteDescriptorSet writes[SsgiBindings] {};
		uint32_t               bound = 0;
		for (uint32_t i = 0; i < SsgiBindings; ++i) {
			// (emissive off: its bindings stay unwritten; the shader reads them only when push.emisc says it is on)
			if (i >= 12 && i <= 14 && !emissive) continue;
			auto& w           = writes[bound++];
			w.dstBinding      = i;
			w.descriptorCount = 1;
			w.descriptorType  = SsgiDescType(i);
			if (i == 13) w.pBufferInfo = &lights_info;
			else if (i == 14) w.pBufferInfo = &exposure_info;
			else w.pImageInfo = &infos[i];
		}
		push.misc[0] = mode;
		push.extra[0] = step;
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_ssgi.layout, 0, bound, writes);
		command.pushConstants(g_ssgi.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		if (mode == 5) command.dispatch(32, 18, 1); // (one workgroup a cell)
		else command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
		global(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		       vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	};
	// SVGF order: trace; temporal (variance-guided count, moments) -> tmp_a; a-trous step 1 -> the colour history;
	// a-trous 2, 4 (, 8 at level 3) ping-ponging tmp_b / tmp_a with the filtered variance in var_a / var_b; composite
	// (bilateral upsample, level 3 anti-flicker against gihist[prev]) -> out.
	// (Bindings a mode does not write still need a valid storage image.)
	if (emissive) {
		run(5, g_ssgi.tmp_a, g_ssgi.tmp_b, g_ssgi.raw, g_ssgi.tmp_b, 0.0f, g_ssgi.render); // the lights (reads only the game's)
		aux_out = &g_ssgi.emi_raw;                                                       // (the trace's emissive output)
	}
	run(0, g_ssgi.tmp_a, g_ssgi.tmp_b, g_ssgi.raw, g_ssgi.gbuf[cur], 0.0f, g_ssgi.trace);
	aux_in = &g_ssgi.moments[prev], aux_out = &g_ssgi.moments[cur];
	run(1, g_ssgi.raw, g_ssgi.history[prev], g_ssgi.tmp_a, g_ssgi.tmp_b, 0.0f, g_ssgi.trace);
	aux_in = &g_ssgi.moments[cur], aux_out = &g_ssgi.var_a;
	run(2, g_ssgi.tmp_a, g_ssgi.raw, g_ssgi.history[cur], g_ssgi.tmp_b, 1.0f, g_ssgi.trace);
	aux_in = &g_ssgi.var_a, aux_out = &g_ssgi.var_b;
	run(2, g_ssgi.history[cur], g_ssgi.raw, g_ssgi.tmp_b, g_ssgi.raw, 2.0f, g_ssgi.trace);
	aux_in = &g_ssgi.var_b, aux_out = &g_ssgi.var_a;
	run(2, g_ssgi.tmp_b, g_ssgi.raw, g_ssgi.tmp_a, g_ssgi.raw, 4.0f, g_ssgi.trace);
	const SsgiImage* final_gi = &g_ssgi.tmp_a;
	if (quality >= 3) {
		aux_in = &g_ssgi.var_a, aux_out = &g_ssgi.var_b;
		run(2, g_ssgi.tmp_a, g_ssgi.raw, g_ssgi.tmp_b, g_ssgi.raw, 8.0f, g_ssgi.trace);
		final_gi = &g_ssgi.tmp_b;
	}
	if (emissive) {
		// The emissive light's denoise, as the bounce's: temporal (its own count / moments) -> emi_tmp, a-trous 1 -> its
		// history, a-trous 2 -> emi_tmp (what the composite reads). var_a / var_b are free again here; the final GI
		// (tmp_a or tmp_b) is not written (the unused storage bindings point at emi_raw).
		aux_in = &g_ssgi.emi_mom[prev], aux_out = &g_ssgi.emi_mom[cur];
		run(1, g_ssgi.emi_raw, g_ssgi.emi_hist[prev], g_ssgi.emi_tmp, g_ssgi.emi_raw, 0.0f, g_ssgi.trace);
		aux_in = &g_ssgi.emi_mom[cur], aux_out = &g_ssgi.var_a;
		run(2, g_ssgi.emi_tmp, g_ssgi.emi_raw, g_ssgi.emi_hist[cur], g_ssgi.emi_raw, 1.0f, g_ssgi.trace);
		aux_in = &g_ssgi.var_a, aux_out = &g_ssgi.var_b;
		run(2, g_ssgi.emi_hist[cur], g_ssgi.emi_raw, g_ssgi.emi_tmp, g_ssgi.emi_raw, 2.0f, g_ssgi.trace);
	}
	aux_in = &g_ssgi.var_a, aux_out = &g_ssgi.var_b;
	run(4, *final_gi, g_ssgi.gihist[prev], g_ssgi.out, g_ssgi.gihist[cur], 0.0f, g_ssgi.render);
	if (reactive) {
		// (rtparticles) The reactive mask from the particle emission share -> g_ssgi.reactive, left for the upscaler.
		Transition(command, g_ssgi.reactive.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
		           vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader);
		run(6, *final_gi, g_ssgi.gihist[prev], g_ssgi.raw, g_ssgi.raw, 0.0f, g_ssgi.render);
		Transition(command, g_ssgi.reactive.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
		           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
		inputs.reactive.image  = g_ssgi.reactive.image.image;
		inputs.reactive.view   = g_ssgi.reactive.view;
		inputs.reactive.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
		inputs.reactive.format = vk::Format::eR8Unorm;
		inputs.reactive.extent = g_ssgi.render;
		inputs.reactive.usage  = g_ssgi.reactive.image.usage;
	}
	// The composite's writes before the upscaler reads.
	global(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite, vk::PipelineStageFlagBits2::eAllCommands,
	       vk::AccessFlagBits2::eMemoryRead);
	if (emissive && g_ssgi.readback != VK_NULL_HANDLE) {
		// The counters (hdr[0..7]) into this frame's slot of the host ring.
		vk::BufferCopy region {0, (g_ssgi.frame % 4) * SsgiStatWords * 4, SsgiStatWords * 4};
		command.copyBuffer(vk::Buffer(g_ssgi.lights), vk::Buffer(g_ssgi.readback), 1, &region);
		global(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
		       vk::AccessFlagBits2::eHostRead);
	}
	if (timed) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_ssgi.pool, entry * 2 + 1);
		++g_ssgi.written;
	}
	++g_ssgi.frame;
	g_ssgi.history_valid = true;
	inputs.color.image  = g_ssgi.out.image.image;
	inputs.color.view   = g_ssgi.out.view;
	inputs.color.layout = vk::ImageLayout::eGeneral;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = g_ssgi.out.image.usage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}
} // namespace

namespace {
// KYTY_GTAO=0|1|2|3 ("advanced ambient occlusion": off / Normal / Alta / Ultra; gtao.comp): GTAO ported from Intel
// XeGTAO (MIT, attribution in gtao.comp), recorded right before the upscaler after SSGI, with SSGI's inputs (depth,
// motion vectors, color, the upscaler's view_to_clip). Normals from depth; horizon search with cosine-weighted
// slices (1: half res, 2 slices x 8 steps; 2: full res, 3 x 12; 3: full res, 4 x 16); SSGI-style temporal
// (motion-vector reprojection, depth/normal disocclusion reset) + edge-aware a-trous (1, 2 / 1, 2, 4 / 1, 2, 4, 8);
// composite: a soft darkening multiplied into the color, faded with distance, sky skipped. KYTY_GTAO_STRENGTH (0.5),
// KYTY_GTAO_RADIUS (metres, 0.35: the same at every distance; KYTY_GTAO_MAXSCREEN 0.1 of the height clamps it only on
// close-ups), KYTY_GTAO_THICKNESS (metres, 0.6 x the radius), KYTY_GTAO_BENT (0: the extra multiplier on dark,
// ambient-dominated pixels, 0..1). Samples on depth texel centres; the a-trous kernel limited to ~the radius in metres.
// Off (the default): nothing created, nothing recorded.
//
// (specocc) KYTY_SPEC_OCCLUSION (strength 0..1, default 0.8 while KYTY_GTAO >= 1; 0 off): specular occlusion from the
// same horizon search (gtao.comp: the reflection lobe against the slices' visible arcs), taking off only the estimated
// environment-specular (IBL) share of the lit color, from the G-buffer material (KYTY_RT_REFLECTIONS' packed copy):
// KYTY_SPEC_OCCLUSION_ROUGH (default 2:0:8, L2.b0), _GLOSS (default 1:16:8g, L1.b2 inverted; 0 none; the rougher wins),
// _METAL (default none; e.g. the field refl9's survey logs), all "<layer>:<shift>:<bits>[g]". Not where the game's SSR
// hit (its captured guide, while KYTY_RT_REFLECTIONS runs), not on bright pixels / local peaks (KYTY_SPEC_OCCLUSION_
// HIGHLIGHT, display-referred, 0.8; _PEAK, ratio to the neighbourhood, 1.6), not on rough materials (_ROUGH_END, 0.7).
// Every 10 s a "GTAO: specular occlusion:" line (a 1/16 pixel sample).
struct GtaoPass {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             point      = nullptr;
	vk::Sampler             linear     = nullptr;
	bool                    failed     = false;
	vk::Extent2D            render {}, trace {};
	SsgiImage               raw, tmp_a, tmp_b, out;
	SsgiImage               history[2], gbuf[2], count[2];
	uint32_t                frame         = 0;
	int                     level         = 0;
	bool                    history_valid = false;
	vk::QueryPool           pool      = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms  = 0.0;
	uint32_t                samples = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
	VkImage       dummy_image    = VK_NULL_HANDLE;
	VmaAllocation dummy_mem      = nullptr;
	vk::ImageView dummy_view     = nullptr;
	bool          dummy_ready    = false;
	bool          normals_logged = false;
	// (specocc) parameters + stats (device local, 128 bytes), its stats copied to a 4-slot host ring each frame
	VkBuffer        so_dev = VK_NULL_HANDLE, so_host = VK_NULL_HANDLE;
	VmaAllocation   so_dev_mem = nullptr, so_host_mem = nullptr;
	const uint32_t* so_mapped  = nullptr;
	uint64_t        so_written = 0;
	std::array<double, 16> so_sum {};
	uint32_t        so_frames = 0;
	std::chrono::steady_clock::time_point so_last_log = std::chrono::steady_clock::now();
} g_gtao;

constexpr uint32_t GtaoBindings = 19;
constexpr uint32_t GtaoSoHeader = 64, GtaoSoStats = 64;

// "<layer>:<shift>:<bits>[g]" -> the shader's field code (layer | shift << 2 | bits << 7 | 4096 inverted); -1 none.
int32_t GtaoFieldEnv(const char* name, int32_t fallback) {
	const char* v = std::getenv(name);
	if (v == nullptr || *v == '\0') return fallback;
	int layer = 0, shift = 0, bits = 0;
	if (std::sscanf(v, "%d:%d:%d", &layer, &shift, &bits) != 3 || layer < 0 || layer > 3 || shift < 0 || bits < 1 || shift + bits > 32 ||
	    bits > 31)
		return -1;
	return layer | (shift << 2) | (bits << 7) | (std::strchr(v, 'g') != nullptr ? 4096 : 0);
}
struct GtaoSoConfig {
	float   strength = 0.8f, highlight = 0.8f, peak = 1.6f, rough_end = 0.7f;
	int32_t rough = 2 | (0 << 2) | (8 << 7), gloss = 1 | (16 << 2) | (8 << 7) | 4096, metal = -1;
};
const GtaoSoConfig& GtaoSo() {
	static const GtaoSoConfig cfg = [] {
		GtaoSoConfig c;
		c.strength  = SsgiEnv("KYTY_SPEC_OCCLUSION", 0.8f, 1.0f);
		c.highlight = std::max(SsgiEnv("KYTY_SPEC_OCCLUSION_HIGHLIGHT", 0.8f, 64.0f), 0.01f);
		c.peak      = std::max(SsgiEnv("KYTY_SPEC_OCCLUSION_PEAK", 1.6f, 16.0f), 1.05f);
		c.rough_end = std::clamp(SsgiEnv("KYTY_SPEC_OCCLUSION_ROUGH_END", 0.7f, 1.0f), 0.4f, 1.0f);
		c.rough     = GtaoFieldEnv("KYTY_SPEC_OCCLUSION_ROUGH", c.rough);
		c.gloss     = GtaoFieldEnv("KYTY_SPEC_OCCLUSION_GLOSS", c.gloss);
		c.metal     = GtaoFieldEnv("KYTY_SPEC_OCCLUSION_METAL", -1);
		if (c.rough < 0 && c.gloss >= 0) c.rough = c.gloss, c.gloss = -1;
		if (c.rough < 0) c.strength = 0.0f; // (no roughness field: no material, nothing to do)
		return c;
	}();
	return cfg;
}

// (specocc) The stats slot written 4 frames ago, summed; every 10 s one line.
void GtaoSoReadStats() {
	if (g_gtao.so_mapped == nullptr || g_graphics_ctx == nullptr) return;
	if (g_gtao.so_written >= 4) {
		const uint32_t slot = static_cast<uint32_t>(g_gtao.so_written % 4);
		(void)vmaInvalidateAllocation(g_graphics_ctx->allocator, g_gtao.so_host_mem, 0, VK_WHOLE_SIZE);
		const uint32_t* c = g_gtao.so_mapped + slot * (GtaoSoStats / 4);
		for (uint32_t i = 0; i < 16; ++i)
			g_gtao.so_sum[i] += c[i];
		++g_gtao.so_frames;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - g_gtao.so_last_log < std::chrono::seconds(10) || g_gtao.so_frames == 0) return;
	const auto&  t   = g_gtao.so_sum;
	const double n   = std::max(t[0], 1.0), mat = std::max(t[1], 1.0);
	const auto   pct = [&](double v, double of) { return 100.0 * v / std::max(of, 1.0); };
	std::printf("GTAO: specular occlusion: last %.0f s, %u frames: material %.1f%% of the pixels (water %.2f%%), occluded "
	            "(spec visibility < 0.8) %.1f%% of those, reduced (> 5%%) %.2f%% of the pixels, mean reduction there %.1f%%; "
	            "skipped: the game's SSR hit %.1f%%, highlights %.2f%%, not the G-buffer's surface %.1f%% (strength %g)\n",
	            std::chrono::duration<double>(now - g_gtao.so_last_log).count(), g_gtao.so_frames, pct(t[1], n), pct(t[7], n),
	            pct(t[6], mat), pct(t[2], n), t[2] > 0 ? t[3] / 10.0 / t[2] : 0.0, pct(t[4], mat), pct(t[5], mat), pct(t[8], n),
	            GtaoSo().strength);
	std::fflush(stdout);
	g_gtao.so_sum.fill(0.0);
	g_gtao.so_frames   = 0;
	g_gtao.so_last_log = now;
}

int GtaoLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_GTAO");
		const int   l     = value != nullptr ? std::clamp(std::atoi(value), 0, 3) : 0;
		if (l > 0) {
			std::printf("GTAO: level %d\n", l);
			std::fflush(stdout);
		}
		return l;
	}();
	return level;
}

struct GtaoPush : SsgiPush {
	float rot[8]; // gtao.comp's rot[2]
};
static_assert(sizeof(GtaoPush) == 128);

std::array<SsgiImage*, 10> GtaoAll() {
	return {&g_gtao.raw,     &g_gtao.tmp_a,   &g_gtao.tmp_b,    &g_gtao.out,      &g_gtao.history[0],
	        &g_gtao.history[1], &g_gtao.gbuf[0], &g_gtao.gbuf[1], &g_gtao.count[0], &g_gtao.count[1]};
}

bool GtaoEnsure(GraphicContext& graphics, vk::Extent2D render, int level) {
	const auto device = graphics.device;
	if (g_gtao.pipeline == nullptr) {
		if (g_gtao.failed) return false;
		g_gtao.failed = true;
		std::array<vk::DescriptorSetLayoutBinding, GtaoBindings> bindings {};
		for (uint32_t i = 0; i < GtaoBindings; ++i) {
			bindings[i].binding         = i;
			bindings[i].descriptorType  = i >= 17 ? vk::DescriptorType::eStorageBuffer
			                              : i < 7 || (i >= 9 && i != 10) ? vk::DescriptorType::eCombinedImageSampler
			                                                             : vk::DescriptorType::eStorageImage;
			bindings[i].descriptorCount = 1;
			bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
		}
		vk::DescriptorSetLayoutCreateInfo set_info {};
		set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		set_info.bindingCount = static_cast<uint32_t>(bindings.size());
		set_info.pBindings    = bindings.data();
		if (device.createDescriptorSetLayout(&set_info, nullptr, &g_gtao.set_layout) != vk::Result::eSuccess) return false;
		vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(GtaoPush)};
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.setLayoutCount         = 1;
		layout_info.pSetLayouts            = &g_gtao.set_layout;
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges    = &push;
		if (device.createPipelineLayout(&layout_info, nullptr, &g_gtao.layout) != vk::Result::eSuccess) return false;
		vk::ShaderModuleCreateInfo module_info {};
		module_info.codeSize    = sizeof(GTAO_SPV);
		module_info.pCode       = GTAO_SPV;
		vk::ShaderModule module = nullptr;
		if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
		pipeline_info.stage.module = module;
		pipeline_info.stage.pName  = "main";
		pipeline_info.layout       = g_gtao.layout;
		const auto created = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_gtao.pipeline);
		device.destroyShaderModule(module, nullptr);
		if (created != vk::Result::eSuccess) {
			g_gtao.pipeline = nullptr;
			return false;
		}
		for (int i = 0; i < 2; ++i) {
			vk::SamplerCreateInfo sampler {};
			sampler.magFilter    = i == 0 ? vk::Filter::eNearest : vk::Filter::eLinear;
			sampler.minFilter    = sampler.magFilter;
			sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
			sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
			sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
			if (device.createSampler(&sampler, nullptr, i == 0 ? &g_gtao.point : &g_gtao.linear) != vk::Result::eSuccess) return false;
		}
		vk::QueryPoolCreateInfo query {};
		query.queryType  = vk::QueryType::eTimestamp;
		query.queryCount = 16;
		if (device.createQueryPool(&query, nullptr, &g_gtao.pool) != vk::Result::eSuccess) g_gtao.pool = nullptr;
		g_gtao.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
		{ // a 1x1 R32_UINT stand-in for the G-buffer normal layer when there is no copy (binding 11)
			VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
			create.imageType   = VK_IMAGE_TYPE_2D;
			create.format      = VK_FORMAT_R32_UINT;
			create.extent      = {1, 1, 1};
			create.mipLevels   = 1;
			create.arrayLayers = 1;
			create.samples     = VK_SAMPLE_COUNT_1_BIT;
			create.tiling      = VK_IMAGE_TILING_OPTIMAL;
			create.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
			VmaAllocationCreateInfo allocate {};
			allocate.usage = VMA_MEMORY_USAGE_AUTO;
			if (vmaCreateImage(graphics.allocator, &create, &allocate, &g_gtao.dummy_image, &g_gtao.dummy_mem, nullptr) != VK_SUCCESS)
				return false;
			vk::ImageViewCreateInfo view_info {};
			view_info.image            = vk::Image(g_gtao.dummy_image);
			view_info.viewType         = vk::ImageViewType::e2D;
			view_info.format           = vk::Format::eR32Uint;
			view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
			if (device.createImageView(&view_info, nullptr, &g_gtao.dummy_view) != vk::Result::eSuccess) return false;
		}
		{ // (specocc) the parameters + stats buffer and its host ring
			VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			buffer.size  = GtaoSoHeader + GtaoSoStats;
			buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
			VmaAllocationCreateInfo dev {};
			dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
			if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_gtao.so_dev, &g_gtao.so_dev_mem, nullptr) != VK_SUCCESS) return false;
			buffer.size  = GtaoSoStats * 4;
			buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			VmaAllocationCreateInfo host {};
			host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
			host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
			VmaAllocationInfo mapped {};
			if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_gtao.so_host, &g_gtao.so_host_mem, &mapped) == VK_SUCCESS)
				g_gtao.so_mapped = static_cast<const uint32_t*>(mapped.pMappedData);
			else
				g_gtao.so_host = VK_NULL_HANDLE; // (no stats line; the effect still runs)
		}
		g_gtao.failed    = false;
	}
	if (g_gtao.out.image.image != nullptr && g_gtao.render == render && g_gtao.level == level) return true;
	const vk::Extent2D trace = level >= 2 ? render : vk::Extent2D {std::max(1u, (render.width + 1) / 2), std::max(1u, (render.height + 1) / 2)};
	for (auto* img : GtaoAll())
		SsgiFreeImage(graphics, *img);
	bool ok = SsgiMakeImage(graphics, g_gtao.out, render);
	for (auto* img : {&g_gtao.raw, &g_gtao.tmp_a, &g_gtao.tmp_b, &g_gtao.history[0], &g_gtao.history[1], &g_gtao.gbuf[0], &g_gtao.gbuf[1],
	                  &g_gtao.count[0], &g_gtao.count[1]})
		ok = ok && SsgiMakeImage(graphics, *img, trace);
	if (!ok) {
		for (auto* img : GtaoAll())
			SsgiFreeImage(graphics, *img);
		g_gtao.failed = true;
		return false;
	}
	g_gtao.render        = render;
	g_gtao.level         = level;
	g_gtao.trace         = trace;
	g_gtao.history_valid = false;
	return true;
}

void GtaoReadTiming() {
	if (g_gtao.pool == nullptr || g_graphics_ctx == nullptr) return;
	while (g_gtao.read + 4 <= g_gtao.written) {
		const auto entry = static_cast<uint32_t>(g_gtao.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (g_graphics_ctx->device.getQueryPoolResults(g_gtao.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                               vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		g_gtao.sum_ms += static_cast<double>(stamps[1] - stamps[0]) * g_gtao.period_ns / 1e6;
		++g_gtao.samples;
		++g_gtao.read;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - g_gtao.last_log >= std::chrono::seconds(5) && g_gtao.samples > 0) {
		std::printf("GTAO: GPU %.3f ms (mean of %u frames)\n", g_gtao.sum_ms / g_gtao.samples, g_gtao.samples);
		std::fflush(stdout);
		g_gtao.sum_ms   = 0.0;
		g_gtao.samples  = 0;
		g_gtao.last_log = now;
	}
}

bool GtaoApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int level = GtaoLevel();
	if (level == 0) return false; // off: nothing created or recorded
	if (HwAoReplacesGtao()) return false; // (KYTY_HWRT_AO: the ray-traced AO in its place, hwrt-gi.cpp)
	if (g_graphics_ctx == nullptr) return false;
	if (VideoOut::MoviePlaying()) {
		g_gtao.history_valid = false;
		return false;
	}
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || inputs.motion.view == nullptr || color.extent.width < 16 ||
	    color.extent.height < 16)
		return false;
	if (!GtaoEnsure(*g_graphics_ctx, color.extent, level)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("GTAO: off (could not create its pipeline or images)\n");
			std::fflush(stdout);
		}
		return false;
	}
	// (10-24 ao: subtle defaults; strength 0.6 -> 0.5, radius 1.0 -> 0.6 m, clamped to 10% of the screen height)
	const float strength  = SsgiEnv("KYTY_GTAO_STRENGTH", 0.5f, 1.0f);
	// (gtaofar) radius 0.6 -> 0.35 m. The screen clamp below (max_r of the height) is what made the near field look
	// right: it holds the world radius at max_r * 2 z / proj_y, i.e. ~0.3 m at 3 m from the camera with the game's
	// projection, while everything past ~6 m got the full 0.6 m (so a staircase or a far wall darkened over whole
	// areas and brightened as the player walked up to it). 0.35 m is that near-field radius held at every distance (the
	// clamp now bites only nearer than ~3.5 m: close-ups). The thin-object thickness keeps its ratio to the radius.
	const float radius    = std::max(SsgiEnv("KYTY_GTAO_RADIUS", 0.35f, 8.0f), 0.05f);
	const float thickness = std::max(SsgiEnv("KYTY_GTAO_THICKNESS", 0.6f * radius, 4.0f), 0.02f);
	const float max_r     = std::max(SsgiEnv("KYTY_GTAO_MAXSCREEN", 0.1f, 0.5f), 0.01f);
	const float bent     = SsgiEnv("KYTY_GTAO_BENT", 0.0f, 1.0f);
	const auto& m        = inputs.view_to_clip;
	GtaoPush    push {};
	const bool  have_proj = m[0] != 0.0f && m[5] != 0.0f && (m[11] != 0.0f || m[15] != 0.0f);
	if (have_proj) {
		push.proj[0] = m[0], push.proj[1] = m[5], push.proj[2] = m[8], push.proj[3] = m[9];
		push.depth_map[0] = m[10], push.depth_map[1] = m[14], push.depth_map[2] = m[11], push.depth_map[3] = m[15];
	} else {
		const float f  = 1.0f / std::tan(0.5f * 1.0472f);
		const float ar = static_cast<float>(color.extent.width) / static_cast<float>(color.extent.height);
		push.proj[0] = f / ar, push.proj[1] = f, push.proj[2] = 0.0f, push.proj[3] = 0.0f;
		push.depth_map[0] = 0.0f, push.depth_map[1] = inputs.near_plane, push.depth_map[2] = 1.0f, push.depth_map[3] = 0.0f;
	}
	push.motion[0] = inputs.motion_scale_x, push.motion[1] = inputs.motion_scale_y;
	push.motion[2] = static_cast<float>(color.extent.width), push.motion[3] = static_cast<float>(color.extent.height);
	push.trace[0] = static_cast<float>(g_gtao.trace.width), push.trace[1] = static_cast<float>(g_gtao.trace.height);
	push.trace[2] = strength, push.trace[3] = radius;
	push.extra[1] = bent;
	push.extra[2] = thickness;
	push.extra[3] = max_r;
	// The G-buffer normal (as SHINE reads it: layer 3, oct 8:8, world space, the G-buffer camera's rotation).
	RtReflGbuffer gbuf;
	const bool    have_gbuf = have_proj && RtReflGetGbuffer(gbuf) && gbuf.extent == color.extent && gbuf.views[3] != nullptr;
	RtReflCamera  camera;
	if (have_gbuf) RtReflGetCamera(m, camera);
	static const bool use_gbuf = [] {
		const char* v = std::getenv("KYTY_GTAO_GBUFFER");
		return v == nullptr || *v != '0';
	}();
	static const bool flip_y = [] {
		const char* v = std::getenv("KYTY_GTAO_FLIPY");
		return v == nullptr || *v == '\0' || *v != '0';
	}();
	const bool normals = use_gbuf && have_gbuf && camera.rows;
	int32_t    nflags  = 0;
	if (normals) {
		const auto& r = camera.view_rows; // 3 rows x 4
		push.rot[0] = r[0], push.rot[1] = r[1], push.rot[2] = r[2], push.rot[3] = r[4];
		push.rot[4] = r[5], push.rot[5] = r[6], push.rot[6] = r[8], push.rot[7] = r[9];
		// row 2 = cross(row 0, row 1) * sign: the sign from the real row 2
		const float cx = r[1] * r[6] - r[2] * r[5], cy = r[2] * r[4] - r[0] * r[6], cz = r[0] * r[5] - r[1] * r[4];
		const bool  neg = cx * r[8] + cy * r[9] + cz * r[10] < 0.0f;
		nflags = 1 | (flip_y ? 2 : 0) | (neg ? 4 : 0);
	}
	if (normals != g_gtao.normals_logged) {
		g_gtao.normals_logged = normals;
		std::printf("GTAO: normals from %s\n", normals ? "the G-buffer (layer 3, oct 8:8, world)" : "depth");
		std::fflush(stdout);
	}
	{
		static bool said = false;
		if (!said) {
			said = true;
			// (gtaofar) where the screen clamp stops shrinking the radius (w = |z C + D|, C = +-1 for the game's)
			const float py      = std::fabs(push.proj[1]);
			const float c_abs   = std::fabs(push.depth_map[2]);
			const float z_clamp = c_abs > 1e-6f ? (radius * py * 0.5f / max_r - std::fabs(push.depth_map[3])) / c_abs : 0.0f;
			std::printf("GTAO: on (level %d, %ux%u, strength %g, radius %g m (whole beyond %.1f m, vertical FOV %.1f deg), "
			            "thickness %g m, max %g of the height, bent %g)%s\n",
			            level, g_gtao.trace.width, g_gtao.trace.height, strength, radius, z_clamp,
			            py > 0.0f ? 2.0f * std::atan(1.0f / py) * 57.29578f : 0.0f, thickness, max_r, bent,
			            have_proj ? "" : " (no projection: 60 deg FOV assumed)");
			std::fflush(stdout);
		}
	}
	// (specocc) What the specular occlusion reads: the material layers (with the G-buffer normal: the same copy), the
	// lighting's depth, the game's SSR guide, the exposure.
	const auto& so_cfg  = GtaoSo();
	const bool  so_mat  = normals && so_cfg.strength > 0.0f && gbuf.views[0] != nullptr && gbuf.views[1] != nullptr && gbuf.views[2] != nullptr;
	const bool  so_dep  = so_mat && gbuf.depth != nullptr;
	RtReflGuide guide;
	const bool  so_gd   = so_mat && RtReflGetGuide(guide) && guide.view != nullptr;
	const bool  so_exp  = inputs.exposure_buffer != nullptr && inputs.exposure_range >= sizeof(float);
	const bool  so_stat = g_gtao.so_host != VK_NULL_HANDLE;
	{
		static int said = -1;
		const int  now  = so_cfg.strength <= 0.0f ? 0 : !so_mat ? 1 : 2 + (so_gd ? 1 : 0) + (so_dep ? 2 : 0);
		if (now != said) {
			said = now;
			if (now == 0) std::printf("GTAO: specular occlusion off (KYTY_SPEC_OCCLUSION=0 or no roughness field)\n");
			else if (now == 1) std::printf("GTAO: specular occlusion waiting (no G-buffer material copy this frame)\n");
			else
				std::printf("GTAO: specular occlusion on (strength %g; roughness %d, gloss %d, metal %d; the game's SSR guide %s, the "
				            "lighting's depth %s, exposure %s)\n",
				            so_cfg.strength, so_cfg.rough, so_cfg.gloss, so_cfg.metal, so_gd ? "yes" : "no (KYTY_RT_REFLECTIONS off)",
				            so_dep ? "yes" : "no", so_exp ? "the game's" : "1.0");
			std::fflush(stdout);
		}
	}
	GtaoSoReadStats();
	{
		std::array<uint32_t, GtaoSoHeader / 4> h {};
		const auto fl = [&](uint32_t i, float v) { std::memcpy(&h[i], &v, sizeof(float)); };
		h[0] = static_cast<uint32_t>(so_cfg.rough), h[1] = static_cast<uint32_t>(so_cfg.gloss), h[2] = static_cast<uint32_t>(so_cfg.metal);
		h[3] = (so_mat ? 1u | 8u : 0u) | (so_gd ? 2u : 0u) | (so_dep ? 4u : 0u) | (so_mat && so_stat ? 16u : 0u);
		h[4] = so_exp ? inputs.exposure_index : 12u;
		fl(8, so_cfg.strength), fl(9, 0.1f), fl(10, 0.6f), fl(11, so_cfg.rough_end); // (guide range: RT reflections' defaults)
		fl(12, 1.0f), fl(13, so_cfg.highlight), fl(14, so_cfg.peak);
		{ // (the last frame's shader atomics and copy before these writes)
			vk::MemoryBarrier2 mb {};
			mb.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
			mb.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
			mb.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
			mb.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
			vk::DependencyInfo dependency {};
			dependency.memoryBarrierCount = 1;
			dependency.pMemoryBarriers    = &mb;
			command.pipelineBarrier2(dependency);
		}
		command.updateBuffer(vk::Buffer(g_gtao.so_dev), 0, sizeof(h), h.data());
		command.fillBuffer(vk::Buffer(g_gtao.so_dev), GtaoSoHeader, GtaoSoStats, 0);
	}
	GtaoReadTiming();
	const auto entry = static_cast<uint32_t>(g_gtao.written % 8);
	const bool timed = g_gtao.pool != nullptr;
	if (timed) {
		command.resetQueryPool(g_gtao.pool, entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_gtao.pool, entry * 2);
	}
	const auto global = [&](vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src, vk::PipelineStageFlags2 dst_stage,
	                        vk::AccessFlags2 dst) {
		vk::MemoryBarrier2 barrier {};
		barrier.srcStageMask  = src_stage;
		barrier.srcAccessMask = src;
		barrier.dstStageMask  = dst_stage;
		barrier.dstAccessMask = dst;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &barrier;
		command.pipelineBarrier2(dependency);
	};
	for (auto* img : GtaoAll())
		if (img->image.state.layout != vk::ImageLayout::eGeneral)
			Transition(command, img->image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
			           vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader);
	global(vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	       vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	const uint32_t cur  = g_gtao.frame & 1u;
	const uint32_t prev = cur ^ 1u;
	push.misc[1] = static_cast<int32_t>(g_gtao.frame);
	push.misc[2] = level;
	push.misc[3] = (inputs.reset ? 1 : 0) | (g_gtao.history_valid ? 2 : 0) | (nflags << 2);
	if (!g_gtao.dummy_ready) {
		g_gtao.dummy_ready = true;
		vk::ImageMemoryBarrier2 barrier {};
		barrier.srcStageMask     = vk::PipelineStageFlagBits2::eNone;
		barrier.dstStageMask     = vk::PipelineStageFlagBits2::eTransfer;
		barrier.dstAccessMask    = vk::AccessFlagBits2::eTransferWrite;
		barrier.oldLayout        = vk::ImageLayout::eUndefined;
		barrier.newLayout        = vk::ImageLayout::eGeneral;
		barrier.image            = vk::Image(g_gtao.dummy_image);
		barrier.subresourceRange = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers    = &barrier;
		command.pipelineBarrier2(dependency);
		vk::ClearColorValue             zero {};
		zero.uint32 = std::array<uint32_t, 4> {0, 0, 0, 0};
		const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		command.clearColorImage(vk::Image(g_gtao.dummy_image), vk::ImageLayout::eGeneral, &zero, 1, &whole);
		global(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
		       vk::AccessFlagBits2::eShaderRead);
	}
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_gtao.pipeline);
	const auto run = [&](int mode, const SsgiImage& a, const SsgiImage& b, const SsgiImage& out, const SsgiImage& gout,
	                     const SsgiImage& aux_in, const SsgiImage& aux_out, float step, vk::Extent2D size) {
		const vk::Sampler       lin = g_gtao.linear, pt = g_gtao.point;
		vk::DescriptorImageInfo infos[GtaoBindings] {};
		infos[0]  = {lin, color.view, color.layout};
		infos[1]  = {pt, inputs.depth.view, inputs.depth.layout};
		infos[2]  = {pt, inputs.motion.view, inputs.motion.layout};
		infos[3]  = {lin, a.view, vk::ImageLayout::eGeneral};
		infos[4]  = {lin, b.view, vk::ImageLayout::eGeneral};
		infos[5]  = {pt, g_gtao.gbuf[cur].view, vk::ImageLayout::eGeneral};
		infos[6]  = {pt, g_gtao.gbuf[prev].view, vk::ImageLayout::eGeneral};
		infos[7]  = {nullptr, out.view, vk::ImageLayout::eGeneral};
		infos[8]  = {nullptr, gout.view, vk::ImageLayout::eGeneral};
		infos[9]  = {pt, aux_in.view, vk::ImageLayout::eGeneral};
		infos[10] = {nullptr, aux_out.view, vk::ImageLayout::eGeneral};
		infos[11] = normals ? vk::DescriptorImageInfo {pt, gbuf.views[3], vk::ImageLayout::eShaderReadOnlyOptimal}
		                    : vk::DescriptorImageInfo {pt, g_gtao.dummy_view, vk::ImageLayout::eGeneral};
		// (specocc) the material layers / lighting depth / guide, or stand-ins the flags tell the shader not to read
		for (uint32_t l = 0; l < 3; ++l)
			infos[12 + l] = so_mat ? vk::DescriptorImageInfo {pt, gbuf.views[l], vk::ImageLayout::eShaderReadOnlyOptimal}
			                       : vk::DescriptorImageInfo {pt, g_gtao.dummy_view, vk::ImageLayout::eGeneral};
		infos[15] = so_dep ? vk::DescriptorImageInfo {pt, gbuf.depth, gbuf.depth_layout} : infos[1];
		infos[16] = so_gd ? vk::DescriptorImageInfo {lin, guide.view, vk::ImageLayout::eShaderReadOnlyOptimal}
		                  : vk::DescriptorImageInfo {pt, g_gtao.history[prev].view, vk::ImageLayout::eGeneral};
		const vk::DescriptorBufferInfo so_buf {vk::Buffer(g_gtao.so_dev), 0, GtaoSoHeader + GtaoSoStats};
		const vk::DescriptorBufferInfo so_ex  = so_exp ? vk::DescriptorBufferInfo {inputs.exposure_buffer, inputs.exposure_base, inputs.exposure_range}
		                                               : so_buf;
		vk::WriteDescriptorSet writes[GtaoBindings] {};
		for (uint32_t i = 0; i < GtaoBindings; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			if (i >= 17) {
				writes[i].descriptorType = vk::DescriptorType::eStorageBuffer;
				writes[i].pBufferInfo    = i == 17 ? &so_buf : &so_ex;
				continue;
			}
			writes[i].descriptorType  = i < 7 || (i >= 9 && i != 10) ? vk::DescriptorType::eCombinedImageSampler : vk::DescriptorType::eStorageImage;
			writes[i].pImageInfo      = &infos[i];
		}
		push.misc[0]  = mode;
		push.extra[0] = step;
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_gtao.layout, 0, GtaoBindings, writes);
		command.pushConstants(g_gtao.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
		global(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		       vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	};
	// (Bindings a mode does not write still need a valid storage image: tmp_b is the scratch target.)
	auto& g = g_gtao;
	run(0, g.tmp_a, g.tmp_b, g.raw, g.gbuf[cur], g.count[prev], g.tmp_b, 0.0f, g.trace);                     // GTAO
	run(1, g.raw, g.history[prev], g.history[cur], g.tmp_b, g.count[prev], g.count[cur], 0.0f, g.trace);      // temporal
	const SsgiImage* src = &g.history[cur];
	const SsgiImage* ping[2] = {&g.tmp_a, &g.raw};
	const int        passes  = level >= 3 ? 4 : level >= 2 ? 3 : 2;
	for (int i = 0; i < passes; ++i) {
		const SsgiImage* dst = ping[i & 1];
		run(2, *src, g.tmp_b, *dst, g.tmp_b, g.count[cur], g.tmp_b, static_cast<float>(1 << i), g.trace);
		src = dst;
	}
	// (shadows-1025) KYTY_GTAO_FLOOR: the composite's darkest multiplier (gtao.comp Composite: 0.7 = at most 30 % off;
	// "Sombras: profundidad" lowers it to 0.55 / 0.45 for deeper creases under tables, carts and feet). Rides in the
	// composite's unused a-trous step slot.
	static const float gtao_floor = std::clamp(SsgiEnv("KYTY_GTAO_FLOOR", 0.7f, 1.0f), 0.3f, 0.95f);
	run(4, *src, g.tmp_b, g.out, g.tmp_b, g.count[cur], g.tmp_b, gtao_floor, g.render);                      // composite
	global(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite, vk::PipelineStageFlagBits2::eAllCommands,
	       vk::AccessFlagBits2::eMemoryRead);
	if (so_stat) { // (specocc) this frame's stats into the host ring (read 4 frames later)
		const vk::BufferCopy copy {GtaoSoHeader, (g_gtao.so_written % 4) * GtaoSoStats, GtaoSoStats};
		command.copyBuffer(vk::Buffer(g_gtao.so_dev), vk::Buffer(g_gtao.so_host), 1, &copy);
		global(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
		       vk::AccessFlagBits2::eHostRead);
		++g_gtao.so_written;
	}
	if (timed) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_gtao.pool, entry * 2 + 1);
		++g_gtao.written;
	}
	++g_gtao.frame;
	g_gtao.history_valid = true;
	inputs.color.image  = g_gtao.out.image.image;
	inputs.color.view   = g_gtao.out.view;
	inputs.color.layout = vk::ImageLayout::eGeneral;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = g_gtao.out.image.usage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}
} // namespace

namespace {
bool Inverse4(const std::array<float, 16>& m, std::array<double, 16>& inv); // (defined with the contact shadows)

// KYTY_SSR=0|1|2|3 ("Reflejos RT": off / Normal / Alta / Ultra; ssr.comp): screen-space reflections, mainly for water,
// puddles and wet floors, recorded right before the upscaler after SSGI and GTAO, with their inputs (depth, motion
// vectors, the pre-upscale color, the upscaler's view_to_clip). 1: half-res trace, 24 steps + 4 binary refinements,
// a-trous 1, 2; 2: full res, 40 + 6, a-trous 1, 2, 4; 3: full res, 64 + 8, longer rays, a-trous 1, 2, 4, 8.
// Rough surfaces: GGX importance-sampled jitter (IGN per frame); temporal accumulation reprojecting by the surface's and
// the hit's motion; a-trous by roughness / depth / normal; composite with Schlick Fresnel x a smoothness mask, faded at
// the screen edges and by ray length; no hit = the game's own image (its own reflections stay).
// Roughness source: KYTY_SSR_GBUF=<render target slot>.<channel r|g|b|a>[g] (g: the channel is gloss) names a channel
// of the G-buffer draws' color targets (logged as "SSR: G-buffer draw ..."); unset, or the target not usable this
// frame: a heuristic mask (depth-derived normals facing up, low local variance, darker than the surroundings).
// KYTY_SSR_STRENGTH (1.0, 0..2), KYTY_SSR_THICKNESS (view units, 0.25), KYTY_SSR_UP (world up axis for the heuristic:
// y (default), z or x; "-y" etc. for the negative one). Off (the default): nothing created, nothing recorded, no hooks.
//
// Packed G-buffer (Demon's Souls: RT0..RT3 R32_UINT, logged as one image's layers 0..3 at the same address, read by the
// deferred lighting CS as one slot): bound as R32_UINT views (bindings 14..17). The first frames run a calibration (mode
// 5): every layer x 5 normal layouts (oct 16:16, oct 12:12 low / high, oct 11:11, 10:10:10) is scored against the
// depth-derived normal in view and world space, and the 16 byte fields' statistics are gathered; the best layout (mean
// |cos| >= 0.85) becomes the normal source (logged "SSR: G-buffer normal ..."), and the up direction is no longer
// needed. Roughness / metal: KYTY_SSR_ROUGH / KYTY_SSR_METAL = <layer>:<shift>:<bits>[g] (g: gloss / inverted); unset:
// the 8-bit field left in the normal's word when its layout leaves one (oct 12:12 -> the other byte), else the
// heuristic mask (logged as a GUESS). KYTY_SSR_NORMAL = <layer>:<layout>[w][n] (w world space, n negated) overrides the
// calibration. Composite: a convex mix (reflection replaces a share of the color), Schlick-with-roughness Fresnel, f0
// 0.04 (0.6 metal), the reflection's brightness capped at 4x the surface's, the sky never a hit.
struct SsrPass {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             point      = nullptr;
	vk::Sampler             linear     = nullptr;
	bool                    failed     = false;
	vk::Extent2D            render {}, trace {};
	SsgiImage               raw, tmp_a, tmp_b, out, mat, hit;
	SsgiImage               history[2], gbuf[2];
	uint32_t                frame         = 0;
	int                     level         = 0;
	bool                    history_valid = false;
	int                     source        = -2; // the roughness source last logged (-1 heuristic, else slot*4+channel)
	vk::QueryPool           pool      = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms  = 0.0;
	uint32_t                samples = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
	// Packed G-buffer: a 1x1 R32_UINT stand-in for unbound layers, the calibration / camera buffer, the chosen decode.
	VkImage                 dummy_image = VK_NULL_HANDLE;
	VmaAllocation           dummy_mem   = nullptr;
	vk::ImageView           dummy_view  = nullptr;
	bool                    dummy_ready = false;
	VkBuffer                calib       = VK_NULL_HANDLE;
	VmaAllocation           calib_mem   = nullptr;
	const int32_t*          calib_mapped = nullptr;
	uint32_t                calib_runs  = 0, calib_wait = 0;
	bool                    calib_done  = false;
	int32_t                 dec_normal = -1, dec_rough = -1, dec_metal = -1;
	// The G-buffer's copy (SsrCaptureGbuffer): 4 layers R32_UINT, one view per layer, the flip it was taken in.
	VkImage                 copy_image = VK_NULL_HANDLE;
	VmaAllocation           copy_mem   = nullptr;
	vk::ImageView           copy_views[4] {};
	vk::Extent2D            copy_extent {};
	vk::ImageLayout         copy_layout = vk::ImageLayout::eUndefined;
	std::atomic<uint64_t>   copy_flip {~0ull};
	bool                    copy_failed = false;
	uint32_t                packed_logged = ~0u;
	std::atomic<uint64_t>   copies {0};       // copies taken (KYTY_RT_REFLECTIONS' periodic line)
	vk::Extent2D            resize_to {};     // a new G-buffer size seen on consecutive attempts (resize after 30)
	uint32_t                resize_seen = 0;
	struct RetiredCopy {
		VkImage       image = VK_NULL_HANDLE;
		VmaAllocation mem   = nullptr;
		vk::ImageView views[4] {};
		uint64_t      flip  = 0;
	};
	std::vector<RetiredCopy> retired_copies; // (frames in flight may still read them: destroyed 8 flips later)
	// KYTY_RT_REFLECTIONS: the depth as the deferred lighting reads it (before water and other forward passes draw):
	// where the final depth is nearer, something was drawn after the G-buffer (water). Taken with the G-buffer copy.
	VkImage                 depth_image = VK_NULL_HANDLE;
	VmaAllocation           depth_mem   = nullptr;
	vk::ImageView           depth_view  = nullptr;
	vk::Extent2D            depth_extent {};
	vk::Format              depth_format = vk::Format::eUndefined;
	vk::ImageLayout         depth_layout = vk::ImageLayout::eUndefined;
	uint64_t                depth_flip   = ~0ull;
	bool                    depth_failed = false;
} g_ssr;
constexpr uint32_t SsrCalibRuns  = 24;  // calibration frames (one every 4th)
constexpr uint32_t SsrCalibInts  = 41 + 16 * 3;
constexpr uint32_t SsrCalibBytes = 48 + SsrCalibInts * 4;

struct SsrPush {
	float   proj[4];
	float   depth_map[4];
	float   motion[4];
	float   trace[4];
	int32_t misc[4];
	float   extra[4];
	float   up[4];
	int32_t dec[4];
};
static_assert(sizeof(SsrPush) == 128);

constexpr uint32_t SsrBindings = 19;
vk::DescriptorType SsrType(uint32_t i) {
	return i == 18 ? vk::DescriptorType::eStorageBuffer
	               : (i == 7 || i == 8 || i == 10 || i == 13) ? vk::DescriptorType::eStorageImage
	                                                          : vk::DescriptorType::eCombinedImageSampler;
}
// "<layer>:<shift>:<bits>[g]" -> the shader's field code (-1: unset or malformed).
int32_t SsrFieldEnv(const char* name) {
	const char* value = std::getenv(name);
	if (value == nullptr) return -1;
	int layer = -1, shift = -1, bits = -1;
	char flag = 0;
	if (std::sscanf(value, "%d:%d:%d%c", &layer, &shift, &bits, &flag) < 3) return -1;
	if (layer < 0 || layer > 3 || shift < 0 || bits < 1 || bits > 16 || shift + bits > 32) return -1;
	return layer | (shift << 2) | (bits << 7) | (flag == 'g' ? 4096 : 0);
}
// "<layer>:<layout>[w][n]" -> the shader's normal code (-1: unset or malformed).
int32_t SsrNormalEnv() {
	const char* value = std::getenv("KYTY_SSR_NORMAL");
	if (value == nullptr) return -1;
	int layer = -1, layout = -1;
	if (std::sscanf(value, "%d:%d", &layer, &layout) < 2 || layer < 0 || layer > 3 || layout < 0 || layout > 4) return -1;
	const std::string v = value;
	return layer | (layout << 2) | (v.find('w') != std::string::npos ? 32 : 0) | (v.find('n') != std::string::npos ? 64 : 0);
}

int SsrLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_SSR");
		return value != nullptr ? std::clamp(std::atoi(value), 0, 3) : 0;
	}();
	return level;
}

// KYTY_SSR_GBUF: slot, channel, gloss (slot -1: unset or malformed).
struct SsrGbufChoice {
	int  slot = -1, channel = 0;
	bool gloss = false;
};
const SsrGbufChoice& SsrChoice() {
	static const SsrGbufChoice choice = [] {
		SsrGbufChoice c;
		const char*   value = std::getenv("KYTY_SSR_GBUF");
		if (value == nullptr || value[0] < '0' || value[0] > '7' || value[1] != '.') return c;
		const char ch = value[2];
		const int  channel = ch == 'r' || ch == 'x' ? 0 : ch == 'g' || ch == 'y' ? 1 : ch == 'b' || ch == 'z' ? 2 : ch == 'a' || ch == 'w' ? 3 : -1;
		if (channel < 0) return c;
		c.slot    = value[0] - '0';
		c.channel = channel;
		c.gloss   = value[3] == 'g';
		return c;
	}();
	return choice;
}

const char* SsrFormatName(vk::Format f) {
	switch (f) {
		case vk::Format::eR8G8B8A8Unorm: return "RGBA8_UNORM";
		case vk::Format::eR8G8B8A8Srgb: return "RGBA8_SRGB";
		case vk::Format::eB8G8R8A8Unorm: return "BGRA8_UNORM";
		case vk::Format::eA2B10G10R10UnormPack32: return "A2B10G10R10_UNORM";
		case vk::Format::eA2R10G10B10UnormPack32: return "A2R10G10B10_UNORM";
		case vk::Format::eR16G16B16A16Sfloat: return "RGBA16F";
		case vk::Format::eR16G16B16A16Unorm: return "RGBA16_UNORM";
		case vk::Format::eB10G11R11UfloatPack32: return "R11G11B10F";
		case vk::Format::eR16G16Sfloat: return "RG16F";
		case vk::Format::eR16G16Unorm: return "RG16_UNORM";
		case vk::Format::eR8G8Unorm: return "RG8_UNORM";
		case vk::Format::eR8Unorm: return "R8_UNORM";
		case vk::Format::eR16Sfloat: return "R16F";
		case vk::Format::eR32Sfloat: return "R32F";
		case vk::Format::eR32Uint: return "R32_UINT";
		case vk::Format::eR32G32Uint: return "RG32_UINT";
		case vk::Format::eR16G16B16A16Uint: return "RGBA16_UINT";
		case vk::Format::eR8G8B8A8Uint: return "RGBA8_UINT";
		case vk::Format::eD32Sfloat: return "D32F";
		case vk::Format::eD32SfloatS8Uint: return "D32F_S8";
		case vk::Format::eD24UnormS8Uint: return "D24_S8";
		case vk::Format::eD16Unorm: return "D16";
		default: return "other";
	}
}
// Filterable, normalized or float: a channel the shader can read as a 0..1 value.
bool SsrFormatUsable(vk::Format f) {
	switch (f) {
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eB10G11R11UfloatPack32:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR8Unorm:
		case vk::Format::eR16Sfloat: return true;
		default: return false;
	}
}

// The G-buffer as its draws on the normal path bind it (any recording thread writes, the upscale reads).
struct SsrGbuffer {
	std::mutex                  mutex;
	std::array<SsrTarget, 8>    targets {};
	uint32_t                    count   = 0;
	uint64_t                    flip    = 0;
	bool                        valid   = false;
	std::vector<std::pair<uint64_t, uint64_t>> logged; // (vs, ps) layouts logged
	std::vector<uint64_t>       consumers;              // compute programs logged as reading it
	std::array<float, 16>       view_proj {};
	bool                        have_vp = false;
	uint64_t                    vp_flip = ~0ull;
	std::array<uint64_t, 2>     packed_addresses {};   // the packed layers' last two addresses (double-buffered)
	std::array<float, 16>       prev_view_proj {};     // the previous frame's (rows x, y, w; z from the current)
	bool                        have_prev = false;
} g_ssr_gb;

bool SsrEnsure(GraphicContext& graphics, vk::Extent2D render, int level) {
	const auto device = graphics.device;
	if (g_ssr.pipeline == nullptr) {
		if (g_ssr.failed) return false;
		g_ssr.failed = true;
		std::array<vk::DescriptorSetLayoutBinding, SsrBindings> bindings {};
		for (uint32_t i = 0; i < SsrBindings; ++i) {
			bindings[i].binding         = i;
			bindings[i].descriptorType  = SsrType(i);
			bindings[i].descriptorCount = 1;
			bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
		}
		vk::DescriptorSetLayoutCreateInfo set_info {};
		set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		set_info.bindingCount = static_cast<uint32_t>(bindings.size());
		set_info.pBindings    = bindings.data();
		if (device.createDescriptorSetLayout(&set_info, nullptr, &g_ssr.set_layout) != vk::Result::eSuccess) return false;
		vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(SsrPush)};
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.setLayoutCount         = 1;
		layout_info.pSetLayouts            = &g_ssr.set_layout;
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges    = &push;
		if (device.createPipelineLayout(&layout_info, nullptr, &g_ssr.layout) != vk::Result::eSuccess) return false;
		vk::ShaderModuleCreateInfo module_info {};
		module_info.codeSize    = sizeof(SSR_SPV);
		module_info.pCode       = SSR_SPV;
		vk::ShaderModule module = nullptr;
		if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
		pipeline_info.stage.module = module;
		pipeline_info.stage.pName  = "main";
		pipeline_info.layout       = g_ssr.layout;
		const auto created = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_ssr.pipeline);
		device.destroyShaderModule(module, nullptr);
		if (created != vk::Result::eSuccess) {
			g_ssr.pipeline = nullptr;
			return false;
		}
		for (int i = 0; i < 2; ++i) {
			vk::SamplerCreateInfo sampler {};
			sampler.magFilter    = i == 0 ? vk::Filter::eNearest : vk::Filter::eLinear;
			sampler.minFilter    = sampler.magFilter;
			sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
			sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
			sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
			if (device.createSampler(&sampler, nullptr, i == 0 ? &g_ssr.point : &g_ssr.linear) != vk::Result::eSuccess) return false;
		}
		vk::QueryPoolCreateInfo query {};
		query.queryType  = vk::QueryType::eTimestamp;
		query.queryCount = 16;
		if (device.createQueryPool(&query, nullptr, &g_ssr.pool) != vk::Result::eSuccess) g_ssr.pool = nullptr;
		g_ssr.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
		{
			VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
			create.imageType   = VK_IMAGE_TYPE_2D;
			create.format      = VK_FORMAT_R32_UINT;
			create.extent      = {1, 1, 1};
			create.mipLevels   = 1;
			create.arrayLayers = 1;
			create.samples     = VK_SAMPLE_COUNT_1_BIT;
			create.tiling      = VK_IMAGE_TILING_OPTIMAL;
			create.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
			VmaAllocationCreateInfo allocate {};
			allocate.usage = VMA_MEMORY_USAGE_AUTO;
			if (vmaCreateImage(graphics.allocator, &create, &allocate, &g_ssr.dummy_image, &g_ssr.dummy_mem, nullptr) != VK_SUCCESS)
				return false;
			vk::ImageViewCreateInfo view_info {};
			view_info.image            = vk::Image(g_ssr.dummy_image);
			view_info.viewType         = vk::ImageViewType::e2D;
			view_info.format           = vk::Format::eR32Uint;
			view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
			if (device.createImageView(&view_info, nullptr, &g_ssr.dummy_view) != vk::Result::eSuccess) return false;
			VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			buffer.size  = SsrCalibBytes;
			buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			VmaAllocationCreateInfo host {};
			host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
			host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
			VmaAllocationInfo mapped {};
			if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_ssr.calib, &g_ssr.calib_mem, &mapped) != VK_SUCCESS) return false;
			g_ssr.calib_mapped = static_cast<const int32_t*>(mapped.pMappedData);
		}
		g_ssr.failed    = false;
	}
	if (g_ssr.out.image.image != nullptr && g_ssr.render == render && g_ssr.level == level) return true;
	const vk::Extent2D trace = level >= 2 ? render : vk::Extent2D {std::max(1u, (render.width + 1) / 2), std::max(1u, (render.height + 1) / 2)};
	const std::array<SsgiImage*, 10> all {&g_ssr.raw, &g_ssr.tmp_a, &g_ssr.tmp_b, &g_ssr.out, &g_ssr.mat, &g_ssr.hit,
	                                      &g_ssr.history[0], &g_ssr.history[1], &g_ssr.gbuf[0], &g_ssr.gbuf[1]};
	for (auto* img : all)
		SsgiFreeImage(graphics, *img);
	bool ok = SsgiMakeImage(graphics, g_ssr.out, render);
	for (auto* img : all)
		if (img != &g_ssr.out) ok = ok && SsgiMakeImage(graphics, *img, trace);
	if (!ok) {
		for (auto* img : all)
			SsgiFreeImage(graphics, *img);
		g_ssr.failed = true;
		return false;
	}
	g_ssr.render        = render;
	g_ssr.level         = level;
	g_ssr.trace         = trace;
	g_ssr.history_valid = false;
	return true;
}

void SsrReadTiming() {
	if (g_ssr.pool == nullptr || g_graphics_ctx == nullptr) return;
	while (g_ssr.read + 4 <= g_ssr.written) {
		const auto entry = static_cast<uint32_t>(g_ssr.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (g_graphics_ctx->device.getQueryPoolResults(g_ssr.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                               vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		g_ssr.sum_ms += static_cast<double>(stamps[1] - stamps[0]) * g_ssr.period_ns / 1e6;
		++g_ssr.samples;
		++g_ssr.read;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - g_ssr.last_log >= std::chrono::seconds(5) && g_ssr.samples > 0) {
		std::printf("SSR: GPU %.3f ms (mean of %u frames)\n", g_ssr.sum_ms / g_ssr.samples, g_ssr.samples);
		std::fflush(stdout);
		g_ssr.sum_ms   = 0.0;
		g_ssr.samples  = 0;
		g_ssr.last_log = now;
	}
}

// World up (KYTY_SSR_UP) in view space: View = P^-1 * view_proj (column vectors; P's column form is the transpose of
// the row-vector view_to_clip).
// The camera's world -> view matrix (rows of the 3x3), as above.
bool SsrViewRotation(const std::array<float, 16>& m, std::array<float, 12>& rows) {
	std::array<float, 16> vp {};
	{
		std::lock_guard lock(g_ssr_gb.mutex);
		if (!g_ssr_gb.have_vp || g_flips_taken.load(std::memory_order_relaxed) > g_ssr_gb.vp_flip + 4) return false;
		vp = g_ssr_gb.view_proj;
	}
	std::array<float, 16> pc {};
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j) pc[i * 4 + j] = m[j * 4 + i];
	std::array<double, 16> pinv {};
	if (!Inverse4(pc, pinv)) return false;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			double d = 0.0;
			for (int k = 0; k < 4; ++k) d += pinv[i * 4 + k] * vp[k * 4 + j];
			if (!std::isfinite(d)) return false;
			rows[i * 4 + j] = static_cast<float>(d);
		}
		rows[i * 4 + 3] = 0.0f;
	}
	return true;
}

// The calibration's verdict (after its frames are done): the normal layout, the roughness field; logged with the scores.
void SsrAnalyze(uint32_t packed_count) {
	if (g_ssr.calib_mapped == nullptr || g_graphics_ctx == nullptr) return;
	(void)vmaInvalidateAllocation(g_graphics_ctx->allocator, g_ssr.calib_mem, 0, VK_WHOLE_SIZE);
	const int32_t* c = g_ssr.calib_mapped + 12;
	const double   n = static_cast<double>(c[0]);
	static const char* const layouts[5] = {"oct 16:16", "oct 12:12 bits 0..23", "oct 12:12 bits 8..31", "oct 11:11 bits 0..21",
	                                       "xyz 10:10:10"};
	std::printf("SSR: G-buffer calibration: %.0f samples over %u frames, %u packed layers\n", n, g_ssr.calib_runs, packed_count);
	int    best       = -1;
	double best_score = 0.0;
	if (n >= 200.0) {
		for (uint32_t layer = 0; layer < packed_count; ++layer) {
			std::string line;
			char        buf[96];
			for (int l = 0; l < 5; ++l)
				for (int space = 0; space < 2; ++space) {
					const double score = c[1 + (layer * 5 + l) * 2 + space] / 1000.0 / n;
					std::snprintf(buf, sizeof(buf), " %s/%s %+.3f;", layouts[l], space ? "world" : "view", score);
					line += buf;
					if (std::abs(score) > std::abs(best_score))
						best_score = score, best = static_cast<int>(layer) | (l << 2) | (space << 5);
				}
			std::printf("SSR: G-buffer layer %u normal fit (mean cos vs depth normal):%s\n", layer, line.c_str());
			line.clear();
			for (int b = 0; b < 4; ++b) {
				const int    f    = static_cast<int>(layer) * 4 + b;
				const double mean = c[41 + f * 3] / 1000.0 / n, sq = c[42 + f * 3] / 1000.0 / n, nov = c[43 + f * 3] / 1000.0 / n;
				std::snprintf(buf, sizeof(buf), " bits %d..%d mean %.3f sd %.3f cov(1-NoV) %+.3f;", b * 8, b * 8 + 7, mean,
				              std::sqrt(std::max(sq - mean * mean, 0.0)), nov);
				line += buf;
			}
			std::printf("SSR: G-buffer layer %u byte fields:%s\n", layer, line.c_str());
		}
	}
	if (best >= 0 && std::abs(best_score) >= 0.85) {
		if (best_score < 0.0) best |= 64;
		if (g_ssr.dec_normal < 0) g_ssr.dec_normal = best;
		std::printf("SSR: G-buffer normal = layer %d, %s, %s space%s (mean cos %.3f)\n", best & 3, layouts[(best >> 2) & 7],
		            (best & 32) ? "world" : "view", (best & 64) ? ", negated" : "", std::abs(best_score));
		// Roughness: the byte the normal's layout leaves free in its word, when it varies (else unknown).
		const int layout = (best >> 2) & 7, layer = best & 3;
		const int shift  = layout == 1 ? 24 : layout == 2 ? 0 : -1;
		if (g_ssr.dec_rough < 0 && shift >= 0) {
			const int    f    = layer * 4 + shift / 8;
			const double mean = c[41 + f * 3] / 1000.0 / n, sq = c[42 + f * 3] / 1000.0 / n;
			if (std::sqrt(std::max(sq - mean * mean, 0.0)) > 0.02) {
				g_ssr.dec_rough = layer | (shift << 2) | (8 << 7);
				std::printf("SSR: roughness = layer %d bits %d..%d (the normal word's free byte; GUESS, set KYTY_SSR_ROUGH to "
				            "override)\n",
				            layer, shift, shift + 7);
			}
		}
	} else {
		std::printf("SSR: G-buffer normal not identified (best |mean cos| %.3f < 0.85): depth-derived normals stay\n",
		            std::abs(best_score));
	}
	if (g_ssr.dec_rough < 0)
		std::printf("SSR: roughness unknown (set KYTY_SSR_ROUGH=<layer>:<shift>:<bits>[g] from the byte fields above): "
		            "heuristic mask with the G-buffer normals\n");
	std::fflush(stdout);
}

bool SsrUpInView(const std::array<float, 16>& m, std::array<float, 3>& up) {
	std::array<float, 16> vp {};
	{
		std::lock_guard lock(g_ssr_gb.mutex);
		if (!g_ssr_gb.have_vp || g_flips_taken.load(std::memory_order_relaxed) > g_ssr_gb.vp_flip + 4) return false;
		vp = g_ssr_gb.view_proj;
	}
	static const std::pair<int, float> axis = [] {
		const char* value = std::getenv("KYTY_SSR_UP");
		std::string v     = value != nullptr ? value : "y";
		float       sign  = 1.0f;
		if (!v.empty() && v[0] == '-') sign = -1.0f, v.erase(0, 1);
		const int a = v == "x" ? 0 : v == "z" ? 2 : 1;
		return std::pair<int, float> {a, sign};
	}();
	std::array<float, 16> pc {};
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j) pc[i * 4 + j] = m[j * 4 + i];
	std::array<double, 16> pinv {};
	if (!Inverse4(pc, pinv)) return false;
	double d[3] {};
	for (int i = 0; i < 3; ++i)
		for (int k = 0; k < 4; ++k) d[i] += pinv[i * 4 + k] * vp[k * 4 + axis.first];
	const double n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
	if (!(n > 1e-9) || !std::isfinite(n)) return false;
	for (int i = 0; i < 3; ++i) up[i] = static_cast<float>(axis.second * d[i] / n);
	return true;
}

bool SsrApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int level = SsrLevel();
	if (level == 0) return false; // off: nothing created or recorded
	if (g_graphics_ctx == nullptr) return false;
	if (VideoOut::MoviePlaying()) {
		g_ssr.history_valid = false;
		return false;
	}
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || inputs.motion.view == nullptr || color.extent.width < 16 ||
	    color.extent.height < 16)
		return false;
	const auto& m = inputs.view_to_clip;
	if (!(m[0] != 0.0f && m[5] != 0.0f && (m[11] != 0.0f || m[15] != 0.0f))) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("SSR: skipped (no camera projection from the upscaler)\n");
			std::fflush(stdout);
		}
		return false;
	}
	if (!SsrEnsure(*g_graphics_ctx, color.extent, level)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("SSR: off (could not create its pipeline or images)\n");
			std::fflush(stdout);
		}
		return false;
	}
	const bool have_material = inputs.material.view != nullptr && inputs.material_channel >= 0 && inputs.material_channel < 4 &&
	                           inputs.material.extent == color.extent;
	uint32_t packed_count = 0;
	bool     from_copy    = false;
	{
		// The copy taken at the deferred lighting this flip or the last (else the live image, which later passes may
		// have overwritten by now).
		const uint64_t copied = g_ssr.copy_flip.load(std::memory_order_acquire);
		const uint64_t flips  = g_flips_taken.load(std::memory_order_relaxed);
		if (copied != ~0ull && flips <= copied + 1 && g_ssr.copy_extent == color.extent) {
			for (uint32_t i = 0; i < 4; ++i) {
				auto& p  = inputs.packed[i];
				p.view   = g_ssr.copy_views[i];
				p.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
				p.extent = g_ssr.copy_extent;
				p.format = vk::Format::eR32Uint;
			}
			inputs.packed_count = 4;
			from_copy           = true;
		}
	}
	while (packed_count < inputs.packed_count && packed_count < 4 && inputs.packed[packed_count].view != nullptr &&
	       inputs.packed[packed_count].extent == color.extent)
		++packed_count;
	if (const uint32_t key = packed_count * 2 + (from_copy ? 1 : 0); key != g_ssr.packed_logged) {
		static uint32_t reports = 0;
		if (reports++ < 8) {
			std::printf("SSR: packed G-buffer source: %u layers, %s\n", packed_count,
			            packed_count == 0 ? "none this frame"
			            : from_copy       ? "copied before the deferred lighting"
			                              : "the live image at the upscaler (may be overwritten)");
			std::fflush(stdout);
		}
		g_ssr.packed_logged = key;
	}
	{
		static const int32_t env_normal = SsrNormalEnv(), env_rough = SsrFieldEnv("KYTY_SSR_ROUGH"),
		                     env_metal = SsrFieldEnv("KYTY_SSR_METAL");
		static bool          said      = false;
		if (!said) {
			said = true;
			if (env_normal >= 0) g_ssr.dec_normal = env_normal, g_ssr.calib_done = true;
			if (env_rough >= 0) g_ssr.dec_rough = env_rough;
			if (env_metal >= 0) g_ssr.dec_metal = env_metal;
			if (env_normal >= 0 || env_rough >= 0 || env_metal >= 0)
				std::printf("SSR: decode from the environment (normal %d, roughness %d, metal %d)\n", env_normal, env_rough, env_metal);
		}
	}
	if (packed_count == 0 && !have_material && g_ssr.copy_flip.load(std::memory_order_relaxed) != ~0ull) {
		// (the copy worked before but not this frame or the last: no G-buffer this frame, SSR skipped rather than
		// traced from stale or missing layers)
		static uint32_t skips = 0;
		if (skips++ < 4) {
			std::printf("SSR: skipped this frame (no G-buffer copy)\n");
			std::fflush(stdout);
		}
		return false;
	}
	const bool use_packed = packed_count > 0 && !have_material;
	static uint32_t live_frames = 0;
	if (use_packed && !from_copy && live_frames < 100000) ++live_frames;
	if (live_frames == 600 && g_ssr.copy_flip.load(std::memory_order_relaxed) == ~0ull) {
		std::printf("SSR: no G-buffer copy after 600 frames: calibrating on the live image\n");
		std::fflush(stdout);
	}
	const bool calib_live = live_frames >= 600 && g_ssr.copy_flip.load(std::memory_order_relaxed) == ~0ull;
	// (every 4th eligible frame, not every 4th frame: a source present every other frame would never line up)
	static uint32_t calib_tick = 0;
	const bool      eligible   = use_packed && (from_copy || calib_live) && !g_ssr.calib_done && g_ssr.calib_runs < SsrCalibRuns;
	const bool      calibrate  = eligible && (calib_tick++ & 3u) == 0;
	if (use_packed && !g_ssr.calib_done && g_ssr.calib_runs >= SsrCalibRuns && ++g_ssr.calib_wait > 8) {
		g_ssr.calib_done = true;
		SsrAnalyze(packed_count);
	}
	const auto usable = [&](int32_t code) { return use_packed && code >= 0 && (code & 3) < static_cast<int32_t>(packed_count) ? code : -1; };
	const int32_t dec_normal = usable(g_ssr.dec_normal), dec_rough = usable(g_ssr.dec_rough), dec_metal = usable(g_ssr.dec_metal);
	std::array<float, 12> view_rows {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
	const bool            have_rows = SsrViewRotation(inputs.view_to_clip, view_rows);
	{
		static int32_t logged = -2;
		const int32_t  key    = dec_normal * 7919 + dec_rough * 31 + dec_metal;
		if (key != logged && dec_normal >= 0) {
			logged = key;
			std::printf("SSR: decoding the packed G-buffer: normal code %d%s, roughness %s, metal %s\n", dec_normal,
			            (dec_normal & 32) != 0 && !have_rows ? " (world space, camera not seen yet)" : "",
			            dec_rough >= 0 ? "from the G-buffer" : "heuristic", dec_metal >= 0 ? "from the G-buffer" : "none");
			std::fflush(stdout);
		}
	}
	const int source = have_material ? SsrChoice().slot * 4 + inputs.material_channel : dec_rough >= 0 ? 1000 + dec_rough : -1;
	if (source != g_ssr.source) {
		static uint32_t reports = 0;
		if (reports++ < 16) {
			static const char channels[] = "rgba";
			if (!have_material && dec_rough >= 0)
				std::printf("SSR: level %d, roughness source packed G-buffer layer %d bits %d..%d%s\n", level, dec_rough & 3,
				            (dec_rough >> 2) & 31, ((dec_rough >> 2) & 31) + ((dec_rough >> 7) & 31) - 1,
				            (dec_rough & 4096) != 0 ? " (gloss)" : "");
			else if (have_material)
				std::printf("SSR: level %d, roughness source G-buffer RT%d.%c (%s, %s)\n", level, SsrChoice().slot,
				            channels[inputs.material_channel], SsrFormatName(inputs.material.format),
				            inputs.material_gloss ? "gloss: roughness = 1 - value" : "roughness");
			else
				std::printf("SSR: level %d, roughness source heuristic (depth-derived up-facing normals + low local variance + low "
				            "luminance; HEURISTIC)%s\n",
				            level, SsrChoice().slot >= 0 ? " - KYTY_SSR_GBUF target not usable this frame" : "");
			std::fflush(stdout);
		}
		g_ssr.source = source;
	}
	std::array<float, 3> up {};
	const bool           have_up = SsrUpInView(m, up);
	if (static bool said_up = false; have_up && !said_up) {
		said_up = true;
		std::printf("SSR: camera seen (up direction and world -> view rotation known)\n");
		std::fflush(stdout);
	}

	const float strength = SsgiEnv("KYTY_SSR_STRENGTH", 0.8f, 2.0f);
	const float thick    = std::max(SsgiEnv("KYTY_SSR_THICKNESS", 0.25f, 4.0f), 0.01f);
	SsrPush     push {};
	push.proj[0] = m[0], push.proj[1] = m[5], push.proj[2] = m[8], push.proj[3] = m[9];
	push.depth_map[0] = m[10], push.depth_map[1] = m[14], push.depth_map[2] = m[11], push.depth_map[3] = m[15];
	push.motion[0] = inputs.motion_scale_x, push.motion[1] = inputs.motion_scale_y;
	push.motion[2] = static_cast<float>(color.extent.width), push.motion[3] = static_cast<float>(color.extent.height);
	push.trace[0] = static_cast<float>(g_ssr.trace.width), push.trace[1] = static_cast<float>(g_ssr.trace.height);
	push.trace[2] = strength, push.trace[3] = level >= 3 ? 60.0f : level >= 2 ? 40.0f : 25.0f;
	push.extra[1] = static_cast<float>(std::max(inputs.material_channel, 0));
	push.extra[2] = inputs.material_gloss ? 1.0f : 0.0f;
	push.extra[3] = thick;
	push.up[0] = up[0], push.up[1] = up[1], push.up[2] = up[2];
	push.dec[0] = (dec_normal & 32) != 0 && !have_rows ? -1 : dec_normal;
	push.dec[1] = dec_rough, push.dec[2] = dec_metal, push.dec[3] = static_cast<int32_t>(packed_count);
	{
		// (on a change of the layer count, held for 120 frames: no line per frame when the source flickers)
		static uint32_t said_layers = ~0u, pending = ~0u, held = 0, lines = 0;
		if (packed_count != pending) pending = packed_count, held = 0;
		else if (held < 1000) ++held;
		if (said_layers != pending && (said_layers == ~0u || held == 120) && lines < 16) {
			said_layers = pending;
			++lines;
			std::printf("SSR: on (level %d, trace %ux%u, strength %g, thickness %g, up %s, packed G-buffer layers %u%s)\n", level,
			            g_ssr.trace.width, g_ssr.trace.height, strength, thick,
			            have_up ? "from the camera" : "unknown (|view y| assumed until the G-buffer normals are decoded)", packed_count,
			            use_packed && !g_ssr.calib_done ? ", calibrating" : "");
			std::fflush(stdout);
		}
	}
	SsrReadTiming();
	const auto entry = static_cast<uint32_t>(g_ssr.written % 8);
	const bool timed = g_ssr.pool != nullptr;
	if (timed) {
		command.resetQueryPool(g_ssr.pool, entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_ssr.pool, entry * 2);
	}
	const auto global = [&](vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src, vk::PipelineStageFlags2 dst_stage,
	                        vk::AccessFlags2 dst) {
		vk::MemoryBarrier2 barrier {};
		barrier.srcStageMask  = src_stage;
		barrier.srcAccessMask = src;
		barrier.dstStageMask  = dst_stage;
		barrier.dstAccessMask = dst;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &barrier;
		command.pipelineBarrier2(dependency);
	};
	auto& g = g_ssr;
	for (auto* img : {&g.raw, &g.tmp_a, &g.tmp_b, &g.out, &g.mat, &g.hit, &g.history[0], &g.history[1], &g.gbuf[0], &g.gbuf[1]})
		if (img->image.state.layout != vk::ImageLayout::eGeneral)
			Transition(command, img->image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
			           vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader);
	global(vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	       vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	const uint32_t cur  = g.frame & 1u;
	const uint32_t prev = cur ^ 1u;
	push.misc[1] = static_cast<int32_t>(g.frame);
	push.misc[2] = level;
	push.misc[3] = (inputs.reset ? 1 : 0) | (g.history_valid ? 2 : 0) | (have_material ? 4 : 0) | (have_up ? 8 : 0);
	if (!g.dummy_ready) {
		g.dummy_ready = true;
		vk::ImageMemoryBarrier2 barrier {};
		barrier.srcStageMask     = vk::PipelineStageFlagBits2::eNone;
		barrier.dstStageMask     = vk::PipelineStageFlagBits2::eTransfer;
		barrier.dstAccessMask    = vk::AccessFlagBits2::eTransferWrite;
		barrier.oldLayout        = vk::ImageLayout::eUndefined;
		barrier.newLayout        = vk::ImageLayout::eGeneral;
		barrier.image            = vk::Image(g.dummy_image);
		barrier.subresourceRange = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers    = &barrier;
		command.pipelineBarrier2(dependency);
		vk::ClearColorValue             zero {};
		zero.uint32 = std::array<uint32_t, 4> {0, 0, 0, 0};
		const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		command.clearColorImage(vk::Image(g.dummy_image), vk::ImageLayout::eGeneral, &zero, 1, &whole);
		command.fillBuffer(vk::Buffer(g.calib), 0, VK_WHOLE_SIZE, 0);
		global(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer,
		       vk::AccessFlagBits2::eTransferWrite);
	}
	command.updateBuffer(vk::Buffer(g.calib), 0, sizeof(float) * view_rows.size(), view_rows.data());
	global(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
	       vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipeline);
	const auto run = [&](int mode, const SsgiImage& a, const SsgiImage& b, const SsgiImage& out, const SsgiImage& gout,
	                     const SsgiImage& aux_in, const SsgiImage& aux_out, const SsgiImage& mat_in, const SsgiImage& mat_out,
	                     float step, vk::Extent2D size) {
		const vk::Sampler       lin = g.linear, pt = g.point;
		vk::DescriptorImageInfo infos[SsrBindings] {};
		infos[0]  = {lin, color.view, color.layout};
		infos[1]  = {pt, inputs.depth.view, inputs.depth.layout};
		infos[2]  = {pt, inputs.motion.view, inputs.motion.layout};
		infos[3]  = {lin, a.view, vk::ImageLayout::eGeneral};
		infos[4]  = {lin, b.view, vk::ImageLayout::eGeneral};
		infos[5]  = {pt, g.gbuf[cur].view, vk::ImageLayout::eGeneral};
		infos[6]  = {pt, g.gbuf[prev].view, vk::ImageLayout::eGeneral};
		infos[7]  = {nullptr, out.view, vk::ImageLayout::eGeneral};
		infos[8]  = {nullptr, gout.view, vk::ImageLayout::eGeneral};
		infos[9]  = {pt, aux_in.view, vk::ImageLayout::eGeneral};
		infos[10] = {nullptr, aux_out.view, vk::ImageLayout::eGeneral};
		infos[11] = have_material ? vk::DescriptorImageInfo {pt, inputs.material.view, inputs.material.layout}
		                          : vk::DescriptorImageInfo {pt, color.view, color.layout};
		infos[12] = {pt, mat_in.view, vk::ImageLayout::eGeneral};
		infos[13] = {nullptr, mat_out.view, vk::ImageLayout::eGeneral};
		for (uint32_t i = 0; i < 4; ++i)
			infos[14 + i] = use_packed && i < packed_count
			                    ? vk::DescriptorImageInfo {pt, inputs.packed[i].view, inputs.packed[i].layout}
			                    : vk::DescriptorImageInfo {pt, g.dummy_view, vk::ImageLayout::eGeneral};
		const vk::DescriptorBufferInfo calib_info {vk::Buffer(g.calib), 0, VK_WHOLE_SIZE};
		vk::WriteDescriptorSet writes[SsrBindings] {};
		for (uint32_t i = 0; i < SsrBindings; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = SsrType(i);
			if (i == 18) writes[i].pBufferInfo = &calib_info;
			else writes[i].pImageInfo = &infos[i];
		}
		push.misc[0]  = mode;
		push.extra[0] = step;
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g.layout, 0, SsrBindings, writes);
		command.pushConstants(g.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
		global(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		       vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	};
	// (Bindings a mode does not write still need a valid storage image: tmp_b is the scratch target; tmp_a the unread
	// sampled one.)
	if (calibrate) {
		run(5, g.tmp_a, g.tmp_a, g.tmp_b, g.tmp_b, g.tmp_a, g.tmp_b, g.tmp_a, g.tmp_b, 0.0f, vk::Extent2D {160, 90});
		++g.calib_runs;
	}
	run(0, g.tmp_a, g.tmp_a, g.raw, g.gbuf[cur], g.tmp_a, g.hit, g.tmp_a, g.mat, 0.0f, g.trace);                 // trace
	run(1, g.raw, g.history[prev], g.history[cur], g.tmp_b, g.hit, g.tmp_b, g.mat, g.tmp_b, 0.0f, g.trace);      // temporal
	const SsgiImage* src     = &g.history[cur];
	const SsgiImage* ping[2] = {&g.tmp_a, &g.raw};
	const int        passes  = level >= 3 ? 4 : level >= 2 ? 3 : 2;
	for (int i = 0; i < passes; ++i) {
		const SsgiImage* dst = ping[i & 1];
		run(2, *src, g.tmp_b, *dst, g.tmp_b, g.hit, g.tmp_b, g.mat, g.tmp_b, static_cast<float>(1 << i), g.trace);
		src = dst;
	}
	run(4, *src, g.tmp_b, g.out, g.tmp_b, g.hit, g.tmp_b, g.mat, g.tmp_b, 0.0f, g.render);                       // composite
	global(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite, vk::PipelineStageFlagBits2::eAllCommands,
	       vk::AccessFlagBits2::eMemoryRead);
	if (timed) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g.pool, entry * 2 + 1);
		++g.written;
	}
	++g.frame;
	g.history_valid = true;
	inputs.color.image  = g.out.image.image;
	inputs.color.view   = g.out.view;
	inputs.color.layout = vk::ImageLayout::eGeneral;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = g.out.image.usage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}
} // namespace

bool SsrWanted() {
	return SsrLevel() > 0 || RtReflectionsWanted() || ShineWanted() || GtaoLevel() > 0 || GbufViewWanted() ||
	       SsgiEmissive() > 0.0f || // (rtfire: the emissive light reads the G-buffer copy's albedo)
	       HwGiWanted();             // (KYTY_HWRT_AO / KYTY_HWRT_GI: its normal and albedo)
}
bool SsrPassWanted() { return SsrLevel() > 0; }

void OnSsrDraw(uint64_t vs_hash, std::span<const uint32_t> srt) {
	if (!IsGbufferVertexShader(vs_hash) || srt.size() < SrtViewProj + 16) return;
	const uint64_t flip = g_flips_taken.load(std::memory_order_relaxed);
	std::lock_guard lock(g_ssr_gb.mutex);
	// The G-buffer pass is drawn this frame. Its targets are known from a normal-path draw (OnSsrGbuffer), but the
	// G-buffer draws almost all take the table path (table-xpr.inc) or records (native-xpr.inc), which call only this
	// hook: without this the targets' flip went stale after the first frames and SsrCaptureGbuffer refused the copy
	// ("no G-buffer pass this frame or the last") nearly every frame. (The copy still proves the consumer binds the same
	// address with a 4-layer R32_UINT backing large enough.)
	if (g_ssr_gb.valid && g_ssr_gb.count > 0 && g_ssr_gb.flip != flip) g_ssr_gb.flip = flip;
	if (g_ssr_gb.vp_flip == flip) return; // (the frame's first G-buffer draw)
	std::array<float, 16> vp {};
	for (uint32_t i = 0; i < 16; ++i) {
		vp[i] = AsFloat(srt[SrtViewProj + i]);
		if (!std::isfinite(vp[i])) return;
	}
	g_ssr_gb.view_proj = vp;
	g_ssr_gb.have_vp   = true;
	g_ssr_gb.vp_flip   = flip;
	// (KYTY_RT_REFLECTIONS: the previous frame's rows x, y and w for the reflections' reprojection)
	g_ssr_gb.have_prev = false;
	if (srt.size() >= SrtPrevViewProj + 12) {
		std::array<float, 16> pv {};
		for (uint32_t i = 0; i < 4; ++i) {
			pv[i]      = AsFloat(srt[SrtPrevViewProj + i]);
			pv[4 + i]  = AsFloat(srt[SrtPrevViewProj + 4 + i]);
			pv[8 + i]  = vp[8 + i];
			pv[12 + i] = AsFloat(srt[SrtPrevViewProj + 8 + i]);
		}
		if (std::all_of(pv.begin(), pv.end(), [](float v) { return std::isfinite(v); })) {
			g_ssr_gb.prev_view_proj = pv;
			g_ssr_gb.have_prev      = true;
		}
	}
}

void OnSsrGbuffer(uint64_t vs_hash, uint64_t ps_hash, std::span<const SsrTarget> targets) {
	if (targets.empty()) return;
	const uint64_t flip = g_flips_taken.load(std::memory_order_relaxed);
	std::lock_guard lock(g_ssr_gb.mutex);
	const uint32_t n = static_cast<uint32_t>(std::min<size_t>(targets.size(), g_ssr_gb.targets.size()));
	// The layout with the most targets wins (the main G-buffer pass; the decals bind fewer).
	if (!(g_ssr_gb.valid && g_ssr_gb.flip == flip && g_ssr_gb.count > n)) {
		for (uint32_t i = 0; i < n; ++i) g_ssr_gb.targets[i] = targets[i];
		g_ssr_gb.count = n;
		g_ssr_gb.flip  = flip;
		g_ssr_gb.valid = true;
		if (const uint64_t a = targets[0].address; a != 0 && targets[0].view.format == vk::Format::eR32Uint &&
		                                           g_ssr_gb.packed_addresses[0] != a) {
			g_ssr_gb.packed_addresses[1] = g_ssr_gb.packed_addresses[0];
			g_ssr_gb.packed_addresses[0] = a;
		}
	}
	const auto key = std::make_pair(vs_hash, ps_hash);
	if (g_ssr_gb.logged.size() >= 24 || std::find(g_ssr_gb.logged.begin(), g_ssr_gb.logged.end(), key) != g_ssr_gb.logged.end()) return;
	g_ssr_gb.logged.push_back(key);
	std::string line;
	char        buf[160];
	for (uint32_t i = 0; i < n; ++i) {
		const auto& t = targets[i];
		std::snprintf(buf, sizeof(buf), " RT%u %s(%d) %ux%u @0x%llx layer %u;", i, SsrFormatName(t.view.format),
		              static_cast<int>(t.view.format), t.width, t.height, static_cast<unsigned long long>(t.address), t.view.base_layer);
		line += buf;
	}
	std::printf("SSR: G-buffer draw VS %016llx PS %016llx:%s\n", static_cast<unsigned long long>(vs_hash),
	            static_cast<unsigned long long>(ps_hash), line.c_str());
	std::fflush(stdout);
}

void OnSsrDispatch(uint64_t cs_hash, std::span<const uint64_t> addresses, std::span<const vk::Format> formats) {
	std::lock_guard lock(g_ssr_gb.mutex);
	if (!g_ssr_gb.valid || g_ssr_gb.consumers.size() >= 32) return;
	if (std::find(g_ssr_gb.consumers.begin(), g_ssr_gb.consumers.end(), cs_hash) != g_ssr_gb.consumers.end()) return;
	std::string line;
	char        buf[128];
	uint32_t    hits = 0;
	for (size_t s = 0; s < addresses.size(); ++s)
		for (uint32_t i = 0; i < g_ssr_gb.count; ++i)
			if (addresses[s] != 0 && addresses[s] == g_ssr_gb.targets[i].address) {
				std::snprintf(buf, sizeof(buf), " slot %zu <- RT%u (%s);", s, i, SsrFormatName(s < formats.size() ? formats[s] : vk::Format::eUndefined));
				line += buf;
				++hits;
			}
	if (hits < 2) return; // (a consumer of the G-buffer reads several of its targets: the deferred lighting)
	g_ssr_gb.consumers.push_back(cs_hash);
	std::printf("SSR: G-buffer consumer CS %016llx (%u of %u targets):%s\n", static_cast<unsigned long long>(cs_hash), hits,
	            g_ssr_gb.count, line.c_str());
	std::fflush(stdout);
}

// Why the copy was not taken (each reason logged the first time, then every 600th time).
void SsrCaptureSkip(const char* reason, uint64_t cs_hash) {
	static std::mutex                              mutex;
	static std::vector<std::pair<std::string, uint32_t>> seen;
	std::lock_guard lock(mutex);
	for (auto& [r, n]: seen)
		if (r == reason) {
			if (++n % 600 == 0) std::printf("SSR: G-buffer copy not taken (CS %016llx): %s (x%u)\n", static_cast<unsigned long long>(cs_hash), reason, n);
			return;
		}
	if (seen.size() >= 32) return;
	seen.emplace_back(reason, 1);
	std::printf("SSR: G-buffer copy not taken (CS %016llx): %s\n", static_cast<unsigned long long>(cs_hash), reason);
	std::fflush(stdout);
}

bool SsrIsDepthFormat(vk::Format f) {
	return f == vk::Format::eD32Sfloat || f == vk::Format::eD32SfloatS8Uint || f == vk::Format::eD24UnormS8Uint ||
	       f == vk::Format::eX8D24UnormPack32 || f == vk::Format::eD16Unorm || f == vk::Format::eD16UnormS8Uint;
}

// KYTY_RT_REFLECTIONS' water detection: the consumer's depth binding (a depth-format backing at least the G-buffer's
// size) copied right after the G-buffer layers, in the same command buffer.
void SsrCaptureDepth(TextureCache& cache, vk::CommandBuffer command, uint64_t cs_hash, std::span<const TextureBinding> images,
                     uint32_t width, uint32_t height, uint64_t flip) {
	if (!RtReflectionsWanted() || g_ssr.depth_failed || g_graphics_ctx == nullptr) return;
	static bool listed = false;
	if (!listed) {
		// (once: what the consumer binds, for the next look at its inputs)
		listed = true;
		std::string line;
		char        buf[160];
		for (size_t i = 0; i < images.size() && i < 24; ++i) {
			const auto& b = images[i];
			if (!b.image_id) continue;
			auto& im = cache.GetImage(b.image_id);
			std::snprintf(buf, sizeof(buf), " %zu: %s %ux%ux%u @0x%llx;", i, SsrFormatName(im.backing.format), im.backing.extent.width,
			              im.backing.extent.height, im.backing.layers, static_cast<unsigned long long>(b.desc.info.data.address));
			line += buf;
		}
		std::printf("SSR: G-buffer consumer CS %016llx binds:%s\n", static_cast<unsigned long long>(cs_hash), line.c_str());
		std::fflush(stdout);
	}
	// (KYTY_RT_REFLECTIONS_DEPTH=<n>: the n-th depth binding of the consumer; Demon's Souls' lighting binds two
	// D32F_S8 images of the render size, the first is the default)
	static const int which = [] {
		const char* v = std::getenv("KYTY_RT_REFLECTIONS_DEPTH");
		return v != nullptr ? std::clamp(std::atoi(v), 0, 7) : 0;
	}();
	const TextureBinding* binding = nullptr;
	int                   seen    = 0;
	for (const auto& b: images) {
		if (!b.image_id) continue;
		auto& im = cache.GetImage(b.image_id);
		if (im.backing.image == nullptr || !SsrIsDepthFormat(im.backing.format)) continue;
		if (im.backing.extent.width < width || im.backing.extent.height < height) continue;
		if (seen++ < which) continue;
		binding = &b;
		break;
	}
	if (binding == nullptr) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("SSR: the G-buffer consumer binds no depth image: no water detection\n");
			std::fflush(stdout);
		}
		return;
	}
	auto&              image = cache.GetImage(binding->image_id);
	const vk::Format   fmt   = image.backing.format;
	const vk::Extent2D extent {width, height};
	if (g_ssr.depth_image != VK_NULL_HANDLE && (g_ssr.depth_extent != extent || g_ssr.depth_format != fmt)) {
		SsrPass::RetiredCopy r;
		r.image = g_ssr.depth_image, r.mem = g_ssr.depth_mem, r.flip = flip, r.views[0] = g_ssr.depth_view;
		g_ssr.retired_copies.push_back(r);
		g_ssr.depth_image = VK_NULL_HANDLE, g_ssr.depth_mem = nullptr, g_ssr.depth_view = nullptr;
		g_ssr.depth_layout = vk::ImageLayout::eUndefined;
	}
	if (g_ssr.depth_image == VK_NULL_HANDLE) {
		g_ssr.depth_failed = true;
		VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		create.imageType   = VK_IMAGE_TYPE_2D;
		create.format      = static_cast<VkFormat>(fmt);
		create.extent      = {width, height, 1};
		create.mipLevels   = 1;
		create.arrayLayers = 1;
		create.samples     = VK_SAMPLE_COUNT_1_BIT;
		create.tiling      = VK_IMAGE_TILING_OPTIMAL;
		create.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo allocate {};
		allocate.usage = VMA_MEMORY_USAGE_AUTO;
		if (vmaCreateImage(g_graphics_ctx->allocator, &create, &allocate, &g_ssr.depth_image, &g_ssr.depth_mem, nullptr) != VK_SUCCESS) {
			g_ssr.depth_image = VK_NULL_HANDLE;
			std::printf("SSR: depth copy image could not be created (%s)\n", SsrFormatName(fmt));
			return;
		}
		vk::ImageViewCreateInfo view_info {};
		view_info.image            = vk::Image(g_ssr.depth_image);
		view_info.viewType         = vk::ImageViewType::e2D;
		view_info.format           = fmt;
		view_info.subresourceRange = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
		if (g_graphics_ctx->device.createImageView(&view_info, nullptr, &g_ssr.depth_view) != vk::Result::eSuccess) {
			g_ssr.depth_view = nullptr;
			return;
		}
		g_ssr.depth_extent = extent, g_ssr.depth_format = fmt;
		g_ssr.depth_failed = false;
		std::printf("SSR: depth copied with the G-buffer (%ux%u %s @0x%llx): water detection\n", width, height, SsrFormatName(fmt),
		            static_cast<unsigned long long>(binding->desc.info.data.address));
		std::fflush(stdout);
	}
	const ImageSubresourceRange range {0, 1, 0, 1};
	// (layout transitions name every aspect of a depth / stencil image; the copy and the view only the depth)
	const vk::ImageAspectFlags full_aspect =
	    (fmt == vk::Format::eD32Sfloat || fmt == vk::Format::eD16Unorm || fmt == vk::Format::eX8D24UnormPack32)
	        ? vk::ImageAspectFlags(vk::ImageAspectFlagBits::eDepth)
	        : (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil);
	image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, range, command);
	const auto barrier = [&](vk::ImageLayout from, vk::ImageLayout to, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
	                         vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
		vk::ImageMemoryBarrier2 b {};
		b.srcStageMask        = src_stage;
		b.srcAccessMask       = src_access;
		b.dstStageMask        = dst_stage;
		b.dstAccessMask       = dst_access;
		b.oldLayout           = from;
		b.newLayout           = to;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image               = vk::Image(g_ssr.depth_image);
		b.subresourceRange    = vk::ImageSubresourceRange {full_aspect, 0, 1, 0, 1};
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers    = &b;
		command.pipelineBarrier2(dependency);
	};
	barrier(g_ssr.depth_layout, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eAllCommands,
	        vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
	vk::ImageCopy copy {};
	copy.srcSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eDepth, 0, 0, 1};
	copy.dstSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eDepth, 0, 0, 1};
	copy.extent         = vk::Extent3D {width, height, 1};
	command.copyImage(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, vk::Image(g_ssr.depth_image),
	                  vk::ImageLayout::eTransferDstOptimal, 1, &copy);
	barrier(vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
	g_ssr.depth_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
	const vk::ImageLayout back = binding->layout == vk::ImageLayout::eUndefined ? vk::ImageLayout::eGeneral : binding->layout;
	image.Transit(back, vk::AccessFlagBits2::eShaderRead, range, command);
	g_ssr.depth_flip = flip;
}

void SsrCaptureGbuffer(GraphicContext& graphics, TextureCache& cache, vk::CommandBuffer command, uint64_t cs_hash,
                       std::span<const TextureBinding> images) {
	if (!SsrWanted() || g_ssr.copy_failed) return;
	if (g_graphics_ctx == nullptr) g_graphics_ctx = &graphics;
	const uint64_t flip = g_flips_taken.load(std::memory_order_relaxed);
	if (g_ssr.copy_flip.load(std::memory_order_relaxed) == flip) return; // (once a frame: the first consumer)
	uint32_t width = 0, height = 0;
	std::array<uint64_t, 2> addresses {};
	{
		std::lock_guard lock(g_ssr_gb.mutex);
		if (!g_ssr_gb.valid || g_ssr_gb.count < 1) return;
		// (a known consumer only: a dispatch before the G-buffer pass must not take the frame's copy)
		if (std::find(g_ssr_gb.consumers.begin(), g_ssr_gb.consumers.end(), cs_hash) == g_ssr_gb.consumers.end()) return;
		if (g_ssr_gb.targets[0].view.format != vk::Format::eR32Uint) return SsrCaptureSkip("RT0 not R32_UINT", cs_hash);
		if (flip > g_ssr_gb.flip + 1) return SsrCaptureSkip("no G-buffer pass this frame or the last", cs_hash);
		addresses = g_ssr_gb.packed_addresses;
		width = g_ssr_gb.targets[0].width, height = g_ssr_gb.targets[0].height;
	}
	const TextureBinding* binding = nullptr;
	for (const auto& b: images)
		if (b.desc.info.data.address != 0 && (b.desc.info.data.address == addresses[0] || b.desc.info.data.address == addresses[1])) {
			binding = &b;
			break;
		}
	if (binding == nullptr) return SsrCaptureSkip("the consumer binds neither G-buffer address", cs_hash);
	if (!binding->image_id) return SsrCaptureSkip("binding without a cache image", cs_hash);
	const uint64_t address = binding->desc.info.data.address;
	auto& image = cache.GetImage(binding->image_id);
	if (image.backing.image == nullptr) return SsrCaptureSkip("no backing image", cs_hash);
	if (image.backing.format != vk::Format::eR32Uint) return SsrCaptureSkip("backing not R32_UINT", cs_hash);
	if (image.backing.layers < 4) return SsrCaptureSkip("backing has < 4 layers", cs_hash);
	if (image.backing.extent.width < width || image.backing.extent.height < height)
		return SsrCaptureSkip("backing smaller than the G-buffer", cs_hash);
	const vk::Extent2D extent {width, height};
	// (copies replaced at a size change: destroyed once no frame in flight can read them)
	for (size_t i = 0; i < g_ssr.retired_copies.size();) {
		auto& r = g_ssr.retired_copies[i];
		if (flip < r.flip + 8) {
			++i;
			continue;
		}
		for (auto& v: r.views)
			if (v != nullptr) g_graphics_ctx->device.destroyImageView(v, nullptr);
		if (r.image != VK_NULL_HANDLE) vmaDestroyImage(g_graphics_ctx->allocator, r.image, r.mem);
		r = g_ssr.retired_copies.back();
		g_ssr.retired_copies.pop_back();
	}
	if (g_ssr.copy_image != VK_NULL_HANDLE && g_ssr.copy_extent != extent) {
		// A new size on 30 consecutive attempts (a resolution change, not a second view in the frame): a new copy.
		if (g_ssr.resize_to != extent) g_ssr.resize_to = extent, g_ssr.resize_seen = 0;
		if (++g_ssr.resize_seen < 30) return SsrCaptureSkip("size changed", cs_hash);
		std::printf("SSR: G-buffer copy resized %ux%u -> %ux%u\n", g_ssr.copy_extent.width, g_ssr.copy_extent.height, width, height);
		std::fflush(stdout);
		SsrPass::RetiredCopy r;
		r.image = g_ssr.copy_image, r.mem = g_ssr.copy_mem, r.flip = flip;
		for (uint32_t i = 0; i < 4; ++i) r.views[i] = g_ssr.copy_views[i], g_ssr.copy_views[i] = nullptr;
		g_ssr.retired_copies.push_back(r);
		g_ssr.copy_image  = VK_NULL_HANDLE;
		g_ssr.copy_mem    = nullptr;
		g_ssr.copy_layout = vk::ImageLayout::eUndefined;
		g_ssr.copy_extent = {};
	}
	g_ssr.resize_seen = 0;
	if (g_ssr.copy_image == VK_NULL_HANDLE) {
		g_ssr.copy_failed = true;
		VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		create.imageType   = VK_IMAGE_TYPE_2D;
		create.format      = VK_FORMAT_R32_UINT;
		create.extent      = {width, height, 1};
		create.mipLevels   = 1;
		create.arrayLayers = 4;
		create.samples     = VK_SAMPLE_COUNT_1_BIT;
		create.tiling      = VK_IMAGE_TILING_OPTIMAL;
		create.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo allocate {};
		allocate.usage = VMA_MEMORY_USAGE_AUTO;
		if (vmaCreateImage(g_graphics_ctx->allocator, &create, &allocate, &g_ssr.copy_image, &g_ssr.copy_mem, nullptr) != VK_SUCCESS) {
			g_ssr.copy_image = VK_NULL_HANDLE;
			std::printf("SSR: G-buffer copy image could not be created\n");
			return;
		}
		for (uint32_t i = 0; i < 4; ++i) {
			vk::ImageViewCreateInfo view_info {};
			view_info.image            = vk::Image(g_ssr.copy_image);
			view_info.viewType         = vk::ImageViewType::e2D;
			view_info.format           = vk::Format::eR32Uint;
			view_info.subresourceRange = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, 0, 1, i, 1};
			if (g_graphics_ctx->device.createImageView(&view_info, nullptr, &g_ssr.copy_views[i]) != vk::Result::eSuccess) return;
		}
		g_ssr.copy_extent = extent;
		g_ssr.copy_failed = false;
	}
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
	LocalVulkanRecording::Drain();
#endif
	const ImageSubresourceRange range {0, 1, 0, 4};
	image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, range, command);
	const auto copy_barrier = [&](vk::ImageLayout from, vk::ImageLayout to, vk::PipelineStageFlags2 src_stage,
	                              vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
		vk::ImageMemoryBarrier2 barrier {};
		barrier.srcStageMask        = src_stage;
		barrier.srcAccessMask       = src_access;
		barrier.dstStageMask        = dst_stage;
		barrier.dstAccessMask       = dst_access;
		barrier.oldLayout           = from;
		barrier.newLayout           = to;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image               = vk::Image(g_ssr.copy_image);
		barrier.subresourceRange    = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 4};
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers    = &barrier;
		command.pipelineBarrier2(dependency);
	};
	copy_barrier(g_ssr.copy_layout, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eAllCommands,
	             vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
	vk::ImageCopy copy {};
	copy.srcSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0, 0, 4};
	copy.dstSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0, 0, 4};
	copy.extent         = vk::Extent3D {width, height, 1};
	command.copyImage(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, vk::Image(g_ssr.copy_image),
	                  vk::ImageLayout::eTransferDstOptimal, 1, &copy);
	copy_barrier(vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eTransfer,
	             vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
	g_ssr.copy_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
	const vk::ImageLayout back = binding->layout == vk::ImageLayout::eUndefined ? vk::ImageLayout::eGeneral : binding->layout;
	image.Transit(back, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, range, command);
	SsrCaptureDepth(cache, command, cs_hash, images, width, height, flip);
	g_ssr.copy_flip.store(flip, std::memory_order_release);
	g_ssr.copies.fetch_add(1, std::memory_order_relaxed);
	static bool said = false;
	if (!said) {
		said = true;
		std::printf("SSR: G-buffer copied before the deferred lighting (%ux%u x4 R32_UINT @0x%llx)\n", width, height,
		            static_cast<unsigned long long>(address));
		std::fflush(stdout);
	}
}

// KYTY_GBUF_VIEW / KYTY_GBUF_TAP (gbuf-view.h, refl7): after a dispatch (renderCompute.cpp, outside a compute chain's
// middle): a dispatch reading the G-buffer logs once what it writes (its outputs: which image holds the game's SSR or
// specular IBL); the KYTY_GBUF_TAP dispatch's written image is copied for the tap view (every time it runs: the last
// one in the frame stays).
namespace {
struct GbufTapState {
	bool                  parsed = false, failed = false;
	uint64_t              hash = 0;
	int                   slot = -1;
	VkImage               image = VK_NULL_HANDLE;
	VmaAllocation         mem   = nullptr;
	vk::ImageView         view  = nullptr;
	vk::Extent2D          extent {};
	vk::Format            format = vk::Format::eUndefined;
	vk::ImageLayout       layout = vk::ImageLayout::eUndefined;
	std::atomic<uint64_t> flip {~0ull};
	std::vector<std::pair<uint64_t, uint64_t>> logged; // (hash, written mask) logged already
	std::vector<SsrPass::RetiredCopy>          retired;
} g_tap;

bool GbufTapFormat(vk::Format f) {
	switch (f) {
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR16Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eB10G11R11UfloatPack32:
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eR8Unorm:
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR16Unorm:
		case vk::Format::eR16G16B16A16Unorm: return true;
		default: return false;
	}
}
} // namespace

void GbufTapAfterDispatch(GraphicContext& graphics, TextureCache& cache, vk::CommandBuffer command, uint64_t cs_hash,
                          std::span<const TextureBinding> images, uint64_t written_mask) {
	if (!GbufViewWanted()) return;
	if (g_graphics_ctx == nullptr) g_graphics_ctx = &graphics;
	if (!g_tap.parsed) {
		g_tap.parsed = true;
		if (const char* v = std::getenv("KYTY_GBUF_TAP"); v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0) {
			unsigned long long h    = 0;
			int                slot = -1;
			if (std::sscanf(v, "%llx:%d", &h, &slot) >= 1) g_tap.hash = h, g_tap.slot = slot;
			std::printf("GBUF: tap CS %016llx slot %s\n", h, slot >= 0 ? std::to_string(slot).c_str() : "(first written)");
			std::fflush(stdout);
		}
	}
	// (what a G-buffer reader writes: logged once per shader and written set)
	{
		std::array<uint64_t, 2> packed {};
		{
			std::lock_guard lock(g_ssr_gb.mutex);
			packed = g_ssr_gb.packed_addresses;
		}
		bool reads = false;
		for (const auto& b: images)
			if (b.desc.info.data.address != 0 && (b.desc.info.data.address == packed[0] || b.desc.info.data.address == packed[1])) reads = true;
		const auto key = std::make_pair(cs_hash, written_mask);
		if ((reads || cs_hash == g_tap.hash) && g_tap.logged.size() < 128 &&
		    std::find(g_tap.logged.begin(), g_tap.logged.end(), key) == g_tap.logged.end()) {
			g_tap.logged.push_back(key);
			std::string line;
			char        buf[200];
			for (size_t i = 0; i < images.size() && i < 32; ++i) {
				const auto& b = images[i];
				if (!b.image_id) continue;
				auto& im = cache.GetImage(b.image_id);
				std::snprintf(buf, sizeof(buf), " %zu%s %s %ux%ux%u mip %u layer %u @0x%llx;", i, (written_mask >> i) & 1u ? " WRITES" : "",
				              SsrFormatName(im.backing.format), im.backing.extent.width, im.backing.extent.height, im.backing.layers,
				              b.desc.view_info.base_level, b.desc.view_info.base_layer, static_cast<unsigned long long>(b.desc.info.data.address));
				line += buf;
			}
			std::printf("GBUF: %s CS %016llx images:%s\n", reads ? "G-buffer reader" : "tap", static_cast<unsigned long long>(cs_hash), line.c_str());
			std::fflush(stdout);
		}
	}
	if (g_tap.failed || g_tap.hash == 0 || cs_hash != g_tap.hash) return;
	const uint64_t flip = g_flips_taken.load(std::memory_order_relaxed);
	int slot = g_tap.slot;
	if (slot < 0)
		for (size_t i = 0; i < images.size() && i < 64; ++i)
			if ((written_mask >> i) & 1u) {
				slot = static_cast<int>(i);
				break;
			}
	if (slot < 0 || static_cast<size_t>(slot) >= images.size() || !images[slot].image_id) return;
	const auto& binding = images[slot];
	auto&       image   = cache.GetImage(binding.image_id);
	if (image.backing.image == nullptr) return;
	const vk::Format fmt = image.backing.format;
	if (!GbufTapFormat(fmt)) {
		g_tap.failed = true;
		std::printf("GBUF: tap slot %d is %s: not a color format the view can show\n", slot, SsrFormatName(fmt));
		std::fflush(stdout);
		return;
	}
	const uint32_t   level = binding.desc.view_info.base_level, layer = binding.desc.view_info.base_layer;
	const vk::Extent2D extent {std::max(1u, image.backing.extent.width >> level), std::max(1u, image.backing.extent.height >> level)};
	for (size_t i = 0; i < g_tap.retired.size();) {
		auto& r = g_tap.retired[i];
		if (flip < r.flip + 8) {
			++i;
			continue;
		}
		if (r.views[0] != nullptr) graphics.device.destroyImageView(r.views[0], nullptr);
		if (r.image != VK_NULL_HANDLE) vmaDestroyImage(graphics.allocator, r.image, r.mem);
		r = g_tap.retired.back();
		g_tap.retired.pop_back();
	}
	if (g_tap.image != VK_NULL_HANDLE && (g_tap.extent != extent || g_tap.format != fmt)) {
		SsrPass::RetiredCopy r;
		r.image = g_tap.image, r.mem = g_tap.mem, r.flip = flip, r.views[0] = g_tap.view;
		g_tap.retired.push_back(r);
		g_tap.image = VK_NULL_HANDLE, g_tap.mem = nullptr, g_tap.view = nullptr, g_tap.layout = vk::ImageLayout::eUndefined;
	}
	if (g_tap.image == VK_NULL_HANDLE) {
		VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		create.imageType   = VK_IMAGE_TYPE_2D;
		create.format      = static_cast<VkFormat>(fmt);
		create.extent      = {extent.width, extent.height, 1};
		create.mipLevels   = 1;
		create.arrayLayers = 1;
		create.samples     = VK_SAMPLE_COUNT_1_BIT;
		create.tiling      = VK_IMAGE_TILING_OPTIMAL;
		create.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo allocate {};
		allocate.usage = VMA_MEMORY_USAGE_AUTO;
		if (vmaCreateImage(graphics.allocator, &create, &allocate, &g_tap.image, &g_tap.mem, nullptr) != VK_SUCCESS) {
			g_tap.image  = VK_NULL_HANDLE;
			g_tap.failed = true;
			return;
		}
		vk::ImageViewCreateInfo view_info {};
		view_info.image            = vk::Image(g_tap.image);
		view_info.viewType         = vk::ImageViewType::e2D;
		view_info.format           = fmt;
		view_info.subresourceRange = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		if (graphics.device.createImageView(&view_info, nullptr, &g_tap.view) != vk::Result::eSuccess) {
			g_tap.view   = nullptr;
			g_tap.failed = true;
			return;
		}
		g_tap.extent = extent, g_tap.format = fmt;
		std::printf("GBUF: tap image %ux%u %s from CS %016llx slot %d (mip %u layer %u @0x%llx)\n", extent.width, extent.height,
		            SsrFormatName(fmt), static_cast<unsigned long long>(cs_hash), slot, level, layer,
		            static_cast<unsigned long long>(binding.desc.info.data.address));
		std::fflush(stdout);
	}
	const ImageSubresourceRange range {level, 1, layer, 1};
	{
		// (the dispatch's storage writes before the copy, also inside a compute chain, whose dispatches have no barriers)
		vk::MemoryBarrier2 mb {};
		mb.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
		mb.srcAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderStorageWrite;
		mb.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		mb.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &mb;
		command.pipelineBarrier2(dependency);
	}
	image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, range, command);
	const auto barrier = [&](vk::ImageLayout from, vk::ImageLayout to, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
	                         vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
		vk::ImageMemoryBarrier2 b {};
		b.srcStageMask        = src_stage;
		b.srcAccessMask       = src_access;
		b.dstStageMask        = dst_stage;
		b.dstAccessMask       = dst_access;
		b.oldLayout           = from;
		b.newLayout           = to;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image               = vk::Image(g_tap.image);
		b.subresourceRange    = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers    = &b;
		command.pipelineBarrier2(dependency);
	};
	barrier(g_tap.layout, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
	        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
	vk::ImageCopy copy {};
	copy.srcSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, level, layer, 1};
	copy.dstSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.extent         = vk::Extent3D {extent.width, extent.height, 1};
	command.copyImage(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, vk::Image(g_tap.image), vk::ImageLayout::eTransferDstOptimal, 1,
	                  &copy);
	barrier(vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
	g_tap.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
	const vk::ImageLayout back = binding.layout == vk::ImageLayout::eUndefined ? vk::ImageLayout::eGeneral : binding.layout;
	image.Transit(back, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, range, command);
	g_tap.flip.store(flip, std::memory_order_release);
}

bool GbufTapGet(GbufTapImage& out) {
	const uint64_t taken = g_tap.flip.load(std::memory_order_acquire);
	const uint64_t flips = g_flips_taken.load(std::memory_order_relaxed);
	if (taken == ~0ull || flips > taken + 1 || g_tap.view == nullptr) return false;
	out.view    = g_tap.view;
	out.extent  = g_tap.extent;
	out.integer = false;
	return true;
}

// (refl8) KYTY_RT_REFLECTIONS' guide (rt-reflections.h RtReflGuide): the game's own SSR result, a half-resolution
// RGBA16F input of the deferred lighting / specular resolve (default CS 9fb5cc274b5e4f82: reads the packed G-buffer,
// both depths, a 256x256 RG16F BRDF LUT and this image; writes the full-resolution lit RGBA16F), copied after that
// dispatch every frame. Its bindings are logged once, and the dispatch that writes the image (the game's SSR pass) once.
namespace {
struct GuideState {
	bool                  parsed = false, failed = false, logged = false, producer_logged = false;
	uint64_t              hash = 0x9fb5cc274b5e4f82ull;
	int                   slot = -1, used_slot = -1;
	uint64_t              address = 0, producer = 0;
	VkImage               image = VK_NULL_HANDLE;
	VmaAllocation         mem   = nullptr;
	vk::ImageView         view  = nullptr;
	vk::Extent2D          extent {};
	vk::Format            format = vk::Format::eUndefined;
	vk::ImageLayout       layout = vk::ImageLayout::eUndefined;
	std::atomic<uint64_t> flip {~0ull};
	uint32_t              misses = 0;
	bool                  miss_logged = false, first_copy_logged = false;
	std::atomic<const char*> status {"the lighting (guide CS) not seen yet"};
	std::vector<SsrPass::RetiredCopy> retired;
} g_guide;

vk::Extent2D BindingExtent(TextureCache& cache, const TextureBinding& b) {
	auto&          im    = cache.GetImage(b.image_id);
	const uint32_t level = b.desc.view_info.base_level;
	return {std::max(1u, im.backing.extent.width >> level), std::max(1u, im.backing.extent.height >> level)};
}
} // namespace

void RtReflGuideAfterDispatch(GraphicContext& graphics, TextureCache& cache, vk::CommandBuffer command, uint64_t cs_hash,
                              std::span<const TextureBinding> images, uint64_t written_mask) {
	if (!RtReflGuideWanted()) return;
	if (!g_guide.parsed) {
		g_guide.parsed = true;
		if (const char* v = std::getenv("KYTY_RT_REFLECTIONS_GUIDE_CS"); v != nullptr && *v != '\0') {
			unsigned long long h = 0;
			if (std::sscanf(v, "%llx", &h) == 1 && h != 0) g_guide.hash = h;
		}
		if (const char* v = std::getenv("KYTY_RT_REFLECTIONS_GUIDE_SLOT"); v != nullptr && *v != '\0') g_guide.slot = std::atoi(v);
	}
	// (the game's SSR pass: the dispatch that writes the guide's image; once its address is known)
	if (g_guide.address != 0 && !g_guide.producer_logged && cs_hash != g_guide.hash) {
		for (size_t i = 0; i < images.size() && i < 64; ++i) {
			if (((written_mask >> i) & 1u) == 0 || !images[i].image_id || images[i].desc.info.data.address != g_guide.address) continue;
			g_guide.producer_logged = true;
			g_guide.producer        = cs_hash;
			std::string line;
			char        buf[200];
			for (size_t k = 0; k < images.size() && k < 32; ++k) {
				const auto& b = images[k];
				if (!b.image_id) continue;
				auto&              im = cache.GetImage(b.image_id);
				const vk::Extent2D e  = BindingExtent(cache, b);
				std::snprintf(buf, sizeof(buf), " %zu%s %s %ux%u @0x%llx;", k, (written_mask >> k) & 1u ? " WRITES" : "",
				              SsrFormatName(im.backing.format), e.width, e.height, static_cast<unsigned long long>(b.desc.info.data.address));
				line += buf;
			}
			std::printf("RT reflections: guide image @0x%llx written by CS %016llx (the game's SSR pass) slot %zu; its images:%s\n",
			            static_cast<unsigned long long>(g_guide.address), static_cast<unsigned long long>(cs_hash), i, line.c_str());
			std::fflush(stdout);
			break;
		}
	}
	if (g_guide.failed || cs_hash != g_guide.hash) return;
	// The slot. (refl8 fix, combo19) The first version took the first RGBA16F input at half the size of the first
	// image the lighting WRITES; that held only on the very first run (a 2560x1440 lighting writing 2560x1440, guide
	// 1280x720). In play the PS5 Pro path renders 1280x720 and the lighting writes a 640x720 RGBA16F (half width), its
	// guide input is 640x360: 640 is not half of 640, no slot ever again ("ran 600 times without a usable slot", the
	// guide copied in 0 frames). Now, in order: KYTY_RT_REFLECTIONS_GUIDE_SLOT; the slot found before when it is still
	// an RGBA float input no larger than the render size; the first RGBA16F input at half the render size (the largest
	// depth binding: the G-buffer's size); the old rule (half the written image).
	vk::Extent2D out_extent {}, ref_extent {};
	for (size_t i = 0; i < images.size() && i < 64; ++i) {
		if (!images[i].image_id) continue;
		const vk::Extent2D e = BindingExtent(cache, images[i]);
		if (((written_mask >> i) & 1u) != 0 && out_extent.width == 0) out_extent = e;
		if (SsrIsDepthFormat(cache.GetImage(images[i].image_id).backing.format) &&
		    uint64_t {e.width} * e.height > uint64_t {ref_extent.width} * ref_extent.height)
			ref_extent = e;
	}
	if (ref_extent.width == 0) ref_extent = out_extent;
	const auto is_rgba_float = [&](size_t i) {
		if (i >= images.size() || !images[i].image_id || ((written_mask >> i) & 1u) != 0) return false;
		const vk::Format f = cache.GetImage(images[i].image_id).backing.format;
		return f == vk::Format::eR16G16B16A16Sfloat || f == vk::Format::eR32G32B32A32Sfloat;
	};
	const auto near_half = [](uint32_t v, uint32_t full) { return v + 1 >= (full + 1) / 2 && v <= (full + 1) / 2 + 1; };
	const auto half_of   = [&](size_t i, vk::Extent2D full) {
		if (full.width == 0 || !is_rgba_float(i)) return false;
		const vk::Extent2D e = BindingExtent(cache, images[i]);
		return near_half(e.width, full.width) && near_half(e.height, full.height);
	};
	int         slot = g_guide.slot;
	const char* rule = "KYTY_RT_REFLECTIONS_GUIDE_SLOT";
	if (slot < 0 && g_guide.used_slot >= 0 && is_rgba_float(static_cast<size_t>(g_guide.used_slot))) {
		const vk::Extent2D e = BindingExtent(cache, images[g_guide.used_slot]);
		// (no larger than the render size, no smaller than a quarter of it: not some other small RGBA float input)
		if (ref_extent.width == 0 || (e.width <= ref_extent.width && e.height <= ref_extent.height && e.width * 4 >= ref_extent.width &&
		                              e.height * 4 >= ref_extent.height))
			slot = g_guide.used_slot, rule = "the slot found before";
	}
	for (size_t i = 0; slot < 0 && i < images.size() && i < 64; ++i)
		if (half_of(i, ref_extent)) slot = static_cast<int>(i), rule = "half the render (depth) size";
	for (size_t i = 0; slot < 0 && i < images.size() && i < 64; ++i)
		if (half_of(i, out_extent)) slot = static_cast<int>(i), rule = "half the written image's size";
	const auto bindings_line = [&] {
		std::string line;
		char        buf[200];
		for (size_t i = 0; i < images.size() && i < 32; ++i) {
			const auto& b = images[i];
			if (!b.image_id) continue;
			auto&              im = cache.GetImage(b.image_id);
			const vk::Extent2D e  = BindingExtent(cache, b);
			std::snprintf(buf, sizeof(buf), " %zu%s %s %ux%u @0x%llx;", i, (written_mask >> i) & 1u ? " WRITES" : "",
			              SsrFormatName(im.backing.format), e.width, e.height, static_cast<unsigned long long>(b.desc.info.data.address));
			line += buf;
		}
		return line;
	};
	if (!g_guide.logged) {
		g_guide.logged = true;
		std::printf("RT reflections: guide CS %016llx images:%s (render size %ux%u, writes %ux%u) -> guide slot %s (%s)\n",
		            static_cast<unsigned long long>(g_guide.hash), bindings_line().c_str(), ref_extent.width, ref_extent.height,
		            out_extent.width, out_extent.height, slot >= 0 ? std::to_string(slot).c_str() : "none",
		            slot >= 0 ? rule : "no RGBA float input at half the render size: the material mask until one is found");
		std::fflush(stdout);
	}
	if (slot < 0 || static_cast<size_t>(slot) >= images.size() || !images[slot].image_id) {
		++g_guide.misses;
		if (!g_guide.miss_logged) {
			// (one-shot: why this run found no guide, with what it bound)
			g_guide.miss_logged = true;
			std::printf("RT reflections: guide CS %016llx run without a usable slot (first time; %s): render size %ux%u, writes %ux%u, "
			            "slot found before %d; images:%s\n",
			            static_cast<unsigned long long>(g_guide.hash),
			            slot >= 0 ? "the given slot is not bound" : "no RGBA float input at half the render or written size",
			            ref_extent.width, ref_extent.height, out_extent.width, out_extent.height, g_guide.used_slot, bindings_line().c_str());
			std::fflush(stdout);
		}
		g_guide.status.store("the lighting ran without a usable slot", std::memory_order_relaxed);
		return;
	}
	if (slot != g_guide.used_slot && g_guide.used_slot >= 0) {
		std::printf("RT reflections: guide slot %d -> %d (%s)\n", g_guide.used_slot, slot, rule);
		std::fflush(stdout);
	}
	const auto& binding = images[slot];
	auto&       image   = cache.GetImage(binding.image_id);
	if (image.backing.image == nullptr) {
		g_guide.status.store("the guide slot's image has no backing", std::memory_order_relaxed);
		return;
	}
	const vk::Format fmt = image.backing.format;
	if (fmt != vk::Format::eR16G16B16A16Sfloat && fmt != vk::Format::eR32G32B32A32Sfloat) {
		g_guide.failed = true;
		g_guide.status.store("the given guide slot is not RGBA float", std::memory_order_relaxed);
		std::printf("RT reflections: guide slot %d is %s, not RGBA float: the material mask instead\n", slot, SsrFormatName(fmt));
		std::fflush(stdout);
		return;
	}
	const uint64_t     flip  = g_flips_taken.load(std::memory_order_relaxed);
	const uint32_t     level = binding.desc.view_info.base_level, layer = binding.desc.view_info.base_layer;
	const vk::Extent2D extent = BindingExtent(cache, binding);
	for (size_t i = 0; i < g_guide.retired.size();) {
		auto& r = g_guide.retired[i];
		if (flip < r.flip + 8) {
			++i;
			continue;
		}
		if (r.views[0] != nullptr) graphics.device.destroyImageView(r.views[0], nullptr);
		if (r.image != VK_NULL_HANDLE) vmaDestroyImage(graphics.allocator, r.image, r.mem);
		r = g_guide.retired.back();
		g_guide.retired.pop_back();
	}
	if (g_guide.image != VK_NULL_HANDLE && (g_guide.extent != extent || g_guide.format != fmt)) {
		SsrPass::RetiredCopy r;
		r.image = g_guide.image, r.mem = g_guide.mem, r.flip = flip, r.views[0] = g_guide.view;
		g_guide.retired.push_back(r);
		g_guide.image = VK_NULL_HANDLE, g_guide.mem = nullptr, g_guide.view = nullptr, g_guide.layout = vk::ImageLayout::eUndefined;
		g_guide.flip.store(~0ull, std::memory_order_release);
	}
	if (g_guide.image == VK_NULL_HANDLE) {
		VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		create.imageType   = VK_IMAGE_TYPE_2D;
		create.format      = static_cast<VkFormat>(fmt);
		create.extent      = {extent.width, extent.height, 1};
		create.mipLevels   = 1;
		create.arrayLayers = 1;
		create.samples     = VK_SAMPLE_COUNT_1_BIT;
		create.tiling      = VK_IMAGE_TILING_OPTIMAL;
		create.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo allocate {};
		allocate.usage = VMA_MEMORY_USAGE_AUTO;
		if (vmaCreateImage(graphics.allocator, &create, &allocate, &g_guide.image, &g_guide.mem, nullptr) != VK_SUCCESS) {
			g_guide.image  = VK_NULL_HANDLE;
			g_guide.failed = true;
			g_guide.status.store("the guide copy image could not be created", std::memory_order_relaxed);
			return;
		}
		vk::ImageViewCreateInfo view_info {};
		view_info.image            = vk::Image(g_guide.image);
		view_info.viewType         = vk::ImageViewType::e2D;
		view_info.format           = fmt;
		view_info.subresourceRange = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		if (graphics.device.createImageView(&view_info, nullptr, &g_guide.view) != vk::Result::eSuccess) {
			g_guide.view   = nullptr;
			g_guide.failed = true;
			g_guide.status.store("the guide copy view could not be created", std::memory_order_relaxed);
			return;
		}
		g_guide.extent = extent, g_guide.format = fmt, g_guide.used_slot = slot;
		g_guide.address = binding.desc.info.data.address, g_guide.producer_logged = false;
		static uint32_t created = 0;
		if (++created <= 8) std::printf("RT reflections: guide image %ux%u %s from CS %016llx slot %d (mip %u layer %u @0x%llx; the lighting writes %ux%u)\n",
		            extent.width, extent.height, SsrFormatName(fmt), static_cast<unsigned long long>(cs_hash), slot, level, layer,
		            static_cast<unsigned long long>(g_guide.address), out_extent.width, out_extent.height);
		std::fflush(stdout);
	}
	const ImageSubresourceRange range {level, 1, layer, 1};
	{
		// (whatever wrote the image, and this dispatch, before the copy; also inside a compute chain)
		vk::MemoryBarrier2 mb {};
		mb.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		mb.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
		mb.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		mb.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &mb;
		command.pipelineBarrier2(dependency);
	}
	image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, range, command);
	const auto barrier = [&](vk::ImageLayout from, vk::ImageLayout to, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
	                         vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
		vk::ImageMemoryBarrier2 b {};
		b.srcStageMask        = src_stage;
		b.srcAccessMask       = src_access;
		b.dstStageMask        = dst_stage;
		b.dstAccessMask       = dst_access;
		b.oldLayout           = from;
		b.newLayout           = to;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image               = vk::Image(g_guide.image);
		b.subresourceRange    = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers    = &b;
		command.pipelineBarrier2(dependency);
	};
	barrier(g_guide.layout, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
	        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
	vk::ImageCopy copy {};
	copy.srcSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, level, layer, 1};
	copy.dstSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.extent         = vk::Extent3D {extent.width, extent.height, 1};
	command.copyImage(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, vk::Image(g_guide.image), vk::ImageLayout::eTransferDstOptimal,
	                  1, &copy);
	barrier(vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);
	g_guide.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
	const vk::ImageLayout back = binding.layout == vk::ImageLayout::eUndefined ? vk::ImageLayout::eGeneral : binding.layout;
	image.Transit(back, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, range, command);
	g_guide.used_slot = slot;
	g_guide.status.store("copied", std::memory_order_relaxed);
	g_guide.flip.store(flip, std::memory_order_release);
	if (!g_guide.first_copy_logged) {
		g_guide.first_copy_logged = true;
		std::printf("RT reflections: guide copied (first time, flip %llu): %ux%u from CS %016llx slot %d @0x%llx, after %u runs without a slot\n",
		            static_cast<unsigned long long>(flip), extent.width, extent.height, static_cast<unsigned long long>(cs_hash), slot,
		            static_cast<unsigned long long>(binding.desc.info.data.address), g_guide.misses);
		std::fflush(stdout);
	}
}

const char* RtReflGuideStatus() { return g_guide.status.load(std::memory_order_relaxed); }

bool RtReflGetGuide(RtReflGuide& out) {
	const uint64_t taken = g_guide.flip.load(std::memory_order_acquire);
	const uint64_t flips = g_flips_taken.load(std::memory_order_relaxed);
	if (taken == ~0ull || flips > taken + 1 || g_guide.view == nullptr) {
		const char* st = g_guide.status.load(std::memory_order_relaxed);
		if (taken != ~0ull && flips > taken + 1 && st != nullptr && std::strcmp(st, "copied") == 0)
			g_guide.status.store("the last copy is older than one flip (the lighting did not run)", std::memory_order_relaxed);
		return false;
	}
	out.view     = g_guide.view;
	out.extent   = g_guide.extent;
	out.cs       = g_guide.hash;
	out.producer = g_guide.producer;
	out.slot     = g_guide.used_slot;
	return true;
}

uint32_t SsrPackedTargets(std::array<SsrTarget, 4>& targets) {
	std::lock_guard lock(g_ssr_gb.mutex);
	if (!g_ssr_gb.valid || g_flips_taken.load(std::memory_order_relaxed) > g_ssr_gb.flip + 4) return 0;
	uint32_t n = 0;
	for (uint32_t i = 0; i < g_ssr_gb.count && n < 4; ++i) {
		const auto& t = g_ssr_gb.targets[i];
		if (t.view.format != vk::Format::eR32Uint || t.address == 0) break; // (the packed targets lead: RT0..RT3)
		targets[n++] = t;
	}
	return n;
}

bool SsrMaterial(SsrTarget& target, int& channel, bool& gloss) {
	const auto& choice = SsrChoice();
	if (choice.slot < 0) return false;
	std::lock_guard lock(g_ssr_gb.mutex);
	if (!g_ssr_gb.valid || static_cast<uint32_t>(choice.slot) >= g_ssr_gb.count) return false;
	if (g_flips_taken.load(std::memory_order_relaxed) > g_ssr_gb.flip + 600) return false; // (not seen for ~10 s)
	const auto& t = g_ssr_gb.targets[choice.slot];
	if (!SsrFormatUsable(t.view.format) || t.address == 0) return false;
	target  = t;
	channel = choice.channel;
	gloss   = choice.gloss;
	return true;
}

// KYTY_RT_REFLECTIONS (rt-reflections.cpp): the G-buffer copy and the camera.
bool RtReflGetGbuffer(RtReflGbuffer& out) {
	const uint64_t copied = g_ssr.copy_flip.load(std::memory_order_acquire);
	const uint64_t flips  = g_flips_taken.load(std::memory_order_relaxed);
	if (copied == ~0ull || flips > copied + 1 || g_ssr.copy_image == VK_NULL_HANDLE) return false;
	for (uint32_t i = 0; i < 4; ++i) {
		if (g_ssr.copy_views[i] == nullptr) return false;
		out.views[i] = g_ssr.copy_views[i];
	}
	out.extent = g_ssr.copy_extent;
	if (g_ssr.depth_view != nullptr && g_ssr.depth_flip == copied && g_ssr.depth_extent == g_ssr.copy_extent) {
		out.depth        = g_ssr.depth_view;
		out.depth_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
	}
	return true;
}

uint64_t RtReflCopies() { return g_ssr.copies.load(std::memory_order_relaxed); }

void RtReflGetCamera(const std::array<float, 16>& view_to_clip, RtReflCamera& out) {
	out.rows     = SsrViewRotation(view_to_clip, out.view_rows);
	out.up_valid = SsrUpInView(view_to_clip, out.up);
	out.prev     = false;
	std::array<float, 16> vp {}, pv {};
	{
		std::lock_guard lock(g_ssr_gb.mutex);
		if (!g_ssr_gb.have_vp || !g_ssr_gb.have_prev || g_flips_taken.load(std::memory_order_relaxed) > g_ssr_gb.vp_flip + 2) return;
		vp = g_ssr_gb.view_proj, pv = g_ssr_gb.prev_view_proj;
	}
	// Current clip -> previous clip = VP_prev * VP^-1 (row-major, column vectors: clip = M * world).
	std::array<double, 16> inv {};
	if (!Inverse4(vp, inv)) return;
	std::array<float, 16> c {};
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j) {
			double d = 0.0;
			for (int k = 0; k < 4; ++k) d += static_cast<double>(pv[i * 4 + k]) * inv[k * 4 + j];
			if (!std::isfinite(d)) return;
			c[i * 4 + j] = static_cast<float>(d);
		}
	out.clip_to_prev = c;
	out.prev         = true;
}

namespace {
// KYTY_RT_SHADOWS=0|1|2|3 ("Sombras RT": off / Low / High / Ultra; cshadow.comp; KYTY_CONTACT_SHADOWS, the old name,
// is read when it is unset): screen-space ray-traced sun shadows on the PS5 Pro path's linear HDR color (before the
// game's exposure, the upscaler and its TAA), recorded first in Upscale (before SSGI / GTAO / SSR). 1: half-res trace,
// 12 steps, temporal + a-trous 1, 2; 2: full res, 20 steps, a-trous 1, 2, 4; 3: full res, 32 steps, rays x1.25,
// slower temporal alpha. The composite only darkens: color * (1 - strength * fade * (1 - visibility)), nothing on the
// sky, faded out between KYTY_RT_SHADOWS_FADE (100 m) and twice that. KYTY_RT_SHADOWS_STRENGTH (0.75, 0..1),
// KYTY_RT_SHADOWS_LENGTH (3.0: world ray length in m, 0.25..10), KYTY_RT_SHADOWS_HOLD (20 s: a sun unseen is kept).
// Every 10 s a "RTSHADOW: stats" line: the share of pixels shadowed, mean occlusion, their depth, the sun's K and the
// overcast factor, the GPU time.
// (10-24, second round) Hard near-black shards on far cliffs (Boletaria, overcast): rays 3.75 m long at 50 m with a
// 1 m thickness hit the next flat facet of the cliff's mesh. Now: rays, strength and thickness scale with distance
// (generous within 6 m, conservative beyond 40 m), a least occluder distance over the receiver's plane, a ~3 degree
// penumbra cone jittered per frame and a wider, softer a-trous (no binary edges), a cap of 75 % darkening; pixels the
// game already shadows (dark against the same surface around them, on the scale its sun gives) get no second shadow;
// and an overcast factor (KYTY_RT_SHADOWS_OVERCAST=0 turns it off): the light probe compares the luminance of what
// faces the sun with what faces away (K, the sun-lit to ambient ratio): K <= 1.3 scales the strength to 0.15, full from 2.4.
//
// Why the first version looked wrong: its depth reconstruction took the render's row 0 as NDC y -1, while the game's
// viewport (GNM, yscale < 0) puts NDC y +1 there: the scene it marched was mirrored vertically against the light it
// was given (shadows cast "upward", lit floors left alone, a lighting vote picking a flipped or view-space axis to
// compensate); its lit mask and light vote used luma thresholds that mean nothing in the pre-exposure HDR range; its
// rays reached 2 m (doubling the game's own shadow-map shadows, grey far away) and its fade followed vote locks
// (popping).
//
// The sun's direction:
//  1. OnShadowConsumer (renderCompute.cpp): a dispatch that samples a square, layered (>= 4), depth-like image of
//     >= 512 px (the sun cascades' shadow-map array; Demon's Souls: 1024x1024x80 at 0x102a400000) is a shadow
//     consumer (the deferred sun lighting / shadow resolve). Its flattened SRT and its constant buffers (<= 64 KiB, the
//     first 4 KiB) are searched for orthographic matrices: 4x4 (either major) with a (0, 0, 0, 1) projective row, or
//     3x4 rows, whose 3x3 rows are mutually orthogonal (|cos| < 0.01), x and y scales equal (a square map) and the
//     depth scale different (so no rigid view / object matrix passes); the camera's own forward is refused. The depth
//     row is the light axis; the cascades must agree (2 degrees), else the axis seen in 3 frames running. The draws
//     into that array (OnShadowDraw) are the fallback source while no consumer is seen.
//  2. The axis is in the matrices' input space (world, or the camera's view) with an unknown sign. Space: the reading
//     that stays put while the camera turns (> 15 degrees) is the right one (2 agreeing turns decide; the draws'
//     matrices are world; KYTY_RT_SHADOWS_SPACE=world|view forces it). Sign: the sun is above the horizon.
//  3. World up and the screen's y convention: a calibration pass (mode 5) counts the depth buffer's floors (normals
//     within 14 degrees of an axis) along each world axis, both signs, in both conventions; the winner (2x the next)
//     decides; priors +y up, row 0 = NDC y +1 (KYTY_RT_SHADOWS_UP=y|-y|z|..., KYTY_RT_SHADOWS_FLIPY=1|0 force them).
//     (10-24) The camera narrows it first: a game camera does not roll, so world up is the axis its right vector stays
//     perpendicular to while it turns (and its screen-up row stays within 70 degrees of), with the sign that puts up at
//     the top of the screen (row 0) in each convention: the floors only choose between the two mirror twins left. The
//     floors alone picked -z in Boletaria (walls facing a camera that looked along +z outnumbered the grazing ground;
//     the game is +y up, as the KBM camera's yaw from x / z assumes), which flips any sun with z > 0 below the horizon.
// The view-space light is the world sun through the current camera every frame; the pass is off while any is unknown.
struct CShadowPass {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	bool                    failed     = false;
	vk::Extent2D            render {}, trace {};
	SsgiImage               raw, tmp_a, tmp_b, out;
	SsgiImage               history[2], gbuf[2];
	SsgiImage               moments[2], var_a, var_b;
	uint32_t                frame         = 0;
	int                     quality       = 0;
	bool                    history_valid = false;
	VkBuffer                vote          = VK_NULL_HANDLE;
	VmaAllocation           vote_mem      = nullptr;
	const uint32_t*         vote_mapped   = nullptr;
	bool                    slot_voted[4] {};
	// Calibration: floors per hypothesis (convention c (0 row 0 = NDC +1, 1 mirrored) * 6 + axis * 2 + negative).
	double                  calib[12] {};
	int                     hyp = -1; // the decided hypothesis (-1: none yet)
	// The camera's evidence on world up (frames; per axis: frames its right row is perpendicular to it and its up row
	// leans toward it, the signed lean summed); the axis it decided (-1: none yet); the hypotheses it leaves.
	double                  cam_frames = 0.0, cam_ok[3] {}, cam_lean[3] {};
	int                     cam_axis = -1;
	uint32_t                cam_mask = 0xfffu;
	// Statistics (every 16th frame, read back 4 frames late; logged every 10 s with the GPU time).
	int64_t                 stat_frame = -1;
	double                  st_pixels = 0.0, st_n = 0.0, st_occ = 0.0, st_depth = 0.0;
	uint32_t                st_samples = 0;
	// The light probe: log2 K smoothed (K the game's sun-lit to facing-away luminance ratio; -1 none yet) and the
	// overcast factor it gives the strength (1 full sun .. 0.15 overcast).
	double                  log_k = -1.0, sun_factor = 1.0;
	// The sun's space: 0 unknown, 1 world, 2 view; turns that voted for each.
	int                     space = 0, world_votes = 0, view_votes = 0;
	bool                    space_ref = false;
	std::array<double, 3>   ref_axis {}, ref_vw {}, ref_fwd {};
	uint64_t                space_flip = ~0ull;
	// The latched world-space sun (toward the light).
	bool                    latched = false, switching = false, said_on = false;
	// (KYTY_HWRT_SHADOW) The pipeline traces the scene TLAS; one was built at least once; the 10 s line's counts.
	bool                    hw = false, hw_seen = false, hw_now = false;
	uint32_t                hw_traced = 0, hw_stale = 0, hw_none = 0;
	std::array<double, 3>   latch_axis {}, switch_axis {};
	float                   fade = 0.0f;
	std::chrono::steady_clock::time_point latch_time {}, last_fresh {};
	vk::QueryPool           pool      = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms  = 0.0;
	uint32_t                samples = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
} g_cs;

struct CShadowPush {
	float   proj[4];
	float   depth_map[4];
	float   motion[4];
	float   trace[4];
	int32_t misc[4];
	float   extra[4];
	float   cand[4][4];
	float   tune[4]; // x: 1 / K (the light probe's sun-lit to ambient ratio)
};
static_assert(sizeof(CShadowPush) == 176);

// (hwrt2s) KYTY_HWRT_SHADOW=0..3: the sun's rays through the scene TLAS (KYTY_HWRT=1, hwrt-scene.h) joined to KYTY_RT_SUN's
// cascades in its trace pass (rt-sun.comp HwSun, the RTSUN_HW_SPV build): contact shadows the maps' bias loses, small
// casters their texels miss, and every caster beyond the last cascade. 1: half-resolution trace, a ray a pixel (first
// hit), 150 m; 2: the same, 300 m; 3: full resolution, two rays (closest hit), 500 m. It runs KYTY_RT_SUN's plumbing at
// its own level when KYTY_RT_SUN is unset. 0 / unset: nothing (KYTY_RT_SUN and everything else exactly as before).
int HwShadowLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_HWRT_SHADOW");
		const int   q     = value != nullptr && *value != '\0' ? std::clamp(std::atoi(value), 0, 3) : 0;
		if (q > 0) {
			std::printf("HWRTSHADOW: KYTY_HWRT_SHADOW=%s read (level %d)%s\n", value, q,
			            HwScene::Requested() ? "" : ": needs KYTY_HWRT=1 too: off");
			std::fflush(stdout);
		}
		return HwScene::Requested() ? q : 0;
	}();
	return level;
}

// (rtsun) KYTY_RT_SUN=0..3: the cascades' visibility pass (frame-gen.h RtSunWanted); 0 when unset (KYTY_HWRT_SHADOW's
// level then, when that is set).
int RtSunLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_RT_SUN");
		int         q     = value != nullptr && *value != '\0' ? std::clamp(std::atoi(value), 0, 3) : 0;
		if ((value == nullptr || *value == '\0') && HwShadowLevel() > 0) {
			q = HwShadowLevel();
			std::printf("RTSUN: KYTY_RT_SUN unset: level %d for KYTY_HWRT_SHADOW\n", q);
			std::fflush(stdout);
			return q;
		}
		if (q > 0) {
			std::printf("RTSUN: KYTY_RT_SUN=%s read (level %d: %s)\n", value, q, q == 1 ? "low" : q == 2 ? "high" : "ultra");
			std::fflush(stdout);
		}
		return q;
	}();
	return level;
}

int ContactQuality() {
	static const int quality = [] {
		if (RtSunLevel() > 0) return RtSunLevel(); // (KYTY_RT_SUN runs this plumbing with its own shader)
		const char* name  = "KYTY_RT_SHADOWS";
		const char* value = std::getenv(name);
		if (value == nullptr || *value == '\0') name = "KYTY_CONTACT_SHADOWS", value = std::getenv(name);
		const int q = value != nullptr ? std::clamp(std::atoi(value), 0, 3) : 0;
		if (q > 0) {
			std::printf("RTSHADOW: %s=%s read (level %d)\n", name, value, q);
			std::fflush(stdout);
		}
		return q;
	}();
	return quality;
}

float RtShadowEnv(const char* name, const char* old_name, float fallback, float hi) {
	const char* value = std::getenv(name);
	if ((value == nullptr || *value == '\0') && old_name != nullptr) value = std::getenv(old_name);
	return value != nullptr && *value != '\0' ? std::clamp(static_cast<float>(std::atof(value)), 0.0f, hi) : fallback;
}

// The probe's findings (any recording thread writes, the upscaler's reads).
struct SunProbe {
	std::mutex              mutex;
	std::array<float, 3>    axis {};    // the light axis in the matrices' input space, sign canonical (largest > 0)
	uint64_t                flip = 0;   // the guest flip it was last seen in
	uint32_t                agree = 0;  // matrices agreeing in that dispatch
	uint32_t                stable = 0; // flips in a row with the same axis
	bool                    valid = false;
	int                     source = 0; // 1 a shadow consumer dispatch, 2 shadow-map draws
	uint64_t                consumer_flip = 0;
	std::array<float, 16>   view_proj {}; // the camera's (g_camera) when it was seen
	bool                    have_camera = false;
	std::array<std::atomic<uint64_t>, 8> consumers {};
	std::array<uint64_t, 8> scanned_flip {};
} g_sun;

bool Orthonormalish(const float* r0, const float* r1, const float* r2, std::array<float, 3>& axis) {
	const auto dot  = [](const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
	const float n0 = std::sqrt(dot(r0, r0)), n1 = std::sqrt(dot(r1, r1)), n2 = std::sqrt(dot(r2, r2));
	if (!(n0 > 1e-6f && n1 > 1e-6f && n2 > 1e-7f) || !(n0 < 1e4f && n1 < 1e4f && n2 < 1e4f)) return false;
	if (std::abs(dot(r0, r1)) > 0.01f * n0 * n1 || std::abs(dot(r0, r2)) > 0.01f * n0 * n2 || std::abs(dot(r1, r2)) > 0.01f * n1 * n2)
		return false;
	if (std::abs(n0 - n1) > 0.02f * std::max(n0, n1)) return false;  // a square shadow map
	if (std::abs(n2 - n0) < 0.05f * std::max(n0, n2)) return false;  // rigid / uniformly scaled: no projection
	axis = {r2[0] / n2, r2[1] / n2, r2[2] / n2};
	// Exactly axis-aligned rows are identity-like constants, not a sun.
	if (std::max({std::abs(axis[0]), std::abs(axis[1]), std::abs(axis[2])}) > 0.9999f) return false;
	int big = 0;
	for (int i = 1; i < 3; ++i)
		if (std::abs(axis[i]) > std::abs(axis[big])) big = i;
	if (axis[big] < 0.0f) axis = {-axis[0], -axis[1], -axis[2]};
	return true;
}

// Orthographic matrices among the words: their light axes.
void FindOrthoAxes(const std::vector<float>& f, std::vector<std::array<float, 3>>& axes) {
	const auto finite = [&](size_t o, size_t n) {
		for (size_t i = 0; i < n; ++i)
			if (!std::isfinite(f[o + i])) return false;
		return true;
	};
	std::array<float, 3> axis {};
	for (size_t o = 0; o + 12 <= f.size() && axes.size() < 16; ++o) {
		if (!finite(o, 12)) continue;
		const float* m = &f[o];
		if (o + 16 <= f.size() && finite(o, 16)) {
			// Row-major (clip = M * p): the w row (0, 0, 0, 1).
			if (m[12] == 0.0f && m[13] == 0.0f && m[14] == 0.0f && m[15] == 1.0f && Orthonormalish(m, m + 4, m + 8, axis)) {
				axes.push_back(axis);
				o += 15;
				continue;
			}
			// Column-major: the w column.
			if (m[3] == 0.0f && m[7] == 0.0f && m[11] == 0.0f && m[15] == 1.0f) {
				const float c0[3] {m[0], m[4], m[8]}, c1[3] {m[1], m[5], m[9]}, c2[3] {m[2], m[6], m[10]};
				if (Orthonormalish(c0, c1, c2, axis)) {
					axes.push_back(axis);
					o += 15;
					continue;
				}
			}
		}
		// 3x4 rows (the projective row left out), 16-byte aligned only: fewer chance matches.
		// (A perspective matrix's w row, along the depth row, follows a spot light's: refused.)
		const bool perspective = o + 16 <= f.size() && finite(o, 16) &&
		                         std::abs(m[12] * m[8] + m[13] * m[9] + m[14] * m[10]) >
		                             0.5f * std::sqrt((m[12] * m[12] + m[13] * m[13] + m[14] * m[14]) * (m[8] * m[8] + m[9] * m[9] + m[10] * m[10])) &&
		                         (m[12] * m[12] + m[13] * m[13] + m[14] * m[14]) > 1e-8f;
		if ((o & 3u) == 0 && !perspective && Orthonormalish(m, m + 4, m + 8, axis)) {
			axes.push_back(axis);
			o += 11;
		}
	}
}

int CameraForwardReject(const std::array<float, 3>& axis, const std::array<float, 16>& vp) {
	// The camera's w row's spatial part is its forward (perspective): a camera matrix, not a light one.
	const float n = std::sqrt(vp[12] * vp[12] + vp[13] * vp[13] + vp[14] * vp[14]);
	if (n < 1e-6f) return 0;
	return std::abs(axis[0] * vp[12] + axis[1] * vp[13] + axis[2] * vp[14]) / n > 0.9994f ? 1 : 0;
}

bool Inverse4(const std::array<float, 16>& m, std::array<double, 16>& inv) {
	// (Row-major, m[i*4+j]; the classic cofactor expansion.)
	double a[16];
	for (int i = 0; i < 16; ++i) a[i] = m[i];
	inv[0]  = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
	inv[4]  = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
	inv[8]  = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
	inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
	inv[1]  = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
	inv[5]  = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
	inv[9]  = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
	inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
	inv[2]  = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
	inv[6]  = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
	inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
	inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
	inv[3]  = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
	inv[7]  = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
	inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
	inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
	const double det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
	if (!(std::abs(det) > 1e-30)) return false;
	for (auto& v: inv) v /= det;
	return true;
}
} // namespace

bool ContactShadowsWanted() { return ContactQuality() > 0; }

bool IsShadowConsumer(uint64_t cs_hash) {
	for (const auto& h: g_sun.consumers)
		if (h.load(std::memory_order_relaxed) == cs_hash) return true;
	return false;
}

void OnShadowConsumer(uint64_t cs_hash, const ShadowConsumer& consumer, std::span<const uint32_t> srt) {
	const uint64_t flip = g_flips_taken.load(std::memory_order_relaxed);
	std::lock_guard lock(g_sun.mutex);
	size_t slot = g_sun.consumers.size();
	bool   first = false;
	for (size_t i = 0; i < g_sun.consumers.size(); ++i) {
		const uint64_t h = g_sun.consumers[i].load(std::memory_order_relaxed);
		if (h == cs_hash) { slot = i; break; }
		if (h == 0) {
			g_sun.consumers[i].store(cs_hash, std::memory_order_relaxed);
			g_sun.scanned_flip[i] = ~0ull;
			slot  = i;
			first = true;
			break;
		}
	}
	if (slot == g_sun.consumers.size() || g_sun.scanned_flip[slot] == flip) return; // (once a flip per program)
	g_sun.scanned_flip[slot] = flip;
	std::vector<float> words(srt.size());
	std::memcpy(words.data(), srt.data(), srt.size() * 4u);
	std::vector<std::array<float, 3>> axes;
	FindOrthoAxes(words, axes);
	for (uint32_t i = 0; i < consumer.buffers; ++i) {
		const uint64_t bytes = std::min<uint64_t>(consumer.buffer_size[i], 4096) & ~uint64_t {3};
		words.assign(bytes / 4, 0.0f);
		if (bytes == 0 || !Libs::LibKernel::Memory::TryReadBackingToHost(consumer.buffer_address[i], words.data(), bytes)) continue;
		FindOrthoAxes(words, axes);
	}
	bool camera = false;
	{
		std::lock_guard vp_lock(g_last_vp_mutex);
		if (g_have_last_vp) g_sun.view_proj = g_last_vp, g_sun.have_camera = camera = true;
	}
	if (g_sun.have_camera)
		axes.erase(std::remove_if(axes.begin(), axes.end(), [&](const auto& a) { return CameraForwardReject(a, g_sun.view_proj) != 0; }),
		           axes.end());
	// The largest group agreeing within 2 degrees.
	uint32_t best = 0, best_n = 0;
	for (uint32_t i = 0; i < axes.size(); ++i) {
		uint32_t n = 0;
		for (const auto& b: axes)
			n += std::abs(axes[i][0] * b[0] + axes[i][1] * b[1] + axes[i][2] * b[2]) > 0.9994f ? 1u : 0u;
		if (n > best_n) best = i, best_n = n;
	}
	static uint32_t reports = 0;
	if (first || (reports < 64 && flip % 600 == 0)) {
		++reports;
		std::printf("RTSHADOW: sun-shadow consumer CS %016llx (shadow array 0x%llx %ux%u x%u layers, SRT %zu words, %u constant "
		            "buffers): %zu orthographic matrices, axis (%.4f %.4f %.4f) agreed by %u%s\n",
		            static_cast<unsigned long long>(cs_hash), static_cast<unsigned long long>(consumer.shadow_address),
		            consumer.shadow_size, consumer.shadow_size, consumer.shadow_layers, srt.size(), consumer.buffers, axes.size(),
		            best_n > 0 ? axes[best][0] : 0.0f, best_n > 0 ? axes[best][1] : 0.0f, best_n > 0 ? axes[best][2] : 0.0f, best_n,
		            camera ? "" : " (no camera this frame)");
		std::fflush(stdout);
	}
	if (best_n == 0) return;
	const auto& a    = axes[best];
	const bool  same = g_sun.valid && std::abs(a[0] * g_sun.axis[0] + a[1] * g_sun.axis[1] + a[2] * g_sun.axis[2]) > 0.9994f;
	g_sun.stable = same ? (g_sun.flip == flip ? g_sun.stable : g_sun.stable + 1) : 1;
	g_sun.agree  = std::max(best_n, g_sun.flip == flip && same ? g_sun.agree : 0u);
	g_sun.axis   = a;
	g_sun.flip   = flip;
	g_sun.valid  = true;
	g_sun.source = 1;
	g_sun.consumer_flip = flip;
}

void OnShadowDraw(uint64_t vs_hash, std::span<const uint32_t> srt, bool shadow_target) {
	if (srt.empty()) return;
	// The RT shadows' camera on every draw path, whatever the upscaler or frame generation gate OnDraw on: a G-buffer
	// draw's view-projection, only when finite and perspective (a w row with a spatial part); else it stays unknown.
	if (IsGbufferVertexShader(vs_hash) && srt.size() >= SrtViewProj + 16) {
		std::array<float, 16> vp {};
		bool                  ok = true;
		for (uint32_t i = 0; i < 16 && ok; ++i) ok = std::isfinite(vp[i] = AsFloat(srt[SrtViewProj + i]));
		if (ok && std::abs(vp[12]) + std::abs(vp[13]) + std::abs(vp[14]) > 1e-6f) {
			std::lock_guard vp_lock(g_last_vp_mutex);
			g_last_vp = vp, g_have_last_vp = true;
		}
	}
	static std::mutex                             mutex;
	static std::unordered_map<uint64_t, uint64_t> learned;  // shadow-pass vertex shaders -> the flip last learned
	static std::unordered_map<uint64_t, uint64_t> probed;   // others: the flip last scanned (every 300 flips at most)
	static std::vector<std::array<float, 3>>      axes;     // this flip's
	static uint64_t                               flip_of = ~0ull;
	static uint32_t                               draws = 0;
	static bool                                   said = false;
	const uint64_t flip = g_flips_taken.load(std::memory_order_relaxed);
	std::lock_guard lock(mutex);
	if (flip != flip_of) {
		// The last flip's axes: the largest group agreeing within 2 degrees becomes the sun.
		uint32_t best = 0, best_n = 0;
		for (uint32_t i = 0; i < axes.size(); ++i) {
			uint32_t n = 0;
			for (const auto& b: axes) n += std::abs(axes[i][0] * b[0] + axes[i][1] * b[1] + axes[i][2] * b[2]) > 0.9994f ? 1u : 0u;
			if (n > best_n) best = i, best_n = n;
		}
		if (best_n >= 2 || (best_n == 1 && axes.size() == 1)) {
			const auto a = axes[best];
			std::lock_guard sun(g_sun.mutex);
			// (A shadow consumer seen lately is the better source: its matrices are the cascades' own, no object's.)
			if (!(g_sun.source == 1 && flip_of <= g_sun.consumer_flip + 120)) {
				const bool same = g_sun.valid && std::abs(a[0] * g_sun.axis[0] + a[1] * g_sun.axis[1] + a[2] * g_sun.axis[2]) > 0.9994f;
				g_sun.stable = same ? g_sun.stable + 1 : 1;
				g_sun.agree  = best_n;
				g_sun.axis   = a;
				g_sun.flip   = flip_of;
				g_sun.valid  = true;
				g_sun.source = 2;
				std::lock_guard vp_lock(g_last_vp_mutex);
				if (g_have_last_vp) g_sun.view_proj = g_last_vp, g_sun.have_camera = true;
			}
		}
		axes.clear();
		flip_of = flip;
		draws   = 0;
	}
	if (draws >= 64) return; // (a sample of a flip's shadow draws)
	const auto known = learned.find(vs_hash);
	if (!shadow_target && known == learned.end()) {
		auto& last = probed[vs_hash];
		if (last != 0 && flip < last + 300) return;
		last = flip + 1;
	}
	std::vector<float> words(std::min<size_t>(srt.size(), 160));
	std::memcpy(words.data(), srt.data(), words.size() * 4u);
	std::vector<std::array<float, 3>> found;
	FindOrthoAxes(words, found);
	{
		std::lock_guard vp_lock(g_last_vp_mutex);
		if (g_have_last_vp)
			found.erase(std::remove_if(found.begin(), found.end(), [&](const auto& a) { return CameraForwardReject(a, g_last_vp) != 0; }),
			            found.end());
	}
	if (found.empty()) return;
	++draws;
	if (shadow_target || known != learned.end()) learned[vs_hash] = flip;
	else if (learned.size() < 64) learned[vs_hash] = flip; // (an ortho matrix in an unknown VS: kept as a candidate)
	axes.push_back(found[0]);
	if (!said) {
		said = true;
		std::printf("RTSHADOW: sun matrix from a shadow draw (VS %016llx, %s, SRT %zu words): light axis (%.4f %.4f %.4f)\n",
		            static_cast<unsigned long long>(vs_hash), shadow_target ? "depth target a shadow-map layer" : "table/record path",
		            srt.size(), found[0][0], found[0][1], found[0][2]);
		std::fflush(stdout);
	}
}

// ---- KYTY_RT_SUN (frame-gen.h RtSunWanted; rt-sun.comp) ----
// What the sun pass knows of the game's cascades (any recording thread writes, the upscaler's thread reads).
// Evidence it builds on (logs of 10-24, KYTY_RT_SHADOWS): the sun-shadow draws (VS daa84716d49bf71d, a depth-only target
// that is a layer of the 1024x1024 x80 array at 0x102a400000) carry an orthographic light matrix in their 32-word SRT
// whose depth row gave the sun (0.3434 0.7004 -0.6257), 44.5 degrees; the deferred lighting dispatches (CS decd2b04b5eee050,
// 7e676bf37482d768, 422c6f1a73af2909, a4f528961552278e, 9d79b940e729615f) sample that array.
namespace {
struct RtSunEntry {
	uint32_t              layer = 0;
	std::array<float, 16> m {}; // row-major, clip = m * (world, 1)
	uint32_t              count = 0;
};
struct RtSunState {
	std::mutex mutex;
	// The draws' matrices: this flip's (per layer, up to 24 sampled draws) and the last complete flip's.
	uint64_t                cur_flip = ~0ull, last_flip = ~0ull;
	std::vector<RtSunEntry> cur, last;
	std::array<uint8_t, 256> sampled {};
	uint64_t                address = 0; // the cascades' array (the draws' depth target)
	// Per vertex shader: the SRT word offset * 4 + layout (0 row-major 4x4, 1 column-major 4x4, 2 3x4 rows) of its
	// orthographic matrix, -1 none (tries in the high bits: looked for at most 8 times).
	std::unordered_map<uint64_t, int32_t> vs_code;
	uint32_t                vs_logged = 0;
	// Our copy of the cascade layers (slot k: copied[k]).
	VkImage                 image = VK_NULL_HANDLE;
	VmaAllocation           mem   = nullptr;
	vk::ImageView           view  = nullptr;
	vk::Format              format = vk::Format::eUndefined;
	vk::Extent2D            extent {};
	vk::ImageLayout         layout = vk::ImageLayout::eUndefined;
	bool                    failed = false, said_copy = false;
	uint64_t                copy_flip = ~0ull, copies = 0;
	std::vector<RtSunEntry> copied;
	// The shader's buffer (4 frame slots of 1 KiB).
	VkBuffer                buf     = VK_NULL_HANDLE;
	VmaAllocation           buf_mem = nullptr;
	uint8_t*                buf_mapped = nullptr;
	// Calibration: the convention (-1 undecided), its candidate, the votes last read, the bias, R (log2 (1 + R), -1 none).
	int                     hyp = -1, cand = -1, cand_n = 0;
	std::array<uint32_t, 8> votes {};
	std::array<double, 32>  hist {};
	double                  bias_m = 0.03, log_r = -1.0;
	uint32_t                r_lit = 0, r_shadow = 0;
	// Statistics for the 10 s line.
	double                  st_dark = 0.0, st_lift = 0.0, st_amount = 0.0, st_pixels = 0.0;
	uint32_t                st_frames = 0, cascades_used = 0;
	float                   texel_min = 0.0f, texel_max = 0.0f;
} g_rs;

constexpr uint32_t kRtSunSlots = 8;      // cascades at most
constexpr uint32_t kRtSunSlotBytes = 1024;

float RtSunEnv(const char* name, float fallback, float lo, float hi) {
	const char* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? std::clamp(static_cast<float>(std::atof(value)), lo, hi) : fallback;
}

bool RtSunHasStencil(vk::Format f) {
	return f == vk::Format::eD32SfloatS8Uint || f == vk::Format::eD24UnormS8Uint || f == vk::Format::eD16UnormS8Uint;
}

// The first orthographic light matrix among the words (FindOrthoAxes' tests): its code (offset * 4 + layout), -1 none.
int32_t RtSunFindMatrix(std::span<const uint32_t> srt) {
	const size_t       n = std::min<size_t>(srt.size(), 160);
	std::vector<float> f(n);
	std::memcpy(f.data(), srt.data(), n * 4u);
	const auto finite = [&](size_t o, size_t k) {
		for (size_t i = 0; i < k; ++i)
			if (!std::isfinite(f[o + i])) return false;
		return true;
	};
	std::array<float, 3> axis {};
	for (size_t o = 0; o + 12 <= n; ++o) {
		if (!finite(o, 12)) continue;
		const float* m = &f[o];
		if (o + 16 <= n && finite(o, 16)) {
			if (m[12] == 0.0f && m[13] == 0.0f && m[14] == 0.0f && m[15] == 1.0f && Orthonormalish(m, m + 4, m + 8, axis))
				return static_cast<int32_t>(o * 4 + 0);
			if (m[3] == 0.0f && m[7] == 0.0f && m[11] == 0.0f && m[15] == 1.0f) {
				const float c0[3] {m[0], m[4], m[8]}, c1[3] {m[1], m[5], m[9]}, c2[3] {m[2], m[6], m[10]};
				if (Orthonormalish(c0, c1, c2, axis)) return static_cast<int32_t>(o * 4 + 1);
			}
		}
		const bool perspective = o + 16 <= n && finite(o, 16) &&
		                         std::abs(m[12] * m[8] + m[13] * m[9] + m[14] * m[10]) >
		                             0.5f * std::sqrt((m[12] * m[12] + m[13] * m[13] + m[14] * m[14]) * (m[8] * m[8] + m[9] * m[9] + m[10] * m[10])) &&
		                         (m[12] * m[12] + m[13] * m[13] + m[14] * m[14]) > 1e-8f;
		if ((o & 3u) == 0 && !perspective && Orthonormalish(m, m + 4, m + 8, axis)) return static_cast<int32_t>(o * 4 + 2);
	}
	return -1;
}

// The matrix at `code`, row-major (clip = m * p); false when it is not there (not finite, w row not (0, 0, 0, 1)).
bool RtSunReadMatrix(std::span<const uint32_t> srt, int32_t code, std::array<float, 16>& m) {
	const size_t o = static_cast<size_t>(code >> 2);
	const int    kind = code & 3;
	if (o + (kind == 2 ? 12u : 16u) > srt.size()) return false;
	float f[16] {};
	std::memcpy(f, srt.data() + o, (kind == 2 ? 12u : 16u) * 4u);
	if (kind == 2) f[12] = 0.0f, f[13] = 0.0f, f[14] = 0.0f, f[15] = 1.0f;
	if (kind == 1) {
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j) m[i * 4 + j] = f[j * 4 + i];
	} else {
		for (int i = 0; i < 16; ++i) m[i] = f[i];
	}
	for (float v: m)
		if (!std::isfinite(v)) return false;
	return m[12] == 0.0f && m[13] == 0.0f && m[14] == 0.0f && m[15] == 1.0f;
}

// One matrix per layer: the one most of its sampled draws used (a cascade's matrix is shared by all its draws).
std::vector<RtSunEntry> RtSunTable(const std::vector<RtSunEntry>& entries) {
	std::vector<RtSunEntry> out;
	for (const auto& e: entries) {
		auto it = std::ranges::find(out, e.layer, &RtSunEntry::layer);
		if (it == out.end()) out.push_back(e);
		else if (e.count > it->count) *it = e;
	}
	std::erase_if(out, [](const RtSunEntry& e) { return e.count < 2; });
	std::ranges::sort(out, {}, &RtSunEntry::layer);
	if (out.size() > kRtSunSlots) out.resize(kRtSunSlots);
	return out;
}
} // namespace

bool RtSunWanted() { return RtSunLevel() > 0; }

void OnSunCascadeDraw(uint64_t vs_hash, std::span<const uint32_t> srt, uint32_t layer, uint64_t target_address) {
	if (srt.empty() || layer >= 256 || !RtSunWanted()) return;
	const uint64_t  flip = g_flips_taken.load(std::memory_order_relaxed);
	std::lock_guard lock(g_rs.mutex);
	if (flip != g_rs.cur_flip) {
		if (!g_rs.cur.empty()) g_rs.last = std::move(g_rs.cur), g_rs.last_flip = g_rs.cur_flip;
		g_rs.cur.clear();
		g_rs.cur_flip = flip;
		g_rs.sampled.fill(0);
	}
	if (g_rs.sampled[layer] >= 24) return; // (a sample of each layer's draws a flip)
	auto it = g_rs.vs_code.find(vs_hash);
	if (it == g_rs.vs_code.end() || (it->second < 0 && (-it->second) < 8)) {
		const int32_t code = RtSunFindMatrix(srt);
		const int32_t was  = it == g_rs.vs_code.end() ? 0 : -it->second;
		g_rs.vs_code[vs_hash] = code >= 0 ? code : -(was + 1);
		if (g_rs.vs_logged < 16 && (code >= 0 || was == 0)) {
			++g_rs.vs_logged;
			std::printf("RTSUN: cascade draw VS %016llx (layer %u of 0x%llx, SRT %zu words): %s at word %d\n",
			            static_cast<unsigned long long>(vs_hash), layer, static_cast<unsigned long long>(target_address), srt.size(),
			            code >= 0 ? (code & 3) == 0 ? "orthographic matrix (row-major 4x4)" : (code & 3) == 1 ? "orthographic matrix (column-major 4x4)"
			                                                                             : "orthographic matrix (3x4 rows)"
			                      : "no orthographic matrix",
			            code >= 0 ? code >> 2 : -1);
			std::fflush(stdout);
		}
		it = g_rs.vs_code.find(vs_hash);
	}
	if (it->second < 0) return;
	std::array<float, 16> m {};
	if (!RtSunReadMatrix(srt, it->second, m)) return;
	++g_rs.sampled[layer];
	g_rs.address = target_address;
	for (auto& e: g_rs.cur)
		if (e.layer == layer && e.m == m) {
			++e.count;
			return;
		}
	if (g_rs.cur.size() < 256) g_rs.cur.push_back({layer, m, 1});
}

void RtSunAfterDispatch(GraphicContext& graphics, TextureCache& cache, vk::CommandBuffer command, uint64_t cs_hash,
                        std::span<const TextureBinding> images) {
	if (!RtSunWanted() || g_rs.failed) return;
	const uint64_t          flip = g_flips_taken.load(std::memory_order_relaxed);
	std::vector<RtSunEntry> table;
	uint64_t                address = 0;
	{
		std::lock_guard lock(g_rs.mutex);
		if (g_rs.address == 0 || g_rs.copy_flip == flip) return;
		address = g_rs.address;
		table   = RtSunTable(g_rs.cur_flip == flip && !g_rs.cur.empty() ? g_rs.cur : g_rs.last);
	}
	if (table.empty()) return;
	const TextureBinding* binding = nullptr;
	for (const auto& b: images)
		if (b.image_id && b.desc.info.data.address == address && b.desc.info.resources.layers >= 4) {
			binding = &b;
			break;
		}
	if (binding == nullptr) return;
	auto& image = cache.GetImage(binding->image_id);
	if (image.backing.image == nullptr) return;
	const vk::Format   fmt    = image.backing.format;
	const vk::Extent2D extent = {image.backing.extent.width, image.backing.extent.height};
	const bool         depth  = SsrIsDepthFormat(fmt);
	for (const auto& e: table)
		if (e.layer >= binding->desc.info.resources.layers) return;
	if (g_rs.image != VK_NULL_HANDLE && (g_rs.format != fmt || g_rs.extent != extent)) {
		// (another format or size: made again; the device idles first, as the passes' own images are remade)
		graphics.device.waitIdle();
		graphics.device.destroyImageView(g_rs.view, nullptr);
		vmaDestroyImage(graphics.allocator, g_rs.image, g_rs.mem);
		g_rs.image = VK_NULL_HANDLE, g_rs.mem = nullptr, g_rs.view = nullptr, g_rs.layout = vk::ImageLayout::eUndefined;
	}
	const vk::ImageAspectFlags all_aspects =
	    depth ? (RtSunHasStencil(fmt) ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil : vk::ImageAspectFlagBits::eDepth)
	          : vk::ImageAspectFlagBits::eColor;
	const vk::ImageAspectFlags copy_aspect = depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;
	if (g_rs.image == VK_NULL_HANDLE) {
		VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		create.imageType   = VK_IMAGE_TYPE_2D;
		create.format      = static_cast<VkFormat>(fmt);
		create.extent      = {extent.width, extent.height, 1};
		create.mipLevels   = 1;
		create.arrayLayers = kRtSunSlots;
		create.samples     = VK_SAMPLE_COUNT_1_BIT;
		create.tiling      = VK_IMAGE_TILING_OPTIMAL;
		create.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo allocate {};
		allocate.usage = VMA_MEMORY_USAGE_AUTO;
		if (vmaCreateImage(graphics.allocator, &create, &allocate, &g_rs.image, &g_rs.mem, nullptr) != VK_SUCCESS) {
			g_rs.image = VK_NULL_HANDLE, g_rs.failed = true;
			std::printf("RTSUN: the cascade copy (%ux%u x%u, %s) could not be created: off\n", extent.width, extent.height, kRtSunSlots,
			            SsrFormatName(fmt));
			std::fflush(stdout);
			return;
		}
		vk::ImageViewCreateInfo view_info {};
		view_info.image            = vk::Image(g_rs.image);
		view_info.viewType         = vk::ImageViewType::e2DArray;
		view_info.format           = fmt;
		view_info.subresourceRange = vk::ImageSubresourceRange {copy_aspect, 0, 1, 0, kRtSunSlots};
		if (graphics.device.createImageView(&view_info, nullptr, &g_rs.view) != vk::Result::eSuccess) {
			vmaDestroyImage(graphics.allocator, g_rs.image, g_rs.mem);
			g_rs.image = VK_NULL_HANDLE, g_rs.view = nullptr, g_rs.failed = true;
			return;
		}
		g_rs.format = fmt, g_rs.extent = extent, g_rs.layout = vk::ImageLayout::eUndefined;
	}
	{
		// (whatever wrote the cascades, this dispatch's reads, our pass's reads of the last copy: before the copy)
		vk::MemoryBarrier2 mb {};
		mb.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		mb.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
		mb.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		mb.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &mb;
		command.pipelineBarrier2(dependency);
	}
	for (const auto& e: table)
		image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, ImageSubresourceRange {0, 1, e.layer, 1},
		              command);
	const auto barrier = [&](vk::ImageLayout from, vk::ImageLayout to, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
	                         vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
		vk::ImageMemoryBarrier2 b {};
		b.srcStageMask        = src_stage;
		b.srcAccessMask       = src_access;
		b.dstStageMask        = dst_stage;
		b.dstAccessMask       = dst_access;
		b.oldLayout           = from;
		b.newLayout           = to;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image               = vk::Image(g_rs.image);
		b.subresourceRange    = vk::ImageSubresourceRange {all_aspects, 0, 1, 0, kRtSunSlots};
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers    = &b;
		command.pipelineBarrier2(dependency);
	};
	barrier(g_rs.layout, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
	        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
	std::array<vk::ImageCopy, kRtSunSlots> copies {};
	for (uint32_t k = 0; k < table.size(); ++k) {
		copies[k].srcSubresource = vk::ImageSubresourceLayers {copy_aspect, 0, table[k].layer, 1};
		copies[k].dstSubresource = vk::ImageSubresourceLayers {copy_aspect, 0, k, 1};
		copies[k].extent         = vk::Extent3D {extent.width, extent.height, 1};
	}
	command.copyImage(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, vk::Image(g_rs.image), vk::ImageLayout::eTransferDstOptimal,
	                  static_cast<uint32_t>(table.size()), copies.data());
	barrier(vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
	g_rs.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
	std::lock_guard lock(g_rs.mutex);
	g_rs.copied    = std::move(table);
	g_rs.copy_flip = flip;
	++g_rs.copies;
	if (!g_rs.said_copy) {
		g_rs.said_copy = true;
		std::string line;
		char        buf[160];
		for (const auto& e: g_rs.copied) {
			const double sx = std::sqrt(double(e.m[0]) * e.m[0] + double(e.m[1]) * e.m[1] + double(e.m[2]) * e.m[2]);
			std::snprintf(buf, sizeof(buf), " layer %u (%.1f m wide, %u of its draws)", e.layer, sx > 0.0 ? 2.0 / sx : 0.0, e.count);
			line += buf;
		}
		std::printf("RTSUN: cascades copied after CS %016llx: array 0x%llx %ux%u %s, %zu cascades:%s\n",
		            static_cast<unsigned long long>(cs_hash), static_cast<unsigned long long>(address), extent.width, extent.height,
		            SsrFormatName(fmt), g_rs.copied.size(), line.c_str());
		std::fflush(stdout);
	}
}

namespace {
bool CShadowEnsure(GraphicContext& graphics, vk::Extent2D render, int quality) {
	const auto device = graphics.device;
	if (g_cs.pipeline == nullptr) {
		if (g_cs.failed) return false;
		g_cs.failed = true;
		if (graphics.GetPhysicalDeviceProperties().limits.maxPushConstantsSize < sizeof(CShadowPush)) return false;
		if (g_ssgi.point == nullptr) {
			for (int i = 0; i < 2; ++i) {
				vk::SamplerCreateInfo sampler {};
				sampler.magFilter    = i == 0 ? vk::Filter::eNearest : vk::Filter::eLinear;
				sampler.minFilter    = sampler.magFilter;
				sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
				sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
				sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
				if (device.createSampler(&sampler, nullptr, i == 0 ? &g_ssgi.point : &g_ssgi.linear) != vk::Result::eSuccess) return false;
			}
		}
		// (KYTY_RT_SUN: rt-sun.comp, two more: 12 the cascades' copy, 13 their matrices' buffer)
		// (KYTY_HWRT_SHADOW, while the device has ray queries: 14 the scene TLAS)
		g_cs.hw = RtSunLevel() > 0 && HwShadowLevel() > 0 && HwScene::Active();
		const uint32_t count = g_cs.hw ? 15u : RtSunLevel() > 0 ? 14u : 12u;
		std::array<vk::DescriptorSetLayoutBinding, 15> bindings {};
		for (uint32_t i = 0; i < count; ++i) {
			bindings[i].binding         = i;
			bindings[i].descriptorType  = i == 14 ? vk::DescriptorType::eAccelerationStructureKHR
			                              : i == 11 || i == 13 ? vk::DescriptorType::eStorageBuffer
			                              : i < 7 || i == 9 || i == 12 ? vk::DescriptorType::eCombinedImageSampler
			                                                           : vk::DescriptorType::eStorageImage;
			bindings[i].descriptorCount = 1;
			bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
		}
		vk::DescriptorSetLayoutCreateInfo set_info {};
		set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		set_info.bindingCount = count;
		set_info.pBindings    = bindings.data();
		if (device.createDescriptorSetLayout(&set_info, nullptr, &g_cs.set_layout) != vk::Result::eSuccess) return false;
		vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(CShadowPush)};
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.setLayoutCount         = 1;
		layout_info.pSetLayouts            = &g_cs.set_layout;
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges    = &push;
		if (device.createPipelineLayout(&layout_info, nullptr, &g_cs.layout) != vk::Result::eSuccess) return false;
		vk::ShaderModuleCreateInfo module_info {};
		module_info.codeSize    = g_cs.hw ? sizeof(RTSUN_HW_SPV) : RtSunLevel() > 0 ? sizeof(RTSUN_SPV) : sizeof(CSHADOW_SPV);
		module_info.pCode       = g_cs.hw ? RTSUN_HW_SPV : RtSunLevel() > 0 ? RTSUN_SPV : CSHADOW_SPV;
		if (HwShadowLevel() > 0) {
			std::printf("HWRTSHADOW: %s\n", g_cs.hw ? "the sun pass traces the scene TLAS (rt-sun.comp with HwSun)"
			                                          : "the device has no ray queries (KYTY_HWRT inactive): KYTY_RT_SUN alone");
			std::fflush(stdout);
		}
		vk::ShaderModule module = nullptr;
		if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
		pipeline_info.stage.module = module;
		pipeline_info.stage.pName  = "main";
		pipeline_info.layout       = g_cs.layout;
		const auto created = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_cs.pipeline);
		device.destroyShaderModule(module, nullptr);
		if (created != vk::Result::eSuccess) {
			g_cs.pipeline = nullptr;
			return false;
		}
		VkBufferCreateInfo create {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		create.size  = 128 * sizeof(uint32_t); // (calibration slots, then statistics; KYTY_RT_SUN's from word 80)
		create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo allocate {};
		allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		allocate.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo mapped {};
		if (vmaCreateBuffer(graphics.allocator, &create, &allocate, &g_cs.vote, &g_cs.vote_mem, &mapped) != VK_SUCCESS) return false;
		g_cs.vote_mapped = static_cast<const uint32_t*>(mapped.pMappedData);
		if (RtSunLevel() > 0) {
			// (KYTY_RT_SUN: the cascades' buffer, 4 frame slots written by the CPU)
			VkBufferCreateInfo sun_create {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			sun_create.size  = 4 * kRtSunSlotBytes;
			sun_create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
			VmaAllocationCreateInfo sun_allocate {};
			sun_allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
			sun_allocate.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
			VmaAllocationInfo sun_mapped {};
			if (vmaCreateBuffer(graphics.allocator, &sun_create, &sun_allocate, &g_rs.buf, &g_rs.buf_mem, &sun_mapped) != VK_SUCCESS)
				return false;
			g_rs.buf_mapped = static_cast<uint8_t*>(sun_mapped.pMappedData);
		}
		vk::QueryPoolCreateInfo query {};
		query.queryType  = vk::QueryType::eTimestamp;
		query.queryCount = 16;
		if (device.createQueryPool(&query, nullptr, &g_cs.pool) != vk::Result::eSuccess) g_cs.pool = nullptr;
		g_cs.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
		g_cs.failed    = false;
	}
	if (g_cs.out.image.image != nullptr && g_cs.render == render && g_cs.quality == quality) return true;
	device.waitIdle();
	SsgiImage* all[] {&g_cs.raw, &g_cs.tmp_a, &g_cs.tmp_b, &g_cs.out, &g_cs.history[0], &g_cs.history[1], &g_cs.gbuf[0],
	                  &g_cs.gbuf[1], &g_cs.moments[0], &g_cs.moments[1], &g_cs.var_a, &g_cs.var_b};
	for (auto* img : all)
		SsgiFreeImage(graphics, *img);
	// (KYTY_HWRT_SHADOW 1, 2: a half-resolution trace whatever KYTY_RT_SUN's level; 3: full)
	const bool         half  = quality < 2 || (g_cs.hw && HwShadowLevel() < 3);
	const vk::Extent2D trace = !half ? render : vk::Extent2D {std::max(render.width / 2, 1u), std::max(render.height / 2, 1u)};
	bool ok = SsgiMakeImage(graphics, g_cs.out, render);
	for (auto* img : all)
		if (img != &g_cs.out) ok = ok && SsgiMakeImage(graphics, *img, trace);
	if (!ok) {
		for (auto* img : all)
			SsgiFreeImage(graphics, *img);
		g_cs.failed = true;
		return false;
	}
	g_cs.render        = render;
	g_cs.quality       = quality;
	g_cs.trace         = trace;
	g_cs.history_valid = false;
	return true;
}

// (KYTY_RT_SUN) This frame's cascades into `slot` of the shader's buffer (rt-sun.comp SunBuf): each copied layer's light
// matrix composed with the camera's view -> world (from the projection and the G-buffer draws' view-projection, as
// CShadowView, with the translation), sorted finest first. False while there is no copy of the last 8 flips.
bool RtSunPrepare(const std::array<float, 16>& m, const std::array<float, 16>& vp, uint32_t slot) {
	if (g_rs.buf_mapped == nullptr || g_rs.view == nullptr) return false;
	std::vector<RtSunEntry> table;
	uint64_t                copy_flip = ~0ull;
	{
		std::lock_guard lock(g_rs.mutex);
		table = g_rs.copied, copy_flip = g_rs.copy_flip;
	}
	const uint64_t flips = g_flips_taken.load(std::memory_order_relaxed);
	if (table.empty() || copy_flip == ~0ull || flips > copy_flip + 8) return false;
	std::array<float, 16> pc {};
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j) pc[i * 4 + j] = m[j * 4 + i];
	std::array<double, 16> pinv {};
	if (!Inverse4(pc, pinv)) return false;
	double v[16] {};
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			for (int k = 0; k < 4; ++k) v[i * 4 + j] += pinv[i * 4 + k] * vp[k * 4 + j];
	for (int i = 0; i < 3; ++i) {
		const double n = std::sqrt(v[i * 4] * v[i * 4] + v[i * 4 + 1] * v[i * 4 + 1] + v[i * 4 + 2] * v[i * 4 + 2]);
		if (!(n > 1e-9) || !std::isfinite(n)) return false;
		for (int j = 0; j < 4; ++j) v[i * 4 + j] /= n;
	}
	// view -> world: the rotation's transpose, minus the rotated translation.
	double vi[16] {};
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) vi[i * 4 + j] = v[j * 4 + i];
		vi[i * 4 + 3] = -(v[0 * 4 + i] * v[3] + v[1 * 4 + i] * v[7] + v[2 * 4 + i] * v[11]);
	}
	vi[15] = 1.0;
	struct Casc {
		double   r[12];
		float    texel, mz;
		uint32_t k;
	};
	const float       size = static_cast<float>(std::max(g_rs.extent.width, 1u));
	std::vector<Casc> cs;
	for (uint32_t k = 0; k < table.size() && k < kRtSunSlots; ++k) {
		const auto&  mm = table[k].m;
		const double sx = std::sqrt(double(mm[0]) * mm[0] + double(mm[1]) * mm[1] + double(mm[2]) * mm[2]);
		const double sz = std::sqrt(double(mm[8]) * mm[8] + double(mm[9]) * mm[9] + double(mm[10]) * mm[10]);
		if (!(sx > 1e-9) || !(sz > 1e-9)) continue;
		Casc c {};
		for (int row = 0; row < 3; ++row)
			for (int j = 0; j < 4; ++j) {
				double d = 0.0;
				for (int l = 0; l < 4; ++l) d += double(mm[row * 4 + l]) * vi[l * 4 + j];
				c.r[row * 4 + j] = d;
			}
		c.texel = static_cast<float>(2.0 / (sx * size));
		c.mz    = static_cast<float>(1.0 / sz);
		c.k     = k;
		cs.push_back(c);
	}
	if (cs.empty()) return false;
	std::ranges::sort(cs, {}, &Casc::texel);
	static const float angle   = RtSunEnv("KYTY_RT_SUN_ANGLE", 0.8f, 0.1f, 5.0f);     // the sun's apparent size (deg)
	static const float reach   = RtSunEnv("KYTY_RT_SUN_REACH", 6.0f, 0.5f, 50.0f);    // occluder heights the penumbra search sees (m)
	static const float pcf     = RtSunEnv("KYTY_RT_SUN_GAMEPCF", 1.0f, 0.0f, 3.0f);   // the game emulation's PCF radius (taps)
	static const float band    = RtSunEnv("KYTY_RT_SUN_BAND", 0.08f, 0.0f, 0.3f);     // the cascade blend band (uv)
	static const float lift    = RtSunEnv("KYTY_RT_SUN_LIFT", 0.75f, 0.0f, 1.0f);     // brightening allowed (share of the sun term)
	static const float contact = RtSunEnv("KYTY_RT_SUN_CONTACT", RtSunLevel() >= 2 ? 0.75f : 0.0f, 0.0f, 1.0f);
	static const float debug   = RtSunEnv("KYTY_RT_SUN_DEBUG", 0.0f, 0.0f, 4.0f);
	static const float forced_hyp  = RtSunEnv("KYTY_RT_SUN_HYP", -1.0f, -1.0f, 7.0f);
	static const float forced_bias = RtSunEnv("KYTY_RT_SUN_BIAS", -1.0f, -1.0f, 1.0f);
	static const float forced_r    = RtSunEnv("KYTY_RT_SUN_R", -1.0f, -1.0f, 100.0f);
	double r = 1.5; // (nothing measured yet: a moderate sun)
	if (g_rs.log_r >= 0.0) r = std::exp2(g_rs.log_r) - 1.0;
	else if (g_cs.log_k >= 0.0) r = std::max(std::exp2(g_cs.log_k) - 1.0, 0.0) / 0.75;
	if (forced_r >= 0.0f) r = forced_r;
	float* f = reinterpret_cast<float*>(g_rs.buf_mapped + slot * kRtSunSlotBytes);
	std::memset(f, 0, kRtSunSlotBytes);
	f[0] = static_cast<float>(cs.size());
	f[1] = 2.0f * std::tan(0.5f * angle * 0.01745329f);
	f[2] = static_cast<float>(r);
	f[3] = 0.0f; // (strength x fade: RtSunSetFade)
	f[4] = forced_hyp >= 0.0f ? std::floor(forced_hyp) : static_cast<float>(g_rs.hyp);
	f[5] = forced_bias >= 0.0f ? forced_bias : static_cast<float>(g_rs.bias_m);
	f[6] = std::floor(debug);
	f[7] = size;
	f[8] = std::floor(pcf), f[9] = contact, f[10] = band, f[11] = lift;
	f[12] = reach;
	for (size_t i = 0; i < cs.size(); ++i) {
		float* c = f + 16 + i * 16;
		for (int j = 0; j < 12; ++j) c[j] = static_cast<float>(cs[i].r[j]);
		c[12] = static_cast<float>(cs[i].k), c[13] = cs[i].texel, c[14] = cs[i].mz, c[15] = 1.0f;
	}
	if (g_cs.hw) {
		// (KYTY_HWRT_SHADOW: rt-sun.comp SunBuf w0..w2, hw: after the 8 cascades, float 144) view -> the TLAS's world (the
		// same G-buffer view-projection the scene capture inverts), the level (0: no TLAS this frame), the far-hit weight
		// inside the cascades, the ray length, the contact range.
		static const float miss    = RtSunEnv("KYTY_HWRT_SHADOW_MISS", 0.5f, 0.0f, 1.0f);
		static const float range   = RtSunEnv("KYTY_HWRT_SHADOW_RANGE", HwShadowLevel() >= 3 ? 500.0f : HwShadowLevel() >= 2 ? 300.0f : 150.0f, 5.0f, 2000.0f);
		static const float contact_m = RtSunEnv("KYTY_HWRT_SHADOW_CONTACT", 1.0f, 0.05f, 20.0f);
		float* w = f + 144;
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 4; ++j) w[i * 4 + j] = static_cast<float>(vi[i * 4 + j]);
		w[12] = g_cs.hw_now ? static_cast<float>(HwShadowLevel()) : 0.0f;
		w[13] = miss, w[14] = range, w[15] = contact_m;
	}
	(void)vmaFlushAllocation(g_graphics_ctx->allocator, g_rs.buf_mem, slot * kRtSunSlotBytes, kRtSunSlotBytes);
	g_rs.cascades_used = static_cast<uint32_t>(cs.size());
	g_rs.texel_min = cs.front().texel, g_rs.texel_max = cs.back().texel;
	return true;
}

void RtSunSetFade(uint32_t slot, float fade) {
	if (g_rs.buf_mapped == nullptr) return;
	static const float strength = RtSunEnv("KYTY_RT_SUN_STRENGTH", 1.0f, 0.0f, 1.0f);
	float* f = reinterpret_cast<float*>(g_rs.buf_mapped + slot * kRtSunSlotBytes);
	f[3] = strength * std::clamp(fade, 0.0f, 1.0f);
	(void)vmaFlushAllocation(g_graphics_ctx->allocator, g_rs.buf_mem, slot * kRtSunSlotBytes, 16);
}

// (KYTY_RT_SUN) The statistics frame's words read back (v = word 64): the composite's counts, the conventions' votes, the
// bias histogram and R's probe (rt-sun.comp mode 7).
void RtSunTakeCalib(const uint32_t* v, double pixels) {
	g_rs.st_dark += v[0], g_rs.st_lift += v[1], g_rs.st_amount += v[2] / 256.0, g_rs.st_pixels += pixels;
	++g_rs.st_frames;
	const uint32_t* w = v + 16; // (word 80)
	for (int h = 0; h < 8; ++h) g_rs.votes[h] = w[h];
	int      best = 0;
	uint32_t second = 0;
	for (int h = 1; h < 8; ++h)
		if (w[h] > w[best]) best = h;
	for (int h = 0; h < 8; ++h)
		if (h != best) second = std::max(second, w[h]);
	static uint32_t logged = 0;
	const bool      decided_now = w[best] >= 100 && w[best] >= second * 5 / 2;
	static const char* conv[] {"z", "z/2+1/2", "1-z", "1/2-z/2"};
	if (decided_now && best != g_rs.hyp) {
		if (best == g_rs.cand) ++g_rs.cand_n;
		else g_rs.cand = best, g_rs.cand_n = 1;
		if (g_rs.cand_n >= 2) {
			const int was = g_rs.hyp;
			g_rs.hyp = best, g_rs.cand = -1, g_rs.cand_n = 0;
			g_rs.hist.fill(0.0);
			g_cs.history_valid = false;
			std::printf("RTSUN: cascades' convention %s: light clip y %s, stored depth = %s (lit receivers on their own depth: %u of the "
			            "samples, next best %u)\n",
			            was >= 0 ? "changed" : "decided", (best & 1) != 0 ? "down (row 0 = +1)" : "up", conv[best >> 1], w[best], second);
			std::fflush(stdout);
		}
	} else if (decided_now) {
		g_rs.cand = -1, g_rs.cand_n = 0;
	}
	if (logged < 6 || (g_rs.hyp < 0 && logged < 40)) {
		++logged;
		std::printf("RTSUN: calibration votes (lit receivers on their own depth, per convention):");
		for (int h = 0; h < 8; ++h) std::printf(" %d:%u", h, w[h]);
		std::printf("%s\n", g_rs.hyp >= 0 ? "" : decided_now ? " (candidate)" : " (undecided: look at sunlit ground and walls)");
		std::fflush(stdout);
	}
	if (g_rs.hyp >= 0) {
		// The bias: the 95th percentile of the lit cluster (stored - receiver height below +0.15 m), over the frames.
		double total = 0.0;
		for (int b = 0; b < 32; ++b) {
			g_rs.hist[b] = g_rs.hist[b] * 0.8 + w[8 + b];
			if (b < 25) total += g_rs.hist[b];
		}
		if (total >= 200.0) {
			double acc = 0.0;
			int    b   = 0;
			for (; b < 25; ++b)
				if ((acc += g_rs.hist[b]) >= 0.95 * total) break;
			const double value = std::clamp((b + 1) / 64.0 - 0.25, 0.005, 0.15);
			g_rs.bias_m        = g_rs.bias_m * 0.7 + value * 0.3;
		}
		const uint32_t cl = w[40], cs = w[43];
		g_rs.r_lit = cl, g_rs.r_shadow = cs;
		if (cl >= 60 && cs >= 60) {
			const double k  = std::exp2((static_cast<double>(w[41]) / cl - static_cast<double>(w[44]) / cs) / 32.0);
			const double nl = static_cast<double>(w[42]) / 256.0 / cl;
			const double r  = std::clamp((k - 1.0) / std::max(nl, 0.3), 0.0, 40.0);
			const double lr = std::log2(1.0 + r);
			g_rs.log_r      = g_rs.log_r < 0.0 ? lr : g_rs.log_r + (lr - g_rs.log_r) * 0.25;
		}
	}
}

// (KYTY_RT_SUN) The 10 s line.
void RtSunLog(double gpu_ms, uint32_t samples) {
	const double px = std::max(g_rs.st_pixels, 1.0);
	std::printf("RTSUN: stats: %.1f %% of pixels darkened, %.1f %% brightened (> 3 %%), mean change %.3f; convention %d (votes",
	            100.0 * g_rs.st_dark / px, 100.0 * g_rs.st_lift / px, g_rs.st_amount / px, g_rs.hyp);
	for (int h = 0; h < 8; ++h) std::printf(" %u", g_rs.votes[h]);
	std::printf("), bias %.3f m, sun/ambient R %.2f (%u lit / %u shadowed samples), %u cascades (texel %.3f..%.3f m), %llu copies; "
	            "GPU %.3f ms (%u samples)\n",
	            g_rs.bias_m, g_rs.log_r >= 0.0 ? std::exp2(g_rs.log_r) - 1.0 : -1.0, g_rs.r_lit, g_rs.r_shadow, g_rs.cascades_used,
	            g_rs.texel_min, g_rs.texel_max, static_cast<unsigned long long>(g_rs.copies), gpu_ms, samples);
	std::fflush(stdout);
	g_rs.st_dark = g_rs.st_lift = g_rs.st_amount = g_rs.st_pixels = 0.0;
	g_rs.st_frames = 0;
	if (g_cs.hw) {
		std::printf("HWRTSHADOW: 10 s: %u frames traced with this frame's TLAS, %u with an older one, %u without (level %d, trace %ux%u; "
		            "the GPU above is the whole sun pass)\n",
		            g_cs.hw_traced, g_cs.hw_stale, g_cs.hw_none, HwShadowLevel(), g_cs.trace.width, g_cs.trace.height);
		std::fflush(stdout);
		g_cs.hw_traced = g_cs.hw_stale = g_cs.hw_none = 0;
	}
}

void CShadowReadTiming() {
	if (g_cs.pool == nullptr || g_graphics_ctx == nullptr) return;
	while (g_cs.read + 4 <= g_cs.written) {
		const auto entry = static_cast<uint32_t>(g_cs.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (g_graphics_ctx->device.getQueryPoolResults(g_cs.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                               vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		g_cs.sum_ms += static_cast<double>(stamps[1] - stamps[0]) * g_cs.period_ns / 1e6;
		++g_cs.samples;
		++g_cs.read;
	}
}

// The statistics read back (4 frames after the composite wrote them) and, every 10 s, the log line.
void CShadowTakeStats() {
	if (g_cs.stat_frame >= 0 && static_cast<int64_t>(g_cs.frame) >= g_cs.stat_frame + 4 && g_cs.vote_mapped != nullptr) {
		g_cs.stat_frame = -1;
		(void)vmaInvalidateAllocation(g_graphics_ctx->allocator, g_cs.vote_mem, 0, VK_WHOLE_SIZE);
		const uint32_t* v = g_cs.vote_mapped + 64;
		g_cs.st_pixels += static_cast<double>(g_cs.render.width) * g_cs.render.height;
		g_cs.st_n += v[0], g_cs.st_occ += v[1] / 256.0, g_cs.st_depth += v[2] / 2.0;
		++g_cs.st_samples;
		if (RtSunLevel() > 0) RtSunTakeCalib(v, static_cast<double>(g_cs.render.width) * g_cs.render.height);
		// The light probe: geometric means of the sun-facing and facing-away luminance (log2 * 32 + 40 * 32 codes).
		if (v[4] >= 200 && v[6] >= 200) {
			const double lk = std::clamp((static_cast<double>(v[5]) / v[4] - static_cast<double>(v[7]) / v[6]) / 32.0, 0.0, 6.0);
			g_cs.log_k      = g_cs.log_k < 0.0 ? lk : g_cs.log_k + (lk - g_cs.log_k) * 0.2; // (~1.3 s at 60 fps)
		}
		// K 1.3 or less: overcast (the sun barely lights what faces it): 0.15; K 2.4 or more: full (sun-facing pixels in
		// the game's own shadows pull K down in full sun too).
		static const bool auto_sun = [] {
			const char* value = std::getenv("KYTY_RT_SHADOWS_OVERCAST");
			return !(value != nullptr && *value == '0');
		}();
		if (auto_sun && g_cs.log_k >= 0.0) {
			const double k = std::exp2(g_cs.log_k), x = std::clamp((k - 1.15) / (2.0 - 1.15), 0.0, 1.0); // (10-24 r3: gentler)
			g_cs.sun_factor = 0.5 + 0.5 * x * x * (3.0 - 2.0 * x);
		}
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - g_cs.last_log < std::chrono::seconds(10)) return;
	if (RtSunLevel() > 0 && (g_cs.st_samples > 0 || g_cs.samples > 0)) {
		RtSunLog(g_cs.samples > 0 ? g_cs.sum_ms / g_cs.samples : 0.0, g_cs.samples);
	} else if (g_cs.st_samples > 0 || g_cs.samples > 0) {
		const double n = g_cs.st_n, px = std::max(g_cs.st_pixels, 1.0);
		std::printf("RTSHADOW: stats: %.1f %% of pixels shadowed (occlusion > 0.1), mean occlusion %.3f (%.3f on those), their "
		            "mean depth %.1f m; sun lit/ambient K %.2f -> overcast factor %.2f; GPU %.3f ms (%u samples, %u frames)\n",
		            100.0 * n / px, g_cs.st_occ / px, n > 0.0 ? g_cs.st_occ / n : 0.0, n > 0.0 ? g_cs.st_depth / n : 0.0,
		            g_cs.log_k >= 0.0 ? std::exp2(g_cs.log_k) : 0.0, g_cs.sun_factor,
		            g_cs.samples > 0 ? g_cs.sum_ms / g_cs.samples : 0.0, g_cs.st_samples, g_cs.samples);
		std::fflush(stdout);
	}
	g_cs.st_pixels = g_cs.st_n = g_cs.st_occ = g_cs.st_depth = 0.0;
	g_cs.st_samples = 0;
	g_cs.sum_ms     = 0.0;
	g_cs.samples    = 0;
	g_cs.last_log   = now;
}

using Rows3 = std::array<std::array<double, 3>, 3>;

// The camera's world -> view rotation (rows: the view axes in world): View = P^-1 * view_proj (column vectors; P's
// column form is the transpose of the row-vector view_to_clip). Rows normalized; false when degenerate.
bool CShadowView(const std::array<float, 16>& m, const std::array<float, 16>& vp, Rows3& v) {
	std::array<float, 16> pc {};
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j) pc[i * 4 + j] = m[j * 4 + i];
	std::array<double, 16> pinv {};
	if (!Inverse4(pc, pinv)) return false;
	for (int i = 0; i < 3; ++i) {
		double n = 0.0;
		for (int j = 0; j < 3; ++j) {
			double d = 0.0;
			for (int k = 0; k < 4; ++k) d += pinv[i * 4 + k] * vp[k * 4 + j];
			v[i][j] = d;
			n += d * d;
		}
		n = std::sqrt(n);
		if (!(n > 1e-9) || !std::isfinite(n)) return false;
		for (int j = 0; j < 3; ++j) v[i][j] /= n;
	}
	// (A view-projection of another camera than the projection's (a shadow cascade, a reflection) leaves no rotation.)
	for (int i = 0; i < 3; ++i) {
		const auto& a = v[i];
		const auto& b = v[(i + 1) % 3];
		if (std::abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) > 0.05) return false;
	}
	return true;
}

std::array<double, 3> Norm3(const std::array<double, 3>& a) {
	const double n = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
	return n > 1e-12 ? std::array<double, 3> {a[0] / n, a[1] / n, a[2] / n} : std::array<double, 3> {0.0, 0.0, 0.0};
}
double Dot3(const std::array<double, 3>& a, const std::array<double, 3>& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
std::array<double, 3> ToView(const Rows3& v, const std::array<double, 3>& w) {
	return Norm3({Dot3(v[0], w), Dot3(v[1], w), Dot3(v[2], w)});
}
std::array<double, 3> ToWorld(const Rows3& v, const std::array<double, 3>& d) {
	std::array<double, 3> w {};
	for (int j = 0; j < 3; ++j) w[j] = v[0][j] * d[0] + v[1][j] * d[1] + v[2][j] * d[2];
	return Norm3(w);
}

// The hypotheses the switches leave (KYTY_RT_SHADOWS_UP, KYTY_RT_SHADOWS_FLIPY) as a 12-bit mask.
uint32_t CShadowAllowed() {
	static const uint32_t mask = [] {
		uint32_t    m  = 0xfffu;
		const char* up = std::getenv("KYTY_RT_SHADOWS_UP");
		if (up != nullptr && *up != '\0') {
			std::string v   = up;
			int         neg = 0;
			if (v[0] == '-') neg = 1, v.erase(0, 1);
			else if (v[0] == '+') v.erase(0, 1);
			const int axis = v == "x" ? 0 : v == "z" ? 2 : 1;
			m &= (1u << (axis * 2 + neg)) | (1u << (6 + axis * 2 + neg));
		}
		const char* flip = std::getenv("KYTY_RT_SHADOWS_FLIPY");
		if (flip != nullptr && *flip == '1') m &= 0x03fu;
		if (flip != nullptr && *flip == '0') m &= 0xfc0u;
		return m;
	}();
	return mask;
}

bool CShadowUpForced() {
	static const bool forced = [] {
		const char* up = std::getenv("KYTY_RT_SHADOWS_UP");
		return up != nullptr && *up != '\0';
	}();
	return forced;
}

// The camera's evidence on world up, every frame (view: the world -> view rows; m: view_to_clip, row vectors): a game
// camera does not roll, so its right row stays perpendicular to world up while it turns; its screen-up row (view y
// times the sign of m[5], which maps it to NDC y) leans toward up unless it looks > 70 degrees up or down.
void CShadowCamera(const Rows3& view, const std::array<float, 16>& m) {
	const double ys = m[5] < 0.0f ? -1.0 : 1.0;
	g_cs.cam_frames += 1.0;
	for (int k = 0; k < 3; ++k) {
		const double r = std::abs(view[0][k]), u = view[1][k] * ys;
		if (r < 0.1 && std::abs(u) > 0.35) g_cs.cam_ok[k] += 1.0, g_cs.cam_lean[k] += u;
	}
	if (g_cs.cam_frames > 4000.0) { // (a window: old views fade)
		g_cs.cam_frames *= 0.5;
		for (int k = 0; k < 3; ++k) g_cs.cam_ok[k] *= 0.5, g_cs.cam_lean[k] *= 0.5;
	}
	const double f = g_cs.cam_frames;
	// The axis: plausible in >= 90 % of >= 120 frames, every other in <= 75 % (the camera turned enough to tell).
	int axis = -1;
	for (int k = 0; k < 3 && f >= 120.0; ++k) {
		bool others = true;
		for (int j = 0; j < 3; ++j)
			if (j != k && g_cs.cam_ok[j] > 0.75 * f) others = false;
		if (g_cs.cam_ok[k] >= 0.9 * f && others) axis = k;
	}
	if (axis >= 0 && axis != g_cs.cam_axis) {
		static const char* names[] {"x", "y", "z"};
		std::printf("RTSHADOW: camera: world up is the %s axis (its right row perpendicular to it in %.0f %% of %.0f frames; "
		            "x %.0f %%, y %.0f %%, z %.0f %%)%s\n",
		            names[axis], 100.0 * g_cs.cam_ok[axis] / f, f, 100.0 * g_cs.cam_ok[0] / f, 100.0 * g_cs.cam_ok[1] / f,
		            100.0 * g_cs.cam_ok[2] / f, g_cs.cam_axis >= 0 ? " (changed)" : "");
		std::fflush(stdout);
		g_cs.cam_axis = axis;
	}
	// The hypotheses left: the decided axis (else every axis still plausible in >= 85 % of the frames), and for an
	// axis whose lean is clear the sign that puts up at the top of the screen (row 0) in each convention.
	uint32_t mask = 0;
	for (int k = 0; k < 3; ++k) {
		if (g_cs.cam_axis >= 0 ? k != g_cs.cam_axis : f >= 30.0 && g_cs.cam_ok[k] < 0.85 * f) continue;
		uint32_t bits = (3u << (k * 2)) | (3u << (6 + k * 2));
		const double lean = g_cs.cam_ok[k] > 0.0 ? g_cs.cam_lean[k] / g_cs.cam_ok[k] : 0.0;
		if (g_cs.cam_ok[k] >= 30.0 && std::abs(lean) > 0.3) {
			// Row 0 = NDC y +1 (hypotheses 0..5): up leans to NDC +y; mirrored (6..11): to NDC -y.
			bits = lean > 0.0 ? (1u << (k * 2)) | (1u << (6 + k * 2 + 1)) : (1u << (k * 2 + 1)) | (1u << (6 + k * 2));
		}
		mask |= bits;
	}
	g_cs.cam_mask = mask != 0 ? mask : 0xfffu;
}

// The hypotheses left by the switches and the camera (the switch alone when KYTY_RT_SHADOWS_UP is set or they disagree).
uint32_t CShadowLeft() {
	const uint32_t env = CShadowAllowed();
	if (CShadowUpForced()) return env;
	const uint32_t both = env & g_cs.cam_mask;
	return both != 0 ? both : env;
}

// The calibration read back (4 frames late) and the decision made (again).
void CShadowTakeCalib(uint32_t slot) {
	if (!g_cs.slot_voted[slot] || g_cs.vote_mapped == nullptr) return;
	g_cs.slot_voted[slot] = false;
	(void)vmaInvalidateAllocation(g_graphics_ctx->allocator, g_cs.vote_mem, 0, VK_WHOLE_SIZE);
	const uint32_t* v   = g_cs.vote_mapped + slot * 16;
	double          sum = 0.0;
	for (int i = 0; i < 12; ++i) g_cs.calib[i] += v[i], sum += g_cs.calib[i];
	if (sum > 4e6)
		for (auto& c: g_cs.calib) c *= 0.5; // (a window: old views fade)
	const uint32_t allowed = CShadowLeft();
	int            best = -1, second = -1;
	for (int i = 0; i < 12; ++i) {
		if ((allowed & (1u << i)) == 0) continue;
		if (best < 0 || g_cs.calib[i] > g_cs.calib[best]) second = best, best = i;
		else if (second < 0 || g_cs.calib[i] > g_cs.calib[second]) second = i;
	}
	if (best < 0) return;
	const double b = g_cs.calib[best], s = second >= 0 ? g_cs.calib[second] : 0.0;
	int          pick = -1;
	// A level camera cannot tell (up, row 0 = NDC +1) from its mirror twin (-up, mirrored): the GNM viewport's prior.
	const auto twin = [](int i) { return (i < 6 ? 6 : 0) + ((i % 6) / 2) * 2 + (1 - (i & 1)); };
	const int  gnm  = best < 6 ? best : twin(best);
	if (std::popcount(allowed) == 1) pick = best;
	else if (b >= 2000.0 && b >= 2.0 * s) pick = best;
	else if (b >= 2000.0 && second == twin(best) && (allowed & (1u << gnm)) != 0) pick = gnm;
	// The camera left only a twin pair: the prior unless the mirrored one has twice the floors.
	else if (g_cs.cam_axis >= 0 && std::popcount(allowed) == 2 && (allowed & (1u << twin(best))) != 0 && (allowed & (1u << gnm)) != 0)
		pick = gnm;
	if (pick < 0 || pick == g_cs.hyp) return;
	// A decided hypothesis changes only on clear evidence (3x).
	if (g_cs.hyp >= 0 && g_cs.calib[pick] < 3.0 * g_cs.calib[g_cs.hyp]) return;
	const int was = g_cs.hyp;
	g_cs.hyp      = pick;
	static const char* axes[] {"+x", "-x", "+y", "-y", "+z", "-z"};
	std::printf("RTSHADOW: calibration: world up %s, render row 0 = NDC y %s (floors %.0f vs next %.0f; camera %s)%s\n", axes[pick % 6],
	            pick < 6 ? "+1 (GNM viewport)" : "-1 (mirrored)", g_cs.calib[pick], s,
	            CShadowUpForced() ? "not asked (KYTY_RT_SHADOWS_UP)" : g_cs.cam_axis >= 0 ? "agrees" : "undecided, axis among those left",
	            was >= 0 ? " (changed)" : "");
	std::fflush(stdout);
	g_cs.history_valid = false;
}

bool CShadowApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int quality = ContactQuality();
	const auto said_once = [](const char* why) {
		static std::string said;
		if (said != why) {
			said = why;
			std::printf("RTSHADOW: off for now (%s)\n", why);
			std::fflush(stdout);
		}
	};
	if (quality == 0 || g_graphics_ctx == nullptr) return false;
	if (VideoOut::MoviePlaying()) return g_cs.history_valid = false, false;
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || inputs.motion.view == nullptr || color.extent.width < 16 ||
	    color.extent.height < 16)
		return said_once("missing color/depth/motion view"), false;
	const auto& m = inputs.view_to_clip;
	if (!(m[0] != 0.0f && m[5] != 0.0f && (m[11] != 0.0f || m[15] != 0.0f)))
		return said_once("no camera projection (view_to_clip)"), false;
	std::array<float, 16> vp {};
	{
		std::lock_guard vp_lock(g_last_vp_mutex);
		if (!g_have_last_vp) return said_once("no camera matrix captured yet"), false;
		vp = g_last_vp;
	}
	Rows3 view {};
	if (!CShadowView(m, vp, view)) return said_once("the camera matrix is degenerate"), false;
	if (!CShadowEnsure(*g_graphics_ctx, color.extent, quality)) return said_once("could not create its pipeline or images"), false;
	// (KYTY_RT_SUN: the cascades copied lately, their matrices into this frame's slot of the buffer; else nothing yet)
	const bool sun_mode = RtSunLevel() > 0;
	const uint32_t sun_slot = g_cs.frame % 4;
	// (KYTY_HWRT_SHADOW: the TLAS built this frame, before this pass on the same command stream; an older frame's while
	// this one built none; no rays until one was ever built)
	vk::AccelerationStructureKHR hw_tlas = nullptr;
	if (g_cs.hw) {
		bool built = false;
		hw_tlas    = HwScene::SceneTlas(&built);
		g_cs.hw_seen = g_cs.hw_seen || (hw_tlas != nullptr && built);
		if (hw_tlas == nullptr || !g_cs.hw_seen) {
			// (the pipeline's layout has the binding: no TLAS yet, no pass)
			++g_cs.hw_none;
			return said_once("KYTY_HWRT_SHADOW: no scene TLAS built yet"), false;
		}
		g_cs.hw_now = true;
		++(built ? g_cs.hw_traced : g_cs.hw_stale);
	}
	if (sun_mode && !RtSunPrepare(m, vp, sun_slot)) {
		// (10-25 godfix: the cascades are copied only on some frames; the latched sun still feeds the god rays and shine in
		// the meantime, in this frame's view, else they lost the sun 250 ms later and popped off and on.)
		if (g_cs.latched && g_cs.hyp >= 0 && g_cs.fade > 0.0f) {
			const auto l_view = ToView(view, g_cs.latch_axis);
			ShineSetSun(l_view, g_cs.fade * static_cast<float>(g_cs.sun_factor));
			GodraysSetSun(l_view, g_cs.fade * static_cast<float>(g_cs.sun_factor));
		}
		return said_once("KYTY_RT_SUN: no sun cascades copied yet (outdoors, in sunlight)"), false;
	}
	CShadowCamera(view, m);
	if (g_cs.hyp >= 0 && (CShadowLeft() & (1u << g_cs.hyp)) == 0) {
		static const char* names[] {"+x", "-x", "+y", "-y", "+z", "-z"};
		std::printf("RTSHADOW: calibration: world up %s (row 0 = NDC y %s) does not fit the camera: calibrating again\n",
		            names[g_cs.hyp % 6], g_cs.hyp < 6 ? "+1" : "-1");
		std::fflush(stdout);
		g_cs.hyp = -1, g_cs.history_valid = false;
	}

	const auto now = std::chrono::steady_clock::now();
	const float dt = g_cs.latch_time.time_since_epoch().count() == 0
	                     ? 0.0f
	                     : std::clamp(std::chrono::duration<float>(now - g_cs.latch_time).count(), 0.0f, 0.25f);
	g_cs.latch_time  = now;
	const float hold = std::max(RtShadowEnv("KYTY_RT_SHADOWS_HOLD", "KYTY_CONTACT_SHADOWS_HOLD", 20.0f, 600.0f), 1.0f);

	// The sun probe's axis (fresh: seen in the last 2 s of flips, by agreeing cascades or 3 flips running).
	std::array<double, 3> raw {};
	bool                  fresh = false, recent = false;
	int                   source = 0;
	const uint64_t        flips  = g_flips_taken.load(std::memory_order_relaxed);
	{
		std::lock_guard lock(g_sun.mutex);
		fresh  = g_sun.valid && (g_sun.agree >= 2 || g_sun.stable >= 3) && flips <= g_sun.flip + 120;
		recent = fresh && flips <= g_sun.flip + 2;
		raw    = {g_sun.axis[0], g_sun.axis[1], g_sun.axis[2]};
		source = g_sun.source;
	}
	raw = Norm3(raw);

	// The space: which reading of the axis stays put while the camera turns.
	static const int forced_space = [] {
		const char* value = std::getenv("KYTY_RT_SHADOWS_SPACE");
		const std::string v = value != nullptr ? value : "";
		return v == "world" ? 1 : v == "view" ? 2 : 0;
	}();
	if (recent && forced_space == 0 && g_cs.space_flip != flips) {
		g_cs.space_flip = flips;
		const std::array<double, 3> fwd = view[2];
		const std::array<double, 3> vw  = ToWorld(view, raw);
		if (!g_cs.space_ref) {
			g_cs.space_ref = true, g_cs.ref_axis = raw, g_cs.ref_vw = vw, g_cs.ref_fwd = fwd;
		} else if (Dot3(fwd, g_cs.ref_fwd) < 0.9659) { // (15 degrees)
			const double dr = std::abs(Dot3(raw, g_cs.ref_axis)), dv = std::abs(Dot3(vw, g_cs.ref_vw));
			if (dr > 0.9986 && dv < 0.9903) ++g_cs.world_votes; // (3 / 8 degrees)
			else if (dv > 0.9986 && dr < 0.9903) ++g_cs.view_votes;
			g_cs.ref_axis = raw, g_cs.ref_vw = vw, g_cs.ref_fwd = fwd;
			const int was = g_cs.space, was_used = was != 0 ? was : source == 2 ? 1 : 0;
			if (g_cs.world_votes >= g_cs.view_votes + 2) g_cs.space = 1;
			else if (g_cs.view_votes >= g_cs.world_votes + 2) g_cs.space = 2;
			if (was != g_cs.space) {
				std::printf("RTSHADOW: the sun axis is in %s space (camera turns: %d world, %d view)\n", g_cs.space == 1 ? "world" : "view",
				            g_cs.world_votes, g_cs.view_votes);
				std::fflush(stdout);
				if (was_used != 0 && was_used != g_cs.space) g_cs.latched = false, g_cs.fade = 0.0f; // (the other reading was in use)
			}
		}
	}
	const int space = forced_space != 0 ? forced_space : g_cs.space != 0 ? g_cs.space : source == 2 ? 1 : 0;

	// The calibration (world up, the screen's y convention): every frame until decided, then every 16th.
	const uint32_t slot = g_cs.frame % 4;
	CShadowTakeCalib(slot);
	const bool calibrate = g_cs.hyp < 0 || g_cs.frame % 16 == 0;

	CShadowPush push {};
	push.proj[0] = m[0], push.proj[1] = m[5], push.proj[2] = m[8], push.proj[3] = m[9];
	push.depth_map[0] = m[10], push.depth_map[1] = m[14], push.depth_map[2] = m[11], push.depth_map[3] = m[15];
	push.motion[0] = inputs.motion_scale_x, push.motion[1] = inputs.motion_scale_y;
	push.motion[2] = static_cast<float>(color.extent.width), push.motion[3] = static_cast<float>(color.extent.height);
	push.trace[0] = static_cast<float>(g_cs.trace.width), push.trace[1] = static_cast<float>(g_cs.trace.height);
	const float strength = RtShadowEnv("KYTY_RT_SHADOWS_STRENGTH", "KYTY_CONTACT_SHADOWS_STRENGTH", 0.5f, 1.0f) /* (10-24 r3: 0.6 looked thin) */;
	const float length   = std::max(RtShadowEnv("KYTY_RT_SHADOWS_LENGTH", "KYTY_CONTACT_SHADOWS_LENGTH", 1.5f, 10.0f), 0.25f); // (rtsart: 1.5 m: contact shadows; long rays only doubled the game's) // (10-24) 3 m (1 m was too short)
	const float fade_m   = std::max(RtShadowEnv("KYTY_RT_SHADOWS_FADE", nullptr, 100.0f, 600.0f), 2.0f); // (10-24) 100-200 m by default; far (dist-1023): up to 600
	push.trace[2] = strength, push.trace[3] = length;
	for (int k = 0; k < 3; ++k) {
		const auto c = ToView(view, {k == 0 ? 1.0 : 0.0, k == 1 ? 1.0 : 0.0, k == 2 ? 1.0 : 0.0});
		push.cand[k][0] = static_cast<float>(c[0]), push.cand[k][1] = static_cast<float>(c[1]), push.cand[k][2] = static_cast<float>(c[2]);
		push.cand[k][3] = 1.0f;
	}
	push.cand[3][0] = quality >= 3 ? 400.0f : quality >= 2 ? 280.0f : 160.0f; // (screen cap, px at 1440p; 10-24: longer)
	push.cand[3][1] = 1.0f;
	push.cand[3][2] = fade_m, push.cand[3][3] = fade_m * 2.0f;

	CShadowReadTiming();
	CShadowTakeStats();
	const auto global = [&](vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src, vk::PipelineStageFlags2 dst_stage,
	                        vk::AccessFlags2 dst) {
		vk::MemoryBarrier2 barrier {};
		barrier.srcStageMask  = src_stage;
		barrier.srcAccessMask = src;
		barrier.dstStageMask  = dst_stage;
		barrier.dstAccessMask = dst;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &barrier;
		command.pipelineBarrier2(dependency);
	};
	SsgiImage* all[] {&g_cs.raw, &g_cs.tmp_a, &g_cs.tmp_b, &g_cs.out, &g_cs.history[0], &g_cs.history[1], &g_cs.gbuf[0],
	                  &g_cs.gbuf[1], &g_cs.moments[0], &g_cs.moments[1], &g_cs.var_a, &g_cs.var_b};
	for (auto* img : all)
		if (img->image.state.layout != vk::ImageLayout::eGeneral)
			Transition(command, img->image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
			           vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader);
	global(vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	       vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	       vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite);
	const uint32_t cur  = g_cs.frame & 1u;
	const uint32_t prev = cur ^ 1u;
	push.misc[1] = static_cast<int32_t>(g_cs.frame);
	push.misc[2] = quality;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_cs.pipeline);
	const SsgiImage* aux_in  = &g_cs.moments[prev];
	const SsgiImage* aux_out = &g_cs.moments[cur];
	const auto run = [&](int mode, const SsgiImage& a, const SsgiImage& b, const SsgiImage& out, const SsgiImage& gout, float step,
	                     vk::Extent2D size) {
		const vk::Sampler       lin = g_ssgi.linear, pt = g_ssgi.point;
		vk::DescriptorImageInfo infos[11] {};
		infos[0]  = {lin, color.view, color.layout};
		infos[1]  = {pt, inputs.depth.view, inputs.depth.layout};
		infos[2]  = {pt, inputs.motion.view, inputs.motion.layout};
		infos[3]  = {lin, a.view, vk::ImageLayout::eGeneral};
		infos[4]  = {lin, b.view, vk::ImageLayout::eGeneral};
		infos[5]  = {pt, g_cs.gbuf[cur].view, vk::ImageLayout::eGeneral};
		infos[6]  = {pt, g_cs.gbuf[prev].view, vk::ImageLayout::eGeneral};
		infos[7]  = {nullptr, out.view, vk::ImageLayout::eGeneral};
		infos[8]  = {nullptr, gout.view, vk::ImageLayout::eGeneral};
		infos[9]  = {lin, aux_in->view, vk::ImageLayout::eGeneral};
		infos[10] = {nullptr, aux_out->view, vk::ImageLayout::eGeneral};
		vk::DescriptorBufferInfo vote_info {vk::Buffer(g_cs.vote), 0, VK_WHOLE_SIZE};
		// (KYTY_RT_SUN: 12 the cascades' copy, 13 this frame's slot of their buffer)
		const vk::DescriptorImageInfo  casc_info {pt, g_rs.view, vk::ImageLayout::eShaderReadOnlyOptimal};
		const vk::DescriptorBufferInfo sun_info {vk::Buffer(g_rs.buf), vk::DeviceSize {sun_slot} * kRtSunSlotBytes, kRtSunSlotBytes};
		const uint32_t                 count = g_cs.hw ? 15u : sun_mode ? 14u : 12u;
		vk::WriteDescriptorSet         writes[15] {};
		vk::WriteDescriptorSetAccelerationStructureKHR as_write {};
		as_write.accelerationStructureCount = 1;
		as_write.pAccelerationStructures    = &hw_tlas;
		for (uint32_t i = 0; i < count; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			if (i == 14) {
				writes[i].descriptorType = vk::DescriptorType::eAccelerationStructureKHR;
				writes[i].pNext          = &as_write;
			} else if (i == 11 || i == 13) {
				writes[i].descriptorType = vk::DescriptorType::eStorageBuffer;
				writes[i].pBufferInfo    = i == 11 ? &vote_info : &sun_info;
			} else if (i == 12) {
				writes[i].descriptorType = vk::DescriptorType::eCombinedImageSampler;
				writes[i].pImageInfo     = &casc_info;
			} else {
				writes[i].descriptorType = i < 7 || i == 9 ? vk::DescriptorType::eCombinedImageSampler : vk::DescriptorType::eStorageImage;
				writes[i].pImageInfo     = &infos[i];
			}
		}
		push.misc[0]  = mode;
		push.extra[0] = step;
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_cs.layout, 0, count, writes);
		command.pushConstants(g_cs.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
		global(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		       vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	};
	if (calibrate) {
		command.fillBuffer(vk::Buffer(g_cs.vote), slot * 64, 64, 0);
		global(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
		       vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
		push.misc[3] = static_cast<int32_t>(slot << 8);
		run(5, g_cs.tmp_a, g_cs.tmp_b, g_cs.raw, g_cs.gbuf[cur], 0.0f, vk::Extent2D {(color.extent.width + 3) / 4, (color.extent.height + 3) / 4});
		global(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite, vk::PipelineStageFlagBits2::eHost,
		       vk::AccessFlagBits2::eHostRead);
		g_cs.slot_voted[slot] = true;
	}
	// The statistics on every 16th frame (one in flight).
	const bool stats = g_cs.stat_frame < 0 && g_cs.frame % 16 == 8;
	if (stats) {
		command.fillBuffer(vk::Buffer(g_cs.vote), 64 * 4, (sun_mode ? 64 : 8) * 4, 0); // (statistics 64..66, light probe 68..71; KYTY_RT_SUN 80..124)
		global(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
		       vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
	}
	const int64_t this_frame = g_cs.frame;
	++g_cs.frame;

	// World up (the calibration's, +y until then) and the world-space sun toward the light, latched.
	const int hyp = g_cs.hyp;
	std::array<double, 3> up {};
	up[hyp >= 0 ? (hyp % 6) / 2 : 1] = hyp >= 0 && (hyp & 1) != 0 ? -1.0 : 1.0;
	if (fresh && space != 0 && hyp >= 0) { // (10-24: once up is known: its sign and the logged elevation are up's)
		std::array<double, 3> sun = space == 1 ? raw : ToWorld(view, raw);
		const double          h   = Dot3(sun, up);
		if (h < 0.0) sun = {-sun[0], -sun[1], -sun[2]};
		// (A sun on the horizon has no reliable sign: kept from the latch when there is one.)
		if (std::abs(h) < 0.03 && g_cs.latched && Dot3(sun, g_cs.latch_axis) < 0.0) sun = {-sun[0], -sun[1], -sun[2]};
		if (!g_cs.latched) {
			g_cs.latched = true, g_cs.latch_axis = sun, g_cs.switching = false;
			std::printf("RTSHADOW: sun latched, world direction to the light (%.4f %.4f %.4f), elevation %.1f deg\n", sun[0], sun[1],
			            sun[2], std::asin(std::clamp(std::abs(h), 0.0, 1.0)) * 57.29578);
			std::fflush(stdout);
		} else {
			const double c = Dot3(sun, g_cs.latch_axis);
			if (c >= 0.9848) { // (within 10 degrees: the same sun, smoothed over ~1 s)
				const double a = 1.0 - std::exp(-dt / 1.0);
				const auto&  l = g_cs.latch_axis;
				g_cs.latch_axis = Norm3({l[0] + (sun[0] - l[0]) * a, l[1] + (sun[1] - l[1]) * a, l[2] + (sun[2] - l[2]) * a});
				g_cs.switching  = false;
			} else if (!g_cs.switching) {
				g_cs.switching = true; // (a scene change: fade out, then take the new sun)
				std::printf("RTSHADOW: sun changed to (%.4f %.4f %.4f) from (%.4f %.4f %.4f): fading over\n", sun[0], sun[1], sun[2],
				            g_cs.latch_axis[0], g_cs.latch_axis[1], g_cs.latch_axis[2]);
				std::fflush(stdout);
			}
			g_cs.switch_axis = sun;
		}
		g_cs.last_fresh = now;
	}
	if (g_cs.latched && g_cs.switching && g_cs.fade <= 0.0f) g_cs.latch_axis = g_cs.switch_axis, g_cs.switching = false;
	const bool absent = g_cs.latched && now - g_cs.last_fresh > std::chrono::duration<float>(hold);
	if (absent && g_cs.fade <= 0.0f) {
		g_cs.latched = false;
		std::printf("RTSHADOW: sun unseen for %g s: dropped\n", hold);
		std::fflush(stdout);
	}
	// The fade: in over 1 s once everything is known, out over 0.3 s on a change or an absence.
	const bool target = g_cs.latched && !g_cs.switching && !absent && hyp >= 0 && space != 0;
	g_cs.fade         = std::clamp(g_cs.fade + (target ? dt / 1.0f : -dt / 0.3f), 0.0f, 1.0f);
	if (!g_cs.latched || g_cs.fade <= 0.0f) {
		g_cs.history_valid = false;
		return said_once(hyp < 0 ? "calibrating world up (turn the camera; look at some floor)"
		                 : !g_cs.latched ? (space == 0 && fresh ? "sun axis found; turn the camera to tell its space (world / view)"
		                                                        : "no sun direction yet")
		                                 : "faded out (sun changed or absent)"),
		       false;
	}
	const auto l_view = ToView(view, g_cs.latch_axis);
	ShineSetSun(l_view, g_cs.fade * static_cast<float>(g_cs.sun_factor)); // (KYTY_SHINE: shine.cpp)
	GodraysSetSun(l_view, g_cs.fade * static_cast<float>(g_cs.sun_factor)); // (KYTY_GODRAYS: godrays.cpp)
	push.extra[1] = static_cast<float>(l_view[0]), push.extra[2] = static_cast<float>(l_view[1]), push.extra[3] = static_cast<float>(l_view[2]);
	push.trace[2] = strength * g_cs.fade * g_cs.fade * (3.0f - 2.0f * g_cs.fade) * static_cast<float>(g_cs.sun_factor); // (smoothstep fade; overcast)
	push.tune[0]  = g_cs.log_k >= 0.0 ? static_cast<float>(1.0 / std::max(std::exp2(g_cs.log_k), 1.25)) : 0.0f;
	if (sun_mode) RtSunSetFade(sun_slot, g_cs.fade * g_cs.fade * (3.0f - 2.0f * g_cs.fade));
	if (!g_cs.said_on) {
		g_cs.said_on = true;
		static const char* axes[] {"+x", "-x", "+y", "-y", "+z", "-z"};
		std::printf("RTSHADOW: on (level %d, trace %ux%u of %ux%u, strength %g, length %g m, fade %g-%g m; light from %s, %s space; "
		            "world dir to the sun (%.4f %.4f %.4f), up %s, row 0 = NDC y %s; the light in view space now (%.3f %.3f %.3f), elevation %.1f deg)\n",
		            quality, g_cs.trace.width, g_cs.trace.height, color.extent.width, color.extent.height, strength, length, fade_m,
		            fade_m * 2.0f, source == 1 ? "the sun-shadow consumer's cascades" : "the shadow-map draws",
		            space == 1 ? "world" : "view", g_cs.latch_axis[0], g_cs.latch_axis[1], g_cs.latch_axis[2], axes[hyp % 6],
		            hyp < 6 ? "+1" : "-1", l_view[0], l_view[1], l_view[2],
		            std::asin(std::clamp(std::abs(Dot3(g_cs.latch_axis, up)), 0.0, 1.0)) * 57.29578);
		std::fflush(stdout);
	}
	const auto entry = static_cast<uint32_t>(g_cs.written % 8);
	const bool timed = g_cs.pool != nullptr;
	if (timed) {
		command.resetQueryPool(g_cs.pool, entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_cs.pool, entry * 2);
	}
	push.misc[3] = (inputs.reset ? 1 : 0) | (g_cs.history_valid ? 2 : 0) | (hyp < 6 ? 4 : 0); // (slot 0: statistics at word 64)
	// trace -> raw (+ gbuf[cur]); temporal (raw, history[prev]) -> tmp_a, count -> moments[cur]; a-trous 1 -> history[cur]
	// (next frame's history); a-trous 2 -> tmp_b; (a-trous 4 -> tmp_a); composite -> out.
	run(0, g_cs.tmp_a, g_cs.tmp_b, g_cs.raw, g_cs.gbuf[cur], 0.0f, g_cs.trace);
	run(1, g_cs.raw, g_cs.history[prev], g_cs.tmp_a, g_cs.tmp_b, 0.0f, g_cs.trace);
	aux_in = &g_cs.moments[cur], aux_out = &g_cs.var_a;
	run(2, g_cs.tmp_a, g_cs.raw, g_cs.history[cur], g_cs.tmp_b, 1.0f, g_cs.trace);
	run(2, g_cs.history[cur], g_cs.raw, g_cs.tmp_b, g_cs.raw, 2.0f, g_cs.trace);
	const SsgiImage* final_vis = &g_cs.tmp_b;
	if (quality >= 2 && !sun_mode) { // (KYTY_RT_SUN: steps 1, 2 only: wider a-trous would soften the contact-hardened edges)
		run(2, g_cs.tmp_b, g_cs.raw, g_cs.tmp_a, g_cs.raw, 4.0f, g_cs.trace);
		final_vis = &g_cs.tmp_a;
	}
	if (quality >= 3 && !sun_mode) { // (10-24: a wider pass: the penumbra soft, no hard shards)
		run(2, g_cs.tmp_a, g_cs.raw, g_cs.tmp_b, g_cs.raw, 8.0f, g_cs.trace);
		final_vis = &g_cs.tmp_b;
	}
	if (stats) // (the light probe: the game's sun strength; push.extra holds this frame's light)
		run(6, g_cs.tmp_a, g_cs.tmp_b, g_cs.raw, g_cs.gbuf[cur], 0.0f, vk::Extent2D {(color.extent.width + 3) / 4, (color.extent.height + 3) / 4});
	if (stats && sun_mode) // (KYTY_RT_SUN: the conventions' votes, the bias histogram, R's probe)
		run(7, g_cs.tmp_a, g_cs.tmp_b, g_cs.raw, g_cs.gbuf[cur], 0.0f, vk::Extent2D {(color.extent.width + 7) / 8, (color.extent.height + 7) / 8});
	aux_in = &g_cs.moments[cur], aux_out = &g_cs.var_b; // (the composite's stability mask: the history count in .x)
	if (stats) push.misc[3] |= 8;
	run(4, *final_vis, g_cs.raw, g_cs.out, g_cs.raw, 0.0f, g_cs.render);
	if (stats) {
		push.misc[3] &= ~8;
		global(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite, vk::PipelineStageFlagBits2::eHost,
		       vk::AccessFlagBits2::eHostRead);
		g_cs.stat_frame = this_frame;
	}
	global(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite, vk::PipelineStageFlagBits2::eAllCommands,
	       vk::AccessFlagBits2::eMemoryRead);
	if (timed) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_cs.pool, entry * 2 + 1);
		++g_cs.written;
	}
	g_cs.history_valid = true;
	inputs.color.image  = g_cs.out.image.image;
	inputs.color.view   = g_cs.out.view;
	inputs.color.layout = vk::ImageLayout::eGeneral;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = g_cs.out.image.usage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}
} // namespace

bool SsgiWanted() { return SsgiQuality() > 0; }

bool Upscale(vk::CommandBuffer command, const UpscaleInputs& inputs_in) {
	UpscaleInputs inputs = inputs_in;
	// (The DLSS-G hudless path sets the context in PrepareInputs; the PS5 Pro path alone needs the renderer's.)
	if (g_graphics_ctx == nullptr && inputs.graphics != nullptr) g_graphics_ctx = inputs.graphics;
	RtPartNoteColor(inputs.color_address, inputs.color.extent); // (rtparticles: the scene color the draws are matched to)
	CShadowApply(command, inputs); // (KYTY_RT_SHADOWS: direct-sun shadows first; inputs.color becomes the composited copy)
	HwLightsApply(command, inputs); // (KYTY_HWRT_LIGHTS: ray-traced local-light shadows, hw-lights.cpp; nothing when off)
	SsgiApply(command, inputs); // (KYTY_SSGI: likewise; nothing when off)
	GtaoApply(command, inputs); // (KYTY_GTAO: likewise, after SSGI; nothing when off)
	HwGiApply(command, inputs); // (KYTY_HWRT_AO / KYTY_HWRT_GI: likewise, hwrt-gi.cpp, in GTAO's place; nothing when off)
	SsrApply(command, inputs); // (KYTY_SSR: likewise, after GTAO; nothing when off)
	RtReflectionsApply(command, inputs); // (KYTY_RT_REFLECTIONS: likewise, rt-reflections.cpp; nothing when off)
	ShineApply(command, inputs); // (KYTY_SHINE: likewise, shine.cpp; nothing when off)
	GodraysApply(command, inputs); // (KYTY_GODRAYS: likewise, godrays.cpp; nothing when off)
	GbufViewApply(command, inputs); // (KYTY_GBUF_VIEW: the G-buffer diagnostic, last: gbuf-view.cpp; nothing when off)
	const bool  ours = g_sr_kind == 2 || g_sr_kind == 3;
	const bool  done = ours ? UpscaleOurs(command, inputs) : UpscaleStreamline(command, inputs);
	const bool  fsr  = g_sr_kind == 3;
	// (FSR: its RCAS, KYTY_UPSCALE_SHARPNESS defaulting to 0.3 in Fsr3::Upscale; DLSS and XeSS: 0 unless set.)
	const float amount = fsr ? [] {
		const char* value = std::getenv("KYTY_UPSCALE_SHARPNESS");
		return value != nullptr && *value != '\0' ? std::clamp(static_cast<float>(std::atof(value)), 0.0f, 1.0f) : 0.3f;
	}()
	                         : SharpenAmount();
	static bool said = false;
	if (!said) {
		said = true;
		std::printf("Super resolution sharpening: %g (%s)\n", amount,
		            fsr ? "FSR RCAS" : g_sr_kind == 2 ? "XeSS + sharpen.comp" : "DLSS + sharpen.comp");
		std::fflush(stdout);
	}
	if (done && !fsr && amount > 0.0f) SharpenOutput(command, inputs, amount);
	return done;
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
                  vk::Rect2D region, vk::Format format) {
	inputs.has_hudless_display = false;
	if (!inputs.valid || !inputs.has_hudless || inputs.hudless.image == nullptr) return;
	ConfigureAs(graphics, inputs.hudless_display, inputs.hudless_display_view, format,
	            vk::Extent3D {backbuffer.width, backbuffer.height, 1});
	Transition(command, inputs.hudless, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
	           vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eTransfer);
	Transition(command, inputs.hudless_display, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
	           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
	// Where RecordPresentCommands put the frame image (its present region: black bars at another aspect or with
	// KYTY_PRESENT_ASPECT=fit), black around it as there; the whole backbuffer otherwise, as before.
	const bool partial = region.extent.width != 0 && region.extent.height != 0 &&
	                     (region.offset.x != 0 || region.offset.y != 0 || region.extent != backbuffer);
	if (partial) {
		const vk::ClearColorValue       black {std::array<float, 4> {0.0f, 0.0f, 0.0f, 1.0f}};
		const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		command.clearColorImage(inputs.hudless_display.image, vk::ImageLayout::eTransferDstOptimal, &black, 1, &whole);
		Transition(command, inputs.hudless_display, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
		           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
	}
	vk::ImageBlit blit {};
	blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(inputs.hudless.extent.width),
                                       static_cast<int32_t>(inputs.hudless.extent.height), 1};
	blit.dstOffsets[1]  = vk::Offset3D {static_cast<int32_t>(backbuffer.width), static_cast<int32_t>(backbuffer.height), 1};
	if (partial) {
		blit.dstOffsets[0] = vk::Offset3D {region.offset.x, region.offset.y, 0};
		blit.dstOffsets[1] = vk::Offset3D {region.offset.x + static_cast<int32_t>(region.extent.width),
		                                   region.offset.y + static_cast<int32_t>(region.extent.height), 1};
	}
	command.blitImage(inputs.hudless.image, vk::ImageLayout::eTransferSrcOptimal, inputs.hudless_display.image,
	                  vk::ImageLayout::eTransferDstOptimal, 1, &blit, vk::Filter::eLinear);
	for (auto* image: {&inputs.hudless, &inputs.hudless_display})
		Transition(command, *image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
		           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
	inputs.has_hudless_display = true;
}

} // namespace Libs::Graphics::FrameGen
