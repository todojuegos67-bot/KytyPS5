#include "frame-gen.h"
#include "frame-gen-hudless.h"
#include "fsr-native.h"

#include "common/logging/log.h"
#include "graphics/guest_gpu/pm4.h"
#include "live-counters.h"
#include "local-platform.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "loader/demonsSoulsHdOutput.h"
#include "loader/demonsSoulsRenderSize.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>

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
#include <sl_helpers.h>
#include <sl_pcl.h>
#include <sl_reflex.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#endif

#ifdef KYTY_FRAMEGEN_FSR
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
// AMD FSR SDK 2.4's API headers (_Build/deps/fsr-sdk/Kits/FidelityFX, see CMakeLists.txt; MIT); its runtime is AMD's
// next to the emulator (FsrLoadRuntime). (Their pragmas are MSVC's: clang skips them.)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#include <api/include/ffx_api.h>
#include <api/include/ffx_api_loader.h>
#include <api/include/vk/ffx_api_vk.h>
#include <framegeneration/include/ffx_framegeneration.h>
#include <framegeneration/include/vk/ffx_api_framegeneration_vk.h>
#include <upscalers/include/ffx_upscale.h>
#pragma clang diagnostic pop

#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "gpu_blit_shaders/fsr_hud_mask_spv.h"
#include "gpu_blit_shaders/gpu_blit_fs_triangle_spv.h"

#include <cfloat>
#include <chrono>
#include <cinttypes>
#include <utility>
#include <cstdio>
#include <filesystem>
#include <string>
#endif

#if defined(KYTY_FRAMEGEN_FSR) || defined(KYTY_FRAMEGEN_NGX)
// FSR's HUD-less image (RecordHudless): the Windows FSR backend's, and FSR 3 built in on Linux (fsr-native.h).
#define KYTY_FRAMEGEN_FSR_HUD 1
#include "gpu_blit_shaders/fsr_hud_mask_spv.h"
#include "gpu_blit_shaders/gpu_blit_fs_triangle_spv.h"

#include <cinttypes>
#include <cstdio>
#endif

#ifdef KYTY_FRAMEGEN_NGX
// NVIDIA DLSS on Linux, where Streamline does not exist: NGX through the bridge libkyty_dlss.so (tools/local/dlss-bridge),
// loaded at run time, the only part that uses NVIDIA's SDK.
#include "kyty-dlss-bridge.h"

#include <dlfcn.h>

#include <chrono>
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
bool     g_ready       = false; // Streamline loaded and the feature available
bool     g_fsr_native  = false; // Linux KYTY_FRAMEGEN_BACKEND=fsr: FSR 3 built in (fsr-native.h), shown by the presenter
bool     g_upscale     = false; // KYTY_UPSCALER=dlss: DLSS super resolution asked for (Streamline loaded for it)
bool     g_upscale_ready = false; // ... and available on the device
Camera   g_camera;              // Kyty.Gpu: the current guest frame
Sources  g_sources;             // Kyty.Gpu: the current guest frame
uint64_t g_guest_frame = 0;     // Kyty.Gpu
std::atomic<bool> g_scene {false}; // the current guest frame ran the TAA pass
std::atomic<bool> g_last_scene {false}; // ... the last one the guest flipped
std::atomic<uint64_t> g_flips_taken {0}; // TakeScene calls (guest flips)
std::atomic<uint64_t> g_drawn_at_flip {0}; // the guest's indexed draw packets when TakeScene last ran
std::atomic<bool> g_menu {false}; // the current guest frame drew a menu (NoteMenu)
std::atomic<int> g_menu_hold {0}; // flips still to go in the middle after the last that drew a menu (TakeMenu)
std::atomic<bool> g_centred {false}; // the current guest frame's interface went to the middle (NoteCentred)
// Why the last frame had no inputs (PrepareInputs; the on/off lines' reason): 0 it had them, 1 no TAA pass, 2 no camera,
// 3 the TAA's images replaced since.
std::atomic<uint32_t> g_inputs_missing {0};

[[maybe_unused]] const char* MissingInputs() {
	switch (g_inputs_missing.load(std::memory_order_relaxed)) {
		case 1: return "no TAA pass: menus, movies, loading";
		case 2: return "no camera";
		default: return "the images replaced";
	}
}

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

// (Re)create `image` as a sampled copy target like `source` (of `size` when given: a blit target); with `attachment`, a
// color attachment too where the format allows it (*attachment tells).
void Configure(GraphicContext& graphics, VulkanImage& image, vk::ImageView& view,
               const VulkanImage& source, bool* attachment, vk::Extent2D size, bool* failed) {
	if (size.width == 0 || size.height == 0) size = {source.extent.width, source.extent.height};
	if (image.image != nullptr && image.format == source.format &&
	    image.extent.width == size.width && image.extent.height == size.height) {
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
	create.extent        = vk::Extent3D {size.width, size.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = source.format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
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
		// (`failed`: an image the frame can do without, the HUD canvas's copy: none this time.)
		if (failed != nullptr) {
			*failed = true;
			return;
		}
		EXIT("frame generation: cannot allocate a %ux%u input image\n", size.width, size.height);
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

#ifdef KYTY_FRAMEGEN_FSR_HUD
// `source` scaled into `target` (a color image; filtered).
void BlitInto(vk::CommandBuffer command, Image& source, VulkanImage& target) {
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	Transition(command, target, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
	           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
	vk::ImageBlit blit {};
	blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(source.backing.extent.width),
	                                    static_cast<int32_t>(source.backing.extent.height), 1};
	blit.dstOffsets[1]  = vk::Offset3D {static_cast<int32_t>(target.extent.width), static_cast<int32_t>(target.extent.height), 1};
	command.blitImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, target.image,
	                  vk::ImageLayout::eTransferDstOptimal, 1, &blit, vk::Filter::eLinear);
	Transition(command, target, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
	           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
}
#endif

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
	PFun_slPCLSetMarker*         pcl_set_marker      = nullptr;
	PFun_slReflexSleep*          reflex_sleep        = nullptr;
	PFun_slEvaluateFeature*      evaluate_feature    = nullptr;
	PFun_slDLSSSetOptions*       dlss_set_options    = nullptr;
	PFun_slDLSSGetOptimalSettings* dlss_get_optimal_settings = nullptr;
};
Api              g_api;
sl::FrameToken*  g_token = nullptr; // present thread: the frame being presented
uint32_t         g_frame_index = 0;
bool             g_options_set = false;
vk::Extent2D     g_options_backbuffer {};
vk::Extent2D     g_options_render {};
vk::Format       g_options_format = vk::Format::eUndefined; // SDR or HDR10 swapchain
uint64_t         g_presents = 0;
// DLSS-G on (generating) or off with its resources kept (eRetainResourcesWhenOff): off after TurnOffAfter presents in a
// row without inputs (menus, movies, loading), on after TurnOnAfter with them, so a fade or a cut does not flap it. On
// with null tags it was asked to generate frames it had no inputs for (Streamline "Failed to find global tag").
constexpr uint32_t TurnOffAfter = 30, TurnOnAfter = 10;
bool               g_mode_on    = false;
uint32_t           g_run_with   = 0, g_run_without = 0; // presents in a row with inputs, without them
// A frame with inputs the presenter left unshown (NoteDroppedFrame): the next one's motion spans two game frames.
std::atomic<bool>  g_discontinuity {false};

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

// How many times this error has come (its text after Streamline's time stamp and thread, "...[1066s:973ms:006us]").
uint64_t ErrorRepeats(const char* message) {
	std::string_view text(message);
	if (const auto stamp = text.find("us]"); stamp != std::string_view::npos) text.remove_prefix(stamp + 3);
	static std::mutex                             mutex;
	static std::unordered_map<uint64_t, uint64_t> counts;
	const uint64_t                                key = std::hash<std::string_view> {}(text);
	std::lock_guard                               lock(mutex);
	return ++counts[key];
}

void LogMessage(sl::LogType type, const char* message) {
	LOGF("Streamline[%d]: %s", static_cast<int>(type), message);
	if (type == sl::LogType::eError && message != nullptr && std::strstr(message, "[sl.dlss]") != nullptr &&
	    std::strstr(message, "evaluate feature failed") != nullptr)
		g_dlss_refusals.fetch_add(1, std::memory_order_relaxed);
	// KYTY_FRAMEGEN_VERBOSE: every message on stdout (the run log), not only errors.
	static const bool verbose = std::getenv("KYTY_FRAMEGEN_VERBOSE") != nullptr;
	if (type != sl::LogType::eError && !verbose) return;
	// The same error again: the first 3 printed, then every 1000th with its count. A tag Streamline refused every frame
	// printed ~115 lines a second, each flushed from the thread that evaluates (issues #85, #129: 143,323 lines).
	const uint64_t repeats = type == sl::LogType::eError && !verbose && message != nullptr ? ErrorRepeats(message) : 1;
	if (repeats <= 3) {
		std::printf("Streamline %s: %s", type == sl::LogType::eError ? "error" : "log", message);
		if (repeats == 3) std::printf("Streamline: that error 3 times; from now on every 1000th of it is printed\n");
	} else if (repeats % 1000 == 0) {
		std::printf("Streamline error (%" PRIu64 " times): %s", repeats, message);
	} else {
		return;
	}
	std::fflush(stdout);
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

// KYTY_FG_HUDLESS: the frame's image without its HUD, laid out as the backbuffer (`game`: where that shows the game
// image; frame-gen-hudless.h), or none for a frame without one (DLSS-G then takes the backbuffer as it is).
void TagHudless(sl::FrameToken& token, const FrameInputs& inputs, const sl::Extent& game) {
	const auto&  image = inputs.hudless.display;
	sl::Resource hudless(sl::ResourceType::eTex2d, static_cast<VkImage>(image.image), nullptr,
	                     static_cast<VkImageView>(inputs.hudless.display_view),
	                     static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
	hudless.width        = image.extent.width;
	hudless.height       = image.extent.height;
	hudless.nativeFormat = static_cast<uint32_t>(image.format);
	hudless.mipLevels    = 1;
	hudless.arrayLayers  = 1;
	hudless.flags        = 0;
	hudless.usage        = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image.usage));
	const sl::ResourceTag tag(inputs.hudless.has_display ? &hudless : nullptr, sl::kBufferTypeHUDLessColor,
	                          sl::ResourceLifecycle::eValidUntilPresent, &game);
	if (const auto result = g_api.set_tag_for_frame(token, sl::ViewportHandle {0}, &tag, 1, nullptr);
	    result != sl::Result::eOk && g_presents < 8) {
		LOGF("frame generation: slSetTagForFrame (HUD-less) failed (%d)\n", static_cast<int>(result));
	}
}
#endif

// KYTY_FRAMEGEN_BACKEND=fsr (UsesFsr): FSR 3.1 in place of Streamline.
bool g_fsr = false; // asked for and its runtime loaded
#ifdef KYTY_FRAMEGEN_FSR
struct FsrState {
	ffxFunctions api {};
	ffxContext   swapchain_context = nullptr; // the frame generation swapchain
	ffxContext   fg_context        = nullptr; // frame generation (made at the first frame with inputs)
	ffxQueryDescSwapchainReplacementFunctionsVK functions {};
	vk::SwapchainKHR swapchain = nullptr;
	vk::Extent2D     display {};
	vk::Format       format = vk::Format::eUndefined;
	bool             hdr    = false; // HDR10 (PQ) swapchain
	bool             sdk2   = false; // the runtime is FSR SDK 2's loader, not FidelityFX SDK 1.1.4's DLL (FsrLoadRuntime)
	std::atomic<bool> fsr4 {false};  // fg_context is FSR 4's frame generation (the KYTY_FPS_HUD panel's name: FsrName)
	vk::Extent2D     max_render {};  // fg_context's
	uint64_t         frame_id = 0;
	bool             mode_on  = false;
	bool             reset    = false; // the next generated frame starts a new history
	uint32_t         run_with = 0, run_without = 0; // presents in a row with inputs, without them
	std::chrono::steady_clock::time_point last_frame {};
	GraphicContext*  graphics = nullptr;
};
FsrState g_fsr_state;
// FSR: generation on after FsrTurnOnAfter frames with inputs in a row; off after FsrTurnOffAfter without them (those
// before are not presented: a frame without inputs between generated ones would be shown as a still).
constexpr uint32_t FsrTurnOnAfter = 10, FsrTurnOffAfter = 4;
std::atomic<bool>  g_fsr_discontinuity {false};

bool FsrOk(ffxReturnCode_t code, const char* what) {
	if (code == FFX_API_RETURN_OK) return true;
	LOGF("frame generation: FSR %s failed (%u)\n", what, static_cast<uint32_t>(code));
	std::printf("Frame generation: FSR %s failed (%u)\n", what, static_cast<uint32_t>(code));
	std::fflush(stdout);
	return false;
}

void FsrMessage(uint32_t type, const wchar_t* message) {
	char text[1024] {};
	WideCharToMultiByte(CP_UTF8, 0, message, -1, text, static_cast<int>(sizeof(text)) - 1, nullptr, nullptr);
	LOGF("FSR[%u]: %s\n", type, text);
	if (std::getenv("KYTY_FRAMEGEN_VERBOSE") != nullptr || type == FFX_API_MESSAGE_TYPE_ERROR) {
		std::printf("FSR %s: %s\n", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", text);
		std::fflush(stdout);
	}
}

// AMD's runtime next to the emulator, only AMD's own (once): FSR SDK 2's loader, amd_fidelityfx_loader_vk.dll, which
// takes the effects from amd_fidelityfx_upscaler_vk.dll and amd_fidelityfx_framegeneration_vk.dll beside it (FSR 4 on
// the Radeon cards AMD lists for it, FSR 3.1 on the others), else FidelityFX SDK 1.1.4's amd_fidelityfx_vk.dll (FSR
// 3.1). False: none.
bool FsrLoadRuntime() {
	static const bool loaded = [] {
		auto&   s = g_fsr_state;
		wchar_t exe[MAX_PATH] {};
		GetModuleFileNameW(nullptr, exe, MAX_PATH);
		const auto      folder = std::filesystem::path(exe).parent_path();
		std::error_code error;
		const auto present = [&](const wchar_t* name) { return std::filesystem::exists(folder / name, error); };
		const auto amd     = [&](const wchar_t* name) { return LocalPlatform::IsAmdSigned((folder / name).wstring()); };
		HMODULE    module  = nullptr;
		if (present(L"amd_fidelityfx_loader_vk.dll")) {
			// (The loader loads the effects beside it by their names: each must be AMD's too.)
			bool ok = amd(L"amd_fidelityfx_loader_vk.dll");
			for (const wchar_t* effect: {L"amd_fidelityfx_upscaler_vk.dll", L"amd_fidelityfx_framegeneration_vk.dll"}) {
				if (present(effect) && !amd(effect)) ok = false;
			}
			if (ok) {
				module = LoadLibraryW((folder / L"amd_fidelityfx_loader_vk.dll").c_str());
				s.sdk2 = module != nullptr;
			} else {
				std::printf("FSR: amd_fidelityfx_loader_vk.dll or an effect DLL beside it is not signed by AMD, not loaded\n");
			}
		}
		if (module == nullptr && present(L"amd_fidelityfx_vk.dll")) {
			if (amd(L"amd_fidelityfx_vk.dll")) {
				module = LoadLibraryW((folder / L"amd_fidelityfx_vk.dll").c_str());
			} else {
				std::printf("FSR: amd_fidelityfx_vk.dll is not signed by AMD, not loaded\n");
			}
		}
		if (module == nullptr) {
			LOGF("FSR: no runtime (amd_fidelityfx_loader_vk.dll or amd_fidelityfx_vk.dll)\n");
			std::fflush(stdout);
			return false;
		}
		ffxLoadFunctions(&s.api, module);
		if (s.api.CreateContext == nullptr || s.api.Configure == nullptr || s.api.Dispatch == nullptr ||
		    s.api.Query == nullptr || s.api.DestroyContext == nullptr) {
			std::printf("FSR: AMD's runtime lacks an entry point, FSR off\n");
			std::fflush(stdout);
			s.sdk2 = false;
			return false;
		}
		ffxConfigureDescGlobalDebug1 debug {};
		debug.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1;
		debug.fpMessage   = FsrMessage;
		debug.debugLevel  = std::getenv("KYTY_FRAMEGEN_VERBOSE") != nullptr ? FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE
		                                                                    : FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_ERRORS;
		(void)s.api.Configure(nullptr, &debug.header);
		std::printf("FSR: AMD's runtime %s\n", s.sdk2 ? "amd_fidelityfx_loader_vk.dll (FSR SDK 2)"
		                                              : "amd_fidelityfx_vk.dll (FidelityFX SDK 1.1.4)");
		std::fflush(stdout);
		return true;
	}();
	return loaded;
}

// The Vulkan backend as the loaded runtime takes it: FSR SDK 2's description (a VkDeviceContext), or SDK 1.1.4's
// older one (which SDK 2 keeps for its FSR 3.1.4 provider only). Made where it is chained (Header points into it).
struct FsrBackend {
	ffxCreateBackendVKDesc            current {};
	ffxCreateBackendVKDesc_DEPRECATED old {};

	explicit FsrBackend(GraphicContext& graphics) {
		current.header.type                    = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
		current.deviceContext.vkDevice         = static_cast<VkDevice>(graphics.device);
		current.deviceContext.vkPhysicalDevice = static_cast<VkPhysicalDevice>(graphics.physical_device);
		current.deviceContext.vkDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
		old.header.type                        = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK_DEPRECATED;
		old.vkDevice                           = current.deviceContext.vkDevice;
		old.vkPhysicalDevice                   = current.deviceContext.vkPhysicalDevice;
		old.vkDeviceProcAddr                   = current.deviceContext.vkDeviceProcAddr;
	}
	ffxApiHeader* Header() { return g_fsr_state.sdk2 ? &current.header : &old.header; }
};

// The provider AMD's runtime chose for a context (its version: "4.1.2" is FSR 4, "3.1.x" FSR 3.1); "?" when unsaid.
std::string FsrProvider(ffxContext* context) {
	ffxQueryGetProviderVersion version {};
	version.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
	if (g_fsr_state.api.Query(context, &version.header) != FFX_API_RETURN_OK || version.versionName == nullptr) {
		return "?";
	}
	return version.versionName;
}

// The video memory AMD's runtime says a context holds, as ", N MiB of video memory" for its log line ("" when unsaid):
// its own allocations, which the emulator sees only as the driver's share (issue #276: on an 8 GB RX 6650 XT, FSR SDK
// 2's frame generation took about 1.1 GB more of it than SDK 1.1.4's).
template <typename Query>
std::string FsrMemory(ffxContext* context, uint64_t type, FfxApiEffectMemoryUsage* Query::* output) {
	FfxApiEffectMemoryUsage usage {};
	Query                   query {};
	query.header.type = type;
	query.*output     = &usage;
	if (*context == nullptr || g_fsr_state.api.Query(context, &query.header) != FFX_API_RETURN_OK || usage.totalUsageInBytes == 0) {
		return "";
	}
	return ", " + std::to_string((usage.totalUsageInBytes + (1u << 19u)) >> 20u) + " MiB of video memory";
}

FfxApiResource FsrImage(const VulkanImage& image) {
	VkImageCreateInfo info {};
	info.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	info.imageType   = VK_IMAGE_TYPE_2D;
	info.format      = static_cast<VkFormat>(image.format);
	info.extent      = {image.extent.width, image.extent.height, 1};
	info.mipLevels   = 1;
	info.arrayLayers = 1;
	info.usage       = static_cast<VkImageUsageFlags>(image.usage);
	return ffxApiGetResourceVK(static_cast<VkImage>(image.image), ffxApiGetImageResourceDescriptionVK(static_cast<VkImage>(image.image), info, 0),
	                           FFX_API_RESOURCE_STATE_COMPUTE_READ);
}

// The game's world->clip matrix split as in Decompose (Streamline's): the camera's axes and position in world space,
// the vertical scale of its projection and its near plane (reversed-Z, infinite far).
struct FsrCamera {
	float position[3] {}, up[3] {}, right[3] {}, forward[3] {};
	float sy = 1.0f, z_near = 0.1f;
	bool  valid = false;
};

FsrCamera FsrDecompose(const std::array<float, 16>& m) {
	FsrCamera  c;
	const auto row = [&](uint32_t i) { return std::array<double, 4> {m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]}; };
	const auto r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
	const double flen = std::sqrt(r3[0] * r3[0] + r3[1] * r3[1] + r3[2] * r3[2]);
	if (flen < 1e-6) return c;
	const double f[3] {r3[0] / flen, r3[1] / flen, r3[2] / flen};
	const double tf = r3[3] / flen;
	const auto split = [&](const std::array<double, 4>& r, double& s, double axis[3], double& t) {
		const double off = (r[0] * f[0] + r[1] * f[1] + r[2] * f[2]) / flen;
		double v[3] {r[0] / flen - off * f[0], r[1] / flen - off * f[1], r[2] / flen - off * f[2]};
		s = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		if (s < 1e-9) return false;
		for (int i = 0; i < 3; ++i) axis[i] = v[i] / s;
		t = (r[3] / flen - off * tf) / s;
		return true;
	};
	double sx = 0, sy = 0, x[3] {}, y[3] {}, tx = 0, ty = 0;
	if (!split(r0, sx, x, tx) || !split(r1, sy, y, ty)) return c;
	const double up[3] {f[1] * x[2] - f[2] * x[1], f[2] * x[0] - f[0] * x[2], f[0] * x[1] - f[1] * x[0]};
	if (up[0] * y[0] + up[1] * y[1] + up[2] * y[2] < 0) {
		sy = -sy;
		ty = -ty;
	}
	for (int i = 0; i < 3; ++i) {
		c.right[i]    = float(x[i]);
		c.up[i]       = float(up[i]);
		c.forward[i]  = float(f[i]);
		c.position[i] = float(-(tx * x[i] + ty * up[i] + tf * f[i]));
	}
	c.sy     = float(sy);
	c.z_near = float(r2[3] / flen);
	c.valid  = true;
	return c;
}

// Frame generation off on the swapchain: it then presents the frames as they are (before either goes).
void FsrStopFrameGeneration() {
	auto& s = g_fsr_state;
	if (s.fg_context == nullptr || s.swapchain == nullptr) return;
	ffxConfigureDescFrameGeneration config {};
	config.header.type            = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
	config.swapChain              = static_cast<VkSwapchainKHR>(s.swapchain);
	config.frameGenerationEnabled = false;
	config.frameID                = ++s.frame_id;
	(void)s.api.Configure(&s.fg_context, &config.header);
}

void FsrDestroyFrameGeneration() {
	auto& s = g_fsr_state;
	if (s.fg_context == nullptr) return;
	FsrStopFrameGeneration();
	(void)s.api.DestroyContext(&s.fg_context, nullptr);
	s.fg_context = nullptr;
	s.mode_on    = false;
}

// The frame generation context for render sizes up to `render` (made again for a larger one).
bool FsrEnsureFrameGeneration(vk::Extent2D render) {
	auto& s = g_fsr_state;
	if (s.fg_context != nullptr && render.width <= s.max_render.width && render.height <= s.max_render.height) return true;
	FsrDestroyFrameGeneration();
	FsrBackend backend(*s.graphics);
	// (FSR SDK 2 wants the API version the emulator was built against.)
	ffxCreateContextDescFrameGenerationVersion version {};
	version.header.type  = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION;
	version.header.pNext = backend.Header();
	version.version      = FFX_FRAMEGENERATION_VERSION;
	ffxCreateContextDescFrameGeneration create {};
	create.header.type  = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
	create.header.pNext = s.sdk2 ? &version.header : backend.Header();
	// The game's depth: reversed-Z with an infinite far plane (Decompose); its vectors at the render size.
	create.flags = FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED | FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE;
	if (std::getenv("KYTY_FRAMEGEN_DEBUG") != nullptr) create.flags |= FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING;
	create.displaySize      = {s.display.width, s.display.height};
	s.max_render            = {std::max(render.width, s.display.width), std::max(render.height, s.display.height)};
	create.maxRenderSize    = {s.max_render.width, s.max_render.height};
	create.backBufferFormat = ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(s.format));
	if (!FsrOk(s.api.CreateContext(&s.fg_context, &create.header, nullptr), "frame generation context")) {
		s.fg_context = nullptr;
		return false;
	}
	LOGF("frame generation: FSR context %ux%u, render up to %ux%u\n", s.display.width, s.display.height,
	     s.max_render.width, s.max_render.height);
	const auto provider = FsrProvider(&s.fg_context);
	s.fsr4.store(provider.rfind("4.", 0) == 0, std::memory_order_relaxed);
	const auto memory    = FsrMemory(&s.fg_context, FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE,
	                                 &ffxQueryDescFrameGenerationGetGPUMemoryUsage::gpuMemoryUsageFrameGeneration);
	const auto swapchain = FsrMemory(&s.swapchain_context, FFX_API_QUERY_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_GPU_MEMORY_USAGE_VK,
	                                 &ffxQueryFrameGenerationSwapChainGetGPUMemoryUsageVK::gpuMemoryUsageFrameGenerationSwapchain);
	std::printf("Frame generation: FSR ready (%ux%u, render up to %ux%u; frame generation %s%s%s)\n", s.display.width,
	            s.display.height, s.max_render.width, s.max_render.height, provider.c_str(), memory.c_str(),
	            swapchain.empty() ? "" : ("; its swapchain" + swapchain).c_str());
	std::fflush(stdout);
	return true;
}

// An upscaler input or output as AMD's runtime takes it: the image's own format (it makes its views: a depth image's
// with the depth aspect; the game's depth is a D32S8 image viewed as R32), the state its layout is in.
FfxApiResource FsrResource(const UpscaleImage& image, bool written) {
	VkImageCreateInfo info {};
	info.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	info.imageType   = VK_IMAGE_TYPE_2D;
	info.format      = static_cast<VkFormat>(image.image_format != vk::Format::eUndefined ? image.image_format : image.format);
	info.extent      = {image.extent.width, image.extent.height, 1};
	info.mipLevels   = 1;
	info.arrayLayers = 1;
	info.samples     = VK_SAMPLE_COUNT_1_BIT;
	info.usage       = static_cast<VkImageUsageFlags>(image.usage);
	const uint32_t state = written ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS
	                       : image.layout == vk::ImageLayout::eGeneral ? FFX_API_RESOURCE_STATE_COMMON
	                                                                   : FFX_API_RESOURCE_STATE_COMPUTE_READ;
	return ffxApiGetResourceVK(static_cast<VkImage>(image.image),
	                           ffxApiGetImageResourceDescriptionVK(static_cast<VkImage>(image.image), info,
	                                                               written ? FFX_API_RESOURCE_USAGE_UAV : FFX_API_RESOURCE_USAGE_READ_ONLY),
	                           state);
}

// AMD's upscaling through FSR SDK 2's runtime (UpscaleWithAmdRuntime), from the inputs FSR 3 built in takes
// (fsr-native.cpp): the game's linear HDR scene before its tone mapping, reversed-Z with an infinite far plane.
struct FsrRuntimeUpscaler {
	std::mutex   mutex;
	ffxContext   context = nullptr;
	vk::Extent2D render {}, display {};
	bool         failed = false, reset = true;
	std::chrono::steady_clock::time_point last {};

	void Destroy(GraphicContext& graphics) {
		if (context == nullptr) return;
		{
			Common::LockGuard lock(graphics.queue_mutex);
			(void)graphics.queue.waitIdle();
		}
		(void)g_fsr_state.api.DestroyContext(&context, nullptr);
		context = nullptr;
	}

	bool Fail(const char* what, ffxReturnCode_t code) {
		failed = true;
		std::printf("Super resolution: AMD's FSR %s failed (%u); FSR 3 built in from now on\n", what,
		            static_cast<uint32_t>(code));
		std::fflush(stdout);
		return false;
	}

	bool Create(GraphicContext& graphics, vk::Extent2D render_size, vk::Extent2D display_size) {
		render  = render_size;
		display = display_size;
		FsrBackend backend(graphics);
		// (FSR SDK 2 wants the API version the emulator was built against.)
		ffxCreateContextDescUpscaleVersion version {};
		version.header.type  = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
		version.header.pNext = backend.Header();
		version.version      = FFX_UPSCALER_VERSION;
		ffxCreateContextDescUpscale create {};
		create.header.type  = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
		create.header.pNext = &version.header;
		create.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
		               FFX_UPSCALE_ENABLE_DEPTH_INFINITE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
		if (std::getenv("KYTY_FSR_DEBUG") != nullptr) create.flags |= FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
		create.maxRenderSize  = {render.width, render.height};
		create.maxUpscaleSize = {display.width, display.height};
		create.fpMessage      = FsrMessage;
		if (const auto code = g_fsr_state.api.CreateContext(&context, &create.header, nullptr); code != FFX_API_RETURN_OK) {
			context = nullptr;
			return Fail("upscaling context", code);
		}
		reset = true;
		std::printf("Super resolution: AMD FSR %s %ux%u -> %ux%u (AMD's runtime%s)\n", FsrProvider(&context).c_str(),
		            render.width, render.height, display.width, display.height,
		            FsrMemory(&context, FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE,
		                      &ffxQueryDescUpscaleGetGPUMemoryUsage::gpuMemoryUsageUpscaler)
		                .c_str());
		std::fflush(stdout);
		return true;
	}

	bool Record(GraphicContext& graphics, vk::CommandBuffer command, const UpscaleInputs& inputs) {
		std::lock_guard lock(mutex);
		if (failed) return false;
		// (The runtime makes its own views of mip level 0 and layer 0.)
		for (const auto* image: {&inputs.color, &inputs.depth, &inputs.motion, &inputs.output}) {
			if (image->base_level != 0 || image->base_layer != 0) return false;
		}
		if (context != nullptr && (render != inputs.color.extent || display != inputs.output.extent)) Destroy(graphics);
		if (context == nullptr && !Create(graphics, inputs.color.extent, inputs.output.extent)) return false;
		static const float sharpness = [] {
			const char* value = std::getenv("KYTY_FSR_SHARPNESS");
			return value != nullptr ? std::clamp(std::strtof(value, nullptr), 0.0f, 1.0f) : 0.6f;
		}();
		ffxDispatchDescUpscale dispatch {};
		dispatch.header.type   = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
		dispatch.commandList   = static_cast<VkCommandBuffer>(command);
		dispatch.color         = FsrResource(inputs.color, false);
		dispatch.depth         = FsrResource(inputs.depth, false);
		dispatch.motionVectors = FsrResource(inputs.motion, false);
		dispatch.output        = FsrResource(inputs.output, true);
		// (The game's reactive mask: the pixels taken from this frame rather than the history, issue #232.)
		if (inputs.reactive.image) dispatch.reactive = FsrResource(inputs.reactive, false);
		// (Render pixels, as DLSS takes them; the game's vectors times their scale: render pixels to the previous position.)
		dispatch.jitterOffset      = {inputs.jitter_x, inputs.jitter_y};
		dispatch.motionVectorScale = {inputs.motion_scale_x, inputs.motion_scale_y};
		dispatch.renderSize        = {render.width, render.height};
		dispatch.upscaleSize       = {display.width, display.height};
		dispatch.enableSharpening  = sharpness > 0.0f;
		dispatch.sharpness         = sharpness;
		const auto now             = std::chrono::steady_clock::now();
		dispatch.frameTimeDelta    = last.time_since_epoch().count() == 0
		                                 ? 16.7f
		                                 : std::chrono::duration<float, std::milli>(now - last).count();
		last                       = now;
		dispatch.preExposure       = 1.0f;
		dispatch.reset             = reset || inputs.reset;
		dispatch.cameraNear        = inputs.near_plane > 0.0f ? inputs.near_plane : 0.1f;
		dispatch.cameraFar         = FLT_MAX;
		// (Row vectors: the projection's y scale is cot(fov / 2).)
		const float y_scale              = std::fabs(inputs.view_to_clip[5]);
		dispatch.cameraFovAngleVertical  = y_scale > 0.0f ? 2.0f * std::atan(1.0f / y_scale) : 1.0f;
		dispatch.viewSpaceToMetersFactor = 1.0f;
		// (What it was given, as the built-in FSR 3 says it: issue #271, FSR 4's picture black under the HUD.)
		if (static uint32_t reports = 0; reports < 2) {
			++reports;
			std::printf("Super resolution: AMD FSR inputs: color %s, depth %s, motion %s, output %s %ux%u; jitter %g %g, motion "
			            "scale %g %g, near %g, vertical field of view %g, frame time %.1f ms, sharpness %g, reset %d\n",
			            vk::to_string(inputs.color.format).c_str(), vk::to_string(inputs.depth.format).c_str(),
			            vk::to_string(inputs.motion.format).c_str(), vk::to_string(inputs.output.format).c_str(),
			            display.width, display.height, dispatch.jitterOffset.x, dispatch.jitterOffset.y,
			            dispatch.motionVectorScale.x, dispatch.motionVectorScale.y, dispatch.cameraNear,
			            dispatch.cameraFovAngleVertical, dispatch.frameTimeDelta, dispatch.enableSharpening ? sharpness : 0.0f,
			            dispatch.reset ? 1 : 0);
			std::fflush(stdout);
		}
		if (const auto code = g_fsr_state.api.Dispatch(&context, &dispatch.header); code != FFX_API_RETURN_OK) {
			return Fail("upscaling", code);
		}
		reset = false;
		return true;
	}
};

FsrRuntimeUpscaler& TheRuntimeUpscaler() {
	static auto* upscaler = new FsrRuntimeUpscaler;
	return *upscaler;
}

#endif

#ifdef KYTY_FRAMEGEN_FSR_HUD
// The game's interface canvas (NoteUiCanvas: its guest address; the GPU thread's): 3840x2160, the HD output's
// 1920x1080, or KYTY_OUTPUT_HEIGHT's output where the interface is scaled to it (loader/demonsSoulsHdOutput.h). With
// the wide output (KYTY_RENDER_ASPECT) as wide as that output: 5120x1440 at 32:9 and 1440, where FSR looked for a
// 2560x1440 canvas, found none and generated the HUD too, which ghosted (issue #250).
std::atomic<uint64_t> g_ui_canvas {0};

vk::Extent2D UiCanvasExtent() {
	namespace Hd = ::Loader::DemonsSoulsHdOutput;
	const uint32_t wide = ::Loader::DemonsSoulsRenderSize::WideOutputWidth();
	if (Hd::InterfaceScaled() || Hd::InterfaceAtOutput())
		return vk::Extent2D {wide != 0 ? wide : Hd::UpscaledWidth(), Hd::UpscaledHeight()};
	return Hd::Active() ? vk::Extent2D {Hd::HdWidth, Hd::HdHeight} : vk::Extent2D {wide != 0 ? wide : 3840u, 2160};
}

// FSR's HUD-less images (RecordHudless): the backbuffer with the HUD painted over by the scene around it, two in turn
// (the interpolation of one may still read it while the next frame's is made).
struct FsrHud {
	vk::Format                   format = vk::Format::eUndefined; // the pipeline's
	vk::Extent2D                 extent {};                       // the images'
	std::array<VulkanImage, 2>   images {};
	std::array<vk::ImageView, 2> views {};
	VulkanImage                  source;                // the backbuffer copied, the pass's input
	vk::ImageView                source_view = nullptr;
	vk::Sampler                  sampler    = nullptr;
	vk::DescriptorSetLayout      set_layout = nullptr;
	vk::PipelineLayout           layout     = nullptr;
	vk::ShaderModule             vertex = nullptr, fragment = nullptr;
	vk::Pipeline                 pipeline = nullptr;
	uint32_t                     index    = 0;
	bool                         ready    = false; // this frame's (images[index]) was recorded
};
FsrHud g_fsr_hud;

struct FsrHudParameters {
	float region[4];
	float extent[2];
	float threshold, invert;
};

void FsrHudDestroyImages(GraphicContext& graphics) {
	auto& h = g_fsr_hud;
	for (uint32_t i = 0; i <= h.images.size(); ++i) {
		auto& view  = i < h.images.size() ? h.views[i] : h.source_view;
		auto& image = i < h.images.size() ? h.images[i] : h.source;
		if (view != nullptr) graphics.device.destroyImageView(view, nullptr);
		view = nullptr;
		if (image.image != nullptr) graphics.DeleteImage(image);
		image.state = {};
	}
	h.extent = {};
	h.ready  = false;
}

vk::ShaderModule FsrShader(GraphicContext& graphics, const uint32_t* code, size_t words) {
	vk::ShaderModuleCreateInfo create {};
	create.codeSize         = words * sizeof(uint32_t);
	create.pCode            = code;
	vk::ShaderModule module = nullptr;
	RequireVulkanSuccess(graphics.device.createShaderModule(&create, nullptr, &module), "FSR HUD shader");
	return module;
}

// The HUD pass's pipeline for `format` and its images of `extent`.
bool FsrHudSetup(GraphicContext& graphics, vk::Format format, vk::Extent2D extent) {
	auto& h = g_fsr_hud;
	if (h.layout == nullptr) {
		vk::SamplerCreateInfo sampler {};
		sampler.magFilter    = vk::Filter::eLinear;
		sampler.minFilter    = vk::Filter::eLinear;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		RequireVulkanSuccess(graphics.device.createSampler(&sampler, nullptr, &h.sampler), "FSR HUD sampler");
		// The canvas and the backbuffer's copy.
		std::array<vk::DescriptorSetLayoutBinding, 2> bindings {};
		for (uint32_t i = 0; i < bindings.size(); ++i) {
			bindings[i].binding            = i;
			bindings[i].descriptorType     = vk::DescriptorType::eCombinedImageSampler;
			bindings[i].descriptorCount    = 1;
			bindings[i].stageFlags         = vk::ShaderStageFlagBits::eFragment;
			bindings[i].pImmutableSamplers = &h.sampler;
		}
		vk::DescriptorSetLayoutCreateInfo set_layout {};
		set_layout.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		set_layout.bindingCount = static_cast<uint32_t>(bindings.size());
		set_layout.pBindings    = bindings.data();
		RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&set_layout, nullptr, &h.set_layout),
		                     "FSR HUD descriptor layout");
		vk::PushConstantRange range {};
		range.stageFlags = vk::ShaderStageFlagBits::eFragment;
		range.size       = sizeof(FsrHudParameters);
		vk::PipelineLayoutCreateInfo layout {};
		layout.setLayoutCount         = 1;
		layout.pSetLayouts            = &h.set_layout;
		layout.pushConstantRangeCount = 1;
		layout.pPushConstantRanges    = &range;
		RequireVulkanSuccess(graphics.device.createPipelineLayout(&layout, nullptr, &h.layout), "FSR HUD pipeline layout");
		h.vertex   = FsrShader(graphics, GPU_BLIT_FS_TRIANGLE_SPV, std::size(GPU_BLIT_FS_TRIANGLE_SPV));
		h.fragment = FsrShader(graphics, FSR_HUD_MASK_SPV, std::size(FSR_HUD_MASK_SPV));
	}
	if (h.pipeline == nullptr || h.format != format) {
		if (h.pipeline != nullptr) graphics.device.destroyPipeline(h.pipeline, nullptr);
		h.pipeline = nullptr;
		std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
		stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
		stages[0].module = h.vertex;
		stages[0].pName  = "main";
		stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
		stages[1].module = h.fragment;
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
		// The HUD's pixels replaced (the shader discards the others).
		vk::PipelineColorBlendAttachmentState blend {};
		blend.blendEnable    = VK_FALSE;
		blend.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                       vk::ColorComponentFlagBits::eB;
		vk::PipelineColorBlendStateCreateInfo color_blend {};
		color_blend.attachmentCount = 1;
		color_blend.pAttachments    = &blend;
		const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamic {};
		dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
		dynamic.pDynamicStates    = dynamic_states.data();
		vk::PipelineRenderingCreateInfo rendering {};
		rendering.colorAttachmentCount    = 1;
		rendering.pColorAttachmentFormats = &format;
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
		create.layout              = h.layout;
		RequireVulkanSuccess(graphics.device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &h.pipeline),
		                     "FSR HUD pipeline");
		h.format = format;
		FsrHudDestroyImages(graphics); // (of the old format)
	}
	if (h.extent != extent || h.images[0].image == nullptr) {
		FsrHudDestroyImages(graphics);
		for (uint32_t i = 0; i < h.images.size(); ++i) {
			vk::ImageCreateInfo create {};
			create.imageType     = vk::ImageType::e2D;
			create.extent        = vk::Extent3D {extent.width, extent.height, 1};
			create.mipLevels     = 1;
			create.arrayLayers   = 1;
			create.format        = format;
			create.tiling        = vk::ImageTiling::eOptimal;
			create.initialLayout = vk::ImageLayout::eUndefined;
			create.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment |
			               vk::ImageUsageFlagBits::eSampled;
			create.sharingMode = vk::SharingMode::eExclusive;
			create.samples     = vk::SampleCountFlagBits::e1;
			if (!graphics.CreateImage(create, h.images[i])) {
				FsrHudDestroyImages(graphics);
				return false;
			}
			vk::ImageViewCreateInfo view {};
			view.image            = h.images[i].image;
			view.viewType         = vk::ImageViewType::e2D;
			view.format           = format;
			view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
			RequireVulkanSuccess(graphics.device.createImageView(&view, nullptr, &h.views[i]), "FSR HUD view");
		}
		vk::ImageCreateInfo create {};
		create.imageType     = vk::ImageType::e2D;
		create.extent        = vk::Extent3D {extent.width, extent.height, 1};
		create.mipLevels     = 1;
		create.arrayLayers   = 1;
		create.format        = format;
		create.tiling        = vk::ImageTiling::eOptimal;
		create.initialLayout = vk::ImageLayout::eUndefined;
		create.usage         = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
		create.sharingMode   = vk::SharingMode::eExclusive;
		create.samples       = vk::SampleCountFlagBits::e1;
		if (!graphics.CreateImage(create, h.source)) {
			FsrHudDestroyImages(graphics);
			return false;
		}
		vk::ImageViewCreateInfo view {};
		view.image            = h.source.image;
		view.viewType         = vk::ImageViewType::e2D;
		view.format           = format;
		view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		RequireVulkanSuccess(graphics.device.createImageView(&view, nullptr, &h.source_view), "FSR HUD source view");
		h.extent = extent;
	}
	return true;
}

void FsrHudDestroy(GraphicContext& graphics) {
	auto& h = g_fsr_hud;
	FsrHudDestroyImages(graphics);
	if (h.pipeline != nullptr) graphics.device.destroyPipeline(h.pipeline, nullptr);
	h.pipeline = nullptr;
	h.format   = vk::Format::eUndefined;
}

void ImageBarrier(vk::CommandBuffer command, vk::Image image, vk::ImageLayout from, vk::ImageLayout to,
                  vk::AccessFlags2 src_access, vk::AccessFlags2 dst_access) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	barrier.srcAccessMask       = src_access;
	barrier.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	barrier.dstAccessMask       = dst_access;
	barrier.oldLayout           = from;
	barrier.newLayout           = to;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image;
	barrier.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

#endif

#ifdef KYTY_FRAMEGEN_FSR
// The swapchain calls this from its present (allowAsyncWorkloads off: on the game queue): the generated frame.
ffxReturnCode_t FsrGenerate(ffxDispatchDescFrameGeneration* params, void* context) {
	if (std::exchange(g_fsr_state.reset, false)) params->reset = true;
	if (g_fsr_state.hdr) {
		params->backbufferTransferFunction = FFX_API_BACKBUFFER_TRANSFER_FUNCTION_PQ;
		params->minMaxLuminance[0]         = 0.0f;
		params->minMaxLuminance[1]         = 1000.0f;
	}
	return g_fsr_state.api.Dispatch(static_cast<ffxContext*>(context), &params->header);
}
#endif

} // namespace

#ifdef KYTY_FRAMEGEN_NGX
namespace {

const KytyDlssApi* g_dlss        = nullptr; // the bridge, when KYTY_UPSCALER=dlss found it
bool               g_dlss_device = false;   // the device has the extensions NGX needs

void DlssLog(int warning, const char* message) {
	LOGF("DLSS: %s\n", message);
	std::printf("DLSS%s: %s\n", warning != 0 ? " (warning)" : "", message);
	std::fflush(stdout);
}

// Every way DLSS cannot run: the game keeps its own TAA (demonsSoulsSuperResolution.cpp), no frames are generated.
void DlssOff(const char* why) {
	LOGF("DLSS: off (%s)\n", why);
	std::printf("DLSS: off (%s)\n", why);
	std::fflush(stdout);
	g_dlss     = nullptr;
	g_upscale  = false;
	g_generate = 0;
}

std::filesystem::path ExecutableFolder() {
	std::error_code error;
	auto            exe = std::filesystem::read_symlink("/proc/self/exe", error);
	return error ? std::filesystem::current_path(error) : exe.parent_path();
}

// The bridge: in KYTY_DLSS_DIR, else next to the emulator, else in a "dlss" folder next to it; NGX finds
// libnvidia-ngx-dlss.so.* (and libnvidia-ngx-dlssg.so.*) in the same folder.
bool LoadBridge() {
	std::vector<std::filesystem::path> candidates;
	if (const char* env = std::getenv("KYTY_DLSS_DIR"); env != nullptr && *env != 0) {
		candidates.emplace_back(env);
	} else {
		const auto exe = ExecutableFolder();
		candidates.push_back(exe);
		candidates.push_back(exe / "dlss");
	}
	for (const auto& folder: candidates) {
		const auto      path = folder / KYTY_DLSS_BRIDGE_LIBRARY;
		std::error_code error;
		if (!std::filesystem::exists(path, error)) continue;
		void* module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
		if (module == nullptr) {
			std::printf("DLSS: cannot load %s (%s)\n", path.c_str(), dlerror());
			return false;
		}
		const auto         get = reinterpret_cast<KytyDlssGetApiFn>(dlsym(module, KYTY_DLSS_BRIDGE_ENTRY));
		const KytyDlssApi* api = get != nullptr ? get() : nullptr;
		if (api == nullptr || api->abi != KYTY_DLSS_BRIDGE_ABI) {
			std::printf("DLSS: %s is not a bridge of ABI %d (build it again with tools/local/dlss-bridge/build-dlss-linux.sh)\n",
			            path.c_str(), KYTY_DLSS_BRIDGE_ABI);
			dlclose(module);
			return false;
		}
		// NGX's own data (logs, its model cache): writable, next to the emulator's other temporary data.
		auto data = std::filesystem::current_path(error) / "_TempData" / "ngx";
		std::filesystem::create_directories(data, error);
		if (error) data = std::filesystem::temp_directory_path(error);
		const auto folder_text = std::filesystem::absolute(folder, error).string();
		if (api->Configure(folder_text.c_str(), data.c_str(), DlssLog) == 0) {
			std::printf("DLSS: the bridge refused its configuration\n");
			return false;
		}
		g_dlss = api;
		LOGF("DLSS: bridge %s loaded\n", path.c_str());
		std::printf("DLSS: bridge %s loaded\n", path.c_str());
		std::fflush(stdout);
		return true;
	}
	std::printf("DLSS: no %s (tools/local/dlss-bridge/build-dlss-linux.sh puts it next to kyty_emulator; or set "
	            "KYTY_DLSS_DIR)\n",
	            KYTY_DLSS_BRIDGE_LIBRARY);
	std::fflush(stdout);
	return false;
}

bool HasStencil(vk::Format format) {
	return format == vk::Format::eD16UnormS8Uint || format == vk::Format::eD24UnormS8Uint ||
	       format == vk::Format::eD32SfloatS8Uint;
}

KytyDlssImage BridgeImage(const UpscaleImage& image, vk::ImageAspectFlags aspect) {
	KytyDlssImage out {};
	out.image  = static_cast<VkImage>(image.image);
	out.view   = static_cast<VkImageView>(image.view);
	out.range  = {static_cast<VkImageAspectFlags>(aspect), image.base_level, 1, image.base_layer, 1};
	out.format = static_cast<VkFormat>(image.format);
	out.width  = image.extent.width;
	out.height = image.extent.height;
	return out;
}

// The depth as NGX samples it (shader read-only) and back to the layout the renderer left it in.
void DepthLayout(vk::CommandBuffer command, const UpscaleImage& depth, vk::ImageLayout from, vk::ImageLayout to,
                 const vk::detail::DispatchLoaderDynamic& dispatch) {
	vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eDepth;
	if (HasStencil(depth.format)) aspect |= vk::ImageAspectFlagBits::eStencil;
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask       = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask       = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	barrier.oldLayout           = from;
	barrier.newLayout           = to;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = depth.image;
	barrier.subresourceRange    = {aspect, depth.base_level, 1, depth.base_layer, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency, dispatch);
}

KytyDlssImage BridgeImage(const VulkanImage& image, vk::ImageView view) {
	KytyDlssImage out {};
	out.image  = static_cast<VkImage>(image.image);
	out.view   = static_cast<VkImageView>(view);
	out.range  = {static_cast<VkImageAspectFlags>(IsDepthFormat(image.format) ? vk::ImageAspectFlagBits::eDepth
	                                                                          : vk::ImageAspectFlagBits::eColor),
	              0, 1, 0, 1};
	out.format = static_cast<VkFormat>(image.format);
	out.width  = image.extent.width;
	out.height = image.extent.height;
	return out;
}

// Frame generation through NGX (the presenter shows the frames, swapchain.cpp).
bool              g_fg_created = false; // the DLSS-G feature exists for these sizes
bool              g_fg_history = false; // the last frame given to it had inputs: the next one interpolates from it
bool              g_fg_failed  = false;
// The sizes NGX did not make DLSS-G for, and when (out of video memory at a resize: issue #215, NGX 0xbad0000d at a
// 646x270 render size, then no frame generation for the rest of the session): frames of those sizes go without it,
// and it is made again at other sizes or after FgRetry; FgRefusals in a row turn it off.
struct FgRefusal {
	vk::Extent2D                          backbuffer {}, render {};
	vk::Format                            format = vk::Format::eUndefined;
	std::chrono::steady_clock::time_point at {};
	uint32_t                              count = 0;
};
constexpr auto     FgRetry    = std::chrono::seconds(30);
constexpr uint32_t FgRefusals = 8;
FgRefusal          g_fg_refused;
// Since when the frames' render size has been half the feature's or less, the backbuffer the same (zero: not now).
std::chrono::steady_clock::time_point g_fg_small_since {};
vk::Extent2D      g_fg_backbuffer {}, g_fg_render {};
vk::Format        g_fg_format = vk::Format::eUndefined;
std::atomic<bool> g_fg_discontinuity {false}; // a frame with inputs left unshown (NoteDroppedFrame)

// Matrices as Streamline and NGX take them: row-major, row vectors (p' = p * M), 16 floats.
using RowMatrix = std::array<float, 16>;

RowMatrix Identity() {
	RowMatrix m {};
	for (uint32_t i = 0; i < 4; ++i) m[i * 4 + i] = 1.0f;
	return m;
}

RowMatrix Transposed(const std::array<float, 16>& m) { // the game's column-vector matrix
	RowMatrix r {};
	for (uint32_t i = 0; i < 4; ++i)
		for (uint32_t j = 0; j < 4; ++j) r[i * 4 + j] = m[j * 4 + i];
	return r;
}

RowMatrix Multiply(const RowMatrix& a, const RowMatrix& b) {
	RowMatrix r {};
	for (uint32_t i = 0; i < 4; ++i)
		for (uint32_t j = 0; j < 4; ++j)
			for (uint32_t k = 0; k < 4; ++k) r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
	return r;
}

bool Inverted(const RowMatrix& m, RowMatrix& out) {
	double a[4][8] {};
	for (uint32_t i = 0; i < 4; ++i) {
		for (uint32_t j = 0; j < 4; ++j) a[i][j] = m[i * 4 + j];
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
	for (uint32_t i = 0; i < 4; ++i)
		for (uint32_t j = 0; j < 4; ++j) out[i * 4 + j] = static_cast<float>(a[i][4 + j]);
	return true;
}

// The game's world->clip matrix split into camera axes, position and projection terms, as the Streamline way does
// (its Decompose): P * V with a perspective P whose w row is (0, 0, 1, 0) and whose z row is constant (reversed-Z,
// infinite far).
struct CameraAxes {
	std::array<float, 3> right {}, up {}, forward {}, position {};
	float                sx = 1.0f, sy = 1.0f, cx = 0.0f, cy = 0.0f, z_near = 0.1f;
	bool                 valid = false;
};

CameraAxes Decompose(const std::array<float, 16>& m) {
	CameraAxes c;
	const auto row = [&](uint32_t i) { return std::array<double, 4> {m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]}; };
	const auto r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
	const double flen = std::sqrt(r3[0] * r3[0] + r3[1] * r3[1] + r3[2] * r3[2]);
	if (flen < 1e-6) return c;
	const double f[3] {r3[0] / flen, r3[1] / flen, r3[2] / flen};
	const double tf    = r3[3] / flen;
	const auto   split = [&](const std::array<double, 4>& r, double& s, double& off, double axis[3], double& t) {
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
	for (int i = 0; i < 3; ++i) {
		c.right[i]   = static_cast<float>(x[i]);
		c.up[i]      = static_cast<float>(up[i]);
		c.forward[i] = static_cast<float>(f[i]);
		// View translation (tx, ty, tf) = -axes . position.
		c.position[i] = static_cast<float>(-(tx * x[i] + ty * up[i] + tf * f[i]));
	}
	c.sx     = static_cast<float>(sx);
	c.sy     = static_cast<float>(sy);
	c.cx     = static_cast<float>(cx);
	c.cy     = static_cast<float>(cy);
	c.z_near = static_cast<float>(r2[3] / flen);
	c.valid  = true;
	return c;
}

void Copy16(float (&out)[16], const RowMatrix& m) {
	for (uint32_t i = 0; i < 16; ++i) out[i] = m[i];
}

} // namespace
#endif

bool Enabled() {
	return g_ready;
}

bool Requested() {
	return g_generate > 0;
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
	if (g_generate == 0 || g_fsr) {
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

bool WantsCamera() {
	return g_ready && !g_camera.valid;
}

bool InputsGathered() {
	return g_sources.valid && g_camera.valid;
}

void BypassCommandHooks(vk::Device device) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// KYTY_FRAMEGEN_HOOKS=1 keeps Streamline's wrappers (for comparison).
	// (Streamline not loaded: neither DLSS-G nor DLSS super resolution.)
	if ((!g_upscale && (g_generate == 0 || g_fsr)) || std::getenv("KYTY_FRAMEGEN_HOOKS") != nullptr) {
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
	direct(dispatcher.vkBeginCommandBuffer, "vkBeginCommandBuffer");
	direct(dispatcher.vkCmdBindPipeline, "vkCmdBindPipeline");
	direct(dispatcher.vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets");
	direct(dispatcher.vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
	std::printf("Frame generation: %u command-buffer calls bypass Streamline\n", replaced);
#else
	(void)device;
#endif
}

void Initialize() {
	const char* value = std::getenv("KYTY_FRAMEGEN");
	g_generate        = value != nullptr ? std::clamp(std::atoi(value), 0, 5) : 0;
	const char* upscaler = std::getenv("KYTY_UPSCALER");
	g_upscale            = upscaler != nullptr && std::string_view(upscaler) == "dlss";
	// Linux: KYTY_FRAMEGEN_BACKEND=fsr is FSR 3 frame interpolation built in (fsr-native.h, KYTY_FRAMEGEN implied), shown by
	// the presenter as NGX's DLSS-G frames are; no NVIDIA runtime. DLSS super resolution may still load the bridge.
	if (FsrNative::InterpolationRequested()) {
		g_generate   = 1;
		g_fsr_native = true;
		std::printf("Frame generation: AMD FSR 3 (2x, built in)\n");
		std::fflush(stdout);
	}
	if (g_generate == 0 && !g_upscale) {
		return;
	}
	if (g_fsr_native && !g_upscale) {
		return;
	}
	if (const char* backend = std::getenv("KYTY_FRAMEGEN_BACKEND");
	    !g_fsr_native && backend != nullptr && std::string(backend) == "fsr") {
		[] {
#ifdef KYTY_FRAMEGEN_FSR
			// AMD's runtime next to the emulator (FsrLoadRuntime); FSR makes one frame per frame.
			if (!FsrLoadRuntime()) {
				std::printf("Frame generation: no AMD runtime next to the emulator (amd_fidelityfx_loader_vk.dll with its "
				            "effect DLLs, or amd_fidelityfx_vk.dll), FSR off\n");
				std::fflush(stdout);
				g_generate = 0;
				return;
			}
			g_generate = 1;
			g_fsr      = true;
			std::printf("Frame generation: AMD FSR (2x)\n");
			std::fflush(stdout);
#else
			LOGF("frame generation: this build has no FSR support\n");
			g_generate = 0;
#endif
		}();
		// DLSS super resolution beside it (issue #244: an RTX 30 has no DLSS-G, so FSR's frame generation, which took FSR 3
		// upscaling in DLSS's place): Streamline below with its DLSS plugin only. AMD's runtime makes its swapchain and
		// presents through the Vulkan loader itself (vulkan-1.dll's imports), past Streamline. KYTY_FSR_FG_DLSS=0: FSR 3
		// upscaling with it, as before.
		static const bool with_dlss = [] {
			const char* value = std::getenv("KYTY_FSR_FG_DLSS");
			return value == nullptr || value[0] != '0';
		}();
		if (!with_dlss) g_upscale = false;
		if (!g_upscale) return;
	}
	// (Every way out without Streamline turns both off, AMD's frame generation aside.)
	struct Off {
		bool done = false;
		~Off() {
			if (!done) {
				if (!g_fsr_native && !g_fsr) g_generate = 0;
				g_upscale = false;
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
		return;
	}
	const auto interposer = dir / "sl.interposer.dll";
	// Only NVIDIA's own (Streamline then checks its plugins' signatures itself).
	if (!LocalPlatform::IsNvidiaSigned(interposer.wstring())) {
		LOGF("frame generation: %s is not signed by NVIDIA: not loaded\n", interposer.string().c_str());
		std::printf("Frame generation: %s is not signed by NVIDIA, not loaded\n", interposer.string().c_str());
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
	    !load(g_api.set_tag_for_frame, "slSetTagForFrame") ||
	    !load(g_api.evaluate_feature, "slEvaluateFeature")) {
		LOGF("frame generation: sl.interposer.dll lacks an entry point\n");
		return;
	}
	static const std::wstring plugin_dir = dir.wstring();
	static const std::wstring log_dir    = (dir.parent_path() / "logs").wstring();
	static const wchar_t*     plugin_paths[] {plugin_dir.c_str()};
	// KYTY_FRAMEGEN_FEATURES=dlssg,reflex,pcl: the plugins to load (a diagnostic subset).
	static std::vector<sl::Feature> features;
	if (g_generate > 0 && !g_fsr) features = {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
	if (const char* list = std::getenv("KYTY_FRAMEGEN_FEATURES"); list != nullptr && g_generate > 0 && !g_fsr) {
		const std::string text = list;
		features.clear();
		if (text.find("dlssg") != std::string::npos) features.push_back(sl::kFeatureDLSS_G);
		if (text.find("reflex") != std::string::npos) features.push_back(sl::kFeatureReflex);
		if (text.find("pcl") != std::string::npos) features.push_back(sl::kFeaturePCL);
	}
	if (g_upscale) features.push_back(sl::kFeatureDLSS);
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
		return;
	}
	// SDL loads the interposer as the Vulkan loader: every instance/device/swapchain call then
	// goes through Streamline, which adds what DLSS-G needs.
	SetEnvironmentVariableW(L"SDL_VULKAN_LIBRARY", interposer.c_str());
	_putenv_s("SDL_VULKAN_LIBRARY", interposer.string().c_str());
	LOGF("frame generation: Streamline loaded from %s, %d generated frame(s), DLSS super resolution %s\n",
	     dir.string().c_str(), g_generate, g_upscale ? "asked for" : "off");
	off.done = true;
#elif defined(KYTY_FRAMEGEN_NGX)
	// (DLSS super resolution, frame generation, or both: the bridge either way. NGX makes the frames, the presenter
	// shows them: swapchain.cpp, PresenterGenerates.)
	if (!LoadBridge()) return;
	off.done = true;
#else
	LOGF("frame generation: this build has no Streamline support\n");
	std::printf("DLSS: this build has no Streamline support\n");
#endif
}

void AddInstanceExtensions(const std::vector<vk::ExtensionProperties>& available, std::vector<const char*>& extensions) {
#ifdef KYTY_FRAMEGEN_NGX
	if (g_dlss == nullptr) return;
	uint32_t                     count  = 0;
	const VkExtensionProperties* wanted = nullptr;
	if (g_dlss->InstanceExtensions(&count, &wanted) == 0) return DlssOff("NGX named no instance extensions");
	for (uint32_t i = 0; i < count; ++i) {
		const char* name = wanted[i].extensionName;
		if (std::none_of(available.begin(), available.end(),
		                 [&](const vk::ExtensionProperties& item) { return std::strcmp(item.extensionName.data(), name) == 0; })) {
			std::printf("DLSS: the Vulkan instance has no %s\n", name);
			return DlssOff("an instance extension NGX needs is missing");
		}
	}
	for (uint32_t i = 0; i < count; ++i) {
		const char* name = wanted[i].extensionName;
		if (std::none_of(extensions.begin(), extensions.end(), [&](const char* have) { return std::strcmp(have, name) == 0; }))
			extensions.push_back(name);
	}
	LOGF("DLSS: %u instance extension(s) for NGX\n", count);
#else
	(void)available;
	(void)extensions;
#endif
}

void AddDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical_device,
                         const std::vector<vk::ExtensionProperties>& available, std::vector<const char*>& extensions) {
#ifdef KYTY_FRAMEGEN_NGX
	if (g_dlss == nullptr) return;
	uint32_t                     count  = 0;
	const VkExtensionProperties* wanted = nullptr;
	if (g_dlss->DeviceExtensions(static_cast<VkInstance>(instance), static_cast<VkPhysicalDevice>(physical_device), &count,
	                             &wanted) == 0)
		return DlssOff("NGX does not offer DLSS with this GPU and driver (an NVIDIA RTX GPU and a recent driver)");
	for (uint32_t i = 0; i < count; ++i) {
		const char* name = wanted[i].extensionName;
		if (std::none_of(available.begin(), available.end(),
		                 [&](const vk::ExtensionProperties& item) { return std::strcmp(item.extensionName.data(), name) == 0; })) {
			std::printf("DLSS: the GPU has no %s\n", name);
			return DlssOff("a device extension NGX needs is missing");
		}
	}
	for (uint32_t i = 0; i < count; ++i) {
		const char* name = wanted[i].extensionName;
		if (std::none_of(extensions.begin(), extensions.end(), [&](const char* have) { return std::strcmp(have, name) == 0; }))
			extensions.push_back(name);
	}
	g_dlss_device = true;
	LOGF("DLSS: %u device extension(s) for NGX\n", count);
#else
	(void)instance;
	(void)physical_device;
	(void)available;
	(void)extensions;
#endif
}

void OnDevice(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device) {
	(void)instance;
	(void)device;
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (g_upscale) {
		sl::AdapterInfo adapter {};
		adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
		bool loaded              = false;
		if (g_api.is_feature_supported(sl::kFeatureDLSS, adapter) == sl::Result::eOk &&
		    g_api.is_feature_loaded(sl::kFeatureDLSS, loaded) == sl::Result::eOk && loaded &&
		    FeatureFunction(sl::kFeatureDLSS, "slDLSSSetOptions", g_api.dlss_set_options) &&
		    FeatureFunction(sl::kFeatureDLSS, "slDLSSGetOptimalSettings", g_api.dlss_get_optimal_settings)) {
			g_upscale_ready = true;
			std::printf("DLSS super resolution: ready\n");
		} else {
			std::printf("DLSS super resolution: not available on this device (the game's image is upscaled without it)\n");
		}
		std::fflush(stdout);
	}
#elif defined(KYTY_FRAMEGEN_NGX)
	if ((g_upscale || g_generate > 0) && g_dlss != nullptr && g_dlss_device) {
		auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
		if (g_dlss->Initialize(static_cast<VkInstance>(instance), static_cast<VkPhysicalDevice>(physical_device),
		                       static_cast<VkDevice>(device), dispatch.vkGetInstanceProcAddr, dispatch.vkGetDeviceProcAddr) != 0) {
			if (g_upscale) {
				g_upscale_ready = true;
				std::printf("DLSS super resolution: ready (NVIDIA NGX)\n");
			}
			if (g_generate > 0 && !g_fsr_native) {
				const int most = g_dlss->FrameGenMaxCount();
				if (most < 1) {
					LOGF("frame generation: NGX offers no DLSS-G on this GPU or driver\n");
					std::printf("Frame generation: not available on this GPU or driver (NVIDIA RTX 40 or newer)\n");
					g_generate = 0;
				} else {
					if (g_generate > most) {
						std::printf("Frame generation: %d generated frame(s) asked for, the GPU makes up to %d\n", g_generate, most);
						g_generate = most;
					}
					g_ready = true;
					InitializeHudless();
					LOGF("frame generation: DLSS-G ready through NGX, %d generated frame(s) per frame\n", g_generate);
					std::printf("Frame generation: DLSS-G ready (NVIDIA NGX, %d generated frame(s) per frame, %dx)\n",
					            g_generate, g_generate + 1);
				}
			}
		} else {
			std::printf("DLSS: NGX did not start on this device (the game keeps its own TAA, no frame generation)\n");
			g_generate = 0;
		}
		std::fflush(stdout);
	}
	// (FSR 3 built in: its contexts are made by the present thread with the first frame it generates.)
	if (g_fsr_native) g_ready = true;
	if (g_generate > 0 && !g_ready) g_generate = 0; // (no NGX: Requested() and the rest see it off)
#endif
	if (g_generate == 0 || g_fsr || g_fsr_native) {
		return;
	}
#ifdef KYTY_FRAMEGEN_STREAMLINE
	sl::AdapterInfo adapter {};
	adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
	bool loaded              = false;
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
	// (Required with DLSS-G, whatever the Reflex mode: Streamline's DLSS-G and Reflex guides.)
	(void)FeatureFunction(sl::kFeatureReflex, "slReflexSleep", g_api.reflex_sleep);
	sl::ReflexOptions reflex {};
	reflex.mode = sl::ReflexMode::eLowLatency;
	if (const auto result = g_api.reflex_set_options(reflex); result != sl::Result::eOk) {
		LOGF("frame generation: slReflexSetOptions failed (%d)\n", static_cast<int>(result));
	}
	g_ready = true;
	InitializeHudless();
	LOGF("frame generation: DLSS-G ready\n");
	std::printf("Frame generation: DLSS-G ready (%d generated frame(s) per frame)\n", g_generate);
	std::fflush(stdout);
#endif
}

void BeforePresent(const FrameInputs* inputs, vk::Extent2D backbuffer, vk::Rect2D region,
                   vk::Format format, uint32_t image_count) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!g_ready || g_fsr) {
		return;
	}
	const sl::ViewportHandle viewport {0};
	const bool               usable = inputs != nullptr && inputs->valid;
	g_run_with    = usable ? g_run_with + 1 : 0;
	g_run_without = usable ? 0 : g_run_without + 1;
	const bool mode_on = g_mode_on ? g_run_without < TurnOffAfter : g_run_with >= TurnOnAfter;
	// DLSS-G options: on while frames carry inputs; sizes are hints for its allocations.
	if (!g_options_set || g_options_format != format || mode_on != g_mode_on ||
	    (usable && (g_options_backbuffer != backbuffer || g_options_render != inputs->render_extent))) {
		sl::DLSSGOptions options {};
		options.mode                = mode_on ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
		options.flags               = sl::DLSSGFlags::eRetainResourcesWhenOff;
		options.numFramesToGenerate = static_cast<uint32_t>(g_generate);
		options.numBackBuffers      = image_count;
		options.colorWidth          = backbuffer.width;
		options.colorHeight         = backbuffer.height;
		options.colorBufferFormat   = static_cast<uint32_t>(format);
		if (HudlessOn()) {
			options.hudLessBufferFormat = static_cast<uint32_t>(format); // (ScaleHudless: the backbuffer's)
		}
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
		if (mode_on != g_mode_on) {
			if (mode_on) std::printf("Frame generation: DLSS-G on\n");
			else std::printf("Frame generation: DLSS-G off: %u frames without inputs (%s)\n", TurnOffAfter, MissingInputs());
			std::fflush(stdout);
			g_mode_on = mode_on;
		}
	}

	++g_frame_index;
	if (g_api.get_new_frame_token(g_token, &g_frame_index) != sl::Result::eOk || g_token == nullptr) {
		g_token = nullptr;
		return;
	}
	auto& token = *g_token;
	// The frame's Reflex sleep, then its markers (all at its present: the game's frame was made before).
	if (g_api.reflex_sleep != nullptr && kyty_local_framegen_sleep.load(std::memory_order_relaxed) != 0)
		g_api.reflex_sleep(token);
	g_api.pcl_set_marker(sl::PCLMarker::eSimulationStart, token);
	g_api.pcl_set_marker(sl::PCLMarker::eSimulationEnd, token);
	g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitStart, token);

	sl::Constants constants {};
	const auto    camera = usable ? Decompose(inputs->camera.view_proj) : CameraFrame {};
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
	const bool discontinuity      = g_discontinuity.exchange(false, std::memory_order_relaxed) || (usable && inputs->cut);
	constants.reset               = usable && !discontinuity ? sl::Boolean::eFalse : sl::Boolean::eTrue;
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
		const auto current  = Transposed(inputs->camera.view_proj);
		const auto previous = Transposed(inputs->camera.prev_view_proj);
		Mat4       clip_to_world;
		if (Inverted(current, clip_to_world)) {
			constants.clipToPrevClip = Multiply(clip_to_world, previous);
			Inverted(constants.clipToPrevClip, constants.prevClipToClip);
		}
		const float width  = static_cast<float>(inputs->render_extent.width);
		const float height = static_cast<float>(inputs->render_extent.height);
		constants.jitterOffset = sl::float2(inputs->camera.jitter_ndc[0] * width * 0.5f,
		                                    -inputs->camera.jitter_ndc[1] * height * 0.5f);
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
		sl::ResourceTag tags[] {
		    sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &full),
		    sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent,
		                    &full),
		    sl::ResourceTag(nullptr, sl::kBufferTypeBackbuffer, sl::ResourceLifecycle::eValidUntilPresent,
		                    &game),
		};
		if (const auto result = g_api.set_tag_for_frame(token, viewport, tags, partial ? 3u : 2u, nullptr);
		    result != sl::Result::eOk && g_presents < 8) {
			LOGF("frame generation: slSetTagForFrame failed (%d)\n", static_cast<int>(result));
		}
		if (HudlessOn()) {
			TagHudless(token, *inputs, game);
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
	if (!g_ready || g_fsr || g_token == nullptr) {
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

void SwapchainChanging() {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// Off before the swapchain is made again (DLSS-G guide: before a resize or a full screen change).
	if (!g_ready || g_fsr || !g_mode_on) return;
	sl::DLSSGOptions options {};
	options.mode                = sl::DLSSGMode::eOff;
	options.flags               = sl::DLSSGFlags::eRetainResourcesWhenOff;
	options.numFramesToGenerate = static_cast<uint32_t>(g_generate);
	if (const auto result = g_api.dlssg_set_options(sl::ViewportHandle {0}, options); result != sl::Result::eOk)
		LOGF("frame generation: slDLSSGSetOptions (off) failed (%d)\n", static_cast<int>(result));
	g_mode_on     = false;
	g_run_with    = 0;
	g_options_set = false; // (the next present sends its sizes again)
#elif defined(KYTY_FRAMEGEN_NGX)
	g_fg_history = false; // (the next frame starts DLSS-G again)
#endif
}

void NoteDroppedFrame() {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	g_discontinuity.store(true, std::memory_order_relaxed);
#elif defined(KYTY_FRAMEGEN_NGX)
	g_fg_discontinuity.store(true, std::memory_order_relaxed);
#endif
#ifdef KYTY_FRAMEGEN_FSR
	g_fsr_discontinuity.store(true, std::memory_order_relaxed);
#endif
	static std::atomic<uint32_t> dropped {0};
	if (const auto count = dropped.fetch_add(1, std::memory_order_relaxed) + 1; count <= 8 || count % 600 == 0)
		LOGF("frame generation: %u frame(s) with inputs left unshown (the next resets DLSS-G's history)\n", count);
}

uint32_t PresentedFrames() {
#ifdef KYTY_FRAMEGEN_FSR
	if (g_fsr) return g_fsr_state.mode_on ? 2u : 1u;
#endif
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

static bool g_cut = false; // Kyty.Gpu: the current guest frame's upscale started a new history (NoteCut)

void NoteCut() {
	g_cut = true;
}

// The guest's indexed draw packets so far (LiveCounters::AddPm4 counts every PM4 packet the render thread runs).
static uint64_t IndexedDrawPackets() {
	using namespace Libs::Graphics::Pm4;
	uint64_t draws = 0;
	for (const uint32_t opcode: {IT_DRAW_INDEX_2, IT_DRAW_INDEX_OFFSET_2, IT_DRAW_INDEX_INDIRECT, IT_DRAW_INDEX_INDIRECT_MULTI})
		draws += LiveCounters::g_pm4[opcode].load(std::memory_order_relaxed);
	return draws;
}

bool TakeScene() {
	// The game's 3D view: its TAA pass ran over a world drawn since the last flip (1-1: ~5,400 indexed draw packets a
	// frame). The start-up logos run the TAA and the lighting over 36 draws without indices (the PlayStation Studios one is
	// the game's own render): at -Fps120 they flipped 120 times a second, at twice their speed (issue #231). One indexed
	// draw is enough: a view filled by a floor or a wall close up draws few (issue #241).
	const uint64_t drawn = IndexedDrawPackets();
	const bool     world = drawn != g_drawn_at_flip.exchange(drawn, std::memory_order_relaxed);
	const bool     scene = g_scene.exchange(false, std::memory_order_relaxed) && world;
	g_last_scene.store(scene, std::memory_order_relaxed);
	g_flips_taken.fetch_add(1, std::memory_order_relaxed);
	return scene;
}

uint64_t GuestFlips() {
	return g_flips_taken.load(std::memory_order_relaxed);
}

void NoteMenu() {
	g_menu.store(true, std::memory_order_relaxed);
}

void NoteCentred() {
	g_centred.store(true, std::memory_order_relaxed);
}

bool TakeMenu() {
	// (A closed menu fades out over a few more frames without the draws NoteMenu takes it by: 20 flips more in the
	// middle, a third of a second, and its fade ends at 16:9 instead of widened across the screen.)
	constexpr int HoldFlips = 20;
	const bool    menu      = g_menu.exchange(false, std::memory_order_relaxed);
	g_menu_hold.store(menu ? HoldFlips : std::max(0, g_menu_hold.load(std::memory_order_relaxed) - 1),
	                  std::memory_order_relaxed);
	return g_centred.exchange(false, std::memory_order_relaxed);
}

bool MenuNoted() {
	return g_menu.load(std::memory_order_relaxed) || g_menu_hold.load(std::memory_order_relaxed) > 0;
}

bool SceneNoted() {
	// (The current frame as TakeScene will tell it: the TAA pass over a world drawn with indices since the last flip. The
	// start-up logos' TAA over draws without indices had their pictures widened across the wide output, which the
	// presenter, not taking them for the 3D view, cut to its 16:9 part: the logos at its right edge, issue #250.)
	return (g_scene.load(std::memory_order_relaxed) &&
	        IndexedDrawPackets() != g_drawn_at_flip.load(std::memory_order_relaxed)) ||
	       g_last_scene.load(std::memory_order_relaxed);
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

void PrepareInputs(GraphicContext& graphics, TextureCache& cache, CommandBuffer& command,
                   FrameInputs& inputs) {
	inputs.valid       = false;
	inputs.cut         = std::exchange(g_cut, false);
	const auto sources = g_sources;
	auto       camera  = g_camera;
	g_sources.valid    = false;
	g_camera.valid     = false;
	++g_guest_frame;
	// The camera comes from the G-buffer's vertex shaders (OnDraw). A view without them (the Nexus floor looked down at,
	// a wall close up: issue #241) takes the last one seen since the last cut, held still (its previous matrix the same:
	// the game's motion vectors carry the motion; only its projection is used then) and without jitter. Only a frame of
	// the game's 3D view (TakeScene, just before: the loading screens run the TAA too, over no world).
	// KYTY_FG_HOLD_CAMERA=0: no inputs for such a frame, as before.
	static const bool hold = [] {
		const char* value = std::getenv("KYTY_FG_HOLD_CAMERA");
		return value == nullptr || value[0] != '0';
	}();
	// KYTY_TEST_DROP_CAMERA=1 (a test of the held camera): after 600 frames with a camera, none has its own.
	static const bool drop = [] {
		const char* value = std::getenv("KYTY_TEST_DROP_CAMERA");
		return value != nullptr && value[0] == '1';
	}();
	static uint32_t cameras = 0;
	if (drop && camera.valid && ++cameras > 600) camera.valid = false;
	static Camera held;
	if (camera.valid) {
		held = camera;
	} else if (inputs.cut) {
		held.valid = false;
	} else if (hold && held.valid && sources.valid && g_last_scene.load(std::memory_order_relaxed)) {
		camera                = held;
		camera.prev_view_proj = camera.view_proj;
		camera.jitter_ndc     = {0.0f, 0.0f};
	}
	if (!sources.valid || !camera.valid) {
		g_inputs_missing.store(sources.valid ? 2u : 1u, std::memory_order_relaxed);
		return;
	}
	g_inputs_missing.store(3u, std::memory_order_relaxed); // (until the images below are found as the TAA left them)
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
	inputs.render_extent = {motion.backing.extent.width, motion.backing.extent.height};
	inputs.camera        = camera;
	inputs.guest_frame   = g_guest_frame;
	inputs.valid         = true;
	g_inputs_missing.store(0u, std::memory_order_relaxed);
#ifdef KYTY_FRAMEGEN_FSR_HUD
	// FSR: the interface canvas as it is at the flip (RecordHudless reads its alpha).
	inputs.ui_valid = false;
	if (const uint64_t canvas = g_ui_canvas.load(std::memory_order_relaxed); (g_fsr || g_fsr_native) && canvas != 0) {
		// (Another image can start at the canvas's address later on - a pass reusing its memory -: the canvas is the
		// one of its size used last, not the first registered there. Each change of state reported, up to 32.)
		static int        state   = 0; // 1: copied, 2: unusable, 3: not in the cache
		static uint32_t   reports = 0;
		static vk::Format format  = vk::Format::eUndefined; // the canvas's, from its first copy
		const vk::Extent2D size  = UiCanvasExtent();
		uint32_t           count = 0;
		auto               id    = cache.FindImageAt(canvas, size.width, size.height, format, &count);
		// (A canvas made again in another format - a game setting changed - is still the canvas: the one of its size used
		// last, its format taken from then on.)
		if (!id && format != vk::Format::eUndefined) id = cache.FindImageAt(canvas, size.width, size.height, vk::Format::eUndefined, &count);
		int                now   = 3;
		if (id) {
			auto& ui = cache.GetImage(id);
			now      = 2;
			if (!IsDepthFormat(ui.backing.format)) {
				// (At most 1080 high, the 4K canvas at half its size, a wider one at the same scale: its alpha is a mask,
				// read a few pixels wide; a quarter of the video memory.)
				const uint32_t high   = std::min(size.height, 1080u);
				bool           failed = false;
				Configure(graphics, inputs.ui, inputs.ui_view, ui.backing, nullptr,
				          vk::Extent2D {std::max(1u, size.width * high / std::max(1u, size.height)), high}, &failed);
				if (!failed) {
					BlitInto(vk_command, ui, inputs.ui);
					inputs.ui_valid = true;
					now             = 1;
					format          = ui.backing.format;
				}
			}
			if (now != state && reports < 32) {
				++reports;
				std::printf("Frame generation: FSR HUD canvas %s (%ux%u, format %d; %u image(s) at 0x%" PRIx64 ")\n",
				            now == 1 ? "copied" : "image unusable", ui.backing.extent.width, ui.backing.extent.height,
				            static_cast<int>(ui.backing.format), count, canvas);
				std::fflush(stdout);
			}
		} else if (now != state && reports < 32) {
			++reports;
			std::printf("Frame generation: FSR HUD canvas at 0x%" PRIx64 " not in the texture cache as %ux%u (%u image(s) there)\n",
			            canvas, size.width, size.height, count);
			std::fflush(stdout);
		}
		state = now;
	} else if ((g_fsr || g_fsr_native) && canvas == 0) {
		static uint32_t scenes = 0;
		if (++scenes == 300) {
			std::printf("Frame generation: no FSR HUD canvas seen in 300 frames with inputs\n");
			std::fflush(stdout);
		}
	}
#endif
}

bool UpscaleWithAmdRuntime(GraphicContext& graphics, vk::CommandBuffer command, const UpscaleInputs& inputs) {
#ifdef KYTY_FRAMEGEN_FSR
	static const bool allowed = [] {
		const char* value = std::getenv("KYTY_FSR_RUNTIME");
		return value == nullptr || value[0] != '0';
	}();
	return allowed && FsrLoadRuntime() && g_fsr_state.sdk2 && TheRuntimeUpscaler().Record(graphics, command, inputs);
#else
	(void)graphics, (void)command, (void)inputs;
	return false;
#endif
}

bool UpscalerReady() {
	return g_upscale_ready;
}

bool PresentEveryFrame() {
	static const char* forced = std::getenv("KYTY_PRESENT_EVERY_FRAME");
	if (forced != nullptr && *forced != '\0') return std::strcmp(forced, "0") != 0;
#ifdef KYTY_FRAMEGEN_STREAMLINE
	return g_upscale_ready && !g_ready;
#else
	return false;
#endif
}

bool Upscale(vk::CommandBuffer command, const UpscaleInputs& inputs, const vk::detail::DispatchLoaderDynamic& dispatch) {
	(void)dispatch;
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
		// limits): PS5 Pro's 1512 of 2160 and 1440 are quality, 1080 performance, the output's own DLAA.
		const double share = static_cast<double>(inputs.color.extent.height) / inputs.output.extent.height;
		sl::DLSSOptions options {};
		options.mode = share >= 0.95   ? sl::DLSSMode::eDLAA
		               : share >= 0.62 ? sl::DLSSMode::eMaxQuality
		               : share >= 0.55 ? sl::DLSSMode::eBalanced
		               : share >= 0.45 ? sl::DLSSMode::eMaxPerformance
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
	auto             color    = resource(inputs.color);
	auto             depth    = resource(inputs.depth);
	auto             motion   = resource(inputs.motion);
	auto             output   = resource(inputs.output);
	auto             reactive = resource(inputs.reactive);
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
	// (The game's reactive mask as DLSS's bias towards the current color: 1 rejects the history; untagged, none.)
	const sl::ResourceTag tags[] {
	    sl::ResourceTag(&color, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &render),
	    sl::ResourceTag(&output, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &full),
	    sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilEvaluate, &render),
	    sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilEvaluate, &render),
	    sl::ResourceTag(inputs.reactive.image ? &reactive : nullptr, sl::kBufferTypeBiasCurrentColorHint,
	                    sl::ResourceLifecycle::eValidUntilEvaluate, &render),
	};
	auto* command_buffer = reinterpret_cast<sl::CommandBuffer*>(static_cast<VkCommandBuffer>(command));
	if (const auto result = g_api.set_tag_for_frame(*token, viewport, tags, static_cast<uint32_t>(std::size(tags)), command_buffer);
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
	if (const auto result = g_api.evaluate_feature(sl::kFeatureDLSS, *token, evaluate_inputs, 1, command_buffer);
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
#elif defined(KYTY_FRAMEGEN_NGX)
	if (!g_upscale_ready || g_dlss == nullptr) return false;
	// (The renderer thread, or the recording worker replaying in order: one at a time.)
	static vk::Extent2D created_render {}, created_output {};
	static bool         created = false, failed = false;
	static uint32_t     reports = 0;
	if (failed) return false; // (the stretch, every frame, without asking NGX again)
	if (!created || created_render != inputs.color.extent || created_output != inputs.output.extent) {
		// The mode by the input's share of the output's height, as Windows' (Streamline) way: PS5 Pro's 1512 of 2160 and
		// 1440 are quality, 1080 performance, the output's own DLAA.
		const double    share = static_cast<double>(inputs.color.extent.height) / inputs.output.extent.height;
		KytyDlssFeature feature {};
		feature.input_width    = inputs.color.extent.width;
		feature.input_height   = inputs.color.extent.height;
		feature.output_width   = inputs.output.extent.width;
		feature.output_height  = inputs.output.extent.height;
		feature.quality        = share >= 0.95 ? 0 : share >= 0.62 ? 1 : share >= 0.55 ? 2 : share >= 0.45 ? 3 : 4;
		feature.depth_inverted = 1;
		feature.hdr            = 1; // (the game's linear HDR scene, before its tone mapping; automatic exposure)
		// KYTY_DLSS_PRESET=<letter> (a test): that DLSS model for every mode instead of the mode's default.
		if (const char* preset = std::getenv("KYTY_DLSS_PRESET"); preset != nullptr && *preset >= 'A' && *preset <= 'O')
			feature.preset = static_cast<uint32_t>(*preset - 'A' + 1);
		if (g_dlss->CreateFeature(static_cast<VkCommandBuffer>(command), &feature) == 0) {
			std::printf("DLSS super resolution: NGX did not make the feature for %ux%u -> %ux%u (the stretch instead)\n",
			            feature.input_width, feature.input_height, feature.output_width, feature.output_height);
			std::fflush(stdout);
			failed = true;
			return false;
		}
		static constexpr const char* Modes[] {"DLAA", "Quality", "Balanced", "Performance", "Ultra Performance"};
		std::printf("DLSS super resolution: %ux%u -> %ux%u, %s (NVIDIA NGX)\n", feature.input_width, feature.input_height,
		            feature.output_width, feature.output_height, Modes[feature.quality]);
		std::fflush(stdout);
		created        = true;
		created_render = inputs.color.extent;
		created_output = inputs.output.extent;
	}
	static auto last = std::chrono::steady_clock::now();
	const auto  now  = std::chrono::steady_clock::now();
	const float frame_ms =
	    std::clamp(std::chrono::duration<float, std::milli>(now - last).count(), 0.0f, 100.0f);
	last = now;

	KytyDlssEvaluate evaluate {};
	evaluate.color  = BridgeImage(inputs.color, vk::ImageAspectFlagBits::eColor);
	evaluate.depth  = BridgeImage(inputs.depth, IsDepthFormat(inputs.depth.format) ? vk::ImageAspectFlagBits::eDepth
	                                                                               : vk::ImageAspectFlagBits::eColor);
	evaluate.motion = BridgeImage(inputs.motion, vk::ImageAspectFlagBits::eColor);
	evaluate.output = BridgeImage(inputs.output, vk::ImageAspectFlagBits::eColor);
	evaluate.jitter_x   = inputs.jitter_x; // (render pixels, as NGX takes them)
	evaluate.jitter_y   = inputs.jitter_y;
	evaluate.mv_scale_x = inputs.motion_scale_x; // (the game's vectors to render pixels, NGX's units)
	evaluate.mv_scale_y = inputs.motion_scale_y;
	evaluate.reset      = inputs.reset ? 1 : 0;
	evaluate.frame_ms   = frame_ms;
	// KYTY_DLSS_SHARPNESS=0..1 (a test): NGX's own sharpening of the output.
	static const float sharpness = [] {
		const char* value = std::getenv("KYTY_DLSS_SHARPNESS");
		return value != nullptr ? std::clamp(static_cast<float>(std::atof(value)), 0.0f, 1.0f) : 0.0f;
	}();
	evaluate.sharpness = sharpness;
	// (KYTY_SR_LOG: the first two evaluations' images and constants.)
	static const bool describe = std::getenv("KYTY_SR_LOG") != nullptr;
	if (describe && reports < 2) {
		++reports;
		const std::pair<const char*, const UpscaleImage*> images[] {
		    {"color", &inputs.color}, {"depth", &inputs.depth}, {"motion", &inputs.motion}, {"output", &inputs.output}};
		for (const auto& [name, image]: images) {
			std::printf("DLSS super resolution: %s image %p view %p layout %d format %d %ux%u level %u layer %u\n", name,
			            static_cast<void*>(static_cast<VkImage>(image->image)),
			            static_cast<void*>(static_cast<VkImageView>(image->view)), static_cast<int>(image->layout),
			            static_cast<int>(image->format), image->extent.width, image->extent.height, image->base_level,
			            image->base_layer);
		}
		std::printf("DLSS super resolution: jitter %g %g, motion scale %g %g, reset %d, %.2f ms\n", inputs.jitter_x,
		            inputs.jitter_y, inputs.motion_scale_x, inputs.motion_scale_y, inputs.reset ? 1 : 0, frame_ms);
		std::fflush(stdout);
	}
	// The depth in the layout NGX samples it in, for the evaluation only.
	const bool depth_moved =
	    IsDepthFormat(inputs.depth.format) && inputs.depth.layout != vk::ImageLayout::eShaderReadOnlyOptimal;
	if (depth_moved)
		DepthLayout(command, inputs.depth, inputs.depth.layout, vk::ImageLayout::eShaderReadOnlyOptimal, dispatch);
	const bool done = g_dlss->Evaluate(static_cast<VkCommandBuffer>(command), &evaluate) != 0;
	if (depth_moved)
		DepthLayout(command, inputs.depth, vk::ImageLayout::eShaderReadOnlyOptimal, inputs.depth.layout, dispatch);
	if (!done) {
		// (As the Streamline way, issue #87: NGX refusing its evaluations three times gives DLSS up for good, the game's
		// image stretched from then on.)
		static uint32_t failures = 0;
		if (++failures >= 3) {
			failed = true;
			std::printf("DLSS super resolution: NGX refused it %u times: the game's image is upscaled without it from here "
			            "on\n",
			            failures);
		} else {
			std::printf("DLSS super resolution: NGX's evaluation failed (the stretch for this frame)\n");
		}
		std::fflush(stdout);
	}
	return done;
#else
	(void)command;
	(void)inputs;
	return false;
#endif
}

void DestroyInputs(GraphicContext& graphics, FrameInputs& inputs) {
	for (auto* view: {&inputs.depth_view, &inputs.motion_view, &inputs.ui_view}) {
		if (*view != nullptr) {
			graphics.device.destroyImageView(*view, nullptr);
			*view = nullptr;
		}
	}
	inputs.ui_valid = false;
	for (auto* image: {&inputs.depth, &inputs.motion, &inputs.ui}) {
		if (image->image != nullptr) {
			graphics.DeleteImage(*image);
		}
	}
	DestroyHudlessInputs(graphics, inputs);
	inputs.valid = false;
}

bool UsesFsr() {
	return g_fsr;
}

bool FsrMarksHud() {
	return g_fsr || g_fsr_native;
}

const char* FsrName() {
#ifdef KYTY_FRAMEGEN_FSR
	if (g_fsr && g_fsr_state.fsr4.load(std::memory_order_relaxed)) return "FSR 4";
#endif
	return "FSR 3";
}

uint64_t VideoMemoryReserve() {
	// Frame generation's: FSR's measured with a 2560x1440 swapchain (the driver's share of the video memory 1.33-1.38 GiB
	// with it, 0.55-0.70 without; and the HUD images), DLSS-G's (NGX's optical flow and interpolation, its swapchain
	// images) and FSR's built in taken alike (issue #215: a 16 GB RTX 5080 with DLSS-G at 3440x1440 left the game a
	// budget it counted as roomy, kept the texture pools and ran out); KYTY_FSR_VRAM_RESERVE_MB overrides it. The super
	// resolution's (DLSS's model and history, or FSR 3's built in): 512 MiB more.
	static const uint64_t generation = [] {
		const char* value = std::getenv("KYTY_FSR_VRAM_RESERVE_MB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 1536ull) << 20u;
	}();
	const bool upscaling = g_upscale || FsrNative::UpscalerRequested();
	return (g_generate > 0 || g_fsr ? generation : 0) + (upscaling ? uint64_t {512} << 20u : 0);
}

void NoteUiCanvas(uint64_t address) {
#ifdef KYTY_FRAMEGEN_FSR_HUD
	if (g_ui_canvas.exchange(address, std::memory_order_relaxed) != address) {
		std::printf("Frame generation: FSR HUD from the interface canvas at 0x%" PRIx64 "\n", address);
		std::fflush(stdout);
	}
#else
	(void)address;
#endif
}

void RecordHudless(GraphicContext& graphics, vk::CommandBuffer command, const FrameInputs* inputs, vk::Image backbuffer,
                   vk::Extent2D extent, vk::Rect2D region, vk::Format format) {
#ifdef KYTY_FRAMEGEN_FSR_HUD
	auto& h = g_fsr_hud;
	h.ready = false;
	// KYTY_FSR_HUD=0: no HUD-less image (FSR interpolates the HUD with the scene); KYTY_FSR_HUD_ALPHA: the canvas alpha
	// that counts as HUD (default 0.02); KYTY_FSR_HUD_INVERT=1: the canvas's alpha taken as 1 - coverage.
	static const bool  enabled   = [] {
		const char* value = std::getenv("KYTY_FSR_HUD");
		return value == nullptr || std::atoi(value) != 0;
	}();
	static const float threshold = [] {
		const char* value = std::getenv("KYTY_FSR_HUD_ALPHA");
		return value != nullptr ? static_cast<float>(std::atof(value)) : 0.02f;
	}();
	static const bool  invert    = [] {
		const char* value = std::getenv("KYTY_FSR_HUD_INVERT");
		return value != nullptr && std::atoi(value) != 0;
	}();
	// (The Windows backend's swapchain while it generates, in its format; or FSR 3 built in, in the backbuffer's.)
	bool generating = g_fsr_native && PresenterGenerates() && format != vk::Format::eUndefined;
#ifdef KYTY_FRAMEGEN_FSR
	if (!g_fsr_native) {
		generating = OwnsSwapchain() && g_fsr_state.fg_context != nullptr && g_fsr_state.mode_on;
		format     = g_fsr_state.format;
	}
#endif
	if (!enabled || !generating || inputs == nullptr || !inputs->valid || !inputs->ui_valid || extent.width == 0 ||
	    extent.height == 0) {
		return;
	}
	if (!FsrHudSetup(graphics, format, extent)) return;
	h.index     = (h.index + 1u) % static_cast<uint32_t>(h.images.size());
	auto& image = h.images[h.index];
	// The backbuffer as the game image's blit left it, copied.
	ImageBarrier(command, backbuffer, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
	             vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eTransferRead);
	Transition(command, image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
	           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
	vk::ImageCopy copy {};
	copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.extent         = vk::Extent3D {extent.width, extent.height, 1};
	command.copyImage(backbuffer, vk::ImageLayout::eTransferSrcOptimal, image.image, vk::ImageLayout::eTransferDstOptimal, copy);
	// (And once more as the pass's input: it reads around each HUD pixel.)
	Transition(command, h.source, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
	           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
	command.copyImage(backbuffer, vk::ImageLayout::eTransferSrcOptimal, h.source.image, vk::ImageLayout::eTransferDstOptimal,
	                  copy);
	ImageBarrier(command, backbuffer, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eTransferDstOptimal,
	             vk::AccessFlagBits2::eTransferRead, vk::AccessFlagBits2::eTransferWrite);
	Transition(command, h.source, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
	           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eFragmentShader);
	// The HUD replaced in the copy by the scene around it.
	Transition(command, image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal,
	           vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
	           vk::PipelineStageFlagBits2::eColorAttachmentOutput);
	vk::RenderingAttachmentInfo color {};
	color.imageView   = h.views[h.index];
	color.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	color.loadOp      = vk::AttachmentLoadOp::eLoad;
	color.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = extent;
	rendering.layerCount           = 1;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &color;
	command.beginRendering(rendering);
	const vk::Viewport viewport {0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
	const vk::Rect2D   scissor {{0, 0}, extent};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &scissor);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, h.pipeline);
	std::array<vk::DescriptorImageInfo, 2> sampled {};
	sampled[0].imageView   = inputs->ui_view;
	sampled[0].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	sampled[1].imageView   = h.source_view;
	sampled[1].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t i = 0; i < writes.size(); ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
		writes[i].pImageInfo      = &sampled[i];
	}
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, h.layout, 0, static_cast<uint32_t>(writes.size()),
	                             writes.data());
	FsrHudParameters parameters {};
	parameters.region[0] = static_cast<float>(region.offset.x);
	parameters.region[1] = static_cast<float>(region.offset.y);
	parameters.region[2] = static_cast<float>(std::max(1u, region.extent.width));
	parameters.region[3] = static_cast<float>(std::max(1u, region.extent.height));
	parameters.extent[0] = static_cast<float>(extent.width);
	parameters.extent[1] = static_cast<float>(extent.height);
	parameters.threshold = threshold;
	parameters.invert    = invert ? 1.0f : 0.0f;
	command.pushConstants(h.layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(parameters), &parameters);
	command.draw(3, 1, 0, 0);
	command.endRendering();
	// KYTY_FSR_HUD_VIEW=1 (diagnostic): the HUD-less image shown instead of the frame (the HUD darkened where FSR is
	// told it is).
	static const bool view = std::getenv("KYTY_FSR_HUD_VIEW") != nullptr;
	if (view) {
		Transition(command, image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
		           vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eTransfer);
		command.copyImage(image.image, vk::ImageLayout::eTransferSrcOptimal, backbuffer, vk::ImageLayout::eTransferDstOptimal,
		                  copy);
	}
	Transition(command, image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
	           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
	static bool reported = false;
	if (!reported) {
		reported = true;
		std::printf("Frame generation: FSR HUD-less image in use (threshold %.3f%s)\n", threshold, invert ? ", inverted" : "");
		std::fflush(stdout);
	}
	h.ready = true;
#else
	(void)graphics, (void)command, (void)inputs, (void)backbuffer, (void)extent, (void)region, (void)format;
#endif
}

void OnFsrQueues(GraphicContext& graphics) {
#ifdef KYTY_FRAMEGEN_FSR
	if (!g_fsr) return;
	if (graphics.fsr_present_queue == nullptr || graphics.fsr_compute_queue == nullptr ||
	    graphics.fsr_acquire_queue == nullptr) {
		LOGF("frame generation: no queues for FSR's swapchain\n");
		std::printf("Frame generation: the GPU lacks the queues FSR's swapchain needs, FSR off\n");
		std::fflush(stdout);
		g_fsr      = false;
		g_generate = 0;
		return;
	}
	g_fsr_state.graphics = &graphics;
	g_ready              = true;
#else
	(void)graphics;
#endif
}

bool OwnsSwapchain() {
	return g_fsr && g_ready;
}

vk::Result CreateSwapchain(GraphicContext& graphics, const vk::SwapchainCreateInfoKHR& info, vk::SwapchainKHR* swapchain) {
#ifdef KYTY_FRAMEGEN_FSR
	auto& s = g_fsr_state;
	ffxCreateContextDescFrameGenerationSwapChainVK create {};
	create.header.type    = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
	create.physicalDevice = static_cast<VkPhysicalDevice>(graphics.physical_device);
	create.device         = static_cast<VkDevice>(graphics.device);
	VkSwapchainKHR handle = VK_NULL_HANDLE;
	create.swapchain      = &handle;
	create.allocator      = nullptr;
	create.createInfo     = static_cast<VkSwapchainCreateInfoKHR>(info);
	// The game queue is the emulator's present queue (presents go through QueuePresent there, under its lock), or without
	// one its graphics queue (under queue_mutex): the swapchain records the generated frame's work on it in the present
	// call.
	const vk::Queue game = graphics.present_queue != nullptr ? graphics.present_queue : graphics.queue;
	create.gameQueue         = {static_cast<VkQueue>(game), graphics.queue_family, nullptr};
	create.asyncComputeQueue = {static_cast<VkQueue>(graphics.fsr_compute_queue), graphics.fsr_compute_family, nullptr};
	create.presentQueue      = {static_cast<VkQueue>(graphics.fsr_present_queue), graphics.fsr_present_family, nullptr};
	create.imageAcquireQueue = {static_cast<VkQueue>(graphics.fsr_acquire_queue), graphics.fsr_compute_family, nullptr};
	// (FSR SDK 2 wants the API version the emulator was built against.)
	ffxCreateContextDescFrameGenerationSwapChainVersionVK version {};
	version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_VERSION_VK;
	version.version     = FFX_FRAMEGENERATION_SWAPCHAIN_VK_VERSION;
	if (s.sdk2) create.header.pNext = &version.header;
	if (!FsrOk(s.api.CreateContext(&s.swapchain_context, &create.header, nullptr), "swapchain")) {
		s.swapchain_context = nullptr;
		return vk::Result::eErrorInitializationFailed;
	}
	s.functions             = {};
	s.functions.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
	if (!FsrOk(s.api.Query(&s.swapchain_context, &s.functions.header), "swapchain functions") ||
	    s.functions.pOutAcquireNextImageKHR == nullptr || s.functions.pOutQueuePresentKHR == nullptr ||
	    s.functions.pOutGetSwapchainImagesKHR == nullptr) {
		(void)s.api.DestroyContext(&s.swapchain_context, nullptr);
		s.swapchain_context = nullptr;
		return vk::Result::eErrorInitializationFailed;
	}
	s.swapchain = handle;
	s.display   = info.imageExtent;
	s.format    = info.imageFormat;
	s.hdr       = info.imageColorSpace == vk::ColorSpaceKHR::eHdr10St2084EXT;
	s.run_with  = 0;
	s.run_without = 0;
	*swapchain  = handle;
	LOGF("frame generation: FSR swapchain %ux%u (%s)\n", s.display.width, s.display.height,
	     FsrProvider(&s.swapchain_context).c_str());
	// (FSR SDK 2's swapchain does not say its version.)
	if (static bool reported = false; !std::exchange(reported, true) && FsrProvider(&s.swapchain_context) != "?") {
		std::printf("Frame generation: FSR swapchain %s\n", FsrProvider(&s.swapchain_context).c_str());
		std::fflush(stdout);
	}
	return vk::Result::eSuccess;
#else
	(void)graphics;
	(void)info;
	(void)swapchain;
	return vk::Result::eErrorInitializationFailed;
#endif
}

void DestroySwapchain(GraphicContext& graphics, vk::SwapchainKHR swapchain) {
	(void)graphics;
	(void)swapchain;
#ifdef KYTY_FRAMEGEN_FSR
	auto& s = g_fsr_state;
	// Frame generation off, then the swapchain: its destruction stops its present thread and waits for its presents.
	// Then FSR's own queues (nothing else submits to them) are waited for, and only then go the frame generation context
	// and the HUD-less images its interpolation reads. Freed before the swapchain, they could still be in use: on an RTX
	// 5090 an F11 back to the window, frame generation on at 5120x2160 since a load, lost the device, and an F11 in the
	// main menu froze the picture while the game ran on (issue #174).
	FsrStopFrameGeneration();
	vk::SwapchainKHR left = nullptr; // FSR SDK 2's swapchain, which its context leaves
	if (s.swapchain_context != nullptr) {
		// FSR SDK 2's context leaves its swapchain to the program, to go after the context through the swapchain's own
		// function (AMD's order, its presents waited for first). Left on the window, the next swapchain could not be made
		// there: F11 or a resized window with frame generation on an RX 9070 XT exited "FSR frame generation swapchain
		// failed" (issue #270). FidelityFX SDK 1.1.4's context destroys its swapchain itself.
		if (s.sdk2) {
			ffxDispatchDescFrameGenerationSwapChainWaitForPresentsVK wait {};
			wait.header.type = FFX_API_DISPATCH_DESC_TYPE_FGSWAPCHAIN_WAIT_FOR_PRESENTS_VK;
			(void)FsrOk(s.api.Dispatch(&s.swapchain_context, &wait.header), "swapchain presents' wait");
			left = s.swapchain;
		}
		(void)s.api.DestroyContext(&s.swapchain_context, nullptr);
		s.swapchain_context = nullptr;
	}
	s.swapchain = nullptr;
	for (const vk::Queue queue: {graphics.fsr_compute_queue, graphics.fsr_present_queue, graphics.fsr_acquire_queue}) {
		if (queue != nullptr) (void)queue.waitIdle();
	}
	// (Only with FSR's queues idle: its last presents' copies may still read the swapchain's images.)
	if (left != nullptr && s.functions.pOutDestroySwapchainFFXAPI != nullptr) {
		s.functions.pOutDestroySwapchainFFXAPI(static_cast<VkDevice>(graphics.device), static_cast<VkSwapchainKHR>(left),
		                                       nullptr, nullptr);
	}
	// (The frame generation context is made for the display size: again with the next swapchain. Its swapchain gone,
	// FsrDestroyFrameGeneration configures nothing.)
	FsrDestroyFrameGeneration();
	FsrHudDestroy(graphics);
	s.functions = {};
#endif
}

vk::Result GetSwapchainImages(vk::Device device, vk::SwapchainKHR swapchain, uint32_t* count, vk::Image* images) {
#ifdef KYTY_FRAMEGEN_FSR
	return static_cast<vk::Result>(g_fsr_state.functions.pOutGetSwapchainImagesKHR(
	    static_cast<VkDevice>(device), static_cast<VkSwapchainKHR>(swapchain), count, reinterpret_cast<VkImage*>(images)));
#else
	(void)device, (void)swapchain, (void)count, (void)images;
	return vk::Result::eErrorInitializationFailed;
#endif
}

vk::Result AcquireNextImage(vk::Device device, vk::SwapchainKHR swapchain, uint64_t timeout, vk::Semaphore semaphore,
                            uint32_t* index) {
#ifdef KYTY_FRAMEGEN_FSR
	return static_cast<vk::Result>(g_fsr_state.functions.pOutAcquireNextImageKHR(
	    static_cast<VkDevice>(device), static_cast<VkSwapchainKHR>(swapchain), timeout, static_cast<VkSemaphore>(semaphore),
	    VK_NULL_HANDLE, index));
#else
	(void)device, (void)swapchain, (void)timeout, (void)semaphore, (void)index;
	return vk::Result::eErrorInitializationFailed;
#endif
}

vk::Result QueuePresent(vk::Queue queue, const vk::PresentInfoKHR& present) {
#ifdef KYTY_FRAMEGEN_FSR
	const VkPresentInfoKHR& info = present;
	return static_cast<vk::Result>(g_fsr_state.functions.pOutQueuePresentKHR(static_cast<VkQueue>(queue), &info));
#else
	(void)queue, (void)present;
	return vk::Result::eErrorInitializationFailed;
#endif
}

bool SkipsFrame(const FrameInputs* inputs) {
#ifdef KYTY_FRAMEGEN_FSR
	auto& s = g_fsr_state;
	if (!OwnsSwapchain() || !s.mode_on || (inputs != nullptr && inputs->valid)) return false;
	if (s.run_without + 1 >= FsrTurnOffAfter) return false; // (RecordPrepare turns generation off with it)
	++s.run_without;
	s.run_with = 0;
	return true;
#else
	(void)inputs;
	return false;
#endif
}

bool PresenterGenerates() {
#ifdef KYTY_FRAMEGEN_NGX
	return g_ready && g_generate > 0 && !g_fg_failed && (g_dlss != nullptr || g_fsr_native);
#else
	return false;
#endif
}

void RecordPrepare(GraphicContext& graphics, vk::CommandBuffer command, const FrameInputs* inputs, vk::Rect2D region) {
	(void)graphics;
#ifdef KYTY_FRAMEGEN_FSR
	auto& s = g_fsr_state;
	if (!OwnsSwapchain() || s.swapchain == nullptr) return;
	const bool usable = inputs != nullptr && inputs->valid;
	s.run_with        = usable ? s.run_with + 1 : 0;
	s.run_without     = usable ? 0 : s.run_without + 1;
	bool mode_on      = s.mode_on ? s.run_without < FsrTurnOffAfter : s.run_with >= FsrTurnOnAfter;
	if (mode_on && usable && !FsrEnsureFrameGeneration(inputs->render_extent)) mode_on = false;
	if (s.fg_context == nullptr) return; // (no frame with inputs yet: the swapchain presents as it is)
	if (mode_on != s.mode_on) {
		if (mode_on) std::printf("Frame generation: FSR on\n");
		else std::printf("Frame generation: FSR off: frames without inputs (%s)\n", MissingInputs());
		std::fflush(stdout);
		s.mode_on = mode_on;
		s.reset   = true;
	}
	// (A frame left unshown before this one: its motion is to the game's previous frame, not to the one shown last.)
	if (g_fsr_discontinuity.exchange(false, std::memory_order_relaxed) || (usable && inputs->cut)) s.reset = true;
	const auto now   = std::chrono::steady_clock::now();
	const float delta = s.last_frame.time_since_epoch().count() == 0
	                        ? 16.7f
	                        : std::chrono::duration<float, std::milli>(now - s.last_frame).count();
	s.last_frame     = now;
	const uint64_t id = ++s.frame_id;

	static const uint32_t debug_flags = [] {
		// KYTY_FRAMEGEN_DEBUG=1: tear and reset lines, 2: FSR's debug view.
		const char* value = std::getenv("KYTY_FRAMEGEN_DEBUG");
		const int   level = value != nullptr ? std::atoi(value) : 0;
		return level >= 2 ? uint32_t(FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW)
		       : level == 1 ? uint32_t(FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_TEAR_LINES | FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_RESET_INDICATORS)
		                    : 0u;
	}();
	ffxConfigureDescFrameGeneration config {};
	config.header.type                        = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
	config.swapChain                          = static_cast<VkSwapchainKHR>(s.swapchain);
	config.presentCallback                    = nullptr;
	config.frameGenerationCallback            = FsrGenerate;
	config.frameGenerationCallbackUserContext = &s.fg_context;
	config.frameGenerationEnabled             = mode_on;
	config.allowAsyncWorkloads                = false;
	config.flags                              = debug_flags;
	config.onlyPresentGenerated               = false;
	// (The HUD-less image of this frame, RecordHudless: FSR takes the HUD from the game's frame.)
	if (g_fsr_hud.ready) config.HUDLessColor = FsrImage(g_fsr_hud.images[g_fsr_hud.index]);
	g_fsr_hud.ready = false;
	config.generationRect                     = {region.offset.x, region.offset.y, static_cast<int32_t>(region.extent.width),
	                                             static_cast<int32_t>(region.extent.height)};
	config.frameID                            = id;
	if (!FsrOk(s.api.Configure(&s.fg_context, &config.header), "configure") || !mode_on) return;

	const auto camera = FsrDecompose(inputs->camera.view_proj);
	const float width  = static_cast<float>(inputs->render_extent.width);
	const float height = static_cast<float>(inputs->render_extent.height);
	// The game's vectors in UV units along the motion (see BeforePresent): FSR wants pixels towards the previous
	// position.
	static const float mv_sign = [] {
		const char* sign = std::getenv("KYTY_FRAMEGEN_MV_SIGN");
		return sign != nullptr && std::atof(sign) > 0 ? 1.0f : -1.0f;
	}();
	static const bool pixels_mv = [] {
		const char* mv = std::getenv("KYTY_FRAMEGEN_MV");
		return mv != nullptr && std::string(mv) == "px";
	}();
	// FSR SDK 2: the camera's axes in the prepare itself (V2; FSR 4's frame generation requires it), else after it.
	ffxDispatchDescFrameGenerationPrepareV2 prepare {};
	prepare.header.type       = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
	prepare.frameID           = id;
	prepare.flags             = debug_flags;
	prepare.commandList       = static_cast<VkCommandBuffer>(command);
	prepare.renderSize        = {inputs->render_extent.width, inputs->render_extent.height};
	// (The sample's convention: the negated offset in pixels of a projection jittered by 2 * offset / size.)
	prepare.jitterOffset      = {-inputs->camera.jitter_ndc[0] * width * 0.5f, inputs->camera.jitter_ndc[1] * height * 0.5f};
	prepare.motionVectorScale = pixels_mv ? FfxApiFloatCoords2D {mv_sign, mv_sign}
	                                      : FfxApiFloatCoords2D {mv_sign * width, mv_sign * height};
	prepare.frameTimeDelta    = delta;
	prepare.reset             = s.reset;
	prepare.cameraNear        = camera.valid ? camera.z_near : 0.1f;
	prepare.cameraFar         = FLT_MAX;
	prepare.cameraFovAngleVertical  = camera.valid ? 2.0f * std::atan(1.0f / std::fabs(camera.sy)) : 1.0f;
	prepare.viewSpaceToMetersFactor = 1.0f;
	prepare.depth             = FsrImage(inputs->depth);
	prepare.motionVectors     = FsrImage(inputs->motion);
	for (int i = 0; i < 3; ++i) {
		prepare.cameraPosition[i] = camera.position[i];
		prepare.cameraUp[i]       = camera.up[i];
		prepare.cameraRight[i]    = camera.right[i];
		prepare.cameraForward[i]  = camera.forward[i];
	}
	if (s.sdk2) {
		(void)FsrOk(s.api.Dispatch(&s.fg_context, &prepare.header), "prepare");
		return;
	}
	// (FidelityFX SDK 1.1.4's runtime: the prepare and its camera that SDK 2 calls deprecated.)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	ffxDispatchDescFrameGenerationPrepareCameraInfo camera_info {};
	camera_info.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO;
	std::copy_n(prepare.cameraPosition, 3, camera_info.cameraPosition);
	std::copy_n(prepare.cameraUp, 3, camera_info.cameraUp);
	std::copy_n(prepare.cameraRight, 3, camera_info.cameraRight);
	std::copy_n(prepare.cameraForward, 3, camera_info.cameraForward);
	ffxDispatchDescFrameGenerationPrepare old {};
	old.header.type             = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
	old.header.pNext            = camera.valid ? &camera_info.header : nullptr;
	old.frameID                 = prepare.frameID;
	old.flags                   = prepare.flags;
	old.commandList             = prepare.commandList;
	old.renderSize              = prepare.renderSize;
	old.jitterOffset            = prepare.jitterOffset;
	old.motionVectorScale       = prepare.motionVectorScale;
	old.frameTimeDelta          = prepare.frameTimeDelta;
	old.cameraNear              = prepare.cameraNear;
	old.cameraFar               = prepare.cameraFar;
	old.cameraFovAngleVertical  = prepare.cameraFovAngleVertical;
	old.viewSpaceToMetersFactor = prepare.viewSpaceToMetersFactor;
	old.depth                   = prepare.depth;
	old.motionVectors           = prepare.motionVectors;
	(void)FsrOk(s.api.Dispatch(&s.fg_context, &old.header), "prepare");
#pragma clang diagnostic pop
#else
	(void)command, (void)inputs, (void)region;
#endif
}

void NotePresented(uint32_t frames) {
	g_presented_frames.store(std::max(frames, 1u), std::memory_order_relaxed);
}

bool PresenterGeneratesHdr() {
	return PresenterGenerates() && g_fsr_native;
}

uint32_t Generate(GraphicContext& graphics, vk::CommandBuffer command, const FrameInputs* inputs, const UpscaleImage& backbuffer,
                  vk::Rect2D region, std::span<const UpscaleImage> outputs, bool hdr) {
#ifdef KYTY_FRAMEGEN_NGX
	if (!PresenterGenerates()) return 0;
	const bool discontinuity =
	    g_fg_discontinuity.exchange(false, std::memory_order_relaxed) || (inputs != nullptr && inputs->cut);
	if (inputs == nullptr || !inputs->valid) {
		// (Menus, movies, loading: the frame shown as it is; the next one with inputs starts DLSS-G again.)
		g_fg_history = false;
		return 0;
	}
	if (g_fsr_native) {
		// FSR 3 built in: one frame between the last one and this, from the same inputs as DLSS-G's.
		if (outputs.empty()) return 0;
		const auto image = [](const VulkanImage& source, vk::ImageView view) {
			return UpscaleImage {source.image, view, vk::ImageLayout::eShaderReadOnlyOptimal, source.format,
			                     vk::Extent2D {source.extent.width, source.extent.height}, source.usage, 0, 0};
		};
		FsrNative::InterpolationInputs fsr;
		fsr.backbuffer   = backbuffer;
		fsr.depth        = image(inputs->depth, inputs->depth_view);
		fsr.motion       = image(inputs->motion, inputs->motion_view);
		fsr.region       = region;
		const float width  = static_cast<float>(inputs->render_extent.width);
		const float height = static_cast<float>(inputs->render_extent.height);
		// (FSR's documented convention, DLSS-G's here: the projection's offset in pixels, y down.)
		fsr.jitter_px = {inputs->camera.jitter_ndc[0] * width * 0.5f, -inputs->camera.jitter_ndc[1] * height * 0.5f};
		if (const auto camera = Decompose(inputs->camera.view_proj); camera.valid) {
			fsr.near_plane = camera.z_near;
			fsr.fov_y      = 2.0f * std::atan(1.0f / std::fabs(camera.sy));
		}
		// The game's frame time: the time between the frames given here, a long gap (a load, a pause) left out.
		static auto last  = std::chrono::steady_clock::time_point {};
		static float frame_ms = 16.7f;
		const auto now = std::chrono::steady_clock::now();
		if (last != std::chrono::steady_clock::time_point {}) {
			const float ms = std::chrono::duration<float, std::milli>(now - last).count();
			if (ms > 2.0f && ms < 100.0f) frame_ms = ms;
		}
		last         = now;
		fsr.frame_ms = frame_ms;
		fsr.reset    = !g_fg_history || discontinuity;
		fsr.pq       = hdr;
		// (This frame's HUD-less image, RecordHudless: FSR takes the HUD and the menus from the game's frame.)
		if (auto& h = g_fsr_hud; h.ready) {
			const auto& hud = h.images[h.index];
			fsr.hudless     = {hud.image,       h.views[h.index], vk::ImageLayout::eShaderReadOnlyOptimal, hud.format,
			                   backbuffer.extent, hud.usage,       0,                                      0};
		}
		const bool made = FsrNative::Interpolate(graphics, command, fsr, outputs[0]);
		g_fg_history    = true;
		return made ? 1u : 0u;
	}
	const auto count = static_cast<uint32_t>(std::min<size_t>(outputs.size(), static_cast<size_t>(g_generate)));
	if (count == 0) return 0;
	// A render size half the feature's or less, the backbuffer the same (#215: 646x270 against 2584x1080, a few times an
	// hour, the game's menus or loading): those frames go without DLSS-G and the feature stays for the size that comes
	// back (made again at each change, it took the memory twice over each time; one such remake ran out of it). Ten
	// seconds of it: made for that size.
	const bool small = g_fg_created && g_fg_backbuffer == backbuffer.extent && g_fg_format == backbuffer.format &&
	                   inputs->render_extent.width * 2 <= g_fg_render.width &&
	                   inputs->render_extent.height * 2 <= g_fg_render.height;
	if (!small) {
		g_fg_small_since = {};
	} else {
		const auto now = std::chrono::steady_clock::now();
		if (g_fg_small_since == std::chrono::steady_clock::time_point {}) g_fg_small_since = now;
		if (now - g_fg_small_since < std::chrono::seconds(10)) {
			g_fg_history = false;
			return 0;
		}
	}
	if (!g_fg_created || g_fg_backbuffer != backbuffer.extent || g_fg_format != backbuffer.format ||
	    g_fg_render != inputs->render_extent) {
		const auto now = std::chrono::steady_clock::now();
		if (g_fg_refused.count != 0 && g_fg_refused.backbuffer == backbuffer.extent &&
		    g_fg_refused.format == backbuffer.format && g_fg_refused.render == inputs->render_extent &&
		    now - g_fg_refused.at < FgRetry)
			return 0;
		KytyDlssgFeature feature {};
		feature.width         = backbuffer.extent.width;
		feature.height        = backbuffer.extent.height;
		feature.format        = static_cast<VkFormat>(backbuffer.format);
		feature.render_width  = inputs->render_extent.width;
		feature.render_height = inputs->render_extent.height;
		if (g_dlss->FrameGenCreate(static_cast<VkCommandBuffer>(command), &feature) == 0) {
			// (The bridge retired the feature before: none now.)
			g_fg_created = false;
			g_fg_refused = {backbuffer.extent, inputs->render_extent, backbuffer.format, now, g_fg_refused.count + 1};
			g_fg_failed  = g_fg_refused.count >= FgRefusals;
			std::printf("Frame generation: NGX did not make DLSS-G for %ux%u (depth and motion %ux%u): %s\n", feature.width,
			            feature.height, feature.render_width, feature.render_height,
			            g_fg_failed ? "off" : "frames go without it, made again at other sizes or in 30 s");
			std::fflush(stdout);
			return 0;
		}
		g_fg_refused.count = 0;
		std::printf("Frame generation: DLSS-G %ux%u, depth and motion %ux%u, %ux (NVIDIA NGX)\n", feature.width,
		            feature.height, feature.render_width, feature.render_height, count + 1);
		std::fflush(stdout);
		g_fg_created    = true;
		g_fg_backbuffer = backbuffer.extent;
		g_fg_format     = backbuffer.format;
		g_fg_render     = inputs->render_extent;
		g_fg_history    = false;
	}
	// (No frame before this one with inputs, or one left unshown in between: this one starts DLSS-G's history and its
	// generated frames are not shown.)
	const bool reset = !g_fg_history || discontinuity;

	KytyDlssgEvaluate evaluate {};
	evaluate.backbuffer = BridgeImage(backbuffer, vk::ImageAspectFlagBits::eColor);
	evaluate.depth      = BridgeImage(inputs->depth, inputs->depth_view);
	evaluate.motion     = BridgeImage(inputs->motion, inputs->motion_view);
	if (HudlessOn() && inputs->hudless.has_display) evaluate.hudless = BridgeImage(inputs->hudless.display, inputs->hudless.display_view);
	// The camera as the Streamline way gives it (BeforePresent): defaults for a frame whose matrix does not split.
	auto view_to_clip = Identity(), clip_to_view = Identity(), clip_to_prev = Identity(), prev_to_clip = Identity();
	evaluate.mv_scale_x = evaluate.mv_scale_y = 1.0f;
	evaluate.camera_up[1]    = 1.0f;
	evaluate.camera_right[0] = 1.0f;
	evaluate.camera_fwd[2]   = 1.0f;
	evaluate.camera_near     = 0.1f;
	evaluate.camera_far      = 10000.0f;
	evaluate.camera_fov      = 1.0f;
	evaluate.camera_aspect   = 16.0f / 9.0f;
	if (const auto camera = Decompose(inputs->camera.view_proj); camera.valid) {
		// Projection in view space: x' = sx*x + cx*z, y' = sy*y + cy*z, z' = near, w' = z.
		RowMatrix projection {};
		projection[0]  = camera.sx;
		projection[5]  = camera.sy;
		projection[8]  = camera.cx;
		projection[9]  = camera.cy;
		projection[11] = 1.0f;
		projection[14] = camera.z_near;
		view_to_clip   = projection;
		(void)Inverted(projection, clip_to_view);
		const auto current  = Transposed(inputs->camera.view_proj);
		const auto previous = Transposed(inputs->camera.prev_view_proj);
		if (RowMatrix clip_to_world {}; Inverted(current, clip_to_world)) {
			clip_to_prev = Multiply(clip_to_world, previous);
			(void)Inverted(clip_to_prev, prev_to_clip);
		}
		const float width  = static_cast<float>(inputs->render_extent.width);
		const float height = static_cast<float>(inputs->render_extent.height);
		evaluate.jitter_x  = inputs->camera.jitter_ndc[0] * width * 0.5f;
		evaluate.jitter_y  = -inputs->camera.jitter_ndc[1] * height * 0.5f;
		// The game's vectors point along the motion, in UV units (its TAA reads the history at uv - mv); DLSS-G wants
		// them towards the previous position, so they are negated (the Streamline way's KYTY_FRAMEGEN_MV_SIGN and
		// KYTY_FRAMEGEN_MV=px, the same tests).
		static const bool pixels_mv = [] {
			const char* mv = std::getenv("KYTY_FRAMEGEN_MV");
			return mv != nullptr && std::string(mv) == "px";
		}();
		static const float mv_sign = [] {
			const char* sign = std::getenv("KYTY_FRAMEGEN_MV_SIGN");
			return sign != nullptr && std::atof(sign) > 0 ? 1.0f : -1.0f;
		}();
		evaluate.mv_scale_x = pixels_mv ? mv_sign / width : mv_sign;
		evaluate.mv_scale_y = pixels_mv ? mv_sign / height : mv_sign;
		for (int i = 0; i < 3; ++i) {
			evaluate.camera_pos[i]   = camera.position[i];
			evaluate.camera_up[i]    = camera.up[i];
			evaluate.camera_right[i] = camera.right[i];
			evaluate.camera_fwd[i]   = camera.forward[i];
		}
		evaluate.camera_near   = camera.z_near;
		evaluate.camera_far    = 100000.0f;
		evaluate.camera_fov    = 2.0f * std::atan(1.0f / std::fabs(camera.sy));
		evaluate.camera_aspect = std::fabs(camera.sy / camera.sx);
	}
	Copy16(evaluate.camera_view_to_clip, view_to_clip);
	Copy16(evaluate.clip_to_camera_view, clip_to_view);
	Copy16(evaluate.clip_to_prev_clip, clip_to_prev);
	Copy16(evaluate.prev_clip_to_clip, prev_to_clip);
	evaluate.depth_inverted = 1;
	evaluate.reset          = reset ? 1 : 0;
	// The backbuffer's part with the game image (KYTY_PRESENT_ASPECT, KYTY_RENDER_ASPECT: black around it).
	if (region.offset.x != 0 || region.offset.y != 0 || region.extent.width != backbuffer.extent.width ||
	    region.extent.height != backbuffer.extent.height) {
		evaluate.backbuffer_x      = static_cast<uint32_t>(region.offset.x);
		evaluate.backbuffer_y      = static_cast<uint32_t>(region.offset.y);
		evaluate.backbuffer_width  = region.extent.width;
		evaluate.backbuffer_height = region.extent.height;
	}
	evaluate.multi_frame_count = count;
	uint32_t made              = 0;
	for (uint32_t i = 0; i < count; ++i) {
		evaluate.output            = BridgeImage(outputs[i], vk::ImageAspectFlagBits::eColor);
		evaluate.multi_frame_index = i + 1;
		if (g_dlss->FrameGenEvaluate(static_cast<VkCommandBuffer>(command), &evaluate) == 0) break;
		++made;
	}
	// (Three in a row: made again after FgRetry, as when NGX did not make it.)
	static uint32_t failures = 0;
	if (made < count) {
		if (++failures >= 3) {
			failures     = 0;
			g_fg_created = false;
			g_fg_refused = {backbuffer.extent, inputs->render_extent, backbuffer.format, std::chrono::steady_clock::now(),
			                g_fg_refused.count + 1};
			g_fg_failed  = g_fg_refused.count >= FgRefusals;
			std::printf("Frame generation: NGX refused DLSS-G 3 times in a row: %s\n",
			            g_fg_failed ? "off from here on" : "made again in 30 s");
		} else {
			std::printf("Frame generation: NGX's DLSS-G evaluation failed (this frame shown alone)\n");
		}
		std::fflush(stdout);
		g_fg_history = false;
		return 0;
	}
	failures     = 0;
	g_fg_history = true;
	return reset ? 0u : made;
#else
	(void)graphics;
	(void)command;
	(void)inputs;
	(void)backbuffer;
	(void)region;
	(void)outputs;
	(void)hdr;
	return 0;
#endif
}

} // namespace Libs::Graphics::FrameGen
