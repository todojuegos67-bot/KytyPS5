// KYTY_GODRAYS=0|1|2|3 ("God rays"; godrays.comp): crepuscular rays from the sun through windows, gaps and trees, and soft
// volumetric halos around torches and fires, on Demon's Souls' PS5 Pro path. Recorded in FrameGen::Upscale after the other
// screen-space passes (last, after shine), on the pre-exposure HDR color, before the upscaler.
//
// Per frame (compute, one pipeline, push descriptors), six dispatches:
//   lights    a 16x9 grid of cells, 64 fixed taps a cell: the brightest emissive tap (KYTY_GODRAYS_LIGHT, exposed 3.0)
//   track     one thread: up to 2 / 4 / 8 lights tracked across frames (motion-reprojected, matched, faded in / out)
//   volume    quarter resolution: the occlusion-aware radial march toward the sun on screen (32 / 48 / 64 taps) and,
//             level >= 2, the view-ray march shadowed toward the sun against the depth (12 / 16 steps x 5 taps);
//             jittered (IGN + golden-ratio rotation per frame), raw
//   temporal  quarter resolution: reprojected history (depth-tested), length up to 20 / 32 (dark) frames when still,
//             ~10 on motion, clamped to the raw neighbourhood (10-25: the old 0.1 EMA of a jittered, sub-texel-moved
//             sample left a visible dither when standing still, worst in dark fog)
//   spatial   quarter resolution: a depth-aware 3x3 bilateral blur before the composite
//   composite full resolution: the volume x the sun's hue + the lights' analytic halos; added display-referred
//             (x the game's exposure), hue-preserving, soft-capped (0.10 / 0.18 / 0.28 x strength exposed luminance)
// The sun direction comes from KYTY_RT_SHADOWS (CShadowApply's latched sun): without it, lights only. The sun on screen
// fades in / out over ~0.5 s as it crosses the frame edge (+-25 %); at level 1 the shafts go with it.
// Bajo / Medio (1 / 2): the in-scattering boosted where light is actually visible (the sky mask of the radial march,
// partly lit view rays = shafts through gaps), halos and the cap raised a little. KYTY_GODRAYS_DEBUG=1: our light only.
// gr4: every stage sanitized (NaN / Inf -> 0, a bad depth -> far), the history restarted on a gap or never-written
// texels, invalid motion vectors rejected, the sky's history short, our light never blackens a pixel; boosts lowered.
// Every 10 s: "GODRAYS: stats" with the GPU ms, the sun on / off screen and the lights used.
//
// (hwrt3v) KYTY_HWRT_VOLUMETRIC=0|1|2|3 (needs KYTY_HWRT=1; the sun from KYTY_RT_SHADOWS / KYTY_RT_SUN / KYTY_HWRT_SHADOW):
// the view-ray march's visibility toward the sun becomes a real ray query into the scene TLAS (hwrt-scene.cpp) at every
// step instead of the depth-buffer test (godrays_hw_spv.h, godrays.comp with KYTY_HWRT_VOLUMETRIC): the shafts are cut by
// the captured geometry, off-screen occluders included, and the march runs at every level (1: 12 steps, 2: 16, 3: 24,
// first hit, 250 m). Quarter resolution with the pass's own temporal + spatial filter and capped additive composite
// (never darkens). KYTY_GODRAYS unset: it runs at the KYTY_HWRT_VOLUMETRIC level. No TLAS (yet) or no frame camera: the
// depth-buffer march as before. Unset / 0: nothing changes (the same pipeline, push size and SPIR-V as before).
#include "local/godrays.h"
#include "local/hwrt-scene.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/godrays_spv.h"
#include "gpu_tiler_shaders/godrays_hw_spv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace Libs::Graphics::FrameGen {
namespace {

constexpr uint32_t Bindings    = 10;
constexpr uint32_t StatsWord   = 4;
constexpr uint32_t StatsCount  = 16;
constexpr uint32_t Cells       = 16 * 9;
constexpr uint32_t FrameBytes  = (24 + 16 * 4 + Cells * 8) * 4;
constexpr uint32_t ExposureIdx = 0; // hdr.x (1.0) when the game gives no exposure

struct GodraysPush {
	float   proj[4];
	float   depth_map[4];
	float   size[4];
	float   sun[4];
	float   sun_uv[4];
	float   motion[4];
	int32_t misc[4];
	float   extra[4];
};
static_assert(sizeof(GodraysPush) == 128);
// (KYTY_HWRT_VOLUMETRIC) the same, then view -> the TLAS's world (rows) and the ray parameters.
struct GodraysPushHw {
	GodraysPush base;
	float       w[12];
	float       hw[4]; // x: 1 = TLAS bound this frame, y: ray length (m), z: march length (m), w: level
};
static_assert(sizeof(GodraysPushHw) == 192);

int HwVolLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_HWRT_VOLUMETRIC");
		const int   q     = value != nullptr && *value != '\0' ? std::clamp(std::atoi(value), 0, 3) : 0;
		if (q > 0) {
			std::printf("HWRTVOL: KYTY_HWRT_VOLUMETRIC=%s read (level %d)%s\n", value, q,
			            HwScene::Requested() ? "" : ": needs KYTY_HWRT=1 too: off");
			std::fflush(stdout);
		}
		return HwScene::Requested() ? q : 0;
	}();
	return level;
}

bool Invert4d(const double* m, double* inv) {
	double a[4][8];
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 8; ++j) a[i][j] = j < 4 ? m[i * 4 + j] : (j - 4 == i ? 1.0 : 0.0);
	for (int c = 0; c < 4; ++c) {
		int p = c;
		for (int r = c + 1; r < 4; ++r)
			if (std::abs(a[r][c]) > std::abs(a[p][c])) p = r;
		if (!(std::abs(a[p][c]) > 1e-20)) return false;
		if (p != c)
			for (int j = 0; j < 8; ++j) std::swap(a[p][j], a[c][j]);
		const double d = a[c][c];
		for (int j = 0; j < 8; ++j) a[c][j] /= d;
		for (int r = 0; r < 4; ++r) {
			if (r == c) continue;
			const double f = a[r][c];
			for (int j = 0; j < 8; ++j) a[r][j] -= f * a[c][j];
		}
	}
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j) inv[i * 4 + j] = a[i][j + 4];
	return true;
}

// View -> world rows (3x4) from the upscaler's view_to_clip (row vectors) and the G-buffer camera's view-projection: the
// same derivation as KYTY_HWRT_SHADOW's (frame-gen.cpp RtSunPrepare).
bool ViewToWorld(const std::array<float, 16>& m, const std::array<float, 16>& vp, float* w) {
	double pc[16] {}, pinv[16] {};
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j) pc[i * 4 + j] = m[j * 4 + i];
	if (!Invert4d(pc, pinv)) return false;
	double v[16] {};
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			for (int k = 0; k < 4; ++k) v[i * 4 + j] += pinv[i * 4 + k] * vp[k * 4 + j];
	for (int i = 0; i < 3; ++i) {
		const double n = std::sqrt(v[i * 4] * v[i * 4] + v[i * 4 + 1] * v[i * 4 + 1] + v[i * 4 + 2] * v[i * 4 + 2]);
		if (!(n > 1e-9) || !std::isfinite(n)) return false;
		for (int j = 0; j < 4; ++j) v[i * 4 + j] /= n;
	}
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) w[i * 4 + j] = static_cast<float>(v[j * 4 + i]);
		w[i * 4 + 3] = static_cast<float>(-(v[0 * 4 + i] * v[3] + v[1 * 4 + i] * v[7] + v[2 * 4 + i] * v[11]));
		for (int j = 0; j < 4; ++j)
			if (!std::isfinite(w[i * 4 + j])) return false;
	}
	return true;
}

struct Img {
	VkImage       image  = VK_NULL_HANDLE;
	VmaAllocation memory = nullptr;
	vk::ImageView view   = nullptr;
};

struct {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             point = nullptr, linear = nullptr;
	bool                    failed = false;
	bool                    hw = false; // (KYTY_HWRT_VOLUMETRIC: the ray query pipeline, binding 10 the TLAS, 192-byte push)
	uint32_t                hw_traced = 0, hw_stale = 0, hw_none = 0;
	// (hwrt3v-2) the last good view -> world rows (held over frames without a camera: never the depth march once traced),
	// the 10 s transition counts: path switches, held frames, sun on / off (hysteresis 0.35 / 0.65), history resets.
	std::array<float, 12>   hw_w {};
	bool                    hw_have_w = false, hw_last_path = false, sun_on = false;
	uint32_t                hw_held = 0, path_switches = 0, sun_ons = 0, sun_offs = 0, resets = 0;
	Img                     out;
	std::array<Img, 2>      vol;
	Img                     raw, filt; // (quarter: this frame's raw volume, the filtered one)
	vk::Extent2D            extent {}, qextent {};
	bool                    ready = false, history_valid = false;
	VkBuffer                frame_buf = VK_NULL_HANDLE, read_buf = VK_NULL_HANDLE;
	VmaAllocation           frame_mem = nullptr, read_mem = nullptr;
	const uint32_t*         mapped = nullptr;
	bool                    cleared = false;
	uint64_t                frame = 0;
	std::array<std::pair<Img, uint64_t>, 8> retired {};
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms = 0.0, max_ms = 0.0;
	uint32_t                samples = 0;
	std::array<uint32_t, StatsCount> stat_last {};
	float                   on_screen = 0.0f; // the smoothed on-screen factor
	float                   sun_scale = -1.0f, sun_feed_min = 1.0f, sun_feed_max = 0.0f; // (hwrt5v-2)
	float                   sun_held  = 0.0f; // the smoothed sun weight (10-25: fades instead of popping when the sun goes stale)
	std::array<float, 3>    sun_dir {};       // the last sun given (view space)
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now(), last_frame {};
} g_gr;

struct {
	std::mutex                            mutex;
	std::array<float, 3>                  dir {};
	float                                 weight = 0.0f;
	std::chrono::steady_clock::time_point when {};
} g_gr_sun;

float EnvFloat(const char* name, float fallback, float lo, float hi) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') return fallback;
	const float v = static_cast<float>(std::atof(value));
	return std::isfinite(v) ? std::clamp(v, lo, hi) : fallback;
}

void Barrier(vk::CommandBuffer command, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src, vk::PipelineStageFlags2 dst_stage,
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
}
void ToGeneral(vk::CommandBuffer command, VkImage image) {
	vk::ImageMemoryBarrier2 b {};
	b.srcStageMask        = vk::PipelineStageFlagBits2::eNone;
	b.dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
	b.dstAccessMask       = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite;
	b.oldLayout           = vk::ImageLayout::eUndefined;
	b.newLayout           = vk::ImageLayout::eGeneral;
	b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image               = vk::Image(image);
	b.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &b;
	command.pipelineBarrier2(dependency);
}

bool MakeImage(GraphicContext& graphics, vk::Format format, vk::Extent2D extent, Img& img) {
	VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create.imageType     = VK_IMAGE_TYPE_2D;
	create.format        = static_cast<VkFormat>(format);
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.samples       = VK_SAMPLE_COUNT_1_BIT;
	create.tiling        = VK_IMAGE_TILING_OPTIMAL;
	create.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
	               VK_IMAGE_USAGE_STORAGE_BIT;
	create.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
	create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO;
	if (vmaCreateImage(graphics.allocator, &create, &allocate, &img.image, &img.memory, nullptr) != VK_SUCCESS) {
		img.image = VK_NULL_HANDLE;
		return false;
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = vk::Image(img.image);
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = format;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &img.view) != vk::Result::eSuccess) {
		img.view = nullptr;
		return false;
	}
	return true;
}
void Destroy(GraphicContext& graphics, Img& img) {
	if (img.view != nullptr) graphics.device.destroyImageView(img.view, nullptr);
	if (img.image != VK_NULL_HANDLE) vmaDestroyImage(graphics.allocator, img.image, img.memory);
	img = {};
}
void Retire(GraphicContext& graphics, Img& img) {
	if (img.image == VK_NULL_HANDLE) return;
	for (auto& slot: g_gr.retired)
		if (slot.first.image == VK_NULL_HANDLE) {
			slot = {img, g_gr.frame};
			img  = {};
			return;
		}
	Destroy(graphics, img); // (no free slot: only after many resizes in 16 frames)
}
void FreeRetired(GraphicContext& graphics) {
	for (auto& slot: g_gr.retired)
		if (slot.first.image != VK_NULL_HANDLE && g_gr.frame >= slot.second + 16) Destroy(graphics, slot.first), slot.second = 0;
}

vk::DescriptorType DescType(uint32_t i) {
	if (i == 10) return vk::DescriptorType::eAccelerationStructureKHR;
	return (i == 6 || i == 7) ? vk::DescriptorType::eStorageBuffer
	       : (i == 4 || i == 5 || i == 9) ? vk::DescriptorType::eStorageImage
	                            : vk::DescriptorType::eCombinedImageSampler;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_gr.pipeline != nullptr) return true;
	if (g_gr.failed) return false;
	g_gr.failed       = true;
	const auto device = graphics.device;
	g_gr.hw = HwVolLevel() > 0 && HwScene::Active() &&
	          graphics.GetPhysicalDeviceProperties().limits.maxPushConstantsSize >= sizeof(GodraysPushHw);
	if (HwVolLevel() > 0) {
		std::printf("HWRTVOL: %s\n", g_gr.hw ? "the god rays' view-ray march traces the scene TLAS (godrays.comp with ray queries)"
		                                      : "no ray queries on this device (KYTY_HWRT inactive) or push constants < 192 B: depth-buffer march");
		std::fflush(stdout);
	}
	const uint32_t nb = g_gr.hw ? Bindings + 1 : Bindings;
	std::array<vk::DescriptorSetLayoutBinding, Bindings + 1> bindings {};
	for (uint32_t i = 0; i < nb; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = DescType(i);
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = nb;
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_gr.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0,
	                                   static_cast<uint32_t>(g_gr.hw ? sizeof(GodraysPushHw) : sizeof(GodraysPush))};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_gr.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_gr.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = g_gr.hw ? sizeof(GODRAYS_HW_SPV) : sizeof(GODRAYS_SPV);
	module_info.pCode       = g_gr.hw ? GODRAYS_HW_SPV : GODRAYS_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_gr.layout;
	const auto created         = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_gr.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_gr.pipeline = nullptr;
		return false;
	}
	for (int i = 0; i < 2; ++i) {
		vk::SamplerCreateInfo sampler {};
		sampler.magFilter    = i == 0 ? vk::Filter::eNearest : vk::Filter::eLinear;
		sampler.minFilter    = sampler.magFilter;
		sampler.mipmapMode   = vk::SamplerMipmapMode::eNearest;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		sampler.maxLod       = VK_LOD_CLAMP_NONE;
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g_gr.point : &g_gr.linear) != vk::Result::eSuccess) return false;
	}
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = FrameBytes;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VmaAllocationCreateInfo dev {};
	dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_gr.frame_buf, &g_gr.frame_mem, nullptr) != VK_SUCCESS) {
		g_gr.frame_buf = VK_NULL_HANDLE;
		return false;
	}
	buffer.size  = StatsCount * 4;
	buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_gr.read_buf, &g_gr.read_mem, &mapped) != VK_SUCCESS) {
		g_gr.read_buf = VK_NULL_HANDLE;
		return false;
	}
	g_gr.mapped = static_cast<const uint32_t*>(mapped.pMappedData);
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = 8 * 2;
	if (device.createQueryPool(&query, nullptr, &g_gr.pool) != vk::Result::eSuccess) g_gr.pool = nullptr;
	g_gr.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g_gr.failed    = false;
	return true;
}

bool EnsureImages(GraphicContext& graphics, vk::Extent2D extent) {
	if (g_gr.out.image != VK_NULL_HANDLE && g_gr.extent == extent) return true;
	Retire(graphics, g_gr.out);
	Retire(graphics, g_gr.vol[0]);
	Retire(graphics, g_gr.vol[1]);
	Retire(graphics, g_gr.raw);
	Retire(graphics, g_gr.filt);
	g_gr.ready = false, g_gr.history_valid = false;
	g_gr.extent  = extent;
	g_gr.qextent = vk::Extent2D {(extent.width + 3) / 4, (extent.height + 3) / 4};
	return MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, extent, g_gr.out) &&
	       MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, g_gr.qextent, g_gr.vol[0]) &&
	       MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, g_gr.qextent, g_gr.vol[1]) &&
	       MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, g_gr.qextent, g_gr.raw) &&
	       MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, g_gr.qextent, g_gr.filt);
}

void ReadTiming(GraphicContext& graphics) {
	if (g_gr.pool == nullptr) return;
	while (g_gr.read + 4 <= g_gr.written) {
		const auto              entry = static_cast<uint32_t>(g_gr.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (graphics.device.getQueryPoolResults(g_gr.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		if (stamps[1] > stamps[0]) {
			const double ms = static_cast<double>(stamps[1] - stamps[0]) * g_gr.period_ns / 1e6;
			g_gr.sum_ms += ms, g_gr.max_ms = std::max(g_gr.max_ms, ms);
			++g_gr.samples;
		}
		++g_gr.read;
	}
}

void Log(int level, bool have_sun) {
	const auto now = std::chrono::steady_clock::now();
	if (now - g_gr.last_log < std::chrono::seconds(10) || g_gr.mapped == nullptr) return;
	g_gr.last_log = now;
	std::array<uint32_t, StatsCount> cur {};
	std::memcpy(cur.data(), g_gr.mapped, sizeof(cur)); // (a few frames late: statistics only)
	std::array<uint32_t, StatsCount> d {};
	for (uint32_t i = 0; i < StatsCount; ++i) d[i] = cur[i] - g_gr.stat_last[i];
	g_gr.stat_last  = cur;
	const double fr = std::max<double>(d[0], 1.0);
	std::printf("GODRAYS: stats (level %d) GPU %.3f ms (max %.3f), sun %s (on screen %.0f %% of frames), lights %.2f a frame\n", level,
	            g_gr.samples > 0 ? g_gr.sum_ms / g_gr.samples : 0.0, g_gr.max_ms,
	            !have_sun ? "unknown (needs KYTY_RT_SHADOWS)" : g_gr.on_screen > 0.5f ? "on screen" : "off screen", 100.0 * d[2] / fr,
	            static_cast<double>(d[1]) / fr);
	std::fflush(stdout);
	g_gr.sum_ms = 0.0, g_gr.max_ms = 0.0, g_gr.samples = 0;
	if (g_gr.hw) {
		std::printf("HWRTVOL: 10 s: %u frames traced with this frame's TLAS, %u with an older one, %u without (depth march; level %d, "
		            "volume %ux%u; the GPU above is the whole god-ray pass)\n",
		            g_gr.hw_traced, g_gr.hw_stale, g_gr.hw_none, HwVolLevel(), g_gr.qextent.width, g_gr.qextent.height);
		std::printf("HWRTVOL: 10 s transitions: path %s, %u switches TLAS<->depth, %u frames on held camera rows; sun %u on / %u off "
		            "(weight %.2f); halos %u on / %u off; history resets %u\n",
		            g_gr.hw_last_path ? "TLAS" : "depth", g_gr.path_switches, g_gr.hw_held, g_gr.sun_ons, g_gr.sun_offs,
		            static_cast<double>(g_gr.sun_held), d[3], d[4], g_gr.resets);
		{
			// (hwrt5v: why the haze is what it is: the lamps tracked, the adaptive threshold, the sun's share)
			float thr = 0.0f;
			std::memcpy(&thr, &cur[13], sizeof(thr));
			const double lights = static_cast<double>(d[1]) / fr;
			const char*  why    = d[0] == 0                         ? "pass not run (no frames: menu / movie / no TLAS yet)"
			                      : lights >= 0.5                   ? "on: lamp halos"
			                      : g_gr.sun_held > 0.05f           ? "on: sun haze only (no lamp above the threshold)"
			                                                        : "off: no lamp above the threshold and no sun";
			std::printf("HWRTVOL: state: %s; lamps %.2f a frame (%u slots), light threshold %.2f (exposed, adaptive; %.2f lit), "
			            "sun weight %.2f (presence; slow scale %.2f, feed %.2f..%.2f in 10 s: RT shadows' fade x overcast)\n",
			            why, lights, g_gr.hw || level >= 3 ? 8u : level == 2 ? 4u : 2u, static_cast<double>(thr),
			            static_cast<double>(EnvFloat("KYTY_GODRAYS_LIGHT", 3.0f, 0.5f, 100.0f)), static_cast<double>(g_gr.sun_held),
			            static_cast<double>(g_gr.sun_scale), static_cast<double>(g_gr.sun_feed_max > 0.0f ? g_gr.sun_feed_min : 0.0f),
			            static_cast<double>(g_gr.sun_feed_max));
			g_gr.sun_feed_min = 1.0f, g_gr.sun_feed_max = 0.0f;
		}
		std::fflush(stdout);
		g_gr.hw_traced = g_gr.hw_stale = g_gr.hw_none = 0;
		g_gr.path_switches = g_gr.hw_held = g_gr.sun_ons = g_gr.sun_offs = g_gr.resets = 0;
	}
}

} // namespace

int GodraysLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_GODRAYS");
		// (hwrt3v: KYTY_GODRAYS unset and KYTY_HWRT_VOLUMETRIC set: the pass at the latter's level)
		if ((value == nullptr || *value == '\0') && HwVolLevel() > 0) return HwVolLevel();
		return value != nullptr ? std::clamp(std::atoi(value), 0, 3) : 0;
	}();
	return level;
}
bool GodraysWanted() { return GodraysLevel() > 0; }

void GodraysSetSun(const std::array<double, 3>& view_dir, float weight) {
	if (!GodraysWanted()) return;
	std::lock_guard lock(g_gr_sun.mutex);
	g_gr_sun.dir    = {static_cast<float>(view_dir[0]), static_cast<float>(view_dir[1]), static_cast<float>(view_dir[2])};
	g_gr_sun.weight = std::isfinite(weight) ? std::clamp(weight, 0.0f, 1.0f) : 0.0f;
	g_gr_sun.when   = std::chrono::steady_clock::now();
}

bool GodraysApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int level = GodraysLevel();
	if (level == 0) return false;
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr) return false;
	if (VideoOut::MoviePlaying()) {
		g_gr.history_valid = false;
		return false;
	}
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || color.extent.width < 16 || color.extent.height < 16 ||
	    inputs.depth.extent != color.extent)
		return false;
	const auto& m = inputs.view_to_clip;
	if (!(m[0] != 0.0f && m[5] != 0.0f && m[11] != 0.0f) || !std::all_of(m.begin(), m.end(), [](float v) { return std::isfinite(v); }))
		return false;
	if (!EnsurePipeline(*graphics) || !EnsureImages(*graphics, color.extent)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("GODRAYS: off (could not create its pipeline or images)\n");
			std::fflush(stdout);
		}
		return false;
	}
	FreeRetired(*graphics);
	ReadTiming(*graphics);

	static const bool row0_up = [] {
		const char* y = std::getenv("KYTY_GODRAYS_FLIPY");
		return y == nullptr || *y == '\0' || *y != '0';
	}();
	const float strength = EnvFloat("KYTY_GODRAYS_STRENGTH", 1.0f, 0.0f, 3.0f);

	float sun_target = 0.0f;
	{
		std::lock_guard lock(g_gr_sun.mutex);
		if (std::chrono::steady_clock::now() - g_gr_sun.when < std::chrono::milliseconds(250)) g_gr.sun_dir = g_gr_sun.dir, sun_target = g_gr_sun.weight;
	}
	// The sun's uv (a direction: w = z * M[11]) and how far on screen it is; smoothed over ~0.5 s.
	const auto now = std::chrono::steady_clock::now();
	const float dt = g_gr.last_frame.time_since_epoch().count() == 0
	                     ? 0.016f
	                     : std::clamp(std::chrono::duration<float>(now - g_gr.last_frame).count(), 0.0f, 0.1f);
	// (10-25: a sun gone stale (no fresh one in 250 ms) fades out over ~0.8 s from the last one given instead of popping off,
	// and a returning one fades back in over ~0.4 s: short gaps in the RT sun's feed no longer flicker the rays.)
	// (hwrt3v-2, KYTY_HWRT_VOLUMETRIC: 2.5 s out, 1.2 s in: an indoor sun feed that comes and goes no longer pulses the haze)
	const float sun_out = g_gr.hw ? 2.5f : 0.8f, sun_in = g_gr.hw ? 1.2f : 0.4f;
	if (g_gr.hw) {
		// (hwrt5v-2: the feed's weight is RT shadows' fade x its overcast factor, and that factor (0.5..1) is measured from
		// the pixels on screen: it moved with every camera turn and the sun haze with it. Here the presence (any weight)
		// drives the fade and the factor only a slow (~8 s) scale, so turning the camera no longer dims the shafts.)
		if (sun_target > 0.0f) {
			g_gr.sun_feed_min = std::min(g_gr.sun_feed_min, sun_target), g_gr.sun_feed_max = std::max(g_gr.sun_feed_max, sun_target);
			g_gr.sun_scale = g_gr.sun_scale < 0.0f ? sun_target : g_gr.sun_scale + (sun_target - g_gr.sun_scale) * std::min(dt / 8.0f, 1.0f);
			sun_target = 1.0f;
		}
	}
	g_gr.sun_held = std::clamp(g_gr.sun_held + std::clamp(sun_target - g_gr.sun_held, -dt / sun_out, dt / sun_in), 0.0f, 1.0f);
	if (g_gr.hw) {
		if (!g_gr.sun_on && g_gr.sun_held > 0.65f) g_gr.sun_on = true, ++g_gr.sun_ons;
		if (g_gr.sun_on && g_gr.sun_held < 0.35f) g_gr.sun_on = false, ++g_gr.sun_offs;
	}
	const std::array<float, 3> sun   = g_gr.sun_dir;
	const float                sun_w = g_gr.sun_held * (g_gr.hw ? std::clamp(g_gr.sun_scale < 0.0f ? 1.0f : g_gr.sun_scale, 0.5f, 1.0f) : 1.0f);
	const bool have_sun = sun_w > 0.0f;
	// (a gap of more than 0.25 s: menus, loading, frames without the pass: the history is stale, restart it)
	if (g_gr.last_frame.time_since_epoch().count() != 0 && now - g_gr.last_frame > std::chrono::milliseconds(250))
		g_gr.history_valid = false;
	g_gr.last_frame = now;
	float su = 0.5f, sv = 0.5f, target = 0.0f;
	if (have_sun) {
		const float w = sun[2] * m[11];
		if (w > 1e-4f) {
			const float nx = (sun[0] * m[0] + sun[2] * m[8]) / w;
			const float ny = (sun[1] * m[5] + sun[2] * m[9]) / w;
			su             = nx * 0.5f + 0.5f;
			sv             = (row0_up ? -ny : ny) * 0.5f + 0.5f;
			const float out = std::max({-su, su - 1.0f, -sv, sv - 1.0f, 0.0f});
			target          = std::clamp(1.0f - out / 0.25f, 0.0f, 1.0f);
		}
	}
	g_gr.on_screen = std::clamp(g_gr.on_screen + std::clamp(target - g_gr.on_screen, -dt / 0.5f, dt / 0.5f), 0.0f, 1.0f);
	Log(level, have_sun);
	{
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("GODRAYS: on (level %d, %ux%u, volume %ux%u, strength %g, sun %s, exposure %s)\n", level, color.extent.width,
			            color.extent.height, g_gr.qextent.width, g_gr.qextent.height, strength,
			            have_sun ? "from RT shadows" : "unknown yet (KYTY_RT_SHADOWS gives it; lights only until then)",
			            inputs.exposure_buffer != nullptr ? "the game's" : "1.0");
			std::fflush(stdout);
		}
	}

	// (rt3fx) KYTY_HWRT_VOLUMETRIC: no lamp / fire halos unless KYTY_HWRT_VOL_HALOS=1. They were tracked from the frame's
	// brightest pixels (flames: particles whose brightness flickers, and that pass the TLAS check on their brazier), so
	// they swelled and faded with the flame - the halos at fires; the game draws its own glow there. The sun shafts stay.
	static const bool hw_halos = [] {
		const char* v = std::getenv("KYTY_HWRT_VOL_HALOS");
		return v != nullptr && v[0] == '1';
	}();
	const int   slots  = g_gr.hw ? (hw_halos ? 8 : 0) : level >= 3 ? 8 : level == 2 ? 4 : 2; // (hwrt5v-2: world-space lamps: 8 slots)
	const int   taps   = level >= 3 ? 64 : level == 2 ? 48 : 32;
	const int   hwl    = g_gr.hw ? HwVolLevel() : 0;
	const int   steps  = hwl >= 3 ? 24 : hwl == 2 ? 16 : hwl == 1 ? 12 : level >= 3 ? 16 : level == 2 ? 12 : 0;
	// (10-25: Bajo / Medio raised a little: 0.10 / 0.18 cap, 0.5 / 0.75 halos, 0.45 / 0.7 sun before)
	const float cap    = (level >= 3 ? 0.28f : level == 2 ? 0.22f : 0.14f) * strength;
	const float lstr   = (level >= 3 ? 1.0f : level == 2 ? 0.95f : 0.7f) * strength;
	const float sstr   = (level >= 3 ? 1.0f : level == 2 ? 0.8f : 0.55f) * strength; // (gr4: 0.85 / 0.6 in gr3: noisy shafts)
	static const bool debug = [] {
		const char* v = std::getenv("KYTY_GODRAYS_DEBUG");
		return v != nullptr && *v == '1';
	}();

	GodraysPush push {};
	push.proj[0] = m[0], push.proj[1] = m[5], push.proj[2] = m[8], push.proj[3] = m[9];
	push.depth_map[0] = m[10], push.depth_map[1] = m[14], push.depth_map[2] = m[11], push.depth_map[3] = m[15];
	push.size[0] = static_cast<float>(color.extent.width), push.size[1] = static_cast<float>(color.extent.height);
	push.size[2] = sstr, push.size[3] = cap;
	const float sun_fade = sun_w * sun_w * (3.0f - 2.0f * sun_w);
	push.sun[0] = sun[0], push.sun[1] = sun[1], push.sun[2] = sun[2], push.sun[3] = have_sun ? sun_fade : 0.0f;
	push.sun_uv[0] = su, push.sun_uv[1] = sv, push.sun_uv[2] = have_sun ? g_gr.on_screen : 0.0f, push.sun_uv[3] = static_cast<float>(taps);
	push.motion[0] = inputs.motion_scale_x, push.motion[1] = inputs.motion_scale_y;
	push.motion[2] = (g_gr.history_valid && !inputs.reset && inputs.motion.view != nullptr) ? 1.0f : 0.0f;
	if (g_gr.hw && push.motion[2] == 0.0f) ++g_gr.resets;
	push.motion[3] = static_cast<float>(steps);
	const bool game_exposure = inputs.exposure_buffer != nullptr && inputs.exposure_range >= sizeof(float);
	push.misc[1] = static_cast<int32_t>(g_gr.frame & 0x7fffffff);
	push.misc[2] = (row0_up ? 1 : 0) | (debug ? 2 : 0);
	push.misc[3] = game_exposure ? static_cast<int32_t>(inputs.exposure_index) : static_cast<int32_t>(ExposureIdx);
	push.extra[0] = static_cast<float>(g_gr.qextent.width), push.extra[1] = static_cast<float>(g_gr.qextent.height);
	push.extra[2] = lstr;
	push.extra[3] = level <= 2 ? 1.0f : 0.0f; // (the visibility boost)

	// (KYTY_HWRT_VOLUMETRIC: the TLAS of this frame's build, an older one while this frame built none; none: the
	// depth-buffer march, hw.x = 0)
	GodraysPushHw hpush {};
	vk::AccelerationStructureKHR hw_tlas = nullptr;
	if (g_gr.hw) {
		bool                  built = false;
		std::array<float, 16> vp {};
		hw_tlas = HwScene::VolumetricTlas(&built, &vp);
		if (hw_tlas == nullptr) {
			// (no TLAS ever built yet: the layout has the binding, so no pass until the first one; start-up only)
			++g_gr.hw_none;
			g_gr.history_valid = false;
			return false;
		}
		// (hwrt3v-2: a frame without the camera keeps the last good rows instead of falling back to the depth march: the
		// two methods never alternate frame to frame once traced)
		std::array<float, 12> rows {};
		if (ViewToWorld(m, vp, rows.data())) {
			g_gr.hw_w = rows, g_gr.hw_have_w = true;
			++(built ? g_gr.hw_traced : g_gr.hw_stale);
		} else if (g_gr.hw_have_w) {
			++g_gr.hw_held;
		} else {
			++g_gr.hw_none;
		}
		const bool traced = g_gr.hw_have_w;
		if (traced) std::memcpy(hpush.w, g_gr.hw_w.data(), sizeof(hpush.w)), hpush.hw[0] = 1.0f;
		if (traced != g_gr.hw_last_path) ++g_gr.path_switches, g_gr.hw_last_path = traced;
		hpush.hw[1] = EnvFloat("KYTY_HWRT_VOLUMETRIC_RANGE", 250.0f, 10.0f, 2000.0f);
		hpush.hw[2] = EnvFloat("KYTY_HWRT_VOLUMETRIC_DIST", hwl >= 3 ? 80.0f : 60.0f, 10.0f, 300.0f);
		hpush.hw[3] = static_cast<float>(hwl);
	}
	const auto entry = static_cast<uint32_t>(g_gr.written % 8);
	const bool timed = g_gr.pool != nullptr && g_gr.written < g_gr.read + 8;
	if (timed) {
		command.resetQueryPool(g_gr.pool, entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_gr.pool, entry * 2);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite);
	if (!g_gr.ready) {
		ToGeneral(command, g_gr.out.image), ToGeneral(command, g_gr.vol[0].image), ToGeneral(command, g_gr.vol[1].image);
		ToGeneral(command, g_gr.raw.image), ToGeneral(command, g_gr.filt.image);
		g_gr.ready = true;
	}
	if (!g_gr.cleared) {
		g_gr.cleared = true;
		command.fillBuffer(vk::Buffer(g_gr.frame_buf), 0, VK_WHOLE_SIZE, 0);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer,
		        vk::AccessFlagBits2::eTransferWrite);
	}
	{
		std::array<float, 4> hdr {1.0f, EnvFloat("KYTY_GODRAYS_LIGHT", 3.0f, 0.5f, 100.0f), static_cast<float>(slots),
		                          EnvFloat("KYTY_GODRAYS_LIGHT_RADIUS", 1.2f, 0.2f, 10.0f)};
		command.updateBuffer(vk::Buffer(g_gr.frame_buf), 0, sizeof(hdr), hdr.data());
	}
	Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_gr.pipeline);

	const auto     gen = vk::ImageLayout::eGeneral;
	const uint32_t cur = static_cast<uint32_t>(g_gr.frame & 1u);
	vk::DescriptorImageInfo infos[Bindings] {};
	infos[0] = {g_gr.linear, color.view, color.layout};
	infos[1] = {g_gr.point, inputs.depth.view, inputs.depth.layout};
	infos[2] = inputs.motion.view != nullptr ? vk::DescriptorImageInfo {g_gr.point, inputs.motion.view, inputs.motion.layout}
	                                         : vk::DescriptorImageInfo {g_gr.point, g_gr.vol[cur ^ 1u].view, gen};
	infos[3] = {g_gr.linear, g_gr.vol[cur ^ 1u].view, gen};
	infos[4] = {nullptr, g_gr.vol[cur].view, gen};
	infos[5] = {nullptr, g_gr.out.view, gen};
	infos[8] = {g_gr.point, g_gr.filt.view, gen}; // (lights / track / volume: not read)
	infos[9] = {nullptr, g_gr.raw.view, gen};     // volume: the raw scatter
	const vk::DescriptorBufferInfo frame_info {vk::Buffer(g_gr.frame_buf), 0, VK_WHOLE_SIZE};
	const vk::DescriptorBufferInfo exposure_info =
	    game_exposure ? vk::DescriptorBufferInfo {inputs.exposure_buffer, inputs.exposure_base, inputs.exposure_range} : frame_info;
	// (binding 10, KYTY_HWRT_VOLUMETRIC: the TLAS; with none this frame the shader never touches it, yet the layout
	// has the binding: never null here, see above)
	vk::WriteDescriptorSetAccelerationStructureKHR as_write {};
	as_write.accelerationStructureCount = 1;
	as_write.pAccelerationStructures    = &hw_tlas;
	const uint32_t         nb = g_gr.hw ? Bindings + 1 : Bindings;
	vk::WriteDescriptorSet writes[Bindings + 1] {};
	const auto push_set = [&] {
		for (uint32_t i = 0; i < nb; ++i) {
			writes[i]                 = vk::WriteDescriptorSet {};
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = DescType(i);
			if (i == 10) writes[i].pNext = &as_write;
			else if (i == 6) writes[i].pBufferInfo = &frame_info;
			else if (i == 7) writes[i].pBufferInfo = &exposure_info;
			else writes[i].pImageInfo = &infos[i];
		}
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_gr.layout, 0, nb, writes);
	};
	push_set();
	const auto run = [&](int mode, uint32_t gx, uint32_t gy) {
		push.misc[0] = mode;
		if (g_gr.hw) {
			hpush.base = push;
			command.pushConstants(g_gr.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(hpush), &hpush);
		} else {
			command.pushConstants(g_gr.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		}
		command.dispatch(gx, gy, 1);
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	};
	run(0, 16, 9);
	run(1, 1, 1);
	run(2, (g_gr.qextent.width + 7) / 8, (g_gr.qextent.height + 7) / 8);
	const uint32_t qx = (g_gr.qextent.width + 7) / 8, qy = (g_gr.qextent.height + 7) / 8;
	// Temporal: raw (8) + last frame's history (3) -> this frame's (4).
	infos[8] = {g_gr.point, g_gr.raw.view, gen};
	infos[9] = {nullptr, g_gr.filt.view, gen}; // (not written)
	push_set();
	run(4, qx, qy);
	// Spatial: this frame's history (8) -> filtered (9).
	infos[8] = {g_gr.point, g_gr.vol[cur].view, gen};
	infos[9] = {nullptr, g_gr.filt.view, gen};
	infos[4] = {nullptr, g_gr.raw.view, gen}; // (not written)
	push_set();
	run(5, qx, qy);
	// The composite reads the filtered volume (binding 3 becomes it; 4 and 9 are not used there).
	infos[3] = {g_gr.linear, g_gr.filt.view, gen};
	infos[4] = {nullptr, g_gr.vol[cur ^ 1u].view, gen};
	infos[8] = {g_gr.point, g_gr.vol[cur].view, gen};
	infos[9] = {nullptr, g_gr.raw.view, gen};
	push_set();
	run(3, (color.extent.width + 7) / 8, (color.extent.height + 7) / 8);
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eMemoryRead);
	{
		vk::BufferCopy region {StatsWord * 4, 0, StatsCount * 4};
		command.copyBuffer(vk::Buffer(g_gr.frame_buf), vk::Buffer(g_gr.read_buf), 1, &region);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
		        vk::AccessFlagBits2::eHostRead);
	}
	if (timed) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_gr.pool, entry * 2 + 1);
		++g_gr.written;
	}
	++g_gr.frame;
	g_gr.history_valid  = true;
	inputs.color.image  = vk::Image(g_gr.out.image);
	inputs.color.view   = g_gr.out.view;
	inputs.color.layout = gen;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc |
	                      vk::ImageUsageFlagBits::eStorage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}

} // namespace Libs::Graphics::FrameGen
