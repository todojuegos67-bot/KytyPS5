// KYTY_RT_REFLECTIONS=0|1|2|3|4 ("RT reflections": off / low / medium / high / ultra; rt-reflections.comp): screen-space ray-traced
// reflections with denoising for Demon's Souls' PS5 Pro path, recorded in FrameGen::Upscale right before the upscaler
// (DLSS / XeSS / FSR), which then reads our composited copy of the game's color. Only there: it is the one point where
// the game hands over its lit HDR color before anti-aliasing together with depth, motion vectors and the projection
// (without KYTY_UPSCALER the game's own TAA runs and nothing here does).
//
// Per frame (all compute, one pipeline, push descriptors):
//   hi-z:      the closest-distance pyramid of the depth (mip 0 at the render size, then 2x2 min down to 1x1)
//   trace:     one GGX VNDF-sampled ray per trace pixel (low / medium / high: half resolution, ultra: full; low and
//              ultra: checkerboard, half the trace pixels a frame), hierarchical-Z march (LevelCfg below)
//   temporal:  reprojection by motion vector + reflection parallax, disocclusion rejection, neighbourhood clamp
//   a-trous:   2 / 3 / 4 edge-aware passes (distance, normal, roughness, luminance variance)
//   composite: bilateral upsample, Fresnel x mask x confidence convex mix over the game's color
// Surface data: the packed R32_UINT G-buffer copy KYTY_SSR's hooks take before the deferred lighting (frame-gen.cpp),
// decoded after a calibration against depth normals (normal: statistically identified; roughness: the normal word's free
// byte, a guess, so weighed by the up-facing / flat / dark heuristic unless set by KYTY_RT_REFLECTIONS_ROUGH); without
// it, depth normals and the heuristic. The game's exposure (its upscaler path's slot 5 buffer) is read so the filters
// work on display-referred values.
//
// Material field: picked by the calibration (the byte that separates puddle-looking pixels most), or
// KYTY_RT_REFLECTIONS_ROUGH; KYTY_RT_REFLECTIONS_WET=lo:hi (wet below lo, dry from hi, roughness units),
// KYTY_RT_REFLECTIONS_MATERIAL (1.0: its scale), KYTY_RT_REFLECTIONS_DEPTH=<n> (the lighting's n-th depth binding as the
// pre-water depth), KYTY_RT_REFLECTIONS_GUESS (0.2: the puddle guess without a material field).
// KYTY_RT_REFLECTIONS_STRENGTH (1.0, 0..3): the reflection's share scale. KYTY_RT_REFLECTIONS_MIN (0.35, 0..0.8): the
// minimum reflectance of wet surfaces (water films, puddles: far above a dry dielectric's 4 %). KYTY_RT_REFLECTIONS_NORMAL
// = <layer>:<layout 0..12>[w][n] and KYTY_RT_REFLECTIONS_ROUGH = <layer>:<shift>:<bits>[g] override the calibration
// (layouts: the shader's DecodeNormal), KYTY_RT_REFLECTIONS_FLIPY=1|0 forces the screen y convention (1: the render's row
// 0 is NDC y +1, the default). Every 10 s one line: the share of pixels the reflections change, the mean weight, the
// hit rate, the G-buffer copy's frames, the GPU time (timestamps, always on; KYTY_RT_REFLECTIONS_TIMING is not needed).
//
// Reflectivity (rt-reflections.comp SurfaceAt): water (the final depth nearer than the depth the deferred lighting read,
// which frame-gen.cpp copies with the G-buffer), a trusted material field (KYTY_RT_REFLECTIONS_ROUGH), and a faint
// puddle guess (KYTY_RT_REFLECTIONS_GUESS, 0.2: only its Fresnel). The minimum reflectance applies to water and smooth
// material only: dry dirt and stone stay as the game draws them (the 10-24 build made every up-facing floor shine).
//
// Why the first version showed nothing (log of 2026-10-24): the G-buffer copy was refused nearly every frame (the
// G-buffer draws take the table path, which did not refresh the G-buffer pass's flip, frame-gen.cpp OnSsrDraw); the
// depth reconstruction mirrored the screen vertically against the game's (GNM) viewport, so the depth normals
// disagreed with the G-buffer's (calibration 0.654) and with world up (floors read as ceilings: the wet-surface mask was
// zero on floors); the mask needed flat AND plain AND up-facing at once (textured wet stone failed "plain"); and the
// composite weight was a dry dielectric's Fresnel (~0.02 seen from above).
#include "local/rt-reflections.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/rt_reflections_spv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace Libs::Graphics::FrameGen {
namespace {

constexpr uint32_t Bindings      = 21;
constexpr uint32_t HeaderBytes   = 160;           // view_rot[3], clip_prev[4], up_view, fparams, mparams
// calib[] (after the header; the shader's CALIB_* / STATS): samples, normal sums per layer x layout x space x screen
// y convention, byte statistics, floor votes, then the cumulative stats counters (never cleared).
constexpr uint32_t NLayouts      = 13;
constexpr uint32_t CalibNormal   = 1;
constexpr uint32_t CalibBytes    = CalibNormal + 4 * NLayouts * 4;
constexpr uint32_t CalibFloors   = CalibBytes + 16 * 3;
constexpr uint32_t CalibWet      = CalibFloors + 2; // puddle guess: count + 16 byte sums; water: the same
constexpr uint32_t CalibHist     = 300;             // 16 byte fields x 16 bins
constexpr uint32_t StatsIndex    = 560;
constexpr uint32_t StatsCount    = 10;
constexpr uint32_t FrameBytes    = 4096;          // header + calibration + stats
constexpr uint32_t ExposureIndex = 128 / 4;       // fparams.x (1.0) when the game gives no exposure
constexpr uint32_t CalibRuns     = 24;
constexpr uint32_t MaxMips       = 12;
static_assert(CalibWet + 34 <= CalibHist && CalibHist + 256 <= StatsIndex);
static_assert(HeaderBytes + (StatsIndex + StatsCount) * 4 <= FrameBytes);

struct RtPush {
	float   proj[4];
	float   depth_map[4];
	float   size[4];
	float   params[4];
	float   motion[4];
	int32_t misc[4];
	int32_t dec[4];
	int32_t misc2[4];
};
static_assert(sizeof(RtPush) == 128);

struct RtImage {
	VkImage                             image  = VK_NULL_HANDLE;
	VmaAllocation                       memory = nullptr;
	vk::ImageView                       view   = nullptr; // every mip
	std::array<vk::ImageView, MaxMips>  mips {};          // one mip each (storage; hi-z only)
	uint32_t                            levels = 1;
	vk::Extent2D                        extent {};
	vk::Format                          format = vk::Format::eUndefined;
	vk::ImageUsageFlags                 usage {};
	bool                                ready = false;    // in the general layout
};

struct Retired {
	RtImage  image;
	uint64_t frame = 0;
};

struct RtPass {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             point = nullptr, linear = nullptr;
	bool                    failed = false; // (the pipeline could not be made: off for the session)
	bool                    images_failed = false;
	vk::Extent2D            render {}, trace {};
	int                     level = 0;
	RtImage                 hiz, rad, aux, tmp_a, tmp_b, out;
	RtImage                 gb[2], hist[2], meta[2];
	RtImage                 dummy; // 1x1 R32_UINT stand-in for absent G-buffer layers
	std::vector<Retired>    retired;
	VkBuffer                frame_buf = VK_NULL_HANDLE; // device-local: the shader's header, sums and counters
	VmaAllocation           frame_dev_mem = nullptr;
	VkBuffer                read_buf = VK_NULL_HANDLE;  // host: its copy after every frame (what the CPU reads)
	VmaAllocation           frame_mem = nullptr;
	// The material field (wet below lo, dry from hi; scale): from KYTY_RT_REFLECTIONS_ROUGH / _WET or the calibration.
	float                   mat_lo = 0.15f, mat_hi = 0.30f, mat_scale = 1.0f;
	bool                    mat_auto = false, mat_forced = false;
	float                   wet_cap = 1.0f;
	const int32_t*          mapped    = nullptr;
	bool                    frame_cleared = false;
	uint64_t                frame = 0;
	bool                    history_valid = false;
	// Calibration (up to 3 attempts, the first once the G-buffer copy has been there 60 frames running) and the
	// chosen decode; the screen's y convention (true: the render's row 0 is NDC y +1, the GNM viewport).
	uint32_t                calib_runs = 0, calib_wait = 0, calib_tick = 0, calib_attempts = 0, calib_packed = 0;
	bool                    calib_done = false, calib_active = false, calib_clear = false, floors_done = false;
	uint64_t                calib_next = 0, gb_streak = 0;
	int32_t                 dec_normal = -1, dec_rough = -1;
	bool                    rough_trusted = false;
	bool                    row0_up = true, row0_forced = false;
	bool                    have_water = false; // (this frame: the G-buffer-time depth is bound)
	// The periodic line (every 10 s): the stats counters' last values and the sums since the last line.
	std::array<uint32_t, StatsCount> stat_last {};
	std::array<uint64_t, StatsCount> stat_sum {};
	uint64_t                stat_frames = 0, stat_gbuf_frames = 0, stat_copies0 = 0;
	std::chrono::steady_clock::time_point stat_time = std::chrono::steady_clock::now();
	// The GPU time (timestamps, always: two a frame).
	// Per pass: a timestamp before the first and after each pass; ring of 8 frames x 16 queries.
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms = 0.0;
	uint32_t                samples = 0;
	std::array<std::array<uint8_t, 16>, 8> stage_of {}; // the pass each interval timed (Stage), per ring entry
	std::array<uint32_t, 8>               stamps_in {};  // timestamps written per ring entry
	std::array<double, 9>                 stage_ms {};   // sums since the last line
	std::array<uint32_t, 9>               stage_frames {};
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
} g_rt;

float EnvFloat(const char* name, float fallback, float lo, float hi) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') return fallback;
	const float v = static_cast<float>(std::atof(value));
	return std::isfinite(v) ? std::clamp(v, lo, hi) : fallback;
}
// "<layer>:<shift>:<bits>[g]" -> the shader's field code (-1: unset or malformed).
int32_t FieldEnv(const char* name) {
	const char* value = std::getenv(name);
	if (value == nullptr) return -1;
	int  layer = -1, shift = -1, bits = -1;
	char flag  = 0;
	if (std::sscanf(value, "%d:%d:%d%c", &layer, &shift, &bits, &flag) < 3) return -1;
	if (layer < 0 || layer > 3 || shift < 0 || bits < 1 || bits > 16 || shift + bits > 32) return -1;
	return layer | (shift << 2) | (bits << 7) | (flag == 'g' ? 4096 : 0);
}
// "<layer>:<layout>[w][n]" -> the shader's normal code (-1: unset or malformed).
int32_t NormalEnv(const char* name) {
	const char* value = std::getenv(name);
	if (value == nullptr) return -1;
	int layer = -1, layout = -1;
	if (std::sscanf(value, "%d:%d", &layer, &layout) < 2 || layer < 0 || layer > 3 || layout < 0 ||
	    layout >= static_cast<int>(NLayouts))
		return -1;
	const std::string v = value;
	return layer | (layout << 2) | (v.find('w') != std::string::npos ? 64 : 0) | (v.find('n') != std::string::npos ? 128 : 0);
}

void FreeImage(GraphicContext& graphics, RtImage& img) {
	if (img.view != nullptr) graphics.device.destroyImageView(img.view, nullptr);
	for (auto& v: img.mips)
		if (v != nullptr && v != img.view) graphics.device.destroyImageView(v, nullptr);
	if (img.image != VK_NULL_HANDLE) vmaDestroyImage(graphics.allocator, img.image, img.memory);
	img = RtImage {};
}
// (Images replaced at a size change may still be read by frames in flight: destroyed 16 frames later.)
void Retire(RtImage& img) {
	if (img.image != VK_NULL_HANDLE) g_rt.retired.push_back({img, g_rt.frame});
	img = RtImage {};
}
void FreeRetired(GraphicContext& graphics) {
	auto& r = g_rt.retired;
	for (size_t i = 0; i < r.size();) {
		if (g_rt.frame >= r[i].frame + 16) {
			FreeImage(graphics, r[i].image);
			r[i] = r.back();
			r.pop_back();
		} else {
			++i;
		}
	}
}

bool MakeImage(GraphicContext& graphics, RtImage& img, vk::Format format, vk::Extent2D extent, uint32_t levels, bool storage) {
	VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create.imageType   = VK_IMAGE_TYPE_2D;
	create.format      = static_cast<VkFormat>(format);
	create.extent      = {extent.width, extent.height, 1};
	create.mipLevels   = levels;
	create.arrayLayers = 1;
	create.samples     = VK_SAMPLE_COUNT_1_BIT;
	create.tiling      = VK_IMAGE_TILING_OPTIMAL;
	create.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
	               (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0u);
	create.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
	create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO;
	if (vmaCreateImage(graphics.allocator, &create, &allocate, &img.image, &img.memory, nullptr) != VK_SUCCESS) {
		img.image = VK_NULL_HANDLE;
		return false;
	}
	img.levels = levels;
	img.extent = extent;
	img.format = format;
	img.usage  = static_cast<vk::ImageUsageFlags>(create.usage);
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = vk::Image(img.image);
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = format;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, levels, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &img.view) != vk::Result::eSuccess) {
		img.view = nullptr;
		return false;
	}
	if (levels == 1) {
		img.mips[0] = img.view;
		return true;
	}
	for (uint32_t m = 0; m < levels && m < MaxMips; ++m) {
		view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, m, 1, 0, 1};
		if (graphics.device.createImageView(&view_info, nullptr, &img.mips[m]) != vk::Result::eSuccess) {
			img.mips[m] = nullptr;
			return false;
		}
	}
	return true;
}
vk::Extent2D MipExtent(const RtImage& img, uint32_t m) {
	return {std::max(1u, img.extent.width >> m), std::max(1u, img.extent.height >> m)};
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

std::array<RtImage*, 13> Sized() {
	return {&g_rt.hiz, &g_rt.rad, &g_rt.aux, &g_rt.tmp_a, &g_rt.tmp_b, &g_rt.out, &g_rt.gb[0], &g_rt.gb[1], &g_rt.hist[0],
	        &g_rt.hist[1], &g_rt.meta[0], &g_rt.meta[1], &g_rt.dummy};
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_rt.pipeline != nullptr) return true;
	if (g_rt.failed) return false;
	g_rt.failed       = true;
	const auto device = graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, Bindings> bindings {};
	for (uint32_t i = 0; i < Bindings; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = (i == 18 || i == 19)                       ? vk::DescriptorType::eStorageBuffer
		                              : (i == 4 || i == 11 || i == 12 || i == 13) ? vk::DescriptorType::eStorageImage
		                                                                          : vk::DescriptorType::eCombinedImageSampler;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = Bindings;
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_rt.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(RtPush)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_rt.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_rt.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(RT_REFLECTIONS_SPV);
	module_info.pCode       = RT_REFLECTIONS_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_rt.layout;
	const auto created         = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_rt.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_rt.pipeline = nullptr;
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
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g_rt.point : &g_rt.linear) != vk::Result::eSuccess) return false;
	}
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = FrameBytes;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VmaAllocationCreateInfo dev {};
	dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_rt.frame_buf, &g_rt.frame_dev_mem, nullptr) != VK_SUCCESS) {
		g_rt.frame_buf = VK_NULL_HANDLE;
		return false;
	}
	buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_rt.read_buf, &g_rt.frame_mem, &mapped) != VK_SUCCESS) {
		g_rt.read_buf = VK_NULL_HANDLE;
		return false;
	}
	g_rt.mapped = static_cast<const int32_t*>(mapped.pMappedData);
	if (!MakeImage(graphics, g_rt.dummy, vk::Format::eR32Uint, {1, 1}, 1, false)) return false;
	{
		vk::QueryPoolCreateInfo query {};
		query.queryType  = vk::QueryType::eTimestamp;
		query.queryCount = 8 * 16;
		if (device.createQueryPool(&query, nullptr, &g_rt.pool) != vk::Result::eSuccess) g_rt.pool = nullptr;
		g_rt.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	}
	g_rt.failed = false;
	return true;
}

// The quality levels (KYTY_RT_REFLECTIONS 1..4). Costs measured with the first version on an RTX 5080 at a 1080p render:
// full-resolution trace, 72 iterations, 3 a-trous passes, every pixel = 8.5-10 ms; most of it the march, then the wet-
// surface mask evaluated twice per pixel (trace and composite) and the full-resolution filters.
struct LevelCfg {
	const char* name;
	bool        full;     // trace at the render resolution (else half: a quarter of the pixels)
	bool        checker;  // half of the trace pixels march each frame (the temporal pass fills the others)
	int         iters;    // hi-z iterations (rough rays: half)
	float       cutoff;   // roughness above which nothing reflects
	int         passes;   // a-trous passes
	float       max_len;  // view-space ray length
	float       cap;      // the longest ray on screen, x the render height
	float       alpha;    // temporal minimum alpha scale
};
constexpr LevelCfg Levels[5] = {
    {"off", false, false, 0, 0.0f, 0, 0.0f, 0.0f, 1.0f},
    {"low", false, true, 24, 0.45f, 1, 200.0f, 0.30f, 1.0f},
    {"medium", false, false, 32, 0.55f, 2, 300.0f, 0.45f, 1.0f},
    {"high", false, false, 48, 0.60f, 3, 800.0f, 0.75f, 0.9f}, // (10-24 far: 400 m / 0.65)
    {"ultra", true, true, 72, 0.65f, 3, 1600.0f, 1.00f, 0.8f}, // (10-24 far: 64 / 600 m / 0.90)
};
const LevelCfg& Cfg(int level) { return Levels[std::clamp(level, 0, 4)]; }

bool EnsureImages(GraphicContext& graphics, vk::Extent2D render, int level) {
	if (g_rt.out.image != VK_NULL_HANDLE && g_rt.render == render && g_rt.level == level) return true;
	for (auto* img: Sized())
		if (img != &g_rt.dummy) Retire(*img);
	const vk::Extent2D trace = Cfg(level).full ? render : vk::Extent2D {std::max(1u, (render.width + 1) / 2), std::max(1u, (render.height + 1) / 2)};
	uint32_t levels = 1;
	while (levels < MaxMips && ((render.width >> levels) > 0 || (render.height >> levels) > 0)) ++levels;
	const auto f16 = vk::Format::eR16G16B16A16Sfloat;
	bool ok = MakeImage(graphics, g_rt.hiz, vk::Format::eR32Sfloat, render, levels, true) &&
	          MakeImage(graphics, g_rt.out, f16, render, 1, true);
	for (auto* img: {&g_rt.rad, &g_rt.aux, &g_rt.tmp_a, &g_rt.tmp_b, &g_rt.gb[0], &g_rt.gb[1], &g_rt.hist[0], &g_rt.hist[1],
	                 &g_rt.meta[0], &g_rt.meta[1]})
		ok = ok && MakeImage(graphics, *img, f16, trace, 1, true);
	if (!ok) {
		for (auto* img: Sized())
			if (img != &g_rt.dummy) Retire(*img);
		g_rt.images_failed = true;
		return false;
	}
	g_rt.render        = render;
	g_rt.trace         = trace;
	g_rt.level         = level;
	g_rt.history_valid = false;
	return true;
}

// Every image not yet in the general layout moved there (contents undefined: the history is invalid anyway).
void PrepareImages(vk::CommandBuffer command) {
	std::vector<vk::ImageMemoryBarrier2> barriers;
	bool                                 dummy = false;
	for (auto* img: Sized()) {
		if (img->ready || img->image == VK_NULL_HANDLE) continue;
		vk::ImageMemoryBarrier2 b {};
		b.srcStageMask        = vk::PipelineStageFlagBits2::eNone;
		b.dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
		b.dstAccessMask       = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite;
		b.oldLayout           = vk::ImageLayout::eUndefined;
		b.newLayout           = vk::ImageLayout::eGeneral;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image               = vk::Image(img->image);
		b.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, img->levels, 0, 1};
		barriers.push_back(b);
		img->ready = true;
		if (img == &g_rt.dummy) dummy = true;
	}
	if (!barriers.empty()) {
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
		dependency.pImageMemoryBarriers    = barriers.data();
		command.pipelineBarrier2(dependency);
	}
	if (dummy) {
		vk::ClearColorValue zero {};
		zero.uint32 = std::array<uint32_t, 4> {0, 0, 0, 0};
		const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		command.clearColorImage(vk::Image(g_rt.dummy.image), vk::ImageLayout::eGeneral, &zero, 1, &whole);
	}
}

enum Stage : uint8_t { StHiz, StCalib, StTrace, StTemporal, StAtrous0, StAtrous1, StAtrous2, StAtrous3, StComposite };
const char* const StageNames[9] = {"hi-z", "calib", "trace", "temporal", "a-trous 1", "a-trous 2", "a-trous 3", "a-trous 4",
                                   "composite"};

void ReadTiming(GraphicContext& graphics) {
	if (g_rt.pool == nullptr) return;
	while (g_rt.read + 4 <= g_rt.written) {
		const auto     entry = static_cast<uint32_t>(g_rt.read % 8);
		const uint32_t n     = g_rt.stamps_in[entry];
		std::array<uint64_t, 16> stamps {};
		if (n >= 2 && graphics.device.getQueryPoolResults(g_rt.pool, entry * 16, n, sizeof(uint64_t) * n, stamps.data(),
		                                                  sizeof(uint64_t), vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		if (n >= 2 && stamps[n - 1] > stamps[0]) {
			g_rt.sum_ms += static_cast<double>(stamps[n - 1] - stamps[0]) * g_rt.period_ns / 1e6;
			++g_rt.samples;
			std::array<bool, 9> seen {};
			for (uint32_t k = 1; k < n; ++k) {
				const uint8_t st = g_rt.stage_of[entry][k];
				if (st >= 9 || stamps[k] < stamps[k - 1]) continue;
				g_rt.stage_ms[st] += static_cast<double>(stamps[k] - stamps[k - 1]) * g_rt.period_ns / 1e6;
				seen[st] = true;
			}
			for (uint32_t st = 0; st < 9; ++st)
				if (seen[st]) ++g_rt.stage_frames[st];
		}
		++g_rt.read;
	}
}

// The stats counters (the shader's atomics on 1 pixel in 16, cumulative: the difference since the last frame, modulo
// 2^32) and, every 10 s, one line that says whether the reflections show: the share of pixels with a weight > 0.05, the
// mean weight, the rays' hit rate, the G-buffer's state, the GPU time.
void ReadStats(GraphicContext& graphics, bool have_gbuf, int level) {
	if (g_rt.mapped == nullptr) return;
	(void)vmaInvalidateAllocation(graphics.allocator, g_rt.frame_mem, 0, VK_WHOLE_SIZE);
	const auto* c = reinterpret_cast<const uint32_t*>(g_rt.mapped + HeaderBytes / 4 + StatsIndex);
	for (uint32_t k = 0; k < StatsCount; ++k) {
		const uint32_t v = c[k];
		// (the first frames: the buffer's clear may not have run yet, its contents are a baseline only)
		if (g_rt.frame > 8) g_rt.stat_sum[k] += static_cast<uint32_t>(v - g_rt.stat_last[k]);
		g_rt.stat_last[k] = v;
	}
	++g_rt.stat_frames;
	if (have_gbuf) ++g_rt.stat_gbuf_frames;
	const auto now = std::chrono::steady_clock::now();
	if (now - g_rt.stat_time < std::chrono::seconds(10)) return;
	const auto&    sm     = g_rt.stat_sum;
	const double   px     = static_cast<double>(std::max<uint64_t>(sm[0], 1));
	const double   area   = (Cfg(level).full ? 1.0 : 0.25) * (Cfg(level).checker ? 0.5 : 1.0); // (rays per render pixel)
	const uint64_t copies = RtReflCopies();
	const bool  mat    = g_rt.rough_trusted || g_rt.mat_auto;
	const char* source = mat ? (g_rt.have_water ? (g_rt.rough_trusted ? "material field (env) + water" : "material field (auto) + water")
	                                            : (g_rt.rough_trusted ? "material field (env)" : "material field (auto)"))
	                     : g_rt.have_water ? "water (depth after the G-buffer) + faint puddle guess"
	                                       : "faint puddle guess only (no depth copy)";
	std::printf("RT reflections: last %.0f s: mask %s marks %.1f%% of the pixels reflective (water %.1f%%, material wet %.1f%%, "
	            "depth changed after the G-buffer %.1f%%); weight > 0.05 on "
	            "%.1f%% of the pixels, mean weight %.3f (%.3f on reflective surfaces), rays on %.1f%%, hit rate %.0f%%, "
	            "G-buffer copy %llu of %llu frames, its normal used on %.1f%% (%s), screen y %s, GPU %.3f ms\n",
	            std::chrono::duration<double>(now - g_rt.stat_time).count(), source, 100.0 * sm[3] / px, 100.0 * sm[7] / px,
	            100.0 * sm[9] / px, 100.0 * sm[8] / px,
	            100.0 * sm[1] / px, sm[2] / 1000.0 / px, sm[3] > 0 ? sm[2] / 1000.0 / static_cast<double>(sm[3]) : 0.0,
	            100.0 * sm[4] / (px * area), sm[4] > 0 ? 100.0 * sm[5] / static_cast<double>(sm[4]) : 0.0,
	            static_cast<unsigned long long>(copies - g_rt.stat_copies0), static_cast<unsigned long long>(g_rt.stat_frames),
	            100.0 * sm[6] / px, g_rt.dec_normal >= 0 ? "decoded" : "not identified: depth normals",
	            g_rt.row0_up ? "row 0 = NDC +1" : "row 0 = NDC -1", g_rt.samples > 0 ? g_rt.sum_ms / g_rt.samples : 0.0);
	{
		// (the wet cap for the next 10 s: material-wet on more than 40 % of the pixels scales it down)
		const double wet = static_cast<double>(sm[9]) / px;
		g_rt.wet_cap     = wet > 0.4 ? static_cast<float>(0.4 / wet) : 1.0f;
		if (wet > 0.4) std::printf("RT reflections: material wet on %.0f%% of the pixels: wet strength x%.2f\n", 100.0 * wet, g_rt.wet_cap);
	}
	std::string passes;
	for (uint32_t st = 0; st < 9; ++st) {
		if (g_rt.stage_frames[st] == 0) continue;
		char buf[64];
		// (calib: its mean over the frames that ran it; the others run every frame)
		std::snprintf(buf, sizeof(buf), "%s%s %.3f", passes.empty() ? "" : ", ", StageNames[st],
		              g_rt.stage_ms[st] / g_rt.stage_frames[st]);
		passes += buf;
	}
	if (!passes.empty()) std::printf("RT reflections: GPU ms per pass (%s): %s\n", Cfg(level).name, passes.c_str());
	std::fflush(stdout);
	g_rt.stage_ms.fill(0.0);
	g_rt.stage_frames.fill(0);
	g_rt.stat_sum.fill(0);
	g_rt.stat_frames = 0, g_rt.stat_gbuf_frames = 0, g_rt.stat_copies0 = copies;
	g_rt.sum_ms = 0.0, g_rt.samples = 0;
	g_rt.stat_time = now;
}

// The calibration's verdict: the normal layout (mean cos vs depth normals >= 0.8, best of layer x layout x view /
// world space x screen y convention; the convention it implies is adopted) and, when its layout leaves a byte free in
// the word, that byte as roughness (a guess: only modulating the wet-surface mask in the shader). Without a normal, the
// floor votes (world up known) decide the y convention. True: a normal identified.
bool Analyze(GraphicContext& graphics, uint32_t packed_count) {
	if (g_rt.mapped == nullptr) return false;
	(void)vmaInvalidateAllocation(graphics.allocator, g_rt.frame_mem, 0, VK_WHOLE_SIZE);
	const int32_t* c = g_rt.mapped + HeaderBytes / 4;
	const double   n = static_cast<double>(c[0]);
	static const char* const layouts[NLayouts] = {"oct 16:16",         "oct 12:12 bits 0..23", "oct 12:12 bits 8..31",
	                                              "oct 11:11 bits 0..21", "xyz 10:10:10",       "oct 16:16 swapped",
	                                              "oct 16:16 snorm",   "spheremap 16:16",      "xyz 11:11:10",
	                                              "oct 8:8 bits 0..15", "oct 8:8 bits 16..31", "hemi-oct 16:16",
	                                              "xyz 10:10:10 bits 2..31"};
	int    best = -1, best_conv = 0;
	double best_score = 0.0, second = 0.0;
	if (n >= 200.0) {
		std::array<double, 4 * NLayouts> per {}; // (the best |score| of each layer x layout: the runner-up's margin)
		for (uint32_t layer = 0; layer < packed_count && layer < 4; ++layer)
			for (uint32_t l = 0; l < NLayouts; ++l)
				for (int space = 0; space < 2; ++space)
					for (int conv = 0; conv < 2; ++conv) {
						const double score = c[CalibNormal + (layer * NLayouts + l) * 4 + space * 2 + conv] / 1000.0 / n;
						per[layer * NLayouts + l] = std::max(per[layer * NLayouts + l], std::abs(score));
						if (std::abs(score) > std::abs(best_score)) {
							best_score = score, best_conv = conv;
							best       = static_cast<int>(layer | (l << 2)) | (space << 6);
						}
					}
		for (uint32_t k = 0; k < per.size(); ++k)
			if (best >= 0 && static_cast<int>(k) != (best & 3) * static_cast<int>(NLayouts) + ((best >> 2) & 15))
				second = std::max(second, per[k]);
	}
	const double floors_cur = c[CalibFloors], floors_mirror = c[CalibFloors + 1];
	std::printf("RT reflections: calibration %u: %.0f samples, %u packed layers, best normal fit %.3f (next layout %.3f), "
	            "floors %.0f (screen y as is) vs %.0f (mirrored)\n",
	            g_rt.calib_attempts + 1, n, packed_count, std::abs(best_score), second, floors_cur, floors_mirror);
	const bool identified = best >= 0 && std::abs(best_score) >= 0.8;
	if (identified) {
		if (best_score < 0.0) best |= 128;
		if (g_rt.dec_normal < 0) g_rt.dec_normal = best;
		if (best_conv == 1 && !g_rt.row0_forced) g_rt.row0_up = !g_rt.row0_up, g_rt.history_valid = false;
		const int layout = (best >> 2) & 15, layer = best & 3;
		std::printf("RT reflections: G-buffer normal = layer %d, %s, %s space%s (mean cos %.3f vs depth normals); screen row 0 "
		            "= NDC y %s\n",
		            layer, layouts[layout], (best & 64) ? "world" : "view", (best & 128) ? ", negated" : "", std::abs(best_score),
		            g_rt.row0_up ? "+1" : "-1");
	} else {
		std::printf("RT reflections: G-buffer normal not identified (best |mean cos| %.3f, %.0f samples): depth normals and "
		            "the wet-surface mask\n",
		            std::abs(best_score), n);
		if (!g_rt.row0_forced && floors_mirror >= 300.0 && floors_mirror > 2.0 * floors_cur) {
			g_rt.row0_up       = !g_rt.row0_up;
			g_rt.history_valid = false;
			std::printf("RT reflections: screen row 0 = NDC y %s (the floor votes)\n", g_rt.row0_up ? "+1" : "-1");
		}
	}
	// The byte fields overall, on puddle-looking pixels and on water (the depth drawn after the G-buffer): a field that
	// differs clearly on water / puddles is the material's (set it with KYTY_RT_REFLECTIONS_ROUGH=<layer>:<shift>:8[g]).
	if (packed_count > 0 && n >= 200.0) {
		const double pud = c[CalibWet], wat = c[CalibWet + 17];
		std::string  line;
		char         buf[96];
		for (uint32_t f = 0; f < 16; ++f) {
			const double all = c[CalibBytes + f * 3] / 1000.0 / n;
			std::snprintf(buf, sizeof(buf), " L%u.b%u %.2f/%s/%s;", f / 4, f % 4, all,
			              pud >= 20 ? std::to_string(c[CalibWet + 1 + f] / 1000.0 / pud).substr(0, 4).c_str() : "-",
			              wat >= 20 ? std::to_string(c[CalibWet + 18 + f] / 1000.0 / wat).substr(0, 4).c_str() : "-");
			line += buf;
		}
		std::printf("RT reflections: G-buffer byte fields, mean all / puddle-looking (%.0f) / water (%.0f):%s\n", pud, wat, line.c_str());
	}
	// The material field, automatically: the byte (outside the decoded normal's bits) that separates the puddle-looking
	// pixels from all the others the most (>= 1 standard deviation and 0.2 apart). Below the overall mean on puddles:
	// roughness (low = wet / smooth); above: gloss (inverted). Wet below a quarter of the way from the puddles' mean to
	// the overall mean, dry from halfway (in roughness units). The 10-24 log: L2.b0 0.40 overall / 0.01 on puddles,
	// L1.b2 0.60 / 0.96.
	if (!g_rt.rough_trusted && g_rt.dec_rough < 0 && packed_count > 0 && n >= 200.0 && c[CalibWet] >= 100 && g_rt.dec_normal >= 0) {
		const int    nlayer = g_rt.dec_normal & 3, nlayout = (g_rt.dec_normal >> 2) & 15;
		const auto   in_normal = [&](int layer, int b) {
			if (layer != nlayer) return false;
			switch (nlayout) {
				case 1: case 3: return b <= 2;
				case 2: return b >= 1;
				case 9: return b <= 1;
				case 10: return b >= 2;
				default: return true;
			}
		};
		const double pud = c[CalibWet];
		int          pick = -1;
		double       best_sep = 0.0, pick_all = 0.0, pick_pud = 0.0;
		for (int f = 0; f < 16; ++f) {
			if (in_normal(f / 4, f % 4)) continue;
			const double all = c[CalibBytes + f * 3] / 1000.0 / n, sq = c[CalibBytes + f * 3 + 1] / 1000.0 / n;
			const double sd  = std::sqrt(std::max(sq - all * all, 0.0));
			const double mp  = c[CalibWet + 1 + f] / 1000.0 / pud;
			if (sd < 0.05) continue;
			const double sep = std::abs(mp - all) / sd;
			if (sep >= 1.0 && std::abs(mp - all) >= 0.2 && sep > best_sep) best_sep = sep, pick = f, pick_all = all, pick_pud = mp;
		}
		if (pick >= 0) {
			const bool   gloss = pick_pud > pick_all;
			const double rp = gloss ? 1.0 - pick_pud : pick_pud, ra = gloss ? 1.0 - pick_all : pick_all;
			g_rt.dec_rough = (pick / 4) | (((pick % 4) * 8) << 2) | (8 << 7) | (gloss ? 4096 : 0);
			if (!g_rt.mat_forced) {
				g_rt.mat_lo = static_cast<float>(rp + (ra - rp) * 0.25);
				g_rt.mat_hi = static_cast<float>(rp + (ra - rp) * 0.5);
			}
			g_rt.mat_auto = true;
			// (the share of all pixels it marks wet, from the field's histogram)
			double wet = 0.0, total = 0.0;
			for (int b = 0; b < 16; ++b) {
				const double v  = (b + 0.5) / 16.0, r = gloss ? 1.0 - v : v;
				const double hc = c[CalibHist + pick * 16 + b];
				total += hc;
				if (r < g_rt.mat_hi) wet += hc * (r < g_rt.mat_lo ? 1.0 : 0.5);
			}
			std::printf("RT reflections: material field = layer %d bits %d..%d%s (mean %.2f overall, %.2f on puddle-looking pixels, "
			            "%.1f sd apart): wet below %.2f, dry from %.2f (roughness units), about %.1f%% of the samples wet "
			            "(KYTY_RT_REFLECTIONS_ROUGH=%d:%d:8%s and KYTY_RT_REFLECTIONS_WET=lo:hi override)\n",
			            pick / 4, (pick % 4) * 8, (pick % 4) * 8 + 7, gloss ? " (gloss: inverted)" : " (roughness)", pick_all, pick_pud,
			            best_sep, g_rt.mat_lo, g_rt.mat_hi, total > 0 ? 100.0 * wet / total : 0.0, pick / 4, (pick % 4) * 8,
			            gloss ? "g" : "");
		} else {
			std::printf("RT reflections: no material field separates the puddle-looking pixels (>= 1 sd): water and the faint "
			            "puddle guess\n");
		}
	} else if (!g_rt.rough_trusted && g_rt.dec_rough < 0 && packed_count > 0) {
		std::printf("RT reflections: material field not looked for (%s): water and the faint puddle guess\n",
		            g_rt.dec_normal < 0 ? "no G-buffer normal" : "too few puddle-looking samples");
	}
	std::fflush(stdout);
	return identified;
}

} // namespace

int RtReflectionsLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_RT_REFLECTIONS");
		return value != nullptr ? std::clamp(std::atoi(value), 0, 4) : 0; // 0 off, 1 low, 2 medium, 3 high, 4 ultra
	}();
	return level;
}
bool RtReflectionsWanted() { return RtReflectionsLevel() > 0; }

bool RtReflectionsApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int level = RtReflectionsLevel();
	if (level == 0) return false; // off: nothing created or recorded
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr) return false;
	if (VideoOut::MoviePlaying()) {
		g_rt.history_valid = false;
		return false;
	}
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || inputs.motion.view == nullptr || color.extent.width < 16 ||
	    color.extent.height < 16 || inputs.depth.extent != color.extent)
		return false;
	const auto& m = inputs.view_to_clip;
	if (!(m[0] != 0.0f && m[5] != 0.0f && (m[11] != 0.0f || m[15] != 0.0f)) ||
	    !std::all_of(m.begin(), m.end(), [](float v) { return std::isfinite(v); })) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("RT reflections: skipped (no camera projection from the upscaler)\n");
			std::fflush(stdout);
		}
		return false;
	}
	if (g_rt.images_failed || !EnsurePipeline(*graphics) || !EnsureImages(*graphics, color.extent, level)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("RT reflections: off (could not create its pipeline or images)\n");
			std::fflush(stdout);
		}
		return false;
	}
	FreeRetired(*graphics);
	ReadTiming(*graphics);

	// The G-buffer copy (this flip or the last), the decode from the environment, the calibration.
	RtReflGbuffer gbuf;
	const bool    have_gbuf    = RtReflGetGbuffer(gbuf) && gbuf.extent == color.extent;
	const uint32_t packed_count = have_gbuf ? 4u : 0u;
	ReadStats(*graphics, have_gbuf, level);
	{
		static bool said = false;
		if (!said) {
			said = true;
			const int32_t n = NormalEnv("KYTY_RT_REFLECTIONS_NORMAL"), r = FieldEnv("KYTY_RT_REFLECTIONS_ROUGH");
			if (n >= 0) g_rt.dec_normal = n;
			if (r >= 0) g_rt.dec_rough = r, g_rt.rough_trusted = true;
			if (const char* w = std::getenv("KYTY_RT_REFLECTIONS_WET"); w != nullptr) {
				float lo = 0.0f, hi = 0.0f;
				if (std::sscanf(w, "%f:%f", &lo, &hi) == 2 && lo >= 0.0f && hi > lo && hi <= 1.0f)
					g_rt.mat_lo = lo, g_rt.mat_hi = hi, g_rt.mat_forced = true;
			}
			g_rt.mat_scale = EnvFloat("KYTY_RT_REFLECTIONS_MATERIAL", 1.0f, 0.0f, 1.0f);
			if (const char* y = std::getenv("KYTY_RT_REFLECTIONS_FLIPY"); y != nullptr && *y != '\0')
				g_rt.row0_up = *y != '0', g_rt.row0_forced = true;
			// (a normal given and the convention forced: nothing left to calibrate)
			if (n >= 0 && g_rt.row0_forced) g_rt.calib_done = true;
			if (n >= 0 || r >= 0 || g_rt.row0_forced) {
				std::printf("RT reflections: decode from the environment (normal %d, roughness %d, screen y %s)\n", n, r,
				            g_rt.row0_forced ? (g_rt.row0_up ? "row 0 = NDC +1" : "row 0 = NDC -1") : "calibrated");
				std::fflush(stdout);
			}
		}
	}
	// The calibration. G-buffer attempts (the ones that count, up to 3, ~30 s apart until a normal fits): once the copy
	// has been there 120 frames running, so a scene and not the first loading frames; its runs wait while the copy is
	// missing. Before any copy, after 900 frames: one floor-vote run (packed layers 0, the y convention only), which
	// does not count (the first version spent its attempt there when the copy came late: "0 packed layers").
	g_rt.gb_streak = have_gbuf ? g_rt.gb_streak + 1 : 0;
	if (!g_rt.calib_done && !g_rt.calib_active && g_rt.frame >= g_rt.calib_next) {
		if (g_rt.gb_streak >= 120) {
			g_rt.calib_active = true, g_rt.calib_clear = true;
			g_rt.calib_runs = 0, g_rt.calib_wait = 0, g_rt.calib_packed = 4;
		} else if (!g_rt.floors_done && g_rt.gb_streak == 0 && g_rt.frame >= 900 && g_rt.calib_attempts == 0) {
			g_rt.floors_done  = true;
			g_rt.calib_active = true, g_rt.calib_clear = true;
			g_rt.calib_runs = 0, g_rt.calib_wait = 0, g_rt.calib_packed = 0;
		}
	}
	const bool calibrate = g_rt.calib_active && g_rt.calib_runs < CalibRuns && (g_rt.calib_tick++ & 3u) == 0 &&
	                       (g_rt.calib_packed == 0 || have_gbuf);
	if (g_rt.calib_active && g_rt.calib_runs >= CalibRuns && ++g_rt.calib_wait > 8) {
		g_rt.calib_active     = false;
		const bool identified = Analyze(*graphics, g_rt.calib_packed);
		if (g_rt.calib_packed > 0) {
			++g_rt.calib_attempts;
			// (done once the normal and a material field are known; else again ~30 s later, up to 6 attempts: the
			// material needs puddle-looking pixels in view)
			const bool material = g_rt.dec_rough >= 0 || g_rt.rough_trusted;
			if ((g_rt.dec_normal >= 0 && material) || g_rt.calib_attempts >= 6) g_rt.calib_done = true;
			else g_rt.calib_next = g_rt.frame + 1800;
			(void)identified;
		}
	}
	const int32_t dec_normal = have_gbuf ? g_rt.dec_normal : -1;
	// (a roughness field is used only when trusted, from KYTY_RT_REFLECTIONS_ROUGH: a guess made dry stone shine)
	const int32_t dec_rough  = have_gbuf && dec_normal >= 0 && (g_rt.rough_trusted || g_rt.mat_auto) ? g_rt.dec_rough : -1;
	const bool    have_mat   = dec_rough >= 0;
	const bool    have_water = have_gbuf && gbuf.depth != nullptr;
	g_rt.have_water          = have_water;

	RtReflCamera camera;
	RtReflGetCamera(m, camera);
	const bool use_normal = dec_normal >= 0 && ((dec_normal & 64) == 0 || camera.rows);

	const float strength = EnvFloat("KYTY_RT_REFLECTIONS_STRENGTH", 1.0f, 0.0f, 3.0f);
	const float min_refl = EnvFloat("KYTY_RT_REFLECTIONS_MIN", 0.35f, 0.0f, 0.8f);
	const float guess    = EnvFloat("KYTY_RT_REFLECTIONS_GUESS", 0.2f, 0.0f, 1.0f);
	const auto& cfg      = Cfg(level);
	const int   iters    = cfg.iters;
	const float cutoff   = cfg.cutoff;
	const int   passes   = cfg.passes;
	const float cap_px   = cfg.cap * static_cast<float>(color.extent.height);
	{
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("RT reflections: on (level %d %s: trace %ux%u of %ux%u%s, hi-z %u mips, %d iterations, rays <= %.0f px, %d "
			            "a-trous passes, strength %g, wet minimum reflectance %g, G-buffer %s, exposure %s, camera %s)\n",
			            level, cfg.name, g_rt.trace.width, g_rt.trace.height, color.extent.width, color.extent.height,
			            cfg.checker ? " checkerboard" : "", g_rt.hiz.levels, iters, cap_px, passes, strength, min_refl, have_gbuf ? "copy found (calibrating)" : "not found yet (depth normals, heuristic)",
			            inputs.exposure_buffer != nullptr ? "the game's" : "1.0", camera.prev ? "matrices" : "motion vectors only");
			std::fflush(stdout);
		}
	}

	RtPush push {};
	push.proj[0] = m[0], push.proj[1] = m[5], push.proj[2] = m[8], push.proj[3] = m[9];
	push.depth_map[0] = m[10], push.depth_map[1] = m[14], push.depth_map[2] = m[11], push.depth_map[3] = m[15];
	push.size[0] = static_cast<float>(color.extent.width), push.size[1] = static_cast<float>(color.extent.height);
	push.size[2] = static_cast<float>(g_rt.trace.width), push.size[3] = static_cast<float>(g_rt.trace.height);
	push.params[0] = strength, push.params[1] = EnvFloat("KYTY_RT_REFLECTIONS_THICKNESS", 0.3f, 0.01f, 4.0f);
	push.params[2] = cfg.max_len;
	push.motion[0] = inputs.motion_scale_x, push.motion[1] = inputs.motion_scale_y;
	push.motion[2] = cutoff, push.motion[3] = cfg.alpha;
	push.misc[1] = static_cast<int32_t>(g_rt.frame & 0x7fffffff);
	push.misc[2] = level;
	push.misc[3] = (inputs.reset ? 1 : 0) | (g_rt.history_valid ? 2 : 0) | (camera.prev ? 4 : 0) | (camera.up_valid ? 8 : 0) |
	               (g_rt.row0_up ? 16 : 0) | (cfg.checker ? 32 : 0) | (have_water ? 64 : 0);
	push.dec[0] = use_normal ? dec_normal : -1, push.dec[1] = use_normal ? dec_rough : -1;
	push.dec[2] = static_cast<int32_t>(packed_count), push.dec[3] = 0;
	const bool game_exposure = inputs.exposure_buffer != nullptr && inputs.exposure_range >= sizeof(float);
	push.misc2[0] = game_exposure ? static_cast<int32_t>(inputs.exposure_index) : static_cast<int32_t>(ExposureIndex);
	push.misc2[1] = static_cast<int32_t>(g_rt.hiz.levels), push.misc2[2] = iters, push.misc2[3] = g_rt.rough_trusted ? 1 : 0;

	const bool timed = g_rt.pool != nullptr && g_rt.written < g_rt.read + 8; // (a ring of 8 frames)
	const auto entry = static_cast<uint32_t>(g_rt.written % 8);
	uint32_t   stamp_n = 0;
	const auto stamp   = [&](uint8_t stage) {
		if (!timed || stamp_n >= 16) return;
		g_rt.stage_of[entry][stamp_n] = stage;
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_rt.pool, entry * 16 + stamp_n);
		++stamp_n;
	};
	if (timed) command.resetQueryPool(g_rt.pool, entry * 16, 16);
	stamp(0xff); // (the start)
	// (What wrote the game's images, and our last frame's reads of the buffer and images.)
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite);
	PrepareImages(command);
	if (!g_rt.frame_cleared) {
		g_rt.frame_cleared = true;
		command.fillBuffer(vk::Buffer(g_rt.frame_buf), 0, VK_WHOLE_SIZE, 0);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer,
		        vk::AccessFlagBits2::eTransferWrite);
	} else if (g_rt.calib_clear) {
		// (a new calibration attempt: its sums from zero, the stats counters kept)
		command.fillBuffer(vk::Buffer(g_rt.frame_buf), HeaderBytes, StatsIndex * 4, 0);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer,
		        vk::AccessFlagBits2::eTransferWrite);
	}
	g_rt.calib_clear = false;
	{
		std::array<float, HeaderBytes / 4> header {};
		std::copy(camera.view_rows.begin(), camera.view_rows.end(), header.begin());
		std::copy(camera.clip_to_prev.begin(), camera.clip_to_prev.end(), header.begin() + 12);
		header[28] = camera.up[0], header[29] = camera.up[1], header[30] = camera.up[2], header[31] = camera.up_valid ? 1.0f : 0.0f;
		header[32] = 1.0f;     // (ExposureIndex: the exposure when the game gives none)
		header[33] = min_refl; // (fparams.y)
		header[34] = cap_px;   // (fparams.z)
		header[35] = have_mat ? 0.0f : guess; // (fparams.w: the puddle guess's scale; off with a material field)
		header[36] = g_rt.mat_lo, header[37] = g_rt.mat_hi, header[38] = g_rt.mat_scale;
		header[39] = g_rt.wet_cap; // (mparams.w: 1, or 0.4 / the wet share when more than 40 % of the pixels are wet) // (mparams)
		command.updateBuffer(vk::Buffer(g_rt.frame_buf), 0, HeaderBytes, header.data());
	}
	Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_rt.pipeline);

	auto&          g    = g_rt;
	const uint32_t cur  = static_cast<uint32_t>(g.frame & 1u);
	const uint32_t prev = cur ^ 1u;
	struct Set {
		const RtImage *a, *b, *gcur, *gprev, *aux, *meta;
		vk::ImageView  hiz_store;
		const RtImage *oa, *ob, *oc;
	};
	const auto run = [&](int mode, const Set& s, vk::Extent2D size) {
		const vk::Sampler       lin = g.linear, pt = g.point;
		const auto              gen = vk::ImageLayout::eGeneral;
		vk::DescriptorImageInfo infos[Bindings] {};
		infos[0]  = {lin, color.view, color.layout};
		infos[1]  = {pt, inputs.depth.view, inputs.depth.layout};
		infos[2]  = {pt, inputs.motion.view, inputs.motion.layout};
		infos[3]  = {pt, g.hiz.view, gen};
		infos[4]  = {nullptr, s.hiz_store, gen};
		infos[5]  = {lin, s.a->view, gen};
		infos[6]  = {lin, s.b->view, gen};
		infos[7]  = {pt, s.gcur->view, gen};
		infos[8]  = {pt, s.gprev->view, gen};
		infos[9]  = {pt, s.aux->view, gen};
		infos[10] = {pt, s.meta->view, gen};
		infos[11] = {nullptr, s.oa->view, gen};
		infos[12] = {nullptr, s.ob->view, gen};
		infos[13] = {nullptr, s.oc->view, gen};
		for (uint32_t i = 0; i < 4; ++i)
			infos[14 + i] = have_gbuf && gbuf.views[i] != nullptr ? vk::DescriptorImageInfo {pt, gbuf.views[i], vk::ImageLayout::eShaderReadOnlyOptimal}
			                                                      : vk::DescriptorImageInfo {pt, g.dummy.view, gen};
		// (binding 20: the G-buffer-time depth; without it the hi-z stands in, never read: misc.w & 64 unset)
		infos[20] = have_water ? vk::DescriptorImageInfo {pt, gbuf.depth, gbuf.depth_layout} : vk::DescriptorImageInfo {pt, g.hiz.view, gen};
		const vk::DescriptorBufferInfo frame_info {vk::Buffer(g.frame_buf), 0, VK_WHOLE_SIZE};
		const vk::DescriptorBufferInfo exposure_info = game_exposure
		                                                   ? vk::DescriptorBufferInfo {inputs.exposure_buffer, inputs.exposure_base, inputs.exposure_range}
		                                                   : frame_info;
		vk::WriteDescriptorSet writes[Bindings] {};
		for (uint32_t i = 0; i < Bindings; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = (i == 18 || i == 19)                       ? vk::DescriptorType::eStorageBuffer
			                            : (i == 4 || i == 11 || i == 12 || i == 13) ? vk::DescriptorType::eStorageImage
			                                                                        : vk::DescriptorType::eCombinedImageSampler;
			if (i == 18) writes[i].pBufferInfo = &frame_info;
			else if (i == 19) writes[i].pBufferInfo = &exposure_info;
			else writes[i].pImageInfo = &infos[i];
		}
		push.misc[0] = mode;
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g.layout, 0, Bindings, writes);
		command.pushConstants(g.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	};
	// (Bindings a mode does not use still need valid images: tmp_a / tmp_b / rad where nothing reads or writes them.)
	const Set base {&g.tmp_a, &g.tmp_a, &g.tmp_a, &g.tmp_a, &g.tmp_a, &g.tmp_a, g.hiz.mips[0], &g.tmp_b, &g.tmp_b, &g.tmp_b};
	// Hi-z pyramid.
	run(6, base, g.render);
	for (uint32_t mip = 1; mip < g.hiz.levels; ++mip) {
		Set s       = base;
		s.hiz_store = g.hiz.mips[mip];
		push.dec[3] = static_cast<int32_t>(mip);
		run(7, s, MipExtent(g.hiz, mip));
	}
	push.dec[3] = 0;
	stamp(StHiz);
	if (calibrate) {
		push.dec[2] = static_cast<int32_t>(g_rt.calib_packed);
		run(5, base, vk::Extent2D {160, 90});
		push.dec[2] = static_cast<int32_t>(packed_count);
		++g.calib_runs;
		stamp(StCalib);
	}
	// Trace, temporal, a-trous, composite.
	{
		Set s = base;
		s.oa = &g.rad, s.ob = &g.gb[cur], s.oc = &g.aux;
		// (checkerboard: a thread per pair of horizontal neighbours)
		run(0, s, cfg.checker ? vk::Extent2D {(g.trace.width + 1) / 2, g.trace.height} : g.trace);
		stamp(StTrace);
	}
	{
		Set s = base;
		s.a = &g.rad, s.b = &g.hist[prev], s.gcur = &g.gb[cur], s.gprev = &g.gb[prev], s.aux = &g.aux, s.meta = &g.meta[prev];
		s.oa = &g.hist[cur], s.ob = &g.meta[cur], s.oc = &g.tmp_b;
		run(1, s, g.trace);
		stamp(StTemporal);
	}
	const RtImage* src     = &g.hist[cur];
	const RtImage* ping[2] = {&g.tmp_a, &g.tmp_b};
	for (int i = 0; i < passes; ++i) {
		const RtImage* dst = ping[i & 1];
		Set            s   = base;
		s.a = src, s.b = src, s.gcur = &g.gb[cur], s.gprev = &g.gb[cur], s.aux = &g.aux, s.meta = &g.meta[cur];
		s.oa = dst, s.ob = &g.rad, s.oc = &g.rad;
		push.params[3] = static_cast<float>(1 << i);
		run(2, s, g.trace);
		stamp(static_cast<uint8_t>(StAtrous0 + std::min(i, 3)));
		src = dst;
	}
	{
		Set s = base;
		s.a = src, s.b = src, s.gcur = &g.gb[cur], s.gprev = &g.gb[cur], s.aux = &g.aux, s.meta = &g.meta[cur];
		s.oa = &g.out, s.ob = &g.rad, s.oc = &g.rad;
		run(4, s, g.render);
		stamp(StComposite);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead);
	{
		// The sums and counters to the host (the CPU reads them a few frames late; the copy is 2.3 KB).
		vk::BufferCopy region {HeaderBytes, HeaderBytes, (StatsIndex + StatsCount) * 4};
		command.copyBuffer(vk::Buffer(g.frame_buf), vk::Buffer(g.read_buf), 1, &region);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
		        vk::AccessFlagBits2::eHostRead);
	}
	if (timed) {
		g.stamps_in[entry] = stamp_n;
		++g.written;
	}
	++g.frame;
	g.history_valid     = true;
	inputs.color.image  = vk::Image(g.out.image);
	inputs.color.view   = g.out.view;
	inputs.color.layout = vk::ImageLayout::eGeneral;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = g.out.usage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}

} // namespace Libs::Graphics::FrameGen
