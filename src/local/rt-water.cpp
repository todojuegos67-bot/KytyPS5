// KYTY_HWRT_WATER=0|1|2|3 ("Reflejos de agua RT": No / Bajo / Medio / Alto; default 0, off; needs KYTY_HWRT=1): sharp
// ray-traced mirror reflections on water, puddles and wet stone (rt-water.h, rt-water.comp). Recorded in
// FrameGen::Upscale right after RtReflectionsApply, on the pre-exposure HDR color before the upscaler.
//
// Why the reflections' log said "water 0.0%" always (logs of 2026-10-27): KYTY_RT_REFLECTIONS finds water as "the final
// depth nearer than the depth the deferred lighting read". Demon's Souls draws its water surface (PS 7a59c1a20cbe5d4a,
// docs/PERF-ZONE-GOAL.md 2026-10-13) forward, after the lighting, as a blended surface: the depth left after it is the
// bed's where it does not write depth, and where it does the up-facing / flat test on the depth's own normals (the water's
// vertex waves) refused it. So water was never seen, and its own SSR guide never covers it either (drawn after the SSR).
//
// Here water is found by its geometry: the water shader's draws (KYTY_HWRT_WATER_PS, default 7a59c1a20cbe5d4a) take
// KYTY_HWRT's capturing pipeline too (table-xpr.inc) and go into a TLAS of their own (hwrt-scene.cpp BuildWaterTlas; never
// in the scene TLAS, so no other pass changes). Per pixel: the camera ray meets that TLAS before the depth buffer's
// surface -> water, with the exact surface point and its triangle's normal (near-level planes snapped level). Puddles /
// wet stone: level floors where the game's own SSR is confident (KYTY_RT_REFLECTIONS' guide, the half-resolution SSR
// result of CS 9fb5cc274b5e4f82; only while KYTY_RT_REFLECTIONS runs). Then one mirror ray (static scene, mask 0x01)
// through a static world-space ripple, Schlick Fresnel (f0 0.02) raised to a minimum reflectance; a hit visible on the
// screen (or the screen's sky) gives the frame's colour; anything else leaves the game's own reflection.
// Stability: no stochastic term at all (one ray, a fixed direction); a temporal history of the virtual image, disocclusion
// by the surface's kind and distance at its previous position, a 3x3 mean +- sigma clamp, a firefly clamp, alpha
// 0.08 still / 0.2 moving; the history is dropped on a cut, a resize or a frame the pass did not run.
//
// Levels (water2: Medio at half resolution too): 1 Bajo (trace at half resolution, rays <= 80 m, puddles from SSR confidence 0.30), 2 Medio (half resolution,
// 150 m, 0.20), 3 Alto (full resolution, 300 m, 0.12, stronger puddles). Options: KYTY_HWRT_WATER_PS=<hex>[,<hex>..],
// KYTY_HWRT_WATER_STRENGTH (0..2), _MIN (water's minimum reflectance, 0..0.9), _RIPPLE (normal slope amplitude, 0..0.2),
// _PUDDLE (puddles' share, 0 off, 0..1.5), _DIST (ray length, m), _FLIPY (1: row 0 = NDC y +1, the default).
// Every 10 s "HWRTWATER: 10 s:" with the water draws captured, the water / puddle share of the screen, the rays' outcome,
// the history's validity, the composited share, the TLAS alignment self-check and the GPU time; the flicker self-check
// has its own "water mirrors" entry in the RTFLICKER line.
#include "local/rt-water.h"
#include "local/env-level.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/rtwater_spv.h"
#include "local/hwrt-scene.h"
#include "local/rt-reflections.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace Libs::Graphics::FrameGen {
namespace {

constexpr uint32_t Bindings  = 14;
constexpr uint32_t SlotBytes = 512;
constexpr uint32_t Slots     = 4;
constexpr uint32_t StatWords = 32;
constexpr uint32_t ExposureFallbackIndex = 59; // (Params p1.w as a float index)

struct Push {
	float   size[4];
	int32_t misc[4];
	int32_t misc2[4];
	float   pad[4];
};
static_assert(sizeof(Push) == 64);

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
	vk::Extent2D            extent {}, trace {};
	Img                     out, raw, raw_aux;
	std::array<Img, 2>      hist, aux;
	bool                    layouts_ready = false, history_valid = false;
	std::array<std::pair<Img, uint64_t>, 16> retired {};
	VkBuffer                param_buf = VK_NULL_HANDLE, stat_buf = VK_NULL_HANDLE;
	VmaAllocation           param_mem = nullptr, stat_mem = nullptr;
	uint8_t*                param_mapped = nullptr;
	volatile uint32_t*      stat_mapped  = nullptr;
	uint64_t                frame = 0;
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms = 0.0, max_ms = 0.0;
	uint32_t                samples = 0;
	std::array<uint32_t, StatWords> stat_last {};
	std::array<double, 16>  prev_vp {};
	bool                    have_prev = false;
	uint64_t                frames = 0, frames_ran = 0, frames_water = 0, frames_guide = 0, frames_still = 0;
	uint64_t                last_draws = 0, last_wframes = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
	bool                    said_on = false;
	const char*             skip_why = nullptr;
	uint64_t                skipped = 0;
} g_w;
std::atomic<uint64_t> g_normal_draws {0};
uint64_t              g_normal_last = 0;

float EnvFloat(const char* name, float fallback, float lo, float hi) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') return fallback;
	const float v = static_cast<float>(std::atof(value));
	return std::isfinite(v) ? std::clamp(v, lo, hi) : fallback;
}

// The water shaders: KYTY_HWRT_WATER_PS or Demon's Souls' water surface.
struct WaterPsList {
	std::array<uint64_t, 16> hashes {};
	uint32_t                 count = 0;
};
const WaterPsList& WaterPs() {
	static const WaterPsList list = [] {
		WaterPsList l;
		const char* value = std::getenv("KYTY_HWRT_WATER_PS");
		std::string s     = value != nullptr && *value != '\0' ? value : "7a59c1a20cbe5d4a";
		size_t      pos   = 0;
		while (pos < s.size() && l.count < l.hashes.size()) {
			size_t end = s.find_first_of(",; ", pos);
			if (end == std::string::npos) end = s.size();
			const std::string item = s.substr(pos, end - pos);
			if (!item.empty()) {
				const uint64_t h = std::strtoull(item.c_str(), nullptr, 16);
				if (h != 0) l.hashes[l.count++] = h;
			}
			pos = end + 1;
		}
		return l;
	}();
	return list;
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
	b.dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader;
	b.dstAccessMask       = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
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

bool MakeImage(GraphicContext& graphics, vk::Extent2D extent, Img& img) {
	VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create.imageType     = VK_IMAGE_TYPE_2D;
	create.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
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
	view_info.format           = vk::Format::eR16G16B16A16Sfloat;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &img.view) != vk::Result::eSuccess) {
		img.view = nullptr;
		return false;
	}
	return true;
}
void DestroyImage(GraphicContext& graphics, Img& img) {
	if (img.view != nullptr) graphics.device.destroyImageView(img.view, nullptr);
	if (img.image != VK_NULL_HANDLE) vmaDestroyImage(graphics.allocator, img.image, img.memory);
	img = {};
}
// (frames in flight may still read it: destroyed 16 frames later)
void Retire(GraphicContext& graphics, Img& img) {
	if (img.image == VK_NULL_HANDLE) return;
	for (auto& slot: g_w.retired)
		if (slot.first.image == VK_NULL_HANDLE) {
			slot = {img, g_w.frame};
			img  = {};
			return;
		}
	graphics.device.waitIdle(); // (all 16 slots busy: only on repeated resizes)
	DestroyImage(graphics, img);
}
void FreeRetired(GraphicContext& graphics) {
	for (auto& slot: g_w.retired)
		if (slot.first.image != VK_NULL_HANDLE && g_w.frame >= slot.second + 16) DestroyImage(graphics, slot.first), slot = {};
}

vk::DescriptorType DescType(uint32_t i) {
	return i == 8 || i == 11 || i == 13 ? vk::DescriptorType::eStorageBuffer
	       : i == 9 || i == 10          ? vk::DescriptorType::eAccelerationStructureKHR
	       : i == 6 || i == 7           ? vk::DescriptorType::eStorageImage
	                                    : vk::DescriptorType::eCombinedImageSampler;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_w.pipeline != nullptr) return true;
	if (g_w.failed) return false;
	g_w.failed        = true;
	const auto device = graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, Bindings> bindings {};
	for (uint32_t i = 0; i < Bindings; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = DescType(i);
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = Bindings;
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_w.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Push)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_w.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_w.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(RTWATER_SPV);
	module_info.pCode       = RTWATER_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_w.layout;
	const auto created         = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_w.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_w.pipeline = nullptr;
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
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g_w.point : &g_w.linear) != vk::Result::eSuccess) return false;
	}
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = SlotBytes * Slots;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_w.param_buf, &g_w.param_mem, &mapped) != VK_SUCCESS) {
		g_w.param_buf = VK_NULL_HANDLE;
		return false;
	}
	g_w.param_mapped = static_cast<uint8_t*>(mapped.pMappedData);
	buffer.size      = StatWords * 4;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_w.stat_buf, &g_w.stat_mem, &mapped) != VK_SUCCESS) {
		g_w.stat_buf = VK_NULL_HANDLE;
		return false;
	}
	g_w.stat_mapped = static_cast<volatile uint32_t*>(mapped.pMappedData);
	for (uint32_t i = 0; i < StatWords; ++i) g_w.stat_mapped[i] = 0;
	(void)vmaFlushAllocation(graphics.allocator, g_w.stat_mem, 0, VK_WHOLE_SIZE);
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = 8 * 2;
	if (device.createQueryPool(&query, nullptr, &g_w.pool) != vk::Result::eSuccess) g_w.pool = nullptr;
	g_w.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g_w.failed    = false;
	return true;
}

bool EnsureImages(GraphicContext& graphics, vk::Extent2D extent, int level) {
	const vk::Extent2D trace = level >= 3 ? extent : vk::Extent2D {std::max(extent.width / 2, 1u), std::max(extent.height / 2, 1u)};
	if (g_w.out.image != VK_NULL_HANDLE && g_w.extent == extent && g_w.trace == trace) return true;
	Img* all[] {&g_w.out, &g_w.raw, &g_w.raw_aux, &g_w.hist[0], &g_w.hist[1], &g_w.aux[0], &g_w.aux[1]};
	for (Img* img: all) Retire(graphics, *img);
	g_w.extent = extent, g_w.trace = trace;
	g_w.layouts_ready = false, g_w.history_valid = false;
	bool ok = MakeImage(graphics, extent, g_w.out);
	for (Img* img: all)
		if (img != &g_w.out) ok = ok && MakeImage(graphics, trace, *img);
	if (!ok)
		for (Img* img: all) Retire(graphics, *img);
	return ok;
}

void ReadTiming(GraphicContext& graphics) {
	if (g_w.pool == nullptr) return;
	while (g_w.read + 4 <= g_w.written) {
		const auto              entry = static_cast<uint32_t>(g_w.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (graphics.device.getQueryPoolResults(g_w.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		if (stamps[1] > stamps[0]) {
			const double ms = static_cast<double>(stamps[1] - stamps[0]) * g_w.period_ns / 1e6;
			g_w.sum_ms += ms, g_w.max_ms = std::max(g_w.max_ms, ms);
			++g_w.samples;
		}
		++g_w.read;
	}
}

bool Inverse4(const double* m, double* out) {
	double inv[16];
	inv[0]  = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
	inv[4]  = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
	inv[8]  = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
	inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
	inv[1]  = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
	inv[5]  = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
	inv[9]  = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
	inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
	inv[2]  = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
	inv[6]  = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
	inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
	inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
	inv[3]  = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
	inv[7]  = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
	inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
	inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
	const double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
	if (!(std::abs(det) > 1e-30) || !std::isfinite(det)) return false;
	for (int i = 0; i < 16; ++i) out[i] = inv[i] / det;
	return true;
}

struct LevelCfg {
	float dist, puddle_conf, puddle_share, ripple, strength, min_refl;
};
LevelCfg Cfg(int level) {
	static const LevelCfg cfgs[] {{80.0f, 0.30f, 0.6f, 0.020f, 0.85f, 0.22f},
	                              {150.0f, 0.20f, 0.8f, 0.025f, 1.00f, 0.28f},
	                              {300.0f, 0.12f, 1.0f, 0.030f, 1.00f, 0.35f}};
	LevelCfg c = cfgs[std::clamp(level, 1, 3) - 1];
	c.dist         = EnvFloat("KYTY_HWRT_WATER_DIST", c.dist, 5.0f, 2000.0f);
	c.puddle_share = EnvFloat("KYTY_HWRT_WATER_PUDDLE", c.puddle_share, 0.0f, 1.5f);
	c.ripple       = EnvFloat("KYTY_HWRT_WATER_RIPPLE", c.ripple, 0.0f, 0.2f);
	c.strength     = EnvFloat("KYTY_HWRT_WATER_STRENGTH", c.strength, 0.0f, 2.0f);
	c.min_refl     = EnvFloat("KYTY_HWRT_WATER_MIN", c.min_refl, 0.0f, 0.9f);
	return c;
}

bool Skip(const char* why) {
	++g_w.skipped, g_w.skip_why = why;
	g_w.history_valid = false;
	g_w.have_prev     = false;
	static const char* said = nullptr;
	if (said != why) {
		said = why;
		std::printf("HWRTWATER: nothing this frame (%s)\n", why);
		std::fflush(stdout);
	}
	return false;
}

void Log(GraphicContext& graphics, int level) {
	const auto now = std::chrono::steady_clock::now();
	if (now - g_w.last_log < std::chrono::seconds(10)) return;
	const double secs = std::chrono::duration<double>(now - g_w.last_log).count();
	g_w.last_log      = now;
	(void)vmaInvalidateAllocation(graphics.allocator, g_w.stat_mem, 0, VK_WHOLE_SIZE);
	std::array<uint32_t, StatWords> cur {}, d {};
	for (uint32_t i = 0; i < StatWords; ++i) cur[i] = g_w.stat_mapped[i], d[i] = cur[i] - g_w.stat_last[i];
	g_w.stat_last = cur;
	uint64_t draws = 0, wframes = 0;
	uint32_t known = 0, built = 0;
	HwScene::WaterCounters(draws, known, built, wframes);
	const double px = std::max<double>(d[0], 1.0), rays = std::max<double>(d[3], 1.0), refl = std::max<double>(d[8], 1.0),
	             comp = std::max<double>(d[12], 1.0);
	std::printf("HWRTWATER: %.0f s (level %d, trace %ux%u): ran %llu of %llu frames (last skip: %s); water draws captured %llu "
	            "(%u water meshes, %u built), water TLAS %llu frames; water %.2f %% of the pixels, puddles %.2f %% (guide %llu "
	            "frames); rays %.2f %%: on-screen hit %.1f %%, sky %.1f %%, hidden / off-screen %.1f %%, miss %.1f %%; history kept "
	            "%.1f %% (disoccluded %.1f %%), fireflies clamped %.2f %%; composited (> 5 %%) %.2f %% of the pixels, mean weight "
	            "%.3f; water draws on the normal path (not captured) %llu; TLAS alignment %.1f %% of %u probes; camera still %llu frames; GPU %.3f ms (max %.3f)\n",
	            secs, level, g_w.trace.width, g_w.trace.height, static_cast<unsigned long long>(g_w.frames_ran),
	            static_cast<unsigned long long>(g_w.frames), g_w.skip_why != nullptr ? g_w.skip_why : "none",
	            static_cast<unsigned long long>(draws - g_w.last_draws), known, built,
	            static_cast<unsigned long long>(wframes - g_w.last_wframes), 100.0 * d[1] / px, 100.0 * d[2] / px,
	            static_cast<unsigned long long>(g_w.frames_guide), 100.0 * d[3] / px, 100.0 * d[4] / rays, 100.0 * d[5] / rays,
	            100.0 * d[6] / rays, 100.0 * d[7] / rays, 100.0 * d[9] / refl, 100.0 * d[10] / refl, 100.0 * d[11] / refl,
	            100.0 * d[13] / comp, d[13] > 0 ? d[14] / 1000.0 / d[13] : 0.0,
	            static_cast<unsigned long long>(g_normal_draws.load(std::memory_order_relaxed) - g_normal_last), d[16] > 0 ? 100.0 * d[17] / d[16] : 0.0, d[16],
	            static_cast<unsigned long long>(g_w.frames_still), g_w.samples > 0 ? g_w.sum_ms / g_w.samples : 0.0, g_w.max_ms);
	std::fflush(stdout);
	g_w.last_draws = draws, g_w.last_wframes = wframes;
	g_normal_last  = g_normal_draws.load(std::memory_order_relaxed);
	g_w.frames = g_w.frames_ran = g_w.frames_water = g_w.frames_guide = g_w.frames_still = 0;
	g_w.sum_ms = g_w.max_ms = 0.0, g_w.samples = 0;
	g_w.skip_why = nullptr;
}

} // namespace

int RtWaterLevel() {
	return 0; // (ds-rt-stable-1027) removed: the owner keeps only RT reflections, RT sun / lamp shadows and volumetric shafts
	static const int level = [] {
		const char* value = std::getenv("KYTY_HWRT_WATER");
		const int   q     = ::Libs::Graphics::FrameGen::ParseEnvLevel(value);
		if (q > 0) {
			std::printf("HWRTWATER: KYTY_HWRT_WATER=%s read (level %d)%s\n", value, q,
			            HwScene::Requested() ? "" : ": needs KYTY_HWRT=1 too: off");
			std::fflush(stdout);
		}
		return HwScene::Requested() ? q : 0;
	}();
	return level;
}

bool RtWaterIsWaterPs(uint64_t ps_hash) {
	if (ps_hash == 0 || RtWaterLevel() == 0) return false;
	const auto& l = WaterPs();
	for (uint32_t i = 0; i < l.count; ++i)
		if (l.hashes[i] == ps_hash) return true;
	return false;
}

void RtWaterNoteNormalDraw() { g_normal_draws.fetch_add(1, std::memory_order_relaxed); }

void RtWaterNoteState(uint64_t vs_hash, uint64_t ps_hash, uint32_t colors, uint32_t depth_width, bool depth_write, int topology) {
	static std::mutex                                  mutex;
	static std::vector<std::pair<uint64_t, uint64_t>> seen;
	std::lock_guard lock(mutex);
	const auto      key = std::make_pair(vs_hash, ps_hash);
	if (seen.size() >= 32 || std::find(seen.begin(), seen.end(), key) != seen.end()) return;
	seen.push_back(key);
	std::printf("HWRTWATER: water draw state VS %016llx PS %016llx: %u colours, depth %u wide, depth writes %d, topology %d -> %s\n",
	            static_cast<unsigned long long>(vs_hash), static_cast<unsigned long long>(ps_hash), colors, depth_width,
	            depth_write ? 1 : 0, topology,
	            depth_width >= 1280 && topology == static_cast<int>(vk::PrimitiveTopology::eTriangleList) ? "captured" : "not captured");
	std::fflush(stdout);
}

bool RtWaterApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int level = RtWaterLevel();
	if (level == 0) return false;
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr) return false;
	++g_w.frames;
	if (g_w.pipeline != nullptr) Log(*graphics, level);
	if (!HwScene::Active()) return Skip("KYTY_HWRT inactive: the device has no ray queries");
	if (VideoOut::MoviePlaying()) return Skip("a movie");
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || color.extent.width < 16 || color.extent.height < 16 ||
	    inputs.depth.extent != color.extent)
		return Skip("missing color / depth");
	bool built = false;
	vk::AccelerationStructureKHR tlas = HwScene::SceneTlas(&built);
	if (tlas == nullptr) return Skip("no scene TLAS built yet");
	std::array<float, 16> vp {};
	if (!HwScene::SceneCamera(vp)) return Skip("no camera matrix captured yet");
	double vpd[16], ivp[16];
	for (int i = 0; i < 16; ++i) vpd[i] = vp[i];
	if (!Inverse4(vpd, ivp)) return Skip("the camera matrix is degenerate");
	double eye[4] {};
	for (int i = 0; i < 4; ++i) eye[i] = ivp[i * 4 + 2];
	if (!(std::abs(eye[3]) > 1e-12)) return Skip("the camera matrix has no perspective");
	const double cam[3] {eye[0] / eye[3], eye[1] / eye[3], eye[2] / eye[3]};
	if (!EnsurePipeline(*graphics) || !EnsureImages(*graphics, color.extent, level)) return Skip("could not create its pipeline or images");
	FreeRetired(*graphics);
	ReadTiming(*graphics);

	HwScene::WaterScene water {};
	const bool have_water = HwScene::CurrentWaterScene(water) && water.tlas != nullptr && water.instances > 0;
	RtReflGuide guide;
	const bool  have_guide = RtReflGetGuide(guide) && guide.view != nullptr;
	static const auto cfg  = Cfg(level);
	static const float flipy = EnvFloat("KYTY_HWRT_WATER_FLIPY", 1.0f, 0.0f, 1.0f);
	if (!have_water && !(have_guide && cfg.puddle_share > 0.0f)) return Skip("no water drawn and no game SSR guide this frame");

	// Still camera: this frame's and the last frame's matrices agree (the projection's jitter is far below the bound).
	bool still = false;
	if (g_w.have_prev) {
		double scale = 0.0, diff = 0.0;
		for (int i = 0; i < 16; ++i) scale = std::max(scale, std::abs(vpd[i])), diff = std::max(diff, std::abs(vpd[i] - g_w.prev_vp[i]));
		still = diff <= 4e-3 * std::max(scale, 1e-9);
	}
	const bool history = g_w.history_valid && g_w.have_prev && !inputs.reset;
	++g_w.frames_ran;
	if (have_water) ++g_w.frames_water;
	if (have_guide) ++g_w.frames_guide;
	if (still) ++g_w.frames_still;
	if (!g_w.said_on) {
		g_w.said_on = true;
		std::printf("HWRTWATER: on (level %d %s: trace %ux%u of %ux%u, rays <= %g m, strength %g, water minimum reflectance %g, "
		            "ripple %g, puddles %g from the game's SSR confidence %g; water shaders:",
		            level, level == 1 ? "Bajo" : level == 2 ? "Medio" : "Alto", g_w.trace.width, g_w.trace.height, color.extent.width,
		            color.extent.height, cfg.dist, cfg.strength, cfg.min_refl, cfg.ripple, cfg.puddle_share, cfg.puddle_conf);
		const auto& l = WaterPs();
		for (uint32_t i = 0; i < l.count; ++i) std::printf(" %016llx", static_cast<unsigned long long>(l.hashes[i]));
		std::printf(")\n");
		std::fflush(stdout);
	}

	// This frame's parameters.
	const uint32_t slot = static_cast<uint32_t>(g_w.frame % Slots);
	{
		std::array<float, SlotBytes / 4> f {};
		for (int i = 0; i < 16; ++i) f[i] = static_cast<float>(ivp[i]);
		for (int i = 0; i < 16; ++i) f[16 + i] = vp[i];
		for (int i = 0; i < 16; ++i) f[32 + i] = static_cast<float>(g_w.have_prev ? g_w.prev_vp[i] : vpd[i]);
		f[48] = static_cast<float>(cam[0]), f[49] = static_cast<float>(cam[1]), f[50] = static_cast<float>(cam[2]), f[51] = 1.0f;
		f[52] = flipy > 0.5f ? -1.0f : 1.0f, f[53] = cfg.dist, f[54] = cfg.puddle_conf, f[55] = cfg.ripple;
		f[56] = cfg.strength, f[57] = cfg.min_refl, f[58] = have_guide ? cfg.puddle_share : 0.0f, f[ExposureFallbackIndex] = 1.0f;
		uint32_t words[4] {static_cast<uint32_t>(water.materials), static_cast<uint32_t>(water.materials >> 32u),
		                   have_water ? water.instances : 0u, 0u};
		std::memcpy(&f[60], words, sizeof(words));
		std::memcpy(g_w.param_mapped + slot * SlotBytes, f.data(), SlotBytes);
		(void)vmaFlushAllocation(graphics->allocator, g_w.param_mem, slot * SlotBytes, SlotBytes);
	}
	for (int i = 0; i < 16; ++i) g_w.prev_vp[i] = vpd[i];
	g_w.have_prev = true;

	Push push {};
	push.size[0] = static_cast<float>(color.extent.width), push.size[1] = static_cast<float>(color.extent.height);
	push.size[2] = static_cast<float>(g_w.trace.width), push.size[3] = static_cast<float>(g_w.trace.height);
	push.misc[1] = static_cast<int32_t>(g_w.frame & 0x7fffffff);
	push.misc[2] = level;
	push.misc[3] = (history ? 1 : 0) | (have_water ? 2 : 0) | (have_guide ? 4 : 0) | (still ? 8 : 0);
	const bool game_exposure = inputs.exposure_buffer != nullptr && inputs.exposure_range >= sizeof(float);
	push.misc2[0] = game_exposure ? static_cast<int32_t>(inputs.exposure_index) : static_cast<int32_t>(ExposureFallbackIndex);

	const auto entry = static_cast<uint32_t>(g_w.written % 8);
	const bool timed = g_w.pool != nullptr && g_w.written < g_w.read + 8;
	if (timed) {
		command.resetQueryPool(g_w.pool, entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_w.pool, entry * 2);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	if (!g_w.layouts_ready) {
		Img* all[] {&g_w.out, &g_w.raw, &g_w.raw_aux, &g_w.hist[0], &g_w.hist[1], &g_w.aux[0], &g_w.aux[1]};
		for (Img* img: all) ToGeneral(command, img->image);
		g_w.layouts_ready = true, g_w.history_valid = false;
		push.misc[3] &= ~1;
	}
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_w.pipeline);

	const auto     gen = vk::ImageLayout::eGeneral;
	const uint32_t cur = static_cast<uint32_t>(g_w.frame & 1u), prev = cur ^ 1u;
	const vk::DescriptorBufferInfo param_info {vk::Buffer(g_w.param_buf), vk::DeviceSize {slot} * SlotBytes, SlotBytes};
	const vk::DescriptorBufferInfo stat_info {vk::Buffer(g_w.stat_buf), 0, StatWords * 4};
	const vk::DescriptorBufferInfo exposure_info = game_exposure
	                                                   ? vk::DescriptorBufferInfo {inputs.exposure_buffer, inputs.exposure_base, inputs.exposure_range}
	                                                   : param_info;
	vk::AccelerationStructureKHR water_tlas = have_water ? water.tlas : tlas; // (a valid descriptor; flag 2 off: unused)
	vk::WriteDescriptorSetAccelerationStructureKHR scene_write {}, water_write {};
	scene_write.accelerationStructureCount = 1;
	scene_write.pAccelerationStructures    = &tlas;
	water_write.accelerationStructureCount = 1;
	water_write.pAccelerationStructures    = &water_tlas;
	const vk::DescriptorImageInfo color_info {g_w.linear, color.view, color.layout};
	const vk::DescriptorImageInfo depth_info {g_w.point, inputs.depth.view, inputs.depth.layout};
	const vk::DescriptorImageInfo guide_info = have_guide ? vk::DescriptorImageInfo {g_w.linear, guide.view, vk::ImageLayout::eShaderReadOnlyOptimal}
	                                                      : vk::DescriptorImageInfo {g_w.linear, g_w.raw.view, gen};
	const auto run = [&](int mode, const Img& src, const Img& src_aux, const Img& out, const Img& out_aux, uint32_t w, uint32_t h) {
		vk::DescriptorImageInfo infos[Bindings] {};
		infos[0]  = color_info;
		infos[1]  = depth_info;
		infos[2]  = {g_w.linear, g_w.hist[prev].view, gen};
		infos[3]  = {g_w.point, g_w.aux[prev].view, gen};
		infos[4]  = {g_w.point, src.view, gen};
		infos[5]  = {g_w.point, src_aux.view, gen};
		infos[6]  = {nullptr, out.view, gen};
		infos[7]  = {nullptr, out_aux.view, gen};
		infos[12] = guide_info;
		vk::WriteDescriptorSet writes[Bindings] {};
		for (uint32_t i = 0; i < Bindings; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = DescType(i);
			if (i == 8) writes[i].pBufferInfo = &param_info;
			else if (i == 11) writes[i].pBufferInfo = &stat_info;
			else if (i == 13) writes[i].pBufferInfo = &exposure_info;
			else if (i == 9) writes[i].pNext = &scene_write;
			else if (i == 10) writes[i].pNext = &water_write;
			else writes[i].pImageInfo = &infos[i];
		}
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_w.layout, 0, Bindings, writes);
		push.misc[0] = mode;
		command.pushConstants(g_w.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch((w + 7) / 8, (h + 7) / 8, 1);
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	};
	const uint32_t tw = g_w.trace.width, th = g_w.trace.height;
	// (mode 0 reads no src; mode 2 writes no aux: the bindings point at images that mode leaves alone)
	run(0, g_w.hist[cur], g_w.aux[cur], g_w.raw, g_w.raw_aux, tw, th);
	run(1, g_w.raw, g_w.raw_aux, g_w.hist[cur], g_w.aux[cur], tw, th);
	run(2, g_w.hist[cur], g_w.aux[cur], g_w.out, g_w.raw_aux, color.extent.width, color.extent.height);
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead);
	if (timed) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_w.pool, entry * 2 + 1);
		++g_w.written;
	}
	++g_w.frame;
	g_w.history_valid   = true;
	inputs.color.image  = vk::Image(g_w.out.image);
	inputs.color.view   = g_w.out.view;
	inputs.color.layout = gen;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc |
	                      vk::ImageUsageFlagBits::eStorage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}

} // namespace Libs::Graphics::FrameGen
