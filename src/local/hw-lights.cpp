// KYTY_HWRT_LIGHTS=0|1|2|3: ray-traced local-light shadows (hw-lights.h). Recorded in FrameGen::Upscale right after
// CShadowApply, on the pre-exposure HDR color, before the upscaler; the scene TLAS (hwrt-scene.cpp SceneTlas) was built
// earlier on the same command stream (after the game's TAA dispatch) with a barrier to compute ray queries.
//
// Lights: rt-local-lights.cpp's shadow-projection apexes (the game's own spot / cube shadow matrices read by its
// lighting: position, forward, cone, reach exact), up to 32 nearest the camera, faded in over 30 flips and out after 60
// flips unseen. Their colour and intensity are not known (the light list's layout is not proven), so the result is a
// ratio correction of the lit color, not a per-light visibility term: color *= 1 - min(s * share * occ, max_dark), occ the
// lights' weighted occlusion by the TLAS (rays), share = W / (W + ambient) the analytic estimate of the local lights' part of
// the pixel's light. Darken only; a reset / cut / first frame darkens nothing (the history count ramps it in over ~6 frames).
//
// Options: KYTY_HWRT_LIGHTS_STRENGTH (1.0), _MAX (max darkening, 0.85), _AMBIENT (0.03: the share's knee), _SOFT (light
// source radius, 0.08 m), _SKIP (rays end this short of the light: its lamp / brazier mesh, 0.3 m), _DIST (lights and
// pixels within this of the camera, 60 m), _FLIPY (1: row 0 = NDC y +1, default), _GUARD (a 1 s window whose mean
// darkening over the screen exceeds it turns the pass off for 5 s, logged; 0.25).
// Every 10 s: "HWRTLIGHTS: stats" with lights used, darkened pixels, their mean darkening and the GPU ms.
#include "local/hw-lights.h"
#include "local/env-level.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/hwlights_spv.h"
#include "local/hwrt-scene.h"
#include "local/rt-local-lights.h"
#include "local/rt-flicker.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace Libs::Graphics::FrameGen {
extern std::mutex            g_last_vp_mutex;
extern std::array<float, 16> g_last_vp;
extern bool                  g_have_last_vp;
namespace {

constexpr uint32_t Bindings   = 12;
constexpr uint32_t MaxLights  = 32;
constexpr uint32_t SlotBytes  = 2048;
constexpr uint32_t Slots      = 4;

struct HwlPush {
	float   size[4];
	float   motion[4];
	int32_t misc[4];
};
static_assert(sizeof(HwlPush) == 48);

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
	Img                     out, raw, tmp_a, tmp_b;
	std::array<Img, 2>      gbuf, hist;
	bool                    layouts_ready = false, history_valid = false;
	std::array<std::pair<Img, uint64_t>, 16> retired {};
	VkBuffer                light_buf = VK_NULL_HANDLE, stat_buf = VK_NULL_HANDLE;
	VmaAllocation           light_mem = nullptr, stat_mem = nullptr;
	uint8_t*                light_mapped = nullptr;
	volatile uint32_t*      stat_mapped  = nullptr;
	uint64_t                frame = 0;
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms = 0.0, max_ms = 0.0;
	uint32_t                samples = 0;
	std::array<uint32_t, 32> stat_last {}, guard_last {};
	uint64_t                frames = 0, frames_lit = 0, lights_sum = 0, no_tlas = 0, stale_tlas = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now(), last_guard = last_log, guard_until {};
	bool                    said_on = false;
	bool                    idle = false; // (ds-rt-shadow) no pixel traced in the last 1 s window: probe only
	uint64_t                idle_frames = 0;
	uint32_t                guard_last_w = 0;
	uint64_t                skipped = 0;
	const char*             skip_why = nullptr;
	std::chrono::steady_clock::time_point last_skip_log = std::chrono::steady_clock::now();
} g_hl;

// (lampshadow-1027) The lights as sent: tracked across frames, their position / forward / reach eased toward each new
// scan (a flame-carried light or a rescan's jitter moves the shadow smoothly, not in 30-flip steps) and their weight
// ramped in / out (vis) when one enters or leaves the nearest 32: the visibility itself is never smoothed here.
struct Track {
	float    pos[3] {}, fwd[3] {}, tan_half = 0, far_d = 0, fade = 0, vis = 0, d = 0;
	uint32_t miss = 0;             // (ds-rt-shadow) frames since a scan last had it
	float    cap_w = 0;            // (ds-rt-shadow) its weight in the per-level light cap, ramped (no pop at the cap's edge)
	bool     point = false, used = false, hit = false, in_cap = false;
};
std::array<Track, 64> g_tracks {};
uint64_t              g_track_jumps = 0;

// (ds-rt-shadow) The scan's lights matched to the tracks one to one, nearest pairs first (a global greedy match). The
// per-light greedy in scan order let a light take its neighbour's track when the scan listed it first (two lamps within
// 0.75 m, a torch and its brazier): both tracks jumped, "light rescans moved 176" in 10 s at the 10-27 lamps, the one
// window with lit pixels, its RTFLICKER 18 % of the touched points, max jump 143 %.
void UpdateTracks(const std::vector<const RtLocal::LiveLight*>& in, const double* cam, bool snap) {
	for (auto& t: g_tracks) t.hit = false;
	struct Pair {
		float    d2;
		uint32_t li, ti;
	};
	std::vector<Pair> pairs;
	for (uint32_t li = 0; li < in.size(); ++li) {
		const auto* l   = in[li];
		const float tol = std::max(0.75f, 0.1f * l->far_d);
		for (uint32_t ti = 0; ti < g_tracks.size(); ++ti) {
			const auto& t = g_tracks[ti];
			if (!t.used || t.point != l->point) continue;
			const float dx = t.pos[0] - l->pos[0], dy = t.pos[1] - l->pos[1], dz = t.pos[2] - l->pos[2];
			const float d2 = dx * dx + dy * dy + dz * dz;
			if (d2 <= tol * tol) pairs.push_back({d2, li, ti});
		}
	}
	std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.d2 < b.d2; });
	std::vector<int32_t> match(in.size(), -1);
	for (const auto& p: pairs)
		if (match[p.li] < 0 && !g_tracks[p.ti].hit) match[p.li] = static_cast<int32_t>(p.ti), g_tracks[p.ti].hit = true;
	for (auto& t: g_tracks) t.hit = false;
	for (uint32_t li = 0; li < in.size(); ++li) {
		const auto* l        = in[li];
		const float fade_out = 1.0f; // (ds-rt-shadow 10-28: the miss ramp (vis) fades a light that is no longer scanned)
		const float fade_in  = std::clamp(static_cast<float>(l->since) / 30.0f, 0.0f, 1.0f);
		const float fade     = fade_out * fade_in;
		Track*      best     = match[li] >= 0 ? &g_tracks[match[li]] : nullptr;
		float       bd       = 0.0f;
		if (best != nullptr) {
			const float dx = best->pos[0] - l->pos[0], dy = best->pos[1] - l->pos[1], dz = best->pos[2] - l->pos[2];
			bd             = dx * dx + dy * dy + dz * dz;
			if (bd > 0.09f) ++g_track_jumps, RtFlicker::Cause(RtFlicker::Lights, "light rescans moved > 0.3 m");
		} else {
			for (auto& t: g_tracks)
				if (!t.used) { best = &t; break; }
			if (best == nullptr) continue;
			*best = {};
			best->used = true, best->point = l->point;
			for (int k = 0; k < 3; ++k) best->pos[k] = l->pos[k], best->fwd[k] = l->fwd[k];
			best->tan_half = l->tan_half, best->far_d = l->far_d, best->fade = fade;
			RtFlicker::Cause(RtFlicker::Lights, "lights entered");
		}
		// (all2-1027) a dead band: a rescan within 0.3 m (the apex's jitter) does not move the light at all; a real move
		// is followed slowly (~1 s); the inputs are constant while the light holds still
		const float a = snap ? 1.0f : bd < 0.09f ? 0.0f : 0.03f;
		for (int k = 0; k < 3; ++k) best->pos[k] += (l->pos[k] - best->pos[k]) * a, best->fwd[k] += (l->fwd[k] - best->fwd[k]) * a;
		const float fl = std::sqrt(best->fwd[0] * best->fwd[0] + best->fwd[1] * best->fwd[1] + best->fwd[2] * best->fwd[2]);
		if (fl > 1e-6f)
			for (float& v: best->fwd) v /= fl;
		best->tan_half += (l->tan_half - best->tan_half) * a;
		best->far_d += (l->far_d - best->far_d) * a;
		best->fade += (fade - best->fade) * (snap ? 1.0f : 0.05f);
		best->hit = true;
	}
	for (auto& t: g_tracks) {
		if (!t.used) continue;
		// (ds-rt-shadow 10-28) a light missing from a scan keeps its shadow for 90 frames (the scans drop and re-find the
		// same lamps: entered / left 12-24 per 10 s), then fades over ~0.5 s; a new one fades in over ~0.5 s
		t.miss = t.hit ? 0u : t.miss + 1u;
		const bool present = t.hit || t.miss < 90u;
		t.vis = snap ? (present ? 1.0f : 0.0f) : std::clamp(t.vis + (present ? 1.0f / 30.0f : -1.0f / 30.0f), 0.0f, 1.0f);
		const double dx = t.pos[0] - cam[0], dy = t.pos[1] - cam[1], dz = t.pos[2] - cam[2];
		t.d = static_cast<float>(std::sqrt(dx * dx + dy * dy + dz * dz));
		if (!present && t.vis <= 0.0f) {
			RtFlicker::Cause(RtFlicker::Lights, "lights left");
			t = {};
		}
	}
}

// (ds-rt-shadow) The per-level cap with hysteresis: the lights in it last frame keep their place unless another is 25 %
// nearer, and a light entering / leaving it ramps its weight over 20 frames. A hard nearest-N cut ("lights 16.00 a frame"
// at the 10-27 lamps: the cap full) dropped and added whole shadows as the eased distances crossed.
void SelectCapped(std::vector<const Track*>& sent, size_t cap, bool snap) {
	std::vector<Track*> cand;
	for (auto& t: g_tracks)
		if (t.used && t.vis > 0.0f && t.fade > 0.0f) cand.push_back(&t);
	std::sort(cand.begin(), cand.end(), [](const Track* a, const Track* b) {
		return (a->in_cap ? a->d * 0.5f : a->d) < (b->in_cap ? b->d * 0.5f : b->d); // (10-28: twice as near to take a place)
	});
	for (size_t i = 0; i < cand.size(); ++i) {
		Track* t        = cand[i];
		const bool was  = t->in_cap;
		t->in_cap       = i < cap;
		if (was != t->in_cap && !snap) RtFlicker::Cause(RtFlicker::Lights, "cap swaps");
		t->cap_w = snap ? (t->in_cap ? 1.0f : 0.0f) : std::clamp(t->cap_w + (t->in_cap ? 1.0f / 30.0f : -1.0f / 30.0f), 0.0f, 1.0f);
	}
	for (auto& t: g_tracks)
		if (!t.used || t.vis <= 0.0f || t.fade <= 0.0f) t.in_cap = false;
	sent.clear();
	for (const Track* t: cand)
		if (t->cap_w > 0.0f && sent.size() < MaxLights) sent.push_back(t);
	// (ds-rt-shadow 10-28) sent in a fixed order (the track's slot), not by distance: the trace picks a light per pixel from
	// the weights' running sum in list order, so a reordering (a light entering, two distances crossing) re-picked the light
	// of every pixel at once and its 1-sample occlusion flipped (20261028-001843: max jump 552-567 % on cap swaps / entries)
	std::sort(sent.begin(), sent.end(), [](const Track* a, const Track* b) { return a < b; });
}

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
	for (auto& slot: g_hl.retired)
		if (slot.first.image == VK_NULL_HANDLE) {
			slot = {img, g_hl.frame};
			img  = {};
			return;
		}
	graphics.device.waitIdle(); // (all 16 slots busy: only on repeated resizes)
	DestroyImage(graphics, img);
}
void FreeRetired(GraphicContext& graphics) {
	for (auto& slot: g_hl.retired)
		if (slot.first.image != VK_NULL_HANDLE && g_hl.frame >= slot.second + 16) DestroyImage(graphics, slot.first), slot = {};
}

vk::DescriptorType DescType(uint32_t i) {
	return i == 9 || i == 11 ? vk::DescriptorType::eStorageBuffer
	       : i == 10         ? vk::DescriptorType::eAccelerationStructureKHR
	       : i == 7 || i == 8 ? vk::DescriptorType::eStorageImage
	                          : vk::DescriptorType::eCombinedImageSampler;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_hl.pipeline != nullptr) return true;
	if (g_hl.failed) return false;
	g_hl.failed       = true;
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
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_hl.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(HwlPush)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_hl.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_hl.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(HWLIGHTS_SPV);
	module_info.pCode       = HWLIGHTS_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_hl.layout;
	const auto created         = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_hl.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_hl.pipeline = nullptr;
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
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g_hl.point : &g_hl.linear) != vk::Result::eSuccess) return false;
	}
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = SlotBytes * Slots;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_hl.light_buf, &g_hl.light_mem, &mapped) != VK_SUCCESS) {
		g_hl.light_buf = VK_NULL_HANDLE;
		return false;
	}
	g_hl.light_mapped = static_cast<uint8_t*>(mapped.pMappedData);
	buffer.size       = 128;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_hl.stat_buf, &g_hl.stat_mem, &mapped) != VK_SUCCESS) {
		g_hl.stat_buf = VK_NULL_HANDLE;
		return false;
	}
	g_hl.stat_mapped = static_cast<volatile uint32_t*>(mapped.pMappedData);
	for (int i = 0; i < 32; ++i) g_hl.stat_mapped[i] = 0;
	(void)vmaFlushAllocation(graphics.allocator, g_hl.stat_mem, 0, VK_WHOLE_SIZE);
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = 8 * 2;
	if (device.createQueryPool(&query, nullptr, &g_hl.pool) != vk::Result::eSuccess) g_hl.pool = nullptr;
	g_hl.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g_hl.failed    = false;
	return true;
}

bool EnsureImages(GraphicContext& graphics, vk::Extent2D extent, int level) {
	// (all2-1027) the trace's pixel budget, whatever the render size (a 2880x1620 render traced at full size cost 9.7 ms):
	// Bajo / Medio ~0.23 M (640x360 at 16:9), Alto ~0.92 M (1280x720); never above the render size
	const double budget = level >= 3 ? 921600.0 : 230400.0;
	const double scale  = std::min(1.0, std::sqrt(budget / std::max(1.0, double(extent.width) * double(extent.height))));
	const vk::Extent2D trace {std::max(static_cast<uint32_t>(extent.width * scale), 1u), std::max(static_cast<uint32_t>(extent.height * scale), 1u)};
	if (g_hl.out.image != VK_NULL_HANDLE && g_hl.extent == extent && g_hl.trace == trace) return true;
	Img* all[] {&g_hl.out, &g_hl.raw, &g_hl.tmp_a, &g_hl.tmp_b, &g_hl.gbuf[0], &g_hl.gbuf[1], &g_hl.hist[0], &g_hl.hist[1]};
	for (Img* img: all) Retire(graphics, *img);
	g_hl.extent = extent, g_hl.trace = trace;
	g_hl.layouts_ready = false, g_hl.history_valid = false;
	bool ok = MakeImage(graphics, extent, g_hl.out);
	for (Img* img: all)
		if (img != &g_hl.out) ok = ok && MakeImage(graphics, trace, *img);
	if (!ok)
		for (Img* img: all) Retire(graphics, *img);
	return ok;
}

void ReadTiming(GraphicContext& graphics) {
	if (g_hl.pool == nullptr) return;
	while (g_hl.read + 4 <= g_hl.written) {
		const auto              entry = static_cast<uint32_t>(g_hl.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (graphics.device.getQueryPoolResults(g_hl.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		if (stamps[1] > stamps[0]) {
			const double ms = static_cast<double>(stamps[1] - stamps[0]) * g_hl.period_ns / 1e6;
			g_hl.sum_ms += ms, g_hl.max_ms = std::max(g_hl.max_ms, ms);
			++g_hl.samples;
		}
		++g_hl.read;
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

} // namespace

int HwLightsLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_HWRT_LIGHTS");
		const int   q     = ::Libs::Graphics::FrameGen::ParseEnvLevel(value);
		if (q > 0) {
			std::printf("HWRTLIGHTS: KYTY_HWRT_LIGHTS=%s read (level %d)%s\n", value, q,
			            HwScene::Requested() ? "" : ": needs KYTY_HWRT=1 too: off");
			std::fflush(stdout);
		}
		return HwScene::Requested() ? q : 0;
	}();
	return level;
}

bool HwLightsApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int level = HwLightsLevel();
	if (level == 0) return false;
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr) return false;
	{
		// (every 10 s, while frames are skipped before the pass runs: how many and why; the stats line below covers the rest)
		const auto t = std::chrono::steady_clock::now();
		if (t - g_hl.last_skip_log >= std::chrono::seconds(10)) {
			if (g_hl.skipped > 0) {
				std::printf("HWRTLIGHTS: 10 s: %llu frames without the pass (last: %s)\n", static_cast<unsigned long long>(g_hl.skipped),
				            g_hl.skip_why != nullptr ? g_hl.skip_why : "?");
				std::fflush(stdout);
			}
			g_hl.skipped = 0, g_hl.last_skip_log = t;
		}
	}
	const auto said_once = [](const char* why) {
		static const char* said = nullptr;
		++g_hl.skipped, g_hl.skip_why = why;
		if (said != why) {
			said = why;
			std::printf("HWRTLIGHTS: nothing this frame (%s)\n", why);
			std::fflush(stdout);
		}
		return false;
	};
	if (!HwScene::Active()) return said_once("KYTY_HWRT inactive: the device has no ray queries");
	if (VideoOut::MoviePlaying()) return g_hl.history_valid = false, false;
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || color.extent.width < 16 || color.extent.height < 16 ||
	    inputs.depth.extent != color.extent)
		return said_once("missing color / depth");
	bool built = false;
	vk::AccelerationStructureKHR tlas = HwScene::SceneTlas(&built);
	if (tlas == nullptr) {
		++g_hl.no_tlas, g_hl.history_valid = false;
		return said_once("no scene TLAS built yet");
	}
	if (!built) ++g_hl.stale_tlas;
	// The camera: the scene capture's own (the TLAS's world), else FrameGen's G-buffer one.
	std::array<float, 16> vp {};
	if (!HwScene::SceneCamera(vp)) {
		std::lock_guard lock(g_last_vp_mutex);
		if (!g_have_last_vp) return said_once("no camera matrix captured yet");
		vp = g_last_vp;
	}
	double vpd[16], ivp[16];
	for (int i = 0; i < 16; ++i) vpd[i] = vp[i];
	if (!Inverse4(vpd, ivp)) return said_once("the camera matrix is degenerate");
	// The camera: the clip-space point x = y = w = 0 (row-major M, clip = M * world).
	double eye[4] {};
	for (int i = 0; i < 4; ++i) eye[i] = ivp[i * 4 + 2];
	if (!(std::abs(eye[3]) > 1e-12)) return said_once("the camera matrix has no perspective");
	const double cam[3] {eye[0] / eye[3], eye[1] / eye[3], eye[2] / eye[3]};

	if (!EnsurePipeline(*graphics) || !EnsureImages(*graphics, color.extent, level)) return said_once("could not create its pipeline or images");
	FreeRetired(*graphics);
	ReadTiming(*graphics);

	static const float strength = EnvFloat("KYTY_HWRT_LIGHTS_STRENGTH", 1.0f, 0.0f, 2.0f);
	static const float max_dark = EnvFloat("KYTY_HWRT_LIGHTS_MAX", 0.85f, 0.0f, 0.95f);
	static const float ambient  = EnvFloat("KYTY_HWRT_LIGHTS_AMBIENT", 0.5f, 0.001f, 100.0f);
	static const float soft     = EnvFloat("KYTY_HWRT_LIGHTS_SOFT", 0.12f, 0.0f, 2.0f);
	static const float skip     = EnvFloat("KYTY_HWRT_LIGHTS_SKIP", 0.3f, 0.0f, 5.0f);
	static const float max_dist = EnvFloat("KYTY_HWRT_LIGHTS_DIST", 60.0f, 5.0f, 500.0f);
	static const float flipy    = EnvFloat("KYTY_HWRT_LIGHTS_FLIPY", 1.0f, 0.0f, 1.0f);
	static const float guard    = EnvFloat("KYTY_HWRT_LIGHTS_GUARD", 0.25f, 0.0f, 1.0f);

	// The lights: seen within 120 flips, by >= 2 scans; nearest first, up to 32.
	std::vector<RtLocal::LiveLight> live;
	RtLocal::LiveLights(live, 120, 2);
	struct Pick {
		float d;
		const RtLocal::LiveLight* l;
	};
	std::vector<Pick> picks;
	for (const auto& l: live) {
		if (!(l.far_d > 0.5f && l.far_d < 100.0f)) continue; // (a light's reach; cameras and cascades have none like it)
		const double dx = l.pos[0] - cam[0], dy = l.pos[1] - cam[1], dz = l.pos[2] - cam[2];
		const double d  = std::sqrt(dx * dx + dy * dy + dz * dz);
		if (d - l.far_d > max_dist) continue;
		picks.push_back({static_cast<float>(d), &l});
	}
	std::sort(picks.begin(), picks.end(), [](const Pick& a, const Pick& b) { return a.d < b.d; });
	if (picks.size() > MaxLights) picks.resize(MaxLights);
	std::vector<const RtLocal::LiveLight*> in;
	for (const auto& p: picks) in.push_back(p.l);
	// (ds-rt-shadow) snapped only on a camera cut: a frame with no light (the history dropped) no longer snaps the next
	// one's weights to full at once (a lamp's shadow popped in when it was the only light and came back into the scan)
	const bool snap = inputs.reset;
	UpdateTracks(in, cam, snap);
	std::vector<const Track*> sent;
	// (all2-1027) per level: Bajo 8, Medio 12, Alto 16 lights (the per-pixel loop and the picks' spread)
	const size_t cap = level == 1 ? 8 : level == 2 ? 12 : 16;
	SelectCapped(sent, cap, snap);

	// The guard: a 1 s window whose mean darkening over the screen passes KYTY_HWRT_LIGHTS_GUARD turns the pass off for 5 s.
	const auto now = std::chrono::steady_clock::now();
	if (now - g_hl.last_guard >= std::chrono::seconds(1)) {
		g_hl.last_guard = now;
		(void)vmaInvalidateAllocation(graphics->allocator, g_hl.stat_mem, 0, VK_WHOLE_SIZE);
		std::array<uint32_t, 32> cur {};
		for (int i = 0; i < 32; ++i) cur[i] = g_hl.stat_mapped[i];
		const double px = cur[0] - g_hl.guard_last[0], dk = cur[2] - g_hl.guard_last[2];
		g_hl.guard_last = cur;
		const double mean = px > 0 ? dk / 1024.0 / px : 0.0;
		// (ds-rt-shadow 10-28) idle when no trace pixel had any light weight in the last second (s[8], the trace's W > 0
		// samples): the lights cover nothing on screen; the pass then only traces every 15th frame, its color left as it
		// is (invisible, ~0 cost), until a weight shows up again
		const bool was_idle = g_hl.idle;
		g_hl.idle           = cur[8] == g_hl.guard_last_w;
		g_hl.guard_last_w   = cur[8];
		if (was_idle != g_hl.idle) RtFlicker::Cause(RtFlicker::Lights, g_hl.idle ? "went idle (no light weight on screen)" : "woke up");
		if (guard > 0.0f && mean > guard && now >= g_hl.guard_until) {
			g_hl.guard_until = now + std::chrono::seconds(5);
			std::printf("HWRTLIGHTS: guard: the screen darkened %.1f %% on average over 1 s (> %.0f %%): off for 5 s\n", 100.0 * mean,
			            100.0 * guard);
			std::fflush(stdout);
		}
	}
	const bool guarded = now < g_hl.guard_until;

	// The log, every 10 s.
	++g_hl.frames;
	g_hl.lights_sum += sent.size();
	if (!sent.empty()) ++g_hl.frames_lit;
	if (now - g_hl.last_log >= std::chrono::seconds(10)) {
		g_hl.last_log = now;
		(void)vmaInvalidateAllocation(graphics->allocator, g_hl.stat_mem, 0, VK_WHOLE_SIZE);
		std::array<uint32_t, 32> cur {}, d {};
		for (int i = 0; i < 32; ++i) cur[i] = g_hl.stat_mapped[i], d[i] = cur[i] - g_hl.stat_last[i];
		g_hl.stat_last = cur;
		const double px = std::max<double>(d[0], 1.0), dp = d[1], dk = d[2];
		std::printf("HWRTLIGHTS: stats (level %d, trace %ux%u) lights %.2f a frame (%llu/%llu frames with any; %zu known now), darkened "
		            "%.1f %% of pixels by %.1f %% on average (%.2f %% of the screen's light), TLAS stale %llu / none %llu frames, GPU %.3f ms (max %.3f)\n",
		            level, g_hl.trace.width, g_hl.trace.height, static_cast<double>(g_hl.lights_sum) / std::max<double>(g_hl.frames, 1.0),
		            static_cast<unsigned long long>(g_hl.frames_lit), static_cast<unsigned long long>(g_hl.frames), live.size(),
		            100.0 * dp / px, dp > 0 ? 100.0 * dk / 1024.0 / dp : 0.0, 100.0 * dk / 1024.0 / px,
		            static_cast<unsigned long long>(g_hl.stale_tlas), static_cast<unsigned long long>(g_hl.no_tlas),
		            g_hl.samples > 0 ? g_hl.sum_ms / g_hl.samples : 0.0, g_hl.max_ms);
		// (rays: hit rate, hits at the light's own end; the camera -> pixel probe's agreement with the TLAS; lit pixels'
		// mean history count and filtered occlusion)
		std::printf("HWRTLIGHTS: rays: %.1f %% of traced shadow rays hit (%u sampled; %.1f %% of hits within 0.5 m of the light), "
		            "TLAS alignment %.1f %% (%u camera->pixel probes hit at the pixel's depth; >~85 %%: the depth's world is the TLAS's), "
		            "lit pixels %.1f %% (W > 0), their history %.1f frames, occlusion %.1f %%; hits at the receiver (< 0.15 m) %.1f %%; "
		            "light rescans moved %llu\n",
		            d[3] > 0 ? 100.0 * d[4] / d[3] : 0.0, d[3], d[4] > 0 ? 100.0 * d[5] / d[4] : 0.0, d[7] > 0 ? 100.0 * d[6] / d[7] : 0.0, d[7],
		            100.0 * d[9] / px, d[9] > 0 ? static_cast<double>(d[10]) / d[9] : 0.0, d[9] > 0 ? 100.0 * d[11] / 1024.0 / d[9] : 0.0,
		            d[4] > 0 ? 100.0 * d[12] / d[4] : 0.0, static_cast<unsigned long long>(g_track_jumps));
		g_track_jumps = 0;
		// (ds-rt-shadow) where the light weight is lost (1 pixel in 256 within the distance)
		std::printf("HWRTLIGHTS: reach: of %u probed pixels %.1f %% within a light's reach, %.1f %% also facing it, %.1f %% also in its cone, "
		            "%.1f %% with a weight W > 0, %.1f %% traced (share >= 3 %%); idle frames %llu "
		            "(reach 0 with lights near: their positions are not in the depth's world; facing but no cone: the forward)\n",
		            d[13], d[13] > 0 ? 100.0 * d[14] / d[13] : 0.0, d[13] > 0 ? 100.0 * d[15] / d[13] : 0.0,
		            d[13] > 0 ? 100.0 * d[16] / d[13] : 0.0, d[13] > 0 ? 100.0 * d[17] / d[13] : 0.0,
		            d[13] > 0 ? 100.0 * d[18] / d[13] : 0.0, static_cast<unsigned long long>(g_hl.idle_frames));
		g_hl.idle_frames = 0;
		std::fflush(stdout);
		g_hl.frames = g_hl.frames_lit = g_hl.lights_sum = g_hl.stale_tlas = g_hl.no_tlas = 0;
		g_hl.sum_ms = g_hl.max_ms = 0.0, g_hl.samples = 0;
	}
	if (!g_hl.said_on) {
		g_hl.said_on = true;
		std::printf("HWRTLIGHTS: on (level %d: %s, %ux%u; strength %g, max %g, ambient %g, source radius %g m, skip %g m, %g m; lights from "
		            "the game's shadow projections eased per frame, weight (r/d)^2 windowed, penumbra = source size (contact-hardening), "
		            "a ratio correction of the lit color)\n",
		            level, level == 1 ? "~0.23 Mpx, 1 ray, 8 lights" : level == 2 ? "~0.23 Mpx, 2 rays, 12 lights" : "~0.92 Mpx, 1 ray, 16 lights", color.extent.width,
		            color.extent.height, strength, max_dark, ambient, soft, skip, max_dist);
		std::fflush(stdout);
	}
	// Nothing to shadow (no light near, or the guard): the color as it is (no pass at all), the history dropped.
	if (sent.empty() || guarded) {
		RtFlicker::Cause(RtFlicker::Lights, guarded ? "guard (pass off)" : "no light near (pass off)");
		g_hl.history_valid = false;
		return false;
	}

	// This frame's slot of the light buffer.
	const uint32_t slot = static_cast<uint32_t>(g_hl.frame % Slots);
	{
		std::array<float, SlotBytes / 4> f {};
		for (int i = 0; i < 16; ++i) f[i] = static_cast<float>(ivp[i]);
		f[16] = static_cast<float>(cam[0]), f[17] = static_cast<float>(cam[1]), f[18] = static_cast<float>(cam[2]), f[19] = 1.0f;
		f[20] = static_cast<float>(sent.size()), f[21] = strength, f[22] = ambient, f[23] = max_dark;
		f[24] = flipy > 0.5f ? -1.0f : 1.0f, f[25] = 0.01f, f[26] = max_dist, f[27] = 0.0f;
		for (size_t i = 0; i < sent.size(); ++i) {
			const auto& l = *sent[i];
			float*      e = &f[32 + i * 12];
			e[0] = l.pos[0], e[1] = l.pos[1], e[2] = l.pos[2], e[3] = l.far_d;
			e[4] = l.fwd[0], e[5] = l.fwd[1], e[6] = l.fwd[2];
			// (the cone: the shadow frustum's half angle; its outer 15 % fades)
			const float half = std::atan(std::max(l.tan_half, 0.01f));
			e[7]  = l.point || half > 1.5f ? -2.0f : std::cos(half);
			e[8]  = soft;
			const float v = l.vis * l.cap_w;
			e[9]  = l.fade * v * v * (3.0f - 2.0f * v);
			e[10] = l.point || half > 1.5f ? 0.0f : std::cos(half * 0.85f);
			e[11] = std::max(skip, 1.5f * soft);
		}
		std::memcpy(g_hl.light_mapped + slot * SlotBytes, f.data(), SlotBytes);
		(void)vmaFlushAllocation(graphics->allocator, g_hl.light_mem, slot * SlotBytes, SlotBytes);
	}

	HwlPush push {};
	push.size[0] = static_cast<float>(color.extent.width), push.size[1] = static_cast<float>(color.extent.height);
	push.size[2] = static_cast<float>(g_hl.trace.width), push.size[3] = static_cast<float>(g_hl.trace.height);
	push.motion[0] = inputs.motion_scale_x, push.motion[1] = inputs.motion_scale_y;
	push.motion[2] = g_hl.history_valid && !inputs.reset && inputs.motion.view != nullptr ? 1.0f : 0.0f;
	push.misc[1]   = static_cast<int32_t>(g_hl.frame & 0x7fffffff);
	push.misc[2]   = level;

	const auto entry = static_cast<uint32_t>(g_hl.written % 8);
	const bool timed = g_hl.pool != nullptr && g_hl.written < g_hl.read + 8;
	if (timed) {
		command.resetQueryPool(g_hl.pool, entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_hl.pool, entry * 2);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	if (!g_hl.layouts_ready) {
		Img* all[] {&g_hl.out, &g_hl.raw, &g_hl.tmp_a, &g_hl.tmp_b, &g_hl.gbuf[0], &g_hl.gbuf[1], &g_hl.hist[0], &g_hl.hist[1]};
		for (Img* img: all) ToGeneral(command, img->image);
		g_hl.layouts_ready = true, g_hl.history_valid = false, push.motion[2] = 0.0f;
	}
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_hl.pipeline);

	const auto     gen = vk::ImageLayout::eGeneral;
	const uint32_t cur = static_cast<uint32_t>(g_hl.frame & 1u), prev = cur ^ 1u;
	const vk::DescriptorBufferInfo light_info {vk::Buffer(g_hl.light_buf), vk::DeviceSize {slot} * SlotBytes, SlotBytes};
	const vk::DescriptorBufferInfo stat_info {vk::Buffer(g_hl.stat_buf), 0, 128};
	vk::WriteDescriptorSetAccelerationStructureKHR as_write {};
	as_write.accelerationStructureCount = 1;
	as_write.pAccelerationStructures    = &tlas;
	const vk::DescriptorImageInfo color_info {g_hl.point, color.view, color.layout};
	const vk::DescriptorImageInfo depth_info {g_hl.point, inputs.depth.view, inputs.depth.layout};
	const vk::DescriptorImageInfo motion_info = inputs.motion.view != nullptr
	                                                ? vk::DescriptorImageInfo {g_hl.point, inputs.motion.view, inputs.motion.layout}
	                                                : vk::DescriptorImageInfo {g_hl.point, g_hl.raw.view, gen};
	// One dispatch: its input (src) and output (out).
	const auto run = [&](int mode, const Img& src, const Img& out, uint32_t w, uint32_t h, float step) {
		const Img& gbuf_out = g_hl.gbuf[cur]; // (written by mode 0 only; the others leave it as it is)
		vk::DescriptorImageInfo infos[Bindings] {};
		infos[0] = color_info;
		infos[1] = depth_info;
		infos[2] = motion_info;
		infos[3] = {g_hl.point, src.view, gen};
		infos[4] = {g_hl.point, g_hl.gbuf[cur].view, gen};
		infos[5] = {g_hl.linear, g_hl.gbuf[prev].view, gen};
		infos[6] = {g_hl.point, g_hl.hist[prev].view, gen};
		infos[7] = {nullptr, out.view, gen};
		infos[8] = {nullptr, gbuf_out.view, gen};
		vk::WriteDescriptorSet writes[Bindings] {};
		for (uint32_t i = 0; i < Bindings; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = DescType(i);
			if (i == 9) writes[i].pBufferInfo = &light_info;
			else if (i == 11) writes[i].pBufferInfo = &stat_info;
			else if (i == 10) writes[i].pNext = &as_write;
			else writes[i].pImageInfo = &infos[i];
		}
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_hl.layout, 0, Bindings, writes);
		push.misc[0]   = mode;
		push.motion[3] = step;
		command.pushConstants(g_hl.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch((w + 7) / 8, (h + 7) / 8, 1);
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	};
	const uint32_t tw = g_hl.trace.width, th = g_hl.trace.height;
	if (g_hl.idle) {
		++g_hl.idle_frames;
		if (g_hl.frame % 15 == 0) run(0, g_hl.hist[prev], g_hl.raw, tw, th, 1.0f); // (the probe: its stats only)
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead);
		if (timed) {
			command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_hl.pool, entry * 2 + 1);
			++g_hl.written;
		}
		++g_hl.frame;
		g_hl.history_valid = false;
		return false;
	}
	run(0, g_hl.hist[prev], g_hl.raw, tw, th, 1.0f);
	run(1, g_hl.raw, g_hl.hist[cur], tw, th, 1.0f);
	run(2, g_hl.hist[cur], g_hl.tmp_a, tw, th, 1.0f);
	run(2, g_hl.tmp_a, g_hl.tmp_b, tw, th, 2.0f);
	const Img* last = &g_hl.tmp_b;
	if (level >= 2) run(2, g_hl.tmp_b, g_hl.tmp_a, tw, th, 4.0f), last = &g_hl.tmp_a;
	run(3, *last, g_hl.out, color.extent.width, color.extent.height, 1.0f);
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead);
	if (timed) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_hl.pool, entry * 2 + 1);
		++g_hl.written;
	}
	++g_hl.frame;
	g_hl.history_valid  = true;
	inputs.color.image  = vk::Image(g_hl.out.image);
	inputs.color.view   = g_hl.out.view;
	inputs.color.layout = gen;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc |
	                      vk::ImageUsageFlagBits::eStorage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}

} // namespace Libs::Graphics::FrameGen
