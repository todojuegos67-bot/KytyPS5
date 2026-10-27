// KYTY_HWRT_CONTACT=0|1|2|3 (rt-contact.h): ray-traced contact shadows. Recorded in FrameGen::Upscale right after
// HwLightsApply, on the pre-exposure HDR color, before SSGI / reflections and the upscaler; the scene TLAS (hwrt-scene.cpp)
// was built earlier on the same command stream.
//
// Why: the game's own contact shadows (r_contactShadowsEnabled, 0.3 m) are screen-space: nothing off screen, behind the
// silhouette or thinner than the depth's thickness casts; its cascades lose the last centimetres (bias, texels) and the
// HWRT sun rays trace static meshes only (mask 0x01). These rays (mask 0x03: static + dynamic meshes captured fresh) give
// feet, armour, props and the bases of things their grounding shadow toward the sun and the strongest local light.
//
// Levels (No / Bajo / Medio / Alto): 1 half-res trace, 1 ray per light, 0.25 m; 2 half res, 1 ray, 0.5 m; 3 full res,
// 2 rays, 1.0 m (x up to 2 with the distance). Seeds are fixed per pixel (no frame index): a still view traces the same
// rays every frame; a temporal pass (reprojection, disocclusion, 1.25 sigma variance clamp, alpha >= 0.08) and an
// edge-aware a-trous (depth / normal stops) clean the penumbra. The composite (rt-contact.comp) darkens the direct term
// only and blends with what is already dark (no double darkening): with KYTY_RT_SUN the sun pass's own visibility V and
// its relighting model (V = 1 on a frame without it: the color then has no RT sun relighting either). ds-fx-all2: the
// old luminance-against-neighbours estimate is gone (it moved with the other passes and the jitter: 37 % flicker).
//
// Options: KYTY_HWRT_CONTACT_STRENGTH (1.0), _MAX (max darkening, 0.6), _LEN (ray length override, m), _SOFT (sun cone
// half-angle, deg, 1.5), _FADE (fade start, 25 m; out at 1.6x), _KNEE (contrast taken as texture, 0.15), _AMBIENT (local
// light share knee, 0.03), _GUARD (1 s mean screen darkening above it: off 5 s, 0.2).
// Every 10 s: "HWRTCONTACT: stats" (darkened pixels, mean darkening, blended-away pixels, ray hit rates, GPU ms).
#include "local/rt-contact.h"
#include "local/env-level.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/rt_contact_spv.h"
#include "local/hwrt-scene.h"
#include "local/rt-local-lights.h"

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

constexpr uint32_t Bindings  = 14;
constexpr uint32_t MaxLights = 8;
constexpr uint32_t SlotBytes = 1024;
constexpr uint32_t Slots     = 4;

struct CtPush {
	float   size[4];
	float   motion[4];
	int32_t misc[4];
};
static_assert(sizeof(CtPush) == 48);

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
	Img                     out, raw, tmp_a, tmp_b, held; // held: the last exact sun visibility (full res)
	bool                    held_valid = false;
	float                   held_w     = 0.0f;
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
	std::array<uint32_t, 12> stat_last {}, guard_last {};
	uint64_t                frames = 0, frames_sun = 0, frames_exact = 0, lights_sum = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now(), last_guard = last_log, guard_until {};
	bool                    said_on = false;
	uint64_t                skipped = 0;
	const char*             skip_why = nullptr;
	std::chrono::steady_clock::time_point last_skip_log = std::chrono::steady_clock::now();
} g_ct;

// The sun as CShadowApply left it this frame (render thread only).
struct {
	std::array<float, 3> dir {};
	float                weight = 0.0f, r = 0.0f;
	vk::ImageView        vis = nullptr;
	vk::Extent2D         vis_extent {};
	bool                 exact = false;
	uint64_t             upscale = ~0ull;
} g_ct_sun;
uint64_t g_ct_upscales = 0;

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
	for (auto& slot: g_ct.retired)
		if (slot.first.image == VK_NULL_HANDLE) {
			slot = {img, g_ct.frame};
			img  = {};
			return;
		}
	graphics.device.waitIdle(); // (all 16 slots busy: only on repeated resizes)
	DestroyImage(graphics, img);
}
void FreeRetired(GraphicContext& graphics) {
	for (auto& slot: g_ct.retired)
		if (slot.first.image != VK_NULL_HANDLE && g_ct.frame >= slot.second + 16) DestroyImage(graphics, slot.first), slot = {};
}

vk::DescriptorType DescType(uint32_t i) {
	return i == 9 || i == 11 ? vk::DescriptorType::eStorageBuffer
	       : i == 10         ? vk::DescriptorType::eAccelerationStructureKHR
	       : i == 7 || i == 8 || i == 13 ? vk::DescriptorType::eStorageImage
	                          : vk::DescriptorType::eCombinedImageSampler;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_ct.pipeline != nullptr) return true;
	if (g_ct.failed) return false;
	g_ct.failed       = true;
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
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_ct.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(CtPush)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_ct.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_ct.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(RT_CONTACT_SPV);
	module_info.pCode       = RT_CONTACT_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_ct.layout;
	const auto created         = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_ct.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_ct.pipeline = nullptr;
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
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g_ct.point : &g_ct.linear) != vk::Result::eSuccess) return false;
	}
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = SlotBytes * Slots;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_ct.light_buf, &g_ct.light_mem, &mapped) != VK_SUCCESS) {
		g_ct.light_buf = VK_NULL_HANDLE;
		return false;
	}
	g_ct.light_mapped = static_cast<uint8_t*>(mapped.pMappedData);
	buffer.size       = 64;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_ct.stat_buf, &g_ct.stat_mem, &mapped) != VK_SUCCESS) {
		g_ct.stat_buf = VK_NULL_HANDLE;
		return false;
	}
	g_ct.stat_mapped = static_cast<volatile uint32_t*>(mapped.pMappedData);
	for (int i = 0; i < 16; ++i) g_ct.stat_mapped[i] = 0;
	(void)vmaFlushAllocation(graphics.allocator, g_ct.stat_mem, 0, VK_WHOLE_SIZE);
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = 8 * 2;
	if (device.createQueryPool(&query, nullptr, &g_ct.pool) != vk::Result::eSuccess) g_ct.pool = nullptr;
	g_ct.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g_ct.failed    = false;
	return true;
}

bool EnsureImages(GraphicContext& graphics, vk::Extent2D extent, int level) {
	// (ds-rt-stable-1027) level 3 traces at half resolution too (2 rays, 64-frame still mean, 2x2 nearest-depth upsample):
	// ~1/4 of the 3.7 ms; KYTY_HWRT_CONTACT_FULL=1 keeps the old full-res trace
	static const bool full = [] { const char* v = std::getenv("KYTY_HWRT_CONTACT_FULL"); return v != nullptr && *v == '1'; }();
	const vk::Extent2D trace = level >= 3 && full ? extent : vk::Extent2D {std::max(extent.width / 2, 1u), std::max(extent.height / 2, 1u)};
	if (g_ct.out.image != VK_NULL_HANDLE && g_ct.extent == extent && g_ct.trace == trace) return true;
	Img* all[] {&g_ct.out, &g_ct.held, &g_ct.raw, &g_ct.tmp_a, &g_ct.tmp_b, &g_ct.gbuf[0], &g_ct.gbuf[1], &g_ct.hist[0], &g_ct.hist[1]};
	for (Img* img: all) Retire(graphics, *img);
	g_ct.extent = extent, g_ct.trace = trace;
	g_ct.layouts_ready = false, g_ct.history_valid = false, g_ct.held_valid = false, g_ct.held_w = 0.0f;
	bool ok = MakeImage(graphics, extent, g_ct.out) && MakeImage(graphics, extent, g_ct.held);
	for (Img* img: all)
		if (img != &g_ct.out && img != &g_ct.held) ok = ok && MakeImage(graphics, trace, *img);
	if (!ok)
		for (Img* img: all) Retire(graphics, *img);
	return ok;
}

void ReadTiming(GraphicContext& graphics) {
	if (g_ct.pool == nullptr) return;
	while (g_ct.read + 4 <= g_ct.written) {
		const auto              entry = static_cast<uint32_t>(g_ct.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (graphics.device.getQueryPoolResults(g_ct.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		if (stamps[1] > stamps[0]) {
			const double ms = static_cast<double>(stamps[1] - stamps[0]) * g_ct.period_ns / 1e6;
			g_ct.sum_ms += ms, g_ct.max_ms = std::max(g_ct.max_ms, ms);
			++g_ct.samples;
		}
		++g_ct.read;
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

int RtContactLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_HWRT_CONTACT");
		const int   q     = ::Libs::Graphics::FrameGen::ParseEnvLevel(value);
		if (q > 0) {
			std::printf("HWRTCONTACT: KYTY_HWRT_CONTACT=%s read (level %d: %s)%s\n", value, q, q == 1 ? "Bajo" : q == 2 ? "Medio" : "Alto",
			            HwScene::Requested() ? "" : ": needs KYTY_HWRT=1 too: off");
			std::fflush(stdout);
		}
		return HwScene::Requested() ? q : 0;
	}();
	return level;
}

void RtContactSetSun(const std::array<double, 3>& world_dir, float weight, float r, vk::ImageView vis, vk::Extent2D vis_extent, bool exact) {
	if (RtContactLevel() == 0) return;
	const double n = std::sqrt(world_dir[0] * world_dir[0] + world_dir[1] * world_dir[1] + world_dir[2] * world_dir[2]);
	if (!(n > 1e-6) || !std::isfinite(n)) return;
	for (int i = 0; i < 3; ++i) g_ct_sun.dir[i] = static_cast<float>(world_dir[i] / n);
	g_ct_sun.weight     = std::isfinite(weight) ? std::clamp(weight, 0.0f, 1.0f) : 0.0f;
	g_ct_sun.r          = std::isfinite(r) ? std::clamp(r, 0.0f, 50.0f) : 1.5f;
	g_ct_sun.vis        = vis;
	g_ct_sun.vis_extent = vis_extent;
	g_ct_sun.exact      = exact && vis != nullptr;
	g_ct_sun.upscale    = g_ct_upscales;
}

bool RtContactApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int level = RtContactLevel();
	if (level == 0) return false;
	const uint64_t this_upscale = g_ct_upscales++;
	const bool     sun_fresh    = g_ct_sun.upscale == this_upscale;
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr) return false;
	{
		const auto t = std::chrono::steady_clock::now();
		if (t - g_ct.last_skip_log >= std::chrono::seconds(10)) {
			if (g_ct.skipped > 0) {
				std::printf("HWRTCONTACT: 10 s: %llu frames without the pass (last: %s)\n", static_cast<unsigned long long>(g_ct.skipped),
				            g_ct.skip_why != nullptr ? g_ct.skip_why : "?");
				std::fflush(stdout);
			}
			g_ct.skipped = 0, g_ct.last_skip_log = t;
		}
	}
	const auto said_once = [](const char* why) {
		static const char* said = nullptr;
		++g_ct.skipped, g_ct.skip_why = why;
		if (said != why) {
			said = why;
			std::printf("HWRTCONTACT: nothing this frame (%s)\n", why);
			std::fflush(stdout);
		}
		g_ct.history_valid = false;
		return false;
	};
	if (!HwScene::Active()) return said_once("KYTY_HWRT inactive: the device has no ray queries");
	if (VideoOut::MoviePlaying()) return g_ct.history_valid = false, false;
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || color.extent.width < 16 || color.extent.height < 16 ||
	    inputs.depth.extent != color.extent)
		return said_once("missing color / depth");
	bool                         built = false;
	vk::AccelerationStructureKHR tlas  = HwScene::SceneTlas(&built);
	if (tlas == nullptr) return said_once("no scene TLAS built yet");
	std::array<float, 16> vp {};
	if (!HwScene::SceneCamera(vp)) {
		std::lock_guard lock(g_last_vp_mutex);
		if (!g_have_last_vp) return said_once("no camera matrix captured yet");
		vp = g_last_vp;
	}
	double vpd[16], ivp[16];
	for (int i = 0; i < 16; ++i) vpd[i] = vp[i];
	if (!Inverse4(vpd, ivp)) return said_once("the camera matrix is degenerate");
	double eye[4] {};
	for (int i = 0; i < 4; ++i) eye[i] = ivp[i * 4 + 2];
	if (!(std::abs(eye[3]) > 1e-12)) return said_once("the camera matrix has no perspective");
	const double cam[3] {eye[0] / eye[3], eye[1] / eye[3], eye[2] / eye[3]};

	if (!EnsurePipeline(*graphics) || !EnsureImages(*graphics, color.extent, level)) return said_once("could not create its pipeline or images");
	FreeRetired(*graphics);
	ReadTiming(*graphics);

	static const float strength = EnvFloat("KYTY_HWRT_CONTACT_STRENGTH", 1.0f, 0.0f, 2.0f);
	static const float max_dark = EnvFloat("KYTY_HWRT_CONTACT_MAX", 0.6f, 0.0f, 0.95f);
	static const float length   = EnvFloat("KYTY_HWRT_CONTACT_LEN", level == 1 ? 0.25f : level == 2 ? 0.5f : 1.0f, 0.02f, 3.0f);
	static const float soft     = EnvFloat("KYTY_HWRT_CONTACT_SOFT", 1.5f, 0.1f, 8.0f);
	static const float fade_m   = EnvFloat("KYTY_HWRT_CONTACT_FADE", 25.0f, 2.0f, 300.0f);
	static const float knee     = EnvFloat("KYTY_HWRT_CONTACT_KNEE", 0.15f, 0.0f, 0.9f);
	static const float ambient  = EnvFloat("KYTY_HWRT_CONTACT_AMBIENT", 0.03f, 0.001f, 100.0f);
	static const float guard    = EnvFloat("KYTY_HWRT_CONTACT_GUARD", 0.2f, 0.0f, 1.0f);

	// The sun: this frame's from CShadowApply; otherwise the last one held, its weight eased out over 0.5 s (and back in
	// over 0.5 s): no pop when the sun pass skips a frame.
	const auto  now = std::chrono::steady_clock::now();
	static auto last_t = now;
	const float dt     = std::clamp(std::chrono::duration<float>(now - last_t).count(), 0.0f, 0.25f);
	last_t             = now;
	// (ds-fx-all2) held: the last weight given, never eased to 0 on a frame the sun pass skipped (sun in 257 of 343
	// frames: the old easing pulsed the shadow); eased toward a new value over 0.5 s.
	static float sun_w = 0.0f;
	const float  target = g_ct_sun.weight;
	sun_w = std::clamp(sun_w + std::clamp(target - sun_w, -dt / 0.5f, dt / 0.5f), 0.0f, 1.0f);
	const bool exact = sun_fresh && g_ct_sun.exact;
	// (ds-rt-stable-1027) RTFLICKER contact 10-28 %, max jump 140 %: the RT sun pass ran on only 18-74 of ~490 frames
	// (HWRTCONTACT 'sun in N frames (M with exact visibility)'), and the composite switched V between that visibility
	// and 1 on every such frame - a sun-shadowed pixel was left alone one frame and darkened by up to K/(1+K) the next.
	// Now the composite stores V on an exact frame and reuses the held copy on every other one (one V every frame);
	// its weight eases in / out over 1 s, so the model never switches in one frame.
	if (exact) g_ct.held_valid = true;
	g_ct.held_w = std::clamp(g_ct.held_w + (g_ct.held_valid ? dt : -dt), 0.0f, 1.0f);

	// The local lights: seen within 120 flips by >= 2 scans, nearest first, up to 8 (the pixel picks the strongest).
	std::vector<RtLocal::LiveLight> live;
	if (RtLocal::Wanted()) RtLocal::LiveLights(live, 120, 2);
	struct Pick {
		float                     d;
		const RtLocal::LiveLight* l;
	};
	std::vector<Pick> picks;
	for (const auto& l: live) {
		if (!(l.far_d > 0.5f && l.far_d < 100.0f)) continue;
		const double dx = l.pos[0] - cam[0], dy = l.pos[1] - cam[1], dz = l.pos[2] - cam[2];
		const double d  = std::sqrt(dx * dx + dy * dy + dz * dz);
		if (d - l.far_d > fade_m * 1.6f) continue;
		picks.push_back({static_cast<float>(d), &l});
	}
	std::sort(picks.begin(), picks.end(), [](const Pick& a, const Pick& b) { return a.d < b.d; });
	if (picks.size() > MaxLights) picks.resize(MaxLights);

	// The guard: a 1 s window whose mean darkening over the screen passes _GUARD turns the pass off for 5 s.
	if (now - g_ct.last_guard >= std::chrono::seconds(1)) {
		g_ct.last_guard = now;
		(void)vmaInvalidateAllocation(graphics->allocator, g_ct.stat_mem, 0, VK_WHOLE_SIZE);
		std::array<uint32_t, 12> cur {};
		for (int i = 0; i < 12; ++i) cur[i] = g_ct.stat_mapped[i];
		const double px = cur[0] - g_ct.guard_last[0], dk = cur[2] - g_ct.guard_last[2];
		g_ct.guard_last = cur;
		const double mean = px > 0 ? dk / 1024.0 / px : 0.0;
		if (guard > 0.0f && mean > guard && now >= g_ct.guard_until) {
			g_ct.guard_until = now + std::chrono::seconds(5);
			std::printf("HWRTCONTACT: guard: the screen darkened %.1f %% on average over 1 s (> %.0f %%): off for 5 s\n", 100.0 * mean,
			            100.0 * guard);
			std::fflush(stdout);
		}
	}
	const bool guarded = now < g_ct.guard_until;

	++g_ct.frames;
	g_ct.lights_sum += picks.size();
	if (sun_w > 0.0f) ++g_ct.frames_sun;
	if (exact) ++g_ct.frames_exact;
	if (now - g_ct.last_log >= std::chrono::seconds(10)) {
		g_ct.last_log = now;
		(void)vmaInvalidateAllocation(graphics->allocator, g_ct.stat_mem, 0, VK_WHOLE_SIZE);
		std::array<uint32_t, 12> cur {}, d {};
		for (int i = 0; i < 12; ++i) cur[i] = g_ct.stat_mapped[i], d[i] = cur[i] - g_ct.stat_last[i];
		g_ct.stat_last = cur;
		const double px = std::max<double>(d[0], 1.0), dp = d[1], dk = d[2];
		std::printf("HWRTCONTACT: stats (level %d, trace %ux%u, rays %.2f m) darkened %.2f %% of pixels by %.1f %% on average (%.3f %% of "
		            "the screen's light); %.2f %% of pixels already in the RT sun pass's shadow (left as they are); sun in %llu/%llu "
		            "frames (%llu with KYTY_RT_SUN's exact visibility), lights %.2f a frame; GPU %.3f ms (max %.3f)\n",
		            level, g_ct.trace.width, g_ct.trace.height, static_cast<double>(length), 100.0 * dp / px, dp > 0 ? 100.0 * dk / 1024.0 / dp : 0.0,
		            100.0 * dk / 1024.0 / px, 100.0 * d[7] / px, static_cast<unsigned long long>(g_ct.frames_sun),
		            static_cast<unsigned long long>(g_ct.frames), static_cast<unsigned long long>(g_ct.frames_exact),
		            static_cast<double>(g_ct.lights_sum) / std::max<double>(g_ct.frames, 1.0), g_ct.samples > 0 ? g_ct.sum_ms / g_ct.samples : 0.0,
		            g_ct.max_ms);
		std::printf("HWRTCONTACT: rays: sun %u sampled, %.1f %% occluded; light %u sampled, %.1f %% occluded; %u receivers on dynamic "
		            "meshes (self-shadow skipped), %u receiver self-hits ignored\n",
		            d[3], d[3] > 0 ? 100.0 * d[4] / d[3] : 0.0, d[5], d[5] > 0 ? 100.0 * d[6] / d[5] : 0.0, d[8], d[9]);
		std::fflush(stdout);
		g_ct.frames = g_ct.frames_sun = g_ct.frames_exact = g_ct.lights_sum = 0;
		g_ct.sum_ms = g_ct.max_ms = 0.0, g_ct.samples = 0;
	}
	if (!g_ct.said_on) {
		g_ct.said_on = true;
		std::printf("HWRTCONTACT: on (level %d: %s, %ux%u; rays %g m toward the sun and the strongest local light, mask static + "
		            "fresh dynamic, one blend formula, held sun; strength %g, max %g, sun cone %g deg, fade %g-%g m, texture knee %g)\n",
		            level, level == 1 ? "half-res, 1 ray" : level == 2 ? "half-res, 1 ray" : (g_ct.trace == color.extent ? "full-res, 2 rays" : "half-res, 2 rays"), color.extent.width,
		            color.extent.height, static_cast<double>(length), static_cast<double>(strength), static_cast<double>(max_dark),
		            static_cast<double>(soft), static_cast<double>(fade_m), static_cast<double>(fade_m * 1.6f), static_cast<double>(knee));
		std::fflush(stdout);
	}
	if (guarded || (sun_w <= 0.0f && picks.empty())) {
		g_ct.history_valid = false;
		return false;
	}

	const uint32_t slot = static_cast<uint32_t>(g_ct.frame % Slots);
	{
		std::array<float, SlotBytes / 4> f {};
		for (int i = 0; i < 16; ++i) f[i] = static_cast<float>(ivp[i]);
		f[16] = static_cast<float>(cam[0]), f[17] = static_cast<float>(cam[1]), f[18] = static_cast<float>(cam[2]), f[19] = 1.0f;
		f[20] = g_ct_sun.dir[0], f[21] = g_ct_sun.dir[1], f[22] = g_ct_sun.dir[2], f[23] = sun_w;
		f[24] = g_ct_sun.r, f[25] = exact ? 2.0f : g_ct.held_valid ? 1.0f : 0.0f, f[26] = length, f[27] = strength;
		f[28] = static_cast<float>(picks.size()), f[29] = -1.0f, f[30] = max_dark, f[31] = fade_m;
		f[32] = fade_m * 1.6f, f[33] = std::tan(soft * 0.01745329f), f[34] = ambient, f[35] = knee;
		f[36] = g_ct.held_w;
		for (size_t i = 0; i < picks.size(); ++i) {
			const auto& l = *picks[i].l;
			float*      e = &f[40 + i * 12];
			e[0] = l.pos[0], e[1] = l.pos[1], e[2] = l.pos[2], e[3] = l.far_d;
			e[4] = l.fwd[0], e[5] = l.fwd[1], e[6] = l.fwd[2];
			const float half = std::atan(std::max(l.tan_half, 0.01f));
			e[7]  = l.point || half > 1.5f ? -2.0f : std::cos(half);
			const float fade_out = 1.0f - std::clamp((static_cast<float>(l.age) - 60.0f) / 60.0f, 0.0f, 1.0f);
			const float fade_in  = std::clamp(static_cast<float>(l.since) / 30.0f, 0.0f, 1.0f);
			e[8]  = fade_out * fade_in;
			e[9]  = l.point || half > 1.5f ? 0.0f : std::cos(half * 0.85f);
			e[10] = 0.3f;  // (the ray ends this short of the light: its lamp / brazier mesh)
			e[11] = 0.08f; // (source radius, m)
		}
		std::memcpy(g_ct.light_mapped + slot * SlotBytes, f.data(), SlotBytes);
		(void)vmaFlushAllocation(graphics->allocator, g_ct.light_mem, slot * SlotBytes, SlotBytes);
	}

	CtPush push {};
	push.size[0] = static_cast<float>(color.extent.width), push.size[1] = static_cast<float>(color.extent.height);
	push.size[2] = static_cast<float>(g_ct.trace.width), push.size[3] = static_cast<float>(g_ct.trace.height);
	push.motion[0] = inputs.motion_scale_x, push.motion[1] = inputs.motion_scale_y;
	push.motion[2] = g_ct.history_valid && !inputs.reset && inputs.motion.view != nullptr ? 1.0f : 0.0f;
	push.misc[1]   = static_cast<int32_t>(g_ct.frame & 0x7fffffff);
	push.misc[2]   = level;

	const auto entry = static_cast<uint32_t>(g_ct.written % 8);
	const bool timed = g_ct.pool != nullptr && g_ct.written < g_ct.read + 8;
	if (timed) {
		command.resetQueryPool(g_ct.pool, entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_ct.pool, entry * 2);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	if (!g_ct.layouts_ready) {
		Img* all[] {&g_ct.out, &g_ct.held, &g_ct.raw, &g_ct.tmp_a, &g_ct.tmp_b, &g_ct.gbuf[0], &g_ct.gbuf[1], &g_ct.hist[0], &g_ct.hist[1]};
		for (Img* img: all) ToGeneral(command, img->image);
		g_ct.layouts_ready = true, g_ct.history_valid = false, push.motion[2] = 0.0f;
	}
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_ct.pipeline);

	const auto     gen = vk::ImageLayout::eGeneral;
	const uint32_t cur = static_cast<uint32_t>(g_ct.frame & 1u), prev = cur ^ 1u;
	const vk::DescriptorBufferInfo light_info {vk::Buffer(g_ct.light_buf), vk::DeviceSize {slot} * SlotBytes, SlotBytes};
	const vk::DescriptorBufferInfo stat_info {vk::Buffer(g_ct.stat_buf), 0, 64};
	vk::WriteDescriptorSetAccelerationStructureKHR as_write {};
	as_write.accelerationStructureCount = 1;
	as_write.pAccelerationStructures    = &tlas;
	const vk::DescriptorImageInfo color_info {g_ct.point, color.view, color.layout};
	const vk::DescriptorImageInfo depth_info {g_ct.point, inputs.depth.view, inputs.depth.layout};
	const vk::DescriptorImageInfo motion_info = inputs.motion.view != nullptr
	                                                ? vk::DescriptorImageInfo {g_ct.point, inputs.motion.view, inputs.motion.layout}
	                                                : vk::DescriptorImageInfo {g_ct.point, g_ct.raw.view, gen};
	// (the sun pass's visibility, bilinear; unused without it: any image of ours bound)
	const vk::DescriptorImageInfo vis_info = exact ? vk::DescriptorImageInfo {g_ct.linear, g_ct_sun.vis, gen}
	                                               : vk::DescriptorImageInfo {g_ct.linear, g_ct.gbuf[prev].view, gen};
	const auto run = [&](int mode, const Img& src, const Img& out, uint32_t w, uint32_t h, float step) {
		vk::DescriptorImageInfo infos[Bindings] {};
		infos[0]  = color_info;
		infos[1]  = depth_info;
		infos[2]  = motion_info;
		infos[3]  = {g_ct.point, src.view, gen};
		infos[4]  = {g_ct.point, g_ct.gbuf[cur].view, gen};
		infos[5]  = {g_ct.linear, g_ct.gbuf[prev].view, gen};
		infos[6]  = {g_ct.point, g_ct.hist[prev].view, gen};
		infos[7]  = {nullptr, out.view, gen};
		infos[8]  = {nullptr, g_ct.gbuf[cur].view, gen};
		infos[12] = vis_info;
		infos[13] = {nullptr, g_ct.held.view, gen};
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
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_ct.layout, 0, Bindings, writes);
		push.misc[0]   = mode;
		push.motion[3] = step;
		command.pushConstants(g_ct.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch((w + 7) / 8, (h + 7) / 8, 1);
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	};
	const uint32_t tw = g_ct.trace.width, th = g_ct.trace.height;
	// trace -> raw (+ gbuf[cur]); temporal -> hist[cur] (next frame's history); a-trous 1, 2 (, 4) -> composite
	run(0, g_ct.hist[prev], g_ct.raw, tw, th, 1.0f);
	run(1, g_ct.raw, g_ct.hist[cur], tw, th, 1.0f);
	run(2, g_ct.hist[cur], g_ct.tmp_a, tw, th, 1.0f);
	run(2, g_ct.tmp_a, g_ct.tmp_b, tw, th, 2.0f);
	const Img* last = &g_ct.tmp_b;
	if (level >= 3) run(2, g_ct.tmp_b, g_ct.tmp_a, tw, th, 4.0f), last = &g_ct.tmp_a;
	run(3, *last, g_ct.out, color.extent.width, color.extent.height, 1.0f);
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead);
	if (timed) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_ct.pool, entry * 2 + 1);
		++g_ct.written;
	}
	++g_ct.frame;
	g_ct.history_valid  = true;
	inputs.color.image  = vk::Image(g_ct.out.image);
	inputs.color.view   = g_ct.out.view;
	inputs.color.layout = gen;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc |
	                      vk::ImageUsageFlagBits::eStorage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}

} // namespace Libs::Graphics::FrameGen
