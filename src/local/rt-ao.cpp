// KYTY_AO=0..3 / No / Bajo / Medio / Alto (ao-1028; rt-ao.comp holds the algorithm and its reasons): screen-space
// ambient occlusion written from scratch, recorded in FrameGen::Upscale in GTAO's old slot (before the reflections),
// on the render-resolution colour before the upscaler. Replaces nothing of the RT passes; no hardware ray tracing.
//
//   Bajo  (1): half-res trace, 2 slices x 4 steps a side, 0.8 m (ao-strong-1029; was 0.6), history 16 frames
//   Medio (2): half-res trace, 3 slices x 6 steps a side, 1.1 m (was 0.8), history 24 frames
//   Alto  (3): half-res trace, 4 slices x 8 steps a side, 1.4 m (was 1.0), history 32 frames
//   Passes: prep (trace-res linear z + normal) -> trace -> temporal -> a-trous (steps 1, 2) -> composite (joint-bilateral
//   upsample + ambient-only apply). (ao2-1028) Alto at full res traced on the game's full-res depth cost 8.4 ms in game
//   (RTX 5080, 720p); every level now traces at half res on its own R32F linear z, and the RTAO line times each pass.
//
//   Ultra (4, KYTY_AO=4/ultra, not in the launcher): Alto's samples, 1.7 m, the strongest look.
// Knobs: KYTY_AO_STRENGTH (1: x the level's 1.1 / 1.25 / 1.4 / 1.55, up to 3), KYTY_AO_POWER (1.6 / 1.9 / 2.2 / 2.5),
// KYTY_AO_SHARE (0.55 / 0.65 / 0.75 / 0.85 of the colour counted as ambient at least), KYTY_AO_FLOOR (0.15, was 0.4: the ambient term keeps at least this), KYTY_AO_RADIUS (metres, the
// level's), KYTY_AO_AMBIENT (6: the ambient light = this x the frame's geometric-mean light x albedo), KYTY_AO_FOG
// (60 m: fade exp(-z / fog), the colour is after the game's fog; 0 off), KYTY_AO_VIEW=1 (the AO term alone),
// KYTY_AO_GAME=keep (the game's own AO, r_enableAmbientShadowing, stays on: ours at half strength; by default
// gameArgs.cpp switches the game's off while KYTY_AO runs, so the occlusion is replaced, never applied twice).
//
// (ds-ptao-1029) "Path tracing" (the same launcher row, the same KYTY_AO levels): with a scene TLAS (KYTY_HWRT=1, which
// KYTY_AO > 0 now asks for by itself, hwrt-scene.cpp, on a GPU with ray queries) the trace is the KYTY_AO_HW build of
// rt-ao.comp: per pixel cosine-weighted rays through the TLAS (real occlusion from off-screen / behind-camera geometry),
// each hit shaded from the frame's lit colour (one bounce), plus soft shadow rays toward the sun (the RT sun pass's
// direction, when it runs) and the nearest torch.comp light (read only); temporal (history 32-48) + 3-4 a-trous passes
// on both results; the composite replaces the ambient term (visibility x ambient + albedo x bounce) and takes the
// occluded share off the direct light. No global dimming: an open surface has visibility 1 and no shadow-ray hit.
//   Bajo  (1): half res, 1 spp, GI rays 3 m, AO radius 0.6 m          Medio (2): full res, 1 spp, 5 m, 0.9 m
//   Alto  (3): full res, 2 spp, 8 m, 1.2 m (4 a-trous passes)          (1 bounce each; the screen radiance carries the rest)
// KYTY_AO_HW=0: the screen-space path even with a TLAS. Knobs: KYTY_AO_GI (bounce intensity 1.0), KYTY_AO_SHADOW (0.6:
// the most the direct light loses), KYTY_AO_SUN_LEN (4 m). Every 10 s one "PTAO:" line (path, spp, bounces, resolution,
// mean factor on the ambient, mean indirect, % darkened > 20 %, GPU ms per pass). The screen-space fallback is back to
// the pre-10-29 look with a modest boost (strength 1.0 / 1.1 / 1.2, power 1.15 / 1.25 / 1.35, share 0.2 / 0.25 / 0.3,
// floor 0.3, radius 0.7 / 0.9 / 1.1 m).
// Every 10 s one "RTAO:" line: GPU ms, mean AO, the share of pixels darkened, the frame light, the normal convention.
// The flicker self-check (KYTY_RT_FLICKER) probes it as "ambient occlusion" (frame-gen.cpp).
#include "local/rt-ao.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/rt_ao_spv.h"
#include "gpu_tiler_shaders/rt_ao_hw_spv.h"
#include "local/hwrt-scene.h"
#include "local/rt-contact.h"
#include "local/torch.h"
#include "local/rt-flicker.h"
#include "local/rt-reflections.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::FrameGen {
namespace {

int ParseLevel(const char* v) {
	if (v == nullptr || *v == '\0') return 0;
	if (*v >= '0' && *v <= '9') return std::clamp(std::atoi(v), 0, 4);
	char w[8] {};
	for (int i = 0; i < 7 && v[i] != '\0'; ++i) w[i] = static_cast<char>(v[i] | 0x20);
	if (std::strncmp(w, "bajo", 4) == 0 || std::strncmp(w, "low", 3) == 0) return 1;
	if (std::strncmp(w, "medi", 4) == 0) return 2;
	if (std::strncmp(w, "alto", 4) == 0 || std::strncmp(w, "high", 4) == 0) return 3;
	if (std::strncmp(w, "ultr", 4) == 0) return 4; // (ao-strong-1029)
	return 0;
}
const char* LevelName(int l) {
	static const char* names[] {"No", "Bajo", "Medio", "Alto", "Ultra"};
	return names[std::clamp(l, 0, 4)];
}
float EnvF(const char* name, float fallback, float lo, float hi) {
	const char* v = std::getenv(name);
	if (v == nullptr || *v == '\0') return fallback;
	const float f = static_cast<float>(std::atof(v));
	return std::isfinite(f) ? std::clamp(f, lo, hi) : fallback;
}

struct Image {
	VkImage       image   = VK_NULL_HANDLE;
	VmaAllocation memory  = VK_NULL_HANDLE;
	vk::ImageView view    = nullptr;
	bool          general = false;
};

// std140 (rt-ao.comp Frame)
struct FrameUbo {
	float   proj[4];
	float   depth_map[4];
	float   motion[4];
	float   trace[4];
	float   par[4];
	int32_t cfg[4];
	int32_t cfg2[4];
	float   rot0[4];
	float   rot1[4];
	float   rot2[4];
	float   par2[4];
	float   ivp[16];
	float   vp[16];
	float   cam[4];
	float   sun[4];
	float   hw[4];
	float   hw2[4];
};
static_assert(sizeof(FrameUbo) == 368);
struct Push {
	int32_t mode = 0;
	float   step = 0.0f;
	int32_t pad[2] {};
};

constexpr uint32_t Bindings = 18, HwBindings = 21;
constexpr uint32_t Stamps = 14; // (ao2) start + one per pass (ds-ptao: up to 13 passes on the path-traced chain)
constexpr uint32_t SlotWords = 32, Slots = 4;

struct State {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             point      = nullptr;
	bool                    failed     = false;
	Image                   dummy_u;
	bool                    dummy_ready = false;
	VkBuffer                ubo = VK_NULL_HANDLE;
	VmaAllocation           ubo_mem = VK_NULL_HANDLE;
	VkBuffer                stats = VK_NULL_HANDLE; // host-visible readback copy (never touched by shaders)
	// (ao3-1028) the counters the shaders add to: device local. They were host-visible memory: every atomic crossed PCIe,
	// serialized (composite 1.1 ms with no geometry, 7.3 ms on a full screen; prep 0.29 ms from the normal vote alone)
	VkBuffer                stats_dev = VK_NULL_HANDLE;
	VmaAllocation           stats_dev_mem = VK_NULL_HANDLE;
	VmaAllocation           stats_mem = VK_NULL_HANDLE;
	uint32_t*               stats_map = nullptr;
	vk::Extent2D            render {}, trace {};
	Image                   raw, tmp_a, tmp_b, out, history[2], gz[2], count[2], lz;
	// (ds-ptao-1029) the path-traced chain: its pipeline, the shadow result B's images, a zero torch-light buffer
	vk::DescriptorSetLayout hw_set_layout = nullptr;
	vk::PipelineLayout      hw_layout     = nullptr;
	vk::Pipeline            hw_pipeline   = nullptr;
	bool                    hw_failed     = false;
	Image                   raw2, tmp2a, tmp2b, history2[2];
	VkBuffer                zero_buf = VK_NULL_HANDLE;
	VmaAllocation           zero_mem = VK_NULL_HANDLE;
	bool                    zero_ready = false, hw_path = false;
	std::array<uint32_t, 8> entry_passes {};
	uint32_t                passes_n = 0;
	const char*             pass_names[Stamps - 1] {};
	uint32_t                frame = 0;
	bool                    history_valid = false;
	// timing
	vk::QueryPool pool = nullptr;
	double        period_ns = 1.0;
	uint64_t      written = 0, read = 0;
	double        ms_sum = 0.0;
	std::array<double, Stamps - 1> pass_ms {};
	uint32_t      ms_n = 0;
	// stats
	uint64_t sum[16] {};
	uint32_t st_frames = 0, frames_on = 0, frames_still = 0;
	float    log_light = 0.0f;
	bool     light_valid = false;
	int      conv = 1; // bit 0 negate y (measured in DS: mirrored), bit 1 negate z
	uint64_t votes[4] {};
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
} g_ao;

std::array<Image*, 17> Images() {
	return {&g_ao.raw, &g_ao.tmp_a, &g_ao.tmp_b, &g_ao.out, &g_ao.history[0], &g_ao.history[1], &g_ao.gz[0], &g_ao.gz[1],
	        &g_ao.count[0], &g_ao.count[1], &g_ao.lz, &g_ao.raw2, &g_ao.tmp2a, &g_ao.tmp2b, &g_ao.history2[0], &g_ao.history2[1], &g_ao.dummy_u};
}

bool MakeImage(GraphicContext& graphics, vk::Format format, vk::Extent2D extent, bool storage, Image& img) {
	VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create.imageType     = VK_IMAGE_TYPE_2D;
	create.format        = static_cast<VkFormat>(format);
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.samples       = VK_SAMPLE_COUNT_1_BIT;
	create.tiling        = VK_IMAGE_TILING_OPTIMAL;
	create.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
	               (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0u);
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
	img.general = false;
	return true;
}
void FreeImage(GraphicContext& graphics, Image& img) {
	if (img.view != nullptr) graphics.device.destroyImageView(img.view, nullptr);
	if (img.image != VK_NULL_HANDLE) vmaDestroyImage(graphics.allocator, img.image, img.memory);
	img = {};
}

vk::DescriptorType DescType(uint32_t i) {
	if (i == 14) return vk::DescriptorType::eUniformBuffer;
	if (i == 15 || i == 19) return vk::DescriptorType::eStorageBuffer;
	if (i == 18) return vk::DescriptorType::eAccelerationStructureKHR;
	if (i == 7 || i == 8 || i == 10 || i == 17) return vk::DescriptorType::eStorageImage;
	return vk::DescriptorType::eCombinedImageSampler;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_ao.pipeline != nullptr) return true;
	if (g_ao.failed) return false;
	g_ao.failed       = true;
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
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_ao.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Push)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_ao.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_ao.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = sizeof(RT_AO_SPV);
	module_info.pCode    = RT_AO_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_ao.layout;
	const auto created = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_ao.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_ao.pipeline = nullptr;
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
	if (device.createSampler(&sampler, nullptr, &g_ao.point) != vk::Result::eSuccess) return false;
	if (!MakeImage(graphics, vk::Format::eR32Uint, {1, 1}, false, g_ao.dummy_u)) return false;
	{
		VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		buffer.size  = sizeof(FrameUbo);
		buffer.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo dev {};
		dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_ao.ubo, &g_ao.ubo_mem, nullptr) != VK_SUCCESS) {
			g_ao.ubo = VK_NULL_HANDLE;
			return false;
		}
	}
	{
		VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		buffer.size  = SlotWords * Slots * 4;
		buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo host {};
		host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo mapped {};
		if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_ao.stats, &g_ao.stats_mem, &mapped) != VK_SUCCESS) {
			g_ao.stats = VK_NULL_HANDLE;
			return false;
		}
		g_ao.stats_map = static_cast<uint32_t*>(mapped.pMappedData);
		std::memset(g_ao.stats_map, 0, SlotWords * Slots * 4);
	}
	{
		VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		buffer.size  = SlotWords * Slots * 4;
		buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		VmaAllocationCreateInfo dev {};
		dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_ao.stats_dev, &g_ao.stats_dev_mem, nullptr) != VK_SUCCESS) {
			g_ao.stats_dev = VK_NULL_HANDLE;
			return false;
		}
	}
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = Stamps * 8;
	if (device.createQueryPool(&query, nullptr, &g_ao.pool) != vk::Result::eSuccess) g_ao.pool = nullptr;
	g_ao.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g_ao.failed    = false;
	return true;
}

// (ds-ptao-1029) the path-traced build's pipeline (made once, only on a device with ray queries)
bool EnsureHwPipeline(GraphicContext& graphics) {
	if (g_ao.hw_pipeline != nullptr) return true;
	if (g_ao.hw_failed) return false;
	g_ao.hw_failed    = true;
	const auto device = graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, HwBindings> bindings {};
	for (uint32_t i = 0; i < HwBindings; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = DescType(i);
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = HwBindings;
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_ao.hw_set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Push)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_ao.hw_set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_ao.hw_layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = sizeof(RT_AO_HW_SPV);
	module_info.pCode    = RT_AO_HW_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_ao.hw_layout;
	const auto created = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_ao.hw_pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_ao.hw_pipeline = nullptr;
		return false;
	}
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = 512;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo dev {};
	dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_ao.zero_buf, &g_ao.zero_mem, nullptr) != VK_SUCCESS) {
		g_ao.zero_buf = VK_NULL_HANDLE;
		return false;
	}
	g_ao.hw_failed = false;
	return true;
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

bool EnsureImages(GraphicContext& graphics, vk::Extent2D render, bool half) {
	const vk::Extent2D trace = half ? vk::Extent2D {std::max(1u, render.width / 2), std::max(1u, render.height / 2)} : render;
	if (g_ao.out.image != VK_NULL_HANDLE && g_ao.render == render && g_ao.trace == trace) return true;
	for (auto* img : Images())
		if (img != &g_ao.dummy_u) FreeImage(graphics, *img);
	bool ok = MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, render, true, g_ao.out);
	ok = ok && MakeImage(graphics, vk::Format::eR32Sfloat, trace, true, g_ao.lz);
	for (auto* img : Images())
		if (img != &g_ao.dummy_u && img != &g_ao.out && img != &g_ao.lz) ok = ok && MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, trace, true, *img);
	if (!ok) {
		for (auto* img : Images())
			if (img != &g_ao.dummy_u) FreeImage(graphics, *img);
		g_ao.failed = true;
		return false;
	}
	g_ao.render        = render;
	g_ao.trace         = trace;
	g_ao.history_valid = false;
	return true;
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
void ToGeneral(vk::CommandBuffer command, Image& img) {
	if (img.general) return;
	vk::ImageMemoryBarrier2 b {};
	b.srcStageMask        = vk::PipelineStageFlagBits2::eNone;
	b.dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
	b.dstAccessMask       = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite;
	b.oldLayout           = vk::ImageLayout::eUndefined;
	b.newLayout           = vk::ImageLayout::eGeneral;
	b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image               = vk::Image(img.image);
	b.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &b;
	command.pipelineBarrier2(dependency);
	img.general = true;
}

void ReadTiming(GraphicContext& graphics) {
	while (g_ao.pool != nullptr && g_ao.read + 4 <= g_ao.written) {
		const auto                   entry = static_cast<uint32_t>(g_ao.read % 8);
		const uint32_t               n     = std::clamp<uint32_t>(g_ao.entry_passes[entry], 1, Stamps - 1) + 1;
		std::array<uint64_t, Stamps> stamps {};
		if (graphics.device.getQueryPoolResults(g_ao.pool, entry * Stamps, n, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		g_ao.ms_sum += static_cast<double>(stamps[n - 1] - stamps[0]) * g_ao.period_ns / 1e6;
		for (uint32_t i = 0; i + 1 < n; ++i)
			g_ao.pass_ms[i] += static_cast<double>(stamps[i + 1] - stamps[i]) * g_ao.period_ns / 1e6;
		++g_ao.ms_n;
		++g_ao.read;
	}
}

// The slot written Slots - 1 frames ago: summed; the frame light; the normal-convention vote; every 10 s the RTAO line.
void ReadStats(GraphicContext& graphics, int level, bool still) {
	if (g_ao.stats_map == nullptr || g_ao.frame < Slots) return;
	(void)vmaInvalidateAllocation(graphics.allocator, g_ao.stats_mem, 0, VK_WHOLE_SIZE);
	const uint32_t* w = g_ao.stats_map + ((g_ao.frame + 1) % Slots) * SlotWords;
	for (int i = 0; i < 16; ++i)
		g_ao.sum[i] += w[i];
	for (int i = 0; i < 4; ++i)
		g_ao.votes[i] += w[8 + i];
	if (w[5] > 0) {
		const float l = static_cast<float>(static_cast<double>(w[4]) / w[5] / 256.0 - 32.0);
		// frozen while the camera holds still (the other passes' changes in the colour must not move every pixel's AO)
		if (!(g_ao.light_valid && still)) g_ao.log_light = g_ao.light_valid ? g_ao.log_light + (l - g_ao.log_light) * 0.05f : l;
		g_ao.light_valid = true;
	}
	++g_ao.st_frames;
	// the normal convention: switch on a clear majority (>= 2000 votes, 2x the current one's)
	{
		uint64_t tot = 0, best = 0;
		int      bi  = g_ao.conv;
		for (int i = 0; i < 4; ++i) {
			tot += g_ao.votes[i];
			if (g_ao.votes[i] > best) best = g_ao.votes[i], bi = i;
		}
		if (tot >= 2000) {
			if (bi != g_ao.conv && best > 2 * g_ao.votes[g_ao.conv]) {
				std::printf("RTAO: G-buffer normal convention %d -> %d (votes %llu / %llu / %llu / %llu)\n", g_ao.conv, bi,
				            static_cast<unsigned long long>(g_ao.votes[0]), static_cast<unsigned long long>(g_ao.votes[1]),
				            static_cast<unsigned long long>(g_ao.votes[2]), static_cast<unsigned long long>(g_ao.votes[3]));
				std::fflush(stdout);
				g_ao.conv          = bi;
				g_ao.history_valid = false;
			}
			for (auto& v: g_ao.votes) v /= 2;
		}
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - g_ao.last_log < std::chrono::seconds(10)) return;
	const auto& s   = g_ao.sum;
	const auto  pct = [](uint64_t a, uint64_t b) { return b > 0 ? 100.0 * static_cast<double>(a) / static_cast<double>(b) : 0.0; };
	const auto  pm  = [](int i) { return g_ao.ms_n > 0 ? g_ao.pass_ms[i] / g_ao.ms_n : 0.0; };
	{ // (ds-ptao-1029) one line: path, spp, bounces, resolution, the factors, darkened > 20 %, GPU ms (all + per pass)
		static const int    spp_by[5] {0, 1, 1, 2, 2};
		const bool          hw  = g_ao.hw_path;
		char                per[512] {};
		size_t              off = 0;
		for (uint32_t i = 0; i < g_ao.passes_n && i + 1 < Stamps && off + 40 < sizeof(per); ++i)
			off += static_cast<size_t>(std::snprintf(per + off, sizeof(per) - off, "%s%s %.3f", i ? ", " : "",
			                                         g_ao.pass_names[i] ? g_ao.pass_names[i] : "?", pm(static_cast<int>(i))));
		std::printf("PTAO: 10 s: path %s (%s), %s %d, bounces %d, %ux%u trace for %ux%u, mean factor on the ambient %.3f, mean "
		            "indirect (bounce / ambient) %.3f, darkened > 20 %%: %.1f%% of the pixels (> 2 %%: %.1f%%), mean visibility %.3f, "
		            "GPU %.3f ms (mean of %u frames); per pass: %s\n",
		            hw ? "HW" : "SS", LevelName(level), hw ? "spp" : "slices", hw ? spp_by[std::clamp(level, 0, 4)] : 0, hw ? 1 : 0,
		            g_ao.trace.width, g_ao.trace.height, g_ao.render.width, g_ao.render.height,
		            s[1] > 0 ? static_cast<double>(s[13]) / 250.0 / s[1] : 1.0, s[1] > 0 ? static_cast<double>(s[15]) / 250.0 / s[1] : 0.0,
		            pct(s[14], s[0]), pct(s[3], s[0]), s[1] > 0 ? static_cast<double>(s[2]) / 1000.0 / s[1] : 1.0,
		            g_ao.ms_n > 0 ? g_ao.ms_sum / g_ao.ms_n : 0.0, g_ao.ms_n, per);
		std::printf("RTAO: 10 s: frame light %.4f; normal convention %d; %u frames on (%u camera still)\n",
		            g_ao.light_valid ? std::exp2(g_ao.log_light) : 0.0f, g_ao.conv, g_ao.frames_on, g_ao.frames_still);
	}
	std::fflush(stdout);
	for (auto& v: g_ao.sum) v = 0;
	g_ao.st_frames = g_ao.frames_on = g_ao.frames_still = 0;
	g_ao.ms_sum = 0.0, g_ao.ms_n = 0;
	g_ao.pass_ms.fill(0.0);
	g_ao.last_log = now;
}

} // namespace

int RtAoLevel() {
	static const int level = [] {
		const int l = ParseLevel(std::getenv("KYTY_AO"));
		if (l > 0) {
			std::printf("RTAO: KYTY_AO=%d (%s) read\n", l, LevelName(l));
			std::fflush(stdout);
		}
		return l;
	}();
	return level;
}

bool RtAoApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int level = RtAoLevel();
	if (level == 0) return false;
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr) return false;
	if (VideoOut::MoviePlaying()) {
		g_ao.history_valid = false;
		return false;
	}
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || inputs.motion.view == nullptr || color.extent.width < 16 ||
	    color.extent.height < 16 || inputs.depth.extent != color.extent)
		return false;
	const auto& m = inputs.view_to_clip;
	if (!(m[0] != 0.0f && m[5] != 0.0f && (m[11] != 0.0f || m[15] != 0.0f)) ||
	    !std::all_of(m.begin(), m.end(), [](float v) { return std::isfinite(v); }))
		return false;
	// (ds-ptao-1029) the screen-space fallback: the 10-29 boosts reverted to the pre-10-29 look with a modest boost
	// (they darkened 53-57 % of the pixels by ~20 %); Ultra (4) stays Alto's samples, a little stronger.
	static constexpr int   slices_by[5] {0, 2, 3, 4, 4}, steps_by[5] {0, 4, 6, 8, 8}, hist_by[5] {0, 16, 24, 32, 32};
	static constexpr float radius_by[5] {0.0f, 0.7f, 0.9f, 1.1f, 1.3f};
	static constexpr float strength_by[5] {0.0f, 1.0f, 1.1f, 1.2f, 1.3f};
	static constexpr float power_by[5] {1.0f, 1.15f, 1.25f, 1.35f, 1.5f};
	static constexpr float share_by[5] {0.0f, 0.2f, 0.25f, 0.3f, 0.35f};
	// (ds-ptao-1029) the path-traced levels: spp, GI range, AO radius, history, a-trous passes, half res
	static constexpr int   hw_spp_by[5] {0, 1, 1, 2, 2}, hw_hist_by[5] {0, 32, 40, 48, 48}, hw_atrous_by[5] {0, 3, 3, 4, 4};
	static constexpr float hw_range_by[5] {0.0f, 3.0f, 5.0f, 8.0f, 10.0f}, hw_ao_by[5] {0.0f, 1.0f, 1.4f, 1.9f, 2.4f};
	// (ds-ptstrong-1029) the 10-29 log: mean factor 0.95-0.96, > 20 % on 6 %: too faint. Stronger visibility (x strength, ^ power),
	// a larger ambient share and a brighter bounce; the a-trous / temporal chain is unchanged.
	static constexpr float hw_str_by[5] {0.0f, 1.5f, 1.7f, 1.9f, 2.3f}, hw_pow_by[5] {1.0f, 1.5f, 1.6f, 1.8f, 2.1f};
	static constexpr float hw_share_by[5] {0.0f, 0.5f, 0.55f, 0.6f, 0.7f}, hw_gi_by[5] {0.0f, 2.0f, 2.2f, 2.5f, 3.0f};
	static const bool      hw_wanted = [] {
		const char* v = std::getenv("KYTY_AO_HW");
		return !(v != nullptr && *v == '0');
	}();

	// the path: hardware when the scene TLAS exists (its camera too) and the ray-query pipeline could be made
	bool                         hw   = false;
	vk::AccelerationStructureKHR tlas = nullptr;
	std::array<float, 16>        vp {};
	double                       ivp[16] {};
	double                       eye[3] {};
	const char*                  why_ss = "KYTY_AO_HW=0";
	if (hw_wanted) {
		why_ss = "KYTY_HWRT inactive (no ray queries on this GPU, or KYTY_HWRT=0)";
		if (HwScene::Active()) {
			bool built = false;
			tlas       = HwScene::SceneTlas(&built);
			why_ss     = "no scene TLAS built yet";
			if (tlas != nullptr && HwScene::SceneCamera(vp)) {
				double vpd[16];
				for (int i = 0; i < 16; ++i) vpd[i] = vp[i];
				why_ss = "the camera matrix is degenerate";
				if (Inverse4(vpd, ivp)) {
					double e[4] {};
					for (int i = 0; i < 4; ++i) e[i] = ivp[i * 4 + 2];
					if (std::abs(e[3]) > 1e-12) {
						for (int i = 0; i < 3; ++i) eye[i] = e[i] / e[3];
						why_ss = "the ray-query pipeline could not be made";
						hw     = EnsureHwPipeline(*graphics);
					}
				}
			}
		}
	}
	{
		static int said = -1;
		if (said != (hw ? 1 : 0)) {
			said = hw ? 1 : 0;
			if (hw) std::printf("PTAO: path tracing on (HW: %s, %d spp, 1 bounce, GI rays %g m, AO radius %g m, sun + torch shadow rays)\n",
			                    LevelName(level), hw_spp_by[level], hw_range_by[level], hw_ao_by[level]);
			else std::printf("PTAO: screen-space AO fallback (%s)\n", why_ss);
			std::fflush(stdout);
		}
	}
	if (hw != g_ao.hw_path) g_ao.history_valid = false;
	g_ao.hw_path    = hw;
	const bool half = hw ? level == 1 : true; // (ao2-1028) the screen-space trace at half res on every level
	if (!EnsurePipeline(*graphics) || !EnsureImages(*graphics, color.extent, half)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("RTAO: off (could not create its pipeline or images)\n");
			std::fflush(stdout);
		}
		return false;
	}
	ReadTiming(*graphics);
	const bool still = RtFlicker::CameraStill();
	++g_ao.frames_on;
	if (still) ++g_ao.frames_still;
	ReadStats(*graphics, level, still);

	static const float radius   = EnvF("KYTY_AO_RADIUS", radius_by[level], 0.1f, 4.0f);
	static const bool  keep     = [] {
		const char* v = std::getenv("KYTY_AO_GAME");
		return v != nullptr && (v[0] | 0x20) == 'k';
	}();
	static const float strength = EnvF("KYTY_AO_STRENGTH", 1.0f, 0.0f, 3.0f) * strength_by[level] * (keep ? 0.5f : 1.0f);
	static const float power    = EnvF("KYTY_AO_POWER", power_by[level], 0.5f, 4.0f);
	static const float share    = EnvF("KYTY_AO_SHARE", share_by[level], 0.0f, 1.0f);
	static const float floor_k  = EnvF("KYTY_AO_FLOOR", 0.3f, 0.0f, 1.0f);
	static const float ambient  = EnvF("KYTY_AO_AMBIENT", 6.0f, 0.0f, 100.0f);
	static const float fog      = EnvF("KYTY_AO_FOG", 80.0f, 0.0f, 100000.0f);
	static const bool  view     = EnvF("KYTY_AO_VIEW", 0.0f, 0.0f, 1.0f) > 0.5f;
	// path traced: the true visibility (power 1), the ambient's share of the colour 0.4, a floor 0.2
	static const float gi_k     = EnvF("KYTY_AO_GI", 1.0f, 0.0f, 3.0f) * hw_gi_by[level];
	static const float sh_max   = EnvF("KYTY_AO_SHADOW", 0.75f, 0.0f, 0.95f);
	static const float sun_len  = EnvF("KYTY_AO_SUN_LEN", 4.0f, 0.2f, 30.0f);
	static const float hw_share = EnvF("KYTY_AO_HW_SHARE", hw_share_by[level], 0.0f, 1.0f);
	static const float hw_str   = EnvF("KYTY_AO_STRENGTH", 1.0f, 0.0f, 3.0f) * hw_str_by[level];
	static const float hw_pow   = EnvF("KYTY_AO_POWER", hw_pow_by[level], 0.5f, 4.0f);

	RtReflGbuffer gbuf;
	const bool    have_gbuf = RtReflGetGbuffer(gbuf) && gbuf.extent == color.extent;
	RtReflCamera  camera;
	if (have_gbuf) RtReflGetCamera(m, camera);
	const bool normals = have_gbuf && gbuf.views[3] != nullptr && camera.rows;
	const bool albedo  = have_gbuf && gbuf.views[0] != nullptr;
	const bool gdepth  = have_gbuf && gbuf.depth != nullptr;
	{
		static int said = -1;
		const int  now  = (normals ? 1 : 0) | (albedo ? 2 : 0) | (gdepth ? 4 : 0) | (hw ? 8 : 0);
		if (now != said) {
			said = now;
			if (hw)
				std::printf("PTAO: on (%s: %d spp, %ux%u trace for %ux%u, history %d, %d a-trous; bounce x%g, direct-shadow max %g, sun rays %g m, "
				            "ambient share %g, strength %g, power %g, floor 0.12 (ptstrong-1029; target mean factor ~0.80-0.85 at Alto); normals from %s, albedo %s)\n",
				            LevelName(level), hw_spp_by[level], g_ao.trace.width, g_ao.trace.height, color.extent.width, color.extent.height,
				            hw_hist_by[level], hw_atrous_by[level], gi_k, sh_max, sun_len, hw_share, hw_str, hw_pow,
				            normals ? "the G-buffer + depth" : "depth", albedo ? "the G-buffer" : "0.35 (no G-buffer)");
			else
				std::printf("RTAO: on (%s: %d slices x %d steps, %g m, history %d, %ux%u trace for %ux%u; strength %g, power %g, share %g, floor %g, ambient x%g, "
				            "fog %g m; normals from %s, albedo %s, forward-drawn surfaces %s; the game's AO %s%s)\n",
				            LevelName(level), slices_by[level], steps_by[level], radius, hist_by[level], g_ao.trace.width, g_ao.trace.height,
				            color.extent.width, color.extent.height, strength, power, share, floor_k, ambient, fog, normals ? "the G-buffer + depth" : "depth",
				            albedo ? "from the G-buffer" : "0.35 (no G-buffer)", gdepth ? "skipped" : "not detectable",
				            keep ? "kept (KYTY_AO_GAME=keep: ours at half strength)" : "replaced (r_enableAmbientShadowing=false)",
				            view ? "; AO view" : "");
			std::fflush(stdout);
		}
	}

	// the torch lights (torch.cpp's GPU buffer, the last frame's tracking) and the sun (the RT sun pass's latch)
	vk::Buffer     torch_buf = nullptr;
	vk::DeviceSize torch_off = 0;
	const bool     torch     = hw && TorchLightsBuffer(torch_buf, torch_off);
	std::array<float, 3> sun_dir {0, 0, 0};
	float                sun_w = 0.0f, sun_r = 0.0f;
	bool                 sun_exact = false;
	if (hw && !RtContactGetSun(sun_dir, sun_w, sun_r, sun_exact)) sun_w = 0.0f;

	const float l_frame = g_ao.light_valid ? std::exp2(g_ao.log_light) : 0.0f;
	FrameUbo ubo {};
	ubo.proj[0] = m[0], ubo.proj[1] = m[5], ubo.proj[2] = m[8], ubo.proj[3] = m[9];
	ubo.depth_map[0] = m[10], ubo.depth_map[1] = m[14], ubo.depth_map[2] = m[11], ubo.depth_map[3] = m[15];
	ubo.motion[0] = inputs.motion_scale_x, ubo.motion[1] = inputs.motion_scale_y;
	ubo.motion[2] = static_cast<float>(color.extent.width), ubo.motion[3] = static_cast<float>(color.extent.height);
	ubo.trace[0] = static_cast<float>(g_ao.trace.width), ubo.trace[1] = static_cast<float>(g_ao.trace.height);
	ubo.trace[2] = hw ? hw_str : strength, ubo.trace[3] = radius;
	ubo.par[0] = hw ? 0.12f : floor_k, ubo.par[1] = fog;
	ubo.par[2] = l_frame * ambient;
	ubo.par[3] = 1.0f / 8.0f; // (ds-ptao) back to 1/8 of the height
	ubo.par2[0] = hw ? hw_pow : power, ubo.par2[1] = hw ? hw_share : share;
	ubo.cfg[0] = level, ubo.cfg[1] = static_cast<int32_t>(g_ao.frame);
	ubo.cfg[2] = (inputs.reset ? 1 : 0) | (g_ao.history_valid ? 2 : 0) | (normals ? 4 : 0) | (albedo ? 8 : 0) | (half ? 16 : 0) |
	             (gdepth ? 32 : 0) | (still ? 64 : 0) | (view ? 128 : 0) | (torch ? 256 : 0);
	ubo.cfg[3] = static_cast<int32_t>(g_ao.frame % Slots);
	ubo.cfg2[0] = slices_by[level], ubo.cfg2[1] = steps_by[level], ubo.cfg2[2] = hw ? hw_hist_by[level] : hist_by[level], ubo.cfg2[3] = g_ao.conv;
	if (normals) {
		const auto& r = camera.view_rows; // 3 rows x 4
		for (int i = 0; i < 3; ++i)
			ubo.rot0[i] = r[i], ubo.rot1[i] = r[4 + i], ubo.rot2[i] = r[8 + i];
	}
	if (hw) {
		for (int i = 0; i < 16; ++i) ubo.ivp[i] = static_cast<float>(ivp[i]), ubo.vp[i] = vp[i];
		ubo.cam[0] = static_cast<float>(eye[0]), ubo.cam[1] = static_cast<float>(eye[1]), ubo.cam[2] = static_cast<float>(eye[2]);
		ubo.cam[3] = -1.0f; // (hw-lights' / rt-contact's clip y convention)
		ubo.sun[0] = sun_dir[0], ubo.sun[1] = sun_dir[1], ubo.sun[2] = sun_dir[2], ubo.sun[3] = std::clamp(sun_w, 0.0f, 1.0f);
		ubo.hw[0] = hw_range_by[level], ubo.hw[1] = hw_ao_by[level], ubo.hw[2] = static_cast<float>(hw_spp_by[level]), ubo.hw[3] = gi_k;
		ubo.hw2[0] = sun_len, ubo.hw2[1] = std::tan(1.5f * 0.01745329f), ubo.hw2[2] = sh_max;
		ubo.hw2[3] = l_frame * 0.8f; // (ptstrong-1029: was 0.5) // (an off-screen hit: a dim estimate, half the frame's mean light)
	}

	const auto entry = static_cast<uint32_t>(g_ao.written % 8);
	const bool timed = g_ao.pool != nullptr;
	if (timed) {
		command.resetQueryPool(g_ao.pool, entry * Stamps, Stamps);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_ao.pool, entry * Stamps);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
	command.updateBuffer(vk::Buffer(g_ao.ubo), 0, sizeof(ubo), &ubo);
	command.fillBuffer(vk::Buffer(g_ao.stats_dev), (g_ao.frame % Slots) * SlotWords * 4, SlotWords * 4, 0);
	if (hw && !g_ao.zero_ready) {
		g_ao.zero_ready = true;
		command.fillBuffer(vk::Buffer(g_ao.zero_buf), 0, 512, 0);
	}
	for (auto* img : Images())
		ToGeneral(command, *img);
	if (!g_ao.dummy_ready) {
		g_ao.dummy_ready = true;
		vk::ClearColorValue zero {};
		zero.uint32 = std::array<uint32_t, 4> {0, 0, 0, 0};
		const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		command.clearColorImage(vk::Image(g_ao.dummy_u.image), vk::ImageLayout::eGeneral, &zero, 1, &whole);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eUniformRead);

	auto&          g    = g_ao;
	const uint32_t cur  = g.frame & 1u;
	const uint32_t prev = cur ^ 1u;
	const vk::PipelineLayout layout = hw ? g.hw_layout : g.layout;
	const uint32_t           nbind  = hw ? HwBindings : Bindings;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, hw ? g.hw_pipeline : g.pipeline);
	uint32_t stamp = 0;
	vk::WriteDescriptorSetAccelerationStructureKHR as_write {};
	as_write.accelerationStructureCount = 1;
	as_write.pAccelerationStructures    = &tlas;
	// run(mode, src, out, gz_out, cnt_out, step, size, hist_prev, shadow (composite: the denoised B))
	const auto run = [&](const char* name, int mode, const Image& a, const Image& out, const Image& gz_out, const Image& cnt_out, float step,
	                     vk::Extent2D size, const Image& hist_prev, const Image& shadow) {
		const vk::Sampler       pt  = g.point;
		const auto              gen = vk::ImageLayout::eGeneral;
		const auto              ro  = vk::ImageLayout::eShaderReadOnlyOptimal;
		vk::DescriptorImageInfo infos[HwBindings] {};
		infos[0]  = {pt, color.view, color.layout};
		infos[1]  = {pt, inputs.depth.view, inputs.depth.layout};
		infos[2]  = {pt, inputs.motion.view, inputs.motion.layout};
		infos[3]  = {pt, a.view, gen};
		infos[4]  = {pt, hist_prev.view, gen};
		infos[5]  = {pt, g.gz[cur].view, gen};
		infos[6]  = {pt, g.gz[prev].view, gen};
		infos[7]  = {nullptr, out.view, gen};
		infos[8]  = {nullptr, gz_out.view, gen};
		infos[9]  = {pt, g.count[prev].view, gen};
		infos[10] = {nullptr, cnt_out.view, gen};
		infos[11] = normals ? vk::DescriptorImageInfo {pt, gbuf.views[3], ro} : vk::DescriptorImageInfo {pt, g.dummy_u.view, gen};
		infos[12] = albedo ? vk::DescriptorImageInfo {pt, gbuf.views[0], ro} : vk::DescriptorImageInfo {pt, g.dummy_u.view, gen};
		infos[16] = {pt, g.lz.view, gen};
		infos[17] = {nullptr, g.lz.view, gen};
		infos[13] = gdepth ? vk::DescriptorImageInfo {pt, gbuf.depth, gbuf.depth_layout}
		                   : vk::DescriptorImageInfo {pt, inputs.depth.view, inputs.depth.layout};
		infos[20] = {pt, shadow.view, gen};
		const vk::DescriptorBufferInfo ubo_info {vk::Buffer(g.ubo), 0, sizeof(FrameUbo)};
		const vk::DescriptorBufferInfo st_info {vk::Buffer(g.stats_dev), 0, SlotWords * Slots * 4};
		const vk::DescriptorBufferInfo tl_info = torch ? vk::DescriptorBufferInfo {torch_buf, torch_off, 512}
		                                               : vk::DescriptorBufferInfo {vk::Buffer(g.zero_buf), 0, 512};
		vk::WriteDescriptorSet         writes[HwBindings] {};
		for (uint32_t i = 0; i < nbind; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = DescType(i);
			if (i == 14) writes[i].pBufferInfo = &ubo_info;
			else if (i == 15) writes[i].pBufferInfo = &st_info;
			else if (i == 19) writes[i].pBufferInfo = &tl_info;
			else if (i == 18) writes[i].pNext = &as_write;
			else writes[i].pImageInfo = &infos[i];
		}
		Push push {};
		push.mode = mode;
		push.step = step;
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, nbind, writes);
		command.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
		if (stamp + 1 < Stamps) {
			g.pass_names[stamp] = name;
			if (timed) command.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, g.pool, entry * Stamps + stamp + 1);
			++stamp;
		}
	};
	// (storage bindings a mode does not write still need a valid image: tmp_b is the scratch target)
	if (!hw) {
		run("prep", 0, g.tmp_a, g.tmp_b, g.gz[cur], g.tmp_b, 0.0f, g.trace, g.history[prev], g.tmp_b);           // -> gz[cur], lz
		run("trace", 1, g.tmp_a, g.raw, g.tmp_b, g.tmp_b, 0.0f, g.trace, g.history[prev], g.tmp_b);              // -> raw
		run("temporal", 2, g.raw, g.history[cur], g.tmp_b, g.count[cur], 0.0f, g.trace, g.history[prev], g.tmp_b);
		run("a-trous 1", 3, g.history[cur], g.tmp_a, g.tmp_b, g.tmp_b, 1.0f, g.trace, g.history[prev], g.tmp_b);
		run("a-trous 2", 3, g.tmp_a, g.raw, g.tmp_b, g.tmp_b, 2.0f, g.trace, g.history[prev], g.tmp_b);
		run("composite", 4, g.raw, g.out, g.tmp_b, g.tmp_b, 0.0f, g.render, g.history[prev], g.tmp_b);
	} else {
		// A = (bounce rgb, visibility) in raw / history; B = (sun occ, torch occ, torch share) in raw2 / history2
		run("prep", 0, g.tmp_a, g.tmp_b, g.gz[cur], g.tmp_b, 0.0f, g.trace, g.history[prev], g.tmp_b);
		run("trace (rays)", 1, g.tmp_a, g.raw, g.raw2, g.tmp_b, 0.0f, g.trace, g.history[prev], g.tmp_b);
		run("temporal A", 2, g.raw, g.history[cur], g.tmp_b, g.count[cur], 0.0f, g.trace, g.history[prev], g.tmp_b);
		run("temporal B", 2, g.raw2, g.history2[cur], g.tmp_b, g.tmp2b, 0.0f, g.trace, g.history2[prev], g.tmp_b);
		// a-trous on A: steps 1, 2, 4 (, 8): history[cur] -> tmp_a -> raw -> tmp_a (-> raw)
		const Image* src = &g.history[cur];
		const Image* pp[2] {&g.tmp_a, &g.raw};
		static const char* an[4] {"a-trous A 1", "a-trous A 2", "a-trous A 4", "a-trous A 8"};
		for (int i = 0; i < hw_atrous_by[level]; ++i) {
			const Image* dst = pp[i & 1];
			run(an[i], 3, *src, *dst, g.tmp_b, g.tmp_b, static_cast<float>(1 << i), g.trace, g.history[prev], g.tmp_b);
			src = dst;
		}
		run("a-trous B 1", 3, g.history2[cur], g.tmp2a, g.tmp_b, g.tmp_b, 1.0f, g.trace, g.history[prev], g.tmp_b);
		run("a-trous B 2", 3, g.tmp2a, g.raw2, g.tmp_b, g.tmp_b, 2.0f, g.trace, g.history[prev], g.tmp_b);
		run("composite", 4, *src, g.out, g.tmp_b, g.tmp_b, 0.0f, g.render, g.history[prev], g.raw2);
	}
	g.passes_n            = stamp;
	g.entry_passes[entry] = stamp;
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead);
	{ // (ao3-1028) this frame's counters, 128 bytes, to the host-visible readback slot
		const vk::DeviceSize off = (g.frame % Slots) * SlotWords * 4;
		const vk::BufferCopy region {off, off, SlotWords * 4};
		command.copyBuffer(vk::Buffer(g.stats_dev), vk::Buffer(g.stats), 1, &region);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
		        vk::AccessFlagBits2::eHostRead);
	}
	if (timed) {
		++g.written; // (the last stamp: written after the composite by run())
	}
	++g.frame;
	g.history_valid     = true;
	inputs.color.image  = vk::Image(g.out.image);
	inputs.color.view   = g.out.view;
	inputs.color.layout = vk::ImageLayout::eGeneral;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc |
	                      vk::ImageUsageFlagBits::eTransferDst;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}

} // namespace Libs::Graphics::FrameGen
