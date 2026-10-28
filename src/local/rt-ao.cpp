// KYTY_AO=0..3 / No / Bajo / Medio / Alto (ao-1028; rt-ao.comp holds the algorithm and its reasons): screen-space
// ambient occlusion written from scratch, recorded in FrameGen::Upscale in GTAO's old slot (before the reflections),
// on the render-resolution colour before the upscaler. Replaces nothing of the RT passes; no hardware ray tracing.
//
//   Bajo  (1): half-res trace, 2 slices x 4 steps a side, 0.6 m, history 16 frames
//   Medio (2): half-res trace, 3 slices x 6 steps a side, 0.8 m, history 24 frames
//   Alto  (3): half-res trace, 4 slices x 8 steps a side, 1.0 m, history 32 frames
//   Passes: prep (trace-res linear z + normal) -> trace -> temporal -> a-trous (steps 1, 2) -> composite (joint-bilateral
//   upsample + ambient-only apply). (ao2-1028) Alto at full res traced on the game's full-res depth cost 8.4 ms in game
//   (RTX 5080, 720p); every level now traces at half res on its own R32F linear z, and the RTAO line times each pass.
//
// Knobs: KYTY_AO_STRENGTH (1), KYTY_AO_FLOOR (0.4: the ambient term keeps at least this), KYTY_AO_RADIUS (metres, the
// level's), KYTY_AO_AMBIENT (6: the ambient light = this x the frame's geometric-mean light x albedo), KYTY_AO_FOG
// (60 m: fade exp(-z / fog), the colour is after the game's fog; 0 off), KYTY_AO_VIEW=1 (the AO term alone),
// KYTY_AO_GAME=keep (the game's own AO, r_enableAmbientShadowing, stays on: ours at half strength; by default
// gameArgs.cpp switches the game's off while KYTY_AO runs, so the occlusion is replaced, never applied twice).
// Every 10 s one "RTAO:" line: GPU ms, mean AO, the share of pixels darkened, the frame light, the normal convention.
// The flicker self-check (KYTY_RT_FLICKER) probes it as "ambient occlusion" (frame-gen.cpp).
#include "local/rt-ao.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/rt_ao_spv.h"
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
	if (*v >= '0' && *v <= '9') return std::clamp(std::atoi(v), 0, 3);
	char w[8] {};
	for (int i = 0; i < 7 && v[i] != '\0'; ++i) w[i] = static_cast<char>(v[i] | 0x20);
	if (std::strncmp(w, "bajo", 4) == 0 || std::strncmp(w, "low", 3) == 0) return 1;
	if (std::strncmp(w, "medi", 4) == 0) return 2;
	if (std::strncmp(w, "alto", 4) == 0 || std::strncmp(w, "high", 4) == 0) return 3;
	return 0;
}
const char* LevelName(int l) {
	static const char* names[] {"No", "Bajo", "Medio", "Alto"};
	return names[std::clamp(l, 0, 3)];
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
};
static_assert(sizeof(FrameUbo) == 160);
struct Push {
	int32_t mode = 0;
	float   step = 0.0f;
	int32_t pad[2] {};
};

constexpr uint32_t Bindings = 18;
constexpr uint32_t Stamps = 7; // (ao2) start, after prep, trace, temporal, a-trous 1, a-trous 2, composite
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
	VkBuffer                stats = VK_NULL_HANDLE;
	VmaAllocation           stats_mem = VK_NULL_HANDLE;
	uint32_t*               stats_map = nullptr;
	vk::Extent2D            render {}, trace {};
	Image                   raw, tmp_a, tmp_b, out, history[2], gz[2], count[2], lz;
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

std::array<Image*, 12> Images() {
	return {&g_ao.raw, &g_ao.tmp_a, &g_ao.tmp_b, &g_ao.out, &g_ao.history[0], &g_ao.history[1], &g_ao.gz[0], &g_ao.gz[1],
	        &g_ao.count[0], &g_ao.count[1], &g_ao.lz, &g_ao.dummy_u};
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
	if (i == 15) return vk::DescriptorType::eStorageBuffer;
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
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = Stamps * 8;
	if (device.createQueryPool(&query, nullptr, &g_ao.pool) != vk::Result::eSuccess) g_ao.pool = nullptr;
	g_ao.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g_ao.failed    = false;
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
		std::array<uint64_t, Stamps> stamps {};
		if (graphics.device.getQueryPoolResults(g_ao.pool, entry * Stamps, Stamps, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		g_ao.ms_sum += static_cast<double>(stamps[Stamps - 1] - stamps[0]) * g_ao.period_ns / 1e6;
		for (uint32_t i = 0; i + 1 < Stamps; ++i)
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
	std::printf("RTAO: 10 s: quality %s (%d), %ux%u trace for %ux%u, GPU %.3f ms (mean of %u frames, all passes), mean AO %.3f, "
	            "darkened (> 2 %%) %.1f%% of the pixels (%.1f%% of the shaded surfaces), mean drop on those surfaces %.2f%%, "
	            "surfaces %.1f%% of the pixels; frame light %.4f; normal convention %d; %u frames on (%u camera still); GPU ms per pass: "
	            "prep %.3f, trace %.3f, temporal %.3f, a-trous 1 %.3f, a-trous 2 %.3f, upsample + composite %.3f\n",
	            LevelName(level), level, g_ao.trace.width, g_ao.trace.height, g_ao.render.width, g_ao.render.height,
	            g_ao.ms_n > 0 ? g_ao.ms_sum / g_ao.ms_n : 0.0, g_ao.ms_n, s[1] > 0 ? static_cast<double>(s[2]) / 1000.0 / s[1] : 1.0,
	            pct(s[3], s[0]), pct(s[3], s[1]), s[1] > 0 ? static_cast<double>(s[6]) / 10.0 / s[1] : 0.0, pct(s[1], s[0]),
	            g_ao.light_valid ? std::exp2(g_ao.log_light) : 0.0f, g_ao.conv, g_ao.frames_on, g_ao.frames_still,
	            pm(0), pm(1), pm(2), pm(3), pm(4), pm(5));
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
	static constexpr int   slices_by[4] {0, 2, 3, 4}, steps_by[4] {0, 4, 6, 8}, hist_by[4] {0, 16, 24, 32};
	static constexpr float radius_by[4] {0.0f, 0.6f, 0.8f, 1.0f};
	const bool             half = true; // (ao2-1028) every level: Alto full res cost 8.4 ms
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
	static const float strength = EnvF("KYTY_AO_STRENGTH", 1.0f, 0.0f, 1.0f) * (keep ? 0.5f : 1.0f);
	static const float floor_k  = EnvF("KYTY_AO_FLOOR", 0.4f, 0.0f, 1.0f);
	static const float ambient  = EnvF("KYTY_AO_AMBIENT", 6.0f, 0.0f, 100.0f);
	static const float fog      = EnvF("KYTY_AO_FOG", 60.0f, 0.0f, 100000.0f);
	static const bool  view     = EnvF("KYTY_AO_VIEW", 0.0f, 0.0f, 1.0f) > 0.5f;

	RtReflGbuffer gbuf;
	const bool    have_gbuf = RtReflGetGbuffer(gbuf) && gbuf.extent == color.extent;
	RtReflCamera  camera;
	if (have_gbuf) RtReflGetCamera(m, camera);
	const bool normals = have_gbuf && gbuf.views[3] != nullptr && camera.rows;
	const bool albedo  = have_gbuf && gbuf.views[0] != nullptr;
	const bool gdepth  = have_gbuf && gbuf.depth != nullptr;
	{
		static int said = -1;
		const int  now  = (normals ? 1 : 0) | (albedo ? 2 : 0) | (gdepth ? 4 : 0);
		if (now != said) {
			said = now;
			std::printf("RTAO: on (%s: %d slices x %d steps, %g m, history %d, %ux%u trace for %ux%u; strength %g, floor %g, ambient x%g, "
			            "fog %g m; normals from %s, albedo %s, forward-drawn surfaces %s; the game's AO %s%s)\n",
			            LevelName(level), slices_by[level], steps_by[level], radius, hist_by[level], g_ao.trace.width, g_ao.trace.height,
			            color.extent.width, color.extent.height, strength, floor_k, ambient, fog, normals ? "the G-buffer + depth" : "depth",
			            albedo ? "from the G-buffer" : "0.35 (no G-buffer)", gdepth ? "skipped" : "not detectable",
			            keep ? "kept (KYTY_AO_GAME=keep: ours at half strength)" : "replaced (r_enableAmbientShadowing=false)",
			            view ? "; AO view" : "");
			std::fflush(stdout);
		}
	}

	FrameUbo ubo {};
	ubo.proj[0] = m[0], ubo.proj[1] = m[5], ubo.proj[2] = m[8], ubo.proj[3] = m[9];
	ubo.depth_map[0] = m[10], ubo.depth_map[1] = m[14], ubo.depth_map[2] = m[11], ubo.depth_map[3] = m[15];
	ubo.motion[0] = inputs.motion_scale_x, ubo.motion[1] = inputs.motion_scale_y;
	ubo.motion[2] = static_cast<float>(color.extent.width), ubo.motion[3] = static_cast<float>(color.extent.height);
	ubo.trace[0] = static_cast<float>(g_ao.trace.width), ubo.trace[1] = static_cast<float>(g_ao.trace.height);
	ubo.trace[2] = strength, ubo.trace[3] = radius;
	ubo.par[0] = floor_k, ubo.par[1] = fog;
	ubo.par[2] = g_ao.light_valid ? std::exp2(g_ao.log_light) * ambient : 0.0f;
	ubo.par[3] = 0.125f;
	ubo.cfg[0] = level, ubo.cfg[1] = static_cast<int32_t>(g_ao.frame);
	ubo.cfg[2] = (inputs.reset ? 1 : 0) | (g_ao.history_valid ? 2 : 0) | (normals ? 4 : 0) | (albedo ? 8 : 0) | (half ? 16 : 0) |
	             (gdepth ? 32 : 0) | (still ? 64 : 0) | (view ? 128 : 0);
	ubo.cfg[3] = static_cast<int32_t>(g_ao.frame % Slots);
	ubo.cfg2[0] = slices_by[level], ubo.cfg2[1] = steps_by[level], ubo.cfg2[2] = hist_by[level], ubo.cfg2[3] = g_ao.conv;
	if (normals) {
		const auto& r = camera.view_rows; // 3 rows x 4
		for (int i = 0; i < 3; ++i)
			ubo.rot0[i] = r[i], ubo.rot1[i] = r[4 + i], ubo.rot2[i] = r[8 + i];
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
	command.fillBuffer(vk::Buffer(g_ao.stats), (g_ao.frame % Slots) * SlotWords * 4, SlotWords * 4, 0);
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
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipeline);
	uint32_t stamp = 0;
	const auto run = [&](int mode, const Image& a, const Image& out, const Image& gz_out, const Image& cnt_out, float step, vk::Extent2D size) {
		const vk::Sampler       pt  = g.point;
		const auto              gen = vk::ImageLayout::eGeneral;
		const auto              ro  = vk::ImageLayout::eShaderReadOnlyOptimal;
		vk::DescriptorImageInfo infos[Bindings] {};
		infos[0]  = {pt, color.view, color.layout};
		infos[1]  = {pt, inputs.depth.view, inputs.depth.layout};
		infos[2]  = {pt, inputs.motion.view, inputs.motion.layout};
		infos[3]  = {pt, a.view, gen};
		infos[4]  = {pt, g.history[prev].view, gen};
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
		const vk::DescriptorBufferInfo ubo_info {vk::Buffer(g.ubo), 0, sizeof(FrameUbo)};
		const vk::DescriptorBufferInfo st_info {vk::Buffer(g.stats), 0, SlotWords * Slots * 4};
		vk::WriteDescriptorSet         writes[Bindings] {};
		for (uint32_t i = 0; i < Bindings; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = DescType(i);
			if (i == 14) writes[i].pBufferInfo = &ubo_info;
			else if (i == 15) writes[i].pBufferInfo = &st_info;
			else writes[i].pImageInfo = &infos[i];
		}
		Push push {};
		push.mode = mode;
		push.step = step;
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g.layout, 0, Bindings, writes);
		command.pushConstants(g.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
		if (timed) command.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, g.pool, entry * Stamps + (++stamp));
	};
	// (storage bindings a mode does not write still need a valid image: tmp_b is the scratch target)
	run(0, g.tmp_a, g.tmp_b, g.gz[cur], g.tmp_b, 0.0f, g.trace);                  // prep -> gz[cur], lz
	run(1, g.tmp_a, g.raw, g.tmp_b, g.tmp_b, 0.0f, g.trace);                      // trace -> raw
	run(2, g.raw, g.history[cur], g.tmp_b, g.count[cur], 0.0f, g.trace);         // temporal -> history[cur], count[cur]
	run(3, g.history[cur], g.tmp_a, g.tmp_b, g.tmp_b, 1.0f, g.trace);            // a-trous 1 -> tmp_a
	run(3, g.tmp_a, g.raw, g.tmp_b, g.tmp_b, 2.0f, g.trace);                     // a-trous 2 -> raw
	run(4, g.raw, g.out, g.tmp_b, g.tmp_b, 0.0f, g.render);                      // composite -> out
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead);
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
