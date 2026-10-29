// KYTY_SMOKE=0|1|2 ("Humo y particulas"; smoke.comp): soft particles and torch-lit smoke for Demon's Souls.
//
// What the game gives: one particle-translucent pass draws every sprite blended onto the HDR scene colour at the render
// size (full resolution: the copy below is taken at the size of the upscaler's colour), depth test on, depth writes
// off. KYTY_RT_PARTICLES' hook (frame-gen.cpp BeforeSceneDraw) copies the colour right before its first draw; the
// copy is taken while this switch is on too (one render-size RGBA16F copy, ~0.05-0.1 ms). D = colour - copy is the
// particle layer: exact for additive sprites, a (c - background) for alpha-blended smoke, so D x f = the sprite drawn
// at alpha x f.
//
// Level 1 (soft): where a sprite was cut by the depth test on a continuous surface (a hard step in D, background and
//   depth continuous across it: smoke meeting a floor or a wall) D fades in over KYTY_SMOKE_SOFT metres (0.35, projected
//   at the pixel's depth, 1.5..24 px), the soft-particle depth fade the sprites lack. Exact distance (row + column
//   passes), motion-reprojected history (alpha 0.4, depth-rejected): stable under the jitter.
// Level 2 (+ torch glow): the smoke's coverage of each pixel is lit by the torch lights torch.cpp tracked this flip
//   (KYTY_TORCH >= 1 or KYTY_TORCH_FLARE: their view position, hue, SLOW power x fade - never the flicker - and their
//   4.5 m local radius), in-scattering integrated along the view ray to the surface (8 fixed steps, no noise), soft-capped;
//   never on fire pixels. KYTY_SMOKE_GLOW (1.0, 0..3).
// KYTY_SMOKE_VIEW=1: the correction only (x4). Every 10 s "SMOKE: stats" (what is active, resolution, %, GPU ms).
#include "local/smoke.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/smoke_spv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::FrameGen {
namespace {

constexpr uint32_t Bindings   = 12;
constexpr uint32_t StatsCount = 16;

struct SmokePush {
	float   proj[4];
	float   depth_map[4];
	float   size[4];
	float   par[4];
	float   motion[4];
	int32_t misc[4];
};
static_assert(sizeof(SmokePush) == 96);

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
	Img                     out, a, b, hist[2];
	vk::Extent2D            extent {};
	bool                    ready = false, history_valid = false;
	VkBuffer                stats_buf = VK_NULL_HANDLE, read_buf = VK_NULL_HANDLE;
	VmaAllocation           stats_mem = nullptr, read_mem = nullptr;
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
	uint64_t                ran = 0, no_pre = 0, torch_frames = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now(), last_frame {};
} g_sm;

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
	for (auto& slot: g_sm.retired)
		if (slot.first.image == VK_NULL_HANDLE) {
			slot = {img, g_sm.frame};
			img  = {};
			return;
		}
	graphics.device.waitIdle();
	Destroy(graphics, img);
}
void FreeRetired(GraphicContext& graphics) {
	for (auto& slot: g_sm.retired)
		if (slot.first.image != VK_NULL_HANDLE && g_sm.frame >= slot.second + 16) Destroy(graphics, slot.first), slot.second = 0;
}

vk::DescriptorType DescType(uint32_t i) {
	return (i >= 9) ? vk::DescriptorType::eStorageBuffer
	       : (i == 7 || i == 8) ? vk::DescriptorType::eStorageImage
	                            : vk::DescriptorType::eCombinedImageSampler;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_sm.pipeline != nullptr) return true;
	if (g_sm.failed) return false;
	g_sm.failed       = true;
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
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_sm.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(SmokePush)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_sm.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_sm.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(SMOKE_SPV);
	module_info.pCode       = SMOKE_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_sm.layout;
	const auto created         = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_sm.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_sm.pipeline = nullptr;
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
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g_sm.point : &g_sm.linear) != vk::Result::eSuccess) return false;
	}
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = 256;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VmaAllocationCreateInfo dev {};
	dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_sm.stats_buf, &g_sm.stats_mem, nullptr) != VK_SUCCESS) {
		g_sm.stats_buf = VK_NULL_HANDLE;
		return false;
	}
	buffer.size  = StatsCount * 4;
	buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_sm.read_buf, &g_sm.read_mem, &mapped) != VK_SUCCESS) {
		g_sm.read_buf = VK_NULL_HANDLE;
		return false;
	}
	g_sm.mapped = static_cast<const uint32_t*>(mapped.pMappedData);
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = 8 * 2;
	if (device.createQueryPool(&query, nullptr, &g_sm.pool) != vk::Result::eSuccess) g_sm.pool = nullptr;
	g_sm.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g_sm.failed    = false;
	return true;
}

bool EnsureImages(GraphicContext& graphics, vk::Extent2D extent) {
	if (g_sm.out.image != VK_NULL_HANDLE && g_sm.extent == extent) return true;
	Img* all[] {&g_sm.out, &g_sm.a, &g_sm.b, &g_sm.hist[0], &g_sm.hist[1]};
	for (Img* img: all) Retire(graphics, *img);
	g_sm.ready = false, g_sm.history_valid = false;
	g_sm.extent = extent;
	for (Img* img: all)
		if (!MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, extent, *img)) return false;
	return true;
}

void ReadTiming(GraphicContext& graphics) {
	if (g_sm.pool == nullptr) return;
	while (g_sm.read + 4 <= g_sm.written) {
		const auto              entry = static_cast<uint32_t>(g_sm.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (graphics.device.getQueryPoolResults(g_sm.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		if (stamps[1] > stamps[0]) {
			const double ms = static_cast<double>(stamps[1] - stamps[0]) * g_sm.period_ns / 1e6;
			g_sm.sum_ms += ms, g_sm.max_ms = std::max(g_sm.max_ms, ms);
			++g_sm.samples;
		}
		++g_sm.read;
	}
}

void Log(int level, vk::Extent2D extent, float soft_m, float glow) {
	const auto now = std::chrono::steady_clock::now();
	if (now - g_sm.last_log < std::chrono::seconds(10) || g_sm.mapped == nullptr) return;
	g_sm.last_log = now;
	std::array<uint32_t, StatsCount> cur {};
	std::memcpy(cur.data(), g_sm.mapped, sizeof(cur)); // (a few frames late: statistics only)
	std::array<uint32_t, StatsCount> d {};
	for (uint32_t i = 0; i < StatsCount; ++i) d[i] = cur[i] - g_sm.stat_last[i];
	g_sm.stat_last  = cur;
	const double fr = std::max<double>(d[0], 1.0);
	const double px = std::max<double>(d[1], 1.0);
	std::printf("SMOKE: stats (level %d: soft particles %s (%.2f m), torch glow %s (x%.2f)) particle layer %ux%u (render size, full res), "
	            "%llu frames run, %llu without the pre-particle copy; particle px %.2f %%, cut edges %.3f %%, softened %.2f %%, "
	            "torch-lit smoke %.2f %%, torch lights %.2f a frame (%llu frames with torch data), GPU %.3f ms (max %.3f)\n",
	            level, soft_m > 0.0f ? "on" : "off", soft_m, level >= 2 && glow > 0.0f ? "on" : "off", glow, extent.width,
	            extent.height, static_cast<unsigned long long>(g_sm.ran), static_cast<unsigned long long>(g_sm.no_pre),
	            100.0 * d[2] / px, 100.0 * d[3] / px, 100.0 * d[4] / px, 100.0 * d[5] / px, d[6] / fr,
	            static_cast<unsigned long long>(g_sm.torch_frames), g_sm.samples > 0 ? g_sm.sum_ms / g_sm.samples : 0.0, g_sm.max_ms);
	std::fflush(stdout);
	g_sm.sum_ms = 0.0, g_sm.max_ms = 0.0, g_sm.samples = 0;
	g_sm.ran = 0, g_sm.no_pre = 0, g_sm.torch_frames = 0;
}

void ReplaceColor(UpscaleInputs& inputs, const Img& img) {
	inputs.color.image  = vk::Image(img.image);
	inputs.color.view   = img.view;
	inputs.color.layout = vk::ImageLayout::eGeneral;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc |
	                      vk::ImageUsageFlagBits::eStorage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
}

} // namespace

int SmokeLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_SMOKE");
		const int   l     = value != nullptr && *value != '\0' ? std::clamp(std::atoi(value), 0, 2) : 0;
		std::printf("SMOKE: KYTY_SMOKE=%d (%s)\n", l, l == 0 ? "off" : l == 1 ? "soft particles" : "soft particles + torch-lit smoke");
		std::fflush(stdout);
		return l;
	}();
	return level;
}
bool SmokeWanted() { return SmokeLevel() > 0; }

bool SmokeApply(vk::CommandBuffer command, UpscaleInputs& inputs, const UpscaleImage& game) {
	const int level = SmokeLevel();
	if (level == 0) return false;
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr) return false;
	if (VideoOut::MoviePlaying()) {
		g_sm.history_valid = false;
		return false;
	}
	const auto& color = inputs.color;
	if (color.view == nullptr || game.view == nullptr || inputs.depth.view == nullptr || color.extent.width < 16 ||
	    color.extent.height < 16 || inputs.depth.extent != color.extent || game.extent != color.extent)
		return false;
	const auto& m = inputs.view_to_clip;
	if (!(m[0] != 0.0f && m[5] != 0.0f && (m[11] != 0.0f || m[15] != 0.0f)) ||
	    !std::all_of(m.begin(), m.end(), [](float v) { return std::isfinite(v); }))
		return false;
	const float soft_m = EnvFloat("KYTY_SMOKE_SOFT", 0.35f, 0.0f, 2.0f);
	const float glow   = level >= 2 ? EnvFloat("KYTY_SMOKE_GLOW", 1.0f, 0.0f, 3.0f) : 0.0f;
	Log(level, color.extent, soft_m, glow);
	const vk::ImageView pre = RtParticlesPreCopy(inputs.color_address, color.extent);
	if (pre == nullptr) {
		++g_sm.no_pre;
		g_sm.history_valid = false;
		return false;
	}
	if (!EnsurePipeline(*graphics) || !EnsureImages(*graphics, color.extent)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("SMOKE: off (could not create its pipeline or images)\n");
			std::fflush(stdout);
		}
		return false;
	}
	FreeRetired(*graphics);
	ReadTiming(*graphics);
	static const bool row0_up = [] {
		const char* y = std::getenv("KYTY_TORCH_FLIPY"); // (the torch lights' view convention: the same as torch.cpp)
		return y == nullptr || *y == '\0' || *y != '0';
	}();
	static const bool debug = [] {
		const char* v = std::getenv("KYTY_SMOKE_VIEW");
		return v != nullptr && *v == '1';
	}();
	vk::Buffer torch_buf    = nullptr;
	float      torch_radius = 4.5f;
	const bool torch        = glow > 0.0f && TorchLightsThisFlip(torch_buf, torch_radius) && torch_buf != nullptr;
	if (torch) ++g_sm.torch_frames;
	{
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("SMOKE: on (level %d, %ux%u, soft %.2f m, glow %.2f, torch lights %s, exposure %s)\n", level, color.extent.width,
			            color.extent.height, soft_m, glow, torch ? "found" : "not this frame (needs Luz de antorchas RT)",
			            inputs.exposure_buffer != nullptr ? "the game's" : "1.0");
			std::fflush(stdout);
		}
	}
	const auto now = std::chrono::steady_clock::now();
	if (g_sm.last_frame.time_since_epoch().count() != 0 && now - g_sm.last_frame > std::chrono::milliseconds(250)) g_sm.history_valid = false;
	g_sm.last_frame = now;

	const bool game_exposure = inputs.exposure_buffer != nullptr && inputs.exposure_range >= sizeof(float);
	SmokePush  push {};
	push.proj[0] = m[0], push.proj[1] = m[5], push.proj[2] = m[8], push.proj[3] = m[9];
	push.depth_map[0] = m[10], push.depth_map[1] = m[14], push.depth_map[2] = m[11], push.depth_map[3] = m[15];
	push.size[0] = static_cast<float>(color.extent.width), push.size[1] = static_cast<float>(color.extent.height);
	push.size[2] = soft_m > 0.0f ? 1.0f : 0.0f, push.size[3] = glow;
	push.par[0] = std::max(soft_m, 0.01f), push.par[1] = 24.0f * static_cast<float>(color.extent.height) / 1440.0f;
	push.par[1] = std::clamp(push.par[1], 6.0f, 32.0f);
	push.par[2] = std::clamp(torch_radius, 1.0f, 24.0f), push.par[3] = 0.6f;
	push.motion[0] = inputs.motion_scale_x, push.motion[1] = inputs.motion_scale_y;
	push.motion[2] = (g_sm.history_valid && !inputs.reset && inputs.motion.view != nullptr) ? 1.0f : 0.0f;
	push.misc[1] = static_cast<int32_t>(g_sm.frame & 0x7fffffff);
	push.misc[2] = (row0_up ? 1 : 0) | (debug ? 2 : 0) | (torch ? 4 : 0) | (game_exposure ? 8 : 0);
	push.misc[3] = game_exposure ? static_cast<int32_t>(inputs.exposure_index) : 0;

	const uint32_t entry       = static_cast<uint32_t>(g_sm.written % 8);
	const bool     timing_open = g_sm.pool != nullptr && g_sm.written < g_sm.read + 8;
	if (timing_open) {
		command.resetQueryPool(g_sm.pool, entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_sm.pool, entry * 2);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite);
	if (!g_sm.ready) {
		Img* all[] {&g_sm.out, &g_sm.a, &g_sm.b, &g_sm.hist[0], &g_sm.hist[1]};
		for (Img* img: all) ToGeneral(command, img->image);
		g_sm.ready = true, g_sm.history_valid = false;
		push.motion[2] = 0.0f;
	}
	if (!g_sm.cleared) {
		g_sm.cleared = true;
		command.fillBuffer(vk::Buffer(g_sm.stats_buf), 0, VK_WHOLE_SIZE, 0);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
		        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	}

	const uint32_t cur = static_cast<uint32_t>(g_sm.frame & 1u), prev = cur ^ 1u;
	const auto     gen = vk::ImageLayout::eGeneral;
	const vk::DescriptorBufferInfo stats_info {vk::Buffer(g_sm.stats_buf), 0, VK_WHOLE_SIZE};
	const vk::DescriptorBufferInfo torch_info = torch ? vk::DescriptorBufferInfo {torch_buf, 0, VK_WHOLE_SIZE} : stats_info;
	const vk::DescriptorBufferInfo exposure_info =
	    game_exposure ? vk::DescriptorBufferInfo {inputs.exposure_buffer, inputs.exposure_base, inputs.exposure_range} : stats_info;
	const auto record = [&](int mode, const Img& src, const Img& hist, const Img& aux) {
		vk::DescriptorImageInfo infos[9] {};
		infos[0] = {g_sm.point, color.view, color.layout};
		infos[1] = {g_sm.point, inputs.depth.view, inputs.depth.layout};
		infos[2] = {g_sm.point, game.view, game.layout};
		infos[3] = {g_sm.point, pre, vk::ImageLayout::eShaderReadOnlyOptimal};
		infos[4] = inputs.motion.view != nullptr ? vk::DescriptorImageInfo {g_sm.point, inputs.motion.view, inputs.motion.layout}
		                                         : vk::DescriptorImageInfo {g_sm.point, g_sm.b.view, gen};
		infos[5] = {g_sm.point, src.view, gen};
		infos[6] = {g_sm.linear, hist.view, gen};
		infos[7] = {nullptr, g_sm.out.view, gen};
		infos[8] = {nullptr, aux.view, gen};
		vk::WriteDescriptorSet writes[Bindings] {};
		for (uint32_t i = 0; i < Bindings; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = DescType(i);
			if (i == 9) writes[i].pBufferInfo = &torch_info;
			else if (i == 10) writes[i].pBufferInfo = &exposure_info;
			else if (i == 11) writes[i].pBufferInfo = &stats_info;
			else writes[i].pImageInfo = &infos[i];
		}
		command.bindPipeline(vk::PipelineBindPoint::eCompute, g_sm.pipeline);
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_sm.layout, 0, Bindings, writes);
		push.misc[0] = mode;
		command.pushConstants(g_sm.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch((color.extent.width + 7) / 8, (color.extent.height + 7) / 8, 1);
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	};
	// edge -> a; rows a -> b; cols b (+ last fade) -> hist[cur]; composite hist[cur] -> out
	record(0, g_sm.b, g_sm.hist[prev], g_sm.a);
	record(1, g_sm.a, g_sm.hist[prev], g_sm.b);
	record(2, g_sm.b, g_sm.hist[prev], g_sm.hist[cur]);
	record(3, g_sm.hist[cur], g_sm.hist[prev], g_sm.a);

	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eMemoryRead);
	vk::BufferCopy region {0, 0, StatsCount * 4};
	command.copyBuffer(vk::Buffer(g_sm.stats_buf), vk::Buffer(g_sm.read_buf), 1, &region);
	Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
	        vk::AccessFlagBits2::eHostRead);
	if (timing_open) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_sm.pool, entry * 2 + 1);
		++g_sm.written;
	}
	g_sm.history_valid = true;
	++g_sm.frame;
	++g_sm.ran;
	ReplaceColor(inputs, g_sm.out);
	return true;
}

} // namespace Libs::Graphics::FrameGen
