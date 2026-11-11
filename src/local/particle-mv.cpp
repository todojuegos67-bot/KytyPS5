// KYTY_FG_PARTICLE_MV=0|1 (particle-mv.comp): particle motion vectors for frame generation (Souls ligero 30x2 / ligero + 30x3).
//
// Why: Demon's Souls draws every particle (fire, embers, sparks, rain - its drops are .cprt particles with
// particles/environment/water textures -, mist, spells) in one particle-translucent pass with depth writes and motion
// vectors off; the motion vector under a flame is the wall's (zero for a static wall). FSR 3.1 frame generation takes
// no reactive / composition mask (only a HUD-less image and a UI layer), so it moved the particles with the wall's
// vector: at 30 real fps they showed at 30. Telas / capas are skinned meshes drawn in the G-buffer with their own
// vectors (not this pass): not touched here.
// How: KYTY_RT_PARTICLES' hook (frame-gen.cpp BeforeSceneDraw) copies the scene colour before the pass's first draw;
// game - copy is the particle layer. At half resolution that layer is block matched against the previous real frame's
// (5x5 footprint, +-KYTY_FG_PARTICLE_MV_RANGE half-res px, default 24 = 48 render px a real frame), with a confidence
// (better than the background's vector and than standing still, on a real signal, over the coverage threshold). At the
// guest flip, after frame generation's own copy of the motion vectors, the particle vector is mixed into THAT copy
// (weight x KYTY_FG_PARTICLE_MV_STRENGTH, 1.0): the game's motion image and the upscaler's input are never written.
// Cost: ~0.2-0.5 ms (one half-res match only on particle pixels, one full-res compose) a real frame.
// Every 10 s "FG particle MV:" stats (particle px, matched px, mean |d|, composed px, GPU ms).
#include "local/particle-mv.h"

#include "local/local-pass-cache.h"
#include "local/smoke.h" // RtParticlesPreCopy

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/particle_mv_spv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::FrameGen {
namespace {

constexpr uint32_t Bindings   = 9;
constexpr uint32_t StatsCount = 8;

struct Push {
	float   size[4];
	float   motion[4];
	float   par[4];
	int32_t misc[4];
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
	Img                     lay[2], pmv, dummy; // dummy: 1x1 RG16F, binding 7 outside the compose
	vk::Extent2D            full {}, half {};
	bool                    ready = false, history_valid = false;
	bool                    measured = false; // this frame's pmv is valid for the compose
	float                   scale[2] {};
	VkBuffer                stats_buf = VK_NULL_HANDLE, read_buf = VK_NULL_HANDLE;
	VmaAllocation           stats_mem = nullptr, read_mem = nullptr;
	const uint32_t*         mapped = nullptr;
	bool                    cleared = false;
	uint64_t                frame = 0, ran = 0, no_pre = 0, composed = 0, not_storage = 0;
	vk::Image               fg_image = nullptr;
	vk::ImageView           fg_view  = nullptr;
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms = 0.0, max_ms = 0.0;
	uint32_t                samples = 0;
	std::array<uint32_t, StatsCount> stat_last {};
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now(), last_frame {};
} g_pm;

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
void Layout(vk::CommandBuffer command, vk::Image image, vk::ImageLayout from, vk::ImageLayout to) {
	vk::ImageMemoryBarrier2 b {};
	b.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	b.srcAccessMask       = vk::AccessFlagBits2::eMemoryWrite;
	b.dstStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	b.dstAccessMask       = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	b.oldLayout           = from;
	b.newLayout           = to;
	b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image               = image;
	b.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &b;
	command.pipelineBarrier2(dependency);
}

bool MakeImage(GraphicContext& graphics, vk::Extent2D extent, Img& img, vk::Format format = vk::Format::eR16G16B16A16Sfloat) {
	VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create.imageType     = VK_IMAGE_TYPE_2D;
	create.format        = static_cast<VkFormat>(format);
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.samples       = VK_SAMPLE_COUNT_1_BIT;
	create.tiling        = VK_IMAGE_TILING_OPTIMAL;
	create.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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

vk::DescriptorType DescType(uint32_t i) {
	return i == 8 ? vk::DescriptorType::eStorageBuffer : i >= 6 ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eCombinedImageSampler;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_pm.pipeline != nullptr) return true;
	if (g_pm.failed) return false;
	g_pm.failed       = true;
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
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_pm.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Push)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_pm.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_pm.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(PARTICLE_MV_SPV);
	module_info.pCode       = PARTICLE_MV_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_pm.layout;
	const auto created         = device.createComputePipelines(LocalPassCache::Get(device), 1, &pipeline_info, nullptr, &g_pm.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_pm.pipeline = nullptr;
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
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g_pm.point : &g_pm.linear) != vk::Result::eSuccess) return false;
	}
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = StatsCount * 4;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VmaAllocationCreateInfo dev {};
	dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_pm.stats_buf, &g_pm.stats_mem, nullptr) != VK_SUCCESS) {
		g_pm.stats_buf = VK_NULL_HANDLE;
		return false;
	}
	buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_pm.read_buf, &g_pm.read_mem, &mapped) != VK_SUCCESS) {
		g_pm.read_buf = VK_NULL_HANDLE;
		return false;
	}
	g_pm.mapped = static_cast<const uint32_t*>(mapped.pMappedData);
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = 8 * 2;
	if (device.createQueryPool(&query, nullptr, &g_pm.pool) != vk::Result::eSuccess) g_pm.pool = nullptr;
	g_pm.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g_pm.failed    = false;
	return true;
}

bool EnsureImages(GraphicContext& graphics, vk::Extent2D full) {
	if (g_pm.pmv.image != VK_NULL_HANDLE && g_pm.full == full) return true;
	if (g_pm.pmv.image != VK_NULL_HANDLE) graphics.device.waitIdle(); // (a resolution change: rare)
	Img* all[] {&g_pm.lay[0], &g_pm.lay[1], &g_pm.pmv};
	for (Img* img: all) Destroy(graphics, *img);
	if (g_pm.dummy.image == VK_NULL_HANDLE && !MakeImage(graphics, vk::Extent2D {1, 1}, g_pm.dummy, vk::Format::eR16G16Sfloat)) return false;
	g_pm.ready = false, g_pm.history_valid = false, g_pm.measured = false;
	g_pm.full = full;
	g_pm.half = vk::Extent2D {(full.width + 1) / 2, (full.height + 1) / 2};
	for (Img* img: all)
		if (!MakeImage(graphics, g_pm.half, *img)) return false;
	return true;
}

void ReadTiming(GraphicContext& graphics) {
	if (g_pm.pool == nullptr) return;
	while (g_pm.read + 4 <= g_pm.written) {
		const auto              entry = static_cast<uint32_t>(g_pm.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (graphics.device.getQueryPoolResults(g_pm.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		if (stamps[1] > stamps[0]) {
			const double ms = static_cast<double>(stamps[1] - stamps[0]) * g_pm.period_ns / 1e6;
			g_pm.sum_ms += ms, g_pm.max_ms = std::max(g_pm.max_ms, ms);
			++g_pm.samples;
		}
		++g_pm.read;
	}
}

void Log() {
	const auto now = std::chrono::steady_clock::now();
	if (now - g_pm.last_log < std::chrono::seconds(10) || g_pm.mapped == nullptr) return;
	g_pm.last_log = now;
	std::array<uint32_t, StatsCount> cur {};
	std::memcpy(cur.data(), g_pm.mapped, sizeof(cur)); // (a few frames late: statistics only)
	std::array<uint32_t, StatsCount> d {};
	for (uint32_t i = 0; i < StatsCount; ++i) d[i] = cur[i] - g_pm.stat_last[i];
	g_pm.stat_last  = cur;
	const double fr   = std::max<double>(static_cast<double>(g_pm.ran), 1.0);
	const double half = std::max<double>(static_cast<double>(g_pm.half.width) * g_pm.half.height, 1.0);
	const double full = std::max<double>(static_cast<double>(g_pm.full.width) * g_pm.full.height, 1.0);
	std::printf("FG particle MV: stats %llu frames measured (%llu without the pre-particle copy), %llu composed into the FG vectors "
	            "(%llu skipped: motion copy not storage RG16F); particle px %.2f %%, matched (moving) %.2f %% of the particle px, "
	            "mean |d| %.1f render px a real frame, composed px %.2f %%; GPU %.3f ms (max %.3f)\n",
	            static_cast<unsigned long long>(g_pm.ran), static_cast<unsigned long long>(g_pm.no_pre),
	            static_cast<unsigned long long>(g_pm.composed), static_cast<unsigned long long>(g_pm.not_storage),
	            100.0 * d[1] / (half * fr), d[1] > 0 ? 100.0 * d[2] / d[1] : 0.0, d[2] > 0 ? 0.5 * d[3] / d[2] : 0.0,
	            g_pm.composed > 0 ? 100.0 * d[4] / (full * static_cast<double>(g_pm.composed)) : 0.0,
	            g_pm.samples > 0 ? g_pm.sum_ms / g_pm.samples : 0.0, g_pm.max_ms);
	std::fflush(stdout);
	g_pm.sum_ms = 0.0, g_pm.max_ms = 0.0, g_pm.samples = 0;
	g_pm.ran = 0, g_pm.no_pre = 0, g_pm.composed = 0, g_pm.not_storage = 0;
}

void Dispatch(vk::CommandBuffer command, Push& push, int mode, const vk::DescriptorImageInfo (&infos)[8], vk::Extent2D grid) {
	const vk::DescriptorBufferInfo stats_info {vk::Buffer(g_pm.stats_buf), 0, VK_WHOLE_SIZE};
	vk::WriteDescriptorSet         writes[Bindings] {};
	for (uint32_t i = 0; i < Bindings; ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = DescType(i);
		if (i == 8) writes[i].pBufferInfo = &stats_info;
		else writes[i].pImageInfo = &infos[i];
	}
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_pm.pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_pm.layout, 0, Bindings, writes);
	push.misc[0] = mode;
	command.pushConstants(g_pm.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch((grid.width + 7) / 8, (grid.height + 7) / 8, 1);
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferRead);
}

Push BasePush() {
	Push push {};
	push.size[0] = static_cast<float>(g_pm.full.width), push.size[1] = static_cast<float>(g_pm.full.height);
	push.size[2] = static_cast<float>(g_pm.half.width), push.size[3] = static_cast<float>(g_pm.half.height);
	push.motion[0] = g_pm.scale[0], push.motion[1] = g_pm.scale[1];
	static const float range = EnvFloat("KYTY_FG_PARTICLE_MV_RANGE", 24.0f, 2.0f, 48.0f);
	static const float thr   = EnvFloat("KYTY_FG_PARTICLE_MV_THRESHOLD", 0.08f, 0.01f, 0.9f);
	static const float str   = EnvFloat("KYTY_FG_PARTICLE_MV_STRENGTH", 1.0f, 0.0f, 1.0f);
	push.motion[2] = std::floor(range / 2.0f) * 2.0f, push.motion[3] = thr;
	push.par[0] = str;
	return push;
}

} // namespace

bool ParticleMvWanted() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_FG_PARTICLE_MV");
		const bool  v     = value != nullptr && *value == '1';
		std::printf("FG particle MV: KYTY_FG_PARTICLE_MV=%d (%s)\n", v ? 1 : 0,
		            v ? "particle vectors measured and mixed into frame generation's motion copy" : "off");
		std::fflush(stdout);
		return v;
	}();
	return on;
}

bool ParticleMvMeasure(vk::CommandBuffer command, const UpscaleInputs& inputs, const UpscaleImage& game) {
	g_pm.measured = false;
	if (!ParticleMvWanted()) return false;
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr) return false;
	if (VideoOut::MoviePlaying() || inputs.reset) {
		g_pm.history_valid = false;
		return false;
	}
	const auto extent = game.extent;
	if (game.view == nullptr || inputs.motion.view == nullptr || extent.width < 16 || extent.height < 16 || inputs.motion.extent != extent ||
	    inputs.motion_scale_x == 0.0f || inputs.motion_scale_y == 0.0f || !std::isfinite(inputs.motion_scale_x) ||
	    !std::isfinite(inputs.motion_scale_y))
		return false;
	const vk::ImageView pre = RtParticlesPreCopy(inputs.color_address, extent);
	if (pre == nullptr) {
		++g_pm.no_pre;
		g_pm.history_valid = false;
		Log();
		return false;
	}
	if (!EnsurePipeline(*graphics) || !EnsureImages(*graphics, extent)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("FG particle MV: off (could not create its pipeline or images)\n");
			std::fflush(stdout);
		}
		return false;
	}
	ReadTiming(*graphics);
	Log();
	const auto now = std::chrono::steady_clock::now();
	if (g_pm.last_frame.time_since_epoch().count() != 0 && now - g_pm.last_frame > std::chrono::milliseconds(250)) g_pm.history_valid = false;
	g_pm.last_frame = now;
	g_pm.scale[0] = inputs.motion_scale_x, g_pm.scale[1] = inputs.motion_scale_y;
	{
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("FG particle MV: on (particle layer %ux%u, matched at %ux%u, motion scale %g %g)\n", extent.width, extent.height,
			            g_pm.half.width, g_pm.half.height, inputs.motion_scale_x, inputs.motion_scale_y);
			std::fflush(stdout);
		}
	}
	const uint32_t entry       = static_cast<uint32_t>(g_pm.written % 8);
	const bool     timing_open = g_pm.pool != nullptr && g_pm.written < g_pm.read + 8;
	if (timing_open) {
		command.resetQueryPool(g_pm.pool, entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_pm.pool, entry * 2);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite);
	if (!g_pm.ready) {
		for (Img* img: {&g_pm.lay[0], &g_pm.lay[1], &g_pm.pmv, &g_pm.dummy})
			Layout(command, vk::Image(img->image), vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
		g_pm.ready = true, g_pm.history_valid = false;
	}
	if (!g_pm.cleared) {
		g_pm.cleared = true;
		command.fillBuffer(vk::Buffer(g_pm.stats_buf), 0, VK_WHOLE_SIZE, 0);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
		        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	}
	const uint32_t cur = static_cast<uint32_t>(g_pm.frame & 1u), prev = cur ^ 1u;
	const auto     gen  = vk::ImageLayout::eGeneral;
	Push           push = BasePush();
	// Binding 7 (the FG motion copy, rg16f) is unused here: the 1x1 dummy stands in.
	vk::DescriptorImageInfo infos[8] {
	    {g_pm.point, game.view, game.layout},
	    {g_pm.point, pre, vk::ImageLayout::eShaderReadOnlyOptimal},
	    {g_pm.point, g_pm.lay[cur].view, gen},
	    {g_pm.point, g_pm.lay[prev].view, gen},
	    {g_pm.linear, inputs.motion.view, inputs.motion.layout},
	    {g_pm.linear, g_pm.pmv.view, gen},
	    {nullptr, g_pm.lay[cur].view, gen},
	    {nullptr, g_pm.dummy.view, gen},
	};
	infos[2] = {g_pm.point, g_pm.pmv.view, gen}; // mode 0 reads no layer: keep lay[cur] out of the sampled set while written
	Dispatch(command, push, 0, infos, g_pm.half);
	if (g_pm.history_valid) {
		infos[2] = {g_pm.point, g_pm.lay[cur].view, gen};
		infos[6] = {nullptr, g_pm.pmv.view, gen};
		infos[5] = {g_pm.linear, g_pm.lay[prev].view, gen};
		Dispatch(command, push, 1, infos, g_pm.half);
		g_pm.measured = true;
	}
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eMemoryRead);
	vk::BufferCopy region {0, 0, StatsCount * 4};
	command.copyBuffer(vk::Buffer(g_pm.stats_buf), vk::Buffer(g_pm.read_buf), 1, &region);
	Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
	        vk::AccessFlagBits2::eHostRead);
	if (timing_open) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_pm.pool, entry * 2 + 1);
		++g_pm.written;
	}
	g_pm.history_valid = true;
	++g_pm.frame;
	++g_pm.ran;
	return g_pm.measured;
}

bool ParticleMvCompose(vk::CommandBuffer command, GraphicContext& graphics, VulkanImage& motion) {
	if (!ParticleMvWanted() || !g_pm.measured || g_pm.pipeline == nullptr) return false;
	g_pm.measured = false; // (once per measured frame)
	if (motion.image == nullptr || motion.extent.width != g_pm.full.width || motion.extent.height != g_pm.full.height) return false;
	if (motion.format != vk::Format::eR16G16Sfloat || !(motion.usage & vk::ImageUsageFlagBits::eStorage)) {
		++g_pm.not_storage;
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("FG particle MV: the motion copy is format %d without storage use: not composed\n", static_cast<int>(motion.format));
			std::fflush(stdout);
		}
		return false;
	}
	if (g_pm.fg_image != motion.image) {
		if (g_pm.fg_view != nullptr) {
			graphics.device.waitIdle();
			graphics.device.destroyImageView(g_pm.fg_view, nullptr);
			g_pm.fg_view = nullptr;
		}
		vk::ImageViewCreateInfo view_info {};
		view_info.image            = motion.image;
		view_info.viewType         = vk::ImageViewType::e2D;
		view_info.format           = vk::Format::eR16G16Sfloat;
		view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		if (graphics.device.createImageView(&view_info, nullptr, &g_pm.fg_view) != vk::Result::eSuccess) {
			g_pm.fg_view = nullptr;
			return false;
		}
		g_pm.fg_image = motion.image;
	}
	const auto layout = motion.state.layout;
	const auto gen    = vk::ImageLayout::eGeneral;
	Layout(command, motion.image, layout, gen);
	Push push = BasePush();
	const vk::DescriptorImageInfo infos[8] {
	    {g_pm.point, g_pm.pmv.view, gen},  {g_pm.point, g_pm.pmv.view, gen}, {g_pm.point, g_pm.lay[0].view, gen},
	    {g_pm.point, g_pm.lay[1].view, gen}, {g_pm.point, g_pm.lay[0].view, gen}, {g_pm.linear, g_pm.pmv.view, gen},
	    {nullptr, g_pm.lay[0].view, gen},  {nullptr, g_pm.fg_view, gen},
	};
	Dispatch(command, push, 2, infos, g_pm.full);
	Layout(command, motion.image, gen, layout);
	++g_pm.composed;
	return true;
}

} // namespace Libs::Graphics::FrameGen
