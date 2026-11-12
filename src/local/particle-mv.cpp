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

#include "graphics/host_gpu/renderer/render.h"
#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/particle_mv_spv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace Libs::Graphics::FrameGen {
namespace {

constexpr uint32_t Bindings   = 15;
constexpr uint32_t StatsCount = 16;

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
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms = 0.0, max_ms = 0.0;
	uint32_t                samples = 0;
	std::array<uint32_t, StatsCount> stat_last {};
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now(), last_frame {};
} g_pm;


// ---- (des-particles-native) KYTY_FG_PARTICLE_NATIVE ----
struct NBuf {
	VkBuffer      buffer = VK_NULL_HANDLE;
	VmaAllocation memory = nullptr;
	uint64_t      size   = 0;
	void*         mapped = nullptr;
};
struct NKey { // one particle record buffer (guest address): this frame's copy and last real frame's, ping-pong
	uint64_t address = 0;
	NBuf     buf[2];
	int      cur = 0;
	uint64_t copied_serial = UINT64_MAX, prev_serial = UINT64_MAX; // serial of buf[cur] / buf[cur ^ 1]
	uint32_t stride = 0, count = 0, prev_stride = 0, prev_count = 0;
	float    vp_prev[16] {};
	bool     vp_prev_valid = false;
	float    vp_last[16] {};
};
struct NDraw {
	int      key = -1;
	uint64_t sort_off = 0, emit_off = 0, args_off = 0, sort_size = 0, emit_size = 0;
	uint32_t sort_stride = 0, sort_count = 0, emit_stride = 0, emit_count = 0, base = 0, instances = 0, first_word = 0;
	bool     indirect = false, prev_valid = false;
	float    vp[16] {}, vpp[16] {};
};
constexpr uint32_t NKeys = 4, NDrawsMax = 16, NParamSlot = 256, NParamSlots = 128;
constexpr uint64_t NRingSize = 8ull << 20, NRecMax = 64ull << 20;
struct {
	bool                 failed = false;
	NBuf                 ring, params; // ring: this frame's sort lists / emitter tables / indirect arguments; params: host-visible
	uint64_t             ring_used = 0;
	uint32_t             param_next = 0;
	std::array<NKey, NKeys> keys {};
	std::vector<NDraw>   draws;
	std::vector<std::pair<uint64_t, NBuf>> graveyard; // (serial, buffer) freed a few frames later
	uint64_t             serial = 0;                  // real frames measured
	uint64_t             alignment = 256;
	// log (every second)
	uint64_t draws_seen = 0, draws_native = 0, draws_skipped = 0, frames = 0;
	uint64_t hits_down = 0, hits_up = 0;
	bool     y_down = true;
	std::array<uint32_t, StatsCount> stat_last {};
	double   sum_ms = 0.0, max_ms = 0.0;
	uint32_t samples = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
} g_pn;

bool MakeBuffer(GraphicContext& graphics, uint64_t size, bool host, NBuf& out) {
	VkBufferCreateInfo info {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	info.size  = size;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VmaAllocationCreateInfo alloc {};
	alloc.usage = host ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (host) alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &info, &alloc, &out.buffer, &out.memory, &mapped) != VK_SUCCESS) {
		out = {};
		return false;
	}
	out.size   = size;
	out.mapped = mapped.pMappedData;
	return true;
}
void FreeLater(NBuf& b) {
	if (b.buffer != VK_NULL_HANDLE) g_pn.graveyard.emplace_back(g_pn.serial, b);
	b = {};
}
void FreeOld(GraphicContext& graphics) {
	auto& g = g_pn.graveyard;
	for (size_t i = 0; i < g.size();) {
		if (g_pn.serial >= g[i].first + 8) {
			vmaDestroyBuffer(graphics.allocator, g[i].second.buffer, g[i].second.memory);
			g[i] = g.back();
			g.pop_back();
		} else {
			++i;
		}
	}
}
uint64_t AlignUp(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

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
	return i >= 8 ? vk::DescriptorType::eStorageBuffer : i >= 6 ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eCombinedImageSampler;
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
	if (ParticleNativeWanted()) return; // (its own line every second)
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

void Dispatch(vk::CommandBuffer command, Push& push, int mode, const vk::DescriptorImageInfo (&infos)[8], vk::Extent2D grid,
              const vk::DescriptorBufferInfo* native = nullptr, uint32_t groups_x = 0, uint32_t groups_y = 0) {
	vk::DescriptorBufferInfo buffers[Bindings - 8] {};
	buffers[0] = {vk::Buffer(g_pm.stats_buf), 0, VK_WHOLE_SIZE};
	for (uint32_t i = 1; i < Bindings - 8; ++i)
		buffers[i] = native != nullptr ? native[i - 1] : vk::DescriptorBufferInfo {vk::Buffer(g_pm.stats_buf), 0, VK_WHOLE_SIZE};
	vk::WriteDescriptorSet writes[Bindings] {};
	for (uint32_t i = 0; i < Bindings; ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = DescType(i);
		if (i >= 8) writes[i].pBufferInfo = &buffers[i - 8];
		else writes[i].pImageInfo = &infos[i];
	}
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_pm.pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_pm.layout, 0, Bindings, writes);
	push.misc[0] = mode;
	command.pushConstants(g_pm.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	if (groups_x != 0) command.dispatch(groups_x, groups_y, 1);
	else command.dispatch((grid.width + 7) / 8, (grid.height + 7) / 8, 1);
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferRead);
}


void NativeLog(GraphicContext& graphics);
// After the frame's measure: this frame's view-projection becomes the next one's previous; the draws and the ring reset.
void NativeEndFrame(GraphicContext& graphics) {
	for (NKey& key: g_pn.keys)
		if (key.address != 0 && key.copied_serial == g_pn.serial) {
			std::memcpy(key.vp_prev, key.vp_last, sizeof(key.vp_prev));
			key.vp_prev_valid = true;
		}
	g_pn.draws.clear();
	g_pn.ring_used = 0;
	++g_pn.serial;
	++g_pn.frames;
	FreeOld(graphics);
	NativeLog(graphics);
}

void NativeLog(GraphicContext& graphics) {
	(void)graphics;
	const auto now = std::chrono::steady_clock::now();
	if (now - g_pn.last_log < std::chrono::seconds(1) || g_pm.mapped == nullptr) return;
	g_pn.last_log = now;
	std::array<uint32_t, StatsCount> cur {};
	std::memcpy(cur.data(), g_pm.mapped, sizeof(cur)); // (a few frames late: statistics only)
	std::array<uint32_t, StatsCount> d {};
	for (uint32_t i = 0; i < StatsCount; ++i) d[i] = cur[i] - g_pn.stat_last[i];
	g_pn.stat_last = cur;
	g_pn.hits_down += d[8], g_pn.hits_up += d[9];
	// Which way the screen's y runs: the side whose particle centres land on covered pixels (decided after a few hundred).
	if (g_pn.hits_down + g_pn.hits_up > 200) g_pn.y_down = g_pn.hits_down >= g_pn.hits_up;
	std::printf("FG particle native: %llu frames, %llu particle draws (%llu with native vectors, %llu skipped); particles %u, with a "
	            "vector %u, recycled slots discarded %u; particle px with a native vector %.1f %% (%u of %u half-res px); y %s "
	            "(%llu / %llu); GPU %.3f ms (max %.3f)\n",
	            static_cast<unsigned long long>(g_pn.frames), static_cast<unsigned long long>(g_pn.draws_seen),
	            static_cast<unsigned long long>(g_pn.draws_native), static_cast<unsigned long long>(g_pn.draws_skipped), d[10], d[5], d[6],
	            d[11] > 0 ? 100.0 * d[7] / d[11] : 0.0, d[7], d[11], g_pn.y_down ? "down" : "up",
	            static_cast<unsigned long long>(g_pn.hits_down), static_cast<unsigned long long>(g_pn.hits_up),
	            g_pm.samples > 0 ? g_pm.sum_ms / g_pm.samples : 0.0, g_pm.max_ms);
	std::fflush(stdout);
	g_pm.sum_ms = 0.0, g_pm.max_ms = 0.0, g_pm.samples = 0;
	g_pn.frames = 0, g_pn.draws_seen = 0, g_pn.draws_native = 0, g_pn.draws_skipped = 0;
}

struct NParams {
	float    vp[16], vpp[16];
	uint32_t sorti[4], reci[4], argi[4];
	float    parf[4];
};
static_assert(sizeof(NParams) == 192);

// This frame's captured draws into pmv (cleared first). False: none ran.
bool NativeRun(GraphicContext& graphics, vk::CommandBuffer command, Push& push, vk::DescriptorImageInfo (&infos)[8]) {
	if (g_pn.draws.empty() || g_pn.params.mapped == nullptr) return false;
	static const float max_step = EnvFloat("KYTY_FG_PARTICLE_NATIVE_MAX_STEP", 3.0f, 0.01f, 1000.0f);
	static const float max_quad = EnvFloat("KYTY_FG_PARTICLE_NATIVE_MAX_QUAD", 160.0f, 4.0f, 1024.0f);
	const vk::ClearColorValue zero(std::array<float, 4> {0.0f, 0.0f, 0.0f, 0.0f});
	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	command.clearColorImage(vk::Image(g_pm.pmv.image), vk::ImageLayout::eGeneral, &zero, 1, &range);
	Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	const uint64_t slot_size = AlignUp(NParamSlot, g_pn.alignment);
	const vk::Buffer ring(g_pn.ring.buffer), stats(g_pm.stats_buf);
	bool ran = false;
	for (const NDraw& nd: g_pn.draws) {
		const NKey& key = g_pn.keys[nd.key];
		if (key.buf[key.cur].buffer == VK_NULL_HANDLE) continue;
		const uint32_t slot = g_pn.param_next++ % NParamSlots;
		NParams        prm {};
		std::memcpy(prm.vp, nd.vp, sizeof(prm.vp));
		std::memcpy(prm.vpp, nd.vpp, sizeof(prm.vpp));
		prm.sorti[0] = nd.sort_stride, prm.sorti[1] = nd.sort_count, prm.sorti[2] = nd.base, prm.sorti[3] = nd.prev_valid ? key.prev_count : 0;
		prm.reci[0] = key.stride, prm.reci[1] = key.count, prm.reci[2] = nd.prev_valid ? 1u : 0u, prm.reci[3] = nd.emit_stride;
		prm.argi[0] = nd.indirect ? 1u : 0u, prm.argi[1] = nd.instances, prm.argi[2] = nd.emit_count, prm.argi[3] = nd.first_word;
		prm.parf[0] = max_step, prm.parf[1] = max_quad, prm.parf[2] = g_pn.y_down ? 1.0f : 0.0f;
		std::memcpy(static_cast<uint8_t*>(g_pn.params.mapped) + slot * slot_size, &prm, sizeof(prm));
		vmaFlushAllocation(graphics.allocator, g_pn.params.memory, slot * slot_size, sizeof(prm));
		const VkBuffer prev = key.buf[key.cur ^ 1].buffer != VK_NULL_HANDLE ? key.buf[key.cur ^ 1].buffer : key.buf[key.cur].buffer;
		const vk::DescriptorBufferInfo native[6] {
		    {ring, nd.sort_off, nd.sort_size},
		    {vk::Buffer(key.buf[key.cur].buffer), 0, VK_WHOLE_SIZE},
		    {vk::Buffer(prev), 0, VK_WHOLE_SIZE},
		    nd.emit_size > 0 ? vk::DescriptorBufferInfo {ring, nd.emit_off, nd.emit_size} : vk::DescriptorBufferInfo {stats, 0, VK_WHOLE_SIZE},
		    nd.indirect ? vk::DescriptorBufferInfo {ring, nd.args_off, 64} : vk::DescriptorBufferInfo {stats, 0, VK_WHOLE_SIZE},
		    {vk::Buffer(g_pn.params.buffer), slot * slot_size, sizeof(NParams)},
		};
		const uint32_t n = std::min<uint32_t>(nd.indirect ? nd.sort_count : nd.instances, 65535u * 64u);
		if (n == 0) continue;
		Dispatch(command, push, 3, infos, g_pm.half, native, std::min(n, 65535u), (n + 65534u) / 65535u);
		++g_pn.draws_native;
		ran = true;
	}
	return ran;
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

bool ParticleNativeWanted() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_FG_PARTICLE_NATIVE");
		const bool  v     = value != nullptr && *value == '1';
		std::printf("FG particle native: KYTY_FG_PARTICLE_NATIVE=%d (%s)\n", v ? 1 : 0,
		            v ? "the particle pass's own simulated positions (this and last real frame) as frame generation's particle vectors"
		              : "off");
		std::fflush(stdout);
		return v;
	}();
	return on;
}

bool ParticleNativeVs(uint64_t vs_hash) {
	static const std::vector<uint64_t> list = [] {
		std::vector<uint64_t> out;
		const char*           v = std::getenv("KYTY_FG_PARTICLE_NATIVE_VS");
		if (v == nullptr || *v == '\0') v = "a64d43c72c1c5c87,b6761454c17bea3c";
		for (const char* at = v; *at != '\0';) {
			char*                    end = nullptr;
			const unsigned long long h   = std::strtoull(at, &end, 16);
			if (end == at) break;
			if (h != 0) out.push_back(h);
			if (*end != ',' && *end != ';') break;
			at = end + 1;
		}
		return out;
	}();
	return std::find(list.begin(), list.end(), vs_hash) != list.end();
}

bool ParticleMvWanted() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_FG_PARTICLE_MV");
		const bool  v     = (value != nullptr && *value == '1') || ParticleNativeWanted(); // (native: the same layer / compose)
		std::printf("FG particle MV: KYTY_FG_PARTICLE_MV=%d (%s)\n", v ? 1 : 0,
		            v ? "particle vectors measured and mixed into frame generation's motion copy" : "off");
		std::fflush(stdout);
		return v;
	}();
	return on;
}


void ParticleNativeCapture(CommandBuffer& command, const RenderState& rendering, const ParticleNativeDraw& d) {
	if (!ParticleNativeWanted()) return;
	++g_pn.draws_seen;
	const uint64_t sort_size = std::min<uint64_t>(d.sort.size, uint64_t {d.sort_count} * d.sort_stride);
	const uint64_t emit_size = d.emit.buffer != nullptr ? std::min<uint64_t>(d.emit.size, uint64_t {d.emit_count} * d.emit_stride) : 0;
	const uint64_t args_size = d.indirect ? std::min<uint64_t>(d.args.size, 64) : 0;
	if (g_pn.failed || g_pn.draws.size() >= NDrawsMax || d.rec.buffer == nullptr || d.sort.buffer == nullptr || d.rec_stride < 32 ||
	    (d.rec_stride & 3u) != 0 || d.sort_stride < 8 || (d.sort_stride & 3u) != 0 || d.rec.size == 0 || d.rec.size > NRecMax ||
	    sort_size == 0 || (d.indirect && args_size < 8) || (d.emit_stride & 3u) != 0) {
		++g_pn.draws_skipped;
		return;
	}
	auto& graphics = command.GetGraphics();
	if (g_pn.ring.buffer == VK_NULL_HANDLE) {
		g_pn.alignment = std::max<uint64_t>(graphics.GetPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment, 16);
		if (!MakeBuffer(graphics, NRingSize, false, g_pn.ring) ||
		    !MakeBuffer(graphics, AlignUp(NParamSlot, g_pn.alignment) * NParamSlots, true, g_pn.params) || g_pn.params.mapped == nullptr) {
			g_pn.failed = true;
			std::printf("FG particle native: off (could not create its buffers)\n");
			std::fflush(stdout);
			return;
		}
	}
	const uint64_t a        = g_pn.alignment;
	const uint64_t sort_off = AlignUp(g_pn.ring_used, a), emit_off = AlignUp(sort_off + sort_size, a), args_off = AlignUp(emit_off + emit_size, a);
	if (args_off + args_size > NRingSize) {
		++g_pn.draws_skipped;
		return;
	}
	// The record buffer's key (its guest address); a new one takes the least recently copied slot.
	int k = -1;
	for (uint32_t i = 0; i < NKeys; ++i)
		if (g_pn.keys[i].address == d.rec_address) k = static_cast<int>(i);
	if (k < 0) {
		uint64_t oldest = UINT64_MAX;
		for (uint32_t i = 0; i < NKeys; ++i) {
			const uint64_t c = g_pn.keys[i].address == 0 ? 0 : g_pn.keys[i].copied_serial + 1;
			if (c < oldest) oldest = c, k = static_cast<int>(i);
		}
		NKey& key = g_pn.keys[k];
		FreeLater(key.buf[0]), FreeLater(key.buf[1]);
		key         = NKey {};
		key.address = d.rec_address;
	}
	NKey&      key   = g_pn.keys[k];
	const bool first = key.copied_serial != g_pn.serial;
	if (first) {
		key.cur ^= 1;
		key.prev_serial = key.copied_serial, key.prev_stride = key.stride, key.prev_count = key.count;
		if (key.buf[key.cur].size < d.rec.size) {
			FreeLater(key.buf[key.cur]);
			if (!MakeBuffer(graphics, AlignUp(d.rec.size + d.rec.size / 4, 1ull << 20), false, key.buf[key.cur])) {
				key.copied_serial = UINT64_MAX, key.prev_serial = UINT64_MAX;
				++g_pn.draws_skipped;
				return;
			}
		}
	} else if (key.stride != d.rec_stride || key.count != d.rec_count) {
		++g_pn.draws_skipped; // (another layout on the same buffer this frame)
		return;
	}
	// Only a pass already open with this draw's state is resumed (attachments loaded, never cleared again: ResumeRendering);
	// otherwise the draw's own BeginRendering opens it afterwards with its real load / clear operations.
	const bool resume = command.RenderingWith(rendering);
	command.EndRendering();
	const auto vk_command = command.Handle();
	Barrier(vk_command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
	        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite);
	if (first) {
		const vk::BufferCopy region {d.rec.offset, 0, d.rec.size};
		vk_command.copyBuffer(d.rec.buffer, vk::Buffer(key.buf[key.cur].buffer), 1, &region);
		key.copied_serial = g_pn.serial, key.stride = d.rec_stride, key.count = d.rec_count;
	}
	const vk::Buffer ring(g_pn.ring.buffer);
	{
		const vk::BufferCopy region {d.sort.offset, sort_off, sort_size};
		vk_command.copyBuffer(d.sort.buffer, ring, 1, &region);
	}
	if (emit_size > 0) {
		const vk::BufferCopy region {d.emit.offset, emit_off, emit_size};
		vk_command.copyBuffer(d.emit.buffer, ring, 1, &region);
	}
	if (args_size > 0) {
		const vk::BufferCopy region {d.args.offset, args_off, args_size};
		vk_command.copyBuffer(d.args.buffer, ring, 1, &region);
	}
	Barrier(vk_command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eTransferRead);
	if (resume) command.ResumeRendering(rendering);
	g_pn.ring_used = args_off + args_size;
	NDraw nd;
	nd.key = k, nd.sort_off = sort_off, nd.emit_off = emit_off, nd.args_off = args_off, nd.sort_size = sort_size, nd.emit_size = emit_size;
	nd.sort_stride = d.sort_stride, nd.sort_count = d.sort_count, nd.emit_stride = d.emit_stride;
	nd.emit_count  = emit_size > 0 && d.emit_stride > 0 ? static_cast<uint32_t>(emit_size / d.emit_stride) : 0;
	nd.base = d.base_instance, nd.instances = d.instance_count, nd.indirect = d.indirect;
	nd.first_word = d.indirect && d.base_from_args ? d.args_first_instance_word + 1 : 0;
	nd.prev_valid = key.prev_serial != UINT64_MAX && key.prev_serial + 1 == g_pn.serial && key.prev_stride == d.rec_stride &&
	                key.vp_prev_valid && key.buf[key.cur ^ 1].buffer != VK_NULL_HANDLE;
	std::memcpy(nd.vp, d.vp, sizeof(nd.vp));
	std::memcpy(nd.vpp, key.vp_prev, sizeof(nd.vpp));
	std::memcpy(key.vp_last, d.vp, sizeof(key.vp_last));
	g_pn.draws.push_back(nd);
	static bool said = false;
	if (!said) {
		said = true;
		std::printf("FG particle native: first draw VS %016llx: records 0x%llx (%u x %u B), sorted list %u x %u B, emitters %u x %u B, %s\n",
		            static_cast<unsigned long long>(d.vs), static_cast<unsigned long long>(d.rec_address), d.rec_count, d.rec_stride,
		            d.sort_count, d.sort_stride, d.emit_count, d.emit_stride, d.indirect ? "indirect" : "direct");
		std::fflush(stdout);
	}
}

bool ParticleMvMeasure(vk::CommandBuffer command, const UpscaleInputs& inputs, const UpscaleImage& game) {
	g_pm.measured = false;
	if (!ParticleMvWanted()) return false;
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr) return false;
	// (des-particles-native) Whatever happens below, this frame's captured draws end with it.
	struct NativeFrameEnd {
		GraphicContext& graphics;
		~NativeFrameEnd() {
			if (ParticleNativeWanted()) NativeEndFrame(graphics);
		}
	} native_end {*graphics};
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
	if (ParticleNativeWanted()) {
		// (des-particles-native) The particle pass's own positions: lay[cur] (coverage) read, pmv written, then counted.
		infos[2] = {g_pm.point, g_pm.lay[cur].view, gen};
		infos[6] = {nullptr, g_pm.pmv.view, gen};
		infos[5] = {g_pm.point, g_pm.lay[prev].view, gen};
		if (NativeRun(*graphics, command, push, infos)) {
			infos[5] = {g_pm.point, g_pm.pmv.view, gen};
			infos[6] = {nullptr, g_pm.lay[prev].view, gen};
			Dispatch(command, push, 4, infos, g_pm.half);
			g_pm.measured = true;
		}
	} else if (g_pm.history_valid) {
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

bool ParticleMvCompose(vk::CommandBuffer command, GraphicContext& graphics, VulkanImage& motion, vk::ImageView view) {
	(void)graphics;
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
	// (11-11) The FG copy's own view (made with the image, same RG16F format; its usage includes storage): a view cached
	// by image handle outlived the image when Configure remade it (a handle value can be reused): device lost.
	if (view == nullptr) return false;
	const auto layout = motion.state.layout;
	const auto gen    = vk::ImageLayout::eGeneral;
	Layout(command, motion.image, layout, gen);
	Push push = BasePush();
	const vk::DescriptorImageInfo infos[8] {
	    {g_pm.point, g_pm.pmv.view, gen},  {g_pm.point, g_pm.pmv.view, gen}, {g_pm.point, g_pm.lay[0].view, gen},
	    {g_pm.point, g_pm.lay[1].view, gen}, {g_pm.point, g_pm.lay[0].view, gen}, {g_pm.linear, g_pm.pmv.view, gen},
	    {nullptr, g_pm.lay[0].view, gen},  {nullptr, view, gen},
	};
	Dispatch(command, push, 2, infos, g_pm.full);
	Layout(command, motion.image, gen, layout);
	++g_pm.composed;
	return true;
}

} // namespace Libs::Graphics::FrameGen
