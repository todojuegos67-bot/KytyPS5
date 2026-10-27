// KYTY_HWRT_PARTICLES=0|1|2|3 ("Particulas RT (HW)"; hwrt-particles.comp): the game's emissive particles - fire, embers,
// sparks, spell glow - light the world, with hardware ray-query shadows into the scene TLAS that KYTY_HWRT=1 builds from
// the G-buffer draws every frame (local/hwrt-scene.h CurrentRayScene, the same RayInfo the reflections read).
//
// Where the emitters come from: KYTY_RT_PARTICLES' hook (frame-gen.cpp BeforeSceneDraw) copies the scene color right
// before the frame's first blended, depth-write-off draw into it (the engine's particle-translucent pass); it is taken
// while this switch is on too. Scene color - copy = the particles' own emission (exact for additive particles). The pass
// runs first in FrameGen::Upscale, on the game's own color, so nothing else is mixed into that difference.
//
// Per frame (compute, one pipeline, push descriptors, six kinds of dispatch):
//   collect   a tile grid (16x9 / 24x14 / 32x18 at levels 1 / 2 / 3): one emitter per tile (power-weighted world position,
//             intensity = emission x the pixels' world area, peak radiance)
//   cdf       one thread: the non-empty emitters' power CDF; the total power smoothed so a burst ramps in (no flash)
//   trace     half res (1, 2) / full res (3): 4 / 6 / 8 candidates by power, RIS by power x cos / d^2, 1 / 1 / 2 TLAS
//             shadow rays per pixel (terminate on first hit), the irradiance
//   temporal  motion-reprojected history (alpha 0.12 / 0.1 / 0.08, distance-tested, clamped to the raw neighbourhood)
//   a-trous   depth-aware 5x5, 2 / 2 / 3 passes
//   composite render size: + albedo / pi x irradiance x strength, soft-capped by the emitters' own radiance
// KYTY_HWRT_PARTICLES_STRENGTH (1.0, 0..4), KYTY_HWRT_PARTICLES_RANGE (m, 8), KYTY_HWRT_PARTICLES_VIEW=1 (the added
// light only, x4). Every 10 s "HWRT particles: stats" (GPU ms, emitters a frame, frames with emitters). The SSGI
// emissive light leaves the particles to this pass while it runs (no double light).
#include "local/local-pass-cache.h"
#include "local/hwrt-particles.h"

#include "local/rt-reflections.h"
#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/hwrt_particles_spv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::FrameGen {
namespace {

constexpr uint32_t Bindings  = 12;
constexpr uint32_t MaxLights = 576;
constexpr uint32_t LightBytes = 32 + MaxLights * 32 + MaxLights * 4 + MaxLights * 4;

struct HpPush {
	float   size[4];
	float   params[4];
	int32_t misc[4];
	int32_t misc2[4];
	float   extra[4];
};
static_assert(sizeof(HpPush) == 80);

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
	Img                     out, raw, tmp, dummy;
	std::array<Img, 2>      hist;
	vk::Extent2D            extent {}, textent {};
	int                     tlevel = 0;
	bool                    ready = false, history_valid = false, dummy_ready = false;
	VkBuffer                light_buf = VK_NULL_HANDLE, read_buf = VK_NULL_HANDLE;
	VmaAllocation           light_mem = nullptr, read_mem = nullptr;
	const float*            mapped = nullptr;
	bool                    cleared = false;
	uint64_t                frame = 0, flip_ran = UINT64_MAX;
	std::array<std::pair<Img, uint64_t>, 8> retired {};
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms = 0.0, max_ms = 0.0;
	uint32_t                samples = 0;
	uint64_t                ran = 0, no_pre = 0, no_tlas = 0;
	double                  emitters_sum = 0.0;
	uint32_t                emitters_n = 0;
	float                   last_frames_with = 0.0f;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now(), last_frame {};
} g_hp;

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
	for (auto& slot: g_hp.retired)
		if (slot.first.image == VK_NULL_HANDLE) {
			slot = {img, g_hp.frame};
			img  = {};
			return;
		}
	Destroy(graphics, img);
}
void FreeRetired(GraphicContext& graphics) {
	for (auto& slot: g_hp.retired)
		if (slot.first.image != VK_NULL_HANDLE && g_hp.frame >= slot.second + 16) Destroy(graphics, slot.first), slot.second = 0;
}

vk::DescriptorType DescType(uint32_t i) {
	if (i == 11) return vk::DescriptorType::eAccelerationStructureKHR;
	if (i == 9 || i == 10) return vk::DescriptorType::eStorageBuffer;
	if (i == 6 || i == 8) return vk::DescriptorType::eStorageImage;
	return vk::DescriptorType::eCombinedImageSampler;
}

bool EnsurePipeline(GraphicContext& graphics) {
	std::lock_guard ensure_lock(LocalPassCache::EnsureMutex()); // (the prewarm thread's, local-pass-cache.h)
	if (g_hp.pipeline != nullptr) return true;
	if (g_hp.failed) return false;
	g_hp.failed       = true;
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
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_hp.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(HpPush)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_hp.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_hp.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(HWRT_PARTICLES_SPV);
	module_info.pCode       = HWRT_PARTICLES_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_hp.layout;
	const auto created         = device.createComputePipelines(LocalPassCache::Get(device), 1, &pipeline_info, nullptr, &g_hp.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_hp.pipeline = nullptr;
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
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g_hp.point : &g_hp.linear) != vk::Result::eSuccess) return false;
	}
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = LightBytes;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VmaAllocationCreateInfo dev {};
	dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_hp.light_buf, &g_hp.light_mem, nullptr) != VK_SUCCESS) {
		g_hp.light_buf = VK_NULL_HANDLE;
		return false;
	}
	buffer.size  = 32;
	buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_hp.read_buf, &g_hp.read_mem, &mapped) != VK_SUCCESS) {
		g_hp.read_buf = VK_NULL_HANDLE;
		return false;
	}
	g_hp.mapped = static_cast<const float*>(mapped.pMappedData);
	if (!MakeImage(graphics, vk::Format::eR32Uint, vk::Extent2D {1, 1}, g_hp.dummy)) return false;
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = 8 * 2;
	if (device.createQueryPool(&query, nullptr, &g_hp.pool) != vk::Result::eSuccess) g_hp.pool = nullptr;
	g_hp.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g_hp.failed    = false;
	return true;
}

bool EnsureImages(GraphicContext& graphics, vk::Extent2D extent, int level) {
	const int tl = level >= 3 ? 1 : 2; // (the trace's divisor)
	if (g_hp.out.image != VK_NULL_HANDLE && g_hp.extent == extent && g_hp.tlevel == tl) return true;
	Retire(graphics, g_hp.out), Retire(graphics, g_hp.raw), Retire(graphics, g_hp.tmp);
	Retire(graphics, g_hp.hist[0]), Retire(graphics, g_hp.hist[1]);
	g_hp.ready = false, g_hp.history_valid = false;
	g_hp.extent  = extent;
	g_hp.tlevel  = tl;
	g_hp.textent = vk::Extent2D {(extent.width + tl - 1) / tl, (extent.height + tl - 1) / tl};
	return MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, extent, g_hp.out) &&
	       MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, g_hp.textent, g_hp.raw) &&
	       MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, g_hp.textent, g_hp.tmp) &&
	       MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, g_hp.textent, g_hp.hist[0]) &&
	       MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, g_hp.textent, g_hp.hist[1]);
}

void ReadTiming(GraphicContext& graphics) {
	if (g_hp.pool == nullptr) return;
	while (g_hp.read + 4 <= g_hp.written) {
		const auto              entry = static_cast<uint32_t>(g_hp.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (graphics.device.getQueryPoolResults(g_hp.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		if (stamps[1] > stamps[0]) {
			const double ms = static_cast<double>(stamps[1] - stamps[0]) * g_hp.period_ns / 1e6;
			g_hp.sum_ms += ms, g_hp.max_ms = std::max(g_hp.max_ms, ms);
			++g_hp.samples;
		}
		++g_hp.read;
	}
	if (g_hp.mapped != nullptr) { // (a few frames late: statistics only)
		const float n = g_hp.mapped[0];
		if (std::isfinite(n)) g_hp.emitters_sum += n, ++g_hp.emitters_n;
	}
}

void Log(int level) {
	const auto now = std::chrono::steady_clock::now();
	if (now - g_hp.last_log < std::chrono::seconds(10)) return;
	g_hp.last_log = now;
	const float with = g_hp.mapped != nullptr ? g_hp.mapped[4] : 0.0f;
	std::printf("HWRT particles: stats (level %d) GPU %.3f ms (max %.3f), %.1f emitters a frame, %.0f frames with emitters in "
	            "the last 10 s, smoothing scale %.2f; %llu frames traced, %llu without the pre-particle copy, %llu without a TLAS\n",
	            level, g_hp.samples > 0 ? g_hp.sum_ms / g_hp.samples : 0.0, g_hp.max_ms,
	            g_hp.emitters_n > 0 ? g_hp.emitters_sum / g_hp.emitters_n : 0.0, std::max(0.0f, with - g_hp.last_frames_with),
	            g_hp.mapped != nullptr ? g_hp.mapped[5] : 0.0f, static_cast<unsigned long long>(g_hp.ran),
	            static_cast<unsigned long long>(g_hp.no_pre), static_cast<unsigned long long>(g_hp.no_tlas));
	std::fflush(stdout);
	g_hp.last_frames_with = with;
	g_hp.sum_ms = 0.0, g_hp.max_ms = 0.0, g_hp.samples = 0, g_hp.emitters_sum = 0.0, g_hp.emitters_n = 0;
}

void SayOnce(const char* text) {
	static bool said = false;
	if (said) return;
	said = true;
	std::printf("HWRT particles: %s\n", text);
	std::fflush(stdout);
}

} // namespace

int HwParticlesLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_HWRT_PARTICLES");
		return value != nullptr && *value != '\0' ? std::clamp(std::atoi(value), 0, 3) : 0;
	}();
	return level;
}
bool HwParticlesWanted() { return HwParticlesLevel() > 0; }
void HwParticlesPrewarm(GraphicContext& graphics) { // (local-pass-cache.h)
	if (HwParticlesLevel() > 0) (void)EnsurePipeline(graphics);
}
bool HwParticlesActive() { return HwParticlesWanted() && g_hp.flip_ran == g_hp.frame; }

bool HwParticlesApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int level = HwParticlesLevel();
	if (level == 0) return false;
	++g_hp.frame; // (one Upscale per flip; HwParticlesActive compares to it)
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr) return false;
	if (VideoOut::MoviePlaying()) {
		g_hp.history_valid = false;
		return false;
	}
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || color.extent.width < 16 || color.extent.height < 16 ||
	    inputs.depth.extent != color.extent)
		return false;
	if (inputs.hwrt_tlas == 0 || !inputs.hwrt_info) {
		++g_hp.no_tlas;
		SayOnce("no scene TLAS yet (needs KYTY_HWRT=1 and a device with ray queries); nothing until there is one");
		g_hp.history_valid = false;
		return false;
	}
	const vk::ImageView pre = RtParticlesPreCopy(inputs.color_address, color.extent);
	if (pre == nullptr) {
		++g_hp.no_pre;
		g_hp.history_valid = false; // (no particles drawn this flip: the history restarts when they return)
		Log(level);
		return false;
	}
	if (!EnsurePipeline(*graphics) || !EnsureImages(*graphics, color.extent, level)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("HWRT particles: off (could not create its pipeline or images)\n");
			std::fflush(stdout);
		}
		return false;
	}
	FreeRetired(*graphics);
	ReadTiming(*graphics);
	Log(level);

	const auto now = std::chrono::steady_clock::now();
	if (g_hp.last_frame.time_since_epoch().count() != 0 && now - g_hp.last_frame > std::chrono::milliseconds(250))
		g_hp.history_valid = false;
	g_hp.last_frame = now;

	const float strength = EnvFloat("KYTY_HWRT_PARTICLES_STRENGTH", 1.0f, 0.0f, 4.0f);
	const float range    = EnvFloat("KYTY_HWRT_PARTICLES_RANGE", 8.0f, 1.0f, 40.0f);
	static const bool view = [] {
		const char* v = std::getenv("KYTY_HWRT_PARTICLES_VIEW");
		return v != nullptr && *v == '1';
	}();
	RtReflGbuffer gbuf;
	const bool    have_albedo = RtReflGetGbuffer(gbuf) && gbuf.extent == color.extent && gbuf.views[0] != nullptr;

	const uint32_t tiles_x = level >= 3 ? 32 : level == 2 ? 24 : 16;
	const uint32_t tiles_y = level >= 3 ? 18 : level == 2 ? 14 : 9;
	const bool     history = g_hp.history_valid && !inputs.reset;

	HpPush push {};
	push.size[0] = static_cast<float>(color.extent.width), push.size[1] = static_cast<float>(color.extent.height);
	push.size[2] = static_cast<float>(g_hp.textent.width), push.size[3] = static_cast<float>(g_hp.textent.height);
	// (2: the internal gain over the physical I x cos / d^2: the game's flames are far dimmer than real fire)
	push.params[0] = strength * 2.0f, push.params[1] = range, push.params[2] = 0.5f;
	push.params[3] = level >= 3 ? 0.08f : level == 2 ? 0.1f : 0.12f;
	push.misc[1]  = static_cast<int32_t>(g_hp.frame & 0x7fffffff);
	push.misc[2]  = level >= 3 ? 8 : level == 2 ? 6 : 4;
	push.misc[3]  = level >= 3 ? 2 : 1;
	push.misc2[0] = static_cast<int32_t>(tiles_x), push.misc2[1] = static_cast<int32_t>(tiles_y);
	push.misc2[2] = level >= 3 ? 2 : 3;
	push.misc2[3] = (have_albedo ? 1 : 0) | (history ? 2 : 0) | (view ? 4 : 0) | (inputs.motion.view != nullptr ? 8 : 0);
	push.extra[0] = inputs.motion_scale_x, push.extra[1] = inputs.motion_scale_y;
	push.extra[3] = 0.02f; // (the emission's luminance below which a pixel is no emitter: smoke, noise)
	{
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("HWRT particles: on (level %d, %ux%u, trace %ux%u, %ux%u emitter tiles, %d candidates, %d shadow ray(s) a "
			            "pixel, strength %g, range %g m, albedo %s)\n",
			            level, color.extent.width, color.extent.height, g_hp.textent.width, g_hp.textent.height, tiles_x, tiles_y,
			            push.misc[2], push.misc[3], strength, range, have_albedo ? "the G-buffer's" : "estimated from the color");
			std::fflush(stdout);
		}
	}

	const auto entry = static_cast<uint32_t>(g_hp.written % 8);
	const bool timed = g_hp.pool != nullptr && g_hp.written < g_hp.read + 8;
	if (timed) {
		command.resetQueryPool(g_hp.pool, entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_hp.pool, entry * 2);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite);
	if (!g_hp.ready) {
		ToGeneral(command, g_hp.out.image), ToGeneral(command, g_hp.raw.image), ToGeneral(command, g_hp.tmp.image);
		ToGeneral(command, g_hp.hist[0].image), ToGeneral(command, g_hp.hist[1].image);
		g_hp.ready = true;
	}
	if (!g_hp.dummy_ready) {
		ToGeneral(command, g_hp.dummy.image);
		const vk::ClearColorValue zero {};
		vk::ImageSubresourceRange range_all {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		command.clearColorImage(vk::Image(g_hp.dummy.image), vk::ImageLayout::eGeneral, &zero, 1, &range_all);
		g_hp.dummy_ready = true;
	}
	if (!g_hp.cleared) {
		g_hp.cleared = true;
		command.fillBuffer(vk::Buffer(g_hp.light_buf), 0, VK_WHOLE_SIZE, 0);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_hp.pipeline);

	const auto     gen = vk::ImageLayout::eGeneral;
	const auto     ro  = vk::ImageLayout::eShaderReadOnlyOptimal;
	const uint32_t cur = static_cast<uint32_t>(g_hp.frame & 1u);
	vk::DescriptorImageInfo infos[Bindings] {};
	infos[0] = {g_hp.point, color.view, color.layout};
	infos[1] = {g_hp.point, inputs.depth.view, inputs.depth.layout};
	infos[2] = inputs.motion.view != nullptr ? vk::DescriptorImageInfo {g_hp.point, inputs.motion.view, inputs.motion.layout}
	                                         : vk::DescriptorImageInfo {g_hp.point, g_hp.tmp.view, gen};
	infos[3] = {g_hp.point, pre, ro};
	infos[4] = have_albedo ? vk::DescriptorImageInfo {g_hp.point, gbuf.views[0], ro} : vk::DescriptorImageInfo {g_hp.point, g_hp.dummy.view, gen};
	infos[5] = {g_hp.linear, g_hp.hist[cur ^ 1u].view, gen};
	infos[6] = {nullptr, g_hp.raw.view, gen};
	infos[7] = {g_hp.point, g_hp.tmp.view, gen};
	infos[8] = {nullptr, g_hp.out.view, gen};
	const vk::DescriptorBufferInfo light_info {vk::Buffer(g_hp.light_buf), 0, VK_WHOLE_SIZE};
	const vk::DescriptorBufferInfo ray_info {inputs.hwrt_info, 0, inputs.hwrt_info_size};
	const auto tlas = vk::AccelerationStructureKHR(reinterpret_cast<VkAccelerationStructureKHR>(inputs.hwrt_tlas));
	vk::WriteDescriptorSetAccelerationStructureKHR as_write {};
	as_write.accelerationStructureCount = 1;
	as_write.pAccelerationStructures    = &tlas;
	vk::WriteDescriptorSet writes[Bindings] {};
	const auto push_set = [&] {
		for (uint32_t i = 0; i < Bindings; ++i) {
			writes[i]                 = vk::WriteDescriptorSet {};
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = DescType(i);
			if (i == 9) writes[i].pBufferInfo = &light_info;
			else if (i == 10) writes[i].pBufferInfo = &ray_info;
			else if (i == 11) writes[i].pNext = &as_write;
			else writes[i].pImageInfo = &infos[i];
		}
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_hp.layout, 0, Bindings, writes);
	};
	const auto run = [&](int mode, uint32_t gx, uint32_t gy) {
		push.misc[0] = mode;
		command.pushConstants(g_hp.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch(gx, gy, 1);
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	};
	const uint32_t tx = (g_hp.textent.width + 7) / 8, ty = (g_hp.textent.height + 7) / 8;
	push_set();
	run(0, tiles_x, tiles_y); // collect
	run(1, 1, 1);             // cdf
	run(2, tx, ty);           // trace -> raw
	// temporal: raw (7) + last history (5) -> this history (6)
	infos[6] = {nullptr, g_hp.hist[cur].view, gen};
	infos[7] = {g_hp.point, g_hp.raw.view, gen};
	push_set();
	run(3, tx, ty);
	// a-trous: hist[cur] -> tmp -> raw (-> tmp at level 3); the history itself stays unfiltered
	const int passes = level >= 3 ? 3 : 2;
	Img*      src    = &g_hp.hist[cur];
	for (int p = 0; p < passes; ++p) {
		Img* dst = (p & 1) == 0 ? &g_hp.tmp : &g_hp.raw;
		infos[6] = {nullptr, dst->view, gen};
		infos[7] = {g_hp.point, src->view, gen};
		push.extra[2] = static_cast<float>(1 << p);
		push_set();
		run(4, tx, ty);
		src = dst;
	}
	// composite: the filtered light (7, linear) onto the color (8)
	infos[6] = {nullptr, g_hp.tmp.view == src->view ? g_hp.raw.view : g_hp.tmp.view, gen}; // (not written)
	infos[7] = {g_hp.linear, src->view, gen};
	push_set();
	run(5, (color.extent.width + 7) / 8, (color.extent.height + 7) / 8);
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eMemoryRead);
	{
		vk::BufferCopy region {0, 0, 32};
		command.copyBuffer(vk::Buffer(g_hp.light_buf), vk::Buffer(g_hp.read_buf), 1, &region);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
		        vk::AccessFlagBits2::eHostRead);
	}
	if (timed) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_hp.pool, entry * 2 + 1);
		++g_hp.written;
	}
	++g_hp.ran;
	g_hp.flip_ran       = g_hp.frame;
	g_hp.history_valid  = true;
	inputs.color.image  = vk::Image(g_hp.out.image);
	inputs.color.view   = g_hp.out.view;
	inputs.color.layout = gen;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc |
	                      vk::ImageUsageFlagBits::eStorage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}

} // namespace Libs::Graphics::FrameGen
