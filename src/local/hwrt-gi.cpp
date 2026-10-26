// KYTY_HWRT_AO=0..3 / KYTY_HWRT_GI=0..3 (hwrt2g): ray-traced ambient occlusion and one-bounce diffuse GI through the
// scene TLAS that KYTY_HWRT=1 builds every frame at the PS5 Pro super-resolution packet (hwrt-scene.cpp), recorded in
// FrameGen::Upscale in GTAO's slot (after it), on the render-resolution color before the upscaler.
//
//  AO: short cosine rays (KYTY_HWRT_AO_RADIUS, 0.8 m) from the depth's world position: true contact occlusion from the
//      real geometry (no screen-space halos around silhouettes, no veil over far walls: the radius is metres at every
//      distance, no screen clamp), the player and enemies included (captured every frame). Replaces GTAO (GtaoApply
//      does nothing) unless KYTY_HWRT_AO_GTAO=1 keeps both. KYTY_HWRT_AO_STRENGTH (0.6), KYTY_HWRT_AO_FLOOR (0.6: at most
//      40 % off a pixel).
//  GI: cosine rays to KYTY_HWRT_GI_RANGE (8 m), closest hit; the hit's radiance is the game's own lit colour where the
//      hit is on screen and not hidden (exact for what the camera sees), a dim estimate off screen; misses add nothing
//      (the sky's light is the game's ambient already). Added as G-buffer albedo x bounce x KYTY_HWRT_GI_INTENSITY (0.7),
//      soft-capped against the pixel's own light: dark creases next to lit walls lift and take their colour; nothing glows.
//  Levels (the higher of the two switches): 1 half res, AO 2 rays + GI 1 ray, 2 a-trous passes; 2 half res, AO 3 + GI 2,
//  3 passes; 3 full res, AO 2 + GI 2, 3 passes. Temporal accumulation (motion vectors, disocclusion, clamped) + a-trous +
//  a depth-aware upsample. KYTY_HWRT_GI_VIEW=1 the AO alone, 2 the added bounce x 4. Every 5 s "HWRTGI: GPU" timing.
//  Off (both 0 / unset): nothing here runs (HwGiWanted false; GTAO, SSGI and the G-buffer copy exactly as before).
#include "local/hwrt-gi.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/hwrt_gi_spv.h"
#include "local/hwrt-scene.h"
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

int EnvLevel(const char* name) {
	const char* v = std::getenv(name);
	return v != nullptr && *v != '\0' ? std::clamp(std::atoi(v), 0, 3) : 0;
}
float EnvFloat(const char* name, float fallback, float lo, float hi) {
	const char* v = std::getenv(name);
	if (v == nullptr || *v == '\0') return fallback;
	const float f = static_cast<float>(std::atof(v));
	return std::isfinite(f) ? std::clamp(f, lo, hi) : fallback;
}
bool EnvBool(const char* name, bool fallback) {
	const char* v = std::getenv(name);
	if (v == nullptr || *v == '\0') return fallback;
	return *v != '0';
}

struct Config {
	int ao = 0, gi = 0, level = 0;
};
const Config& Cfg() {
	static const Config c = [] {
		Config r;
		r.ao = EnvLevel("KYTY_HWRT_AO");
		r.gi = EnvLevel("KYTY_HWRT_GI");
		if (r.ao > 0 || r.gi > 0) {
			std::printf("HWRTGI: KYTY_HWRT_AO=%d KYTY_HWRT_GI=%d read%s\n", r.ao, r.gi,
			            HwScene::Requested() ? "" : ": needs KYTY_HWRT=1 too: off");
			std::fflush(stdout);
			if (!HwScene::Requested()) r.ao = r.gi = 0;
		}
		r.level = std::max(r.ao, r.gi);
		return r;
	}();
	return c;
}

struct Image {
	VkImage       image  = VK_NULL_HANDLE;
	VmaAllocation memory = VK_NULL_HANDLE;
	vk::ImageView view   = nullptr;
	bool          general = false;
};

// std140 (hwrt-gi.comp Frame)
struct FrameUbo {
	float   inv[16];
	float   vp[16];
	float   cam[4];
	float   motion[4];
	float   trace[4];
	float   par[4];
	int32_t cfg[4];
	int32_t cfg2[4];
};
static_assert(sizeof(FrameUbo) == 224);
struct Push {
	int32_t mode = 0;
	float   step = 0.0f;
	int32_t pad[2] {};
};

constexpr uint32_t Bindings = 15;

struct State {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             point = nullptr, linear = nullptr;
	bool                    failed = false;
	Image                   dummy;
	bool                    dummy_ready = false;
	VkBuffer                ubo = VK_NULL_HANDLE;
	VmaAllocation           ubo_mem = VK_NULL_HANDLE;
	vk::Extent2D            render {}, trace {};
	Image                   raw, tmp_a, tmp_b, out, history[2], gbuf[2], count[2];
	uint32_t                frame = 0;
	bool                    history_valid = false;
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms = 0.0;
	uint32_t                samples = 0, frames_on = 0, frames_no_scene = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
} g_gi;

std::array<Image*, 10> TraceImages() {
	return {&g_gi.raw, &g_gi.tmp_a, &g_gi.tmp_b, &g_gi.history[0], &g_gi.history[1], &g_gi.gbuf[0], &g_gi.gbuf[1], &g_gi.count[0],
	        &g_gi.count[1], &g_gi.out};
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
	if (i == 13) return vk::DescriptorType::eAccelerationStructureKHR;
	if (i == 14) return vk::DescriptorType::eUniformBuffer;
	if (i == 7 || i == 8 || i == 10) return vk::DescriptorType::eStorageImage;
	return vk::DescriptorType::eCombinedImageSampler;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_gi.pipeline != nullptr) return true;
	if (g_gi.failed) return false;
	g_gi.failed       = true;
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
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_gi.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Push)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_gi.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_gi.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = sizeof(HWRT_GI_SPV);
	module_info.pCode    = HWRT_GI_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_gi.layout;
	const auto created = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_gi.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_gi.pipeline = nullptr;
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
		sampler.maxLod       = 0.0f;
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g_gi.point : &g_gi.linear) != vk::Result::eSuccess) return false;
	}
	if (!MakeImage(graphics, vk::Format::eR32Uint, {1, 1}, false, g_gi.dummy)) return false;
	{
		VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		buffer.size  = sizeof(FrameUbo);
		buffer.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo dev {};
		dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_gi.ubo, &g_gi.ubo_mem, nullptr) != VK_SUCCESS) {
			g_gi.ubo = VK_NULL_HANDLE;
			return false;
		}
	}
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = 16;
	if (device.createQueryPool(&query, nullptr, &g_gi.pool) != vk::Result::eSuccess) g_gi.pool = nullptr;
	g_gi.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g_gi.failed    = false;
	return true;
}

bool EnsureImages(GraphicContext& graphics, vk::Extent2D render, int level) {
	if (g_gi.out.image != VK_NULL_HANDLE && g_gi.render == render) return true;
	const vk::Extent2D trace = level >= 3 ? render : vk::Extent2D {std::max(1u, (render.width + 1) / 2), std::max(1u, (render.height + 1) / 2)};
	for (auto* img : TraceImages())
		FreeImage(graphics, *img);
	bool ok = MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, render, true, g_gi.out);
	for (auto* img : TraceImages())
		if (img != &g_gi.out) ok = ok && MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, trace, true, *img);
	if (!ok) {
		for (auto* img : TraceImages())
			FreeImage(graphics, *img);
		g_gi.failed = true;
		return false;
	}
	g_gi.render        = render;
	g_gi.trace         = trace;
	g_gi.history_valid = false;
	return true;
}

void ReadTiming(GraphicContext& graphics) {
	while (g_gi.pool != nullptr && g_gi.read + 4 <= g_gi.written) {
		const auto entry = static_cast<uint32_t>(g_gi.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (graphics.device.getQueryPoolResults(g_gi.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		g_gi.sum_ms += static_cast<double>(stamps[1] - stamps[0]) * g_gi.period_ns / 1e6;
		++g_gi.samples;
		++g_gi.read;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - g_gi.last_log >= std::chrono::seconds(5) && (g_gi.samples > 0 || g_gi.frames_no_scene > 0)) {
		std::printf("HWRTGI: GPU %.3f ms (mean of %u frames; %u frames on, %u without a scene)\n",
		            g_gi.samples > 0 ? g_gi.sum_ms / g_gi.samples : 0.0, g_gi.samples, g_gi.frames_on, g_gi.frames_no_scene);
		std::fflush(stdout);
		g_gi.sum_ms   = 0.0;
		g_gi.samples  = 0;
		g_gi.frames_on = g_gi.frames_no_scene = 0;
		g_gi.last_log = now;
	}
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

} // namespace

bool HwGiWanted() { return Cfg().level > 0; }
bool HwAoReplacesGtao() {
	static const bool keep = EnvBool("KYTY_HWRT_AO_GTAO", false);
	return Cfg().ao > 0 && !keep && HwScene::Active();
}

bool HwGiApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const auto& cfg = Cfg();
	if (cfg.level == 0) return false; // off: nothing created or recorded
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr || !HwScene::Active()) return false;
	if (VideoOut::MoviePlaying()) {
		g_gi.history_valid = false;
		return false;
	}
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || inputs.motion.view == nullptr || color.extent.width < 16 ||
	    color.extent.height < 16 || inputs.depth.extent != color.extent)
		return false;
	HwScene::GiSceneInfo scene;
	if (!HwScene::GiScene(scene)) {
		++g_gi.frames_no_scene;
		g_gi.history_valid = false;
		ReadTiming(*graphics);
		return false;
	}
	if (!EnsurePipeline(*graphics) || !EnsureImages(*graphics, color.extent, cfg.level)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("HWRTGI: off (could not create its pipeline or images)\n");
			std::fflush(stdout);
		}
		return false;
	}
	ReadTiming(*graphics);
	++g_gi.frames_on;

	static const float ao_radius   = EnvFloat("KYTY_HWRT_AO_RADIUS", 0.8f, 0.05f, 8.0f);
	static const float ao_strength = EnvFloat("KYTY_HWRT_AO_STRENGTH", 0.6f, 0.0f, 1.0f);
	static const float ao_floor    = EnvFloat("KYTY_HWRT_AO_FLOOR", 0.6f, 0.2f, 1.0f);
	static const float gi_int      = EnvFloat("KYTY_HWRT_GI_INTENSITY", 0.7f, 0.0f, 4.0f);
	static const float gi_range    = EnvFloat("KYTY_HWRT_GI_RANGE", 8.0f, 0.5f, 100.0f);
	static const float fade_start  = EnvFloat("KYTY_HWRT_GI_FADE", 120.0f, 5.0f, 5000.0f);
	static const int   view        = std::clamp(static_cast<int>(EnvFloat("KYTY_HWRT_GI_VIEW", 0.0f, 0.0f, 2.0f)), 0, 2);
	static const bool  flip_y      = EnvBool("KYTY_HWRT_GI_FLIPY", false);
	static const bool  use_normal  = EnvBool("KYTY_HWRT_GI_GBUF_NORMAL", true);
	static const bool  use_albedo  = EnvBool("KYTY_HWRT_GI_GBUF_ALBEDO", true);
	const int          level       = cfg.level;
	const int          ao_rays     = cfg.ao == 0 ? 0 : level == 1 ? 2 : level == 2 ? 3 : 2;
	const int          gi_rays     = cfg.gi == 0 ? 0 : level == 1 ? 1 : 2;
	const int          passes      = level == 1 ? 2 : 3;
	const bool         half        = g_gi.trace != g_gi.render;

	RtReflGbuffer gbuf;
	const bool    have_gbuf = RtReflGetGbuffer(gbuf) && gbuf.extent == color.extent;
	const bool    normals   = use_normal && have_gbuf && gbuf.views[3] != nullptr;
	const bool    albedo    = use_albedo && cfg.gi > 0 && have_gbuf && gbuf.views[0] != nullptr;
	{
		static int said = -1;
		const int  now  = (normals ? 1 : 0) | (albedo ? 2 : 0);
		if (now != said) {
			said = now;
			std::printf("HWRTGI: on (AO level %d: %d rays, %g m, strength %g, floor %g; GI level %d: %d rays, %g m, intensity %g; "
			            "%ux%u trace for %ux%u; normals from %s, albedo from %s%s%s)\n",
			            cfg.ao, ao_rays, ao_radius, ao_strength, ao_floor, cfg.gi, gi_rays, gi_range, gi_int, g_gi.trace.width,
			            g_gi.trace.height, color.extent.width, color.extent.height, normals ? "the G-buffer + depth" : "depth",
			            albedo ? "the G-buffer" : cfg.gi > 0 ? "the colour's chroma" : "-",
			            HwAoReplacesGtao() ? "; replaces GTAO" : "", view ? "; debug view" : "");
			std::fflush(stdout);
		}
	}

	FrameUbo ubo {};
	std::memcpy(ubo.inv, scene.inv.data(), sizeof(ubo.inv));
	std::memcpy(ubo.vp, scene.vp.data(), sizeof(ubo.vp));
	ubo.cam[0] = scene.cam[0], ubo.cam[1] = scene.cam[1], ubo.cam[2] = scene.cam[2], ubo.cam[3] = ao_radius;
	ubo.motion[0] = inputs.motion_scale_x, ubo.motion[1] = inputs.motion_scale_y;
	ubo.motion[2] = static_cast<float>(color.extent.width), ubo.motion[3] = static_cast<float>(color.extent.height);
	ubo.trace[0] = static_cast<float>(g_gi.trace.width), ubo.trace[1] = static_cast<float>(g_gi.trace.height);
	ubo.trace[2] = cfg.ao > 0 ? ao_strength : 0.0f, ubo.trace[3] = gi_int;
	ubo.par[0] = gi_range, ubo.par[1] = ao_floor, ubo.par[2] = fade_start, ubo.par[3] = fade_start * 2.0f;
	ubo.cfg[0] = ao_rays, ubo.cfg[1] = gi_rays, ubo.cfg[2] = level;
	ubo.cfg[3] = (inputs.reset ? 1 : 0) | (g_gi.history_valid ? 2 : 0) | (normals ? 4 : 0) | (albedo ? 8 : 0) | (half ? 16 : 0) |
	             (cfg.ao > 0 ? 32 : 0) | (cfg.gi > 0 ? 64 : 0) | (flip_y ? 128 : 0);
	ubo.cfg2[0] = static_cast<int32_t>(g_gi.frame), ubo.cfg2[1] = view;

	const auto entry = static_cast<uint32_t>(g_gi.written % 8);
	const bool timed = g_gi.pool != nullptr;
	if (timed) {
		command.resetQueryPool(g_gi.pool, entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_gi.pool, entry * 2);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
	command.updateBuffer(vk::Buffer(g_gi.ubo), 0, sizeof(ubo), &ubo);
	for (auto* img : TraceImages())
		ToGeneral(command, *img);
	if (!g_gi.dummy_ready) {
		g_gi.dummy_ready = true;
		ToGeneral(command, g_gi.dummy);
		vk::ClearColorValue             zero {};
		zero.uint32 = std::array<uint32_t, 4> {0, 0, 0, 0};
		const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eTransfer,
		        vk::AccessFlagBits2::eTransferWrite);
		command.clearColorImage(vk::Image(g_gi.dummy.image), vk::ImageLayout::eGeneral, &zero, 1, &whole);
	}
	// (the AS build before the ray queries: hwrt-scene.cpp's own barrier covers compute; this one covers everything else)
	Barrier(command,
	        vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
	        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
	        vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eUniformRead |
	            vk::AccessFlagBits2::eAccelerationStructureReadKHR);

	const uint32_t cur  = g_gi.frame & 1u;
	const uint32_t prev = cur ^ 1u;
	auto&          g    = g_gi;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipeline);
	const auto run = [&](int mode, const Image& a, const Image& b, const Image& out, const Image& gout, const Image& aux_in,
	                     const Image& aux_out, float step, vk::Extent2D size) {
		const vk::Sampler       lin = g.linear, pt = g.point;
		const auto              gen = vk::ImageLayout::eGeneral;
		vk::DescriptorImageInfo infos[13] {};
		infos[0]  = {pt, color.view, color.layout};
		infos[1]  = {pt, inputs.depth.view, inputs.depth.layout};
		infos[2]  = {pt, inputs.motion.view, inputs.motion.layout};
		infos[3]  = {lin, a.view, gen};
		infos[4]  = {lin, b.view, gen};
		infos[5]  = {pt, g.gbuf[cur].view, gen};
		infos[6]  = {pt, g.gbuf[prev].view, gen};
		infos[7]  = {nullptr, out.view, gen};
		infos[8]  = {nullptr, gout.view, gen};
		infos[9]  = {pt, aux_in.view, gen};
		infos[10] = {nullptr, aux_out.view, gen};
		infos[11] = normals ? vk::DescriptorImageInfo {pt, gbuf.views[3], vk::ImageLayout::eShaderReadOnlyOptimal}
		                    : vk::DescriptorImageInfo {pt, g.dummy.view, gen};
		infos[12] = albedo ? vk::DescriptorImageInfo {pt, gbuf.views[0], vk::ImageLayout::eShaderReadOnlyOptimal}
		                   : vk::DescriptorImageInfo {pt, g.dummy.view, gen};
		vk::WriteDescriptorSetAccelerationStructureKHR as_write {};
		as_write.accelerationStructureCount = 1;
		as_write.pAccelerationStructures    = &scene.tlas;
		const vk::DescriptorBufferInfo ubo_info {vk::Buffer(g.ubo), 0, sizeof(FrameUbo)};
		vk::WriteDescriptorSet         writes[Bindings] {};
		for (uint32_t i = 0; i < Bindings; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = DescType(i);
			if (i == 13) writes[i].pNext = &as_write;
			else if (i == 14) writes[i].pBufferInfo = &ubo_info;
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
	};
	// (bindings a mode does not write still need a valid storage image: tmp_b is the scratch target)
	run(0, g.tmp_a, g.tmp_b, g.raw, g.gbuf[cur], g.count[prev], g.tmp_b, 0.0f, g.trace);                // trace
	run(1, g.raw, g.history[prev], g.history[cur], g.tmp_b, g.count[prev], g.count[cur], 0.0f, g.trace); // temporal
	const Image* src     = &g.history[cur];
	const Image* ping[2] = {&g.tmp_a, &g.raw};
	for (int i = 0; i < passes; ++i) {
		const Image* dst = ping[i & 1];
		run(2, *src, g.tmp_b, *dst, g.tmp_b, g.count[cur], g.tmp_b, static_cast<float>(1 << i), g.trace);
		src = dst;
	}
	run(3, *src, g.tmp_b, g.out, g.tmp_b, g.count[cur], g.tmp_b, 0.0f, g.render); // composite
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead);
	if (timed) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g.pool, entry * 2 + 1);
		++g.written;
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
