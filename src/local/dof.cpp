// KYTY_DOF=0..3 / No / Suave / Cinematico / Fuerte (ds-dof-1029; dof.comp holds the algorithm): a distance-only depth
// of field ("al fondo las cosas mas borrosas"), recorded in FrameGen::Upscale after every other pass (the lens flare,
// the god rays) and before the upscaler; the HUD is drawn after the upscaler and stays sharp.
//
// Why not the game's own: Demon's Souls' r_dofEnable is the only DoF convar the launcher (and the docs) know, an on /
// off switch; the game drives its DoF's focus from the cutscene / menu cameras, and in gameplay it does nothing at a
// distance. There is no known convar for a focus distance or a far blur, so the distance blur is ours.
//
//   Suave      (1): focus 40 m -> full at 3.0x (120 m), radius 0.6 % of the height, 24 taps, highlight boost 1.0
//   Cinematico (2): focus 30 m -> full at 2.5x (75 m),  radius 1.1 % of the height, 36 taps, highlight boost 2.0
//   Fuerte     (3): focus 25 m -> full at 2.2x (55 m),  radius 1.7 % of the height, 48 taps, highlight boost 2.5
//   Passes (half res except the last): prep (colour + nearest view z + CoC) -> bokeh gather disc -> temporal
//   (reprojected, z-rejected, neighbourhood-clamped) -> composite (full res, depth-aware upsample, in-focus pixels
//   untouched). The sky counts as far (full blur).
// Knobs: KYTY_DOF_FOCUS (metres, the level's), KYTY_DOF_RAMP (x the focus where the blur is full, the level's),
// KYTY_DOF_RADIUS (x the level's radius, 0..3), KYTY_DOF_BOKEH (x the highlight boost, 0..4), KYTY_DOF_VIEW=1 (the blur
// amount alone, white = full).
// Every 10 s one "DOF:" line: the level, the focus and ramp end, the share of pixels blurred, GPU ms (and per pass).
#include "local/local-pass-cache.h" // (ds-perf-1031: the passes' pipelines in the disk cache)
#include "local/dof.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/dof_spv.h"
#include "local/rt-flicker.h"

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
	if (std::strncmp(w, "suav", 4) == 0 || std::strncmp(w, "subt", 4) == 0 || std::strncmp(w, "low", 3) == 0) return 1;
	if (std::strncmp(w, "cine", 4) == 0 || std::strncmp(w, "medi", 4) == 0) return 2;
	if (std::strncmp(w, "fuer", 4) == 0 || std::strncmp(w, "stro", 4) == 0 || std::strncmp(w, "high", 4) == 0) return 3;
	return 0;
}
const char* LevelName(int l) {
	static const char* names[] {"No", "Suave", "Cinematico", "Fuerte"};
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

// std140 (dof.comp Frame)
struct FrameUbo {
	float   proj[4];
	float   depth_map[4];
	float   motion[4];
	float   half_[4];
	float   par[4];
	int32_t cfg[4];
	int32_t cfg2[4];
};
static_assert(sizeof(FrameUbo) == 112);
struct Push {
	int32_t mode = 0;
	int32_t pad[3] {};
};

constexpr uint32_t Bindings = 12;
constexpr uint32_t Stamps   = 5; // start, after prep, bokeh, temporal, composite
constexpr uint32_t SlotWords = 16, Slots = 4;

struct State {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             point = nullptr, linear = nullptr;
	bool                    failed = false;
	VkBuffer                ubo = VK_NULL_HANDLE, one = VK_NULL_HANDLE, stats = VK_NULL_HANDLE, stats_dev = VK_NULL_HANDLE;
	VmaAllocation           ubo_mem = VK_NULL_HANDLE, one_mem = VK_NULL_HANDLE, stats_mem = VK_NULL_HANDLE, stats_dev_mem = VK_NULL_HANDLE;
	uint32_t*               stats_map = nullptr;
	bool                    one_ready = false;
	vk::Extent2D            render {}, half {};
	Image                   prep, bokeh, hist[2], lz[2], out;
	uint32_t                frame = 0;
	bool                    history_valid = false;
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  ms_sum = 0.0;
	std::array<double, Stamps - 1> pass_ms {};
	uint32_t                ms_n = 0;
	uint64_t                px = 0, blurred = 0;
	uint32_t                frames_on = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
} g_dof;

std::array<Image*, 7> Images() { return {&g_dof.prep, &g_dof.bokeh, &g_dof.hist[0], &g_dof.hist[1], &g_dof.lz[0], &g_dof.lz[1], &g_dof.out}; }

bool MakeImage(GraphicContext& graphics, vk::Format format, vk::Extent2D extent, Image& img) {
	VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create.imageType     = VK_IMAGE_TYPE_2D;
	create.format        = static_cast<VkFormat>(format);
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.samples       = VK_SAMPLE_COUNT_1_BIT;
	create.tiling        = VK_IMAGE_TILING_OPTIMAL;
	create.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
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
	if (i == 9) return vk::DescriptorType::eUniformBuffer;
	if (i == 10 || i == 11) return vk::DescriptorType::eStorageBuffer;
	if (i == 7 || i == 8) return vk::DescriptorType::eStorageImage;
	return vk::DescriptorType::eCombinedImageSampler;
}

bool MakeBuffer(GraphicContext& graphics, VkDeviceSize size, VkBufferUsageFlags usage, bool host, VkBuffer& buf, VmaAllocation& mem,
                void** map) {
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = size;
	buffer.usage = usage;
	VmaAllocationCreateInfo info {};
	info.usage = host ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (host) info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &info, &buf, &mem, &mapped) != VK_SUCCESS) {
		buf = VK_NULL_HANDLE;
		return false;
	}
	if (map != nullptr) *map = mapped.pMappedData;
	return true;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_dof.pipeline != nullptr) return true;
	if (g_dof.failed) return false;
	g_dof.failed      = true;
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
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_dof.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Push)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_dof.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_dof.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = sizeof(DOF_SPV);
	module_info.pCode    = DOF_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_dof.layout;
	const auto created = device.createComputePipelines(LocalPassCache::Get(device), 1, &pipeline_info, nullptr, &g_dof.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_dof.pipeline = nullptr;
		return false;
	}
	for (int k = 0; k < 2; ++k) {
		vk::SamplerCreateInfo sampler {};
		sampler.magFilter    = k == 0 ? vk::Filter::eNearest : vk::Filter::eLinear;
		sampler.minFilter    = sampler.magFilter;
		sampler.mipmapMode   = vk::SamplerMipmapMode::eNearest;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		sampler.maxLod       = 0.0f;
		if (device.createSampler(&sampler, nullptr, k == 0 ? &g_dof.point : &g_dof.linear) != vk::Result::eSuccess) return false;
	}
	if (!MakeBuffer(graphics, sizeof(FrameUbo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, g_dof.ubo,
	                g_dof.ubo_mem, nullptr))
		return false;
	if (!MakeBuffer(graphics, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, g_dof.one, g_dof.one_mem,
	                nullptr))
		return false;
	void* map = nullptr;
	if (!MakeBuffer(graphics, SlotWords * Slots * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true,
	                g_dof.stats, g_dof.stats_mem, &map))
		return false;
	g_dof.stats_map = static_cast<uint32_t*>(map);
	std::memset(g_dof.stats_map, 0, SlotWords * Slots * 4);
	if (!MakeBuffer(graphics, SlotWords * Slots * 4,
	                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false,
	                g_dof.stats_dev, g_dof.stats_dev_mem, nullptr))
		return false;
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = Stamps * 8;
	if (device.createQueryPool(&query, nullptr, &g_dof.pool) != vk::Result::eSuccess) g_dof.pool = nullptr;
	g_dof.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g_dof.failed    = false;
	return true;
}

bool EnsureImages(GraphicContext& graphics, vk::Extent2D render) {
	const vk::Extent2D half {std::max(1u, (render.width + 1) / 2), std::max(1u, (render.height + 1) / 2)};
	if (g_dof.out.image != VK_NULL_HANDLE && g_dof.render == render) return true;
	for (auto* img: Images()) FreeImage(graphics, *img);
	bool ok = MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, render, g_dof.out);
	ok      = ok && MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, half, g_dof.prep);
	ok      = ok && MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, half, g_dof.bokeh);
	for (int i = 0; i < 2; ++i) {
		ok = ok && MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, half, g_dof.hist[i]);
		ok = ok && MakeImage(graphics, vk::Format::eR32Sfloat, half, g_dof.lz[i]);
	}
	if (!ok) {
		for (auto* img: Images()) FreeImage(graphics, *img);
		g_dof.failed = true;
		return false;
	}
	g_dof.render        = render;
	g_dof.half          = half;
	g_dof.history_valid = false;
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
	while (g_dof.pool != nullptr && g_dof.read + 4 <= g_dof.written) {
		const auto                   entry = static_cast<uint32_t>(g_dof.read % 8);
		std::array<uint64_t, Stamps> stamps {};
		if (graphics.device.getQueryPoolResults(g_dof.pool, entry * Stamps, Stamps, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		g_dof.ms_sum += static_cast<double>(stamps[Stamps - 1] - stamps[0]) * g_dof.period_ns / 1e6;
		for (uint32_t i = 0; i + 1 < Stamps; ++i)
			g_dof.pass_ms[i] += static_cast<double>(stamps[i + 1] - stamps[i]) * g_dof.period_ns / 1e6;
		++g_dof.ms_n;
		++g_dof.read;
	}
}

void ReadStats(GraphicContext& graphics, int level, float focus, float ramp_end, float radius) {
	if (g_dof.stats_map == nullptr || g_dof.frame < Slots) return;
	(void)vmaInvalidateAllocation(graphics.allocator, g_dof.stats_mem, 0, VK_WHOLE_SIZE);
	const uint32_t* w = g_dof.stats_map + ((g_dof.frame + 1) % Slots) * SlotWords;
	g_dof.px += w[0];
	g_dof.blurred += w[1];
	const auto now = std::chrono::steady_clock::now();
	if (now - g_dof.last_log < std::chrono::seconds(10)) return;
	const auto pm = [](int i) { return g_dof.ms_n > 0 ? g_dof.pass_ms[i] / g_dof.ms_n : 0.0; };
	std::printf("DOF: 10 s: level %s (%d), focus %.1f m (full blur at %.1f m), max radius %.1f half-res px, %ux%u half for %ux%u, "
	            "blurred %.1f%% of the pixels, GPU %.3f ms (mean of %u frames; prep %.3f, bokeh %.3f, temporal %.3f, composite %.3f), "
	            "%u frames on\n",
	            LevelName(level), level, focus, ramp_end, radius, g_dof.half.width, g_dof.half.height, g_dof.render.width, g_dof.render.height,
	            g_dof.px > 0 ? 100.0 * static_cast<double>(g_dof.blurred) / static_cast<double>(g_dof.px) : 0.0,
	            g_dof.ms_n > 0 ? g_dof.ms_sum / g_dof.ms_n : 0.0, g_dof.ms_n, pm(0), pm(1), pm(2), pm(3), g_dof.frames_on);
	std::fflush(stdout);
	g_dof.px = g_dof.blurred = 0;
	g_dof.frames_on          = 0;
	g_dof.ms_sum = 0.0, g_dof.ms_n = 0;
	g_dof.pass_ms.fill(0.0);
	g_dof.last_log = now;
}

} // namespace

int DofLevel() {
	static const int level = [] {
		const int l = ParseLevel(std::getenv("KYTY_DOF"));
		if (l > 0) {
			std::printf("DOF: KYTY_DOF=%d (%s) read\n", l, LevelName(l));
			std::fflush(stdout);
		}
		return l;
	}();
	return level;
}

bool DofApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int level = DofLevel();
	if (level == 0) return false;
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr) return false;
	if (VideoOut::MoviePlaying()) {
		g_dof.history_valid = false;
		return false;
	}
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || color.extent.width < 16 || color.extent.height < 16 ||
	    inputs.depth.extent != color.extent)
		return false;
	const auto& m = inputs.view_to_clip;
	if (!(m[0] != 0.0f && m[5] != 0.0f && (m[11] != 0.0f || m[15] != 0.0f)) ||
	    !std::all_of(m.begin(), m.end(), [](float v) { return std::isfinite(v); }))
		return false;
	static constexpr float focus_by[4] {0.0f, 40.0f, 30.0f, 25.0f};
	static constexpr float ramp_by[4] {1.0f, 3.0f, 2.5f, 2.2f};
	static constexpr float radius_by[4] {0.0f, 0.006f, 0.011f, 0.017f}; // of the render height (full-res px)
	static constexpr int   taps_by[4] {0, 24, 36, 48};
	static constexpr float boost_by[4] {0.0f, 1.0f, 2.0f, 2.5f};
	static const float     focus  = EnvF("KYTY_DOF_FOCUS", focus_by[level], 2.0f, 2000.0f);
	static const float     ramp   = EnvF("KYTY_DOF_RAMP", ramp_by[level], 1.1f, 10.0f);
	static const float     rscale = EnvF("KYTY_DOF_RADIUS", 1.0f, 0.0f, 3.0f);
	static const float     boost  = EnvF("KYTY_DOF_BOKEH", 1.0f, 0.0f, 4.0f) * boost_by[level];
	static const bool      view   = EnvF("KYTY_DOF_VIEW", 0.0f, 0.0f, 1.0f) > 0.5f;
	if (!EnsurePipeline(*graphics) || !EnsureImages(*graphics, color.extent)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("DOF: off (could not create its pipeline or images)\n");
			std::fflush(stdout);
		}
		return false;
	}
	// in half-res pixels, capped so the gather stays cheap
	const float radius = std::min(radius_by[level] * rscale * static_cast<float>(color.extent.height) * 0.5f, 24.0f);
	ReadTiming(*graphics);
	ReadStats(*graphics, level, focus, focus * ramp, radius);
	const bool still = RtFlicker::CameraStill();
	++g_dof.frames_on;
	const bool game_exposure = inputs.exposure_buffer != nullptr && inputs.exposure_range >= sizeof(float);
	{
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("DOF: on (%s: focus %g m, full blur at %g m, radius %.1f half-res px, %d taps, highlight boost %g, %ux%u half for "
			            "%ux%u, exposure %s%s)\n",
			            LevelName(level), focus, focus * ramp, radius, taps_by[level], boost, g_dof.half.width, g_dof.half.height,
			            color.extent.width, color.extent.height, game_exposure ? "the game's" : "1.0", view ? "; blur view" : "");
			std::fflush(stdout);
		}
	}

	FrameUbo ubo {};
	ubo.proj[0] = m[0], ubo.proj[1] = m[5], ubo.proj[2] = m[8], ubo.proj[3] = m[9];
	ubo.depth_map[0] = m[10], ubo.depth_map[1] = m[14], ubo.depth_map[2] = m[11], ubo.depth_map[3] = m[15];
	ubo.motion[0] = inputs.motion_scale_x, ubo.motion[1] = inputs.motion_scale_y;
	ubo.motion[2] = static_cast<float>(color.extent.width), ubo.motion[3] = static_cast<float>(color.extent.height);
	ubo.half_[0] = static_cast<float>(g_dof.half.width), ubo.half_[1] = static_cast<float>(g_dof.half.height);
	ubo.half_[2] = radius, ubo.half_[3] = focus;
	ubo.par[0] = focus * ramp, ubo.par[1] = boost, ubo.par[2] = 1.0f, ubo.par[3] = 0.125f;
	const bool motion = inputs.motion.view != nullptr;
	ubo.cfg[0] = level, ubo.cfg[1] = static_cast<int32_t>(g_dof.frame);
	ubo.cfg[2] = ((g_dof.history_valid && !inputs.reset) ? 1 : 0) | (motion ? 2 : 0) | (still ? 4 : 0);
	ubo.cfg[3] = static_cast<int32_t>(g_dof.frame % Slots);
	ubo.cfg2[0] = taps_by[level], ubo.cfg2[1] = game_exposure ? static_cast<int32_t>(inputs.exposure_index) : 0, ubo.cfg2[2] = view ? 1 : 0;

	auto&      g     = g_dof;
	const auto entry = static_cast<uint32_t>(g.written % 8);
	const bool timed = g.pool != nullptr;
	if (timed) {
		command.resetQueryPool(g.pool, entry * Stamps, Stamps);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g.pool, entry * Stamps);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
	command.updateBuffer(vk::Buffer(g.ubo), 0, sizeof(ubo), &ubo);
	if (!g.one_ready) {
		g.one_ready = true;
		const float one[4] {1.0f, 1.0f, 1.0f, 1.0f};
		command.updateBuffer(vk::Buffer(g.one), 0, sizeof(one), one);
	}
	command.fillBuffer(vk::Buffer(g.stats_dev), (g.frame % Slots) * SlotWords * 4, SlotWords * 4, 0);
	for (auto* img: Images()) ToGeneral(command, *img);
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eUniformRead);

	const uint32_t cur  = g.frame & 1u;
	const uint32_t prev = cur ^ 1u;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipeline);
	uint32_t   stamp = 0;
	const auto run   = [&](int mode, const Image& a, const Image& out, vk::Extent2D size) {
        const vk::Sampler       pt  = g.point;
        const auto              gen = vk::ImageLayout::eGeneral;
        vk::DescriptorImageInfo infos[Bindings] {};
        infos[0] = {pt, color.view, color.layout};
        infos[1] = {pt, inputs.depth.view, inputs.depth.layout};
        infos[2] = motion ? vk::DescriptorImageInfo {pt, inputs.motion.view, inputs.motion.layout} : vk::DescriptorImageInfo {pt, g.lz[prev].view, gen};
        infos[3] = {pt, a.view, gen};
        infos[4] = {g.linear, g.hist[prev].view, gen};
        infos[5] = {pt, g.lz[cur].view, gen};
        infos[6] = {pt, g.lz[prev].view, gen};
        infos[7] = {nullptr, out.view, gen};
        infos[8] = {nullptr, g.lz[cur].view, gen};
        const vk::DescriptorBufferInfo ubo_info {vk::Buffer(g.ubo), 0, sizeof(FrameUbo)};
        const vk::DescriptorBufferInfo st_info {vk::Buffer(g.stats_dev), 0, SlotWords * Slots * 4};
        const vk::DescriptorBufferInfo exp_info = game_exposure
                                                      ? vk::DescriptorBufferInfo {inputs.exposure_buffer, inputs.exposure_base, inputs.exposure_range}
                                                      : vk::DescriptorBufferInfo {vk::Buffer(g.one), 0, 16};
        vk::WriteDescriptorSet writes[Bindings] {};
        for (uint32_t i = 0; i < Bindings; ++i) {
            writes[i].dstBinding      = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = DescType(i);
            if (i == 9) writes[i].pBufferInfo = &ubo_info;
            else if (i == 10) writes[i].pBufferInfo = &st_info;
            else if (i == 11) writes[i].pBufferInfo = &exp_info;
            else writes[i].pImageInfo = &infos[i];
        }
        Push push {};
        push.mode = mode;
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g.layout, 0, Bindings, writes);
        command.pushConstants(g.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
        command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
        Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
        if (timed) command.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, g.pool, entry * Stamps + (++stamp));
	};
	run(0, g.bokeh, g.prep, g.half);         // prep -> prep, lz[cur]
	run(1, g.prep, g.bokeh, g.half);         // bokeh gather -> bokeh
	run(2, g.bokeh, g.hist[cur], g.half);    // temporal -> hist[cur]
	run(3, g.hist[cur], g.out, g.render);    // composite -> out (full res)
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite, vk::PipelineStageFlagBits2::eAllCommands,
	        vk::AccessFlagBits2::eMemoryRead);
	{
		const vk::DeviceSize   off = (g.frame % Slots) * SlotWords * 4;
		const vk::BufferCopy region {off, off, SlotWords * 4};
		command.copyBuffer(vk::Buffer(g.stats_dev), vk::Buffer(g.stats), 1, &region);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
		        vk::AccessFlagBits2::eHostRead);
	}
	if (timed) ++g.written;
	++g.frame;
	g.history_valid     = true;
	inputs.color.image  = vk::Image(g.out.image);
	inputs.color.view   = g.out.view;
	inputs.color.layout = vk::ImageLayout::eGeneral;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc |
	                     vk::ImageUsageFlagBits::eTransferDst;
	inputs.color.level = 0;
	inputs.color.layer = 0;
	return true;
}

} // namespace Libs::Graphics::FrameGen
