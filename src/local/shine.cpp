// KYTY_SHINE=0|1|2|3 ("Shine": off / sun / + 2 local lights / + 4 local lights; shine.comp): makes armour, shields and
// props read sharper and shinier on Demon's Souls' PS5 Pro path. Recorded in FrameGen::Upscale after the other
// screen-space passes (RT shadows, SSGI, GTAO, SSR, RT reflections), on the pre-exposure HDR color, before the upscaler.
//
// Per frame (compute, one pipeline, push descriptors), three dispatches:
//   lights    (level >= 2): a 32x18 grid, 16 taps a cell: the brightest emissive tap above KYTY_SHINE_LIGHT (exposed
//             luminance, 3.0) that is not sky and nearer than 60 m
//   select    one thread: the 2 / 4 strongest with suppression, their view positions from the depth, intensity smoothed
//   composite full resolution: the G-buffer material (KYTY_RT_REFLECTIONS' packed copy: roughness byte, albedo L0.b0..2,
//             world normal L3 oct 8:8) -> CAS-like sharpen gated by smooth / metal / no depth edge / no foliage-like
//             normal noise; GGX highlights from the sun (KYTY_RT_SHADOWS' latched direction; without it a soft sky fill
//             from world up) and the lights, display-referred (/ the game's exposure), soft-capped, Toksvig-like widened
//             where the normal varies (temporally stable). Sky and fog (beyond ~35..110 m) are untouched.
// Decode overrides: KYTY_SHINE_NORMAL ("<layer>:<layout>[w][n]", default 3:9w), KYTY_SHINE_ROUGH ("<layer>:<shift>:<bits>[g]",
// default 2:0:8), KYTY_SHINE_GLOSS (same syntax, default none; e.g. 1:16:8g), KYTY_SHINE_FLIPY (1: row 0 = NDC y +1, default).
// Every 10 s: "SHINE: stats" with the share of pixels highlighted / sharpened, the mean lights found and the GPU ms.
#include "local/shine.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/shine_spv.h"
#include "local/rt-reflections.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace Libs::Graphics::FrameGen {
namespace {

constexpr uint32_t Bindings    = 9;
constexpr uint32_t HeaderWords = 16;
constexpr uint32_t StatsWord   = 16;
constexpr uint32_t StatsCount  = 16;
constexpr uint32_t Cells       = 32 * 18;
constexpr uint32_t FrameBytes  = (64 + Cells * 4) * 4;
constexpr uint32_t ExposureIdx = 12; // hparams.x (1.0) when the game gives no exposure

struct ShinePush {
	float   proj[4];
	float   depth_map[4];
	float   size[4];
	float   sun[4];
	float   fill[4];
	int32_t misc[4];
	int32_t dec[4];
};
static_assert(sizeof(ShinePush) == 112);

struct {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             point = nullptr, linear = nullptr;
	bool                    failed = false;
	VkImage                 image  = VK_NULL_HANDLE;
	VmaAllocation           memory = nullptr;
	vk::ImageView           view   = nullptr;
	vk::Extent2D            extent {};
	bool                    image_ready = false;
	VkImage                 dummy = VK_NULL_HANDLE;
	VmaAllocation           dummy_mem = nullptr;
	vk::ImageView           dummy_view = nullptr;
	bool                    dummy_ready = false;
	VkBuffer                frame_buf = VK_NULL_HANDLE, read_buf = VK_NULL_HANDLE;
	VmaAllocation           frame_mem = nullptr, read_mem = nullptr;
	const uint32_t*         mapped = nullptr;
	bool                    cleared = false;
	uint64_t                frame = 0;
	// Retired output images (a size change; frames in flight may still read them).
	std::array<std::pair<std::pair<VkImage, VmaAllocation>, std::pair<vk::ImageView, uint64_t>>, 4> retired {};
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms = 0.0;
	uint32_t                samples = 0;
	std::array<uint32_t, StatsCount> stat_last {};
	uint64_t                frames = 0, gbuf_frames = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
} g_sh;

struct {
	std::mutex                            mutex;
	std::array<float, 3>                  dir {};
	float                                 weight = 0.0f;
	std::chrono::steady_clock::time_point when {};
} g_shine_sun;

float EnvFloat(const char* name, float fallback, float lo, float hi) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') return fallback;
	const float v = static_cast<float>(std::atof(value));
	return std::isfinite(v) ? std::clamp(v, lo, hi) : fallback;
}
// "<layer>:<shift>:<bits>[g]" -> the shader's field code (fallback: unset or malformed; "none" -> -1).
int32_t FieldEnv(const char* name, int32_t fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') return fallback;
	if (std::strcmp(value, "none") == 0) return -1;
	int  layer = -1, shift = -1, bits = -1;
	char flag  = 0;
	if (std::sscanf(value, "%d:%d:%d%c", &layer, &shift, &bits, &flag) < 3) return fallback;
	if (layer < 0 || layer > 3 || shift < 0 || bits < 1 || bits > 16 || shift + bits > 32) return fallback;
	return layer | (shift << 2) | (bits << 7) | (flag == 'g' ? 4096 : 0);
}
// "<layer>:<layout>[w][n]" -> the shader's normal code (layouts 0, 5, 9, 10 of rt-reflections.comp).
int32_t NormalEnv(const char* name, int32_t fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') return fallback;
	int layer = -1, layout = -1;
	if (std::sscanf(value, "%d:%d", &layer, &layout) < 2 || layer < 0 || layer > 3 ||
	    !(layout == 0 || layout == 5 || layout == 9 || layout == 10))
		return fallback;
	const std::string v = value;
	return layer | (layout << 2) | (v.find('w') != std::string::npos ? 64 : 0) | (v.find('n') != std::string::npos ? 128 : 0);
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

bool MakeImage(GraphicContext& graphics, vk::Format format, vk::Extent2D extent, bool storage, VkImage& image, VmaAllocation& memory,
               vk::ImageView& view) {
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
	if (vmaCreateImage(graphics.allocator, &create, &allocate, &image, &memory, nullptr) != VK_SUCCESS) {
		image = VK_NULL_HANDLE;
		return false;
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = vk::Image(image);
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = format;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &view) != vk::Result::eSuccess) {
		view = nullptr;
		return false;
	}
	return true;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_sh.pipeline != nullptr) return true;
	if (g_sh.failed) return false;
	g_sh.failed       = true;
	const auto device = graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, Bindings> bindings {};
	for (uint32_t i = 0; i < Bindings; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = i >= 7 ? vk::DescriptorType::eStorageBuffer
		                              : i == 6 ? vk::DescriptorType::eStorageImage
		                                       : vk::DescriptorType::eCombinedImageSampler;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = Bindings;
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_sh.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(ShinePush)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_sh.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_sh.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(SHINE_SPV);
	module_info.pCode       = SHINE_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_sh.layout;
	const auto created         = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_sh.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_sh.pipeline = nullptr;
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
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g_sh.point : &g_sh.linear) != vk::Result::eSuccess) return false;
	}
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = FrameBytes;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VmaAllocationCreateInfo dev {};
	dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_sh.frame_buf, &g_sh.frame_mem, nullptr) != VK_SUCCESS) {
		g_sh.frame_buf = VK_NULL_HANDLE;
		return false;
	}
	buffer.size  = StatsCount * 4;
	buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_sh.read_buf, &g_sh.read_mem, &mapped) != VK_SUCCESS) {
		g_sh.read_buf = VK_NULL_HANDLE;
		return false;
	}
	g_sh.mapped = static_cast<const uint32_t*>(mapped.pMappedData);
	if (!MakeImage(graphics, vk::Format::eR32Uint, {1, 1}, false, g_sh.dummy, g_sh.dummy_mem, g_sh.dummy_view)) return false;
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = 8 * 2;
	if (device.createQueryPool(&query, nullptr, &g_sh.pool) != vk::Result::eSuccess) g_sh.pool = nullptr;
	g_sh.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g_sh.failed    = false;
	return true;
}

bool EnsureImage(GraphicContext& graphics, vk::Extent2D extent) {
	if (g_sh.image != VK_NULL_HANDLE && g_sh.extent == extent) return true;
	if (g_sh.image != VK_NULL_HANDLE) {
		// (retired, destroyed 16 frames later)
		auto& slot = g_sh.retired[g_sh.frame % g_sh.retired.size()];
		if (slot.first.first != VK_NULL_HANDLE) {
			graphics.device.destroyImageView(slot.second.first, nullptr);
			vmaDestroyImage(graphics.allocator, slot.first.first, slot.first.second);
		}
		slot = {{g_sh.image, g_sh.memory}, {g_sh.view, g_sh.frame}};
		g_sh.image = VK_NULL_HANDLE, g_sh.memory = nullptr, g_sh.view = nullptr;
	}
	g_sh.image_ready = false;
	g_sh.extent      = extent;
	return MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, extent, true, g_sh.image, g_sh.memory, g_sh.view);
}
void FreeRetired(GraphicContext& graphics) {
	for (auto& slot: g_sh.retired)
		if (slot.first.first != VK_NULL_HANDLE && g_sh.frame >= slot.second.second + 16) {
			graphics.device.destroyImageView(slot.second.first, nullptr);
			vmaDestroyImage(graphics.allocator, slot.first.first, slot.first.second);
			slot = {};
		}
}

void ReadTiming(GraphicContext& graphics) {
	if (g_sh.pool == nullptr) return;
	while (g_sh.read + 4 <= g_sh.written) {
		const auto               entry = static_cast<uint32_t>(g_sh.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (graphics.device.getQueryPoolResults(g_sh.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		if (stamps[1] > stamps[0]) {
			g_sh.sum_ms += static_cast<double>(stamps[1] - stamps[0]) * g_sh.period_ns / 1e6;
			++g_sh.samples;
		}
		++g_sh.read;
	}
}

void Log(bool have_gbuf, bool have_sun, int level) {
	++g_sh.frames;
	if (have_gbuf) ++g_sh.gbuf_frames;
	const auto now = std::chrono::steady_clock::now();
	if (now - g_sh.last_log < std::chrono::seconds(10) || g_sh.mapped == nullptr) return;
	g_sh.last_log = now;
	std::array<uint32_t, StatsCount> cur {};
	std::memcpy(cur.data(), g_sh.mapped, sizeof(cur)); // (a few frames late: statistics only)
	std::array<uint32_t, StatsCount> d {};
	for (uint32_t i = 0; i < StatsCount; ++i) d[i] = cur[i] - g_sh.stat_last[i];
	g_sh.stat_last  = cur;
	const double px = std::max<double>(d[0], 1.0);
	std::printf("SHINE: stats (level %d) highlighted %.1f %% of pixels, sharpened %.1f %%, lights %.2f a frame, sun %s, G-buffer %llu/%llu "
	            "frames, GPU %.3f ms\n",
	            level, 100.0 * d[1] / px, 100.0 * d[2] / px, static_cast<double>(d[3]) / std::max<double>(static_cast<double>(g_sh.frames), 1.0),
	            have_sun ? "yes" : "no (sky fill)", static_cast<unsigned long long>(g_sh.gbuf_frames), static_cast<unsigned long long>(g_sh.frames),
	            g_sh.samples > 0 ? g_sh.sum_ms / g_sh.samples : 0.0);
	std::fflush(stdout);
	g_sh.frames = g_sh.gbuf_frames = 0;
	g_sh.sum_ms = 0.0, g_sh.samples = 0;
}

} // namespace

int ShineLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_SHINE");
		return value != nullptr ? std::clamp(std::atoi(value), 0, 3) : 0;
	}();
	return level;
}
bool ShineWanted() { return ShineLevel() > 0; }

void ShineSetSun(const std::array<double, 3>& view_dir, float weight) {
	if (!ShineWanted()) return;
	std::lock_guard lock(g_shine_sun.mutex);
	g_shine_sun.dir    = {static_cast<float>(view_dir[0]), static_cast<float>(view_dir[1]), static_cast<float>(view_dir[2])};
	g_shine_sun.weight = std::isfinite(weight) ? std::clamp(weight, 0.0f, 1.0f) : 0.0f;
	g_shine_sun.when   = std::chrono::steady_clock::now();
}

bool ShineApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int level = ShineLevel();
	if (level == 0) return false;
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr || VideoOut::MoviePlaying()) return false;
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || color.extent.width < 16 || color.extent.height < 16 ||
	    inputs.depth.extent != color.extent)
		return false;
	const auto& m = inputs.view_to_clip;
	if (!(m[0] != 0.0f && m[5] != 0.0f && (m[11] != 0.0f || m[15] != 0.0f)) ||
	    !std::all_of(m.begin(), m.end(), [](float v) { return std::isfinite(v); }))
		return false;
	if (!EnsurePipeline(*graphics) || !EnsureImage(*graphics, color.extent)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("SHINE: off (could not create its pipeline or images)\n");
			std::fflush(stdout);
		}
		return false;
	}
	FreeRetired(*graphics);
	ReadTiming(*graphics);

	RtReflGbuffer gbuf;
	const bool    have_gbuf = RtReflGetGbuffer(gbuf) && gbuf.extent == color.extent;
	RtReflCamera  camera;
	RtReflGetCamera(m, camera);
	static const int32_t normal_code = NormalEnv("KYTY_SHINE_NORMAL", 3 | (9 << 2) | 64);
	static const int32_t rough_code  = FieldEnv("KYTY_SHINE_ROUGH", 2 | (0 << 2) | (8 << 7));
	static const int32_t gloss_code  = FieldEnv("KYTY_SHINE_GLOSS", -1);
	static const bool    row0_up     = [] {
		const char* y = std::getenv("KYTY_SHINE_FLIPY");
		return y == nullptr || *y == '\0' || *y != '0';
	}();
	const float strength = EnvFloat("KYTY_SHINE_STRENGTH", 1.0f, 0.0f, 3.0f);
	const float sharpen  = EnvFloat("KYTY_SHINE_SHARPEN", 0.4f, 0.0f, 1.0f);
	const bool  normals  = have_gbuf && ((normal_code & 64) == 0 || camera.rows);

	std::array<float, 3> sun {};
	float                sun_w = 0.0f;
	{
		std::lock_guard lock(g_shine_sun.mutex);
		if (std::chrono::steady_clock::now() - g_shine_sun.when < std::chrono::milliseconds(250)) sun = g_shine_sun.dir, sun_w = g_shine_sun.weight;
	}
	const bool have_sun = sun_w > 0.0f;
	Log(have_gbuf, have_sun, level);
	{
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("SHINE: on (level %d, %ux%u, strength %g, sharpen %g, normal code %d, roughness %d, gloss %d, G-buffer %s, "
			            "sun %s, exposure %s)\n",
			            level, color.extent.width, color.extent.height, strength, sharpen, normal_code, rough_code, gloss_code,
			            have_gbuf ? "found" : "not yet", have_sun ? "from RT shadows" : "unknown (KYTY_RT_SHADOWS gives it; sky fill until then)",
			            inputs.exposure_buffer != nullptr ? "the game's" : "1.0");
			std::fflush(stdout);
		}
	}
	// Without the G-buffer there is no material: nothing to gate on, so the game's color stays as it is.
	if (!have_gbuf) return false;

	ShinePush push {};
	push.proj[0] = m[0], push.proj[1] = m[5], push.proj[2] = m[8], push.proj[3] = m[9];
	push.depth_map[0] = m[10], push.depth_map[1] = m[14], push.depth_map[2] = m[11], push.depth_map[3] = m[15];
	push.size[0] = static_cast<float>(color.extent.width), push.size[1] = static_cast<float>(color.extent.height);
	push.size[2] = strength, push.size[3] = sharpen;
	push.sun[0] = sun[0], push.sun[1] = sun[1], push.sun[2] = sun[2], push.sun[3] = have_sun ? 1.2f * sun_w : 0.0f;
	push.fill[0] = camera.up[0], push.fill[1] = camera.up[1], push.fill[2] = camera.up[2];
	push.fill[3] = camera.up_valid ? (have_sun ? 0.1f : 0.35f) : 0.0f;
	const bool game_exposure = inputs.exposure_buffer != nullptr && inputs.exposure_range >= sizeof(float);
	push.misc[1] = static_cast<int32_t>(g_sh.frame & 0x7fffffff);
	push.misc[2] = (row0_up ? 1 : 0) | (normals ? 2 : 0);
	push.misc[3] = game_exposure ? static_cast<int32_t>(inputs.exposure_index) : static_cast<int32_t>(ExposureIdx);
	push.dec[0] = normal_code, push.dec[1] = rough_code, push.dec[2] = gloss_code, push.dec[3] = level;

	const auto entry = static_cast<uint32_t>(g_sh.written % 8);
	const bool timed = g_sh.pool != nullptr && g_sh.written < g_sh.read + 8;
	if (timed) {
		command.resetQueryPool(g_sh.pool, entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_sh.pool, entry * 2);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite);
	if (!g_sh.image_ready) ToGeneral(command, g_sh.image), g_sh.image_ready = true;
	if (!g_sh.dummy_ready) ToGeneral(command, g_sh.dummy), g_sh.dummy_ready = true;
	if (!g_sh.cleared) {
		g_sh.cleared = true;
		command.fillBuffer(vk::Buffer(g_sh.frame_buf), 0, VK_WHOLE_SIZE, 0);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer,
		        vk::AccessFlagBits2::eTransferWrite);
	}
	{
		std::array<float, HeaderWords> header {};
		std::copy(camera.view_rows.begin(), camera.view_rows.end(), header.begin());
		header[12] = 1.0f;                                             // (ExposureIdx)
		header[13] = EnvFloat("KYTY_SHINE_LIGHT", 3.0f, 0.5f, 100.0f); // the emissive threshold (exposed luminance)
		header[14] = EnvFloat("KYTY_SHINE_LIGHT_RADIUS", 3.0f, 0.5f, 20.0f);
		header[15] = level >= 3 ? 4.0f : level == 2 ? 2.0f : 0.0f;
		command.updateBuffer(vk::Buffer(g_sh.frame_buf), 0, sizeof(header), header.data());
	}
	Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_sh.pipeline);

	const auto gen = vk::ImageLayout::eGeneral;
	vk::DescriptorImageInfo infos[Bindings] {};
	infos[0] = {g_sh.linear, color.view, color.layout};
	infos[1] = {g_sh.point, inputs.depth.view, inputs.depth.layout};
	for (uint32_t i = 0; i < 4; ++i)
		infos[2 + i] = gbuf.views[i] != nullptr ? vk::DescriptorImageInfo {g_sh.point, gbuf.views[i], vk::ImageLayout::eShaderReadOnlyOptimal}
		                                        : vk::DescriptorImageInfo {g_sh.point, g_sh.dummy_view, gen};
	infos[6] = {nullptr, g_sh.view, gen};
	const vk::DescriptorBufferInfo frame_info {vk::Buffer(g_sh.frame_buf), 0, VK_WHOLE_SIZE};
	const vk::DescriptorBufferInfo exposure_info =
	    game_exposure ? vk::DescriptorBufferInfo {inputs.exposure_buffer, inputs.exposure_base, inputs.exposure_range} : frame_info;
	vk::WriteDescriptorSet writes[Bindings] {};
	for (uint32_t i = 0; i < Bindings; ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = i >= 7 ? vk::DescriptorType::eStorageBuffer
		                            : i == 6 ? vk::DescriptorType::eStorageImage
		                                     : vk::DescriptorType::eCombinedImageSampler;
		if (i == 7) writes[i].pBufferInfo = &frame_info;
		else if (i == 8) writes[i].pBufferInfo = &exposure_info;
		else writes[i].pImageInfo = &infos[i];
	}
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_sh.layout, 0, Bindings, writes);
	const auto run = [&](int mode, uint32_t gx, uint32_t gy) {
		push.misc[0] = mode;
		command.pushConstants(g_sh.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch(gx, gy, 1);
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	};
	// (level 1: no light search; select still runs once to clear the lights, then never again)
	static bool lights_cleared = false;
	if (level >= 2) run(0, 4, 3);
	if (level >= 2 || !lights_cleared) run(1, 1, 1), lights_cleared = true;
	run(2, (color.extent.width + 7) / 8, (color.extent.height + 7) / 8);
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eMemoryRead);
	{
		vk::BufferCopy region {StatsWord * 4, 0, StatsCount * 4};
		command.copyBuffer(vk::Buffer(g_sh.frame_buf), vk::Buffer(g_sh.read_buf), 1, &region);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
		        vk::AccessFlagBits2::eHostRead);
	}
	if (timed) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_sh.pool, entry * 2 + 1);
		++g_sh.written;
	}
	++g_sh.frame;
	inputs.color.image  = vk::Image(g_sh.image);
	inputs.color.view   = g_sh.view;
	inputs.color.layout = gen;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc |
	                      vk::ImageUsageFlagBits::eStorage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}

} // namespace Libs::Graphics::FrameGen
