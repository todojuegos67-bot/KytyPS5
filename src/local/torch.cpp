// KYTY_TORCH=0|1|2|3 ("Luz de antorchas RT") and KYTY_TORCH_FLARE=0|1 ("Destellos de lente"); torch.comp. Firelight and
// a lens flare from torches, braziers and bonfires on Demon's Souls' PS5 Pro path, screen-space / G-buffer only (no
// acceleration structure, no ray query: works with KYTY_RT_HW_EXPERIMENTAL off).
//
// TorchApply (in FrameGen::Upscale after GTAO, before SSR / RT reflections, so those reflect the lit walls):
//   detect  a 32x18 grid, 8x8 taps a cell: fire pixels (exposed luminance > KYTY_TORCH_LIGHT (1.5), warm, lit colour
//           far above the G-buffer albedo (x4..10), 0.8..120 m)
//   track   one thread: cells clustered into blobs (8-connected, depth-coherent); 2 / 4 / 8 tracked lights (Bajo / Medio
//           / Alto) reprojected with the motion vectors, matched, smoothed; the flicker from each blob's own luminance
//           (fast / slow power, 0.8..1.2, eased); the flare's visibility depth-tested at the source
//   light   full resolution: G-buffer normal + albedo + roughness, windowed 1/r^2, N.L, a 8 / 12 / 16-step march in the
//           depth buffer toward the light (no light through walls / characters), a GGX glint; display-referred,
//           soft-capped, faded with distance; sky, far plane and the fires themselves untouched. No blur, no
//           history: nothing to smear into a halo, nothing to boil.
// TorchFlareApply (last, after the god rays; the HUD is drawn after the upscaler): starburst, streak, 4 ghosts, a faint
//   ring, per visible tracked light, faded at the screen edges and with its visibility.
// Without the G-buffer copy (it comes with KYTY_RT_REFLECTIONS / Reflejos RT) the depth's normals and the colour's chroma
// stand in. Every 10 s: "TORCH: stats" (blobs a frame, lights, lit-pixel %, flares, GPU ms); the RTFLICKER line has
// "torch light" and "lens flare".
#include "local/torch.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/torch_spv.h"
#include "local/rt-flicker.h"
#include "local/rt-reflections.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace Libs::Graphics::FrameGen {
namespace {

constexpr uint32_t Bindings    = 10;
constexpr uint32_t HeaderWords = 16;
constexpr uint32_t StatsWord   = 16;
constexpr uint32_t StatsCount  = 16;
constexpr uint32_t Cells       = 32 * 18;
constexpr uint32_t FrameBytes  = (160 + Cells * 8) * 4;
constexpr uint32_t ExposureIdx = 12; // hp.x (1.0) when the game gives no exposure

struct TorchPush {
	float   proj[4];
	float   depth_map[4];
	float   size[4];
	float   light[4];
	float   flare[4];
	float   motion[4];
	int32_t misc[4];
	int32_t dec[4];
};
static_assert(sizeof(TorchPush) == 128);

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
	Img                     out, flare, dummy;
	vk::Extent2D            extent {};
	bool                    ready = false, dummy_ready = false, history_valid = false;
	VkBuffer                frame_buf = VK_NULL_HANDLE, read_buf = VK_NULL_HANDLE;
	VmaAllocation           frame_mem = nullptr, read_mem = nullptr;
	const uint32_t*         mapped = nullptr;
	bool                    cleared = false;
	uint64_t                frame = 0;
	uint64_t                tracked_frame = ~0ull; // (g_tc.frame when TorchApply last tracked: the flare needs this frame's)
	std::array<std::pair<Img, uint64_t>, 8> retired {};
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms = 0.0, max_ms = 0.0;
	uint32_t                samples = 0;
	std::array<uint32_t, StatsCount> stat_last {};
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now(), last_frame {};
	// What the flare pass needs from TorchApply's frame.
	TorchPush               push {};
	bool                    game_exposure = false;
} g_tc;


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
	for (auto& slot: g_tc.retired)
		if (slot.first.image == VK_NULL_HANDLE) {
			slot = {img, g_tc.frame};
			img  = {};
			return;
		}
	Destroy(graphics, img); // (no free slot: only after many resizes in 16 frames)
}
void FreeRetired(GraphicContext& graphics) {
	for (auto& slot: g_tc.retired)
		if (slot.first.image != VK_NULL_HANDLE && g_tc.frame >= slot.second + 16) Destroy(graphics, slot.first), slot.second = 0;
}

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

vk::DescriptorType DescType(uint32_t i) {
	return (i == 7 || i == 8) ? vk::DescriptorType::eStorageBuffer
	       : i == 6           ? vk::DescriptorType::eStorageImage
	                          : vk::DescriptorType::eCombinedImageSampler;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_tc.pipeline != nullptr) return true;
	if (g_tc.failed) return false;
	g_tc.failed       = true;
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
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_tc.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(TorchPush)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_tc.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_tc.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(TORCH_SPV);
	module_info.pCode       = TORCH_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_tc.layout;
	const auto created         = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_tc.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_tc.pipeline = nullptr;
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
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g_tc.point : &g_tc.linear) != vk::Result::eSuccess) return false;
	}
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = FrameBytes;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VmaAllocationCreateInfo dev {};
	dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_tc.frame_buf, &g_tc.frame_mem, nullptr) != VK_SUCCESS) {
		g_tc.frame_buf = VK_NULL_HANDLE;
		return false;
	}
	buffer.size  = StatsCount * 4;
	buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_tc.read_buf, &g_tc.read_mem, &mapped) != VK_SUCCESS) {
		g_tc.read_buf = VK_NULL_HANDLE;
		return false;
	}
	g_tc.mapped = static_cast<const uint32_t*>(mapped.pMappedData);
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = 8 * 2;
	if (device.createQueryPool(&query, nullptr, &g_tc.pool) != vk::Result::eSuccess) g_tc.pool = nullptr;
	g_tc.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	if (!MakeImage(graphics, vk::Format::eR32Uint, {1, 1}, g_tc.dummy)) return false;
	g_tc.failed = false;
	return true;
}

bool EnsureImages(GraphicContext& graphics, vk::Extent2D extent) {
	if (g_tc.out.image != VK_NULL_HANDLE && g_tc.extent == extent) return true;
	Retire(graphics, g_tc.out);
	Retire(graphics, g_tc.flare);
	g_tc.ready = false, g_tc.history_valid = false;
	g_tc.extent = extent;
	return MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, extent, g_tc.out) &&
	       MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, extent, g_tc.flare);
}

// (both passes' GPU time together: TorchApply opens the timestamp pair, the last pass of the frame closes it)
void ReadTiming(GraphicContext& graphics) {
	if (g_tc.pool == nullptr) return;
	while (g_tc.read + 4 <= g_tc.written) {
		const auto              entry = static_cast<uint32_t>(g_tc.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (graphics.device.getQueryPoolResults(g_tc.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		if (stamps[1] > stamps[0]) {
			const double ms = static_cast<double>(stamps[1] - stamps[0]) * g_tc.period_ns / 1e6;
			g_tc.sum_ms += ms, g_tc.max_ms = std::max(g_tc.max_ms, ms);
			++g_tc.samples;
		}
		++g_tc.read;
	}
}

void Log(int level, bool flare, bool have_gbuf) {
	const auto now = std::chrono::steady_clock::now();
	if (now - g_tc.last_log < std::chrono::seconds(10) || g_tc.mapped == nullptr) return;
	g_tc.last_log = now;
	std::array<uint32_t, StatsCount> cur {};
	std::memcpy(cur.data(), g_tc.mapped, sizeof(cur)); // (a few frames late: statistics only)
	std::array<uint32_t, StatsCount> d {};
	for (uint32_t i = 0; i < StatsCount; ++i) d[i] = cur[i] - g_tc.stat_last[i];
	g_tc.stat_last  = cur;
	const double fr = std::max<double>(d[0], 1.0);
	std::printf("TORCH: stats (level %d, flare %s, G-buffer %s) blobs %.2f a frame, lights %.2f a frame, lit pixels %.2f %%, "
	            "flares %.2f a frame, GPU %.3f ms (max %.3f)\n",
	            level, flare ? "on" : "off", have_gbuf ? "yes" : "no (depth normals)", d[1] / fr, d[2] / fr,
	            d[4] > 0 ? 100.0 * d[5] / d[4] : 0.0, d[3] / fr, g_tc.samples > 0 ? g_tc.sum_ms / g_tc.samples : 0.0, g_tc.max_ms);
	std::fflush(stdout);
	g_tc.sum_ms = 0.0, g_tc.max_ms = 0.0, g_tc.samples = 0;
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

// One push-descriptor set + dispatches of the given modes; out: the storage image written.
void Record(vk::CommandBuffer command, const UpscaleInputs& inputs, const RtReflGbuffer* gbuf, const Img& out, TorchPush push,
            std::initializer_list<int> modes) {
	const auto gen = vk::ImageLayout::eGeneral;
	vk::DescriptorImageInfo infos[Bindings] {};
	infos[0] = {g_tc.linear, inputs.color.view, inputs.color.layout};
	infos[1] = {g_tc.point, inputs.depth.view, inputs.depth.layout};
	for (uint32_t i = 0; i < 4; ++i)
		infos[2 + i] = gbuf != nullptr && gbuf->views[i] != nullptr
		                   ? vk::DescriptorImageInfo {g_tc.point, gbuf->views[i], vk::ImageLayout::eShaderReadOnlyOptimal}
		                   : vk::DescriptorImageInfo {g_tc.point, g_tc.dummy.view, gen};
	infos[6] = {nullptr, out.view, gen};
	infos[9] = inputs.motion.view != nullptr ? vk::DescriptorImageInfo {g_tc.point, inputs.motion.view, inputs.motion.layout}
	                                         : vk::DescriptorImageInfo {g_tc.point, g_tc.out.view, gen};
	const vk::DescriptorBufferInfo frame_info {vk::Buffer(g_tc.frame_buf), 0, VK_WHOLE_SIZE};
	const vk::DescriptorBufferInfo exposure_info =
	    g_tc.game_exposure ? vk::DescriptorBufferInfo {inputs.exposure_buffer, inputs.exposure_base, inputs.exposure_range} : frame_info;
	vk::WriteDescriptorSet writes[Bindings] {};
	for (uint32_t i = 0; i < Bindings; ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = DescType(i);
		if (i == 7) writes[i].pBufferInfo = &frame_info;
		else if (i == 8) writes[i].pBufferInfo = &exposure_info;
		else writes[i].pImageInfo = &infos[i];
	}
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_tc.pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_tc.layout, 0, Bindings, writes);
	const uint32_t gx = (inputs.color.extent.width + 7) / 8, gy = (inputs.color.extent.height + 7) / 8;
	for (int mode: modes) {
		push.misc[0] = mode;
		command.pushConstants(g_tc.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		if (mode == 0) command.dispatch(32, 18, 1);
		else if (mode == 1) command.dispatch(1, 1, 1);
		else command.dispatch(gx, gy, 1);
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	}
}

void FinishFrame(vk::CommandBuffer command) {
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eMemoryRead);
	vk::BufferCopy region {StatsWord * 4, 0, StatsCount * 4};
	command.copyBuffer(vk::Buffer(g_tc.frame_buf), vk::Buffer(g_tc.read_buf), 1, &region);
	Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
	        vk::AccessFlagBits2::eHostRead);
}

bool g_timing_open = false;
uint32_t g_timing_entry = 0;

void CloseTiming(vk::CommandBuffer command) {
	if (!g_timing_open) return;
	g_timing_open = false;
	command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_tc.pool, g_timing_entry * 2 + 1);
	++g_tc.written;
}

} // namespace

int TorchLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_TORCH");
		return value != nullptr ? std::clamp(std::atoi(value), 0, 3) : 0;
	}();
	return level;
}
bool TorchFlareWanted() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TORCH_FLARE");
		return value != nullptr && std::atoi(value) > 0;
	}();
	return on;
}

bool TorchApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int  level = TorchLevel();
	const bool flare = TorchFlareWanted();
	if (level == 0 && !flare) return false;
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr) return false;
	if (VideoOut::MoviePlaying()) {
		g_tc.history_valid = false;
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
	if (!EnsurePipeline(*graphics) || !EnsureImages(*graphics, color.extent)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("TORCH: off (could not create its pipeline or images)\n");
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
	static const int32_t normal_code = NormalEnv("KYTY_TORCH_NORMAL", 3 | (9 << 2) | 64);
	static const int32_t rough_code  = FieldEnv("KYTY_TORCH_ROUGH", 2 | (0 << 2) | (8 << 7));
	static const bool    row0_up     = [] {
		const char* y = std::getenv("KYTY_TORCH_FLIPY");
		return y == nullptr || *y == '\0' || *y != '0';
	}();
	static const bool debug = [] {
		const char* v = std::getenv("KYTY_TORCH_DEBUG");
		return v != nullptr && *v == '1';
	}();
	const bool  gbuf_ok  = have_gbuf && ((normal_code & 64) == 0 || camera.rows);
	const float strength = EnvFloat("KYTY_TORCH_STRENGTH", 1.0f, 0.0f, 3.0f);
	const float fstr     = EnvFloat("KYTY_TORCH_FLARE_STRENGTH", 1.0f, 0.0f, 3.0f);
	Log(level, flare, gbuf_ok);
	{
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("TORCH: on (light level %d, flare %s, %ux%u, strength %g, flare strength %g, G-buffer %s, exposure %s; "
			            "screen-space only, no hardware RT)\n",
			            level, flare ? "on" : "off", color.extent.width, color.extent.height, strength, fstr,
			            gbuf_ok ? "found" : "not yet (depth normals until Reflejos RT copies it)",
			            inputs.exposure_buffer != nullptr ? "the game's" : "1.0");
			std::fflush(stdout);
		}
	}
	const auto now = std::chrono::steady_clock::now();
	const float dt = g_tc.last_frame.time_since_epoch().count() == 0
	                     ? 0.016f
	                     : std::clamp(std::chrono::duration<float>(now - g_tc.last_frame).count(), 0.001f, 0.1f);
	if (g_tc.last_frame.time_since_epoch().count() != 0 && now - g_tc.last_frame > std::chrono::milliseconds(250)) g_tc.history_valid = false;
	g_tc.last_frame = now;

	const int   slots = level >= 3 ? 8 : level == 2 ? 4 : level == 1 ? 2 : 4; // (flare only: 4)
	const int   steps = level >= 3 ? 16 : level == 2 ? 12 : 8;
	const float lstr  = (level >= 3 ? 1.2f : level == 2 ? 1.0f : 0.8f) * strength;

	TorchPush push {};
	push.proj[0] = m[0], push.proj[1] = m[5], push.proj[2] = m[8], push.proj[3] = m[9];
	push.depth_map[0] = m[10], push.depth_map[1] = m[14], push.depth_map[2] = m[11], push.depth_map[3] = m[15];
	push.size[0] = static_cast<float>(color.extent.width), push.size[1] = static_cast<float>(color.extent.height);
	push.size[2] = lstr, push.size[3] = 1.2f * lstr; // (the added-luminance soft cap, exposed)
	push.light[0] = EnvFloat("KYTY_TORCH_RANGE", 1.0f, 0.3f, 3.0f), push.light[1] = 0.05f, push.light[2] = static_cast<float>(steps);
	push.light[3] = 130.0f;
	push.flare[0] = fstr, push.flare[1] = static_cast<float>(color.extent.width) / static_cast<float>(color.extent.height);
	push.motion[0] = inputs.motion_scale_x, push.motion[1] = inputs.motion_scale_y;
	push.motion[2] = (g_tc.history_valid && !inputs.reset && inputs.motion.view != nullptr) ? 1.0f : 0.0f;
	push.motion[3] = dt;
	g_tc.game_exposure = inputs.exposure_buffer != nullptr && inputs.exposure_range >= sizeof(float);
	push.misc[1] = static_cast<int32_t>(g_tc.frame & 0x7fffffff);
	push.misc[2] = (row0_up ? 1 : 0) | (gbuf_ok ? 2 : 0) | (inputs.motion.view != nullptr ? 4 : 0) | (debug ? 8 : 0);
	push.misc[3] = g_tc.game_exposure ? static_cast<int32_t>(inputs.exposure_index) : static_cast<int32_t>(ExposureIdx);
	push.dec[0] = normal_code, push.dec[1] = rough_code;

	g_timing_entry = static_cast<uint32_t>(g_tc.written % 8);
	g_timing_open  = g_tc.pool != nullptr && g_tc.written < g_tc.read + 8;
	if (g_timing_open) {
		command.resetQueryPool(g_tc.pool, g_timing_entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_tc.pool, g_timing_entry * 2);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite);
	if (!g_tc.ready) ToGeneral(command, g_tc.out.image), ToGeneral(command, g_tc.flare.image), g_tc.ready = true;
	if (!g_tc.dummy_ready) ToGeneral(command, g_tc.dummy.image), g_tc.dummy_ready = true;
	if (!g_tc.cleared || !g_tc.history_valid) {
		// (first use, or after a gap / a resize: the tracked lights start over)
		g_tc.cleared = true;
		if (g_tc.frame == 0) command.fillBuffer(vk::Buffer(g_tc.frame_buf), 0, VK_WHOLE_SIZE, 0);
		else command.fillBuffer(vk::Buffer(g_tc.frame_buf), 32 * 4, 128 * 4, 0); // (the lights only: the stats keep counting)
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer,
		        vk::AccessFlagBits2::eTransferWrite);
	}
	{
		std::array<float, HeaderWords> header {};
		std::copy(camera.view_rows.begin(), camera.view_rows.end(), header.begin());
		header[12] = 1.0f;                                              // (ExposureIdx)
		header[13] = EnvFloat("KYTY_TORCH_LIGHT", 1.5f, 0.3f, 100.0f); // the fire threshold (exposed luminance)
		header[14] = static_cast<float>(slots);
		command.updateBuffer(vk::Buffer(g_tc.frame_buf), 0, sizeof(header), header.data());
	}
	Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	g_tc.push = push;
	const RtReflGbuffer* gb = gbuf_ok ? &gbuf : nullptr;
	if (level > 0) Record(command, inputs, gb, g_tc.out, push, {0, 1, 2});
	else Record(command, inputs, gb, g_tc.out, push, {0, 1});
	g_tc.tracked_frame = g_tc.frame;
	g_tc.history_valid = true;
	if (!flare) {
		FinishFrame(command);
		CloseTiming(command);
		++g_tc.frame;
	}
	if (level == 0) return false;
	ReplaceColor(inputs, g_tc.out);
	return true;
}

bool TorchFlareApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	if (!TorchFlareWanted() || command == nullptr || g_tc.pipeline == nullptr) return false;
	if (g_tc.tracked_frame != g_tc.frame) return false; // (TorchApply did not track this frame)
	const auto& color = inputs.color;
	bool ok = color.view != nullptr && color.extent == g_tc.extent && inputs.depth.view != nullptr;
	if (ok) {
		Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
		        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
		TorchPush push = g_tc.push;
		push.misc[2] &= ~2; // (no G-buffer reads here)
		Record(command, inputs, nullptr, g_tc.flare, push, {3});
	}
	FinishFrame(command);
	CloseTiming(command);
	++g_tc.frame;
	if (!ok) return false;
	ReplaceColor(inputs, g_tc.flare);
	return true;
}

} // namespace Libs::Graphics::FrameGen
