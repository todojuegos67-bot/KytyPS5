// KYTY_RT_FLICKER: the ray-traced passes' flicker self-check (rt-flicker.h, rt-flicker.comp).
#include "local/rt-flicker.h"

#include "gpu_tiler_shaders/rtflicker_spv.h"
#include "local/hwrt-scene.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::FrameGen::RtFlicker {
namespace {

struct Push {
	int32_t size[4];
	float   par[4];
};
static_assert(sizeof(Push) == 32);
constexpr uint32_t MaxIndex = 40; // (rt3fx) st[40 + pass]: the largest jump this 10 s, x 1e-4 of the light
static_assert(Count * 4 <= MaxIndex && MaxIndex + Count <= 64);

struct Grid {
	VkImage       image  = VK_NULL_HANDLE;
	VmaAllocation memory = nullptr;
	vk::ImageView view   = nullptr;
	vk::Extent2D  extent {};
	bool          fresh  = true; // (its first use: cleared, no history)
	uint64_t      last_frame = 0;
};

struct {
	bool                  failed = false;
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout    layout   = nullptr;
	vk::Pipeline          pipeline = nullptr;
	vk::Sampler           point    = nullptr;
	VkBuffer              stat_buf = VK_NULL_HANDLE;
	VmaAllocation         stat_mem = nullptr;
	volatile uint32_t*    stat     = nullptr;
	std::array<uint32_t, Count * 4> last {};
	std::array<Grid, Count> grids {};
	std::array<std::pair<Grid, uint64_t>, 8> retired {};
	std::array<float, 16> prev_vp {};
	bool                  have_prev = false, still = false;
	uint64_t              frame = 0, frames = 0, still_frames = 0;
	std::array<uint64_t, Count> ran {};
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
} g_f;

bool Enabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RT_FLICKER");
		return !(v != nullptr && v[0] == '0');
	}();
	return on;
}
float Threshold() {
	static const float t = [] {
		const char* v = std::getenv("KYTY_RT_FLICKER_PCT");
		const float p = v != nullptr && *v != '\0' ? static_cast<float>(std::atof(v)) : 10.0f;
		return std::clamp(std::isfinite(p) ? p : 10.0f, 1.0f, 100.0f) / 100.0f;
	}();
	return t;
}
const char* Name(uint32_t p) {
	static const char* names[] {"particles light", "sun shadows", "lamp/torch shadows", "reflections", "ao", "volumetric", "metal"};
	return p < Count ? names[p] : "?";
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_f.pipeline != nullptr) return true;
	if (g_f.failed) return false;
	g_f.failed        = true;
	const auto device = graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, 4> bindings {};
	for (uint32_t i = 0; i < 4; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = i < 2 ? vk::DescriptorType::eCombinedImageSampler
		                              : i == 2 ? vk::DescriptorType::eStorageImage
		                                       : vk::DescriptorType::eStorageBuffer;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = static_cast<uint32_t>(bindings.size());
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_f.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Push)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_f.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_f.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(RTFLICKER_SPV);
	module_info.pCode       = RTFLICKER_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_f.layout;
	const auto created         = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_f.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) return g_f.pipeline = nullptr, false;
	vk::SamplerCreateInfo sampler {};
	sampler.magFilter    = vk::Filter::eNearest;
	sampler.minFilter    = vk::Filter::eNearest;
	sampler.mipmapMode   = vk::SamplerMipmapMode::eNearest;
	sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	sampler.maxLod       = VK_LOD_CLAMP_NONE;
	if (device.createSampler(&sampler, nullptr, &g_f.point) != vk::Result::eSuccess) return false;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = 256;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_f.stat_buf, &g_f.stat_mem, &mapped) != VK_SUCCESS) {
		g_f.stat_buf = VK_NULL_HANDLE;
		return false;
	}
	g_f.stat = static_cast<volatile uint32_t*>(mapped.pMappedData);
	for (uint32_t i = 0; i < 64; ++i) g_f.stat[i] = 0;
	(void)vmaFlushAllocation(graphics.allocator, g_f.stat_mem, 0, VK_WHOLE_SIZE);
	g_f.failed = false;
	std::printf("RTFLICKER: self-check on (KYTY_RT_FLICKER=0 turns it off): per RT pass, %% of the points it touches whose "
	            "contribution changed > %.0f %% of the pixel's light frame to frame, camera still; target ~0\n",
	            static_cast<double>(Threshold() * 100.0f));
	std::fflush(stdout);
	return true;
}

void Destroy(GraphicContext& graphics, Grid& grid) {
	if (grid.view != nullptr) graphics.device.destroyImageView(grid.view, nullptr);
	if (grid.image != VK_NULL_HANDLE) vmaDestroyImage(graphics.allocator, grid.image, grid.memory);
	grid = {};
}

bool EnsureGrid(GraphicContext& graphics, Grid& grid, vk::Extent2D extent) {
	if (grid.image != VK_NULL_HANDLE && grid.extent == extent) return true;
	if (grid.image != VK_NULL_HANDLE) {
		bool parked = false;
		for (auto& slot: g_f.retired)
			if (slot.first.image == VK_NULL_HANDLE) {
				slot = {grid, g_f.frame}, grid = {}, parked = true;
				break;
			}
		if (!parked) graphics.device.waitIdle(), Destroy(graphics, grid);
	}
	VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create.imageType     = VK_IMAGE_TYPE_2D;
	create.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.samples       = VK_SAMPLE_COUNT_1_BIT;
	create.tiling        = VK_IMAGE_TILING_OPTIMAL;
	create.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateImage(graphics.allocator, &create, &allocate, &grid.image, &grid.memory, nullptr) != VK_SUCCESS) {
		grid.image = VK_NULL_HANDLE;
		return false;
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = vk::Image(grid.image);
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = vk::Format::eR16G16B16A16Sfloat;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &grid.view) != vk::Result::eSuccess) {
		grid.view = nullptr;
		Destroy(graphics, grid);
		return false;
	}
	grid.extent = extent;
	grid.fresh  = true;
	return true;
}

void MaybeLog() {
	const auto now = std::chrono::steady_clock::now();
	if (now - g_f.last_log < std::chrono::seconds(10) || g_f.stat == nullptr) return;
	g_f.last_log = now;
	std::array<uint32_t, Count * 4> cur {}, d {};
	for (uint32_t i = 0; i < Count * 4; ++i) cur[i] = g_f.stat[i], d[i] = cur[i] - g_f.last[i];
	g_f.last = cur;
	char line[2048];
	int  n = std::snprintf(line, sizeof(line), "RTFLICKER: 10 s: camera still %llu of %llu frames;",
	                       static_cast<unsigned long long>(g_f.still_frames), static_cast<unsigned long long>(g_f.frames));
	for (uint32_t p = 0; p < Count && n > 0 && n < static_cast<int>(sizeof(line)); ++p) {
		if (g_f.ran[p] == 0) {
			n += std::snprintf(line + n, sizeof(line) - n, " %s off;", Name(p));
			continue;
		}
		const double touched = static_cast<double>(std::max(d[p * 4 + 1], 1u));
		const uint32_t jump  = g_f.stat[MaxIndex + p];
		g_f.stat[MaxIndex + p] = 0; // (rt3fx: per 10 s; a frame in flight may land in the next window)
		n += std::snprintf(line + n, sizeof(line) - n, " %s: changed > %.0f %% %.3f %% (> %.0f %% %.3f %%) of %u touched points, max jump %.1f %% (%u probed, %llu frames);",
		                   Name(p), static_cast<double>(Threshold() * 100.0f), 100.0 * d[p * 4 + 2] / touched,
		                   static_cast<double>(Threshold() * 200.0f), 100.0 * d[p * 4 + 3] / touched, d[p * 4 + 1], jump / 100.0, d[p * 4],
		                   static_cast<unsigned long long>(g_f.ran[p]));
	}
	std::printf("%s\n", line);
	std::fflush(stdout);
	g_f.frames = g_f.still_frames = 0;
	g_f.ran    = {};
}

} // namespace

void BeginFrame(const UpscaleInputs& inputs) {
	if (!Enabled()) return;
	++g_f.frame;
	++g_f.frames;
	std::array<float, 16> vp {};
	const bool have = HwScene::SceneCamera(vp);
	bool       still = false;
	if (have && g_f.have_prev && !inputs.reset) {
		float scale = 1.0f, diff = 0.0f;
		for (int i = 0; i < 16; ++i) scale = std::max(scale, std::abs(vp[i])), diff = std::max(diff, std::abs(vp[i] - g_f.prev_vp[i]));
		still = diff <= 4e-3f * scale; // (the TAA jitter moves the projection by ~1e-3)
	}
	g_f.prev_vp   = vp;
	g_f.have_prev = have;
	g_f.still     = still;
	if (still) ++g_f.still_frames;
	MaybeLog();
}

void Probe(vk::CommandBuffer command, GraphicContext* graphics, Pass pass, const UpscaleImage& before, const UpscaleImage& after) {
	if (!Enabled() || graphics == nullptr || command == nullptr || pass >= Count) return;
	if (before.view == nullptr || after.view == nullptr || before.view == after.view) return;
	if (before.extent != after.extent || after.extent.width < 8 || after.extent.height < 8) return;
	if (!EnsurePipeline(*graphics)) return;
	for (auto& slot: g_f.retired)
		if (slot.first.image != VK_NULL_HANDLE && g_f.frame >= slot.second + 16) Destroy(*graphics, slot.first), slot = {};
	auto&              grid = g_f.grids[pass];
	const vk::Extent2D ge {(after.extent.width + 3) / 4, (after.extent.height + 3) / 4};
	if (!EnsureGrid(*graphics, grid, ge)) return;
	const bool history = !grid.fresh && grid.last_frame + 1 == g_f.frame; // (the pass ran last frame too)
	if (grid.fresh) {
		vk::ImageMemoryBarrier2 b {};
		b.srcStageMask     = vk::PipelineStageFlagBits2::eNone;
		b.dstStageMask     = vk::PipelineStageFlagBits2::eClear;
		b.dstAccessMask    = vk::AccessFlagBits2::eTransferWrite;
		b.oldLayout        = vk::ImageLayout::eUndefined;
		b.newLayout        = vk::ImageLayout::eGeneral;
		b.image            = vk::Image(grid.image);
		b.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		vk::DependencyInfo dep {};
		dep.imageMemoryBarrierCount = 1;
		dep.pImageMemoryBarriers    = &b;
		command.pipelineBarrier2(dep);
		const vk::ClearColorValue       zero {std::array<float, 4> {0.0f, 0.0f, 0.0f, 0.0f}};
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		command.clearColorImage(vk::Image(grid.image), vk::ImageLayout::eGeneral, &zero, 1, &range);
		grid.fresh = false;
	}
	grid.last_frame = g_f.frame;
	++g_f.ran[pass];
	// The pass's writes (and the clear) -> these reads; this frame's grid write after last frame's.
	vk::MemoryBarrier2 mb {};
	mb.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eClear;
	mb.srcAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite;
	mb.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	mb.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	vk::DependencyInfo dep {};
	dep.memoryBarrierCount = 1;
	dep.pMemoryBarriers    = &mb;
	command.pipelineBarrier2(dep);
	vk::DescriptorImageInfo  b_info {g_f.point, before.view, before.layout};
	vk::DescriptorImageInfo  a_info {g_f.point, after.view, after.layout};
	vk::DescriptorImageInfo  g_info {nullptr, grid.view, vk::ImageLayout::eGeneral};
	vk::DescriptorBufferInfo s_info {vk::Buffer(g_f.stat_buf), 0, VK_WHOLE_SIZE};
	std::array<vk::WriteDescriptorSet, 4> writes {};
	for (uint32_t i = 0; i < 4; ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
	}
	writes[0].descriptorType = writes[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
	writes[0].pImageInfo     = &b_info;
	writes[1].pImageInfo     = &a_info;
	writes[2].descriptorType = vk::DescriptorType::eStorageImage;
	writes[2].pImageInfo     = &g_info;
	writes[3].descriptorType = vk::DescriptorType::eStorageBuffer;
	writes[3].pBufferInfo    = &s_info;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_f.pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_f.layout, 0, static_cast<uint32_t>(writes.size()), writes.data());
	Push push {};
	push.size[0] = static_cast<int32_t>(after.extent.width);
	push.size[1] = static_cast<int32_t>(after.extent.height);
	push.size[2] = static_cast<int32_t>(ge.width);
	push.size[3] = static_cast<int32_t>(ge.height);
	push.par[0]  = Threshold();
	push.par[1]  = g_f.still ? 1.0f : 0.0f;
	push.par[2]  = history ? 1.0f : 0.0f;
	push.par[3]  = static_cast<float>(pass);
	command.pushConstants(g_f.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch((ge.width + 7) / 8, (ge.height + 7) / 8, 1);
	// (the grid's write -> the next frame's read; the color reads stay ahead of the next pass's writes)
	mb.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	mb.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	mb.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	mb.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	command.pipelineBarrier2(dep);
}

} // namespace Libs::Graphics::FrameGen::RtFlicker
