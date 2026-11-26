// (rtq2 b, compilacion 2) KYTY_RTN: "Reflejos RT nuevos" - trace (a quarter of the half-size pixels a frame), temporal +
// spatial denoise, and the result blended into the game's specular IBL image right after the next frame's IBL dispatch
// (RtNuevoInject), so the game composites it with its own BRDF (rt-nuevo.h, rt-nuevo.comp).
// Built on rt-water.cpp's proven plumbing (same inputs from Upscale, same TLAS / camera from hwrt-scene, same world
// reconstruction, push descriptors, timestamps). Every descriptor is checked non-null before recording; any missing input
// skips the frame (logged once per reason) and records nothing.
#include "local/local-pass-cache.h"
#include "local/rt-nuevo.h"
#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/rtnuevo_spv.h"
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

constexpr uint32_t Bindings  = 15;
constexpr uint32_t ParamBytes = 256; // ivp 64 + vp 64 + pvp 64 + cam 16 + p0 16
constexpr uint32_t Slots     = 8;   // (Upscale and inject each take one a frame)
constexpr uint32_t StatWords = 10;

struct Push {
	float   size[4];
	int32_t misc[4];
	int32_t tsz[4];
	float   isz[4];
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
	bool                    failed = false, layouts_ready = false;
	vk::Extent2D            extent {}, trace {};
	Img                     raw, raw_aux, res, dbg;
	std::array<Img, 2>      hist, haux;
	uint32_t                cur = 0;            // the history written last (hist[cur], haux[cur])
	bool                    history_valid = false;
	std::array<double, 16>  prev_vp {};         // the last Upscale's camera (the history's frame)
	bool                    have_prev = false;
	bool                    res_ready = false;  // res / haux[cur] hold the last Upscale's result (for the next inject)
	uint64_t                inject_frames = 0, inject_skipped = 0;
	const char*             inject_why = nullptr;
	VkImage                 ibl_image = VK_NULL_HANDLE; // (the game's IBL image our storage view is of)
	vk::ImageView           ibl_view = nullptr;
	vk::Format              ibl_format = vk::Format::eUndefined;
	uint32_t                ibl_level = 0, ibl_layer = 0;
	std::array<std::pair<vk::ImageView, uint64_t>, 32> retired_views {};
	std::array<std::pair<Img, uint64_t>, 8> retired {};
	VkBuffer                param_buf = VK_NULL_HANDLE, stat_buf = VK_NULL_HANDLE;
	VmaAllocation           param_mem = nullptr, stat_mem = nullptr;
	uint8_t*                param_mapped = nullptr;
	volatile uint32_t*      stat_mapped  = nullptr;
	uint64_t                frame = 0;
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms = 0.0, max_ms = 0.0;
	uint32_t                samples = 0;
	std::array<uint32_t, StatWords> stat_last {};
	uint64_t                frames = 0, frames_ran = 0, skipped = 0;
	const char*             skip_why = nullptr;
	bool                    said_on = false;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
} g_n;

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
	b.dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader;
	b.dstAccessMask       = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
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

bool MakeImage(GraphicContext& graphics, vk::Extent2D extent, Img& img) {
	VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create.imageType     = VK_IMAGE_TYPE_2D;
	create.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
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
	view_info.format           = vk::Format::eR16G16B16A16Sfloat;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &img.view) != vk::Result::eSuccess) {
		img.view = nullptr;
		return false;
	}
	return true;
}
void DestroyImage(GraphicContext& graphics, Img& img) {
	if (img.view != nullptr) graphics.device.destroyImageView(img.view, nullptr);
	if (img.image != VK_NULL_HANDLE) vmaDestroyImage(graphics.allocator, img.image, img.memory);
	img = {};
}
void Retire(GraphicContext& graphics, Img& img) {
	if (img.image == VK_NULL_HANDLE) return;
	for (auto& slot: g_n.retired)
		if (slot.first.image == VK_NULL_HANDLE) {
			slot = {img, g_n.frame};
			img  = {};
			return;
		}
	(void)graphics.device.waitIdle(); // (all slots busy: only on repeated resizes)
	DestroyImage(graphics, img);
}
void FreeRetired(GraphicContext& graphics) {
	for (auto& slot: g_n.retired)
		if (slot.first.image != VK_NULL_HANDLE && g_n.frame >= slot.second + 16) DestroyImage(graphics, slot.first), slot = {};
}

vk::DescriptorType DescType(uint32_t i) {
	return i == 10 || i == 12 || i == 14 ? vk::DescriptorType::eStorageBuffer
	       : i == 11                     ? vk::DescriptorType::eAccelerationStructureKHR
	       : i == 6 || i == 7 || i == 13 ? vk::DescriptorType::eStorageImage
	                                     : vk::DescriptorType::eCombinedImageSampler;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_n.pipeline != nullptr) return true;
	if (g_n.failed) return false;
	g_n.failed        = true;
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
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_n.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Push)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_n.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_n.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(RTNUEVO_SPV);
	module_info.pCode       = RTNUEVO_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_n.layout;
	const auto created         = device.createComputePipelines(LocalPassCache::Get(device), 1, &pipeline_info, nullptr, &g_n.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_n.pipeline = nullptr;
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
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g_n.point : &g_n.linear) != vk::Result::eSuccess) return false;
	}
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = ParamBytes * Slots;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_n.param_buf, &g_n.param_mem, &mapped) != VK_SUCCESS) {
		g_n.param_buf = VK_NULL_HANDLE;
		return false;
	}
	g_n.param_mapped = static_cast<uint8_t*>(mapped.pMappedData);
	buffer.size      = StatWords * 4;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_n.stat_buf, &g_n.stat_mem, &mapped) != VK_SUCCESS) {
		g_n.stat_buf = VK_NULL_HANDLE;
		return false;
	}
	g_n.stat_mapped = static_cast<volatile uint32_t*>(mapped.pMappedData);
	for (uint32_t i = 0; i < StatWords; ++i) g_n.stat_mapped[i] = 0;
	(void)vmaFlushAllocation(graphics.allocator, g_n.stat_mem, 0, VK_WHOLE_SIZE);
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = 16 * 2; // (8 Upscale + 8 inject entries)
	if (device.createQueryPool(&query, nullptr, &g_n.pool) != vk::Result::eSuccess) g_n.pool = nullptr;
	g_n.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g_n.failed    = false;
	return true;
}

Img* AllImages(size_t i) {
	Img* all[] {&g_n.raw, &g_n.raw_aux, &g_n.res, &g_n.dbg, &g_n.hist[0], &g_n.hist[1], &g_n.haux[0], &g_n.haux[1]};
	return i < 8 ? all[i] : nullptr;
}
bool EnsureImages(GraphicContext& graphics, vk::Extent2D extent) {
	const vk::Extent2D trace {std::max(extent.width / 2, 1u), std::max(extent.height / 2, 1u)};
	bool all_there = true;
	for (size_t i = 0; i < 8; ++i) all_there = all_there && AllImages(i)->image != VK_NULL_HANDLE;
	if (all_there && g_n.extent == extent && g_n.trace == trace) return true;
	for (size_t i = 0; i < 8; ++i) Retire(graphics, *AllImages(i));
	g_n.extent = extent, g_n.trace = trace, g_n.layouts_ready = false, g_n.history_valid = false, g_n.res_ready = false;
	bool ok = true;
	for (size_t i = 0; i < 8; ++i) ok = ok && MakeImage(graphics, i == 3 ? extent : trace, *AllImages(i));
	if (!ok)
		for (size_t i = 0; i < 8; ++i) Retire(graphics, *AllImages(i));
	return ok;
}

// (two rings of 8 timestamp pairs: 0 the Upscale part, 1 the inject)
struct Ring {
	uint64_t written = 0, read = 0;
	double   sum_ms = 0.0, max_ms = 0.0;
	uint32_t samples = 0;
};
std::array<Ring, 2> g_ring {};
void ReadTiming(GraphicContext& graphics) {
	if (g_n.pool == nullptr) return;
	for (uint32_t r = 0; r < 2; ++r) {
		auto& ring = g_ring[r];
		while (ring.read + 4 <= ring.written) {
			const auto              entry = r * 8 + static_cast<uint32_t>(ring.read % 8);
			std::array<uint64_t, 2> stamps {};
			if (graphics.device.getQueryPoolResults(g_n.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
			                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
				break;
			if (stamps[1] > stamps[0]) {
				const double ms = static_cast<double>(stamps[1] - stamps[0]) * g_n.period_ns / 1e6;
				ring.sum_ms += ms, ring.max_ms = std::max(ring.max_ms, ms);
				++ring.samples;
			}
			++ring.read;
		}
	}
}
bool TimeBegin(vk::CommandBuffer command, uint32_t r) {
	auto& ring = g_ring[r];
	if (g_n.pool == nullptr || ring.written >= ring.read + 8) return false;
	const uint32_t entry = r * 8 + static_cast<uint32_t>(ring.written % 8);
	command.resetQueryPool(g_n.pool, entry * 2, 2);
	command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_n.pool, entry * 2);
	return true;
}
void TimeEnd(vk::CommandBuffer command, uint32_t r) {
	auto&          ring  = g_ring[r];
	const uint32_t entry = r * 8 + static_cast<uint32_t>(ring.written % 8);
	command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_n.pool, entry * 2 + 1);
	++ring.written;
}

bool Inverse4(const double* m, double* out) {
	double inv[16];
	inv[0]  = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
	inv[4]  = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
	inv[8]  = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
	inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
	inv[1]  = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
	inv[5]  = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
	inv[9]  = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
	inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
	inv[2]  = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
	inv[6]  = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
	inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
	inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
	inv[3]  = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
	inv[7]  = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
	inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
	inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
	const double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
	if (!(std::abs(det) > 1e-30) || !std::isfinite(det)) return false;
	for (int i = 0; i < 16; ++i) out[i] = inv[i] / det;
	return true;
}

bool Skip(const char* why) {
	++g_n.skipped, g_n.skip_why = why;
	g_n.history_valid = false, g_n.res_ready = false, g_n.have_prev = false;
	static const char* said = nullptr;
	if (said != why) {
		said = why;
		std::printf("[rtn] nada este frame (%s)\n", why);
		std::fflush(stdout);
	}
	return false;
}

void Log(GraphicContext& graphics) {
	const auto now = std::chrono::steady_clock::now();
	if (now - g_n.last_log < std::chrono::seconds(10)) return;
	const double secs = std::chrono::duration<double>(now - g_n.last_log).count();
	g_n.last_log      = now;
	(void)vmaInvalidateAllocation(graphics.allocator, g_n.stat_mem, 0, VK_WHOLE_SIZE);
	std::array<uint32_t, StatWords> cur {}, d {};
	for (uint32_t i = 0; i < StatWords; ++i) cur[i] = g_n.stat_mapped[i], d[i] = cur[i] - g_n.stat_last[i];
	g_n.stat_last = cur;
	const double px = std::max<double>(d[0], 1.0), traced = std::max<double>(d[2], 1.0), rays = std::max<double>(d[3], 1.0),
	             inj = std::max<double>(d[7], 1.0);
	const double up = g_ring[0].samples > 0 ? g_ring[0].sum_ms / g_ring[0].samples : 0.0;
	const double in = g_ring[1].samples > 0 ? g_ring[1].sum_ms / g_ring[1].samples : 0.0;
	std::printf("[rtn] %.0f s (nivel %d, traza %ux%u de %ux%u, 1/4 por frame): corrio %llu de %llu frames (ultimo salto: %s); "
	            "trazados %.1f %% de los pixeles de traza; de ellos en la mascara del juego %.1f %%, rayos %.1f %%, rugosos saltados %.1f %%, "
	            "normales giradas %.1f %%; aciertos visibles %.1f %% de los rayos; historia conservada %.1f %%; inyectado en el IBL del "
	            "juego %llu frames (saltado %llu: %s), peso medio %.3f; GPU traza+filtros %.3f ms (max %.3f) + inyeccion %.3f ms (max %.3f)\n",
	            secs, RtNuevoLevel(), g_n.trace.width, g_n.trace.height, g_n.extent.width, g_n.extent.height,
	            static_cast<unsigned long long>(g_n.frames_ran), static_cast<unsigned long long>(g_n.frames),
	            g_n.skip_why != nullptr ? g_n.skip_why : "ninguno", 100.0 * d[2] / px, 100.0 * d[1] / traced, 100.0 * d[3] / traced,
	            100.0 * d[6] / traced, 100.0 * d[5] / traced, 100.0 * d[4] / rays, 100.0 * d[9] / px,
	            static_cast<unsigned long long>(g_n.inject_frames), static_cast<unsigned long long>(g_n.inject_skipped),
	            g_n.inject_why != nullptr ? g_n.inject_why : "ninguno", d[7] > 0 ? d[8] / 1000.0 / inj : 0.0, up, g_ring[0].max_ms, in,
	            g_ring[1].max_ms);
	std::fflush(stdout);
	g_n.frames = g_n.frames_ran = g_n.inject_frames = g_n.inject_skipped = 0;
	for (auto& ring: g_ring) ring.sum_ms = ring.max_ms = 0.0, ring.samples = 0;
	g_n.skip_why = g_n.inject_why = nullptr;
}

} // namespace

int RtNuevoLevel() {
	static const int level = [] {
		const char* v = std::getenv("KYTY_RTN");
		const int   l = v != nullptr && v[0] >= '1' && v[0] <= '2' && v[1] == '\0' ? v[0] - '0' : 0;
		if (l > 0) {
			std::printf("[rtn] KYTY_RTN=%d: reflejos RT nuevos, compilacion 1 (solo traza%s)%s\n", l,
			            l == 2 ? " + vista de depuracion: azul mascara del juego, verde acierto, rojo fallo" : "; la imagen no cambia",
			            HwScene::Requested() ? "" : ": necesita KYTY_HWRT=1 (y KYTY_RT_HW_EXPERIMENTAL=1): apagado");
			std::fflush(stdout);
		}
		return HwScene::Requested() ? l : 0;
	}();
	return level;
}
bool RtNuevoWanted() { return RtNuevoLevel() > 0; }

namespace {
// Writes the frame's parameters into a free slot; returns its index.
uint32_t PutParams(GraphicContext& graphics, const double* ivp, const double* vp, const double* pvp, const double* cam, float flipy,
                   float dist, float bias, float strength) {
	static uint32_t next = 0;
	const uint32_t  slot = next++ % Slots;
	std::array<float, ParamBytes / 4> f {};
	for (int i = 0; i < 16; ++i) f[i] = static_cast<float>(ivp[i]);
	for (int i = 0; i < 16; ++i) f[16 + i] = static_cast<float>(vp[i]);
	for (int i = 0; i < 16; ++i) f[32 + i] = static_cast<float>(pvp[i]);
	f[48] = static_cast<float>(cam[0]), f[49] = static_cast<float>(cam[1]), f[50] = static_cast<float>(cam[2]), f[51] = 1.0f;
	f[52] = flipy > 0.5f ? -1.0f : 1.0f, f[53] = dist, f[54] = bias, f[55] = strength;
	f[56] = 1.0f; // (the exposure without the game's: index 56 of this buffer)
	std::memcpy(g_n.param_mapped + slot * ParamBytes, f.data(), ParamBytes);
	(void)vmaFlushAllocation(graphics.allocator, g_n.param_mem, slot * ParamBytes, ParamBytes);
	return slot;
}
// The camera: view-projection, its inverse, the eye. False: none / degenerate.
bool RtnCamera(double* vpd, double* ivp, double* cam) {
	std::array<float, 16> vp {};
	if (!HwScene::SceneCamera(vp)) return false;
	for (int i = 0; i < 16; ++i) vpd[i] = vp[i];
	if (!Inverse4(vpd, ivp)) return false;
	double eye[4] {};
	for (int i = 0; i < 4; ++i) eye[i] = ivp[i * 4 + 2];
	if (!(std::abs(eye[3]) > 1e-12)) return false;
	for (int i = 0; i < 3; ++i) cam[i] = eye[i] / eye[3];
	return true;
}
struct Cfg {
	float flipy, dist, bias, strength;
};
const Cfg& Config() {
	static const Cfg c {EnvFloat("KYTY_HWRT_WATER_FLIPY", 1.0f, 0.0f, 1.0f), // (the same convention as rt-water)
	                    EnvFloat("KYTY_RTN_DIST", 60.0f, 5.0f, 2000.0f), EnvFloat("KYTY_RTN_BIAS", 0.03f, 0.0f, 1.0f),
	                    EnvFloat("KYTY_RTN_STRENGTH", 1.0f, 0.0f, 1.0f)};
	return c;
}
struct Binds {
	std::array<vk::DescriptorImageInfo, Bindings> img {};
	vk::DescriptorBufferInfo param {}, stat {}, exposure {};
	vk::AccelerationStructureKHR tlas = nullptr;
};
void PushBinds(vk::CommandBuffer command, Binds& b) {
	vk::WriteDescriptorSetAccelerationStructureKHR as {};
	as.accelerationStructureCount = 1;
	as.pAccelerationStructures    = &b.tlas;
	vk::WriteDescriptorSet writes[Bindings] {};
	for (uint32_t i = 0; i < Bindings; ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = DescType(i);
		if (i == 10) writes[i].pBufferInfo = &b.param;
		else if (i == 12) writes[i].pBufferInfo = &b.stat;
		else if (i == 14) writes[i].pBufferInfo = &b.exposure;
		else if (i == 11) writes[i].pNext = &as;
		else writes[i].pImageInfo = &b.img[i];
	}
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_n.layout, 0, Bindings, writes);
}
void Dispatch(vk::CommandBuffer command, Push& push, int mode, uint32_t w, uint32_t h) {
	push.misc[0] = mode;
	command.pushConstants(g_n.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch((w + 7) / 8, (h + 7) / 8, 1);
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
}
} // namespace

bool RtNuevoApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int level = RtNuevoLevel();
	if (level == 0) return false;
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr) return false;
	++g_n.frames;
	if (g_n.pipeline != nullptr) Log(*graphics);
	if (!HwScene::Active()) return Skip("KYTY_HWRT inactivo: el dispositivo no tiene ray queries");
	if (VideoOut::MoviePlaying()) return Skip("un video");
	const auto& color = inputs.color;
	const auto& depth = inputs.depth;
	if (color.view == nullptr || depth.view == nullptr || color.extent.width < 16 || color.extent.height < 16 || depth.extent != color.extent)
		return Skip("sin color / profundidad");
	if (depth.layout == vk::ImageLayout::eUndefined || color.layout == vk::ImageLayout::eUndefined) return Skip("color / profundidad sin layout");
	RtReflGbuffer gbuf;
	if (!RtReflGetGbuffer(gbuf) || gbuf.extent != color.extent || gbuf.views[1] == nullptr || gbuf.views[2] == nullptr ||
	    gbuf.views[3] == nullptr)
		return Skip("sin copia del G-buffer de este frame (o de otro tamano)");
	vk::ImageView tiles_view = nullptr;
	vk::Extent2D  tiles_extent {};
	if (!RtNuevoGetTiles(tiles_view, tiles_extent) || tiles_view == nullptr || tiles_extent.width == 0 || tiles_extent.height == 0)
		return Skip("sin la mascara de tiles del SSR del juego (el IBL no corrio)");
	const uint32_t tile = (color.extent.width + tiles_extent.width - 1) / tiles_extent.width;
	if (tile < 8 || tile > 32 || (color.extent.height + tile - 1) / tile > tiles_extent.height + 1) return Skip("la mascara de tiles no encaja con el tamano de render");
	bool built = false;
	vk::AccelerationStructureKHR tlas = HwScene::SceneTlas(&built);
	if (tlas == nullptr) return Skip("sin TLAS de la escena");
	double vpd[16], ivp[16], cam[3];
	if (!RtnCamera(vpd, ivp, cam)) return Skip("sin camara (o degenerada)");
	if (!EnsurePipeline(*graphics) || !EnsureImages(*graphics, color.extent)) return Skip("no se pudo crear su pipeline o sus imagenes");
	FreeRetired(*graphics);
	ReadTiming(*graphics);
	const Cfg& cfg = Config();
	++g_n.frames_ran;
	if (!g_n.said_on) {
		g_n.said_on = true;
		std::printf("[rtn] on (compilacion 2): traza %ux%u (1/4 de los pixeles por frame) de %ux%u, tiles %ux%u (%u px), rayos <= %g m, "
		            "fuerza %g; se compone en el IBL del juego tras su dispatch del frame siguiente\n",
		            g_n.trace.width, g_n.trace.height, color.extent.width, color.extent.height, tiles_extent.width, tiles_extent.height,
		            tile, cfg.dist, cfg.strength);
		std::fflush(stdout);
	}
	const bool history = g_n.history_valid && g_n.have_prev && !inputs.reset;
	const uint32_t slot = PutParams(*graphics, ivp, vpd, g_n.have_prev ? g_n.prev_vp.data() : vpd, cam, cfg.flipy, cfg.dist, cfg.bias,
	                                cfg.strength);
	for (int i = 0; i < 16; ++i) g_n.prev_vp[i] = vpd[i];
	g_n.have_prev = true;

	Push push {};
	push.size[0] = static_cast<float>(color.extent.width), push.size[1] = static_cast<float>(color.extent.height);
	push.size[2] = static_cast<float>(g_n.trace.width), push.size[3] = static_cast<float>(g_n.trace.height);
	push.misc[1] = static_cast<int32_t>(g_n.frame & 0x7fffffff);
	push.misc[2] = static_cast<int32_t>(tile);
	push.misc[3] = (history ? 1 : 0) | 2;
	push.tsz[0]  = static_cast<int32_t>(tiles_extent.width), push.tsz[1] = static_cast<int32_t>(tiles_extent.height);
	const bool game_exposure = inputs.exposure_buffer != nullptr && inputs.exposure_range >= sizeof(float);
	push.tsz[2] = game_exposure ? static_cast<int32_t>(inputs.exposure_index) : 56;

	const bool timed = TimeBegin(command, 0);
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	if (!g_n.layouts_ready) {
		for (size_t i = 0; i < 8; ++i) ToGeneral(command, AllImages(i)->image);
		g_n.layouts_ready = true;
		push.misc[3] &= ~1;
	}
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_n.pipeline);
	const auto gen = vk::ImageLayout::eGeneral;
	const auto ro  = vk::ImageLayout::eShaderReadOnlyOptimal;
	const uint32_t cur = g_n.cur ^ 1u, prev = g_n.cur;
	Binds b;
	b.param    = {vk::Buffer(g_n.param_buf), vk::DeviceSize {slot} * ParamBytes, ParamBytes};
	b.stat     = {vk::Buffer(g_n.stat_buf), 0, StatWords * 4};
	b.exposure = game_exposure ? vk::DescriptorBufferInfo {inputs.exposure_buffer, inputs.exposure_base, inputs.exposure_range} : b.param;
	b.tlas     = tlas;
	b.img[0]   = {g_n.point, depth.view, depth.layout};
	b.img[1]   = {g_n.linear, color.view, color.layout};
	b.img[2]   = {g_n.point, gbuf.views[1], ro};
	b.img[3]   = {g_n.point, gbuf.views[2], ro};
	b.img[4]   = {g_n.point, gbuf.views[3], ro};
	b.img[5]   = {g_n.point, tiles_view, ro};
	// mode 0: trace -> raw / raw_aux (src_a / src_b / ibl unused: images it never writes)
	b.img[6]  = {nullptr, g_n.raw.view, gen};
	b.img[7]  = {nullptr, g_n.raw_aux.view, gen};
	b.img[8]  = {g_n.linear, g_n.res.view, gen};
	b.img[9]  = {g_n.point, g_n.haux[prev].view, gen};
	b.img[13] = {nullptr, g_n.dbg.view, gen};
	PushBinds(command, b);
	Dispatch(command, push, 0, g_n.trace.width, g_n.trace.height);
	// mode 1: temporal: raw + the previous history (colour linear at 1, its aux at 13) -> hist[cur] / haux[cur]
	b.img[1]  = {g_n.linear, g_n.hist[prev].view, gen};
	b.img[6]  = {nullptr, g_n.hist[cur].view, gen};
	b.img[7]  = {nullptr, g_n.haux[cur].view, gen};
	b.img[8]  = {g_n.linear, g_n.raw.view, gen};
	b.img[9]  = {g_n.point, g_n.raw_aux.view, gen};
	b.img[13] = {nullptr, g_n.haux[prev].view, gen};
	PushBinds(command, b);
	Dispatch(command, push, 1, g_n.trace.width, g_n.trace.height);
	// mode 2: spatial: hist[cur] / haux[cur] -> res (out_b: raw_aux, not written)
	b.img[1]  = {g_n.linear, color.view, color.layout};
	b.img[6]  = {nullptr, g_n.res.view, gen};
	b.img[7]  = {nullptr, g_n.raw_aux.view, gen};
	b.img[8]  = {g_n.linear, g_n.hist[cur].view, gen};
	b.img[9]  = {g_n.point, g_n.haux[cur].view, gen};
	b.img[13] = {nullptr, g_n.dbg.view, gen};
	PushBinds(command, b);
	Dispatch(command, push, 2, g_n.trace.width, g_n.trace.height);
	if (level == 2) {
		// mode 3: debug overlay -> dbg (src_a: res)
		b.img[6] = {nullptr, g_n.dbg.view, gen};
		b.img[7] = {nullptr, g_n.raw_aux.view, gen};
		b.img[8] = {g_n.linear, g_n.res.view, gen};
		b.img[9] = {g_n.point, g_n.haux[cur].view, gen};
		b.img[13] = {nullptr, g_n.raw.view, gen};
		PushBinds(command, b);
		Dispatch(command, push, 3, color.extent.width, color.extent.height);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead);
	if (timed) TimeEnd(command, 0);
	++g_n.frame;
	g_n.cur           = cur;
	g_n.history_valid = true;
	g_n.res_ready     = true;
	if (level != 2) return true;
	inputs.color.image  = vk::Image(g_n.dbg.image);
	inputs.color.view   = g_n.dbg.view;
	inputs.color.layout = gen;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc |
	                      vk::ImageUsageFlagBits::eStorage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}

bool RtNuevoInjectReady() { return RtNuevoLevel() > 0 && g_n.res_ready && g_n.pipeline != nullptr; }

bool RtNuevoInject(GraphicContext& graphics, vk::CommandBuffer command, vk::Image image, vk::Format format, vk::ImageUsageFlags usage,
                   vk::Extent2D extent, uint32_t level, uint32_t layer) {
	const auto skip = [&](const char* why) {
		++g_n.inject_skipped, g_n.inject_why = why;
		return false;
	};
	if (!RtNuevoInjectReady()) return skip("sin resultado del frame anterior");
	if (image == nullptr || extent.width == 0 || extent.height == 0) return skip("sin imagen del IBL");
	if (format != vk::Format::eR16G16B16A16Sfloat) return skip("el IBL no es RGBA16F");
	if (!(usage & vk::ImageUsageFlagBits::eStorage)) return skip("el IBL no admite escritura (storage)");
	RtReflGbuffer gbuf;
	if (!RtReflGetGbuffer(gbuf) || gbuf.extent != g_n.extent || gbuf.depth == nullptr || gbuf.views[1] == nullptr ||
	    gbuf.views[2] == nullptr || gbuf.views[3] == nullptr)
		return skip("sin copia del G-buffer / profundidad de este frame");
	bool built = false;
	vk::AccelerationStructureKHR tlas = HwScene::SceneTlas(&built);
	if (tlas == nullptr) return skip("sin TLAS");
	double vpd[16], ivp[16], cam[3];
	if (!RtnCamera(vpd, ivp, cam)) return skip("sin camara");
	// (our storage view of the game's IBL image: a new one every inject - the cache may have destroyed and recreated the image
	// under the same handle -, each destroyed 16 frames later, when no command buffer can still use it)
	for (auto& r: g_n.retired_views)
		if (r.first != nullptr && g_n.frame >= r.second + 16) graphics.device.destroyImageView(r.first, nullptr), r = {};
	if (g_n.ibl_view != nullptr) {
		bool put = false;
		for (auto& r: g_n.retired_views)
			if (r.first == nullptr) {
				r   = {g_n.ibl_view, g_n.frame};
				put = true;
				break;
			}
		if (!put) {
			(void)graphics.device.waitIdle();
			graphics.device.destroyImageView(g_n.ibl_view, nullptr);
		}
		g_n.ibl_view = nullptr;
	}
	{
		vk::ImageViewUsageCreateInfo view_usage {};
		view_usage.usage = vk::ImageUsageFlagBits::eStorage;
		vk::ImageViewCreateInfo view_info {};
		view_info.pNext            = &view_usage;
		view_info.image            = image;
		view_info.viewType         = vk::ImageViewType::e2D;
		view_info.format           = format;
		view_info.subresourceRange = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, level, 1, layer, 1};
		if (graphics.device.createImageView(&view_info, nullptr, &g_n.ibl_view) != vk::Result::eSuccess) {
			g_n.ibl_view = nullptr;
			return skip("no se pudo crear la vista del IBL");
		}
	}
	const Cfg&     cfg  = Config();
	// (pvp: the camera of the Upscale that made the result, the last one)
	const uint32_t slot = PutParams(graphics, ivp, vpd, g_n.prev_vp.data(), cam, cfg.flipy, cfg.dist, cfg.bias, cfg.strength);
	Push push {};
	push.size[0] = static_cast<float>(g_n.extent.width), push.size[1] = static_cast<float>(g_n.extent.height);
	push.size[2] = static_cast<float>(g_n.trace.width), push.size[3] = static_cast<float>(g_n.trace.height);
	push.isz[0]  = static_cast<float>(extent.width), push.isz[1] = static_cast<float>(extent.height);
	push.misc[1] = static_cast<int32_t>(g_n.frame & 0x7fffffff);
	push.misc[3] = 0;
	const bool timed = TimeBegin(command, 1);
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_n.pipeline);
	const auto gen = vk::ImageLayout::eGeneral;
	const auto ro  = vk::ImageLayout::eShaderReadOnlyOptimal;
	Binds b;
	b.param    = {vk::Buffer(g_n.param_buf), vk::DeviceSize {slot} * ParamBytes, ParamBytes};
	b.stat     = {vk::Buffer(g_n.stat_buf), 0, StatWords * 4};
	b.exposure = b.param;
	b.tlas     = tlas;
	b.img[0]   = {g_n.point, gbuf.depth, gbuf.depth_layout};
	b.img[1]   = {g_n.linear, g_n.dbg.view, gen};
	b.img[2]   = {g_n.point, gbuf.views[1], ro};
	b.img[3]   = {g_n.point, gbuf.views[2], ro};
	b.img[4]   = {g_n.point, gbuf.views[3], ro};
	b.img[5]   = {g_n.point, gbuf.views[2], ro}; // (an R32_UINT view for the unused tile binding)
	b.img[6]   = {nullptr, g_n.raw.view, gen};
	b.img[7]   = {nullptr, g_n.raw_aux.view, gen};
	b.img[8]   = {g_n.linear, g_n.res.view, gen};
	b.img[9]   = {g_n.point, g_n.haux[g_n.cur].view, gen};
	b.img[13]  = {nullptr, g_n.ibl_view, gen};
	PushBinds(command, b);
	Dispatch(command, push, 4, extent.width, extent.height);
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	if (timed) TimeEnd(command, 1);
	++g_n.inject_frames;
	g_n.res_ready = false; // (one inject per result: a second IBL run in the same frame leaves it)
	return true;
}

} // namespace Libs::Graphics::FrameGen
