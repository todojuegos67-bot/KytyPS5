// KYTY_HWRT_VOLSHADOW=0|1|2|3: ray-traced shadowing of the god rays' view-ray march (volshadow.h). Called by GodraysApply
// (godrays.cpp) before its own dispatches, on the same command stream, after the scene TLAS was built this frame
// (hwrt-scene.cpp, after the game's TAA) with a barrier to compute ray queries.
//
// Per frame, three quarter-resolution dispatches on godrays' quarter grid:
//   trace     per texel: 6 / 10 / 16 jittered points on the camera ray (to the surface, or 60 m: godrays' own march
//             length), one TLAS ray each toward the sun (300 m, first hit) -> the lit fraction
//   temporal  reprojected history (depth-tested), up to 24 frames still / 8 moving, 3x3 variance clamp
//   spatial   depth-aware 3x3 bilateral; written as mix(1, fraction, fade): the fade ramps in over 0.5 s once a TLAS
//             exists and out over 0.3 s when the sun weight drops, so godrays never pops between its two paths
// godrays.comp then uses min(depth-tested lit length, fraction x march length): only light the approved volumetric
// already had can be removed, never added (no indoor haze: godrays' openness gate stays on top).
// The sun's world direction: godrays' view-space direction through view_to_clip and the scene camera's inverse
// view-projection (the TLAS's world), logged once with the RTSHADOW one to compare. Every 10 s: "VOLSHADOW: stats" with
// rays, occluded share, the TLAS alignment probe and the GPU ms.
// Options: KYTY_HWRT_VOLSHADOW_LEN (sun ray length, 300 m).
#include "local/volshadow.h"

#include "gpu_tiler_shaders/volshadow_spv.h"
#include "graphics/presentation/videoOut.h"
#include "local/hwrt-scene.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace Libs::Graphics::FrameGen {
extern std::mutex            g_last_vp_mutex;
extern std::array<float, 16> g_last_vp;
extern bool                  g_have_last_vp;
namespace {

constexpr uint32_t Bindings  = 8;
constexpr uint32_t SlotBytes = 256;
constexpr uint32_t Slots     = 4;

struct VsPush {
	float   depth_map[4];
	float   size[4];
	float   motion[4];
	int32_t misc[4];
};
static_assert(sizeof(VsPush) == 64);

struct Img {
	VkImage       image  = VK_NULL_HANDLE;
	VmaAllocation memory = nullptr;
	vk::ImageView view   = nullptr;
};

struct {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             point      = nullptr;
	bool                    failed     = false;
	vk::Extent2D            extent {};
	Img                     raw, out;
	std::array<Img, 2>      hist;
	bool                    layouts_ready = false, history_valid = false;
	std::array<std::pair<Img, uint64_t>, 16> retired {};
	VkBuffer                scene_buf = VK_NULL_HANDLE, stat_buf = VK_NULL_HANDLE;
	VmaAllocation           scene_mem = nullptr, stat_mem = nullptr;
	uint8_t*                scene_mapped = nullptr;
	volatile uint32_t*      stat_mapped  = nullptr;
	uint64_t                frame = 0;
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms = 0.0, max_ms = 0.0;
	uint32_t                samples = 0;
	std::array<uint32_t, 4> stat_last {};
	float                   fade = 0.0f;
	uint64_t                frames = 0, stale_tlas = 0, skipped = 0;
	const char*             skip_why = nullptr;
	bool                    said_on = false;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now(), last_frame {};
} g_vs;

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
	create.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
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
	for (auto& slot: g_vs.retired)
		if (slot.first.image == VK_NULL_HANDLE) {
			slot = {img, g_vs.frame};
			img  = {};
			return;
		}
	graphics.device.waitIdle(); // (all slots busy: only on repeated resizes)
	DestroyImage(graphics, img);
}
void FreeRetired(GraphicContext& graphics) {
	for (auto& slot: g_vs.retired)
		if (slot.first.image != VK_NULL_HANDLE && g_vs.frame >= slot.second + 16) DestroyImage(graphics, slot.first), slot = {};
}

vk::DescriptorType DescType(uint32_t i) {
	return i == 5 || i == 7 ? vk::DescriptorType::eStorageBuffer
	       : i == 6         ? vk::DescriptorType::eAccelerationStructureKHR
	       : i == 4         ? vk::DescriptorType::eStorageImage
	                        : vk::DescriptorType::eCombinedImageSampler;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_vs.pipeline != nullptr) return true;
	if (g_vs.failed) return false;
	g_vs.failed       = true;
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
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_vs.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(VsPush)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_vs.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_vs.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(VOLSHADOW_SPV);
	module_info.pCode       = VOLSHADOW_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_vs.layout;
	const auto created         = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_vs.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_vs.pipeline = nullptr;
		return false;
	}
	vk::SamplerCreateInfo sampler {};
	sampler.magFilter    = vk::Filter::eNearest;
	sampler.minFilter    = vk::Filter::eNearest;
	sampler.mipmapMode   = vk::SamplerMipmapMode::eNearest;
	sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	sampler.maxLod       = VK_LOD_CLAMP_NONE;
	if (device.createSampler(&sampler, nullptr, &g_vs.point) != vk::Result::eSuccess) return false;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = SlotBytes * Slots;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_vs.scene_buf, &g_vs.scene_mem, &mapped) != VK_SUCCESS) {
		g_vs.scene_buf = VK_NULL_HANDLE;
		return false;
	}
	g_vs.scene_mapped = static_cast<uint8_t*>(mapped.pMappedData);
	buffer.size       = 64;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_vs.stat_buf, &g_vs.stat_mem, &mapped) != VK_SUCCESS) {
		g_vs.stat_buf = VK_NULL_HANDLE;
		return false;
	}
	g_vs.stat_mapped = static_cast<volatile uint32_t*>(mapped.pMappedData);
	for (int i = 0; i < 16; ++i) g_vs.stat_mapped[i] = 0;
	(void)vmaFlushAllocation(graphics.allocator, g_vs.stat_mem, 0, VK_WHOLE_SIZE);
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = 8 * 2;
	if (device.createQueryPool(&query, nullptr, &g_vs.pool) != vk::Result::eSuccess) g_vs.pool = nullptr;
	g_vs.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g_vs.failed    = false;
	return true;
}

bool EnsureImages(GraphicContext& graphics, vk::Extent2D extent) {
	if (g_vs.out.image != VK_NULL_HANDLE && g_vs.extent == extent) return true;
	Img* all[] {&g_vs.raw, &g_vs.out, &g_vs.hist[0], &g_vs.hist[1]};
	for (Img* img: all) Retire(graphics, *img);
	g_vs.extent        = extent;
	g_vs.layouts_ready = false, g_vs.history_valid = false;
	bool ok = true;
	for (Img* img: all) ok = ok && MakeImage(graphics, extent, *img);
	if (!ok)
		for (Img* img: all) Retire(graphics, *img);
	return ok;
}

void ReadTiming(GraphicContext& graphics) {
	if (g_vs.pool == nullptr) return;
	while (g_vs.read + 4 <= g_vs.written) {
		const auto              entry = static_cast<uint32_t>(g_vs.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (graphics.device.getQueryPoolResults(g_vs.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		if (stamps[1] > stamps[0]) {
			const double ms = static_cast<double>(stamps[1] - stamps[0]) * g_vs.period_ns / 1e6;
			g_vs.sum_ms += ms, g_vs.max_ms = std::max(g_vs.max_ms, ms);
			++g_vs.samples;
		}
		++g_vs.read;
	}
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

// A view-space point -> world: view_to_clip (column-major, as godrays.comp reads it), then the scene's clip -> world.
bool ViewToWorld(const std::array<float, 16>& m, const double* ivp, const double v[3], double out[3]) {
	const double clip[4] {m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12], m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13],
	                      m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14], m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15]};
	double w[4] {};
	for (int i = 0; i < 4; ++i) w[i] = ivp[i * 4 + 0] * clip[0] + ivp[i * 4 + 1] * clip[1] + ivp[i * 4 + 2] * clip[2] + ivp[i * 4 + 3] * clip[3];
	if (!(std::abs(w[3]) > 1e-12) || !std::isfinite(w[3])) return false;
	for (int i = 0; i < 3; ++i) out[i] = w[i] / w[3];
	return true;
}

vk::ImageView Skip(const char* why) {
	static const char* said = nullptr;
	++g_vs.skipped, g_vs.skip_why = why;
	g_vs.history_valid = false;
	if (said != why) {
		said = why;
		std::printf("VOLSHADOW: nothing this frame (%s): godrays' own march unchanged\n", why);
		std::fflush(stdout);
	}
	return nullptr;
}

} // namespace

int VolShadowLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_HWRT_VOLSHADOW");
		const int   q     = value != nullptr && *value != '\0' ? std::clamp(std::atoi(value), 0, 3) : 0;
		if (q > 0) {
			std::printf("VOLSHADOW: KYTY_HWRT_VOLSHADOW=%s read (level %d: %s)%s\n", value, q, q == 1 ? "Bajo" : q == 2 ? "Medio" : "Alto",
			            HwScene::Requested() ? "" : ": needs KYTY_HWRT=1 too: off");
			std::fflush(stdout);
		}
		return HwScene::Requested() ? q : 0;
	}();
	return level;
}

vk::ImageView VolShadowRun(vk::CommandBuffer command, UpscaleInputs& inputs, vk::Extent2D qextent, const std::array<float, 3>& sun_view,
                           float sun_weight, bool row0_up) {
	const int level = VolShadowLevel();
	if (level == 0) return nullptr;
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr || qextent.width == 0 || qextent.height == 0) return nullptr;
	if (!HwScene::Active()) return Skip("KYTY_HWRT inactive: the device has no ray queries");
	bool                         built = false;
	vk::AccelerationStructureKHR tlas  = HwScene::SceneTlas(&built);
	if (tlas == nullptr) return Skip("no scene TLAS built yet");
	std::array<float, 16> vp {};
	if (!HwScene::SceneCamera(vp)) {
		std::lock_guard lock(g_last_vp_mutex);
		if (!g_have_last_vp) return Skip("no camera matrix captured yet");
		vp = g_last_vp;
	}
	double vpd[16], ivp[16];
	for (int i = 0; i < 16; ++i) vpd[i] = vp[i];
	if (!Inverse4(vpd, ivp)) return Skip("the camera matrix is degenerate");
	double eye[4] {};
	for (int i = 0; i < 4; ++i) eye[i] = ivp[i * 4 + 2];
	if (!(std::abs(eye[3]) > 1e-12)) return Skip("the camera matrix has no perspective");
	const double cam[3] {eye[0] / eye[3], eye[1] / eye[3], eye[2] / eye[3]};
	// The sun in world space: two view points 2 m ahead, the second 0.5 m toward the sun.
	const auto&  m  = inputs.view_to_clip;
	const double zf = m[11] > 0.0f ? 2.0 : -2.0;
	const double a[3] {0.0, 0.0, zf};
	const double b[3] {0.5 * sun_view[0], 0.5 * sun_view[1], zf + 0.5 * sun_view[2]};
	double       wa[3], wb[3];
	if (!ViewToWorld(m, ivp, a, wa) || !ViewToWorld(m, ivp, b, wb)) return Skip("the sun's world direction is degenerate");
	double       sw[3] {wb[0] - wa[0], wb[1] - wa[1], wb[2] - wa[2]};
	const double sl = std::sqrt(sw[0] * sw[0] + sw[1] * sw[1] + sw[2] * sw[2]);
	if (!(sl > 1e-9) || !std::isfinite(sl)) return Skip("the sun's world direction is degenerate");
	for (double& c: sw) c /= sl;

	if (!EnsurePipeline(*graphics) || !EnsureImages(*graphics, qextent)) return Skip("could not create its pipeline or images");
	FreeRetired(*graphics);
	ReadTiming(*graphics);
	if (!built) ++g_vs.stale_tlas;

	const auto  now = std::chrono::steady_clock::now();
	const float dt  = g_vs.last_frame.time_since_epoch().count() == 0
	                      ? 0.016f
	                      : std::clamp(std::chrono::duration<float>(now - g_vs.last_frame).count(), 0.0f, 0.1f);
	if (g_vs.last_frame.time_since_epoch().count() != 0 && now - g_vs.last_frame > std::chrono::milliseconds(250)) g_vs.history_valid = false;
	g_vs.last_frame = now;
	// (in over 0.5 s once traced; follows the sun weight down: no cut is needed where godrays adds nothing)
	const float target = sun_weight > 0.0f ? 1.0f : 0.0f;
	g_vs.fade          = std::clamp(g_vs.fade + std::clamp(target - g_vs.fade, -dt / 0.3f, dt / 0.5f), 0.0f, 1.0f);
	static const float ray_len = EnvFloat("KYTY_HWRT_VOLSHADOW_LEN", 300.0f, 10.0f, 2000.0f);
	const int          steps   = level >= 3 ? 16 : level == 2 ? 10 : 6;

	++g_vs.frames;
	if (now - g_vs.last_log >= std::chrono::seconds(10)) {
		g_vs.last_log = now;
		(void)vmaInvalidateAllocation(graphics->allocator, g_vs.stat_mem, 0, VK_WHOLE_SIZE);
		std::array<uint32_t, 4> cur {}, d {};
		for (int i = 0; i < 4; ++i) cur[i] = g_vs.stat_mapped[i], d[i] = cur[i] - g_vs.stat_last[i];
		g_vs.stat_last = cur;
		std::printf("VOLSHADOW: stats (level %d, %ux%u, %d points a ray) %llu frames (TLAS stale %llu, skipped %llu: %s), sun rays %u sampled, "
		            "%.1f %% occluded by the TLAS, alignment %.1f %% (%u camera->texel probes hit at the depth; >~85 %%: the depth's world is "
		            "the TLAS's), fade %.2f, sun world dir (%.4f %.4f %.4f), GPU %.3f ms (max %.3f)\n",
		            level, qextent.width, qextent.height, steps, static_cast<unsigned long long>(g_vs.frames),
		            static_cast<unsigned long long>(g_vs.stale_tlas), static_cast<unsigned long long>(g_vs.skipped),
		            g_vs.skip_why != nullptr ? g_vs.skip_why : "-", d[0], d[0] > 0 ? 100.0 * d[1] / d[0] : 0.0,
		            d[2] > 0 ? 100.0 * d[3] / d[2] : 0.0, d[2], static_cast<double>(g_vs.fade), sw[0], sw[1], sw[2],
		            g_vs.samples > 0 ? g_vs.sum_ms / g_vs.samples : 0.0, g_vs.max_ms);
		std::fflush(stdout);
		g_vs.frames = g_vs.stale_tlas = g_vs.skipped = 0;
		g_vs.sum_ms = g_vs.max_ms = 0.0, g_vs.samples = 0;
	}
	if (!g_vs.said_on) {
		g_vs.said_on = true;
		std::printf("VOLSHADOW: on (level %d, quarter %ux%u, %d TLAS rays a texel toward the sun, %g m; world dir to the sun (%.4f %.4f %.4f), "
		            "camera (%.2f %.2f %.2f); godrays keeps min(depth-tested, TLAS) lit length)\n",
		            level, qextent.width, qextent.height, steps, static_cast<double>(ray_len), sw[0], sw[1], sw[2], cam[0], cam[1], cam[2]);
		std::fflush(stdout);
	}

	const uint32_t slot = static_cast<uint32_t>(g_vs.frame % Slots);
	{
		std::array<float, SlotBytes / 4> f {};
		for (int i = 0; i < 16; ++i) f[i] = static_cast<float>(ivp[i]);
		f[16] = static_cast<float>(cam[0]), f[17] = static_cast<float>(cam[1]), f[18] = static_cast<float>(cam[2]), f[19] = 1.0f;
		f[20] = static_cast<float>(sw[0]), f[21] = static_cast<float>(sw[1]), f[22] = static_cast<float>(sw[2]), f[23] = ray_len;
		f[24] = row0_up ? -1.0f : 1.0f, f[25] = 60.0f, f[26] = g_vs.fade, f[27] = 0.0f;
		std::memcpy(g_vs.scene_mapped + slot * SlotBytes, f.data(), SlotBytes);
		(void)vmaFlushAllocation(graphics->allocator, g_vs.scene_mem, slot * SlotBytes, SlotBytes);
	}

	VsPush push {};
	push.depth_map[0] = m[10], push.depth_map[1] = m[14], push.depth_map[2] = m[11], push.depth_map[3] = m[15];
	push.size[0] = static_cast<float>(inputs.color.extent.width), push.size[1] = static_cast<float>(inputs.color.extent.height);
	push.size[2] = static_cast<float>(qextent.width), push.size[3] = static_cast<float>(qextent.height);
	push.motion[0] = inputs.motion_scale_x, push.motion[1] = inputs.motion_scale_y;
	push.motion[2] = g_vs.history_valid && !inputs.reset && inputs.motion.view != nullptr ? 1.0f : 0.0f;
	push.motion[3] = static_cast<float>(steps);
	push.misc[1]   = static_cast<int32_t>(g_vs.frame & 0x7fffffff);

	const auto entry = static_cast<uint32_t>(g_vs.written % 8);
	const bool timed = g_vs.pool != nullptr && g_vs.written < g_vs.read + 8;
	if (timed) {
		command.resetQueryPool(g_vs.pool, entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_vs.pool, entry * 2);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	if (!g_vs.layouts_ready) {
		Img* all[] {&g_vs.raw, &g_vs.out, &g_vs.hist[0], &g_vs.hist[1]};
		for (Img* img: all) ToGeneral(command, img->image);
		g_vs.layouts_ready = true, g_vs.history_valid = false, push.motion[2] = 0.0f;
	}
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_vs.pipeline);

	const auto     gen = vk::ImageLayout::eGeneral;
	const uint32_t cur = static_cast<uint32_t>(g_vs.frame & 1u), prev = cur ^ 1u;
	const vk::DescriptorBufferInfo scene_info {vk::Buffer(g_vs.scene_buf), vk::DeviceSize {slot} * SlotBytes, SlotBytes};
	const vk::DescriptorBufferInfo stat_info {vk::Buffer(g_vs.stat_buf), 0, 64};
	vk::WriteDescriptorSetAccelerationStructureKHR as_write {};
	as_write.accelerationStructureCount = 1;
	as_write.pAccelerationStructures    = &tlas;
	const vk::DescriptorImageInfo depth_info {g_vs.point, inputs.depth.view, inputs.depth.layout};
	const vk::DescriptorImageInfo motion_info = inputs.motion.view != nullptr
	                                                ? vk::DescriptorImageInfo {g_vs.point, inputs.motion.view, inputs.motion.layout}
	                                                : vk::DescriptorImageInfo {g_vs.point, g_vs.hist[prev].view, gen};
	const auto run = [&](int mode, const Img& src, const Img& out) {
		vk::DescriptorImageInfo infos[Bindings] {};
		infos[0] = depth_info;
		infos[1] = motion_info;
		infos[2] = {g_vs.point, src.view, gen};
		infos[3] = {g_vs.point, g_vs.hist[prev].view, gen};
		infos[4] = {nullptr, out.view, gen};
		vk::WriteDescriptorSet writes[Bindings] {};
		for (uint32_t i = 0; i < Bindings; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = DescType(i);
			if (i == 5) writes[i].pBufferInfo = &scene_info;
			else if (i == 7) writes[i].pBufferInfo = &stat_info;
			else if (i == 6) writes[i].pNext = &as_write;
			else writes[i].pImageInfo = &infos[i];
		}
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_vs.layout, 0, Bindings, writes);
		push.misc[0] = mode;
		command.pushConstants(g_vs.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch((qextent.width + 7) / 8, (qextent.height + 7) / 8, 1);
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	};
	run(0, g_vs.hist[prev], g_vs.raw);       // trace -> raw (src unused)
	run(1, g_vs.raw, g_vs.hist[cur]);        // temporal: raw + hist[prev] -> hist[cur]
	run(2, g_vs.hist[cur], g_vs.out);        // spatial -> out (mix(1, fraction, fade))
	if (timed) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_vs.pool, entry * 2 + 1);
		++g_vs.written;
	}
	++g_vs.frame;
	g_vs.history_valid = true;
	return g_vs.out.view;
}

} // namespace Libs::Graphics::FrameGen
