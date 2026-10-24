// KYTY_REALISM=0|1|2|3 ("Modo Souls realista": off / suave / medio / fuerte; realism.h, realism.comp, material.cpp).
//
// The preset (each value only when the variable is not already set, so every individual KYTY_* var overrides it):
//   level             1 suave            2 medio            3 fuerte
//   KYTY_RT_SHADOWS   1 (str 0.45)       1 (0.55)           2 (0.60)
//   KYTY_GTAO         1 (str 0.50)       1 (0.60)           1 (0.70)
//   KYTY_SSGI         1 (int 0.35, ao 0) 1 (0.45, ao 0)     1 (0.55, ao 0)   (GTAO is the one occlusion)
//   KYTY_RT_REFLECTIONS 1 low           2 medium           2 medium
//   KYTY_REALISM_SPEC 0.35               0.50               0.70             (the sun's missing specular)
//   KYTY_REALISM_AO   1                  1                  1                (GTAO on the ambient share only)
//   KYTY_REALISM_TONE 0                  0.15               0.30             (local contrast + filmic toe)
//   GPU budget        2.5 ms             3.3 ms             4.0 ms           (1440p render, RTX 5080: the target the
//                                                                             10 s line compares the measured total to)
// Other knobs: KYTY_REALISM_FADE=<start>:<end> metres (35:110: every term fades out with distance, the fog guard),
// KYTY_REALISM_AMBIENT (0.3: the ambient irradiance as a share of the sun's, for the visibility estimate),
// KYTY_REALISM_FLIPY=1|0 (the screen y convention when the RT shadow pass has none), the material overrides
// (material.h). Every 10 s: "Realism: GPU ms" (each pass's mean and the total against the budget) and "Realism: last
// 10 s" (what the composite did). Off (the default): nothing set, created or recorded.
#include "local/realism.h"

#include "local/material.h"
#include "local/rt-reflections.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/realism_spv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace Libs::Graphics::FrameGen {
namespace {

// ---- the preset ----
struct PresetVar {
	const char* name;
	const char* value[3];
};
constexpr PresetVar Preset[] = {
    {"KYTY_RT_SHADOWS", {"1", "1", "2"}},
    {"KYTY_RT_SHADOWS_STRENGTH", {"0.45", "0.55", "0.6"}},
    {"KYTY_GTAO", {"1", "1", "1"}},
    {"KYTY_GTAO_STRENGTH", {"0.5", "0.6", "0.7"}},
    {"KYTY_SSGI", {"1", "1", "1"}},
    {"KYTY_SSGI_INTENSITY", {"0.35", "0.45", "0.55"}},
    {"KYTY_SSGI_AO", {"0", "0", "0"}},
    {"KYTY_RT_REFLECTIONS", {"1", "2", "2"}},
    {"KYTY_REALISM_SPEC", {"0.35", "0.5", "0.7"}},
    {"KYTY_REALISM_AO", {"1", "1", "1"}},
    {"KYTY_REALISM_TONE", {"0", "0.15", "0.3"}},
};
constexpr const char* LevelNames[4] = {"off", "suave", "medio", "fuerte"};
constexpr double      Budget[4]     = {0.0, 2.5, 3.3, 4.0};

std::string g_preset_log;
int         g_level = -1;

int ReadLevel() {
	const char* v = std::getenv("KYTY_REALISM");
	return v != nullptr && *v != '\0' ? std::clamp(std::atoi(v), 0, 3) : 0;
}
void SetEnvKeep(const char* name, const char* value) {
#ifdef _WIN32
	_putenv_s(name, value);
#else
	setenv(name, value, 0);
#endif
}
bool ApplyPreset() {
	if (g_level >= 0) return true;
	g_level = ReadLevel();
	if (g_level == 0) return true;
	std::string set, kept;
	for (const auto& v: Preset) {
		const char* cur = std::getenv(v.name);
		if (cur != nullptr && *cur != '\0') {
			kept += std::string(" ") + v.name + "=" + cur;
			continue;
		}
		SetEnvKeep(v.name, v.value[g_level - 1]);
		set += std::string(" ") + v.name + "=" + v.value[g_level - 1];
	}
	g_preset_log = std::string("Realism: KYTY_REALISM=") + std::to_string(g_level) + " (" + LevelNames[g_level] + ", GPU budget " +
	               std::to_string(Budget[g_level]).substr(0, 3) + " ms): set" + (set.empty() ? " nothing" : set) +
	               "; kept the user's:" + (kept.empty() ? " none" : kept);
	return true;
}
// (Before main: the passes read their variables lazily, the first frame at the earliest.)
const bool g_preset_applied = ApplyPreset();

float EnvFloat(const char* name, float fallback, float lo, float hi) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') return fallback;
	const float v = static_cast<float>(std::atof(value));
	return std::isfinite(v) ? std::clamp(v, lo, hi) : fallback;
}

// ---- GPU resources ----
constexpr uint32_t Bindings    = 14;
constexpr uint32_t CpuBytes    = 96;               // view_rot[3], up_view, cpu_i, cpu_f
constexpr uint32_t StateOff    = 96;               // gpu_state
constexpr uint32_t HistOff     = 112;              // hist[32]
constexpr uint32_t StatsOff    = HistOff + 32 * 4; // stats[16]
constexpr uint32_t CalibOff    = StatsOff + 16 * 4;
constexpr uint32_t FrameBytes  = 2048;
constexpr uint32_t CalibRuns   = 32;
constexpr uint32_t OneBitsIdx  = 19;               // cpu_i.w holds 1.0f: the exposure when the game gives none
static_assert(CalibOff + MatCalInts * 4 <= FrameBytes);

struct ReImage {
	VkImage             image  = VK_NULL_HANDLE;
	VmaAllocation       memory = nullptr;
	vk::ImageView       view   = nullptr;
	vk::Extent2D        extent {};
	vk::ImageUsageFlags usage {};
	bool                ready = false;
};
struct Retired {
	ReImage  image;
	uint64_t frame = 0;
};

struct ReState {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             point = nullptr, linear = nullptr;
	bool                    failed = false, images_failed = false;
	ReImage                 out, lum, dummy_u, dummy_f;
	std::vector<Retired>    retired;
	vk::Extent2D            render {};
	VkBuffer                frame_buf = VK_NULL_HANDLE, read_buf = VK_NULL_HANDLE;
	VmaAllocation           frame_dev = nullptr, frame_host = nullptr;
	const uint8_t*          mapped = nullptr;
	bool                    cleared = false;
	uint64_t                frame = 0;
	// This frame's hand-overs.
	RealismAo               ao {};
	bool                    have_ao = false;
	RealismSun              sun {};
	float                   sun_fade = 0.0f;
	std::array<float, 3>    sun_view {0, 1, 0};
	// The material decode's fade (a change ramps in over 1.5 s) and the calibration.
	uint32_t                mat_version = ~0u;
	float                   mat_fade = 0.0f;
	uint32_t                calib_runs = 0, calib_wait = 0, calib_tick = 0, calib_attempts = 0;
	bool                    calib_active = false, calib_clear = false, calib_done = false;
	uint64_t                calib_next = 0, gb_streak = 0;
	std::chrono::steady_clock::time_point last = {};
	// Statistics and the GPU time.
	std::array<uint32_t, 16> stat_last {};
	std::array<uint64_t, 16> stat_sum {};
	float                   esun = 0.0f, esun_raw = 0.0f;
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	std::chrono::steady_clock::time_point stat_time = std::chrono::steady_clock::now();
} g_re;

// Every pass's GPU ms since the last line.
std::array<double, static_cast<int>(RealismPass::Count)>   g_ms_sum {};
std::array<uint32_t, static_cast<int>(RealismPass::Count)> g_ms_n {};
std::chrono::steady_clock::time_point                       g_ms_time = std::chrono::steady_clock::now();

void FreeImage(GraphicContext& graphics, ReImage& img) {
	if (img.view != nullptr) graphics.device.destroyImageView(img.view, nullptr);
	if (img.image != VK_NULL_HANDLE) vmaDestroyImage(graphics.allocator, img.image, img.memory);
	img = ReImage {};
}
void Retire(ReImage& img) {
	if (img.image != VK_NULL_HANDLE) g_re.retired.push_back({img, g_re.frame});
	img = ReImage {};
}
void FreeRetired(GraphicContext& graphics) {
	auto& r = g_re.retired;
	for (size_t i = 0; i < r.size();) {
		if (g_re.frame >= r[i].frame + 16) {
			FreeImage(graphics, r[i].image);
			r[i] = r.back();
			r.pop_back();
		} else {
			++i;
		}
	}
}
bool MakeImage(GraphicContext& graphics, ReImage& img, vk::Format format, vk::Extent2D extent, bool storage) {
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
	img.extent = extent;
	img.usage  = static_cast<vk::ImageUsageFlags>(create.usage);
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
vk::DescriptorType TypeOf(uint32_t i) {
	return i == 12 || i == 13 ? vk::DescriptorType::eStorageBuffer
	       : i == 10 || i == 11 ? vk::DescriptorType::eStorageImage
	                            : vk::DescriptorType::eCombinedImageSampler;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_re.pipeline != nullptr) return true;
	if (g_re.failed) return false;
	g_re.failed       = true;
	const auto device = graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, Bindings> bindings {};
	for (uint32_t i = 0; i < Bindings; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = TypeOf(i);
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = Bindings;
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_re.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, 128};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_re.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_re.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(REALISM_SPV);
	module_info.pCode       = REALISM_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_re.layout;
	const auto created         = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_re.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_re.pipeline = nullptr;
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
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g_re.point : &g_re.linear) != vk::Result::eSuccess) return false;
	}
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = FrameBytes;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VmaAllocationCreateInfo dev {};
	dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_re.frame_buf, &g_re.frame_dev, nullptr) != VK_SUCCESS) {
		g_re.frame_buf = VK_NULL_HANDLE;
		return false;
	}
	buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_re.read_buf, &g_re.frame_host, &mapped) != VK_SUCCESS) {
		g_re.read_buf = VK_NULL_HANDLE;
		return false;
	}
	g_re.mapped = static_cast<const uint8_t*>(mapped.pMappedData);
	if (!MakeImage(graphics, g_re.dummy_u, vk::Format::eR32Uint, {1, 1}, false) ||
	    !MakeImage(graphics, g_re.dummy_f, vk::Format::eR16G16B16A16Sfloat, {1, 1}, false))
		return false;
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = 8 * 2;
	if (device.createQueryPool(&query, nullptr, &g_re.pool) != vk::Result::eSuccess) g_re.pool = nullptr;
	g_re.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g_re.failed    = false;
	return true;
}

bool EnsureImages(GraphicContext& graphics, vk::Extent2D render) {
	if (g_re.out.image != VK_NULL_HANDLE && g_re.render == render) return true;
	Retire(g_re.out), Retire(g_re.lum);
	const vk::Extent2D lum {std::max(1u, (render.width + 3) / 4), std::max(1u, (render.height + 3) / 4)};
	if (!MakeImage(graphics, g_re.out, vk::Format::eR16G16B16A16Sfloat, render, true) ||
	    !MakeImage(graphics, g_re.lum, vk::Format::eR16G16B16A16Sfloat, lum, true)) {
		Retire(g_re.out), Retire(g_re.lum);
		g_re.images_failed = true;
		return false;
	}
	g_re.render = render;
	return true;
}

void PrepareImages(vk::CommandBuffer command) {
	std::vector<vk::ImageMemoryBarrier2> barriers;
	bool                                 dummies = false;
	for (auto* img: {&g_re.out, &g_re.lum, &g_re.dummy_u, &g_re.dummy_f}) {
		if (img->ready || img->image == VK_NULL_HANDLE) continue;
		vk::ImageMemoryBarrier2 b {};
		b.srcStageMask        = vk::PipelineStageFlagBits2::eNone;
		b.dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
		b.dstAccessMask       = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite;
		b.oldLayout           = vk::ImageLayout::eUndefined;
		b.newLayout           = vk::ImageLayout::eGeneral;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image               = vk::Image(img->image);
		b.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		barriers.push_back(b);
		img->ready = true;
		if (img == &g_re.dummy_u || img == &g_re.dummy_f) dummies = true;
	}
	if (!barriers.empty()) {
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
		dependency.pImageMemoryBarriers    = barriers.data();
		command.pipelineBarrier2(dependency);
	}
	if (dummies) {
		const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		vk::ClearColorValue             zero {};
		zero.uint32 = std::array<uint32_t, 4> {0, 0, 0, 0};
		command.clearColorImage(vk::Image(g_re.dummy_u.image), vk::ImageLayout::eGeneral, &zero, 1, &whole);
		const vk::ClearColorValue one {std::array<float, 4> {1.0f, 1.0f, 1.0f, 1.0f}};
		command.clearColorImage(vk::Image(g_re.dummy_f.image), vk::ImageLayout::eGeneral, &one, 1, &whole);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
		        vk::AccessFlagBits2::eShaderRead);
	}
}

void ReadTiming(GraphicContext& graphics) {
	if (g_re.pool == nullptr) return;
	while (g_re.read + 4 <= g_re.written) {
		const auto              entry = static_cast<uint32_t>(g_re.read % 8);
		std::array<uint64_t, 2> stamps {};
		if (graphics.device.getQueryPoolResults(g_re.pool, entry * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		if (stamps[1] > stamps[0]) RealismNoteGpuMs(RealismPass::Realism, static_cast<double>(stamps[1] - stamps[0]) * g_re.period_ns / 1e6);
		++g_re.read;
	}
}

// The counters (cumulative on the GPU, read a few frames late) and, every 10 s, the two lines.
void ReadStats(GraphicContext& graphics, int level) {
	if (g_re.mapped == nullptr) return;
	(void)vmaInvalidateAllocation(graphics.allocator, g_re.frame_host, 0, VK_WHOLE_SIZE);
	const auto* c = reinterpret_cast<const uint32_t*>(g_re.mapped + StatsOff);
	for (uint32_t k = 0; k < 16; ++k) {
		if (g_re.frame > 8) g_re.stat_sum[k] += static_cast<uint32_t>(c[k] - g_re.stat_last[k]);
		g_re.stat_last[k] = c[k];
	}
	const auto* st = reinterpret_cast<const float*>(g_re.mapped + StateOff);
	g_re.esun = st[0], g_re.esun_raw = st[2];
	const auto now = std::chrono::steady_clock::now();
	if (now - g_re.stat_time < std::chrono::seconds(10)) return;
	const auto&  s  = g_re.stat_sum;
	const double px = static_cast<double>(std::max<uint64_t>(s[0], 1));
	const auto&  md = MaterialGet();
	std::printf("Realism: last %.0f s (%s): G-buffer matched on %.1f%% of the shaded pixels, smooth/wet %.1f%%, water %.1f%%, metal "
	            "%.1f%%, sun visible %.1f%%; highlight added on %.2f%% (mean %.3f display units there), AO darkening mean %.1f%%, "
	            "tone change mean %.3f stops; sun irradiance estimate %.3g (exposed; last %.3g, %llu samples), material fade %.2f, "
	            "decode v%u (rough %d, wet %d, albedo %d, metal %d), sky pixels skipped %llu\n",
	            std::chrono::duration<double>(now - g_re.stat_time).count(), LevelNames[level], 100.0 * s[1] / px, 100.0 * s[4] / px,
	            100.0 * s[2] / px, 100.0 * s[3] / px, 100.0 * s[11] / px, 100.0 * s[5] / px,
	            s[5] > 0 ? s[6] / 1000.0 / static_cast<double>(s[5]) : 0.0, 100.0 * s[7] / 1000.0 / px, s[8] / 1000.0 / px, g_re.esun,
	            g_re.esun_raw, static_cast<unsigned long long>(s[9]), g_re.mat_fade, md.version, md.code[MatRough], md.code[MatWet],
	            md.code[MatAlbedo], md.code[MatMetal], static_cast<unsigned long long>(s[10]));
	std::fflush(stdout);
	g_re.stat_sum.fill(0);
	g_re.stat_time = now;
}

void LogPassTimes(int level) {
	const auto now = std::chrono::steady_clock::now();
	if (now - g_ms_time < std::chrono::seconds(10)) return;
	static const char* names[] = {"RT shadows", "SSGI", "GTAO", "SSR", "RT reflections", "realism"};
	std::string line;
	double      total = 0.0;
	for (int p = 0; p < static_cast<int>(RealismPass::Count); ++p) {
		char buf[64];
		if (g_ms_n[p] > 0) {
			const double ms = g_ms_sum[p] / g_ms_n[p];
			total += ms;
			std::snprintf(buf, sizeof(buf), "%s%s %.2f", line.empty() ? "" : ", ", names[p], ms);
		} else {
			std::snprintf(buf, sizeof(buf), "%s%s -", line.empty() ? "" : ", ", names[p]);
		}
		line += buf;
	}
	std::printf("Realism: GPU ms per pass (10 s means; - = not running): %s; total %.2f ms (level %d %s, budget %.1f ms: %s)\n", line.c_str(),
	            total, level, LevelNames[level], Budget[level], total <= Budget[level] ? "within" : "OVER");
	std::fflush(stdout);
	g_ms_sum.fill(0.0);
	g_ms_n.fill(0);
	g_ms_time = now;
}

} // namespace

int RealismLevel() {
	ApplyPreset();
	(void)g_preset_applied;
	return g_level;
}
bool RealismWanted() { return RealismLevel() > 0; }

void RealismNoteGpuMs(RealismPass pass, double ms) {
	const int p = static_cast<int>(pass);
	if (p < 0 || p >= static_cast<int>(RealismPass::Count) || !(ms >= 0.0) || ms > 1000.0) return;
	g_ms_sum[p] += ms;
	++g_ms_n[p];
}

bool RealismTakesAo() {
	static const bool on = EnvFloat("KYTY_REALISM_AO", 0.0f, 0.0f, 1.0f) > 0.0f;
	return RealismWanted() && on && !g_re.failed && !g_re.images_failed && !VideoOut::MoviePlaying();
}
void RealismSetAo(const RealismAo& ao) {
	g_re.ao      = ao;
	g_re.have_ao = ao.ao != nullptr && ao.geom != nullptr && ao.extent.width > 0;
}
void RealismSetSun(const RealismSun& sun) { g_re.sun = sun; }

bool RealismApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int level = RealismLevel();
	if (level == 0) return false;
	const bool have_ao = g_re.have_ao;
	g_re.have_ao       = false; // (this frame's only)
	const RealismSun sun_in = g_re.sun;
	g_re.sun.valid     = false;
	{
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("%s\n", g_preset_log.c_str());
			std::fflush(stdout);
		}
	}
	LogPassTimes(level);
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
	if (g_re.images_failed || !EnsurePipeline(*graphics) || !EnsureImages(*graphics, color.extent)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("Realism: off (could not create its pipeline or images)\n");
			std::fflush(stdout);
		}
		return false;
	}
	FreeRetired(*graphics);
	ReadTiming(*graphics);
	ReadStats(*graphics, level);

	const auto  now = std::chrono::steady_clock::now();
	const float dt  = g_re.last.time_since_epoch().count() == 0 ? 0.016f
	                                                            : std::clamp(std::chrono::duration<float>(now - g_re.last).count(), 0.0f, 0.25f);
	g_re.last = now;

	// The G-buffer copy, the camera, the material decode.
	RtReflGbuffer gbuf;
	const bool    have_gbuf  = RtReflGetGbuffer(gbuf) && gbuf.extent == color.extent;
	const bool    have_water = have_gbuf && gbuf.depth != nullptr;
	RtReflCamera  camera;
	RtReflGetCamera(m, camera);
	MaterialRefresh();
	g_re.gb_streak = have_gbuf ? g_re.gb_streak + 1 : 0;
	// Calibration attempts (the material decode's correlations): after 180 frames of G-buffer copies, 32 runs every
	// 4th frame, read 8 frames later; again ~30 s later until the albedo is known and confirmed (2 attempts), at most 6.
	if (!g_re.calib_done && !g_re.calib_active && g_re.frame >= g_re.calib_next && g_re.gb_streak >= 180) {
		g_re.calib_active = true, g_re.calib_clear = true, g_re.calib_runs = 0, g_re.calib_wait = 0;
	}
	const bool calibrate = g_re.calib_active && g_re.calib_runs < CalibRuns && (g_re.calib_tick++ & 3u) == 0 && have_gbuf;
	if (g_re.calib_active && g_re.calib_runs >= CalibRuns && ++g_re.calib_wait > 8) {
		g_re.calib_active = false;
		(void)vmaInvalidateAllocation(graphics->allocator, g_re.frame_host, 0, VK_WHOLE_SIZE);
		++g_re.calib_attempts;
		MaterialAnalyze(reinterpret_cast<const int32_t*>(g_re.mapped + CalibOff), g_re.calib_attempts);
		if ((MaterialAlbedoKnown() && g_re.calib_attempts >= 2) || g_re.calib_attempts >= 6) g_re.calib_done = true;
		else g_re.calib_next = g_re.frame + 1800;
	}
	const MaterialDecode& md = MaterialGet();
	if (md.version != g_re.mat_version) {
		if (g_re.mat_version != ~0u) g_re.mat_fade = 0.0f; // (a decode change: ramp in again)
		g_re.mat_version = md.version;
	}
	g_re.mat_fade = std::min(g_re.mat_fade + dt / 1.5f, 1.0f);
	const bool world_normal = md.code[MatNormal] >= 0 && (md.code[MatNormal] & 64) != 0;
	const bool use_gbuf     = have_gbuf && md.code[MatNormal] >= 0 && (!world_normal || camera.rows);

	// The sun (the RT shadow pass's, in its view space), faded in / out over 0.5 s.
	static const int flip_env = [] {
		const char* v = std::getenv("KYTY_REALISM_FLIPY");
		return v != nullptr && *v != '\0' ? (*v != '0' ? 1 : 0) : -1;
	}();
	if (sun_in.valid) g_re.sun_view = sun_in.view;
	g_re.sun_fade = std::clamp(g_re.sun_fade + (sun_in.valid ? dt / 0.5f : -dt / 0.5f), 0.0f, 1.0f);
	const float sun_fade = std::min(g_re.sun_fade, sun_in.valid ? sun_in.fade : g_re.sun_fade);
	RtReflByteTable table;
	const bool      have_table = RtReflGetByteTable(table);
	const bool      row0_up    = flip_env >= 0 ? flip_env == 1 : sun_in.valid ? sun_in.row0_up : have_table ? table.row0_up : true;

	const float spec      = EnvFloat("KYTY_REALISM_SPEC", 0.5f, 0.0f, 1.5f);
	const float ao_scale  = EnvFloat("KYTY_REALISM_AO", 0.0f, 0.0f, 1.0f);
	const float tone      = EnvFloat("KYTY_REALISM_TONE", 0.0f, 0.0f, 1.0f);
	const float ambient   = EnvFloat("KYTY_REALISM_AMBIENT", 0.3f, 0.05f, 2.0f);
	static const std::pair<float, float> fade_m = [] {
		float a = 35.0f, b = 110.0f;
		if (const char* v = std::getenv("KYTY_REALISM_FADE"); v != nullptr) {
			float x = 0, y = 0;
			if (std::sscanf(v, "%f:%f", &x, &y) == 2 && x >= 1.0f && y > x) a = x, b = y;
		}
		return std::pair<float, float> {a, b};
	}();
	const bool  ao_on       = have_ao && ao_scale > 0.0f;
	const bool  game_exposure = inputs.exposure_buffer != nullptr && inputs.exposure_range >= sizeof(float);
	{
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("Realism: on (level %d %s: specular %g, AO on the ambient share %s (GTAO strength %g x %g), tone %g, fade %g-%g m, "
			            "ambient/sun %g; G-buffer %s, sun %s, exposure %s, screen row 0 = NDC y %s)\n",
			            level, LevelNames[level], spec, ao_on ? "yes" : "no (GTAO off or composited by itself)", g_re.ao.strength, ao_scale,
			            tone, fade_m.first, fade_m.second, ambient, have_gbuf ? "copy found" : "not found yet",
			            sun_in.valid ? "from RT shadows" : "none yet (RT shadows off or still calibrating: no highlight)",
			            game_exposure ? "the game's" : "1.0", row0_up ? "+1" : "-1");
			std::fflush(stdout);
		}
	}

	// Push constants (realism.comp's layout).
	struct {
		float   proj[4], depth_map[4], size[4], sun[4], params[4], params2[4];
		int32_t misc[4], dec[4];
	} push {};
	static_assert(sizeof(push) == 128);
	const vk::Extent2D lum = g_re.lum.extent;
	push.proj[0] = m[0], push.proj[1] = m[5], push.proj[2] = m[8], push.proj[3] = m[9];
	push.depth_map[0] = m[10], push.depth_map[1] = m[14], push.depth_map[2] = m[11], push.depth_map[3] = m[15];
	push.size[0] = static_cast<float>(color.extent.width), push.size[1] = static_cast<float>(color.extent.height);
	push.size[2] = static_cast<float>(lum.width), push.size[3] = static_cast<float>(lum.height);
	{
		const auto&  l = g_re.sun_view;
		const double n = std::sqrt(static_cast<double>(l[0]) * l[0] + static_cast<double>(l[1]) * l[1] + static_cast<double>(l[2]) * l[2]);
		const bool   ok = n > 1e-6 && std::isfinite(n);
		push.sun[0] = ok ? static_cast<float>(l[0] / n) : 0.0f, push.sun[1] = ok ? static_cast<float>(l[1] / n) : 1.0f;
		push.sun[2] = ok ? static_cast<float>(l[2] / n) : 0.0f, push.sun[3] = ok && use_gbuf ? sun_fade : 0.0f;
	}
	push.params[0] = spec, push.params[1] = g_re.ao.strength * ao_scale, push.params[2] = tone, push.params[3] = g_re.mat_fade;
	push.params2[0] = std::clamp(md.smooth_lo, 0.0f, 0.95f), push.params2[1] = std::max(md.smooth_hi, push.params2[0] + 0.05f);
	push.params2[2] = std::clamp(md.wet_lo, 0.0f, 0.95f), push.params2[3] = std::max(md.wet_hi, push.params2[2] + 0.05f);
	const bool stats_frame = (g_re.frame & 3u) == 0; // (the counters and the sun estimate's samples)
	push.misc[1] = static_cast<int32_t>(g_re.frame & 0x7fffffff);
	push.misc[2] = (use_gbuf ? 1 : 0) | (use_gbuf && have_water ? 2 : 0) | (row0_up ? 4 : 0) | (push.sun[3] > 0.0f ? 8 : 0) |
	               (ao_on ? 16 : 0) | (stats_frame ? 32 : 0);
	push.misc[3] = game_exposure ? static_cast<int32_t>(inputs.exposure_index) : static_cast<int32_t>(OneBitsIdx);
	push.dec[0] = use_gbuf ? md.code[MatNormal] : -1;
	push.dec[1] = use_gbuf ? md.code[MatRough] : -1;
	push.dec[2] = use_gbuf ? md.code[MatWet] : -1;
	push.dec[3] = use_gbuf ? md.code[MatAlbedo] : -1;

	const bool timed = g_re.pool != nullptr && g_re.written < g_re.read + 8;
	const auto entry = static_cast<uint32_t>(g_re.written % 8);
	if (timed) {
		command.resetQueryPool(g_re.pool, entry * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_re.pool, entry * 2);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite);
	PrepareImages(command);
	if (!g_re.cleared) {
		g_re.cleared = true;
		command.fillBuffer(vk::Buffer(g_re.frame_buf), 0, VK_WHOLE_SIZE, 0);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer,
		        vk::AccessFlagBits2::eTransferWrite);
	} else if (g_re.calib_clear) {
		command.fillBuffer(vk::Buffer(g_re.frame_buf), CalibOff, FrameBytes - CalibOff, 0);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer,
		        vk::AccessFlagBits2::eTransferWrite);
	}
	g_re.calib_clear = false;
	{
		std::array<float, CpuBytes / 4> header {};
		std::copy(camera.view_rows.begin(), camera.view_rows.end(), header.begin());
		header[12] = camera.up[0], header[13] = camera.up[1], header[14] = camera.up[2], header[15] = camera.up_valid ? 1.0f : 0.0f;
		const int32_t metal = use_gbuf ? md.code[MatMetal] : -1;
		const int32_t ints[4] = {metal, static_cast<int32_t>(g_re.ao.extent.width), static_cast<int32_t>(g_re.ao.extent.height), 0};
		std::memcpy(&header[16], ints, sizeof(ints));
		const float one = 1.0f;
		std::memcpy(&header[OneBitsIdx], &one, sizeof(one));
		header[20] = 1.0f - std::exp(-4.0f * dt / 1.5f); // (the sun estimate's EMA: ~1.5 s, updated every 4th frame)
		header[21] = ambient, header[22] = fade_m.first, header[23] = fade_m.second;
		command.updateBuffer(vk::Buffer(g_re.frame_buf), 0, CpuBytes, header.data());
	}
	Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_re.pipeline);

	const auto gen = vk::ImageLayout::eGeneral;
	const auto run = [&](int mode, vk::Extent2D size) {
		vk::DescriptorImageInfo infos[Bindings] {};
		infos[0] = {g_re.linear, color.view, color.layout};
		infos[1] = {g_re.point, inputs.depth.view, inputs.depth.layout};
		infos[2] = use_gbuf && have_water ? vk::DescriptorImageInfo {g_re.point, gbuf.depth, gbuf.depth_layout}
		                                  : vk::DescriptorImageInfo {g_re.point, inputs.depth.view, inputs.depth.layout};
		for (uint32_t i = 0; i < 4; ++i)
			infos[3 + i] = use_gbuf && gbuf.views[i] != nullptr
			                   ? vk::DescriptorImageInfo {g_re.point, gbuf.views[i], vk::ImageLayout::eShaderReadOnlyOptimal}
			                   : vk::DescriptorImageInfo {g_re.point, g_re.dummy_u.view, gen};
		infos[7]  = ao_on ? vk::DescriptorImageInfo {g_re.point, g_re.ao.ao, gen} : vk::DescriptorImageInfo {g_re.point, g_re.dummy_f.view, gen};
		infos[8]  = ao_on ? vk::DescriptorImageInfo {g_re.point, g_re.ao.geom, gen} : vk::DescriptorImageInfo {g_re.point, g_re.dummy_f.view, gen};
		infos[9]  = {g_re.linear, g_re.lum.view, gen};
		infos[10] = {nullptr, g_re.out.view, gen};
		infos[11] = {nullptr, g_re.lum.view, gen};
		const vk::DescriptorBufferInfo frame_info {vk::Buffer(g_re.frame_buf), 0, VK_WHOLE_SIZE};
		const vk::DescriptorBufferInfo exposure_info =
		    game_exposure ? vk::DescriptorBufferInfo {inputs.exposure_buffer, inputs.exposure_base, inputs.exposure_range} : frame_info;
		vk::WriteDescriptorSet writes[Bindings] {};
		for (uint32_t i = 0; i < Bindings; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = TypeOf(i);
			if (i == 12) writes[i].pBufferInfo = &frame_info;
			else if (i == 13) writes[i].pBufferInfo = &exposure_info;
			else writes[i].pImageInfo = &infos[i];
		}
		push.misc[0] = mode;
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_re.layout, 0, Bindings, writes);
		command.pushConstants(g_re.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	};
	// (The lum pass writes the image binding 9 also names: binding 9 is not read in mode 0.)
	run(0, lum);
	if (calibrate) {
		run(1, vk::Extent2D {160, 90});
		++g_re.calib_runs;
	}
	run(2, color.extent);
	if (stats_frame) run(3, vk::Extent2D {1, 1});
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead);
	{
		vk::BufferCopy region {StateOff, StateOff, FrameBytes - StateOff};
		command.copyBuffer(vk::Buffer(g_re.frame_buf), vk::Buffer(g_re.read_buf), 1, &region);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
		        vk::AccessFlagBits2::eHostRead);
	}
	if (timed) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_re.pool, entry * 2 + 1);
		++g_re.written;
	}
	++g_re.frame;
	inputs.color.image  = vk::Image(g_re.out.image);
	inputs.color.view   = g_re.out.view;
	inputs.color.layout = gen;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = g_re.out.usage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}

} // namespace Libs::Graphics::FrameGen
