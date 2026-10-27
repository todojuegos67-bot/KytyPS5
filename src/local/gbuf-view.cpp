// KYTY_GBUF_VIEW / KYTY_GBUF_TAP (gbuf-view.h; refl7): the G-buffer diagnostic. One compute pipeline (gbuf-view.comp),
// push descriptors; per frame a stats dispatch (256 x 144 samples) into one of 4 slots of a device buffer, copied to a
// host buffer the CPU reads 3 frames later, then (unless the view is "stats") the view into our own image, which the
// upscaler then reads in place of the game's color. Nothing of this exists while both variables are unset.
#include "local/gbuf-view.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/gbuf_view_spv.h"
#include "local/rt-reflections.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace Libs::Graphics::FrameGen {
namespace {

constexpr uint32_t Bindings  = 11;
constexpr uint32_t SlotUints = 512;
constexpr uint32_t Slots     = 4;
constexpr uint32_t StProbe   = 400;

enum View : int32_t { VField = 0, VMosaic = 1, VNormal = 2, VForward = 3, VTap = 4, VStats = 5, VRaw = 6 };

struct Push {
	int32_t misc[4];
	int32_t misc2[4];
	float   size[4];
	float   params[4];
};
static_assert(sizeof(Push) == 64);

struct Img {
	VkImage       image  = VK_NULL_HANDLE;
	VmaAllocation memory = nullptr;
	vk::ImageView view   = nullptr;
	vk::Extent2D  extent {};
	vk::ImageUsageFlags usage {};
	bool          ready = false; // (in the general layout)
};

struct State {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             point = nullptr, linear = nullptr;
	bool                    failed = false;
	Img                     out, dummy_u, dummy_f;
	std::vector<std::pair<Img, uint64_t>> retired;
	VkBuffer                dev_buf = VK_NULL_HANDLE, host_buf = VK_NULL_HANDLE;
	VmaAllocation           dev_mem = nullptr, host_mem = nullptr;
	const uint32_t*         mapped = nullptr;
	bool                    cleared = false;
	uint64_t                frame = 0;
	// The view (KYTY_GBUF_VIEW, then kyty-gbuf-view.txt) and its name for the log.
	int32_t                 view = VStats, field = 2 | (0 << 2) | (8 << 7);
	std::string             view_name;
	std::filesystem::file_time_type file_time {};
	// Sums since the last line (5 s).
	std::array<double, SlotUints> sums {};
	uint64_t                frames = 0, frames_gbuf = 0;
	std::array<uint32_t, 9> probe {};
	bool                    probe_valid = false;
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
} g_gv;

bool EnvSet(const char* name) {
	const char* v = std::getenv(name);
	return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
}

// "<view>" -> view, field; false: not understood.
bool ParseView(std::string text, int32_t& view, int32_t& field) {
	while (!text.empty() && (text.back() == '\r' || text.back() == '\n' || text.back() == ' ')) text.pop_back();
	while (!text.empty() && text.front() == ' ') text.erase(0, 1);
	for (auto& c: text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	if (text.empty() || text == "1" || text == "stats" || text == "off") return view = VStats, true;
	if (text == "mosaic" || text == "all") return view = VMosaic, true;
	if (text == "normal") return view = VNormal, true;
	if (text == "forward" || text == "water") return view = VForward, true;
	if (text == "tap") return view = VTap, true;
	if (text == "raw") return view = VRaw, true;
	int layer = -1, b = -1, shift = -1, bits = -1;
	if (std::sscanf(text.c_str(), "l%d.b%d", &layer, &b) == 2 && layer >= 0 && layer < 4 && b >= 0 && b < 4) {
		view  = VField;
		field = layer | ((b * 8) << 2) | (8 << 7);
		return true;
	}
	if (std::sscanf(text.c_str(), "l%d:%d:%d", &layer, &shift, &bits) == 3 || std::sscanf(text.c_str(), "%d:%d:%d", &layer, &shift, &bits) == 3) {
		if (layer < 0 || layer > 3 || shift < 0 || bits < 1 || bits > 16 || shift + bits > 32) return false;
		view  = VField;
		field = layer | (shift << 2) | (bits << 7);
		return true;
	}
	return false;
}

void SetView(const std::string& text, const char* from) {
	int32_t view = g_gv.view, field = g_gv.field;
	if (!ParseView(text, view, field)) {
		std::printf("GBUF: view \"%s\" (%s) not understood: L<layer>.b<byte>, <layer>:<shift>:<bits>, mosaic, raw, normal, "
		            "forward, tap or stats\n",
		            text.c_str(), from);
		std::fflush(stdout);
		return;
	}
	g_gv.view = view, g_gv.field = field, g_gv.view_name = text;
	std::printf("GBUF: view %s (%s)\n", text.empty() ? "stats" : text.c_str(), from);
	std::fflush(stdout);
}

// kyty-gbuf-view.txt in the working directory: its first line, whenever the file changes (checked once a second).
void PollViewFile() {
	if (g_gv.frame % 60 != 0) return;
	std::error_code error;
	const std::filesystem::path path = "kyty-gbuf-view.txt";
	const auto time = std::filesystem::last_write_time(path, error);
	if (error || time == g_gv.file_time) return;
	g_gv.file_time = time;
	std::ifstream in(path);
	std::string   line;
	std::getline(in, line);
	SetView(line, "kyty-gbuf-view.txt");
}

void FreeImg(GraphicContext& graphics, Img& img) {
	if (img.view != nullptr) graphics.device.destroyImageView(img.view, nullptr);
	if (img.image != VK_NULL_HANDLE) vmaDestroyImage(graphics.allocator, img.image, img.memory);
	img = Img {};
}

bool MakeImg(GraphicContext& graphics, Img& img, vk::Format format, vk::Extent2D extent, bool storage) {
	VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create.imageType   = VK_IMAGE_TYPE_2D;
	create.format      = static_cast<VkFormat>(format);
	create.extent      = {extent.width, extent.height, 1};
	create.mipLevels   = 1;
	create.arrayLayers = 1;
	create.samples     = VK_SAMPLE_COUNT_1_BIT;
	create.tiling      = VK_IMAGE_TILING_OPTIMAL;
	create.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
	               (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0u);
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

void ToGeneral(vk::CommandBuffer command, Img& img) {
	if (img.ready || img.image == VK_NULL_HANDLE) return;
	vk::ImageMemoryBarrier2 b {};
	b.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	b.dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader;
	b.dstAccessMask       = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
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
	img.ready = true;
}

vk::DescriptorType TypeOf(uint32_t i) {
	return i == 8 ? vk::DescriptorType::eStorageImage
	       : (i == 9 || i == 10) ? vk::DescriptorType::eStorageBuffer
	                             : vk::DescriptorType::eCombinedImageSampler;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_gv.pipeline != nullptr) return true;
	if (g_gv.failed) return false;
	g_gv.failed       = true;
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
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_gv.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Push)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_gv.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_gv.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(GBUF_VIEW_SPV);
	module_info.pCode       = GBUF_VIEW_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_gv.layout;
	const auto created         = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &g_gv.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_gv.pipeline = nullptr;
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
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g_gv.point : &g_gv.linear) != vk::Result::eSuccess) return false;
	}
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = SlotUints * Slots * 4;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VmaAllocationCreateInfo dev {};
	dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_gv.dev_buf, &g_gv.dev_mem, nullptr) != VK_SUCCESS) {
		g_gv.dev_buf = VK_NULL_HANDLE;
		return false;
	}
	buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_gv.host_buf, &g_gv.host_mem, &mapped) != VK_SUCCESS) {
		g_gv.host_buf = VK_NULL_HANDLE;
		return false;
	}
	g_gv.mapped = static_cast<const uint32_t*>(mapped.pMappedData);
	if (!MakeImg(graphics, g_gv.dummy_u, vk::Format::eR32Uint, {1, 1}, false)) return false;
	if (!MakeImg(graphics, g_gv.dummy_f, vk::Format::eR16G16B16A16Sfloat, {1, 1}, false)) return false;
	g_gv.failed = false;
	return true;
}

// The slot written 3 frames ago (the GPU is done with it by now: frames in flight are fewer): its sums added up.
void Collect(GraphicContext& graphics) {
	if (g_gv.mapped == nullptr || g_gv.frame < Slots) return;
	(void)vmaInvalidateAllocation(graphics.allocator, g_gv.host_mem, 0, VK_WHOLE_SIZE);
	const uint32_t* s = g_gv.mapped + ((g_gv.frame + 1) % Slots) * SlotUints;
	for (uint32_t i = 0; i < StProbe; ++i) g_gv.sums[i] += static_cast<double>(s[i]);
	if (s[0] != 0) {
		std::memcpy(g_gv.probe.data(), s + StProbe, sizeof(g_gv.probe));
		g_gv.probe_valid = true;
	}
}

float AsFloat(uint32_t v) {
	float f = 0.0f;
	std::memcpy(&f, &v, sizeof(f));
	return f;
}

void Report(bool have_gbuf, bool have_depth, bool have_tap) {
	const auto now = std::chrono::steady_clock::now();
	if (now - g_gv.last < std::chrono::seconds(5)) return;
	g_gv.last      = now;
	const auto& s  = g_gv.sums;
	const double n = s[0];
	std::printf("GBUF: last 5 s: %llu frames, G-buffer copy in %llu, pre-lighting depth %s, tap %s, view %s; %.0f samples/frame "
	            "(sky excluded), forward %.1f %%\n",
	            static_cast<unsigned long long>(g_gv.frames), static_cast<unsigned long long>(g_gv.frames_gbuf), have_depth ? "yes" : "no",
	            have_tap ? "yes" : "no", g_gv.view_name.empty() ? "stats" : g_gv.view_name.c_str(),
	            g_gv.frames ? n / static_cast<double>(g_gv.frames) : 0.0, n > 0 ? 100.0 * s[3] / n : 0.0);
	if (n > 100) {
		const double my = s[1] / n, vy = std::max(s[2] / n - my * my, 1e-9);
		std::string  line;
		char         buf[256];
		for (int f = 0; f < 16; ++f) {
			const uint32_t o  = 16 + f * 20;
			const double   mx = s[o + 16] / n, vx = std::max(s[o + 17] / n - mx * mx, 0.0);
			const double   r  = vx > 1e-6 ? (s[o + 18] / n - mx * my) / std::sqrt(vx * vy) : 0.0;
			// (the histogram as one hex digit per bin: its share of the samples x 15, rounded up so a present bin shows)
			char hist[17];
			for (int b = 0; b < 16; ++b) {
				const double share = s[o + b] / n;
				const int    d     = share <= 0.0 ? 0 : std::clamp(static_cast<int>(std::ceil(share * 15.0)), 1, 15);
				hist[b]            = "0123456789abcdef"[d];
			}
			hist[16] = 0;
			std::snprintf(buf, sizeof(buf), "GBUF:   L%d.b%d mean %5.1f sd %5.1f hist %s r(color) %+.2f forward mean %s\n", f >> 2, f & 3,
			              mx, std::sqrt(vx), hist, r, s[3] > 0 ? std::to_string(static_cast<int>(s[o + 19] / s[3] + 0.5)).c_str() : "-");
			line += buf;
		}
		std::fputs(line.c_str(), stdout);
	}
	if (g_gv.probe_valid) {
		const auto& p = g_gv.probe;
		std::printf("GBUF:   probe (crosshair): L0 %08x L1 %08x L2 %08x L3 %08x | bytes L0 %u %u %u %u, L1 %u %u %u %u, L2 %u %u %u "
		            "%u, L3 %u %u %u %u | depth %.7g, pre-lighting depth %.7g | color %.3g %.3g %.3g\n",
		            p[0], p[1], p[2], p[3], p[0] & 255, (p[0] >> 8) & 255, (p[0] >> 16) & 255, p[0] >> 24, p[1] & 255, (p[1] >> 8) & 255,
		            (p[1] >> 16) & 255, p[1] >> 24, p[2] & 255, (p[2] >> 8) & 255, (p[2] >> 16) & 255, p[2] >> 24, p[3] & 255,
		            (p[3] >> 8) & 255, (p[3] >> 16) & 255, p[3] >> 24, AsFloat(p[4]), AsFloat(p[5]), AsFloat(p[6]), AsFloat(p[7]),
		            AsFloat(p[8]));
	}
	(void)have_gbuf;
	std::fflush(stdout);
	g_gv.sums.fill(0.0);
	g_gv.frames = 0, g_gv.frames_gbuf = 0;
}

} // namespace

bool GbufViewWanted() {
	static const bool wanted = EnvSet("KYTY_GBUF_VIEW") || EnvSet("KYTY_GBUF_TAP");
	return wanted;
}

bool GbufViewApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	if (!GbufViewWanted()) return false;
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr || VideoOut::MoviePlaying()) return false;
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || color.extent.width < 16 || color.extent.height < 16 ||
	    inputs.depth.extent != color.extent)
		return false;
	if (!EnsurePipeline(*graphics)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("GBUF: off (could not create its pipeline)\n");
			std::fflush(stdout);
		}
		return false;
	}
	if (g_gv.frame == 0) {
		const char* v = std::getenv("KYTY_GBUF_VIEW");
		SetView(v != nullptr && std::strcmp(v, "0") != 0 ? v : "stats", "KYTY_GBUF_VIEW");
	}
	PollViewFile();
	// (the view image at the render size: a new one at a size change, the old one destroyed 16 frames later)
	for (size_t i = 0; i < g_gv.retired.size();) {
		if (g_gv.frame >= g_gv.retired[i].second + 16) {
			FreeImg(*graphics, g_gv.retired[i].first);
			g_gv.retired[i] = g_gv.retired.back();
			g_gv.retired.pop_back();
		} else {
			++i;
		}
	}
	if (g_gv.out.image != VK_NULL_HANDLE && g_gv.out.extent != color.extent) {
		g_gv.retired.emplace_back(g_gv.out, g_gv.frame);
		g_gv.out = Img {};
	}
	if (g_gv.out.image == VK_NULL_HANDLE && !MakeImg(*graphics, g_gv.out, vk::Format::eR16G16B16A16Sfloat, color.extent, true)) {
		FreeImg(*graphics, g_gv.out);
		return false;
	}
	Collect(*graphics);

	RtReflGbuffer gbuf;
	const bool    have_gbuf  = RtReflGetGbuffer(gbuf) && gbuf.extent == color.extent;
	const bool    have_depth = have_gbuf && gbuf.depth != nullptr;
	GbufTapImage  tap;
	const bool    have_tap = GbufTapGet(tap) && tap.view != nullptr;
	++g_gv.frames;
	if (have_gbuf) ++g_gv.frames_gbuf;

	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite);
	ToGeneral(command, g_gv.out);
	ToGeneral(command, g_gv.dummy_u);
	ToGeneral(command, g_gv.dummy_f);
	const uint32_t slot = static_cast<uint32_t>(g_gv.frame % Slots);
	command.fillBuffer(vk::Buffer(g_gv.dev_buf), slot * SlotUints * 4, SlotUints * 4, 0);
	Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, g_gv.pipeline);

	const bool game_exposure = inputs.exposure_buffer != nullptr && inputs.exposure_range >= sizeof(float);
	Push       push {};
	push.misc[1]  = g_gv.view;
	push.misc[2]  = g_gv.field;
	push.misc[3]  = (have_gbuf ? 1 : 0) | (have_depth ? 2 : 0) | (have_tap ? 4 : 0);
	push.misc2[0] = static_cast<int32_t>(slot * SlotUints);
	push.misc2[1] = 0;
	push.misc2[2] = game_exposure ? static_cast<int32_t>(inputs.exposure_index) : -1;
	push.size[0] = static_cast<float>(color.extent.width), push.size[1] = static_cast<float>(color.extent.height);
	push.size[2] = 1.0f / push.size[0], push.size[3] = 1.0f / push.size[1];
	push.params[1] = 1e-4f;

	const auto run = [&](int mode, uint32_t gx, uint32_t gy) {
		const auto              gen = vk::ImageLayout::eGeneral;
		vk::DescriptorImageInfo infos[Bindings] {};
		infos[0] = {g_gv.point, color.view, color.layout};
		infos[1] = {g_gv.point, inputs.depth.view, inputs.depth.layout};
		for (uint32_t i = 0; i < 4; ++i)
			infos[2 + i] = have_gbuf && gbuf.views[i] != nullptr ? vk::DescriptorImageInfo {g_gv.point, gbuf.views[i], vk::ImageLayout::eShaderReadOnlyOptimal}
			                                                     : vk::DescriptorImageInfo {g_gv.point, g_gv.dummy_u.view, gen};
		infos[6] = have_depth ? vk::DescriptorImageInfo {g_gv.point, gbuf.depth, gbuf.depth_layout} : vk::DescriptorImageInfo {g_gv.point, inputs.depth.view, inputs.depth.layout};
		infos[7] = have_tap ? vk::DescriptorImageInfo {g_gv.linear, tap.view, vk::ImageLayout::eShaderReadOnlyOptimal}
		                    : vk::DescriptorImageInfo {g_gv.point, g_gv.dummy_f.view, gen};
		infos[8] = {nullptr, g_gv.out.view, gen};
		const vk::DescriptorBufferInfo stats_info {vk::Buffer(g_gv.dev_buf), 0, VK_WHOLE_SIZE};
		const vk::DescriptorBufferInfo exposure_info =
		    game_exposure ? vk::DescriptorBufferInfo {inputs.exposure_buffer, inputs.exposure_base, inputs.exposure_range} : stats_info;
		vk::WriteDescriptorSet writes[Bindings] {};
		for (uint32_t i = 0; i < Bindings; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = TypeOf(i);
			if (i == 9) writes[i].pBufferInfo = &stats_info;
			else if (i == 10) writes[i].pBufferInfo = &exposure_info;
			else writes[i].pImageInfo = &infos[i];
		}
		push.misc[0] = mode;
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_gv.layout, 0, Bindings, writes);
		command.pushConstants(g_gv.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch(gx, gy, 1);
	};
	run(1, 256 / 16, 144 / 16); // stats
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite, vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eTransferRead);
	{
		vk::BufferCopy region {slot * SlotUints * 4, slot * SlotUints * 4, SlotUints * 4};
		command.copyBuffer(vk::Buffer(g_gv.dev_buf), vk::Buffer(g_gv.host_buf), 1, &region);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
		        vk::AccessFlagBits2::eHostRead);
	}
	const bool show = g_gv.view != VStats;
	if (show) {
		run(0, (color.extent.width + 15) / 16, (color.extent.height + 15) / 16);
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead);
	}
	Report(have_gbuf, have_depth, have_tap);
	++g_gv.frame;
	if (!show) return false;
	inputs.color.image  = vk::Image(g_gv.out.image);
	inputs.color.view   = g_gv.out.view;
	inputs.color.layout = vk::ImageLayout::eGeneral;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = g_gv.out.usage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}

} // namespace Libs::Graphics::FrameGen
