// KYTY_RT_CAPTURE=1: see rt-capture.h. Never touches the game's images: a compute pass reads their views into a
// host-visible buffer; the files are written on a worker thread 8 frames later.
#include "local/rt-capture.h"

#include "gpu_tiler_shaders/rt_capture_f_spv.h"
#include "gpu_tiler_shaders/rt_capture_u_spv.h"
#include "local/godrays.h"
#include "local/hwrt-scene.h"
#include "local/rt-contact.h"
#include "local/rt-flicker.h"
#include "local/rt-local-lights.h"
#include "local/rt-reflections.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libs::Graphics::FrameGen::RtCapture {
namespace {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------------------------------------------------
// PNG (uncompressed deflate: stored blocks), 8-bit gray or RGB.
uint32_t Crc(const uint8_t* p, size_t n, uint32_t c = 0xffffffffu) {
	static uint32_t table[256];
	static bool     init = false;
	if (!init) {
		for (uint32_t i = 0; i < 256; ++i) {
			uint32_t k = i;
			for (int j = 0; j < 8; ++j) k = (k & 1u) ? 0xedb88320u ^ (k >> 1) : k >> 1;
			table[i] = k;
		}
		init = true;
	}
	for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xffu] ^ (c >> 8);
	return c;
}
void Be32(std::vector<uint8_t>& v, uint32_t x) {
	v.push_back(static_cast<uint8_t>(x >> 24)), v.push_back(static_cast<uint8_t>(x >> 16));
	v.push_back(static_cast<uint8_t>(x >> 8)), v.push_back(static_cast<uint8_t>(x));
}
void Chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data) {
	Be32(out, static_cast<uint32_t>(data.size()));
	std::vector<uint8_t> td(type, type + 4);
	td.insert(td.end(), data.begin(), data.end());
	out.insert(out.end(), td.begin(), td.end());
	Be32(out, Crc(td.data(), td.size()) ^ 0xffffffffu);
}
bool WritePng(const fs::path& path, uint32_t w, uint32_t h, int channels, const std::vector<uint8_t>& px) {
	std::vector<uint8_t> raw;
	raw.reserve(static_cast<size_t>(h) * (w * channels + 1));
	for (uint32_t y = 0; y < h; ++y) {
		raw.push_back(0);
		raw.insert(raw.end(), px.begin() + static_cast<size_t>(y) * w * channels, px.begin() + static_cast<size_t>(y + 1) * w * channels);
	}
	std::vector<uint8_t> z {0x78, 0x01};
	uint32_t             a = 1, b = 0;
	for (uint8_t c: raw) a = (a + c) % 65521u, b = (b + a) % 65521u;
	for (size_t off = 0; off < raw.size() || off == 0;) {
		const size_t n    = std::min<size_t>(65535, raw.size() - off);
		const bool   last = off + n >= raw.size();
		z.push_back(last ? 1 : 0);
		z.push_back(static_cast<uint8_t>(n)), z.push_back(static_cast<uint8_t>(n >> 8));
		z.push_back(static_cast<uint8_t>(~n)), z.push_back(static_cast<uint8_t>((~n) >> 8));
		z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
		off += n;
		if (last) break;
	}
	Be32(z, (b << 16) | a);
	std::vector<uint8_t> out {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
	std::vector<uint8_t> ihdr;
	Be32(ihdr, w), Be32(ihdr, h);
	ihdr.push_back(8), ihdr.push_back(channels == 1 ? 0 : 2), ihdr.push_back(0), ihdr.push_back(0), ihdr.push_back(0);
	Chunk(out, "IHDR", ihdr);
	Chunk(out, "IDAT", z);
	Chunk(out, "IEND", {});
	std::ofstream f(path, std::ios::binary);
	f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
	return f.good();
}

// ---------------------------------------------------------------------------------------------------------------------
struct Buf {
	VkBuffer      buffer = VK_NULL_HANDLE;
	VmaAllocation memory = nullptr;
	void*         mapped = nullptr;
	size_t        bytes  = 0;
};

// One image read back.
struct Item {
	std::string  name;   // file stem
	std::string  what;   // its description in frame.json
	std::string  format; // the view's format
	vk::Extent2D extent {};
	bool         uint_  = false; // 1 word a texel (else 4 floats)
	Buf          buf;
};

struct Set {
	std::string        kind; // still / moving
	fs::path           dir;
	uint64_t           frame = 0;
	std::vector<Item>  items;
	std::string        json; // the frame description (camera, lights, ...)
};

struct {
	bool                  failed = false;
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout    layout = nullptr;
	vk::Pipeline          pipe_f = nullptr, pipe_u = nullptr;
	vk::Sampler           point = nullptr;
	uint64_t              frame = 0;
	int                   state = 0; // 0 waiting for the world + still, 1 still taken, 2 moving taken, 3 done
	uint32_t              moving_run = 0;
	fs::path              root;
	std::array<std::array<float, 2>, 64> jitter {};
	uint32_t              jitter_n = 0;
	std::vector<std::pair<uint64_t, std::unique_ptr<Set>>> pending;
	std::atomic<int>      writers {0};
	std::string           log_lines;
	std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
} g_c;

fs::path Root() {
	fs::path base;
	if (const char* v = std::getenv("KYTY_RT_CAPTURE_DIR"); v != nullptr && *v != '\0') {
		base = v;
	} else {
#ifdef _WIN32
		wchar_t buf[MAX_PATH] {};
		const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
		base = n > 0 ? fs::path(buf).parent_path() / "logs" / "rtcapture" : fs::path("logs") / "rtcapture";
#else
		base = fs::path("logs") / "rtcapture";
#endif
	}
	const std::time_t t = std::time(nullptr);
	std::tm           tm {};
#ifdef _WIN32
	localtime_s(&tm, &t);
#else
	localtime_r(&t, &tm);
#endif
	char stamp[32];
	std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tm);
	return base / stamp;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g_c.pipe_f != nullptr && g_c.pipe_u != nullptr) return true;
	if (g_c.failed) return false;
	g_c.failed        = true;
	const auto device = graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, 2> bindings {};
	for (uint32_t i = 0; i < 2; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = i == 0 ? vk::DescriptorType::eCombinedImageSampler : vk::DescriptorType::eStorageBuffer;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = 2;
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_c.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, 16};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_c.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_c.layout) != vk::Result::eSuccess) return false;
	const auto make = [&](const uint32_t* code, size_t bytes, vk::Pipeline& out) {
		vk::ShaderModuleCreateInfo module_info {};
		module_info.codeSize    = bytes;
		module_info.pCode       = code;
		vk::ShaderModule module = nullptr;
		if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
		vk::ComputePipelineCreateInfo info {};
		info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
		info.stage.module = module;
		info.stage.pName  = "main";
		info.layout       = g_c.layout;
		const auto r      = device.createComputePipelines(nullptr, 1, &info, nullptr, &out);
		device.destroyShaderModule(module, nullptr);
		if (r != vk::Result::eSuccess) out = nullptr;
		return r == vk::Result::eSuccess;
	};
	if (!make(RT_CAPTURE_F_SPV, sizeof(RT_CAPTURE_F_SPV), g_c.pipe_f) || !make(RT_CAPTURE_U_SPV, sizeof(RT_CAPTURE_U_SPV), g_c.pipe_u)) return false;
	vk::SamplerCreateInfo sampler {};
	sampler.magFilter = sampler.minFilter = vk::Filter::eNearest;
	sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	if (device.createSampler(&sampler, nullptr, &g_c.point) != vk::Result::eSuccess) return false;
	g_c.failed = false;
	return true;
}

bool MakeBuf(GraphicContext& graphics, size_t bytes, Buf& b) {
	VkBufferCreateInfo info {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	info.size  = bytes;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &info, &host, &b.buffer, &b.memory, &mapped) != VK_SUCCESS) return b.buffer = VK_NULL_HANDLE, false;
	b.mapped = mapped.pMappedData, b.bytes = bytes;
	return b.mapped != nullptr;
}

// Records the readback of one view (a sampled view in `layout`).
void Read(GraphicContext& graphics, vk::CommandBuffer command, Set& set, const char* name, const char* what, vk::ImageView view,
          vk::ImageLayout layout, vk::Format format, vk::Extent2D extent, bool is_uint) {
	if (view == nullptr || extent.width == 0 || extent.height == 0) return;
	Item it;
	it.name = name, it.what = what, it.format = vk::to_string(format), it.extent = extent, it.uint_ = is_uint;
	const size_t texels = static_cast<size_t>(extent.width) * extent.height;
	if (!MakeBuf(graphics, texels * (is_uint ? 4 : 16), it.buf)) return;
	vk::DescriptorImageInfo  img {g_c.point, view, layout};
	vk::DescriptorBufferInfo bi {vk::Buffer(it.buf.buffer), 0, VK_WHOLE_SIZE};
	std::array<vk::WriteDescriptorSet, 2> w {};
	w[0].dstBinding = 0, w[0].descriptorCount = 1, w[0].descriptorType = vk::DescriptorType::eCombinedImageSampler, w[0].pImageInfo = &img;
	w[1].dstBinding = 1, w[1].descriptorCount = 1, w[1].descriptorType = vk::DescriptorType::eStorageBuffer, w[1].pBufferInfo = &bi;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, is_uint ? g_c.pipe_u : g_c.pipe_f);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g_c.layout, 0, 2, w.data());
	const int32_t push[4] {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 0, 0};
	command.pushConstants(g_c.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push);
	command.dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
	set.items.push_back(std::move(it));
}

// ---------------------------------------------------------------------------------------------------------------------
float Tm(float x) { return std::isfinite(x) && x > 0.0f ? std::sqrt(x / (1.0f + x)) : 0.0f; }
uint8_t U8(float x) { return static_cast<uint8_t>(std::clamp(x, 0.0f, 1.0f) * 255.0f + 0.5f); }

void WriteItem(const Set& set, const Item& it, const std::vector<uint32_t>& d, std::string& json, std::string& readme) {
	const uint32_t W = it.extent.width, H = it.extent.height;
	const size_t   n = static_cast<size_t>(W) * H;
	{
		std::ofstream f(set.dir / (it.name + ".bin"), std::ios::binary);
		f.write(reinterpret_cast<const char*>(d.data()), static_cast<std::streamsize>(d.size() * 4));
	}
	char buf[1024];
	std::snprintf(buf, sizeof(buf), "    {\"name\": \"%s\", \"what\": \"%s\", \"format\": \"%s\", \"width\": %u, \"height\": %u, \"bin\": \"%s.bin\", "
	              "\"bin_layout\": \"%s\"",
	              it.name.c_str(), it.what.c_str(), it.format.c_str(), W, H, it.name.c_str(),
	              it.uint_ ? "uint32 per texel, row-major" : "4 x float32 (r g b a as sampled) per texel, row-major");
	json += buf;
	if (it.uint_) {
		// per bit: share of texels with it set; per byte: a 16-bin histogram; one PNG per byte
		std::array<uint64_t, 32>                     bits {};
		std::array<std::array<uint64_t, 16>, 4>      hist {};
		std::array<std::vector<uint8_t>, 4>          bytes;
		for (auto& b: bytes) b.resize(n);
		uint64_t zero = 0;
		for (size_t i = 0; i < n; ++i) {
			const uint32_t v = d[i];
			if (v == 0) ++zero;
			for (int k = 0; k < 32; ++k) bits[k] += (v >> k) & 1u;
			for (int k = 0; k < 4; ++k) {
				const uint8_t b = static_cast<uint8_t>(v >> (8 * k));
				bytes[k][i] = b, ++hist[k][b >> 4];
			}
		}
		for (int k = 0; k < 4; ++k) WritePng(set.dir / (it.name + "_byte" + std::to_string(k) + ".png"), W, H, 1, bytes[k]);
		json += ", \"zero_share\": " + std::to_string(static_cast<double>(zero) / n) + ", \"bit_set_share\": [";
		std::string varying;
		for (int k = 0; k < 32; ++k) {
			const double s = static_cast<double>(bits[k]) / n;
			json += (k ? ", " : "") + std::to_string(s);
			if (s > 0.02 && s < 0.98) varying += " " + std::to_string(k);
		}
		json += "], \"byte_hist16\": [";
		for (int k = 0; k < 4; ++k) {
			json += k ? ", [" : "[";
			for (int j = 0; j < 16; ++j) json += (j ? ", " : "") + std::to_string(hist[k][j]);
			json += "]";
		}
		json += "]";
		readme += "  " + it.name + " (" + it.what + "): words zero on " + std::to_string(100.0 * zero / n).substr(0, 5) + " %; bits that vary (2-98 % set):" +
		          (varying.empty() ? " none" : varying) + "; PNGs " + it.name + "_byte0..3.png\n";
	} else {
		const auto F = [&](size_t i, int c) {
			float f;
			std::memcpy(&f, &d[i * 4 + c], 4);
			return f;
		};
		std::array<float, 4> lo {1e30f, 1e30f, 1e30f, 1e30f}, hi {-1e30f, -1e30f, -1e30f, -1e30f};
		uint64_t             at0 = 0, at1 = 0, bad = 0;
		for (size_t i = 0; i < n; ++i)
			for (int c = 0; c < 4; ++c) {
				const float f = F(i, c);
				if (!std::isfinite(f)) {
					++bad;
					continue;
				}
				lo[c] = std::min(lo[c], f), hi[c] = std::max(hi[c], f);
				if (c == 0) at0 += f == 0.0f, at1 += f == 1.0f;
			}
		std::snprintf(buf, sizeof(buf), ", \"min\": [%g, %g, %g, %g], \"max\": [%g, %g, %g, %g], \"r_exactly_0\": %g, \"r_exactly_1\": %g, \"non_finite\": %llu",
		              lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3], static_cast<double>(at0) / n, static_cast<double>(at1) / n,
		              static_cast<unsigned long long>(bad));
		json += buf;
		std::vector<uint8_t> px(n * 3);
		const bool depth = it.name.find("depth") != std::string::npos, motion = it.name == "motion";
		for (size_t i = 0; i < n; ++i) {
			if (depth) {
				const float z = F(i, 0), t = hi[0] > lo[0] ? (z - lo[0]) / (hi[0] - lo[0]) : 0.0f;
				px[i * 3] = px[i * 3 + 1] = px[i * 3 + 2] = U8(std::pow(std::clamp(t, 0.0f, 1.0f), 0.25f));
			} else if (motion) {
				px[i * 3] = U8(0.5f + F(i, 0) * 0.05f), px[i * 3 + 1] = U8(0.5f + F(i, 1) * 0.05f), px[i * 3 + 2] = 128;
			} else {
				for (int c = 0; c < 3; ++c) px[i * 3 + c] = U8(Tm(F(i, c)));
			}
		}
		WritePng(set.dir / (it.name + ".png"), W, H, 3, px);
		std::snprintf(buf, sizeof(buf), "  %s (%s, %s %ux%u): r %g..%g, g %g..%g; r == 0 on %.2f %%, r == 1 on %.2f %%%s\n", it.name.c_str(),
		              it.what.c_str(), it.format.c_str(), W, H, lo[0], hi[0], lo[1], hi[1], 100.0 * at0 / n, 100.0 * at1 / n,
		              depth ? (at0 > at1 ? " (the far plane at 0: reversed Z)" : at1 > at0 ? " (the far plane at 1: standard Z)" : "") : "");
		readme += buf;
	}
	json += "}";
}

void WriteSet(std::unique_ptr<Set> set, std::vector<std::vector<uint32_t>> data) {
	std::error_code ec;
	fs::create_directories(set->dir, ec);
	std::string json = "{\n  \"set\": \"" + set->kind + "\",\n" + set->json + ",\n  \"images\": [\n";
	std::string readme;
	for (size_t i = 0; i < set->items.size(); ++i) {
		WriteItem(*set, set->items[i], data[i], json, readme);
		json += i + 1 < set->items.size() ? ",\n" : "\n";
	}
	json += "  ]\n}\n";
	std::ofstream(set->dir / "frame.json") << json;
	std::ofstream(set->dir / "README.txt") << "KYTY_RT_CAPTURE, set '" << set->kind << "' (frame " << set->frame << ")\n\nImages:\n" << readme
	                                       << "\nSee frame.json for the camera, the jitter, the sun, the lights and the full statistics.\n";
	std::printf("RTCAPTURE: %s set written to %s (%zu images)\n", set->kind.c_str(), set->dir.string().c_str(), set->items.size());
	std::fflush(stdout);
	--g_c.writers;
}

std::string M16(const float* m) {
	std::string s = "[";
	for (int i = 0; i < 16; ++i) s += (i ? ", " : "") + std::to_string(m[i]);
	return s + "]";
}

std::string Describe(const UpscaleInputs& in) {
	std::string j;
	char        b[512];
	std::snprintf(b, sizeof(b), "  \"render\": [%u, %u], \"output\": [%u, %u],\n  \"near\": %g, \"far\": %g, \"motion_scale\": [%g, %g], \"reset\": %s,\n",
	              in.color.extent.width, in.color.extent.height, in.output.extent.width, in.output.extent.height, in.near_plane, in.far_plane,
	              in.motion_scale_x, in.motion_scale_y, in.reset ? "true" : "false");
	j += b;
	j += "  \"formats\": {\"color\": \"" + vk::to_string(in.color.format) + "\", \"depth\": \"" + vk::to_string(in.depth.format) +
	     "\", \"motion\": \"" + vk::to_string(in.motion.format) + "\", \"output\": \"" + vk::to_string(in.output.format) + "\"},\n";
	j += "  \"view_to_clip_rows\": " + M16(in.view_to_clip.data()) + ",\n";
	std::array<float, 16> vp {};
	j += std::string("  \"scene_view_projection\": ") + (HwScene::SceneCamera(vp) ? M16(vp.data()) : "null") + ",\n";
	j += "  \"jitter_px_last_frames\": [";
	const uint32_t cnt = std::min<uint32_t>(g_c.jitter_n, 64);
	for (uint32_t i = 0; i < cnt; ++i) {
		const auto& v = g_c.jitter[(g_c.jitter_n - cnt + i) % 64];
		std::snprintf(b, sizeof(b), "%s[%g, %g]", i ? ", " : "", v[0], v[1]);
		j += b;
	}
	j += "],\n";
	std::array<float, 3> sd {};
	float                sw = 0, sr = 0;
	bool                 ex = false;
	if (RtContactGetSun(sd, sw, sr, ex)) {
		std::snprintf(b, sizeof(b), "  \"sun\": {\"dir_to_sun_world\": [%g, %g, %g], \"weight\": %g, \"radiance_over_ambient\": %g, \"rt_exact\": %s},\n", sd[0],
		              sd[1], sd[2], sw, sr, ex ? "true" : "false");
		j += b;
	} else {
		j += "  \"sun\": null,\n";
	}
	j += "  \"volumetric\": " + GodraysDescribe() + ",\n"; // (vol-1027: the god rays' sun as fed, its weight, the march path)
	std::vector<RtLocal::LiveLight> lights;
	if (RtLocal::Wanted()) RtLocal::LiveLights(lights, 120, 1);
	j += "  \"local_lights\": [";
	for (size_t i = 0; i < lights.size(); ++i) {
		const auto& l = lights[i];
		std::snprintf(b, sizeof(b), "%s\n    {\"pos\": [%g, %g, %g], \"fwd\": [%g, %g, %g], \"tan_half\": %g, \"reach\": %g, \"point\": %s, \"seen\": %u, \"age\": %llu}",
		              i ? "," : "", l.pos[0], l.pos[1], l.pos[2], l.fwd[0], l.fwd[1], l.fwd[2], l.tan_half, l.far_d, l.point ? "true" : "false", l.seen,
		              static_cast<unsigned long long>(l.age));
		j += b;
	}
	j += "],\n";
	uint64_t wd = 0, wf = 0;
	uint32_t wk = 0, wb = 0;
	HwScene::WaterCounters(wd, wk, wb, wf);
	std::snprintf(b, sizeof(b), "  \"tlas\": {\"active\": %s, \"this_frame\": %s, \"water_draws_noted\": %llu, \"water_meshes_known\": %u, \"water_meshes_built\": %u},\n",
	              HwScene::Active() ? "true" : "false", in.hwrt_tlas != 0 ? "true" : "false", static_cast<unsigned long long>(wd), wk, wb);
	j += b;
	RtReflGbuffer gb;
	const bool    have = RtReflGetGbuffer(gb);
	std::snprintf(b, sizeof(b), "  \"gbuffer_copy\": %s, \"material_target\": %s, \"material_channel\": %d, \"material_is_gloss\": %s",
	              have ? "true" : "false", in.material.view != nullptr ? ("\"" + vk::to_string(in.material.format) + "\"").c_str() : "null",
	              in.material_channel, in.material_gloss ? "true" : "false");
	j += b;
	return j;
}

} // namespace

bool Enabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RT_CAPTURE");
		const bool  b = v != nullptr && v[0] == '1';
		if (b) {
			std::printf("RTCAPTURE: on (KYTY_RT_CAPTURE=1): in the world, hold the camera still ~3 s (the still set), then walk ~2 s (the moving set)\n");
			std::fflush(stdout);
		}
		return b;
	}();
	return on;
}

void OnFrame(vk::CommandBuffer command, const UpscaleInputs& inputs) {
	if (!Enabled() || inputs.graphics == nullptr || command == nullptr) return;
	++g_c.frame;
	g_c.jitter[g_c.jitter_n % 64] = {inputs.jitter_x, inputs.jitter_y};
	++g_c.jitter_n;
	GraphicContext& graphics = *inputs.graphics;
	// finished sets: 8 frames after their copy, onto a worker thread
	for (auto& p: g_c.pending) {
		if (p.second == nullptr || g_c.frame < p.first + 8) continue;
		std::vector<std::vector<uint32_t>> data;
		for (auto& it: p.second->items) {
			(void)vmaInvalidateAllocation(graphics.allocator, it.buf.memory, 0, VK_WHOLE_SIZE);
			const auto* w = static_cast<const uint32_t*>(it.buf.mapped);
			data.emplace_back(w, w + it.buf.bytes / 4);
			vmaDestroyBuffer(graphics.allocator, it.buf.buffer, it.buf.memory);
			it.buf = {};
		}
		++g_c.writers;
		std::thread(WriteSet, std::move(p.second), std::move(data)).detach();
	}
	g_c.pending.erase(std::remove_if(g_c.pending.begin(), g_c.pending.end(), [](const auto& p) { return p.second == nullptr; }), g_c.pending.end());
	if (g_c.state == 2 && g_c.pending.empty() && g_c.writers.load() == 0) {
		g_c.state = 3;
		std::ofstream(g_c.root / "README.txt")
		    << "KYTY_RT_CAPTURE: the RT passes' view of the game's frame.\n"
		       "still/ : camera held still; moving/ : camera moving. Each has frame.json (camera, jitter, sun, local lights, TLAS,\n"
		       "formats, statistics), README.txt (a summary per image), .bin raw data (exact bits) and PNG views.\n"
		       "gbuf0..3: the game's packed G-buffer layers (R32_UINT) as copied at its deferred lighting; one PNG per byte.\n"
		       "Not in this capture (needs hooks inside the game's command stream): the stencil, the shadow cascades / spot-shadow\n"
		       "pool, the fog / volumetric buffers, the game's SSR image and the GPU pass timeline; see the RTSHADOW / RT reflections\n"
		       "lines of the log for what the emulator already identified of those.\n";
		std::printf("RTCAPTURE: done %s\n", g_c.root.string().c_str());
		std::fflush(stdout);
		return;
	}
	if (g_c.state >= 2) return;
	// when: in the world (the G-buffer copy), still 2 s (state 0) / moving 0.5 s (state 1)
	RtReflGbuffer gb;
	const bool    have_gb = RtReflGetGbuffer(gb);
	const bool    world   = have_gb || inputs.hwrt_tlas != 0;
	const auto    since   = std::chrono::steady_clock::now() - g_c.started;
	if (!world || since < std::chrono::seconds(5)) return;
	const uint32_t still = RtFlicker::StillFrames();
	g_c.moving_run       = still == 0 ? g_c.moving_run + 1 : 0;
	const bool take      = (g_c.state == 0 && still >= 60) || (g_c.state == 1 && g_c.moving_run >= 15);
	if (!take || !EnsurePipeline(graphics)) return;
	if (g_c.root.empty()) g_c.root = Root();
	auto set   = std::make_unique<Set>();
	set->kind  = g_c.state == 0 ? "still" : "moving";
	set->dir   = g_c.root / set->kind;
	set->frame = g_c.frame;
	set->json  = Describe(inputs);
	// (the game's images as the upscaler gets them, our G-buffer copy in its read-only layout)
	vk::MemoryBarrier2 mb {};
	mb.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	mb.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	mb.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	mb.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
	vk::DependencyInfo dep {};
	dep.memoryBarrierCount = 1;
	dep.pMemoryBarriers    = &mb;
	command.pipelineBarrier2(dep);
	Read(graphics, command, *set, "color", "the lit HDR colour before the upscaler (pre-exposure)", inputs.color.view, inputs.color.layout,
	     inputs.color.format, inputs.color.extent, false);
	Read(graphics, command, *set, "depth", "the depth the upscaler gets", inputs.depth.view, inputs.depth.layout, inputs.depth.format,
	     inputs.depth.extent, false);
	Read(graphics, command, *set, "motion", "the motion vectors (x motion_scale = render pixels to the previous frame)", inputs.motion.view,
	     inputs.motion.layout, inputs.motion.format, inputs.motion.extent, false);
	if (have_gb) {
		for (uint32_t i = 0; i < 4; ++i) {
			const std::string nm = "gbuf" + std::to_string(i);
			Read(graphics, command, *set, nm.c_str(), "a packed G-buffer layer (RT0..3)", gb.views[i], vk::ImageLayout::eShaderReadOnlyOptimal,
			     vk::Format::eR32Uint, gb.extent, true);
		}
		if (gb.depth != nullptr)
			Read(graphics, command, *set, "gbuf_depth", "the depth at the deferred lighting (before water / transparents)", gb.depth, gb.depth_layout,
			     vk::Format::eD32Sfloat, gb.extent, false);
	}
	if (inputs.material.view != nullptr)
		Read(graphics, command, *set, "material", "the target KYTY_SSR_GBUF names as the material (roughness / gloss)", inputs.material.view,
		     inputs.material.layout, inputs.material.format, inputs.material.extent, false);
	mb.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	mb.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	mb.dstStageMask  = vk::PipelineStageFlagBits2::eHost | vk::PipelineStageFlagBits2::eAllCommands;
	mb.dstAccessMask = vk::AccessFlagBits2::eHostRead | vk::AccessFlagBits2::eMemoryRead;
	command.pipelineBarrier2(dep);
	std::printf("RTCAPTURE: %s set recorded (frame %llu, %zu images%s)\n", set->kind.c_str(), static_cast<unsigned long long>(g_c.frame),
	            set->items.size(), have_gb ? "" : "; no G-buffer copy: turn RT reflections on for it");
	std::fflush(stdout);
	g_c.pending.emplace_back(g_c.frame, std::move(set));
	++g_c.state;
}

} // namespace Libs::Graphics::FrameGen::RtCapture
