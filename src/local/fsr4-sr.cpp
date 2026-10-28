#include "fsr4-sr.h"
// AMD FSR 4 (source-v07 INT8 / DOT4 model) on the PS5 Pro path: see fsr4-sr.h. The runtime is ours (KytyPS5, GPL-2.0);
// the model's SPIR-V and weights are AMD's (MIT, fsr4_shaders/LICENSE-FSR4-v07.txt). The dispatch sizes, the constant
// buffer and RCAS's constants follow AMD's FidelityFX SDK FSR 4 provider (MIT) for its non-WMMA (DOT4) path.

#include "frame-gen.h"
#include "local-platform.h"
#include "gpu_tiler_shaders/fsr4_prep_spv.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

namespace Libs::Graphics::Fsr4 {

namespace {

const char* const PresetNames[] = {"native", "quality", "balanced", "performance", "ultraperf", "drs"};
const char* const PresetTitles[] = {"FSR 4 Native AA", "FSR 4 Quality", "FSR 4 Balanced", "FSR 4 Performance",
                                    "FSR 4 Ultra Performance", "FSR 4 Dynamic (DRS)"};
constexpr int PresetAuto = -1;

// AMD's scratch sizes (shader_selector.cpp GetScratchSize: the network's tensors for that output size).
constexpr vk::DeviceSize Scratch1080 = 20880256;
constexpr vk::DeviceSize Scratch2160 = 83232256;

bool EnvOn(const char* name, bool fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') return fallback;
	return value[0] != '0';
}

int RequestedPreset() {
	static const int preset = [] {
		const char* value = std::getenv("KYTY_FSR4_PRESET");
		const std::string v = value != nullptr ? value : "";
		if (v.empty()) return 1;
		if (v == "auto") return PresetAuto;
		if (v == "native" || v == "nativeaa" || v == "aa") return 0;
		if (v == "ultraperformance" || v == "ultra_performance") return 4;
		for (int i = 0; i < 6; ++i)
			if (v == PresetNames[i]) return i;
		std::printf("FSR 4: KYTY_FSR4_PRESET=%s unknown (native, quality, balanced, performance, ultraperf, drs, auto): quality\n",
		            v.c_str());
		return 1;
	}();
	return preset;
}

// AMD's own choice by the ratio (ffx_provider_fsr4 Dispatch).
int PresetForRatio(float ratio) {
	int preset = 0;
	if (ratio >= 1.50f) preset = 1;
	if (ratio >= 1.69f) preset = 2;
	if (ratio >= 1.99f) preset = 3;
	if (ratio >= 2.99f) preset = 4;
	return preset;
}

std::filesystem::path AssetDir() {
	if (const char* dir = std::getenv("KYTY_FSR4_DIR"); dir != nullptr && *dir != '\0') return dir;
#if defined(_WIN32)
	wchar_t exe[MAX_PATH] {};
	GetModuleFileNameW(nullptr, exe, MAX_PATH);
	return std::filesystem::path(exe).parent_path() / "fsr4_shaders";
#else
	std::error_code error;
	const auto exe = std::filesystem::read_symlink("/proc/self/exe", error);
	return (error ? std::filesystem::current_path() : exe.parent_path()) / "fsr4_shaders";
#endif
}

bool ReadFile(const std::filesystem::path& path, std::vector<char>& data) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) return false;
	const auto size = static_cast<size_t>(file.tellg());
	data.resize(size);
	file.seekg(0);
	return static_cast<bool>(file.read(data.data(), static_cast<std::streamsize>(size))) || size == 0;
}

// ---- A minimal SPIR-V reflection: each descriptor's (set, binding, type, name), the entry points, the local size ----
enum class Kind { SampledImage, StorageImage, Sampler, CombinedImageSampler, UniformBuffer, StorageBuffer };
struct Resource {
	uint32_t    set = 0, binding = 0;
	Kind        kind = Kind::SampledImage;
	std::string name;
};
struct Reflection {
	std::vector<std::string> entries;
	std::array<uint32_t, 3>  local {0, 0, 0};
	std::vector<Resource>    resources;
	bool                     push_constants = false;
	bool                     derivatives = false, dot = false;
};

std::string SpvString(const uint32_t* words, size_t count) {
	std::string s(reinterpret_cast<const char*>(words), count * 4);
	return s.substr(0, s.find('\0'));
}

bool Reflect(const std::vector<char>& data, Reflection& out) {
	if (data.size() < 20 || data.size() % 4 != 0) return false;
	std::vector<uint32_t> w(data.size() / 4);
	std::memcpy(w.data(), data.data(), data.size());
	if (w[0] != 0x07230203u) return false;
	struct Type {
		uint32_t op = 0;
		std::vector<uint32_t> args;
	};
	std::map<uint32_t, Type>        types;
	std::map<uint32_t, std::string> names;
	std::map<uint32_t, uint32_t>    sets, bindings;
	std::map<uint32_t, bool>        buffer_block;
	struct Var {
		uint32_t id, ptr, sc;
	};
	std::vector<Var> vars;
	for (size_t i = 5; i < w.size();) {
		const uint32_t count = w[i] >> 16, op = w[i] & 0xFFFFu;
		if (count == 0 || i + count > w.size()) return false;
		const uint32_t* a = &w[i + 1];
		const size_t    n = count - 1;
		switch (op) {
			case 5: if (n >= 1) names[a[0]] = SpvString(a + 1, n - 1); break;          // OpName
			case 15: if (n >= 3) out.entries.push_back(SpvString(a + 2, n - 2)); break; // OpEntryPoint
			case 16: if (n >= 5 && a[1] == 17) out.local = {a[2], a[3], a[4]}; break; // OpExecutionMode LocalSize
			case 17: // OpCapability
				if (n >= 1 && (a[0] == 5350 || a[0] == 5288)) out.derivatives = true;
				if (n >= 1 && (a[0] == 6019 || a[0] == 6018)) out.dot = true;
				break;
			case 71: // OpDecorate
				if (n >= 3 && a[1] == 34) sets[a[0]] = a[2];
				if (n >= 3 && a[1] == 33) bindings[a[0]] = a[2];
				if (n >= 2 && a[1] == 3) buffer_block[a[0]] = true;
				break;
			case 25: case 26: case 27: case 28: case 29: case 30: case 32: // image, sampler, sampled image, arrays, struct, pointer
				if (n >= 1) types[a[0]] = {op, std::vector<uint32_t>(a + 1, a + n)};
				break;
			case 59: if (n >= 3) vars.push_back({a[1], a[0], a[2]}); break; // OpVariable
			default: break;
		}
		i += count;
	}
	for (const auto& v: vars) {
		if (v.sc == 9) {
			out.push_constants = true;
			continue;
		}
		if (v.sc != 0 && v.sc != 2 && v.sc != 12) continue; // UniformConstant, Uniform, StorageBuffer
		auto ptr = types.find(v.ptr);
		if (ptr == types.end() || ptr->second.op != 32 || ptr->second.args.size() < 2) continue;
		uint32_t t = ptr->second.args[1];
		while (types.count(t) != 0 && (types[t].op == 28 || types[t].op == 29) && !types[t].args.empty()) t = types[t].args[0];
		const auto found = types.find(t);
		if (found == types.end()) continue;
		Resource r;
		r.set     = sets.count(v.id) != 0 ? sets[v.id] : 0;
		r.binding = bindings.count(v.id) != 0 ? bindings[v.id] : 0;
		r.name    = names.count(v.id) != 0 ? names[v.id] : "";
		switch (found->second.op) {
			case 25: r.kind = found->second.args.size() > 5 && found->second.args[5] == 2 ? Kind::StorageImage : Kind::SampledImage; break;
			case 26: r.kind = Kind::Sampler; break;
			case 27: r.kind = Kind::CombinedImageSampler; break;
			case 30: r.kind = v.sc == 12 || buffer_block.count(t) != 0 ? Kind::StorageBuffer : Kind::UniformBuffer; break;
			default: continue;
		}
		out.resources.push_back(r);
	}
	std::sort(out.resources.begin(), out.resources.end(),
	          [](const Resource& x, const Resource& y) { return x.set != y.set ? x.set < y.set : x.binding < y.binding; });
	return true;
}

bool KindFromManifest(const std::string& type, Kind& kind) {
	if (type == "sampled_image") kind = Kind::SampledImage;
	else if (type == "storage_image") kind = Kind::StorageImage;
	else if (type == "sampler") kind = Kind::Sampler;
	else if (type == "combined_image_sampler") kind = Kind::CombinedImageSampler;
	else if (type == "uniform_buffer") kind = Kind::UniformBuffer;
	else if (type == "storage_buffer") kind = Kind::StorageBuffer;
	else return false;
	return true;
}

vk::DescriptorType DescriptorType(Kind kind) {
	switch (kind) {
		case Kind::SampledImage: return vk::DescriptorType::eSampledImage;
		case Kind::StorageImage: return vk::DescriptorType::eStorageImage;
		case Kind::Sampler: return vk::DescriptorType::eSampler;
		case Kind::CombinedImageSampler: return vk::DescriptorType::eCombinedImageSampler;
		case Kind::UniformBuffer: return vk::DescriptorType::eUniformBuffer;
		case Kind::StorageBuffer: return vk::DescriptorType::eStorageBuffer;
	}
	return vk::DescriptorType::eSampledImage;
}

// One module of the preset: its file, entry point and descriptors (the manifest's, checked against the SPIR-V).
struct Module {
	std::filesystem::path path;
	std::string           entry;
	std::array<uint32_t, 3> local {0, 0, 0};
	std::vector<Resource> resources;
	std::vector<char>     code;
};

// Loads `name` (the manifest's artifact) and checks it: size, entry point, local size, descriptors. `file` overrides the
// file read (opt/ variants, which the manifest does not list: same interface, checked the same way).
bool LoadModule(const nlohmann::json& artifacts, const std::filesystem::path& dir, const std::string& name,
                const std::filesystem::path& file, Module& out, std::string& why) {
	const auto it = artifacts.find(name);
	if (it == artifacts.end()) {
		why = name + ": not in the manifest";
		return false;
	}
	const auto& art = *it;
	out.path = file.empty() ? dir / name : file;
	if (!ReadFile(out.path, out.code)) {
		why = out.path.string() + ": missing";
		return false;
	}
	if (file.empty() && art.value("size", static_cast<uint64_t>(0)) != out.code.size()) {
		why = name + ": size " + std::to_string(out.code.size()) + ", the manifest says " +
		      std::to_string(art.value("size", static_cast<uint64_t>(0)));
		return false;
	}
	out.entry = art.value("entry_point", std::string("main"));
	const auto local = art.value("local_size", std::vector<uint32_t> {});
	if (local.size() == 3) out.local = {local[0], local[1], local[2]};
	out.resources.clear();
	for (const auto& r: art.value("resources", nlohmann::json::array())) {
		Resource res;
		res.set     = r.value("set", 0u);
		res.binding = r.value("binding", 0u);
		res.name    = r.value("name", std::string());
		if (!KindFromManifest(r.value("type", std::string()), res.kind)) {
			why = name + ": descriptor type " + r.value("type", std::string()) + " unknown";
			return false;
		}
		if (res.set != 0) {
			why = name + ": descriptor set " + std::to_string(res.set) + " (one set expected)";
			return false;
		}
		out.resources.push_back(res);
	}
	std::sort(out.resources.begin(), out.resources.end(), [](const Resource& x, const Resource& y) { return x.binding < y.binding; });
	Reflection reflected;
	if (!Reflect(out.code, reflected)) {
		why = out.path.string() + ": not SPIR-V";
		return false;
	}
	if (std::find(reflected.entries.begin(), reflected.entries.end(), out.entry) == reflected.entries.end()) {
		why = name + ": no entry point " + out.entry;
		return false;
	}
	if (reflected.local != out.local) {
		why = name + ": local size differs from the manifest";
		return false;
	}
	if (reflected.push_constants) {
		why = name + ": push constants (none expected)";
		return false;
	}
	if (reflected.resources.size() != out.resources.size()) {
		why = name + ": " + std::to_string(reflected.resources.size()) + " descriptors, the manifest lists " +
		      std::to_string(out.resources.size());
		return false;
	}
	for (size_t i = 0; i < out.resources.size(); ++i) {
		const auto& a = reflected.resources[i];
		const auto& b = out.resources[i];
		if (a.set != b.set || a.binding != b.binding || a.kind != b.kind) {
			why = name + ": descriptor " + std::to_string(b.binding) + " (" + b.name + ") differs from the SPIR-V";
			return false;
		}
	}
	return true;
}

// rcas.spv / fsr4-prep: not in the manifest, their descriptors from the SPIR-V itself.
bool LoadReflected(const std::filesystem::path& path, Module& out, std::string& why) {
	out.path = path;
	if (!ReadFile(path, out.code)) {
		why = path.string() + ": missing";
		return false;
	}
	Reflection reflected;
	if (!Reflect(out.code, reflected) || reflected.entries.empty()) {
		why = path.string() + ": not SPIR-V";
		return false;
	}
	out.entry     = reflected.entries[0];
	out.local     = reflected.local;
	out.resources = reflected.resources;
	return true;
}

// ---- Device support (device creation) ----
struct DeviceSupport {
	bool checked = false;
	bool dot = false, int8 = false, int16 = false, float16 = false, derivatives = false, accelerated = false;
	bool derivatives_ext = false;
	VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivatives_features {};
} g_dev;

// ---- Runtime state ----
struct Pipe {
	Module                  module;
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
};
struct Buffer {
	VkBuffer      buffer = VK_NULL_HANDLE;
	VmaAllocation memory = VK_NULL_HANDLE;
	vk::DeviceSize size  = 0;
};
struct Target {
	VulkanImage   image;
	vk::ImageView view = nullptr;
};

struct State {
	bool        ready_checked = false, ready = false, failed = false;
	int         preset = 1;  // the model in use
	bool        res2160 = true;
	std::filesystem::path dir;
	nlohmann::json        manifest;
	std::vector<char>     initializers, pre_weights;
	std::array<Pipe, 14>  model {}; // pre, pass1..12, post
	Pipe                  rcas, prep;
	bool                  built = false;
	vk::Sampler           linear = nullptr, nearest = nullptr;
	Buffer                scratch, init, weights, constants, rcas_constants;
	bool                  uploaded = false;
	vk::Extent2D          size {};   // our images' (the aligned upscale size)
	vk::Extent2D          render {}; // the depth copy's
	Target                history, reprojected, recurrent, rcas_temp, out16, exposure, depth;
	bool                  first = true;
	uint64_t              frames = 0;
} g;

void DestroyPipe(vk::Device device, Pipe& p) {
	if (p.pipeline) device.destroyPipeline(p.pipeline);
	if (p.layout) device.destroyPipelineLayout(p.layout);
	if (p.set_layout) device.destroyDescriptorSetLayout(p.set_layout);
	p.pipeline   = nullptr;
	p.layout     = nullptr;
	p.set_layout = nullptr;
}

bool BuildPipe(vk::Device device, Pipe& p) {
	std::vector<vk::DescriptorSetLayoutBinding> bindings;
	for (const auto& r: p.module.resources) {
		vk::DescriptorSetLayoutBinding b {};
		b.binding         = r.binding;
		b.descriptorType  = DescriptorType(r.kind);
		b.descriptorCount = 1;
		b.stageFlags      = vk::ShaderStageFlagBits::eCompute;
		bindings.push_back(b);
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = static_cast<uint32_t>(bindings.size());
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &p.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, 32};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts    = &p.set_layout;
	if (p.module.path.empty()) { // (fsr4-prep: our own shader, with push constants)
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges    = &push;
	}
	if (device.createPipelineLayout(&layout_info, nullptr, &p.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = p.module.code.size();
	module_info.pCode    = reinterpret_cast<const uint32_t*>(p.module.code.data());
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = p.module.entry.c_str();
	pipeline_info.layout       = p.layout;
	const auto created = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &p.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		p.pipeline = nullptr;
		return false;
	}
	return true;
}

std::string ModelName(int preset) {
	return std::string("fsr4_model_v07_i8_") + PresetNames[preset];
}

// The preset's manifest, weights and modules (the 1080 or 2160 variant), each checked against the manifest.
bool LoadPreset(int preset, bool res2160, std::string& why) {
	const std::string model = ModelName(preset);
	std::vector<char> text;
	if (!ReadFile(g.dir / (model + "_shader_manifest.json"), text)) {
		why = (g.dir / (model + "_shader_manifest.json")).string() + ": missing";
		return false;
	}
	g.manifest = nlohmann::json::parse(text.begin(), text.end(), nullptr, false);
	if (g.manifest.is_discarded() || !g.manifest.contains("artifacts")) {
		why = model + "_shader_manifest.json: not a manifest";
		return false;
	}
	const auto& perm = g.manifest.value("permutation", nlohmann::json::object());
	if (perm.value("FSR4_ENABLE_DOT4", 0) != 1 || perm.value("WMMA_ENABLED", 1) != 0 || perm.value("FFX_MLSR_LOW_RES_MV", 0) != 1 ||
	    perm.value("FFX_MLSR_AUTOEXPOSURE_ENABLED", 1) != 0 || perm.value("FFX_MLSR_COLORSPACE", 1) != 0 ||
	    perm.value("FFX_MLSR_JITTERED_MOTION_VECTORS", 1) != 0) {
		why = model + ": a permutation this runtime does not drive (DOT4, no WMMA, low-res linear MV, input exposure)";
		return false;
	}
	const auto& artifacts = g.manifest["artifacts"];
	for (const char* blob: {"initializers.bin", "pre_weights.bin"}) {
		const std::string name = model + "_" + blob;
		auto& data = std::string_view(blob) == "initializers.bin" ? g.initializers : g.pre_weights;
		if (!ReadFile(g.dir / name, data)) {
			why = name + ": missing";
			return false;
		}
		const auto it = artifacts.find(name);
		if (it == artifacts.end() || it->value("size", static_cast<uint64_t>(0)) != data.size()) {
			why = name + ": its size does not match the manifest";
			return false;
		}
	}
	if (g.pre_weights.size() != 1024) {
		why = model + "_pre_weights.bin: 1024 bytes expected (cbPass_Weights)";
		return false;
	}
	const bool opt = EnvOn("KYTY_FSR4_OPT", false);
	const std::string res = res2160 ? "2160" : "1080";
	for (int i = 0; i < 14; ++i) {
		const std::string pass = i == 0 ? "pre" : i == 13 ? "post" : "pass" + std::to_string(i);
		const std::string name = model + "_" + res + "_" + pass + ".spv";
		std::filesystem::path file;
		if (opt && (i == 11 || i == 13) && std::filesystem::exists(g.dir / "opt" / name)) file = g.dir / "opt" / name;
		if (!LoadModule(artifacts, g.dir, name, file, g.model[i].module, why)) return false;
	}
	if (!LoadReflected(g.dir / "rcas.spv", g.rcas.module, why)) return false;
	return true;
}

bool CreateBuffer(GraphicContext& graphics, Buffer& b, vk::DeviceSize size, VkBufferUsageFlags usage) {
	if (b.buffer != VK_NULL_HANDLE && b.size >= size) return true;
	if (b.buffer != VK_NULL_HANDLE) {
		(void)graphics.device.waitIdle();
		vmaDestroyBuffer(graphics.allocator, b.buffer, b.memory);
		b.buffer = VK_NULL_HANDLE;
	}
	VkBufferCreateInfo create {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	create.size  = size;
	create.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo dev {};
	dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(graphics.allocator, &create, &dev, &b.buffer, &b.memory, nullptr) != VK_SUCCESS) {
		b.buffer = VK_NULL_HANDLE;
		return false;
	}
	b.size = size;
	return true;
}

void DestroyTarget(GraphicContext& graphics, Target& t) {
	if (t.view) graphics.device.destroyImageView(t.view);
	t.view = nullptr;
	if (t.image.image) graphics.DeleteImage(t.image);
	t.image.image = nullptr;
}

bool CreateTarget(GraphicContext& graphics, Target& t, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage) {
	DestroyTarget(graphics, t);
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = usage;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, t.image)) {
		t.image.image = nullptr;
		return false;
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = t.image.image;
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = format;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &t.view) != vk::Result::eSuccess) {
		t.view = nullptr;
		return false;
	}
	t.image.state = {};
	return true;
}

bool Build(GraphicContext& graphics, int preset, bool res2160) {
	const auto device = graphics.device;
	const auto start  = std::chrono::steady_clock::now();
	if (g.built) {
		(void)device.waitIdle();
		for (auto& p: g.model) DestroyPipe(device, p);
		DestroyPipe(device, g.rcas);
		g.built = false;
	}
	std::string why;
	if (!LoadPreset(preset, res2160, why)) {
		std::printf("FSR 4: %s: off (stretched)\n", why.c_str());
		std::fflush(stdout);
		return false;
	}
	for (auto& p: g.model) {
		if (!BuildPipe(device, p)) {
			std::printf("FSR 4: %s: its pipeline was not created: off (stretched)\n", p.module.path.filename().string().c_str());
			std::fflush(stdout);
			return false;
		}
	}
	if (!BuildPipe(device, g.rcas)) return false;
	if (g.prep.pipeline == nullptr) {
		g.prep.module.code.assign(reinterpret_cast<const char*>(FSR4_PREP_SPV), reinterpret_cast<const char*>(FSR4_PREP_SPV) + sizeof(FSR4_PREP_SPV));
		g.prep.module.entry = "main";
		Reflection reflected;
		if (!Reflect(g.prep.module.code, reflected)) return false;
		g.prep.module.resources = reflected.resources;
		if (!BuildPipe(device, g.prep)) return false;
	}
	if (g.linear == nullptr) {
		vk::SamplerCreateInfo sampler {};
		sampler.magFilter    = vk::Filter::eLinear;
		sampler.minFilter    = vk::Filter::eLinear;
		sampler.mipmapMode   = vk::SamplerMipmapMode::eNearest;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		if (device.createSampler(&sampler, nullptr, &g.linear) != vk::Result::eSuccess) return false;
		sampler.magFilter = vk::Filter::eNearest;
		sampler.minFilter = vk::Filter::eNearest;
		if (device.createSampler(&sampler, nullptr, &g.nearest) != vk::Result::eSuccess) return false;
	}
	if (!CreateBuffer(graphics, g.scratch, res2160 ? Scratch2160 : Scratch1080, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
	    !CreateBuffer(graphics, g.init, std::max<vk::DeviceSize>(g.initializers.size(), 256), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
	    !CreateBuffer(graphics, g.weights, 1024, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
	    !CreateBuffer(graphics, g.constants, 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) || // (also fsr4-prep's stand-in exposure buffer)
	    !CreateBuffer(graphics, g.rcas_constants, 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) {
		std::printf("FSR 4: its buffers were not created (%.1f MB scratch): off (stretched)\n",
		            static_cast<double>(res2160 ? Scratch2160 : Scratch1080) / 1e6);
		std::fflush(stdout);
		return false;
	}
	g.uploaded = false;
	g.preset   = preset;
	g.res2160  = res2160;
	g.built    = true;
	g.first    = true;
	const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
	std::printf("FSR 4: %s model (%s, %s variant%s) loaded from %s: 14 passes + RCAS, descriptors as the manifest, %.0f ms\n",
	            PresetTitles[preset], ModelName(preset).c_str(), res2160 ? "2160" : "1080",
	            EnvOn("KYTY_FSR4_OPT", false) ? ", opt pass 11 + post" : "", g.dir.string().c_str(), ms);
	std::fflush(stdout);
	return true;
}

uint16_t HalfBits(float value) {
	uint32_t f = 0;
	std::memcpy(&f, &value, 4);
	const uint32_t sign = (f >> 16) & 0x8000u;
	const int32_t  exp  = static_cast<int32_t>((f >> 23) & 0xFFu) - 127 + 15;
	uint32_t       mant = f & 0x7FFFFFu;
	if (exp <= 0) return static_cast<uint16_t>(sign);
	if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
	mant += 0x1000u; // round
	if (mant & 0x800000u) return static_cast<uint16_t>(sign | ((exp + 1) << 10));
	return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

// AMD's MLSR_Optimized_Constants (104 bytes, the reflected offsets).
struct Constants {
	float    inv_size[2], scale[2], inv_scale[2], jitter[2], mv_scale[2], tex_size[2], max_render_size[2], mv_jitter_cancel[2];
	uint32_t width, height, reset, width_lr;
	uint32_t height_lr;
	float    pre_exposure, previous_pre_exposure;
	uint32_t rcas_enabled;
	float    rcas_sharpness, pad;
};
static_assert(sizeof(Constants) == 104);

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
void ComputeBarrier(vk::CommandBuffer command) {
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
	        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
}

void Update(vk::CommandBuffer command, const Buffer& b, const void* data, size_t size) {
	for (size_t at = 0; at < size; at += 65536) {
		const size_t n = std::min<size_t>(65536, size - at);
		command.updateBuffer(b.buffer, at, n, static_cast<const char*>(data) + at);
	}
}

uint32_t Div(uint32_t a, uint32_t b) {
	return (a + b - 1) / b;
}

float HalfToFloat(uint16_t h) {
	const uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu;
	uint32_t f = 0;
	if (exp == 0) {
		if (mant == 0) f = sign;
		else {
			float v = std::ldexp(static_cast<float>(mant), -24);
			return (h & 0x8000u) ? -v : v;
		}
	} else if (exp == 31) f = sign | 0x7F800000u | (mant << 13);
	else f = sign | ((exp + 112u) << 23) | (mant << 13);
	float v = 0.0f;
	std::memcpy(&v, &f, 4);
	return v;
}

// The check that FSR 4's result reached the game's image (KYTY_FSR4_PROBE=0: none): at a few frames, the centre texel
// of our result and of the game's output after our blit, and the exposure the model read, copied to a host buffer and
// logged some frames later.
constexpr int ProbePoints = 5; // the centre and the four quarter points
struct ProbeState {
	VkBuffer      buffer = VK_NULL_HANDLE;
	VmaAllocation memory = VK_NULL_HANDLE;
	const uint8_t* mapped = nullptr;
	Target        game; // 5x1 rgba16f: the output's texels at the probe points blitted here
	uint64_t      pending = 0; // the frame the copies were recorded at (0: none)
	bool          failed = false;
} g_probe;

void ProbeRead() {
	if (g_probe.pending == 0 || g.frames < g_probe.pending + 8) return;
	const auto* h = reinterpret_cast<const uint16_t*>(g_probe.mapped);
	float exposure = 0.0f;
	std::memcpy(&exposure, g_probe.mapped + 96, 4);
	bool  finite = true, lit = false;
	float diff = 0.0f, mag = 0.0f;
	std::string points;
	for (int p = 0; p < ProbePoints; ++p) {
		float ours[3], game[3];
		for (int i = 0; i < 3; ++i) {
			ours[i] = HalfToFloat(h[p * 4 + i]);
			game[i] = HalfToFloat(h[24 + p * 4 + i]);
			finite  = finite && std::isfinite(ours[i]) && std::isfinite(game[i]);
			diff    = std::max(diff, std::fabs(ours[i] - game[i]));
			mag     = std::max(mag, std::fabs(ours[i]));
			lit     = lit || std::fabs(ours[i]) > 0.002f;
		}
		char text[96];
		std::snprintf(text, sizeof(text), "%s%.3f %.3f %.3f / %.3f %.3f %.3f", p == 0 ? "" : "; ", ours[0], ours[1], ours[2], game[0],
		              game[1], game[2]);
		points += text;
	}
	const bool reached = finite && diff <= 0.05f * std::max(mag, 0.05f);
	std::printf("FSR 4 check (frame %llu; ours / the game's output at the centre and four quarter points): %s; exposure read "
	            "%.4f: %s\n",
	            static_cast<unsigned long long>(g_probe.pending), points.c_str(), exposure,
	            !finite   ? "NaN/Inf in FSR 4's result"
	            : !lit    ? "all black (a loading screen?): nothing shown"
	            : reached ? "FSR 4's output reached the game's image"
	                      : "the game's image differs from FSR 4's result");
	std::fflush(stdout);
	g_probe.pending = 0;
}

void Probe(vk::CommandBuffer command, const UpscaleDesc& desc, vk::Extent2D target) {
	static const bool on = EnvOn("KYTY_FSR4_PROBE", true);
	const uint64_t frame = g.frames + 1;
	if (!on || g_probe.failed || g_probe.pending != 0 || !(frame == 1800 || frame % 3600 == 0)) return;
	if (!(desc.output_usage & vk::ImageUsageFlagBits::eTransferSrc)) return;
	if (g_probe.buffer == VK_NULL_HANDLE) {
		g_probe.failed = true;
		auto* graphics = desc.graphics;
		if (graphics == nullptr) return;
		VkBufferCreateInfo create {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		create.size  = 128;
		create.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo host {};
		host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo info {};
		if (vmaCreateBuffer(graphics->allocator, &create, &host, &g_probe.buffer, &g_probe.memory, &info) != VK_SUCCESS) {
			g_probe.buffer = VK_NULL_HANDLE;
			return;
		}
		g_probe.mapped = static_cast<const uint8_t*>(info.pMappedData);
		if (!CreateTarget(*graphics, g_probe.game, {ProbePoints, 1}, vk::Format::eR16G16B16A16Sfloat,
		                  vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc))
			return;
		g_probe.failed = false;
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite, vk::PipelineStageFlagBits2::eAllTransfer,
	        vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite);
	// The probe points as fractions of the image (the centre, then the four quarter points).
	constexpr float fx[ProbePoints] {0.5f, 0.25f, 0.75f, 0.25f, 0.75f}, fy[ProbePoints] {0.5f, 0.25f, 0.25f, 0.75f, 0.75f};
	FrameGen::Transition(command, g_probe.game.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
	                     vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eBlit);
	vk::BufferImageCopy copy {};
	copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.imageExtent      = vk::Extent3D {1, 1, 1};
	for (int p = 0; p < ProbePoints; ++p) {
		const auto tx = static_cast<int32_t>(fx[p] * target.width), ty = static_cast<int32_t>(fy[p] * target.height);
		// The output texel our (tx, ty) went to (its top-left one at an integer ratio).
		const auto ox = static_cast<int32_t>(static_cast<int64_t>(tx) * desc.upscale.width / target.width);
		const auto oy = static_cast<int32_t>(static_cast<int64_t>(ty) * desc.upscale.height / target.height);
		copy.bufferOffset = static_cast<vk::DeviceSize>(p) * 8;
		copy.imageOffset  = vk::Offset3D {tx, ty, 0};
		command.copyImageToBuffer(g.out16.image.image, vk::ImageLayout::eTransferSrcOptimal, g_probe.buffer, 1, &copy);
		vk::ImageBlit blit {};
		blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, desc.output_level, desc.output_layer, 1};
		blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		blit.srcOffsets[0]  = vk::Offset3D {ox, oy, 0};
		blit.srcOffsets[1]  = vk::Offset3D {ox + 1, oy + 1, 1};
		blit.dstOffsets[0]  = vk::Offset3D {p, 0, 0};
		blit.dstOffsets[1]  = vk::Offset3D {p + 1, 1, 1};
		command.blitImage(desc.output, vk::ImageLayout::eGeneral, g_probe.game.image.image, vk::ImageLayout::eTransferDstOptimal, 1,
		                  &blit, vk::Filter::eNearest);
	}
	FrameGen::Transition(command, g_probe.game.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
	                     vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eCopy);
	copy.bufferOffset = 48;
	copy.imageOffset  = vk::Offset3D {0, 0, 0};
	copy.imageExtent  = vk::Extent3D {ProbePoints, 1, 1};
	command.copyImageToBuffer(g_probe.game.image.image, vk::ImageLayout::eTransferSrcOptimal, g_probe.buffer, 1, &copy);
	// The exposure texel (general).
	copy.imageExtent  = vk::Extent3D {1, 1, 1};
	copy.bufferOffset = 96;
	command.copyImageToBuffer(g.exposure.image.image, vk::ImageLayout::eGeneral, g_probe.buffer, 1, &copy);
	Barrier(command, vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
	        vk::AccessFlagBits2::eHostRead);
	g_probe.pending = frame;
}

} // namespace

float Sharpness() {
	// KYTY_FSR4_SHARPNESS=<0..1>, else the launcher's KYTY_UPSCALE_SHARPNESS in full as FSR 3.1 takes it (10-28: half
	// of it, on top of the blur of a linear 1080 -> 2160 -> 1080, looked like no sharpening at all), else 0.3.
	static const float amount = [] {
		const auto read = [](const char* name, float& out) {
			const char* value = std::getenv(name);
			if (value == nullptr || *value == '\0') return false;
			out = std::clamp(static_cast<float>(std::atof(value)), 0.0f, 1.0f);
			return true;
		};
		float v = 0.0f;
		if (read("KYTY_FSR4_SHARPNESS", v)) return v;
		if (read("KYTY_UPSCALE_SHARPNESS", v)) return v;
		return 0.3f;
	}();
	return amount;
}

bool SrPathWanted() {
	static const bool wanted = [] {
		const char* value = std::getenv("KYTY_UPSCALER");
		return value != nullptr && std::string(value) == "fsr4";
	}();
	return wanted;
}

const char* Name() {
	const int preset = RequestedPreset();
	return preset == PresetAuto ? (g.built ? PresetTitles[g.preset] : "FSR 4 (auto preset)") : PresetTitles[preset];
}

void AddDeviceExtensions(const std::vector<vk::ExtensionProperties>& available, std::vector<const char*>& extensions) {
	if (!SrPathWanted()) return;
	const auto has = [&](const char* name) {
		return std::any_of(available.begin(), available.end(), [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
	};
	const auto added = [&](const char* name) {
		return std::any_of(extensions.begin(), extensions.end(), [&](const char* e) { return std::strcmp(e, name) == 0; });
	};
	for (const char* name: {VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME, VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME}) {
		if (has(name)) {
			if (!added(name)) extensions.push_back(name);
			break;
		}
	}
	// (Core in Vulkan 1.3; the extension too where listed, for older loaders' validation.)
	if (has(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) && !added(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME))
		extensions.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
}

void* ChainDeviceFeatures(vk::PhysicalDevice physical_device, const std::vector<const char*>& extensions,
                          vk::PhysicalDeviceVulkan12Features& features12, vk::PhysicalDeviceVulkan13Features& features13,
                          vk::PhysicalDeviceFeatures& features, void* next) {
	if (!SrPathWanted()) return next;
	g_dev.checked = true;
	const auto added = [&](const char* name) {
		return std::any_of(extensions.begin(), extensions.end(), [&](const char* e) { return std::strcmp(e, name) == 0; });
	};
	g_dev.derivatives_ext = added(VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME) || added(VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
	vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR derivatives {};
	vk::PhysicalDeviceVulkan13Features                    s13 {};
	vk::PhysicalDeviceVulkan12Features                    s12 {};
	s12.pNext = &s13;
	if (g_dev.derivatives_ext) s13.pNext = &derivatives;
	vk::PhysicalDeviceFeatures2 s {};
	s.pNext = &s12;
	physical_device.getFeatures2(&s);
	vk::PhysicalDeviceShaderIntegerDotProductProperties dot_props {};
	vk::PhysicalDeviceProperties2                       props {};
	props.pNext = &dot_props;
	physical_device.getProperties2(&props);
	g_dev.dot         = s13.shaderIntegerDotProduct == VK_TRUE;
	g_dev.int8        = s12.shaderInt8 == VK_TRUE;
	g_dev.float16     = s12.shaderFloat16 == VK_TRUE;
	g_dev.int16       = s.features.shaderInt16 == VK_TRUE;
	g_dev.derivatives = g_dev.derivatives_ext && derivatives.computeDerivativeGroupLinear == VK_TRUE;
	g_dev.accelerated = dot_props.integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE ||
	                    dot_props.integerDotProduct4x8BitPackedMixedSignednessAccelerated == VK_TRUE;
	if (g_dev.dot) features13.shaderIntegerDotProduct = VK_TRUE;
	if (g_dev.int8) features12.shaderInt8 = VK_TRUE;
	if (g_dev.float16) features12.shaderFloat16 = VK_TRUE;
	if (g_dev.int16) features.shaderInt16 = VK_TRUE;
	std::printf("FSR 4 device: %s: shaderIntegerDotProduct %s (4x8 packed %saccelerated), shaderInt8 %s, shaderInt16 %s, "
	            "shaderFloat16 %s, computeDerivativeGroupLinear %s\n",
	            static_cast<const char*>(props.properties.deviceName), g_dev.dot ? "yes" : "NO", g_dev.accelerated ? "" : "not ", g_dev.int8 ? "yes" : "NO",
	            g_dev.int16 ? "yes" : "NO", g_dev.float16 ? "yes" : "NO",
	            g_dev.derivatives ? "yes" : g_dev.derivatives_ext ? "NO" : "NO (no compute shader derivatives extension)");
	std::fflush(stdout);
	if (!g_dev.derivatives) return next;
	g_dev.derivatives_features                              = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
	g_dev.derivatives_features.computeDerivativeGroupLinear = VK_TRUE;
	g_dev.derivatives_features.pNext                        = next;
	return &g_dev.derivatives_features;
}

bool Ready() {
	if (!SrPathWanted()) return false;
	if (g.ready_checked) return g.ready;
	g.ready_checked = true;
	if (!g_dev.checked || !g_dev.dot || !g_dev.int8 || !g_dev.int16 || !g_dev.float16 || !g_dev.derivatives) {
		std::printf("FSR 4: this GPU lacks what the INT8/DOT4 model needs (shaderIntegerDotProduct, shaderInt8, shaderInt16, "
		            "shaderFloat16, computeDerivativeGroupLinear): off\n");
		std::fflush(stdout);
		return false;
	}
	g.dir = AssetDir();
	// Every preset checked here (at 2160 and 1080) so a broken package shows at start, not mid-game.
	const int  requested = RequestedPreset();
	std::string why;
	int         checked = 0;
	for (int preset = 0; preset < 6; ++preset) {
		for (bool res2160: {false, true}) {
			if (!LoadPreset(preset, res2160, why)) {
				if (preset == requested || requested == PresetAuto) {
					std::printf("FSR 4: %s: off\n", why.c_str());
					std::fflush(stdout);
					return false;
				}
				std::printf("FSR 4: (another preset) %s\n", why.c_str());
			} else {
				++checked;
			}
		}
	}
	std::printf("FSR 4: %d of 12 preset variants in %s match their manifest (descriptors, entry points, local sizes, sizes); "
	            "%s requested\n",
	            checked, g.dir.string().c_str(), Name());
	std::fflush(stdout);
	g.ready = true;
	return true;
}

bool Upscale(GraphicContext& graphics, const UpscaleDesc& desc) {
	if (!Ready() || g.failed || desc.command == nullptr || desc.color_view == nullptr || desc.depth_view == nullptr ||
	    desc.motion_view == nullptr || desc.output_view == nullptr || desc.render.width == 0 || desc.upscale.width == 0)
		return false;
	const auto command = desc.command;
	ProbeRead();
	// The model's output: the display's size (as DLSS's KYTY_DLSS_OUT_DISPLAY), linearly blitted up into the game's
	// 3840x2160 output for its later stages, which the presenter then scales to the display. 10-28: on a 1080 monitor
	// the model made 3840x2160 from 720 (a 3x ratio, the 2160 variant) only for the presenter to shrink it back to
	// 1080: wasteful and soft. KYTY_FSR4_OUT_DISPLAY=0: the game's output size as before.
	const vk::Extent2D target = [&] {
		static const uint32_t display = [] {
			const char* value = std::getenv("KYTY_DISPLAY_HEIGHT");
			if (value != nullptr && *value != '\0') return static_cast<uint32_t>(std::max(0, std::atoi(value)));
			return LocalPlatform::DisplayHeight(nullptr);
		}();
		static const bool on = EnvOn("KYTY_FSR4_OUT_DISPLAY", true);
		if (!on || display < 360 || display >= desc.upscale.height || display < desc.render.height) return desc.upscale;
		const auto width = (static_cast<uint32_t>(std::lround(static_cast<double>(display) * desc.upscale.width / desc.upscale.height)) + 1u) & ~1u;
		return vk::Extent2D {width, display};
	}();
	const bool scaled = target != desc.upscale;
	// An integer ratio (1080 into 2160): nearest, each pixel a 2x2 block, which the presenter's linear 2x downscale
	// (its samples on the blocks' shared corners) averages back to the pixel itself, so the display gets the model's
	// RCAS-sharpened pixels unchanged. 10-28: a linear blit up and the presenter's linear blit down blurred it twice.
	const bool integer = scaled && desc.upscale.width % target.width == 0 && desc.upscale.height % target.height == 0 &&
	                     desc.upscale.width / target.width == desc.upscale.height / target.height;
	if (scaled && !(desc.output_usage & vk::ImageUsageFlagBits::eTransferDst)) return false;
	const uint32_t W = (target.width + 7u) & ~7u;
	const uint32_t H = (target.height + 7u) & ~7u;
	if (W > 3840 || H > 2160) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("FSR 4: output %ux%u above 3840x2160 (no 4320 variant shipped): stretched\n", desc.upscale.width, desc.upscale.height);
			std::fflush(stdout);
		}
		return false;
	}
	const bool res2160 = W > 1920 || H > 1080;
	const float ratio  = static_cast<float>(target.height) / static_cast<float>(desc.render.height);
	const int   preset = RequestedPreset() == PresetAuto ? (g.built ? g.preset : PresetForRatio(ratio)) : RequestedPreset();
	if (!g.built || g.res2160 != res2160 || g.preset != preset) {
		if (!Build(graphics, preset, res2160)) {
			g.failed = true;
			return false;
		}
	}
	// Our images at the aligned upscale size; the depth copy at the render size.
	const auto sampled_storage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
	if (g.size.width != W || g.size.height != H) {
		(void)graphics.device.waitIdle();
		const vk::Extent2D size {W, H};
		if (!CreateTarget(graphics, g.history, size, vk::Format::eR16G16B16A16Sfloat, sampled_storage) ||
		    !CreateTarget(graphics, g.reprojected, size, vk::Format::eR16G16B16A16Sfloat, sampled_storage) ||
		    !CreateTarget(graphics, g.recurrent, size, vk::Format::eR8G8B8A8Unorm, sampled_storage) ||
		    !CreateTarget(graphics, g.rcas_temp, size, vk::Format::eR16G16B16A16Sfloat, sampled_storage) ||
		    !CreateTarget(graphics, g.out16, size, vk::Format::eR16G16B16A16Sfloat, sampled_storage | vk::ImageUsageFlagBits::eTransferSrc) ||
		    (g.exposure.image.image == nullptr &&
		     !CreateTarget(graphics, g.exposure, {1, 1}, vk::Format::eR32Sfloat, sampled_storage | vk::ImageUsageFlagBits::eTransferSrc))) {
			std::printf("FSR 4: its %ux%u images were not created: off (stretched)\n", W, H);
			std::fflush(stdout);
			g.failed = true;
			return false;
		}
		g.size  = size;
		g.first = true;
	}
	if (g.render != desc.render) {
		(void)graphics.device.waitIdle();
		if (!CreateTarget(graphics, g.depth, desc.render, vk::Format::eR32Sfloat, sampled_storage)) {
			g.failed = true;
			return false;
		}
		g.render = desc.render;
		g.first  = true;
	}
	// Where the result goes: the output itself (rgba16f with storage), else our rgba16f blitted into it.
	const bool direct = !scaled && desc.output_format == vk::Format::eR16G16B16A16Sfloat && (desc.output_usage & vk::ImageUsageFlagBits::eStorage);
	if (!direct && !(desc.output_usage & vk::ImageUsageFlagBits::eTransferDst)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("FSR 4: output format %d usage 0x%x: neither storage rgba16f nor a blit target: stretched\n",
			            static_cast<int>(desc.output_format), static_cast<uint32_t>(static_cast<VkImageUsageFlags>(desc.output_usage)));
			std::fflush(stdout);
		}
		return false;
	}
	const float sharpness = Sharpness();
	const bool rcas = sharpness > 0.0f;

	// Previous work on our images and the inputs before anything here.
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eAllTransfer,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite);
	for (Target* t: {&g.history, &g.reprojected, &g.recurrent, &g.rcas_temp, &g.out16, &g.exposure, &g.depth}) {
		if (t->image.state.layout != vk::ImageLayout::eGeneral)
			FrameGen::Transition(command, t->image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
			                     vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
			                     vk::PipelineStageFlagBits2::eComputeShader);
	}

	// Constants (in the command stream: in order with the frames in flight).
	Constants c {};
	c.inv_size[0] = 1.0f / static_cast<float>(W);
	c.inv_size[1] = 1.0f / static_cast<float>(H);
	c.scale[0]    = static_cast<float>(W) / static_cast<float>(desc.render.width);
	c.scale[1]    = static_cast<float>(H) / static_cast<float>(desc.render.height);
	c.inv_scale[0] = 1.0f / c.scale[0];
	c.inv_scale[1] = 1.0f / c.scale[1];
	c.jitter[0]    = desc.jitter[0];
	c.jitter[1]    = desc.jitter[1];
	// The model wants UV offsets to the previous frame: render pixels over the render size (the TAA's own UV vectors,
	// scale 0, point the other way: -1, as FSR 3.1 takes them).
	static const float sr_sign = [] {
		const char* value = std::getenv("KYTY_SR_MV_SIGN");
		return value != nullptr && *value != '\0' && std::atof(value) < 0 ? -1.0f : 1.0f;
	}();
	const bool have_scale = desc.mv_scale[0] != 0.0f || desc.mv_scale[1] != 0.0f;
	c.mv_scale[0] = have_scale ? sr_sign * desc.mv_scale[0] / static_cast<float>(desc.render.width) : -1.0f;
	c.mv_scale[1] = have_scale ? sr_sign * desc.mv_scale[1] / static_cast<float>(desc.render.height) : -1.0f;
	c.tex_size[0] = static_cast<float>(W);
	c.tex_size[1] = static_cast<float>(H);
	c.max_render_size[0] = static_cast<float>(desc.render.width);
	c.max_render_size[1] = static_cast<float>(desc.render.height);
	c.width     = W;
	c.height    = H;
	c.reset     = desc.reset || g.first ? 1u : 0u;
	c.width_lr  = desc.render.width;
	c.height_lr = desc.render.height;
	c.pre_exposure          = 1.0f;
	c.previous_pre_exposure = 1.0f;
	c.rcas_enabled          = rcas ? 1u : 0u;
	c.rcas_sharpness        = sharpness;
	std::array<uint8_t, 256> cb {};
	std::memcpy(cb.data(), &c, sizeof(c));
	Update(command, g.constants, cb.data(), cb.size());
	// RCAS: AMD's FsrRcasCon with the provider's remap (sharpness 1 = 0 stops, the sharpest).
	std::array<uint32_t, 64> rc {};
	const float stops = -2.0f * sharpness + 2.0f;
	const float lin   = std::exp2(-stops);
	std::memcpy(&rc[0], &lin, 4);
	rc[1] = static_cast<uint32_t>(HalfBits(lin)) | (static_cast<uint32_t>(HalfBits(lin)) << 16);
	const float one = 1.0f;
	std::memcpy(&rc[4], &one, 4); // preExposure
	Update(command, g.rcas_constants, rc.data(), sizeof(rc));
	if (!g.uploaded) {
		Update(command, g.init, g.initializers.data(), g.initializers.size());
		Update(command, g.weights, g.pre_weights.data(), g.pre_weights.size());
		g.uploaded = true;
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferWrite,
	        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eUniformRead | vk::AccessFlagBits2::eShaderRead);

	// The depth copy and the exposure (fsr4-prep.comp).
	{
		static const bool game_exposure = [] { // KYTY_FSR4_EXPOSURE=1: the model sees exposure 1 (A/B)
			const char* value = std::getenv("KYTY_FSR4_EXPOSURE");
			return !(value != nullptr && std::string(value) == "1");
		}();
		const bool has_exposure = game_exposure && desc.exposure_buffer != nullptr && desc.exposure_range >= sizeof(float);
		const vk::DescriptorImageInfo  depth_in {g.nearest, desc.depth_view, desc.depth_layout};
		const vk::DescriptorBufferInfo values = has_exposure ? vk::DescriptorBufferInfo {desc.exposure_buffer, desc.exposure_base, desc.exposure_range}
		                                                     : vk::DescriptorBufferInfo {g.constants.buffer, 0, 256};
		const vk::DescriptorImageInfo depth_out {nullptr, g.depth.view, vk::ImageLayout::eGeneral};
		const vk::DescriptorImageInfo exposure_out {nullptr, g.exposure.view, vk::ImageLayout::eGeneral};
		std::array<vk::WriteDescriptorSet, 4> writes {};
		for (uint32_t i = 0; i < 4; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
		}
		writes[0].descriptorType = vk::DescriptorType::eCombinedImageSampler;
		writes[0].pImageInfo     = &depth_in;
		writes[1].descriptorType = vk::DescriptorType::eStorageBuffer;
		writes[1].pBufferInfo    = &values;
		writes[2].descriptorType = vk::DescriptorType::eStorageImage;
		writes[2].pImageInfo     = &depth_out;
		writes[3].descriptorType = vk::DescriptorType::eStorageImage;
		writes[3].pImageInfo     = &exposure_out;
		static const bool invert = EnvOn("KYTY_FSR4_DEPTH_INVERT", true);
		const uint32_t push[8] {has_exposure ? desc.exposure_index : 0u, has_exposure ? 1u : 0u, invert ? 1u : 0u, desc.render.width,
		                        desc.render.height, 0, 0, 0};
		command.bindPipeline(vk::PipelineBindPoint::eCompute, g.prep.pipeline);
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g.prep.layout, 0, 4, writes.data());
		command.pushConstants(g.prep.layout, vk::ShaderStageFlagBits::eCompute, 0, 32, push);
		command.dispatch(Div(desc.render.width, 8), Div(desc.render.height, 8), 1);
		ComputeBarrier(command);
	}

	// Each pass's descriptors by the manifest's names.
	const vk::ImageView final_view = direct ? desc.output_view : g.out16.view;
	const vk::ImageView mlsr_out   = rcas ? g.rcas_temp.view : final_view;
	const auto run = [&](Pipe& p, uint32_t x, uint32_t y) -> bool {
		std::vector<vk::DescriptorImageInfo>  images(p.module.resources.size());
		std::vector<vk::DescriptorBufferInfo> buffers(p.module.resources.size());
		std::vector<vk::WriteDescriptorSet>   writes;
		for (size_t i = 0; i < p.module.resources.size(); ++i) {
			const auto& r = p.module.resources[i];
			const auto& n = r.name;
			vk::WriteDescriptorSet w {};
			w.dstBinding      = r.binding;
			w.descriptorCount = 1;
			w.descriptorType  = DescriptorType(r.kind);
			auto image = [&](vk::ImageView view, vk::ImageLayout layout) {
				images[i]    = vk::DescriptorImageInfo {nullptr, view, layout};
				w.pImageInfo = &images[i];
			};
			auto buffer = [&](const Buffer& b, vk::DeviceSize size) {
				buffers[i]    = vk::DescriptorBufferInfo {b.buffer, 0, size};
				w.pBufferInfo = &buffers[i];
			};
			if (n == "r_input_color") image(desc.color_view, desc.color_layout);
			else if (n == "r_velocity") image(desc.motion_view, desc.motion_layout);
			else if (n == "r_depth") image(g.depth.view, vk::ImageLayout::eGeneral);
			else if (n == "r_input_exposure") image(g.exposure.view, vk::ImageLayout::eGeneral);
			else if (n == "r_history_color" || n == "rw_history_color") image(g.history.view, vk::ImageLayout::eGeneral);
			else if (n == "r_recurrent_0" || n == "rw_recurrent_0") image(g.recurrent.view, vk::ImageLayout::eGeneral);
			else if (n == "r_reprojected_color" || n == "rw_reprojected_color") image(g.reprojected.view, vk::ImageLayout::eGeneral);
			else if (n == "rw_mlsr_output_color") image(mlsr_out, vk::ImageLayout::eGeneral);
			else if (n == "r_rcas_input") image(g.rcas_temp.view, vk::ImageLayout::eGeneral);
			else if (n == "rw_rcas_output") image(final_view, vk::ImageLayout::eGeneral);
			else if (n == "g_history_sampler") {
				images[i]    = vk::DescriptorImageInfo {g.linear, nullptr, vk::ImageLayout::eUndefined};
				w.pImageInfo = &images[i];
			} else if (n == "ScratchBuffer") buffer(g.scratch, g.scratch.size);
			else if (n == "InitializerBuffer") buffer(g.init, g.initializers.size());
			else if (n == "cbPass_Weights") buffer(g.weights, 1024);
			else if (n == "MLSR_Optimized_Constants" || n == "cbRCAS") buffer(n == "cbRCAS" ? g.rcas_constants : g.constants, 256);
			else {
				std::printf("FSR 4: %s binding %u (%s) unknown to this runtime: off\n", p.module.path.filename().string().c_str(),
				            r.binding, n.c_str());
				std::fflush(stdout);
				return false;
			}
			writes.push_back(w);
		}
		command.bindPipeline(vk::PipelineBindPoint::eCompute, p.pipeline);
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, p.layout, 0, static_cast<uint32_t>(writes.size()), writes.data());
		command.dispatch(x, y, 1);
		ComputeBarrier(command);
		return true;
	};
	// AMD's non-WMMA dispatch sizes (the encoder levels at 1/2, 1/4, 1/8 of the aligned output).
	const uint32_t e[3][2] = {{W / 2, H / 2}, {W / 4, H / 4}, {W / 8, H / 8}};
	std::array<std::array<uint32_t, 2>, 14> d {};
	d[0]  = {Div(W, 16), Div(H, 16)};
	d[1]  = d[2] = {Div(e[0][0], 64), e[0][1]};
	d[3]  = d[4] = d[5] = {Div(e[1][0], 64), e[1][1]};
	d[6]  = d[7] = d[8] = {Div(e[2][0], 64), e[2][1]};
	d[9]  = {Div(e[2][0], 8), Div(e[2][1], 8)};
	d[10] = d[11] = {Div(e[1][0], 64), e[1][1]};
	d[12] = {Div(e[0][0], 64), e[0][1]};
	d[13] = {Div(W, 16), Div(H, 16)};
	for (int i = 0; i < 14; ++i) {
		if (!run(g.model[i], d[i][0], d[i][1])) {
			g.failed = true;
			return false;
		}
	}
	if (rcas && !run(g.rcas, Div(W, 16), Div(H, 16))) {
		g.failed = true;
		return false;
	}
	if (!direct) {
		FrameGen::Transition(command, g.out16.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
		                     vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eBlit);
		Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
		        vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite);
		vk::ImageBlit blit {};
		blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, desc.output_level, desc.output_layer, 1};
		blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(target.width), static_cast<int32_t>(target.height), 1};
		blit.dstOffsets[1]  = vk::Offset3D {static_cast<int32_t>(desc.upscale.width), static_cast<int32_t>(desc.upscale.height), 1};
		command.blitImage(g.out16.image.image, vk::ImageLayout::eTransferSrcOptimal, desc.output, vk::ImageLayout::eGeneral, 1, &blit,
		                  scaled && !integer ? vk::Filter::eLinear : vk::Filter::eNearest);
		Probe(command, desc, target);
		FrameGen::Transition(command, g.out16.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
		                     vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader);
	}
	// The result before whatever reads or writes the output next (the exposure pass, the composite).
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eBlit,
	        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eAllCommands,
	        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	if (g.frames++ == 0) {
		std::printf("FSR 4: first frame %ux%u -> %ux%u (%s, %s variant), RCAS %s, depth %s, exposure %s, output %s\n", desc.render.width,
		            desc.render.height, target.width, target.height, PresetTitles[g.preset], g.res2160 ? "2160" : "1080",
		            rcas ? "on" : "off", EnvOn("KYTY_FSR4_DEPTH_INVERT", true) ? "flipped" : "as is",
		            desc.exposure_buffer != nullptr ? "the game's" : "1.0", direct ? "written in place" : scaled ? (integer ? "at the display's size, blitted (nearest, an integer ratio) into the game's output" : "at the display's size, blitted (linear) into the game's output") : "via rgba16f + blit");
		if (scaled)
			std::printf("FSR 4: the game's output stays %ux%u (its HUD and composite); the model makes the display's %ux%u from %ux%u (%.2fx)\n",
			            desc.upscale.width, desc.upscale.height, target.width, target.height, desc.render.width, desc.render.height, ratio);
		std::printf("FSR 4: RCAS %s, strength %.3f (KYTY_FSR4_SHARPNESS, else KYTY_UPSCALE_SHARPNESS), on the model's %ux%u result, "
		            "the last pass before the game's output\n",
		            rcas ? "on" : "off", sharpness, target.width, target.height);
		std::fflush(stdout);
	}
	g.first = false;
	return true;
}

} // namespace Libs::Graphics::Fsr4
