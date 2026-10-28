// AMD FSR 4 (the source-v07 INT8 / DOT4 model) on Vulkan: KytyPS5's own runtime (src/local/fsr4-sr.cpp, its "fsr4 path"),
// ported from its vk:: / VMA wrappers to the plain Vulkan calls of this injector. The model's SPIR-V and weights are AMD's
// (MIT, fsr4_shaders/LICENSE-FSR4-v07.txt), shipped in fsr4_shaders\ next to the exe: per preset a pre pass, 12 network
// passes and a post pass (1080 and 2160 variants), the preset's initializers.bin and pre_weights.bin, and a
// shader_manifest.json. Every module is checked against the manifest (entry point, local size, descriptors reflected from
// the SPIR-V) before a pipeline is built; dispatch sizes and constants are those of AMD's FidelityFX SDK FSR 4 provider
// (MIT) for its non-WMMA (DOT4) path. Our fsr4_prep.comp writes the depth (flipped when reversed: the shipped modules
// dilate motion vectors by the closest forward depth) and the 1x1 exposure (none: the model reads 1). Sharpening is not
// the model's RCAS here: one RCAS for every upscaler, at the display size (upscale.cpp).
// Needs shaderIntegerDotProduct, shaderInt8, shaderInt16, shaderFloat16, computeDerivativeGroupLinear and
// VK_KHR_push_descriptor (added at device creation where the device has them).
#include "upscale_backend.h"

#include "shaders/fsr4_prep_spv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#ifdef INJ_FSR4
#include <nlohmann/json.hpp>
#endif

namespace inj::upscale::fsr4 {
namespace {

struct DeviceSupport {
	bool checked = false;
	bool dot = false, int8 = false, int16 = false, float16 = false, derivatives = false, accelerated = false;
} g_dev;

std::filesystem::path AssetDir() {
	return std::filesystem::path(ExeDir()) / "fsr4_shaders";
}
bool AssetsPresent() {
	std::error_code e;
	return std::filesystem::exists(AssetDir() / "fsr4_model_v07_i8_quality_shader_manifest.json", e);
}

} // namespace

void DeviceExtensions(const std::vector<VkExtensionProperties>& available, std::vector<const char*>& names) {
	if (!AssetsPresent()) return;
	const auto has = [&](const char* name) {
		return std::any_of(available.begin(), available.end(), [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
	};
	if (has("VK_KHR_compute_shader_derivatives")) names.push_back("VK_KHR_compute_shader_derivatives");
	else if (has(VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME)) names.push_back(VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
}

void DeviceFeatures(VkPhysicalDevice physical, VkPhysicalDeviceFeatures& f, VkPhysicalDeviceVulkan12Features& f12,
                    VkPhysicalDeviceVulkan13Features& f13, bool derivatives_ext, bool& want_derivatives) {
	g_dev = {};
	if (!AssetsPresent()) return;
	auto features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(R().gipa(R().instance, "vkGetPhysicalDeviceFeatures2"));
	auto props2    = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(R().gipa(R().instance, "vkGetPhysicalDeviceProperties2"));
	if (features2 == nullptr || props2 == nullptr) return;
	g_dev.checked = true;
	VkPhysicalDeviceComputeShaderDerivativesFeaturesNV derivatives {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV};
	VkPhysicalDeviceVulkan13Features s13 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
	VkPhysicalDeviceVulkan12Features s12 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	VkPhysicalDeviceFeatures2        s {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	s.pNext   = &s12;
	s12.pNext = &s13;
	if (derivatives_ext) s13.pNext = &derivatives;
	features2(physical, &s);
	VkPhysicalDeviceShaderIntegerDotProductProperties dot {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES};
	VkPhysicalDeviceProperties2 props {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
	props.pNext = &dot;
	props2(physical, &props);
	g_dev.dot         = s13.shaderIntegerDotProduct == VK_TRUE;
	g_dev.int8        = s12.shaderInt8 == VK_TRUE;
	g_dev.float16     = s12.shaderFloat16 == VK_TRUE;
	g_dev.int16       = s.features.shaderInt16 == VK_TRUE;
	g_dev.derivatives = derivatives_ext && derivatives.computeDerivativeGroupLinear == VK_TRUE;
	g_dev.accelerated = dot.integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE ||
	                    dot.integerDotProduct4x8BitPackedMixedSignednessAccelerated == VK_TRUE;
	f13.shaderIntegerDotProduct = g_dev.dot ? VK_TRUE : f13.shaderIntegerDotProduct;
	f12.shaderInt8    = g_dev.int8 ? VK_TRUE : f12.shaderInt8;
	f12.shaderFloat16 = g_dev.float16 ? VK_TRUE : f12.shaderFloat16;
	f.shaderInt16     = g_dev.int16 ? VK_TRUE : f.shaderInt16;
	want_derivatives  = g_dev.derivatives;
	Log("[fsr4] device %s: shaderIntegerDotProduct %s (4x8 packed %saccelerated), shaderInt8 %s, shaderInt16 %s, shaderFloat16 %s, "
	    "computeDerivativeGroupLinear %s",
	    props.properties.deviceName, g_dev.dot ? "yes" : "NO", g_dev.accelerated ? "" : "not ", g_dev.int8 ? "yes" : "NO",
	    g_dev.int16 ? "yes" : "NO", g_dev.float16 ? "yes" : "NO",
	    g_dev.derivatives ? "yes" : derivatives_ext ? "NO" : "NO (no compute shader derivatives extension)");
}

#ifdef INJ_FSR4
namespace {

const char* const PresetNames[]  = {"native", "quality", "balanced", "performance", "ultraperf", "drs"};
const char* const PresetTitles[] = {"FSR 4 Native AA", "FSR 4 Quality", "FSR 4 Balanced", "FSR 4 Performance",
                                    "FSR 4 Ultra Performance", "FSR 4 Dynamic (DRS)"};
// AMD's scratch sizes (shader_selector.cpp GetScratchSize: the network's tensors for that output size).
constexpr VkDeviceSize Scratch1080 = 20880256;
constexpr VkDeviceSize Scratch2160 = 83232256;

bool ReadFile(const std::filesystem::path& path, std::vector<char>& data) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) return false;
	const auto size = static_cast<size_t>(file.tellg());
	data.resize(size);
	file.seekg(0);
	return static_cast<bool>(file.read(data.data(), static_cast<std::streamsize>(size))) || size == 0;
}

// ---- A minimal SPIR-V reflection (KytyPS5's): each descriptor's (set, binding, type, name), entry points, local size ----
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
	struct Type { uint32_t op = 0; std::vector<uint32_t> args; };
	std::map<uint32_t, Type>        types;
	std::map<uint32_t, std::string> names;
	std::map<uint32_t, uint32_t>    sets, bindings;
	std::map<uint32_t, bool>        buffer_block;
	struct Var { uint32_t id, ptr, sc; };
	std::vector<Var> vars;
	for (size_t i = 5; i < w.size();) {
		const uint32_t count = w[i] >> 16, op = w[i] & 0xFFFFu;
		if (count == 0 || i + count > w.size()) return false;
		const uint32_t* a = &w[i + 1];
		const size_t    n = count - 1;
		switch (op) {
		case 5: if (n >= 1) names[a[0]] = SpvString(a + 1, n - 1); break;          // OpName
		case 15: if (n >= 3) out.entries.push_back(SpvString(a + 2, n - 2)); break; // OpEntryPoint
		case 16: if (n >= 5 && a[1] == 17) out.local = {a[2], a[3], a[4]}; break;   // OpExecutionMode LocalSize
		case 71:                                                                    // OpDecorate
			if (n >= 3 && a[1] == 34) sets[a[0]] = a[2];
			if (n >= 3 && a[1] == 33) bindings[a[0]] = a[2];
			if (n >= 2 && a[1] == 3) buffer_block[a[0]] = true;
			break;
		case 25: case 26: case 27: case 28: case 29: case 30: case 32:
			if (n >= 1) types[a[0]] = {op, std::vector<uint32_t>(a + 1, a + n)};
			break;
		case 59: if (n >= 3) vars.push_back({a[1], a[0], a[2]}); break;             // OpVariable
		default: break;
		}
		i += count;
	}
	for (const auto& v: vars) {
		if (v.sc == 9) { out.push_constants = true; continue; }
		if (v.sc != 0 && v.sc != 2 && v.sc != 12) continue;
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

VkDescriptorType DescriptorType(Kind kind) {
	switch (kind) {
	case Kind::SampledImage: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	case Kind::StorageImage: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	case Kind::Sampler: return VK_DESCRIPTOR_TYPE_SAMPLER;
	case Kind::CombinedImageSampler: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	case Kind::UniformBuffer: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	case Kind::StorageBuffer: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	}
	return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
}

struct Module {
	std::filesystem::path   path;
	std::string             entry;
	std::array<uint32_t, 3> local {0, 0, 0};
	std::vector<Resource>   resources;
	std::vector<char>       code;
};

// Loads `name` (the manifest's artifact) and checks it against the manifest and its own reflection.
bool LoadModule(const nlohmann::json& artifacts, const std::filesystem::path& dir, const std::string& name,
                const std::filesystem::path& file, Module& out, std::string& why) {
	const auto it = artifacts.find(name);
	if (it == artifacts.end()) { why = name + ": not in the manifest"; return false; }
	const auto& art = *it;
	out.path = file.empty() ? dir / name : file;
	if (!ReadFile(out.path, out.code)) { why = out.path.string() + ": missing"; return false; }
	if (file.empty() && art.value("size", static_cast<uint64_t>(0)) != out.code.size()) {
		why = name + ": size " + std::to_string(out.code.size()) + ", the manifest says " + std::to_string(art.value("size", static_cast<uint64_t>(0)));
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
		if (!KindFromManifest(r.value("type", std::string()), res.kind)) { why = name + ": descriptor type unknown"; return false; }
		if (res.set != 0) { why = name + ": descriptor set " + std::to_string(res.set) + " (one set expected)"; return false; }
		out.resources.push_back(res);
	}
	std::sort(out.resources.begin(), out.resources.end(), [](const Resource& x, const Resource& y) { return x.binding < y.binding; });
	Reflection reflected;
	if (!Reflect(out.code, reflected)) { why = out.path.string() + ": not SPIR-V"; return false; }
	if (std::find(reflected.entries.begin(), reflected.entries.end(), out.entry) == reflected.entries.end()) {
		why = name + ": no entry point " + out.entry;
		return false;
	}
	if (reflected.local != out.local) { why = name + ": local size differs from the manifest"; return false; }
	if (reflected.push_constants) { why = name + ": push constants (none expected)"; return false; }
	if (reflected.resources.size() != out.resources.size()) {
		why = name + ": " + std::to_string(reflected.resources.size()) + " descriptors, the manifest lists " + std::to_string(out.resources.size());
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

struct Pipe {
	Module                module;
	VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
	VkPipelineLayout      layout = VK_NULL_HANDLE;
	VkPipeline            pipeline = VK_NULL_HANDLE;
};

void DestroyPipe(Pipe& p) {
	if (R().device == VK_NULL_HANDLE) { p.pipeline = VK_NULL_HANDLE; p.layout = VK_NULL_HANDLE; p.set_layout = VK_NULL_HANDLE; return; }
	if (p.pipeline) R().vkDestroyPipeline(R().device, p.pipeline, nullptr);
	if (p.layout) R().vkDestroyPipelineLayout(R().device, p.layout, nullptr);
	if (p.set_layout) R().vkDestroyDescriptorSetLayout(R().device, p.set_layout, nullptr);
	p.pipeline = VK_NULL_HANDLE;
	p.layout = VK_NULL_HANDLE;
	p.set_layout = VK_NULL_HANDLE;
}

bool BuildPipe(Pipe& p, uint32_t push_bytes) {
	std::vector<VkDescriptorSetLayoutBinding> bindings;
	for (const auto& r: p.module.resources) {
		VkDescriptorSetLayoutBinding b {};
		b.binding         = r.binding;
		b.descriptorType  = DescriptorType(r.kind);
		b.descriptorCount = 1;
		b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
		bindings.push_back(b);
	}
	VkDescriptorSetLayoutCreateInfo si {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	si.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
	si.bindingCount = static_cast<uint32_t>(bindings.size());
	si.pBindings    = bindings.data();
	if (R().vkCreateDescriptorSetLayout(R().device, &si, nullptr, &p.set_layout) != VK_SUCCESS) return false;
	VkPushConstantRange push {VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
	VkPipelineLayoutCreateInfo li {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	li.setLayoutCount = 1;
	li.pSetLayouts    = &p.set_layout;
	if (push_bytes != 0) {
		li.pushConstantRangeCount = 1;
		li.pPushConstantRanges    = &push;
	}
	if (R().vkCreatePipelineLayout(R().device, &li, nullptr, &p.layout) != VK_SUCCESS) return false;
	VkShaderModuleCreateInfo mi {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	mi.codeSize = p.module.code.size();
	mi.pCode    = reinterpret_cast<const uint32_t*>(p.module.code.data());
	VkShaderModule module = VK_NULL_HANDLE;
	if (R().vkCreateShaderModule(R().device, &mi, nullptr, &module) != VK_SUCCESS) return false;
	VkComputePipelineCreateInfo pi {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	pi.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pi.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	pi.stage.module = module;
	pi.stage.pName  = p.module.entry.c_str();
	pi.layout       = p.layout;
	const VkResult r = R().vkCreateComputePipelines(R().device, VK_NULL_HANDLE, 1, &pi, nullptr, &p.pipeline);
	R().vkDestroyShaderModule(R().device, module, nullptr);
	if (r != VK_SUCCESS) { p.pipeline = VK_NULL_HANDLE; return false; }
	return true;
}

std::string ModelName(int preset) {
	return std::string("fsr4_model_v07_i8_") + PresetNames[preset];
}

struct Assets {
	nlohmann::json        manifest;
	std::vector<char>     initializers, pre_weights;
	std::array<Module, 14> modules;
};

// The preset's manifest, weights and modules (the 1080 or 2160 variant), each checked against the manifest.
bool LoadPreset(int preset, bool res2160, Assets& a, std::string& why) {
	const auto dir = AssetDir();
	const std::string model = ModelName(preset);
	std::vector<char> text;
	if (!ReadFile(dir / (model + "_shader_manifest.json"), text)) { why = model + "_shader_manifest.json: missing"; return false; }
	a.manifest = nlohmann::json::parse(text.begin(), text.end(), nullptr, false);
	if (a.manifest.is_discarded() || !a.manifest.contains("artifacts")) { why = model + "_shader_manifest.json: not a manifest"; return false; }
	const auto& perm = a.manifest.value("permutation", nlohmann::json::object());
	if (perm.value("FSR4_ENABLE_DOT4", 0) != 1 || perm.value("WMMA_ENABLED", 1) != 0 || perm.value("FFX_MLSR_LOW_RES_MV", 0) != 1 ||
	    perm.value("FFX_MLSR_AUTOEXPOSURE_ENABLED", 1) != 0 || perm.value("FFX_MLSR_COLORSPACE", 1) != 0 ||
	    perm.value("FFX_MLSR_JITTERED_MOTION_VECTORS", 1) != 0) {
		why = model + ": a permutation this runtime does not drive (DOT4, no WMMA, low-res linear MV, input exposure)";
		return false;
	}
	const auto& artifacts = a.manifest["artifacts"];
	for (const char* blob: {"initializers.bin", "pre_weights.bin"}) {
		const std::string name = model + "_" + blob;
		auto& data = std::string_view(blob) == "initializers.bin" ? a.initializers : a.pre_weights;
		if (!ReadFile(dir / name, data)) { why = name + ": missing"; return false; }
		const auto it = artifacts.find(name);
		if (it == artifacts.end() || it->value("size", static_cast<uint64_t>(0)) != data.size()) {
			why = name + ": its size does not match the manifest";
			return false;
		}
	}
	if (a.pre_weights.size() != 1024) { why = model + "_pre_weights.bin: 1024 bytes expected (cbPass_Weights)"; return false; }
	const std::string res = res2160 ? "2160" : "1080";
	for (int i = 0; i < 14; ++i) {
		const std::string pass = i == 0 ? "pre" : i == 13 ? "post" : "pass" + std::to_string(i);
		const std::string name = model + "_" + res + "_" + pass + ".spv";
		if (!LoadModule(artifacts, dir, name, {}, a.modules[i], why)) return false;
	}
	return true;
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

uint32_t Div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

void ComputeBarrier(VkCommandBuffer cb) {
	VkMemoryBarrier m {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	m.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	m.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	R().vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);
}

void Update(VkCommandBuffer cb, const OwnBuffer& b, const void* data, size_t size) {
	for (size_t at = 0; at < size; at += 65536) {
		const size_t n = std::min<size_t>(65536, size - at);
		R().vkCmdUpdateBuffer(cb, b.buffer, at, n & ~size_t(3), static_cast<const char*>(data) + at);
	}
}

bool g_ready_checked = false, g_ready = false;
std::string g_why;

class Fsr4 final : public Upscaler {
public:
	~Fsr4() override {
		for (auto& p: model_) DestroyPipe(p);
		for (auto& p: old_pipes_) DestroyPipe(p);
		for (auto& i: old_images_) DestroyOwnImage(i);
		DestroyPipe(prep_);
		for (OwnImage* t: {&history_, &reprojected_, &recurrent_, &exposure_, &depth_}) DestroyOwnImage(*t);
		for (OwnBuffer* b: {&scratch_, &init_, &weights_, &constants_}) DestroyOwnBuffer(*b);
		if (R().device != VK_NULL_HANDLE) {
			if (linear_) R().vkDestroySampler(R().device, linear_, nullptr);
			if (nearest_) R().vkDestroySampler(R().device, nearest_, nullptr);
		}
	}
	const char* Name() const override { return "FSR 4"; }
	std::string Detail() const override {
		return built_ ? std::string(PresetTitles[preset_]) + " model (" + ModelName(preset_) + ", " + (res2160_ ? "2160" : "1080") + " variant)" : "";
	}

	bool Dispatch(const Frame& f) override {
		if (failed_) return false;
		const uint32_t W = (f.display.width + 7u) & ~7u, H = (f.display.height + 7u) & ~7u;
		if (W > 3840 || H > 2160) {
			if (!said_size_) Log("[fsr4] output %ux%u above 3840x2160 (no 4320 variant shipped): off", f.display.width, f.display.height);
			said_size_ = true;
			return false;
		}
		const bool res2160 = W > 1920 || H > 1080;
		const int  preset  = std::clamp(f.quality - 1, 0, 4);  // native, quality, balanced, performance, ultra performance
		if (!built_ || res2160_ != res2160 || preset_ != preset) {
			if (!Build(preset, res2160)) { failed_ = true; return false; }
		}
		const VkImageUsageFlags ss = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		if (size_.width != W || size_.height != H) {
			// (Frames already recorded may still use the old ones: kept until this upscaler is destroyed, after the device idles.)
			for (OwnImage* t: {&history_, &reprojected_, &recurrent_}) { old_images_.push_back(*t); *t = {}; }
			if (!CreateOwnImage(history_, VK_FORMAT_R16G16B16A16_SFLOAT, {W, H}, ss) ||
			    !CreateOwnImage(reprojected_, VK_FORMAT_R16G16B16A16_SFLOAT, {W, H}, ss) ||
			    !CreateOwnImage(recurrent_, VK_FORMAT_R8G8B8A8_UNORM, {W, H}, ss) ||
			    (exposure_.image == VK_NULL_HANDLE && !CreateOwnImage(exposure_, VK_FORMAT_R32_SFLOAT, {1, 1}, ss | VK_IMAGE_USAGE_TRANSFER_SRC_BIT))) {
				Log("[fsr4] its %ux%u images were not created: off", W, H);
				failed_ = true;
				return false;
			}
			size_ = {W, H};
			first_ = true;
		}
		if (render_.width != f.render.width || render_.height != f.render.height) {
			old_images_.push_back(depth_);
			depth_ = {};
			if (!CreateOwnImage(depth_, VK_FORMAT_R32_SFLOAT, f.render, ss)) { failed_ = true; return false; }
			render_ = f.render;
			first_ = true;
		}
		const VkCommandBuffer cb = f.cb;
		for (OwnImage* t: {&history_, &reprojected_, &recurrent_, &exposure_, &depth_})
			if (t->layout != VK_IMAGE_LAYOUT_GENERAL) Transition(cb, *t, VK_IMAGE_LAYOUT_GENERAL);

		Constants c {};
		c.inv_size[0]  = 1.0f / static_cast<float>(W);
		c.inv_size[1]  = 1.0f / static_cast<float>(H);
		c.scale[0]     = static_cast<float>(W) / static_cast<float>(f.render.width);
		c.scale[1]     = static_cast<float>(H) / static_cast<float>(f.render.height);
		c.inv_scale[0] = 1.0f / c.scale[0];
		c.inv_scale[1] = 1.0f / c.scale[1];
		c.jitter[0]    = f.jitter[0];
		c.jitter[1]    = f.jitter[1];
		// The model wants UV offsets to the previous frame: render pixels over the render size.
		const bool have = f.mv_scale[0] != 0.0f || f.mv_scale[1] != 0.0f;
		c.mv_scale[0] = have ? f.mv_scale[0] / static_cast<float>(f.render.width) : 0.0f;
		c.mv_scale[1] = have ? f.mv_scale[1] / static_cast<float>(f.render.height) : 0.0f;
		c.tex_size[0] = static_cast<float>(W);
		c.tex_size[1] = static_cast<float>(H);
		c.max_render_size[0] = static_cast<float>(f.render.width);
		c.max_render_size[1] = static_cast<float>(f.render.height);
		c.width     = W;
		c.height    = H;
		c.reset     = f.reset || first_ ? 1u : 0u;
		c.width_lr  = f.render.width;
		c.height_lr = f.render.height;
		c.pre_exposure = c.previous_pre_exposure = 1.0f;
		c.rcas_enabled = 0;  // (our RCAS after it)
		std::array<uint8_t, 256> cbuf {};
		std::memcpy(cbuf.data(), &c, sizeof(c));
		Update(cb, constants_, cbuf.data(), cbuf.size());
		if (!uploaded_) {
			Update(cb, init_, assets_.initializers.data(), assets_.initializers.size());
			Update(cb, weights_, assets_.pre_weights.data(), assets_.pre_weights.size());
			uploaded_ = true;
		}
		VkMemoryBarrier m {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		m.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		m.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
		R().vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);

		// The depth (flipped when reversed) and the exposure (fsr4_prep.comp).
		{
			VkDescriptorImageInfo  depth_in {nearest_, f.depth->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
			VkDescriptorBufferInfo values {constants_.buffer, 0, 256};
			VkDescriptorImageInfo  depth_out {VK_NULL_HANDLE, depth_.view, VK_IMAGE_LAYOUT_GENERAL};
			VkDescriptorImageInfo  exposure_out {VK_NULL_HANDLE, exposure_.view, VK_IMAGE_LAYOUT_GENERAL};
			VkWriteDescriptorSet w[4] {};
			for (uint32_t i = 0; i < 4; ++i) {
				w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				w[i].dstBinding = i;
				w[i].descriptorCount = 1;
			}
			w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &depth_in;
			w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         w[1].pBufferInfo = &values;
			w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          w[2].pImageInfo = &depth_out;
			w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          w[3].pImageInfo = &exposure_out;
			const uint32_t push[8] {0u, 0u, f.reversed ? 1u : 0u, f.render.width, f.render.height, 0, 0, 0};
			R().vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, prep_.pipeline);
			R().vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE, prep_.layout, 0, 4, w);
			R().vkCmdPushConstants(cb, prep_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, push);
			R().vkCmdDispatch(cb, Div(f.render.width, 8), Div(f.render.height, 8), 1);
			ComputeBarrier(cb);
		}

		// Each pass's descriptors by the manifest's names.
		const auto run = [&](Pipe& p, uint32_t x, uint32_t y) -> bool {
			const size_t n = p.module.resources.size();
			std::vector<VkDescriptorImageInfo>  images(n);
			std::vector<VkDescriptorBufferInfo> buffers(n);
			std::vector<VkWriteDescriptorSet>   writes;
			for (size_t i = 0; i < n; ++i) {
				const auto& r = p.module.resources[i];
				const auto& nm = r.name;
				VkWriteDescriptorSet w {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
				w.dstBinding      = r.binding;
				w.descriptorCount = 1;
				w.descriptorType  = DescriptorType(r.kind);
				const auto image = [&](VkImageView view, VkImageLayout layout) {
					images[i] = {VK_NULL_HANDLE, view, layout};
					w.pImageInfo = &images[i];
				};
				const auto buffer = [&](const OwnBuffer& b, VkDeviceSize size) {
					buffers[i] = {b.buffer, 0, size};
					w.pBufferInfo = &buffers[i];
				};
				if (nm == "r_input_color") image(f.color->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				else if (nm == "r_velocity") image(f.motion->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				else if (nm == "r_depth") image(depth_.view, VK_IMAGE_LAYOUT_GENERAL);
				else if (nm == "r_input_exposure") image(exposure_.view, VK_IMAGE_LAYOUT_GENERAL);
				else if (nm == "r_history_color" || nm == "rw_history_color") image(history_.view, VK_IMAGE_LAYOUT_GENERAL);
				else if (nm == "r_recurrent_0" || nm == "rw_recurrent_0") image(recurrent_.view, VK_IMAGE_LAYOUT_GENERAL);
				else if (nm == "r_reprojected_color" || nm == "rw_reprojected_color") image(reprojected_.view, VK_IMAGE_LAYOUT_GENERAL);
				else if (nm == "rw_mlsr_output_color") image(f.output->view, VK_IMAGE_LAYOUT_GENERAL);
				else if (nm == "g_history_sampler") {
					images[i] = {linear_, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
					w.pImageInfo = &images[i];
				} else if (nm == "ScratchBuffer") buffer(scratch_, scratch_.size);
				else if (nm == "InitializerBuffer") buffer(init_, assets_.initializers.size());
				else if (nm == "cbPass_Weights") buffer(weights_, 1024);
				else if (nm == "MLSR_Optimized_Constants") buffer(constants_, 256);
				else {
					Log("[fsr4] %s binding %u (%s) unknown to this runtime: off", p.module.path.filename().string().c_str(), r.binding, nm.c_str());
					return false;
				}
				writes.push_back(w);
			}
			R().vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
			R().vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, static_cast<uint32_t>(writes.size()), writes.data());
			R().vkCmdDispatch(cb, x, y, 1);
			ComputeBarrier(cb);
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
			if (!run(model_[i], d[i][0], d[i][1])) { failed_ = true; return false; }
		}
		if (frames_++ == 0)
			Log("[fsr4] first frame %ux%u -> %ux%u (%s, %s variant), depth %s, mv scale (%.5f, %.5f) UV, jitter (%.3f, %.3f)", f.render.width,
			    f.render.height, f.display.width, f.display.height, PresetTitles[preset_], res2160_ ? "2160" : "1080",
			    f.reversed ? "flipped (reversed)" : "as is", c.mv_scale[0], c.mv_scale[1], f.jitter[0], f.jitter[1]);
		first_ = false;
		return true;
	}

private:
	bool Build(int preset, bool res2160) {
		const auto start = std::chrono::steady_clock::now();
		for (auto& p: model_) { old_pipes_.push_back(p); p.pipeline = VK_NULL_HANDLE; p.layout = VK_NULL_HANDLE; p.set_layout = VK_NULL_HANDLE; }
		built_ = false;
		std::string why;
		if (!LoadPreset(preset, res2160, assets_, why)) { Log("[fsr4] %s: off", why.c_str()); return false; }
		assets_.initializers.resize((assets_.initializers.size() + 3) & ~size_t(3), 0);  // (vkCmdUpdateBuffer: multiples of 4)
		for (int i = 0; i < 14; ++i) {
			model_[i].module = assets_.modules[i];
			if (!BuildPipe(model_[i], 0)) {
				Log("[fsr4] %s: its pipeline was not created: off", model_[i].module.path.filename().string().c_str());
				return false;
			}
		}
		if (prep_.pipeline == VK_NULL_HANDLE) {
			prep_.module.code.assign(reinterpret_cast<const char*>(FSR4_PREP_SPV), reinterpret_cast<const char*>(FSR4_PREP_SPV) + sizeof(FSR4_PREP_SPV));
			prep_.module.entry = "main";
			Reflection reflected;
			if (!Reflect(prep_.module.code, reflected)) return false;
			prep_.module.resources = reflected.resources;
			if (!BuildPipe(prep_, 32)) return false;
		}
		if (linear_ == VK_NULL_HANDLE) {
			VkSamplerCreateInfo s {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
			s.magFilter = s.minFilter = VK_FILTER_LINEAR;
			s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
			s.addressModeU = s.addressModeV = s.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			if (R().vkCreateSampler(R().device, &s, nullptr, &linear_) != VK_SUCCESS) return false;
			s.magFilter = s.minFilter = VK_FILTER_NEAREST;
			if (R().vkCreateSampler(R().device, &s, nullptr, &nearest_) != VK_SUCCESS) return false;
		}
		const VkDeviceSize scratch = res2160 ? Scratch2160 : Scratch1080;
		const auto ensure = [](OwnBuffer& b, VkDeviceSize size, VkBufferUsageFlags usage) {
			if (b.buffer != VK_NULL_HANDLE && b.size >= size) return true;
			DestroyOwnBuffer(b);
			return CreateOwnBuffer(b, size, usage, false);
		};
		if (!ensure(scratch_, scratch, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
		    !ensure(init_, std::max<VkDeviceSize>((assets_.initializers.size() + 3) & ~size_t(3), 256), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
		    !ensure(weights_, 1024, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
		    !ensure(constants_, 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
			Log("[fsr4] its buffers were not created (%.1f MB scratch): off", static_cast<double>(scratch) / 1e6);
			return false;
		}
		uploaded_ = false;
		preset_   = preset;
		res2160_  = res2160;
		built_    = true;
		first_    = true;
		const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
		Log("[fsr4] %s model (%s, %s variant) loaded from %ls: 14 passes, descriptors as the manifest, %.0f ms", PresetTitles[preset],
		    ModelName(preset).c_str(), res2160 ? "2160" : "1080", AssetDir().wstring().c_str(), ms);
		return true;
	}

	Assets      assets_;
	std::array<Pipe, 14> model_ {};
	std::vector<Pipe>     old_pipes_;
	std::vector<OwnImage> old_images_;
	Pipe        prep_;
	VkSampler   linear_ = VK_NULL_HANDLE, nearest_ = VK_NULL_HANDLE;
	OwnBuffer   scratch_, init_, weights_, constants_;
	OwnImage    history_, reprojected_, recurrent_, exposure_, depth_;
	VkExtent2D  size_ {}, render_ {};
	int         preset_ = 1;
	bool        res2160_ = false, built_ = false, uploaded_ = false, first_ = true, failed_ = false, said_size_ = false;
	uint64_t    frames_ = 0;
};

} // namespace

bool Ready(std::string& why) {
	if (!AssetsPresent()) { why = "no esta la carpeta fsr4_shaders junto al exe"; return false; }
	if (!g_dev.checked || !g_dev.dot || !g_dev.int8 || !g_dev.int16 || !g_dev.float16 || !g_dev.derivatives) {
		why = "a la GPU le falta lo que pide el modelo INT8/DOT4 (dot product, int8, int16, fp16, derivadas en compute)";
		return false;
	}
	if (!PushDescriptorsAvailable()) { why = "sin VK_KHR_push_descriptor"; return false; }
	if (!g_ready_checked) {  // every preset's 1080 / 2160 variant checked once, so a broken package shows at once
		g_ready_checked = true;
		int checked = 0;
		std::string first_why;
		for (int preset = 0; preset < 5; ++preset) {
			for (bool res2160: {false, true}) {
				Assets a;
				std::string w;
				if (LoadPreset(preset, res2160, a, w)) ++checked;
				else if (first_why.empty()) first_why = w;
			}
		}
		g_ready = checked == 10;
		g_why   = g_ready ? std::string() : "fsr4_shaders incompleto: " + first_why;
		Log("[fsr4] %d of 10 preset variants in %ls match their manifest (descriptors, entry points, local sizes, sizes)%s%s", checked,
		    AssetDir().wstring().c_str(), g_ready ? "" : ": ", first_why.c_str());
	}
	why = g_why;
	return g_ready;
}

std::unique_ptr<Upscaler> Create(std::string& why) {
	if (!Ready(why)) return nullptr;
	return std::make_unique<Fsr4>();
}

#else  // built without nlohmann/json (the manifest reader)

bool Ready(std::string& why) {
	why = "esta build no incluye FSR 4";
	return false;
}
std::unique_ptr<Upscaler> Create(std::string& why) {
	why = "this build has no FSR 4";
	return nullptr;
}

#endif

} // namespace inj::upscale::fsr4
