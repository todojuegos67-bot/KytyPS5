#include "upscale.h"

#include "camera.h"
#include "overlay.h"
#include "upscale_backend.h"

#include "shaders/metric_spv.h"
#include "shaders/rcas_spv.h"

#include <imgui.h>

#include <algorithm>
#include <cstdarg>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace inj::upscale {

// ================================================================ names, modes
const char* KindName(int kind) {
	switch (kind) {
	case kDlss: return "DLSS";
	case kFsr4: return "FSR 4";
	case kFsr3: return "FSR 3.1";
	case kXess: return "XeSS";
	default: return "off";
	}
}
const char* QualityName(int q) {
	switch (q) {
	case 1: return "Native AA";
	case 2: return "Quality";
	case 3: return "Balanced";
	case 4: return "Performance";
	case 5: return "Ultra Performance";
	default: return "Auto";
	}
}
int QualityForRatio(float r) {
	if (r < 1.15f) return 1;
	if (r < 1.60f) return 2;
	if (r < 1.85f) return 3;
	if (r < 2.50f) return 4;
	return 5;
}
float ScaleForQuality(int q) {
	switch (q) {
	case 1: return 1.0f;
	case 2: return 1.0f / 1.5f;
	case 3: return 1.0f / 1.7f;
	case 4: return 0.5f;
	case 5: return 1.0f / 3.0f;
	default: return 1.0f / 1.5f;
	}
}

// ================================================================ Vulkan helpers
bool IsDepthFormat(VkFormat f) {
	switch (f) {
	case VK_FORMAT_D16_UNORM:
	case VK_FORMAT_X8_D24_UNORM_PACK32:
	case VK_FORMAT_D32_SFLOAT:
	case VK_FORMAT_D16_UNORM_S8_UINT:
	case VK_FORMAT_D24_UNORM_S8_UINT:
	case VK_FORMAT_D32_SFLOAT_S8_UINT: return true;
	default: return false;
	}
}
bool HasStencil(VkFormat f) {
	return f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}
bool IsFloatFormat(VkFormat f) {
	switch (f) {
	case VK_FORMAT_R16G16B16A16_SFLOAT:
	case VK_FORMAT_R32G32B32A32_SFLOAT:
	case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
	case VK_FORMAT_R16G16B16_SFLOAT:
	case VK_FORMAT_R32G32B32_SFLOAT: return true;
	default: return false;
	}
}
VkImageAspectFlags ViewAspect(VkFormat f) {
	return IsDepthFormat(f) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
}
VkImageAspectFlags BarrierAspect(VkFormat f) {
	if (!IsDepthFormat(f)) return VK_IMAGE_ASPECT_COLOR_BIT;
	return VK_IMAGE_ASPECT_DEPTH_BIT | (HasStencil(f) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
}

void Transition(VkCommandBuffer cb, VkImage image, VkFormat format, VkImageLayout from, VkImageLayout to) {
	if (from == to && from != VK_IMAGE_LAYOUT_GENERAL) return;
	VkImageMemoryBarrier b {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
	b.srcAccessMask       = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
	b.dstAccessMask       = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
	b.oldLayout           = from;
	b.newLayout           = to;
	b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image               = image;
	b.subresourceRange    = {BarrierAspect(format), 0, 1, 0, 1};
	R().vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}
void Transition(VkCommandBuffer cb, OwnImage& img, VkImageLayout to) {
	Transition(cb, img.image, img.format, img.layout, to);
	img.layout = to;
}
void FullBarrier(VkCommandBuffer cb) {
	VkMemoryBarrier m {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	m.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
	m.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
	R().vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);
}

bool CreateOwnImage(OwnImage& img, VkFormat format, VkExtent2D extent, VkImageUsageFlags usage) {
	img = {};
	VkImageCreateInfo ci {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	ci.imageType   = VK_IMAGE_TYPE_2D;
	ci.format      = format;
	ci.extent      = {extent.width, extent.height, 1};
	ci.mipLevels   = 1;
	ci.arrayLayers = 1;
	ci.samples     = VK_SAMPLE_COUNT_1_BIT;
	ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
	ci.usage       = usage;
	if (R().vkCreateImage(R().device, &ci, nullptr, &img.image) != VK_SUCCESS) { img.image = VK_NULL_HANDLE; return false; }
	VkMemoryRequirements req {};
	R().vkGetImageMemoryRequirements(R().device, img.image, &req);
	VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	ai.allocationSize  = req.size;
	ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (R().vkAllocateMemory(R().device, &ai, nullptr, &img.memory) != VK_SUCCESS ||
	    R().vkBindImageMemory(R().device, img.image, img.memory, 0) != VK_SUCCESS) {
		DestroyOwnImage(img);
		return false;
	}
	VkImageViewCreateInfo vi {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
	vi.image            = img.image;
	vi.viewType         = VK_IMAGE_VIEW_TYPE_2D;
	vi.format           = format;
	vi.subresourceRange = {ViewAspect(format), 0, 1, 0, 1};
	if (R().vkCreateImageView(R().device, &vi, nullptr, &img.view) != VK_SUCCESS) {
		DestroyOwnImage(img);
		return false;
	}
	img.format = format;
	img.extent = extent;
	img.usage  = usage;
	img.layout = VK_IMAGE_LAYOUT_UNDEFINED;
	return true;
}
void DestroyOwnImage(OwnImage& img) {
	if (R().device == VK_NULL_HANDLE) { img = {}; return; }
	if (img.view) R().vkDestroyImageView(R().device, img.view, nullptr);
	if (img.image) R().vkDestroyImage(R().device, img.image, nullptr);
	if (img.memory) R().vkFreeMemory(R().device, img.memory, nullptr);
	img = {};
}

bool CreateOwnBuffer(OwnBuffer& b, VkDeviceSize size, VkBufferUsageFlags usage, bool host) {
	b = {};
	VkBufferCreateInfo ci {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	ci.size  = size;
	ci.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	if (R().vkCreateBuffer(R().device, &ci, nullptr, &b.buffer) != VK_SUCCESS) { b.buffer = VK_NULL_HANDLE; return false; }
	VkMemoryRequirements req {};
	R().vkGetBufferMemoryRequirements(R().device, b.buffer, &req);
	VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	ai.allocationSize  = req.size;
	ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, host ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
	                                                              : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (R().vkAllocateMemory(R().device, &ai, nullptr, &b.memory) != VK_SUCCESS ||
	    R().vkBindBufferMemory(R().device, b.buffer, b.memory, 0) != VK_SUCCESS) {
		DestroyOwnBuffer(b);
		return false;
	}
	if (host && R().vkMapMemory(R().device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped) != VK_SUCCESS) {
		DestroyOwnBuffer(b);
		return false;
	}
	b.size = size;
	return true;
}
void DestroyOwnBuffer(OwnBuffer& b) {
	if (R().device == VK_NULL_HANDLE) { b = {}; return; }
	if (b.buffer) R().vkDestroyBuffer(R().device, b.buffer, nullptr);
	if (b.memory) R().vkFreeMemory(R().device, b.memory, nullptr);  // (unmapped with it)
	b = {};
}

namespace {
bool g_push_descriptors = false;
std::unordered_set<std::string> g_device_extensions;
} // namespace

bool PushDescriptorsAvailable() {
	return g_push_descriptors && R().vkCmdPushDescriptorSetKHR != nullptr;
}
bool DeviceHas(const char* extension) {
	return g_device_extensions.count(extension) != 0;
}

bool CreateComputePipe(ComputePipe& p, const uint32_t* code, size_t bytes, const char* entry,
                       const std::vector<VkDescriptorType>& bindings, uint32_t push_bytes) {
	p = {};
	std::vector<VkDescriptorSetLayoutBinding> b(bindings.size());
	for (size_t i = 0; i < bindings.size(); ++i) {
		b[i].binding         = static_cast<uint32_t>(i);
		b[i].descriptorType  = bindings[i];
		b[i].descriptorCount = 1;
		b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	VkDescriptorSetLayoutCreateInfo si {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	si.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
	si.bindingCount = static_cast<uint32_t>(b.size());
	si.pBindings    = b.data();
	if (R().vkCreateDescriptorSetLayout(R().device, &si, nullptr, &p.set_layout) != VK_SUCCESS) { p = {}; return false; }
	VkPushConstantRange range {VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
	VkPipelineLayoutCreateInfo li {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	li.setLayoutCount         = 1;
	li.pSetLayouts            = &p.set_layout;
	li.pushConstantRangeCount = push_bytes ? 1u : 0u;
	li.pPushConstantRanges    = push_bytes ? &range : nullptr;
	if (R().vkCreatePipelineLayout(R().device, &li, nullptr, &p.layout) != VK_SUCCESS) { DestroyComputePipe(p); return false; }
	VkShaderModuleCreateInfo mi {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	mi.codeSize = bytes;
	mi.pCode    = code;
	VkShaderModule module = VK_NULL_HANDLE;
	if (R().vkCreateShaderModule(R().device, &mi, nullptr, &module) != VK_SUCCESS) { DestroyComputePipe(p); return false; }
	VkComputePipelineCreateInfo pi {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	pi.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pi.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	pi.stage.module = module;
	pi.stage.pName  = entry;
	pi.layout       = p.layout;
	const VkResult r = R().vkCreateComputePipelines(R().device, VK_NULL_HANDLE, 1, &pi, nullptr, &p.pipeline);
	R().vkDestroyShaderModule(R().device, module, nullptr);
	if (r != VK_SUCCESS) { p.pipeline = VK_NULL_HANDLE; DestroyComputePipe(p); return false; }
	return true;
}
void DestroyComputePipe(ComputePipe& p) {
	if (R().device == VK_NULL_HANDLE) { p = {}; return; }
	if (p.pipeline) R().vkDestroyPipeline(R().device, p.pipeline, nullptr);
	if (p.layout) R().vkDestroyPipelineLayout(R().device, p.layout, nullptr);
	if (p.set_layout) R().vkDestroyDescriptorSetLayout(R().device, p.set_layout, nullptr);
	p = {};
}

// ================================================================ instance / device creation
void BeforeCreateInstance() {
	InternalScope scope;  // (Streamline and XeSS load "vulkan-1.dll": ours)
	if (Cfg().up_streamline) dlss::BeforeCreateInstance();
	xess::BeforeCreateInstance();
}

void InstanceExtensions(const std::vector<VkExtensionProperties>& available, std::vector<const char*>& extensions) {
	std::vector<const char*> wanted;
	dlss::InstanceExtensions(wanted);
	xess::InstanceExtensions(wanted);
	for (const char* name: wanted) {
		const bool has = std::any_of(available.begin(), available.end(), [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
		const bool added = std::any_of(extensions.begin(), extensions.end(), [&](const char* e) { return std::strcmp(e, name) == 0; });
		if (has && !added) {
			extensions.push_back(name);
			Log("[up] instance extension added: %s", name);
		} else if (!has) {
			Log("[up] instance extension %s not on this system", name);
		}
	}
}

void DevicePatch::Restore() {
	for (size_t i = 0; i < saved.size(); ++i) std::memcpy(saved_at[i], saved[i].data(), saved[i].size());
}

namespace {

struct ChainNode { VkStructureType sType; void* pNext; };

void* FindInChain(const void* head, VkStructureType type) {
	for (auto* n = static_cast<const ChainNode*>(head); n != nullptr; n = static_cast<const ChainNode*>(n->pNext))
		if (n->sType == type) return const_cast<ChainNode*>(n);
	return nullptr;
}

// dst |= src & support, over the VkBool32 fields after sType / pNext.
void OrFeatures(void* dst, const void* src, const void* support, size_t size) {
	constexpr size_t head = sizeof(VkStructureType) + sizeof(void*) + (sizeof(void*) - sizeof(VkStructureType));
	for (size_t at = head; at + sizeof(VkBool32) <= size; at += sizeof(VkBool32)) {
		VkBool32 s = VK_FALSE, sup = VK_TRUE, d = VK_FALSE;
		std::memcpy(&s, static_cast<const uint8_t*>(src) + at, sizeof(s));
		if (support != nullptr) std::memcpy(&sup, static_cast<const uint8_t*>(support) + at, sizeof(sup));
		std::memcpy(&d, static_cast<uint8_t*>(dst) + at, sizeof(d));
		if (s && sup && !d) {
			d = VK_TRUE;
			std::memcpy(static_cast<uint8_t*>(dst) + at, &d, sizeof(d));
		}
	}
}

uint32_t g_queue_counts[16] {};  // queues the device was made with, per family (Streamline's queue base)

} // namespace

void PatchDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo& in, VkDeviceCreateInfo& out, DevicePatch& patch) {
	auto& r = R();
	auto enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(LoaderProc("vkEnumerateDeviceExtensionProperties"));
	auto features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(LoaderProc("vkGetPhysicalDeviceFeatures2"));
	std::vector<VkExtensionProperties> available;
	if (enumerate != nullptr) {
		uint32_t n = 0;
		enumerate(physical, nullptr, &n, nullptr);
		available.resize(n);
		enumerate(physical, nullptr, &n, available.data());
		available.resize(n);
	}
	const auto has = [&](const char* name) {
		return std::any_of(available.begin(), available.end(), [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
	};
	patch.extensions.assign(in.ppEnabledExtensionNames, in.ppEnabledExtensionNames + in.enabledExtensionCount);
	const auto added = [&](const char* name) {
		return std::any_of(patch.extensions.begin(), patch.extensions.end(), [&](const char* e) { return std::strcmp(e, name) == 0; });
	};
	const auto add = [&](const char* name, const char* who) {
		if (added(name)) return true;
		if (!has(name)) {
			Log("[up] device extension %s (%s) not on this device", name, who);
			return false;
		}
		patch.extensions.push_back(name);
		Log("[up] device extension added: %s (%s)", name, who);
		return true;
	};
	add(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, "our passes, FSR 4");
	std::vector<const char*> wanted;
	fsr4::DeviceExtensions(available, wanted);
	for (const char* n: wanted) add(n, "FSR 4");
	const bool derivatives_ext = added("VK_KHR_compute_shader_derivatives") || added(VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
	wanted.clear();
	xess::DeviceExtensions(physical, wanted);
	for (const char* n: wanted) add(n, "XeSS");
	wanted.clear();
	dlss::DeviceExtensions(wanted);
	for (const char* n: wanted) add(n, "DLSS");
	out.enabledExtensionCount   = static_cast<uint32_t>(patch.extensions.size());
	out.ppEnabledExtensionNames = patch.extensions.data();

	// Features: ORed into the game's VkPhysicalDeviceFeatures2 / Vulkan 1.2 / 1.3 structures (where supported).
	auto* f2  = static_cast<VkPhysicalDeviceFeatures2*>(FindInChain(in.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2));
	auto* f12 = static_cast<VkPhysicalDeviceVulkan12Features*>(FindInChain(in.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES));
	auto* f13 = static_cast<VkPhysicalDeviceVulkan13Features*>(FindInChain(in.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES));
	if (f2 == nullptr || f12 == nullptr || f13 == nullptr || features2 == nullptr) {
		Log("[up] the game's device has no Vulkan 1.2 / 1.3 feature structures (%d %d %d): no features added (FSR 4 / DLSS may be off)",
		    f2 != nullptr, f12 != nullptr, f13 != nullptr);
		return;
	}
	for (void* s: {static_cast<void*>(f2), static_cast<void*>(f12), static_cast<void*>(f13)}) {
		const size_t size = FeatureStructSize(static_cast<ChainNode*>(s)->sType);
		patch.saved.emplace_back(static_cast<uint8_t*>(s), static_cast<uint8_t*>(s) + size);
		patch.saved_at.push_back(s);
	}
	// What the device has.
	VkPhysicalDeviceVulkan13Features s13 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
	VkPhysicalDeviceVulkan12Features s12 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	VkPhysicalDeviceFeatures2        s2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	s2.pNext  = &s12;
	s12.pNext = &s13;
	features2(physical, &s2);
	// What the backends want.
	VkPhysicalDeviceVulkan13Features w13 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
	VkPhysicalDeviceVulkan12Features w12 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	VkPhysicalDeviceFeatures2        w2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	bool want_derivatives = false;
	fsr4::DeviceFeatures(physical, w2.features, w12, w13, derivatives_ext, want_derivatives);
	dlss::DeviceFeatures(w12, w13);
	OrFeatures(f2, &w2, &s2, sizeof(VkPhysicalDeviceFeatures2));
	OrFeatures(f12, &w12, &s12, sizeof(VkPhysicalDeviceVulkan12Features));
	OrFeatures(f13, &w13, &s13, sizeof(VkPhysicalDeviceVulkan13Features));
	// New structures chained in front of the game's.
	const auto chain = [&](const void* src, size_t size) {
		patch.owned.emplace_back(static_cast<const uint8_t*>(src), static_cast<const uint8_t*>(src) + size);
		auto* node  = reinterpret_cast<ChainNode*>(patch.owned.back().data());
		node->pNext = const_cast<void*>(out.pNext);
		out.pNext   = node;
	};
	if (want_derivatives && FindInChain(in.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV) == nullptr) {
		VkPhysicalDeviceComputeShaderDerivativesFeaturesNV d {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV};
		d.computeDerivativeGroupLinear = VK_TRUE;
		chain(&d, sizeof(d));
	}
	// XeSS: its own structures, merged (ORed into ours where the game has the same type, else chained in).
	if (void* x = xess::DeviceFeatureChain(physical)) {
		for (auto* n = static_cast<ChainNode*>(x); n != nullptr;) {
			auto* next = static_cast<ChainNode*>(n->pNext);
			const size_t size = FeatureStructSize(n->sType);
			void* mine = FindInChain(out.pNext, n->sType);
			const void* support = n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 ? static_cast<const void*>(&s2)
			                    : n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES ? static_cast<const void*>(&s12)
			                    : n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES ? static_cast<const void*>(&s13)
			                                                                                      : nullptr;
			if (mine != nullptr && size != 0) {
				OrFeatures(mine, n, support, size);
			} else if (mine == nullptr && size != 0) {
				chain(n, size);
			} else {
				Log("[up] XeSS asks for feature structure %d (unknown size): not merged", n->sType);
			}
			n = next;
		}
	}
}

namespace {
struct DeviceRecord { std::unordered_set<std::string> extensions; uint32_t queues[16] {}; };
std::mutex g_records_mutex;
std::unordered_map<VkDevice, DeviceRecord> g_records;
} // namespace

void OnDeviceCreated(VkDevice device, const VkDeviceCreateInfo& info) {
	DeviceRecord rec;
	for (uint32_t i = 0; i < info.enabledExtensionCount; ++i) rec.extensions.insert(info.ppEnabledExtensionNames[i]);
	for (uint32_t i = 0; i < info.queueCreateInfoCount; ++i)
		if (info.pQueueCreateInfos[i].queueFamilyIndex < 16) rec.queues[info.pQueueCreateInfos[i].queueFamilyIndex] += info.pQueueCreateInfos[i].queueCount;
	Log("[up] device %p made with %u extensions (push descriptors %s)", static_cast<void*>(device), info.enabledExtensionCount,
	    rec.extensions.count(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) ? "on" : "off");
	std::lock_guard lock(g_records_mutex);
	g_records[device] = std::move(rec);
}

void OnActivate(VkDevice device) {
	std::lock_guard lock(g_records_mutex);
	auto it = g_records.find(device);
	if (it == g_records.end()) return;
	g_device_extensions = it->second.extensions;
	g_push_descriptors  = DeviceHas(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
	std::copy(std::begin(it->second.queues), std::end(it->second.queues), std::begin(g_queue_counts));
}
uint32_t QueueCount(uint32_t family) {
	return family < 16 ? g_queue_counts[family] : 0;
}

// ================================================================ command buffer state (restored after our work)
namespace {

struct Binds {  // descriptor-set binds of one bind point, in order; a later bind that covers an earlier one replaces it
	struct Bind { VkPipelineLayout layout; uint32_t first; std::vector<VkDescriptorSet> sets; std::vector<uint32_t> dynamic; };
	std::vector<Bind> list;
	void Add(VkPipelineLayout layout, uint32_t first, uint32_t count, const VkDescriptorSet* sets, uint32_t dynamic_count, const uint32_t* dynamic) {
		list.erase(std::remove_if(list.begin(), list.end(), [&](const Bind& b) {
			           return b.first >= first && b.first + b.sets.size() <= first + count;
		           }),
		           list.end());
		if (list.size() >= 8) list.erase(list.begin());
		list.push_back({layout, first, std::vector<VkDescriptorSet>(sets, sets + count),
		                std::vector<uint32_t>(dynamic, dynamic + (dynamic != nullptr ? dynamic_count : 0))});
	}
	void Replay(VkCommandBuffer cb, VkPipelineBindPoint point) const {
		for (const auto& b: list)
			R().vkCmdBindDescriptorSets(cb, point, b.layout, b.first, static_cast<uint32_t>(b.sets.size()), b.sets.data(),
			                            static_cast<uint32_t>(b.dynamic.size()), b.dynamic.data());
	}
};
struct CbState {
	VkPipeline compute = VK_NULL_HANDLE, graphics = VK_NULL_HANDLE;
	Binds      compute_sets, graphics_sets;
	bool       graphics_disturbed = false;  // a third-party runtime bound graphics state inside our work
	struct Push { VkPipelineLayout layout; VkShaderStageFlags stages; uint32_t offset; std::vector<uint8_t> data; };
	std::vector<Push> pushes;  // the last per (stages, offset)
};
std::mutex g_cb_mutex;
std::unordered_map<VkCommandBuffer, CbState> g_cbs;
thread_local VkCommandBuffer t_cb = VK_NULL_HANDLE;
thread_local CbState*        t_state = nullptr;

CbState& StateOf(VkCommandBuffer cb) {
	if (t_cb == cb && t_state != nullptr) return *t_state;
	std::lock_guard lock(g_cb_mutex);
	t_cb    = cb;
	t_state = &g_cbs[cb];  // (node-based: the reference stays valid)
	return *t_state;
}

void Restore(VkCommandBuffer cb) {
	CbState& s = StateOf(cb);
	if (s.compute != VK_NULL_HANDLE) R().vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.compute);
	s.compute_sets.Replay(cb, VK_PIPELINE_BIND_POINT_COMPUTE);
	if (s.graphics_disturbed) {
		if (s.graphics != VK_NULL_HANDLE) R().vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s.graphics);
		s.graphics_sets.Replay(cb, VK_PIPELINE_BIND_POINT_GRAPHICS);
		s.graphics_disturbed = false;
	}
	for (const auto& p: s.pushes)
		R().vkCmdPushConstants(cb, p.layout, p.stages, p.offset, static_cast<uint32_t>(p.data.size()), p.data.data());
}

} // namespace

void OnBeginCommandBuffer(VkCommandBuffer cb) {
	CbState& s = StateOf(cb);
	s.compute = s.graphics = VK_NULL_HANDLE;
	s.compute_sets.list.clear();
	s.graphics_sets.list.clear();
	s.graphics_disturbed = false;
	s.pushes.clear();
}
void OnBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipeline pipeline) {
	CbState& s = StateOf(cb);
	if (Internal()) {  // (a third-party runtime inside our work: only noted)
		if (point == VK_PIPELINE_BIND_POINT_GRAPHICS) s.graphics_disturbed = true;
		return;
	}
	if (point == VK_PIPELINE_BIND_POINT_COMPUTE) s.compute = pipeline;
	else if (point == VK_PIPELINE_BIND_POINT_GRAPHICS) s.graphics = pipeline;
}
void OnBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipelineLayout layout, uint32_t first, uint32_t count,
                          const VkDescriptorSet* sets, uint32_t dynamic_count, const uint32_t* dynamic) {
	if (count == 0) return;
	CbState& s = StateOf(cb);
	if (Internal()) {
		if (point == VK_PIPELINE_BIND_POINT_GRAPHICS) s.graphics_disturbed = true;
		return;
	}
	if (point == VK_PIPELINE_BIND_POINT_COMPUTE) s.compute_sets.Add(layout, first, count, sets, dynamic_count, dynamic);
	else if (point == VK_PIPELINE_BIND_POINT_GRAPHICS) s.graphics_sets.Add(layout, first, count, sets, dynamic_count, dynamic);
}
void OnPushConstants(VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stages, uint32_t offset, uint32_t size,
                     const void* data) {
	if (Internal()) return;
	CbState& s = StateOf(cb);
	for (auto& p: s.pushes) {
		if (p.stages == stages && p.offset == offset) {
			p.layout = layout;
			p.data.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
			return;
		}
	}
	if (s.pushes.size() < 16) s.pushes.push_back({layout, stages, offset, std::vector<uint8_t>(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size)});
}

// ================================================================ the frame
namespace {

using Clock = std::chrono::steady_clock;

std::mutex g_mutex;  // everything below

VkExtent2D g_display {};
VkFormat   g_display_format = VK_FORMAT_UNDEFINED;

struct Targets {
	OwnImage color, depth, motion, zero_motion, output, sharp, crop_game, crop_ours;
};
Targets     g_t;
OwnBuffer   g_metric;           // 4 uints: game, ours (host visible)
VkQueryPool g_queries = VK_NULL_HANDLE;
float       g_ts_period = 1.0f;
ComputePipe g_rcas, g_metric_pipe;
VkSampler   g_point = VK_NULL_HANDLE;
bool        g_pipes_tried = false, g_pipes_ok = false;

std::unique_ptr<Upscaler> g_backend;
int         g_backend_kind = kOff;
int         g_failed_kind  = kOff;  // creation failed for this one (retried now and then, or when the menu changes it)
uint64_t    g_failed_at    = 0;
std::string g_backend_why;

struct Retired {
	std::unique_ptr<Upscaler> backend;
	std::vector<OwnImage>     images;
	uint64_t                  at = 0;
};
std::vector<Retired> g_retired;

struct FrameMarks { int bb_passes = 0; bool done = false; bool camera = false; };
FrameMarks g_frame;
uint64_t   g_runs = 0, g_dispatched = 0, g_presents = 0;
bool       g_ran_last = false;
VkImage    g_last_input = VK_NULL_HANDLE;
VkExtent2D g_last_render {};
int        g_last_quality = 0;
Clock::time_point g_last_time {};

// Status (menu, OSD, log).
struct Status {
	std::string state = "off";
	std::string where;
	std::string inputs;
	std::string detail;
	VkExtent2D  render {}, display {};
	float       ratio = 0;
	int         quality = 0;
	float       jitter[2] {0, 0};
	const char* jitter_kind = "none";
	float       gpu_ms = 0;
	float       metric_game = 0, metric_ours = 0;
	uint64_t    metric_frame = 0;
	bool        hud_covered = false;
	bool        running = false;
	std::string why_not;
};
Status g_status;
bool   g_logged_on = false;
char   g_osd[192] {};

// Readbacks: timestamps (4 slots) and the metric (one pending).
struct Pending { uint64_t index = 0; bool valid = false; };
Pending  g_ts_pending[4];
Pending  g_metric_pending;
bool     g_metric_has_ours = false;

void RetireTargets(OwnImage& img) {
	if (img.image == VK_NULL_HANDLE) return;
	if (g_retired.empty() || g_retired.back().at != g_presents) g_retired.push_back({nullptr, {}, g_presents});
	g_retired.back().images.push_back(img);
	img = {};
}

bool Ensure(OwnImage& img, VkFormat format, VkExtent2D extent, VkImageUsageFlags usage) {
	if (img.image != VK_NULL_HANDLE && img.format == format && img.extent.width == extent.width && img.extent.height == extent.height &&
	    (img.usage & usage) == usage)
		return true;
	RetireTargets(img);
	if (!CreateOwnImage(img, format, extent, usage)) {
		Log("[up] image %ux%u format %d not created", extent.width, extent.height, format);
		return false;
	}
	return true;
}

bool EnsurePipes() {
	if (g_pipes_tried) return g_pipes_ok;
	g_pipes_tried = true;
	if (!PushDescriptorsAvailable()) {
		Log("[up] VK_KHR_push_descriptor not enabled: no RCAS sharpening, no sharpness metric");
		return false;
	}
	g_pipes_ok = CreateComputePipe(g_rcas, RCAS_SPV, sizeof(RCAS_SPV), "main",
	                               {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}, 16) &&
	             CreateComputePipe(g_metric_pipe, METRIC_SPV, sizeof(METRIC_SPV), "main",
	                               {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, 4);
	if (g_pipes_ok) {
		VkSamplerCreateInfo si {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
		si.magFilter = si.minFilter = VK_FILTER_NEAREST;
		si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		g_pipes_ok = R().vkCreateSampler(R().device, &si, nullptr, &g_point) == VK_SUCCESS;
	}
	if (g_pipes_ok) g_pipes_ok = CreateOwnBuffer(g_metric, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
	VkQueryPoolCreateInfo qi {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
	qi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
	qi.queryCount = 8;
	if (R().vkCreateQueryPool(R().device, &qi, nullptr, &g_queries) != VK_SUCCESS) g_queries = VK_NULL_HANDLE;
	VkPhysicalDeviceProperties props {};
	R().vkGetPhysicalDeviceProperties(R().physical, &props);
	g_ts_period = props.limits.timestampPeriod;
	Log("[up] RCAS / metric pipelines %s, timestamps %s", g_pipes_ok ? "ready" : "NOT created", g_queries ? "on" : "off");
	return g_pipes_ok;
}

void PushImage(VkWriteDescriptorSet& w, VkDescriptorImageInfo& info, uint32_t binding, VkDescriptorType type, VkImageView view,
               VkImageLayout layout) {
	info  = {type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ? g_point : VK_NULL_HANDLE, view, layout};
	w     = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
	w.dstBinding      = binding;
	w.descriptorCount = 1;
	w.descriptorType  = type;
	w.pImageInfo      = &info;
}

// RCAS: src (GENERAL) -> dst (GENERAL).
void Sharpen(VkCommandBuffer cb, OwnImage& src, OwnImage& dst, float sharpness, bool hdr) {
	Transition(cb, dst, VK_IMAGE_LAYOUT_GENERAL);
	VkWriteDescriptorSet w[2];
	VkDescriptorImageInfo i[2];
	PushImage(w[0], i[0], 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, src.view, VK_IMAGE_LAYOUT_GENERAL);
	PushImage(w[1], i[1], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, dst.view, VK_IMAGE_LAYOUT_GENERAL);
	struct { float sharpness; uint32_t hdr, width, height; } push {std::exp2(-(2.0f - 2.0f * sharpness)), hdr ? 1u : 0u,
	                                                                 src.extent.width, src.extent.height};
	R().vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_rcas.pipeline);
	R().vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_rcas.layout, 0, 2, w);
	R().vkCmdPushConstants(cb, g_rcas.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
	R().vkCmdDispatch(cb, (src.extent.width + 7) / 8, (src.extent.height + 7) / 8, 1);
	FullBarrier(cb);
}

// The centre 256x256 of the backbuffer into `crop` (SHADER_READ_ONLY after).
void Crop(VkCommandBuffer cb, VkImage bb, VkFormat bb_format, VkImageLayout bb_layout, OwnImage& crop) {
	Transition(cb, bb, bb_format, bb_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	Transition(cb, crop, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	VkImageBlit blit {};
	blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	const int32_t x = static_cast<int32_t>(g_display.width / 2) - 128, y = static_cast<int32_t>(g_display.height / 2) - 128;
	blit.srcOffsets[0] = {x, y, 0};
	blit.srcOffsets[1] = {x + 256, y + 256, 1};
	blit.dstOffsets[1] = {256, 256, 1};
	R().vkCmdBlitImage(cb, bb, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, crop.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
	Transition(cb, bb, bb_format, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, bb_layout);
	Transition(cb, crop, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void Metric(VkCommandBuffer cb, OwnImage& crop, uint32_t slot) {
	VkWriteDescriptorSet w[2];
	VkDescriptorImageInfo i;
	PushImage(w[0], i, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, crop.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	VkDescriptorBufferInfo b {g_metric.buffer, 0, 64};
	w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
	w[1].dstBinding      = 1;
	w[1].descriptorCount = 1;
	w[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	w[1].pBufferInfo     = &b;
	R().vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_metric_pipe.pipeline);
	R().vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_metric_pipe.layout, 0, 2, w);
	R().vkCmdPushConstants(cb, g_metric_pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &slot);
	R().vkCmdDispatch(cb, 16, 16, 1);
}

// The game's image into our copy (left in SHADER_READ_ONLY_OPTIMAL; the game's left as it was).
bool CopyIn(VkCommandBuffer cb, const analyzer::Image& src, OwnImage& dst, const char* what, std::string& why) {
	if (src.layout == VK_IMAGE_LAYOUT_UNDEFINED || src.layout == VK_IMAGE_LAYOUT_PREINITIALIZED) {
		why = std::string(what) + ": its layout is not known yet";
		return false;
	}
	if (!(src.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
		why = std::string(what) + ": not copyable (made before the injector could add TRANSFER_SRC)";
		return false;
	}
	const VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	if (!Ensure(dst, src.format, src.extent, usage)) {
		why = std::string(what) + ": our copy not created";
		return false;
	}
	Transition(cb, src.image, src.format, src.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	Transition(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	VkImageCopy region {};
	region.srcSubresource = {ViewAspect(src.format), 0, 0, 1};
	region.dstSubresource = region.srcSubresource;
	region.extent         = {src.extent.width, src.extent.height, 1};
	R().vkCmdCopyImage(cb, src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
	Transition(cb, src.image, src.format, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, src.layout);
	Transition(cb, dst, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	return true;
}

std::unique_ptr<Upscaler> CreateBackend(int kind, std::string& why) {
	switch (kind) {
	case kDlss: return dlss::Create(why);
	case kFsr4: return fsr4::Create(why);
	case kFsr3: return fsr3::Create(why);
	case kXess: return xess::Create(why);
	default: why = "off"; return nullptr;
	}
}

std::string Fmt(const char* fmt, ...) {
	char buf[512];
	va_list args;
	va_start(args, fmt);
	std::vsnprintf(buf, sizeof buf, fmt, args);
	va_end(args);
	return buf;
}

// The work, right before the game's pass `where` (outside any render pass).
void Run(VkCommandBuffer cb, VkImageLayout bb_layout, const char* where, bool hud_covered) {
	g_frame.done = true;
	auto& s = Cfg();
	const auto v = analyzer::View(cb);
	const bool pre = s.up_input == 1 && v.pretaa.image != VK_NULL_HANDLE && v.pretaa.confidence >= s.min_confidence;
	const analyzer::Image& color = pre ? v.pretaa : v.scene;
	auto skip = [&](std::string why) {
		if (g_status.why_not != why) Log("[up] %s waiting: %s", KindName(s.up_backend), why.c_str());
		g_status.why_not = why;
		g_status.running = false;
		g_status.state   = "esperando";
		g_ran_last = false;
	};
	if (v.backbuffer.image == VK_NULL_HANDLE || v.backbuffer.extent.width != g_display.width || v.backbuffer.extent.height != g_display.height)
		return skip("the backbuffer (display-size image the frame ends in) is not found yet");
	if (v.depth.image == VK_NULL_HANDLE || v.depth.confidence < s.min_confidence) return skip("the depth buffer is not found yet");
	if (color.image == VK_NULL_HANDLE || color.confidence < s.min_confidence * 0.5f) return skip("the scene image is not found yet");
	const VkExtent2D render = color.extent, display = g_display;
	if (render.width > display.width || render.height > display.height) return skip("the scene image is larger than the display");
	if ((render.width == display.width || render.height == display.height) && !s.up_native)
		return skip(Fmt("the game renders at the display size (%ux%u): lower its own resolution scale (e.g. %d%% for Quality) "
		                "or tick 'tambien a resolucion nativa'", render.width, render.height, static_cast<int>(ScaleForQuality(2) * 100 + 0.5f)));
	if (v.depth.extent.width != render.width || v.depth.extent.height != render.height)
		return skip(Fmt("the depth (%ux%u) is not at the scene's size (%ux%u)", v.depth.extent.width, v.depth.extent.height, render.width, render.height));
	if (!(v.backbuffer.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return skip("the backbuffer cannot be written by a blit");

	// Camera (this frame's scan is complete: the scene passes are recorded).
	const float ratio = static_cast<float>(display.height) / static_cast<float>(render.height);
	camera::SetInjection(s.up_jitter == 2, ratio);
	camera::EndFrame(render);
	g_frame.camera = true;
	const camera::State cam = camera::Current();

	// The backend.
	if (g_backend == nullptr || g_backend_kind != s.up_backend) {
		if (g_failed_kind == s.up_backend && g_presents < g_failed_at + 240) return skip(g_backend_why);  // (retried every ~240 frames)
		if (g_backend != nullptr) {
			g_retired.push_back({std::move(g_backend), {}, g_presents});
			g_backend = nullptr;
		}
		std::string why;
		{
			InternalScope scope;
			g_backend = CreateBackend(s.up_backend, why);
		}
		g_backend_kind = s.up_backend;
		g_logged_on    = false;
		if (g_backend == nullptr) {
			const std::string text = std::string(KindName(s.up_backend)) + " not available: " + why;
			if (g_failed_kind != s.up_backend || text != g_backend_why) Log("[up] %s", text.c_str());
			g_failed_kind = s.up_backend;
			g_failed_at   = g_presents;
			g_backend_why = text;
			return skip(g_backend_why);
		}
		g_failed_kind = kOff;
		g_ran_last    = false;
	}

	InternalScope scope;  // (the backends call back into our exports)
	FullBarrier(cb);
	const uint32_t ts_slot = static_cast<uint32_t>(g_runs % 4);
	const bool ts = g_queries != VK_NULL_HANDLE && !g_ts_pending[ts_slot].valid;
	if (ts) {
		R().vkCmdResetQueryPool(cb, g_queries, ts_slot * 2, 2);
		R().vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_queries, ts_slot * 2);
	}
	std::string why;
	bool ok = CopyIn(cb, color, g_t.color, pre ? "pre-TAA colour" : "scene colour", why) && CopyIn(cb, v.depth, g_t.depth, "depth", why);
	const bool motion_ok = ok && v.motion.image != VK_NULL_HANDLE && v.motion.confidence >= s.min_confidence &&
	                       v.motion.extent.width == render.width && v.motion.extent.height == render.height &&
	                       CopyIn(cb, v.motion, g_t.motion, "motion vectors", why);
	OwnImage* motion = &g_t.motion;
	if (ok && !motion_ok) {  // zero motion (the upscaler then leans on its own history clamping)
		if (Ensure(g_t.zero_motion, VK_FORMAT_R16G16_SFLOAT, render, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
			if (g_t.zero_motion.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
				Transition(cb, g_t.zero_motion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
				VkClearColorValue zero {};
				VkImageSubresourceRange range {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
				R().vkCmdClearColorImage(cb, g_t.zero_motion.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
				Transition(cb, g_t.zero_motion, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			}
			motion = &g_t.zero_motion;
		} else {
			ok = false;
			why = "zero motion image not created";
		}
	}
	const VkImageUsageFlags out_usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
	                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	ok = ok && Ensure(g_t.output, VK_FORMAT_R16G16B16A16_SFLOAT, display, out_usage);
	const bool pipes = EnsurePipes();
	const bool metric = pipes && g_runs % 120 == 60 && !g_metric_pending.valid &&
	                    Ensure(g_t.crop_game, VK_FORMAT_R16G16B16A16_SFLOAT, {256, 256}, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT) &&
	                    Ensure(g_t.crop_ours, VK_FORMAT_R16G16B16A16_SFLOAT, {256, 256}, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT) &&
	                    display.width >= 256 && display.height >= 256;
	if (ok) {
		if (metric) {
			R().vkCmdFillBuffer(cb, g_metric.buffer, 0, 64, 0);
			Crop(cb, v.backbuffer.image, v.backbuffer.format, bb_layout, g_t.crop_game);  // the game's own result
		}
		Transition(cb, g_t.output, VK_IMAGE_LAYOUT_GENERAL);
		// Jitter: the camera's, by the menu's rule (see upscale.h).
		Frame f;
		f.cb      = cb;
		f.color   = &g_t.color;
		f.depth   = &g_t.depth;
		f.motion  = motion;
		f.output  = &g_t.output;
		f.render  = render;
		f.display = display;
		const char* jitter_kind = "none";
		if (s.up_jitter == 2 && cam.injecting) jitter_kind = "injected (Halton)";
		else if (s.up_jitter == 0 && cam.jitter_detected && (pre || !v.taa_found)) jitter_kind = "detected";
		else if (s.up_jitter == 0 && cam.jitter_detected) jitter_kind = "detected, not used (the scene image is after the game's TAA)";
		if (std::strcmp(jitter_kind, "detected") == 0 || cam.injecting) {
			f.jitter[0] = cam.jitter_px[0];
			f.jitter[1] = cam.jitter_px[1];
		}
		const float sign = static_cast<float>(s.mv_sign);
		f.mv_scale[0] = motion_ok ? (s.mv_units == 0 ? sign * render.width : sign) : 0.0f;
		f.mv_scale[1] = motion_ok ? (s.mv_units == 0 ? sign * render.height : sign) : 0.0f;
		f.reversed = v.reversed_depth;
		f.hdr      = IsFloatFormat(color.format);
		f.quality  = s.up_quality == 0 ? QualityForRatio(ratio) : s.up_quality;
		const auto now = Clock::now();
		f.dt_ms = g_last_time == Clock::time_point {} ? 16.7f : std::clamp(std::chrono::duration<float, std::milli>(now - g_last_time).count(), 1.0f, 100.0f);
		g_last_time = now;
		f.reset = !g_ran_last || g_last_input != color.image || g_last_render.width != render.width || g_last_render.height != render.height ||
		          g_last_quality != f.quality;
		f.cam   = cam;
		f.index = g_runs;
		const bool dispatched = g_backend->Dispatch(f);
		FullBarrier(cb);
		if (dispatched) {
			OwnImage* final_image = &g_t.output;
			const bool sharpen = pipes && s.up_sharpness > 0.001f && Ensure(g_t.sharp, VK_FORMAT_R16G16B16A16_SFLOAT, display, out_usage);
			if (sharpen) {
				Sharpen(cb, g_t.output, g_t.sharp, std::clamp(s.up_sharpness, 0.0f, 1.0f), f.hdr);
				final_image = &g_t.sharp;
			}
			if (!s.up_original) {
				Transition(cb, *final_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
				Transition(cb, v.backbuffer.image, v.backbuffer.format, bb_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
				VkImageBlit blit {};
				blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
				blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
				const int32_t w = static_cast<int32_t>(s.up_split ? display.width / 2 : display.width);
				blit.srcOffsets[1] = {w, static_cast<int32_t>(display.height), 1};
				blit.dstOffsets[1] = blit.srcOffsets[1];
				R().vkCmdBlitImage(cb, final_image->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, v.backbuffer.image,
				                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
				Transition(cb, v.backbuffer.image, v.backbuffer.format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, bb_layout);
				Transition(cb, *final_image, VK_IMAGE_LAYOUT_GENERAL);
			}
			++g_dispatched;
			if (metric) {
				g_metric_has_ours = !s.up_original && !s.up_split;
				if (g_metric_has_ours) Crop(cb, v.backbuffer.image, v.backbuffer.format, bb_layout, g_t.crop_ours);  // what is on screen now
				FullBarrier(cb);
				Metric(cb, g_t.crop_game, 0);
				if (g_metric_has_ours) Metric(cb, g_t.crop_ours, 1);
				VkMemoryBarrier m {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
				m.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				m.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
				R().vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);
				g_metric_pending = {g_runs, true};
			}
			// Status / log.
			g_status.running     = true;
			g_status.state       = s.up_original ? "on (A/B: original on screen)" : s.up_split ? "on (split)" : "on";
			g_status.why_not.clear();
			g_status.where       = where;
			g_status.hud_covered = hud_covered;
			g_status.render      = render;
			g_status.display     = display;
			g_status.ratio       = ratio;
			g_status.quality     = f.quality;
			g_status.jitter[0]   = f.jitter[0];
			g_status.jitter[1]   = f.jitter[1];
			g_status.jitter_kind = jitter_kind;
			g_status.detail      = g_backend->Detail();
			g_status.inputs = Fmt("colour %s fmt %d %ux%u%s | depth fmt %d %ux%u %s | motion %s", pre ? "pre-TAA" : "scene (post-TAA)",
			                      color.format, render.width, render.height, f.hdr ? " HDR" : "", v.depth.format, v.depth.extent.width,
			                      v.depth.extent.height, f.reversed ? "reversed" : "forward",
			                      motion_ok ? Fmt("fmt %d %ux%u, scale (%.0f, %.0f)", v.motion.format, v.motion.extent.width,
			                                      v.motion.extent.height, f.mv_scale[0], f.mv_scale[1]).c_str()
			                                : "none (zero vectors)");
			if (!g_logged_on) {
				g_logged_on = true;
				Log("[up] %s on: %ux%u -> %ux%u (x%.2f, %s%s) | %s | jitter %s (%.3f, %.3f px; camera %s) | RCAS %.2f at %ux%u | placement: %s%s",
				    g_backend->Name(), render.width, render.height, display.width, display.height, ratio, QualityName(f.quality),
				    g_status.detail.empty() ? "" : (", " + g_status.detail).c_str(), g_status.inputs.c_str(), jitter_kind, f.jitter[0], f.jitter[1],
				    cam.found ? Fmt("fov %.1f near %.3f", cam.fov * 57.2958f, cam.near_plane).c_str() : "not found",
				    sharpen ? s.up_sharpness : 0.0f, display.width, display.height, where,
				    hud_covered ? " (no UI pass seen: the HUD drawn in the composite is covered)" : "");
			}
		} else {
			why = "the upscaler did not record (see the lines above)";
		}
	}
	if (ts) {
		R().vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_queries, ts_slot * 2 + 1);
		g_ts_pending[ts_slot] = {g_runs, true};
	}
	FullBarrier(cb);
	Restore(cb);  // DXVK's compute pipeline, descriptor sets and push constants as they were
	++g_runs;
	if (!ok || !why.empty()) {
		if (g_status.why_not != why) Log("[up] %s: not this frame: %s", g_backend->Name(), why.c_str());
		g_status.why_not = why;
		g_status.running = false;
		g_ran_last = false;
		return;
	}
	g_ran_last     = true;
	g_last_input   = color.image;
	g_last_render  = render;
	g_last_quality = s.up_quality == 0 ? QualityForRatio(ratio) : s.up_quality;
}

void DestroyAll() {
	g_backend = nullptr;
	g_retired.clear();
	for (OwnImage* i: {&g_t.color, &g_t.depth, &g_t.motion, &g_t.zero_motion, &g_t.output, &g_t.sharp, &g_t.crop_game, &g_t.crop_ours})
		DestroyOwnImage(*i);
	DestroyOwnBuffer(g_metric);
	DestroyComputePipe(g_rcas);
	DestroyComputePipe(g_metric_pipe);
	if (g_point != VK_NULL_HANDLE && R().device != VK_NULL_HANDLE) R().vkDestroySampler(R().device, g_point, nullptr);
	if (g_queries != VK_NULL_HANDLE && R().device != VK_NULL_HANDLE) R().vkDestroyQueryPool(R().device, g_queries, nullptr);
	g_point = VK_NULL_HANDLE;
	g_queries = VK_NULL_HANDLE;
	g_pipes_tried = g_pipes_ok = false;
	g_backend_kind = g_failed_kind = kOff;
	g_frame = {};
	g_ran_last = false;
	for (auto& p: g_ts_pending) p = {};
	g_metric_pending = {};
	g_status = {};
	g_logged_on = false;
}

} // namespace

void SetDisplay(VkExtent2D display, VkFormat format) {
	std::lock_guard lock(g_mutex);
	g_display = display;
	g_display_format = format;
}

void OnFirstPresent() {
	InternalScope scope;
	dlss::OnFirstPresent(R().device);
}

void OnDeviceDestroyed() {
	std::lock_guard lock(g_mutex);
	DestroyAll();
	{
		std::lock_guard cbl(g_cb_mutex);
		g_cbs.clear();
	}
	g_device_extensions.clear();
	g_push_descriptors = false;
}

void BeforeRendering(VkCommandBuffer cb, const VkRenderingInfo* info, const analyzer::PassInfo& pass) {
	if (Internal() || info == nullptr || (info->flags & VK_RENDERING_RESUMING_BIT)) return;
	std::lock_guard lock(g_mutex);
	if (pass.swapchain) {  // DXVK's present blit: the frame ends here
		if (Cfg().up_backend != kOff && !g_frame.done && g_frame.bb_passes >= 1) {
			const auto v = analyzer::View(cb);
			Run(cb, analyzer::LayoutIn(cb, v.backbuffer.image), "before DXVK's present blit", true);
		}
		if (!g_frame.camera) camera::EndFrame(g_last_render);
		g_frame = {};
		return;
	}
	if (!pass.backbuffer) return;
	if (++g_frame.bb_passes == 2 && !g_frame.done && Cfg().up_backend != kOff)
		Run(cb, pass.color0_layout, "before the first UI pass (backbuffer pass 2)", false);
}

void OnPresent() {
	std::vector<Retired> dead;
	{
		std::lock_guard lock(g_mutex);
		++g_presents;
		if (Cfg().up_backend == kOff && g_backend != nullptr) {  // turned off from the menu: let it go
			g_retired.push_back({std::move(g_backend), {}, g_presents});
			g_backend = nullptr;
			g_backend_kind = kOff;
			g_status = {};
			g_logged_on = false;
		}
		if (g_failed_kind != kOff && g_failed_kind != Cfg().up_backend) g_failed_kind = kOff;
		for (auto it = g_retired.begin(); it != g_retired.end();) {
			if (g_presents >= it->at + 8) { dead.push_back(std::move(*it)); it = g_retired.erase(it); }
			else ++it;
		}
		// Timestamps.
		for (uint32_t slot = 0; slot < 4; ++slot) {
			auto& p = g_ts_pending[slot];
			if (!p.valid || g_runs < p.index + 4) continue;
			uint64_t t[2] {};
			if (R().vkGetQueryPoolResults(R().device, g_queries, slot * 2, 2, sizeof(t), t, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
				g_status.gpu_ms = static_cast<float>(static_cast<double>(t[1] - t[0]) * g_ts_period / 1e6);
			p.valid = false;
		}
		// The sharpness metric.
		if (g_metric_pending.valid && g_runs >= g_metric_pending.index + 6 && g_metric.mapped != nullptr) {
			const auto* sums = static_cast<const uint32_t*>(g_metric.mapped);
			g_status.metric_game  = static_cast<float>(sums[0]) / 1024.0f / 65536.0f;
			g_status.metric_ours  = g_metric_has_ours ? static_cast<float>(sums[1]) / 1024.0f / 65536.0f : 0.0f;
			g_status.metric_frame = g_metric_pending.index;
			g_metric_pending.valid = false;
			if (g_status.running)
				Log("[up] %s frame %llu: %ux%u -> %ux%u (%s) | gpu %.2f ms (copies + upscale + RCAS + blit) | jitter %s (%.3f, %.3f px) | "
				    "sharpness metric (mean |laplacian| of luma, centre 256x256): game %.5f -> ours %s",
				    KindName(g_backend_kind), static_cast<unsigned long long>(g_status.metric_frame), g_status.render.width,
				    g_status.render.height, g_status.display.width, g_status.display.height, QualityName(g_status.quality),
				    g_status.gpu_ms, g_status.jitter_kind, g_status.jitter[0], g_status.jitter[1], g_status.metric_game,
				    g_metric_has_ours ? Fmt("%.5f (%+.0f%%)", g_status.metric_ours,
				                            g_status.metric_game > 0 ? 100.0f * (g_status.metric_ours / g_status.metric_game - 1.0f) : 0.0f).c_str()
				                      : "(not measured: A/B original or split on)");
		}
		if (g_status.running)
			std::snprintf(g_osd, sizeof g_osd, " | %s %ux%u->%ux%u %.2f ms%s", KindName(g_backend_kind), g_status.render.width,
			              g_status.render.height, g_status.display.width, g_status.display.height, g_status.gpu_ms,
			              Cfg().up_original ? " (A/B: original)" : "");
		else if (Cfg().up_backend != kOff)
			std::snprintf(g_osd, sizeof g_osd, " | %s esperando", KindName(Cfg().up_backend));
		else
			g_osd[0] = '\0';
	}
	if (!dead.empty()) {
		std::lock_guard qlock(QueueLock());
		R().vkDeviceWaitIdle(R().device);
		InternalScope scope;
		for (auto& d: dead) {
			d.backend = nullptr;
			for (auto& i: d.images) DestroyOwnImage(i);
		}
	}
}

const char* OsdText() {
	return g_osd;
}

void DrawMenu() {
	auto& s = Cfg();
	std::string why[kKinds];
	bool ready[kKinds] {};
	ready[kDlss] = dlss::Ready(why[kDlss]);
	ready[kFsr4] = fsr4::Ready(why[kFsr4]);
	ready[kFsr3] = fsr3::Ready(why[kFsr3]);
	ready[kXess] = xess::Ready(why[kXess]);
	char labels[kKinds][96];
	std::snprintf(labels[0], sizeof labels[0], "Off");
	for (int k = 1; k < kKinds; ++k)
		std::snprintf(labels[k], sizeof labels[k], "%s%s", k == kDlss ? "DLSS (Streamline)" : KindName(k), ready[k] ? "" : " (no disponible)");
	if (ImGui::BeginCombo("Reescalador", labels[s.up_backend])) {
		for (int k = 0; k < kKinds; ++k)
			if (ImGui::Selectable(labels[k], s.up_backend == k)) s.up_backend = k;
		ImGui::EndCombo();
	}
	if (s.up_backend != kOff && !ready[s.up_backend]) ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "%s", why[s.up_backend].c_str());
	ImGui::Combo("Modo", &s.up_quality, "Auto (segun la escala del juego)\0Native AA (100%)\0Quality (67%)\0Balanced (58%)\0Performance (50%)\0Ultra Performance (33%)\0");
	if (s.up_quality != 0)
		ImGui::TextDisabled("Pon la escala de resolucion del juego en ~%d%%", static_cast<int>(ScaleForQuality(s.up_quality) * 100 + 0.5f));
	ImGui::SliderFloat("Nitidez (RCAS, a resolucion de pantalla)", &s.up_sharpness, 0.0f, 1.0f, "%.2f");
	char ab[80];
	std::snprintf(ab, sizeof ab, "A/B: ver la imagen original del juego (%s)", KeyName(s.ab_key));
	ImGui::Checkbox(ab, &s.up_original);
	ImGui::Checkbox("Pantalla dividida (izq. reescalado, der. original)", &s.up_split);
	ImGui::Combo("Jitter", &s.up_jitter, "Auto (el detectado en la proyeccion)\0Ninguno\0Inyectar Halton (con el TAA del juego apagado)\0");
	ImGui::Combo("Entrada de color", &s.up_input, "Imagen de escena (tras el TAA del juego)\0Pre-TAA (experimental)\0");
	ImGui::Checkbox("Tambien a resolucion nativa (solo AA)", &s.up_native);
	camera::DrawMenu();
	Status st;
	{
		std::lock_guard lock(g_mutex);
		st = g_status;
	}
	if (s.up_backend == kOff) return;
	if (st.running) {
		ImGui::TextColored(ImVec4(0.4f, 1, 0.4f, 1), "%s %s: %ux%u -> %ux%u (x%.2f, %s)", KindName(s.up_backend), st.state.c_str(),
		                   st.render.width, st.render.height, st.display.width, st.display.height, st.ratio, QualityName(st.quality));
		if (!st.detail.empty()) ImGui::TextDisabled("%s", st.detail.c_str());
		ImGui::TextWrapped("Entradas: %s", st.inputs.c_str());
		ImGui::Text("Jitter: %s (%.3f, %.3f px)", st.jitter_kind, st.jitter[0], st.jitter[1]);
		ImGui::Text("GPU: %.2f ms  |  nitidez (metrica): juego %.5f -> nuestro %.5f", st.gpu_ms, st.metric_game, st.metric_ours);
		ImGui::TextWrapped("Punto: %s", st.where.c_str());
		if (st.hud_covered) ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1), "Sin pase de UI separado: el HUD del juego queda tapado");
		if (s.up_jitter == 2) ImGui::TextDisabled("Inyectar Halton: apaga el TAA / AA del juego, o el jitter no llega al reescalador");
	} else {
		ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1), "Esperando: %s", st.why_not.empty() ? "primer frame" : st.why_not.c_str());
	}
}

} // namespace inj::upscale
