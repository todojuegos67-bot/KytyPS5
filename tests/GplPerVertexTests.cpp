// Pixel shaders with per-vertex inputs (PerVertexKHR, VK_KHR_fragment_shader_barycentric) linked from graphics pipeline
// library parts must draw exactly what the whole pipeline draws. The recompiler emits such inputs for custom
// interpolation and for smooth FP16_INTERP_MODE inputs, whose packed halves are unpacked per vertex, then blended.
// Each case is drawn by a whole pipeline and by four library builds (four parts fast-linked, the same LTO-linked,
// pre-rasterization and fragment shader in one part, fragment shader and output in one part); the targets must match
// bit for bit. A device without libraries with fast linking or barycentrics skips.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

#include "GplPerVertexShaders.inc"

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "GplPerVertexTests:%d: %s\n", __LINE__, #condition);              \
			std::abort();                                                                          \
		}                                                                                          \
	} while (false)
#define CHECK_VK(call) CHECK((call) == VK_SUCCESS)

constexpr uint32_t Width   = 64;
constexpr uint32_t Height  = 64;
constexpr uint32_t Targets = 4;
constexpr uint32_t Verts   = 8 * 8 * 2 * 3; // 8x8 cells, two triangles each
constexpr float    Clear   = -12345.0f;

#define VK_FUNCTIONS(X)                                                                            \
	X(vkCreateInstance) X(vkDestroyInstance) X(vkEnumeratePhysicalDevices)                         \
	X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceProperties2)                             \
	X(vkGetPhysicalDeviceFeatures2) X(vkGetPhysicalDeviceMemoryProperties)                         \
	X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkEnumerateDeviceExtensionProperties)            \
	X(vkCreateDevice) X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateShaderModule)               \
	X(vkDestroyShaderModule) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout)                  \
	X(vkCreateGraphicsPipelines) X(vkDestroyPipeline) X(vkCreateImage) X(vkDestroyImage)           \
	X(vkGetImageMemoryRequirements) X(vkBindImageMemory) X(vkCreateImageView)                      \
	X(vkDestroyImageView) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements)    \
	X(vkBindBufferMemory) X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory)                       \
	X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers)                     \
	X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkCmdPipelineBarrier2) X(vkCmdBeginRendering)  \
	X(vkCmdEndRendering) X(vkCmdBindPipeline) X(vkCmdDraw) X(vkCmdCopyImageToBuffer)               \
	X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkDeviceWaitIdle)

struct Vk {
#define VK_MEMBER(name) PFN_##name name = nullptr;
	VK_FUNCTIONS(VK_MEMBER)
#undef VK_MEMBER
};
Vk vk;

PFN_vkGetInstanceProcAddr LoadLoader() {
#if defined(_WIN32)
	const auto library = LoadLibraryA("vulkan-1.dll");
	return library == nullptr ? nullptr
	                          : reinterpret_cast<PFN_vkGetInstanceProcAddr>(
	                                reinterpret_cast<void*>(GetProcAddress(library, "vkGetInstanceProcAddr")));
#else
	void* library = dlopen("libvulkan.so.1", RTLD_NOW);
	if (library == nullptr) library = dlopen("libvulkan.1.dylib", RTLD_NOW);
	return library == nullptr ? nullptr
	                          : reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
#endif
}

enum class Build { Whole, Fast, Lto, PreRasterAndFragment, FragmentAndOutput };
constexpr std::array<Build, 4> LibraryBuilds {Build::Fast, Build::Lto, Build::PreRasterAndFragment,
                                              Build::FragmentAndOutput};

const char* BuildName(Build build) {
	switch (build) {
		case Build::Whole: return "whole";
		case Build::Fast: return "four parts, fast link";
		case Build::Lto: return "four parts, LTO link";
		case Build::PreRasterAndFragment: return "pre-rasterization + fragment part";
		case Build::FragmentAndOutput: return "fragment + output part";
	}
	return "?";
}

class Harness {
public:
	// False when the device lacks what the test needs; the test then skips.
	bool Init() {
		const auto get_proc = LoadLoader();
		if (get_proc == nullptr) return Skip("no Vulkan loader");
		vk.vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(get_proc(nullptr, "vkCreateInstance"));
		VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
		app.pApplicationName = "GplPerVertexTests";
		app.apiVersion       = VK_API_VERSION_1_3;
		VkInstanceCreateInfo instance_info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
		instance_info.pApplicationInfo = &app;
		if (vk.vkCreateInstance == nullptr || vk.vkCreateInstance(&instance_info, nullptr, &m_instance) != VK_SUCCESS) {
			return Skip("no Vulkan 1.3 instance");
		}
#define VK_LOAD(name) vk.name = reinterpret_cast<PFN_##name>(get_proc(m_instance, #name));
		VK_FUNCTIONS(VK_LOAD)
#undef VK_LOAD

		uint32_t count = 0;
		vk.vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
		std::vector<VkPhysicalDevice> devices(count);
		vk.vkEnumeratePhysicalDevices(m_instance, &count, devices.data());
		for (const auto device: devices) {
			if (Suitable(device)) {
				m_physical = device;
				break;
			}
		}
		if (m_physical == VK_NULL_HANDLE) return Skip("no device with fast-linked libraries and barycentrics");
		VkPhysicalDeviceProperties properties {};
		vk.vkGetPhysicalDeviceProperties(m_physical, &properties);
		std::printf("device %s, driver 0x%08x\n", properties.deviceName, properties.driverVersion);
		vk.vkGetPhysicalDeviceMemoryProperties(m_physical, &m_memory);

		VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentric {
		    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
		barycentric.fragmentShaderBarycentric = VK_TRUE;
		VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT library {
		    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT};
		library.graphicsPipelineLibrary = VK_TRUE;
		library.pNext                   = &barycentric;
		VkPhysicalDeviceVulkan13Features features13 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
		features13.dynamicRendering = VK_TRUE;
		features13.synchronization2 = VK_TRUE;
		features13.pNext            = &library;
		VkPhysicalDeviceFeatures2 features {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
		features.pNext = &features13;
		const std::array<const char*, 3> extensions {VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
		                                             VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME,
		                                             VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME};
		const float             priority = 1.0f;
		VkDeviceQueueCreateInfo queue {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
		queue.queueFamilyIndex = m_queue_family;
		queue.queueCount       = 1;
		queue.pQueuePriorities = &priority;
		VkDeviceCreateInfo device_info {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
		device_info.pNext                   = &features;
		device_info.queueCreateInfoCount    = 1;
		device_info.pQueueCreateInfos       = &queue;
		device_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
		device_info.ppEnabledExtensionNames = extensions.data();
		CHECK_VK(vk.vkCreateDevice(m_physical, &device_info, nullptr, &m_device));
		vk.vkGetDeviceQueue(m_device, m_queue_family, 0, &m_queue);
		CreateTargets();
		return true;
	}

	~Harness() {
		if (m_device != VK_NULL_HANDLE) {
			vk.vkDeviceWaitIdle(m_device);
			for (const auto pipeline: m_pipelines) vk.vkDestroyPipeline(m_device, pipeline, nullptr);
			for (const auto layout: m_layouts) vk.vkDestroyPipelineLayout(m_device, layout, nullptr);
			for (const auto module: m_modules) vk.vkDestroyShaderModule(m_device, module, nullptr);
			vk.vkDestroyCommandPool(m_device, m_pool, nullptr);
			vk.vkDestroyBuffer(m_device, m_readback, nullptr);
			for (uint32_t t = 0; t < Targets; t++) {
				vk.vkDestroyImageView(m_device, m_views[t], nullptr);
				vk.vkDestroyImage(m_device, m_images[t], nullptr);
			}
			for (const auto memory: m_allocations) vk.vkFreeMemory(m_device, memory, nullptr);
			vk.vkDestroyDevice(m_device, nullptr);
		}
		if (m_instance != VK_NULL_HANDLE) vk.vkDestroyInstance(m_instance, nullptr);
	}

	// Draws the case with every build and compares each library build with the whole pipeline.
	void Check(const char* name, const uint32_t* vert, size_t vert_size, const uint32_t* frag, size_t frag_size) {
		const auto vs = Module(vert, vert_size);
		const auto fs = Module(frag, frag_size);
		const auto whole = Draw(Pipeline(vs, fs, Build::Whole));
		uint32_t   covered = 0;
		for (uint32_t pixel = 0; pixel < Width * Height; pixel++) covered += whole[pixel * 4] != Clear ? 1u : 0u;
		CHECK(covered > Width * Height / 4); // the triangles were drawn
		for (const auto build: LibraryBuilds) {
			const auto parts = Draw(Pipeline(vs, fs, build));
			uint32_t   differ = 0;
			uint32_t   first  = 0;
			for (uint32_t i = 0; i < parts.size(); i++) {
				if (std::memcmp(&parts[i], &whole[i], sizeof(float)) != 0 && differ++ == 0) first = i;
			}
			if (differ != 0) {
				const auto pixel = (first % (Width * Height * 4)) / 4;
				std::fprintf(stderr, "%s, %s: %u of %zu floats differ; first target %u pixel (%u,%u): %g against %g\n",
				             name, BuildName(build), differ, parts.size(), first / (Width * Height * 4), pixel % Width,
				             pixel / Width, parts[first], whole[first]);
			}
			CHECK(differ == 0);
		}
		std::printf("[gpl]  %-40s ok\n", name);
	}

private:
	static bool Skip(const char* why) {
		std::printf("GplPerVertexTests skipped: %s\n", why);
		return false;
	}

	bool Suitable(VkPhysicalDevice device) {
		uint32_t count = 0;
		vk.vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
		std::vector<VkExtensionProperties> extensions(count);
		vk.vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());
		const auto has = [&](const char* name) {
			for (const auto& extension: extensions) {
				if (std::strcmp(extension.extensionName, name) == 0) return true;
			}
			return false;
		};
		if (!has(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) || !has(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME) ||
		    !has(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME)) {
			return false;
		}
		VkPhysicalDeviceProperties properties {};
		vk.vkGetPhysicalDeviceProperties(device, &properties);
		if (properties.apiVersion < VK_API_VERSION_1_3) return false;
		VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentric {
		    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
		VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT library {
		    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT};
		library.pNext = &barycentric;
		VkPhysicalDeviceVulkan13Features features13 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
		features13.pNext = &library;
		VkPhysicalDeviceFeatures2 features {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
		features.pNext = &features13;
		vk.vkGetPhysicalDeviceFeatures2(device, &features);
		VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT library_properties {
		    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT};
		VkPhysicalDeviceProperties2 properties2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
		properties2.pNext = &library_properties;
		vk.vkGetPhysicalDeviceProperties2(device, &properties2);
		if (features13.dynamicRendering == VK_FALSE || features13.synchronization2 == VK_FALSE ||
		    library.graphicsPipelineLibrary == VK_FALSE || barycentric.fragmentShaderBarycentric == VK_FALSE ||
		    library_properties.graphicsPipelineLibraryFastLinking == VK_FALSE ||
		    library_properties.graphicsPipelineLibraryIndependentInterpolationDecoration == VK_FALSE) {
			return false;
		}
		vk.vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
		std::vector<VkQueueFamilyProperties> families(count);
		vk.vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());
		for (uint32_t family = 0; family < count; family++) {
			if ((families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
				m_queue_family = family;
				return true;
			}
		}
		return false;
	}

	VkDeviceMemory Allocate(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags flags) {
		for (uint32_t type = 0; type < m_memory.memoryTypeCount; type++) {
			if ((requirements.memoryTypeBits & (1u << type)) != 0 &&
			    (m_memory.memoryTypes[type].propertyFlags & flags) == flags) {
				VkMemoryAllocateInfo info {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
				info.allocationSize  = requirements.size;
				info.memoryTypeIndex = type;
				VkDeviceMemory memory = VK_NULL_HANDLE;
				CHECK_VK(vk.vkAllocateMemory(m_device, &info, nullptr, &memory));
				m_allocations.push_back(memory);
				return memory;
			}
		}
		CHECK(false);
		return VK_NULL_HANDLE;
	}

	void CreateTargets() {
		for (uint32_t t = 0; t < Targets; t++) {
			VkImageCreateInfo image {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
			image.imageType   = VK_IMAGE_TYPE_2D;
			image.format      = VK_FORMAT_R32G32B32A32_SFLOAT;
			image.extent      = {Width, Height, 1};
			image.mipLevels   = 1;
			image.arrayLayers = 1;
			image.samples     = VK_SAMPLE_COUNT_1_BIT;
			image.tiling      = VK_IMAGE_TILING_OPTIMAL;
			image.usage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
			CHECK_VK(vk.vkCreateImage(m_device, &image, nullptr, &m_images[t]));
			VkMemoryRequirements requirements {};
			vk.vkGetImageMemoryRequirements(m_device, m_images[t], &requirements);
			CHECK_VK(vk.vkBindImageMemory(m_device, m_images[t],
			                              Allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), 0));
			VkImageViewCreateInfo view {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
			view.image            = m_images[t];
			view.viewType         = VK_IMAGE_VIEW_TYPE_2D;
			view.format           = image.format;
			view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
			CHECK_VK(vk.vkCreateImageView(m_device, &view, nullptr, &m_views[t]));
		}
		VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		buffer.size  = TargetBytes() * Targets;
		buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		CHECK_VK(vk.vkCreateBuffer(m_device, &buffer, nullptr, &m_readback));
		VkMemoryRequirements requirements {};
		vk.vkGetBufferMemoryRequirements(m_device, m_readback, &requirements);
		const auto memory =
		    Allocate(requirements, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		CHECK_VK(vk.vkBindBufferMemory(m_device, m_readback, memory, 0));
		CHECK_VK(vk.vkMapMemory(m_device, memory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&m_mapped)));
		VkCommandPoolCreateInfo pool {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
		pool.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		pool.queueFamilyIndex = m_queue_family;
		CHECK_VK(vk.vkCreateCommandPool(m_device, &pool, nullptr, &m_pool));
		VkCommandBufferAllocateInfo command {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		command.commandPool        = m_pool;
		command.commandBufferCount = 1;
		CHECK_VK(vk.vkAllocateCommandBuffers(m_device, &command, &m_command));
		for (const auto flags: {VkPipelineLayoutCreateFlags {0}, VkPipelineLayoutCreateFlags {
		                                                             VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT}}) {
			VkPipelineLayoutCreateInfo layout {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
			layout.flags = flags;
			m_layouts.push_back(VK_NULL_HANDLE);
			CHECK_VK(vk.vkCreatePipelineLayout(m_device, &layout, nullptr, &m_layouts.back()));
		}
	}

	static VkDeviceSize TargetBytes() { return VkDeviceSize {Width} * Height * 4 * sizeof(float); }

	VkShaderModule Module(const uint32_t* code, size_t size) {
		VkShaderModuleCreateInfo info {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
		info.codeSize = size;
		info.pCode    = code;
		m_modules.push_back(VK_NULL_HANDLE);
		CHECK_VK(vk.vkCreateShaderModule(m_device, &info, nullptr, &m_modules.back()));
		return m_modules.back();
	}

	VkPipeline Pipeline(VkShaderModule vs, VkShaderModule fs, Build build) {
		std::array<VkFormat, Targets> formats {};
		formats.fill(VK_FORMAT_R32G32B32A32_SFLOAT);
		VkPipelineRenderingCreateInfo rendering {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
		rendering.colorAttachmentCount    = Targets;
		rendering.pColorAttachmentFormats = formats.data();
		VkPipelineShaderStageCreateInfo vertex_stage {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
		vertex_stage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
		vertex_stage.module = vs;
		vertex_stage.pName  = "main";
		auto fragment_stage   = vertex_stage;
		fragment_stage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragment_stage.module = fs;
		VkPipelineVertexInputStateCreateInfo   vertex_input {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
		VkPipelineInputAssemblyStateCreateInfo assembly {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
		assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		const VkViewport                  viewport {0.0f, 0.0f, float(Width), float(Height), 0.0f, 1.0f};
		const VkRect2D                    scissor {{0, 0}, {Width, Height}};
		VkPipelineViewportStateCreateInfo viewport_state {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
		viewport_state.viewportCount = 1;
		viewport_state.pViewports    = &viewport;
		viewport_state.scissorCount  = 1;
		viewport_state.pScissors     = &scissor;
		VkPipelineRasterizationStateCreateInfo raster {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
		raster.polygonMode = VK_POLYGON_MODE_FILL;
		raster.cullMode    = VK_CULL_MODE_NONE;
		raster.lineWidth   = 1.0f;
		VkPipelineMultisampleStateCreateInfo multisample {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
		multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		VkPipelineDepthStencilStateCreateInfo depth_stencil {
		    VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
		std::array<VkPipelineColorBlendAttachmentState, Targets> attachments {};
		for (auto& attachment: attachments) attachment.colorWriteMask = 0xfu;
		VkPipelineColorBlendStateCreateInfo blend {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
		blend.attachmentCount = Targets;
		blend.pAttachments    = attachments.data();

		VkPipeline pipeline = VK_NULL_HANDLE;
		if (build == Build::Whole) {
			const std::array             stages {vertex_stage, fragment_stage};
			VkGraphicsPipelineCreateInfo info {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
			info.pNext               = &rendering;
			info.stageCount          = 2;
			info.pStages             = stages.data();
			info.pVertexInputState   = &vertex_input;
			info.pInputAssemblyState = &assembly;
			info.pViewportState      = &viewport_state;
			info.pRasterizationState = &raster;
			info.pMultisampleState   = &multisample;
			info.pDepthStencilState  = &depth_stencil;
			info.pColorBlendState    = &blend;
			info.layout              = m_layouts[0];
			CHECK_VK(vk.vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
			m_pipelines.push_back(pipeline);
			return pipeline;
		}
		const bool lto  = build == Build::Lto;
		const auto part = [&](VkGraphicsPipelineLibraryFlagsEXT kinds) {
			VkGraphicsPipelineLibraryCreateInfoEXT library {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT};
			library.flags            = kinds;
			auto part_rendering      = rendering;
			part_rendering.pNext     = &library;
			VkGraphicsPipelineCreateInfo info {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
			info.pNext = &part_rendering;
			info.flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR |
			             (lto ? VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT : 0u);
			std::array<VkPipelineShaderStageCreateInfo, 2> stages {};
			uint32_t                                       stage_count = 0;
			if ((kinds & VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT) != 0) {
				info.pVertexInputState   = &vertex_input;
				info.pInputAssemblyState = &assembly;
			}
			if ((kinds & VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT) != 0) {
				stages[stage_count++]    = vertex_stage;
				info.pViewportState      = &viewport_state;
				info.pRasterizationState = &raster;
				info.layout              = m_layouts[1];
			}
			if ((kinds & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT) != 0) {
				stages[stage_count++]   = fragment_stage;
				info.pMultisampleState  = &multisample;
				info.pDepthStencilState = &depth_stencil;
				info.layout             = m_layouts[1];
			}
			if ((kinds & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT) != 0) {
				info.pMultisampleState = &multisample;
				info.pColorBlendState  = &blend;
			}
			info.stageCount = stage_count;
			info.pStages    = stage_count != 0 ? stages.data() : nullptr;
			VkPipeline result = VK_NULL_HANDLE;
			CHECK_VK(vk.vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &info, nullptr, &result));
			m_pipelines.push_back(result); // a part must outlive the pipelines linked from it
			return result;
		};
		constexpr auto vertex_input_part = VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT;
		constexpr auto raster_part       = VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT;
		constexpr auto fragment_part     = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT;
		constexpr auto output_part       = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT;
		std::vector<VkPipeline> parts;
		if (build == Build::PreRasterAndFragment) {
			parts = {part(vertex_input_part), part(raster_part | fragment_part), part(output_part)};
		} else if (build == Build::FragmentAndOutput) {
			parts = {part(vertex_input_part), part(raster_part), part(fragment_part | output_part)};
		} else {
			parts = {part(vertex_input_part), part(raster_part), part(fragment_part), part(output_part)};
		}
		VkPipelineLibraryCreateInfoKHR linked {VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR};
		linked.libraryCount = static_cast<uint32_t>(parts.size());
		linked.pLibraries   = parts.data();
		VkGraphicsPipelineCreateInfo link {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
		link.pNext  = &linked;
		link.flags  = lto ? VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT : 0u;
		link.layout = m_layouts[1];
		CHECK_VK(vk.vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &link, nullptr, &pipeline));
		m_pipelines.push_back(pipeline);
		return pipeline;
	}

	std::vector<float> Draw(VkPipeline pipeline) {
		VkCommandBufferBeginInfo begin {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		CHECK_VK(vk.vkBeginCommandBuffer(m_command, &begin));
		std::array<VkImageMemoryBarrier2, Targets> barriers {};
		for (uint32_t t = 0; t < Targets; t++) {
			auto& barrier            = barriers[t];
			barrier.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
			barrier.srcStageMask     = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
			barrier.srcAccessMask    = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
			barrier.dstStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
			barrier.dstAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
			barrier.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
			barrier.newLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			barrier.image            = m_images[t];
			barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		}
		VkDependencyInfo dependency {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
		dependency.imageMemoryBarrierCount = Targets;
		dependency.pImageMemoryBarriers    = barriers.data();
		vk.vkCmdPipelineBarrier2(m_command, &dependency);
		std::array<VkRenderingAttachmentInfo, Targets> attachments {};
		for (uint32_t t = 0; t < Targets; t++) {
			auto& attachment       = attachments[t];
			attachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
			attachment.imageView   = m_views[t];
			attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			attachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
			attachment.clearValue.color = {{Clear, Clear, Clear, Clear}};
		}
		VkRenderingInfo rendering {VK_STRUCTURE_TYPE_RENDERING_INFO};
		rendering.renderArea           = {{0, 0}, {Width, Height}};
		rendering.layerCount           = 1;
		rendering.colorAttachmentCount = Targets;
		rendering.pColorAttachments    = attachments.data();
		vk.vkCmdBeginRendering(m_command, &rendering);
		vk.vkCmdBindPipeline(m_command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		vk.vkCmdDraw(m_command, Verts, 1, 0, 0);
		vk.vkCmdEndRendering(m_command);
		for (auto& barrier: barriers) {
			barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
			barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
			barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
			barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
			barrier.oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			barrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		}
		vk.vkCmdPipelineBarrier2(m_command, &dependency);
		for (uint32_t t = 0; t < Targets; t++) {
			VkBufferImageCopy copy {};
			copy.bufferOffset     = t * TargetBytes();
			copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
			copy.imageExtent      = {Width, Height, 1};
			vk.vkCmdCopyImageToBuffer(m_command, m_images[t], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_readback, 1,
			                          &copy);
		}
		CHECK_VK(vk.vkEndCommandBuffer(m_command));
		VkSubmitInfo submit {VK_STRUCTURE_TYPE_SUBMIT_INFO};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers    = &m_command;
		CHECK_VK(vk.vkQueueSubmit(m_queue, 1, &submit, VK_NULL_HANDLE));
		CHECK_VK(vk.vkQueueWaitIdle(m_queue));
		return {m_mapped, m_mapped + TargetBytes() * Targets / sizeof(float)};
	}

	VkInstance                       m_instance     = VK_NULL_HANDLE;
	VkPhysicalDevice                 m_physical     = VK_NULL_HANDLE;
	VkPhysicalDeviceMemoryProperties m_memory {};
	uint32_t                         m_queue_family = 0;
	VkDevice                         m_device       = VK_NULL_HANDLE;
	VkQueue                          m_queue        = VK_NULL_HANDLE;
	std::array<VkImage, Targets>     m_images {};
	std::array<VkImageView, Targets> m_views {};
	VkBuffer                         m_readback = VK_NULL_HANDLE;
	float*                           m_mapped   = nullptr;
	VkCommandPool                    m_pool     = VK_NULL_HANDLE;
	VkCommandBuffer                  m_command  = VK_NULL_HANDLE;
	std::vector<VkDeviceMemory>      m_allocations;
	std::vector<VkPipelineLayout>    m_layouts; // [0] whole pipelines, [1] parts and their links (independent sets)
	std::vector<VkShaderModule>      m_modules;
	std::vector<VkPipeline>          m_pipelines;
};

} // namespace

int main() {
	Harness harness;
	if (!harness.Init()) return 0;
	// Packed f16 pairs blended per vertex with the barycentrics, next to a flat and a smooth input, an unread output
	// and a clip distance.
	harness.Check("packed f16 inputs blended per vertex", kPackedVert, sizeof(kPackedVert), kPackedFrag,
	              sizeof(kPackedFrag));
	// Raw bits through per-vertex inputs: f32 denormals (an f16 pair whose high half is zero), NaN payloads, infinities.
	harness.Check("raw bits through per-vertex inputs", kBitsVert, sizeof(kBitsVert), kBitsFrag, sizeof(kBitsFrag));
	std::puts("GPL per-vertex input tests passed");
}
