#include "common/assert.h"
#include "common/upstreamPaths.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/fsr3FrameGen.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/upscaler/dlss.h"
#include "graphics/host_gpu/renderer/upscaler/fsr.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/openxrHeadset.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/presentation/window/windowInternal.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "kernel/memory.h"
#include "libs/controller.h"
#include "loader/systemContent.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vk_platform.h>

// IWYU pragma: no_include <intrin.h>

namespace Libs::Graphics {

struct VulkanExtensions {
	bool enable_validation_layers = false;

	std::vector<const char*>             required_extensions;
	std::vector<vk::ExtensionProperties> available_extensions;
	std::vector<const char*>             required_layers;
	std::vector<vk::LayerProperties>     available_layers;
};

vk::PhysicalDeviceVulkan11Features WindowContext::RequiredVulkan11Features() noexcept {
	vk::PhysicalDeviceVulkan11Features features {};
	features.storageBuffer16BitAccess = VK_TRUE;
	features.shaderDrawParameters     = VK_TRUE;
	return features;
}

vk::PhysicalDeviceVulkan12Features WindowContext::RequiredVulkan12Features() noexcept {
	vk::PhysicalDeviceVulkan12Features features {};
	features.samplerMirrorClampToEdge  = VK_TRUE;
	features.timelineSemaphore         = VK_TRUE;
	features.shaderOutputLayer         = VK_TRUE;
	features.shaderOutputViewportIndex = VK_TRUE;
	features.bufferDeviceAddress       = VK_TRUE;
	features.shaderBufferInt64Atomics  = VK_TRUE;
	features.storageBuffer8BitAccess   = VK_TRUE;
	features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
	features.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
	features.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
	features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
	features.descriptorBindingUpdateUnusedWhilePending    = VK_TRUE;
	features.descriptorBindingPartiallyBound              = VK_TRUE;
	return features;
}

vk::PhysicalDeviceVulkan13Features WindowContext::RequiredVulkan13Features() noexcept {
	vk::PhysicalDeviceVulkan13Features features {};
	features.dynamicRendering = VK_TRUE;
	features.synchronization2 = VK_TRUE;
	return features;
}

static bool HasExtension(const std::vector<vk::ExtensionProperties>& extensions, const char* name) {
	return std::any_of(extensions.begin(), extensions.end(),
	                   [name](const auto& ext) { return strcmp(ext.extensionName, name) == 0; });
}

// A compute shader that queries an image's LOD (ImageQueryLod) declares ComputeDerivativeGroupQuadsKHR
// (spirvEmitterModule.cpp): the device enables the feature when it has it. Without it the SPIR-V is invalid for the
// device, as it always was; most drivers accept it.
static bool HasExtension(const std::vector<const char*>& extensions, const char* name);
static void ChainComputeDerivatives(vk::PhysicalDevice physical_device, const std::vector<const char*>& extensions,
                                    vk::DeviceCreateInfo& create_info,
                                    vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR& derivatives) {
	if (!HasExtension(extensions, VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME)) return;
	vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR supported {};
	vk::PhysicalDeviceFeatures2                           features {};
	features.pNext = &supported;
	physical_device.getFeatures2(&features);
	if (!supported.computeDerivativeGroupQuads) return;
	derivatives                             = {};
	derivatives.computeDerivativeGroupQuads = VK_TRUE;
	derivatives.pNext                       = const_cast<void*>(create_info.pNext);
	create_info.pNext                       = &derivatives;
}

static bool HasExtension(const std::vector<const char*>& extensions, const char* name) {
	return std::any_of(extensions.begin(), extensions.end(),
	                   [name](const char* ext) { return strcmp(ext, name) == 0; });
}

static bool HasLayer(const std::vector<vk::LayerProperties>& layers, const char* name) {
	return std::any_of(layers.begin(), layers.end(),
	                   [name](const auto& layer) { return strcmp(layer.layerName, name) == 0; });
}

static void GetSurfaceCapabilities(vk::PhysicalDevice physical_device, vk::SurfaceKHR surface,
                                   SurfaceCapabilities& r) {
	RequireVulkanSuccess(physical_device.getSurfaceCapabilitiesKHR(surface, &r.capabilities),
	                     "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

	r.formats = EnumerateVulkan<vk::SurfaceFormatKHR>( // @suppress("Ambiguous problem")
	    "vkGetPhysicalDeviceSurfaceFormatsKHR", [&](uint32_t* count, vk::SurfaceFormatKHR* values) {
		    return physical_device.getSurfaceFormatsKHR(surface, count, values);
	    });
	EXIT_NOT_IMPLEMENTED(r.formats.empty());

	r.present_modes = EnumerateVulkan<vk::PresentModeKHR>( // @suppress("Ambiguous problem")
	    "vkGetPhysicalDeviceSurfacePresentModesKHR",
	    [&](uint32_t* count, vk::PresentModeKHR* values) {
		    return physical_device.getSurfacePresentModesKHR(surface, count, values);
	    });
	EXIT_NOT_IMPLEMENTED(r.present_modes.empty());
}

static bool CheckFormat(vk::PhysicalDevice device, vk::Format format,
                        vk::FormatFeatureFlags features) {
	return (device.getFormatProperties(format).optimalTilingFeatures & features) == features;
}

static uint32_t VulkanFindQueueFamily(vk::PhysicalDevice device, vk::SurfaceKHR surface) {
	EXIT_IF(device == nullptr);
	EXIT_IF(surface == nullptr);

	uint32_t queue_family_count = 0;
	device.getQueueFamilyProperties(&queue_family_count, nullptr);
	std::vector<vk::QueueFamilyProperties> queue_families(queue_family_count);
	device.getQueueFamilyProperties(&queue_family_count, queue_families.data());

	const auto required = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
	for (uint32_t family = 0; family < queue_family_count; family++) {
		const auto& properties             = queue_families[family];
		vk::Bool32  presentation_supported = VK_FALSE;
		RequireVulkanSuccess(device.getSurfaceSupportKHR(family, surface, &presentation_supported),
		                     "vkGetPhysicalDeviceSurfaceSupportKHR");

		LOGF("\tqueue family: %s [count = %u], [present = %s]\n",
		     vk::to_string(properties.queueFlags).c_str(), properties.queueCount,
		     (presentation_supported == VK_TRUE ? "true" : "false"));
		if (properties.queueCount != 0 && (properties.queueFlags & required) == required &&
		    presentation_supported == VK_TRUE) {
			LOGF("\tselected universal queue family %u\n", family);
			return family;
		}
	}
	return static_cast<uint32_t>(-1);
}

// On failure out_device is null and out_rejections says why each device was skipped.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static void VulkanFindPhysicalDevice(vk::Instance instance, vk::SurfaceKHR surface,
                                     const std::vector<const char*>& device_extensions,
                                     vk::PhysicalDevice& out_device, uint32_t& out_queue_family,
                                     std::string& out_rejections) {
	EXIT_IF(instance == nullptr);
	EXIT_IF(surface == nullptr);

	auto devices = EnumerateVulkan<vk::PhysicalDevice>(
	    "vkEnumeratePhysicalDevices", [&](uint32_t* count, vk::PhysicalDevice* values) {
		    return instance.enumeratePhysicalDevices(count, values);
	    });
	EXIT_NOT_IMPLEMENTED(devices.empty());

	if (Config::GetGpuIndex() >= 0) {
		if (static_cast<size_t>(Config::GetGpuIndex()) < devices.size()) {
			devices = {devices[Config::GetGpuIndex()]};
		} else {
			LOGF("Vulkan GPU index %d is unavailable; selecting automatically\n",
			     Config::GetGpuIndex());
		}
	} else if (vk::PhysicalDevice headset; Vr::OpenXrPhysicalDevice(instance, headset)) {
		devices = {headset}; // The headset runtime renders on this GPU.
	}

	vk::PhysicalDevice best_device       = nullptr;
	uint32_t           best_queue_family = static_cast<uint32_t>(-1);
	bool               best_discrete     = false;
	vk::DeviceSize     best_local        = 0;

	for (const auto& device: devices) {
		bool skip_device = false;

		vk::PhysicalDeviceProperties device_properties {};
		device.getProperties(&device_properties);

		const auto reject = [&](const std::string& reason) {
			LOGF("%s\n", reason.c_str());
			out_rejections +=
			    skip_device ? "; " : fmt::format("\n  {}: ", device_properties.deviceName.data());
			out_rejections += reason;
			skip_device = true;
		};

		LOGF("Vulkan device: %s\n", device_properties.deviceName.data());
		if (device_properties.apiVersion < VULKAN_TARGET_API_VERSION) {
			reject(fmt::format("Vulkan {}.{} is required, but device supports only {}.{}.{}",
			                   VK_VERSION_MAJOR(VULKAN_TARGET_API_VERSION),
			                   VK_VERSION_MINOR(VULKAN_TARGET_API_VERSION),
			                   VK_VERSION_MAJOR(device_properties.apiVersion),
			                   VK_VERSION_MINOR(device_properties.apiVersion),
			                   VK_VERSION_PATCH(device_properties.apiVersion)));
			continue;
		}

		vk::PhysicalDeviceFeatures2 device_features2 {};

		vk::PhysicalDeviceVulkan13Features features13 {};

		vk::PhysicalDeviceColorWriteEnableFeaturesEXT color_write_ext {};
		vk::PhysicalDeviceImageViewMinLodFeaturesEXT  image_view_min_lod {};
		color_write_ext.pNext = &image_view_min_lod;

		vk::PhysicalDeviceDepthClipEnableFeaturesEXT depth_clip_enable {};
		depth_clip_enable.pNext = &color_write_ext;

		vk::PhysicalDeviceDepthClipControlFeaturesEXT depth_clip_control {};
		depth_clip_control.pNext = &depth_clip_enable;

		vk::PhysicalDeviceVulkan12Features features12 {};
		vk::PhysicalDeviceVulkan11Features features11 {};
		vk::PhysicalDeviceRobustness2FeaturesEXT robustness2 {};
#if defined(__APPLE__)
		features12.pNext = &depth_clip_control;
#else
		vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR fragment_barycentric {};
		fragment_barycentric.pNext = &depth_clip_control;
		features12.pNext           = &fragment_barycentric;
#endif
		robustness2.pNext      = features12.pNext;
		features11.pNext       = &robustness2;
		features12.pNext       = &features11;
		features13.pNext       = &features12;
		device_features2.pNext = &features13;

		device.getFeatures2(&device_features2);
		const auto required_features12 = WindowContext::RequiredVulkan12Features();
		const auto required_features13 = WindowContext::RequiredVulkan13Features();

		const auto queue_family = VulkanFindQueueFamily(device, surface);
		if (queue_family == static_cast<uint32_t>(-1)) {
			reject("No universal graphics, compute, and presentation queue");
		}

		const auto check_feature = [&](vk::Bool32 supported, const char* name,
		                               vk::Bool32 required = VK_TRUE) {
			if (required == VK_TRUE && supported != VK_TRUE) {
				reject(fmt::format("{} is not supported", name));
			}
		};

#if defined(__APPLE__)
		if (color_write_ext.colorWriteEnable != VK_TRUE) {
			LOGF("colorWriteEnable is not supported\n");
		}
#else
		check_feature(color_write_ext.colorWriteEnable, "colorWriteEnable");
#endif
		check_feature(image_view_min_lod.minLod, "image view minLod");

		check_feature(depth_clip_control.depthClipControl, "depthClipControl");
#if defined(__APPLE__)
		if (depth_clip_enable.depthClipEnable != VK_TRUE) {
			LOGF("depthClipEnable is not supported\n");
		}
#else
		check_feature(depth_clip_enable.depthClipEnable, "depthClipEnable");
#endif
#if !defined(__APPLE__)
		check_feature(device_features2.features.depthClamp, "depthClamp");
#endif

		check_feature(features12.samplerMirrorClampToEdge, "samplerMirrorClampToEdge",
		              required_features12.samplerMirrorClampToEdge);
		check_feature(features11.storageBuffer16BitAccess, "storageBuffer16BitAccess",
		              WindowContext::RequiredVulkan11Features().storageBuffer16BitAccess);
		check_feature(features11.shaderDrawParameters, "shaderDrawParameters",
		              WindowContext::RequiredVulkan11Features().shaderDrawParameters);
		check_feature(features12.storageBuffer8BitAccess, "storageBuffer8BitAccess",
		              required_features12.storageBuffer8BitAccess);
		check_feature(features12.timelineSemaphore, "timelineSemaphore",
		              required_features12.timelineSemaphore);
		check_feature(features12.shaderOutputLayer, "shaderOutputLayer",
		              required_features12.shaderOutputLayer);
		check_feature(features12.shaderOutputViewportIndex, "shaderOutputViewportIndex",
		              required_features12.shaderOutputViewportIndex);
		check_feature(features12.bufferDeviceAddress, "bufferDeviceAddress",
		              required_features12.bufferDeviceAddress);
		check_feature(features12.shaderBufferInt64Atomics, "shaderBufferInt64Atomics",
		              required_features12.shaderBufferInt64Atomics);
		check_feature(features12.shaderSampledImageArrayNonUniformIndexing,
		              "shaderSampledImageArrayNonUniformIndexing",
		              required_features12.shaderSampledImageArrayNonUniformIndexing);
		check_feature(features12.shaderStorageImageArrayNonUniformIndexing,
		              "shaderStorageImageArrayNonUniformIndexing",
		              required_features12.shaderStorageImageArrayNonUniformIndexing);
		check_feature(features12.shaderStorageBufferArrayNonUniformIndexing,
		              "shaderStorageBufferArrayNonUniformIndexing",
		              required_features12.shaderStorageBufferArrayNonUniformIndexing);
		check_feature(features12.descriptorBindingSampledImageUpdateAfterBind,
		              "descriptorBindingSampledImageUpdateAfterBind",
		              required_features12.descriptorBindingSampledImageUpdateAfterBind);
		check_feature(features12.descriptorBindingUpdateUnusedWhilePending,
		              "descriptorBindingUpdateUnusedWhilePending",
		              required_features12.descriptorBindingUpdateUnusedWhilePending);
		check_feature(features12.descriptorBindingPartiallyBound, "descriptorBindingPartiallyBound",
		              required_features12.descriptorBindingPartiallyBound);
		check_feature(features13.robustImageAccess, "robustImageAccess");
		check_feature(robustness2.nullDescriptor, "nullDescriptor");
		check_feature(features13.dynamicRendering, "dynamicRendering",
		              required_features13.dynamicRendering);
		check_feature(features13.synchronization2, "synchronization2",
		              required_features13.synchronization2);
		check_feature(device_features2.features.sampleRateShading, "sampleRateShading");
		check_feature(device_features2.features.depthBiasClamp, "depthBiasClamp");
		check_feature(device_features2.features.shaderClipDistance, "shaderClipDistance");
		check_feature(device_features2.features.shaderCullDistance, "shaderCullDistance");
		check_feature(device_features2.features.largePoints, "largePoints");
		check_feature(device_features2.features.multiViewport, "multiViewport");
		check_feature(device_features2.features.fillModeNonSolid, "fillModeNonSolid");
		check_feature(device_features2.features.fragmentStoresAndAtomics,
		              "fragmentStoresAndAtomics");

		check_feature(device_features2.features.samplerAnisotropy, "samplerAnisotropy");
		check_feature(device_features2.features.robustBufferAccess, "robustBufferAccess");
#if defined(__APPLE__)
		if (device_features2.features.depthBounds != VK_TRUE) {
			LOGF("depthBounds is not supported\n");
		}
#else
		check_feature(device_features2.features.depthBounds, "depthBounds");
#endif
		check_feature(device_features2.features.shaderStorageImageWriteWithoutFormat,
		              "shaderStorageImageWriteWithoutFormat");

		check_feature(device_features2.features.shaderImageGatherExtended,
		              "shaderImageGatherExtended");
		check_feature(device_features2.features.shaderResourceMinLod, "shaderResourceMinLod");

		check_feature(device_features2.features.independentBlend, "independentBlend");
		check_feature(device_features2.features.tessellationShader, "tessellationShader");
		check_feature(device_features2.features.drawIndirectFirstInstance,
		              "drawIndirectFirstInstance");
		check_feature(device_features2.features.multiDrawIndirect, "multiDrawIndirect");

		if (!skip_device) {
			auto available_extensions = EnumerateVulkan<vk::ExtensionProperties>(
			    "vkEnumerateDeviceExtensionProperties",
			    [&](uint32_t* count, vk::ExtensionProperties* values) {
				    return device.enumerateDeviceExtensionProperties(nullptr, count, values);
			    });
			EXIT_NOT_IMPLEMENTED(available_extensions.empty());

			for (const char* ext: device_extensions) {
				if (!HasExtension(available_extensions, ext)) {
					reject(fmt::format("{} is not supported", ext));
				}
			}

			if (skip_device) {
				for (const auto& ext: available_extensions) {
					LOGF("Vulkan available extension: %s, version = %u\n", ext.extensionName.data(),
					     ext.specVersion);
				}
			}
		}

		SurfaceCapabilities candidate_capabilities;
		if (!skip_device) {
			GetSurfaceCapabilities(device, surface, candidate_capabilities);

			if (!(candidate_capabilities.capabilities.supportedUsageFlags &
			      vk::ImageUsageFlagBits::eTransferDst)) {
				reject("Surface cannot be destination of blit");
			}
		}

		const auto check_format = [&](vk::Format format, vk::FormatFeatureFlags features,
		                              const char* usage) {
			if (!skip_device && !CheckFormat(device, format, features)) {
				reject(fmt::format("Format vk::Format::e{} cannot be used as {}",
				                   vk::to_string(format), usage));
			}
		};

		const auto depth_features = vk::FormatFeatureFlagBits::eDepthStencilAttachment;
		const auto texture_features =
		    vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferDst;
		const auto storage_features =
		    vk::FormatFeatureFlagBits::eStorageImage | vk::FormatFeatureFlagBits::eTransferDst;

		check_format(vk::Format::eD32Sfloat, depth_features, "depth buffer");
		check_format(vk::Format::eD32SfloatS8Uint, depth_features, "depth buffer");
		check_format(vk::Format::eD16Unorm, depth_features, "depth buffer");
		check_format(vk::Format::eBc3SrgbBlock, texture_features, "texture");
		check_format(vk::Format::eR8G8B8A8Srgb, texture_features, "texture");
		check_format(vk::Format::eR8Unorm, texture_features, "texture");
		check_format(vk::Format::eR8G8Unorm, texture_features, "texture");

		for (const auto [srgb, unorm]:
		     {std::pair {vk::Format::eR8G8B8A8Srgb, vk::Format::eR8G8B8A8Unorm},
		      std::pair {vk::Format::eB8G8R8A8Srgb, vk::Format::eB8G8R8A8Unorm}}) {
			if (!skip_device && !CheckFormat(device, srgb, storage_features)) {
				LOGF("Format vk::Format::e%s cannot be used as storage image\n",
				     vk::to_string(srgb).c_str());
				check_format(unorm, storage_features, "storage image");
			}
		}

		if (!skip_device && device_properties.limits.maxSamplerAnisotropy < 16.0f) {
			reject("maxSamplerAnisotropy < 16.0f");
		}

		if (skip_device) {
			continue;
		}

		// A discrete GPU first, then the most device-local memory, the first on a tie: the launcher shows and stamps
		// the card with the most dedicated memory (wolverineLauncher.cpp), and it was the last discrete one here.
		const bool     discrete = device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
		vk::DeviceSize local    = 0;
		for (const auto& heap: device.getMemoryProperties().memoryHeaps) {
			if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) local = std::max(local, heap.size);
		}
		if (best_device == nullptr || (discrete && !best_discrete) || (discrete == best_discrete && local > best_local)) {
			best_device       = device;
			best_queue_family = queue_family;
			best_discrete     = discrete;
			best_local        = local;
		}
	}

	out_device       = best_device;
	out_queue_family = best_queue_family;
}

// KYTY_SPARSE_GEOMETRY (on unless 0): game buffers of 1 GiB or more are sparse. Every page is bound, to memory where the guest has mapped
// some and else to one shared zero page (aliased), so no read meets an unbound page: AMD's scalar loads can hang on one
// (vkd3d-proton, device_workarounds.c). Without aliasing, unbound pages must read 0 (residencyNonResidentStrict).
static void ChooseSparseBuffers(GraphicContext& graphics, vk::PhysicalDevice physical_device,
                                const vk::PhysicalDeviceFeatures& supported, bool queue_binds_sparse) {
	const char* value  = std::getenv("KYTY_SPARSE_GEOMETRY");
	const bool  wanted = value == nullptr || value[0] != '0';
	const bool  strict = physical_device.getProperties().sparseProperties.residencyNonResidentStrict;
	graphics.sparse_buffers = wanted && queue_binds_sparse && supported.sparseBinding && supported.sparseResidencyBuffer &&
	                          (supported.sparseResidencyAliased || strict);
	graphics.sparse_buffers_aliased = graphics.sparse_buffers && supported.sparseResidencyAliased;
	vk::PhysicalDeviceVulkan13Properties properties13 {};
	vk::PhysicalDeviceProperties2        properties2 {};
	properties2.pNext = &properties13;
	physical_device.getProperties2(&properties2);
	graphics.max_buffer_size = properties13.maxBufferSize;
	LOGF("Sparse game buffers: %s%s, maxBufferSize %llu MiB\n", graphics.sparse_buffers ? "enabled" : "off",
	     graphics.sparse_buffers ? (graphics.sparse_buffers_aliased ? " (zero page)" : " (unbound pages)") : "",
	     static_cast<unsigned long long>(graphics.max_buffer_size >> 20u));
}

static vk::Device VulkanCreateDevice(GraphicContext& graphics, vk::SurfaceKHR surface,
	                                 const std::vector<const char*>& device_extensions) {
	const auto physical_device = graphics.physical_device;
	const auto queue_family    = graphics.queue_family;
	EXIT_IF(physical_device == nullptr);
	EXIT_IF(queue_family == static_cast<uint32_t>(-1));

	const std::array<float, 2> queue_priority {1.0f, 1.0f};
	vk::DeviceQueueCreateInfo  queue_create_info {};
	queue_create_info.queueFamilyIndex = queue_family;
	queue_create_info.queueCount       = 1;
	queue_create_info.pQueuePriorities = queue_priority.data();

	vk::PhysicalDeviceColorWriteEnableFeaturesEXT color_write_ext {};
	color_write_ext.colorWriteEnable = VK_TRUE;

	vk::PhysicalDeviceDepthClipEnableFeaturesEXT depth_clip_enable {};
	depth_clip_enable.pNext = &color_write_ext;
	depth_clip_enable.depthClipEnable = VK_TRUE;

	vk::PhysicalDeviceDepthClipControlFeaturesEXT depth_clip_control {};
	vk::PhysicalDeviceImageViewMinLodFeaturesEXT  image_view_min_lod {};
	image_view_min_lod.minLod = VK_TRUE;
	depth_clip_control.pNext  = &image_view_min_lod;
	// MoltenVK lacks VK_EXT_depth_clip_enable and VK_EXT_color_write_enable, so drop those
	// feature structs from the chain on macOS (the renderer falls back to default depth
	// clipping and static color-write masks).
#if !defined(__APPLE__)
	image_view_min_lod.pNext = &depth_clip_enable;
#endif
	depth_clip_control.depthClipControl = VK_TRUE;

	const bool workgroup_layout_extension =
	    HasExtension(device_extensions, VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
	vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR supported_workgroup_layout {};
	vk::PhysicalDeviceVulkan12Features supported_features12 {};
	supported_features12.pNext = workgroup_layout_extension ? &supported_workgroup_layout : nullptr;
	vk::PhysicalDeviceVulkan13Features supported_features13 {};
	supported_features13.pNext = &supported_features12;

	vk::PhysicalDeviceRobustness2FeaturesEXT supported_robustness2 {};
	supported_robustness2.pNext = supported_features13.pNext;
	supported_features13.pNext = &supported_robustness2;

	const bool mesh_extension = HasExtension(device_extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME);
	vk::PhysicalDeviceMeshShaderFeaturesEXT supported_mesh {};
	supported_mesh.pNext = &supported_features13;
	vk::PhysicalDeviceFeatures2 supported_features2 {};
	supported_features2.pNext = mesh_extension ? static_cast<void*>(&supported_mesh)
	                                           : static_cast<void*>(&supported_features13);
	const bool feedback_extensions =
	    HasExtension(device_extensions, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME) &&
	    HasExtension(device_extensions, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME);
	vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT feedback_layout {};
	vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT feedback_dynamic {};
	if (feedback_extensions) {
		feedback_dynamic.pNext = supported_features2.pNext;
		feedback_layout.pNext  = &feedback_dynamic;
		supported_features2.pNext = &feedback_layout;
	}
	const bool provoking_extension =
	    HasExtension(device_extensions, VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME);
	vk::PhysicalDeviceProvokingVertexFeaturesEXT provoking_vertex {};
	if (provoking_extension) {
		provoking_vertex.pNext = supported_features2.pNext;
		supported_features2.pNext = &provoking_vertex;
	}
	const bool maintenance8_extension =
	    HasExtension(device_extensions, VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
	vk::PhysicalDeviceMaintenance8FeaturesKHR maintenance8 {};
	if (maintenance8_extension) {
		maintenance8.pNext = supported_features2.pNext;
		supported_features2.pNext = &maintenance8;
	}
	const bool maintenance6_extension =
	    HasExtension(device_extensions, VK_KHR_MAINTENANCE_6_EXTENSION_NAME);
	vk::PhysicalDeviceMaintenance6FeaturesKHR maintenance6 {};
	if (maintenance6_extension) {
		maintenance6.pNext = supported_features2.pNext;
		supported_features2.pNext = &maintenance6;
	}
	const bool shader_clock_extension =
	    HasExtension(device_extensions, VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
	vk::PhysicalDeviceShaderClockFeaturesKHR shader_clock {};
	if (shader_clock_extension) {
		shader_clock.pNext = supported_features2.pNext;
		supported_features2.pNext = &shader_clock;
	}
	const bool image_atomic_int64_extension =
	    HasExtension(device_extensions, VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME);
	vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT image_atomic_int64 {};
	if (image_atomic_int64_extension) {
		image_atomic_int64.pNext = supported_features2.pNext;
		supported_features2.pNext = &image_atomic_int64;
	}
	const bool conditional_rendering_extension =
	    HasExtension(device_extensions, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
	vk::PhysicalDeviceConditionalRenderingFeaturesEXT conditional_rendering {};
	if (conditional_rendering_extension) {
		conditional_rendering.pNext = supported_features2.pNext;
		supported_features2.pNext   = &conditional_rendering;
	}
	const bool gpl_extension = HasExtension(device_extensions, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
	vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT supported_gpl {};
	if (gpl_extension) {
		supported_gpl.pNext       = supported_features2.pNext;
		supported_features2.pNext = &supported_gpl;
	}
	const bool barycentric_extension =
	    HasExtension(device_extensions, VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
	vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR supported_barycentric {};
	if (barycentric_extension) {
		supported_barycentric.pNext = supported_features2.pNext;
		supported_features2.pNext   = &supported_barycentric;
	}
	const bool ray_query_extensions = HasExtension(device_extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME);
	vk::PhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure {};
	vk::PhysicalDeviceRayQueryFeaturesKHR              ray_query {};
	if (ray_query_extensions) {
		ray_query.pNext              = &acceleration_structure;
		acceleration_structure.pNext = supported_features2.pNext;
		supported_features2.pNext    = &ray_query;
	}
	const bool fsr = Fsr::Enabled();
	vk::PhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_matrix {};
	if (fsr && HasExtension(device_extensions, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
		cooperative_matrix.pNext  = supported_features2.pNext;
		supported_features2.pNext = &cooperative_matrix;
	}
	vk::PhysicalDeviceShaderFloat8FeaturesEXT float8 {};
	if (fsr && HasExtension(device_extensions, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME)) {
		float8.pNext              = supported_features2.pNext;
		supported_features2.pNext = &float8;
	}
	physical_device.getFeatures2(&supported_features2);
	const bool barycentric_supported =
	    barycentric_extension && supported_barycentric.fragmentShaderBarycentric == VK_TRUE;
	const bool ray_query_enabled =
	    ray_query_extensions && acceleration_structure.accelerationStructure && ray_query.rayQuery;
	if (!barycentric_supported) {
		Log::WriteToConsoleAndLog(
		    "WARNING: Your GPU is not fully supported (fragmentShaderBarycentric is unavailable). "
		    "Some games may render incorrectly or crash.\n");
	}
	graphics.shader_image_int64_atomics_enabled = image_atomic_int64.shaderImageInt64Atomics;
	graphics.sampler_filter_minmax_enabled      = supported_features12.samplerFilterMinmax;

	auto features12 = WindowContext::RequiredVulkan12Features();
	features12.shaderSharedInt64Atomics = supported_features12.shaderSharedInt64Atomics;
	features12.samplerFilterMinmax      = supported_features12.samplerFilterMinmax;
	{
		// The image heap is one set of twelve arrays of 65,536 sampled images, written while bound.
		constexpr uint32_t HeapImages = 12u * 65536u;
		// And the game's sampler heap, a float and an integer sampler a record.
		constexpr uint32_t HeapSamplers = 2u * 256u;
		vk::PhysicalDeviceVulkan12Properties properties12 {};
		vk::PhysicalDeviceProperties2        properties2 {};
		properties2.pNext = &properties12;
		physical_device.getProperties2(&properties2);
		graphics.guest_image_heap_enabled =
		    supported_features12.descriptorBindingSampledImageUpdateAfterBind &&
		    supported_features12.descriptorBindingPartiallyBound &&
		    supported_features12.descriptorBindingUpdateUnusedWhilePending &&
		    properties12.maxDescriptorSetUpdateAfterBindSampledImages >= HeapImages &&
		    properties12.maxPerStageDescriptorUpdateAfterBindSampledImages >= HeapImages &&
		    properties12.maxPerStageUpdateAfterBindResources >= HeapImages + 1u &&
		    properties12.maxUpdateAfterBindDescriptorsInAllPools >= HeapImages + 1u + HeapSamplers &&
		    properties12.maxDescriptorSetUpdateAfterBindSamplers >= HeapSamplers &&
		    properties12.maxPerStageDescriptorUpdateAfterBindSamplers >= HeapSamplers;
		// KYTY_IMAGE_HEAP=0 leaves the heap off: upstream's per-draw descriptor tables.
		if (const char* off = std::getenv("KYTY_IMAGE_HEAP"); off != nullptr && off[0] == '0') {
			graphics.guest_image_heap_enabled = false;
		}
		// The three descriptor-binding features it needs are required ones (the resident image heap's).
		LOGF("Image heap: %s\n", graphics.guest_image_heap_enabled ? "enabled" : "unavailable on this device");
	}
	vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR workgroup_layout {};
	workgroup_layout.workgroupMemoryExplicitLayout =
	    supported_workgroup_layout.workgroupMemoryExplicitLayout;
	workgroup_layout.pNext = &depth_clip_control;
	features12.pNext = workgroup_layout_extension ? static_cast<void*>(&workgroup_layout)
	                                             : static_cast<void*>(&depth_clip_control);
	auto features11 = WindowContext::RequiredVulkan11Features();
	features11.pNext = features12.pNext;
	features12.pNext = &features11;
	if (!features12.shaderSharedInt64Atomics || !workgroup_layout.workgroupMemoryExplicitLayout) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "WARNING: Native 64-bit LDS atomics are unavailable: shaderSharedInt64Atomics={}, "
		    "workgroupMemoryExplicitLayout={}. Continuing with the selected Vulkan device.\n",
		    features12.shaderSharedInt64Atomics != VK_FALSE,
		    workgroup_layout.workgroupMemoryExplicitLayout != VK_FALSE));
	}
	graphics.mesh_shader_enabled = mesh_extension && supported_mesh.meshShader;
	// Upstream abc509d33: indirect mesh draws launch their records' groups from a task shader.
	graphics.task_shader_enabled =
	    graphics.mesh_shader_enabled && supported_mesh.taskShader && KYTY_UPSTREAM_PATH("abc509d33");

	vk::PhysicalDeviceSubgroupSizeControlProperties subgroup_size_control {};

	vk::PhysicalDeviceVulkan11Properties properties11 {};
	properties11.pNext = &subgroup_size_control;

	vk::PhysicalDeviceFloatControlsProperties float_controls {};
	float_controls.pNext = &properties11;
	vk::PhysicalDeviceRobustness2PropertiesEXT robustness2_properties {};
	robustness2_properties.pNext = &float_controls;
	vk::PhysicalDeviceProperties2 properties2 {};
	properties2.pNext = &robustness2_properties;

	if (graphics.mesh_shader_enabled) {
		subgroup_size_control.pNext = &graphics.mesh_shader_properties;
	}
	vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT gpl_properties {};
	if (gpl_extension) {
		gpl_properties.pNext = properties2.pNext;
		properties2.pNext    = &gpl_properties;
	}
	vk::PhysicalDeviceMaintenance6PropertiesKHR maintenance6_properties {};
	if (maintenance6_extension) {
		maintenance6_properties.pNext = properties2.pNext;
		properties2.pNext             = &maintenance6_properties;
	}
	physical_device.getProperties2(&properties2);
	// Enabled for this property only: without it, a cube's uncompressed alias views break the spec.
	graphics.block_texel_view_multiple_layers =
	    maintenance6_extension && maintenance6.maintenance6 &&
	    maintenance6_properties.blockTexelViewCompatibleMultipleLayers;
	graphics.robust_storage_buffer_alignment = static_cast<uint32_t>(
	    std::max(robustness2_properties.robustStorageBufferAccessSizeAlignment, vk::DeviceSize {1}));
	graphics.graphics_pipeline_library_enabled =
	    gpl_extension && supported_gpl.graphicsPipelineLibrary && gpl_properties.graphicsPipelineLibraryFastLinking;
	if (gpl_extension) {
		LOGF("Vulkan graphics pipeline library: %s\n", graphics.graphics_pipeline_library_enabled ? "on" : "no fast linking");
	}
	EXIT_NOT_IMPLEMENTED(!(properties11.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eVote));

	graphics.subgroup_size                 = properties11.subgroupSize;
	graphics.min_subgroup_size             = subgroup_size_control.minSubgroupSize;
	graphics.max_subgroup_size             = subgroup_size_control.maxSubgroupSize;
	graphics.required_subgroup_size_stages = subgroup_size_control.requiredSubgroupSizeStages;
	graphics.compute_subgroup_size_control_enabled =
	    supported_features13.subgroupSizeControl == VK_TRUE &&
	    (graphics.required_subgroup_size_stages & vk::ShaderStageFlagBits::eCompute) &&
	    subgroup_size_control.minSubgroupSize <= 64 &&
	    subgroup_size_control.maxSubgroupSize >= 64;

	graphics.compute_full_subgroups_enabled =
	    supported_features13.computeFullSubgroups == VK_TRUE &&
	    graphics.min_subgroup_size == graphics.subgroup_size &&
	    graphics.max_subgroup_size == graphics.subgroup_size;

	LOGF("Vulkan subgroup: default=%u min=%u max=%u stages=0x%08x size_control=%s wave64=%s "
	     "full_subgroups=%s\n",
	     graphics.subgroup_size, graphics.min_subgroup_size, graphics.max_subgroup_size,
	     static_cast<vk::ShaderStageFlags::MaskType>(graphics.required_subgroup_size_stages),
	     graphics.compute_subgroup_size_control_enabled ? "true" : "false",
	     graphics.SupportsComputeWave64() ? "true" : "false",
	     graphics.compute_full_subgroups_enabled ? "true" : "false");
	graphics.provoking_vertex_last_enabled = provoking_extension && provoking_vertex.provokingVertexLast;
	graphics.attachment_feedback_loop_enabled =
	    feedback_extensions && feedback_layout.attachmentFeedbackLoopLayout &&
	    feedback_dynamic.attachmentFeedbackLoopDynamicState;
	graphics.maintenance8_enabled = maintenance8_extension && maintenance8.maintenance8;
	graphics.r8_srgb_views   = graphics.SamplesFiltered(vk::Format::eR8Srgb);
	graphics.r8g8_srgb_views = graphics.SamplesFiltered(vk::Format::eR8G8Srgb);
	// The device clock ticks at timestampPeriod nanoseconds.
	if (shader_clock_extension && shader_clock.shaderDeviceClock) {
		graphics.device_clock_period_ps = static_cast<uint32_t>(
		    std::lround(graphics.physical_device_properties.limits.timestampPeriod * 1000.0f));
	}
	graphics.conditional_rendering_enabled =
	    conditional_rendering_extension && conditional_rendering.conditionalRendering;
	LOGF("Vulkan maintenance8 support: %s\n",
	     graphics.maintenance8_enabled ? "true" : "false");
	LOGF("Vulkan block texel views of several layers (maintenance6): %s\n",
	     graphics.block_texel_view_multiple_layers ? "true" : "false");
	LOGF("Vulkan depth feedback support: %s\n",
	     graphics.attachment_feedback_loop_enabled ? "true" : "false");
	if (graphics.mesh_shader_enabled) {
		LOGF("Vulkan MeshEXT: invocations=%u vertices=%u primitives=%u shared=%u\n",
		     graphics.mesh_shader_properties.maxMeshWorkGroupInvocations,
		     graphics.mesh_shader_properties.maxMeshOutputVertices,
		     graphics.mesh_shader_properties.maxMeshOutputPrimitives,
		     graphics.mesh_shader_properties.maxMeshSharedMemorySize);
	}
	// VulkanFindPhysicalDevice already checked the required creation features. These
	// requirements are specific to this creation path and are not part of device selection.
	EXIT_NOT_IMPLEMENTED(supported_features2.features.shaderInt64 != VK_TRUE);
	EXIT_NOT_IMPLEMENTED(supported_features2.features.vertexPipelineStoresAndAtomics != VK_TRUE);
	EXIT_NOT_IMPLEMENTED(supported_features2.features.dualSrcBlend != VK_TRUE);
	vk::PhysicalDeviceFeatures device_features {};
	device_features.fragmentStoresAndAtomics = VK_TRUE;
	device_features.samplerAnisotropy        = VK_TRUE;
	device_features.robustBufferAccess       = VK_TRUE;
#if !defined(__APPLE__)
	device_features.depthBounds = VK_TRUE; // unsupported by MoltenVK
	device_features.depthClamp  = VK_TRUE;
#endif
	device_features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
	device_features.shaderStorageImageMultisample        =
	    supported_features2.features.shaderStorageImageMultisample;
	device_features.shaderImageGatherExtended            = VK_TRUE;
	device_features.shaderResourceMinLod                 = VK_TRUE;
	device_features.independentBlend                     = VK_TRUE;
	device_features.dualSrcBlend                         = VK_TRUE;
	device_features.tessellationShader                   = VK_TRUE;
	device_features.drawIndirectFirstInstance            = VK_TRUE;
	device_features.multiDrawIndirect                    = VK_TRUE;
	device_features.sampleRateShading                    = VK_TRUE;
	device_features.depthBiasClamp                       = VK_TRUE;
	device_features.shaderClipDistance                   = VK_TRUE;
	device_features.shaderCullDistance                   = VK_TRUE;
	device_features.largePoints                          = VK_TRUE;
	device_features.multiViewport                        = VK_TRUE;
	device_features.fillModeNonSolid                      = VK_TRUE;
	device_features.vertexPipelineStoresAndAtomics       = VK_TRUE;
	// Storage image sample counts beyond 1x require the feature.
	if (!device_features.shaderStorageImageMultisample) {
		graphics.physical_device_properties.limits.storageImageSampleCounts =
		    vk::SampleCountFlagBits::e1;
	}
	graphics.sample_rate_shading_enabled                 = true;
	{
		// Streamed textures hold only their loaded levels (sparse residency) where the device binds
		// sparse images on our queue; KYTY_TEXTURE_SPARSE=0 makes them ordinary images as before.
		uint32_t family_count = 0;
		physical_device.getQueueFamilyProperties(&family_count, nullptr);
		std::vector<vk::QueueFamilyProperties> families(family_count);
		physical_device.getQueueFamilyProperties(&family_count, families.data());
		const char* sparse = std::getenv("KYTY_TEXTURE_SPARSE");
		graphics.sparse_textures_enabled =
		    supported_features2.features.sparseBinding && supported_features2.features.sparseResidencyImage2D &&
		    queue_family < family_count &&
		    static_cast<bool>(families[queue_family].queueFlags & vk::QueueFlagBits::eSparseBinding) &&
		    !(sparse != nullptr && sparse[0] == '0');
		// The BDA page table (768 MB flat) is bound only where the guest has buffers; KYTY_SPARSE_PAGE_TABLE=0 keeps it dense.
		const char* sparse_table = std::getenv("KYTY_SPARSE_PAGE_TABLE");
		ChooseSparseBuffers(graphics, physical_device, supported_features2.features,
		                    queue_family < family_count &&
		                        static_cast<bool>(families[queue_family].queueFlags & vk::QueueFlagBits::eSparseBinding));
		graphics.sparse_page_table =
		    supported_features2.features.sparseBinding && supported_features2.features.sparseResidencyBuffer &&
		    physical_device.getProperties().sparseProperties.residencyNonResidentStrict && queue_family < family_count &&
		    static_cast<bool>(families[queue_family].queueFlags & vk::QueueFlagBits::eSparseBinding) &&
		    !(sparse_table != nullptr && sparse_table[0] == '0');
		device_features.sparseBinding          = graphics.sparse_textures_enabled || graphics.sparse_page_table || graphics.sparse_buffers;
		device_features.sparseResidencyImage2D = graphics.sparse_textures_enabled;
		device_features.sparseResidencyBuffer  = graphics.sparse_page_table || graphics.sparse_buffers;
		device_features.sparseResidencyAliased = graphics.sparse_buffers_aliased;
		// Binds go to a second queue of the family where it has one (as vkd3d-proton binds on its own
		// submission thread): the GPU thread does not wait for the driver's binds.
		graphics.sparse_queue_index =
		    (graphics.sparse_textures_enabled || graphics.sparse_page_table || graphics.sparse_buffers) && families[queue_family].queueCount > 1 ? 1u : 0u;
		queue_create_info.queueCount = graphics.sparse_queue_index + 1u;
		LOGF("Sparse textures: %s, sparse page table: %s, binds on queue %u\n",
		     graphics.sparse_textures_enabled ? "enabled" : "off", graphics.sparse_page_table ? "enabled" : "off",
		     graphics.sparse_queue_index);
	}
	// Occlusion queries reset their queries from the host and count samples exactly where they can.
	graphics.occlusion_queries_enabled =
	    Config::OcclusionQueriesEnabled() && supported_features12.hostQueryReset;
	graphics.occlusion_query_precise_enabled =
	    graphics.occlusion_queries_enabled && supported_features2.features.occlusionQueryPrecise;
	features12.hostQueryReset             = graphics.occlusion_queries_enabled;
	device_features.occlusionQueryPrecise = graphics.occlusion_query_precise_enabled;
	LOGF("Vulkan ray query: %s\n", ray_query_enabled ? "true" : "false");
	LOGF("Vulkan occlusion queries: %s, precise: %s\n",
	     graphics.occlusion_queries_enabled ? "true" : "false",
	     graphics.occlusion_query_precise_enabled ? "true" : "false");
	device_features.shaderInt64 = VK_TRUE;
	device_features.shaderFloat64 =
	    supported_features2.features.shaderFloat64 &&
	    float_controls.shaderSignedZeroInfNanPreserveFloat64;
	// if (device_features.shaderFloat64 && !float_controls.shaderDenormPreserveFloat64) {
	// 	Log::WriteToConsoleAndLog(
	// 	    "WARNING: Vulkan device does not guarantee FP64 denormal preservation; "
	// 	    "continuing with native FP64 arithmetic. Very small values may be flushed to zero.\n");
	// }

	vk::PhysicalDeviceRobustness2FeaturesEXT robustness2 {};
#if defined(__APPLE__)
	robustness2.pNext = &features12;
#else
	vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR fragment_barycentric {};
	if (barycentric_supported) {
		fragment_barycentric.pNext                     = &features12;
		fragment_barycentric.fragmentShaderBarycentric = VK_TRUE;
		robustness2.pNext                              = &fragment_barycentric;
	} else {
		robustness2.pNext = &features12;
	}
#endif
	robustness2.robustBufferAccess2 = supported_robustness2.robustBufferAccess2;
	graphics.robust_buffer_access2  = robustness2.robustBufferAccess2 == VK_TRUE;
	robustness2.robustImageAccess2  = supported_robustness2.robustImageAccess2;
	robustness2.nullDescriptor      = VK_TRUE;

	auto features13 = WindowContext::RequiredVulkan13Features();
	features13.pNext = &robustness2;
	features13.robustImageAccess   = supported_features13.robustImageAccess;
	features13.subgroupSizeControl =
	    graphics.compute_subgroup_size_control_enabled ? VK_TRUE : VK_FALSE;
	features13.computeFullSubgroups = graphics.compute_full_subgroups_enabled ? VK_TRUE : VK_FALSE;
	if (fsr) {
		device_features.shaderInt16            = supported_features2.features.shaderInt16;
		features12.shaderFloat16               = supported_features12.shaderFloat16;
		features12.shaderInt8                  = supported_features12.shaderInt8;
		features12.separateDepthStencilLayouts = supported_features12.separateDepthStencilLayouts;
		features13.shaderIntegerDotProduct     = supported_features13.shaderIntegerDotProduct;
		features13.computeFullSubgroups        = supported_features13.computeFullSubgroups;
	}

	LOGF("Vulkan robustness: robustImageAccess=%s robustImageAccess2=%s robustBufferAccess2=%s "
	     "storage alignment=%u\n",
	     features13.robustImageAccess == VK_TRUE ? "true" : "false",
	     robustness2.robustImageAccess2 == VK_TRUE ? "true" : "false",
	     graphics.robust_buffer_access2 ? "true" : "false", graphics.robust_storage_buffer_alignment);

	vk::DeviceCreateInfo create_info {};
	vk::PhysicalDeviceMeshShaderFeaturesEXT mesh_features {};
	mesh_features.pNext                 = &features13;
	mesh_features.meshShader            = graphics.mesh_shader_enabled;
	mesh_features.taskShader            = graphics.task_shader_enabled;
	feedback_dynamic.pNext =
	    mesh_extension ? static_cast<void*>(&mesh_features) : static_cast<void*>(&features13);
	create_info.pNext = graphics.attachment_feedback_loop_enabled
	                        ? static_cast<void*>(&feedback_layout)
	                        : feedback_dynamic.pNext;
	if (graphics.provoking_vertex_last_enabled) {
		provoking_vertex.pNext = const_cast<void*>(create_info.pNext);
		provoking_vertex.transformFeedbackPreservesProvokingVertex = VK_FALSE;
		create_info.pNext = &provoking_vertex;
	}
	if (graphics.shader_image_int64_atomics_enabled) {
		image_atomic_int64.pNext = const_cast<void*>(create_info.pNext);
		image_atomic_int64.sparseImageInt64Atomics = VK_FALSE;
		create_info.pNext = &image_atomic_int64;
	}
	if (graphics.maintenance8_enabled) {
		maintenance8.pNext = const_cast<void*>(create_info.pNext);
		create_info.pNext = &maintenance8;
	}
	if (graphics.block_texel_view_multiple_layers) {
		maintenance6 = {};
		maintenance6.maintenance6 = VK_TRUE;
		maintenance6.pNext = const_cast<void*>(create_info.pNext);
		create_info.pNext = &maintenance6;
	}
	vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl_features {};
	if (graphics.graphics_pipeline_library_enabled) {
		gpl_features.graphicsPipelineLibrary = VK_TRUE;
		gpl_features.pNext                   = const_cast<void*>(create_info.pNext);
		create_info.pNext                    = &gpl_features;
	}
	if (ray_query_enabled) {
		acceleration_structure                       = {};
		acceleration_structure.accelerationStructure = VK_TRUE;
		ray_query                                    = {};
		ray_query.rayQuery                           = VK_TRUE;
		ray_query.pNext              = &acceleration_structure;
		acceleration_structure.pNext = const_cast<void*>(create_info.pNext);
		create_info.pNext            = &ray_query;
	}
	if (cooperative_matrix.cooperativeMatrix) {
		cooperative_matrix = {};
		cooperative_matrix.cooperativeMatrix = VK_TRUE;
		cooperative_matrix.pNext             = const_cast<void*>(create_info.pNext);
		create_info.pNext                    = &cooperative_matrix;
	}
	if (float8.shaderFloat8) {
		float8 = {};
		float8.shaderFloat8 = VK_TRUE;
		float8.pNext        = const_cast<void*>(create_info.pNext);
		create_info.pNext   = &float8;
	}
	if (graphics.device_clock_period_ps != 0u) {
		shader_clock.pNext = const_cast<void*>(create_info.pNext);
		create_info.pNext = &shader_clock;
	}
	if (graphics.conditional_rendering_enabled) {
		conditional_rendering.pNext = const_cast<void*>(create_info.pNext);
		create_info.pNext = &conditional_rendering;
	}
	// The renderer only uses the Vulkan 1.3 core dynamic states, but RADV (Mesa 26.3) creates its
	// PS epilog cache only when this EXT feature is enabled and crashes in vkCmdDraw when a
	// depth- or stencil-writing fragment shader needs an epilog.
	vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extended_dynamic_state {};
	extended_dynamic_state.extendedDynamicState = VK_TRUE;
	if (HasExtension(device_extensions, VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME)) {
		extended_dynamic_state.pNext = const_cast<void*>(create_info.pNext);
		create_info.pNext = &extended_dynamic_state;
	}
	// Frame generation's swapchain presents and acquires on queues of its own (KYTY_FSR3_FG=1).
	std::vector<vk::DeviceQueueCreateInfo> queue_infos {queue_create_info};
	Fsr3::FrameGen::AddQueues(physical_device, surface, queue_infos);
	create_info.pQueueCreateInfos       = queue_infos.data();
	create_info.queueCreateInfoCount    = static_cast<uint32_t>(queue_infos.size());
	create_info.enabledExtensionCount   = static_cast<uint32_t>(device_extensions.size());
	create_info.ppEnabledExtensionNames = device_extensions.data();
	create_info.pEnabledFeatures        = &device_features;
	vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR derivatives {};
	ChainComputeDerivatives(physical_device, device_extensions, create_info, derivatives);

	vk::Device device = nullptr;

	auto result = physical_device.createDevice(&create_info, nullptr, &device);
	if (result != vk::Result::eSuccess) {
		LOGF("vkCreateDevice failed: %s\n", vk::to_string(result).c_str());
		return nullptr;
	}

	return device;
}

// Upstream 8810d3869 builds the Vulkan device from structure chains. This is that construction carrying
// everything lean's VulkanCreateDevice enables or reads (sparse binding and its second queue, the image heap
// limits, GPL, maintenance6, memory priority, full subgroups, samplerFilterMinmax, the robust storage alignment),
// so nothing is missing with KYTY_UPSTREAM_PATHS=8810d3869. Lean's CreateVulkan still picks the extensions.
static vk::Device VulkanCreateDeviceChains(GraphicContext& graphics, vk::SurfaceKHR surface,
                                           const std::vector<const char*>& device_extensions) {
	const auto physical_device = graphics.physical_device;
	const auto queue_family    = graphics.queue_family;
	EXIT_IF(physical_device == nullptr);
	EXIT_IF(queue_family == static_cast<uint32_t>(-1));
	const auto has = [&](const char* name) { return HasExtension(device_extensions, name); };

	// Optional feature structs stay in the query only with their extension.
	vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
	                   vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceRobustness2FeaturesEXT,
	                   vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR,
	                   vk::PhysicalDeviceMeshShaderFeaturesEXT,
	                   vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT,
	                   vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT,
	                   vk::PhysicalDeviceProvokingVertexFeaturesEXT, vk::PhysicalDeviceMaintenance8FeaturesKHR,
	                   vk::PhysicalDeviceMaintenance6FeaturesKHR, vk::PhysicalDeviceShaderClockFeaturesKHR,
	                   vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT,
	                   vk::PhysicalDeviceConditionalRenderingFeaturesEXT,
	                   vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT,
	                   vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR,
	                   vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceRayQueryFeaturesKHR,
	                   vk::PhysicalDeviceCooperativeMatrixFeaturesKHR, vk::PhysicalDeviceShaderFloat8FeaturesEXT>
	    supported;
	auto& [supported2, supported12, supported13, supported_robustness2, supported_workgroup, supported_mesh,
	       feedback_layout, feedback_dynamic, provoking_vertex, maintenance8, maintenance6, shader_clock,
	       image_atomic_int64, conditional_rendering, supported_gpl, supported_barycentric,
	       supported_acceleration_structure, supported_ray_query, cooperative_matrix, float8] = supported;
	const bool workgroup_layout_extension = has(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
	const bool mesh_extension             = has(VK_EXT_MESH_SHADER_EXTENSION_NAME);
	const bool feedback_extensions        = has(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME) &&
	                                 has(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME);
	const bool provoking_extension             = has(VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME);
	const bool maintenance8_extension          = has(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
	const bool maintenance6_extension          = has(VK_KHR_MAINTENANCE_6_EXTENSION_NAME);
	const bool shader_clock_extension          = has(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
	const bool image_atomic_int64_extension    = has(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME);
	const bool conditional_rendering_extension = has(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
	const bool gpl_extension                   = has(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
	const bool barycentric_extension           = has(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
	if (!workgroup_layout_extension) supported.unlink<vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>();
	if (!mesh_extension) supported.unlink<vk::PhysicalDeviceMeshShaderFeaturesEXT>();
	if (!feedback_extensions) {
		supported.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT>();
		supported.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT>();
	}
	if (!provoking_extension) supported.unlink<vk::PhysicalDeviceProvokingVertexFeaturesEXT>();
	if (!maintenance8_extension) supported.unlink<vk::PhysicalDeviceMaintenance8FeaturesKHR>();
	if (!maintenance6_extension) supported.unlink<vk::PhysicalDeviceMaintenance6FeaturesKHR>();
	if (!shader_clock_extension) supported.unlink<vk::PhysicalDeviceShaderClockFeaturesKHR>();
	if (!image_atomic_int64_extension) supported.unlink<vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT>();
	if (!conditional_rendering_extension) supported.unlink<vk::PhysicalDeviceConditionalRenderingFeaturesEXT>();
	if (!gpl_extension) supported.unlink<vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>();
	if (!barycentric_extension) supported.unlink<vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>();
	// Ray queries take all three extensions or none (CreateVulkan adds them with --native-rt).
	const bool ray_query_extension = has(VK_KHR_RAY_QUERY_EXTENSION_NAME);
	if (!ray_query_extension) {
		supported.unlink<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();
		supported.unlink<vk::PhysicalDeviceRayQueryFeaturesKHR>();
	}
	// FSR's ML upscalers (AMD's DLLs) run on cooperative matrices; CreateVulkan adds these with them.
	const bool fsr = Fsr::Enabled();
	if (!has(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) supported.unlink<vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>();
	if (!has(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME)) supported.unlink<vk::PhysicalDeviceShaderFloat8FeaturesEXT>();
	physical_device.getFeatures2(&supported2);
	const bool barycentric_supported =
	    barycentric_extension && supported_barycentric.fragmentShaderBarycentric == VK_TRUE;
	const bool ray_query_enabled = ray_query_extension && supported_acceleration_structure.accelerationStructure &&
	                               supported_ray_query.rayQuery;
	LOGF("Vulkan ray query: %s\n", ray_query_enabled ? "true" : "false");
	if (!barycentric_supported) {
		Log::WriteToConsoleAndLog(
		    "WARNING: Your GPU is not fully supported (fragmentShaderBarycentric is unavailable). "
		    "Some games may render incorrectly or crash.\n");
	}
	graphics.shader_image_int64_atomics_enabled =
	    image_atomic_int64_extension && image_atomic_int64.shaderImageInt64Atomics;
	graphics.sampler_filter_minmax_enabled = supported12.samplerFilterMinmax;

	auto features12                     = WindowContext::RequiredVulkan12Features();
	features12.shaderSharedInt64Atomics = supported12.shaderSharedInt64Atomics;
	features12.samplerFilterMinmax      = supported12.samplerFilterMinmax;
	{
		// The image heap is one set of twelve arrays of 65,536 sampled images, written while bound.
		constexpr uint32_t HeapImages = 12u * 65536u;
		// And the game's sampler heap, a float and an integer sampler a record.
		constexpr uint32_t HeapSamplers = 2u * 256u;
		vk::StructureChain<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceVulkan12Properties> heap_chain;
		physical_device.getProperties2(&heap_chain.get<vk::PhysicalDeviceProperties2>());
		const auto& properties12 = heap_chain.get<vk::PhysicalDeviceVulkan12Properties>();
		graphics.guest_image_heap_enabled =
		    supported12.descriptorBindingSampledImageUpdateAfterBind && supported12.descriptorBindingPartiallyBound &&
		    supported12.descriptorBindingUpdateUnusedWhilePending &&
		    properties12.maxDescriptorSetUpdateAfterBindSampledImages >= HeapImages &&
		    properties12.maxPerStageDescriptorUpdateAfterBindSampledImages >= HeapImages &&
		    properties12.maxPerStageUpdateAfterBindResources >= HeapImages + 1u &&
		    properties12.maxUpdateAfterBindDescriptorsInAllPools >= HeapImages + 1u + HeapSamplers &&
		    properties12.maxDescriptorSetUpdateAfterBindSamplers >= HeapSamplers &&
		    properties12.maxPerStageDescriptorUpdateAfterBindSamplers >= HeapSamplers;
		// KYTY_IMAGE_HEAP=0 leaves the heap off: upstream's per-draw descriptor tables.
		if (const char* off = std::getenv("KYTY_IMAGE_HEAP"); off != nullptr && off[0] == '0') {
			graphics.guest_image_heap_enabled = false;
		}
		LOGF("Image heap: %s\n", graphics.guest_image_heap_enabled ? "enabled" : "unavailable on this device");
	}
	const auto workgroup_memory_explicit_layout =
	    workgroup_layout_extension ? supported_workgroup.workgroupMemoryExplicitLayout : vk::Bool32 {VK_FALSE};
	if (!features12.shaderSharedInt64Atomics || !workgroup_memory_explicit_layout) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "WARNING: Native 64-bit LDS atomics are unavailable: shaderSharedInt64Atomics={}, "
		    "workgroupMemoryExplicitLayout={}. Continuing with the selected Vulkan device.\n",
		    features12.shaderSharedInt64Atomics != VK_FALSE, workgroup_memory_explicit_layout != VK_FALSE));
	}
	graphics.mesh_shader_enabled = mesh_extension && supported_mesh.meshShader;
	// Upstream abc509d33: indirect mesh draws launch their records' groups from a task shader.
	graphics.task_shader_enabled =
	    graphics.mesh_shader_enabled && supported_mesh.taskShader && KYTY_UPSTREAM_PATH("abc509d33");

	vk::StructureChain<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRobustness2PropertiesEXT,
	                   vk::PhysicalDeviceFloatControlsProperties, vk::PhysicalDeviceVulkan11Properties,
	                   vk::PhysicalDeviceSubgroupSizeControlProperties, vk::PhysicalDeviceMeshShaderPropertiesEXT,
	                   vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT,
	                   vk::PhysicalDeviceMaintenance6PropertiesKHR>
	    properties;
	auto& [properties2, robustness2_properties, float_controls, properties11, subgroup_size_control,
	       mesh_properties, gpl_properties, maintenance6_properties] = properties;
	if (!graphics.mesh_shader_enabled) properties.unlink<vk::PhysicalDeviceMeshShaderPropertiesEXT>();
	if (!gpl_extension) properties.unlink<vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT>();
	if (!maintenance6_extension) properties.unlink<vk::PhysicalDeviceMaintenance6PropertiesKHR>();
	physical_device.getProperties2(&properties2);
	if (graphics.mesh_shader_enabled) {
		graphics.mesh_shader_properties       = mesh_properties;
		graphics.mesh_shader_properties.pNext = nullptr;
	}
	// Enabled for this property only: without it, a cube's uncompressed alias views break the spec.
	graphics.block_texel_view_multiple_layers = maintenance6_extension && maintenance6.maintenance6 &&
	                                            maintenance6_properties.blockTexelViewCompatibleMultipleLayers;
	graphics.robust_storage_buffer_alignment = static_cast<uint32_t>(
	    std::max(robustness2_properties.robustStorageBufferAccessSizeAlignment, vk::DeviceSize {1}));
	graphics.graphics_pipeline_library_enabled =
	    gpl_extension && supported_gpl.graphicsPipelineLibrary && gpl_properties.graphicsPipelineLibraryFastLinking;
	if (gpl_extension) {
		LOGF("Vulkan graphics pipeline library: %s\n", graphics.graphics_pipeline_library_enabled ? "on" : "no fast linking");
	}
	EXIT_NOT_IMPLEMENTED(!(properties11.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eVote));

	graphics.subgroup_size                 = properties11.subgroupSize;
	graphics.min_subgroup_size             = subgroup_size_control.minSubgroupSize;
	graphics.max_subgroup_size             = subgroup_size_control.maxSubgroupSize;
	graphics.required_subgroup_size_stages = subgroup_size_control.requiredSubgroupSizeStages;
	graphics.compute_subgroup_size_control_enabled =
	    supported13.subgroupSizeControl == VK_TRUE &&
	    (graphics.required_subgroup_size_stages & vk::ShaderStageFlagBits::eCompute) &&
	    subgroup_size_control.minSubgroupSize <= 64 && subgroup_size_control.maxSubgroupSize >= 64;
	graphics.compute_full_subgroups_enabled = supported13.computeFullSubgroups == VK_TRUE &&
	                                          graphics.min_subgroup_size == graphics.subgroup_size &&
	                                          graphics.max_subgroup_size == graphics.subgroup_size;
	LOGF("Vulkan subgroup: default=%u min=%u max=%u stages=0x%08x size_control=%s wave64=%s "
	     "full_subgroups=%s\n",
	     graphics.subgroup_size, graphics.min_subgroup_size, graphics.max_subgroup_size,
	     static_cast<vk::ShaderStageFlags::MaskType>(graphics.required_subgroup_size_stages),
	     graphics.compute_subgroup_size_control_enabled ? "true" : "false",
	     graphics.SupportsComputeWave64() ? "true" : "false",
	     graphics.compute_full_subgroups_enabled ? "true" : "false");
	graphics.provoking_vertex_last_enabled = provoking_extension && provoking_vertex.provokingVertexLast;
	graphics.attachment_feedback_loop_enabled = feedback_extensions && feedback_layout.attachmentFeedbackLoopLayout &&
	                                            feedback_dynamic.attachmentFeedbackLoopDynamicState;
	graphics.maintenance8_enabled = maintenance8_extension && maintenance8.maintenance8;
	graphics.r8_srgb_views        = graphics.SamplesFiltered(vk::Format::eR8Srgb);
	graphics.r8g8_srgb_views      = graphics.SamplesFiltered(vk::Format::eR8G8Srgb);
	// The device clock ticks at timestampPeriod nanoseconds.
	if (shader_clock_extension && shader_clock.shaderDeviceClock) {
		graphics.device_clock_period_ps = static_cast<uint32_t>(
		    std::lround(graphics.physical_device_properties.limits.timestampPeriod * 1000.0f));
	}
	graphics.conditional_rendering_enabled = conditional_rendering_extension && conditional_rendering.conditionalRendering;
	LOGF("Vulkan maintenance8 support: %s\n", graphics.maintenance8_enabled ? "true" : "false");
	LOGF("Vulkan block texel views of several layers (maintenance6): %s\n",
	     graphics.block_texel_view_multiple_layers ? "true" : "false");
	LOGF("Vulkan depth feedback support: %s\n", graphics.attachment_feedback_loop_enabled ? "true" : "false");
	if (graphics.mesh_shader_enabled) {
		LOGF("Vulkan MeshEXT: invocations=%u vertices=%u primitives=%u shared=%u\n",
		     graphics.mesh_shader_properties.maxMeshWorkGroupInvocations,
		     graphics.mesh_shader_properties.maxMeshOutputVertices,
		     graphics.mesh_shader_properties.maxMeshOutputPrimitives,
		     graphics.mesh_shader_properties.maxMeshSharedMemorySize);
	}
	EXIT_NOT_IMPLEMENTED(supported2.features.shaderInt64 != VK_TRUE);
	EXIT_NOT_IMPLEMENTED(supported2.features.vertexPipelineStoresAndAtomics != VK_TRUE);
	EXIT_NOT_IMPLEMENTED(supported2.features.dualSrcBlend != VK_TRUE);
	vk::PhysicalDeviceFeatures device_features {};
	device_features.fragmentStoresAndAtomics = VK_TRUE;
	device_features.samplerAnisotropy        = VK_TRUE;
	device_features.robustBufferAccess       = VK_TRUE;
#if !defined(__APPLE__)
	device_features.depthBounds = VK_TRUE; // unsupported by MoltenVK
	device_features.depthClamp  = VK_TRUE;
#endif
	device_features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
	device_features.shaderStorageImageMultisample        = supported2.features.shaderStorageImageMultisample;
	device_features.shaderImageGatherExtended            = VK_TRUE;
	device_features.shaderResourceMinLod                 = VK_TRUE;
	device_features.independentBlend                     = VK_TRUE;
	device_features.dualSrcBlend                         = VK_TRUE;
	device_features.tessellationShader                   = VK_TRUE;
	device_features.drawIndirectFirstInstance            = VK_TRUE;
	device_features.multiDrawIndirect                    = VK_TRUE;
	device_features.sampleRateShading                    = VK_TRUE;
	device_features.depthBiasClamp                       = VK_TRUE;
	device_features.shaderClipDistance                   = VK_TRUE;
	device_features.shaderCullDistance                   = VK_TRUE;
	device_features.largePoints                          = VK_TRUE;
	device_features.multiViewport                        = VK_TRUE;
	device_features.fillModeNonSolid                     = VK_TRUE;
	device_features.vertexPipelineStoresAndAtomics       = VK_TRUE;
	// Storage image sample counts beyond 1x require the feature.
	if (!device_features.shaderStorageImageMultisample) {
		graphics.physical_device_properties.limits.storageImageSampleCounts = vk::SampleCountFlagBits::e1;
	}
	graphics.sample_rate_shading_enabled = true;
	{
		// As lean's VulkanCreateDevice: sparse textures and the sparse page table, binds on a second queue.
		uint32_t family_count = 0;
		physical_device.getQueueFamilyProperties(&family_count, nullptr);
		std::vector<vk::QueueFamilyProperties> families(family_count);
		physical_device.getQueueFamilyProperties(&family_count, families.data());
		const char* sparse = std::getenv("KYTY_TEXTURE_SPARSE");
		graphics.sparse_textures_enabled =
		    supported2.features.sparseBinding && supported2.features.sparseResidencyImage2D &&
		    queue_family < family_count &&
		    static_cast<bool>(families[queue_family].queueFlags & vk::QueueFlagBits::eSparseBinding) &&
		    !(sparse != nullptr && sparse[0] == '0');
		const char* sparse_table = std::getenv("KYTY_SPARSE_PAGE_TABLE");
		ChooseSparseBuffers(graphics, physical_device, supported2.features,
		                    queue_family < family_count &&
		                        static_cast<bool>(families[queue_family].queueFlags & vk::QueueFlagBits::eSparseBinding));
		graphics.sparse_page_table =
		    supported2.features.sparseBinding && supported2.features.sparseResidencyBuffer &&
		    physical_device.getProperties().sparseProperties.residencyNonResidentStrict && queue_family < family_count &&
		    static_cast<bool>(families[queue_family].queueFlags & vk::QueueFlagBits::eSparseBinding) &&
		    !(sparse_table != nullptr && sparse_table[0] == '0');
		device_features.sparseBinding          = graphics.sparse_textures_enabled || graphics.sparse_page_table || graphics.sparse_buffers;
		device_features.sparseResidencyImage2D = graphics.sparse_textures_enabled;
		device_features.sparseResidencyBuffer  = graphics.sparse_page_table || graphics.sparse_buffers;
		device_features.sparseResidencyAliased = graphics.sparse_buffers_aliased;
		graphics.sparse_queue_index =
		    (graphics.sparse_textures_enabled || graphics.sparse_page_table || graphics.sparse_buffers) && families[queue_family].queueCount > 1
		        ? 1u
		        : 0u;
		LOGF("Sparse textures: %s, sparse page table: %s, binds on queue %u\n",
		     graphics.sparse_textures_enabled ? "enabled" : "off", graphics.sparse_page_table ? "enabled" : "off",
		     graphics.sparse_queue_index);
	}
	// Occlusion queries reset their queries from the host and count samples exactly where they can.
	graphics.occlusion_queries_enabled = Config::OcclusionQueriesEnabled() && supported12.hostQueryReset;
	graphics.occlusion_query_precise_enabled =
	    graphics.occlusion_queries_enabled && supported2.features.occlusionQueryPrecise;
	features12.hostQueryReset             = graphics.occlusion_queries_enabled;
	device_features.occlusionQueryPrecise = graphics.occlusion_query_precise_enabled;
	LOGF("Vulkan occlusion queries: %s, precise: %s\n", graphics.occlusion_queries_enabled ? "true" : "false",
	     graphics.occlusion_query_precise_enabled ? "true" : "false");
	device_features.shaderInt64 = VK_TRUE;
	device_features.shaderFloat64 =
	    supported2.features.shaderFloat64 && float_controls.shaderSignedZeroInfNanPreserveFloat64;

	vk::PhysicalDeviceRobustness2FeaturesEXT robustness2 {};
	robustness2.robustBufferAccess2 = supported_robustness2.robustBufferAccess2;
	robustness2.robustImageAccess2  = supported_robustness2.robustImageAccess2;
	robustness2.nullDescriptor      = VK_TRUE;
	graphics.robust_buffer_access2  = robustness2.robustBufferAccess2 == VK_TRUE;

	auto features13                 = WindowContext::RequiredVulkan13Features();
	features13.robustImageAccess    = supported13.robustImageAccess;
	features13.subgroupSizeControl  = graphics.compute_subgroup_size_control_enabled ? VK_TRUE : VK_FALSE;
	features13.computeFullSubgroups = graphics.compute_full_subgroups_enabled ? VK_TRUE : VK_FALSE;
	if (fsr) {
		device_features.shaderInt16            = supported2.features.shaderInt16;
		features12.shaderFloat16               = supported12.shaderFloat16;
		features12.shaderInt8                  = supported12.shaderInt8;
		features12.separateDepthStencilLayouts = supported12.separateDepthStencilLayouts;
		features13.shaderIntegerDotProduct     = supported13.shaderIntegerDotProduct;
		features13.computeFullSubgroups        = supported13.computeFullSubgroups;
	}
	LOGF("Vulkan robustness: robustImageAccess=%s robustImageAccess2=%s robustBufferAccess2=%s "
	     "storage alignment=%u\n",
	     features13.robustImageAccess == VK_TRUE ? "true" : "false",
	     robustness2.robustImageAccess2 == VK_TRUE ? "true" : "false",
	     graphics.robust_buffer_access2 ? "true" : "false", graphics.robust_storage_buffer_alignment);

	const std::array<float, 2> queue_priority {1.0f, 1.0f};
	vk::DeviceQueueCreateInfo  queue_create_info {};
	queue_create_info.queueFamilyIndex = queue_family;
	queue_create_info.queueCount       = graphics.sparse_queue_index + 1u;
	queue_create_info.pQueuePriorities = queue_priority.data();
	vk::DeviceCreateInfo create_info {};
	// Frame generation's swapchain presents and acquires on queues of its own (KYTY_FSR3_FG=1).
	std::vector<vk::DeviceQueueCreateInfo> queue_infos {queue_create_info};
	Fsr3::FrameGen::AddQueues(physical_device, surface, queue_infos);
	create_info.pQueueCreateInfos       = queue_infos.data();
	create_info.queueCreateInfoCount    = static_cast<uint32_t>(queue_infos.size());
	create_info.enabledExtensionCount   = static_cast<uint32_t>(device_extensions.size());
	create_info.ppEnabledExtensionNames = device_extensions.data();
	create_info.pEnabledFeatures        = &device_features;

	vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR workgroup_layout {};
	workgroup_layout.workgroupMemoryExplicitLayout = workgroup_memory_explicit_layout;
	vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentric {};
	barycentric.fragmentShaderBarycentric = VK_TRUE;
	vk::PhysicalDeviceMeshShaderFeaturesEXT mesh_features {};
	mesh_features.meshShader = VK_TRUE;
	mesh_features.taskShader = graphics.task_shader_enabled;
	vk::PhysicalDeviceMaintenance6FeaturesKHR maintenance6_features {};
	maintenance6_features.maintenance6 = VK_TRUE;
	vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl_features {};
	gpl_features.graphicsPipelineLibrary = VK_TRUE;
	vk::PhysicalDeviceDepthClipControlFeaturesEXT depth_clip_control {};
	depth_clip_control.depthClipControl = VK_TRUE;
	vk::PhysicalDeviceImageViewMinLodFeaturesEXT image_view_min_lod {};
	image_view_min_lod.minLod = VK_TRUE;
	vk::PhysicalDeviceDepthClipEnableFeaturesEXT depth_clip_enable {};
	depth_clip_enable.depthClipEnable = VK_TRUE;
	vk::PhysicalDeviceColorWriteEnableFeaturesEXT color_write {};
	color_write.colorWriteEnable = VK_TRUE;
	// The renderer only uses the Vulkan 1.3 core dynamic states, but RADV (Mesa 26.3) creates its PS epilog cache
	// only when this EXT feature is enabled and crashes in vkCmdDraw when a depth- or stencil-writing fragment
	// shader needs an epilog.
	vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extended_dynamic_state {};
	extended_dynamic_state.extendedDynamicState = VK_TRUE;
	provoking_vertex.transformFeedbackPreservesProvokingVertex = VK_FALSE;
	image_atomic_int64.sparseImageInt64Atomics                 = VK_FALSE;

	vk::StructureChain create {create_info,
	                           WindowContext::RequiredVulkan11Features(),
	                           features12,
	                           features13,
	                           robustness2,
	                           depth_clip_control,
	                           image_view_min_lod,
	                           depth_clip_enable,
	                           color_write,
	                           workgroup_layout,
	                           barycentric,
	                           mesh_features,
	                           feedback_layout,
	                           feedback_dynamic,
	                           provoking_vertex,
	                           image_atomic_int64,
	                           maintenance8,
	                           maintenance6_features,
	                           gpl_features,
	                           shader_clock,
	                           conditional_rendering,
	                           extended_dynamic_state,
	                           vk::PhysicalDeviceAccelerationStructureFeaturesKHR {.accelerationStructure = VK_TRUE},
	                           vk::PhysicalDeviceRayQueryFeaturesKHR {.rayQuery = VK_TRUE},
	                           cooperative_matrix,
	                           float8};
	if (!workgroup_layout_extension) create.unlink<vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>();
#if defined(__APPLE__)
	// MoltenVK lacks these extensions (see CreateVulkan).
	create.unlink<vk::PhysicalDeviceDepthClipEnableFeaturesEXT>();
	create.unlink<vk::PhysicalDeviceColorWriteEnableFeaturesEXT>();
	create.unlink<vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>();
#else
	if (!barycentric_supported) create.unlink<vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>();
#endif
	if (!graphics.mesh_shader_enabled) create.unlink<vk::PhysicalDeviceMeshShaderFeaturesEXT>();
	if (!graphics.attachment_feedback_loop_enabled) {
		create.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT>();
		create.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT>();
	}
	if (!graphics.provoking_vertex_last_enabled) create.unlink<vk::PhysicalDeviceProvokingVertexFeaturesEXT>();
	if (!graphics.shader_image_int64_atomics_enabled) {
		create.unlink<vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT>();
	}
	if (!graphics.maintenance8_enabled) create.unlink<vk::PhysicalDeviceMaintenance8FeaturesKHR>();
	if (!graphics.block_texel_view_multiple_layers) create.unlink<vk::PhysicalDeviceMaintenance6FeaturesKHR>();
	if (!graphics.graphics_pipeline_library_enabled) {
		create.unlink<vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>();
	}
	if (graphics.device_clock_period_ps == 0u) create.unlink<vk::PhysicalDeviceShaderClockFeaturesKHR>();
	if (!graphics.conditional_rendering_enabled) create.unlink<vk::PhysicalDeviceConditionalRenderingFeaturesEXT>();
	if (!has(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME)) {
		create.unlink<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
	}
	if (!cooperative_matrix.cooperativeMatrix) {
		create.unlink<vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>();
	}
	if (!float8.shaderFloat8) create.unlink<vk::PhysicalDeviceShaderFloat8FeaturesEXT>();
	if (!ray_query_enabled) {
		create.unlink<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();
		create.unlink<vk::PhysicalDeviceRayQueryFeaturesKHR>();
	}

	vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR derivatives {};
	ChainComputeDerivatives(physical_device, device_extensions, create.get(), derivatives);
	vk::Device device = nullptr;
	const auto result = physical_device.createDevice(&create.get(), nullptr, &device);
	if (result != vk::Result::eSuccess) {
		LOGF("vkCreateDevice failed: %s\n", vk::to_string(result).c_str());
		return nullptr;
	}
	LOGF("upstream 8810d3869: device built from structure chains\n");
	return device;
}

static void VulkanGetExtensions(VulkanExtensions& r) {
	uint32_t required_extensions_count = 0;

	const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&required_extensions_count);
	EXIT_NOT_IMPLEMENTED(extensions == nullptr);
	EXIT_NOT_IMPLEMENTED(required_extensions_count == 0);
	r.required_extensions.assign(extensions, extensions + required_extensions_count);

	r.available_extensions =
	    EnumerateVulkan<vk::ExtensionProperties>( // @suppress("Ambiguous problem")
	        "vkEnumerateInstanceExtensionProperties",
	        [](uint32_t* count, vk::ExtensionProperties* values) {
		        return vk::enumerateInstanceExtensionProperties(nullptr, count, values);
	        });

	r.enable_validation_layers = Config::VulkanValidationEnabled();
	if (Config::HdrEnabled() &&
	    HasExtension(r.available_extensions, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME) &&
	    !HasExtension(r.required_extensions, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME)) {
		r.required_extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
	}

	if (HasExtension(r.available_extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
		r.required_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	} else {
		r.enable_validation_layers = false;
	}

	for (const char* ext: r.required_extensions) {
		LOGF("Vulkan required extension: %s\n", ext);
	}

	for (const auto& ext: r.available_extensions) {
		LOGF("Vulkan available extension: %s, version = %u\n", ext.extensionName.data(),
		     ext.specVersion);
	}

	r.available_layers = EnumerateVulkan<vk::LayerProperties>( // @suppress("Ambiguous problem")
	    "vkEnumerateInstanceLayerProperties", [](uint32_t* count, vk::LayerProperties* values) {
		    return vk::enumerateInstanceLayerProperties(count, values);
	    });

	for (const auto& l: r.available_layers) {
		LOGF("Vulkan available layer: %s, specVersion = %u, implVersion = %u, %s\n",
		     l.layerName.data(), l.specVersion, l.implementationVersion, l.description.data());
	}

	r.required_layers = {"VK_LAYER_KHRONOS_validation"};

	if (r.enable_validation_layers) {
		for (const char* l: r.required_layers) {
			if (!HasLayer(r.available_layers, l)) {
				LOGF("no validation layer: %s\n", l);
				r.enable_validation_layers = false;
				break;
			}
		}
	}

	if (r.enable_validation_layers) {
		auto available_extensions = EnumerateVulkan<vk::ExtensionProperties>(
		    "vkEnumerateInstanceExtensionProperties",
		    [](uint32_t* count, vk::ExtensionProperties* values) {
			    return vk::enumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation",
			                                                    count, values);
		    });

		for (const auto& ext: available_extensions) {
			LOGF("VK_LAYER_KHRONOS_validation available extension: %s, version = %u\n",
			     ext.extensionName.data(), ext.specVersion);
		}

		if (HasExtension(available_extensions, VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME)) {
			r.required_extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
		} else {
			r.enable_validation_layers = false;
		}
	}
}

static VKAPI_ATTR vk::Bool32 VKAPI_CALL VulkanDebugMessengerCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT      message_severity,
    vk::DebugUtilsMessageTypeFlagsEXT             message_types,
    const vk::DebugUtilsMessengerCallbackDataEXT* callback_data, void* /*user_data*/) {
	EXIT_IF(callback_data == nullptr);
	EXIT_IF(callback_data->pMessage == nullptr);

	const char*     severity_str   = nullptr;
	fmt::text_style severity_style = Log::Color::Default;
	bool            skip           = false;
	bool            error          = false;
	bool            debug_printf   = false;
	switch (message_severity) {
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
			severity_str   = "V";
			severity_style = Log::Color::BrightWhite;
			skip           = true;
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
			if ((message_types & vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation) &&
			    Config::SpirvDebugPrintfEnabled() && callback_data->pMessageIdName != nullptr &&
			    strcmp(callback_data->pMessageIdName, "UNASSIGNED-DEBUG-PRINTF") == 0) {
				debug_printf   = true;
				severity_style = Log::Color::BrightYellow;
				skip           = true;
			} else {
				severity_str   = "I";
				severity_style = Log::Color::Default;
				skip           = true;
			}
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
			severity_str   = "W";
			severity_style = Log::Color::Red;
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
			severity_str   = "E";
			severity_style = Log::Color::BrightRed;
			// Only validation errors are fatal; GENERAL-type errors can come
			// from unrelated loader/layer issues (e.g. a broken overlay).
			error = static_cast<bool>(message_types &
			                          vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
			break;
		default: severity_str = "?";
	}

	if (error) {
		EXIT("[Vulkan][%s][%u]: %s\n", severity_str, static_cast<uint32_t>(message_types),
		     callback_data->pMessage);
	}

	if (!skip) {
		LOGF_COLOR(severity_style, "[Vulkan][%s][%u]: %s\n", severity_str,
		           static_cast<uint32_t>(message_types), callback_data->pMessage);
	}

	if (debug_printf) {
		auto strs = Common::Split(std::string(callback_data->pMessage), '|');
		if (!strs.empty()) {
			LOGF_COLOR(severity_style, "%s\n", strs[strs.size() - 1].c_str());
		}
	}

	return VK_FALSE;
}

static VKAPI_ATTR vk::Result VKAPI_CALL VulkanCreateDebugUtilsMessengerEXT(
    vk::Instance instance, const vk::DebugUtilsMessengerCreateInfoEXT* create_info,
    const vk::AllocationCallbacks* allocator, vk::DebugUtilsMessengerEXT* messenger) {
	EXIT_IF(instance == nullptr);

	if (auto func = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDebugUtilsMessengerEXT; func != nullptr) {
		return instance.createDebugUtilsMessengerEXT(create_info, allocator, messenger);
	}
	return vk::Result::eErrorExtensionNotPresent;
}

static void VulkanCheckInstanceVersion() {
	uint32_t version = VK_API_VERSION_1_0;

	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkEnumerateInstanceVersion != nullptr) {
		auto result = vk::enumerateInstanceVersion(&version);
		if (result != vk::Result::eSuccess) {
			EXIT("Could not query Vulkan loader version: %s\n", vk::to_string(result).c_str());
		}
	}

	LOGF("Vulkan loader version: %u.%u.%u\n", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version),
	     VK_VERSION_PATCH(version));
	if (version < VULKAN_TARGET_API_VERSION) {
		EXIT("Vulkan %u.%u is required, but loader supports only %u.%u.%u\n",
		     VK_VERSION_MAJOR(VULKAN_TARGET_API_VERSION),
		     VK_VERSION_MINOR(VULKAN_TARGET_API_VERSION), VK_VERSION_MAJOR(version),
		     VK_VERSION_MINOR(version), VK_VERSION_PATCH(version));
	}
}

void WindowContext::CreateVulkan() {
	EXIT_IF(window == nullptr);
	EXIT_IF(graphic_ctx.instance != nullptr);
	EXIT_IF(graphic_ctx.physical_device != nullptr);
	EXIT_IF(graphic_ctx.device != nullptr);
	EXIT_IF(surface != nullptr);

	auto get_instance_proc_addr =
	    reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
	if (get_instance_proc_addr == nullptr) {
		EXIT("Could not load Vulkan: %s\n", SDL_GetError());
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(get_instance_proc_addr);

	Vr::OpenXrInit();
	Dlss::Load();
	Fsr::Load();

	VulkanExtensions r;
	VulkanGetExtensions(r);
	Vr::OpenXrAppendExtensions(r.required_extensions, r.available_extensions, false);
	Dlss::AppendExtensions(r.required_extensions, r.available_extensions, false);
	VulkanCheckInstanceVersion();

	vk::ApplicationInfo app_info {};
	app_info.pApplicationName   = "Kyty";
	app_info.applicationVersion = 1;
	app_info.pEngineName        = "Kyty";
	app_info.engineVersion      = 1;
	app_info.apiVersion         = VULKAN_TARGET_API_VERSION; // NOLINT

	if (Config::SpirvDebugPrintfEnabled() && Config::GpuAssistedValidationEnabled()) {
		EXIT("--spirv-debug-printf and --gpu-assisted-validation are mutually exclusive\n");
	}

	vk::ValidationFeatureEnableEXT enabled_features[2]    = {};
	uint32_t                       enabled_features_count = 0;
	if (Config::SpirvDebugPrintfEnabled()) {
		enabled_features[enabled_features_count++] = vk::ValidationFeatureEnableEXT::eDebugPrintf;
	}
	if (Config::GpuAssistedValidationEnabled()) {
		enabled_features[enabled_features_count++] = vk::ValidationFeatureEnableEXT::eGpuAssisted;
		LOGF("Vulkan GPU-assisted validation is enabled; expect a large slowdown\n");
		std::printf("Vulkan GPU-assisted validation is enabled; expect a large slowdown\n");
		std::fflush(stdout);
	}

	vk::ValidationFeaturesEXT validation_features {};
	validation_features.enabledValidationFeatureCount  = enabled_features_count;
	validation_features.pEnabledValidationFeatures     = enabled_features;

	vk::DebugUtilsMessengerCreateInfoEXT dbg_create_info {};
	dbg_create_info.pNext           = &validation_features;
	dbg_create_info.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
	                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
	                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
	                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
	dbg_create_info.messageType     = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
	                                  vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
	                                  vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
	dbg_create_info.pfnUserCallback = VulkanDebugMessengerCallback;

	vk::InstanceCreateInfo inst_info {};
	inst_info.pNext                   = (r.enable_validation_layers ? &dbg_create_info : nullptr);
#if defined(__APPLE__)
	// MoltenVK requires VK_KHR_portability_enumeration + flag to surface
	// portability devices. Without this, enumeratePhysicalDevices hides the
	// MoltenVK adapter on some driver versions.
	if (HasExtension(r.available_extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
		if (!HasExtension(r.required_extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
			r.required_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		}
		inst_info.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
		LOGF("Vulkan instance: enabled %s\n", VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
	}
#endif
	inst_info.pApplicationInfo        = &app_info;
	inst_info.enabledExtensionCount   = static_cast<uint32_t>(r.required_extensions.size());
	inst_info.ppEnabledExtensionNames = r.required_extensions.data();
	inst_info.enabledLayerCount =
	    (r.enable_validation_layers ? static_cast<uint32_t>(r.required_layers.size()) : 0);
	inst_info.ppEnabledLayerNames =
	    (r.enable_validation_layers ? r.required_layers.data() : nullptr);

	const vk::Result result = vk::createInstance(&inst_info, nullptr, &graphic_ctx.instance);
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eErrorIncompatibleDriver:
			EXIT("Unable to find a compatible Vulkan Driver");
		default: EXIT("Could not create a Vulkan instance (for unknown reasons)");
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(graphic_ctx.instance);

	if (r.enable_validation_layers) {
		dbg_create_info.pNext = nullptr;
		if (VulkanCreateDebugUtilsMessengerEXT(graphic_ctx.instance, &dbg_create_info, nullptr,
		                                       &graphic_ctx.debug_messenger) !=
		    vk::Result::eSuccess) {
			EXIT("Could not create debug messenger");
		}
	}

	vk::SurfaceKHR::CType native_surface = VK_NULL_HANDLE;
	if (!SDL_Vulkan_CreateSurface(window, static_cast<vk::Instance::CType>(graphic_ctx.instance),
	                              nullptr, &native_surface)) {
		EXIT("Could not create a Vulkan surface");
	}
	surface = native_surface;

	std::vector<const char*> device_extensions = {
	    VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME,
	    VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
	    VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, "VK_KHR_maintenance1"};

#if defined(__APPLE__)
	// MoltenVK lacks VK_EXT_depth_clip_enable and VK_EXT_color_write_enable; the renderer
	// falls back to default depth clipping and static color-write masks on macOS. It also
	// requires VK_KHR_portability_subset per the Vulkan portability spec.
	device_extensions.push_back("VK_KHR_portability_subset");
#else
	device_extensions.push_back(VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME);
	device_extensions.push_back(VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME);
#endif

	if (Config::SpirvDebugPrintfEnabled()) {
		device_extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
	}

	size_t      dlss_extensions_from = device_extensions.size(); // where Dlss::AppendExtensions starts
	std::string rejected_devices;
	VulkanFindPhysicalDevice(graphic_ctx.instance, surface, device_extensions,
	                         graphic_ctx.physical_device, graphic_ctx.queue_family,
	                         rejected_devices);

	if (graphic_ctx.physical_device == nullptr) {
		EXIT("Could not find suitable device:%s", rejected_devices.c_str());
	}

	vk::PhysicalDevicePushDescriptorProperties push_descriptor_properties {};
	vk::PhysicalDeviceProperties2              physical_device_properties {};
	physical_device_properties.pNext = &push_descriptor_properties;
	graphic_ctx.physical_device.getProperties2(&physical_device_properties);
	graphic_ctx.physical_device_properties = physical_device_properties.properties;
	// NVIDIA reloads coherent loads without a barrier; AMD's drivers need one (SpirvEmitter.h).
	ShaderRecompiler::Spirv::SetCoherentLoadAcquire(graphic_ctx.physical_device_properties.vendorID != 0x10deu);
	graphic_ctx.max_push_descriptors       = push_descriptor_properties.maxPushDescriptors;
	graphic_ctx.physical_device.getMemoryProperties(&graphic_ctx.physical_device_memory_properties);
	const auto& device_properties = graphic_ctx.GetPhysicalDeviceProperties();

	LOGF("Select device: %s\n", device_properties.deviceName.data());

	const vk::PhysicalDeviceImageFormatInfo2 block_texel_view_info {
	    .format = vk::Format::eBc1RgbaUnormBlock,
	    .type = vk::ImageType::e2D,
	    .tiling = vk::ImageTiling::eOptimal,
	    .usage = vk::ImageUsageFlagBits::eSampled,
	    .flags = vk::ImageCreateFlagBits::eBlockTexelViewCompatible,
	};
	const auto block_texel_view_props =
	    graphic_ctx.physical_device.getImageFormatProperties2(block_texel_view_info);
	graphic_ctx.supports_block_texel_view = block_texel_view_props.result == vk::Result::eSuccess;
	LOGF("Block Texel View support: %s\n", graphic_ctx.supports_block_texel_view ? "Yes" : "No");

	{
		auto available_extensions = EnumerateVulkan<vk::ExtensionProperties>(
		    "vkEnumerateDeviceExtensionProperties",
		    [&](uint32_t* count, vk::ExtensionProperties* values) {
			    return graphic_ctx.physical_device.enumerateDeviceExtensionProperties(
			        nullptr, count, values);
		    });

		if (HasExtension(available_extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
			device_extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
			graphic_ctx.memory_budget_ext_enabled = true;
		}
		for (const auto* extension: {VK_KHR_MAINTENANCE_8_EXTENSION_NAME,
		                             VK_KHR_MAINTENANCE_6_EXTENSION_NAME,
		                             VK_KHR_SHADER_CLOCK_EXTENSION_NAME,
		                             VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME,
		                             VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME,
		                             VK_EXT_MESH_SHADER_EXTENSION_NAME,
		                             VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME,
		                             VK_EXT_DEPTH_RANGE_UNRESTRICTED_EXTENSION_NAME,
		                             VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME,
		                             VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME,
		                             VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,
		                             VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME}) {
			if (HasExtension(available_extensions, extension)) {
				device_extensions.push_back(extension);
			}
		}
		// KYTY_GPL=1: graphics pipelines from libraries (pipelineCache.cpp); off by default until proven.
		if (const char* gpl = std::getenv("KYTY_GPL");
		    gpl != nullptr && gpl[0] != '0' && HasExtension(available_extensions, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) &&
		    HasExtension(available_extensions, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME)) {
			device_extensions.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
			device_extensions.push_back(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
		}
		if (HasExtension(available_extensions, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME) &&
		    HasExtension(available_extensions, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME)) {
			device_extensions.push_back(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME);
			device_extensions.push_back(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME);
		}
		// Ray queries take all three extensions or none (--native-rt).
		if (Config::NativeRtEnabled() &&
		    HasExtension(available_extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) &&
		    HasExtension(available_extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
		    HasExtension(available_extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME)) {
			device_extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
			device_extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
			device_extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
		}
		// FSR's ML upscalers (AMD's DLLs, --fsr-folder) run on cooperative matrices.
		if (Fsr::Enabled()) {
			for (const auto* extension: {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME}) {
				if (HasExtension(available_extensions, extension)) {
					device_extensions.push_back(extension);
				}
			}
		}
		Vr::OpenXrAppendExtensions(device_extensions, available_extensions, true);
		dlss_extensions_from = device_extensions.size();
		Dlss::AppendExtensions(device_extensions, available_extensions, true);
	}

	const auto create_device = [&] {
		return KYTY_UPSTREAM_PATH("8810d3869") ? VulkanCreateDeviceChains(graphic_ctx, surface, device_extensions)
		                                       : VulkanCreateDevice(graphic_ctx, surface, device_extensions);
	};
	LibKernel::Memory::ReleaseDriverAddressRoom(); // NGX's CUDA side reserves its arena there as the device is made
	graphic_ctx.device = create_device();
	if (graphic_ctx.device == nullptr && device_extensions.size() > dlss_extensions_from) {
		// NVIDIA's driver has refused a valid device with NGX's extensions inside the emulator (2026-11-11, 5090).
		printf("DLSS: the Vulkan device refused NGX's extensions; DLSS is off, PSSR upscales linearly\n");
		device_extensions.resize(dlss_extensions_from);
		Dlss::Shutdown();
		graphic_ctx.device = create_device();
	}
	if (graphic_ctx.device == nullptr) {
		EXIT("Could not create device");
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(graphic_ctx.device);
	graphic_ctx.device.getQueue(graphic_ctx.queue_family, 0, &graphic_ctx.queue);
	EXIT_IF(graphic_ctx.queue == nullptr);
	if (graphic_ctx.sparse_queue_index != 0u) {
		graphic_ctx.device.getQueue(graphic_ctx.queue_family, graphic_ctx.sparse_queue_index, &graphic_ctx.sparse_queue);
	}
	Dlss::Init(graphic_ctx, VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
	           VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr);
	Fsr3::FrameGen::Start(graphic_ctx);
	Fsr::Init(graphic_ctx, VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr);

	if (!graphic_ctx.CreateAllocator()) {
		EXIT("Could not create Vulkan memory allocator");
	}

	render_context = std::make_unique<RenderContext>(graphic_ctx);
	LibKernel::Memory::InstallGpuResources(render_context.get());
	presenter = std::make_unique<Presenter>(*this);
	Vr::OpenXrStartSession(graphic_ctx);
}

void WindowContext::RefreshSurfaceCapabilities() {
	EXIT_IF(graphic_ctx.physical_device == nullptr || surface == nullptr);
	GetSurfaceCapabilities(graphic_ctx.physical_device, surface, surface_capabilities);
}

void WindowContext::RecreateSurface() {
	EXIT_IF(window == nullptr || graphic_ctx.instance == nullptr);
	if (surface != nullptr) {
		graphic_ctx.instance.destroySurfaceKHR(surface, nullptr);
		surface = nullptr;
	}
	vk::SurfaceKHR::CType native_surface = VK_NULL_HANDLE;
	if (!SDL_Vulkan_CreateSurface(window, static_cast<vk::Instance::CType>(graphic_ctx.instance),
	                              nullptr, &native_surface)) {
		EXIT("Could not recreate the Vulkan surface: %s\n", SDL_GetError());
	}
	surface = native_surface;
}

WindowContext::~WindowContext() {
	Vr::OpenXrShutdown();
	ShutdownSystemOverlayInput();
	presenter.reset();
	LibKernel::Memory::InstallGpuResources(nullptr);
	render_context.reset();
	Dlss::Shutdown();
	Fsr::Shutdown();

	if (graphic_ctx.device != nullptr) {
		RequireVulkanSuccess(graphic_ctx.device.waitIdle(), "wait for Vulkan device shutdown");
		graphic_ctx.DestroyAllocator();
		graphic_ctx.device.destroy(nullptr);
		graphic_ctx.device = nullptr;
		graphic_ctx.queue  = nullptr;
	}
	if (surface != nullptr) {
		graphic_ctx.instance.destroySurfaceKHR(surface, nullptr);
		surface = nullptr;
	}
	if (graphic_ctx.debug_messenger != nullptr) {
		graphic_ctx.instance.destroyDebugUtilsMessengerEXT(graphic_ctx.debug_messenger, nullptr);
		graphic_ctx.debug_messenger = nullptr;
	}
	if (graphic_ctx.instance != nullptr) {
		graphic_ctx.instance.destroy(nullptr);
		graphic_ctx.instance        = nullptr;
		graphic_ctx.physical_device = nullptr;
	}
	if (window != nullptr) {
		SDL_DestroyWindow(window);
		window = nullptr;
	}
	SDL_QuitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD);
}

} // namespace Libs::Graphics
