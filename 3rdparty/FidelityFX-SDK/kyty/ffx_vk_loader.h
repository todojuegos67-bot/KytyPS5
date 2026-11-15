// Kyty: force-included into the SDK's ffx_vk.cpp. The backend calls these few Vulkan functions
// directly; Kyty links no Vulkan import library (it loads Vulkan at run time), so they are renamed
// here and defined in src/graphics/host_gpu/renderer/fsr3.cpp, through Kyty's dispatcher.
#pragma once

#define vkEnumerateDeviceExtensionProperties kyty_ffx_vkEnumerateDeviceExtensionProperties
#define vkGetPhysicalDeviceFeatures          kyty_ffx_vkGetPhysicalDeviceFeatures
#define vkGetPhysicalDeviceFeatures2         kyty_ffx_vkGetPhysicalDeviceFeatures2
#define vkGetPhysicalDeviceMemoryProperties  kyty_ffx_vkGetPhysicalDeviceMemoryProperties
#define vkGetPhysicalDeviceProperties        kyty_ffx_vkGetPhysicalDeviceProperties
#define vkGetPhysicalDeviceProperties2       kyty_ffx_vkGetPhysicalDeviceProperties2
#define vkCreateBuffer                       kyty_ffx_vkCreateBuffer
#define vkGetDeviceProcAddr                  kyty_ffx_vkGetDeviceProcAddr
