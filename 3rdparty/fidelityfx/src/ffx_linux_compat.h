// Build support for the FidelityFX SDK sources in the emulator: on Linux the MSVC library extensions they use; on Linux
// and Windows their few global Vulkan calls routed to the emulator's dispatcher (below).
#pragma once

#if !defined(_WIN32)

#include <algorithm>
#include <bit>
#include <cmath>
#include <codecvt>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <locale>
#include <string>

#ifndef _countof
#define _countof(array) (sizeof(array) / sizeof((array)[0]))
#endif

template <size_t N>
inline int wcscpy_s(wchar_t (&destination)[N], const wchar_t* source) {
	std::wcsncpy(destination, source, N - 1);
	destination[N - 1] = L'\0';
	return 0;
}

inline int wcscpy_s(wchar_t* destination, size_t size, const wchar_t* source) {
	if (size == 0) return 1;
	std::wcsncpy(destination, source, size - 1);
	destination[size - 1] = L'\0';
	return 0;
}

template <size_t N>
inline int strcpy_s(char (&destination)[N], const char* source) {
	std::strncpy(destination, source, N - 1);
	destination[N - 1] = '\0';
	return 0;
}

inline int strcpy_s(char* destination, size_t size, const char* source) {
	if (size == 0) return 1;
	std::strncpy(destination, source, size - 1);
	destination[size - 1] = '\0';
	return 0;
}

template <size_t N>
inline int wcscat_s(wchar_t (&destination)[N], const wchar_t* source) {
	std::wcsncat(destination, source, N - std::wcslen(destination) - 1);
	return 0;
}

template <size_t N, typename... Args>
inline int swprintf_s(wchar_t (&buffer)[N], const wchar_t* format, Args... args) {
	return std::swprintf(buffer, N, format, args...);
}

template <size_t N, typename... Args>
inline int sprintf_s(char (&buffer)[N], const char* format, Args... args) {
	return std::snprintf(buffer, N, format, args...);
}

template <typename... Args>
inline int sprintf_s(char* buffer, size_t size, const char* format, Args... args) {
	return std::snprintf(buffer, size, format, args...);
}

#endif

// The emulator loads Vulkan at run time (SDL), so libvulkan's (vulkan-1.dll's) exported entry points are not linked.
// The few global calls in the Vulkan backend are routed to the emulator's dispatcher instead.
#include <vulkan/vulkan.h>
#ifdef __cplusplus
extern "C" {
#endif
void     kyty_ffx_vkGetPhysicalDeviceFeatures(VkPhysicalDevice device, VkPhysicalDeviceFeatures* features);
void     kyty_ffx_vkGetPhysicalDeviceFeatures2(VkPhysicalDevice device, VkPhysicalDeviceFeatures2* features);
void     kyty_ffx_vkGetPhysicalDeviceProperties(VkPhysicalDevice device, VkPhysicalDeviceProperties* properties);
void     kyty_ffx_vkGetPhysicalDeviceProperties2(VkPhysicalDevice device, VkPhysicalDeviceProperties2* properties);
void     kyty_ffx_vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice device, VkPhysicalDeviceMemoryProperties* properties);
VkResult kyty_ffx_vkEnumerateDeviceExtensionProperties(VkPhysicalDevice device, const char* layer, uint32_t* count, VkExtensionProperties* properties);
PFN_vkVoidFunction kyty_ffx_vkGetDeviceProcAddr(VkDevice device, const char* name);
#ifdef __cplusplus
}
#endif
#define vkGetPhysicalDeviceFeatures              kyty_ffx_vkGetPhysicalDeviceFeatures
#define vkGetPhysicalDeviceFeatures2             kyty_ffx_vkGetPhysicalDeviceFeatures2
#define vkGetPhysicalDeviceProperties            kyty_ffx_vkGetPhysicalDeviceProperties
#define vkGetPhysicalDeviceProperties2           kyty_ffx_vkGetPhysicalDeviceProperties2
#define vkGetPhysicalDeviceMemoryProperties      kyty_ffx_vkGetPhysicalDeviceMemoryProperties
#define vkEnumerateDeviceExtensionProperties     kyty_ffx_vkEnumerateDeviceExtensionProperties
#define vkGetDeviceProcAddr                      kyty_ffx_vkGetDeviceProcAddr
