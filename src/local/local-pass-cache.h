#pragma once
// The local passes' pipelines (RT reflections and HWRT particles): one VkPipelineCache
// kept on disk (_PipelineCache/local/rt-passes.bin), and the mutex their Ensure* functions take, so that the prewarm
// thread (PrewarmLocalPasses, started at the first upscale: the title screen) and the flip path never make one twice.
// (10-27: with KYTY_HWRT_REFL the first world frame waited 4.7-5.4 s on the flip while these were made there.)
// The cache is the driver's to validate: data of another driver or GPU is ignored by vkCreatePipelineCache.

#include "local/frame-gen.h" // (vulkan.hpp)

#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <string>
#include <mutex>
#include <vector>

namespace Libs::Graphics::LocalPassCache {

inline std::recursive_mutex& EnsureMutex() {
	static std::recursive_mutex mutex;
	return mutex;
}

inline const char* Path() { return "_PipelineCache/local/rt-passes.bin"; }

inline std::atomic<VkPipelineCache> g_cache {VK_NULL_HANDLE};
inline std::atomic<VkDevice>        g_device {VK_NULL_HANDLE};

// (ds-perf-1031) KYTY_PASS_DISK_CACHE=0: the local passes other than HWRT particles make their pipelines with no cache
// (as before ds-perf-1031); default: every local compute pass (RT reflections, torch, godrays, SSGI, DLSS NR, sun
// shadows, ...) through this disk cache, so their first use in a later session takes the driver's cache hit, not a
// compile on the flip (10-31 log: 206-1465 ms each on first use).
inline bool DiskCacheOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_PASS_DISK_CACHE");
		return !(value != nullptr && value[0] == '0');
	}();
	return on;
}

inline vk::PipelineCache Get(vk::Device device) {
	static std::once_flag once;
	std::call_once(once, [device] {
		std::printf("Pass disk cache: %s (KYTY_PASS_DISK_CACHE=0: off)\n", DiskCacheOn() ? "on" : "off");
		std::vector<char> data;
		if (FILE* f = std::fopen(Path(), "rb")) {
			std::fseek(f, 0, SEEK_END);
			const long size = std::ftell(f);
			std::fseek(f, 0, SEEK_SET);
			if (size > 0 && size < (256l << 20)) {
				data.resize(static_cast<size_t>(size));
				if (std::fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
			}
			std::fclose(f);
		}
		vk::PipelineCacheCreateInfo info {};
		info.initialDataSize = data.size();
		info.pInitialData    = data.empty() ? nullptr : data.data();
		vk::PipelineCache cache = nullptr;
		if (device.createPipelineCache(&info, nullptr, &cache) != vk::Result::eSuccess) {
			info.initialDataSize = 0, info.pInitialData = nullptr; // (the data refused: an empty cache)
			if (device.createPipelineCache(&info, nullptr, &cache) != vk::Result::eSuccess) cache = nullptr;
		}
		std::printf("RT passes: pipeline cache %s (%zu bytes read)\n", cache ? Path() : "none", data.size());
		std::fflush(stdout);
		g_device.store(static_cast<VkDevice>(device));
		g_cache.store(static_cast<VkPipelineCache>(cache));
	});
	return DiskCacheOn() ? vk::PipelineCache(g_cache.load()) : vk::PipelineCache(nullptr);
}

// Written after the prewarm and after each pipeline made on the flip path (a few a session).
inline void Save() {
	const VkPipelineCache cache  = g_cache.load();
	const VkDevice        device = g_device.load();
	if (cache == VK_NULL_HANDLE || device == VK_NULL_HANDLE) return;
	static std::mutex     mutex;
	std::lock_guard       lock(mutex);
	size_t size = 0;
	const vk::Device        dev(device);
	const vk::PipelineCache vcache(cache);
	if (dev.getPipelineCacheData(vcache, &size, nullptr) != vk::Result::eSuccess || size == 0) return;
	std::vector<char> data(size);
	if (dev.getPipelineCacheData(vcache, &size, data.data()) != vk::Result::eSuccess) return;
	std::error_code error;
	std::filesystem::create_directories(std::filesystem::path(Path()).parent_path(), error);
	const std::string temp = std::string(Path()) + ".tmp";
	if (FILE* f = std::fopen(temp.c_str(), "wb")) {
		const bool ok = std::fwrite(data.data(), 1, size, f) == size;
		std::fclose(f);
		if (ok) std::filesystem::rename(temp, Path(), error);
	}
}

} // namespace Libs::Graphics::LocalPassCache
