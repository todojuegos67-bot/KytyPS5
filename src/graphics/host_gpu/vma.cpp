#include <atomic>
#include <cstdio>
#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include "graphics/host_gpu/vramStats.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <string>

#include <fmt/format.h>

namespace Libs::Graphics {

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

GraphicContext::VramSample GraphicContext::SampleVram() const {
	VramSample sample {};
	if (!memory_budget_ext_enabled || allocator == nullptr || physical_device == nullptr) {
		return sample;
	}
	const auto& properties = physical_device_memory_properties;
	uint32_t    heap       = 0;
	while (heap < properties.memoryHeapCount &&
	       !static_cast<bool>(properties.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
		heap++;
	}
	if (heap >= properties.memoryHeapCount) {
		return sample;
	}
	auto query = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties2;
	if (query == nullptr) {
		query = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties2KHR;
	}
	if (query == nullptr) {
		return sample;
	}
	VkPhysicalDeviceMemoryBudgetPropertiesEXT budget {};
	budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
	VkPhysicalDeviceMemoryProperties2 props2 {};
	props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
	props2.pNext = &budget;
	query(static_cast<VkPhysicalDevice>(physical_device), &props2);
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	sample.valid       = budget.heapBudget[heap] != 0;
	sample.usage       = budget.heapUsage[heap];
	sample.budget      = budget.heapBudget[heap];
	sample.vma_alloc   = budgets[heap].statistics.allocationBytes;
	sample.vma_blocks  = budgets[heap].statistics.blockBytes;
	sample.driver_side = sample.usage > sample.vma_blocks ? sample.usage - sample.vma_blocks : 0;
	return sample;
}

bool GraphicContext::EnsureVramHeadroom(uint64_t bytes, const char* what) {
	if (!memory_budget_ext_enabled) return true;
	if (WithinVramBudget(bytes)) return true;
	if (m_vram_reclaim) m_vram_reclaim(bytes, false);
	if (WithinVramBudget(bytes)) return true;
	if (m_vram_reclaim) m_vram_reclaim(bytes, true);
	const bool ok = WithinVramBudget(bytes);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 24) {
		const auto s = SampleVram();
		std::printf("VRAM headroom: %s needs ~%.0f MiB, heap usage %.0f/%.0f MiB (driver-side %.0f MiB, vma blocks %.0f MiB) "
		            "after reclaim: %s\n",
		            what, bytes / 1048576.0, s.usage / 1048576.0, s.budget / 1048576.0, s.driver_side / 1048576.0,
		            s.vma_blocks / 1048576.0, ok ? "fits" : "still over the budget");
		std::fflush(stdout);
	}
	return ok;
}

// VRAM (5 s): driver heap usage/budget plus what the emulator holds per category, so a leak or
// over-allocation shows which cache grows. Called from the texture cache GC (Thread_Gpu).
void GraphicContext::ReportVram(std::string_view extra) const {
	if (allocator == nullptr) return;
	constexpr double MiB = 1024.0 * 1024.0;
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	std::string line = "VRAM (5 s):";
	if (const auto fresh = SampleVram(); fresh.valid) {
		line += fmt::format(" fresh heap0 {:.0f}/{:.0f} MiB = vma blocks {:.0f} (used {:.0f}) + driver-side {:.0f} |",
		                    fresh.usage / MiB, fresh.budget / MiB, fresh.vma_blocks / MiB, fresh.vma_alloc / MiB,
		                    fresh.driver_side / MiB);
	}
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool local = static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                                     vk::MemoryHeapFlagBits::eDeviceLocal);
		line += fmt::format(" heap{}{} {:.0f}/{:.0f} MiB (vma {:.0f} MiB in {} allocs)", heap, local ? "[vram]" : "",
		                    budgets[heap].usage / MiB, budgets[heap].budget / MiB,
		                    budgets[heap].statistics.allocationBytes / MiB,
		                    budgets[heap].statistics.allocationCount);
	}
	line += " |";
	for (int c = 0; c < VramStats::Count; c++) {
		line += fmt::format(" {} {:.0f} MiB/{} live (+{} -{})", VramStats::kNames[c],
		                    VramStats::g_bytes[c].load(std::memory_order_relaxed) / MiB,
		                    VramStats::g_count[c].load(std::memory_order_relaxed),
		                    VramStats::g_created[c].exchange(0, std::memory_order_relaxed),
		                    VramStats::g_destroyed[c].exchange(0, std::memory_order_relaxed));
	}
	line += fmt::format(" | {} | images outside VRAM so far {}{}", extra,
	                    VramStats::g_images_outside_vram.load(std::memory_order_relaxed),
	                    memory_budget_ext_enabled ? "" : " (no VK_EXT_memory_budget: usage is VMA's own)");
	std::printf("%s\n", line.c_str());
	std::fflush(stdout);
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	if (const auto fresh = SampleVram(); fresh.valid) {
		return fresh.usage;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint32_t GraphicContext::GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto count = physical_device_memory_properties.memoryHeapCount;
	for (uint32_t heap = 0; heap < count; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		heaps[heap] = {.usage        = budgets[heap].usage,
		               .budget       = budgets[heap].budget,
		               .allocations  = budgets[heap].statistics.allocationCount,
		               .device_local = static_cast<bool>(properties.flags &
		                                                 vk::MemoryHeapFlagBits::eDeviceLocal)};
	}
	return count;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

// True when a device-local allocation of `bytes` keeps the first device-local heap (heap0 on the
// target) at or below its VK_EXT_memory_budget budget minus a 512 MiB safety margin.
bool GraphicContext::WithinVramBudget(uint64_t bytes) const {
	constexpr uint64_t kSafetyMargin = 512ull * 1024 * 1024;
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return true;
	}
	const auto& properties = physical_device_memory_properties;
	uint32_t    heap       = 0;
	while (heap < properties.memoryHeapCount &&
	       !static_cast<bool>(properties.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
		heap++;
	}
	if (heap >= properties.memoryHeapCount) {
		return true;
	}
	// Fresh reading: the VMA cache can be ~30 allocation operations behind, and the driver's own
	// allocations (pipelines) never pass through VMA at all.
	if (const auto fresh = SampleVram(); fresh.valid) {
		const uint64_t margin = fresh.budget > kSafetyMargin ? kSafetyMargin : 0;
		return fresh.usage + bytes + margin <= fresh.budget;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const uint64_t budget = budgets[heap].budget;
	const uint64_t margin = budget > kSafetyMargin ? kSafetyMargin : 0;
	return budgets[heap].usage + bytes + margin <= budget;
}

// Memory the driver will need for an image, from its create info (no image object needed).
// Returns 0 when the entry point is unavailable, which disables the gate for that image.
uint64_t GraphicContext::ImageAllocationSize(const vk::ImageCreateInfo& image_info) const {
	const auto query = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceImageMemoryRequirements;
	if (query == nullptr || device == nullptr) {
		return 0;
	}
	VkDeviceImageMemoryRequirements info {};
	info.sType       = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS;
	info.pCreateInfo = static_cast<const vk::ImageCreateInfo::NativeType*>(image_info);
	VkMemoryRequirements2 reqs {};
	reqs.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
	query(static_cast<VkDevice>(device), &info, &reqs);
	return reqs.memoryRequirements.size;
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	// Budget gate, before the driver is asked for device-local memory it does not have. Idle
	// images are freed first; then the handler waits for submitted GPU work and frees again. If
	// the image still does not fit, it is placed in a non-device-local memory type (system RAM)
	// rather than overflowing VRAM, which the driver otherwise answers with device loss.
	const auto size = ImageAllocationSize(image_info);
	if (size != 0 && !WithinVramBudget(size)) {
		if (m_vram_reclaim) m_vram_reclaim(size, false);
		if (!WithinVramBudget(size) && m_vram_reclaim) m_vram_reclaim(size, true);
		if (!WithinVramBudget(size)) {
			uint32_t host_types = 0;
			for (uint32_t t = 0; t < physical_device_memory_properties.memoryTypeCount; t++) {
				if (!static_cast<bool>(physical_device_memory_properties.memoryTypes[t].propertyFlags &
				                       vk::MemoryPropertyFlagBits::eDeviceLocal)) {
					host_types |= 1u << t;
				}
			}
			if (host_types != 0) {
				alloc_info.requiredFlags = 0;
				alloc_info.memoryTypeBits = host_types;
				static std::atomic<uint32_t> logged_gate {0};
				if (logged_gate.fetch_add(1, std::memory_order_relaxed) < 4) {
					VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
					vmaGetHeapBudgets(allocator, budgets);
					std::printf("VRAM budget: image %ux%u needs %.0f MiB, heap usage %.0f/%.0f MiB "
					            "after reclaim and wait; placed outside VRAM\n",
					            image_info.extent.width, image_info.extent.height,
					            size / 1048576.0, budgets[0].usage / 1048576.0,
					            budgets[0].budget / 1048576.0);
				}
			}
		}
	}

	vk::Image::CType native_image = VK_NULL_HANDLE;
	auto              result       = static_cast<vk::Result>(
	    vmaCreateImage(allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
	                   &alloc_info, &native_image, &image.allocation, nullptr));
	if (result != vk::Result::eSuccess) {
		// Video memory is full (Ghost of Yotei streams past 16 GB): place the image in any memory
		// type the driver accepts, usually system RAM. Slower to sample, but the game goes on.
		LogMemoryBudget();
		alloc_info.requiredFlags  = 0;
		alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		native_image              = VK_NULL_HANDLE;
		image.allocation          = nullptr;
		result = static_cast<vk::Result>(vmaCreateImage(
		    allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info), &alloc_info,
		    &native_image, &image.allocation, nullptr));
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			std::printf("VMA: video memory full, image %ux%u placed outside VRAM: %s\n",
			            image_info.extent.width, image_info.extent.height,
			            result == vk::Result::eSuccess ? "ok" : "failed");
		}
	}
	image.image = native_image;
	if (result != vk::Result::eSuccess) {
		return false;
	}
	{
		VmaAllocationInfo info {};
		vmaGetAllocationInfo(allocator, image.allocation, &info);
		VramStats::Add(VramStats::Images, info.size);
		if (alloc_info.requiredFlags == 0) VramStats::g_images_outside_vram.fetch_add(1, std::memory_order_relaxed);
	}

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	{
		VmaAllocationInfo info {};
		vmaGetAllocationInfo(allocator, image.allocation, &info);
		VramStats::Remove(VramStats::Images, info.size);
	}
	vmaDestroyImage(allocator, image.image, image.allocation);
	image.image      = nullptr;
	image.allocation = nullptr;
}

} // namespace Libs::Graphics
