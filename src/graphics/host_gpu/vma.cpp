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
#include "common/liveSwitches.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/addressBindingReport.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <iterator>

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
	// KYTY_VMA_BLOCK_MB=N (environment, default 64; 256 = VMA's own default): device memory
	// blocks of N MiB. A block returns to the driver only when it is empty; with Wolverine's
	// texture and buffer churn, 256 MiB blocks held 5.7 GiB of allocations in 8.8 GiB of blocks
	// while the device was over its budget (run 95: Windows paged 2-4 GB to system memory).
	// 64 MiB: 0.45-0.7 GiB trapped (run 96), old spot 7 -> 15-16 fps.
	uint64_t block_megabytes = 256; // 64 once the deleted-image read (PLAN P3) is fixed
	if (const char* value = std::getenv("KYTY_VMA_BLOCK_MB"); value != nullptr) {
		block_megabytes = std::strtoull(value, nullptr, 10);
	}
	if (block_megabytes >= 4 && block_megabytes <= 1024) {
		info.preferredLargeHeapBlockSize = static_cast<VkDeviceSize>(block_megabytes) << 20u;
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
	TrimImagePool(0, true);
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

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
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

uint64_t GraphicContext::GetDriverMemoryBudget() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		if (!discrete || (physical_device_memory_properties.memoryHeaps[heap].flags &
		                  vk::MemoryHeapFlagBits::eDeviceLocal)) {
			budget += budgets[heap].budget;
		}
	}
	return budget;
}

void GraphicContext::GetDeviceAllocationStats(uint64_t& allocation_bytes,
                                              uint64_t& block_bytes) const {
	allocation_bytes = 0;
	block_bytes      = 0;
	if (allocator == nullptr) {
		return;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		if (physical_device_memory_properties.memoryHeaps[heap].flags &
		    vk::MemoryHeapFlagBits::eDeviceLocal) {
			allocation_bytes += budgets[heap].statistics.allocationBytes;
			block_bytes += budgets[heap].statistics.blockBytes;
		}
	}
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

// Research: KYTY_IMAGE_RECYCLE_MB=N (live, 0 = off). Wolverine reuses the same memory for
// differently sized and formatted render targets within a frame, so the texture cache deletes
// and recreates ~50 images per frame (~100/s, ~500 MB/s of driver allocations, kernel calls and
// dedicated-memory frees on Thread_Gpu). A deleted image (only after the GPU is done with it:
// the texture cache defers the erase to its tick) is parked, up to N MiB, and an identical
// CreateImage takes it back. It starts in UNDEFINED layout like a new image, so its old
// contents are never observed. Parked images unused for 2 s are destroyed.
static int64_t ImageRecycleMegabytes() {
	// Wolverine run 16: 256 MiB reuses ~450 of ~451 image creations per 5 s with ~20 MiB parked;
	// frame time unchanged there (116.8 ms either way), allocations and kernel calls gone.
	static auto& megabytes = Common::LiveSwitches::Get("KYTY_IMAGE_RECYCLE_MB", 256);
	return megabytes.load(std::memory_order_relaxed);
}

decltype(GraphicContext::PooledImage::key)
GraphicContext::PoolKey(const vk::ImageCreateInfo& info) {
	return {info.flags,         info.imageType,    info.format,    info.extent.width,
	        info.extent.height, info.extent.depth, info.mipLevels, info.arrayLayers,
	        info.samples,       info.tiling,       info.usage,     info.sharingMode};
}

void GraphicContext::TrimImagePool(uint64_t cap_bytes, bool all) {
	constexpr auto MaxAge = std::chrono::seconds(2);
	// KYTY_IMAGE_QUARANTINE_MS=N (live, default 0, workaround): a parked image keeps its memory at
	// least N ms whatever the cap. Wolverine reads some deleted images 2-12 s later through a
	// descriptor not yet found (runs 88-95); inside its old memory that read is harmless, after
	// the memory is released (dedicated allocations) it loses the device.
	static auto&     quarantine_ms = Common::LiveSwitches::Get("KYTY_IMAGE_QUARANTINE_MS", 0);
	const auto       quarantine =
	    std::chrono::milliseconds(std::max<int64_t>(0, quarantine_ms.load(std::memory_order_relaxed)));
	const auto       now    = std::chrono::steady_clock::now();
	std::scoped_lock lock(m_image_pool_mutex);
	// Oldest first: entries are appended as they are parked.
	size_t keep_from = 0;
	while (keep_from < m_image_pool.size() &&
	       (all || ((m_image_pool_bytes > cap_bytes ||
	                 now - m_image_pool[keep_from].parked > MaxAge) &&
	                now - m_image_pool[keep_from].parked >= quarantine))) {
		auto& entry = m_image_pool[keep_from];
		if (address_binding_report_enabled) {
			AddressBindingAnnotate(reinterpret_cast<uint64_t>(static_cast<VkImage>(entry.image)),
			                       "destroyed by the pool trim");
		}
		vmaDestroyImage(allocator, entry.image, entry.allocation);
		m_image_pool_bytes -= entry.bytes;
		keep_from++;
	}
	m_image_pool.erase(m_image_pool.begin(),
	                   m_image_pool.begin() + static_cast<std::ptrdiff_t>(keep_from));
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	vk::Image::CType native_image = VK_NULL_HANDLE;
	auto             result       = vk::Result::eErrorUnknown;
	bool             from_pool    = false;
	if (image_info.pNext == nullptr && image_info.initialLayout == vk::ImageLayout::eUndefined &&
	    ImageRecycleMegabytes() > 0) {
		const auto       key = PoolKey(image_info);
		std::scoped_lock lock(m_image_pool_mutex);
		// Newest first, so the entries kept warm are the ones reused.
		for (auto it = m_image_pool.rbegin(); it != m_image_pool.rend(); ++it) {
			if (it->key == key) {
				native_image     = it->image;
				image.allocation = it->allocation;
				if (address_binding_report_enabled) {
					AddressBindingAnnotate(reinterpret_cast<uint64_t>(native_image),
					                       "taken from the image pool");
				}
				m_image_pool_bytes -= it->bytes;
				m_image_pool.erase(std::next(it).base());
				result    = vk::Result::eSuccess;
				from_pool = true;
				break;
			}
		}
	}

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	// KYTY_IMAGE_DEDICATED_KB (live, default 0 = off): images at least this large get their own
	// device memory. Sharing VMA blocks, freed images left holes the driver still counts:
	// Wolverine run 52 held 7.9 GiB of allocations in 9.9 GiB of blocks, and turning the camera
	// paged up to 2.5 GiB out to system memory (run 53). 1024 fixed that (run 54) but turned a
	// stale bindless texture read, harmless inside a live block, into device losts (runs 54-58)
	// while KYTY_BINDLESS_EVICT was on; off until that read is fixed.
	static auto& dedicated_kb = Common::LiveSwitches::Get("KYTY_IMAGE_DEDICATED_KB", 0);
	if (const auto kb = dedicated_kb.load(std::memory_order_relaxed); kb > 0) {
		// Estimated from the texel count at 4 bytes per texel (formats and mips vary).
		const auto bytes = static_cast<uint64_t>(image_info.extent.width) *
		                   image_info.extent.height * image_info.extent.depth *
		                   image_info.arrayLayers * 4u;
		if (bytes >= static_cast<uint64_t>(kb) * 1024u) {
			alloc_info.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
		}
	}

	if (ImageRecycleMegabytes() > 0) {
		static std::atomic<uint64_t> reused {0};
		static std::atomic<uint64_t> allocated {0};
		static std::atomic<int64_t>  report_ms {0};
		(result == vk::Result::eSuccess ? reused : allocated)++;
		const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		                        std::chrono::steady_clock::now().time_since_epoch())
		                        .count();
		if (auto last = report_ms.load();
		    now_ms - last >= 5000 && report_ms.compare_exchange_strong(last, now_ms)) {
			std::scoped_lock lock(m_image_pool_mutex);
			::printf("Image pool (5 s): reused %" PRIu64 ", allocated %" PRIu64
			         ", parked %zu (%.1f MiB)\n",
			         reused.exchange(0), allocated.exchange(0), m_image_pool.size(),
			         static_cast<double>(m_image_pool_bytes) / 1048576.0);
			std::fflush(stdout);
		}
	}
	if (result != vk::Result::eSuccess) {
		result = static_cast<vk::Result>(vmaCreateImage(
		    allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info), &alloc_info,
		    &native_image, &image.allocation, nullptr));
	}
	if (result != vk::Result::eSuccess) {
		alloc_info.requiredFlags  = 0;
		alloc_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		result                    = static_cast<vk::Result>(vmaCreateImage(
            allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
            &alloc_info, &native_image, &image.allocation, nullptr));
		static std::atomic<uint32_t> spill_count {0};
		if (const auto seen = spill_count.fetch_add(1, std::memory_order_relaxed); seen < 32) {
			LOGF("Image spilled to host memory (%u): %ux%ux%u layers=%u levels=%u format=%d -> %s\n",
			     seen + 1, image_info.extent.width, image_info.extent.height,
			     image_info.extent.depth, image_info.arrayLayers, image_info.mipLevels,
			     static_cast<int>(image_info.format), vk::to_string(result).c_str());
		}
	}
	image.image = native_image;
	if (result != vk::Result::eSuccess) {
		LogMemoryBudget();
		return false;
	}
	if (!from_pool && address_binding_report_enabled) {
		AddressBindingAnnotate(reinterpret_cast<uint64_t>(native_image), "created");
	}

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.tiling     = image_info.tiling;
	image.sharing    = image_info.sharingMode;
	image.recyclable =
	    image_info.pNext == nullptr && image_info.initialLayout == vk::ImageLayout::eUndefined;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	const auto megabytes = ImageRecycleMegabytes();
	const auto cap       = static_cast<uint64_t>(std::max<int64_t>(megabytes, 0)) << 20u;
	static auto& quarantine_ms = Common::LiveSwitches::Get("KYTY_IMAGE_QUARANTINE_MS", 0);
	const bool   quarantine    = quarantine_ms.load(std::memory_order_relaxed) > 0;
	const bool   reusable      = megabytes > 0 && image.recyclable;
	if (reusable || quarantine) {
		vk::ImageCreateInfo info {};
		info.flags       = image.flags;
		info.imageType   = image.image_type;
		info.format      = image.format;
		info.extent      = image.extent;
		info.mipLevels   = image.mip_levels;
		info.arrayLayers = image.layers;
		info.samples     = static_cast<vk::SampleCountFlagBits>(image.samples);
		info.tiling      = image.tiling;
		info.usage       = image.usage;
		info.sharingMode = image.sharing;
		if (!reusable) {
			// Only held for the quarantine: a key no CreateImage asks for.
			info.imageType = static_cast<vk::ImageType>(0x7fffffff);
		}
		VmaAllocationInfo allocation {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation);
		{
			std::scoped_lock lock(m_image_pool_mutex);
			m_image_pool.push_back({PoolKey(info), image.image, image.allocation, allocation.size,
			                        std::chrono::steady_clock::now()});
			if (address_binding_report_enabled) {
				AddressBindingAnnotate(reinterpret_cast<uint64_t>(static_cast<VkImage>(image.image)),
				                       reusable ? "parked in the image pool" : "held (quarantine)");
			}
			m_image_pool_bytes += allocation.size;
		}
		image.image      = nullptr;
		image.allocation = nullptr;
	} else {
		if (address_binding_report_enabled) {
			AddressBindingAnnotate(reinterpret_cast<uint64_t>(static_cast<VkImage>(image.image)),
			                       "destroyed directly");
		}
		vmaDestroyImage(allocator, image.image, image.allocation);
		image.image      = nullptr;
		image.allocation = nullptr;
	}
	TrimImagePool(cap, megabytes <= 0 && !quarantine);
}

} // namespace Libs::Graphics
