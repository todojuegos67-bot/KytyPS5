#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "frame-gen.h"
#include "live-counters.h"
#include "local-platform.h"

#include <cstdio>

#include <algorithm>
#include <cinttypes>

namespace Libs::Graphics {

void FlushBufferReclaimer(); // streamBuffer.cpp (KYTY_BUFFER_RECLAIM)

// Local diagnostic (live `vma <path>`): VMA is thread-safe, so the live thread writes it.
static VmaAllocator          g_report_allocator = nullptr;
static const GraphicContext* g_report_context   = nullptr;
// Freed images kept for reuse (KYTY_IMAGE_POOL): at most 1 GiB, and a 16th of the GPU's memory budget.
static uint64_t g_image_pool_limit = 1024ull << 20;
static void WriteVmaReport(const char* path) {
	if (g_report_allocator == nullptr) return;
	if (g_report_context != nullptr) g_report_context->LogVideoMemory("live");
	char* json = nullptr;
	vmaBuildStatsString(g_report_allocator, &json, VK_TRUE);
	if (FILE* file = std::fopen(path, "wb"); file != nullptr) {
		std::fputs(json, file);
		std::fclose(file);
	}
	vmaFreeStatsString(g_report_allocator, json);
}

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
	// VMA's blocks: 64 MiB on a GPU with less than 13 GiB of video memory (KYTY_SIMULATE_VRAM_MB counts), VMA's 256 MiB
	// on bigger ones. Such a GPU runs past its budget and the driver keeps part of the game's memory in system memory: 1-1
	// tours with 8 GB, each order once: walk 70.0 vs 67.4 and 72.2 vs 69.6 fps, spin 68.3 vs 65.9 and 71.4 vs 69.1, the
	// route's later two thirds 2-6% faster in both (the free space inside the blocks no smaller, 787 vs 545 MiB at the
	// end: it is where the driver puts the smaller blocks). KYTY_VMA_BLOCK_MB=<n>: n MiB.
	{
		uint64_t video = 0;
		for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++)
			if (physical_device_memory_properties.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
				video = std::max<uint64_t>(video, physical_device_memory_properties.memoryHeaps[heap].size);
		if (const char* simulate = std::getenv("KYTY_SIMULATE_VRAM_MB"); simulate != nullptr && *simulate != '\0')
			video = std::min<uint64_t>(video, std::strtoull(simulate, nullptr, 10) << 20u);
		const char* text = std::getenv("KYTY_VMA_BLOCK_MB");
		if (text != nullptr && std::strtoull(text, nullptr, 10) > 0) {
			info.preferredLargeHeapBlockSize = std::strtoull(text, nullptr, 10) << 20u;
		} else if (video != 0 && video < (uint64_t {13} << 30u)) {
			info.preferredLargeHeapBlockSize = uint64_t {64} << 20u;
		}
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	// KYTY_SIMULATE_VRAM_MB=<n>: a GPU with n MiB of video memory, for tests on a bigger one. Unlike an allocator limit
	// (what VMA puts in system memory types the driver still places in video memory while there is room), the rest of
	// the largest device-local heap is taken here at the highest priority and kept, so the driver and OS move the game's
	// memory to system memory as on such a GPU. Budget and usage (GetTotalMemoryBudget, GetDeviceMemoryUsage) leave it
	// out.
	if (const char* simulate = std::getenv("KYTY_SIMULATE_VRAM_MB"); simulate != nullptr) {
		const uint64_t simulated = std::strtoull(simulate, nullptr, 10) << 20u;
		uint32_t       heap      = 0;
		for (uint32_t i = 1; i < physical_device_memory_properties.memoryHeapCount; i++) {
			if (physical_device_memory_properties.memoryHeaps[i].size > physical_device_memory_properties.memoryHeaps[heap].size &&
			    (physical_device_memory_properties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal))
				heap = i;
		}
		uint32_t type = UINT32_MAX;
		for (uint32_t i = 0; i < physical_device_memory_properties.memoryTypeCount && type == UINT32_MAX; i++) {
			const auto& candidate = physical_device_memory_properties.memoryTypes[i];
			if (candidate.heapIndex == heap && candidate.propertyFlags == vk::MemoryPropertyFlagBits::eDeviceLocal) type = i;
		}
		const uint64_t size = physical_device_memory_properties.memoryHeaps[heap].size;
		for (uint64_t left = simulated > 0 && simulated < size ? size - simulated : 0; left > 0 && type != UINT32_MAX;) {
			const uint64_t                    chunk = std::min<uint64_t>(left, uint64_t {1} << 30u);
			vk::MemoryPriorityAllocateInfoEXT priority {};
			priority.priority = 1.0f;
			vk::MemoryAllocateInfo info {};
			info.pNext           = memory_priority_enabled ? &priority : nullptr;
			info.allocationSize  = chunk;
			info.memoryTypeIndex = type;
			vk::DeviceMemory memory = nullptr;
			if (device.allocateMemory(&info, nullptr, &memory) != vk::Result::eSuccess) break;
			simulation_ballast.push_back(memory);
			simulation_ballast_bytes += chunk;
			left -= chunk;
		}
		std::printf("Simulated video memory: %" PRIu64 " MiB of heap %u (%" PRIu64 " MiB) taken, priorities %s\n",
		            simulation_ballast_bytes >> 20u, heap, size >> 20u, memory_priority_enabled ? "on" : "off");
	}
	{
		vk::PhysicalDeviceIDProperties id {};
		vk::PhysicalDeviceProperties2  device_properties {};
		device_properties.pNext = &id;
		physical_device.getProperties2(&device_properties);
		if (id.deviceLUIDValid) (void)LocalPlatform::OpenVideoMemoryAdapter(id.deviceLUID.data());
	}
	// Frame generation's and super resolution's own allocations (FSR's about 1.3 GiB at 2560x1440: interpolation, optical
	// flow, its swapchain; DLSS-G's alike; outside this allocator but in its budget) leave the game that much less, so a
	// budget short of 12 GiB after them counts as small (a 16 GB card whose budget other programs had cut to 12.2 GiB ran
	// out with FSR; issue #215: a 16 GB RTX 5080 with DLSS-G and DLSS ran out with the texture pools kept).
	const uint64_t reserved = Libs::Graphics::FrameGen::VideoMemoryReserve();
	small_video_memory      = GetTotalMemoryBudget() < (uint64_t {12} << 30u) + reserved;
	uint64_t video_memory   = 0; // (the largest device-local heap, less a simulation's ballast)
	for (uint32_t i = 0; i < physical_device_memory_properties.memoryHeapCount; i++)
		if (physical_device_memory_properties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
			video_memory = std::max<uint64_t>(video_memory, physical_device_memory_properties.memoryHeaps[i].size);
	reduce_texture_levels = video_memory - std::min(video_memory, simulation_ballast_bytes) <= (uint64_t {12} << 30u);
	if (reserved != 0) {
		std::printf("Video memory: %" PRIu64 " MiB of budget, %" PRIu64 " MiB of it left to frame generation and super "
		            "resolution: %s\n",
		            GetTotalMemoryBudget() >> 20u, reserved >> 20u,
		            small_video_memory ? "the game's use sized as on a card short of video memory (texture pools retire)"
		                               : "room enough for the game's");
	}
	// The game buffers' pool (graphicContext.h): VMA's device-local type for them, VMA's block sizes (dedicated memory
	// for big ones, as in the default pools).
	if (const char* defrag = std::getenv("KYTY_BUFFER_DEFRAG");
	    defrag == nullptr ? small_video_memory : std::strcmp(defrag, "0") != 0) {
		VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		buffer.size  = uint64_t {64} << 10u;
		buffer.usage = static_cast<VkBufferUsageFlags>(AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress);
		VmaAllocationCreateInfo allocation {};
		allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		uint32_t type    = 0;
		VmaPoolCreateInfo pool {};
		if (vmaFindMemoryTypeIndexForBufferInfo(allocator, &buffer, &allocation, &type) == VK_SUCCESS) {
			pool.memoryTypeIndex = type;
			if (vmaCreatePool(allocator, &pool, &game_buffer_pool) != VK_SUCCESS) game_buffer_pool = nullptr;
		}
		std::printf("Game buffers: %s\n", game_buffer_pool != nullptr ? "own pool, compacted (KYTY_BUFFER_DEFRAG=0 turns it off)"
		                                                               : "pool failed, default pools");
	}
	LogVideoMemory("start");
	g_report_allocator          = allocator;
	g_report_context            = this;
	LiveCounters::g_vma_report = WriteVmaReport;
	g_image_pool_limit         = std::min<uint64_t>(1024ull << 20, GetTotalMemoryBudget() / 16);
	return true;
}

static void DestroyImagePool(VmaAllocator allocator);

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	LiveCounters::g_vma_report = nullptr;
	g_report_allocator         = nullptr;
	g_report_context           = nullptr;
	FlushBufferReclaimer();
	DestroyImagePool(allocator);
	if (game_buffer_pool != nullptr) {
		// (Only empty: a buffer still alive frees its memory into it later.)
		VmaStatistics statistics {};
		vmaGetPoolStatistics(allocator, game_buffer_pool, &statistics);
		if (statistics.allocationCount == 0) vmaDestroyPool(allocator, game_buffer_pool);
		game_buffer_pool = nullptr;
	}
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
	for (const auto memory: simulation_ballast) device.freeMemory(memory);
	simulation_ballast.clear();
	simulation_ballast_bytes = 0;
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

void GraphicContext::RefreshMemoryBudget() {
	static std::atomic<uint32_t> frame {0};
	if (allocator != nullptr) vmaSetCurrentFrameIndex(allocator, frame.fetch_add(1, std::memory_order_relaxed) + 1);
}

bool GraphicContext::QueryLocalVideoMemory(uint64_t* usage, uint64_t* budget) const {
	if (!LocalPlatform::QueryVideoMemory(usage, budget)) return false;
	*usage -= std::min(*usage, simulation_ballast_bytes);
	*budget -= std::min(*budget, simulation_ballast_bytes);
	return true;
}

void GraphicContext::LogVideoMemory(const char* label) const {
	uint64_t video = 0, budget = 0, shared = 0;
	if (allocator == nullptr || !QueryLocalVideoMemory(&video, &budget) || !LocalPlatform::QuerySharedGpuMemory(&shared)) return;
	// VMA's memory by kind of memory type: device-local (video memory), the driver's system memory type without
	// flags (NVIDIA places it in video memory while there is room), host-visible.
	VmaTotalStatistics statistics {};
	vmaCalculateStatistics(allocator, &statistics);
	uint64_t local = 0, plain = 0, host = 0;
	const auto& properties = GetPhysicalDeviceMemoryProperties();
	for (uint32_t type = 0; type < properties.memoryTypeCount; type++) {
		const auto flags = properties.memoryTypes[type].propertyFlags;
		const auto bytes = statistics.memoryType[type].statistics.blockBytes;
		if (flags & vk::MemoryPropertyFlagBits::eDeviceLocal) local += bytes;
		else if (flags & vk::MemoryPropertyFlagBits::eHostVisible) host += bytes;
		else plain += bytes;
	}
	// What of the plain type the GPU's system memory does not hold is in video memory; the rest of the video memory in
	// use is the driver's own (pipelines, shader local memory, descriptors).
	const uint64_t plain_in_system = std::min(plain, shared - std::min(shared, host));
	const uint64_t plain_in_video  = plain - plain_in_system;
	const uint64_t driver          = video - std::min(video, local + plain_in_video);
	std::printf("Video memory (%s): %" PRIu64 " MiB in use of %" PRIu64 " MiB (VMA device-local %" PRIu64 " MiB, plain type %" PRIu64
	            " MiB in video memory, driver %" PRIu64 " MiB); GPU system memory %" PRIu64 " MiB (plain type %" PRIu64
	            " MiB, host-visible %" PRIu64 " MiB)\n",
	            label, video >> 20u, budget >> 20u, local >> 20u, plain_in_video >> 20u, driver >> 20u, shared >> 20u,
	            plain_in_system >> 20u, host >> 20u);
	std::fflush(stdout);
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
	return usage - std::min(usage, simulation_ballast_bytes);
}

uint64_t GraphicContext::GetVideoMemoryRoom() const {
	// (Without VK_EXT_memory_budget VMA counts only its own blocks against a guessed budget: no figure to go by.)
	if (allocator == nullptr || !CanReportMemoryUsage()) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto& heaps = physical_device_memory_properties.memoryHeaps;
	uint32_t    video = UINT32_MAX;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool larger = video == UINT32_MAX || heaps[heap].size > heaps[video].size;
		if ((heaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) && larger) video = heap;
	}
	if (video == UINT32_MAX) {
		return 0;
	}
	// (The driver's usage holds KYTY_SIMULATE_VRAM_MB's ballast: the room left is the same.)
	const auto& budget = budgets[video];
	return budget.budget > budget.usage ? budget.budget - budget.usage : 0;
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
	budget -= std::min(budget, simulation_ballast_bytes);
	usage -= std::min(usage, simulation_ballast_bytes);
	local -= std::min(local, simulation_ballast_bytes);
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

// Recycling of freed images (KYTY_IMAGE_POOL). The game aliases transient render
// targets in one heap, so the texture cache deletes and re-creates the same few
// images every frame. An image reaches DeleteImage only after the GPU is done
// with it; a new image always starts in eUndefined layout, so reusing the VkImage
// and its memory for identical creation parameters is equivalent to a fresh
// allocation.
extern "C" {
volatile std::atomic<uint32_t> kyty_local_image_pool_mode {1};
}
namespace {
struct PooledImage {
	std::array<uint32_t, 10> key {};
	VkImage                  image      = VK_NULL_HANDLE;
	VmaAllocation            allocation = nullptr;
	uint64_t                 bytes      = 0;
};
std::mutex               g_image_pool_mutex;
std::vector<PooledImage> g_image_pool;
uint64_t                 g_image_pool_bytes = 0;

std::array<uint32_t, 10> ImagePoolKey(const vk::ImageCreateInfo& info) {
	return {static_cast<uint32_t>(static_cast<VkImageCreateFlags>(info.flags)), static_cast<uint32_t>(info.imageType),
	        static_cast<uint32_t>(info.format), info.extent.width, info.extent.height, info.extent.depth,
	        info.mipLevels, info.arrayLayers,
	        static_cast<uint32_t>(info.samples) | (static_cast<uint32_t>(info.tiling) << 16u),
	        static_cast<uint32_t>(static_cast<VkImageUsageFlags>(info.usage))};
}
std::array<uint32_t, 10> ImagePoolKey(const VulkanImage& image) {
	vk::ImageCreateInfo info {};
	info.flags       = image.flags;
	info.imageType   = image.image_type;
	info.format      = image.format;
	info.extent      = image.extent;
	info.mipLevels   = image.mip_levels;
	info.arrayLayers = image.layers;
	info.samples     = static_cast<vk::SampleCountFlagBits>(image.samples);
	info.tiling      = vk::ImageTiling::eOptimal;
	info.usage       = image.usage;
	return ImagePoolKey(info);
}
} // namespace

static void DestroyImagePool(VmaAllocator allocator) {
	std::scoped_lock lock(g_image_pool_mutex);
	for (const auto& pooled: g_image_pool) {
		vmaDestroyImage(allocator, pooled.image, pooled.allocation);
	}
	g_image_pool.clear();
	g_image_pool_bytes = 0;
}

void GraphicContext::TrimImagePool() {
	if (allocator != nullptr) DestroyImagePool(allocator);
}

// Out of video memory: the first few times say so in the log, with the heaps' budgets.
void GraphicContext::ReportMemoryFallback(const char* what, uint64_t bytes) const {
	static std::atomic<uint32_t> reported {0};
	if (reported.fetch_add(1, std::memory_order_relaxed) >= 8) return;
	std::printf("Vulkan: video memory full, %s (%" PRIu64 " bytes)\n", what, bytes);
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < GetPhysicalDeviceMemoryProperties().memoryHeapCount; i++) {
		std::printf("  heap %u: usage %" PRIu64 " MiB of budget %" PRIu64 " MiB\n", i, budgets[i].usage >> 20u,
		            budgets[i].budget >> 20u);
	}
	std::fflush(stdout);
}

// Issue #228: an RX 9070 with 5.4 GB of its 15.4 GB of video memory free refused a 108 MB buffer at a world load, the
// PC's RAM nearly full (2 GB of 24 GB left), and the emulator exited: which memory refused it was not known.
void GraphicContext::DescribeAllocationFailure(const char* what, uint64_t bytes) const {
	static std::atomic<uint32_t> described {0};
	if (described.fetch_add(1, std::memory_order_relaxed) >= 4) return;
	VmaTotalStatistics statistics {};
	vmaCalculateStatistics(allocator, &statistics);
	const auto commit = LocalPlatform::SystemCommit();
	std::printf("Vulkan: %s of %" PRIu64 " MiB not made: VMA holds %u memory objects (the GPU's limit %u); the PC has %" PRIu64
	            " MiB of RAM left and %" PRIu64 " MiB to commit of %" PRIu64 " MiB\n",
	            what, bytes >> 20u, statistics.total.statistics.blockCount, physical_device_properties.limits.maxMemoryAllocationCount,
	            LocalPlatform::AvailablePhysicalMemory() >> 20u, commit.available >> 20u, commit.limit >> 20u);
	// The driver asked directly for the same size (with a device address, as VMA asks), in the first memory type of each
	// kind: video memory, and system memory the CPU sees.
	const auto& memory = GetPhysicalDeviceMemoryProperties();
	for (const bool video: {true, false}) {
		for (uint32_t type = 0; type < memory.memoryTypeCount; type++) {
			const auto flags = memory.memoryTypes[type].propertyFlags;
			if (static_cast<bool>(flags & vk::MemoryPropertyFlagBits::eDeviceLocal) != video ||
			    (!video && !(flags & vk::MemoryPropertyFlagBits::eHostVisible)))
				continue;
			vk::MemoryAllocateFlagsInfo address {};
			address.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
			vk::MemoryAllocateInfo info {};
			info.pNext           = &address;
			info.allocationSize  = std::max<uint64_t>(bytes, uint64_t {64} << 10u);
			info.memoryTypeIndex = type;
			vk::DeviceMemory probe  = nullptr;
			const auto       result = device.allocateMemory(&info, nullptr, &probe);
			if (result == vk::Result::eSuccess) device.freeMemory(probe, nullptr);
			std::printf("  the driver for the same size in memory type %u (heap %u, %s): %s\n", type,
			            memory.memoryTypes[type].heapIndex, video ? "video memory" : "system memory", vk::to_string(result).c_str());
			break;
		}
	}
	std::fflush(stdout);
}

bool GraphicContext::RetryAllocation(const char* what, uint64_t bytes, const std::function<bool()>& create) {
	DescribeAllocationFailure(what, bytes);
	const auto start = std::chrono::steady_clock::now();
	for (int attempt = 0; attempt < 100; attempt++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		TrimImagePool();
		if (create()) {
			const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
			std::printf("Vulkan: %s of %" PRIu64 " MiB made after waiting %lld ms\n", what, bytes >> 20u,
			            static_cast<long long>(waited.count()));
			std::fflush(stdout);
			return true;
		}
	}
	std::printf("Vulkan: %s of %" PRIu64 " MiB still not made after 10 s\n", what, bytes >> 20u);
	std::fflush(stdout);
	return false;
}

// The textures made in the host-preferred type (CreateImage): the memory type VMA chose and its heap (a heap Vulkan lists as system
// memory: every sample over PCIe, unless the driver keeps the type in video memory, as NVIDIA's Windows driver does while there is
// room: the Video memory lines' plain type), printed at every further 256 MiB of them (the GPU thread's; not freed counts).
void GraphicContext::NoteHostPreferredTexture(VmaAllocation allocation, double host_at) {
	// (The GPU thread makes the textures, the present thread its frame image: both reach here.)
	static std::mutex mutex;
	std::scoped_lock  lock(mutex);
	static uint64_t   images = 0, bytes = 0, system_bytes = 0, next = 0;
	VmaAllocationInfo info {};
	vmaGetAllocationInfo(allocator, allocation, &info);
	const auto& memory = GetPhysicalDeviceMemoryProperties();
	const bool  system = info.memoryType < memory.memoryTypeCount &&
	                    !(memory.memoryHeaps[memory.memoryTypes[info.memoryType].heapIndex].flags & vk::MemoryHeapFlagBits::eDeviceLocal);
	images++;
	bytes += info.size;
	system_bytes += system ? info.size : 0;
	if (bytes < next) return;
	next = bytes + (uint64_t {256} << 20u);
	// (Each heap's usage against its budget, VK_EXT_memory_budget through VMA: where the no-flags images actually live.)
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	std::string heaps;
	for (uint32_t heap = 0; heap < memory.memoryHeapCount; heap++) {
		char part[96];
		std::snprintf(part, sizeof(part), "%s heap %u (%s) %" PRIu64 " of %" PRIu64 " MiB", heap == 0 ? "" : ",", heap,
		              (memory.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) ? "video" : "system",
		              static_cast<uint64_t>(budgets[heap].usage) >> 20u, static_cast<uint64_t>(budgets[heap].budget) >> 20u);
		heaps += part;
	}
	std::printf("Textures placed past the threshold (from %.2f of the budget, KYTY_TEXTURE_HOST_AT; mode %s): %" PRIu64 " made, %" PRIu64
	            " MiB, %" PRIu64 " MiB of them in a heap listed as system memory (memory type %u now; where the driver keeps them: the Video memory "
	            "lines' plain type); heaps:%s\n",
	            host_at, std::getenv("KYTY_TEXTURE_HOST_MODE") != nullptr ? std::getenv("KYTY_TEXTURE_HOST_MODE") : "host", images,
	            bytes >> 20u, system_bytes >> 20u, info.memoryType, heaps.c_str());
	std::fflush(stdout);
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image, bool guest_texture) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	if (kyty_local_image_pool_mode.load(std::memory_order_relaxed) != 0 && image_info.pNext == nullptr &&
	    image_info.tiling == vk::ImageTiling::eOptimal &&
	    image_info.initialLayout == vk::ImageLayout::eUndefined) {
		const auto key = ImagePoolKey(image_info);
		std::scoped_lock lock(g_image_pool_mutex);
		for (auto it = g_image_pool.begin(); it != g_image_pool.end(); ++it) {
			if (it->key != key) {
				continue;
			}
			image.image      = it->image;
			image.allocation = it->allocation;
			g_image_pool_bytes -= it->bytes;
			g_image_pool.erase(it);
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
	}

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	// On a GPU that cannot hold the game's working set (small_video_memory: Boletaria uses ~11 GB), an image only
	// sampled and copied (a block-compressed texture) goes to the driver's memory type without flags once half the
	// budget is in use: NVIDIA places it in video memory while there is room and in system memory after, so render
	// targets, depth, storage images and buffers, which the GPU reads and writes every frame, keep video memory (it
	// fails device-local allocations past its video memory: what came last went to system memory, 8 GB at 1-1:
	// 22 fps, textures yielding: ~45). It counts that type under the video memory heap: on 8 GB, half is reached
	// before the first texture.
	// KYTY_TEXTURE_HOST_AT=<fraction> (boot env, (0, 1], default 0.5: the half above): the budget use from which textures go to that
	// type (PR #284: on Linux, where that type is system memory, the 1-1 pixel-heavy table draws read such textures over PCIe).
	static const double host_at = [] {
		const char*  value    = std::getenv("KYTY_TEXTURE_HOST_AT");
		const double fraction = value != nullptr ? std::strtod(value, nullptr) : 0.0;
		const double used     = fraction > 0.0 && fraction <= 1.0 ? fraction : 0.5;
		// (The bench's engage line: read at the first image, early in the boot.)
		if (value != nullptr) std::printf("Textures: host-preferred from %.2f of the budget (KYTY_TEXTURE_HOST_AT=%s)\n", used, value);
		return used;
	}();
	// KYTY_TEXTURE_HOST_MODE (boot env; default: the above): pd = past the threshold a guest texture may go anywhere but prefers video
	// memory (requiredFlags 0, preferredFlags DEVICE_LOCAL, no AUTO_PREFER_HOST: VMA, given AUTO_PREFER_HOST and no host access, marks
	// DEVICE_LOCAL not preferred, vk_mem_alloc.h ~4347, so those textures asked for a type outside video memory up front, the driver never
	// choosing): system memory only when video memory fails; and the emulator's own images (the presenter's frame, frame generation's
	// inputs, the HUD source) always video memory, as render targets are.
	static const bool prefer_device = [] {
		const char* value = std::getenv("KYTY_TEXTURE_HOST_MODE");
		const bool  pd    = value != nullptr && std::strcmp(value, "pd") == 0;
		if (pd) std::printf("Textures: host mode pd (KYTY_TEXTURE_HOST_MODE=pd: guest textures prefer video memory past the threshold, "
		                    "the emulator's images always video memory)\n");
		return pd;
	}();
	const bool texture = IsTextureUsage(image_info.usage) && (!prefer_device || guest_texture);
	const bool host_preferred =
	    texture && small_video_memory &&
	    static_cast<double>(GetDeviceMemoryUsage()) >= static_cast<double>(GetTotalMemoryBudget()) * host_at;
	if (host_preferred) {
		alloc_info.requiredFlags = 0;
		if (prefer_device) alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		else alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	}
	vk::Result result = vk::Result::eSuccess;
	const auto create = [&] {
		vk::Image::CType native_image = VK_NULL_HANDLE;
		result                        = static_cast<vk::Result>(
		    vmaCreateImage(allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
		                   &alloc_info, &native_image, &image.allocation, nullptr));
		image.image = native_image;
		return result == vk::Result::eSuccess;
	};
	// Out of video memory: without the freed images kept for reuse, then in system memory (slower
	// to sample, but the game goes on).
	if (!create()) {
		TrimImagePool();
		if (!create()) {
			alloc_info.requiredFlags  = 0;
			alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
			const bool created        = create();
			VmaAllocationInfo allocated {};
			if (created) vmaGetAllocationInfo(allocator, image.allocation, &allocated);
			if (created) {
				ReportMemoryFallback("an image is in system memory", allocated.size);
			} else {
				// (Which memory ran out: OUT_OF_HOST_MEMORY is the system's commit, OUT_OF_DEVICE_MEMORY the GPU's.)
				const auto what = "an image could not be created (" + vk::to_string(result) + ")";
				ReportMemoryFallback(what.c_str(), 0);
				// (No retry here: an image that may fail, a texture pool's view, is drawn without it instead.)
				vk::Image      probe = nullptr;
				vk::DeviceSize size  = 0;
				if (device.createImage(&image_info, nullptr, &probe) == vk::Result::eSuccess) {
					size = device.getImageMemoryRequirements(probe).size;
					device.destroyImage(probe, nullptr);
				}
				DescribeAllocationFailure("an image", size);
				return false;
			}
		}
	}

	if (host_preferred) NoteHostPreferredTexture(image.allocation, host_at);
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

	if (kyty_local_image_pool_mode.load(std::memory_order_relaxed) != 0) {
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		std::scoped_lock lock(g_image_pool_mutex);
		g_image_pool.push_back({ImagePoolKey(image), image.image, image.allocation, allocation_info.size});
		g_image_pool_bytes += allocation_info.size;
		while (g_image_pool_bytes > g_image_pool_limit && !g_image_pool.empty()) {
			auto& oldest = g_image_pool.front();
			vmaDestroyImage(allocator, oldest.image, oldest.allocation);
			g_image_pool_bytes -= oldest.bytes;
			g_image_pool.erase(g_image_pool.begin());
		}
		image.image      = nullptr;
		image.allocation = nullptr;
		return;
	}
	vmaDestroyImage(allocator, image.image, image.allocation);
	image.image      = nullptr;
	image.allocation = nullptr;
}

} // namespace Libs::Graphics
