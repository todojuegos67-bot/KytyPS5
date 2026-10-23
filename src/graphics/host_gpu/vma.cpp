#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
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
#include "live-counters.h"
#include "local-platform.h"
#include "soft-release.h"

#include <cstdio>
#include "kytyGitVersion.h"

#include <algorithm>
#include <cinttypes>

namespace Libs::Graphics {

void FlushBufferReclaimer(); // streamBuffer.cpp (KYTY_BUFFER_RECLAIM)

// Local diagnostic (live `vma <path>`): VMA is thread-safe, so the live thread writes it.
static VmaAllocator          g_report_allocator = nullptr;
static const GraphicContext* g_report_context   = nullptr;
// Freed images kept for reuse (KYTY_IMAGE_POOL): at most 1 GiB, and a 16th of the GPU's memory budget.
static uint64_t g_image_pool_limit = 1024ull << 20;
static std::mutex g_image_pool_mutex;
static uint64_t   g_image_pool_bytes = 0;
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

// Tagged commit of Vulkan memory (LocalPlatform::MemoryTagAdd): every vkAllocateMemory (VMA's, the ballast's, the
// stream buffers') with the process's private commit before and after it, serialized so that the delta is its own
// (other threads' allocations meanwhile still add noise). NVIDIA's Windows driver can charge device-local memory to
// the process's commit as well: this tells how much, by kind, in the "Private breakdown:" line.
namespace {
PFN_vkAllocateMemory           g_real_allocate = nullptr;
PFN_vkFreeMemory               g_real_free     = nullptr;
VkPhysicalDeviceMemoryProperties g_memory_properties {};
std::mutex                     g_tag_mutex;
std::atomic<bool>              g_tag_ballast {false};
struct TaggedMemory {
	int     tag;
	int64_t requested, committed;
};
std::unordered_map<uint64_t, TaggedMemory>* g_tagged = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL TaggedAllocateMemory(VkDevice device, const VkMemoryAllocateInfo* info,
                                                    const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory) {
	std::lock_guard lock(g_tag_mutex);
	const auto before = LocalPlatform::ProcessMemory().private_bytes;
	const auto result = g_real_allocate(device, info, callbacks, memory);
	const auto after  = LocalPlatform::ProcessMemory().private_bytes;
	if (result == VK_SUCCESS && memory != nullptr && *memory != VK_NULL_HANDLE && info != nullptr) {
		const auto flags = info->memoryTypeIndex < g_memory_properties.memoryTypeCount
		                       ? g_memory_properties.memoryTypes[info->memoryTypeIndex].propertyFlags
		                       : 0u;
		const int tag = g_tag_ballast.load() ? LocalPlatform::MemoryTagVulkanBallast
		                : (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 ? LocalPlatform::MemoryTagVulkanDeviceLocal
		                                                                     : LocalPlatform::MemoryTagVulkanHost;
		const TaggedMemory entry {tag, static_cast<int64_t>(info->allocationSize),
		                          after > before ? static_cast<int64_t>(after - before) : 0};
		if (g_tagged == nullptr) g_tagged = new std::unordered_map<uint64_t, TaggedMemory>();
		(*g_tagged)[reinterpret_cast<uint64_t>(*memory)] = entry;
		LocalPlatform::MemoryTagAdd(entry.tag, entry.requested, entry.committed);
	}
	return result;
}

VKAPI_ATTR void VKAPI_CALL TaggedFreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* callbacks) {
	{
		std::lock_guard lock(g_tag_mutex);
		if (g_tagged != nullptr && memory != VK_NULL_HANDLE) {
			if (auto it = g_tagged->find(reinterpret_cast<uint64_t>(memory)); it != g_tagged->end()) {
				LocalPlatform::MemoryTagAdd(it->second.tag, -it->second.requested, -it->second.committed);
				g_tagged->erase(it);
			}
		}
	}
	g_real_free(device, memory, callbacks);
}

void InstallMemoryTags(VkPhysicalDevice physical_device) {
	if (!LocalPlatform::MemoryTagsEnabled() || g_real_allocate != nullptr) return;
	auto& dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;
	if (dispatcher.vkAllocateMemory == nullptr || dispatcher.vkFreeMemory == nullptr) return;
	dispatcher.vkGetPhysicalDeviceMemoryProperties(physical_device, &g_memory_properties);
	g_real_allocate             = dispatcher.vkAllocateMemory;
	g_real_free                 = dispatcher.vkFreeMemory;
	dispatcher.vkAllocateMemory = TaggedAllocateMemory;
	dispatcher.vkFreeMemory     = TaggedFreeMemory;
}
} // namespace

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
	InstallMemoryTags(static_cast<VkPhysicalDevice>(physical_device));
	if (g_real_allocate != nullptr) {
		functions.vkAllocateMemory = TaggedAllocateMemory;
		functions.vkFreeMemory     = TaggedFreeMemory;
	}

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
	// Memory blocks of 64 MiB (VMA's default is 256): a block stays allocated while any allocation of it lives, so
	// the game's streaming left 21 blocks of 256 MiB with 2.7 GB free inside them at Boletaria (11.9 GB held of
	// which 8.9 in use); smaller blocks hold less around each straggler. KYTY_VMA_BLOCK_MB=<n> for an A/B.
	{
		uint64_t block_mb = LocalPlatform::LowMemVram() ? 32 : 64;
		if (const char* text = std::getenv("KYTY_VMA_BLOCK_MB"); text != nullptr && std::strtoull(text, nullptr, 10) > 0)
			block_mb = std::strtoull(text, nullptr, 10);
		info.preferredLargeHeapBlockSize = block_mb << 20u;
		std::printf("Video memory: allocator blocks of %llu MiB\n", static_cast<unsigned long long>(block_mb));
	}
	// KYTY_VRAM_LIMIT_MB=<n>: video memory as on a GPU with n MiB (VMA's heap size limit): its budget,
	// and allocations past it failing, for tests of memory pressure on smaller GPUs.
	std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> heap_limits {};
	if (const char* text = std::getenv("KYTY_VRAM_LIMIT_MB"); text != nullptr && std::strtoull(text, nullptr, 10) > 0) {
		heap_limits.fill(VK_WHOLE_SIZE);
		for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
			if (physical_device_memory_properties.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
				heap_limits[heap] = std::strtoull(text, nullptr, 10) << 20u;
			}
		}
		info.pHeapSizeLimit = heap_limits.data();
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	// KYTY_SIMULATE_VRAM_MB=<n>: a GPU with n MiB of video memory, for tests on a bigger one. Unlike
	// KYTY_VRAM_LIMIT_MB (an allocator limit: what it puts in system memory types the driver still places in video
	// memory while there is room), the rest of the largest device-local heap is taken here at the highest priority
	// and kept, so the driver and OS move the game's memory to system memory as on such a GPU. Budget and usage
	// (GetTotalMemoryBudget, GetDeviceMemoryUsage) leave it out.
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
		g_tag_ballast = true;
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
		g_tag_ballast = false;
		LocalPlatform::SetSimulationBallast(simulation_ballast_bytes);
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
	// (The GPU's own budget, not the caches' cap: a 16 GB card is not short of video memory.)
	// KYTY_SMALL_VRAM=1 (run-windows.ps1 -Turbo): as on a small GPU whatever this one has (textures yield, no warmup).
	const char* small_env = std::getenv("KYTY_SMALL_VRAM");
	small_video_memory = GetTotalMemoryBudget(false) < (uint64_t {12} << 30u) || (small_env != nullptr && *small_env == '1');
	LocalPlatform::StartWorkingSetTrim();
	SoftRelease::Start();
	LocalPlatform::KeepResident();
	LocalPlatform::StartMemoryBreakdown();
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
	g_image_pool_limit         = std::min<uint64_t>(LocalPlatform::LowMemVram() ? 256ull << 20 : 1024ull << 20, GetTotalMemoryBudget() / 32);
	{
		// The caches' video memory budget, in the run log with the build (to tell builds apart in a log).
		uint64_t local = 0;
		for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
			if (physical_device_memory_properties.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
				local += physical_device_memory_properties.memoryHeaps[heap].size;
			}
		}
		std::printf("Video memory: %llu MiB on the GPU; the caches' budget is %llu MiB (build %s)\n",
		            static_cast<unsigned long long>(local >> 20u),
		            static_cast<unsigned long long>(GetTotalMemoryBudget() >> 20u), KYTY_GIT_HASH);
		std::fflush(stdout);
	}
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
	if (std::strcmp(label, "30 s") == 0) {
		// By category: the freed images kept for reuse, the game buffers' own pool, the rest of VMA's device-local
		// memory (textures and render targets of the texture cache, effect and upscaler images, stream rings).
		uint64_t pool_bytes = 0;
		{
			std::scoped_lock lock(g_image_pool_mutex);
			pool_bytes = g_image_pool_bytes;
		}
		VmaStatistics buffers {};
		if (game_buffer_pool != nullptr) vmaGetPoolStatistics(allocator, game_buffer_pool, &buffers);
		const uint64_t used_local = [&] {
			uint64_t used = 0;
			for (uint32_t type = 0; type < properties.memoryTypeCount; type++)
				if (properties.memoryTypes[type].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)
					used += statistics.memoryType[type].statistics.allocationBytes;
			return used;
		}();
		const uint64_t other = used_local - std::min(used_local, pool_bytes + buffers.allocationBytes);
		std::printf("Video memory by category (30 s): game buffer pool %" PRIu64 " MiB (blocks %" PRIu64
		            "), image reuse pool %" PRIu64 " MiB, images and other device-local %" PRIu64
		            " MiB, free inside VMA blocks %" PRIu64 " MiB, driver %" PRIu64 " MiB, room under the budget %" PRId64
		            " MiB\n",
		            buffers.allocationBytes >> 20u, buffers.blockBytes >> 20u, pool_bytes >> 20u, other >> 20u,
		            (local - std::min(local, used_local)) >> 20u, driver >> 20u,
		            (static_cast<int64_t>(budget) - static_cast<int64_t>(video)) / (1 << 20));
	}
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

// Allocations in the heaps that are not device-local (system RAM the driver maps for the GPU): staging and
// readback buffers, and anything the budget pushed out of video memory.
uint64_t GraphicContext::GetHostMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!device_local) usage += budgets[heap].usage;
	}
	return usage;
}

uint64_t GraphicContext::GetTotalMemoryBudget(bool capped) const {
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
		uint64_t result = budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
		if (!capped) return result;
		// KYTY_VRAM_BUDGET_MB=<n> sets the caches' budget outright.
		static const uint64_t forced = [] {
			const char* text = std::getenv("KYTY_VRAM_BUDGET_MB");
			return text != nullptr ? std::strtoull(text, nullptr, 10) << 20u : uint64_t {0};
		}();
		if (forced != 0) return std::min(result, forced);
		// The caches collect from the GPU's memory less 3 GB, at most 8 GB: 8 GB on a 12 GB card and up
		// one, so the whole process (pipelines and the driver's own allocations on top) stays near 11-12 GB with
		// room for Windows, the desktop and a browser. (13.4 GB on a 16 GB card: 13.3 GB in use after 8 minutes,
		// near 15 GB with the rest of the system.) small_video_memory reads the driver's budget, not this cap.
		constexpr uint64_t GiB = 1024ull * 1024 * 1024;
		// 8 GB at most: a 16 GB card with the desktop, an overlay and a browser stayed near 15 GB in use at 10 GB.
		if (local > 4 * GiB) result = std::min(result, std::min(local - 3 * GiB, 8 * GiB));
		// KYTY_LOWMEM_VRAM=1: at most 6 GB, the GPU's memory less 4 GB (not below 3 GB), only on cards above 8.5 GB.
		// On 8 GB it gave 4096 instead of the default 5120 and hitched far more (simulated 8 GB: 25-97 hitch windows
		// at 4096 against 3-17 in most runs at 5120): the collector then deletes and re-uploads textures mid-play.
		if (LocalPlatform::LowMemVram() && local > 8 * GiB + GiB / 2)
			result = std::min(result, std::max(std::min(local - 4 * GiB, 6 * GiB), 3 * GiB));
		return result;
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
volatile std::atomic<uint32_t> kyty_local_image_pool_mode {0};
}
namespace {
struct PooledImage {
	std::array<uint32_t, 10> key {};
	VkImage                  image      = VK_NULL_HANDLE;
	VmaAllocation            allocation = nullptr;
	uint64_t                 bytes      = 0;
};
std::vector<PooledImage> g_image_pool;

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

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
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
	const bool texture = IsTextureUsage(image_info.usage);
	// KYTY_LOWMEM_VRAM=1: no redirect. The plain type's allocations are a second pool outside the caches' budget
	// (2593 MiB of it at 4096 budget on a simulated 8 GB card, the process's private commit +4.6 GB, NVIDIA charging
	// it to commit); device-local textures stay under the budget the caches collect to.
	if (texture && small_video_memory && !LocalPlatform::LowMemVram() &&
	    GetDeviceMemoryUsage() >= GetTotalMemoryBudget(false) / 2) {
		alloc_info.requiredFlags = 0;
		alloc_info.usage         = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
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
				return false;
			}
		}
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
