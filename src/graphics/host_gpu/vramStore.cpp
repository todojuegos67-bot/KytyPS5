// (mem1 etapa 2, 11-23) The native offload's store in video memory: see local/vram-store.h.
//
// One host-visible staging buffer of 1 MiB (the offload's chunk, FileChunk), device-local blocks of 64 MiB grown on
// demand, one command buffer and one fence: every read or write is a GPU copy, submitted and waited for at once under
// the store's lock. The queue is the second queue of the graphics family (the one only presents use, under
// present_queue_mutex): never behind the frame's own work on the graphics queue. Without that queue the store stays off.
// Calls may come from a game thread's fault (a touch of a range taken): nothing here touches guest memory.

#include "graphics/host_gpu/graphicContext.h"
#include "local/vram-store.h"
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "local/vulkan-recording.h"
#include "local/vram-drv.h" // (VRAM-DRV)
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace Libs::Graphics {

namespace {

const auto& Dispatch() {
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	return LocalVulkanRecording::DirectDispatch();
#else
	return VULKAN_HPP_DEFAULT_DISPATCHER;
#endif
}

constexpr uint64_t BlockBytes   = uint64_t {64} << 20u;
constexpr uint64_t StagingBytes = uint64_t {1} << 20u;
constexpr uint64_t MinUseful    = uint64_t {1} << 30u;

struct Block {
	VkBuffer       buffer = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct Store {
	std::mutex         mutex;
	GraphicContext*    graphics = nullptr;
	bool               probed   = false;
	bool               ok       = false;
	VkDevice           device   = VK_NULL_HANDLE;
	VkQueue            queue    = VK_NULL_HANDLE;
	uint32_t           family   = 0;
	VkCommandPool      pool     = VK_NULL_HANDLE;
	VkCommandBuffer    command  = VK_NULL_HANDLE;
	VkFence            fence    = VK_NULL_HANDLE;
	VkBuffer           staging  = VK_NULL_HANDLE;
	VkDeviceMemory     staging_memory = VK_NULL_HANDLE;
	uint8_t*           mapped   = nullptr;
	bool               coherent = true;
	uint32_t           device_type = UINT32_MAX;
	std::vector<Block> blocks;
};

Store& Get() {
	static auto* store = new Store; // (never destroyed: detached threads may still call)
	return *store;
}

uint32_t FindType(const VkPhysicalDeviceMemoryProperties& props, uint32_t bits, VkMemoryPropertyFlags need,
                  VkMemoryPropertyFlags prefer, VkMemoryPropertyFlags avoid) {
	uint32_t best = UINT32_MAX;
	int      best_score = -1;
	for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
		if ((bits & (1u << i)) == 0) continue;
		const auto flags = props.memoryTypes[i].propertyFlags;
		if ((flags & need) != need) continue;
		int score = 1;
		if ((flags & prefer) == prefer) score += 2;
		if ((flags & avoid) == 0) score += 1;
		if (score > best_score) {
			best_score = score;
			best       = i;
		}
	}
	return best;
}

bool MakeBuffer(Store& s, uint64_t bytes, VkBufferUsageFlags usage, bool host, VkBuffer* buffer, VkDeviceMemory* memory,
                uint32_t* type_out) {
	const auto&        d = Dispatch();
	VkBufferCreateInfo info {};
	info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	info.size        = bytes;
	info.usage       = usage;
	info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	if (d.vkCreateBuffer(s.device, &info, nullptr, buffer) != VK_SUCCESS) return false;
	VkMemoryRequirements req {};
	d.vkGetBufferMemoryRequirements(s.device, *buffer, &req);
	const VkPhysicalDeviceMemoryProperties props = s.graphics->physical_device_memory_properties;
	uint32_t type = UINT32_MAX;
	if (host) {
		type = FindType(props, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
		                VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	} else {
		type = FindType(props, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	}
	if (type == UINT32_MAX) {
		d.vkDestroyBuffer(s.device, *buffer, nullptr);
		*buffer = VK_NULL_HANDLE;
		return false;
	}
	VkMemoryAllocateInfo allocate {};
	allocate.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocate.allocationSize  = req.size;
	allocate.memoryTypeIndex = type;
	VramDrv::Scope vram_drv(VramDrv::CatVramStore); // (VRAM-DRV)
	if (d.vkAllocateMemory(s.device, &allocate, nullptr, memory) != VK_SUCCESS) {
		d.vkDestroyBuffer(s.device, *buffer, nullptr);
		*buffer = VK_NULL_HANDLE;
		*memory = VK_NULL_HANDLE;
		return false;
	}
	if (d.vkBindBufferMemory(s.device, *buffer, *memory, 0) != VK_SUCCESS) {
		d.vkFreeMemory(s.device, *memory, nullptr);
		d.vkDestroyBuffer(s.device, *buffer, nullptr);
		*buffer = VK_NULL_HANDLE;
		*memory = VK_NULL_HANDLE;
		return false;
	}
	if (type_out != nullptr) *type_out = type;
	return true;
}

// (store's lock held) Whether one more block fits: the caches keep their capped budget (the store's bytes are added to
// it, vma.cpp), and the driver's budget keeps the margin.
bool BlockFits(Store& s) {
	auto&          r         = VramStore::Get();
	const uint64_t allocated = r.allocated.load(std::memory_order_relaxed);
	if (allocated + BlockBytes > r.cap.load(std::memory_order_relaxed)) return false;
	const uint64_t room = s.graphics->GetVideoMemoryRoom();
	return room >= BlockBytes + VramStore::MarginBytes() / 2;
}

// (store's lock held) The blocks over [offset, offset + bytes) exist (made when writing; a read of a missing one fails).
bool EnsureBlocks(Store& s, uint64_t offset, uint64_t bytes, bool make) {
	const size_t last = static_cast<size_t>((offset + bytes - 1) / BlockBytes);
	if (s.blocks.size() <= last) {
		if (!make) return false;
		s.blocks.resize(last + 1);
	}
	for (size_t b = static_cast<size_t>(offset / BlockBytes); b <= last; b++) {
		if (s.blocks[b].buffer != VK_NULL_HANDLE) continue;
		if (!make) return false;
		auto& r = VramStore::Get();
		if (!BlockFits(s)) {
			r.refused.fetch_add(1, std::memory_order_relaxed);
			return false;
		}
		Block block;
		if (!MakeBuffer(s, BlockBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &block.buffer,
		                &block.memory, nullptr)) {
			r.refused.fetch_add(1, std::memory_order_relaxed);
			return false;
		}
		s.blocks[b] = block;
		r.allocated.fetch_add(BlockBytes, std::memory_order_relaxed);
	}
	return true;
}

// (store's lock held) One copy between the staging buffer and the blocks, submitted and waited for.
bool Copy(Store& s, bool to_blocks, uint64_t offset, uint64_t bytes) {
	const auto& d = Dispatch();
	if (d.vkResetCommandBuffer(s.command, 0) != VK_SUCCESS) return false;
	VkCommandBufferBeginInfo begin {};
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	if (d.vkBeginCommandBuffer(s.command, &begin) != VK_SUCCESS) return false;
	uint64_t done = 0;
	while (done < bytes) {
		const uint64_t at    = offset + done;
		const size_t   b     = static_cast<size_t>(at / BlockBytes);
		const uint64_t inner = at % BlockBytes;
		const uint64_t n     = std::min(bytes - done, BlockBytes - inner);
		VkBufferCopy   region {};
		region.size = n;
		if (to_blocks) {
			region.srcOffset = done;
			region.dstOffset = inner;
			d.vkCmdCopyBuffer(s.command, s.staging, s.blocks[b].buffer, 1, &region);
		} else {
			region.srcOffset = inner;
			region.dstOffset = done;
			d.vkCmdCopyBuffer(s.command, s.blocks[b].buffer, s.staging, 1, &region);
		}
		done += n;
	}
	VkMemoryBarrier barrier {};
	barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = to_blocks ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_HOST_READ_BIT;
	d.vkCmdPipelineBarrier(s.command, VK_PIPELINE_STAGE_TRANSFER_BIT,
	                       to_blocks ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0,
	                       nullptr);
	if (d.vkEndCommandBuffer(s.command) != VK_SUCCESS) return false;
	VkSubmitInfo submit {};
	submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &s.command;
	VkResult result = VK_SUCCESS;
	{
		Common::LockGuard lock(s.graphics->present_queue_mutex);
		result = d.vkQueueSubmit(s.queue, 1, &submit, s.fence);
	}
	if (result != VK_SUCCESS) return false;
	result = d.vkWaitForFences(s.device, 1, &s.fence, VK_TRUE, uint64_t {5} * 1000 * 1000 * 1000);
	(void)d.vkResetFences(s.device, 1, &s.fence);
	return result == VK_SUCCESS;
}

bool Io(bool write, uint64_t offset, uint8_t* data, uint64_t bytes) {
	auto& r = VramStore::Get();
	auto& s = Get();
	if (bytes == 0 || bytes > StagingBytes || data == nullptr) return false;
	const auto      start = std::chrono::steady_clock::now();
	std::lock_guard lock(s.mutex);
	if (!s.ok) return false;
	const auto& d = Dispatch();
	bool        ok = EnsureBlocks(s, offset, bytes, write);
	if (ok && write) {
		std::memcpy(s.mapped, data, bytes);
		if (!s.coherent) {
			VkMappedMemoryRange range {};
			range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			range.memory = s.staging_memory;
			range.size   = VK_WHOLE_SIZE;
			ok           = d.vkFlushMappedMemoryRanges(s.device, 1, &range) == VK_SUCCESS;
		}
		ok = ok && Copy(s, true, offset, bytes);
	} else if (ok) {
		ok = Copy(s, false, offset, bytes);
		if (ok && !s.coherent) {
			VkMappedMemoryRange range {};
			range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			range.memory = s.staging_memory;
			range.size   = VK_WHOLE_SIZE;
			ok           = d.vkInvalidateMappedMemoryRanges(s.device, 1, &range) == VK_SUCCESS;
		}
		if (ok) std::memcpy(data, s.mapped, bytes);
	}
	const auto us = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
	if (!ok) {
		r.failed.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	if (write) {
		r.writes.fetch_add(1, std::memory_order_relaxed);
		r.write_bytes.fetch_add(bytes, std::memory_order_relaxed);
		r.write_us.fetch_add(us, std::memory_order_relaxed);
	} else {
		r.reads.fetch_add(1, std::memory_order_relaxed);
		r.read_bytes.fetch_add(bytes, std::memory_order_relaxed);
		r.read_us.fetch_add(us, std::memory_order_relaxed);
		uint64_t max = r.read_us_max.load(std::memory_order_relaxed);
		while (us > max && !r.read_us_max.compare_exchange_weak(max, us, std::memory_order_relaxed)) {
		}
	}
	return true;
}

bool Probe() {
	auto&           r = VramStore::Get();
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	if (s.probed) return s.ok;
	s.probed       = true;
	GraphicContext* g = s.graphics;
	const auto      off = [&](const char* why) {
        std::printf("[mem1-etapa2] apagada: %s; el almacen sigue en disco como antes\n", why);
        std::fflush(stdout);
        return false;
	};
	if (g == nullptr || g->device == nullptr) return off("sin dispositivo Vulkan");
	if (g->physical_device_properties.deviceType != vk::PhysicalDeviceType::eDiscreteGpu) return off("la GPU no es dedicada");
	if (!g->present_queue_created || g->present_queue == nullptr)
		return off("sin cola aparte de la familia grafica (la de presentar)");
	if (!g->CanReportMemoryUsage()) return off("el driver no da su presupuesto de VRAM");
	// Room: the driver's budget (less its headroom) over the caches' capped budget, less the margin.
	const uint64_t uncapped = g->GetTotalMemoryBudget(false);
	const uint64_t capped   = g->GetTotalMemoryBudget(true) - std::min(g->GetTotalMemoryBudget(true), r.allocated.load());
	const uint64_t margin   = VramStore::MarginBytes();
	const uint64_t room     = uncapped > capped + margin ? uncapped - capped - margin : 0;
	const uint64_t cap      = std::min(VramStore::AskedBytes(), room) / BlockBytes * BlockBytes;
	if (cap < MinUseful) {
		char why[256];
		std::snprintf(why, sizeof(why),
		              "VRAM insuficiente: presupuesto del driver %llu MiB, el de las caches %llu MiB, margen %llu MiB: quedan %llu MiB "
		              "(minimo 1024)",
		              static_cast<unsigned long long>(uncapped >> 20u), static_cast<unsigned long long>(capped >> 20u),
		              static_cast<unsigned long long>(margin >> 20u), static_cast<unsigned long long>(room >> 20u));
		return off(why);
	}
	s.device = static_cast<VkDevice>(g->device);
	s.queue  = static_cast<VkQueue>(g->present_queue);
	s.family = g->queue_family;
	const auto&             d = Dispatch();
	VkCommandPoolCreateInfo pool {};
	pool.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	pool.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	pool.queueFamilyIndex = s.family;
	if (d.vkCreateCommandPool(s.device, &pool, nullptr, &s.pool) != VK_SUCCESS) return off("no se pudo crear el command pool");
	VkCommandBufferAllocateInfo allocate {};
	allocate.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocate.commandPool        = s.pool;
	allocate.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocate.commandBufferCount = 1;
	if (d.vkAllocateCommandBuffers(s.device, &allocate, &s.command) != VK_SUCCESS) return off("no se pudo crear el command buffer");
	VkFenceCreateInfo fence {};
	fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	if (d.vkCreateFence(s.device, &fence, nullptr, &s.fence) != VK_SUCCESS) return off("no se pudo crear la fence");
	uint32_t type = UINT32_MAX;
	if (!MakeBuffer(s, StagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, &s.staging,
	                &s.staging_memory, &type))
		return off("no se pudo crear el buffer de paso");
	s.coherent = (s.graphics->physical_device_memory_properties.memoryTypes[type].propertyFlags &
	              vk::MemoryPropertyFlagBits::eHostCoherent) != vk::MemoryPropertyFlags {};
	void* mapped = nullptr;
	if (d.vkMapMemory(s.device, s.staging_memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS || mapped == nullptr)
		return off("no se pudo mapear el buffer de paso");
	s.mapped = static_cast<uint8_t*>(mapped);
	r.cap.store(cap, std::memory_order_relaxed);
	s.ok = true;
	r.active.store(true, std::memory_order_release);
	std::printf("[mem1-etapa2] encendida (KYTY_MEM1_STAGE2=1; =0 la apaga): el almacen del offload en la VRAM, hasta %llu MiB "
	            "(presupuesto del driver %llu MiB, el de las caches %llu MiB + lo del almacen, margen %llu MiB); VirtualMemo sin "
	            "ventana del emulador en %d s sale de la RAM\n",
	            static_cast<unsigned long long>(cap >> 20u), static_cast<unsigned long long>(uncapped >> 20u),
	            static_cast<unsigned long long>(capped >> 20u), static_cast<unsigned long long>(margin >> 20u),
	            VramStore::IdleSeconds());
	std::fflush(stdout);
	return true;
}

} // namespace

// (vma.cpp, CreateAllocator) The store registered: its resources are made at the first probe (ram-guard's RAM18 thread,
// once the game runs).
void VramStoreRegister(GraphicContext& graphics) {
	if (!VramStore::Asked()) return;
	Get().graphics = &graphics;
	auto& r        = VramStore::Get();
	r.io.store(&Io, std::memory_order_release);
	r.probe.store(&Probe, std::memory_order_release);
}

} // namespace Libs::Graphics
