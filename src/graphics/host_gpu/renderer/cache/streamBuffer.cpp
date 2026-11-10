#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "device-fault.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>
#include <vk_mem_alloc.h>

extern "C" {
// KYTY_BUFFER_RECLAIM: 1: a retired buffer's memory is freed on a worker thread. Freeing a
// dedicated allocation is a kernel call (~0.1 ms on Windows) the render thread paid for every
// such buffer the cache retired while the game streams. VMA synchronizes internally, and a buffer
// is destroyed only after the GPU is done with it.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_buffer_reclaim_mode {1};
}

namespace Libs::Graphics {

namespace {

bool UseStatsOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_STAGING_STATS");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

class BufferReclaimer {
public:
	~BufferReclaimer() {
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_wake.notify_one();
		if (m_thread.joinable()) m_thread.join();
	}
	void Push(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation) {
		{
			std::lock_guard lock(m_mutex);
			if (!m_thread.joinable()) m_thread = std::thread([this] { Run(); });
			m_pending.push_back({allocator, buffer, allocation});
		}
		m_wake.notify_one();
	}
	// Returns once every pushed buffer is destroyed (before the allocator goes away).
	void Flush() {
		Release();
		std::unique_lock lock(m_mutex);
		m_idle.wait(lock, [this] { return m_pending.empty() && !m_busy; });
	}
	// BufferCache::Defragment: no memory is freed here while a pass over the game buffers' pool is open; what is pushed
	// meanwhile waits for its end. A pass names allocations of erased buffers waiting here, and VMA reads its moves till
	// the pass ends: freed meanwhile, they were read freed (8 GB GPUs: access violations in vmaGetAllocationInfo from
	// Defragment, its block gone, 10-23). False while frees are queued or run: the pass waits for a later flip. (Begun
	// over them, a pass lists allocations of erased buffers, which Defragment can only ignore: each takes a move of the
	// pass's 64, and VMA counts its block as one it cannot empty.)
	bool TryHold() {
		const std::unique_lock freeing(m_free_mutex, std::try_to_lock);
		if (!freeing.owns_lock()) return false;
		{
			const std::lock_guard lock(m_mutex);
			if (!m_pending.empty() || m_busy) return false;
		}
		m_hold = true;
		return true;
	}
	void Release() {
		std::vector<Item> held;
		{
			const std::lock_guard freeing(m_free_mutex);
			m_hold = false;
			held.swap(m_held);
		}
		if (held.empty()) return;
		{
			const std::lock_guard lock(m_mutex);
			m_pending.insert(m_pending.end(), held.begin(), held.end());
		}
		m_wake.notify_one();
	}

private:
	struct Item {
		VmaAllocator  allocator;
		VkBuffer      buffer;
		VmaAllocation allocation;
	};
	void Run() {
		std::vector<Item> batch;
		std::unique_lock  lock(m_mutex);
		for (;;) {
			m_wake.wait(lock, [this] { return m_stop || !m_pending.empty(); });
			if (m_pending.empty() && m_stop) return;
			batch.swap(m_pending);
			m_busy = true;
			lock.unlock();
			for (const auto& item: batch) {
				const std::lock_guard freeing(m_free_mutex);
				if (m_hold && item.allocation != nullptr) {
					m_held.push_back(item);
					continue;
				}
				vmaDestroyBuffer(item.allocator, item.buffer, item.allocation);
			}
			batch.clear();
			lock.lock();
			m_busy = false;
			m_idle.notify_all();
		}
	}
	std::mutex              m_mutex;
	std::condition_variable m_wake, m_idle;
	std::vector<Item>       m_pending;
	bool                    m_busy = false, m_stop = false;
	std::thread             m_thread;
	std::mutex              m_free_mutex; // (held over each free: TryHold)
	std::vector<Item>       m_held;
	bool                    m_hold = false;
};

BufferReclaimer& Reclaimer() {
	static BufferReclaimer reclaimer;
	return reclaimer;
}

constexpr size_t WATCHES_INITIAL_RESERVE = 0x4000;
constexpr size_t WATCHES_RESERVE_CHUNK   = 0x1000;

[[nodiscard]] VmaAllocationCreateFlags AllocationFlags(MemoryUsage usage) {
	switch (usage) {
		case MemoryUsage::Upload:
		case MemoryUsage::Stream:
			return VMA_ALLOCATION_CREATE_MAPPED_BIT |
			       VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
		case MemoryUsage::Download:
			return VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		case MemoryUsage::DeviceLocal: return {};
	}
	return {};
}

[[nodiscard]] VmaMemoryUsage AllocationUsage(MemoryUsage usage) {
	switch (usage) {
		case MemoryUsage::DeviceLocal:
		case MemoryUsage::Stream: return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		case MemoryUsage::Upload:
		case MemoryUsage::Download: return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	}
	return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
}

[[nodiscard]] bool AlignUp(uint64_t value, uint64_t alignment, uint64_t& result) {
	if (alignment == 0) {
		result = value;
		return true;
	}
	const auto remainder = value % alignment;
	if (remainder == 0) {
		result = value;
		return true;
	}
	const auto increment = alignment - remainder;
	if (value > std::numeric_limits<uint64_t>::max() - increment) {
		return false;
	}
	result = value + increment;
	return true;
}

// KYTY_TEST_BUFFER_SPILL_MB: the least size of the next buffer made in system memory (0: none; Buffer::Buffer).
std::atomic<uint64_t> g_forced_spill_bytes {[] {
	const char* value = std::getenv("KYTY_TEST_BUFFER_SPILL_MB");
	const auto  mib   = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull; // ("-1": ULLONG_MAX, off)
	return mib < (uint64_t {1} << 44u) ? mib << 20u : uint64_t {0};
}()};

// KYTY_TEST_BUFFER_FAIL_MB: the least size of the next buffer whose first 5 tries fail (0: none; Buffer::Buffer).
std::atomic<uint64_t> g_forced_failure_bytes {[] {
	const char* value = std::getenv("KYTY_TEST_BUFFER_FAIL_MB");
	const auto  mib   = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
	return mib < (uint64_t {1} << 44u) ? mib << 20u : uint64_t {0};
}()};

[[nodiscard]] bool TakeAtLeast(std::atomic<uint64_t>& least_bytes, uint64_t size) {
	auto least = least_bytes.load(std::memory_order_relaxed);
	while (least != 0 && size >= least) {
		if (least_bytes.compare_exchange_weak(least, 0, std::memory_order_relaxed)) return true;
	}
	return false;
}

[[nodiscard]] bool TakeForcedSpill(uint64_t size) {
	return TakeAtLeast(g_forced_spill_bytes, size);
}

} // namespace

Buffer::Buffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
               uint64_t cpu_address, vk::BufferUsageFlags flags, uint64_t size)
    : m_graphics(&graphics), m_scheduler(&scheduler), m_usage(usage), m_cpu_address(cpu_address),
      m_size(size) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(graphics.allocator == nullptr || size == 0);

	vk::BufferCreateInfo buffer_info {};
	buffer_info.size        = size;
	buffer_info.usage       = flags;
	// KYTY_READBACK_QUEUE: the transfer queue reads guest buffers and writes download buffers.
	const uint32_t families[] {graphics.queue_family, graphics.readback_family};
	if (graphics.readback_family != static_cast<uint32_t>(-1)) {
		buffer_info.sharingMode           = vk::SharingMode::eConcurrent;
		buffer_info.queueFamilyIndexCount = 2;
		buffer_info.pQueueFamilyIndices   = families;
	}

	// A buffer with device addresses is placed in VMA's blocks as any other (the allocator is made with buffer device
	// addresses; VMA itself gives a big one its own memory). Each its own allocation, as before, cost a kernel call:
	// ~100 us a buffer (2.4 us placed), 8400 buffers while a save loads and 100-400 in a frame where the game streams
	// a new area.
	const bool              with_bda = bool(flags & vk::BufferUsageFlagBits::eShaderDeviceAddress);
	// On a GPU short of video memory a GPU buffer asks for video memory whatever VMA's budget says, and goes elsewhere
	// only when the driver has none: NVIDIA counts what its system memory type holds, the textures put there
	// (GraphicContext::CreateImage), under the video memory heap's usage, so the budget sent buffers to system memory
	// while there was room (Shrine of Storms with 8 GB: 2 buffers, 140 MB).
	const bool ignore_budget = usage == MemoryUsage::DeviceLocal && graphics.small_video_memory;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags = (ignore_budget ? VmaAllocationCreateFlags {} : VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT) |
	                        AllocationFlags(usage);
	allocation_info.usage = AllocationUsage(usage);
	allocation_info.preferredFlags = usage == MemoryUsage::DeviceLocal
	                                     ? VkMemoryPropertyFlags {}
	                                     : VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	// KYTY_TEST_BUFFER_SPILL_MB=<n> (a test): the first game buffer (device-local, with device addresses and a guest
	// address) of at least n MiB is made in the memory types without video memory, as one is when video memory is
	// full, so that its move back can be watched (BufferCache::MoveSpilledBufferBack). (A GPU without such types: no
	// change.)
	const bool forced_spill =
	    usage == MemoryUsage::DeviceLocal && with_bda && cpu_address != 0 && TakeForcedSpill(size);
	if (forced_spill) {
		const auto& memory = graphics.physical_device_memory_properties;
		for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
			const auto properties = static_cast<VkMemoryPropertyFlags>(memory.memoryTypes[i].propertyFlags);
			if ((properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) allocation_info.memoryTypeBits |= 1u << i;
		}
		allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	}

	// A game buffer goes in the game buffers' pool (graphicContext.h), with no room there where VMA puts it.
	if (usage == MemoryUsage::DeviceLocal && with_bda && cpu_address != 0 && !forced_spill)
		allocation_info.pool = graphics.game_buffer_pool;

	// KYTY_TEST_BUFFER_FAIL_MB=<n> (a test): the first game buffer of at least n MiB fails its first 5 tries, as when
	// memory runs short, so that the retry (GraphicContext::RetryAllocation) can be watched.
	int forced_failures =
	    usage == MemoryUsage::DeviceLocal && with_bda && cpu_address != 0 && TakeAtLeast(g_forced_failure_bytes, size) ? 5 : 0;

	VmaAllocationInfo allocation_result {};
	VkBuffer          native_buffer = VK_NULL_HANDLE;
	const auto        create        = [&] {
		if (forced_failures > 0) {
			forced_failures--;
			return false;
		}
		return vmaCreateBuffer(graphics.allocator, static_cast<const VkBufferCreateInfo*>(buffer_info),
		                       &allocation_info, &native_buffer, &m_allocation, &allocation_result) == VK_SUCCESS;
	};
	bool created = create();
	if (!created && allocation_info.pool != nullptr) {
		allocation_info.pool = nullptr;
		created              = create();
	}
	// Over the video memory budget VMA places it in system memory; when even that fails, without
	// the images kept for reuse and over budget; then again for a while (issue #228).
	if (!created) {
		graphics.TrimImagePool();
		allocation_info.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
		if (!create()) {
			graphics.ReportMemoryFallback("a buffer could not be created", size);
			(void)graphics.RetryAllocation("a buffer", size, create);
		}
	}
	EXIT_NOT_IMPLEMENTED(native_buffer == VK_NULL_HANDLE);

	m_buffer = native_buffer;
	if (with_bda) {
		vk::BufferDeviceAddressInfo address_info {};
		address_info.buffer = m_buffer;
		m_device_address    = graphics.device.getBufferAddress(address_info);
		EXIT_IF(m_device_address == 0);
		DeviceFault::NoteCreated(m_device_address, size, cpu_address);
	}

	VkMemoryPropertyFlags properties = 0;
	vmaGetAllocationMemoryProperties(graphics.allocator, m_allocation, &properties);
	m_coherent = (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
	if (usage == MemoryUsage::DeviceLocal && (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) {
		m_in_system_memory = true;
		graphics.ReportMemoryFallback(forced_spill ? "a GPU buffer is in system memory (KYTY_TEST_BUFFER_SPILL_MB)"
		                                           : "a GPU buffer is in system memory",
		                              size);
	}
	if (allocation_result.pMappedData != nullptr) {
		m_mapped = {static_cast<uint8_t*>(allocation_result.pMappedData),
		            static_cast<size_t>(size)};
	}
}

Buffer::Buffer(GraphicContext& graphics, CommandScheduler& scheduler, uint64_t cpu_address, vk::BufferUsageFlags flags,
               uint64_t size, VmaAllocation allocation)
    : m_graphics(&graphics), m_scheduler(&scheduler), m_usage(MemoryUsage::DeviceLocal), m_cpu_address(cpu_address),
      m_size(size) {
	EXIT_IF(graphics.allocator == nullptr || size == 0 || allocation == nullptr);
	// (As the constructor above makes one: the same requirements, those the destination was reserved with.)
	vk::BufferCreateInfo buffer_info {};
	buffer_info.size  = size;
	buffer_info.usage = flags;
	const uint32_t families[] {graphics.queue_family, graphics.readback_family};
	if (graphics.readback_family != static_cast<uint32_t>(-1)) {
		buffer_info.sharingMode           = vk::SharingMode::eConcurrent;
		buffer_info.queueFamilyIndexCount = 2;
		buffer_info.pQueueFamilyIndices   = families;
	}
	const auto native_device = static_cast<VkDevice>(graphics.device);
	const auto raw           = static_cast<VkBufferCreateInfo>(buffer_info);
	VkBuffer   native_buffer = VK_NULL_HANDLE;
	EXIT_NOT_IMPLEMENTED(VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateBuffer(native_device, &raw, nullptr, &native_buffer) != VK_SUCCESS);
	EXIT_NOT_IMPLEMENTED(vmaBindBufferMemory(graphics.allocator, allocation, native_buffer) != VK_SUCCESS);
	m_buffer     = native_buffer;
	m_allocation = allocation;
	if (flags & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
		vk::BufferDeviceAddressInfo address_info {};
		address_info.buffer = m_buffer;
		m_device_address    = graphics.device.getBufferAddress(address_info);
		EXIT_IF(m_device_address == 0);
		DeviceFault::NoteCreated(m_device_address, size, cpu_address);
	}
}

Buffer::~Buffer() {
	DeviceFault::NoteDestroyed(m_device_address);
	if (m_buffer != nullptr) {
		if (kyty_local_buffer_reclaim_mode.load(std::memory_order_relaxed) != 0) {
			Reclaimer().Push(m_graphics->allocator, m_buffer, m_allocation);
		} else {
			vmaDestroyBuffer(m_graphics->allocator, m_buffer, m_allocation);
		}
	}
}

void FlushBufferReclaimer() {
	Reclaimer().Flush();
}

bool HoldBufferFrees() {
	return Reclaimer().TryHold();
}

void ReleaseBufferFrees() {
	Reclaimer().Release();
}

vk::DeviceAddress Buffer::BufferDeviceAddress() const noexcept {
	EXIT_IF(m_device_address == 0);
	return m_device_address;
}

bool Buffer::IsInBounds(uint64_t address, uint64_t size) const noexcept {
	return address >= m_cpu_address && size <= Size() && address - m_cpu_address <= Size() - size;
}

void Buffer::Write(uint64_t offset, const void* source, uint64_t size) {
	EXIT_IF(source == nullptr || m_mapped.empty() || offset > Size() || size > Size() - offset);
	std::memcpy(m_mapped.data() + offset, source, static_cast<size_t>(size));
	Flush(offset, size);
}

void Buffer::Flush(uint64_t offset, uint64_t size) {
	EXIT_IF(m_mapped.empty() || offset > Size() || size > Size() - offset);
	if (!IsCoherent() && size != 0) {
		const auto result =
		    vmaFlushAllocation(m_graphics->allocator, m_allocation, offset, size);
		EXIT_NOT_IMPLEMENTED(static_cast<vk::Result>(result) != vk::Result::eSuccess);
	}
}

void Buffer::Invalidate(uint64_t offset, uint64_t size) {
	EXIT_IF(m_mapped.empty() || offset > Size() || size > Size() - offset);
	if (!IsCoherent() && size != 0) {
		const auto result =
		    vmaInvalidateAllocation(m_graphics->allocator, m_allocation, offset, size);
		EXIT_NOT_IMPLEMENTED(static_cast<vk::Result>(result) != vk::Result::eSuccess);
	}
}

vk::BufferMemoryBarrier Buffer::Barrier(uint64_t offset, uint64_t size, vk::AccessFlags source,
                                        vk::AccessFlags destination) const {
	if (Handle() == nullptr || size == 0 || offset > Size() || size > Size() - offset) {
		EXIT("Buffer: invalid DMA barrier, handle=%p offset=0x%016" PRIx64 " size=0x%016" PRIx64
		     " capacity=0x%016" PRIx64 "\n",
		     static_cast<const void*>(Handle()), offset, size, Size());
	}
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = source;
	barrier.dstAccessMask       = destination;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = Handle();
	barrier.offset              = offset;
	barrier.size                = size;
	return barrier;
}

void Buffer::CopyFrom(CommandBuffer& command, const Buffer& source, uint64_t source_offset,
                      uint64_t destination_offset, uint64_t size, vk::AccessFlags source_before,
                      vk::AccessFlags destination_before, vk::AccessFlags source_after,
                      vk::AccessFlags destination_after) {
	if (size == 0 || source_offset > source.Size() || size > source.Size() - source_offset ||
	    destination_offset > Size() || size > Size() - destination_offset) {
		EXIT("Buffer: invalid copy range\n");
	}
	if (source.Handle() == Handle() && source_offset < destination_offset + size &&
	    destination_offset < source_offset + size) {
		EXIT("Buffer: overlapping self-copy\n");
	}
	written_serial = Scheduler().CommandSerial();
	command.EndRendering();
	const vk::BufferMemoryBarrier before[] = {
	    source.Barrier(source_offset, size, source_before, vk::AccessFlagBits::eTransferRead),
	    Barrier(destination_offset, size, destination_before, vk::AccessFlagBits::eTransferWrite),
	};
	const auto host_access  = vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eHostWrite;
	auto       before_stage = vk::PipelineStageFlags {vk::PipelineStageFlagBits::eAllCommands};
	if (static_cast<bool>((source_before | destination_before) & host_access)) {
		before_stage |= vk::PipelineStageFlagBits::eHost;
	}
	const auto native = command.Handle();
	native.pipelineBarrier(before_stage, vk::PipelineStageFlagBits::eTransfer,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 2, before, 0, nullptr);
	const vk::BufferCopy copy {source_offset, destination_offset, size};
	native.copyBuffer(source.Handle(), Handle(), 1, &copy);
	const vk::BufferMemoryBarrier after[] = {
	    source.Barrier(source_offset, size, vk::AccessFlagBits::eTransferRead, source_after),
	    Barrier(destination_offset, size, vk::AccessFlagBits::eTransferWrite, destination_after),
	};
	auto after_stage = vk::PipelineStageFlags {vk::PipelineStageFlagBits::eAllCommands};
	if (static_cast<bool>((source_after | destination_after) & host_access)) {
		after_stage |= vk::PipelineStageFlagBits::eHost;
	}
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, after_stage,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 2, after, 0, nullptr);
}

void Buffer::CopyInRun(CommandBuffer& command, const Buffer& source, uint64_t source_offset,
                       uint64_t destination_offset, uint64_t size) {
	if (size == 0 || source_offset > source.Size() || size > source.Size() - source_offset ||
	    destination_offset > Size() || size > Size() - destination_offset) {
		EXIT("Buffer: invalid copy range\n");
	}
	if (source.Handle() == Handle() && source_offset < destination_offset + size &&
	    destination_offset < source_offset + size) {
		EXIT("Buffer: overlapping self-copy\n");
	}
	written_serial = Scheduler().CommandSerial();
	command.EndRendering();
	const vk::BufferCopy copy {source_offset, destination_offset, size};
	command.CopyRunHandle(source.Handle(), source_offset, Handle(), destination_offset, size)
	    .copyBuffer(source.Handle(), Handle(), 1, &copy);
}

void Buffer::Fill(uint64_t offset, uint64_t size, uint32_t value) {
	if (((offset | size) & 3u) != 0) {
		EXIT("Buffer: fill range must be dword aligned\n");
	}
	auto& command = Scheduler().Current();
	written_serial = Scheduler().CommandSerial();
	command.EndRendering();
	const auto before =
	    Barrier(offset, size, vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
	            vk::AccessFlagBits::eTransferWrite);
	const auto native = command.Handle();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 1, &before, 0, nullptr);
	native.fillBuffer(Handle(), offset, size, value);
	const auto after = Barrier(offset, size, vk::AccessFlagBits::eTransferWrite,
	                           vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
}

StreamBuffer::StreamBuffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
                           uint64_t size, vk::BufferUsageFlags flags)
    : Buffer(graphics, scheduler, usage, 0, flags, size),
      m_current_watches(WATCHES_INITIAL_RESERVE), m_previous_watches(WATCHES_INITIAL_RESERVE) {}

bool StreamBuffer::NormalizeReservation(bool coherent, uint64_t atom, uint64_t& size,
                                        uint64_t& alignment) {
	if (coherent) {
		return true;
	}
	if (!AlignUp(size, atom, size)) {
		return false;
	}
	const auto divisor = std::gcd(alignment, atom);
	if (alignment != 0 && alignment / divisor > UINT64_MAX / atom) {
		return false;
	}
	alignment = alignment == 0 ? atom : alignment / divisor * atom;
	return true;
}

std::pair<uint8_t*, uint64_t> StreamBuffer::Map(uint64_t size, uint64_t alignment,
                                                bool allow_wait) {
	if (Mapped().empty()) {
		return {nullptr, 0};
	}
	uint64_t   mapped_size = size;
	const auto atom        = Graphics().physical_device_properties.limits.nonCoherentAtomSize;
	if (!NormalizeReservation(IsCoherent(), atom, mapped_size, alignment)) {
		return {nullptr, 0};
	}
	if (mapped_size > Size()) {
		NoteRefusal(mapped_size);
		return {nullptr, 0};
	}

	uint64_t aligned_offset = 0;
	if (!AlignUp(m_offset, alignment, aligned_offset)) {
		return {nullptr, 0};
	}

	const bool wrap = aligned_offset > Size() - mapped_size;
	if (wrap) {
		aligned_offset = 0;
	}

	auto wait_cursor = wrap ? size_t {0} : m_wait_cursor;
	auto wait_bound  = wrap ? uint64_t {0} : m_wait_bound;
	auto invalidation_mark =
	    wrap ? std::optional<size_t> {m_current_watch_cursor} : m_invalidation_mark;
	auto& pending_watches = wrap ? m_current_watches : m_previous_watches;
	if (!WaitPendingOperations(pending_watches, invalidation_mark, aligned_offset + mapped_size,
	                           allow_wait, wait_cursor, wait_bound)) {
		NoteRefusal(mapped_size);
		return {nullptr, 0};
	}

	if (wrap) {
		m_invalidation_mark    = invalidation_mark;
		m_current_watch_cursor = 0;
		std::swap(m_previous_watches, m_current_watches);
	}
	m_wait_cursor = wait_cursor;
	m_wait_bound  = wait_bound;
	m_offset      = aligned_offset;
	m_mapped_size = mapped_size;
	if (UseStatsOn() && Usage() == MemoryUsage::Upload) NoteUse(mapped_size, wrap);
	return {Mapped().data() + m_offset, m_offset};
}

// The bytes the GPU may still read: the reservations of ticks not done, a suffix of the last lap's
// watches (the earlier ones are done: ticks rise) and of this lap's, and the one just made.
uint64_t StreamBuffer::InFlight() {
	const auto first_busy = [this](const std::vector<Watch>& watches, size_t count) {
		size_t low = 0, high = count;
		while (low < high) {
			const size_t middle = (low + high) / 2;
			if (Scheduler().IsFree(watches[middle].tick)) {
				low = middle + 1;
			} else {
				high = middle;
			}
		}
		return low;
	};
	const auto busy_bytes = [&](const std::vector<Watch>& watches, size_t count) -> uint64_t {
		const size_t first = first_busy(watches, count);
		if (first >= count) return 0;
		return watches[count - 1].upper_bound - (first == 0 ? 0 : watches[first - 1].upper_bound);
	};
	uint64_t bytes = m_mapped_size;
	if (m_invalidation_mark.has_value())
		bytes += busy_bytes(m_previous_watches, *m_invalidation_mark);
	return bytes + busy_bytes(m_current_watches, m_current_watch_cursor);
}

void StreamBuffer::NoteRefusal(uint64_t size) {
	if (!UseStatsOn() || Usage() != MemoryUsage::Upload) return;
	m_use.refused++;
	m_use.refused_bytes += size;
}

void StreamBuffer::NoteBypass(uint64_t size) {
	if (!UseStatsOn() || Usage() != MemoryUsage::Upload) return;
	m_use.bypassed++;
	m_use.bypassed_bytes += size;
}

void StreamBuffer::NoteUse(uint64_t size, bool wrap) {
	using Clock    = std::chrono::steady_clock;
	const auto now = Clock::now();
	if (m_use.maps == 0) m_use.since = m_use.second = now;
	m_use.maps++;
	m_use.bytes += size;
	m_use.laps += wrap ? 1 : 0;
	m_use.largest   = std::max(m_use.largest, size);
	m_use.in_flight = std::max(m_use.in_flight, InFlight());
	if (now - m_use.second >= std::chrono::seconds(1)) {
		m_use.busiest      = std::max(m_use.busiest, m_use.second_bytes);
		m_use.second_bytes = 0;
		m_use.second       = now;
	}
	m_use.second_bytes += size;
	if (now - m_use.since < std::chrono::seconds(10)) return;
	m_use.busiest        = std::max(m_use.busiest, m_use.second_bytes);
	constexpr double mib = 1024.0 * 1024.0;
	std::printf(
	    "Upload ring %.0f MiB: %.0f MiB in %.0f s (%llu reservations, largest %.1f MiB, busiest "
	    "second %.0f MiB), %llu laps, most in flight %.1f MiB, %llu waits for the GPU (%.1f ms), "
	    "%llu refused (%.0f MiB), %llu around the ring (%.0f MiB)\n",
	    static_cast<double>(Size()) / mib, static_cast<double>(m_use.bytes) / mib,
	    std::chrono::duration<double>(now - m_use.since).count(),
	    static_cast<unsigned long long>(m_use.maps), static_cast<double>(m_use.largest) / mib,
	    static_cast<double>(m_use.busiest) / mib, static_cast<unsigned long long>(m_use.laps),
	    static_cast<double>(m_use.in_flight) / mib, static_cast<unsigned long long>(m_use.waits),
	    static_cast<double>(m_use.wait_us) / 1000.0, static_cast<unsigned long long>(m_use.refused),
	    static_cast<double>(m_use.refused_bytes) / mib,
	    static_cast<unsigned long long>(m_use.bypassed),
	    static_cast<double>(m_use.bypassed_bytes) / mib);
	std::fflush(stdout);
	m_use = {};
}

void StreamBuffer::Commit() {
	if (Usage() != MemoryUsage::Download && m_mapped_size != 0) {
		Flush(m_offset, m_mapped_size);
	}

	m_offset += m_mapped_size;
	const auto tick = Scheduler().CurrentTick();
	if (m_current_watch_cursor != 0 && m_current_watches[m_current_watch_cursor - 1].tick == tick) {
		m_current_watches[m_current_watch_cursor - 1].upper_bound = m_offset;
		return;
	}
	if (m_current_watch_cursor + 1 >= m_current_watches.size()) {
		m_current_watches.resize(m_current_watches.size() + WATCHES_RESERVE_CHUNK);
	}
	auto& watch       = m_current_watches[m_current_watch_cursor++];
	watch.upper_bound = m_offset;
	watch.tick        = tick;
}

void StreamBuffer::ResolvePendingTicks(uint64_t pending, uint64_t tick) noexcept {
	// The watches in use only (what waits reads: the current ones to the cursor, the previous to the invalidation
	// mark). Both vectors hold at least WATCHES_INITIAL_RESERVE: a whole scan of the two, for each of a speculation
	// thread's rings at each translation, was ~10% of that thread.
	const auto resolve = [&](std::vector<Watch>& watches, size_t count) {
		for (size_t i = 0, n = std::min(count, watches.size()); i < n; ++i)
			if (watches[i].tick == pending) watches[i].tick = tick;
	};
	resolve(m_current_watches, m_current_watch_cursor);
	if (m_invalidation_mark) resolve(m_previous_watches, *m_invalidation_mark);
}

uint64_t StreamBuffer::Copy(const void* source, uint64_t size, uint64_t alignment) {
	EXIT_IF(source == nullptr);
	const auto [data, offset] = Map(size, alignment);
	EXIT_IF(data == nullptr);
	std::memcpy(data, source, static_cast<size_t>(size));
	Commit();
	return offset;
}

bool StreamBuffer::WaitPendingOperations(const std::vector<Watch>& watches,
                                         std::optional<size_t>     invalidation_mark,
                                         uint64_t requested_upper_bound, bool allow_wait,
                                         size_t& wait_cursor, uint64_t& wait_bound) {
	if (!invalidation_mark.has_value()) {
		return true;
	}
	while (requested_upper_bound > wait_bound && wait_cursor < *invalidation_mark) {
		const auto& watch = watches[wait_cursor];
		const bool  busy  = !Scheduler().IsFree(watch.tick);
		if (busy && !allow_wait) {
			return false;
		}
		const bool timed = busy && UseStatsOn() && Usage() == MemoryUsage::Upload;
		const auto start =
		    timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		Scheduler().Wait(watch.tick);
		if (timed) {
			m_use.waits++;
			m_use.wait_us +=
			    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                              std::chrono::steady_clock::now() - start)
				                          .count());
		}
		if (Usage() == MemoryUsage::Download) {
			Scheduler().WaitPriorityOperations(watch.tick);
		}
		wait_bound = watch.upper_bound;
		++wait_cursor;
	}
	return true;
}

} // namespace Libs::Graphics
