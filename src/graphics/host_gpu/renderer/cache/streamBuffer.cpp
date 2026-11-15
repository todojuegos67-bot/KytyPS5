#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

namespace {

constexpr size_t WATCHES_INITIAL_RESERVE = 0x4000;
constexpr size_t WATCHES_RESERVE_CHUNK   = 0x1000;

constexpr vk::AccessFlags MEMORY_ACCESS =
    vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;

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
	const auto aligned = Common::AlignUp(value, alignment);
	if (aligned < value) {
		return false;
	}
	result = aligned;
	return true;
}

} // namespace

Buffer::Buffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
               uint64_t cpu_address, vk::BufferUsageFlags flags, uint64_t size, bool sparse_residency, bool zero_page)
    : m_graphics(&graphics), m_scheduler(&scheduler), m_usage(usage), m_cpu_address(cpu_address),
      m_size(size) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(graphics.allocator == nullptr || size == 0);
	if (sparse_residency) {
		EXIT_IF(usage != MemoryUsage::DeviceLocal);
		// A device that offers sparse buffers but fails here gets the dense buffer: page 0 is bound and unbound once.
		VmaAllocation probe = nullptr;
		const auto    sparse_works = [&] {
			vk::BufferCreateInfo sparse_info {};
			sparse_info.flags = vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency;
			if (zero_page) sparse_info.flags |= vk::BufferCreateFlagBits::eSparseAliased;
			sparse_info.size  = size;
			sparse_info.usage = flags;
			if (graphics.device.createBuffer(&sparse_info, nullptr, &m_buffer) != vk::Result::eSuccess) return false;
			graphics.device.getBufferMemoryRequirements(m_buffer, &m_page_requirements);
			m_page_size = m_page_requirements.alignment;
			if (m_page_size == 0) return false;
			// A buffer's sparse binds are whole blocks, its last one too (VUID-VkSparseMemoryBind-resourceOffset-09491): a guest
			// range that ends inside a block (a 1075 MiB buffer at 0x1184b58000, 59-fg-trask2) gets a Vulkan buffer to the block's end.
			m_sparse_size = Common::AlignUp(size, m_page_size);
			if (m_sparse_size != size) {
				graphics.device.destroyBuffer(m_buffer, nullptr);
				m_buffer         = nullptr;
				sparse_info.size = m_sparse_size;
				if (graphics.device.createBuffer(&sparse_info, nullptr, &m_buffer) != vk::Result::eSuccess) return false;
				graphics.device.getBufferMemoryRequirements(m_buffer, &m_page_requirements);
				if (m_page_requirements.alignment != m_page_size) return false;
			}
			// A game buffer (zero page) is backed in 2 MiB units: Marvel's Wolverine's geometry pool commits 2 MiB-aligned 2 MiB
			// pages, so it costs the same memory as 64 KiB units with 32 times fewer allocations and binds (43,776 binds of the
			// zero page at 64 KiB held the title back by ~9 s).
			if (zero_page) m_page_size = Common::AlignUp(std::max<uint64_t>(m_page_size, uint64_t {2} << 20u), m_page_size);
			vk::DeviceMemory memory;
			uint64_t         memory_offset = 0;
			probe = graphics.AllocateSparse(m_page_requirements, m_page_size, memory, memory_offset);
			if (probe == nullptr) return false;
			vk::Fence           fence;
			vk::FenceCreateInfo fence_info {};
			if (graphics.device.createFence(&fence_info, nullptr, &fence) != vk::Result::eSuccess) return false;
			bool ok = true;
			for (const vk::DeviceMemory bound: {memory, vk::DeviceMemory {}}) { // bind, then unbind
				const vk::SparseMemoryBind bind {0, std::min(m_page_size, m_sparse_size), bound, bound ? memory_offset : 0};
				const vk::SparseBufferMemoryBindInfo buffer_bind {m_buffer, 1, &bind};
				vk::BindSparseInfo                   info {};
				info.bufferBindCount = 1;
				info.pBufferBinds    = &buffer_bind;
				{
					Common::LockGuard lock(graphics.queue_mutex);
					ok = graphics.queue.bindSparse(1, &info, fence) == vk::Result::eSuccess;
				}
				ok = ok && graphics.device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX) == vk::Result::eSuccess &&
				     graphics.device.resetFences(1, &fence) == vk::Result::eSuccess;
				if (!ok) break;
			}
			graphics.device.destroyFence(fence, nullptr);
			return ok;
		};
		const bool sparse = sparse_works();
		if (!sparse && m_buffer != nullptr) {
			graphics.device.destroyBuffer(m_buffer, nullptr); // before its probe page goes: that may still be bound
			m_buffer = nullptr;
		}
		if (probe != nullptr) vmaFreeMemory(graphics.allocator, probe);
		if (sparse) {
			m_pages.assign((size + m_page_size - 1) / m_page_size, nullptr);
			if (flags & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
				vk::BufferDeviceAddressInfo address_info {};
				address_info.buffer = m_buffer;
				m_device_address    = graphics.device.getBufferAddress(address_info);
				EXIT_IF(m_device_address == 0);
			}
			if (zero_page) {
				// Every page on the one zero page (binds out before the next submit), then zeroed through page 0, which
				// reaches it: the page stays zero as long as nothing writes outside the backed pages.
				vk::DeviceMemory memory;
				uint64_t         memory_offset = 0;
				m_zero_page = graphics.AllocateSparse(m_page_requirements, m_page_size, memory, memory_offset);
				EXIT_NOT_IMPLEMENTED(m_zero_page == nullptr);
				for (uint64_t begin = 0; begin < m_sparse_size; begin += m_page_size) {
					m_scheduler->BindSparseBuffer(m_buffer,
					                             {begin, std::min(m_page_size, m_sparse_size - begin), memory, memory_offset});
				}
				Fill(0, std::min(m_page_size, size), 0);
			}
			return;
		}
		m_page_size = 0;
		LOGF("Sparse page table: a test bind failed, the table is a dense buffer\n");
	}

	vk::BufferCreateInfo buffer_info {};
	buffer_info.size        = size;
	buffer_info.usage       = flags;

	const bool with_bda = bool(flags & vk::BufferUsageFlagBits::eShaderDeviceAddress);
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags =
	    VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | AllocationFlags(usage);
	allocation_info.usage = AllocationUsage(usage);
	allocation_info.preferredFlags = usage == MemoryUsage::DeviceLocal
	                                     ? VkMemoryPropertyFlags {}
	                                     : VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	VmaAllocationInfo allocation_result {};
	VkBuffer          native_buffer = VK_NULL_HANDLE;
	const auto        result        = static_cast<vk::Result>(vmaCreateBuffer(
	    graphics.allocator, static_cast<const VkBufferCreateInfo*>(buffer_info), &allocation_info,
	    &native_buffer, &m_allocation, &allocation_result));
	if (result != vk::Result::eSuccess) {
		graphics.LogMemoryBudget();
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	m_buffer = native_buffer;
	if (with_bda) {
		vk::BufferDeviceAddressInfo address_info {};
		address_info.buffer = m_buffer;
		m_device_address    = graphics.device.getBufferAddress(address_info);
		EXIT_IF(m_device_address == 0);
	}

	VkMemoryPropertyFlags properties = 0;
	vmaGetAllocationMemoryProperties(graphics.allocator, m_allocation, &properties);
	m_coherent = (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
	if (allocation_result.pMappedData != nullptr) {
		m_mapped = {static_cast<uint8_t*>(allocation_result.pMappedData),
		            static_cast<size_t>(size)};
	}
}

Buffer::~Buffer() {
	if (m_buffer != nullptr && IsSparse()) {
		m_graphics->device.destroyBuffer(m_buffer, nullptr);
		for (auto* page: m_pages) {
			if (page != nullptr) vmaFreeMemory(m_graphics->allocator, page);
		}
		if (m_zero_page != nullptr) vmaFreeMemory(m_graphics->allocator, m_zero_page);
	} else if (m_buffer != nullptr) {
		vmaDestroyBuffer(m_graphics->allocator, m_buffer, m_allocation);
	}
}

void Buffer::Back(uint64_t offset, uint64_t size) {
	if (!IsSparse() || size == 0) return;
	EXIT_IF(offset > m_size || size > m_size - offset);
	// Bound memory starts undefined; this submit waits for the binds. One fill per run of new pages.
	uint64_t run_begin = 0;
	uint64_t run_end   = 0;
	for (uint64_t page = offset / m_page_size, last = (offset + size - 1) / m_page_size; page <= last; ++page) {
		if (m_pages[page] != nullptr) continue;
		const uint64_t   begin = page * m_page_size;
		const uint64_t   bytes = std::min(m_page_size, m_sparse_size - begin); // whole blocks: the last page may be shorter
		vk::DeviceMemory memory;
		uint64_t         memory_offset = 0;
		m_pages[page] = m_graphics->AllocateSparse(m_page_requirements, m_page_size, memory, memory_offset);
		EXIT_NOT_IMPLEMENTED(m_pages[page] == nullptr);
		m_backed_pages++;
		m_scheduler->BindSparseBuffer(m_buffer, {begin, bytes, memory, memory_offset});
		if (run_end != begin) {
			if (run_end != run_begin) Fill(run_begin, run_end - run_begin, 0);
			run_begin = begin;
		}
		run_end = begin + std::min(bytes, m_size - begin); // the fill stays in the guest range
	}
	if (run_end != run_begin) Fill(run_begin, run_end - run_begin, 0);
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
	EXIT_IF(m_usage != MemoryUsage::Download || offset > Size() || size > Size() - offset);
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
                      uint64_t destination_offset, uint64_t size) {
	if (size == 0 || source_offset > source.Size() || size > source.Size() - source_offset ||
	    destination_offset > Size() || size > Size() - destination_offset) {
		EXIT("Buffer: invalid copy range\n");
	}
	if (source.Handle() == Handle() && source_offset < destination_offset + size &&
	    destination_offset < source_offset + size) {
		EXIT("Buffer: overlapping self-copy\n");
	}
	command.EndRendering();
	const vk::BufferMemoryBarrier before[] = {
	    source.Barrier(source_offset, size, vk::AccessFlagBits::eMemoryWrite,
	                   vk::AccessFlagBits::eTransferRead),
	    Barrier(destination_offset, size, MEMORY_ACCESS, vk::AccessFlagBits::eTransferWrite),
	};
	const auto native = command.Recorder();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 2, before, 0, nullptr);
	const vk::BufferCopy copy {source_offset, destination_offset, size};
	native.copyBuffer(source.Handle(), Handle(), 1, &copy);
	const vk::BufferMemoryBarrier after[] = {
	    source.Barrier(source_offset, size, vk::AccessFlagBits::eTransferRead, MEMORY_ACCESS),
	    Barrier(destination_offset, size, vk::AccessFlagBits::eTransferWrite, MEMORY_ACCESS),
	};
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 2, after, 0, nullptr);
}

void Buffer::Fill(uint64_t offset, uint64_t size, uint32_t value) {
	if (((offset | size) & 3u) != 0) {
		EXIT("Buffer: fill range must be dword aligned\n");
	}
	auto& command = Scheduler().Current();
	command.EndRendering();
	const auto before = Barrier(offset, size, MEMORY_ACCESS, vk::AccessFlagBits::eTransferWrite);
	const auto native = command.Recorder();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 1, &before, 0, nullptr);
	native.fillBuffer(Handle(), offset, size, value);
	const auto after = Barrier(offset, size, vk::AccessFlagBits::eTransferWrite, MEMORY_ACCESS);
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
}

StreamBuffer::StreamBuffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
                           uint64_t size)
    : Buffer(graphics, scheduler, usage, 0,
             // A mesh draw's draw dwords are read by device address from the stream ring.
             usage == MemoryUsage::Stream ? AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress
                                          : AllFlags,
             size),
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
	return {Mapped().data() + m_offset, m_offset};
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

StreamBuffer::Download StreamBuffer::MapDownload(uint64_t size, uint64_t alignment) {
	if (const auto [mapped, offset] = Map(size, alignment); mapped != nullptr) {
		Commit();
		return {this, mapped, offset, nullptr};
	}
	auto owner = std::make_unique<Buffer>(Graphics(), Scheduler(), Usage(), 0, AllFlags, size);
	return {owner.get(), owner->Mapped().data(), 0, std::move(owner)};
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
		if (!Scheduler().IsFree(watch.tick) && !allow_wait) {
			return false;
		}
		Scheduler().Wait(watch.tick);
		if (Usage() == MemoryUsage::Download) {
			Scheduler().WaitPriorityOperations(watch.tick);
		}
		wait_bound = watch.upper_bound;
		++wait_cursor;
	}
	return true;
}

} // namespace Libs::Graphics
