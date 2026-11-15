#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "common/upstreamPaths.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {
namespace {

constexpr uint32_t   DescriptorHeapCount = 1024;
constexpr std::array DescriptorPoolSizes = {
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 1024},
};

template <typename Pred, typename Fn>
void ForEachRun(size_t begin, size_t end, Pred&& pred, Fn&& fn) {
	while (begin < end) {
		if (!pred(begin)) {
			++begin;
			continue;
		}
		auto run = begin + 1u;
		while (run < end && pred(run)) ++run;
		fn(begin, run);
		begin = run;
	}
}

// DiffImageHeap over the entries of words [first, last): slot 0 is never an entry.
void DiffEntries(std::span<const uint32_t> guest, std::vector<uint32_t>& mirror, size_t first, size_t last,
                 std::vector<ImageHeapChange>& changes) {
	for (size_t word = std::max<size_t>(8u, first & ~size_t {7}); word < last && word + 8 <= guest.size(); word += 8) {
		if (std::memcmp(&guest[word], &mirror[word], 32) == 0) continue;
		ImageHeapChange change {.slot = static_cast<uint32_t>(word / 8)};
		// One read: the game may be writing the entry, and the next diff then sees the rest.
		std::memcpy(change.descriptor.data(), &guest[word], 32);
		std::memcpy(&mirror[word], change.descriptor.data(), 32);
		changes.push_back(change);
	}
}

uint64_t NowMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

} // namespace

GuestImageHeap::GuestImageHeap(GraphicContext& graphics, PageManager& page_manager)
    : m_graphics(graphics), m_page_manager(page_manager) {
	namespace IR = ShaderRecompiler::IR;
	if (const char* mode = std::getenv("KYTY_HEAP_WATCH"); mode != nullptr && mode[0] >= '0' && mode[0] <= '2') {
		m_watch_mode = mode[0] - '0';
	}
	if (!graphics.guest_image_heap_enabled) return;
	LOGF("Image heap: page watch %s\n", m_watch_mode == 0 ? "off" : m_watch_mode == 2 ? "on, verify" : "on");
	std::vector<vk::DescriptorSetLayoutBinding> bindings;
	std::vector<vk::DescriptorBindingFlags>     flags;
	for (const auto numeric_class: {Prospero::TextureNumericClass::Float, Prospero::TextureNumericClass::Uint,
	                                Prospero::TextureNumericClass::Sint}) {
		for (const auto dimension: {ShaderRecompiler::Decoder::ImageDimension::Dim2D,
		                            ShaderRecompiler::Decoder::ImageDimension::Dim2DArray,
		                            ShaderRecompiler::Decoder::ImageDimension::Dim3D,
		                            ShaderRecompiler::Decoder::ImageDimension::Dim1D}) {
			IR::ImageResource image;
			image.resource_class = IR::ImageResourceClass::Sampled;
			image.numeric_class  = numeric_class;
			image.dimension      = dimension;
			bindings.push_back({IR::ImageBindingIndex(*IR::DescriptorBindingForImage(image)),
			                    vk::DescriptorType::eSampledImage, IR::ImageHeapSlots,
			                    vk::ShaderStageFlagBits::eAll, nullptr});
			flags.push_back(vk::DescriptorBindingFlagBits::eUpdateAfterBind |
			                vk::DescriptorBindingFlagBits::ePartiallyBound |
			                vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending);
		}
	}
	const auto image_count = static_cast<uint32_t>(bindings.size()) * IR::ImageHeapSlots;
	bindings.push_back({IR::ImageHeapInfoBinding, vk::DescriptorType::eStorageBuffer, 1,
	                    vk::ShaderStageFlagBits::eAll, nullptr});
	flags.push_back({});
	for (const auto binding: {IR::ImageHeapFloatSamplerBinding, IR::ImageHeapIntegerSamplerBinding}) {
		bindings.push_back({binding, vk::DescriptorType::eSampler, IR::ImageHeapSamplerSlots,
		                    vk::ShaderStageFlagBits::eAll, nullptr});
		flags.push_back(vk::DescriptorBindingFlagBits::eUpdateAfterBind |
		                vk::DescriptorBindingFlagBits::ePartiallyBound |
		                vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending);
	}
	vk::DescriptorSetLayoutBindingFlagsCreateInfo binding_flags {};
	binding_flags.bindingCount  = static_cast<uint32_t>(flags.size());
	binding_flags.pBindingFlags = flags.data();
	vk::DescriptorSetLayoutCreateInfo layout {};
	layout.pNext        = &binding_flags;
	layout.flags        = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
	layout.bindingCount = static_cast<uint32_t>(bindings.size());
	layout.pBindings    = bindings.data();
	const std::array sizes {vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, image_count},
	                        vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, 1},
	                        vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 2u * IR::ImageHeapSamplerSlots}};
	vk::DescriptorPoolCreateInfo pool {};
	pool.flags         = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
	pool.maxSets       = 1;
	pool.poolSizeCount = static_cast<uint32_t>(sizes.size());
	pool.pPoolSizes    = sizes.data();
	vk::BufferCreateInfo buffer {};
	buffer.size  = IR::ImageHeapSlots * sizeof(uint32_t);
	buffer.usage = vk::BufferUsageFlagBits::eStorageBuffer;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	allocate.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo info {};
	const auto        raw = static_cast<VkBufferCreateInfo>(buffer);
	vk::DescriptorSetAllocateInfo allocate_set {};
	allocate_set.descriptorSetCount = 1;
	allocate_set.pSetLayouts        = &graphics.guest_image_heap_layout;
	EXIT_IF(graphics.device.createDescriptorSetLayout(&layout, nullptr, &graphics.guest_image_heap_layout) !=
	            vk::Result::eSuccess ||
	        graphics.device.createDescriptorPool(&pool, nullptr, &m_pool) != vk::Result::eSuccess ||
	        (allocate_set.descriptorPool = m_pool,
	         graphics.device.allocateDescriptorSets(&allocate_set, &m_set)) != vk::Result::eSuccess ||
	        vmaCreateBuffer(graphics.allocator, &raw, &allocate, &m_info_buffer, &m_info_allocation, &info) !=
	            VK_SUCCESS);
	m_info = static_cast<uint32_t*>(info.pMappedData);
	std::fill_n(m_info, IR::ImageHeapSlots, 0u);
	m_images.resize(IR::ImageHeapSlots);
	const vk::DescriptorBufferInfo info_range {vk::Buffer(m_info_buffer), 0, VK_WHOLE_SIZE};
	vk::WriteDescriptorSet         write {};
	write.dstSet          = m_set;
	write.dstBinding      = IR::ImageHeapInfoBinding;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &info_range;
	graphics.device.updateDescriptorSets(1, &write, 0, nullptr);
	graphics.guest_image_heap = this;
}

GuestImageHeap::~GuestImageHeap() {
	if (m_set == nullptr) return;
	m_graphics.guest_image_heap = nullptr;
	m_graphics.device.destroyDescriptorPool(m_pool, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_graphics.guest_image_heap_layout, nullptr);
	m_graphics.guest_image_heap_layout = nullptr;
	vmaDestroyBuffer(m_graphics.allocator, m_info_buffer, m_info_allocation);
}

void GuestImageHeap::Watch(uint64_t base, uint64_t size, const RenderContext& context) {
	// Only pages whose write faults reach RenderContext::HandleFault are watched; else Diff compares all of the heap.
	// shortcut: checked when the heap is found or grows, not when its memory is unmapped later (one heap that lives
	// as long as the game, in Marvel's Wolverine); unmapping it while watched would leave a fault nobody handles.
	const auto mappable = [&] {
		const auto origin = Common::AlignDown(base, TRACKER_PAGE_SIZE);
		const bool mapped = context.IsMapped(origin, Common::AlignUp(base + size, TRACKER_PAGE_SIZE) - origin);
		if (!mapped && m_mappable.load(std::memory_order_relaxed)) LOGF("Image heap: not GPU-mapped, page watch off\n");
		m_mappable.store(mapped, std::memory_order_release);
	};
	if (m_base.load(std::memory_order_relaxed) == base) {
		// The same heap: a draw's table descriptor covers the slots in use so far, which grow.
		if (size > m_size.load(std::memory_order_relaxed)) {
			mappable();
			m_size.store(size, std::memory_order_release);
		}
		return;
	}
	mappable();
	// shortcut: one heap at a time; a game with several heaps would make them take turns here.
	for (uint32_t slot = 1; slot < m_images.size(); ++slot) {
		if (m_images[slot].image || m_info[slot] != 0u) Clear(slot);
		m_images[slot].waiting = false;
		m_images[slot].misses  = 0;
	}
	m_pending.clear();
	m_waiting.clear();
	m_lost.clear();
	LOGF("Image heap: the game's heap at 0x%016" PRIx64 ", %" PRIu64 " bytes\n", base, size);
	m_size.store(size, std::memory_order_release);
	m_base.store(base, std::memory_order_release);
}

std::vector<ImageHeapChange> GuestImageHeap::Diff() {
	const auto base = m_base.load(std::memory_order_acquire);
	if (base == 0) return {};
	const auto      words = std::min<uint64_t>(m_size.load(std::memory_order_acquire), m_images.size() * 32u) / 4u;
	std::lock_guard lock(m_guest_mutex);
	if (m_guest_base != base) {
		// shortcut: the first submission after the heap is found is one late; its draws read
		// empty slots until the next one.
		Unwatch(); // the old heap's pages
		m_guest_base = base;
		m_guest.assign(words, 0u);
	} else if (m_guest.size() < words) {
		m_guest.resize(words, 0u); // the heap grew: its new entries are reported, the others kept
	}
	const std::span<const uint32_t> guest {reinterpret_cast<const uint32_t*>(base), m_guest.size()};
	if (m_watch_mode == 0 || !m_mappable.load(std::memory_order_acquire)) {
		Unwatch();
		return ShaderRecompiler::IR::DiffImageHeap(guest, m_guest);
	}
	// Watch before reading, so a write during or after the read faults its page again (ImageHeap::Refresh).
	const auto origin = Common::AlignDown(base, TRACKER_PAGE_SIZE);
	const auto pages  = (Common::AlignUp(base + guest.size_bytes(), TRACKER_PAGE_SIZE) - origin) / TRACKER_PAGE_SIZE;
	m_runs.clear();
	{
		std::lock_guard watch(m_watch_mutex);
		if (!m_watching || m_watch_origin != origin) {
			m_dirty_pages.assign(pages, uint8_t {1});
			m_watch_origin = origin;
			m_watching     = true;
		} else if (m_watched_bytes < guest.size_bytes()) {
			// The heap grew: its new pages start unwatched, and so does the page its old end shares with them, whose new
			// entries no diff has read (the first verify stand counted them as 49 misses).
			const auto edge = (base + m_watched_bytes - origin) / TRACKER_PAGE_SIZE;
			if (edge < m_dirty_pages.size() && m_dirty_pages[edge] == 0u) {
				m_dirty_pages[edge] = 1u;
				m_page_manager.UpdatePageWatchers<false>(origin + edge * TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE);
			}
			m_dirty_pages.resize(pages, uint8_t {1});
		}
		m_watched_bytes = guest.size_bytes();
		// Before the first watch: a fault on a watched page must find the heap's bytes.
		m_watch_begin.store(origin, std::memory_order_release);
		m_watch_end.store(origin + pages * TRACKER_PAGE_SIZE, std::memory_order_release);
		ForEachRun(
		    0, pages, [&](size_t page) { return m_dirty_pages[page] != 0u; },
		    [&](size_t begin, size_t end) {
			    std::fill(m_dirty_pages.begin() + begin, m_dirty_pages.begin() + end, uint8_t {0});
			    m_page_manager.UpdatePageWatchers<true>(origin + begin * TRACKER_PAGE_SIZE, (end - begin) * TRACKER_PAGE_SIZE);
			    m_runs.emplace_back(begin, end);
		    });
	}
	std::vector<ImageHeapChange> changes;
	for (const auto& [begin, end]: m_runs) {
		const auto first = (std::max(base, origin + begin * TRACKER_PAGE_SIZE) - base) / 4u;
		const auto last  = (std::min(base + guest.size_bytes(), origin + end * TRACKER_PAGE_SIZE) - base) / 4u;
		// An entry across a page edge is compared whole.
		DiffEntries(guest, m_guest, first, Common::AlignUp(last, uint64_t {8}), changes);
		m_stat_pages += end - begin;
	}
	++m_stat_diffs;
	m_stat_of += pages;
	if (m_watch_mode == 2) {
		// Verify: what the whole diff still finds was written without a fault (a miss) or after the watch (late).
		const auto extra = ShaderRecompiler::IR::DiffImageHeap(guest, m_guest);
		std::lock_guard watch(m_watch_mutex);
		for (const auto& change: extra) {
			const auto address = base + change.slot * 32ull;
			const auto page    = (address - origin) / TRACKER_PAGE_SIZE;
			const auto page2   = (address + 31u - origin) / TRACKER_PAGE_SIZE;
			const bool late    = m_dirty_pages[page] != 0u || (page2 < m_dirty_pages.size() && m_dirty_pages[page2] != 0u);
			if (!late && m_stat_misses < 8u) {
				LOGF("Image heap watch: miss at slot %u (0x%016" PRIx64 ")\n", change.slot, address);
			}
			++(late ? m_stat_late : m_stat_misses);
			changes.push_back(change);
		}
	}
	if (const auto now = NowMs(); m_stat_start == 0) {
		m_stat_start = now;
	} else if (now - m_stat_start >= 5000u) {
		LOGF("Image heap watch (5 s): %" PRIu64 " diffs, %" PRIu64 " pages compared of %" PRIu64 ", %" PRIu64
		     " faults, %" PRIu64 " late, %" PRIu64 " misses\n",
		     m_stat_diffs, m_stat_pages, m_stat_of, m_stat_faults.exchange(0, std::memory_order_relaxed), m_stat_late,
		     m_stat_misses);
		m_stat_start = now;
		m_stat_diffs = m_stat_pages = m_stat_of = m_stat_late = 0; // misses stay summed
	}
	return changes;
}

void GuestImageHeap::Unwatch() {
	std::lock_guard watch(m_watch_mutex);
	if (!m_watching) return;
	m_watch_begin.store(0, std::memory_order_release);
	m_watch_end.store(0, std::memory_order_release);
	ForEachRun(
	    0, m_dirty_pages.size(), [&](size_t page) { return m_dirty_pages[page] == 0u; },
	    [&](size_t begin, size_t end) {
		    m_page_manager.UpdatePageWatchers<false>(m_watch_origin + begin * TRACKER_PAGE_SIZE,
		                                             (end - begin) * TRACKER_PAGE_SIZE);
	    });
	m_dirty_pages.clear();
	m_watching = false;
}

void GuestImageHeap::InvalidateMemory(uint64_t address, uint64_t size) {
	if (address >= m_watch_end.load(std::memory_order_acquire) ||
	    address + size <= m_watch_begin.load(std::memory_order_acquire)) {
		return;
	}
	std::lock_guard watch(m_watch_mutex);
	if (!m_watching) return;
	const auto end   = m_watch_origin + m_dirty_pages.size() * TRACKER_PAGE_SIZE;
	const auto first = std::max(address, m_watch_origin);
	const auto last  = std::min(address + size, end);
	if (first >= last) return;
	m_stat_faults.fetch_add(1, std::memory_order_relaxed);
	ForEachRun(
	    (first - m_watch_origin) / TRACKER_PAGE_SIZE, (last - 1u - m_watch_origin) / TRACKER_PAGE_SIZE + 1u,
	    [&](size_t page) { return m_dirty_pages[page] == 0u; },
	    [&](size_t begin, size_t stop) {
		    std::fill(m_dirty_pages.begin() + begin, m_dirty_pages.begin() + stop, uint8_t {1});
		    m_page_manager.UpdatePageWatchers<false>(m_watch_origin + begin * TRACKER_PAGE_SIZE,
		                                             (stop - begin) * TRACKER_PAGE_SIZE);
	    });
}

void GuestImageHeap::Queue(std::vector<ImageHeapChange>&& changes) {
	// The game wrote the entry again: it is resolved as new.
	for (const auto& change: changes) {
		m_images.at(change.slot).waiting = false;
		m_images.at(change.slot).misses  = 0;
	}
	if (m_pending.empty()) {
		m_pending = std::move(changes);
	} else {
		m_pending.insert(m_pending.end(), changes.begin(), changes.end());
	}
}

void GuestImageHeap::Stale(Common::SlotId image) {
	// Write faults call this on other threads: only the image is noted, TakeWork finds its slots.
	std::lock_guard lock(m_stale_mutex);
	if (std::ranges::find(m_stale, image) == m_stale.end()) m_stale.push_back(image);
}

void GuestImageHeap::Wait(uint32_t slot, uint64_t address) {
	Clear(slot);
	m_images.at(slot).waiting = true;
	auto& slots = m_waiting[address];
	if (std::ranges::find(slots, slot) == slots.end()) slots.push_back(slot);
}

void GuestImageHeap::Arrived(uint64_t address) {
	const auto found = m_waiting.find(address);
	if (found == m_waiting.end()) return;
	for (const auto slot: found->second) {
		// A slot whose entry changed since is resolved by that change.
		if (m_images[slot].waiting && m_images[slot].misses < 4u) {
			++m_images[slot].misses;
			Retry(slot);
		}
	}
	m_waiting.erase(found);
}

void GuestImageHeap::Retry(uint32_t slot) {
	++m_retried;
	m_pending.push_back({slot, m_images.at(slot).descriptor});
}

std::vector<ImageHeapChange> GuestImageHeap::TakeWork(std::span<const Common::SlotId> targets) {
	if (m_frame >= m_lost_frame + 30u) {
		m_lost_frame = m_frame;
		RetryLost();
	}
	auto work = std::move(m_pending);
	m_pending.clear();
	last_retried = std::exchange(m_retried, 0u);
	const auto changed = work.size();
	for (const auto& change: work) m_images[change.slot].descriptor = change.descriptor;
	std::vector<Common::SlotId> stale;
	{
		std::lock_guard lock(m_stale_mutex);
		stale.swap(m_stale);
	}
	for (const auto image: stale) {
		if (std::ranges::find(targets, image) != targets.end()) {
			std::lock_guard lock(m_stale_mutex);
			if (std::ranges::find(m_stale, image) == m_stale.end()) m_stale.push_back(image);
			continue;
		}
		const auto found = m_slots.find(image.index);
		if (found == m_slots.end()) continue;
		for (const auto slot: found->second) {
			if (m_images[slot].image == image) work.push_back({slot, m_images[slot].descriptor});
		}
	}
	last_stale = static_cast<uint32_t>(work.size() - changed);
	return work;
}

std::vector<std::pair<uint32_t, std::array<uint32_t, 4>>> GuestImageHeap::SamplerChanges(uint64_t base, uint64_t size) {
	namespace IR     = ShaderRecompiler::IR;
	const auto words = std::min<uint64_t>(size, uint64_t {IR::ImageHeapSamplerSlots} * 16u) / 4u;
	if (m_sampler_base != base || m_samplers.size() != words) {
		m_sampler_base = base;
		// Nothing matches it, so every record is written once, an all-zero one too.
		m_samplers.assign(words, 0xffffffffu);
	}
	// shortcut: read by the GPU thread when a draw reads it, not as of the submission (the image
	// heap's diff); the game makes its samplers at start-up, so a change in flight is not expected.
	const auto* guest = reinterpret_cast<const uint32_t*>(base);
	std::vector<std::pair<uint32_t, std::array<uint32_t, 4>>> changes;
	for (size_t word = 0; word + 4 <= words; word += 4) {
		if (std::memcmp(&guest[word], &m_samplers[word], 16) == 0) continue;
		std::array<uint32_t, 4> record {};
		std::memcpy(record.data(), &guest[word], 16);
		std::memcpy(&m_samplers[word], record.data(), 16);
		changes.emplace_back(static_cast<uint32_t>(word / 4), record);
	}
	return changes;
}

void GuestImageHeap::ShowSampler(uint32_t slot, vk::Sampler float_sampler, vk::Sampler integer_sampler) {
	namespace IR = ShaderRecompiler::IR;
	EXIT_IF(slot >= IR::ImageHeapSamplerSlots);
	const std::array             infos {vk::DescriptorImageInfo {float_sampler, nullptr, {}},
	                                    vk::DescriptorImageInfo {integer_sampler, nullptr, {}}};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t integer = 0; integer < 2u; ++integer) {
		writes[integer].dstSet          = m_set;
		writes[integer].dstBinding      = integer != 0u ? IR::ImageHeapIntegerSamplerBinding : IR::ImageHeapFloatSamplerBinding;
		writes[integer].dstArrayElement = slot;
		writes[integer].descriptorCount = 1;
		writes[integer].descriptorType  = vk::DescriptorType::eSampler;
		writes[integer].pImageInfo      = &infos[integer];
	}
	WriteDescriptor(writes.data(), static_cast<uint32_t>(writes.size()));
}

void GuestImageHeap::WriteInfo(uint32_t slot, uint32_t info) {
	if (m_scheduler == nullptr || !m_scheduler->Threaded()) {
		m_info[slot] = info;
		return;
	}
	struct P {
		uint32_t* where;
		uint32_t  value;
	};
	m_scheduler->Recorder().PutBack(P {m_info + slot, info}, [](CmdBack&, const P& p) { *p.where = p.value; });
}

void GuestImageHeap::WriteDescriptor(const vk::WriteDescriptorSet* writes, uint32_t count) {
	if (m_scheduler == nullptr || !m_scheduler->Threaded()) {
		m_graphics.device.updateDescriptorSets(count, writes, 0, nullptr);
		return;
	}
	m_scheduler->Recorder().UpdateDescriptorSets(m_graphics.device, count, writes);
}

bool GuestImageHeap::Holds(Common::SlotId image) const {
	const auto found = m_slots.find(image.index);
	return found != m_slots.end() &&
	       std::ranges::any_of(found->second, [&](uint32_t slot) { return m_images[slot].image == image; });
}

// shortcut: a slot is rewritten (view, info) when commands are recorded, so a command buffer
// still pending, or a draw recorded earlier in the current one, can read the slot's new texture
// (or the null image) instead of its old one, and a pending one breaks Vulkan's rule for update-
// after-bind descriptors. Slots change 10-50 times in 300 frames at Marvel's Wolverine's title;
// waiting for the GPU on each would stall. Double-buffer the set if a glitch or a fault shows it.
void GuestImageHeap::Show(uint32_t slot, uint32_t binding, Common::SlotId image, vk::ImageView view,
                     vk::ImageLayout layout, uint32_t info) {
	// A null view is a native null descriptor.
	EXIT_IF(slot >= m_images.size());
	auto& shown = m_shown[binding];
	if (shown.empty()) shown.resize(m_images.size());
	if (shown[slot].view != view || shown[slot].layout != layout) {
		const vk::DescriptorImageInfo image_info {nullptr, view, layout};
		vk::WriteDescriptorSet        write {};
		write.dstSet          = m_set;
		write.dstBinding      = binding;
		write.dstArrayElement = slot;
		write.descriptorCount = 1;
		write.descriptorType  = vk::DescriptorType::eSampledImage;
		write.pImageInfo      = &image_info;
		WriteDescriptor(&write, 1);
		shown[slot] = {view, layout};
	}
	if (image) m_images[slot].misses = 0;
	if (m_images[slot].image != image) {
		m_images[slot].image = image;
		// A slot emptied and shown again with the same image is listed once.
		if (image) {
			auto& slots = m_slots[image.index];
			if (std::ranges::find(slots, slot) == slots.end()) slots.push_back(slot);
		}
	}
	WriteInfo(slot, info);
}

void GuestImageHeap::Clear(uint32_t slot) {
	EXIT_IF(slot >= m_images.size());
	WriteInfo(slot, 0u);
	m_images[slot].image = {};
	// Its image may go now; a later view may get the same handle.
	for (auto& [binding, shown]: m_shown) {
		if (!shown.empty()) shown[slot] = {};
	}
}

void GuestImageHeap::Forget(Common::SlotId image) {
	const auto found = m_slots.find(image.index);
	if (found == m_slots.end()) return;
	const auto slots = std::move(found->second);
	m_slots.erase(found);
	for (const auto slot: slots) {
		if (m_images[slot].image != image) continue;
		ShaderTextureResource entry {};
		std::memcpy(&entry, m_images[slot].descriptor.data(), sizeof(entry));
		Wait(slot, entry.Base40());
		m_lost.try_emplace(slot, uint8_t {0});
	}
}

void GuestImageHeap::RetryLost() {
	// KYTY_HEAP_WAIT_ANY: slots wait as before the fire-light fix (only an image registered at their address retries).
	static const bool wait_any = std::getenv("KYTY_HEAP_WAIT_ANY") != nullptr;
	if (wait_any) {
		m_lost.clear();
		return;
	}
	for (auto it = m_lost.begin(); it != m_lost.end();) {
		const auto& record = m_images[it->first];
		if (record.image || !record.waiting || ++it->second > 40u) {
			it = m_lost.erase(it);
			continue;
		}
		Retry(it->first);
		++it;
	}
}

DescriptorHeap::DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore)
    : m_graphics(graphics), m_master_semaphore(master_semaphore) {
	CreateDescriptorPool();
	CreateKeptPool();
}

void DescriptorHeap::CreateKeptPool() {
	// Kept group sets: an effect program's stage holds up to about 25,000 candidates (Marvel's Wolverine, 2026-11-07);
	// the sets in use, one a program and stage, and those released while the GPU may still read them.
	constexpr uint32_t                  KeptDescriptors = 1u << 18u;
	const std::array<vk::DescriptorPoolSize, 2> kept_sizes {vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, KeptDescriptors},
	                                                        vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, KeptDescriptors}};
	vk::DescriptorPoolCreateInfo kept {};
	kept.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
	kept.maxSets       = 256;
	kept.poolSizeCount = static_cast<uint32_t>(kept_sizes.size());
	kept.pPoolSizes    = kept_sizes.data();
	EXIT_IF(m_graphics.device.createDescriptorPool(&kept, nullptr, &m_kept_pool) != vk::Result::eSuccess);
}

DescriptorHeap::~DescriptorHeap() {
	m_graphics.device.destroyDescriptorPool(m_current_pool, nullptr);
	if (!m_kept_released.empty()) m_master_semaphore.Wait(m_kept_released.back().second);
	m_graphics.device.destroyDescriptorPool(m_kept_pool, nullptr);
	for (const auto& [pool, tick]: m_kept_retired) {
		m_master_semaphore.Wait(tick);
		m_graphics.device.destroyDescriptorPool(pool, nullptr);
	}
	m_graphics.device.destroyDescriptorSetLayout(m_graphics.empty_set_layout, nullptr);
	m_graphics.empty_set_layout = nullptr;
	for (const auto& [pool, tick]: m_pending_pools) {
		m_master_semaphore.Wait(tick);
		m_graphics.device.destroyDescriptorPool(pool, nullptr);
	}
}

vk::DescriptorSet DescriptorHeap::Commit(vk::DescriptorSetLayout layout) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF((layout == nullptr) && !KYTY_UPSTREAM_PATH("86139e36f"));

	auto& batch = m_sets[layout];
	if (batch.size != 0) {
		return batch.sets[--batch.size];
	}
	if (Allocate(layout, batch)) {
		return batch.sets[--batch.size];
	}

	m_pending_pools.emplace_back(m_current_pool, m_master_semaphore.CurrentTick());
	if (const auto& [pool, tick] = m_pending_pools.front(); m_master_semaphore.IsFree(tick)) {
		m_current_pool = pool;
		m_pending_pools.pop_front();
		EXIT_IF(m_graphics.device.resetDescriptorPool(m_current_pool, {}) != vk::Result::eSuccess);
	} else {
		CreateDescriptorPool();
	}

	m_sets.clear();
	auto& fresh_batch = m_sets[layout];
	EXIT_IF(!Allocate(layout, fresh_batch));
	return fresh_batch.sets[--fresh_batch.size];
}

bool DescriptorHeap::Allocate(vk::DescriptorSetLayout layout, Batch& batch) {
	std::array<vk::DescriptorSetLayout, DescriptorSetBatch> layouts;
	layouts.fill(layout);

	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool = m_current_pool;
	allocate.pSetLayouts    = layouts.data();

	for (;;) {
		allocate.descriptorSetCount = batch.allocation;
		const auto result = m_graphics.device.allocateDescriptorSets(&allocate, batch.sets.data());
		if (result == vk::Result::eSuccess) {
			batch.size = batch.allocation;
			return true;
		}
		EXIT_IF(result != vk::Result::eErrorOutOfPoolMemory &&
		        result != vk::Result::eErrorFragmentedPool);
		if (batch.allocation == 1) {
			return false;
		}
		batch.allocation /= 2;
	}
}

vk::DescriptorSet DescriptorHeap::AllocateKept(vk::DescriptorSetLayout layout) {
	while (!m_kept_retired.empty() && m_master_semaphore.IsFree(m_kept_retired.front().second)) {
		m_graphics.device.destroyDescriptorPool(m_kept_retired.front().first, nullptr);
		m_kept_retired.pop_front();
	}
	while (!m_kept_released.empty() && m_master_semaphore.IsFree(m_kept_released.front().second)) {
		const auto set = m_kept_released.front().first;
		EXIT_IF(m_graphics.device.freeDescriptorSets(m_kept_pool, 1, &set) != vk::Result::eSuccess);
		m_kept_released.pop_front();
	}
	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool     = m_kept_pool;
	allocate.descriptorSetCount = 1;
	allocate.pSetLayouts        = &layout;
	vk::DescriptorSet set       = nullptr;
	const auto        result    = m_graphics.device.allocateDescriptorSets(&allocate, &set);
	if (result == vk::Result::eSuccess) return set;
	EXIT_IF(result != vk::Result::eErrorOutOfPoolMemory && result != vk::Result::eErrorFragmentedPool);
	// Full: the pool goes with every set in it once the GPU is past the work recorded so far; its released sets go too.
	m_kept_retired.emplace_back(m_kept_pool, m_master_semaphore.CurrentTick());
	m_kept_released.clear();
	CreateKeptPool();
	++m_kept_generation;
	allocate.descriptorPool = m_kept_pool;
	EXIT_IF(m_graphics.device.allocateDescriptorSets(&allocate, &set) != vk::Result::eSuccess);
	return set;
}

void DescriptorHeap::ReleaseKept(vk::DescriptorSet set) {
	m_kept_released.emplace_back(set, m_master_semaphore.CurrentTick());
}

void DescriptorHeap::CreateDescriptorPool() {
	vk::DescriptorPoolCreateInfo create {};
	create.maxSets       = DescriptorHeapCount;
	create.poolSizeCount = static_cast<uint32_t>(DescriptorPoolSizes.size());
	create.pPoolSizes    = DescriptorPoolSizes.data();
	EXIT_IF(m_graphics.device.createDescriptorPool(&create, nullptr, &m_current_pool) !=
	        vk::Result::eSuccess);
}

} // namespace Libs::Graphics
