#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_

#include "common/common.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <array>
#include <deque>
#include <unordered_map>
#include <utility>
#include <atomic>
#include <mutex>
#include <span>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

struct GraphicContext;
class MasterSemaphore;
class CommandScheduler;
class PageManager;
class RenderContext;

class DescriptorHeap {
public:
	DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore);
	~DescriptorHeap();
	KYTY_CLASS_NO_COPY(DescriptorHeap);

	[[nodiscard]] vk::DescriptorSet Commit(vk::DescriptorSetLayout layout);
	// A set kept from draw to draw (image groups, RenderExecutor::CommitBindings), from a pool of its own. When that is
	// full it is replaced, and KeptPoolGeneration changes: every set allocated before is gone (the old pool is destroyed
	// once the GPU is past the work recorded so far) and must be neither bound nor released again. A released set is
	// freed once the GPU is past the work recorded so far.
	[[nodiscard]] vk::DescriptorSet AllocateKept(vk::DescriptorSetLayout layout);
	void                            ReleaseKept(vk::DescriptorSet set);
	[[nodiscard]] uint64_t          KeptPoolGeneration() const { return m_kept_generation; }

private:
	static constexpr uint32_t DescriptorSetBatch = 32;

	struct Batch {
		std::array<vk::DescriptorSet, DescriptorSetBatch> sets {};
		uint32_t                                          size       = 0;
		uint32_t                                          allocation = DescriptorSetBatch;
	};

	[[nodiscard]] bool Allocate(vk::DescriptorSetLayout layout, Batch& batch);
	void               CreateDescriptorPool();
	void               CreateKeptPool();

	GraphicContext&                                     m_graphics;
	MasterSemaphore&                                    m_master_semaphore;
	vk::DescriptorPool                                  m_current_pool = nullptr;
	std::deque<std::pair<vk::DescriptorPool, uint64_t>> m_pending_pools;
	std::unordered_map<vk::DescriptorSetLayout, Batch>  m_sets;
	vk::DescriptorPool                                  m_kept_pool = nullptr;
	std::deque<std::pair<vk::DescriptorSet, uint64_t>>  m_kept_released;
	std::deque<std::pair<vk::DescriptorPool, uint64_t>> m_kept_retired;
	uint64_t                                            m_kept_generation = 0;
};

using ShaderRecompiler::IR::ImageHeapChange;

// The game's descriptor heap as one long-lived descriptor set, set 1 of every guest pipeline
// layout (ShaderIR.h, ImageHeapSet): an array of sampled images a binding kind with a slot an
// element, and a dword of info a slot. A shader reads a texture by the slot number the game gave
// it, so a shader does not depend on which textures exist. The host fills the slots from the
// game's own writes to its heap: the game writes an entry when it makes a view, changes only its
// min LOD while it streams, and fills it with 0xFEFEFEFE when it frees it (Wolverine Engine.md
// 5.1). Nothing is worked out per draw.
class GuestImageHeap {
public:
	GuestImageHeap(GraphicContext& graphics, PageManager& page_manager);
	~GuestImageHeap();
	KYTY_CLASS_NO_COPY(GuestImageHeap);

	// With the recording thread, a slot's descriptors and info change in the command order, as they do without
	// it: between the submits of the command buffers queued before and after the change (design rule 5).
	void UseScheduler(CommandScheduler& scheduler) { m_scheduler = &scheduler; }

	[[nodiscard]] vk::DescriptorSet Set() const { return m_set; }
	using Descriptor = std::array<uint32_t, 8>;

	// GPU thread: a draw reads the game's heap at this address. A heap that moves starts over.
	void Watch(uint64_t base, uint64_t size, const RenderContext& context);
	// Game thread, when it submits work: the heap's entries that changed since its last submission,
	// so that the work sees the heap as the game left it, as on the PS5.
	[[nodiscard]] std::vector<ImageHeapChange> Diff();
	// Any thread: the CPU or the host wrote [address, address + size) (a write fault, host I/O, a backing write). The next
	// Diff compares those pages again: it compares only pages written since it last read them, which it watches
	// (KYTY_HEAP_WATCH: 1 or unset on, 0 off, 2 verify against the whole diff).
	void InvalidateMemory(uint64_t address, uint64_t size);
	// GPU thread, when it starts that work.
	void Queue(std::vector<ImageHeapChange>&& changes);
	// The image's layout or texels changed: its slots are resolved again.
	void Stale(Common::SlotId image);
	// The slot's image went before it was shown: the next draw that reads the heap resolves it again.
	void Retry(uint32_t slot);
	// The slot has no image (none yet, or its image went): it reads as empty until the texture cache
	// registers an image at `address`, and is then resolved from images that exist only. So a slot
	// over a render target never makes an image of its own, and two slots over one memory never
	// evict each other's. A slot stops waiting after a few retries that found nothing (the game makes
	// render targets at the same memory every frame), until the game writes its entry again.
	void Wait(uint32_t slot, uint64_t address);
	void Arrived(uint64_t address);
	[[nodiscard]] bool Waiting(uint32_t slot) const { return m_images.at(slot).waiting; }
	// Frame count for TakeWork's lost-slot retry (every 30 frames).
	void NextFrame() { ++m_frame; }
	// Of the last TakeWork: entries that were stale slots, and retried slots.
	uint32_t last_stale   = 0;
	uint32_t last_retried = 0;
	// What the next draw that reads the heap resolves: changed entries and stale slots. A stale image among `targets`
	// (what this draw renders to) stays listed: no slot can show it during the draw (RenderExecutor::CommitBindings), and
	// resolving a slot binds its image, which made the target General for that draw, changed its layout and so made it
	// stale again, about 530 layout changes a frame at Marvel's Wolverine's Trask outpost. The first draw that reads the
	// heap and doesn't render to it resolves it, takes it through the barrier to its read layout and shows it.
	[[nodiscard]] std::vector<ImageHeapChange> TakeWork(std::span<const Common::SlotId> targets);
	// GPU thread: the records of the game's sampler heap that changed since the last call.
	[[nodiscard]] std::vector<std::pair<uint32_t, std::array<uint32_t, 4>>> SamplerChanges(uint64_t base,
	                                                                                     uint64_t size);
	// A sampler heap record as a float and as an integer sampler (ShaderIR.h).
	void ShowSampler(uint32_t slot, vk::Sampler float_sampler, vk::Sampler integer_sampler);

	[[nodiscard]] Common::SlotId Image(uint32_t slot) const { return m_images.at(slot).image; }
	// The heap shows the image: the texture cache keeps it (the game frees it by its heap entry).
	[[nodiscard]] bool Holds(Common::SlotId image) const;
	// Shows `view` of `image` at `slot` of the array at `binding`, and gives the slot its info.
	void Show(uint32_t slot, uint32_t binding, Common::SlotId image, vk::ImageView view,
	          vk::ImageLayout layout, uint32_t info);
	void Clear(uint32_t slot);
	// The image is being destroyed: its slots read as empty.
	void Forget(Common::SlotId image);
	// Slots whose image went (Forget) are resolved again every 30 frames (TakeWork) until they show an image, the game
	// writes them, or 40 tries pass: nothing else resolves a slot whose memory was unmapped and mapped again unchanged.
	void RetryLost();

private:
	struct Shown {
		vk::ImageView   view   = nullptr;
		vk::ImageLayout layout = vk::ImageLayout::eUndefined;
	};
	struct Slot {
		Common::SlotId image;
		Descriptor     descriptor {};
		bool           waiting = false;
		uint8_t        misses  = 0; // retries since it last showed an image
	};

	void WriteInfo(uint32_t slot, uint32_t info);
	void WriteDescriptor(const vk::WriteDescriptorSet* writes, uint32_t count);

	GraphicContext&                                     m_graphics;
	CommandScheduler*                                   m_scheduler = nullptr;
	vk::DescriptorPool                                  m_pool = nullptr;
	vk::DescriptorSet                                   m_set  = nullptr;
	VkBuffer                                            m_info_buffer = VK_NULL_HANDLE;
	VmaAllocation                                       m_info_allocation {};
	uint32_t*                                           m_info = nullptr;
	std::unordered_map<uint32_t, std::vector<Shown>>    m_shown; // by binding, then slot
	std::vector<Slot>                                   m_images; // by slot
	std::unordered_map<uint32_t, std::vector<uint32_t>> m_slots;  // by image index
	std::vector<ImageHeapChange>                        m_pending;
	std::unordered_map<uint32_t, uint8_t>               m_lost; // slot -> tries (Forget, RetryLost)
	uint64_t                                            m_lost_frame = 0;
	std::unordered_map<uint64_t, std::vector<uint32_t>> m_waiting; // by address
	std::mutex                                          m_stale_mutex; // Stale comes from write faults too
	std::vector<Common::SlotId>                         m_stale;
	uint32_t                                            m_retried = 0;
	uint64_t                                            m_frame   = 0;
	// The game thread's copy of the heap, and where the GPU thread found the heap.
	std::mutex                                          m_guest_mutex;
	std::vector<uint32_t>                               m_guest;
	uint64_t                                            m_guest_base = 0;
	std::atomic<uint64_t>                               m_base {0};
	std::atomic<uint64_t>                               m_size {0};
	// The heap's pages by tracker page from m_watch_origin: 1 = not watched, compared by the next Diff; 0 = watched, unchanged.
	void                                                Unwatch();
	PageManager&                                        m_page_manager;
	int                                                 m_watch_mode = 1;
	bool                                                m_watching   = false;
	std::atomic<bool>                                   m_mappable {false}; // a write fault there reaches HandleFault
	std::mutex                                          m_watch_mutex; // the page bytes: Diff and write faults
	std::vector<uint8_t>                                m_dirty_pages;
	std::vector<std::pair<size_t, size_t>>              m_runs;
	uint64_t                                            m_watch_origin  = 0;
	uint64_t                                            m_watched_bytes = 0; // the heap's size at the last diff
	std::atomic<uint64_t>                               m_watch_begin {0}, m_watch_end {0}; // for a check without the lock
	std::atomic<uint64_t>                               m_stat_faults {0};
	uint64_t m_stat_start = 0, m_stat_diffs = 0, m_stat_pages = 0, m_stat_of = 0, m_stat_late = 0, m_stat_misses = 0;
	uint64_t                                            m_sampler_base = 0;
	std::vector<uint32_t>                               m_samplers; // the game's sampler heap, as shown
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
