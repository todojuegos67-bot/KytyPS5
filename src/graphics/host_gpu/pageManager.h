#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <memory>
#include <vector>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute, Unknown };

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	[[nodiscard]] uint64_t GetPageSize() const;
	// A hint only: callers must still check exact GPU ownership before reading
	// a backing alias. A missing hint retains the normal faulting guest load.
	[[nodiscard]] bool HasReadWatchers(uint64_t vaddr, uint64_t size) const noexcept;
	// HasReadWatchers' filter, for callers that test it inline: bit g of word g / 64 is clear when no page of the
	// 64 KiB granule g (of the tracker's address space) has read watchers.
	static constexpr uint64_t                   READ_GRANULE_BITS = 16;
	[[nodiscard]] const std::atomic<uint64_t>* ReadGranules() const noexcept;

	// Restores the watchers' protection after the host protection of watched pages was
	// changed behind the tracker's back (a guest mprotect). Unwatched pages keep theirs.
	void ReapplyProtection(uint64_t vaddr, uint64_t size);

	// KYTY_ASYNC_REPROTECT: while a sink is set on this thread, adding write watchers
	// updates the page state but records the address ranges instead of changing the host
	// protection; ReapplyProtection of those ranges applies the current state later.
	struct DeferredRange {
		uint64_t address = 0, size = 0;
	};
	static void SetDeferredWriteProtectSink(std::vector<DeferredRange>* sink) noexcept;

	// Sets every page of the range to its watchers' current protection (writable when none).
	void SyncProtection(uint64_t vaddr, uint64_t size);

	// While set on this thread, watchers released inside [begin, end) update the page state
	// without making the pages writable on the host: the range is about to be unmapped, which
	// drops its protection anyway (a texture pool made of 64 KiB mappings cost one VirtualProtect
	// per mapping for each image the unmap deleted). Protections are still applied: a skipped one
	// would leave a watched page writable and writes to it unseen. If the unmap fails, the caller
	// syncs the range (SyncProtection). end == 0 clears it.
	static void SetUnmappingRange(uint64_t begin, uint64_t end) noexcept;
	// While set on this thread (the GPU thread), the read protection of pages GPU work will write (no access) waits,
	// with the others pending, for FlushDeferredProtection or for another protection of its pages (which comes after
	// it): a range next to a pending one extends it (consecutive dispatches writing adjacent ranges made a
	// VirtualProtect call each, ~110 a frame at 1-1; a write protection between two of them, or another buffer's
	// range, ended the one range pending before). The work runs on the GPU and the guest can learn that it ran only
	// after a submission, a label or a guest command of the GPU thread: it flushes before each of them.
	static void DeferReadProtection(bool on) noexcept;
	static void FlushDeferredProtection() noexcept;
	// Moves with each flush of the calling thread's deferred read protections: on the GPU thread, before each point at
	// which the guest can learn that work it submitted ran (above). Until then the guest writes no memory that work
	// reads: guest memory the GPU thread's translation reads twice in one epoch was not changed by the guest in
	// between, unless the guest raced its own GPU work.
	[[nodiscard]] static uint64_t ObservationEpoch() noexcept;
	// A write fault on a page no watcher holds: its host protection is not the trackers'. Gives
	// the page the guest's own protection back (writable if the guest's is) and returns true;
	// false when the page is watched. Checked and changed under the page's lock, so a watcher
	// protecting the page meanwhile is not undone.
	[[nodiscard]] bool RestoreIfUnwatched(uint64_t vaddr) noexcept;

	template <bool track>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
