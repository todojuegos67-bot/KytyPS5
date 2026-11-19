#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <memory>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute, Unknown };

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	[[nodiscard]] uint64_t GetPageSize() const;

	template <bool track>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

	// KYTY_SCAN_STAMP (kernel/scanStamp.h), any thread, lock-free: every page of the range has
	// been given a host protection without write access by the tracker, made already (not only
	// deferred), and none has been made writable since. False when the switch is off.
	[[nodiscard]] static bool HostWriteProtected(uint64_t vaddr, uint64_t size) noexcept;

	// Whether the page holding vaddr is read-protected for tracking. Unlocked: a hint that can
	// be stale by the time the caller acts on it.
	[[nodiscard]] bool IsReadWatched(uint64_t vaddr) const noexcept;
	// Whether a write to the page holding vaddr faults for tracking (it is write- or
	// read-protected). Unlocked: a hint as IsReadWatched.
	[[nodiscard]] bool IsWriteWatched(uint64_t vaddr) const noexcept;

	// The host protection of [vaddr, vaddr + size) was replaced behind the tracker (a view mapped
	// again with its own access by a partial unmap, a guest mprotect that allows writes): the
	// watched pages get the protection their counts ask for again, so CPU writes to them fault
	// (and reach the GPU) once more. Unwatched pages keep what they were given. Any thread,
	// outside the guest address-space lock.
	void ResyncHostProtection(uint64_t vaddr, uint64_t size);

	// Round 111, deferred protection (KYTY_ASYNC_MEMORY_SYNC=1, KYTY_BISECT's memsync:async):
	// while a DeferScope made with `on` lives on a thread, the tighter host protection that
	// adding watchers of this manager asks for there is left to a worker thread ("GpuProtect"),
	// which gives each page the protection its counts ask for when it gets to it, adjacent ranges
	// in one call. Taking watchers away still changes the protection at once. Jobs posted to the
	// worker (an upload's guest reads) run after the protections deferred before them. The
	// callers make sure nothing relies on deferred work before WaitDeferred(DeferredIssued())
	// returns: the GPU thread waits before its own stores to guest memory (labels, WRITE_DATA,
	// fills and copies on the CPU), and a command buffer is submitted after it.
	class DeferScope final {
	public:
		DeferScope(PageManager& manager, bool on);
		~DeferScope();
		KYTY_CLASS_NO_COPY(DeferScope);

	private:
		const void* m_previous;
	};
	// Runs `job` on the worker after the protections deferred before it (any thread).
	void PostDeferred(Common::UniqueFunction<void> job);
	// The deferred protections and jobs asked for so far (a count: 0 while none was).
	[[nodiscard]] uint64_t DeferredIssued() const noexcept;
	// Whether some are still to be done.
	[[nodiscard]] bool DeferredPending() const noexcept;
	// Returns once the first `issued` deferred protections and jobs are done (any thread but the
	// worker). With `help` (round 156), a batch the worker has not started is done by the caller:
	// only where it holds none of the locks the batch takes (the page regions', the guest address
	// space's, the memory tracker's).
	void WaitDeferred(uint64_t issued, bool help = false);
	// WaitDeferred(DeferredIssued()) when some are pending.
	void FinishDeferred(bool help = false) {
		if (DeferredPending()) {
			WaitDeferred(DeferredIssued(), help);
		}
	}

	struct Impl;

private:
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
