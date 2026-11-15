#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "common/virtualMemory.h"
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

	template <bool track, bool is_read = false>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

	// The host protection of the range was replaced behind the tracker (a view mapped again with its own
	// access by a partial unmap, a guest mprotect): protects the watched pages again, so CPU accesses to
	// them fault once more. Unwatched pages keep what they were given. Outside the guest address-space lock.
	void ResyncHostProtection(uint64_t vaddr, uint64_t size);

	// A guest view's pages just committed (first touch, lazy protection): gives each page its final host protection in one
	// protect per run, under the region lock, so no tracker change is lost and no page is opened for a moment: the tracker's
	// where it watches the page, else the guest's (guest_mode, per page; its NoAccess wins). Protects directly, not through
	// the guest address space, whose lock the caller may not take. guest_mode must take no lock the tracker holds.
	// guest_space: protect through the guest address space instead (a guest mprotect, whose range can hold first-touch
	// views' reserved granules), as the tracker's own protections do; the caller holds no state lock.
	void ApplyHostProtection(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode (*guest_mode)(uint64_t, void*),
	                         void* context, bool guest_space = false);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
