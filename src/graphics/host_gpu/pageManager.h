#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "common/virtualMemory.h"
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

	// The runs of pages of [vaddr, vaddr + size) a watcher holds, with their watchers' protection (the other pages left
	// out), by the pages' state read without their regions' locks: for a caller holding the guest address-space lock,
	// which every protection change of the trackers waits for, so that one made meanwhile is applied after the caller's.
	struct WatchedRun {
		uint64_t                    address = 0, size = 0;
		Common::VirtualMemory::Mode mode    = Common::VirtualMemory::Mode::NoAccess;
	};
	void WatchedRuns(uint64_t vaddr, uint64_t size, std::vector<WatchedRun>* runs) const;

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
	// (FLUSH-ASYNC, KYTY_FLUSH_ASYNC=1) Before a packet that lets the guest learn nothing: the ranges go to a worker.
	static void FlushDeferredProtectionAsync() noexcept;
	// Moves with each flush of the calling thread's deferred read protections: on the GPU thread, before each point at
	// which the guest can learn that work it submitted ran (above). Until then the guest writes no memory that work
	// reads: guest memory the GPU thread's translation reads twice in one epoch was not changed by the guest in
	// between, unless the guest raced its own GPU work. (KYTY_NATIVE_MEMORY: such a write is seen only at the next
	// SweepCpuWrites, at the next submission or satisfied WAIT_REG_MEM, which is when the GPU may use it anyway.)
	[[nodiscard]] static uint64_t ObservationEpoch() noexcept;

	// KYTY_NATIVE_MEMORY=1 (off by default; read once): the watchers keep their counts but never change the host
	// protection (Protect does nothing: no read-only or no-access VirtualProtect, no fault of any kind; read watchers and
	// the texture offload's holds included). CPU writes are learnt from Windows' write watch instead
	// (LibKernel::Memory::NativeWriteWatch, MEM_WRITE_WATCH memory) by SweepCpuWrites; reads are not seen at all.
	[[nodiscard]] static bool NativeWriteWatch() noexcept;
	// (nativemem1) KYTY_WRITE_WATCH=1 with NativeWriteWatch (LibKernel::Memory::NativeReadFaultsEnabled): the write
	// watchers still never protect (the write watch sees the writes), but read watchers and the offload's holds make
	// their pages no-access again, as without native memory: CPU reads of GPU-written memory fault and are read back
	// (GpuResourceManager::NativeFlushGpuWrites does nothing then). SweepCpuWrites reads only the regions a write watcher
	// holds pages of (KYTY_WRITE_WATCH_FULL_EVERY: some sweeps only the recently written ones).
	[[nodiscard]] static bool NativeReadFaults() noexcept;
	// (nativemem1) SweepCpuWrites' counts since the last call (KYTY_WRITE_WATCH only): full and partial sweeps, regions
	// read and their bytes.
	static void TakeSweepStats(uint64_t* full, uint64_t* partial, uint64_t* regions, uint64_t* queried_bytes) noexcept;
	// NativeWriteWatch only (nothing otherwise; GpuResourceManager::SweepCpuWrites, the GPU thread): the write watch of
	// every tracker region that has a watched page now read and reset at once, with the pages CaptureCpuWrites kept since
	// the last sweep; appended to `runs` (page runs, in address order): the pages written since the last sweep that a
	// watcher holds now (write, read or offload: the pages a write fault was taken on before). Pages no watcher holds
	// are dropped (their write faulted nothing before either). Regions where the write watch could not be read report
	// every page a write watcher holds (said once). Takes the guest address space's lock (inside NativeWriteWatch), then
	// each region's lock alone: no lock of the caller's may be held.
	// (ww2, KYTY_WRITE_WATCH_MODE=poll) `image_runs` (may be null): the pages TakeWritesBeforeGpuWrite read since the last
	// sweep that a write watcher holds, for the images over them only (TextureCache::InvalidateCpuWrittenPages).
	// KYTY_WRITE_WATCH_MODE=protect: nothing (the writes fault).
	void SweepCpuWrites(std::vector<DeferredRange>* runs, std::vector<DeferredRange>* image_runs = nullptr);
	// (ww2) KYTY_WRITE_WATCH_MODE (with KYTY_NATIVE_MEMORY=1 and KYTY_WRITE_WATCH=1; both false otherwise, and with
	// KYTY_WRITE_WATCH_MODE=sweep, the sweeps as they were). WatchPolled, the default: the sweeps run only at the
	// hand-overs no sweep has covered (SweptSince; a WAIT_REG_MEM on a value the game wrote). WatchProtects: the write
	// watchers protect read-only and the game's writes fault, as without native memory; no sweep.
	[[nodiscard]] static bool WatchPolled() noexcept;
	[[nodiscard]] static bool WatchProtects() noexcept;
	// (ww2, WatchPolled) The count of full sweeps begun, moved before a sweep reads the write watch. A submission
	// stamped with it when the game queued it (GuestGpu::Enqueue: its writes came before) needs no sweep of its own once
	// SweptSince(stamp): a sweep that began after it was queued has read every write made before it. Always false
	// outside WatchPolled.
	[[nodiscard]] static uint64_t SweepStamp() noexcept;
	[[nodiscard]] static bool     SweptSince(uint64_t stamp) noexcept;
	// (ww2, WatchPolled; nothing otherwise - `pages` left empty -) Right BEFORE the buffers make [vaddr, vaddr + size)
	// GPU-written (BufferCache::UploadDirtyRanges, the GPU thread): the pages the game wrote there since their last
	// reset, read and reset now, in `pages` (the caller marks them CPU-dirty at once: uploaded before the GPU writes, as
	// their write faults did) and kept for the images over them at the next sweep. Before, the next sweep found such a
	// page no-access and dropped its write ("written pages dropped").
	static void TakeWritesBeforeGpuWrite(uint64_t vaddr, uint64_t size, std::vector<uint64_t>* pages) noexcept;
	// (ww2, WatchProtects; false otherwise) Whether some page of the range may be write-protected by a watcher (a hint,
	// read without locks): the emulator's own writes there open the pages first (LibKernel::Memory, NativeGuardedAccess).
	[[nodiscard]] bool HasWriteProtection(uint64_t vaddr, uint64_t size) const noexcept;
	// (ww2) Since the last call: written pages left out because no-access when swept; TakeWritesBeforeGpuWrite's and
	// CaptureAfterNoAccess's calls, pages and time (microseconds, both); times the sweep's region list was built again.
	struct WatchStats {
		uint64_t noaccess_pages = 0, before_gpu_calls = 0, before_gpu_pages = 0, after_protect_calls = 0,
		         after_protect_pages = 0, list_rebuilds = 0, capture_us = 0;
	};
	static void TakeWatchStats(WatchStats* stats) noexcept;
	// NativeWriteWatch only (nothing otherwise; any thread): right BEFORE the emulator writes [vaddr, vaddr + size) of
	// guest memory itself and then clears the write watch there (LibKernel::Memory::NativeForgetWrites: GPU results
	// written back), the pages the game wrote there since the last sweep are read (and reset) now and kept for the next
	// SweepCpuWrites: the forgetting would lose them. Takes the address space's lock, then a leaf mutex.
	static void CaptureCpuWrites(uint64_t vaddr, uint64_t size) noexcept;
	// KYTY_PROTECT_PAUSE (default 1; =0 off): while a dangerous window lasts (low free RAM, a zone change seen by the
	// texture probe), no page is write-protected; the pages that would have been are resynced (taken as CPU-written)
	// when it ends. NoteZoneSignal: the probe's unmapped texture bytes and streamed copy bytes.
	static void NoteZoneSignal(uint64_t unmapped_bytes, uint64_t copied_bytes) noexcept;
	// GPU thread (GpuResourceManager::SyncProtectPause): the ranges to protect again (ending only) and to resync; true
	// when there is work. EndProtectPause after it.
	[[nodiscard]] static bool TakeProtectPause(std::vector<DeferredRange>* reprotect, std::vector<DeferredRange>* resync,
	                                           bool* ending) noexcept;
	static void KeepProtectPauseResync(uint64_t vaddr, uint64_t size) noexcept;
	static void EndProtectPause(bool ending, uint64_t pages) noexcept;
	// (Before SweepCpuWrites the write watch was read lazily, by each cache at some of its lookups. Kept for those
	// callers - TextureCache's TakeImageCpuWrites -, it now does nothing: the sweep invalidates every consumer at once.)
	enum class WriteConsumer : uint32_t { Buffers = 0, Images = 1, Count = 2 };
	void TakeCpuWrites(uint64_t /*vaddr*/, uint64_t /*size*/, WriteConsumer /*consumer*/,
	                   std::vector<DeferredRange>* /*runs*/) noexcept {}
	// A write fault on a page no watcher holds: its host protection is not the trackers'. Gives
	// the page the guest's own protection back (writable if the guest's is) and returns true;
	// false when the page is watched. Checked and changed under the page's lock, so a watcher
	// protecting the page meanwhile is not undone.
	[[nodiscard]] bool RestoreIfUnwatched(uint64_t vaddr) noexcept;
	// From the first to the end of the last page of [vaddr, vaddr + size) a watcher holds (its host protection is the
	// trackers'), by the pages' state: no system call. False when no page of the range has one.
	[[nodiscard]] bool WatchedSpan(uint64_t vaddr, uint64_t size, uint64_t* first, uint64_t* end) const noexcept;

	// (raiz8, KYTY_RAIZ=1, KYTY_RAIZ_ASYNC_RELEASE - on by default; =0 off -) The GPU thread's read releases (a read
	// watcher dropped: no access -> its watchers' readable protection) only change the pages' state there: the pages are
	// marked and keep their no access - and their read hints: the emulator's own accesses keep opening windows - until
	// one of these applies their current state's protection (the kernel puts the kept bytes in then, as at any release):
	//  - ApplyDeferredReleaseAt: a fault on such a page (GpuResourceManager::HandleFault, before anything else); true when
	//    the page was marked (the access is retried; a page watched again by then faults again on the usual path);
	//  - ApplyDeferredReleases: the guest thread that waited for the release (BufferCache::ReadMemory, after its
	//    SendCommandSync), over its range;
	//  - the release thread, for what is left.
	// A release only makes pages less protected, so a late one hides no write (a write faults and applies it first).
	[[nodiscard]] bool ApplyDeferredReleaseAt(uint64_t vaddr) noexcept;
	static void        ApplyDeferredReleases(uint64_t vaddr, uint64_t size) noexcept;

	template <bool track>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	// The texture offload's hold on whole pages (texture-offload.h): no access while it lasts, a count of its own beside
	// the watchers (the memory tracker's read watcher of a page GPU work wrote stays its own); reads of them fault
	// (HasReadWatchers).
	template <bool track>
	void UpdateReadWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
	friend struct DeferredReleaseAccess; // (raiz8: the release thread, pageManager.cpp)
};

// (raiz8) GuestGpu::IsGpuThread for the page manager (defined in gpuResourceManager.cpp: pageManager.cpp does not
// include the command processor's headers).
[[nodiscard]] bool PageManagerOnGpuThread() noexcept;

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
