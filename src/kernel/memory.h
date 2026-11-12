#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/virtualMemory.h"

#include <string>
#include <utility>
#include <vector>

namespace Libs::Graphics {
class RenderContext;
enum class PageFaultAccess;
} // namespace Libs::Graphics

namespace Libs::LibKernel::Memory {

void Initialize();
void Shutdown();

struct Lifecycle {
	static constexpr const char* name       = "Memory";
	static constexpr auto        initialize = Libs::LibKernel::Memory::Initialize;
	static constexpr auto        shutdown   = Libs::LibKernel::Memory::Shutdown;
};

using callback_func_t = void (*)(uintptr_t addr, size_t size);

constexpr uint32_t KERNEL_MAXIMUM_NAME_LENGTH = 32;
constexpr uint64_t kExtendedMemoryBase       = 0x080000000000ull;
constexpr uint64_t kExtendedMemorySize       = 512ull * 1024 * 1024 * 1024;

struct VirtualQueryInfo {
	uintptr_t start;
	uintptr_t end;
	uint64_t  offset;
	int32_t   protection;
	int32_t   memory_type;
	uint32_t  is_flexible  : 1;
	uint32_t  is_direct    : 1;
	uint32_t  is_stack     : 1;
	uint32_t  is_pooled    : 1;
	uint32_t  is_committed : 1;
	uint32_t  is_gpu_prt   : 1;
	uint32_t  amm_usage    : 1;
	uint32_t  reserved     : 1;
	char      name[KERNEL_MAXIMUM_NAME_LENGTH];
	uint8_t   gpu_mask_id;
	uint8_t   reserved2;
};

static_assert(sizeof(VirtualQueryInfo) == 72, "VirtualQueryInfo struct size is incorrect");

struct KernelBatchMapEntry {
	void*         start;
	uint64_t      offset;
	uint64_t      length;
	unsigned char protection;
	unsigned char type;
	int16_t       reserved;
	int32_t       operation;
};

static_assert(sizeof(KernelBatchMapEntry) == 32, "KernelBatchMapEntry struct size is incorrect");

struct KernelMemoryPoolBatchEntry {
	uint32_t op;
	uint32_t flags;
	union {
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
			uint8_t  type;
		} commit;
		struct {
			void*    addr;
			uint64_t len;
		} decommit;
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
		} protect;
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
			uint8_t  type;
		} type_protect;
		struct {
			void*    dst;
			void*    src;
			uint64_t len;
		} move;
		uintptr_t padding[3];
	};
};

static_assert(sizeof(KernelMemoryPoolBatchEntry) == 32,
              "KernelMemoryPoolBatchEntry struct size is incorrect");

struct KernelMemoryPoolBlockStats {
	int32_t available_flushed_blocks;
	int32_t available_cached_blocks;
	int32_t allocated_flushed_blocks;
	int32_t allocated_cached_blocks;
};

static_assert(sizeof(KernelMemoryPoolBlockStats) == 16,
              "KernelMemoryPoolBlockStats struct size is incorrect");

void                   RegisterCallbacks(callback_func_t alloc_func, callback_func_t free_func);
void                   SetFlexibleMemorySize(uint64_t size);
int AllocateDirectMemory(int64_t search_start, int64_t search_end, size_t size, size_t alignment,
                         int memory_type, int64_t* phys_addr_out, bool automatic = false);
int MapAutomaticMemory(uint64_t vaddr, size_t size, int type, int prot);
bool                   TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size);
bool                   TryReadBacking(uint64_t vaddr, void* data, uint64_t size);
bool                   TryReadBufferBacking(uint64_t vaddr, void* data, uint64_t size);

// Diagnostics: who wrote guest memory. The emulator's own writes into it (TryWriteBacking, and the
// GPU thread's stores, fills and copies done on the CPU) and game threads' write faults, the last
// 16384 and 65536 of them, so a check that finds bytes it did not expect there (the texture
// cache's light grids) can name the writes that touched them, and when.
enum class WriteSource : uint8_t {
	Other,          // TryWriteBacking from a caller that does not name itself
	Download,       // a buffer download put into guest memory (BufferCache)
	FastReadback,   // a fast readback put into guest memory (BufferCache)
	StoreInPlace,   // a store to GPU-written bytes done on both copies (BufferCache::WriteClean)
	ImageWriteBack, // an image written back to guest memory (TextureCache)
	MetadataClear,  // color metadata rewritten as expanded keys (TextureCache)
	GpuThreadStore, // WRITE_DATA done on the CPU
	Fill,           // a DMA fill done on the CPU
	Copy,           // a DMA copy done on the CPU
	FileRead,       // an AMPR file read
	CpuWriteFault,  // a game thread's store faulted there (the faulting byte's page)
	Count
};
[[nodiscard]] const char* WriteSourceName(WriteSource source) noexcept;
void                      JournalWrite(uint64_t vaddr, uint64_t size, WriteSource source) noexcept;
// The source TryWriteBacking journals this thread's writes as now (WriteSourceScope).
[[nodiscard]] WriteSource CurrentWriteSource() noexcept;
// The source TryWriteBacking journals this thread's writes as, while it lives.
class WriteSourceScope final {
public:
	explicit WriteSourceScope(WriteSource source) noexcept;
	~WriteSourceScope();
	WriteSourceScope(const WriteSourceScope&)            = delete;
	WriteSourceScope& operator=(const WriteSourceScope&) = delete;

private:
	WriteSource m_previous;
};
// The journaled writes that touched [vaddr, vaddr + size) in the last `ms` milliseconds, newest
// first, at most `max` of them: "<source> <address>+<size> <age> ms ago (<thread>)", joined by
// "; ", or "none".
[[nodiscard]] std::string DescribeJournal(uint64_t vaddr, uint64_t size, uint64_t ms, size_t max);
// The steady-clock time the journal stamps its entries with, in nanoseconds.
[[nodiscard]] uint64_t JournalNowNs() noexcept;
// The guest's direct memory: the bytes allocated now and the size it may use (round 80, for the
// Perf line: a race's load came within 0.4 GB of it). Any thread.
struct DirectMemoryUse {
	uint64_t allocated = 0;
	uint64_t size      = 0;
	uint64_t blocks    = 0;
	// Of `allocated`, the bytes given room past the direct memory (round 81).
	uint64_t overflow = 0;
};
[[nodiscard]] DirectMemoryUse GetDirectMemoryUse();
// KYTY_BISECT's backing:locked (round 82): copies to and from guest memory hold the backing
// store's lock while they copy, as before round 82.
void SetBackingCopyLocked(bool locked);
// KYTY_BISECT's protect:exclusive (round 135): the page tracker's protections take the address
// space's lock exclusively, as before they shared it.
void SetProtectLockExclusive(bool exclusive);
// GPU thread: some byte of [vaddr, vaddr + size) was written by GPU work into a buffer, so the
// GPU holds it and a host read would wait for the GPU.
[[nodiscard]] bool     IsGpuBufferWritten(uint64_t vaddr, uint64_t size);
bool                   TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size);
// GPU thread: the backing bytes of [vaddr, vaddr + size) when one mapping holds them and no GPU
// work has written any of them, else nullptr. Current until the GPU thread marks them
// GPU-written or the guest remaps them.
[[nodiscard]] const uint8_t* FindGpuCleanBacking(uint64_t vaddr, uint64_t size);
// Any thread: the backing bytes of [vaddr, vaddr + size) when one mapping holds them, without the
// GPU-dirty check (the caller knows them clean: DrawPrep's KYTY_MT_PROG seal).
[[nodiscard]] const uint8_t* FindBackingUnchecked(uint64_t vaddr, uint64_t size);
// Lets the calling thread make the GPU thread's clean-backing queries (FindGpuCleanBacking,
// TryReadGpuCleanBacking) while the GPU thread waits for it and makes only the same read-only
// queries itself: the stage-preparation helper (pipelineCache.cpp, StagePrepWorker).
void SetGpuReadDelegate(bool delegate);
// GPU thread or delegate: copies [vaddr, vaddr + size) from the backing store when no GPU work has
// written any of it. Never downloads, faults or waits; false otherwise.
[[nodiscard]] bool TryReadGpuCleanBacking(uint64_t vaddr, void* data, uint64_t size);
// Any thread, for a read that faulted at fault_vaddr on a page protected because the GPU wrote to
// it: reads [vaddr, vaddr + size) from the backing store when none of those bytes is GPU-written.
bool TryReadCleanFaultingBytes(uint64_t fault_vaddr, uint64_t vaddr, void* data, uint64_t size);
// A game thread's write fault: whether the faulting page is protected because the GPU wrote to it
// (rather than only watched for CPU writes).
[[nodiscard]] bool IsGpuWrittenPage(uint64_t fault_vaddr) noexcept;
// Round 133, for a crash report: the GPU's view of [vaddr, vaddr + size): whether its page holds
// GPU-written bytes, whether these bytes are among them or being downloaded, and which accesses
// to the page fault.
[[nodiscard]] std::string DescribeGpuBytes(uint64_t vaddr, uint64_t size);
// A game thread, for the store at `pc` that faulted at fault_vaddr on such a page: writes [vaddr,
// vaddr + size) to both copies (guest memory and the GPU's) through the GPU thread and leaves the
// page protected, instead of downloading what the GPU wrote there first. False to decline (the
// page's or the store's budget is spent, a download of these bytes is in flight, no cached buffer
// holds them); the fault then takes the normal path.
bool TryWriteGpuFaultingBytes(uint64_t pc, uint64_t fault_vaddr, uint64_t vaddr, const void* data,
                              uint64_t size);
[[nodiscard]] uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size);
// Like ClampRangeSize, but 0 for a range that starts outside committed memory instead of EXIT.
[[nodiscard]] uint64_t TryClampRangeSize(uint64_t vaddr, uint64_t size);
// Whether every byte of the range is committed guest memory, by the kernel's own table of guest
// mappings (what sceKernelVirtualQuery reports), without asking the host. An empty range is, and
// so is every range before the kernel's memory is set up (unit tests).
[[nodiscard]] bool     IsCommittedRange(uint64_t vaddr, uint64_t size);
// The committed parts of [vaddr, vaddr + size), in order, as (start, size): the pages a GPU write
// over the range lands in when some of it is not mapped (sparse memory). At most `max_spans`;
// false (and no spans) when there are more.
[[nodiscard]] bool     CommittedSpans(uint64_t vaddr, uint64_t size, size_t max_spans,
                                      std::vector<std::pair<uint64_t, uint64_t>>* spans);
// From vaddr to the end of the last committed byte of [vaddr, vaddr + size) (0 when none is).
[[nodiscard]] uint64_t CommittedExtent(uint64_t vaddr, uint64_t size);
void                   WriteBacking(uint64_t vaddr, const void* data, uint64_t size) noexcept;
// Diagnostics: KYTY_TRACE_MEMORY=0x<address>+<size>[,...]. What happens to these guest ranges is
// logged as "Trace memory: ...": the kernel's mappings over them (maps, unmaps, protection
// changes, including the memory manager's), file reads into them, and, from the GPU side, buffer
// writes, fills and copies into them and cached buffers cut short at a hole in them.
[[nodiscard]] bool TracesMemory(uint64_t vaddr, uint64_t size);
void               TraceMemory(const char* format, ...) KYTY_FORMAT_PRINTF(1, 2);
// The kernel's mappings over [vaddr, vaddr + size): each piece's offset from vaddr, size and type,
// the holes between them included ("+0x0 0x8c00000 Direct rw, +0x8c00000 0x400000 hole, ...").
[[nodiscard]] std::string DescribeMappings(uint64_t vaddr, uint64_t size);
// Round 124, diagnostics, any thread: the GPU's images over [vaddr, vaddr + size), one line each
// (at most `count`), or none before the GPU is set up.
[[nodiscard]] std::vector<std::string> DescribeGpuImages(uint64_t vaddr, uint64_t size,
                                                         size_t count);
void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
void                   InstallGpuResources(Graphics::RenderContext* renderer) noexcept;
[[nodiscard]] bool HandleGpuFault(Graphics::PageFaultAccess access, uint64_t fault_vaddr) noexcept;

int KYTY_SYSV_ABI KernelMapNamedFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags,
                                               const char* name);
int KYTY_SYSV_ABI KernelMapFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags);
int KYTY_SYSV_ABI KernelSetVirtualRangeName(const void* addr, uint64_t len, const char* name);
int KYTY_SYSV_ABI KernelClearVirtualRangeName(const void* addr, uint64_t len);
int KYTY_SYSV_ABI KernelMunmap(uint64_t vaddr, size_t len);
size_t KYTY_SYSV_ABI KernelGetDirectMemorySize();
int KYTY_SYSV_ABI    KernelAvailableDirectMemorySize(int64_t search_start, int64_t search_end,
                                                     size_t alignment, int64_t* phys_addr_out,
                                                     size_t* size_out);
int KYTY_SYSV_ABI    KernelGetPageTableStats(int* cpu_total, int* cpu_available, int* gpu_total,
                                             int* gpu_available);
int KYTY_SYSV_ABI KernelAllocateDirectMemory(int64_t search_start, int64_t search_end, size_t len,
                                             size_t alignment, int memory_type,
                                             int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelAllocateMainDirectMemory(size_t len, size_t alignment, int memory_type,
                                                 int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelCheckedReleaseDirectMemory(int64_t start, size_t len);
int KYTY_SYSV_ABI KernelReleaseDirectMemory(int64_t start, size_t len);
int KYTY_SYSV_ABI KernelMapDirectMemory(void** addr, size_t len, int prot, int flags,
                                        int64_t direct_memory_start, size_t alignment);
int KYTY_SYSV_ABI KernelMapDirectMemory2(void** addr, size_t len, int type, int prot, int flags,
                                         int64_t direct_memory_start, size_t alignment);
int KYTY_SYSV_ABI KernelMapNamedDirectMemory(void** addr, size_t len, int prot, int flags,
                                             int64_t direct_memory_start, size_t alignment,
                                             const char* name);
int KYTY_SYSV_ABI KernelSetPrtAperture(int index, void* addr, size_t len);
int KYTY_SYSV_ABI KernelGetPrtAperture(int index, void** addr, size_t* len);
int KYTY_SYSV_ABI KernelIsAddressSanitizerEnabled();
int KYTY_SYSV_ABI KernelQueryMemoryProtection(void* addr, void** start, void** end, int* prot);
int KYTY_SYSV_ABI KernelDirectMemoryQuery(int64_t offset, int flags, void* info, size_t info_size);
int KYTY_SYSV_ABI KernelVirtualQuery(const void* addr, int flags, VirtualQueryInfo* info,
                                     uint64_t info_size);
int KYTY_SYSV_ABI KernelIsStack(void* addr, void** start, void** end);
int KYTY_SYSV_ABI KernelReserveVirtualRange(void** addr, size_t len, int flags, size_t alignment);
int KYTY_SYSV_ABI KernelAvailableFlexibleMemorySize(size_t* size);
int KYTY_SYSV_ABI KernelConfiguredFlexibleMemorySize(size_t* size);
int KYTY_SYSV_ABI KernelMprotect(const void* addr, size_t len, int prot);
int KYTY_SYSV_ABI KernelMtypeprotect(const void* addr, size_t len, int type, int prot);
int KYTY_SYSV_ABI KernelBatchMap(KernelBatchMapEntry* entries, int num_entries,
                                 int* num_entries_out);
int KYTY_SYSV_ABI KernelBatchMap2(KernelBatchMapEntry* entries, int num_entries,
                                  int* num_entries_out, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolExpand(int64_t search_start, int64_t search_end, size_t len,
                                         size_t alignment, int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelMemoryPoolReserve(void* addr_in, size_t len, size_t alignment, int flags,
                                          void** addr_out);
int KYTY_SYSV_ABI KernelMemoryPoolCommit(void* addr, size_t len, int type, int prot, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolDecommit(void* addr, size_t len, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolBatch(const KernelMemoryPoolBatchEntry* entries, int num_entries,
                                        int* num_entries_out, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolGetBlockStats(KernelMemoryPoolBlockStats* output,
                                                size_t                      output_size);

uint64_t AllocateProgramMemory(uint64_t search_addr, uint64_t size,
                               Common::VirtualMemory::Mode mode, const char* name);
void SetProgramMemoryProtection(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode);
uint64_t AllocateRuntimeMemory(uint64_t search_addr, uint64_t size,
                               Common::VirtualMemory::Mode mode, const char* name,
                               bool fixed = false);
uint64_t AllocateGuestStackMemory(uint64_t search_addr, uint64_t size,
                                  Common::VirtualMemory::Mode mode, const char* name);
bool     ProtectGuestMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode,
                            Common::VirtualMemory::Mode* old_mode = nullptr);
// Transient PageManager watch state; does not change the guest mapping's semantic protection.
bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode);
// Round 127: a live thread's stack in memory the game gave it; the page tracker's protections
// leave it writable until it is removed (GuestAddressSpace::AddLiveStack).
void AddLiveThreadStack(uint64_t vaddr, uint64_t size);
void RemoveLiveThreadStack(uint64_t vaddr, uint64_t size);
bool FreeGuestMemory(uint64_t vaddr, uint64_t size);

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
void     TestBeforeNextBackingMap(callback_func_t callback);
void     TestFailNextPhysicalMemoryUnmap();
void     TestFailPhysicalMemoryUnmapAfter(uint32_t successful_unmaps);
void     TestFailGuestBackingStoreUnmapAfter(uint32_t successful_unmaps);
void     TestFailNextFixedReserveRangeRegistration();
void     TestFailNextVirtualRangeReplacement();
bool     TestPlaceholderRangeIsFree(uint64_t vaddr, uint64_t size);
bool     TestGuestAddressRangeIsOwned(uint64_t vaddr, uint64_t size);
bool     TestGuestBackingOutsideAddressSpace();
uint64_t TestGuestBackingSize();
// The room past the direct and flexible memory for direct allocations that do not fit (round 81).
uint64_t TestDirectMemoryOverflowSize();
bool     TestGuestFreeRangeBounds();
#endif

} // namespace Libs::LibKernel::Memory

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_ */
