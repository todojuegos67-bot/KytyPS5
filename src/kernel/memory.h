#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/virtualMemory.h"

#include <utility>
#include <vector>

namespace Libs::Graphics {
class GpuResourceManager;
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
bool                   TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size);
// With frame generation: the address range held free above the guest since start-up, released right before the
// Vulkan device is made (ReserveGuestRegions, issue #45). Nothing without it.
void                   ReleaseFrameGenAddressHole();
// A write of bytes the GPU does not own (no GPU result is written back there while it lasts), one guest mapping's:
// it does not wait behind TryWriteBacking's writes of GPU results. False when it cannot (TryWriteBacking can).
bool                   TryWriteCpuBacking(uint64_t vaddr, const void* data, uint64_t size);
bool                   TryReadBacking(uint64_t vaddr, void* data, uint64_t size);
// Destination must be ordinary host storage: copying must not enter guest fault
// handling. Used by uploads; guest destinations keep TryReadBacking's lock path.
bool                   TryReadBackingToHost(uint64_t vaddr, void* data, uint64_t size);
// The backing bytes of a guest range inside one mapping (readable regardless of the guest
// range's protection), or null.
const uint8_t*         TryGetBackingPointer(uint64_t vaddr, uint64_t size);
// The end of the backing mapping `vaddr` is in (0: none): a range up to it has one backing pointer.
uint64_t               BackingMappingEnd(uint64_t vaddr);
// The backing view (every guest mapping's bytes, at their backing offset): its base and size; false when there is none.
bool                   BackingView(uint64_t* base, uint64_t* size);
// Diagnostics (live `ranges`): the guest's virtual ranges (start, size, direct memory offset, protection, memory type,
// kind, the name the game gave them) to `path`.
bool                   WriteGuestRanges(const char* path);
// The guest's protection over [vaddr, vaddr + size): whether some of it lets the GPU access it, whether some lets the
// CPU write it. False when the range is not all mapped.
bool                   GuestRangeAccess(uint64_t vaddr, uint64_t size, bool* gpu, bool* cpu_write);
// The same for a range that spans mappings: one (pointer, bytes) piece per mapping, in order.
bool TryGetBackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces);
// The same for a range with unmapped parts: a hole is a (null, bytes) piece. False if none of it is mapped.
bool TryGetSparseBackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces);
bool                   TryReadGpuCleanBackingToHost(uint64_t vaddr, void* data, uint64_t size);
// Avoid false-sharing read faults only when exact ownership proves clean.
bool TryReadGpuCleanBackingOnWatchedPage(uint64_t vaddr, void* data, uint64_t size);
bool TryReadGpuShaderSpan(uint64_t vaddr, void* data, uint64_t size, bool clean);
bool                   SyncGpuCleanBacking(uint64_t vaddr, uint64_t size);
// A read of the range faults (a page the GPU wrote, its bytes read back when the GPU thread reads them): a thread
// that is not the GPU thread must not read it.
bool                   ReadFaults(uint64_t vaddr, uint64_t size);
// The shader whose resources the GPU thread evaluates (KYTY_SYNC_LOG names it on a drain).
extern thread_local uint64_t g_srt_shader_hash;
bool                   IsUniqueGuestBackingRange(uint64_t vaddr, uint64_t size);
// The other virtual ranges mapping the same bytes ({vaddr, size} each), for a range IsUniqueGuestBackingRange refused.
bool                   GuestBackingAliases(uint64_t vaddr, uint64_t size, std::vector<std::pair<uint64_t, uint64_t>>* aliases);
// The same over consecutive mappings (each mapping's part unique, none of it unmapped): for a caller that reads and
// writes the range through its guest addresses only (not through one backing pointer).
bool                   IsUniqueGuestBackingRangeAcrossMappings(uint64_t vaddr, uint64_t size);
// On failure, optionally return a static diagnostic string (no allocation).
// The mapped parts of a guest range as they are, unmapped parts as zeros (a texture pool span
// whose memory the game released). False when the backing store is unavailable.
bool                   TryReadMappedOrZero(uint64_t vaddr, void* data, uint64_t size);
// A page no GPU watcher holds that still faults on writes: back to the guest's own writable
// protection. False when the guest itself made the range read-only (or it is unknown).
bool                   RestoreGuestWritable(uint64_t vaddr, uint64_t size);
bool                   TryReadPrtBacking(uint64_t vaddr, void* data, uint64_t size,
                                         const char** failure_reason = nullptr);
[[nodiscard]] uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size);
// The end of the guest mapping (one committed virtual range) `vaddr` lies in; 0 when none.
[[nodiscard]] uint64_t MappingEnd(uint64_t vaddr);
// Whether guest mappings cover all of [vaddr, vaddr + size), and the parts they cover (as
// (address, size), in order): what can be read of a range that reaches into unmapped memory.
[[nodiscard]] bool     IsFullyMapped(uint64_t vaddr, uint64_t size);
void                   MappedParts(uint64_t vaddr, uint64_t size, std::vector<std::pair<uint64_t, uint64_t>>* parts);
// `source` names the path in crash reports (PrintWriteBacksOverlapping).
void                   WriteBacking(uint64_t vaddr, const void* data, uint64_t size,
                                    const char* source = "image") noexcept;
// Crash reports: the latest GPU write-backs into guest memory (WriteBacking, newest first) that overlap
// [vaddr, vaddr + size); false when none does.
bool                   PrintWriteBacksOverlapping(uint64_t vaddr, uint64_t size);
void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
// Before a host write of all of [vaddr, vaddr + size) (a file read, a large guest memmove): as InvalidateMemory, but
// GPU-written bytes the write replaces whole are dropped instead of read back first.
void                   InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size);
// Before plain stores of the host over all of [vaddr, vaddr + size) (a file read's copy): where the host tells the
// pages' protection (Windows), the span of those a store would fault on (the trackers' write or read protection) is
// made the host's at once (InvalidateMemoryForWrite). Nothing when no page would fault: the trackers and their locks are
// left alone (InvalidateMemoryForWrite takes the texture cache's lock, which the render thread holds for long).
bool PrepareHostOverwrite(uint64_t vaddr, uint64_t size);
// After such a write (PrepareHostOverwrite's result `prepared`): the images over the range are refreshed from the bytes
// now there. An image the render thread uploaded while the copy was under way could read pages before the copy reached
// them and protect them after (the archstone menu's preview pictures stayed garbled until the menu was opened again).
// Not for small reads with nothing protected (audio pieces). KYTY_HOST_WRITE_REFRESH=0: off.
void FinishHostOverwrite(uint64_t vaddr, uint64_t size, bool prepared);
// Prepare a host write using normal coherent invalidation; false means use guest faults.
bool TryPrepareHostWrite(uint64_t vaddr, uint64_t size);
// The same for a write of all of the range that reads none of it (InvalidateMemoryForWrite).
bool TryPrepareHostOverwrite(uint64_t vaddr, uint64_t size);
// The images over the range only (a GPU-written image's bytes here are stale: they are rewritten as InvalidateMemory
// would make them).
bool TryPrepareHostImageWrite(uint64_t vaddr, uint64_t size);
void                   InstallGpuResources(Graphics::GpuResourceManager* resources) noexcept;
// Whether a guest mapping holds vaddr once a mapping change under way is done (it waits for one).
[[nodiscard]] bool     IsGuestMappedAfterChanges(uint64_t vaddr);
// Whether the address is in a view a partial unmap is mapping again (Windows; statistics: remap_window_writes).
[[nodiscard]] bool     InRemapWindow(uint64_t vaddr) noexcept;
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
bool FreeGuestMemory(uint64_t vaddr, uint64_t size);

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
void     TestFailNextPhysicalMemoryUnmap();
void     TestFailPhysicalMemoryUnmapAfter(uint32_t successful_unmaps);
void     TestFailGuestBackingStoreUnmapAfter(uint32_t successful_unmaps);
void     TestFailNextFixedReserveRangeRegistration();
bool     TestPlaceholderRangeIsFree(uint64_t vaddr, uint64_t size);
bool     TestGuestAddressRangeIsOwned(uint64_t vaddr, uint64_t size);
bool     TestGuestBackingOutsideAddressSpace();
uint64_t TestGuestBackingSize();
bool     TestGuestFreeRangeBounds();
#endif

} // namespace Libs::LibKernel::Memory

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_ */
