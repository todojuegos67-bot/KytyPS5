#ifndef KYTY_COMMON_VIRTUALMEMORY_H_
#define KYTY_COMMON_VIRTUALMEMORY_H_

#include "common/common.h"

namespace Common {

namespace VirtualMemory {

enum class Mode : uint32_t {
	NoAccess         = 0,
	Read             = 1,
	Write            = 2,
	ReadWrite        = Read | Write,
	Execute          = 4,
	ExecuteRead      = Execute | Read,
	ExecuteWrite     = Execute | Write,
	ExecuteReadWrite = Execute | Read | Write,
};

inline bool IsExecute(Mode mode) {
	return (mode == Mode::Execute || mode == Mode::ExecuteRead || mode == Mode::ExecuteWrite ||
	        mode == Mode::ExecuteReadWrite);
}

void Init();

uint64_t Alloc(uint64_t address, uint64_t size, Mode mode);
uint64_t AllocAligned(uint64_t address, uint64_t size, Mode mode, uint64_t alignment);
bool     AllocFixed(uint64_t address, uint64_t size, Mode mode);
bool     Commit(uint64_t address, uint64_t size, Mode mode);
uint64_t Reserve(uint64_t address, uint64_t size);
uint64_t ReserveAligned(uint64_t address, uint64_t size, uint64_t alignment);
bool     ReserveFixed(uint64_t address, uint64_t size);
bool     Decommit(uint64_t address, uint64_t size);
bool     Free(uint64_t address);
bool     FreeRange(uint64_t address, uint64_t size);
bool     Protect(uint64_t address, uint64_t size, Mode mode, Mode* old_mode = nullptr);
bool     FlushInstructionCache(uint64_t address, uint64_t size);

// The GPU trackers' protection of guest views (Protect elsewhere). Linux with KYTY_UFFD_WP=1: their write watch (Read)
// of a view registered by WriteWatchMapped is a userfaultfd write protection, not an mprotect (no VMA split or merge,
// ~45% cheaper a protect-fault-unprotect cycle); a write to such a page raises SIGBUS with the write error bit.
bool ProtectWriteWatch(uint64_t address, uint64_t size, Mode mode);
// A view of the guest's shared backing mapped at [address, address + size) with `mode`, or unmapped.
void WriteWatchMapped(uint64_t address, uint64_t size, Mode mode);
void WriteWatchUnmapped(uint64_t address, uint64_t size);
// While on for this thread (the GPU thread's flush of deferred read protections), ProtectWriteWatch's no-access
// protection of short runs of registered views may drop their PTEs under userfaultfd minor mode instead of an mprotect
// (Linux, KYTY_UFFD_MINOR=1: sysLinuxVirtual.cpp); elsewhere nothing changes.
void ZapNoAccess(bool on);

} // namespace VirtualMemory

} // namespace Common

#endif /* KYTY_COMMON_VIRTUALMEMORY_H_ */
