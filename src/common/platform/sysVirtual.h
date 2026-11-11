#ifndef KYTY_COMMON_PLATFORM_SYSVIRTUAL_H_
#define KYTY_COMMON_PLATFORM_SYSVIRTUAL_H_

#include "common/common.h"
#include "common/virtualMemory.h"

namespace Common {

void SysVirtualInit();

uint64_t SysVirtualAlloc(uint64_t address, uint64_t size, VirtualMemory::Mode mode);
uint64_t SysVirtualAllocAligned(uint64_t address, uint64_t size, VirtualMemory::Mode mode,
                                uint64_t alignment);
bool     SysVirtualAllocFixed(uint64_t address, uint64_t size, VirtualMemory::Mode mode);
bool     SysVirtualCommit(uint64_t address, uint64_t size, VirtualMemory::Mode mode);
uint64_t SysVirtualReserve(uint64_t address, uint64_t size);
uint64_t SysVirtualReserveAligned(uint64_t address, uint64_t size, uint64_t alignment);
bool     SysVirtualReserveFixed(uint64_t address, uint64_t size);
bool     SysVirtualDecommit(uint64_t address, uint64_t size);
bool     SysVirtualFree(uint64_t address);
bool     SysVirtualFreeRange(uint64_t address, uint64_t size);
bool     SysVirtualProtect(uint64_t address, uint64_t size, VirtualMemory::Mode mode,
                           VirtualMemory::Mode* old_mode = nullptr);
bool     SysVirtualFlushInstructionCache(uint64_t address, uint64_t size);
bool     SysVirtualProtectWriteWatch(uint64_t address, uint64_t size, VirtualMemory::Mode mode);
void     SysVirtualWriteWatchMapped(uint64_t address, uint64_t size, VirtualMemory::Mode mode);
void     SysVirtualWriteWatchUnmapped(uint64_t address, uint64_t size);
// (Linux) VirtualMemory::ZapNoAccess; a SIGBUS of a minor-mode page, or of a write protection the bits lost, served
// (sysLinuxVirtual.cpp).
void     SysVirtualZapNoAccess(bool on);
bool     SysVirtualResolveMinorFault(uint64_t address, bool present, bool write);

} // namespace Common

#endif /* KYTY_COMMON_PLATFORM_SYSVIRTUAL_H_ */
