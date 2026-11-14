#pragma once
// KYTY_PIN_WATCHED=1 (Windows; off by default for now): no protection change of guest memory on a page Windows can page
// out.
//
// Why: every reboot in the field logs of 11-14 (5 of them, minidumps) is bug check 0x1A / 0x41202 (MEMORY_MANAGEMENT):
// Windows' memory manager trims / pages a page of the shared-section-backed guest store while the emulator changes that
// page's protection (VirtualProtect of the write and read watches, ~46,000 a minute; issues #5 and #50, memory.cpp
// FlushTrim, memoryAddressSpace.inc BusyUnlocked, local-platform.h KeepResident). It needs RAM pressure (16 GB PCs).
//
// What: every protection change of guest memory goes through GuestLazyCommit::Protect (SysVirtualProtect, the address
// space's ApplyProtection, the texture offload's alias pages). Around it (Begin/Finish) the pages of the range that are
// views of a section (MEM_MAPPED) are VirtualLock'ed BEFORE the protection changes: a locked page stays in the working
// set, so its physical page (shared by every view) cannot be paged while its protection changes. A page that keeps a
// protection other than read-write (a watch, a guest mprotect) stays locked; when the last such page of its 64 KiB
// granule is made writable again the granule is unlocked AFTER that (VirtualUnlock only of pages QueryWorkingSetEx says
// are locked: VirtualUnlock of an unlocked page removes it from the working set, a trim).
//
// Budget: the locked bytes are capped (KYTY_PIN_BUDGET_MB, default physical RAM / 6 clamped to 512..3072 MiB: 2730 MiB
// on a 16 GB PC); the working-set minimum is raised (soft) to hold them. The GPU trackers' write watches (read-only,
// the bulk) ask first (`optional`): over the budget the page manager does not protect the range at all and queues it
// as written (PinWatched::QueueUnprotectedWrite): drained on the GPU thread before each guest command / submission,
// the caches drop their copies and upload again (correctness first: always CPU-dirty). Other protections (read
// watches of GPU-written memory, offload holds, guest mprotect) cannot be skipped: they may go 512 MiB over the budget;
// past that, or when VirtualLock fails, the protection is still made and counted (`protects on unpinned pages`, which
// must stay 0).
//
// Also, while on: the emulator's own working-set trims are off (KYTY_TRIM_RAM_SECONDS, the RAM budget's
// TrimWorkingSetTo, KYTY_RAM_FLUSH_TRIM, Kyty.Trim of parked ranges): they page guest memory, and their VirtualUnlock
// would unlock the pins.
//
// A line every 30 s: "Pin watched (30 s): ...".

#include <cstdint>
#include <utility>
#include <vector>

namespace PinWatched {

struct Token {
	uint64_t begin      = 0;
	uint64_t end        = 0;
	bool     active     = false; // Finish has work to do
	bool     persistent = false; // the new protection is not read-write: the pages stay pinned
	bool     fallback   = false; // optional and not pinned: the caller does not protect
};

#if defined(_WIN32)
bool Enabled();
// Before a protection change of [address, address + size) to `protect` (a PAGE_* value). `optional`: the caller may
// skip the protection (a write watch) when the pages cannot be pinned; Token::fallback then says so.
Token Begin(uint64_t address, uint64_t size, uint32_t protect, bool optional);
// After it (`applied`: the protection was changed).
void Finish(const Token& token, bool applied);
// The range is unmapped: its pins went with it (no VirtualUnlock).
void Forget(uint64_t address, uint64_t size);
// A write watch left unprotected: drained as a write of the range (TakeUnprotectedWrites).
void QueueUnprotectedWrite(uint64_t address, uint64_t size);
bool TakeUnprotectedWrites(std::vector<std::pair<uint64_t, uint64_t>>* out);
// Once, from a trim that is skipped: "<name> off (KYTY_PIN_WATCHED=1)".
void NoteTrimSkipped(const char* name);
#else
inline bool  Enabled() { return false; }
inline Token Begin(uint64_t, uint64_t, uint32_t, bool) { return {}; }
inline void  Finish(const Token&, bool) {}
inline void  Forget(uint64_t, uint64_t) {}
inline void  QueueUnprotectedWrite(uint64_t, uint64_t) {}
inline bool  TakeUnprotectedWrites(std::vector<std::pair<uint64_t, uint64_t>>*) { return false; }
inline void  NoteTrimSkipped(const char*) {}
#endif

// RAII around one protection change (mandatory).
struct Scope {
	Token token;
	bool  applied = false;
	Scope(uint64_t address, uint64_t size, uint32_t protect): token(Begin(address, size, protect, false)) {}
	~Scope() {
		if (token.active) Finish(token, applied);
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;
};

} // namespace PinWatched
