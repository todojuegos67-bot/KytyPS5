#ifndef EMULATOR_SRC_LOCAL_RAM_GUARD_H_
#define EMULATOR_SRC_LOCAL_RAM_GUARD_H_
// RAM guard: the low-RAM profile, the memory-pressure response ("vaciado seguro") and a periodic RAM line.
//
// Low-RAM profile (KYTY_LOWRAM=1 on, 0 off; default on with 20 GiB of physical memory or less, KYTY_SIMULATE_RAM_MB
// counted): before anything reads them, the switches below are given low-RAM defaults where the environment does not
// set them (ApplyLowRamDefaults, the first thing main does):
//   KYTY_SOFT_RELEASE=1          heaps compacted and idle scratch buffers freed on quiet frames (no working set trim)
//   KYTY_RAM_DIET=1              driver pipelines idle 180 s released, replaced unoptimized ones after 60 s
//   KYTY_PREFETCH_MIN_AVAIL_MB=3072  the shader prefetch waits while less RAM than that is left
//   KYTY_TEXTURE_OFFLOAD=1       clean sampled textures' guest bytes dropped from RAM (texture-offload.h)
//   KYTY_TEXTURE_SPILL=1         the levels KYTY_TEXTURE_REDUCE leaves out of large textures, which nothing reads, moved
//                                to a scratch file and dropped from RAM; read back on a touch (texture-offload.h)
//   KYTY_BACKING_ON_DEMAND=1     the guest store's page-file section committed as the game maps it, not 13.5 GiB at
//                                launch (KYTY_LOWRAM_ON_DEMAND=0 keeps it all committed; never with KYTY_BACKING_FILE or
//                                KYTY_GUEST_LAZY_COMMIT)
// Nothing of the render resolution or the game's detail changes.
//
// Pressure watcher (KYTY_RAM_GUARD=0 turns it off; Windows): a thread checks every second the physical memory left,
// the commit headroom (commit limit less charge) and Windows' low-memory signal (CreateMemoryResourceNotification).
//   low       available < KYTY_RAM_LOW_AVAIL_MB (default 1/16 of the RAM, at least 1024) or headroom <
//             KYTY_RAM_LOW_COMMIT_MB (default 2048) or Windows signals low memory
//   critical  under half of either
// (Images are collected and pipelines idle 60 s released only when critical.)
// Under pressure it releases only memory that holds nothing in use or can be made again, logging each pass
// ("RAM: pressure ..., freed N MB"): the heaps' free blocks decommitted, idle scratch buffers freed, registered free
// pools discarded (soft-release.h), and, read by their owners: driver pipelines idle 60 s destroyed when critical
// (made again from the static store at their next binding), images no draw used for 600 flips collected from video
// memory (uploaded again from guest memory when used; GPU-written ones only after their download, as the collector
// does), the shader prefetch held. Never the working set (no EmptyWorkingSet / SetProcessWorkingSetSize), never the
// page file, never guest memory.
// A line every KYTY_RAM_LOG_SECONDS (default 30, 0 none): private commit, working set, available RAM, commit headroom.
// With it, "RAM breakdown" (KYTY_WS_BREAKDOWN=1|0; default on in the low-RAM profile or a simulation): the working set
// by kind, sampled (QueryWorkingSetEx on one page in every KYTY_WS_BREAKDOWN_STRIDE_KB, default 64 KiB, in batches
// with the thread yielding between them): guest memory at the game's views and at the emulator's store alias (and
// pages resident at both, which the working set counts twice and RAM once; "never written": zero pages someone read),
// the emulator's private memory by allocation size (and the most resident allocations), other mapped views, code; the
// guest store's commit and how much of it was never written (commit only, no RAM unless read).
//
// Simulation (to test the flush on a big PC): KYTY_SIMULATE_RAM_MB=<n> (as on a PC with n MiB) makes the watcher
// count n MiB of RAM less KYTY_SIMULATE_OS_MB (default 3072, Windows and other programs) less the emulator's working
// set as available, and a commit limit of n + KYTY_SIMULATE_PAGEFILE_MB (default n: a page file Windows
// grew to the RAM's size) MiB charged with the emulator's
// private bytes, the guest store's commit and KYTY_SIMULATE_OS_COMMIT_MB (default 4096). KYTY_SIMULATE_COMMIT_MB=<m>
// sets that commit limit alone (also without KYTY_SIMULATE_RAM_MB). The real figures still count when lower.
// In a simulation the run ends as that PC's would: when the simulated commit is used up (no headroom left) the
// process is closed with "RAM simulada agotada: en un PC de 16 GB el juego se habria cerrado" in the log
// (KYTY_SIMULATE_RAM_ABORT=0 keeps it running on the real memory). The periodic line also counts the seconds the
// simulated PC spent under 1 GiB / 512 MiB of available RAM (where a real one pages: stutter).
//
// Periodic flush ("vaciado", KYTY_RAM_FLUSH_SECONDS=<n>, default 0 = off; the launcher's test uses 5): every n
// seconds, whatever the pressure, up to KYTY_RAM_FLUSH_MB (default 1024) are released, in order of safety, on the
// guard's thread, stopping when the amount is reached:
//   1. caches: emulator memory that holds nothing in use and is made again when needed (freed images kept for reuse,
//      pooled tiling scratch buffers, idle scratch buffers, the heaps' free blocks). Lowers the commit.
//   2. guest memory the game has released (direct memory released, flexible memory unmapped) whose pages are still
//      committed: dropped (DiscardVirtualMemory). The game's data is not touched (no mapping of it is left), the
//      commit stays (a page-file section's pages cannot be decommitted): it lowers the RAM in use only.
//   3. KYTY_RAM_FLUSH_TRIM=1 only: pages of the guest store's emulator-side alias taken out of the working set,
//      a few MiB at a time with the address space quiet (VirtualUnlock): pages the game's own views still use stay in
//      RAM; the others go to the standby list (or the page file). No commit change, and may cost page faults.
// One line a pass: MB per category, the time it took and the longest lock a game thread could wait on, private
// bytes, working set and commit (simulated when so) before and after.
#include <cstdint>

namespace RamGuard {

// The low-RAM profile is on (KYTY_LOWRAM, or 20 GiB of RAM or less).
[[nodiscard]] bool LowRam();
// main, first: the low-RAM defaults set where unset (see above). Logs what it set.
void ApplyLowRamDefaults();
// Starts the watcher thread once (nothing when off or not on Windows).
void Start();

enum Level : int { None = 0, Low = 1, Critical = 2 };
// The pressure now (one relaxed load).
[[nodiscard]] int PressureLevel();
// Physical memory left as the watcher counts it (simulated when KYTY_SIMULATE_RAM_MB is set); 0 when unknown.
[[nodiscard]] uint64_t AvailablePhysical();

// The guest backing store: its writable alias and size, and how it is charged to the commit limit.
enum class StoreCharge : int { AllCommitted = 0, OnDemand = 1, File = 2, Lazy = 3 };
void NoteGuestStore(const void* alias, uint64_t size, StoreCharge charge);

// What an owner released because of the pressure (pipelines, images): counted into the next pressure line.
enum Released : int { Pipelines = 0, Images = 1, ReleasedKinds = 2 };
void NoteReleased(Released kind, uint64_t count, uint64_t bytes);

// A source of re-creatable or free memory called on each pressure pass (on the watcher's thread) with the level;
// returns the bytes it dropped. It must drop nothing in use.
void RegisterFlush(const char* name, uint64_t (*flush)(int level));

// A source for the periodic flush (KYTY_RAM_FLUSH_SECONDS): called on the guard's thread with the bytes still wanted;
// drops at most about that much and returns the bytes it dropped (the time it holds a lock a game thread takes, at
// most `*held_us`, the longest such hold in microseconds). It must drop nothing in use.
enum FlushClass : int { FlushCaches = 0, FlushGuestFreed = 1, FlushTrim = 2, FlushClasses = 3 };
using BudgetFlushFn = uint64_t (*)(uint64_t budget_bytes, uint64_t* held_us);
void RegisterBudgetFlush(const char* name, FlushClass kind, BudgetFlushFn flush);
// The periodic flush is on (KYTY_RAM_FLUSH_SECONDS > 0 and the guard on).
[[nodiscard]] bool PeriodicFlush();

} // namespace RamGuard

#endif // EMULATOR_SRC_LOCAL_RAM_GUARD_H_
