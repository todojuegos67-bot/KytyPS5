#ifndef EMULATOR_SRC_LOCAL_RAM_GUARD_H_
#define EMULATOR_SRC_LOCAL_RAM_GUARD_H_
// RAM guard: the low-RAM profile, the memory-pressure response ("vaciado seguro") and a periodic RAM line.
//
// Low-RAM profile (KYTY_LOWRAM=1 on, 0 off; default on with 20 GiB of physical memory or less, KYTY_SIMULATE_RAM_MB
// counted): before anything reads them, the switches below are given low-RAM defaults where the environment does not
// set them (ApplyLowRamDefaults, the first thing main does):
//   KYTY_SOFT_RELEASE=1          heaps compacted and idle scratch buffers freed on quiet frames (no working set trim)
//   KYTY_RAM_DIET=1              driver pipelines idle 180 s released (only under RAM pressure), replaced unoptimized after 60 s
//   KYTY_PREFETCH_MIN_AVAIL_MB=1536  the shader prefetch waits while less RAM than that is left
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
//
// What the flush turns on with it (ApplyLowRamDefaults, where the environment does not set them; the switches it set
// are logged on the "RAM flush (vaciado)" line), because the pass alone can only return what is free at that moment:
//   KYTY_BACKING_ON_DEMAND=1   the guest store's page-file section committed as the game maps it (never with
//                              KYTY_BACKING_FILE / KYTY_GUEST_LAZY_COMMIT): the game maps ~11.3 of the 13.5 GiB,
//                              ~2.2-2.5 GiB less commit for the whole run. The 16 GB profile's default since 10-23.
//   KYTY_RAM_DIET=1            driver pipelines idle 180 s destroyed (made again from the static store at their next
//                              binding, ~0.4 ms on a 5090), the unoptimized ones a build replaced 60 s later: ~220 KB
//                              of the driver's RAM and commit each. Counted on the flush line ("owners" part).
//   KYTY_SOFT_RELEASE=1        idle scratch buffers registered (the pass frees those idle 10 s), heaps compacted.
// KYTY_RAM_FLUSH_LEVEL=2 ("vaciado fuerte") adds:
//   KYTY_TEXTURE_OFFLOAD=1     clean sampled-only textures' guest bytes dropped from RAM (texture-offload.h; the
//                              16 GB default; 0.8-2 GB of RAM on a 5090 desktop, 1-1 -> Stonefang -> 1-1).
//   KYTY_RAM_FLUSH_IMAGE_IDLE_SECONDS=120   images no draw used for that long (counted in flips at 60/s) leave video
//                              memory whatever the budget (the texture collector, on the render thread, at most 40 a
//                              flip; never the GPU-written ones): on NVIDIA every device-local allocation is charged
//                              to the process's commit as well (private bytes), so it lowers commit, not RAM. 0 off.
//                              Returning to a place after that long uploads its textures again (a few MB a frame for
//                              some seconds): level 2 only.
// A census line with the first pass and then every 12th ("RAM census:"): the figures the owners register
// (RegisterFigure): guest store committed / mapped / freed by the game, Vulkan device-local (= driver commit on
// NVIDIA) and host memory, image reuse pool, buffer cache, pipelines with driver handles. With
// KYTY_MEMORY_BREAKDOWN_SECONDS / KYTY_MEMORY_TAGS=1 the "Private breakdown:" line tells the rest by caller.
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
// KYTY_RAM_FLUSH_SECONDS as set (0 off), read once; and KYTY_RAM_FLUSH_LEVEL (1, or 2 for the strong flush).
[[nodiscard]] int64_t FlushSeconds();
[[nodiscard]] int     FlushLevel();
// KYTY_RAM_FLUSH_IMAGE_IDLE_SECONDS: the texture collector drops images no draw used for this long (0: not for the
// flush). Default 0 at level 1, 120 at level 2.
[[nodiscard]] uint64_t FlushImageIdleSeconds();

// RAM governor (ramtarget-1025, KYTY_RAM_TARGET_MB=<n>, default off): the PC's RAM in use (total less available, as the
// "RAM flush" line and the overlay count it, simulated with KYTY_SIMULATE_RAM_MB) held at about n MiB, adaptively. The
// guard samples it every second; below the soft threshold nothing extra runs (no cost). Thresholds (MiB):
//   KYTY_RAM_SOFT_MB       (default target - 512): stage 1, freeing starts ahead of streaming spikes: the flush pass
//                          every 2 s, idle pipelines after 60 s, texture offload takes the longest-idle verified ones.
//   target                 stage 2: the pass every second, pipelines after 30 s, offload scans every 30 flips.
//   KYTY_RAM_EMERGENCY_MB  (default target + 512, at most total - 512): stage 3, an offload scan at the next flip with
//                          the hot window cut to 3 s, pipelines after 15 s.
// Texture offload under the governor takes only enough MiB to get back under the soft threshold (less 256 MiB of
// hysteresis), longest-idle first, never a texture touched within KYTY_RAM_TARGET_HOT_SECONDS (default 10, at 60
// flips a second); only the clean, sampled-only, download-verified ones it always took. One "RAM target:" line a pass.
enum GovernorStageId : int { GovIdle = 0, GovSoft = 1, GovOver = 2, GovEmergency = 3 };
[[nodiscard]] bool     GovernorOn();
[[nodiscard]] int      GovernorStage();          // (one relaxed load; 0 when off)
[[nodiscard]] uint64_t GovernorOffloadDemand();  // bytes texture offload should still take now (0: none)
void                   NoteGovernorOffload(uint64_t bytes); // what a scan took for the demand
[[nodiscard]] bool     TakeGovernorUrgent();     // stage 3 was entered or held: scan at once (cleared by the call)

// A figure for the census line (bytes now; cheap and thread-safe: it runs on the guard's thread).
void RegisterFigure(const char* name, uint64_t (*figure)());

} // namespace RamGuard

#endif // EMULATOR_SRC_LOCAL_RAM_GUARD_H_
