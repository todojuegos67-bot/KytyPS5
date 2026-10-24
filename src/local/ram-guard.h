#ifndef EMULATOR_SRC_LOCAL_RAM_GUARD_H_
#define EMULATOR_SRC_LOCAL_RAM_GUARD_H_
// bruno-astro-1025: the todojuegos67-bot fork's RAM guard (low-RAM profile, memory-pressure flush, RAM log line,
// periodic flush), reduced to what brunoKytyPs5's tree can release safely from another thread.
//
// Low-RAM profile (KYTY_LOWRAM=1 on, 0 off; default on with 20 GiB of RAM or less, KYTY_SIMULATE_RAM_MB counted): the
// pressure watcher's thresholds a quarter higher and the periodic flush on every 30 s unless KYTY_RAM_FLUSH_SECONDS
// says otherwise. (The fork's own low-RAM switches - soft release, the RAM diet of idle pipelines, the guest store
// committed on demand, the prefetch floor - drive its caches, which this tree does not have: not set here.)
//
// Pressure watcher (KYTY_RAM_GUARD=0 off): a thread checks every second the RAM left, the commit headroom (commit limit
// less charge) and Windows' low-memory signal:
//   low       available < KYTY_RAM_LOW_AVAIL_MB (default 1/16 of the RAM, at least 1024) or headroom <
//             KYTY_RAM_LOW_COMMIT_MB (2048) or Windows signals low memory
//   critical  under half of either
// On entering low or critical (and every 10 s while it lasts) a flush pass releases what holds nothing in use: the
// process heaps' free blocks (HeapCompact) and the sources registered with RegisterFlush, logged as
// "RAM: pressure <level> (...), freed N MB". Never the working set, the page file or guest memory.
// A line every KYTY_RAM_LOG_SECONDS (default 30, 0 none): "RAM (30 s): private ..., working set ..., available ...,
// commit headroom ...".
// Periodic flush ("vaciado", KYTY_RAM_FLUSH_SECONDS=<s>, 0 off; KYTY_RAM_FLUSH_MB caps a pass, default 1024): the same
// release every s seconds, "RAM flush: freed N MB in T ms".
// Simulation (KYTY_SIMULATE_RAM_MB=<n>): the watcher counts n MiB of RAM less KYTY_SIMULATE_OS_MB (3072) less the
// emulator's working set as available, and a commit limit of n + KYTY_SIMULATE_PAGEFILE_MB (default n) MiB charged with
// the emulator's private bytes and KYTY_SIMULATE_OS_COMMIT_MB (4096); the real figures still count when lower.
#include <cstdint>

namespace RamGuard {

[[nodiscard]] bool LowRam();
// main, first: logs the profile.
void ApplyLowRamDefaults();
// Starts the watcher thread once (nothing when off).
void Start();

enum Level : int { None = 0, Low = 1, Critical = 2 };
[[nodiscard]] int PressureLevel();

// A source of re-creatable or free memory called on the watcher's thread with the level (0 for the periodic flush) and
// a byte budget; returns the bytes it dropped. It must drop nothing in use.
void RegisterFlush(const char* name, uint64_t (*flush)(int level, uint64_t budget));

} // namespace RamGuard

#endif // EMULATOR_SRC_LOCAL_RAM_GUARD_H_
