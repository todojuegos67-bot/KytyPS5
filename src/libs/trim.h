#pragma once
// (recorte1) KYTY_TRIM=1 (default off: without it nothing here runs and nothing changes).
//
// What the PS5-only services of Demon's Souls 01.007.000 cost here was checked first (eboot imports, the HLE code and the
// logs): the emulator already runs them as bookkeeping only (network forced offline, PSN signed out, the Http/Ssl/WebApi
// pools never allocated, no Audio3d/Voice/Ngs2 imported, trophies local through UDS events). There is nothing there
// worth cutting, and trophies are never touched. What KYTY_TRIM=1 does instead:
//   1. One pass over the emulator's C++ heap when the shader prefetch has translated everything (or 180 s after start
//      if it never does): _heapmin + HeapCompact + HeapOptimizeResources. The prefetch builds and frees ~1.5 GiB of
//      plans in 15 s on 32 threads (its "Shader prefetch memory" line); what the heap keeps committed afterwards and no
//      longer uses is given back to Windows (decommitted: nothing written to the page file). The "[trim] heap" line says
//      the heap's commit and the process's private bytes and working set before and after: the RAM given back.
//      Guest memory and every cache's live data are left alone: only free heap blocks are decommitted.
//   2. A thread census every KYTY_TRIM_CENSUS_S seconds (default 30, 0 = off): "[trim-cpu]" lines with each thread
//      group's share of a CPU in the period (the game's own threads by their names, the audio devices, the emulator's
//      workers) and how many threads only wait (under 0.1 %). Measurement only: the data for the next cut.
// For a periodic heap pass while playing there is already KYTY_SOFT_RELEASE=1 (it waits for quiet frames).
namespace Libs::Trim {

[[nodiscard]] bool Enabled();
// At emulator start (after the libraries are initialized): starts the trim thread. Nothing when off.
void Start();
// The shader prefetch has translated every program and spilled its plans: the heap pass runs now (on the trim thread,
// not the caller's). Any thread, any number of times (only the first counts). Nothing when off.
void AfterShaderPrefetch();

} // namespace Libs::Trim
