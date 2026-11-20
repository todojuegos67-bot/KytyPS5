#pragma once
// (mem1) KYTY_MEM1=1 (off by default; Windows, native guest memory: KYTY_RAIZ=1). What the mem1 lab (INFORME.md of
// mem1) found and what this switch does about it:
//  - The emulator's demand on a 16 GB PC (the game's ~9.5 GB written + the emulator's ~2 GB) is over the RAM Windows
//    leaves it (p50 1.6 GB over, p90 4.4 GB [M, 190 samples of 59 runs]). While it is over, Windows keeps "working set +
//    compression" where its room is, whatever cold memory is taken out: the cuts of RAM16-RAM20 (0.4-1.2 GB) only shrank
//    the compression store, never the total (simulated: 0.0 GB at p50 for any cut under ~1.6 GB).
//  - Only one cut is that large: the streamed textures' zone (VirtualMemo, 2.2-3.4 GB committed, ~2.2 GB in RAM, the
//    CPU never reads it [Ghidra]) kept in video memory instead (simulated -0.8 GB at p50 alone, -1.5 GB with the
//    rest). It needs the texture and buffer caches to upload from that copy on the GPU (no return to RAM): not done
//    here. This switch measures what that stage would take and when the emulator still reads that zone, and makes the
//    safe cuts:
//     1. the C++ heap's free blocks (malloc/new, ~1.7-1.9 GB committed, ~0.9 GB of it free after the shader prefetch
//        [M trim log]) given back (MEM_RESET + out of the working set: no page-file or compression copy); passes at
//        60 s, 120 s, then every 180 s, under the heap's lock (its time measured; periodic passes stop if one held it
//        over KYTY_MEM1_HEAP_MAX_MS, default 20);
//     2. the image upload ring at 128 MiB (KYTY_STAGING_RING_MB=128 unless set; the port's own low-RAM default that the
//        test launcher did not set: 256 MiB in RAM in every log);
//     3. the "[mem1-vm]" census every 30 s: VirtualMemo by where it is (RAM / taken / out of RAM), whether a GPU cache
//        image is over it, and how long ago an emulator window last read or wrote it (NoteWindow below);
//     4. the "[mem1]" line every 10 s: GB given back by each measure, the RAM the user counts (the emulator's private RAM
//        + Windows' compression) and the available memory, reads back and their time, losses, hitches it may cause.
// Off: one static bool read in the emulator's window hook (only installed with the native texture offload).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace Mem1 {

inline bool On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_MEM1");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return on;
}

// VirtualMemoryGPU's zone (Ghidra 01.007.000: base 0x10_0000_0000, 0x80d9a0), in 64 KiB granules.
constexpr uint64_t VmBase      = 0x1000000000ull;
constexpr uint64_t VmEnd       = 0x1400000000ull;
constexpr uint64_t GranuleBits = 16;
constexpr uint64_t Granule     = uint64_t {1} << GranuleBits;
constexpr uint64_t Granules    = (VmEnd - VmBase) >> GranuleBits; // 262144 (1 MiB of stamps)

inline std::chrono::steady_clock::time_point Epoch() {
	static const auto epoch = std::chrono::steady_clock::now();
	return epoch;
}

// Seconds since the epoch, plus 1 (0 = never).
inline uint32_t NowStamp() {
	return 1u + static_cast<uint32_t>(
	                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - Epoch()).count());
}

// When an emulator window last covered each granule (NowStamp; 0: never). Never freed (detached threads read it).
inline std::atomic<uint32_t>* WindowStamps() {
	static auto* stamps = new std::atomic<uint32_t>[Granules]();
	return stamps;
}

// (Any thread, the emulator's window hook: texture-offload.h NativeBeforeWindow.) A window over [vaddr, vaddr + bytes).
inline void NoteWindow(uint64_t vaddr, uint64_t bytes) {
	if (!On() || bytes == 0) return;
	const uint64_t lo = std::max(vaddr, VmBase);
	const uint64_t hi = std::min(vaddr + bytes, VmEnd);
	if (lo >= hi || vaddr + bytes < vaddr) return;
	const uint32_t now    = NowStamp();
	auto*          stamps = WindowStamps();
	const uint64_t last   = (hi - 1 - VmBase) >> GranuleBits;
	for (uint64_t g = (lo - VmBase) >> GranuleBits; g <= last; ++g)
		if (stamps[g].load(std::memory_order_relaxed) != now) stamps[g].store(now, std::memory_order_relaxed);
}

} // namespace Mem1
