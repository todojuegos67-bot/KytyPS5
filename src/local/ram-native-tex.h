#pragma once
// (ram1b) KYTY_RAM_NATIVE_TEX=1|2 (off by default): Demon's Souls' textures held once, the PC's way.
//
// Ghidra 01.007.000 (eboot, ram1 + ram1b blocks): the 12 texture arrays of 0xac4160 (ETextureArrayID::Particle_*,
// LightGobo2D/Cube, BakedShadow; 0xac3f20 with flags 3 and 7: every layer committed on demand) live in VirtualMemoryGPU
// (0x10_0000_0000..); a layer slot is committed whole (faces x slice, 0x901030, op 0) when the game takes it (0xac4980)
// and decommitted whole (0x901110, op 2) when its deferred free list releases it (0xac4e00). The CPU only writes the
// layers (stream read 0x8fed14 / 0x8fedc0, the Agc tiler 0x1d2fb70); no read of a texture's bytes after its upload.
// Logs (11-17/11-18, 16 GB PC, 8 GB of video memory simulated): those arrays' guest ranges 972 MiB, 414-511 MiB of them
// committed; render targets 0.10 GB of guest copy in RAM.
//
// =1: (a) a sampled colour array of that zone, created on the GPU thread, is a sparse (partially resident) image where
//     the device has sparse residency for 2D images and the format: video memory only for the layers the game has
//     committed - bound at its creation, at each map over it, unbound at each unmap (the game's own commit/decommit,
//     sparseImage.cpp) -; anything else, or any refusal, is the dense image as before.
//     (b) The guest copy of what the GPU caches hold - the arrays' committed layers (any image of the zone with layers)
//     and the render targets (RTShared/RTPersist arenas) -, clean for two scans, goes to the RAM17 disk store and out of
//     RAM (the RAM18 cold-range path: read watchers, the RAM18 thread writes and drops; any touch, by any thread, reads it
//     back from the file and is counted). Needs native memory (KYTY_RAIZ=1) and turns on KYTY_RAM17 (ram-guard.cpp).
// =2: verification. (a) as =1, and every 5 s one sparse image's bound layers are compared, layer by layer (XXH3 of every
//     level), with a dense copy uploaded from guest memory by the current path: "DISCREPANCIA" must stay 0. (b) does not
//     drop anything: the guest copy stays; what =1 would release is counted.
// Every 5 s: the "[rnt]" line (video memory bound vs dense, layers bound/unbound, RAM released now and the touches that
// brought it back, the verification, and the first-use probe of the zone's memory no image covers yet - build 2's datum).
// Off: one static int read where it is asked.
//
// (ram2) KYTY_RAM2=1|2 (off by default; with KYTY_RAM_NATIVE_TEX=1 only): the zone's memory no image has used for
// KYTY_RAM2_AGE_S (default 60) since the game mapped it, in RAM, to the disk store and out of RAM, by the same cold-range
// path (texture-offload.h Ram2TakeNow): read watchers on, the RAM18 thread writes it to the disk store and drops it; any
// touch, by any thread, reads it back from the file. Every range read back is verified: its bytes in guest memory after
// the read hashed against those written ("DISCREPANCIA" must stay 0). =2 takes nothing and counts what =1 would take.
// Ghidra 01.007.000: the streamed textures' chunks are committed (op 0) and filled by a memcpy from the loaded chunk
// (0xaa4a60, the job of TextureStreamingChunk::ProcessTextureData 0xaa4820; CTexture::SetCurrentStreamingChunk 0x901330
// the same) or by the file read of 0x8fe750 (0x8fed14), the T# then pointed at them (0x901540); decommitted (op 2) when
// the chunk goes down (0x901330). No CPU read of those bytes found. The emulator makes an image when a draw samples the
// T#: a chunk streamed in for an object nobody draws waits. Logs 11-18 (64 GB PC, 3 runs): of what got its first image,
// after 60 s or more 0-19 MiB a run; waiting 60 s or more at a time 1.0-2.6 GB, of what waited 10 s or more 86-93 % in RAM.
// (ram2b) First run (11-18 21:22): 2.5 GB out of RAM, 0 discrepancies, then a hang. Cause: the game's heaps (cpuheap,
// cpubkt) map 2 MiB blocks in the same address zone; ram2 took some; an emulator write (a file read into a heap buffer)
// over a range taken and not dropped yet marked it "missed" (110 "MISSED host write" lines, none in runs without ram2),
// and a missed cold range is never restored (the GPU path skips cold ranges): the game's heap allocator (0x802610)
// writing there faulted for ever, each fault sending a restore to the GPU thread that did nothing (render thread idle at
// 100 % CPU), its lock held - the other threads spinning on it (0x802f90, 0x802610). Now: only the VirtualMemo
// allocator's ranges (memory map labels), never a missed ram2 range (a host write lets it go), no GPU restore for a ram2
// range, and the protection (Ram2Block): a window where a host write, a repeated fault or a slow one happens is out of
// ram2 for the rest of the run.

#include <algorithm>
#include <array>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h> // (ram2a: K32QueryWorkingSetEx, the waiting memory's residency)
#endif
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>

namespace Loader::DemonsSoulsMemMap { // (ram2b: demonsSoulsMemMap.h; lock-free, no guest read)
size_t RangesOf(const char* const* labels, size_t label_count, uint64_t* out, size_t max_pairs) noexcept;
bool   LabelAt(uint64_t address, char* out, size_t out_size) noexcept;
} // namespace Loader::DemonsSoulsMemMap

namespace RamNativeTex {

inline int Mode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_RAM_NATIVE_TEX");
		const int   m     = value != nullptr && (value[0] == '1' || value[0] == '2') && value[1] == '\0' ? value[0] - '0' : 0;
		if (m != 0) {
			std::printf("[rnt] KYTY_RAM_NATIVE_TEX=%d: arrays de texturas dispersos en la VRAM (solo capas comprometidas)%s; "
			            "linea [rnt] cada 5 s\n",
			            m,
			            m == 1 ? " + copia guest de arrays y RT al almacen en disco (RAM17) y fuera de RAM"
			                   : " + VERIFICACION (=2): capas comparadas con la via actual, la copia guest se queda");
			std::fflush(stdout);
		}
		return m;
	}();
	return mode;
}
inline bool On() {
	return Mode() != 0;
}

// (ram2) KYTY_RAM2: 0 off, 1 take (and verify every range read back), 2 count only. Needs KYTY_RAM_NATIVE_TEX=1.
inline int Ram2Mode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_RAM2");
		const int   asked = value != nullptr && (value[0] == '1' || value[0] == '2') && value[1] == '\0' ? value[0] - '0' : 0;
		if (asked == 0) return 0;
		if (Mode() != 1) {
			std::printf("[ram2] KYTY_RAM2=%d pedido pero apagado: necesita KYTY_RAM_NATIVE_TEX=1 (la sonda y la ruta en frio)\n", asked);
			std::fflush(stdout);
			return 0;
		}
		std::printf("[ram2] KYTY_RAM2=%d: memoria de la zona VirtualMemo sin imagen desde hace %s s o mas (KYTY_RAM2_AGE_S) y en RAM "
		            "%s; hasta %s MiB por pasada (KYTY_RAM2_MB); linea [ram2] cada 5 s\n",
		            asked, std::getenv("KYTY_RAM2_AGE_S") != nullptr ? std::getenv("KYTY_RAM2_AGE_S") : "60",
		            asked == 1 ? "al almacen en disco y fuera de RAM (ruta en frio; cualquier toque la lee del disco y se verifica)"
		                       : "solo contada (=2: no se toca nada)",
		            std::getenv("KYTY_RAM2_MB") != nullptr ? std::getenv("KYTY_RAM2_MB") : "128");
		std::fflush(stdout);
		return asked;
	}();
	return mode;
}
inline int Ram2AgeSeconds() {
	static const int seconds = [] {
		const char* value = std::getenv("KYTY_RAM2_AGE_S");
		const long  asked = value != nullptr && value[0] != '\0' ? std::strtol(value, nullptr, 10) : 60;
		return static_cast<int>(std::clamp<long>(asked, 10, 3600));
	}();
	return seconds;
}
inline uint64_t Ram2Budget() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_RAM2_MB");
		const unsigned long long asked = value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) : 128;
		return static_cast<uint64_t>(std::clamp<unsigned long long>(asked, 16, 4096)) << 20u;
	}();
	return bytes;
}

// (ram2) The render thread's last scan (TextureCache::RntScan): what waits that long now and where it went.
struct Ram2ScanCounters {
	std::atomic<uint64_t> waiting {0};    // waiting >= the age (the probe's waits, mapped or not)
	std::atomic<uint64_t> aged {0};       // of it, looked at this scan and mapped
	std::atomic<uint64_t> under_image {0}; // of it, under an image of the cache (not ours)
	std::atomic<uint64_t> taken {0};      // of it, a range taken already (any kind)
	std::atomic<uint64_t> held {0};       // of it, held back: touched or read back a short while ago
	std::atomic<uint64_t> other_label {0}; // (ram2b) of it, not the VirtualMemo allocator's (cpuheap, cpubkt... live there too)
	std::atomic<uint64_t> blocked {0};     // (ram2b) of it, in a window the protection took out (Ram2Block)
	std::atomic<uint64_t> not_ram {0};    // of it, not in RAM (no sample in the working set) or not plain read-write/read-only
	std::atomic<uint64_t> proposed {0};   // of it, proposed now (=2: what =1 would take)
	std::atomic<uint64_t> scan_us {0}, scan_us_max {0}, scan_us_last {0}, scans {0};
};
inline Ram2ScanCounters& Ram2Scan() {
	static auto* counters = new Ram2ScanCounters;
	return *counters;
}

// (ram2b) The protection. A window of 64 MiB where a ram2 range went wrong - a host write over it (the emulator writing
// that memory: not the streamed textures' kind), a fault repeated on the same page, a fault or a read back over
// KYTY_RAM2_SLOW_MS (default 100) - is taken out for the rest of the run: nothing more is taken there. Counted and logged
// in "[ram2]" (the first 16 with their address and the game allocator's label).
constexpr uint64_t Ram2WindowBits = 26; // 64 MiB
struct Ram2Guard {
	std::mutex                   mutex;
	std::vector<uint64_t>        windows;     // sorted window numbers (address >> Ram2WindowBits)
	std::atomic<uint64_t>        count {0};   // windows taken out
	std::atomic<uint64_t>        host_writes {0}, repeats {0}, slow {0}, escapes {0}, logged {0};
	std::atomic<uint64_t>        slow_us_max {0};
};
inline Ram2Guard& Ram2Guards() {
	static auto* guard = new Ram2Guard;
	return *guard;
}
inline int Ram2SlowMs() {
	static const int ms = [] {
		const char* value = std::getenv("KYTY_RAM2_SLOW_MS");
		const long  asked = value != nullptr && value[0] != '\0' ? std::strtol(value, nullptr, 10) : 100;
		return static_cast<int>(std::clamp<long>(asked, 5, 5000));
	}();
	return ms;
}
// Any thread: [begin, end) meets a window taken out.
inline bool Ram2Blocked(uint64_t begin, uint64_t end) {
	auto& guard = Ram2Guards();
	if (guard.count.load(std::memory_order_acquire) == 0 || end <= begin) return false;
	std::lock_guard lock(guard.mutex);
	const auto      it = std::lower_bound(guard.windows.begin(), guard.windows.end(), begin >> Ram2WindowBits);
	return it != guard.windows.end() && *it <= ((end - 1) >> Ram2WindowBits);
}
// Any thread (no other lock of the offload's held): the window of `address` taken out, `why` logged (the first 16).
inline void Ram2Block(uint64_t address, const char* why, const char* label) {
	auto&          guard  = Ram2Guards();
	const uint64_t window = address >> Ram2WindowBits;
	bool           added  = false;
	{
		std::lock_guard lock(guard.mutex);
		const auto      it = std::lower_bound(guard.windows.begin(), guard.windows.end(), window);
		if (it == guard.windows.end() || *it != window) {
			guard.windows.insert(it, window);
			added = true;
		}
	}
	if (added) guard.count.fetch_add(1, std::memory_order_release);
	if (guard.logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::printf("[ram2] PROTECCION: %s en 0x%llx (%s): zona 0x%llx..0x%llx fuera de ram2 el resto de la partida%s\n", why,
		            static_cast<unsigned long long>(address), label != nullptr && label[0] != '\0' ? label : "sin etiqueta",
		            static_cast<unsigned long long>(window << Ram2WindowBits),
		            static_cast<unsigned long long>((window + 1) << Ram2WindowBits), added ? "" : " (ya estaba)");
		std::fflush(stdout);
	}
}

// (ram2) The verification's hash of a range's bytes (64-bit words; a range is whole pages).
inline uint64_t Ram2Hash(uint64_t seed, const void* data, uint64_t size) {
	const auto* bytes = static_cast<const uint8_t*>(data);
	uint64_t    h     = seed ^ 0x9e3779b97f4a7c15ull ^ size;
	uint64_t    i     = 0;
	for (; i + 8 <= size; i += 8) {
		uint64_t word = 0;
		std::memcpy(&word, bytes + i, 8);
		h ^= word + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
		h *= 0xff51afd7ed558ccdull;
	}
	for (; i < size; i++) h = (h ^ bytes[i]) * 0x100000001b3ull;
	h ^= h >> 33u;
	h *= 0xc4ceb9fe1a85ec53ull;
	return h ^ (h >> 33u);
}

#if defined(_WIN32)
// (ram2; render thread) Whether chunk [begin, begin + size) is all committed, private and plain read-write or read-only
// (no-access: someone else's watchers) with a page in RAM among one sampled each 256 KiB (memory never written or paged
// out: nothing to gain, left alone).
inline bool Ram2Plain(uint64_t begin, uint64_t size) {
	constexpr uint64_t Sample = uint64_t {256} << 10u;
	for (uint64_t at = begin; at < begin + size;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) return false;
		const DWORD protect = info.Protect & 0xffu;
		if (info.State != MEM_COMMIT || info.Type != MEM_PRIVATE || (info.Protect & (PAGE_GUARD | PAGE_NOCACHE)) != 0 ||
		    (protect != PAGE_READWRITE && protect != PAGE_READONLY))
			return false;
		const uint64_t stop = reinterpret_cast<uint64_t>(info.BaseAddress) + static_cast<uint64_t>(info.RegionSize);
		if (stop <= at) return false;
		at = stop;
	}
	PSAPI_WORKING_SET_EX_INFORMATION samples[8] {};
	DWORD                            n = 0;
	for (uint64_t at = begin; at < begin + size && n < 8; at += Sample) samples[n++].VirtualAddress = reinterpret_cast<void*>(at);
	if (n == 0 || K32QueryWorkingSetEx(GetCurrentProcess(), samples, n * sizeof(samples[0])) == 0) return false;
	for (DWORD i = 0; i < n; i++)
		if (samples[i].VirtualAttributes.Valid != 0) return true;
	return false;
}
#else
inline bool Ram2Plain(uint64_t /*begin*/, uint64_t /*size*/) {
	return true;
}
#endif

// The VirtualMemoryGPU zone (Ghidra 01.007.000: its base 0x10_0000_0000, 0x80d9a0; [memmap] "VirtualMemo").
constexpr uint64_t ZoneBegin = 0x1000000000ull, ZoneEnd = 0x1400000000ull;
inline bool InZone(uint64_t begin, uint64_t end) {
	return begin >= ZoneBegin && end <= ZoneEnd && end > begin;
}

// (a) Sparse arrays: numbers kept by sparseImage.cpp (any thread reads them).
struct SparseCounters {
	std::atomic<uint64_t> images {0};       // sparse images alive
	std::atomic<uint64_t> made {0};         // made so far
	std::atomic<uint64_t> refused {0};      // candidates made dense (format, metadata aspect, sizes, thread)
	std::atomic<uint64_t> bound_bytes {0};  // video memory bound to layers now
	std::atomic<uint64_t> dense_bytes {0};  // what the same images take dense (their memory requirements)
	std::atomic<uint64_t> chunk_bytes {0};  // video memory allocated for layers (bound or kept free for reuse)
	std::atomic<uint64_t> layers_bound {0}; // binds so far
	std::atomic<uint64_t> layers_unbound {0};
	std::atomic<uint64_t> bind_calls {0};
	std::atomic<uint64_t> bind_us {0};
	std::atomic<uint64_t> alloc_failed {0}; // a layer left unbound: no memory for it (counted, logged)
};
inline SparseCounters& Sparse() {
	static auto* counters = new SparseCounters;
	return *counters;
}

// (=2) The verification's numbers (TextureCache::RntVerify, the offload worker compares).
struct VerifyCounters {
	std::atomic<uint64_t> runs {0}, layers {0}, equal {0}, discrepancy {0}, skipped {0}, unproven_format {0};
};
inline VerifyCounters& Verify() {
	static auto* counters = new VerifyCounters;
	return *counters;
}

// (b, =2) What =1 would release now (render thread's last scan).
inline std::atomic<uint64_t>& WouldRelease(int kind) {
	static std::atomic<uint64_t> bytes[2] {};
	return bytes[kind & 1];
}

// ---- First-use probe (build 2): how long the zone's committed memory waits for its first image ----
// The GPU thread notes each map of the zone (TextureCache::MapMemory); an image made over it (InsertImage, any thread)
// or an unmap (UnmapMemory) ends the wait. Counted by age; the line shows what still waits, by age.
struct Probe {
	std::mutex                                                                     mutex;
	std::map<uint64_t, std::pair<uint64_t, std::chrono::steady_clock::time_point>> waiting; // begin -> (end, when mapped)
	// Bytes whose first image came after <1 s, <10 s, <60 s, <300 s, later; unmapped with no image; and the latency sum.
	uint64_t used[5] {}, unmapped_unused = 0, used_count = 0;
	double   used_ms_sum = 0.0, used_ms_max = 0.0;
};
inline Probe& GetProbe() {
	static auto* probe = new Probe;
	return *probe;
}
inline int AgeBucket(double seconds) {
	return seconds < 1 ? 0 : seconds < 10 ? 1 : seconds < 60 ? 2 : seconds < 300 ? 3 : 4;
}
// (GPU thread) A map of [begin, end) in the zone: waits for its first image (what was waiting under it starts again).
inline void ProbeMap(uint64_t begin, uint64_t end) {
	if (!On() || !InZone(begin, end)) return;
	auto&           probe = GetProbe();
	std::lock_guard lock(probe.mutex);
	auto            it = probe.waiting.lower_bound(begin);
	if (it != probe.waiting.begin() && std::prev(it)->second.first > begin) --it;
	while (it != probe.waiting.end() && it->first < end) it = probe.waiting.erase(it);
	if (probe.waiting.size() < 200000) probe.waiting.emplace(begin, std::make_pair(end, std::chrono::steady_clock::now()));
}
// An image over [begin, end) (`image`) or an unmap of it: the waits under it end.
inline void ProbeEnd(uint64_t begin, uint64_t end, bool image) {
	if (!On() || end <= begin || end <= ZoneBegin || begin >= ZoneEnd) return;
	auto&           probe = GetProbe();
	const auto      now   = std::chrono::steady_clock::now();
	std::lock_guard lock(probe.mutex);
	auto            it = probe.waiting.lower_bound(begin);
	if (it != probe.waiting.begin() && std::prev(it)->second.first > begin) --it;
	while (it != probe.waiting.end() && it->first < end) {
		const uint64_t bytes = it->second.first - it->first;
		if (image) {
			const double ms = std::chrono::duration<double, std::milli>(now - it->second.second).count();
			probe.used[AgeBucket(ms / 1000.0)] += bytes;
			probe.used_count++;
			probe.used_ms_sum += ms;
			if (ms > probe.used_ms_max) probe.used_ms_max = ms;
		} else if (Ram2Mode() != 0) {
			// (ram2) Only the part unmapped ends its wait: the rest of the map stays mapped, its age kept (the game
			// decommits a texture's upper chunks and keeps the lower ones, 0x901330).
			const uint64_t lo = it->first, hi = it->second.first;
			const auto     when = it->second.second;
			const uint64_t cut_lo = std::max(lo, begin), cut_hi = std::min(hi, end);
			probe.unmapped_unused += cut_hi > cut_lo ? cut_hi - cut_lo : 0;
			it = probe.waiting.erase(it);
			if (lo < cut_lo) probe.waiting.emplace(lo, std::make_pair(cut_lo, when));
			if (cut_hi < hi) {
				it = probe.waiting.emplace(cut_hi, std::make_pair(hi, when)).first;
				++it;
			}
			continue;
		} else {
			probe.unmapped_unused += bytes;
		}
		it = probe.waiting.erase(it);
	}
}
// (ram2; render thread) The waits of `min_seconds` or more now: (begin, end), oldest first.
inline void ProbeAged(double min_seconds, std::vector<std::pair<uint64_t, uint64_t>>* out) {
	out->clear();
	auto&                                                        probe = GetProbe();
	const auto                                                   now   = std::chrono::steady_clock::now();
	std::vector<std::pair<std::chrono::steady_clock::time_point, std::pair<uint64_t, uint64_t>>> aged;
	{
		std::lock_guard lock(probe.mutex);
		for (const auto& [begin, item]: probe.waiting)
			if (std::chrono::duration<double>(now - item.second).count() >= min_seconds) aged.push_back({item.second, {begin, item.first}});
	}
	std::sort(aged.begin(), aged.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
	out->reserve(aged.size());
	for (const auto& item: aged) out->push_back(item.second);
}
// The probe's part of the "[rnt]" line (into `out`).
inline void ProbeLine(char* out, size_t size) {
	constexpr double MiB = 1024.0 * 1024.0;
	auto&            probe = GetProbe();
	const auto       now   = std::chrono::steady_clock::now();
	uint64_t         waiting[5] {};
	std::lock_guard  lock(probe.mutex);
	for (const auto& [begin, item]: probe.waiting)
		waiting[AgeBucket(std::chrono::duration<double>(now - item.second).count())] += item.first - begin;
	// (ram2a) Build 2's datum: how much of what has waited 10 s or more is in RAM (the working set), one page sampled
	// per 64 KiB (at most 16384 samples, spread evenly): committed memory nobody touched costs commit, not RAM.
	double resident_mib = -1.0, sampled_mib = 0.0;
#if defined(_WIN32)
	{
		constexpr uint64_t Step = 64 * 1024, MaxSamples = 16384;
		uint64_t           total = 0;
		for (const auto& [begin, item]: probe.waiting)
			if (std::chrono::duration<double>(now - item.second).count() >= 10.0) total += item.first - begin;
		if (total > 0) {
			const uint64_t step = std::max<uint64_t>(Step, (total / MaxSamples + Step - 1) / Step * Step);
			std::vector<PSAPI_WORKING_SET_EX_INFORMATION> q;
			q.reserve(static_cast<size_t>(total / step + 16));
			for (const auto& [begin, item]: probe.waiting) {
				if (std::chrono::duration<double>(now - item.second).count() < 10.0) continue;
				for (uint64_t a = begin; a < item.first && q.size() < MaxSamples + 16; a += step) {
					PSAPI_WORKING_SET_EX_INFORMATION e {};
					e.VirtualAddress = reinterpret_cast<void*>(a);
					q.push_back(e);
				}
			}
			if (!q.empty() && K32QueryWorkingSetEx(GetCurrentProcess(), q.data(), static_cast<DWORD>(q.size() * sizeof(q[0]))) != 0) {
				size_t valid = 0;
				for (const auto& e: q) valid += e.VirtualAttributes.Valid != 0 ? 1 : 0;
				sampled_mib  = total / MiB;
				resident_mib = sampled_mib * static_cast<double>(valid) / static_cast<double>(q.size());
			}
		}
	}
#endif
	std::snprintf(out, size,
	              "sonda primer uso (zona VirtualMemo, desde el arranque): con imagen tras <1 s %.0f, <10 s %.0f, <60 s %.0f, <300 s "
	              "%.0f, mas %.0f MiB (%llu, media %.0f ms, max %.0f ms); desmapeado sin imagen %.0f MiB; esperando ahora <10 s %.0f, "
	              "<60 s %.0f, <300 s %.0f, mas %.0f MiB; de lo que espera >= 10 s en RAM %.0f de %.0f MiB (ram2a, -1: sin dato)",
	              probe.used[0] / MiB, probe.used[1] / MiB, probe.used[2] / MiB, probe.used[3] / MiB, probe.used[4] / MiB,
	              static_cast<unsigned long long>(probe.used_count),
	              probe.used_count != 0 ? probe.used_ms_sum / static_cast<double>(probe.used_count) : 0.0, probe.used_ms_max,
	              probe.unmapped_unused / MiB, (waiting[0] + waiting[1]) / MiB, waiting[2] / MiB, waiting[3] / MiB, waiting[4] / MiB, resident_mib, sampled_mib);
}

} // namespace RamNativeTex
