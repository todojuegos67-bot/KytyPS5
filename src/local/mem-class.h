#pragma once
// Memory classification (KYTY_MEM_CLASS=0 turns it off; diagnostics, step 1 of keeping GPU data out of host RAM): what
// each 64 KiB granule of the guest's committed memory is actually used for, by guest virtual address. Bits only (a few
// 2 MiB bitmaps, set with atomics; the marks check the bit before writing it), summed every 30 s by a thread of its own:
//   - texture (sampled): an image the texture cache binds as a sampled texture (TextureCache::FindTexture);
//   - GPU-written: render targets, depth targets, storage images, buffers bound for writing;
//   - GPU buffers: other ranges the buffer cache binds (vertex, index, constant, texel, ...: BufferCache::ObtainBuffer);
//   - from disk: destinations of file reads (AprRead, sceKernelRead/Pread);
//   - CPU after GPU: a CPU access the emulator traps after the granule was bound by the GPU: write faults on watched
//     pages (the caches' write watches, first writes), read faults on GPU-written pages, game touches of offloaded
//     textures. CPU reads of memory nothing read-watches (most textures and meshes) cannot be seen: "never read by the
//     CPU afterwards" means no trapped access, a lower bound on CPU use, not a proof.
// Not seen: GPU accesses through the BDA page table, DMA / CP writes, constant data copied into upload rings, aliases
// of the same backing mapped at another address.

#include "kernel/memory.h"
#include "fresh-memory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <thread>
#include <vector>

namespace MemClass {

constexpr uint64_t GranuleBits = 16;
constexpr uint64_t Limit       = uint64_t {1} << 40u; // (guest user space ends at 0xfc00000000)
constexpr uint64_t Words       = (Limit >> GranuleBits) / 64;

enum Map : int { Tex = 0, GpuWrite, Buf, Disk, CpuWrite, CpuRead, MapCount };

struct State {
	std::atomic<uint64_t>* maps[MapCount] {};
};

inline void StartReporter();

inline bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_MEM_CLASS");
		// (11-13 RAM diet: measurement only, off by default; KYTY_MEM_CLASS=1 turns it on.)
		return value != nullptr && value[0] != '\0' && std::string_view(value) != "0";
	}();
	return on;
}

inline State* Get() {
	static State* state = []() -> State* {
		if (!Enabled()) return nullptr;
		auto* made = new State;
		for (auto& map: made->maps) {
			// (calloc: zero pages committed only where a bit is set)
			map = static_cast<std::atomic<uint64_t>*>(std::calloc(Words, sizeof(std::atomic<uint64_t>)));
			if (map == nullptr) return nullptr;
		}
		StartReporter();
		return made;
	}();
	return state;
}

inline bool Test(const std::atomic<uint64_t>* map, uint64_t granule) {
	return (map[granule >> 6u].load(std::memory_order_relaxed) & (uint64_t {1} << (granule & 63u))) != 0;
}

inline void SetBits(std::atomic<uint64_t>* map, uint64_t vaddr, uint64_t size) {
	if (map == nullptr || size == 0 || vaddr >= Limit) return;
	const uint64_t end  = size > Limit - vaddr ? Limit : vaddr + size;
	const uint64_t last = (end - 1) >> GranuleBits;
	for (uint64_t granule = vaddr >> GranuleBits; granule <= last;) {
		const uint64_t lo   = granule & 63u;
		const uint64_t hi   = std::min<uint64_t>(63u, last - (granule & ~uint64_t {63}));
		const uint64_t n    = hi - lo + 1;
		const uint64_t mask = (n == 64 ? ~uint64_t {0} : ((uint64_t {1} << n) - 1)) << lo;
		auto&          word = map[granule >> 6u];
		if ((word.load(std::memory_order_relaxed) & mask) != mask) word.fetch_or(mask, std::memory_order_relaxed);
		granule = (granule | 63u) + 1;
	}
}

inline void Mark(Map which, uint64_t vaddr, uint64_t size) {
	if (auto* state = Get()) SetBits(state->maps[which], vaddr, size);
}

// A CPU access the emulator trapped: counted only on granules the GPU used before.
inline void MarkCpu(bool write, uint64_t vaddr, uint64_t size) {
	auto* state = Get();
	if (state == nullptr || size == 0 || vaddr >= Limit) return;
	const uint64_t end = size > Limit - vaddr ? Limit : vaddr + size;
	for (uint64_t granule = vaddr >> GranuleBits; granule <= (end - 1) >> GranuleBits; ++granule) {
		if (!Test(state->maps[Tex], granule) && !Test(state->maps[GpuWrite], granule) && !Test(state->maps[Buf], granule))
			continue;
		SetBits(state->maps[write ? CpuWrite : CpuRead], granule << GranuleBits, 1);
	}
}

enum Cat : int { CatTex = 0, CatGpuWrite, CatBuf, CatCpu, CatUnknown, CatCount };

struct Sums {
	uint64_t cat[CatCount] {};
	uint64_t candidate = 0, candidate_tex = 0, candidate_buf = 0; // from disk, GPU only after, no trapped CPU access
	uint64_t gpu_only  = 0;                                        // GPU used, no trapped CPU access after (any source)
	uint64_t cpu_after = 0, disk = 0, written = 0;
	void     Add(const Sums& o) {
        for (int i = 0; i < CatCount; ++i) cat[i] += o.cat[i];
        candidate += o.candidate;
        candidate_tex += o.candidate_tex;
        candidate_buf += o.candidate_buf;
        gpu_only += o.gpu_only;
        cpu_after += o.cpu_after;
        disk += o.disk;
        written += o.written;
	}
};

// Whether the backing granule of guest granule `granule` was ever written (fresh memory), or -1 when unknown.
inline int WrittenOf(uint64_t vaddr) {
	auto* fresh = FreshMemory::Get();
	if (fresh == nullptr) return -1;
	const auto* backing = Libs::LibKernel::Memory::TryGetBackingPointer(vaddr, 1);
	if (backing == nullptr) return -1;
	const auto granule = FreshMemory::GranuleOf(*fresh, reinterpret_cast<uint64_t>(backing));
	if (granule == UINT64_MAX) return -1;
	return FreshMemory::IsWritten(*fresh, granule) ? 1 : 0;
}

inline Sums Classify(const State& state, uint64_t start, uint64_t size, bool gpu_prot) {
	Sums sums;
	if (size == 0) return sums;
	const uint64_t end = start + size;
	for (uint64_t at = start; at < end;) {
		const uint64_t next  = std::min(end, (at | ((uint64_t {1} << GranuleBits) - 1)) + 1);
		const uint64_t bytes = next - at;
		if (at >= Limit) {
			sums.cat[CatUnknown] += end - at;
			break;
		}
		const uint64_t g    = at >> GranuleBits;
		const bool     tex  = Test(state.maps[Tex], g);
		const bool     gw   = Test(state.maps[GpuWrite], g);
		const bool     buf  = Test(state.maps[Buf], g);
		const bool     disk = Test(state.maps[Disk], g);
		const bool     cpu  = Test(state.maps[CpuWrite], g) || Test(state.maps[CpuRead], g);
		const bool     seen = tex || gw || buf;
		const int      written = WrittenOf(at);
		if (gw) sums.cat[CatGpuWrite] += bytes;
		else if (tex) sums.cat[CatTex] += bytes;
		else if (buf) sums.cat[CatBuf] += bytes;
		else if (!gpu_prot || written == 1 || disk) sums.cat[CatCpu] += bytes;
		else sums.cat[CatUnknown] += bytes;
		if (seen && !cpu) {
			sums.gpu_only += bytes;
			if (disk) {
				sums.candidate += bytes;
				if (tex && !gw) sums.candidate_tex += bytes;
				else if (!gw) sums.candidate_buf += bytes;
			}
		}
		if (seen && cpu) sums.cpu_after += bytes;
		if (disk) sums.disk += bytes;
		if (written == 1) sums.written += bytes;
		at = next;
	}
	return sums;
}

inline unsigned long long MiB(uint64_t bytes) {
	return static_cast<unsigned long long>((bytes + (uint64_t {1} << 19u)) >> 20u);
}

inline void Report() {
	auto* state = Get();
	if (state == nullptr) return;
	const auto ranges = Libs::LibKernel::Memory::CommittedRanges();
	if (ranges.empty()) return;
	struct Block {
		uint64_t start = 0, size = 0;
		Sums     sums;
		char     name[Libs::LibKernel::Memory::KERNEL_MAXIMUM_NAME_LENGTH] {};
	};
	std::vector<Block> blocks;
	Sums               total;
	for (const auto& range: ranges) {
		const auto sums = Classify(*state, range.start, range.size, range.gpu);
		total.Add(sums);
		if (!blocks.empty() && blocks.back().start + blocks.back().size == range.start) {
			blocks.back().size += range.size;
			blocks.back().sums.Add(sums);
			continue;
		}
		Block block;
		block.start = range.start;
		block.size  = range.size;
		block.sums  = sums;
		std::snprintf(block.name, sizeof(block.name), "%s", range.name);
		blocks.push_back(block);
	}
	uint64_t all = 0;
	for (const auto bytes: total.cat) all += bytes;
	std::printf("Memory classes: %llu MiB committed = texture (sampled) %llu, render targets/GPU-written %llu, GPU buffers "
	            "%llu, CPU-only %llu, unknown %llu MiB | from disk then GPU only (native-PC candidates) %llu MiB (textures "
	            "%llu, buffers %llu); GPU-used with no CPU access trapped after %llu MiB, with one %llu MiB; filled from disk "
	            "%llu MiB, ever written %llu MiB (KYTY_MEM_CLASS=0 turns it off; off by default)\n",
	            MiB(all), MiB(total.cat[CatTex]), MiB(total.cat[CatGpuWrite]), MiB(total.cat[CatBuf]), MiB(total.cat[CatCpu]),
	            MiB(total.cat[CatUnknown]), MiB(total.candidate), MiB(total.candidate_tex), MiB(total.candidate_buf),
	            MiB(total.gpu_only), MiB(total.cpu_after), MiB(total.disk), MiB(total.written));
	const auto count = std::min<size_t>(10, blocks.size());
	std::partial_sort(blocks.begin(), blocks.begin() + static_cast<std::ptrdiff_t>(count), blocks.end(),
	                  [](const Block& a, const Block& b) { return a.size > b.size; });
	for (size_t i = 0; i < count; ++i) {
		const auto& b = blocks[i];
		std::printf("Memory classes:   0x%010llx %6llu MiB tex %llu rt/gw %llu buf %llu cpu %llu unk %llu | cand %llu "
		            "disk %llu cpu-after %llu  '%s'\n",
		            static_cast<unsigned long long>(b.start), MiB(b.size), MiB(b.sums.cat[CatTex]),
		            MiB(b.sums.cat[CatGpuWrite]), MiB(b.sums.cat[CatBuf]), MiB(b.sums.cat[CatCpu]),
		            MiB(b.sums.cat[CatUnknown]), MiB(b.sums.candidate), MiB(b.sums.disk), MiB(b.sums.cpu_after), b.name);
	}
	std::fflush(stdout);
}

inline void StartReporter() {
	std::thread([] {
		for (;;) {
			std::this_thread::sleep_for(std::chrono::seconds(30));
			Report();
		}
	}).detach();
}

} // namespace MemClass
