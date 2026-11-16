#include "loader/demonsSoulsMemMap.h"

#include "common/guardedCopy.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h" // (ram1: MappedParts, TryGetBackingPieces for the residency lines)

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#include <windows.h> // IWYU pragma: keep
#include <psapi.h>   // (ram1: K32QueryWorkingSetEx)
#ifdef small
#undef small // (rpcndr.h: "#define small char" breaks the Known::small field)
#endif
#endif

// Measurement only (see the header). Every guest read is a GuardedCopy from a background thread; the fault path only
// reads the published snapshot and the two recent-range rings (atomics, no lock, no allocation).
namespace Loader::DemonsSoulsMemMap {
namespace {

// ---- Demon's Souls 01.007.000 (Ghidra; vaddr + 0x900000000 at run time) ----
// CRenderBackend singleton (0xe00 bytes, ctor 0x902630): stored at eboot+0x2f65f58 by 0xaa3890.
//   +0x18 / +0x1f8  the two PerFrameRenderContexts (BeginFrame 0x903bb0 picks one through backend+0x10):
//                   +0x10/+0x18 vector<CRenderContext*>, +0xf8/+0x100 vector<CComputeContext*>
//   CRenderContext (0x911100): +0 -> Agc DCB object {+0 begin, +8 end, +0x10 cursor, +0x18 top} (init 0x1d0a4d0);
//                   the 5 MiB block: Alloc returns at 0x911304, from the RenderContextGPU dispatcher backend+0x558.
//   CComputeContext (0x914c40): is itself the Agc DCB object {+0 begin, +8 end} (Alloc returns at 0x914cb6).
//   +0x930 0xe0000 block of 7 small DCBs (0x8000 each, every 0x20000; first DCB object at +0x948)
//   +0x8d8 RenderContextFrameCPU 130 MiB, +0x8e0 RenderContextFrameGPU 20 MiB, +0xde8 112 MiB pool
// Allocators: every engine allocator (bpe::memory, base ctor 0x8018e0: +0x10 name, +0x18 type 1 CPU / 2 GPU) is
//   registered by 0x7f8a10 in the table eboot+0x2ea0df0[1..255]. Each one lists the page blocks it took from the
//   PlatformPageAllocator (0x801d00): sentinel at +0x50 (next +0x58, prev +0x60), node {+8 next, +0x10 prev,
//   +0x18 owner sentinel, +0x58 size, +0x60 base}. HeapAllocator CPU 0x2ee4880 (1 GiB block, 0x7fbc28),
//   GPU heap 100 MiB (0x7f756f); PageAllocator/BucketAllocator children of the CPUMemory/GPUMemory dispatchers.
constexpr uint64_t Image      = 0x900000000ull;
constexpr uint64_t BackendPtr = Image + 0x2f65f58;
constexpr uint64_t AllocTable = Image + 0x2ea0df0;
constexpr int64_t  WindowMs   = 5000;

constexpr size_t MaxLabels = 48;
constexpr size_t MaxRanges = 32768;
constexpr size_t PerAlloc  = 8192;
enum : uint32_t { LDcb = 0, LCtx, LCpuHeap, LGpuHeap, LPool, LRing, LFixed };
constexpr const char* FixedNames[LFixed] = {"dcb", "ctx", "cpuheap", "gpuheap", "pool", "ring"};

bool Enabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_MEMMAP");
		return !(v != nullptr && v[0] == '0');
	}();
	return on;
}

int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

template <typename T>
bool Peek(uint64_t address, T* out) {
	return address >= 0x10000 && address < (uint64_t {1} << 47u) &&
	       Common::GuardedCopy(out, reinterpret_cast<const void*>(address), sizeof(T)) == 0;
}

bool Matches(uint64_t vaddr, const uint8_t* bytes, size_t size) {
	uint8_t got[64] {};
	return size <= sizeof(got) && vaddr + size > vaddr &&
	       Common::GuardedCopy(got, reinterpret_cast<const void*>(Image + vaddr), size) == 0 &&
	       std::memcmp(got, bytes, size) == 0;
}

// The code the layouts above come from, as 01.007.000 has it. Two groups: backend (rings, pool, contexts) and
// allocators (table + page-block lists).
bool BackendCodeMatches() {
	static constexpr uint8_t Singleton[] = {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0xc5, 0xf8, 0x57, 0xc0, 0xc5,
	                                        0xf8, 0x11, 0x07, 0x48, 0x8b, 0x1d, 0xb3, 0x26, 0x4c, 0x02};
	static constexpr uint8_t Cpu[]  = {0x48, 0x89, 0x83, 0xd8, 0x08, 0x00, 0x00};
	static constexpr uint8_t Gpu[]  = {0x48, 0x89, 0x83, 0xe0, 0x08, 0x00, 0x00};
	static constexpr uint8_t Pool[] = {0x48, 0x89, 0x83, 0xe8, 0x0d, 0x00, 0x00};
	// 0x903bf5: mov rax,[r14+0x10]; vmovups [rax],xmm0; mov rcx,[rax+0x10]; mov rdx,[rax+0xf8]; mov [rax+0x18],rcx;
	// mov [rax+0x100],rdx (the per-frame vectors)
	static constexpr uint8_t Frame[] = {0x49, 0x8b, 0x46, 0x10, 0xc5, 0xf8, 0x11, 0x00, 0x48, 0x8b,
	                                    0x48, 0x10, 0x48, 0x8b, 0x90, 0xf8, 0x00, 0x00, 0x00, 0x48,
	                                    0x89, 0x48, 0x18, 0x48, 0x89, 0x90, 0x00, 0x01, 0x00, 0x00};
	// 0x911ca9 (CRenderContext::Submit): mov rax,[rdi]; mov rsi,[rax]; mov rdx,[rax+0x10]
	static constexpr uint8_t Submit[] = {0x48, 0x8b, 0x07, 0x48, 0x8b, 0x30, 0x48, 0x8b, 0x50, 0x10};
	// 0x1d0a4de: mov [rdi],rsi; 0x1d0a4ef: mov [rdi+8],rdx (DCB init: begin, end)
	static constexpr uint8_t DcbBegin[] = {0x48, 0x89, 0x37};
	static constexpr uint8_t DcbEnd[]   = {0x48, 0x89, 0x57, 0x08};
	return Matches(0xaa3890, Singleton, sizeof(Singleton)) && Matches(0x902f33, Cpu, sizeof(Cpu)) &&
	       Matches(0x902f3f, Gpu, sizeof(Gpu)) && Matches(0x902e7e, Pool, sizeof(Pool)) &&
	       Matches(0x903bf5, Frame, sizeof(Frame)) && Matches(0x911ca9, Submit, sizeof(Submit)) &&
	       Matches(0x1d0a4de, DcbBegin, sizeof(DcbBegin)) && Matches(0x1d0a4ef, DcbEnd, sizeof(DcbEnd));
}

bool AllocatorCodeMatches() {
	// 0x7f8b5a: lea rax,[rip -> 0x2ea0df0] (the registration table)
	static constexpr uint8_t Table[] = {0x48, 0x8d, 0x05, 0x8f, 0x82, 0x6a, 0x02};
	// 0x801d45: the page-block node appended to the allocator's list (+0x50 sentinel, +0x58 size, +0x60 base)
	static constexpr uint8_t List[] = {0x48, 0x8d, 0x4b, 0x50, 0x48, 0x89, 0x00, 0x48, 0x89, 0x48, 0x08, 0x48,
	                                   0x8b, 0x53, 0x60, 0x48, 0x89, 0x50, 0x10, 0x48, 0x89, 0x43, 0x60, 0x48,
	                                   0x8b, 0x50, 0x10, 0x48, 0x89, 0x42, 0x08, 0xff, 0x43, 0x70, 0x48, 0x89,
	                                   0x48, 0x18, 0x48, 0x8b, 0x48, 0x58, 0x48, 0x01, 0x4b, 0x48};
	// 0x8018f8: mov [rdi+0x10],rsi (name); mov [rdi+0x18],edx (type)
	static constexpr uint8_t Name[] = {0x48, 0x89, 0x77, 0x10, 0x89, 0x57, 0x18};
	return Matches(0x7f8b5a, Table, sizeof(Table)) && Matches(0x801d45, List, sizeof(List)) &&
	       Matches(0x8018f8, Name, sizeof(Name));
}

// ---- KYTY_VOLATILE: the granules to make volatile (written by the map thread only, read by any thread) ----
constexpr size_t      VolMax     = 4096; // (16 GiB)
constexpr uint64_t    VolGranule = 22;   // (4 MiB, the tracker's region)
std::atomic<uint64_t> g_vol[VolMax] {};
std::atomic<size_t>   g_vol_count {0};

bool VolCtx(); // (v6, below)
// KYTY_BIG_WINDOW=1: the granules found (DefaultTagg + ctx) are NOT made volatile; a CPU write fault inside one
// unprotects 1 MiB at once instead of 16 KiB (gpuResourceManager), so tracking stays exact with far fewer faults.
bool BigWindow() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_BIG_WINDOW");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}
struct VolConfig {
	bool                     on = false;
	std::vector<std::string> prefixes;
};
const VolConfig& Vol() {
	static const VolConfig config = [] {
		VolConfig   c;
		const char* v = std::getenv("KYTY_VOLATILE");
		c.on          = (v != nullptr && std::strcmp(v, "1") == 0) || BigWindow();
		if (!c.on) return c;
		const char* base  = std::getenv("KYTY_VOLATILE_ALLOCATORS");
		const char* extra = std::getenv("KYTY_VOLATILE_EXTRA_ALLOCATORS");
		std::string text  = base != nullptr ? base : "DefaultTagg";
		if (extra != nullptr && extra[0] != 0) text += std::string(",") + extra;
		size_t pos = 0;
		while (pos <= text.size()) {
			size_t next = text.find(',', pos);
			if (next == std::string::npos) next = text.size();
			std::string item = text.substr(pos, next - pos);
			while (!item.empty() && item.front() == ' ') item.erase(item.begin());
			while (!item.empty() && item.back() == ' ') item.pop_back();
			if (!item.empty()) c.prefixes.push_back(item);
			pos = next + 1;
		}
		std::string shown;
		for (const auto& p: c.prefixes) shown += (shown.empty() ? "" : ",") + p;
		std::printf("[memmap] volatile: asignadores \"%s\"%s\n", shown.c_str(), VolCtx() ? " + ctx" : " (ctx no: KYTY_VOLATILE_CTX=1)");
		std::fflush(stdout);
		return c;
	}();
	return config;
}

bool VolAllocator(const char* name) {
	// (v6) Compared without spaces, as the [faults] label is ("Default Tagged..." -> "DefaultTagg"): the name has
	// spaces, so a plain prefix never matched and nothing was registered.
	char packed[40];
	size_t n = 0;
	for (const char* p = name; *p != 0 && n + 1 < sizeof(packed); p++)
		if (*p != ' ') packed[n++] = *p;
	packed[n] = 0;
	for (const auto& p: Vol().prefixes)
		if (std::strncmp(packed, p.c_str(), p.size()) == 0) return true;
	return false;
}

// (v6) Command buffers (ctx) are not volatile unless KYTY_VOLATILE_CTX=1: a DCB uploaded half-written lost the GPU.
bool VolCtx() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_VOLATILE_CTX");
		const char* w = std::getenv("KYTY_VOLATILE");
		// (big window) ctx is safe there: still tracked, only unprotected in bigger steps.
		return (v != nullptr && v[0] == '1') || (BigWindow() && !(w != nullptr && w[0] == '1'));
	}();
	return on;
}

// (Map thread only.) The granules of [begin, begin + size) not given yet.
void VolAdd(uint64_t begin, uint64_t size, uint64_t kind) {
	static std::unordered_set<uint64_t> seen;
	if (!Vol().on || size == 0 || begin + size < begin || begin + size > (uint64_t {1} << 47u)) return;
	for (uint64_t g = begin >> VolGranule; g <= (begin + size - 1) >> VolGranule; g++) {
		const size_t n = g_vol_count.load(std::memory_order_relaxed);
		if (n == VolMax) return;
		if (!seen.insert(g).second) continue;
		g_vol[n].store(g << 1u | kind, std::memory_order_relaxed);
		g_vol_count.store(n + 1, std::memory_order_release);
	}
}

// ---- Recent ranges (dcb, ctx): a ring of [begin, end) with the time they were seen ----
template <size_t N>
struct Recent {
	struct Slot {
		std::atomic<uint64_t> begin {0}, end {0};
		std::atomic<int64_t>  ms {0};
	};
	std::atomic<uint64_t> next {0};
	Slot                  slots[N];
	void Add(uint64_t begin, uint64_t end, int64_t ms) noexcept {
		auto& s = slots[next.fetch_add(1, std::memory_order_relaxed) % N];
		s.end.store(0, std::memory_order_relaxed); // (a reader in between sees an empty slot)
		s.begin.store(begin, std::memory_order_relaxed);
		s.ms.store(ms, std::memory_order_relaxed);
		s.end.store(end, std::memory_order_release);
	}
};
Recent<65536> g_dcb;
Recent<4096>  g_ctx;

// (v4, KYTY_VOLATILE) The context DCB blocks of the last poll (both per-frame sets) and the small DCBs' block, for
// CtxBlockOf. Written by the map thread only, each pair end-first-cleared so a reader never pairs two polls' halves
// into a range wider than one block (it may miss one meanwhile: the frame re-dirty covers it).
constexpr size_t      CtxNowMax = 129;
std::atomic<uint64_t> g_ctx_now[CtxNowMax][2] {};
std::atomic<size_t>   g_ctx_now_count {0};
void PublishCtxNow(size_t i, uint64_t begin, uint64_t end) {
	if (i >= CtxNowMax) return;
	if (g_ctx_now[i][0].load(std::memory_order_relaxed) == begin &&
	    g_ctx_now[i][1].load(std::memory_order_relaxed) == end)
		return;
	g_ctx_now[i][1].store(0, std::memory_order_release);
	g_ctx_now[i][0].store(begin, std::memory_order_release);
	g_ctx_now[i][1].store(end, std::memory_order_release);
}

// ---- The published snapshot (allocators, rings, pool, small DCBs): two buffers, a sequence number each ----
struct Snapshot {
	std::atomic<uint32_t> seq {0};
	std::atomic<uint32_t> labels {0}, count {0};
	std::atomic<uint64_t> name[MaxLabels][2] {};
	std::atomic<uint64_t> begin[MaxRanges] {}; // label << 56 | begin
	std::atomic<uint64_t> end[MaxRanges] {};
};
Snapshot         g_snap[2];
std::atomic<int> g_current {-1};

// ---- (ram1) The render-target arenas: RTSharedAllocator / RTPersistAllocator page blocks (map thread writes) ----
// Ghidra 01.007.000: both are reserved whole at start-up (PS5CRenderDisplay 0x90ad80 picks the size tables 0x24f3e80 /
// 0x24f8c30; 0x91c680 -> 0x801d00 RequestVirtualPages): 687.25 / 897.4 MiB, one block each. Up to RtMax blocks each.
constexpr size_t      RtMax = 4;
std::atomic<uint64_t> g_rt[2][RtMax][2] {}; // [arena][block] {begin, end}; end stored last, cleared first
uint64_t              g_rt_build[2][RtMax][2] {};
size_t                g_rt_build_n[2] {};

// 0 RTShared*, 1 RTPersist*, -1 other (the raw name: "RTSharedAllocator", "RTPersistAllocator").
int RtKind(const char* name) {
	if (std::strncmp(name, "RTShared", 8) == 0) return 0;
	if (std::strncmp(name, "RTPersist", 9) == 0) return 1;
	return -1;
}

void PublishRt() {
	for (int a = 0; a < 2; a++) {
		for (size_t i = 0; i < RtMax; i++) {
			const uint64_t b = i < g_rt_build_n[a] ? g_rt_build[a][i][0] : 0;
			const uint64_t e = i < g_rt_build_n[a] ? g_rt_build[a][i][1] : 0;
			if (g_rt[a][i][0].load(std::memory_order_relaxed) == b && g_rt[a][i][1].load(std::memory_order_relaxed) == e)
				continue;
			g_rt[a][i][1].store(0, std::memory_order_release);
			g_rt[a][i][0].store(b, std::memory_order_release);
			g_rt[a][i][1].store(e, std::memory_order_release);
		}
	}
}

// ---- The builder (background thread only) ----
struct Range {
	uint64_t begin, end;
	uint32_t label;
};
struct Builder {
	char     names[MaxLabels][16] {};
	uint32_t labels = 0;
	Range    ranges[MaxRanges] {};
	uint32_t count = 0, dropped = 0;
	uint32_t allocators = 0, bad_lists = 0;
	void     Reset() {
        labels = 0;
        count = dropped = allocators = bad_lists = 0;
        for (const char* n: FixedNames) (void)Label(n);
	}
	uint32_t Label(const char* name) {
		for (uint32_t i = 0; i < labels; i++)
			if (std::strcmp(names[i], name) == 0) return i;
		if (labels == MaxLabels) return MaxLabels - 1; // (the last one takes the rest)
		std::snprintf(names[labels], sizeof(names[labels]), "%s", name);
		return labels++;
	}
	void Add(uint32_t label, uint64_t begin, uint64_t size) {
		if (size == 0 || begin + size < begin || begin + size > (uint64_t {1} << 47u)) return;
		if (count == MaxRanges) {
			dropped++;
			return;
		}
		ranges[count++] = {begin, begin + size, label};
	}
};
Builder g_build;

// Short label for an allocator: its name (type picks cpu/gpu for the three generic kinds).
void LabelFor(const char* name, uint32_t type, char* out, size_t size) {
	const char* side = type == 2 ? "gpu" : "cpu";
	if (std::strcmp(name, "HeapAllocator") == 0) std::snprintf(out, size, "%sheap", side);
	else if (std::strcmp(name, "PageAllocator") == 0) std::snprintf(out, size, "%spage", side);
	else if (std::strncmp(name, "Bucket", 6) == 0) std::snprintf(out, size, "%sbkt", side);
	else {
		size_t n = 0;
		for (const char* p = name; *p != 0 && n + 1 < size && n < 11; p++)
			if (*p != ' ') out[n++] = *p;
		out[n] = 0;
		if (n == 0) std::snprintf(out, size, "alloc%u", type);
	}
}

void ReadAllocators() {
	auto& b = g_build;
	g_rt_build_n[0] = g_rt_build_n[1] = 0;
	for (uint64_t i = 1; i < 0x100; i++) {
		uint64_t object = 0, name_ptr = 0;
		uint32_t type   = 0;
		if (!Peek(AllocTable + i * 8, &object) || object == 0) continue;
		if (!Peek(object + 0x10, &name_ptr) || !Peek(object + 0x18, &type)) continue;
		char name[40] {};
		if (name_ptr >= 0x10000 && name_ptr < (uint64_t {1} << 47u)) {
			const uint64_t missing = Common::GuardedCopy(name, reinterpret_cast<const void*>(name_ptr), sizeof(name) - 1);
			name[sizeof(name) - 1 - std::min<uint64_t>(missing, sizeof(name) - 1)] = 0;
			for (char& c: name)
				if (c != 0 && (c < 0x20 || c > 0x7e)) c = '_';
		}
		char label[16];
		LabelFor(name, type, label, sizeof(label));
		const uint32_t l = b.Label(label);
		const bool     vol = Vol().on && VolAllocator(name);
		const int      rt  = RtKind(name);
		b.allocators++;
		const uint64_t sentinel = object + 0x50;
		uint64_t       node     = 0;
		if (!Peek(object + 0x58, &node)) continue;
		for (size_t n = 0; node != sentinel; n++) {
			uint64_t owner = 0, size = 0, base = 0, next = 0;
			if (n == PerAlloc || !Peek(node + 0x18, &owner) || owner != sentinel || !Peek(node + 0x58, &size) ||
			    !Peek(node + 0x60, &base) || !Peek(node + 8, &next)) {
				b.bad_lists++; // (moved while read, or not this layout: what was read so far stays)
				break;
			}
			if (size < (uint64_t {1} << 40u)) {
				b.Add(l, base, size);
				if (vol) VolAdd(base, size, 0);
				if (rt >= 0 && size != 0 && g_rt_build_n[rt] < RtMax) {
					g_rt_build[rt][g_rt_build_n[rt]][0] = base;
					g_rt_build[rt][g_rt_build_n[rt]][1] = base + size;
					g_rt_build_n[rt]++;
				}
			}
			node = next;
		}
	}
}

struct Known {
	uint64_t ring_cpu = 0, ring_gpu = 0, pool = 0, small = 0;
};

Known ReadBackend(uint64_t backend) {
	Known k;
	auto& b = g_build;
	(void)Peek(backend + 0x8d8, &k.ring_cpu);
	(void)Peek(backend + 0x8e0, &k.ring_gpu);
	(void)Peek(backend + 0xde8, &k.pool);
	uint64_t small = 0, first = 0;
	if (Peek(backend + 0x930, &small) && Peek(backend + 0x948, &first) && small != 0 && first == small) k.small = small;
	if (k.ring_cpu != 0) b.Add(LRing, k.ring_cpu, 0x8200000);
	if (k.ring_gpu != 0) b.Add(LRing, k.ring_gpu, 0x1400000);
	if (k.pool != 0) b.Add(LPool, k.pool, 0x7000000);
	if (k.small != 0) {
		b.Add(LCtx, k.small, 0xe0000);
		if (VolCtx()) VolAdd(k.small, 0xe0000, 1);
		if (Vol().on) PublishCtxNow(0, k.small, k.small + 0xe0000);
	}
	return k;
}

// The contexts of both per-frame sets; each range goes into g_ctx when it is new or every second.
uint32_t PollContexts(uint64_t backend, int64_t now) {
	static uint64_t last[128][2] {};
	static uint32_t last_n  = 0;
	static int64_t  last_ms = 0;
	uint64_t        cur[128][2];
	uint32_t        n = 0;
	auto            take = [&](uint64_t dcb_object) {
        uint64_t begin = 0, end = 0;
        if (n < 128 && Peek(dcb_object, &begin) && Peek(dcb_object + 8, &end) && end > begin &&
            end - begin <= (uint64_t {64} << 20u)) {
            cur[n][0] = begin;
            cur[n][1] = end;
            n++;
        }
	};
	for (const uint64_t set: {backend + 0x18, backend + 0x1f8}) {
		for (const uint64_t vec: {set + 0x10, set + 0xf8}) {
			uint64_t first = 0, past = 0;
			if (!Peek(vec, &first) || !Peek(vec + 8, &past) || past < first || past - first > 64 * 8) continue;
			uint64_t items[64] {};
			const uint64_t bytes = past - first;
			if (bytes == 0 || Common::GuardedCopy(items, reinterpret_cast<const void*>(first), bytes) != 0) continue;
			for (uint64_t i = 0; i < bytes / 8; i++) {
				if (items[i] == 0) continue;
				if (vec == set + 0x10) {
					uint64_t dcb = 0;
					if (Peek(items[i], &dcb)) take(dcb);
				} else {
					take(items[i]);
				}
			}
		}
	}
	const bool refresh = now - last_ms >= 1000;
	for (uint32_t i = 0; i < n; i++) {
		bool seen = false;
		for (uint32_t j = 0; j < last_n && !seen; j++) seen = last[j][0] == cur[i][0] && last[j][1] == cur[i][1];
		if (!seen || refresh) g_ctx.Add(cur[i][0], cur[i][1], now);
		if (!seen && VolCtx()) VolAdd(cur[i][0], cur[i][1] - cur[i][0], 1);
	}
	if (refresh) last_ms = now;
	if (Vol().on) {
		for (uint32_t i = 0; i < n; i++) PublishCtxNow(1 + i, cur[i][0], cur[i][1]);
		const size_t count = std::max<size_t>(g_ctx_now_count.load(std::memory_order_relaxed), 1 + n);
		for (size_t i = 1 + n; i < count && i < CtxNowMax; i++) PublishCtxNow(i, 0, 0);
		g_ctx_now_count.store(std::min<size_t>(1 + n, CtxNowMax), std::memory_order_release);
	}
	std::memcpy(last, cur, sizeof(uint64_t) * 2 * n);
	last_n = n;
	return n;
}

void Publish() {
	const auto& b    = g_build;
	const int   cur  = g_current.load(std::memory_order_acquire);
	const int   next = cur == 0 ? 1 : 0;
	auto&       s    = g_snap[next];
	s.seq.fetch_add(1, std::memory_order_acq_rel); // odd: being written
	std::atomic_thread_fence(std::memory_order_release);
	s.labels.store(b.labels, std::memory_order_relaxed);
	for (uint32_t i = 0; i < b.labels; i++) {
		uint64_t words[2] {};
		std::memcpy(words, b.names[i], sizeof(words));
		s.name[i][0].store(words[0], std::memory_order_relaxed);
		s.name[i][1].store(words[1], std::memory_order_relaxed);
	}
	for (uint32_t i = 0; i < b.count; i++) {
		s.begin[i].store((uint64_t {b.ranges[i].label} << 56u) | b.ranges[i].begin, std::memory_order_relaxed);
		s.end[i].store(b.ranges[i].end, std::memory_order_relaxed);
	}
	s.count.store(b.count, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_release);
	s.seq.fetch_add(1, std::memory_order_release); // even: done
	g_current.store(next, std::memory_order_release);
}

// "[memmap]": per label, the number of ranges, the MiB they cover and the 4 largest. Returns a signature (labels,
// counts, MiB) so it is printed again only when that changes.
uint64_t Signature() {
	const auto& b = g_build;
	uint64_t    h = 1469598103934665603ull;
	for (uint32_t i = 0; i < b.count; i++) {
		h = (h ^ b.ranges[i].label) * 1099511628211ull;
		h = (h ^ ((b.ranges[i].end - b.ranges[i].begin) >> 20u)) * 1099511628211ull;
	}
	return h ^ b.count;
}

void Print(const Known& k, uint32_t contexts, bool backend_ok, bool alloc_ok) {
	const auto& b = g_build;
	std::printf("[memmap] %u asignadores leidos, %u rangos%s%s; backend %s, asignadores %s; contextos ahora %u\n",
	            b.allocators, b.count, b.dropped != 0 ? " (lleno: faltan algunos)" : "",
	            b.bad_lists != 0 ? " (alguna lista cambio al leerla)" : "", backend_ok ? "ok" : "codigo distinto/no",
	            alloc_ok ? "ok" : "codigo distinto/no", contexts);
	std::printf("[memmap]   ring CPU 0x%010llx+130 MiB, ring GPU 0x%010llx+20 MiB, pool 0x%010llx+112 MiB, DCB "
	            "pequenos 0x%010llx+896 KiB\n",
	            static_cast<unsigned long long>(k.ring_cpu), static_cast<unsigned long long>(k.ring_gpu),
	            static_cast<unsigned long long>(k.pool), static_cast<unsigned long long>(k.small));
	for (uint32_t l = LCpuHeap; l < b.labels; l++) {
		if (l == LPool || l == LRing) continue;
		uint32_t n = 0;
		uint64_t total = 0;
		uint32_t top[4] = {~0u, ~0u, ~0u, ~0u};
		for (uint32_t i = 0; i < b.count; i++) {
			if (b.ranges[i].label != l) continue;
			n++;
			const uint64_t size = b.ranges[i].end - b.ranges[i].begin;
			total += size;
			for (int t = 0; t < 4; t++) {
				if (top[t] == ~0u || size > b.ranges[top[t]].end - b.ranges[top[t]].begin) {
					for (int u = 3; u > t; u--) top[u] = top[u - 1];
					top[t] = i;
					break;
				}
			}
		}
		if (n == 0) continue;
		char line[256];
		int  used = std::snprintf(line, sizeof(line), "[memmap]   %-11s %5u rangos %7.1f MiB:", b.names[l], n,
		                          static_cast<double>(total) / (1024.0 * 1024.0));
		for (int t = 0; t < 4 && top[t] != ~0u && used > 0 && static_cast<size_t>(used) < sizeof(line); t++) {
			const auto& r = b.ranges[top[t]];
			used += std::snprintf(line + used, sizeof(line) - used, " 0x%010llx+%.1fM",
			                      static_cast<unsigned long long>(r.begin),
			                      static_cast<double>(r.end - r.begin) / (1024.0 * 1024.0));
		}
		std::printf("%s\n", line);
	}
	std::fflush(stdout);
}

// ---- (ram1) "[memmap-ram]": the RAM each label holds (map thread, every 10 s) ----
bool ResidencyOn() {
	static const bool on = [] {
		// (ram2) Off unless asked (=1): each pass is 90 000-130 000 working-set queries (logs 11-15 04:54: 33-72 ms,
		// 552 and 916 ms under paging) every 10 s; the "[ram]" line (KYTY_RAM_LINE=1) gives the totals with none.
		const char* v = std::getenv("KYTY_MEMMAP_RESIDENCY");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

#if defined(_WIN32)
struct RamSample {
	uint64_t page   = 0; // guest address (the game's view)
	uint64_t alias  = 0; // the emulator's alias of it, 0 when the game does not map it
	uint32_t weight = 0; // bytes it stands for
	uint16_t label  = 0;
	bool     mapped = false;
	uint8_t  state  = 0; // 1 game view in the working set, 2 alias in the working set
};

// The system's page lists (NtQuerySystemInformation, SystemMemoryListInformation = 80); false when refused.
bool PageLists(uint64_t* free_mib, uint64_t* standby_mib, uint64_t* modified_mib, uint64_t* modified_pf_mib) {
	struct MemoryLists {
		ULONG_PTR zero, free, modified, modified_no_write, bad;
		ULONG_PTR standby_by_priority[8];
		ULONG_PTR repurposed_by_priority[8];
		ULONG_PTR modified_page_file;
	};
	using Query           = LONG(WINAPI*)(int, void*, ULONG, ULONG*);
	static const auto fn  = reinterpret_cast<Query>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtQuerySystemInformation")));
	if (fn == nullptr) return false;
	MemoryLists lists {};
	ULONG       got = 0;
	if (fn(80, &lists, sizeof(lists), &got) < 0) return false;
	constexpr uint64_t PagesPerMiB = 256;
	uint64_t           standby     = 0;
	for (const auto n: lists.standby_by_priority) standby += n;
	*free_mib        = (static_cast<uint64_t>(lists.free) + lists.zero) / PagesPerMiB;
	*standby_mib     = standby / PagesPerMiB;
	*modified_mib    = static_cast<uint64_t>(lists.modified) / PagesPerMiB;
	*modified_pf_mib = static_cast<uint64_t>(lists.modified_page_file) / PagesPerMiB;
	return true;
}

void ReportResidency() {
	const auto& b = g_build;
	if (b.count == 0) return;
	const auto           t0         = std::chrono::steady_clock::now();
	constexpr uint64_t   Granule    = uint64_t {64} << 10u;
	constexpr uint64_t   MaxSamples = uint64_t {1} << 17u;
	uint64_t             total      = 0;
	for (uint32_t i = 0; i < b.count; i++) total += b.ranges[i].end - b.ranges[i].begin;
	// One sample per `stride` (64 KiB, more when the labels cover more than 8 GiB), each standing for its bytes.
	const uint64_t stride = std::max(Granule, ((total / MaxSamples) + Granule - 1) & ~(Granule - 1));
	std::vector<RamSample>                          samples;
	std::vector<std::pair<uint64_t, uint64_t>>      parts;
	std::vector<std::pair<const uint8_t*, uint64_t>> pieces;
	std::vector<uint64_t>                           mapped_bytes(b.labels, 0), range_bytes(b.labels, 0);
	samples.reserve(static_cast<size_t>(std::min<uint64_t>(total / stride + b.count, MaxSamples * 2)));
	for (uint32_t i = 0; i < b.count; i++) {
		const auto& r = b.ranges[i];
		range_bytes[r.label] += r.end - r.begin;
		Libs::LibKernel::Memory::MappedParts(r.begin, r.end - r.begin, &parts);
		for (const auto& [pb, ps]: parts) mapped_bytes[r.label] += ps;
		// (The alias of each mapped part looked up once: one shared lock of the address space a part, not a sample.)
		size_t   part = 0, piece = 0;
		uint64_t piece_begin = 0;
		bool     have_pieces = false;
		for (uint64_t at = r.begin; at < r.end; at += stride) {
			RamSample sample;
			sample.page   = at & ~uint64_t {0xfff};
			sample.weight = static_cast<uint32_t>(std::min(stride, r.end - at));
			sample.label  = static_cast<uint16_t>(r.label);
			while (part < parts.size() && parts[part].first + parts[part].second <= at) {
				part++;
				have_pieces = false;
			}
			sample.mapped = part < parts.size() && parts[part].first <= at;
			if (sample.mapped) {
				if (!have_pieces) {
					have_pieces = true;
					pieces.clear();
					if (!Libs::LibKernel::Memory::TryGetBackingPieces(parts[part].first, parts[part].second, pieces))
						pieces.clear();
					piece       = 0;
					piece_begin = parts[part].first;
				}
				while (piece < pieces.size() && at >= piece_begin + pieces[piece].second) piece_begin += pieces[piece++].second;
				if (piece < pieces.size() && pieces[piece].first != nullptr && at >= piece_begin)
					sample.alias = (reinterpret_cast<uint64_t>(pieces[piece].first) + (at - piece_begin)) & ~uint64_t {0xfff};
			}
			samples.push_back(sample);
		}
	}
	// The working-set state of both pages of each sample, 1024 samples a call, the CPU yielded between calls (the
	// kernel's walk holds the working-set lock the game's page faults wait for).
	constexpr size_t                              Batch = 1024;
	std::vector<PSAPI_WORKING_SET_EX_INFORMATION> entries(Batch * 2);
	HANDLE                                        process = GetCurrentProcess();
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
	bool ok = true;
	for (size_t begin = 0; begin < samples.size() && ok; begin += Batch) {
		const size_t n = std::min(Batch, samples.size() - begin);
		for (size_t i = 0; i < n; i++) {
			const auto& sample                = samples[begin + i];
			entries[i * 2]                    = {};
			entries[i * 2 + 1]                = {};
			entries[i * 2].VirtualAddress     = reinterpret_cast<PVOID>(sample.page);
			entries[i * 2 + 1].VirtualAddress = reinterpret_cast<PVOID>(sample.alias != 0 ? sample.alias : sample.page);
		}
		if (begin != 0) SwitchToThread();
		if (K32QueryWorkingSetEx(process, entries.data(), static_cast<DWORD>(n * 2 * sizeof(entries[0]))) == 0) {
			ok = false;
			break;
		}
		for (size_t i = 0; i < n; i++) {
			auto&      sample = samples[begin + i];
			const bool view   = (entries[i * 2].VirtualAttributes.Flags & 1u) != 0;
			const bool alias  = sample.alias != 0 && (entries[i * 2 + 1].VirtualAttributes.Flags & 1u) != 0;
			sample.state      = static_cast<uint8_t>((view ? 1 : 0) | (alias ? 2 : 0));
		}
	}
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
	if (!ok) {
		std::printf("[memmap-ram] K32QueryWorkingSetEx fallo (0x%08x): sin medida\n", static_cast<unsigned>(GetLastError()));
		std::fflush(stdout);
		return;
	}
	struct Totals {
		uint64_t ram = 0, view = 0, alias = 0, dead = 0;
	};
	std::vector<Totals> by(b.labels);
	for (const auto& sample: samples) {
		auto& t = by[sample.label];
		if (sample.state != 0) t.ram += sample.weight;
		if ((sample.state & 1u) != 0) t.view += sample.weight;
		if ((sample.state & 2u) != 0) t.alias += sample.weight;
		if (!sample.mapped && (sample.state & 1u) != 0) t.dead += sample.weight;
	}
	constexpr double MiB = 1024.0 * 1024.0;
	const double     ms  = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	uint64_t         all_ram = 0, all_dead = 0;
	for (const auto& t: by) {
		all_ram += t.ram;
		all_dead += t.dead;
	}
	uint64_t   free_mib = 0, standby_mib = 0, modified_mib = 0, modified_pf_mib = 0;
	const bool lists = PageLists(&free_mib, &standby_mib, &modified_mib, &modified_pf_mib);
	char       sys[160];
	if (lists)
		std::snprintf(sys, sizeof(sys), "sistema: libre %llu, standby %llu, modificada %llu (de pagefile %llu)",
		              static_cast<unsigned long long>(free_mib), static_cast<unsigned long long>(standby_mib),
		              static_cast<unsigned long long>(modified_mib), static_cast<unsigned long long>(modified_pf_mib));
	else
		std::snprintf(sys, sizeof(sys), "sistema: listas de paginas no disponibles (Windows lo negó)");
	std::printf("[memmap-ram] 10 s, MiB (KYTY_MEMMAP_RESIDENCY=1; muestra cada %llu KiB, %zu muestras, %.1f ms): "
	            "en RAM en los asignadores %.0f (de ello no mapeado por el juego %.0f); %s\n",
	            static_cast<unsigned long long>(stride >> 10u), samples.size(), ms, all_ram / MiB, all_dead / MiB, sys);
	// Per label: rango / mapeado / en RAM (vista del juego, alias) / en RAM sin mapear. (Labels whose ranges overlap -
	// a heap's blocks inside VirtualMemory - count in both.)
	char   line[512];
	size_t used = 0;
	line[0]     = 0;
	for (uint32_t l = 0; l < b.labels; l++) {
		if (range_bytes[l] < (uint64_t {1} << 20u)) continue;
		const auto& t = by[l];
		char        item[160];
		const int   n = std::snprintf(item, sizeof(item), "%s %.0f/%.0f/%.0f(j%.0f a%.0f)/%.0f", b.names[l],
		                              range_bytes[l] / MiB, mapped_bytes[l] / MiB, t.ram / MiB, t.view / MiB, t.alias / MiB,
		                              t.dead / MiB);
		if (n <= 0) continue;
		if (used != 0 && used + static_cast<size_t>(n) + 2 > 220) {
			std::printf("[memmap-ram]  %s\n", line);
			used    = 0;
			line[0] = 0;
		}
		const int m = std::snprintf(line + used, sizeof(line) - used, "%s%s", used == 0 ? "" : "  ", item);
		if (m > 0) used = std::min(sizeof(line) - 1, used + static_cast<size_t>(m));
	}
	if (used != 0) std::printf("[memmap-ram]  %s\n", line);
	std::printf("[memmap-ram]  (por etiqueta: rango/mapeado/en RAM(j vista del juego, a alias del emulador)/en RAM sin "
	            "mapear)\n");
	std::fflush(stdout);
}
#else
void ReportResidency() {}
#endif

void Thread() {
	bool     checked = false, backend_ok = false, alloc_ok = false;
	int64_t  last_build = 0, last_print = 0, last_ram = 0;
	uint64_t last_sig   = 0;
	uint32_t contexts   = 0;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		const int64_t now = NowMs();
		if (!checked) {
			checked    = true;
			backend_ok = BackendCodeMatches();
			alloc_ok   = AllocatorCodeMatches();
			std::printf("[memmap] on (KYTY_MEMMAP=0 lo apaga): backend %s, asignadores %s\n",
			            backend_ok ? "ok" : "codigo distinto, sin ring/pool/ctx",
			            alloc_ok ? "ok" : "codigo distinto, sin heaps");
			std::fflush(stdout);
			if (!backend_ok && !alloc_ok) return; // (dcb from the submits still works)
		}
		uint64_t backend = 0;
		if (backend_ok && (!Peek(BackendPtr, &backend) || backend == 0)) backend = 0;
		if (backend != 0) contexts = PollContexts(backend, now);
		if (now - last_build < 2000) continue;
		last_build = now;
		g_build.Reset();
		Known k;
		if (backend != 0) k = ReadBackend(backend);
		if (alloc_ok) {
			ReadAllocators();
			PublishRt();
		}
		Publish();
		if (alloc_ok && ResidencyOn() && now - last_ram >= 10000) {
			last_ram = now;
			ReportResidency();
		}
		const uint64_t sig = Signature();
		if (sig != last_sig && (last_print == 0 || now - last_print >= 10000)) {
			last_sig   = sig;
			last_print = now;
			Print(k, contexts, backend_ok && backend != 0, alloc_ok);
		}
	}
}

void StartOnce() {
	static std::atomic<int> state {0}; // 0 not yet, 1 started / not this game
	int expected = 0;
	if (state.load(std::memory_order_relaxed) != 0 || !state.compare_exchange_strong(expected, 1)) return;
	if (!Libs::Graphics::DemonsSouls::IsSupportedGame()) return;
	std::thread(Thread).detach();
}

// Marks [b, e) clipped to the region (4 KiB pages, 512 per 2 MiB) in `bits`.
void Mark(uint64_t* bits, uint64_t region, uint64_t pages, uint64_t b, uint64_t e) {
	const uint64_t region_end = region + pages * 4096;
	if (e <= region || b >= region_end || e <= b) return;
	const uint64_t first = (std::max(b, region) - region) >> 12u;
	const uint64_t last  = (std::min(e, region_end) - 1 - region) >> 12u;
	for (uint64_t p = first; p <= last; p++) bits[p >> 6u] |= uint64_t {1} << (p & 63u);
}

uint32_t Percent(const uint64_t* bits, uint64_t pages) {
	uint64_t set = 0;
	for (uint64_t w = 0; w < 8; w++) set += static_cast<uint64_t>(std::popcount(bits[w]));
	return pages == 0 ? 0 : static_cast<uint32_t>(set * 100 / pages);
}

template <size_t N>
void MarkRecent(const Recent<N>& ring, uint64_t* bits, uint64_t region, uint64_t pages, int64_t now) {
	for (const auto& s: ring.slots) {
		const uint64_t e = s.end.load(std::memory_order_acquire);
		if (e == 0) continue;
		const uint64_t b  = s.begin.load(std::memory_order_relaxed);
		const int64_t  ms = s.ms.load(std::memory_order_relaxed);
		if (now - ms > WindowMs) continue;
		Mark(bits, region, pages, b, e);
	}
}

} // namespace

void NoteSubmit(const void* commands, uint64_t size_in_bytes) noexcept {
	if (!Enabled() || commands == nullptr || size_in_bytes == 0) return;
	const uint64_t begin = reinterpret_cast<uint64_t>(commands);
	g_dcb.Add(begin, begin + size_in_bytes, NowMs());
	StartOnce();
}

size_t SubmitsSince(uint64_t* cursor, uint64_t* out, size_t max) noexcept {
	if (!Enabled() || cursor == nullptr) return 0;
	constexpr uint64_t N    = 65536;
	const uint64_t     next = g_dcb.next.load(std::memory_order_acquire);
	uint64_t           at   = *cursor;
	if (at > next || next - at > N) at = next > N ? next - N : 0; // (fell behind: the oldest still kept)
	size_t n = 0;
	for (; at < next && n < max; at++) {
		const auto&    slot = g_dcb.slots[at % N];
		const uint64_t end  = slot.end.load(std::memory_order_acquire);
		const uint64_t b    = slot.begin.load(std::memory_order_relaxed);
		if (end == 0 || end <= b) continue; // (being written, or overwritten meanwhile: the frame re-dirty covers it)
		out[2 * n]     = b;
		out[2 * n + 1] = end;
		n++;
	}
	*cursor = at;
	return n;
}

bool CtxBlockOf(uint64_t address, uint64_t* begin, uint64_t* end) noexcept {
	const size_t count = std::min(g_ctx_now_count.load(std::memory_order_acquire), CtxNowMax);
	for (size_t i = 0; i < count; i++) {
		const uint64_t e = g_ctx_now[i][1].load(std::memory_order_acquire);
		const uint64_t b = g_ctx_now[i][0].load(std::memory_order_acquire);
		if (e == 0 || g_ctx_now[i][1].load(std::memory_order_acquire) != e) continue;
		if (address >= b && address < e) {
			*begin = b;
			*end   = e;
			return true;
		}
	}
	return false;
}

// (ram2) The engine's page allocator (PS5PlatformPageAllocator.cpp), 01.007.000 [Ghidra]: its state is at
// *(Image + 0x2ee4d60), two memory categories of 0xa0 bytes. In each, +0x70 = 64 KiB pages in released blocks kept
// mapped (ReleasePages 0x80f550 puts a fully committed block in the category's free list and adds its pages here;
// the first-fit reuse 0x80e7a0 takes them back; they are unmapped - UnmapPhysicalPages 0x80eb90 - only when +0x74
// reaches 0, in the commit path of 0x80d9a0), +0x74 = pages free in the category's bitmap (not mapped; a commit takes
// the lowest free bit first). GetFreeMemory 0x80d540 adds the four: mov rcx,[0x2ee4d60]; mov edx,[rcx+0x74]; mov
// eax,[rcx+0x114]; add edx,[rcx+0x70]; add eax,[rcx+0x110]; add rax,rdx; shl rax,16 - the bytes checked below.
bool PageAllocatorFree(uint64_t* cached_bytes, uint64_t* unmapped_bytes) noexcept {
	static constexpr uint8_t FreeMemory[] = {0x48, 0x8b, 0x0d, 0x19, 0x78, 0x6d, 0x02, 0x8b, 0x51, 0x74, 0x8b,
	                                         0x81, 0x14, 0x01, 0x00, 0x00, 0x03, 0x51, 0x70, 0x03, 0x81, 0x10,
	                                         0x01, 0x00, 0x00, 0x48, 0x01, 0xd0, 0x48, 0xc1, 0xe0, 0x10};
	// (Not IsSupportedGame: it latches its first answer, and this may be asked before the game's metadata is loaded.
	// The code's bytes at the image's address say it is this game and version.)
	if (!Matches(0x80d540, FreeMemory, sizeof(FreeMemory))) return false;
	uint64_t state    = 0;
	uint32_t cached_0 = 0, free_0 = 0, cached_1 = 0, free_1 = 0;
	if (!Peek(Image + 0x2ee4d60, &state) || state == 0 || !Peek(state + 0x70, &cached_0) || !Peek(state + 0x74, &free_0) ||
	    !Peek(state + 0x110, &cached_1) || !Peek(state + 0x114, &free_1)) {
		return false;
	}
	*cached_bytes   = (uint64_t {cached_0} + cached_1) << 16u;
	*unmapped_bytes = (uint64_t {free_0} + free_1) << 16u;
	return true;
}

// (ram3, KYTY_DS_FREECACHE_RESET) See the header. Ghidra 01.007.000: every function that changes the allocator's
// lists (0x80d210, 0x80d640, 0x80e620, 0x80f550, 0x80f930, 0x8107a0, 0x810e90) takes the spinlock at 0x2ee4b60
// (xor eax,eax; lock cmpxchg [0x2ee4b60],r15 with r15 = -1; released by a store of 0), and flushes its queued
// sceKernelBatchMap entries (0x80f2c0) before releasing it. A category (state + c * 0xa0, c = 0, 1): +0x28 the free
// list's sentinel link, +0x38 its first link, +0x48 its node count, +0x70 pages in free blocks. A node
// (PageAllocationInfo, allocated apart from the memory it describes: 0x80d0a0): +0x20 link {+0 the node, +0x10 the
// next link, +0x18 the list's sentinel}, +0x50 bytes mapped, +0x58 size, +0x60 base.
int WithPageAllocatorFreeBlocks(FreeBlocksFn fn, void* user, uint64_t* cached_bytes) noexcept {
	// 0x80f5c8: xor eax,eax; lock cmpxchg [rip -> 0x2ee4b60],r15; jne
	static constexpr uint8_t LockCode[] = {0x31, 0xc0, 0xf0, 0x4c, 0x0f, 0xb1, 0x3d, 0x8d, 0x55, 0x6d, 0x02, 0x75, 0xbb};
	// 0x80f625 (ReleasePages): mov rcx,[rbx+0x58]; sub [rax+rdx+0x80],rcx; mov r9,rcx; mov r8,[rbx+0x50]; shr r9,16;
	// sub [rax+rdx+0x90],r8; add [rax+rdx+0x70],r9d; cmp rcx,r8
	static constexpr uint8_t ReleaseCode[] = {0x48, 0x8b, 0x4b, 0x58, 0x48, 0x29, 0x8c, 0x10, 0x80, 0x00, 0x00, 0x00, 0x49,
	                                          0x89, 0xc9, 0x4c, 0x8b, 0x43, 0x50, 0x49, 0xc1, 0xe9, 0x10, 0x4c, 0x29, 0x84,
	                                          0x10, 0x90, 0x00, 0x00, 0x00, 0x44, 0x01, 0x4c, 0x10, 0x70, 0x4c, 0x39, 0xc1};
	// 0x80f606: lea rdx,[rcx+rcx*4] ... 0x80f61a: shl edx,5 (the category's 0xa0 bytes)
	static constexpr uint8_t StrideCode[] = {0xc1, 0xe2, 0x05, 0xff, 0x4c, 0x10, 0x20};
	// 0x80e7b8 (the first-fit reuse): cmp [rdi+0x70],r13d; jb; mov rax,[rdi+0x38]; mov r12d,ecx; lea rcx,[rdi+0x28]
	static constexpr uint8_t ReuseCode[] = {0x44, 0x39, 0x6f, 0x70, 0x0f, 0x82, 0xe0, 0x01, 0x00, 0x00, 0x48,
	                                        0x8b, 0x47, 0x38, 0x41, 0x89, 0xcc, 0x48, 0x8d, 0x4f, 0x28};
	// 0x80e805: mov rsi,[r15+0x58] (a free node's size, read through the list)
	static constexpr uint8_t NodeCode[] = {0x49, 0x8b, 0x77, 0x58};
	if (fn == nullptr || !Matches(0x80f5c8, LockCode, sizeof(LockCode)) || !Matches(0x80f625, ReleaseCode, sizeof(ReleaseCode)) ||
	    !Matches(0x80f61a, StrideCode, sizeof(StrideCode)) || !Matches(0x80e7b8, ReuseCode, sizeof(ReuseCode)) ||
	    !Matches(0x80e805, NodeCode, sizeof(NodeCode))) {
		return 1;
	}
	uint64_t state = 0;
	int64_t  word  = 0;
	if (!Peek(Image + 0x2ee4d60, &state) || state == 0 || !Peek(Image + 0x2ee4b60, &word)) return 1;
	static std::atomic<bool> busy {false};
	if (busy.exchange(true, std::memory_order_acquire)) return 2;
	constexpr size_t MaxBlocks = 8192;
	static FreeBlock blocks[MaxBlocks];
	auto*            lock = reinterpret_cast<std::atomic<int64_t>*>(Image + 0x2ee4b60);
	bool             held = false;
	for (int attempt = 0; attempt < 64 && !held; attempt++) {
		int64_t expected = 0;
		held             = lock->compare_exchange_strong(expected, -1, std::memory_order_acq_rel);
		if (!held) std::this_thread::yield();
	}
	if (!held) {
		busy.store(false, std::memory_order_release);
		return 2;
	}
	size_t   count  = 0;
	uint64_t cached = 0;
	bool     ok     = true;
	for (int c = 0; c < 2 && ok; c++) {
		const uint64_t category = state + static_cast<uint64_t>(c) * 0xa0;
		const uint64_t sentinel = category + 0x28;
		uint64_t       link     = 0;
		int32_t        nodes    = 0;
		uint32_t       pages    = 0;
		ok = Peek(category + 0x38, &link) && Peek(category + 0x48, &nodes) && Peek(category + 0x70, &pages) && nodes >= 0 &&
		     static_cast<size_t>(nodes) <= MaxBlocks;
		cached += uint64_t {pages} << 16u;
		int32_t walked = 0;
		while (ok && link != sentinel) {
			uint64_t node = 0, next = 0, owner = 0, mapped = 0, size = 0, base = 0;
			ok = walked < nodes && count < MaxBlocks && Peek(link, &node) && node != 0 && link == node + 0x20 &&
			     Peek(node + 0x30, &next) && Peek(node + 0x38, &owner) && owner == sentinel && Peek(node + 0x50, &mapped) &&
			     Peek(node + 0x58, &size) && Peek(node + 0x60, &base) && size != 0 &&
			     mapped <= size && size <= (uint64_t {1} << 36u) && ((base | size | mapped) & 0xffffu) == 0 && base >= 0x10000 &&
			     base < (uint64_t {1} << 47u);
			if (!ok) break;
			blocks[count++] = {base, size, mapped};
			walked++;
			link = next;
		}
		ok = ok && walked == nodes;
	}
	if (ok) fn(user, blocks, count);
	lock->store(0, std::memory_order_release);
	busy.store(false, std::memory_order_release);
	if (cached_bytes != nullptr) *cached_bytes = cached;
	return ok ? 0 : 3;
}

int RtArenaOf(uint64_t begin, uint64_t end) noexcept {
	if (!Enabled() || end <= begin) return 0;
	for (int a = 0; a < 2; a++) {
		for (size_t i = 0; i < RtMax; i++) {
			const uint64_t e = g_rt[a][i][1].load(std::memory_order_acquire);
			if (e == 0) continue;
			const uint64_t b = g_rt[a][i][0].load(std::memory_order_acquire);
			if (g_rt[a][i][1].load(std::memory_order_acquire) != e) continue; // (being changed: not counted)
			if (begin >= b && end <= e) return a + 1;
		}
	}
	return 0;
}

bool InBigWindowGranule(uint64_t vaddr) noexcept {
	if (!BigWindow()) return false;
	const uint64_t g     = vaddr >> VolGranule;
	const size_t   count = g_vol_count.load(std::memory_order_acquire);
	for (size_t i = 0; i < count; i++)
		if ((g_vol[i].load(std::memory_order_relaxed) >> 1u) == g) return true;
	return false;
}

size_t VolatileGranules(size_t from, uint64_t* out, size_t max) noexcept {
	const size_t count = g_vol_count.load(std::memory_order_acquire);
	size_t       n     = 0;
	for (size_t i = from; i < count && n < max; i++) out[n++] = g_vol[i].load(std::memory_order_relaxed);
	return n;
}

size_t RangesOf(const char* const* labels, size_t label_count, uint64_t* out, size_t max_pairs) noexcept {
	if (!Enabled() || labels == nullptr || out == nullptr) return SIZE_MAX;
	const int cur = g_current.load(std::memory_order_acquire);
	if (cur < 0) return SIZE_MAX;
	const auto&    s  = g_snap[cur];
	const uint32_t s1 = s.seq.load(std::memory_order_acquire);
	if ((s1 & 1u) != 0) return SIZE_MAX;
	const uint32_t count = std::min<uint32_t>(s.labels.load(std::memory_order_relaxed), MaxLabels);
	const uint32_t cnt   = std::min<uint32_t>(s.count.load(std::memory_order_relaxed), MaxRanges);
	bool           wanted[MaxLabels] {};
	for (uint32_t i = 0; i < count; i++) {
		char           name[17] {};
		const uint64_t words[2] = {s.name[i][0].load(std::memory_order_relaxed), s.name[i][1].load(std::memory_order_relaxed)};
		std::memcpy(name, words, 16);
		for (size_t l = 0; l < label_count && !wanted[i]; l++) wanted[i] = labels[l] != nullptr && std::strcmp(name, labels[l]) == 0;
	}
	size_t n = 0;
	for (uint32_t i = 0; i < cnt && n < max_pairs; i++) {
		const uint64_t v = s.begin[i].load(std::memory_order_relaxed);
		const uint32_t l = static_cast<uint32_t>(v >> 56u);
		if (l >= count || !wanted[l]) continue;
		out[2 * n]     = v & ((uint64_t {1} << 56u) - 1);
		out[2 * n + 1] = s.end[i].load(std::memory_order_relaxed);
		n++;
	}
	std::atomic_thread_fence(std::memory_order_acquire);
	return s.seq.load(std::memory_order_relaxed) == s1 ? n : SIZE_MAX;
}

bool LabelAt(uint64_t address, char* out, size_t out_size) noexcept {
	if (out == nullptr || out_size == 0) return false;
	out[0] = 0;
	if (!Enabled()) return false;
	if (const int cur = g_current.load(std::memory_order_acquire); cur >= 0) {
		const auto&    s  = g_snap[cur];
		const uint32_t s1 = s.seq.load(std::memory_order_acquire);
		if ((s1 & 1u) == 0) {
			const uint32_t count = std::min<uint32_t>(s.labels.load(std::memory_order_relaxed), MaxLabels);
			const uint32_t cnt   = std::min<uint32_t>(s.count.load(std::memory_order_relaxed), MaxRanges);
			uint64_t       best  = UINT64_MAX;
			uint32_t       label = MaxLabels;
			for (uint32_t i = 0; i < cnt; i++) {
				const uint64_t v = s.begin[i].load(std::memory_order_relaxed);
				const uint64_t b = v & ((uint64_t {1} << 56u) - 1);
				const uint64_t e = s.end[i].load(std::memory_order_relaxed);
				if (address < b || address >= e || e - b >= best) continue;
				best  = e - b;
				label = static_cast<uint32_t>(v >> 56u);
			}
			char name[17] {};
			if (label < count) {
				const uint64_t words[2] = {s.name[label][0].load(std::memory_order_relaxed),
				                           s.name[label][1].load(std::memory_order_relaxed)};
				std::memcpy(name, words, 16);
			}
			std::atomic_thread_fence(std::memory_order_acquire);
			if (s.seq.load(std::memory_order_relaxed) == s1 && name[0] != 0) {
				std::snprintf(out, out_size, "%s", name);
				return true;
			}
		}
	}
	const int64_t now = NowMs();
	for (const auto& slot: g_ctx.slots) {
		const uint64_t e = slot.end.load(std::memory_order_acquire);
		if (e == 0 || address >= e || address < slot.begin.load(std::memory_order_relaxed)) continue;
		if (now - slot.ms.load(std::memory_order_relaxed) > WindowMs) continue;
		std::snprintf(out, out_size, "%s", FixedNames[LCtx]);
		return true;
	}
	return false;
}

void Tags(uint64_t begin, uint64_t end, char* out, size_t out_size) noexcept {
	if (out == nullptr || out_size == 0) return;
	out[0] = 0;
	if (!Enabled() || end <= begin) return;
	const uint64_t pages = std::min<uint64_t>((end - begin) >> 12u, 512);
	uint64_t       bits[MaxLabels][8] {};
	const int64_t  now = NowMs();
	MarkRecent(g_dcb, bits[LDcb], begin, pages, now);
	MarkRecent(g_ctx, bits[LCtx], begin, pages, now);
	char     names[MaxLabels][17] {};
	uint32_t labels = 0;
	bool     known  = false;
	if (const int cur = g_current.load(std::memory_order_acquire); cur >= 0) {
		const auto&    s  = g_snap[cur];
		const uint32_t s1 = s.seq.load(std::memory_order_acquire);
		if ((s1 & 1u) == 0) {
			labels             = std::min<uint32_t>(s.labels.load(std::memory_order_relaxed), MaxLabels);
			const uint32_t cnt = std::min<uint32_t>(s.count.load(std::memory_order_relaxed), MaxRanges);
			for (uint32_t i = 0; i < labels; i++) {
				const uint64_t words[2] = {s.name[i][0].load(std::memory_order_relaxed),
				                           s.name[i][1].load(std::memory_order_relaxed)};
				std::memcpy(names[i], words, 16);
			}
			for (uint32_t i = 0; i < cnt; i++) {
				const uint64_t v = s.begin[i].load(std::memory_order_relaxed);
				const uint32_t l = static_cast<uint32_t>(v >> 56u);
				if (l < MaxLabels) Mark(bits[l], begin, pages, v & ((uint64_t {1} << 56u) - 1), s.end[i].load(std::memory_order_relaxed));
			}
			std::atomic_thread_fence(std::memory_order_acquire);
			known = s.seq.load(std::memory_order_relaxed) == s1;
		}
	}
	size_t used = 0;
	auto   put  = [&](const char* fmt, const char* name, uint32_t pct, bool unknown) {
        if (used >= out_size) return;
        const int n = unknown ? std::snprintf(out + used, out_size - used, "%s%s ?", used == 0 ? "" : "  ", name)
                              : std::snprintf(out + used, out_size - used, fmt, used == 0 ? "" : "  ", name, pct);
        if (n > 0) used = std::min(out_size - 1, used + static_cast<size_t>(n));
	};
	for (uint32_t l = 0; l < LFixed; l++)
		put("%s%s %u%%", FixedNames[l], Percent(bits[l], pages), l >= LCpuHeap && !known);
	if (!known) return;
	// The other allocators that cover >= 1 % of it, most first (4 at most).
	uint32_t order[MaxLabels];
	uint32_t pct[MaxLabels];
	uint32_t n = 0;
	for (uint32_t l = LFixed; l < labels; l++) {
		pct[l] = Percent(bits[l], pages);
		if (pct[l] >= 1) order[n++] = l;
	}
	std::sort(order, order + n, [&](uint32_t a, uint32_t b) { return pct[a] > pct[b]; });
	for (uint32_t i = 0; i < n && i < 4; i++) put("%s%s %u%%", names[order[i]], pct[order[i]], false);
}

} // namespace Loader::DemonsSoulsMemMap
