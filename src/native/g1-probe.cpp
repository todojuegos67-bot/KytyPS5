// (g1) The G1 probe and the STATE switch's reporting (g1-probe.h). Measurement code: nothing here changes what the game
// sees or what the GPU draws (the STATE path itself is native/state-cache.inc).
#include "native/g1-probe.h"

#include "native/native-render.h"
#include "native/pass-cut.h"
#include "native/pass-registry.h"

#include "loader/demonsSoulsMemMap.h"
#include "local/live-census.h"
#include "local/port2.h"

#include <xxhash.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace G1 {
namespace {

uint64_t Fnv(const uint8_t* p, uint32_t n) {
	uint64_t h = 0xcbf29ce484222325ull;
	for (uint32_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
	return h;
}

// ---- the eboot (01.007.000; Ghidra: G1Info.java, bytes from the entry to the function's end) ----
struct Checked {
	uint64_t    offset;
	uint32_t    size;
	uint64_t    fnv;
	const char* what;
};
constexpr Checked BinderCopies[] = {
    {0x1d182f0, 358, 0xa68698ac58338a27ull, "CX pairs block 0x1d182f0 (memcpy returns to 0x1d18434)"},
    {0x1d18460, 358, 0x85012972b96dc2e9ull, "SH pairs block 0x1d18460 (memcpy returns to 0x1d185a4)"},
    {0x1d185d0, 358, 0x4b99f06c3b55c63full, "UC pairs block 0x1d185d0 (memcpy returns to 0x1d18714)"},
};
constexpr uint64_t CopyReturn[3] = {0x1d18434, 0x1d185a4, 0x1d18714};

// Known CPU readers of GPU-written memory (Ghidra function bodies, G1Info.java).
struct Site {
	uint64_t    lo, hi;
	const char* name;
};
constexpr Site KnownSites[] = {
    {0xa1f560, 0xa1f5a7, "0xa1f560 (bits XPR)"},       {0xa1f290, 0xa1f2ea, "0xa1f290 (bits XPR)"},
    {0xad8bc0, 0xad8fd9, "0xad8bc0 (via 0xa1f290)"},   {0xad92f0, 0xad9641, "0xad92f0 (via 0xa1f290)"},
    {0xa1f020, 0xa1f081, "0xa1f020 (contadores cull)"}, {0x909940, 0x90994e, "0x909940 (accesor)"},
    {0x909970, 0x9099e6, "0x909970 (accesor copia)"},
};
constexpr uint32_t KnownCount = sizeof(KnownSites) / sizeof(KnownSites[0]);
constexpr uint32_t SiteOtherEboot = KnownCount, SiteOutside = KnownCount + 1, Sites = KnownCount + 2;

std::atomic<uint64_t> g_base {0}, g_size {0};

// ---- memory-map labels (a thread of its own paints a 64 KiB table; any thread reads it lock-free) ----
constexpr uint32_t MaxLabels   = 32; // 0: none
constexpr uint32_t GranuleBits = 16;
struct LabelTable {
	uint64_t             lo = 0, hi = 0;
	std::vector<uint8_t> gran;
};
std::atomic<LabelTable*> g_table {nullptr};
char                     g_names[MaxLabels][20] {};
std::atomic<uint32_t>    g_label_count {1};
std::atomic<uint64_t>    g_unknown[64] {};
std::atomic<uint32_t>    g_unknown_next {0};
uint32_t LabelOf(uint64_t a) {
	const LabelTable* t = g_table.load(std::memory_order_acquire);
	if (t == nullptr || a < t->lo || a >= t->hi) return 0;
	return t->gran[(a - t->lo) >> GranuleBits];
}
uint32_t LabelNoted(uint64_t a) {
	const uint32_t l = LabelOf(a);
	if (l == 0) {
		const uint32_t i = g_unknown_next.fetch_add(1, std::memory_order_relaxed);
		if ((i & 7u) == 0) g_unknown[(i >> 3u) & 63u].store(a, std::memory_order_relaxed);
	}
	return l;
}
const char* LabelName(uint32_t l) { return l == 0 ? "sin_etiqueta" : g_names[l]; }

void LabelThread() {
	using namespace Loader::DemonsSoulsMemMap;
	std::vector<LabelTable*> retired;
	std::vector<uint64_t>    pairs(2 * 16384);
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(2));
		for (auto& u: g_unknown) {
			const uint64_t a = u.exchange(0, std::memory_order_relaxed);
			if (a == 0) continue;
			char name[20] {};
			if (!LabelAt(a, name, sizeof(name)) || name[0] == 0) continue;
			const uint32_t n     = g_label_count.load(std::memory_order_relaxed);
			bool           known = false;
			for (uint32_t i = 1; i < n; ++i) known = known || std::strcmp(g_names[i], name) == 0;
			if (known || n >= MaxLabels) continue;
			std::snprintf(g_names[n], sizeof(g_names[n]), "%s", name);
			g_label_count.store(n + 1, std::memory_order_release);
		}
		struct Painted {
			uint64_t b, e;
			uint8_t  l;
		};
		std::vector<Painted> all;
		const uint32_t       n = g_label_count.load(std::memory_order_acquire);
		bool                 ok = true;
		for (uint32_t l = 1; l < n && ok; ++l) {
			const char*  name = g_names[l];
			const size_t got  = RangesOf(&name, 1, pairs.data(), pairs.size() / 2);
			if (got == SIZE_MAX) ok = false;
			for (size_t i = 0; ok && i < got; ++i)
				if (pairs[2 * i + 1] > pairs[2 * i]) all.push_back({pairs[2 * i], pairs[2 * i + 1], static_cast<uint8_t>(l)});
		}
		if (!ok || all.empty()) continue;
		uint64_t lo = UINT64_MAX, hi = 0;
		for (const auto& p: all) {
			lo = std::min(lo, p.b);
			hi = std::max(hi, p.e);
		}
		lo &= ~((uint64_t {1} << GranuleBits) - 1u);
		constexpr uint64_t MaxSpan = uint64_t {64} << 30u;
		if (hi - lo > MaxSpan) hi = lo + MaxSpan;
		auto* t = new LabelTable;
		t->lo   = lo;
		t->hi   = hi;
		t->gran.assign(static_cast<size_t>((hi - lo + (uint64_t {1} << GranuleBits) - 1u) >> GranuleBits), 0);
		std::sort(all.begin(), all.end(), [](const Painted& a, const Painted& b) { return a.e - a.b > b.e - b.b; });
		for (const auto& p: all) {
			if (p.b >= hi) continue;
			const uint64_t b = (p.b - lo) >> GranuleBits, e = (std::min(p.e, hi) - lo + (uint64_t {1} << GranuleBits) - 1u) >> GranuleBits;
			for (uint64_t g = b; g < e && g < t->gran.size(); ++g) t->gran[g] = p.l;
		}
		retired.push_back(g_table.exchange(t, std::memory_order_acq_rel));
		// (A reader holds a table for a few instructions: two refreshes later nobody does.)
		while (retired.size() > 2) {
			delete retired.front();
			retired.erase(retired.begin());
		}
	}
}

// ---- A, game side: the binder's copies ----
enum SrcClass : uint32_t { SrcEboot = 0, SrcStack = 1, SrcHeap = 2, SrcClasses = 3 };
constexpr const char* SrcNames[SrcClasses] = {"eboot", "pila", "monton"};
enum Repeat : uint32_t { RepNew = 0, RepSame = 1, RepChanged = 2, Repeats = 3 };
struct CopyCounters {
	std::atomic<uint64_t> copies[3] {}, pairs[3] {};
	std::atomic<uint64_t> by_src[3][SrcClasses] {}, by_src_pairs[3][SrcClasses] {};
	std::atomic<uint64_t> rep[3][Repeats] {}, rep_pairs[3][Repeats] {};
	std::atomic<uint64_t> rep_src[SrcClasses][Repeats] {};
	std::atomic<uint64_t> heap_label[MaxLabels] {};
	std::atomic<uint64_t> cycles {0};
};
CopyCounters g_copies;
struct SourceEntry {
	uint64_t src = 0, hash = 0, frame = 0;
	uint32_t n = 0, space = 0;
};
constexpr uint32_t        SourceBits = 15;
SourceEntry               g_sources[1u << SourceBits];
std::atomic_flag          g_source_locks[64] {};

// ---- B ----
constexpr uint32_t Origins = 11; // Raiz::ProtOriginCount
constexpr const char* OriginNames[Origins] = {"otro", "read-track", "read-flush", "read-release", "write-track", "write-release",
                                              "offload", "sync", "reapply", "restore", "deferred-release"};
constexpr uint32_t NoPassSlot = NativeRender::MaxPasses; // (no pass open / no map)
struct CohCounters {
	// GPU thread only (plain)
	uint64_t gpu_ns[Origins][MaxLabels] {}, gpu_pages[Origins][MaxLabels] {}, gpu_calls[Origins][MaxLabels] {};
	uint64_t pass_ns[NativeRender::MaxPasses + 1][Origins] {};
	uint64_t track_pages_pass[NativeRender::MaxPasses + 1] {}, track_pages_label[MaxLabels] {};
	uint64_t tq6_cycles[NativeRender::MaxPasses + 1] {}, tq6_calls[NativeRender::MaxPasses + 1] {};
	// any thread
	std::atomic<uint64_t> other_ns[Origins] {};
	std::atomic<uint64_t> reads[Sites][MaxLabels] {}, writes[Sites][MaxLabels] {};
};
CohCounters g_coh_c;
// Other eboot sites of read faults: offset -> count (open addressing, CAS insert).
struct SiteSlot {
	std::atomic<uint64_t> offset {0};
	std::atomic<uint64_t> reads {0};
	std::atomic<uint32_t> label {0};
};
constexpr uint32_t SiteSlots = 512;
SiteSlot           g_site_slots[SiteSlots];

uint32_t PassNow() {
	if (!NativeRender::On() || !NativeRender::Cut::OnGpuThread()) return NoPassSlot;
	const uint32_t p = NativeRender::Cut::Innermost();
	return p < NativeRender::MaxPasses ? p : NoPassSlot;
}

// ---- report state (GPU thread) ----
struct Window {
	bool     on = true; // STATE window
	double   seconds = 0, frames = 0;
	double   frame_ms = 0, p50 = 0, p99 = 0, low1 = 0, fps = 0, cpu = 0, ws_mib = 0, priv_mib = 0;
	double   block_ms = 0, sub_ms = 0;
};
std::vector<Window> g_windows;
std::vector<double> g_frame_ms;
int64_t             g_last_flip = 0, g_window_start = 0, g_last_summary = 0, g_start = 0;
uint64_t            g_window_tsc = 0;
uint64_t            g_window_frames = 0;
double              g_tsc_per_ns = 3.0;
uint64_t            g_last_sub_cycles = 0;
double              g_last_cpu_s = 0;
// cumulative (RESUMEN)
struct Totals {
	double frames = 0;
	double copies[3] {}, cpairs[3] {}, rep[3][Repeats] {}, rep_pairs[3][Repeats] {}, by_src[3][SrcClasses] {};
	double blocks[3] {}, pairs[3] {}, sync_ms[3] {}, apply_ms[3] {}, total_ms[3] {}, raw[3] {}, raw_diff[3] {}, ign[3] {},
	    call[3] {}, call_diff[3] {}, fake[3] {}, skipped[3] {}, repeat_blocks[3] {}, repeat_pairs[3] {}, probe_ms = 0;
	double gpu_ms[Origins][MaxLabels] {}, tq6_ms = 0, tq6_pass[NativeRender::MaxPasses + 1] {},
	    pass_ms[NativeRender::MaxPasses + 1][Origins] {};
	double reads[Sites][MaxLabels] {};
	double walk_ms[2][WalkPoints] {}, walk_steps[2] {};
	double set_calls = 0, set_full = 0, set_held = 0, set_checked = 0, set_refreshed = 0, set_ok = 0, rep_clean = 0, rep_dirty = 0;
	double st_discrepancias = 0, st_verified = 0;
};
Totals g_tot;

double ProcessCpuSeconds() {
#if defined(_WIN32)
	FILETIME c {}, e {}, k {}, u {};
	if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u) == 0) return 0.0;
	const auto f = [](const FILETIME& t) { return static_cast<double>((uint64_t {t.dwHighDateTime} << 32u) | t.dwLowDateTime) * 1e-7; };
	return f(k) + f(u);
#else
	rusage r {};
	getrusage(RUSAGE_SELF, &r);
	return static_cast<double>(r.ru_utime.tv_sec + r.ru_stime.tv_sec) + static_cast<double>(r.ru_utime.tv_usec + r.ru_stime.tv_usec) * 1e-6;
#endif
}
void ProcessMemory(double* ws_mib, double* priv_mib) {
	*ws_mib = *priv_mib = 0;
#if defined(_WIN32)
	PROCESS_MEMORY_COUNTERS_EX pmc {};
	pmc.cb = sizeof(pmc);
	if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc)) != 0) {
		*ws_mib   = static_cast<double>(pmc.WorkingSetSize) / 1048576.0;
		*priv_mib = static_cast<double>(pmc.PrivateUsage) / 1048576.0;
	}
#endif
}
double Percentile(std::vector<double> v, double p) {
	if (v.empty()) return 0.0;
	std::sort(v.begin(), v.end());
	const double at = p * static_cast<double>(v.size() - 1);
	const auto   i  = static_cast<size_t>(at);
	const double f  = at - static_cast<double>(i);
	return i + 1 < v.size() ? v[i] * (1.0 - f) + v[i + 1] * f : v[i];
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
Switches ReadSwitches() noexcept {
	const auto num = [](const char* name, uint32_t top) {
		const char* t = std::getenv(name);
		const auto  v = t == nullptr || t[0] == 0 ? 0u : static_cast<uint32_t>(std::strtoul(t, nullptr, 10));
		return v <= top ? v : 0u;
	};
	Switches s;
	s.probe = num("KYTY_G1_PROBE", 1);
	s.state = num("KYTY_G1_STATE", 2);
	s.fixed = num("KYTY_G1_FIJO", 1) == 1;
	if (std::getenv("KYTY_G1_COH") != nullptr)
		std::printf("[g1] KYTY_G1_COH no existe en esta build (la coherencia solo se mide: KYTY_G1_PROBE=1); ignorado\n");
	if (s.probe != 0 || s.state != 0) {
		std::printf("[g1] KYTY_G1_PROBE=%u KYTY_G1_STATE=%u%s: %s%s\n", s.probe, s.state, s.fixed ? " KYTY_G1_FIJO=1" : "",
		            s.probe != 0 ? "sonda A (estado indirecto) + B (coherencia de lo que escribe la GPU) + NP1 (recorrido del draw); " : "",
		            s.state == 1   ? (s.fixed ? "STATE fijo ON (delta de estado indirecto)" : "STATE alterna 5 s ON / 5 s OFF, linea [g1] ALTERNA")
		            : s.state == 2 ? "STATE verificacion: los dos caminos, registros comparados (DISCREPANCIA debe ser 0)"
		                           : "STATE apagado");
		std::fflush(stdout);
	}
	return s;
}

void Install(uint64_t base, uint64_t size, bool is_eboot) noexcept {
	if (!On() || !is_eboot) return;
	for (const auto& c: BinderCopies) {
		if (size < c.offset + c.size) {
			std::printf("[g1] DESACTIVADO: el eboot es demasiado corto para ser 01.007.000\n");
			return;
		}
		const uint64_t h = Fnv(reinterpret_cast<const uint8_t*>(base + c.offset), c.size);
#ifdef G1_SELFTEST
		if (false) {
#else
		if (h != c.fnv) {
#endif
			std::printf("[g1] DESACTIVADO: eboot+0x%llx (%s) hash 0x%016llx, 01.007.000 tiene 0x%016llx\n",
			            static_cast<unsigned long long>(c.offset), c.what, static_cast<unsigned long long>(h),
			            static_cast<unsigned long long>(c.fnv));
			std::fflush(stdout);
			return;
		}
	}
	g_base.store(base, std::memory_order_relaxed);
	g_size.store(size, std::memory_order_relaxed);
	if (ProbeMode() != 0) {
		g_copy_lo.store(base + CopyReturn[0], std::memory_order_relaxed);
		g_coh.store(true, std::memory_order_relaxed);
		std::thread(LabelThread).detach();
	}
	std::printf("[g1] eboot 01.007.000: copias de pares del binder comprobadas (3 funciones)%s\n",
	            ProbeMode() != 0 ? "; sonda activa" : "");
	std::fflush(stdout);
}

void OnBinderCopySlow(uint64_t ret, const void* dst, const void* src, size_t n) noexcept {
	const uint64_t base = g_base.load(std::memory_order_relaxed);
	uint32_t       space = 3;
	for (uint32_t s = 0; s < 3; ++s)
		if (ret == base + CopyReturn[s]) space = s;
	if (space == 3 || n == 0 || (n & 7u) != 0) return;
	(void)dst;
	const uint64_t t0    = __rdtsc();
	const auto     a     = reinterpret_cast<uint64_t>(src);
	const uint32_t pairs = static_cast<uint32_t>(n / 8u);
	char           here  = 0;
	const auto     sp    = reinterpret_cast<uint64_t>(&here);
	uint32_t       cls   = SrcHeap;
	if (a >= base && a < base + g_size.load(std::memory_order_relaxed)) cls = SrcEboot;
	else if (a + 0x40000u > sp && a < sp + 0x40000u) cls = SrcStack;
	const uint64_t hash  = XXH3_64bits(src, n);
	const uint64_t frame = g_frame.load(std::memory_order_relaxed);
	uint32_t       rep   = RepNew;
	{
		const uint64_t key  = (a * 0x9E3779B97F4A7C15ull) ^ space;
		const uint32_t slot = static_cast<uint32_t>(key >> (64u - SourceBits));
		auto&          lock = g_source_locks[slot & 63u];
		while (lock.test_and_set(std::memory_order_acquire)) _mm_pause();
		SourceEntry& e = g_sources[slot];
		if (e.src == a && e.space == space && e.frame + 1 >= frame) rep = e.hash == hash && e.n == pairs ? RepSame : RepChanged;
		e.src   = a;
		e.space = space;
		e.n     = pairs;
		e.hash  = hash;
		e.frame = frame;
		lock.clear(std::memory_order_release);
	}
	auto& c = g_copies;
	c.copies[space].fetch_add(1, std::memory_order_relaxed);
	c.pairs[space].fetch_add(pairs, std::memory_order_relaxed);
	c.by_src[space][cls].fetch_add(1, std::memory_order_relaxed);
	c.by_src_pairs[space][cls].fetch_add(pairs, std::memory_order_relaxed);
	c.rep[space][rep].fetch_add(1, std::memory_order_relaxed);
	c.rep_pairs[space][rep].fetch_add(pairs, std::memory_order_relaxed);
	c.rep_src[cls][rep].fetch_add(1, std::memory_order_relaxed);
	if (cls == SrcHeap) c.heap_label[LabelNoted(a)].fetch_add(1, std::memory_order_relaxed);
	c.cycles.fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
}

void NoteProtectSlow(bool gpu, uint32_t origin, uint64_t vaddr, uint64_t size, uint64_t ns) noexcept {
	if (origin >= Origins) origin = 0;
	if (!gpu) {
		g_coh_c.other_ns[origin].fetch_add(ns, std::memory_order_relaxed);
		return;
	}
	const uint32_t l = LabelNoted(vaddr);
	g_coh_c.gpu_ns[origin][l] += ns;
	g_coh_c.gpu_pages[origin][l] += (size + 0xfffu) >> 12u;
	g_coh_c.gpu_calls[origin][l] += 1;
	g_coh_c.pass_ns[PassNow()][origin] += ns;
}

void NoteReadTrackSlow(uint64_t begin, uint64_t end) noexcept {
	if (!t_gpu || end <= begin) return;
	const uint64_t pages = (end - begin + 0xfffu) >> 12u;
	g_coh_c.track_pages_pass[PassNow()] += pages;
	g_coh_c.track_pages_label[LabelNoted(begin)] += pages;
}

void NoteTq6Slow(uint64_t cycles) noexcept {
	if (!t_gpu) return;
	const uint32_t p = PassNow();
	g_coh_c.tq6_cycles[p] += cycles;
	g_coh_c.tq6_calls[p] += 1;
}

void NoteGpuFaultSlow(bool read, uint64_t rip, uint64_t rsp, uint64_t vaddr) noexcept {
	const uint64_t base = g_base.load(std::memory_order_relaxed), size = g_size.load(std::memory_order_relaxed);
	const auto     in_eboot = [&](uint64_t a) { return a >= base && a < base + size; };
	uint64_t       at       = rip;
	if (!in_eboot(at)) {
		// Host code (the libc memcpy HLE, another HLE): the guest caller.
		if (t_memcpy_ret != 0) at = t_memcpy_ret;
		else if (rsp != 0 && in_eboot(*reinterpret_cast<const uint64_t*>(rsp))) at = *reinterpret_cast<const uint64_t*>(rsp);
	}
	uint32_t site = SiteOutside;
	if (in_eboot(at)) {
		const uint64_t off = at - base;
		site               = SiteOtherEboot;
		for (uint32_t i = 0; i < KnownCount; ++i)
			if (off >= KnownSites[i].lo && off < KnownSites[i].hi) site = i;
		if (site == SiteOtherEboot && read) {
			const uint32_t h = static_cast<uint32_t>((off * 0x9E3779B97F4A7C15ull) >> 55u) & (SiteSlots - 1u);
			for (uint32_t probe = 0; probe < 16; ++probe) {
				auto&    s   = g_site_slots[(h + probe) & (SiteSlots - 1u)];
				uint64_t cur = s.offset.load(std::memory_order_relaxed);
				if (cur == 0 && s.offset.compare_exchange_strong(cur, off, std::memory_order_relaxed)) cur = off;
				if (cur != off) continue;
				s.reads.fetch_add(1, std::memory_order_relaxed);
				s.label.store(LabelOf(vaddr), std::memory_order_relaxed);
				break;
			}
		}
	}
	const uint32_t l = LabelNoted(vaddr);
	(read ? g_coh_c.reads : g_coh_c.writes)[site][l].fetch_add(1, std::memory_order_relaxed);
}

void WalkStepSlow() noexcept {
	auto&          w   = g_walk_state;
	const uint64_t now = __rdtsc();
	// (A walk that returned after a draw closes its step at the next walk's first step: such a step is dropped.)
	if (w.draw && (w.set & 1u) != 0 && static_cast<double>(now - w.mark[0]) > g_tsc_per_ns * 2e6) {
		++w.dropped;
	} else if (w.draw && (w.set & 1u) != 0) {
		const uint32_t kind = (w.set & (1u << WObserved)) != 0 ? 0u : 1u;
		++w.steps[kind];
		uint32_t prev = WTop;
		for (uint32_t p = 1; p < WalkPoints; ++p) {
			if ((w.set & (1u << p)) == 0) continue;
			w.cyc[kind][prev] += w.mark[p] - w.mark[prev];
			prev = p;
		}
		w.cyc[kind][prev] += now - w.mark[prev];
	}
	w.set     = 1u;
	w.draw    = false;
	w.mark[0] = now;
}

// ---------------------------------------------------------------------------------------------------------------------
void FlipSlow(int64_t flip_ns) noexcept {
	if (!t_gpu) {
		t_gpu = true;
		g_walk.store(ProbeMode() != 0, std::memory_order_relaxed);
	}
	g_frame.fetch_add(1, std::memory_order_relaxed);
	const uint64_t tsc = __rdtsc();
	if (g_last_flip != 0) g_frame_ms.push_back(static_cast<double>(flip_ns - g_last_flip) / 1e6);
	g_last_flip = flip_ns;
	++g_window_frames;
	if (g_window_start == 0) {
		g_window_start = g_start = g_last_summary = flip_ns;
		g_window_tsc                               = tsc;
		g_last_sub_cycles                          = LiveCensus::g_kind_cycles[LiveCensus::Submission];
		g_last_cpu_s                               = ProcessCpuSeconds();
		g_window_frames                            = 0;
		g_frame_ms.clear();
		return;
	}
	if (flip_ns - g_window_start < 5'000'000'000LL) return;

	// ---- one window ----
	const double secs   = static_cast<double>(flip_ns - g_window_start) / 1e9;
	const double frames = g_window_frames > 0 ? static_cast<double>(g_window_frames) : 1.0;
	if (tsc > g_window_tsc) g_tsc_per_ns = static_cast<double>(tsc - g_window_tsc) / static_cast<double>(flip_ns - g_window_start);
	const double ms_per_cycle = 1.0 / (g_tsc_per_ns * 1e6);
	const auto   pf           = [&](double cycles) { return cycles * ms_per_cycle / frames; }; // ms a frame
	Window       w;
	w.on       = StateActive();
	w.seconds  = secs;
	w.frames   = frames;
	w.fps      = frames / secs;
	w.frame_ms = secs * 1000.0 / frames;
	w.p50      = Percentile(g_frame_ms, 0.50);
	w.p99      = Percentile(g_frame_ms, 0.99);
	w.low1     = w.p99 > 0 ? 1000.0 / w.p99 : 0.0;
	const double cpu_s = ProcessCpuSeconds();
	w.cpu              = (cpu_s - g_last_cpu_s) / secs * 100.0;
	g_last_cpu_s       = cpu_s;
	ProcessMemory(&w.ws_mib, &w.priv_mib);
	const uint64_t sub = LiveCensus::g_kind_cycles[LiveCensus::Submission];
	w.sub_ms           = pf(static_cast<double>(sub - g_last_sub_cycles));
	g_last_sub_cycles  = sub;
	auto& b            = g_blocks;
	double blocks_cyc  = 0;
	for (int s = 0; s < 3; ++s) blocks_cyc += static_cast<double>(b.cyc_total[s]);
	w.block_ms = pf(blocks_cyc);
	g_tot.frames += frames;

	const char* p2 = Port2::Mode() == 0 ? "off" : Port2::Mode() == 2 ? "verif" : Port2::g_window_on.load(std::memory_order_relaxed) ? "ON" : "OFF";
	std::printf("[g1] %.1f s, %.0f frames | STATE %s | PORT2 %s | fps %.1f, 1%% low %.1f, frame p50 %.2f p99 %.2f ms | CPU %.0f%% "
	            "(nucleos) | RAM ws %.0f MiB privada %.0f MiB | submissions %.2f ms/frame (KYTY_RENDER_STATS) | bloques indirectos %.3f "
	            "ms/frame\n",
	            secs, frames, StateMode() == 0 ? "off" : StateMode() == 2 ? "verif" : w.on ? "ON" : "OFF", p2, w.fps, w.low1, w.p50,
	            w.p99, w.cpu, w.ws_mib, w.priv_mib, w.sub_ms, w.block_ms);

	if (ProbeMode() != 0) {
		// ---- A, game side ----
		auto&       c  = g_copies;
		const char* sp[3] = {"CX", "SH", "UC"};
		std::printf("[g1] A juego (memcpy del binder)/frame:");
		for (int s = 0; s < 3; ++s) {
			const double n  = static_cast<double>(c.copies[s].exchange(0, std::memory_order_relaxed));
			const double pr = static_cast<double>(c.pairs[s].exchange(0, std::memory_order_relaxed));
			double       r[Repeats], rp[Repeats], bs[SrcClasses], bsp[SrcClasses];
			for (uint32_t k = 0; k < Repeats; ++k) {
				r[k]  = static_cast<double>(c.rep[s][k].exchange(0, std::memory_order_relaxed));
				rp[k] = static_cast<double>(c.rep_pairs[s][k].exchange(0, std::memory_order_relaxed));
				g_tot.rep[s][k] += r[k];
				g_tot.rep_pairs[s][k] += rp[k];
			}
			for (uint32_t k = 0; k < SrcClasses; ++k) {
				bs[k]  = static_cast<double>(c.by_src[s][k].exchange(0, std::memory_order_relaxed));
				bsp[k] = static_cast<double>(c.by_src_pairs[s][k].exchange(0, std::memory_order_relaxed));
				g_tot.by_src[s][k] += bs[k];
			}
			g_tot.copies[s] += n;
			g_tot.cpairs[s] += pr;
			const double nn = n > 0 ? n : 1, pp = pr > 0 ? pr : 1;
			std::printf(" | %s copias %.0f pares %.0f; origen eboot %.0f%% pila %.0f%% monton %.0f%% (pares %.0f/%.0f/%.0f%%); "
			            "repetido_igual %.1f%% (pares %.1f%%) repetido_distinto %.1f%% nuevo %.1f%%",
			            sp[s], n / frames, pr / frames, 100 * bs[0] / nn, 100 * bs[1] / nn, 100 * bs[2] / nn, 100 * bsp[0] / pp,
			            100 * bsp[1] / pp, 100 * bsp[2] / pp, 100 * r[RepSame] / nn, 100 * rp[RepSame] / pp,
			            100 * r[RepChanged] / nn, 100 * r[RepNew] / nn);
		}
		std::printf(" | por origen repetido_igual/distinto/nuevo:");
		for (uint32_t k = 0; k < SrcClasses; ++k) {
			double r[Repeats];
			for (uint32_t q = 0; q < Repeats; ++q) r[q] = static_cast<double>(c.rep_src[k][q].exchange(0, std::memory_order_relaxed));
			const double t = r[0] + r[1] + r[2] > 0 ? r[0] + r[1] + r[2] : 1;
			std::printf(" %s %.0f/%.0f/%.0f%%", SrcNames[k], 100 * r[RepSame] / t, 100 * r[RepChanged] / t, 100 * r[RepNew] / t);
		}
		std::printf(" | monton por etiqueta:");
		for (uint32_t l = 0; l < g_label_count.load(std::memory_order_acquire); ++l) {
			const double v = static_cast<double>(c.heap_label[l].exchange(0, std::memory_order_relaxed));
			if (v > 0) std::printf(" %s=%.0f", LabelName(l), v / frames);
		}
		std::printf(" | coste sonda juego %.3f ms/frame (hilos del juego)\n",
		            pf(static_cast<double>(c.cycles.exchange(0, std::memory_order_relaxed))));

		// ---- A, GPU side ----
		std::printf("[g1] A GPU (bloques *_REG_INDIRECT del hilo GPU)/frame:");
		for (int s = 0; s < 3; ++s) {
			const double n = static_cast<double>(b.blocks[s]), pr = static_cast<double>(b.pairs[s]);
			const double nn = n > 0 ? n : 1, pp = pr > 0 ? pr : 1;
			const double raw = static_cast<double>(b.raw[s]), call = static_cast<double>(b.call[s]);
			std::printf(" | %s bloques %.0f pares %.0f; ns/bloque sync %.0f aplicar %.0f total %.0f (%.3f ms/frame); pares raw %.1f%% "
			            "ign %.1f%% call %.1f%% fake %.1f%% salto %.1f%%; DELTA raw distintos %.1f%% de raw, call distintos %.1f%% de "
			            "call (sombra); contenido ya aplicado %.1f%% bloques (%.1f%% pares), igual al anterior %.1f%%",
			            sp[s], n / frames, pr / frames, static_cast<double>(b.cyc_sync[s]) / g_tsc_per_ns / nn,
			            static_cast<double>(b.cyc_apply[s]) / g_tsc_per_ns / nn, static_cast<double>(b.cyc_total[s]) / g_tsc_per_ns / nn,
			            pf(static_cast<double>(b.cyc_total[s])), 100 * raw / pp, 100 * static_cast<double>(b.ign[s]) / pp,
			            100 * call / pp, 100 * static_cast<double>(b.fake[s]) / pp, 100 * static_cast<double>(b.skipped[s]) / pp,
			            raw > 0 ? 100 * static_cast<double>(b.raw_diff[s]) / raw : 0.0,
			            call > 0 ? 100 * static_cast<double>(b.call_diff[s]) / call : 0.0,
			            100 * static_cast<double>(b.repeat_blocks[s]) / nn, 100 * static_cast<double>(b.repeat_pairs[s]) / pp,
			            100 * static_cast<double>(b.same_as_previous[s]) / nn);
			g_tot.blocks[s] += n;
			g_tot.pairs[s] += pr;
			g_tot.sync_ms[s] += static_cast<double>(b.cyc_sync[s]) * ms_per_cycle;
			g_tot.apply_ms[s] += static_cast<double>(b.cyc_apply[s]) * ms_per_cycle;
			g_tot.total_ms[s] += static_cast<double>(b.cyc_total[s]) * ms_per_cycle;
			g_tot.raw[s] += raw;
			g_tot.raw_diff[s] += static_cast<double>(b.raw_diff[s]);
			g_tot.ign[s] += static_cast<double>(b.ign[s]);
			g_tot.call[s] += call;
			g_tot.call_diff[s] += static_cast<double>(b.call_diff[s]);
			g_tot.fake[s] += static_cast<double>(b.fake[s]);
			g_tot.skipped[s] += static_cast<double>(b.skipped[s]);
			g_tot.repeat_blocks[s] += static_cast<double>(b.repeat_blocks[s]);
			g_tot.repeat_pairs[s] += static_cast<double>(b.repeat_pairs[s]);
		}
		g_tot.probe_ms += static_cast<double>(b.probe_cycles) * ms_per_cycle;
		std::printf(" | otros hilos %llu bloques | coste sonda %.3f ms/frame (dentro de 'aplicar' no: restado)\n",
		            static_cast<unsigned long long>(g_other_blocks.exchange(0, std::memory_order_relaxed)), pf(static_cast<double>(b.probe_cycles)));

		// ---- B ----
		auto&          h      = g_coh_c;
		const uint32_t labels = g_label_count.load(std::memory_order_acquire);
		std::printf("[g1] B hilo GPU por etiqueta (ms/frame, paginas/frame):");
		for (uint32_t o = 0; o < Origins; ++o) {
			double sum = 0;
			for (uint32_t l = 0; l < MaxLabels; ++l) sum += static_cast<double>(h.gpu_ns[o][l]);
			if (sum == 0) continue;
			std::printf(" | %s %.3f ms:", OriginNames[o], sum / 1e6 / frames);
			for (uint32_t l = 0; l < labels; ++l)
				if (h.gpu_ns[o][l] != 0)
					std::printf(" %s=%.3f/%.0fp", LabelName(l), static_cast<double>(h.gpu_ns[o][l]) / 1e6 / frames,
					            static_cast<double>(h.gpu_pages[o][l]) / frames);
		}
		std::printf(" | otros hilos:");
		for (uint32_t o = 0; o < Origins; ++o) {
			const double v = static_cast<double>(h.other_ns[o].exchange(0, std::memory_order_relaxed));
			if (v > 0) std::printf(" %s=%.3f", OriginNames[o], v / 1e6 / frames);
		}
		std::printf(" | paginas marcadas no-access (read-track) por etiqueta:");
		for (uint32_t l = 0; l < labels; ++l)
			if (h.track_pages_label[l] != 0) std::printf(" %s=%.0f", LabelName(l), static_cast<double>(h.track_pages_label[l]) / frames);
		std::printf("\n");
		// by pass
		double tq6_total = 0;
		for (uint32_t p = 0; p <= NativeRender::MaxPasses; ++p) tq6_total += static_cast<double>(h.tq6_cycles[p]);
		std::printf("[g1] B por pasada (KYTY_NATIVE_RENDER=map; ms/frame): tq6 total %.3f%s |", pf(tq6_total),
		            tq6_total == 0 ? " (sin KYTY_RENDER_STATS no se mide)" : "");
		std::vector<uint32_t> ids;
		for (uint32_t p = 0; p <= NativeRender::MaxPasses; ++p) {
			double v = static_cast<double>(h.tq6_cycles[p]) * ms_per_cycle;
			for (uint32_t o = 0; o < Origins; ++o) v += static_cast<double>(h.pass_ns[p][o]) / 1e6;
			if (v > 0 || h.track_pages_pass[p] != 0) ids.push_back(p);
		}
		const auto pass_cost = [&](uint32_t p) {
			double v = static_cast<double>(h.tq6_cycles[p]) * ms_per_cycle;
			for (uint32_t o = 0; o < Origins; ++o) v += static_cast<double>(h.pass_ns[p][o]) / 1e6;
			return v;
		};
		std::sort(ids.begin(), ids.end(), [&](uint32_t a2, uint32_t b2) { return pass_cost(a2) > pass_cost(b2); });
		for (size_t i = 0; i < ids.size() && i < 12; ++i) {
			const uint32_t p = ids[i];
			std::printf(" \"%s\" tq6=%.3f (%.0f) read-flush=%.3f write-release=%.3f read-track=%.3f marcadas=%.0fp;",
			            p == NoPassSlot ? "(ninguna)" : NativeRender::PassRegistry::Name(p), pf(static_cast<double>(h.tq6_cycles[p])),
			            static_cast<double>(h.tq6_calls[p]) / frames, static_cast<double>(h.pass_ns[p][2]) / 1e6 / frames,
			            static_cast<double>(h.pass_ns[p][5]) / 1e6 / frames, static_cast<double>(h.pass_ns[p][1]) / 1e6 / frames,
			            static_cast<double>(h.track_pages_pass[p]) / frames);
		}
		std::printf("\n");
		// reads of GPU-written pages by site
		std::printf("[g1] B lecturas de CPU de paginas escritas por la GPU (fallos de lectura atendidos)/frame por sitio:");
		for (uint32_t s = 0; s < Sites; ++s) {
			double sum = 0, wsum = 0;
			for (uint32_t l = 0; l < MaxLabels; ++l) {
				const double r = static_cast<double>(h.reads[s][l].exchange(0, std::memory_order_relaxed));
				const double wr = static_cast<double>(h.writes[s][l].exchange(0, std::memory_order_relaxed));
				g_tot.reads[s][l] += r;
				sum += r;
				wsum += wr;
				(void)wr;
			}
			const char* name = s < KnownCount ? KnownSites[s].name : s == SiteOtherEboot ? "eboot otros" : "fuera del eboot";
			std::printf(" %s=%.2f (escrituras %.2f);", name, sum / frames, wsum / frames);
		}
		std::printf(" | top eboot otros:");
		{
			std::vector<std::pair<uint64_t, uint64_t>> top;
			for (auto& s: g_site_slots) {
				const uint64_t o = s.offset.load(std::memory_order_relaxed), r = s.reads.load(std::memory_order_relaxed);
				if (o != 0 && r != 0) top.emplace_back(r, o);
			}
			std::sort(top.rbegin(), top.rend());
			for (size_t i = 0; i < top.size() && i < 10; ++i)
				std::printf(" eboot+0x%llx=%llu", static_cast<unsigned long long>(top[i].second), static_cast<unsigned long long>(top[i].first));
			std::printf(" (acumulado)");
		}
		std::printf("\n");
		for (uint32_t o = 0; o < Origins; ++o)
			for (uint32_t l = 0; l < MaxLabels; ++l) g_tot.gpu_ms[o][l] += static_cast<double>(h.gpu_ns[o][l]) / 1e6;
		for (uint32_t p = 0; p <= NativeRender::MaxPasses; ++p) {
			g_tot.tq6_pass[p] += static_cast<double>(h.tq6_cycles[p]) * ms_per_cycle;
			for (uint32_t o = 0; o < Origins; ++o) g_tot.pass_ms[p][o] += static_cast<double>(h.pass_ns[p][o]) / 1e6;
		}
		g_tot.tq6_ms += tq6_total * ms_per_cycle;
		std::memset(h.gpu_ns, 0, sizeof(h.gpu_ns));
		std::memset(h.gpu_pages, 0, sizeof(h.gpu_pages));
		std::memset(h.gpu_calls, 0, sizeof(h.gpu_calls));
		std::memset(h.pass_ns, 0, sizeof(h.pass_ns));
		std::memset(h.track_pages_pass, 0, sizeof(h.track_pages_pass));
		std::memset(h.track_pages_label, 0, sizeof(h.track_pages_label));
		std::memset(h.tq6_cycles, 0, sizeof(h.tq6_cycles));
		std::memset(h.tq6_calls, 0, sizeof(h.tq6_calls));

		// ---- NP1 ----
		auto&             wk    = g_walk_state;
		static const char* WalkNames[WalkPoints] = {"preambulo", "decodificar", "comprobaciones_del_walk", "entrada(render1/prefetch)",
		                                            "observador", "try", "despues"};
		std::printf("[g1] NP1 recorrido de paquetes de draw (hilo GPU): con try %.0f pasos/frame, sin try %.0f (descartados >2 ms %.1f) |",
		            static_cast<double>(wk.steps[0]) / frames, static_cast<double>(wk.steps[1]) / frames,
		            static_cast<double>(wk.dropped) / frames);
		wk.dropped = 0;
		for (int k = 0; k < 2; ++k) {
			const double st = wk.steps[k] > 0 ? static_cast<double>(wk.steps[k]) : 1;
			std::printf(" %s us/paso:", k == 0 ? "con try" : "sin try");
			double sum = 0;
			for (uint32_t p = 0; p < WalkPoints; ++p) {
				const double us = static_cast<double>(wk.cyc[k][p]) / g_tsc_per_ns / 1000.0 / st;
				sum += us;
				if (wk.cyc[k][p] != 0) std::printf(" %s=%.3f", WalkNames[p], us);
				g_tot.walk_ms[k][p] += static_cast<double>(wk.cyc[k][p]) * ms_per_cycle;
			}
			std::printf(" (suma %.3f, %.3f ms/frame) |", sum, sum * st / 1000.0 / frames);
			g_tot.walk_steps[k] += static_cast<double>(wk.steps[k]);
		}
		auto&        sc  = g_sets;
		const double sc_calls = static_cast<double>(sc.calls.exchange(0)), sc_full = static_cast<double>(sc.full.exchange(0)),
		             sc_held = static_cast<double>(sc.held.exchange(0)), sc_checked = static_cast<double>(sc.checked.exchange(0)),
		             sc_ref = static_cast<double>(sc.refreshed.exchange(0)), sc_ok = static_cast<double>(sc.ok.exchange(0)),
		             rc = static_cast<double>(sc.reprove_clean.exchange(0)), rd = static_cast<double>(sc.reprove_dirty.exchange(0));
		std::printf(" sets probados (fase 7) %.0f/frame (completos %.0f), usos sostenidos %.0f comprobados %.0f refrescados %.1f, sets OK "
		            "%.0f fallidos %.0f | rangos reprobados (fase 6) limpios %.0f sucios %.0f/frame (resueltos de nuevo, fase 5: 'resolved' de "
		            "[render])\n",
		            sc_calls / frames, sc_full / frames, sc_held / frames, sc_checked / frames, sc_ref / frames, sc_ok / frames,
		            (sc_calls - sc_ok) / frames, rc / frames, rd / frames);
		g_tot.set_calls += sc_calls;
		g_tot.set_full += sc_full;
		g_tot.set_held += sc_held;
		g_tot.set_checked += sc_checked;
		g_tot.set_refreshed += sc_ref;
		g_tot.set_ok += sc_ok;
		g_tot.rep_clean += rc;
		g_tot.rep_dirty += rd;
		std::memset(wk.cyc, 0, sizeof(wk.cyc));
		wk.steps[0] = wk.steps[1] = 0;
	}

	// ---- STATE ----
	if (StateMode() != 0) {
		const char* sp[3] = {"CX", "SH", "UC"};
		std::printf("[g1] STATE %s:", StateMode() == 2 ? "verif" : w.on ? "ON" : "OFF");
		for (int s = 0; s < 3; ++s) {
			const double hit = static_cast<double>(b.st_hit[s]), miss = static_cast<double>(b.st_miss[s]);
			std::printf(" | %s aciertos %.0f fallos %.0f /frame (%.1f%%), stores hechos %.0f evitados %.0f, llamadas %.0f evitadas %.0f",
			            sp[s], hit / frames, miss / frames, hit + miss > 0 ? 100 * hit / (hit + miss) : 0.0,
			            static_cast<double>(b.st_stores[s]) / frames, static_cast<double>(b.st_kept[s]) / frames,
			            static_cast<double>(b.st_calls[s]) / frames, static_cast<double>(b.st_not_called[s]) / frames);
		}
		g_tot.st_verified += static_cast<double>(b.st_verified);
		g_tot.st_discrepancias += static_cast<double>(b.st_discrepancias);
		std::printf(" | verificados %llu DISCREPANCIA %llu (acumulado %.0f)\n", static_cast<unsigned long long>(b.st_verified),
		            static_cast<unsigned long long>(b.st_discrepancias), g_tot.st_discrepancias);
	}
	g_blocks = BlockCounters {};

	// ---- alternation (STATE=1) ----
	g_windows.push_back(w);
	if (StateMode() == 1 && !Get().fixed) {
		// Pairs: each OFF window with the ON window right after it.
		std::vector<double> d_block, d_sub, d_frame, d_fps, d_low, d_cpu;
		for (size_t i = 0; i + 1 < g_windows.size(); ++i) {
			const Window& off = g_windows[i];
			const Window& on  = g_windows[i + 1];
			if (off.on || !on.on) continue;
			d_block.push_back(off.block_ms - on.block_ms);
			d_sub.push_back(off.sub_ms - on.sub_ms);
			d_frame.push_back(off.frame_ms - on.frame_ms);
			d_fps.push_back(on.fps - off.fps);
			d_low.push_back(on.low1 - off.low1);
			d_cpu.push_back(off.cpu - on.cpu);
		}
		if (!d_block.empty()) {
			const auto mean = [](const std::vector<double>& v) {
				double s = 0;
				for (double x: v) s += x;
				return s / static_cast<double>(v.size());
			};
			std::printf("[g1] ALTERNA %zu pares OFF->ON | AHORRO MEDIDO (OFF-ON, media / p50): bloques indirectos %.3f / %.3f ms/frame, "
			            "submissions %.3f / %.3f ms/frame, frame %.3f / %.3f ms | fps +%.2f / +%.2f, 1%% low +%.2f / +%.2f, CPU -%.1f / "
			            "-%.1f puntos\n",
			            d_block.size(), mean(d_block), Percentile(d_block, 0.5), mean(d_sub), Percentile(d_sub, 0.5), mean(d_frame),
			            Percentile(d_frame, 0.5), mean(d_fps), Percentile(d_fps, 0.5), mean(d_low), Percentile(d_low, 0.5),
			            mean(d_cpu), Percentile(d_cpu, 0.5));
		}
		const bool next = !g_window_on.load(std::memory_order_relaxed);
		g_window_on.store(next, std::memory_order_relaxed);
		std::printf("[g1] ventana -> STATE %s (las lineas siguientes, [vram] [ram] [fps+] incluidas, son de esta ventana)\n", next ? "ON" : "OFF");
	}
	if (g_windows.size() > 4096) g_windows.erase(g_windows.begin(), g_windows.begin() + 2048);

	// ---- RESUMEN every 60 s ----
	if (flip_ns - g_last_summary >= 60'000'000'000LL && ProbeMode() != 0) {
		g_last_summary  = flip_ns;
		const double fr = g_tot.frames > 0 ? g_tot.frames : 1;
		double       copies = 0, same = 0, pairs = 0, same_pairs = 0;
		for (int s = 0; s < 3; ++s) {
			copies += g_tot.copies[s];
			pairs += g_tot.cpairs[s];
			same += g_tot.rep[s][RepSame];
			same_pairs += g_tot.rep_pairs[s][RepSame];
		}
		double a_ms = 0, a_raw = 0, a_raw_diff = 0, a_call = 0, a_call_diff = 0, a_pairs = 0, a_ign = 0, a_fake = 0;
		for (int s = 0; s < 3; ++s) {
			a_ms += g_tot.total_ms[s];
			a_raw += g_tot.raw[s];
			a_raw_diff += g_tot.raw_diff[s];
			a_call += g_tot.call[s];
			a_call_diff += g_tot.call_diff[s];
			a_pairs += g_tot.pairs[s];
			a_ign += g_tot.ign[s];
			a_fake += g_tot.fake[s];
		}
		const double rule_a = copies > 0 ? 100 * same / copies : 0;
		std::printf("[g1] RESUMEN %.0f s, %.0f frames | A: copias repetidas con el mismo contenido %.1f%% (pares %.1f%%) -> REGLA A (>=80%%) %s; "
		            "bloques %.3f ms/frame (sync %.3f aplicar %.3f); pares/frame %.0f: raw %.1f%% (distintos %.1f%%) ign+fake %.1f%% call "
		            "%.1f%% (distintos %.1f%%); TECHO STATE (raw iguales + ign + fake sin llamada, a 4 ns/par) ~%.3f ms/frame [H]\n",
		            static_cast<double>(flip_ns - g_start) / 1e9, g_tot.frames, rule_a, pairs > 0 ? 100 * same_pairs / pairs : 0.0,
		            rule_a >= 80 ? "CUMPLE" : "NO CUMPLE", a_ms / fr, (g_tot.sync_ms[0] + g_tot.sync_ms[1] + g_tot.sync_ms[2]) / fr,
		            (g_tot.apply_ms[0] + g_tot.apply_ms[1] + g_tot.apply_ms[2]) / fr, a_pairs / fr, a_pairs > 0 ? 100 * a_raw / a_pairs : 0.0,
		            a_raw > 0 ? 100 * a_raw_diff / a_raw : 0.0, a_pairs > 0 ? 100 * (a_ign + a_fake) / a_pairs : 0.0,
		            a_pairs > 0 ? 100 * a_call / a_pairs : 0.0, a_call > 0 ? 100 * a_call_diff / a_call : 0.0,
		            ((a_raw - a_raw_diff) + a_ign + a_fake) * 4e-6 / fr);
		// B: labels whose every read comes from known readers (or none at all)
		const uint32_t labels  = g_label_count.load(std::memory_order_acquire);
		double         known_ms = 0, unknown_ms = 0;
		std::printf("[g1] RESUMEN B por etiqueta (hilo GPU read-flush+write-release+read-track ms/frame; lecturas conocidas/otras por frame):");
		for (uint32_t l = 0; l < labels; ++l) {
			const double ms = (g_tot.gpu_ms[1][l] + g_tot.gpu_ms[2][l] + g_tot.gpu_ms[5][l]) / fr;
			double       kn = 0, ot = 0;
			for (uint32_t s = 0; s < Sites; ++s) (s < KnownCount ? kn : ot) += g_tot.reads[s][l];
			if (ms == 0 && kn + ot == 0) continue;
			(ot == 0 ? known_ms : unknown_ms) += ms;
			std::printf(" %s=%.3f (%.2f/%.2f)%s;", LabelName(l), ms, kn / fr, ot / fr, ot == 0 ? " lectores_conocidos" : "");
		}
		std::printf("\n[g1] RESUMEN B: TECHO B (etiquetas con solo lectores conocidos) %.3f ms/frame, resto %.3f; tq6 %.3f ms/frame (por pasada, sin "
		            "etiqueta: cota aparte) -> REGLA B (>=1,5) %s (%s)\n",
		            known_ms, unknown_ms, g_tot.tq6_ms / fr, known_ms >= 1.5 ? "CUMPLE" : "NO CUMPLE",
		            known_ms + g_tot.tq6_ms / fr >= 1.5 ? "con tq6 entero llegaria" : "ni con tq6 entero llega");
		// NP1
		double outside = 0, steps = g_tot.walk_steps[0] + g_tot.walk_steps[1];
		for (int k = 0; k < 2; ++k)
			for (uint32_t p = 0; p < WalkPoints; ++p)
				if (!(k == 0 && p == WObserved)) outside += g_tot.walk_ms[k][p];
		std::printf("[g1] RESUMEN NP1: pasos de draw %.0f/frame, fuera del try %.3f ms/frame (%.3f us/paso); sets probados %.0f/frame: usos "
		            "sostenidos %.0f%%, comprobados %.0f%%, refrescados %.2f%%, sets fallidos %.2f%%; reprobados limpios %.0f%% de %.0f/frame\n",
		            steps / fr, outside / fr, steps > 0 ? outside * 1000 / steps : 0.0, g_tot.set_calls / fr,
		            g_tot.set_held + g_tot.set_checked > 0 ? 100 * g_tot.set_held / (g_tot.set_held + g_tot.set_checked) : 0.0,
		            g_tot.set_held + g_tot.set_checked > 0 ? 100 * g_tot.set_checked / (g_tot.set_held + g_tot.set_checked) : 0.0,
		            g_tot.set_held + g_tot.set_checked > 0 ? 100 * g_tot.set_refreshed / (g_tot.set_held + g_tot.set_checked) : 0.0,
		            g_tot.set_calls > 0 ? 100 * (g_tot.set_calls - g_tot.set_ok) / g_tot.set_calls : 0.0,
		            g_tot.rep_clean + g_tot.rep_dirty > 0 ? 100 * g_tot.rep_clean / (g_tot.rep_clean + g_tot.rep_dirty) : 0.0,
		            (g_tot.rep_clean + g_tot.rep_dirty) / fr);
	}
	std::fflush(stdout);
	g_window_start  = flip_ns;
	g_window_tsc    = tsc;
	g_window_frames = 0;
	g_frame_ms.clear();
}

} // namespace G1
