#pragma once
// (PROT-MENOS2) KYTY_PROT_MENOS (default: on with native memory, off without it; =0 off, =1 on): pages the game writes again and again stop being
// write-protected ("hot" pages, RegionManager).
//
// Why (log 20261124-054756-507: 73,600 protect_calls/s on average, [fps+]; earlier logs: cpuheap 39-80 % of them):
// the general heap (cpuheap) shares pages between data the GPU reads and engine objects the CPU rewrites every frame.
// Ghidra (eboot 01.007.000): the top writer of cpuheap pages GPU work reads ([anuncio] "VETADA antes", 106 M refusals
// in the logs) is eboot+0xab4939 in FUN_00ab46f0, a per-frame loop over the draw objects of a list (param_3+0x28..0x30,
// each object's sub-list at +0xa0..+0xa8) that stores a distance fade/LOD factor at object+0x178 (and +0x174/+0x17c)
// for every object, every frame. Those fields are CPU bookkeeping, but they live on the same 4 KiB pages as bytes the
// GPU reads, so every frame: write fault (unprotect) -> upload -> reprotect, for each such page.
//
// What changes: a page that takes a second CPU write fault inside a 500 ms window becomes hot. A hot page is never
// write-protected again; instead it is made CPU-dirty again before EVERY submission (MemoryTracker::HotRedirty, same
// point as KYTY_VOLATILE), so whatever GPU work uses it uploads the current bytes - no stale data (the window is the
// one KYTY_VOLATILE has: a write landing during a submission's own work is taken at the next submission). Every 2 s a
// region's hot pages go back to the normal watch (dirty first, so nothing is lost); pages still written every frame
// become hot again after two faults. At most kMaxHotPages pages are hot (the rest stay as today).
//
// (PROT-MENOS2) Log 20261124-063947-246: making EVERY clean hot page dirty before every submission re-uploaded ~14,800
// pages ~7 times a frame (3.5 M pages/s, upload_bytes ~480 MB/frame, frame 13 -> 27.6 ms). Now, with native memory
// (KYTY_RAIZ=1: guest memory is MEM_WRITE_WATCH), HotStep reads the write watch (GetWriteWatch + reset, one call per run
// of hot pages) and makes dirty only the clean hot pages written since the last read: what a write fault did before,
// without the faults. Clean hot pages not written stay clean (no upload) and are looked at again next submission.
// Without native memory (or KYTY_PROT_MENOS_WW=0) the PROT-MENOS behaviour (all of them dirty).
// A demoted page still clean is write-protected again (normal watch) and the write watch read once more after that,
// so a write landing in between makes it dirty as a fault would.
#include "local/frame-nativo.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// (PROT-MENOS2) kernel/memory.h (declared here: regionManager.h includes this header).
namespace Libs::LibKernel::Memory {
bool     NativeMemoryEnabled();
uint64_t NativeTakeWritten(uint64_t vaddr, uint64_t size, uint64_t* pages, uint64_t max, bool* failed);
uint64_t NativePeekWritten(uint64_t vaddr, uint64_t size, uint64_t* pages, uint64_t max, bool* failed); // (PROT-MENOS4)
} // namespace Libs::LibKernel::Memory

namespace ProtMenos {

constexpr uint64_t kWindowMs    = 500;
constexpr uint64_t kDemoteMs    = 2000;
constexpr int64_t  kMaxHotPages = 16384; // 64 MiB re-uploaded at most, when every hot page is used
constexpr size_t   kMaxRegions  = 1u << 16u;

[[nodiscard]] inline bool On() noexcept {
	static const bool on = [] {
		// (PROT-MENOS2) Encendido por defecto con memoria nativa (KYTY_NATIVE_MEMORY: MEM_WRITE_WATCH, solo se resube lo
		// escrito). Sin ella, apagado (log 063947: 480 MB/frame). KYTY_PROT_MENOS=0 lo apaga, =1 lo fuerza.
		const char* v = std::getenv("KYTY_PROT_MENOS");
		if (v != nullptr && v[0] == '0') return false;
		if (v != nullptr && v[0] == '1') return true;
		return false; // (11-24: apagado por defecto; con el va lento en Boletaria)
	}();
	return on;
}

// (PROT-MENOS2) Only the written hot pages made dirty (write watch). Needs native memory (MEM_WRITE_WATCH).
[[nodiscard]] inline bool WatchOn() noexcept {
	static const bool on = [] {
		// (PROT-MENOS3, 11-24) Antes devolvia false siempre: con KYTY_PROT_MENOS=1 corria el modo viejo (todas las
		// paginas calientes sucias antes de CADA envio). Log 20261124-082011-604 (intro, mismo tramo que 081541/080436):
		// [fps] cpu 14.5 ms/frame frente a 1.3-1.4 sin PROT-MENOS, frame 21-27 ms frente a 16.7; SLOW Frame queue_run
		// ~24 ms con 4-14 comandos y draw_us <100 us (tiempo en HotStep/BDA, no en dibujar). Ahora sigue a On():
		// solo se resube lo que la vigilancia de escritura dice escrito (ab46f0 escribe +0x174..+0x193 una vez por frame).
		// Necesita memoria nativa (MEM_WRITE_WATCH). KYTY_PROT_MENOS_WW=0 vuelve al modo viejo.
		const char* v = std::getenv("KYTY_PROT_MENOS_WW");
		if (v != nullptr && v[0] == '0') return false;
		return On() && ::Libs::LibKernel::Memory::NativeMemoryEnabled();
	}();
	return on;
}

[[nodiscard]] inline uint64_t NowMs() noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline std::atomic<int64_t>  g_hot_pages {0};      // now
inline std::atomic<uint64_t> g_promoted {0};       // pages made hot (total)
inline std::atomic<uint64_t> g_refused {0};        // pages refused by the cap (total)
inline std::atomic<uint64_t> g_redirty_pages {0};  // pages made CPU-dirty again before submissions (total)
inline std::atomic<uint64_t> g_demoted {0};        // pages given back to the normal watch (total)
// (PROT-MENOS2) Totals: GetWriteWatch reads (and those refused: the run taken as written), clean hot pages left clean
// (not written: no re-upload), pages made dirty by the read after a demotion, microseconds in HotRedirty.
inline std::atomic<uint64_t> g_ww_calls {0};
inline std::atomic<uint64_t> g_ww_failed {0};
inline std::atomic<uint64_t> g_ww_unchanged {0};
inline std::atomic<uint64_t> g_ww_late {0};
inline std::atomic<uint64_t> g_step_us {0};

// (PROT-MENOS4) For the [prot-menos] line (Report, every 5 s, always with KYTY_PROT_MENOS=1): game frames (flips),
// HotRedirty calls (one per submission), its time and HotStep's (under the lock), span reads without reset, CPU write
// faults over tracked pages, hot pages kept at a demotion because they were written in the last window.
inline std::atomic<uint64_t> g_frames {0};
inline std::atomic<uint64_t> g_hot_calls {0};
inline std::atomic<uint64_t> g_redirty_us {0};
inline std::atomic<uint64_t> g_peek_calls {0};
inline std::atomic<uint64_t> g_cpu_writes {0};
inline std::atomic<uint64_t> g_kept {0};
inline std::atomic<size_t>   g_hot_regions {0};

inline void NoteFlip() noexcept { g_frames.fetch_add(1, std::memory_order_relaxed); }

// (PROT-MENOS5) Log 20261124-105007-224: 298.8 submissions/frame x ~51 regions = 15,335 GetWriteWatch/frame, HotRedirty
// 29.9 ms/frame (frame 26-30 ms against ~14 ms without PROT-MENOS). Ghidra (eboot 01.007.000): the hot fields
// (+0x170..+0x19f of each draw object, vmovss at 0xa55a8f..0xa55add in FUN_00a558e0) are written in ONE burst per frame:
// FUN_00a56270 walks the frame's update list once (+0x229160..+0x229168) and calls FUN_00a558e0 per object; it is called
// once, from the per-frame update stage FUN_00a227d0, before that stage queues the frame's jobs and builds
// (0xa10c80, 0xac5fc0, 0xa345f0, 0xa41340, 0xab90f0). So the write watch is read (HotStep over every region) at the
// first submission after each flip and then at most once every GAP us (KYTY_PROT_MENOS_GAP_US, default 2000; 0 = every
// submission, the PROT-MENOS4 behaviour). No write is lost: a clean hot page stays clean and its watch bit set until a
// scan takes it; a skipped scan only delays it <= GAP us, the same kind of window KYTY_VOLATILE has.
inline std::atomic<uint64_t> g_scans {0};

// (PROT-MENOS6) Lazy check at use. ScanDue (PROT-MENOS5) left the submissions between two scans reading hot pages the
// game wrote after the last scan (log 20261124-110544-630: 1.1 scans for ~300 submissions per frame, then GPU device
// lost and broken graphics). Every submission now moves g_submit_gen (HotRedirty, one fetch_add); the GPU thread's
// tracker queries (CpuModificationEpoch, IsRegionCpuModified, IsRegionFullyCpuModified, ForEachUploadRange) run
// RegionManager::HotLazy on each region they touch: the region's write watch is read once per submission, and only
// when that submission's work asks about it and it has clean hot pages. No write between two submissions is missed.
inline std::atomic<uint64_t> g_submit_gen {1};
inline std::atomic<uint64_t> g_lazy_calls {0}; // lazy HotSteps (a region read for a submission)
inline std::atomic<uint64_t> g_lazy_ns {0};
inline thread_local bool     t_gpu_thread = false; // (set by HotRedirty: lazy checks only on the GPU thread)
[[nodiscard]] inline bool LazyOn() noexcept {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_PROT_MENOS_LAZY");
		if (v != nullptr && v[0] == '0') return false;
		return On();
	}();
	return on;
} // full scans (HotStep over the regions), for [prot-menos]

[[nodiscard]] inline uint64_t GapUs() noexcept {
	static const uint64_t gap = [] {
		const char* v = std::getenv("KYTY_PROT_MENOS_GAP_US");
		if (v != nullptr && v[0] >= '0' && v[0] <= '9') return static_cast<uint64_t>(std::strtoull(v, nullptr, 10));
		return static_cast<uint64_t>(2000);
	}();
	return gap;
}
[[nodiscard]] inline uint64_t NowUs() noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
// One thread (HotRedirty, GPU thread).
[[nodiscard]] inline bool ScanDue() noexcept {
	static uint64_t last_frame = ~0ull;
	static uint64_t last_us    = 0;
	const uint64_t  frame      = g_frames.load(std::memory_order_relaxed);
	const uint64_t  now        = NowUs();
	if (frame == last_frame && now - last_us < GapUs()) return false;
	last_frame = frame;
	last_us    = now;
	g_scans.fetch_add(1, std::memory_order_relaxed);
	return true;
}

// Regions that ever had hot pages (RegionManager*, never freed by the tracker). Appended under the region's lock.
inline std::atomic<void*>  g_regions[kMaxRegions] {};
inline std::atomic<size_t> g_region_count {0};

[[nodiscard]] inline bool ListRegion(void* manager) noexcept {
	const size_t i = g_region_count.fetch_add(1, std::memory_order_acq_rel);
	if (i >= kMaxRegions) {
		g_region_count.fetch_sub(1, std::memory_order_acq_rel);
		return false;
	}
	g_regions[i].store(manager, std::memory_order_release);
	return true;
}

// Takes up to `want` pages of the cap; returns how many were granted.
[[nodiscard]] inline int64_t Reserve(int64_t want) noexcept {
	int64_t cur = g_hot_pages.load(std::memory_order_relaxed);
	for (;;) {
		const int64_t room = kMaxHotPages - cur;
		if (room <= 0) return 0;
		const int64_t take = want < room ? want : room;
		if (g_hot_pages.compare_exchange_weak(cur, cur + take, std::memory_order_acq_rel)) return take;
	}
}
inline void Release(int64_t pages) noexcept { g_hot_pages.fetch_sub(pages, std::memory_order_acq_rel); }

// (PROT-MENOS4) One thread (HotRedirty, GPU thread). Every 5 s, independent of the write audit.
inline void Report() noexcept {
	struct Snap {
		uint64_t frames, calls, redirty_us, step_us, ww, peek, pages, cpu_writes, promoted, demoted, kept, late, failed, unchanged, scans, lazy, lazy_ns;
	};
	static uint64_t last_ms = 0;
	static Snap     last {};
	const uint64_t  now = NowMs();
	if (last_ms == 0) {
		last_ms = now;
		return;
	}
	if (now - last_ms < 5000) return;
	const auto ld = [](const std::atomic<uint64_t>& a) { return a.load(std::memory_order_relaxed); };
	const Snap cur {ld(g_frames), ld(g_hot_calls), ld(g_redirty_us), ld(g_step_us), ld(g_ww_calls), ld(g_peek_calls),
	                ld(g_redirty_pages), ld(g_cpu_writes), ld(g_promoted), ld(g_demoted), ld(g_kept), ld(g_ww_late),
	                ld(g_ww_failed), ld(g_ww_unchanged), ld(g_scans), ld(g_lazy_calls), ld(g_lazy_ns)};
	const double secs   = static_cast<double>(now - last_ms) / 1000.0;
	const double frames = cur.frames > last.frames ? static_cast<double>(cur.frames - last.frames) : 1.0;
	const auto   pf     = [&](uint64_t a, uint64_t b) { return static_cast<double>(a - b) / frames; };
	std::printf("[prot-menos] %.1f s: %.0f frames, envios/frame %.1f, HotRedirty %.0f us/frame (HotStep %.0f), GetWriteWatch "
	            "%.1f/frame (lecturas %.1f + reset %.1f, fallidas %llu), calientes %lld pag (%.1f MiB, %zu regiones), "
	            "resubido %.2f MiB/frame, sin escribir %.0f pag/frame, fallos escritura CPU %.0f/s, promovidas %llu, "
	            "degradadas %llu, mantenidas %llu, tardias %llu, ww=%d, lecturas completas/frame %.1f (gap %llu us), perezosas %.1f/frame %.0f us/frame (lazy=%d)\n",
	            secs, frames, pf(cur.calls, last.calls), pf(cur.redirty_us, last.redirty_us), pf(cur.step_us, last.step_us),
	            pf(cur.ww + cur.peek, last.ww + last.peek), pf(cur.peek, last.peek), pf(cur.ww, last.ww),
	            static_cast<unsigned long long>(cur.failed - last.failed),
	            static_cast<long long>(g_hot_pages.load(std::memory_order_relaxed)),
	            static_cast<double>(g_hot_pages.load(std::memory_order_relaxed)) * 4096.0 / (1024.0 * 1024.0),
	            g_hot_regions.load(std::memory_order_relaxed), pf(cur.pages, last.pages) * 4096.0 / (1024.0 * 1024.0),
	            pf(cur.unchanged, last.unchanged), static_cast<double>(cur.cpu_writes - last.cpu_writes) / secs,
	            static_cast<unsigned long long>(cur.promoted - last.promoted),
	            static_cast<unsigned long long>(cur.demoted - last.demoted), static_cast<unsigned long long>(cur.kept - last.kept),
	            static_cast<unsigned long long>(cur.late - last.late), WatchOn() ? 1 : 0,
	            pf(cur.scans, last.scans), static_cast<unsigned long long>(GapUs()), pf(cur.lazy, last.lazy),
	            pf(cur.lazy_ns, last.lazy_ns) / 1000.0, LazyOn() ? 1 : 0);
	std::fflush(stdout);
	last    = cur;
	last_ms = now;
}

} // namespace ProtMenos
