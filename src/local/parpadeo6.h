#pragma once
// (parpadeo6) KYTY_PARPADEO6=1: the XPR chunk bits settled before the game reads them (0xa1f290's entry, the hook of
// parpadeo5), so that 0xad8bc0 / 0xad92f0 never record their draws from OLD bytes.
//
// Why (native/parpadeo6/INFORME.md):
//   - The game waits on the CPU for the XPR label (0xab6080 -> 0x8351b0) and then reads the bits a GPU dispatch wrote.
//     The emulator writes that label when it translates the RELEASE_MEM (CommandProcessor::WriteAtEndOfPipe), before
//     the GPU runs the dispatch; that alone is harmless: the bits' pages are no-access (GPU-written) and a read of them
//     faults and is read back behind the dispatch, the GPU waited for (BufferCache::ReadMemory -> BeginGuestReadback /
//     ReadMemoryOnGpu: the copy recorded after everything translated so far, CopyGuestReadback waits for its tick).
//   - With KYTY_RAIZ the readback's bytes are KEPT aside and put in when the page is released (lazy write-back). The
//     put-in first makes the run READ-ONLY with its OLD bytes and only then compares and puts the GPU's in
//     (NativePendingMergeOpenFast: "the old bytes readable from here"). Any other game thread reading the page meanwhile
//     reads old bytes without a fault. The passes 40..45 (Depth, GBuffer, ...) all wait on the same label and read
//     bits in neighbouring frame-allocator memory at the same moment: one thread's fault and put-in expose old bytes to
//     the others. The "[raiz]" line measures it: ~40,000 pages/s readable with old bytes, ~190 us each, an estimated
//     ~40 game reads of old bytes a second (20261116-084528-244). RAIZ=0 has no kept bytes (alias view): no window.
//   - Memory::RaizSettleRange (raiz12) closes it for one reader: what the GPU still owns read back, the deferred
//     releases applied by this thread, then until no put-in is under way over the range and no kept bytes are left
//     (at most 4 ms). After that no page of the range shows old bytes until the GPU writes it again (next frame).
// Here: every 0xa1f290 call settles its bits (and mask2 on the 0xad92f0 path) with that mode 1, whatever
// KYTY_RAIZ_SETTLE says, before the game's code runs. Nothing without KYTY_RAIZ (no put-ins there).
// Cost: the readback the game's own read fault would make, a moment earlier; a wait only while a put-in is under way.
// No new lock order: the game thread waits for the GPU thread as at any read fault; the GPU thread never waits here.
//
// Line "[parpadeo6]" every 5 s: calls, ranges settled, how many waited (readback / put-in) and their microseconds.
// Off by default (read once). KYTY_PARPADEO5 still measures (its flags then show what the settle found).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Parpadeo6 {

// (parpadeo7) KYTY_PARPADEO7=1 turns it on too: its prefetch keeps this settle as the net (local/parpadeo7.h).
[[nodiscard]] inline bool On() noexcept {
	static const bool on = [] {
		const char* v  = std::getenv("KYTY_PARPADEO6");
		const char* v7 = std::getenv("KYTY_PARPADEO7");
		return (v != nullptr && v[0] == '1' && v[1] == 0) || (v7 != nullptr && v7[0] == '1' && v7[1] == 0);
	}();
	return on;
}

// Memory::RaizSettle* flags (kernel/memory.h), repeated here so this header needs nothing else.
enum : uint32_t { SettleBusy = 1u, SettlePending = 2u, SettleReadback = 4u, SettleWaited = 8u, SettleLeft = 16u };

struct Counters {
	std::atomic<uint64_t> calls {0};         // 0xa1f290 calls settled (with bits)
	std::atomic<uint64_t> ranges {0};        // ranges settled (bits, and mask2 on path B)
	std::atomic<uint64_t> busy {0};          // a put-in under way over the range at entry (old bytes readable then)
	std::atomic<uint64_t> pending {0};       // kept GPU bytes not in yet at entry
	std::atomic<uint64_t> readback {0};      // read back here (the game's fault, made first)
	std::atomic<uint64_t> readback_us {0};   // ... the time of those calls
	std::atomic<uint64_t> waited {0};        // waited for a put-in / kept bytes (no readback)
	std::atomic<uint64_t> waited_us {0};     // ... the time of those calls
	std::atomic<uint64_t> left {0};          // still under way after the 4 ms cap (the game may read old bytes)
	std::atomic<uint64_t> total_us {0};      // the settle's whole time, every call
	std::atomic<uint64_t> max_us {0};        // the longest call
	std::atomic<uint64_t> changed {0};       // (with KYTY_PARPADEO5) bytes readable before the settle that it changed
};
inline Counters g_c;
inline std::atomic<int64_t> g_last_ns {0};

inline int64_t NowNs() noexcept {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One call's settle (the game thread, in the hook): `flags` of its ranges ORed, `us` its time.
inline void Note(uint32_t flags, uint32_t ranges, uint64_t us) noexcept {
	auto& c = g_c;
	c.calls.fetch_add(1, std::memory_order_relaxed);
	c.ranges.fetch_add(ranges, std::memory_order_relaxed);
	if (flags & SettleBusy) c.busy.fetch_add(1, std::memory_order_relaxed);
	if (flags & SettlePending) c.pending.fetch_add(1, std::memory_order_relaxed);
	if (flags & SettleReadback) {
		c.readback.fetch_add(1, std::memory_order_relaxed);
		c.readback_us.fetch_add(us, std::memory_order_relaxed);
	} else if (flags & SettleWaited) {
		c.waited.fetch_add(1, std::memory_order_relaxed);
		c.waited_us.fetch_add(us, std::memory_order_relaxed);
	}
	if (flags & SettleLeft) c.left.fetch_add(1, std::memory_order_relaxed);
	c.total_us.fetch_add(us, std::memory_order_relaxed);
	uint64_t m = c.max_us.load(std::memory_order_relaxed);
	while (us > m && !c.max_us.compare_exchange_weak(m, us, std::memory_order_relaxed)) {
	}
}
inline void NoteChanged() noexcept { g_c.changed.fetch_add(1, std::memory_order_relaxed); }

// The 5 s line: printed by the first game thread past the 5 s (one CAS), the counters taken as they are.
inline void MaybeReport() noexcept {
	const int64_t now  = NowNs();
	int64_t       last = g_last_ns.load(std::memory_order_relaxed);
	if (last == 0) {
		g_last_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 5'000'000'000ll || !g_last_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
	auto&          c        = g_c;
	const auto     take     = [](std::atomic<uint64_t>& a) { return static_cast<unsigned long long>(a.exchange(0, std::memory_order_relaxed)); };
	const auto     calls    = take(c.calls);
	const auto     ranges   = take(c.ranges);
	const auto     busy     = take(c.busy);
	const auto     pending  = take(c.pending);
	const auto     rb       = take(c.readback);
	const auto     rb_us    = take(c.readback_us);
	const auto     wt       = take(c.waited);
	const auto     wt_us    = take(c.waited_us);
	const auto     left     = take(c.left);
	const auto     total_us = take(c.total_us);
	const auto     max_us   = take(c.max_us);
	const auto     changed  = take(c.changed);
	const double   seconds  = static_cast<double>(now - last) / 1e9;
	std::printf("[parpadeo6] %.1f s (KYTY_PARPADEO6=1, bits del XPR asentados en 0xa1f290 antes de que el juego los lea): llamadas %llu, "
	            "rangos %llu; ESPERAS %llu (%.2f ms en total): leidos de vuelta %llu (%.2f ms), esperando put-in/guardados %llu "
	            "(%.2f ms); al entrar: put-in en curso %llu, guardados sin meter %llu; SIN ASENTAR en 4 ms %llu; tiempo del arreglo "
	            "%.2f ms en total (%.3f ms/s), max %llu us; bytes viejos cambiados por el asentado %llu (solo con KYTY_PARPADEO5)\n",
	            seconds, calls, ranges, rb + wt, static_cast<double>(rb_us + wt_us) / 1000.0, rb, static_cast<double>(rb_us) / 1000.0,
	            wt, static_cast<double>(wt_us) / 1000.0, busy, pending, left, static_cast<double>(total_us) / 1000.0,
	            seconds > 0.0 ? static_cast<double>(total_us) / 1000.0 / seconds : 0.0, max_us, changed);
	std::fflush(stdout);
}

} // namespace Parpadeo6
