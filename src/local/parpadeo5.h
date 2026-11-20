#pragma once
// (parpadeo5) KYTY_PARPADEO5: what the game reads when it decides how many XPR draws it records, and with what bytes.
//
// Ghidra (projelf_par1; native/parpadeo5/ghidra), Demon's Souls 01.007.000:
//   0xa0de50 sets the view's two labels *(view+0x120)+0x120 and +0x1c0 to 0 and launches the job "XPRPreRender"
//     (0xa0a980 -> 0xa0b020 / 0xa0b480). That job records, per XPR section, a compute dispatch that writes the
//     section's chunk-visibility bits into frame-allocator memory (0xa1f930: idx 0..3; 0xa20750 via 0xa0b120: shadow
//     idx 4..), stores the bits' addresses in the table *(view+0x58) + section*0x2d0 + idx*8, and ends with a
//     RELEASE_MEM of 1 into the label (0x912960: +0x120 after the shadow culls, +0x1c0 after 0xa1f930's).
//   0xab6080 (one call per pass; pass entry = *(view+0x58) + 0x2cd30 + i*0x28, filled by 0xa0d2d0 / 0xa0d740: entries
//     0..39 "XPR_kSMLT_Shadow*" wait on +0x120, 40..45 Depth / DepthFadeOut / DepthFadeOutInvLODFade / GBuffer / Debug
//     on +0x1c0) WAITS ON THE CPU until *entry[0] == 1 (0x8351b0(.., entry[0], 1, 0 (==), 1) at 0xab6107: a spin with
//     yields) and then calls 0xa1f290(section, view, entry) for each section of the pass (the call at 0xab630a).
//   0xa1f290 picks the section's bits (table[idx], idx from entry+0x20/+0x24) and tail-jumps to
//     0xad8bc0(mesh, bits, entry, view)   when config(+0x4c) == 0 and (kind 0, 6 or *(view+0x68)+0xcac != 1), or
//     0xad92f0(mesh, mask2, bits', entry, view) otherwise (the index-cull path).
//   0xad8bc0 reads the bits ON THE CPU and records, for every chunk c < *(u16*)(mesh+0x64) with its bit set and
//     (*(u32*)(*(mesh+0x50)+0x20)[c] >> kind) & 1, one DRAW_INDEX_INDIRECT at offset c*0x14 of the base *(mesh+0x40)
//     (SET_BASE 0x909460(.., 0x1c, 4, 1, 0x800, *(mesh+0x40))). So a base's draws in a frame = the union over the
//     calls of the chunks drawn: exactly what parpadeo.h counts per base on the GPU thread ("DRAWS CAEN (CPU)").
// The emulator writes a RELEASE_MEM's label when it TRANSLATES the packet (WriteAtEndOfPipe), and with native memory
// relies on the GPU-written pages being no-access until their bytes are in (NativeFlushGpuWrites returns at once).
// So the game reads the bits right after the label shows 1: whatever guest memory holds then is what it draws from.
//
// KYTY_PARPADEO5 (off by default; read at first use):
//   1  measures, changes nothing the game sees: a hook on 0xa1f290's entry computes the bits exactly as the game will
//      read them (when their pages are no-access the readback the game's fault would make is made first, the same work
//      a moment earlier), the draws 0xad8bc0 / 0xad92f0 will record, and per args base the union of a CPU frame. Flags:
//      the label not 1 at entry, the pages no-access, and - at the next frame's call of the same (section, pass) - the
//      old bits read again: different bytes there mean the GPU's bytes landed AFTER the game read them.
//      The GPU thread (parpadeo.h, needs KYTY_PARPADEO >= 1) classifies each read range as the buffer cache sees it
//      (CPU-written / GPU-written / unknown) and, on each "DRAWS CAEN (CPU)", prints the game's last frames for that
//      base.
//   2  as 1, and the arreglo candidato: the bits settled before the game reads them (Memory::RaizSettleRange: read
//      back if the GPU owns them, deferred releases applied, put-ins waited for); the bytes before and after the
//      settle are compared (a difference is a read of old bytes, measured and repaired).
// Lines: "[parpadeo5] ..." every 5 s (counters); "[parpadeo5] DRAWS CAEN EN EL JUEGO" (the first 64): the middle of
// three consecutive CPU frames of a base under 80 % of both neighbours (and by >= 4), with that frame's flags.
//
// Threads: the game's job threads (the hook) and the GPU thread (the checks): one mutex.

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace Parpadeo5 {

[[nodiscard]] inline int Level() noexcept {
	static const int level = [] {
		const char* v = std::getenv("KYTY_PARPADEO5");
		if (v == nullptr || v[0] < '0' || v[0] > '2' || v[1] != 0) return 0;
		return v[0] - '0';
	}();
	return level;
}

enum Flag : uint32_t {
	FlagLabelNot1      = 1u,     // the label 0xab6080 waited for was not 1 at 0xa1f290's entry
	FlagNoAccess       = 2u,     // the bits' pages were no-access at entry (GPU bytes not in: read back first)
	FlagSettleBusy     = 4u,     // (2) a put-in under way over the bits at entry
	FlagSettlePending  = 8u,     // (2) kept GPU bytes over the bits not in yet
	FlagSettleReadback = 16u,    // (2) read back by the settle
	FlagSettleLeft     = 32u,    // (2) still busy after the settle's cap (4 ms)
	FlagSettleChanged  = 64u,    // (2) the bits differ after the settle: the game would have read OLD bytes
	FlagLaterChanged   = 128u,   // the bits, read again at the next frame, differ from what the game read
	FlagPathB          = 256u,   // 0xad92f0 (index-cull path)
	FlagUnreadable     = 512u,   // the bits could not be read even after the readback (counted as 0 draws here)
	FlagGpuCpuDirty    = 1024u,  // (GPU thread, after the read) the buffer cache holds the range as CPU-written
	FlagGpuGpuWritten  = 2048u,  // (GPU thread) ... as GPU-written (not read back yet)
	FlagGpuUnknown     = 4096u,  // (GPU thread) ... not in any buffer
};
inline void FlagsText(uint32_t f, char* out, size_t size) {
	std::snprintf(out, size, "%s%s%s%s%s%s%s%s%s%s%s%s%s", (f & FlagLabelNot1) ? " etiqueta-no-1" : "",
	              (f & FlagNoAccess) ? " bits-sin-acceso(GPU)" : "", (f & FlagSettleBusy) ? " put-in-en-curso" : "",
	              (f & FlagSettlePending) ? " guardados-sin-meter" : "", (f & FlagSettleReadback) ? " leidos-de-vuelta" : "",
	              (f & FlagSettleLeft) ? " SIN-ASENTAR-en-4ms" : "", (f & FlagSettleChanged) ? " BYTES-VIEJOS-CORREGIDOS" : "",
	              (f & FlagLaterChanged) ? " BITS-CAMBIARON-DESPUES-DE-LEERLOS" : "", (f & FlagPathB) ? " ruta-ad92f0" : "",
	              (f & FlagUnreadable) ? " ilegibles" : "", (f & FlagGpuCpuDirty) ? " cache:sucios-de-CPU" : "",
	              (f & FlagGpuGpuWritten) ? " cache:escritos-por-GPU" : "", (f & FlagGpuUnknown) ? " cache:sin-buffer" : "");
	if (out[0] == '\0') std::snprintf(out, size, " nada");
}
inline uint64_t UnixMs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}
inline uint64_t Hash(const uint64_t* words, size_t n) {
	uint64_t h = 0xcbf29ce484222325ull;
	for (size_t i = 0; i < n; ++i) h = (h ^ words[i]) * 0x100000001b3ull + (h >> 29u);
	return h;
}

// One CPU frame of a base: the union of the chunks its calls drew (path A), plus path B's draws (not deduplicated).
struct FrameRec {
	uint64_t unix_ms = 0;
	uint32_t draws = 0, calls = 0, flags = 0, settle_fixed = 0, later_changed = 0;
};
inline constexpr uint32_t RecRing = 8;
struct BaseAgg {
	uint64_t                        base = 0;
	std::vector<uint64_t>           drawn; // (path A: chunk bits of this frame's union)
	uint32_t                        extra = 0;
	std::array<uint64_t, 64>        keys {};
	uint32_t                        nkeys = 0;
	FrameRec                        cur {};
	std::array<FrameRec, RecRing>   ring {};
	uint64_t                        n = 0; // frames closed
};
// The last read of a (section, pass): to read its bits again at the key's next call.
struct KeyRec {
	uint64_t key = 0, bits_at = 0, bytes = 0, hash = 0, frame = 0;
	uint32_t base_slot = UINT32_MAX;
	bool     readable = false;
};
// A read range for the GPU thread to classify.
struct Pending {
	uint64_t bits_at = 0, bytes = 0, frame = 0;
	uint32_t base_slot = 0;
};
struct Counters {
	uint64_t calls = 0, skipped = 0, path_b = 0, label_not1 = 0, no_access = 0, unreadable = 0, settle_busy = 0, settle_pending = 0,
	         settle_readback = 0, settle_left = 0, settle_changed = 0, later_compared = 0, later_changed = 0, draws = 0, frames = 0,
	         drops = 0, drops_flagged = 0, gpu_checked = 0, gpu_cpu_dirty = 0, gpu_gpu_written = 0, gpu_unknown = 0, settle_us = 0;
};
struct State {
	std::mutex                            mutex;
	std::array<BaseAgg, 512>              bases {};
	uint32_t                              count = 0;
	std::array<KeyRec, 8192>              keys {};
	std::array<Pending, 1024>             pending {};
	uint32_t                              npending = 0;
	uint64_t                              pending_lost = 0;
	Counters                              c {};
	uint32_t                              logged = 0;
	std::chrono::steady_clock::time_point last {};
};
[[nodiscard]] inline State& S() {
	static State state;
	return state;
}

inline BaseAgg* BaseOf(State& s, uint64_t base, uint32_t* slot) {
	for (uint32_t i = 0; i < s.count; ++i)
		if (s.bases[i].base == base) {
			*slot = i;
			return &s.bases[i];
		}
	if (s.count >= s.bases.size()) return nullptr;
	*slot = s.count;
	auto& b = s.bases[s.count++];
	b.base  = base;
	return &b;
}
inline KeyRec* KeyOf(State& s, uint64_t key) {
	const uint64_t h = (key * 0x9E3779B97F4A7C15ull) >> 51u;
	for (uint32_t probe = 0; probe < 64; ++probe) {
		auto& k = s.keys[(h + probe) % s.keys.size()];
		if (k.key == key) return &k;
		if (k.key == 0) {
			k     = {};
			k.key = key;
			return &k;
		}
	}
	return nullptr;
}
// The frame `frame` of base `b`, still the current one or kept in the ring.
inline FrameRec* FrameOf(BaseAgg& b, uint64_t frame) {
	if (frame == b.n) return &b.cur;
	if (frame < b.n && b.n - frame <= RecRing) return &b.ring[frame % RecRing];
	return nullptr;
}
// Closes the base's current frame and judges the middle of the last three (under the mutex).
inline void CloseFrame(State& s, BaseAgg& b) {
	uint32_t union_draws = 0;
	for (const uint64_t w: b.drawn) union_draws += static_cast<uint32_t>(std::popcount(w));
	b.cur.draws = union_draws + b.extra;
	b.ring[b.n % RecRing] = b.cur;
	++b.n;
	++s.c.frames;
	std::fill(b.drawn.begin(), b.drawn.end(), 0);
	b.extra = 0;
	b.nkeys = 0;
	b.cur   = {};
	if (b.n < 3) return;
	const FrameRec& f0 = b.ring[(b.n - 3) % RecRing];
	const FrameRec& f1 = b.ring[(b.n - 2) % RecRing];
	const FrameRec& f2 = b.ring[(b.n - 1) % RecRing];
	const uint32_t  low = (std::min)(f0.draws, f2.draws);
	if (!(f1.draws * 5 < low * 4 && low >= f1.draws + 4)) return;
	++s.c.drops;
	const uint32_t bad = FlagLabelNot1 | FlagSettleChanged | FlagLaterChanged | FlagSettleLeft | FlagUnreadable | FlagGpuCpuDirty;
	if ((f1.flags & bad) != 0) ++s.c.drops_flagged;
	if (s.logged++ >= 64) return;
	char ft[400];
	FlagsText(f1.flags, ft, sizeof(ft));
	std::printf("[parpadeo5] DRAWS CAEN EN EL JUEGO (CPU, 0xad8bc0/0xad92f0): base 0x%" PRIx64 ": %u draws (frame anterior %u, siguiente %u), "
	            "%u llamadas; bits corregidos por el asentado en %u llamadas, cambiados despues de leerlos en %u; unix_ms~%" PRIu64
	            "; al leer:%s\n",
	            b.base, f1.draws, f0.draws, f2.draws, f1.calls, f1.settle_fixed, f1.later_changed, f1.unix_ms, ft);
	std::fflush(stdout);
}

// The 5 s line (under the mutex).
inline void Report(State& s) {
	const auto now = std::chrono::steady_clock::now();
	if (s.last == std::chrono::steady_clock::time_point {}) s.last = now;
	const double seconds = std::chrono::duration<double>(now - s.last).count();
	if (seconds < 5.0) return;
	s.last         = now;
	const auto& c  = s.c;
	const auto  u  = [](uint64_t v) { return static_cast<unsigned long long>(v); };
	std::printf("[parpadeo5] %.1f s (KYTY_PARPADEO5=%d): 0xa1f290 llamadas %llu (sin bits %llu, ruta 0xad92f0 %llu), draws %llu, bases %u, "
	            "frames de base %llu; al leer: etiqueta no 1 %llu, bits sin acceso (GPU) %llu, ilegibles %llu; asentado: put-in en curso "
	            "%llu, guardados %llu, leidos de vuelta %llu, sin asentar en 4 ms %llu, BYTES VIEJOS CORREGIDOS %llu (%.2f ms en "
	            "total); relectura al frame siguiente: %llu comparadas, CAMBIADAS %llu; cache en el hilo GPU: %llu rangos, sucios de "
	            "CPU %llu, escritos por GPU %llu, sin buffer %llu (perdidos %llu); DRAWS CAEN EN EL JUEGO %llu (con marca %llu)\n",
	            seconds, Level(), u(c.calls), u(c.skipped), u(c.path_b), u(c.draws), s.count, u(c.frames), u(c.label_not1), u(c.no_access),
	            u(c.unreadable), u(c.settle_busy), u(c.settle_pending), u(c.settle_readback), u(c.settle_left), u(c.settle_changed),
	            static_cast<double>(c.settle_us) / 1000.0, u(c.later_compared), u(c.later_changed), u(c.gpu_checked), u(c.gpu_cpu_dirty),
	            u(c.gpu_gpu_written), u(c.gpu_unknown), u(s.pending_lost), u(c.drops), u(c.drops_flagged));
	std::fflush(stdout);
	s.c            = {};
	s.pending_lost = 0;
}

// What one 0xa1f290 call records (the hook, a game thread). `drawn`: the chunks drawn (path A; nullptr for path B),
// `words` of them. `later`: the previous read of this key compared again (0 none, 1 same, 2 different).
struct Call {
	uint64_t        key = 0, base = 0, bits_at = 0, bytes = 0, hash = 0;
	const uint64_t* drawn = nullptr;
	size_t          words = 0;
	uint32_t        draws = 0, flags = 0;
	bool            readable = false, skipped = false;
	uint64_t        settle_us = 0;
};
// The previous read of `key`, to compare again before the new call (the hook reads the old range itself, outside the
// mutex). False: none.
inline bool PreviousRead(uint64_t key, uint64_t* bits_at, uint64_t* bytes, uint64_t* hash) {
	auto&                  s = S();
	const std::scoped_lock lock(s.mutex);
	KeyRec*                k = KeyOf(s, key);
	if (k == nullptr || !k->readable || k->bits_at == 0) return false;
	*bits_at = k->bits_at, *bytes = k->bytes, *hash = k->hash;
	return true;
}
inline void Note(const Call& call, int later) {
	auto&                  s = S();
	const std::scoped_lock lock(s.mutex);
	auto&                  c = s.c;
	++c.calls;
	if (call.skipped) {
		++c.skipped;
		return Report(s);
	}
	c.settle_us += call.settle_us;
	if (call.flags & FlagPathB) ++c.path_b;
	if (call.flags & FlagLabelNot1) ++c.label_not1;
	if (call.flags & FlagNoAccess) ++c.no_access;
	if (call.flags & FlagUnreadable) ++c.unreadable;
	if (call.flags & FlagSettleBusy) ++c.settle_busy;
	if (call.flags & FlagSettlePending) ++c.settle_pending;
	if (call.flags & FlagSettleReadback) ++c.settle_readback;
	if (call.flags & FlagSettleLeft) ++c.settle_left;
	if (call.flags & FlagSettleChanged) ++c.settle_changed;
	c.draws += call.draws;
	KeyRec* k = KeyOf(s, call.key);
	// The previous read of this key, compared again: credited to the frame it was read in.
	if (later != 0 && k != nullptr) {
		++c.later_compared;
		if (later == 2) {
			++c.later_changed;
			if (k->base_slot < s.count)
				if (FrameRec* f = FrameOf(s.bases[k->base_slot], k->frame); f != nullptr) {
					f->flags |= FlagLaterChanged;
					++f->later_changed;
				}
		}
	}
	uint32_t slot = 0;
	BaseAgg* b    = BaseOf(s, call.base, &slot);
	if (b != nullptr) {
		bool seen = false;
		for (uint32_t i = 0; i < b->nkeys && !seen; ++i) seen = b->keys[i] == call.key;
		if (seen) CloseFrame(s, *b);
		if (b->nkeys < b->keys.size()) b->keys[b->nkeys++] = call.key;
		if (b->cur.calls == 0) b->cur.unix_ms = UnixMs();
		++b->cur.calls;
		b->cur.flags |= call.flags;
		if (call.flags & FlagSettleChanged) ++b->cur.settle_fixed;
		if (call.drawn != nullptr) {
			if (b->drawn.size() < call.words) b->drawn.resize(call.words, 0);
			for (size_t w = 0; w < call.words; ++w) b->drawn[w] |= call.drawn[w];
		} else {
			b->extra += call.draws;
		}
		if (s.npending < s.pending.size()) s.pending[s.npending++] = {call.bits_at, call.bytes, b->n, slot};
		else ++s.pending_lost;
	}
	if (k != nullptr) {
		k->bits_at   = call.bits_at;
		k->bytes     = call.bytes;
		k->hash      = call.hash;
		k->readable  = call.readable;
		k->base_slot = b != nullptr ? slot : UINT32_MAX;
		k->frame     = b != nullptr ? b->n : 0;
	}
	Report(s);
}

// The GPU thread: the ranges read since, as the buffer cache holds them now. `classify(vaddr, size)` returns
// FlagGpuCpuDirty / FlagGpuGpuWritten / FlagGpuUnknown (or 0: registered, clean).
template <typename Classify>
void CheckPending(Classify&& classify) {
	if (Level() == 0) return;
	auto&                  s = S();
	const std::scoped_lock lock(s.mutex);
	for (uint32_t i = 0; i < s.npending; ++i) {
		const auto&    p = s.pending[i];
		const uint32_t f = classify(p.bits_at, p.bytes);
		++s.c.gpu_checked;
		if (f & FlagGpuCpuDirty) ++s.c.gpu_cpu_dirty;
		if (f & FlagGpuGpuWritten) ++s.c.gpu_gpu_written;
		if (f & FlagGpuUnknown) ++s.c.gpu_unknown;
		if (p.base_slot < s.count)
			if (FrameRec* fr = FrameOf(s.bases[p.base_slot], p.frame); fr != nullptr) fr->flags |= f;
	}
	s.npending = 0;
}

// For "DRAWS CAEN (CPU)" on the GPU thread: the game's last CPU frames of that base (oldest first).
inline void PrintRecent(uint64_t base) {
	if (Level() == 0) return;
	auto&                  s = S();
	const std::scoped_lock lock(s.mutex);
	for (uint32_t i = 0; i < s.count; ++i) {
		auto& b = s.bases[i];
		if (b.base != base) continue;
		std::printf("[parpadeo5]   lo que grabo el juego para 0x%" PRIx64 " (gancho 0xa1f290), ultimos frames:", base);
		const uint64_t first = b.n > RecRing ? b.n - RecRing : 0;
		for (uint64_t f = first; f < b.n; ++f) {
			const auto& r = b.ring[f % RecRing];
			char        ft[400];
			FlagsText(r.flags, ft, sizeof(ft));
			std::printf(" | ms %" PRIu64 ": %u draws, %u llamadas,%s", r.unix_ms, r.draws, r.calls, ft);
		}
		std::printf("\n");
		std::fflush(stdout);
		return;
	}
	std::printf("[parpadeo5]   base 0x%" PRIx64 ": el gancho 0xa1f290 no la ha visto (no la graba 0xad8bc0/0xad92f0)\n", base);
	std::fflush(stdout);
}

} // namespace Parpadeo5
