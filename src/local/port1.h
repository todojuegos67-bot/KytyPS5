#ifndef EMULATOR_SRC_LOCAL_PORT1_H_
#define EMULATOR_SRC_LOCAL_PORT1_H_
// (port1) KYTY_PORT1=1|2 (off by default): stage 1 of the native-port plan (port0). Measures what an Agc layer at
// the API level ("Agc nativa": record state, resources and draws as calls instead of building and walking PM4) would
// see and what it would not.
// Ghidra (01.007.000, port0/lab): the game imports 211 libSceAgc + 44 libSceAgcDriver functions but calls only 56 of
// them (167 call sites). Every draw and dispatch goes through an import (DrawIndexIndirect 1d0b110, DrawIndexOffset
// 1d0b070, Dispatch 1d0a900, ...), and the emulator writes those packets in CommandBuffer::AllocateDW. The per-draw
// user data (SET_SH_REG), instead, is written by the game's own copy of the Agc binder (1d0cfd0 / 1d0dbc0 copy the
// values straight into packet space the binder reserved, 1d0c450), with no import call.
// Here:
//  * API side (game's job threads): every packet the emulator's Agc writes (AllocateDW) is entered in a direct-mapped
//    table by header address with its first dwords (once written: at the thread's next AllocateDW or at the submit). The dwords the game wrote into the same command
//    buffer between two API packets are counted (the "gap": what a native Agc would have to take from the game's binder).
//  * GPU side (the command processor's packet loop): each packet is looked up: written by the API or by the game,
//    by class (draw/dispatch, SET_SH_REG, SET_CONTEXT_REG, SET_UCONFIG_REG, NOP, other), and the game-written dwords
//    between two draws. Mode 2 also compares each covered draw's dwords with the ones the API wrote (DISCREPANCIA =
//    the packet changed after the API call: an API-level record would have been wrong for it).
// Nothing is changed in what is drawn: the stage's saving is 0 by design (it decides stage 2 with data).
// One line [port1] every 5 s (with or without KYTY_RENDER_STATS). Off: one static load in AllocateDW and in the loop.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <x86intrin.h>

namespace Port1 {

[[nodiscard]] inline uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* text = std::getenv("KYTY_PORT1");
		const auto  m = text == nullptr || text[0] == '\0' ? 0u : static_cast<uint32_t>(std::strtoul(text, nullptr, 10));
		if (m != 0)
			std::printf("port1: KYTY_PORT1=%u: Agc API packets noted and looked up by the command processor%s\n", m,
			            m == 2 ? " (verification: each covered draw's dwords compared with the API's)" : "");
		return m;
	}();
	return mode;
}

// Kinds of API packet (what wrote it).
enum Kind : uint32_t { KNone, KOther, KDrawIndexIndirect, KDrawIndexOffset, KDrawOther, KDispatch, Kinds };
inline constexpr const char* KindNames[Kinds] = {"-", "otro", "DrawIndexIndirect", "DrawIndexOffset", "otro_draw", "dispatch"};

struct Entry {
	std::atomic<uint64_t> key {0}; // header address (0 = empty)
	uint32_t              kind = 0;
	uint32_t              size_dw = 0;
	uint32_t              dw[5] {};
	uint32_t              has_dw = 0;
};
inline constexpr uint32_t TableBits = 18; // 256K entries (~10 MiB, allocated at the first use)
inline Entry*             g_table = nullptr;
inline std::atomic<bool>  g_table_ready {false};

[[nodiscard]] inline uint32_t Slot(uint64_t a) {
	return static_cast<uint32_t>(((a >> 2u) * 0x9E3779B97F4A7C15ull) >> (64u - TableBits));
}
inline Entry* Table() {
	if (g_table_ready.load(std::memory_order_acquire)) return g_table;
	static const bool once = [] {
		g_table = new Entry[1u << TableBits];
		g_table_ready.store(true, std::memory_order_release);
		return true;
	}();
	(void)once;
	return g_table;
}

// ---- API side (any game thread) ----
inline std::atomic<uint64_t> g_api_packets {0}, g_api_dw {0}, g_gap_dw {0}, g_gaps {0}, g_overwrites {0};
inline std::atomic<uint64_t> g_api_kind[Kinds] {};
inline std::atomic<uint64_t> g_api_cycles {0}, g_api_timed {0};
struct ThreadAcc {
	const void* last_buf = nullptr;
	uint64_t    last_end = 0;
	uint64_t    packets = 0, dw = 0, gap = 0, gaps = 0, over = 0, n = 0;
	~ThreadAcc() { Flush(); }
	void Flush() {
		g_api_packets.fetch_add(packets, std::memory_order_relaxed);
		g_api_dw.fetch_add(dw, std::memory_order_relaxed);
		g_gap_dw.fetch_add(gap, std::memory_order_relaxed);
		g_gaps.fetch_add(gaps, std::memory_order_relaxed);
		g_overwrites.fetch_add(over, std::memory_order_relaxed);
		packets = dw = gap = gaps = over = 0;
	}
};
inline thread_local ThreadAcc t_acc;

// The packets the emulator's Agc wrote in [ptr, ptr + size_dw) (complete by the time it is called again on this
// thread, or at the submit): each header is entered with its first dwords; draws and dispatches by their opcode.
inline void EnterWritten(const uint32_t* ptr, uint32_t size_dw) {
	auto*    table = Table();
	uint32_t off = 0;
	while (off < size_dw) {
		const uint32_t* p = ptr + off;
		const uint32_t  h = p[0];
		const uint32_t  len = (h >> 30u) == 3u ? ((h >> 16u) & 0x3fffu) + 2u : 1u;
		const uint32_t  op = (h >> 30u) == 3u ? (h >> 8u) & 0xffu : 0x100u;
		const auto      a = reinterpret_cast<uint64_t>(p);
		auto&           e = table[Slot(a)];
		const auto      old = e.key.load(std::memory_order_relaxed);
		if (old != 0 && old != a) ++t_acc.over;
		e.key.store(0, std::memory_order_relaxed);
		const uint32_t kind = op == 0x25 ? KDrawIndexIndirect : op == 0x35 ? KDrawIndexOffset
		                    : (op == 0x27 || op == 0x2d || op == 0x38 || op == 0x24 || op == 0x2c) ? KDrawOther
		                    : (op == 0x15 || op == 0x16) ? KDispatch : KOther;
		e.kind = kind;
		e.size_dw = len;
		const uint32_t n = len < 5u ? len : 5u;
		const uint32_t m = off + n <= size_dw ? n : size_dw - off;
		for (uint32_t i = 0; i < m; ++i) e.dw[i] = p[i];
		e.has_dw = m;
		e.key.store(a, std::memory_order_release);
		if (kind != KOther) g_api_kind[kind].fetch_add(1, std::memory_order_relaxed);
		off += len;
	}
}
inline thread_local const uint32_t* t_pending = nullptr;
inline thread_local uint32_t        t_pending_dw = 0;
inline void FlushPending() {
	if (t_pending == nullptr) return;
	EnterWritten(t_pending, t_pending_dw);
	t_pending = nullptr;
}

// From CommandBuffer::AllocateDW: `ptr` = the packet(s) the emulator's Agc is about to write in buffer `buf`.
inline void OnAlloc(const void* buf, const uint32_t* ptr, uint32_t size_dw) {
	const bool     timed = (t_acc.n++ & 63u) == 0;
	const uint64_t t0 = timed ? __rdtsc() : 0;
	FlushPending();
	const auto a = reinterpret_cast<uint64_t>(ptr);
	if (t_acc.last_buf == buf && a >= t_acc.last_end && a - t_acc.last_end < (64ull << 20)) {
		t_acc.gap += (a - t_acc.last_end) / 4u;
		t_acc.gaps += a != t_acc.last_end;
	}
	t_acc.last_buf = buf;
	t_acc.last_end = a + 4ull * size_dw;
	t_pending = ptr;
	t_pending_dw = size_dw;
	++t_acc.packets;
	t_acc.dw += size_dw;
	if ((t_acc.n & 255u) == 0) t_acc.Flush();
	if (timed) {
		g_api_cycles.fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
		g_api_timed.fetch_add(1, std::memory_order_relaxed);
	}
}

// ---- GPU side (the command processor's thread; read at the flip on the same thread) ----
enum Cls : uint32_t { CDraw, CDispatch, CSh, CCx, CUc, CNop, COther, Classes };
inline constexpr const char* ClsNames[Classes] = {"draw", "dispatch", "sh_reg", "context_reg", "uconfig_reg", "nop", "otro"};
inline uint64_t g_pk[Classes][2] {}; // packets [class][0 = game, 1 = API]
inline uint64_t g_dw[Classes][2] {}; // dwords
inline uint64_t g_draw_kind[Kinds] {}; // covered draws/dispatches by API kind
inline uint64_t g_discrepancias = 0, g_compared = 0;
inline uint64_t g_gap_game_dw = 0, g_gap_game_pk = 0, g_gaps_seen = 0; // game-written between two draws/dispatches
inline uint64_t g_gap_hist[6] {};    // game dwords before a draw: 0, 1-8, 9-32, 33-128, 129-512, >512
inline uint64_t t_since_dw = 0, t_since_pk = 0;
inline bool     t_seen_draw = false;
inline uint64_t g_lookup_cycles = 0, g_lookups = 0;

[[nodiscard]] inline uint32_t ClassOf(uint32_t opcode, bool draw, bool dispatch) {
	if (draw) return CDraw;
	if (dispatch) return CDispatch;
	switch (opcode) {
		case 0x76: return CSh;
		case 0x69: return CCx;
		case 0x79: return CUc;
		case 0x10: return CNop;
		default: return COther;
	}
}

// `packet` = the header, `len` = its dwords (bounded by the buffer).
inline void OnPacket(uint32_t opcode, const uint32_t* packet, uint32_t len) {
	if (!g_table_ready.load(std::memory_order_acquire)) return;
	const bool     timed = (g_lookups++ & 63u) == 0;
	const uint64_t t0 = timed ? __rdtsc() : 0;
	const bool     draw = opcode == 0x25 || opcode == 0x35 || opcode == 0x27 || opcode == 0x2d || opcode == 0x38 || opcode == 0x24;
	const bool     dispatch = opcode == 0x15 || opcode == 0x16;
	const uint32_t c = ClassOf(opcode, draw, dispatch);
	const auto     a = reinterpret_cast<uint64_t>(packet);
	const auto&    e = g_table[Slot(a)];
	const bool     api = e.key.load(std::memory_order_acquire) == a;
	++g_pk[c][api ? 1 : 0];
	g_dw[c][api ? 1 : 0] += len;
	if (draw || dispatch) {
		if (api) {
			++g_draw_kind[e.kind < Kinds ? e.kind : 0];
			if (Mode() == 2 && e.has_dw != 0) {
				++g_compared;
				const uint32_t n = e.has_dw < len ? e.has_dw : len;
				if (std::memcmp(e.dw, packet, 4ull * n) != 0) ++g_discrepancias;
			}
		}
		if (t_seen_draw) {
			++g_gaps_seen;
			g_gap_game_dw += t_since_dw;
			g_gap_game_pk += t_since_pk;
			const uint64_t d = t_since_dw;
			++g_gap_hist[d == 0 ? 0 : d <= 8 ? 1 : d <= 32 ? 2 : d <= 128 ? 3 : d <= 512 ? 4 : 5];
		}
		t_seen_draw = true;
		t_since_dw = t_since_pk = 0;
	} else if (!api) {
		t_since_dw += len;
		++t_since_pk;
	}
	if (timed) g_lookup_cycles += __rdtsc() - t0;
}
// The packets a run took after its first one (TryNativeXprDraws consumed `consumed` dwords from `packet`).
inline void OnRun(const uint32_t* packet, uint32_t consumed) {
	uint32_t off = 0;
	bool     first = true;
	while (off < consumed) {
		const uint32_t h = packet[off];
		const uint32_t len = (h >> 30u) == 3u ? ((h >> 16u) & 0x3fffu) + 2u : 1u;
		if (!first) OnPacket((h >> 30u) == 3u ? (h >> 8u) & 0xffu : 0x100u, packet + off, len);
		first = false;
		off += len;
	}
}

inline void Report(int64_t now_ns) {
	if (Mode() == 0) return;
	static int64_t  last_ns = 0;
	static uint64_t last_tsc = 0, frames = 0;
	++frames;
	const uint64_t tsc = __rdtsc();
	if (last_ns == 0) {
		last_ns = now_ns;
		last_tsc = tsc;
		frames = 0;
		return;
	}
	if (now_ns - last_ns < 5'000'000'000ll) return;
	const double secs = static_cast<double>(now_ns - last_ns) / 1e9;
	const double ghz = static_cast<double>(tsc - last_tsc) / static_cast<double>(now_ns - last_ns); // TSC ticks per ns
	const double f = frames == 0 ? 1.0 : static_cast<double>(frames);
	t_acc.Flush();
	const uint64_t api_packets = g_api_packets.exchange(0), api_dw = g_api_dw.exchange(0), gap_dw = g_gap_dw.exchange(0);
	const uint64_t gaps = g_gaps.exchange(0), over = g_overwrites.exchange(0);
	const uint64_t api_cyc = g_api_cycles.exchange(0), api_timed = g_api_timed.exchange(0);
	uint64_t       kinds[Kinds];
	for (uint32_t k = 0; k < Kinds; ++k) kinds[k] = g_api_kind[k].exchange(0);
	char line[2048];
	int  n = std::snprintf(line, sizeof(line),
	                       "[port1] %.1f s, %llu frames | API (hilos del juego)/frame: paquetes %.0f (%.0f dw), escritos por el juego "
	                       "entre dos paquetes API %.0f dw en %.0f huecos, pisadas de tabla %.0f, coste %.0f ns/paquete | API por tipo/frame:",
	                       secs, static_cast<unsigned long long>(frames), api_packets / f, api_dw / f, gap_dw / f, gaps / f, over / f,
	                       api_timed ? static_cast<double>(api_cyc) / static_cast<double>(api_timed) / ghz : 0.0);
	for (uint32_t k = KDrawIndexIndirect; k < Kinds && n < 1900; ++k)
		n += std::snprintf(line + n, sizeof(line) - n, " %s=%.0f", KindNames[k], kinds[k] / f);
	n += std::snprintf(line + n, sizeof(line) - n, " | GPU/frame (juego/API) paquetes:");
	for (uint32_t c = 0; c < Classes && n < 1900; ++c)
		n += std::snprintf(line + n, sizeof(line) - n, " %s=%.0f/%.0f (%.0f/%.0f dw)", ClsNames[c], g_pk[c][0] / f, g_pk[c][1] / f,
		                   g_dw[c][0] / f, g_dw[c][1] / f);
	const uint64_t dd = g_pk[CDraw][0] + g_pk[CDraw][1] + g_pk[CDispatch][0] + g_pk[CDispatch][1];
	const uint64_t dd_api = g_pk[CDraw][1] + g_pk[CDispatch][1];
	n += std::snprintf(line + n, sizeof(line) - n,
	                   " | draws+dispatches cubiertos por la API %.1f%% | entre dos draws, escrito por el juego: %.1f dw / %.1f paquetes "
	                   "(hist dw 0:%.0f%% 1-8:%.0f%% 9-32:%.0f%% 33-128:%.0f%% 129-512:%.0f%% >512:%.0f%%) | consulta %.0f ns/paquete",
	                   dd ? 100.0 * dd_api / dd : 0.0, g_gaps_seen ? static_cast<double>(g_gap_game_dw) / g_gaps_seen : 0.0,
	                   g_gaps_seen ? static_cast<double>(g_gap_game_pk) / g_gaps_seen : 0.0,
	                   g_gaps_seen ? 100.0 * g_gap_hist[0] / g_gaps_seen : 0.0, g_gaps_seen ? 100.0 * g_gap_hist[1] / g_gaps_seen : 0.0,
	                   g_gaps_seen ? 100.0 * g_gap_hist[2] / g_gaps_seen : 0.0, g_gaps_seen ? 100.0 * g_gap_hist[3] / g_gaps_seen : 0.0,
	                   g_gaps_seen ? 100.0 * g_gap_hist[4] / g_gaps_seen : 0.0, g_gaps_seen ? 100.0 * g_gap_hist[5] / g_gaps_seen : 0.0,
	                   g_lookups ? static_cast<double>(g_lookup_cycles) / (static_cast<double>(g_lookups) / 64.0) / ghz : 0.0);
	if (Mode() == 2 && n < 2000)
		n += std::snprintf(line + n, sizeof(line) - n, " | VERIFICA draws comparados %.0f/frame, DISCREPANCIA %llu", g_compared / f,
		                   static_cast<unsigned long long>(g_discrepancias));
	std::printf("%s\n", line);
	std::fflush(stdout);
	std::memset(g_pk, 0, sizeof(g_pk));
	std::memset(g_dw, 0, sizeof(g_dw));
	std::memset(g_draw_kind, 0, sizeof(g_draw_kind));
	std::memset(g_gap_hist, 0, sizeof(g_gap_hist));
	g_compared = g_gap_game_dw = g_gap_game_pk = g_gaps_seen = g_lookup_cycles = g_lookups = 0; // discrepancias: cumulative
	last_ns = now_ns;
	last_tsc = tsc;
	frames = 0;
}

} // namespace Port1

#endif // EMULATOR_SRC_LOCAL_PORT1_H_
