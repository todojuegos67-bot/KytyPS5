#ifndef EMULATOR_SRC_LOCAL_PORT2_H_
#define EMULATOR_SRC_LOCAL_PORT2_H_
// (port2) KYTY_PORT2=1|2 (off by default): stage E2 of the native-port plan (port0/PLAN.md), "Agc nativa".
//
// What the GPU thread stops doing: walking, packet by packet, the register state the game's Agc binder writes before
// every draw and dispatch (the user data SET_SH_REG, the game's user-data markers NOP "hu", context / uconfig / index
// state, indirect register blocks). Those packets are taken as a flow of operations, decoded on the game's job thread
// that recorded them, and applied in one step when the command processor reaches that point of the DCB/ACB.
//
// Where the flow is recorded (Ghidra, 01.007.000; port2/INFORME.md): the game's static copy of Agc does not write
// complete packets by itself. Its binder (1d0c450) reserves the user data "hole" with the Agc import
// SetShRegisterRangeDirect (1d0a6e0 -> the emulator's AgcCbSetShRegisterRangeDirect, here) and fills it later with
// 1d0cfd0 (a block) / 1d0dbc0 (one dword), up to the draw; 1d10840 (-> 1d0edb0) clears the binder's "emitted" flags
// right after the draw, so the next draw reserves a new hole. Every draw and dispatch goes through a 5-54 byte
// wrapper (1d0b110 DrawIndexIndirect, 1d0b070 DrawIndexOffset, 1d0b050 DrawIndexAuto, 1d0b080 DrawIndirect, 1d0b0d0
// DrawIndexIndirectMulti, 1d0a900 Dispatch, 1d0b040 / 1d0ab80 DispatchIndirect) that checks the modifier and tail-jumps
// to the import: the emulator's Agc (libs/agc.cpp) is the hook, at the exact moment the draw's state is complete. So:
//   * at each draw / dispatch the emulator's Agc writes, a 6-dword marker (a NOP R_ZERO of its own, not sealed: any
//     reader passes over it as a plain NOP) right after the draw packet;
//   * at the next draw / dispatch on the same command buffer, the dwords between that marker and the new packet (what
//     the binder, the game and the other Agc calls wrote: the next draw's state) are decoded - only register / index
//     state, the binder's markers and fillers; anything else (labels, EOP / RELEASE_MEM, waits, barriers, DMA,
//     predication, the linear-copy announcements, any other NOP) leaves the marker a plain NOP - into a record of a
//     host ring, and the marker is sealed with the record's position and the span's length;
//   * the command processor, at a sealed marker (outside speculation, state passes, chunks, captures), applies the
//     record with the same handlers as the walk (user data straight into the SGPR arrays, with the same marker types
//     and the draw state observer's shadow), and passes over the span: it lands on the draw packet, which runs as today.
// The game's packets stay written (the game's binder works on pointers into them: they are its own bookkeeping, and
// the fallback); the PM4 of a span that was not sealed, or whose record is gone, is walked as always. The order of the
// command buffer is the order of the flow: one marker per span, the span right after it.
// The eboot's binder and wrappers are compared byte by byte (a hash per function) when it loads; any other bytes: off,
// said once ("[port2] DESACTIVADO").
//
// 1: on, alternating 5 s ON / 5 s OFF (OFF: no markers written at all, the stream is today's) and a [port2] line every
//    5 s (operations per frame by type, PM4 packets avoided, fallbacks by reason, submissions / frame ms ON vs OFF,
//    AHORRO MEDIDO). 2: verification, always recording: each sealed span is applied from the flow on a copy of the
//    registers, then the span's own PM4 runs through the handlers; the two register states are compared field by field
//    (DISCREPANCIA by field) and the PM4's is kept (the image is always the PM4's).
// Off: one static load in AllocateDW, in the packet loop at a NOP of this length, and at the flip.

#include "graphics/guest_gpu/pm4.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <x86intrin.h>

namespace Port2 {

namespace P = Libs::Graphics::Pm4;

[[nodiscard]] inline uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* text = std::getenv("KYTY_PORT2");
		const auto  m    = text == nullptr || text[0] == '\0' ? 0u : static_cast<uint32_t>(std::strtoul(text, nullptr, 10));
		const auto  v    = m <= 2 ? m : 0u;
		if (v != 0)
			std::printf("port2: KYTY_PORT2=%u: Agc nativa (the binder's state as a flow, applied without walking its PM4)%s\n", v,
			            v == 2 ? " - verification: flow and PM4 both applied and compared, the image is the PM4's"
			                   : " - alternating 5 s ON / 5 s OFF, [port2] line every 5 s");
		return v;
	}();
	return mode;
}

// The eboot's binder and wrappers (InstallCheck, loader/demonsSoulsCopy.cpp): 0 not looked at, 1 01.007.000's, -1 not.
inline std::atomic<int32_t> g_eboot {0};
inline std::atomic<bool>    g_window_on {true}; // (1) the ON window; flipped at the [port2] line
[[nodiscard]] inline bool   Recording() {
    const uint32_t m = Mode();
    return m != 0 && g_eboot.load(std::memory_order_relaxed) == 1 && (m == 2 || g_window_on.load(std::memory_order_relaxed));
}

// ---- the eboot check (01.007.000; offsets = Ghidra addresses, FNV-1a 64 of the function's bytes) ----
struct EbootFunction {
	uint64_t    offset;
	uint32_t    size;
	uint64_t    fnv;
	const char* what;
};
inline constexpr EbootFunction EbootFunctions[] = {
    {0x1d0c450, 641, 0xa68e609afd45faa0ull, "binder: reserve user data (SetShRegisterRangeDirect / top-down block)"},
    {0x1d0c340, 267, 0x981d1cc281e31306ull, "binder: reserve extended user data"},
    {0x1d0cfd0, 187, 0xec0564a9bedf111eull, "binder: copy user data block into the hole"},
    {0x1d0dbc0, 175, 0x0f3366df18b0427full, "binder: one user data dword into the hole"},
    {0x1d10840, 40, 0xaf62f47dba730013ull, "binder: not emitted after the draw"},
    {0x1d0edb0, 43, 0xd7ceea5293342565ull, "binder: emitted flags cleared"},
    {0x1d18770, 28, 0x6daa8b97b77488d5ull, "binder: second stage reset"},
    {0x1d0a6e0, 143, 0x389dfce7a99839e7ull, "SetShRegisterRangeDirect wrapper"},
    {0x1d0a770, 144, 0x8002721810ef52afull, "SetShRegisterRangeDirect wrapper (pointer)"},
    {0x1d0fc90, 10, 0x7bbe31ebcc89aa76ull, "GetDataPacketPayloadAddress wrapper"},
    {0x1d0a900, 19, 0x7a3a8f29ed2f3e5dull, "Dispatch wrapper"},
    {0x1d0ab80, 18, 0xf356682a0a52f3aeull, "Acb DispatchIndirect wrapper"},
    {0x1d0b040, 18, 0x70153eecd265d7edull, "Dcb DispatchIndirect wrapper"},
    {0x1d0b050, 21, 0x31a17cb91dff1aadull, "DrawIndexAuto wrapper"},
    {0x1d0b070, 21, 0xa0bcbd6771464a95ull, "DrawIndexOffset wrapper"},
    {0x1d0b080, 21, 0x119b8dd072fc7ff6ull, "DrawIndirect wrapper"},
    {0x1d0b0d0, 54, 0x32e82978696794adull, "DrawIndexIndirectMulti wrapper"},
    {0x1d0b110, 21, 0xb1fef3376f8c0696ull, "DrawIndexIndirect wrapper"},
};
[[nodiscard]] inline uint64_t Fnv1a64(const uint8_t* bytes, uint32_t size) {
	uint64_t h = 0xcbf29ce484222325ull;
	for (uint32_t i = 0; i < size; ++i) h = (h ^ bytes[i]) * 0x100000001b3ull;
	return h;
}

// ---- the marker (a NOP R_ZERO of 6 dwords) ----
constexpr uint32_t MarkerDw = 6;
constexpr uint32_t Header   = 0xC0000000u | ((MarkerDw - 2u) << 16u) | (0x10u << 8u); // KYTY_PM4(6, IT_NOP, R_ZERO)
constexpr uint32_t Magic0 = 0x3254504bu, Magic1 = 0x776f6c66u;  // (not a "hu" marker's 0x6875xxxx first word)
constexpr uint32_t IdWord = 3, SpanWord = 5;                    // [3][4]: nonce (not sealed) / record (sealed); [5]: span
constexpr uint32_t MaxSpanDw = 1024;                            // longer spans are walked
[[nodiscard]] inline bool IsMarker(const uint32_t* p, uint64_t remaining_dw) {
	return remaining_dw >= MarkerDw && p[0] == Header && p[1] == Magic0 && p[2] == Magic1;
}
// Not sealed (no span after it, a span left to the walk, or dropped): a plain NOP.
[[nodiscard]] inline bool IsEmptyMarker(const uint32_t* p, uint64_t remaining_dw) {
	return IsMarker(p, remaining_dw) && p[SpanWord] == 0;
}

// ---- the record (a host ring) ----
struct Record {
	uint64_t abs;     // its position (never 0)
	uint64_t marker;  // the marker's address
	uint32_t span_dw; // the span it stands for
	uint32_t words;   // operation words after it
	uint32_t packets; // PM4 packets of the span (fillers included)
	uint32_t fillers; // one-dword NOPs among them
};
static_assert(sizeof(Record) == 32);
enum OpKind : uint32_t { OUd = 1, OGen = 2, OMk = 3 };
enum Stage : uint32_t { SPs, SGs, SHs, SCs, Stages };
// Operation types (the [port2] line's "por tipo").
enum Type : uint32_t { TUd, TSh, TCx, TUc, TIdx, TInd, TMk, TNop, Types };
inline constexpr const char* TypeNames[Types] = {"user_data", "sh", "context", "uconfig", "indice", "indirecto", "marcador_hu", "relleno"};
// One word per operation, then its values: user data (stage, first slot, count, the span offset; then the values), a
// packet for its handler (type, length, offset; then the packet, header included), a binder marker (id, offset).
[[nodiscard]] constexpr uint32_t UdWord(uint32_t stage, uint32_t slot, uint32_t n, uint32_t off) {
	return OUd | stage << 4u | slot << 8u | n << 14u | off << 20u;
}
[[nodiscard]] constexpr uint32_t GenWord(uint32_t type, uint32_t len, uint32_t off) { return OGen | type << 4u | len << 8u | off << 20u; }
[[nodiscard]] constexpr uint32_t MkWord(uint32_t id, uint32_t off) { return OMk | (id & 0xfffu) << 4u | off << 20u; }
[[nodiscard]] constexpr uint32_t KindOf(uint32_t w) { return w & 15u; }
[[nodiscard]] constexpr uint32_t UdStage(uint32_t w) { return (w >> 4u) & 15u; }
[[nodiscard]] constexpr uint32_t UdSlot(uint32_t w) { return (w >> 8u) & 63u; }
[[nodiscard]] constexpr uint32_t UdCount(uint32_t w) { return (w >> 14u) & 63u; }
[[nodiscard]] constexpr uint32_t GenType(uint32_t w) { return (w >> 4u) & 15u; }
[[nodiscard]] constexpr uint32_t GenLen(uint32_t w) { return (w >> 8u) & 0xfffu; }
[[nodiscard]] constexpr uint32_t MkId(uint32_t w) { return (w >> 4u) & 0xfffu; }
[[nodiscard]] constexpr uint32_t OffOf(uint32_t w) { return (w >> 20u) & 0xfffu; }
inline constexpr uint32_t UdBase[Stages]     = {P::SPI_SHADER_USER_DATA_PS_0, P::SPI_SHADER_USER_DATA_GS_0,
                                                P::SPI_SHADER_USER_DATA_HS_0, P::COMPUTE_USER_DATA_0};
inline constexpr uint32_t UdCapacity[Stages] = {32, 32, 32, 16};

constexpr uint64_t                RingBytes = uint64_t {16} << 20u; // (~15 frames of records; allocated at the first seal)
inline std::atomic<uint8_t*>      g_ring {nullptr};
inline std::atomic<uint64_t>      g_head {64};
[[nodiscard]] inline uint8_t*     Ring() {
    if (auto* r = g_ring.load(std::memory_order_acquire)) return r;
    static uint8_t* const once = [] {
        auto* r = static_cast<uint8_t*>(std::calloc(1, RingBytes));
        g_ring.store(r, std::memory_order_release);
        return r;
    }();
    return once;
}
// A record's position (0: none).
[[nodiscard]] inline uint64_t RingAlloc(uint32_t bytes) {
	for (int i = 0; i < 4; ++i) {
		const uint64_t abs = g_head.fetch_add(bytes, std::memory_order_relaxed);
		if (abs % RingBytes + bytes <= RingBytes) return abs;
	}
	return 0;
}

// ---- fallbacks ----
// Recording side (game threads): why a span was not sealed.
enum Fall : uint32_t {
	FNoRoom, FSegment, FFar, FBroken, FCollision, FSubmit, FPred, FOpcode, FHuOther, FNopOther, FBadLen, FShNop, FTooBig,
	FRing, Falls
};
inline constexpr const char* FallNames[Falls] = {"sin_sitio", "otro_segmento", "hueco_largo", "marcador_pisado", "colision_tabla",
                                                 "submit", "predicado", "otro_paquete", "marcador_hu_otro", "nop_otro",
                                                 "longitud", "sh_nop", "flujo_grande", "anillo"};
// GPU side: why a sealed marker was walked instead.
enum GFall : uint32_t { GOff, GLost, GBeyond, GLength, GRace, GFalls };
inline constexpr const char* GFallNames[GFalls] = {"gpu_fuera_de_via", "gpu_registro_perdido", "gpu_fuera_de_buffer",
                                                   "gpu_longitud_handler", "gpu_carrera_anillo"};

inline std::atomic<uint64_t> g_rec_markers {0}, g_rec_sealed {0}, g_rec_empty {0}, g_rec_packets {0}, g_rec_cycles {0},
    g_rec_timed {0};
inline std::atomic<uint64_t> g_rec_fall[Falls] {};
inline std::atomic<uint64_t> g_rec_op[257] {}; // FOpcode by opcode (256: not type 3)
struct RecAcc {
	uint64_t markers = 0, sealed = 0, empty = 0, packets = 0, cycles = 0, timed = 0, n = 0;
	uint64_t fall[Falls] {};
	~RecAcc() { Flush(); }
	void Flush() {
		g_rec_markers.fetch_add(markers, std::memory_order_relaxed);
		g_rec_sealed.fetch_add(sealed, std::memory_order_relaxed);
		g_rec_empty.fetch_add(empty, std::memory_order_relaxed);
		g_rec_packets.fetch_add(packets, std::memory_order_relaxed);
		g_rec_cycles.fetch_add(cycles, std::memory_order_relaxed);
		g_rec_timed.fetch_add(timed, std::memory_order_relaxed);
		for (uint32_t f = 0; f < Falls; ++f)
			if (fall[f] != 0) g_rec_fall[f].fetch_add(fall[f], std::memory_order_relaxed);
		markers = sealed = empty = packets = cycles = timed = 0;
		std::memset(fall, 0, sizeof(fall));
	}
	void Tick() {
		if ((++n & 127u) == 0) Flush();
	}
};
inline thread_local RecAcc t_rec;

// ---- decoding a span (game thread) ----
// The span [begin, begin + span) as operations into `out` (at most `cap` words). False: `why` (and `bad_op`).
[[nodiscard]] inline bool Decode(const uint32_t* begin, uint32_t span, uint32_t* out, uint32_t cap, uint32_t& words,
                                 uint32_t& packets, uint32_t& fillers, Fall& why, uint32_t& bad_op) {
	words = packets = fillers = 0;
	uint32_t at = 0;
	while (at < span) {
		const uint32_t h = begin[at];
		if (h == 0x80000000u) {
			++at;
			++packets;
			++fillers;
			continue;
		}
		if ((h >> 30u) != 3u) return bad_op = 256, why = FOpcode, false;
		if ((h & 1u) != 0) return why = FPred, false; // (predicated: decided when the GPU gets there)
		const uint32_t len = ((h >> 16u) & 0x3fffu) + 2u;
		if (len > span - at) return why = FBadLen, false;
		const uint32_t op = (h >> 8u) & 0xffu;
		const auto     put = [&](uint32_t type) {
            if (words + 1u + len > cap) return why = FTooBig, false;
            out[words++] = GenWord(type, len, at);
            std::memcpy(out + words, begin + at, 4ull * len);
            words += len;
            return true;
		};
		switch (op) {
			case P::IT_NOP: {
				if (((h >> 2u) & (P::R_NUM - 1u)) != P::R_ZERO) return why = FNopOther, false; // (custom NOP commands)
				// The binder's user-data markers ("hu": 0 none, 4 V#, 0xd region); any other R_ZERO NOP (the linear-copy
				// announcement, these markers, unknown ones) is walked.
				if (len < 2u || (begin[at + 1u] & 0xffff0000u) != 0x68750000u) return why = FNopOther, false;
				const uint32_t id = begin[at + 1u] & 0xfffu;
				if (id != 0u && id != 4u && id != 0xdu) return why = FHuOther, false;
				if (words + 1u > cap) return why = FTooBig, false;
				out[words++] = MkWord(id, at);
				break;
			}
			case P::IT_SET_SH_REG: {
				if (len < 3u) return why = FBadLen, false;
				const uint32_t offset = begin[at + 1u], n = len - 2u;
				if (offset == P::SH_NOP) return why = FShNop, false; // (its handler's length is not the header's)
				uint32_t stage = Stages;
				for (uint32_t s = 0; s < Stages; ++s)
					if (offset >= UdBase[s] && offset < UdBase[s] + UdCapacity[s]) stage = s;
				if (stage != Stages && offset - UdBase[stage] + n <= UdCapacity[stage]) {
					if (words + 1u + n > cap) return why = FTooBig, false;
					out[words++] = UdWord(stage, offset - UdBase[stage], n, at);
					std::memcpy(out + words, begin + at + 2u, 4ull * n);
					words += n;
				} else if (!put(TSh)) {
					return false;
				}
				break;
			}
			case P::IT_SET_CONTEXT_REG:
				if (!put(TCx)) return false;
				break;
			case P::IT_SET_UCONFIG_REG:
			case P::IT_SET_UCONFIG_REG_INDEX:
				if (!put(TUc)) return false;
				break;
			case P::IT_INDEX_BASE:
			case P::IT_INDEX_BUFFER_SIZE:
			case P::IT_INDEX_TYPE:
			case P::IT_NUM_INSTANCES:
			case P::IT_SET_BASE:
				if (!put(TIdx)) return false;
				break;
			case P::IT_SET_SH_REG_INDIRECT:
			case P::IT_SET_CONTEXT_REG_INDIRECT:
			case P::IT_SET_UCONFIG_REG_INDIRECT:
				if (!put(TInd)) return false;
				break;
			default: return bad_op = op, why = FOpcode, false;
		}
		at += len;
		++packets;
	}
	return true;
}

// ---- the pending marker of each command buffer (game threads) ----
struct Pending {
	std::atomic_flag         busy = ATOMIC_FLAG_INIT;
	std::atomic<const void*> buf {nullptr};
	std::atomic<uint32_t*>   marker {nullptr}; // (atomic: looked at without the lock first)
	const void*              bottom = nullptr;
	const void*              top    = nullptr;
	uint64_t                 nonce  = 0;
};
constexpr uint32_t   PendingSlots = 1024;
inline Pending       g_pending[PendingSlots];
inline std::atomic<uint64_t> g_nonce {1};
[[nodiscard]] inline Pending& PendingOf(const void* buf) {
	const auto a = reinterpret_cast<uint64_t>(buf);
	return g_pending[((a >> 4u) * 0x9E3779B97F4A7C15ull) >> 54u]; // (10 bits)
}
struct PendingLock {
	Pending& p;
	explicit PendingLock(Pending& pending): p(pending) {
		while (p.busy.test_and_set(std::memory_order_acquire)) _mm_pause();
	}
	~PendingLock() { p.busy.clear(std::memory_order_release); }
	PendingLock(const PendingLock&)            = delete;
	PendingLock& operator=(const PendingLock&) = delete;
};

// (graphicsRun.cpp, TryDrawIndirectRun) The draws of the last run (its dwords are no longer 5 per draw over markers).
inline thread_local uint32_t t_run_draws = 0;

inline thread_local bool     t_draw     = false; // (set by a draw / dispatch of the emulator's Agc right before its AllocateDW)
inline thread_local uint32_t t_suppress = 0;     // (no marker: the linear-copy hook checks the exact packets its function wrote)
inline void                  MarkDraw() {
    if (Mode() != 0 && t_suppress == 0) t_draw = true;
}
struct Suppress {
	Suppress() { ++t_suppress; }
	~Suppress() { --t_suppress; }
	Suppress(const Suppress&)            = delete;
	Suppress& operator=(const Suppress&) = delete;
};
[[nodiscard]] inline bool TakeDraw() { return std::exchange(t_draw, false); }

// The marker's span [marker + MarkerDw, end), decoded and sealed (`p` locked, its marker checked).
inline void Seal(Pending& p, const uint32_t* end) {
	uint32_t* const marker = p.marker.load(std::memory_order_relaxed);
	const uint32_t* begin  = marker + MarkerDw;
	const auto      span   = static_cast<uint32_t>(end - begin);
	if (span == 0) {
		++t_rec.empty;
		return;
	}
	constexpr uint32_t        Cap = 2u * MaxSpanDw + 64u;
	thread_local uint32_t     scratch[Cap];
	uint32_t                  words = 0, packets = 0, fillers = 0, bad_op = 0;
	Fall                      why = Falls;
	if (!Decode(begin, span, scratch, Cap, words, packets, fillers, why, bad_op)) {
		++t_rec.fall[why];
		if (why == FOpcode) g_rec_op[bad_op].fetch_add(1, std::memory_order_relaxed);
		return;
	}
	uint8_t* const ring  = Ring();
	const auto     bytes = static_cast<uint32_t>((sizeof(Record) + 4ull * words + 7u) & ~uint64_t {7});
	const uint64_t abs   = ring != nullptr ? RingAlloc(bytes) : 0;
	if (abs == 0) {
		++t_rec.fall[FRing];
		return;
	}
	auto* rec    = reinterpret_cast<Record*>(ring + abs % RingBytes);
	rec->abs     = 0; // (invalid while it is written)
	std::atomic_thread_fence(std::memory_order_release);
	rec->marker  = reinterpret_cast<uint64_t>(marker);
	rec->span_dw = span;
	rec->words   = words;
	rec->packets = packets;
	rec->fillers = fillers;
	std::memcpy(rec + 1, scratch, 4ull * words);
	std::atomic_thread_fence(std::memory_order_release);
	rec->abs = abs;
	marker[IdWord]     = static_cast<uint32_t>(abs);
	marker[IdWord + 1] = static_cast<uint32_t>(abs >> 32u);
	std::atomic_thread_fence(std::memory_order_release);
	reinterpret_cast<volatile uint32_t*>(marker)[SpanWord] = span; // (last: sealed)
	++t_rec.sealed;
	t_rec.packets += packets;
}

// From CommandBuffer::AllocateDW, `ptr` = where the API's packet(s) begin (after any grow), before the cursor moves:
// a draw / dispatch seals the buffer's pending marker; any packet checks it is still in the same segment.
template <class B>
inline void OnAlloc(B* buf, const uint32_t* ptr, bool draw) {
	if (g_eboot.load(std::memory_order_relaxed) != 1) return;
	auto& p = PendingOf(buf);
	if (p.marker.load(std::memory_order_relaxed) == nullptr || p.buf.load(std::memory_order_relaxed) != buf) return;
	const bool     timed = draw && (t_rec.n & 15u) == 0;
	const uint64_t t0    = timed ? __rdtsc() : 0;
	{
		PendingLock     lock(p);
		uint32_t* const m = p.marker.load(std::memory_order_relaxed);
		if (m == nullptr || p.buf.load(std::memory_order_relaxed) != buf) return;
		const uint32_t* const end_of_marker = m + MarkerDw;
		Fall                  why           = Falls;
		if (p.bottom != buf->bottom || p.top != buf->top || ptr < end_of_marker) why = FSegment;
		else if (ptr - end_of_marker > MaxSpanDw) why = FFar;
		else if (m[0] != Header || m[1] != Magic0 || m[2] != Magic1 || m[IdWord] != static_cast<uint32_t>(p.nonce) ||
		         m[IdWord + 1] != static_cast<uint32_t>(p.nonce >> 32u) || m[SpanWord] != 0)
			why = FBroken;
		if (why != Falls) {
			++t_rec.fall[why];
			p.marker.store(nullptr, std::memory_order_relaxed);
		} else if (draw) {
			Seal(p, ptr);
			p.marker.store(nullptr, std::memory_order_relaxed);
		}
	}
	if (timed) {
		t_rec.cycles += __rdtsc() - t0;
		++t_rec.timed;
	}
	t_rec.Tick();
}

// From CommandBuffer::AllocateDW after a draw / dispatch's packet was allocated (the cursor past it): its marker, not
// sealed, right after it - when the buffer has the room without its grow callback (never asked for more room).
template <class B>
inline void AfterDrawAlloc(B* buf) {
	if (!Recording()) return;
	++t_rec.markers;
	const auto up = reinterpret_cast<uintptr_t>(buf->cursor_up), down = reinterpret_cast<uintptr_t>(buf->cursor_down);
	const uint64_t available = down > up ? (down - up) / 4u : 0u;
	if (available <= buf->reserved_dw || available - buf->reserved_dw < MarkerDw) {
		++t_rec.fall[FNoRoom];
		t_rec.Tick();
		return;
	}
	uint32_t* const marker = buf->cursor_up;
	const uint64_t  nonce  = g_nonce.fetch_add(1, std::memory_order_relaxed);
	marker[0]              = Header;
	marker[1]              = Magic0;
	marker[2]              = Magic1;
	marker[IdWord]         = static_cast<uint32_t>(nonce);
	marker[IdWord + 1]     = static_cast<uint32_t>(nonce >> 32u);
	marker[SpanWord]       = 0;
	buf->cursor_up         = marker + MarkerDw;
	auto&       p          = PendingOf(buf);
	PendingLock lock(p);
	if (p.marker.load(std::memory_order_relaxed) != nullptr && p.buf.load(std::memory_order_relaxed) != buf)
		++t_rec.fall[FCollision]; // (another buffer's span: left to the walk)
	p.buf.store(buf, std::memory_order_relaxed);
	p.bottom = buf->bottom;
	p.top    = buf->top;
	p.nonce  = nonce;
	p.marker.store(marker, std::memory_order_relaxed);
	t_rec.Tick();
}

// At every submit (DCB or ACB): every pending marker is left a plain NOP (no span is sealed after the GPU may read it).
inline void OnSubmit() {
	if (g_eboot.load(std::memory_order_relaxed) != 1) return;
	uint64_t dropped = 0;
	for (auto& p: g_pending) {
		if (p.marker.load(std::memory_order_relaxed) == nullptr) continue;
		PendingLock lock(p);
		if (p.marker.load(std::memory_order_relaxed) != nullptr) ++dropped;
		p.marker.store(nullptr, std::memory_order_relaxed);
	}
	t_rec.fall[FSubmit] += dropped;
	t_rec.Tick();
}

// ---- the command processor's side ----
// The record a sealed marker stands for (null: `why`); `span`: its span.
[[nodiscard]] inline const Record* Find(const uint32_t* marker, uint32_t remaining_dw, uint32_t& span, GFall& why) {
	span = marker[SpanWord];
	if (span > remaining_dw - MarkerDw) return why = GBeyond, nullptr;
	const uint64_t abs  = marker[IdWord] | uint64_t {marker[IdWord + 1]} << 32u;
	uint8_t* const ring = g_ring.load(std::memory_order_acquire);
	if (ring == nullptr || abs == 0 || abs % 8u != 0 || abs % RingBytes > RingBytes - sizeof(Record)) return why = GLost, nullptr;
	const auto* rec = reinterpret_cast<const Record*>(ring + abs % RingBytes);
	if (rec->abs != abs || rec->marker != reinterpret_cast<uint64_t>(marker) || rec->span_dw != span ||
	    abs % RingBytes + sizeof(Record) + 4ull * rec->words > RingBytes)
		return why = GLost, nullptr;
	return rec;
}

// GPU-side counters (the command processors' threads; read at the flip).
inline std::atomic<uint64_t> g_gpu_applied {0}, g_gpu_packets {0}, g_gpu_dw {0}, g_gpu_empty {0}, g_gpu_cycles {0},
    g_gpu_timed {0};
inline std::atomic<uint64_t> g_gpu_type[Types] {};
inline std::atomic<uint64_t> g_gpu_fall[GFalls] {};
// (2) Verification.
enum Field : uint32_t { VCx, VUc, VUdPs, VUdGs, VUdHs, VUdCs, VShOther, VMarker, VIndex, VLength, VChanged, Fields };
inline constexpr const char* FieldNames[Fields] = {"context", "uconfig", "ud_ps", "ud_gs", "ud_hs", "ud_cs", "sh_otro",
                                                   "marcador", "indice/bases/instancias", "longitud", "pm4_cambiado_tras_sellar"};
inline std::atomic<uint64_t> g_verified {0}, g_discrepancias {0};
inline std::atomic<uint64_t> g_field[Fields] {};

} // namespace Port2

#endif // EMULATOR_SRC_LOCAL_PORT2_H_
