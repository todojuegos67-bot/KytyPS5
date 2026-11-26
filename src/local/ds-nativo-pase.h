#pragma once
// (F5 pase 1) KYTY_NATIVO_PASE_SSRCLEAR (off by default; Demon's Souls 01.007.000 only): the first pass of the frame
// drawn natively from the game's own data instead of the emulator's PM4 -> Vulkan translation.
//
// The pass (Ghidra, 01.007.000): SSRPreRender 0x9d6030 opens the profiler scopes "SSRPreRender" and "SSRClear"
// (ScopeBegin 0x999220) and calls 0x9ad540(ctx_holder, texture) with texture = *(*(view + 0x38) + 0x70), the SSR
// target. 0x9ad540 is the engine's texture clear and has no other caller: with a command buffer (ctx = *holder != 0)
// it jumps to 0x9ad560, which binds the engine's clear compute shader (shader table entry 0x1d70 / 0x10), puts the
// texture's T# (texture + 0xbc, 32 bytes) in resource slot 0 and dispatches ((width + 7) / 8, (height + 7) / 8, 1).
// One texture, one dispatch, nothing read: the whole pass is "this image = 0".
//
// The hook (loader/demonsSoulsNativoPase.inc) runs the game's function whole, as the linear copy hook does: it writes
// this announcement at the command buffer's cursor first, and seals it after the function returned when the packets
// it wrote are only shader register writes and, last, that dispatch:
//
//   [0]       a type-3 IT_NOP of R_ZERO, PacketDwords long (every reader passes over an R_ZERO NOP that is not a marker)
//   [1] [2]   Magic0, Magic1
//   [3..10]   the texture's T# (8 dwords), as the game's function put it in the resource slot
//   [11]      the dwords of the game's own packets that follow (0: not sealed - a plain NOP)
//   [12][13]  XXH3 of those dwords as they were when the game's function returned
//   [14]      the dwords of the game's packets up to and including its dispatch (=1 passes over these only: the
//             EVENT_WRITEs after the dispatch, CS_PARTIAL_FLUSH 0x407 and 0x410, still run, so the game's sync stays)
//
// The command processor (graphicsRun.cpp), at a sealed packet whose following dwords still hash the same:
//   1 (nativo):   clears the image natively (RenderExecutor::DsNativeClear: the texture cache's image of that T#,
//                 vkCmdClearColorImage to 0) and passes over the game's packets (no shader bound, no descriptors, no
//                 dispatch translated). A refusal (no image of exactly that T#, more levels / layers than the view,
//                 a speculation, a state pass, a frame capture) leaves the game's packets to run as always.
//   2 (verifica): both: the native clear first, then the game's packets as always (the image shown is the emulated
//                 one); the log counts the native clears made and refused, so =1 is only tried once =2 shows them made.
// "[nativo-pase] SSRClear" line every 5 s (GPU thread).

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <xxhash.h>

namespace DsNativoPase {

constexpr uint32_t PacketDwords = 15;
// KYTY_PM4(PacketDwords, IT_NOP, R_ZERO): 0xC0000000 | (PacketDwords - 2) << 16 | 0x10 << 8 (not the linear copy's).
constexpr uint32_t Header = 0xC0000000u | ((PacketDwords - 2u) << 16u) | (0x10u << 8u);
constexpr uint32_t Magic0 = 0x5053414eu, Magic1 = 0x52435353u; // "NASP" "SSCR" (not a marker's 0x6875xxxx)
constexpr uint32_t TSharpWord = 3, SpanWord = 11, HashWord = 12, SkipWord = 14;
// The game's packets of one clear: the shader's registers, the user data and the dispatch.
constexpr uint32_t MaxSpanDwords = 256;

// KYTY_NATIVO_PASE_SSRCLEAR=1 / 2, read at its first use (the port sets its settings' variables after the start).
[[nodiscard]] inline uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* text = std::getenv("KYTY_NATIVO_PASE_SSRCLEAR");
		if (text == nullptr || text[0] == '\0') return 0u; // (11-26 limpieza) apagado por defecto // (11-24) validated: ON by default, =0 off
		if (text[1] != '\0') return 0u;
		return text[0] == '1' ? 1u : text[0] == '2' ? 2u : 0u;
	}();
	return mode;
}

[[nodiscard]] inline uint64_t SpanHash(const uint32_t* dwords, uint32_t count) {
	return XXH3_64bits(dwords, size_t {count} * sizeof(uint32_t));
}
[[nodiscard]] inline bool IsPacket(const uint32_t* packet, uint32_t remaining) {
	return remaining >= PacketDwords && packet[0] == Header && packet[1] == Magic0 && packet[2] == Magic1;
}
[[nodiscard]] inline bool IsSealed(const uint32_t* packet, uint32_t remaining) {
	const uint32_t span = packet[SpanWord];
	return span != 0 && span <= MaxSpanDwords && span <= remaining - PacketDwords &&
	       SpanHash(packet + PacketDwords, span) == (uint64_t {packet[HashWord]} | uint64_t {packet[HashWord + 1]} << 32u);
}

// Counters (relaxed). Game threads: Calls, Sealed, NoRoom, OtherPackets. GPU thread: the rest.
enum Counter : uint32_t { Calls, Sealed, NoRoom, OtherPackets, Native, Refused, Verified, Counters };
inline std::atomic<uint64_t> g_counters[Counters] {};
inline void Count(Counter counter) { g_counters[counter].fetch_add(1, std::memory_order_relaxed); }

// The GPU thread, at each announcement: one line every 5 s.
inline void Report() {
	using Clock = std::chrono::steady_clock;
	static Clock::time_point last = Clock::now();
	const auto               now  = Clock::now();
	if (now - last < std::chrono::seconds(5)) return;
	last = now;
	const auto v = [](Counter c) { return static_cast<unsigned long long>(g_counters[c].load(std::memory_order_relaxed)); };
	std::printf("[nativo-pase] SSRClear modo=%u (%s): llamadas=%llu selladas=%llu sin_sitio=%llu otros_paquetes=%llu | "
	            "nativos=%llu (pase del juego saltado) verificados=%llu (nativo + emulado) rechazados=%llu (emulado)\n",
	            Mode(), Mode() == 1 ? "nativo" : "verifica", v(Calls), v(Sealed), v(NoRoom), v(OtherPackets), v(Native),
	            v(Verified), v(Refused));
	std::fflush(stdout);
}

} // namespace DsNativoPase
