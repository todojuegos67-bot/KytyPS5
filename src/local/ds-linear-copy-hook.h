#pragma once
// (render4) KYTY_DS_LINEAR_COPY_HOOK (off by default; Demon's Souls 01.007.000 only): the game's linear copy
// (eboot+0x912e60: cs_memset32 bound, three temporary buffers, {count, period} and a dispatch of (count + 63) / 64
// groups: dst[i] = src[i % period]) announced by a private packet the hook writes into the game's command buffer just
// before the packets the game's own function writes (loader/demonsSoulsCopy.cpp):
//
//   [0]      a type-3 IT_NOP of R_ZERO, PacketDwords long (every reader of the stream passes over an R_ZERO NOP whose
//            first word is not a marker's: a state pass, a speculation, the draw-state observer)
//   [1] [2]  Magic0, Magic1
//   [3] [4]  the destination address      [5] [6]  the source address
//   [7]      count (dwords)               [8]      period (dwords; >= count: a plain copy)
//   [9]      the dwords of the game's own packets that follow (0: not sealed - a plain NOP)
//   [10][11] XXH3 of those dwords as they were when the game's function returned
//
// The command processor (CommandProcessor::ProcessPm4), at a sealed packet whose following dwords still hash the
// same, makes the copy as the dispatch's register check does (DemonsSouls::TryLinearCopyRanges) and passes over the
// game's packets; when it does not (a refusal, every 64th use, a frame capture, a state pass, a speculation) it passes
// over the NOP alone and the game's packets run as they always did. The game's own function always runs: the game's
// state is what it was without the hook.
//
// 1: on. 2: verify - nothing is passed over; the ranges the packet announces are compared with those the dispatch
// that follows copies (the counters below; "[ds-copy-hook]" lines for the first differences).

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <xxhash.h>

namespace DsLinearCopyHook {

constexpr uint32_t PacketDwords = 12;
// KYTY_PM4(PacketDwords, IT_NOP, R_ZERO): 0xC0000000 | (PacketDwords - 2) << 16 | 0x10 << 8.
constexpr uint32_t Header = 0xC0000000u | ((PacketDwords - 2u) << 16u) | (0x10u << 8u);
constexpr uint32_t Magic0 = 0x4b594c43u, Magic1 = 0x4f50591fu; // (not a marker's 0x6875xxxx first word)
constexpr uint32_t DstWord = 3, SrcWord = 5, CountWord = 7, PeriodWord = 8, SpanWord = 9, HashWord = 10;
// The game's packets of one copy: a register table, the 12 user data registers and the dispatch are 24 dwords.
constexpr uint32_t MaxSpanDwords = 64;

// KYTY_DS_LINEAR_COPY_HOOK, read at its first use (the port sets its settings' variables after the process started).
[[nodiscard]] inline uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* text = std::getenv("KYTY_DS_LINEAR_COPY_HOOK");
		if (text == nullptr || text[0] == '\0') return 0u;
		return text[0] == '1' && text[1] == '\0' ? 1u : text[0] == '2' && text[1] == '\0' ? 2u : 0u;
	}();
	return mode;
}

[[nodiscard]] inline uint64_t SpanHash(const uint32_t* dwords, uint32_t count) {
	return XXH3_64bits(dwords, size_t {count} * sizeof(uint32_t));
}

// A hook packet at `packet` with `remaining` dwords left in its buffer (sealed or not).
[[nodiscard]] inline bool IsPacket(const uint32_t* packet, uint32_t remaining) {
	return remaining >= PacketDwords && packet[0] == Header && packet[1] == Magic0 && packet[2] == Magic1;
}
// ... sealed, the game's packets after it whole in the buffer and unchanged since.
[[nodiscard]] inline bool IsSealed(const uint32_t* packet, uint32_t remaining) {
	const uint32_t span = packet[SpanWord];
	return span != 0 && span <= MaxSpanDwords && span <= remaining - PacketDwords &&
	       SpanHash(packet + PacketDwords, span) == (uint64_t {packet[HashWord]} | uint64_t {packet[HashWord + 1]} << 32u);
}
[[nodiscard]] inline uint64_t Address(const uint32_t* packet, uint32_t word) {
	return uint64_t {packet[word]} | uint64_t {packet[word + 1]} << 32u;
}

// Counters (any thread; relaxed). The game's threads: calls of the hook, packets sealed, and why one was not (0 the
// copy is not a plain one, 1 no room in the command buffer, 2 the game's packets are not the ones expected). The
// render thread: copies made here, sealed packets left to the game's packets, and the verify mode's results.
enum Counter : uint32_t {
	Calls, Sealed, NotPlain, NoRoom, OtherPackets, Taken, Passed, VerifySame, VerifyDifferent, VerifyRefused, Counters
};
inline std::atomic<uint64_t> g_counters[Counters] {};
inline void Count(Counter counter) { g_counters[counter].fetch_add(1, std::memory_order_relaxed); }

} // namespace DsLinearCopyHook
