#pragma once
// [audio-exacto] AudioOut2 as the PS5 library does it, from what Demon's Souls sends (Ghidra, FUN_01ddbe30 = the
// game's per-grain port update before AudioOut2ContextPush, and FUN_01dda120 = its reroute check):
//  - main ports (8/2/1 ch): attribute 1 = one gain per channel (the game's master level, this+0x84). On the console the
//    gain scales the port; the emulator only used it on the pad ports.
//  - AudioOut2PortGetState.num_channels: the game builds its speaker mask from it (1 -> 0x1ff1, 2 -> 0x1ff3,
//    6 -> 0x1ff7, 8 -> 0x1fff) and mixes for that layout. The console reports the real output's channels; the
//    emulator reported the port's (8), so the game mixed 7.1 for a stereo device and the host downmixed it.
//  - object ports: attributes 3 (position x,y,z), 4 (spread), 1 (gain), 2 (priority), 0 (PCM), 6 (reset).
//    The position was ignored (centre); KYTY_AUDIO_POS pans by it (sign not proven: =1 / =2 mirrored).
// KYTY_AUDIO_EXACTO=1 (default off): main-port gains + real channel count. KYTY_AUDIO_POS=1|2 (default off).
#include <cstdio>
#include <cstdlib>

namespace AudioExacto {

inline bool Enabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_AUDIO_EXACTO");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

// 0 off, 1 pan by position (+x right), 2 mirrored (+x left).
inline int PosMode() {
	static const int m = [] {
		const char* v = std::getenv("KYTY_AUDIO_POS");
		return v == nullptr ? 0 : (v[0] == '1' ? 1 : (v[0] == '2' ? 2 : 0));
	}();
	return m;
}

} // namespace AudioExacto
