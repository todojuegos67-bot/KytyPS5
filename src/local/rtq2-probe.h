#pragma once
// (rtq2) KYTY_RTQ2_PROBE=1: F12 records ONE whole frame with the existing frame capture (frame-capture.h: every draw and
// dispatch, its shaders' hashes, bound targets / textures / buffers with guest address, format and size; the batched
// draw and table-dispatch paths step aside for that frame so every call is seen), into
// KYTY_RTQ2_PROBE_DIR (default "rtq2-capturas" next to the exe)/capN/frame-*.jsonl, with the constants of the game's
// IBL compute (9fb5cc274b5e4f82) and of KYTY_RTQ2_PROBE_HASHES. Non-blocking: the key only arms it; the next guest flip
// starts it (FrameCapture::OnFlip). For the floor / water / fog / metal question of rtq2: which pass writes the SSR
// (0x2ac920000 in the 20261119-053939 log), which pixel shaders write the G-buffer array (and the material class bits
// they write), the fog chain. Rendering unchanged apart from that one frame taking the normal paths.
#include "local/frame-capture.h"
#include "loader/demonsSoulsWarp.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace Rtq2Probe {

[[nodiscard]] inline bool Wanted() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RTQ2_PROBE");
		const bool  r = v != nullptr && v[0] == '1';
		if (r) {
			std::printf("[rtq2] sonda on: F12 graba un frame completo (frame-capture) por pulsacion\n");
			std::fflush(stdout);
		}
		return r;
	}();
	return on;
}

inline void Trigger() {
	static std::atomic<int> count {0};
	if (FrameCapture::g_active.load() || FrameCapture::g_pending.load() > 0) {
		Loader::DemonsSoulsWarp::ShowNotice("Captura en curso: espera un momento y pulsa F12 otra vez", 2);
		return;
	}
	const int   n    = ++count;
	const char* base = std::getenv("KYTY_RTQ2_PROBE_DIR");
	std::string dir  = std::string(base != nullptr && *base != 0 ? base : "rtq2-capturas") + "/cap" + std::to_string(n);
	std::string hashes = "9fb5cc274b5e4f82";
	if (const char* more = std::getenv("KYTY_RTQ2_PROBE_HASHES"); more != nullptr && *more != 0) hashes += std::string(",") + more;
	{
		const std::lock_guard lock(FrameCapture::g_mutex);
		FrameCapture::g_dir = dir;
	}
	FrameCapture::g_data_hashes.clear();
	for (size_t begin = 0; begin < hashes.size();) {
		const auto end = std::min(hashes.find(',', begin), hashes.size());
		if (end > begin) FrameCapture::g_data_hashes.push_back(std::strtoull(hashes.substr(begin, end - begin).c_str(), nullptr, 16));
		begin = end + 1;
	}
	FrameCapture::g_data_counts.assign(FrameCapture::g_data_hashes.size(), 0u);
	FrameCapture::g_done.store(0);
	FrameCapture::g_pending.store(1, std::memory_order_release);
	std::printf("[rtq2] captura %d armada -> %s (un frame)\n", n, dir.c_str());
	std::fflush(stdout);
	Loader::DemonsSoulsWarp::ShowNotice("Captura " + std::to_string(n) + " hecha (F12)", 3);
}

} // namespace Rtq2Probe
