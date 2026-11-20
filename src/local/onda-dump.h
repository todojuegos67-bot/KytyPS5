#pragma once
// (onda4) KYTY_ONDA_DUMP=1: F12 dumps to disk, during KYTY_ONDA_DUMP_FRAMES guest frames (default 40) one out of every
// KYTY_ONDA_DUMP_STEP (default 2), the images of the passes the floor "onda" goes through, as raw texel files plus an
// index (index.jsonl) that tools/onda_dump.py turns into PNG strips / GIFs on Linux:
//   - every image bound to the game's TAA (a44e580f4f0cfb2b / 934a6e1197b99148) BEFORE it runs (colour before the TAA,
//     depth, motion vectors, history) and the images it writes AFTER (colour after the TAA, the new history);
//   - every image bound to the game's IBL (9fb5cc274b5e4f82) before (G-buffer array layers, depth, SSR result
//     0x2ac920000 / 0x2ac4a0000, SSR tile mask 0x2acda0000) and what it writes after (IBL specular 0x2886a0000);
//   - KYTY_ONDA_DUMP_HASHES (more CS hashes, comma separated, hex) the same way;
//   - KYTY_ONDA_DUMP_ADDRS (guest addresses, default the four above): after ANY dispatch that writes one of them (its
//     producer: the SSR passes, the tile classification), that image.
// The copies are recorded on the GPU in the frame's own command buffer (image -> host-visible staging buffer, no wait);
// the GPU thread hands a copy to a writer thread only once the scheduler's tick for it is done (IsFree) and at least
// 8 frames have passed; the writer writes the file and frees the staging buffer. Nothing waits for the GPU. Staging held
// at once is capped (KYTY_ONDA_DUMP_MB, default 2048): over it a copy is skipped and counted, never blocked.
// Output: KYTY_ONDA_DUMP_DIR (default "capturas/onda" next to the exe)/sesionN/. F12 also records the first frame's
// call list (frame-capture jsonl) into sesionN/llamadas, to map slots to passes.
#include "local/frame-capture.h"
#include "loader/demonsSoulsWarp.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace OndaDump {

struct Config {
	bool                  on     = false;
	uint32_t              frames = 40, step = 2;
	uint64_t              budget = 2048ull << 20u;
	std::string           base   = "capturas/onda";
	std::vector<uint64_t> hashes; // besides the TAA
	std::vector<uint64_t> addrs;
};

inline std::vector<uint64_t> ParseHexList(const char* text) {
	std::vector<uint64_t> out;
	if (text == nullptr) return out;
	const std::string s(text);
	for (size_t begin = 0; begin < s.size();) {
		const auto end = std::min(s.find(',', begin), s.size());
		if (end > begin) {
			const uint64_t v = std::strtoull(s.substr(begin, end - begin).c_str(), nullptr, 16);
			if (v != 0) out.push_back(v);
		}
		begin = end + 1;
	}
	return out;
}

inline const Config& Cfg() {
	static const Config cfg = [] {
		Config c;
		const char* v = std::getenv("KYTY_ONDA_DUMP");
		c.on          = v != nullptr && v[0] == '1';
		if (const char* f = std::getenv("KYTY_ONDA_DUMP_FRAMES"); f != nullptr && std::atoi(f) > 0)
			c.frames = static_cast<uint32_t>(std::min(std::atoi(f), 600));
		if (const char* f = std::getenv("KYTY_ONDA_DUMP_STEP"); f != nullptr && std::atoi(f) > 0)
			c.step = static_cast<uint32_t>(std::min(std::atoi(f), 60));
		if (const char* f = std::getenv("KYTY_ONDA_DUMP_MB"); f != nullptr && std::atoi(f) > 0)
			c.budget = static_cast<uint64_t>(std::atoi(f)) << 20u;
		if (const char* d = std::getenv("KYTY_ONDA_DUMP_DIR"); d != nullptr && *d != 0) c.base = d;
		c.hashes = ParseHexList("9fb5cc274b5e4f82");
		for (const auto h: ParseHexList(std::getenv("KYTY_ONDA_DUMP_HASHES"))) c.hashes.push_back(h);
		const char* a = std::getenv("KYTY_ONDA_DUMP_ADDRS");
		c.addrs       = ParseHexList(a != nullptr ? a : "2ac920000,2ac4a0000,2acda0000,2886a0000");
		if (c.on) {
			std::printf("[onda4] volcado on: F12 graba %u frames (1 de cada %u) a %s, tope %llu MB\n", c.frames, c.step,
			            c.base.c_str(), static_cast<unsigned long long>(c.budget >> 20u));
			std::fflush(stdout);
		}
		return c;
	}();
	return cfg;
}

[[nodiscard]] inline bool Wanted() { return Cfg().on; }

// Window thread -> Kyty.Gpu.
inline std::atomic<int>  g_armed {0};     // a session number waiting for the next dispatch
inline std::atomic<bool> g_recording {false};
inline std::mutex        g_dir_mutex;
inline std::string       g_dir;           // the session's folder (set before g_armed)

// While true the batched dispatch path steps aside (every dispatch takes DispatchDirect's normal path).
[[nodiscard]] inline bool Recording() {
	return g_recording.load(std::memory_order_relaxed) || g_armed.load(std::memory_order_relaxed) != 0;
}

// F12 (window thread): arms a session; refused while one runs.
inline void Arm() {
	static std::atomic<int> sessions {0};
	if (Recording()) {
		Loader::DemonsSoulsWarp::ShowNotice("Volcado de la onda en curso: espera a que acabe", 2);
		return;
	}
	const int n = ++sessions;
	const std::string dir = Cfg().base + "/sesion" + std::to_string(n);
	{
		const std::lock_guard lock(g_dir_mutex);
		g_dir = dir;
	}
	// The first frame's call list (slots -> passes), when no frame capture runs.
	if (!FrameCapture::g_active.load() && FrameCapture::g_pending.load() <= 0) {
		{
			const std::lock_guard lock(FrameCapture::g_mutex);
			FrameCapture::g_dir = dir + "/llamadas";
		}
		FrameCapture::g_data_hashes = ParseHexList("a44e580f4f0cfb2b,9fb5cc274b5e4f82");
		FrameCapture::g_data_counts.assign(FrameCapture::g_data_hashes.size(), 0u);
		FrameCapture::g_done.store(0);
		FrameCapture::g_pending.store(1, std::memory_order_release);
	}
	g_armed.store(n, std::memory_order_release);
	std::printf("[onda4] sesion %d armada -> %s\n", n, dir.c_str());
	std::fflush(stdout);
	Loader::DemonsSoulsWarp::ShowNotice("Volcado onda " + std::to_string(n) + ": grabando...", 2);
}

} // namespace OndaDump
