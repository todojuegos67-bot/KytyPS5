#ifndef KYTY_GRAPHICS_DRAWDIFF_H_
#define KYTY_GRAPHICS_DRAWDIFF_H_

// KYTY_DRAW_DIFF=1 (live, default 0): finds what flickers or pops. Standing still, consecutive
// frames issue the same draws and dispatches; this records, for 8 frames in a row, every draw
// (vertex-stage hash, pixel hash, index count, instance count, indirect) and dispatch (compute
// hash, group count) that reached the GPU, and prints the keys whose count changes between those
// frames ("Draw diff"), plus the draws the emulator dropped and why ("Draw skips"). A key missing
// in some frames is a candidate for something that appears and disappears.
// Thread_Gpu only (draws, dispatches and flips are recorded there); costs one relaxed load when off.

#include "common/liveSwitches.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace Libs::Graphics::DrawDiff {

inline constexpr int kFrames = 8;

// Guest flips so far (always counted) and the shader whose resources are being looked up on this
// thread (set by the program cache, used by the fresh-read report).
inline std::atomic<uint64_t> g_frame {0};
inline thread_local uint64_t t_shader = 0;
inline uint64_t              Frame() { return g_frame.load(std::memory_order_relaxed); }

struct Key {
	uint64_t a        = 0; // vertex-stage (or compute) hash
	uint64_t b        = 0; // pixel hash
	uint32_t count    = 0; // index/vertex count or group count
	uint32_t instance = 0;
	uint32_t kind     = 0; // 0 draw, 1 indirect draw, 2 dispatch, 3 indirect dispatch
	bool     operator<(const Key& o) const {
		return std::tie(a, b, count, instance, kind) < std::tie(o.a, o.b, o.count, o.instance, o.kind);
	}
};

struct State {
	std::mutex                                       mutex;
	int                                              frame = -1; // -1 idle, 0..kFrames-1 recording
	int                                              cooldown = 0;
	std::map<Key, std::array<uint32_t, kFrames>>     keys;
	std::map<std::pair<uint64_t, std::string>, uint64_t> skips;
	std::map<uint64_t, uint64_t>                         gpu_freed; // address -> frame freed with GPU contents
	std::map<std::string, uint64_t>                      fresh;     // fresh-read description -> count
	std::map<uint64_t, std::string>                      insert_note; // address -> what overlapped it at creation
	std::map<std::string, uint64_t>                      writeback;   // writeback result -> count
	std::map<std::string, uint64_t>                      history;     // history-read description -> count
};

inline State& Get() {
	static State s;
	return s;
}

inline bool Enabled() {
	static auto& on = Common::LiveSwitches::Get("KYTY_DRAW_DIFF", 0);
	return on.load(std::memory_order_relaxed) != 0;
}

inline void Note(uint64_t a, uint64_t b, uint32_t count, uint32_t instance, uint32_t kind) {
	if (!Enabled()) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	if (s.frame < 0) {
		return;
	}
	s.keys[Key {a, b, count, instance, kind}][static_cast<size_t>(s.frame)]++;
}

inline void NoteSkip(uint64_t hash, const char* reason) {
	if (!Enabled()) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	s.skips[{hash, reason}]++;
}

inline const char* KindName(uint32_t kind) {
	switch (kind) {
		case 0: return "draw";
		case 1: return "draw-indirect";
		case 2: return "dispatch";
		default: return "dispatch-indirect";
	}
}

// An image freed while it held GPU-written contents.
inline void NoteGpuFree(uint64_t address) {
	if (!Enabled()) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	s.gpu_freed[address] = Frame();
}

// What FindImage found over a range when it created a new image there (last creation per address).
inline void NoteInsert(uint64_t address, std::string note) {
	if (!Enabled()) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	s.insert_note[address] = std::move(note);
}

inline void NoteWriteback(uint64_t address, const char* result) {
	if (!Enabled()) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	char            buf[96];
	std::snprintf(buf, sizeof(buf), "0x%010llx %s", static_cast<unsigned long long>(address), result);
	s.writeback[buf]++;
}

// A sampled read of an image the GPU last wrote in the previous frame: a temporal history (TAA,
// denoiser, reprojection). Lists which shader reads which image and which shader wrote it.
inline void NoteHistoryRead(uint64_t reader, uint64_t address, const char* format, uint32_t width,
                            uint32_t height, uint64_t writer, uint64_t age) {
	if (!Enabled()) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	char            buf[256];
	std::snprintf(buf, sizeof(buf), "shader %016llx reads 0x%010llx %ux%u %s written %llu frame(s) before by %016llx",
	              static_cast<unsigned long long>(reader), static_cast<unsigned long long>(address), width,
	              height, format, static_cast<unsigned long long>(age),
	              static_cast<unsigned long long>(writer));
	s.history[buf]++;
}

// A sampled read of an image created this frame that the GPU has not written yet: whatever it
// holds came from guest memory (or nothing). If the game expects last frame's contents there
// (a temporal history), this is the flicker.
inline void NoteFreshRead(uint64_t address, const char* format, uint32_t width, uint32_t height,
                          bool buffer_modified, uint32_t created_line = 0) {
	if (!Enabled()) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	const auto      freed  = s.gpu_freed.find(address);
	const bool      lost   = freed != s.gpu_freed.end() && Frame() - freed->second <= 2;
	const auto      note   = s.insert_note.find(address);
	char            buf[768];
	std::snprintf(buf, sizeof(buf), "shader %016llx reads 0x%010llx %ux%u %s%s%s [created at line %u; overlapped: %s]",
	              static_cast<unsigned long long>(t_shader), static_cast<unsigned long long>(address),
	              width, height, format, buffer_modified ? " (from GPU buffer copy)" : "",
	              lost ? " <- GPU contents dropped here 0-2 frames ago" : "", created_line,
	              note != s.insert_note.end() ? note->second.c_str() : "?");
	s.fresh[buf]++;
}

// Called once per guest flip.
inline void EndFrame() {
	g_frame.fetch_add(1, std::memory_order_relaxed);
	if (!Enabled()) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	if (s.frame < 0) {
		// Start a capture every ~5 s (about 100 flips at 20 fps, fewer when slower).
		if (s.cooldown-- > 0) {
			return;
		}
		s.keys.clear();
		s.frame = 0;
		return;
	}
	if (++s.frame < kFrames) {
		return;
	}
	s.frame    = -1;
	s.cooldown = 100;
	std::vector<std::pair<Key, std::array<uint32_t, kFrames>>> changing;
	size_t                                                     stable = 0;
	for (const auto& [key, counts]: s.keys) {
		const auto [lo, hi] = std::minmax_element(counts.begin(), counts.end());
		if (*lo == *hi) {
			stable++;
		} else {
			changing.emplace_back(key, counts);
		}
	}
	// Keys absent in some frames first, then by how much they vary.
	std::sort(changing.begin(), changing.end(), [](const auto& x, const auto& y) {
		const auto zx = std::count(x.second.begin(), x.second.end(), 0u);
		const auto zy = std::count(y.second.begin(), y.second.end(), 0u);
		return zx != zy ? zx > zy : x.first < y.first;
	});
	::printf("Draw diff (%d frames): %zu keys stable, %zu change\n", kFrames, stable, changing.size());
	for (size_t i = 0; i < changing.size() && i < 40; i++) {
		const auto& [key, counts] = changing[i];
		::printf("  %-17s a=%016" PRIx64 " b=%016" PRIx64 " n=%u inst=%u :", KindName(key.kind), key.a,
		         key.b, key.count, key.instance);
		for (const auto c: counts) {
			::printf(" %u", c);
		}
		::printf("\n");
	}
	if (!s.fresh.empty()) {
		std::vector<std::pair<uint64_t, std::string>> rows;
		for (const auto& [k, n]: s.fresh) {
			rows.emplace_back(n, k);
		}
		std::sort(rows.rbegin(), rows.rend());
		::printf("Fresh image reads (since last report):\n");
		for (size_t i = 0; i < rows.size() && i < 25; i++) {
			::printf("  %6" PRIu64 "  %s\n", rows[i].first, rows[i].second.c_str());
		}
		s.fresh.clear();
	}
	if (!s.history.empty()) {
		std::vector<std::pair<uint64_t, std::string>> rows;
		for (const auto& [k, n]: s.history) {
			rows.emplace_back(n, k);
		}
		std::sort(rows.rbegin(), rows.rend());
		::printf("History reads (sampled images written in an earlier frame, since last report):\n");
		for (size_t i = 0; i < rows.size() && i < 40; i++) {
			::printf("  %6" PRIu64 "  %s\n", rows[i].first, rows[i].second.c_str());
		}
		s.history.clear();
	}
	if (!s.writeback.empty()) {
		::printf("Image writebacks to the buffer cache (since last report):\n");
		for (const auto& [k, n]: s.writeback) {
			::printf("  %6" PRIu64 "  %s\n", n, k.c_str());
		}
		s.writeback.clear();
	}
	if (s.insert_note.size() > 4096) {
		s.insert_note.clear();
	}
	if (s.gpu_freed.size() > 4096) {
		s.gpu_freed.clear();
	}
	if (!s.skips.empty()) {
		::printf("Draw skips (since last report):\n");
		for (const auto& [k, n]: s.skips) {
			::printf("  %8" PRIu64 "  %016" PRIx64 "  %s\n", n, k.first, k.second.c_str());
		}
		s.skips.clear();
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics::DrawDiff

#endif
