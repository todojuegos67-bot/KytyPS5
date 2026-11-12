#pragma once
// Live probe `skipcs <guest code address hex>`: dispatches of that compute program, or draws of that pixel shader, are not run (toggles; `skipcs 0`
// clears all). For finding which pass draws an artifact; the image keeps its previous contents. Up to 8 programs.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

namespace SkipShader {
inline std::atomic<uint64_t> g_cs[8] {};
inline std::atomic<bool>     g_any {false};

inline bool Skipped(uint64_t address) noexcept {
	if (address == 0 || !g_any.load(std::memory_order_relaxed)) return false;
	for (auto& slot : g_cs)
		if (slot.load(std::memory_order_relaxed) == address) return true;
	return false;
}

inline void Toggle(uint64_t address) noexcept {
	if (address == 0) {
		for (auto& slot : g_cs) slot.store(0);
		g_any.store(false);
		return;
	}
	for (auto& slot : g_cs)
		if (slot.load() == address) {
			slot.store(0);
			return;
		}
	for (auto& slot : g_cs)
		if (slot.load() == 0) {
			slot.store(address);
			g_any.store(true);
			return;
		}
}

// Paint mode (debug, off by default): finding which pass draws an artifact by leaving passes out, from settings.ini.
//  KYTY_PAINT=<class>          draws of that class are not run: 1 no color target (depth / shadow), 2 color target 3000+
//                              wide (4K / HUD canvas), 3 1700-2999 (full-res scene), 4 900-1699 (half), 5 400-899
//                              (quarter), 6 narrower.
//  KYTY_PAINT_SKIP_PS=a,b      draws of these pixel shaders (guest code addresses, hex, as KYTY_PAINT_LIST prints them)
//  KYTY_PAINT_SKIP_CS=a,b      dispatches of these compute shaders (same; with SKIP_PS up to 8 in all, as `skipcs`)
//  KYTY_PAINT_SKIP_RT=WxH,...  draws into a color target of that size (KYTY_PAINT_LIST's sizes)
//  KYTY_PAINT_LIST=1           once a second: the draws (PS, class, target size / format) and dispatches (CS), counted.
// Any of them sends every draw the normal path (as `skipcs`). A pass left out keeps its target's previous contents.
namespace Paint {
struct Config {
	bool                                    active = false;
	bool                                    list   = false;
	uint32_t                                klass  = 0;
	std::vector<std::pair<uint32_t, uint32_t>> sizes;
};

inline uint32_t ClassOf(uint32_t width, bool has_color) noexcept {
	if (!has_color || width == 0) return 1;
	if (width >= 3000) return 2;
	if (width >= 1700) return 3;
	if (width >= 900) return 4;
	if (width >= 400) return 5;
	return 6;
}

inline void ParseAddresses(const char* name) {
	const char* text = std::getenv(name);
	if (text == nullptr) return;
	const char* p = text;
	while (*p != '\0') {
		char*          end     = nullptr;
		const uint64_t address = std::strtoull(p, &end, 16);
		if (end == p) {
			++p;
			continue;
		}
		if (address != 0 && !Skipped(address)) Toggle(address);
		std::printf("Paint: %s 0x%llx left out\n", name, static_cast<unsigned long long>(address));
		p = end;
	}
}

inline Config Load() {
	Config      c;
	const char* klass = std::getenv("KYTY_PAINT");
	if (klass != nullptr) c.klass = static_cast<uint32_t>(std::strtoul(klass, nullptr, 10));
	if (c.klass > 6) c.klass = 0;
	const char* list = std::getenv("KYTY_PAINT_LIST");
	c.list = list != nullptr && list[0] != '\0' && list[0] != '0';
	if (const char* rt = std::getenv("KYTY_PAINT_SKIP_RT"); rt != nullptr) {
		const char* p = rt;
		while (*p != '\0') {
			char*               end = nullptr;
			const unsigned long w   = std::strtoul(p, &end, 10);
			if (end == p) {
				++p;
				continue;
			}
			p = end;
			if (*p != 'x' && *p != 'X') continue;
			++p;
			const unsigned long h = std::strtoul(p, &end, 10);
			if (end == p) continue;
			p = end;
			c.sizes.emplace_back(static_cast<uint32_t>(w), static_cast<uint32_t>(h));
		}
	}
	ParseAddresses("KYTY_PAINT_SKIP_PS");
	ParseAddresses("KYTY_PAINT_SKIP_CS");
	c.active = c.list || c.klass != 0 || !c.sizes.empty() || g_any.load();
	if (c.active) {
		g_any.store(true); // every draw on the normal path, where it is seen
		std::printf("Paint: active: class %u left out (0 none), %zu target sizes left out", c.klass, c.sizes.size());
		for (const auto& size: c.sizes) std::printf(" %ux%u", size.first, size.second);
		std::printf(", shaders left out via KYTY_PAINT_SKIP_PS/CS above, list %s\n", c.list ? "on" : "off");
		std::fflush(stdout);
	}
	return c;
}

inline const Config& Get() {
	static const Config config = Load();
	return config;
}
// (Read at the first dispatch/draw: settings.ini `extra` is in the environment by then, not at static init.)

// Key: kind ('D' draw, 'C' dispatch), shader address, width, height, format, channel type.
using Key = std::tuple<char, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t>;
struct Table {
	std::mutex                            mutex;
	std::map<Key, uint32_t>               rows;
	std::chrono::steady_clock::time_point last {};
	uint32_t                              skipped = 0;
};
inline Table& GetTable() {
	static Table table;
	return table;
}

inline void Note(char kind, uint64_t address, uint32_t width, uint32_t height, uint32_t format, uint32_t type, bool skipped) {
	auto&                       t = GetTable();
	std::lock_guard<std::mutex> lock(t.mutex);
	if (t.rows.size() < 4096) ++t.rows[Key {kind, address, width, height, format, type}];
	if (skipped) ++t.skipped;
	const auto now = std::chrono::steady_clock::now();
	if (now - t.last < std::chrono::seconds(1)) return;
	if (t.last != std::chrono::steady_clock::time_point {}) {
		std::vector<std::pair<uint32_t, Key>> sorted;
		sorted.reserve(t.rows.size());
		for (const auto& [key, count]: t.rows) sorted.emplace_back(count, key);
		std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
		std::printf("[paint] 1 s: %zu rows, %u left out (kind class shader count target WxH format/type)\n", sorted.size(),
		            t.skipped);
		size_t shown = 0;
		for (const auto& [count, key]: sorted) {
			if (++shown > 80) break;
			const char kind = std::get<0>(key);
			if (kind == 'D')
				std::printf("[paint]   PS class %u 0x%016llx x%u  %ux%u fmt %u/%u\n",
				            ClassOf(std::get<2>(key), std::get<2>(key) != 0),
				            static_cast<unsigned long long>(std::get<1>(key)), count, std::get<2>(key), std::get<3>(key),
				            std::get<4>(key), std::get<5>(key));
			else
				std::printf("[paint]   CS         0x%016llx x%u  groups %ux%ux%u\n",
				            static_cast<unsigned long long>(std::get<1>(key)), count, std::get<2>(key), std::get<3>(key),
				            std::get<4>(key));
		}
		std::fflush(stdout);
	}
	t.rows.clear();
	t.skipped = 0;
	t.last    = now;
}

// A draw (pixel shader `ps`, color target 0 `width` x `height`, 0 without one): true when left out.
inline bool SkipDraw(uint64_t ps, bool has_color, uint32_t width, uint32_t height, uint32_t format, uint32_t type) {
	const auto& c = Get();
	if (!c.active) return false;
	if (!has_color) width = height = format = type = 0;
	bool skip = c.klass != 0 && ClassOf(width, has_color) == c.klass;
	if (!skip && has_color)
		for (const auto& size: c.sizes)
			if (size.first == width && size.second == height) skip = true;
	if (!skip) skip = Skipped(ps);
	if (c.list) Note('D', ps, width, height, format, type, skip);
	return skip;
}

// A dispatch (KYTY_PAINT_LIST's count; KYTY_PAINT_SKIP_CS is `Skipped`).
inline void NoteDispatch(uint64_t cs, uint32_t x, uint32_t y, uint32_t z) {
	const auto& c = Get();
	if (c.list) Note('C', cs, x, y, z, 0, Skipped(cs));
}
} // namespace Paint
} // namespace SkipShader
