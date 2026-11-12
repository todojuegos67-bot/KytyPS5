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
#include <string>
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
	bool                                    live   = false; // KYTY_PAINT_LIVE=1: hotkeys choose what is left out
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

inline std::atomic<bool> g_live {false}; // set at Load (first draw); the hotkeys do nothing before

inline Config Load() {
	Config      c;
	if (const char* live = std::getenv("KYTY_PAINT_LIVE"); live != nullptr && live[0] != '\0' && live[0] != '0') c.live = true;
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
	c.active = c.live || c.list || c.klass != 0 || !c.sizes.empty() || g_any.load();
	if (c.active) {
		g_any.store(true); // every draw on the normal path, where it is seen
		std::printf("Paint: active: class %u left out (0 none), %zu target sizes left out", c.klass, c.sizes.size());
		for (const auto& size: c.sizes) std::printf(" %ux%u", size.first, size.second);
		std::printf(", shaders left out via KYTY_PAINT_SKIP_PS/CS above, list %s\n", c.list ? "on" : "off");
		std::fflush(stdout);
	}
	if (c.live) {
		std::printf("[paint-live] on: F7 next group, F6 previous, F8 off, F9 first/second half, F10 narrow into the half left out\n");
		std::fflush(stdout);
		g_live.store(true);
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

// KYTY_PAINT_LIVE: in-game bisection. Counts of the last full second (CS addresses; PS addresses of draws with a color
// target, any size: the scene's targets are not always exactly 1280x720, which left that list empty) are snapshot when a
// group is chosen from off (F8 off retakes it). Groups, in order:
//  0 off, 1 blended draws (color target 0 blending), 2 depth test without depth write, 3 draws into 3840x2160,
//  4 draws into 1280x720, 5 draws with alpha-to-mask (DB_ALPHA_TO_MASK on, not disabled by the shader), 6 draws whose
//  PS may discard (DB_SHADER_CONTROL.KILL_ENABLE), 7 CS range, 8 PS(color) range, then each CS alone (by count), then
//  each PS alone.
// A range group leaves out list[lo, hi); F9 halves it (first half, second half, whole), F10 makes the half left out
// the new whole. Only with live on: a mutex per draw/dispatch.
namespace Live {
enum Kind : uint32_t { Off = 0, Blend, DepthNoWrite, Rt4k, Rt720, AlphaToMask, PsKill, CsRange, PsRange, CsOne, PsOne };
inline constexpr uint32_t FixedGroups = 9; // 0..6 the kinds above, 7 CS range, 8 PS range
struct State {
	std::mutex                            mutex;
	std::map<uint64_t, uint32_t>          cs_cur, ps_cur, cs_prev, ps_prev;
	std::chrono::steady_clock::time_point last {};
	std::vector<uint64_t>                 cs_list, ps_list; // snapshot, by count
	uint32_t                              group = 0;        // index in the order above
	Kind                                  kind  = Off;
	size_t                                lo = 0, hi = 0;   // the whole of a range (or the one entry)
	uint32_t                              half = 0;         // 0 whole, 1 first half, 2 second half
	std::vector<uint64_t>                 out;              // sorted addresses left out (CS / PS kinds)
	std::map<uint64_t, uint32_t>          ps_rt;            // PS -> its last color target, (width << 16) | height
};
inline State& GetState() {
	static State s;
	return s;
}

inline void Roll(State& s) { // (locked)
	const auto now = std::chrono::steady_clock::now();
	if (now - s.last < std::chrono::seconds(1)) return;
	s.cs_prev.swap(s.cs_cur);
	s.ps_prev.swap(s.ps_cur);
	s.cs_cur.clear();
	s.ps_cur.clear();
	s.last = now;
}

inline std::vector<uint64_t> Sorted(const std::map<uint64_t, uint32_t>& m) {
	std::vector<std::pair<uint32_t, uint64_t>> v;
	for (const auto& [a, n]: m) v.emplace_back(n, a);
	std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
	std::vector<uint64_t> r;
	r.reserve(v.size());
	for (const auto& e: v) r.push_back(e.second);
	return r;
}

inline uint32_t GroupCount(const State& s) {
	return FixedGroups + static_cast<uint32_t>(s.cs_list.size() + s.ps_list.size());
}

inline void LeftOutRange(const State& s, size_t* lo, size_t* hi) {
	*lo = s.lo;
	*hi = s.hi;
	if (s.half == 0 || s.hi - s.lo < 2) return;
	const size_t mid = s.lo + (s.hi - s.lo) / 2;
	if (s.half == 1) *hi = mid;
	else *lo = mid;
}

inline void Rebuild(State& s) { // (locked) out from kind/lo/hi/half
	s.out.clear();
	const std::vector<uint64_t>* list = (s.kind == CsRange || s.kind == CsOne) ? &s.cs_list
	                                    : (s.kind == PsRange || s.kind == PsOne) ? &s.ps_list
	                                                                             : nullptr;
	if (list == nullptr) return;
	size_t lo = 0, hi = 0;
	LeftOutRange(s, &lo, &hi);
	for (size_t i = lo; i < hi && i < list->size(); ++i) s.out.push_back((*list)[i]);
	std::sort(s.out.begin(), s.out.end());
}

inline void SelectGroup(State& s, uint32_t g) { // (locked)
	s.group = g;
	s.half  = 0;
	s.lo = s.hi = 0;
	const uint32_t ncs = static_cast<uint32_t>(s.cs_list.size());
	if (g <= 6) s.kind = static_cast<Kind>(g);
	else if (g == 7) { s.kind = CsRange; s.hi = s.cs_list.size(); }
	else if (g == 8) { s.kind = PsRange; s.hi = s.ps_list.size(); }
	else if (g - FixedGroups < ncs) { s.kind = CsOne; s.lo = g - FixedGroups; s.hi = s.lo + 1; }
	else { s.kind = PsOne; s.lo = g - FixedGroups - ncs; s.hi = s.lo + 1; }
	Rebuild(s);
}

inline std::string Describe(const State& s) { // (locked)
	char buf[512];
	const uint32_t total = GroupCount(s);
	const char*    name  = "";
	switch (s.kind) {
		case Off: name = "nothing (off)"; break;
		case Blend: name = "blended draws"; break;
		case DepthNoWrite: name = "draws with depth test, no depth write"; break;
		case Rt4k: name = "draws into 3840x2160"; break;
		case Rt720: name = "draws into 1280x720"; break;
		case AlphaToMask: name = "draws with alpha-to-mask enabled"; break;
		case PsKill: name = "draws whose PS uses discard/kill"; break;
		case CsRange: name = "CS list range"; break;
		case PsRange: name = "PS(color target) list range"; break;
		case CsOne: name = "CS"; break;
		case PsOne: name = "PS(color target)"; break;
	}
	size_t lo = 0, hi = 0;
	LeftOutRange(s, &lo, &hi);
	if (s.kind == PsOne) {
		const uint64_t ps = s.out.empty() ? 0ull : s.out[0];
		const auto     rt = s.ps_rt.find(ps);
		const uint32_t wh = rt != s.ps_rt.end() ? rt->second : 0u;
		std::snprintf(buf, sizeof(buf), "group %u/%u: %s #%zu 0x%016llx (target %ux%u)", s.group, total - 1, name, lo,
		              static_cast<unsigned long long>(ps), wh >> 16u, wh & 0xffffu);
	} else if (s.kind == CsOne)
		std::snprintf(buf, sizeof(buf), "group %u/%u: %s #%zu 0x%016llx", s.group, total - 1, name, lo,
		              static_cast<unsigned long long>(s.out.empty() ? 0ull : s.out[0]));
	else if (s.kind == CsRange || s.kind == PsRange)
		std::snprintf(buf, sizeof(buf), "group %u/%u: %s [%zu..%zu) of %zu (%zu left out)%s", s.group, total - 1, name, lo, hi,
		              s.kind == CsRange ? s.cs_list.size() : s.ps_list.size(), s.out.size(),
		              s.half == 1 ? " first half" : s.half == 2 ? " second half" : "");
	else std::snprintf(buf, sizeof(buf), "group %u/%u: %s", s.group, total - 1, name);
	return buf;
}

// A hotkey (SDL keycode as F6..F10 = 6..10). Returns the label (empty: not a live key / live off).
inline std::string Key(int f) {
	if (!g_live.load(std::memory_order_relaxed) || f < 6 || f > 10) return {};
	auto&                       s = GetState();
	std::lock_guard<std::mutex> lock(s.mutex);
	if (f == 8) {
		SelectGroup(s, 0);
	} else if (f == 6 || f == 7) {
		if (s.group == 0) {
			s.cs_list = Sorted(s.cs_prev.empty() ? s.cs_cur : s.cs_prev);
			s.ps_list = Sorted(s.ps_prev.empty() ? s.ps_cur : s.ps_prev);
		}
		const uint32_t total = GroupCount(s);
		SelectGroup(s, f == 7 ? (s.group + 1) % total : (s.group + total - 1) % total);
	} else if (f == 9) {
		if ((s.kind == CsRange || s.kind == PsRange) && s.hi - s.lo >= 2) {
			s.half = (s.half + 1) % 3;
			Rebuild(s);
		}
	} else if (f == 10) {
		if ((s.kind == CsRange || s.kind == PsRange) && s.half != 0) {
			size_t lo = 0, hi = 0;
			LeftOutRange(s, &lo, &hi);
			s.lo   = lo;
			s.hi   = hi;
			s.half = 0;
			Rebuild(s);
		}
	}
	std::string label = Describe(s);
	std::printf("[paint-live] now leaving out: %s\n", label.c_str());
	if (!s.out.empty()) {
		std::printf("[paint-live]   addresses:");
		for (size_t i = 0; i < s.out.size() && i < 64; ++i) std::printf(" %llx", static_cast<unsigned long long>(s.out[i]));
		std::printf(s.out.size() > 64 ? " ...\n" : "\n");
	}
	std::fflush(stdout);
	return label;
}

inline bool Has(const State& s, uint64_t a) { return std::binary_search(s.out.begin(), s.out.end(), a); }

inline bool Draw(uint64_t ps, bool has_color, uint32_t width, uint32_t height, bool blend, bool ztest, bool zwrite,
                 bool alpha_to_mask, bool kill) {
	auto&                       s = GetState();
	std::lock_guard<std::mutex> lock(s.mutex);
	Roll(s);
	const bool is720 = has_color && width == 1280 && height == 720;
	if (has_color && ps != 0 && s.ps_cur.size() < 4096) {
		++s.ps_cur[ps];
		if (s.ps_rt.size() < 8192 || s.ps_rt.count(ps) != 0)
			s.ps_rt[ps] = ((width & 0xffffu) << 16u) | (height & 0xffffu);
	}
	switch (s.kind) {
		case Off: return false;
		case Blend: return has_color && blend;
		case DepthNoWrite: return ztest && !zwrite;
		case Rt4k: return has_color && width == 3840 && height == 2160;
		case Rt720: return is720;
		case AlphaToMask: return alpha_to_mask;
		case PsKill: return kill;
		case PsRange:
		case PsOne: return has_color && Has(s, ps);
		default: return false;
	}
}

inline bool Dispatch(uint64_t cs) {
	auto&                       s = GetState();
	std::lock_guard<std::mutex> lock(s.mutex);
	Roll(s);
	if (cs != 0 && s.cs_cur.size() < 4096) ++s.cs_cur[cs];
	return (s.kind == CsRange || s.kind == CsOne) && Has(s, cs);
}
} // namespace Live

// A draw (pixel shader `ps`, color target 0 `width` x `height`, 0 without one): true when left out.
inline bool SkipDraw(uint64_t ps, bool has_color, uint32_t width, uint32_t height, uint32_t format, uint32_t type,
                     bool blend = false, bool ztest = false, bool zwrite = false, bool alpha_to_mask = false,
                     bool kill = false) {
	const auto& c = Get();
	if (!c.active) return false;
	if (!has_color) width = height = format = type = 0;
	bool skip = c.klass != 0 && ClassOf(width, has_color) == c.klass;
	if (c.live && Live::Draw(ps, has_color, width, height, blend, ztest, zwrite, alpha_to_mask, kill)) skip = true;
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

// KYTY_PAINT_LIVE: true when this dispatch is left out by the live group (counts it too).
inline bool SkipDispatchLive(uint64_t cs) {
	const auto& c = Get();
	return c.live && Live::Dispatch(cs);
}
} // namespace Paint
} // namespace SkipShader
