// KYTY_RTLS_DEBUG / KYTY_RT_LOCAL_SHADOWS step 1: the game's local light list, found and proven (rt-local-lights.h).
#include "local/rt-local-lights.h"
#include "local/hwrt-scene.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <tuple>
#include <unordered_map>

namespace Libs::Graphics::FrameGen {
// (frame-gen.cpp: the last G-buffer view-projection, row-major, clip = M * world; captured while RtLocal::Wanted())
extern std::mutex            g_last_vp_mutex;
extern std::array<float, 16> g_last_vp;
extern bool                  g_have_last_vp;
uint64_t                     GuestFlips();
} // namespace Libs::Graphics::FrameGen
namespace Libs::LibKernel::Memory {
bool TryReadBackingToHost(uint64_t vaddr, void* data, uint64_t size);
} // namespace Libs::LibKernel::Memory

namespace RtLocal {
namespace {

using V3 = std::array<double, 3>;
double Dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double Len(const V3& a) { return std::sqrt(Dot(a, a)); }
V3     Sub(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

struct Config {
	int                   debug = 0, shadows = 0;
	bool                  scan_all = false;
	std::vector<uint64_t> programs {0x9fb5cc274b5e4f82ull};
	uint32_t              period = 30;
	double                tol    = 0.05;
	int                   dumps  = 0;
	std::string           dump_dir = "_TempData/rtls";
	uint64_t              max_bytes = 256u << 10;
	uint32_t              min_match = 3;
	bool                  on = false, hw_lights = false, layout_search = false;
};

const Config& Cfg() {
	static const Config cfg = [] {
		Config c;
		const auto env = [](const char* n) -> const char* {
			const char* v = std::getenv(n);
			return v != nullptr && *v != '\0' ? v : nullptr;
		};
		if (const char* v = env("KYTY_RTLS_DEBUG")) c.debug = std::clamp(std::atoi(v), 0, 3);
		if (const char* v = env("KYTY_RT_LOCAL_SHADOWS")) c.shadows = std::clamp(std::atoi(v), 0, 3);
		if (const char* v = env("KYTY_RTLS_SCAN_ALL")) c.scan_all = *v == '1';
		if (const char* v = env("KYTY_RTLS_PERIOD")) c.period = static_cast<uint32_t>(std::clamp(std::atoi(v), 1, 100000));
		if (const char* v = env("KYTY_RTLS_TOL")) c.tol = std::clamp(std::atof(v), 0.001, 1.0);
		if (const char* v = env("KYTY_RTLS_DUMP")) c.dumps = std::clamp(std::atoi(v), 0, 64);
		if (const char* v = env("KYTY_RTLS_DUMP_DIR")) c.dump_dir = v;
		if (const char* v = env("KYTY_RTLS_MIN_MATCH")) c.min_match = static_cast<uint32_t>(std::clamp(std::atoi(v), 2, 64));
		if (const char* v = env("KYTY_RTLS_MAX_KB")) c.max_bytes = uint64_t(std::clamp(std::atoi(v), 4, 16384)) * 1024u;
		if (const char* v = env("KYTY_RTLS_CS")) {
			c.programs.clear();
			for (const char* at = v; *at != '\0';) {
				char*                    end = nullptr;
				const unsigned long long h   = std::strtoull(at, &end, 16);
				if (end == at) break;
				if (h != 0) c.programs.push_back(h);
				if (*end != ',' && *end != ';') break;
				at = end + 1;
			}
		}
		// (KYTY_HWRT_LIGHTS=1..3 with KYTY_HWRT=1, hw-lights.cpp: the apexes feed the ray-traced shadow pass; scans every 4
		// flips unless KYTY_RTLS_PERIOD says otherwise)
		// (KYTY_HWRT_CONTACT too, rt-contact.cpp: its rays toward the strongest local light)
		const char* hw_contact = env("KYTY_HWRT_CONTACT");
		if (const char* v = env("KYTY_HWRT_LIGHTS"); ((v != nullptr && std::atoi(v) > 0) || (hw_contact != nullptr && std::atoi(hw_contact) > 0)) &&
		                                             env("KYTY_HWRT") != nullptr && *env("KYTY_HWRT") != '0' && Libs::Graphics::HwScene::RtHwExperimental()) {
			c.hw_lights = true;
			if (env("KYTY_RTLS_PERIOD") == nullptr) c.period = 4;
		}
		c.layout_search = c.debug > 0 || c.shadows > 0 || c.dumps > 0 || c.scan_all;
		c.on = c.layout_search || c.hw_lights;
		if (c.on) {
			std::printf("RTLS: local light discovery on (KYTY_RTLS_DEBUG=%d, KYTY_RT_LOCAL_SHADOWS=%d, scan %s, every %u flips, "
			            "tolerance %.3f m, %d dumps)\n",
			            c.debug, c.shadows, c.scan_all ? "every normal-path CS" : "the lighting, its culling and shadow-pool consumers",
			            c.period, c.tol, c.dumps);
			if (c.shadows > 0)
				std::printf("RTLS: KYTY_RT_LOCAL_SHADOWS=%d: the shadow pass (step 2) is not in this build: it waits for the light "
				            "list's layout to be proven by this discovery (the \"RTLS: proven\" line); nothing is darkened\n",
				            c.shadows);
			std::fflush(stdout);
		}
		return c;
	}();
	return cfg;
}

// A perspective projection found in guest data: a shadowed light (or a camera).
struct Apex {
	V3       pos {}, fwd {};
	double   near_d = 0, far_d = 0, tan_x = 0, tan_y = 0;
	uint32_t faces  = 1;     // distinct forwards seen with this apex (a cube: >= 4)
	std::vector<V3> fwds;
	uint64_t first_flip = 0, last_flip = 0, cs = 0;
	uint32_t seen = 0;
};

// Rows of M (row-major unless `col`), clip = M * (p, 1).
bool PerspectiveApex(const float* m, bool col, Apex& out, bool light = true) {
	const auto R = [&](int i, int j) -> double { return col ? m[j * 4 + i] : m[i * 4 + j]; };
	{
		// (fast reject, most windows of any buffer: the depth row's spatial part along w's)
		const auto F = [&](int i, int j) { return col ? m[j * 4 + i] : m[i * 4 + j]; };
		const float d23 = F(2, 0) * F(3, 0) + F(2, 1) * F(3, 1) + F(2, 2) * F(3, 2);
		const float q2 = F(2, 0) * F(2, 0) + F(2, 1) * F(2, 1) + F(2, 2) * F(2, 2), q3 = F(3, 0) * F(3, 0) + F(3, 1) * F(3, 1) + F(3, 2) * F(3, 2);
		if (!(d23 * d23 > 0.999f * q2 * q3) || !(q3 > 1e-10f)) return false;
	}
	for (int i = 0; i < 16; ++i)
		if (!std::isfinite(m[i]) || std::abs(m[i]) > 1e7f) return false;
	const V3     r0 {R(0, 0), R(0, 1), R(0, 2)}, r1 {R(1, 0), R(1, 1), R(1, 2)}, r2 {R(2, 0), R(2, 1), R(2, 2)},
	    r3 {R(3, 0), R(3, 1), R(3, 2)};
	const double n0 = Len(r0), n1 = Len(r1), n2 = Len(r2), n3 = Len(r3);
	if (n3 < 1e-5 || n0 < 1e-5 || n1 < 1e-5 || n2 < 1e-9) return false;
	// The depth row's spatial part along w's (any perspective); x and y rows not along it, nor along each other.
	// (exactly parallel in a real projection, z = A w + B: float rounding only)
	if (std::abs(Dot(r2, r3)) < (1.0 - 1e-5) * n2 * n3) return false;
	if (std::abs(Dot(r0, r3)) > 0.98 * n0 * n3 || std::abs(Dot(r1, r3)) > 0.98 * n1 * n3 || std::abs(Dot(r0, r1)) > 0.98 * n0 * n1)
		return false;
	// Solve r0.p = -R03, r1.p = -R13, r3.p = -R33.
	const double a[3][3] = {{r0[0], r0[1], r0[2]}, {r1[0], r1[1], r1[2]}, {r3[0], r3[1], r3[2]}};
	const double b[3]    = {-R(0, 3), -R(1, 3), -R(3, 3)};
	const auto   det3    = [](const double (&k)[3][3]) {
        return k[0][0] * (k[1][1] * k[2][2] - k[1][2] * k[2][1]) - k[0][1] * (k[1][0] * k[2][2] - k[1][2] * k[2][0]) +
               k[0][2] * (k[1][0] * k[2][1] - k[1][1] * k[2][0]);
	};
	const double d = det3(a);
	if (!(std::abs(d) > 1e-6 * n0 * n1 * n3)) return false;
	V3 p {};
	for (int c = 0; c < 3; ++c) {
		double k[3][3];
		std::memcpy(k, a, sizeof(k));
		for (int r = 0; r < 3; ++r) k[r][c] = b[r];
		p[c] = det3(k) / d;
	}
	if (!(Len(p) < 1e5)) return false;
	const V3 f {r3[0] / n3, r3[1] / n3, r3[2] / n3};
	// x / y scales (the parts of r0 / r1 across the forward) per unit w: 1 / tan of the half angles (with an atlas
	// scale folded in, a multiple of it).
	const auto across = [&](const V3& r) {
		const double k = Dot(r, f);
		return Len(V3 {r[0] - k * f[0], r[1] - k * f[1], r[2] - k * f[2]}) / n3;
	};
	const double sx = across(r0), sy = across(r1);
	if (!(sx > 0.02 && sx < 200.0 && sy > 0.02 && sy < 200.0)) return false;
	// The x and y rows across the forward are perpendicular (the view's right and up; an atlas scale / bias folded into
	// x and y keeps that) and of a similar scale (a shadow map's aspect, a camera's).
	{
		const auto across_v = [&](const V3& r) {
			const double k = Dot(r, f);
			return V3 {r[0] - k * f[0], r[1] - k * f[1], r[2] - k * f[2]};
		};
		const V3 a0 = across_v(r0), a1 = across_v(r1);
		if (std::abs(Dot(a0, a1)) > 0.002 * Len(a0) * Len(a1)) return false;
		if (sx > 4.0 * sy || sy > 4.0 * sx) return false;
	}
	// Depth along the forward at which z / w = t: (alpha d + beta) / (gamma d + delta) = t.
	const double alpha = Dot(r2, f), beta = Dot(r2, p) + R(2, 3), gamma = n3, delta = Dot(r3, p) + R(3, 3);
	const auto   at    = [&](double t) {
        const double den = alpha - t * gamma;
        return std::abs(den) > 1e-12 ? (t * delta - beta) / den : -1.0;
	};
	double ds[3] = {at(0.0), at(1.0), at(-1.0)}, lo = 1e30, hi = 0;
	for (double v: ds)
		if (v > 1e-4 && v < 1e6) lo = std::min(lo, v), hi = std::max(hi, v);
	// A light's shadow projection has a finite near and far within a light's reach (a camera's: an infinite or a far
	// plane kilometres away).
	if (light && (!(lo < 1e29) || !(hi > lo * 1.01) || hi > 500.0)) return false;
	out.pos    = p;
	out.fwd    = f;
	out.near_d = lo < 1e29 ? lo : 0.0;
	out.far_d  = hi;
	out.tan_x  = 1.0 / sx;
	out.tan_y  = 1.0 / sy;
	return true;
}

// Returns the end (in words) of the last projection found, 0: none.
size_t FindApexes(const float* f, size_t n, std::vector<Apex>& out) {
	size_t end = 0;
	for (size_t o = 0; o + 16 <= n && out.size() < 512; o += 4) {
		Apex a;
		if (PerspectiveApex(f + o, false, a) || PerspectiveApex(f + o, true, a)) {
			out.push_back(a);
			end = o + 16;
			o += 12;
		}
	}
	return end;
}

bool CameraApex(Apex& cam) {
	std::array<float, 16> vp {};
	{
		std::lock_guard lock(Libs::Graphics::FrameGen::g_last_vp_mutex);
		if (!Libs::Graphics::FrameGen::g_have_last_vp) return false;
		vp = Libs::Graphics::FrameGen::g_last_vp;
	}
	return PerspectiveApex(vp.data(), false, cam, false);
}

bool LastViewProj(std::array<float, 16>& vp) {
	std::lock_guard lock(Libs::Graphics::FrameGen::g_last_vp_mutex);
	if (!Libs::Graphics::FrameGen::g_have_last_vp) return false;
	vp = Libs::Graphics::FrameGen::g_last_vp;
	return true;
}

// The proven (or best) layout of the light list.
constexpr int kNone = -100000; // (a field not identified)

struct Layout {
	uint64_t cs = 0;
	int      slot = -1;
	uint32_t stride = 0, residue = 0; // words: entry i's position at word residue + i * stride of the buffer
	uint64_t address = 0, size = 0;
	uint32_t matched = 0;             // distinct apexes matched in the last scan
	uint32_t count = 0, first = 0;    // plausible entries and the first one's index
	uint32_t scans = 0;               // scans (distinct flips) that agreed
	uint64_t last_flip = 0;
	int      radius_field = kNone, dir_field = kNone, cone_field = kNone; // words relative to the position
	int      cone_kind = 0;           // 1 cos(half), 2 tan(half), 3 half angle (rad)
	int      radius_kind = 0;         // 1 far, 2 1/far, 3 far^2, 4 1/far^2
	bool     proven = false;
	std::string field_report;
};

struct LightOut {
	V3     pos {};
	double radius = 0;
	bool   matched = false;
};

struct Program {
	uint64_t last_scan = ~0ull;
	uint32_t scans = 0;
	bool     logged = false;
	// (KYTY_HWRT_LIGHTS alone: the buffer slots that held projections and how far into them (words, 0: none yet), and the
	// round-robin slot probed in full this scan: every scan reads the known extents plus one other slot, bounded)
	std::array<uint32_t, 32> extent {};
	uint32_t                 probe = 0;
};

struct State {
	std::mutex                             mutex;
	std::unordered_map<uint64_t, Program>  programs;
	std::vector<uint64_t>                  tile_images;   // the lighting's small inputs (tile data)
	std::vector<uint64_t>                  culling;       // programs writing one of them
	std::vector<Apex>                      apexes;        // shadowed lights, deduplicated
	std::map<std::tuple<uint64_t, int, uint32_t, uint32_t>, std::pair<uint32_t, uint64_t>> evidence; // scans, last flip
	Layout                                 layout;
	std::vector<LightOut>                  lights;
	uint64_t                               lights_flip = 0;
	uint64_t                               last_log = 0;
	uint64_t                               buffers_scanned = 0, dispatches_scanned = 0;
	int                                    dumps_left = -1;
	std::string                            best_note;
} g;

bool Plausible(const float* p, const V3* cam) {
	if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) return false;
	if (p[0] == 0.0f && p[1] == 0.0f && p[2] == 0.0f) return false;
	if (std::abs(p[0]) > 1e5f || std::abs(p[1]) > 1e5f || std::abs(p[2]) > 1e5f) return false;
	// (denormals / tiny integers read as floats)
	if (std::abs(p[0]) + std::abs(p[1]) + std::abs(p[2]) < 1e-6f) return false;
	if (cam != nullptr) {
		const V3 d {p[0] - (*cam)[0], p[1] - (*cam)[1], p[2] - (*cam)[2]};
		if (Len(d) > 2000.0) return false;
	}
	return true;
}

void AddApex(const Apex& a, uint64_t flip, uint64_t cs, const Apex* cam) {
	// The camera (this frame's or a recent one's: within 0.5 m, its forward) is not a light.
	if (cam != nullptr && Len(Sub(a.pos, cam->pos)) < 0.5 && Dot(a.fwd, cam->fwd) > 0.999) return;
	for (auto& e: g.apexes) {
		if (Len(Sub(e.pos, a.pos)) < 0.02) {
			bool known = false;
			for (const auto& f: e.fwds) known = known || Dot(f, a.fwd) > 0.999;
			if (!known && e.fwds.size() < 8) e.fwds.push_back(a.fwd), e.faces = static_cast<uint32_t>(e.fwds.size());
			if (e.last_flip != flip) ++e.seen;
			e.last_flip = flip;
			return;
		}
	}
	if (g.apexes.size() >= 256) return;
	Apex n       = a;
	n.fwds.assign(1, a.fwd);
	n.first_flip = n.last_flip = flip;
	n.cs         = cs;
	n.seen       = 1;
	g.apexes.push_back(n);
}

bool ReadGuest(uint64_t address, uint64_t size, std::vector<float>& words) {
	words.clear();
	size = std::min(size, Cfg().max_bytes) & ~uint64_t {3};
	if (address == 0 || size < 16) return false;
	words.resize(size / 4);
	uint64_t done = 0;
	while (done < size) {
		const uint64_t chunk = std::min<uint64_t>(65536, size - done);
		if (!Libs::LibKernel::Memory::TryReadBackingToHost(address + done, reinterpret_cast<uint8_t*>(words.data()) + done, chunk)) break;
		done += chunk;
	}
	words.resize(done / 4);
	return words.size() >= 4;
}

struct Hit {
	uint32_t word;
	uint32_t apex;
};

// Word offsets where a float3 equals an apex (within the tolerance).
void FindHits(const std::vector<float>& w, const std::vector<const Apex*>& apexes, std::vector<Hit>& hits) {
	hits.clear();
	if (apexes.empty() || w.size() < 3) return;
	std::vector<std::pair<float, uint32_t>> xs;
	for (uint32_t i = 0; i < apexes.size(); ++i) xs.emplace_back(static_cast<float>(apexes[i]->pos[0]), i);
	std::sort(xs.begin(), xs.end());
	const double tol = Cfg().tol;
	const float  xlo = xs.front().first - static_cast<float>(tol) - 1e-3f, xhi = xs.back().first + static_cast<float>(tol) + 1e-3f;
	for (uint32_t o = 0; o + 3 <= w.size() && hits.size() < 1024; ++o) {
		const float x = w[o];
		if (!(x >= xlo && x <= xhi)) continue;
		auto it = std::lower_bound(xs.begin(), xs.end(), std::pair<float, uint32_t> {x - static_cast<float>(tol) - 1e-3f, 0u});
		for (; it != xs.end() && it->first <= x + tol + 1e-3; ++it) {
			const Apex& a = *apexes[it->second];
			const double t = std::max(tol, 1e-4 * Len(a.pos));
			if (std::abs(w[o] - a.pos[0]) < t && std::abs(w[o + 1] - a.pos[1]) < t && std::abs(w[o + 2] - a.pos[2]) < t) {
				hits.push_back({o, it->second});
				break;
			}
		}
	}
}

struct Candidate {
	uint32_t stride = 0, residue = 0, distinct = 0, count = 0, first = 0;
	std::vector<std::pair<uint32_t, uint32_t>> matched; // entry index, apex
};

// The stride and residue that hold the most distinct apexes, the entries between them plausible positions.
bool BestLayout(const std::vector<float>& w, const std::vector<Hit>& hits, uint32_t apex_count, const V3* cam, Candidate& best) {
	std::vector<uint32_t> distinct_apexes;
	for (const auto& h: hits) distinct_apexes.push_back(h.apex);
	std::sort(distinct_apexes.begin(), distinct_apexes.end());
	distinct_apexes.erase(std::unique(distinct_apexes.begin(), distinct_apexes.end()), distinct_apexes.end());
	if (distinct_apexes.size() < 2) return false;
	bool found = false;
	for (uint32_t s = 4; s <= 192; ++s) {
		// residue -> apexes there
		std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> by;
		for (const auto& h: hits) by[h.word % s].emplace_back(h.word / s, h.apex);
		for (auto& [r, list]: by) {
			std::vector<uint32_t> ap;
			for (const auto& [i, a]: list) ap.push_back(a);
			std::sort(ap.begin(), ap.end());
			ap.erase(std::unique(ap.begin(), ap.end()), ap.end());
			const uint32_t distinct = static_cast<uint32_t>(ap.size());
			if (distinct < 2 || distinct < best.distinct) continue;
			// Plausibility of every entry between the first and last match (90 %), then the run around them.
			uint32_t lo = ~0u, hi = 0;
			for (const auto& [i, a]: list) lo = std::min(lo, i), hi = std::max(hi, i);
			const uint32_t entries = static_cast<uint32_t>((w.size() - r) / s);
			uint32_t       bad     = 0;
			for (uint32_t i = lo; i <= hi; ++i)
				if (r + i * s + 3 > w.size() || !Plausible(&w[r + i * s], cam)) ++bad;
			if (bad * 10 > (hi - lo + 1)) continue;
			uint32_t first = lo, last = hi;
			while (first > 0 && Plausible(&w[r + (first - 1) * s], cam)) --first;
			while (last + 1 < entries && r + (last + 1) * s + 3 <= w.size() && Plausible(&w[r + (last + 1) * s], cam)) ++last;
			// More distinct apexes wins; on a tie the smaller stride (a multiple of the true one ties only when every
			// match falls on its multiples; a divisor of it fails the plausibility of the entries between).
			if (distinct > best.distinct || (distinct == best.distinct && s < best.stride)) {
				best.stride   = s;
				best.residue  = r;
				best.distinct = distinct;
				best.first    = first;
				best.count    = last - first + 1;
				best.matched  = list;
				found         = true;
			}
		}
	}
	(void)apex_count;
	return found;
}

// The fields of the proven struct against the matched apexes' facts, and the value ranges of the rest.
void ReportFields(const std::vector<float>& w, const Candidate& c, const std::vector<const Apex*>& apexes, Layout& layout) {
	const int    s   = static_cast<int>(c.stride);
	std::string  rep;
	char         buf[256];
	const auto   entry_word = [&](uint32_t i, int k) -> int64_t { return int64_t(c.residue) + int64_t(i) * s + k; };
	const auto   word_at    = [&](uint32_t i, int k, float& out) {
        const int64_t o = entry_word(i, k);
        if (o < 0 || o >= int64_t(w.size())) return false;
        out = w[o];
        return true;
	};
	layout.radius_field = layout.dir_field = layout.cone_field = kNone;
	const int lo_k = -static_cast<int>(c.residue % c.stride), hi_k = lo_k + s;
	for (int k = lo_k; k < hi_k; ++k) {
		if (k >= 0 && k < 3) continue; // the position
		uint32_t r_far[4] {}, r_cone[3] {}, r_dir = 0, n_spot = 0, n_all = 0;
		for (const auto& [i, ai]: c.matched) {
			const Apex& a = *apexes[ai];
			float       v = 0;
			if (!word_at(i, k, v) || !std::isfinite(v)) continue;
			++n_all;
			const auto near_rel = [](double x, double y) { return std::abs(x - y) <= 0.03 * std::max(std::abs(y), 1e-6); };
			if (a.far_d > 0) {
				r_far[0] += near_rel(v, a.far_d), r_far[1] += near_rel(v, 1.0 / a.far_d), r_far[2] += near_rel(v, a.far_d * a.far_d),
				    r_far[3] += near_rel(v, 1.0 / (a.far_d * a.far_d));
			}
			if (a.faces < 4) {
				++n_spot;
				// (the cone from the projection's x / y scale, or from half of it: a 0.5 NDC-to-UV scale folded in)
				for (const double t: {std::max(a.tan_x, a.tan_y), 0.5 * std::max(a.tan_x, a.tan_y)}) {
					const double half = std::atan(t);
					r_cone[0] += near_rel(v, std::cos(half)), r_cone[1] += near_rel(v, std::tan(half)), r_cone[2] += near_rel(v, half);
				}
				float d0 = 0, d1 = 0, d2 = 0;
				if (word_at(i, k, d0) && word_at(i, k + 1, d1) && word_at(i, k + 2, d2)) {
					const double dd = std::abs(d0 * a.fwd[0] + d1 * a.fwd[1] + d2 * a.fwd[2]);
					if (dd > 0.995 && std::abs(std::sqrt(double(d0) * d0 + double(d1) * d1 + double(d2) * d2) - 1.0) < 0.01) ++r_dir;
				}
			}
		}
		// Ranges over all plausible entries.
		float    fmin = 1e30f, fmax = -1e30f;
		uint32_t umin = ~0u, umax = 0, finite = 0, n = 0;
		for (uint32_t i = c.first; i < c.first + c.count; ++i) {
			float v = 0;
			if (!word_at(i, k, v)) continue;
			++n;
			uint32_t u = 0;
			std::memcpy(&u, &v, 4);
			umin = std::min(umin, u), umax = std::max(umax, u);
			if (std::isfinite(v) && std::abs(v) < 1e9f) ++finite, fmin = std::min(fmin, v), fmax = std::max(fmax, v);
		}
		std::string tag;
		const uint32_t need = std::max<uint32_t>(2, (n_all * 3 + 3) / 4);
		static const char* far_names[4]  = {"radius = shadow far", "1/far", "far^2", "1/far^2"};
		static const char* cone_names[3] = {"cos(half cone)", "tan(half cone)", "half cone (rad)"};
		// (every field that matches is tagged; the first is the one used)
		const auto add = [&](const char* t) {
			tag += tag.empty() ? "" : ", ";
			tag += t;
		};
		for (int q = 0; q < 4; ++q)
			if (r_far[q] >= need) {
				if (layout.radius_field == kNone) layout.radius_field = k, layout.radius_kind = q + 1;
				add(far_names[q]);
			}
		const uint32_t need_spot = std::max<uint32_t>(2, (n_spot * 3 + 3) / 4);
		for (int q = 0; q < 3; ++q)
			if (n_spot > 0 && r_cone[q] >= need_spot) {
				if (layout.cone_field == kNone) layout.cone_field = k, layout.cone_kind = q + 1;
				add(cone_names[q]);
			}
		if (n_spot > 0 && r_dir >= need_spot) {
			if (layout.dir_field == kNone) layout.dir_field = k;
			add("direction xyz = shadow forward");
		}
		std::snprintf(buf, sizeof(buf), "\n  RTLS:   +%3d B: float [%g, %g] (%u/%u finite), uint [0x%x, 0x%x]%s%s", (k - lo_k) * 4,
		              finite ? fmin : 0.0f, finite ? fmax : 0.0f, finite, n, umin, umax, tag.empty() ? "" : "  <- ", tag.c_str());
		rep += buf;
	}
	layout.field_report = rep;
}

void Dump(uint64_t flip, uint64_t cs, std::span<const DispatchBuffer> buffers, const std::vector<std::vector<float>>& data,
          std::span<const uint32_t> srt) {
	if (g.dumps_left < 0) g.dumps_left = Cfg().dumps;
	if (g.dumps_left <= 0) return;
	--g.dumps_left;
	std::error_code ec;
	std::filesystem::create_directories(Cfg().dump_dir, ec);
	char name[512];
	std::snprintf(name, sizeof(name), "%s/rtls-%06llu-%016llx.txt", Cfg().dump_dir.c_str(), static_cast<unsigned long long>(flip),
	              static_cast<unsigned long long>(cs));
	if (FILE* t = std::fopen(name, "w")) {
		std::array<float, 16> vp {};
		if (LastViewProj(vp)) {
			std::fprintf(t, "camera_view_proj_row_major");
			for (float v: vp) std::fprintf(t, " %.9g", v);
			std::fprintf(t, "\n");
		}
		for (const auto& a: g.apexes)
			std::fprintf(t, "apex %.5f %.5f %.5f fwd %.5f %.5f %.5f near %.4f far %.4f tan %.5f %.5f faces %u seen %u cs %016llx\n", a.pos[0],
			             a.pos[1], a.pos[2], a.fwd[0], a.fwd[1], a.fwd[2], a.near_d, a.far_d, a.tan_x, a.tan_y, a.faces, a.seen,
			             static_cast<unsigned long long>(a.cs));
		for (size_t i = 0; i < buffers.size(); ++i)
			std::fprintf(t, "buffer %zu 0x%llx size 0x%llx %s read 0x%zx\n", i, static_cast<unsigned long long>(buffers[i].address),
			             static_cast<unsigned long long>(buffers[i].size), buffers[i].written ? "written" : "read",
			             i < data.size() ? data[i].size() * 4 : size_t {0});
		std::fprintf(t, "srt %zu words\n", srt.size());
		std::fclose(t);
	}
	for (size_t i = 0; i < data.size(); ++i) {
		if (data[i].empty()) continue;
		std::snprintf(name, sizeof(name), "%s/rtls-%06llu-%016llx-b%02zu.bin", Cfg().dump_dir.c_str(), static_cast<unsigned long long>(flip),
		              static_cast<unsigned long long>(cs), i);
		if (FILE* b = std::fopen(name, "wb")) std::fwrite(data[i].data(), 4, data[i].size(), b), std::fclose(b);
	}
	if (!srt.empty()) {
		std::snprintf(name, sizeof(name), "%s/rtls-%06llu-%016llx-srt.bin", Cfg().dump_dir.c_str(), static_cast<unsigned long long>(flip),
		              static_cast<unsigned long long>(cs));
		if (FILE* b = std::fopen(name, "wb")) std::fwrite(srt.data(), 4, srt.size(), b), std::fclose(b);
	}
	std::printf("RTLS: dumped CS %016llx's %zu buffers at flip %llu to %s (%d dumps left)\n", static_cast<unsigned long long>(cs),
	            data.size(), static_cast<unsigned long long>(flip), Cfg().dump_dir.c_str(), g.dumps_left);
}

void LogFlip(uint64_t flip) {
	const auto& c = Cfg();
	if (c.debug == 0 && c.shadows == 0) return;
	const uint64_t every = c.debug >= 2 ? c.period : c.period * 4;
	if (g.last_log != 0 && flip < g.last_log + every) return;
	g.last_log = flip;
	uint32_t live = 0, cubes = 0;
	for (const auto& a: g.apexes)
		if (a.last_flip + 120 >= flip) ++live, cubes += a.faces >= 4;
	const Layout& l = g.layout;
	if (l.proven) {
		std::string list;
		char        b[160];
		for (size_t i = 0; i < g.lights.size() && i < 8; ++i) {
			std::snprintf(b, sizeof(b), " [%zu] (%.2f %.2f %.2f%s r %.2f)", i, g.lights[i].pos[0], g.lights[i].pos[1], g.lights[i].pos[2],
			              g.lights[i].matched ? " *" : "", g.lights[i].radius);
			list += b;
		}
		std::printf("RTLS: flip %llu: %zu lights in CS %016llx buffer %d @0x%llx (stride %u B, position at +%u B, radius %s%s); "
		            "%u shadow apexes matched, %u scans agree; shadow apexes live %u (%u cubes):%s\n",
		            static_cast<unsigned long long>(flip), g.lights.size(), static_cast<unsigned long long>(l.cs), l.slot,
		            static_cast<unsigned long long>(l.address), l.stride * 4, (l.residue % l.stride) * 4,
		            l.radius_field != kNone ? "field +" : "unknown", l.radius_field != kNone ? std::to_string((int(l.residue % l.stride) + l.radius_field) * 4).c_str() : "",
		            l.matched, l.scans, live, cubes, list.c_str());
	} else {
		std::printf("RTLS: flip %llu: no light list proven yet (%zu programs, %llu dispatches / %llu buffers scanned; %u shadow apexes "
		            "live (%u cubes), %zu in all)%s%s\n",
		            static_cast<unsigned long long>(flip), g.programs.size(), static_cast<unsigned long long>(g.dispatches_scanned),
		            static_cast<unsigned long long>(g.buffers_scanned), live, cubes, g.apexes.size(), g.best_note.empty() ? "" : "; best: ",
		            g.best_note.c_str());
	}
	std::fflush(stdout);
}

} // namespace

bool Wanted() { return Cfg().on; }

namespace {
// (KYTY_HWRT_LIGHTS alone: the render thread's per-program filter, before any binding list is built) A program found
// uninteresting (not the lighting, its culling or a shadow-pool consumer) stays so until the lighting's tile images or
// the culling programs change (g_gen); an interesting one is let through once every period flips.
struct Filter {
	uint8_t  cls = 0; // 0 unknown, 1 interesting, 2 not
	uint32_t gen = 0;
	uint64_t next_flip = 0;
};
thread_local std::unordered_map<uint64_t, Filter> t_filter;
std::atomic<uint32_t>                              g_gen {1};
} // namespace

bool WantsDispatch(uint64_t cs_hash) {
	const auto& c = Cfg();
	if (!c.on) return false;
	if (c.layout_search) return true;
	auto& f = t_filter[cs_hash];
	if (f.cls == 2 && f.gen == g_gen.load(std::memory_order_relaxed)) return false;
	const uint64_t flip = Libs::Graphics::FrameGen::GuestFlips();
	if (f.cls == 1 && flip < f.next_flip) return false;
	f.next_flip = flip + c.period;
	return true;
}

void LiveLights(std::vector<LiveLight>& out, uint64_t max_age, uint32_t min_seen) {
	out.clear();
	if (!Cfg().on) return;
	const uint64_t flip = Libs::Graphics::FrameGen::GuestFlips();
	std::lock_guard lock(g.mutex);
	for (const auto& a: g.apexes) {
		if (a.seen < min_seen || a.last_flip + max_age < flip) continue;
		LiveLight l;
		for (int i = 0; i < 3; ++i) l.pos[i] = static_cast<float>(a.pos[i]), l.fwd[i] = static_cast<float>(a.fwd[i]);
		l.tan_half = static_cast<float>(std::max(a.tan_x, a.tan_y));
		l.far_d    = static_cast<float>(a.far_d);
		l.point    = a.faces >= 4;
		l.seen     = a.seen;
		l.age      = flip > a.last_flip ? flip - a.last_flip : 0;
		l.since    = flip > a.first_flip ? flip - a.first_flip : 0;
		out.push_back(l);
	}
}
bool OverlayWanted() { return Cfg().debug > 0; }

void OnDispatch(uint64_t cs_hash, std::span<const DispatchBuffer> buffers, std::span<const DispatchImage> images,
                std::span<const uint32_t> srt) {
	const auto& c = Cfg();
	if (!c.on) return;
	const uint64_t flip = Libs::Graphics::FrameGen::GuestFlips();
	std::lock_guard lock(g.mutex);
	const bool lighting = std::find(c.programs.begin(), c.programs.end(), cs_hash) != c.programs.end();
	if (lighting) {
		// Its small inputs (tile data: an image at most 1/8 of its largest in both axes).
		uint32_t big_w = 0, big_h = 0;
		for (const auto& im: images) big_w = std::max(big_w, im.width), big_h = std::max(big_h, im.height);
		for (const auto& im: images)
			if (!im.written && im.address != 0 && im.width >= 16 && im.height >= 8 && im.width * 8 <= big_w && im.height * 8 <= big_h &&
			    std::find(g.tile_images.begin(), g.tile_images.end(), im.address) == g.tile_images.end() && g.tile_images.size() < 16) {
				g.tile_images.push_back(im.address);
				g_gen.fetch_add(1, std::memory_order_relaxed);
				std::printf("RTLS: lighting CS %016llx reads a %ux%u image @0x%llx (tile data: its writer is the light culling)\n",
				            static_cast<unsigned long long>(cs_hash), im.width, im.height, static_cast<unsigned long long>(im.address));
			}
	}
	bool culling = std::find(g.culling.begin(), g.culling.end(), cs_hash) != g.culling.end();
	if (!culling && !lighting)
		for (const auto& im: images)
			if (im.written && std::find(g.tile_images.begin(), g.tile_images.end(), im.address) != g.tile_images.end()) {
				culling = true;
				if (g.culling.size() < 16) g.culling.push_back(cs_hash), g_gen.fetch_add(1, std::memory_order_relaxed);
				std::printf("RTLS: CS %016llx writes the lighting's tile image @0x%llx (the light culling): scanned too\n",
				            static_cast<unsigned long long>(cs_hash), static_cast<unsigned long long>(im.address));
				break;
			}
	bool pool = false;
	for (const auto& im: images)
		pool = pool || (!im.written && im.depth_like && im.layers >= 6 && im.width == im.height && im.width >= 256);
	if (!c.layout_search) {
		auto& f = t_filter[cs_hash];
		f.cls   = lighting || culling || pool ? 1 : 2;
		f.gen   = g_gen.load(std::memory_order_relaxed);
	}
	if (!lighting && !culling && !pool && !c.scan_all) return;
	auto& prog = g.programs[cs_hash];
	if (prog.last_scan != ~0ull && flip < prog.last_scan + c.period) return;
	prog.last_scan = flip;
	++prog.scans;
	++g.dispatches_scanned;
	if (!prog.logged && c.debug >= 1) {
		prog.logged = true;
		std::printf("RTLS: scanning CS %016llx (%s%s%s%s): %zu buffers, %zu images, SRT %zu words\n", static_cast<unsigned long long>(cs_hash),
		            lighting ? "lighting " : "", culling ? "culling " : "", pool ? "shadow-pool consumer " : "",
		            c.scan_all && !lighting && !culling && !pool ? "scan-all" : "", buffers.size(), images.size(), srt.size());
	}
	Apex       cam {};
	const bool have_cam = CameraApex(cam);
	const V3*  cam_pos  = have_cam ? &cam.pos : nullptr;
	// 1. Every perspective projection this dispatch can read: shadowed lights.
	std::vector<Apex> found;
	{
		std::vector<float> f(srt.size());
		if (!srt.empty()) std::memcpy(f.data(), srt.data(), srt.size() * 4);
		FindApexes(f.data(), f.size(), found);
	}
	// (KYTY_HWRT_LIGHTS alone needs the projections only: no light-list search, and bounded reads (the known extents plus
	// one other slot a scan, <= 64 KiB), so a scan's cost stays flat on the render thread)
	const bool lights_only = !c.layout_search;
	if (lights_only && buffers.size() > 0) prog.probe = (prog.probe + 1) % static_cast<uint32_t>(std::min<size_t>(buffers.size(), 32));
	std::vector<std::vector<float>> data(buffers.size());
	for (size_t i = 0; i < buffers.size(); ++i) {
		if (buffers[i].written) continue;
		uint64_t size = buffers[i].size;
		if (lights_only) {
			const uint32_t known = i < 32 ? prog.extent[i] : 0;
			if (known > 0 && i != prog.probe) size = std::min<uint64_t>(size, (uint64_t {known} + 64) * 4);
			else if (i == prog.probe) size = std::min<uint64_t>(size, 64u << 10);
			else continue;
		}
		if (ReadGuest(buffers[i].address, size, data[i])) {
			++g.buffers_scanned;
			// (projections: in the first 256 KiB; a light list is searched in all that was read)
			const size_t end = FindApexes(data[i].data(), std::min<size_t>(data[i].size(), 65536), found);
			if (lights_only && i < 32 && end > prog.extent[i]) prog.extent[i] = static_cast<uint32_t>(end);
		}
	}
	for (const auto& a: found) AddApex(a, flip, cs_hash, have_cam ? &cam : nullptr);
	// (stale: not seen for 600 flips)
	g.apexes.erase(std::remove_if(g.apexes.begin(), g.apexes.end(), [&](const Apex& a) { return a.last_flip + 600 < flip; }),
	               g.apexes.end());
	if (c.dumps > 0) Dump(flip, cs_hash, buffers, data, srt);
	if (c.debug >= 2 && !found.empty()) {
		std::printf("RTLS: CS %016llx flip %llu: %zu perspective projections (apex of the first (%.3f %.3f %.3f), far %.2f, tan %.3f)\n",
		            static_cast<unsigned long long>(cs_hash), static_cast<unsigned long long>(flip), found.size(), found[0].pos[0],
		            found[0].pos[1], found[0].pos[2], found[0].far_d, found[0].tan_x);
	}
	if (lights_only) return; // (KYTY_HWRT_LIGHTS: the apexes are all it reads)
	// 2. The light list: float3s at the apexes, one stride.
	std::vector<const Apex*> live;
	for (const auto& a: g.apexes)
		if (a.last_flip + 120 >= flip) live.push_back(&a);
	if (live.size() < 2) {
		LogFlip(flip);
		return;
	}
	std::vector<Hit> hits;
	for (size_t i = 0; i < data.size(); ++i) {
		if (data[i].empty()) continue;
		FindHits(data[i], live, hits);
		Candidate cand;
		if (!BestLayout(data[i], hits, static_cast<uint32_t>(live.size()), cam_pos, cand)) continue;
		auto& ev = g.evidence[{cs_hash, static_cast<int>(i), cand.stride, cand.residue % cand.stride}];
		if (ev.second != flip) ++ev.first, ev.second = flip;
		char note[256];
		std::snprintf(note, sizeof(note), "CS %016llx buffer %zu @0x%llx stride %u B residue %u B: %u apexes, %u entries, %u scans",
		              static_cast<unsigned long long>(cs_hash), i, static_cast<unsigned long long>(buffers[i].address), cand.stride * 4,
		              (cand.residue % cand.stride) * 4, cand.distinct, cand.count, ev.first);
		if (c.debug >= 2) std::printf("RTLS: candidate %s\n", note);
		Layout& l = g.layout;
		const bool same = l.cs == cs_hash && l.slot == static_cast<int>(i) && l.stride == cand.stride &&
		                  l.residue % l.stride == cand.residue % cand.stride;
		// Take it when it is the layout already held, or holds more apexes than it.
		if (!same && l.proven && cand.distinct <= l.matched && l.last_flip + 300 > flip) continue;
		g.best_note = note;
		const bool was_proven = l.proven && same;
		l.cs      = cs_hash;
		l.slot    = static_cast<int>(i);
		l.stride  = cand.stride;
		l.residue = cand.residue;
		l.address = buffers[i].address;
		l.size    = buffers[i].size;
		l.matched = cand.distinct;
		l.count   = cand.count;
		l.first   = cand.first;
		l.scans   = ev.first;
		l.last_flip = flip;
		l.proven  = ev.first >= 3 && cand.distinct >= c.min_match;
		ReportFields(data[i], cand, live, l);
		// The lights.
		g.lights.clear();
		for (uint32_t e = cand.first; e < cand.first + cand.count && g.lights.size() < 1024; ++e) {
			const float* p = &data[i][cand.residue + size_t(e) * cand.stride];
			LightOut     o;
			o.pos = {p[0], p[1], p[2]};
			for (const auto& [mi, ai]: cand.matched) o.matched = o.matched || mi == e;
			if (l.radius_field != kNone) {
				const int64_t ro = int64_t(cand.residue) + int64_t(e) * cand.stride + l.radius_field;
				if (ro >= 0 && ro < int64_t(data[i].size())) {
					const double v = data[i][ro];
					o.radius = l.radius_kind == 1 ? v : l.radius_kind == 2 ? (v > 0 ? 1.0 / v : 0) : l.radius_kind == 3 ? std::sqrt(std::max(v, 0.0))
					                                                                                 : (v > 0 ? 1.0 / std::sqrt(v) : 0);
				}
			}
			g.lights.push_back(o);
		}
		g.lights_flip = flip;
		static std::set<std::tuple<uint64_t, int, uint32_t, uint32_t>> said_proven; // (each layout logged once)
		if (l.proven && !was_proven && said_proven.insert({cs_hash, l.slot, l.stride, l.residue % l.stride}).second) {
			std::printf("RTLS: proven: the local light list is CS %016llx's buffer %d @0x%llx (0x%llx bytes): %u entries of %u bytes from "
			            "entry %u, the position (float3, world space) at +%u B of each (if the array starts at the buffer's start); %u "
			            "distinct shadow apexes (spot / cube projections the game reads) found at that stride, %u scans agree. Fields, "
			            "relative to the struct (word ranges over the %u entries; matched against the shadow projections' far / cone / "
			            "forward):%s%s\n",
			            static_cast<unsigned long long>(cs_hash), l.slot, static_cast<unsigned long long>(l.address),
			            static_cast<unsigned long long>(l.size), l.count, l.stride * 4, l.first, (l.residue % l.stride) * 4, l.matched,
			            l.scans, l.count, l.field_report.c_str(),
			            l.count <= l.matched ? "\n  RTLS: (every entry matched a shadow projection: this may be the shadowed views only, not "
			                                   "the whole light list; lights without shadows would be missing)"
			                                 : "");
			std::fflush(stdout);
		}
	}
	LogFlip(flip);
}

bool DebugMarkers(float x0, float y0, float w, float h, std::vector<Marker>& out) {
	out.clear();
	if (Cfg().debug == 0) return false;
	std::array<float, 16> vp {};
	const bool            have_vp = LastViewProj(vp);
	std::lock_guard       lock(g.mutex);
	const uint64_t        flip = Libs::Graphics::FrameGen::GuestFlips();
	const auto project = [&](const V3& p, float& x, float& y, double& wclip) {
		const double cx = vp[0] * p[0] + vp[1] * p[1] + vp[2] * p[2] + vp[3];
		const double cy = vp[4] * p[0] + vp[5] * p[1] + vp[6] * p[2] + vp[7];
		wclip           = vp[12] * p[0] + vp[13] * p[1] + vp[14] * p[2] + vp[15];
		if (wclip < 0.05) return false;
		const double nx = cx / wclip, ny = cy / wclip;
		if (std::abs(nx) > 1.2 || std::abs(ny) > 1.2) return false;
		// (GNM viewport: render row 0 = NDC y +1)
		x = x0 + static_cast<float>((nx * 0.5 + 0.5) * w);
		y = y0 + static_cast<float>((0.5 - ny * 0.5) * h);
		return true;
	};
	const double xscale = std::sqrt(double(vp[0]) * vp[0] + double(vp[1]) * vp[1] + double(vp[2]) * vp[2]);
	uint32_t     shown_apex = 0, shown_lights = 0;
	if (have_vp) {
		for (const auto& a: g.apexes) {
			if (a.last_flip + 120 < flip) continue;
			float  x = 0, y = 0;
			double wc = 0;
			if (!project(a.pos, x, y, wc)) continue;
			Marker m;
			m.x = x, m.y = y, m.radius = 9.0f, m.shape = 1;
			m.rgba  = a.faces >= 4 ? 0xff00a0ffu : 0xff00ffffu; // cube orange, spot yellow
			char b[48];
			std::snprintf(b, sizeof(b), "%s %.1fm", a.faces >= 4 ? "pt" : "spot", a.far_d);
			m.label = b;
			out.push_back(std::move(m));
			++shown_apex;
		}
		for (size_t i = 0; i < g.lights.size() && g.lights_flip + 600 >= flip; ++i) {
			const auto& l = g.lights[i];
			float       x = 0, y = 0;
			double      wc = 0;
			if (!project(l.pos, x, y, wc)) continue;
			Marker m;
			m.x = x, m.y = y, m.shape = 0;
			m.radius = l.radius > 0 ? static_cast<float>(std::clamp(l.radius * xscale / wc * 0.5 * w, 4.0, 2000.0)) : 6.0f;
			m.rgba   = g.layout.proven ? 0xc000ff00u : 0xa000a0a0u;
			m.label  = std::to_string(i);
			out.push_back(std::move(m));
			++shown_lights;
		}
	}
	Marker t;
	t.x = x0 + 24.0f, t.y = y0 + 24.0f, t.shape = 2, t.rgba = 0xffffffffu;
	char b[200];
	std::snprintf(b, sizeof(b), "RTLS: %s, %zu lights (%u on screen), %u shadow apexes on screen%s", g.layout.proven ? "list proven" : "not proven",
	              g.lights.size(), shown_lights, shown_apex, have_vp ? "" : " (no camera: needs an upscaler / frame generation)");
	t.label = b;
	out.push_back(std::move(t));
	return true;
}

} // namespace RtLocal
