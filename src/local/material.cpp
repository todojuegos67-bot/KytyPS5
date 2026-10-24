// The material decode (material.h): rt-reflections' byte table (the per-byte means overall / on puddle-looking pixels,
// taken by its calibration) plus realism.comp's correlations (lit chroma, lit luma, roughness) decide which bytes of the
// packed G-buffer hold roughness, wetness, albedo and metal. Every decision is logged with its numbers and the env var
// that overrides it. Nothing here touches the GPU; realism.cpp passes the codes to its shader (material.glsl).
#include "local/material.h"

#include "local/rt-reflections.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace Libs::Graphics::FrameGen {
namespace {

constexpr const char* FieldNames[MatFields] = {"normal", "roughness", "wetness", "albedo", "metal"};
constexpr const char* EnvNames[MatFields]   = {"KYTY_REALISM_NORMAL", "KYTY_REALISM_ROUGH", "KYTY_REALISM_WET",
                                               "KYTY_REALISM_ALBEDO", "KYTY_REALISM_METAL"};
// The 10-24 log's fields (the first build's calibration, reported by the user): the defaults until a calibration runs.
constexpr int32_t DefaultNormal = 3 | (9 << 2) | 64;        // layer 3, oct 8:8 bits 0..15, world space (mean cos 0.983)
constexpr int32_t DefaultRough  = 2 | (0 << 2) | (8 << 7);  // L2.b0: 0.40 overall, 0.01 on puddles
constexpr int32_t DefaultWet    = 1 | (16 << 2) | (8 << 7); // L1.b2: 0.60 overall, 0.96 on puddles

MaterialDecode g_dec;
bool           g_init = false;
bool           g_forced[MatFields] {};
uint32_t       g_table_attempt = 0;
bool           g_albedo_known = false;

void Say(const char* fmt, const std::string& a) {
	std::printf(fmt, a.c_str());
	std::fflush(stdout);
}

std::string FieldText(int field, int32_t c) {
	if (c < 0) return "none";
	char buf[96];
	if (field == MatNormal) {
		std::snprintf(buf, sizeof(buf), "layer %d layout %d%s%s", c & 3, (c >> 2) & 15, (c & 64) ? " world" : " view",
		              (c & 128) ? " negated" : "");
	} else if (field == MatAlbedo) {
		const int s = (c >> 2) & 31;
		std::snprintf(buf, sizeof(buf), "L%d.b%d..b%d%s%s", c & 3, s / 8, s / 8 + 2, (c & 64) ? " (b, g, r)" : " (r, g, b)",
		              (c & 128) ? " linear" : " sRGB");
	} else {
		const int s = (c >> 2) & 31, bits = (c >> 7) & 31;
		if (bits == 8 && s % 8 == 0) std::snprintf(buf, sizeof(buf), "L%d.b%d%s", c & 3, s / 8, (c & 4096) ? " inverted" : "");
		else std::snprintf(buf, sizeof(buf), "layer %d bits %d..%d%s", c & 3, s, s + bits - 1, (c & 4096) ? " inverted" : "");
	}
	return buf;
}

// "<layer>:<shift>:<bits>[g]" (fields), "<layer>:<layout>[w][n]" (normal), "<layer>:<shift>[r][l]" (albedo), "none".
bool ParseEnv(int field, int32_t& code) {
	const char* value = std::getenv(EnvNames[field]);
	if (value == nullptr || *value == '\0') return false;
	const std::string v = value;
	if (v == "none" || v == "-1") {
		code = -1;
		return true;
	}
	int a = -1, b = -1, c = -1;
	if (field == MatNormal) {
		if (std::sscanf(value, "%d:%d", &a, &b) < 2 || a < 0 || a > 3 || b < 0 || b > 12) return false;
		code = a | (b << 2) | (v.find('w') != std::string::npos ? 64 : 0) | (v.find('n') != std::string::npos ? 128 : 0);
		return true;
	}
	if (field == MatAlbedo) {
		if (std::sscanf(value, "%d:%d", &a, &b) < 2 || a < 0 || a > 3 || (b != 0 && b != 8)) return false;
		const std::string flags = v.substr(v.rfind(':') + 1);
		code = a | (b << 2) | (flags.find('r') != std::string::npos ? 64 : 0) | (flags.find('l') != std::string::npos ? 128 : 0);
		return true;
	}
	if (std::sscanf(value, "%d:%d:%d", &a, &b, &c) < 3 || a < 0 || a > 3 || b < 0 || c < 1 || c > 16 || b + c > 32) return false;
	code = a | (b << 2) | (c << 7) | (v.find('g') != std::string::npos ? 4096 : 0);
	return true;
}

// The bytes the decoded normal occupies (rt-reflections' rule: its layout's bits in its layer).
bool InNormal(int32_t normal, int layer, int b) {
	if (normal < 0 || layer != (normal & 3)) return false;
	switch ((normal >> 2) & 15) {
		case 1: case 3: return b <= 2;
		case 2: return b >= 1;
		case 9: return b <= 1;
		case 10: return b >= 2;
		default: return true;
	}
}
int ByteOf(int32_t field_code) { return field_code < 0 ? -1 : (field_code & 3) * 4 + ((field_code >> 2) & 31) / 8; }

void Set(int field, int32_t code, int confidence, const std::string& source) {
	if (g_forced[field]) return;
	if (g_dec.code[field] == code && g_dec.confidence[field] == confidence) {
		g_dec.source[field] = source;
		return;
	}
	g_dec.code[field] = code, g_dec.confidence[field] = confidence, g_dec.source[field] = source;
	++g_dec.version;
	static const char* conf[] = {"none", "low", "medium", "high"};
	std::printf("Material: %s = %s (confidence %s; %s; override %s)\n", FieldNames[field], FieldText(field, code).c_str(),
	            conf[std::clamp(confidence, 0, 3)], source.c_str(), EnvNames[field]);
	std::fflush(stdout);
}

void Init() {
	if (g_init) return;
	g_init = true;
	g_dec.code       = {DefaultNormal, DefaultRough, DefaultWet, -1, -1};
	g_dec.confidence = {1, 1, 1, 0, 0};
	g_dec.source     = {"default from the 10-24 log (mean cos 0.983)", "default from the 10-24 log (L2.b0 0.40 overall / 0.01 on puddles)",
	                    "default from the 10-24 log (L1.b2 0.60 / 0.96)", "not identified yet (lit chroma stands in)", "not identified yet"};
	std::string env;
	for (int f = 0; f < MatFields; ++f) {
		int32_t code = -1;
		if (!ParseEnv(f, code)) {
			if (std::getenv(EnvNames[f]) != nullptr) std::printf("Material: %s=%s not understood, ignored\n", EnvNames[f], std::getenv(EnvNames[f]));
			continue;
		}
		g_dec.code[f] = code, g_dec.confidence[f] = 3, g_dec.source[f] = std::string("environment (") + EnvNames[f] + ")";
		g_forced[f] = true;
		env += std::string(" ") + FieldNames[f] + "=" + FieldText(f, code);
		if (f == MatAlbedo) g_albedo_known = code >= 0;
	}
	if (const char* s = std::getenv("KYTY_REALISM_SMOOTH"); s != nullptr) {
		float lo = 0, hi = 0;
		if (std::sscanf(s, "%f:%f", &lo, &hi) == 2 && lo >= 0.0f && hi > lo && hi <= 1.0f) g_dec.smooth_lo = lo, g_dec.smooth_hi = hi;
	}
	std::printf("Material: decode starts from the 10-24 log's fields (normal L3 oct 8:8 world, roughness L2.b0, wetness L1.b2; "
	            "albedo and metal unknown until the calibration)%s%s\n",
	            env.empty() ? "" : "; from the environment:", env.c_str());
	std::fflush(stdout);
	++g_dec.version;
}

} // namespace

const MaterialDecode& MaterialGet() {
	Init();
	return g_dec;
}
bool MaterialAlbedoKnown() { return g_albedo_known; }

bool MaterialRefresh() {
	Init();
	RtReflByteTable t;
	if (!RtReflGetByteTable(t) || t.attempt == g_table_attempt) return false;
	g_table_attempt         = t.attempt;
	const uint32_t version0 = g_dec.version;
	char           buf[256];
	if (t.dec_normal >= 0) {
		std::snprintf(buf, sizeof(buf), "rt-reflections' calibration, mean cos %.3f against depth normals", t.normal_fit);
		Set(MatNormal, t.dec_normal, t.normal_fit >= 0.95 ? 3 : t.normal_fit >= 0.85 ? 2 : 1, buf);
	}
	const int32_t normal = g_dec.code[MatNormal];
	// Roughness / wetness: the byte the puddle-looking pixels separate on the most, downward / upward (>= 1 sd, 0.2 apart).
	int    rough = -1, wet = -1;
	double rough_sep = 0.0, wet_sep = 0.0;
	if (t.puddles >= 100) {
		for (int f = 0; f < 16; ++f) {
			if (InNormal(normal, f / 4, f % 4) || t.sd[f] < 0.05 || t.puddle[f] < 0.0) continue;
			const double d = t.puddle[f] - t.mean[f], sep = std::abs(d) / t.sd[f];
			if (sep < 1.0 || std::abs(d) < 0.2) continue;
			if (d < 0 && sep > rough_sep) rough_sep = sep, rough = f;
			if (d > 0 && sep > wet_sep) wet_sep = sep, wet = f;
		}
	}
	const auto conf = [](double sep) { return sep >= 2.0 ? 3 : sep >= 1.5 ? 2 : 1; };
	if (t.rough_trusted && t.dec_rough >= 0) {
		Set(MatRough, t.dec_rough, 3, "rt-reflections' KYTY_RT_REFLECTIONS_ROUGH");
		g_dec.smooth_lo = t.mat_lo, g_dec.smooth_hi = std::max(t.mat_hi, t.mat_lo + 0.05f);
	} else if (rough >= 0) {
		const double rp = t.puddle[rough], ra = t.mean[rough];
		std::snprintf(buf, sizeof(buf), "rt-reflections' byte table: mean %.2f overall, %.2f on %.0f puddle-looking samples, %.1f sd apart",
		              ra, rp, t.puddles, rough_sep);
		Set(MatRough, (rough / 4) | (((rough % 4) * 8) << 2) | (8 << 7), conf(rough_sep), buf);
		g_dec.smooth_lo = static_cast<float>(rp + (ra - rp) * 0.4), g_dec.smooth_hi = static_cast<float>(rp + (ra - rp) * 0.9);
	} else if (wet >= 0) {
		const double rp = 1.0 - t.puddle[wet], ra = 1.0 - t.mean[wet];
		std::snprintf(buf, sizeof(buf), "1 - the wetness / gloss byte (no byte falls on puddles; it rises %.1f sd)", wet_sep);
		Set(MatRough, (wet / 4) | (((wet % 4) * 8) << 2) | (8 << 7) | 4096, conf(wet_sep) - 1, buf);
		g_dec.smooth_lo = static_cast<float>(rp + (ra - rp) * 0.4), g_dec.smooth_hi = static_cast<float>(rp + (ra - rp) * 0.9);
	}
	if (wet >= 0 && ByteOf(g_dec.code[MatRough]) != wet) {
		std::snprintf(buf, sizeof(buf), "rt-reflections' byte table: mean %.2f overall, %.2f on puddle-looking samples, %.1f sd apart",
		              t.mean[wet], t.puddle[wet], wet_sep);
		Set(MatWet, (wet / 4) | (((wet % 4) * 8) << 2) | (8 << 7), conf(wet_sep), buf);
		g_dec.wet_lo = static_cast<float>(t.mean[wet] + (t.puddle[wet] - t.mean[wet]) * 0.3);
		g_dec.wet_hi = static_cast<float>(t.puddle[wet]);
	}
	std::printf("Material: rt-reflections' byte table attempt %u (%.0f samples, %.0f puddle-looking, %.0f water): roughness %s, "
	            "wetness %s, smooth below %.2f, rough from %.2f\n",
	            t.attempt, t.samples, t.puddles, t.water, FieldText(MatRough, g_dec.code[MatRough]).c_str(),
	            FieldText(MatWet, g_dec.code[MatWet]).c_str(), g_dec.smooth_lo, g_dec.smooth_hi);
	std::fflush(stdout);
	return g_dec.version != version0;
}

bool MaterialAnalyze(const int32_t* c, uint32_t attempt) {
	Init();
	const double n = c[MatCalN];
	if (n < 300.0) {
		std::printf("Material: calibration %u: %.0f samples (fewer than 300 matched G-buffer pixels in view): nothing decided\n",
		            attempt, n);
		std::fflush(stdout);
		return false;
	}
	const uint32_t version0 = g_dec.version;
	const auto     E        = [&](int i) { return c[i] / 1000.0 / n; };
	double         mc[3], sc[3];
	for (int k = 0; k < 3; ++k) mc[k] = E(MatCalChroma + k), sc[k] = std::sqrt(std::max(E(MatCalChroma + 3 + k) - mc[k] * mc[k], 0.0));
	const double mY = E(MatCalChroma + 6), sY = std::sqrt(std::max(E(MatCalChroma + 7) - mY * mY, 0.0));
	const double mR = E(MatCalRough), sR = std::sqrt(std::max(E(MatCalRough + 1) - mR * mR, 0.0));
	struct Stat {
		double m, sd, cc[3], cy, ext, hi, y_hi, r_hi, cr;
	} st[16];
	std::string table;
	for (int f = 0; f < 16; ++f) {
		const int o = MatCalField + f * MatCalPer;
		Stat&     s = st[f];
		s.m         = E(o), s.sd = std::sqrt(std::max(E(o + 1) - s.m * s.m, 0.0));
		for (int k = 0; k < 3; ++k) s.cc[k] = s.sd > 1e-3 && sc[k] > 1e-4 ? (E(o + 2 + k) - s.m * mc[k]) / (s.sd * sc[k]) : 0.0;
		s.cy  = s.sd > 1e-3 && sY > 1e-4 ? (E(o + 5) - s.m * mY) / (s.sd * sY) : 0.0;
		s.ext = c[o + 6] / n;
		const double nh = c[o + 7];
		s.hi   = nh / n;
		s.y_hi = nh > 0 ? c[o + 8] / 1000.0 / nh : 0.0;
		s.r_hi = nh > 0 ? c[o + 9] / 1000.0 / nh : 0.0;
		s.cr   = s.sd > 1e-3 && sR > 1e-4 ? (E(o + 10) - s.m * mR) / (s.sd * sR) : 0.0;
		char buf[128];
		std::snprintf(buf, sizeof(buf), " L%d.b%d %.2f+-%.2f chroma %+.2f/%+.2f/%+.2f luma %+.2f rough %+.2f 0|1 %.0f%%;", f / 4, f % 4,
		              s.m, s.sd, s.cc[0], s.cc[1], s.cc[2], s.cy, s.cr, 100.0 * s.ext);
		table += buf;
	}
	std::printf("Material: calibration %u: %.0f samples; lit luma %.3f+-%.3f, roughness %.2f+-%.2f; per byte (mean+-sd, "
	            "correlation with the lit chroma r/g/b, lit luma, roughness, share at 0 or 1):%s\n",
	            attempt, n, mY, sY, mR, sR, table.c_str());
	std::fflush(stdout);

	const int32_t normal = g_dec.code[MatNormal];
	const int     rough = ByteOf(g_dec.code[MatRough]), wet = ByteOf(g_dec.code[MatWet]);
	const auto    taken = [&](int f) { return InNormal(normal, f / 4, f % 4) || f == rough || f == wet; };
	char          buf[256];

	// Albedo: three consecutive bytes following the lit chroma channel by channel (either order) and the lit luma.
	{
		double best = -1.0, second = -1.0, best_l = 0.0;
		int    pick = -1;
		for (int layer = 0; layer < 4; ++layer)
			for (int s = 0; s < 2; ++s)
				for (int rev = 0; rev < 2; ++rev) {
					bool   ok = true;
					double sc3 = 0.0, lum = 0.0;
					for (int k = 0; k < 3; ++k) {
						const int f = layer * 4 + s + k;
						if (taken(f) || st[f].sd < 0.03) ok = false;
						sc3 += st[f].cc[rev ? 2 - k : k] / 3.0;
						lum += st[f].cy / 3.0;
					}
					if (!ok) continue;
					const double score = sc3 + 0.5 * std::max(lum, 0.0);
					if (score > best) second = best, best = score, best_l = lum, pick = layer | ((s * 8) << 2) | (rev ? 64 : 0);
					else second = std::max(second, score);
				}
		if (pick >= 0 && best - 0.5 * std::max(best_l, 0.0) >= 0.25 && best_l >= 0.1) {
			const double chroma = best - 0.5 * std::max(best_l, 0.0);
			std::snprintf(buf, sizeof(buf), "realism calibration: chroma correlation %.2f, luma %.2f, next candidate %.2f", chroma, best_l,
			              second);
			Set(MatAlbedo, pick, chroma >= 0.5 && best - second >= 0.15 ? 3 : chroma >= 0.35 ? 2 : 1, buf);
			if (!g_forced[MatAlbedo]) g_albedo_known = true;
		} else {
			std::printf("Material: albedo not identified (best three bytes: chroma+luma score %.2f, luma %.2f; needs chroma >= 0.25): "
			            "the lit color's chroma stands in\n",
			            best, best_l);
		}
		const Stat& b3 = st[3];
		std::printf("Material: L0.b3 (mean %.2f, %.0f%% at 0 or 1, luma correlation %+.2f): %s\n", b3.m, 100.0 * b3.ext, b3.cy,
		            b3.ext > 0.8 ? "two-valued: a flag or coverage, not used"
		            : b3.m > 0.8 && b3.cy > 0.15 ? "bright, follows the lit luma: baked AO / cavity likely (not used: the game applies it)"
		                                         : "not used");
	}
	// Wetness vs gloss: a byte that is just 1 - roughness adds nothing.
	if (wet >= 0 && rough >= 0 && !g_forced[MatWet]) {
		const double r = st[wet].cr;
		if (r <= -0.7) {
			std::snprintf(buf, sizeof(buf), "realism calibration: correlation with roughness %.2f: a gloss copy of it, not wetness", r);
			Set(MatWet, -1, 2, buf);
		} else {
			std::printf("Material: wetness %s correlates %.2f with roughness: an independent wetness / specular level (kept)\n",
			            FieldText(MatWet, g_dec.code[MatWet]).c_str(), r);
		}
	}
	// Metal: a two-valued byte whose set pixels are darker and smoother than the rest.
	{
		int    pick = -1;
		double best = 0.0;
		std::string near;
		const int alb0 = g_dec.code[MatAlbedo] >= 0 ? ByteOf(g_dec.code[MatAlbedo]) : -1;
		for (int f = 0; f < 16; ++f) {
			if (taken(f) || (alb0 >= 0 && f >= alb0 && f < alb0 + 3 && f / 4 == alb0 / 4)) continue;
			const Stat&  s     = st[f];
			const double dark  = mY > 1e-4 ? s.y_hi / mY : 1.0;
			const bool   ok    = s.ext >= 0.85 && s.hi >= 0.01 && s.hi <= 0.5 && dark <= 0.75 && s.r_hi <= mR - 0.05;
			const double score = (1.0 - dark) * s.ext;
			if (s.ext >= 0.7 && s.hi >= 0.005 && s.hi <= 0.6) {
				char b2[96];
				std::snprintf(b2, sizeof(b2), " L%d.b%d (set %.1f%%, luma x%.2f, roughness %.2f vs %.2f)", f / 4, f % 4, 100.0 * s.hi, dark,
				              s.r_hi, mR);
				near += b2;
			}
			if (ok && score > best) best = score, pick = f;
		}
		if (pick >= 0) {
			const Stat& s = st[pick];
			std::snprintf(buf, sizeof(buf), "realism calibration: %.0f%% of samples at 0 or 1, set on %.1f%%, those %.2fx as bright and "
			              "roughness %.2f vs %.2f",
			              100.0 * s.ext, 100.0 * s.hi, mY > 0 ? s.y_hi / mY : 0.0, s.r_hi, mR);
			Set(MatMetal, (pick / 4) | (((pick % 4) * 8) << 2) | (8 << 7), s.ext >= 0.95 && s.y_hi / std::max(mY, 1e-4) <= 0.5 ? 2 : 1, buf);
		} else {
			std::printf("Material: metal not identified (needs a byte at 0 or 1 on >= 85%% of samples, set pixels <= 0.75x as bright "
			            "and smoother); two-valued candidates:%s\n",
			            near.empty() ? " none" : near.c_str());
		}
	}
	std::fflush(stdout);
	return g_dec.version != version0;
}

} // namespace Libs::Graphics::FrameGen
