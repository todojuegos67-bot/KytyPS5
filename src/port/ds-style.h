// (trucos11) The F2 menu's look, Demon's Souls remake style, all drawn here (no game art, no logos): blue-black stone
// with slow fog and a strong vignette, bone-white serif titles with tracking, small caps subtitles, thin rules with a
// rhombus, candle-lit selection plates that breathe, aged-metal medallion icons (filled, shaded silhouettes on a
// bevelled dark disc). Header-only, ImGui draw lists only: ds-menu.cpp uses it, and so does the off-line capture
// program (trucos11/capturas), so the PNGs show this very code.
#pragma once

#include "imgui.h"
#include "imgui_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

namespace DsStyle {

// ---- Palette (sRGB 0..255) ----
struct Rgb {
	int r, g, b;
};
constexpr Rgb kInk {7, 8, 11}, kStone {22, 25, 32}, kStoneHi {36, 40, 50};
constexpr Rgb kBone {226, 219, 202}, kBoneDim {150, 145, 133};
constexpr Rgb kGold {170, 146, 98}, kGoldHi {222, 198, 142}, kBronze {92, 72, 44};
constexpr Rgb kCandle {255, 172, 92}, kSoul {196, 216, 255};

inline ImU32 C(Rgb c, float a) {
	return IM_COL32(c.r, c.g, c.b, static_cast<int>(255.0f * std::clamp(a, 0.0f, 1.0f)));
}
inline float Hash(int i) {
	uint32_t x = static_cast<uint32_t>(i) * 2654435761u;
	x ^= x >> 15;
	x *= 2246822519u;
	x ^= x >> 13;
	return static_cast<float>(x & 0xffffff) / 16777215.0f;
}
inline float Breath(float time, float speed = 1.4f, float seed = 0.0f) {
	return 0.5f + 0.5f * std::sin(time * speed + seed);
}

// ---- Background: stone gradient, slow fog, stone grain, strong vignette ----
inline void Vignette(ImDrawList* dl, ImVec2 a, ImVec2 b, float alpha) {
	const float w = b.x - a.x, h = b.y - a.y;
	const ImU32 d = IM_COL32(0, 0, 0, static_cast<int>(235 * alpha)), z = IM_COL32(0, 0, 0, 0);
	const float ex = std::round(w * 0.28f), ey = std::round(h * 0.32f);
	dl->AddRectFilledMultiColor(a, {a.x + ex, b.y}, d, z, z, d);
	dl->AddRectFilledMultiColor({b.x - ex, a.y}, b, z, d, d, z);
	dl->AddRectFilledMultiColor(a, {b.x, a.y + ey}, d, d, z, z);
	dl->AddRectFilledMultiColor({a.x, b.y - ey}, b, z, z, d, d);
}
inline void Backdrop(ImDrawList* dl, ImVec2 a, ImVec2 b, float alpha, float time) {
	const float w = b.x - a.x, h = b.y - a.y, u = std::min(w / 1920.0f, h / 1080.0f);
	dl->AddRectFilledMultiColor(a, b, C(kStone, 0.94f * alpha), C(kStone, 0.94f * alpha), C(kInk, 0.97f * alpha), C(kInk, 0.97f * alpha));
	// Fog: large soft banks drifting sideways, each a stack of faint ellipses.
	for (int i = 0; i < 14; i++) {
		const float sp = 0.004f + 0.006f * Hash(i * 5 + 1);
		const float x  = a.x + w * std::fmod(Hash(i * 5) + time * sp * (i % 2 ? 1.0f : -1.0f) + 2.0f, 1.0f);
		const float y  = a.y + h * (0.35f + 0.6f * Hash(i * 5 + 2));
		const float rx = (260.0f + 360.0f * Hash(i * 5 + 3)) * u, ry = rx * 0.32f;
		for (int k = 6; k >= 1; k--) {
			const float f = static_cast<float>(k) / 6.0f;
			dl->AddEllipseFilled({x, y}, {rx * f, ry * f}, C(kStoneHi, alpha * 0.018f), 0.0f, 40);
		}
	}
	// Stone grain: fixed faint specks.
	for (int i = 0; i < 700; i++) {
		const float x = a.x + w * Hash(i * 3 + 11), y = a.y + h * Hash(i * 3 + 12), s = std::max(1.0f, std::round(u * (1.0f + Hash(i * 3 + 13))));
		dl->AddRectFilled({std::round(x), std::round(y)}, {std::round(x) + s, std::round(y) + s}, Hash(i) > 0.5f ? C(kStoneHi, alpha * 0.10f) : C(kInk, alpha * 0.18f));
	}
	Vignette(dl, a, b, alpha);
}
// Soul motes: faint, rising slowly.
inline void Motes(ImDrawList* dl, ImVec2 a, ImVec2 b, float alpha, float time, int count = 56) {
	const float w = b.x - a.x, h = b.y - a.y, u = std::min(w / 1920.0f, h / 1080.0f);
	for (int i = 0; i < count; i++) {
		const float sp = 0.010f + 0.018f * Hash(i * 3 + 1);
		const float y  = a.y + h * (1.0f - std::fmod(Hash(i * 3 + 2) + time * sp, 1.0f));
		const float x  = a.x + w * Hash(i * 3) + std::sin(time * 0.5f + i) * 16.0f * u;
		const float fl = Breath(time, 1.1f + Hash(i), i * 1.7f);
		const float r  = (1.0f + 1.8f * Hash(i * 7)) * u;
		dl->AddCircleFilled({x, y}, r * 4.0f, C(kSoul, alpha * 0.03f * fl), 12);
		dl->AddCircleFilled({x, y}, r, C(kSoul, alpha * 0.45f * fl), 8);
	}
}

// ---- Typography ----
// UTF-8 upper case for ASCII and Latin-1 letters (á -> Á: C3 A0..BE -> C3 80..9E).
inline std::string Upper(const char* s) {
	std::string o;
	for (const unsigned char* p = reinterpret_cast<const unsigned char*>(s); *p; p++) {
		if (*p >= 'a' && *p <= 'z') o += static_cast<char>(*p - 32);
		else if (*p == 0xC3 && p[1] >= 0xA0 && p[1] <= 0xBE && p[1] != 0xB7) o += static_cast<char>(0xC3), o += static_cast<char>(*++p - 0x20);
		else o += static_cast<char>(*p);
	}
	return o;
}
inline int CharLen(unsigned char c) {
	return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
}
inline float TrackedWidth(ImFont* f, float size, const char* t, float tracking) {
	float w = 0;
	int   n = 0;
	for (const char* p = t; *p; n++) {
		const int k = CharLen(static_cast<unsigned char>(*p));
		w += f->CalcTextSizeA(size, FLT_MAX, 0.0f, p, p + k).x;
		p += k;
	}
	return w + tracking * std::max(0, n - 1);
}
inline void TrackedText(ImDrawList* dl, ImFont* f, float size, ImVec2 at, ImU32 col, const char* t, float tracking) {
	float x = std::round(at.x);
	for (const char* p = t; *p;) {
		const int k = CharLen(static_cast<unsigned char>(*p));
		dl->AddText(f, size, {x, std::round(at.y)}, col, p, p + k);
		x += f->CalcTextSizeA(size, FLT_MAX, 0.0f, p, p + k).x + tracking;
		p += k;
	}
}
// A title: bone white, tracked, a soft dark halo under it (no blur: two offset dark copies).
inline float Title(ImDrawList* dl, ImFont* f, float size, ImVec2 at, const char* t, float alpha, float tracking_em = 0.08f) {
	const float tr = std::round(size * tracking_em);
	TrackedText(dl, f, size, {at.x + 2, at.y + 2}, IM_COL32(0, 0, 0, static_cast<int>(200 * alpha)), t, tr);
	TrackedText(dl, f, size, {at.x, at.y + 1}, IM_COL32(0, 0, 0, static_cast<int>(140 * alpha)), t, tr);
	TrackedText(dl, f, size, at, C(kBone, alpha), t, tr);
	return TrackedWidth(f, size, t, tr);
}
// Small caps: the text in capitals, the first letter of each word at full size, the rest at 78 %, muted gold.
inline float SmallCaps(ImDrawList* dl, ImFont* f, float size, ImVec2 at, const char* t, ImU32 col, float tracking_em = 0.12f) {
	const std::string up    = Upper(t);
	const float       small = std::round(size * 0.78f), tr = std::round(size * tracking_em);
	float             x     = std::round(at.x);
	bool              first = true;
	for (const char* p = up.c_str(); *p;) {
		const int   k  = CharLen(static_cast<unsigned char>(*p));
		const float sz = first ? size : small;
		dl->AddText(f, sz, {x, std::round(at.y + (size - sz) * 0.82f)}, col, p, p + k);
		x += f->CalcTextSizeA(sz, FLT_MAX, 0.0f, p, p + k).x + tr;
		first = *p == ' ';
		p += k;
	}
	return x - at.x;
}

// ---- Rules and frames ----
inline void Rhombus(ImDrawList* dl, ImVec2 c, float r, ImU32 col, bool filled = true) {
	const ImVec2 p[4] = {{c.x, c.y - r}, {c.x + r, c.y}, {c.x, c.y + r}, {c.x - r, c.y}};
	if (filled) dl->AddConvexPolyFilled(p, 4, col);
	else dl->AddPolyline(p, 4, col, ImDrawFlags_Closed, 1.0f);
}
// A thin rule fading out at both ends, a rhombus in the middle and two small ones beside it.
inline void DiamondRule(ImDrawList* dl, float x0, float x1, float y, float alpha, float unit) {
	y             = std::round(y) + 0.5f;
	const float c = std::round((x0 + x1) * 0.5f), t = std::max(1.0f, std::round(unit));
	const float g = 18 * unit;
	dl->AddRectFilledMultiColor({x0, y - t * 0.5f}, {c - g, y + t * 0.5f}, C(kGold, 0), C(kGold, alpha * 0.8f), C(kGold, alpha * 0.8f), C(kGold, 0));
	dl->AddRectFilledMultiColor({c + g, y - t * 0.5f}, {x1, y + t * 0.5f}, C(kGold, alpha * 0.8f), C(kGold, 0), C(kGold, 0), C(kGold, alpha * 0.8f));
	Rhombus(dl, {c, y}, 6 * unit, C(kGoldHi, alpha));
	Rhombus(dl, {c, y}, 10 * unit, C(kGold, alpha * 0.7f), false);
	Rhombus(dl, {c - g + 3 * unit, y}, 2.5f * unit, C(kGold, alpha));
	Rhombus(dl, {c + g - 3 * unit, y}, 2.5f * unit, C(kGold, alpha));
}
// A panel: translucent ink, a thin bone border, an inner gold hairline, rhombi in the corners.
inline void Panel(ImDrawList* dl, ImVec2 a, ImVec2 b, float alpha, float unit) {
	const float t = std::max(1.0f, std::round(unit));
	a             = {std::round(a.x), std::round(a.y)};
	b             = {std::round(b.x), std::round(b.y)};
	dl->AddRectFilledMultiColor(a, b, C(kStone, alpha * 0.80f), C(kStone, alpha * 0.80f), C(kInk, alpha * 0.88f), C(kInk, alpha * 0.88f));
	dl->AddRect(a, b, C(kBoneDim, alpha * 0.55f), 0, 0, t);
	dl->AddRect({a.x + 5 * t, a.y + 5 * t}, {b.x - 5 * t, b.y - 5 * t}, C(kGold, alpha * 0.35f), 0, 0, t);
	for (int q = 0; q < 4; q++) Rhombus(dl, {q & 1 ? b.x : a.x, q & 2 ? b.y : a.y}, 5 * unit, C(kGoldHi, alpha * 0.9f));
}
// The selection plate: dark, thin border; hovered / selected: a warm candle light from inside that breathes, the border
// warming to gold. `hot` and `sel` are 0..1 (eased by the caller).
inline void CandlePlate(ImDrawList* dl, ImVec2 a, ImVec2 b, float hot, float sel, float alpha, float time, float unit) {
	const float t = std::max(1.0f, std::round(unit));
	a             = {std::round(a.x), std::round(a.y)};
	b             = {std::round(b.x), std::round(b.y)};
	const float k = std::max(hot, sel), br = 0.75f + 0.25f * Breath(time, 1.6f);
	dl->AddRectFilledMultiColor(a, b, C(kStone, alpha * 0.55f), C(kStone, alpha * 0.55f), C(kInk, alpha * 0.70f), C(kInk, alpha * 0.70f));
	if (k > 0.001f) {
		const float cx = (a.x + b.x) * 0.5f, hgt = b.y - a.y;
		// The glow: from the bottom centre, warm, fading up and to the sides (a fan of gradients).
		dl->AddRectFilledMultiColor({a.x, a.y + hgt * 0.35f}, {cx, b.y}, C(kCandle, 0), C(kCandle, alpha * 0.10f * k * br), C(kCandle, alpha * 0.16f * k * br), C(kCandle, 0));
		dl->AddRectFilledMultiColor({cx, a.y + hgt * 0.35f}, {b.x, b.y}, C(kCandle, alpha * 0.10f * k * br), C(kCandle, 0), C(kCandle, 0), C(kCandle, alpha * 0.16f * k * br));
		dl->AddRectFilledMultiColor(a, {b.x, a.y + hgt * 0.35f}, C(kCandle, 0), C(kCandle, 0), C(kCandle, alpha * 0.10f * k * br), C(kCandle, alpha * 0.10f * k * br));
		dl->AddRectFilled({a.x + t, b.y - 2 * t}, {b.x - t, b.y - t}, C(kCandle, alpha * 0.35f * k * br));
	}
	const ImU32 border = k > 0.01f ? C(kGoldHi, alpha * (0.35f + 0.55f * k)) : C(kBoneDim, alpha * 0.30f);
	dl->AddRect(a, b, border, 0, 0, t);
	if (k > 0.01f) {
		const float c = 10 * unit;
		for (int q = 0; q < 4; q++) {
			const ImVec2 p {q & 1 ? b.x : a.x, q & 2 ? b.y : a.y};
			const float  sx = q & 1 ? -1.0f : 1.0f, sy = q & 2 ? -1.0f : 1.0f;
			dl->AddLine(p, {p.x + sx * c, p.y}, C(kGoldHi, alpha * k), 2 * t);
			dl->AddLine(p, {p.x, p.y + sy * c}, C(kGoldHi, alpha * k), 2 * t);
		}
	}
}
// A switch: iron track; on: a gold-lit track and a soul-light knob with a halo.
inline void Toggle(ImDrawList* dl, ImVec2 a, ImVec2 b, float on, float alpha, float time) {
	const float h = b.y - a.y, r = h * 0.5f;
	dl->AddRectFilled(a, b, C(kInk, alpha), r);
	if (on > 0.01f) dl->AddRectFilledMultiColor({a.x + r * 0.5f, a.y + 2}, {b.x - r * 0.5f, b.y - 2}, C(kBronze, alpha * on), C(kGold, alpha * on), C(kGold, alpha * on), C(kBronze, alpha * on));
	dl->AddRect(a, b, C(on > 0.5f ? kGoldHi : kBoneDim, alpha * 0.8f), r, 0, 1.0f);
	const ImVec2 k {a.x + r + (b.x - a.x - h) * on, a.y + r};
	if (on > 0.01f)
		for (int g = 4; g >= 1; g--) dl->AddCircleFilled(k, r * (0.5f + 0.22f * g), C(kSoul, alpha * 0.05f * on * (0.6f + 0.4f * Breath(time, 2.0f))), 32);
	dl->AddCircleFilled(k, r * 0.72f, on > 0.5f ? C(kSoul, alpha) : C(kBoneDim, alpha), 32);
	dl->AddCircle(k, r * 0.72f, IM_COL32(0, 0, 0, static_cast<int>(160 * alpha)), 32, 1.0f);
}

// ---- Icons: filled silhouettes in aged metal on a bevelled dark medallion ----
enum class Icon { Back, Trophy, Pad, Keyboard, Power, Sword, Shield, Helm, Ring, Potion, Soul, Sun, Skull, Chest, Flame, Stone, Gear, Eye };

namespace Detail {
// Silhouette of an icon at centre c, half-size h (filled where it should be solid, stroked where thin), in `col`.
inline void Shape(ImDrawList* dl, Icon k, ImVec2 c, float h, ImU32 col, ImU32 hole) {
	const auto  P = [&](float x, float y) { return ImVec2 {c.x + x * h, c.y + y * h}; };
	const float w = std::max(1.5f, h * 0.11f);
	switch (k) {
		case Icon::Sword: {
			const ImVec2 blade[5] = {P(0, -0.98f), P(0.12f, -0.8f), P(0.1f, 0.3f), P(-0.1f, 0.3f), P(-0.12f, -0.8f)};
			dl->AddConvexPolyFilled(blade, 5, col);
			dl->AddLine(P(0, -0.8f), P(0, 0.25f), hole, std::max(1.0f, w * 0.35f)); // fuller
			const ImVec2 guard[6] = {P(-0.5f, 0.3f), P(0.5f, 0.3f), P(0.42f, 0.42f), P(0.1f, 0.42f), P(-0.1f, 0.42f), P(-0.42f, 0.42f)};
			dl->AddConvexPolyFilled(guard, 6, col);
			dl->AddRectFilled(P(-0.07f, 0.42f), P(0.07f, 0.78f), col);
			dl->AddCircleFilled(P(0, 0.86f), h * 0.12f, col, 16);
			break;
		}
		case Icon::Shield: {
			dl->PathLineTo(P(-0.7f, -0.78f));
			dl->PathBezierQuadraticCurveTo(P(0, -0.92f), P(0.7f, -0.78f), 10);
			dl->PathBezierCubicCurveTo(P(0.72f, 0.1f), P(0.42f, 0.6f), P(0, 0.95f), 16);
			dl->PathBezierCubicCurveTo(P(-0.42f, 0.6f), P(-0.72f, 0.1f), P(-0.7f, -0.78f), 16);
			dl->PathFillConcave(col);
			dl->AddLine(P(0, -0.7f), P(0, 0.75f), hole, w * 0.7f);
			dl->AddLine(P(-0.55f, -0.18f), P(0.55f, -0.18f), hole, w * 0.7f);
			dl->AddCircleFilled(P(0, -0.18f), h * 0.14f, col, 16);
			break;
		}
		case Icon::Helm: {
			dl->PathArcTo(P(0, -0.05f), h * 0.66f, 3.14159f, 6.2832f, 24);
			dl->PathLineTo(P(0.66f, 0.8f));
			dl->PathLineTo(P(-0.66f, 0.8f));
			dl->PathFillConvex(col);
			dl->AddRectFilled(P(-0.5f, 0.02f), P(0.5f, 0.12f), hole);
			dl->AddLine(P(0, 0.12f), P(0, 0.8f), hole, w * 0.6f);
			for (int i = 0; i < 4; i++) dl->AddCircleFilled(P(-0.35f + 0.1f * i, 0.35f + 0.1f * i), w * 0.35f, hole, 8);
			for (int i = 0; i < 4; i++) dl->AddCircleFilled(P(0.35f - 0.1f * i, 0.35f + 0.1f * i), w * 0.35f, hole, 8);
			dl->AddTriangleFilled(P(-0.1f, -0.7f), P(0.1f, -0.7f), P(0, -1.0f), col); // crest
			break;
		}
		case Icon::Ring: {
			dl->AddCircle(P(0, 0.22f), h * 0.52f, col, 40, h * 0.18f);
			const ImVec2 gem[4] = {P(0, -0.85f), P(0.28f, -0.5f), P(0, -0.2f), P(-0.28f, -0.5f)};
			dl->AddConvexPolyFilled(gem, 4, col);
			dl->AddLine(P(-0.28f, -0.5f), P(0.28f, -0.5f), hole, std::max(1.0f, w * 0.4f));
			break;
		}
		case Icon::Potion: {
			dl->AddRectFilled(P(-0.16f, -0.92f), P(0.16f, -0.72f), col, 2.0f);
			dl->AddRectFilled(P(-0.12f, -0.72f), P(0.12f, -0.38f), col);
			dl->AddCircleFilled(P(0, 0.22f), h * 0.66f, col, 40);
			dl->PathArcTo(P(0, 0.22f), h * 0.48f, 0.35f, 2.8f, 20);
			dl->PathStroke(hole, 0, w * 0.6f);
			dl->AddCircleFilled(P(-0.25f, 0.0f), h * 0.1f, hole, 12);
			break;
		}
		case Icon::Soul: {
			dl->PathLineTo(P(0, -0.98f));
			dl->PathBezierCubicCurveTo(P(0.25f, -0.55f), P(0.72f, -0.1f), P(0.55f, 0.45f), 14);
			dl->PathBezierCubicCurveTo(P(0.42f, 0.88f), P(-0.42f, 0.88f), P(-0.55f, 0.45f), 14);
			dl->PathBezierCubicCurveTo(P(-0.7f, -0.05f), P(-0.15f, -0.45f), P(0, -0.98f), 14);
			dl->PathFillConcave(col);
			dl->AddCircleFilled(P(0, 0.38f), h * 0.26f, hole, 20);
			dl->AddCircleFilled(P(0, 0.38f), h * 0.14f, col, 16);
			break;
		}
		case Icon::Sun: {
			for (int i = 0; i < 12; i++) {
				const float a0 = i * 0.5236f, a1 = a0 + 0.13f, a2 = a0 - 0.13f;
				const ImVec2 t[3] = {P(std::cos(a0) * 0.98f, std::sin(a0) * 0.98f), P(std::cos(a1) * 0.5f, std::sin(a1) * 0.5f), P(std::cos(a2) * 0.5f, std::sin(a2) * 0.5f)};
				dl->AddConvexPolyFilled(t, 3, col);
			}
			dl->AddCircleFilled(c, h * 0.46f, col, 32);
			dl->AddCircle(c, h * 0.32f, hole, 32, w * 0.6f);
			break;
		}
		case Icon::Skull: {
			dl->AddCircleFilled(P(0, -0.15f), h * 0.66f, col, 40);
			dl->AddRectFilled(P(-0.38f, 0.2f), P(0.38f, 0.72f), col, h * 0.12f);
			dl->AddEllipseFilled(P(-0.26f, -0.08f), {h * 0.18f, h * 0.21f}, hole, 0.0f, 16);
			dl->AddEllipseFilled(P(0.26f, -0.08f), {h * 0.18f, h * 0.21f}, hole, 0.0f, 16);
			dl->AddTriangleFilled(P(0, 0.12f), P(-0.09f, 0.32f), P(0.09f, 0.32f), hole);
			for (int i = -1; i <= 1; i++) dl->AddLine(P(i * 0.14f, 0.5f), P(i * 0.14f, 0.72f), hole, std::max(1.0f, w * 0.45f));
			break;
		}
		case Icon::Chest: {
			dl->PathArcTo(P(0, -0.12f), h * 0.82f, 3.14159f, 6.2832f, 24);
			dl->PathFillConvex(col);
			dl->AddRectFilled(P(-0.82f, -0.12f), P(0.82f, 0.78f), col);
			dl->AddLine(P(-0.82f, -0.05f), P(0.82f, -0.05f), hole, w * 0.6f);
			dl->AddLine(P(-0.5f, -0.8f), P(-0.5f, 0.78f), hole, w * 0.45f);
			dl->AddLine(P(0.5f, -0.8f), P(0.5f, 0.78f), hole, w * 0.45f);
			dl->AddRectFilled(P(-0.13f, 0.02f), P(0.13f, 0.34f), hole, 2.0f);
			dl->AddCircleFilled(P(0, 0.14f), h * 0.05f, col, 8);
			break;
		}
		case Icon::Trophy: {
			dl->PathArcTo(P(0, -0.35f), h * 0.55f, 0.0f, 3.14159f, 20);
			dl->PathLineTo(P(-0.55f, -0.85f));
			dl->PathLineTo(P(0.55f, -0.85f));
			dl->PathFillConvex(col);
			dl->PathArcTo(P(-0.55f, -0.45f), h * 0.28f, 1.5708f, 4.712f, 12);
			dl->PathStroke(col, 0, w);
			dl->PathArcTo(P(0.55f, -0.45f), h * 0.28f, -1.5708f, 1.5708f, 12);
			dl->PathStroke(col, 0, w);
			dl->AddRectFilled(P(-0.09f, 0.15f), P(0.09f, 0.55f), col);
			dl->AddRectFilled(P(-0.45f, 0.55f), P(0.45f, 0.8f), col, 2.0f);
			dl->AddCircleFilled(P(0, -0.45f), h * 0.14f, hole, 12);
			break;
		}
		case Icon::Pad: {
			dl->AddRectFilled(P(-0.85f, -0.42f), P(0.85f, 0.4f), col, h * 0.35f);
			dl->AddCircleFilled(P(-0.5f, 0.42f), h * 0.32f, col, 20);
			dl->AddCircleFilled(P(0.5f, 0.42f), h * 0.32f, col, 20);
			dl->AddRectFilled(P(-0.6f, -0.06f), P(-0.3f, 0.04f), hole);
			dl->AddRectFilled(P(-0.5f, -0.16f), P(-0.4f, 0.14f), hole);
			for (int i = 0; i < 4; i++) dl->AddCircleFilled(P(0.45f + 0.14f * std::cos(i * 1.5708f), -0.01f + 0.14f * std::sin(i * 1.5708f)), w * 0.55f, hole, 10);
			break;
		}
		case Icon::Keyboard: {
			dl->AddRectFilled(P(-0.92f, -0.5f), P(0.92f, 0.5f), col, h * 0.1f);
			for (int r = 0; r < 3; r++)
				for (int q = 0; q < 6; q++) dl->AddRectFilled(P(-0.76f + q * 0.27f, -0.36f + r * 0.24f), P(-0.58f + q * 0.27f, -0.2f + r * 0.24f), hole, 1.0f);
			dl->AddRectFilled(P(-0.45f, 0.3f), P(0.45f, 0.4f), hole);
			break;
		}
		case Icon::Power: {
			dl->PathArcTo(P(0, 0.08f), h * 0.66f, -1.5708f + 0.75f, -1.5708f + 6.2832f - 0.75f, 28);
			dl->PathStroke(col, 0, w * 1.6f);
			dl->AddLine(P(0, -0.9f), P(0, -0.05f), col, w * 1.6f);
			break;
		}
		case Icon::Back: {
			dl->PathArcTo(P(0.05f, 0.12f), h * 0.55f, -1.5708f, 1.5708f, 18);
			dl->PathLineTo(P(-0.55f, 0.67f));
			dl->PathStroke(col, 0, w * 1.5f);
			dl->AddLine(P(-0.5f, -0.43f), P(0.05f, -0.43f), col, w * 1.5f);
			dl->AddTriangleFilled(P(-0.9f, -0.43f), P(-0.42f, -0.78f), P(-0.42f, -0.08f), col);
			break;
		}
		case Icon::Flame: {
			dl->PathLineTo(P(0, -0.95f));
			dl->PathBezierCubicCurveTo(P(0.8f, -0.2f), P(0.7f, 0.8f), P(0, 0.85f), 16);
			dl->PathBezierCubicCurveTo(P(-0.7f, 0.8f), P(-0.8f, -0.05f), P(-0.25f, -0.35f), 16);
			dl->PathBezierCubicCurveTo(P(-0.1f, -0.1f), P(0.1f, -0.45f), P(0, -0.95f), 10);
			dl->PathFillConcave(col);
			dl->PathLineTo(P(0, -0.2f));
			dl->PathBezierCubicCurveTo(P(0.4f, 0.2f), P(0.35f, 0.7f), P(0, 0.72f), 10);
			dl->PathBezierCubicCurveTo(P(-0.35f, 0.7f), P(-0.35f, 0.25f), P(0, -0.2f), 10);
			dl->PathFillConcave(hole);
			break;
		}
		case Icon::Stone: {
			const ImVec2 s[6] = {P(-0.55f, -0.6f), P(0.3f, -0.8f), P(0.8f, -0.15f), P(0.55f, 0.65f), P(-0.3f, 0.8f), P(-0.85f, 0.1f)};
			dl->AddConvexPolyFilled(s, 6, col);
			dl->AddLine(P(-0.55f, -0.6f), P(0.1f, 0.0f), hole, w * 0.5f);
			dl->AddLine(P(0.1f, 0.0f), P(0.8f, -0.15f), hole, w * 0.5f);
			dl->AddLine(P(0.1f, 0.0f), P(-0.3f, 0.8f), hole, w * 0.5f);
			break;
		}
		case Icon::Gear: {
			for (int i = 0; i < 10; i++) {
				const float a = i * 0.6283f;
				const ImVec2 p0 = P(std::cos(a - 0.17f) * 0.62f, std::sin(a - 0.17f) * 0.62f), p1 = P(std::cos(a - 0.12f) * 0.92f, std::sin(a - 0.12f) * 0.92f);
				const ImVec2 p2 = P(std::cos(a + 0.12f) * 0.92f, std::sin(a + 0.12f) * 0.92f), p3 = P(std::cos(a + 0.17f) * 0.62f, std::sin(a + 0.17f) * 0.62f);
				const ImVec2 t[4] = {p0, p1, p2, p3};
				dl->AddConvexPolyFilled(t, 4, col);
			}
			dl->AddCircleFilled(c, h * 0.68f, col, 36);
			dl->AddCircleFilled(c, h * 0.26f, hole, 24);
			break;
		}
		case Icon::Eye: {
			dl->PathArcTo(P(0, 0.9f), h * 1.25f, -2.3f, -0.84f, 18);
			dl->PathArcTo(P(0, -0.9f), h * 1.25f, 0.84f, 2.3f, 18);
			dl->PathFillConvex(col);
			dl->AddCircleFilled(c, h * 0.32f, hole, 24);
			dl->AddCircleFilled(c, h * 0.16f, col, 16);
			break;
		}
	}
}
} // namespace Detail

// The icon in aged metal: a dark engraved shadow, the silhouette shaded top-light gold to deep bronze (vertex gradient),
// a fine bright edge on the upper left. `hole`: the colour cut-outs take (the medallion's recess).
inline void MetalIcon(ImDrawList* dl, Icon k, ImVec2 c, float size, float alpha, float bright, ImU32 hole, float time = 0.0f, float hot = 0.0f) {
	size *= 1.0f + 0.06f * hot; // (trucos13) selected: a little larger
	const float h = size * 0.5f;
	Detail::Shape(dl, k, {c.x + h * 0.06f, c.y + h * 0.08f}, h, IM_COL32(0, 0, 0, static_cast<int>(200 * alpha)), IM_COL32(0, 0, 0, 0));
	const int v0 = dl->VtxBuffer.Size;
	Detail::Shape(dl, k, c, h, IM_COL32(255, 255, 255, static_cast<int>(255 * alpha)), hole);
	const int v1 = dl->VtxBuffer.Size;
	// Gold top-left -> bronze bottom-right, kept alpha (the cut-outs keep their own colour: they are re-coloured below).
	const Rgb top {std::min(255, kGoldHi.r + static_cast<int>(30 * bright)), std::min(255, kGoldHi.g + static_cast<int>(30 * bright)), std::min(255, kGoldHi.b + static_cast<int>(25 * bright))};
	ImGui::ShadeVertsLinearColorGradientKeepAlpha(dl, v0, v1, {c.x - h, c.y - h}, {c.x + h, c.y + h}, C(top, 1), C(kBronze, 1));
	// The cut-outs (drawn in `hole`) were shaded too: put their colour back (they are the vertices with hole's alpha).
	const int hole_a = static_cast<int>((hole >> IM_COL32_A_SHIFT) & 0xff);
	// (trucos13) A metallic shimmer: a bright diagonal band sweeping across the metal every few seconds (stronger when
	// selected); the selected icon also sways a few degrees.
	const float cyc  = std::fmod(time * 0.35f + c.x * 0.0007f, 1.0f) * 3.2f - 1.1f; // band position along the diagonal, -1.1..2.1
	const float sh_k = 0.45f + 0.55f * hot;
	const float ang  = 0.07f * hot * std::sin(time * 1.6f), ca = std::cos(ang), sa = std::sin(ang);
	for (int i = v0; i < v1; i++) {
		ImDrawVert& vx = dl->VtxBuffer[i];
		const int   va = static_cast<int>((vx.col >> IM_COL32_A_SHIFT) & 0xff);
		const bool  is_hole = hole_a > 0 && hole_a != static_cast<int>(255 * alpha) && va == hole_a;
		if (is_hole) vx.col = hole;
		else {
			const float d = ((vx.pos.x - c.x) + (vx.pos.y - c.y)) / (2.0f * h) * 0.5f + 0.5f; // 0 top-left .. 1 bottom-right
			const float b = std::max(0.0f, 1.0f - std::fabs(d - cyc) * 5.0f) * sh_k;
			if (b > 0.0f) {
				int r = (vx.col >> IM_COL32_R_SHIFT) & 0xff, g = (vx.col >> IM_COL32_G_SHIFT) & 0xff, bb = (vx.col >> IM_COL32_B_SHIFT) & 0xff;
				r = std::min(255, r + static_cast<int>((255 - r) * b * 0.8f));
				g = std::min(255, g + static_cast<int>((250 - g) * b * 0.8f));
				bb = std::min(255, bb + static_cast<int>((230 - bb) * b * 0.8f));
				vx.col = IM_COL32(r, g, bb, va);
			}
		}
		if (ang != 0.0f) {
			const float x = vx.pos.x - c.x, y = vx.pos.y - c.y;
			vx.pos = {c.x + x * ca - y * sa, c.y + x * sa + y * ca};
		}
	}
}
// (trucos13) Sparks leaping from a selected medallion: short arcs outwards with gravity, born at the rim, white-gold,
// fading; a deterministic cycle per spark (no state).
inline void Sparks(ImDrawList* dl, ImVec2 c, float r, float k, float alpha, float time) {
	if (k <= 0.01f) return;
	for (int i = 0; i < 14; i++) {
		const float period = 0.9f + 0.7f * Hash(i * 3);
		const float lt     = std::fmod(time + Hash(i * 3 + 1) * period, period) / period; // 0..1 life
		const float an     = -1.5708f + (Hash(i * 3 + 2) - 0.5f) * 3.0f + std::floor((time + Hash(i * 3 + 1) * period) / period) * 0.9f;
		const float sp     = r * (1.2f + 1.0f * Hash(i * 7));
		const ImVec2 p {c.x + std::cos(an) * (r + sp * lt), c.y + std::sin(an) * (r + sp * lt) + r * 1.2f * lt * lt};
		const float  a = alpha * k * (1.0f - lt) * (0.6f + 0.4f * std::sin(time * 30.0f + i));
		dl->AddCircleFilled(p, 4.0f, C(kCandle, a * 0.18f), 10);
		dl->AddCircleFilled(p, 1.5f, IM_COL32(255, 240, 200, static_cast<int>(255 * std::clamp(a, 0.0f, 1.0f))), 6);
	}
}
// A medallion: bevelled ring (light from the upper left), a deep recess, the metal icon; `hot` 0..1 warms it with a
// breathing candle halo, `soul` 0..1 adds a pale soul glow (an active cheat, the chosen category).
inline void Medallion(ImDrawList* dl, Icon k, ImVec2 c, float r, float alpha, float hot, float time, float soul = 0.0f) {
	c = {std::round(c.x), std::round(c.y)};
	const float br = Breath(time, 1.5f, c.x * 0.01f);
	if (hot > 0.01f)
		for (int g = 5; g >= 1; g--) dl->AddCircleFilled(c, r * (1.0f + 0.09f * g), C(kCandle, alpha * 0.025f * hot * (0.7f + 0.3f * br)), 48);
	if (soul > 0.01f)
		for (int g = 5; g >= 1; g--) dl->AddCircleFilled(c, r * (0.95f + 0.1f * g), C(kSoul, alpha * 0.03f * soul * (0.6f + 0.4f * br)), 48);
	dl->AddCircleFilled({c.x + r * 0.06f, c.y + r * 0.1f}, r * 1.02f, IM_COL32(0, 0, 0, static_cast<int>(170 * alpha)), 48); // drop shadow
	// Bevel ring: a gradient disc (light upper left), then the recess inside it.
	const int v0 = dl->VtxBuffer.Size;
	dl->AddCircleFilled(c, r, IM_COL32(255, 255, 255, static_cast<int>(255 * alpha)), 64);
	ImGui::ShadeVertsLinearColorGradientKeepAlpha(dl, v0, dl->VtxBuffer.Size, {c.x - r, c.y - r}, {c.x + r, c.y + r},
	                                              C(hot > 0.5f ? kGoldHi : kGold, 1), C(kInk, 1));
	const int v1 = dl->VtxBuffer.Size;
	dl->AddCircleFilled(c, r * 0.86f, IM_COL32(255, 255, 255, static_cast<int>(255 * alpha)), 64);
	ImGui::ShadeVertsLinearColorGradientKeepAlpha(dl, v1, dl->VtxBuffer.Size, {c.x - r, c.y - r}, {c.x + r, c.y + r}, C(kInk, 1), C(kStoneHi, 1));
	dl->AddCircle(c, r * 0.86f, IM_COL32(0, 0, 0, static_cast<int>(200 * alpha)), 64, std::max(1.0f, r * 0.03f));
	dl->AddCircle(c, r * 0.78f, C(kGold, alpha * (0.25f + 0.35f * hot)), 64, 1.0f);
	const ImU32 recess = C(kInk, alpha * 0.98f);
	MetalIcon(dl, k, c, r * 1.05f, alpha, hot + soul, recess, time, hot);
	// (trucos13) A pulse of glow on the rim when selected.
	if (hot > 0.01f) dl->AddCircle(c, r * (1.0f + 0.05f * Breath(time, 3.0f)), C(kGoldHi, alpha * hot * 0.35f * Breath(time, 3.0f)), 64, 2.0f);
}


// ======================= (trucos12) the spectacle layer =======================
// Embers and soul motes rising with a bright core and wide faint halo (ImGui has no additive blend: the "add" look is
// a white-hot core over two warm halos), flickering; some drift sideways in the heat.
inline void Embers(ImDrawList* dl, ImVec2 a, ImVec2 b, float alpha, float time, int count = 70) {
	const float w = b.x - a.x, h = b.y - a.y, u = std::min(w / 1920.0f, h / 1080.0f);
	for (int i = 0; i < count; i++) {
		const bool  soul = (i % 3) == 0;
		const float sp   = (soul ? 0.012f : 0.03f) + 0.03f * Hash(i * 5 + 1);
		const float ph   = std::fmod(Hash(i * 5 + 2) + time * sp, 1.0f);
		const float y    = a.y + h * (1.05f - ph * 1.1f);
		const float x    = a.x + w * Hash(i * 5) + std::sin(time * (0.4f + Hash(i)) + i) * 40.0f * u * ph;
		const float life = std::sin(ph * 3.14159f); // born dim, brightest mid-flight, dies out
		const float fl   = 0.55f + 0.45f * std::sin(time * (6.0f + 5.0f * Hash(i * 9)) + i);
		const float k    = alpha * life * fl;
		const float r    = (soul ? 1.4f : 1.0f) * (0.8f + 1.4f * Hash(i * 7)) * u;
		const Rgb   hue  = soul ? kSoul : kCandle;
		dl->AddCircleFilled({x, y}, r * 7.0f, C(hue, k * 0.035f), 14);
		dl->AddCircleFilled({x, y}, r * 3.0f, C(hue, k * 0.16f), 10);
		dl->AddCircleFilled({x, y}, r, soul ? C(kSoul, k) : IM_COL32(255, 236, 200, static_cast<int>(255 * std::clamp(k, 0.0f, 1.0f))), 8);
	}
}
// Volumetric fog in three layers drifting at different speeds (slow parallax), denser low on the screen.
inline void FogLayers(ImDrawList* dl, ImVec2 a, ImVec2 b, float alpha, float time) {
	const float w = b.x - a.x, h = b.y - a.y, u = std::min(w / 1920.0f, h / 1080.0f);
	for (int layer = 0; layer < 3; layer++) {
		const float speed = 0.004f * (layer + 1), depth = 0.5f + 0.25f * layer;
		for (int i = 0; i < 9; i++) {
			const int   s  = layer * 31 + i;
			const float x  = a.x + w * (std::fmod(Hash(s) + time * speed + 3.0f, 1.4f) - 0.2f);
			const float y  = a.y + h * (0.45f + 0.5f * Hash(s + 7)) + std::sin(time * 0.2f + s) * 10.0f * u;
			const float rx = (300.0f + 400.0f * Hash(s + 3)) * u * depth, ry = rx * (0.22f + 0.1f * Hash(s + 4));
			for (int k = 5; k >= 1; k--) dl->AddEllipseFilled({x, y}, {rx * k / 5.0f, ry * k / 5.0f}, C(layer == 2 ? kStoneHi : kStone, alpha * 0.022f), 0.0f, 36);
		}
	}
}
// Gold with a bloom: faint widening halos around a thin stroke.
inline void BloomLine(ImDrawList* dl, ImVec2 p0, ImVec2 p1, float alpha, float thick) {
	dl->AddLine(p0, p1, C(kGoldHi, alpha * 0.05f), thick * 7.0f);
	dl->AddLine(p0, p1, C(kGoldHi, alpha * 0.12f), thick * 3.5f);
	dl->AddLine(p0, p1, C(kGoldHi, alpha), thick);
}
// Gothic filigree for a corner: a pointed arch stem, two curling scrolls, thorns, a small rosette. `sx`, `sy` = +-1
// (direction into the panel). Size s (px).
inline void Rosette(ImDrawList* dl, ImVec2 c, float r, float alpha, float time) {
	for (int g = 3; g >= 1; g--) dl->AddCircleFilled(c, r * (1.0f + 0.35f * g), C(kGoldHi, alpha * 0.03f), 32);
	for (int i = 0; i < 8; i++) {
		const float an = i * 0.7854f + time * 0.05f;
		const ImVec2 tip {c.x + std::cos(an) * r, c.y + std::sin(an) * r};
		const ImVec2 l {c.x + std::cos(an - 0.3f) * r * 0.45f, c.y + std::sin(an - 0.3f) * r * 0.45f};
		const ImVec2 rr {c.x + std::cos(an + 0.3f) * r * 0.45f, c.y + std::sin(an + 0.3f) * r * 0.45f};
		const ImVec2 petal[3] = {l, tip, rr};
		dl->AddConvexPolyFilled(petal, 3, C(i % 2 ? kGold : kGoldHi, alpha * 0.9f));
	}
	dl->AddCircleFilled(c, r * 0.32f, C(kInk, alpha), 20);
	dl->AddCircle(c, r * 0.32f, C(kGoldHi, alpha), 20, 1.2f);
	dl->AddCircleFilled(c, r * 0.12f, C(kSoul, alpha * (0.6f + 0.4f * Breath(time, 2.0f))), 12);
}
inline void Filigree(ImDrawList* dl, ImVec2 o, float s, float sx, float sy, float alpha, float time, float unit) {
	const auto  P  = [&](float x, float y) { return ImVec2 {o.x + sx * x * s, o.y + sy * y * s}; };
	const float t1 = std::max(1.0f, unit * 1.3f);
	const ImU32 g  = C(kGold, alpha), gh = C(kGoldHi, alpha);
	// Two bars along the edges with pointed ends.
	BloomLine(dl, P(0.0f, 0.0f), P(1.0f, 0.0f), alpha, t1);
	BloomLine(dl, P(0.0f, 0.0f), P(0.0f, 1.0f), alpha, t1);
	dl->AddTriangleFilled(P(1.0f, -0.035f), P(1.0f, 0.035f), P(1.12f, 0.0f), gh);
	dl->AddTriangleFilled(P(-0.035f, 1.0f), P(0.035f, 1.0f), P(0.0f, 1.12f), gh);
	// Scrolls curling inwards from the bars.
	for (int k = 0; k < 2; k++) {
		const bool  hx = k == 0;
		const float m  = 0.55f;
		const ImVec2 st = hx ? P(m, 0.0f) : P(0.0f, m);
		dl->PathLineTo(st);
		if (hx) dl->PathBezierCubicCurveTo(P(m, 0.18f), P(m - 0.22f, 0.3f), P(m - 0.3f, 0.16f), 16);
		else dl->PathBezierCubicCurveTo(P(0.18f, m), P(0.3f, m - 0.22f), P(0.16f, m - 0.3f), 16);
		dl->PathStroke(g, 0, t1);
		dl->AddCircleFilled(hx ? P(m - 0.3f, 0.16f) : P(0.16f, m - 0.3f), s * 0.025f, gh, 10);
		// thorns along the bar
		for (int j = 1; j <= 3; j++) {
			const float f = 0.18f * j + 0.05f;
			if (hx) dl->AddTriangleFilled(P(f - 0.02f, 0.0f), P(f + 0.02f, 0.0f), P(f, 0.07f), g);
			else dl->AddTriangleFilled(P(0.0f, f - 0.02f), P(0.0f, f + 0.02f), P(0.07f, f), g);
		}
	}
	// The pointed arch in the corner and its rosette.
	dl->PathLineTo(P(0.0f, 0.42f));
	dl->PathBezierQuadraticCurveTo(P(0.05f, 0.08f), P(0.42f, 0.0f), 14);
	dl->PathStroke(g, 0, t1);
	Rosette(dl, P(0.17f, 0.17f), s * 0.1f, alpha, time);
}
// The header crest: a rosette between two mirrored scroll wings, over the title.
inline void Crest(ImDrawList* dl, ImVec2 c, float s, float alpha, float time, float unit) {
	const float t1 = std::max(1.0f, unit * 1.3f);
	for (int side = -1; side <= 1; side += 2) {
		const auto P = [&](float x, float y) { return ImVec2 {c.x + side * x * s, c.y + y * s}; };
		dl->PathLineTo(P(0.12f, 0.0f));
		dl->PathBezierCubicCurveTo(P(0.35f, -0.25f), P(0.6f, 0.2f), P(0.85f, -0.05f), 20);
		dl->PathStroke(C(kGold, alpha), 0, t1);
		dl->PathLineTo(P(0.3f, -0.05f));
		dl->PathBezierCubicCurveTo(P(0.45f, 0.18f), P(0.62f, 0.12f), P(0.62f, -0.02f), 14);
		dl->PathStroke(C(kGold, alpha * 0.8f), 0, t1);
		dl->AddTriangleFilled(P(0.85f, -0.09f), P(0.85f, -0.01f), P(1.0f, -0.05f), C(kGoldHi, alpha));
		dl->AddCircleFilled(P(0.62f, -0.02f), s * 0.018f, C(kGoldHi, alpha), 10);
	}
	Rosette(dl, c, s * 0.1f, alpha, time);
}
// A god-ray: a warm vertical beam over a rect (top wide and faint, focused down), breathing.
inline void GodRay(ImDrawList* dl, ImVec2 a, ImVec2 b, float k, float alpha, float time) {
	if (k <= 0.001f) return;
	const float cx = (a.x + b.x) * 0.5f, w = b.x - a.x, br = 0.8f + 0.2f * Breath(time, 1.3f);
	const float top = a.y - (b.y - a.y) * 0.9f;
	for (int i = 0; i < 3; i++) {
		const float hw0 = w * (0.18f + 0.12f * i), hw1 = w * (0.42f + 0.1f * i);
		const ImVec2 q[4] = {{cx - hw0, top}, {cx + hw0, top}, {cx + hw1, b.y}, {cx - hw1, b.y}};
		const int v0 = dl->VtxBuffer.Size;
		dl->AddConvexPolyFilled(q, 4, IM_COL32(255, 255, 255, 255));
		ImGui::ShadeVertsLinearColorGradientKeepAlpha(dl, v0, dl->VtxBuffer.Size, {cx, top}, {cx, b.y}, C(kCandle, 0.0f), C(kCandle, 1.0f));
		const int a8 = static_cast<int>(255 * std::clamp(alpha * k * br * (0.05f - 0.012f * i), 0.0f, 1.0f));
		for (int v = v0; v < dl->VtxBuffer.Size; v++) {
			ImU32& col = dl->VtxBuffer[v].col;
			const int ca = static_cast<int>((col >> IM_COL32_A_SHIFT) & 0xff);
			col = (col & ~IM_COL32_A_MASK) | (static_cast<ImU32>(ca * a8 / 255) << IM_COL32_A_SHIFT);
		}
	}
}
// A pulsing soul aura around a rect.
inline void SoulAura(ImDrawList* dl, ImVec2 a, ImVec2 b, float k, float alpha, float time) {
	if (k <= 0.001f) return;
	const float p = 0.6f + 0.4f * Breath(time, 2.2f);
	for (int g = 5; g >= 1; g--) {
		const float e = g * 5.0f;
		dl->AddRect({a.x - e, a.y - e}, {b.x + e, b.y + e}, C(kSoul, alpha * k * p * 0.035f), e, 0, 5.0f);
	}
}
// Title letter by letter: each glyph fades in with a short bright flash, `t` seconds since the page opened.
inline float TitleReveal(ImDrawList* dl, ImFont* f, float size, ImVec2 at, const char* text, float alpha, float t, float tracking_em = 0.10f) {
	const float tr = std::round(size * tracking_em);
	float       x  = std::round(at.x);
	int         i  = 0;
	for (const char* p = text; *p; i++) {
		const int   n  = CharLen(static_cast<unsigned char>(*p));
		const float lt = std::clamp((t - i * 0.045f) / 0.35f, 0.0f, 1.0f);
		const float fl = std::max(0.0f, 1.0f - std::fabs(lt - 0.5f) * 2.0f); // the flash, peaking mid-fade
		if (lt > 0.0f) {
			dl->AddText(f, size, {x + 2, std::round(at.y) + 2}, IM_COL32(0, 0, 0, static_cast<int>(200 * alpha * lt)), p, p + n);
			if (fl > 0.01f)
				for (int g = 2; g >= 1; g--)
					for (int d = 0; d < 4; d++)
						dl->AddText(f, size, {x + (d % 2 ? g : -g), std::round(at.y) + (d / 2 ? g : -g)}, C(kGoldHi, alpha * fl * 0.15f), p, p + n);
			dl->AddText(f, size, {x, std::round(at.y)}, C(kBone, alpha * lt), p, p + n);
		}
		x += f->CalcTextSizeA(size, FLT_MAX, 0.0f, p, p + n).x + tr;
		p += n;
	}
	return x - at.x - tr;
}
// A light sweep across a rect (page change): a slanted bright band moving left to right as p goes 0..1.
inline void LightSweep(ImDrawList* dl, ImVec2 a, ImVec2 b, float p, float alpha) {
	if (p <= 0.0f || p >= 1.0f) return;
	const float w = b.x - a.x, x = a.x - w * 0.3f + p * w * 1.6f, bw = w * 0.12f;
	const ImVec2 q0[4] = {{x - bw, a.y}, {x, a.y}, {x - bw * 0.6f, b.y}, {x - bw * 1.6f, b.y}};
	const ImVec2 q1[4] = {{x, a.y}, {x + bw, a.y}, {x + bw * 0.4f - bw, b.y}, {x - bw * 0.6f, b.y}};
	const float  k     = alpha * std::sin(p * 3.14159f) * 0.10f;
	dl->AddQuadFilled(q0[0], q0[1], q0[2], q0[3], C(kGoldHi, k * 0.5f));
	dl->AddQuadFilled(q1[0], q1[1], q1[2], q1[3], C(kGoldHi, k));
}
// The shop's chosen item: a large medallion with a halo whose light rays turn slowly.
inline void Showcase(ImDrawList* dl, Icon k, ImVec2 c, float r, float alpha, float time) {
	for (int i = 0; i < 16; i++) {
		const float an = i * 0.3927f + time * 0.15f, l = r * (1.9f + 0.4f * Hash(i)), wd = 0.05f;
		const ImVec2 q[3] = {c, {c.x + std::cos(an - wd) * l, c.y + std::sin(an - wd) * l}, {c.x + std::cos(an + wd) * l, c.y + std::sin(an + wd) * l}};
		const int v0 = dl->VtxBuffer.Size;
		dl->AddConvexPolyFilled(q, 3, C(kGoldHi, alpha * 0.10f));
		ImGui::ShadeVertsLinearColorGradientKeepAlpha(dl, v0, dl->VtxBuffer.Size, c, {c.x + std::cos(an) * l, c.y + std::sin(an) * l}, C(kGoldHi, 1), C(kInk, 1));
		for (int v = v0; v < dl->VtxBuffer.Size; v++) {
			const ImVec2 d {dl->VtxBuffer[v].pos.x - c.x, dl->VtxBuffer[v].pos.y - c.y};
			const float  f = 1.0f - std::min(1.0f, std::sqrt(d.x * d.x + d.y * d.y) / l);
			ImU32&       col = dl->VtxBuffer[v].col;
			col = (col & ~IM_COL32_A_MASK) | (static_cast<ImU32>(255 * alpha * 0.22f * f) << IM_COL32_A_SHIFT);
		}
	}
	for (int g = 6; g >= 1; g--) dl->AddCircleFilled(c, r * (1.0f + 0.12f * g), C(kSoul, alpha * 0.025f * (0.7f + 0.3f * Breath(time, 1.4f))), 48);
	Medallion(dl, k, c, r, alpha, 1.0f, time, 0.8f);
}

// ======================= (trucos13) typography and ash =======================
// A rich title: dark outline, a thin gold rim, the letters shaded top bone-white -> bottom warm gold (vertex gradient on
// the glyph quads), revealed letter by letter (t: seconds since shown; t >= 10: all shown).
inline float GoldTitle(ImDrawList* dl, ImFont* f, float size, ImVec2 at, const char* text, float alpha, float t, float tracking_em = 0.10f) {
	const float tr = std::round(size * tracking_em);
	float       x  = std::round(at.x);
	const float y  = std::round(at.y);
	int         i  = 0;
	for (const char* p = text; *p; i++) {
		const int   n  = CharLen(static_cast<unsigned char>(*p));
		const float lt = std::clamp((t - i * 0.045f) / 0.35f, 0.0f, 1.0f);
		const float fl = std::max(0.0f, 1.0f - std::fabs(lt - 0.5f) * 2.0f);
		if (lt > 0.0f) {
			const float a = alpha * lt;
			dl->AddText(f, size, {x + 3, y + 4}, IM_COL32(0, 0, 0, static_cast<int>(170 * a)), p, p + n); // drop shadow
			for (int d = 0; d < 8; d++) {                                                                  // dark outline
				const float ox = d < 3 ? -1.5f : d < 5 ? 0.0f : 1.5f, oy = (d == 0 || d == 5) ? -1.5f : (d == 2 || d == 7) ? 1.5f : (d == 3 ? -1.5f : d == 4 ? 1.5f : 0.0f);
				dl->AddText(f, size, {x + ox, y + oy}, IM_COL32(8, 6, 4, static_cast<int>(230 * a)), p, p + n);
			}
			for (int d = 0; d < 4; d++) dl->AddText(f, size, {x + (d % 2 ? 0.8f : -0.8f), y + (d / 2 ? 0.8f : -0.8f)}, C(kGold, a * 0.9f), p, p + n); // gold rim
			const int v0 = dl->VtxBuffer.Size;
			dl->AddText(f, size, {x, y}, IM_COL32(255, 255, 255, static_cast<int>(255 * a)), p, p + n);
			ImGui::ShadeVertsLinearColorGradientKeepAlpha(dl, v0, dl->VtxBuffer.Size, {x, y + size * 0.2f}, {x, y + size * 0.95f}, C(kBone, 1), C(kGoldHi, 1));
			if (fl > 0.01f)
				for (int d = 0; d < 4; d++) dl->AddText(f, size, {x + (d % 2 ? 2.0f : -2.0f), y + (d / 2 ? 2.0f : -2.0f)}, C(kGoldHi, a * fl * 0.2f), p, p + n);
		}
		x += f->CalcTextSizeA(size, FLT_MAX, 0.0f, p, p + n).x + tr;
		p += n;
	}
	return x - at.x - tr;
}
// An illuminated initial: the first letter large in a dark square with a gold double frame, corner dots and a faint
// red-brown ground (as a manuscript capital), then the rest of the title in GoldTitle. Returns the total width.
inline float IlluminatedTitle(ImDrawList* dl, ImFont* f, float size, ImVec2 at, const char* text, float alpha, float t, float time) {
	if (*text == 0) return 0.0f;
	const int    n   = CharLen(static_cast<unsigned char>(*text));
	const float  cs  = std::round(size * 1.55f), box = std::round(size * 1.5f);
	const ImVec2 b0 {std::round(at.x), std::round(at.y - (box - size) * 0.5f)}, b1 {b0.x + box, b0.y + box};
	const float  a   = alpha * std::clamp(t / 0.3f, 0.0f, 1.0f);
	dl->AddRectFilledMultiColor(b0, b1, IM_COL32(48, 18, 14, static_cast<int>(230 * a)), IM_COL32(30, 12, 10, static_cast<int>(230 * a)),
	                            IM_COL32(14, 8, 8, static_cast<int>(240 * a)), IM_COL32(24, 10, 9, static_cast<int>(240 * a)));
	dl->AddRect(b0, b1, C(kGoldHi, a), 0, 0, 1.5f);
	dl->AddRect({b0.x + 4, b0.y + 4}, {b1.x - 4, b1.y - 4}, C(kGold, a * 0.7f), 0, 0, 1.0f);
	for (int q = 0; q < 4; q++) Rhombus(dl, {q & 1 ? b1.x : b0.x, q & 2 ? b1.y : b0.y}, 4.0f, C(kGoldHi, a));
	for (int i = 0; i < 6; i++) dl->AddCircleFilled({b0.x + box * (0.18f + 0.13f * i), b1.y - 8}, 1.3f, C(kGold, a * 0.8f), 6);
	const ImVec2 csz = f->CalcTextSizeA(cs, FLT_MAX, 0.0f, text, text + n);
	const ImVec2 cp {std::round(b0.x + (box - csz.x) * 0.5f), std::round(b0.y + (box - csz.y) * 0.5f)};
	std::string first(text, n);
	GoldTitle(dl, f, cs, cp, first.c_str(), a, 10.0f, 0.0f);
	const float rest = GoldTitle(dl, f, size, {b1.x + std::round(size * 0.25f), at.y}, text + n, alpha, t - 0.15f);
	return box + std::round(size * 0.25f) + rest;
}
// Ash: grey flakes and a few embers drifting on a turbulent (curl-noise) wind; they flare and die, shy away from a point
// (the cursor / the selection) and scatter outwards on Burst (opening, a page change). State kept across frames.
struct Ash {
	static constexpr int N = 160;
	float x[N] {}, y[N] {}, vx[N] {}, vy[N] {}, life[N] {}, size[N] {};
	bool  ember[N] {};
	bool  init = false;
	float w = 0, h = 0;
	void Spawn(int i, bool anywhere) {
		x[i]     = Hash(i * 13 + static_cast<int>(life[i] * 997)) * w;
		y[i]     = anywhere ? Hash(i * 17 + static_cast<int>(x[i])) * h : h + 10.0f;
		vx[i]    = vy[i] = 0;
		life[i]  = 0.0f;
		size[i]  = 0.8f + 2.2f * Hash(i * 19 + static_cast<int>(y[i]));
		ember[i] = Hash(i * 23 + static_cast<int>(x[i])) < 0.18f;
	}
	void Burst(ImVec2 c, float strength) {
		for (int i = 0; i < N; i++) {
			const float dx = x[i] - c.x, dy = y[i] - c.y, d = std::sqrt(dx * dx + dy * dy) + 30.0f;
			vx[i] += dx / d * strength;
			vy[i] += dy / d * strength;
		}
	}
	// One step: dt seconds, `repel` a point (or {-1e9,..}), the screen size.
	void Update(float dt, float time, ImVec2 repel, float sw, float sh) {
		if (!init || w != sw || h != sh) {
			w = sw, h = sh, init = true;
			for (int i = 0; i < N; i++) {
				life[i] = Hash(i * 31);
				Spawn(i, true);
				life[i] = Hash(i * 37);
			}
		}
		const float u = std::min(w / 1920.0f, h / 1080.0f);
		for (int i = 0; i < N; i++) {
			// Curl of a smooth potential psi(x, y, t): v = (d psi/dy, -d psi/dx), here from two sine layers.
			const float px = x[i] * 0.0023f / u, py = y[i] * 0.0023f / u, tt = time * 0.12f;
			const float dpx = std::cos(px + tt) * std::cos(py * 1.3f - tt) * 1.0f + std::cos(px * 2.1f - tt * 1.7f) * 0.5f;
			const float dpy = -std::sin(px + tt) * std::sin(py * 1.3f - tt) * 1.3f + std::cos(py * 2.3f + tt) * 0.6f;
			const float fx = dpy * 22.0f * u, fy = -dpx * 22.0f * u - (ember[i] ? 26.0f : 9.0f) * u; // wind + rise (embers rise faster)
			float ax = (fx - vx[i]) * 1.5f, ay = (fy - vy[i]) * 1.5f;
			const float rx = x[i] - repel.x, ry = y[i] - repel.y, rd2 = rx * rx + ry * ry, rr = 220.0f * u;
			if (rd2 < rr * rr && rd2 > 1.0f) {
				const float d = std::sqrt(rd2), k = (1.0f - d / rr) * 900.0f * u;
				ax += rx / d * k, ay += ry / d * k;
			}
			vx[i] += ax * dt, vy[i] += ay * dt;
			x[i] += vx[i] * dt, y[i] += vy[i] * dt;
			life[i] += dt / (ember[i] ? 5.0f : 12.0f);
			if (life[i] >= 1.0f || y[i] < -20.0f || x[i] < -40.0f || x[i] > w + 40.0f) Spawn(i, Hash(i * 41 + static_cast<int>(time * 7.0f)) < 0.6f); // (most reborn anywhere, faded in: the screen stays evenly filled)
		}
	}
	void Draw(ImDrawList* dl, float alpha, float time) const {
		if (!init) return;
		const float u = std::min(w / 1920.0f, h / 1080.0f);
		for (int i = 0; i < N; i++) {
			const float fade = std::sin(std::clamp(life[i], 0.0f, 1.0f) * 3.14159f);
			const float r    = size[i] * u;
			if (ember[i]) {
				const float fl = 0.5f + 0.5f * std::sin(time * (7.0f + 4.0f * Hash(i)) + i); // flares and dims
				const float a  = alpha * fade * fl;
				dl->AddCircleFilled({x[i], y[i]}, r * 5.0f, C(kCandle, a * 0.05f), 12);
				dl->AddCircleFilled({x[i], y[i]}, r * 1.8f, C(kCandle, a * 0.35f), 8);
				dl->AddCircleFilled({x[i], y[i]}, r * 0.8f, IM_COL32(255, 230, 190, static_cast<int>(255 * std::clamp(a, 0.0f, 1.0f))), 6);
			} else {
				// A grey flake: a small tilted quad, tumbling.
				const float an = time * (0.8f + Hash(i)) + i, ca = std::cos(an) * r * 1.6f, sa = std::sin(an) * r * 0.7f;
				const ImU32 col = IM_COL32(150, 146, 140, static_cast<int>(255 * std::clamp(alpha * fade * 0.45f, 0.0f, 1.0f)));
				dl->AddQuadFilled({x[i] - ca, y[i] - sa}, {x[i] + sa, y[i] - ca * 0.5f}, {x[i] + ca, y[i] + sa}, {x[i] - sa, y[i] + ca * 0.5f}, col);
			}
		}
	}
};

} // namespace DsStyle
