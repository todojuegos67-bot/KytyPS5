#ifndef EMULATOR_SRC_LOCAL_HUD_ANCHOR_H_
#define EMULATOR_SRC_LOCAL_HUD_ANCHOR_H_
// KYTY_RENDER_ASPECT (Demon's Souls, src/loader/demonsSoulsRenderSize.h) widens the game's 3D, and the presenter widens
// its whole 16:9 output to that aspect, the HUD with it. Here the HUD keeps its proportions. The game draws it into its
// 3840x2160 interface canvas with three vertex shaders, in batches: one draw holds the text of the health bars and the
// souls, another the item names and the boss's name. In a frame showing the 3D, each vertex of such a draw is placed in
// one of the HUD's areas (Areas: the top left - health, magic, stamina -, the bottom left - equipment, items -, the top
// right - souls, the items picked up under them - and the bottom middle - boss health, prompts) when it comes after the
// HUD's first draw in the frame,
// its round icon in the top left corner (menus draw none; the markers over the world come before it, often inside the
// areas). A draw wholly in one area gets its viewport squeezed back to 16:9 around the area's side (the left edge, the
// right edge, the centre); a draw over
// several is drawn once for each, with its viewport and a scissor around where the area then lies (the areas stay
// apart, squeezed), and what it has outside them (a dialogue's text in the same batch) once for each rectangle of the
// canvas outside them, as it was. Widened again, each element stands at its side of the screen as drawn for 16:9. The
// rest - menus, fades, markers over the world (lock-on, enemy health) - stays as it was (stretched, the markers on
// their targets).
// With the wide output (DemonsSoulsRenderSize::WideOutputWidth: the game's output as wide as the 3D, 5120x2160 at 21:9)
// nothing is widened after: the game lays its interface out at 3840x2160 against the canvas's left edge. In a frame
// showing the 3D the HUD's areas are moved to their sides then (unstretched as drawn), and every other draw into the
// canvas - menus, fades, markers - is widened to the canvas (StretchScope): the same picture as the 16:9 output widened.
#include "common/common.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/shader/shader.h"
#include "loader/demonsSoulsRenderSize.h"

#include "draw-state-observer.h"
#include "frame-gen.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <utility>

namespace Libs::Graphics::HudAnchor {

// The interface's vertex shaders (01.007.000): their positions float x, y (z), as clip coordinates in the V# of the
// user data's first four words, or (PixelShader) as canvas pixels in the first V# of a table two pointers away.
constexpr std::array<uint64_t, 2> ClipShaders {0x7e880a66bf1b69c6ULL, 0xb2f062027869d4eeULL};
constexpr uint64_t                PixelShader = 0xbd39a2eba2d20d8bULL;
constexpr uint32_t                CanvasWidth = 3840, CanvasHeight = 2160;
constexpr uint32_t                MaxVertices = 8192;

// The HUD's areas in clip coordinates (y up; frames of elements reach a little past the canvas's edges) and the side
// each is held to (-1 left, 0 centre, 1 right).
struct Area {
	float x0, x1, y0, y1, side;
};
constexpr std::array<Area, 5> Areas {{
    {-1.1f, 0.3f, 0.6f, 1.1f, -1.0f},     // health, magic, stamina, their buffs
    {-1.1f, -0.55f, -1.1f, -0.25f, -1.0f}, // equipment, item counts and names
    {0.4f, 1.1f, 0.7f, 1.1f, 1.0f},        // souls
    {-0.55f, 0.55f, -0.95f, -0.35f, 0.0f}, // boss health, prompts
    {0.3f, 1.1f, 0.3f, 0.7f, 1.0f},        // the items picked up, under the souls (their top reaches into the souls')
}};
constexpr uint32_t Outside = 1u << Areas.size(); // (Classify: a vertex in none of them)

// The HUD's round icon in the top left corner (with its frame) and nothing else: the HUD shows.
constexpr Area Icon {-1.02f, -0.85f, 0.78f, 1.05f, -1.0f};

// The pixel shader of the equipment's icons masks them by their screen position: a constant matrix (rows 0 and 1 of the
// V# in its user data's words 16-19, x y _ w each) maps the fragment's window x, y to the slot's mask. Squeezed, its x
// coefficients are made to undo the squeeze (MaskPatch); restored when the HUD is drawn as it is.
constexpr uint64_t MaskedShader = 0x4d86d2e2bb1b49cdULL;

struct MaskPatch {
	uint64_t address = 0;              // row 0 (row 1 follows at the V#'s stride)
	uint32_t stride  = 0;
	float    original[2][4] {}, patched[2][4] {};
};

struct State {
	std::array<std::pair<uint64_t, uint64_t>, 256> hashes {}; // shader address -> hash (direct mapped)
	uint64_t icon_frame = 0;              // FrameGen::GuestFlips() + 1 of the frame the icon was last drawn in
	std::array<MaskPatch, 32> patches {};  // (by address, round robin)
	uint32_t                  next_patch = 0;
	bool                      patched    = false; // any patch outstanding
	uint64_t                  canvas     = 0;     // the wide output's interface canvas (an interface shader's target)
};
inline State g_state; // (the GPU thread's)

// The wide output's canvas width (0: none).
inline uint32_t WideWidth() {
	return ::Loader::DemonsSoulsRenderSize::WideOutputWidth();
}

// The width of the interface canvas the draw goes into: the game's 3840, or the wide output's (the performance mode's;
// its Cinematic mode keeps 3840); 0 for another target.
inline uint32_t CanvasWidthOf(const HW::Context& ctx) {
	const auto& target = ctx.GetRenderTarget(0);
	const auto  width  = target.attrib2.width + 1u;
	if (target.base.addr == 0 || target.attrib2.height + 1u != CanvasHeight) return 0;
	return width == CanvasWidth || (WideWidth() != 0 && width == WideWidth()) ? width : 0;
}

inline uint64_t HashAt(uint64_t address) {
	if (address == 0) return 0;
	auto& entry = g_state.hashes[(address >> 8u) % g_state.hashes.size()];
	if (entry.first != address) entry = {address, ShaderHashAt(address)};
	return entry.second;
}

inline uint64_t PixelShaderHash(const HW::PixelShaderInfo& ps) {
	return HashAt(ps.ps_regs.data_addr);
}

// The masked icon draw's matrix rows (address and stride), when readable.
inline bool MaskRows(const HW::PixelShaderInfo& ps, uint64_t& address, uint32_t& stride) {
	if (PixelShaderHash(ps) != MaskedShader || ps.ps_user_sgpr.count < 20) return false;
	address = ps.ps_user_sgpr.value[16] | (static_cast<uint64_t>(ps.ps_user_sgpr.value[17] & 0xffffu) << 32u);
	stride  = (ps.ps_user_sgpr.value[17] >> 16u) & 0x3fffu;
	return address != 0 && stride >= 16 && HostMemoryRangeIsReadable(address, stride + 16u);
}

inline void ReadRows(uint64_t address, uint32_t stride, float rows[2][4]) {
	std::memcpy(rows[0], reinterpret_cast<const void*>(address), 16);
	std::memcpy(rows[1], reinterpret_cast<const void*>(address + stride), 16);
}

inline void WriteRows(uint64_t address, uint32_t stride, const float rows[2][4]) {
	std::memcpy(reinterpret_cast<void*>(address), rows[0], 16);
	std::memcpy(reinterpret_cast<void*>(address + stride), rows[1], 16);
}

// A masked icon draw left as it is: rows this patched (and the game did not write again) get their values back.
inline void RestoreMask(const HW::PixelShaderInfo& ps) {
	uint64_t address = 0;
	uint32_t stride  = 0;
	if (!g_state.patched || !MaskRows(ps, address, stride)) return;
	for (auto& patch: g_state.patches) {
		if (patch.address != address) continue;
		float rows[2][4] {};
		ReadRows(address, stride, rows);
		if (std::memcmp(rows, patch.patched, sizeof(rows)) == 0) WriteRows(address, stride, patch.original);
		patch.address = 0;
	}
}

// A masked icon draw placed at window x' = scale * x + shift, so its mask is read at x = (x' - shift) / scale (from its
// own values: rows this patched before, the game not writing them again, keep theirs).
inline void PatchMask(const HW::PixelShaderInfo& ps, float scale, float shift) {
	uint64_t address = 0;
	uint32_t stride  = 0;
	if (!MaskRows(ps, address, stride)) return;
	auto& state = g_state;
	float rows[2][4] {};
	ReadRows(address, stride, rows);
	MaskPatch* patch = nullptr;
	for (auto& p: state.patches) {
		if (p.address == address) patch = &p;
	}
	if (patch == nullptr || patch->stride != stride || std::memcmp(rows, patch->patched, sizeof(rows)) != 0) {
		if (patch == nullptr) patch = &state.patches[state.next_patch++ % state.patches.size()];
		patch->address = address;
		patch->stride  = stride;
		std::memcpy(patch->original, rows, sizeof(rows));
	}
	std::memcpy(patch->patched, patch->original, sizeof(rows));
	for (auto& row: patch->patched) {
		const float x = row[0];
		row[0]        = x / scale;
		row[3] -= x * shift / scale;
	}
	WriteRows(address, stride, patch->patched);
	state.patched = true;
}

// A V# in four words: base address, stride.
inline bool ReadVsharp(const uint32_t* words, uint64_t& base, uint32_t& stride) {
	base   = words[0] | (static_cast<uint64_t>(words[1] & 0xffffu) << 32u);
	stride = (words[1] >> 16u) & 0x3fffu;
	return base != 0 && stride >= 12;
}

// The areas the draw's vertices lie in (a bit each, Outside for one in none), from its vertices in guest memory; 0 when
// it is no interface draw into the canvas in a frame showing the 3D, or its vertices are not readable.
inline uint32_t Classify(const HW::Context& ctx, const HW::VertexShaderInfo& vs, uint32_t first, uint32_t count) {
	if (::Loader::DemonsSoulsRenderSize::WideAspect() <= 0.0 || count == 0) return 0;
	const auto& target = ctx.GetRenderTarget(0);
	const auto  width  = CanvasWidthOf(ctx);
	if (width == 0) return 0;
	auto&          state = g_state;
	const uint64_t hash  = HashAt(vs.es_regs.data_addr);
	const bool     clip  = std::find(ClipShaders.begin(), ClipShaders.end(), hash) != ClipShaders.end();
	if (!clip && hash != PixelShader) return 0;
	// (An interface shader's target: the canvas the wide output's other draws into it are widened in, StretchScope.)
	if (width != CanvasWidth) state.canvas = target.base.addr;
	if (!FrameGen::SceneNoted()) return 0;
	const auto& viewport = ctx.GetScreenViewport().viewports[0];
	const auto& user   = vs.gs_user_sgpr;
	uint64_t    base   = 0;
	uint32_t    stride = 0;
	float       scale[2] {};
	if (clip) {
		if (user.count < 4 || !ReadVsharp(user.value, base, stride)) return 0;
	} else {
		// As its code reads them: the user data's first two words point to two pointers, the first to its V#s, the
		// second to constants whose words 170 and 171 scale the pixels (1/3840, 1/2160): clip = 2 * pixels * scale - 1,
		// y negated.
		if (user.count < 2) return 0;
		const uint64_t root = user.value[0] | (static_cast<uint64_t>(user.value[1] & 0xffffu) << 32u);
		uint64_t       pointers[2] {};
		if (root == 0 || !HostMemoryRangeIsReadable(root, sizeof(pointers))) return 0;
		std::memcpy(pointers, reinterpret_cast<const void*>(root), sizeof(pointers));
		const uint64_t vsharps = pointers[0] & 0xffffffffffffull, constants = (pointers[1] & 0xffffffffffffull) + 680u;
		uint32_t       words[4] {};
		if (vsharps == 0 || !HostMemoryRangeIsReadable(vsharps, sizeof(words)) ||
		    !HostMemoryRangeIsReadable(constants, sizeof(scale)))
			return 0;
		std::memcpy(words, reinterpret_cast<const void*>(vsharps), sizeof(words));
		std::memcpy(scale, reinterpret_cast<const void*>(constants), sizeof(scale));
		if (!ReadVsharp(words, base, stride) || !(scale[0] > 0.0f) || !(scale[1] > 0.0f)) return 0;
	}
	count = std::min(count, MaxVertices);
	if (!HostMemoryRangeIsReadable(base + static_cast<uint64_t>(first) * stride, static_cast<uint64_t>(count) * stride))
		return 0;
	uint32_t areas = 0;
	bool     icon  = true;
	for (uint32_t i = 0; i < count; ++i) {
		float xy[2] {};
		std::memcpy(xy, reinterpret_cast<const void*>(base + static_cast<uint64_t>(first + i) * stride), sizeof(xy));
		if (!std::isfinite(xy[0]) || !std::isfinite(xy[1])) return Outside;
		if (!clip) {
			xy[0] = 2.0f * xy[0] * scale[0] - 1.0f;
			xy[1] = 1.0f - 2.0f * xy[1] * scale[1];
		}
		// (The wide canvas: the window x the viewport puts it at, in the 3840 wide layout's clip coordinates.)
		if (width != CanvasWidth) xy[0] = 2.0f * (viewport.xoffset + xy[0] * viewport.xscale) / CanvasWidth - 1.0f;
		icon = icon && xy[0] >= Icon.x0 && xy[0] <= Icon.x1 && xy[1] >= Icon.y0 && xy[1] <= Icon.y1;
		uint32_t bit = Outside;
		for (uint32_t a = 0; a < Areas.size(); ++a) {
			const auto& area = Areas[a];
			if (xy[0] >= area.x0 && xy[0] <= area.x1 && xy[1] >= area.y0 && xy[1] <= area.y1) {
				bit = 1u << a;
				break;
			}
		}
		areas |= bit;
	}
	// The HUD is what the frame draws from its round icon on: the markers over the world (enemies' health bars and the
	// damage dealt, the lock-on) come before it, at their targets, often inside the HUD's areas (a bar above an enemy in
	// the top left one), and are no HUD.
	const auto frame = FrameGen::GuestFlips() + 1;
	if (icon) state.icon_frame = frame;
	return state.icon_frame == frame ? areas : 0u;
}

// The draws a draw is drawn as: once for each of `areas` (Scope), and with `outside` once for each rectangle of the
// canvas outside the areas (Complement, ClipScope) as it is; none of them: drawn as it is.
struct Plan {
	uint32_t areas   = 0;
	bool     outside = false;
	[[nodiscard]] bool Single() const { return !outside && std::popcount(areas) == 1; }
	[[nodiscard]] bool Several() const { return areas != 0 && !Single(); }
};
inline Plan Split(uint32_t areas) {
	return {areas & (Outside - 1u), (areas & Outside) != 0 && (areas & (Outside - 1u)) != 0};
}

// The canvas outside the areas (as drawn: the viewport unchanged) in window pixels, as rectangles: bands between the
// areas' tops and bottoms, the gaps between the areas in each. (The wide output's: in the 3840 wide layout at the left
// edge, as the draw is widened after (StretchScope, Widened).)
struct Rect {
	int left = 0, top = 0, right = 0, bottom = 0;
};
inline uint32_t Complement(const HW::Context& ctx, std::array<Rect, 32>& out) {
	const auto&                    viewport = ctx.GetScreenViewport().viewports[0];
	const bool                     wide     = CanvasWidthOf(ctx) > CanvasWidth;
	const auto window_x = [&](float x) { return wide ? (x + 1.0f) * (CanvasWidth / 2.0f) : viewport.xoffset + x * viewport.xscale; };
	std::array<Rect, Areas.size()> held {};
	std::array<int, 2 * Areas.size() + 2> ys {0, static_cast<int>(CanvasHeight)};
	uint32_t                       ny = 2;
	for (uint32_t a = 0; a < Areas.size(); ++a) {
		const auto& area = Areas[a];
		const float x0 = window_x(area.x0), x1 = window_x(area.x1);
		const float y0 = viewport.yoffset + area.y0 * viewport.yscale, y1 = viewport.yoffset + area.y1 * viewport.yscale;
		auto&       r  = held[a];
		r.left   = std::clamp(static_cast<int>(std::floor(std::min(x0, x1))), 0, static_cast<int>(CanvasWidth));
		r.right  = std::clamp(static_cast<int>(std::ceil(std::max(x0, x1))), 0, static_cast<int>(CanvasWidth));
		r.top    = std::clamp(static_cast<int>(std::floor(std::min(y0, y1))), 0, static_cast<int>(CanvasHeight));
		r.bottom = std::clamp(static_cast<int>(std::ceil(std::max(y0, y1))), 0, static_cast<int>(CanvasHeight));
		ys[ny++] = r.top;
		ys[ny++] = r.bottom;
	}
	std::sort(ys.begin(), ys.begin() + ny);
	ny = static_cast<uint32_t>(std::unique(ys.begin(), ys.begin() + ny) - ys.begin());
	uint32_t count = 0;
	for (uint32_t b = 0; b + 1 < ny; ++b) {
		const int top = ys[b], bottom = ys[b + 1];
		if (bottom <= top) continue;
		std::array<std::pair<int, int>, Areas.size()> spans {};
		uint32_t                                       ns = 0;
		for (const auto& r: held) {
			if (r.top < bottom && r.bottom > top && r.right > r.left) spans[ns++] = {r.left, r.right};
		}
		std::sort(spans.begin(), spans.begin() + ns);
		int x = 0;
		for (uint32_t s = 0; s <= ns; ++s) {
			const int gap_end = s < ns ? spans[s].first : static_cast<int>(CanvasWidth);
			if (gap_end > x && count < out.size()) {
				// (Merged with the rectangle above of the same columns.)
				const auto above = std::find_if(out.begin(), out.begin() + count, [&](const Rect& r) {
					return r.left == x && r.right == gap_end && r.bottom == top;
				});
				if (above != out.begin() + count) {
					above->bottom = bottom;
				} else {
					out[count++] = {x, top, gap_end, bottom};
				}
			}
			if (s < ns) x = std::max(x, spans[s].second);
		}
	}
	return count;
}

// For the length of one draw of the canvas outside the areas: the generic scissor narrowed to `rect`; restored after it.
class ClipScope {
public:
	ClipScope(HW::Context& ctx, const Rect& rect): m_ctx(&ctx) {
		const auto& screen = ctx.GetScreenViewport();
		m_left = screen.generic_scissor_left, m_top = screen.generic_scissor_top;
		m_right = screen.generic_scissor_right, m_bottom = screen.generic_scissor_bottom;
		m_window_offset = screen.generic_scissor_window_offset_enable;
		const int left = std::max(m_left, rect.left), top = std::max(m_top, rect.top);
		ctx.SetGenericScissor(left, top, std::max(left, std::min(m_right, rect.right)),
		                      std::max(top, std::min(m_bottom, rect.bottom)), m_window_offset);
		Changed();
	}
	~ClipScope() {
		m_ctx->SetGenericScissor(m_left, m_top, m_right, m_bottom, m_window_offset);
		Changed();
	}
	KYTY_CLASS_NO_COPY(ClipScope);

private:
	static void Changed() {
		auto& shadow = DrawStateObserver::g_shadow;
		shadow.key_changes |= 1u;
		shadow.dirty = true;
	}

	HW::Context* m_ctx = nullptr;
	bool         m_window_offset = false;
	int          m_left = 0, m_top = 0, m_right = 0, m_bottom = 0;
};

// For the length of one draw of an area (the GPU thread, around both of its ways): the viewport squeezed around the
// area's side (the wide output's: moved to it), and with `scissor` the generic scissor around where the area then lies;
// restored after it. The draw state the table and native ways keep is keyed by the registers: they hash them again
// (DrawStateObserver::g_shadow) both times.
class Scope {
public:
	Scope(HW::Context& ctx, const HW::PixelShaderInfo& ps, uint32_t area, bool scissor) {
		const double wide = ::Loader::DemonsSoulsRenderSize::WideAspect();
		if (wide <= 0.0 || area >= Areas.size()) return;
		const auto&  screen   = ctx.GetScreenViewport();
		const auto&  viewport = screen.viewports[0];
		const auto&  a        = Areas[area];
		const auto   width    = std::max(CanvasWidthOf(ctx), CanvasWidth);
		m_ctx                 = &ctx;
		m_xscale              = viewport.xscale;
		m_xoffset             = viewport.xoffset;
		// Where the draw lands, as a window x' = scale * x + shift of the x it is drawn at.
		float scale = 1.0f, shift = 0.0f;
		if (width != CanvasWidth) {
			shift = static_cast<float>(width - CanvasWidth) * (a.side + 1.0f) / 2.0f;
		} else {
			scale = static_cast<float>((16.0 / 9.0) / wide);
			shift = (m_xoffset + a.side * m_xscale) * (1.0f - scale);
		}
		ctx.SetViewportXScale(0, m_xscale * scale);
		ctx.SetViewportXOffset(0, m_xoffset * scale + shift);
		// (One area's draw: a draw of several shares its rows between them.)
		if (!scissor) {
			if (scale != 1.0f || shift != 0.0f) {
				PatchMask(ps, scale, shift);
			} else {
				RestoreMask(ps);
			}
		}
		if (scissor) {
			m_scissor = true;
			m_left = screen.generic_scissor_left, m_top = screen.generic_scissor_top;
			m_right = screen.generic_scissor_right, m_bottom = screen.generic_scissor_bottom;
			m_window_offset = screen.generic_scissor_window_offset_enable;
			// The area's corners in window pixels, where the viewport now places them (the wide output's layout: 3840
			// wide at the left edge).
			const auto drawn_x  = [&](float x) {
				return width != CanvasWidth ? (x + 1.0f) * (CanvasWidth / 2.0f) : m_xoffset + x * m_xscale;
			};
			const auto window_x = [&](float x) { return scale * drawn_x(x) + shift; };
			const auto window_y = [&](float y) { return viewport.yoffset + y * viewport.yscale; };
			const int   left     = std::max(m_left, static_cast<int>(std::floor(window_x(a.x0))));
			const int   right    = std::min(m_right, static_cast<int>(std::ceil(window_x(a.x1))));
			const float top_y = window_y(a.y1), bottom_y = window_y(a.y0);
			const int   top    = std::max(m_top, static_cast<int>(std::floor(std::min(top_y, bottom_y))));
			const int   bottom = std::min(m_bottom, static_cast<int>(std::ceil(std::max(top_y, bottom_y))));
			ctx.SetGenericScissor(left, top, std::max(left, right), std::max(top, bottom), m_window_offset);
		}
		Changed();
	}
	~Scope() {
		if (m_ctx == nullptr) return;
		m_ctx->SetViewportXScale(0, m_xscale);
		m_ctx->SetViewportXOffset(0, m_xoffset);
		if (m_scissor) m_ctx->SetGenericScissor(m_left, m_top, m_right, m_bottom, m_window_offset);
		Changed();
	}
	KYTY_CLASS_NO_COPY(Scope);

private:
	static void Changed() {
		auto& shadow = DrawStateObserver::g_shadow;
		shadow.key_changes |= 1u;
		shadow.dirty = true;
	}

	HW::Context* m_ctx     = nullptr;
	float        m_xscale  = 0.0f;
	float        m_xoffset = 0.0f;
	bool         m_scissor = false, m_window_offset = false;
	int          m_left = 0, m_top = 0, m_right = 0, m_bottom = 0;
};

// A draw into the wide output's interface canvas in a frame showing the 3D: widened (StretchScope) unless it is the HUD's.
inline bool Stretched(const HW::Context& ctx) {
	if (g_state.canvas == 0 || !FrameGen::SceneNoted() || CanvasWidthOf(ctx) <= CanvasWidth) return false;
	return ctx.GetRenderTarget(0).base.addr == g_state.canvas;
}

// A rectangle of the 3840 wide layout where StretchScope's widening puts it.
inline Rect Widened(const Rect& rect) {
	const float k = static_cast<float>(WideWidth()) / CanvasWidth;
	return {static_cast<int>(std::floor(rect.left * k)), rect.top, static_cast<int>(std::ceil(rect.right * k)), rect.bottom};
}

// For the length of one draw into the wide output's interface canvas that is not the HUD's (menus, fades, markers over
// the world): widened from the 3840 wide layout at the left edge to the canvas - the viewport, a scissor narrower than
// the canvas (the layout's) and a masked icon's mask with it -, the picture the 16:9 output widened showed; restored
// after it.
class StretchScope {
public:
	StretchScope(HW::Context& ctx, const HW::PixelShaderInfo& ps): m_ctx(&ctx), m_screen(ctx.GetScreenViewport()) {
		const auto  width    = static_cast<int>(WideWidth());
		const float k        = static_cast<float>(width) / CanvasWidth;
		const auto& viewport = m_screen.viewports[0];
		const auto  widen    = [&](int x) { return std::clamp(static_cast<int>(std::lround(x * k)), 0, width); };
		ctx.SetViewportXScale(0, viewport.xscale * k);
		ctx.SetViewportXOffset(0, viewport.xoffset * k);
		if (m_screen.generic_scissor_left > 0 || m_screen.generic_scissor_right < width)
			ctx.SetGenericScissor(widen(m_screen.generic_scissor_left), m_screen.generic_scissor_top,
			                      widen(m_screen.generic_scissor_right), m_screen.generic_scissor_bottom,
			                      m_screen.generic_scissor_window_offset_enable);
		if (viewport.viewport_scissor_left > 0 || viewport.viewport_scissor_right < width)
			ctx.SetViewportScissor(0, widen(viewport.viewport_scissor_left), viewport.viewport_scissor_top,
			                       widen(viewport.viewport_scissor_right), viewport.viewport_scissor_bottom,
			                       viewport.viewport_scissor_window_offset_enable);
		PatchMask(ps, k, 0.0f); // (only a masked icon's)
		Changed();
	}
	~StretchScope() {
		const auto& viewport = m_screen.viewports[0];
		m_ctx->SetViewportXScale(0, viewport.xscale);
		m_ctx->SetViewportXOffset(0, viewport.xoffset);
		m_ctx->SetGenericScissor(m_screen.generic_scissor_left, m_screen.generic_scissor_top, m_screen.generic_scissor_right,
		                         m_screen.generic_scissor_bottom, m_screen.generic_scissor_window_offset_enable);
		m_ctx->SetViewportScissor(0, viewport.viewport_scissor_left, viewport.viewport_scissor_top,
		                          viewport.viewport_scissor_right, viewport.viewport_scissor_bottom,
		                          viewport.viewport_scissor_window_offset_enable);
		Changed();
	}
	KYTY_CLASS_NO_COPY(StretchScope);

private:
	static void Changed() {
		auto& shadow = DrawStateObserver::g_shadow;
		shadow.key_changes |= 1u;
		shadow.dirty = true;
	}

	HW::Context*      m_ctx = nullptr;
	HW::ScreenViewport m_screen; // (as it was)
};

} // namespace Libs::Graphics::HudAnchor

#endif // EMULATOR_SRC_LOCAL_HUD_ANCHOR_H_
