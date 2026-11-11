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
#include "loader/demonsSoulsHdOutput.h"
#include "loader/demonsSoulsRenderSize.h"

#include "draw-state-observer.h"
#include "frame-gen.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <string_view>
#include <utility>
#include <vector>

namespace Libs::Graphics::HudAnchor {

// The interface's vertex shaders (01.007.000): their positions float x, y (z), as clip coordinates in the V# of the
// user data's first four words, or (PixelShader) as canvas pixels in the first V# of a table two pointers away.
constexpr std::array<uint64_t, 2> ClipShaders {0x7e880a66bf1b69c6ULL, 0xb2f062027869d4eeULL};
constexpr uint64_t                PixelShader = 0xbd39a2eba2d20d8bULL;
// The interface canvas: 3840x2160, the HD output's 1920x1080 (its own layout, the same areas in clip coordinates and
// the same shaders; the pixel shader's scale constants then 1/1920, 1/1080), or KYTY_OUTPUT_HEIGHT's output where the
// game lays its 4K layout out at that size (DemonsSoulsHdOutput::InterfaceAtOutput: 2560x1440 for 1440).
inline uint32_t CanvasWidth() {
	namespace Hd = ::Loader::DemonsSoulsHdOutput;
	return Hd::InterfaceAtOutput() ? Hd::UpscaledWidth() : Hd::Active() ? Hd::HdWidth : 3840u;
}
inline uint32_t CanvasHeight() {
	namespace Hd = ::Loader::DemonsSoulsHdOutput;
	return Hd::InterfaceAtOutput() ? Hd::UpscaledHeight() : Hd::Active() ? Hd::HdHeight : 2160u;
}
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
// (Classify: a marker over the world, held to the centre: Scope's area MarkerArea.)
constexpr uint32_t Marker     = Outside << 1u;
constexpr uint32_t MarkerArea = Areas.size();

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
	uint64_t icon_frame = 0;              // `frame` + 1 of the frame the icon was last drawn in
	uint64_t frame      = 0;              // the guest flips the GPU thread's command stream has reached (NoteFlip)
	std::array<MaskPatch, 32> patches {};  // (by address, round robin)
	uint32_t                  next_patch = 0;
	bool                      patched    = false; // any patch outstanding
	uint64_t                  canvas     = 0;     // the wide output's interface canvas (an interface shader's target)
};
inline State g_state; // (the GPU thread's)

// A guest flip in the GPU thread's command stream (a GPU flip, a CPU flip's preparation): the frame Classify tells the
// HUD's draws (after the icon) and the markers over the world (before it) by, in the order the draws come. (The
// presenter's count, FrameGen::GuestFlips, runs on its own thread: not in step with the draws, the markers were missed.)
inline void NoteFlip() {
	++g_state.frame;
}

// The wide output's canvas width (0: none).
inline uint32_t WideWidth() {
	return ::Loader::DemonsSoulsRenderSize::WideOutputWidth();
}

// KYTY_PRESENT_ASPECT=fit with a wider aspect (KYTY_RENDER_ASPECT): the game's menus at 16:9 in the middle of the
// picture (StretchScope), which the presenter shows between black bars (FrameGen::TakeMenu), not widened across it
// like the rest of the interface: the wide output's 16:9 part there, or where the 3D is squeezed into the game's own
// 16:9 output (its TAA) the middle part the presenter widens back to 16:9, the menu squeezed into it as much.
inline bool MenuBars() {
	static const bool fit = [] {
		const char* value = std::getenv("KYTY_PRESENT_ASPECT");
		return value != nullptr && std::string_view(value) == "fit";
	}();
	return fit && ::Loader::DemonsSoulsRenderSize::WideAspect() > 0.0;
}

// The current frame's interface goes to the middle: MenuBars, and this frame or the last drew a menu (NoteMenuDraw).
inline bool MenuCentred() {
	return MenuBars() && FrameGen::MenuNoted();
}

// The frame shows the 3D view with its HUD (Classify): not a menu in the middle.
inline bool HudFrame() {
	return FrameGen::SceneNoted() && !MenuCentred();
}

// The width of the interface canvas the draw goes into: the game's 3840, or the wide output's (the performance mode's;
// its Cinematic mode keeps 3840); 0 for another target.
inline uint32_t CanvasWidthOf(const HW::Context& ctx) {
	const auto& target = ctx.GetRenderTarget(0);
	const auto  width  = target.attrib2.width + 1u;
	if (target.base.addr == 0 || target.attrib2.height + 1u != CanvasHeight()) return 0;
	return width == CanvasWidth() || (WideWidth() != 0 && width == WideWidth()) ? width : 0;
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
inline void PatchMask(const HW::PixelShaderInfo& ps, float scale, float shift, float yscale = 1.0f) {
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
		row[1] /= yscale; // (ScaleScope's y too)
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

// The lock-on reticle: drawn after the icon among the HUD's draws (not with the markers before it), two quads of
// ClipShaders[1] that sample their own places of the interface's texture array. The shader's third V# (user data
// words 8-11) holds each vertex's u, v and layer; the reticle's (01.007.000, 1-1 frame dumps):
struct Sprite {
	float u0, u1, v0, v1, layer;
};
constexpr std::array<Sprite, 2> ReticleSprites {{
    {0.9873f, 0.9990f, 0.2471f, 0.2588f, 1.0f},
    {0.9824f, 0.9980f, 0.7134f, 0.7290f, 2.0f},
}};

// A quad of ClipShaders[1] (four vertices from `first`) samples one of the reticle's sprites.
inline bool IsReticle(const HW::VertexShaderInfo& vs, uint32_t first) {
	const auto& user = vs.gs_user_sgpr;
	uint64_t    base = 0;
	uint32_t    stride = 0;
	if (user.count < 12 || !ReadVsharp(user.value + 8, base, stride) || stride < 12 ||
	    !HostMemoryRangeIsReadable(base + static_cast<uint64_t>(first) * stride, 4u * stride))
		return false;
	float u0 = 2.0f, u1 = -1.0f, v0 = 2.0f, v1 = -1.0f, layer = -1.0f;
	for (uint32_t i = 0; i < 4; ++i) {
		float uvl[3] {};
		std::memcpy(uvl, reinterpret_cast<const void*>(base + static_cast<uint64_t>(first + i) * stride), sizeof(uvl));
		if (i > 0 && uvl[2] != layer) return false;
		layer = uvl[2];
		u0 = std::min(u0, uvl[0]), u1 = std::max(u1, uvl[0]), v0 = std::min(v0, uvl[1]), v1 = std::max(v1, uvl[1]);
	}
	constexpr float e = 0.002f;
	return std::any_of(ReticleSprites.begin(), ReticleSprites.end(), [&](const Sprite& s) {
		return s.layer == layer && std::fabs(s.u0 - u0) < e && std::fabs(s.u1 - u1) < e && std::fabs(s.v0 - v0) < e &&
		       std::fabs(s.v1 - v1) < e;
	});
}

inline float OutputScale();
inline bool  OutputTarget(const HW::Context& ctx);

// KYTY_FRAMEGEN_BACKEND=fsr: an interface draw names the canvas (its render target) whose alpha marks the HUD for FSR
// (FrameGen::NoteUiCanvas). With the interface scaled to KYTY_OUTPUT_HEIGHT's output the canvas is the output's size
// (OutputTarget): at 1440 FSR found none and generated the HUD too, which flickered (issue #170). With the wide output
// the HUD's canvas is as wide as it (WideWidth; issue #250, the same ghosting at 32:9).
inline void NoteCanvas(const HW::Context& ctx, const HW::VertexShaderInfo& vs) {
	const auto&    target = ctx.GetRenderTarget(0);
	const uint32_t width  = WideWidth() != 0 ? WideWidth() : CanvasWidth();
	const bool     canvas = OutputScale() > 0.0f ? OutputTarget(ctx)
	                                             : target.attrib2.width + 1u == width &&
	                                                  target.attrib2.height + 1u == CanvasHeight();
	if (target.base.addr == 0 || !canvas) return;
	const uint64_t hash = HashAt(vs.es_regs.data_addr);
	if (hash == PixelShader || std::find(ClipShaders.begin(), ClipShaders.end(), hash) != ClipShaders.end())
		FrameGen::NoteUiCanvas(target.base.addr);
}

// The vertices of each primitive of a draw (`strip`: each vertex with the two before it), 0 when they are not looked at.
inline uint32_t PrimitiveCorners(Prospero::PrimitiveType primitive, bool& strip) {
	using Prospero::PrimitiveType;
	strip = primitive == PrimitiveType::kTriStrip;
	switch (primitive) {
		case PrimitiveType::kTriList:
		case PrimitiveType::kTriStrip:
		case PrimitiveType::kRectList:
		case PrimitiveType::kRectListLegacy: return 3;
		case PrimitiveType::kQuadListLegacy: return 4;
		default: return 0;
	}
}

// The areas the draw's vertices lie in (a bit each, Outside for one in none), from its vertices in guest memory; 0 when
// it is no interface draw into the canvas in a frame showing the 3D, or its vertices are not readable.
inline uint32_t Classify(const HW::Context& ctx, const HW::VertexShaderInfo& vs, uint32_t first, uint32_t count,
                         Prospero::PrimitiveType primitive) {
	if (count != 0 && FrameGen::FsrMarksHud()) NoteCanvas(ctx, vs);
	if (::Loader::DemonsSoulsRenderSize::WideAspect() <= 0.0 || count == 0) return 0;
	const auto& target = ctx.GetRenderTarget(0);
	const auto  width  = CanvasWidthOf(ctx);
	if (width == 0) return 0;
	auto&          state = g_state;
	const uint64_t hash  = HashAt(vs.es_regs.data_addr);
	const bool     clip  = std::find(ClipShaders.begin(), ClipShaders.end(), hash) != ClipShaders.end();
	if (!clip && hash != PixelShader) return 0;
	// (An interface shader's target: the canvas the wide output's other draws into it are widened in, StretchScope; with
	// the 3D squeezed into the game's own 16:9 output, its canvas, where a menu shown is squeezed into the middle.)
	if (width != CanvasWidth() || WideWidth() == 0) state.canvas = target.base.addr;
	if (!HudFrame()) return 0;
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
	float    min_x = 2.0f, max_x = -2.0f;
	// A primitive over an area's edge: a corner in an area and another on the canvas outside them, or in an area held
	// to another side. The HUD's own (glyphs, frames, icons) each lie in one area; a menu's (the toolbelt's backdrop
	// at 43:18 reaches into the picked-up items' and the health's areas, issue #106) are cut at those edges by drawing
	// each area's part apart, its pieces moved to different sides: such a draw is left as it is, as menus are.
	bool       strip   = false;
	const auto corners = PrimitiveCorners(primitive, strip);
	uint32_t   prim_areas = 0, prim_corner = 0;
	bool       prim_out = false, straddles = false;
	// (Strips: the last two corners' area bits and whether each was on the canvas outside the areas.)
	std::array<std::pair<uint32_t, bool>, 2> last {};
	const auto over_edge = [](uint32_t in_areas, bool out) {
		if (in_areas == 0) return false;
		if (out) return true;
		float side = 0.0f;
		bool  any  = false;
		for (uint32_t a = 0; a < Areas.size(); ++a) {
			if ((in_areas & (1u << a)) == 0) continue;
			if (any && Areas[a].side != side) return true;
			side = Areas[a].side;
			any  = true;
		}
		return false;
	};
	for (uint32_t i = 0; i < count; ++i) {
		float xy[2] {};
		std::memcpy(xy, reinterpret_cast<const void*>(base + static_cast<uint64_t>(first + i) * stride), sizeof(xy));
		if (!std::isfinite(xy[0]) || !std::isfinite(xy[1])) return Outside;
		if (!clip) {
			xy[0] = 2.0f * xy[0] * scale[0] - 1.0f;
			xy[1] = 1.0f - 2.0f * xy[1] * scale[1];
		}
		// (The wide canvas: the window x the viewport puts it at, in the 3840 wide layout's clip coordinates.)
		if (width != CanvasWidth()) xy[0] = 2.0f * (viewport.xoffset + xy[0] * viewport.xscale) / CanvasWidth() - 1.0f;
		icon  = icon && xy[0] >= Icon.x0 && xy[0] <= Icon.x1 && xy[1] >= Icon.y0 && xy[1] <= Icon.y1;
		min_x = std::min(min_x, xy[0]);
		max_x = std::max(max_x, xy[0]);
		uint32_t bit = Outside;
		for (uint32_t a = 0; a < Areas.size(); ++a) {
			const auto& area = Areas[a];
			if (xy[0] >= area.x0 && xy[0] <= area.x1 && xy[1] >= area.y0 && xy[1] <= area.y1) {
				bit = 1u << a;
				break;
			}
		}
		areas |= bit;
		if (corners != 0 && !straddles) {
			const uint32_t in_area = bit & (Outside - 1u);
			const bool     out     = bit == Outside && std::fabs(xy[0]) <= 1.0f && std::fabs(xy[1]) <= 1.0f;
			if (strip) {
				straddles = i >= 2 && over_edge(in_area | last[0].first | last[1].first, out || last[0].second || last[1].second);
				last[0]   = last[1];
				last[1]   = {in_area, out};
			} else {
				prim_areas |= in_area;
				prim_out = prim_out || out;
				if (++prim_corner == corners) {
					straddles   = over_edge(prim_areas, prim_out);
					prim_areas  = 0;
					prim_out    = false;
					prim_corner = 0;
				}
			}
		}
	}
	// The HUD is what the frame draws from its round icon on: the markers over the world (enemies' health bars and the
	// damage dealt) come before it, at their targets, often inside the HUD's areas (a bar above an enemy in the top left
	// one), and are no HUD; the lock-on reticle comes after it (IsReticle).
	const auto frame = state.frame + 1;
	if (icon) state.icon_frame = frame;
	// The lock-on reticle, drawn after the icon: a marker over the world as the ones before it (below).
	if (state.icon_frame == frame && clip && hash == ClipShaders[1] && count == 4 && areas == Outside && IsReticle(vs, first))
		return Marker;
	if (state.icon_frame == frame) {
		static const bool log = std::getenv("KYTY_HUD_LOG") != nullptr;
		static uint64_t logged_frame = 0;
		if (log && straddles && logged_frame != frame) {
			logged_frame = frame;
			std::printf("HUD anchor: frame %llu draw of %u vertices (%s) over an area's edge (areas %#x, x %.2f to %.2f): "
			            "drawn as it is\n",
			            static_cast<unsigned long long>(frame), count, clip ? "clip" : "pixels", areas, min_x, max_x);
		}
		return straddles ? 0u : areas;
	}
	// The game places those markers with its own 16:9 projection of their targets: aspect / (16/9) times as far from the
	// screen's centre as the wider 3D shows the targets (the lock-on and an enemy's health bar beside an enemy off the
	// centre, the user 10-19). A draw before the icon in a frame after one with the HUD is drawn as an area held to the
	// centre (Scope, MarkerArea): back over its target, unstretched. They come batched, the lock-on and the health bars
	// in one draw that also reaches the canvas's left edge (36 vertices, x -0.98 to 0.09: a slot parked there); only a
	// draw over the whole width (a fade over the screen) is left as it is.
	if (state.icon_frame + 1 == frame && !(min_x <= -0.99f && max_x >= 0.99f)) {
		static bool noted = false;
		if (!noted) {
			noted = true;
			std::printf("HUD anchor: markers over the world held to the screen's centre (first at frame %llu)\n",
			            static_cast<unsigned long long>(frame));
		}
		return Marker;
	}
	return 0u;
}

// The draws a draw is drawn as: once for each of `areas` (Scope), and with `outside` once for each rectangle of the
// canvas outside the areas (Complement, ClipScope) as it is; none of them: drawn as it is.
struct Plan {
	uint32_t areas   = 0;
	bool     outside = false;
	bool     marker  = false; // a marker over the world (Classify): drawn as MarkerArea
	[[nodiscard]] bool Single() const { return !outside && std::popcount(areas) == 1; }
	[[nodiscard]] bool Several() const { return areas != 0 && !Single(); }
};
inline Plan Split(uint32_t areas) {
	return {areas & (Outside - 1u), (areas & Outside) != 0 && (areas & (Outside - 1u)) != 0, (areas & Marker) != 0};
}

// The canvas outside the areas (as drawn: the viewport unchanged) in window pixels, as rectangles: bands between the
// areas' tops and bottoms, the gaps between the areas in each. (The wide output's: in the 3840 wide layout at the left
// edge, as the draw is widened after (StretchScope, Widened).)
struct Rect {
	int left = 0, top = 0, right = 0, bottom = 0;
};
inline uint32_t Complement(const HW::Context& ctx, std::array<Rect, 32>& out) {
	const auto&                    viewport = ctx.GetScreenViewport().viewports[0];
	const bool                     wide     = CanvasWidthOf(ctx) > CanvasWidth();
	const auto window_x = [&](float x) { return wide ? (x + 1.0f) * (CanvasWidth() / 2.0f) : viewport.xoffset + x * viewport.xscale; };
	std::array<Rect, Areas.size()> held {};
	std::array<int, 2 * Areas.size() + 2> ys {0, static_cast<int>(CanvasHeight())};
	uint32_t                       ny = 2;
	for (uint32_t a = 0; a < Areas.size(); ++a) {
		const auto& area = Areas[a];
		const float x0 = window_x(area.x0), x1 = window_x(area.x1);
		const float y0 = viewport.yoffset + area.y0 * viewport.yscale, y1 = viewport.yoffset + area.y1 * viewport.yscale;
		auto&       r  = held[a];
		r.left   = std::clamp(static_cast<int>(std::floor(std::min(x0, x1))), 0, static_cast<int>(CanvasWidth()));
		r.right  = std::clamp(static_cast<int>(std::ceil(std::max(x0, x1))), 0, static_cast<int>(CanvasWidth()));
		r.top    = std::clamp(static_cast<int>(std::floor(std::min(y0, y1))), 0, static_cast<int>(CanvasHeight()));
		r.bottom = std::clamp(static_cast<int>(std::ceil(std::max(y0, y1))), 0, static_cast<int>(CanvasHeight()));
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
			const int gap_end = s < ns ? spans[s].first : static_cast<int>(CanvasWidth());
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
		if (wide <= 0.0 || area > MarkerArea || (scissor && area == MarkerArea)) return;
		const auto&  screen   = ctx.GetScreenViewport();
		const auto&  viewport = screen.viewports[0];
		// (A marker over the world: held to the centre, MarkerArea.)
		const auto&  a        = area < Areas.size() ? Areas[area] : Area {-1.0f, 1.0f, -1.0f, 1.0f, 0.0f};
		const auto   width    = std::max(CanvasWidthOf(ctx), CanvasWidth());
		m_ctx                 = &ctx;
		m_xscale              = viewport.xscale;
		m_xoffset             = viewport.xoffset;
		// Where the draw lands, as a window x' = scale * x + shift of the x it is drawn at.
		float scale = 1.0f, shift = 0.0f;
		if (width != CanvasWidth()) {
			shift = static_cast<float>(width - CanvasWidth()) * (a.side + 1.0f) / 2.0f;
			// A wide output narrower than the aspect (held at DemonsSoulsRenderSize::MaxWideOutputWidth, 32:9): the
			// presenter widens the canvas after, so the area is narrowed as much around its side (none otherwise).
			if (const double widened = wide * CanvasHeight() / width; widened > 1.01) {
				scale = static_cast<float>(1.0 / widened);
				shift += (a.side + 1.0f) * (CanvasWidth() / 2.0f) * (1.0f - scale);
			}
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
				return width != CanvasWidth() ? (x + 1.0f) * (CanvasWidth() / 2.0f) : m_xoffset + x * m_xscale;
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
// Where the 3D is squeezed into the game's own 16:9 output, a draw into its canvas while a menu is shown in the middle.
inline bool Stretched(const HW::Context& ctx) {
	if (g_state.canvas == 0 || !FrameGen::SceneNoted()) return false;
	if (CanvasWidthOf(ctx) <= CanvasWidth() && (WideWidth() != 0 || !MenuCentred())) return false;
	return ctx.GetRenderTarget(0).base.addr == g_state.canvas;
}

// StretchScope's x' = k x + shift of the 3840 wide layout: widened from its left edge across the canvas, or, a menu shown
// with MenuBars (MenuCentred), at its own size in the middle (issue #250: the menus at 16:9 between black bars). With
// the 3D squeezed into the game's own 16:9 output, which the presenter widens: the menu squeezed about the middle as
// much (16:9 over the aspect), the part the presenter shows widened back to 16:9.
inline std::pair<float, float> StretchTransform() {
	const float wide = static_cast<float>(WideWidth()), layout = static_cast<float>(CanvasWidth());
	if (!MenuCentred()) return {wide / layout, 0.0f};
	if (wide != 0.0f) return {1.0f, (wide - layout) / 2.0f};
	const auto squeeze = static_cast<float>((16.0 / 9.0) / ::Loader::DemonsSoulsRenderSize::WideAspect());
	return {squeeze, layout * (1.0f - squeeze) / 2.0f};
}

// A rectangle of the 3840 wide layout where StretchScope's widening puts it.
inline Rect Widened(const Rect& rect) {
	const auto [k, shift] = StretchTransform();
	return {static_cast<int>(std::floor(rect.left * k + shift)), rect.top, static_cast<int>(std::ceil(rect.right * k + shift)),
	        rect.bottom};
}

// For the length of one draw into the wide output's interface canvas that is not the HUD's (menus, fades, markers over
// the world): widened from the 3840 wide layout at the left edge to the canvas - the viewport, a scissor narrower than
// the canvas (the layout's) and a masked icon's mask with it -, the picture the 16:9 output widened showed; restored
// after it. A menu shown with MenuBars: moved to the middle at its own size instead (StretchTransform).
class StretchScope {
public:
	StretchScope(HW::Context& ctx, const HW::PixelShaderInfo& ps): m_ctx(&ctx), m_screen(ctx.GetScreenViewport()) {
		const auto width      = static_cast<int>(std::max(WideWidth(), CanvasWidth()));
		const auto [k, shift] = StretchTransform();
		const auto& viewport  = m_screen.viewports[0];
		const auto  widen     = [&](int x) { return std::clamp(static_cast<int>(std::lround(x * k + shift)), 0, width); };
		ctx.SetViewportXScale(0, viewport.xscale * k);
		ctx.SetViewportXOffset(0, viewport.xoffset * k + shift);
		if (m_screen.generic_scissor_left > 0 || m_screen.generic_scissor_right < width)
			ctx.SetGenericScissor(widen(m_screen.generic_scissor_left), m_screen.generic_scissor_top,
			                      widen(m_screen.generic_scissor_right), m_screen.generic_scissor_bottom,
			                      m_screen.generic_scissor_window_offset_enable);
		if (viewport.viewport_scissor_left > 0 || viewport.viewport_scissor_right < width)
			ctx.SetViewportScissor(0, widen(viewport.viewport_scissor_left), viewport.viewport_scissor_top,
			                       widen(viewport.viewport_scissor_right), viewport.viewport_scissor_bottom,
			                       viewport.viewport_scissor_window_offset_enable);
		PatchMask(ps, k, shift); // (only a masked icon's)
		if (shift != 0.0f) FrameGen::NoteCentred();
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

// KYTY_OUTPUT_HEIGHT above 1080 with KYTY_OUTPUT_LAYOUT=hd (DemonsSoulsHdOutput::InterfaceScaled; by default the game
// lays out its 4K layout at the output's size, nothing scaled here): the game lays its interface out in the HD
// layout's 1920x1080 and turns the layout's pixels into clip coordinates by the canvas's own size (2560x1440 for 1440),
// the viewport the whole canvas: the HUD, the menus, the movies and the loading screens showed at their HD pixels, in
// the canvas's top left corner. Their draws into a target of the output's size get the viewport scaled by output /
// 1080 about the canvas's corner, so a layout pixel lands where the output's is (ScaleScope; a masked icon's mask with
// it; the scissors are the whole canvas). KYTY_OUTPUT_TRACE=1 lists the draws into such targets, a line for each vertex
// and pixel shader pair.
inline float OutputScale() {
	namespace Hd = ::Loader::DemonsSoulsHdOutput;
	return Hd::InterfaceScaled() ? static_cast<float>(Hd::UpscaledHeight()) / static_cast<float>(Hd::HdHeight) : 0.0f;
}

inline bool OutputTarget(const HW::Context& ctx) {
	namespace Hd = ::Loader::DemonsSoulsHdOutput;
	const auto& target = ctx.GetRenderTarget(0);
	return target.base.addr != 0 && target.attrib2.width + 1u == Hd::UpscaledWidth() &&
	       target.attrib2.height + 1u == Hd::UpscaledHeight();
}

// The HD layout's other draws at 1920x1080 (01.007.000, KYTY_OUTPUT_TRACE): the movies (the intro, the title's backdrop)
// and the loading screens.
constexpr std::array<uint64_t, 2> HdLayoutShaders {0x05ca0d08ae41fd2eULL, 0xa64d43c72c1c5c87ULL};
// Draws of the HD layout whose vertex shader also draws over the whole canvas (full-screen passes, not scaled): told by
// the pixel shader too (01.007.000, KYTY_OUTPUT_TRACE, issue #187). The menus' glow behind the selected slot (it was
// drawn at the slot's layout pixels: a green box above and left of it at 1440) and the glow of their title band (its
// scissor in layout pixels).
constexpr std::array<std::pair<uint64_t, uint64_t>, 2> HdLayoutPairs {
    {{0x3ec043bd86013cafULL, 0x91cd581e82574995ULL}, {0x5d9a2e8e34030f35ULL, 0xae51f2cf2ca8c121ULL}}};

// With MenuBars, an interface draw of those two (the glow behind a menu's selected slot, of its title band): the frame
// shows one of the game's menus (FrameGen::NoteMenu). 1-1 at 32:9, the equipment menu and its submenus: in each of
// their frames, early on (before the canvas's other draws); never in play.
[[gnu::noinline]] inline void NoteMenuDraw(const HW::VertexShaderInfo& vs, const HW::PixelShaderInfo& ps) {
	if (!MenuBars()) return;
	const auto pair = std::make_pair(HashAt(vs.es_regs.data_addr), PixelShaderHash(ps));
	if (std::find(HdLayoutPairs.begin(), HdLayoutPairs.end(), pair) != HdLayoutPairs.end()) FrameGen::NoteMenu();
}

inline bool Scaled(const HW::Context& ctx, const HW::VertexShaderInfo& vs, const HW::PixelShaderInfo& ps) {
	if (OutputScale() <= 0.0f || !OutputTarget(ctx)) return false;
	const uint64_t hash   = HashAt(vs.es_regs.data_addr);
	const bool     scaled = hash == PixelShader || std::find(ClipShaders.begin(), ClipShaders.end(), hash) != ClipShaders.end() ||
	                    std::find(HdLayoutShaders.begin(), HdLayoutShaders.end(), hash) != HdLayoutShaders.end() ||
	                    std::find(HdLayoutPairs.begin(), HdLayoutPairs.end(), std::make_pair(hash, PixelShaderHash(ps))) !=
	                        HdLayoutPairs.end();
	static const bool trace = [] {
		const char* value = std::getenv("KYTY_OUTPUT_TRACE");
		return value != nullptr && *value != 0 && *value != '0';
	}();
	if (trace) {
		static std::vector<std::pair<uint64_t, uint64_t>> seen;
		const auto pair = std::make_pair(hash, PixelShaderHash(ps));
		if (seen.size() < 400 && std::find(seen.begin(), seen.end(), pair) == seen.end()) {
			seen.push_back(pair);
			const auto& screen   = ctx.GetScreenViewport();
			const auto& viewport = screen.viewports[0];
			std::printf("Output trace: frame %" PRIu64 " vs %016" PRIx64 " ps %016" PRIx64 " viewport (%.0f,%.0f) %.0fx%.0f "
			            "scissors (%d,%d)-(%d,%d) (%d,%d)-(%d,%d)%s\n",
			            g_state.frame, pair.first, pair.second, viewport.xoffset - std::fabs(viewport.xscale),
			            viewport.yoffset - std::fabs(viewport.yscale), 2.0f * std::fabs(viewport.xscale),
			            2.0f * std::fabs(viewport.yscale), screen.generic_scissor_left, screen.generic_scissor_top,
			            screen.generic_scissor_right, screen.generic_scissor_bottom, viewport.viewport_scissor_left,
			            viewport.viewport_scissor_top, viewport.viewport_scissor_right, viewport.viewport_scissor_bottom,
			            scaled ? " -> scaled" : "");
			std::fflush(stdout);
		}
	}
	return scaled;
}

class ScaleScope {
public:
	ScaleScope(HW::Context& ctx, const HW::PixelShaderInfo& ps): m_ctx(&ctx), m_screen(ctx.GetScreenViewport()) {
		namespace Hd = ::Loader::DemonsSoulsHdOutput;
		const float k        = OutputScale();
		const auto  width    = static_cast<int>(Hd::UpscaledWidth());
		const auto  height   = static_cast<int>(Hd::UpscaledHeight());
		const auto& viewport = m_screen.viewports[0];
		const auto  scale    = [&](int v, int most) { return std::clamp(static_cast<int>(std::lround(v * k)), 0, most); };
		// (window = offset + clip * scale: both times k, a layout pixel p at k * p.)
		ctx.SetViewportXScale(0, viewport.xscale * k);
		ctx.SetViewportXOffset(0, viewport.xoffset * k);
		ctx.SetViewportYScale(0, viewport.yscale * k);
		ctx.SetViewportYOffset(0, viewport.yoffset * k);
		// (A scissor over the whole canvas stays so; a narrower one, in the layout's pixels as the positions are, is scaled.)
		const auto whole = [&](int left, int top, int right, int bottom) {
			return left <= 0 && top <= 0 && right >= width - 1 && bottom >= height - 1;
		};
		if (!whole(m_screen.generic_scissor_left, m_screen.generic_scissor_top, m_screen.generic_scissor_right,
		           m_screen.generic_scissor_bottom))
			ctx.SetGenericScissor(scale(m_screen.generic_scissor_left, width), scale(m_screen.generic_scissor_top, height),
			                      scale(m_screen.generic_scissor_right, width), scale(m_screen.generic_scissor_bottom, height),
			                      m_screen.generic_scissor_window_offset_enable);
		if (!whole(viewport.viewport_scissor_left, viewport.viewport_scissor_top, viewport.viewport_scissor_right,
		           viewport.viewport_scissor_bottom))
			ctx.SetViewportScissor(0, scale(viewport.viewport_scissor_left, width), scale(viewport.viewport_scissor_top, height),
			                       scale(viewport.viewport_scissor_right, width), scale(viewport.viewport_scissor_bottom, height),
			                       viewport.viewport_scissor_window_offset_enable);
		PatchMask(ps, k, 0.0f, k); // (only a masked icon's)
		Changed();
		static bool reported = false;
		if (!reported) {
			reported = true;
			std::printf("Demon's Souls output: the interface's HD layout scaled %.3fx to the %dx%d output\n", k, width, height);
			std::fflush(stdout);
		}
	}
	~ScaleScope() {
		const auto& viewport = m_screen.viewports[0];
		m_ctx->SetViewportXScale(0, viewport.xscale);
		m_ctx->SetViewportXOffset(0, viewport.xoffset);
		m_ctx->SetViewportYScale(0, viewport.yscale);
		m_ctx->SetViewportYOffset(0, viewport.yoffset);
		m_ctx->SetGenericScissor(m_screen.generic_scissor_left, m_screen.generic_scissor_top, m_screen.generic_scissor_right,
		                         m_screen.generic_scissor_bottom, m_screen.generic_scissor_window_offset_enable);
		m_ctx->SetViewportScissor(0, viewport.viewport_scissor_left, viewport.viewport_scissor_top,
		                          viewport.viewport_scissor_right, viewport.viewport_scissor_bottom,
		                          viewport.viewport_scissor_window_offset_enable);
		Changed();
	}
	KYTY_CLASS_NO_COPY(ScaleScope);

private:
	static void Changed() {
		auto& shadow = DrawStateObserver::g_shadow;
		shadow.key_changes |= 1u;
		shadow.dirty = true;
	}

	HW::Context*       m_ctx = nullptr;
	HW::ScreenViewport m_screen; // (as it was)
};

} // namespace Libs::Graphics::HudAnchor

#endif // EMULATOR_SRC_LOCAL_HUD_ANCHOR_H_
