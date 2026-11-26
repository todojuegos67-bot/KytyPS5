// (pt1) BB_PATHTRACING=1: ray-traced sun shadows and reflections injected before the game's lighting resolves
// (pathtrace.h has the design and the switches; pt.comp the passes). Everything here runs on the GPU (translation)
// thread; nothing runs and nothing is allocated unless BB_PATHTRACING=1.
//
// Reference for the method: the GPL-2.0 Bloodborne PC path-tracing mod by João Vitor Alves (its docs/PATH_TRACING.md:
// capture of the G-buffer draws' transformed vertices, BLAS per mesh kept across frames, one TLAS a frame, ray-query
// compute passes, camera-reprojected temporal accumulation, composition before the post chain, F9, a stats line).
// This file is our own code for this renderer: no source of that mod is copied or adapted line by line.
#include "local/pathtrace.h"

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "local/frame-gen.h"
#include "local/hwrt-scene.h"
#include "local/local-pass-cache.h"
#include "local/rt-reflections.h"
#include "native/native-render.h"
#include "native/pass-cut.h"
#include "native/pass-registry.h"
#include "gpu_tiler_shaders/pt_spv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>

// (<windows.h>, through vk_mem_alloc.h, defines MemoryBarrier; vk::MemoryBarrier2 is meant here.)
#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

namespace Libs::Graphics::PathTrace {
namespace {

// ---- switches and presets ----------------------------------------------------------------------------------------

const char* Env(const char* name) {
	const char* v = std::getenv(name);
	return v != nullptr && *v != '\0' ? v : nullptr;
}
float EnvF(const char* name, float fallback, float lo, float hi) {
	const char* v = Env(name);
	if (v == nullptr) return fallback;
	const float f = static_cast<float>(std::atof(v));
	return std::isfinite(f) ? std::clamp(f, lo, hi) : fallback;
}
int EnvI(const char* name, int fallback, int lo, int hi) {
	const char* v = Env(name);
	return v == nullptr ? fallback : std::clamp(std::atoi(v), lo, hi);
}
bool SameText(const char* a, const char* b) {
	for (; *a != '\0' && *b != '\0'; ++a, ++b)
		if ((*a | 0x20) != (*b | 0x20)) return false;
	return *a == *b;
}

// One quality level (BB_PATHTRACING_QUALITY). Bajo = the cheap targets (shadows 1/2 + checkerboard, static occluders
// only); each level up raises resolution, distance and history; ultra traces at full resolution.
// (pt1e) refl_scale / rough_cutoff / refl_spp / refl_history only serve the legacy reflections (BB_PT_REFL_LEGACY=1);
// the default reflections are rt-reflections' fill (fill_block: trace pixels per hardware ray, per axis).
struct Preset {
	const char* name;
	const char* name_es;
	float       shadow_scale, refl_scale; // trace resolution, fraction of the render size per axis
	float       rough_cutoff;             // reflections only below this roughness
	float       shadow_dist, refl_dist;   // world units
	int         refl_spp;
	float       shadow_history, refl_history; // frames
	bool        gi;                       // one diffuse bounce (alto / ultra): NOT implemented in pt1 (logged)
	uint32_t    capture_mverts;           // the capture buffer (Mi vertices, 16 B each)
	uint32_t    mesh_budget_mb;           // vertices + indices + BLAS
	uint32_t    keep_frames;              // a mesh not drawn stays in the TLAS this long
	// (pt1e)
	bool        shadow_checker;           // half the shadow texels traced a frame (the temporal pass fills the others)
	bool        shadow_dynamic;           // moving meshes cast our traced shadow (mask 0x03; else 0x01: the game's own
	                                      // cascades keep the characters' shadows, min() keeps them)
	uint32_t    fill_block;               // reflections' fill: one ray per fill_block x fill_block trace pixels
	uint32_t    refit_every;              // moving meshes refit every N frames (the scene's main GPU cost: 150-190 a frame)
	float       budget_ms;                // GPU ms a frame (scene + shadows + fill) at 1440p on an RTX 5080 (the request)
};
constexpr Preset kPresets[4] = {
    // (pt1c) mesh budgets raised: 192 MB was full in every pt1 / pt1b window (the scene of 1-1 is ~1.4 M triangles,
    // ~190 MB of BLAS + vertices) and new meshes were refused: the TLAS missed geometry.
    // (pt1e) bajo keeps 320 MB: the PT total measured 312 MB with it (log 074330), inside the 300-400 MB asked; less
    // would refuse meshes (more missing geometry).
    {"low", "bajo", 0.5f, 0.25f, 0.25f, 80.0f, 40.0f, 1, 16.0f, 12.0f, false, 2, 320, 300, true, false, 2, 3, 2.0f},
    {"medium", "medio", 0.5f, 0.5f, 0.4f, 120.0f, 60.0f, 1, 20.0f, 16.0f, false, 3, 448, 450, false, false, 2, 2, 3.0f},
    {"high", "alto", 0.75f, 0.5f, 0.5f, 180.0f, 100.0f, 1, 24.0f, 20.0f, true, 4, 640, 600, true, true, 1, 1, 4.0f},
    {"ultra", "ultra", 1.0f, 1.0f, 0.6f, 300.0f, 150.0f, 2, 32.0f, 24.0f, true, 6, 896, 600, true, true, 1, 1, 5.0f},
};

int ParseQuality(const char* v) {
	if (v == nullptr) return 0;
	if (*v >= '1' && *v <= '4' && v[1] == '\0') return *v - '1';
	for (int i = 0; i < 4; ++i)
		if (SameText(v, kPresets[i].name) || SameText(v, kPresets[i].name_es)) return i;
	return 0;
}

struct Config {
	bool  requested = false, shadows = true, reflections = true, cycle = false, force = false, replace = false;
	bool  capture_off = false; // (pt1b) BB_PT_CAPTURE_WHEN_OFF=1: keep capturing / building while F9 has it off (instant toggle)
	bool  refl_legacy = false; // (pt1e) BB_PT_REFL_LEGACY=1: pt1d's reflections into the game's SSR image (not the fill)
	bool  ab = false;          // (pt1e) BB_PT_AB=1: 5 s on, 5 s off, in turn (the A/B line)
	bool  budget = true;       // (pt1e) BB_PT_BUDGET=0: the presets' trace scales fixed (no budget control)
	int   quality = 0, debug = 0, channel = -1, rough_layer = 2, rough_byte = 0;
	float sun_deg = 0.5f, strength = 1.0f, gain = 0.0f;
	std::vector<std::string> capture_scopes;
};
const Config& Cfg() {
	static const Config c = [] {
		Config r;
		const char* m = Env("BB_PATHTRACING");
		r.requested   = m != nullptr && *m != '0';
		if (!r.requested) return r;
		r.quality     = ParseQuality(Env("BB_PATHTRACING_QUALITY"));
		r.shadows     = EnvI("BB_PT_SHADOWS", 1, 0, 1) != 0;
		r.reflections = EnvI("BB_PT_REFLECTIONS", 1, 0, 1) != 0;
		r.cycle       = EnvI("BB_PT_CYCLE", 0, 0, 1) != 0;
		r.force       = EnvI("BB_PT_FORCE", 0, 0, 1) != 0;
		r.capture_off = EnvI("BB_PT_CAPTURE_WHEN_OFF", 0, 0, 1) != 0;
		r.refl_legacy = EnvI("BB_PT_REFL_LEGACY", 0, 0, 1) != 0;
		r.ab          = EnvI("BB_PT_AB", 0, 0, 1) != 0;
		r.budget      = EnvI("BB_PT_BUDGET", 1, 0, 1) != 0;
		r.debug       = EnvI("BB_PT_DEBUG", 0, 0, 3); // (pt1c) 1 tint, 2 our shadow alone in its channel, 3 our reflection alone
		r.channel     = EnvI("BB_PT_SHADOW_CHANNEL", -1, -1, 3);
		r.sun_deg     = EnvF("BB_PT_SUN_ANGLE", 0.5f, 0.0f, 5.0f);
		r.strength    = EnvF("BB_PT_SHADOW_STRENGTH", 1.0f, 0.0f, 1.0f);
		r.gain        = EnvF("BB_PT_REFL_GAIN", 0.0f, 0.0f, 64.0f);
		if (const char* sm = Env("BB_PT_SHADOW_MODE")) r.replace = SameText(sm, "replace");
		if (const char* rg = Env("BB_PT_ROUGH")) {
			if (SameText(rg, "none")) {
				r.rough_layer = -1;
			} else {
				r.rough_layer = std::clamp(std::atoi(rg), 0, 3);
				const char* dot = std::strchr(rg, '.');
				r.rough_byte    = dot != nullptr ? std::clamp(std::atoi(dot + 1), 0, 3) : 0;
			}
		}
		std::string scopes = Env("BB_PT_CAPTURE_SCOPES") != nullptr ? Env("BB_PT_CAPTURE_SCOPES") : "GBuffer,VirtualGeo";
		for (size_t at = 0; at < scopes.size();) {
			size_t end = scopes.find(',', at);
			if (end == std::string::npos) end = scopes.size();
			if (end > at) r.capture_scopes.push_back(scopes.substr(at, end - at));
			at = end + 1;
		}
		return r;
	}();
	return c;
}

// The live preset (BB_PT_CYCLE moves it every 5 s window; 4 = off) with the BB_PT_* overrides.
struct Live {
	int   preset = 0;
	bool  on     = true;
	float shadow_scale = 0.5f, refl_scale = 0.25f, rough_cutoff = 0.25f, shadow_dist = 80.0f, refl_dist = 40.0f;
	float shadow_history = 16.0f, refl_history = 12.0f;
	int   refl_spp = 1;
	// (pt1e)
	bool     shadow_checker = true, shadow_dynamic = false;
	uint32_t fill_block = 2, refit_every = 3;
	float    budget_ms = 2.0f;
	float    scale = 1.0f; // the budget control's factor on the trace scales (0.5 .. 1)
};
Live MakeLive(int preset) {
	const Preset& p = kPresets[std::clamp(preset, 0, 3)];
	Live l;
	l.preset         = preset;
	l.shadow_scale   = EnvF("BB_PT_SHADOW_SCALE", p.shadow_scale, 0.125f, 1.0f);
	l.refl_scale     = EnvF("BB_PT_REFL_SCALE", p.refl_scale, 0.125f, 1.0f);
	l.rough_cutoff   = EnvF("BB_PT_ROUGH_CUTOFF", p.rough_cutoff, 0.0f, 1.0f);
	l.shadow_dist    = EnvF("BB_PT_SHADOW_DIST", p.shadow_dist, 1.0f, 5000.0f);
	l.refl_dist      = EnvF("BB_PT_REFL_DIST", p.refl_dist, 1.0f, 5000.0f);
	l.refl_spp       = EnvI("BB_PT_REFL_SPP", p.refl_spp, 1, 4);
	l.shadow_history = EnvF("BB_PT_SHADOW_HISTORY", p.shadow_history, 1.0f, 64.0f);
	l.refl_history   = EnvF("BB_PT_REFL_HISTORY", p.refl_history, 1.0f, 64.0f);
	l.shadow_checker = EnvI("BB_PT_SHADOW_CHECKER", p.shadow_checker ? 1 : 0, 0, 1) != 0;
	l.shadow_dynamic = EnvI("BB_PT_SHADOW_DYNAMIC", p.shadow_dynamic ? 1 : 0, 0, 1) != 0;
	l.fill_block     = static_cast<uint32_t>(EnvI("BB_PT_FILL_BLOCK", static_cast<int>(p.fill_block), 1, 4));
	l.refit_every    = static_cast<uint32_t>(EnvI("BB_PT_REFIT_EVERY", static_cast<int>(p.refit_every), 1, 8));
	l.budget_ms      = EnvF("BB_PT_BUDGET_MS", p.budget_ms, 0.25f, 50.0f);
	return l;
}

// ---- the pass roles of the game's profiler scopes (names from the frame map: src/native/pass-table.inc) ----------
enum Role : uint8_t {
	RNone = 0,
	RGBuffer,
	RVirtualGeo,
	RCapture,         // another scope named in BB_PT_CAPTURE_SCOPES
	RResolveSunShadows,
	RRenderLights,
	RResolveLighting,
	RRenderSSR,
	RRenderIBL,
	RCascade,         // "RenderShadowCascade N" (computed names)
	RPostProcessing,  // (pt1c) the debug tint before the game's post chain
	RCount
};
// ---- our images and buffers ---------------------------------------------------------------------------------------
struct OwnImage {
	VkImage       image  = VK_NULL_HANDLE;
	VmaAllocation memory = nullptr;
	vk::ImageView view   = nullptr;
	vk::Extent2D  extent {};
	vk::Format    format = vk::Format::eUndefined;
	bool          general = false;
	uint64_t      bytes   = 0;
};

constexpr uint32_t Bindings   = 27; // (pt1e: 19..26 the reflections' fill)
constexpr uint32_t StatWords  = 512;
constexpr uint32_t Slots      = 8;   // frame slots (stats, timestamps): read 5 frames late
constexpr uint32_t ReadDelay  = 5;
constexpr uint32_t Queries    = 10;  // per slot: shadow start / trace / denoise / compose, reflection the same, (pt1e) fill start / end
constexpr uint32_t RetireWait = 8;   // frames before a retired view / image is destroyed

struct FrameUbo {
	float   inv[16], vp[16], pvp[16];
	float   cam[4], pcam[4], proj[4], depth_map[4], sun[4];
	float   render[4], strace[4], rtrace[4], target[4], par[4], par2[4];
	int32_t cfg[4], cfg2[4];
	// (pt1e) the reflections' fill: rt-reflections' view_to_clip (proj, depth_map), the fill's sizes and options
	float   fproj[4], fdepth[4], fsize[4], fpar[4];
	int32_t cfg3[4];
};
static_assert(sizeof(FrameUbo) == 3 * 64 + 18 * 16 && sizeof(FrameUbo) <= 512);

struct Push {
	int32_t mode = 0, step = 0, pad0 = 0, pad1 = 0;
};

// A game image seen this frame: the texture cache slot, the subresource, and the frame it was seen.
struct Seen {
	uint32_t   index = UINT32_MAX, generation = 0, level = 0, layer = 0;
	uint64_t   frame = UINT64_MAX;
	vk::Format format = vk::Format::eUndefined;
	vk::Extent2D extent {};
	[[nodiscard]] bool Fresh(uint64_t now) const { return index != UINT32_MAX && frame == now; }
};

enum Job : uint8_t { JBuild = 0, JShadows, JReflections, JDebug };
struct PendingJob {
	Job      job   = JBuild;
	int32_t  queue = -1;
	uint64_t frame = 0;
};

// Skip reasons (5 s line): why a pass did not run or did not compose this frame.
enum Why : uint32_t {
	WOff = 0, WNoTlas, WNoCamera, WNoProj, WNoSun, WNoDepth, WNoTarget, WNoLit, WTargetFormat, WImages, WLate, WNoRough, WOrder, WCount
};
const char* WhyName(Why w) {
	static const char* n[] {"apagado (F9)", "sin TLAS de este frame", "sin camara", "sin view_to_clip", "sin sol",
	                        "sin profundidad del GBuffer", "sin imagen destino", "sin color iluminado", "formato destino no escribible",
	                        "imagenes propias", "tarde (otro frame)", "sin rugosidad del GBuffer",
	                        "orden (la pasada fuente no termino antes)"};
	return n[w < WCount ? w : 0];
}

struct State {
	bool ready = false, failed = false, said_on = false;
	GraphicContext* graphics = nullptr;
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             point = nullptr, linear = nullptr;
	VkBuffer      ubo = VK_NULL_HANDLE, stats = VK_NULL_HANDLE;
	VmaAllocation ubo_mem = nullptr, stats_mem = nullptr;
	uint32_t*     stats_map = nullptr;
	vk::QueryPool pool = nullptr;
	double        period_ns = 1.0;
	OwnImage dummy_u, dummy_f; // R32_UINT and RGBA16F 1x1 (sampled + storage)
	bool     dummies_general = false;
	OwnImage s_raw, s_hist[2], s_filt, r_raw, r_aux, r_hist[2], r_haux[2], r_filt;
	bool     s_history = false, r_history = false;
	uint64_t s_last = 0, r_last = 0; // the frame each history was last written (valid only from the frame before)
	struct Retired {
		uint64_t      frame = 0;
		vk::ImageView view  = nullptr;
		OwnImage      image;
	};
	std::deque<Retired> retired;

	// frame bookkeeping (GPU thread)
	uint64_t frame = 1;           // our flip counter
	uint64_t slot_frame[Slots] {}; // the frame each slot holds (0 none)
	bool     slot_used[Slots] {}, slot_timed[Slots] {};
	uint32_t s_cur = 0, r_cur = 0;
	Live     live;
	bool     live_on = true;
	int      cycle_window = 0;

	// scope roles
	uint8_t role[NativeRender::MaxPasses] {};
	bool    role_known[NativeRender::MaxPasses] {};
	int     open_count[RCount] {};

	// what the scopes showed this frame
	Seen depth, gb_layer[4], mask, ssr, lit;
	std::array<Seen, 8> lit_cand {};
	uint32_t            lit_n = 0;
	uint64_t            built_frame = 0, resolve_sun_frame = 0, ssr_frame = 0;
	std::deque<PendingJob> jobs;

	// camera (this frame's and the last one's)
	std::array<float, 16> inv {}, vp {}, pvp {};
	std::array<float, 4>  cam {}, pcam {};
	uint64_t              cam_frame = 0, pcam_frame = 0;
	vk::AccelerationStructureKHR tlas = nullptr;
	std::array<float, 16> view_to_clip {};
	bool                  have_proj = false;
	// the sun (toward the light, world)
	std::array<float, 3>  sun {};
	uint64_t              sun_frame = 0;
	std::vector<std::array<float, 3>> sun_axes; // this frame's candidates
	uint64_t              sun_axes_frame = 0;
	uint32_t              sun_draws = 0, sun_found = 0;
	bool                  sun_said = false;

	// calibrations
	bool   y_flip = true; // (hwrt5a, measured in DS: mirrored, 92 % vs 51 % agreement)
	double self_n = 0, self_a = 0, self_b = 0, self_miss = 0, self_near = 0, self_far = 0;
	double last_agree = 0, last_cover = 0, last_near = 0, last_far = 0; // (pt1d) the last judgement, for the 5 s line
	bool   self_judged = false; // (pt1e) at least one judgement made
	uint64_t sun_flip_frame = 0;
	float    sun_sign = 1.0f;
	bool   depth_ok = false;
	double sh_n = 0, sh_sv = 0, sh_svv = 0, sh_ch[4][3] {};
	int    sh_channel = -1;
	double sh_r[4] {};
	double rf_hist_a[64] {}, rf_hist_b[64] {}, rf_hist_c[64] {};
	double rf_gain = 0.0;
	int    rf_premul = -1; // -1 unknown
	bool   rf_ok = false;
	double rf_spread = 0.0;

	// the 5 s window
	std::chrono::steady_clock::time_point window_start = std::chrono::steady_clock::now();
	uint64_t win_frames = 0;
	double   win_gpu[Queries] {};
	uint32_t win_gpu_n[2] {};
	double   win_cpu_ms = 0.0;
	uint64_t win_skips[2][WCount] {};
	uint64_t win_runs[3] {};
	double   win_stat[96] {}; // summed words 0..63 of the read slots; (pt1e) 64..95: the fill's words 288..319
	double   win_rough[16] {};
	HwScene::PtTotals last_totals {};
	uint64_t win_capture_by_role[RCount] {}, win_capture_refused = 0;
	// per preset (BB_PT_CYCLE): the window GPU ms totals
	double   preset_ms[5] {};
	uint32_t preset_windows[5] {};
	// one-time logs
	uint32_t said_mask_n = 0;
	// (pt1b) the lit colour and the SSR result kept across frames (the same render targets every frame; checked each use)
	Seen lit_cache, ssr_cache;
	uint32_t win_lit_cand = 0;
	// (pt1b) sun-shadow image candidates: images written at the render size (or half) by a CS of ResolveSunShadows or
	// RenderLights, tried one at a time until one channel is the sun's visibility (correlation, range 0..1).
	struct Cand {
		uint64_t cs = 0;
		uint32_t binding = 0;
		Role     scope = RNone;
		Seen     last;
		bool     rejected = false;
		uint32_t trials = 0;
		bool     structural = false; // (pt1c) unorm, written in ResolveSunShadows by a CS that reads a layered depth (the cascades)
	};
	vk::Extent2D render_last {};
	uint64_t     win_debug = 0;      // (pt1c) the GBuffer depth's size, kept across frames
	Seen         taa_in;              // (pt1c) the colour the game's TAA reads (binding 0): the lit HDR image
	bool         said_taa = false;
	std::vector<Cand> cands;
	int      cand_cur = -1, cand_locked = -1;
	uint32_t cand_evals = 0;
	std::vector<uint64_t> cs_logged;
	double sh_over[4] {};
	bool said_mask = false, said_ssr = false, said_lit = false, said_depth = false, said_rough = false, said_consumer = false;
	bool said_mask_bad = false;
	uint64_t own_bytes = 0;
	// (pt1e) The sun mask's image kept across frames (kRTT_SunShadowResolve: one render target, made once per render size
	// in FUN_00917c30 [Ghidra]); pt1d needed the writing dispatch's hook in the same frame, which ran in ~5 % of frames (log
	// 074330: "sin imagen destino" 237 of 250): the dispatch is usually translated off the hook (another queue / path).
	Seen     mask_cache;
	uint64_t win_mask_cached = 0;
	// (pt1e) The first scope seen with a dispatch reading the mask (its consumer: the shadows compose before it).
	Role     consumer = RNone;
	uint64_t consumer_cs = 0;
	uint32_t consumer_binding = 0;
	// (pt1e) The reflections' fill: the per-mesh radiance cache, its timing and statistics, the A/B.
	VkBuffer      cache = VK_NULL_HANDLE;
	VmaAllocation cache_mem = nullptr;
	bool          cache_cleared = false;
	bool          slot_fill[Slots] {}, slot_on[Slots] {};
	uint64_t      fill_frame = 0;
	double        win_fill_gpu = 0.0;
	uint32_t      win_fill_n = 0;
	uint64_t      win_fill_runs = 0, win_fill_skips = 0;
	const char*   fill_why = nullptr;
	double        ab_ms[2] {};  // rt-reflections' whole chain, GPU ms a frame: [0] without our fill, [1] with it
	uint32_t      ab_n[2] {};
	double        ab_pt_ms[2] {}; // our GPU ms a frame (scene + shadows + fill) in the A/B's off / on windows
	uint32_t      ab_pt_n[2] {};
	// (pt1e) F9 / BB_PT_AB / cycle state per frame of the window: ON and OFF frames apart, and the proof that OFF costs 0
	uint64_t win_on_frames = 0, win_off_frames = 0, win_off_jobs = 0, win_off_slots = 0;
	double   win_off_gpu = 0.0;
	// (pt1e) budget control: the last window's ms a frame (ON frames), changes made
	double   last_total = 0.0;
	uint32_t scale_changes = 0;
} g;

Role RoleOf(uint32_t id) {
	if (id >= NativeRender::MaxPasses) return RNone;
	if (!g.role_known[id]) {
		g.role_known[id] = true;
		const char* name = NativeRender::PassRegistry::Name(id);
		Role        r    = RNone;
		if (name != nullptr) {
			using NativeRender::PassRegistry;
			if (PassRegistry::SameName(name, "GBuffer")) r = RGBuffer;
			else if (PassRegistry::SameName(name, "VirtualGeo")) r = RVirtualGeo;
			else if (PassRegistry::SameName(name, "ResolveSunShadows")) r = RResolveSunShadows;
			else if (PassRegistry::SameName(name, "RenderLights")) r = RRenderLights;
			else if (PassRegistry::SameName(name, "ResolveLighting")) r = RResolveLighting;
			else if (PassRegistry::SameName(name, "RenderSSR")) r = RRenderSSR;
			else if (PassRegistry::SameName(name, "RenderIBL")) r = RRenderIBL;
			else if (std::strncmp(name, "RenderShadowCascade", 19) == 0) r = RCascade;
			else if (PassRegistry::SameName(name, "PostProcessing")) r = RPostProcessing;
			if (r == RNone)
				for (const auto& s: Cfg().capture_scopes)
					if (PassRegistry::SameName(name, s.c_str())) r = RCapture;
		}
		g.role[id] = r;
	}
	return static_cast<Role>(g.role[id]);
}

bool CaptureRole(Role r) {
	if (r == RCapture) return true;
	for (const auto& s: Cfg().capture_scopes) {
		if (r == RGBuffer && NativeRender::PassRegistry::SameName(s.c_str(), "GBuffer")) return true;
		if (r == RVirtualGeo && NativeRender::PassRegistry::SameName(s.c_str(), "VirtualGeo")) return true;
	}
	return false;
}

std::atomic<bool> g_cascade_open {false};
std::atomic<bool> g_live_toggle {false}; // F9 pressed (window thread) -> applied on the GPU thread

// ---- Vulkan helpers ------------------------------------------------------------------------------------------------
void Barrier(vk::CommandBuffer cmd, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src, vk::PipelineStageFlags2 dst_stage,
             vk::AccessFlags2 dst) {
	vk::MemoryBarrier2 b {};
	b.srcStageMask  = src_stage;
	b.srcAccessMask = src;
	b.dstStageMask  = dst_stage;
	b.dstAccessMask = dst;
	vk::DependencyInfo dep {};
	dep.memoryBarrierCount = 1;
	dep.pMemoryBarriers    = &b;
	cmd.pipelineBarrier2(&dep);
}

bool MakeImage(GraphicContext& graphics, vk::Format format, vk::Extent2D extent, OwnImage& img) {
	VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create.imageType     = VK_IMAGE_TYPE_2D;
	create.format        = static_cast<VkFormat>(format);
	create.extent        = {std::max(extent.width, 1u), std::max(extent.height, 1u), 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.samples       = VK_SAMPLE_COUNT_1_BIT;
	create.tiling        = VK_IMAGE_TILING_OPTIMAL;
	create.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	create.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
	create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	VmaAllocationInfo info {};
	if (vmaCreateImage(graphics.allocator, &create, &allocate, &img.image, &img.memory, &info) != VK_SUCCESS) {
		img.image = VK_NULL_HANDLE;
		return false;
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = vk::Image(img.image);
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = format;
	view_info.subresourceRange = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &img.view) != vk::Result::eSuccess) {
		vmaDestroyImage(graphics.allocator, img.image, img.memory);
		img = {};
		return false;
	}
	img.extent  = {create.extent.width, create.extent.height};
	img.format  = format;
	img.general = false;
	img.bytes   = info.size;
	g.own_bytes += img.bytes;
	return true;
}
void RetireImage(OwnImage& img) {
	if (img.image == VK_NULL_HANDLE) return;
	g.own_bytes -= std::min(g.own_bytes, img.bytes);
	State::Retired r;
	r.frame = g.frame;
	r.image = img;
	g.retired.push_back(r);
	img = {};
}
void RetireView(vk::ImageView view) {
	if (view == nullptr) return;
	State::Retired r;
	r.frame = g.frame;
	r.view  = view;
	g.retired.push_back(r);
}
void CollectRetired(GraphicContext& graphics) {
	while (!g.retired.empty() && g.retired.front().frame + RetireWait <= g.frame) {
		auto& r = g.retired.front();
		if (r.view != nullptr) graphics.device.destroyImageView(r.view, nullptr);
		if (r.image.view != nullptr) graphics.device.destroyImageView(r.image.view, nullptr);
		if (r.image.image != VK_NULL_HANDLE) vmaDestroyImage(graphics.allocator, r.image.image, r.image.memory);
		g.retired.pop_front();
	}
}
void ToGeneral(vk::CommandBuffer cmd, OwnImage& img) {
	if (img.general || img.image == VK_NULL_HANDLE) return;
	vk::ImageMemoryBarrier2 b {};
	b.srcStageMask        = vk::PipelineStageFlagBits2::eTopOfPipe;
	b.dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
	b.dstAccessMask       = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite;
	b.oldLayout           = vk::ImageLayout::eUndefined;
	b.newLayout           = vk::ImageLayout::eGeneral;
	b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image               = vk::Image(img.image);
	b.subresourceRange    = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::DependencyInfo dep {};
	dep.imageMemoryBarrierCount = 1;
	dep.pImageMemoryBarriers    = &b;
	cmd.pipelineBarrier2(&dep);
	// a defined start (the temporal passes read the history before it is written once)
	vk::ClearColorValue zero {};
	zero.float32 = std::array<float, 4> {0.0f, 0.0f, 0.0f, 0.0f};
	if (img.format == vk::Format::eR32Uint) zero.uint32 = std::array<uint32_t, 4> {0, 0, 0, 0};
	const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	cmd.clearColorImage(vk::Image(img.image), vk::ImageLayout::eGeneral, &zero, 1, &whole);
	Barrier(cmd, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	img.general = true;
}

vk::DescriptorType DescType(uint32_t i) {
	if (i == 0) return vk::DescriptorType::eAccelerationStructureKHR;
	if (i == 17 || i == 25 || i == 26) return vk::DescriptorType::eStorageBuffer;
	if (i == 18) return vk::DescriptorType::eUniformBuffer;
	if ((i >= 5 && i <= 16) || i == 24) return vk::DescriptorType::eStorageImage;
	return vk::DescriptorType::eCombinedImageSampler; // (1..4, 19..23)
}
// (pt1e) The per-mesh radiance cache of the reflections' fill (pt.comp Fill*): CacheSlots x (vec4 mean + uvec4 sums + uint
// owner + uint idle frames); the technique (one mean lit radiance per mesh, learned from on-screen hits, used for hits off
// screen) is the GPL-2.0 Bloodborne path-tracing mod's per-BLAS material cache (its docs, "slice 5"); our own code.
constexpr uint32_t CacheSlots = 32768;
constexpr uint32_t CacheBytes = CacheSlots * (16 + 16 + 4 + 4);

bool Init(GraphicContext& graphics) {
	if (g.ready) return true;
	if (g.failed) return false;
	g.failed          = true;
	const auto device = graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, Bindings> bindings {};
	for (uint32_t i = 0; i < Bindings; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = DescType(i);
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = Bindings;
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Push)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = sizeof(PT_SPV);
	module_info.pCode    = PT_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g.layout;
	const auto created = device.createComputePipelines(LocalPassCache::Get(device), 1, &pipeline_info, nullptr, &g.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g.pipeline = nullptr;
		return false;
	}
	for (int i = 0; i < 2; ++i) {
		vk::SamplerCreateInfo sampler {};
		sampler.magFilter    = i == 0 ? vk::Filter::eNearest : vk::Filter::eLinear;
		sampler.minFilter    = sampler.magFilter;
		sampler.mipmapMode   = vk::SamplerMipmapMode::eNearest;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		sampler.maxLod       = 0.0f;
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g.point : &g.linear) != vk::Result::eSuccess) return false;
	}
	if (!MakeImage(graphics, vk::Format::eR32Uint, {1, 1}, g.dummy_u)) return false;
	if (!MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, {1, 1}, g.dummy_f)) return false;
	{
		VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		buffer.size  = 512;
		buffer.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo dev {};
		dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g.ubo, &g.ubo_mem, nullptr) != VK_SUCCESS) {
			g.ubo = VK_NULL_HANDLE;
			return false;
		}
	}
	{
		VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		buffer.size  = StatWords * Slots * 4;
		buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo host {};
		host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo mapped {};
		if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g.stats, &g.stats_mem, &mapped) != VK_SUCCESS) {
			g.stats = VK_NULL_HANDLE;
			return false;
		}
		g.stats_map = static_cast<uint32_t*>(mapped.pMappedData);
		if (g.stats_map == nullptr) return false;
		std::memset(g.stats_map, 0, StatWords * Slots * 4);
	}
	{
		// (pt1e) the fill's per-mesh radiance cache (device local, cleared on its first use)
		VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		buffer.size  = CacheBytes;
		buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo dev {};
		dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g.cache, &g.cache_mem, nullptr) != VK_SUCCESS) {
			g.cache = VK_NULL_HANDLE;
			return false;
		}
	}
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = Slots * Queries;
	if (device.createQueryPool(&query, nullptr, &g.pool) != vk::Result::eSuccess) g.pool = nullptr;
	g.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	g.graphics  = &graphics;
	g.failed    = false;
	g.ready     = true;
	std::printf("[pt] listo: pipeline pt.comp creado, %s; timestamps %s\n", g.pool != nullptr ? "consultas GPU" : "sin consultas",
	            g.pool != nullptr ? "si" : "no");
	std::fflush(stdout);
	return true;
}

// (Re)makes the trace images at the sizes the live preset asks for this render size. False: could not.
bool EnsureImages(GraphicContext& graphics, vk::Extent2D render) {
	const auto scaled = [&](float s) {
		return vk::Extent2D {std::max(1u, static_cast<uint32_t>(std::ceil(static_cast<float>(render.width) * s))),
		                     std::max(1u, static_cast<uint32_t>(std::ceil(static_cast<float>(render.height) * s)))};
	};
	const auto f16x4 = vk::Format::eR16G16B16A16Sfloat;
	bool       ok    = true;
	if (Cfg().shadows) {
		const vk::Extent2D se = scaled(std::max(0.25f, g.live.shadow_scale * g.live.scale)); // (pt1e: the budget's factor)
		if (g.s_raw.image == VK_NULL_HANDLE || g.s_raw.extent != se) {
			for (OwnImage* i: {&g.s_raw, &g.s_hist[0], &g.s_hist[1], &g.s_filt}) RetireImage(*i);
			ok = ok && MakeImage(graphics, f16x4, se, g.s_raw) && MakeImage(graphics, f16x4, se, g.s_hist[0]) &&
			     MakeImage(graphics, f16x4, se, g.s_hist[1]) && MakeImage(graphics, f16x4, se, g.s_filt);
			g.s_history = false;
		}
	}
	if (Cfg().reflections && Cfg().refl_legacy) { // (pt1e: the fill needs none of these; ultra's were 92 MB)
		const vk::Extent2D re = scaled(g.live.refl_scale);
		if (g.r_raw.image == VK_NULL_HANDLE || g.r_raw.extent != re) {
			for (OwnImage* i: {&g.r_raw, &g.r_aux, &g.r_hist[0], &g.r_hist[1], &g.r_haux[0], &g.r_haux[1], &g.r_filt}) RetireImage(*i);
			ok = ok && MakeImage(graphics, f16x4, re, g.r_raw) && MakeImage(graphics, f16x4, re, g.r_aux) &&
			     MakeImage(graphics, f16x4, re, g.r_hist[0]) && MakeImage(graphics, f16x4, re, g.r_hist[1]) &&
			     MakeImage(graphics, f16x4, re, g.r_haux[0]) && MakeImage(graphics, f16x4, re, g.r_haux[1]) &&
			     MakeImage(graphics, f16x4, re, g.r_filt);
			g.r_history = false;
		}
	}
	return ok;
}

// A view of a game image subresource for this frame only (retired after RetireWait frames: the image may be gone by
// then; the frame's commands that use it are done).
vk::ImageView FrameView(GraphicContext& graphics, Image& image, vk::Format format, vk::ImageAspectFlags aspect, uint32_t level,
                        uint32_t layer) {
	vk::ImageViewCreateInfo info {};
	info.image            = image.backing.image;
	info.viewType         = vk::ImageViewType::e2D;
	info.format           = format;
	info.subresourceRange = vk::ImageSubresourceRange {aspect, level, 1, layer, 1};
	vk::ImageView view    = nullptr;
	if (graphics.device.createImageView(&info, nullptr, &view) != vk::Result::eSuccess) return nullptr;
	RetireView(view);
	return view;
}

bool IsDepthFormat(vk::Format f) {
	return f == vk::Format::eD32Sfloat || f == vk::Format::eD32SfloatS8Uint || f == vk::Format::eD16Unorm ||
	       f == vk::Format::eD24UnormS8Uint || f == vk::Format::eD16UnormS8Uint || f == vk::Format::eX8D24UnormPack32;
}
bool IsFloatColour(vk::Format f) {
	return f == vk::Format::eR16G16B16A16Sfloat || f == vk::Format::eB10G11R11UfloatPack32 || f == vk::Format::eR32G32B32A32Sfloat ||
	       f == vk::Format::eE5B9G9R9UfloatPack32;
}
const char* FormatName(vk::Format f) {
	switch (f) {
		case vk::Format::eR8Unorm: return "R8_UNORM";
		case vk::Format::eR8G8Unorm: return "RG8_UNORM";
		case vk::Format::eR8G8B8A8Unorm: return "RGBA8_UNORM";
		case vk::Format::eR16Sfloat: return "R16F";
		case vk::Format::eR16G16Sfloat: return "RG16F";
		case vk::Format::eR16G16B16A16Sfloat: return "RGBA16F";
		case vk::Format::eB10G11R11UfloatPack32: return "R11G11B10F";
		case vk::Format::eR32Sfloat: return "R32F";
		case vk::Format::eR32Uint: return "R32_UINT";
		case vk::Format::eR16Unorm: return "R16_UNORM";
		case vk::Format::eR16G16Unorm: return "RG16_UNORM";
		case vk::Format::eD32SfloatS8Uint: return "D32F_S8";
		case vk::Format::eD32Sfloat: return "D32F";
		default: return "otro";
	}
}

// ---- the trace passes ----------------------------------------------------------------------------------------------
struct Bound {
	Image*        image = nullptr;
	vk::ImageView view  = nullptr;
	vk::Extent2D  extent {};
	vk::Format    format = vk::Format::eUndefined;
};

// The game image a Seen names, its subresource moved to GENERAL for our compute access, and a view of it.
bool BindGame(GraphicContext& graphics, TextureCache& cache, vk::CommandBuffer cmd, const Seen& s, bool write, bool depth_aspect,
              Bound& out, vk::ImageView* storage_view = nullptr) {
	Image* image = cache.TryGetImage(ImageId(s.index, s.generation));
	if (image == nullptr || !image->registered || image->backing.image == nullptr || image->backing.samples != 1) return false;
	const auto& b = image->backing;
	if (s.level >= b.mip_levels || s.layer >= b.layers) return false;
	const vk::Format f = b.format;
	if (depth_aspect != IsDepthFormat(f)) return false;
	const ImageSubresourceRange range {s.level, 1, s.layer, 1};
	vk::AccessFlags2 access = vk::AccessFlagBits2::eShaderSampledRead | vk::AccessFlagBits2::eShaderStorageRead;
	if (write) access |= vk::AccessFlagBits2::eShaderStorageWrite;
	image->Transit(vk::ImageLayout::eGeneral, access, range, cmd);
	out.image  = image;
	out.format = f;
	out.extent = {std::max(1u, b.extent.width >> s.level), std::max(1u, b.extent.height >> s.level)};
	out.view   = FrameView(graphics, *image, f, depth_aspect ? vk::ImageAspectFlags(vk::ImageAspectFlagBits::eDepth)
	                                                         : vk::ImageAspectFlags(vk::ImageAspectFlagBits::eColor),
	                       s.level, s.layer);
	if (out.view == nullptr) return false;
	if (storage_view != nullptr) {
		*storage_view = nullptr;
		if ((b.usage & vk::ImageUsageFlagBits::eStorage) &&
		    (graphics.GetFormatProperties(f).optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage))
			*storage_view = FrameView(graphics, *image, f, vk::ImageAspectFlagBits::eColor, s.level, s.layer);
	}
	return true;
}

void FillUbo(FrameUbo& u, vk::Extent2D render, vk::Extent2D strace, vk::Extent2D rtrace, vk::Extent2D target, int flags, int channel,
             uint32_t slot, uint32_t hist, int debug) {
	std::memset(&u, 0, sizeof(u));
	std::memcpy(u.inv, g.inv.data(), 64);
	std::memcpy(u.vp, g.vp.data(), 64);
	std::memcpy(u.pvp, g.pvp.data(), 64);
	u.cam[0] = g.cam[0], u.cam[1] = g.cam[1], u.cam[2] = g.cam[2], u.cam[3] = 1.0f;
	const bool prev = g.pcam_frame != 0 && g.pcam_frame + 1 == g.cam_frame;
	u.pcam[0] = g.pcam[0], u.pcam[1] = g.pcam[1], u.pcam[2] = g.pcam[2], u.pcam[3] = prev ? 1.0f : 0.0f;
	const auto& m = g.view_to_clip;
	u.proj[0] = m[0], u.proj[1] = m[5], u.proj[2] = m[8], u.proj[3] = m[9];
	u.depth_map[0] = m[10], u.depth_map[1] = m[14], u.depth_map[2] = m[11], u.depth_map[3] = m[15];
	u.sun[0] = g.sun[0] * g.sun_sign, u.sun[1] = g.sun[1] * g.sun_sign, u.sun[2] = g.sun[2] * g.sun_sign;
	u.sun[3] = static_cast<float>(std::cos(Cfg().sun_deg * 3.14159265358979 / 180.0));
	const auto size4 = [](float* d, vk::Extent2D e) {
		d[0] = static_cast<float>(e.width), d[1] = static_cast<float>(e.height);
		d[2] = 1.0f / static_cast<float>(std::max(e.width, 1u)), d[3] = 1.0f / static_cast<float>(std::max(e.height, 1u));
	};
	size4(u.render, render);
	size4(u.strace, strace);
	size4(u.rtrace, rtrace);
	size4(u.target, target);
	u.par[0]  = g.live.shadow_dist;
	u.par[1]  = g.live.refl_dist;
	u.par[2]  = g.live.rough_cutoff;
	u.par[3]  = Cfg().strength;
	u.par2[0] = Cfg().gain > 0.0f ? Cfg().gain : static_cast<float>(g.rf_gain > 0.0 ? g.rf_gain : 1.0);
	u.par2[1] = g.live.shadow_history;
	u.par2[2] = g.live.refl_history;
	u.par2[3] = 1.0f;
	u.cfg[0]  = static_cast<int32_t>(g.frame & 0x7fffffffu);
	u.cfg[1]  = flags;
	u.cfg[2]  = channel;
	u.cfg[3]  = Cfg().rough_layer < 0 ? -1 : (Cfg().rough_byte | (Cfg().rough_layer << 8));
	u.cfg2[0] = static_cast<int32_t>(slot * StatWords);
	u.cfg2[1] = static_cast<int32_t>(hist);
	u.cfg2[2] = debug;
	u.cfg2[3] = g.live.refl_spp;
}

struct PassImages {
	vk::DescriptorImageInfo info[Bindings] {};
};

// (pt1e) `exposure`: binding 26 (the game's exposure buffer for the fill; else the stats buffer stands in, never read)
void Dispatch(vk::CommandBuffer cmd, const PassImages& pi, int mode, vk::Extent2D size, const vk::DescriptorBufferInfo* exposure = nullptr) {
	vk::WriteDescriptorSetAccelerationStructureKHR as_write {};
	as_write.accelerationStructureCount = 1;
	as_write.pAccelerationStructures    = &g.tlas;
	const vk::DescriptorBufferInfo ubo_info {vk::Buffer(g.ubo), 0, sizeof(FrameUbo)};
	const vk::DescriptorBufferInfo st_info {vk::Buffer(g.stats), 0, StatWords * Slots * 4};
	const vk::DescriptorBufferInfo cache_info {vk::Buffer(g.cache), 0, CacheBytes};
	vk::WriteDescriptorSet         writes[Bindings] {};
	for (uint32_t i = 0; i < Bindings; ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = DescType(i);
		if (i == 0) writes[i].pNext = &as_write;
		else if (i == 17) writes[i].pBufferInfo = &st_info;
		else if (i == 18) writes[i].pBufferInfo = &ubo_info;
		else if (i == 25) writes[i].pBufferInfo = &cache_info;
		else if (i == 26) writes[i].pBufferInfo = exposure != nullptr ? exposure : &st_info;
		else writes[i].pImageInfo = &pi.info[i];
	}
	Push push {};
	push.mode = mode;
	cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g.layout, 0, Bindings, writes);
	cmd.pushConstants(g.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	cmd.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
	Barrier(cmd, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eShaderWrite,
	        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
}

PassImages BaseImages(const Bound& depth) {
	PassImages pi;
	const auto gen = vk::ImageLayout::eGeneral;
	for (uint32_t i = 1; i < Bindings; ++i) pi.info[i] = {g.point, g.dummy_f.view, gen};
	pi.info[1] = {g.point, depth.view, gen};
	pi.info[2] = {g.point, g.dummy_u.view, gen};
	pi.info[3] = {g.linear, g.dummy_f.view, gen};
	pi.info[4] = {g.point, g.dummy_f.view, gen};
	for (uint32_t i = 5; i <= 16; ++i) pi.info[i] = {nullptr, g.dummy_f.view, gen};
	for (uint32_t i = 19; i <= 23; ++i) pi.info[i] = {g.point, g.dummy_f.view, gen}; // (pt1e: the fill's inputs)
	pi.info[24] = {nullptr, g.dummy_f.view, gen};                                     // (pt1e: the fill's output)
	const auto own = [&](uint32_t b, const OwnImage& img) {
		if (img.view != nullptr) pi.info[b] = {nullptr, img.view, gen};
	};
	own(6, g.s_raw), own(7, g.s_hist[0]), own(8, g.s_hist[1]), own(9, g.s_filt);
	own(10, g.r_raw), own(11, g.r_aux), own(12, g.r_hist[0]), own(13, g.r_hist[1]), own(14, g.r_haux[0]), own(15, g.r_haux[1]);
	own(16, g.r_filt);
	return pi;
}

uint32_t SlotOf(uint64_t frame) { return static_cast<uint32_t>(frame % Slots); }

// The frame's first GPU work: its statistics slot zeroed, its timestamps reset.
void BeginSlot(vk::CommandBuffer cmd) {
	const uint32_t s = SlotOf(g.frame);
	if (g.slot_frame[s] == g.frame) return;
	g.slot_frame[s] = g.frame;
	g.slot_used[s]  = true;
	g.slot_timed[s] = false;
	g.slot_fill[s]  = false;
	g.slot_on[s]    = g.live_on && g.live.on; // (pt1e: the proof that an OFF frame records nothing)
	Barrier(cmd, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
	cmd.fillBuffer(vk::Buffer(g.stats), vk::DeviceSize {s} * StatWords * 4, StatWords * 4, 0);
	if (g.pool != nullptr) cmd.resetQueryPool(g.pool, s * Queries, Queries);
	Barrier(cmd, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
}
void Stamp(vk::CommandBuffer cmd, uint32_t q) {
	if (g.pool == nullptr) return;
	const uint32_t s = SlotOf(g.frame);
	cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g.pool, s * Queries + q);
	if (q == 0 || q == 4) g.slot_timed[s] = true;
	if (q == 8) g.slot_fill[s] = true; // (pt1e: the fill's pair, read apart)
}
void UploadUbo(vk::CommandBuffer cmd, const FrameUbo& u) {
	Barrier(cmd, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eUniformRead, vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eTransferWrite);
	cmd.updateBuffer(vk::Buffer(g.ubo), 0, sizeof(u), &u);
	Barrier(cmd, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eUniformRead);
}

// The game's lit colour of this frame: the last float colour image ResolveLighting wrote at the depth's size (its
// draws' targets and its dispatches' written images, a ring of the last 8). (pt1b) pt1 found it in 1 of ~300 frames on
// the PC: the candidates of a frame were capped at the first 8. The choice is kept across frames (the same render target
// every frame) and checked at each use: still registered, float colour, the render size.
bool UsableLit(TextureCache& cache, vk::Extent2D render, Seen& s) {
	if (s.index == UINT32_MAX) return false;
	Image* image = cache.TryGetImage(ImageId(s.index, s.generation));
	if (image == nullptr || !image->registered || image->backing.image == nullptr) return false;
	const vk::Extent2D e {std::max(1u, image->backing.extent.width >> s.level), std::max(1u, image->backing.extent.height >> s.level)};
	if (!IsFloatColour(image->backing.format) || e != render) return false;
	s.format = image->backing.format;
	s.extent = e;
	return true;
}
bool PickLit(TextureCache& cache, vk::Extent2D render, Seen& out) {
	// (pt1c) First the image the game's TAA reads (the same render target every frame: at RenderIBL it already holds this
	// frame's lighting, without the specular IBL); pt1b found no ResolveLighting candidate at the render size at all.
	{
		Seen t = g.taa_in;
		if (UsableLit(cache, render, t)) {
			out = t;
			return true;
		}
	}
	g.win_lit_cand += std::min<uint32_t>(g.lit_n, static_cast<uint32_t>(g.lit_cand.size()));
	const uint32_t n = std::min<uint32_t>(g.lit_n, static_cast<uint32_t>(g.lit_cand.size()));
	for (uint32_t k = 0; k < n; ++k) {
		Seen s = g.lit_cand[(g.lit_n - 1 - k) % g.lit_cand.size()];
		if (!UsableLit(cache, render, s)) continue;
		out         = s;
		g.lit_cache = s;
		if (!g.said_lit) {
			g.said_lit = true;
			std::printf("[pt] color iluminado (ResolveLighting): %s %ux%u, capa %u, nivel %u (de %u candidatos)\n", FormatName(s.format),
			            s.extent.width, s.extent.height, s.layer, s.level, g.lit_n);
			std::fflush(stdout);
		}
		return true;
	}
	Seen c = g.lit_cache;
	if (UsableLit(cache, render, c)) {
		out = c;
		return true;
	}
	return false;
}

// (pt1e) The shadows' gate: min(game, ours) only adds shadow, so the TLAS geometry the capture misses ("detras" in the
// autotest: GPU-written draws) costs nothing but some of our shadow; extra or stale geometry ("delante") would draw a
// wrong shadow. Composition needs: the hits at the depth's distance >= 90 % (as pt1d), or, for min() only, the hits
// nearer than the depth <= 5 % (log 074330: 1-7 % "delante", 2-46 % "detras").
bool ShadowGateOk() {
	if (g.depth_ok) return true;
	return !Cfg().replace && g.last_cover >= 0.5 && g.last_near <= 0.05 && g.self_judged;
}
int ShadowFlags(bool gate) {
	int f = 0;
	if (g.y_flip) f |= 1;
	if (g.s_history && g.s_last + 1 == g.frame) f |= 2;
	const bool calibrated = Cfg().channel >= 0 || g.sh_channel >= 0;
	if ((calibrated && ShadowGateOk()) || Cfg().force) f |= 4;
	if (Cfg().replace) f |= 8;
	f |= 64;  // dynamic bit in the custom index (hwrt-scene.cpp under BB_PATHTRACING)
	f |= 256; // the self-test every frame (1 pixel of 16x16, two rays)
	if (g.live.shadow_checker) f |= 2048;                                  // (pt1e) half the texels a frame
	if (gate && calibrated && !Cfg().replace && Cfg().debug != 2) f |= 4096; // (pt1e) no ray where the game's mask is dark
	if (!g.live.shadow_dynamic) f |= 8192;                                 // (pt1e) static occluders only (mask 0x01)
	return f;
}
int ReflFlags(bool rough) {
	int f = 0;
	if (g.y_flip) f |= 1;
	if (g.r_history && g.r_last + 1 == g.frame) f |= 2;
	const bool calibrated = Cfg().gain > 0.0f || g.rf_ok;
	if ((calibrated && g.depth_ok) || Cfg().force) f |= 16;
	if (g.rf_premul == 1) f |= 32;
	f |= 64;
	if (rough) f |= 128;
	if (!Cfg().shadows) f |= 256; // (the depth <-> TLAS self-test runs in the reflection trace when there are no shadows)
	return f;
}

bool Skip(int part, Why why) {
	++g.win_skips[part][why];
	return false;
}

// Common preconditions of a trace this frame (the camera of this frame's GBuffer and its TLAS).
bool SceneReady(int part) {
	if (!g.live_on || !g.live.on) return Skip(part, WOff);
	if (g.built_frame != g.frame || g.tlas == nullptr) return Skip(part, WNoTlas);
	if (g.cam_frame != g.frame) return Skip(part, WNoCamera);
	if (!g.have_proj) return Skip(part, WNoProj);
	if (!g.depth.Fresh(g.frame)) return Skip(part, WNoDepth);
	return true;
}

// (pt1b) The sun-shadow candidate in use: the one whose calibration passed, else the one on trial (the first not
// rejected). -1: none (log: no image of the render size written in ResolveSunShadows / RenderLights yet).
int CurrentCand() {
	if (g.cand_locked >= 0) return g.cand_locked;
	if (g.cand_cur >= 0 && g.cand_cur < static_cast<int>(g.cands.size()) && !g.cands[g.cand_cur].rejected) return g.cand_cur;
	for (size_t i = 0; i < g.cands.size(); ++i)
		if (!g.cands[i].rejected) {
			g.cand_cur = static_cast<int>(i);
			g.cand_evals = 0;
			g.sh_n = g.sh_sv = g.sh_svv = 0.0;
			std::memset(g.sh_ch, 0, sizeof(g.sh_ch));
			std::memset(g.sh_over, 0, sizeof(g.sh_over));
			return g.cand_cur;
		}
	return -1;
}

// (pt1e) The mask image for this frame: the one the writing dispatch was seen with this frame, else the same image kept
// from an earlier frame (still registered, same format and size). Only while ResolveSunShadows ran this frame (the job's
// rule: it writes the image every frame before its consumers).
bool MaskTarget(TextureCache& cache, int ci, Seen& out) {
	if (ci < 0 || ci >= static_cast<int>(g.cands.size())) return false;
	const Seen& last = g.cands[ci].last;
	if (last.Fresh(g.frame)) {
		g.mask_cache = last;
		out          = last;
		return true;
	}
	if (g.resolve_sun_frame != g.frame) return false;
	Seen c = g.mask_cache;
	if (c.index != UINT32_MAX && c.index == last.index && c.generation == last.generation)
		if (Image* im = cache.TryGetImage(ImageId(c.index, c.generation));
		    im != nullptr && im->registered && im->backing.image != nullptr && im->backing.format == last.format &&
		    vk::Extent2D {std::max(1u, im->backing.extent.width >> c.level), std::max(1u, im->backing.extent.height >> c.level)} == last.extent) {
			c.frame = g.frame;
			out     = c;
			++g.win_mask_cached;
			return true;
		}
	return false;
}

void RunShadows(CommandBuffer& buffer) {
	if (!Cfg().shadows || !SceneReady(0)) return;
	if (g.sun_frame == 0 || g.sun_frame + 120 < g.frame) return (void)Skip(0, WNoSun);
	const int ci = CurrentCand();
	auto& graphics = buffer.GetContext().GetGraphics();
	auto& cache    = buffer.GetContext().GetTextureCache();
	// (pt1c) No target this frame: the trace still runs (the debug tint needs it); only the composition is left out.
	// (pt1e) The target kept across frames (MaskTarget); the calibration samples are taken in the trace now, so a target
	// must be there for it too.
	Seen target;
	const bool have_target = MaskTarget(cache, ci, target);
	if (!have_target) ++g.win_skips[0][WNoTarget];
	if (have_target) g.mask = target;
	const auto t0  = std::chrono::steady_clock::now();
	buffer.EndRendering();
	const vk::CommandBuffer cmd = buffer.Handle();
	Bound depth, mask;
	vk::ImageView mask_store = nullptr;
	if (!BindGame(graphics, cache, cmd, g.depth, false, true, depth)) return (void)Skip(0, WNoDepth);
	const bool bound = have_target && BindGame(graphics, cache, cmd, g.mask, true, false, mask, &mask_store);
	bool compose = bound;
	if (compose && mask_store == nullptr) {
		if (!g.said_mask_bad) {
			g.said_mask_bad = true;
			std::printf("[pt] sombras: la imagen destino (%s) no admite escritura como storage: solo medida\n", FormatName(mask.format));
			std::fflush(stdout);
		}
		++g.win_skips[0][WTargetFormat];
		compose = false;
	}
	if (!EnsureImages(graphics, depth.extent)) return (void)Skip(0, WImages);
	BeginSlot(cmd);
	for (OwnImage* i: {&g.dummy_u, &g.dummy_f, &g.s_raw, &g.s_hist[0], &g.s_hist[1], &g.s_filt}) ToGeneral(cmd, *i);
	Barrier(cmd, vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
	        vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eAccelerationStructureWriteKHR, vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eAccelerationStructureReadKHR);
	const uint32_t slot = SlotOf(g.frame);
	g.s_cur ^= 1u;
	FrameUbo u {};
	FillUbo(u, depth.extent, g.s_raw.extent, g.r_raw.image != VK_NULL_HANDLE ? g.r_raw.extent : vk::Extent2D {1, 1},
	        bound ? mask.extent : vk::Extent2D {1, 1}, ShadowFlags(bound) | (bound ? 16384 : 0), Cfg().channel >= 0 ? Cfg().channel : g.sh_channel,
	        slot, g.s_cur, Cfg().debug == 2 ? 1 : 0);
	UploadUbo(cmd, u);
	PassImages pi = BaseImages(depth);
	if (bound) pi.info[4] = {g.point, mask.view, vk::ImageLayout::eGeneral};
	if (compose) pi.info[5] = {nullptr, mask_store, vk::ImageLayout::eGeneral};
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipeline);
	Stamp(cmd, 0);
	// (pt1e) checkerboard: a thread per pair of horizontal texels (the traced one chosen by the frame's parity)
	Dispatch(cmd, pi, 0, g.live.shadow_checker ? vk::Extent2D {(g.s_raw.extent.width + 1) / 2, g.s_raw.extent.height} : g.s_raw.extent);
	Stamp(cmd, 1);
	Dispatch(cmd, pi, 1, g.s_raw.extent);
	Dispatch(cmd, pi, 2, g.s_raw.extent);
	Stamp(cmd, 2);
	if (compose) Dispatch(cmd, pi, 3, mask.extent);
	Stamp(cmd, 3);
	// our writes -> whatever the game does next
	Barrier(cmd, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	g.s_history = true;
	g.s_last    = g.frame;
	++g.win_runs[1];
	g.win_cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void RunReflections(CommandBuffer& buffer) {
	if (!Cfg().reflections || !SceneReady(1)) return;
	if (g.ssr.Fresh(g.frame)) g.ssr_cache = g.ssr;
	// (pt1b) the SSR result of this frame, else the same image as last time (RenderSSR ran this frame: the job's rule)
	if (!g.ssr.Fresh(g.frame)) {
		Image* im = g.ssr_cache.index != UINT32_MAX ? buffer.GetContext().GetTextureCache().TryGetImage(ImageId(g.ssr_cache.index, g.ssr_cache.generation))
		                                            : nullptr;
		if (im == nullptr || !im->registered || im->backing.format != vk::Format::eR16G16B16A16Sfloat) return (void)Skip(1, WNoTarget);
		g.ssr       = g.ssr_cache;
		g.ssr.frame = g.frame;
	}
	auto& graphics = buffer.GetContext().GetGraphics();
	auto& cache    = buffer.GetContext().GetTextureCache();
	const auto t0  = std::chrono::steady_clock::now();
	buffer.EndRendering();
	const vk::CommandBuffer cmd = buffer.Handle();
	Bound depth, ssr, lit, rough;
	vk::ImageView ssr_store = nullptr;
	if (!BindGame(graphics, cache, cmd, g.depth, false, true, depth)) return (void)Skip(1, WNoDepth);
	Seen lit_seen;
	if (!PickLit(cache, depth.extent, lit_seen)) return (void)Skip(1, WNoLit);
	if (!BindGame(graphics, cache, cmd, lit_seen, false, false, lit)) return (void)Skip(1, WNoLit);
	if (!BindGame(graphics, cache, cmd, g.ssr, true, false, ssr, &ssr_store) || ssr_store == nullptr) return (void)Skip(1, WTargetFormat);
	bool have_rough = false;
	if (Cfg().rough_layer >= 0) {
		const Seen& rl = g.gb_layer[Cfg().rough_layer];
		have_rough     = rl.Fresh(g.frame) && BindGame(graphics, cache, cmd, rl, false, false, rough) && rough.format == vk::Format::eR32Uint;
		if (!g.said_rough) {
			g.said_rough = true;
			std::printf("[pt] rugosidad: capa %d byte %d del array del GBuffer: %s\n", Cfg().rough_layer, Cfg().rough_byte,
			            have_rough ? "enlazada (R32_UINT)" : "NO encontrada: se trata toda superficie como brillante solo hasta el corte");
			std::fflush(stdout);
		}
	}
	// Without the roughness every surface would count as a mirror: no reflections then (BB_PT_ROUGH=none forces it).
	if (!have_rough && Cfg().rough_layer >= 0) return (void)Skip(1, WNoRough);
	if (!EnsureImages(graphics, depth.extent)) return (void)Skip(1, WImages);
	BeginSlot(cmd);
	for (OwnImage* i: {&g.dummy_u, &g.dummy_f, &g.r_raw, &g.r_aux, &g.r_hist[0], &g.r_hist[1], &g.r_haux[0], &g.r_haux[1], &g.r_filt})
		ToGeneral(cmd, *i);
	Barrier(cmd, vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
	        vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eAccelerationStructureWriteKHR, vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eAccelerationStructureReadKHR);
	const uint32_t slot = SlotOf(g.frame);
	g.r_cur ^= 1u;
	FrameUbo u {};
	FillUbo(u, depth.extent, g.s_raw.image != VK_NULL_HANDLE ? g.s_raw.extent : vk::Extent2D {1, 1}, g.r_raw.extent, ssr.extent,
	        ReflFlags(have_rough), -1, slot, g.r_cur, Cfg().debug == 3 ? 2 : 0);
	UploadUbo(cmd, u);
	PassImages pi = BaseImages(depth);
	if (have_rough) pi.info[2] = {g.point, rough.view, vk::ImageLayout::eGeneral};
	pi.info[3] = {g.linear, lit.view, vk::ImageLayout::eGeneral};
	pi.info[4] = {g.point, ssr.view, vk::ImageLayout::eGeneral};
	pi.info[5] = {nullptr, ssr_store, vk::ImageLayout::eGeneral};
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipeline);
	Stamp(cmd, 4);
	Dispatch(cmd, pi, 4, g.r_raw.extent);
	Stamp(cmd, 5);
	Dispatch(cmd, pi, 5, g.r_raw.extent);
	Dispatch(cmd, pi, 6, g.r_raw.extent);
	Stamp(cmd, 6);
	Dispatch(cmd, pi, 7, ssr.extent);
	Stamp(cmd, 7);
	Barrier(cmd, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	g.r_history = true;
	g.r_last    = g.frame;
	++g.win_runs[2];
	g.win_cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// (pt1c) BB_PT_DEBUG=1: the tint of the lit HDR image at the start of PostProcessing (pt.comp DebugTint).
void RunDebug(CommandBuffer& buffer) {
	if (Cfg().debug != 1 || !g.live_on || !g.live.on || g.tlas == nullptr) return;
	const bool sh = g.s_last == g.frame && g.s_filt.image != VK_NULL_HANDLE, rf = g.r_last == g.frame && g.r_filt.image != VK_NULL_HANDLE;
	if (!sh && !rf) return;
	if (!g.depth.Fresh(g.frame) || g.taa_in.index == UINT32_MAX) return;
	auto& graphics = buffer.GetContext().GetGraphics();
	auto& cache    = buffer.GetContext().GetTextureCache();
	buffer.EndRendering();
	const vk::CommandBuffer cmd = buffer.Handle();
	Bound depth, lit;
	vk::ImageView lit_store = nullptr;
	if (!BindGame(graphics, cache, cmd, g.depth, false, true, depth)) return;
	Seen t = g.taa_in;
	if (!UsableLit(cache, depth.extent, t) || !BindGame(graphics, cache, cmd, t, true, false, lit, &lit_store) || lit_store == nullptr) return;
	FrameUbo u {};
	FillUbo(u, depth.extent, g.s_filt.image != VK_NULL_HANDLE ? g.s_filt.extent : vk::Extent2D {1, 1},
	        g.r_filt.image != VK_NULL_HANDLE ? g.r_filt.extent : vk::Extent2D {1, 1}, lit.extent, (sh ? 512 : 0) | (rf ? 1024 : 0), -1,
	        SlotOf(g.frame), 0, 0);
	UploadUbo(cmd, u);
	PassImages pi = BaseImages(depth);
	pi.info[4]    = {g.point, lit.view, vk::ImageLayout::eGeneral};
	pi.info[5]    = {nullptr, lit_store, vk::ImageLayout::eGeneral};
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipeline);
	Dispatch(cmd, pi, 8, lit.extent);
	Barrier(cmd, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	++g.win_debug;
}

void RunBuild(CommandBuffer& buffer) {
	auto& graphics = buffer.GetContext().GetGraphics();
	const auto t0  = std::chrono::steady_clock::now();
	buffer.EndRendering();
	const vk::CommandBuffer cmd = buffer.Handle();
	HwScene::EndFrameAtScope(graphics, cmd);
	HwScene::GiSceneInfo scene;
	if (HwScene::GiScene(scene)) {
		g.built_frame = g.frame;
		g.tlas        = scene.tlas;
		if (g.cam_frame != 0 && g.cam_frame != g.frame) {
			g.pvp        = g.vp;
			g.pcam       = g.cam;
			g.pcam_frame = g.cam_frame;
		}
		g.inv       = scene.inv;
		g.vp        = scene.vp;
		g.cam       = scene.cam;
		g.cam_frame = g.frame;
	}
	++g.win_runs[0];
	(void)t0; // (pt1b: the build's CPU time is HwScene's, in "captura"; pt1 counted it twice)
}

// ---- (pt1e) the reflections' fill (rt-reflections.cpp calls RecordReflFill) -------------------------------------------
// Why not the reflection passes of pt1..pt1d: they decided glossiness from a G-buffer byte (40-81 % of the pixels traced
// in Boletaria, dry stone included) and wrote into the game's SSR image. "Reflejos RT" already reflects only where the
// game does (its SSR guide, its material response: rt-reflections.cpp refl8 / rt5), and looks better (the user's A/B): the
// hardware rays only fill its screen-space misses there.
uint32_t FillBlock() { return std::clamp<uint32_t>(g.live.fill_block + (g.live.scale < 0.7f ? 1u : 0u), 1u, 4u); }

const char* FillBlocked() {
	if (!Cfg().requested || !g.ready || g.pipeline == nullptr || g.cache == VK_NULL_HANDLE) return "sin iniciar";
	if (!g.live_on || !g.live.on) return "apagado (F9)";
	if (!Cfg().reflections) return "BB_PT_REFLECTIONS=0";
	if (Cfg().refl_legacy) return "BB_PT_REFL_LEGACY=1";
	if (g.built_frame != g.frame || g.tlas == nullptr) return "sin TLAS de este frame";
	if (g.cam_frame != g.frame) return "sin camara";
	if (!ShadowGateOk() && !Cfg().force) return "autotest (geometria de mas en el TLAS)";
	return nullptr;
}

void FillFrameUbo(FrameUbo& u, const ReflFillIo& io, bool history) {
	FillUbo(u, io.render, vk::Extent2D {1, 1}, io.trace, io.render, (g.y_flip ? 1 : 0) | 64 | (history ? 2 : 0) | (io.row0_up ? 16384 : 0) |
	                                                                     (Cfg().debug == 1 ? 32768 : 0),
	        -1, SlotOf(g.frame), 0, 0);
	const auto& m = io.view_to_clip;
	u.fproj[0] = m[0], u.fproj[1] = m[5], u.fproj[2] = m[8], u.fproj[3] = m[9];
	u.fdepth[0] = m[10], u.fdepth[1] = m[14], u.fdepth[2] = m[11], u.fdepth[3] = m[15];
	u.fsize[0] = static_cast<float>(io.render.width), u.fsize[1] = static_cast<float>(io.render.height);
	u.fsize[2] = static_cast<float>(io.trace.width), u.fsize[3] = static_cast<float>(io.trace.height);
	u.fpar[0]  = g.live.refl_dist * 4.0f; // (the ray's reach, world units)
	u.fpar[1]  = 0.85f;                   // share of a hit coloured from the previous frame
	u.fpar[2]  = 0.6f;                    // share of a hit coloured from its mesh's learned radiance
	u.fpar[3]  = 4.0f;                    // frames of learning before a mesh's radiance is used
	u.cfg3[0]  = io.exposure != nullptr ? static_cast<int32_t>(io.exposure_index) : -1;
	u.cfg3[1]  = static_cast<int32_t>(FillBlock());
	u.cfg3[2]  = 16; // (the learning grid: one primary ray per 16 x 16 render pixels)
	u.cfg3[3]  = static_cast<int32_t>(std::clamp(io.rough_cutoff, 0.0f, 1.0f) * 1000.0f + 0.5f); // (per mille)
}

// ---- the sun from the cascades' light matrix -----------------------------------------------------------------------
// An orthographic 4x4 (row-major: rows x, y, depth, then (0, 0, 0, 1)) or its 3x4 rows: mutually orthogonal rows,
// x and y the same scale (a square map), depth a different one. The depth row is the light axis.
bool FindOrtho(std::span<const uint32_t> srt, std::array<float, 3>& axis) {
	const size_t n = std::min<size_t>(srt.size(), 96);
	for (size_t i = 0; i + 12 <= n; ++i) {
		float m[16];
		for (size_t k = 0; k < 12; ++k) std::memcpy(&m[k], &srt[i + k], 4);
		bool finite = true;
		for (size_t k = 0; k < 12; ++k) finite = finite && std::isfinite(m[k]) && std::fabs(m[k]) < 1e7f;
		if (!finite) continue;
		const auto row = [&](int r) { return std::array<double, 3> {m[r * 4], m[r * 4 + 1], m[r * 4 + 2]}; };
		const auto dot = [](const std::array<double, 3>& a, const std::array<double, 3>& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
		const auto r0 = row(0), r1 = row(1), r2 = row(2);
		const double l0 = std::sqrt(dot(r0, r0)), l1 = std::sqrt(dot(r1, r1)), l2 = std::sqrt(dot(r2, r2));
		if (l0 < 1e-9 || l1 < 1e-9 || l2 < 1e-9) continue;
		if (std::fabs(dot(r0, r1)) / (l0 * l1) > 0.01 || std::fabs(dot(r0, r2)) / (l0 * l2) > 0.01 || std::fabs(dot(r1, r2)) / (l1 * l2) > 0.01)
			continue;
		if (std::fabs(l0 - l1) > 0.01 * std::max(l0, l1)) continue;
		if (std::fabs(l2 - l0) < 0.01 * std::max(l0, l2)) continue; // a rigid / uniform matrix, not a projection
		// a 4x4: its fourth row must be (0, 0, 0, 1)
		if (i + 16 <= n) {
			float w[4];
			for (size_t k = 0; k < 4; ++k) std::memcpy(&w[k], &srt[i + 12 + k], 4);
			const bool proj_row = std::fabs(w[0]) < 1e-6f && std::fabs(w[1]) < 1e-6f && std::fabs(w[2]) < 1e-6f && std::fabs(w[3] - 1.0f) < 1e-4f;
			const bool affine   = std::fabs(m[3]) + std::fabs(m[7]) + std::fabs(m[11]) > 0.0f;
			if (!proj_row && !affine) continue;
		}
		axis = {static_cast<float>(r2[0] / l2), static_cast<float>(r2[1] / l2), static_cast<float>(r2[2] / l2)};
		return true;
	}
	return false;
}

void SettleSun() {
	if (g.sun_axes.empty()) return;
	uint32_t best = 0, best_n = 0;
	for (uint32_t i = 0; i < g.sun_axes.size(); ++i) {
		uint32_t n = 0;
		for (const auto& b: g.sun_axes) {
			const auto& a = g.sun_axes[i];
			n += std::fabs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) > 0.99985f ? 1u : 0u; // 1 degree
		}
		if (n > best_n) best = i, best_n = n;
	}
	auto a = g.sun_axes[best];
	// Toward the light: the sign that puts it above the horizon (world +y up: the camera's yaw is about y, KBM / CAM1).
	if (a[1] < 0.0f) a = {-a[0], -a[1], -a[2]};
	g.sun       = a;
	g.sun_frame = g.sun_axes_frame;
	++g.sun_found;
	if (!g.sun_said) {
		g.sun_said = true;
		std::printf("[pt] sol: eje de la luz de la cascada (%.4f %.4f %.4f), elevacion %.1f grados, %u de %zu draws de cascada de acuerdo\n",
		            a[0], a[1], a[2], std::asin(std::clamp(a[1], -1.0f, 1.0f)) * 57.29578f, best_n, g.sun_axes.size());
		std::fflush(stdout);
	}
	g.sun_axes.clear();
}

// ---- statistics read back and calibrations -------------------------------------------------------------------------
double Median64(const double* h, double* spread) {
	double total = 0.0;
	for (int i = 0; i < 64; ++i) total += h[i];
	if (total <= 0.0) return 0.0;
	const auto quant = [&](double q) {
		double acc = 0.0;
		for (int i = 0; i < 64; ++i) {
			acc += h[i];
			if (acc >= q * total) return (static_cast<double>(i) + 0.5) / 8.0 - 4.0;
		}
		return 4.0;
	};
	if (spread != nullptr) *spread = (quant(0.75) - quant(0.25)) * 0.5;
	return quant(0.5);
}

void ReadStats() {
	if (g.stats_map == nullptr || g.frame < ReadDelay) return;
	const uint64_t f = g.frame - ReadDelay;
	const uint32_t s = SlotOf(f);
	if (!g.slot_used[s] || g.slot_frame[s] != f) return;
	g.slot_used[s] = false;
	(void)vmaInvalidateAllocation(g.graphics->allocator, g.stats_mem, vk::DeviceSize {s} * StatWords * 4, StatWords * 4);
	const uint32_t* w = g.stats_map + s * StatWords;
	for (int i = 0; i < 64; ++i) g.win_stat[i] += w[i];
	for (int i = 0; i < 16; ++i) g.win_rough[i] += w[256 + i];
	// self-test
	g.self_n += w[0], g.self_a += w[1], g.self_b += w[2], g.self_miss += w[3];
	g.self_near += w[5], g.self_far += w[6];
	// shadow channel sums
	g.sh_n += w[16], g.sh_sv += w[17] / 1024.0, g.sh_svv += w[18] / 1024.0;
	for (int c = 0; c < 4; ++c)
		for (int k = 0; k < 3; ++k) g.sh_ch[c][k] += w[20 + c * 3 + k] / 1024.0;
	for (int c = 0; c < 4; ++c) g.sh_over[c] += w[36 + c]; // (pt1b: samples above 1.01: not a visibility)
	for (int i = 0; i < 64; ++i) {
		g.rf_hist_a[i] += w[64 + i];
		g.rf_hist_b[i] += w[128 + i];
		g.rf_hist_c[i] += w[192 + i];
	}
	for (int i = 0; i < 32; ++i) g.win_stat[64 + i] += w[288 + i]; // (pt1e: the fill's counters -> window words 64..95)
	// (pt1e) the proof that OFF frames record nothing: a slot used by an OFF frame
	if (!g.slot_on[s]) ++g.win_off_slots;
	// (pt1e) the fill's GPU time (its own pair of queries)
	if (g.pool != nullptr && g.slot_fill[s]) {
		std::array<uint64_t, 4> r {};
		if (g.graphics->device.getQueryPoolResults(g.pool, s * Queries + 8, 2, sizeof(r), r.data(), 2 * sizeof(uint64_t),
		                                           vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability) ==
		        vk::Result::eSuccess &&
		    r[1] != 0 && r[3] != 0 && r[2] >= r[0]) {
			const double ms = static_cast<double>(r[2] - r[0]) * g.period_ns / 1e6;
			g.win_fill_gpu += ms;
			++g.win_fill_n;
			if (!g.slot_on[s]) g.win_off_gpu += ms;
		}
		g.slot_fill[s] = false;
	}
	// timestamps
	if (g.pool != nullptr && g.slot_timed[s]) {
		// (each half on its own: a pass that did not run this frame leaves its four queries unavailable)
		for (uint32_t half = 0; half < 2; ++half) {
			std::array<uint64_t, 8> r {};
			if (g.graphics->device.getQueryPoolResults(g.pool, s * Queries + half * 4, 4, sizeof(r), r.data(), 2 * sizeof(uint64_t),
			                                           vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability) !=
			    vk::Result::eSuccess)
				continue;
			if (r[1] == 0 || r[3] == 0 || r[5] == 0 || r[7] == 0) continue;
			const auto ms = [&](uint32_t a, uint32_t b) { return static_cast<double>(r[b * 2] - r[a * 2]) * g.period_ns / 1e6; };
			g.win_gpu[half * 4 + 0] += ms(0, 1), g.win_gpu[half * 4 + 1] += ms(1, 2), g.win_gpu[half * 4 + 2] += ms(2, 3);
			++g.win_gpu_n[half];
			if (!g.slot_on[s]) g.win_off_gpu += ms(0, 3);
		}
		g.slot_timed[s] = false;
	}
}

void Calibrate() {
	// Depth <-> TLAS (the screen's y convention and the depth's distance against the captured geometry).
	// (pt1d) The gate before any composition: of the primary rays (the shadow rays' mask 0x03) that hit, >= 90 % must hit
	// at the depth's distance (3 % + 5 cm); the share that hit nothing is the TLAS's coverage, logged apart (pt1c's 50 %
	// counted the misses - geometry the capture does not have - as disagreement). Nearer / farther tell why it fails.
	if (g.self_n >= 200.0) {
		const double a = g.self_a / g.self_n, b = g.self_b / g.self_n;
		if (g.self_b > g.self_a * 1.2 && b > 0.5) {
			g.y_flip = !g.y_flip;
			g.s_history = g.r_history = false;
			std::printf("[pt] autotest: la convencion y contraria acierta mas (%.0f %% frente a %.0f %%): se cambia\n", b * 100.0, a * 100.0);
		}
		const double hits  = std::max(1.0, g.self_n - g.self_miss);
		g.self_judged      = true;
		g.last_agree       = g.self_a / hits;
		g.last_cover       = hits / g.self_n;
		g.last_near        = g.self_near / hits;
		g.last_far         = g.self_far / hits;
		const bool was     = g.depth_ok;
		g.depth_ok         = g.last_agree >= 0.9 && g.last_cover >= 0.5;
		if (was != g.depth_ok)
			std::printf("[pt] autotest profundidad<->TLAS: de los rayos que chocan, %.0f %% a la distancia de la profundidad (%.0f %% delante, %.0f %% "
			            "detras); cobertura %.0f %%; y %s -> composicion %s\n",
			            g.last_agree * 100.0, g.last_near * 100.0, g.last_far * 100.0, g.last_cover * 100.0, g.y_flip ? "invertida" : "directa",
			            g.depth_ok ? "permitida" : "BLOQUEADA (< 90 % o cobertura < 50 %)");
		std::fflush(stdout);
		g.self_near = g.self_far = 0.0;
		g.self_n = g.self_a = g.self_b = g.self_miss = 0.0;
	}
	// The sun-shadow channel: Pearson r of each channel of the candidate image with our visibility; (pt1b) and its range:
	// a visibility stays within 0..1 (a lighting image does not, and min() with it would be wrong). A candidate that fails
	// three judgements (with shadow in view) is dropped and the next one tried.
	if (g.sh_n >= 20000.0) {
		const double n = g.sh_n, mv = g.sh_sv / n, vv = g.sh_svv / n - mv * mv;
		int    best   = -1;
		double best_r = 0.0;
		for (int c = 0; c < 4; ++c) {
			const double mx = g.sh_ch[c][0] / n, vx = g.sh_ch[c][1] / n - mx * mx, cov = g.sh_ch[c][2] / n - mx * mv;
			g.sh_r[c] = vv > 1e-6 && vx > 1e-6 ? cov / std::sqrt(vv * vx) : 0.0;
			const bool in_range = g.sh_over[c] / n < 0.01;
			if (in_range && g.sh_r[c] > best_r) best_r = g.sh_r[c], best = c;
		}
		const int ci = g.cand_locked >= 0 ? g.cand_locked : g.cand_cur;
		if (vv > 0.01 && ci >= 0 && ci < static_cast<int>(g.cands.size())) { // (enough shadow in view to judge)
			auto& cand = g.cands[ci];
			// (pt1c) a structural mask (unorm, resolved from the cascades) needs r >= 0.25 (our TLAS still misses the GPU-culled
			// geometry: the agreement is partial); any other image r >= 0.5 as before.
			if (best >= 0 && best_r >= (cand.structural ? 0.25 : 0.5)) {
				if (g.cand_locked != ci || g.sh_channel != best)
					std::printf("[pt] sombras: candidata #%d (CS %016llx binding %u) canal %d = visibilidad del sol (r %.2f %.2f %.2f %.2f, fuera de "
					            "0..1 %.1f/%.1f/%.1f/%.1f %%, n %.0f): composicion permitida\n",
					            ci, static_cast<unsigned long long>(cand.cs), cand.binding, best, g.sh_r[0], g.sh_r[1], g.sh_r[2], g.sh_r[3],
					            g.sh_over[0] / n * 100.0, g.sh_over[1] / n * 100.0, g.sh_over[2] / n * 100.0, g.sh_over[3] / n * 100.0, n);
				g.cand_locked = ci;
				g.sh_channel  = best;
				g.cand_evals  = 0;
			} else if (g.cand_locked == ci && g.sh_r[g.sh_channel >= 0 ? g.sh_channel : 0] < 0.2) {
				std::printf("[pt] sombras: la candidata #%d dejo de correlacionar (r %.2f): composicion parada, se vuelve a probar\n", ci,
				            g.sh_r[g.sh_channel >= 0 ? g.sh_channel : 0]);
				g.cand_locked = -1;
				g.sh_channel  = -1;
			} else if (cand.structural && best < 0 && *std::min_element(g.sh_r, g.sh_r + 4) <= -0.25 && g.frame > g.sun_flip_frame + 600) {
				// (pt1d) The mask is the sun's (structure + [G]) but our visibility anti-correlates: our sun points the wrong way.
				g.sun_flip_frame = g.frame;
				g.sun_sign       = -g.sun_sign;
				g.s_history      = false;
				std::printf("[pt] sombras: la mascara del sol ANTI-correlaciona con nuestra traza (r %.2f %.2f %.2f %.2f): se invierte el sentido del "
				            "sol\n", g.sh_r[0], g.sh_r[1], g.sh_r[2], g.sh_r[3]);
			} else if (g.cand_locked != ci && ++g.cand_evals >= 3) {
				cand.rejected = true;
				g.cand_cur    = -1;
				g.cand_evals  = 0;
				std::printf("[pt] sombras: candidata #%d (CS %016llx binding %u) descartada: ningun canal es la visibilidad del sol (r %.2f %.2f %.2f "
				            "%.2f; fuera de 0..1 %.0f/%.0f/%.0f/%.0f %%)\n",
				            ci, static_cast<unsigned long long>(cand.cs), cand.binding, g.sh_r[0], g.sh_r[1], g.sh_r[2], g.sh_r[3],
				            g.sh_over[0] / n * 100.0, g.sh_over[1] / n * 100.0, g.sh_over[2] / n * 100.0, g.sh_over[3] / n * 100.0);
				bool left = false;
				for (const auto& c: g.cands) left |= !c.rejected;
				if (!left)
					std::printf("[pt] sombras: NINGUNA imagen de ResolveSunShadows / RenderLights a resolucion de pantalla es la visibilidad del sol "
					            "(%zu probadas): sin composicion de sombras (ver las lineas [pt] CS ...)\n", g.cands.size());
			}
			std::fflush(stdout);
		}
		g.sh_n = g.sh_sv = g.sh_svv = 0.0;
		std::memset(g.sh_ch, 0, sizeof(g.sh_ch));
		std::memset(g.sh_over, 0, sizeof(g.sh_over));
	}
	// The SSR colour scale and alpha convention.
	double na = 0.0, nb = 0.0;
	for (int i = 0; i < 64; ++i) na += g.rf_hist_a[i], nb += g.rf_hist_b[i];
	if (na >= 2000.0) {
		double spread = 0.0;
		const double med = Median64(g.rf_hist_a, &spread);
		g.rf_spread      = spread;
		const bool was   = g.rf_ok;
		g.rf_ok          = spread < 1.0; // half the inter-quartile range under one stop
		g.rf_gain        = std::pow(2.0, med);
		if (nb >= 300.0) {
			const double mb = Median64(g.rf_hist_b, nullptr), mc = Median64(g.rf_hist_c, nullptr);
			g.rf_premul = std::fabs(mc - med) < std::fabs(mb - med) ? 1 : 0;
		}
		if (was != g.rf_ok || !was)
			std::printf("[pt] reflejos: escala SSR del juego / nuestra = %.3f (log2 %.2f, dispersion %.2f, n %.0f), alfa %s -> composicion %s\n",
			            g.rf_gain, med, spread, na,
			            g.rf_premul < 0 ? "sin muestras (se trata como no premultiplicado)" : g.rf_premul == 1 ? "premultiplicado" : "no premultiplicado",
			            g.rf_ok ? "permitida" : "BLOQUEADA (dispersion alta)");
		std::fflush(stdout);
		std::memset(g.rf_hist_a, 0, sizeof(g.rf_hist_a));
		std::memset(g.rf_hist_b, 0, sizeof(g.rf_hist_b));
		std::memset(g.rf_hist_c, 0, sizeof(g.rf_hist_c));
	}
}

double HeapUsageMb(GraphicContext& graphics, double* budget_mb) {
	const VkPhysicalDeviceMemoryProperties* props = nullptr;
	vmaGetMemoryProperties(graphics.allocator, &props);
	if (props == nullptr) return 0.0;
	std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets {};
	vmaGetHeapBudgets(graphics.allocator, budgets.data());
	double use = 0.0, bud = 0.0;
	for (uint32_t h = 0; h < props->memoryHeapCount; ++h)
		if (props->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
			use += static_cast<double>(budgets[h].usage) / 1048576.0;
			bud += static_cast<double>(budgets[h].budget) / 1048576.0;
		}
	if (budget_mb != nullptr) *budget_mb = bud;
	return use;
}

void Log() {
	const auto now = std::chrono::steady_clock::now();
	const double secs = std::chrono::duration<double>(now - g.window_start).count();
	if (secs < 5.0) return;
	const double frames = static_cast<double>(std::max<uint64_t>(g.win_frames, 1));
	const HwScene::PtTotals t = HwScene::GetPtTotals();
	const HwScene::PtTotals& p = g.last_totals;
	const double scene_frames  = static_cast<double>(std::max<uint64_t>(t.frames - p.frames, 1));
	const double cap_cpu       = (t.cpu_ms - p.cpu_ms) / frames;
	const double build_gpu     = t.timed > p.timed ? (t.gpu_ms - p.gpu_ms) / static_cast<double>(t.timed - p.timed) : 0.0;
	const double sn = std::max<uint32_t>(g.win_gpu_n[0], 1), rn = std::max<uint32_t>(g.win_gpu_n[1], 1);
	const double sh[3] {g.win_gpu[0] / sn, g.win_gpu[1] / sn, g.win_gpu[2] / sn};
	const double rf[3] {g.win_gpu[4] / rn, g.win_gpu[5] / rn, g.win_gpu[6] / rn};
	// per-frame costs: a pass that ran on fewer frames than the window costs its share
	// (pt1e) per ON frame (F9 / A/B / cycle windows): OFF frames record nothing (their own line proves it)
	const double on_frames = static_cast<double>(std::max<uint64_t>(g.win_on_frames, 1));
	const double builds    = static_cast<double>(t.frames - p.frames);
	const double sh_share = g.win_gpu_n[0] / on_frames, rf_share = g.win_gpu_n[1] / on_frames;
	const double fill_ms  = g.win_fill_n > 0 ? g.win_fill_gpu / g.win_fill_n : 0.0, fill_share = g.win_fill_n / on_frames;
	const double scene_ms = g.win_on_frames > 0 ? build_gpu * std::min(builds, on_frames) / on_frames : 0.0;
	const double sh_ms    = (sh[0] + sh[1] + sh[2]) * sh_share;
	const double total    = scene_ms + sh_ms + (rf[0] + rf[1] + rf[2]) * rf_share + fill_ms * fill_share;
	const int    preset   = g.live.on ? g.live.preset : 4;
	const char*  pname    = preset < 4 ? kPresets[preset].name_es : "apagado";
	g.preset_ms[preset] += total;
	++g.preset_windows[preset];
	if (Cfg().ab) {
		const int ph = g.live.on ? 1 : 0;
		g.ab_pt_ms[ph] += total;
		++g.ab_pt_n[ph];
	}
	double budget_mb = 0.0;
	const double heap_mb  = HeapUsageMb(*g.graphics, &budget_mb);
	const double own_mb   = static_cast<double>(g.own_bytes + 512 + StatWords * Slots * 4 + (g.cache != VK_NULL_HANDLE ? CacheBytes : 0)) / 1048576.0;
	const double fixed_mb = static_cast<double>(t.fixed_bytes) / 1048576.0;
	const double mesh_mb  = static_cast<double>(t.mesh_bytes) / 1048576.0;
	const double pt_mb    = own_mb + fixed_mb + mesh_mb;
	std::printf("[pt] 5 s: preset %s, %s; %.0f frames (%.1f fps de juego): %llu ON, %llu OFF; GPU ms por frame ON: escena (captura+BLAS+TLAS) "
	            "%.3f (%.3f por build) | sombras %.3f (traza %.3f temporal+espacial %.3f composicion %.3f, en %.0f %% de frames ON) | relleno de "
	            "reflejos %.3f (%.3f por frame con relleno, en %.0f %%)%s | TOTAL %.3f ms (presupuesto %.1f, escala %.2f)\n",
	            pname, g.live_on ? "ON (F9)" : "OFF (F9)", frames, frames / secs, static_cast<unsigned long long>(g.win_on_frames),
	            static_cast<unsigned long long>(g.win_off_frames), scene_ms, build_gpu, sh_ms, sh[0], sh[1], sh[2], sh_share * 100.0, fill_ms * fill_share,
	            fill_ms, fill_share * 100.0, rf_share > 0.0 ? " | reflejos antiguos (BB_PT_REFL_LEGACY) en el total" : "", total, g.live.budget_ms,
	            g.live.scale);
	// (pt1e) OFF must cost 0: no job ran, no slot was recorded, no GPU time, no capture (the scene's notes and builds)
	if (g.win_off_frames > 0)
		std::printf("[pt] 5 s: OFF %llu frames: trabajos PT %llu, frames con trabajo GPU %llu, GPU %.3f ms, builds de escena en total %.0f "
		            "(%s)\n",
		            static_cast<unsigned long long>(g.win_off_frames), static_cast<unsigned long long>(g.win_off_jobs),
		            static_cast<unsigned long long>(g.win_off_slots), g.win_off_gpu, builds,
		            g.win_off_jobs == 0 && g.win_off_slots == 0 && g.win_off_gpu == 0.0 ? "coste 0: OK" : "NO es 0: revisar");
	std::printf("[pt] 5 s: CPU hilo GPU ms/frame: captura (notas de draws + build) %.3f, nuestro registro %.3f, total %.3f (objetivo <= 0,3) | "
	            "draws vistos %.0f/frame, capturados %.1f/frame, mallas nuevas %.1f/frame, refits %.1f/frame, rechazadas por presupuesto %llu\n",
	            cap_cpu, g.win_cpu_ms / frames, cap_cpu + g.win_cpu_ms / frames, static_cast<double>(t.draws - p.draws) / scene_frames,
	            static_cast<double>(t.captured - p.captured) / scene_frames, static_cast<double>(t.new_meshes - p.new_meshes) / scene_frames,
	            static_cast<double>(t.refits - p.refits) / scene_frames, static_cast<unsigned long long>(t.budget_skips - p.budget_skips));
	std::printf("[pt] 5 s: VRAM PT %.1f MB (imagenes propias %.1f, escena fija %.1f, mallas %.1f de las que BLAS %.1f; presupuesto mallas %u MB) | "
	            "VRAM del proceso %.0f de %.0f MB | escena: %u instancias TLAS, %u mallas (%u moviles, %u con discard fuera de los rayos)\n",
	            pt_mb, own_mb, fixed_mb, mesh_mb, static_cast<double>(t.blas_bytes) / 1048576.0,
	            static_cast<unsigned>(MeshBudgetBytes() >> 20u), heap_mb, budget_mb, t.instances, t.meshes, t.dynamic, t.discard);
	const double* w = g.win_stat;
	const auto pct = [](double a, double b) { return b > 0.0 ? a / b * 100.0 : 0.0; };
	std::printf("[pt] 5 s: sol (%.3f %.3f %.3f) visto hace %llu frames | sombras: %.0f %% ocluidos (%.0f %% por mallas moviles), historia perdida %.1f %%, "
	            "canal %d (r %.2f/%.2f/%.2f/%.2f), compuestos %.1f %% | autotest %s, y %s\n",
	            g.sun[0], g.sun[1], g.sun[2], static_cast<unsigned long long>(g.sun_frame != 0 ? g.frame - g.sun_frame : 0), pct(w[9], w[8]),
	            pct(w[10], w[9]), pct(w[11], w[8]), Cfg().channel >= 0 ? Cfg().channel : g.sh_channel, g.sh_r[0], g.sh_r[1], g.sh_r[2],
	            g.sh_r[3], pct(w[32], w[33]), g.depth_ok ? "OK" : "pendiente/no casa", g.y_flip ? "invertida" : "directa");
	if (Cfg().refl_legacy) {
		double rough_total = 0.0;
		for (double r: g.win_rough) rough_total += r;
		double glossy = 0.0;
		for (int i = 0; i < 16; ++i)
			if ((static_cast<double>(i) + 0.5) / 16.0 <= g.live.rough_cutoff) glossy += g.win_rough[i];
		std::printf("[pt] 5 s: reflejos: superficies bajo el corte %.2f: %.0f %% | rayos %.0f: en pantalla %.0f %%, fuera/ocultos %.0f %% (se queda el "
		            "cubemap), sin impacto %.0f %%, cielo %.0f %% | historia perdida %.1f %% | escala %.3f (disp %.2f) %s, alfa %s, px compuestos ~%.0f/frame\n",
		            g.live.rough_cutoff, pct(glossy, rough_total), w[41], pct(w[42], w[41]), pct(w[43], w[41]), pct(w[44], w[41]), pct(w[45], w[41]),
		            pct(w[46], w[40]), Cfg().gain > 0.0f ? Cfg().gain : g.rf_gain, g.rf_spread, g.rf_ok || Cfg().gain > 0.0f ? "OK" : "pendiente",
		            g.rf_premul < 0 ? "?" : g.rf_premul == 1 ? "premult" : "directo", w[47] * 16.0 / frames);
	} else {
		// (pt1e) the fill over "Reflejos RT" (pt.comp FillStat words 288.. -> window words 64..)
		const double* f = w + 64;
		std::printf("[pt] 5 s: relleno de reflejos (sobre Reflejos RT): en %.0f %% de los frames ON%s%s; Reflejos RT refleja %.1f %% de su traza; "
		            "su rayo de pantalla fallo en %.1f %% de esos px (agua omitida %.1f %%); rayos HW %.0f/frame (1 por %ux%u px), impacto %.0f %%: "
		            "color de este frame %.0f %%, del frame anterior %.0f %%, de la cache por malla %.0f %%, sin color %.0f %% (queda el del juego); "
		            "px rellenados %.1f %% de los reflectantes; aprendizaje %.0f rayos/frame (%.0f %% validos)\n",
		            g.win_fill_runs / on_frames * 100.0, FrameGen::RtReflectionsLevel() <= 0 ? " (Reflejos RT apagado: sin reflejos de PT)" : g.win_fill_skips > 0 ? ", no: " : "",
		            FrameGen::RtReflectionsLevel() > 0 && g.win_fill_skips > 0 && g.fill_why != nullptr ? g.fill_why : "",
		            pct(f[0], f[13]), pct(f[1], f[0]), pct(f[11], f[0]), f[2] / std::max(1.0, static_cast<double>(g.win_fill_n)), FillBlock(), FillBlock(),
		            pct(f[3], f[2]), pct(f[4], f[3]), pct(f[5], f[3]), pct(f[6], f[3]), pct(f[7], f[3]), pct(f[8], f[0]),
		            f[9] / std::max(1.0, static_cast<double>(g.win_fill_n)), pct(f[10], f[9]));
		// (pt1e) A/B against "Reflejos RT" alone: its whole chain's GPU ms a frame with and without our fill (rt-reflections'
		// own timestamps), and with BB_PT_AB=1 our whole cost in the windows with and without PT
		std::printf("[pt] A/B reflejos: Reflejos RT solo %.3f ms/frame (%u frames) | Reflejos RT + relleno PT %.3f ms/frame (%u frames) -> "
		            "diferencia %+.3f ms (la pasada del relleno mide %.3f ms)%s\n",
		            g.ab_n[0] > 0 ? g.ab_ms[0] / g.ab_n[0] : 0.0, g.ab_n[0], g.ab_n[1] > 0 ? g.ab_ms[1] / g.ab_n[1] : 0.0, g.ab_n[1],
		            g.ab_n[0] > 0 && g.ab_n[1] > 0 ? g.ab_ms[1] / g.ab_n[1] - g.ab_ms[0] / g.ab_n[0] : 0.0, fill_ms,
		            g.ab_n[0] == 0 ? " (sin frames sin relleno: pulsa F9 o usa BB_PT_AB=1)" : g.ab_n[1] == 0 ? " (sin frames con relleno)" : "");
		if (Cfg().ab)
			std::printf("[pt] A/B PT (BB_PT_AB=1, ventanas de 5 s): sin PT %.3f ms (%u ventanas) | con PT %.3f ms (%u ventanas)\n",
			            g.ab_pt_n[0] > 0 ? g.ab_pt_ms[0] / g.ab_pt_n[0] : 0.0, g.ab_pt_n[0], g.ab_pt_n[1] > 0 ? g.ab_pt_ms[1] / g.ab_pt_n[1] : 0.0,
			            g.ab_pt_n[1]);
		std::memset(g.ab_ms, 0, sizeof(g.ab_ms));
		std::memset(g.ab_n, 0, sizeof(g.ab_n));
	}
	std::string why[2];
	for (int part = 0; part < 2; ++part)
		for (uint32_t k = 0; k < WCount; ++k)
			if (g.win_skips[part][k] != 0) why[part] += std::string(why[part].empty() ? "" : ", ") + WhyName(static_cast<Why>(k)) + " " +
			                                            std::to_string(g.win_skips[part][k]);
	std::printf("[pt] 5 s: ejecuciones: escena %llu, sombras %llu, reflejos %llu | sin sombras: %s | sin reflejos: %s | captura por pasada: "
	            "GBuffer %llu, VirtualGeo %llu, otras %llu, fuera de ambito %llu%s\n",
	            static_cast<unsigned long long>(g.win_runs[0]), static_cast<unsigned long long>(g.win_runs[1]),
	            static_cast<unsigned long long>(g.win_runs[2]), why[0].empty() ? "-" : why[0].c_str(), why[1].empty() ? "-" : why[1].c_str(),
	            static_cast<unsigned long long>(g.win_capture_by_role[RGBuffer]), static_cast<unsigned long long>(g.win_capture_by_role[RVirtualGeo]),
	            static_cast<unsigned long long>(g.win_capture_by_role[RCapture]), static_cast<unsigned long long>(g.win_capture_refused),
	            kPresets[g.live.preset].gi ? " | GI 1 rebote: NO implementada en pt1 (pendiente)" : "");
	// (pt1b) at a glance: the share of the target images' pixels our passes changed in this window
	{
		const double sh_vis = w[33] > 0.0 ? w[32] / w[33] * 100.0 : 0.0, rf_vis = w[49] > 0.0 ? w[47] / w[49] * 100.0 : 0.0;
		const int    ci     = g.cand_locked;
		std::printf("[pt] VISIBLE: sombras %.1f %% px, reflejos %.1f %% px%s | sombras: candidata %s (%zu vistas, %zu descartadas) | reflejos: color "
		            "iluminado %s (%.1f candidatos/frame), SSR %s | escena: %llu reordenaciones tomadas como estaticas, PS con descarte por alfa %u / "
		            "por otra cosa %u\n",
		            sh_vis, rf_vis, Cfg().debug == 1 ? (g.win_debug > 0 ? " (vista de depuracion: rojo = sombra trazada, azul = reflejo trazado)" : " (depuracion pedida pero sin pasar: falta color del TAA o trazas)") : "",
		            ci >= 0 ? "fijada" : g.cand_cur >= 0 ? "en prueba" : "ninguna", g.cands.size(),
		            static_cast<size_t>(std::count_if(g.cands.begin(), g.cands.end(), [](const State::Cand& c) { return c.rejected; })),
		            g.lit_cache.index != UINT32_MAX ? "fijado" : "no encontrado", static_cast<double>(g.win_lit_cand) / frames,
		            g.ssr_cache.index != UINT32_MAX ? "fijado" : "no", static_cast<unsigned long long>(t.reorders - p.reorders), t.ps_alpha,
		            t.ps_discard_other);
		// (pt1c) what the TLAS misses: draws whose arguments / indices the GPU wrote (GPU-culled geometry) are not captured
		std::printf("[pt] 5 s: autotest (ultimo juicio): %.0f %% de los impactos a la distancia de la profundidad, %.0f %% delante, %.0f %% detras, "
		            "cobertura %.0f %% -> %s; sol %s\n",
		            g.last_agree * 100.0, g.last_near * 100.0, g.last_far * 100.0, g.last_cover * 100.0,
		            g.depth_ok ? "composicion permitida" : ShadowGateOk() ? "sombras (min) y relleno permitidos: <= 5 % delante; lo que falta solo quita sombra"
		                                                      : "composicion BLOQUEADA (objetivo >= 90 %, o <= 5 % delante)",
		            g.sun_sign > 0.0f ? "por encima del horizonte" : "INVERTIDO por la mascara");
		std::printf("[pt] 5 s: draws sin capturar: escritos por la GPU %.1f/frame (argumentos indirectos %.1f, indices %.1f), sin pipeline de "
		            "captura %.1f/frame; color HDR del TAA %s\n",
		            static_cast<double>(t.gpu_written - p.gpu_written) / on_frames,
		            static_cast<double>(t.gpu_written_args - p.gpu_written_args) / on_frames,
		            static_cast<double>(t.gpu_written_indices - p.gpu_written_indices) / on_frames,
		            static_cast<double>(t.no_variant - p.no_variant) / on_frames,
		            g.taa_in.index != UINT32_MAX ? "fijado" : "no visto");
	}
	if (Cfg().cycle) {
		std::string per;
		for (int i = 0; i < 5; ++i)
			if (g.preset_windows[i] != 0) {
				char text[64];
				std::snprintf(text, sizeof(text), "%s%s %.3f ms (%u)", per.empty() ? "" : ", ", i < 4 ? kPresets[i].name_es : "apagado",
				              g.preset_ms[i] / g.preset_windows[i], g.preset_windows[i]);
				per += text;
			}
		std::printf("[pt] 5 s: GPU ms medios por preset (ventanas): %s\n", per.c_str());
	}
	std::fflush(stdout);
	g.last_totals = t;
	g.window_start = now;
	g.win_frames   = 0;
	std::memset(g.win_gpu, 0, sizeof(g.win_gpu));
	g.win_gpu_n[0] = g.win_gpu_n[1] = 0;
	g.win_cpu_ms = 0.0;
	std::memset(g.win_skips, 0, sizeof(g.win_skips));
	std::memset(g.win_runs, 0, sizeof(g.win_runs));
	std::memset(g.win_stat, 0, sizeof(g.win_stat));
	std::memset(g.win_rough, 0, sizeof(g.win_rough));
	std::memset(g.win_capture_by_role, 0, sizeof(g.win_capture_by_role));
	g.win_capture_refused = 0;
	g.win_lit_cand        = 0;
	g.win_debug           = 0;
	// (pt1e)
	{
		const double sh_vis = w[33] > 0.0 ? w[32] / w[33] * 100.0 : 0.0; // (compose: pixels changed of the sampled ones)
		char cs[48] = "";
		if (g.consumer != RNone) std::snprintf(cs, sizeof(cs), " (CS %016llx)", static_cast<unsigned long long>(g.consumer_cs));
		std::printf("[pt] 5 s: sombras: destino %s (de un frame anterior en %llu frames), compuestos %.1f %% px; consumidor de la mascara %s%s; rayos "
		            "solo donde la mascara del juego esta iluminada%s, %s, ocluyentes %s; escena: refit de mallas moviles cada %u frames\n",
		            g.mask_cache.index != UINT32_MAX ? "fijado" : "no visto", static_cast<unsigned long long>(g.win_mask_cached), sh_vis,
		            g.consumer == RResolveLighting ? "ResolveLighting" : g.consumer == RRenderLights ? "RenderLights" : "no visto (se compone al empezar RenderLights)",
		            cs,
		            (Cfg().channel >= 0 || g.sh_channel >= 0) ? "" : " (aun no: canal sin calibrar)", g.live.shadow_checker ? "damero" : "todos los texels",
		            g.live.shadow_dynamic ? "estaticos y moviles" : "solo estaticos (los personajes: cascadas del juego)", RefitEvery());
	}
	// (pt1e) budget control (ON windows of a fixed preset): over budget -> the trace scales down; under 60 % -> back up
	if (g.win_on_frames > 0 && Cfg().budget && !Cfg().cycle) {
		const float before = g.live.scale;
		if (total > g.live.budget_ms && g.live.scale > 0.5f)
			g.live.scale = std::max(0.5f, g.live.scale * std::max(0.7f, static_cast<float>(std::sqrt(g.live.budget_ms / std::max(total, 1e-3)))));
		else if (total < 0.6 * g.live.budget_ms && g.live.scale < 1.0f)
			g.live.scale = std::min(1.0f, g.live.scale * 1.15f);
		if (g.live.scale != before) {
			++g.scale_changes;
			g.s_history = false;
			std::printf("[pt] presupuesto: %.3f ms frente a %.1f ms (%s): escala de traza %.2f -> %.2f (sombras %.2f de la resolucion, relleno 1 "
			            "rayo por %ux%u)\n",
			            total, g.live.budget_ms, pname, before, g.live.scale, std::max(0.25f, g.live.shadow_scale * g.live.scale), FillBlock(), FillBlock());
		}
	}
	g.last_total     = total;
	g.win_on_frames  = g.win_off_frames = g.win_off_jobs = g.win_off_slots = 0;
	g.win_off_gpu    = 0.0;
	g.win_fill_gpu   = 0.0;
	g.win_fill_n     = 0;
	g.win_fill_runs  = g.win_fill_skips = 0;
	g.fill_why       = nullptr;
	g.win_mask_cached = 0;
	if (Cfg().ab && !Cfg().cycle) { // (pt1e) BB_PT_AB=1: the next window the other way
		g.live.on   = !g.live.on;
		g.s_history = g.r_history = false;
		std::printf("[pt] A/B: siguiente ventana %s\n", g.live.on ? "CON path tracing" : "SIN path tracing");
		std::fflush(stdout);
	}
	if (Cfg().cycle) {
		g.cycle_window = (g.cycle_window + 1) % 5;
		g.live         = MakeLive(g.cycle_window < 4 ? g.cycle_window : 0);
		g.live.on      = g.cycle_window < 4;
		g.s_history = g.r_history = false;
		std::printf("[pt] ciclo: siguiente ventana %s\n", g.cycle_window < 4 ? kPresets[g.cycle_window].name_es : "apagado");
		std::fflush(stdout);
	}
}

bool OnGpu() { return NativeRender::Cut::OnGpuThread(); }

} // namespace

// ---- public ---------------------------------------------------------------------------------------------------------

bool Requested() { return Cfg().requested; }

bool On() { return Cfg().requested && g.live_on && g.live.on; }

void ToggleLive() {
	if (Requested()) g_live_toggle.store(true, std::memory_order_release);
}

const char* PresetName() { return kPresets[std::clamp(Cfg().quality, 0, 3)].name_es; }

uint32_t CaptureVertices() {
	if (!Requested()) return 0;
	const int q = Cfg().cycle ? 3 : Cfg().quality;
	return kPresets[q].capture_mverts << 20u;
}
uint64_t MeshBudgetBytes() {
	if (!Requested()) return 0;
	static const uint64_t v = [] {
		const int q  = Cfg().cycle ? 3 : Cfg().quality;
		const int mb = EnvI("BB_PT_MESH_MB", static_cast<int>(kPresets[q].mesh_budget_mb), 32, 8192);
		return uint64_t {static_cast<uint32_t>(mb)} << 20u;
	}();
	return v;
}
uint32_t KeepFrames() {
	if (!Requested()) return 0;
	return kPresets[Cfg().cycle ? 3 : Cfg().quality].keep_frames;
}
uint32_t RefitEvery() {
	if (!Requested()) return 0;
	return std::max(1u, g.live.refit_every);
}

void ScopeBegin(uint32_t pass_id, int32_t queue) {
	if (!Requested()) return;
	const Role r = RoleOf(pass_id);
	if (r == RNone) return;
	++g.open_count[r];
	// (pt1e) off (F9 / an A/B or cycle window): no job, no target notes, no sun search in the cascades' draws; the scope
	// counts stay right for when it is on
	const bool on = g.live_on && g.live.on;
	if (!on) return;
	if (r == RCascade) g_cascade_open.store(true, std::memory_order_relaxed);
	if (r == RGBuffer || r == RResolveLighting) NativeRender::g_pt_targets.store(true, std::memory_order_relaxed);
	if (r == RResolveLighting) g.lit_n = 0;
	if (r == RPostProcessing && Cfg().debug == 1) g.jobs.push_back({JDebug, queue, g.frame}); // (pt1c)
	// Shadows: before the game's light passes read the sun-shadow image (RenderLights is the first DCB scope after
	// ResolveSunShadows and the GBuffer, frame map order-real). Reflections: before the IBL pass reads the SSR result.
	// (pt1e) Before the first pass seen reading the mask (log: the PS5 Pro path's RenderLights dispatches read no RG8 image
	// of the render size, ResolveLighting's CS c0dfd916e841a78e reads one); RenderLights until a consumer is seen.
	const Role shadow_at = g.consumer == RResolveLighting ? RResolveLighting : RRenderLights;
	if (r == shadow_at) {
		// (pt1d) before the light passes read the sun mask; with no mask identified yet the trace still runs (debug view,
		// calibration of the next candidate)
		if (g.resolve_sun_frame == g.frame) g.jobs.push_back({JShadows, queue, g.frame});
		else if (Cfg().shadows) ++g.win_skips[0][WOrder];
	}
	if (r == RRenderIBL && Cfg().refl_legacy) {
		if (g.ssr_frame == g.frame) g.jobs.push_back({JReflections, queue, g.frame});
		else if (Cfg().reflections) ++g.win_skips[1][WOrder];
	}
}

void ScopeEnd(uint32_t pass_id, int32_t queue) {
	if (!Requested()) return;
	const Role r = RoleOf(pass_id);
	if (r == RNone) return;
	g.open_count[r] = std::max(0, g.open_count[r] - 1);
	if (r == RCascade && g.open_count[RCascade] == 0) g_cascade_open.store(false, std::memory_order_relaxed);
	if ((r == RGBuffer || r == RResolveLighting) && g.open_count[RGBuffer] == 0 && g.open_count[RResolveLighting] == 0)
		NativeRender::g_pt_targets.store(false, std::memory_order_relaxed);
	if (r == RGBuffer && g.built_frame != g.frame && ((g.live_on && g.live.on) || Cfg().capture_off)) g.jobs.push_back({JBuild, queue, g.frame});
	if (r == RResolveSunShadows) g.resolve_sun_frame = g.frame;
	if (r == RRenderSSR) g.ssr_frame = g.frame;
}

bool WorkPending() {
	if (!Requested() || g.jobs.empty()) return false;
	return g.jobs.front().queue == NativeRender::Cut::Queue();
}

void RunWork(CommandBuffer& buffer) {
	while (!g.jobs.empty() && g.jobs.front().queue == NativeRender::Cut::Queue()) {
		const PendingJob job = g.jobs.front();
		g.jobs.pop_front();
		if (buffer.IsInvalid()) continue;
		if (job.frame != g.frame) {
			++g.win_skips[job.job == JReflections ? 1 : 0][WLate];
			continue;
		}
		if ((!g.live_on || !g.live.on) && !(job.job == JBuild && Cfg().capture_off)) { // (pt1e: F9 went off after it was queued)
			++g.win_off_jobs;
			continue;
		}
		auto& graphics = buffer.GetContext().GetGraphics();
		if (!HwScene::Active()) {
			if (!g.said_on) {
				g.said_on = true;
				std::printf("[pt] BB_PATHTRACING=1 pero la escena por hardware no esta activa (el dispositivo no tiene ray query o fallo al "
				            "crearla; ver lineas HWRT): sin path tracing\n");
				std::fflush(stdout);
			}
			g.jobs.clear();
			return;
		}
		if (!Init(graphics)) {
			g.jobs.clear();
			return;
		}
		// (pt1d) Our compute work pushes its own constants and descriptors. The draw path's fast repeat of a draw keeps the
		// push constants it pushed before (CommandBuffer::GraphicsGeneration): after our work it must push them again.
		// pt1c's shadows at the end of RenderLights, right before ResolveLighting's draw, are the suspect of the whole
		// screen turning red (no write to a game image was logged).
		struct Invalidate {
			CommandBuffer& b;
			~Invalidate() { b.InvalidateGraphicsState(); }
		} invalidate {buffer};
		switch (job.job) {
			case JBuild: RunBuild(buffer); break;
			case JShadows: RunShadows(buffer); break;
			case JReflections: RunReflections(buffer); break;
			case JDebug: RunDebug(buffer); break;
		}
	}
}

void Flip() {
	if (!Requested()) return;
	static bool said = false;
	if (!said) {
		said = true;
		std::printf("[pt] BB_PATHTRACING=1: preset %s%s%s; sombras %s, reflejos %s; GI no implementada (pt1); F9 = encender/apagar; captura en %s\n",
		            PresetName(), Cfg().cycle ? " (BB_PT_CYCLE: rota los 4 presets y apagado cada 5 s)" : "",
		            Cfg().ab ? " (BB_PT_AB: 5 s con y 5 s sin)" : "", Cfg().shadows ? "si" : "no",
		            !Cfg().reflections ? "no" : Cfg().refl_legacy ? "si (antiguos, en la imagen SSR del juego: BB_PT_REFL_LEGACY=1)"
		                                                          : "relleno de Reflejos RT (solo donde Reflejos RT refleja y su rayo de pantalla falla)",
		            Env("BB_PT_CAPTURE_SCOPES") != nullptr ? Env("BB_PT_CAPTURE_SCOPES") : "GBuffer,VirtualGeo");
		if (!NativeRender::On())
			std::printf("[pt] AVISO: el mapa del frame (KYTY_NATIVE_RENDER) no esta activo: sin ambitos del juego no hay path tracing\n");
		// (DLSS Ray Reconstruction: our passes write into the game's own intermediate images, not into a noisy colour the
		// upscaler would denoise, so RR has nowhere to plug in; the DLL's presence is only reported.)
		// Detected at run time: the DLL next to the exe and Streamline (the CI release has it; the local build does not).
		FILE*      rr  = std::fopen("nvngx_dlssd.dll", "rb");
		const bool dll = rr != nullptr;
		if (rr != nullptr) std::fclose(rr);
		const bool sl = FrameGen::StreamlineLoaded();
		std::printf("[pt] DLSS Ray Reconstruction: nvngx_dlssd.dll %s, Streamline %s -> RR %s; denoiser ACTIVO: propio (temporal con "
		            "reproyeccion de camara + espacial con bordes). RR no se conecta en pt1: estas pasadas escriben en imagenes intermedias "
		            "del juego (mascara del sol, SSR) antes de su iluminacion, y RR necesita el color final ruidoso en lugar del reescalador\n",
		            dll ? "presente" : "no encontrada", sl ? "cargado" : "no cargado",
		            dll && sl ? "disponible pero no aplicable" : "no disponible");
		std::fflush(stdout);
		g.live = MakeLive(Cfg().cycle ? 0 : Cfg().quality);
	}
	if (g_live_toggle.exchange(false, std::memory_order_acq_rel)) {
		g.live_on = !g.live_on;
		g.s_history = g.r_history = false;
		std::printf("[pt] F9: path tracing %s\n", g.live_on ? "ENCENDIDO" : "APAGADO");
		std::fflush(stdout);
	}
	if (g.live_on && g.live.on) ++g.win_on_frames; // (pt1e: this frame's state, ON / OFF windows apart)
	else ++g.win_off_frames;
	SettleSun();
	if (g.ready) {
		ReadStats();
		Calibrate();
		CollectRetired(*g.graphics);
	}
	++g.win_frames;
	if (g.ready) Log();
	g.jobs.clear(); // (a job whose packet never came this frame)
	++g.frame;
}

bool CaptureHere() {
	if (!Requested()) return true;
	// (pt1b) Off (F9 / a cycle window) costs nothing unless BB_PT_CAPTURE_WHEN_OFF=1: no capture, no build (the BLAS cache
	// stays; on again it is used at once).
	if ((!g.live_on || !g.live.on) && !Cfg().capture_off) return false;
	for (int r = RGBuffer; r <= RCapture; ++r)
		if (g.open_count[r] > 0 && CaptureRole(static_cast<Role>(r))) {
			++g.win_capture_by_role[r];
			return true;
		}
	++g.win_capture_refused;
	return false;
}

bool WantDispatchHook() {
	if (!Requested() || !OnGpu()) return false;
	if (!g.live_on || !g.live.on) return false; // (pt1b: off, the dispatches keep the table path)
	if (!Cfg().shadows && !Cfg().refl_legacy) return false; // (pt1e: nothing to find)
	// (pt1e) ResolveLighting only until its consumer of the mask and its lit colour are known (it held ~45 dispatches a
	// frame on the slow path); RenderSSR only for the legacy reflections.
	const bool lit_scope = g.open_count[RResolveLighting] > 0 && (g.consumer == RNone || Cfg().refl_legacy);
	return g.open_count[RResolveSunShadows] > 0 || lit_scope || (Cfg().refl_legacy && g.open_count[RRenderSSR] > 0); // (pt1d: not RenderLights)
}

void AfterDispatch(TextureCache& cache, std::span<const TextureBinding> images, uint64_t written_mask, uint64_t cs_hash) {
	if (!WantDispatchHook()) return;
	const bool sun = g.open_count[RResolveSunShadows] > 0, lit = g.open_count[RResolveLighting] > 0, ssr = g.open_count[RRenderSSR] > 0;
	const bool lights = g.open_count[RRenderLights] > 0;
	// The render size: the GBuffer's depth of this frame.
	// (pt1c) The last known one: ResolveSunShadows runs on the ACB before this frame's GBuffer draws are translated, so
	// pt1b saw "no render size" for exactly the dispatch that writes the sun mask (log 065150: CS 2871939ca9c6753e).
	if (g.depth.index != UINT32_MAX)
		if (Image* d = cache.TryGetImage(ImageId(g.depth.index, g.depth.generation)); d != nullptr && d->backing.image != nullptr)
			g.render_last = {std::max(1u, d->backing.extent.width >> g.depth.level), std::max(1u, d->backing.extent.height >> g.depth.level)};
	const vk::Extent2D render = g.render_last;
	// Does this CS read a layered depth image (>= 4 layers: the sun cascades)?
	bool reads_cascades = false;
	for (size_t i = 0; i < images.size() && i < 64; ++i) {
		if (!images[i].image_id || ((written_mask >> i) & 1u) != 0) continue;
		if (Image* im = cache.TryGetImage(images[i].image_id); im != nullptr && im->backing.image != nullptr && IsDepthFormat(im->backing.format) &&
		                                                       im->backing.layers >= 4)
			reads_cascades = true;
	}
	// (pt1e) The mask's consumer: a dispatch outside ResolveSunShadows reading the mask image (by its cache slot).
	if (g.consumer == RNone && !sun && g.mask_cache.index != UINT32_MAX)
		for (size_t i = 0; i < images.size() && i < 64; ++i) {
			if (!images[i].image_id || ((written_mask >> i) & 1u) != 0) continue;
			if (images[i].image_id.index != g.mask_cache.index || images[i].image_id.generation != g.mask_cache.generation) continue;
			g.consumer         = lit ? RResolveLighting : lights ? RRenderLights : RNone;
			g.consumer_cs      = cs_hash;
			g.consumer_binding = static_cast<uint32_t>(i);
			if (g.consumer != RNone) {
				std::printf("[pt] sombras: la mascara del sol la lee el CS %016llx (binding %zu) en %s: la composicion va al principio de "
				            "ese ambito\n", static_cast<unsigned long long>(cs_hash), i, lit ? "ResolveLighting" : "RenderLights");
				std::fflush(stdout);
			}
			break;
		}
	// (pt1b) Every CS of these scopes logged once with all its images (format, size, layers, read / written): the data the
	// target choice rests on.
	const bool log_cs = std::find(g.cs_logged.begin(), g.cs_logged.end(), cs_hash) == g.cs_logged.end() && g.cs_logged.size() < 96;
	std::string line;
	for (size_t i = 0; i < images.size() && i < 64; ++i) {
		const auto& b = images[i];
		if (!b.image_id) continue;
		Image* image = cache.TryGetImage(b.image_id);
		if (image == nullptr || image->backing.image == nullptr) continue;
		Seen s;
		s.index      = b.image_id.index;
		s.generation = b.image_id.generation;
		s.level      = b.desc.view_info.base_level;
		s.layer      = b.desc.view_info.base_layer;
		s.frame      = g.frame;
		s.format     = image->backing.format;
		s.extent     = {std::max(1u, image->backing.extent.width >> s.level), std::max(1u, image->backing.extent.height >> s.level)};
		const bool writes = ((written_mask >> i) & 1u) != 0;
		if (log_cs) {
			char text[96];
			std::snprintf(text, sizeof(text), " %zu:%s%s %ux%ux%u", i, writes ? "W " : "", FormatName(s.format), s.extent.width, s.extent.height,
			              image->backing.layers);
			line += text;
		}
		if (!writes || IsDepthFormat(s.format)) continue;
		// Sun-shadow candidates: written at the render size or half of it, in ResolveSunShadows or RenderLights.
		const auto near_size = [](uint32_t a, uint32_t r) { return a + 1 >= r && a <= r + 1; };
		const bool screen = render.width != 0 && ((near_size(s.extent.width, render.width) && near_size(s.extent.height, render.height)) ||
		                                          (near_size(s.extent.width, render.width / 2) && near_size(s.extent.height, render.height / 2)));
		// (pt1d) [G] the engine resolves the sun into kRTT_SunShadowResolve (id 0x62, render size, FUN_00917c30) inside
		// "LightRendering::ResolveSunShadowJob" (FUN_00a0a800 -> FUN_009dffe0, scope ResolveSunShadows): only an image written
		// there, unorm, by a CS that reads the cascades is a candidate (pt1b / pt1c tried RenderLights' lighting images:
		// 32 of them filled the list and the mask never got a place).
		const bool unorm_fmt = s.format == vk::Format::eR8Unorm || s.format == vk::Format::eR8G8Unorm || s.format == vk::Format::eR8G8B8A8Unorm ||
		                       s.format == vk::Format::eR16Unorm || s.format == vk::Format::eR16G16Unorm;
		if (sun && screen && unorm_fmt && reads_cascades) {
			auto it = std::find_if(g.cands.begin(), g.cands.end(), [&](const State::Cand& c) { return c.cs == cs_hash && c.binding == i; });
			const bool unorm = s.format == vk::Format::eR8Unorm || s.format == vk::Format::eR8G8Unorm || s.format == vk::Format::eR8G8B8A8Unorm ||
			                   s.format == vk::Format::eR16Unorm || s.format == vk::Format::eR16G16Unorm;
			if (it == g.cands.end() && g.cands.size() < 32) {
				State::Cand c;
				c.cs = cs_hash, c.binding = static_cast<uint32_t>(i), c.scope = sun ? RResolveSunShadows : RRenderLights;
				c.structural = sun && unorm && reads_cascades;
				// (pt1c) the structural ones first: the screen-space sun mask is a unorm image the cascades are resolved into
				if (c.structural) {
					g.cands.insert(g.cands.begin(), c);
					it = g.cands.begin();
					if (g.cand_cur >= 0 && g.cand_locked < 0) g.cand_cur = -1; // (try it next)
					if (g.cand_locked >= 0) ++g.cand_locked;
				} else {
					g.cands.push_back(c);
					it = g.cands.end() - 1;
				}
				std::printf("[pt] candidata a sombra del sol: %s CS %016llx binding %zu, %s %ux%u (render %ux%u)%s\n",
				            sun ? "ResolveSunShadows" : "RenderLights", static_cast<unsigned long long>(cs_hash), i, FormatName(s.format),
				            s.extent.width, s.extent.height, render.width, render.height,
				            c.structural ? " - ESTRUCTURAL (unorm, lee las cascadas): se prueba primero" : "");
				std::fflush(stdout);
			}
			if (it != g.cands.end()) it->last = s;
		}
		if (ssr && Cfg().refl_legacy && s.format == vk::Format::eR16G16B16A16Sfloat) {
			g.ssr = s;
			if (!g.said_ssr) {
				g.said_ssr = true;
				std::printf("[pt] RenderSSR escribe RGBA16F %ux%u (CS %016llx, binding %zu): destino de los reflejos (render %ux%u)\n", s.extent.width,
				            s.extent.height, static_cast<unsigned long long>(cs_hash), i, render.width, render.height);
				std::fflush(stdout);
			}
		}
		if (lit && IsFloatColour(s.format)) {
			g.lit_cand[g.lit_n % g.lit_cand.size()] = s; // (a ring: the last ones written win)
			++g.lit_n;
		}
	}
	if (log_cs) {
		g.cs_logged.push_back(cs_hash);
		std::printf("[pt] CS %016llx en %s:%s\n", static_cast<unsigned long long>(cs_hash),
		            sun ? "ResolveSunShadows" : lights ? "RenderLights" : lit ? "ResolveLighting" : ssr ? "RenderSSR" : "?", line.c_str());
		std::fflush(stdout);
	}
}

void NoteTaaInput(TextureCache& cache, std::span<const TextureBinding> images, uint64_t written_mask, uint64_t cs_hash) {
	if (!Requested() || !OnGpu() || g.render_last.width == 0) return;
	for (size_t i = 0; i < images.size() && i < 64; ++i) {
		if (!images[i].image_id || ((written_mask >> i) & 1u) != 0) continue;
		Image* im = cache.TryGetImage(images[i].image_id);
		if (im == nullptr || im->backing.image == nullptr || !IsFloatColour(im->backing.format)) continue;
		Seen s;
		s.index = images[i].image_id.index, s.generation = images[i].image_id.generation;
		s.level = images[i].desc.view_info.base_level, s.layer = images[i].desc.view_info.base_layer;
		s.frame  = g.frame;
		s.format = im->backing.format;
		s.extent = {std::max(1u, im->backing.extent.width >> s.level), std::max(1u, im->backing.extent.height >> s.level)};
		if (s.extent != g.render_last) continue;
		const bool changed = g.taa_in.index != s.index || g.taa_in.generation != s.generation;
		g.taa_in = s;
		if (!g.said_taa || changed) {
			g.said_taa = true;
			std::printf("[pt] color HDR iluminado = lo que lee %s (CS %016llx binding %zu): %s %ux%u, imagen %u: fuente de los reflejos y "
			            "destino de la vista de depuracion\n",
			            cs_hash == 0 ? "el paquete de superresolucion PS5 Pro (en lugar del TAA)" : "el TAA del juego",
			            static_cast<unsigned long long>(cs_hash), i, FormatName(s.format), s.extent.width, s.extent.height, s.index);
			std::fflush(stdout);
		}
		return;
	}
}

void NoteTarget(uint32_t index, uint32_t generation, uint32_t level, uint32_t layer, bool depth) {
	if (!Requested() || !OnGpu()) return;
	Seen s;
	s.index = index, s.generation = generation, s.level = level, s.layer = layer, s.frame = g.frame;
	if (g.open_count[RGBuffer] > 0) {
		if (depth) {
			g.depth = s;
			if (!g.said_depth) {
				g.said_depth = true;
				std::printf("[pt] profundidad del GBuffer: imagen %u, nivel %u, capa %u\n", index, level, layer);
				std::fflush(stdout);
			}
		} else if (layer < 4) {
			g.gb_layer[layer] = s; // (the array's layers: the format is checked when bound)
		}
	}
	if (g.open_count[RResolveLighting] > 0 && !depth) {
		g.lit_cand[g.lit_n % g.lit_cand.size()] = s;
		++g.lit_n;
	}
}

void OnDraw(uint64_t /*vs_hash*/, std::span<const uint32_t> flattened_srt) {
	if (!g_cascade_open.load(std::memory_order_relaxed) || !OnGpu()) return;
	if (g.sun_axes_frame != g.frame) {
		g.sun_axes.clear();
		g.sun_axes_frame = g.frame;
	}
	if (g.sun_axes.size() >= 32) return;
	++g.sun_draws;
	std::array<float, 3> axis {};
	if (FindOrtho(flattened_srt, axis)) g.sun_axes.push_back(axis);
}

bool ReflFillReady() {
	const char* why = !OnGpu() ? "otro hilo" : FillBlocked();
	if (why == nullptr) return true;
	if (Requested() && g.ready) { // (the 5 s line says why the fill did not run)
		++g.win_fill_skips;
		g.fill_why = why;
	}
	return false;
}

bool RecordReflFill(vk::CommandBuffer command, const ReflFillIo& io) {
	if (command == nullptr || !Requested()) return false;
	if (const char* why = FillBlocked(); why != nullptr || !OnGpu()) {
		++g.win_fill_skips;
		g.fill_why = why != nullptr ? why : "otro hilo";
		return false;
	}
	if (io.rad == nullptr || io.aux == nullptr || io.gb_cur == nullptr || io.gb_prev == nullptr || io.prev_screen == nullptr ||
	    io.rad_out == nullptr || io.depth == nullptr || io.color == nullptr || io.trace.width == 0 || io.render.width == 0) {
		++g.win_fill_skips;
		g.fill_why = "imagenes de Reflejos RT";
		return false;
	}
	const auto t0 = std::chrono::steady_clock::now();
	const vk::CommandBuffer cmd = command;
	BeginSlot(cmd);
	for (OwnImage* i: {&g.dummy_u, &g.dummy_f}) ToGeneral(cmd, *i);
	if (!g.cache_cleared) {
		g.cache_cleared = true;
		Barrier(cmd, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
		        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
		cmd.fillBuffer(vk::Buffer(g.cache), 0, CacheBytes, 0);
	}
	// (what rt-reflections' trace wrote, the TLAS build, the cache clear -> our reads)
	Barrier(cmd, vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
	        vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eAccelerationStructureWriteKHR | vk::AccessFlagBits2::eTransferWrite,
	        vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eAccelerationStructureReadKHR);
	FrameUbo u {};
	FillFrameUbo(u, io, io.history);
	UploadUbo(cmd, u);
	Bound depth;
	depth.view    = io.depth;
	depth.extent  = io.render;
	PassImages pi = BaseImages(depth);
	const auto gen = vk::ImageLayout::eGeneral;
	pi.info[1]  = {g.point, io.depth, io.depth_layout};
	pi.info[3]  = {g.linear, io.color, io.color_layout};
	pi.info[19] = {g.point, io.rad, gen};
	pi.info[20] = {g.point, io.aux, gen};
	pi.info[21] = {g.point, io.gb_cur, gen};
	pi.info[22] = {g.point, io.gb_prev, gen};
	pi.info[23] = {g.linear, io.prev_screen, gen};
	pi.info[24] = {nullptr, io.rad_out, gen};
	const vk::DescriptorBufferInfo exposure {io.exposure, io.exposure_base, io.exposure_range};
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipeline);
	Stamp(cmd, 8);
	const uint32_t stride = 16, block = FillBlock();
	Dispatch(cmd, pi, 9, vk::Extent2D {(io.render.width + stride - 1) / stride, (io.render.height + stride - 1) / stride},
	         io.exposure != nullptr ? &exposure : nullptr);
	Dispatch(cmd, pi, 10, vk::Extent2D {(io.trace.width + block - 1) / block, (io.trace.height + block - 1) / block},
	         io.exposure != nullptr ? &exposure : nullptr);
	Dispatch(cmd, pi, 11, vk::Extent2D {256, CacheSlots / 256}, io.exposure != nullptr ? &exposure : nullptr);
	Stamp(cmd, 9);
	g.fill_frame = g.frame;
	++g.win_fill_runs;
	g.win_cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	return true;
}

void NoteReflectionsMs(double ms, bool with_fill) {
	if (!Requested() || !(ms >= 0.0) || ms > 1000.0) return;
	g.ab_ms[with_fill ? 1 : 0] += ms;
	++g.ab_n[with_fill ? 1 : 0];
}

void NoteViewToClip(const float* m16) {
	if (!Requested() || m16 == nullptr) return;
	for (int i = 0; i < 16; ++i)
		if (!std::isfinite(m16[i])) return;
	std::memcpy(g.view_to_clip.data(), m16, 64);
	g.have_proj = std::fabs(m16[0]) > 1e-6f && std::fabs(m16[5]) > 1e-6f;
}

} // namespace Libs::Graphics::PathTrace

// (native-render.h) The render targets a draw writes, while BB_PATHTRACING has a GBuffer / ResolveLighting scope open.
namespace NativeRender {
void NotePtTarget(uint32_t index, uint32_t generation, uint32_t level, uint32_t layer, bool depth) {
	Libs::Graphics::PathTrace::NoteTarget(index, generation, level, layer, depth);
}
} // namespace NativeRender
