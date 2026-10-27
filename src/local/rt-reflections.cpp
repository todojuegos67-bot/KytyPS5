// KYTY_RT_REFLECTIONS=0|1|2|3|4|5 ("RT reflections": off / low / medium / high / ultra / extreme; rt-reflections.comp): screen-space ray-traced
// reflections with denoising for Demon's Souls' PS5 Pro path, recorded in FrameGen::Upscale right before the upscaler
// (DLSS / XeSS / FSR), which then reads our composited copy of the game's color. Only there: it is the one point where
// the game hands over its lit HDR color before anti-aliasing together with depth, motion vectors and the projection
// (without KYTY_UPSCALER the game's own TAA runs and nothing here does).
//
// Per frame (all compute, one pipeline, push descriptors):
//   hi-z:      the closest-distance pyramid of the depth (mip 0 at the render size, then 2x2 min down to 1x1)
//   trace:     one GGX VNDF-sampled ray per trace pixel (low / medium / high: half resolution, ultra: full; low and
//              ultra: checkerboard, half the trace pixels a frame), hierarchical-Z march (LevelCfg below)
//   temporal:  reprojection by motion vector + reflection parallax, disocclusion rejection, neighbourhood clamp
//   a-trous:   2 / 3 / 4 edge-aware passes (distance, normal, roughness, luminance variance)
//   composite: bilateral upsample, Fresnel x mask x confidence convex mix over the game's color
// Surface data: the packed R32_UINT G-buffer copy KYTY_SSR's hooks take before the deferred lighting (frame-gen.cpp),
// decoded after a calibration against depth normals (normal: statistically identified; roughness: the normal word's free
// byte, a guess, so weighed by the up-facing / flat / dark heuristic unless set by KYTY_RT_REFLECTIONS_ROUGH); without
// it, depth normals and the heuristic. The game's exposure (its upscaler path's slot 5 buffer) is read so the filters
// work on display-referred values.
//
// Material roughness (refl6): KYTY_RT_REFLECTIONS_ROUGH (default 2:0:8, L2.b0), KYTY_RT_REFLECTIONS_GLOSS (default
// 1:16:8g, L1.b2 inverted; 0 off), KYTY_RT_REFLECTIONS_METAL_FIELD (refl9: default found by the survey), KYTY_RT_REFLECTIONS_RANGE=lo:hi (0.1:0.4;
// the older KYTY_RT_REFLECTIONS_WET=lo:hi is read the same way), KYTY_RT_REFLECTIONS_MATERIAL (1.0: its scale),
// KYTY_RT_REFLECTIONS_DEPTH=<n> (the lighting's n-th depth binding as the pre-water depth), KYTY_RT_REFLECTIONS_GUESS
// (0: the puddle guess, only without a material field).
// KYTY_RT_REFLECTIONS_STRENGTH (per level 1.3 / 1.6 / 1.8 / 2.0, 0..3): the reflection's share scale.
// KYTY_RT_REFLECTIONS_MIN (0.4, 0..0.8): the minimum reflectance of water. KYTY_RT_REFLECTIONS_NORMAL
// = <layer>:<layout 0..12>[w][n] and KYTY_RT_REFLECTIONS_ROUGH = <layer>:<shift>:<bits>[g] override the calibration
// (layouts: the shader's DecodeNormal), KYTY_RT_REFLECTIONS_FLIPY=1|0 forces the screen y convention (1: the render's row
// 0 is NDC y +1, the default). Every 10 s one line: the share of pixels the reflections change, the mean weight, the
// hit rate, the G-buffer copy's frames, the GPU time (timestamps, always on; KYTY_RT_REFLECTIONS_TIMING is not needed).
//
// Reflectivity (rt-reflections.comp SurfaceAt), refl6: per pixel from the material's roughness, on any surface facing any
// way (floors, walls, metal): the rougher of the roughness field (default L2.b0, KYTY_RT_REFLECTIONS_ROUGH) and 1 - the
// gloss field (default L1.b2, KYTY_RT_REFLECTIONS_GLOSS=<layer>:<shift>:<bits>g, 0 off). A mirror at full strength below
// 0.1, weaker and blurrier (the GGX lobe widens) up to 0.4, nothing from there (KYTY_RT_REFLECTIONS_RANGE=0.1:0.4). The
// share is a Schlick Fresnel with f0 0.04 (dielectric) or from KYTY_RT_REFLECTIONS_METAL (a metal field, same syntax;
// default none): smooth dry stone seen from above barely changes, at grazing angles it mirrors. Water (the final depth
// nearer than the depth the deferred lighting read, which frame-gen.cpp copies with the G-buffer) stays up-facing only and
// keeps the minimum reflectance. Moving objects (characters: motion vector vs the camera's reprojection;
// KYTY_RT_REFLECTIONS_MOVING=0 off; switched off by itself for 10 s when it flags over 35 % of the screen) and the sky
// never get one. The periodic line gives the share of pixels per roughness bucket, to tune the range from a log.
//
// Why the first version showed nothing (log of 2026-10-24): the G-buffer copy was refused nearly every frame (the
// G-buffer draws take the table path, which did not refresh the G-buffer pass's flip, frame-gen.cpp OnSsrDraw); the
// depth reconstruction mirrored the screen vertically against the game's (GNM) viewport, so the depth normals
// disagreed with the G-buffer's (calibration 0.654) and with world up (floors read as ceilings: the wet-surface mask was
// zero on floors); the mask needed flat AND plain AND up-facing at once (textured wet stone failed "plain"); and the
// composite weight was a dry dielectric's Fresnel (~0.02 seen from above).
//
// (refl8) Guided by the game's own reflections (the default; KYTY_RT_REFLECTIONS_GUIDE=0: refl6's material mask): the
// game's SSR result, a half-resolution RGBA16F input of its deferred lighting (CS 9fb5cc274b5e4f82, which also reads the
// G-buffer, both depths and the 256x256 RG16F BRDF LUT), is copied after that dispatch (frame-gen.cpp
// RtReflGuideAfterDispatch) and decides which surfaces reflect and how much: where its confidence (a) is high and our
// sharper ray hits, the game's SSR color (rgb) is replaced by ours at the game's share (a x Fresnel x
// KYTY_RT_REFLECTIONS_GUIDE_STRENGTH, 1.0); where our ray misses or leaves the screen, the game's pixel is unchanged. No
// material guesses add a reflection; the roughness field only narrows the sharpening to mirror-like surfaces. Water (drawn
// after the lighting, so not in the game's SSR) keeps its depth-detected mirror. KYTY_RT_REFLECTIONS_GUIDE_RANGE=lo:hi
// (0.1:0.6, the confidence from which / to which the game reflects), KYTY_RT_REFLECTIONS_GUIDE_PREMUL=1 (its rgb is
// premultiplied by a), KYTY_RT_REFLECTIONS_GUIDE_CS=<hash>, KYTY_RT_REFLECTIONS_GUIDE_SLOT=<n> (default: the slot found
// before, else the first RGBA16F input at half the render (depth) size, else at half the written image's size). Logged: the lighting's images once, the dispatch that writes the guide (the
// game's SSR pass) once, and every 10 s "RT reflections: guide:" with the share of pixels where the game reflects and
// where we sharpened.
//
// Why refl8 showed nothing (logs of 2026-10-25): the guide slot was found once, on a 2560x1440 lighting run, and never
// again: in play the lighting renders 1280x720 and writes a 640x720 image, so "half the written image" matched no input
// (640x360) and the guide was copied in 0 frames; and while it was missing the guided mode masked everything but water
// (0 % in Demon's Souls). Fixed: the slot by the render size and kept, the material mask while there is no guide, and the
// guided composite also mixes our hit in up to the wet minimum reflectance on mirror-like up-facing pixels the game's SSR
// is confident on (the game's own share alone, a dielectric's Fresnel, changed the pixel by ~1-4 %).
//
// (refl9) Metals and glossy surfaces, any orientation (guided mode). Metal (armour, swords, shields): the G-buffer's metal
// field, found at run time by the metal survey (SurveyDecide: the near-binary byte set on characters far more often than
// on the static world; logged with its candidates; KYTY_RT_REFLECTIONS_METAL_FIELD=<layer>:<shift>:<bits>[g] sets it, 0
// none), our hit through a sharper lobe than the material's (ExtraLobe, then the a-trous blur by roughness) mixed in up to
// KYTY_RT_REFLECTIONS_METAL (0.35) fading out by a roughness of 0.65, tinted by the albedo's hue
// (KYTY_RT_REFLECTIONS_METAL_TINT 0.7), on characters too. Glossy dielectrics (polished / wet stone, marble, glass,
// polished wood): up to KYTY_RT_REFLECTIONS_GLOSSY (0.25) x a soft Fresnel, floors from a roughness of 0.30 down, walls
// only from 0.12 down (at 70 %). Both at 65 % where the game's own SSR is not confident, full where it is; a miss leaves the game's
// pixel (its IBL); never over its highlights, never darker than 25 % of it in all. Every 10 s "RT reflections: metal /
// glossy:" with the metal and glossy shares, where our hit was added and the mean added weight.
// (refl9b, user feedback on combo20: tiny puddles and speckled wet mud sparkled; large puddles wanted stronger and
// sharper) Floors and water pass an area test (Coverage: 16 taps of the trace's mask within KYTY_RT_REFLECTIONS_MIN_AREA
// render pixels at 720p, 10; 0 off): a patch covering under ~35-70 % of that disc fades out, one covering over ~75-95 %
// is a pool: its weight x (1 + KYTY_RT_REFLECTIONS_POOL_BOOST, 0.6) and its own full-resolution mirror ray up to a
// roughness of 0.25 (no filtered blur mixed in). Fireflies: the floor's reflection at most 3x (pools 6x) the pixel's
// luminance, the temporal pass clamps a sample to its neighbourhood (mean + 2.5 sigma) and takes in a sample that differs
// 2x from its history at half the rate.
// (refl10, precision and depth; the user's Ultra: KYTY_RT_REFLECTIONS=4) What limited it, from the code and the logs:
// mirrors at ultra came from the temporal history (a fifth of a fresh sample a frame through a bilinear fetch, half the
// pixels traced per frame), and the composite's own sharp mirror ray (half-resolution levels only) never ran there,
// pools' included: soft, lagging mirrors; the history of untraced (checkerboard) and missed pixels was reprojected by the
// surface's motion, not the reflected hit's (virtual image), so the reflection slid with the floor (smear / ghost
// trails); every pixel's hit was the entry of a flat screen-facing box (stepped edges); the G-buffer normal is
// octahedral 8:8 (~0.45 degree steps: banding across flat puddles, ~0.4 m at 25 m); rays hit 36-58 % of the time.
// Changes: KYTY_RT_REFLECTIONS_FRESH_MIRROR (ultra / extreme: this frame's mirror ray, the trace's or the composite's
// own), KYTY_RT_REFLECTIONS_VIRTUAL (every level: the neighbours' hit distance for the virtual image),
// KYTY_RT_REFLECTIONS_BICUBIC (ultra / extreme: Catmull-Rom history below a roughness of 0.3),
// KYTY_RT_REFLECTIONS_REFINE (ultra / extreme: bisection on the bilinear surface), KYTY_RT_REFLECTIONS_NORMAL_SNAP
// (degrees, 2; 0 off: on floors / ramps where the 8-bit G-buffer normal agrees with the depth's plane that closely, the
// plane's exact normal),
// KYTY_RT_REFLECTIONS_RPP (rays per glossy pixel, 1..4), KYTY_RT_REFLECTIONS_ITERS (8..256) / _CAP (x the render
// height, 0.1..4) override a level's march; level 5 "extreme": every pixel every frame, 128 iterations, rays up to 2.5x
// the render height, 2 rays per glossy pixel. A 10 s line "RT reflections: rays:" says why rays miss.
#include "local/local-pass-cache.h"
#include "local/rt-reflections.h"
#include "local/rt-flicker.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/rt_reflections_spv.h"
#include "gpu_tiler_shaders/rt_reflections_hw_spv.h"
#include "gpu_tiler_shaders/rt_reflections_hwm_spv.h"
#include "local/hwrt-scene.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace Libs::Graphics::FrameGen {
namespace {

constexpr uint32_t Bindings      = 22;
constexpr uint32_t HeaderBytes   = 208;           // view_rot[3], clip_prev[4], up_view, fparams, mparams, gparams, xparams, yparams
// calib[] (after the header; the shader's CALIB_* / STATS): samples, normal sums per layer x layout x space x screen
// y convention, byte statistics, floor votes, then the cumulative stats counters (never cleared).
constexpr uint32_t NLayouts      = 13;
constexpr uint32_t CalibNormal   = 1;
constexpr uint32_t CalibBytes    = CalibNormal + 4 * NLayouts * 4;
constexpr uint32_t CalibFloors   = CalibBytes + 16 * 3;
constexpr uint32_t CalibWet      = CalibFloors + 2; // puddle guess: count + 16 byte sums; water: the same
constexpr uint32_t CalibHist     = 300;             // 16 byte fields x 16 bins
constexpr uint32_t StatsIndex    = 560;
constexpr uint32_t StatsCount    = 40; // (refl10: 30..32 why rays miss, 33 the composite's mirror rays; hwrt4: 34..39 the metal pass)
// (refl9) The metal survey (the shader's SURVEY, mode 8; never cleared): static / moving sample counts, set-bit counts
// per class x layer x bit, byte histograms per class x field x 16 bins.
constexpr uint32_t SurveyIndex   = 600;
constexpr uint32_t SurveyBits    = 2;
constexpr uint32_t SurveyHist    = 258;
constexpr uint32_t SurveyCount   = 770;
constexpr uint32_t MetalHist     = 1400;          // (hwrt4) the metal pixels' roughness, 10 bins (cumulative)
constexpr uint32_t FrameBytes    = 8192;          // header + calibration + stats + survey
constexpr uint32_t MetalAmb      = 1420;          // (rt3fx) the metal pass's smoothed frame light: 2 x 4 floats (the shader's MT_AMB)
constexpr uint32_t ExposureIndex = 128 / 4;       // fparams.x (1.0) when the game gives no exposure
constexpr uint32_t CalibRuns     = 24;
constexpr uint32_t MaxMips       = 12;
static_assert(CalibWet + 34 <= CalibHist && CalibHist + 256 <= StatsIndex);
static_assert(HeaderBytes + (StatsIndex + StatsCount) * 4 <= FrameBytes);
static_assert(StatsIndex + StatsCount <= SurveyIndex && HeaderBytes + (SurveyIndex + SurveyCount) * 4 <= FrameBytes);
static_assert(SurveyIndex + SurveyCount <= MetalHist && HeaderBytes + (MetalHist + 12) * 4 <= FrameBytes);
static_assert(SurveyHist == SurveyBits + 2 * 128 && SurveyHist + 2 * 256 == SurveyCount);
static_assert(MetalHist + 12 <= MetalAmb && HeaderBytes + (MetalAmb + 8) * 4 <= FrameBytes);

struct RtPush {
	float   proj[4];
	float   depth_map[4];
	float   size[4];
	float   params[4];
	float   motion[4];
	int32_t misc[4];
	int32_t dec[4];
	int32_t misc2[4];
};
static_assert(sizeof(RtPush) == 128);

struct RtImage {
	VkImage                             image  = VK_NULL_HANDLE;
	VmaAllocation                       memory = nullptr;
	vk::ImageView                       view   = nullptr; // every mip
	std::array<vk::ImageView, MaxMips>  mips {};          // one mip each (storage; hi-z only)
	uint32_t                            levels = 1;
	vk::Extent2D                        extent {};
	vk::Format                          format = vk::Format::eUndefined;
	vk::ImageUsageFlags                 usage {};
	bool                                ready = false;    // in the general layout
};

struct Retired {
	RtImage  image;
	uint64_t frame = 0;
};

struct RtPass {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	// (hwrt2, KYTY_HWRT_REFL) the pass with ray queries into the scene TLAS: + binding 22 (RayInfo), 23 (the TLAS)
	vk::DescriptorSetLayout hw_set_layout = nullptr;
	vk::PipelineLayout      hw_layout     = nullptr;
	vk::Pipeline            hw_pipeline   = nullptr;
	bool                    hw_failed     = false;
	// (hwrt3, KYTY_HWRT_REFL_MATERIAL) the same with binding 24, the albedo array of the off-screen hits
	vk::DescriptorSetLayout hwm_set_layout = nullptr;
	vk::PipelineLayout      hwm_layout     = nullptr;
	vk::Pipeline            hwm_pipeline   = nullptr;
	bool                    hwm_failed     = false;
	uint64_t                hwm_frames     = 0;
	uint64_t                hw_frames = 0, sw_frames = 0;
	vk::Sampler             point = nullptr, linear = nullptr;
	bool                    failed = false; // (the pipeline could not be made: off for the session)
	bool                    images_failed = false;
	vk::Extent2D            render {}, trace {};
	int                     level = 0;
	RtImage                 hiz, rad, aux, tmp_a, tmp_b, out;
	RtImage                 gb[2], hist[2], meta[2];
	RtImage                 dummy; // 1x1 R32_UINT stand-in for absent G-buffer layers
	std::vector<Retired>    retired;
	VkBuffer                frame_buf = VK_NULL_HANDLE; // device-local: the shader's header, sums and counters
	VmaAllocation           frame_dev_mem = nullptr;
	VkBuffer                read_buf = VK_NULL_HANDLE;  // host: its copy after every frame (what the CPU reads)
	VmaAllocation           frame_mem = nullptr;
	// The material roughness range (a mirror below lo, nothing from hi; scale): KYTY_RT_REFLECTIONS_RANGE / _MATERIAL.
	float                   mat_lo = 0.10f, mat_hi = 0.40f, mat_scale = 1.0f;
	bool                    mat_auto = false, mat_forced = false;
	// (refl6) The gloss field (1 - gloss as a second roughness: the rougher wins) and the metal field (-1: none).
	int32_t                 dec_gloss = -1, dec_metal = -1;
	// (refl9) The metal / glossy paths' strengths (KYTY_RT_REFLECTIONS_METAL / _GLOSSY) and the metal survey (the field
	// found from the G-buffer itself unless KYTY_RT_REFLECTIONS_METAL_FIELD gives it or turns it off).
	float                   metal_strength = 0.35f, glossy_strength = 0.25f, metal_tint = 0.7f;
	float                   min_area = 10.0f, pool_boost = 0.6f; // (refl9b: KYTY_RT_REFLECTIONS_MIN_AREA / _POOL_BOOST)
	bool                    survey_on = false, survey_done = false;
	uint64_t                survey_runs = 0, survey_next_log = 0;
	bool                    moving_on = true, moving_auto_off = false;
	const int32_t*          mapped    = nullptr;
	bool                    frame_cleared = false;
	uint64_t                frame = 0;
	bool                    history_valid = false;
	// Calibration (up to 3 attempts, the first once the G-buffer copy has been there 60 frames running) and the
	// chosen decode; the screen's y convention (true: the render's row 0 is NDC y +1, the GNM viewport).
	uint32_t                calib_runs = 0, calib_wait = 0, calib_tick = 0, calib_attempts = 0, calib_packed = 0;
	bool                    calib_done = false, calib_active = false, calib_clear = false, floors_done = false;
	uint64_t                calib_next = 0, gb_streak = 0;
	int32_t                 dec_normal = -1, dec_rough = -1;
	bool                    rough_trusted = false;
	bool                    row0_up = true, row0_forced = false;
	bool                    have_water = false; // (this frame: the G-buffer-time depth is bound)
	// (refl8) The guide (the game's own SSR): its confidence range, rgb premultiplied, strength; the last one seen.
	float                   guide_lo = 0.1f, guide_hi = 0.6f, guide_strength = 1.0f;
	bool                    guide_premul = false, have_guide = false;
	RtReflGuide             guide_last {};
	uint64_t                stat_guide_frames = 0;
	// The periodic line (every 10 s): the stats counters' last values and the sums since the last line.
	std::array<uint32_t, StatsCount> stat_last {};
	std::array<uint64_t, StatsCount> stat_sum {};
	uint64_t                stat_frames = 0, stat_gbuf_frames = 0, stat_copies0 = 0;
	std::chrono::steady_clock::time_point stat_time = std::chrono::steady_clock::now();
	// The GPU time (timestamps, always: two a frame).
	// Per pass: a timestamp before the first and after each pass; ring of 8 frames x 16 queries.
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	double                  sum_ms = 0.0;
	uint32_t                samples = 0;
	std::array<std::array<uint8_t, 16>, 8> stage_of {}; // the pass each interval timed (Stage), per ring entry
	std::array<uint32_t, 8>               stamps_in {};  // timestamps written per ring entry
	std::array<double, 11>                stage_ms {};   // sums since the last line
	std::array<uint32_t, 11>              stage_frames {};
	// (hwrt4, KYTY_HWRT_METAL) the metal reflections' images (trace size: level 1 half the render, 2 full; mout: render) and state
	RtImage                 mrad, maux, mout, mgb[2], mhist[2], mmeta[2], mtmp[2];
	vk::Extent2D            mtrace {};
	int                     mlevel = 0;
	bool                    mhist_valid = false;
	uint64_t                m_frames = 0, m_tlas_frames = 0;
	std::array<uint32_t, 10> m_hist_last {};
	uint32_t                m_px_last = 0, m_dark_last = 0;
	uint64_t                m_dark_frames = 0, m_dark_px = 0, m_all_px = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
} g_rt;

float EnvFloat(const char* name, float fallback, float lo, float hi) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') return fallback;
	const float v = static_cast<float>(std::atof(value));
	return std::isfinite(v) ? std::clamp(v, lo, hi) : fallback;
}
// "<layer>:<shift>:<bits>[g]" -> the shader's field code (-1: unset or malformed).
int32_t FieldEnv(const char* name) {
	const char* value = std::getenv(name);
	if (value == nullptr) return -1;
	int  layer = -1, shift = -1, bits = -1;
	char flag  = 0;
	if (std::sscanf(value, "%d:%d:%d%c", &layer, &shift, &bits, &flag) < 3) return -1;
	if (layer < 0 || layer > 3 || shift < 0 || bits < 1 || bits > 16 || shift + bits > 32) return -1;
	return layer | (shift << 2) | (bits << 7) | (flag == 'g' ? 4096 : 0);
}
// "<layer>:<layout>[w][n]" -> the shader's normal code (-1: unset or malformed).
int32_t NormalEnv(const char* name) {
	const char* value = std::getenv(name);
	if (value == nullptr) return -1;
	int layer = -1, layout = -1;
	if (std::sscanf(value, "%d:%d", &layer, &layout) < 2 || layer < 0 || layer > 3 || layout < 0 ||
	    layout >= static_cast<int>(NLayouts))
		return -1;
	const std::string v = value;
	return layer | (layout << 2) | (v.find('w') != std::string::npos ? 64 : 0) | (v.find('n') != std::string::npos ? 128 : 0);
}

void FreeImage(GraphicContext& graphics, RtImage& img) {
	if (img.view != nullptr) graphics.device.destroyImageView(img.view, nullptr);
	for (auto& v: img.mips)
		if (v != nullptr && v != img.view) graphics.device.destroyImageView(v, nullptr);
	if (img.image != VK_NULL_HANDLE) vmaDestroyImage(graphics.allocator, img.image, img.memory);
	img = RtImage {};
}
// (Images replaced at a size change may still be read by frames in flight: destroyed 16 frames later.)
void Retire(RtImage& img) {
	if (img.image != VK_NULL_HANDLE) g_rt.retired.push_back({img, g_rt.frame});
	img = RtImage {};
}
void FreeRetired(GraphicContext& graphics) {
	auto& r = g_rt.retired;
	for (size_t i = 0; i < r.size();) {
		if (g_rt.frame >= r[i].frame + 16) {
			FreeImage(graphics, r[i].image);
			r[i] = r.back();
			r.pop_back();
		} else {
			++i;
		}
	}
}

bool MakeImage(GraphicContext& graphics, RtImage& img, vk::Format format, vk::Extent2D extent, uint32_t levels, bool storage) {
	VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create.imageType   = VK_IMAGE_TYPE_2D;
	create.format      = static_cast<VkFormat>(format);
	create.extent      = {extent.width, extent.height, 1};
	create.mipLevels   = levels;
	create.arrayLayers = 1;
	create.samples     = VK_SAMPLE_COUNT_1_BIT;
	create.tiling      = VK_IMAGE_TILING_OPTIMAL;
	create.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
	               (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0u);
	create.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
	create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO;
	if (vmaCreateImage(graphics.allocator, &create, &allocate, &img.image, &img.memory, nullptr) != VK_SUCCESS) {
		img.image = VK_NULL_HANDLE;
		return false;
	}
	img.levels = levels;
	img.extent = extent;
	img.format = format;
	img.usage  = static_cast<vk::ImageUsageFlags>(create.usage);
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = vk::Image(img.image);
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = format;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, levels, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &img.view) != vk::Result::eSuccess) {
		img.view = nullptr;
		return false;
	}
	if (levels == 1) {
		img.mips[0] = img.view;
		return true;
	}
	for (uint32_t m = 0; m < levels && m < MaxMips; ++m) {
		view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, m, 1, 0, 1};
		if (graphics.device.createImageView(&view_info, nullptr, &img.mips[m]) != vk::Result::eSuccess) {
			img.mips[m] = nullptr;
			return false;
		}
	}
	return true;
}
vk::Extent2D MipExtent(const RtImage& img, uint32_t m) {
	return {std::max(1u, img.extent.width >> m), std::max(1u, img.extent.height >> m)};
}

void Barrier(vk::CommandBuffer command, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src, vk::PipelineStageFlags2 dst_stage,
             vk::AccessFlags2 dst) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = src_stage;
	barrier.srcAccessMask = src;
	barrier.dstStageMask  = dst_stage;
	barrier.dstAccessMask = dst;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

bool HwReflWanted();
int  HwReflQuality() {
	static const int q = [] {
		if (!HwReflWanted()) return 3;
		const char* v = std::getenv("KYTY_HWRT_REFL_QUALITY");
		const int   r = v != nullptr && *v != '\0' ? std::clamp(std::atoi(v), 1, 4) : 3;
		std::printf("HWRTQ: reflections quality %d (%s)%s\n", r, r == 1 ? "Bajo" : r == 2 ? "Medio" : r == 3 ? "Alto" : "Ultra",
		            v != nullptr && *v != '\0' ? "" : " (KYTY_HWRT_REFL_QUALITY unset: default)");
		std::fflush(stdout);
		return r;
	}();
	return q;
}

// (hwrt4) KYTY_HWRT_METAL=1|2 (default 0, off: nothing created, nothing recorded): ray-traced reflections on metal
// (armour, weapons, shields, metal fittings) through the scene TLAS, independent of the game's SSR guide. Level 1: one
// GGX-sampled ray per pixel at half the render resolution; 2: at full. Needs KYTY_HWRT=1 and KYTY_HWRT_REFL=1 (the TLAS
// bound to this pass) and runs inside the RT reflections (KYTY_HWRT_REFL turns them on). Settings:
//   KYTY_HWRT_METAL_FIELD  the metal field, <layer>:<shift>:<bits>[g] (2:0:8, L2 bits 0..7: the refl9 metal survey's
//                          pick, near-binary, set on 62 % of the characters' pixels and 8 % of the static world's)
//   KYTY_HWRT_METAL_ROUGH  the roughness field (1:24:8, L1 bits 24..31: the byte the oct 12:12 normal leaves free in its
//                          word; "g" suffix: it is gloss)
//   KYTY_HWRT_METAL_ROUGH_MAX  no ray from this roughness (0.85: worn plate reads 0.5-0.8 and keeps a soft, blurred
//                          reflection; fades from 70 % of it)
//   KYTY_HWRT_METAL_STRENGTH   the composite's weight scale (1.0, 0..2)
//   KYTY_HWRT_METAL_TINT       how much of the base colour's hue the F0 keeps (1.0 physical; 0 grey F0 of its luminance)
//   KYTY_HWRT_METAL_DIST       the reflection ray's length (world units, 200)
//   KYTY_HWRT_METAL_MIN_F0     plausibility: a metal pixel's base colour (linear, brightest channel) must reach this to
//                          count fully (0.12; fades from 60 % of it; 0 off): dark stone or dirt with the metal bit set
//                          is not bare metal
//   KYTY_HWRT_METAL_SELF=0     do not skip the pixel's own TLAS instance near the surface
//   KYTY_HWRT_METAL_VIEW=1|2|3|4  debug views: 1 metal field as grey, 2 roughness field as grey, 3 F0 (the albedo) on
//                          metal, 4 our reflection alone (envBRDF x hit) on metal, black elsewhere
struct MetalCfg {
	int     level = 0, view = 0;
	int32_t metal = -1, rough = -1;
	float   rough_max = 0.6f, strength = 1.0f, tint = 1.0f, dist = 200.0f;
	bool    self_skip = true;
	float   min_f0 = 0.12f;
	int     quality = 2, passes = 2; // (hwrtq) KYTY_HWRT_METAL_QUALITY
	bool    guess = false;            // (rt3fx) KYTY_HWRT_METAL_GUESS
};
const MetalCfg& Metal() {
	static const MetalCfg c = [] {
		MetalCfg m;
		const char* v = std::getenv("KYTY_HWRT_METAL");
		m.level       = v != nullptr ? std::clamp(std::atoi(v), 0, 2) : 0;
		if (m.level == 0) return m;
		// (hwrtq) KYTY_HWRT_METAL_QUALITY=1..4 (Bajo / Medio / Alto / Ultra; unset: 2 at KYTY_HWRT_METAL=1, 3 at 2, today's):
		// 1 half res, 100 m rays, no ray above roughness 0.6, 1 a-trous pass; 2 half res, 200 m, 0.85, 2 passes; 3 full res,
		// 200 m, 0.85, 2 passes; 4 full res, 400 m, 0.9, 3 passes. The _DIST / _ROUGH_MAX knobs still override.
		const char* qv = std::getenv("KYTY_HWRT_METAL_QUALITY");
		m.quality      = qv != nullptr && *qv != '\0' ? std::clamp(std::atoi(qv), 1, 4) : (m.level >= 2 ? 3 : 2);
		m.level        = m.quality >= 3 ? 2 : 1;
		m.passes       = m.quality == 1 ? 1 : m.quality == 4 ? 3 : 2;
		std::printf("HWRTQ: metal quality %d (%s)%s\n", m.quality,
		            m.quality == 1 ? "Bajo" : m.quality == 2 ? "Medio" : m.quality == 3 ? "Alto" : "Ultra",
		            qv != nullptr && *qv != '\0' ? "" : " (KYTY_HWRT_METAL_QUALITY unset: from KYTY_HWRT_METAL)");
		const char* vw = std::getenv("KYTY_HWRT_METAL_VIEW");
		m.view         = vw != nullptr ? std::clamp(std::atoi(vw), 0, 4) : 0;
		m.metal        = std::getenv("KYTY_HWRT_METAL_FIELD") != nullptr ? FieldEnv("KYTY_HWRT_METAL_FIELD") : (2 | (0 << 2) | (8 << 7));
		m.rough        = std::getenv("KYTY_HWRT_METAL_ROUGH") != nullptr ? FieldEnv("KYTY_HWRT_METAL_ROUGH") : (1 | (24 << 2) | (8 << 7));
		m.rough_max    = EnvFloat("KYTY_HWRT_METAL_ROUGH_MAX", m.quality == 1 ? 0.6f : m.quality == 4 ? 0.9f : 0.85f, 0.05f, 1.0f);
		m.strength     = EnvFloat("KYTY_HWRT_METAL_STRENGTH", 1.0f, 0.0f, 2.0f);
		m.tint         = EnvFloat("KYTY_HWRT_METAL_TINT", 1.0f, 0.0f, 1.0f);
		m.dist         = EnvFloat("KYTY_HWRT_METAL_DIST", m.quality == 1 ? 100.0f : m.quality == 4 ? 400.0f : 200.0f, 1.0f, 5000.0f);
		m.min_f0       = EnvFloat("KYTY_HWRT_METAL_MIN_F0", 0.12f, 0.0f, 1.0f);
		const char* sf = std::getenv("KYTY_HWRT_METAL_SELF");
		m.self_skip    = !(sf != nullptr && sf[0] == '0');
		// (rt3fx) KYTY_HWRT_METAL_GUESS=1: hits off the screen or hidden there are shaded by a guess (the captured albedo
		// under the frame's mean light, or grey). Default 0: only hits visible on the screen count (the frame's own colour
		// there); any other hit is a miss and the game's own shading stays - no colour is invented.
		const char* gs = std::getenv("KYTY_HWRT_METAL_GUESS");
		m.guess        = gs != nullptr && gs[0] == '1';
		const auto fs  = [](int32_t f, char* buf, size_t n) {
			if (f < 0) std::snprintf(buf, n, "none");
			else std::snprintf(buf, n, "L%d bits %d..%d%s", f & 3, (f >> 2) & 31, ((f >> 2) & 31) + ((f >> 7) & 31) - 1, (f & 4096) ? " inverted" : "");
		};
		char mf[48], rf[48];
		fs(m.metal, mf, sizeof(mf)), fs(m.rough, rf, sizeof(rf));
		std::printf("RT metal: KYTY_HWRT_METAL=%d (%s resolution, 1 GGX ray per pixel into the scene TLAS), metal field %s, roughness "
		            "field %s, rays below roughness %.2f, strength %g, F0 tint %g, ray length %g, self-instance skip %s, view %d, off-screen guesses %s (KYTY_HWRT_METAL_GUESS)%s\n",
		            m.level, m.level >= 2 ? "full" : "half", mf, rf, static_cast<double>(m.rough_max), static_cast<double>(m.strength),
		            static_cast<double>(m.tint), static_cast<double>(m.dist), m.self_skip ? "on" : "off", m.view, m.guess ? "on" : "off",
		            HwReflWanted() ? "" : " (KYTY_HWRT_REFL is not 1: no TLAS bound, nothing is traced)");
		std::fflush(stdout);
		return m;
	}();
	return c;
}
bool EnsureMetalImages(GraphicContext& graphics) {
	const vk::Extent2D render = g_rt.render;
	const vk::Extent2D trace  = Metal().level >= 2 ? render : vk::Extent2D {std::max(1u, (render.width + 1) / 2), std::max(1u, (render.height + 1) / 2)};
	if (g_rt.mout.image != VK_NULL_HANDLE && g_rt.mtrace == trace && g_rt.mout.extent == render) return true;
	for (auto* img: {&g_rt.mrad, &g_rt.maux, &g_rt.mout, &g_rt.mgb[0], &g_rt.mgb[1], &g_rt.mhist[0], &g_rt.mhist[1], &g_rt.mmeta[0],
	                 &g_rt.mmeta[1], &g_rt.mtmp[0], &g_rt.mtmp[1]})
		Retire(*img);
	const auto f16 = vk::Format::eR16G16B16A16Sfloat;
	bool       ok  = MakeImage(graphics, g_rt.mout, f16, render, 1, true);
	for (auto* img: {&g_rt.mrad, &g_rt.maux, &g_rt.mgb[0], &g_rt.mgb[1], &g_rt.mhist[0], &g_rt.mhist[1], &g_rt.mmeta[0], &g_rt.mmeta[1],
	                 &g_rt.mtmp[0], &g_rt.mtmp[1]})
		ok = ok && MakeImage(graphics, *img, f16, trace, 1, true);
	if (!ok) {
		for (auto* img: {&g_rt.mrad, &g_rt.maux, &g_rt.mout, &g_rt.mgb[0], &g_rt.mgb[1], &g_rt.mhist[0], &g_rt.mhist[1], &g_rt.mmeta[0],
		                 &g_rt.mmeta[1], &g_rt.mtmp[0], &g_rt.mtmp[1]})
			Retire(*img);
		return false;
	}
	g_rt.mtrace      = trace;
	g_rt.mhist_valid = false;
	return true;
}

std::array<RtImage*, 24> Sized() {
	return {&g_rt.hiz, &g_rt.rad, &g_rt.aux, &g_rt.tmp_a, &g_rt.tmp_b, &g_rt.out, &g_rt.gb[0], &g_rt.gb[1], &g_rt.hist[0],
	        &g_rt.hist[1], &g_rt.meta[0], &g_rt.meta[1], &g_rt.dummy, &g_rt.mrad, &g_rt.maux, &g_rt.mout, &g_rt.mgb[0],
	        &g_rt.mgb[1], &g_rt.mhist[0], &g_rt.mhist[1], &g_rt.mmeta[0], &g_rt.mmeta[1], &g_rt.mtmp[0], &g_rt.mtmp[1]};
}

bool EnsurePipeline(GraphicContext& graphics) {
	std::lock_guard ensure_lock(LocalPassCache::EnsureMutex()); // (the prewarm thread's, local-pass-cache.h)
	if (g_rt.pipeline != nullptr) return true;
	if (g_rt.failed) return false;
	g_rt.failed       = true;
	const auto device = graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, Bindings> bindings {};
	for (uint32_t i = 0; i < Bindings; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = (i == 18 || i == 19)                       ? vk::DescriptorType::eStorageBuffer
		                              : (i == 4 || i == 11 || i == 12 || i == 13) ? vk::DescriptorType::eStorageImage
		                                                                          : vk::DescriptorType::eCombinedImageSampler;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = Bindings;
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_rt.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(RtPush)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_rt.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_rt.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(RT_REFLECTIONS_SPV);
	module_info.pCode       = RT_REFLECTIONS_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_rt.layout;
	const auto created         = device.createComputePipelines(LocalPassCache::Get(device), 1, &pipeline_info, nullptr, &g_rt.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_rt.pipeline = nullptr;
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
		sampler.maxLod       = VK_LOD_CLAMP_NONE;
		if (device.createSampler(&sampler, nullptr, i == 0 ? &g_rt.point : &g_rt.linear) != vk::Result::eSuccess) return false;
	}
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = FrameBytes;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VmaAllocationCreateInfo dev {};
	dev.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &dev, &g_rt.frame_buf, &g_rt.frame_dev_mem, nullptr) != VK_SUCCESS) {
		g_rt.frame_buf = VK_NULL_HANDLE;
		return false;
	}
	buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &g_rt.read_buf, &g_rt.frame_mem, &mapped) != VK_SUCCESS) {
		g_rt.read_buf = VK_NULL_HANDLE;
		return false;
	}
	g_rt.mapped = static_cast<const int32_t*>(mapped.pMappedData);
	if (!MakeImage(graphics, g_rt.dummy, vk::Format::eR32Uint, {1, 1}, 1, false)) return false;
	{
		vk::QueryPoolCreateInfo query {};
		query.queryType  = vk::QueryType::eTimestamp;
		query.queryCount = 8 * 16;
		if (device.createQueryPool(&query, nullptr, &g_rt.pool) != vk::Result::eSuccess) g_rt.pool = nullptr;
		g_rt.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	}
	g_rt.failed = false;
	return true;
}

// (hwrt2) KYTY_HWRT_REFL=1: the trace's misses (off the screen, behind something, toward the camera) take a hardware
// ray query into the scene captured by KYTY_HWRT (local/hwrt-scene.h). Needs KYTY_HWRT=1; turns the reflections on
// (high) when KYTY_RT_REFLECTIONS is not set.
bool HwReflWanted() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_HWRT_REFL");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

bool EnsureHwPipeline(GraphicContext& graphics) {
	std::lock_guard ensure_lock(LocalPassCache::EnsureMutex()); // (the prewarm thread's, local-pass-cache.h)
	if (g_rt.hw_pipeline != nullptr) return true;
	if (g_rt.hw_failed) return false;
	g_rt.hw_failed    = true;
	const auto device = graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, Bindings + 2> bindings {};
	for (uint32_t i = 0; i < Bindings + 2; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = i == Bindings + 1                                       ? vk::DescriptorType::eAccelerationStructureKHR
		                              : (i == 18 || i == 19 || i == Bindings)                 ? vk::DescriptorType::eStorageBuffer
		                              : (i == 4 || i == 11 || i == 12 || i == 13)             ? vk::DescriptorType::eStorageImage
		                                                                                      : vk::DescriptorType::eCombinedImageSampler;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = Bindings + 2;
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_rt.hw_set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(RtPush)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_rt.hw_set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_rt.hw_layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(RT_REFLECTIONS_HW_SPV);
	module_info.pCode       = RT_REFLECTIONS_HW_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_rt.hw_layout;
	const auto created         = device.createComputePipelines(LocalPassCache::Get(device), 1, &pipeline_info, nullptr, &g_rt.hw_pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_rt.hw_pipeline = nullptr;
		std::printf("RT reflections: the hardware ray query pipeline could not be made (KYTY_HWRT_REFL): screen-space only\n");
		std::fflush(stdout);
		return false;
	}
	g_rt.hw_failed = false;
	std::printf("RT reflections: hardware ray queries into the scene TLAS for the screen-space misses (KYTY_HWRT_REFL=1)\n");
	std::fflush(stdout);
	return true;
}

// (hwrt3) KYTY_HWRT_REFL_MATERIAL=1 (with KYTY_HWRT_REFL=1): the ray query pass whose off-screen hits take the captured
// meshes' base colour (binding 24: the albedo array of local/hwrt-scene.cpp). Off: this pipeline is never made.
bool EnsureHwMaterialPipeline(GraphicContext& graphics) {
	std::lock_guard ensure_lock(LocalPassCache::EnsureMutex()); // (the prewarm thread's, local-pass-cache.h)
	if (g_rt.hwm_pipeline != nullptr) return true;
	if (g_rt.hwm_failed) return false;
	g_rt.hwm_failed   = true;
	const auto device = graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, Bindings + 3> bindings {};
	for (uint32_t i = 0; i < Bindings + 3; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = i == Bindings + 1                                       ? vk::DescriptorType::eAccelerationStructureKHR
		                              : (i == 18 || i == 19 || i == Bindings)                 ? vk::DescriptorType::eStorageBuffer
		                              : (i == 4 || i == 11 || i == 12 || i == 13)             ? vk::DescriptorType::eStorageImage
		                                                                                      : vk::DescriptorType::eCombinedImageSampler;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = Bindings + 3;
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g_rt.hwm_set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(RtPush)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g_rt.hwm_set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g_rt.hwm_layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(RT_REFLECTIONS_HWM_SPV);
	module_info.pCode       = RT_REFLECTIONS_HWM_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g_rt.hwm_layout;
	const auto created         = device.createComputePipelines(LocalPassCache::Get(device), 1, &pipeline_info, nullptr, &g_rt.hwm_pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g_rt.hwm_pipeline = nullptr;
		std::printf("RT reflections: the material ray query pipeline could not be made (KYTY_HWRT_REFL_MATERIAL): flat off-screen hits\n");
		std::fflush(stdout);
		return false;
	}
	g_rt.hwm_failed = false;
	std::printf("RT reflections: off-screen hits take the captured meshes' base colour (KYTY_HWRT_REFL_MATERIAL=1)\n");
	std::fflush(stdout);
	return true;
}

// The quality levels (KYTY_RT_REFLECTIONS 1..4). Costs measured with the first version on an RTX 5080 at a 1080p render:
// full-resolution trace, 72 iterations, 3 a-trous passes, every pixel = 8.5-10 ms; most of it the march, then the wet-
// surface mask evaluated twice per pixel (trace and composite) and the full-resolution filters.
struct LevelCfg {
	const char* name;
	bool        full;     // trace at the render resolution (else half: a quarter of the pixels)
	bool        checker;  // half of the trace pixels march each frame (the temporal pass fills the others)
	int         iters;    // hi-z iterations (rough rays: half)
	float       cutoff;   // roughness above which nothing reflects
	int         passes;   // a-trous passes
	float       max_len;  // view-space ray length
	float       cap;      // the longest ray on screen, x the render height
	float       alpha;    // temporal minimum alpha scale
};
constexpr LevelCfg Levels[6] = {
    {"off", false, false, 0, 0.0f, 0, 0.0f, 0.0f, 1.0f},
    {"low", false, true, 24, 0.45f, 1, 200.0f, 0.30f, 1.0f},
    {"medium", false, false, 32, 0.55f, 2, 300.0f, 0.45f, 1.0f},
    {"high", false, false, 48, 0.60f, 3, 800.0f, 0.75f, 0.9f}, // (10-24 far: 400 m / 0.65)
    {"ultra", true, true, 72, 0.65f, 3, 1600.0f, 1.00f, 0.8f}, // (10-24 far: 64 / 600 m / 0.90)
    // (refl10) every pixel every frame (no checkerboard), longer rays (2.5 x the render height on screen: the whole
    // screen; 128 iterations), 2 rays on glossy surfaces (KYTY_RT_REFLECTIONS_RPP). About 2-2.5x ultra's GPU time.
    {"extreme", true, false, 128, 0.65f, 3, 3000.0f, 2.50f, 0.8f},
};
constexpr int MaxLevel = 5;
const LevelCfg& Cfg(int level) { return Levels[std::clamp(level, 0, MaxLevel)]; }

bool EnsureImages(GraphicContext& graphics, vk::Extent2D render, int level) {
	if (g_rt.out.image != VK_NULL_HANDLE && g_rt.render == render && g_rt.level == level) return true;
	for (auto* img: Sized())
		if (img != &g_rt.dummy) Retire(*img);
	const vk::Extent2D trace = Cfg(level).full ? render : vk::Extent2D {std::max(1u, (render.width + 1) / 2), std::max(1u, (render.height + 1) / 2)};
	uint32_t levels = 1;
	while (levels < MaxMips && ((render.width >> levels) > 0 || (render.height >> levels) > 0)) ++levels;
	const auto f16 = vk::Format::eR16G16B16A16Sfloat;
	bool ok = MakeImage(graphics, g_rt.hiz, vk::Format::eR32Sfloat, render, levels, true) &&
	          MakeImage(graphics, g_rt.out, f16, render, 1, true);
	for (auto* img: {&g_rt.rad, &g_rt.aux, &g_rt.tmp_a, &g_rt.tmp_b, &g_rt.gb[0], &g_rt.gb[1], &g_rt.hist[0], &g_rt.hist[1],
	                 &g_rt.meta[0], &g_rt.meta[1]})
		ok = ok && MakeImage(graphics, *img, f16, trace, 1, true);
	if (!ok) {
		for (auto* img: Sized())
			if (img != &g_rt.dummy) Retire(*img);
		g_rt.images_failed = true;
		return false;
	}
	g_rt.render        = render;
	g_rt.trace         = trace;
	g_rt.level         = level;
	g_rt.history_valid = false;
	return true;
}

// Every image not yet in the general layout moved there (contents undefined: the history is invalid anyway).
void PrepareImages(vk::CommandBuffer command) {
	std::vector<vk::ImageMemoryBarrier2> barriers;
	bool                                 dummy = false;
	for (auto* img: Sized()) {
		if (img->ready || img->image == VK_NULL_HANDLE) continue;
		vk::ImageMemoryBarrier2 b {};
		b.srcStageMask        = vk::PipelineStageFlagBits2::eNone;
		b.dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
		b.dstAccessMask       = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite;
		b.oldLayout           = vk::ImageLayout::eUndefined;
		b.newLayout           = vk::ImageLayout::eGeneral;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image               = vk::Image(img->image);
		b.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, img->levels, 0, 1};
		barriers.push_back(b);
		img->ready = true;
		if (img == &g_rt.dummy) dummy = true;
	}
	if (!barriers.empty()) {
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
		dependency.pImageMemoryBarriers    = barriers.data();
		command.pipelineBarrier2(dependency);
	}
	if (dummy) {
		vk::ClearColorValue zero {};
		zero.uint32 = std::array<uint32_t, 4> {0, 0, 0, 0};
		const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		command.clearColorImage(vk::Image(g_rt.dummy.image), vk::ImageLayout::eGeneral, &zero, 1, &whole);
	}
}

enum Stage : uint8_t { StHiz, StCalib, StTrace, StTemporal, StAtrous0, StAtrous1, StAtrous2, StAtrous3, StComposite, StSurvey, StMetal };
constexpr uint32_t NStages = 11;
const char* const StageNames[NStages] = {"hi-z", "calib", "trace", "temporal", "a-trous 1", "a-trous 2", "a-trous 3", "a-trous 4",
                                         "composite", "metal survey", "metal (KYTY_HWRT_METAL)"};

void ReadTiming(GraphicContext& graphics) {
	if (g_rt.pool == nullptr) return;
	while (g_rt.read + 4 <= g_rt.written) {
		const auto     entry = static_cast<uint32_t>(g_rt.read % 8);
		const uint32_t n     = g_rt.stamps_in[entry];
		std::array<uint64_t, 16> stamps {};
		if (n >= 2 && graphics.device.getQueryPoolResults(g_rt.pool, entry * 16, n, sizeof(uint64_t) * n, stamps.data(),
		                                                  sizeof(uint64_t), vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		if (n >= 2 && stamps[n - 1] > stamps[0]) {
			g_rt.sum_ms += static_cast<double>(stamps[n - 1] - stamps[0]) * g_rt.period_ns / 1e6;
			++g_rt.samples;
			std::array<bool, NStages> seen {};
			for (uint32_t k = 1; k < n; ++k) {
				const uint8_t st = g_rt.stage_of[entry][k];
				if (st >= NStages || stamps[k] < stamps[k - 1]) continue;
				g_rt.stage_ms[st] += static_cast<double>(stamps[k] - stamps[k - 1]) * g_rt.period_ns / 1e6;
				seen[st] = true;
			}
			for (uint32_t st = 0; st < NStages; ++st)
				if (seen[st]) ++g_rt.stage_frames[st];
		}
		++g_rt.read;
	}
}

// The stats counters (the shader's atomics on 1 pixel in 16, cumulative: the difference since the last frame, modulo
// 2^32) and, every 10 s, one line that says whether the reflections show: the share of pixels with a weight > 0.05, the
// mean weight, the rays' hit rate, the G-buffer's state, the GPU time.
void ReadStats(GraphicContext& graphics, bool have_gbuf, int level) {
	if (g_rt.mapped == nullptr) return;
	(void)vmaInvalidateAllocation(graphics.allocator, g_rt.frame_mem, 0, VK_WHOLE_SIZE);
	const auto* c = reinterpret_cast<const uint32_t*>(g_rt.mapped + HeaderBytes / 4 + StatsIndex);
	for (uint32_t k = 0; k < StatsCount; ++k) {
		const uint32_t v = c[k];
		// (the first frames: the buffer's clear may not have run yet, its contents are a baseline only)
		if (g_rt.frame > 8) g_rt.stat_sum[k] += static_cast<uint32_t>(v - g_rt.stat_last[k]);
		g_rt.stat_last[k] = v;
	}
	if (Metal().level > 0 && g_rt.frame > 8) {
		// (hwrt4) per frame: the share of the metal pixels the composite changed whose result would have been > 30 %
		// darker than the game's (before the 0.85 floor); a frame over 5 % counts as a dark burst
		const auto*    mh = reinterpret_cast<const uint32_t*>(g_rt.mapped + HeaderBytes / 4 + MetalHist);
		const uint32_t np = mh[10] - g_rt.m_px_last, nd = mh[11] - g_rt.m_dark_last;
		g_rt.m_px_last = mh[10], g_rt.m_dark_last = mh[11];
		g_rt.m_all_px += np, g_rt.m_dark_px += nd;
		if (np >= 50 && nd * 20u > np) ++g_rt.m_dark_frames;
	}
	++g_rt.stat_frames;
	if (have_gbuf) ++g_rt.stat_gbuf_frames;
	if (g_rt.have_guide) ++g_rt.stat_guide_frames;
	const auto now = std::chrono::steady_clock::now();
	if (now - g_rt.stat_time < std::chrono::seconds(10)) return;
	const auto&    sm     = g_rt.stat_sum;
	const double   px     = static_cast<double>(std::max<uint64_t>(sm[0], 1));
	const double   area   = (Cfg(level).full ? 1.0 : 0.25) * (Cfg(level).checker ? 0.5 : 1.0); // (rays per render pixel)
	const uint64_t copies = RtReflCopies();
	const bool  mat    = g_rt.dec_rough >= 0 && g_rt.dec_normal >= 0;
	const char* source = g_rt.stat_guide_frames > 0 ? (g_rt.have_water ? "the game's SSR (guide) + water" : "the game's SSR (guide)")
	                     : mat ? (g_rt.have_water ? "material roughness + water" : "material roughness")
	                     : g_rt.have_water ? "water (depth after the G-buffer) only"
	                                       : "nothing (no G-buffer material, no depth copy)";
	std::printf("RT reflections: last %.0f s: mask %s marks %.1f%% of the pixels reflective (water %.1f%%, "
	            "depth changed after the G-buffer %.1f%%); weight > 0.05 on "
	            "%.1f%% of the pixels (%.1f%% on roughness < 0.1), mean weight %.3f (%.3f on reflective surfaces), rays on %.1f%%, "
	            "hit rate %.0f%%, G-buffer copy %llu of %llu frames, its normal used on %.1f%% (%s), screen y %s, GPU %.3f ms\n",
	            std::chrono::duration<double>(now - g_rt.stat_time).count(), source, 100.0 * sm[3] / px, 100.0 * sm[7] / px,
	            100.0 * sm[8] / px, 100.0 * sm[1] / px, 100.0 * sm[19] / px,
	            sm[2] / 1000.0 / px, sm[3] > 0 ? sm[2] / 1000.0 / static_cast<double>(sm[3]) : 0.0,
	            100.0 * sm[4] / (px * area), sm[4] > 0 ? 100.0 * sm[5] / static_cast<double>(sm[4]) : 0.0,
	            static_cast<unsigned long long>(copies - g_rt.stat_copies0), static_cast<unsigned long long>(g_rt.stat_frames),
	            100.0 * sm[6] / px, g_rt.dec_normal >= 0 ? "decoded" : "not identified: depth normals",
	            g_rt.row0_up ? "row 0 = NDC +1" : "row 0 = NDC -1", g_rt.samples > 0 ? g_rt.sum_ms / g_rt.samples : 0.0);
	{
		// (refl6) The roughness buckets (shares of all non-sky pixels; the G-buffer pixels are the "normal used" share):
		// the material roughness used (the rougher of the two fields) and the roughness field alone.
		const auto pc = [&](uint32_t k) { return 100.0 * sm[k] / px; };
		const auto fc = [](int32_t f, char* buf, size_t n) {
			if (f < 0) std::snprintf(buf, n, "none");
			else std::snprintf(buf, n, "L%d bits %d..%d%s", f & 3, (f >> 2) & 31, ((f >> 2) & 31) + ((f >> 7) & 31) - 1,
			                   (f & 4096) ? " as 1 - gloss" : "");
		};
		char rf[48], gf[48], mf[48];
		fc(g_rt.dec_rough, rf, sizeof(rf)), fc(g_rt.dec_gloss, gf, sizeof(gf)), fc(g_rt.dec_metal, mf, sizeof(mf));
		std::printf("RT reflections: last %.0f s: roughness buckets (%% of the pixels) < 0.1 / 0.1-0.25 / 0.25-0.4 / >= 0.4: used "
		            "(rougher of %s and %s) %.1f / %.1f / %.1f / %.1f, roughness field alone %.1f / %.1f / %.1f / %.1f; mirror "
		            "below %.2f, nothing from %.2f; metal (%s) > 0.5 on %.1f%%; moving (characters) %.1f%% (%s)\n",
		            std::chrono::duration<double>(now - g_rt.stat_time).count(), rf, gf, pc(9), pc(10), pc(11), pc(12), pc(13),
		            pc(14), pc(15), pc(16), g_rt.mat_lo, g_rt.mat_hi, mf, pc(18), pc(17),
		            !g_rt.moving_on ? "not skipped: off" : g_rt.moving_auto_off ? "not skipped: too many, the camera's matrices disagree" : "skipped");
		// (more than 35 % of the screen "moving": the camera reprojection does not match the motion vectors; not used for
		// the next 10 s)
		if (g_rt.moving_on) g_rt.moving_auto_off = sm[0] > 100 && pc(17) > 35.0;
	}
	{
		// (refl9) The metal / glossy paths: the share of the pixels that are metal / glossy (G-buffer surfaces), where our
		// hit was added (> 5 %), the mean added weight over all pixels and over those two.
		const auto pc  = [&](uint32_t k) { return 100.0 * sm[k] / px; };
		const double on = static_cast<double>(sm[25] + sm[26]);
		char         mf[64];
		if (g_rt.dec_metal < 0) std::snprintf(mf, sizeof(mf), "%s", g_rt.survey_on && !g_rt.survey_done ? "none yet: surveying" : "none");
		else std::snprintf(mf, sizeof(mf), "L%d bits %d..%d%s", g_rt.dec_metal & 3, (g_rt.dec_metal >> 2) & 31,
		                   ((g_rt.dec_metal >> 2) & 31) + std::max((g_rt.dec_metal >> 7) & 31, 1) - 1, (g_rt.dec_metal & 4096) ? " inverted" : "");
		std::printf("RT reflections: metal / glossy: last %.0f s: metal %.1f%% of the pixels (field %s), glossy %.1f%%; our hit "
		            "added (> 5%%) on metal %.1f%%, glossy %.1f%%; mean added weight %.4f (%.3f on metal + glossy pixels); strengths "
		            "metal %g, glossy %g\n",
		            std::chrono::duration<double>(now - g_rt.stat_time).count(), pc(25), mf, pc(26), pc(28), pc(29), sm[27] / 1000.0 / px,
		            on > 0 ? sm[27] / 1000.0 / on : 0.0, g_rt.metal_strength, g_rt.glossy_strength);
	}
	if (sm[4] > 0) {
		// (refl10) Why the rays miss: the depth limit (iterations: KYTY_RT_REFLECTIONS_ITERS), the screen or the ray's
		// length (KYTY_RT_REFLECTIONS_CAP; off screen is what a screen-space ray cannot see), the hit's validation
		// (silhouettes, back faces, the reflector's own plane, the screen-edge fade); the rest: rays toward the camera.
		const double rays = static_cast<double>(sm[4]);
		std::printf("RT reflections: rays: last %.0f s: hit %.1f%%, missed: out of iterations %.1f%%, off the screen / ray end %.1f%%, "
		            "refused by the hit test %.1f%%; the composite's own mirror rays on %.2f%% of the pixels\n",
		            std::chrono::duration<double>(now - g_rt.stat_time).count(), 100.0 * sm[5] / rays, 100.0 * sm[30] / rays,
		            100.0 * sm[31] / rays, 100.0 * sm[32] / rays, 100.0 * sm[33] / px);
	}
	if (Metal().level > 0) {
		// (hwrt4) The metal reflections: the share of G-buffer pixels whose metal field is set, of them the ones traced
		// (roughness below the cut-off), the TLAS hit rate, the pixels whose own instance was found (self-skip), the share
		// of all pixels the composite changed by > 5 %.
		const double gp = static_cast<double>(std::max<uint64_t>(sm[34], 1));
		std::printf("RT metal: last %.0f s: metal field set on %.1f%% of the G-buffer pixels, rays on %.1f%% (roughness < %.2f), TLAS "
		            "hit %.0f%%, own instance found %.0f%%; changed > 5%% on %.1f%% of the screen; TLAS bound %llu of %llu frames\n",
		            std::chrono::duration<double>(now - g_rt.stat_time).count(), 100.0 * sm[35] / gp, 100.0 * sm[36] / gp,
		            static_cast<double>(Metal().rough_max), sm[36] > 0 ? 100.0 * sm[37] / static_cast<double>(sm[36]) : 0.0,
		            sm[36] > 0 ? 100.0 * sm[38] / static_cast<double>(sm[36]) : 0.0, 100.0 * sm[39] / px,
		            static_cast<unsigned long long>(g_rt.m_tlas_frames), static_cast<unsigned long long>(g_rt.m_frames));
		std::printf("RT metal: darkening: %.1f%% of the changed metal pixels would have been > 30%% darker than the game's (floored "
		            "at 85%%); frames with a burst (> 5%% of them) %llu of %llu\n",
		            g_rt.m_all_px ? 100.0 * g_rt.m_dark_px / static_cast<double>(g_rt.m_all_px) : 0.0,
		            static_cast<unsigned long long>(g_rt.m_dark_frames), static_cast<unsigned long long>(g_rt.m_frames));
		g_rt.m_dark_frames = 0, g_rt.m_dark_px = 0, g_rt.m_all_px = 0;
		g_rt.m_tlas_frames = 0, g_rt.m_frames = 0;
		const auto* mh = reinterpret_cast<const uint32_t*>(g_rt.mapped + HeaderBytes / 4 + MetalHist);
		std::array<uint32_t, 10> d {};
		uint64_t                 total = 0;
		for (int k = 0; k < 10; ++k) d[k] = mh[k] - g_rt.m_hist_last[k], g_rt.m_hist_last[k] = mh[k], total += d[k];
		std::string h;
		for (int k = 0; k < 10; ++k) {
			char b[32];
			std::snprintf(b, sizeof(b), "%s%.1f", k ? " / " : "", total ? 100.0 * d[k] / static_cast<double>(total) : 0.0);
			h += b;
		}
		std::printf("RT metal: roughness on metal pixels (%% per 0.1 bin, 0-0.1 ... 0.9-1): %s (rays below %.2f, fading from %.2f)\n",
		            h.c_str(), static_cast<double>(Metal().rough_max), static_cast<double>(Metal().rough_max * 0.7f));
	}
	std::string passes;
	for (uint32_t st = 0; st < NStages; ++st) {
		if (g_rt.stage_frames[st] == 0) continue;
		char buf[64];
		// (calib: its mean over the frames that ran it; the others run every frame)
		std::snprintf(buf, sizeof(buf), "%s%s %.3f", passes.empty() ? "" : ", ", StageNames[st],
		              g_rt.stage_ms[st] / g_rt.stage_frames[st]);
		passes += buf;
	}
	if (!passes.empty()) std::printf("RT reflections: GPU ms per pass (%s): %s\n", Cfg(level).name, passes.c_str());
	if (RtReflGuideWanted()) {
		// (refl8) The one line on the guide: where the game reflects (its SSR confident) and where we sharpened it.
		const auto        pc = [&](uint32_t k) { return 100.0 * sm[k] / px; };
		const RtReflGuide& gl = g_rt.guide_last;
		char              producer[96];
		if (gl.producer != 0) std::snprintf(producer, sizeof(producer), "CS %016llx", static_cast<unsigned long long>(gl.producer));
		else std::snprintf(producer, sizeof(producer), "not seen (a draw, or not a G-buffer-time compute)");
		std::printf("RT reflections: guide: last %.0f s: the game reflects (its SSR confidence > %.2f) on %.1f%% of the pixels, we "
		            "sharpened %.1f%% (our hit replaced > 5%% of its SSR), water mirror %.1f%%; its mean confidence %.3f, rgb "
		            "without confidence on %.1f%%; guide %llu of %llu frames (%ux%u from CS %016llx slot %d, written by %s; now: %s)\n",
		            std::chrono::duration<double>(now - g_rt.stat_time).count(), g_rt.guide_lo, pc(20), pc(21), pc(23),
		            sm[22] / 1000.0 / px, pc(24), static_cast<unsigned long long>(g_rt.stat_guide_frames),
		            static_cast<unsigned long long>(g_rt.stat_frames), gl.extent.width, gl.extent.height,
		            static_cast<unsigned long long>(gl.cs), gl.slot, producer, RtReflGuideStatus());
	}
	std::fflush(stdout);
	g_rt.stage_ms.fill(0.0);
	g_rt.stage_frames.fill(0);
	g_rt.stat_sum.fill(0);
	g_rt.stat_frames = 0, g_rt.stat_gbuf_frames = 0, g_rt.stat_guide_frames = 0, g_rt.stat_copies0 = copies;
	g_rt.sum_ms = 0.0, g_rt.samples = 0;
	g_rt.stat_time = now;
}

// The calibration's verdict: the normal layout (mean cos vs depth normals >= 0.8, best of layer x layout x view /
// world space x screen y convention; the convention it implies is adopted) and, when its layout leaves a byte free in
// the word, that byte as roughness (a guess: only modulating the wet-surface mask in the shader). Without a normal, the
// floor votes (world up known) decide the y convention. True: a normal identified.
bool Analyze(GraphicContext& graphics, uint32_t packed_count) {
	if (g_rt.mapped == nullptr) return false;
	(void)vmaInvalidateAllocation(graphics.allocator, g_rt.frame_mem, 0, VK_WHOLE_SIZE);
	const int32_t* c = g_rt.mapped + HeaderBytes / 4;
	const double   n = static_cast<double>(c[0]);
	static const char* const layouts[NLayouts] = {"oct 16:16",         "oct 12:12 bits 0..23", "oct 12:12 bits 8..31",
	                                              "oct 11:11 bits 0..21", "xyz 10:10:10",       "oct 16:16 swapped",
	                                              "oct 16:16 snorm",   "spheremap 16:16",      "xyz 11:11:10",
	                                              "oct 8:8 bits 0..15", "oct 8:8 bits 16..31", "hemi-oct 16:16",
	                                              "xyz 10:10:10 bits 2..31"};
	int    best = -1, best_conv = 0;
	double best_score = 0.0, second = 0.0;
	if (n >= 200.0) {
		std::array<double, 4 * NLayouts> per {}; // (the best |score| of each layer x layout: the runner-up's margin)
		for (uint32_t layer = 0; layer < packed_count && layer < 4; ++layer)
			for (uint32_t l = 0; l < NLayouts; ++l)
				for (int space = 0; space < 2; ++space)
					for (int conv = 0; conv < 2; ++conv) {
						const double score = c[CalibNormal + (layer * NLayouts + l) * 4 + space * 2 + conv] / 1000.0 / n;
						per[layer * NLayouts + l] = std::max(per[layer * NLayouts + l], std::abs(score));
						if (std::abs(score) > std::abs(best_score)) {
							best_score = score, best_conv = conv;
							best       = static_cast<int>(layer | (l << 2)) | (space << 6);
						}
					}
		for (uint32_t k = 0; k < per.size(); ++k)
			if (best >= 0 && static_cast<int>(k) != (best & 3) * static_cast<int>(NLayouts) + ((best >> 2) & 15))
				second = std::max(second, per[k]);
	}
	const double floors_cur = c[CalibFloors], floors_mirror = c[CalibFloors + 1];
	std::printf("RT reflections: calibration %u: %.0f samples, %u packed layers, best normal fit %.3f (next layout %.3f), "
	            "floors %.0f (screen y as is) vs %.0f (mirrored)\n",
	            g_rt.calib_attempts + 1, n, packed_count, std::abs(best_score), second, floors_cur, floors_mirror);
	const bool identified = best >= 0 && std::abs(best_score) >= 0.8;
	if (identified) {
		if (best_score < 0.0) best |= 128;
		if (g_rt.dec_normal < 0) g_rt.dec_normal = best;
		if (best_conv == 1 && !g_rt.row0_forced) g_rt.row0_up = !g_rt.row0_up, g_rt.history_valid = false;
		const int layout = (best >> 2) & 15, layer = best & 3;
		std::printf("RT reflections: G-buffer normal = layer %d, %s, %s space%s (mean cos %.3f vs depth normals); screen row 0 "
		            "= NDC y %s\n",
		            layer, layouts[layout], (best & 64) ? "world" : "view", (best & 128) ? ", negated" : "", std::abs(best_score),
		            g_rt.row0_up ? "+1" : "-1");
	} else {
		std::printf("RT reflections: G-buffer normal not identified (best |mean cos| %.3f, %.0f samples): depth normals and "
		            "the wet-surface mask\n",
		            std::abs(best_score), n);
		if (!g_rt.row0_forced && floors_mirror >= 300.0 && floors_mirror > 2.0 * floors_cur) {
			g_rt.row0_up       = !g_rt.row0_up;
			g_rt.history_valid = false;
			std::printf("RT reflections: screen row 0 = NDC y %s (the floor votes)\n", g_rt.row0_up ? "+1" : "-1");
		}
	}
	// The byte fields overall, on puddle-looking pixels and on water (the depth drawn after the G-buffer): a field that
	// differs clearly on water / puddles is the material's (set it with KYTY_RT_REFLECTIONS_ROUGH=<layer>:<shift>:8[g]).
	if (packed_count > 0 && n >= 200.0) {
		const double pud = c[CalibWet], wat = c[CalibWet + 17];
		std::string  line;
		char         buf[96];
		for (uint32_t f = 0; f < 16; ++f) {
			const double all = c[CalibBytes + f * 3] / 1000.0 / n;
			std::snprintf(buf, sizeof(buf), " L%u.b%u %.2f/%s/%s;", f / 4, f % 4, all,
			              pud >= 20 ? std::to_string(c[CalibWet + 1 + f] / 1000.0 / pud).substr(0, 4).c_str() : "-",
			              wat >= 20 ? std::to_string(c[CalibWet + 18 + f] / 1000.0 / wat).substr(0, 4).c_str() : "-");
			line += buf;
		}
		std::printf("RT reflections: G-buffer byte fields, mean all / puddle-looking (%.0f) / water (%.0f):%s\n", pud, wat, line.c_str());
		// (refl6) The histograms of the material candidates (roughness L2.b0, gloss L1.b2, L2.b3, L1.b3), % per 1/16.
		for (const uint32_t f: {8u, 6u, 11u, 7u}) {
			double tot = 0.0;
			for (uint32_t b = 0; b < 16; ++b) tot += c[CalibHist + f * 16 + b];
			std::string h;
			for (uint32_t b = 0; b < 16; ++b) {
				std::snprintf(buf, sizeof(buf), "%s%.0f", b ? " " : "", tot > 0 ? 100.0 * c[CalibHist + f * 16 + b] / tot : 0.0);
				h += buf;
			}
			std::printf("RT reflections: L%u.b%u histogram (%% per 1/16, 0 -> 1): %s\n", f / 4, f % 4, h.c_str());
		}
	}
	// The material field, automatically: the byte (outside the decoded normal's bits) that separates the puddle-looking
	// pixels from all the others the most (>= 1 standard deviation and 0.2 apart). Below the overall mean on puddles:
	// roughness (low = wet / smooth); above: gloss (inverted). Wet below a quarter of the way from the puddles' mean to
	// the overall mean, dry from halfway (in roughness units). The 10-24 log: L2.b0 0.40 overall / 0.01 on puddles,
	// L1.b2 0.60 / 0.96.
	if (!g_rt.rough_trusted && g_rt.dec_rough < 0 && packed_count > 0 && n >= 200.0 && c[CalibWet] >= 100 && g_rt.dec_normal >= 0) {
		const int    nlayer = g_rt.dec_normal & 3, nlayout = (g_rt.dec_normal >> 2) & 15;
		const auto   in_normal = [&](int layer, int b) {
			if (layer != nlayer) return false;
			switch (nlayout) {
				case 1: case 3: return b <= 2;
				case 2: return b >= 1;
				case 9: return b <= 1;
				case 10: return b >= 2;
				default: return true;
			}
		};
		const double pud = c[CalibWet];
		int          pick = -1;
		double       best_sep = 0.0, pick_all = 0.0, pick_pud = 0.0;
		for (int f = 0; f < 16; ++f) {
			if (in_normal(f / 4, f % 4)) continue;
			const double all = c[CalibBytes + f * 3] / 1000.0 / n, sq = c[CalibBytes + f * 3 + 1] / 1000.0 / n;
			const double sd  = std::sqrt(std::max(sq - all * all, 0.0));
			const double mp  = c[CalibWet + 1 + f] / 1000.0 / pud;
			if (sd < 0.05) continue;
			const double sep = std::abs(mp - all) / sd;
			if (sep >= 1.0 && std::abs(mp - all) >= 0.2 && sep > best_sep) best_sep = sep, pick = f, pick_all = all, pick_pud = mp;
		}
		if (pick >= 0) {
			const bool   gloss = pick_pud > pick_all;
			const double rp = gloss ? 1.0 - pick_pud : pick_pud, ra = gloss ? 1.0 - pick_all : pick_all;
			g_rt.dec_rough = (pick / 4) | (((pick % 4) * 8) << 2) | (8 << 7) | (gloss ? 4096 : 0);
			if (!g_rt.mat_forced) {
				g_rt.mat_lo = static_cast<float>(rp + (ra - rp) * 0.15); // (refl3: tighter, was 0.25 / 0.5)
				g_rt.mat_hi = static_cast<float>(rp + (ra - rp) * 0.35);
			}
			g_rt.mat_auto = true;
			// (the share of all pixels it marks wet, from the field's histogram)
			double wet = 0.0, total = 0.0;
			for (int b = 0; b < 16; ++b) {
				const double v  = (b + 0.5) / 16.0, r = gloss ? 1.0 - v : v;
				const double hc = c[CalibHist + pick * 16 + b];
				total += hc;
				if (r < g_rt.mat_hi) wet += hc * (r < g_rt.mat_lo ? 1.0 : 0.5);
			}
			std::printf("RT reflections: material field = layer %d bits %d..%d%s (mean %.2f overall, %.2f on puddle-looking pixels, "
			            "%.1f sd apart): wet below %.2f, dry from %.2f (roughness units), about %.1f%% of the samples wet "
			            "(KYTY_RT_REFLECTIONS_ROUGH=%d:%d:8%s and KYTY_RT_REFLECTIONS_WET=lo:hi override)\n",
			            pick / 4, (pick % 4) * 8, (pick % 4) * 8 + 7, gloss ? " (gloss: inverted)" : " (roughness)", pick_all, pick_pud,
			            best_sep, g_rt.mat_lo, g_rt.mat_hi, total > 0 ? 100.0 * wet / total : 0.0, pick / 4, (pick % 4) * 8,
			            gloss ? "g" : "");
		} else {
			std::printf("RT reflections: no material field separates the puddle-looking pixels (>= 1 sd): water and the faint "
			            "puddle guess\n");
		}
	} else if (!g_rt.rough_trusted && g_rt.dec_rough < 0 && packed_count > 0) {
		std::printf("RT reflections: material field not looked for (%s): water and the faint puddle guess\n",
		            g_rt.dec_normal < 0 ? "no G-buffer normal" : "too few puddle-looking samples");
	}
	std::fflush(stdout);
	return identified;
}

// (refl9) The decoded normal's bits (the survey never offers them as the metal field).
bool InNormalBits(int layer, int bit) {
	const int n = g_rt.dec_normal;
	if (n < 0 || (n & 3) != layer) return false;
	static constexpr int lo[NLayouts] = {0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16, 0, 2};
	static constexpr int hi[NLayouts] = {31, 23, 31, 21, 29, 31, 31, 31, 31, 15, 31, 31, 31};
	const int l = (n >> 2) & 15;
	return l < static_cast<int>(NLayouts) && bit >= lo[l] && bit <= hi[l];
}
// Two field codes share bits (same layer, overlapping ranges).
bool FieldsOverlap(int32_t a, int32_t b) {
	if (a < 0 || b < 0 || (a & 3) != (b & 3)) return false;
	const int as = (a >> 2) & 31, ae = as + std::max((a >> 7) & 31, 1), bs = (b >> 2) & 31, be = bs + std::max((b >> 7) & 31, 1);
	return as < be && bs < ae;
}

// (refl9) The metal field from the survey (mode 8): the game's G-buffer pack shader is not available here, so the field
// is found the way the normal and the roughness were, statistically, against a ground truth the frame has: characters
// (moving pixels: armour, swords, shields; also skin and cloth) against the static world (mostly stone and wood). A
// metal field is near-binary (a metalness texture: >= 75 % of the samples in the end bins), set on a minority of the
// static world (<= 30 %) and on clearly more of the characters (>= 12 %, >= 2x the static share + 5 points), but not on
// nearly all of them (<= 85 %: that is an object / character flag, not a material). Candidates: the 16 bytes as unorm8
// outside the decoded normal (single bits only logged); the polarity whichever side the characters favour.
// Logged with the top candidates so a log is enough to check or override it (KYTY_RT_REFLECTIONS_METAL_FIELD).
void SurveyDecide(bool force) {
	if (g_rt.mapped == nullptr || g_rt.survey_done) return;
	const int32_t* c  = g_rt.mapped + HeaderBytes / 4 + SurveyIndex;
	const double   ns = c[0], nm = c[1];
	const bool     enough = (nm >= 3000.0 && ns >= 30000.0) || (force && nm >= 1000.0 && ns >= 10000.0);
	if (!enough) {
		if (g_rt.frame >= g_rt.survey_next_log) {
			g_rt.survey_next_log = g_rt.frame + 3600;
			std::printf("RT reflections: metal survey: %.0f static / %.0f moving (character) G-buffer samples so far (deciding at "
			            "30000 / 3000; characters count while they move against the camera)\n",
			            ns, nm);
			std::fflush(stdout);
		}
		return;
	}
	g_rt.survey_done = true;
	struct Cand {
		int32_t code;
		double  ms, mm, bin, score;
	};
	std::vector<Cand> all;
	const auto consider = [&](int32_t code, double hs, double hm, double bin) {
		// (hs / hm: the shares of the static / moving samples on the "high" side)
		const bool   inv = hm < hs;
		const double ms = inv ? 1.0 - hs : hs, mm = inv ? 1.0 - hm : hm;
		double       score = mm - ms;
		const bool   ok = bin >= 0.75 && ms <= 0.30 && mm >= 0.12 && mm <= 0.85 && mm >= 2.0 * ms + 0.05;
		if (!ok) score = -1.0 + score * 0.5; // (kept for the log, ranked below the passing ones)
		all.push_back({code | (inv ? 4096 : 0), ms, mm, bin, score});
	};
	for (int f = 0; f < 16; ++f) {
		const int layer = f / 4, b = f % 4;
		if (InNormalBits(layer, b * 8) || InNormalBits(layer, b * 8 + 7)) continue;
		double s_tot = 0, m_tot = 0, s_hi = 0, m_hi = 0, ends = 0;
		for (int k = 0; k < 16; ++k) {
			const double hs = c[SurveyHist + f * 16 + k], hm = c[SurveyHist + 256 + f * 16 + k];
			s_tot += hs, m_tot += hm;
			if (k >= 8) s_hi += hs, m_hi += hm;
			if (k <= 1 || k >= 14) ends += hs + hm;
		}
		if (s_tot < 1 || m_tot < 1) continue;
		consider(layer | ((b * 8) << 2) | (8 << 7), s_hi / s_tot, m_hi / m_tot, ends / (s_tot + m_tot));
	}
	// (single bits are logged, not picked: the top bit of any continuous value (an albedo channel) is "binary" too and
	// can differ on characters; a 1-bit flag found in the log is set with KYTY_RT_REFLECTIONS_METAL_FIELD=<l>:<bit>:1)
	std::vector<Cand> bits;
	for (int layer = 0; layer < 4; ++layer)
		for (int bit = 0; bit < 32; ++bit) {
			if (InNormalBits(layer, bit)) continue;
			const double ps = c[SurveyBits + layer * 32 + bit] / std::max(ns, 1.0), pm = c[SurveyBits + 128 + layer * 32 + bit] / std::max(nm, 1.0);
			if (ps < 0.005 && pm < 0.005) continue;
			const bool   inv = pm < ps;
			const double a = inv ? 1.0 - ps : ps, b = inv ? 1.0 - pm : pm;
			bits.push_back({layer | (bit << 2) | (1 << 7) | (inv ? 4096 : 0), a, b, 1.0, b - a});
		}
	std::sort(bits.begin(), bits.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });
	std::sort(all.begin(), all.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });
	const auto name = [](int32_t f) {
		char buf[48];
		const int sh = (f >> 2) & 31, bits = (f >> 7) & 31;
		if (bits == 1) std::snprintf(buf, sizeof(buf), "L%d bit %d%s", f & 3, sh, (f & 4096) ? " clear" : "");
		else std::snprintf(buf, sizeof(buf), "L%d bits %d..%d%s", f & 3, sh, sh + bits - 1, (f & 4096) ? " inverted" : "");
		return std::string(buf);
	};
	std::string line;
	for (size_t i = 0; i < all.size() && i < 6; ++i) {
		char buf[128];
		std::snprintf(buf, sizeof(buf), "%s %s static %.1f%% / characters %.1f%% (binary %.0f%%)%s", i ? ";" : "", name(all[i].code).c_str(),
		              100.0 * all[i].ms, 100.0 * all[i].mm, 100.0 * all[i].bin, all[i].score >= 0.0 ? "" : " rejected");
		line += buf;
	}
	std::printf("RT reflections: metal survey: %.0f static / %.0f moving samples; best candidates:%s\n", ns, nm, line.c_str());
	const Cand* pick = nullptr;
	for (const auto& a: all)
		if (a.score >= 0.0) {
			pick = &a;
			break;
		}
	{
		std::string bl;
		for (size_t i = 0; i < bits.size() && i < 6; ++i) {
			char buf[96];
			std::snprintf(buf, sizeof(buf), "%s %s static %.1f%% / characters %.1f%%", i ? ";" : "", name(bits[i].code).c_str(),
			              100.0 * bits[i].ms, 100.0 * bits[i].mm);
			bl += buf;
		}
		std::printf("RT reflections: metal survey: single bits most set on characters (not picked):%s\n", bl.c_str());
	}
	if (pick == nullptr) {
		std::printf("RT reflections: metal survey: no field separates the characters' metal (the glossy path alone: metals "
		            "reflect by their roughness, at the glossy strength; KYTY_RT_REFLECTIONS_METAL_FIELD=<layer>:<shift>:<bits>[g] "
		            "sets one)\n");
	} else {
		g_rt.dec_metal = pick->code;
		std::printf("RT reflections: metal field = %s (static %.1f%%, characters %.1f%%; KYTY_RT_REFLECTIONS_METAL_FIELD=%d:%d:%d%s)\n",
		            name(pick->code).c_str(), 100.0 * pick->ms, 100.0 * pick->mm, pick->code & 3, (pick->code >> 2) & 31,
		            (pick->code >> 7) & 31, (pick->code & 4096) ? "g" : "");
		// (the roughness field was a guess too: when it is the metal field, a metal read as rough and never reflected; the
		// gloss field alone is the roughness then)
		if (!g_rt.rough_trusted && FieldsOverlap(g_rt.dec_rough, pick->code)) {
			std::printf("RT reflections: the roughness field %s is the metal field: roughness from the gloss field alone\n",
			            name(g_rt.dec_rough).c_str());
			g_rt.dec_rough = g_rt.dec_gloss >= 0 ? g_rt.dec_gloss : g_rt.dec_rough;
			if (g_rt.dec_rough == g_rt.dec_gloss) g_rt.dec_gloss = -1;
		}
	}
	std::fflush(stdout);
}

} // namespace

int RtReflectionsLevel() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_RT_REFLECTIONS");
		if (value == nullptr && HwReflWanted()) return 3; // (KYTY_HWRT_REFL alone: high)
		return value != nullptr ? std::clamp(std::atoi(value), 0, MaxLevel) : 0; // 0 off, 1 low, 2 medium, 3 high, 4 ultra, 5 extreme
	}();
	return level;
}
bool RtReflectionsWanted() { return RtReflectionsLevel() > 0; }
// (local-pass-cache.h) The pipelines a world's first frame would make on the flip path, made beside the title screen
// with the same guards as RtReflectionsApply's: the base pass when on, the ray query pass with KYTY_HWRT_REFL and a
// scene, its material variant with KYTY_HWRT_REFL_MATERIAL (KYTY_HWRT_METAL reuses these).
void RtReflectionsPrewarm(GraphicContext& graphics) {
	if (RtReflectionsLevel() == 0) return;
	if (!EnsurePipeline(graphics)) return;
	if (HwReflWanted() && HwScene::Active() && EnsureHwPipeline(graphics) && HwScene::MaterialActive()) (void)EnsureHwMaterialPipeline(graphics);
}
// (refl8) The game's SSR as the guide: on with the reflections unless KYTY_RT_REFLECTIONS_GUIDE=0.
bool RtReflGuideWanted() {
	static const bool wanted = [] {
		const char* value = std::getenv("KYTY_RT_REFLECTIONS_GUIDE");
		return RtReflectionsLevel() > 0 && !(value != nullptr && value[0] == '0');
	}();
	return wanted;
}

bool RtReflectionsApply(vk::CommandBuffer command, UpscaleInputs& inputs) {
	const int level = RtReflectionsLevel();
	if (level == 0) return false; // off: nothing created or recorded
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr) return false;
	if (VideoOut::MoviePlaying()) {
		g_rt.history_valid = false;
		g_rt.mhist_valid   = false;
		return false;
	}
	const auto& color = inputs.color;
	if (color.view == nullptr || inputs.depth.view == nullptr || inputs.motion.view == nullptr || color.extent.width < 16 ||
	    color.extent.height < 16 || inputs.depth.extent != color.extent)
		return false;
	const auto& m = inputs.view_to_clip;
	if (!(m[0] != 0.0f && m[5] != 0.0f && (m[11] != 0.0f || m[15] != 0.0f)) ||
	    !std::all_of(m.begin(), m.end(), [](float v) { return std::isfinite(v); })) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("RT reflections: skipped (no camera projection from the upscaler)\n");
			std::fflush(stdout);
		}
		return false;
	}
	if (g_rt.images_failed || !EnsurePipeline(*graphics) || !EnsureImages(*graphics, color.extent, level)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("RT reflections: off (could not create its pipeline or images)\n");
			std::fflush(stdout);
		}
		return false;
	}
	FreeRetired(*graphics);
	ReadTiming(*graphics);

	// The G-buffer copy (this flip or the last), the decode from the environment, the calibration.
	RtReflGbuffer gbuf;
	const bool    have_gbuf    = RtReflGetGbuffer(gbuf) && gbuf.extent == color.extent;
	const uint32_t packed_count = have_gbuf ? 4u : 0u;
	// (refl8) The game's own SSR result (the guide), taken in this flip or the last.
	const bool  guide_mode = RtReflGuideWanted();
	RtReflGuide guide;
	const bool  have_guide = guide_mode && RtReflGetGuide(guide) && guide.view != nullptr;
	if (have_guide) g_rt.guide_last = guide;
	g_rt.have_guide = have_guide;
	ReadStats(*graphics, have_gbuf, level);
	{
		static bool said = false;
		if (!said) {
			said = true;
			const int32_t n = NormalEnv("KYTY_RT_REFLECTIONS_NORMAL"), r = FieldEnv("KYTY_RT_REFLECTIONS_ROUGH");
			if (n >= 0) g_rt.dec_normal = n;
			if (r >= 0) g_rt.dec_rough = r, g_rt.rough_trusted = true;
			if (const char* w = std::getenv("KYTY_RT_REFLECTIONS_WET"); w != nullptr) {
				float lo = 0.0f, hi = 0.0f;
				if (std::sscanf(w, "%f:%f", &lo, &hi) == 2 && lo >= 0.0f && hi > lo && hi <= 1.0f)
					g_rt.mat_lo = lo, g_rt.mat_hi = hi, g_rt.mat_forced = true;
			}
				// (refl6) The roughness range: a mirror below lo, nothing from hi (KYTY_RT_REFLECTIONS_RANGE=lo:hi).
			if (const char* w = std::getenv("KYTY_RT_REFLECTIONS_RANGE"); w != nullptr) {
				float lo = 0.0f, hi = 0.0f;
				if (std::sscanf(w, "%f:%f", &lo, &hi) == 2 && lo >= 0.0f && hi > lo && hi <= 1.0f)
					g_rt.mat_lo = lo, g_rt.mat_hi = hi, g_rt.mat_forced = true;
			}
			// (refl6) The material roughness by default: L2.b0 (roughness) and L1.b2 (gloss), the rougher of the two.
			if (r < 0) g_rt.dec_rough = 2 | (0 << 2) | (8 << 7);
			g_rt.dec_gloss = 1 | (16 << 2) | (8 << 7) | 4096;
			if (const char* gl = std::getenv("KYTY_RT_REFLECTIONS_GLOSS"); gl != nullptr && *gl != '\0') {
				const int32_t gc = FieldEnv("KYTY_RT_REFLECTIONS_GLOSS");
				g_rt.dec_gloss   = gc >= 0 ? gc : -1; // ("0" or anything malformed: no gloss field)
			}
			// (refl9) KYTY_RT_REFLECTIONS_METAL is the metal path's strength (0..1, 0.35); the older field syntax
			// (<layer>:<shift>:<bits>[g]) is still read as the field. KYTY_RT_REFLECTIONS_METAL_FIELD: the field, "0" none,
			// unset / "auto": the survey finds it. KYTY_RT_REFLECTIONS_GLOSSY: the glossy dielectrics' strength (0..1, 0.25).
			g_rt.survey_on = true;
			if (const char* mt = std::getenv("KYTY_RT_REFLECTIONS_METAL"); mt != nullptr && *mt != '\0') {
				if (std::strchr(mt, ':') != nullptr) {
					g_rt.dec_metal = FieldEnv("KYTY_RT_REFLECTIONS_METAL");
					g_rt.survey_on = false;
				} else {
					g_rt.metal_strength = EnvFloat("KYTY_RT_REFLECTIONS_METAL", 0.35f, 0.0f, 1.0f);
				}
			}
			if (const char* mf = std::getenv("KYTY_RT_REFLECTIONS_METAL_FIELD"); mf != nullptr && *mf != '\0' && std::strcmp(mf, "auto") != 0) {
				g_rt.dec_metal = FieldEnv("KYTY_RT_REFLECTIONS_METAL_FIELD"); // ("0" or malformed: none)
				g_rt.survey_on = false;
			}
			g_rt.glossy_strength = EnvFloat("KYTY_RT_REFLECTIONS_GLOSSY", 0.25f, 0.0f, 1.0f);
			g_rt.metal_tint      = EnvFloat("KYTY_RT_REFLECTIONS_METAL_TINT", 0.7f, 0.0f, 1.0f);
			g_rt.min_area        = EnvFloat("KYTY_RT_REFLECTIONS_MIN_AREA", 10.0f, 0.0f, 64.0f);
			g_rt.pool_boost      = EnvFloat("KYTY_RT_REFLECTIONS_POOL_BOOST", 0.6f, 0.0f, 2.0f);
			if (g_rt.dec_metal >= 0 && !g_rt.rough_trusted && FieldsOverlap(g_rt.dec_rough, g_rt.dec_metal) && g_rt.dec_gloss >= 0)
				g_rt.dec_rough = g_rt.dec_gloss, g_rt.dec_gloss = -1; // (as SurveyDecide: a metal field is not a roughness)
			std::printf("RT reflections: metal strength %g, glossy strength %g, metal tint %g, minimum area radius %g px (720p), pool "
			            "boost %g, metal field %s\n", g_rt.metal_strength, g_rt.glossy_strength, g_rt.metal_tint, g_rt.min_area, g_rt.pool_boost, g_rt.survey_on ? "by the survey (auto)" : g_rt.dec_metal >= 0 ? "from the environment" : "none");
			if (const char* mv = std::getenv("KYTY_RT_REFLECTIONS_MOVING"); mv != nullptr && *mv == '0') g_rt.moving_on = false;
			g_rt.mat_scale = EnvFloat("KYTY_RT_REFLECTIONS_MATERIAL", 1.0f, 0.0f, 1.0f);
			// (refl8) The guide's settings.
			if (const char* w = std::getenv("KYTY_RT_REFLECTIONS_GUIDE_RANGE"); w != nullptr) {
				float lo = 0.0f, hi = 0.0f;
				if (std::sscanf(w, "%f:%f", &lo, &hi) == 2 && lo >= 0.0f && hi > lo && hi <= 1.0f) g_rt.guide_lo = lo, g_rt.guide_hi = hi;
			}
			if (const char* pm = std::getenv("KYTY_RT_REFLECTIONS_GUIDE_PREMUL"); pm != nullptr && *pm == '1') g_rt.guide_premul = true;
			g_rt.guide_strength = EnvFloat("KYTY_RT_REFLECTIONS_GUIDE_STRENGTH", 1.0f, 0.0f, 4.0f);
			if (const char* y = std::getenv("KYTY_RT_REFLECTIONS_FLIPY"); y != nullptr && *y != '\0')
				g_rt.row0_up = *y != '0', g_rt.row0_forced = true;
			// (a normal given and the convention forced: nothing left to calibrate)
			if (n >= 0 && g_rt.row0_forced) g_rt.calib_done = true;
			if (n >= 0 || r >= 0 || g_rt.row0_forced) {
				std::printf("RT reflections: decode from the environment (normal %d, roughness %d, screen y %s)\n", n, r,
				            g_rt.row0_forced ? (g_rt.row0_up ? "row 0 = NDC +1" : "row 0 = NDC -1") : "calibrated");
				std::fflush(stdout);
			}
		}
	}
	// The calibration. G-buffer attempts (the ones that count, up to 3, ~30 s apart until a normal fits): once the copy
	// has been there 120 frames running, so a scene and not the first loading frames; its runs wait while the copy is
	// missing. Before any copy, after 900 frames: one floor-vote run (packed layers 0, the y convention only), which
	// does not count (the first version spent its attempt there when the copy came late: "0 packed layers").
	g_rt.gb_streak = have_gbuf ? g_rt.gb_streak + 1 : 0;
	if (!g_rt.calib_done && !g_rt.calib_active && g_rt.frame >= g_rt.calib_next) {
		if (g_rt.gb_streak >= 120) {
			g_rt.calib_active = true, g_rt.calib_clear = true;
			g_rt.calib_runs = 0, g_rt.calib_wait = 0, g_rt.calib_packed = 4;
		} else if (!g_rt.floors_done && g_rt.gb_streak == 0 && g_rt.frame >= 900 && g_rt.calib_attempts == 0) {
			g_rt.floors_done  = true;
			g_rt.calib_active = true, g_rt.calib_clear = true;
			g_rt.calib_runs = 0, g_rt.calib_wait = 0, g_rt.calib_packed = 0;
		}
	}
	const bool calibrate = g_rt.calib_active && g_rt.calib_runs < CalibRuns && (g_rt.calib_tick++ & 3u) == 0 &&
	                       (g_rt.calib_packed == 0 || have_gbuf);
	if (g_rt.calib_active && g_rt.calib_runs >= CalibRuns && ++g_rt.calib_wait > 8) {
		g_rt.calib_active     = false;
		const bool identified = Analyze(*graphics, g_rt.calib_packed);
		if (g_rt.calib_packed > 0) {
			++g_rt.calib_attempts;
			// (done once the normal and a material field are known; else again ~30 s later, up to 6 attempts: the
			// material needs puddle-looking pixels in view)
			const bool material = g_rt.dec_rough >= 0 || g_rt.rough_trusted;
			if ((g_rt.dec_normal >= 0 && material) || g_rt.calib_attempts >= 6) g_rt.calib_done = true;
			else g_rt.calib_next = g_rt.frame + 1800;
			(void)identified;
		}
	}
	const int32_t dec_normal = have_gbuf ? g_rt.dec_normal : -1;
	// (refl6: the roughness field whenever the G-buffer's normal is known; a field sharing the normal's layer is not used
	// unless given in the environment)
	const auto    field_ok   = [&](int32_t f, bool trusted) { return f >= 0 && (trusted || dec_normal < 0 || (f & 3) != (dec_normal & 3)); };
	const int32_t dec_rough  = have_gbuf && dec_normal >= 0 && field_ok(g_rt.dec_rough, g_rt.rough_trusted) ? g_rt.dec_rough : -1;
	const int32_t dec_gloss  = dec_rough >= 0 && field_ok(g_rt.dec_gloss, false) ? g_rt.dec_gloss : -1;
	const int32_t dec_metal  = dec_rough >= 0 && field_ok(g_rt.dec_metal, true) ? g_rt.dec_metal : -1;
	const bool    have_mat   = dec_rough >= 0;
	const bool    have_water = have_gbuf && gbuf.depth != nullptr;
	g_rt.have_water          = have_water;

	RtReflCamera camera;
	RtReflGetCamera(m, camera);
	const bool use_normal = dec_normal >= 0 && ((dec_normal & 64) == 0 || camera.rows);
	// (refl9) The metal survey: every 16th frame while undecided, with the G-buffer's normal (its surface test) and the
	// camera's matrices (the moving test); decided once enough characters were seen (SurveyDecide).
	const bool survey = g_rt.survey_on && !g_rt.survey_done && have_gbuf && use_normal && camera.prev && g_rt.moving_on &&
	                    !g_rt.moving_auto_off && (g_rt.frame & 15u) == 7u;
	if (g_rt.survey_on && !g_rt.survey_done) SurveyDecide(g_rt.survey_runs > 1100); // (~5 min of surveyed frames: the minimum)

	// (refl6: per level by default, so the medium level shows: low 1.3, medium 1.6, high 1.8, ultra 2.0)
	static constexpr float DefStrength[MaxLevel + 1] = {0.0f, 1.3f, 1.6f, 1.8f, 2.0f, 2.0f};
	const float strength = EnvFloat("KYTY_RT_REFLECTIONS_STRENGTH", DefStrength[std::clamp(level, 0, MaxLevel)], 0.0f, 3.0f);
	const float min_refl = EnvFloat("KYTY_RT_REFLECTIONS_MIN", 0.4f, 0.0f, 0.8f);
	const float guess    = EnvFloat("KYTY_RT_REFLECTIONS_GUESS", 0.0f, 0.0f, 1.0f);
	const auto& cfg      = Cfg(level);
	// (refl10) The march's depth: KYTY_RT_REFLECTIONS_ITERS (hi-z iterations, 8..256) and KYTY_RT_REFLECTIONS_CAP (the
	// longest ray on screen x the render height, 0.1..4) override the level's.
	const int   iters    = static_cast<int>(EnvFloat("KYTY_RT_REFLECTIONS_ITERS", static_cast<float>(cfg.iters), 8.0f, 256.0f));
	const float cutoff   = cfg.cutoff;
	// (hwrtq) KYTY_HWRT_REFL_QUALITY=1..4 (Bajo / Medio / Alto / Ultra; only with KYTY_HWRT_REFL=1; unset: 3, today's):
	// the ray length (screen march + the TLAS miss ray) x 0.5 / 0.75 / 1 / 1.5, a-trous passes -1 / 0 / 0 / 0, rays per
	// glossy pixel 1 / 1 / 1 / 2 (unless KYTY_RT_REFLECTIONS_RPP is set).
	const int   hwq      = HwReflQuality();
	const float len_q    = hwq == 1 ? 0.5f : hwq == 2 ? 0.75f : hwq == 4 ? 1.5f : 1.0f;
	const int   passes   = hwq == 1 ? std::max(1, cfg.passes - 1) : cfg.passes;
	const float cap_px   = EnvFloat("KYTY_RT_REFLECTIONS_CAP", cfg.cap, 0.1f, 4.0f) * static_cast<float>(color.extent.height);
	// (refl10) Precision switches (1 on, 0 off; defaults per level): sub-pixel hit refinement, bicubic history and a fresh
	// mirror every frame at the full-resolution trace (ultra and extreme), the virtual-image reprojection for pixels
	// without their own hit (every level), floors' plane normal for the 8-bit G-buffer one (degrees, 2; 0 off), rays per pixel
	// on glossy surfaces (extreme 2, else 1).
	const auto  env_on   = [](const char* name, bool fallback) {
		const char* v = std::getenv(name);
		return v == nullptr || *v == '\0' ? fallback : *v != '0';
	};
	const bool  refine   = env_on("KYTY_RT_REFLECTIONS_REFINE", level >= 4);
	const bool  bicubic  = env_on("KYTY_RT_REFLECTIONS_BICUBIC", level >= 4);
	const bool  fresh    = env_on("KYTY_RT_REFLECTIONS_FRESH_MIRROR", level >= 4);
	const bool  vfill    = env_on("KYTY_RT_REFLECTIONS_VIRTUAL", true);
	const float snap_deg = EnvFloat("KYTY_RT_REFLECTIONS_NORMAL_SNAP", 2.0f, 0.0f, 10.0f);
	const int   rpp      = static_cast<int>(EnvFloat("KYTY_RT_REFLECTIONS_RPP", level >= 5 || hwq >= 4 ? 2.0f : 1.0f, 1.0f, 4.0f));
	{
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("RT reflections: on (level %d %s: trace %ux%u of %ux%u%s, hi-z %u mips, %d iterations, rays <= %.0f px, %d "
			            "a-trous passes, strength %g, water minimum reflectance %g, roughness: a mirror below %.2f, nothing from %.2f, G-buffer %s, "
			            "exposure %s, camera %s, %s; refl10: %d ray(s) per glossy pixel, hit refinement %s, bicubic history %s, "
			            "fresh mirrors %s, virtual-image fill %s, floor normal snap %g deg)\n",
			            level, cfg.name, g_rt.trace.width, g_rt.trace.height, color.extent.width, color.extent.height,
			            cfg.checker ? " checkerboard" : "", g_rt.hiz.levels, iters, cap_px, passes, strength, min_refl, g_rt.mat_lo, g_rt.mat_hi, have_gbuf ? "copy found (calibrating)" : "not found yet (depth normals, heuristic)",
			            inputs.exposure_buffer != nullptr ? "the game's" : "1.0", camera.prev ? "matrices" : "motion vectors only",
			            guide_mode ? (have_guide ? "guided by the game's SSR (found)" : "guided by the game's SSR (not found yet: the material mask until it is)")
			                       : "material mask (KYTY_RT_REFLECTIONS_GUIDE=0)",
			            rpp, refine ? "on" : "off", bicubic ? "on" : "off", fresh ? "on" : "off", vfill ? "on" : "off",
			            static_cast<double>(snap_deg));
			std::fflush(stdout);
		}
	}

	RtPush push {};
	push.proj[0] = m[0], push.proj[1] = m[5], push.proj[2] = m[8], push.proj[3] = m[9];
	push.depth_map[0] = m[10], push.depth_map[1] = m[14], push.depth_map[2] = m[11], push.depth_map[3] = m[15];
	push.size[0] = static_cast<float>(color.extent.width), push.size[1] = static_cast<float>(color.extent.height);
	push.size[2] = static_cast<float>(g_rt.trace.width), push.size[3] = static_cast<float>(g_rt.trace.height);
	push.params[0] = strength, push.params[1] = EnvFloat("KYTY_RT_REFLECTIONS_THICKNESS", 0.3f, 0.01f, 4.0f);
	push.params[2] = cfg.max_len * len_q;
	push.motion[0] = inputs.motion_scale_x, push.motion[1] = inputs.motion_scale_y;
	push.motion[2] = cutoff, push.motion[3] = cfg.alpha;
	push.misc[1] = static_cast<int32_t>(g_rt.frame & 0x7fffffff);
	push.misc[2] = level;
	push.misc[3] = (inputs.reset ? 1 : 0) | (g_rt.history_valid ? 2 : 0) | (camera.prev ? 4 : 0) | (camera.up_valid ? 8 : 0) |
	               (g_rt.row0_up ? 16 : 0) | (cfg.checker ? 32 : 0) | (have_water ? 64 : 0) |
	               (g_rt.moving_on && !g_rt.moving_auto_off ? 128 : 0) | (have_guide ? 256 : 0) | (guide_mode && have_guide ? 512 : 0) | (refine ? 1024 : 0) | (bicubic ? 2048 : 0) |
	               (fresh ? 4096 : 0) | (vfill ? 8192 : 0);
	// (combo19: guided only while the guide is there; without it the material mask, refl6, instead of water only: the
	// guide never found in play was the whole reason nothing changed)
	push.dec[0] = use_normal ? dec_normal : -1, push.dec[1] = use_normal ? dec_rough : -1;
	push.dec[2] = static_cast<int32_t>(packed_count), push.dec[3] = 0;
	const bool game_exposure = inputs.exposure_buffer != nullptr && inputs.exposure_range >= sizeof(float);
	push.misc2[0] = game_exposure ? static_cast<int32_t>(inputs.exposure_index) : static_cast<int32_t>(ExposureIndex);
	push.misc2[1] = static_cast<int32_t>(g_rt.hiz.levels), push.misc2[2] = iters;
	push.misc2[3] = use_normal && dec_metal >= 0 ? dec_metal + 1 : 0; // (the metal field + 1; 0: none)

	const bool timed = g_rt.pool != nullptr && g_rt.written < g_rt.read + 8; // (a ring of 8 frames)
	const auto entry = static_cast<uint32_t>(g_rt.written % 8);
	uint32_t   stamp_n = 0;
	const auto stamp   = [&](uint8_t stage) {
		if (!timed || stamp_n >= 16) return;
		g_rt.stage_of[entry][stamp_n] = stage;
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g_rt.pool, entry * 16 + stamp_n);
		++stamp_n;
	};
	if (timed) command.resetQueryPool(g_rt.pool, entry * 16, 16);
	stamp(0xff); // (the start)
	// (What wrote the game's images, and our last frame's reads of the buffer and images.)
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite);
	PrepareImages(command);
	if (!g_rt.frame_cleared) {
		g_rt.frame_cleared = true;
		command.fillBuffer(vk::Buffer(g_rt.frame_buf), 0, VK_WHOLE_SIZE, 0);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer,
		        vk::AccessFlagBits2::eTransferWrite);
	} else if (g_rt.calib_clear) {
		// (a new calibration attempt: its sums from zero, the stats counters kept)
		command.fillBuffer(vk::Buffer(g_rt.frame_buf), HeaderBytes, StatsIndex * 4, 0);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer,
		        vk::AccessFlagBits2::eTransferWrite);
	}
	g_rt.calib_clear = false;
	{
		std::array<float, HeaderBytes / 4> header {};
		std::copy(camera.view_rows.begin(), camera.view_rows.end(), header.begin());
		std::copy(camera.clip_to_prev.begin(), camera.clip_to_prev.end(), header.begin() + 12);
		header[28] = camera.up[0], header[29] = camera.up[1], header[30] = camera.up[2], header[31] = camera.up_valid ? 1.0f : 0.0f;
		header[32] = 1.0f;     // (ExposureIndex: the exposure when the game gives none)
		header[33] = min_refl; // (fparams.y)
		header[34] = cap_px;   // (fparams.z)
		header[35] = have_mat ? 0.0f : guess; // (fparams.w: the puddle guess's scale; off with a material field)
		header[36] = g_rt.mat_lo, header[37] = g_rt.mat_hi, header[38] = g_rt.mat_scale; // (mparams)
		header[39] = 0.0f;
		header[40] = g_rt.guide_lo, header[41] = g_rt.guide_hi, header[42] = g_rt.guide_premul ? 1.0f : 0.0f; // (gparams)
		header[43] = g_rt.guide_strength;
		// (refl9, xparams) the metal / glossy paths; the albedo layer L0 (its bits 0..23) unless the normal lives there
		header[44] = dec_metal >= 0 ? g_rt.metal_strength : 0.0f, header[45] = g_rt.glossy_strength, header[46] = g_rt.metal_tint;
		header[47] = (packed_count > 0 && (dec_normal < 0 || (dec_normal & 3) != 0)) ? 0.0f : -1.0f;
		// (refl9b, yparams) the minimum reflective area's radius in render pixels (KYTY_RT_REFLECTIONS_MIN_AREA, given at a
		// 720-line render, scaled to this one) and the large pools' boost
		header[48] = g_rt.min_area * static_cast<float>(color.extent.height) / 720.0f, header[49] = g_rt.pool_boost;
		header[50] = static_cast<float>(rpp), header[51] = snap_deg; // (refl10, yparams.z / w)
		command.updateBuffer(vk::Buffer(g_rt.frame_buf), 0, HeaderBytes, header.data());
	}
	Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	// (hwrt2) the ray query pipeline when the frame has a scene TLAS (KYTY_HWRT_REFL)
	const bool hw = HwReflWanted() && inputs.hwrt_tlas != 0 && inputs.hwrt_info && EnsureHwPipeline(*graphics);
	if (hw) ++g_rt.hw_frames;
	else ++g_rt.sw_frames;
	if (HwReflWanted() && ((g_rt.hw_frames + g_rt.sw_frames) % 600) == 1) {
		std::printf("RT reflections: hardware ray queries in %llu of %llu frames (KYTY_HWRT_REFL; none: no scene TLAS that frame)\n",
		            static_cast<unsigned long long>(g_rt.hw_frames), static_cast<unsigned long long>(g_rt.hw_frames + g_rt.sw_frames));
		std::fflush(stdout);
	}
	// (hwrt3) the material variant while the albedo array exists (KYTY_HWRT_REFL_MATERIAL)
	vk::ImageView albedo_view    = nullptr;
	vk::Sampler   albedo_sampler = nullptr;
	const bool    hwm = hw && HwScene::MaterialActive() && HwScene::MaterialAtlas(albedo_view, albedo_sampler) &&
	                 EnsureHwMaterialPipeline(*graphics);
	if (hwm && (++g_rt.hwm_frames % 600) == 1) {
		std::printf("RT reflections: material off-screen hits in %llu of %llu frames (KYTY_HWRT_REFL_MATERIAL)\n",
		            static_cast<unsigned long long>(g_rt.hwm_frames), static_cast<unsigned long long>(g_rt.hw_frames + g_rt.sw_frames));
		std::fflush(stdout);
	}
	const vk::PipelineLayout pass_layout = hwm ? g_rt.hwm_layout : hw ? g_rt.hw_layout : g_rt.layout;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, hwm ? g_rt.hwm_pipeline : hw ? g_rt.hw_pipeline : g_rt.pipeline);

	auto&          g    = g_rt;
	const uint32_t cur  = static_cast<uint32_t>(g.frame & 1u);
	const uint32_t prev = cur ^ 1u;
	struct Set {
		const RtImage *a, *b, *gcur, *gprev, *aux, *meta;
		vk::ImageView  hiz_store;
		const RtImage *oa, *ob, *oc;
	};
	const auto run = [&](int mode, const Set& s, vk::Extent2D size) {
		const vk::Sampler       lin = g.linear, pt = g.point;
		const auto              gen = vk::ImageLayout::eGeneral;
		vk::DescriptorImageInfo infos[Bindings] {};
		infos[0]  = {lin, color.view, color.layout};
		infos[1]  = {pt, inputs.depth.view, inputs.depth.layout};
		infos[2]  = {pt, inputs.motion.view, inputs.motion.layout};
		infos[3]  = {pt, g.hiz.view, gen};
		infos[4]  = {nullptr, s.hiz_store, gen};
		infos[5]  = {lin, s.a->view, gen};
		infos[6]  = {lin, s.b->view, gen};
		infos[7]  = {pt, s.gcur->view, gen};
		infos[8]  = {pt, s.gprev->view, gen};
		infos[9]  = {pt, s.aux->view, gen};
		infos[10] = {pt, s.meta->view, gen};
		infos[11] = {nullptr, s.oa->view, gen};
		infos[12] = {nullptr, s.ob->view, gen};
		infos[13] = {nullptr, s.oc->view, gen};
		for (uint32_t i = 0; i < 4; ++i)
			infos[14 + i] = have_gbuf && gbuf.views[i] != nullptr ? vk::DescriptorImageInfo {pt, gbuf.views[i], vk::ImageLayout::eShaderReadOnlyOptimal}
			                                                      : vk::DescriptorImageInfo {pt, g.dummy.view, gen};
		// (binding 20: the G-buffer-time depth; without it the hi-z stands in, never read: misc.w & 64 unset)
		infos[20] = have_water ? vk::DescriptorImageInfo {pt, gbuf.depth, gbuf.depth_layout} : vk::DescriptorImageInfo {pt, g.hiz.view, gen};
		// (binding 21, refl8: the game's SSR, linear; without it the hi-z stands in, never read: misc.w & 256 unset)
		infos[21] = have_guide ? vk::DescriptorImageInfo {lin, guide.view, vk::ImageLayout::eShaderReadOnlyOptimal}
		                       : vk::DescriptorImageInfo {pt, g.hiz.view, gen};
		const vk::DescriptorBufferInfo frame_info {vk::Buffer(g.frame_buf), 0, VK_WHOLE_SIZE};
		const vk::DescriptorBufferInfo exposure_info = game_exposure
		                                                   ? vk::DescriptorBufferInfo {inputs.exposure_buffer, inputs.exposure_base, inputs.exposure_range}
		                                                   : frame_info;
		vk::WriteDescriptorSet writes[Bindings + 3] {};
		for (uint32_t i = 0; i < Bindings; ++i) {
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = (i == 18 || i == 19)                       ? vk::DescriptorType::eStorageBuffer
			                            : (i == 4 || i == 11 || i == 12 || i == 13) ? vk::DescriptorType::eStorageImage
			                                                                        : vk::DescriptorType::eCombinedImageSampler;
			if (i == 18) writes[i].pBufferInfo = &frame_info;
			else if (i == 19) writes[i].pBufferInfo = &exposure_info;
			else writes[i].pImageInfo = &infos[i];
		}
		push.misc[0] = mode;
		// (hwrt2: the scene's RayInfo and TLAS)
		const vk::DescriptorBufferInfo ray_info {inputs.hwrt_info, 0, inputs.hwrt_info_size};
		const auto tlas = vk::AccelerationStructureKHR(reinterpret_cast<VkAccelerationStructureKHR>(inputs.hwrt_tlas));
		vk::WriteDescriptorSetAccelerationStructureKHR as_write {};
		as_write.accelerationStructureCount = 1;
		as_write.pAccelerationStructures    = &tlas;
		if (hw) {
			writes[Bindings].dstBinding          = Bindings;
			writes[Bindings].descriptorCount     = 1;
			writes[Bindings].descriptorType      = vk::DescriptorType::eStorageBuffer;
			writes[Bindings].pBufferInfo         = &ray_info;
			writes[Bindings + 1].pNext           = &as_write;
			writes[Bindings + 1].dstBinding      = Bindings + 1;
			writes[Bindings + 1].descriptorCount = 1;
			writes[Bindings + 1].descriptorType  = vk::DescriptorType::eAccelerationStructureKHR;
		}
		// (hwrt3: binding 24, the albedo array)
		const vk::DescriptorImageInfo albedo_info {albedo_sampler, albedo_view, vk::ImageLayout::eGeneral};
		if (hwm) {
			writes[Bindings + 2].dstBinding      = Bindings + 2;
			writes[Bindings + 2].descriptorCount = 1;
			writes[Bindings + 2].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
			writes[Bindings + 2].pImageInfo      = &albedo_info;
		}
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, pass_layout, 0, hwm ? Bindings + 3 : hw ? Bindings + 2 : Bindings, writes);
		command.pushConstants(pass_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
		Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	};
	// (Bindings a mode does not use still need valid images: tmp_a / tmp_b / rad where nothing reads or writes them.)
	const Set base {&g.tmp_a, &g.tmp_a, &g.tmp_a, &g.tmp_a, &g.tmp_a, &g.tmp_a, g.hiz.mips[0], &g.tmp_b, &g.tmp_b, &g.tmp_b};
	// Hi-z pyramid.
	run(6, base, g.render);
	for (uint32_t mip = 1; mip < g.hiz.levels; ++mip) {
		Set s       = base;
		s.hiz_store = g.hiz.mips[mip];
		push.dec[3] = static_cast<int32_t>(mip);
		run(7, s, MipExtent(g.hiz, mip));
	}
	push.dec[3] = use_normal ? dec_gloss : -1; // (outside the hi-z: the gloss field)
	stamp(StHiz);
	if (calibrate) {
		push.dec[2] = static_cast<int32_t>(g_rt.calib_packed);
		run(5, base, vk::Extent2D {160, 90});
		push.dec[2] = static_cast<int32_t>(packed_count);
		++g.calib_runs;
		stamp(StCalib);
	}
	if (survey) {
		run(8, base, vk::Extent2D {160, 90});
		++g.survey_runs;
		stamp(StSurvey);
	}
	// Trace, temporal, a-trous, composite.
	{
		Set s = base;
		s.oa = &g.rad, s.ob = &g.gb[cur], s.oc = &g.aux;
		// (checkerboard: a thread per pair of horizontal neighbours)
		run(0, s, cfg.checker ? vk::Extent2D {(g.trace.width + 1) / 2, g.trace.height} : g.trace);
		stamp(StTrace);
	}
	{
		Set s = base;
		s.a = &g.rad, s.b = &g.hist[prev], s.gcur = &g.gb[cur], s.gprev = &g.gb[prev], s.aux = &g.aux, s.meta = &g.meta[prev];
		s.oa = &g.hist[cur], s.ob = &g.meta[cur], s.oc = &g.tmp_b;
		run(1, s, g.trace);
		stamp(StTemporal);
	}
	const RtImage* src     = &g.hist[cur];
	const RtImage* ping[2] = {&g.tmp_a, &g.tmp_b};
	for (int i = 0; i < passes; ++i) {
		const RtImage* dst = ping[i & 1];
		Set            s   = base;
		s.a = src, s.b = src, s.gcur = &g.gb[cur], s.gprev = &g.gb[cur], s.aux = &g.aux, s.meta = &g.meta[cur];
		s.oa = dst, s.ob = &g.rad, s.oc = &g.rad;
		push.params[3] = static_cast<float>(1 << i);
		run(2, s, g.trace);
		stamp(static_cast<uint8_t>(StAtrous0 + std::min(i, 3)));
		src = dst;
	}
	{
		Set s = base;
		s.a = src, s.b = src, s.gcur = &g.gb[cur], s.gprev = &g.gb[cur], s.aux = &g.aux, s.meta = &g.meta[cur];
		s.b  = &g.rad; // (refl10: the trace's raw output: this frame's mirror hits at the full-resolution trace)
		s.oa = &g.out, s.ob = &g.rad, s.oc = &g.rad;
		run(4, s, g.render);
		stamp(StComposite);
	}
	// (hwrt4, KYTY_HWRT_METAL) Metal reflections over the composite: trace (TLAS), temporal, 2 a-trous passes, composite.
	const RtImage* final_out = &g.out;
	if (Metal().level > 0 && EnsureMetalImages(*graphics)) {
		PrepareImages(command);
		const MetalCfg& mc     = Metal();
		const RtPush    saved  = push;
		const bool      reset  = inputs.reset || !g.mhist_valid;
		++g.m_frames;
		if (hw) ++g.m_tlas_frames;
		push.size[2] = static_cast<float>(g.mtrace.width), push.size[3] = static_cast<float>(g.mtrace.height);
		push.params[0] = mc.strength, push.params[1] = mc.rough_max, push.params[2] = mc.dist, push.params[3] = 1.0f;
		push.motion[2] = mc.tint, push.motion[3] = static_cast<float>(mc.view);
		push.misc[2]   = mc.level;
		push.dec[1]    = have_gbuf ? mc.rough : -1;
		// (views 1..3 show G-buffer fields only: no rays traced, so a view costs nothing beyond its passes)
		const bool rays = hw && (mc.view == 0 || mc.view == 4);
		push.dec[3]    = (reset ? 0 : 1) | (mc.self_skip ? 2 : 0) | (rays ? 4 : 0) | (std::clamp(static_cast<int32_t>(mc.min_f0 * 1000.0f + 0.5f), 0, 1023) << 8) | (mc.guess ? (1 << 18) : 0);
		push.misc2[3]  = have_gbuf && mc.metal >= 0 ? mc.metal + 1 : 0;
		{
			Set s = base;
			s.oa = &g.mrad, s.ob = &g.mgb[cur], s.oc = &g.maux;
			run(9, s, g.mtrace);
		}
		{
			Set s = base;
			s.a = &g.mrad, s.b = &g.mhist[prev], s.gcur = &g.mgb[cur], s.gprev = &g.mgb[prev], s.aux = &g.maux, s.meta = &g.mmeta[prev];
			s.oa = &g.mhist[cur], s.ob = &g.mmeta[cur], s.oc = &g.mtmp[1];
			run(10, s, g.mtrace);
		}
		const RtImage* msrc = &g.mhist[cur];
		for (int i = 0; i < mc.passes; ++i) {
			const RtImage* dst = &g.mtmp[i & 1];
			Set            s   = base;
			s.a = msrc, s.b = msrc, s.gcur = &g.mgb[cur], s.gprev = &g.mgb[cur], s.aux = &g.maux, s.meta = &g.mmeta[cur];
			s.oa = dst, s.ob = &g.mrad, s.oc = &g.mrad;
			push.params[3] = static_cast<float>(1 << i);
			run(11, s, g.mtrace);
			msrc = dst;
		}
		{
			Set s = base;
			s.a = &g.out, s.b = msrc, s.gcur = &g.mgb[cur], s.gprev = &g.mgb[cur], s.aux = &g.maux, s.meta = &g.mmeta[cur];
			s.oa = &g.mout, s.ob = &g.mrad, s.oc = &g.mrad;
			run(12, s, g.render);
		}
		stamp(StMetal);
		push           = saved;
		g.mhist_valid  = true;
		final_out      = &g.mout;
	}
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead);
	{
		// The sums, counters and the metal survey to the host (the CPU reads them a few frames late; the copy is 5.5 KB).
		vk::BufferCopy region {HeaderBytes, HeaderBytes, (Metal().level > 0 ? MetalHist + 12 : SurveyIndex + SurveyCount) * 4};
		command.copyBuffer(vk::Buffer(g.frame_buf), vk::Buffer(g.read_buf), 1, &region);
		Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
		        vk::AccessFlagBits2::eHostRead);
	}
	if (timed) {
		g.stamps_in[entry] = stamp_n;
		++g.written;
	}
	++g.frame;
	g.history_valid     = true;
	if (final_out == &g.mout) {
		// (rt3fx, KYTY_RT_FLICKER) the metal pass alone: its input (the reflections' composite) against its output
		const auto as_input = [](const RtImage& im) {
			UpscaleImage u {};
			u.image = vk::Image(im.image), u.view = im.view, u.layout = vk::ImageLayout::eGeneral, u.format = im.format;
			u.extent = im.extent, u.usage = im.usage;
			return u;
		};
		RtFlicker::Probe(command, graphics, RtFlicker::Metal, as_input(g.out), as_input(g.mout));
	}
	inputs.color.image  = vk::Image(final_out->image);
	inputs.color.view   = final_out->view;
	inputs.color.layout = vk::ImageLayout::eGeneral;
	inputs.color.format = vk::Format::eR16G16B16A16Sfloat;
	inputs.color.usage  = final_out->usage;
	inputs.color.level  = 0;
	inputs.color.layer  = 0;
	return true;
}

} // namespace Libs::Graphics::FrameGen
