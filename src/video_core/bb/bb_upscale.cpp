// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/bb/bb_upscale.h"
#include "video_core/bb/bb_dlss5.h"
#include "core/bb_respatch.h"
#include "video_core/bb/bb_fg.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "video_core/bb/bb_analyze.h"
#include "video_core/bb/bb_audio_check.h"
#include "video_core/bb/bb_metrics.h"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

namespace BB::SR {

const char* KindName(Kind kind) {
    switch (kind) {
    case Kind::Off:
        return "off";
    case Kind::Native:
        return "native";
    case Kind::Dlss:
        return "DLSS";
    case Kind::Xess:
        return "XeSS";
    case Kind::Fsr3:
        return "FSR 3.1";
    case Kind::Fsr4:
        return "FSR 4";
    }
    return "?";
}

const char* QualityName(Quality quality) {
    switch (quality) {
    case Quality::Native:
        return "native AA (DLAA)";
    case Quality::UltraQualityPlus:
        return "ultra quality plus";
    case Quality::UltraQuality:
        return "ultra quality";
    case Quality::Quality:
        return "quality";
    case Quality::Balanced:
        return "balanced";
    case Quality::Performance:
        return "performance";
    case Quality::UltraPerformance:
        return "ultra performance";
    case Quality::Dynamic:
        return "dynamic";
    }
    return "?";
}

double QualityRatio(Kind kind, Quality quality) {
    const bool xess = kind == Kind::Xess;
    switch (quality) {
    case Quality::Native:
        return 1.0;
    case Quality::UltraQualityPlus:
        return 1.3;
    case Quality::UltraQuality:
        return 1.5;
    case Quality::Quality:
        return xess ? 1.7 : 1.5;
    case Quality::Balanced:
        return xess ? 2.0 : 1.724;
    case Quality::Performance:
        return xess ? 2.3 : 2.0;
    case Quality::UltraPerformance:
        return 3.0;
    case Quality::Dynamic:
        return 1.5;
    }
    return 1.0;
}

std::atomic<uint32_t> g_backend_resets{0};
std::atomic<const char*> g_backend_reset_why{""};
void NoteBackendReset(const char* why) {
    g_backend_resets.fetch_add(1, std::memory_order_relaxed);
    g_backend_reset_why.store(why, std::memory_order_relaxed);
}

} // namespace BB::SR

namespace BB::Upscale {

namespace {

std::string Lower(const char* value) {
    std::string s = value != nullptr ? value : "";
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

const char* Env(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : nullptr;
}

// A crash guard: the upscaler's risky start-up steps (Streamline load, device creation, the first
// evaluation) run with this file present; a crash leaves it behind and the next start runs native.
constexpr const char* GuardFile = "bb_upscale_guard.txt";
std::string g_guard_skipped;
// 0.39: BB_FG_NATIVE (default 1): frame generation asked (BB_FG=amd-* / dlss-*) with no upscaler
// (BB_UPSCALER empty, off or native): the module runs in its native mode, with the scene analysis
// (G-buffer depth, camera, motion vectors incl. object vectors, the HUD-less copy) feeding frame
// generation only. No upscaler is loaded or dispatched and the presented image is not touched.
bool g_fg_native = false;
bool FgNativeWanted() {
    const char* sw = std::getenv("BB_FG_NATIVE");
    if (sw != nullptr && sw[0] == '0') {
        return false;
    }
    const char* fg = std::getenv("BB_FG");
    if (fg == nullptr) {
        return false;
    }
    std::string v = fg;
    for (auto& ch : v) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (v.starts_with("amd") || v.starts_with("fsr")) {
        // 0.41: BB_FG_NATIVE_AMD=0: AMD FSR 3 frame generation keeps the old behaviour (no inputs
        // without an upscaler); DLSS-G unaffected.
        const char* amd = std::getenv("BB_FG_NATIVE_AMD");
        return amd == nullptr || amd[0] != '0';
    }
    return v.starts_with("dlss");
}

void GuardBegin(const char* kind, const char* step) {
    if (std::FILE* f = std::fopen(GuardFile, "w")) {
        std::fprintf(f, "%s\n%s\n", kind, step);
        std::fclose(f);
    }
}
void GuardEnd() {
    std::remove(GuardFile);
}

Config ParseConfig() {
    Config c;
    const std::string up = Lower(Env("BB_UPSCALER"));
    if (up == "dlss") {
        c.kind = SR::Kind::Dlss;
    } else if (up == "dlaa") {
        c.kind = SR::Kind::Dlss;
        c.quality = SR::Quality::Native;
    } else if (up == "xess") {
        c.kind = SR::Kind::Xess;
    } else if (up == "fsr3" || up == "fsr" || up == "fsr31" || up == "fsr3.1") {
        c.kind = SR::Kind::Fsr3;
    } else if (up == "fsr4") {
        c.kind = SR::Kind::Fsr4;
    } else if (up == "native") {
        c.kind = SR::Kind::Native;
    }
    // 0.90: DLSS 5 with no upscaler (Reescalado off / native): the same native analysis as frame
    // generation (G-buffer depth, camera + object motion vectors at the game's size), with
    // DLSS 5 on the native image at that point (BBFgNativeAnalyze). FG may be off.
    const bool dlss5_native = [] {
        const char* d5 = std::getenv("BB_DLSS5");
        return d5 != nullptr && std::atof(d5) > 0.0;
    }();
    if ((c.kind == SR::Kind::Off || c.kind == SR::Kind::Native) &&
        (FgNativeWanted() || dlss5_native)) {
        c.kind = SR::Kind::Native; // analysis only, for frame generation (g_fg_native)
        g_fg_native = true;
    }
    if (c.kind != SR::Kind::Off && c.kind != SR::Kind::Native) {
        if (std::FILE* f = std::fopen(GuardFile, "r")) {
            char kind[64]{}, step[128]{};
            const bool read = std::fgets(kind, sizeof(kind), f) != nullptr &&
                              std::fgets(step, sizeof(step), f) != nullptr;
            std::fclose(f);
            if (Env("BB_UPSCALE_RETRY") != nullptr) {
                GuardEnd();
            } else if (read) {
                std::string k = kind, st = step;
                while (!k.empty() && (k.back() == '\n' || k.back() == '\r')) k.pop_back();
                while (!st.empty() && (st.back() == '\n' || st.back() == '\r')) st.pop_back();
                g_guard_skipped = fmt::format(
                    "{} crashed the previous start during \"{}\": running NATIVE this time "
                    "(delete {} or set BB_UPSCALE_RETRY=1 to try it again)",
                    k, st, GuardFile);
                c.kind = SR::Kind::Native;
            }
        }
    }
    const std::string q = Lower(Env("BB_UPSCALE_MODE"));
    if (up != "dlaa" && !q.empty()) {
        if (q == "native" || q == "dlaa" || q == "aa" || q == "nativeaa") {
            c.quality = SR::Quality::Native;
        } else if (q == "ultraquality+" || q == "ultraqualityplus" || q == "uqplus") {
            c.quality = SR::Quality::UltraQualityPlus;
        } else if (q == "ultraquality" || q == "uq") {
            c.quality = SR::Quality::UltraQuality;
        } else if (q == "quality") {
            c.quality = SR::Quality::Quality;
        } else if (q == "balanced") {
            c.quality = SR::Quality::Balanced;
        } else if (q == "performance" || q == "perf") {
            c.quality = SR::Quality::Performance;
        } else if (q == "ultraperformance" || q == "ultraperf" || q == "ultra_performance") {
            c.quality = SR::Quality::UltraPerformance;
        } else if (q == "drs" || q == "dynamic" || q == "auto") {
            c.quality = SR::Quality::Dynamic;
        }
    }
    if (const char* s = Env("BB_UPSCALE_SHARPNESS") ? Env("BB_UPSCALE_SHARPNESS") : Env("BB_SHARPNESS")) {
        c.sharpness = std::clamp(static_cast<float>(std::atof(s)), 0.0f, 1.0f);
    }
    if (const char* s = Env("BB_UPSCALE_JITTER")) {
        c.jitter = s[0] != '0';
    }
    c.mv_reconstruct = Lower(Env("BB_MV")) != "zero";
    c.mv_upscale_zero = Lower(Env("BB_MV_UPSCALE")) == "zero";
    if (const char* s = Env("BB_HUDLESS_COPY")) {
        c.hudless_copy = s[0] != '0';
    }
    const std::string at = Lower(Env("BB_UPSCALE_AT"));
    c.force_mode = at == "scene" ? 1 : at == "composite" ? 2 : 0;
    if (const char* s = Env("BB_UPSCALE_TO_WINDOW")) {
        c.to_window = s[0] != '0';
    }
    if (const char* s = Env("BB_UPSCALE_SRC")) {
        c.force_source = std::strtoull(s, nullptr, 0);
    }
    if (const char* s = Env("BB_UPSCALE_MENUS")) {
        c.menus = s[0] != '0';
    }
    if (const char* s = Env("BB_UPSCALE_LOG_EVERY")) {
        c.log_every = std::max(60, std::atoi(s));
    }
#if defined(_WIN32)
    // FSR 4's ported runtime picks its model from BB_FSR4_PRESET: from the mode unless given.
    if (c.kind == SR::Kind::Fsr4 && Env("BB_FSR4_PRESET") == nullptr) {
        const char* preset = "quality";
        switch (c.quality) {
        case SR::Quality::Native:
            preset = "native";
            break;
        case SR::Quality::Balanced:
            preset = "balanced";
            break;
        case SR::Quality::Performance:
            preset = "performance";
            break;
        case SR::Quality::UltraPerformance:
            preset = "ultraperf";
            break;
        case SR::Quality::Dynamic:
            preset = "drs";
            break;
        default:
            break;
        }
        _putenv_s("BB_FSR4_PRESET", preset);
    }
    if (c.kind == SR::Kind::Fsr4) {
        _putenv_s("BB_UPSCALER", "fsr4");
    }
#endif
    return c;
}

// BB_ANALYZE dumps also go to their own file in the working folder, bb_analyze_<upscaler>.log
// (rewritten per run of that upscaler, so a later run of another one keeps it).
// One file and one lock for every AnalyzeLog instantiation: a function-local static inside the
// template was a separate FILE* (each opened with "w") per argument list, so the instantiations
// overwrote each other's lines in the same file (owner's 2026-11-02 dumps: cut lines).
void AnalyzeWrite(const std::string& text) {
    static std::mutex m;
    static std::FILE* file = [] {
        const char* up = std::getenv("BB_UPSCALER");
        std::string name = std::string("bb_analyze_") + (up && *up ? up : "off") + ".log";
        return std::fopen(name.c_str(), "w");
    }();
    static u32 lines = 0;
    std::scoped_lock lk{m};
    if (file != nullptr && lines++ < 200000) {
        std::fprintf(file, "%s\n", text.c_str());
        std::fflush(file);
    }
}

template <typename... Args>
void AnalyzeLog(fmt::format_string<Args...> format, Args&&... args) {
    const std::string text = fmt::format(format, std::forward<Args>(args)...);
    LOG_INFO(Render_Vulkan, "{}", text);
    AnalyzeWrite(text);
}

bool AnalyzeOn() {
    return Analyze::Enabled();
}

// ---- Camera ----
struct Mat {
    double m[16]{};
};

Mat Multiply(const Mat& a, const Mat& b) {
    Mat r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double s = 0;
            for (int k = 0; k < 4; ++k) s += a.m[i * 4 + k] * b.m[k * 4 + j];
            r.m[i * 4 + j] = s;
        }
    return r;
}

bool Invert(const Mat& in, Mat& out) {
    double a[4][8]{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) a[i][j] = in.m[i * 4 + j];
        a[i][4 + i] = 1.0;
    }
    for (int c = 0; c < 4; ++c) {
        int pivot = c;
        for (int r = c + 1; r < 4; ++r)
            if (std::fabs(a[r][c]) > std::fabs(a[pivot][c])) pivot = r;
        if (std::fabs(a[pivot][c]) < 1e-15) return false;
        for (int j = 0; j < 8; ++j) std::swap(a[c][j], a[pivot][j]);
        const double s = 1.0 / a[c][c];
        for (int j = 0; j < 8; ++j) a[c][j] *= s;
        for (int r = 0; r < 4; ++r) {
            if (r == c) continue;
            const double f = a[r][c];
            for (int j = 0; j < 8; ++j) a[r][j] -= f * a[c][j];
        }
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) out.m[i * 4 + j] = a[i][4 + j];
    return true;
}

double Dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
double Len3(const double* a) {
    return std::sqrt(Dot3(a, a));
}

// A world -> clip perspective view-projection (rows: clip_i = dot(row_i, p)): its w row is the
// camera's forward axis (unit length), its z row parallel to it, its x and y rows the scaled right
// and up axes, about perpendicular to it and to each other.
bool LooksLikeViewProj(const Mat& v) {
    const double* r0 = &v.m[0];
    const double* r1 = &v.m[4];
    const double* r2 = &v.m[8];
    const double* r3 = &v.m[12];
    for (double x : v.m)
        if (!std::isfinite(x) || std::fabs(x) > 1e7) return false;
    const double n0 = Len3(r0), n1 = Len3(r1), n2 = Len3(r2), n3 = Len3(r3);
    // (Scale invariant: the game's view may carry a world scale, so the w row need not be unit.)
    // The w row is the camera's forward axis: unit length (owner's XeSS dump, 10-29: the camera
    // has it at 1.000; a constant with a 19.97-long row passed the scale-free test and won).
    if (n3 < 0.97 || n3 > 1.03) return false;
    if (n0 < 1e-4 * n3 || n1 < 1e-4 * n3 || n2 < 1e-9) return false;
    const double aspect = n0 / n1;
    if (aspect < 0.2 || aspect > 5.0) return false;
    if (std::fabs(Dot3(r0, r1)) / (n0 * n1) > 0.05) return false;
    if (std::fabs(Dot3(r0, r3)) / (n0 * n3) > 0.25 || std::fabs(Dot3(r1, r3)) / (n1 * n3) > 0.25)
        return false;
    const double c[3]{r2[1] * r3[2] - r2[2] * r3[1], r2[2] * r3[0] - r2[0] * r3[2],
                      r2[0] * r3[1] - r2[1] * r3[0]};
    if (Len3(c) / (n2 * n3) > 0.05) return false;
    // A bare projection (axis aligned, no translation) is not the camera.
    const bool axis_aligned =
        std::fabs(r3[0]) + std::fabs(r3[1]) < 1e-6 && std::fabs(r0[1]) + std::fabs(r0[2]) < 1e-6;
    if (axis_aligned && std::fabs(r3[3]) < 1e-6) return false;
    return true;
}

struct Candidate {
    Mat m;
    u32 count = 0;
    u64 last_draw = ~0ULL;
};

// ---- State ----
struct FrameState {
    bool fg_native_done = false; // 0.39: the FG-only analysis ran (or was tried) this frame
    bool fgn_late = false;       // 1.05: this frame's FG-only inputs were taken at present
    bool fgn_fail = false;       // 1.05: the FG-only analysis ran but gave no inputs
    // G-buffer
    VAddr gb_depth = 0;
    u32 gb_w = 0, gb_h = 0, gb_mrt = 0;
    std::array<Target, 8> gb_colors{};
    u32 gb_num_colors = 0;
    Target gb_depth_target;
    u64 gb_vs_hash = 0, gb_ps_hash = 0;
    u32 gb_draws = 0;
    bool depth_greater = true;
    Viewport viewport;
    // the scene colour and the composite source this frame
    VAddr scene_color = 0;
    VAddr comp_src = 0;
    u32 comp_w = 0, comp_h = 0;
    // sample ordering for the learned injection point
    u32 qual_samples = 0;
    u32 last_write_after = 0;
    bool post_seen = false; // a draw into the presented image sampled a render-size image
    // 0.97 shared injection point (DCport placement): the composite target (the output-size
    // image the game's last post-processing pass draws into, flip or not), the HUD start found
    // this frame (draw index, target) and how it was found.
    VAddr post_target = 0;
    // 1.28: every post-processing target of this frame (the composite / UI image is not always
    // the last one: Bloodborne's later post passes into 0x24ffb0000 moved post_target off its UI
    // image 0x24e7c0000, so its HUD draws never matched and the HUD point missed).
    std::array<VAddr, 16> post_targets{};
    u32 num_post_targets = 0;
    u64 post_draw = 0;
    VAddr hud_target = 0;
    u64 hud_draw = 0;
    const char* hud_how = "";
    std::string trace; // the output-size draws after the scene (first frames, for the log)
    // draws
    u64 draws = 0;
    u32 scene_draws = 0;
    u32 scanned = 0;
    bool camera_ok = false; // a translated view-projection seen in two scene draws
    // 0.23: the camera of the game's 864-byte scene constant block (BlockCamera), the largest
    // pass's (render size area) this frame; 0 = none seen.
    u64 block_area = 0;
    u64 scanned_draw = ~0ULL;
    u32 flip_draws = 0, hud_draws = 0, hud_other = 0, src_rewrites = 0;
    bool injected = false, inject_failed = false, redirecting = false, hudless = false;
    u32 native_post = 0;      // 0.73: the upscaler ended the scene passes (native chain)
    u32 native_redirects = 0; // later samples of the source given the upscaled output
    int miss = -1;            // 0.84: the last reason a qualifying sample was refused (Refuse)
    bool good_src_drawn = false; // 0.84: the kept in-place source was drawn this frame
    bool present = false; // upscaled to the window size this frame
    std::vector<Candidate> camera;
};

struct Learned {
    VAddr gb_depth = 0;
    u32 render_w = 0, render_h = 0;
    VAddr scene_color = 0;
    VAddr comp_src = 0;
    u32 skip = 0;
    u32 skip_age = 0;
    u32 stable = 0;
    bool have_prev_vp = false;
    Mat prev_vp;
    // 0.90: OnFlip rolls prev_vp to this frame's camera before its present-time plan
    // (FillPresentPlan "at present"): that plan's clip_to_prev_clip was vp * inv(vp) = identity,
    // zero camera motion on every present-time frame (0.89 log: "camera delta max 0.000 NDC" in
    // all 111 seconds, generated frames smeared on every camera turn, DLSS-G and FSR 3 alike).
    // The camera of the frame before, kept for the rest of OnFlip.
    bool have_vp_before_roll = false;
    Mat vp_before_roll;
    bool rolled = false; // set between the roll and the end of OnFlip
    bool last_injected = false;
    bool last_fg_native = false; // 0.39: the previous frame had FG-only (native) inputs
    // 0.97: the last HUD start found (draw index, target): a frame that misses the point uses it.
    VAddr hud_target = 0;
    u64 hud_draw = 0;
};

struct Stats {
    u64 frames = 0, injected = 0, failed = 0;
    double gpu_ms_sum = 0, gpu_ms_max = 0;
    u32 gpu_samples = 0;
    u64 flip_draws = 0, hud_draws = 0, hud_other = 0, src_rewrites = 0, hudless = 0;
    u64 camera_frames = 0;
    double stage_ms[kStages]{};
    u32 stage_samples = 0;
    u32 big_w = 0, big_h = 0; // the largest colour target drawn (render-size check)
    u64 big_draws = 0;        // draws into a colour target larger than the upscaler's input
};

std::mutex g_mutex;
FrameState g_frame;
Learned g_learned;
Stats g_stats;
Plan g_plan;
std::array<VAddr, 8> g_flips{};
u32 g_flip_index = 0;
u32 g_output_w = 0, g_output_h = 0;
u64 g_frame_no = 0;
u32 g_jitter_index = 0;
float g_jitter[2]{};
u64 g_jittered_draws = 0;
u64 g_jittered_frame_start = 0;
u64 g_jittered_last_frame = 0;
// 0.79 [taa]: the length of the Halton cycle in use and the scene draws seen per frame (with and
// without a camera found in their vertex constants: those are not jittered unless BB_TAA_ALL=1).
u32 g_jitter_phases = 8;
u64 g_taa_scene_draws = 0, g_taa_nocam_draws = 0;
u64 g_taa_last_scene = 0, g_taa_last_nocam = 0;
u64 g_taa_dispatches = 0;
// 1.27 [taa-stab]: per-second jitter stability counters.
struct TaaStab {
    u64 frames = 0, frames_jittered = 0, frames_cam = 0, frames_nocam = 0, draws = 0,
        fullscreen_skipped = 0, nowrite_skipped = 0, post_point_skipped = 0, mismatch = 0,
        zero_passed = 0, nonzero_passed = 0;
} g_stab;
std::chrono::steady_clock::time_point g_last_flip{};
float g_delta_ms = 16.7f;
std::string g_last_point;
bool g_first_eval_pending = false;
// The first-frame verdict: the window-size path is only known to work once the presenter has
// shown the upscaled image (OnPresentResult); a failed attempt is retried a few times.
bool g_eval_await_present = false;
u32 g_eval_attempts = 0;
constexpr u32 EvalMaxAttempts = 5;
// Window-size path failures in a row; it is switched off only after PresentMaxFails of them.
u32 g_present_fails = 0;
constexpr u32 PresentMaxFails = 3;
// Upscaling to the window size (SetPresentSize).
u32 g_present_w = 0, g_present_h = 0;
bool g_present_dead = false;
bool g_late_upscale = false; // no HUD point this frame: the presenter upscales the flip image
bool g_flip_scene = false;   // the flip being presented had a 3D scene (set at OnFlip)
// The last upscale's formats (SetUpscaleFormats), for the init log.
std::string g_up_formats = "?";
bool g_said_presented = false;
bool g_said_nothing = false;
u64 g_total_upscaled = 0;

// the current draw
struct DrawState {
    bool valid = false;
    bool compute = false;
    std::array<Target, 8> colors{};
    u32 num_colors = 0;
    Target depth{};
    bool has_depth = false;
    bool scene = false;
    bool targets_flip = false;
    bool sampled_upscaled = false;
    bool samples_render_size = false; // samples a render-size colour image it does not draw into
    bool analyze_dump = false;
    bool vertex_camera = false; // a vertex-stage constant holds the forward view-projection
} g_draw;
bool g_scan_vertex = false;

bool g_dump_frame = false;
u32 g_dump_lines = 0;
u32 g_dump_cbs = 0;
std::string g_dump_last;
u32 g_dump_repeats = 0;

bool IsFlip(VAddr addr) {
    return addr != 0 && std::find(g_flips.begin(), g_flips.end(), addr) != g_flips.end();
}

bool IsHdr(vk::Format f) {
    return f == vk::Format::eR16G16B16A16Sfloat || f == vk::Format::eB10G11R11UfloatPack32 ||
           f == vk::Format::eR32G32B32A32Sfloat;
}

bool IsColorSampleable(vk::Format f) {
    switch (f) {
    case vk::Format::eR16G16B16A16Sfloat:
    case vk::Format::eB10G11R11UfloatPack32:
    case vk::Format::eR32G32B32A32Sfloat:
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eA2B10G10R10UnormPack32:
    case vk::Format::eA2R10G10B10UnormPack32:
        return true;
    default:
        return false;
    }
}

bool NativeChainLive();
bool SceneMode() {
    const auto& c = Cfg();
    if (c.force_mode == 1) return true;
    if (c.force_mode == 2) return false;
    // 0.84: the native chain always upscales to the output size (19:13 run: its first upscale
    // ran 1706x960 -> 1706x960 in scene mode, learned from the frame before the scaling).
    if (NativeChainLive()) return false;
    return g_learned.render_w == g_output_w && g_learned.render_h == g_output_h;
}

bool NativeChainLive();
VAddr LearnedSource() {
    if (Cfg().force_source != 0) return Cfg().force_source;
    // 0.80 native chain: the upscaler's source is the lit scene colour (the HDR target the
    // scene's forward / transparent / fog passes draw with the scene depth), at its first read
    // after its last write: lighting, transparencies and fog stay before the upscaler, bloom,
    // tonemap and the HUD after it. (0.76 took the first output-size draw's internal-size input,
    // which became a G-buffer target: self-check 3 of the 07:27 run read R8G8B8A8Unorm data.)
    if (NativeChainLive() && g_learned.scene_color != 0) {
        return g_learned.scene_color;
    }
    return SceneMode() ? g_learned.scene_color : g_learned.comp_src;
}

void FillPresentPlan(VAddr flip, const char* where);
void FirstEvalFailed(std::string_view why);
void PresentFailedLocked(std::string_view why);

bool Active() {
    return Cfg().kind != SR::Kind::Off && Cfg().kind != SR::Kind::Native && BackendReady();
}

// The game renders its whole frame at the presented image's size (scene mode), and the window is
// larger: the upscaler's output is the window size (WantPresentUpscale) instead of in place.
bool PresentMode() {
    const auto& c = Cfg();
    if (!c.to_window || g_present_dead || c.force_mode != 0 || c.force_source != 0 || !Active()) {
        return false;
    }
    if (g_present_w == 0 || g_present_h == 0 || g_learned.render_w == 0 ||
        g_learned.render_w != g_output_w || g_learned.render_h != g_output_h) {
        return false;
    }
    if (g_present_w == g_output_w && g_present_h == g_output_h) {
        return false; // input == output: native AA in place, as before
    }
    return g_present_w >= g_output_w && g_present_h >= g_output_h;
}

// 0.48 (bb-fix-048): native AA (the game renders at the window size, input == output). The
// in-place point needs more samples of the scene colour than the learned skip (the largest write
// count of the last 240 frames): after one busy frame it misses every frame for ~240 flips (the
// user's 23:10:58 log: whole [3d] windows of "other / no injection point 60", DLSS-G turned off
// with no inputs). A 3D frame the in-place point missed now goes through the window-size path at
// the same size (before the HUD, or at present): every 3D frame is anti-aliased and gives frame
// generation its depth, motion vectors and camera. BB_NATIVE_FALLBACK=0: 0.46's behaviour.
bool NativeFallbackOn() {
    static const bool on = [] {
        const char* v = std::getenv("BB_NATIVE_FALLBACK");
        return v == nullptr || *v != '0';
    }();
    return on;
}
bool NativeFallback() {
    const auto& c = Cfg();
    if (!NativeFallbackOn() || !c.to_window || g_present_dead || c.force_mode != 0 ||
        c.force_source != 0 || !Active()) {
        return false;
    }
    return g_present_w != 0 && g_present_h != 0 && g_learned.render_w != 0 &&
           g_learned.render_w == g_output_w && g_learned.render_h == g_output_h &&
           g_present_w == g_output_w && g_present_h == g_output_h;
}
// 0.97: the shared injection point (after post-processing, before the HUD) is the default for
// every frame the game renders at its output size: upscaler / DLAA, DLSS 5 and frame generation
// inputs all run there. BB_INJECT=inplace: the in-place scene-colour point first (0.96). The
// scene-upscale modes (internal-size scene: native chain, resolution patch) keep their own point.
// 1.02 (bb-native-102): native AA with an upscaler (DLAA / FSR 4 native / ratio 1). The 0.97 HUD
// point only takes the presented image for the upscaler, but Bloodborne composites its scene into
// its UI image, so at ratio 1 every frame missed it and went to the present-time path (the
// finished image, HUD included: AA and DLSS 5 on the HUD, and the frames where the HUD point did
// land on the flip switched paths: overlays off and on). At ratio 1 the resolution patch is a
// no-op (scene == output), so the scene-upscale composite point does not exist either. Native AA
// therefore uses the in-place point on the HDR scene colour (pre-0.97 / pre-0.93b DLAA), DLSS 5
// right after the upscaler there, the HUD drawn by the game afterwards. BB_INJECT=hud: 0.97.
bool NativeAaUpscaler() {
    if (Cfg().kind == SR::Kind::Off || Cfg().kind == SR::Kind::Native || FgNative()) return false;
    return SR::QualityRatio(Cfg().kind, Cfg().quality) <= 1.01 &&
           Cfg().quality != SR::Quality::Dynamic;
}
bool HudPointOn() {
    static const int on = [] {
        const char* v = std::getenv("BB_INJECT");
        if (v != nullptr && std::string_view{v} == "inplace") return 0;
        if (v != nullptr && std::string_view{v} == "hud") return 2;
        return 1;
    }();
    if (on == 0 || (on == 1 && NativeAaUpscaler())) {
        static bool said = false;
        if (!said && on == 1) {
            said = true;
            Checkpoint("[inject] 1.02 native AA (ratio 1): in-place point on the HDR scene colour "
                       "for the upscaler / DLAA, DLSS 5 after it, HUD untouched; the HUD point "
                       "(present-time) is not used (BB_INJECT=hud: 0.97 behaviour)");
        }
        return false;
    }
    return !NativeChainLive() && g_learned.render_w != 0 &&
           g_learned.render_w == g_output_w && g_learned.render_h == g_output_h;
}
// The window-size path may run this frame (to the window, or native AA's same-size fallback).
bool WindowPathOk() {
    // 0.93: never while the native chain scales the scene (its frames upscale only in place).
    if (NativeChainLive()) return false;
    return PresentMode() || NativeFallback();
}
bool g_last_window_path = false; // the last upscaled frame went through the window-size path
// Native fallback hold: after a frame on the window-size path, stay on it this many flips (no
// history reset ping-pong between the two paths; the in-place miss lasts ~240 flips anyway).
// BB_NATIVE_HOLD=<flips> (default 240, 0 = back to the in-place point on the next frame).
u32 NativeHoldFlips() {
    static const u32 v = [] {
        const char* e = std::getenv("BB_NATIVE_HOLD");
        return e == nullptr || *e == '\0' ? 240u : static_cast<u32>(std::max(0, std::atoi(e)));
    }();
    return v;
}
u32 g_native_hold = 0;

float Halton(u32 index, u32 base) {
    float f = 1.0f, r = 0.0f;
    while (index > 0) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(index % base);
        index /= base;
    }
    return r;
}

const Candidate* BestCamera() {
    // Bloodborne keeps a camera-relative view-projection (no translation) next to the inverse of
    // the full one (owner's dump): the full one is preferred, since only it moves with the camera's
    // position; the camera-relative one only when nothing else is there.
    const auto translated = [](const Candidate& c) {
        return std::fabs(c.m.m[3]) + std::fabs(c.m.m[7]) + std::fabs(c.m.m[15]) > 1e-3;
    };
    const Candidate* best = nullptr;
    for (const auto& c : g_frame.camera) {
        if (translated(c) && (best == nullptr || c.count > best->count)) best = &c;
    }
    // The full (translated) view-projection is often only in a few draws' constants (lighting /
    // fog passes): seen twice in the frame's scene draws is enough. Requiring a third of all
    // scanned draws rejected it in most gameplay frames ("no camera yet", no jitter, zero MV).
    if (best != nullptr && best->count >= 2) return best;
    best = nullptr;
    for (const auto& c : g_frame.camera) {
        if (best == nullptr || c.count > best->count) best = &c;
    }
    if (best == nullptr) return nullptr;
    const u32 need = std::max<u32>(3, g_frame.scanned / 8);
    return best->count >= need ? best : nullptr;
}

// 0.23: Bloodborne's scene constant block (864 bytes; deadinside28's vk_camera_motion.cpp
// OnConstants): [0] far plane 3000, [1] 1 / far, [4] [5] the render size, floats 8-19 the view
// (3x4, row-major, translation in the 4th column), 180-191 its inverse, 52-67 the projection
// (4x4 row-major). Its view-projection is the frame's camera outright (a candidate no scanned
// constant outranks); the scan keeps running for the jitter's vertex-camera test.
// BB_CAMERA_BLOCK=0: the 0.22 scan only.
bool BlockCamera(FrameState& f, const float* d, size_t size) {
    static const bool on = [] {
        const char* v = std::getenv("BB_CAMERA_BLOCK");
        return v == nullptr || v[0] != '0';
    }();
    if (!on || size < 864 || d[0] != 3000.0f || std::fabs(d[1] * d[0] - 1.0f) > 1e-3f ||
        !(d[4] >= 64.0f) || !(d[5] >= 64.0f) || d[4] > 16384.0f || d[5] > 16384.0f) {
        return false;
    }
    const u64 area = static_cast<u64>(d[4]) * static_cast<u64>(d[5]);
    if (area <= f.block_area) {
        return false; // (a smaller pass: a reflection's camera; or this one again)
    }
    Mat view, proj;
    for (int i = 0; i < 12; ++i) view.m[i] = d[8 + i];
    view.m[15] = 1.0;
    for (int i = 0; i < 16; ++i) proj.m[i] = d[52 + i];
    const Mat vp = Multiply(proj, view);
    static u32 rejected = 0;
    if (!LooksLikeViewProj(vp)) {
        if (rejected++ < 4) {
            Checkpoint(fmt::format("camera block: rejected (view-projection not camera-like): "
                                   "proj {:.4g} {:.4g} {:.4g} {:.4g} / w row {:.4g} {:.4g} {:.4g} "
                                   "{:.4g}",
                                   d[52], d[57], d[62], d[63], d[64], d[65], d[66], d[67]));
        }
        return false;
    }
    f.block_area = area;
    std::erase_if(f.camera, [](const Candidate& c) { return c.count >= 1000000; });
    f.camera.push_back({vp, 1000000, f.draws});
    f.camera_ok = true;
    static bool said = false;
    if (!said) {
        said = true;
        Checkpoint(fmt::format("camera block: the scene constant block's camera ({}x{} pass): "
                               "proj x {:.4f} y {:.4f} z {:.6g} {:.6g}, position {:.3f} {:.3f} "
                               "{:.3f}",
                               d[4], d[5], d[52], d[57], d[62], d[63], d[180 + 3], d[180 + 7],
                               d[180 + 11]));
    }
    return false; // (the scan still looks at it: vertex-camera / jitter marking)
}

int g_depth_detected = -1; // 1 reversed-Z, 0 forward, -1 not yet known

int RtShadowLevel() {
    static const int level = [] {
        const char* v = std::getenv("BB_RT_SHADOW");
        if (v != nullptr && *v >= '1' && *v <= '9') {
            return *v - '0';
        }
        // BB_RT_REFLECT=1 (the launcher's checkbox) also logs the camera.
        const char* r = std::getenv("BB_RT_REFLECT");
        return r != nullptr && std::atoi(r) >= 1 ? 1 : 0;
    }();
    return level;
}

struct RtShadowStats {
    u64 calls = 0;
    u64 valid = 0;
    u64 prev = 0;
};
RtShadowStats g_rts;

void LogRtShadowCamera(const Plan& p) {
    if (RtShadowLevel() == 0) return;
    ++g_rts.calls;
    g_rts.valid += p.camera_valid;
    g_rts.prev += p.camera_valid && g_learned.have_prev_vp;
    if (g_rts.calls % 600 != 1) return;
    const auto row = [](const float* m) {
        return fmt::format("[{:+.4f} {:+.4f} {:+.4f} {:+.4f}]", m[0], m[1], m[2], m[3]);
    };
    const float* v = p.view_to_clip;
    const float* c = p.clip_to_prev_clip;
    Checkpoint(fmt::format(
        "BB_RT_SHADOW camera: {} (valid {}/{} plans, with prev {}), view_proj {}{}{}{}, "
        "clip_to_prev {}{}{}{}, jitter ({:+.4f},{:+.4f}), fov_y {:.3f}, near {:.3f}, far {:.1f}, "
        "depth {}",
        p.camera_valid ? "valid" : "invalid", g_rts.valid, g_rts.calls, g_rts.prev, row(v),
        row(v + 4), row(v + 8), row(v + 12), row(c), row(c + 4), row(c + 8), row(c + 12),
        p.jitter[0], p.jitter[1], p.fov_y, p.z_near, p.z_far,
        p.depth_inverted ? "inverted" : "forward"));
}

// The view-projection with the scene viewport (pixels) and depth buffer mapping, and its inverse,
// for the reflection pass (bb_rtrefl.comp): the same mapping bb_mv.comp unprojects with.
void FillPixelMatrices(const Mat& vp, Plan& p) {
    p.reflect_valid = false;
    const Viewport& v = p.viewport;
    Mat viewport{};
    viewport.m[0] = v.xscale;
    viewport.m[3] = v.xoffset;
    viewport.m[5] = v.yscale;
    viewport.m[7] = v.yoffset;
    viewport.m[10] = v.zscale != 0.0f ? v.zscale : 1.0f;
    viewport.m[11] = v.zoffset;
    viewport.m[15] = 1.0;
    const Mat to_pix = Multiply(viewport, vp);
    Mat inv{};
    if (!Invert(to_pix, inv) || std::fabs(inv.m[14]) < 1e-12) {
        return;
    }
    for (int i = 0; i < 16; ++i) {
        p.world_to_pix[i] = static_cast<float>(to_pix.m[i]);
        p.pix_to_world[i] = static_cast<float>(inv.m[i]);
    }
    p.reflect_valid = true;
}

void FillCamera(Plan& p) {
    p.camera_valid = false;
    p.reflect_valid = false; // g_plan persists: no stale reflection matrices without a camera
    const Candidate* cam = Cfg().mv_reconstruct ? BestCamera() : nullptr;
    if (cam == nullptr) {
        p.depth_inverted = g_depth_detected >= 0 ? g_depth_detected == 1 : g_frame.depth_greater;
        return;
    }
    const Mat& vp = cam->m;
    Mat inv;
    if (!Invert(vp, inv)) return;
    const bool have_prev =
        g_learned.rolled ? g_learned.have_vp_before_roll : g_learned.have_prev_vp;
    const Mat prev = !have_prev            ? vp
                     : g_learned.rolled ? g_learned.vp_before_roll
                                        : g_learned.prev_vp;
    const Mat to_prev = Multiply(prev, inv);
    for (int i = 0; i < 16; ++i) {
        p.view_to_clip[i] = static_cast<float>(vp.m[i]);
        p.clip_to_prev_clip[i] = static_cast<float>(to_prev.m[i]);
    }
    const double* r1 = &vp.m[4];
    const double* r2 = &vp.m[8];
    const double* r3 = &vp.m[12];
    p.fov_y = static_cast<float>(2.0 * std::atan(Len3(r3) / std::max(Len3(r1), 1e-9)));
    // ndc_z = alpha + beta / w (w the view depth)
    const double alpha = Dot3(r2, r3) / Dot3(r3, r3);
    const double beta = r2[3] - alpha * r3[3];
    // (The depth direction stays the depth test's, as without a camera: a value that changes
    // between frames recreated the FSR 3.1 / XeSS contexts, a stutter each time.)
    (void)beta;
    p.depth_inverted = g_frame.depth_greater;
    const double w0 = std::fabs(alpha) > 1e-9 ? -beta / alpha : 1e30;
    const double w1 = std::fabs(1.0 - alpha) > 1e-9 ? beta / (1.0 - alpha) : 1e30;
    double z_near = std::min(std::fabs(w0), std::fabs(w1));
    double z_far = std::max(std::fabs(w0), std::fabs(w1));
    if (!(z_near > 1e-4)) z_near = 0.1;
    p.z_near = static_cast<float>(z_near);
    p.z_far = z_far > 1e7 ? 0.0f : static_cast<float>(z_far);
    // The depth convention from the projection: the window depth of a point at infinity
    // (ndc_z -> alpha) against one at the near plane, through the scene viewport's z mapping.
    if (g_depth_detected < 0) {
        const double zs = p.viewport.zscale != 0.0f ? p.viewport.zscale : 1.0;
        const double zo = p.viewport.zoffset;
        const double far_d = alpha * zs + zo;
        const double near_d = (alpha + beta / z_near) * zs + zo;
        if (std::isfinite(far_d) && std::isfinite(near_d) && std::fabs(far_d - near_d) > 0.1) {
            g_depth_detected = far_d < near_d ? 1 : 0;
            Checkpoint(fmt::format(
                "depth convention: {} (detected from the camera projection: far plane depth "
                "{:.3f}, near plane depth {:.3f}; the G-buffer depth test says {}), used by "
                "every upscaler{}",
                g_depth_detected == 1 ? "reversed (far = 0)" : "forward (far = 1)", far_d,
                near_d, g_frame.depth_greater ? "greater (reversed)" : "less (forward)",
                std::getenv("BB_DEPTH_INVERT") ? " (BB_DEPTH_INVERT overrides it)" : ""));
        }
    }
    p.depth_inverted = g_depth_detected >= 0 ? g_depth_detected == 1 : g_frame.depth_greater;
    p.camera_valid = true;
    FillPixelMatrices(vp, p);
}

void LogPoint(const std::string& text) {
    if (text != g_last_point) {
        g_last_point = text;
        LOG_INFO(Render_Vulkan, "[BB_UPSCALE] {}", text);
    }
}

std::string Describe(const Target& t) {
    return fmt::format("{:#x} {}x{} {}", t.addr, t.width, t.height, vk::to_string(t.format));
}

} // namespace

void Checkpoint(std::string_view text) {
    static std::mutex m;
    static std::FILE* file = nullptr;
    static u32 lines = 0;
    std::scoped_lock lk{m};
    if (file == nullptr && lines == 0) {
        // Appended, the last 8 runs kept: a crashing run survives the native run after it, and
        // an A/B of every upscaler in one session stays readable.
        std::string old;
        if (std::FILE* in = std::fopen("bb_upscale_init.log", "rb")) {
            char buf[4096];
            std::size_t n;
            while ((n = std::fread(buf, 1, sizeof(buf), in)) > 0) old.append(buf, n);
            std::fclose(in);
        }
        const std::string marker = "==== run ";
        std::vector<std::size_t> starts;
        for (std::size_t at = old.find(marker); at != std::string::npos;
             at = old.find(marker, at + 1)) {
            starts.push_back(at);
        }
        if (starts.size() >= 8) {
            old = old.substr(starts[starts.size() - 7]);
        }
        file = std::fopen("bb_upscale_init.log", "wb");
        if (file != nullptr) {
            std::fwrite(old.data(), 1, old.size(), file);
            const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            char when[64]{};
            std::strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
            std::fprintf(file,
                         "==== run %s (BB_UPSCALER=%s, BB_UPSCALE_MODE=%s, BB_SCENE_UPSCALE=%s) "
                         "====\n",
                         when, Env("BB_UPSCALER") ? Env("BB_UPSCALER") : "",
                         Env("BB_UPSCALE_MODE") ? Env("BB_UPSCALE_MODE") : "",
                         Env("BB_SCENE_UPSCALE") ? Env("BB_SCENE_UPSCALE") : "");
            std::fflush(file);
        }
    }
    if (file != nullptr && lines < 4000) {
        std::fprintf(file, "%.*s\n", static_cast<int>(text.size()), text.data());
        std::fflush(file);
    }
    ++lines;
    LOG_INFO(Render_Vulkan, "[BB_UPSCALE] {}", text);
    Common::Log::Flush();
}

const Config& Cfg() {
    static const Config config = ParseConfig();
    return config;
}

namespace {
// Streamline is loaded for DLSS and for DLSS frame generation (BB_FG=dlss-*, bb_fg.h).
bool Streamline() {
    return Cfg().kind == SR::Kind::Dlss || BB::FG::StreamlineWanted();
}
} // namespace

namespace {
std::atomic<bool> g_ss_frozen{false};
std::atomic<bool> g_ss_off{false}; // turned off at run time by the safety net
u32 g_ss_frames = 0, g_ss_scene = 0, g_ss_good = 0, g_ss_windows = 0;
std::array<VAddr, 64> g_ss_addrs{};
// 0.73 native chain (BB_SCENE_UPSCALE=native): which version of a scene target a pass gets is
// decided per pass, not per address. Bloodborne reuses the same guest addresses across the frame
// (the G-buffer's colour 0 0x27a3c8000 of the 11-04 analysis shows up again in the 11-05 03:19
// run, bound with the scaled targets after the upscaler, and two more addresses joined the same
// way), and 0.71 scaled an address for the whole frame: once an output-size pass shared an address
// (or was bound with one) it was shrunk too, the shrink spread through every pass bound with it,
// and the game's output chain ended at 1706x960 inside the 2560x1440 flip (the top-left 1/1.5)
// while post-processing read internal-size images at output-size pixel positions (the flat 3D).
//   phase 0: before this frame's scene passes, phase 1: the scene passes (from the first draw
//   with the scene depth until the upscaler runs), phase 2: after the upscaler (post-processing,
//   HUD, flip). Only phase-1 draws use the internal-size images; phase 0 / 2 and every compute
//   dispatch use an output-size twin at the same address, filled from the internal-size image
//   (bilinear colour, nearest depth) the first time it is used after the scene wrote it
//   (TwinAccess, texture_cache BBTwinSync). The upscaler's source keeps its internal-size image
//   after the upscaler (the samples of it are redirected to the upscaled output).
std::atomic<int> g_ss_phase{0};
std::atomic<bool> g_ss_compute{false};
std::atomic<VAddr> g_ss_depth{0};
std::atomic<VAddr> g_ss_source{0};
std::array<std::atomic<u8>, 64> g_ss_valid{};  // bit 0: internal-size image current, bit 1: output
std::array<std::atomic<u8>, 64> g_ss_class{};  // classification logged: 1 scene, 2 output, 4 compute
std::atomic<u32> g_ss_twin_copies{0}, g_ss_twin_fail{0}, g_ss_post_scaled{0}, g_ss_post_refused{0};
// 0.76: the frame number of the scene passes (counted at their start) and, per address and
// version, the frame its content was last made; reads of content older than the last frame are
// counted as stale (twins filled once and read for many frames). g_ss_purge: the chain turned off,
// the texture cache brings every address back to one output-size image (BBNativeOff).
std::atomic<u32> g_ss_frame{1}, g_ss_stale_reads{0};
std::array<std::array<std::atomic<u32>, 2>, 64> g_ss_age{};
std::atomic<bool> g_ss_purge{false};
// 0.83: per address: stale output-size reads (count, by a dispatch / draw), the frame of its first
// use and whether that first use was a read (a history image: read before it is written in the
// frame), and the pin: a history address keeps one size (the output size) in every pass, so the
// game's history is never resampled between sizes (BB_NATIVE_HISTORY=copy: 0.80 behaviour).
std::array<std::atomic<u32>, 64> g_ss_stale_by{}, g_ss_seen{}, g_ss_hist{};
std::array<std::atomic<bool>, 64> g_ss_pinned{};
std::array<std::atomic<u8>, 64> g_ss_stale_kind{};
// 0.80: compute pipelines of the scene passes run at the internal size (their grid scaled, the
// game's size constants patched) once they were seen touching a scene target; g_ss_compute_scaled
// for the dispatch being bound, g_ss_compute_touched: it looked up a scene target.
std::atomic<bool> g_ss_compute_scaled{false}, g_ss_compute_touched{false};
// 0.80 content check: PSNR (dB, x100) of the downscaled output against the internal lit colour.
std::atomic<int> g_ss_psnr_last{-1};
std::atomic<u32> g_ss_psnr_n{0}, g_ss_psnr_low{0};
// 0.76: why a qualifying sample of the upscaler's source was not the injection point.
enum Refuse : u32 { RfPresent, RfHold, RfSkip, RfInactive, RfFailed, RfSize, RfTarget, RfCount };
std::array<std::atomic<u32>, RfCount> g_ss_refuse{};
// 0.84 in-place stability: the source and sample index of the last in-place upscale are kept
// (a frame whose scene-colour guess or sample count differs does not move the point), the frame
// it last worked, and per-reason counts of the 3D frames that missed it.
// 0.85 session path lock (BB_PATH_LOCK, default on): 0 undecided, 1 in place, 2 window /
// present-time path. Decided from the first 300 3D frames (in place in >= 95% of them, else the
// window path); after that the other path is never used, so the upscaler's input and history never
// switch. Resets of the upscaler history, and frames per second on the locked path.
int g_lock = 0;
u32 g_lock_frames = 0, g_lock_inplace = 0;
u32 g_sec_frames = 0, g_sec_locked = 0, g_sec_resets = 0;
// bb-dcideas-094 (DCport 0.5.3 lesson: the HUD reset FSR's history every frame): resets per
// second by reason, printed as [sr-hist] whether the path lock is on or not.
u32 g_sec_upscales = 0, g_rs_first = 0, g_rs_cut = 0, g_rs_size = 0, g_rs_path = 0;
bool PathLockOn() {
    static const bool on = [] {
        const char* v = std::getenv("BB_PATH_LOCK");
        return v == nullptr || v[0] != '0';
    }();
    return on;
}
double g_sec_cam_max = 0.0;
// 0.85: with the path lock the upscaler's history is reset only on its first frame and on a real
// camera cut, by the same test as DLSS-G (bb-dlssg-085): the mean displacement (NDC, the screen
// 2 wide) of four clip points through clip_to_prev_clip above 1.0 (BB_SR_CUT_NDC). The path-state
// resets (last frame not in place, window path before) are not used. BB_PATH_LOCK=0: as before.
void SrReset(Plan& p) {
    ++g_sec_upscales;
    // A render or output size change always resets (the history no longer matches); nothing
    // else that changes per frame (HUD draws, render-target switches) does.
    static u32 last_rw = 0, last_rh = 0, last_ow = 0, last_oh = 0;
    const bool size_changed = last_rw != 0 && (p.render_w != last_rw || p.render_h != last_rh ||
                                               p.output_w != last_ow || p.output_h != last_oh);
    last_rw = p.render_w;
    last_rh = p.render_h;
    last_ow = p.output_w;
    last_oh = p.output_h;
    if (!PathLockOn()) {
        if (size_changed) p.reset = true;
        if (p.reset) {
            ++g_sec_resets;
            ++(size_changed ? g_rs_size : g_rs_path);
        }
        return;
    }
    static const double cut = [] {
        const char* e = std::getenv("BB_SR_CUT_NDC");
        return e != nullptr && *e != '\0' ? std::atof(e) : 3.0; // bb-dlssgpace-089: 1.0 cut on turns (1.03-1.04 NDC)
    }();
    static bool first = true;
    double delta = 0.0;
    if (p.camera_valid) {
        const float* m = p.clip_to_prev_clip;
        const double pts[4][2] = {{-0.5, -0.5}, {0.5, -0.5}, {-0.5, 0.5}, {0.5, 0.5}};
        double sum = 0.0;
        for (const auto& pt : pts) {
            // 1.05: far-plane points (z 0.5 sat at the lens with forward depth: 3.9-4.7 NDC on
            // ordinary turns, 5-14 history resets per second).
            const double v[4] = {pt[0], pt[1], p.depth_inverted ? 1e-4 : 0.9999, 1.0};
            double o[4]{};
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) o[i] += m[i * 4 + j] * v[j];
            if (std::fabs(o[3]) < 1e-9) {
                sum = 1e9;
                break;
            }
            const double dx = o[0] / o[3] - pt[0], dy = o[1] / o[3] - pt[1];
            sum += std::sqrt(dx * dx + dy * dy);
        }
        delta = sum >= 1e9 ? 1e9 : sum / 4.0;
    }
    g_sec_cam_max = std::max(g_sec_cam_max, std::min(delta, 99.0));
    // 1.05: a real discontinuity only: above the threshold and 4x the recent average.
    static double avg = 0.0;
    const double dclamp = std::min(delta, 99.0);
    const bool is_cut = delta > cut && dclamp > 4.0 * std::max(avg, 0.05);
    avg = avg * 0.8 + dclamp * 0.2;
    p.reset = first || is_cut || size_changed;
    if (p.reset) {
        ++g_sec_resets;
        ++(first ? g_rs_first : size_changed ? g_rs_size : g_rs_cut);
    }
    first = false;
}
VAddr g_good_src = 0;
u32 g_good_index = 0;
u64 g_good_frame = 0, g_good_src_seen = 0;
std::array<u32, RfCount + 2> g_miss_counts{};
u32 g_miss_kept = 0;
std::atomic<u32> g_ss_count{0};
u32 g_ss_w = 0, g_ss_h = 0, g_ss_sw = 0, g_ss_sh = 0;
std::vector<VAddr> g_ss_candidates;

bool SceneScaleWanted() {
    static const bool on = [] {
        // 0.23: BB_SCENE_UPSCALE=1 is the startup Resolution Patch (core/bb_respatch.cpp); the
        // 0.22 shrinking of the scene's targets only with BB_SCENE_UPSCALE=shrink.
        return BB::ResPatch::ShrinkWanted();
    }();
    return on;
}
} // namespace

namespace {
// BB_SCENE_NATIVE_MIXED=scale: the 0.71 per-address behaviour (an address is scaled for the whole
// frame, a target bound with scaled ones is scaled too), for comparison only.
bool NativeLegacy() {
    static const bool on = [] {
        const char* v = std::getenv("BB_SCENE_NATIVE_MIXED");
        return v != nullptr && std::string_view{v} == "scale";
    }();
    return on;
}
bool NativePerPass();
bool NativeChainLive() {
    return NativePerPass() && g_ss_frozen.load(std::memory_order_relaxed) &&
           !g_ss_off.load(std::memory_order_relaxed);
}
bool NativePerPass() {
    return BB::ResPatch::NativeChain() && !NativeLegacy();
}
// 0.93: BB_NATIVE_CHECK_OFF=1: a failed native chain check / safety net turns scene scaling off
// (0.84). Default: it only logs (and dumps one frame); the internal-resolution path stays.
bool NativeCheckOff() {
    static const bool on = [] {
        const char* v = std::getenv("BB_NATIVE_CHECK_OFF");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}
// 0.84: the frame the native chain check asks to be dumped (BB_DUMP_FRAME) before it turns
// scene scaling off, and the frame from which it turns off.
std::atomic<u64> g_dump_request{0};
u64 g_off_at = 0;
// Every way scene scaling turns off goes through here: the texture cache then copies each
// address's current content to its output-size image and frees the internal-size one.
void TurnOff() {
    if (!g_ss_off.exchange(true, std::memory_order_relaxed) &&
        g_ss_frozen.load(std::memory_order_relaxed) && BB::ResPatch::NativeChain()) {
        g_ss_purge.store(true, std::memory_order_release);
    }
}
int SceneIndex(VAddr addr) {
    const u32 n = g_ss_count.load(std::memory_order_acquire);
    for (u32 i = 0; i < n; ++i) {
        if (g_ss_addrs[i] == addr) return static_cast<int>(i);
    }
    return -1;
}
// One line per address and class: how the passes of the frame use it.
void Classify(int idx, u8 bit, VAddr addr) {
    if (idx < 0 || (g_ss_class[idx].fetch_or(bit, std::memory_order_relaxed) & bit) != 0) {
        return;
    }
    Checkpoint(fmt::format(
        "native chain: {:#x} {}", addr,
        bit == 1 ? fmt::format("used by the scene passes: internal size {}x{}", g_ss_sw, g_ss_sh)
        : bit == 2
            ? fmt::format("used after the upscaler / outside the scene passes (output chain): "
                          "output-size twin {}x{}, filled from the internal-size image",
                          g_ss_w, g_ss_h)
            : fmt::format("used by a compute dispatch (dispatch grids are in guest pixels): "
                          "output-size twin {}x{}",
                          g_ss_w, g_ss_h)));
}
} // namespace

bool ScaleImage(VAddr addr, u32& w, u32& h) {
    if (!g_ss_frozen.load(std::memory_order_acquire) || g_ss_off.load(std::memory_order_relaxed) ||
        w != g_ss_w || h != g_ss_h) {
        return false;
    }
    const int idx = SceneIndex(addr);
    if (idx < 0) {
        return false;
    }
    if (NativePerPass() && g_ss_pinned[idx].load(std::memory_order_relaxed)) {
        return false; // 0.83: a history address, one size (output) everywhere
    }
    if (NativePerPass()) {
        const int phase = g_ss_phase.load(std::memory_order_relaxed);
        const bool compute = g_ss_compute.load(std::memory_order_relaxed);
        // 0.80: the scene passes (draws, and the compute pipelines learned as scene compute) use
        // the internal-size image; after the upscaler every pass, the source included, uses the
        // output-size image, which for the source holds the upscaled image (BBNativeFillSource).
        if (compute && phase == 1) {
            g_ss_compute_touched.store(true, std::memory_order_relaxed);
        }
        const bool scaled =
            phase == 1 && (!compute || g_ss_compute_scaled.load(std::memory_order_relaxed));
        Classify(idx, scaled ? 1 : compute ? 4 : 2, addr);
        if (!scaled) {
            return false;
        }
    }
    w = g_ss_sw;
    h = g_ss_sh;
    return true;
}

bool TwinAccess(VAddr addr, u32 w, u32 h, bool scaled, bool write, bool depth) {
    if (!NativePerPass() || !g_ss_frozen.load(std::memory_order_acquire) ||
        g_ss_off.load(std::memory_order_relaxed)) {
        return false;
    }
    if ((scaled && (w != g_ss_sw || h != g_ss_sh)) || (!scaled && (w != g_ss_w || h != g_ss_h))) {
        return false;
    }
    const int idx = SceneIndex(addr);
    if (idx < 0) {
        return false;
    }
    if (g_ss_pinned[idx].load(std::memory_order_relaxed) && !scaled) {
        // A pinned history address: its output-size image is the only one in use. One last copy
        // if the internal-size image still holds the current content.
        const u8 v = g_ss_valid[idx].load(std::memory_order_relaxed);
        if ((v & 2) == 0 && (v & 1) != 0) return true;
        g_ss_valid[idx].store(2, std::memory_order_relaxed);
        return false;
    }
    const u8 mine = scaled ? 1 : 2, other = scaled ? 2 : 1;
    const u8 valid = g_ss_valid[idx].load(std::memory_order_relaxed);
    {
        // 0.83 history detection: the first use of the address in this frame is a read.
        static const bool history_copy = [] {
            const char* v = std::getenv("BB_NATIVE_HISTORY");
            return v != nullptr && std::string_view{v} == "copy";
        }();
        const u32 frame = g_ss_frame.load(std::memory_order_relaxed);
        if (g_ss_seen[idx].exchange(frame, std::memory_order_relaxed) != frame && !write &&
            g_ss_phase.load(std::memory_order_relaxed) != 2 && !history_copy &&
            !g_ss_pinned[idx].load(std::memory_order_relaxed) &&
            addr != g_ss_depth.load(std::memory_order_relaxed) &&
            addr != g_ss_source.load(std::memory_order_relaxed) &&
            g_ss_hist[idx].fetch_add(1, std::memory_order_relaxed) + 1 == 3) {
            g_ss_pinned[idx].store(true, std::memory_order_relaxed);
            Checkpoint(fmt::format(
                "native chain: {:#x} is a history image (read before it is written in 3 frames, "
                "first by a {}): kept at the output size {}x{} in every pass from now on "
                "(BB_NATIVE_HISTORY=copy: resampled per pass as 0.80)",
                addr, g_ss_compute.load(std::memory_order_relaxed) ? "dispatch" : "draw", g_ss_w,
                g_ss_h));
        }
    }
    // 0.76: every use of a version that is not current copies the current one first, scene-pass
    // writes included. 0.73 skipped the copy for a scene-pass target ("the scene rebuilds it"),
    // but a target bound by the scene passes is often only blended onto: after a compute dispatch
    // (which uses the output-size twin) wrote it, the scene draws blended onto an internal-size
    // image that still held the first frame's picture (06:24 run: the spawn view fixed on screen).
    (void)depth;
    const bool copy = (valid & mine) == 0 && (valid & other) != 0;
    if (!copy && !write && (valid & mine) != 0 &&
        g_ss_phase.load(std::memory_order_relaxed) == 2) {
        const u32 age = g_ss_age[idx][scaled ? 0 : 1].load(std::memory_order_relaxed);
        if (age + 1 < g_ss_frame.load(std::memory_order_relaxed)) {
            g_ss_stale_reads.fetch_add(1, std::memory_order_relaxed);
            if (g_ss_stale_by[idx].fetch_add(1, std::memory_order_relaxed) == 0) {
                Checkpoint(fmt::format(
                    "native chain: stale read of {:#x} ({} image, content from frame {}, now {}) by "
                    "a {} after the upscaler",
                    addr, scaled ? "internal-size" : "output-size", age,
                    g_ss_frame.load(std::memory_order_relaxed),
                    g_ss_compute.load(std::memory_order_relaxed) ? "dispatch" : "draw"));
            }
        }
    }
    if (!copy) {
        TwinDone(addr, scaled, write, false);
    }
    return copy;
}

void TwinDone(VAddr addr, bool scaled, bool write, bool copied) {
    const int idx = SceneIndex(addr);
    if (idx < 0) {
        return;
    }
    const u8 mine = scaled ? 1 : 2;
    if (write || copied) {
        g_ss_age[idx][scaled ? 0 : 1].store(g_ss_frame.load(std::memory_order_relaxed),
                                            std::memory_order_relaxed);
    }
    if (write) {
        g_ss_valid[idx].store(mine, std::memory_order_relaxed);
    } else {
        g_ss_valid[idx].fetch_or(mine, std::memory_order_relaxed);
    }
    if (copied) {
        const u32 n = g_ss_twin_copies.fetch_add(1, std::memory_order_relaxed);
        if (n == 0) {
            Checkpoint(fmt::format("native chain: first twin copy {:#x}: {} -> {} ({})", addr,
                                   scaled ? "output size" : "internal size",
                                   scaled ? "internal size" : "output size",
                                   write ? "bound as a target" : "read"));
        }
    }
}

void TwinFailed(VAddr addr, std::string_view why) {
    if (g_ss_twin_fail.fetch_add(1, std::memory_order_relaxed) < 4) {
        Checkpoint(fmt::format("WARNING: native chain: twin copy of {:#x} not possible: {}", addr,
                               why));
    }
}

void BeginDraw(VAddr depth_addr, u32 colors) {
    g_ss_compute.store(false, std::memory_order_relaxed);
    // The scene passes start at the first draw with the scene depth; a G-buffer draw (3 or more
    // colour targets with the scene depth) also starts them after the upscaler, so the phases do
    // not depend on when the flip is seen (0.76).
    const int phase = g_ss_phase.load(std::memory_order_relaxed);
    if (depth_addr != 0 && depth_addr == g_ss_depth.load(std::memory_order_relaxed) &&
        (phase == 0 || (phase == 2 && colors >= 3))) {
        g_ss_phase.store(1, std::memory_order_relaxed);
        g_ss_frame.fetch_add(1, std::memory_order_relaxed);
    }
}

bool NativeScenePhase() {
    return NativePerPass() && g_ss_frozen.load(std::memory_order_relaxed) &&
           !g_ss_off.load(std::memory_order_relaxed) &&
           g_ss_phase.load(std::memory_order_relaxed) == 1;
}

void SetComputeScaled(bool on) {
    g_ss_compute_scaled.store(on, std::memory_order_relaxed);
    g_ss_compute_touched.store(false, std::memory_order_relaxed);
}

bool TakeComputeTouched() {
    return g_ss_compute_touched.exchange(false, std::memory_order_relaxed);
}

void OnContentCheck(double psnr_db) {
    g_ss_psnr_last.store(static_cast<int>(psnr_db * 100.0), std::memory_order_relaxed);
    g_ss_psnr_n.fetch_add(1, std::memory_order_relaxed);
    if (psnr_db < 20.0) g_ss_psnr_low.fetch_add(1, std::memory_order_relaxed);
    static u32 said = 0;
    if (said++ < 3) {
        Checkpoint(fmt::format("native chain content: PSNR {:.1f} dB (the output-size image of the "
                               "source, downscaled, against the internal lit colour; >= 20 ok): {}",
                               psnr_db, psnr_db >= 20.0 ? "ok" : "LOW"));
    }
}

u64 FrameNumber() {
    return g_frame_no;
}

u64 DumpRequest() {
    return g_dump_request.load(std::memory_order_relaxed);
}

bool SceneScaleLive() {
    return g_ss_frozen.load(std::memory_order_acquire) && !g_ss_off.load(std::memory_order_relaxed);
}

int PreferVersion(VAddr addr) {
    if (!g_ss_frozen.load(std::memory_order_acquire)) return -1;
    if (g_ss_off.load(std::memory_order_relaxed) || !NativePerPass()) return 0;
    const int idx = SceneIndex(addr);
    if (idx < 0) return -1;
    return (g_ss_valid[idx].load(std::memory_order_relaxed) & 2) != 0 ? 0 : 1;
}

bool TakeOffPurge(std::vector<std::pair<VAddr, bool>>& addrs) {
    if (!g_ss_purge.exchange(false, std::memory_order_acquire)) {
        return false;
    }
    const u32 n = g_ss_count.load(std::memory_order_acquire);
    addrs.clear();
    for (u32 i = 0; i < n; ++i) {
        // (address, the internal-size image holds the only current content)
        addrs.emplace_back(g_ss_addrs[i], g_ss_valid[i].load(std::memory_order_relaxed) == 1);
    }
    return true;
}

void OnOffPurged(u32 copied, u32 freed) {
    Checkpoint(fmt::format("native chain off: {} addresses back to one output-size image ({} "
                           "internal-size images copied up first, {} freed); no twin, redirect "
                           "or internal-size target is used from here",
                           g_ss_count.load(std::memory_order_relaxed), copied, freed));
}

bool NativePostPhase() {
    return NativePerPass() && g_ss_frozen.load(std::memory_order_relaxed) &&
           !g_ss_off.load(std::memory_order_relaxed) &&
           g_ss_phase.load(std::memory_order_relaxed) == 2;
}

void OnPostScaledTarget(VAddr addr) {
    const u32 n = g_ss_post_scaled.fetch_add(1, std::memory_order_relaxed);
    if (n == 0) {
        Checkpoint(fmt::format("native chain: {:#x} bound as an internal-size target after the "
                               "upscaler (the upscaler's source written again by post-processing; "
                               "counted by the native chain check)",
                               addr));
    }
}

void AddSceneTarget(VAddr addr) {
    if (!g_ss_frozen.load(std::memory_order_acquire) || g_ss_off.load(std::memory_order_relaxed)) {
        return;
    }
    const u32 n = g_ss_count.load(std::memory_order_acquire);
    if (n >= g_ss_addrs.size()) {
        return;
    }
    for (u32 i = 0; i < n; ++i) {
        if (g_ss_addrs[i] == addr) {
            // 0.83: a pinned history image drawn together with internal-size targets: it must be
            // the same size as them, so it goes back to the per-pass versions for good.
            if (g_ss_pinned[i].exchange(false, std::memory_order_relaxed)) {
                g_ss_hist[i].store(1u << 30, std::memory_order_relaxed);
                Checkpoint(fmt::format("native chain: history image {:#x} drawn with internal-size "
                                       "targets: back to per-pass sizes",
                                       addr));
            }
            return;
        }
    }
    bool after_upscale = false, flip = false;
    {
        std::scoped_lock lk{g_mutex};
        after_upscale = g_frame.injected;
        flip = IsFlip(addr);
    }
    // 0.73: with per-pass versions only a scene pass (phase 1) adds a target: a scene target of
    // the guest scene size bound with the internal-size ones (a ring of G-buffer images, an
    // address the learning frame did not use). After the upscaler, or a flip buffer, it is part of
    // the output chain and stays at the output size; scene scaling stays on (0.72 turned it off).
    if (NativePerPass() &&
        (after_upscale || flip || g_ss_phase.load(std::memory_order_relaxed) != 1)) {
        if (g_ss_post_refused.fetch_add(1, std::memory_order_relaxed) < 8) {
            Checkpoint(fmt::format(
                "native chain: {:#x} ({}x{}, {}) bound with internal-size targets outside the "
                "scene passes: output chain, kept at the output size (scene upscale stays on)",
                addr, g_ss_w, g_ss_h, flip ? "a flip buffer" : "output-size target"));
        }
        return;
    }
    g_ss_valid[n].store(2, std::memory_order_relaxed);
    g_ss_class[n].store(0, std::memory_order_relaxed);
    g_ss_addrs[n] = addr;
    g_ss_count.store(n + 1, std::memory_order_release);
    Checkpoint(fmt::format("scene upscale: {:#x} ({}x{}) bound with scaled targets, scaled too "
                           "({} scene targets)",
                           addr, g_ss_w, g_ss_h, n + 1));
    if (BB::ResPatch::NativeChain()) {
        Checkpoint(fmt::format("native chain: {:#x} {}", addr,
                               after_upscale || flip
                                   ? "bound after the upscaler ran / a flip buffer: scaled anyway "
                                     "(BB_SCENE_NATIVE_MIXED=scale)"
                                   : "bound in the scene passes (before the upscaler): scaled "
                                     "there, output-size twin after it"));
    }
}

bool SceneScale(u32& guest_w, u32& guest_h, u32& host_w, u32& host_h) {
    if (!g_ss_frozen.load(std::memory_order_acquire) || g_ss_off.load(std::memory_order_relaxed)) {
        return false;
    }
    guest_w = g_ss_w;
    guest_h = g_ss_h;
    host_w = g_ss_sw;
    host_h = g_ss_sh;
    return true;
}

float LiveSharpness() {
    static std::atomic<float> value{Cfg().sharpness};
    static const char* const path = std::getenv("BB_UPSCALE_SHARPNESS_FILE");
    if (path == nullptr || *path == '\0') {
        return value.load(std::memory_order_relaxed);
    }
    static std::mutex mutex;
    static std::chrono::steady_clock::time_point next{};
    static std::filesystem::file_time_type seen{};
    std::unique_lock lk{mutex, std::try_to_lock};
    if (!lk.owns_lock()) {
        return value.load(std::memory_order_relaxed);
    }
    const auto now = std::chrono::steady_clock::now();
    if (now < next) {
        return value.load(std::memory_order_relaxed);
    }
    next = now + std::chrono::milliseconds(500);
    std::error_code ec;
    const auto stamp = std::filesystem::last_write_time(path, ec);
    if (ec || stamp == seen) {
        return value.load(std::memory_order_relaxed);
    }
    seen = stamp;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto eq = line.find('=');
        const std::string text = eq == std::string::npos ? line : line.substr(eq + 1);
        if (text.empty()) {
            continue;
        }
        char* end = nullptr;
        const float v = std::strtof(text.c_str(), &end);
        if (end != text.c_str() && std::isfinite(v)) {
            const float clamped = std::clamp(v, 0.0f, 1.0f);
            if (clamped != value.load(std::memory_order_relaxed)) {
                LOG_INFO(Render_Vulkan, "[BB_UPSCALE] sharpness {:.2f} (live, {})", clamped, path);
            }
            value.store(clamped, std::memory_order_relaxed);
            break;
        }
    }
    return value.load(std::memory_order_relaxed);
}

bool Enabled() {
    return Cfg().kind != SR::Kind::Off;
}

bool FgNative() {
    return Cfg().kind == SR::Kind::Native && g_fg_native;
}

void PreInstance() {
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
    const auto& c = Cfg();
    // The Windows audio session check (bb_audio_check.h).
    // Now in every mode (native too) so the fork and the official exe can be compared;
    // BB_AUDIO_CHECK=0 turns it off.
    if (const char* ac = Env("BB_AUDIO_CHECK"); ac == nullptr || ac[0] != '0') {
        BB::Audio::StartCheck();
    }
    if (!g_guard_skipped.empty()) {
        Checkpoint(g_guard_skipped);
    }
    if (c.kind == SR::Kind::Off && !Streamline()) {
        return;
    }
    LOG_INFO(Render_Vulkan,
             "[BB_UPSCALE] asked: upscaler {} mode {} (ratio {:.3f}), sharpness {:.2f}, jitter {}, "
             "motion vectors {}, injection {}, HUD-less copy {}",
             SR::KindName(c.kind), SR::QualityName(c.quality),
             SR::QualityRatio(c.kind, c.quality), c.sharpness, c.jitter ? "on" : "off",
             c.mv_reconstruct ? "reconstructed from depth + camera" : "zero",
             c.force_mode == 1 ? "scene" : c.force_mode == 2 ? "composite" : "auto",
             c.hudless_copy ? "on" : "off");
    if (c.mv_upscale_zero) {
        LOG_INFO(Render_Vulkan,
                 "BB_MV_UPSCALE=zero: upscaler motion vectors zeroed, frame generation keeps the "
                 "camera vectors");
    }
    if (FgNative()) {
        Checkpoint(fmt::format(
            "frame generation runs WITHOUT an upscaler (BB_FG={}, BB_UPSCALER={}, BB_FG_NATIVE "
            "on): native image, depth + camera + motion vectors built for frame generation only "
            "(BB_FG_NATIVE=0: the old behaviour, no inputs without an upscaler)",
            Env("BB_FG") ? Env("BB_FG") : "", Env("BB_UPSCALER") ? Env("BB_UPSCALER") : "none"));
    }
    if (c.kind != SR::Kind::Off && c.kind != SR::Kind::Native) {
        GuardBegin(SR::KindName(c.kind), "loading the upscaler and creating the Vulkan device");
    }
    if (Streamline()) {
#if defined(_WIN32)
        // Implicit Vulkan capture layers that crash inside vkCreateDevice under Streamline (owner's
        // log: OBS's we-graphics-hook64.dll, 0xc0000005): turned off for this process only, before
        // the Vulkan loader loads. (RTSS is left on.)
        for (const char* var : {"DISABLE_VULKAN_OBS_CAPTURE", "DISABLE_VK_LAYER_VALVE_steam_overlay_1",
                                "DISABLE_VK_LAYER_VALVE_steam_fossilize_1"}) {
            if (Env(var) == nullptr) {
                _putenv_s(var, "1");
                SetEnvironmentVariableA(var, "1");
            }
        }
        // RTSS's implicit Vulkan layer makes the driver's vkCreateDevice fail when
        // VK_NVX_binary_import (which DLSS needs) is enabled (owner's dry runs, 10-30): off unless
        // BB_DLSS_KEEP_RTSS=1.
        // (10-30: the real cause was the address space NGX needs, see address_space.cpp; RTSS
        // only crashed on the device the driver had refused. Kept on; BB_DLSS_NO_RTSS=1 turns it
        // off.)
        if (Env("BB_DLSS_NO_RTSS") != nullptr && Env("DISABLE_RTSS_LAYER") == nullptr) {
            _putenv_s("DISABLE_RTSS_LAYER", "1");
            SetEnvironmentVariableA("DISABLE_RTSS_LAYER", "1");
            Checkpoint("DLSS: DISABLE_RTSS_LAYER=1 (RTSS's Vulkan layer breaks device creation with "
                       "VK_NVX_binary_import; BB_DLSS_KEEP_RTSS=1 keeps it)");
        }
        if (Env("VK_LOADER_LAYERS_DISABLE") == nullptr) {
            _putenv_s("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_OBS_HOOK");
            SetEnvironmentVariableA("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_OBS_HOOK");
        }
        Checkpoint("DLSS: implicit layers off for this process: DISABLE_VULKAN_OBS_CAPTURE=1, "
                   "DISABLE_VK_LAYER_VALVE_steam_overlay_1=1, "
                   "DISABLE_VK_LAYER_VALVE_steam_fossilize_1=1, "
                   "VK_LOADER_LAYERS_DISABLE=VK_LAYER_OBS_HOOK");
#endif
        // The Vulkan loader's own messages (VK_LOADER_DEBUG) to bb_vkloader.log: the loader writes
        // them to stderr, which is redirected to that file for this run.
        if (Env("VK_LOADER_DEBUG") == nullptr) {
            _putenv_s("VK_LOADER_DEBUG", "error,warn,layer,driver");
            SetEnvironmentVariableA("VK_LOADER_DEBUG", "error,warn,layer,driver");
        }
        if (std::freopen("bb_vkloader.log", "w", stderr) != nullptr) {
            std::setvbuf(stderr, nullptr, _IONBF, 0);
            Checkpoint(fmt::format("DLSS: VK_LOADER_DEBUG={} -> bb_vkloader.log",
                                   Env("VK_LOADER_DEBUG") ? Env("VK_LOADER_DEBUG") : ""));
        }
        Checkpoint("DLSS: loading Streamline (slInit)");
        if (!SR::Dlss::Preload()) {
            Checkpoint("DLSS: Streamline not loaded: DLSS off, the game runs native");
        } else if (const wchar_t* loader = SR::Dlss::LoaderPath()) {
#if defined(_WIN32)
            char narrow[1024]{};
            std::wcstombs(narrow, loader, sizeof(narrow) - 1);
            // (SDL is not pointed at the interposer: shadPS4 makes its surface itself, and the
            // interposer imports vulkan-1.dll anyway, so SDL's load only finds that module.)
            Checkpoint(fmt::format("DLSS: Streamline ready, interposer {}; vulkan-1.dll loaded: {}",
                                   narrow, GetModuleHandleA("vulkan-1.dll") != nullptr));
#endif
        }
    } else if (c.kind == SR::Kind::Xess) {
        SR::Xess::Load();
    } else if (c.kind == SR::Kind::Fsr3) {
        SR::Fsr3::Load();
    }
}

const wchar_t* VulkanLoaderOverride() {
    return Streamline() ? SR::Dlss::LoaderPath() : nullptr;
}

void AddDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                         const std::vector<std::string>& available,
                         std::vector<const char*>& extensions) {
    const auto add = [&](const char* name) {
        const bool has = std::find(available.begin(), available.end(), name) != available.end();
        const bool added = std::any_of(extensions.begin(), extensions.end(),
                                       [&](const char* e) { return std::strcmp(e, name) == 0; });
        if (has && !added) {
            extensions.push_back(name);
        }
        Checkpoint(fmt::format("device extension {} for {}: {}", name, SR::KindName(Cfg().kind),
                               has ? "added" : "NOT available"));
    };
    if (Streamline() && SR::Dlss::Manual()) {
        // Manual hooking: what Streamline's features asked for (slGetFeatureRequirements).
        for (const auto& name : SR::Dlss::RequiredDeviceExtensions()) {
            // (Not the EXT buffer device address with the core / KHR one: both is invalid.)
            if (name == VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) continue;
            add(name.c_str());
        }
    }
    if (Streamline() && SR::Dlss::LoaderPath() != nullptr) {
        // Streamline enables VK_NV_low_latency2 (Reflex) on the device whatever plugins it loads,
        // without the present-id extension that needs: the driver then refuses (or crashes in)
        // vkCreateDevice. The Demon's Souls fork adds it the same way (vulkanWindow.cpp).
        add(VK_KHR_PRESENT_ID_EXTENSION_NAME);
    }
    if (Cfg().kind == SR::Kind::Fsr3) {
        // The FidelityFX backend calls vkGetBufferMemoryRequirements2KHR by the extension's name
        // (null without the extension): the Demon's Souls fork adds both (vulkanWindow.cpp).
        add(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
        add(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
    }
    if (Cfg().kind == SR::Kind::Xess) {
        SR::Xess::AddDeviceExtensions(instance, physical, available, extensions);
    } else if (Cfg().kind == SR::Kind::Fsr4) {
        SR::Fsr4::AddDeviceExtensions(available, extensions);
    }    // 0.42 DLSS 5 (bb_dlss5.h): NGX's two vendor extensions, only with BB_DLSS5 1..3 on NVIDIA.
    Dlss5::AddDeviceExtensions(physical, available, extensions);
}

void* ChainDeviceFeatures(vk::PhysicalDevice physical, const std::vector<const char*>& extensions,
                          vk::PhysicalDeviceVulkan12Features& f12,
                          vk::PhysicalDeviceVulkan13Features& f13, vk::PhysicalDeviceFeatures& f,
                          void* next) {
    if (Streamline()) {
        // The Vulkan 1.2 / 1.3 features Streamline's features ask for, turned on in our own
        // structs where the device has them, so nothing needs to be appended to the chain.
        vk::PhysicalDeviceVulkan13Features s13{};
        vk::PhysicalDeviceVulkan12Features s12{};
        s12.pNext = &s13;
        vk::PhysicalDeviceFeatures2 s{};
        s.pNext = &s12;
        physical.getFeatures2(&s);
#define BB_F12(name)                                                                           \
    {#name, &vk::PhysicalDeviceVulkan12Features::name}
#define BB_F13(name)                                                                           \
    {#name, &vk::PhysicalDeviceVulkan13Features::name}
        static const std::pair<const char*, vk::Bool32 vk::PhysicalDeviceVulkan12Features::*> t12[] = {
            BB_F12(bufferDeviceAddress), BB_F12(timelineSemaphore), BB_F12(descriptorIndexing),
            BB_F12(runtimeDescriptorArray), BB_F12(descriptorBindingPartiallyBound),
            BB_F12(descriptorBindingVariableDescriptorCount),
            BB_F12(descriptorBindingUpdateUnusedWhilePending),
            BB_F12(descriptorBindingSampledImageUpdateAfterBind),
            BB_F12(descriptorBindingStorageImageUpdateAfterBind),
            BB_F12(shaderSampledImageArrayNonUniformIndexing),
            BB_F12(shaderStorageImageArrayNonUniformIndexing), BB_F12(scalarBlockLayout),
            BB_F12(hostQueryReset), BB_F12(shaderFloat16), BB_F12(shaderInt8),
            BB_F12(storageBuffer8BitAccess), BB_F12(uniformAndStorageBuffer8BitAccess),
            BB_F12(vulkanMemoryModel), BB_F12(imagelessFramebuffer), BB_F12(separateDepthStencilLayouts),
            BB_F12(uniformBufferStandardLayout), BB_F12(shaderSubgroupExtendedTypes)};
        static const std::pair<const char*, vk::Bool32 vk::PhysicalDeviceVulkan13Features::*> t13[] = {
            BB_F13(synchronization2), BB_F13(dynamicRendering), BB_F13(maintenance4),
            BB_F13(shaderDemoteToHelperInvocation), BB_F13(subgroupSizeControl),
            BB_F13(computeFullSubgroups), BB_F13(pipelineCreationCacheControl),
            BB_F13(shaderIntegerDotProduct)};
#undef BB_F12
#undef BB_F13
        for (const auto& name : SR::Dlss::RequiredFeatures12()) {
            for (const auto& [n, m] : t12) {
                if (name == n) {
                    const bool has = s12.*m == VK_TRUE;
                    if (has) f12.*m = VK_TRUE;
                    Checkpoint(fmt::format("DLSS: vk12 {} {}", n, has ? "on" : "NOT supported"));
                }
            }
        }
        for (const auto& name : SR::Dlss::RequiredFeatures13()) {
            for (const auto& [n, m] : t13) {
                if (name == n) {
                    const bool has = s13.*m == VK_TRUE;
                    if (has) f13.*m = VK_TRUE;
                    Checkpoint(fmt::format("DLSS: vk13 {} {}", n, has ? "on" : "NOT supported"));
                }
            }
        }
    }
    if (Streamline() && SR::Dlss::LoaderPath() != nullptr) {
        // VK_KHR_present_id enabled for Streamline's Reflex (VK_NV_low_latency2) needs its feature
        // turned on too.
        const bool has = std::any_of(extensions.begin(), extensions.end(), [](const char* e) {
            return std::strcmp(e, VK_KHR_PRESENT_ID_EXTENSION_NAME) == 0;
        });
        if (has) {
            static VkPhysicalDevicePresentIdFeaturesKHR present_id{};
            present_id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR;
            present_id.presentId = VK_TRUE;
            present_id.pNext = next;
            Checkpoint("DLSS: VkPhysicalDevicePresentIdFeaturesKHR{presentId} chained");
            // Not a return: with DLSS-G / Reflex (Streamline) on, XeSS's and FSR 4's own features
            // still have to be chained (FSR 4 was off as 'this GPU lacks ...' under DLSS-G).
            next = &present_id;
        }
    }
    if (Cfg().kind == SR::Kind::Xess) {
        return SR::Xess::ChainDeviceFeatures(extensions, next);
    }
    if (Cfg().kind == SR::Kind::Fsr4) {
        return SR::Fsr4::ChainDeviceFeatures(physical, extensions, f12, f13, f, next);
    }
    return next;
}

void OnDevice(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device,
              VmaAllocator allocator, vk::Queue queue, u32 family) {
    switch (Cfg().kind) {
    case SR::Kind::Dlss:
        SR::Dlss::OnDevice(instance, physical, device);
        break;
    case SR::Kind::Xess:
        SR::Xess::OnDevice(instance, physical, device);
        break;
    case SR::Kind::Fsr3:
        SR::Fsr3::OnDevice(physical, device, VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr);
        break;
    case SR::Kind::Fsr4:
        SR::Fsr4::OnDevice(instance, physical, device, allocator, queue, family);
        break;
    default:
        break;
    }
    Dlss5::OnDevice(instance, physical, device, allocator);
    if (Enabled()) {
        Checkpoint(fmt::format("{} back end {}", SR::KindName(Cfg().kind),
                               BackendReady() ? "ready"
                                              : "NOT available: the game runs native (no upscale)"));
        GuardEnd();
    }
}

namespace {
bool g_backend_dead = false;
using EvaluateFn = bool (*)(const SR::Desc&);

#if defined(_WIN32)
// (No C++ objects in here: SEH only.)
bool CallGuarded(EvaluateFn fn, const SR::Desc* desc, unsigned long* code) {
    __try {
        return fn(*desc);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *code = GetExceptionCode();
        return false;
    }
}
#endif
} // namespace

namespace {
#if defined(_WIN32)
struct FaultInfo {
    unsigned long code = 0;
    void* address = nullptr;
};
int FaultFilter(EXCEPTION_POINTERS* info, FaultInfo* out) {
    out->code = info->ExceptionRecord->ExceptionCode;
    out->address = info->ExceptionRecord->ExceptionAddress;
    return EXCEPTION_EXECUTE_HANDLER;
}
bool CallFn(void (*fn)(void*), void* arg, FaultInfo* fault) {
    __try {
        fn(arg);
        return true;
    } __except (FaultFilter(GetExceptionInformation(), fault)) {
        return false;
    }
}
#endif
} // namespace

std::string Where(const void* p) {
#if defined(_WIN32)
    char module[MAX_PATH] = "?";
    HMODULE handle = nullptr;
    if (p != nullptr &&
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(p), &handle) &&
        handle != nullptr) {
        GetModuleFileNameA(handle, module, MAX_PATH);
        return fmt::format("{} ({}+{:#x})", p, module,
                           reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(handle));
    }
#endif
    return fmt::format("{}", p);
}

void DryRunDevice(VkInstance instance, VkPhysicalDevice physical, const VkDeviceCreateInfo* info) {
#if defined(_WIN32)
    HMODULE vk1 = GetModuleHandleA("vulkan-1.dll");
    if (vk1 == nullptr || info == nullptr) {
        return;
    }
    const auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        reinterpret_cast<void*>(GetProcAddress(vk1, "vkGetInstanceProcAddr")));
    if (gipa == nullptr) {
        return;
    }
    const auto create = reinterpret_cast<PFN_vkCreateDevice>(gipa(instance, "vkCreateDevice"));
    const auto destroy = reinterpret_cast<PFN_vkDestroyDevice>(gipa(instance, "vkDestroyDevice"));
    if (create == nullptr || destroy == nullptr) {
        return;
    }
    struct Try {
        PFN_vkCreateDevice create;
        PFN_vkDestroyDevice destroy;
        VkPhysicalDevice physical;
        VkDeviceCreateInfo ci;
        VkResult result;
    };
    const auto attempt = [&](const std::vector<const char*>& exts, const std::string& label) {
        Try t{create, destroy, physical, *info, VK_ERROR_UNKNOWN};
        t.ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
        t.ci.ppEnabledExtensionNames = exts.data();
        std::string fault;
        const bool ok = GuardedCall(
            [](void* p) {
                auto* x = static_cast<Try*>(p);
                VkDevice d = VK_NULL_HANDLE;
                x->result = x->create(x->physical, &x->ci, nullptr, &d);
                if (x->result == VK_SUCCESS && d != VK_NULL_HANDLE) {
                    x->destroy(d, nullptr);
                }
            },
            &t, fault);
        Checkpoint(fmt::format("DLSS dry run (system loader) {}: {}", label,
                               ok ? vk::to_string(static_cast<vk::Result>(t.result))
                                  : "CRASH " + fault));
        return ok && t.result == VK_SUCCESS;
    };
    std::vector<const char*> base(info->ppEnabledExtensionNames,
                                  info->ppEnabledExtensionNames + info->enabledExtensionCount);
    attempt(base, "as we create it");
    std::vector<const char*> added;
    for (const auto& name : SR::Dlss::RequiredDeviceExtensions()) {
        if (std::none_of(base.begin(), base.end(), [&](const char* e) { return name == e; }) &&
            std::none_of(added.begin(), added.end(), [&](const char* e) { return name == e; })) {
            added.push_back(name.c_str());
        }
    }
    auto all = base;
    all.insert(all.end(), added.begin(), added.end());
    if (attempt(all, "with Streamline's extensions")) {
        return;
    }
    for (std::size_t i = 0; i < added.size(); ++i) {
        auto without = base;
        for (std::size_t j = 0; j < added.size(); ++j) {
            if (j != i) without.push_back(added[j]);
        }
        attempt(without, fmt::format("with Streamline's extensions but {}", added[i]));
    }
    // The same on a throwaway instance made without RTSS's layer (when the real one has it).
    if (Env("DISABLE_RTSS_LAYER") == nullptr) {
        _putenv_s("DISABLE_RTSS_LAYER", "1");
        SetEnvironmentVariableA("DISABLE_RTSS_LAYER", "1");
        const auto create_instance =
            reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        VkInstance inst = VK_NULL_HANDLE;
        if (create_instance != nullptr && create_instance(&ici, nullptr, &inst) == VK_SUCCESS) {
            const auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
                gipa(inst, "vkEnumeratePhysicalDevices"));
            const auto destroy_instance =
                reinterpret_cast<PFN_vkDestroyInstance>(gipa(inst, "vkDestroyInstance"));
            uint32_t n = 1;
            VkPhysicalDevice pd = VK_NULL_HANDLE;
            enumerate(inst, &n, &pd);
            if (pd != VK_NULL_HANDLE) {
                const auto create2 = reinterpret_cast<PFN_vkCreateDevice>(gipa(inst, "vkCreateDevice"));
                const auto destroy2 = reinterpret_cast<PFN_vkDestroyDevice>(gipa(inst, "vkDestroyDevice"));
                Try t{create2, destroy2, pd, *info, VK_ERROR_UNKNOWN};
                t.ci.enabledExtensionCount = static_cast<uint32_t>(all.size());
                t.ci.ppEnabledExtensionNames = all.data();
                std::string fault;
                const bool ok = GuardedCall(
                    [](void* p) {
                        auto* x = static_cast<Try*>(p);
                        VkDevice d = VK_NULL_HANDLE;
                        x->result = x->create(x->physical, &x->ci, nullptr, &d);
                        if (x->result == VK_SUCCESS && d != VK_NULL_HANDLE) x->destroy(d, nullptr);
                    },
                    &t, fault);
                Checkpoint(fmt::format("DLSS dry run (system loader, new instance WITHOUT the RTSS "
                                       "layer) with Streamline's extensions: {}",
                                       ok ? vk::to_string(static_cast<vk::Result>(t.result))
                                          : "CRASH " + fault));
            }
            destroy_instance(inst, nullptr);
        }
        _putenv_s("DISABLE_RTSS_LAYER", "");
        SetEnvironmentVariableA("DISABLE_RTSS_LAYER", nullptr);
    }
    for (std::size_t i = 0; i < base.size(); ++i) {
        auto without = all;
        without.erase(without.begin() + static_cast<std::ptrdiff_t>(i));
        attempt(without, fmt::format("with Streamline's extensions, without our {}", base[i]));
    }
#endif
}

bool GuardedCall(void (*fn)(void*), void* arg, std::string& fault) {
#if defined(_WIN32)
    FaultInfo info{};
    if (CallFn(fn, arg, &info)) {
        return true;
    }
    char module[MAX_PATH] = "?";
    HMODULE handle = nullptr;
    uintptr_t offset = 0;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(info.address), &handle) &&
        handle != nullptr) {
        GetModuleFileNameA(handle, module, MAX_PATH);
        offset = reinterpret_cast<uintptr_t>(info.address) - reinterpret_cast<uintptr_t>(handle);
    }
    fault = fmt::format("exception {:#x} at {} ({}+{:#x})", info.code, info.address, module, offset);
    return false;
#else
    fn(arg);
    return true;
#endif
}

bool RunBackend(const SR::Desc& desc) {
    if (g_backend_dead) {
        return false;
    }
    EvaluateFn fn = nullptr;
    switch (Cfg().kind) {
    case SR::Kind::Dlss:
        fn = &SR::Dlss::Evaluate;
        break;
    case SR::Kind::Xess:
        fn = &SR::Xess::Evaluate;
        break;
    case SR::Kind::Fsr3:
        fn = &SR::Fsr3::Evaluate;
        break;
    case SR::Kind::Fsr4:
        fn = &SR::Fsr4::Evaluate;
        break;
    default:
        return false;
    }
    static u32 failures = 0;
#if defined(_WIN32)
    unsigned long code = 0;
    const bool ok = CallGuarded(fn, &desc, &code);
    if (code != 0) {
        g_backend_dead = true;
        Checkpoint(fmt::format("{} crashed inside its dispatch (exception {:#x}): turned off, the "
                               "game runs native",
                               SR::KindName(Cfg().kind), code));
        return false;
    }
#else
    const bool ok = fn(desc);
#endif
    if (!ok && ++failures >= 8) {
        g_backend_dead = true;
        Checkpoint(fmt::format("{} failed {} times: turned off, the game runs native",
                               SR::KindName(Cfg().kind), failures));
    }
    if (ok) failures = 0;
    return ok;
}

bool BackendReady() {
    if (g_backend_dead) return false;
    static int cached = -1;
    switch (Cfg().kind) {
    case SR::Kind::Dlss:
        return SR::Dlss::Ready();
    case SR::Kind::Xess:
        return SR::Xess::Ready();
    case SR::Kind::Fsr3:
        return SR::Fsr3::Ready();
    case SR::Kind::Fsr4:
        if (cached < 0) cached = SR::Fsr4::Ready() ? 1 : 0;
        return cached == 1;
    default:
        return false;
    }
}

int RtReflectLevel() {
    static const int level = [] {
        const char* s = Env("BB_RT_REFLECT");
        return s != nullptr ? std::atoi(s) : 0;
    }();
    return level;
}

bool RtReflect() {
    return RtReflectLevel() >= 1;
}

void LogGbuffer(const FrameState& f) {
    static u64 flips = 0;
    if (!RtReflect() || f.gb_depth == 0 || flips++ % 600 != 0) {
        return;
    }
    Checkpoint(fmt::format("rt-reflect: G-buffer pass {}x{}, {} targets, {} draws, depth {}, "
                           "last draw vs {:#018x} ps {:#018x}",
                           f.gb_w, f.gb_h, f.gb_mrt, f.gb_draws, Describe(f.gb_depth_target),
                           f.gb_vs_hash, f.gb_ps_hash));
    for (u32 i = 0; i < f.gb_num_colors; ++i) {
        if (f.gb_colors[i].addr == 0)
            continue;
        Checkpoint(fmt::format("rt-reflect:   RT{} {} (ps {:#018x})", i, Describe(f.gb_colors[i]),
                               f.gb_ps_hash));
    }
}

void OnDraw(std::span<const Target> colors, const Target* depth, bool depth_greater,
            const Viewport& viewport, u64 vs_hash, u64 ps_hash) {
    if (!Enabled() && !AnalyzeOn() && !RtReflect()) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    auto& f = g_frame;
    ++f.draws;
    g_draw = {};
    g_draw.valid = true;
    g_draw.num_colors = static_cast<u32>(std::min<size_t>(colors.size(), 8));
    u32 mrt = 0;
    for (u32 i = 0; i < g_draw.num_colors; ++i) {
        g_draw.colors[i] = colors[i];
        if (colors[i].addr != 0) ++mrt;
        if (IsFlip(colors[i].addr)) g_draw.targets_flip = true;
        const u64 area = u64(colors[i].width) * colors[i].height;
        if (colors[i].addr != 0 && area > u64(g_stats.big_w) * g_stats.big_h) {
            g_stats.big_w = colors[i].width;
            g_stats.big_h = colors[i].height;
        }
        if (colors[i].addr != 0 && g_plan.render_w != 0 &&
            (colors[i].width > g_plan.render_w || colors[i].height > g_plan.render_h)) {
            ++g_stats.big_draws;
        }
    }
    g_draw.has_depth = depth != nullptr && depth->addr != 0;
    if (g_draw.has_depth) g_draw.depth = *depth;

    // The G-buffer pass: the most colour targets with a depth target (3 or more).
    if (g_draw.has_depth && mrt >= 3 && mrt >= f.gb_mrt) {
        f.gb_depth = depth->addr;
        f.gb_w = depth->width;
        f.gb_h = depth->height;
        f.gb_mrt = mrt;
        f.depth_greater = depth_greater;
        f.viewport = viewport;
        f.gb_colors = g_draw.colors;
        f.gb_num_colors = g_draw.num_colors;
        f.gb_depth_target = *depth;
        f.gb_vs_hash = vs_hash;
        f.gb_ps_hash = ps_hash;
        ++f.gb_draws;
    }
    const VAddr scene_depth = f.gb_depth != 0 ? f.gb_depth : g_learned.gb_depth;
    g_draw.scene = !f.injected && g_draw.has_depth && depth->addr == scene_depth;
    if (g_draw.scene && SceneScaleWanted() && !g_ss_frozen.load(std::memory_order_relaxed)) {
        const auto add = [](VAddr a) {
            if (a != 0 && g_ss_candidates.size() < 16 &&
                std::find(g_ss_candidates.begin(), g_ss_candidates.end(), a) ==
                    g_ss_candidates.end()) {
                g_ss_candidates.push_back(a);
            }
        };
        add(depth->addr);
        for (u32 i = 0; i < g_draw.num_colors; ++i) {
            if (colors[i].width == depth->width && colors[i].height == depth->height) {
                add(colors[i].addr);
            }
        }
    }
    // 0.73 motion-vector probe: a game-made velocity buffer would be a two-channel target of the
    // scene size (R16G16 / R8G8). Every such target drawn is logged once with the pass it is in;
    // the vectors stay the in-emulator ones unless the game is shown to make its own.
    for (u32 i = 0; i < g_draw.num_colors; ++i) {
        const Target& c = colors[i];
        const bool two = c.format == vk::Format::eR16G16Sfloat ||
                         c.format == vk::Format::eR16G16Snorm ||
                         c.format == vk::Format::eR16G16Unorm ||
                         c.format == vk::Format::eR8G8Snorm || c.format == vk::Format::eR8G8Unorm ||
                         c.format == vk::Format::eR32G32Sfloat;
        if (!two || c.addr == 0 || f.gb_w == 0) continue;
        static std::array<VAddr, 16> seen{};
        static u32 n_seen = 0;
        if (n_seen >= seen.size() ||
            std::find(seen.begin(), seen.begin() + n_seen, c.addr) != seen.begin() + n_seen) {
            continue;
        }
        seen[n_seen++] = c.addr;
        Checkpoint(fmt::format(
            "[mv-probe] two-channel target {} (c{}) drawn {}, {} the upscaler, draw {} (vs {:#x} "
            "ps {:#x}); scene size {}x{}: {}",
            Describe(c), i,
            g_draw.has_depth && depth->addr == scene_depth ? "with the scene depth"
            : g_draw.has_depth                              ? "with another depth"
                                                            : "without depth",
            f.injected ? "after" : "before", f.draws, vs_hash, ps_hash, f.gb_w, f.gb_h,
            c.width == f.gb_w && c.height == f.gb_h ? "a velocity buffer candidate"
                                                    : "not the scene size"));
    }
    if (g_draw.scene) {
        ++f.scene_draws;
        // The scene colour: the lit image the forward passes draw into (one or two targets).
        if (mrt <= 2 && colors.size() > 0 && IsHdr(colors[0].format) &&
            colors[0].width == depth->width && colors[0].height == depth->height) {
            f.scene_color = colors[0].addr;
            if (colors[0].addr == g_good_src) f.good_src_drawn = true;
        }
    }
    // Writes to the injection source.
    const VAddr src = LearnedSource();
    for (u32 i = 0; i < g_draw.num_colors; ++i) {
        if (src != 0 && colors[i].addr == src) {
            // 0.84: only writes before the upscaler move the injection point. Post-processing
            // reusing the scene colour's memory after it (19:13 run: 'learned skip 298', 93 of
            // 300 frames, '0 with later samples redirected') raised the learned skip past the
            // real point, so the next frames waited for a sample that never came.
            if (!f.injected) f.last_write_after = f.qual_samples;
            if (f.injected && f.redirecting) {
                ++f.src_rewrites;
                f.redirecting = false;
            }
        }
    }
    // HUD stage: draws into the presented image, and output-size draws after the injection.
    if (g_draw.targets_flip) {
        ++f.flip_draws;
    }
    if (g_dump_frame && g_dump_lines < 8000) {
        g_draw.analyze_dump = true;
        std::string line = fmt::format("[BB_ANALYZE] frame={} draw {} targets:", g_frame_no, f.draws);
        for (u32 i = 0; i < g_draw.num_colors; ++i) {
            if (colors[i].addr) line += fmt::format(" c{}={}", i, Describe(colors[i]));
        }
        if (g_draw.has_depth) line += fmt::format(" d={}", Describe(*depth));
        if (g_draw.targets_flip) line += " [FLIP]";
        if (g_draw.scene) line += " [SCENE]";
        const std::string key = line.substr(line.find("targets:"));
        if (key == g_dump_last) {
            ++g_dump_repeats;
            g_draw.analyze_dump = false;
        } else {
            if (g_dump_repeats > 0) {
                AnalyzeLog("[BB_ANALYZE]   ... {} more draw(s) with the same targets",
                         g_dump_repeats);
            }
            g_dump_repeats = 0;
            g_dump_last = key;
            AnalyzeLog("{}", line);
            ++g_dump_lines;
        }
    }
}

void OnDispatch() {
    g_ss_compute.store(true, std::memory_order_relaxed); // 0.73: dispatches use output-size twins
    if (!Enabled() && !AnalyzeOn()) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    ++g_frame.draws;
    g_draw = {};
    g_draw.valid = true;
    g_draw.compute = true;
    if (g_dump_frame && g_dump_lines < 8000) {
        g_draw.analyze_dump = true;
        AnalyzeLog("[BB_ANALYZE] frame={} dispatch {}", g_frame_no, g_frame.draws);
        ++g_dump_lines;
    }
}

static bool all_scene_on();

u64 g_diag_asked = 0, g_diag_inactive = 0, g_diag_noscene = 0, g_diag_calls = 0, g_diag_small = 0;

bool WantCameraScan(bool vertex) {
    g_scan_vertex = vertex;
    ++g_diag_asked;
    if ((!Active() && !AnalyzeOn() && !FgNative()) || !Cfg().mv_reconstruct) {
        ++g_diag_inactive;
        return false;
    }
    std::scoped_lock lk{g_mutex};
    // 1.12 (bb-cpcpu-112): with BB_TAA_ALL (the default) Jitter() never reads vertex_camera, so
    // the jitter term no longer rescans every scene draw's vertex constants (up to 4 KB, 2000
    // matrix tests each) once the frame's camera is known.
    const bool want = g_draw.valid && g_draw.scene &&
                      (g_frame.scanned < 160 || !g_frame.camera_ok ||
                       (vertex && Cfg().jitter && !g_draw.vertex_camera &&
                        (!all_scene_on() || g_draw.analyze_dump)));
    if (!want) ++g_diag_noscene;
    return want;
}

void OnVertexConstants(const void* data, size_t size) {
    ++g_diag_calls;
    if (data == nullptr || size < 64) {
        ++g_diag_small;
        return;
    }
    std::scoped_lock lk{g_mutex};
    auto& f = g_frame;
    if (BlockCamera(f, static_cast<const float*>(data), size)) {
        return;
    }
    // 1.12 (bb-cpcpu-112): once the scene constant block gave the frame's camera (a candidate of
    // count 1000000 no scanned one can outrank, BestCamera), the matrix scan only still matters
    // for the vertex-camera mark, which only Jitter() without BB_TAA_ALL reads. The block test
    // above keeps running, so a larger pass's block still replaces it.
    if (f.block_area > 0 && f.camera_ok && !g_draw.analyze_dump &&
        (all_scene_on() || !g_scan_vertex || g_draw.vertex_camera)) {
        return;
    }
    const u64 draw = f.draws;
    bool counted_draw = f.scanned_draw == draw;
    if (g_draw.analyze_dump && g_dump_lines < 8000 && g_dump_cbs < 64) {
        ++g_dump_cbs;
        const float* v = static_cast<const float*>(data);
        std::string text;
        for (size_t i = 0; i < std::min<size_t>(size / 4, 48); ++i) {
            text += fmt::format("{}{:.4g}", i % 4 == 0 ? (i ? " | " : "") : " ", v[i]);
        }
        AnalyzeLog("[BB_ANALYZE]   cb {:#x} size {} : {}",
                 reinterpret_cast<uintptr_t>(data), size, text);
        ++g_dump_lines;
    }
    const size_t n = std::min<size_t>(size, 4096) / 4;
    const float* w = static_cast<const float*>(data);
    const bool jitter_only = f.scanned >= 160 && f.camera_ok;
    for (size_t at = 0; at + 16 <= n; at += 4) {
        for (int layout = 0; layout < 2; ++layout) {
            Mat m;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    m.m[i * 4 + j] = layout == 0 ? w[at + i * 4 + j] : w[at + j * 4 + i];
            bool hit = LooksLikeViewProj(m);
            bool inverse = false;
            if (hit && g_scan_vertex) {
                g_draw.vertex_camera = true;
                if (jitter_only) return;
            }
            if (jitter_only) {
                continue;
            }
            if (!hit && std::fabs(m.m[15]) + std::fabs(m.m[11]) + std::fabs(m.m[14]) > 1e-9) {
                // The inverse view-projection (deferred lighting's position reconstruction).
                Mat inv;
                if (Invert(m, inv) && LooksLikeViewProj(inv)) {
                    m = inv;
                    hit = true;
                    inverse = true;
                }
            }
            if (!hit) {
                continue;
            }
            if (g_draw.analyze_dump && g_dump_lines < 8000) {
                AnalyzeLog(
                         "[BB_ANALYZE]   camera candidate in cb {:#x}+{:#x} ({} layout{}): "
                         "{:.4g} {:.4g} {:.4g} {:.4g} / {:.4g} {:.4g} {:.4g} {:.4g} / {:.4g} "
                         "{:.4g} {:.4g} {:.4g} / {:.4g} {:.4g} {:.4g} {:.4g}",
                         reinterpret_cast<uintptr_t>(data), at * 4, layout == 0 ? "row" : "column",
                         inverse ? ", inverse" : "", m.m[0], m.m[1], m.m[2], m.m[3], m.m[4],
                         m.m[5], m.m[6], m.m[7], m.m[8], m.m[9], m.m[10], m.m[11], m.m[12],
                         m.m[13], m.m[14], m.m[15]);
                ++g_dump_lines;
            }
            if (!counted_draw) {
                ++f.scanned;
                f.scanned_draw = draw;
                counted_draw = true;
            }
            auto it = std::find_if(f.camera.begin(), f.camera.end(), [&](const Candidate& c) {
                for (int k = 0; k < 16; ++k)
                    if (std::fabs(c.m.m[k] - m.m[k]) > 1e-5 * (1.0 + std::fabs(m.m[k])))
                        return false;
                return true;
            });
            if (it == f.camera.end()) {
                if (f.camera.size() < 64) f.camera.push_back({m, 1, draw});
            } else if (it->last_draw != draw) {
                ++it->count;
                it->last_draw = draw;
                if (it->count >= 2 && std::fabs(m.m[3]) + std::fabs(m.m[7]) + std::fabs(m.m[15]) > 1e-3) {
                    f.camera_ok = true;
                }
            }
        }
    }
    if (!counted_draw) {
        ++f.scanned;
        f.scanned_draw = draw;
    }
}

Sample OnSample(VAddr addr, u32 width, u32 height, vk::Format format) {
    if (!Enabled() && !AnalyzeOn()) {
        return Sample::Keep;
    }
    std::scoped_lock lk{g_mutex};
    auto& f = g_frame;
    if (g_draw.analyze_dump && g_dump_lines < 8000) {
        AnalyzeLog("[BB_ANALYZE]   samples {:#x} {}x{} {}{}{}", addr, width, height,
                 vk::to_string(format), addr == f.scene_color ? " [SCENE COLOUR]" : "",
                 addr == f.gb_depth ? " [SCENE DEPTH]" : "");
        ++g_dump_lines;
    }
    bool targets_it = false;
    for (u32 i = 0; i < g_draw.num_colors; ++i) {
        if (g_draw.colors[i].addr == addr) targets_it = true;
    }
    if (!targets_it && f.gb_w != 0 && width == f.gb_w && height == f.gb_h) {
        g_draw.samples_render_size = true;
    }
    if (!IsColorSampleable(format)) {
        return Sample::Keep;
    }
    const bool out_size = g_draw.num_colors > 0 && g_draw.colors[0].width == g_output_w &&
                          g_draw.colors[0].height == g_output_h;
    // Composite candidate: an output-size draw sampling a render-size image after the scene.
    if (f.comp_src == 0 && !g_draw.compute && out_size && !targets_it && f.gb_w != 0 &&
        width == f.gb_w && height == f.gb_h && (f.gb_w != g_output_w || f.gb_h != g_output_h)) {
        f.comp_src = addr;
        f.comp_w = width;
        f.comp_h = height;
    }
    const VAddr src = LearnedSource();
    if (src == 0 || addr != src || targets_it || g_draw.scene) {
        return Sample::Keep;
    }
    if (f.injected) {
        if (f.redirecting) {
            g_draw.sampled_upscaled = true;
            ++f.native_redirects;
            return Sample::Redirect;
        }
        return Sample::Keep;
    }
    // A qualifying sample of the source: the injection is at the one after its last write.
    const u32 index = f.qual_samples++;
    // 0.76 native chain: a refused qualifying sample is counted by reason (native chain check
    // line), and, unless it waits for a later sample (learned skip), it still ends the scene
    // passes: post-processing then works on output-size images instead of scaled ones.
    const bool chain_live = NativePerPass() && g_ss_frozen.load(std::memory_order_relaxed) &&
                            !g_ss_off.load(std::memory_order_relaxed);
    const auto refuse = [&](Refuse why) {
        f.miss = static_cast<int>(why);
        if (chain_live) {
            g_ss_refuse[why].fetch_add(1, std::memory_order_relaxed);
            if (why != RfSkip && g_ss_phase.load(std::memory_order_relaxed) == 1) {
                g_ss_source.store(addr, std::memory_order_relaxed);
                g_ss_phase.store(2, std::memory_order_relaxed);
            }
        }
        return Sample::Keep;
    };
    if (PathLockOn() && g_lock == 2) {
        return Sample::Keep; // 0.85: the session is locked to the window path
    }
    if (PresentMode() || (HudPointOn() && WindowPathOk())) {
        return refuse(RfPresent); // upscaled before the HUD instead (WantPresentUpscale)
    }
    // 0.76: no native-fallback hold in the native chain: the in-place point is the only place
    // the internal-size scene is upscaled, a hold would leave it unscaled for 240 flips.
    if (g_native_hold > 0 && NativeFallback() && !chain_live) {
        return Sample::Keep; // 0.48: native fallback hold (window-size path at the same size)
    }
    // 0.84: once the in-place point worked, its sample index is kept (the learned skip is the
    // largest over 240 frames, so frames with fewer samples of the source missed it and fell back
    // to the present-time path: the 19:07 run switched between the two paths every few frames).
    const u32 need = (g_good_src != 0 && addr == g_good_src) ? std::min(g_learned.skip, g_good_index)
                                                            : g_learned.skip;
    if (index < need) {
        return refuse(RfSkip);
    }
    if (!Active() || f.inject_failed) {
        return refuse(!Active() ? RfInactive : RfFailed);
    }
    if (width != (f.gb_w ? f.gb_w : g_learned.render_w) ||
        height != (f.gb_h ? f.gb_h : g_learned.render_h)) {
        return refuse(RfSize);
    }
    const bool scene = SceneMode();
    // 0.76: in the native chain the draw that first reads the scene colour may draw into an
    // address shared with a scene target (looked up before the upscaler ran, so internal size);
    // the upscale still goes to the output size (the game's guest size).
    const bool chain_point = chain_live && !scene && width == g_ss_sw && height == g_ss_sh;
    if (!scene && !out_size && !chain_point) {
        return refuse(RfTarget);
    }
    // The plan.
    Plan& p = g_plan;
    p = {};
    p.scene_mode = scene;
    p.color_addr = addr;
    p.depth_addr = f.gb_depth != 0 ? f.gb_depth : g_learned.gb_depth;
    p.render_w = width;
    p.render_h = height;
    p.output_w = scene ? width : chain_point ? g_ss_w : g_draw.colors[0].width;
    p.output_h = scene ? height : chain_point ? g_ss_h : g_draw.colors[0].height;
    p.hdr = IsHdr(format);
    // 0.84: the history is reset only when the in-place point has not run for 30 frames (or the
    // last upscale was the present-time path's post-processed image), not after one missed frame.
    p.reset = g_good_frame == 0 || g_frame_no > g_good_frame + 30;
    if (NativeFallbackOn() && g_last_window_path && !(PathLockOn() && g_lock == 1)) {
        p.reset = true; // (the history was the window-size path's post-processed image)
    }
    p.delta_ms = g_delta_ms;
    p.viewport = f.viewport;
    p.depth_inverted = f.depth_greater;
    FillCamera(p);
    SrReset(p);
    if (const char* inv = std::getenv("BB_DEPTH_INVERT"); inv != nullptr && *inv != '\0') {
        p.depth_inverted = inv[0] == '1';
    }
    const bool jitter_on =
        Cfg().jitter && p.camera_valid && g_jittered_draws > g_jittered_frame_start;
    p.jitter[0] = jitter_on ? g_jitter[0] : 0.0f;
    p.jitter[1] = jitter_on ? g_jitter[1] : 0.0f;
    LogRtShadowCamera(p);
    const bool chain = !scene && g_ss_frozen.load(std::memory_order_relaxed) &&
                       !g_ss_off.load(std::memory_order_relaxed) && width == g_ss_sw &&
                       height == g_ss_sh;
    LogPoint(fmt::format(
        "injection point: {} mode, source {:#x} {}x{} {} ({}), depth {:#x}, output {}x{}, "
        "after {} earlier sample(s) of it, motion vectors {}, depth {}",
        scene   ? "scene (pre-post HDR)"
        : chain ? "native chain (scene targets at the internal size, upscaled where "
                  "post-processing first reads the scene colour)"
                : "composite (post-tonemap)",
        addr, width, height,
        vk::to_string(format), p.hdr ? "HDR" : "LDR", p.depth_addr, p.output_w, p.output_h,
        g_learned.skip, p.camera_valid ? "reconstructed (camera found)" : "zero (no camera)",
        p.depth_inverted ? "inverted" : "forward"));
    static bool first = true;
    if (first) {
        first = false;
        GuardBegin(SR::KindName(Cfg().kind), "the first upscale");
        Checkpoint(fmt::format("{}: first upscale {}x{} -> {}x{}", SR::KindName(Cfg().kind),
                               p.render_w, p.render_h, p.output_w, p.output_h));
        LOG_INFO(Render_Vulkan, "[BB_UPSCALE] upscale {}x{} -> {}x{} (window {}x{}){}",
                 p.render_w, p.render_h, p.output_w, p.output_h, g_present_w, g_present_h,
                 p.render_w == p.output_w && p.render_h == p.output_h
                     ? ": same size, native AA only (the game already renders at this size)"
                     : "");
        Checkpoint(fmt::format("upscale {}x{} -> {}x{} (window {}x{})", p.render_w, p.render_h,
                               p.output_w, p.output_h, g_present_w, g_present_h));
        g_first_eval_pending = true;
    }
    return Sample::Inject;
}

void SetPresentSize(u32 width, u32 height) {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    if (width != g_present_w || height != g_present_h) {
        LOG_INFO(Render_Vulkan, "[BB_UPSCALE] presentation (window) size {}x{}", width, height);
    }
    g_present_w = width;
    g_present_h = height;
}

namespace {
// 0.36: BB_3D_SKIP_RELAX=1: the window-size path needs one sample of the source, not more than the
// learned in-place skip (the largest write count of the last 240 frames, which stays high for 240
// frames after one busy frame and then rejects ordinary gameplay frames).
bool SkipRelax() {
    static const bool on = [] {
        const char* v = std::getenv("BB_3D_SKIP_RELAX");
        return v == nullptr || *v != '0'; // 0.38: on by default (the user log proved it)
    }();
    return on;
}
u32 SkipNeeded() {
    return SkipRelax() ? 0u : g_learned.skip;
}
// 0.40: BB_UPFIX (default on; BB_UPFIX=0 restores 0.38). The window-size path upscales the
// finished post-processed image, after every write to the source, so the in-place sample count
// does not apply to it: at present (the finished flip, no HUD point found) no sample of the
// source is needed at all, and a 3D frame that still misses the upscaler is scaled to the window
// with the spatial FSR 1 pass instead of a plain bilinear stretch (SceneSpatialFallback).
bool UpfixOn() {
    static const bool on = [] {
        const char* v = std::getenv("BB_UPFIX");
        return v == nullptr || *v != '0';
    }();
    return on;
}
bool LateSamplesOk(u32 qual_samples) {
    return UpfixOn() || qual_samples > SkipNeeded();
}
// Per-reason counts of frames not upscaled (not 3D), printed about once a second.
struct Not3D {
    u32 frames = 0, up = 0, no_depth = 0, no_scene = 0, size = 0, samples = 0, failed = 0,
        no_point = 0, max_q = 0, max_skip = 0;
    // 0.48: how the upscaled ones went (in place, window-size before the HUD, at present), and
    // the in-place counts of the frames with no point (samples of the source vs the learned skip).
    u32 inplace = 0, hud = 0, late = 0, np_max_q = 0, np_max_skip = 0, np_src_moved = 0;
} g_not3d;
bool g_native_said = false;
// 1.05 (bb-fgpure-105): frame generation with no upscaler (FgNative) had no path lock: the 0.85
// lock counts f.injected, which the FG-only analysis never sets, so every 3D frame chose again
// between the HUD point (mid-frame, HUD-less image) and the finished flip at present (HUD
// included, vectors composed late), and a frame with neither gave frame generation no inputs
// (DLSS-G / FSR 3 FG: no generated frame, then a history reset on the next one). The FG-only
// path is now locked for the session from the first 300 3D frames like the upscaler's:
// 1 = before the HUD (>= 95% of frames had the point; a miss still takes the present-time inputs
// so no frame is without inputs), 2 = at present for every frame (the HUD point is not used).
// BB_FGN_LOCK=hud / present forces one, BB_FGN_LOCK=0 the old per-frame choice.
int g_fgn_lock = 0;
u32 g_inject_traced = 0; // 1.16: [inject] traces logged
u32 g_fgn_lock_frames = 0, g_fgn_lock_hud = 0;
int FgnLockForced() {
    static const int v = [] {
        const char* e = std::getenv("BB_FGN_LOCK");
        if (e == nullptr || *e == '\0') return -1;
        const std::string_view s{e};
        if (s == "hud") return 1;
        if (s == "present") return 2;
        if (s == "0") return 0;
        return -1;
    }();
    return v;
}
struct FgnSec {
    u32 frames = 0, hud = 0, late = 0, none = 0, fail = 0, resets = 0, switches = 0, cam = 0;
    int last = 0;
    double int_sum = 0, int_sq = 0, int_min = 1e9, int_max = 0;
    u32 ints = 0;
} g_fgn_sec;
} // namespace

namespace {
std::atomic<bool> g_hud_canvas_ok{true};
}
bool HudCanvasOk() {
    return g_hud_canvas_ok.load(std::memory_order_relaxed);
}
void SetHudCanvasOk(bool ok, std::string_view why) {
    if (g_hud_canvas_ok.exchange(ok) != ok) {
        Checkpoint(fmt::format("[hud] HUD canvas {} for the session: {}", ok ? "on" : "OFF", why));
    }
}

bool WantPresentUpscale(VAddr& flip) {
    if (!Enabled()) {
        return false;
    }
    std::scoped_lock lk{g_mutex};
    auto& f = g_frame;
    const bool fgn = FgNative();
    if (!g_draw.valid || g_draw.compute) {
        return false;
    }
    // 0.97 shared injection point (the DCport placement: scene -> the game's post-processing ->
    // upscaler / DLAA / DLSS 5 / frame generation inputs -> HUD). Generic, from the command stream:
    // after the scene (G-buffer depth and scene draws), the post-processing passes are the
    // output-size draws without depth that sample a render-size image; the last one's target is
    // the composite target (the flip, or the game's UI image). The HUD starts at the first later
    // draw into that target that samples no render-size image (2D, no depth). A frame where that
    // draw is not seen uses the last known HUD start (its draw index and target).
    VAddr cand = 0;
    const bool after_scene = f.gb_depth != 0 && f.scene_draws > 0;
    if (after_scene && !g_draw.has_depth) {
        for (u32 i = 0; i < g_draw.num_colors; ++i) {
            const auto& t = g_draw.colors[i];
            if (t.addr == 0 || t.width != f.gb_w || t.height != f.gb_h) continue;
            if (g_draw.samples_render_size) {
                f.post_target = t.addr;
                if (std::find(f.post_targets.begin(),
                              f.post_targets.begin() + f.num_post_targets,
                              t.addr) == f.post_targets.begin() + f.num_post_targets &&
                    f.num_post_targets < f.post_targets.size()) {
                    f.post_targets[f.num_post_targets++] = t.addr;
                }
                f.post_draw = f.draws;
                f.post_seen = true;
            } else if (cand == 0 &&
                       (t.addr == f.post_target || IsFlip(t.addr) ||
                        std::find(f.post_targets.begin(),
                                  f.post_targets.begin() + f.num_post_targets,
                                  t.addr) != f.post_targets.begin() + f.num_post_targets)) {
                cand = t.addr;
                f.hud_how = t.addr == f.post_target ? "first 2D draw into the composite target"
                                                    : "first 2D draw into the presented image";
            }
            if (g_inject_traced < 6 && f.trace.size() < 1500) {
                f.trace += fmt::format(" #{}:{:#x}{}{}", f.draws, t.addr,
                                       g_draw.samples_render_size ? "/post" : "/2d",
                                       IsFlip(t.addr) ? "/flip" : "");
            }
            break;
        }
        // The last known HUD start: this draw index (or later) into the same target.
        if (cand == 0 && g_learned.hud_draw != 0 && f.draws >= g_learned.hud_draw &&
            f.post_seen) {
            for (u32 i = 0; i < g_draw.num_colors; ++i) {
                if (g_draw.colors[i].addr == g_learned.hud_target &&
                    g_draw.colors[i].width == f.gb_w && g_draw.colors[i].height == f.gb_h) {
                    cand = g_learned.hud_target;
                    f.hud_how = "last known HUD start (draw index and target)";
                    break;
                }
            }
        }
    }
    if (cand == 0 || f.injected || f.inject_failed || f.fg_native_done ||
        (fgn && g_fgn_lock == 2) ||
        !(fgn ? !g_present_dead : (WindowPathOk() || HudPointOn()))) {
        return false;
    }
    if (f.gb_w != g_output_w || f.gb_h != g_output_h || !f.post_seen) {
        return false;
    }
    // The upscaler path composites the HUD at present as (presented image - this copy): only the
    // presented image itself is a valid point for it. DLSS 5 and the frame generation inputs
    // (written back in place) take the composite target too. BB_INJECT_COMPOSITE=1: both.
    static const bool any_target = [] {
        const char* v = std::getenv("BB_INJECT_COMPOSITE");
        return v != nullptr && v[0] == '1';
    }();
    // 1.16: in scaled modes the HUD canvas takes the UI image too (the scene is upscaled there,
    // the HUD drawn at the output size, the game's final pass re-run on the canvas).
    const bool canvas_target = !fgn && BB::ResPatch::HudSharp() && HudCanvasOk();
    if (!fgn && !any_target && !canvas_target && !IsFlip(cand)) {
        f.hud_how = "composite target is not the presented image (upscaler waits for the flip)";
        return false;
    }
    flip = cand;
    f.hud_target = cand;
    f.hud_draw = f.draws;
    FillPresentPlan(flip, "before the first HUD draw");
    return true;
}

namespace {
void FillPresentPlan(VAddr flip, const char* where) {
    auto& f = g_frame;
    Plan& p = g_plan;
    p = {};
    p.scene_mode = false;
    p.color_addr = flip;
    p.depth_addr = f.gb_depth;
    p.render_w = f.gb_w;
    p.render_h = f.gb_h;
    p.output_w = g_present_w;
    p.output_h = g_present_h;
    p.hdr = false; // the post-processed (tonemapped) image
    p.reset = !g_learned.last_injected;
    if (!PresentMode() && !FgNative()) {
        // 0.48 native fallback: the history was the in-place path's HDR scene colour.
        p.reset = p.reset || !g_last_window_path;
        if (PathLockOn() && g_lock == 2 && g_last_window_path) {
            p.reset = false; // 0.85: locked to this path, one missed frame is not a camera cut
        }
        // 0.80: the hold (in-place off for BB_NATIVE_HOLD flips) only after an in-place upscale
        // really failed. 0.48-0.79 set it on every present-time frame, so the frames it held were
        // processed at present and renewed it: after the first missed frame (temporal not ready)
        // the in-place point was never tried again (07:54 run: "in place 0, at present 60" in
        // every window), and DLAA ran on the post-processed sRGB image instead of the HDR scene.
        if (f.inject_failed) {
            g_native_hold = NativeHoldFlips();
        }
        if (!g_native_said) {
            g_native_said = true;
            Checkpoint(fmt::format("[3d] native AA: a 3D frame the in-place point missed is "
                                   "processed on the presented image {}x{} ({}), so every 3D "
                                   "frame is anti-aliased and frame generation keeps its inputs "
                                   "(BB_NATIVE_FALLBACK=0 turns it off)",
                                   f.gb_w, f.gb_h, where));
        }
    }
    if (FgNative()) {
        // 0.39: frame generation without an upscaler: inputs only, at the game's own size.
        p.output_w = p.render_w;
        p.output_h = p.render_h;
        p.reset = !g_learned.last_fg_native;
    }
    p.delta_ms = g_delta_ms;
    p.viewport = f.viewport;
    p.depth_inverted = f.depth_greater;
    FillCamera(p);
    SrReset(p);
    if (const char* inv = std::getenv("BB_DEPTH_INVERT"); inv != nullptr && *inv != '\0') {
        p.depth_inverted = inv[0] == '1';
    }
    const bool jitter_on =
        Cfg().jitter && p.camera_valid && g_jittered_draws > g_jittered_frame_start;
    p.jitter[0] = jitter_on ? g_jitter[0] : 0.0f;
    p.jitter[1] = jitter_on ? g_jitter[1] : 0.0f;
    LogRtShadowCamera(p);
    LogPoint(fmt::format(
        "injection point: window-size upscale {}, source {:#x} (presented image, LDR) {}x{}, "
        "depth {:#x}, output {}x{} (window), motion vectors {}, depth {}",
        where, flip, p.render_w, p.render_h, p.depth_addr, p.output_w, p.output_h,
        p.camera_valid ? "reconstructed (camera found)" : "zero (no camera)",
        p.depth_inverted ? "inverted" : "forward"));
    if (FgNative()) {
        if (p.reset) ++g_fgn_sec.resets;
        if (p.camera_valid) ++g_fgn_sec.cam;
        static bool said = false;
        if (!said) {
            said = true;
            Checkpoint(fmt::format("frame generation WITHOUT an upscaler (BB_FG_NATIVE): first "
                                   "3D inputs {}x{} {}, camera {}, nothing upscaled or resampled",
                                   p.render_w, p.render_h, where,
                                   p.camera_valid ? "found" : "not found"));
        }
        return;
    }
    static bool first = true;
    if (first) {
        first = false;
        GuardBegin(SR::KindName(Cfg().kind), "the first upscale");
        Checkpoint(fmt::format("{}: {}x{} -> {}x{} (game output {}x{} upscaled to the window "
                               "size, HUD added on top at present)",
                               SR::KindName(Cfg().kind), p.render_w, p.render_h, p.output_w,
                               p.output_h, g_output_w, g_output_h));
        LOG_INFO(Render_Vulkan, "[BB_UPSCALE] upscale {}x{} -> {}x{} (window {}x{})", p.render_w,
                 p.render_h, p.output_w, p.output_h, g_present_w, g_present_h);
        Checkpoint(fmt::format("upscale {}x{} -> {}x{} (window {}x{})", p.render_w, p.render_h,
                               p.output_w, p.output_h, g_present_w, g_present_h));
        g_first_eval_pending = true;
    }
}
} // namespace

namespace {
void FirstEvalFailed(std::string_view why) {
    ++g_eval_attempts;
    if (g_eval_attempts < EvalMaxAttempts) {
        g_first_eval_pending = true; // judge the next upscaled frame
        Checkpoint(fmt::format("{}: upscaled frame check {} of {} failed: {}; retrying",
                               SR::KindName(Cfg().kind), g_eval_attempts, EvalMaxAttempts, why));
        return;
    }
    g_first_eval_pending = false;
    GuardEnd();
    Checkpoint(fmt::format("{}: first upscaled frame FAILED after {} attempts (last: {}); the "
                           "upscale stays on and is judged again by the per-frame counters",
                           SR::KindName(Cfg().kind), g_eval_attempts, why));
}

void PresentFailedLocked(std::string_view why) {
    ++g_present_fails;
    if (g_present_dead) {
        return;
    }
    if (g_present_fails < PresentMaxFails) {
        Checkpoint(fmt::format("{}: window-size upscale failed ({} of {} in a row): {}; retrying",
                               SR::KindName(Cfg().kind), g_present_fails, PresentMaxFails, why));
        return;
    }
    g_present_dead = true;
    Checkpoint(fmt::format("{}: window-size upscale off after {} failures in a row (last: {}): "
                           "back to the in-place upscale at the game's size",
                           SR::KindName(Cfg().kind), g_present_fails, why));
}
} // namespace

bool LastFrameUpscaled() {
    std::scoped_lock lk{g_mutex};
    return g_learned.last_injected;
}

bool MenuUpscaleWanted() {
    if (!Enabled()) {
        return false;
    }
    std::scoped_lock lk{g_mutex};
    // 0.69: from the first frame. FSR 1 is the presenter's own pass and needs no upscaler back
    // end, so the menus and loading screens before the back end is ready (or with the native
    // kind, or after a back-end failure) are no longer a plain stretch.
    if (!Cfg().menus || g_flip_scene) {
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        Checkpoint("2D frames (menus, loading screens, movies): scaled to the window with the "
                   "spatial FSR 1 pass (BB_UPSCALE_MENUS=0 turns it off)");
    }
    return true;
}

bool SceneSpatialFallback() {
    if (!Enabled() || !UpfixOn()) {
        return false;
    }
    std::scoped_lock lk{g_mutex};
    // 0.69: also a 3D frame while no temporal upscaler can run (back end not ready yet, native
    // kind, or turned off by a failure): the whole frame (3D + HUD) gets FSR 1, not a stretch.
    if (!g_flip_scene || (Active() && !PresentMode())) {
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        Checkpoint("3D frame missed the upscaler: scaled to the window with the spatial FSR 1 "
                   "pass instead of a plain stretch (BB_UPFIX=0 turns it off)");
    }
    return true;
}

bool FlipIs2D() {
    std::scoped_lock lk{g_mutex};
    return !g_flip_scene;
}

float Menu2DRcas() {
    static const float v = [] {
        const char* e = std::getenv("BB_2D_RCAS");
        if (e == nullptr || *e == '\0') return 0.15f;
        return static_cast<float>(std::atof(e));
    }();
    return v;
}

const char* Menu2DWhyNot() {
    if (!Enabled()) return "upscaler off";
    if (!Cfg().menus) return "BB_UPSCALE_MENUS=0";
    std::scoped_lock lk{g_mutex};
    if (g_flip_scene) return "3D flip";
    return "";
}

void Note2DFlip(int kind, const char* reason) {
    static std::mutex m;
    static u32 n = 0, fsr = 0, easu = 0, same = 0, plain = 0;
    static std::string reasons;
    std::scoped_lock lk{m};
    ++n;
    switch (kind) {
    case 0: ++fsr; break;
    case 1: ++easu; break;
    case 2: ++same; break;
    default:
        ++plain;
        if (reason != nullptr && *reason != '\0' && reasons.find(reason) == std::string::npos &&
            reasons.size() < 200) {
            reasons += reasons.empty() ? reason : std::string(", ") + reason;
        }
        break;
    }
    if (n >= 60) {
        Checkpoint(fmt::format("[2d] last {} flips: {} FSR1 (EASU + RCAS {:.2f}), {} FSR1 EASU only, "
                               "{} plain stretch{}{}{}, {} already the window size",
                               n, fsr, Menu2DRcas(), easu, plain, plain ? " (" : "",
                               plain ? reasons : std::string(), plain ? ")" : "", same));
        n = fsr = easu = same = plain = 0;
        reasons.clear();
    }
}

bool TakeLateUpscale() {
    std::scoped_lock lk{g_mutex};
    const bool late = g_late_upscale;
    g_late_upscale = false;
    return late;
}

void OnPresentUpscaled(bool ok) {
    std::scoped_lock lk{g_mutex};
    if (ok) {
        g_frame.injected = true;
        g_frame.redirecting = false;
        g_frame.hudless = true;
        g_frame.present = true;
        return;
    }
    g_frame.inject_failed = true;
    PresentFailedLocked("the window-size upscale before the HUD failed");
}

void OnFgNativeAnalyzed(bool ok) {
    std::scoped_lock lk{g_mutex};
    g_frame.fg_native_done = true; // once per frame either way; nothing else changes
    g_frame.fgn_fail = !ok;
    if (ok) {
        return;
    }
    static u32 fails = 0;
    if (fails++ % 600 == 0) {
        LOG_WARNING(Render_Vulkan,
                    "[BB_UPSCALE] frame generation without an upscaler: no inputs this frame "
                    "({} so far)",
                    fails);
    }
}

void OnPresentFailed(std::string_view why) {
    std::scoped_lock lk{g_mutex};
    PresentFailedLocked(why);
}

void OnPresentResult(bool presented, std::string_view why) {
    std::scoped_lock lk{g_mutex};
    if (presented) {
        g_present_fails = 0;
    }
    if (!g_eval_await_present) {
        return;
    }
    g_eval_await_present = false;
    if (presented) {
        GuardEnd();
        if (!g_said_presented) {
            g_said_presented = true;
            Checkpoint(fmt::format("upscaled {}x{} -> {}x{}, format {}, presented ({}, window-size "
                                   "path, HUD composited on top)",
                                   g_plan.render_w, g_plan.render_h, g_plan.output_w,
                                   g_plan.output_h, g_up_formats, SR::KindName(Cfg().kind)));
        }
        Checkpoint(fmt::format("{}: first upscaled frame done: {}x{} presented at the window size "
                               "{}x{} (attempt {})",
                               SR::KindName(Cfg().kind), g_plan.render_w, g_plan.render_h,
                               g_plan.output_w, g_plan.output_h, g_eval_attempts + 1));
        return;
    }
    FirstEvalFailed(fmt::format("the presenter did not show the window-size image: {}",
                                why.empty() ? "unknown" : why));
}

const Plan& CurrentPlan() {
    return g_plan;
}

void SetUpscaleFormats(std::string text) {
    std::scoped_lock lk{g_mutex};
    g_up_formats = std::move(text);
}

void OnInjected(bool ok, float) {
    std::scoped_lock lk{g_mutex};
    if (ok) {
        g_frame.injected = true;
        g_frame.redirecting = true;
        g_draw.sampled_upscaled = true;
        // 0.84: the in-place decision that worked, kept for the next frames.
        g_good_src = g_plan.color_addr;
        g_good_index = g_frame.qual_samples > 0 ? g_frame.qual_samples - 1 : 0;
        g_good_frame = g_frame_no;
        g_good_src_seen = g_frame_no;
        // 0.73: the scene passes end here; post-processing gets the output-size versions.
        g_ss_source.store(g_plan.color_addr, std::memory_order_relaxed);
        if (g_ss_phase.load(std::memory_order_relaxed) == 1) {
            g_ss_phase.store(2, std::memory_order_relaxed);
            ++g_frame.native_post;
        }
    } else {
        g_frame.inject_failed = true;
    }
}

void OnGpuTime(double ms) {
    std::scoped_lock lk{g_mutex};
    g_stats.gpu_ms_sum += ms;
    g_stats.gpu_ms_max = std::max(g_stats.gpu_ms_max, ms);
    ++g_stats.gpu_samples;
}

void OnStageTimes(const double (&ms)[kStages]) {
    std::scoped_lock lk{g_mutex};
    for (u32 i = 0; i < kStages; ++i) g_stats.stage_ms[i] += ms[i];
    ++g_stats.stage_samples;
}

static bool all_scene_on() {
    // 0.80: BB_TAA_ALL on by default (=0 turns it off). The 07:54 run jittered 227 of 578 scene
    // draws (351 without a camera found) while the upscaler was told the whole frame was
    // jittered. Only draws with the scene depth before the upscaler are scene draws, so shadows,
    // post-processing and the HUD are never jittered.
    static const bool on = [] {
        const char* v = std::getenv("BB_TAA_ALL");
        return v == nullptr || v[0] != '0';
    }();
    return on;
}

bool Jitter(float& x, float& y, bool fullscreen) {
    if (!Active() || !Cfg().jitter) {
        return false;
    }
    std::scoped_lock lk{g_mutex};
    // 0.79: BB_TAA_ALL=1 jitters every draw of the scene pass (a real TAA engine jitters all of
    // its projections), not only those whose vertex constants hold a recognised view-projection.
    const bool all_scene = all_scene_on();
    if (g_draw.valid && g_draw.scene) {
        ++g_taa_scene_draws;
        if (!g_draw.vertex_camera) ++g_taa_nocam_draws;
    }
    if (!g_draw.valid || !g_draw.scene || (!g_draw.vertex_camera && !all_scene) ||
        !g_learned.have_prev_vp || (!PresentMode() && LearnedSource() == 0)) {
        return false;
    }
    // 1.11: never after the in-place point ran this frame. The upscaler / DLAA has already
    // resolved the jittered scene there, so a later draw still flagged as a scene draw (the HUD
    // health / stamina bars drawn with the G-buffer depth bound; BB_TAA_ALL jitters every such
    // draw) kept a sub-pixel wobble nothing removed: the bars shimmered / haloed while the
    // camera moved. BB_JITTER_AFTER_POINT=1: the old behaviour.
    static const bool after_point = [] {
        const char* v = std::getenv("BB_JITTER_AFTER_POINT");
        return v != nullptr && v[0] == '1';
    }();
    if (!after_point && g_frame.injected && !g_frame.present) {
        ++g_stab.post_point_skipped;
        static u64 skipped = 0;
        if (skipped++ % 6000 == 0) {
            Checkpoint(fmt::format("[taa] 1.11: scene-flagged draw after the in-place point not "
                                   "jittered (HUD bars), {} so far",
                                   skipped));
        }
        return false;
    }
    // 1.27 (bb-dlaa-127): full-screen passes before the in-place point are never jittered. With
    // native AA the point is the first read of the HDR scene colour, so the deferred lighting /
    // composite passes that write it (full-screen triangles with the G-buffer depth bound for a
    // stencil / depth-bounds test) were jittered too: their UVs come from the shifted vertices,
    // so they re-sampled the already-jittered G-buffer at pixel - jitter a second time (double
    // offset, a phase-dependent bilinear blur) while DLAA was told one offset: the lit image
    // trembled, and looked right only on phases near zero. Geometry is jittered, screen passes
    // are not. BB_JITTER_FULLSCREEN=1: the old behaviour. BB_JITTER_NOWRITE=1 (diagnosis): also
    // skip every draw without depth writes.
    static const bool fs_old = [] {
        const char* v = std::getenv("BB_JITTER_FULLSCREEN");
        return v != nullptr && v[0] == '1';
    }();
    if (fullscreen && !fs_old) {
        ++g_stab.fullscreen_skipped;
        return false;
    }
    ++g_jittered_draws;
    ++g_stab.draws;
    x = g_jitter[0];
    y = g_jitter[1];
    return x != 0.0f || y != 0.0f;
}

// 0.79 [taa]: the jitter handed to the upscaler, with each SDK's sign convention applied, and the
// log line. Called once per upscaler dispatch with the jitter the plan carries (the offset the
// scene passes were rasterised with, render pixels, +x right, +y down). All four SDKs take the
// offset as applied to the projection in pixels (FSR 3.1: NDC x += 2*jx/w, y -= 2*jy/h, i.e. a
// shift of the picture by (+jx,+jy) screen pixels; DLSS / XeSS the same pixel offset): sign +1
// each. BB_JITTER_SIGN_<DLSS|FSR3|FSR4|XESS>=-1 flips only what that SDK receives (diagnosis).
void TaaPass(const float (&applied)[2], bool camera_valid, float (&passed)[2]) {
    const auto kind = Cfg().kind;
    const char* env = kind == SR::Kind::Dlss   ? "BB_JITTER_SIGN_DLSS"
                      : kind == SR::Kind::Fsr3 ? "BB_JITTER_SIGN_FSR3"
                      : kind == SR::Kind::Fsr4 ? "BB_JITTER_SIGN_FSR4"
                                               : "BB_JITTER_SIGN_XESS";
    const char* v = std::getenv(env);
    const float sign = v != nullptr && std::atof(v) < 0 ? -1.0f : 1.0f;
    passed[0] = applied[0] * sign;
    passed[1] = applied[1] * sign;
    std::scoped_lock lk{g_mutex};
    if (applied[0] != 0.0f || applied[1] != 0.0f) ++g_stab.nonzero_passed; else ++g_stab.zero_passed;
    // Scene draws were jittered this frame but the upscaler is told zero (or the reverse).
    const bool drew_jit = g_jittered_draws != g_jittered_frame_start;
    if (drew_jit != (applied[0] != 0.0f || applied[1] != 0.0f)) ++g_stab.mismatch;
    const u64 n = ++g_taa_dispatches;
    if (n != 1 && n % 600 != 0) {
        return;
    }
    const bool yes = applied[0] != 0.0f || applied[1] != 0.0f;
    const std::string line = fmt::format(
        "[taa] jitter applied to game projection: {}, sequence Halton(2,3)-0.5 over {} phases "
        "(FSR 3 phase count, 8*ratio^2), phase {}, offset {:+.4f},{:+.4f} px (render), passed to "
        "{} as {:+.4f},{:+.4f}{}, mv excludes jitter: {} (bb_mv.comp samples at pixel - jitter; "
        "no jitter-cancellation / motionVectorsJittered), scene draws jittered last frame {}/{} "
        "({} without a camera found{}), how: viewport offset of the G-buffer-depth passes "
        "(equivalent to translating the projection by the same pixels; HUD, post-processing and "
        "other-size passes untouched)",
        yes ? "yes" : "no", g_jitter_phases, g_jitter_index, applied[0], applied[1],
        SR::KindName(kind), passed[0], passed[1], sign < 0 ? " (sign flipped by env)" : "",
        camera_valid ? "yes" : "n/a (zero vectors, no camera yet)", g_jittered_last_frame,
        g_taa_last_scene, g_taa_last_nocam, all_scene_on() ? ", all scene draws (BB_TAA_ALL default on)" : ", BB_TAA_ALL=0");
    LOG_INFO(Render_Vulkan, "{}", line);
    Checkpoint(line);
}

bool WantHudlessCopy() {
    std::scoped_lock lk{g_mutex};
    if (!g_draw.valid || !g_draw.targets_flip) {
        return false;
    }
    if (g_frame.injected && !g_draw.sampled_upscaled) {
        ++g_frame.hud_draws;
    }
    // (Frame generation needs the image before the HUD every frame: bb_fg.h.)
    if ((!Cfg().hudless_copy && !BB::FG::WantInputs() && !BB::Analyze::Enabled()) ||
        !g_frame.injected || g_frame.hudless ||
        g_draw.sampled_upscaled) {
        return false;
    }
    g_frame.hudless = true;
    return true;
}

void OnHudlessCopied() {
    std::scoped_lock lk{g_mutex};
    ++g_stats.hudless;
}

namespace {
// 1.05: the FG-only path lock and the per-second [fg-pure] line (under g_mutex, from OnFlip).
void FgnNoteFrame(const FrameState& f) {
    auto& s = g_fgn_sec;
    const bool scene3d = f.gb_depth != 0 && f.scene_draws > 0;
    const auto now = std::chrono::steady_clock::now();
    static auto prev_flip = now;
    static bool have_prev = false;
    if (scene3d) {
        const int path = !f.fg_native_done || f.fgn_fail ? 0 : f.fgn_late ? 2 : 1;
        ++s.frames;
        if (path == 1) ++s.hud;
        if (path == 2) ++s.late;
        if (!f.fg_native_done) ++s.none;
        if (f.fgn_fail) ++s.fail;
        if (path != 0 && s.last != 0 && path != s.last) ++s.switches;
        if (path != 0) s.last = path;
        if (have_prev) {
            const double ms = std::chrono::duration<double, std::milli>(now - prev_flip).count();
            if (ms < 200.0) {
                s.int_sum += ms;
                s.int_sq += ms * ms;
                s.int_min = std::min(s.int_min, ms);
                s.int_max = std::max(s.int_max, ms);
                ++s.ints;
            }
        }
        if (g_fgn_lock == 0 && path != 0) {
            const int forced = FgnLockForced();
            if (forced > 0) {
                g_fgn_lock = forced;
            } else if (forced < 0) {
                ++g_fgn_lock_frames;
                if (path == 1) ++g_fgn_lock_hud;
                if (g_fgn_lock_frames >= 300) {
                    g_fgn_lock = g_fgn_lock_hud * 100 >= g_fgn_lock_frames * 95 ? 1 : 2;
                }
            }
            if (g_fgn_lock != 0) {
                Checkpoint(fmt::format(
                    "[fg-pure] FG without an upscaler: path locked={} for the session ({} of the "
                    "first {} 3D frames had the HUD point{}; BB_FGN_LOCK=hud/present/0)",
                    g_fgn_lock == 1 ? "before-HUD" : "at-present", g_fgn_lock_hud,
                    g_fgn_lock_frames, forced > 0 ? ", forced" : ""));
            }
        }
    }
    prev_flip = now;
    have_prev = scene3d;
    static auto last = now;
    if (now - last >= std::chrono::seconds(1)) {
        last = now;
        if (s.frames > 0) {
            const double mean = s.ints ? s.int_sum / s.ints : 0.0;
            const double sd =
                s.ints ? std::sqrt(std::max(0.0, s.int_sq / s.ints - mean * mean)) : 0.0;
            Checkpoint(fmt::format(
                "[fg-pure] last second: {} 3D frames, inputs before-HUD {} / at-present {} / none "
                "{} (failed {}), camera MVs {}, depth G-buffer, object MVs composed, HUD-less {}, "
                "path={} switches {}, resets {}, guest flip interval mean {:.2f} ms sd {:.2f} "
                "min {:.2f} max {:.2f}",
                s.frames, s.hud, s.late, s.none, s.fail, s.cam, s.hud,
                g_fgn_lock == 1 ? "locked before-HUD" : g_fgn_lock == 2 ? "locked at-present"
                                                                        : "undecided",
                s.switches, s.resets, mean, sd, s.ints ? s.int_min : 0.0, s.int_max));
        }
        const int keep = s.last;
        s = {};
        s.last = keep;
    }
}
} // namespace

void OnFlip(VAddr addr, u32 width, u32 height) {
    if (!Enabled() && !AnalyzeOn()) {
        if (RtReflect()) {
            std::scoped_lock lk{g_mutex};
            LogGbuffer(g_frame);
            g_frame = {};
        }
        return;
    }
    std::scoped_lock lk{g_mutex};
    auto& f = g_frame;
    g_flip_scene = f.gb_depth != 0; // (the presenter asks for this flip after this call)
    g_flips[g_flip_index++ % g_flips.size()] = addr;
    g_output_w = width;
    g_output_h = height;
    // Safety net: scene scaling must give a real upscale (composite injection, render size
    // below the output) in most 3D frames, or it is turned off for the session and the scene
    // targets go back to the full size (an unplayable image otherwise: white / a corner).
    {
        // 0.73: the next frame starts outside the scene. 0.76: only from the post-processing phase
        // (or a frame whose scene colour was never sampled), so a flip seen while the next frame's
        // scene passes already run does not end them half-way.
        int post = 2;
        if (!g_ss_phase.compare_exchange_strong(post, 0, std::memory_order_relaxed) &&
            post == 1 && f.qual_samples == 0) {
            g_ss_phase.store(0, std::memory_order_relaxed);
        }
    }
    {
        // 0.73 motion-vector source, once: the G-buffer's targets (a game velocity output would be
        // one of them or a two-channel target logged by [mv-probe]).
        static bool said = false;
        if (!said && f.gb_depth != 0 && f.gb_num_colors > 0 && f.gb_draws > 8) {
            said = true;
            std::string list;
            bool two = false;
            for (u32 i = 0; i < f.gb_num_colors; ++i) {
                const Target& c = f.gb_colors[i];
                if (c.addr == 0) continue;
                list += fmt::format(" c{}={}", i, Describe(c));
                const auto cf = c.format;
                two |= cf == vk::Format::eR16G16Sfloat || cf == vk::Format::eR16G16Snorm ||
                       cf == vk::Format::eR16G16Unorm || cf == vk::Format::eR8G8Snorm ||
                       cf == vk::Format::eR8G8Unorm || cf == vk::Format::eR32G32Sfloat;
            }
            Checkpoint(fmt::format(
                "[mv-source] G-buffer targets:{} d={}; {}; motion vectors used: camera from depth + "
                "the game's view-projection (scene block), objects from their G-buffer vertex "
                "shaders' clip positions this / last frame (bb_objmv)",
                list, Describe(f.gb_depth_target),
                two ? "a two-channel target IS in the G-buffer (see [mv-probe])"
                    : "no two-channel (velocity) target in the G-buffer"));
        }
    }
    // 0.73 native chain check: the upscaled image must reach the game's output chain, not only be
    // produced. A frame counts when the upscaler ended the scene passes (post-processing then used
    // the output-size versions) and no internal-size target was bound after it.
    if (g_off_at != 0 && g_frame_no >= g_off_at) {
        g_off_at = 0;
        TurnOff();
        Checkpoint("WARNING: scene upscale turned OFF for this session (native chain check): the "
                   "upscaled image did not reach the output chain");
        LOG_WARNING(Render_Vulkan, "[BB_UPSCALE] native chain check failed: scene upscale off");
    }
    static u32 nc_frames = 0, nc_scene = 0, nc_reach = 0, nc_redirect = 0, nc_scaled = 0,
               nc_windows = 0, nc_last_scaled = 0, nc_copies0 = 0, nc_stale0 = 0;
    if (NativePerPass() && g_ss_frozen.load(std::memory_order_relaxed) &&
        !g_ss_off.load(std::memory_order_relaxed)) {
        const u32 ps = g_ss_post_scaled.load(std::memory_order_relaxed);
        const bool scaled_after = ps != nc_last_scaled;
        nc_last_scaled = ps;
        ++nc_frames;
        if (f.gb_depth != 0) ++nc_scene;
        if (scaled_after) ++nc_scaled;
        if (f.native_redirects > 0) ++nc_redirect;
        if (f.injected && !f.inject_failed && f.native_post > 0 && !scaled_after) ++nc_reach;
        if (nc_frames >= 300) {
            const bool first = nc_windows++ == 0;
            // 0.80: the content check too: most sampled frames must give an output whose
            // downscale matches the internal lit colour (PSNR >= 20 dB).
            const u32 pn = g_ss_psnr_n.exchange(0), plow = g_ss_psnr_low.exchange(0);
            const bool bad = (nc_scene >= 150 && nc_reach * 2 < nc_scene) ||
                             (pn >= 3 && plow * 2 > pn);
            (void)first;
            {
                Checkpoint(fmt::format(
                    "native chain check: {} of {} 3D frames ({} frames) gave post-processing the "
                    "upscaled image at the output size ({} with later samples redirected to it, "
                    "{} with an internal-size target after the upscaler); twin copies {} ({} not "
                    "possible): {}; 0.76: twin copies {:.1f} per frame, stale twins read {} "
                    "(content older than the last frame), source samples refused: before-HUD "
                    "mode {}, learned skip {}, inactive {}, failed {}, size {}, target {}; 0.80 content: "
                    "{} samples, {} below 20 dB, last PSNR {:.1f} dB; 0.83 stale by address:{}; history "
                    "images kept at the output size: {}",
                    nc_reach, nc_scene, nc_frames, nc_redirect, nc_scaled,
                    g_ss_twin_copies.load(std::memory_order_relaxed),
                    g_ss_twin_fail.load(std::memory_order_relaxed), bad ? "FAILED" : "ok",
                    double(g_ss_twin_copies.load(std::memory_order_relaxed) - nc_copies0) /
                        double(std::max<u32>(1, nc_frames)),
                    g_ss_stale_reads.load(std::memory_order_relaxed) - nc_stale0,
                    g_ss_refuse[RfPresent].exchange(0), g_ss_refuse[RfSkip].exchange(0),
                    g_ss_refuse[RfInactive].exchange(0), g_ss_refuse[RfFailed].exchange(0),
                    g_ss_refuse[RfSize].exchange(0), g_ss_refuse[RfTarget].exchange(0), pn, plow,
                    g_ss_psnr_last.load(std::memory_order_relaxed) / 100.0, [] {
                        std::string list;
                        const u32 n = g_ss_count.load(std::memory_order_relaxed);
                        for (u32 i = 0; i < n; ++i) {
                            const u32 c = g_ss_stale_by[i].exchange(0, std::memory_order_relaxed);
                            if (c != 0) list += fmt::format(" {:#x} x{}", g_ss_addrs[i], c);
                        }
                        return list.empty() ? std::string(" none") : list;
                    }(), [] {
                        u32 k = 0;
                        const u32 n = g_ss_count.load(std::memory_order_relaxed);
                        for (u32 i = 0; i < n; ++i) k += g_ss_pinned[i].load() ? 1 : 0;
                        return k;
                    }()));
            }
            // 0.93: a failed check no longer turns the chain off by default (that sent every
            // later frame to the window path); it dumps one frame for diagnosis and keeps the
            // internal-resolution path. BB_NATIVE_CHECK_OFF=1: 0.84's behaviour (turn off).
            const bool check_off = NativeCheckOff();
            static bool dumped = false;
            if (bad && !check_off && !dumped) {
                dumped = true;
                g_dump_request.store(g_frame_no + 1, std::memory_order_relaxed);
                Checkpoint(fmt::format("native chain check below target: frame {} is dumped "
                                       "(bb_dump); the native chain stays on (BB_NATIVE_CHECK_OFF=1 "
                                       "turns it off instead)",
                                       g_frame_no + 1));
            }
            if (bad && check_off && g_off_at == 0) {
                // 0.84: the next frame is dumped with the chain still active (BB_DUMP_FRAME,
                // into bb_dump), then scaling turns off two frames later.
                g_dump_request.store(g_frame_no + 1, std::memory_order_relaxed);
                g_off_at = g_frame_no + 3;
                Checkpoint(fmt::format("native chain check FAILED: frame {} is dumped with the "
                                       "chain active (bb_dump), scene upscale turns off at frame {}",
                                       g_frame_no + 1, g_off_at));
            }
            if (false) {
                Checkpoint("WARNING: scene upscale turned OFF for this session (native chain "
                           "check): the upscaled image did not reach the output chain");
                LOG_WARNING(Render_Vulkan, "[BB_UPSCALE] native chain check failed: {} of {}",
                            nc_reach, nc_scene);
            }
            nc_frames = nc_scene = nc_reach = nc_redirect = nc_scaled = 0;
            nc_copies0 = g_ss_twin_copies.load(std::memory_order_relaxed);
            nc_stale0 = g_ss_stale_reads.load(std::memory_order_relaxed);
        }
    }
    if (g_ss_frozen.load(std::memory_order_relaxed) && !g_ss_off.load(std::memory_order_relaxed)) {
        ++g_ss_frames;
        if (f.gb_depth != 0) ++g_ss_scene;
        if (f.injected && !f.inject_failed && g_plan.output_w > g_plan.render_w) ++g_ss_good;
        if (g_ss_frames >= 300) {
            const bool first = g_ss_windows++ == 0;
            const bool bad = first ? g_ss_good < 150
                                   : g_ss_scene >= 150 && g_ss_good * 2 < g_ss_scene;
            if (bad && NativePerPass() && !NativeCheckOff()) {
                Checkpoint(fmt::format("scene upscale check: {} of {} 3D frames upscaled (scene "
                                       "size -> output) in {} frames, below target; the native "
                                       "chain stays on (BB_NATIVE_CHECK_OFF=1 turns it off)",
                                       g_ss_good, g_ss_scene, g_ss_frames));
            } else if (bad) {
                TurnOff();
                Checkpoint(fmt::format(
                    "WARNING: scene upscale turned OFF for this session (safety net): in {} frames "
                    "{} had a 3D scene and only {} were upscaled from the scene size to the "
                    "output; back to the normal path",
                    g_ss_frames, g_ss_scene, g_ss_good));
                LOG_WARNING(Render_Vulkan, "[BB_UPSCALE] scene upscale turned off: {} of {} "
                            "3D frames upscaled", g_ss_good, g_ss_scene);
            } else if (first) {
                Checkpoint(fmt::format("scene upscale check: {} of {} 3D frames upscaled "
                                       "(scene size -> output) in the first {} frames: kept",
                                       g_ss_good, g_ss_scene, g_ss_frames));
            }
            g_ss_frames = g_ss_scene = g_ss_good = 0;
        }
    }
    if (SceneScaleWanted() && !g_ss_frozen.load(std::memory_order_relaxed)) {
        const double ratio = SR::QualityRatio(Cfg().kind, Cfg().quality);
        if (Active() && ratio > 1.01 && f.gb_depth != 0 && f.scene_draws > 0 &&
            !g_ss_candidates.empty() && f.gb_w == width && f.gb_h == height) {
            g_ss_w = f.gb_w;
            g_ss_h = f.gb_h;
            g_ss_sw = std::max<u32>(8, static_cast<u32>(std::lround(f.gb_w / ratio)) & ~1u);
            g_ss_sh = std::max<u32>(8, static_cast<u32>(std::lround(f.gb_h / ratio)) & ~1u);
            const u32 count = static_cast<u32>(g_ss_candidates.size());
            std::string list;
            for (u32 i = 0; i < count; ++i) {
                g_ss_addrs[i] = g_ss_candidates[i];
                g_ss_valid[i].store(2, std::memory_order_relaxed); // the full-size images hold it
                g_ss_class[i].store(0, std::memory_order_relaxed);
                list += fmt::format(" {:#x}", g_ss_addrs[i]);
            }
            g_ss_depth.store(f.gb_depth, std::memory_order_relaxed);
            g_ss_phase.store(0, std::memory_order_relaxed);
            g_ss_count.store(count, std::memory_order_release);
            g_ss_frozen.store(true, std::memory_order_release);
            Checkpoint(fmt::format("scene upscale (BB_SCENE_UPSCALE={}): {} scene targets of {}x{} "
                                   "rendered at {}x{} (ratio {:.3f}), post-processing and HUD at "
                                   "{}x{}:{}",
                                   BB::ResPatch::NativeChain() ? "native" : "shrink", count,
                                   g_ss_w, g_ss_h, g_ss_sw, g_ss_sh, ratio, width,
                                   height, list));
        }
        if (!g_ss_frozen.load(std::memory_order_relaxed) && f.gb_depth != 0 &&
            (f.gb_w != width || f.gb_h != height || ratio <= 1.01)) {
            static bool said = false;
            if (!said) {
                said = true;
                Checkpoint(fmt::format(
                    "scene upscale (BB_SCENE_UPSCALE=1) not active: {}", ratio <= 1.01
                        ? std::string("the mode is native (ratio 1), nothing to upscale")
                        : fmt::format("the game renders its scene at {}x{} but outputs {}x{}; it "
                                      "needs the resolution patch at the output size (internal "
                                      "= window), the mode then sets the scene size",
                                      f.gb_w, f.gb_h, width, height)));
            }
        }
        g_ss_candidates.clear();
    }
    const auto now = std::chrono::steady_clock::now();
    if (g_last_flip.time_since_epoch().count() != 0) {
        g_delta_ms = std::chrono::duration<float, std::milli>(now - g_last_flip).count();
    }
    g_last_flip = now;

    LogGbuffer(f);

    // Learn this frame's layout for the next.
    const VAddr src_before = LearnedSource();
    if (f.gb_depth != 0) {
        const bool same = f.gb_depth == g_learned.gb_depth && f.gb_w == g_learned.render_w &&
                          f.gb_h == g_learned.render_h && f.scene_color == g_learned.scene_color;
        g_learned.stable = same ? g_learned.stable + 1 : 0;
        g_learned.gb_depth = f.gb_depth;
        g_learned.render_w = f.gb_w;
        g_learned.render_h = f.gb_h;
        g_learned.scene_color = f.scene_color;
        // 0.84: the source the in-place point last worked with stays the source while the scene
        // still draws it (seen in the last 120 frames), whatever target this frame drew last.
        if (g_good_src != 0) {
            if (f.good_src_drawn) g_good_src_seen = g_frame_no;
            if (g_frame_no <= g_good_src_seen + 120) {
                g_learned.scene_color = g_good_src;
            } else {
                Checkpoint(fmt::format("[3d] in-place source {:#x} not drawn for 120 frames: "
                                       "learning the source again",
                                       g_good_src));
                g_good_src = 0;
            }
        }
        if (f.comp_src != 0) g_learned.comp_src = f.comp_src;
        if (src_before == LearnedSource()) {
            // The largest over the last 240 frames: a frame with one more write than usual must
            // not have its upscale before that write.
            if (f.last_write_after >= g_learned.skip) {
                g_learned.skip = f.last_write_after;
                g_learned.skip_age = 0;
            } else if (++g_learned.skip_age > 240) {
                g_learned.skip = f.last_write_after;
                g_learned.skip_age = 0;
            }
        } else {
            g_learned.skip = 0;
        }
    }
    const Candidate* cam = Cfg().mv_reconstruct ? BestCamera() : nullptr;
    if (cam != nullptr && Enabled() && g_frame_no % Cfg().log_every == 0) {
        const auto& m = cam->m.m;
        LOG_INFO(Render_Vulkan,
                 "[BB_UPSCALE] frame {}: camera view-projection (seen in {} draws): {:.4g} {:.4g} "
                 "{:.4g} {:.4g} / {:.4g} {:.4g} {:.4g} {:.4g} / {:.4g} {:.4g} {:.4g} {:.4g} / "
                 "{:.4g} {:.4g} {:.4g} {:.4g}",
                 g_frame_no, cam->count, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9],
                 m[10], m[11], m[12], m[13], m[14], m[15]);
    }
    g_learned.vp_before_roll = g_learned.prev_vp;
    g_learned.have_vp_before_roll = g_learned.have_prev_vp;
    g_learned.rolled = true;
    if (cam != nullptr) {
        g_learned.prev_vp = cam->m;
        g_learned.have_prev_vp = true;
        ++g_stats.camera_frames;
    } else {
        g_learned.have_prev_vp = false;
    }
    const bool prev_injected = g_learned.last_injected;
    g_learned.last_injected = f.injected;
    if (f.injected) {
        g_last_window_path = f.present; // (the late path below sets it again)
    }
    if (g_native_hold > 0) --g_native_hold;
    const bool prev_fg_native = g_learned.last_fg_native;
    g_learned.last_fg_native = f.fg_native_done;
    // The scene was drawn but no HUD point was found (no HUD this frame, e.g. a cut scene): the
    // presenter upscales the finished image instead (TakeLateUpscale).
    g_late_upscale = false;
    // 0.84: a 3D frame that missed the in-place point while it works (it ran in the last 120
    // frames) is not sent to the present-time path: switching the upscaler's input between the
    // HDR scene colour and the post-processed image reset its history every switch (19:07 run:
    // the background blurred and cleared in pulses). The frame keeps the game's own image.
    // (the whole session once the in-place point has worked: never mix the two paths, 19:24 run:
    // consecutive frames alternated sharper / softer and the HUD changed brightness)
    bool keep_inplace = !PresentMode() && g_good_frame != 0;
    // 0.85 session path lock.
    const bool scene3d = f.gb_depth != 0 && f.scene_draws > 0;
    if (PathLockOn()) {
        // 0.93: the native chain (internal-resolution scene, upscaler in place before
        // post-processing) is the in-place path by construction: it locks in place as soon as the
        // scene targets are scaled, and the window / present-time path is never chosen for it
        // (0.85-0.91: the first 300 3D frames decided, < 95% in place locked=window, and with
        // g_lock == 2 every source sample was skipped before the chain could end its scene
        // passes, so the native chain check then saw almost no frame reach post-processing and
        // turned the chain off: "path=window" for every ratio > 1).
        static bool chain_locked = false;
        if (NativeChainLive()) {
            if (g_lock != 1) {
                Checkpoint(fmt::format(
                    "[path] locked=in-place for the session: native chain (scene {}x{} -> {}x{}, "
                    "upscaler on the lit scene colour before post-processing and HUD){}",
                    g_ss_sw, g_ss_sh, g_ss_w, g_ss_h,
                    g_lock == 2 ? " (was window: overridden)" : ""));
                g_lock = 1;
                chain_locked = true;
            }
        } else if (chain_locked && g_ss_off.load(std::memory_order_relaxed)) {
            chain_locked = false;
            g_lock = 0; // the chain was turned off: decide again from the next 300 3D frames
            g_lock_frames = g_lock_inplace = 0;
            Checkpoint("[path] native chain off: the path is decided again");
        }
        if (g_lock == 0 && scene3d) {
            ++g_lock_frames;
            if (f.injected) ++g_lock_inplace;
            if (g_lock_frames >= 300) {
                g_lock = g_lock_inplace * 100 >= g_lock_frames * 95 ? 1 : 2;
                // 1.02: native AA locks in place when the in-place point carried most frames
                // (the present-time path there runs on the image with the HUD); window only when
                // the in-place point really does not work, for the whole session.
                if (NativeAaUpscaler() && g_lock == 2 && g_lock_inplace * 2 >= g_lock_frames) {
                    g_lock = 1;
                }
                if (NativeAaUpscaler()) {
                    Checkpoint(fmt::format(
                        "[path] 1.02 native AA (ratio 1, {}): {} of {} 3D frames in place -> "
                        "locked={} for the session{}",
                        SR::KindName(Cfg().kind), g_lock_inplace, g_lock_frames,
                        g_lock == 1 ? "in-place" : "window",
                        g_lock == 1 ? "; DLSS 5 runs after the upscaler on the scene colour"
                                    : " (fallback: same-size present-time AA, DLSS 5 after it)"));
                }
                // 0.93b: the window path only when it can run at all (the game's whole frame
                // smaller than the window, or native AA). When the scene renders below the
                // game's output size (resolution patch / composite point) there is no window
                // path: locking to it skipped every in-place sample and nothing was upscaled.
                if (g_lock == 2 && !WindowPathOk()) g_lock = 1;
                Checkpoint(fmt::format(
                    "[path] locked={} for the session: {} of the first {} 3D frames were upscaled "
                    "in place{} (BB_PATH_LOCK=0: per-frame choice)",
                    g_lock == 1 ? "in-place" : "window", g_lock_inplace, g_lock_frames,
                    g_lock == 1 ? ""
                                : " (< 95%: the in-place point is not reliable here, every 3D "
                                  "frame goes through the window / present-time path)"));
            }
        }
        // 1.29: a scaled frame (the game's whole frame below the window: resolution patch) has no
        // in-place point (refused as before-HUD mode): its path is the window / HUD point. 1.27's
        // log locked in-place from 299/300 early frames (before the window size was known), then
        // every scaled frame was refused in place and kept off the present path: 0 upscaled.
        if (g_lock == 1 && PresentMode() && !NativeChainLive()) {
            g_lock = 2;
            Checkpoint("[path] locked=window for the session: the frame is scaled (game "
                       "output below the window), the in-place point does not apply (1.29)");
        }
        if (g_lock == 1) keep_inplace = true;
        if (g_lock == 2) keep_inplace = false;
        if (scene3d) {
            ++g_sec_frames;
            if ((g_lock == 1 && f.injected) || g_lock == 2) ++g_sec_locked;
        }
        static auto last = std::chrono::steady_clock::now();
        const auto now_t = std::chrono::steady_clock::now();
        if (now_t - last >= std::chrono::seconds(1)) {
            last = now_t;
            if (g_sec_frames > 0) {
                Checkpoint(fmt::format("[path] locked={}, frames this second on locked path {}/{}, "
                                       "resets {} (camera delta max {:.3f} NDC; a reset only "
                                       "above BB_SR_CUT_NDC, default 3.0)",
                                       g_lock == 1 ? "in-place" : g_lock == 2 ? "window"
                                                                              : "undecided",
                                       g_sec_locked, g_sec_frames, g_sec_resets, g_sec_cam_max));
            }
            g_sec_frames = g_sec_locked = g_sec_resets = 0;
            g_sec_cam_max = 0.0;
        }
    }
    {
        static auto last = std::chrono::steady_clock::now();
        const auto now_t = std::chrono::steady_clock::now();
        if (now_t - last >= std::chrono::seconds(1)) {
            last = now_t;
            const u32 be = SR::g_backend_resets.exchange(0, std::memory_order_relaxed);
            const u32 total = g_rs_first + g_rs_cut + g_rs_size + g_rs_path;
            if (g_sec_upscales > 0 || be > 0) {
                Checkpoint(fmt::format(
                    "[sr-hist] history resets this second: {} of {} upscales (first {}, camera "
                    "cut {}, size change {}, path switch {}); back end re-creates {}{}{}; HUD "
                    "draws and target changes never reset",
                    total, g_sec_upscales, g_rs_first, g_rs_cut, g_rs_size, g_rs_path, be,
                    be > 0 ? ", last: " : "",
                    be > 0 ? SR::g_backend_reset_why.load(std::memory_order_relaxed) : ""));
            }
            g_sec_upscales = g_rs_first = g_rs_cut = g_rs_size = g_rs_path = 0;
        }
    }
    {
        static u32 n_frames = 0;
        ++n_frames;
        if (f.gb_depth != 0 && f.scene_draws > 0 && !f.injected) {
            const u32 why = f.miss >= 0                 ? static_cast<u32>(f.miss)
                            : f.qual_samples == 0      ? u32(RfCount)
                                                       : u32(RfCount + 1);
            ++g_miss_counts[why];
            if (keep_inplace) ++g_miss_kept;
        }
        if (n_frames >= 60) {
            u32 total = 0;
            for (const u32 c : g_miss_counts) total += c;
            if (total > 0) {
                Checkpoint(fmt::format(
                    "[3d] in-place misses, last {} flips: {} (before-HUD mode {}, waiting for a "
                    "later sample {}, inactive {}, failed {}, size {}, target {}, source not "
                    "sampled {}, other {}); {} kept on the in-place decision (no present-time "
                    "switch, history kept); in-place source {:#x} index {}",
                    n_frames, total, g_miss_counts[RfPresent], g_miss_counts[RfSkip],
                    g_miss_counts[RfInactive], g_miss_counts[RfFailed], g_miss_counts[RfSize],
                    g_miss_counts[RfTarget], g_miss_counts[RfCount], g_miss_counts[RfCount + 1],
                    g_miss_kept, g_good_src, g_good_index));
            }
            g_miss_counts = {};
            g_miss_kept = 0;
            n_frames = 0;
        }
    }
    // 0.97 [inject]: the HUD start found this frame is kept as the last known one; a 3D frame
    // that reaches the flip without it is logged with the reason (it then goes to the present-
    // time path below, the only fallback).
    {
        const bool scene3d_i = f.gb_depth != 0 && f.scene_draws > 0;
        static u32 miss_logged = 0, miss_sec = 0, hit_sec = 0;
        static auto miss_t = std::chrono::steady_clock::now();
        if (f.hud_draw != 0) {
            ++hit_sec;
            if (f.hud_target != g_learned.hud_target ||
                (f.hud_draw != g_learned.hud_draw && g_frame_no % 600 == 0)) {
                Checkpoint(fmt::format("[inject] HUD point: draw {} into {:#x} ({}), composite "
                                       "target {:#x} (last post-processing draw {}); upscaler / "
                                       "DLAA, DLSS 5 and frame generation inputs run there",
                                       f.hud_draw, f.hud_target, f.hud_how, f.post_target,
                                       f.post_draw));
            }
            g_learned.hud_target = f.hud_target;
            g_learned.hud_draw = f.hud_draw;
        } else if (scene3d_i && (HudPointOn() || (FgNative() && g_fgn_lock != 2)) &&
                   !f.injected && !f.fg_native_done) {
            ++miss_sec;
            const std::string why =
                f.gb_w != g_output_w || f.gb_h != g_output_h
                    ? fmt::format("scene {}x{} is not the output size {}x{}", f.gb_w, f.gb_h,
                                  g_output_w, g_output_h)
                : !f.post_seen ? std::string("no post-processing draw (output size, no depth, "
                                             "sampling a render-size image) after the scene")
                : f.inject_failed ? std::string("the dispatch at the HUD point failed")
                : fmt::format("no 2D draw into the composite target {:#x} after its last "
                              "post-processing draw {} (last known HUD start: draw {} into "
                              "{:#x}, {} draws this frame)",
                              f.post_target, f.post_draw, g_learned.hud_draw,
                              g_learned.hud_target, f.draws);
            if (miss_logged < 20) {
                ++miss_logged;
                Checkpoint(fmt::format("[inject] miss reason (frame {}): {}; present-time path",
                                       g_frame_no, why));
            }
        }
        if (scene3d_i && !f.trace.empty()) {
            // 1.16: traced in any part of the session (the 1.14 log reached 3D after frame 2400).
            if (g_inject_traced < 6 && (g_frame_no % 97) == 0) {
                ++g_inject_traced;
                Checkpoint(fmt::format("[inject] output-size draws after the scene, frame {}:{}",
                                       g_frame_no, f.trace));
            }
        }
        const auto now_i = std::chrono::steady_clock::now();
        if (now_i - miss_t >= std::chrono::seconds(1)) {
            miss_t = now_i;
            if (hit_sec + miss_sec > 0 && (miss_sec > 0 || g_frame_no % 600 < 120)) {
                Checkpoint(fmt::format("[inject] last second: HUD point {} frames, missed {}",
                                       hit_sec, miss_sec));
            }
            hit_sec = miss_sec = 0;
        }
    }
    if (WindowPathOk() && !keep_inplace && !f.injected && !f.inject_failed && f.gb_depth != 0 &&
        f.gb_w == width && f.gb_h == height && f.scene_draws > 0 &&
        LateSamplesOk(f.qual_samples)) {
        g_learned.last_injected = prev_injected;
        FillPresentPlan(addr, "at present (no HUD draw found this frame)");
        g_late_upscale = true;
        f.injected = true;
        g_learned.last_injected = true;
        g_last_window_path = true;
    } else if (FgNative() && !g_present_dead && !f.fg_native_done && !f.injected &&
               !f.inject_failed && f.gb_depth != 0 && f.gb_w == width && f.gb_h == height &&
               LateSamplesOk(f.qual_samples)) {
        // 0.39: no HUD point this frame: the FG-only analysis on the finished image at present.
        // 0.51 (bb-fgnfix-051): the same sample rule as the upscalers' late path (0.40 upfix): the
        // finished flip needs no sample of the source. With qual_samples > SkipNeeded() almost every
        // 3D frame had no point ("other / no injection point 60") and frame generation no inputs.
        g_learned.last_fg_native = prev_fg_native;
        FillPresentPlan(addr, g_fgn_lock == 2 ? "at present (FG-only path locked at present)"
                                              : "at present (no HUD draw found this frame)");
        g_late_upscale = true;
        f.fg_native_done = true;
        f.fgn_late = true;
        g_learned.last_fg_native = true;
    }
    if (FgNative()) FgnNoteFrame(f);
    {
        auto& n = g_not3d;
        ++n.frames;
        if (f.injected) {
            ++n.up;
            if (g_late_upscale) {
                ++n.late;
            } else if (f.present) {
                ++n.hud;
            } else {
                ++n.inplace;
            }
        } else if (f.inject_failed) {
            ++n.failed;
        } else if (f.gb_depth == 0) {
            ++n.no_depth;
        } else if (f.scene_draws == 0) {
            ++n.no_scene;
        } else if (f.gb_w != width || f.gb_h != height) {
            ++n.size;
        } else if (!LateSamplesOk(f.qual_samples)) {
            ++n.samples;
            n.max_q = std::max(n.max_q, f.qual_samples);
            n.max_skip = std::max(n.max_skip, g_learned.skip);
        } else {
            ++n.no_point;
            n.np_max_q = std::max(n.np_max_q, f.qual_samples);
            n.np_max_skip = std::max(n.np_max_skip, g_learned.skip);
            if (f.scene_color != 0 && f.scene_color != LearnedSource()) ++n.np_src_moved;
        }
        if (Enabled() && n.frames >= 60) {
            const bool any3d = n.up + n.no_scene + n.size + n.samples + n.failed + n.no_point > 0;
            if (n.up != n.frames) {
                Checkpoint(fmt::format(
                    "[3d] last {} flips: {} upscaled (in place {}, before HUD {}, at present {}); "
                    "not 3D: no G-buffer depth {}, no scene draws "
                    "{}, scene size != flip {}, source sampled too few times {} (max {} seen, "
                    "needs > {}; learned skip {}, relax {}, upfix {}), dispatch failed {}, other / no "
                    "injection point {} (in-place samples max {} vs learned skip {}, source moved "
                    "{}; native fallback {})",
                    n.frames, n.up, n.inplace, n.hud, n.late, n.no_depth, n.no_scene, n.size,
                    n.samples, n.max_q, SkipNeeded(), n.max_skip, SkipRelax() ? "on" : "off",
                    UpfixOn() ? "on" : "off", n.failed, n.no_point, n.np_max_q, n.np_max_skip,
                    n.np_src_moved,
                    !NativeFallbackOn() ? "off (BB_NATIVE_FALLBACK=0)"
                    : NativeFallback()  ? "active"
                                        : "on (not native)"));
            } else if (any3d && NativeFallbackOn() && n.hud + n.late > 0) {
                Checkpoint(fmt::format("[3d] last {} flips: all {} upscaled (in place {}, before "
                                       "HUD {}, at present {})",
                                       n.frames, n.up, n.inplace, n.hud, n.late));
            }
            n = {};
        }
    }
    // The first-frame verdict, after this frame's late (at present) decision: before, a frame
    // upscaled at present still read as not injected here and was reported FAILED.
    if (g_first_eval_pending) {
        if (f.injected && (f.present || g_late_upscale)) {
            // Window-size path: the verdict comes from the presenter (OnPresentResult).
            g_first_eval_pending = false;
            g_eval_await_present = true;
        } else if (f.injected) {
            g_first_eval_pending = false;
            GuardEnd();
            Checkpoint(fmt::format("{}: first upscaled frame done (in place, {}x{})",
                                   SR::KindName(Cfg().kind), g_output_w, g_output_h));
            if (!g_said_presented) {
                g_said_presented = true;
                Checkpoint(fmt::format(
                    "upscaled {}x{} -> {}x{}, format {}, presented ({}, in place{})",
                    g_plan.render_w, g_plan.render_h, g_plan.output_w, g_plan.output_h,
                    g_up_formats, SR::KindName(Cfg().kind),
                    g_plan.render_w == g_plan.output_w && g_plan.render_h == g_plan.output_h
                        ? ": same size, anti-aliasing only, NOT an upscale"
                        : ""));
            }
        } else {
            const std::string why =
                f.inject_failed   ? "the upscale dispatch failed (see the back end's log)"
                : f.gb_depth == 0 ? "no 3D scene this frame (menu / loading)"
                : f.gb_w != width || f.gb_h != height
                    ? fmt::format("scene {}x{} is not the presented size {}x{}", f.gb_w, f.gb_h,
                                  width, height)
                : f.qual_samples <= g_learned.skip
                    ? fmt::format("the source was sampled {} time(s), {} needed",
                                  f.qual_samples, g_learned.skip + 1)
                : !PresentMode() ? std::string("window-size mode is off (present path disabled or "
                                               "sizes do not match)")
                                 : std::string("no injection point this frame");
            FirstEvalFailed(why);
        }
    }

    ++g_stats.frames;
    g_total_upscaled += f.injected ? 1 : 0;
    g_stats.injected += f.injected ? 1 : 0;
    g_stats.failed += f.inject_failed ? 1 : 0;
    g_stats.flip_draws += f.flip_draws;
    g_stats.hud_draws += f.hud_draws;
    g_stats.hud_other += f.hud_other;
    g_stats.src_rewrites += f.src_rewrites;

    if (Enabled() && g_frame_no % Cfg().log_every == 0) {
        const auto& c = Cfg();
        const double ratio = SR::QualityRatio(c.kind, c.quality);
        const bool to_window = PresentMode();
        // The size the upscale should reach: the window (presentation) when the game's frame is
        // not larger than it, else the game's own output.
        const bool win_target = g_present_w != 0 && g_present_h != 0 &&
                                g_present_w >= g_output_w && g_present_h >= g_output_h;
        const u32 out_w = to_window ? g_present_w : g_output_w;
        const u32 out_h = to_window ? g_present_h : g_output_h;
        const u32 want_w = win_target ? g_present_w : g_output_w;
        const u32 want_h = win_target ? g_present_h : g_output_h;
        const u32 rec_w = static_cast<u32>(std::lround(want_w / ratio));
        const u32 rec_h = static_cast<u32>(std::lround(want_h / ratio));
        const bool scene = SceneMode();
        LOG_INFO(Render_Vulkan,
                 "[BB_UPSCALE] frame {}: upscaler {} ({}, {}), {} mode, internal {}x{} -> output "
                 "{}x{} (mode wants internal {}x{}), source {:#x}, depth {:#x}, MV {}, jitter {} "
                 "({:+.3f},{:+.3f}, {} 3D draws jittered last frame), upscale GPU ms avg {:.3f} max {:.3f} ({} samples), injected "
                 "{}/{} frames, failed {}",
                 g_frame_no, SR::KindName(c.kind), SR::QualityName(c.quality),
                 BackendReady() ? "ready" : "not ready",
                 g_learned.render_w == 0 ? "no 3D scene seen yet (menu / loading / intro)"
                 : to_window             ? "window-size (before the HUD)"
                 : scene                 ? "scene"
                                         : "composite",
                 g_learned.render_w, g_learned.render_h, out_w, out_h, rec_w, rec_h,
                 LearnedSource(), g_learned.gb_depth,
                 !c.mv_reconstruct ? "zero (BB_MV=zero)"
                 : c.mv_upscale_zero      ? "zero for the upscalers (BB_MV_UPSCALE=zero)"
                 : g_learned.have_prev_vp ? "reconstructed (depth + camera)"
                                          : "zero (camera not found)",
                 Cfg().jitter && g_learned.have_prev_vp ? "on" : "off", g_jitter[0], g_jitter[1],
                 g_jittered_last_frame, g_stats.gpu_samples ? g_stats.gpu_ms_sum / g_stats.gpu_samples : 0.0,
                 g_stats.gpu_ms_max, g_stats.gpu_samples, g_stats.injected, g_stats.frames,
                 g_stats.failed);
        LOG_INFO(Render_Vulkan,
                 "[BB_UPSCALE] frame {}: HUD separation: draws into the presented image {}, HUD "
                 "draws after the upscale {} (never upscaled), upscale source drawn into after "
                 "the upscale {} (should stay 0), HUD-less copies {}, camera found in {}/{} frames "
                 "(this frame: {} draws scanned, {} matrix candidates, best seen {} times; scan "
                 "asked {} inactive {} not-scene {} buffers {} too small {})",
                 g_frame_no, g_stats.flip_draws, g_stats.hud_draws, g_stats.src_rewrites,
                 g_stats.hudless, g_stats.camera_frames, g_stats.frames, f.scanned,
                 f.camera.size(),
                 f.camera.empty() ? 0u
                                  : std::max_element(f.camera.begin(), f.camera.end(),
                                                     [](const Candidate& a, const Candidate& b) {
                                                         return a.count < b.count;
                                                     })->count,
                 g_diag_asked, g_diag_inactive, g_diag_noscene, g_diag_calls, g_diag_small);
        if (Active() && scene && !to_window && g_learned.render_w != 0 &&
            g_learned.render_w >= want_w && g_learned.render_h >= want_h && ratio > 1.01) {
            LOG_INFO(Render_Vulkan,
                     "[BB_UPSCALE] the game renders its 3D at {}x{}, not below the window {}x{}: "
                     "{} runs as native AA (nothing to upscale). For {} set the game's internal "
                     "resolution (patch) to {}x{}, or none (native 1920x1080) on this window.",
                     g_learned.render_w, g_learned.render_h, want_w, want_h, SR::KindName(c.kind),
                     SR::QualityName(c.quality), rec_w, rec_h);
        }
        // The periodic line in bb_upscale_init.log.
        Checkpoint(fmt::format(
            "[periodic] frame {}: {} upscaled {} of the last {} frames ({} in all, {} failed), "
            "{}x{} -> {}x{} ({}), window {}x{}, game output {}x{}, GPU {:.2f} ms{}{}",
            g_frame_no, SR::KindName(c.kind), g_stats.injected, g_stats.frames, g_total_upscaled,
            g_stats.failed, g_plan.render_w, g_plan.render_h, g_plan.output_w, g_plan.output_h,
            to_window ? "window-size" : "in place", g_present_w, g_present_h, g_output_w,
            g_output_h, g_stats.gpu_samples ? g_stats.gpu_ms_sum / g_stats.gpu_samples : 0.0,
            [&] {
                const double n = g_stats.stage_samples ? g_stats.stage_samples : 1.0;
                const auto& t = g_stats.stage_ms;
                return fmt::format(
                    " (stages ms: copy {:.2f}, MV+upscaler {:.2f}, alpha {:.2f}, HUD composite "
                    "{:.2f}, present pass {:.2f}; total {:.2f}; {} samples); largest colour "
                    "target {}x{}, {} draws into targets larger than {}x{}",
                    t[0] / n, t[1] / n, t[2] / n, t[3] / n, t[4] / n,
                    (t[0] + t[1] + t[2] + t[3] + t[4]) / n, g_stats.stage_samples, g_stats.big_w,
                    g_stats.big_h, g_stats.big_draws, g_plan.render_w, g_plan.render_h);
            }(),
            g_learned.render_w == 0
                ? "; no 3D scene drawn yet (no depth + 3-target G-buffer pass: title screen, "
                  "menu or loading), nothing to upscale until gameplay"
                : ""));
        // Nothing to upscale: the game already renders at (or above) the window size.
        if (!g_said_nothing && Active() && g_learned.render_w != 0 && g_present_w != 0 &&
            g_learned.render_w >= g_present_w && g_learned.render_h >= g_present_h) {
            g_said_nothing = true;
            Checkpoint(fmt::format(
                "WARNING: NOTHING TO UPSCALE: the game renders its 3D at {}x{} and the window "
                "(swapchain) is {}x{}: {} runs as anti-aliasing only. For a real upscale set the "
                "Resolution Patch below the window, e.g. {}x{} ({}), or make the window larger "
                "(DSR 3840x2160 in exclusive fullscreen).",
                g_learned.render_w, g_learned.render_h, g_present_w, g_present_h,
                SR::KindName(c.kind),
                static_cast<u32>(std::lround(g_present_w / (ratio < 1.01 ? 1.5 : ratio))),
                static_cast<u32>(std::lround(g_present_h / (ratio < 1.01 ? 1.5 : ratio))),
                ratio < 1.01 ? "the native AA mode never upscales; quality = window / 1.5"
                             : SR::QualityName(c.quality)));
        }
        g_stats = {};
    }

    if (BB::Metrics::PanelShown()) { // (bb-metrics-043) the in-game panel's Escala / Resol.
        const auto& mc = Cfg();
        const bool up = Active() && f.injected;
        BB::Metrics::Upscale u;
        u.kind = static_cast<int>(mc.kind);
        u.quality = static_cast<int>(mc.quality);
        u.running = up;
        u.scene = g_learned.render_w != 0;
        u.render_w = up ? g_plan.render_w : (g_learned.render_w != 0 ? g_learned.render_w : g_output_w);
        u.render_h = up ? g_plan.render_h : (g_learned.render_h != 0 ? g_learned.render_h : g_output_h);
        u.output_w = up ? g_plan.output_w : g_output_w;
        u.output_h = up ? g_plan.output_h : g_output_h;
        BB::Metrics::SetUpscale(u);
    }
    if (BB::Metrics::Enabled()) {
        const auto& mc = Cfg();
        if (Active() && f.injected) {
            BB::Metrics::SetUpscaleLine(fmt::format(
                "{} {} {}x{} -> {}x{}", SR::KindName(mc.kind), SR::QualityName(mc.quality),
                g_plan.render_w, g_plan.render_h, g_plan.output_w, g_plan.output_h));
        } else if (mc.kind == SR::Kind::Off || mc.kind == SR::Kind::Native) {
            BB::Metrics::SetUpscaleLine("off");
        } else {
            BB::Metrics::SetUpscaleLine(fmt::format("{} (not running: native)", SR::KindName(mc.kind)));
        }
        // The real sizes for the overlay's "Resol." row (the config's internal size is not what
        // the game renders): the game's 3D size (its output size until a 3D scene is seen) and
        // what reaches the window.
        const bool up = Active() && f.injected;
        const u32 gw = g_learned.render_w != 0 ? g_learned.render_w : g_output_w;
        const u32 gh = g_learned.render_h != 0 ? g_learned.render_h : g_output_h;
        const u32 ow = up ? g_plan.output_w : g_output_w;
        const u32 oh = up ? g_plan.output_h : g_output_h;
        BB::Metrics::SetResLine(fmt::format(
            "game {}x{} -> output {}x{} (window {}x{}, {})", gw, gh, ow, oh, g_present_w,
            g_present_h,
            up ? "upscaled"
            : g_learned.render_w == 0 ? "no 3D scene yet: title / menu / loading"
                                      : "not upscaled"));
    }

    // Jitter for the next frame.
    g_jittered_last_frame = g_jittered_draws - g_jittered_frame_start;
    g_jittered_frame_start = g_jittered_draws;
    if (g_taa_scene_draws > 0) {
        ++g_stab.frames;
        if (g_jittered_last_frame > 0) ++g_stab.frames_jittered;
        if (g_taa_nocam_draws < g_taa_scene_draws) ++g_stab.frames_cam; else ++g_stab.frames_nocam;
    }
    {
        static auto stab_last = std::chrono::steady_clock::now();
        const auto stab_now = std::chrono::steady_clock::now();
        if (stab_now - stab_last >= std::chrono::seconds(1)) {
            stab_last = stab_now;
            const auto& t = g_stab;
            if (t.frames > 0) {
                Checkpoint(fmt::format(
                    "[taa-stab] frames {} jittered {} (camera in vertex constants {}, none {}; "
                    "block camera {}), jittered draws {}, full-screen passes not jittered {}, "
                    "post-point draws not jittered {}, upscaler told jitter {} / zero {}, "
                    "applied-vs-passed mismatches {} (should stay 0), phases {}",
                    t.frames, t.frames_jittered, t.frames_cam, t.frames_nocam,
                    g_learned.have_prev_vp ? "yes" : "no", t.draws, t.fullscreen_skipped,
                    t.post_point_skipped, t.nonzero_passed, t.zero_passed, t.mismatch,
                    g_jitter_phases));
            }
            g_stab = {};
        }
    }
    const double ratio = g_learned.render_w != 0
                             ? static_cast<double>(PresentMode() ? g_present_w : g_output_w) /
                                   g_learned.render_w
                             : 1.0;
    // 0.79: the length FSR 3's ffxFsr3UpscalerGetJitterPhaseCount gives (8 * ratio^2, truncated),
    // the cycle every back end is told about; Halton(2,3) index (phase + 1), minus 0.5.
    // 0.91: at ratio 1 (DLAA / native AA) an 8-phase cycle repeats every 8 frames and shows as a
    // visible on/off shimmer (grass, foliage). Use at least 32 phases; BB_JITTER_PHASES overrides.
    static const u32 min_phases = [] {
        const char* v = std::getenv("BB_JITTER_PHASES");
        const int n = v != nullptr ? std::atoi(v) : 32;
        return static_cast<u32>(std::clamp(n, 8, 128));
    }();
    const u32 phases = std::clamp<u32>(static_cast<u32>(8.0 * ratio * ratio + 1e-6), min_phases, 128);
    g_jitter_phases = phases;
    g_taa_last_scene = g_taa_scene_draws;
    g_taa_last_nocam = g_taa_nocam_draws;
    g_taa_scene_draws = g_taa_nocam_draws = 0;
    g_jitter_index = (g_jitter_index + 1) % phases;
    g_jitter[0] = Halton(g_jitter_index + 1, 2) - 0.5f;
    g_jitter[1] = Halton(g_jitter_index + 1, 3) - 0.5f;
    if (const char* s = std::getenv("BB_JITTER_SIGN"); s != nullptr && std::atof(s) < 0) {
        g_jitter[0] = -g_jitter[0];
        g_jitter[1] = -g_jitter[1];
    }

    ++g_frame_no;
    g_dump_frame = AnalyzeOn() &&
                   (g_frame_no % 1800 == 900 ||
                    (g_ss_frozen.load(std::memory_order_relaxed) &&
                     !g_ss_off.load(std::memory_order_relaxed) && g_ss_windows == 0 &&
                     g_ss_frames == 30));
    g_dump_lines = 0;
    g_dump_cbs = 0;
    g_dump_last.clear();
    g_dump_repeats = 0;
    g_frame = {};
    g_draw = {};
    g_learned.rolled = false;
}

bool HudSharpScale(u32 render_w, u32 render_h, u32 out_w, u32 out_h) {
    static const bool off = [] {
        const char* v = std::getenv("BB_HUD_SCALE");
        return v != nullptr && (std::string_view{v} == "bilinear" || std::string_view{v} == "0");
    }();
    return !off && out_w >= render_w && out_h >= render_h &&
           (out_w > render_w || out_h > render_h);
}

void NoteFullFrame(const char* path, u32 in_w, u32 in_h, u32 out_w, u32 out_h, Hud2D hud,
                   bool is_3d) {
    static std::mutex m;
    static auto start = std::chrono::steady_clock::now();
    static auto last = start;
    static std::string last_key;
    static u32 frames = 0, d3 = 0, hud_up = 0;
    std::scoped_lock lk{m};
    ++frames;
    d3 += is_3d ? 1 : 0;
    const bool scaled2d = hud == Hud2D::Upscaled || hud == Hud2D::Spatial;
    hud_up += scaled2d ? 1 : 0;
    const char* hud_s = hud == Hud2D::Upscaled    ? "YES (window path: HUD layer EASU-like)"
                        : hud == Hud2D::Spatial    ? "YES (whole frame FSR 1 EASU)"
                        : hud == Hud2D::Native     ? "native (drawn by the game at the window size)"
                        : hud == Hud2D::Downscaled ? "NO (internal > window: downscaled)"
                                                   : "NO (plain stretch)";
    const SR::Kind kind = Cfg().kind;
    const std::string up = kind == SR::Kind::Off ? std::string("off (emulator FSR 1 setting)")
                                                 : std::string(SR::KindName(kind));
    const std::string key = fmt::format("{}|{}|{}x{}|{}x{}|{}", up, path, in_w, in_h, out_w,
                                        out_h, hud_s);
    const auto now = std::chrono::steady_clock::now();
    const double since = std::chrono::duration<double>(now - start).count();
    const double gap = std::chrono::duration<double>(now - last).count();
    const bool changed = key != last_key;
    if (!changed && gap < (since < 120.0 ? 1.0 : 30.0)) {
        return;
    }
    Checkpoint(fmt::format("[frame] t={:.0f}s upscaler={} path={} internal={}x{} window={}x{} "
                           "2D layer scaled={} | last {} flips: {} 3D, {} with 2D scaled{}",
                           since, up, path, in_w, in_h, out_w, out_h, hud_s, frames, d3, hud_up,
                           changed ? " (changed)" : ""));
    last = now;
    last_key = key;
    frames = d3 = hud_up = 0;
}

} // namespace BB::Upscale
