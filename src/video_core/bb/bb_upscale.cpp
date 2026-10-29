// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/bb/bb_upscale.h"

#include <algorithm>
#include <chrono>
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
    if (const char* s = Env("BB_UPSCALE_SHARPNESS")) {
        c.sharpness = std::clamp(static_cast<float>(std::atof(s)), 0.0f, 1.0f);
    }
    if (const char* s = Env("BB_UPSCALE_JITTER")) {
        c.jitter = s[0] != '0';
    }
    c.mv_reconstruct = Lower(Env("BB_MV")) != "zero";
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
        if (Env("BB_FSR4_SHARPNESS") == nullptr) {
            _putenv_s("BB_FSR4_SHARPNESS", fmt::format("{}", c.sharpness).c_str());
        }
    }
#endif
    return c;
}

// BB_ANALYZE dumps also go to their own file in the working folder, bb_analyze_<upscaler>.log
// (rewritten per run of that upscaler, so a later run of another one keeps it).
template <typename... Args>
void AnalyzeLog(fmt::format_string<Args...> format, Args&&... args) {
    const std::string text = fmt::format(format, std::forward<Args>(args)...);
    LOG_INFO(Render_Vulkan, "{}", text);
    static std::FILE* file = [] {
        const char* up = std::getenv("BB_UPSCALER");
        std::string name = std::string("bb_analyze_") + (up && *up ? up : "off") + ".log";
        return std::fopen(name.c_str(), "w");
    }();
    static u32 lines = 0;
    if (file != nullptr && lines++ < 200000) {
        std::fprintf(file, "%s\n", text.c_str());
        std::fflush(file);
    }
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
    // G-buffer
    VAddr gb_depth = 0;
    u32 gb_w = 0, gb_h = 0, gb_mrt = 0;
    bool depth_greater = true;
    Viewport viewport;
    // the scene colour and the composite source this frame
    VAddr scene_color = 0;
    VAddr comp_src = 0;
    u32 comp_w = 0, comp_h = 0;
    // sample ordering for the learned injection point
    u32 qual_samples = 0;
    u32 last_write_after = 0;
    // draws
    u64 draws = 0;
    u32 scene_draws = 0;
    u32 scanned = 0;
    u64 scanned_draw = ~0ULL;
    u32 flip_draws = 0, hud_draws = 0, hud_other = 0, src_rewrites = 0;
    bool injected = false, inject_failed = false, redirecting = false, hudless = false;
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
    bool last_injected = false;
};

struct Stats {
    u64 frames = 0, injected = 0, failed = 0;
    double gpu_ms_sum = 0, gpu_ms_max = 0;
    u32 gpu_samples = 0;
    u64 flip_draws = 0, hud_draws = 0, hud_other = 0, src_rewrites = 0, hudless = 0;
    u64 camera_frames = 0;
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
std::chrono::steady_clock::time_point g_last_flip{};
float g_delta_ms = 16.7f;
std::string g_last_point;
bool g_first_eval_pending = false;
// Upscaling to the window size (SetPresentSize).
u32 g_present_w = 0, g_present_h = 0;
bool g_present_dead = false;
bool g_late_upscale = false; // no HUD point this frame: the presenter upscales the flip image

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
} g_draw;

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

bool SceneMode() {
    const auto& c = Cfg();
    if (c.force_mode == 1) return true;
    if (c.force_mode == 2) return false;
    return g_learned.render_w == g_output_w && g_learned.render_h == g_output_h;
}

VAddr LearnedSource() {
    if (Cfg().force_source != 0) return Cfg().force_source;
    return SceneMode() ? g_learned.scene_color : g_learned.comp_src;
}

void FillPresentPlan(VAddr flip, const char* where);

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
    if (best == nullptr) {
        for (const auto& c : g_frame.camera) {
            if (best == nullptr || c.count > best->count) best = &c;
        }
    }
    if (best == nullptr) return nullptr;
    const u32 need = std::max<u32>(3, g_frame.scanned / 3);
    return best->count >= need ? best : nullptr;
}

void FillCamera(Plan& p) {
    p.camera_valid = false;
    const Candidate* cam = Cfg().mv_reconstruct ? BestCamera() : nullptr;
    if (cam == nullptr) {
        p.depth_inverted = g_frame.depth_greater;
        return;
    }
    const Mat& vp = cam->m;
    Mat inv;
    if (!Invert(vp, inv)) return;
    const Mat prev = g_learned.have_prev_vp ? g_learned.prev_vp : vp;
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
    p.camera_valid = true;
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
        // Appended, the last 3 runs kept: a crashing run survives the native run after it.
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
        if (starts.size() >= 3) {
            old = old.substr(starts[starts.size() - 2]);
        }
        file = std::fopen("bb_upscale_init.log", "wb");
        if (file != nullptr) {
            std::fwrite(old.data(), 1, old.size(), file);
            const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            char when[64]{};
            std::strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
            std::fprintf(file, "==== run %s (BB_UPSCALER=%s) ====\n", when,
                         Env("BB_UPSCALER") ? Env("BB_UPSCALER") : "");
            std::fflush(file);
        }
    }
    if (file != nullptr && lines < 600) {
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

bool Enabled() {
    return Cfg().kind != SR::Kind::Off;
}

void PreInstance() {
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
    const auto& c = Cfg();
    if (!g_guard_skipped.empty()) {
        Checkpoint(g_guard_skipped);
    }
    if (c.kind == SR::Kind::Off) {
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
    if (c.kind != SR::Kind::Off && c.kind != SR::Kind::Native) {
        GuardBegin(SR::KindName(c.kind), "loading the upscaler and creating the Vulkan device");
    }
    if (c.kind == SR::Kind::Dlss) {
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
    return Cfg().kind == SR::Kind::Dlss ? SR::Dlss::LoaderPath() : nullptr;
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
    if (Cfg().kind == SR::Kind::Dlss && SR::Dlss::Manual()) {
        // Manual hooking: what Streamline's features asked for (slGetFeatureRequirements).
        for (const auto& name : SR::Dlss::RequiredDeviceExtensions()) {
            // (Not the EXT buffer device address with the core / KHR one: both is invalid.)
            if (name == VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) continue;
            add(name.c_str());
        }
    }
    if (Cfg().kind == SR::Kind::Dlss && SR::Dlss::LoaderPath() != nullptr) {
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
    }
}

void* ChainDeviceFeatures(vk::PhysicalDevice physical, const std::vector<const char*>& extensions,
                          vk::PhysicalDeviceVulkan12Features& f12,
                          vk::PhysicalDeviceVulkan13Features& f13, vk::PhysicalDeviceFeatures& f,
                          void* next) {
    if (Cfg().kind == SR::Kind::Dlss) {
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
    if (Cfg().kind == SR::Kind::Dlss && SR::Dlss::LoaderPath() != nullptr) {
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
            return &present_id;
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

void OnDraw(std::span<const Target> colors, const Target* depth, bool depth_greater,
            const Viewport& viewport) {
    if (!Enabled() && !AnalyzeOn()) {
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
    }
    const VAddr scene_depth = f.gb_depth != 0 ? f.gb_depth : g_learned.gb_depth;
    g_draw.scene = !f.injected && g_draw.has_depth && depth->addr == scene_depth;
    if (g_draw.scene) {
        ++f.scene_draws;
        // The scene colour: the lit image the forward passes draw into (one or two targets).
        if (mrt <= 2 && colors.size() > 0 && IsHdr(colors[0].format) &&
            colors[0].width == depth->width && colors[0].height == depth->height) {
            f.scene_color = colors[0].addr;
        }
    }
    // Writes to the injection source.
    const VAddr src = LearnedSource();
    for (u32 i = 0; i < g_draw.num_colors; ++i) {
        if (src != 0 && colors[i].addr == src) {
            f.last_write_after = f.qual_samples;
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
    if (g_dump_frame && g_dump_lines < 1500) {
        g_draw.analyze_dump = true;
        std::string line = fmt::format("[BB_ANALYZE] frame={} draw {} targets:", g_frame_no, f.draws);
        for (u32 i = 0; i < g_draw.num_colors; ++i) {
            if (colors[i].addr) line += fmt::format(" c{}={}", i, Describe(colors[i]));
        }
        if (g_draw.has_depth) line += fmt::format(" d={}", Describe(*depth));
        if (g_draw.targets_flip) line += " [FLIP]";
        if (g_draw.scene) line += " [SCENE]";
        const std::string key = line.substr(line.find("targets:"));
        if (key == g_dump_last && !g_draw.scene) {
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
    if (!Enabled() && !AnalyzeOn()) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    ++g_frame.draws;
    g_draw = {};
    g_draw.valid = true;
    g_draw.compute = true;
    if (g_dump_frame && g_dump_lines < 1500) {
        g_draw.analyze_dump = true;
        AnalyzeLog("[BB_ANALYZE] frame={} dispatch {}", g_frame_no, g_frame.draws);
        ++g_dump_lines;
    }
}

u64 g_diag_asked = 0, g_diag_inactive = 0, g_diag_noscene = 0, g_diag_calls = 0, g_diag_small = 0;

bool WantCameraScan() {
    ++g_diag_asked;
    if ((!Active() && !AnalyzeOn()) || !Cfg().mv_reconstruct) {
        ++g_diag_inactive;
        return false;
    }
    std::scoped_lock lk{g_mutex};
    const bool want = g_draw.valid && g_draw.scene && g_frame.scanned < 160;
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
    const u64 draw = f.draws;
    bool counted_draw = f.scanned_draw == draw;
    if (g_draw.analyze_dump && g_dump_lines < 1500 && g_dump_cbs < 64) {
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
    for (size_t at = 0; at + 16 <= n; at += 4) {
        for (int layout = 0; layout < 2; ++layout) {
            Mat m;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    m.m[i * 4 + j] = layout == 0 ? w[at + i * 4 + j] : w[at + j * 4 + i];
            bool hit = LooksLikeViewProj(m);
            bool inverse = false;
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
            if (g_draw.analyze_dump && g_dump_lines < 1500) {
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
    if (g_draw.analyze_dump && g_dump_lines < 1500) {
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
            return Sample::Redirect;
        }
        return Sample::Keep;
    }
    // A qualifying sample of the source: the injection is at the one after its last write.
    const u32 index = f.qual_samples++;
    if (PresentMode()) {
        return Sample::Keep; // upscaled before the HUD instead (WantPresentUpscale)
    }
    if (index < g_learned.skip || !Active() || f.inject_failed) {
        return Sample::Keep;
    }
    if (width != (f.gb_w ? f.gb_w : g_learned.render_w) ||
        height != (f.gb_h ? f.gb_h : g_learned.render_h)) {
        return Sample::Keep;
    }
    const bool scene = SceneMode();
    if (!scene && !out_size) {
        return Sample::Keep;
    }
    // The plan.
    Plan& p = g_plan;
    p = {};
    p.scene_mode = scene;
    p.color_addr = addr;
    p.depth_addr = f.gb_depth != 0 ? f.gb_depth : g_learned.gb_depth;
    p.render_w = width;
    p.render_h = height;
    p.output_w = scene ? width : g_draw.colors[0].width;
    p.output_h = scene ? height : g_draw.colors[0].height;
    p.hdr = IsHdr(format);
    p.reset = !g_learned.last_injected;
    p.delta_ms = g_delta_ms;
    p.viewport = f.viewport;
    p.depth_inverted = f.depth_greater;
    FillCamera(p);
    const bool jitter_on = Cfg().jitter && p.camera_valid;
    p.jitter[0] = jitter_on ? g_jitter[0] : 0.0f;
    p.jitter[1] = jitter_on ? g_jitter[1] : 0.0f;
    LogPoint(fmt::format(
        "injection point: {} mode, source {:#x} {}x{} {} ({}), depth {:#x}, output {}x{}, "
        "after {} earlier sample(s) of it, motion vectors {}, depth {}",
        scene ? "scene (pre-post HDR)" : "composite (post-tonemap)", addr, width, height,
        vk::to_string(format), p.hdr ? "HDR" : "LDR", p.depth_addr, p.output_w, p.output_h,
        g_learned.skip, p.camera_valid ? "reconstructed (camera found)" : "zero (no camera)",
        p.depth_inverted ? "inverted" : "forward"));
    static bool first = true;
    if (first) {
        first = false;
        GuardBegin(SR::KindName(Cfg().kind), "the first upscale");
        Checkpoint(fmt::format("{}: first upscale {}x{} -> {}x{}", SR::KindName(Cfg().kind),
                               p.render_w, p.render_h, p.output_w, p.output_h));
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

bool WantPresentUpscale(VAddr& flip) {
    if (!Enabled()) {
        return false;
    }
    std::scoped_lock lk{g_mutex};
    auto& f = g_frame;
    if (!g_draw.valid || g_draw.compute || !g_draw.targets_flip || f.injected ||
        f.inject_failed || !PresentMode()) {
        return false;
    }
    // After the scene and its last read by the post-processing; the draw into the presented image
    // that samples a render-size image is the game's final post-processing pass, not the HUD.
    if (f.gb_depth == 0 || f.scene_draws == 0 || f.qual_samples <= g_learned.skip ||
        g_draw.samples_render_size || f.gb_w != g_output_w || f.gb_h != g_output_h) {
        return false;
    }
    flip = 0;
    for (u32 i = 0; i < g_draw.num_colors; ++i) {
        const auto& t = g_draw.colors[i];
        if (IsFlip(t.addr) && t.width == f.gb_w && t.height == f.gb_h) {
            flip = t.addr;
            break;
        }
    }
    if (flip == 0) {
        return false;
    }
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
    p.delta_ms = g_delta_ms;
    p.viewport = f.viewport;
    p.depth_inverted = f.depth_greater;
    FillCamera(p);
    const bool jitter_on = Cfg().jitter && p.camera_valid;
    p.jitter[0] = jitter_on ? g_jitter[0] : 0.0f;
    p.jitter[1] = jitter_on ? g_jitter[1] : 0.0f;
    LogPoint(fmt::format(
        "injection point: window-size upscale {}, source {:#x} (presented image, LDR) {}x{}, "
        "depth {:#x}, output {}x{} (window), motion vectors {}, depth {}",
        where, flip, p.render_w, p.render_h, p.depth_addr, p.output_w, p.output_h,
        p.camera_valid ? "reconstructed (camera found)" : "zero (no camera)",
        p.depth_inverted ? "inverted" : "forward"));
    static bool first = true;
    if (first) {
        first = false;
        GuardBegin(SR::KindName(Cfg().kind), "the first upscale");
        Checkpoint(fmt::format("{}: {}x{} -> {}x{} (game output {}x{} upscaled to the window "
                               "size, HUD added on top at present)",
                               SR::KindName(Cfg().kind), p.render_w, p.render_h, p.output_w,
                               p.output_h, g_output_w, g_output_h));
        g_first_eval_pending = true;
    }
}
} // namespace

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
    if (!g_present_dead) {
        g_present_dead = true;
        Checkpoint(fmt::format("{}: the window-size upscale failed: back to the in-place upscale "
                               "at the game's size",
                               SR::KindName(Cfg().kind)));
    }
}

void OnPresentFailed(std::string_view why) {
    std::scoped_lock lk{g_mutex};
    if (!g_present_dead) {
        g_present_dead = true;
        Checkpoint(fmt::format("{}: window-size upscale off ({}): back to the in-place upscale "
                               "at the game's size",
                               SR::KindName(Cfg().kind), why));
    }
}

const Plan& CurrentPlan() {
    return g_plan;
}

void OnInjected(bool ok, float) {
    std::scoped_lock lk{g_mutex};
    if (ok) {
        g_frame.injected = true;
        g_frame.redirecting = true;
        g_draw.sampled_upscaled = true;
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

bool Jitter(float& x, float& y) {
    if (!Active() || !Cfg().jitter) {
        return false;
    }
    std::scoped_lock lk{g_mutex};
    if (!g_draw.valid || !g_draw.scene || !g_learned.have_prev_vp || LearnedSource() == 0) {
        return false;
    }
    x = g_jitter[0];
    y = g_jitter[1];
    return x != 0.0f || y != 0.0f;
}

bool WantHudlessCopy() {
    std::scoped_lock lk{g_mutex};
    if (!g_draw.valid || !g_draw.targets_flip) {
        return false;
    }
    if (g_frame.injected && !g_draw.sampled_upscaled) {
        ++g_frame.hud_draws;
    }
    if (!Cfg().hudless_copy || !g_frame.injected || g_frame.hudless || g_draw.sampled_upscaled) {
        return false;
    }
    g_frame.hudless = true;
    return true;
}

void OnHudlessCopied() {
    std::scoped_lock lk{g_mutex};
    ++g_stats.hudless;
}

void OnFlip(VAddr addr, u32 width, u32 height) {
    if (!Enabled() && !AnalyzeOn()) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    auto& f = g_frame;
    g_flips[g_flip_index++ % g_flips.size()] = addr;
    g_output_w = width;
    g_output_h = height;
    const auto now = std::chrono::steady_clock::now();
    if (g_last_flip.time_since_epoch().count() != 0) {
        g_delta_ms = std::chrono::duration<float, std::milli>(now - g_last_flip).count();
    }
    g_last_flip = now;

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
    if (cam != nullptr) {
        g_learned.prev_vp = cam->m;
        g_learned.have_prev_vp = true;
        ++g_stats.camera_frames;
    } else {
        g_learned.have_prev_vp = false;
    }
    g_learned.last_injected = f.injected;
    if (g_first_eval_pending) {
        g_first_eval_pending = false;
        GuardEnd();
        Checkpoint(fmt::format("{}: first upscaled frame {}", SR::KindName(Cfg().kind),
                               f.injected ? "done" : "FAILED: the game runs native"));
    }
    // The scene was drawn but no HUD point was found (no HUD this frame, e.g. a cut scene): the
    // presenter upscales the finished image instead (TakeLateUpscale).
    g_late_upscale = false;
    if (PresentMode() && !f.injected && !f.inject_failed && f.gb_depth != 0 &&
        f.gb_w == width && f.gb_h == height && f.qual_samples > g_learned.skip) {
        FillPresentPlan(addr, "at present (no HUD draw found this frame)");
        g_late_upscale = true;
        f.injected = true;
        g_learned.last_injected = true;
    }

    ++g_stats.frames;
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
        const u32 out_w = to_window ? g_present_w : g_output_w;
        const u32 out_h = to_window ? g_present_h : g_output_h;
        const u32 rec_w = static_cast<u32>(std::lround(out_w / ratio));
        const u32 rec_h = static_cast<u32>(std::lround(out_h / ratio));
        const bool scene = SceneMode();
        LOG_INFO(Render_Vulkan,
                 "[BB_UPSCALE] frame {}: upscaler {} ({}, {}), {} mode, internal {}x{} -> output "
                 "{}x{} (mode wants internal {}x{}), source {:#x}, depth {:#x}, MV {}, jitter {} "
                 "({:+.3f},{:+.3f}), upscale GPU ms avg {:.3f} max {:.3f} ({} samples), injected "
                 "{}/{} frames, failed {}",
                 g_frame_no, SR::KindName(c.kind), SR::QualityName(c.quality),
                 BackendReady() ? "ready" : "not ready",
                 to_window ? "window-size (before the HUD)" : scene ? "scene" : "composite",
                 g_learned.render_w, g_learned.render_h, out_w, out_h, rec_w, rec_h,
                 LearnedSource(), g_learned.gb_depth,
                 !c.mv_reconstruct ? "zero (BB_MV=zero)"
                 : g_learned.have_prev_vp ? "reconstructed (depth + camera)"
                                          : "zero (camera not found)",
                 Cfg().jitter && g_learned.have_prev_vp ? "on" : "off", g_jitter[0], g_jitter[1],
                 g_stats.gpu_samples ? g_stats.gpu_ms_sum / g_stats.gpu_samples : 0.0,
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
        if (Active() && scene && !to_window && (g_learned.render_w == g_output_w) &&
            ratio > 1.01) {
            LOG_INFO(Render_Vulkan,
                     "[BB_UPSCALE] the game renders its 3D at the output size {}x{}: {} runs as "
                     "native AA. For {} set the game's internal resolution (patch) to {}x{}.",
                     g_output_w, g_output_h, SR::KindName(c.kind), SR::QualityName(c.quality),
                     rec_w, rec_h);
        }
        g_stats = {};
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
    }

    // Jitter for the next frame.
    const double ratio = g_learned.render_w != 0
                             ? static_cast<double>(PresentMode() ? g_present_w : g_output_w) /
                                   g_learned.render_w
                             : 1.0;
    const u32 phases = std::clamp<u32>(static_cast<u32>(std::ceil(8.0 * ratio * ratio)), 8, 64);
    g_jitter_index = (g_jitter_index + 1) % phases;
    g_jitter[0] = Halton(g_jitter_index + 1, 2) - 0.5f;
    g_jitter[1] = Halton(g_jitter_index + 1, 3) - 0.5f;
    if (const char* s = std::getenv("BB_JITTER_SIGN"); s != nullptr && std::atof(s) < 0) {
        g_jitter[0] = -g_jitter[0];
        g_jitter[1] = -g_jitter[1];
    }

    ++g_frame_no;
    g_dump_frame = AnalyzeOn() && (g_frame_no % 1800 == 900);
    g_dump_lines = 0;
    g_dump_cbs = 0;
    g_dump_last.clear();
    g_dump_repeats = 0;
    g_frame = {};
    g_draw = {};
}

} // namespace BB::Upscale
