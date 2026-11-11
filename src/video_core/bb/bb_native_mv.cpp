// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// BB native motion vectors: see bb_native_mv.h.

#include "video_core/bb/bb_native_mv.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "core/memory.h"
#include "video_core/bb/bb_upscale.h"

namespace BB::NativeMv {

namespace {

constexpr u32 kSamples = 24;     // bones compared per object (spread over the skeleton)
constexpr u32 kSlots = 4;        // guest frames kept per object
constexpr u32 kMaxBones = 1024;
constexpr u64 kMarkerReach = 0x800; // draw packet within this many bytes after the marker
constexpr u64 kRecentFrames = 30;
constexpr float kTolT = 0.004f;  // metres a frame (base), + kTolTRel * |t|
constexpr float kTolTRel = 0.0005f;
constexpr float kTolR = 0.004f;

using Clock = std::chrono::steady_clock;

struct Mat34 {
    float m[12];
};

struct Slot {
    u64 frame = 0;
    u32 n = 0;
    bool has_prev = false;
    int state = -1; // -1 not classified, 0 static, 1 moving
    std::array<Mat34, kSamples> cur{};
    std::array<Mat34, kSamples> prev{};
};

struct Obj {
    u64 last_frame = 0;
    u64 last_moving = 0;
    u32 bones = 0;
    bool validated = false;
    std::array<Slot, kSlots> slots{};
};

struct Marker {
    u64 bones;
    u64 key;
    u64 frame;
    bool skinned;
};

struct Tls {
    u64 bones = 0;
    u64 key = 0;
    u64 frame = 0;
    bool skinned = false;
    bool valid = false;
};
thread_local Tls t_id;

std::atomic<bool> g_on{true};
std::atomic<bool> g_hooks{false};
std::atomic<u64> g_frame{1};
std::atomic<u64> g_dm_begin{0}, g_dm_end{0};

std::mutex g_mx; // objects + markers
std::unordered_map<u64, Obj> g_objs;
std::map<u64, Marker> g_markers;

// Frame camera (common transform) cache, GPU thread under g_mx.
struct CamCache {
    u64 frame = 0;
    u32 used = 0;
    bool valid = false;
    Mat34 c{};
};
CamCache g_cam;
struct Eligible {
    u64 frame = 0;
    u32 count = 0;
};
std::array<Eligible, kSlots> g_eligible{};

DrawId g_cur;
DrawId g_none;

// Statistics.
std::atomic<u64> st_hook_ns{0}, st_hook_calls{0}, st_emit_calls{0};
u64 st_packets = 0, st_matched = 0, st_scene = 0, st_scene_id = 0, st_scene_moving = 0,
    st_vec = 0, st_vec_prev = 0, st_gpu_ns = 0, st_classified = 0, st_moving_obj = 0,
    st_skinned_scene_id = 0, st_skinned_scene_moving = 0, st_frames = 0;
Clock::time_point st_t0 = Clock::now();

u64 Mix(u64 a, u64 b) {
    u64 h = a ^ (b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2));
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    return h;
}

// 3x4 row-major affine (translation in column 3).
Mat34 Mul(const Mat34& a, const Mat34& b) {
    Mat34 r;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            float v = a.m[i * 4 + 0] * b.m[0 * 4 + j] + a.m[i * 4 + 1] * b.m[1 * 4 + j] +
                      a.m[i * 4 + 2] * b.m[2 * 4 + j];
            if (j == 3) v += a.m[i * 4 + 3];
            r.m[i * 4 + j] = v;
        }
    }
    return r;
}

bool Inverse(const Mat34& a, Mat34& r) {
    const float* m = a.m;
    const float c00 = m[5] * m[10] - m[6] * m[9];
    const float c01 = m[6] * m[8] - m[4] * m[10];
    const float c02 = m[4] * m[9] - m[5] * m[8];
    const float det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    if (!(std::fabs(det) > 1e-12f) || !std::isfinite(det)) return false;
    const float id = 1.0f / det;
    float inv[9];
    inv[0] = c00 * id;
    inv[1] = (m[2] * m[9] - m[1] * m[10]) * id;
    inv[2] = (m[1] * m[6] - m[2] * m[5]) * id;
    inv[3] = c01 * id;
    inv[4] = (m[0] * m[10] - m[2] * m[8]) * id;
    inv[5] = (m[2] * m[4] - m[0] * m[6]) * id;
    inv[6] = c02 * id;
    inv[7] = (m[1] * m[8] - m[0] * m[9]) * id;
    inv[8] = (m[0] * m[5] - m[1] * m[4]) * id;
    for (int i = 0; i < 3; ++i) {
        r.m[i * 4 + 0] = inv[i * 3 + 0];
        r.m[i * 4 + 1] = inv[i * 3 + 1];
        r.m[i * 4 + 2] = inv[i * 3 + 2];
        r.m[i * 4 + 3] = -(inv[i * 3 + 0] * m[3] + inv[i * 3 + 1] * m[7] + inv[i * 3 + 2] * m[11]);
    }
    return true;
}

bool Finite(const Mat34& a) {
    for (float v : a.m) {
        if (!std::isfinite(v) || std::fabs(v) > 1e7f) return false;
    }
    return true;
}

// The object's own frame-to-frame transform (root bone): cur0 x prev0^-1.
bool ObjDelta(const Slot& s, Mat34& out) {
    Mat34 inv;
    if (!Inverse(s.prev[0], inv)) return false;
    out = Mul(s.cur[0], inv);
    return Finite(out);
}

// The frame's common transform: component-wise median of the objects' own transforms (most
// objects are static: identity in world space, the camera change when the bones are
// camera-relative).
const Mat34* FrameCommon(u64 frame) {
    const u32 slot = frame % kSlots;
    const u32 eligible = g_eligible[slot].frame == frame ? g_eligible[slot].count : 0u;
    if (g_cam.valid && g_cam.frame == frame && eligible * 4 <= g_cam.used * 5 + 16) {
        return &g_cam.c;
    }
    if (eligible < 3) {
        g_cam = {frame, eligible, false, {}};
        return nullptr;
    }
    std::array<std::vector<float>, 12> comp;
    for (auto& v : comp) v.reserve(eligible);
    for (const auto& [b, o] : g_objs) {
        const Slot& s = o.slots[slot];
        if (s.frame != frame || !s.has_prev) continue;
        Mat34 d;
        if (!ObjDelta(s, d)) continue;
        for (int i = 0; i < 12; ++i) comp[i].push_back(d.m[i]);
    }
    if (comp[0].size() < 3) {
        g_cam = {frame, eligible, false, {}};
        return nullptr;
    }
    Mat34 c;
    for (int i = 0; i < 12; ++i) {
        auto& v = comp[i];
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        c.m[i] = v[v.size() / 2];
    }
    g_cam = {frame, eligible, true, c};
    return &g_cam.c;
}

bool Moved(const Slot& s, const Mat34& c) {
    for (u32 i = 0; i < s.n; ++i) {
        const Mat34 p = Mul(c, s.prev[i]);
        const Mat34& q = s.cur[i];
        for (int r = 0; r < 3; ++r) {
            const float t = q.m[r * 4 + 3];
            if (std::fabs(p.m[r * 4 + 3] - t) > kTolT + kTolTRel * std::fabs(t)) return true;
            for (int k = 0; k < 3; ++k) {
                if (std::fabs(p.m[r * 4 + k] - q.m[r * 4 + k]) > kTolR) return true;
            }
        }
    }
    return false;
}

// GPU thread, under g_mx: 1 moving, 0 static, -1 unknown (first frame seen, or no common
// transform yet).
int Classify(u64 bones, u64 frame, bool& recent) {
    recent = false;
    const auto it = g_objs.find(bones);
    if (it == g_objs.end()) return -1;
    Obj& o = it->second;
    Slot& s = o.slots[frame % kSlots];
    if (s.frame != frame) return -1;
    if (s.state < 0 && s.has_prev) {
        const Mat34* c = FrameCommon(frame);
        if (c != nullptr) {
            s.state = Moved(s, *c) ? 1 : 0;
            ++st_classified;
            if (s.state == 1) {
                ++st_moving_obj;
                o.last_moving = std::max(o.last_moving, frame);
            }
        }
    }
    recent = o.last_moving != 0 && frame >= o.last_moving && frame - o.last_moving <= kRecentFrames;
    return s.state;
}

std::string LiveFile() {
    const char* v = std::getenv("BB_NATIVE_MV_FILE");
    return v != nullptr && *v != '\0' ? std::string(v) : std::string("bb_native_mv.txt");
}

void PollLive() {
    static Clock::time_point next{};
    static std::filesystem::file_time_type seen{};
    static const std::string path = LiveFile();
    const auto now = Clock::now();
    if (now < next) return;
    next = now + std::chrono::milliseconds(500);
    std::error_code ec;
    const auto stamp = std::filesystem::last_write_time(path, ec);
    if (ec || stamp == seen) return;
    seen = stamp;
    std::ifstream in(path);
    std::string text;
    std::getline(in, text);
    const auto eq = text.find('=');
    if (eq != std::string::npos) text = text.substr(eq + 1);
    if (text.empty()) return;
    const bool on = text[0] != '0';
    if (on != g_on.load()) {
        g_on.store(on);
        const std::string msg =
            fmt::format("[native-mv] live switch ({}): {}", path,
                        on ? "on (game-identity object vectors, static objects camera-only)"
                           : "off (previous behaviour)");
        LOG_INFO(Render_Vulkan, "{}", msg);
        BB::Upscale::Checkpoint(msg);
    }
}

void Log(const std::string& text) {
    LOG_INFO(Render_Vulkan, "{}", text);
    BB::Upscale::Checkpoint(text);
}

} // namespace

bool On() {
    return g_on.load(std::memory_order_relaxed) && g_hooks.load(std::memory_order_relaxed);
}

bool HooksInstalled() {
    return g_hooks.load(std::memory_order_relaxed);
}

void SetHooksInstalled(bool installed, const char* detail) {
    static const bool env_on = [] {
        const char* v = std::getenv("BB_NATIVE_MV");
        return v == nullptr || *v == '\0' || (v[0] != '0' && std::string(v) != "off");
    }();
    g_on.store(env_on);
    g_hooks.store(installed);
    Log(fmt::format("[native-mv] {} ({}); BB_NATIVE_MV={} (live file {})",
                    installed ? "guest hooks in place" : "no guest hooks", detail,
                    env_on ? 1 : 0, LiveFile()));
}

void SetDrawMeshRange(u64 begin, u64 end) {
    g_dm_begin.store(begin);
    g_dm_end.store(end);
}

void OnGuestFlip() {
    g_frame.fetch_add(1, std::memory_order_relaxed);
}

void GuestDrawMesh(u64 model, u64 mesh, u64 instance) {
    t_id.valid = false;
    if (!On() || instance == 0 || model == 0) return;
    const auto t0 = Clock::now();
    const u64 bones = *reinterpret_cast<const u64*>(instance);
    if (bones == 0 || (bones & 15) != 0) return;
    s32 count = 0;
    std::memcpy(&count, reinterpret_cast<const void*>(model + 0x1c), 4);
    const u32 n = count >= 1 && count <= s32(kMaxBones) ? u32(count) : 1u;
    const u64 frame = g_frame.load(std::memory_order_relaxed);
    {
        std::scoped_lock lk{g_mx};
        Obj& o = g_objs[bones];
        if (!o.validated || o.bones != n) {
            auto* memory = Core::Memory::Instance();
            if (memory == nullptr || !memory->IsValidMapping(bones, u64(n) * 48)) {
                g_objs.erase(bones);
                return;
            }
            o.validated = true;
            o.bones = n;
        }
        if (o.last_frame != frame) {
            Slot& s = o.slots[frame % kSlots];
            const Slot& p = o.slots[(frame + kSlots - 1) % kSlots];
            const u32 ns = std::min(n, kSamples);
            s.frame = frame;
            s.n = ns;
            s.state = -1;
            const auto* src = reinterpret_cast<const Mat34*>(bones);
            for (u32 i = 0; i < ns; ++i) {
                const u32 b = ns > 1 ? u32(u64(i) * (n - 1) / (ns - 1)) : 0u;
                s.cur[i] = src[b];
            }
            s.has_prev = o.last_frame + 1 == frame && p.frame + 1 == frame && p.n == ns;
            if (s.has_prev) {
                s.prev = p.cur;
                Eligible& e = g_eligible[frame % kSlots];
                if (e.frame != frame) e = {frame, 0};
                ++e.count;
            }
            o.last_frame = frame;
        }
    }
    t_id.bones = bones;
    t_id.key = Mix(bones, mesh);
    t_id.frame = frame;
    t_id.skinned = n > 1;
    t_id.valid = true;
    st_hook_calls.fetch_add(1, std::memory_order_relaxed);
    st_hook_ns.fetch_add(u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 Clock::now() - t0)
                                 .count()),
                         std::memory_order_relaxed);
}

void GuestDrawEmit(u64 gx_context, u64 return_address) {
    if (!t_id.valid || gx_context == 0) return;
    if (return_address < g_dm_begin.load(std::memory_order_relaxed) ||
        return_address >= g_dm_end.load(std::memory_order_relaxed)) {
        return;
    }
    const auto t0 = Clock::now();
    const u64 cmdptr = *reinterpret_cast<const u64*>(gx_context + 0x20);
    if (cmdptr == 0) return;
    {
        std::scoped_lock lk{g_mx};
        if (g_markers.size() > 200000) g_markers.clear();
        auto it = g_markers.upper_bound(cmdptr);
        while (it != g_markers.end() && it->first <= cmdptr + kMarkerReach) {
            it = g_markers.erase(it);
        }
        g_markers[cmdptr] = Marker{t_id.bones, t_id.key, t_id.frame, t_id.skinned};
    }
    st_emit_calls.fetch_add(1, std::memory_order_relaxed);
    st_hook_ns.fetch_add(u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 Clock::now() - t0)
                                 .count()),
                         std::memory_order_relaxed);
}

void SetDrawPacket(u64 packet_address) {
    g_cur = DrawId{};
    if (!On()) return;
    const auto t0 = Clock::now();
    ++st_packets;
    {
        std::scoped_lock lk{g_mx};
        auto it = g_markers.upper_bound(packet_address);
        if (it != g_markers.begin()) {
            --it;
            if (packet_address - it->first < kMarkerReach) {
                const Marker m = it->second;
                g_markers.erase(it);
                ++st_matched;
                bool recent = false;
                const int state = Classify(m.bones, m.frame, recent);
                g_cur.key = m.key;
                g_cur.moving = state == 1;
                g_cur.recent = recent || state == 1;
                g_cur.skinned = m.skinned;
            }
        }
    }
    st_gpu_ns += u64(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
}

const DrawId& Current() {
    return On() ? g_cur : g_none;
}

void NoteSceneDraw(bool motion_candidate) {
    if (!motion_candidate) return;
    ++st_scene;
    if (g_cur.key != 0) {
        ++st_scene_id;
        if (g_cur.moving) ++st_scene_moving;
        if (g_cur.skinned) {
            ++st_skinned_scene_id;
            if (g_cur.moving) ++st_skinned_scene_moving;
        }
    }
}

void NoteVectorDraw(bool with_previous) {
    ++st_vec;
    if (with_previous) ++st_vec_prev;
}

void OnPresent() {
    PollLive();
    ++st_frames;
    const auto now = Clock::now();
    const double sec = std::chrono::duration<double>(now - st_t0).count();
    if (sec < 1.0) return;
    st_t0 = now;
    u64 tracked = 0, frame = g_frame.load();
    {
        std::scoped_lock lk{g_mx};
        for (auto it = g_objs.begin(); it != g_objs.end();) {
            if (it->second.last_frame + 120 < frame) {
                it = g_objs.erase(it);
            } else {
                if (it->second.last_frame + 60 >= frame) ++tracked;
                ++it;
            }
        }
    }
    const u64 hook_calls = st_hook_calls.exchange(0), emits = st_emit_calls.exchange(0);
    const u64 hook_ns = st_hook_ns.exchange(0);
    const double frames = std::max<double>(1.0, double(st_frames));
    if (HooksInstalled() || hook_calls != 0) {
        const double cov_id = st_scene ? 100.0 * double(st_scene_id) / double(st_scene) : 0.0;
        const double cov_char = st_skinned_scene_id
                                    ? 100.0 * double(st_skinned_scene_moving) /
                                          double(st_skinned_scene_id)
                                    : 0.0;
        Log(fmt::format(
            "[native-mv] {}: {} objects tracked (game bone arrays), moving {:.1f} a frame of {:.1f} "
            "classified; scene draws {:.0f}/frame, with game identity {:.1f}% ({:.0f}/frame), "
            "moving {:.1f}/frame; character (skinned) draws moving {:.1f}% of {:.0f}/frame; "
            "{:.1f} draws/frame with game-identity previous transforms (vectors), {:.1f} without "
            "history; draw packets {:.0f}/frame, matched {:.0f}/frame; DrawMesh calls {:.0f}/frame, "
            "draw markers {:.0f}/frame; CPU added: guest hooks {:.3f} ms/frame, GPU thread {:.3f} "
            "ms/frame",
            On() ? "on" : "off", tracked, double(st_moving_obj) / frames,
            double(st_classified) / frames, double(st_scene) / frames, cov_id,
            double(st_scene_id) / frames, double(st_scene_moving) / frames, cov_char,
            double(st_skinned_scene_id) / frames, double(st_vec_prev) / frames,
            double(st_vec - st_vec_prev) / frames, double(st_packets) / frames,
            double(st_matched) / frames, double(hook_calls) / frames, double(emits) / frames,
            double(hook_ns) / 1e6 / frames, double(st_gpu_ns) / 1e6 / frames));
    }
    st_packets = st_matched = st_scene = st_scene_id = st_scene_moving = st_vec = st_vec_prev = 0;
    st_gpu_ns = st_classified = st_moving_obj = st_skinned_scene_id = st_skinned_scene_moving = 0;
    st_frames = 0;
}

} // namespace BB::NativeMv
