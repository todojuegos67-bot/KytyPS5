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

// Per-character GPU skin rings (SkinRingUpdate, eboot 0x21c9ce0 base / 0x21c8c50 1.09): the
// ring object keeps [ring+0x9c] palette slots, arrays [ring+0x10] / [ring+0x18] (one entry per
// slot), index [ring+0x98], skeleton [ring+0xb0] (bone count +0x1c). Its draws read a slot's
// palette: the palette's GPU address names the character.
struct Ring {
    u64 bones = 0;
    u64 frame = 0;
    u32 n = 0;
    bool skinned = true;
    std::array<u64, 16> entries{}; // [ring+0x10][i], [ring+0x18][i]
    u32 count = 0;
    u32 valid_arrays = 0;
    u64 mapped_frame = 0;
};
std::unordered_map<u64, Ring> g_rings;
std::map<u64, std::pair<u64, u64>> g_ring_addr; // palette buffer address -> (ring, last seen)
constexpr u64 kRingSpan = 0x10000;
std::atomic<int> g_ring_offset{0x1018}; // the GX buffer object's V# at +0x18 (1.09 0x2168a20)
u64 g_present = 1, g_map_present = 0;       // -1: entries are the addresses; else [entry + offset]
std::vector<u64> g_missed;                // palette addresses of draws without identity (learning)
u64 st_ring_calls = 0, st_ring_hits = 0, st_ring_miss = 0;

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

// A buffer V#'s first 8 bytes: base address bits 0..43.
u64 VsharpBase(u64 v) {
    return (v & 0xffffffffull) | (((v >> 32) & 0xfffull) << 32);
}

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

const Slot* FindSlot(const Obj& o, u64 frame) {
    for (const Slot& s : o.slots) {
        if (s.frame == frame) return &s;
    }
    return nullptr;
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
        const Slot* sp = FindSlot(o, frame);
        if (sp == nullptr || !sp->has_prev) continue;
        const Slot& s = *sp;
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
    Slot* sp = const_cast<Slot*>(FindSlot(o, frame));
    if (sp == nullptr) return -1;
    Slot& s = *sp;
    if (s.state < 0 && s.has_prev) {
        static const Mat34 ident{{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}};
        const Mat34* c = FrameCommon(frame);
        {
            const bool moved = Moved(s, ident) && (c == nullptr || Moved(s, *c));
            s.state = moved ? 1 : 0;
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

// Under g_mx: this frame's record of an object's bones (false: not readable).
bool RecordBones(u64 bones, u32 n, u64 frame) {
    Obj& o = g_objs[bones];
    if (!o.validated || o.bones != n) {
        auto* memory = Core::Memory::Instance();
        if (memory == nullptr || !memory->IsValidMapping(bones, u64(n) * 48)) {
            g_objs.erase(bones);
            return false;
        }
        o.validated = true;
        o.bones = n;
    }
    if (o.last_frame != frame) {
        // The newest earlier record (guest flips may skip a number when the game flips
        // through more than one entry point).
        u32 pi = 0;
        for (u32 k = 1; k < kSlots; ++k) {
            if (o.slots[k].frame > o.slots[pi].frame) pi = k;
        }
        const u32 si = (pi + 1) % kSlots;
        Slot& s = o.slots[si];
        const Slot& p = o.slots[pi];
        const u32 ns = std::min(n, kSamples);
        s.frame = frame;
        s.n = ns;
        s.state = -1;
        const auto* src = reinterpret_cast<const Mat34*>(bones);
        for (u32 i = 0; i < ns; ++i) {
            const u32 b = ns > 1 ? u32(u64(i) * (n - 1) / (ns - 1)) : 0u;
            s.cur[i] = src[b];
        }
        s.has_prev = p.frame != 0 && p.frame < frame && frame - p.frame <= 4 && p.n == ns;
        if (s.has_prev) {
            s.prev = p.cur;
            Eligible& e = g_eligible[frame % kSlots];
            if (e.frame != frame) e = {frame, 0};
            ++e.count;
        }
        o.last_frame = frame;
    }
    return true;
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
        if (!RecordBones(bones, n, frame)) return;
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

void GuestSkinRing(u64 ring, u64 bones) {
    if (!On() || ring == 0 || bones == 0 || (bones & 15) != 0) return;
    const auto t0 = Clock::now();
    const u64 skel = *reinterpret_cast<const u64*>(ring + 0xb0);
    s32 count = 0, slots = 0;
    if (skel != 0) std::memcpy(&count, reinterpret_cast<const void*>(skel + 0x1c), 4);
    std::memcpy(&slots, reinterpret_cast<const void*>(ring + 0x9c), 4);
    const u32 n = count >= 1 && count <= s32(kMaxBones) ? u32(count) : 1u;
    const u64 frame = g_frame.load(std::memory_order_relaxed);
    {
        std::scoped_lock lk{g_mx};
        ++st_ring_calls;
        if (!RecordBones(bones, n, frame)) return;
        Ring& r = g_rings[ring];
        const bool fresh = r.bones != bones || r.count == 0 || frame > r.mapped_frame + 120;
        r.bones = bones;
        r.frame = frame;
        r.n = n;
        r.skinned = n > 1;
        if (slots >= 1 && slots <= 8) {
            const u64 arr[2] = {*reinterpret_cast<const u64*>(ring + 0x10),
                                *reinterpret_cast<const u64*>(ring + 0x18)};
            if (fresh) {
                auto* memory = Core::Memory::Instance();
                r.valid_arrays = 0;
                for (u32 k = 0; k < 2; ++k) {
                    if (arr[k] != 0 && memory != nullptr &&
                        memory->IsValidMapping(arr[k], u64(slots) * 8)) {
                        r.valid_arrays |= 1u << k;
                    }
                }
                r.mapped_frame = frame;
            }
            r.count = 0;
            for (u32 k = 0; k < 2; ++k) {
                if ((r.valid_arrays & (1u << k)) == 0) continue;
                for (s32 i = 0; i < slots; ++i) {
                    r.entries[r.count++] = reinterpret_cast<const u64*>(arr[k])[i];
                }
            }
            const int off = g_ring_offset.load();
            const int o = off & 0xfff;
            auto* memory = Core::Memory::Instance();
            for (u32 i = 0; i < r.count; ++i) {
                u64 addr = r.entries[i];
                if (off >= 0) {
                    if (addr == 0 || memory == nullptr || !memory->IsValidMapping(addr, u64(o) + 8)) {
                        continue;
                    }
                    addr = *reinterpret_cast<const u64*>(addr + o);
                    if (off & 0x1000) addr = VsharpBase(addr);
                }
                if (addr != 0) g_ring_addr[addr] = {ring, g_present};
            }
        }
    }
    st_hook_calls.fetch_add(1, std::memory_order_relaxed);
    st_hook_ns.fetch_add(u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 Clock::now() - t0)
                                 .count()),
                         std::memory_order_relaxed);
}

DrawId LookupPalette(u64 palette_address) {
    DrawId id{};
    if (!On() || palette_address == 0) return id;
    std::scoped_lock lk{g_mx};
    if (g_map_present != g_present) {
        // The slots' buffer objects may move their memory on every update: map them again once a
        // frame, reading where each entry keeps its GPU address now.
        g_map_present = g_present;
        for (auto e = g_ring_addr.begin(); e != g_ring_addr.end();) {
            e = e->second.second + 8 < g_present ? g_ring_addr.erase(e) : std::next(e);
        }
        auto* memory = Core::Memory::Instance();
        const int off = g_ring_offset.load();
        const int o = off & 0xfff;
        for (const auto& [ring, r] : g_rings) {
            for (u32 i = 0; i < r.count; ++i) {
                u64 addr = r.entries[i];
                if (off >= 0) {
                    if (addr == 0 || memory == nullptr || !memory->IsValidMapping(addr, u64(o) + 8)) {
                        continue;
                    }
                    addr = *reinterpret_cast<const u64*>(addr + o);
                    if (off & 0x1000) addr = VsharpBase(addr);
                }
                if (addr != 0) g_ring_addr[addr] = {ring, g_present};
            }
        }
    }
    auto it = g_ring_addr.upper_bound(palette_address);
    if (it != g_ring_addr.begin()) --it;
    if (it == g_ring_addr.end() || it->first > palette_address ||
        palette_address - it->first >= kRingSpan) {
        ++st_ring_miss;
        if (!g_rings.empty() && g_missed.size() < 256) g_missed.push_back(palette_address);
        return id;
    }
    const auto r = g_rings.find(it->second.first);
    if (r == g_rings.end()) return id;
    ++st_ring_hits;
    bool recent = false;
    const int state = Classify(r->second.bones, r->second.frame, recent);
    id.key = Mix(it->second.first, 0x72696e67ull);
    id.moving = state == 1;
    id.recent = recent || state == 1;
    id.skinned = r->second.skinned;
    return id;
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

void NoteSceneDraw(const DrawId& id) {
    ++st_scene;
    if (id.key != 0) {
        ++st_scene_id;
        if (id.moving) ++st_scene_moving;
        if (id.skinned) {
            ++st_skinned_scene_id;
            if (id.moving) ++st_skinned_scene_moving;
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
    {
        std::scoped_lock lk{g_mx};
        ++g_present;
    }
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
    std::string ring_text;
    {
        std::scoped_lock lk{g_mx};
        // Ring palettes not found by their entries: learn where the entry keeps the address.
        if (st_ring_hits == 0 && !g_missed.empty() && !g_rings.empty()) {
            std::sort(g_missed.begin(), g_missed.end());
            std::array<u32, 33> votes{};
            std::array<u32, 32> vvotes{};
            auto* memory = Core::Memory::Instance();
            u32 checked = 0;
            for (const auto& [ring, r] : g_rings) {
                for (u32 i = 0; i < r.count && checked < 512; ++i, ++checked) {
                    const u64 e = r.entries[i];
                    const auto near = [&](u64 v) {
                        const auto m = std::lower_bound(g_missed.begin(), g_missed.end(), v);
                        return v != 0 && m != g_missed.end() && *m - v < kRingSpan;
                    };
                    if (near(e)) ++votes[32];
                    if (e == 0 || memory == nullptr || !memory->IsValidMapping(e, 0x100)) continue;
                    for (u32 o = 0; o < 32; ++o) {
                        const u64 v = reinterpret_cast<const u64*>(e)[o];
                        if (near(v)) ++votes[o];
                        if (near(VsharpBase(v))) ++vvotes[o];
                    }
                }
            }
            u32 best = 0;
            bool vs = false;
            for (u32 o = 0; o < 32; ++o) {
                if (votes[o] > (vs ? vvotes[best] : votes[best])) best = o, vs = false;
                if (vvotes[o] > (vs ? vvotes[best] : votes[best])) best = o, vs = true;
            }
            const u32 best_votes = vs ? vvotes[best] : votes[best];
            ring_text = fmt::format("; ring learning: {} missed palettes, entry hits direct {}, "
                                    "best entry+0x{:x}{} ({} hits)",
                                    g_missed.size(), votes[32], best * 8, vs ? " (V#)" : "",
                                    best_votes);
            if (best_votes >= 2 && best_votes > votes[32]) {
                g_ring_offset.store(int(best * 8) + (vs ? 0x1000 : 0));
                g_map_present = 0;
                ring_text += fmt::format(" -> palette address at [entry+0x{:x}]", best * 8);
            }
        }
        g_missed.clear();
        ring_text = fmt::format("; skin rings {} (updates {:.1f}/frame), ring palette draws {:.1f}/frame, "
                                "palette draws without game identity {:.1f}/frame, entry offset {}{}",
                                g_rings.size(), double(st_ring_calls) / std::max<u64>(1, st_frames),
                                double(st_ring_hits) / std::max<u64>(1, st_frames),
                                double(st_ring_miss) / std::max<u64>(1, st_frames),
                                g_ring_offset.load(), ring_text);
        st_ring_calls = st_ring_hits = st_ring_miss = 0;
        for (auto it = g_rings.begin(); it != g_rings.end();) {
            if (it->second.frame + 120 < frame) {
                it = g_rings.erase(it);
            } else {
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
            "ms/frame{}",
            On() ? "on" : "off", tracked, double(st_moving_obj) / frames,
            double(st_classified) / frames, double(st_scene) / frames, cov_id,
            double(st_scene_id) / frames, double(st_scene_moving) / frames, cov_char,
            double(st_skinned_scene_id) / frames, double(st_vec_prev) / frames,
            double(st_vec - st_vec_prev) / frames, double(st_packets) / frames,
            double(st_matched) / frames, double(hook_calls) / frames, double(emits) / frames,
            double(hook_ns) / 1e6 / frames, double(st_gpu_ns) / 1e6 / frames, ring_text));
    }
    st_packets = st_matched = st_scene = st_scene_id = st_scene_moving = st_vec = st_vec_prev = 0;
    st_gpu_ns = st_classified = st_moving_obj = st_skinned_scene_id = st_skinned_scene_moving = 0;
    st_frames = 0;
}

} // namespace BB::NativeMv
