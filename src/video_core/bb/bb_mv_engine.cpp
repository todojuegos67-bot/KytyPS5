// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-mv-engine: see bb_mv_engine.h.

#include "video_core/bb/bb_mv_engine.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "core/memory.h"
#include "video_core/bb/bb_upscale.h"

namespace BB::MvEngine {
namespace {

using Clock = std::chrono::steady_clock;

constexpr u32 kHist = 4;              // records kept per palette address
constexpr u64 kMaxOffset = 0x4000;    // V# base up to this far into a palette
constexpr u64 kKeepFrames = 16;       // records older than this (GPU frames) are dropped

struct Rec {
    u64 hash = 0; // first matrix (48 bytes) when the game handed the palette to the sink
    u64 key = 0;
    u64 sink = 0;
    u64 stamp = 0;
};
struct Slot {
    std::array<Rec, kHist> r{};
    u32 next = 0;
};

struct Tls {
    u64 model = 0, sink = 0, mesh = 0, bones = 0;
    u32 consumer = 0;
    bool valid = false;
};
thread_local Tls t_cur;

std::mutex g_mx;
std::map<u64, Slot> g_pal; // palette start -> records
std::atomic<u64> g_stamp{1};
std::atomic<u64> g_guest_sets{0}, g_guest_begins{0}, g_guest_nobegin{0}, g_recorded{0};
std::atomic<bool> g_hooks{false};
std::string g_hook_detail;

// GPU thread.
u64 g_pass = 0;
std::unordered_map<u64, u64> g_cam_of;     // block key -> camera key (this frame)
std::unordered_map<u64, u32> g_pass_n;     // pass -> camera blocks seen this frame
std::unordered_set<u64> g_passes_frame, g_sinks_frame;
std::array<u64, size_t(Stat::Count)> g_st{};
std::array<u64, size_t(Miss::Count)> g_miss{};
u64 g_frames = 0, g_passes_sum = 0, g_sinks_sum = 0;
Clock::time_point g_t0 = Clock::now();

// Rigid ring records (DrawMesh, guest threads): bound current slot -> previous slot.
struct RigidRec {
    u64 prev = 0;
    u64 key = 0; // the ring object
    u64 stamp = 0;
    u64 size = 0; // the slot buffer's size ([obj+0x2c]: descriptor +0x14)
};
std::mutex g_rmx;
std::map<u64, RigidRec> g_rigid;
std::atomic<u64> g_dm_accept{0}, g_dm_reject{0};
std::atomic<u64> g_dm_override{0}, g_dm_same{0}, g_dm_single{0};
std::atomic<u64> g_dm_calls{0}, g_dm_rigid{0}, g_dm_skinned{0}, g_dm_palette{0};
std::atomic<u32> g_hook_dm{0}, g_hook_cam{0};
std::string g_eng_detail;
constexpr u64 kRigidMaxOffset = 0x100;
constexpr u64 kRigidMaxAge = 4; // GPU frames

// Engine camera records (SSAO reprojection site, render thread).
struct CamRec {
    std::array<float, 16> cur_view{}, cur_proj{}, prev_view{}, prev_proj{};
    bool prev_valid = false;
    u64 seq = 0;
};
std::mutex g_cmx;
std::array<CamRec, 8> g_cam{};
u64 g_cam_seq = 0;
std::atomic<u64> g_frame_seq{0}; // render frames (the SSAO camera site runs once a frame)
struct RingUpd {
    u64 last = 0, before = 0;
};
std::unordered_map<u64, RingUpd> g_ring_upd; // under g_rmx
std::atomic<u64> g_cam_calls{0};
u32 g_cam_layout_hits[2]{}; // row-major / transposed matches (GPU thread)
std::array<u32, 216> g_delta_n{};
std::array<float, 216> g_delta_v{};
u64 g_delta_blocks = 0;
std::array<u32, 216> g_sdelta_n{};
std::array<float, 216> g_sdelta_v{}, g_sdelta_p{};
u64 g_sdelta_blocks = 0;
bool g_dumped_nomatch = false;

// bb-mv-skin: ring slot records (guest DrawMesh, under g_rmx).
struct SlotRec {
    u64 prev = 0;   // previous slot base (Engine)
    u64 key = 0;    // ring object x kind (x bone)
    u64 stamp = 0;  // GPU frame stamp when recorded
    u64 size = 0;   // slot buffer size ([obj+0x2c])
    SlotMode mode = SlotMode::None;
    bool skinned = false;
};
std::map<u64, SlotRec> g_slots;
struct RingSeen {
    u64 period = 0, pperiod = 0;
    s32 idx = -1, pidx = -1;
    bool mixed = false;
};
std::unordered_map<u64, RingSeen> g_seen;
std::atomic<u64> g_flip{1};
enum class SlotWhy : u8 {
    Engine = 0, Unchanged, Own, NotLastFrame, Mixed, Advanced, SizeDiffers, Override,
    Count
};
std::array<std::atomic<u64>, size_t(SlotWhy::Count)> g_slot_why_skin{}, g_slot_why_rigid{};
std::atomic<u64> g_slot_rings1{0}, g_slot_rings2{0};
constexpr u64 kSlotMaxAge = 4; // GPU frames

u64 Mix(u64 a, u64 b) {
    u64 h = a ^ (b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2));
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    return h;
}

u64 Hash48(const void* p) {
    u64 v[6];
    std::memcpy(v, p, 48);
    u64 h = 0x6d76656e67ull;
    for (u64 x : v) h = Mix(h, x);
    return h;
}

void Log(const std::string& text) {
    LOG_INFO(Render_Vulkan, "{}", text);
    BB::Upscale::Checkpoint(text);
}

double Pct(u64 a, u64 b) {
    return b != 0 ? 100.0 * double(a) / double(b) : 0.0;
}

} // namespace

bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_MV_ENGINE");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

bool SlotsEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_MV_ENGINE_SLOTS");
        return Enabled() && !(v != nullptr && v[0] == '0');
    }();
    return on;
}

void OnGuestFlip() {
    g_flip.fetch_add(1, std::memory_order_relaxed);
}

void SetHooks(bool installed, const char* detail) {
    g_hooks = installed;
    {
        std::scoped_lock lk{g_mx};
        g_hook_detail = detail != nullptr ? detail : "";
    }
    Log(fmt::format("[mv-engine] hooks {}: {}", installed ? "in place" : "NOT installed",
                    detail != nullptr ? detail : ""));
}

void GuestPaletteBegin(u64 model, u64 sink, u64 mesh, u64 bones, u32 consumer) {
    t_cur = {model, sink, mesh, bones, consumer, bones != 0};
    g_guest_begins.fetch_add(1, std::memory_order_relaxed);
}

void GuestPaletteSet(u64 palette, u32 consumer) {
    g_guest_sets.fetch_add(1, std::memory_order_relaxed);
    if (!t_cur.valid || palette == 0 || t_cur.consumer != consumer) {
        g_guest_nobegin.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // The palette has just been written by the game (the loop before the call): first matrix.
    const u64 h = Hash48(reinterpret_cast<const void*>(palette));
    const u64 key = Mix(Mix(Mix(t_cur.bones, t_cur.mesh), consumer), 0x6f626a);
    std::scoped_lock lk{g_mx};
    Slot& s = g_pal[palette];
    s.r[s.next % kHist] = {h, key, t_cur.sink, g_stamp.load(std::memory_order_relaxed)};
    ++s.next;
    g_recorded.fetch_add(1, std::memory_order_relaxed);
}

void SetEngineHooks(u32 drawmesh, u32 camera, const char* detail) {
    g_hook_dm = drawmesh;
    g_hook_cam = camera;
    {
        std::scoped_lock lk{g_mx};
        g_eng_detail = detail != nullptr ? detail : "";
    }
    Log(fmt::format("[mv-engine] engine-transform hooks: DrawMesh rigid ring {}, engine camera "
                    "{}: {}",
                    drawmesh != 0 ? "in place" : "NOT installed",
                    camera != 0 ? "in place" : "NOT installed", detail != nullptr ? detail : ""));
}

namespace {
// bb-mv-skin: records the slot buffers DrawMesh binds as current (both branches) and what their
// previous is. Which slot was displayed last frame comes from DrawMesh itself: the ring index the
// object was drawn with in the previous guest frame (flip period). Called with g_rmx held.
void RecordSlots(u64 ring, bool skinned, u64 key_base, s32 idx, s32 pidx, s32 slots,
                 const u64* cur_objs, const u64* prev_objs, u32 n, bool overridden) {
    auto& why = skinned ? g_slot_why_skin : g_slot_why_rigid;
    const u64 P = g_flip.load(std::memory_order_relaxed);
    RingSeen& r = g_seen[ring];
    if (r.period != P) {
        r.pperiod = r.period;
        r.pidx = r.idx;
        r.period = P;
        r.idx = idx;
        r.mixed = false;
    } else if (r.idx != idx) {
        r.mixed = true; // the ring advanced inside one guest frame: no single "last frame"
    }
    (slots <= 1 ? g_slot_rings1 : g_slot_rings2).fetch_add(1, std::memory_order_relaxed);
    const u64 stamp = g_stamp.load(std::memory_order_relaxed);
    for (u32 k = 0; k < n; ++k) {
        const u64 cur_obj = cur_objs[k];
        if (cur_obj == 0)
            continue;
        const u64 cur = *reinterpret_cast<const u64*>(cur_obj + 0x20);
        const u64 size = *reinterpret_cast<const u32*>(cur_obj + 0x2c);
        if (cur == 0 || size == 0)
            continue;
        SlotRec rec{};
        rec.key = Mix(Mix(key_base, k), skinned ? 0x736b696eull : 0x72696764ull);
        rec.stamp = stamp;
        rec.size = size;
        rec.skinned = skinned;
        SlotWhy w;
        if (overridden) {
            // +0x30 binds [sink+0x1c8] instead of the slot: this slot is not what the VS reads.
            g_slots.erase(cur);
            why[size_t(SlotWhy::Override)].fetch_add(1, std::memory_order_relaxed);
            continue;
        } else if (slots <= 1) {
            rec.mode = SlotMode::Own; // one slot: rewritten in place, keep our own copy
            w = SlotWhy::Own;
        } else if (r.mixed) {
            w = SlotWhy::Mixed;
        } else if (r.pperiod == 0 || r.pperiod + 1 != P) {
            w = SlotWhy::NotLastFrame; // not drawn in the previous guest frame
        } else if (r.pidx == idx) {
            rec.mode = SlotMode::Unchanged; // not advanced: last frame showed this same slot
            w = SlotWhy::Unchanged;
        } else if (r.pidx != pidx) {
            w = SlotWhy::Advanced; // advanced twice or more: the previous slot is not last frame's
        } else {
            const u64 prev_obj = prev_objs[k];
            const u64 prev = prev_obj != 0 ? *reinterpret_cast<const u64*>(prev_obj + 0x20) : 0;
            const u64 psize = prev_obj != 0 ? *reinterpret_cast<const u32*>(prev_obj + 0x2c) : 0;
            if (prev == 0 || prev == cur || psize != size) {
                w = SlotWhy::SizeDiffers;
            } else {
                rec.mode = SlotMode::Engine;
                rec.prev = prev;
                w = SlotWhy::Engine;
            }
        }
        why[size_t(w)].fetch_add(1, std::memory_order_relaxed);
        g_slots[cur] = rec;
    }
}
} // namespace

// Mirrors DrawMesh 0x21d0640 up to the sink calls (only the reads the game itself makes on the
// branch it takes): the rigid ring branch's current / previous slot pointers.
void GuestDrawMesh(u64 model, u64 sink, u64 mesh, u64 material, u64 instance) {
    (void)model;
    g_dm_calls.fetch_add(1, std::memory_order_relaxed);
    if (instance == 0 || sink == 0 || mesh == 0 || material == 0)
        return;
    if (SlotsEnabled()) {
        const auto rd8s = [](u64 a) { return *reinterpret_cast<const u8*>(a); };
        const auto rd64s = [](u64 a) { return *reinterpret_cast<const u64*>(a); };
        const auto rd32s = [](u64 a) { return *reinterpret_cast<const s32*>(a); };
        const u64 ring = rd64s(instance + 8);
        if (ring == 0) {
            g_dm_palette.fetch_add(1, std::memory_order_relaxed); // BonePalette (0x21d083a)
            return;
        }
        bool rigid;
        if (rd8s(material + 0x2c) != 0) {
            rigid = false; // 0x21d0700 -> 0x21d0759
        } else if ((rd8s(sink + 0x171) & 1) == 0 || rd8s(material + 0x1d) == 0) {
            rigid = true; // 0x21d0809
        } else {
            rigid = rd8s(ring + 0xa4) == 0; // 0x21d0749
        }
        const u64 rmodel = rd64s(ring + 0xb0);
        const bool full = rd8s(rmodel + 0x200) != 0;
        const s32 idx = rd32s(ring + 0x98);
        const s32 slots = rd32s(ring + 0x9c);
        s32 pidx = idx - 1; // 0x21d07b1 / 0x21d08fb: idx - 1, + [ring+0x9c] when idx <= 0
        if (idx < 1)
            pidx += slots;
        if (!rigid) {
            g_dm_skinned.fetch_add(1, std::memory_order_relaxed);
            if (!full)
                return; // 0x21d077d: the sink gets 0, 0 (no ring slot bound)
            // 0x21d077f..0x21d0792 (current) / 0x21d07d5..0x21d07f7 (previous)
            const u64 a = rd64s(ring + 0x10), b = rd64s(ring + 0x18);
            const u64 cur[2] = {rd64s(a + u64(s64(idx)) * 8), rd64s(b + u64(s64(idx)) * 8)};
            const u64 prv[2] = {rd64s(a + u64(s64(pidx)) * 8), rd64s(b + u64(s64(pidx)) * 8)};
            std::scoped_lock lk{g_rmx};
            RecordSlots(ring, true, Mix(ring, 0x736b), idx, pidx, slots, cur, prv, 2, false);
            return;
        }
        const u64 bone = *reinterpret_cast<const u16*>(mesh + 2);
        const auto slot = [&](s32 i) -> u64 {
            if (full)
                return rd64s(rd64s(ring + 8) + u64(s64(i)) * 8);
            return rd64s(rd64s(rd64s(ring) + u64(s64(i)) * 0x28 + 8) + bone * 8);
        };
        const bool overridden = rd64s(sink + 0x1c8) != 0 && rd32s(sink + 0x108) == 1;
        if (overridden)
            g_dm_override.fetch_add(1, std::memory_order_relaxed);
        g_dm_rigid.fetch_add(1, std::memory_order_relaxed);
        if (slots <= 1)
            g_dm_single.fetch_add(1, std::memory_order_relaxed);
        const u64 cur[1] = {slot(idx)};
        const u64 prv[1] = {slot(pidx)};
        std::scoped_lock lk{g_rmx};
        RecordSlots(ring, false, full ? Mix(ring, 0x66756c6c) : Mix(ring, bone), idx, pidx, slots,
                    cur, prv, 1, overridden);
        return;
    }
    const auto rd8 = [](u64 a) { return *reinterpret_cast<const u8*>(a); };
    const auto rd64 = [](u64 a) { return *reinterpret_cast<const u64*>(a); };
    const auto rd32 = [](u64 a) { return *reinterpret_cast<const s32*>(a); };
    const u64 ring = rd64(instance + 8);
    if (ring == 0) {
        g_dm_palette.fetch_add(1, std::memory_order_relaxed); // BonePalette branch (0x21d083a)
        return;
    }
    bool rigid;
    if (rd8(material + 0x2c) != 0) {
        rigid = false; // 0x21d0700 -> skinned branch 0x21d0759
    } else if ((rd8(sink + 0x171) & 1) == 0 || rd8(material + 0x1d) == 0) {
        rigid = true; // 0x21d0809
    } else {
        rigid = rd8(ring + 0xa4) == 0; // 0x21d0749: 0 -> 0x21d0ff9 (rigid)
    }
    if (!rigid) {
        g_dm_skinned.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u64 rmodel = rd64(ring + 0xb0);
    const bool full = rd8(rmodel + 0x200) != 0;
    const s32 idx = rd32(ring + 0x98);
    s32 pidx = idx - 1;
    if (idx < 1)
        pidx += rd32(ring + 0x9c);
    const u64 bone = *reinterpret_cast<const u16*>(mesh + 2);
    const auto slot = [&](s32 i) -> u64 {
        if (full)
            return rd64(rd64(ring + 8) + u64(s64(i)) * 8);
        return rd64(rd64(rd64(ring) + u64(s64(i)) * 0x28 + 8) + bone * 8);
    };
    // The slots are GX buffer objects, not memory: sink vtable +0x30 = 0x224d880 / +0x38 =
    // 0x224d9d0 (vtable 0x53b7880, set by the sink constructor 0x224c940) store the object in
    // GX constant slot 5 / 6 ([[sink+8]]+0x6a8 / +0x6b0; flushed to state +0xc148+slot*8 at
    // 0x21a04e0). The shader's V# is the object's V# at +0x18 (bb_native_mv.cpp, 1.09
    // 0x2168a20). +0x30 binds [sink+0x1c8] instead when it is set and [sink+0x108] == 1.
    const u64 cur_obj = slot(idx);
    const u64 prev_obj = slot(pidx);
    const bool overridden = rd64(sink + 0x1c8) != 0 && rd32(sink + 0x108) == 1;
    if (overridden)
        g_dm_override.fetch_add(1, std::memory_order_relaxed);
    if (cur_obj == 0 || prev_obj == 0)
        return;
    // GX buffer object: descriptor at +0x18 whose +8 is the GPU address (+0x14 size): the slot
    // update 0x2568a20 -> 0x26b3560 writes at (offset + [desc+8]) = [obj+0x20]. (+0x18 itself is
    // not a V#: b9a295b0c collapsed every slot onto 2 values.)
    const auto base = [&](u64 obj) { return rd64(obj + 0x20); };
    if (rd32(ring + 0x9c) <= 1)
        g_dm_single.fetch_add(1, std::memory_order_relaxed);
    const u64 cur = base(cur_obj), prev = base(prev_obj);
    if (cur == 0 || prev == 0)
        return;
    g_dm_rigid.fetch_add(1, std::memory_order_relaxed);
    if (cur == prev)
        g_dm_same.fetch_add(1, std::memory_order_relaxed);
    // The previous slot holds last frame's matrices only when SkinRingUpdate 0x21c8c50 wrote the
    // ring in this render frame and in the one before (consecutive): otherwise the "previous"
    // slot is older (object not updated every frame: static props, 30 Hz logic) and the record
    // would give a static object a vector. Rejected (and any earlier record of the slot
    // dropped): 1-slot rings, cur == prev, the [sink+0x1c8] override (+0x30 binds another
    // buffer), rings not updated in two consecutive frames.
    const u64 F = g_frame_seq.load(std::memory_order_relaxed);
    bool consecutive = false;
    {
        std::scoped_lock lk{g_rmx};
        if (const auto it = g_ring_upd.find(ring); it != g_ring_upd.end()) {
            consecutive = it->second.last + 1 >= F && it->second.last == it->second.before + 1;
        }
        if (overridden || cur == prev || rd32(ring + 0x9c) < 2 || !consecutive) {
            g_rigid.erase(cur);
            g_dm_reject.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        // Size of both slots: a V# inside a neighbouring buffer (slots are allocated next to each
        // other) must never resolve to this record.
        const u64 size = *reinterpret_cast<const u32*>(cur_obj + 0x2c);
        if (size == 0 || size != *reinterpret_cast<const u32*>(prev_obj + 0x2c)) {
            g_rigid.erase(cur);
            g_dm_reject.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        g_rigid[cur] = {prev, Mix(ring, bone), g_stamp.load(std::memory_order_relaxed), size};
    }
    g_dm_accept.fetch_add(1, std::memory_order_relaxed);
}

void GuestRingUpdate(u64 ring) {
    if (ring == 0)
        return;
    const u64 F = g_frame_seq.load(std::memory_order_relaxed);
    std::scoped_lock lk{g_rmx};
    RingUpd& u = g_ring_upd[ring];
    if (u.last != F) {
        u.before = u.last;
        u.last = F;
    }
    if (g_ring_upd.size() > 65536)
        g_ring_upd.clear();
}

void GuestCamera(const u64* regs, u64 rbp, const u8* xmm) {
    const u64 obj = regs[0]; // rbx
    if (obj == 0 || rbp == 0 || xmm == nullptr)
        return;
    g_cam_calls.fetch_add(1, std::memory_order_relaxed);
    CamRec r;
    std::memcpy(r.cur_view.data(), xmm + 0x40, 64); // xmm4..7 -> [obj+0x1c0] (0xe670df)
    for (u32 i = 0; i < 4; ++i) {                   // [rbp-0x1f0-i*0x10] -> [obj+0x200+i*0x10]
        std::memcpy(r.cur_proj.data() + i * 4,
                    reinterpret_cast<const void*>(rbp - 0x1f0 - u64(i) * 0x10), 16);
    }
    std::memcpy(r.prev_view.data(), reinterpret_cast<const void*>(obj + 0x1c0), 64);
    std::memcpy(r.prev_proj.data(), reinterpret_cast<const void*>(obj + 0x200), 64);
    r.prev_valid = *reinterpret_cast<const u8*>(obj + 0x240) != 0;
    std::scoped_lock lk{g_cmx};
    r.seq = ++g_cam_seq;
    g_frame_seq.store(r.seq, std::memory_order_relaxed);
    g_cam[r.seq % g_cam.size()] = r;
}

bool ViewEqual(const float* a, const float* b) {
    for (u32 i = 0; i < 12; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
            return false;
        if (std::fabs(a[i] - b[i]) > 1e-4f * std::max(1.0f, std::fabs(a[i])))
            return false;
    }
    return true;
}

u32 EngineCameraPrev(const float* block, float prev_view[12]) {
    std::array<CamRec, 8> recs;
    {
        std::scoped_lock lk{g_cmx};
        recs = g_cam;
    }
    std::sort(recs.begin(), recs.end(),
              [](const CamRec& a, const CamRec& b) { return a.seq > b.seq; });
    const float* v = block + 8;
    for (const CamRec& r : recs) {
        if (r.seq == 0)
            break;
        for (u32 layout = 0; layout < 2; ++layout) {
            float c[12], p[12];
            for (u32 row = 0; row < 3; ++row) {
                for (u32 col = 0; col < 4; ++col) {
                    const u32 src = layout == 0 ? row * 4 + col : col * 4 + row;
                    c[row * 4 + col] = r.cur_view[src];
                    p[row * 4 + col] = r.prev_view[src];
                }
            }
            if (!ViewEqual(v, c))
                continue;
            ++g_cam_layout_hits[layout];
            if (!r.prev_valid)
                return 2;
            std::memcpy(prev_view, p, sizeof(p));
            return 1;
        }
    }
    if (!g_dumped_nomatch && recs[0].seq != 0) {
        g_dumped_nomatch = true;
        std::string a, b;
        for (u32 i = 0; i < 12; ++i)
            a += fmt::format(" {:.5g}", v[i]);
        for (u32 i = 0; i < 16; ++i)
            b += fmt::format(" {:.5g}", recs[0].cur_view[i]);
        Log(fmt::format("[mv-engine-cam] first unmatched block view (floats 8..19){} | engine "
                        "view 4x4{}",
                        a, b));
    }
    return 0;
}

void BlockDelta(const float* cur, const float* prev) {
    ++g_delta_blocks;
    // Camera standing still (view bit-identical): what still changes is time / wind / jitter.
    if (std::memcmp(cur + 8, prev + 8, 48) == 0) {
        ++g_sdelta_blocks;
        for (u32 i = 0; i < 216; ++i) {
            if ((i >= 8 && i < 20) || (i >= 52 && i < 68))
                continue;
            if (std::memcmp(cur + i, prev + i, 4) != 0) {
                ++g_sdelta_n[i];
                g_sdelta_v[i] = cur[i];
                g_sdelta_p[i] = prev[i];
            }
        }
    }
    for (u32 i = 0; i < 216; ++i) {
        if ((i >= 8 && i < 20) || (i >= 52 && i < 68))
            continue;
        if (std::memcmp(cur + i, prev + i, 4) != 0) {
            ++g_delta_n[i];
            g_delta_v[i] = cur[i];
        }
    }
}

bool LookupRigid(u64 addr, u64& prev_addr, u64& offset, u64& key) {
    std::scoped_lock lk{g_rmx};
    auto it = g_rigid.upper_bound(addr);
    if (it == g_rigid.begin())
        return false;
    --it;
    if (addr - it->first >= std::min(kRigidMaxOffset, it->second.size))
        return false;
    // Only records of the last few frames: an older slot address may have been freed and reused
    // by another object, whose "previous" would then be unrelated memory.
    if (it->second.stamp + kRigidMaxAge < g_stamp.load(std::memory_order_relaxed))
        return false;
    offset = addr - it->first;
    prev_addr = it->second.prev + offset;
    key = it->second.key;
    return true;
}

bool LookupSlot(u64 addr, u64 size, SlotHit& out) {
    std::scoped_lock lk{g_rmx};
    auto it = g_slots.upper_bound(addr);
    if (it == g_slots.begin())
        return false;
    --it;
    const SlotRec& r = it->second;
    const u64 off = addr - it->first;
    if (off >= r.size)
        return false;
    // Old records: the slot's memory may belong to another object by now.
    if (r.stamp + kSlotMaxAge < g_stamp.load(std::memory_order_relaxed))
        return false;
    out = {};
    out.skinned = r.skinned;
    out.key = r.key;
    out.offset = off;
    // A V# running past the slot's end would read past the previous slot too.
    out.mode = off + size <= r.size ? r.mode : SlotMode::None;
    if (out.mode == SlotMode::Engine)
        out.prev_addr = r.prev + off;
    return true;
}

bool LookupPalette(Core::MemoryManager* memory, bool gpu_modified, u64 addr, ObjId& out,
                   Miss& why) {
    why = Miss::None;
    if (gpu_modified) {
        why = Miss::GpuWritten; // not a CPU frame-allocator palette (skin ring: own identity)
        return false;
    }
    u64 start = 0;
    Slot slot;
    {
        std::scoped_lock lk{g_mx};
        auto it = g_pal.upper_bound(addr);
        if (it == g_pal.begin()) {
            why = Miss::NoRecord;
            return false;
        }
        --it;
        if (addr - it->first >= kMaxOffset) {
            why = Miss::NoRecord;
            return false;
        }
        start = it->first;
        slot = it->second;
    }
    if (memory == nullptr || !memory->IsValidMapping(start, 48)) {
        why = Miss::Unmapped;
        return false;
    }
    const u64 h = Hash48(reinterpret_cast<const void*>(start));
    // The newest record whose first matrix is still in guest memory (exact).
    const Rec* best = nullptr;
    for (const Rec& r : slot.r) {
        if (r.key != 0 && r.hash == h && (best == nullptr || r.stamp > best->stamp)) best = &r;
    }
    if (best == nullptr) {
        why = Miss::Stale;
        return false;
    }
    out = {best->key, best->sink, addr - start};
    g_sinks_frame.insert(best->sink);
    return true;
}

void SetPass(u64 pass_key) {
    g_pass = pass_key;
}

u64 CameraKey(u64 block_key, u32& ordinal) {
    const u64 k = Mix(block_key, g_pass);
    if (const auto it = g_cam_of.find(k); it != g_cam_of.end()) {
        ordinal = u32(it->second & 0xffff);
        return it->second;
    }
    g_passes_frame.insert(g_pass);
    ordinal = g_pass_n[g_pass]++;
    const u64 ck = (Mix(Mix(g_pass, 0x63616d70617373ull), ordinal) & ~0xffffull) | ordinal;
    g_cam_of[k] = ck;
    return ck;
}

void Count(Stat s, u64 n) {
    g_st[size_t(s)] += n;
}

void CountMiss(Miss m) {
    ++g_miss[size_t(m)];
}

void OnFrame() {
    if (!Enabled()) return;
    ++g_frames;
    g_passes_sum += g_passes_frame.size();
    g_sinks_sum += g_sinks_frame.size();
    g_cam_of.clear();
    g_pass_n.clear();
    g_passes_frame.clear();
    g_sinks_frame.clear();
    const u64 stamp = g_stamp.fetch_add(1) + 1;
    const auto now = Clock::now();
    if (now - g_t0 < std::chrono::seconds(1)) return;
    const double secs = std::chrono::duration<double>(now - g_t0).count();
    g_t0 = now;
    size_t addrs = 0;
    {
        std::scoped_lock lk{g_mx};
        for (auto it = g_pal.begin(); it != g_pal.end();) {
            u64 newest = 0;
            for (const Rec& r : it->second.r) newest = std::max(newest, r.stamp);
            if (newest + kKeepFrames < stamp) {
                it = g_pal.erase(it);
            } else {
                ++it;
            }
        }
        addrs = g_pal.size();
    }
    size_t rigid_addrs = 0;
    {
        std::scoped_lock lk{g_rmx};
        for (auto it = g_rigid.begin(); it != g_rigid.end();) {
            if (it->second.stamp + kKeepFrames < stamp) {
                it = g_rigid.erase(it);
            } else {
                ++it;
            }
        }
        rigid_addrs = g_rigid.size();
        for (auto it = g_slots.begin(); it != g_slots.end();) {
            if (it->second.stamp + kKeepFrames < stamp) {
                it = g_slots.erase(it);
            } else {
                ++it;
            }
        }
        if (g_seen.size() > 65536)
            g_seen.clear();
    }
    const auto S = [](Stat s) { return g_st[size_t(s)]; };
    const auto M = [](Miss m) { return g_miss[size_t(m)]; };
    const double f = double(std::max<u64>(1, g_frames));
    Log(fmt::format(
        "[mv-engine] {:.1f} s {} frames | hooks {} | caves entered: BonePalette entry {:.1f}/frame, "
        "SetPalette site {:.1f}/frame; recorded {:.1f}/frame (site without entry record {}), {} "
        "addresses (0 entries: the game never took BonePalette, every instance drawn has a GPU "
        "skin ring: DrawMesh eboot+0x21d0740 test [instance+8]) | draws {:.0f}/frame, engine identity {:.1f}% | palettes "
        "{:.0f}/frame: identity {:.1f}%, previous transform found {:.1f}% of identified, new "
        "object {:.1f}/frame | fallbacks: no record {:.1f}/frame, GPU-written {:.1f}/frame, "
        "stale {:.1f}/frame, unmapped {:.1f}/frame, old path used {:.1f}/frame",
        secs, g_frames, g_hooks.load() ? "on" : "off", double(g_guest_begins.exchange(0)) / f,
        double(g_guest_sets.exchange(0)) / f, double(g_recorded.exchange(0)) / f,
        g_guest_nobegin.exchange(0), addrs, double(S(Stat::Draws)) / f,
        Pct(S(Stat::EngineDraws), S(Stat::Draws)), double(S(Stat::PalLookups)) / f,
        Pct(S(Stat::PalFound), S(Stat::PalLookups)), Pct(S(Stat::PalPrev), S(Stat::PalFound)),
        double(S(Stat::PalNew)) / f, double(M(Miss::NoRecord)) / f,
        double(M(Miss::GpuWritten)) / f, double(M(Miss::Stale)) / f,
        double(M(Miss::Unmapped)) / f, double(S(Stat::FallbackOld)) / f));
    // Short line of its own (the long line above is cut by some log viewers).
    Log(fmt::format("[mv-engine-cam] camera blocks {:.1f}/frame: pass key {:.1f}%, previous "
                    "found {:.1f}%, fallback {:.1f}% | passes {:.1f}/frame, engine sinks "
                    "{:.1f}/frame",
                    double(S(Stat::CamLookups)) / f, Pct(S(Stat::CamPass), S(Stat::CamLookups)),
                    Pct(S(Stat::CamPassPrev), S(Stat::CamPass)),
                    Pct(S(Stat::CamFallback), S(Stat::CamLookups)), double(g_passes_sum) / f,
                    double(g_sinks_sum) / f));
    // bb-mv-engine (engine transforms): coverage of the rigid ring records and the engine camera.
    Log(fmt::format("[mv-engine-rigid] DrawMesh {:.1f}/frame: rigid ring {:.1f}, skinned ring "
                    "{:.1f}, BonePalette {:.1f}; {} slot records | lookups {:.1f}/frame: engine "
                    "previous world {:.1f}%, no copy {:.1f}/frame (hook {}) | sink+0x1c8 "
                    "override {:.1f}/frame, previous slot == current {:.1f}/frame, ring of 1 slot "
                    "{:.1f}/frame | records accepted {:.1f}/frame, rejected {:.1f}/frame",
                    double(g_dm_calls.exchange(0)) / f, double(g_dm_rigid.exchange(0)) / f,
                    double(g_dm_skinned.exchange(0)) / f, double(g_dm_palette.exchange(0)) / f,
                    rigid_addrs, double(S(Stat::RigidLookups)) / f,
                    Pct(S(Stat::RigidFound), S(Stat::RigidLookups)),
                    double(S(Stat::RigidNoSnap)) / f, g_hook_dm.load() ? "on" : "off",
                    double(g_dm_override.exchange(0)) / f, double(g_dm_same.exchange(0)) / f,
                    double(g_dm_single.exchange(0)) / f, double(g_dm_accept.exchange(0)) / f,
                    double(g_dm_reject.exchange(0)) / f));
    if (SlotsEnabled()) {
        // bb-mv-skin: what the velocity pass got for the engine's ring slots.
        const u64 sl = S(Stat::SlotSkin), rl = S(Stat::SlotRigid);
        const u64 found = S(Stat::SlotEngine) + S(Stat::SlotUnchanged) + S(Stat::SlotOwn);
        const u64 camonly = S(Stat::SlotOwnMissing) + S(Stat::SlotNone) + S(Stat::SlotLiveFail) +
                            S(Stat::PalNoRecordCamOnly);
        Log(fmt::format(
            "[mv-engine-slots] skinned previous found {:.1f}% of {:.1f} buffers/frame (draws "
            "{:.1f}/frame, all buffers with previous {:.1f}%) | rigid previous found {:.1f}% of "
            "{:.1f} buffers/frame | camera-only fallback {:.1f}/frame (no previous {:.1f}, 1-slot "
            "ring without last frame's copy {:.1f}, previous slot not cached {:.1f}, palette "
            "without engine record {:.1f}) | previous from: engine slot {:.1f}%, unchanged ring "
            "{:.1f}%, own copy {:.1f}%",
            Pct(S(Stat::SlotSkinPrev), sl), double(sl) / f, double(S(Stat::SkinDraws)) / f,
            Pct(S(Stat::SkinDrawsPrev), S(Stat::SkinDraws)), Pct(S(Stat::SlotRigidPrev), rl),
            double(rl) / f, double(camonly) / f, double(S(Stat::SlotNone)) / f,
            double(S(Stat::SlotOwnMissing)) / f, double(S(Stat::SlotLiveFail)) / f,
            double(S(Stat::PalNoRecordCamOnly)) / f, Pct(S(Stat::SlotEngine), found),
            Pct(S(Stat::SlotUnchanged), found), Pct(S(Stat::SlotOwn), found)));
        const auto W = [&](std::array<std::atomic<u64>, size_t(SlotWhy::Count)>& a) {
            std::array<u64, size_t(SlotWhy::Count)> v{};
            for (size_t i = 0; i < v.size(); ++i)
                v[i] = a[i].exchange(0);
            return fmt::format("engine slot {:.1f}, unchanged {:.1f}, 1-slot own copy {:.1f}, "
                               "not drawn last frame {:.1f}, advanced inside a frame {:.1f}, "
                               "advanced 2+ {:.1f}, previous slot other size {:.1f}, "
                               "sink+0x1c8 override {:.1f}",
                               double(v[0]) / f, double(v[1]) / f, double(v[2]) / f,
                               double(v[3]) / f, double(v[4]) / f, double(v[5]) / f,
                               double(v[6]) / f, double(v[7]) / f);
        };
        const std::string ws = W(g_slot_why_skin), wr = W(g_slot_why_rigid);
        Log(fmt::format("[mv-engine-slots] guest records/frame: skinned {} | rigid {} | rings "
                        "drawn with 1 slot {:.1f}, 2+ slots {:.1f}, flip {}",
                        ws, wr, double(g_slot_rings1.exchange(0)) / f,
                        double(g_slot_rings2.exchange(0)) / f, g_flip.load()));
    }
    const u64 cl = S(Stat::CamEngLookups);
    Log(fmt::format("[mv-engine-cam2] engine camera {:.1f}/frame (hook {}) | blocks {:.1f}/frame: "
                    "engine previous used {:.1f}%, fallback: no CPU copy {:.1f}%, not the engine "
                    "view {:.1f}%, no engine previous {:.1f}%, previous view not among last "
                    "frame's blocks {:.1f}% | view layout row {} / transposed {}",
                    double(g_cam_calls.exchange(0)) / f, g_hook_cam.load() ? "on" : "off",
                    double(cl) / f, Pct(S(Stat::CamEngUsed), cl), Pct(S(Stat::CamEngNoCpu), cl),
                    Pct(S(Stat::CamEngNoMatch), cl), Pct(S(Stat::CamEngNoPrev), cl),
                    Pct(S(Stat::CamEngNoBlock), cl), g_cam_layout_hits[0], g_cam_layout_hits[1]));
    g_cam_layout_hits[0] = g_cam_layout_hits[1] = 0;
    if (g_delta_blocks != 0) {
        // Floats (besides view / projection) that change between the engine-paired blocks: time /
        // wind / jitter constants the vertex animation can read (reproduced by last frame's block).
        std::vector<u32> idx;
        for (u32 i = 0; i < 216; ++i) {
            if (g_delta_n[i] != 0)
                idx.push_back(i);
        }
        std::sort(idx.begin(), idx.end(), [](u32 a, u32 b) { return g_delta_n[a] > g_delta_n[b]; });
        std::string t;
        for (size_t k = 0; k < std::min<size_t>(idx.size(), 12); ++k) {
            t += fmt::format(" d[{}]={:.5g}({:.0f}%)", idx[k], g_delta_v[idx[k]],
                             Pct(g_delta_n[idx[k]], g_delta_blocks));
        }
        Log(fmt::format("[mv-engine-wind] {} paired blocks, {} other floats change:{}",
                        g_delta_blocks, idx.size(), t));
        g_delta_n.fill(0);
        g_delta_blocks = 0;
    }
    {
        std::string t;
        u32 nchg = 0;
        for (u32 i = 0; i < 216; ++i) {
            if (g_sdelta_n[i] == 0)
                continue;
            ++nchg;
            if (nchg <= 24)
                t += fmt::format(" d[{}]={:.6g}<-{:.6g}({:.0f}%)", i, g_sdelta_v[i],
                                 g_sdelta_p[i], Pct(g_sdelta_n[i], g_sdelta_blocks));
        }
        Log(fmt::format("[mv-engine-wind-static] {} paired blocks with the camera still, {} "
                        "floats change:{}",
                        g_sdelta_blocks, nchg, t));
        g_sdelta_n.fill(0);
        g_sdelta_blocks = 0;
    }
    g_st.fill(0);
    g_miss.fill(0);
    g_frames = g_passes_sum = g_sinks_sum = 0;
}

} // namespace BB::MvEngine
