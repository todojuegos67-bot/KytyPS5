// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Object collision diagnostics (see bb_objcol.h). Havok 2014.1 hknp, 1.09 eboot:
//   0xb55b20 hknpWorld::addBodies(world, const hknpBodyId* ids, int n, int mode, u32 flags):
//            n <= 0 or mode == 0: returns without doing anything; mode 2: queued on the pending
//            list (world+0xb8 / +0xc8), committed later by 0xb56360 with mode 1; otherwise the
//            bodies go into the broad phase (static: quantized AABB into body+0x50).
//   0xb56940 hknpWorld::removeBodies(world, ids, n, mode)
//   world+0x20 hkArray<hknpBody> (data, size +0x28, capacity +0x2c), 0x90 bytes a body:
//            +0x00 transform (translation +0x30), +0x40 flags (1 static, 2 dynamic, 4 keyframed),
//            +0x44 collision filter info, +0x48 shape, +0x72 broad-phase layer.
// FD4 (0xa7b870, the game's addBody wrapper) only adds a body when flags & 3 is non-zero.

#include "core/bb_objcol.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>
#include <zlib.h>
#include "common/logging/log.h"
#include "common/singleton.h"
#include "core/file_sys/fs.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace BB::ObjCol {
namespace {

constexpr int W = -1;

bool ReadEnv() {
    const char* v = std::getenv("BB_OBJCOL_DIAG");
    return v != nullptr && v[0] == '1';
}

const bool g_enabled = ReadEnv();

// Copies guest memory without faulting on unmapped / protected pages.
bool SafeCopy(void* dst, u64 src, u64 len) {
    if (len == 0) {
        return true;
    }
    if (src == 0 || src + len < src) {
        return false;
    }
#ifdef _WIN32
    SIZE_T got = 0;
    return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(src), dst, len, &got) !=
               0 &&
           got == len;
#else
    iovec local{dst, len};
    iovec remote{reinterpret_cast<void*>(src), len};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == static_cast<ssize_t>(len);
#endif
}

template <typename T>
bool Rd(u64 addr, T& out) {
    return SafeCopy(&out, addr, sizeof(T));
}

u32 Le32(const u8* p) {
    u32 v;
    std::memcpy(&v, p, 4);
    return v;
}
u64 Le64(const u8* p) {
    u64 v;
    std::memcpy(&v, p, 8);
    return v;
}
u32 Be32(const u8* p) {
    return (u32(p[0]) << 24) | (u32(p[1]) << 16) | (u32(p[2]) << 8) | u32(p[3]);
}

double Now() {
    using namespace std::chrono;
    static const auto t0 = steady_clock::now();
    return duration<double>(steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------- binders (DCX / BND4)

struct BndEntry {
    std::string name;
    u64 offset = 0;
    u64 size = 0;
};

std::string BaseName(const std::string& s) {
    const auto p = s.find_last_of("\\/");
    return p == std::string::npos ? s : s.substr(p + 1);
}

std::string ReadName(const u8* p, size_t n, u64 off, bool unicode) {
    std::string out;
    if (off >= n) {
        return out;
    }
    if (unicode) {
        for (u64 i = off; i + 1 < n && out.size() < 260; i += 2) {
            const u16 c = u16(p[i]) | (u16(p[i + 1]) << 8);
            if (c == 0) {
                break;
            }
            out.push_back(c < 0x80 ? char(c) : '?');
        }
    } else {
        for (u64 i = off; i < n && p[i] != 0 && out.size() < 260; ++i) {
            out.push_back(char(p[i]));
        }
    }
    return out;
}

// BND4 (SoulsFormats layout): count +0x0c, entry header size +0x20, unicode +0x30, format
// +0x31 (bit order reversed unless +0x0a is 0 or the raw byte says so), entries from +0x40:
// flags, -1, size (i64), [uncompressed size i64], data offset (u32 / i64), [id], name offset.
bool ParseBnd4(const u8* p, size_t n, std::vector<BndEntry>& out) {
    if (n < 0x40 || std::memcmp(p, "BND4", 4) != 0 || p[0x09] != 0) {
        return false;
    }
    const s32 count = s32(Le32(p + 0x0c));
    const u64 hdr = Le64(p + 0x20);
    if (count <= 0 || count > 4096 || hdr < 0x18 || hdr > 0x40) {
        return false;
    }
    const u64 table_end = 0x40 + u64(count) * hdr;
    if (table_end > n) {
        return false;
    }
    const bool unicode = p[0x30] == 1;
    const u8 raw = p[0x31];
    const bool bit_big_endian = p[0x0a] == 0;
    u8 format = raw;
    if (!(bit_big_endian || ((raw & 1) != 0 && (raw & 0x80) == 0))) {
        format = 0;
        for (int b = 0; b < 8; ++b) {
            format |= u8(((raw >> b) & 1) << (7 - b));
        }
    }
    struct Layout {
        bool comp, longoff, ids;
    };
    std::vector<Layout> layouts;
    layouts.push_back({(format & 0x20) != 0, (format & 0x10) != 0, (format & 0x02) != 0});
    for (int c = 0; c < 8; ++c) {
        layouts.push_back({(c & 1) != 0, (c & 2) != 0, (c & 4) != 0});
    }
    for (const auto& l : layouts) {
        const u64 size = 16 + (l.comp ? 8 : 0) + (l.longoff ? 8 : 4) + (l.ids ? 4 : 0) + 4;
        if (size != hdr) {
            continue;
        }
        bool ok = true;
        out.clear();
        for (s32 i = 0; i < count && ok; ++i) {
            const u8* e = p + 0x40 + u64(i) * hdr;
            BndEntry be;
            be.size = Le64(e + 8);
            const u8* o = e + 16 + (l.comp ? 8 : 0);
            be.offset = l.longoff ? Le64(o) : Le32(o);
            const u64 name_off = Le32(e + hdr - 4);
            ok = be.size < (1ull << 30) && be.offset >= table_end && be.offset < (1ull << 31) &&
                 name_off >= table_end && name_off < n;
            if (ok) {
                be.name = BaseName(ReadName(p, n, name_off, unicode));
                ok = !be.name.empty();
            }
            out.push_back(std::move(be));
        }
        if (ok) {
            return true;
        }
    }
    out.clear();
    return false;
}

bool Inflate(const u8* src, size_t n, u64 out_size, std::vector<u8>& out) {
    if (out_size == 0 || out_size > (512ull << 20)) {
        return false;
    }
    out.assign(out_size, 0);
    z_stream zs{};
    if (inflateInit(&zs) != Z_OK) {
        return false;
    }
    zs.next_in = const_cast<Bytef*>(src);
    zs.avail_in = uInt(n);
    zs.next_out = out.data();
    zs.avail_out = uInt(out_size);
    const int r = inflate(&zs, Z_FINISH);
    const u64 got = zs.total_out;
    inflateEnd(&zs);
    if (r != Z_STREAM_END && got != out_size) {
        return false;
    }
    out.resize(got);
    return true;
}

// DCX: "DCS\0" +0x18 (uncompressed / compressed size, big endian), "DCP\0" +0x24 with the
// format, "DCA\0" then the zlib stream.
bool Undcx(const std::vector<u8>& f, std::vector<u8>& out, std::string& why) {
    if (f.size() < 0x50 || std::memcmp(f.data(), "DCX\0", 4) != 0) {
        out = f;
        return true;
    }
    if (std::memcmp(f.data() + 0x18, "DCS\0", 4) != 0 ||
        std::memcmp(f.data() + 0x24, "DCP\0", 4) != 0) {
        why = "DCX header not recognised";
        return false;
    }
    const std::string format(reinterpret_cast<const char*>(f.data() + 0x28), 4);
    if (format != "DFLT") {
        why = "DCX format " + format;
        return false;
    }
    const u64 usize = Be32(f.data() + 0x1c);
    const u64 csize = Be32(f.data() + 0x20);
    for (size_t i = 0x2c; i + 8 < std::min<size_t>(f.size(), 0x100); i += 4) {
        if (std::memcmp(f.data() + i, "DCA\0", 4) == 0) {
            const size_t data = i + Be32(f.data() + i + 4);
            if (data >= f.size()) {
                break;
            }
            const size_t avail = std::min<size_t>(csize ? csize : f.size(), f.size() - data);
            if (!Inflate(f.data() + data, avail, usize, out)) {
                why = "inflate failed";
                return false;
            }
            return true;
        }
    }
    why = "no DCA block";
    return false;
}

bool Contains(const u8* p, size_t n, const char* s) {
    const size_t k = std::strlen(s);
    if (k == 0 || n < k) {
        return false;
    }
    const u8* end = p + n - k + 1;
    for (const u8* q = p; q < end;) {
        q = static_cast<const u8*>(std::memchr(q, s[0], size_t(end - q)));
        if (q == nullptr) {
            return false;
        }
        if (std::memcmp(q, s, k) == 0) {
            return true;
        }
        ++q;
    }
    return false;
}

// "oNNNNNN" in a binder entry name.
std::string ModelId(const std::string& name) {
    for (size_t i = 0; i + 7 <= name.size(); ++i) {
        if (name[i] != 'o' && name[i] != 'O') {
            continue;
        }
        bool digits = true;
        for (size_t k = 1; k < 7; ++k) {
            digits = digits && name[i + k] >= '0' && name[i + k] <= '9';
        }
        if (digits && (i + 7 == name.size() || name[i + 7] < '0' || name[i + 7] > '9')) {
            return "o" + name.substr(i + 1, 6);
        }
    }
    return {};
}

// ---------------------------------------------------------------- state

enum class Kind : u8 { Unknown, NoPhysics, Solid, Destructible, Failed };

const char* KindName(Kind k) {
    switch (k) {
    case Kind::NoPhysics:
        return "no collision data";
    case Kind::Solid:
        return "static-solid";
    case Kind::Destructible:
        return "destructible/physics";
    case Kind::Failed:
        return "not parsed";
    default:
        return "pending";
    }
}

struct Model {
    std::string id;
    std::string map;
    Kind kind = Kind::Unknown;
    std::string files; // hkx entries with their class
    // runtime (bodies attributed to this binder)
    u32 added = 0, in_world = 0, removed = 0, not_added = 0, noop = 0;
    u32 st_static = 0, st_dynamic = 0, st_keyframed = 0;
    bool removal_logged = false;
};

enum class State : u8 { None, InWorld, Deferred, Ignored, Removed };

struct Body {
    u64 shape = 0;
    int owner = -1;
    State state = State::None;
    u32 flags = 0;
    bool logged = false;
};

struct World {
    std::unordered_map<u32, Body> bodies;
    std::unordered_map<u32, double> unadded_since; // census: body slots never added
    double last_census = 0.0;
    double last_call = 0.0;
};

struct Range {
    u64 lo, hi;
    int model;
};

std::mutex g_mutex;
std::vector<Model> g_models;
std::unordered_map<std::string, int> g_model_index;
std::string g_map = "(boot)";
std::vector<int> g_map_models;
std::unordered_map<u64, World> g_worlds;
std::vector<Range> g_ranges;
std::set<u64> g_no_owner_blocks;
u64 g_eboot = 0;
u32 g_body_lines = 0;
std::atomic<bool> g_hooked{false};
u64 st_calls[3] = {}, st_bodies = 0, st_owned = 0, st_bad_id = 0, st_removed = 0, st_scans = 0;
double g_last_summary = 0.0;
std::string g_last_summary_text;

int ModelFor(const std::string& id) {
    if (auto it = g_model_index.find(id); it != g_model_index.end()) {
        return it->second;
    }
    Model m;
    m.id = id;
    g_models.push_back(m);
    const int i = int(g_models.size() - 1);
    g_model_index[id] = i;
    return i;
}

// ---------------------------------------------------------------- file side worker

struct Job {
    std::string guest;
    int model;
};
std::mutex g_job_mutex;
std::condition_variable g_job_cv;
std::deque<Job> g_jobs;
bool g_worker_started = false;

void Analyse(const Job& job) {
    std::vector<u8> raw, bnd;
    std::string why;
    auto* mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();
    auto file = mnt->Open(job.guest, false);
    if (!file || !file->IsOpen()) {
        why = "host open failed";
    } else {
        raw.resize(file->Size());
        const s64 got = raw.empty() ? 0 : file->Read(raw.data(), raw.size());
        if (got != s64(raw.size())) {
            why = fmt::format("short read {} of {} bytes", got, raw.size());
        }
    }
    std::vector<BndEntry> entries;
    if (why.empty() && Undcx(raw, bnd, why) && !ParseBnd4(bnd.data(), bnd.size(), entries)) {
        why = "BND4 not recognised";
    }
    Kind kind = Kind::Failed;
    std::string files;
    if (why.empty()) {
        bool physics = false, breakable = false;
        for (const auto& e : entries) {
            std::string lower = e.name;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](char c) { return char(std::tolower(u8(c))); });
            if (!lower.ends_with(".hkx")) {
                continue;
            }
            if (e.offset + e.size > bnd.size()) {
                files += fmt::format(" {}(out of range)", e.name);
                continue;
            }
            std::vector<u8> inner;
            const u8* p = bnd.data() + e.offset;
            size_t n = e.size;
            if (n >= 4 && std::memcmp(p, "DCX\0", 4) == 0) {
                std::string iw;
                if (Undcx(std::vector<u8>(p, p + n), inner, iw)) {
                    p = inner.data();
                    n = inner.size();
                }
            }
            const bool brk = Contains(p, n, "hkdBreakable") || Contains(p, n, "hkdShape") ||
                             Contains(p, n, "BreakableBody") || Contains(p, n, "hkdDestruction") ||
                             Contains(p, n, "hkdFracture");
            const bool phys = brk || Contains(p, n, "hknpPhysicsSystemData") ||
                              Contains(p, n, "hknpShape") || Contains(p, n, "hknpBody") ||
                              Contains(p, n, "hkpRigidBody") || Contains(p, n, "hkpPhysicsData") ||
                              Contains(p, n, "hkpShape") || Contains(p, n, "hknpCompressedMesh") ||
                              Contains(p, n, "hknpConvex");
            const char* cls = brk                          ? "breakable"
                              : phys                       ? "physics"
                              : Contains(p, n, "hclCloth") ? "cloth"
                              : (Contains(p, n, "hkaSkeleton") || Contains(p, n, "hkaAnim"))
                                  ? "anim"
                                  : "other";
            files += fmt::format(" {}({} {} KB)", e.name, cls, (n + 1023) / 1024);
            physics = physics || phys;
            breakable = breakable || brk;
        }
        kind = !physics ? Kind::NoPhysics : breakable ? Kind::Destructible : Kind::Solid;
        if (files.empty()) {
            files = " (no .hkx entry)";
        }
    }
    std::string map, id;
    {
        std::scoped_lock lk{g_mutex};
        auto& m = g_models[job.model];
        m.kind = kind;
        m.files = why.empty() ? files : " " + why;
        map = m.map;
        id = m.id;
    }
    LOG_INFO(Core, "[objcol] objbnd {} ({}): {} -{}", id, map, KindName(kind),
             why.empty() ? files : " " + why);
}

void Worker() {
    for (;;) {
        Job job;
        {
            std::unique_lock lk{g_job_mutex};
            g_job_cv.wait(lk, [] { return !g_jobs.empty(); });
            job = std::move(g_jobs.front());
            g_jobs.pop_front();
        }
        Analyse(job);
    }
}

// ---------------------------------------------------------------- owner of a shape

void AddRanges(u64 base, const std::vector<BndEntry>& entries) {
    std::string id;
    for (const auto& e : entries) {
        if (id.empty()) {
            id = ModelId(e.name);
        }
    }
    if (id.empty()) {
        return;
    }
    const int model = ModelFor(id);
    for (const auto& e : entries) {
        g_ranges.push_back({base + e.offset, base + e.offset + e.size, model});
    }
}

int LookupRange(u64 addr) {
    for (auto it = g_ranges.rbegin(); it != g_ranges.rend(); ++it) {
        if (addr >= it->lo && addr < it->hi) {
            return it->model;
        }
    }
    return -1;
}

// The game keeps each objbnd in memory and loads its .hkx in place: the shape lies inside a
// binder entry. Scans back from the shape for a BND4 header whose entry contains it.
int OwnerOf(u64 shape) {
    if (shape == 0) {
        return -1;
    }
    if (const int m = LookupRange(shape); m >= 0) {
        return m;
    }
    const u64 block = shape >> 18;
    if (g_no_owner_blocks.contains(block)) {
        return -1;
    }
    ++st_scans;
    constexpr u64 Chunk = 0x40000, Window = 0x1000000;
    static std::vector<u8> buf(Chunk);
    static std::vector<u8> head(0x10000);
    u64 top = shape & ~u64(15);
    int candidates = 0;
    for (u64 scanned = 0; scanned < Window && top > Chunk && candidates < 6;) {
        const u64 lo = top - Chunk;
        if (!SafeCopy(buf.data(), lo, Chunk)) {
            break;
        }
        for (s64 off = s64(Chunk) - 16; off >= 0; off -= 16) {
            if (std::memcmp(buf.data() + off, "BND4", 4) != 0) {
                continue;
            }
            const u64 b = lo + u64(off);
            ++candidates;
            const u64 n = std::min<u64>(head.size(), shape - b);
            std::vector<BndEntry> entries;
            if (n >= 0x40 && SafeCopy(head.data(), b, n) && ParseBnd4(head.data(), n, entries)) {
                for (const auto& e : entries) {
                    if (shape >= b + e.offset && shape < b + e.offset + e.size) {
                        AddRanges(b, entries);
                        const int m = LookupRange(shape);
                        if (m < 0) {
                            g_no_owner_blocks.insert(block);
                        }
                        return m;
                    }
                }
            }
            if (candidates >= 6) {
                break;
            }
        }
        top = lo;
        scanned += Chunk;
    }
    g_no_owner_blocks.insert(block);
    return -1;
}

// ---------------------------------------------------------------- bodies

struct BodyBytes {
    u8 b[0x90];
    u32 Flags() const {
        return Le32(b + 0x40);
    }
    u32 Filter() const {
        return Le32(b + 0x44);
    }
    u64 Shape() const {
        return Le64(b + 0x48);
    }
    u8 Layer() const {
        return b[0x72];
    }
    float Pos(int i) const {
        float f;
        std::memcpy(&f, b + 0x30 + i * 4, 4);
        return f;
    }
};

const char* Motion(u32 flags) {
    return (flags & 4) ? "keyframed" : (flags & 2) ? "dynamic" : (flags & 1) ? "static" : "none";
}

bool BodyArray(u64 world, u64& data, s32& size) {
    u8 arr[16];
    if (!SafeCopy(arr, world + 0x20, sizeof(arr))) {
        return false;
    }
    data = Le64(arr);
    size = s32(Le32(arr + 8));
    return data != 0 && size > 0 && size <= (1 << 20);
}

void Summary(bool force);

double g_last_census = 0.0;
u32 g_unowned_unadded = 0, g_unowned_lines = 0;
std::set<u64> g_unowned_logged;

void Census(u64 world, World& w) {
    u64 data;
    s32 size;
    if (!BodyArray(world, data, size)) {
        return;
    }
    const s32 n = std::min(size, 1 << 16);
    std::vector<u8> all(size_t(n) * 0x90);
    if (!SafeCopy(all.data(), data, all.size())) {
        return;
    }
    const double now = Now();
    std::unordered_map<u32, double> still;
    int scans = 0;
    for (s32 id = 0; id < n; ++id) {
        BodyBytes bb;
        std::memcpy(bb.b, all.data() + size_t(id) * 0x90, 0x90);
        const u32 flags = bb.Flags();
        const u64 shape = bb.Shape();
        if ((flags & 3) == 0 || shape == 0) {
            continue;
        }
        const auto it = w.bodies.find(u32(id));
        if (it != w.bodies.end() && it->second.shape == shape &&
            it->second.state == State::InWorld) {
            if (it->second.owner >= 0) {
                auto& m = g_models[it->second.owner];
                ++m.in_world;
                if (flags & 4) {
                    ++m.st_keyframed;
                } else if (flags & 2) {
                    ++m.st_dynamic;
                } else {
                    ++m.st_static;
                }
            }
            continue;
        }
        int owner = it != w.bodies.end() && it->second.shape == shape ? it->second.owner : -1;
        if (owner < 0 && (flags & 1) && scans < 64) {
            owner = LookupRange(shape);
            if (owner < 0 && !g_no_owner_blocks.contains(shape >> 18)) {
                ++scans;
                owner = OwnerOf(shape);
            }
        }
        const auto prev = w.unadded_since.find(u32(id));
        const double since = prev != w.unadded_since.end() ? prev->second : now;
        still[u32(id)] = since;
        if (now - since < 10.0 || (flags & 1) == 0) {
            continue;
        }
        if (owner >= 0) {
            ++g_models[owner].not_added;
            continue;
        }
        ++g_unowned_unadded;
        if (g_unowned_lines < 100 && g_unowned_logged.insert((u64(id) << 32) ^ shape).second) {
            ++g_unowned_lines;
            u64 vt = 0;
            Rd(shape, vt);
            LOG_INFO(Core,
                     "[objcol] static body {} (no objbnd owner, shape vtable eboot+{:#x}) layer {} "
                     "filter {:#x} at ({:.1f}, {:.1f}, {:.1f}) exists but is not in the world",
                     id, vt - g_eboot, bb.Layer(), bb.Filter(), bb.Pos(0), bb.Pos(1), bb.Pos(2));
        }
    }
    w.unadded_since = std::move(still);
    w.last_census = now;
}

void CensusAll() {
    for (auto& m : g_models) {
        m.in_world = m.not_added = m.st_static = m.st_dynamic = m.st_keyframed = 0;
    }
    g_unowned_unadded = 0;
    const double now = Now();
    for (auto& [world, w] : g_worlds) {
        if (now - w.last_call < 300.0) {
            Census(world, w);
        }
    }
    g_last_census = Now();
}

std::string Join(const std::vector<std::string>& v, size_t max) {
    std::string s;
    for (size_t i = 0; i < v.size() && i < max; ++i) {
        s += (i ? " " : "") + v[i];
    }
    if (v.size() > max) {
        s += fmt::format(" (+{} more)", v.size() - max);
    }
    return s.empty() ? "none" : s;
}

void Summary(bool force) {
    const double now = Now();
    if (!force && now - g_last_summary < 30.0) {
        return;
    }
    u32 objbnds = 0, solid = 0, destr = 0, nophys = 0, failed = 0, solid_ok = 0, destr_ok = 0;
    std::vector<std::string> never, unadded, removed, ignored, failed_ids;
    for (int i : g_map_models) {
        const auto& m = g_models[i];
        ++objbnds;
        if (m.kind == Kind::NoPhysics) {
            ++nophys;
            continue;
        }
        if (m.kind == Kind::Failed) {
            ++failed;
            failed_ids.push_back(m.id);
            continue;
        }
        if (m.kind != Kind::Solid && m.kind != Kind::Destructible) {
            continue;
        }
        const bool is_solid = m.kind == Kind::Solid;
        is_solid ? ++solid : ++destr;
        if (m.in_world > 0) {
            is_solid ? ++solid_ok : ++destr_ok;
        }
        const std::string tag = is_solid ? m.id : m.id + "(d)";
        if (m.not_added > 0) {
            unadded.push_back(fmt::format("{}x{}", tag, m.not_added));
        }
        if (m.in_world == 0 && m.added == 0 && m.not_added == 0) {
            never.push_back(tag);
        }
        if (m.in_world == 0 && m.removed > 0) {
            removed.push_back(fmt::format("{}x{}", tag, m.removed));
        }
        if (m.noop > 0) {
            ignored.push_back(fmt::format("{}x{}", tag, m.noop));
        }
    }
    const std::string text = fmt::format(
        "[objcol] map {}: objbnds {} (static-solid {}, destructible {}, no collision data {}, "
        "not parsed {}) | in the Havok world: static-solid {}/{}, destructible {}/{} | MISSING "
        "solid objects: no body ever added: {} | created but not in the world (10 s+): {} | "
        "added then removed by the game: {} | add ignored (mode 0): {} | not parsed: {} | static "
        "bodies without an objbnd owner not in the world: {} | bodies added {} (binder known {}), "
        "add calls immediate {} deferred {} ignored {}, removes {}, bad ids {}, owner scans {}, "
        "worlds {}, hooks {}",
        g_map, objbnds, solid, destr, nophys, failed, solid_ok, solid, destr_ok, destr,
        Join(never, 40), Join(unadded, 40), Join(removed, 40), Join(ignored, 20),
        Join(failed_ids, 20), g_unowned_unadded, st_bodies, st_owned, st_calls[1], st_calls[2],
        st_calls[0], st_removed, st_bad_id, st_scans, g_worlds.size(),
        g_hooked.load() ? "on" : "off");
    const std::string key = text.substr(0, text.find(" | bodies added"));
    if (!force && key == g_last_summary_text) {
        return;
    }
    g_last_summary_text = key;
    g_last_summary = now;
    LOG_INFO(Core, "{}", text);
}

void OnAdd(u64 world, u64 ids_ptr, s32 n, s32 mode, u64 ret) {
    if (n <= 0 || n > (1 << 16)) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    ++st_calls[mode == 0 ? 0 : mode == 2 ? 2 : 1];
    u64 data;
    s32 size;
    if (!BodyArray(world, data, size)) {
        return;
    }
    std::vector<u32> ids(size_t(std::min(n, 8192)));
    if (!SafeCopy(ids.data(), ids_ptr, ids.size() * 4)) {
        return;
    }
    auto& w = g_worlds[world];
    w.last_call = Now();
    for (const u32 id : ids) {
        BodyBytes bb;
        if (id >= u32(size) || !SafeCopy(bb.b, data + u64(id) * 0x90, 0x90)) {
            ++st_bad_id;
            continue;
        }
        auto& b = w.bodies[id];
        const u64 shape = bb.Shape();
        if (b.shape != shape) {
            b = Body{};
            b.shape = shape;
            b.owner = OwnerOf(shape);
        }
        b.flags = bb.Flags();
        if (mode == 0) {
            if (b.state == State::InWorld) {
                continue;
            }
            b.state = State::Ignored;
            if (b.owner >= 0) {
                ++g_models[b.owner].noop;
            }
            continue;
        }
        if (mode == 2) {
            if (b.state != State::InWorld) {
                b.state = State::Deferred;
            }
            continue;
        }
        const bool first = b.state != State::InWorld && b.state != State::Removed;
        b.state = State::InWorld;
        w.unadded_since.erase(id);
        ++st_bodies;
        if (b.owner < 0) {
            continue;
        }
        ++st_owned;
        auto& m = g_models[b.owner];
        if (first) {
            ++m.added;
        }
        if (!b.logged && (b.flags & 5) && g_body_lines < 4000) {
            b.logged = true;
            ++g_body_lines;
            LOG_INFO(Core,
                     "[objcol] body {} {} ({}): {} layer {} filter {:#x} at ({:.1f}, {:.1f}, "
                     "{:.1f}) added by eboot+{:#x}",
                     id, m.id, KindName(m.kind), Motion(b.flags), bb.Layer(), bb.Filter(),
                     bb.Pos(0), bb.Pos(1), bb.Pos(2), ret - g_eboot);
        }
    }
    if (Now() - g_last_census >= 5.0) {
        CensusAll();
        Summary(false);
    }
}

void OnRemove(u64 world, u64 ids_ptr, s32 n, u64 ret) {
    if (n <= 0 || n > (1 << 16)) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    std::vector<u32> ids(size_t(std::min(n, 8192)));
    if (!SafeCopy(ids.data(), ids_ptr, ids.size() * 4)) {
        return;
    }
    auto& w = g_worlds[world];
    w.last_call = Now();
    for (const u32 id : ids) {
        auto it = w.bodies.find(id);
        if (it == w.bodies.end() || it->second.state == State::Removed) {
            continue;
        }
        ++st_removed;
        it->second.state = State::Removed;
        if (it->second.owner < 0) {
            continue;
        }
        auto& m = g_models[it->second.owner];
        ++m.removed;
        if (!m.removal_logged && (it->second.flags & 5) && g_body_lines < 4000) {
            m.removal_logged = true;
            ++g_body_lines;
            LOG_INFO(Core, "[objcol] body {} {} ({}): {} removed from the world by eboot+{:#x}", id,
                     m.id, KindName(m.kind), Motion(it->second.flags), ret - g_eboot);
        }
    }
}

void PS4_SYSV_ABI HookAdd(const u64* regs) {
    // regs: rbx r11 r10 r9 r8 rdi rsi rdx rcx rax, return address
    OnAdd(regs[5], regs[6], s32(regs[7]), s32(regs[8]), regs[10]);
}

void PS4_SYSV_ABI HookRemove(const u64* regs) {
    OnRemove(regs[5], regs[6], s32(regs[7]), regs[10]);
}

// ---------------------------------------------------------------- detours

// addBodies: prologue (20 bytes relocated), argument spills, test edx / test ecx, the
// world+0x498 flag check.
constexpr int SigAdd[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54,
                          0x53, 0x48, 0x81, 0xec, 0xf8, 0x00, 0x00, 0x00, 0x89, 0x55, 0x8c, 0x48,
                          0x89, 0x75, 0x98, 0x49, 0x89, 0xff, 0x4c, 0x89, 0x7d, 0xa8, 0x48, 0x8b,
                          0x1d, W,    W,    W,    W,    0x48, 0x8b, 0x03, 0x48, 0x89, 0x45, 0xd0,
                          0x85, 0xd2, 0x0f, 0x8e, W,    W,    W,    W,    0x85, 0xc9, 0x0f, 0x84,
                          W,    W,    W,    W,    0x41, 0x80, 0xbf, 0x98, 0x04, 0x00, 0x00, 0x00};
constexpr u32 AddPrologue = 20;

// removeBodies: prologue (17 bytes relocated), spills, test r11d, the world+0x10 / +0x4e0 /
// +0xb8 / +0xc8 (pending add lists) addresses.
constexpr int SigRemove[] = {
    0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48,
    0x83, 0xec, 0x58, 0x89, 0x4d, 0x88, 0x41, 0x89, 0xd3, 0x44, 0x89, 0x5d, 0x8c, 0x49,
    0x89, 0xf4, 0x4c, 0x89, 0x65, 0xb8, 0x48, 0x89, 0x7d, 0xd0, 0x45, 0x85, 0xdb, 0x0f,
    0x8e, W,    W,    W,    W,    0x48, 0x8d, 0x47, 0x10, 0x48, 0x89, 0x45, 0x80, 0x48,
    0x8d, 0x87, 0xe0, 0x04, 0x00, 0x00, 0x48, 0x89, 0x45, 0xc8, 0x48, 0x8d, 0x87, 0xb8,
    0x00, 0x00, 0x00, 0x48, 0x89, 0x45, 0x98, 0x48, 0x8d, 0x87, 0xc8, 0x00, 0x00, 0x00};
constexpr u32 RemovePrologue = 17;

std::optional<u64> Find(const u8* data, u64 size, const int* sig, u32 n) {
    std::optional<u64> found;
    for (u64 i = 0; i + n <= size; ++i) {
        if (data[i] != u8(sig[0])) {
            continue;
        }
        u32 k = 1;
        while (k < n && (sig[k] == W || data[i + k] == u8(sig[k]))) {
            ++k;
        }
        if (k == n) {
            if (found) {
                return std::nullopt;
            }
            found = i;
        }
    }
    return found;
}

void* AllocExec(size_t n) {
#ifdef _WIN32
    return VirtualAlloc(nullptr, n, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
    void* p =
        mmap(nullptr, n, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}

// Saves the caller-saved registers and xmm0..7, calls host(regs), restores them, runs the
// relocated prologue and jumps back after it (same cave as bb_native_mv_hook.cpp).
std::vector<u8> BuildCave(u64 host, const u8* prologue, u32 len, u64 back) {
    std::vector<u8> e;
    const auto put = [&](std::initializer_list<u8> v) { e.insert(e.end(), v); };
    const auto put64 = [&](u64 v) {
        for (int i = 0; i < 8; ++i) {
            e.push_back(u8(v >> (i * 8)));
        }
    };
    put({0x50, 0x51, 0x52, 0x56, 0x57, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53, 0x53});
    put({0x48, 0x89, 0xe3});
    put({0x48, 0x83, 0xe4, 0xf0});
    put({0x48, 0x81, 0xec, 0x80, 0x00, 0x00, 0x00});
    for (u8 i = 0; i < 8; ++i) {
        if (i == 0) {
            put({0xf3, 0x0f, 0x7f, 0x04, 0x24});
        } else {
            put({0xf3, 0x0f, 0x7f, u8(0x44 | (i << 3)), 0x24, u8(i * 16)});
        }
    }
    put({0x48, 0x89, 0xdf});
    put({0x48, 0xb8});
    put64(host);
    put({0xff, 0xd0});
    for (u8 i = 0; i < 8; ++i) {
        if (i == 0) {
            put({0xf3, 0x0f, 0x6f, 0x04, 0x24});
        } else {
            put({0xf3, 0x0f, 0x6f, u8(0x44 | (i << 3)), 0x24, u8(i * 16)});
        }
    }
    put({0x48, 0x89, 0xdc});
    put({0x5b, 0x41, 0x5b, 0x41, 0x5a, 0x41, 0x59, 0x41, 0x58, 0x5f, 0x5e, 0x5a, 0x59, 0x58});
    e.insert(e.end(), prologue, prologue + len);
    put({0xff, 0x25, 0x00, 0x00, 0x00, 0x00});
    put64(back);
    return e;
}

void Detour(u8* at, const u8* to) {
    u8 j[14] = {0xff, 0x25, 0, 0, 0, 0};
    const u64 v = reinterpret_cast<u64>(to);
    std::memcpy(j + 6, &v, 8);
    std::memcpy(at, j, sizeof(j));
}

std::string MapOf(const std::string& path) {
    // /map/mapstudio/mAA_BB_CC_DD.msb.dcx
    const auto name = BaseName(path);
    const auto dot = name.find('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

} // namespace

bool Enabled() {
    return g_enabled;
}

void OnGuestFileOpen(const char* guest_path, const char* /*host_path*/) {
    if (!g_enabled || guest_path == nullptr) {
        return;
    }
    const std::string p = guest_path;
    if (p.find("/mapstudio/") != std::string::npos && p.find(".msb") != std::string::npos) {
        std::scoped_lock lk{g_mutex};
        if (!g_map_models.empty()) {
            CensusAll();
            Summary(true);
        }
        g_map = MapOf(p);
        g_map_models.clear();
        g_last_summary_text.clear();
        g_last_summary = Now();
        LOG_INFO(Core, "[objcol] map {} loading (hooks {})", g_map, g_hooked.load() ? "on" : "off");
        return;
    }
    if (p.find("/obj/") == std::string::npos || !p.ends_with(".objbnd.dcx")) {
        return;
    }
    const std::string id = ModelId(BaseName(p));
    if (id.empty()) {
        return;
    }
    int model;
    {
        std::scoped_lock lk{g_mutex};
        model = ModelFor(id);
        auto& m = g_models[model];
        if (m.map.empty()) {
            m.map = g_map;
        }
        if (std::find(g_map_models.begin(), g_map_models.end(), model) == g_map_models.end()) {
            g_map_models.push_back(model);
        }
        if (m.kind != Kind::Unknown) {
            return;
        }
    }
    std::scoped_lock lk{g_job_mutex};
    g_jobs.push_back({p, model});
    if (!g_worker_started) {
        g_worker_started = true;
        std::thread(Worker).detach();
    }
    g_job_cv.notify_one();
}

void OnEbootPatched(uintptr_t eboot_base, u64 image_size) {
    if (!g_enabled) {
        return;
    }
    g_eboot = eboot_base;
    const u8* img = reinterpret_cast<const u8*>(eboot_base);
    const u64 code = std::min<u64>(image_size, 0x4000000);
    const auto add = Find(img, code, SigAdd, std::size(SigAdd));
    const auto rem = Find(img, code, SigRemove, std::size(SigRemove));
    if (!add || !rem) {
        LOG_WARNING(Core,
                    "[objcol] hknpWorld::addBodies {} / removeBodies {}: game code untouched, only "
                    "the objbnd file analysis runs",
                    add ? fmt::format("eboot+{:#x}", *add) : std::string("not found (unique)"),
                    rem ? fmt::format("eboot+{:#x}", *rem) : std::string("not found (unique)"));
        return;
    }
    u8* cave = static_cast<u8*>(AllocExec(4096));
    if (cave == nullptr) {
        LOG_WARNING(Core, "[objcol] no executable memory for the detours");
        return;
    }
    u8* at_add = reinterpret_cast<u8*>(eboot_base + *add);
    u8* at_rem = reinterpret_cast<u8*>(eboot_base + *rem);
    const auto c1 = BuildCave(reinterpret_cast<u64>(&HookAdd), at_add, AddPrologue,
                              reinterpret_cast<u64>(at_add + AddPrologue));
    const auto c2 = BuildCave(reinterpret_cast<u64>(&HookRemove), at_rem, RemovePrologue,
                              reinterpret_cast<u64>(at_rem + RemovePrologue));
    std::memcpy(cave, c1.data(), c1.size());
    u8* cave2 = cave + ((c1.size() + 15) & ~size_t(15));
    std::memcpy(cave2, c2.data(), c2.size());
    Detour(at_add, cave);
    Detour(at_rem, cave2);
    g_hooked = true;
    LOG_INFO(Core,
             "[objcol] on: hknpWorld::addBodies eboot+{:#x}, removeBodies eboot+{:#x} hooked; "
             "objbnd analysis on a worker thread; per-map summary lines '[objcol] map'",
             *add, *rem);
}

} // namespace BB::ObjCol
