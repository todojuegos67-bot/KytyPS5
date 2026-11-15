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
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fmt/format.h>
#include <zlib.h>
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/singleton.h"
#include "core/file_sys/fs.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#if defined(__linux__)
#include <sys/uio.h>
#endif
#include <unistd.h>
#endif

namespace BB::ObjCol {
namespace {

constexpr int W = -1;

bool ReadEnv() {
    const char* v = std::getenv("BB_OBJCOL_DIAG");
    return v != nullptr && v[0] == '1';
}

bool ReadFix() {
    const char* v = std::getenv("BB_OBJCOL_FIX");
    return v != nullptr && v[0] == '1';
}

const bool g_fix = ReadFix();
const bool g_enabled = ReadEnv() || g_fix;

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
#elif defined(__linux__)
    iovec local{dst, len};
    iovec remote{reinterpret_cast<void*>(src), len};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == static_cast<ssize_t>(len);
#else
    (void)dst;
    return false;
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
        return "solid (no fragment hkx)";
    case Kind::Destructible:
        return "breakable (_1.hkx fragments)";
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
    struct Pending {
        double t;
        u64 ret;
        bool logged;
    };
    std::unordered_map<u32, Pending> deferred; // mode-2 adds waiting for the commit
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
u64 g_eboot = 0;
u32 g_body_lines = 0;
std::atomic<bool> g_hooked{false};
u64 st_calls[3] = {}, st_bodies = 0, st_owned = 0, st_bad_id = 0, st_removed = 0;
u64 st_deferred_bodies = 0, st_committed_bodies = 0, st_deferred_removed = 0;
u64 st_deferred_before_step = 0, st_added_before_step = 0;
double g_first_step = -1.0;
double g_last_summary = 0.0;
std::string g_last_summary_text;
std::set<std::string> g_mod_logged;
u32 g_mod_lines = 0;

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

struct SnapBody {
    float p[3];
    u64 vt;
    u32 flags, filter, id;
    u32 layer;
};

struct Job {
    std::string guest;
    int model;
    std::string host; // set: MSB comparison job
    // set: boot/reload body comparison (file I/O off the hooked guest threads)
    std::vector<SnapBody> snap;
    std::string map;
    int visit = 0;
    double since = 0.0;
    bool compare = false;
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
        const u64 fsize = file->Size();
        if (fsize > (256ull << 20)) {
            why = fmt::format("file too large ({} bytes)", fsize);
        } else {
            raw.resize(fsize);
        }
    }
    if (why.empty() && file) {
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
            breakable = breakable || brk || (phys && lower.ends_with("_1.hkx"));
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

// ---------------------------------------------------------------- MSB (map layout) source check

std::string Join(const std::vector<std::string>& v, size_t max);
std::string MapOf(const std::string& path);

const char* SourceOf(const std::string& host) {
    if (host.find("-mods") != std::string::npos) {
        return "MODS folder";
    }
    if (host.find("-patch") != std::string::npos || host.find("-UPDATE") != std::string::npos) {
        return "update/patch";
    }
    return host.empty() ? "unknown" : "base game";
}

bool ReadHost(const std::string& utf8, std::vector<u8>& out) {
    std::ifstream f(std::filesystem::path(reinterpret_cast<const char8_t*>(utf8.c_str())),
                    std::ios::binary);
    if (!f) {
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return !out.empty() && out.size() < (256ull << 20);
}

// UTF-16 / ASCII tokens "<letter><6 digits>_<4 digits>" (MSB part names: o = object,
// h = collision, m = map piece, c = character...) and "<letter><6 digits>" (model names).
void MsbNames(const std::vector<u8>& d, std::set<std::string>& parts,
              std::set<std::string>& models) {
    const auto take = [&](const std::string& t) {
        if (t.size() == 12 && t[7] == '_' && std::isalpha(u8(t[0]))) {
            bool ok = true;
            for (size_t i = 1; i < 12 && ok; ++i) {
                ok = i == 7 || (t[i] >= '0' && t[i] <= '9');
            }
            if (ok) {
                parts.insert(t);
            }
        } else if (t.size() == 7 && std::isalpha(u8(t[0]))) {
            bool ok = true;
            for (size_t i = 1; i < 7 && ok; ++i) {
                ok = t[i] >= '0' && t[i] <= '9';
            }
            if (ok) {
                models.insert(t);
            }
        }
    };
    for (int step = 1; step <= 2; ++step) {
        for (size_t start = 0; start < size_t(step); ++start) {
            std::string cur;
            for (size_t i = start; i + step <= d.size(); i += step) {
                const u16 c = step == 1 ? d[i] : u16(d[i] | (d[i + 1] << 8));
                if (c >= 0x21 && c < 0x7f) {
                    cur.push_back(char(c));
                    continue;
                }
                take(cur);
                cur.clear();
            }
            take(cur);
        }
    }
}

std::string Diff(const std::set<std::string>& a, const std::set<std::string>& b, char kind,
                 size_t max) {
    std::vector<std::string> v;
    for (const auto& x : a) {
        if ((kind == 0 || x[0] == kind) && !b.contains(x)) {
            v.push_back(x);
        }
    }
    return Join(v, max);
}

size_t Count(const std::set<std::string>& a, char kind) {
    return size_t(
        std::count_if(a.begin(), a.end(), [&](const std::string& x) { return x[0] == kind; }));
}

struct MsbPart;
std::vector<MsbPart> MsbParts(const std::vector<u8>& d);
void StoreMsbParts(const std::string& map, const std::vector<u8>& mod_msb,
                   const std::set<std::string>& vanilla_parts, bool have_vanilla,
                   const std::vector<u8>& vanilla_msb);

void AnalyseMsb(const Job& job) {
    const std::string& host = job.host;
    std::vector<u8> raw, msb;
    std::string why;
    if (!ReadHost(host, raw) || !Undcx(raw, msb, why)) {
        LOG_INFO(Core, "[objcol] msb {} served from {} ({}): not readable {}", job.guest, host,
                 SourceOf(host), why);
        return;
    }
    std::set<std::string> parts, models;
    MsbNames(msb, parts, models);
    std::string text = fmt::format(
        "[objcol] msb {} served from {} ({}), {} bytes crc {:08x}: object parts {}, collision "
        "parts {}, object models {}",
        MapOf(job.guest), host, SourceOf(host), raw.size(),
        u32(crc32(0, raw.data(), uInt(raw.size()))), Count(parts, 'o'), Count(parts, 'h'),
        Count(models, 'o'));
    const auto at = host.find("-mods");
    if (at != std::string::npos) {
        bool found = false;
        for (const char* repl : {"-patch", "-UPDATE", ""}) {
            const std::string van = host.substr(0, at) + repl + host.substr(at + 5);
            std::vector<u8> vraw, vmsb;
            std::string vwhy;
            if (!ReadHost(van, vraw) || !Undcx(vraw, vmsb, vwhy)) {
                continue;
            }
            found = true;
            std::set<std::string> vparts, vmodels;
            MsbNames(vmsb, vparts, vmodels);
            StoreMsbParts(MapOf(job.guest), msb, vparts, true, vmsb);
            text += fmt::format(
                " | vanilla {} {} bytes crc {:08x}{}: object parts {}, collision parts {} | "
                "object parts only in vanilla: {} | only in the mod: {} | collision parts only in "
                "vanilla: {} | only in the mod: {} | object models only in vanilla: {} | only in "
                "the mod: {}",
                van, vraw.size(), u32(crc32(0, vraw.data(), uInt(vraw.size()))),
                vraw == raw ? " (identical)" : "", Count(vparts, 'o'), Count(vparts, 'h'),
                Diff(vparts, parts, 'o', 60), Diff(parts, vparts, 'o', 60),
                Diff(vparts, parts, 'h', 60), Diff(parts, vparts, 'h', 60),
                Diff(vmodels, models, 'o', 40), Diff(models, vmodels, 'o', 40));
            break;
        }
        if (!found) {
            text += " | no vanilla copy found next to the mods folder (-patch / -UPDATE / base)";
            StoreMsbParts(MapOf(job.guest), msb, {}, false, {});
        }
    } else {
        StoreMsbParts(MapOf(job.guest), msb, {}, false, {});
    }
    LOG_INFO(Core, "{}", text);
}

// MSB part records (MSBB / MSB3 family): +0x00 name offset (i64, from the record), +0x08 type,
// +0x0c index, +0x10 model index, +0x18 sib offset, +0x20 position, +0x2c rotation, +0x38 scale.
// Records are found from their UTF-16 names and kept only when the scale is plausible.
struct MsbPart {
    std::string name;
    float pos[3];
    s32 type, index, model;
    bool mod_only = false;
};

u32 st_msb_names = 0, st_msb_backref = 0;
std::string g_msb_sample;

// Names in UTF-16 or single-byte; the record is the 4-aligned position up to 0x2000 bytes
// before the name whose first i64 is the distance to it. Field run 4: the first version (UTF-16
// only, 0x800 back, scale check) matched 0 of 1731 parts, so only the position and rotation are
// checked now and the first record found is dumped.
std::vector<MsbPart> MsbParts(const std::vector<u8>& d) {
    std::vector<MsbPart> out;
    const size_t n = d.size();
    st_msb_names = st_msb_backref = 0;
    g_msb_sample.clear();
    for (int wide = 1; wide >= 0; --wide) {
        const size_t step = wide ? 2 : 1;
        for (size_t i = 0; i + 13 * step <= n; i += step) {
            if (d[i] != 'o' || (wide && d[i + 1] != 0) || (i >= step && d[i - step] != 0)) {
                continue;
            }
            std::string t;
            for (size_t k = i; k + step <= n && t.size() < 13; k += step) {
                if (d[k] == 0 || (wide && d[k + 1] != 0)) {
                    break;
                }
                t.push_back(char(d[k]));
            }
            if (t.size() != 12 || t[7] != '_' || (d[i + 12 * step] != 0)) {
                continue;
            }
            ++st_msb_names;
            for (size_t back = 8 + (i & 3); back <= 0x2000 && back <= i; back += 4) {
                const size_t r = i - back; // 4-aligned
                if (r + 0x50 > n || Le64(d.data() + r) != back) {
                    continue;
                }
                ++st_msb_backref;
                float f[6];
                std::memcpy(f, d.data() + r + 0x20, sizeof(f));
                bool ok = true;
                for (int k = 0; k < 3; ++k) {
                    ok = ok && std::isfinite(f[k]) && std::fabs(f[k]) < 1e5f &&
                         std::isfinite(f[3 + k]) && std::fabs(f[3 + k]) <= 1000.f;
                }
                if (g_msb_sample.empty()) {
                    for (size_t k = 0; k < 0x50; k += 4) {
                        g_msb_sample += fmt::format("{:08x} ", Le32(d.data() + r + k));
                    }
                    g_msb_sample += fmt::format("(name {} at +{:#x}, {})", t, back,
                                                wide ? "UTF-16" : "single-byte");
                }
                if (!ok) {
                    break;
                }
                MsbPart p;
                p.name = t;
                std::copy(f, f + 3, p.pos);
                p.type = s32(Le32(d.data() + r + 8));
                p.index = s32(Le32(d.data() + r + 0xc));
                p.model = s32(Le32(d.data() + r + 0x10));
                out.push_back(std::move(p));
                break;
            }
        }
        if (!out.empty()) {
            break;
        }
    }
    return out;
}

std::map<std::string, std::vector<MsbPart>> g_msb_parts; // by map, under g_mutex

void StoreMsbParts(const std::string& map, const std::vector<u8>& mod_msb,
                   const std::set<std::string>& vanilla_parts, bool have_vanilla,
                   const std::vector<u8>& vanilla_msb) {
    auto parts = MsbParts(mod_msb);
    const u32 names = st_msb_names, backrefs = st_msb_backref;
    const std::string sample = g_msb_sample;
    std::map<s32, std::string> vanilla_index;
    if (have_vanilla) {
        for (const auto& v : MsbParts(vanilla_msb)) {
            vanilla_index.emplace(v.index, v.name);
        }
    }
    std::string mods;
    u32 listed = 0;
    for (auto& p : parts) {
        p.mod_only = have_vanilla && !vanilla_parts.contains(p.name);
        if (p.mod_only && ++listed > 60) {
            if (listed == 61) {
                mods += " (more not listed)";
            }
        } else if (p.mod_only) {
            const auto it = vanilla_index.find(p.index);
            mods += fmt::format("{}{} index {} type {} model {} at ({:.1f}, {:.1f}, {:.1f}){}",
                                mods.empty() ? "" : "; ", p.name, p.index, p.type, p.model,
                                p.pos[0], p.pos[1], p.pos[2],
                                it != vanilla_index.end()
                                    ? " (vanilla index " + std::to_string(p.index) + " is " +
                                          it->second + ")"
                                    : "");
        }
    }
    LOG_INFO(Core,
             "[objcol] msb {} object part records {} (with position; names {}, records "
             "pointing at a name {}; first record: {}){}",
             map, parts.size(), names, backrefs, sample.empty() ? std::string("none") : sample,
             mods.empty() ? std::string() : " | mod-only parts: " + mods);
    std::scoped_lock lk{g_mutex};
    g_msb_parts[map] = std::move(parts);
}

void CompareJob(const Job& job);

void Worker() {
    for (;;) {
        Job job;
        {
            std::unique_lock lk{g_job_mutex};
            g_job_cv.wait(lk, [] { return !g_jobs.empty(); });
            job = std::move(g_jobs.front());
            g_jobs.pop_front();
        }
        try {
            if (job.compare) {
                CompareJob(job);
            } else if (job.host.empty()) {
                Analyse(job);
            } else {
                AnalyseMsb(job);
            }
        } catch (...) {
        }
    }
}

// ---------------------------------------------------------------- owner of a shape

int LookupRange(u64 addr) {
    for (auto it = g_ranges.rbegin(); it != g_ranges.rend(); ++it) {
        if (addr >= it->lo && addr < it->hi) {
            return it->model;
        }
    }
    return -1;
}

// Field run 1 (ba42d5e98): no shape lay inside an in-memory binder (the .hkx is not loaded in
// place), so binder attribution is off; '[objcol] player' lines attribute by position instead.
int OwnerOf(u64 shape) {
    return shape == 0 ? -1 : LookupRange(shape);
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
        if (owner < 0 && (flags & 1)) {
            owner = OwnerOf(shape);
        }
        const auto prev = w.unadded_since.find(u32(id));
        const double since = prev != w.unadded_since.end() ? prev->second : now;
        still[u32(id)] = since;
        if (now - since < 10.0) {
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
                     "[objcol] {} body {} (shape vtable eboot+{:#x}) layer {} filter {:#x} at "
                     "({:.1f}, {:.1f}, {:.1f}) exists but is not in the world ({})",
                     Motion(flags), id, vt - g_eboot, bb.Layer(), bb.Filter(), bb.Pos(0), bb.Pos(1),
                     bb.Pos(2),
                     it == w.bodies.end() || it->second.shape != shape ? "never added"
                     : it->second.state == State::Removed              ? "removed"
                     : it->second.state == State::Deferred ? "add deferred, not committed"
                                                           : "add ignored");
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
    const std::string per_object =
        st_owned == 0
            ? std::string("per-object body attribution unavailable (see '[objcol] player' lines)")
            : fmt::format("in the Havok world: solid {}/{}, breakable {}/{} | MISSING: no body "
                          "ever added: {} | created but not in the world (10 s+): {} | added "
                          "then removed: {} | add ignored (mode 0): {}",
                          solid_ok, solid, destr_ok, destr, Join(never, 40), Join(unadded, 40),
                          Join(removed, 40), Join(ignored, 20));
    const std::string text = fmt::format(
        "[objcol] map {}: objbnds {} (solid {}, breakable {}, no collision data {}, not parsed "
        "{}: {}) | {} | static bodies created but not in the world (10 s+): {} | bodies added {} "
        "(binder known {}), add calls immediate {} deferred {} ignored {}, removes {}, bad ids "
        "{}, worlds {}, hooks {}",
        g_map, objbnds, solid, destr, nophys, failed, Join(failed_ids, 20), per_object,
        g_unowned_unadded, st_bodies, st_owned, st_calls[1], st_calls[2], st_calls[0], st_removed,
        st_bad_id, g_worlds.size(), g_hooked.load() ? "on" : "off");
    const std::string key = text.substr(0, text.find(" | bodies added"));
    if (!force && key == g_last_summary_text) {
        return;
    }
    g_last_summary_text = key;
    g_last_summary = now;
    LOG_INFO(Core, "{}", text);
}

// ---------------------------------------------------------------- boot / reload checks
// Static and keyframed bodies in the world 30 s and 90 s after each map layout opens are written
// to user/log/objcol_last_<map>.txt and compared with objcol_ref_<map>.txt (the largest set seen
// for that map in any session), by shape class and position: a cold boot that loses object
// collision shows the missing bodies against a session where they were there.

double g_map_open = -1.0, g_map_tick = 0.0;
int g_map_visit = 0;
bool g_cmp30 = false, g_cmp90 = false;
struct PendingCmp {
    std::string map;
    int visit;
    double open;
    bool done[2];
};
std::vector<PendingCmp> g_pending_cmp;
std::map<std::string, int> g_visits;
u32 g_deferred_lines = 0;
u32 g_world_lines = 0;

std::vector<SnapBody> Snapshot() {
    std::vector<SnapBody> out;
    const double now = Now();
    for (auto& [world, w] : g_worlds) {
        u64 data;
        s32 size;
        if (now - w.last_call > 300.0 || !BodyArray(world, data, size)) {
            continue;
        }
        const s32 n = std::min(size, 1 << 16);
        std::vector<u8> all(size_t(n) * 0x90);
        if (!SafeCopy(all.data(), data, all.size())) {
            continue;
        }
        for (const auto& [id, b] : w.bodies) {
            if (b.state != State::InWorld || id >= u32(n)) {
                continue;
            }
            BodyBytes bb;
            std::memcpy(bb.b, all.data() + size_t(id) * 0x90, 0x90);
            if (bb.Shape() != b.shape || (bb.Flags() & 5) == 0) {
                continue;
            }
            u64 vt = 0;
            Rd(b.shape, vt);
            out.push_back({{bb.Pos(0), bb.Pos(1), bb.Pos(2)},
                           vt - g_eboot,
                           bb.Flags(),
                           bb.Filter(),
                           id,
                           bb.Layer()});
        }
    }
    return out;
}

std::filesystem::path SnapPath(const char* kind, const std::string& map) {
    std::string name = map;
    for (char& c : name) {
        if (!std::isalnum(u8(c)) && c != '_' && c != '-') {
            c = '_';
        }
    }
    return Common::FS::GetUserPath(Common::FS::PathType::LogDir) /
           fmt::format("objcol_{}_{}.txt", kind, name);
}

void WriteSnap(const std::filesystem::path& path, const std::vector<SnapBody>& snap,
               const std::string& header) {
    std::ofstream f(path, std::ios::trunc);
    f << header << '\n';
    for (const auto& b : snap) {
        f << fmt::format("{:.2f} {:.2f} {:.2f} {:x} {:x} {} {:x} {}\n", b.p[0], b.p[1], b.p[2],
                         b.vt, b.flags, b.layer, b.filter, b.id);
    }
}

bool ReadSnap(const std::filesystem::path& path, std::vector<SnapBody>& snap, std::string& header) {
    std::ifstream f(path);
    if (!f || !std::getline(f, header)) {
        return false;
    }
    if (header.size() > 200) {
        header.resize(200);
    }
    std::string line;
    while (snap.size() < (1u << 17) && std::getline(f, line)) {
        SnapBody b{};
        unsigned long long vt = 0;
        unsigned flags = 0, layer = 0, filter = 0, id = 0;
        if (std::sscanf(line.c_str(), "%f %f %f %llx %x %u %x %u", &b.p[0], &b.p[1], &b.p[2], &vt,
                        &flags, &layer, &filter, &id) == 8) {
            b.vt = vt;
            b.flags = flags;
            b.layer = layer;
            b.filter = filter;
            b.id = id;
            snap.push_back(b);
        }
    }
    return true;
}

// The game shifts the Havok world origin (hknpWorld::shiftWorld 0xb95c60, called from the
// FD4 command flush 0x1806e90), so two sessions can differ by a constant offset: the most common
// offset between bodies of the same shape class (0.5 m bins).
std::array<float, 3> CommonOffset(const std::vector<SnapBody>& a, const std::vector<SnapBody>& b) {
    std::map<std::tuple<int, int, int>, int> votes;
    for (size_t i = 0; i < a.size() && i < 300; ++i) {
        for (const auto& y : b) {
            if (a[i].vt != y.vt) {
                continue;
            }
            ++votes[{int(std::lround((a[i].p[0] - y.p[0]) * 2.f)),
                     int(std::lround((a[i].p[1] - y.p[1]) * 2.f)),
                     int(std::lround((a[i].p[2] - y.p[2]) * 2.f))}];
        }
    }
    std::array<float, 3> best{0.f, 0.f, 0.f};
    int top = 0;
    for (const auto& [k, v] : votes) {
        if (v > top) {
            top = v;
            best = {std::get<0>(k) * 0.5f, std::get<1>(k) * 0.5f, std::get<2>(k) * 0.5f};
        }
    }
    return best;
}

// Bodies of a with no body of the same shape class within 0.6 m in b (b shifted by off).
std::vector<const SnapBody*> Unmatched(const std::vector<SnapBody>& a,
                                       const std::vector<SnapBody>& b,
                                       const std::array<float, 3>& off) {
    std::unordered_map<u64, std::vector<const SnapBody*>> by_shape;
    for (const auto& y : b) {
        by_shape[y.vt].push_back(&y);
    }
    std::vector<const SnapBody*> out;
    for (const auto& x : a) {
        bool hit = false;
        const auto it = by_shape.find(x.vt);
        if (it == by_shape.end()) {
            out.push_back(&x);
            continue;
        }
        for (const SnapBody* yp : it->second) {
            const auto& y = *yp;
            const float dx = x.p[0] - y.p[0] - off[0], dy = x.p[1] - y.p[1] - off[1],
                        dz = x.p[2] - y.p[2] - off[2];
            if (x.vt == y.vt && dx * dx + dy * dy + dz * dz < 0.36f) {
                hit = true;
                break;
            }
        }
        if (!hit) {
            out.push_back(&x);
        }
    }
    return out;
}

std::string ListSnap(const std::vector<const SnapBody*>& v, size_t max) {
    std::string s;
    for (size_t i = 0; i < v.size() && i < max; ++i) {
        const auto& b = *v[i];
        s += fmt::format("{}({:.1f}, {:.1f}, {:.1f}) {} layer {} filter {:#x} shape eboot+{:#x}",
                         i ? "; " : "", b.p[0], b.p[1], b.p[2], Motion(b.flags), b.layer, b.filter,
                         b.vt);
    }
    if (v.size() > max) {
        s += fmt::format(" (+{} more)", v.size() - max);
    }
    return s.empty() ? "none" : s;
}

// Under g_mutex on a hooked guest thread: only the snapshot; the files and the comparison run on
// the worker.
void CompareMap(double now) {
    Job job;
    job.compare = true;
    job.snap = Snapshot();
    job.map = g_map;
    job.visit = g_map_visit;
    job.since = now - g_map_open;
    std::scoped_lock jl{g_job_mutex};
    if (g_jobs.size() >= 64) {
        return;
    }
    g_jobs.push_back(std::move(job));
    if (!g_worker_started) {
        g_worker_started = true;
        std::thread(Worker).detach();
    }
    g_job_cv.notify_one();
}

void CompareJob(const Job& job) {
    const auto& cur = job.snap;
    const std::string header =
        fmt::format("map {} session {} visit {} at +{:.0f} s: {} static/keyframed bodies", job.map,
                    std::time(nullptr), job.visit, job.since, cur.size());
    try {
        WriteSnap(SnapPath("last", job.map), cur, header);
        std::vector<SnapBody> ref;
        std::string ref_header;
        const bool have_ref = ReadSnap(SnapPath("ref", job.map), ref, ref_header);
        if (have_ref) {
            const auto off = CommonOffset(ref, cur);
            const auto missing = Unmatched(ref, cur, off);
            const auto extra = Unmatched(cur, ref, {-off[0], -off[1], -off[2]});
            LOG_INFO(Core,
                     "[objcol] map {} visit #{} (+{:.0f} s): {} static/keyframed bodies in the "
                     "world vs {} in the reference ({}), world offset reference-now ({:.1f}, "
                     "{:.1f}, {:.1f}) | MISSING now {} (reference positions): {} | only now {}: {}",
                     job.map, job.visit, job.since, cur.size(), ref.size(), ref_header, off[0],
                     off[1], off[2], missing.size(), ListSnap(missing, 40), extra.size(),
                     ListSnap(extra, 20));
        } else {
            LOG_INFO(Core,
                     "[objcol] map {} visit #{} (+{:.0f} s): {} static/keyframed bodies in the "
                     "world; first reference written",
                     job.map, job.visit, job.since, cur.size());
        }
        if (!have_ref || cur.size() > ref.size()) {
            WriteSnap(SnapPath("ref", job.map), cur, header);
        }
    } catch (...) {
    }
}

// ---------------------------------------------------------------- object physics systems
// Decompiled (1.09):
//   0xb36780 hknpPhysicsSystem::addToWorld(system, mode, flags): collects the system's body ids
//            ([system+0x20], count [system+0x28]) that are not yet added and calls
//            addBodies([system+0x18] world, ids, n, mode, flags).
//   0xb368e0 hknpPhysicsSystem::removeFromWorld(system).
//   The game's physics-system wrapper (vtable 0x52978a0, system at wrapper+0x18, "in world" byte
//   at wrapper+0x40): slot +0x78 0xa7f2d0 = addToWorld(mode 1, flags 0) and wrapper+0x40 = 1;
//   slot +0x80 0xa7f300 = removeFromWorld and wrapper+0x40 = 0; slot +0x38 0xa7ed70 rebuilds
//   the system and re-adds it only when wrapper+0x40 is already set.

struct SysRec {
    u32 adds = 0, removes = 0, rebuilds = 0;
    bool in_world = false;
    u32 n = 0, flags = 0;
    float pos[3] = {};
    double t = 0.0;
    std::string map;
};
std::unordered_map<u64, SysRec> g_systems;  // by hknpPhysicsSystem
std::unordered_map<u64, SysRec> g_wrappers; // by the game's wrapper (rebuild calls)
u64 st_sys_add = 0, st_sys_add_mode = 0, st_sys_remove = 0, st_rebuild = 0;

void SysInfo(u64 sys, SysRec& r) {
    u8 h[0x30];
    if (!SafeCopy(h, sys, sizeof(h))) {
        return;
    }
    const u64 world = Le64(h + 0x18), ids = Le64(h + 0x20);
    r.n = Le32(h + 0x28);
    u32 id0 = 0;
    u64 data;
    s32 size;
    BodyBytes bb;
    if (r.n == 0 || r.n > 4096 || !Rd(ids, id0) || !BodyArray(world, data, size) ||
        id0 >= u32(size) || !SafeCopy(bb.b, data + u64(id0) * 0x90, 0x90)) {
        return;
    }
    r.flags = bb.Flags();
    for (int k = 0; k < 3; ++k) {
        r.pos[k] = bb.Pos(k);
    }
}

void OnSysAdd(u64 sys, s32 mode) {
    std::scoped_lock lk{g_mutex};
    auto& r = g_systems[sys];
    ++r.adds;
    ++st_sys_add;
    st_sys_add_mode += mode != 1;
    r.in_world = true;
    r.t = Now();
    r.map = g_map;
    SysInfo(sys, r);
}

void OnSysRemove(u64 sys) {
    std::scoped_lock lk{g_mutex};
    auto& r = g_systems[sys];
    ++r.removes;
    ++st_sys_remove;
    r.in_world = false;
    r.t = Now();
    r.map = g_map;
    SysInfo(sys, r);
}

void OnRebuild(u64 wrapper) {
    std::scoped_lock lk{g_mutex};
    auto& r = g_wrappers[wrapper];
    ++r.rebuilds;
    ++st_rebuild;
    r.t = Now();
    r.map = g_map;
}

std::string SysText(const SysRec& r) {
    return fmt::format("({:.1f}, {:.1f}, {:.1f}) {} bodies {} [{}]", r.pos[0], r.pos[1], r.pos[2],
                       r.n, Motion(r.flags), r.map);
}

// The game's wrappers seen rebuilding a system whose "in world" byte is still 0, and systems
// removed and not added again.
void SystemsReport(double now) {
    std::vector<std::string> removed, idle;
    u32 in_world = 0;
    for (const auto& [sys, r] : g_systems) {
        if (r.in_world) {
            ++in_world;
        } else {
            removed.push_back(SysText(r) + fmt::format(" removed {:.0f} s ago", now - r.t));
        }
    }
    for (auto& [wrapper, r] : g_wrappers) {
        u8 flag = 1;
        u64 sys = 0;
        if (!Rd(wrapper + 0x40, flag) || !Rd(wrapper + 0x18, sys) || flag != 0 || sys == 0) {
            continue;
        }
        const auto it = g_systems.find(sys);
        if (it != g_systems.end() && it->second.in_world) {
            continue;
        }
        SysInfo(sys, r);
        idle.push_back(SysText(r) + fmt::format(" (rebuilt {} times, never added)", r.rebuilds));
    }
    LOG_INFO(Core,
             "[objcol] map {} visit #{} (+{:.0f} s): physics systems addToWorld {} (mode != 1: "
             "{}), removeFromWorld {}, in the world now {} | removed and not added again {}: {} "
             "| game wrappers rebuilt but not in the world {}: {}",
             g_map, g_map_visit, now - g_map_open, st_sys_add, st_sys_add_mode, st_sys_remove,
             in_world, removed.size(), Join(removed, 30), idle.size(), Join(idle, 30));
}

// ---------------------------------------------------------------- BB_OBJCOL_FIX
// The world step (0xb597d0, timer "StAddBodies") commits the pending (mode 2) adds of
// [world+0xb8]/[world+0xc0] and [world+0xc8]/[world+0xd0] unconditionally (field run 3: always
// committed), so pending adds are only watched and logged here.

bool IsPtr(u64 v);
void FixPass(double now, u64 world);

struct PendingWatch {
    u32 c0 = 0, d0 = 0;
    double since = -1.0;
    bool logged = false;
};
std::unordered_map<u64, PendingWatch> g_pending_watch;
u64 st_flush = 0;

void OnFlush(u64 fd4) {
    const double now = Now();
    u64 world = 0;
    u32 counts[5];
    if (!Rd(fd4 + 8, world) || !IsPtr(world)) {
        return;
    }
    if (SafeCopy(counts, world + 0xc0, sizeof(counts))) {
        const u32 c0 = counts[0], d0 = counts[4];
        std::scoped_lock lk{g_mutex};
        ++st_flush;
        auto& pw = g_pending_watch[world];
        if (c0 == 0 && d0 == 0) {
            pw = PendingWatch{};
        } else if (pw.since < 0.0 || pw.c0 != c0 || pw.d0 != d0) {
            pw = PendingWatch{c0, d0, now, false};
        } else if (now - pw.since >= 2.0 && !pw.logged) {
            pw.logged = true;
            LOG_INFO(Core, "[objcol] world {:#x}: {} + {} pending body adds unchanged for {:.1f} s",
                     world, c0, d0, now - pw.since);
        }
    }
    if (g_fix) {
        FixPass(now, world);
    }
}

// ---------------------------------------------------------------- object physics wrappers
// Decompiled (1.09): every object physics system is built by the game's wrapper (ctor 0xa7d4a0,
// vtable 0x52978a0, system at +0x18, "in world" byte at +0x40) through 0x6e8700 ->
// hknpPhysicsSystem ctor 0xb359c0 with r8 = r9 = 0 (all four call sites 0xa7d592, 0xa7dcae,
// 0xa7efcd, 0x180c374: "xor r8d,r8d; xor r9d,r9d"). r8 is the addition mode the ctor passes to
// hknpWorld::createBody 0xb564a0 (0xb35e4a "mov edx,[rbp-0x11c]"), and addBodies with mode 0
// returns at once: the bodies are created outside the world. Only the wrapper's slot +0x78
// (0xa7f2d0: addToWorld(sys, 1, 0); [w+0x40] = 1) puts them in; slot +0x80 (0xa7f300) takes them
// out and clears +0x40; the rebuild slot +0x38 (0xa7ed70) re-adds only when +0x40 is set.

struct Wrapper {
    double since;
    bool fixed;
};
std::unordered_map<u64, Wrapper> g_wrap; // under g_mutex
std::recursive_mutex g_wrap_call;        // held while the game's addToWorld runs on a wrapper
u64 g_add_wrap = 0, g_wrap_vt = 0, g_sys_vt = 0;
std::unordered_map<u64, double> g_fix_pass; // by world, under g_mutex
std::map<std::string, u32> g_fixed_by_map;

void OnWrapCtor(u64 w) {
    std::scoped_lock lk{g_mutex};
    g_wrap[w] = Wrapper{Now(), false};
}

void OnWrapDtor(u64 w) {
    std::scoped_lock call{g_wrap_call};
    std::scoped_lock lk{g_mutex};
    g_wrap.erase(w);
}

// MSB object parts against the bodies in the world: the MSB-to-world offset is voted from the
// parts and the in-world static / keyframed bodies (0.5 m bins), then every mod-only part gets
// its nearest body (any motion, any state) and the vanilla parts' match rate is the baseline.
const char* StateName(const World* w, u32 id, u64 shape);

void PartsReport(double now) {
    const auto pit = g_msb_parts.find(g_map);
    if (pit == g_msb_parts.end() || pit->second.empty()) {
        return;
    }
    const auto& parts = pit->second;
    struct B {
        float p[3];
        u32 id, flags, filter, layer;
        const char* state;
    };
    std::vector<B> bodies;
    for (auto& [world, w] : g_worlds) {
        u64 data;
        s32 size;
        if (now - w.last_call > 300.0 || !BodyArray(world, data, size)) {
            continue;
        }
        const s32 n = std::min(size, 1 << 16);
        std::vector<u8> all(size_t(n) * 0x90);
        if (!SafeCopy(all.data(), data, all.size())) {
            continue;
        }
        for (s32 id = 0; id < n; ++id) {
            BodyBytes bb;
            std::memcpy(bb.b, all.data() + size_t(id) * 0x90, 0x90);
            if ((bb.Flags() & 7) == 0 || bb.Shape() == 0) {
                continue;
            }
            bodies.push_back({{bb.Pos(0), bb.Pos(1), bb.Pos(2)},
                              u32(id),
                              bb.Flags(),
                              bb.Filter(),
                              bb.Layer(),
                              StateName(&w, u32(id), bb.Shape())});
        }
    }
    std::map<std::tuple<int, int, int>, int> votes;
    size_t used = 0;
    for (const auto& p : parts) {
        if (p.mod_only || used >= 400) {
            continue;
        }
        ++used;
        for (const auto& b : bodies) {
            if ((b.flags & 5) == 0 || std::strcmp(b.state, "in-world") != 0) {
                continue;
            }
            ++votes[{int(std::lround((b.p[0] - p.pos[0]) * 2.f)),
                     int(std::lround((b.p[1] - p.pos[1]) * 2.f)),
                     int(std::lround((b.p[2] - p.pos[2]) * 2.f))}];
        }
    }
    float off[3] = {};
    int top = 0;
    for (const auto& [k, v] : votes) {
        if (v > top) {
            top = v;
            off[0] = std::get<0>(k) * 0.5f;
            off[1] = std::get<1>(k) * 0.5f;
            off[2] = std::get<2>(k) * 0.5f;
        }
    }
    const auto nearest = [&](const MsbPart& p, float& d) -> const B* {
        const B* best = nullptr;
        d = 1e30f;
        for (const auto& b : bodies) {
            float e = 0.f;
            for (int k = 0; k < 3; ++k) {
                const float t = b.p[k] - p.pos[k] - off[k];
                e += t * t;
            }
            if (e < d) {
                d = e;
                best = &b;
            }
        }
        d = std::sqrt(d);
        return best;
    };
    u32 vanilla = 0, vanilla_hit = 0, vanilla_hit_world = 0;
    std::string mods;
    u32 mod_n = 0, mod_hit_world = 0;
    for (const auto& p : parts) {
        float d;
        const B* b = nearest(p, d);
        const bool hit = b != nullptr && d < 0.6f;
        const bool in_world = hit && std::strcmp(b->state, "in-world") == 0;
        if (!p.mod_only) {
            ++vanilla;
            vanilla_hit += hit;
            vanilla_hit_world += in_world;
            continue;
        }
        ++mod_n;
        mod_hit_world += in_world;
        if (mod_n > 60) {
            continue;
        }
        mods += fmt::format(
            "{}{} ({:.1f}, {:.1f}, {:.1f}): {}", mods.empty() ? "" : "; ", p.name, p.pos[0],
            p.pos[1], p.pos[2],
            b == nullptr ? std::string("no body")
                         : fmt::format("nearest body #{} {} {} layer {} filter {:#x} at {:.2f} m",
                                       b->id, Motion(b->flags), b->state, b->layer, b->filter, d));
    }
    LOG_INFO(Core,
             "[objcol] map {} visit #{} (+{:.0f} s): MSB object parts {} | MSB->world offset "
             "({:.1f}, {:.1f}, {:.1f}) ({} votes) | parts in the vanilla MSB with a body within "
             "0.6 m: {} of {} ({} in the world) | mod-only parts {} ({} with an in-world body "
             "within 0.6 m; added by BB_OBJCOL_FIX so far {}): {}",
             g_map, g_map_visit, now - g_map_open, parts.size(), off[0], off[1], off[2], top,
             vanilla_hit, vanilla, vanilla_hit_world, mod_n, mod_hit_world,
             g_fixed_by_map.contains(g_map) ? g_fixed_by_map[g_map] : 0u,
             mods.empty() ? "none"
             : mod_n > 60 ? mods + " (more not listed)"
                          : mods);
}

// BB_OBJCOL_FIX: for the parts of a served MSB that the vanilla MSB of the same map does not
// have (found per map from the two files, no list of names), an intact object physics system
// (static or keyframed bodies, at most 8; fragment systems are dynamic) that the game built
// but never added, never removed, whose wrapper's "in world" byte is 0, at the part's position
// (MSB->world offset voted from the vanilla parts) is added the way a map reload adds it: the
// game's own wrapper addToWorld 0xa7f2d0, on the game thread, at the entry of the physics command
// flush of the system's own world (the flush adds bodies to that world itself at +0x242). Only
// wrappers whose bodies were all never added (or added with mode 0) qualify, and the offset vote
// runs only when such a wrapper exists, for the maps opened in the last 90 s.
// Vanilla parts are never touched; a mod object that was added and later broken or removed is
// never re-added.
void FixPass(double now, u64 world) {
    if (g_add_wrap == 0) {
        return;
    }
    std::vector<std::pair<u64, std::string>> todo;
    {
        std::scoped_lock lk{g_mutex};
        double& last = g_fix_pass[world];
        if (now - last < 5.0) {
            return;
        }
        last = now;
        std::set<std::string> maps;
        for (const auto& pc : g_pending_cmp) {
            if (now - pc.open < 180.0) {
                maps.insert(pc.map);
            }
        }
        if (maps.empty() || g_wrap.empty()) {
            return;
        }
        // Wrappers of this flush's world that were built, never added and never removed (system
        // and every body), intact (static / keyframed, at most 8 bodies).
        struct Cand {
            u64 w;
            SysRec r;
            Wrapper* rec;
        };
        std::vector<Cand> cands;
        const auto wit = g_worlds.find(world);
        u64 data = 0;
        s32 size = 0;
        if (!BodyArray(world, data, size)) {
            return;
        }
        for (auto& [w, rec] : g_wrap) {
            if (rec.fixed || now - rec.since < 3.0) {
                continue;
            }
            u64 vt = 0, sys = 0, svt = 0;
            u8 in = 1;
            if (!Rd(w, vt) || vt != g_wrap_vt || !Rd(w + 0x40, in) || in != 0 ||
                !Rd(w + 0x18, sys) || !IsPtr(sys) || !Rd(sys, svt) || svt != g_sys_vt) {
                continue;
            }
            u8 h[0x30];
            if (!SafeCopy(h, sys, sizeof(h)) || Le64(h + 0x18) != world) {
                continue;
            }
            if (const auto it = g_systems.find(sys);
                it != g_systems.end() && (it->second.adds || it->second.removes)) {
                continue;
            }
            SysRec r;
            SysInfo(sys, r);
            if (r.n == 0 || r.n > 8 || (r.flags & 5) == 0) {
                continue;
            }
            u32 ids[8];
            bool fresh = SafeCopy(ids, Le64(h + 0x20), r.n * 4);
            for (u32 k = 0; k < r.n && fresh; ++k) {
                BodyBytes bb;
                fresh = ids[k] < u32(size) && SafeCopy(bb.b, data + u64(ids[k]) * 0x90, 0x90);
                if (fresh && wit != g_worlds.end()) {
                    const char* st = StateName(&wit->second, ids[k], bb.Shape());
                    fresh =
                        std::strcmp(st, "never-added") == 0 || std::strcmp(st, "add-ignored") == 0;
                }
            }
            if (fresh) {
                cands.push_back({w, r, &rec});
            }
        }
        if (cands.empty()) {
            return;
        }
        const auto inworld = Snapshot();
        for (const auto& [map, parts] : g_msb_parts) {
            if (!maps.contains(map)) {
                continue;
            }
            std::vector<const MsbPart*> mods;
            for (const auto& p : parts) {
                if (p.mod_only) {
                    mods.push_back(&p);
                }
            }
            if (mods.empty()) {
                continue;
            }
            std::unordered_map<u64, int> votes;
            size_t used = 0;
            for (const auto& p : parts) {
                if (p.mod_only || used >= 400) {
                    continue;
                }
                ++used;
                for (const auto& b : inworld) {
                    u64 key = 0;
                    for (int k = 0; k < 3; ++k) {
                        key = (key << 21) |
                              (u64(u32(std::lround((b.p[k] - p.pos[k]) * 2.f)) + (1u << 20)) &
                               0x1fffff);
                    }
                    ++votes[key];
                }
            }
            float off[3] = {};
            int top = 0;
            for (const auto& [k, v] : votes) {
                if (v > top) {
                    top = v;
                    for (int i = 0; i < 3; ++i) {
                        off[i] = float(s64((k >> (42 - 21 * i)) & 0x1fffff) - (1 << 20)) * 0.5f;
                    }
                }
            }
            if (top < 20) {
                continue; // offset not established
            }
            for (auto& c : cands) {
                if (c.rec->fixed) {
                    continue;
                }
                for (const MsbPart* p : mods) {
                    float d2 = 0.f;
                    for (int k = 0; k < 3; ++k) {
                        const float t = c.r.pos[k] - p->pos[k] - off[k];
                        d2 += t * t;
                    }
                    if (d2 < 0.36f) {
                        todo.emplace_back(c.w, fmt::format("{} {} at ({:.1f}, {:.1f}, {:.1f}) {} "
                                                           "bodies {}",
                                                           map, p->name, c.r.pos[0], c.r.pos[1],
                                                           c.r.pos[2], c.r.n, Motion(c.r.flags)));
                        c.rec->fixed = true;
                        break;
                    }
                }
            }
        }
    }
    if (todo.empty()) {
        return;
    }
    using AddFn = void(PS4_SYSV_ABI*)(u64);
    u32 done = 0;
    std::string list;
    for (const auto& [w, what] : todo) {
        std::scoped_lock call{g_wrap_call};
        {
            std::scoped_lock lk{g_mutex};
            u64 vt = 0;
            u8 in = 1;
            if (!g_wrap.contains(w) || !Rd(w, vt) || vt != g_wrap_vt || !Rd(w + 0x40, in) ||
                in != 0) {
                continue;
            }
            ++g_fixed_by_map[what.substr(0, what.find(' '))];
        }
        reinterpret_cast<AddFn>(g_add_wrap)(w);
        ++done;
        list += (list.empty() ? "" : "; ") + what;
    }
    LOG_INFO(Core,
             "[objcol-fix] {} mod-added object(s) whose collision the game built but never added "
             "now added with the game's wrapper addToWorld eboot+{:#x}: {}",
             done, g_add_wrap - g_eboot, list.empty() ? "none" : list);
}

void MapTick(double now) {
    if (now - g_map_tick < 1.0) {
        return;
    }
    g_map_tick = now;
    for (auto& [world, w] : g_worlds) {
        u64 data;
        s32 size;
        const bool arr = BodyArray(world, data, size);
        for (auto& [id, d] : w.deferred) {
            if (now - d.t < 5.0) {
                continue;
            }
            if (d.logged || g_deferred_lines >= 100) {
                continue;
            }
            d.logged = true;
            ++g_deferred_lines;
            BodyBytes bb{};
            const bool ok = arr && id < u32(size) && SafeCopy(bb.b, data + u64(id) * 0x90, 0x90);
            LOG_INFO(
                Core,
                "[objcol] deferred add of body {} ({} layer {} filter {:#x} at ({:.1f}, "
                "{:.1f}, {:.1f})) queued by eboot+{:#x} at {:.2f} s not committed after {:.0f} "
                "s (map {}, character steps {})",
                id, ok ? Motion(bb.Flags()) : "?", ok ? bb.Layer() : 0, ok ? bb.Filter() : 0,
                ok ? bb.Pos(0) : 0.f, ok ? bb.Pos(1) : 0.f, ok ? bb.Pos(2) : 0.f, d.ret - g_eboot,
                d.t, now - d.t, g_map, g_first_step < 0.0 ? "not started" : "running");
        }
    }
    // Every map open gets its +30 s and +90 s reports, also when another map block opened since
    // (Yharnam opens m24_01 and m27_00 within seconds of each other).
    for (auto& pc : g_pending_cmp) {
        for (int stage = 0; stage < 2; ++stage) {
            const double at = stage == 0 ? 30.0 : 90.0;
            if (pc.done[stage] || now - pc.open < at) {
                continue;
            }
            pc.done[stage] = true;
            const std::string map = g_map;
            const int visit = g_map_visit;
            const double open = g_map_open;
            g_map = pc.map;
            g_map_visit = pc.visit;
            g_map_open = pc.open;
            CompareMap(now);
            SystemsReport(now);
            PartsReport(now);
            g_map = map;
            g_map_visit = visit;
            g_map_open = open;
        }
    }
    std::erase_if(g_pending_cmp, [](const PendingCmp& pc) { return pc.done[0] && pc.done[1]; });
}

std::set<std::string> g_paths;
u32 g_path_lines = 0;

// The game's functions keep rbp frames (push rbp; mov rbp,rsp): from addBodies' entry, the
// return address and then [rbp+8] of each caller frame give the call path.
std::string ShapeVt(u64 shape) {
    u64 vt = 0;
    if (!IsPtr(shape) || !Rd(shape, vt)) {
        return "?";
    }
    return vt >= g_eboot && vt < g_eboot + 0x6000000 ? fmt::format("eboot+{:#x}", vt - g_eboot)
                                                     : fmt::format("{:#x}", vt);
}

// body+0x88 holds the game's owner of the body (0x1e3fc70 stores "param_1 + 0x20" there):
// its address and the vtable of the object it points into.
std::string OwnerText(u64 owner) {
    u64 vt = 0;
    if (!IsPtr(owner) || !Rd(owner, vt)) {
        return owner ? fmt::format("{:#x}", owner) : std::string("none");
    }
    return fmt::format("{:#x} vtable {}", owner,
                       vt >= g_eboot && vt < g_eboot + 0x6000000
                           ? fmt::format("eboot+{:#x}", vt - g_eboot)
                           : fmt::format("{:#x}", vt));
}

std::string CallPath(u64 ret, u64 caller_rbp) {
    std::string s = fmt::format("eboot+{:#x}", ret - g_eboot);
    u64 fp = caller_rbp;
    for (int k = 0; k < 6 && IsPtr(fp); ++k) {
        u64 frame[2];
        if (!SafeCopy(frame, fp, sizeof(frame)) || frame[1] < g_eboot ||
            frame[1] >= g_eboot + 0x5000000) {
            break;
        }
        s += fmt::format(" <- eboot+{:#x}", frame[1] - g_eboot);
        if (frame[0] <= fp) {
            break;
        }
        fp = frame[0];
    }
    return s;
}

void OnAdd(u64 world, u64 ids_ptr, s32 n, s32 mode, u64 ret, u64 caller_rbp) {
    if (n <= 0 || n > (1 << 16)) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    if (!g_worlds.contains(world) && g_world_lines < 64) {
        ++g_world_lines;
        LOG_INFO(Core,
                 "[objcol] world {:#x} first addBodies at {:.2f} s (mode {}, {} bodies, by "
                 "eboot+{:#x}), map {}, character steps {}",
                 world, Now(), mode, n, ret - g_eboot, g_map,
                 g_first_step < 0.0 ? std::string("not started")
                                    : fmt::format("since {:.2f} s", g_first_step));
    }
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
    std::string path; // walked once per call, only while add path lines are still logged
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
        if ((b.flags & 5) && g_path_lines < 120) {
            if (path.empty()) {
                path = CallPath(ret, caller_rbp);
            }
            const std::string key = fmt::format("{} {:#x} {}", Motion(b.flags), bb.Filter(), path);
            if (g_paths.insert(key).second) {
                ++g_path_lines;
                LOG_INFO(Core,
                         "[objcol] add path: {} body #{} filter {:#x} layer {} shape vtable {} "
                         "at ({:.1f}, {:.1f}, {:.1f}), mode {}, owner (body+0x88) {}, map {}: {}",
                         Motion(b.flags), id, bb.Filter(), bb.Layer(), ShapeVt(shape), bb.Pos(0),
                         bb.Pos(1), bb.Pos(2), mode, OwnerText(Le64(bb.b + 0x88)), g_map, path);
            }
        }
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
            ++st_deferred_bodies;
            if (g_first_step < 0.0) {
                ++st_deferred_before_step;
            }
            w.deferred.try_emplace(id, World::Pending{Now(), ret, false});
            continue;
        }
        if (w.deferred.erase(id) != 0) {
            ++st_committed_bodies;
        }
        if (g_first_step < 0.0) {
            ++st_added_before_step;
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
    MapTick(Now());
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
        if (w.deferred.erase(id) != 0) {
            ++st_deferred_removed;
        }
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

// No exception may unwind into the guest frames.
void GuardedAdd(const u64* regs, u64 caller_rbp) {
    try {
        OnAdd(regs[5], regs[6], s32(regs[7]), s32(regs[8]), regs[10], caller_rbp);
    } catch (...) {
    }
}

void GuardedRemove(const u64* regs) {
    try {
        OnRemove(regs[5], regs[6], s32(regs[7]), regs[10]);
    } catch (...) {
    }
}

void PS4_SYSV_ABI HookAdd(const u64* regs, u64 rbp) {
    // regs: rbx r11 r10 r9 r8 rdi rsi rdx rcx rax, return address; rbp: the caller's frame
    GuardedAdd(regs, rbp);
}

void PS4_SYSV_ABI HookRemove(const u64* regs) {
    GuardedRemove(regs);
}

// ---------------------------------------------------------------- player proximity
//   0xb2f010 hknpCharacterProxy::integrate(proxy, ...) (Havok timer "LtCharacterProxyIntegrate"):
//            proxy+0x30 body id, +0x34 collision filter info of its queries, +0x38 world,
//            +0x40 position; its shape casts use the filter object [[world+0x150]+0x5e8].
//   0x553e878 WorldChrMan (1.09, checked against the code that names it); [WorldChrMan+0x60]
//            is its most used member (109 of ~600 sampled uses) and is taken as the local
//            player: the player's proxy is found from it by a bounded pointer search.

constexpr u64 WorldChrManVaddr = 0x553e878;
constexpr u64 WorldChrManUse = 0x1222ee3; // lea r15,[rip+WorldChrMan]; ...; lea rcx,"WorldChrMan"

u64 g_wcm = 0;
std::unordered_map<u64, double> g_proxies;
std::vector<u32> g_player_path;
u64 g_player_proxy = 0;
double g_player_tick = 0.0, g_player_search = -100.0, g_player_line = -100.0, g_proxy_list = -100.0;
float g_last_pos[3] = {1e9f, 1e9f, 1e9f};
std::string g_last_near;
bool g_proxy_logged = false;
u32 g_player_lines = 0;
constexpr u32 MaxPlayerLines = 6000;
u64 st_proxy_calls = 0;
u32 g_search_fails = 0;

bool IsPtr(u64 v) {
    return v > 0x10000 && v < (1ull << 47) && (v & 7) == 0;
}

u64 Follow(u64 root, const std::vector<u32>& path) {
    u64 p = root;
    for (const u32 off : path) {
        u64 v;
        if (!Rd(p + off, v) || !IsPtr(v)) {
            return 0;
        }
        p = v;
    }
    return p;
}

// Breadth first from root (first level [lo, hi), then 0x400 per object, depth <= 4): the
// shallowest pointer to a live character proxy.
bool SearchPath(u64 root, u32 lo, u32 hi, std::vector<u32>& path, u64& found) {
    struct Node {
        u64 addr;
        std::vector<u32> path;
    };
    std::deque<Node> q;
    std::unordered_set<u64> seen{root};
    q.push_back({root, {}});
    constexpr u32 Budget = 12000;
    u32 queued = 1;
    std::vector<u8> buf(hi - lo > 0x400 ? hi - lo : 0x400);
    for (u32 budget = Budget; !q.empty() && budget > 0; --budget) {
        Node node = std::move(q.front());
        q.pop_front();
        const u32 first = node.path.empty() ? lo : 0;
        u32 len = node.path.empty() ? hi - lo : 0x400;
        while (len >= 0x40 && !SafeCopy(buf.data(), node.addr + first, len)) {
            len /= 2;
        }
        if (len < 0x40) {
            continue;
        }
        for (u32 off = 0; off + 8 <= len; off += 8) {
            const u64 v = Le64(buf.data() + off);
            if (!IsPtr(v)) {
                continue;
            }
            if (g_proxies.contains(v)) {
                path = node.path;
                path.push_back(first + off);
                found = v;
                return true;
            }
            // Breadth first: a node queued past the budget would never be read.
            if (node.path.size() < 3 && queued < Budget && seen.insert(v).second) {
                ++queued;
                auto next = node.path;
                next.push_back(first + off);
                q.push_back({v, std::move(next)});
            }
        }
    }
    return false;
}

struct ProxyBytes {
    u8 b[0x50];
    s32 BodyId() const {
        return s32(Le32(b + 0x30));
    }
    u32 Filter() const {
        return Le32(b + 0x34);
    }
    u64 World() const {
        return Le64(b + 0x38);
    }
    float Pos(int i) const {
        float f;
        std::memcpy(&f, b + 0x40 + i * 4, 4);
        return f;
    }
};

const char* StateName(const World* w, u32 id, u64 shape) {
    if (w == nullptr) {
        return "untracked";
    }
    const auto it = w->bodies.find(id);
    if (it == w->bodies.end() || it->second.shape != shape) {
        return "never-added";
    }
    switch (it->second.state) {
    case State::InWorld:
        return "in-world";
    case State::Removed:
        return "removed";
    case State::Deferred:
        return "deferred";
    case State::Ignored:
        return "add-ignored";
    default:
        return "never-added";
    }
}

void LogPlayerProxy(u64 proxy, const ProxyBytes& pb) {
    std::string body = "none";
    const u64 world = pb.World();
    u64 data;
    s32 size;
    if (pb.BodyId() >= 0 && BodyArray(world, data, size) && pb.BodyId() < size) {
        BodyBytes bb;
        if (SafeCopy(bb.b, data + u64(pb.BodyId()) * 0x90, 0x90)) {
            body =
                fmt::format("#{} {} flags {:#x} layer {} filter {:#x} at ({:.1f}, {:.1f}, {:.1f})",
                            pb.BodyId(), Motion(bb.Flags()), bb.Flags(), bb.Layer(), bb.Filter(),
                            bb.Pos(0), bb.Pos(1), bb.Pos(2));
        }
    }
    std::string filter = "n/a";
    u64 mgr = 0, fo = 0, vt = 0;
    u32 words[64];
    if (Rd(world + 0x150, mgr) && IsPtr(mgr) && Rd(mgr + 0x5e8, fo) && IsPtr(fo) && Rd(fo, vt) &&
        SafeCopy(words, fo, sizeof(words))) {
        filter = fmt::format("{:#x} vtable eboot+{:#x}:", fo, vt - g_eboot);
        for (int i = 0; i < 64; ++i) {
            filter += fmt::format(" {:08x}", words[i]);
        }
    }
    std::string path;
    for (const u32 off : g_player_path) {
        path += fmt::format(" +{:#x}", off);
    }
    LOG_INFO(Core,
             "[objcol] player proxy {:#x} (path from [WorldChrMan+0x60]:{}): query filter "
             "info {:#x} (layer {}), world {:#x}, position ({:.1f}, {:.1f}, {:.1f}), own body {} | "
             "world collision filter {}",
             proxy, path, pb.Filter(), pb.Filter() & 0x1f, world, pb.Pos(0), pb.Pos(1), pb.Pos(2),
             body, filter);
}

void NearPlayer(const ProxyBytes& pb, double now) {
    const u64 world = pb.World();
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
    float q[12] = {};
    const bool have_q =
        SafeCopy(q, world + 0x180, sizeof(q)) && q[8] > 0.f && q[9] > 0.f && q[10] > 0.f;
    const auto wit = g_worlds.find(world);
    const World* w = wit == g_worlds.end() ? nullptr : &wit->second;
    // Field run 2: proxy+0x40 reads (1, 0, 0) for every character; the position is the
    // translation of the proxy's own body (+0x30).
    float P[3] = {pb.Pos(0), pb.Pos(1), pb.Pos(2)};
    if (pb.BodyId() >= 0 && pb.BodyId() < n) {
        BodyBytes own;
        std::memcpy(own.b, all.data() + size_t(pb.BodyId()) * 0x90, 0x90);
        for (int k = 0; k < 3; ++k) {
            P[k] = own.Pos(k);
        }
    }
    struct Near {
        float d;
        std::string text;
        std::string key;
    };
    std::vector<Near> nearby;
    bool inside_any = false;
    for (s32 id = 0; id < n; ++id) {
        BodyBytes bb;
        std::memcpy(bb.b, all.data() + size_t(id) * 0x90, 0x90);
        const u32 flags = bb.Flags();
        const u64 shape = bb.Shape();
        if ((flags & 3) == 0 || shape == 0 || id == pb.BodyId()) {
            continue;
        }
        const char* state = StateName(w, u32(id), shape);
        float mn[3], mx[3];
        bool box = false;
        if ((flags & 1) && have_q && std::strcmp(state, "in-world") == 0) {
            s16 qa[8];
            std::memcpy(qa, bb.b + 0x50, sizeof(qa));
            box = true;
            for (int k = 0; k < 3; ++k) {
                mn[k] = float(qa[k]) / q[8 + k] - q[k];
                mx[k] = float(qa[k + 4]) / q[8 + k] - q[k];
                box = box && mn[k] <= mx[k];
            }
        }
        float d2 = 0.f;
        for (int k = 0; k < 3; ++k) {
            const float c = box ? std::clamp(P[k], mn[k], mx[k]) : bb.Pos(k);
            d2 += (P[k] - c) * (P[k] - c);
        }
        const float d = std::sqrt(d2);
        if (!(d <= (box ? 3.f : 4.f))) { // NaN positions too
            continue;
        }
        const bool inside = box && d == 0.f;
        inside_any = inside_any || inside;
        const std::string size_text = box ? fmt::format(" box {:.1f}x{:.1f}x{:.1f}", mx[0] - mn[0],
                                                        mx[1] - mn[1], mx[2] - mn[2])
                                          : " (origin)";
        nearby.push_back(
            {d,
             fmt::format("#{} {} layer {} filter {:#x} {}{} d {:.1f}{}", id, Motion(flags),
                         bb.Layer(), bb.Filter(), state, size_text, d, inside ? " INSIDE" : ""),
             fmt::format("{}:{}:{}", id, state, inside)});
    }
    std::sort(nearby.begin(), nearby.end(), [](const Near& a, const Near& b) { return a.d < b.d; });
    if (nearby.size() > 10) {
        nearby.resize(10);
    }
    std::string key, list;
    for (const auto& e : nearby) {
        key += e.key + ";";
        list += (list.empty() ? "" : " | ") + e.text;
    }
    float moved = 0.f;
    for (int k = 0; k < 3; ++k) {
        moved += (P[k] - g_last_pos[k]) * (P[k] - g_last_pos[k]);
    }
    moved = std::sqrt(moved);
    if (key == g_last_near && !(moved > 1.f && now - g_player_line >= 5.0)) {
        return;
    }
    if (g_player_lines >= MaxPlayerLines) {
        return;
    }
    if (++g_player_lines == MaxPlayerLines) {
        LOG_INFO(Core, "[objcol] player: line cap ({}) reached, no further player lines",
                 MaxPlayerLines);
    }
    g_last_near = key;
    g_player_line = now;
    std::copy(P, P + 3, g_last_pos);
    LOG_INFO(Core,
             "[objcol] player ({:.1f}, {:.1f}, {:.1f}) query filter {:#x}{}: bodies within 3 m: {}",
             P[0], P[1], P[2], pb.Filter(), inside_any ? " [inside a static box]" : "",
             list.empty() ? "none" : list);
}

void PlayerTick(double now) {
    for (auto it = g_proxies.begin(); it != g_proxies.end();) {
        it = now - it->second > 2.0 ? g_proxies.erase(it) : std::next(it);
    }
    u64 pins = 0;
    if (g_wcm != 0) {
        u64 wcm, v;
        if (Rd(g_wcm, wcm) && IsPtr(wcm) && Rd(wcm + 0x60, v) && IsPtr(v)) {
            pins = v;
        }
    }
    u64 proxy = 0;
    if (pins != 0 && !g_player_path.empty()) {
        proxy = Follow(pins, g_player_path);
        if (!g_proxies.contains(proxy)) {
            proxy = 0;
        }
    }
    if (proxy == 0 && pins != 0 && now - g_player_search >= (g_search_fails < 3 ? 5.0 : 60.0)) {
        g_player_search = now;
        ++g_search_fails;
        std::vector<u32> path;
        u64 found = 0;
        if (SearchPath(pins, 0, 0x2000, path, found)) {
            g_player_path = path;
            proxy = found;
            g_search_fails = 0;
            g_proxy_logged = false;
        }
    }
    if (proxy == 0) {
        if (now - g_proxy_list >= 10.0 && g_player_lines < MaxPlayerLines) {
            ++g_player_lines;
            g_proxy_list = now;
            std::string list;
            int k = 0;
            for (const auto& [pp, t] : g_proxies) {
                ProxyBytes pb;
                if (k++ < 8 && SafeCopy(pb.b, pp, sizeof(pb.b))) {
                    list += fmt::format(" ({:.1f}, {:.1f}, {:.1f}) filter {:#x};", pb.Pos(0),
                                        pb.Pos(1), pb.Pos(2), pb.Filter());
                }
            }
            LOG_INFO(Core,
                     "[objcol] player proxy not found (WorldChrMan {}, PlayerIns {:#x}); "
                     "character proxies {} (integrate calls {}):{}",
                     g_wcm ? "ok" : "not validated", pins, g_proxies.size(), st_proxy_calls,
                     list.empty() ? " none" : list);
        }
        return;
    }
    ProxyBytes pb;
    if (!SafeCopy(pb.b, proxy, sizeof(pb.b))) {
        return;
    }
    g_player_proxy = proxy;
    if (!g_proxy_logged) {
        g_proxy_logged = true;
        LogPlayerProxy(proxy, pb);
    }
    NearPlayer(pb, now);
}

void OnProxy(u64 proxy) {
    if (!IsPtr(proxy)) {
        return;
    }
    // Never waits: a physics thread skips this call while the census or a search holds the lock.
    std::unique_lock lk{g_mutex, std::try_to_lock};
    if (!lk.owns_lock()) {
        return;
    }
    ++st_proxy_calls;
    const double now = Now();
    if (g_first_step < 0.0) {
        g_first_step = now;
        LOG_INFO(Core,
                 "[objcol] first character step at {:.2f} s (map {}): before it {} bodies added "
                 "immediately, {} queued deferred",
                 now, g_map, st_added_before_step, st_deferred_before_step);
    }
    g_proxies[proxy] = now;
    MapTick(now);
    if (now - g_player_tick < 0.5) {
        return;
    }
    g_player_tick = now;
    PlayerTick(now);
}

void GuardedProxy(const u64* regs) {
    try {
        OnProxy(regs[5]); // rdi: the proxy
    } catch (...) {
    }
}

void PS4_SYSV_ABI HookProxy(const u64* regs) {
    GuardedProxy(regs);
}

void PS4_SYSV_ABI HookSysAdd(const u64* regs) {
    try {
        OnSysAdd(regs[5], s32(regs[6]));
    } catch (...) {
    }
}

void PS4_SYSV_ABI HookSysRemove(const u64* regs) {
    try {
        OnSysRemove(regs[5]);
    } catch (...) {
    }
}

void PS4_SYSV_ABI HookRebuild(const u64* regs) {
    try {
        OnRebuild(regs[5]);
    } catch (...) {
    }
}

void PS4_SYSV_ABI HookWrapCtor(const u64* regs) {
    try {
        OnWrapCtor(regs[5]);
    } catch (...) {
    }
}

void PS4_SYSV_ABI HookWrapDtor(const u64* regs) {
    try {
        OnWrapDtor(regs[5]);
    } catch (...) {
    }
}

void PS4_SYSV_ABI HookFlush(const u64* regs) {
    try {
        OnFlush(regs[5]);
    } catch (...) {
    }
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

// hknpCharacterProxy::integrate: prologue (20 bytes relocated) and the argument moves up to the
// stack guard load (the Windows TLS patch may rewrite what follows).
constexpr int SigProxy[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54,
                            0x53, 0x48, 0x81, 0xec, 0x38, 0x03, 0x00, 0x00, 0x4c, 0x89, 0xc3, 0x49,
                            0x89, 0xce, 0x4c, 0x89, 0xb5, 0x80, 0xfd, 0xff, 0xff, 0x48, 0x89, 0x95,
                            0x20, 0xfd, 0xff, 0xff, 0x49, 0x89, 0xf7, 0x4c, 0x89, 0xbd, 0x18, 0xfd,
                            0xff, 0xff, 0x49, 0x89, 0xfc, 0x48, 0x8b, 0x05, W,    W,    W,    W,
                            0x48, 0x8b, 0x00, 0x48, 0x89, 0x45, 0xd0};
constexpr u32 ProxyPrologue = 20;

// hknpPhysicsSystem::addToWorld (14 relocated) / removeFromWorld (18 relocated).
constexpr int SigSysAdd[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                             0x41, 0x54, 0x53, 0x50, 0x41, 0x89, 0xd7, 0x89, 0x75, 0xd4,
                             0x49, 0x89, 0xfd, 0x45, 0x8b, 0x75, 0x28, 0x31, 0xdb, 0x41,
                             0xbc, 0x00, 0x00, 0x00, 0x00, 0x45, 0x85, 0xf6, 0x0f, 0x84};
constexpr u32 SysAddPrologue = 14;
constexpr int SigSysRemove[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                0x41, 0x54, 0x53, 0x49, 0x89, 0xff, 0x45, 0x8b,
                                0x67, 0x28, 0x31, 0xdb, 0x45, 0x85, 0xe4, 0x74};
constexpr u32 SysRemovePrologue = 18;
// The game's physics-system wrapper: rebuild (slot +0x38, 17 relocated), addToWorld (+0x78),
// removeFromWorld (+0x80); vtable 0x52978a0 in 1.09.
constexpr int SigRebuild[] = {
    0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec,
    0x38, 0xc5, 0xfa, 0x11, 0x45, 0xac, 0x49, 0x89, 0xfd, 0x48, 0x8b, 0x05, W,    W,    W,    W,
    0x48, 0x8b, 0x00, 0x48, 0x89, 0x45, 0xd0, 0x48, 0xc7, 0x45, 0xc0, 0x00, 0x00, 0x00, 0x00};
constexpr u32 RebuildPrologue = 17;
constexpr int SigAddWrap[] = {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb,
                              0x48, 0x8b, 0x7b, 0x18, 0xbe, 0x01, 0x00, 0x00, 0x00,
                              0x31, 0xd2, 0xb9, 0x01, 0x00, 0x00, 0x00, 0xe8, W,
                              W,    W,    W,    0xc6, 0x43, 0x40, 0x01};
constexpr int SigRemoveWrap[] = {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b,
                                 0x7b, 0x18, 0xe8, W,    W,    W,    W,    0xc6, 0x43, 0x40, 0x00};
constexpr u64 WrapperVtable = 0x52978a0;
constexpr u64 SystemVtable = 0x5261c40; // the game's hknpPhysicsSystem subclass (0x6e8700)
// Wrapper ctor 0xa7d4a0 (15 relocated; lea of the vtable at +30) and dtor 0xa7d930 (14
// relocated; lea of the vtable at +14).
constexpr int SigWrapCtor[] = {
    0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x10,
    0x49, 0x89, 0xd6, 0x49, 0x89, 0xfc, 0x41, 0xc7, 0x44, 0x24, 0x08, 0x01, 0x00, 0xff, 0xff,
    0x48, 0x8d, 0x05, W,    W,    W,    W,    0x48, 0x83, 0xc0, 0x10, 0x49, 0x89, 0x04, 0x24};
constexpr u32 WrapCtorPrologue = 15;
constexpr int SigWrapDtor[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53,
                               0x49, 0x89, 0xfe, 0x48, 0x8d, 0x05, W,    W,    W,    W,    0x48,
                               0x83, 0xc0, 0x10, 0x49, 0x89, 0x06, 0x49, 0x8b, 0x7e, 0x18};
constexpr u32 WrapDtorPrologue = 14;
// The game's physics command flush (20 relocated); its addBodies call at +0x242.
constexpr int SigFlush[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54,
                            0x53, 0x48, 0x81, 0xec, 0xe8, 0x00, 0x00, 0x00, 0x49, 0x89, 0xfc, 0x48,
                            0x8b, 0x1d, W,    W,    W,    W,    0x48, 0x8b, 0x03, 0x48, 0x89, 0x45,
                            0xd0, 0x49, 0x83, 0x7c, 0x24, 0x08, 0x00, 0x0f, 0x84, W,    W,    W,
                            W,    0x48, 0xc7, 0x85, 0xf8, 0xfe, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00};
constexpr u32 FlushPrologue = 20, FlushAddCall = 0x242;
s64 CallTarget(const u8* img, u64 at) {
    if (img[at] != 0xe8) {
        return -1;
    }
    s32 d;
    std::memcpy(&d, img + at + 1, 4);
    return s64(at) + 5 + d;
}

// lea rax,[rip+d] within [from, to) of the image whose target is the given string.
bool LeaToString(const u8* img, u64 size, u64 from, u64 to, const char* str) {
    const size_t k = std::strlen(str) + 1;
    for (u64 i = from; i + 7 <= to && i + 7 <= size; ++i) {
        if (img[i] != 0x48 || img[i + 1] != 0x8d || (img[i + 2] & 0xc7) != 0x05) {
            continue;
        }
        s32 d;
        std::memcpy(&d, img + i + 3, 4);
        const s64 t = s64(i) + 7 + d;
        if (t > 0 && u64(t) + k <= size && std::memcmp(img + t, str, k) == 0) {
            return true;
        }
    }
    return false;
}

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
std::vector<u8> BuildCave(u64 host, const u8* prologue, u32 len, u64 back, bool pass_rbp = false) {
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
    if (pass_rbp) {
        put({0x48, 0x89, 0xee}); // mov rsi, rbp (the caller's frame at a function entry)
    }
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

namespace {
void FileOpened(const std::string& p, const char* host, u64 size) {
    if (host != nullptr && std::strstr(host, "-mods") != nullptr) {
        std::scoped_lock lk{g_mutex};
        if (g_mod_lines < 400 && g_mod_logged.insert(p).second) {
            ++g_mod_lines;
            LOG_INFO(Core, "[objcol] mod file {} served from {} ({} bytes)", p, host, size);
        }
    }
    if (p.find(".hkxbhd") != std::string::npos || p.find(".hkxbdt") != std::string::npos) {
        LOG_INFO(Core, "[objcol] collision file {} served from {} ({} bytes)", p, host ? host : "?",
                 size);
        return;
    }
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
        g_map_open = Now();
        g_cmp30 = g_cmp90 = false;
        g_map_visit = ++g_visits[g_map];
        g_pending_cmp.push_back({g_map, g_map_visit, g_map_open, {false, false}});
        u64 pending = 0, in_world = 0;
        for (const auto& [world, w] : g_worlds) {
            pending += w.deferred.size();
            for (const auto& [id, b] : w.bodies) {
                in_world += b.state == State::InWorld;
            }
        }
        LOG_INFO(Core,
                 "[objcol] map {} loading (visit #{} this session, {:.1f} s after boot, character "
                 "steps {}; worlds {}, bodies in the world {}, deferred adds pending {}; deferred "
                 "so far {} queued / {} committed / {} removed before commit) (hooks {}), layout "
                 "served from {} ({})",
                 g_map, g_map_visit, g_map_open,
                 g_first_step < 0.0 ? std::string("not started")
                                    : fmt::format("since {:.1f} s", g_first_step),
                 g_worlds.size(), in_world, pending, st_deferred_bodies, st_committed_bodies,
                 st_deferred_removed, g_hooked.load() ? "on" : "off", host ? host : "?",
                 SourceOf(host ? host : ""));
        if (host != nullptr) {
            std::scoped_lock jl{g_job_mutex};
            g_jobs.push_back({p, -1, host});
            if (!g_worker_started) {
                g_worker_started = true;
                std::thread(Worker).detach();
            }
            g_job_cv.notify_one();
        }
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
    g_jobs.push_back({p, model, {}});
    if (!g_worker_started) {
        g_worker_started = true;
        std::thread(Worker).detach();
    }
    g_job_cv.notify_one();
}
} // namespace

void OnGuestFileOpen(const char* guest_path, const char* host_path, u64 size) {
    if (!g_enabled || guest_path == nullptr) {
        return;
    }
    try {
        FileOpened(guest_path, host_path, size);
    } catch (...) {
    }
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
                              reinterpret_cast<u64>(at_add + AddPrologue), true);
    const auto c2 = BuildCave(reinterpret_cast<u64>(&HookRemove), at_rem, RemovePrologue,
                              reinterpret_cast<u64>(at_rem + RemovePrologue));
    std::memcpy(cave, c1.data(), c1.size());
    u8* cave2 = cave + ((c1.size() + 15) & ~size_t(15));
    std::memcpy(cave2, c2.data(), c2.size());
    Detour(at_add, cave);
    Detour(at_rem, cave2);
    g_hooked = true;

    // Player proximity: the character proxy integrate entry and the WorldChrMan global.
    std::string player = "character proxy hook off";
    const auto prx = Find(img, code, SigProxy, std::size(SigProxy));
    if (prx &&
        LeaToString(img, image_size, *prx + 0x40, *prx + 0xa0, "LtCharacterProxyIntegrate")) {
        u8* at_prx = reinterpret_cast<u8*>(eboot_base + *prx);
        const auto c3 = BuildCave(reinterpret_cast<u64>(&HookProxy), at_prx, ProxyPrologue,
                                  reinterpret_cast<u64>(at_prx + ProxyPrologue));
        u8* cave3 = cave2 + ((c2.size() + 15) & ~size_t(15));
        std::memcpy(cave3, c3.data(), c3.size());
        Detour(at_prx, cave3);
        player = fmt::format("hknpCharacterProxy::integrate eboot+{:#x} hooked", *prx);
    } else {
        player = prx ? "hknpCharacterProxy::integrate timer string differs: not hooked"
                     : "hknpCharacterProxy::integrate not found (unique): not hooked";
    }
    const u64 use = WorldChrManUse;
    if (use + 0x30 < image_size && img[use] == 0x4c && img[use + 1] == 0x8d &&
        img[use + 2] == 0x3d) {
        s32 d;
        std::memcpy(&d, img + use + 3, 4);
        if (u64(s64(use) + 7 + d) == WorldChrManVaddr &&
            LeaToString(img, image_size, use, use + 0x30, "WorldChrMan")) {
            g_wcm = eboot_base + WorldChrManVaddr;
        }
    }
    LOG_INFO(Core,
             "[objcol] on: hknpWorld::addBodies eboot+{:#x}, removeBodies eboot+{:#x} hooked; {}; "
             "WorldChrMan eboot+{:#x} {}; objbnd analysis on a worker thread; lines '[objcol] "
             "map' (per map), '[objcol] player' (position and bodies within 3 m)",
             *add, *rem, player, WorldChrManVaddr,
             g_wcm ? "validated" : "not validated (no player position)");

    // Object physics systems, the game's wrapper and its command flush (every relocated byte and
    // every call target compared first; any mismatch: that part is not hooked).
    u8* cave_b = static_cast<u8*>(AllocExec(4096));
    u8* next = cave_b;
    const auto place = [&](const std::vector<u8>& c) {
        u8* at = next;
        std::memcpy(at, c.data(), c.size());
        next += (c.size() + 15) & ~size_t(15);
        return at;
    };
    const auto hook = [&](u64 off, u64 host, u32 len) {
        u8* at = reinterpret_cast<u8*>(eboot_base + off);
        Detour(at, place(BuildCave(host, at, len, reinterpret_cast<u64>(at + len))));
    };
    const auto sys_add = Find(img, code, SigSysAdd, std::size(SigSysAdd));
    const auto sys_rem = Find(img, code, SigSysRemove, std::size(SigSysRemove));
    const auto rebuild = Find(img, code, SigRebuild, std::size(SigRebuild));
    const auto add_wrap = Find(img, code, SigAddWrap, std::size(SigAddWrap));
    const auto rem_wrap = Find(img, code, SigRemoveWrap, std::size(SigRemoveWrap));
    const auto flush = Find(img, code, SigFlush, std::size(SigFlush));
    // Called before the relocations: a slot holds 0 (RELA), the offset or the address.
    const auto slot_is = [&](u32 off, std::optional<u64> fn) {
        u64 v = 1;
        if (fn && WrapperVtable + off + 8 <= image_size) {
            std::memcpy(&v, img + WrapperVtable + off, 8);
        }
        return fn && (v == 0 || v == *fn || v == eboot_base + *fn);
    };
    std::string objs;
    const bool sys_ok = cave_b && sys_add && sys_rem && rebuild && add_wrap && rem_wrap &&
                        CallTarget(img, *add_wrap + 25) == s64(*sys_add) &&
                        CallTarget(img, *rem_wrap + 13) == s64(*sys_rem) &&
                        slot_is(0x38, rebuild) && slot_is(0x78, add_wrap) &&
                        slot_is(0x80, rem_wrap);
    if (sys_ok) {
        hook(*sys_add, reinterpret_cast<u64>(&HookSysAdd), SysAddPrologue);
        hook(*sys_rem, reinterpret_cast<u64>(&HookSysRemove), SysRemovePrologue);
        hook(*rebuild, reinterpret_cast<u64>(&HookRebuild), RebuildPrologue);
        objs = fmt::format("hknpPhysicsSystem::addToWorld eboot+{:#x} / removeFromWorld "
                           "eboot+{:#x}, wrapper rebuild eboot+{:#x} hooked",
                           *sys_add, *sys_rem, *rebuild);
    } else {
        objs = "physics-system hooks off (signature, call target or vtable check failed)";
    }
    const bool flush_ok = cave_b && flush && CallTarget(img, *flush + FlushAddCall) == s64(*add);
    if (flush_ok) {
        hook(*flush, reinterpret_cast<u64>(&HookFlush), FlushPrologue);
    }
    // Wrapper ctor (sets the vtable 0x5297890 + 0x10) and dtor (0xa7d930, slot 0's callee).
    const auto wctor = Find(img, code, SigWrapCtor, std::size(SigWrapCtor));
    const auto wdtor = Find(img, code, SigWrapDtor, std::size(SigWrapDtor));
    const auto lea_vt = [&](std::optional<u64> f, u32 at) {
        if (!f || img[*f + at] != 0x48 || img[*f + at + 1] != 0x8d) {
            return false;
        }
        s32 d;
        std::memcpy(&d, img + *f + at + 3, 4);
        return s64(*f) + at + 7 + d == s64(WrapperVtable - 0x10);
    };
    const bool fix_ok = sys_ok && flush_ok && lea_vt(wctor, 30) && lea_vt(wdtor, 14) &&
                        slot_is(0x00, wdtor) && CallTarget(img, *add_wrap + 25) == s64(*sys_add);
    if (fix_ok) {
        hook(*wctor, reinterpret_cast<u64>(&HookWrapCtor), WrapCtorPrologue);
        hook(*wdtor, reinterpret_cast<u64>(&HookWrapDtor), WrapDtorPrologue);
        g_add_wrap = eboot_base + *add_wrap;
        g_wrap_vt = eboot_base + WrapperVtable;
        g_sys_vt = eboot_base + SystemVtable;
    }
    LOG_INFO(Core, "[objcol] {}; physics command flush {}; wrapper ctor/dtor {}; BB_OBJCOL_FIX {}",
             objs, flush_ok ? fmt::format("eboot+{:#x} hooked", *flush) : std::string("not hooked"),
             fix_ok ? fmt::format("eboot+{:#x} / eboot+{:#x} hooked", *wctor, *wdtor)
                    : std::string("checks failed"),
             g_fix ? (fix_ok ? "on (mod-added objects' built-but-never-added collision is added "
                               "with the game's wrapper addToWorld)"
                             : "requested but a check failed: off")
                   : "off");
}

} // namespace BB::ObjCol
