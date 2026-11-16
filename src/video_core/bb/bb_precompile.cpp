// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/bb/bb_precompile.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <miniz.h>

#include "common/logging/log.h"
#include "common/path_util.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace BBPrecompile {

namespace {

constexpr u32 FileMagic = 0x43534242; // "BBSC"
constexpr u32 FileFormat = 1;
constexpr const char* ShippedName = "bb-shadercache.bin";

struct Section {
    std::vector<u8> profile;
    CacheVersions versions{};
    std::map<std::string, std::vector<u8>> entries; // file name -> contents (lazy-decoded)
    std::vector<u8> packed;                         // compressed entries as read from disk
    u64 raw_size{};
    u32 num_entries{};
    bool decoded{};
};

// State remembered from WarmUp, used by the export on close.
struct Context {
    std::mutex m;
    bool active{};
    std::filesystem::path cache_dir;
    std::vector<u8> profile;
    CacheVersions versions{};
    ProfileMatcher matches;
    bool shipped_needs_update{};
};

Context& Ctx() {
    static Context c;
    return c;
}

bool EnvIs1(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] == '1';
}

std::filesystem::path ExeDir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH * 4]{};
    const DWORD n = GetModuleFileNameW(nullptr, buf, static_cast<DWORD>(std::size(buf)));
    if (n > 0 && n < std::size(buf)) {
        return std::filesystem::path{buf}.parent_path();
    }
#else
    std::error_code ec;
    auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        return p.parent_path();
    }
#endif
    return std::filesystem::current_path();
}

// Where the shipped cache is read from / written to.
std::filesystem::path ShippedPath() {
    if (const char* v = std::getenv("BB_SHADERCACHE"); v && v[0]) {
        return std::filesystem::path{v};
    }
    return ExeDir() / ShippedName;
}

// Reader helpers over a byte buffer.
struct Cursor {
    std::span<const u8> d;
    size_t pos{};
    bool ok{true};
    template <typename T>
    T Get() {
        T v{};
        if (pos + sizeof(T) > d.size()) {
            ok = false;
            return v;
        }
        std::memcpy(&v, d.data() + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }
    std::span<const u8> Bytes(size_t n) {
        if (pos + n > d.size()) {
            ok = false;
            return {};
        }
        auto s = d.subspan(pos, n);
        pos += n;
        return s;
    }
};

template <typename T>
void Put(std::vector<u8>& out, const T& v) {
    const auto* p = reinterpret_cast<const u8*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

std::vector<u8> ReadWholeFile(const std::filesystem::path& path) {
    std::ifstream f{path, std::ios::binary | std::ios::ate};
    if (!f) {
        return {};
    }
    const auto size = static_cast<size_t>(f.tellg());
    std::vector<u8> data(size);
    f.seekg(0);
    if (size && !f.read(reinterpret_cast<char*>(data.data()), size)) {
        return {};
    }
    return data;
}

bool ParseFile(std::span<const u8> file, std::vector<Section>& sections) {
    Cursor c{file};
    if (c.Get<u32>() != FileMagic || c.Get<u32>() != FileFormat) {
        return false;
    }
    const u32 count = c.Get<u32>();
    for (u32 i = 0; i < count && c.ok; ++i) {
        Section s{};
        const u32 psize = c.Get<u32>();
        auto prof = c.Bytes(psize);
        s.profile.assign(prof.begin(), prof.end());
        s.versions.shader_binary = c.Get<u32>();
        s.versions.shader_meta = c.Get<u32>();
        s.versions.pipeline_key = c.Get<u32>();
        s.num_entries = c.Get<u32>();
        s.raw_size = c.Get<u64>();
        const u64 packed_size = c.Get<u64>();
        auto packed = c.Bytes(static_cast<size_t>(packed_size));
        s.packed.assign(packed.begin(), packed.end());
        if (c.ok) {
            sections.push_back(std::move(s));
        }
    }
    return c.ok;
}

bool SafeName(const std::string& name) {
    return !name.empty() && name.size() < 256 && name.find_first_of("/\\:") == std::string::npos &&
           name.find("..") == std::string::npos;
}

bool Decode(Section& s) {
    if (s.decoded) {
        return true;
    }
    std::vector<u8> raw(static_cast<size_t>(s.raw_size));
    mz_ulong len = static_cast<mz_ulong>(raw.size());
    if (mz_uncompress(raw.data(), &len, s.packed.data(), static_cast<mz_ulong>(s.packed.size())) !=
            MZ_OK ||
        len != raw.size()) {
        return false;
    }
    Cursor c{raw};
    for (u32 i = 0; i < s.num_entries && c.ok; ++i) {
        const u16 nlen = c.Get<u16>();
        auto n = c.Bytes(nlen);
        std::string name{reinterpret_cast<const char*>(n.data()), n.size()};
        const u32 size = c.Get<u32>();
        auto data = c.Bytes(size);
        if (c.ok && SafeName(name)) {
            s.entries[std::move(name)] = std::vector<u8>(data.begin(), data.end());
        }
    }
    if (!c.ok) {
        return false;
    }
    s.decoded = true;
    s.packed.clear();
    return true;
}

bool Encode(Section& s) {
    std::vector<u8> raw;
    for (const auto& [name, data] : s.entries) {
        Put(raw, static_cast<u16>(name.size()));
        raw.insert(raw.end(), name.begin(), name.end());
        Put(raw, static_cast<u32>(data.size()));
        raw.insert(raw.end(), data.begin(), data.end());
    }
    s.raw_size = raw.size();
    s.num_entries = static_cast<u32>(s.entries.size());
    mz_ulong bound = mz_compressBound(static_cast<mz_ulong>(raw.size()));
    s.packed.resize(bound);
    if (mz_compress2(s.packed.data(), &bound, raw.data(), static_cast<mz_ulong>(raw.size()),
                     MZ_BEST_SPEED) != MZ_OK) {
        return false;
    }
    s.packed.resize(bound);
    return true;
}

bool IsCacheFile(const std::filesystem::path& p) {
    const auto ext = p.extension().string();
    return ext == ".meta" || ext == ".spv" || ext == ".key";
}

} // namespace

bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_PRECOMPILE");
        return !(v && v[0] == '0');
    }();
    return on;
}

u32 NumThreads() {
    if (const char* v = std::getenv("BB_PRECOMPILE_THREADS"); v && v[0]) {
        return std::clamp<u32>(static_cast<u32>(std::atoi(v)), 1u, 16u);
    }
    const u32 hc = std::max(2u, std::thread::hardware_concurrency());
    return std::clamp<u32>(hc / 2, 1u, 8u);
}

bool WriteFileAtomic(const std::filesystem::path& path, std::span<const u8> data,
                     const char* tmp_suffix) {
    auto tmp = path;
    tmp += tmp_suffix;
    {
        std::ofstream f{tmp, std::ios::binary | std::ios::trunc};
        if (!f || !f.write(reinterpret_cast<const char*>(data.data()), data.size())) {
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

u32 ImportShipped(const std::filesystem::path& cache_dir, std::span<const u8> profile,
                  const CacheVersions& versions, const ProfileMatcher& matches) {
    auto& ctx = Ctx();
    {
        std::scoped_lock lk{ctx.m};
        ctx.active = true;
        ctx.cache_dir = cache_dir;
        ctx.profile.assign(profile.begin(), profile.end());
        ctx.versions = versions;
        ctx.matches = matches;
        ctx.shipped_needs_update = false;
    }

    const auto path = ShippedPath();
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        LOG_INFO(Render, "[precompile] no shipped cache at {} (BB_SHADERCACHE_EXPORT=1 makes one)",
                 path.string());
        return 0;
    }
    std::vector<Section> sections;
    if (!ParseFile(ReadWholeFile(path), sections)) {
        LOG_WARNING(Render, "[precompile] shipped cache {} is corrupt, ignored (will be rewritten)",
                    path.string());
        std::scoped_lock lk{ctx.m};
        ctx.shipped_needs_update = true;
        return 0;
    }
    Section* match = nullptr;
    for (auto& s : sections) {
        if (matches(s.profile)) {
            match = &s;
            break;
        }
    }
    if (!match) {
        LOG_INFO(Render,
                 "[precompile] shipped cache has {} section(s), none for this GPU/driver profile; "
                 "this run's cache will be added to it on exit",
                 sections.size());
        std::scoped_lock lk{ctx.m};
        ctx.shipped_needs_update = true;
        return 0;
    }
    if (!(match->versions == versions)) {
        LOG_WARNING(Render,
                    "[precompile] shipped cache is from another build (spv {} meta {} key {}, now "
                    "{} {} {}): ignored; it is regenerated from this run's cache on exit",
                    match->versions.shader_binary, match->versions.shader_meta,
                    match->versions.pipeline_key, versions.shader_binary, versions.shader_meta,
                    versions.pipeline_key);
        std::scoped_lock lk{ctx.m};
        ctx.shipped_needs_update = true;
        return 0;
    }
    if (!Decode(*match)) {
        LOG_WARNING(Render, "[precompile] shipped cache section failed to decode, ignored");
        std::scoped_lock lk{ctx.m};
        ctx.shipped_needs_update = true;
        return 0;
    }
    u32 imported = 0;
    for (const auto& [name, data] : match->entries) {
        const auto dst = cache_dir / name;
        if (std::filesystem::exists(dst, ec)) {
            continue;
        }
        std::ofstream f{dst, std::ios::binary | std::ios::trunc};
        if (f && f.write(reinterpret_cast<const char*>(data.data()), data.size())) {
            ++imported;
        }
    }
    LOG_INFO(Render, "[precompile] shipped cache {}: {} entries, {} imported into {}",
             path.filename().string(), match->entries.size(), imported, cache_dir.string());
    return imported;
}

void OnCacheClosed() {
    auto& ctx = Ctx();
    std::scoped_lock lk{ctx.m};
    if (!ctx.active || !Enabled()) {
        return;
    }
    ctx.active = false;
    const bool forced = EnvIs1("BB_SHADERCACHE_EXPORT");
    if (!forced && !ctx.shipped_needs_update) {
        return;
    }
    const auto path = ShippedPath();

    std::vector<Section> sections;
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
        if (!ParseFile(ReadWholeFile(path), sections)) {
            sections.clear();
        }
    }
    // Keep other GPU profiles' sections; merge into ours (same versions) or replace it.
    Section* mine = nullptr;
    for (auto& s : sections) {
        if (ctx.matches && ctx.matches(s.profile)) {
            mine = &s;
            break;
        }
    }
    if (mine && !(mine->versions == ctx.versions && Decode(*mine))) {
        mine->entries.clear();
        mine->decoded = true;
    }
    if (!mine) {
        mine = &sections.emplace_back();
        mine->decoded = true;
    }
    mine->profile = ctx.profile;
    mine->versions = ctx.versions;

    u32 local = 0;
    for (const auto& entry : std::filesystem::directory_iterator{ctx.cache_dir, ec}) {
        if (!entry.is_regular_file() || !IsCacheFile(entry.path())) {
            continue;
        }
        auto data = ReadWholeFile(entry.path());
        if (data.empty()) {
            continue;
        }
        // Leave out blobs from older cache versions still lying in the user cache.
        const auto ext = entry.path().extension().string();
        u32 head[2]{};
        std::memcpy(head, data.data(), std::min<size_t>(data.size(), sizeof(head)));
        if ((ext == ".meta" && (head[0] != ctx.versions.shader_meta ||
                                head[1] != ctx.versions.shader_binary)) ||
            (ext == ".key" && head[0] != ctx.versions.pipeline_key)) {
            continue;
        }
        mine->entries[entry.path().filename().string()] = std::move(data);
        ++local;
    }
    if (mine->entries.empty()) {
        LOG_INFO(Render, "[precompile] nothing to export");
        return;
    }
    std::vector<u8> out;
    Put(out, FileMagic);
    Put(out, FileFormat);
    Put(out, static_cast<u32>(sections.size()));
    for (auto& s : sections) {
        if (s.decoded && !Encode(s)) {
            LOG_WARNING(Render, "[precompile] export: compression failed");
            return;
        }
        Put(out, static_cast<u32>(s.profile.size()));
        out.insert(out.end(), s.profile.begin(), s.profile.end());
        Put(out, s.versions.shader_binary);
        Put(out, s.versions.shader_meta);
        Put(out, s.versions.pipeline_key);
        Put(out, s.num_entries);
        Put(out, s.raw_size);
        Put(out, static_cast<u64>(s.packed.size()));
        out.insert(out.end(), s.packed.begin(), s.packed.end());
    }
    if (!WriteFileAtomic(path, out, ".tmp")) {
        LOG_WARNING(Render, "[precompile] could not write {} (read-only folder?)", path.string());
        return;
    }
    LOG_INFO(Render,
             "[precompile] exported {}: {} entries for this GPU profile ({} from this PC), {} "
             "section(s), {} KiB",
             path.string(), mine->entries.size(), local, sections.size(), out.size() / 1024);
}

} // namespace BBPrecompile
