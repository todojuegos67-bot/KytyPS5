// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bb-gpu-prof (bb_gpu_profile.h): GPU timestamps per pass, BB_GPU_PROFILE=1.

#include "video_core/bb/bb_gpu_profile.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "core/bb_game_gfx.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace BB::GpuProf {

namespace {

constexpr u32 kPairs = 16384;           // scopes in flight (2 queries each)
constexpr u32 kQueries = kPairs * 2;
constexpr u32 kBegunTimeout = 32;       // frames: a scope never ended is dropped
constexpr u32 kEndedTimeout = 240;      // frames: a scope never executed (buffer discarded)
constexpr u32 kTopLines = 3;            // lines of the sorted list
constexpr u32 kPerLine = 10;            // labels per line

struct LabelInfo {
    std::string name;
    Kind kind;
    u32 group;
};

struct Acc {
    u64 ns = 0;
    u64 n = 0;
    u64 max_ns = 0; // the longest single scope (spikes, e.g. once-a-second statistics passes)
};

struct Rec {
    u32 label = 0;
    u32 frame = 0;
    u8 state = 0; // 0 free, 1 begun, 2 ended
};

struct State {
    std::mutex m;
    bool tried = false;
    bool ok = false;
    vk::Device device{};
    vk::QueryPool pool{};
    double ns_per_tick = 1.0;
    u64 mask = ~0ull;
    std::vector<u32> family_bits; // timestampValidBits per queue family
    std::vector<LabelInfo> labels{{"(none)", Kind::Pass, 0}};
    std::unordered_map<std::string, u32> by_name;
    std::unordered_map<u64, u32> by_key;
    std::vector<Acc> acc{Acc{}};
    std::vector<Rec> recs = std::vector<Rec>(kPairs);
    u64 head = 0; // next record (monotonic; ring index = head % kPairs)
    u64 tail = 0; // oldest unresolved record
    u32 frame = 0;
    u64 sec_frames = 0, sec_dropped = 0, sec_lost = 0, sec_scopes = 0, sec_count = 0;
    std::chrono::steady_clock::time_point sec_start{};
    std::vector<u64> results = std::vector<u64>(kQueries * 2);
};

State& S() {
    static State s;
    return s;
}

u32 AddLabel(State& s, std::string name, Kind kind, u32 group) {
    const u32 id = static_cast<u32>(s.labels.size());
    s.labels.push_back({std::move(name), kind, group});
    s.acc.push_back({});
    return id;
}

void Resolve(State& s) {
    if (!s.ok || s.tail == s.head) return;
    const u64 pending = s.head - s.tail;
    // Read every pending pair with availability (one or two calls around the ring's end).
    const u32 first = static_cast<u32>(s.tail % kPairs);
    const u32 n1 = static_cast<u32>(std::min<u64>(pending, kPairs - first));
    const u32 n2 = static_cast<u32>(pending - n1);
    const auto flags = vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability;
    const auto read = [&](u32 pair, u32 count, u64* out) {
        // eNotReady is expected (scopes still in flight): availability tells which are done.
        (void)s.device.getQueryPoolResults(s.pool, pair * 2, count * 2,
                                           count * 2 * 2 * sizeof(u64), out, 2 * sizeof(u64),
                                           flags);
    };
    std::fill_n(s.results.begin(), pending * 4, 0ull);
    read(first, n1, s.results.data());
    if (n2 > 0) read(0, n2, s.results.data() + u64(n1) * 4);
    u64 done = 0;
    for (; done < pending; ++done) {
        Rec& r = s.recs[(s.tail + done) % kPairs];
        const u64* q = s.results.data() + done * 4; // t0, avail0, t1, avail1
        const u32 age = s.frame - r.frame;
        if (r.state == 2 && q[1] != 0 && q[3] != 0) {
            const u64 dt = (q[2] - q[0]) & s.mask;
            // A begin stamp after the end one cannot happen in order; take it as lost.
            if (dt < (s.mask >> 1)) {
                auto& a = s.acc[r.label];
                const u64 ns = static_cast<u64>(double(dt) * s.ns_per_tick);
                a.ns += ns;
                ++a.n;
                a.max_ns = std::max(a.max_ns, ns);
                if (const u32 g = s.labels[r.label].group; g != 0) {
                    s.acc[g].ns += ns;
                    ++s.acc[g].n;
                }
                ++s.sec_scopes;
            } else {
                ++s.sec_lost;
            }
        } else if ((r.state == 1 && age > kBegunTimeout) || (r.state == 2 && age > kEndedTimeout) ||
                   r.state == 0) {
            ++s.sec_lost;
        } else {
            break; // in flight: everything after it waits (results arrive nearly in order)
        }
        r.state = 0;
    }
    if (done == 0) return;
    // Host reset of the consumed queries (VK 1.2 hostQueryReset): free for the next scopes.
    const u32 r1 = static_cast<u32>(std::min<u64>(done, kPairs - first));
    s.device.resetQueryPool(s.pool, first * 2, r1 * 2);
    if (done > r1) s.device.resetQueryPool(s.pool, 0, static_cast<u32>(done - r1) * 2);
    s.tail += done;
}

std::string Ms(double v) {
    return fmt::format("{:.3f}", v);
}

void LogSecond(State& s, double el) {
    const double f = static_cast<double>(std::max<u64>(1, s.sec_frames));
    struct Item {
        u32 id;
        double ms;
        double n;
        double max_ms;
    };
    std::vector<Item> passes, groups, env;
    double timed = 0.0, busy = 0.0;
    for (u32 i = 1; i < s.labels.size(); ++i) {
        const auto& a = s.acc[i];
        if (a.n == 0) continue;
        const Item it{i, double(a.ns) / 1e6 / f, double(a.n) / f, double(a.max_ns) / 1e6};
        const auto& l = s.labels[i];
        if (l.kind == Kind::Envelope) {
            env.push_back(it);
            busy += it.ms;
        } else if (l.kind == Kind::Group) {
            groups.push_back(it); // summed from its passes
        } else {
            passes.push_back(it);
            timed += it.ms;
        }
    }
    const auto by_cost = [](const Item& a, const Item& b) { return a.ms > b.ms; };
    std::sort(passes.begin(), passes.end(), by_cost);
    std::sort(groups.begin(), groups.end(), by_cost);
    std::sort(env.begin(), env.end(), by_cost);
    std::string e;
    for (const auto& it : env) {
        e += fmt::format("{}{} {} ({:.1f}/f)", e.empty() ? "" : ", ", s.labels[it.id].name,
                         Ms(it.ms), it.n);
    }
    LOG_INFO(Render_Vulkan,
             "[gpu-prof] {:.1f} s, {} frames: GPU busy (submits) {} ms/frame [{}] | timed {} "
             "ms/frame in {:.0f} scopes/frame | lost {} dropped {} (BB_GPU_PROFILE={})",
             el, s.sec_frames, Ms(busy), e, Ms(timed), double(s.sec_scopes) / f, s.sec_lost,
             s.sec_dropped, Level());
    const auto list = [&](const char* title, const std::vector<Item>& items, u32 lines) {
        for (u32 line = 0; line < lines; ++line) {
            const u32 from = line * kPerLine;
            if (from >= items.size()) break;
            std::string t;
            for (u32 k = from; k < std::min<u32>(from + kPerLine, u32(items.size())); ++k) {
                const auto& it = items[k];
                t += fmt::format("{}{} {} ({:.1f}/f, max {})", t.empty() ? "" : " | ",
                                 Ms(it.ms), s.labels[it.id].name, it.n, Ms(it.max_ms));
            }
            LOG_INFO(Render_Vulkan, "[gpu-prof] {} {}-{} (ms/frame): {}", title, from + 1,
                     std::min<u32>(from + kPerLine, u32(items.size())), t);
        }
    };
    list("top", passes, kTopLines);
    list("by target", groups, 1);
    // bb-game-gfx: the game's own post-effect settings (read-only), every 10 s.
    if (++s.sec_count % 10 == 1) {
        if (const std::string g = BB::GameGfx::Describe(); !g.empty()) {
            LOG_INFO(Render_Vulkan, "{}", g);
        }
    }
    for (auto& a : s.acc) a = {};
    s.sec_frames = s.sec_dropped = s.sec_lost = s.sec_scopes = 0;
}

} // namespace

int Level() {
    static const int level = [] {
        const char* v = std::getenv("BB_GPU_PROFILE");
        if (v == nullptr || v[0] < '1' || v[0] > '9') return 0;
        return v[0] - '0';
    }();
    return level;
}

bool On() {
    static const bool on = Level() > 0;
    return on;
}

void Init(const Vulkan::Instance& instance) {
    if (!On()) return;
    auto& s = S();
    std::scoped_lock lk{s.m};
    if (s.tried) return;
    s.tried = true;
    const auto pd = instance.GetPhysicalDevice();
    const auto chain =
        pd.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features>();
    const bool host_reset = chain.get<vk::PhysicalDeviceVulkan12Features>().hostQueryReset;
    const auto props = pd.getProperties();
    const auto families = pd.getQueueFamilyProperties();
    const u32 family = instance.GetGraphicsQueueFamilyIndex();
    const u32 bits = family < families.size() ? families[family].timestampValidBits : 0;
    if (!host_reset || bits == 0 || props.limits.timestampPeriod <= 0.0f) {
        LOG_WARNING(Render_Vulkan,
                    "[gpu-prof] off: the device has no {} (BB_GPU_PROFILE needs host query reset "
                    "and graphics-queue timestamps)",
                    !host_reset ? "hostQueryReset" : "timestamps");
        return;
    }
    s.family_bits.clear();
    for (const auto& f : families) s.family_bits.push_back(f.timestampValidBits);
    s.device = instance.GetDevice();
    auto [r, pool] = s.device.createQueryPool(vk::QueryPoolCreateInfo{
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = kQueries,
    });
    if (r != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "[gpu-prof] off: query pool not created ({})",
                    vk::to_string(r));
        return;
    }
    s.pool = pool;
    s.device.resetQueryPool(s.pool, 0, kQueries);
    s.ns_per_tick = props.limits.timestampPeriod;
    s.mask = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
    s.sec_start = std::chrono::steady_clock::now();
    s.ok = true;
    LOG_INFO(Render_Vulkan,
             "[gpu-prof] on (BB_GPU_PROFILE={}): {} timestamp pairs, {:.2f} ns/tick, {} valid "
             "bits; one '[gpu-prof]' summary a second (ms per guest frame, serialized timings: "
             "the stamps add a little GPU time of their own)",
             Level(), kPairs, s.ns_per_tick, bits);
}

bool FamilyHasTimestamps(u32 family) {
    if (!On()) return false;
    auto& s = S();
    std::scoped_lock lk{s.m};
    return s.ok && family < s.family_bits.size() && s.family_bits[family] != 0;
}

u32 Label(std::string_view name, Kind kind) {
    auto& s = S();
    std::scoped_lock lk{s.m};
    std::string key(name);
    if (const auto it = s.by_name.find(key); it != s.by_name.end()) return it->second;
    const u32 id = AddLabel(s, key, kind, 0);
    s.by_name.emplace(std::move(key), id);
    return id;
}

u32 KeyLookup(u64 key) {
    auto& s = S();
    std::scoped_lock lk{s.m};
    const auto it = s.by_key.find(key);
    return it != s.by_key.end() ? it->second : 0u;
}

u32 KeyInsert(u64 key, const std::string& name, u32 group) {
    auto& s = S();
    std::scoped_lock lk{s.m};
    if (const auto it = s.by_key.find(key); it != s.by_key.end()) return it->second;
    const u32 id = AddLabel(s, name, Kind::Pass, group);
    s.by_key.emplace(key, id);
    return id;
}

u32 Begin(vk::CommandBuffer cmd, u32 label) {
    if (!On() || !cmd || label == 0) return 0;
    auto& s = S();
    std::scoped_lock lk{s.m};
    if (!s.ok) return 0;
    if (s.head - s.tail >= kPairs) {
        ++s.sec_dropped;
        return 0;
    }
    const u32 i = static_cast<u32>(s.head % kPairs);
    ++s.head;
    s.recs[i] = {label, s.frame, 1};
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, s.pool, i * 2);
    return i + 1;
}

void End(vk::CommandBuffer cmd, u32 token) {
    if (token == 0 || !cmd) return;
    auto& s = S();
    std::scoped_lock lk{s.m};
    Rec& r = s.recs[token - 1];
    if (r.state != 1) return;
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, s.pool, (token - 1) * 2 + 1);
    r.state = 2;
}

void Frame() {
    if (!On()) return;
    auto& s = S();
    std::scoped_lock lk{s.m};
    if (!s.ok) return;
    ++s.frame;
    ++s.sec_frames;
    Resolve(s);
    const auto now = std::chrono::steady_clock::now();
    const double el = std::chrono::duration<double>(now - s.sec_start).count();
    if (el >= 1.0) {
        LogSecond(s, el);
        s.sec_start = now;
    }
}

} // namespace BB::GpuProf
