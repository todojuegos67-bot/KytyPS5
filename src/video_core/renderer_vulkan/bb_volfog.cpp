// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-volfog: froxel volumetric fog (see bb_volfog.h and host_shaders/bb_volfog.comp).
//
// Point-light shader table: the GXLightAcc point-light pixel-shader hashes, their light count and
// the per-light data stride (64 bytes, 96 for the shadowed families) are taken from João Vitor
// Alves' "Bloodborne PC Path Tracing" mod (GPL-2.0, rt/rt_lights.h, docs/PATH_TRACING.md "Live
// layout"), which read them from the shipped shader bundle's reflection and a live dump: per light
// floats [0..2] view-space position, [3] 1/r^2, [4..6] diffuse colour; cbLightAccParam (208 B)
// floats 32..47 = g_CameraMatrix (view -> world, 3x4 rows). The same project documents
// cbSceneParam (the 864-byte scene block): FC_ProjSpaceToShadowMatrix[4] at byte 448,
// FC_CascadeSelectDist at 704, FC_MatrixInvView at 720 (this port's float 180, already used).
// Everything else here is this port's own code.
#include "video_core/renderer_vulkan/bb_volfog.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <tuple>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "core/memory.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/host_shaders/bb_volfog_apply_comp.h"
#include "video_core/host_shaders/bb_volfog_apply_r11_comp.h"
#include "video_core/host_shaders/bb_volfog_inject_comp.h"
#include "video_core/host_shaders/bb_volfog_integrate_comp.h"
#include "video_core/host_shaders/bb_volfog_light_comp.h"
#include "video_core/host_shaders/bb_volfog_measure_comp.h"
#include "video_core/host_shaders/bb_volfog_measure_r11_comp.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace BB {

namespace {

template <typename T>
bool Invert4(const T* m, double out[16]) {
    double a[4][8];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            a[r][c] = m[r * 4 + c];
            a[r][4 + c] = r == c ? 1.0 : 0.0;
        }
    }
    for (int c = 0; c < 4; ++c) {
        int piv = c;
        for (int r = c + 1; r < 4; ++r) {
            if (std::fabs(a[r][c]) > std::fabs(a[piv][c]))
                piv = r;
        }
        if (std::fabs(a[piv][c]) < 1e-12)
            return false;
        if (piv != c) {
            for (int j = 0; j < 8; ++j)
                std::swap(a[c][j], a[piv][j]);
        }
        const double inv = 1.0 / a[c][c];
        for (int j = 0; j < 8; ++j)
            a[c][j] *= inv;
        for (int r = 0; r < 4; ++r) {
            if (r == c)
                continue;
            const double f = a[r][c];
            for (int j = 0; j < 8; ++j)
                a[r][j] -= f * a[c][j];
        }
    }
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            out[r * 4 + c] = a[r][4 + c];
    return true;
}
// Row-major (column vectors) -> GLSL std140 mat4 (column-major).
void ToGlsl(const double* m, float* out) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            out[c * 4 + r] = static_cast<float>(m[r * 4 + c]);
}
void Mul4(const double* a, const double* b, double* out) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            double v = 0.0;
            for (int k = 0; k < 4; ++k)
                v += a[r * 4 + k] * b[k * 4 + c];
            out[r * 4 + c] = v;
        }
}
void Xform(const double* m, const double p[3], double out[4]) {
    for (int r = 0; r < 4; ++r) {
        out[r] = m[r * 4] * p[0] + m[r * 4 + 1] * p[1] + m[r * 4 + 2] * p[2] + m[r * 4 + 3];
    }
}

float EnvFloat(const char* name, float def, float lo, float hi) {
    const char* e = std::getenv(name);
    if (!e || !e[0])
        return def;
    char* end = nullptr;
    const float v = std::strtof(e, &end);
    return end != e && std::isfinite(v) ? std::clamp(v, lo, hi) : def;
}
float Lum(const float* c) {
    return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2];
}

struct alignas(16) VfParams {
    float inv_proj[16];
    float inv_view[16];
    float prev_view[16];
    float prev_proj[16];
    float shadow[4][16];
    float tile[4][4];
    float tbound[4][4];
    float cdist[4];
    u32 grid[4];
    u32 sizes[4];
    float cam[4];
    float moon_dir[4];
    float moon_col[4];
    float amb_col[4];
    float fog[4];
    float fog2[4];
    float wind[4];
    float misc[4];
    float misc2[4];
    float misc3[4];
    float gfog[4];
    float gfog_col[4];
    float expo[4];
    float layer[4][4];
};
static_assert(sizeof(VfParams) <= 1024);

struct GpuLight {
    float pos_r[4];
    float col[4];
};

// GXLightAcc point-light pixel shaders (see the file header for the source): hash, light count,
// shadowed (per-light stride 96 instead of 64). Spot / line / directional families are not used.
struct PointShader {
    u32 hash;
    u8 count;
    bool shadowed;
};
constexpr std::array<PointShader, 64> PointShaders{{
    {0x370174f7u, 1, false},  {0x600f86b0u, 2, false},  {0x89f9ee3eu, 3, false},
    {0x149432ffu, 4, false},  {0x5de04266u, 5, false},  {0x5d7150dau, 6, false},
    {0x8d5387d9u, 7, false},  {0xf81596cdu, 8, false},  {0x0b2ace4cu, 9, false},
    {0x201cb43du, 10, false}, {0x5f76eb3du, 11, false}, {0x799bfa32u, 12, false},
    {0xf40da260u, 1, true},   {0xb5b52d63u, 2, true},   {0xbb2ac1cbu, 3, true},
    {0x205c82b3u, 4, true},   {0x7cc2c044u, 1, false},  {0x186d69d5u, 2, false},
    {0xd22cbfd1u, 3, false},  {0x3105494cu, 4, false},  {0xd75fe2ffu, 5, false},
    {0x07d07cf3u, 6, false},  {0xf2531727u, 7, false},  {0xc6edb19au, 8, false},
    {0xb59b4c4eu, 9, false},  {0x505ebdbeu, 10, false}, {0x22b357f3u, 11, false},
    {0x9f9ef267u, 12, false}, {0xe96447c3u, 1, true},   {0xa22c7f71u, 2, true},
    {0x6b42fc89u, 3, true},   {0x0ef50047u, 4, true},   {0x6ab6cecbu, 1, false},
    {0xecda90f6u, 2, false},  {0xd776fb27u, 3, false},  {0xc023d70fu, 4, false},
    {0xb47e0972u, 5, false},  {0xb3c63c01u, 6, false},  {0x43d26ee3u, 7, false},
    {0xb464a1f2u, 8, false},  {0xa422d554u, 9, false},  {0x428a80bfu, 10, false},
    {0xd3a7b5a0u, 11, false}, {0xe4daeaaeu, 12, false}, {0x8aa6908eu, 1, true},
    {0x0a8a518bu, 2, true},   {0x5e210809u, 3, true},   {0x2994a496u, 4, true},
    {0x57a66dedu, 1, false},  {0xdea6db05u, 2, false},  {0xc3eb1d3au, 3, false},
    {0x42bd4804u, 4, false},  {0xba81fc0au, 5, false},  {0x3cd2ac34u, 6, false},
    {0x2aaa5cb1u, 7, false},  {0xe1610d5bu, 8, false},  {0x3aafbe57u, 9, false},
    {0xdb92e75cu, 10, false}, {0x093a99d8u, 11, false}, {0x65ff4c81u, 12, false},
    {0x98506debu, 1, true},   {0x07aa7c3eu, 2, true},   {0xea36a1b0u, 3, true},
    {0x40727870u, 4, true},
}};
const PointShader* FindPointShader(u64 hash) {
    for (const auto& s : PointShaders) {
        if (s.hash == hash)
            return &s;
    }
    return nullptr;
}

std::atomic<int> g_debug{-1};
int DebugEnv() {
    static const int v = int(EnvFloat("BB_VOLFOG_DEBUG", 0.0f, 0.0f, 5.0f));
    return v;
}

// Starting values (HIPOTESIS: not tuned against the game's look yet; every one has an override).
// haze / mist: own medium, only where the game has no fog of its own; shaft: moon light inside
// the game's fog; cap: added light <= cap x the scene's log-average luminance; lcap: point-light
// light at a froxel <= lcap x that luminance.
// high: density of the high thin band (1/m); edge: shaft edge sharpness (1: off).
struct Preset {
    const char* name;
    u32 w, h, d;
    float haze, mist, noise, scatter, lights, shaft, cap, lcap, high, edge;
};
constexpr std::array<Preset, 4> Presets{{
    {"off", 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
    {"sutil", 128, 72, 64, 0.0f, 0.0f, 0.40f, 1.0f, 0.30f, 0.6f, 0.15f, 2.0f, 0.0f, 1.0f},
    {"normal", 160, 90, 80, 0.0012f, 0.004f, 0.50f, 1.0f, 0.50f, 1.0f, 0.25f, 3.0f, 0.0006f, 1.5f},
    {"denso", 192, 108, 96, 0.0025f, 0.010f, 0.60f, 1.0f, 0.80f, 1.5f, 0.40f, 4.0f, 0.0012f, 2.0f},
}};

// FUN_02651f10's fog CB (slot 6 of the forward-lit draws; layout in bb_upscale.h): v2.x / .z / .w
// and v4.w are register copies of v0.x / .y / .w and v1.w.
bool IsFogCb(const u32* d, size_t bytes) {
    if (bytes < 84 || d[8] != d[0] || d[10] != d[1] || d[11] != d[3] || d[19] != d[7])
        return false;
    const float* f = reinterpret_cast<const float*>(d);
    if (!(std::fabs(f[3]) <= 10.001f) || f[3] == 0.0f)
        return false;
    for (int i : {0, 1, 4, 5, 6, 7, 9}) {
        if (!std::isfinite(f[i]) || (i != 9 && f[i] < 0.0f))
            return false;
    }
    return true;
}

constexpr vk::MemoryBarrier2 AllBefore{
    .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
    .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
    .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
    .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
};
constexpr vk::MemoryBarrier2 ComputeToCompute{
    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
};
bool Sampleable(vk::ImageLayout l) {
    return l == vk::ImageLayout::eGeneral || l == vk::ImageLayout::eShaderReadOnlyOptimal ||
           l == vk::ImageLayout::eDepthStencilReadOnlyOptimal ||
           l == vk::ImageLayout::eDepthReadOnlyOptimal || l == vk::ImageLayout::eReadOnlyOptimal;
}
vk::ImageAspectFlags DepthAspect(bool stencil) {
    return stencil ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil
                   : vk::ImageAspectFlags{vk::ImageAspectFlagBits::eDepth};
}
// Game images moved to GENERAL for a pass (when not sampleable / to be written) and back.
struct Transitions {
    struct Tr {
        vk::Image image;
        vk::ImageLayout from;
        vk::ImageAspectFlags aspect;
    };
    std::array<Tr, 3> trs{};
    u32 n = 0;
    void Add(vk::Image image, vk::ImageLayout from, vk::ImageAspectFlags aspect, bool write) {
        if (!image || (write ? from == vk::ImageLayout::eGeneral : Sampleable(from)))
            return;
        for (u32 i = 0; i < n; ++i) {
            if (trs[i].image == image)
                return;
        }
        trs[n++] = {image, from, aspect};
    }
    void Record(vk::CommandBuffer cmd, bool to_general) const {
        std::array<vk::ImageMemoryBarrier2, 3> b{};
        for (u32 i = 0; i < n; ++i) {
            b[i] = vk::ImageMemoryBarrier2{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask =
                    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                .oldLayout = to_general ? trs[i].from : vk::ImageLayout::eGeneral,
                .newLayout = to_general ? vk::ImageLayout::eGeneral : trs[i].from,
                .image = trs[i].image,
                .subresourceRange{trs[i].aspect, 0, 1, 0, 1},
            };
        }
        cmd.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &AllBefore,
            .imageMemoryBarrierCount = n,
            .pImageMemoryBarriers = b.data(),
        });
    }
};
} // namespace

void VolFogCycleDebug() {
    int v = g_debug.load();
    if (v < 0)
        v = DebugEnv();
    v = (v + 1) % 6;
    g_debug.store(v);
    LOG_INFO(Render_Vulkan, "[volfog] Shift+F7 debug view: {}",
             v == 1   ? "1 fog only (in-scattered light)"
             : v == 2 ? "2 fog off (A/B)"
             : v == 3 ? "3 transmittance (white = clear)"
             : v == 4 ? "4 point lights only (added light)"
             : v == 5 ? "5 moon only (added light, no point lights)"
                      : "off (fog applied)");
}

bool VolFog::IsPointLightPs(u64 hash) {
    return FindPointShader(hash) != nullptr;
}

int VolFog::Level() {
    static const int lv = [] {
        const char* e = std::getenv("BB_VOLFOG");
        return e && e[0] >= '0' && e[0] <= '3' ? e[0] - '0' : 0;
    }();
    return lv;
}

VolFog::VolFog(const Vulkan::Instance& instance_) : instance{instance_} {}

VolFog::~VolFog() {
    const auto device = instance.GetDevice();
    if (!device)
        return;
    (void)device.waitIdle();
    FreeImage(vis);
    for (auto& i : scat)
        FreeImage(i);
    FreeImage(integ);
    FreeBuffer(params);
    FreeBuffer(lights_buf);
    FreeBuffer(counters);
    for (auto& p : pipes) {
        if (p)
            device.destroyPipeline(p);
    }
    if (layout)
        device.destroyPipelineLayout(layout);
    if (set_layout)
        device.destroyDescriptorSetLayout(set_layout);
    if (queries)
        device.destroyQueryPool(queries);
    if (nearest)
        device.destroySampler(nearest);
    if (linear)
        device.destroySampler(linear);
}

bool VolFog::MakeBuffer(Buf& b, vk::DeviceSize size, vk::BufferUsageFlags usage) {
    const VkBufferCreateInfo bci = vk::BufferCreateInfo{.size = size, .usage = usage};
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkBuffer buffer{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance.GetAllocator(), &bci, &aci, &buffer, &b.alloc, &info) !=
        VK_SUCCESS) {
        b = {};
        return false;
    }
    b.buffer = buffer;
    b.mapped = info.pMappedData;
    b.size = size;
    std::memset(b.mapped, 0, size);
    return true;
}

void VolFog::FreeBuffer(Buf& b) {
    if (b.buffer)
        vmaDestroyBuffer(instance.GetAllocator(), b.buffer, b.alloc);
    b = {};
}

bool VolFog::MakeImage3D(Img& i, vk::Format format) {
    const VkImageCreateInfo ici = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e3D,
        .format = format,
        .extent = {gw, gh, gd},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eTransferDst,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkImage image{};
    if (vmaCreateImage(instance.GetAllocator(), &ici, &aci, &image, &i.alloc, nullptr) !=
        VK_SUCCESS) {
        i = {};
        return false;
    }
    i.image = image;
    auto [r, v] = instance.GetDevice().createImageView({
        .image = i.image,
        .viewType = vk::ImageViewType::e3D,
        .format = format,
        .subresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
    if (r != vk::Result::eSuccess) {
        FreeImage(i);
        return false;
    }
    i.view = v;
    return true;
}

void VolFog::FreeImage(Img& i) {
    if (i.view)
        instance.GetDevice().destroyImageView(i.view);
    if (i.image)
        vmaDestroyImage(instance.GetAllocator(), i.image, i.alloc);
    i = {};
}

bool VolFog::Init() {
    if (tried)
        return ok;
    tried = true;
    level = Level();
    if (level <= 0)
        return false;
    const Preset& pr = Presets[level];
    level_name = pr.name;
    gw = pr.w;
    gh = pr.h;
    gd = pr.d;
    if (const char* e = std::getenv("BB_VOLFOG_GRID"); e && e[0]) {
        u32 a = 0, b = 0, c = 0;
        if (std::sscanf(e, "%ux%ux%u", &a, &b, &c) == 3) {
            gw = std::clamp(a, 32u, 320u);
            gh = std::clamp(b, 18u, 180u);
            gd = std::clamp(c, 16u, 128u);
        }
    }
    far_dist = EnvFloat("BB_VOLFOG_DIST", 100.0f, 20.0f, 400.0f);
    const float dens = EnvFloat("BB_VOLFOG_DENSITY", 1.0f, 0.0f, 20.0f);
    haze = pr.haze * dens;
    mist = pr.mist * EnvFloat("BB_VOLFOG_MIST", 1.0f, 0.0f, 20.0f) * dens;
    noise_amp = pr.noise;
    scatter = pr.scatter * EnvFloat("BB_VOLFOG_STRENGTH", 1.0f, 0.0f, 10.0f);
    lights_scale = pr.lights * EnvFloat("BB_VOLFOG_LIGHTS", 1.0f, 0.0f, 20.0f);
    shaft_scale = pr.shaft * EnvFloat("BB_VOLFOG_SHAFTS", 1.0f, 0.0f, 10.0f);
    cap_rel = EnvFloat("BB_VOLFOG_CAP", pr.cap, 0.0f, 10.0f);
    light_cap_rel = EnvFloat("BB_VOLFOG_LIGHT_CAP", pr.lcap, 0.0f, 100.0f);
    light_max_r = EnvFloat("BB_VOLFOG_LIGHT_MAXR", 12.0f, 0.5f, 200.0f);
    if (const char* e = std::getenv("BB_VOLFOG_LIGHTS_ALL"); e && e[0] == '1')
        lights_all = true;
    game_fog_mul = EnvFloat("BB_VOLFOG_GAMEFOG_SCALE", 1.0f, 0.0f, 10.0f);
    g_moon = EnvFloat("BB_VOLFOG_G", 0.5f, 0.0f, 0.9f);
    mist_h = EnvFloat("BB_VOLFOG_MIST_HEIGHT", 1.2f, 0.1f, 20.0f);
    // Layers: ground mist, mid haze (above) and a high thin band; BB_VOLFOG_LAYERS=0 drops the band.
    high_dens = pr.high * dens * EnvFloat("BB_VOLFOG_HIGH", 1.0f, 0.0f, 20.0f);
    if (const char* e = std::getenv("BB_VOLFOG_LAYERS"); e && e[0] == '0')
        high_dens = 0.0f;
    high_h = EnvFloat("BB_VOLFOG_HIGH_HEIGHT", 40.0f, 5.0f, 400.0f);
    high_thick = EnvFloat("BB_VOLFOG_HIGH_THICK", 8.0f, 1.0f, 100.0f);
    edge_k = EnvFloat("BB_VOLFOG_EDGE", pr.edge, 1.0f, 4.0f);
    if (high_dens > 0.0f) {
        LOG_INFO(Render_Vulkan,
                 "[volfog] layers: ground mist (drift x1.0), mid haze (drift x0.6), high band "
                 "{:.5f}/m at {:.0f} m +- {:.0f} m above the ground (drift x-0.4); shaft edge x{:.2f}",
                 high_dens, high_h, high_thick, edge_k);
    }
    haze_h = EnvFloat("BB_VOLFOG_HAZE_HEIGHT", 30.0f, 1.0f, 1000.0f);
    ground_below = EnvFloat("BB_VOLFOG_GROUND", 1.8f, -50.0f, 50.0f);
    if (const char* e = std::getenv("BB_VOLFOG_WIND"); e && e[0]) {
        float v[3]{};
        const int n = std::sscanf(e, "%f,%f,%f", &v[0], &v[1], &v[2]);
        if (n == 2) {
            wind[0] = std::clamp(v[0], -20.0f, 20.0f);
            wind[1] = 0.0f;
            wind[2] = std::clamp(v[1], -20.0f, 20.0f);
            wind_src = "BB_VOLFOG_WIND (x, z)";
        } else if (n == 3) {
            for (int i = 0; i < 3; ++i)
                wind[i] = std::clamp(v[i], -20.0f, 20.0f);
            wind_src = "BB_VOLFOG_WIND (x, y, z)";
        }
    }
    if (const char* e = std::getenv("BB_VOLFOG_SHADOW_CMP"); e && (e[0] == '0' || e[0] == '1')) {
        shadow_cmp_env = e[0] - '0';
    }
    if (const char* e = std::getenv("BB_VOLFOG_GAMEFOG"); e && e[0] == '0')
        game_fog_tint = false;
    if (const char* e = std::getenv("BB_VOLFOG_WIND_PROBE"); e && e[0] == '1')
        wind_probe = true;
    t0 = std::chrono::steady_clock::now();

    const auto device = instance.GetDevice();
    ts_period = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
    {
        const vk::SamplerCreateInfo sci{
            .magFilter = vk::Filter::eNearest,
            .minFilter = vk::Filter::eNearest,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        };
        auto [r1, s1] = device.createSampler(sci);
        vk::SamplerCreateInfo lci = sci;
        lci.magFilter = vk::Filter::eLinear;
        lci.minFilter = vk::Filter::eLinear;
        auto [r2, s2] = device.createSampler(lci);
        if (r1 != vk::Result::eSuccess || r2 != vk::Result::eSuccess)
            return false;
        nearest = s1;
        linear = s2;
    }
    constexpr auto C = vk::ShaderStageFlagBits::eCompute;
    using DT = vk::DescriptorType;
    const std::array<vk::DescriptorSetLayoutBinding, 13> b{{
        {0, DT::eUniformBuffer, 1, C},
        {1, DT::eCombinedImageSampler, 1, C},
        {2, DT::eStorageImage, 1, C},
        {3, DT::eCombinedImageSampler, 1, C},
        {4, DT::eCombinedImageSampler, 1, C},
        {5, DT::eStorageImage, 1, C},
        {6, DT::eStorageBuffer, 1, C},
        {7, DT::eStorageBuffer, 1, C},
        {8, DT::eCombinedImageSampler, 1, C},
        {9, DT::eStorageImage, 1, C},
        {10, DT::eCombinedImageSampler, 1, C},
        {11, DT::eCombinedImageSampler, 1, C},
        {12, DT::eStorageImage, 1, C},
    }};
    auto [lr, l] = device.createDescriptorSetLayout({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
        .bindingCount = static_cast<u32>(b.size()),
        .pBindings = b.data(),
    });
    if (lr != vk::Result::eSuccess)
        return false;
    set_layout = l;
    auto [pr2, pl] = device.createPipelineLayout({.setLayoutCount = 1, .pSetLayouts = &set_layout});
    if (pr2 != vk::Result::eSuccess)
        return false;
    layout = pl;
    // Order matches pipes[]: inject, light, integrate, apply, apply R11, measure, measure R11.
    const std::array<std::span<const u32>, 7> code{
        BB_VOLFOG_INJECT_COMP,  BB_VOLFOG_LIGHT_COMP,   BB_VOLFOG_INTEGRATE_COMP,
        BB_VOLFOG_APPLY_COMP,   BB_VOLFOG_APPLY_R11_COMP, BB_VOLFOG_MEASURE_COMP,
        BB_VOLFOG_MEASURE_R11_COMP};
    static_assert(code.size() == std::tuple_size_v<decltype(pipes)>);
    for (size_t i = 0; i < code.size(); ++i) {
        const auto module = Vulkan::CompileSPV(code[i], device);
        if (!module) {
            LOG_ERROR(Render_Vulkan, "[volfog] shader {} failed to compile: fog off", i);
            return false;
        }
        auto [cr, pipe] = device.createComputePipeline(
            {}, vk::ComputePipelineCreateInfo{
                    .stage{.stage = C, .module = module, .pName = "main"},
                    .layout = layout,
                });
        device.destroyShaderModule(module);
        if (cr != vk::Result::eSuccess) {
            LOG_ERROR(Render_Vulkan, "[volfog] pipeline {} failed: fog off", i);
            return false;
        }
        pipes[i] = pipe;
    }
    if (!MakeImage3D(vis, vk::Format::eR16Sfloat) ||
        !MakeImage3D(scat[0], vk::Format::eR16G16B16A16Sfloat) ||
        !MakeImage3D(scat[1], vk::Format::eR16G16B16A16Sfloat) ||
        !MakeImage3D(integ, vk::Format::eR16G16B16A16Sfloat)) {
        LOG_ERROR(Render_Vulkan, "[volfog] {}x{}x{} froxel images: allocation failed, fog off", gw,
                  gh, gd);
        return false;
    }
    using U = vk::BufferUsageFlagBits;
    if (!MakeBuffer(params, ParamSize * Ring, U::eUniformBuffer) ||
        !MakeBuffer(lights_buf, sizeof(GpuLight) * MaxLights * Ring, U::eStorageBuffer) ||
        !MakeBuffer(counters, 64 * Ring, U::eStorageBuffer | U::eTransferDst)) {
        LOG_ERROR(Render_Vulkan, "[volfog] buffers: allocation failed, fog off");
        return false;
    }
    auto [qr, qp] = device.createQueryPool({
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = Ring * 4,
    });
    if (qr == vk::Result::eSuccess)
        queries = qp;
    ok = true;
    const double mb = double(gw) * gh * gd * (2.0 + 8.0 * 3.0) / (1024.0 * 1024.0);
    LOG_INFO(
        Render_Vulkan,
        "[volfog] BB_VOLFOG={} ({}): froxels {}x{}x{} to {:.0f} m ({:.1f} MB); medium = the "
        "game's own distance fog (FUN_02651f10 CB) x{:.2f}, moon shafts in it x{:.2f}; own "
        "haze where the game has none {:.4f}/m (H {:.0f} m), ground mist {:.4f}/m (h {:.1f} "
        "m, {:.1f} m below the camera); noise {:.2f}, scatter x{:.2f}, added light cap x{:.2f} "
        "scene log-avg luminance, point lights (lamp-like, radius <= {:.0f} m{}) x{:.2f} "
        "capped x{:.1f}, moon HG g {:.2f}, wind ({:.2f}, {:.2f}, {:.2f}) m/s [{}], game fog "
        "tint {}, debug {} (Shift+F7 cycles){}",
        level, level_name, gw, gh, gd, far_dist, mb, game_fog_mul, shaft_scale, haze, haze_h, mist,
        mist_h, ground_below, noise_amp, scatter, cap_rel, light_max_r,
        lights_all ? ", BB_VOLFOG_LIGHTS_ALL: all" : "", lights_scale, light_cap_rel, g_moon,
        wind[0], wind[1], wind[2], wind_src, game_fog_tint ? "on" : "off", DebugEnv(),
        std::getenv("BB_VOLFOG_WISPS") && std::getenv("BB_VOLFOG_WISPS")[0] == '1'
            ? "; BB_VOLFOG_WISPS=1 ignored (sprite-to-puff not implemented)"
            : "");
    return true;
}

bool VolFog::Camera() {
    cam_ok = false;
    if (!BB::Upscale::RtSceneBlock(block))
        return false;
    double v[16]{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            v[r * 4 + c] = block[8 + r * 4 + c];
    v[15] = 1.0;
    double p[16];
    for (int i = 0; i < 16; ++i)
        p[i] = block[52 + i];
    if (!Invert4(p, invP) || !Invert4(v, invV))
        return false;
    std::memcpy(V, v, sizeof(V));
    std::memcpy(P, p, sizeof(P));
    Mul4(P, V, PV);
    cam_ok = true;
    return true;
}

// FC_ProjSpaceToShadowMatrix[4] (scene block floats 112..175) and FC_CascadeSelectDist (176..179),
// from the path-tracing mod's reflection of the game's shaders. Checked offline on a scene block
// of this port's own log (run_native1, BB_RT_VOLUMETRIC block dump): matrix rows x the camera's
// clip position give atlas uv with the 2x2 quadrant offsets included (cascade 0 top-left, 1
// top-right, 2 bottom-left, 3 bottom-right), the product with proj x view is affine, and
// FC_CascadeSelectDist holds THREE transition distances (x, y, z; w = 0): cascade 3 covers
// everything beyond z. Light direction = null space of the matrix rows 0/1 (that block: (0,
// 0.707, -0.707), equal to scene block floats 80-82). The resolver still tries every convention
// on sample points of this frame's frustum, scores each, and keeps the last good one when a frame
// scores lower (sticky), so a passing outlier never drops the shadows.
void VolFog::ResolveCascades() {
    casc_ok = false;
    moon_from_casc = false;
    const float* M = block + 112;
    const float* cd = block + 176;
    for (int i = 0; i < 64; ++i) {
        if (!std::isfinite(M[i])) {
            casc_why = "FC_ProjSpaceToShadowMatrix not finite";
            return;
        }
    }
    casc_dist_ok = std::isfinite(cd[0]) && std::isfinite(cd[1]) && std::isfinite(cd[2]) &&
                   cd[0] > 0.05f && cd[1] > cd[0] && cd[2] > cd[1] && cd[2] < 1e5f;
    const bool cd_end = casc_dist_ok && std::isfinite(cd[3]) && cd[3] > cd[2];
    double lo[4], hi[4];
    if (casc_dist_ok) {
        lo[0] = 0.3;
        hi[0] = cd[0];
        for (int i = 1; i < 3; ++i) {
            lo[i] = cd[i - 1];
            hi[i] = cd[i];
        }
        lo[3] = cd[2];
        hi[3] = cd_end ? cd[3] : cd[2] * 1.4;
    } else {
        const double l[4] = {0.3, 3, 8, 20}, h[4] = {3, 8, 20, 30};
        std::copy(l, l + 4, lo);
        std::copy(h, h + 4, hi);
    }
    struct Pt {
        double w[3];
        int c;
    };
    std::vector<Pt> pts;
    pts.reserve(4 * 3 * 9);
    for (int ci = 0; ci < 4; ++ci) {
        for (double fz : {0.1, 0.5, 0.9}) {
            const double depth = lo[ci] + (hi[ci] - lo[ci]) * fz;
            for (double u : {0.1, 0.5, 0.9}) {
                for (double vv : {0.1, 0.5, 0.9}) {
                    const double ndc[3] = {u * 2.0 - 1.0, 1.0 - 2.0 * vv, 0.5};
                    double r[4];
                    Xform(invP, ndc, r);
                    if (std::fabs(r[3]) < 1e-12)
                        continue;
                    double vp[3] = {r[0] / r[3], r[1] / r[3], r[2] / r[3]};
                    if (std::fabs(vp[2]) < 1e-9)
                        continue;
                    const double k = depth / std::fabs(vp[2]);
                    for (double& x : vp)
                        x *= k;
                    double wp[4];
                    Xform(invV, vp, wp);
                    pts.push_back({{wp[0], wp[1], wp[2]}, ci});
                }
            }
        }
    }
    if (pts.empty()) {
        casc_why = "no sample points (projection)";
        return;
    }
    struct Fit {
        double W[4][16];
        float score = -1.0f;
        bool ndc = false;
    };
    std::array<Fit, 4> fits{};
    for (int combo = 0; combo < 4; ++combo) {
        Fit& f = fits[combo];
        for (int i = 0; i < 4; ++i) {
            double m[16];
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    m[r * 4 + c] = (combo & 1) ? M[i * 16 + c * 4 + r] : M[i * 16 + r * 4 + c];
            Mul4(m, combo < 2 ? PV : V, f.W[i]);
        }
        u32 in_uv = 0, in_ndc = 0, n = 0;
        double wmin[4] = {1e30, 1e30, 1e30, 1e30}, wmax[4] = {-1e30, -1e30, -1e30, -1e30};
        for (const auto& pt : pts) {
            double s[4];
            Xform(f.W[pt.c], pt.w, s);
            if (!(std::fabs(s[3]) > 1e-12))
                continue;
            wmin[pt.c] = std::min(wmin[pt.c], s[3]);
            wmax[pt.c] = std::max(wmax[pt.c], s[3]);
            const double x = s[0] / s[3], y = s[1] / s[3], z = s[2] / s[3];
            ++n;
            in_uv +=
                x >= -0.001 && x <= 1.001 && y >= -0.001 && y <= 1.001 && z >= -0.01 && z <= 1.01;
            in_ndc +=
                x >= -1.001 && x <= 1.001 && y >= -1.001 && y <= 1.001 && z >= -1.01 && z <= 1.01;
        }
        if (n == 0)
            continue;
        // Orthographic light: w (nearly) constant per cascade (the block's own matrices vary
        // by ~0.3% over 20 m, float precision of the game's product).
        bool affine = true;
        for (int i = 0; i < 4; ++i) {
            if (wmax[i] >= wmin[i] &&
                (wmax[i] - wmin[i]) > 0.02 * std::max(std::fabs(wmax[i]), std::fabs(wmin[i]))) {
                affine = false;
            }
        }
        f.ndc = in_ndc > in_uv + n / 10;
        f.score = float(f.ndc ? in_ndc : in_uv) / float(pts.size());
        if (!affine)
            f.score *= 0.5f;
    }
    int best = 0;
    for (int c = 1; c < 4; ++c) {
        if (fits[c].score > fits[best].score)
            best = c;
    }
    // Sticky: the convention found earlier is kept while it still fits most points.
    if (casc_combo >= 0 && fits[casc_combo].score >= 0.6f &&
        fits[casc_combo].score + 0.15f >= fits[best].score) {
        best = casc_combo;
    }
    casc_score = fits[best].score;
    if (casc_score < 0.6f) {
        casc_why = fmt::format("cascade matrices not recognised (best fit {:.0f}%, combo {}; "
                               "select {:.1f}/{:.1f}/{:.1f}/{:.1f})",
                               casc_score * 100.0f, best, cd[0], cd[1], cd[2], cd[3]);
        return;
    }
    if (casc_combo != best && casc_score >= 0.85f) {
        casc_combo = best;
    } else if (casc_combo < 0) {
        casc_why = fmt::format("cascade fit {:.0f}% < 85% (combo {})", casc_score * 100.0f, best);
        return;
    }
    const Fit& F = fits[best];
    casc_ndc = F.ndc;
    rmap[0] = casc_ndc ? 0.5f : 1.0f;
    rmap[1] = casc_ndc ? -0.5f : 1.0f; // GNM: image row 0 = NDC y +1 (as the bb-rt passes)
    rmap[2] = casc_ndc ? 0.5f : 0.0f;
    rmap[3] = casc_ndc ? 0.5f : 0.0f;
    // Tiles: each cascade's points covering at most a quarter of the atlas = offsets included;
    // its quadrant (from the centre) bounds the PCF and the range test.
    bool baked = true;
    double cu[4]{}, cv[4]{};
    for (int ci = 0; ci < 4; ++ci) {
        double mn[2] = {1e9, 1e9}, mx[2] = {-1e9, -1e9};
        for (const auto& pt : pts) {
            if (pt.c != ci)
                continue;
            double s[4];
            Xform(F.W[ci], pt.w, s);
            const double u = s[0] / s[3] * rmap[0] + rmap[2];
            const double v = s[1] / s[3] * rmap[1] + rmap[3];
            mn[0] = std::min(mn[0], u);
            mx[0] = std::max(mx[0], u);
            mn[1] = std::min(mn[1], v);
            mx[1] = std::max(mx[1], v);
        }
        baked = baked && (mx[0] - mn[0]) <= 0.55 && (mx[1] - mn[1]) <= 0.55;
        cu[ci] = 0.5 * (mn[0] + mx[0]);
        cv[ci] = 0.5 * (mn[1] + mx[1]);
    }
    casc_baked = baked;
    for (int i = 0; i < 4; ++i) {
        ToGlsl(F.W[i], shadow_w2s[i]);
        float qx, qy;
        if (baked) {
            std::memcpy(tile[i], rmap, sizeof(rmap));
            qx = cu[i] >= 0.5 ? 0.5f : 0.0f;
            qy = cv[i] >= 0.5 ? 0.5f : 0.0f;
        } else {
            tile[i][0] = rmap[0] * 0.5f;
            tile[i][1] = rmap[1] * 0.5f;
            tile[i][2] = rmap[2] * 0.5f + float(i & 1) * 0.5f;
            tile[i][3] = rmap[3] * 0.5f + float(i >> 1) * 0.5f;
            qx = float(i & 1) * 0.5f;
            qy = float(i >> 1) * 0.5f;
        }
        tbound[i][0] = qx;
        tbound[i][1] = qy;
        tbound[i][2] = qx + 0.5f;
        tbound[i][3] = qy + 0.5f;
        cdist[i] = casc_dist_ok ? (i < 3 ? cd[i] : (cd_end ? cd[3] : 0.0f)) : -1.0f;
    }
    const double* W0 = F.W[0];
    const double r0[3] = {W0[0], W0[1], W0[2]}, r1[3] = {W0[4], W0[5], W0[6]};
    double d[3] = {r0[1] * r1[2] - r0[2] * r1[1], r0[2] * r1[0] - r0[0] * r1[2],
                   r0[0] * r1[1] - r0[1] * r1[0]};
    const double dl = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (!(dl > 1e-12)) {
        casc_why = "cascade 0 rows 0 / 1 parallel";
        return;
    }
    for (double& x : d)
        x /= dl;
    float lb[72];
    u64 age = 0;
    double sign = d[1] >= 0.0 ? 1.0 : -1.0;
    if (std::fabs(d[1]) < 0.1 && BB::Upscale::RtLightBlock(lb, age) && age <= 600) {
        const double dot = d[0] * lb[0] + d[1] * lb[1] + d[2] * lb[2];
        sign = (dot >= 0.0) == (lb[1] >= 0.0) ? 1.0 : -1.0;
    }
    for (int i = 0; i < 3; ++i)
        casc_dir[i] = float(d[i] * sign);
    const double dz = (W0[8] * casc_dir[0] + W0[9] * casc_dir[1] + W0[10] * casc_dir[2]) /
                      (std::fabs(W0[15]) > 1e-12 ? W0[15] : 1.0);
    shadow_cmp = shadow_cmp_env >= 0 ? shadow_cmp_env : (dz < 0.0 ? 0 : 1);
    moon_from_casc = true;
    casc_ok = true;
    casc_why = "ok";
}

void VolFog::MoonAndAmbient() {
    float lb[72];
    u64 age = 0;
    const bool cb = BB::Upscale::RtLightBlock(lb, age) && age <= 600;
    float cb_dir[3]{};
    bool cb_dir_ok = false;
    if (cb) {
        const float m = std::max({lb[16], lb[17], lb[18]});
        if (m > 1e-5f) {
            const float k = m > 4.0f ? 4.0f / m : 1.0f;
            for (int i = 0; i < 3; ++i)
                last_cb_col[i] = lb[16 + i] * k;
            have_cb_col = true;
        }
        const float l = std::sqrt(lb[0] * lb[0] + lb[1] * lb[1] + lb[2] * lb[2]);
        if (l > 0.5f) {
            const float s = lb[1] >= 0.0f ? 1.0f : -1.0f; // toward the light, above the horizon
            for (int i = 0; i < 3; ++i)
                cb_dir[i] = s * lb[i] / l;
            cb_dir_ok = true;
        }
    }
    if (moon_from_casc) {
        std::memcpy(moon_dir, casc_dir, sizeof(moon_dir));
        moon_src = "cascade shadow matrices (FC_ProjSpaceToShadowMatrix)";
        moon_on = true;
    } else if (cb_dir_ok) {
        std::memcpy(moon_dir, cb_dir, sizeof(moon_dir));
        moon_src = "light-set CB slot 5 light 0 (FUN_026deeb0)";
        moon_on = true;
    } else {
        moon_src = "none";
        moon_on = false;
    }
    if (have_cb_col) {
        std::memcpy(moon_col, last_cb_col, sizeof(moon_col));
        col_src = cb ? "light-set CB light 0 diffuse" : "light-set CB light 0 diffuse (last seen)";
    } else {
        moon_col[0] = 0.30f;
        moon_col[1] = 0.36f;
        moon_col[2] = 0.50f;
        col_src = "fallback cold blue-grey (no light-set CB seen)";
    }
    // Ambient: a cold sky term proportional to the moon, tinted by the game's fog colour when its
    // fog CB (slot 6) is fresh.
    const float base[3] = {0.55f, 0.62f, 0.80f};
    const float k = 0.04f + 0.08f * std::min(Lum(moon_col), 4.0f);
    for (int i = 0; i < 3; ++i)
        amb[i] = base[i] * k;
    amb_src = "cold sky x moon";
    if (game_fog_tint && game_fog_weight > 0.0f) {
        const float m = std::max({game_fog.col[0], game_fog.col[1], game_fog.col[2]});
        if (m > 1e-5f) {
            const float l = Lum(amb);
            float t[3];
            for (int i = 0; i < 3; ++i)
                t[i] = game_fog.col[i] / m;
            const float lt = std::max(Lum(t), 1e-3f), lb2 = std::max(Lum(base), 1e-3f);
            for (int i = 0; i < 3; ++i)
                amb[i] = l * (0.5f * base[i] / lb2 + 0.5f * t[i] / lt);
            amb_src = "cold sky x moon, 50% game fog colour";
        }
    }
}

void VolFog::OnDrawCbs(const Upscale::DrawCb* cbs, size_t n) {
    if (!ok)
        return;
    for (size_t i = 0; i < n; ++i) {
        if (cbs[i].stage != 1 || cbs[i].bytes < 84 || cbs[i].bytes > 320 ||
            !IsFogCb(cbs[i].data, cbs[i].bytes)) {
            continue;
        }
        ++acc_fog_cbs;
        const float* f = reinterpret_cast<const float*>(cbs[i].data);
        // v0.y = 1 / Fog-Dist = 0: this draw's part has no distance fog (sky, effects).
        if (!(f[1] > 0.0f) || !(f[0] > 0.0f)) {
            ++acc_fog_off;
            break;
        }
        bool found = false;
        for (auto& g : acc_fog) {
            if (g.amount == f[0] && g.rate == f[1] && g.start == f[9]) {
                ++g.count;
                found = true;
                break;
            }
        }
        if (!found && acc_fog.size() < 16) {
            GameFogSet g{};
            g.amount = f[0];
            g.rate = f[1];
            g.start = f[9];
            g.inv_h = f[3];
            for (int k = 0; k < 3; ++k)
                g.col[k] = f[4 + k];
            g.mix = f[7];
            g.h_amount = f[12];
            g.h_base = f[13];
            g.h_scale = f[14];
            g.count = 1;
            acc_fog.push_back(g);
        }
        break;
    }
    const PointShader* ps = nullptr;
    for (size_t i = 0; i < n && !ps; ++i) {
        if (cbs[i].stage == 1)
            ps = FindPointShader(cbs[i].hash);
    }
    if (!ps)
        return;
    const u32 stride = ps->shadowed ? 96u : 64u;
    const size_t need = size_t(ps->count) * stride;
    const float* cam = nullptr; // cbLightAccParam g_CameraMatrix
    const float* arr = nullptr;
    size_t arr_bytes = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto& c = cbs[i];
        if (c.stage != 1)
            continue;
        if (c.bytes == 208) {
            cam = reinterpret_cast<const float*>(c.data) + 32;
        } else if (c.bytes % 16 == 0 && c.bytes >= need && c.bytes != 864) {
            // Live: count x 144 bytes, the lights packed at the family stride (PT mod dump).
            if (!arr || (c.bytes == size_t(ps->count) * 144 && arr_bytes != c.bytes)) {
                arr = reinterpret_cast<const float*>(c.data);
                arr_bytes = c.bytes;
            }
        }
    }
    if (!arr)
        return;
    ++acc_draws;
    if (cam)
        ++acc_cam_cb;
    for (u32 k = 0; k < ps->count; ++k) {
        const float* f = arr + k * (stride / 4);
        bool fin = true;
        for (int j = 0; j < 7; ++j)
            fin = fin && std::isfinite(f[j]);
        if (!fin || !(f[3] > 0.0f))
            continue;
        const float r = 1.0f / std::sqrt(f[3]);
        if (!(r > 0.1f && r < 200.0f))
            continue;
        const float col[3] = {std::clamp(f[4], 0.0f, 64.0f), std::clamp(f[5], 0.0f, 64.0f),
                              std::clamp(f[6], 0.0f, 64.0f)};
        const float lum = Lum(col);
        if (!(lum > 1e-3f))
            continue; // specular-only (the player's blue fill light) / off
        // Halos only for lamp-like sources: small radius and warm (fires, lamps, torches, the
        // player's lantern). Large or cold lights are the artists' fill / bounce lights.
        if (!lights_all) {
            if (r > light_max_r) {
                ++acc_rej_big;
                continue;
            }
            if (!(col[0] >= 1.1f * col[2] && col[0] >= 0.75f * col[1])) {
                ++acc_rej_cold;
                continue;
            }
        }
        double w[3];
        if (cam) {
            for (int i = 0; i < 3; ++i) {
                w[i] = double(cam[i * 4]) * f[0] + double(cam[i * 4 + 1]) * f[1] +
                       double(cam[i * 4 + 2]) * f[2] + cam[i * 4 + 3];
            }
        } else if (cam_ok) {
            const double v[3] = {f[0], f[1], f[2]};
            double o[4];
            Xform(invV, v, o);
            std::copy(o, o + 3, w);
        } else {
            continue;
        }
        if (!(std::fabs(w[0]) < 1e5 && std::fabs(w[1]) < 1e5 && std::fabs(w[2]) < 1e5))
            continue;
        ++acc_parsed;
        // The same lamp is drawn in many light batches: one entry per position.
        bool dup = false;
        for (auto& L : acc_lights) {
            if (std::fabs(L.pos[0] - w[0]) < 0.02 && std::fabs(L.pos[1] - w[1]) < 0.02 &&
                std::fabs(L.pos[2] - w[2]) < 0.02) {
                dup = true;
                break;
            }
        }
        if (dup)
            continue;
        Light L{{float(w[0]), float(w[1]), float(w[2])}, r, {col[0], col[1], col[2]}, lum};
        if (acc_lights.size() < 256) {
            acc_lights.push_back(L);
        }
    }
}

void VolFog::FrameLightsDone() {
    frame_lights.swap(acc_lights);
    acc_lights.clear();
    frame_rej_big = acc_rej_big;
    frame_rej_cold = acc_rej_cold;
    acc_rej_big = acc_rej_cold = 0;
    // The frame's game fog: the set most forward-lit draws used.
    const GameFogSet* best = nullptr;
    for (const auto& g : acc_fog) {
        if (!best || g.count > best->count)
            best = &g;
    }
    frame_fog_sets = u32(acc_fog.size());
    frame_fog_off = acc_fog_off;
    frame_fog_cbs = acc_fog_cbs;
    const auto now = std::chrono::steady_clock::now();
    if (best) {
        game_fog = *best;
        game_fog_have = true;
        game_fog_seen = now;
        ++n_fog_frames;
    } else {
        ++n_nofog_frames;
    }
    acc_fog.clear();
    acc_fog_off = acc_fog_cbs = 0;
    // Held 3 s after the last frame that showed it, then faded out over 2 s (an area without
    // fog binds only fog-off sets).
    const double age =
        game_fog_have ? std::chrono::duration<double>(now - game_fog_seen).count() : 1e9;
    game_fog_weight = float(std::clamp(1.0 - (age - 3.0) / 2.0, 0.0, 1.0));
    frame_draws = acc_draws;
    frame_parsed = acc_parsed;
    frame_cam_cb = acc_cam_cb;
    acc_draws = acc_parsed = acc_cam_cb = 0;
}

void VolFog::NoteSkip(const char* why) {
    ++n_skip;
    last_skip = why;
}

void VolFog::ReadTimers() {
    if (!queries)
        return;
    for (u32 s = 0; s < Ring; ++s) {
        for (int k = 0; k < 2; ++k) {
            auto& used = k == 0 ? inj_used[s] : app_used[s];
            if (!used)
                continue;
            std::array<u64, 2> ts{};
            if (instance.GetDevice().getQueryPoolResults(
                    queries, s * 4 + k * 2, 2, sizeof(ts), ts.data(), sizeof(u64),
                    vk::QueryResultFlagBits::e64) == vk::Result::eSuccess) {
                used = false;
                const double ms = double(ts[1] - ts[0]) * ts_period * 1e-6;
                if (ms > 0.0 && ms < 1000.0) {
                    (k == 0 ? inj_ms : app_ms) += ms;
                    ++(k == 0 ? inj_n : app_n);
                }
            }
        }
    }
}

// First use: every froxel image to GENERAL, cleared (no history).
void VolFog::InitImages(vk::CommandBuffer cmd) {
    if (!fresh)
        return;
    std::array<vk::ImageMemoryBarrier2, 4> b{};
    const vk::Image imgs[4] = {vis.image, scat[0].image, scat[1].image, integ.image};
    for (int i = 0; i < 4; ++i) {
        b[i] = vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = imgs[i],
            .subresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
    }
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 4, .pImageMemoryBarriers = b.data()});
    const vk::ClearColorValue zero{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}};
    const vk::ImageSubresourceRange rng{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    for (int i = 0; i < 4; ++i) {
        cmd.clearColorImage(imgs[i], vk::ImageLayout::eGeneral, zero, rng);
    }
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &AllBefore});
    fresh = false;
    have_prev = false;
    hist_why = "start";
}

bool VolFog::Inject(vk::CommandBuffer cmd, const InjectInput& in, const char*& why) {
    why = "none";
    if (!Init()) {
        why = "off";
        return false;
    }
    if (!Camera()) {
        why = "no camera (864 B scene block) yet";
        return false;
    }
    ++frame;
    frame_jitter = float(std::fmod(double(frame) * 0.6180339887498949, 1.0));
    frame_has_jitter = true;
    atlas_addr = in.atlas_addr;
    atlas_w = in.atlas_w;
    atlas_h = in.atlas_h;
    inject_ps = in.ps_hash;
    ResolveCascades();
    if (!casc_ok) {
        why = "cascade matrices not recognised";
        return false;
    }
    ReadTimers();
    const u32 slot = ring++ % Ring;
    VfParams prm{};
    ToGlsl(invP, prm.inv_proj);
    ToGlsl(invV, prm.inv_view);
    std::memcpy(prm.shadow, shadow_w2s, sizeof(prm.shadow));
    std::memcpy(prm.tile, tile, sizeof(prm.tile));
    std::memcpy(prm.tbound, tbound, sizeof(tbound));
    std::memcpy(prm.cdist, cdist, sizeof(cdist));
    prm.grid[0] = gw;
    prm.grid[1] = gh;
    prm.grid[2] = gd;
    prm.grid[3] = u32(frame);
    prm.cam[3] = far_dist;
    prm.misc[2] = 1.0f;
    prm.misc[3] = frame_jitter;
    prm.misc2[3] = float(shadow_cmp);
    prm.misc3[0] = 0.0015f;
    std::memcpy(static_cast<u8*>(params.mapped) + slot * ParamSize, &prm, sizeof(prm));
    inj_slot_last = slot;
    inj_slot_hist[inj_hist_pos++ % inj_slot_hist.size()] = slot;

    if (queries) {
        cmd.resetQueryPool(queries, slot * 4, 2);
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, queries, slot * 4);
        inj_used[slot] = true;
    }
    Transitions tr;
    tr.Add(in.atlas, in.atlas_layout, DepthAspect(in.atlas_stencil), false);
    tr.Record(cmd, true);
    InitImages(cmd);
    cmd.fillBuffer(counters.buffer, slot * 64, 64, 0);
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &AllBefore});
    const vk::DescriptorBufferInfo prm_info{params.buffer, slot * ParamSize, ParamSize};
    const vk::DescriptorImageInfo atlas_info{
        nearest, in.atlas_view,
        Sampleable(in.atlas_layout) ? in.atlas_layout : vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo vis_info{{}, vis.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorBufferInfo cnt_info{counters.buffer, slot * 64, 64};
    const std::array<vk::WriteDescriptorSet, 4> wr{{
        {.dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eUniformBuffer,
         .pBufferInfo = &prm_info},
        {.dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .pImageInfo = &atlas_info},
        {.dstBinding = 2,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .pImageInfo = &vis_info},
        {.dstBinding = 6,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageBuffer,
         .pBufferInfo = &cnt_info},
    }};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipes[0]);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, wr);
    cmd.dispatch((gw + 7) / 8, (gh + 7) / 8, gd);
    if (queries) {
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, queries, slot * 4 + 1);
    }
    tr.Record(cmd, false);
    if (tr.n == 0) {
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &AllBefore});
    }
    ++n_inject;
    return true;
}

bool VolFog::Apply(vk::CommandBuffer cmd, const ApplyInput& in, const char*& why) {
    why = "none";
    if (!Init()) {
        why = "off";
        return false;
    }
    for (const auto& pl : pipes) {
        if (!pl) {
            why = "pipeline missing";
            return false;
        }
    }
    const bool injected = in.injected && frame_has_jitter && casc_ok && !fresh;
    // No shadow-mask pass this frame: the game draws no moon shadow (interiors), no moon fog.
    shadow_pass = frame_has_jitter;
    if (!frame_has_jitter) {
        // No shadow-mask pass this frame (interiors, loading): a new frame starts here.
        if (!Camera()) {
            why = "no camera (864 B scene block) yet";
            return false;
        }
        ++frame;
        frame_jitter = float(std::fmod(double(frame) * 0.6180339887498949, 1.0));
        moon_from_casc = false;
    } else if (!cam_ok) {
        why = "no camera";
        return false;
    }
    frame_has_jitter = false;
    if (!injected) {
        ++n_noinject;
        moon_from_casc = false;
    }
    ReadTimers();
    MoonAndAmbient();

    // History: the previous apply within 0.25 s, no camera cut, same images.
    const auto now = std::chrono::steady_clock::now();
    bool hist = have_prev && !fresh;
    if (!hist) {
        // reason already set
    } else if (now - last_apply > std::chrono::milliseconds(250) || frame != last_apply_frame + 1) {
        hist = false;
        hist_why = "a frame without fog";
    } else {
        double dt = 0.0;
        for (int k = 0; k < 3; ++k) {
            const double e = invV[k * 4 + 3] - prev_cam[k];
            dt += e * e;
        }
        const double fwd = V[8] * prevV[8] + V[9] * prevV[9] + V[10] * prevV[10];
        if (!(std::sqrt(dt) < 8.0) || !(fwd > 0.94)) {
            hist = false;
            hist_why = "camera cut";
        }
    }
    if (!hist)
        ++hist_resets;
    last_apply = now;

    // Ground reference: a smoothed camera height minus the camera's usual height above the floor.
    const float cam_y = float(invV[7]);
    const float tsec = float(std::fmod(std::chrono::duration<double>(now - t0).count(), 4096.0));
    if (!cam_y_init || std::fabs(cam_y - cam_y_smooth) > 10.0f) {
        cam_y_smooth = cam_y;
        cam_y_init = true;
    } else {
        cam_y_smooth += (cam_y - cam_y_smooth) * 0.05f;
    }

    const u32 slot = ring++ % Ring;
    // Lights: the strongest (luminance x radius) of the frame, at most MaxLights.
    std::vector<Light> ls = frame_lights;
    std::sort(ls.begin(), ls.end(),
              [](const Light& a, const Light& b) { return a.lum * a.r > b.lum * b.r; });
    const u32 nl = std::min<u32>(u32(ls.size()), MaxLights);
    auto* gl = reinterpret_cast<GpuLight*>(static_cast<u8*>(lights_buf.mapped) +
                                           slot * sizeof(GpuLight) * MaxLights);
    for (u32 i = 0; i < nl; ++i) {
        gl[i] = GpuLight{{ls[i].pos[0], ls[i].pos[1], ls[i].pos[2], ls[i].r},
                         {ls[i].col[0], ls[i].col[1], ls[i].col[2], 0.0f}};
    }
    lights_max = std::max(lights_max, nl);
    lights_sum += nl;

    int dbg = g_debug.load();
    if (dbg < 0)
        dbg = DebugEnv();
    VfParams prm{};
    ToGlsl(invP, prm.inv_proj);
    ToGlsl(invV, prm.inv_view);
    if (hist) {
        ToGlsl(prevV, prm.prev_view);
        ToGlsl(prevP, prm.prev_proj);
    }
    prm.grid[0] = gw;
    prm.grid[1] = gh;
    prm.grid[2] = gd;
    prm.grid[3] = u32(frame);
    prm.sizes[0] = in.hdr_w;
    prm.sizes[1] = in.hdr_h;
    prm.sizes[2] = in.depth_w;
    prm.sizes[3] = in.depth_h;
    prm.cam[0] = float(invV[3]);
    prm.cam[1] = float(invV[7]);
    prm.cam[2] = float(invV[11]);
    prm.cam[3] = far_dist;
    const bool moon_lit = moon_on;
    for (int i = 0; i < 3; ++i) {
        prm.moon_dir[i] = moon_dir[i];
        prm.moon_col[i] = moon_col[i] * (injected ? 1.0f : unshadowed_moon);
        prm.amb_col[i] = amb[i];
        prm.wind[i] = wind[i];
    }
    prm.moon_dir[3] = !moon_lit || !shadow_pass ? 0.0f : injected ? 2.0f : 1.0f;
    prm.moon_col[3] = g_moon;
    prm.amb_col[3] = g_lights;
    // The own medium only where the game has no fog of its own.
    const float own = game_fog_have ? 1.0f - game_fog_weight : 1.0f;
    prm.fog[0] = haze * own;
    prm.fog[1] = haze_h;
    prm.fog[2] = mist * own;
    prm.fog[3] = mist_h;
    prm.fog2[0] = cam_y_smooth - ground_below;
    prm.fog2[1] = noise_amp;
    prm.fog2[2] = noise_freq;
    prm.fog2[3] = tsec;
    prm.wind[3] = albedo;
    prm.misc[0] = temporal;
    prm.misc[1] = hist ? 1.0f : 0.0f;
    prm.misc[2] = 1.0f;
    prm.misc[3] = frame_jitter;
    prm.misc2[0] = scatter;
    prm.misc2[1] = float(dbg);
    prm.misc2[2] = float(nl);
    prm.misc2[3] = float(shadow_cmp);
    prm.misc3[1] = lights_scale;
    prm.misc3[2] = 2.0f * far_dist / float(gd);
    if (game_fog_have && game_fog_weight > 0.0f) {
        prm.gfog[0] = std::clamp(game_fog.amount, 0.0f, 1.0f) * game_fog_mul;
        prm.gfog[1] = std::clamp(game_fog.rate, 0.0f, 10.0f);
        prm.gfog[2] = std::clamp(game_fog.start, -1000.0f, 10000.0f);
        prm.gfog[3] = game_fog_weight;
    }
    {
        // Moon light inside the game's fog, in its own units: the fog colour (v1.rgb, rgb^2) as
        // GXLightComposite mixes it with the scene luma (v1.w), half way to the moon's hue.
        const float fl = Lum(game_fog.col);
        const float ls = scene_lum_log > 0.0f ? scene_lum_log : 0.05f;
        const float base = fl * std::max(1.0f + game_fog.mix * (ls - 1.0f), 0.05f);
        const float ml = std::max(Lum(moon_col), 1e-4f);
        for (int i = 0; i < 3; ++i) {
            const float hue = game_fog_have && fl > 1e-6f ? game_fog.col[i] / fl : 1.0f;
            prm.gfog_col[i] = base * (0.5f * hue + 0.5f * moon_col[i] / ml);
        }
        prm.gfog_col[3] = shaft_scale;
    }
    prm.expo[0] = cap_rel;
    prm.expo[1] = light_cap_rel;
    prm.expo[2] = std::fabs(P[0]) > 1e-6 ? float(2.0 / (std::fabs(P[0]) * gw)) : 0.01f;
    // Layer drift shares (ground 1, mid 0.6) and the high band; the band is own medium only.
    prm.layer[0][0] = 1.0f;
    prm.layer[1][0] = 0.6f;
    prm.layer[2][0] = high_dens * own;
    prm.layer[2][1] = high_h;
    prm.layer[2][2] = high_thick;
    prm.layer[2][3] = -0.4f;
    prm.layer[3][0] = edge_k;
    std::memcpy(static_cast<u8*>(params.mapped) + slot * ParamSize, &prm, sizeof(prm));
    app_slot_hist[app_hist_pos++ % app_slot_hist.size()] = slot;

    if (queries) {
        cmd.resetQueryPool(queries, slot * 4 + 2, 2);
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, queries, slot * 4 + 2);
        app_used[slot] = true;
    }
    Transitions tr;
    tr.Add(in.hdr, in.hdr_layout, vk::ImageAspectFlagBits::eColor, true);
    tr.Add(in.depth, in.depth_layout, DepthAspect(in.depth_stencil), false);
    tr.Record(cmd, true);
    InitImages(cmd);
    cmd.fillBuffer(counters.buffer, slot * 64, 64, 0);
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &AllBefore});
    const vk::DescriptorBufferInfo cnt_info{counters.buffer, slot * 64, 64};
    const u32 cur = u32(frame & 1), prv = cur ^ 1u;
    const vk::DescriptorBufferInfo prm_info{params.buffer, slot * ParamSize, ParamSize};
    const vk::DescriptorBufferInfo light_info{
        lights_buf.buffer, slot * sizeof(GpuLight) * MaxLights, sizeof(GpuLight) * MaxLights};
    const vk::DescriptorImageInfo vis_tex{nearest, vis.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo hist_tex{linear, scat[prv].view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo scat_img{{}, scat[cur].view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo scat_tex{nearest, scat[cur].view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo integ_img{{}, integ.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo integ_tex{linear, integ.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo depth_tex{
        nearest, in.depth_view,
        Sampleable(in.depth_layout) ? in.depth_layout : vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo hdr_img{{}, in.hdr_storage_view, vk::ImageLayout::eGeneral};
    using DT = vk::DescriptorType;
    const auto ubo = vk::WriteDescriptorSet{.dstBinding = 0,
                                            .descriptorCount = 1,
                                            .descriptorType = DT::eUniformBuffer,
                                            .pBufferInfo = &prm_info};
    const auto cnt_w = vk::WriteDescriptorSet{.dstBinding = 6,
                                              .descriptorCount = 1,
                                              .descriptorType = DT::eStorageBuffer,
                                              .pBufferInfo = &cnt_info};
    const auto hdr_w = vk::WriteDescriptorSet{.dstBinding = 12,
                                              .descriptorCount = 1,
                                              .descriptorType = DT::eStorageImage,
                                              .pImageInfo = &hdr_img};
    // 0. The scene's log-average luminance before the fog (caps the added light).
    {
        const std::array<vk::WriteDescriptorSet, 3> wr{{ubo, cnt_w, hdr_w}};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, in.hdr_r11 ? pipes[6] : pipes[5]);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, wr);
        cmd.dispatch(8, 5, 1);
    }
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &ComputeToCompute});
    // 1. Light.
    {
        const std::array<vk::WriteDescriptorSet, 6> wr{{
            ubo,
            cnt_w,
            {.dstBinding = 3,
             .descriptorCount = 1,
             .descriptorType = DT::eCombinedImageSampler,
             .pImageInfo = &vis_tex},
            {.dstBinding = 4,
             .descriptorCount = 1,
             .descriptorType = DT::eCombinedImageSampler,
             .pImageInfo = &hist_tex},
            {.dstBinding = 5,
             .descriptorCount = 1,
             .descriptorType = DT::eStorageImage,
             .pImageInfo = &scat_img},
            {.dstBinding = 7,
             .descriptorCount = 1,
             .descriptorType = DT::eStorageBuffer,
             .pBufferInfo = &light_info},
        }};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipes[1]);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, wr);
        cmd.dispatch((gw + 7) / 8, (gh + 7) / 8, gd);
    }
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &ComputeToCompute});
    // 2. Integrate.
    {
        const std::array<vk::WriteDescriptorSet, 3> wr{{
            ubo,
            {.dstBinding = 8,
             .descriptorCount = 1,
             .descriptorType = DT::eCombinedImageSampler,
             .pImageInfo = &scat_tex},
            {.dstBinding = 9,
             .descriptorCount = 1,
             .descriptorType = DT::eStorageImage,
             .pImageInfo = &integ_img},
        }};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipes[2]);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, wr);
        cmd.dispatch((gw + 7) / 8, (gh + 7) / 8, 1);
    }
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &ComputeToCompute});
    // 3. Apply (debug 2: the fog is computed but not applied, for an A/B).
    if (dbg != 2) {
        const std::array<vk::WriteDescriptorSet, 5> wr{{
            cnt_w,
            ubo,
            {.dstBinding = 10,
             .descriptorCount = 1,
             .descriptorType = DT::eCombinedImageSampler,
             .pImageInfo = &depth_tex},
            {.dstBinding = 11,
             .descriptorCount = 1,
             .descriptorType = DT::eCombinedImageSampler,
             .pImageInfo = &integ_tex},
            {.dstBinding = 12,
             .descriptorCount = 1,
             .descriptorType = DT::eStorageImage,
             .pImageInfo = &hdr_img},
        }};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, in.hdr_r11 ? pipes[4] : pipes[3]);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, wr);
        cmd.dispatch((in.hdr_w + 7) / 8, (in.hdr_h + 7) / 8, 1);
    }
    if (queries) {
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, queries, slot * 4 + 3);
    }
    tr.Record(cmd, false);
    if (tr.n == 0) {
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &AllBefore});
    }
    std::memcpy(prevV, V, sizeof(prevV));
    std::memcpy(prevP, P, sizeof(prevP));
    for (int k = 0; k < 3; ++k)
        prev_cam[k] = invV[k * 4 + 3];
    have_prev = true;
    last_apply_frame = frame;
    ++n_apply;
    Log(!logged_start);
    logged_start = true;
    return true;
}

// BB_VOLFOG_WIND_PROBE=1 (diagnostic, read only): the game's WindWorld (Ghidra, eboot 1.09).
// FUN_01d29220 creates it (0x98 bytes, constructor FUN_02111820, vtable 0x574c5e0) into the
// global at Ghidra 0x5940060 (vaddr 0x5540060); FUN_02112490 adds wind instances (0x110 bytes,
// constructor FUN_0210fc00, vtable 0x574c5a0) to its id-keyed tree at +0x70 (node: +0x19 nil
// flag, +0x20 id, +0x28 instance). Instance: +0x18 type (0..3), +0x20..+0x5f a 4x4 transform
// (identity by default), +0x60 the type's data (a point with w = 1 for types 0..2). How a wind
// vector is evaluated from these is not decoded: the fog does not use them (HIPOTESIS only).
std::string VolFog::WindProbe() {
    const uintptr_t base = MemoryPatcher::g_eboot_address;
    if (!base)
        return "eboot base unknown";
    auto* mem = Core::Memory::Instance();
    const auto rd = [&](u64 a, auto& v) {
        if (a < 0x10000 || !mem->IsValidMapping(a, sizeof(v)))
            return false;
        std::memcpy(&v, reinterpret_cast<const void*>(a), sizeof(v));
        return true;
    };
    u64 ww = 0, head = 0, root = 0;
    if (!rd(base + 0x5540060, ww) || !ww)
        return "WindWorld not created";
    if (!rd(ww + 0x70, head) || !rd(head + 8, root))
        return "WindWorld tree unreadable";
    std::array<u32, 5> types{};
    std::string first;
    u64 stack[64];
    int sp = 0, visited = 0;
    stack[sp++] = root;
    while (sp > 0 && visited < 64) {
        const u64 node = stack[--sp];
        u8 nil = 1;
        if (!rd(node + 0x19, nil) || nil)
            continue;
        ++visited;
        u64 inst = 0, l = 0, r = 0;
        rd(node + 0x28, inst);
        u32 type = 4;
        float m[20]{};
        if (inst && rd(inst + 0x18, type) && rd(inst + 0x20, m)) {
            ++types[std::min<u32>(type, 4)];
            if (first.empty()) {
                first = fmt::format(
                    "first: type {} row2 ({:.2f} {:.2f} {:.2f}) row3 ({:.1f} {:.1f} "
                    "{:.1f}) data ({:.2f} {:.2f} {:.2f} {:.2f})",
                    type, m[8], m[9], m[10], m[12], m[13], m[14], m[16], m[17], m[18], m[19]);
            }
        }
        if (rd(node, l) && sp < 63)
            stack[sp++] = l;
        if (rd(node + 0x10, r) && sp < 63)
            stack[sp++] = r;
    }
    return fmt::format("WindWorld {:#x}: {} instances (types 0/1/2/3: {}/{}/{}/{}){}{}", ww,
                       visited, types[0], types[1], types[2], types[3], first.empty() ? "" : "; ",
                       first);
}

void VolFog::Log(bool force) {
    const auto now = std::chrono::steady_clock::now();
    if (!force && last_log != std::chrono::steady_clock::time_point{} &&
        now - last_log < std::chrono::seconds(5)) {
        return;
    }
    const double secs = last_log == std::chrono::steady_clock::time_point{}
                            ? 0.0
                            : std::chrono::duration<double>(now - last_log).count();
    last_log = now;
    // Counters of the inject from a few runs ago (finished on the GPU).
    const auto* cmap = static_cast<const u8*>(counters.mapped);
    const u32* cn = reinterpret_cast<const u32*>(
        cmap + inj_slot_hist[inj_hist_pos % inj_slot_hist.size()] * 64);
    const u32 tested = cn[0], lit = cn[1], outside = cn[2];
    const u32* ca = reinterpret_cast<const u32*>(
        cmap + app_slot_hist[app_hist_pos % app_slot_hist.size()] * 64);
    if (ca[9] > 0) {
        scene_lum_log = std::exp2(float(int(ca[8])) / (256.0f * float(ca[9])));
    }
    const std::string gfog =
        game_fog_have
            ? fmt::format("amount {:.3f}, 1/dist {:.4f} (dist {:.1f} m), start {:.1f} m, colour "
                          "({:.4f}, {:.4f}, {:.4f}) luma-mix {:.2f}, height fog amount {:.3f} base "
                          "{:.1f} scale {:.4f}; weight {:.2f}; frames with fog {} / without {}; "
                          "this frame: {} fog CBs, {} sets, {} fog-off",
                          game_fog.amount, game_fog.rate,
                          game_fog.rate > 0.0f ? 1.0f / game_fog.rate : 0.0f, game_fog.start,
                          game_fog.col[0], game_fog.col[1], game_fog.col[2], game_fog.mix,
                          game_fog.h_amount, game_fog.h_base, game_fog.h_scale, game_fog_weight,
                          n_fog_frames, n_nofog_frames, frame_fog_cbs, frame_fog_sets,
                          frame_fog_off)
            : fmt::format("none seen (frames without {}; this frame {} fog CBs, {} fog-off)",
                          n_nofog_frames, frame_fog_cbs, frame_fog_off);
    std::string shadow;
    if (casc_ok) {
        shadow = fmt::format(
            "game cascade atlas {:#x} {}x{} (ps {:#x}; matrices {} {}, {}, {}, select {}, compare "
            "{}{}; "
            "fit {:.0f}%), lit {:.0f}% of {} tested froxels ({} outside the cascades)",
            atlas_addr, atlas_w, atlas_h, inject_ps, casc_combo < 2 ? "proj-space" : "view-space",
            casc_combo & 1 ? "transposed" : "rows", casc_ndc ? "NDC out" : "uv out",
            casc_baked ? "tiles in the matrices" : "2x2 quadrants (HIPOTESIS)",
            casc_dist_ok ? fmt::format("by FC_CascadeSelectDist {:.1f}/{:.1f}/{:.1f}/{:.1f} m",
                                       cdist[0], cdist[1], cdist[2], cdist[3])
                         : std::string("by range"),
            shadow_cmp ? "z >= d" : "z <= d", shadow_cmp_env >= 0 ? " (BB_VOLFOG_SHADOW_CMP)" : "",
            casc_score * 100.0f, tested ? 100.0 * lit / tested : 0.0, tested, outside);
    } else {
        shadow = fmt::format("none ({}): moon unshadowed at x{:.2f}", casc_why, unshadowed_moon);
    }
    const double inj = inj_n ? inj_ms / inj_n : 0.0, app = app_n ? app_ms / app_n : 0.0;
    LOG_INFO(Render_Vulkan,
             "[volfog] {}{}: BB_VOLFOG={} ({}), froxels {}x{}x{} to {:.0f} m; fog applied {} "
             "(inject {}, frames without inject {}), skipped {} (last: {}); GPU {:.3f} ms "
             "(inject {:.3f} + light/integrate/apply {:.3f}); moon {} dir ({:.2f}, {:.2f}, {:.2f}) "
             "[{}] colour ({:.2f}, {:.2f}, {:.2f}) [{}]; shadow: {}; game fog: {}; scene log-avg "
             "luminance {:.4f} (added light cap x{:.2f}); point lights {} lamp-like (max {}, avg "
             "{:.1f}; rejected: radius > {:.0f} m {}, cold / fill {}; light draws {}, parsed {}, "
             "with cbLightAccParam {}; x{:.2f}); ambient [{}]; wind "
             "({:.2f}, {:.2f}, {:.2f}) m/s [{}]; history restarts {} (last: {}); HDR reader ps "
             "{:#x}{}{}",
             force ? "start" : "5 s", force ? "" : fmt::format(" ({:.1f} s)", secs), level,
             level_name, gw, gh, gd, far_dist, n_apply, n_inject, n_noinject, n_skip, last_skip,
             inj + app, inj, app,
             !moon_on       ? "no"
             : !shadow_pass ? "no (no shadow-mask pass this frame: interior)"
                            : "yes",
             moon_dir[0], moon_dir[1], moon_dir[2], moon_src, moon_col[0], moon_col[1], moon_col[2],
             col_src, shadow, gfog, scene_lum_log, cap_rel, frame_lights.size(), lights_max,
             n_apply ? double(lights_sum) / double(n_apply) : 0.0, light_max_r, frame_rej_big,
             frame_rej_cold, frame_draws, frame_parsed, frame_cam_cb, lights_scale, amb_src,
             wind[0], wind[1], wind[2], wind_src, hist_resets, hist_why, last_hook_ps,
             wind_probe ? "; wind probe: " : "", wind_probe ? WindProbe() : "");
    n_apply = n_inject = n_noinject = n_skip = 0;
    n_fog_frames = n_nofog_frames = 0;
    inj_ms = app_ms = 0.0;
    inj_n = app_n = 0;
    lights_max = 0;
    lights_sum = 0;
    hist_resets = 0;
}

} // namespace BB
