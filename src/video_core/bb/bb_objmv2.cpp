// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// BB object motion vectors v2: see bb_objmv2.h.

#include "video_core/bb/bb_objmv2.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "core/memory.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/bb/bb_native_mv.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/host_shaders/bb_objmv2_comp.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include <vk_mem_alloc.h>
#include <xxhash.h>

namespace BB::ObjMv2 {

namespace {

constexpr u32 kSlots = 4;                       // frames in flight (snapshots, tables, stats)
constexpr u64 kArenaBytes = 24ull << 20;        // snapshots a frame
constexpr u64 kTableBytes = 2ull << 20;         // table entries a frame
constexpr u64 kHeaderBytes = 4096;              // zero entry + dummy dword
constexpr u32 kMaxSnapBytes = 64u << 10;        // larger buffers: read the current one
constexpr u32 kStatWords = 16; // word 11: attachment 7 texels != 0; 12-15: the player mask (bb-player-mv)
constexpr u64 kStatPeriod = 60;                 // statistics / check frame: 1 in 60
constexpr u32 kStaticSample = 4;                // check frames: 1 static draw in 4 checked

enum Cat : u32 { CatSkinned = 0, CatSmallSkel, CatRigid, CatBlend, CatCount };
constexpr const char* kCatName[CatCount] = {"skinned(characters+enemies)",
                                            "small-skeleton(weapons/props)", "rigid",
                                            "blended(particles/effects)"};

struct SnapBuf {
    u64 addr;    // device address of the snapshot (0: none)
    u32 dwords;
    u32 bytes;   // guest size (0: not hashed)
    u64 hash;    // content hash (moving-draw test)
    u64 chash = 0;    // bb-gpu6: content hash of the guest buffer (in-frame dedupe)
    u32 rel_off = 0;  // bb-gpu6: relative bone transforms in rel_cur / rel_prev (12 floats each)
    u32 rel_n = 0;    // matrix pairs (0: none)
};

struct Rec {
    u32 first; // index into bufs
    u32 count;
};

struct State {
    bool tried = false, ready = false;
    bool check = true;
    vk::Device device{};
    VmaAllocator allocator{};
    // Host-visible, coherent, device address: header | kSlots x (arena | table).
    vk::Buffer buf{};
    VmaAllocation buf_alloc{};
    u8* mapped = nullptr;
    u64 base_addr = 0;
    // Stats (host visible): kSlots x kStatWords.
    vk::Buffer stats{};
    VmaAllocation stats_alloc{};
    u32* stats_mapped = nullptr;
    vk::QueryPool queries{};
    std::array<bool, kSlots> slot_used{};
    // Compose pass.
    vk::Sampler sampler{};
    vk::DescriptorSetLayout set_layout{};
    vk::PipelineLayout layout{};
    vk::Pipeline pipeline{};
    // Attachment 7 and the check's previous colours (ping-pong).
    vk::Image image{};
    VmaAllocation image_alloc{};
    vk::ImageView view{};
    vk::Extent2D last_render{}; // (11-13) the last composed render size
    std::array<vk::Image, 2> prev{};
    std::array<VmaAllocation, 2> prev_alloc{};
    std::array<vk::ImageView, 2> prev_view{};
    u32 prev_cur = 0;
    bool prev_valid = false;
    u32 width = 0, height = 0;
    bool cleared = false;  // attachment 7 cleared this frame (a motion pass happened)
    bool composed = false; // composed this frame
    // Frame state.
    u64 frame = 0;
    u32 slot = 0;
    u64 arena_used = 0, table_used = 0;
    u64 scene_depth = 0;
    std::vector<SnapBuf> bufs_cur, bufs_prev;
    std::unordered_map<u64, Rec> map_cur, map_prev;
    std::unordered_map<u64, u32> occ;
    std::unordered_map<u64, SnapBuf> dedupe;
    // bb-gpu6: relative transforms of consecutive 3x4 matrices (M[i]^-1 * M[i+1]) of each hashed
    // buffer, this frame and the last: unchanged by a transform common to all of them (the
    // camera, when the game multiplies it in), changed by animation.
    std::vector<float> rel_cur, rel_prev;
    bool rel_test = true;
    u64 sec_camera_only = 0, sec_dedupe = 0, sec_bytes_hashed = 0, sec_bytes_copied = 0;
    bool history = false; // map_prev is exactly the previous frame
    bool reset_pending = false;
    // Per-second statistics.
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    u64 sec_draws = 0, sec_vec_draws = 0, sec_resets = 0, sec_frames = 0, sec_overflow = 0;
    u64 sec_cat[CatCount]{}, sec_cat_vec[CatCount]{};
    double sec_cpu_ns = 0;
    u64 sec_px_obj = 0, sec_px_blend = 0, sec_px_all = 0, sec_chk = 0, sec_eo = 0, sec_ec = 0,
        sec_wo = 0, sec_wc = 0;
    double sec_gpu_ms = 0, sec_gpu_stat_ms = 0;
    u32 sec_gpu_n = 0, sec_gpu_stat_n = 0;
    u64 sec_static_px = 0, sec_static_abs = 0, sec_static_zero = 0, sec_diag_draws = 0,
        sec_skipped = 0;
    // Mode: the snapshots hold no camera (static draws give ~0): add the camera vector.
    bool additive = true;
    bool stat_frame = false;   // this frame: statistics + check
    u64 color_stored = ~0ull;  // frame whose colour is in prev[0]
    u32 pp = 0;                // (bb-player-mv) prev image read this frame (the other one is written)
    u64 sec_pl_px = 0, sec_pl_changed = 0, sec_pl_before = 0, sec_pl_after = 0;
    u32 frame_vec_draws = 0, static_seen = 0;
    std::array<bool, kSlots> slot_stat{};
    double ts_period = 1.0;
    u64 sec_gpu_bufs = 0;
    u64 sec_nonzero = 0, sec_stat_frames = 0, sec_not_composed = 0;
    bool logged_nonzero = false, logged_not_composed = false;
    // bb-velocity: native velocity buffer. Snapshots of this frame by (address, size) and by game
    // identity (ring palette slot, camera block ordinal); GPU-written buffers copied on the GPU.
    bool velocity = true;
    std::unordered_map<u64, SnapBuf> vel_addr_cur, vel_addr_prev, vel_id_cur, vel_id_prev;
    std::unordered_map<u64, u32> cam_ord; // camera block (address, size) -> ordinal this frame
    std::unordered_map<u64, u64> role_cache; // vs hash -> read-only V# handles
    std::unordered_map<u64, bool> cam_checked; // camera block -> matched by order, similar
    std::unordered_map<u64, BB::NativeMv::DrawId> pal_ids; // palette address -> game identity
    u64 vs_cam_rejected = 0, vs_cam_unchecked = 0;
    std::unordered_map<u32, std::array<float, 216>> cam_cpu_cur, cam_cpu_prev; // by order
    // Palettes without a game identity whose address changes every frame (double-buffered):
    // matched by their order among the frame's palettes of that vertex shader and size, accepted
    // only when the first 64 floats still look like last frame's (CPU copies of guest memory).
    std::unordered_map<u64, u64> pal_ord_of;  // (address, size) -> order key this frame
    std::unordered_map<u64, u32> pal_ord_n;   // (vs hash, size) -> palettes seen this frame
    std::unordered_map<u64, bool> pal_checked; // order key -> similar to last frame's
    std::unordered_map<u64, std::array<float, 64>> pal_cpu_cur, pal_cpu_prev; // by order key
    u64 vs_pal_ord = 0, vs_pal_rejected = 0, vs_pal_unchecked = 0;
    // bb-skin-mv: ring palettes whose slot key had no copy last frame, matched by their order
    // among the same game skin ring's palettes (game identity: no data check needed); palettes
    // the GPU wrote (no CPU check possible) accepted by order when their address offset from the
    // first palette of their group is the same as last frame (same buffer layout).
    std::unordered_map<u64, u64> ring_ord_of; // (address, size) -> ring order key this frame
    std::unordered_map<u64, u32> ring_ord_n;  // ring key -> palettes seen this frame
    std::unordered_map<u64, u64> pal_first;   // group -> first palette address this frame
    std::unordered_map<u64, u64> pal_rel_cur, pal_rel_prev; // order key -> address - first
    std::unordered_map<u64, u32> pal_n_prev;  // group -> palettes seen last frame
    u64 vs_ring_ord = 0, vs_pal_layout = 0, vs_skin_draws = 0, vs_skin_draws_all = 0;
    // bb-skin-mv2: why a palette buffer had no previous copy (per buffer, 1 s), and draws whose
    // other palette buffers matched (the unmatched one reads its current data).
    std::array<u64, 8> vs_why{};
    u64 vs_pal_partial = 0, vs_multi_pal = 0, vs_mesh = 0;
    std::unordered_map<u64, u32> mesh_occ; // mesh draw -> occurrences this frame
    std::unordered_map<u64, u8> ring_seen_cur, ring_seen_prev; // ring keys drawn this / last frame
    bool skin_order = true; // BB_SKIN_MV=0: off (previous behaviour)
    u64 vs_frames = 0, vs_draws = 0, vs_native = 0, vs_static = 0, vs_skinned = 0,
        vs_skinned_ring = 0, vs_skinned_addr = 0, vs_no_cam = 0, vs_no_pal = 0, vs_blend = 0,
        vs_gpu_copies = 0, vs_bytes = 0, vs_cam_addr = 0, vs_cam_ord = 0, vs_ring_seen = 0;
    double vs_cpu_ns = 0;
} g;

void FlushVelocityCopies(vk::CommandBuffer cmd); // bb-velocity (below)

// Statistics / check frame: 1 in 60, or 1 in 30 below 75 fps (at least one a second at 30 fps).
u64 g_stat_period = kStatPeriod;
bool IsStatFrame(u64 frame) {
    return (frame % g_stat_period) == 0;
}

bool EnvOff(const char* name) {
    const char* s = std::getenv(name);
    return s != nullptr && (s[0] == '0' || std::string(s) == "off");
}

void Log(const std::string& text) {
    LOG_INFO(Render_Vulkan, "[objmv2] {}", text);
    BB::Upscale::Checkpoint("[objmv2] " + text);
}

u64 Mix(u64 h, u64 v) {
    h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    return h;
}

void DestroyImages() {
    if (g.view) g.device.destroyImageView(g.view);
    if (g.image) vmaDestroyImage(g.allocator, static_cast<VkImage>(g.image), g.image_alloc);
    g.view = vk::ImageView{};
    g.image = vk::Image{};
    for (u32 i = 0; i < 2; ++i) {
        if (g.prev_view[i]) g.device.destroyImageView(g.prev_view[i]);
        if (g.prev[i]) vmaDestroyImage(g.allocator, static_cast<VkImage>(g.prev[i]), g.prev_alloc[i]);
        g.prev_view[i] = vk::ImageView{};
        g.prev[i] = vk::Image{};
    }
    g.width = g.height = 0;
}

bool CreateImage(u32 w, u32 h, vk::ImageUsageFlags usage, vk::Image& image, VmaAllocation& alloc,
                 vk::ImageView& view) {
    const VkImageCreateInfo ci = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .extent = {w, h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkImage raw{};
    if (vmaCreateImage(g.allocator, &ci, &aci, &raw, &alloc, nullptr) != VK_SUCCESS) {
        return false;
    }
    image = raw;
    auto [result, v] = g.device.createImageView(vk::ImageViewCreateInfo{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
    if (result != vk::Result::eSuccess) return false;
    view = v;
    return true;
}

bool EnsureImages(u32 w, u32 h) {
    if (g.image && g.width == w && g.height == h) return true;
    if (g.image) {
        (void)g.device.waitIdle();
        DestroyImages();
    }
    if (!CreateImage(w, h,
                     vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                     g.image, g.image_alloc, g.view)) {
        DestroyImages();
        return false;
    }
    for (u32 i = 0; i < 2; ++i) {
        if (!CreateImage(w, h, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
                         g.prev[i], g.prev_alloc[i], g.prev_view[i])) {
            DestroyImages();
            return false;
        }
    }
    g.width = w;
    g.height = h;
    g.cleared = false;
    g.prev_valid = false;
    // Resolution change: no history.
    g.map_prev.clear();
    g.history = false;
    ++g.sec_resets;
    Log(fmt::format("vector image {}x{} (RGBA16F), history reset", w, h));
    return true;
}

void Barrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout old_layout,
             vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
             vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
}

u64 SlotArenaOffset(u32 slot) {
    return kHeaderBytes + u64(slot) * (kArenaBytes + kTableBytes);
}
u64 SlotTableOffset(u32 slot) {
    return SlotArenaOffset(slot) + kArenaBytes;
}

void ReadStats(u32 slot) {
    if (!g.slot_used[slot]) return;
    g.slot_used[slot] = false;
    const u32* s = g.stats_mapped + slot * kStatWords;
    g.sec_px_obj += s[0];
    g.sec_px_blend += s[1];
    g.sec_px_all += s[2];
    g.sec_chk += s[3];
    g.sec_eo += s[4];
    g.sec_ec += s[5];
    g.sec_wo += s[6];
    g.sec_wc += s[7];
    g.sec_static_px += s[8];
    g.sec_static_abs += s[9];
    g.sec_static_zero += s[10];
    g.sec_pl_px += s[12];
    g.sec_pl_changed += s[13];
    g.sec_pl_before += s[14];
    g.sec_pl_after += s[15];
    if (g.slot_stat[slot]) {
        g.sec_nonzero += s[11];
        ++g.sec_stat_frames;
    }
    if (g.slot_stat[slot] && !g.logged_nonzero) {
        g.logged_nonzero = true;
        Log(fmt::format("attachment 7: {} non-zero texels in the first statistics frame", s[11]));
    }
    std::array<u64, 2> ts{};
    const auto r = g.device.getQueryPoolResults(g.queries, slot * 2, 2, sizeof(ts), ts.data(),
                                                sizeof(u64), vk::QueryResultFlagBits::e64);
    if (r == vk::Result::eSuccess && ts[1] >= ts[0]) {
        const double ms = double(ts[1] - ts[0]) * g.ts_period * 1e-6;
        if (g.slot_stat[slot]) {
            g.sec_gpu_stat_ms += ms;
            ++g.sec_gpu_stat_n;
        } else {
            g.sec_gpu_ms += ms;
            ++g.sec_gpu_n;
        }
    }
}

} // namespace

bool Init(const Vulkan::Instance& instance) {
    if (g.tried) return g.ready;
    g.tried = true;
    if (EnvOff("BB_OBJMV2")) {
        Log("off (BB_OBJMV2=0): camera-only motion vectors (bb_mv.comp)");
        return false;
    }
    if (!instance.IsBufferDeviceAddressSupported()) {
        Log("off: buffer device address not supported");
        return false;
    }
    g.check = !EnvOff("BB_OBJMV2_CHECK");
    {
        const char* mv = std::getenv("BB_OBJMV_MOVE");
        g.rel_test = !(mv != nullptr && std::strcmp(mv, "hash") == 0);
    }
    g.device = instance.GetDevice();
    g.allocator = instance.GetAllocator();
    g.ts_period = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
    const auto make_buffer = [&](vk::DeviceSize size, vk::BufferUsageFlags usage, vk::Buffer& b,
                                 VmaAllocation& a, void** mapped, u64* address) {
        const VkBufferCreateInfo ci = vk::BufferCreateInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        };
        VmaAllocationCreateInfo aci{};
        aci.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT |
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkBuffer raw{};
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(g.allocator, &ci, &aci, &raw, &a, &info) != VK_SUCCESS) {
            return false;
        }
        b = raw;
        *mapped = info.pMappedData;
        if (address) {
            *address = g.device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = b});
            if (*address == 0) return false;
        }
        return *mapped != nullptr;
    };
    void* m = nullptr;
    void* sm = nullptr;
    if (!make_buffer(kHeaderBytes + kSlots * (kArenaBytes + kTableBytes),
                     vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eShaderDeviceAddress |
                         vk::BufferUsageFlagBits::eTransferDst,
                     g.buf, g.buf_alloc, &m, &g.base_addr) ||
        !make_buffer(kSlots * kStatWords * 4,
                     vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eTransferDst,
                     g.stats, g.stats_alloc, &sm, nullptr)) {
        Log("off: buffers not created");
        return false;
    }
    g.mapped = static_cast<u8*>(m);
    g.stats_mapped = static_cast<u32*>(sm);
    std::memset(g.mapped, 0, kHeaderBytes);
    std::memset(g.stats_mapped, 0, kSlots * kStatWords * 4);
    // Zero entry: every handle reads its current buffer (use 0), dummy dword at 2048.
    for (u32 h = 0; h < Shader::NUM_BUFFERS; ++h) {
        u32* e = reinterpret_cast<u32*>(g.mapped + h * 16);
        const u64 dummy = g.base_addr + 2048;
        e[0] = u32(dummy);
        e[1] = u32(dummy >> 32);
        e[2] = 1;
        e[3] = 0;
    }
    auto [qr, qp] = g.device.createQueryPool(vk::QueryPoolCreateInfo{
        .queryType = vk::QueryType::eTimestamp, .queryCount = kSlots * 2});
    if (qr != vk::Result::eSuccess) {
        Log("off: query pool");
        return false;
    }
    g.queries = qp;

    auto [sr, sampler] = g.device.createSampler(vk::SamplerCreateInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .maxLod = 0.0f,
    });
    g.sampler = sampler;
    const auto bind = [](u32 b, vk::DescriptorType t) {
        return vk::DescriptorSetLayoutBinding{.binding = b,
                                              .descriptorType = t,
                                              .descriptorCount = 1,
                                              .stageFlags = vk::ShaderStageFlagBits::eCompute};
    };
    const std::array bindings{
        bind(0, vk::DescriptorType::eCombinedImageSampler),
        bind(1, vk::DescriptorType::eStorageImage),
        bind(2, vk::DescriptorType::eStorageBuffer),
        bind(3, vk::DescriptorType::eCombinedImageSampler),
        bind(4, vk::DescriptorType::eCombinedImageSampler),
        bind(5, vk::DescriptorType::eStorageImage),
    };
    auto [lr, set_layout] = g.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    });
    const vk::PushConstantRange push{
        .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = 16};
    auto [plr, layout] = g.device.createPipelineLayout(vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push,
    });
    const auto module = Vulkan::CompileSPV(BB_OBJMV2_COMP, g.device);
    auto [pr, pipeline] = g.device.createComputePipeline(
        {}, vk::ComputePipelineCreateInfo{
                .stage = vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eCompute,
                                                           .module = module,
                                                           .pName = "main"},
                .layout = layout,
            });
    g.device.destroyShaderModule(module);
    if (sr != vk::Result::eSuccess || lr != vk::Result::eSuccess ||
        plr != vk::Result::eSuccess || pr != vk::Result::eSuccess) {
        Log("off: compose pass not created");
        return false;
    }
    g.set_layout = set_layout;
    g.layout = layout;
    g.pipeline = pipeline;
    g.bufs_cur.reserve(16384);
    g.bufs_prev.reserve(16384);
    g.map_cur.reserve(8192);
    g.map_prev.reserve(8192);
    g.occ.reserve(8192);
    g.dedupe.reserve(8192);
    g.velocity = !EnvOff("BB_VELOCITY");
    g.skin_order = !EnvOff("BB_SKIN_MV");
    if (g.velocity) {
        g.additive = false; // absolute vectors: the previous camera is in the VS
        Vulkan::Scheduler::bb_end_rendering_hook = &FlushVelocityCopies;
    }
    g.ready = true;
    Log(g.velocity ? "[velocity] on (bb-clean-velocity): every opaque G-buffer draw writes its own "
                     "velocity in its vertex shader (previous 864 B camera block, previous skin "
                     "ring palette), bb_mv.comp only where none is written; BB_VELOCITY=0 off"
                   : "[velocity] off (BB_VELOCITY=0)");
    Log(fmt::format("on: scene draws run their vertex shader once more with last frame's buffers "
                    "(GPU), vectors in attachment 7, composed over bb_mv; snapshot arena {} MiB x "
                    "{}, check {}",
                    kArenaBytes >> 20, kSlots, g.check ? "on" : "off (BB_OBJMV2_CHECK=0)"));
    return true;
}

bool Active() {
    return g.ready;
}

void NoteSceneDepth(u64 depth_address) {
    g.scene_depth = depth_address;
}

Kind Want(const AmdGpu::Regs& regs) {
    if (!g.ready || g.scene_depth == 0) return Kind::None;
    if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::VgtStages::Vs) return Kind::None;
    if (!regs.depth_buffer.DepthValid() || regs.depth_buffer.NumSamples() != 1) return Kind::None;
    if (regs.depth_buffer.DepthAddress() != g.scene_depth) return Kind::None;
    if (regs.color_control.mode != AmdGpu::ColorControl::OperationMode::Normal) return Kind::None;
    u32 bound = 0;
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        if (regs.color_buffers[cb] && regs.color_target_mask.GetMask(cb)) {
            if (cb == Attachment) return Kind::None;
            ++bound;
        }
    }
    if (bound == 0) return Kind::None;
    // Other draws of the scene pass: attachment 7 kept bound, not written (no pass restart).
    if (!regs.depth_control.depth_enable || regs.IsClipDisabled() || !regs.color_buffers[0] ||
        !regs.color_target_mask.GetMask(0)) {
        return Kind::Mask;
    }
    const auto& bc = regs.blend_control[0];
    if (!bc.enable) {
        return regs.depth_control.depth_write_enable ? Kind::Opaque : Kind::Mask;
    }
    // Alpha-blended (particles, effects) only: additive / multiplicative passes (lights, glow)
    // have no surface of their own.
    using F = AmdGpu::BlendControl::BlendFactor;
    if (bc.color_src_factor == F::SrcAlpha && bc.color_dst_factor == F::OneMinusSrcAlpha) {
        return Kind::Blend;
    }
    return Kind::Mask;
}

void FillKey(Kind kind, u32& mrt_mask, u32& num_color_attachments, Shader::PsColorBuffer& cb7,
             vk::ColorComponentFlags& write_mask7, u32& cb_shader_mask_raw, void* blend7) {
    mrt_mask |= 1u << Attachment;
    num_color_attachments = Attachment + 1;
    std::memset(&cb7, 0, sizeof(cb7));
    cb7.data_format = AmdGpu::DataFormat::Format16_16_16_16;
    cb7.num_format = AmdGpu::NumberFormat::Float;
    cb7.swizzle = AmdGpu::IdentityMapping;
    auto& bc = *static_cast<AmdGpu::BlendControl*>(blend7);
    std::memset(&bc, 0, sizeof(bc));
    if (kind == Kind::Mask) {
        write_mask7 = vk::ColorComponentFlags{};
        return;
    }
    write_mask7 = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                  vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    cb_shader_mask_raw |= 0xFu << (Attachment * 4);
    if (kind == Kind::Blend) {
        using F = AmdGpu::BlendControl::BlendFactor;
        bc.enable = 1;
        bc.color_src_factor = F::SrcAlpha;
        bc.color_dst_factor = F::OneMinusSrcAlpha;
        bc.color_func = AmdGpu::BlendControl::BlendFunc::Add;
        bc.separate_alpha_blend = 1;
        bc.alpha_src_factor = F::Zero;
        bc.alpha_dst_factor = F::One;
        bc.alpha_func = AmdGpu::BlendControl::BlendFunc::Add;
    }
}

bool IsMotionKey(u32 mrt_mask, const Shader::PsColorBuffer& cb7) {
    return g.ready && (mrt_mask & (1u << Attachment)) != 0 &&
           cb7.data_format == AmdGpu::DataFormat::Format16_16_16_16 &&
           cb7.num_format == AmdGpu::NumberFormat::Float;
}

void Attach(Vulkan::Scheduler& scheduler, Vulkan::RenderState& state) {
    if (!g.ready || state.num_color_attachments <= Attachment) return;
    const u32 w = std::max<u32>(BB::Upscale::g_scene_w, state.width);
    const u32 h = std::max<u32>(BB::Upscale::g_scene_h, state.height);
    if (!EnsureImages(w, h)) return;
    auto& att = state.color_attachments[Attachment];
    att.image_view = g.view;
    att.image_layout = vk::ImageLayout::eGeneral;
    att.clear_value = {0u, 0u, 0u, 0u};
    att.is_clear = 0;
    if (!g.cleared) {
        // First motion pass of the frame: last frame's compose sampled it.
        scheduler.EndRendering();
        Barrier(scheduler.CommandBuffer(), g.image, vk::ImageLayout::eUndefined,
                vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::AccessFlagBits2::eColorAttachmentWrite |
                    vk::AccessFlagBits2::eColorAttachmentRead);
        att.is_clear = 1;
        g.cleared = true;
    }
}

// bb-gpu6: M[i]^-1 * M[i+1] of the 3x4 matrices (rows [R | t], column vectors) in `src`.
// Returns false when a matrix is not invertible (not a matrix palette).
bool RelTransforms(const float* m, u32 count, std::vector<float>& out) {
    for (u32 i = 0; i + 1 < count; ++i) {
        const float* a = m + i * 12;
        const float* b = m + (i + 1) * 12;
        const double a00 = a[0], a01 = a[1], a02 = a[2], a10 = a[4], a11 = a[5], a12 = a[6],
                     a20 = a[8], a21 = a[9], a22 = a[10];
        const double c00 = a11 * a22 - a12 * a21, c01 = a02 * a21 - a01 * a22,
                     c02 = a01 * a12 - a02 * a11, c10 = a12 * a20 - a10 * a22,
                     c11 = a00 * a22 - a02 * a20, c12 = a02 * a10 - a00 * a12,
                     c20 = a10 * a21 - a11 * a20, c21 = a01 * a20 - a00 * a21,
                     c22 = a00 * a11 - a01 * a10;
        const double det = a00 * c00 + a01 * c10 + a02 * c20;
        if (!(std::fabs(det) > 1e-12) || !std::isfinite(det)) return false;
        const double inv[9] = {c00 / det, c01 / det, c02 / det, c10 / det, c11 / det,
                               c12 / det, c20 / det, c21 / det, c22 / det};
        const double dt[3] = {double(b[3]) - a[3], double(b[7]) - a[7], double(b[11]) - a[11]};
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                out.push_back(float(inv[r * 3 + 0] * b[0 * 4 + c] + inv[r * 3 + 1] * b[1 * 4 + c] +
                                    inv[r * 3 + 2] * b[2 * 4 + c]));
            }
            out.push_back(float(inv[r * 3 + 0] * dt[0] + inv[r * 3 + 1] * dt[1] +
                                inv[r * 3 + 2] * dt[2]));
        }
    }
    return true;
}

bool RelDiffer(const float* a, const float* b, u32 pairs) {
    for (u32 i = 0; i < pairs * 12; ++i) {
        const bool translation = (i % 4) == 3;
        const float tol = translation ? 2e-3f * (1.0f + std::fabs(a[i])) : 2e-3f;
        if (!(std::fabs(a[i] - b[i]) <= tol)) return true;
    }
    return false;
}

namespace {
// ---------------------------------------------------------------------------------------------
// bb-velocity (native velocity buffer, as a game with TAA writes it).
//
// Identification of a vertex shader's buffers (the table to extend; phase 2 adds rigid world
// transforms): the role comes from the V# the shader reads, no CPU scan of the draw.
//   864 B                                   camera block (view-projection): last frame's copy, by
//                                           address, else by its order among this frame's blocks
//   640..16384 B, multiple of 16, not 864   bone palette: last frame's copy of the same game skin
//                                           ring slot (BB::NativeMv), else same address + size
//   anything else                           current buffer (world transform: phase 2)
enum class VelRole : u8 { Current = 0, Camera = 1, Palette = 2 };

VelRole IdentifyVelocityBuffer(u32 size) {
    if (size == 864) return VelRole::Camera;
    if (size >= 640 && size <= 16384 && size % 16 == 0) return VelRole::Palette;
    return VelRole::Current;
}

// This frame's copy of a guest buffer (once per address, size and frame), always made by the GPU
// from the buffer cache (the skin rings' palettes and the camera blocks are written by the
// command processor: guest memory is stale for them). The copies are deferred to the end of the
// current rendering scope (Scheduler::bb_end_rendering_hook): no render pass is cut for them, so
// no attachment is loaded or cleared again. The data is read by the next frame only.
struct PendingCopy {
    vk::Buffer src;
    u64 src_off, dst_off, size;
};
std::vector<PendingCopy> g_pending;

void FlushVelocityCopies(vk::CommandBuffer cmd) {
    if (g_pending.empty()) return;
    const vk::MemoryBarrier2 before{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &before});
    for (const auto& c : g_pending) {
        cmd.copyBuffer(c.src, g.buf, vk::BufferCopy{c.src_off, c.dst_off, c.size});
    }
    const vk::MemoryBarrier2 after{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eVertexShader |
                        vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eHostRead,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &after});
    g.vs_gpu_copies += g_pending.size();
    g_pending.clear();
}

bool VelSnap(VideoCore::BufferCache& buffer_cache, Vulkan::Scheduler& scheduler, u64 addr,
             u32 size, SnapBuf& out) {
    const u64 key = Mix(addr, size);
    if (const auto it = g.vel_addr_cur.find(key); it != g.vel_addr_cur.end()) {
        out = it->second;
        return out.addr != 0;
    }
    SnapBuf sb{0, 0, 0, 0};
    const u64 bytes = (u64(size) + 15) & ~u64(15);
    const u64 off = SlotArenaOffset(g.slot) + g.arena_used;
    if (g.arena_used + bytes > kArenaBytes) {
        ++g.sec_overflow;
    } else if (const auto [buffer, buffer_off] = buffer_cache.ObtainBuffer(addr, size);
               buffer != nullptr) {
        g_pending.push_back({buffer->Handle(), buffer_off, off, size});
        sb.addr = g.base_addr + off;
        sb.dwords = size / 4;
        sb.bytes = size;
        g.arena_used += bytes;
        g.vs_bytes += size;
        if (!scheduler.IsRendering()) {
            FlushVelocityCopies(scheduler.CommandBuffer());
        }
    }
    g.vel_addr_cur[key] = sb;
    out = sb;
    return sb.addr != 0;
}

// Camera blocks matched by their order in the frame (not by address): accepted only when the
// block still looks like last frame's block of that order (most of its 216 floats within 5 %).
// Both sides are CPU copies of guest memory (864 B, once per block and frame; the GPU snapshot
// may still be in flight). A GPU-written block (guest memory stale) cannot be checked: no vector.
bool CameraCpuCopy(Core::MemoryManager* memory, VideoCore::BufferCache& buffer_cache, u64 addr,
                   std::array<float, 216>& out) {
    if (buffer_cache.IsRegionGpuModified(addr, 864) || !memory->IsValidMapping(addr, 864)) {
        return false;
    }
    std::memcpy(out.data(), reinterpret_cast<const void*>(addr), 864);
    return true;
}

bool PaletteSimilar(const std::array<float, 64>& a, const std::array<float, 64>& b) {
    u32 close = 0;
    for (u32 i = 0; i < 64; ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (a[i] == b[i] || d <= 0.05f * std::max(1.0f, std::fabs(a[i]))) ++close;
    }
    return close >= 64 * 3 / 4;
}

bool CameraSimilar(const std::array<float, 216>& a, const std::array<float, 216>& b) {
    u32 close = 0;
    for (u32 i = 0; i < 216; ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (a[i] == b[i] || d <= 0.05f * std::max(1.0f, std::fabs(a[i]))) ++close;
    }
    return close >= 216 * 3 / 4;
}

void VelocityDraw(Shader::PushData& push, const Shader::Info& vs, Core::MemoryManager* memory,
                  VideoCore::BufferCache& buffer_cache, Vulkan::Scheduler& scheduler, u64 vs_hash,
                  bool blend, u64 mesh) {
    const auto t0 = std::chrono::steady_clock::now();
    ++g.vs_draws;
    ++g.sec_draws;
    if (blend) {
        // Particles / effects: no vector of their own, bb_mv.comp stays under them.
        ++g.vs_blend;
        return;
    }
    const u32 n = static_cast<u32>(std::min<size_t>(vs.buffers.size(), Shader::NUM_BUFFERS));
    // Shader cache: the handles that are plain read-only V#s (the only ones that take a role).
    u64 cand = 0;
    if (const auto it = g.role_cache.find(vs_hash); it != g.role_cache.end()) {
        cand = it->second;
    } else {
        for (u32 h = 0; h < n; ++h) {
            if (!vs.buffers[h].IsSpecial() && !vs.buffers[h].is_written) cand |= 1ull << h;
        }
        g.role_cache[vs_hash] = cand;
    }
    std::array<SnapBuf, Shader::NUM_BUFFERS> prev{};
    std::array<bool, Shader::NUM_BUFFERS> use{};
    bool camera = false, ok = true, skinned = false, ring = false;
    u32 pal_matched = 0, pal_failed = 0, pal_bufs = 0;
    for (u32 h = 0; h < n && ok; ++h) {
        if ((cand & (1ull << h)) == 0) continue;
        const auto sharp = vs.buffers[h].GetSharp(vs);
        const u32 size = sharp.GetSize();
        const u64 addr = sharp.base_address;
        const VelRole role = IdentifyVelocityBuffer(size);
        if (role == VelRole::Current || addr == 0) continue;
        SnapBuf cur{0, 0, 0, 0};
        const bool have_cur = VelSnap(buffer_cache, scheduler, addr, size, cur);
        const u64 akey = Mix(addr, size);
        const SnapBuf* found = nullptr;
        if (role == VelRole::Camera) {
            // The camera block: same address last frame, else the same order among the frame's
            // camera blocks (the game places it anew every frame).
            u32 ord;
            if (const auto it = g.cam_ord.find(akey); it != g.cam_ord.end()) {
                ord = it->second;
            } else {
                ord = static_cast<u32>(g.cam_ord.size());
                g.cam_ord[akey] = ord;
                if (have_cur) g.vel_id_cur[Mix(0x63616d65ull, ord)] = cur;
                std::array<float, 216> c;
                if (CameraCpuCopy(memory, buffer_cache, addr, c)) g.cam_cpu_cur[ord] = c;
            }
            if (const auto it = g.vel_addr_prev.find(akey);
                it != g.vel_addr_prev.end() && it->second.addr != 0) {
                found = &it->second;
                ++g.vs_cam_addr;
            } else if (const auto it2 = g.vel_id_prev.find(Mix(0x63616d65ull, ord));
                       it2 != g.vel_id_prev.end() && it2->second.addr != 0) {
                // Per address and frame: the similarity check runs once per camera block.
                const auto chk = g.cam_checked.try_emplace(akey, false);
                if (chk.second) {
                    const auto c = g.cam_cpu_cur.find(ord);
                    const auto p = g.cam_cpu_prev.find(ord);
                    if (c == g.cam_cpu_cur.end() || p == g.cam_cpu_prev.end()) {
                        ++g.vs_cam_unchecked;
                    } else {
                        chk.first->second = CameraSimilar(c->second, p->second);
                    }
                }
                if (chk.first->second) {
                    found = &it2->second;
                    ++g.vs_cam_ord;
                } else {
                    ++g.vs_cam_rejected;
                }
            }
            if (found == nullptr) {
                ++g.vs_no_cam;
                ok = false;
                break;
            }
            camera = true;
        } else {
            if (!skinned) ++g.vs_skin_draws_all;
            skinned = true;
            ++pal_bufs;
            u32 why = 0; // index into vs_why when no previous copy is found
            // Game identity of the palette, once per address and frame (no lock per draw).
            const auto pid = g.pal_ids.find(addr);
            const auto id = pid != g.pal_ids.end()
                                ? pid->second
                                : g.pal_ids.emplace(addr, BB::NativeMv::LookupPalette(addr))
                                      .first->second;
            u64 rkey = 0;
            if (id.key != 0) {
                ++g.vs_ring_seen;
                g.ring_seen_cur[id.key] = 1;
                const u64 ikey = Mix(Mix(id.key, id.slot_offset), size);
                if (have_cur) g.vel_id_cur[ikey] = cur;
                if (const auto it = g.vel_id_prev.find(ikey);
                    it != g.vel_id_prev.end() && it->second.addr != 0) {
                    found = &it->second;
                    ring = true;
                }
                if (g.skin_order) {
                    // Order among this ring's palettes (once per address and frame).
                    if (const auto it = g.ring_ord_of.find(akey); it != g.ring_ord_of.end()) {
                        rkey = it->second;
                    } else {
                        rkey = Mix(Mix(Mix(id.key, 0x726f7264ull), g.ring_ord_n[id.key]++), size);
                        g.ring_ord_of[akey] = rkey;
                        if (have_cur) g.vel_id_cur[rkey] = cur;
                    }
                }
            }
            if (found == nullptr) {
                if (const auto it = g.vel_addr_prev.find(akey);
                    it != g.vel_addr_prev.end() && it->second.addr != 0) {
                    found = &it->second;
                }
            }
            if (found == nullptr && rkey != 0) {
                if (const auto it = g.vel_id_prev.find(rkey);
                    it != g.vel_id_prev.end() && it->second.addr != 0 &&
                    it->second.dwords == size / 4) {
                    found = &it->second;
                    ring = true;
                    ++g.vs_ring_ord;
                }
            }
            // bb-vel-mesh-first: a palette without a ring identity is matched first to last
            // frame's palette of the same mesh draw (vertex program, index buffer, counts,
            // occurrence, handle, size) and only then by draw order: the order changes whenever
            // a character / enemy / part is drawn in another place of the frame, and a similar
            // palette of another instance passed the order check (wrong previous pose: the
            // character's limbs and coat smeared in the generated frames). BB_VEL_MESH_FIRST=0:
            // as before (order first, mesh last).
            static const bool mesh_first = [] {
                const char* v = std::getenv("BB_VEL_MESH_FIRST");
                return !(v != nullptr && v[0] == '0');
            }();
            if (found == nullptr && id.key == 0 && mesh_first && mesh != 0) {
                const u64 mkey = Mix(Mix(mesh, h), size);
                if (const auto it = g.vel_id_prev.find(mkey);
                    it != g.vel_id_prev.end() && it->second.addr != 0 &&
                    it->second.dwords == size / 4) {
                    found = &it->second;
                    ++g.vs_mesh;
                }
            }
            if (found == nullptr && id.key == 0) {
                u64 okey;
                if (const auto it = g.pal_ord_of.find(akey); it != g.pal_ord_of.end()) {
                    okey = it->second;
                } else {
                    const u64 group = Mix(Mix(0x70616cull, vs_hash), size);
                    okey = Mix(group, g.pal_ord_n[group]++);
                    g.pal_ord_of[akey] = okey;
                    const u64 first = g.pal_first.try_emplace(group, addr).first->second;
                    g.pal_rel_cur[okey] = addr - first;
                    if (have_cur) g.vel_id_cur[okey] = cur;
                    if (!buffer_cache.IsRegionGpuModified(addr, 256) &&
                        memory->IsValidMapping(addr, 256)) {
                        std::array<float, 64> c;
                        std::memcpy(c.data(), reinterpret_cast<const void*>(addr), 256);
                        g.pal_cpu_cur[okey] = c;
                    }
                }
                const auto oprev = g.vel_id_prev.find(okey);
                if (oprev == g.vel_id_prev.end() || oprev->second.addr == 0) {
                    why = 4;
                } else if (oprev->second.dwords != size / 4) {
                    why = 5;
                }
                if (const auto it = oprev;
                    it != g.vel_id_prev.end() && it->second.addr != 0 &&
                    it->second.dwords == size / 4) {
                    const auto chk = g.pal_checked.try_emplace(okey, false);
                    if (chk.second) {
                        const auto c = g.pal_cpu_cur.find(okey);
                        const auto p = g.pal_cpu_prev.find(okey);
                        // BB_VEL_PAL_ORDER=trust: also accept palettes the GPU wrote (stale
                        // guest memory, no check possible).
                        static const bool trust = [] {
                            const char* v = std::getenv("BB_VEL_PAL_ORDER");
                            return v != nullptr && std::strcmp(v, "trust") == 0;
                        }();
                        if (c == g.pal_cpu_cur.end() || p == g.pal_cpu_prev.end()) {
                            ++g.vs_pal_unchecked;
                            chk.first->second = trust;
                            if (!trust && g.skin_order) {
                                // GPU-written palette: same place in the same buffer layout
                                // (offset from the group's first palette, group of 2+ last
                                // frame) as last frame's palette of that order.
                                const u64 group = Mix(Mix(0x70616cull, vs_hash), size);
                                const auto rc = g.pal_rel_cur.find(okey);
                                const auto rp = g.pal_rel_prev.find(okey);
                                const auto np = g.pal_n_prev.find(group);
                                if (rc != g.pal_rel_cur.end() && rp != g.pal_rel_prev.end() &&
                                    np != g.pal_n_prev.end() && np->second >= 2 &&
                                    rc->second == rp->second) {
                                    chk.first->second = true;
                                    ++g.vs_pal_layout;
                                }
                            }
                        } else if (PaletteSimilar(c->second, p->second)) {
                            chk.first->second = true;
                        } else {
                            ++g.vs_pal_rejected;
                        }
                    }
                    if (chk.first->second) {
                        found = &it->second;
                        ++g.vs_pal_ord;
                    } else {
                        why = g.pal_cpu_cur.count(okey) != 0 && g.pal_cpu_prev.count(okey) != 0
                                  ? 6u
                                  : 7u;
                    }
                }
            }
            if (mesh != 0) {
                // Last frame's palette of this handle in the same mesh draw (fallback: layout
                // shifted or ring grew). Stored for every palette, used only when all else failed.
                const u64 mkey = Mix(Mix(mesh, h), size);
                if (have_cur) g.vel_id_cur.try_emplace(mkey, cur);
                if (found == nullptr) {
                    if (const auto it = g.vel_id_prev.find(mkey);
                        it != g.vel_id_prev.end() && it->second.addr != 0 &&
                        it->second.dwords == size / 4) {
                        found = &it->second;
                        ++g.vs_mesh;
                    }
                }
            }
            if (found == nullptr) {
                if (!have_cur) {
                    why = 0; // no copy of this frame (snapshot arena full / no buffer)
                } else if (id.key != 0) {
                    why = 1; // ring palette: none of slot, ring order or address last frame
                    if (rkey != 0) {
                        if (const auto it = g.vel_id_prev.find(rkey);
                            it != g.vel_id_prev.end() && it->second.addr != 0) {
                            why = 2; // ring order found, other size
                        } else if (g.ring_seen_prev.count(id.key) != 0) {
                            why = 3; // ring drawn last frame with fewer palettes
                        }
                    }
                } else if (why == 0) {
                    why = 4;
                }
                ++g.vs_why[why];
                ++pal_failed;
                if (!g.skin_order) {
                    ++g.vs_no_pal;
                    ok = false;
                    break;
                }
                // Keep going: the draw's later buffers are still copied for the next frame.
                continue;
            }
            ++pal_matched;
        }
        prev[h] = *found;
        use[h] = true;
    }
    if (ok && pal_bufs > 1) ++g.vs_multi_pal;
    if (ok && pal_failed != 0) {
        if (pal_matched != 0) {
            // Another palette of this draw has last frame's copy: the unmatched one reads its
            // current data (use 0) instead of the whole draw losing its vector.
            ++g.vs_pal_partial;
        } else {
            ++g.vs_no_pal;
            ok = false;
        }
    }

    const u64 bytes = u64(n) * 16;
    if (ok && camera && g.history && g.table_used + bytes <= kTableBytes) {
        const u64 off = SlotTableOffset(g.slot) + g.table_used;
        u32* e = reinterpret_cast<u32*>(g.mapped + off);
        const u64 dummy = g.base_addr + 2048;
        for (u32 h = 0; h < n; ++h) {
            const u64 a = use[h] ? prev[h].addr : dummy;
            e[h * 4 + 0] = u32(a);
            e[h * 4 + 1] = u32(a >> 32);
            e[h * 4 + 2] = use[h] ? prev[h].dwords : 1u;
            e[h * 4 + 3] = use[h] ? 1u : 0u;
        }
        g.table_used += bytes;
        const u64 ea = g.base_addr + off;
        push.objmv_entry = {u32(ea), u32(ea >> 32)};
        // Static draws: 1 in kStaticSample on statistics frames is class 2 (compared with
        // bb_mv.comp's camera vector: the [velocity] static check), else class 1.
        const bool diag = !skinned && g.stat_frame && (g.static_seen++ % kStaticSample) == 0;
        push.objmv_flags = diag ? 3u : 1u;
        ++g.vs_native;
        ++g.frame_vec_draws;
        ++g.sec_vec_draws;
        if (diag) ++g.sec_diag_draws;
        if (skinned) {
            ++g.vs_skinned;
            ++g.vs_skin_draws;
            if (ring) {
                ++g.vs_skinned_ring;
            } else {
                ++g.vs_skinned_addr;
            }
        } else {
            ++g.vs_static;
        }
    } else if (ok && camera && g.history) {
        ++g.sec_overflow;
    }
    g.vs_cpu_ns +=
        std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

void OnDraw(Shader::PushData& push, const Shader::Info& vs, Core::MemoryManager* memory,
            VideoCore::BufferCache& buffer_cache, Vulkan::Scheduler& scheduler,
            u64 vs_hash, u64 identity, u32 count, u32 instances, bool blend, bool ui_phase) {
    push.objmv_flags = 0;
    push.objmv_entry = {u32(g.base_addr), u32(g.base_addr >> 32)}; // zero entry
    if (!g.ready || ui_phase) return;
    if (g.velocity) {
        // Mesh identity (vertex program, index buffer, counts, occurrence this frame): palettes
        // whose order / layout moved are matched to last frame's palette of the same mesh draw.
        u64 mesh = 0;
        if (!blend && g.skin_order) {
            const u64 m = Mix(Mix(Mix(vs_hash, identity), count), instances);
            mesh = Mix(Mix(m, 0x6d657368ull), g.mesh_occ[m]++);
        }
        VelocityDraw(push, vs, memory, buffer_cache, scheduler, vs_hash, blend, mesh);
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    ++g.sec_draws;

    // This draw's identity: vertex program, index / argument buffer, counts, occurrence.
    u64 k = Mix(Mix(Mix(vs_hash, identity), count), instances);
    const u32 occ = g.occ[k]++;
    k = Mix(k, occ);

    // Snapshot this frame's read-only buffers (for the next frame).
    const u32 n = static_cast<u32>(std::min<size_t>(vs.buffers.size(), Shader::NUM_BUFFERS));
    const u32 first = static_cast<u32>(g.bufs_cur.size());
    u32 largest = 0;
    const u64 arena_off = SlotArenaOffset(g.slot);
    for (u32 hnd = 0; hnd < n; ++hnd) {
        const auto& desc = vs.buffers[hnd];
        SnapBuf sb{0, 0, 0, 0};
        const void* src = nullptr;
        u64 size = 0, key = 0, chash = 0;
        if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::Flatbuf && !vs.flattened_ud_buf.empty()) {
                src = vs.flattened_ud_buf.data();
                size = vs.flattened_ud_buf.size() * 4;
            }
        } else if (!desc.is_written) {
            const auto vsharp = desc.GetSharp(vs);
            size = vsharp.GetSize();
            if (vsharp.base_address != 0 && size != 0 && size <= kMaxSnapBytes) {
                key = Mix(vsharp.base_address, size);
                const auto* guest = reinterpret_cast<const void*>(vsharp.base_address);
                if (buffer_cache.IsRegionGpuModified(vsharp.base_address, size)) {
                    // Written by the GPU: guest memory is stale, keep the current buffer.
                    ++g.sec_gpu_bufs;
                    key = 0;
                } else if (memory->IsValidMapping(vsharp.base_address, size)) {
                    // bb-gpu6: the in-frame dedupe compares content hashes of the guest copy
                    // (cached memory) instead of memcmp against the snapshot arena (host-visible
                    // write-combined memory: every read was uncached, ~4 us a draw).
                    chash = XXH3_64bits(guest, size);
                    g.sec_bytes_hashed += size;
                    if (const auto it = g.dedupe.find(key);
                        it != g.dedupe.end() && it->second.addr != 0 && it->second.chash == chash) {
                        sb = it->second;
                        largest = std::max<u32>(largest, u32(size));
                        ++g.sec_dedupe;
                    } else {
                        src = guest;
                    }
                }
            }
        }
        if (src != nullptr && size >= 4 && size <= kMaxSnapBytes) {
            const u64 bytes = (size + 15) & ~u64(15);
            if (g.arena_used + bytes <= kArenaBytes) {
                std::memcpy(g.mapped + arena_off + g.arena_used, src, size);
                g.sec_bytes_copied += size;
                sb.addr = g.base_addr + arena_off + g.arena_used;
                sb.dwords = u32(size / 4);
                sb.chash = chash;
                // Bone palettes / transforms (3x4 matrices): hashed for the moving-draw test.
                if (!desc.IsSpecial() && size >= 96 && size % 48 == 0) {
                    sb.bytes = u32(size);
                    sb.hash = chash != 0 ? chash : XXH3_64bits(src, size);
                    if (g.rel_test && size <= 48u * 256u) {
                        const u32 off = u32(g.rel_cur.size());
                        if (RelTransforms(static_cast<const float*>(src), u32(size / 48),
                                          g.rel_cur)) {
                            sb.rel_off = off;
                            sb.rel_n = u32(size / 48) - 1;
                        } else {
                            g.rel_cur.resize(off);
                        }
                    }
                }
                g.arena_used += bytes;
                if (key != 0) g.dedupe[key] = sb;
                if (!desc.IsSpecial()) largest = std::max<u32>(largest, u32(size));
            } else {
                ++g.sec_overflow;
            }
        }
        g.bufs_cur.push_back(sb);
    }
    g.map_cur[k] = Rec{first, n};

    // Category (statistics): bone palettes are 48-byte matrices.
    u32 cat = CatRigid;
    if (blend) cat = CatBlend;
    else if (largest >= 640 && largest % 48 == 0) cat = CatSkinned;
    else if (largest >= 96 && largest < 640 && largest % 48 == 0) cat = CatSmallSkel;
    ++g.sec_cat[cat];

    // Last frame's snapshots of the same draw. Only moving draws (a bone palette / transform
    // changed since last frame, or blended) run the vertex shader twice; static draws keep the
    // camera vector, except a sample of them on check frames (class 2: compared, never used).
    if (g.history) {
        if (const auto it = g.map_prev.find(k); it != g.map_prev.end() && it->second.count == n) {
            bool moving = blend;
            bool camera_only = false;
            bool any = false;
            const bool skeleton = cat == CatSkinned || cat == CatSmallSkel;
            for (u32 hnd = 0; hnd < n; ++hnd) {
                const SnapBuf& pb = g.bufs_prev[it->second.first + hnd];
                const SnapBuf& cb = g.bufs_cur[first + hnd];
                any |= pb.addr != 0 && pb.dwords != 0;
                if (skeleton && cb.bytes != 0 && (pb.bytes != cb.bytes || pb.hash != cb.hash)) {
                    // bb-gpu6 (BB_OBJMV_MOVE=rel, default): a palette whose matrices all changed
                    // by one common transform (the camera multiplied in, the 369 log's 90 %
                    // "moving" coverage while turning: the scenery got object vectors) is not
                    // moving; one whose bones moved relative to each other is (animation: the
                    // characters keep their object vectors). BB_OBJMV_MOVE=hash: any change.
                    if (g.rel_test && pb.bytes == cb.bytes && pb.rel_n != 0 &&
                        pb.rel_n == cb.rel_n && pb.rel_off + pb.rel_n * 12 <= g.rel_prev.size() &&
                        cb.rel_off + cb.rel_n * 12 <= g.rel_cur.size() &&
                        !RelDiffer(g.rel_prev.data() + pb.rel_off, g.rel_cur.data() + cb.rel_off,
                                   cb.rel_n)) {
                        camera_only = true;
                    } else {
                        moving = true;
                    }
                }
            }
            if (camera_only && !moving) ++g.sec_camera_only;
            const bool diag = !moving && g.stat_frame && (g.static_seen++ % kStaticSample) == 0;
            const u64 bytes = u64(n) * 16;
            if (any && (moving || diag) && g.table_used + bytes <= kTableBytes) {
                const u64 off = SlotTableOffset(g.slot) + g.table_used;
                u32* e = reinterpret_cast<u32*>(g.mapped + off);
                const u64 dummy = g.base_addr + 2048;
                for (u32 hnd = 0; hnd < n; ++hnd) {
                    const SnapBuf& pb = g.bufs_prev[it->second.first + hnd];
                    const bool use = pb.addr != 0 && pb.dwords != 0;
                    const u64 a = use ? pb.addr : dummy;
                    e[hnd * 4 + 0] = u32(a);
                    e[hnd * 4 + 1] = u32(a >> 32);
                    e[hnd * 4 + 2] = use ? pb.dwords : 1u;
                    e[hnd * 4 + 3] = use ? 1u : 0u;
                }
                g.table_used += bytes;
                const u64 ea = g.base_addr + off;
                push.objmv_entry = {u32(ea), u32(ea >> 32)};
                push.objmv_flags = diag ? 3u : 1u;
                if (diag) {
                    ++g.sec_diag_draws;
                } else {
                    ++g.sec_vec_draws;
                    ++g.sec_cat_vec[cat];
                    ++g.frame_vec_draws;
                }
            } else if (any && (moving || diag)) {
                ++g.sec_overflow;
            }
        }
    }
    g.sec_cpu_ns +=
        std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
}

bool ObjectTarget(vk::ImageView& view, float& scale_x, float& scale_y) {
    if (!g.ready || !g.image || !g.view || g.width == 0 || g.height == 0 || g.last_render.width == 0) {
        return false;
    }
    view = g.view;
    scale_x = float(g.last_render.width) / float(g.width);
    scale_y = float(g.last_render.height) / float(g.height);
    return true;
}

void Compose(vk::CommandBuffer cmd, vk::ImageView mv_view, vk::ImageView color_view,
             vk::Extent2D render, bool reset) {
    if (!g.ready || g.composed || !g.cleared || !g.image) return;
    if (render.width > g.width || render.height > g.height) return;
    g.composed = true;
    g.last_render = render;
    const u32 slot = g.slot;
    ReadStats(slot);
    const bool stat = g.stat_frame;
    // bb-player-mv: the player mask (moving opaque pixels in the screen's centre): every
    // frame its vectors are measured against last frame's colour and refined (own vector,
    // +-1 px, the neighbours' vectors: the one whose 3x3 patch matches best). Needs last
    // frame's colour every frame (ping-pong prev[0] / prev[1]). BB_PLAYER_MV_REFINE=0: off.
    static const bool refine = [] {
        const char* v = std::getenv("BB_PLAYER_MV_REFINE");
        return !(v != nullptr && v[0] == '0');
    }();
    const bool store_color = color_view && ((g.check && IsStatFrame(g.frame + 1)) || refine);
    if (!stat && !store_color && (g.frame_vec_draws == 0 || reset)) {
        // Nothing moving drew this frame: the camera vectors stay as they are.
        ++g.sec_skipped;
        if (reset) {
            ++g.sec_resets;
            g.reset_pending = true;
        }
        return;
    }
    if (reset) {
        ++g.sec_resets;
        g.reset_pending = true; // next frame: no history (camera cut)
    }
    cmd.fillBuffer(g.stats, slot * kStatWords * 4, kStatWords * 4, 0);
    cmd.resetQueryPool(g.queries, slot * 2, 2);
    // The stats clear and bb_mv.comp's camera vectors (compute) before this pass.
    const vk::MemoryBarrier2 fill_done{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer |
                        vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite |
                         vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                         vk::AccessFlagBits2::eShaderStorageWrite,
    };
    const vk::ImageMemoryBarrier2 obj_ready{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = g.image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    const u32 rd = refine ? g.pp : 0u; // previous colour read
    const u32 next = refine ? 1u - g.pp : 0u; // written (single image without refine)
    const vk::ImageMemoryBarrier2 prev_barriers[2] = {
        {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
         .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
         .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
         .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
         .oldLayout = vk::ImageLayout::eGeneral,
         .newLayout = vk::ImageLayout::eGeneral,
         .image = g.prev[1],
         .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}},
        {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
         .srcAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
         .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
         .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
         .oldLayout = vk::ImageLayout::eUndefined,
         .newLayout = vk::ImageLayout::eGeneral,
         .image = g.prev[next],
         .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}},
    };
    // prev[1] is unused (bound as a dummy sampled image): only prev[0] needs a layout.
    const bool prev_ready = g.prev_valid;
    const vk::ImageMemoryBarrier2 images[2] = {obj_ready, prev_barriers[1]};
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                            .pMemoryBarriers = &fill_done,
                                            .imageMemoryBarrierCount =
                                                (store_color || !prev_ready) ? 2u : 1u,
                                            .pImageMemoryBarriers = images});
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, g.queries, slot * 2);
    const vk::DescriptorImageInfo obj_info{g.sampler, g.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo mv_info{{}, mv_view, vk::ImageLayout::eGeneral};
    const vk::DescriptorBufferInfo stats_info{g.stats, slot * kStatWords * 4, kStatWords * 4};
    const vk::DescriptorImageInfo cur_info{g.sampler, color_view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo prev_info{g.sampler, g.prev_view[rd], vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo next_info{{}, g.prev_view[next], vk::ImageLayout::eGeneral};
    const auto w = [](u32 b, vk::DescriptorType t, const vk::DescriptorImageInfo* i,
                      const vk::DescriptorBufferInfo* bi) {
        return vk::WriteDescriptorSet{.dstBinding = b,
                                      .descriptorCount = 1,
                                      .descriptorType = t,
                                      .pImageInfo = i,
                                      .pBufferInfo = bi};
    };
    const std::array writes{
        w(0, vk::DescriptorType::eCombinedImageSampler, &obj_info, nullptr),
        w(1, vk::DescriptorType::eStorageImage, &mv_info, nullptr),
        w(2, vk::DescriptorType::eStorageBuffer, nullptr, &stats_info),
        w(3, vk::DescriptorType::eCombinedImageSampler, &cur_info, nullptr),
        w(4, vk::DescriptorType::eCombinedImageSampler, &prev_info, nullptr),
        w(5, vk::DescriptorType::eStorageImage, &next_info, nullptr),
    };
    const bool check = stat && g.check && color_view && !reset && g.prev_valid &&
                       g.color_stored + 1 == g.frame;
    const bool refine_now = refine && color_view && !reset && g.prev_valid &&
                            g.color_stored + 1 == g.frame;
    const std::array<u32, 4> push{render.width, render.height, reset ? 0u : 1u,
                                  (stat ? 1u : 0u) | (g.additive ? 2u : 0u) |
                                      (store_color ? 4u : 0u) | (check ? 8u : 0u) |
                                      (refine_now ? 16u : 0u)};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g.layout, 0, writes);
    cmd.pushConstants(g.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push.data());
    cmd.dispatch((render.width + 7) / 8, (render.height + 7) / 8, 1);
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, g.queries, slot * 2 + 1);
    const vk::MemoryBarrier2 done{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eHostRead,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &done});
    g.slot_used[slot] = true;
    g.slot_stat[slot] = stat;
    if (store_color && !reset) {
        g.color_stored = g.frame;
        g.prev_valid = true;
        if (refine) g.pp = next;
    }
}

void OnFlip() {
    if (!g.ready) return;
    // bb-velocity: a flip without a G-buffer motion pass since the last one (a re-presented or
    // repeated guest image) is not a frame: history and snapshots stay as they are.
    if (g.velocity && !g.cleared) return;
    ++g.frame;
    ++g.sec_frames;
    const bool had_pass = g.cleared;
    if (had_pass && !g.composed) {
        ++g.sec_not_composed;
        if (!g.logged_not_composed) {
            g.logged_not_composed = true;
            Log("frame ended with a motion pass but no compose pass");
        }
    }
    // History = exactly this frame, if it had the pass (and no camera cut).
    if (had_pass && !g.reset_pending) {
        std::swap(g.map_prev, g.map_cur);
        std::swap(g.bufs_prev, g.bufs_cur);
        std::swap(g.rel_prev, g.rel_cur);
        std::swap(g.vel_addr_prev, g.vel_addr_cur);
        std::swap(g.vel_id_prev, g.vel_id_cur);
        std::swap(g.cam_cpu_prev, g.cam_cpu_cur);
        std::swap(g.pal_cpu_prev, g.pal_cpu_cur);
        std::swap(g.pal_rel_prev, g.pal_rel_cur);
        std::swap(g.ring_seen_prev, g.ring_seen_cur);
        g.pal_n_prev.clear();
        for (const auto& [grp, cnt] : g.pal_ord_n) g.pal_n_prev[grp] = cnt;
        g.history = true;
    } else {
        if (g.history) ++g.sec_resets;
        g.map_prev.clear();
        g.bufs_prev.clear();
        g.rel_prev.clear();
        g.vel_addr_prev.clear();
        g.vel_id_prev.clear();
        g.cam_cpu_prev.clear();
        g.pal_cpu_prev.clear();
        g.pal_rel_prev.clear();
        g.pal_n_prev.clear();
        g.ring_seen_prev.clear();
        g.history = false;
        if (!had_pass) g.prev_valid = false;
    }
    g.reset_pending = false;
    g.map_cur.clear();
    g.bufs_cur.clear();
    g.rel_cur.clear();
    g.vel_addr_cur.clear();
    g.vel_id_cur.clear();
    g.cam_ord.clear();
    g.cam_checked.clear();
    g.cam_cpu_cur.clear();
    g.pal_cpu_cur.clear();
    g.pal_ord_of.clear();
    g.pal_ord_n.clear();
    g.pal_checked.clear();
    g.pal_rel_cur.clear();
    g.pal_first.clear();
    g.ring_ord_of.clear();
    g.ring_ord_n.clear();
    g.ring_seen_cur.clear();
    g.mesh_occ.clear();
    g.pal_ids.clear();
    if (had_pass) ++g.vs_frames;
    g.occ.clear();
    g.dedupe.clear();
    g.cleared = false;
    g.composed = false;
    g.frame_vec_draws = 0;
    g.static_seen = 0;
    g.stat_frame = IsStatFrame(g.frame);
    g.slot = (g.slot + 1) % kSlots;
    g.arena_used = 0;
    g.table_used = 0;

    const auto now = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(now - g.t0).count();
    if (secs < 1.0) return;
    {
        const u64 period = double(g.sec_frames) / secs < 75.0 ? kStatPeriod / 2 : kStatPeriod;
        g_stat_period = period; // a stored colour without its check frame is simply unused
    }
    g.t0 = now;
    const double cov = g.sec_px_all ? 100.0 * double(g.sec_px_obj) / double(g.sec_px_all) : 0.0;
    std::string cats;
    for (u32 c = 0; c < CatCount; ++c) {
        cats += fmt::format("{}{} {}/{}", c ? ", " : "", kCatName[c], g.sec_cat_vec[c],
                            g.sec_cat[c]);
    }
    const double eo = g.sec_chk ? double(g.sec_eo) / 1024.0 / double(g.sec_chk) : 0.0;
    const double ec = g.sec_chk ? double(g.sec_ec) / 1024.0 / double(g.sec_chk) : 0.0;
    // Static draws: |obj - cam| (absolute mode) and |obj| (additive mode), render pixels.
    const double st_abs =
        g.sec_static_px ? double(g.sec_static_abs) / 16.0 / double(g.sec_static_px) : 0.0;
    const double st_zero =
        g.sec_static_px ? double(g.sec_static_zero) / 16.0 / double(g.sec_static_px) : 0.0;
    if (!g.velocity && g.sec_static_px >= 2000) {
        // Pick the mode whose static-draw error is smaller (hysteresis 0.1 px).
        if (g.additive && st_abs + 0.1 < st_zero) g.additive = false;
        else if (!g.additive && st_zero + 0.1 < st_abs) g.additive = true;
    }
    const double st_err = g.additive ? st_zero : st_abs;
    Log(fmt::format(
        "draws with object vectors {}/{} (moving only), static checked {}, coverage {:.1f}% "
        "(blended {:.1f}%), GPU {:.3f} ms (compose), {:.3f} ms on check frames, skipped {}, CPU "
        "{:.3f} us/draw added, history resets {}, frames {}, overflow {} | {} | mode {} | "
        "static-draw error {:.3f} px over {} px ({}; |obj-cam| {:.3f}, |obj| {:.3f}) | check: {} "
        "px, reprojection error obj {:.4f} vs cam {:.4f} ({}), obj better {} / cam better {} | "
        "attachment 7 non-zero texels {:.0f}/frame, frames not composed {}, GPU-written buffers "
        "kept current {}",
        g.sec_vec_draws, g.sec_draws, g.sec_diag_draws, cov,
        g.sec_px_all ? 100.0 * double(g.sec_px_blend) / double(g.sec_px_all) : 0.0,
        g.sec_gpu_n ? g.sec_gpu_ms / g.sec_gpu_n : 0.0,
        g.sec_gpu_stat_n ? g.sec_gpu_stat_ms / g.sec_gpu_stat_n : 0.0, g.sec_skipped,
        g.sec_draws ? g.sec_cpu_ns / 1000.0 / double(g.sec_draws) : 0.0, g.sec_resets,
        g.sec_frames, g.sec_overflow, cats,
        g.additive ? "additive (object motion + camera vector)" : "absolute",
        st_err, g.sec_static_px,
        g.sec_static_px == 0 ? "no static px" : (st_err <= 0.5 ? "PASS" : "FAIL"), st_abs, st_zero,
        g.sec_chk, eo, ec,
        g.sec_chk == 0 ? "no differing px" : (eo <= ec ? "OBJECT VECTORS WIN" : "CAMERA WINS"),
        g.sec_wo, g.sec_wc,
        g.sec_stat_frames ? double(g.sec_nonzero) / double(g.sec_stat_frames) : 0.0,
        g.sec_not_composed, g.sec_gpu_bufs));
    if (g.sec_pl_px > 0) {
        Log(fmt::format("[player-mv] player mask {:.0f} px/frame, vectors refined {:.1f}%, patch error "
                        "{:.4f} -> {:.4f} (BB_PLAYER_MV_REFINE=0 off)",
                        double(g.sec_pl_px) / double(std::max<u64>(1, g.sec_frames)),
                        100.0 * double(g.sec_pl_changed) / double(g.sec_pl_px),
                        double(g.sec_pl_before) / 1024.0 / double(g.sec_pl_px),
                        double(g.sec_pl_after) / 1024.0 / double(g.sec_pl_px)));
    }
    g.sec_pl_px = g.sec_pl_changed = g.sec_pl_before = g.sec_pl_after = 0;
    if (g.velocity) {
        // Native share of the G-buffer pixels (statistics frames): pixels written by a draw's own
        // vector (class 1 + the sampled static draws, class 2) over all pixels; the rest keeps
        // bb_mv.comp (backup: sky, particles, draws without a previous camera / palette).
        const u64 native_raw = g.sec_px_obj - std::min(g.sec_px_obj, g.sec_px_blend) +
                               g.sec_static_px;
        const double native_px =
            g.sec_px_all ? 100.0 * double(native_raw) / double(g.sec_px_all) : 0.0;
        const double f = double(std::max<u64>(1, g.vs_frames));
        Log(fmt::format(
            "[velocity] native velocity draws {:.1f}/frame of {:.1f} (static {:.1f}, skinned "
            "{:.1f}: ring palette {:.1f}, palette by address {:.1f}; ring palettes seen {:.1f}) | "
            "G-buffer px native {:.1f}%, backup bb_mv {:.1f}% | static check |native-cam| "
            "{:.3f} px over {} px ({}) | previous camera by address {:.1f} / by order {:.1f} "
            "(rejected as different {:.1f}, uncheckable {:.1f}), without {:.1f}/frame; without previous palette {:.1f}/frame; blended (backup) "
            "{:.1f}/frame | palettes without identity matched by order {:.1f}/frame (rejected as "
            "different {:.1f}, uncheckable {:.1f}) | GPU copies {:.1f}/frame, snapshot {:.0f} "
            "B/frame, CPU {:.3f} us/draw",
            double(g.vs_native) / f, double(g.vs_draws) / f, double(g.vs_static) / f,
            double(g.vs_skinned) / f, double(g.vs_skinned_ring) / f,
            double(g.vs_skinned_addr) / f, double(g.vs_ring_seen) / f, native_px,
            g.sec_px_all ? 100.0 - native_px : 0.0, st_abs, g.sec_static_px,
            g.sec_static_px == 0 ? "no static px" : (st_abs <= 0.5 ? "PASS" : "FAIL"),
            double(g.vs_cam_addr) / f, double(g.vs_cam_ord) / f, double(g.vs_cam_rejected) / f,
            double(g.vs_cam_unchecked) / f, double(g.vs_no_cam) / f,
            double(g.vs_no_pal) / f, double(g.vs_blend) / f, double(g.vs_pal_ord) / f,
            double(g.vs_pal_rejected) / f, double(g.vs_pal_unchecked) / f,
            double(g.vs_gpu_copies) / f,
            double(g.vs_bytes) / f,
            g.vs_draws ? g.vs_cpu_ns / 1000.0 / double(g.vs_draws) : 0.0));
        g.vs_frames = g.vs_draws = g.vs_native = g.vs_static = g.vs_skinned = g.vs_skinned_ring =
            g.vs_skinned_addr = g.vs_no_cam = g.vs_no_pal = g.vs_blend = g.vs_gpu_copies =
                g.vs_bytes = g.vs_cam_addr = g.vs_cam_ord = g.vs_ring_seen = g.vs_cam_rejected =
                    g.vs_cam_unchecked = g.vs_pal_ord = g.vs_pal_rejected =
                        g.vs_pal_unchecked = 0;
        g.vs_cpu_ns = 0;
        Log(fmt::format("[skin-mv] skinned (bone palette) draws with own vectors {:.1f}/frame of "
                        "{:.1f} ({:.1f}%) | extra from ring order {:.1f}/frame, GPU-written by "
                        "layout {:.1f}/s | {}",
                        double(g.vs_skin_draws) / f, double(g.vs_skin_draws_all) / f,
                        g.vs_skin_draws_all ? 100.0 * double(g.vs_skin_draws) /
                                                  double(g.vs_skin_draws_all)
                                            : 0.0,
                        double(g.vs_ring_ord) / f, double(g.vs_pal_layout),
                        g.skin_order ? "on (BB_SKIN_MV=0: off)" : "off (BB_SKIN_MV=0)"));
        Log(fmt::format("[skin-mv] palette buffers without last frame's copy, per frame: no copy "
                        "this frame {:.1f}, ring new/not drawn last frame {:.1f}, ring order other "
                        "size {:.1f}, ring with more palettes than last frame {:.1f}, order new "
                        "{:.1f}, order other size {:.1f}, order data differs {:.1f}, order "
                        "GPU-written + layout moved {:.1f} | draws kept by their other palette "
                        "{:.1f}/frame, draws with 2+ palette buffers {:.1f}/frame | palettes "
                        "matched by mesh draw (index buffer + occurrence) {:.1f}/frame",
                        double(g.vs_why[0]) / f, double(g.vs_why[1]) / f,
                        double(g.vs_why[2]) / f, double(g.vs_why[3]) / f,
                        double(g.vs_why[4]) / f, double(g.vs_why[5]) / f,
                        double(g.vs_why[6]) / f, double(g.vs_why[7]) / f,
                        double(g.vs_pal_partial) / f, double(g.vs_multi_pal) / f,
                        double(g.vs_mesh) / f));
        g.vs_why.fill(0);
        g.vs_pal_partial = g.vs_multi_pal = g.vs_mesh = 0;
        g.vs_skin_draws = g.vs_skin_draws_all = g.vs_ring_ord = g.vs_pal_layout = 0;
    }
    Log(fmt::format("[objmv-gpu6] moving test {}: palettes changed only by a common transform "
                    "(camera) kept static {}/s; CPU per draw: hashed {:.0f} B, copied {:.0f} B, "
                    "dedupe hits {}",
                    g.rel_test ? "rel (BB_OBJMV_MOVE=rel)" : "hash (BB_OBJMV_MOVE=hash)",
                    g.sec_camera_only,
                    g.sec_draws ? double(g.sec_bytes_hashed) / double(g.sec_draws) : 0.0,
                    g.sec_draws ? double(g.sec_bytes_copied) / double(g.sec_draws) : 0.0,
                    g.sec_dedupe));
    g.sec_camera_only = g.sec_dedupe = g.sec_bytes_hashed = g.sec_bytes_copied = 0;
    g.sec_gpu_bufs = 0;
    g.sec_nonzero = g.sec_stat_frames = g.sec_not_composed = 0;
    g.sec_draws = g.sec_vec_draws = g.sec_resets = g.sec_frames = g.sec_overflow = 0;
    for (u32 c = 0; c < CatCount; ++c) g.sec_cat[c] = g.sec_cat_vec[c] = 0;
    g.sec_cpu_ns = 0;
    g.sec_px_obj = g.sec_px_blend = g.sec_px_all = g.sec_chk = g.sec_eo = g.sec_ec = g.sec_wo =
        g.sec_wc = 0;
    g.sec_gpu_ms = g.sec_gpu_stat_ms = 0;
    g.sec_gpu_n = g.sec_gpu_stat_n = 0;
    g.sec_static_px = g.sec_static_abs = g.sec_static_zero = g.sec_diag_draws = g.sec_skipped = 0;
}

} // namespace BB::ObjMv2
