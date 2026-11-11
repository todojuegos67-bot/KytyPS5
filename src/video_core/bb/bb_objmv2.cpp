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
constexpr u32 kStatWords = 12; // word 11: attachment 7 texels != 0
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
    u64 pkey = 0;     // gpu7: (address, size) of a hashed palette-like buffer (shared-use count)
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
    bool rel_test = false;
    // gpu7: draws a frame reading each palette-like buffer (address, size), this frame and the
    // last. A buffer read by many draws is a per-frame / per-pass constant block (it holds the
    // camera: in the 370 log every draw changed with it, coverage 100 %), not a skeleton.
    std::unordered_map<u64, u32> use_cur, use_prev;
    u32 shared_min = 48;
    u64 sec_shared = 0;
    u64 sec_ui_skipped = 0;
    // bb-149-objmv: the 149 composes at its HUD start or at the present, which can come after
    // the guest flip: the frame's state now rolls over at the next frame's first motion draw
    // (flip_pending), and v2 hands back to v1 when it composes nothing for 2 s of 3D frames.
    bool flip_pending = false;
    u32 sec_composed = 0, sec_motion_frames = 0, bad_secs = 0; // pipeline lookups refused as screen-space (depth test Always/Never)
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
    u32 frame_vec_draws = 0, static_seen = 0;
    std::array<bool, kSlots> slot_stat{};
    double ts_period = 1.0;
    u64 sec_gpu_bufs = 0;
    u64 sec_nonzero = 0, sec_stat_frames = 0, sec_not_composed = 0;
    bool logged_nonzero = false, logged_not_composed = false;
    // bb-velocity: native velocity buffer (every G-buffer draw writes its own vector in its
    // vertex shader: previous view-projection, previous bone palette). Snapshots of this frame
    // by (address, size) and by game identity (ring palette, camera block ordinal).
    bool velocity = true;
    std::unordered_map<u64, SnapBuf> vel_addr_cur, vel_addr_prev, vel_id_cur, vel_id_prev;
    std::unordered_map<u64, u32> cam_ord; // camera block (address, size) -> ordinal this frame
    std::unordered_map<u64, u64> role_cache; // vs hash -> handles that may hold camera / palette
    u64 vs_frames = 0, vs_draws = 0, vs_native = 0, vs_static = 0, vs_skinned = 0,
        vs_skinned_ring = 0, vs_skinned_addr = 0, vs_no_cam = 0, vs_no_pal = 0, vs_blend = 0,
        vs_gpu = 0, vs_bytes = 0;
    double vs_cpu_ns = 0;
} g;

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
    // bb-149: buffer device addresses are always enabled on this base (v1 relies on them too).
    g.check = !EnvOff("BB_OBJMV2_CHECK");
    {
        const char* mv = std::getenv("BB_OBJMV_MOVE");
        g.rel_test = mv != nullptr && std::strcmp(mv, "rel") == 0;
        const char* sh = std::getenv("BB_OBJMV_SHARED");
        g.shared_min = sh != nullptr && *sh != '\0' ? u32(std::max(0, std::atoi(sh))) : 48u;
        Log(fmt::format("[objmv-gpu7] moving test: a changed palette moves its draw unless {} "
                        "draws a frame read it (BB_OBJMV_SHARED, 0: off){}",
                        g.shared_min, g.rel_test ? ", plus the relative-bone test" : ""));
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
                         vk::BufferUsageFlagBits::eShaderDeviceAddress,
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
    if (g.velocity) g.additive = false; // absolute vectors: the previous camera is in the VS
    g.ready = true;
    Log(g.velocity ? "[velocity] on (bb-velocity): every G-buffer draw writes its own velocity "
                     "in its vertex shader (previous view-projection from the 864 B camera "
                     "block, previous bone palette from the skin ring), bb_mv.comp only where "
                     "none is written; BB_VELOCITY=0: objmv2 as bb-149-objmv"
                   : "[velocity] off (BB_VELOCITY=0)");
    Log(fmt::format("on: scene draws run their vertex shader once more with last frame's buffers "
                    "(GPU), vectors in attachment 7, composed over bb_mv; snapshot arena {} MiB x "
                    "{}, check {}",
                    kArenaBytes >> 20, kSlots, g.check ? "on" : "off (BB_OBJMV2_CHECK=0)"));
    return true;
}

void Rollover(); // bb-149-objmv: the frame rollover (deferred from OnFlip)

bool Active() {
    return g.ready;
}

void NoteSceneDepth(u64 depth_address) {
    g.scene_depth = depth_address;
}

Kind Want(const AmdGpu::Regs& regs) {
    // bb-149-objmv: the scene depth from the 149 plan (as bb_objmv.cpp's WantPipeline).
    if (g.ready && BB::Upscale::Enabled()) {
        const VAddr plan_depth = BB::Upscale::CurrentPlan().depth_addr;
        if (plan_depth != 0) g.scene_depth = plan_depth;
    }
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
    // bb-149-objmv (HUD duplicates under DLSS-G, build 376): a draw whose depth test passes
    // everything (Always / Never) is screen-space: the HUD / UI (Scaleform) or a full-screen pass,
    // never a moving surface. It keeps attachment 7 bound but writes no object vector, so its
    // pixels keep the camera vector. BB_OBJMV_UI=1 restores 376.
    static const bool ui_old = [] {
        const char* v = std::getenv("BB_OBJMV_UI");
        const bool old = v != nullptr && *v == '1';
        Log(old ? "[objmv-ui] BB_OBJMV_UI=1: depth-test-always draws may get object vectors (376)"
                : "[objmv-ui] draws with depth test Always/Never (HUD, UI, full-screen passes) "
                  "get no object vector");
        return old;
    }();
    // bb-149-objmv: screen-space draws (no depth test, clip disabled: HUD, UI, full-screen
    // passes) get the stock pipeline, without attachment 7: in 376 they took the Mask variant, an
    // extra attachment the 149's HUD redirection / HUD-less capture does not expect.
    if (!ui_old && (!regs.depth_control.depth_enable || regs.IsClipDisabled())) {
        ++g.sec_ui_skipped;
        return Kind::None;
    }
    if (!regs.depth_control.depth_enable || regs.IsClipDisabled() || !regs.color_buffers[0] ||
        !regs.color_target_mask.GetMask(0)) {
        return Kind::Mask;
    }
    {
        using CF = AmdGpu::CompareFunc;
        const auto f = regs.depth_control.depth_func;
        if (!ui_old && (f == CF::Always || f == CF::Never)) {
            ++g.sec_ui_skipped;
            return Kind::None;
        }
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
    if (g.ready && g.flip_pending) Rollover();
    if (!g.ready || state.num_color_attachments <= Attachment) return;
    // bb-149-objmv: the scene's render size from the 149 plan.
    const auto& plan = BB::Upscale::CurrentPlan();
    const u32 w = std::max<u32>(plan.render_w, state.width);
    const u32 h = std::max<u32>(plan.render_h, state.height);
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

// bb-velocity helpers live in the anonymous namespace
namespace {
// ---------------------------------------------------------------------------------------------
// bb-velocity (native velocity buffer, as a game with TAA writes it).
//
// Identification of a vertex shader's buffers (the table to extend; phase 2 adds rigid world
// transforms): the role comes from the V# the shader reads, no CPU scan of the draw.
//   864 B                                   camera block (view-projection): previous frame's copy
//   640..16384 B, multiple of 16, not 864   bone palette: previous frame's copy of the same game
//                                           skin ring slot (BB::NativeMv), else same address+size
//   anything else                           current buffer (world transform: phase 2)
enum class VelRole : u8 { Current = 0, Camera = 1, Palette = 2 };

VelRole IdentifyVelocityBuffer(u32 size) {
    if (size == 864) return VelRole::Camera;
    if (size >= 640 && size <= 16384 && size % 16 == 0) return VelRole::Palette;
    return VelRole::Current;
}

// This frame's copy of a guest buffer (once per address, size and frame).
bool VelSnap(Core::MemoryManager* memory, VideoCore::BufferCache& buffer_cache, u64 addr,
             u32 size, SnapBuf& out) {
    const u64 key = Mix(addr, size);
    if (const auto it = g.vel_addr_cur.find(key); it != g.vel_addr_cur.end()) {
        out = it->second;
        return out.addr != 0;
    }
    SnapBuf sb{0, 0, 0, 0};
    const u64 bytes = (u64(size) + 15) & ~u64(15);
    if (buffer_cache.IsRegionGpuModified(addr, size)) {
        ++g.vs_gpu; // GPU-written: guest memory is stale
    } else if (memory->IsValidMapping(addr, size) && g.arena_used + bytes <= kArenaBytes) {
        std::memcpy(g.mapped + SlotArenaOffset(g.slot) + g.arena_used,
                    reinterpret_cast<const void*>(addr), size);
        sb.addr = g.base_addr + SlotArenaOffset(g.slot) + g.arena_used;
        sb.dwords = size / 4;
        sb.bytes = size;
        g.arena_used += bytes;
        g.vs_bytes += size;
    } else {
        ++g.sec_overflow;
    }
    g.vel_addr_cur[key] = sb;
    out = sb;
    return sb.addr != 0;
}

void VelocityDraw(Shader::PushData& push, const Shader::Info& vs, Core::MemoryManager* memory,
                  VideoCore::BufferCache& buffer_cache, u64 vs_hash, bool blend) {
    const auto t0 = std::chrono::steady_clock::now();
    ++g.vs_draws;
    ++g.sec_draws;
    if (blend) {
        // Particles / effects: no vector of their own, bb_mv.comp stays under them.
        ++g.vs_blend;
        return;
    }
    const u32 n = static_cast<u32>(std::min<size_t>(vs.buffers.size(), Shader::NUM_BUFFERS));
    // Shader cache: handles that are plain read-only V#s (the only ones that can take a role).
    u64 cand;
    if (const auto it = g.role_cache.find(vs_hash); it != g.role_cache.end()) {
        cand = it->second;
    } else {
        cand = 0;
        for (u32 h = 0; h < n; ++h) {
            if (!vs.buffers[h].IsSpecial() && !vs.buffers[h].is_written) cand |= 1ull << h;
        }
        g.role_cache[vs_hash] = cand;
    }
    std::array<SnapBuf, Shader::NUM_BUFFERS> prev{};
    std::array<bool, Shader::NUM_BUFFERS> use{};
    bool camera = false, ok = true, skinned = false, ring = false;
    for (u32 h = 0; h < n && ok; ++h) {
        if ((cand & (1ull << h)) == 0) continue;
        const auto sharp = vs.buffers[h].GetSharp(vs);
        const u32 size = sharp.GetSize();
        const u64 addr = sharp.base_address;
        const VelRole role = IdentifyVelocityBuffer(size);
        if (role == VelRole::Current || addr == 0) continue;
        SnapBuf cur{0, 0, 0, 0};
        const bool have_cur = VelSnap(memory, buffer_cache, addr, size, cur);
        const u64 akey = Mix(addr, size);
        const SnapBuf* found = nullptr;
        if (role == VelRole::Camera) {
            // The camera block: same address last frame, else the same ordinal (the game may
            // place it anew every frame).
            u32 ord;
            if (const auto it = g.cam_ord.find(akey); it != g.cam_ord.end()) {
                ord = it->second;
            } else {
                ord = static_cast<u32>(g.cam_ord.size());
                g.cam_ord[akey] = ord;
                if (have_cur) g.vel_id_cur[Mix(0x63616d65ull, ord)] = cur;
            }
            if (const auto it = g.vel_addr_prev.find(akey);
                it != g.vel_addr_prev.end() && it->second.addr != 0) {
                found = &it->second;
            } else if (const auto it2 = ord < 4 ? g.vel_id_prev.find(Mix(0x63616d65ull, ord))
                                                : g.vel_id_prev.end();
                       it2 != g.vel_id_prev.end() && it2->second.addr != 0) {
                found = &it2->second;
            }
            if (found == nullptr) {
                ++g.vs_no_cam;
                ok = false;
                break;
            }
            camera = true;
        } else {
            skinned = true;
            const auto id = BB::NativeMv::LookupPalette(addr);
            if (id.key != 0) {
                const u64 ikey = Mix(Mix(id.key, id.slot_offset), size);
                if (have_cur) g.vel_id_cur[ikey] = cur;
                if (const auto it = g.vel_id_prev.find(ikey);
                    it != g.vel_id_prev.end() && it->second.addr != 0) {
                    found = &it->second;
                    ring = true;
                }
            }
            if (found == nullptr) {
                if (const auto it = g.vel_addr_prev.find(akey);
                    it != g.vel_addr_prev.end() && it->second.addr != 0) {
                    found = &it->second;
                }
            }
            if (found == nullptr) {
                ++g.vs_no_pal;
                ok = false;
                break;
            }
        }
        prev[h] = *found;
        use[h] = true;
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
            ring ? ++g.vs_skinned_ring : ++g.vs_skinned_addr;
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
            VideoCore::BufferCache& buffer_cache,
            u64 vs_hash, u64 identity, u32 count, u32 instances, bool blend, bool ui_phase) {
    push.objmv_flags = 0;
    push.objmv_entry = {u32(g.base_addr), u32(g.base_addr >> 32)}; // zero entry
    if (g.ready && g.flip_pending) Rollover();
    if (!g.ready || ui_phase) return;
    if (g.velocity) {
        VelocityDraw(push, vs, memory, buffer_cache, vs_hash, blend);
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
                if (size >= 96 && size % 48 == 0) {
                    sb.pkey = key;
                    ++g.use_cur[key];
                }
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
                    const auto use = cb.pkey != 0 && g.shared_min != 0 ? g.use_prev.find(cb.pkey)
                                                                         : g.use_prev.end();
                    if (use != g.use_prev.end() && use->second >= g.shared_min) {
                        // gpu7: a constant block shared by many draws (the camera's): static.
                        camera_only = true;
                        ++g.sec_shared;
                    } else if (g.rel_test && pb.bytes == cb.bytes && pb.rel_n != 0 &&
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

void Compose(vk::CommandBuffer cmd, vk::ImageView mv_view, vk::ImageView color_view,
             vk::Extent2D render, bool reset) {
    if (!g.ready || g.composed || !g.cleared || !g.image) return;
    if (render.width > g.width || render.height > g.height) return;
    g.composed = true;
    ++g.sec_composed;
    const u32 slot = g.slot;
    ReadStats(slot);
    const bool stat = g.stat_frame;
    const bool store_color = g.check && color_view && IsStatFrame(g.frame + 1);
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
    const u32 next = 0; // single previous-colour image (written the frame before a check)
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
    const vk::DescriptorImageInfo prev_info{g.sampler, g.prev_view[0], vk::ImageLayout::eGeneral};
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
    const std::array<u32, 4> push{render.width, render.height, reset ? 0u : 1u,
                                  (stat ? 1u : 0u) | (g.additive ? 2u : 0u) |
                                      (store_color ? 4u : 0u) | (check ? 8u : 0u)};
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
    if (store_color) {
        g.color_stored = g.frame;
        g.prev_valid = true;
    }
}

void OnFlip() {
    if (!g.ready) return;
    g.flip_pending = true;
}

void Rollover() {
    g.flip_pending = false;
    ++g.frame;
    ++g.sec_frames;
    const bool had_pass = g.cleared;
    if (had_pass) ++g.sec_motion_frames;
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
        g.history = true;
    } else {
        if (g.history) ++g.sec_resets;
        g.map_prev.clear();
        g.bufs_prev.clear();
        g.rel_prev.clear();
        g.vel_addr_prev.clear();
        g.vel_id_prev.clear();
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
    if (had_pass) ++g.vs_frames;
    std::swap(g.use_prev, g.use_cur);
    g.use_cur.clear();
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
    {
        static const bool fallback = !EnvOff("BB_OBJMV2_FALLBACK");
        if (g.sec_motion_frames >= 5 && g.sec_composed == 0) {
            ++g.bad_secs;
        } else if (g.sec_composed != 0) {
            g.bad_secs = 0;
        }
        Log(fmt::format("[objmv-149] composed {} of {} frames with motion draws this second{}",
                        g.sec_composed, g.sec_motion_frames,
                        g.bad_secs ? fmt::format(" ({} s in a row without a compose)", g.bad_secs)
                                   : std::string()));
        g.sec_composed = g.sec_motion_frames = 0;
        if (fallback && g.bad_secs >= 2) {
            g.ready = false;
            Log("[objmv-149] v2 composed no frame for 2 s of 3D frames: off for this session, the "
                "149's own object vectors (v1) take over (BB_OBJMV2_FALLBACK=0 keeps v2)");
            return;
        }
    }
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
    if (g.velocity) {
        // Native share of the G-buffer pixels (statistics frames): pixels written by a draw's own
        // vector (moving / skinned class 1 + the sampled static draws, class 2) over all pixels;
        // the rest keeps bb_mv.comp (backup: sky, particles, draws without a previous camera).
        const double native_px =
            g.sec_px_all ? 100.0 * double(g.sec_px_obj - g.sec_px_blend + g.sec_static_px) /
                               double(g.sec_px_all)
                         : 0.0;
        const double f = double(std::max<u64>(1, g.vs_frames));
        Log(fmt::format(
            "[velocity] native velocity draws {:.1f}/frame of {:.1f} (static {:.1f}, skinned "
            "{:.1f}: ring palette {:.1f}, palette by address {:.1f}) | G-buffer px native {:.1f}% "
            "(sampled statics included), backup bb_mv {:.1f}% | static check |native-cam| "
            "{:.3f} px over {} px ({}) | without previous camera {:.1f}/frame, without previous "
            "palette {:.1f}/frame, blended (backup) {:.1f}/frame, GPU-written skipped {} | CPU "
            "{:.3f} us/draw, copied {:.0f} B/frame",
            double(g.vs_native) / f, double(g.vs_draws) / f, double(g.vs_static) / f,
            double(g.vs_skinned) / f, double(g.vs_skinned_ring) / f,
            double(g.vs_skinned_addr) / f, native_px, g.sec_px_all ? 100.0 - native_px : 0.0,
            st_abs, g.sec_static_px,
            g.sec_static_px == 0 ? "no static px" : (st_abs <= 0.5 ? "PASS" : "FAIL"),
            double(g.vs_no_cam) / f, double(g.vs_no_pal) / f, double(g.vs_blend) / f, g.vs_gpu,
            g.vs_draws ? g.vs_cpu_ns / 1000.0 / double(g.vs_draws) : 0.0, double(g.vs_bytes) / f));
        g.vs_frames = g.vs_draws = g.vs_native = g.vs_static = g.vs_skinned = g.vs_skinned_ring =
            g.vs_skinned_addr = g.vs_no_cam = g.vs_no_pal = g.vs_blend = g.vs_gpu = g.vs_bytes = 0;
        g.vs_cpu_ns = 0;
    }
    Log(fmt::format("[objmv-gpu6] moving test {}: changed palettes kept static {}/s (shared "
                    "constant blocks, read by >= {} draws: {}); CPU per draw: hashed {:.0f} B, "
                    "copied {:.0f} B, dedupe hits {}",
                    g.rel_test ? "shared+rel (BB_OBJMV_MOVE=rel)" : "shared",
                    g.sec_camera_only, g.shared_min, g.sec_shared,
                    g.sec_draws ? double(g.sec_bytes_hashed) / double(g.sec_draws) : 0.0,
                    g.sec_draws ? double(g.sec_bytes_copied) / double(g.sec_draws) : 0.0,
                    g.sec_dedupe));
    g.sec_camera_only = g.sec_dedupe = g.sec_bytes_hashed = g.sec_bytes_copied = 0;
    if (g.sec_ui_skipped != 0) {
        Log(fmt::format("[objmv-ui] screen-space draws (depth test Always/Never) without object "
                        "vectors: {} pipeline lookups/s",
                        g.sec_ui_skipped));
    }
    g.sec_shared = 0;
    g.sec_ui_skipped = 0;
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
