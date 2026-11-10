// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// BB object motion vectors v2: see bb_objmv2.h.

#include "video_core/bb/bb_objmv2.h"

#include <algorithm>
#include <array>
#include <chrono>
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
#include "video_core/bb/bb_upscale.h"
#include "video_core/host_shaders/bb_objmv2_comp.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include <vk_mem_alloc.h>

namespace BB::ObjMv2 {

namespace {

constexpr u32 kSlots = 4;                       // frames in flight (snapshots, tables, stats)
constexpr u64 kArenaBytes = 24ull << 20;        // snapshots a frame
constexpr u64 kTableBytes = 2ull << 20;         // table entries a frame
constexpr u64 kHeaderBytes = 4096;              // zero entry + dummy dword
constexpr u32 kMaxSnapBytes = 64u << 10;        // larger buffers: read the current one
constexpr u32 kStatWords = 8;

enum Cat : u32 { CatSkinned = 0, CatSmallSkel, CatRigid, CatBlend, CatCount };
constexpr const char* kCatName[CatCount] = {"skinned(characters+enemies)",
                                            "small-skeleton(weapons/props)", "rigid",
                                            "blended(particles/effects)"};

struct SnapBuf {
    u64 addr;    // device address of the snapshot (0: none)
    u32 dwords;
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
    bool history = false; // map_prev is exactly the previous frame
    bool reset_pending = false;
    // Per-second statistics.
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    u64 sec_draws = 0, sec_vec_draws = 0, sec_resets = 0, sec_frames = 0, sec_overflow = 0;
    u64 sec_cat[CatCount]{}, sec_cat_vec[CatCount]{};
    double sec_cpu_ns = 0;
    u64 sec_px_obj = 0, sec_px_blend = 0, sec_px_all = 0, sec_chk = 0, sec_eo = 0, sec_ec = 0,
        sec_wo = 0, sec_wc = 0;
    double sec_gpu_ms = 0;
    u32 sec_gpu_n = 0;
    double ts_period = 1.0;
} g;

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
    std::array<u64, 2> ts{};
    const auto r = g.device.getQueryPoolResults(g.queries, slot * 2, 2, sizeof(ts), ts.data(),
                                                sizeof(u64), vk::QueryResultFlagBits::e64);
    if (r == vk::Result::eSuccess && ts[1] >= ts[0]) {
        g.sec_gpu_ms += double(ts[1] - ts[0]) * g.ts_period * 1e-6;
        ++g.sec_gpu_n;
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
    g.ready = true;
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
    if (!regs.depth_control.depth_enable) return Kind::None;
    if (regs.color_control.mode != AmdGpu::ColorControl::OperationMode::Normal) return Kind::None;
    if (regs.IsClipDisabled()) return Kind::None;
    u32 bound = 0;
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        if (regs.color_buffers[cb] && regs.color_target_mask.GetMask(cb)) {
            if (cb == Attachment) return Kind::None;
            ++bound;
        }
    }
    if (bound == 0 || !regs.color_buffers[0] || !regs.color_target_mask.GetMask(0)) {
        return Kind::None;
    }
    const auto& bc = regs.blend_control[0];
    if (!bc.enable) {
        return regs.depth_control.depth_write_enable ? Kind::Opaque : Kind::None;
    }
    // Alpha-blended (particles, effects) only: additive / multiplicative passes (lights, glow)
    // have no surface of their own.
    using F = AmdGpu::BlendControl::BlendFactor;
    if (bc.color_src_factor == F::SrcAlpha && bc.color_dst_factor == F::OneMinusSrcAlpha) {
        return Kind::Blend;
    }
    return Kind::None;
}

void FillKey(Kind kind, u32& mrt_mask, u32& num_color_attachments, Shader::PsColorBuffer& cb7,
             vk::ColorComponentFlags& write_mask7, u32& cb_shader_mask_raw, void* blend7) {
    mrt_mask |= 1u << Attachment;
    num_color_attachments = Attachment + 1;
    std::memset(&cb7, 0, sizeof(cb7));
    cb7.data_format = AmdGpu::DataFormat::Format16_16_16_16;
    cb7.num_format = AmdGpu::NumberFormat::Float;
    cb7.swizzle = AmdGpu::IdentityMapping;
    write_mask7 = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                  vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    cb_shader_mask_raw |= 0xFu << (Attachment * 4);
    auto& bc = *static_cast<AmdGpu::BlendControl*>(blend7);
    std::memset(&bc, 0, sizeof(bc));
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

void OnDraw(Shader::PushData& push, const Shader::Info& vs, Core::MemoryManager* memory,
            u64 vs_hash, u64 identity, u32 count, u32 instances, bool blend, bool ui_phase) {
    push.objmv_flags = 0;
    push.objmv_entry = {u32(g.base_addr), u32(g.base_addr >> 32)}; // zero entry
    if (!g.ready || ui_phase) return;
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
        SnapBuf sb{0, 0};
        const void* src = nullptr;
        u64 size = 0, key = 0;
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
                if (const auto it = g.dedupe.find(key); it != g.dedupe.end()) {
                    sb = it->second;
                    largest = std::max<u32>(largest, u32(size));
                } else if (memory->IsValidMapping(vsharp.base_address, size)) {
                    src = reinterpret_cast<const void*>(vsharp.base_address);
                }
            }
        }
        if (src != nullptr && size >= 4 && size <= kMaxSnapBytes) {
            const u64 bytes = (size + 15) & ~u64(15);
            if (g.arena_used + bytes <= kArenaBytes) {
                std::memcpy(g.mapped + arena_off + g.arena_used, src, size);
                sb.addr = g.base_addr + arena_off + g.arena_used;
                sb.dwords = u32(size / 4);
                g.arena_used += bytes;
                if (key != 0) g.dedupe.emplace(key, sb);
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

    // Last frame's snapshots of the same draw.
    if (g.history) {
        if (const auto it = g.map_prev.find(k); it != g.map_prev.end() && it->second.count == n) {
            const u64 bytes = u64(n) * 16;
            if (n > 0 && g.table_used + bytes <= kTableBytes) {
                const u64 off = SlotTableOffset(g.slot) + g.table_used;
                u32* e = reinterpret_cast<u32*>(g.mapped + off);
                const u64 dummy = g.base_addr + 2048;
                bool any = false;
                for (u32 hnd = 0; hnd < n; ++hnd) {
                    const SnapBuf& pb = g.bufs_prev[it->second.first + hnd];
                    const bool use = pb.addr != 0 && pb.dwords != 0;
                    const u64 a = use ? pb.addr : dummy;
                    e[hnd * 4 + 0] = u32(a);
                    e[hnd * 4 + 1] = u32(a >> 32);
                    e[hnd * 4 + 2] = use ? pb.dwords : 1u;
                    e[hnd * 4 + 3] = use ? 1u : 0u;
                    any |= use;
                }
                g.table_used += bytes;
                const u64 ea = g.base_addr + off;
                push.objmv_entry = {u32(ea), u32(ea >> 32)};
                if (any) {
                    push.objmv_flags = 1;
                    ++g.sec_vec_draws;
                    ++g.sec_cat_vec[cat];
                }
            } else if (n > 0) {
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
    const u32 slot = g.slot;
    ReadStats(slot);
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
    const u32 next = g.prev_cur ^ 1u;
    const vk::ImageMemoryBarrier2 prev_barriers[2] = {
        {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
         .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
         .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
         .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
         .oldLayout = g.prev_valid ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined,
         .newLayout = vk::ImageLayout::eGeneral,
         .image = g.prev[g.prev_cur],
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
    const vk::ImageMemoryBarrier2 images[3] = {obj_ready, prev_barriers[0], prev_barriers[1]};
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                            .pMemoryBarriers = &fill_done,
                                            .imageMemoryBarrierCount = 3,
                                            .pImageMemoryBarriers = images});
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, g.queries, slot * 2);
    const vk::DescriptorImageInfo obj_info{g.sampler, g.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo mv_info{{}, mv_view, vk::ImageLayout::eGeneral};
    const vk::DescriptorBufferInfo stats_info{g.stats, slot * kStatWords * 4, kStatWords * 4};
    const vk::DescriptorImageInfo cur_info{g.sampler, color_view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo prev_info{g.sampler, g.prev_view[g.prev_cur],
                                            vk::ImageLayout::eGeneral};
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
    const bool check = g.check && g.prev_valid && !reset && color_view;
    const std::array<u32, 4> push{render.width, render.height, reset ? 0u : 1u,
                                  (check ? 1u : 0u) | (g.check && color_view ? 2u : 0u)};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g.layout, 0, writes);
    cmd.pushConstants(g.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push.data());
    cmd.dispatch((render.width + 7) / 8, (render.height + 7) / 8, 1);
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, g.queries, slot * 2 + 1);
    const vk::MemoryBarrier2 done{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eHostRead,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &done});
    g.slot_used[slot] = true;
    if (g.check && color_view) {
        g.prev_cur = next;
        g.prev_valid = true;
    }
}

void OnFlip() {
    if (!g.ready) return;
    ++g.frame;
    ++g.sec_frames;
    const bool had_pass = g.cleared;
    // History = exactly this frame, if it had the pass (and no camera cut).
    if (had_pass && !g.reset_pending) {
        std::swap(g.map_prev, g.map_cur);
        std::swap(g.bufs_prev, g.bufs_cur);
        g.history = true;
    } else {
        if (g.history) ++g.sec_resets;
        g.map_prev.clear();
        g.bufs_prev.clear();
        g.history = false;
        if (!had_pass) g.prev_valid = false;
    }
    g.reset_pending = false;
    g.map_cur.clear();
    g.bufs_cur.clear();
    g.occ.clear();
    g.dedupe.clear();
    g.cleared = false;
    g.composed = false;
    g.slot = (g.slot + 1) % kSlots;
    g.arena_used = 0;
    g.table_used = 0;

    const auto now = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(now - g.t0).count();
    if (secs < 1.0) return;
    g.t0 = now;
    const double cov = g.sec_px_all ? 100.0 * double(g.sec_px_obj) / double(g.sec_px_all) : 0.0;
    std::string cats;
    for (u32 c = 0; c < CatCount; ++c) {
        cats += fmt::format("{}{} {}/{}", c ? ", " : "", kCatName[c], g.sec_cat_vec[c],
                            g.sec_cat[c]);
    }
    const double eo = g.sec_chk ? double(g.sec_eo) / 1024.0 / double(g.sec_chk) : 0.0;
    const double ec = g.sec_chk ? double(g.sec_ec) / 1024.0 / double(g.sec_chk) : 0.0;
    Log(fmt::format(
        "draws with object vectors {}/{}, coverage {:.1f}% (blended {:.1f}%), GPU {:.3f} ms "
        "(compose), CPU {:.3f} us/draw added, history resets {}, frames {}, overflow {} | {} | "
        "check: {} px, reprojection error obj {:.4f} vs cam {:.4f} ({}), obj better {} / cam "
        "better {}",
        g.sec_vec_draws, g.sec_draws, cov,
        g.sec_px_all ? 100.0 * double(g.sec_px_blend) / double(g.sec_px_all) : 0.0,
        g.sec_gpu_n ? g.sec_gpu_ms / g.sec_gpu_n : 0.0,
        g.sec_draws ? g.sec_cpu_ns / 1000.0 / double(g.sec_draws) : 0.0, g.sec_resets,
        g.sec_frames, g.sec_overflow, cats, g.sec_chk, eo, ec,
        g.sec_chk == 0 ? "no differing px" : (eo <= ec ? "OBJECT VECTORS WIN" : "CAMERA WINS"),
        g.sec_wo, g.sec_wc));
    g.sec_draws = g.sec_vec_draws = g.sec_resets = g.sec_frames = g.sec_overflow = 0;
    for (u32 c = 0; c < CatCount; ++c) g.sec_cat[c] = g.sec_cat_vec[c] = 0;
    g.sec_cpu_ns = 0;
    g.sec_px_obj = g.sec_px_blend = g.sec_px_all = g.sec_chk = g.sec_eo = g.sec_ec = g.sec_wo =
        g.sec_wc = 0;
    g.sec_gpu_ms = 0;
    g.sec_gpu_n = 0;
}

} // namespace BB::ObjMv2
