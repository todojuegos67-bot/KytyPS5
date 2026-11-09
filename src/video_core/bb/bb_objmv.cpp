// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-FileCopyrightText: Copyright 2026 bbport contributors (deadinside28/bloodborne_pc:
//                         gpu/upscale/vk_object_motion.cpp, docs/motion_vectors.md,
//                         GPL-2.0-or-later; the approach and the bone palette classification)
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-clean-021-objmv: object motion vectors, see bb_objmv.h.

#include "video_core/bb/bb_objmv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fmt/format.h>
#include <tsl/robin_map.h>
#include <xxhash.h>
#include "common/logging/log.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/pixel_format.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/host_shaders/bb_objmv_comp.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include <vk_mem_alloc.h>

namespace BB::ObjMotion {

namespace {

using Clock = std::chrono::steady_clock;

constexpr u32 kFrameSlots = 4;        // parameter ring: frames in flight
constexpr u32 kParamsPerFrame = 4096; // entries of 32 bytes; entry 0 of each slot unused
constexpr u32 kMaxVerticesPerDraw = 1u << 20;
constexpr u32 kMaxInstances = 256;
constexpr u32 kQuerySlots = 8;

enum class Palette : u32 { None, Small, Large, Rigid };

// A draw's history under one identity.
struct Rec {
    u64 hash = 0;      // its constants (palette / object transform)
    u32 seen = 0;      // frame last seen (+1: 0 = never)
    u32 stored = 0;    // frame whose positions it stored (+1)
    u32 base = 0;      // where
    u32 count = 0;
};

struct Pending {
    bool valid = false;
    u64 key = 0, key2 = 0; // identities (key2: by bone palette address, large skeletons)
    u32 prev_base = 0, prev_count = 0;
    bool has_prev = false;
    u32 vertices = 0, instances = 0;
    Palette palette = Palette::None;
};

struct Retired {
    vk::Image image{};
    VmaAllocation alloc{};
    vk::ImageView view{};
    u32 frame = 0;
};

struct State {
    bool tried = false, ready = false;
    vk::Device device{};
    VmaAllocator allocator{};
    vk::Buffer params{};
    VmaAllocation params_alloc{};
    u32* params_mapped = nullptr;
    u64 params_address = 0;
    vk::Buffer positions{};
    VmaAllocation positions_alloc{};
    u64 positions_address = 0;
    u32 per_half = 0;
    // The motion image: attachment 7 (vector + depth bits).
    vk::Image image{};
    VmaAllocation image_alloc{};
    vk::ImageView view{};
    u32 width = 0, height = 0;
    std::vector<Retired> retired;
    bool cleared = false; // this frame's first scene draw cleared it
    bool written = false; // a motion draw wrote it this frame
    bool latched = false; // ... in the frame the flip just ended
    // Compose pass.
    vk::DescriptorSetLayout set_layout{};
    vk::PipelineLayout layout{};
    vk::Pipeline pipeline{};
    vk::QueryPool queries{};
    double ms_per_tick = 0;
    u64 compose_no = 0;
    // History.
    u32 frame = 2; // 0 in a record: never (frame - 1 is never 0)
    u64 scene_depth = 0, scene_depth_next = 0;
    u32 cursor = 0, params_used = 0;
    tsl::robin_map<u64, Rec> recs;
    tsl::robin_map<u64, u32> ordinals;
    tsl::robin_map<u64, std::pair<u32, u32>> index_ranges;
    u32 rigid_seen = 0, rigid_changed = 0;
    bool rigid_camera_bound = false;
    Pending pending;
    // Per second.
    Clock::time_point since = Clock::now();
    u64 s_frames = 0, s_scene = 0, s_tracked = 0, s_own = 0, s_overflow = 0, s_composes = 0;
    u64 s_cpu_ns = 0;
    double s_gpu_ms = 0;
    u64 s_gpu_n = 0;
};
State g;

bool EnvOff() {
    const char* s = std::getenv("BB_OBJMV");
    return s != nullptr && (s[0] == '0' || std::strcmp(s, "off") == 0);
}

bool RigidOn() {
    static const bool on = [] {
        const char* s = std::getenv("BB_OBJMV_RIGID");
        return s == nullptr || s[0] != '0';
    }();
    return on;
}

void Log(const std::string& text) {
    BB::Upscale::Checkpoint("[objmv] " + text);
}

u64 Mix(u64 h, u64 v) {
    // Cheap 64-bit mix (identities only; contents use XXH3).
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h * 0xff51afd7ed558ccdull;
}

bool CreateBuffer(vk::DeviceSize size, bool host, vk::Buffer& buffer, VmaAllocation& alloc,
                  void** mapped, u64& address) {
    const VkBufferCreateInfo ci = vk::BufferCreateInfo{
        .size = size,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                 vk::BufferUsageFlagBits::eShaderDeviceAddress,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    VmaAllocationCreateInfo aci{};
    aci.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    if (host) {
        aci.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    } else {
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }
    VkBuffer raw{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(g.allocator, &ci, &aci, &raw, &alloc, &info) != VK_SUCCESS) {
        return false;
    }
    buffer = raw;
    if (mapped) *mapped = info.pMappedData;
    address = g.device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = buffer});
    return address != 0 && (!mapped || *mapped);
}

bool Init(const Vulkan::Instance& instance) {
    g.tried = true;
    if (EnvOff()) {
        Log("off (BB_OBJMV=0): camera-only motion vectors");
        return false;
    }
    if (!instance.IsBufferDeviceAddressSupported()) {
        Log("off: no buffer device address");
        return false;
    }
    g.device = instance.GetDevice();
    g.allocator = instance.GetAllocator();
    u32 mi = 4;
    if (const char* s = std::getenv("BB_OBJMV_VERTICES_MI")) mi = std::clamp(std::atoi(s), 1, 16);
    g.per_half = mi << 20;
    void* mapped = nullptr;
    if (!CreateBuffer(vk::DeviceSize(kFrameSlots) * kParamsPerFrame * 32, true, g.params,
                      g.params_alloc, &mapped, g.params_address) ||
        !CreateBuffer(vk::DeviceSize(g.per_half) * 2 * 16, false, g.positions, g.positions_alloc,
                      nullptr, g.positions_address)) {
        Log("buffers not created: off");
        return false;
    }
    g.params_mapped = static_cast<u32*>(mapped);
    std::memset(g.params_mapped, 0, size_t(kFrameSlots) * kParamsPerFrame * 32);

    const std::array<vk::DescriptorSetLayoutBinding, 3> bindings{{
        {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute},
    }};
    auto [lr, l] = g.device.createDescriptorSetLayout({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    });
    if (lr != vk::Result::eSuccess) return false;
    g.set_layout = l;
    const vk::PushConstantRange range{vk::ShaderStageFlagBits::eCompute, 0, 16};
    auto [pr, pl] = g.device.createPipelineLayout({
        .setLayoutCount = 1,
        .pSetLayouts = &g.set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    });
    if (pr != vk::Result::eSuccess) return false;
    g.layout = pl;
    const auto module = Vulkan::CompileSPV(BB_OBJMV_COMP, g.device);
    if (!module) return false;
    const vk::ComputePipelineCreateInfo ci{
        .stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
        .layout = g.layout,
    };
    auto [cr, p] = g.device.createComputePipeline({}, ci);
    g.device.destroyShaderModule(module);
    if (cr != vk::Result::eSuccess) {
        Log("compose pass not created: off");
        return false;
    }
    g.pipeline = p;
    auto [qr, q] = g.device.createQueryPool({
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = kQuerySlots * 2,
    });
    if (qr == vk::Result::eSuccess) {
        g.queries = q;
        g.ms_per_tick = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod / 1e6;
    }
    g.recs.reserve(16384);
    g.ordinals.reserve(4096);
    g.ready = true;
    Log(fmt::format("on: scene draws that moved since the last frame get their own motion vectors "
                    "({} Mi positions a frame, {} MiB; rigid objects {})",
                    mi, (u64(g.per_half) * 2 * 16) >> 20,
                    RigidOn() ? "on" : "off (BB_OBJMV_RIGID=0)"));
    return true;
}

void DestroyRetired(bool all) {
    std::erase_if(g.retired, [all](const Retired& r) {
        if (!all && g.frame < r.frame + 8) return false;
        g.device.destroyImageView(r.view);
        vmaDestroyImage(g.allocator, static_cast<VkImage>(r.image), r.alloc);
        return true;
    });
}

// Never shrinks; a larger one replaces it (the old one is destroyed 8 frames later: no wait).
bool EnsureImage(u32 width, u32 height) {
    if (g.image && g.width >= width && g.height >= height) return true;
    width = std::max(width, g.width);
    height = std::max(height, g.height);
    if (g.image) {
        g.retired.push_back({g.image, g.image_alloc, g.view, g.frame});
        g.image = vk::Image{};
        g.view = vk::ImageView{};
        g.width = g.height = 0;
    }
    const VkImageCreateInfo ci = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32G32Uint,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage |
                 vk::ImageUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkImage raw{};
    if (vmaCreateImage(g.allocator, &ci, &aci, &raw, &g.image_alloc, nullptr) != VK_SUCCESS) {
        return false;
    }
    g.image = raw;
    auto [result, v] = g.device.createImageView(vk::ImageViewCreateInfo{
        .image = g.image,
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR32G32Uint,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
    if (result != vk::Result::eSuccess) {
        vmaDestroyImage(g.allocator, raw, g.image_alloc);
        g.image = vk::Image{};
        return false;
    }
    g.view = v;
    g.width = width;
    g.height = height;
    g.cleared = false;
    Log(fmt::format("motion image {}x{}", width, height));
    return true;
}

// The draw's bone palette among its vertex constant buffers (deadinside28's ClassifyBuffer):
// 864 bytes is the scene block; 640..16384 a character skeleton; 96..639 in 48-byte (3x4)
// matrices a small skeleton (weapons, props). No palette but own constants (16..1024 bytes, not
// the scene block): a rigid object's transform. `hash`: their contents (large: sampled).
Palette Classify(const Shader::Info& vs, VAddr& large_address, u64& hash) {
    Palette found = Palette::None;
    u64 rigid = 0, small = 0;
    bool any = false;
    for (const auto& desc : vs.buffers) {
        if (desc.IsSpecial() || desc.is_written) continue;
        const auto sharp = desc.GetSharp(vs);
        const u32 size = sharp.GetSize();
        if (sharp.base_address == 0 || size == 864 || size < 16 || size > 16384) continue;
        const auto* p = reinterpret_cast<const u8*>(sharp.base_address);
        if (size >= 640 && size % 16 == 0) {
            // Skinned: the first and last KiB of the palette (cheap; animated bones show there).
            large_address = sharp.base_address;
            const u32 head = std::min<u32>(size, 1024);
            u64 h = XXH3_64bits(p, head);
            if (size > head) h = XXH3_64bits_withSeed(p + size - head, head, h);
            hash = h;
            return Palette::Large;
        }
        if (size <= 1024) {
            rigid = XXH3_64bits_withSeed(p, size, rigid);
            any = true;
        }
        if (size >= 96 && size < 640 && size % 48 == 0 && found == Palette::None) {
            found = Palette::Small;
            small = XXH3_64bits(p, size);
        }
    }
    if (found == Palette::Small) {
        hash = small;
        return found;
    }
    if (any && RigidOn()) {
        hash = rigid;
        return Palette::Rigid;
    }
    return Palette::None;
}

struct Range {
    bool valid = false;
    u32 min = 0, max = 0;
};
// The [min, max] vertex index of an indexed draw (cached per index block: address, count, type,
// head and tail of its contents): only the vertices it names get position slots.
Range IndexRange(const AmdGpu::Regs& regs, u32 index_offset) {
    const bool i16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const u32 size = i16 ? 2u : 4u;
    const VAddr base = regs.index_base_address.Address<VAddr>();
    const u32 count = std::min(regs.num_indices, regs.max_index_size);
    if (base == 0 || count == 0 || count > (4u << 20)) return {};
    const VAddr addr = base + u64(index_offset) * size;
    const u8* p = reinterpret_cast<const u8*>(addr);
    const u64 bytes = u64(count) * size;
    const u64 head = std::min<u64>(bytes, 256);
    u64 key = XXH3_64bits_withSeed(p, head, addr ^ (u64(count) << 1) ^ size);
    key = XXH3_64bits_withSeed(p + (bytes - head), head, key);
    if (const auto it = g.index_ranges.find(key); it != g.index_ranges.end()) {
        return {true, it->second.first, it->second.second};
    }
    u32 lo = ~0u, hi = 0;
    const u32 restart = i16 ? 0xFFFFu : 0xFFFFFFFFu;
    for (u32 i = 0; i < count; ++i) {
        u32 v;
        if (i16) {
            u16 x;
            std::memcpy(&x, p + size_t(i) * 2, 2);
            v = x;
        } else {
            std::memcpy(&v, p + size_t(i) * 4, 4);
        }
        if (v == restart) continue;
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    if (lo > hi) return {};
    if (g.index_ranges.size() > 65536) g.index_ranges.clear();
    g.index_ranges[key] = {lo, hi};
    return {true, lo, hi};
}

} // namespace

bool WantPipeline(const Vulkan::Instance& instance, const AmdGpu::Regs& regs) {
    if (!g.ready) {
        if (g.tried || !BB::Upscale::Capture()) return false;
        if (!Init(instance)) return false;
    }
    if (g.scene_depth == 0 || !BB::Upscale::Capture()) return false;
    if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::VgtStages::Vs) return false;
    if (!regs.depth_buffer.DepthValid() || regs.depth_buffer.NumSamples() != 1 ||
        regs.depth_buffer.DepthAddress() != g.scene_depth) {
        return false;
    }
    if (!regs.depth_control.depth_enable || !regs.depth_control.depth_write_enable) return false;
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable) return false;
    if (regs.IsClipDisabled()) return false;
    if (regs.blend_control[0].enable) return false; // dual source blending takes location 0/1
    u32 bound = 0;
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        if (regs.color_buffers[cb] && regs.color_target_mask.GetMask(cb)) {
            if (cb == Attachment) return false;
            ++bound;
        }
    }
    return bound >= 3;
}

bool Decide(const AmdGpu::Regs& regs, const Shader::Info& vs,
            const Shader::Gcn::FetchShaderData* fetch, u64 vs_hash, bool indirect) {
    const auto t0 = Clock::now();
    g.pending.valid = false;
    ++g.s_scene;
    const auto done = [&](bool r) {
        g.s_cpu_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0)
                          .count();
        return r;
    };
    // Indirect draws: their counts are on the GPU; camera motion.
    if (indirect || fetch == nullptr || fetch->Empty() || fetch->attributes.empty()) {
        return done(false);
    }
    VAddr large_address = 0;
    u64 hash = 0;
    const Palette palette = Classify(vs, large_address, hash);
    if (palette == Palette::None) return done(false);
    const auto vb = fetch->attributes[0].GetSharp(vs);
    if (vb.GetStride() == 0 || vb.num_records == 0) return done(false);
    const u32 instances = regs.num_instances.NumInstances();
    if (instances > kMaxInstances) return done(false);

    // Identity across frames: shader, counts, class (+ the streams for non-skinned draws; a
    // skinned draw's streams may live in a per-frame ring), and its order among identical
    // draws this frame. Large skeletons also by their palette's address (one per character).
    u64 id = Mix(vs_hash, regs.num_indices);
    id = Mix(id, (u64(vb.num_records) << 32) | instances);
    id = Mix(id, u64(palette));
    if (palette != Palette::Large) {
        id = Mix(id, vb.base_address);
        id = Mix(id, regs.index_base_address.Address());
    }
    const u32 ordinal = g.ordinals[id]++;
    const u64 key = Mix(id, ordinal);
    u64 key2 = 0;
    if (palette == Palette::Large && large_address != 0) {
        key2 = Mix(Mix(id, large_address), 0x736b696eull);
    }

    const u32 last = g.frame - 1;
    Rec rec = g.recs[key];
    if (key2 != 0) {
        if (const auto it = g.recs.find(key2); it != g.recs.end() && it->second.seen == last) {
            rec = it->second;
        }
    }
    const bool changed = rec.seen == last && rec.hash != hash;
    {
        Rec& r = g.recs[key];
        r.hash = hash;
        r.seen = g.frame;
        if (key2 != 0) {
            Rec& r2 = g.recs[key2];
            r2.hash = hash;
            r2.seen = g.frame;
        }
    }
    if (palette == Palette::Rigid) {
        ++g.rigid_seen;
        if (changed) ++g.rigid_changed;
        // Most rigid draws changing every frame: the camera is in their constants (camera
        // motion exact for them); their positions would take the budget from the skinned ones.
        if (g.rigid_camera_bound) return done(false);
    }
    if (!changed) return done(false);
    if (g.params_used + 1 >= kParamsPerFrame) {
        ++g.s_overflow;
        return done(false);
    }
    auto& p = g.pending;
    p.valid = true;
    p.key = key;
    p.key2 = key2;
    p.has_prev = rec.stored == last;
    p.prev_base = rec.base;
    p.prev_count = rec.count;
    p.vertices = vb.num_records;
    p.instances = instances;
    p.palette = palette;
    return done(true);
}

void FillKey(bool tracked, u32& mrt_mask, u32& num_color_attachments, Shader::PsColorBuffer& cb7,
             vk::ColorComponentFlags& write_mask7) {
    if (mrt_mask & (1u << Attachment)) return; // the shader writes MRT 7 itself
    mrt_mask |= 1u << Attachment;
    num_color_attachments = Attachment + 1;
    std::memset(&cb7, 0, sizeof(cb7));
    cb7.data_format = AmdGpu::DataFormat::Format32_32;
    cb7.num_format = AmdGpu::NumberFormat::Uint;
    cb7.swizzle = AmdGpu::IdentityMapping;
    write_mask7 = tracked ? vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
                          : vk::ColorComponentFlags{};
}

bool IsMotionKey(u32 mrt_mask, const Shader::PsColorBuffer& cb7) {
    return g.ready && (mrt_mask & (1u << Attachment)) != 0 &&
           cb7.data_format == AmdGpu::DataFormat::Format32_32 &&
           cb7.num_format == AmdGpu::NumberFormat::Uint;
}

void Attach(Vulkan::Scheduler& scheduler, Vulkan::RenderState& state) {
    if (!g.ready || state.num_color_attachments <= Attachment) return;
    if (!EnsureImage(state.width, state.height)) return;
    auto& attachment = state.color_attachments[Attachment];
    attachment.image_view = g.view;
    attachment.image_layout = vk::ImageLayout::eGeneral;
    attachment.clear_value = {};
    attachment.is_clear = 0;
    if (g.cleared) return;
    // The frame's first scene draw: clear outside the render pass (every later scene draw then
    // has the same render state: no extra render pass restarts), and order last frame's
    // positions (vertex shader stores) before this frame's loads / stores.
    g.cleared = true;
    scheduler.EndRendering();
    const auto cmd = scheduler.CommandBuffer();
    const vk::MemoryBarrier2 positions{
        .srcStageMask = vk::PipelineStageFlagBits2::eVertexShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eVertexShader,
        .dstAccessMask =
            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
    };
    vk::ImageMemoryBarrier2 b{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eClear,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = g.image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                            .pMemoryBarriers = &positions,
                                            .imageMemoryBarrierCount = 1,
                                            .pImageMemoryBarriers = &b});
    const vk::ClearColorValue none{std::array<uint32_t, 4>{0u, 0xFFFFFFFFu, 0u, 0u}};
    const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    cmd.clearColorImage(g.image, vk::ImageLayout::eGeneral, none, range);
    b.srcStageMask = vk::PipelineStageFlagBits2::eClear;
    b.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
    b.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    b.dstAccessMask =
        vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead;
    b.oldLayout = vk::ImageLayout::eGeneral;
    cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                            .pImageMemoryBarriers = &b});
}

void OnDraw(Shader::PushData& push, const AmdGpu::Regs& regs, bool indexed, u32 index_offset,
            u32 vertex_offset, u32 instance_offset) {
    push.motion_param = 0;
    push.motion_params_addr = {u32(g.params_address), u32(g.params_address >> 32)};
    push.motion_positions_addr = {u32(g.positions_address), u32(g.positions_address >> 32)};
    if (!g.ready || !g.pending.valid) return;
    const auto t0 = Clock::now();
    auto p = g.pending;
    g.pending.valid = false;
    u32 vertices = p.vertices;
    u32 range_first = 0;
    if (indexed) {
        if (const auto r = IndexRange(regs, index_offset); r.valid && r.max < vertices) {
            range_first = r.min;
            vertices = r.max - r.min + 1;
        }
    }
    const u64 count = u64(vertices) * p.instances;
    const auto finish = [&] {
        g.s_cpu_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0)
                          .count();
    };
    if (vertices == 0 || count > kMaxVerticesPerDraw || g.cursor + count > g.per_half) {
        ++g.s_overflow;
        return finish();
    }
    const u32 half = g.frame & 1;
    const u32 store_base = half * g.per_half + g.cursor;
    g.cursor += static_cast<u32>(count);
    u32 flags = 1; // store
    u32 load_base = 0;
    if (p.has_prev && p.prev_count == count && p.prev_base != store_base) {
        flags |= 2; // load
        load_base = p.prev_base;
        ++g.s_own;
    }
    ++g.s_tracked;
    for (const u64 k : {p.key, p.key2}) {
        if (k == 0) continue;
        Rec& r = g.recs[k];
        r.stored = g.frame;
        r.base = store_base;
        r.count = static_cast<u32>(count);
    }
    const u32 slot = g.frame % kFrameSlots;
    const u32 index = slot * kParamsPerFrame + (++g.params_used);
    u32* entry = g.params_mapped + size_t(index) * 8;
    entry[0] = store_base;
    entry[1] = load_base;
    entry[2] = vertices;
    entry[3] = flags;
    entry[4] = vertex_offset + range_first;
    entry[5] = instance_offset;
    entry[6] = p.instances;
    entry[7] = 0;
    push.motion_param = index;
    g.written = true;
    finish();
}

void NoteSceneDepth(u64 depth_address) {
    g.scene_depth_next = depth_address;
}

void Compose(vk::CommandBuffer cmd, vk::ImageView mv_view, vk::ImageView depth_view,
             vk::Sampler sampler, vk::Extent2D render) {
    if (!g.ready || !g.latched || !g.image || g.width < render.width ||
        g.height < render.height) {
        return;
    }
    const u32 q = static_cast<u32>(g.compose_no % kQuerySlots);
    if (g.queries) {
        if (g.compose_no >= 4) {
            // Four composes back: long finished; not ready -> skipped (never waited for).
            const u32 r = static_cast<u32>((g.compose_no - 4) % kQuerySlots);
            u64 t[2]{};
            if (g.device.getQueryPoolResults(g.queries, r * 2, 2, sizeof(t), t, sizeof(u64),
                                             vk::QueryResultFlagBits::e64) ==
                    vk::Result::eSuccess &&
                t[1] >= t[0]) {
                g.s_gpu_ms += double(t[1] - t[0]) * g.ms_per_tick;
                ++g.s_gpu_n;
            }
        }
        cmd.resetQueryPool(g.queries, q * 2, 2);
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, g.queries, q * 2);
    }
    ++g.compose_no;
    // The G-buffer's attachment 7 and the camera pass's vectors, for this pass.
    const vk::MemoryBarrier2 mb{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput |
                        vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask =
            vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &mb});
    const vk::DescriptorImageInfo obj_info{{}, g.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo mv_info{{}, mv_view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo depth_info{sampler, depth_view,
                                             vk::ImageLayout::eDepthStencilReadOnlyOptimal};
    const std::array writes{
        vk::WriteDescriptorSet{.dstBinding = 0,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageImage,
                               .pImageInfo = &obj_info},
        vk::WriteDescriptorSet{.dstBinding = 1,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageImage,
                               .pImageInfo = &mv_info},
        vk::WriteDescriptorSet{.dstBinding = 2,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &depth_info},
    };
    // Depth tolerance: a 24-bit / 16-bit unorm depth target quantizes what the fragment wrote.
    const float tolerance = 1.0f / 32768.0f;
    u32 tol_bits;
    std::memcpy(&tol_bits, &tolerance, 4);
    const std::array<u32, 4> push{render.width, render.height, tol_bits, 0u};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g.layout, 0, writes);
    cmd.pushConstants(g.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push.data());
    cmd.dispatch((render.width + 7) / 8, (render.height + 7) / 8, 1);
    if (g.queries) {
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, g.queries, q * 2 + 1);
    }
    ++g.s_composes;
}

void OnFlip() {
    if (!g.ready) return;
    // The frame that just ended: what Compose (right after, at this present) reads.
    g.latched = g.written;
    g.written = false;
    g.cleared = false;
    g.scene_depth = g.scene_depth_next != 0 ? g.scene_depth_next : g.scene_depth;
    ++g.frame;
    g.cursor = 0;
    g.params_used = 0;
    g.ordinals.clear();
    g.rigid_camera_bound = g.rigid_seen >= 64 && g.rigid_changed * 2 > g.rigid_seen;
    g.rigid_seen = g.rigid_changed = 0;
    if (g.recs.size() > 200000) g.recs.clear(); // forgets history once; nothing breaks
    DestroyRetired(false);
    ++g.s_frames;
    const auto now = Clock::now();
    const double secs = std::chrono::duration<double>(now - g.since).count();
    if (secs < 1.0) return;
    const double f = double(std::max<u64>(g.s_frames, 1));
    LOG_INFO(Render_Vulkan,
             "[objmv] draws with own vector {:.0f} of {:.0f}, cpu {:.3f} ms, gpu {:.3f} ms "
             "(per frame; tracked {:.0f}, over budget {}, composes {} of {} frames{})",
             g.s_own / f, g.s_scene / f, double(g.s_cpu_ns) / 1e6 / f,
             g.s_gpu_n ? g.s_gpu_ms / double(g.s_gpu_n) : 0.0, g.s_tracked / f, g.s_overflow,
             g.s_composes, g.s_frames, g.rigid_camera_bound ? ", rigid paused: camera in them" : "");
    g.since = now;
    g.s_frames = g.s_scene = g.s_tracked = g.s_own = g.s_overflow = g.s_composes = 0;
    g.s_cpu_ns = 0;
    g.s_gpu_ms = 0;
    g.s_gpu_n = 0;
}

} // namespace BB::ObjMotion
