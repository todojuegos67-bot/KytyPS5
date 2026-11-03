// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-FileCopyrightText: Copyright 2026 bbport contributors (deadinside28/bloodborne_pc:
//                         gpu/upscale/vk_object_motion.cpp, motion_history.h, docs/motion_vectors.md,
//                         GPL-2.0-or-later; the approach and the bone palette classification)
// SPDX-License-Identifier: GPL-2.0-or-later
// BB object motion vectors: see bb_objmv.h.

#include "video_core/bb/bb_objmv.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <fmt/format.h>
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

constexpr u32 kFrameSlots = 4;        // parameter ring: frames in flight
constexpr u32 kParamsPerFrame = 8192; // entries of 32 bytes; entry 0 of slot 0 is all zero
constexpr u32 kMaxVerticesPerDraw = 1u << 20;
constexpr u32 kMaxInstances = 256;

struct Entry {
    u32 base = 0;
    u32 count = 0;
};

struct State {
    bool tried = false;
    bool ready = false;
    bool off = false;
    vk::Device device{};
    VmaAllocator allocator{};
    // Parameters (host visible, coherent) and positions (device local: two halves of vec4).
    vk::Buffer params{};
    VmaAllocation params_alloc{};
    u32* params_mapped = nullptr;
    u64 params_address = 0;
    vk::Buffer positions{};
    VmaAllocation positions_alloc{};
    u64 positions_address = 0;
    u32 per_half = 0; // positions per frame
    // The motion image (render-target size, R32G32_UINT: half2 vector, valid).
    vk::Image image{};
    VmaAllocation image_alloc{};
    vk::ImageView view{};
    u32 width = 0, height = 0;
    bool written = false;   // this frame's G-buffer pass wrote it
    bool latched = false;   // the same, for the frame that the guest flip has just ended
    bool cleared = false;   // this frame's first motion pass cleared it
    // The compose pass.
    vk::DescriptorSetLayout set_layout{};
    vk::PipelineLayout layout{};
    vk::Pipeline pipeline{};
    // History.
    u64 frame = 0;
    u32 cursor = 0;
    u32 params_used = 0;
    std::unordered_map<u64, Entry> prev, cur;
    std::unordered_map<u64, u32> ordinals;
    std::unordered_map<u64, u64> palettes; // small skeletons: last palette hash
    // Stats (since the last report).
    u64 st_motion_draws = 0, st_tracked = 0, st_loaded = 0, st_overflow = 0, st_gated = 0,
        st_frames = 0, st_composed = 0, st_composed_late = 0, st_overflow_large = 0;
    u32 st_overflow_max = 0; // largest dropped draw (positions)
};
State g;

bool EnvOff() {
    const char* s = std::getenv("BB_OBJMV");
    return s != nullptr && (s[0] == '0' || std::string(s) == "off");
}

void Log(const std::string& text) {
    LOG_INFO(Render_Vulkan, "[BB_OBJMV] {}", text);
    BB::Upscale::Checkpoint("[objmv] " + text);
}

bool CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, bool host, vk::Buffer& buffer,
                  VmaAllocation& alloc, void** mapped, u64& address) {
    const VkBufferCreateInfo ci = vk::BufferCreateInfo{
        .size = size,
        .usage = usage | vk::BufferUsageFlagBits::eShaderDeviceAddress,
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
    if (g.tried) return g.ready;
    g.tried = true;
    if (EnvOff()) {
        g.off = true;
        Log("off (BB_OBJMV=0): camera-only motion vectors");
        return false;
    }
    g.device = instance.GetDevice();
    g.allocator = instance.GetAllocator();
    // 4 Mi positions a half (128 MiB of device memory with both halves). Build 77 at 2 Mi dropped
    // 59256 of the 99109 draws that needed positions in 1800 frames (60 %) over the budget.
    u32 mi = 4;
    if (const char* s = std::getenv("BB_OBJMV_VERTICES_MI")) mi = std::clamp(std::atoi(s), 1, 16);
    g.per_half = mi << 20;
    void* mapped = nullptr;
    if (!CreateBuffer(vk::DeviceSize(kFrameSlots) * kParamsPerFrame * 32,
                      vk::BufferUsageFlagBits::eStorageBuffer, true, g.params, g.params_alloc,
                      &mapped, g.params_address) ||
        !CreateBuffer(vk::DeviceSize(g.per_half) * 2 * 16, vk::BufferUsageFlagBits::eStorageBuffer,
                      false, g.positions, g.positions_alloc, nullptr, g.positions_address)) {
        Log("buffers not created: off");
        return false;
    }
    g.params_mapped = static_cast<u32*>(mapped);
    std::memset(g.params_mapped, 0, size_t(kFrameSlots) * kParamsPerFrame * 32);

    const std::array bindings{
        vk::DescriptorSetLayoutBinding{.binding = 0,
                                       .descriptorType = vk::DescriptorType::eStorageImage,
                                       .descriptorCount = 1,
                                       .stageFlags = vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{.binding = 1,
                                       .descriptorType = vk::DescriptorType::eStorageImage,
                                       .descriptorCount = 1,
                                       .stageFlags = vk::ShaderStageFlagBits::eCompute},
    };
    auto [set_result, set_layout] =
        g.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
    const vk::PushConstantRange push{
        .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = 16};
    auto [layout_result, layout] = g.device.createPipelineLayout(vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push,
    });
    const auto module = Vulkan::CompileSPV(BB_OBJMV_COMP, g.device);
    auto [pipe_result, pipeline] = g.device.createComputePipeline(
        {}, vk::ComputePipelineCreateInfo{
                .stage = vk::PipelineShaderStageCreateInfo{.stage =
                                                               vk::ShaderStageFlagBits::eCompute,
                                                           .module = module,
                                                           .pName = "main"},
                .layout = layout,
            });
    g.device.destroyShaderModule(module);
    if (set_result != vk::Result::eSuccess || layout_result != vk::Result::eSuccess ||
        pipe_result != vk::Result::eSuccess) {
        Log("compose pass not created: off");
        return false;
    }
    g.set_layout = set_layout;
    g.layout = layout;
    g.pipeline = pipeline;
    g.ready = true;
    Log(fmt::format("on: skinned / animated G-buffer draws get their own motion vectors "
                    "({} Mi positions a frame, {} MiB)",
                    mi, (u64(g.per_half) * 2 * 16) >> 20));
    return true;
}

void DestroyImage() {
    if (g.view) g.device.destroyImageView(g.view);
    if (g.image) vmaDestroyImage(g.allocator, static_cast<VkImage>(g.image), g.image_alloc);
    g.view = vk::ImageView{};
    g.image = vk::Image{};
    g.image_alloc = {};
    g.width = g.height = 0;
}

bool EnsureImage(u32 width, u32 height) {
    if (g.image && g.width == width && g.height == height) return true;
    if (g.image) {
        (void)g.device.waitIdle();
        DestroyImage();
    }
    const VkImageCreateInfo ci = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32G32Uint,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage,
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
    auto [result, view] = g.device.createImageView(vk::ImageViewCreateInfo{
        .image = g.image,
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR32G32Uint,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
    if (result != vk::Result::eSuccess) {
        DestroyImage();
        return false;
    }
    g.view = view;
    g.width = width;
    g.height = height;
    g.cleared = false; // new image: undefined layout, cleared by the next motion pass
    Log(fmt::format("motion image {}x{}", width, height));
    return true;
}

void ImageBarrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout old_layout,
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
    cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                            .pImageMemoryBarriers = &barrier});
}

u64 Hash(const void* data, size_t size, u64 seed = 0) {
    return XXH3_64bits_withSeed(data, size, seed);
}

// The draw's bone palette among its vertex shader's constant buffers (deadinside28's
// Motion::ClassifyBuffer): 864 bytes is the scene block; 640..16384 a character skeleton; 96..639
// in 48-byte (3x4) matrices a small skeleton (weapons, props: most of them static).
enum class Palette { None, Small, Large };
Palette Classify(const Shader::Info& vs, VAddr& small_address, u32& small_size) {
    Palette found = Palette::None;
    for (const auto& desc : vs.buffers) {
        if (desc.IsSpecial() || desc.is_written) continue;
        const auto sharp = desc.GetSharp(vs);
        const u32 size = sharp.GetSize();
        if (size == 864 || size < 96 || size > 16384) continue;
        if (size >= 640 && size % 16 == 0) return Palette::Large;
        if (size < 640 && size % 48 == 0 && found == Palette::None) {
            found = Palette::Small;
            small_address = sharp.base_address;
            small_size = size;
        }
    }
    return found;
}

} // namespace

bool WantPipeline(const AmdGpu::Regs& regs) {
    if (!g.ready || !BB::Upscale::Enabled()) return false;
    if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::VgtStages::Vs) return false;
    if (!regs.depth_buffer.DepthValid() || regs.depth_buffer.NumSamples() != 1) return false;
    const auto& plan = BB::Upscale::CurrentPlan();
    if (plan.depth_addr == 0 || regs.depth_buffer.DepthAddress() != plan.depth_addr) return false;
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable) return false;
    if (regs.IsClipDisabled()) return false;
    if (regs.blend_control[0].enable) return false; // dual source blending would take location 0/1
    u32 bound = 0;
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        if (regs.color_buffers[cb] && regs.color_target_mask.GetMask(cb)) {
            if (cb == Attachment) return false;
            ++bound;
        }
    }
    return bound >= 3;
}

void FillKey(u32& mrt_mask, u32& num_color_attachments, Shader::PsColorBuffer& cb7,
             vk::ColorComponentFlags& write_mask7) {
    if (mrt_mask & (1u << Attachment)) return; // the shader writes MRT 7 itself
    mrt_mask |= 1u << Attachment;
    num_color_attachments = Attachment + 1;
    std::memset(&cb7, 0, sizeof(cb7));
    cb7.data_format = AmdGpu::DataFormat::Format32_32;
    cb7.num_format = AmdGpu::NumberFormat::Uint;
    cb7.swizzle = AmdGpu::IdentityMapping;
    write_mask7 = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG;
}

bool IsMotionKey(u32 mrt_mask, const Shader::PsColorBuffer& cb7) {
    return (mrt_mask & (1u << Attachment)) != 0 &&
           cb7.data_format == AmdGpu::DataFormat::Format32_32 &&
           cb7.num_format == AmdGpu::NumberFormat::Uint && g.ready;
}

void Attach(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
            Vulkan::RenderState& state) {
    if (!g.ready || state.num_color_attachments <= Attachment) return;
    // The G-buffer targets' size (attachments 0..6 already limited state.width / height).
    if (!EnsureImage(state.width, state.height)) return;
    auto& attachment = state.color_attachments[Attachment];
    attachment.image_view = g.view;
    attachment.image_layout = vk::ImageLayout::eGeneral;
    attachment.clear_value = {0u, 0u, 0u, 0u};
    attachment.is_clear = 0;
    if (!g.cleared) {
        // First motion pass of the frame: last frame's compose read it (compute).
        scheduler.EndRendering();
        ImageBarrier(scheduler.CommandBuffer(), g.image, vk::ImageLayout::eUndefined,
                     vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
                     vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                     vk::AccessFlagBits2::eColorAttachmentWrite |
                         vk::AccessFlagBits2::eColorAttachmentRead);
        attachment.is_clear = 1;
        g.cleared = true;
        g.written = true;
    }
}

void EnsureInit(const Vulkan::Instance& instance) {
    if (!g.tried && BB::Upscale::Enabled()) Init(instance);
}

void OnDraw(const Vulkan::Instance& instance, Shader::PushData& push, bool motion_pipeline,
            const Shader::Info& vs, const Shader::Gcn::FetchShaderData& fetch,
            const AmdGpu::Regs& regs, bool indexed, u32 index_offset, u32 vertex_offset,
            u32 instance_offset, u64 vs_hash) {
    if (!motion_pipeline || !g.ready) return;
    push.motion_param = 0;
    push.motion_params_addr = {u32(g.params_address), u32(g.params_address >> 32)};
    push.motion_positions_addr = {u32(g.positions_address), u32(g.positions_address >> 32)};
    ++g.st_motion_draws;

    VAddr palette_address = 0;
    u32 palette_size = 0;
    const Palette palette = Classify(vs, palette_address, palette_size);
    if (palette == Palette::None || fetch.Empty() || fetch.attributes.empty()) return;
    const auto vb = fetch.attributes[0].GetSharp(vs);
    if (vb.GetStride() == 0) return;
    const u32 vertices = vb.num_records;
    const u32 instances = regs.num_instances.NumInstances();
    if (vertices == 0 || vertices > kMaxVerticesPerDraw || instances > kMaxInstances ||
        u64(vertices) * instances > kMaxVerticesPerDraw) {
        return;
    }
    const u32 count = vertices * instances;

    // The draw's identity across frames: shader, vertex and index streams, range, instances,
    // and its order among identical draws this frame.
    std::array<u64, 8> id{};
    id[0] = vs_hash;
    id[1] = vb.base_address;
    id[2] = indexed ? regs.index_base_address.Address() + u64(index_offset) : 0;
    id[3] = regs.num_indices;
    id[4] = (u64(vertex_offset) << 32) | instance_offset;
    id[5] = (u64(instances) << 32) | u32(regs.index_size);
    id[6] = fetch.attributes.size() > 1 ? fetch.attributes[1].GetSharp(vs).base_address : 0;
    const u64 base_key = Hash(id.data(), sizeof(id));
    const u32 ordinal = g.ordinals[base_key]++;
    const u64 key = Hash(&ordinal, sizeof(ordinal), base_key);

    if (palette == Palette::Small) {
        // Most small skeletons are static map pieces: while the palette does not change the
        // camera motion is exact and nothing is stored.
        const u64 h = Hash(reinterpret_cast<const void*>(palette_address), palette_size);
        auto [it, inserted] = g.palettes.try_emplace(key, h);
        const bool changed = !inserted && it->second != h;
        it->second = h;
        if (!changed) {
            ++g.st_gated;
            return;
        }
    }

    const u32 half = static_cast<u32>(g.frame & 1);
    if (g.cursor + count > g.per_half || g.params_used + 1 >= kParamsPerFrame) {
        // Dropped draws keep camera motion. Split by palette class, with the largest one, so the
        // next log shows whether character skeletons are among them.
        ++g.st_overflow;
        if (palette == Palette::Large) ++g.st_overflow_large;
        g.st_overflow_max = std::max(g.st_overflow_max, count);
        return;
    }
    const u32 store_base = half * g.per_half + g.cursor;
    g.cursor += count;
    u32 flags = 1; // store
    u32 load_base = 0;
    if (const auto it = g.prev.find(key); it != g.prev.end() && it->second.count == count) {
        load_base = it->second.base;
        flags |= 2;
        ++g.st_loaded;
    }
    g.cur[key] = {store_base, count};
    ++g.st_tracked;

    const u32 slot = static_cast<u32>(g.frame % kFrameSlots);
    const u32 index = slot * kParamsPerFrame + (++g.params_used);
    u32* entry = g.params_mapped + size_t(index) * 8;
    entry[0] = store_base;
    entry[1] = load_base;
    entry[2] = vertices;
    entry[3] = flags;
    entry[4] = vertex_offset;
    entry[5] = instance_offset;
    entry[6] = instances;
    entry[7] = 0;
    push.motion_param = index;
}

void Compose(vk::CommandBuffer cmd, vk::Image mv_image, vk::ImageView mv_view, vk::Extent2D render,
             bool reset, bool late) {
    // In the frame's own command stream (before the flip) the G-buffer pass of this frame is
    // `written`. The presenter's late upscale runs after the flip (OnFlip has already cleared
    // `written`), so it reads the flag that OnFlip latched for the frame it presents.
    const bool wrote = late ? g.latched : g.written;
    if (!g.ready || !wrote || reset || g.width != render.width || g.height != render.height) {
        return;
    }
    ImageBarrier(cmd, g.image, vk::ImageLayout::eGeneral,
                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                 vk::AccessFlagBits2::eColorAttachmentWrite,
                 vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    ImageBarrier(cmd, mv_image, vk::ImageLayout::eGeneral,
                 vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                 vk::PipelineStageFlagBits2::eComputeShader,
                 vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    const vk::DescriptorImageInfo obj_info{{}, g.view, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo mv_info{{}, mv_view, vk::ImageLayout::eGeneral};
    const std::array writes{
        vk::WriteDescriptorSet{.dstBinding = 0,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageImage,
                               .pImageInfo = &obj_info},
        vk::WriteDescriptorSet{.dstBinding = 1,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageImage,
                               .pImageInfo = &mv_info},
    };
    const std::array<u32, 4> push{render.width, render.height, 1u, 0u};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g.layout, 0, writes);
    cmd.pushConstants(g.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push.data());
    cmd.dispatch((render.width + 7) / 8, (render.height + 7) / 8, 1);
    ++g.st_composed;
    if (late) ++g.st_composed_late;
}

void OnFlip() {
    if (!g.ready) return;
    ++g.frame;
    ++g.st_frames;
    g.prev = std::move(g.cur);
    g.cur.clear();
    g.ordinals.clear();
    g.cursor = 0;
    g.params_used = 0;
    g.cleared = false;
    g.latched = g.written;
    g.written = false;
    if (g.palettes.size() > 65536) g.palettes.clear();
    if (g.st_frames >= 1800) {
        Log(fmt::format("last {} frames: {} G-buffer draws with the motion pipeline, {} tracked "
                        "(skinned / moving), {} with last frame's positions (object vectors), {} "
                        "small skeletons unchanged (camera motion), {} over the budget, composed "
                        "into the motion vectors {} times ({} after the flip); budget {} positions "
                        "a half: {} of the over-budget draws were large skeletons, the largest "
                        "{} positions",
                        g.st_frames, g.st_motion_draws, g.st_tracked, g.st_loaded, g.st_gated,
                        g.st_overflow, g.st_composed, g.st_composed_late, g.per_half,
                        g.st_overflow_large, g.st_overflow_max));
        g.st_frames = g.st_motion_draws = g.st_tracked = g.st_loaded = g.st_gated =
            g.st_overflow = g.st_composed = g.st_composed_late = g.st_overflow_large = 0;
        g.st_overflow_max = 0;
    }
}

} // namespace BB::ObjMotion
