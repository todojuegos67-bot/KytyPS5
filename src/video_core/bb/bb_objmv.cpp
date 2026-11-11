// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-FileCopyrightText: Copyright 2026 bbport contributors (deadinside28/bloodborne_pc:
//                         gpu/upscale/vk_object_motion.cpp, motion_history.h, docs/motion_vectors.md,
//                         GPL-2.0-or-later; the approach and the bone palette classification)
// SPDX-License-Identifier: GPL-2.0-or-later
// BB object motion vectors: see bb_objmv.h.

#include "video_core/bb/bb_objmv.h"
#include "video_core/bb/bb_native_mv.h"

#include <algorithm>
#include <atomic>
#include <chrono>
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
    // The motion images (render-target size, R32G32_UINT: half2 vector + class). 1.20: two, by
    // frame parity: the presenter's late compose (after the guest flip) reads the frame it
    // presents while the next frame's G-buffer pass already clears / writes the other one. With a
    // single image the late compose could read the next frame's character vectors (or a cleared
    // image) over this frame's depth: the character's vectors one frame off its pixels.
    std::array<vk::Image, 2> images{};
    std::array<VmaAllocation, 2> image_allocs{};
    std::array<vk::ImageView, 2> views{};
    // 1.20 BB_MV_VIEW: the composed motion as colour (render size, RGBA8).
    vk::Image viz{};
    VmaAllocation viz_alloc{};
    vk::ImageView viz_view{};
    bool viz_valid = false; // composed this frame (late or not)
    // 1.20 BB_OBJMV_DEBUG: per-frame statistics of the compose pass (host visible ring).
    vk::Buffer stats{};
    VmaAllocation stats_alloc{};
    u32* stats_mapped = nullptr;
    u64 stats_address = 0;
    u64 compose_no = 0;
    std::array<u64, 16> dbg{};
    u64 dbg_frames = 0;
    u64 dbg_exact = 0, dbg_loose = 0, dbg_new = 0, dbg_count_changed = 0, dbg_large = 0;
    std::chrono::steady_clock::time_point dbg_since = std::chrono::steady_clock::now();
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
    // 1.20: the same draw by a looser identity (shader, index count, vertex count, instances,
    // palette size and order; no buffer addresses): a character whose vertex / index streams or
    // second stream live in a per-frame ring changes address every frame and never found its
    // previous positions by the exact identity: camera-only motion, cross-faded by FG.
    std::unordered_map<u64, Entry> prev_loose, cur_loose;
    std::unordered_map<u64, u32> loose_ordinals;
    std::unordered_map<u64, u64> palettes; // small skeletons: last palette hash
    // 0.71 rigid objects (no bone palette): last hash of the draw's own vertex constants (its
    // object transform); a draw whose constants changed since the previous frame moved by itself.
    std::unordered_map<u64, u64> rigid;
    u32 rigid_used = 0;                     // positions stored for rigid draws this frame
    u32 rigid_seen = 0, rigid_changed = 0;  // this frame
    bool rigid_camera_bound = false;        // last frame: most rigid draws changed (camera in them)
    // Stats (since the last report).
    u64 st_motion_draws = 0, st_tracked = 0, st_loaded = 0, st_overflow = 0, st_gated = 0,
        st_frames = 0, st_composed = 0, st_composed_late = 0, st_overflow_large = 0;
    u32 st_overflow_max = 0; // largest dropped draw (positions)
    u64 st_compacted = 0;    // 1.03: indexed draws stored for their index range only
    // 1.03: per second, what reached the frame-generation input (NoteFgInput).
    u64 fg_frames = 0, fg_with_objects = 0, fg_object_draws = 0;
    u64 composed_frame = ~0ull; // 1.41: guest frame of the last compose's object vectors
    u32 composed_draws = 0;  // draws with last frame's positions in the frame just composed
    u32 loaded_frame = 0, loaded_last = 0; // the same, this frame / the frame the flip ended
    std::unordered_map<u64, std::pair<u32, u32>> index_ranges; // cached [min, max] by index block
    u64 st_rigid_tracked = 0, st_rigid_static = 0, st_rigid_budget = 0, st_rigid_camera = 0;
    // 1.29: identity of a skinned draw by its bone palette's guest address (one per character /
    // enemy instance when the game keeps it there), tried before the shader / stream identity:
    // several enemies of one model share shader, vertex and index streams and were told apart
    // only by draw order, which changes with sorting (an enemy given another one's positions).
    std::unordered_map<u64, Entry> prev_pal, cur_pal;
    std::unordered_map<u64, u32> pal_ordinals;
    // 1.29 [fg-skinmv] per second (draws).
    u64 sk_pal = 0, sk_exact = 0, sk_loose = 0, sk_none = 0, sk_overflow = 0, sk_frames = 0;
    u64 ot_prev = 0, ot_none = 0;
    std::chrono::steady_clock::time_point sk_since = std::chrono::steady_clock::now();
};
State g;

bool RigidOn() {
    static const bool on = [] {
        const char* s = std::getenv("BB_OBJMV_RIGID");
        return s == nullptr || s[0] != '0';
    }();
    return on;
}

bool DebugOn() {
    static const bool on = [] {
        const char* s = std::getenv("BB_OBJMV_DEBUG");
        return s != nullptr && s[0] != '0';
    }();
    return on;
}

bool ViewOn() {
    static const bool on = [] {
        const char* s = std::getenv("BB_MV_VIEW");
        return s != nullptr && s[0] != '0';
    }();
    return on;
}

bool PaletteIdOn() {
    static const bool on = [] {
        const char* s = std::getenv("BB_OBJMV_PALETTE_ID");
        return s == nullptr || s[0] != '0';
    }();
    return on;
}

bool LooseOn() {
    static const bool on = [] {
        const char* s = std::getenv("BB_OBJMV_LOOSE");
        return s == nullptr || s[0] != '0';
    }();
    return on;
}

// 1.40 (bb-fgamd-140): object-vector pixel coverage of the composed motion image, always counted
// (one global add per 8x8 group; BB_OBJMV_COVERAGE=0 off), read by the [fg-amd] line.
bool CoverageOn() {
    static const bool on = [] {
        const char* s = std::getenv("BB_OBJMV_COVERAGE");
        return s == nullptr || s[0] != '0';
    }();
    return on;
}

// 1.40: an upscaler history reset (path switch, first frame of the FG-only path, camera cut) no
// longer drops the object vectors of that frame: the per-vertex previous positions are the
// draw's own and stay valid (a draw without them keeps the camera motion anyway). Up to 1.39
// every reset frame gave frame generation camera-only motion: characters and enemies were
// interpolated as if they stood still on those frames. BB_OBJMV_RESET_SKIP=1: the old rule.
bool ResetSkip() {
    static const bool on = [] {
        const char* s = std::getenv("BB_OBJMV_RESET_SKIP");
        return s != nullptr && s[0] == '1';
    }();
    return on;
}

struct Coverage {
    std::atomic<u64> obj{0}, chr{0}, all{0}, chr_still{0}, chr_own{0};
    std::atomic<u32> composes{0}, read{0}, reset_frames{0}, reset_skipped{0}, not_written{0};
} g_cov;

constexpr u32 kStatSlots = 4; // frames in flight: a slot is read back 3 composes later
constexpr u32 kStatWords = 16;

bool g_native_only = false;

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
    const char* nmv = std::getenv("BB_NATIVE_MV");
    const bool native_env = nmv == nullptr || *nmv == '\0' || (nmv[0] != '0' && std::string(nmv) != "off");
    if (EnvOff() && native_env) {
        // bb-native-mv: native-only mode. v1's motion pipeline runs, but only draws the game
        // names (skin ring / DrawMesh) and that move get a vector; nothing is reconstructed.
        g_native_only = true;
        Log("native-only (BB_OBJMV=0, BB_NATIVE_MV=1): object vectors only for moving game "
            "objects (characters by skin ring), everything else keeps the 149 camera vector");
    } else if (EnvOff()) {
        g.off = true;
        Log("off (BB_OBJMV=0): camera-only motion vectors");
        return false;
    }
    g.device = instance.GetDevice();
    g.allocator = instance.GetAllocator();
    // 4 Mi positions a half (128 MiB of device memory with both halves). Build 77 at 2 Mi dropped
    // 59256 of the 99109 draws that needed positions in 1800 frames (60 %) over the budget.
    u32 mi = 8;  // 8 Mi: the character's large skinned draws overflowed 4 Mi and lost their vectors (ghosting)
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
    void* stats_mapped = nullptr;
    if (!CreateBuffer(vk::DeviceSize(kStatSlots) * kStatWords * 4,
                      vk::BufferUsageFlagBits::eStorageBuffer |
                          vk::BufferUsageFlagBits::eTransferDst,
                      true, g.stats, g.stats_alloc, &stats_mapped, g.stats_address)) {
        Log("buffers not created: off");
        return false;
    }
    g.params_mapped = static_cast<u32*>(mapped);
    g.stats_mapped = static_cast<u32*>(stats_mapped);
    std::memset(g.stats_mapped, 0, size_t(kStatSlots) * kStatWords * 4);
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
        vk::DescriptorSetLayoutBinding{.binding = 2,
                                       .descriptorType = vk::DescriptorType::eStorageBuffer,
                                       .descriptorCount = 1,
                                       .stageFlags = vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{.binding = 3,
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
                    "({} Mi positions a frame, {} MiB); 1.20: motion image per frame parity, "
                    "loose identity {}, BB_OBJMV_DEBUG {}, BB_MV_VIEW {}",
                    mi, (u64(g.per_half) * 2 * 16) >> 20,
                    LooseOn() ? "on" : "off (BB_OBJMV_LOOSE=0)", DebugOn() ? "on" : "off",
                    ViewOn() ? "on" : "off"));
    return true;
}

void DestroyImage() {
    for (u32 i = 0; i < 2; ++i) {
        if (g.views[i]) g.device.destroyImageView(g.views[i]);
        if (g.images[i]) {
            vmaDestroyImage(g.allocator, static_cast<VkImage>(g.images[i]), g.image_allocs[i]);
        }
        g.views[i] = vk::ImageView{};
        g.images[i] = vk::Image{};
        g.image_allocs[i] = {};
    }
    if (g.viz_view) g.device.destroyImageView(g.viz_view);
    if (g.viz) vmaDestroyImage(g.allocator, static_cast<VkImage>(g.viz), g.viz_alloc);
    g.viz_view = vk::ImageView{};
    g.viz = vk::Image{};
    g.viz_alloc = {};
    g.viz_valid = false;
    g.width = g.height = 0;
}

bool CreateImage(u32 width, u32 height, vk::Format format, vk::ImageUsageFlags usage,
                 vk::Image& image, VmaAllocation& alloc, vk::ImageView& view) {
    const VkImageCreateInfo ci = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
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
        .format = format,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
    if (result != vk::Result::eSuccess) return false;
    view = v;
    return true;
}

bool EnsureImage(u32 width, u32 height) {
    if (g.images[0] && g.width == width && g.height == height) return true;
    if (g.images[0]) {
        (void)g.device.waitIdle();
        DestroyImage();
    }
    for (u32 i = 0; i < 2; ++i) {
        if (!CreateImage(width, height, vk::Format::eR32G32Uint,
                         vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage,
                         g.images[i], g.image_allocs[i], g.views[i])) {
            DestroyImage();
            return false;
        }
    }
    if (!CreateImage(width, height, vk::Format::eR8G8B8A8Unorm,
                     vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
                     g.viz, g.viz_alloc, g.viz_view)) {
        DestroyImage();
        return false;
    }
    g.width = width;
    g.height = height;
    g.cleared = false; // new images: undefined layout, cleared by the next motion pass
    Log(fmt::format("motion images 2x {}x{}", width, height));
    return true;
}

void ImageBarrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout old_layout,
                  vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
                  vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
                  bool positions = false) {
    // 1.20 `positions`: also make last frame's stored clip positions (vertex shader atomics,
    // device memory) visible to this frame's vertex shader loads. Without it nothing ordered the
    // two frames' accesses to the position buffer beyond submission order.
    const vk::MemoryBarrier2 memory{
        .srcStageMask = vk::PipelineStageFlagBits2::eVertexShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite |
                         vk::AccessFlagBits2::eShaderStorageRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eVertexShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                         vk::AccessFlagBits2::eShaderStorageWrite,
    };
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
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = positions ? 1u : 0u,
                                            .pMemoryBarriers = &memory,
                                            .imageMemoryBarrierCount = 1,
                                            .pImageMemoryBarriers = &barrier});
}

u64 Hash(const void* data, size_t size, u64 seed = 0) {
    return XXH3_64bits_withSeed(data, size, seed);
}

// The draw's bone palette among its vertex shader's constant buffers (deadinside28's
// Motion::ClassifyBuffer): 864 bytes is the scene block; 640..16384 a character skeleton; 96..639
// in 48-byte (3x4) matrices a small skeleton (weapons, props: most of them static).
// 0.71: no palette but per-draw constants (a rigid object's transform, 16..1024 bytes, not the
// scene block): Rigid, with their hash in rigid_hash.
enum class Palette { None, Small, Large, Rigid };
Palette Classify(const Shader::Info& vs, VAddr& small_address, u32& small_size, u64& rigid_hash,
                 VAddr& large_address) {
    Palette found = Palette::None;
    u64 h = 0;
    bool any = false;
    for (const auto& desc : vs.buffers) {
        if (desc.IsSpecial() || desc.is_written) continue;
        const auto sharp = desc.GetSharp(vs);
        const u32 size = sharp.GetSize();
        if (size != 864 && size >= 16 && size <= 1024 && sharp.base_address != 0) {
            h = XXH3_64bits_withSeed(reinterpret_cast<const void*>(sharp.base_address), size, h);
            any = true;
        }
        if (size == 864 || size < 96 || size > 16384) continue;
        if (size >= 640 && size % 16 == 0) {
            large_address = sharp.base_address;
            return Palette::Large;
        }
        if (size < 640 && size % 48 == 0 && found == Palette::None) {
            found = Palette::Small;
            small_address = sharp.base_address;
            small_size = size;
        }
    }
    if (found == Palette::None && any && RigidOn()) {
        rigid_hash = h;
        return Palette::Rigid;
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
    const u32 cur = static_cast<u32>(g.frame & 1);
    attachment.image_view = g.views[cur];
    attachment.image_layout = vk::ImageLayout::eGeneral;
    attachment.clear_value = {0u, 0u, 0u, 0u};
    attachment.is_clear = 0;
    if (!g.cleared) {
        // First motion pass of the frame: last frame's compose read it (compute).
        scheduler.EndRendering();
        ImageBarrier(scheduler.CommandBuffer(), g.images[cur], vk::ImageLayout::eUndefined,
                     vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
                     vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                     vk::AccessFlagBits2::eColorAttachmentWrite |
                         vk::AccessFlagBits2::eColorAttachmentRead,
                     true);
        attachment.is_clear = 1;
        g.cleared = true;
        g.written = true;
    }
}

namespace {
struct Range {
    bool valid = false;
    u32 min = 0, max = 0;
};
// The [min, max] vertex index of an indexed draw, from the guest index buffer (cached per block:
// address, count, type and a sample of its contents, so a rewritten block is scanned again).
Range IndexRange(const AmdGpu::Regs& regs, u32 index_offset) {
    static const bool on = [] {
        const char* s = std::getenv("BB_OBJMV_INDEX_RANGE");
        return s == nullptr || s[0] != '0';
    }();
    if (!on) return {};
    const bool i16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const u32 size = i16 ? 2u : 4u;
    const VAddr base = regs.index_base_address.Address<VAddr>();
    const u32 count = std::min(regs.num_indices, regs.max_index_size);
    if (base == 0 || count == 0 || count > (16u << 20)) return {};
    const VAddr addr = base + u64(index_offset) * size;
    const u8* p = reinterpret_cast<const u8*>(addr);
    const u64 bytes = u64(count) * size;
    const u64 head = std::min<u64>(bytes, 256);
    u64 key = Hash(p, head, addr ^ (u64(count) << 1) ^ size);
    key = Hash(p + (bytes - head), head, key);
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
    g.index_ranges[key] = {lo, hi};
    return {true, lo, hi};
}
} // namespace

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
    u64 rigid_hash = 0;
    VAddr large_address = 0;
    const Palette palette = Classify(vs, palette_address, palette_size, rigid_hash, large_address);
    if (palette == Palette::None || fetch.Empty() || fetch.attributes.empty()) return;
    const auto vb = fetch.attributes[0].GetSharp(vs);
    if (vb.GetStride() == 0) return;
    u32 vertices = vb.num_records;
    // 1.03: an indexed draw only reaches the vertices its indices name. Skinned meshes share one
    // large vertex buffer between many draws, and storing num_records positions per draw ran the
    // per-frame budget out (the "over the budget ... large skeletons" lines): the draws that came
    // last, often the player character, kept camera-only motion and judged in generated frames.
    // Store [min index, max index] only (the slot base moves with it: entry[4]).
    u32 range_first = 0;
    if (indexed) {
        if (const auto r = IndexRange(regs, index_offset); r.valid) {
            if (r.max < vertices) {
                range_first = r.min;
                vertices = r.max - r.min + 1;
                ++g.st_compacted;
            }
        }
    }
    const u32 instances = regs.num_instances.NumInstances();
    if (vertices == 0 || vertices > kMaxVerticesPerDraw || instances > kMaxInstances ||
        u64(vertices) * instances > kMaxVerticesPerDraw) {
        return;
    }
    const u32 count = vertices * instances;

    // bb-native-mv: a draw the game's DrawMesh issued (bb_native_mv.h) is keyed by the game's own
    // object (bone matrices + mesh) and gets an object vector only while that object moves in the
    // world; static ones keep the camera vector. The reconstructed identity below is not used.
    BB::NativeMv::DrawId nid = BB::NativeMv::Current();
    if (nid.key == 0 && BB::NativeMv::On()) {
        // Characters: the bone palette's GPU address is one of a game skin ring's slots.
        nid = BB::NativeMv::LookupPalette(palette == Palette::Large ? large_address
                                                                     : palette_address);
    }
    if (nid.key != 0) {
        BB::NativeMv::NoteSceneDraw(nid);
        if (!nid.moving && !nid.recent) return;
        const u32 half = static_cast<u32>(g.frame & 1);
        if (g.cursor + count > g.per_half || g.params_used + 1 >= kParamsPerFrame) {
            ++g.st_overflow;
            return;
        }
        std::array<u64, 4> nidk{nid.key, vs_hash, regs.num_indices, count};
        const u64 nbase = Hash(nidk.data(), sizeof(nidk), 0x6e6174697665ull);
        const u32 nord = g.ordinals[nbase]++;
        const u64 nkey = Hash(&nord, sizeof(nord), nbase);
        const u32 store_base = half * g.per_half + g.cursor;
        g.cursor += count;
        u32 flags = 1;
        if (nid.skinned) flags |= 4;
        u32 load_base = 0;
        bool found = false;
        if (nid.moving) {
            if (const auto it = g.prev.find(nkey);
                it != g.prev.end() && it->second.count == count && it->second.base != store_base) {
                load_base = it->second.base;
                found = true;
                flags |= 2;
                ++g.st_loaded;
                ++g.loaded_frame;
            }
            BB::NativeMv::NoteVectorDraw(found);
        }
        g.cur[nkey] = {store_base, count};
        ++g.st_tracked;
        const u32 slot = static_cast<u32>(g.frame % kFrameSlots);
        const u32 index = slot * kParamsPerFrame + (++g.params_used);
        u32* entry = g.params_mapped + size_t(index) * 8;
        entry[0] = store_base;
        entry[1] = load_base;
        entry[2] = vertices;
        entry[3] = flags;
        entry[4] = vertex_offset + range_first;
        entry[5] = instance_offset;
        entry[6] = instances;
        entry[7] = 0;
        push.motion_param = index;
        return;
    }
    if (BB::NativeMv::On()) BB::NativeMv::NoteSceneDraw(nid);
    if (g_native_only) return;

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
    // 1.20 loose identity (see State::prev_loose).
    std::array<u64, 5> lid{};
    lid[0] = vs_hash;
    lid[1] = regs.num_indices;
    lid[2] = (u64(vertices) << 32) | instances;
    lid[3] = (u64(static_cast<u32>(palette)) << 32) | palette_size;
    lid[4] = indexed ? 1 : 0;
    const u64 loose_base = Hash(lid.data(), sizeof(lid), 0x6f626a6d76ull);
    const u32 loose_ordinal = g.loose_ordinals[loose_base]++;
    const u64 loose_key = Hash(&loose_ordinal, sizeof(loose_ordinal), loose_base);

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
    if (palette == Palette::Rigid) {
        // A static object's constants do not change and the camera motion is exact for it. When
        // most of them change every frame the constants hold the camera (a per-draw
        // view-projection): the camera motion is exact for those too, and storing every draw would
        // take the budget from the skinned ones, so rigid tracking waits for a frame without that.
        ++g.rigid_seen;
        auto [it, inserted] = g.rigid.try_emplace(key, rigid_hash);
        const bool changed = !inserted && it->second != rigid_hash;
        it->second = rigid_hash;
        if (!changed) {
            ++g.st_rigid_static;
            return;
        }
        ++g.rigid_changed;
        if (g.rigid_camera_bound) {
            ++g.st_rigid_camera;
            return;
        }
        if (count > 65536 || g.rigid_used + count > g.per_half / 4) {
            ++g.st_rigid_budget;
            return;
        }
        g.rigid_used += count;
        ++g.st_rigid_tracked;
    }

    const u32 half = static_cast<u32>(g.frame & 1);
    if (g.cursor + count > g.per_half || g.params_used + 1 >= kParamsPerFrame) {
        // Dropped draws keep camera motion. Split by palette class, with the largest one, so the
        // next log shows whether character skeletons are among them.
        ++g.st_overflow;
        if (palette == Palette::Large) {
            ++g.st_overflow_large;
            ++g.sk_overflow;
        }
        g.st_overflow_max = std::max(g.st_overflow_max, count);
        return;
    }
    const u32 store_base = half * g.per_half + g.cursor;
    g.cursor += count;
    u32 flags = 1; // store
    u32 load_base = 0;
    const bool large = palette == Palette::Large;
    if (large) {
        flags |= 4; // class for the compose pass's statistics / view
        ++g.dbg_large;
    }
    bool found = false;
    u64 pal_key = 0;
    if (large && large_address != 0 && PaletteIdOn()) {
        std::array<u64, 4> pid{vs_hash, large_address, regs.num_indices, count};
        const u64 pal_base = Hash(pid.data(), sizeof(pid), 0x736b696e6d76ull);
        const u32 pal_ordinal = g.pal_ordinals[pal_base]++;
        pal_key = Hash(&pal_ordinal, sizeof(pal_ordinal), pal_base);
        if (const auto it = g.prev_pal.find(pal_key);
            it != g.prev_pal.end() && it->second.count == count && it->second.base != store_base) {
            load_base = it->second.base;
            found = true;
            ++g.sk_pal;
        }
    }
    if (found) {
        // palette identity
    } else if (const auto it = g.prev.find(key); it != g.prev.end()) {
        if (it->second.count == count) {
            load_base = it->second.base;
            found = true;
            if (large) {
                ++g.dbg_exact;
                ++g.sk_exact;
            }
        } else if (large) {
            ++g.dbg_count_changed;
        }
    }
    if (!found && LooseOn()) {
        if (const auto it = g.prev_loose.find(loose_key);
            it != g.prev_loose.end() && it->second.count == count) {
            load_base = it->second.base;
            found = true;
            if (large) {
                ++g.dbg_loose;
                ++g.sk_loose;
            }
        }
    }
    if (found) {
        flags |= 2;
        ++g.st_loaded;
        ++g.loaded_frame;
    } else if (large) {
        ++g.dbg_new;
        ++g.sk_none;
    }
    if (!large) {
        if (found) ++g.ot_prev; else ++g.ot_none;
    }
    if (pal_key != 0) g.cur_pal[pal_key] = {store_base, count};
    g.cur[key] = {store_base, count};
    g.cur_loose[loose_key] = {store_base, count};
    ++g.st_tracked;

    const u32 slot = static_cast<u32>(g.frame % kFrameSlots);
    const u32 index = slot * kParamsPerFrame + (++g.params_used);
    u32* entry = g.params_mapped + size_t(index) * 8;
    entry[0] = store_base;
    entry[1] = load_base;
    entry[2] = vertices;
    entry[3] = flags;
    entry[4] = vertex_offset + range_first;
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
    g.composed_draws = 0;
    g.viz_valid = false;
    if (g.ready) {
        if (reset) g_cov.reset_frames.fetch_add(1, std::memory_order_relaxed);
        if (!wrote) g_cov.not_written.fetch_add(1, std::memory_order_relaxed);
        if (reset && wrote && ResetSkip()) g_cov.reset_skipped.fetch_add(1, std::memory_order_relaxed);
    }
    if (!g.ready || !wrote || (reset && ResetSkip()) || g.width != render.width ||
        g.height != render.height) {
        return;
    }
    // 1.20: the motion image of the frame being composed (late: the one the flip just ended,
    // OnFlip has already advanced g.frame).
    const u32 index = static_cast<u32>((late ? g.frame - 1 : g.frame) & 1);
    g.composed_frame = late ? g.frame - 1 : g.frame;
    const bool debug = DebugOn() && g.stats_mapped != nullptr;
    const bool coverage = CoverageOn() && g.stats_mapped != nullptr;
    const bool stats = debug || coverage;
    const bool view = ViewOn();
    const u32 stat_slot = static_cast<u32>(g.compose_no % kStatSlots);
    if (stats) {
        // The slot's last use was kStatSlots composes ago (finished on the GPU by now).
        u32* words = g.stats_mapped + size_t(stat_slot) * kStatWords;
        if (coverage && g.compose_no >= kStatSlots && words[13] != 0) {
            g_cov.obj.fetch_add(words[11], std::memory_order_relaxed);
            g_cov.chr.fetch_add(words[12], std::memory_order_relaxed);
            g_cov.all.fetch_add(words[13], std::memory_order_relaxed);
            g_cov.chr_still.fetch_add(words[14], std::memory_order_relaxed);
            g_cov.chr_own.fetch_add(words[15], std::memory_order_relaxed);
            g_cov.read.fetch_add(1, std::memory_order_relaxed);
        }
        if (debug && g.compose_no >= kStatSlots) {
            for (u32 k = 0; k < kStatWords; ++k) {
                if (k == 4) {
                    g.dbg[4] = std::max<u64>(g.dbg[4], words[k]);
                } else {
                    g.dbg[k] += words[k];
                }
            }
            ++g.dbg_frames;
        }
        cmd.fillBuffer(g.stats, vk::DeviceSize(stat_slot) * kStatWords * 4, kStatWords * 4, 0u);
        const vk::MemoryBarrier2 fill{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                             vk::AccessFlagBits2::eShaderStorageWrite,
        };
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &fill});
    }
    ++g.compose_no;
    ImageBarrier(cmd, g.images[index], vk::ImageLayout::eGeneral,
                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                 vk::AccessFlagBits2::eColorAttachmentWrite,
                 vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    ImageBarrier(cmd, mv_image, vk::ImageLayout::eGeneral,
                 vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                 vk::PipelineStageFlagBits2::eComputeShader,
                 vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    ImageBarrier(cmd, g.viz, vk::ImageLayout::eUndefined, vk::PipelineStageFlagBits2::eAllCommands,
                 vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eComputeShader,
                 vk::AccessFlagBits2::eShaderWrite);
    const vk::DescriptorImageInfo obj_info{{}, g.views[index], vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo mv_info{{}, mv_view, vk::ImageLayout::eGeneral};
    const vk::DescriptorBufferInfo stats_info{g.stats, 0, vk::DeviceSize(kStatSlots) * kStatWords * 4};
    const vk::DescriptorBufferInfo stats_slot_info{g.stats, vk::DeviceSize(stat_slot) * kStatWords * 4,
                                                   kStatWords * 4};
    const vk::DescriptorImageInfo viz_info{{}, g.viz_view, vk::ImageLayout::eGeneral};
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
                               .descriptorType = vk::DescriptorType::eStorageBuffer,
                               .pBufferInfo = stats ? &stats_slot_info : &stats_info},
        vk::WriteDescriptorSet{.dstBinding = 3,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageImage,
                               .pImageInfo = &viz_info},
    };
    const std::array<u32, 4> push{render.width, render.height, 1u,
                                  (debug ? 1u : 0u) | (view ? 2u : 0u) | (coverage ? 4u : 0u)};
    g_cov.composes.fetch_add(1, std::memory_order_relaxed);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g.layout, 0, writes);
    cmd.pushConstants(g.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push.data());
    cmd.dispatch((render.width + 7) / 8, (render.height + 7) / 8, 1);
    g.viz_valid = view;
    ++g.st_composed;
    if (late) ++g.st_composed_late;
    g.composed_draws = late ? g.loaded_last : g.loaded_frame;
}

void DebugView(vk::CommandBuffer cmd, vk::Image output, vk::ImageLayout output_layout,
               vk::Extent2D output_size) {
    if (!g.ready || !ViewOn() || !g.viz_valid || !g.viz || !output) return;
    g.viz_valid = false;
    ImageBarrier(cmd, g.viz, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                 vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eBlit,
                 vk::AccessFlagBits2::eTransferRead);
    const auto to = [&](vk::ImageLayout from, vk::ImageLayout layout, vk::PipelineStageFlags2 src,
                        vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst,
                        vk::AccessFlags2 dst_access) {
        const vk::ImageMemoryBarrier2 b{
            .srcStageMask = src,
            .srcAccessMask = src_access,
            .dstStageMask = dst,
            .dstAccessMask = dst_access,
            .oldLayout = from,
            .newLayout = layout,
            .image = output,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                .pImageMemoryBarriers = &b});
    };
    to(output_layout, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eAllCommands,
       vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
       vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite);
    const vk::ImageBlit region{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                 vk::Offset3D{s32(g.width), s32(g.height), 1}},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstOffsets = std::array{vk::Offset3D{0, 0, 0},
                                 vk::Offset3D{s32(output_size.width), s32(output_size.height), 1}},
    };
    // The viz image is in General (blit sources may be General).
    cmd.blitImage(g.viz, vk::ImageLayout::eGeneral, output, vk::ImageLayout::eTransferDstOptimal,
                  region, vk::Filter::eNearest);
    to(vk::ImageLayout::eTransferDstOptimal, output_layout, vk::PipelineStageFlagBits2::eBlit,
       vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eAllCommands,
       vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
           vk::AccessFlagBits2::eTransferRead);
}

void OnFlip() {
    if (!g.ready) return;
    ++g.frame;
    ++g.st_frames;
    g.prev = std::move(g.cur);
    g.cur.clear();
    g.ordinals.clear();
    g.prev_loose = std::move(g.cur_loose);
    g.cur_loose.clear();
    g.loose_ordinals.clear();
    g.prev_pal = std::move(g.cur_pal);
    g.cur_pal.clear();
    g.pal_ordinals.clear();
    ++g.sk_frames;
    {
        const auto now = std::chrono::steady_clock::now();
        if (now - g.sk_since >= std::chrono::seconds(1)) {
            const u64 with = g.sk_pal + g.sk_exact + g.sk_loose;
            const u64 all = with + g.sk_none;
            Log(fmt::format(
                "[fg-skinmv] 1.29 {} frames: skinned (bone palette) draws {} with true per-vertex "
                "previous positions ({:.1f} %: {} by palette identity, {} exact, {} loose), {} "
                "without (camera only), {} over the budget | rigid / small-skeleton draws {} with "
                "previous positions, {} without | palette identity {}",
                g.sk_frames, with, all ? 100.0 * double(with) / double(all) : 0.0, g.sk_pal,
                g.sk_exact, g.sk_loose, g.sk_none, g.sk_overflow, g.ot_prev, g.ot_none,
                PaletteIdOn() ? "on" : "off (BB_OBJMV_PALETTE_ID=0)"));
            g.sk_pal = g.sk_exact = g.sk_loose = g.sk_none = g.sk_overflow = g.sk_frames = 0;
            g.ot_prev = g.ot_none = 0;
            g.sk_since = now;
        }
    }
    if (DebugOn()) {
        const auto now = std::chrono::steady_clock::now();
        if (now - g.dbg_since >= std::chrono::seconds(1)) {
            const auto avg = [](u64 sum16, u64 n) { return n ? double(sum16) / 16.0 / double(n) : 0.0; };
            const u64 frames = std::max<u64>(g.dbg_frames, 1);
            Log(fmt::format(
                "[objmv-debug] {:.1f} s, {} composes read back: CHARACTER (large skeleton) {:.0f} px "
                "a frame, |object mv| avg {:.2f} px (max {:.1f}), |camera mv| there avg {:.2f} px, "
                "{:.1f} % differ from camera by > 1 px, {:.1f} % near zero (< 0.25 px) | other "
                "objects {:.0f} px a frame, |object| {:.2f} vs |camera| {:.2f} | camera-only {:.0f} "
                "px a frame, |camera| {:.2f} | large-skeleton draws: {} found by exact identity, "
                "{} by loose identity, {} without last frame (camera only), {} vertex count "
                "changed",
                std::chrono::duration<double>(now - g.dbg_since).count(), g.dbg_frames,
                double(g.dbg[0]) / frames, avg(g.dbg[1], g.dbg[0]), double(g.dbg[4]) / 16.0,
                avg(g.dbg[2], g.dbg[0]), g.dbg[0] ? 100.0 * g.dbg[3] / g.dbg[0] : 0.0,
                g.dbg[0] ? 100.0 * g.dbg[5] / g.dbg[0] : 0.0, double(g.dbg[6]) / frames,
                avg(g.dbg[7], g.dbg[6]), avg(g.dbg[8], g.dbg[6]), double(g.dbg[9]) / frames,
                avg(g.dbg[10], g.dbg[9]), g.dbg_exact, g.dbg_loose, g.dbg_new,
                g.dbg_count_changed));
            g.dbg.fill(0);
            g.dbg_frames = g.dbg_exact = g.dbg_loose = g.dbg_new = g.dbg_count_changed =
                g.dbg_large = 0;
            g.dbg_since = now;
        }
    }
    g.cursor = 0;
    g.params_used = 0;
    g.cleared = false;
    g.latched = g.written;
    g.written = false;
    g.loaded_last = g.loaded_frame;
    g.loaded_frame = 0;
    if (g.index_ranges.size() > 65536) g.index_ranges.clear();
    if (g.palettes.size() > 65536) g.palettes.clear();
    if (g.rigid.size() > 65536) g.rigid.clear();
    g.rigid_camera_bound = g.rigid_seen >= 64 && g.rigid_changed * 2 > g.rigid_seen;
    g.rigid_seen = g.rigid_changed = g.rigid_used = 0;
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
        Log(fmt::format("1.03: {} indexed draws stored for their index range only (not the whole "
                        "vertex buffer)",
                        g.st_compacted));
        g.st_compacted = 0;
        Log(fmt::format("rigid objects (BB_OBJMV_RIGID): last {} frames {} draws moved by their "
                        "own constants and got object vectors, {} unchanged (camera motion), {} "
                        "skipped (camera in the constants), {} over the rigid budget",
                        g.st_frames, g.st_rigid_tracked, g.st_rigid_static, g.st_rigid_camera,
                        g.st_rigid_budget));
        g.st_rigid_tracked = g.st_rigid_static = g.st_rigid_budget = g.st_rigid_camera = 0;
        g.st_frames = g.st_motion_draws = g.st_tracked = g.st_loaded = g.st_gated =
            g.st_overflow = g.st_composed = g.st_composed_late = g.st_overflow_large = 0;
        g.st_overflow_max = 0;
    }
}

u64 FrameNo() {
    return g.frame;
}

u64 ComposedFrame() {
    return g.composed_frame;
}

u32 ComposedDraws() {
    return g.ready ? g.composed_draws : 0;
}

CoverageStats TakeCoverage() {
    CoverageStats c;
    c.object_px = g_cov.obj.exchange(0, std::memory_order_relaxed);
    c.character_px = g_cov.chr.exchange(0, std::memory_order_relaxed);
    c.all_px = g_cov.all.exchange(0, std::memory_order_relaxed);
    c.composes = g_cov.composes.exchange(0, std::memory_order_relaxed);
    c.frames_read = g_cov.read.exchange(0, std::memory_order_relaxed);
    c.reset_frames = g_cov.reset_frames.exchange(0, std::memory_order_relaxed);
    c.reset_skipped = g_cov.reset_skipped.exchange(0, std::memory_order_relaxed);
    c.not_written = g_cov.not_written.exchange(0, std::memory_order_relaxed);
    c.character_still_px = g_cov.chr_still.exchange(0, std::memory_order_relaxed);
    c.character_own_px = g_cov.chr_own.exchange(0, std::memory_order_relaxed);
    c.enabled = g.ready && CoverageOn();
    c.reset_skip_rule = ResetSkip();
    return c;
}

} // namespace BB::ObjMotion
