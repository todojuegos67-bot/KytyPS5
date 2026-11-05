// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-imgfg-078: see bb_imgfg.h. An original design (block optical flow on a luma pyramid,
// consistency-gated warps, HUD protection); nothing of any closed product was read or reused.

#include "video_core/bb/bb_imgfg.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "video_core/bb/bb_display.h"
#include "video_core/bb/bb_kyty_compat.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/host_shaders/bb_imgfg_capture_comp.h"
#include "video_core/host_shaders/bb_imgfg_flow_comp.h"
#include "video_core/host_shaders/bb_imgfg_flowup_comp.h"
#include "video_core/host_shaders/bb_imgfg_hud_comp.h"
#include "video_core/host_shaders/bb_imgfg_interp_comp.h"
#include "video_core/host_shaders/bb_imgfg_me_comp.h"
#include "video_core/host_shaders/bb_imgfg_pyr_comp.h"

namespace BB::ImgFg {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kLevels = 4;
constexpr uint32_t kSlots = 8;          // frames in flight (command buffers, semaphores, stats)
constexpr uint32_t kStatsStride = 256;  // bytes between the slots' counters
constexpr uint32_t kStamps = 4;         // timestamps per slot

// Tuned on the captured runs (tools: the offline test in the 0.78 report).
constexpr float kEps = 0.03f;       // local contrast normalisation: deviation floor
constexpr float kSigma = 0.06f;     // edge-aware upsampling: luma weight
constexpr float kSc0 = 0.08f;       // photometric trust: base scale
constexpr float kSck = 3.0f;        // ... and per local contrast
constexpr float kConfGain = 8.0f;
constexpr float kConsK = 0.3f;
constexpr float kHudStill = 0.006f; // static: luma difference below
constexpr float kHudRange = 0.12f;  // high contrast: 5x5 range above (half resolution luma)
constexpr float kHudlessDiff = 0.02f;

struct Img {
    vk::Image image{};
    vk::ImageView view{};
    vk::DeviceMemory memory{};
    vk::Extent2D extent{};
};

struct Set { // one real frame's pictures
    Img main;   // what the flow and the warps see (HUD-less image when given, else the backbuffer)
    Img final;  // the backbuffer (the HUD)
    std::array<Img, kLevels> lum;   // raw luma, level 0 is half resolution
    std::array<Img, kLevels> norm;  // locally normalised
};

struct Pipes {
    vk::Pipeline capture{}, pyr{}, me{}, flow{}, flowup{}, hud{}, interp{};
};

struct Stats {
    double real = 0, generated = 0;
    double flow_ms = 0, interp_ms = 0;
    uint64_t gpu_samples = 0;
    uint64_t incons = 0, hud = 0, total = 0;
    Clock::time_point start{};
};

struct NoteState {
    bool new_frame = false;
    bool has_motion = false;
    bool has_hudless = false;
    bool reset = false;
    vk::ImageView motion{};
    vk::ImageView hudless{};
    vk::Extent2D backbuffer{};
};

struct G {
    bool requested = false;
    bool parsed = false;
    int mult = 2;
    bool gamedata = true;
    // device
    vk::PhysicalDevice physical{};
    vk::Device device{};
    uint32_t family = 0;
    vk::Queue queue{};
    vk::PhysicalDeviceMemoryProperties memory{};
    float timestamp_ns = 1.0f;
    bool ready_device = false;
    // resources
    bool built = false;
    bool failed = false;
    vk::Extent2D extent{};
    vk::Format format = vk::Format::eUndefined;
    vk::CommandPool pool{};
    std::array<vk::CommandBuffer, kSlots> cmd{};
    std::array<vk::Fence, kSlots> fence{};
    std::array<bool, kSlots> in_flight{};
    std::array<std::array<vk::Semaphore, 2>, kSlots> acquired{};
    std::array<std::array<vk::Semaphore, 2>, kSlots> done{};
    std::array<vk::Semaphore, kSlots> real_wait{};
    std::array<bool, kSlots> slot_generated{};
    vk::QueryPool queries{};
    vk::Buffer stats_buffer{};
    vk::DeviceMemory stats_memory{};
    uint8_t* stats_map = nullptr;
    vk::DescriptorSetLayout set_layout{};
    vk::PipelineLayout layout{};
    Pipes pipes{};
    vk::Sampler linear{}, nearest{};
    std::array<Set, 2> sets{};
    int cur = 0;
    std::array<std::array<Img, kLevels>, 2> flow{};  // [direction][level] grids
    Img ff, bf, fp, bp, hdiff;
    std::array<Img, 2> hud{};
    int hud_cur = 0;
    std::array<Img, 2> mid{};
    uint32_t slot = 0;
    uint32_t w = 0, h = 0, hw = 0, hh = 0;
    std::array<uint32_t, kLevels> lw{}, lh{}, gw{}, gh{};
    bool have_prev = false;
    bool hud_valid = false;
    std::unordered_map<uint64_t, vk::ImageView> swap_views;
    vk::SwapchainKHR sc_handle{};
    std::vector<vk::Image> sc_images;
    // frame state
    NoteState note{};
    Clock::time_point last_entry{};
    double interval_s = 0.0;
    Clock::time_point grid{};
    bool grid_valid = false;
    bool hudless_off = false;
    int hudless_bad = 0;
    bool last_used_hudless = false;
    bool prev_hudless = false;
    double lag_s = 0.0;
    std::atomic<bool> generating{false};
    Stats stats{};
    int acquire_fail = 0;
};

G g;

bool Parse() {
    if (g.parsed) return g.requested;
    g.parsed = true;
    const char* v = std::getenv("BB_FG");
    std::string s = v != nullptr ? v : "";
    for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    for (const char* p : {"image-", "img-", "imgfg-"}) {
        if (s.rfind(p, 0) == 0) {
            std::string m = s.substr(std::strlen(p));
            if (!m.empty() && m[0] == 'x') m = m.substr(1);
            g.mult = (m == "3") ? 3 : 2;
            g.requested = true;
        }
    }
    const char* d = std::getenv("BB_IMGFG_GAMEDATA");
    g.gamedata = d == nullptr || std::strcmp(d, "0") != 0;
    return g.requested;
}

uint32_t Half(uint32_t v) {
    return (v + 1) / 2;
}

uint32_t FindMemory(uint32_t bits, vk::MemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g.memory.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) != 0 && (g.memory.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

void Destroy(Img& img) {
    if (img.view) g.device.destroyImageView(img.view);
    if (img.image) g.device.destroyImage(img.image);
    if (img.memory) g.device.freeMemory(img.memory);
    img = {};
}

bool Make(Img& img, vk::Format format, uint32_t w, uint32_t h, vk::ImageUsageFlags usage) {
    const vk::ImageCreateInfo ci{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {w, h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    if (g.device.createImage(&ci, nullptr, &img.image) != vk::Result::eSuccess) return false;
    vk::MemoryRequirements req{};
    g.device.getImageMemoryRequirements(img.image, &req);
    const uint32_t type = FindMemory(req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (type == UINT32_MAX) return false;
    const vk::MemoryAllocateInfo mi{.allocationSize = req.size, .memoryTypeIndex = type};
    if (g.device.allocateMemory(&mi, nullptr, &img.memory) != vk::Result::eSuccess) return false;
    if (g.device.bindImageMemory(img.image, img.memory, 0) != vk::Result::eSuccess) return false;
    const vk::ImageViewCreateInfo vi{
        .image = img.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    if (g.device.createImageView(&vi, nullptr, &img.view) != vk::Result::eSuccess) return false;
    img.extent = {w, h};
    return true;
}

template <class A>
bool MakePipeline(vk::Pipeline& out, const A& code) {
    const vk::ShaderModuleCreateInfo mi{.codeSize = sizeof(code), .pCode = std::data(code)};
    vk::ShaderModule module{};
    if (g.device.createShaderModule(&mi, nullptr, &module) != vk::Result::eSuccess) return false;
    vk::ComputePipelineCreateInfo pi{};
    pi.stage.stage = vk::ShaderStageFlagBits::eCompute;
    pi.stage.module = module;
    pi.stage.pName = "main";
    pi.layout = g.layout;
    const auto r = g.device.createComputePipelines(vk::PipelineCache{}, 1, &pi, nullptr, &out);
    g.device.destroyShaderModule(module, nullptr);
    return r == vk::Result::eSuccess;
}

void Teardown() {
    if (!g.device) return;
    (void)g.device.waitIdle();
    for (auto& [k, v] : g.swap_views) g.device.destroyImageView(v);
    g.swap_views.clear();
    for (auto& s : g.sets) {
        Destroy(s.main);
        Destroy(s.final);
        for (auto& i : s.lum) Destroy(i);
        for (auto& i : s.norm) Destroy(i);
    }
    for (auto& d : g.flow) {
        for (auto& i : d) Destroy(i);
    }
    Destroy(g.ff);
    Destroy(g.bf);
    Destroy(g.fp);
    Destroy(g.bp);
    Destroy(g.hdiff);
    for (auto& i : g.hud) Destroy(i);
    for (auto& i : g.mid) Destroy(i);
    for (uint32_t s = 0; s < kSlots; ++s) {
        if (g.fence[s]) g.device.destroyFence(g.fence[s]);
        if (g.real_wait[s]) g.device.destroySemaphore(g.real_wait[s]);
        for (int j = 0; j < 2; ++j) {
            if (g.acquired[s][j]) g.device.destroySemaphore(g.acquired[s][j]);
            if (g.done[s][j]) g.device.destroySemaphore(g.done[s][j]);
        }
        g.fence[s] = {};
        g.real_wait[s] = {};
        g.acquired[s] = {};
        g.done[s] = {};
    }
    if (g.pool) g.device.destroyCommandPool(g.pool);
    g.pool = {};
    if (g.queries) g.device.destroyQueryPool(g.queries);
    g.queries = {};
    if (g.stats_map) g.device.unmapMemory(g.stats_memory);
    g.stats_map = nullptr;
    if (g.stats_buffer) g.device.destroyBuffer(g.stats_buffer);
    if (g.stats_memory) g.device.freeMemory(g.stats_memory);
    g.stats_buffer = {};
    g.stats_memory = {};
    g.built = false;
    g.have_prev = false;
    g.hud_valid = false;
    g.in_flight = {};
}

void TransitionAll(vk::CommandBuffer cmd) {
    std::vector<vk::ImageMemoryBarrier> b;
    const auto add = [&](const Img& i) {
        if (!i.image) return;
        b.push_back(vk::ImageMemoryBarrier{
            .srcAccessMask = {},
            .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = i.image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        });
    };
    for (auto& s : g.sets) {
        add(s.main);
        add(s.final);
        for (auto& i : s.lum) add(i);
        for (auto& i : s.norm) add(i);
    }
    for (auto& d : g.flow) {
        for (auto& i : d) add(i);
    }
    add(g.ff);
    add(g.bf);
    add(g.fp);
    add(g.bp);
    add(g.hdiff);
    for (auto& i : g.hud) add(i);
    for (auto& i : g.mid) add(i);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader, {},
                        {}, {}, b);
}

bool Build(vk::Extent2D extent, vk::Format format) {
    Teardown();
    g.failed = true;
    g.extent = extent;
    g.format = format;
    g.w = extent.width;
    g.h = extent.height;
    g.hw = Half(g.w);
    g.hh = Half(g.h);
    g.lw[0] = g.hw;
    g.lh[0] = g.hh;
    for (int l = 1; l < kLevels; ++l) {
        g.lw[l] = Half(g.lw[l - 1]);
        g.lh[l] = Half(g.lh[l - 1]);
    }
    for (int l = 0; l < kLevels; ++l) {
        g.gw[l] = (g.lw[l] + 3) / 4;
        g.gh[l] = (g.lh[l] + 3) / 4;
    }
    const auto sg = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled;
    for (auto& s : g.sets) {
        if (!Make(s.main, vk::Format::eR8G8B8A8Unorm, g.w, g.h, sg)) return false;
        if (!Make(s.final, vk::Format::eR8G8B8A8Unorm, g.w, g.h, sg)) return false;
        for (int l = 0; l < kLevels; ++l) {
            if (!Make(s.lum[l], vk::Format::eR32Sfloat, g.lw[l], g.lh[l], sg)) return false;
            if (!Make(s.norm[l], vk::Format::eR32Sfloat, g.lw[l], g.lh[l], sg)) return false;
        }
    }
    for (auto& d : g.flow) {
        for (int l = 0; l < kLevels; ++l) {
            if (!Make(d[l], vk::Format::eR16G16B16A16Sfloat, g.gw[l], g.gh[l], sg)) return false;
        }
    }
    if (!Make(g.ff, vk::Format::eR16G16B16A16Sfloat, g.gw[0], g.gh[0], sg)) return false;
    if (!Make(g.bf, vk::Format::eR16G16B16A16Sfloat, g.gw[0], g.gh[0], sg)) return false;
    if (!Make(g.fp, vk::Format::eR16G16B16A16Sfloat, g.w, g.h, sg)) return false;
    if (!Make(g.bp, vk::Format::eR16G16B16A16Sfloat, g.w, g.h, sg)) return false;
    if (!Make(g.hdiff, vk::Format::eR32Sfloat, g.hw, g.hh, sg)) return false;
    for (auto& i : g.hud) {
        if (!Make(i, vk::Format::eR8Unorm, g.hw, g.hh, sg)) return false;
    }
    const auto mid_usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
    for (auto& i : g.mid) {
        if (!Make(i, vk::Format::eR8G8B8A8Unorm, g.w, g.h, mid_usage)) return false;
    }

    // Command buffers, fences, semaphores, queries, the counters.
    const vk::CommandPoolCreateInfo pci{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = g.family,
    };
    if (g.device.createCommandPool(&pci, nullptr, &g.pool) != vk::Result::eSuccess) return false;
    const vk::CommandBufferAllocateInfo cai{
        .commandPool = g.pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = kSlots,
    };
    if (g.device.allocateCommandBuffers(&cai, g.cmd.data()) != vk::Result::eSuccess) return false;
    for (uint32_t s = 0; s < kSlots; ++s) {
        const vk::FenceCreateInfo fci{};
        if (g.device.createFence(&fci, nullptr, &g.fence[s]) != vk::Result::eSuccess) return false;
        const vk::SemaphoreCreateInfo sci{};
        if (g.device.createSemaphore(&sci, nullptr, &g.real_wait[s]) != vk::Result::eSuccess) return false;
        for (int j = 0; j < 2; ++j) {
            if (g.device.createSemaphore(&sci, nullptr, &g.acquired[s][j]) != vk::Result::eSuccess) return false;
            if (g.device.createSemaphore(&sci, nullptr, &g.done[s][j]) != vk::Result::eSuccess) return false;
        }
    }
    const vk::QueryPoolCreateInfo qci{.queryType = vk::QueryType::eTimestamp, .queryCount = kSlots * kStamps};
    if (g.device.createQueryPool(&qci, nullptr, &g.queries) != vk::Result::eSuccess) return false;
    const vk::BufferCreateInfo bci{
        .size = static_cast<vk::DeviceSize>(kSlots) * kStatsStride,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    if (g.device.createBuffer(&bci, nullptr, &g.stats_buffer) != vk::Result::eSuccess) return false;
    vk::MemoryRequirements req{};
    g.device.getBufferMemoryRequirements(g.stats_buffer, &req);
    const uint32_t type = FindMemory(req.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible |
                                                             vk::MemoryPropertyFlagBits::eHostCoherent);
    if (type == UINT32_MAX) return false;
    const vk::MemoryAllocateInfo mi{.allocationSize = req.size, .memoryTypeIndex = type};
    if (g.device.allocateMemory(&mi, nullptr, &g.stats_memory) != vk::Result::eSuccess) return false;
    if (g.device.bindBufferMemory(g.stats_buffer, g.stats_memory, 0) != vk::Result::eSuccess) return false;
    void* mapped = nullptr;
    if (g.device.mapMemory(g.stats_memory, 0, VK_WHOLE_SIZE, {}, &mapped) != vk::Result::eSuccess) return false;
    g.stats_map = static_cast<uint8_t*>(mapped);
    std::memset(g.stats_map, 0, static_cast<size_t>(kSlots) * kStatsStride);

    // One-time layout transition of everything to GENERAL.
    {
        const vk::CommandBuffer cmd = g.cmd[0];
        const vk::CommandBufferBeginInfo bi{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
        (void)cmd.begin(&bi);
        TransitionAll(cmd);
        (void)cmd.end();
        const vk::SubmitInfo si{.commandBufferCount = 1, .pCommandBuffers = &cmd};
        {
            std::lock_guard lock{Vulkan::Scheduler::submit_mutex};
            if (g.queue.submit(1, &si, g.fence[0]) != vk::Result::eSuccess) return false;
        }
        if (g.device.waitForFences(1, &g.fence[0], VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) return false;
        (void)g.device.resetFences(1, &g.fence[0]);
    }
    g.built = true;
    g.failed = false;
    g.have_prev = false;
    g.hud_valid = false;
    g.cur = 0;
    g.slot = 0;
    return true;
}

bool BuildPipelines() {
    if (g.layout) return true;
    std::array<vk::DescriptorSetLayoutBinding, 13> b{};
    for (uint32_t i = 0; i < 13; ++i) {
        b[i].binding = i;
        b[i].descriptorType = i < 8 ? vk::DescriptorType::eCombinedImageSampler
                              : i < 12 ? vk::DescriptorType::eStorageImage
                                       : vk::DescriptorType::eStorageBuffer;
        b[i].descriptorCount = 1;
        b[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
    }
    const vk::DescriptorSetLayoutCreateInfo sli{
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = 13,
        .pBindings = b.data(),
    };
    if (g.device.createDescriptorSetLayout(&sli, nullptr, &g.set_layout) != vk::Result::eSuccess) return false;
    const vk::PushConstantRange pcr{vk::ShaderStageFlagBits::eCompute, 0, 128};
    const vk::PipelineLayoutCreateInfo pli{
        .setLayoutCount = 1,
        .pSetLayouts = &g.set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    if (g.device.createPipelineLayout(&pli, nullptr, &g.layout) != vk::Result::eSuccess) return false;
    if (!MakePipeline(g.pipes.capture, BB_IMGFG_CAPTURE_COMP) || !MakePipeline(g.pipes.pyr, BB_IMGFG_PYR_COMP) ||
        !MakePipeline(g.pipes.me, BB_IMGFG_ME_COMP) || !MakePipeline(g.pipes.flow, BB_IMGFG_FLOW_COMP) ||
        !MakePipeline(g.pipes.flowup, BB_IMGFG_FLOWUP_COMP) || !MakePipeline(g.pipes.hud, BB_IMGFG_HUD_COMP) ||
        !MakePipeline(g.pipes.interp, BB_IMGFG_INTERP_COMP)) {
        return false;
    }
    vk::SamplerCreateInfo sc{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .maxLod = 0.0f,
    };
    if (g.device.createSampler(&sc, nullptr, &g.linear) != vk::Result::eSuccess) return false;
    sc.magFilter = vk::Filter::eNearest;
    sc.minFilter = vk::Filter::eNearest;
    if (g.device.createSampler(&sc, nullptr, &g.nearest) != vk::Result::eSuccess) return false;
    return true;
}

// ---- recording helpers ----
struct Bind {
    uint32_t binding;
    vk::ImageView view;
    vk::Sampler sampler;  // set: a combined image sampler, else a storage image
    vk::ImageLayout layout = vk::ImageLayout::eGeneral;
};

void Dispatch(vk::CommandBuffer cmd, vk::Pipeline pipe, std::initializer_list<Bind> binds, const void* pc,
              uint32_t pc_size, uint32_t gx, uint32_t gy, uint32_t slot = UINT32_MAX) {
    std::array<vk::DescriptorImageInfo, 13> infos{};
    std::array<vk::WriteDescriptorSet, 13> writes{};
    uint32_t n = 0;
    for (const Bind& b : binds) {
        infos[n] = vk::DescriptorImageInfo{.sampler = b.sampler, .imageView = b.view, .imageLayout = b.layout};
        writes[n] = vk::WriteDescriptorSet{
            .dstBinding = b.binding,
            .descriptorCount = 1,
            .descriptorType = b.sampler ? vk::DescriptorType::eCombinedImageSampler : vk::DescriptorType::eStorageImage,
            .pImageInfo = &infos[n],
        };
        ++n;
    }
    vk::DescriptorBufferInfo bi{};
    if (slot != UINT32_MAX) {
        bi = vk::DescriptorBufferInfo{g.stats_buffer, static_cast<vk::DeviceSize>(slot) * kStatsStride, 16};
        writes[n] = vk::WriteDescriptorSet{
            .dstBinding = 12,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &bi,
        };
        ++n;
    }
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g.layout, 0, n, writes.data());
    cmd.pushConstants(g.layout, vk::ShaderStageFlagBits::eCompute, 0, pc_size, pc);
    cmd.dispatch(gx, gy, 1);
}

void Sync(vk::CommandBuffer cmd, vk::PipelineStageFlags dst = vk::PipelineStageFlagBits::eComputeShader,
          vk::AccessFlags dst_access = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite) {
    const vk::MemoryBarrier mb{.srcAccessMask = vk::AccessFlagBits::eShaderWrite, .dstAccessMask = dst_access};
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, dst, {}, mb, {}, {});
}

uint32_t Groups(uint32_t n) {
    return (n + 7) / 8;
}

vk::ImageView SwapView(vk::Image image, vk::Format format) {
    const uint64_t key = reinterpret_cast<uint64_t>(static_cast<VkImage>(image));
    if (const auto it = g.swap_views.find(key); it != g.swap_views.end()) return it->second;
    const vk::ImageViewCreateInfo vi{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    vk::ImageView view{};
    if (g.device.createImageView(&vi, nullptr, &view) != vk::Result::eSuccess) return {};
    g.swap_views[key] = view;
    return view;
}

void WaitUntil(Clock::time_point t) {
    for (;;) {
        const auto now = Clock::now();
        if (now >= t) return;
        const auto left = t - now;
        if (left > std::chrono::microseconds(1800)) {
            std::this_thread::sleep_for(left - std::chrono::microseconds(1500));
        } else {
            std::this_thread::yield();
        }
    }
}

double Ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

// Reads a finished slot's counters and timestamps into the second's statistics.
void Collect(uint32_t slot) {
    if (!g.in_flight[slot]) return;
    (void)g.device.waitForFences(1, &g.fence[slot], VK_TRUE, UINT64_MAX);
    (void)g.device.resetFences(1, &g.fence[slot]);
    g.in_flight[slot] = false;
    if (!g.slot_generated[slot]) return;
    std::array<uint64_t, kStamps> t{};
    const auto r = g.device.getQueryPoolResults(g.queries, slot * kStamps, kStamps, sizeof(t), t.data(),
                                                sizeof(uint64_t),
                                                vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (r == vk::Result::eSuccess && t[2] >= t[1] && t[1] >= t[0]) {
        g.stats.flow_ms += (t[1] - t[0]) * g.timestamp_ns * 1e-6;
        g.stats.interp_ms += (t[2] - t[1]) * g.timestamp_ns * 1e-6;
        ++g.stats.gpu_samples;
    }
    const uint32_t* c = reinterpret_cast<const uint32_t*>(g.stats_map + static_cast<size_t>(slot) * kStatsStride);
    g.stats.incons += c[0];
    g.stats.hud += c[1];
    g.stats.total += c[2];
    // The HUD-less image is only trusted while the backbuffer and it agree: a mask over most of the
    // picture means they are not the same encoding (or the frame is not what it claims to be).
    if (g.last_used_hudless && c[2] > 0) {
        if (static_cast<double>(c[1]) > 0.45 * static_cast<double>(c[2])) {
            if (++g.hudless_bad >= 5 && !g.hudless_off) {
                g.hudless_off = true;
                BBLogf("[imgfg] the HUD-less image differs from the backbuffer over %.0f%% of the picture: "
                       "not used any more (pure image path)",
                       100.0 * c[1] / c[2]);
            }
        } else {
            g.hudless_bad = 0;
        }
    }
}

void Report(Clock::time_point now) {
    Stats& s = g.stats;
    const double win = std::chrono::duration<double>(now - s.start).count();
    if (win < 1.0) return;
    const double flow = s.gpu_samples ? s.flow_ms / s.gpu_samples : 0.0;
    const double interp = s.gpu_samples ? s.interp_ms / s.gpu_samples : 0.0;
    BB::FG::Printf("[imgfg] real %.1f fps, generated %.1f, flow ms %.2f (gpu), interp ms %.2f, inconsistent "
                   "pixels %.2f %%, hud-protected %.2f %%",
                   s.real / win, s.generated / win, flow, interp,
                   s.total ? 100.0 * s.incons / s.total : 0.0, s.total ? 100.0 * s.hud / s.total : 0.0);
    s = {};
    s.start = now;
}

} // namespace

// ---------------------------------------------------------------------------------------------

bool Requested() {
    return Parse();
}

int Multiplier() {
    Parse();
    return g.mult;
}

bool Generating() {
    return g.generating.load(std::memory_order_relaxed);
}

bool WantInputs() {
    return Parse() && g.gamedata;
}

vk::ImageUsageFlags SwapchainUsage(vk::PhysicalDevice physical, vk::SurfaceKHR surface) {
    if (!Parse()) return {};
    const auto [result, caps] = physical.getSurfaceCapabilitiesKHR(surface);
    if (result != vk::Result::eSuccess || !(caps.supportedUsageFlags & vk::ImageUsageFlagBits::eSampled)) {
        BBLogf("[imgfg] the surface does not allow sampled swapchain images: image frame generation off");
        g.failed = true;
        return {};
    }
    return vk::ImageUsageFlagBits::eSampled;
}

uint32_t SwapchainExtraImages() {
    return Parse() ? 1u : 0u;
}

void OnDevice(vk::PhysicalDevice physical, vk::Device device, uint32_t family, vk::Queue queue) {
    if (!Parse()) return;
    g.physical = physical;
    g.device = device;
    g.family = family;
    g.queue = queue;
    g.memory = physical.getMemoryProperties();
    g.timestamp_ns = physical.getProperties().limits.timestampPeriod;
    g.ready_device = true;
    BBLogf("[imgfg] BB_FG=image-x%d: image-only frame generation (block optical flow on a luma pyramid, "
           "consistency-gated warps, HUD protection; game data %s)",
           g.mult, g.gamedata ? "used when given (BB_IMGFG_GAMEDATA=0: never)" : "off");
}

void OnSwapchain(vk::SwapchainKHR swapchain) {
    if (!Parse() || !g.ready_device) return;
    // (Swapchain::Destroy has waited for the device; the views of its images go with it.)
    for (auto& [k, v] : g.swap_views) g.device.destroyImageView(v);
    g.swap_views.clear();
    g.have_prev = false;
    g.hud_valid = false;
    g.grid_valid = false;
    g.sc_images.clear();
    g.sc_handle = vk::SwapchainKHR{};
    if (!swapchain) {
        g.generating = false;
    }
}

void NoteFrame(bool new_game_frame, const BB::FG::Inputs* inputs, vk::Extent2D backbuffer, vk::Rect2D region) {
    if (!Parse()) return;
    NoteState& n = g.note;
    n = {};
    n.new_frame = new_game_frame;
    n.backbuffer = backbuffer;
    if (inputs != nullptr && inputs->valid && g.gamedata) {
        n.reset = inputs->reset;
        n.has_motion = inputs->motion.view && inputs->motion.extent.width > 0;
        n.motion = inputs->motion.view;
        const bool whole = region.offset.x == 0 && region.offset.y == 0 && region.extent == backbuffer;
        n.has_hudless = inputs->has_hudless && inputs->hudless.view && inputs->hudless.extent == backbuffer &&
                        whole && !g.hudless_off;
        n.hudless = inputs->hudless.view;
    }
}

bool BeforeRealPresent(const PresentIn& in, vk::Semaphore* wait) {
    if (!Parse() || !g.ready_device || g.failed || !g.note.new_frame) return false;
    const auto entry = Clock::now();
    if (in.format != vk::Format::eR8G8B8A8Unorm && in.format != vk::Format::eB8G8R8A8Unorm) {
        BBLogf("[imgfg] the swapchain format %s is not 8-bit RGBA / BGRA: image frame generation off",
               vk::to_string(in.format).c_str());
        g.failed = true;
        return false;
    }
    if (!BuildPipelines()) {
        BBLogf("[imgfg] compute pipelines could not be created: image frame generation off");
        g.failed = true;
        return false;
    }
    if (!g.built || g.extent != in.extent || g.format != in.format) {
        if (!Build(in.extent, in.format)) {
            BBLogf("[imgfg] resources for %ux%u could not be created: image frame generation off",
                   in.extent.width, in.extent.height);
            Teardown();
            g.failed = true;
            return false;
        }
        BBLogf("[imgfg] resources %ux%u (flow at %ux%u, vectors %ux%u, 4 levels), swapchain format %s",
               g.w, g.h, g.lw[0], g.lh[0], g.gw[0], g.gh[0], vk::to_string(in.format).c_str());
    }
    if (g.stats.start == Clock::time_point{}) g.stats.start = entry;

    // The real frame interval (the pacer's grid at the guest flip, seen here).
    bool pause = false;
    if (g.last_entry != Clock::time_point{}) {
        const double dt = std::chrono::duration<double>(entry - g.last_entry).count();
        if (dt > 0.1) {
            pause = true;
        } else if (g.interval_s <= 0.0) {
            g.interval_s = dt;
        } else if (dt > 0.5 * g.interval_s && dt < 2.0 * g.interval_s) {
            g.interval_s += 0.1 * (dt - g.interval_s);
        }
    }
    g.last_entry = entry;
    if (pause) {
        g.have_prev = false;
        g.hud_valid = false;
        g.grid_valid = false;
    }
    const int mult = g.mult;
    const bool use_hudless = g.note.has_hudless;
    const bool generate = g.have_prev && !g.note.reset && g.interval_s > 0.002 && g.interval_s < 0.1 &&
                          g.prev_hudless == use_hudless;

    const uint32_t slot = g.slot;
    g.slot = (g.slot + 1) % kSlots;
    Collect(slot);

    if (g.sc_handle != in.swapchain || g.sc_images.empty()) {
        g.sc_handle = in.swapchain;
        uint32_t n = 0;
        (void)g.device.getSwapchainImagesKHR(in.swapchain, &n, nullptr);
        g.sc_images.resize(n);
        (void)g.device.getSwapchainImagesKHR(in.swapchain, &n, g.sc_images.data());
    }
    // Swapchain images for the intermediate frames.
    std::array<uint32_t, 2> out_index{};
    int n_out = 0;
    if (generate) {
        for (int j = 0; j < mult - 1; ++j) {
            uint32_t index = 0;
            const auto r = g.device.acquireNextImageKHR(in.swapchain, 100'000'000ull, g.acquired[slot][j], vk::Fence{},
                                                        &index);
            if (r != vk::Result::eSuccess && r != vk::Result::eSuboptimalKHR) {
                if (++g.acquire_fail <= 3) {
                    BBLogf("[imgfg] acquiring a swapchain image for a generated frame returned %s",
                           vk::to_string(r).c_str());
                }
                // (a semaphore signalled by a successful acquire must still be waited on)
                break;
            }
            out_index[n_out++] = index;
        }
    }
    const bool do_generate = generate && n_out >= 1;
    // If an acquire succeeded but a later one failed, those images are presented as they are: not
    // possible without content, so give up this frame's generation (they will be unused; the
    // acquired image is returned to the swapchain by presenting a copy of the real frame).
    const vk::CommandBuffer cmd = g.cmd[slot];
    cmd.reset();
    const vk::CommandBufferBeginInfo bi{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
    (void)cmd.begin(&bi);
    cmd.resetQueryPool(g.queries, slot * kStamps, kStamps);
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, g.queries, slot * kStamps + 0);
    cmd.fillBuffer(g.stats_buffer, static_cast<vk::DeviceSize>(slot) * kStatsStride, 16, 0);
    {
        const vk::MemoryBarrier mb{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                                   .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, mb,
                            {}, {});
    }
    // The real frame's backbuffer to a readable layout (it waits for the frame's own rendering:
    // everything submitted before on this queue).
    const vk::ImageSubresourceRange color{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    {
        const vk::ImageMemoryBarrier b{
            .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::ePresentSrcKHR,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = in.image,
            .subresourceRange = color,
        };
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader, {},
                            {}, {}, b);
    }
    Set& B = g.sets[g.cur];
    Set& A = g.sets[1 - g.cur];
    const vk::ImageView back = SwapView(in.image, in.format);
    if (!back) {
        (void)cmd.end();
        BBLogf("[imgfg] image view of the backbuffer could not be created: image frame generation off");
        g.failed = true;
        return false;
    }
    // 1. capture
    {
        struct {
            int32_t size[4];
            int32_t flags[4];
        } pc{{static_cast<int32_t>(g.w), static_cast<int32_t>(g.h), static_cast<int32_t>(g.hw),
              static_cast<int32_t>(g.hh)},
             {use_hudless ? 1 : 0, 0, 0, 0}};
        Dispatch(cmd, g.pipes.capture,
                 {{0, back, g.linear, vk::ImageLayout::eShaderReadOnlyOptimal},
                  {1, use_hudless ? g.note.hudless : back, g.linear, vk::ImageLayout::eShaderReadOnlyOptimal},
                  {8, B.main.view, {}},
                  {9, B.final.view, {}},
                  {10, B.lum[0].view, {}},
                  {11, g.hdiff.view, {}}},
                 &pc, sizeof(pc), Groups(g.hw), Groups(g.hh));
    }
    Sync(cmd);
    // 2. luma pyramid, then the normalised pictures
    for (int l = 1; l < kLevels; ++l) {
        struct {
            int32_t size[4];
            int32_t flags[4];
            float params[4];
        } pc{{static_cast<int32_t>(g.lw[l - 1]), static_cast<int32_t>(g.lh[l - 1]), static_cast<int32_t>(g.lw[l]),
              static_cast<int32_t>(g.lh[l])},
             {0, 0, 0, 0},
             {kEps, 0, 0, 0}};
        Dispatch(cmd, g.pipes.pyr, {{0, B.lum[l - 1].view, g.nearest}, {8, B.lum[l].view, {}}}, &pc, sizeof(pc),
                 Groups(g.lw[l]), Groups(g.lh[l]));
        Sync(cmd);
    }
    for (int l = 0; l < kLevels; ++l) {
        struct {
            int32_t size[4];
            int32_t flags[4];
            float params[4];
        } pc{{static_cast<int32_t>(g.lw[l]), static_cast<int32_t>(g.lh[l]), static_cast<int32_t>(g.lw[l]),
              static_cast<int32_t>(g.lh[l])},
             {1, 0, 0, 0},
             {kEps, 0, 0, 0}};
        Dispatch(cmd, g.pipes.pyr, {{0, B.lum[l].view, g.nearest}, {8, B.norm[l].view, {}}}, &pc, sizeof(pc),
                 Groups(g.lw[l]), Groups(g.lh[l]));
    }
    Sync(cmd);

    if (do_generate) {
        // 3. the flow, coarse to fine, both directions
        const bool mv = g.note.has_motion;
        for (int l = kLevels - 1; l >= 0; --l) {
            for (int dir = 0; dir < 2; ++dir) {
                const Img& src = dir == 0 ? A.norm[l] : B.norm[l];
                const Img& dst = dir == 0 ? B.norm[l] : A.norm[l];
                const bool top = l == kLevels - 1;
                struct {
                    int32_t size[4];
                    int32_t info[4];
                    float prior[4];
                } pc{{static_cast<int32_t>(g.lw[l]), static_cast<int32_t>(g.lh[l]), static_cast<int32_t>(g.gw[l]),
                      static_cast<int32_t>(g.gh[l])},
                     {top ? 1 : static_cast<int32_t>(g.gw[l + 1]), top ? 1 : static_cast<int32_t>(g.gh[l + 1]),
                      (top ? 4 : 1) | (mv ? 2 : 0), 0},
                     {dir == 0 ? -1.0f : 1.0f, 0, 0, 0}};
                // (no parent at the coarsest level: any image of the right type is bound, unread)
                const Img& parent = top ? g.flow[dir][l - 1] : g.flow[dir][l + 1];
                Dispatch(cmd, g.pipes.me,
                         {{0, src.view, g.nearest},
                          {1, dst.view, g.nearest},
                          {2, parent.view, g.linear},
                          {3, mv ? g.note.motion : B.main.view, g.linear,
                           mv ? vk::ImageLayout::eShaderReadOnlyOptimal : vk::ImageLayout::eGeneral},
                          {8, g.flow[dir][l].view, {}}},
                         &pc, sizeof(pc), Groups(g.gw[l]), Groups(g.gh[l]));
            }
            Sync(cmd);
        }
        // 4. median, consistency, guide
        {
            struct {
                int32_t size[4];
            } pc{{static_cast<int32_t>(g.gw[0]), static_cast<int32_t>(g.gh[0]), static_cast<int32_t>(g.hw),
                  static_cast<int32_t>(g.hh)}};
            Dispatch(cmd, g.pipes.flow,
                     {{0, g.flow[0][0].view, g.linear},
                      {1, g.flow[1][0].view, g.linear},
                      {2, A.lum[0].view, g.nearest},
                      {3, B.lum[0].view, g.nearest},
                      {8, g.ff.view, {}},
                      {9, g.bf.view, {}}},
                     &pc, sizeof(pc), Groups(g.gw[0]), Groups(g.gh[0]));
            Sync(cmd);
        }
        // 5. one vector per pixel, edge-aware
        {
            struct {
                int32_t size[4];
                int32_t half[4];
                float params[4];
            } pc{{static_cast<int32_t>(g.w), static_cast<int32_t>(g.h), static_cast<int32_t>(g.gw[0]),
                  static_cast<int32_t>(g.gh[0])},
                 {static_cast<int32_t>(g.hw), static_cast<int32_t>(g.hh), 0, 0},
                 {kSigma, 0, 0, 0}};
            Dispatch(cmd, g.pipes.flowup,
                     {{0, g.ff.view, g.nearest},
                      {1, g.bf.view, g.nearest},
                      {2, A.lum[0].view, g.nearest},
                      {3, B.lum[0].view, g.nearest},
                      {8, g.fp.view, {}},
                      {9, g.bp.view, {}}},
                     &pc, sizeof(pc), Groups(g.w), Groups(g.h));
        }
        // 6. the HUD mask
        {
            struct {
                int32_t size[4];
                float params[4];
            } pc{{static_cast<int32_t>(g.hw), static_cast<int32_t>(g.hh), use_hudless ? 1 : 0, g.hud_valid ? 1 : 0},
                 {kHudStill, kHudRange, kHudlessDiff, 0.75f}};
            Dispatch(cmd, g.pipes.hud,
                     {{0, A.lum[0].view, g.nearest},
                      {1, B.lum[0].view, g.nearest},
                      {2, g.hdiff.view, g.nearest},
                      {3, g.hud[1 - g.hud_cur].view, g.nearest},
                      {8, g.hud[g.hud_cur].view, {}}},
                     &pc, sizeof(pc), Groups(g.hw), Groups(g.hh));
            Sync(cmd);
        }
        cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, g.queries, slot * kStamps + 1);
        // 7. the intermediate frames
        for (int j = 1; j <= n_out; ++j) {
            struct {
                int32_t size[4];
                float p0[4];
                float p1[4];
            } pc{{static_cast<int32_t>(g.w), static_cast<int32_t>(g.h), static_cast<int32_t>(g.hw),
                  static_cast<int32_t>(g.hh)},
                 {static_cast<float>(j) / static_cast<float>(mult), kSc0, kSck, kConfGain},
                 {kConsK, 0.0f, j == 1 ? 1.0f : 0.0f, 0}};
            Dispatch(cmd, g.pipes.interp,
                     {{0, A.main.view, g.linear},
                      {1, B.main.view, g.linear},
                      {2, g.fp.view, g.linear},
                      {3, g.bp.view, g.linear},
                      {4, g.hud[g.hud_cur].view, g.linear},
                      {5, B.final.view, g.linear},
                      {6, A.lum[0].view, g.nearest},
                      {7, B.lum[0].view, g.nearest},
                      {8, g.mid[j - 1].view, {}}},
                     &pc, sizeof(pc), Groups(g.w), Groups(g.h), slot);
        }
        Sync(cmd, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
        cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, g.queries, slot * kStamps + 2);
        // 8. into the intermediate swapchain images (a blit: the swapchain may be BGRA)
        for (int j = 0; j < n_out; ++j) {
            const vk::Image dst = g.sc_images[out_index[j]];
            const vk::ImageMemoryBarrier to_dst{
                .srcAccessMask = {},
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = dst,
                .subresourceRange = color,
            };
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {},
                                {}, to_dst);
            vk::ImageBlit region{};
            region.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
            region.srcOffsets[0] = vk::Offset3D{0, 0, 0};
            region.srcOffsets[1] = vk::Offset3D{static_cast<int32_t>(g.w), static_cast<int32_t>(g.h), 1};
            region.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
            region.dstOffsets[0] = vk::Offset3D{0, 0, 0};
            region.dstOffsets[1] = vk::Offset3D{static_cast<int32_t>(g.w), static_cast<int32_t>(g.h), 1};
            cmd.blitImage(g.mid[j].image, vk::ImageLayout::eGeneral, dst, vk::ImageLayout::eTransferDstOptimal, 1,
                          &region, vk::Filter::eNearest);
            const vk::ImageMemoryBarrier to_present{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = {},
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::ePresentSrcKHR,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = dst,
                .subresourceRange = color,
            };
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe, {},
                                {}, {}, to_present);
        }
    } else {
        cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, g.queries, slot * kStamps + 1);
        cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, g.queries, slot * kStamps + 2);
    }
    {
        const vk::ImageMemoryBarrier back_to_present{
            .srcAccessMask = vk::AccessFlagBits::eShaderRead,
            .dstAccessMask = {},
            .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = in.image,
            .subresourceRange = color,
        };
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eBottomOfPipe, {},
                            {}, {}, back_to_present);
    }
    (void)cmd.end();

    // Submit: waits for the acquired images, signals the intermediate frames' presents and the
    // real frame's (the capture reads its image).
    std::array<vk::Semaphore, 2> wait_sem{};
    std::array<vk::PipelineStageFlags, 2> wait_stage{};
    std::array<vk::Semaphore, 3> signal_sem{};
    uint32_t n_signal = 0;
    for (int j = 0; j < n_out; ++j) {
        wait_sem[j] = g.acquired[slot][j];
        wait_stage[j] = vk::PipelineStageFlagBits::eAllCommands;
        signal_sem[n_signal++] = g.done[slot][j];
    }
    signal_sem[n_signal++] = g.real_wait[slot];
    const vk::SubmitInfo si{
        .waitSemaphoreCount = static_cast<uint32_t>(n_out),
        .pWaitSemaphores = wait_sem.data(),
        .pWaitDstStageMask = wait_stage.data(),
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
        .signalSemaphoreCount = n_signal,
        .pSignalSemaphores = signal_sem.data(),
    };
    vk::Result sr;
    {
        std::lock_guard lock{Vulkan::Scheduler::submit_mutex};
        sr = g.queue.submit(1, &si, g.fence[slot]);
    }
    if (sr != vk::Result::eSuccess) {
        BBLogf("[imgfg] queue submit returned %s: image frame generation off", vk::to_string(sr).c_str());
        g.failed = true;
        return false;
    }
    g.in_flight[slot] = true;
    g.slot_generated[slot] = do_generate;
    g.last_used_hudless = use_hudless;
    g.prev_hudless = use_hudless;
    *wait = g.real_wait[slot];

    // The history: this frame becomes the previous one.
    g.cur = 1 - g.cur;
    if (do_generate) g.hud_cur = 1 - g.hud_cur;
    g.hud_valid = do_generate;
    g.have_prev = true;
    ++g.stats.real;

    // Present the intermediate frames at the grid, the real one follows. The grid is the pacer's:
    // the real frames come at its steps; a frame that arrives past its slot starts the grid over.
    if (do_generate) {
        // The GPU work first (a generated frame must be ready when its present is made, or it is
        // shown late and the intervals alternate short / long): the wait is timed, and the grid
        // starts that much after the frame's arrival.
        const auto submitted = Clock::now();
        (void)g.device.waitForFences(1, &g.fence[slot], VK_TRUE, 50'000'000ull);
        const double lag = std::chrono::duration<double>(Clock::now() - submitted).count();
        g.lag_s = g.lag_s <= 0.0 ? lag : g.lag_s + 0.1 * (lag - g.lag_s);
        const auto base = entry + std::chrono::duration_cast<Clock::duration>(
                                      std::chrono::duration<double>(g.lag_s + 0.0005));
        const auto step = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(g.interval_s));
        auto first = base;
        if (g.grid_valid) {
            const auto expect = g.grid + step;
            // on the grid unless late (past 1 ms) or early (more than 3 ms): then the new arrival leads
            if (base <= expect + std::chrono::milliseconds(1) && base + std::chrono::milliseconds(3) >= expect) {
                first = expect;
            }
        }
        g.grid = first;
        g.grid_valid = true;
        const auto sub = step / mult;
        for (int j = 0; j < n_out; ++j) {
            WaitUntil(first + sub * j);
            uint64_t id = Display::IssueImage(true);
            VkPresentIdKHR present_id{};
            present_id.sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR;
            present_id.swapchainCount = 1;
            present_id.pPresentIds = &id;
            const vk::PresentInfoKHR pi{
                .pNext = id != 0 ? &present_id : nullptr,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &g.done[slot][j],
                .swapchainCount = 1,
                .pSwapchains = &in.swapchain,
                .pImageIndices = &out_index[j],
            };
            {
                std::lock_guard lock{Vulkan::Scheduler::submit_mutex};
                (void)g.queue.presentKHR(&pi);
            }
            ++g.stats.generated;
        }
        // the real frame at the last step of the interval
        WaitUntil(first + sub * (mult - 1));
        g.generating = true;
    } else {
        g.generating = false;
    }
    Report(Clock::now());
    return true;
}

} // namespace BB::ImgFg
