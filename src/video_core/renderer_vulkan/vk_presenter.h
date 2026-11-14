// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include "video_core/renderer_vulkan/bb_present_sr.h"
#include "video_core/renderer_vulkan/bb_hud_fg.h"

#include <condition_variable>

#include "core/libraries/videoout/buffer.h"
#include "imgui/imgui_texture.h"
#include "video_core/renderer_vulkan/host_passes/fsr_pass.h"
#include "video_core/renderer_vulkan/host_passes/pp_pass.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Frontend {
class WindowSDL;
}

namespace AmdGpu {
struct Liverpool;
}

namespace Vulkan {

struct Frame {
    u32 width;
    u32 height;
    VmaAllocation allocation;
    vk::Image image;
    vk::ImageView image_view;
    vk::Fence present_done;
    vk::Semaphore ready_semaphore;
    u64 ready_tick;
    bool is_hdr{false};
    u8 id{};
    BB::FG::Inputs fg{}; // bb-clean-021-fg: this frame's frame generation inputs (valid: 3D frame)
    // bb-hudless: the upscaled scene without the HUD, through the same post-process pass as the
    // frame (its size and format): FSR 3 FG's HUD-less colour (DLSS-G: kBufferTypeHUDLessColor).
    vk::Image hudless_image{};
    VmaAllocation hudless_allocation{};
    vk::ImageView hudless_view{};
    // bb-fsr3-hudless: an R8G8B8A8Unorm storage view of it (8-bit UNORM frames with storage only).
    vk::ImageView hudless_rgba_view{};
    u32 hudless_width = 0, hudless_height = 0;
    vk::Format hudless_format = vk::Format::eUndefined;
    // bb-clean-021-bbport: DLSS Frame Generation (Supermedo's Frame::generated): the frames shown
    // before this one, storage images of its size; and this frame's UNORM view for NGX.
    struct Generated {
        vk::Image image{};
        VmaAllocation allocation{};
        vk::ImageView view{}; // UNORM (storage)
        vk::ImageView rgba_view{}; // bb-clean: R8G8B8A8 storage view (HUD overlay pass)
        vk::Format format = vk::Format::eUndefined;
    };
    std::array<Generated, 3> generated{};
    u32 generated_count = 0;
    vk::ImageView unorm_view{};
    vk::Format unorm_format = vk::Format::eUndefined;
    // bb-clean: this frame's HUD layer for FSR 3 FG (bb_hud_fg.h), valid when has_ui_layer.
    BB::FG::Image ui_layer{};
    bool has_ui_layer = false;

    ImTextureID imgui_texture;
};

enum SchedulerType {
    Draw,
    Present,
    CpuFlip,
};

class Rasterizer;

class Presenter {
public:
    Presenter(Frontend::WindowSDL& window, AmdGpu::Liverpool* liverpool);
    ~Presenter();

    HostPasses::PostProcessingPass::Settings& GetPPSettingsRef() {
        return pp_settings;
    }

    HostPasses::FsrPass::Settings& GetFsrSettingsRef() {
        return fsr_settings;
    }

    Frontend::WindowSDL& GetWindow() const {
        return window;
    }

    Rasterizer& GetRasterizer() const {
        return *rasterizer.get();
    }

    bool IsHDRSupported() const {
        return swapchain.HasHDR();
    }

    void SetHDR(bool enable) {
        if (!IsHDRSupported()) {
            return;
        }
        swapchain.SetHDR(enable);
        pp_settings.hdr = enable ? 1 : 0;
    }

    VideoCore::Image& RegisterVideoOutSurface(
        const Libraries::VideoOut::BufferAttributeGroup& attribute, VAddr cpu_address) {
        vo_buffers_addr.emplace_back(cpu_address);
        auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
        const auto image_id = texture_cache.FindImage(desc);
        auto& image = texture_cache.GetImage(image_id);
        image.usage.vo_surface = 1u;
        return image;
    }

    bool IsVideoOutSurface(const AmdGpu::ColorBuffer& color_buffer) const;

    Frame* PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                        VAddr cpu_address);

    Frame* PrepareBlankFrame(bool present_thread);

    void Present(Frame* frame, bool is_reusing_frame = false, bool is_game_frame = true);
    Frame* PrepareLastFrame();

private:
    Frame* GetRenderFrame();

    void RecreateFrame(Frame* frame, u32 width, u32 height);

    void SetExpectedGameSize(s32 width, s32 height);

private:
    float expected_ratio{1920.0 / 1080.0f};
    u32 expected_frame_width{1920};
    u32 expected_frame_height{1080};

    Frontend::WindowSDL& window;
    Instance instance;
    HostPasses::FsrPass fsr_pass;
    HostPasses::FsrPass::Settings fsr_settings{};
    HostPasses::PostProcessingPass::Settings pp_settings{};
    HostPasses::PostProcessingPass pp_pass;
    AmdGpu::Liverpool* liverpool;
    Scheduler draw_scheduler;
    Scheduler present_scheduler;
    Scheduler flip_scheduler;
    Swapchain swapchain;
    Runtime runtime;
    std::unique_ptr<Rasterizer> rasterizer;
    std::unique_ptr<BB::PresentSr> bb_sr; // bb-clean-021: DLSS / FSR 4 (null when off)
    std::unique_ptr<BB::HudFg> bb_hud_fg; // bb-clean: the HUD kept out of frame generation
    double bb_wait_ms = 0.0;
    std::chrono::steady_clock::time_point bb_last_flip{};
    // bb-clean-021-bbport: Supermedo's frame generation present loop (Present, EnsureGenerated,
    // UpdateFrameGenLimit) for DLSS-G.
    u32 BbEnsureGenerated(Frame* frame, u32 count);
    void BbDestroyGenerated(Frame* frame, u32 from = 0);
    void BbUpdateFrameGenLimit(bool wanted, double present_ms);
    bool BbPresentGenerated(Frame* frame, u32 index, bool first);
    std::atomic<double> bb_game_frame_ms{16.6};
    std::chrono::steady_clock::time_point bb_last_game_present{};
    bool bb_imgui_ui_shown = false; // bb-dlssg-ui
    std::atomic<u32> bb_frame_gen_limit{3};
    std::chrono::steady_clock::time_point bb_limit_window{};
    u32 bb_limit_frames = 0, bb_limit_up_seconds = 0;
    double bb_present_busy = 0.0;
    std::chrono::steady_clock::time_point bb_frame_gen_hold_until{};
    int bb_frame_gen_hold_seconds = 0;
    // The one-line-per-second log (bb-clean-021-bbport): HUD cut, HUD-less copy, DLSS-G frames.
    std::atomic<u32> bb_sec_cut{0}, bb_sec_hudless{0}, bb_sec_hudless_failed{0};
    std::atomic<u32> bb_sec_game{0}, bb_sec_generated_made{0}, bb_sec_generated_shown{0},
        bb_sec_presents{0};
    u32 bb_minimized_skips = 0; // bb-alttab: frames not shown while minimized
    std::chrono::steady_clock::time_point bb_sec_start{};
    VideoCore::TextureCache& texture_cache;
    vk::UniqueCommandPool command_pool;
    std::vector<Frame> present_frames;
    std::queue<Frame*> free_queue;
    Frame* last_submit_frame;
    std::mutex free_mutex;
    std::condition_variable free_cv;
    std::condition_variable_any frame_cv;
    std::optional<ImGui::RefCountedTexture> splash_img;
    std::vector<VAddr> vo_buffers_addr;
};

} // namespace Vulkan
